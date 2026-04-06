"""
RedisConsumerDataset — PyTorch Dataset that reads from Redis.

The publisher (run via `ocr-publish`) streams HuggingFace data, augments
images, and stores pickled payloads in Redis.  This dataset pops from that
queue and returns (PIL Image, label_str) pairs compatible with AlignCollate.

Key design decisions (ported from the private repo):
- SRANDMEMBER for O(count) random sampling instead of O(N) SMEMBERS
- Local deque of prefetched indices (avoids per-image Redis round-trips)
- Proactive prefetch when local queue falls below prefetch_threshold
- Batched pipeline cleanup (delete + srem) at cleanup_threshold
- Auto-detection of zlib compression (publisher --compress flag)
"""

from __future__ import annotations

import io
import logging
import pickle
import time
import zlib
from collections import deque
from typing import Optional

import redis as _redis
from PIL import Image
from torch.utils.data import Dataset

from ocr_aster.data.publisher import REDIS_AVAILABLE, REDIS_KEY_PREFIX

logger = logging.getLogger(__name__)


class RedisConsumerDataset(Dataset):
    """
    Map-style Dataset backed by a Redis queue filled by HFRedisPublisher.

    __getitem__ ignores the index and always returns the next available
    image from Redis.  __len__ returns a large artificial epoch size so
    that a DataLoader runs for far longer than any training run.

    Args:
        redis_host:          Redis host (default: localhost)
        redis_port:          Redis port (default: 6379)
        redis_db:            Redis DB index (default: 0)
        redis_password:      optional password
        max_retries:         retries in __getitem__ before raising RuntimeError
        fetch_batch_size:    number of indices to grab per SRANDMEMBER call
        cleanup_threshold:   trigger cleanup after this many consumed indices
        prefetch_threshold:  trigger a new fetch when local queue drops below this
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        max_retries: int = 10,
        fetch_batch_size: int = 5000,
        cleanup_threshold: int = 4000,
        prefetch_threshold: int = 1000,
    ) -> None:
        self._max_retries        = max_retries
        self._fetch_batch_size   = fetch_batch_size
        self._cleanup_threshold  = cleanup_threshold
        self._prefetch_threshold = prefetch_threshold

        self._local_batch: deque[str] = deque()
        self._consumed: list[str]     = []

        self._total_batches    = 0
        self._processed_count  = 0
        self._consecutive_miss = 0

        self._redis = _redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,
        )
        try:
            self._redis.ping()
            logger.info("Redis consumer connected.")
        except Exception as exc:
            raise ConnectionError(f"Cannot connect to Redis: {exc}") from exc

        self._metrics = {
            "total_fetched":  0,
            "total_consumed": 0,
            "errors":         0,
            "total_fetch_s":  0.0,
            "total_clean_s":  0.0,
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Artificial epoch size — training is iteration-controlled."""
        return 10_000_000

    def __getitem__(self, _: int) -> tuple[Image.Image, str]:
        """
        Fetch the next available (PIL Image, label) from Redis.

        Blocks (with retries) until data is available.  Raises RuntimeError
        if the publisher appears to have stopped.
        """
        for attempt in range(self._max_retries):
            try:
                idx = self._get_next_index()
                if idx is None:
                    logger.info(
                        f"Waiting for publisher… attempt {attempt + 1}/{self._max_retries}"
                    )
                    time.sleep(1.0)
                    continue

                payload = self._fetch_payload(idx)
                if payload is None:
                    # Stale index — remove from set and skip
                    self._consecutive_miss += 1
                    if self._consecutive_miss in (3, 10, 50) or (
                        self._consecutive_miss > 3 and self._consecutive_miss % 100 == 0
                    ):
                        logger.warning(
                            f"Publisher lag — {self._consecutive_miss} consecutive misses"
                        )
                    try:
                        self._redis.srem(REDIS_AVAILABLE, idx)
                    except Exception:
                        pass
                    continue

                # Successful fetch
                if self._consecutive_miss >= 3:
                    logger.info(
                        f"Publisher recovered after {self._consecutive_miss} misses."
                    )
                self._consecutive_miss = 0

                data = payload  # dict with 'image_bytes' and 'label'
                img   = Image.open(io.BytesIO(data["image_bytes"]))
                label = data["label"]

                self._consumed.append(idx)
                self._processed_count += 1
                self._metrics["total_consumed"] += 1

                if len(self._consumed) >= self._cleanup_threshold:
                    self._cleanup()

                return img, label

            except Exception as exc:
                self._metrics["errors"] += 1
                logger.error(f"Consumer __getitem__ error: {exc}")
                time.sleep(0.5)

        raise RuntimeError(
            "No data from Redis after max retries. Is the publisher running?\n"
            "  Start it with:  ocr-publish --config <path>"
        )

    # ------------------------------------------------------------------
    # Startup helper
    # ------------------------------------------------------------------

    def wait_for_images(self, min_images: int = 10, timeout: float = 60.0) -> bool:
        """
        Block until Redis has at least `min_images` available.

        Returns True when ready, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                n = self._redis.scard(REDIS_AVAILABLE) or 0
                if n >= min_images:
                    logger.info(f"Redis ready — {n} images available.")
                    return True
                logger.info(f"Waiting for publisher… ({n}/{min_images})")
            except Exception as exc:
                logger.error(f"Redis error while waiting: {exc}")
            time.sleep(1.0)

        n = self._redis.scard(REDIS_AVAILABLE) or 0
        logger.warning(f"Timeout waiting for publisher — only {n} images available.")
        return n >= min_images

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_next_index(self) -> Optional[str]:
        """Return the next index from the local prefetch buffer, refilling if needed."""
        if len(self._local_batch) <= self._prefetch_threshold:
            self._fetch_batch()

        if not self._local_batch:
            return None

        return self._local_batch.popleft()

    def _fetch_batch(self) -> None:
        """Grab a batch of random indices from Redis into the local deque."""
        t0 = time.monotonic()
        try:
            n_available = self._redis.scard(REDIS_AVAILABLE) or 0
            if n_available == 0:
                return

            count = min(self._fetch_batch_size, n_available)
            raw = self._redis.srandmember(REDIS_AVAILABLE, count)
            if not raw:
                return

            indices = [
                b.decode("utf-8") if isinstance(b, bytes) else str(b)
                for b in raw
            ]
            self._local_batch.extend(indices)
            self._total_batches += 1
            self._metrics["total_fetched"] += len(indices)
            self._metrics["total_fetch_s"] += time.monotonic() - t0

            avg = self._metrics["total_fetch_s"] / self._total_batches
            logger.debug(
                f"Batch #{self._total_batches}: {len(indices)} of {n_available} "
                f"({time.monotonic() - t0:.2f}s, avg {avg:.2f}s)"
            )
        except Exception as exc:
            logger.error(f"_fetch_batch error: {exc}")

    def _fetch_payload(self, idx: str) -> Optional[dict]:
        """Get and deserialize one payload from Redis."""
        try:
            key  = REDIS_KEY_PREFIX.format(idx)
            data = self._redis.get(key)
            if data is None:
                return None

            # Auto-detect zlib compression (magic bytes \x78\x9c or \x78\x01)
            if len(data) >= 2 and data[:2] in (b"\x78\x9c", b"\x78\x01"):
                data = zlib.decompress(data)

            return pickle.loads(data)  # noqa: S301 — internal trusted data
        except Exception as exc:
            logger.error(f"_fetch_payload({idx}) error: {exc}")
            return None

    def _cleanup(self) -> None:
        """Delete consumed keys from Redis in a single pipeline call."""
        if not self._consumed:
            return
        t0 = time.monotonic()
        try:
            pipe = self._redis.pipeline()
            for idx in self._consumed:
                pipe.delete(REDIS_KEY_PREFIX.format(idx))
                pipe.srem(REDIS_AVAILABLE, idx)
            pipe.execute()
            elapsed = time.monotonic() - t0
            self._metrics["total_clean_s"] += elapsed
            logger.debug(f"Cleaned {len(self._consumed)} keys ({elapsed:.2f}s)")
        except Exception as exc:
            logger.error(f"_cleanup error: {exc}")
        finally:
            self._consumed.clear()

    def get_metrics(self) -> dict:
        """Return a snapshot of consumer performance metrics."""
        avg_fetch = (
            self._metrics["total_fetch_s"] / max(1, self._total_batches)
        )
        return {
            **self._metrics,
            "batches_fetched":  self._total_batches,
            "images_processed": self._processed_count,
            "local_queue_size": len(self._local_batch),
            "pending_cleanup":  len(self._consumed),
            "avg_fetch_s":      avg_fetch,
        }

    def __del__(self) -> None:
        try:
            if self._consumed:
                self._cleanup()
            self._redis.close()
        except Exception:
            pass

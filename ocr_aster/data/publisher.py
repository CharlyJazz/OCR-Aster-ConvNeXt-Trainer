"""
HFRedisPublisher — HuggingFace → augmentation → Redis.

Run as a standalone process in its own terminal:

    ocr-publish --config configs/training/my_config.yaml

The training loop (train.py) reads from Redis via RedisConsumerDataset
(consumer.py). The two processes are fully decoupled — the publisher can
run on a different machine, restart independently, or be swapped out.

Redis storage format:
    key:  processed_img:{dataset_idx}_{sample_idx}_{timestamp}
    val:  pickle.dumps({'image_bytes': PNG bytes, 'label': str,
                        'dataset_index': int, 'sample_index': int,
                        'timestamp': float})
    ttl:  14400 s (4 h)
    set:  publisher:available_indices  ← unique_ids for consumer SRANDMEMBER

Threading model inside the publisher process:
    Main thread    : HF streaming + future submission + Redis stores
    Worker threads : N_WORKERS parallel PIL augmentation tasks
"""

from __future__ import annotations

import io
import itertools
import logging
import pickle
import random
import time
import zlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterator

import redis as _redis
from datasets import load_dataset
from PIL import Image

from ocr_aster.config.schema import DatasetSourceConfig, TrainingConfig
from ocr_aster.data.augmentation import AugmentationPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis key schema (shared with consumer.py)
# ---------------------------------------------------------------------------

REDIS_KEY_PREFIX = "processed_img:{}"
REDIS_AVAILABLE  = "publisher:available_indices"


# ---------------------------------------------------------------------------
# HF streaming helpers (also used by dataset.py for validation)
# ---------------------------------------------------------------------------

def _load_hf_stream(src: DatasetSourceConfig) -> Iterator[dict]:
    """Open a HuggingFace dataset as a (conceptually) infinite iterator."""
    ds = load_dataset(
        src.repo_id,
        split=src.split,
        streaming=src.streaming,
        trust_remote_code=True,
    )
    return itertools.cycle(iter(ds))


def _apply_filter(stream: Iterator[dict], filter_fn) -> Iterator[dict]:
    def _filtered():
        for sample in stream:
            if filter_fn(sample):
                yield sample
    return _filtered()


def _pil_from_sample(sample: dict, image_column: str) -> Image.Image:
    """Extract a PIL Image from a HuggingFace sample dict."""
    raw = sample[image_column]
    if isinstance(raw, Image.Image):
        return raw
    if isinstance(raw, dict):
        if "bytes" in raw and raw["bytes"]:
            return Image.open(io.BytesIO(raw["bytes"]))
        if "path" in raw and raw["path"]:
            return Image.open(raw["path"])
    raise ValueError(
        f"Cannot convert sample[{image_column!r}] to PIL Image: {type(raw)}"
    )


# ---------------------------------------------------------------------------
# Per-sample augmentation worker (runs in ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _process_one(
    sample: dict,
    src: DatasetSourceConfig,
    src_idx: int,
    sample_idx: int,
    aug: AugmentationPipeline,
) -> dict | None:
    """
    Augment one HF sample. Called in a worker thread.

    Returns a storable dict or None if the sample should be skipped.
    """
    try:
        img = _pil_from_sample(sample, src.image_column)
        label: str = sample.get(src.label_column, "")
        if not label:
            return None

        try:
            img = aug(img)
        except Exception:
            pass  # augmentation failures are non-fatal

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return {
            "image_bytes": buf.getvalue(),
            "label": label,
            "dataset_index": src_idx,
            "sample_index": sample_idx,
            "timestamp": time.time(),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Publisher class
# ---------------------------------------------------------------------------

class HFRedisPublisher:
    """
    Streams from HuggingFace datasets, applies augmentation, and publishes
    processed images to Redis for the training loop to consume.

    Intended to run in a separate terminal:

        ocr-publish --config configs/training/my_config.yaml

    Args:
        config:        full TrainingConfig (datasets, augmentation, phases, redis)
        flush_on_start: FLUSHDB the Redis DB on startup (recommended for fresh runs)
    """

    _LOG_EVERY_N = 200   # log stats every N successfully stored images
    _CHECK_PHASE_EVERY = 64  # re-evaluate active phase every N samples submitted

    def __init__(
        self,
        config: TrainingConfig,
        flush_on_start: bool = True,
    ) -> None:
        self._config = config
        self._stop = False

        # Redis connection
        rc = config.redis
        self._redis = _redis.Redis(
            host=rc.host, port=rc.port, db=rc.db, decode_responses=False
        )
        self._redis.ping()   # fail fast if Redis is not reachable

        self._compress      = rc.compress
        self._max_memory_mb = rc.max_memory_mb
        self._ttl           = rc.ttl_seconds
        self._n_workers     = rc.n_workers

        if flush_on_start:
            logger.info("Flushing Redis DB on startup…")
            self._redis.flushdb()
            logger.info("Redis flushed.")

        # Augmentation pipeline (rebuilt on phase change)
        self._aug = AugmentationPipeline(level=config.augmentation.level)
        self._current_phase_name: str | None = None

        # HF streams (built at run() time to avoid holding connections in __init__)
        self._streams: list[tuple[int, DatasetSourceConfig, Iterator[dict]]] = []

        # Counters
        self._publish_count = 0
        self._stored_count  = 0
        self._error_count   = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main publisher loop. Runs until stop() is called or KeyboardInterrupt.
        """
        self._streams = self._build_streams()
        logger.info(
            f"Publisher starting — {len(self._streams)} dataset(s), "
            f"{self._n_workers} workers, max_memory={self._max_memory_mb} MB"
        )

        pending: deque[Future] = deque()

        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            while not self._stop:
                # ── Memory backpressure ────────────────────────────────────
                if not self._check_memory():
                    time.sleep(2.0)
                    continue

                # ── Fill in-flight pipeline ────────────────────────────────
                while len(pending) < self._n_workers * 2 and not self._stop:
                    src_tuple = _pick_stream(self._streams)
                    if src_tuple is None:
                        break
                    src_idx, src, stream = src_tuple
                    sample = next(stream, None)
                    if sample is None:
                        continue

                    aug = self._aug   # snapshot — safe to read without lock
                    future = executor.submit(
                        _process_one, sample, src, src_idx,
                        self._publish_count, aug,
                    )
                    pending.append(future)
                    self._publish_count += 1

                    if self._publish_count % self._CHECK_PHASE_EVERY == 0:
                        self._maybe_update_phase()

                # ── Collect completed futures ──────────────────────────────
                while pending and pending[0].done():
                    result = pending.popleft().result()
                    if result is not None:
                        self._store(result)

                # ── Yield CPU when pipeline is full ───────────────────────
                if len(pending) >= self._n_workers * 2:
                    time.sleep(0.002)

    def stop(self) -> None:
        """Signal the publisher loop to exit after the current batch."""
        self._stop = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_streams(
        self,
    ) -> list[tuple[int, DatasetSourceConfig, Iterator[dict]]]:
        result = []
        for i, src in enumerate(self._config.datasets):
            stream = _load_hf_stream(src)
            if src.source_filter:
                fn = eval(src.source_filter)   # noqa: S307 — user-controlled config
                stream = _apply_filter(stream, fn)
            result.append((i, src, stream))
        return result

    def _store(self, data: dict) -> None:
        """Serialize and push one processed image to Redis."""
        try:
            payload = pickle.dumps(data)
            if self._compress:
                payload = zlib.compress(payload)

            unique_id = (
                f"{data['dataset_index']}_"
                f"{data['sample_index']}_"
                f"{data['timestamp']}"
            )
            key = REDIS_KEY_PREFIX.format(unique_id)

            pipe = self._redis.pipeline()
            pipe.setex(key, self._ttl, payload)
            pipe.sadd(REDIS_AVAILABLE, unique_id)
            pipe.execute()

            self._stored_count += 1
            if self._stored_count % self._LOG_EVERY_N == 0:
                queue_size = self._redis.scard(REDIS_AVAILABLE) or 0
                logger.info(
                    f"[publisher] stored={self._stored_count:,}  "
                    f"redis_queue={queue_size:,}  "
                    f"errors={self._error_count}"
                )
        except Exception as exc:
            self._error_count += 1
            logger.error(f"Redis store error: {exc}")

    def _check_memory(self) -> bool:
        """Return True if Redis has room for more images."""
        try:
            info = self._redis.info("memory")
            used_mb = info.get("used_memory", 0) / (1024 * 1024)
            if used_mb >= self._max_memory_mb:
                logger.warning(
                    f"Redis memory limit reached: {used_mb:.0f} MB / "
                    f"{self._max_memory_mb} MB — pausing"
                )
                return False
        except Exception:
            pass
        return True

    def _maybe_update_phase(self) -> None:
        """Rebuild augmentation pipeline when the curriculum phase changes."""
        phase = self._config.active_phase(self._publish_count)
        phase_name = phase.name if phase else None
        if phase_name == self._current_phase_name:
            return

        self._current_phase_name = phase_name
        if phase is None or not phase.data_augmentation:
            self._aug = AugmentationPipeline(level="off")
        else:
            self._aug = AugmentationPipeline.from_lists(
                level=phase.data_augmentation_level,
                straug_augs=phase.straug_augs,
                albumentations_augs=phase.albumentations_augs,
            )
        logger.info(f"[publisher] Phase → {phase_name!r}  aug rebuilt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_stream(
    streams: list[tuple[int, DatasetSourceConfig, Iterator[dict]]],
) -> tuple[int, DatasetSourceConfig, Iterator[dict]] | None:
    if not streams:
        return None
    if len(streams) == 1:
        return streams[0]
    weights = [src.weight for _, src, _ in streams]
    return random.choices(streams, weights=weights, k=1)[0]

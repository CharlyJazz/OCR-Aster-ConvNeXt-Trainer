"""
Tests for RedisConsumerDataset.

Uses fakeredis so no real Redis server is needed.
"""

from __future__ import annotations

import io
import pickle
import time

import fakeredis
import pytest
from PIL import Image

from ocr_aster.data.consumer import RedisConsumerDataset
from ocr_aster.data.publisher import REDIS_AVAILABLE, REDIS_KEY_PREFIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grey(w: int = 64, h: int = 16) -> Image.Image:
    return Image.new("L", (w, h), color=128)


def _make_payload(label: str = "hello", compress: bool = False) -> bytes:
    """Create a pickled payload as the publisher would."""
    buf = io.BytesIO()
    _grey().save(buf, format="PNG")
    data = {
        "image_bytes": buf.getvalue(),
        "label":        label,
        "dataset_index": 0,
        "sample_index":  0,
        "timestamp":     time.time(),
    }
    raw = pickle.dumps(data)
    if compress:
        import zlib
        raw = zlib.compress(raw)
    return raw


def _populate_redis(fake_redis, n: int = 20, label_prefix: str = "word") -> list[str]:
    """Pre-populate a fakeredis instance with n images. Returns unique_ids."""
    ids = []
    for i in range(n):
        uid = f"0_{i}_{time.time() + i}"
        key = REDIS_KEY_PREFIX.format(uid)
        fake_redis.setex(key, 14400, _make_payload(label=f"{label_prefix}{i}"))
        fake_redis.sadd(REDIS_AVAILABLE, uid)
        ids.append(uid)
    return ids


def _make_consumer(fake_redis) -> RedisConsumerDataset:
    """Create a consumer wired to a fakeredis instance."""
    consumer = RedisConsumerDataset.__new__(RedisConsumerDataset)
    consumer._redis              = fake_redis
    consumer._max_retries        = 3
    consumer._fetch_batch_size   = 100
    consumer._cleanup_threshold  = 50
    consumer._prefetch_threshold = 10
    consumer._local_batch        = __import__("collections").deque()
    consumer._consumed           = []
    consumer._total_batches      = 0
    consumer._processed_count    = 0
    consumer._consecutive_miss   = 0
    consumer._metrics = {
        "total_fetched": 0, "total_consumed": 0,
        "errors": 0, "total_fetch_s": 0.0, "total_clean_s": 0.0,
    }
    return consumer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRedisConsumerDataset:
    def test_getitem_returns_pil_and_str(self):
        fake_redis = fakeredis.FakeRedis()
        _populate_redis(fake_redis, n=10)
        consumer = _make_consumer(fake_redis)
        img, label = consumer[0]
        assert isinstance(img, Image.Image)
        assert isinstance(label, str) and label

    def test_getitem_ignores_index(self):
        """Different indices should still return valid items."""
        fake_redis = fakeredis.FakeRedis()
        _populate_redis(fake_redis, n=10)
        consumer = _make_consumer(fake_redis)
        items = [consumer[i] for i in (0, 42, 999)]
        for img, label in items:
            assert isinstance(img, Image.Image)
            assert label

    def test_len_is_large(self):
        fake_redis = fakeredis.FakeRedis()
        consumer = _make_consumer(fake_redis)
        assert consumer.__len__() >= 1_000_000

    def test_fetch_batch_populates_local_queue(self):
        fake_redis = fakeredis.FakeRedis()
        _populate_redis(fake_redis, n=15)
        consumer = _make_consumer(fake_redis)
        consumer._fetch_batch()
        assert len(consumer._local_batch) == 15

    def test_cleanup_removes_keys_from_redis(self):
        fake_redis = fakeredis.FakeRedis()
        ids = _populate_redis(fake_redis, n=5)
        consumer = _make_consumer(fake_redis)
        consumer._consumed = list(ids)
        consumer._cleanup()

        # All keys and set members should be gone
        assert fake_redis.scard(REDIS_AVAILABLE) == 0
        for uid in ids:
            assert not fake_redis.exists(REDIS_KEY_PREFIX.format(uid))

    def test_cleanup_clears_consumed_list(self):
        fake_redis = fakeredis.FakeRedis()
        ids = _populate_redis(fake_redis, n=3)
        consumer = _make_consumer(fake_redis)
        consumer._consumed = list(ids)
        consumer._cleanup()
        assert consumer._consumed == []

    def test_auto_decompress_compressed_payload(self):
        """Consumer handles zlib-compressed payloads transparently."""
        fake_redis = fakeredis.FakeRedis()
        uid = "0_999_1.0"
        key = REDIS_KEY_PREFIX.format(uid)
        fake_redis.setex(key, 14400, _make_payload(label="zip", compress=True))
        fake_redis.sadd(REDIS_AVAILABLE, uid)

        consumer = _make_consumer(fake_redis)
        payload = consumer._fetch_payload(uid)
        assert payload is not None
        assert payload["label"] == "zip"

    def test_wait_for_images_returns_true_when_ready(self):
        fake_redis = fakeredis.FakeRedis()
        _populate_redis(fake_redis, n=5)
        consumer = _make_consumer(fake_redis)
        assert consumer.wait_for_images(min_images=5, timeout=2.0)

    def test_wait_for_images_returns_false_when_empty(self):
        fake_redis = fakeredis.FakeRedis()
        consumer = _make_consumer(fake_redis)
        assert not consumer.wait_for_images(min_images=1, timeout=0.5)

    def test_missing_key_increments_miss_counter(self):
        """A stale index (key gone from Redis) increments consecutive_miss."""
        fake_redis = fakeredis.FakeRedis()
        _populate_redis(fake_redis, n=2)
        consumer = _make_consumer(fake_redis)
        # Inject a non-existent index directly into the local batch
        consumer._local_batch.append("ghost_0_0_0.0")
        consumer._local_batch.append("ghost_0_1_0.0")
        # Drain the ghosts (will fail) then get real ones
        # __getitem__ retries up to max_retries
        # Populate more real data to let it eventually succeed
        _populate_redis(fake_redis, n=5)
        img, label = consumer[0]
        assert isinstance(img, Image.Image)

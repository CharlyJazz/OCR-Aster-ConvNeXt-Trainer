"""
Tests for HFRedisPublisher and its helpers.

HFRedisPublisher is meant to run in its own process, so we test it by
calling _process_one and _store directly rather than spinning up the full
run() loop — that would need real Redis and real HuggingFace.

Redis is faked with fakeredis; HuggingFace is mocked via unittest.mock.
"""

from __future__ import annotations

import io
import itertools
import pickle
from unittest.mock import patch

import fakeredis
import pytest
from PIL import Image

from ocr_aster.config.schema import (
    AugmentationConfig,
    DatasetSourceConfig,
    MLflowConfig,
    TrainingConfig,
)
from ocr_aster.data.augmentation import AugmentationPipeline
from ocr_aster.data.publisher import (
    HFRedisPublisher,
    REDIS_AVAILABLE,
    REDIS_KEY_PREFIX,
    _apply_filter,
    _pick_stream,
    _pil_from_sample,
    _process_one,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grey(w: int = 64, h: int = 16) -> Image.Image:
    return Image.new("L", (w, h), color=128)


def _make_config(num_datasets: int = 1) -> TrainingConfig:
    datasets = [
        DatasetSourceConfig(
            repo_id=f"user/ds_{i}",
            split="train",
            weight=1.0,
            image_column="image",
            label_column="label",
            streaming=True,
        )
        for i in range(num_datasets)
    ]
    return TrainingConfig(
        experiment_name="test",
        imgH=32, imgW=128,
        input_channel=1, output_channel=512, hidden_size=512,
        batch_max_length=25, sensitive=False,
        character="abcdefghijklmnopqrstuvwxyz",
        tf_start=1.0, tf_end=0.3, tf_decay_iters=300_000,
        batch_size=4, num_iter=1000, lr=1e-4,
        weight_decay=0.02, grad_clip=1.0,
        valInterval=100, save_every_n_iterations=500,
        checkpoints_dir="checkpoints/", saved_model="",
        datasets=datasets, val_dataset=datasets[0],
        phases=[],
        augmentation=AugmentationConfig(enabled=False, level="off"),
        mlflow=MLflowConfig(enabled=False),
    )


def _fake_samples(n: int = 50) -> list[dict]:
    return [{"image": _grey(), "label": f"word{i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# _pil_from_sample
# ---------------------------------------------------------------------------

class TestPilFromSample:
    def test_pil_image_passthrough(self):
        img = _grey()
        assert _pil_from_sample({"image": img}, "image") is img

    def test_bytes_dict(self):
        img = _grey()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _pil_from_sample({"image": {"bytes": buf.getvalue(), "path": None}}, "image")
        assert isinstance(result, Image.Image)

    def test_unknown_type_raises(self):
        with pytest.raises(Exception):
            _pil_from_sample({"image": 42}, "image")


# ---------------------------------------------------------------------------
# _process_one
# ---------------------------------------------------------------------------

class TestProcessOne:
    def _src(self):
        return DatasetSourceConfig(
            repo_id="x/y", split="train", weight=1.0,
            image_column="image", label_column="label", streaming=True,
        )

    def _aug(self):
        return AugmentationPipeline(level="off")

    def test_returns_dict_with_expected_keys(self):
        src  = self._src()
        sample = {"image": _grey(), "label": "hello"}
        result = _process_one(sample, src, 0, 0, self._aug())
        assert result is not None
        assert {"image_bytes", "label", "dataset_index", "sample_index", "timestamp"} \
               <= result.keys()

    def test_image_bytes_is_valid_png(self):
        src = self._src()
        result = _process_one({"image": _grey(), "label": "hi"}, src, 0, 0, self._aug())
        assert result is not None
        img = Image.open(io.BytesIO(result["image_bytes"]))
        assert isinstance(img, Image.Image)

    def test_empty_label_returns_none(self):
        src = self._src()
        result = _process_one({"image": _grey(), "label": ""}, src, 0, 0, self._aug())
        assert result is None

    def test_dataset_index_and_sample_index_stored(self):
        src = self._src()
        result = _process_one({"image": _grey(), "label": "ok"}, src, 3, 42, self._aug())
        assert result["dataset_index"] == 3
        assert result["sample_index"]  == 42


# ---------------------------------------------------------------------------
# _pick_stream
# ---------------------------------------------------------------------------

class TestPickStream:
    def _stream(self):
        return iter(itertools.cycle([{"image": _grey(), "label": "x"}]))

    def test_single_stream_always_returned(self):
        src = DatasetSourceConfig(
            repo_id="x/y", split="train", weight=1.0,
            image_column="image", label_column="label", streaming=True,
        )
        streams = [(0, src, self._stream())]
        assert _pick_stream(streams) is streams[0]

    def test_empty_returns_none(self):
        assert _pick_stream([]) is None

    def test_multi_stream_returns_one(self):
        src = DatasetSourceConfig(
            repo_id="x/y", split="train", weight=1.0,
            image_column="image", label_column="label", streaming=True,
        )
        streams = [(i, src, self._stream()) for i in range(3)]
        result = _pick_stream(streams)
        assert result in streams


# ---------------------------------------------------------------------------
# HFRedisPublisher._store
# ---------------------------------------------------------------------------

class TestPublisherStore:
    """Test _store() in isolation using a fakeredis server."""

    def _publisher(self, fake_redis) -> HFRedisPublisher:
        config = _make_config()
        pub = HFRedisPublisher.__new__(HFRedisPublisher)
        pub._config        = config
        pub._redis         = fake_redis
        pub._compress      = False
        pub._ttl           = 14400
        pub._stored_count  = 0
        pub._error_count   = 0
        pub._LOG_EVERY_N   = 200
        return pub

    def test_store_adds_key_and_index(self):
        fake_redis = fakeredis.FakeRedis()
        pub = self._publisher(fake_redis)
        data = {
            "image_bytes": b"png_bytes",
            "label": "hello",
            "dataset_index": 0,
            "sample_index": 1,
            "timestamp": 1.0,
        }
        pub._store(data)

        assert fake_redis.scard(REDIS_AVAILABLE) == 1
        unique_id = fake_redis.srandmember(REDIS_AVAILABLE, 1)[0].decode()
        key = REDIS_KEY_PREFIX.format(unique_id)
        assert fake_redis.exists(key)

    def test_stored_payload_round_trips(self):
        fake_redis = fakeredis.FakeRedis()
        pub = self._publisher(fake_redis)
        img = _grey()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = {
            "image_bytes": buf.getvalue(),
            "label": "world",
            "dataset_index": 0,
            "sample_index": 5,
            "timestamp": 2.0,
        }
        pub._store(data)

        uid = fake_redis.srandmember(REDIS_AVAILABLE, 1)[0].decode()
        raw = fake_redis.get(REDIS_KEY_PREFIX.format(uid))
        loaded = pickle.loads(raw)
        assert loaded["label"] == "world"
        assert loaded["image_bytes"] == buf.getvalue()

    def test_store_increments_counter(self):
        fake_redis = fakeredis.FakeRedis()
        pub = self._publisher(fake_redis)
        data = {
            "image_bytes": b"x", "label": "a",
            "dataset_index": 0, "sample_index": 0, "timestamp": 0.0,
        }
        pub._store(data)
        pub._store(data)
        assert pub._stored_count == 2

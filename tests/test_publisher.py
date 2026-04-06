"""
Tests for HFPublisher — HuggingFace is fully mocked, no network needed.
"""

from __future__ import annotations

import itertools
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ocr_aster.config.schema import (
    AugmentationConfig,
    DatasetSourceConfig,
    MLflowConfig,
    TrainingConfig,
)
from ocr_aster.data.publisher import HFPublisher, _pil_from_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grey(w: int = 64, h: int = 16) -> Image.Image:
    return Image.new("L", (w, h), color=128)


def _make_config(
    num_datasets: int = 1,
    weights: list[float] | None = None,
) -> TrainingConfig:
    if weights is None:
        weights = [1.0] * num_datasets

    datasets = [
        DatasetSourceConfig(
            repo_id=f"user/dataset_{i}",
            split="train",
            weight=weights[i],
            image_column="image",
            label_column="label",
            streaming=True,
        )
        for i in range(num_datasets)
    ]
    return TrainingConfig(
        experiment_name="test",
        imgH=32,
        imgW=128,
        input_channel=1,
        output_channel=512,
        hidden_size=512,
        batch_max_length=25,
        sensitive=False,
        character="abcdefghijklmnopqrstuvwxyz",
        tf_start=1.0,
        tf_end=0.3,
        tf_decay_iters=300_000,
        batch_size=4,
        num_iter=1000,
        lr=1e-4,
        weight_decay=0.02,
        grad_clip=1.0,
        valInterval=100,
        save_every_n_iterations=500,
        checkpoints_dir="checkpoints/",
        saved_model="",
        datasets=datasets,
        val_dataset=datasets[0],
        phases=[],
        augmentation=AugmentationConfig(enabled=False, level="off"),
        mlflow=MLflowConfig(enabled=False),
    )


def _fake_hf_stream(n: int = 100) -> list[dict]:
    """Return a finite list of fake HF samples."""
    return [
        {"image": _grey(), "label": f"word{i}"}
        for i in range(n)
    ]


def _patch_load_dataset(samples: list[dict]):
    """Context manager that mocks datasets.load_dataset."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(return_value=iter(samples))
    mock_ds.iter = MagicMock(return_value=iter(samples))

    def fake_load(**kwargs):
        return mock_ds

    return patch("ocr_aster.data.publisher.load_dataset", side_effect=fake_load)


# ---------------------------------------------------------------------------
# _pil_from_sample
# ---------------------------------------------------------------------------

class TestPilFromSample:
    def test_pil_image_passthrough(self):
        img = _grey()
        result = _pil_from_sample({"image": img}, "image")
        assert result is img

    def test_bytes_dict(self):
        import io
        img = _grey()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        sample = {"image": {"bytes": buf.getvalue(), "path": None}}
        result = _pil_from_sample(sample, "image")
        assert isinstance(result, Image.Image)

    def test_unknown_type_raises(self):
        with pytest.raises((ValueError, Exception)):
            _pil_from_sample({"image": 42}, "image")


# ---------------------------------------------------------------------------
# HFPublisher — batch shape and content
# ---------------------------------------------------------------------------

class TestPublisherBatches:
    def test_batch_size_respected(self):
        config = _make_config()
        samples = _fake_hf_stream(200)

        with patch("ocr_aster.data.publisher.load_dataset",
                   return_value=iter(itertools.cycle(samples))):
            pub = HFPublisher(config, batch_size=4, augment=False)
            batch = pub.get_batch(timeout=10.0)
            pub.stop()

        assert len(batch) == 4

    def test_batch_contains_pil_and_str(self):
        config = _make_config()
        samples = _fake_hf_stream(200)

        with patch("ocr_aster.data.publisher.load_dataset",
                   return_value=iter(itertools.cycle(samples))):
            pub = HFPublisher(config, batch_size=4, augment=False)
            batch = pub.get_batch(timeout=10.0)
            pub.stop()

        for img, label in batch:
            assert isinstance(img, Image.Image)
            assert isinstance(label, str)
            assert len(label) > 0

    def test_multiple_batches(self):
        config = _make_config()
        samples = _fake_hf_stream(500)

        with patch("ocr_aster.data.publisher.load_dataset",
                   return_value=iter(itertools.cycle(samples))):
            pub = HFPublisher(config, batch_size=8, augment=False, queue_maxsize=4)
            batches = [pub.get_batch(timeout=10.0) for _ in range(3)]
            pub.stop()

        assert len(batches) == 3
        for b in batches:
            assert len(b) == 8

    def test_context_manager_stops(self):
        config = _make_config()
        samples = _fake_hf_stream(200)

        with patch("ocr_aster.data.publisher.load_dataset",
                   return_value=iter(itertools.cycle(samples))):
            with HFPublisher(config, batch_size=4, augment=False) as pub:
                batch = pub.get_batch(timeout=10.0)

        assert len(batch) == 4
        assert pub._stop_event.is_set()


# ---------------------------------------------------------------------------
# Curriculum filter
# ---------------------------------------------------------------------------

class TestCurriculumFilter:
    def test_filter_removes_samples(self):
        config = _make_config()
        # Only samples whose label starts with "word0" should pass
        samples = _fake_hf_stream(200)

        with patch("ocr_aster.data.publisher.load_dataset",
                   return_value=iter(itertools.cycle(samples))):
            pub = HFPublisher(
                config,
                batch_size=4,
                augment=False,
                filter_fn=lambda s: s["label"].startswith("word0"),
            )
            batch = pub.get_batch(timeout=15.0)
            pub.stop()

        for _, label in batch:
            assert label.startswith("word0")


# ---------------------------------------------------------------------------
# Multi-dataset weighted choice (smoke test)
# ---------------------------------------------------------------------------

class TestMultiDataset:
    def test_two_datasets_no_crash(self):
        config = _make_config(num_datasets=2, weights=[0.7, 0.3])
        samples = _fake_hf_stream(200)

        # Both datasets return the same fake samples
        call_count = {"n": 0}

        def fake_load(*args, **kwargs):
            call_count["n"] += 1
            return iter(itertools.cycle(samples))

        with patch("ocr_aster.data.publisher.load_dataset", side_effect=fake_load):
            pub = HFPublisher(config, batch_size=4, augment=False)
            batch = pub.get_batch(timeout=15.0)
            pub.stop()

        assert len(batch) == 4
        # Both dataset streams should have been opened
        assert call_count["n"] == 2

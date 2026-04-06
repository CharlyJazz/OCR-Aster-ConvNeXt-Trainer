"""
HFPublisher — multi-dataset, threaded data publisher.

Loads one or more HuggingFace datasets (streaming mode), applies weighted
sampling and optional curriculum filters, augments images in background
threads, and pushes ready batches into a thread-safe queue.

The training loop calls publisher.get_batch() which blocks until a batch
is available — this hides data-loading latency behind GPU compute.
"""

from __future__ import annotations

import itertools
import queue
import random
import threading
from typing import Callable, Iterator

from datasets import load_dataset
from PIL import Image

from ocr_aster.config.schema import DatasetSourceConfig, TrainingConfig
from ocr_aster.data.augmentation import AugmentationPipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_hf_stream(src: DatasetSourceConfig) -> Iterator[dict]:
    """Open a HuggingFace dataset as an infinite streaming iterator."""
    ds = load_dataset(
        src.repo_id,
        split=src.split,
        streaming=src.streaming,
        trust_remote_code=True,
    )
    # Wrap in an infinite cycle so we never run out of data mid-epoch
    return itertools.cycle(iter(ds))


def _apply_filter(
    stream: Iterator[dict],
    filter_fn: Callable[[dict], bool] | None,
) -> Iterator[dict]:
    if filter_fn is None:
        return stream

    def filtered():
        for sample in stream:
            if filter_fn(sample):
                yield sample

    return filtered()


def _pil_from_sample(sample: dict, image_column: str) -> Image.Image:
    """Extract a PIL Image from a HF sample dict."""
    raw = sample[image_column]
    if isinstance(raw, Image.Image):
        return raw
    # HF datasets sometimes returns a dict {"bytes": ..., "path": ...}
    if isinstance(raw, dict):
        if "bytes" in raw and raw["bytes"]:
            import io
            return Image.open(io.BytesIO(raw["bytes"]))
        if "path" in raw and raw["path"]:
            return Image.open(raw["path"])
    raise ValueError(f"Cannot convert sample[{image_column!r}] to PIL Image: {type(raw)}")


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

class HFPublisher:
    """
    Streams batches of (PIL Image, label_str) pairs from one or more
    HuggingFace datasets.

    Args:
        config:          full TrainingConfig (datasets, augmentation, phases)
        batch_size:      number of samples per batch
        augment:         whether to apply augmentation (False for validation)
        queue_maxsize:   max number of pre-fetched batches in the internal queue
        filter_fn:       optional callable(sample_dict) → bool for curriculum
    """

    def __init__(
        self,
        config: TrainingConfig,
        batch_size: int,
        augment: bool = True,
        queue_maxsize: int = 8,
        filter_fn: Callable[[dict], bool] | None = None,
    ) -> None:
        self._config = config
        self._batch_size = batch_size
        self._augment = augment
        self._filter_fn = filter_fn
        self._queue: queue.Queue[list[tuple[Image.Image, str]]] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._stop_event = threading.Event()

        aug_level = config.augmentation.level if augment else "off"
        self._augmentation = AugmentationPipeline(level=aug_level)
        self._augmentation_lock = threading.Lock()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch(self, timeout: float = 60.0) -> list[tuple[Image.Image, str]]:
        """
        Block until a batch is ready and return it.

        Returns:
            list of (PIL Image, label_str) pairs, length == batch_size
        """
        return self._queue.get(timeout=timeout)

    def update_phase(self, iteration: int) -> None:
        """
        Call from the training loop when the iteration counter advances.

        Checks if the active curriculum phase has changed and, if so,
        rebuilds the augmentation pipeline from that phase's transform lists.
        """
        if not self._augment:
            return
        phase = self._config.active_phase(iteration)
        if phase is None:
            return
        if not phase.data_augmentation:
            new_aug = AugmentationPipeline(level="off")
        else:
            new_aug = AugmentationPipeline.from_lists(
                level=phase.data_augmentation_level,
                straug_augs=phase.straug_augs,
                albumentations_augs=phase.albumentations_augs,
            )
        with self._augmentation_lock:
            self._augmentation = new_aug

    def stop(self) -> None:
        """Signal the worker thread to stop."""
        self._stop_event.set()

    def __enter__(self) -> "HFPublisher":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        streams = self._build_streams()

        batch: list[tuple[Image.Image, str]] = []
        while not self._stop_event.is_set():
            src, stream = self._weighted_choice(streams)
            sample = next(stream)

            try:
                img = _pil_from_sample(sample, src.image_column)
            except Exception:
                continue

            label: str = sample.get(src.label_column, "")
            if not label:
                continue

            if self._augment:
                try:
                    with self._augmentation_lock:
                        aug = self._augmentation
                    img = aug(img)
                except Exception:
                    pass  # augmentation errors are non-fatal

            batch.append((img, label))

            if len(batch) == self._batch_size:
                self._queue.put(batch, timeout=5.0)
                batch = []

    def _build_streams(
        self,
    ) -> list[tuple[DatasetSourceConfig, Iterator[dict]]]:
        streams = []
        for src in self._config.datasets:
            stream = _load_hf_stream(src)
            stream = _apply_filter(stream, self._filter_fn)
            streams.append((src, stream))
        return streams

    def _weighted_choice(
        self, streams: list[tuple[DatasetSourceConfig, Iterator[dict]]]
    ) -> tuple[DatasetSourceConfig, Iterator[dict]]:
        if len(streams) == 1:
            return streams[0]
        weights = [s.weight for s, _ in streams]
        return random.choices(streams, weights=weights, k=1)[0]

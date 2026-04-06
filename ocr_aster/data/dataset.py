"""
HFOCRDataset — PyTorch IterableDataset backed by HFPublisher.

Yields (image_tensor, label_str) pairs.  The DataLoader wraps this with
AlignCollate as the collate_fn to produce (B, 1, H, W) tensors.

Validation datasets don't go through the publisher — they're loaded once
as a regular (non-streaming) HF dataset and iterated in order.
"""

from __future__ import annotations

from typing import Iterator

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from ocr_aster.config.schema import DatasetSourceConfig, TrainingConfig
from ocr_aster.data.collate import AlignCollate
from ocr_aster.data.publisher import HFPublisher


# ---------------------------------------------------------------------------
# Training dataset (IterableDataset backed by HFPublisher)
# ---------------------------------------------------------------------------

class HFOCRDataset(IterableDataset):
    """
    PyTorch IterableDataset that pulls (PIL Image, label) pairs from an
    HFPublisher one sample at a time.

    The publisher runs in a background thread — calling __iter__ just
    drains the queue.

    Args:
        publisher:   running HFPublisher instance
        max_samples: stop after this many samples (None = infinite)
    """

    def __init__(
        self,
        publisher: HFPublisher,
        max_samples: int | None = None,
    ) -> None:
        self._publisher = publisher
        self._max_samples = max_samples
        # Internal buffer — we receive full batches from publisher,
        # but yield individual samples
        self._buffer: list[tuple[Image.Image, str]] = []

    def __iter__(self) -> Iterator[tuple[Image.Image, str]]:
        count = 0
        while True:
            if self._max_samples is not None and count >= self._max_samples:
                return

            if not self._buffer:
                self._buffer = self._publisher.get_batch()

            yield self._buffer.pop(0)
            count += 1


# ---------------------------------------------------------------------------
# Validation dataset (standard map-style, loaded once, no augmentation)
# ---------------------------------------------------------------------------

class HFValDataset(torch.utils.data.Dataset):
    """
    Non-streaming validation dataset.  Loads the full split into memory
    (suitable for typical val sets of a few thousand samples).

    Args:
        src:          DatasetSourceConfig for the validation split
        imgH, imgW:   resize target — must match training config
    """

    def __init__(
        self,
        src: DatasetSourceConfig,
        imgH: int = 120,
        imgW: int = 280,
    ) -> None:
        from datasets import load_dataset

        self._src = src
        self._imgH = imgH
        self._imgW = imgW

        ds = load_dataset(
            src.repo_id,
            split=src.split,
            streaming=False,
            trust_remote_code=True,
        )
        self._samples = list(ds)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        sample = self._samples[idx]
        from ocr_aster.data.publisher import _pil_from_sample
        img = _pil_from_sample(sample, self._src.image_column)
        label: str = sample.get(self._src.label_column, "")
        return img, label


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_val_dataloader(
    config: TrainingConfig,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for the validation split declared in config.

    Returns batches of (image_tensor B×1×H×W, label_strings list[str]).
    """
    collate = AlignCollate(
        imgH=config.imgH,
        imgW=config.imgW,
        keep_ratio=False,
        adjust_contrast=False,
    )
    ds = HFValDataset(
        src=config.val_dataset,
        imgH=config.imgH,
        imgW=config.imgW,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

"""
Validation dataset and DataLoader builder.

Training data comes from RedisConsumerDataset (consumer.py), which reads
images pre-processed by the publisher (run_publisher.py).

Validation data is loaded once from HuggingFace (non-streaming) and kept
in memory for the full training run.
"""

from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import DataLoader

from ocr_aster.config.schema import DatasetSourceConfig, TrainingConfig
from ocr_aster.data.collate import AlignCollate
from ocr_aster.data.publisher import _pil_from_sample


# ---------------------------------------------------------------------------
# Validation dataset (map-style, loaded once, no augmentation)
# ---------------------------------------------------------------------------

class HFValDataset(torch.utils.data.Dataset):
    """
    Non-streaming validation dataset. Loads the full split into memory
    (suitable for typical val sets of a few thousand samples).

    Args:
        src:       DatasetSourceConfig for the validation split
        imgH, imgW: kept for API compatibility — resize happens in AlignCollate
    """

    def __init__(
        self,
        src: DatasetSourceConfig,
        imgH: int = 120,
        imgW: int = 280,
    ) -> None:
        from datasets import load_dataset

        self._src  = src
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
        img    = _pil_from_sample(sample, self._src.image_column)
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
    collate = AlignCollate(imgH=config.imgH, imgW=config.imgW)
    ds = HFValDataset(src=config.val_dataset, imgH=config.imgH, imgW=config.imgW)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

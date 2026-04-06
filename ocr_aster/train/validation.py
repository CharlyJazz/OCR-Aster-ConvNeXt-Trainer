"""
Validation loop — runs the model over a dataset and returns a ValidationResult.

Designed to be called from train.py every `valInterval` iterations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ocr_aster.model.model import AsterConvNeXt
from ocr_aster.train.metrics import (
    AccuracyByLength,
    CharacterErrorRate,
    ConfidenceCalibration,
    ExactMatchAccuracy,
    NormEditDistance,
    TopKCharacterConfusions,
    ValidationResult,
)
from ocr_aster.train.utils import AttnLabelConverter


@torch.no_grad()
def run_validation(
    model: AsterConvNeXt,
    dataloader: DataLoader,
    converter: AttnLabelConverter,
    batch_max_length: int,
    iteration: int,
    device: torch.device,
    max_batches: int | None = None,
) -> ValidationResult:
    """
    Run full validation pass and collect all metrics.

    Args:
        model:            AsterConvNeXt in eval mode
        dataloader:       validation DataLoader yielding (images, labels)
                          where labels is a list[str]
        converter:        AttnLabelConverter matching the training config
        batch_max_length: maximum output length
        iteration:        current training iteration (for logging)
        device:           compute device
        max_batches:      cap number of batches (useful for quick checks)

    Returns:
        ValidationResult with all metrics populated
    """
    model.eval()

    accuracy = ExactMatchAccuracy()
    cer = CharacterErrorRate()
    ned = NormEditDistance()
    by_length = AccuracyByLength()
    confusions = TopKCharacterConfusions(k=10)
    calibration = ConfidenceCalibration()

    total_loss = 0.0
    total_samples = 0
    total_batches = 0

    for batch in dataloader:
        if max_batches is not None and total_batches >= max_batches:
            break

        images, label_strings = batch
        images = images.to(device)

        B = images.size(0)
        text_for_pred, text_for_loss, _ = converter.encode(label_strings, batch_max_length)
        text_for_pred = text_for_pred.to(device)
        text_for_loss = text_for_loss.to(device)

        # Forward — inference mode (no teacher forcing)
        logits = model(
            images,
            targets=None,
            max_length=batch_max_length,
            teacher_forcing_ratio=0.0,
        )
        # logits: (B, max_len, num_classes)

        # Loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            text_for_loss[:, :batch_max_length].reshape(-1),
            ignore_index=0,
        )
        total_loss += loss.item() * B
        total_samples += B

        # Decode predictions
        pred_indices = logits.argmax(dim=-1)           # (B, max_len)
        predictions = converter.decode(pred_indices)

        # Per-sample softmax confidence (mean over predicted chars)
        probs = torch.softmax(logits, dim=-1)           # (B, max_len, C)
        top_probs = probs.max(dim=-1).values            # (B, max_len)
        confidences = top_probs.mean(dim=-1).tolist()  # (B,)

        # Update all metrics
        accuracy.update(predictions, label_strings)
        cer.update(predictions, label_strings)
        ned.update(predictions, label_strings)
        by_length.update(predictions, label_strings)
        confusions.update(predictions, label_strings)
        calibration.update(predictions, label_strings, confidences)

        total_batches += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return ValidationResult(
        iteration=iteration,
        num_samples=total_samples,
        accuracy=accuracy.value,
        cer=cer.value,
        norm_edit_distance=ned.value,
        val_loss=avg_loss,
        accuracy_by_length=by_length.value,
        counts_by_length=by_length.counts,
        top_confusions=confusions.value,
        avg_conf_correct=calibration.avg_correct,
        avg_conf_incorrect=calibration.avg_incorrect,
    )

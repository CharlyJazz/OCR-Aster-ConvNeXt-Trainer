"""
Forward pass with mixed-precision AMP and attention cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from ocr_aster.model.model import AsterConvNeXt
from ocr_aster.train.utils import AttnLabelConverter


def forward_pass(
    model: AsterConvNeXt,
    images: Tensor,
    text_for_pred: Tensor,
    text_for_loss: Tensor,
    lengths: Tensor,
    converter: AttnLabelConverter,
    teacher_forcing_ratio: float,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """
    One forward pass with AMP + attention cross-entropy loss.

    Args:
        model:                 AsterConvNeXt
        images:                (B, 1, H, W)
        text_for_pred:         (B, max_len+1) — GO + chars, fed to decoder
        text_for_loss:         (B, max_len+1) — chars + EOS, used as CE target
        lengths:               (B,) — actual string lengths
        converter:             AttnLabelConverter (for EOS index)
        teacher_forcing_ratio: τ ∈ [0, 1]
        device:                compute device
        label_smoothing:       ∈ [0, 1). 0.0 = standard CE, 0.1 = soft targets.
                               Reduces overconfidence and improves calibration.

    Returns:
        loss:  scalar tensor
        preds: (B, max_len) — argmax predicted indices
    """
    batch_max_length = text_for_pred.size(1) - 1  # exclude GO token

    with autocast(device_type=device.type, enabled=device.type in ("cuda", "cpu")):
        logits = model(
            images,
            targets=text_for_pred,
            max_length=batch_max_length,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        # logits: (B, max_len, num_classes)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),               # (B*max_len, C)
            text_for_loss[:, :batch_max_length].reshape(-1),   # (B*max_len,)
            ignore_index=0,
            label_smoothing=label_smoothing,
        )

    preds = logits.detach().argmax(dim=-1)  # (B, max_len)
    return loss, preds

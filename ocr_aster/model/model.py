"""
AsterConvNeXt — full model assembly.

ConvNeXt → BiLSTMEncoder → ASTERDecoder

This is the only model in this repository.
No factory, no switches, no alternatives.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ocr_aster.model.convnext import ConvNeXtOCR
from ocr_aster.model.encoder import BiLSTMEncoder
from ocr_aster.model.decoder import ASTERDecoder


class AsterConvNeXt(nn.Module):
    """
    Full OCR model: ConvNeXt + BiLSTM + ASTER v2 attention decoder.

    Args:
        num_classes:           vocabulary size (characters + GO + EOS + padding)
        input_channel:         1 for grayscale
        output_channel:        ConvNeXt output channels = BiLSTM hidden size (512)
        hidden_size:           BiLSTM and decoder GRU hidden size (512)
        embed_dim:             character embedding dimension in decoder
        convnext_depths:       blocks per ConvNeXt stage
        convnext_drop_path:    stochastic depth rate
    """

    def __init__(
        self,
        num_classes: int,
        input_channel: int = 1,
        output_channel: int = 512,
        hidden_size: int = 512,
        embed_dim: int = 256,
        convnext_depths: tuple[int, int, int] = (3, 3, 9),
        convnext_drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = ConvNeXtOCR(
            input_channel=input_channel,
            output_channel=output_channel,
            depths=convnext_depths,
            drop_path_rate=convnext_drop_path,
        )
        self.encoder = BiLSTMEncoder(
            input_size=output_channel,
            hidden_size=hidden_size,
        )
        self.decoder = ASTERDecoder(
            num_classes=num_classes,
            hidden_size=hidden_size,
            embed_dim=embed_dim,
        )

    def forward(
        self,
        images: Tensor,
        targets: Tensor | None = None,
        max_length: int = 28,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:
        """
        Args:
            images:                (B, 1, H, W) — grayscale, typically (B, 1, 120, 280)
            targets:               (B, max_length+1) — for training; None for inference
            max_length:            maximum output sequence length
            teacher_forcing_ratio: τ ∈ [0,1], only used during training

        Returns:
            logits: (B, max_length, num_classes)
        """
        # 1. Visual features
        features = self.backbone(images)           # (T=35, B, output_channel)

        # 2. Contextual encoding + bridge
        encoded, bridge = self.encoder(features)   # (T, B, H), (B, H)

        # 3. Attention decoding
        logits = self.decoder(
            encoded,
            bridge,
            targets=targets,
            max_length=max_length,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )                                          # (B, max_length, num_classes)

        return logits

    @torch.no_grad()
    def generate(self, images: Tensor, max_length: int = 28) -> Tensor:
        """
        Greedy inference — no targets, no teacher forcing.

        Returns:
            predictions: (B, max_length) — argmax character indices
        """
        self.eval()
        logits = self.forward(images, targets=None, max_length=max_length)
        return logits.argmax(dim=-1)  # (B, max_length)

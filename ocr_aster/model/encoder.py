"""
BiLSTM Encoder with LayerNorm gate (ASTER v2 addition).

Receives the ConvNeXt feature sequence (T=35, B, 512) and builds
contextual representations by reading it both left-to-right and
right-to-left simultaneously.

v2 addition: LayerNorm is applied to the feature map BEFORE the LSTM.
This stabilizes gradient flow through the long sequence and was a
measurable improvement over the v1 baseline.

Also returns the final hidden state, which the ASTER decoder uses
as its initial GRU state (encoder-decoder bridge).
"""

import torch
import torch.nn as nn
from torch import Tensor


class BidirectionalLSTM(nn.Module):
    """
    Single projection layer: BiLSTM → linear.

    Two of these are stacked in the full encoder (see BiLSTMEncoder).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            x: (T, B, input_size)

        Returns:
            output: (T, B, output_size)
            (h_n, c_n): final hidden and cell states from the LSTM
                h_n: (2, B, hidden_size)  — 2 because bidirectional
        """
        output, (h_n, c_n) = self.lstm(x)
        output = self.linear(output)
        return output, (h_n, c_n)


class BiLSTMEncoder(nn.Module):
    """
    Two-layer BiLSTM encoder for OCR sequence modeling.

    Forward pass:
        1. LayerNorm on input features  (v2 addition — before LSTM)
        2. BiLSTM layer 1: (T, B, 512) → (T, B, 512)
        3. BiLSTM layer 2: (T, B, 512) → (T, B, 512)
        4. Return encoded sequence + final hidden state for decoder bridge

    The final hidden state from layer 2 is averaged across the two
    directions (fwd + bwd) and used to warm-start the ASTER decoder's GRU,
    rather than initializing from zeros.

    Args:
        input_size: feature size coming from ConvNeXt (= output_channel = 512)
        hidden_size: BiLSTM hidden size per direction (512 → 1024 total → projected to 512)
    """

    def __init__(self, input_size: int = 512, hidden_size: int = 512) -> None:
        super().__init__()
        # v2: LayerNorm before LSTM
        self.norm = nn.LayerNorm(input_size)

        self.lstm1 = BidirectionalLSTM(input_size, hidden_size, hidden_size)
        self.lstm2 = BidirectionalLSTM(hidden_size, hidden_size, hidden_size)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (T, B, input_size) — ConvNeXt output sequence

        Returns:
            encoded: (T, B, hidden_size) — full contextual sequence for attention
            bridge:  (B, hidden_size)    — final hidden state for decoder init
        """
        # LayerNorm expects (*, C) — permute, norm, permute back
        x = self.norm(x.permute(1, 0, 2)).permute(1, 0, 2)  # (T, B, C) unchanged shape

        encoded, (h_n1, _) = self.lstm1(x)
        encoded, (h_n2, _) = self.lstm2(encoded)

        # Bridge: average forward and backward final hidden states from layer 2
        # h_n2 shape: (2, B, hidden_size) — dim 0 is [forward, backward]
        bridge = (h_n2[0] + h_n2[1]) / 2.0  # (B, hidden_size)

        return encoded, bridge

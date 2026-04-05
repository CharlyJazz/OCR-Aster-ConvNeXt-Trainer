"""
Additive (Bahdanau) attention for ASTER v2 decoder.

v2 optimization: the encoder projection (W_e · H_enc) is computed ONCE
per sequence and cached. In v1 it was recomputed at every decoding step,
which was O(T × max_len) multiplications. With caching it's O(T) once,
then O(1) per step — a measurable speedup on long sequences.

Attention formula:
    e_ti = v · tanh(W_h · h_{t-1} + W_e · H_enc_i)
    α_t  = softmax(e_t)
    c_t  = Σ_i α_ti · H_enc_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdditiveAttention(nn.Module):
    """
    Additive attention with cached encoder projection.

    Args:
        hidden_size: decoder GRU hidden size
        num_embeddings: number of character classes (vocabulary size)
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # query projection
        self.W_e = nn.Linear(hidden_size, hidden_size, bias=False)  # key projection (cached)
        self.v = nn.Linear(hidden_size, 1, bias=False)              # energy scalar

    def forward(
        self,
        hidden: Tensor,
        encoded: Tensor,
        encoded_proj: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            hidden:       (B, hidden_size)   — decoder GRU state h_{t-1}
            encoded:      (T, B, hidden_size) — full encoder output H_enc
            encoded_proj: (T, B, hidden_size) — W_e·H_enc, pre-computed once

        Returns:
            context: (B, hidden_size) — weighted sum over encoder positions
            alpha:   (B, T)           — attention weights (for visualization)
        """
        # Query projection: (B, hidden_size) → (1, B, hidden_size)
        query_proj = self.W_h(hidden).unsqueeze(0)  # (1, B, H)

        # Energy: e_ti = v · tanh(query_proj + encoded_proj)
        # encoded_proj: (T, B, H) + query_proj: (1, B, H) → broadcast (T, B, H)
        energy = self.v(torch.tanh(query_proj + encoded_proj))  # (T, B, 1)
        energy = energy.squeeze(2)                               # (T, B)

        # Attention weights
        alpha = F.softmax(energy, dim=0)   # (T, B) — softmax over positions

        # Context vector: weighted sum
        # encoded: (T, B, H), alpha: (T, B) → need (T, B, 1) to broadcast
        context = (encoded * alpha.unsqueeze(2)).sum(dim=0)  # (B, H)

        return context, alpha.permute(1, 0)  # context: (B,H), alpha: (B,T)

    def project_encoder(self, encoded: Tensor) -> Tensor:
        """
        Pre-compute W_e · H_enc for the full sequence.
        Call this ONCE before the decoding loop, then pass the result
        to every forward() call.

        Args:
            encoded: (T, B, hidden_size)

        Returns:
            (T, B, hidden_size)
        """
        return self.W_e(encoded)

"""
ASTER v2 Attention Decoder.

Generates one character at a time by attending over the encoded feature
sequence. Three v2 improvements over the baseline:

1. Encoder-Decoder Bridge
   The GRU hidden state is initialized from the encoder's final state
   (projected to hidden_size) instead of zeros. This gives the decoder
   a warm start with a global summary of the input image.

2. Scheduled Teacher Forcing
   During training, at each step the decoder either receives the ground-truth
   previous character (teacher forcing) or its own previous prediction
   (free-running), controlled by ratio τ ∈ [0, 1].
   τ decays from 1.0 → 0.3 over training — see train.py for the schedule.

3. Cached Encoder Projection
   W_e · H_enc is computed once before the decoding loop (in AdditiveAttention)
   and reused at every step.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ocr_aster.model.attention import AdditiveAttention


class ASTERDecoder(nn.Module):
    """
    GRU-based attention decoder for sequence recognition.

    Args:
        num_classes:  vocabulary size (number of output characters + EOS + padding)
        hidden_size:  GRU hidden size, must match encoder output (512)
        embed_dim:    character embedding dimension
    """

    # Special token indices (set by AttnLabelConverter at train time)
    GO_TOKEN = 0   # start-of-sequence
    EOS_TOKEN = 1  # end-of-sequence

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 512,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Character embedding
        self.embedding = nn.Embedding(num_classes, embed_dim)

        # Attention module (with cached encoder projection)
        self.attention = AdditiveAttention(hidden_size)

        # GRU decoder cell
        # Input: [context (H) ; embedded char (embed_dim)]
        self.gru = nn.GRUCell(hidden_size + embed_dim, hidden_size)

        # Output projection: hidden → character logits
        self.out_proj = nn.Linear(hidden_size, num_classes)

        # Bridge: project encoder final state → decoder initial state
        # encoder bridge is (B, hidden_size) and GRU hidden is (B, hidden_size)
        self.bridge = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def init_hidden(self, encoder_bridge: Tensor) -> Tensor:
        """
        Initialize GRU hidden state from encoder final state.
        This is the encoder-decoder bridge (v2 addition).

        Args:
            encoder_bridge: (B, hidden_size) from BiLSTMEncoder

        Returns:
            hidden: (B, hidden_size)
        """
        return self.bridge(encoder_bridge)

    def forward_step(
        self,
        char_input: Tensor,
        hidden: Tensor,
        encoded: Tensor,
        encoded_proj: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Single decoding step.

        Args:
            char_input:   (B,)              — character index fed as input
            hidden:       (B, hidden_size)  — current GRU state
            encoded:      (T, B, hidden_size)
            encoded_proj: (T, B, hidden_size) — pre-computed W_e · H_enc

        Returns:
            logits: (B, num_classes)
            hidden: (B, hidden_size) — updated GRU state
            alpha:  (B, T)           — attention weights
        """
        embedded = self.embedding(char_input)              # (B, embed_dim)
        context, alpha = self.attention(hidden, encoded, encoded_proj)  # (B, H), (B, T)
        gru_input = torch.cat([context, embedded], dim=1)  # (B, H + embed_dim)
        hidden = self.gru(gru_input, hidden)               # (B, H)
        logits = self.out_proj(hidden)                     # (B, num_classes)
        return logits, hidden, alpha

    def forward(
        self,
        encoded: Tensor,
        encoder_bridge: Tensor,
        targets: Tensor | None = None,
        max_length: int = 28,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:
        """
        Full decoding pass (training or inference).

        Training (targets provided):
            Teacher forcing ratio τ controls how often ground-truth
            previous character is fed vs. the model's own prediction.
            τ=1.0 → always teacher forcing (standard cross-entropy)
            τ=0.0 → always free-running (REINFORCE territory)

        Inference (targets=None):
            Greedy decoding: argmax at each step, stop at EOS.

        Args:
            encoded:               (T, B, hidden_size) — encoder output
            encoder_bridge:        (B, hidden_size)    — for GRU init
            targets:               (B, max_length+1)   — including GO token at pos 0
            max_length:            max output length
            teacher_forcing_ratio: τ ∈ [0, 1], only used when targets given

        Returns:
            logits: (B, max_length, num_classes)
        """
        B = encoded.size(1)

        # Pre-compute encoder projection once (v2 optimization)
        encoded_proj = self.attention.project_encoder(encoded)

        # Initialize decoder hidden state from encoder bridge (v2)
        hidden = self.init_hidden(encoder_bridge)

        # Start with GO token for every sample in the batch
        char_input = torch.full((B,), self.GO_TOKEN, dtype=torch.long, device=encoded.device)

        all_logits: list[Tensor] = []

        for t in range(max_length):
            logits, hidden, _ = self.forward_step(char_input, hidden, encoded, encoded_proj)
            all_logits.append(logits.unsqueeze(1))  # (B, 1, num_classes)

            if targets is not None:
                # Training: scheduled teacher forcing
                if random.random() < teacher_forcing_ratio:
                    char_input = targets[:, t + 1]  # ground-truth next char
                else:
                    char_input = logits.argmax(dim=-1)  # model's prediction
            else:
                # Inference: greedy
                char_input = logits.argmax(dim=-1)

        return torch.cat(all_logits, dim=1)  # (B, max_length, num_classes)

"""
Label encoding/decoding for the attention decoder.

The attention model uses special tokens:
  index 0 → [GO]  start-of-sequence (fed as first decoder input)
  index 1 → [EOS] end-of-sequence   (model predicts this to stop)
  index 2..N → actual characters from the config character set

Example with character="abc":
  vocab = ['[GO]', '[EOS]', 'a', 'b', 'c']
  num_class = 5
"""

from __future__ import annotations

import torch
from torch import Tensor


class AttnLabelConverter:
    """
    Converts between strings and index tensors for the attention decoder.

    Args:
        character: string of all allowed characters (no duplicates)
    """

    GO = "[GO]"
    EOS = "[EOS]"

    def __init__(self, character: str) -> None:
        # Deduplicate while preserving order
        chars = list(dict.fromkeys(character))
        self.vocab = [self.GO, self.EOS] + chars
        self.char2idx: dict[str, int] = {c: i for i, c in enumerate(self.vocab)}
        self.num_class = len(self.vocab)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], batch_max_length: int) -> tuple[Tensor, Tensor]:
        """
        Encode a batch of strings into padded index tensors.

        Returns:
            text_for_pred: (B, batch_max_length + 1)
                Position 0 is always GO. Positions 1..L are character indices.
                Remaining positions are padded with EOS.

            text_for_loss: (B, batch_max_length + 1)
                Position 0..L-1 are character indices.
                Position L is EOS. Remaining are EOS-padded.
                Used as target for cross-entropy loss.

            length: (B,) — actual string lengths (without GO/EOS)
        """
        B = len(texts)
        # +1 for the GO/EOS token
        text_for_pred = torch.zeros(B, batch_max_length + 1, dtype=torch.long)
        text_for_loss = torch.zeros(B, batch_max_length + 1, dtype=torch.long)

        for i, text in enumerate(texts):
            # Truncate silently if text exceeds max length
            text = text[:batch_max_length]

            # text_for_pred: [GO, c0, c1, ..., cn, EOS, EOS, ...]
            text_for_pred[i, 0] = self.char2idx[self.GO]
            for j, ch in enumerate(text):
                idx = self.char2idx.get(ch, self.char2idx[self.EOS])
                text_for_pred[i, j + 1] = idx

            # text_for_loss: [c0, c1, ..., cn, EOS, EOS, ...]
            for j, ch in enumerate(text):
                idx = self.char2idx.get(ch, self.char2idx[self.EOS])
                text_for_loss[i, j] = idx
            text_for_loss[i, len(text)] = self.char2idx[self.EOS]

        lengths = torch.tensor([len(t[:batch_max_length]) for t in texts], dtype=torch.long)
        return text_for_pred, text_for_loss, lengths

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, indices: Tensor) -> list[str]:
        """
        Decode a batch of index tensors back to strings.

        Args:
            indices: (B, L) — argmax outputs from the decoder

        Returns:
            List of decoded strings. Stops at first EOS token.
        """
        results = []
        eos_idx = self.char2idx[self.EOS]
        for row in indices:
            chars = []
            for idx in row.tolist():
                if idx == eos_idx:
                    break
                if idx < len(self.vocab):
                    ch = self.vocab[idx]
                    if ch not in (self.GO, self.EOS):
                        chars.append(ch)
            results.append("".join(chars))
        return results


class Averager:
    """Running average — used to track loss and metrics across iterations."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def add(self, value: float | Tensor, count: int = 1) -> None:
        if isinstance(value, Tensor):
            value = value.item()
        self._sum += value * count
        self._count += count

    @property
    def val(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    def __repr__(self) -> str:
        return f"Averager(val={self.val:.4f}, count={self._count})"

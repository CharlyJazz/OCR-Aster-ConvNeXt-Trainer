"""
Dataset-agnostic OCR validation metrics.

All metrics operate on plain Python strings — no domain knowledge,
no chord types, no music. Works for any text recognition task.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import editdistance


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """All metrics for one validation run."""

    iteration: int
    num_samples: int

    # Core
    accuracy: float = 0.0          # exact match rate
    cer: float = 0.0               # character error rate
    norm_edit_distance: float = 0.0 # 1 - edit_dist / max(len_pred, len_gt)
    val_loss: float = 0.0

    # Accuracy by length group
    accuracy_by_length: dict[str, float] = field(default_factory=dict)
    counts_by_length: dict[str, int] = field(default_factory=dict)

    # Top-K character confusions: list of (gt_char, pred_char, count)
    top_confusions: list[tuple[str, str, int]] = field(default_factory=list)

    # Confidence calibration
    avg_conf_correct: float = 0.0
    avg_conf_incorrect: float = 0.0

    @property
    def calibration_gap(self) -> float:
        return self.avg_conf_correct - self.avg_conf_incorrect

    def summary(self) -> str:
        lines = [
            f"Accuracy:            {self.accuracy * 100:.2f}%",
            f"Norm Edit Distance:  {self.norm_edit_distance:.4f}",
            f"CER:                 {self.cer:.4f}",
            f"Val Loss:            {self.val_loss:.4f}",
            f"Samples:             {self.num_samples}",
        ]
        if self.accuracy_by_length:
            lines.append("\nAccuracy by length:")
            for group, acc in self.accuracy_by_length.items():
                n = self.counts_by_length.get(group, 0)
                lines.append(f"  {group:>8}: {acc * 100:6.2f}%  ({n} samples)")
        if self.top_confusions:
            lines.append("\nTop character confusions (GT → Pred):")
            for gt, pred, cnt in self.top_confusions[:10]:
                lines.append(f"  '{gt}' → '{pred}': {cnt} times")
        if self.avg_conf_correct or self.avg_conf_incorrect:
            lines.append(f"\nCalibration gap: {self.calibration_gap:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual metric classes
# ---------------------------------------------------------------------------

class ExactMatchAccuracy:
    """Fraction of predictions that exactly match the ground truth."""

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, predictions: list[str], targets: list[str]) -> None:
        for pred, gt in zip(predictions, targets):
            self._total += 1
            if pred == gt:
                self._correct += 1

    @property
    def value(self) -> float:
        return self._correct / self._total if self._total > 0 else 0.0

    def reset(self) -> None:
        self._correct = self._total = 0


class CharacterErrorRate:
    """
    CER = sum(edit_distance(pred, gt)) / sum(len(gt))

    Lower is better. 0.0 = perfect, 1.0 = completely wrong on average.
    """

    def __init__(self) -> None:
        self._total_dist = 0
        self._total_len = 0

    def update(self, predictions: list[str], targets: list[str]) -> None:
        for pred, gt in zip(predictions, targets):
            self._total_dist += editdistance.eval(pred, gt)
            self._total_len += max(len(gt), 1)

    @property
    def value(self) -> float:
        return self._total_dist / self._total_len if self._total_len > 0 else 0.0

    def reset(self) -> None:
        self._total_dist = self._total_len = 0


class NormEditDistance:
    """
    NED = 1 - edit_distance / max(len(pred), len(gt))

    Higher is better. 1.0 = perfect match.
    """

    def __init__(self) -> None:
        self._total = 0.0
        self._count = 0

    def update(self, predictions: list[str], targets: list[str]) -> None:
        for pred, gt in zip(predictions, targets):
            denom = max(len(pred), len(gt), 1)
            self._total += 1.0 - editdistance.eval(pred, gt) / denom
            self._count += 1

    @property
    def value(self) -> float:
        return self._total / self._count if self._count > 0 else 0.0

    def reset(self) -> None:
        self._total = 0.0
        self._count = 0


# Length group boundaries: (label, max_length_inclusive)
_LENGTH_GROUPS = [
    ("1-5",   5),
    ("6-10",  10),
    ("11-20", 20),
    ("21+",   10_000),
]


class AccuracyByLength:
    """
    Exact-match accuracy broken down by ground-truth string length.

    Groups: 1-5, 6-10, 11-20, 21+
    """

    def __init__(self) -> None:
        self._correct: dict[str, int] = {g: 0 for g, _ in _LENGTH_GROUPS}
        self._total: dict[str, int] = {g: 0 for g, _ in _LENGTH_GROUPS}

    def _group(self, length: int) -> str:
        for label, max_len in _LENGTH_GROUPS:
            if length <= max_len:
                return label
        return "21+"

    def update(self, predictions: list[str], targets: list[str]) -> None:
        for pred, gt in zip(predictions, targets):
            grp = self._group(len(gt))
            self._total[grp] += 1
            if pred == gt:
                self._correct[grp] += 1

    @property
    def value(self) -> dict[str, float]:
        return {
            g: self._correct[g] / self._total[g] if self._total[g] > 0 else 0.0
            for g, _ in _LENGTH_GROUPS
        }

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._total)

    def reset(self) -> None:
        for g, _ in _LENGTH_GROUPS:
            self._correct[g] = self._total[g] = 0


class TopKCharacterConfusions:
    """
    Tracks the most frequent GT → Predicted character substitutions.

    Useful for diagnosing systematic errors: 'O'↔'0', 'l'↔'1', case errors.
    Only substitutions are counted (not insertions or deletions).
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self._counter: Counter = Counter()

    def update(self, predictions: list[str], targets: list[str]) -> None:
        for pred, gt in zip(predictions, targets):
            if pred == gt:
                continue
            # Align character by character at the shorter length
            for gt_ch, pred_ch in zip(gt, pred):
                if gt_ch != pred_ch:
                    self._counter[(gt_ch, pred_ch)] += 1

    @property
    def value(self) -> list[tuple[str, str, int]]:
        return [
            (gt_ch, pred_ch, cnt)
            for (gt_ch, pred_ch), cnt in self._counter.most_common(self.k)
        ]

    def reset(self) -> None:
        self._counter.clear()


class ConfidenceCalibration:
    """
    Compares mean softmax confidence for correct vs incorrect predictions.

    A well-calibrated model shows a large gap:
        correct confidence >> incorrect confidence
    """

    def __init__(self) -> None:
        self._conf_correct: list[float] = []
        self._conf_incorrect: list[float] = []

    def update(
        self,
        predictions: list[str],
        targets: list[str],
        confidences: list[float],
    ) -> None:
        for pred, gt, conf in zip(predictions, targets, confidences):
            if pred == gt:
                self._conf_correct.append(conf)
            else:
                self._conf_incorrect.append(conf)

    @property
    def avg_correct(self) -> float:
        return sum(self._conf_correct) / len(self._conf_correct) if self._conf_correct else 0.0

    @property
    def avg_incorrect(self) -> float:
        return sum(self._conf_incorrect) / len(self._conf_incorrect) if self._conf_incorrect else 0.0

    @property
    def calibration_gap(self) -> float:
        return self.avg_correct - self.avg_incorrect

    def reset(self) -> None:
        self._conf_correct.clear()
        self._conf_incorrect.clear()

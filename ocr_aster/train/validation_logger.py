"""
Writes structured validation reports to disk.
MLflow logging is handled separately in monitoring/tracker.py.
"""

from __future__ import annotations

from pathlib import Path

from ocr_aster.train.metrics import ValidationResult


_SEPARATOR = "=" * 80


def write_report(result: ValidationResult, log_path: Path) -> None:
    """
    Append a formatted validation report to a log file.

    Args:
        result:   populated ValidationResult
        log_path: path to the log file (created if it doesn't exist)
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    report = _format_report(result)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(report + "\n\n")


def _format_report(result: ValidationResult) -> str:
    lines = [
        _SEPARATOR,
        f"VALIDATION — Iteration {result.iteration:,}",
        _SEPARATOR,
        f"Accuracy (%):          {result.accuracy * 100:>7.2f}",
        f"Norm Edit Distance:    {result.norm_edit_distance:>7.4f}",
        f"CER:                   {result.cer:>7.4f}",
        f"Validation Loss:       {result.val_loss:>7.4f}",
        f"Samples Evaluated:     {result.num_samples:>7,}",
    ]

    if result.accuracy_by_length:
        lines.append("")
        lines.append("Accuracy by Length:")
        lines.append("-" * 40)
        for group, acc in result.accuracy_by_length.items():
            n = result.counts_by_length.get(group, 0)
            bar = _bar(acc, width=20)
            lines.append(f"  {group:>6} chars: {bar} {acc * 100:5.1f}%  ({n:,} samples)")

    if result.top_confusions:
        lines.append("")
        lines.append("Top Character Confusions (GT → Pred):")
        lines.append("-" * 40)
        for i, (gt, pred, cnt) in enumerate(result.top_confusions, 1):
            lines.append(f"  {i:>2}. '{gt}' → '{pred}': {cnt:,} times")

    if result.avg_conf_correct or result.avg_conf_incorrect:
        lines.append("")
        lines.append("Confidence Calibration:")
        lines.append("-" * 40)
        lines.append(f"  Correct:    {result.avg_conf_correct:.4f}")
        lines.append(f"  Incorrect:  {result.avg_conf_incorrect:.4f}")
        lines.append(f"  Gap:        {result.calibration_gap:.4f}")

    lines.append(_SEPARATOR)
    return "\n".join(lines)


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

"""
CLI entry point.

Usage:
    uv run ocr-train --config configs/training/aster_v2_iiit5k.yaml
    python -m ocr_aster.train.run --config configs/training/aster_v2_iiit5k.yaml
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ocr-train",
        description="Train ConvNeXt + BiLSTM + ASTER v2 on a HuggingFace dataset.",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to training YAML config (e.g. configs/training/aster_v2_iiit5k.yaml)",
    )
    args = parser.parse_args()

    # Lazy import so --help is instant even without heavy deps loaded
    from ocr_aster.train.train import train

    try:
        train(args.config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

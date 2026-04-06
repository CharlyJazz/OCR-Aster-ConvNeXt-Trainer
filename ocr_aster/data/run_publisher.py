"""
Publisher entry point — runs in its own terminal.

Usage:
    ocr-publish --config configs/training/my_config.yaml
    ocr-publish --config configs/training/my_config.yaml --no-flush
    ocr-publish --config configs/training/my_config.yaml --redis-host 192.168.1.10

Start this BEFORE the training loop:
    Terminal 1:  ocr-publish --config <path>
    Terminal 2:  ocr-train   --config <path>    (after publisher prints "Publisher starting")
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ocr-publish",
        description="Stream HuggingFace -> augment -> publish to Redis for ocr-train.",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to training YAML config",
    )
    parser.add_argument(
        "--redis-host",
        default=None,
        metavar="HOST",
        help="Override redis.host from config",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Override redis.port from config",
    )
    parser.add_argument(
        "--no-flush",
        action="store_true",
        help="Do NOT flush Redis on startup (useful when resuming a crashed publisher)",
    )
    args = parser.parse_args()

    # Lazy imports so --help is instant
    from ocr_aster.config.loader import load_config
    from ocr_aster.data.publisher import HFRedisPublisher

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # CLI overrides take priority over YAML redis section
    if args.redis_host:
        config.redis.host = args.redis_host
    if args.redis_port:
        config.redis.port = args.redis_port

    publisher = None
    try:
        publisher = HFRedisPublisher(
            config=config,
            flush_on_start=not args.no_flush,
        )
        logger.info(
            f"Publisher ready — redis={config.redis.host}:{config.redis.port}  "
            f"datasets={len(config.datasets)}  workers={config.redis.n_workers}\n"
            f"Now start the trainer:  ocr-train --config {args.config}"
        )
        publisher.run()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — stopping publisher.")
    except ConnectionError as exc:
        logger.error(f"Cannot connect to Redis: {exc}")
        logger.error("Make sure Redis is running:  redis-server")
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Publisher error: {exc}")
        sys.exit(1)
    finally:
        if publisher is not None:
            publisher.stop()
        logger.info("Publisher stopped.")


if __name__ == "__main__":
    main()

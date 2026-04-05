"""
YAML config loader with environment variable substitution.

Usage:
    config = load_config("configs/training/aster_v2_iiit5k.yaml")

Environment variable substitution:
    Any ${VAR_NAME} in the YAML is replaced with os.environ["VAR_NAME"].
    Missing env vars raise a clear error at load time.
"""

from __future__ import annotations

import os
import re
import yaml
from pathlib import Path

from ocr_aster.config.schema import TrainingConfig


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR} with os.environ['VAR']. Raises if var is missing."""

    def replace(match: re.Match) -> str:
        var = match.group(1)
        if var not in os.environ:
            raise EnvironmentError(
                f"Config references environment variable '${{{var}}}' "
                f"but it is not set. Export it before running."
            )
        return os.environ[var]

    return _ENV_VAR_PATTERN.sub(replace, text)


def load_config(path: str | Path) -> TrainingConfig:
    """
    Load and validate a training config from a YAML file.

    Steps:
      1. Read raw YAML text
      2. Substitute ${ENV_VAR} references
      3. Parse YAML → dict
      4. Validate with Pydantic v2 (unknown fields raise ValidationError)
      5. Return TrainingConfig

    Args:
        path: path to the YAML config file

    Returns:
        Validated TrainingConfig instance

    Raises:
        FileNotFoundError: if the YAML file does not exist
        EnvironmentError: if a referenced env var is missing
        pydantic.ValidationError: if any field is invalid or unknown
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    raw = _substitute_env_vars(raw)
    data = yaml.safe_load(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(data)}")

    return TrainingConfig(**data)

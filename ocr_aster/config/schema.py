"""
Pydantic v2 schemas for all configuration.

One YAML file = one experiment. All fields validated at load time —
unknown fields raise errors, required fields are documented here.
"""

from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetSourceConfig(BaseModel):
    """A single HuggingFace dataset source."""

    model_config = {"extra": "forbid"}

    repo_id: str = Field(..., description="HuggingFace repo, e.g. 'user/my-dataset'")
    split: str = Field("train", description="Dataset split: train / validation / test")
    weight: float = Field(1.0, gt=0.0, description="Sampling weight relative to other datasets")
    image_column: str = Field("image", description="Column name containing PIL images")
    label_column: str = Field("label", description="Column name containing text labels")
    streaming: bool = Field(True, description="Stream from Hub without full download")
    trust_remote_code: bool = Field(False)

    # Optional per-source filter (applied before curriculum filter)
    # e.g. "lambda s: len(s['label']) <= 10"
    source_filter: str | None = Field(None, description="Optional lambda filter string")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class AugmentationConfig(BaseModel):
    """Controls the augmentation pipeline applied by the publisher threads."""

    model_config = {"extra": "forbid"}

    enabled: bool = True
    level: str = Field(
        "medium",
        pattern="^(off|low|medium|high|all)$",
        description="off | low | medium | high | all",
    )


# ---------------------------------------------------------------------------
# Curriculum phase
# ---------------------------------------------------------------------------

class PhaseConfig(BaseModel):
    """
    A curriculum learning phase. Training transitions to the next phase
    automatically when `from_iter` is reached.
    """

    model_config = {"extra": "forbid"}

    name: str
    from_iter: int = Field(..., ge=0)
    to_iter: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    lr: float = Field(..., gt=0.0)
    augmentation_level: str = Field(
        "medium",
        pattern="^(off|low|medium|high|all)$",
    )
    # Optional lambda filter on samples for this phase
    # e.g. "lambda s: len(s['label']) <= 6"
    dataset_filter: str | None = None

    @model_validator(mode="after")
    def to_iter_after_from_iter(self) -> PhaseConfig:
        if self.to_iter <= self.from_iter:
            raise ValueError(
                f"Phase '{self.name}': to_iter ({self.to_iter}) must be > from_iter ({self.from_iter})"
            )
        return self


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

class MLflowConfig(BaseModel):
    """MLflow experiment tracking settings."""

    model_config = {"extra": "forbid"}

    enabled: bool = True
    tracking_uri: str = Field("mlruns/", description="Local path or remote URI")
    experiment_name: str = "aster-convnext"
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main training config
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """
    Full training configuration. Loaded from a YAML file via config/loader.py.

    Architecture is fixed: ConvNeXt → BiLSTM → ASTER v2.
    This config only controls sizes, training dynamics, and data.
    """

    model_config = {"extra": "forbid"}

    # ── Experiment ──────────────────────────────────────────────────────────
    experiment_name: str = "aster-v2-run"

    # ── Model geometry ──────────────────────────────────────────────────────
    imgH: int = Field(120, gt=0, description="Input image height")
    imgW: int = Field(280, gt=0, description="Input image width")
    input_channel: int = Field(1, description="1 = grayscale, 3 = RGB")
    output_channel: int = Field(512, gt=0, description="ConvNeXt output = BiLSTM hidden size")
    hidden_size: int = Field(512, gt=0, description="BiLSTM and GRU hidden size")
    embed_dim: int = Field(256, gt=0, description="Character embedding dimension")
    batch_max_length: int = Field(28, gt=0, description="Maximum output sequence length")
    sensitive: bool = Field(True, description="Case-sensitive character set")

    # ── Character set ────────────────────────────────────────────────────────
    # Built by concatenating number + symbol + lang_char in loader,
    # or set directly here.
    character: str = Field(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        min_length=1,
    )

    # ── Scheduled teacher forcing ────────────────────────────────────────────
    tf_start: float = Field(1.0, ge=0.0, le=1.0)
    tf_end: float = Field(0.3, ge=0.0, le=1.0)
    tf_decay_iters: int = Field(300_000, gt=0)

    # ── Training dynamics ────────────────────────────────────────────────────
    batch_size: int = Field(32, gt=0)
    num_iter: int = Field(300_000, gt=0)
    lr: float = Field(1e-4, gt=0.0)
    weight_decay: float = Field(0.02, ge=0.0)
    beta1: float = Field(0.9, gt=0.0, lt=1.0)
    beta2: float = Field(0.999, gt=0.0, lt=1.0)
    eps: float = Field(1e-8, gt=0.0)
    grad_clip: float = Field(1.0, gt=0.0)
    manualSeed: int = 1111

    # ── Checkpointing ────────────────────────────────────────────────────────
    checkpoints_dir: str = "checkpoints/"
    saved_model: str = Field("", description="Path to checkpoint to resume from")
    save_every_n_iterations: int = Field(10_000, gt=0)
    valInterval: int = Field(25_000, gt=0)
    save_metrics_csv_every: int = Field(100, gt=0)
    save_log_gradient_every: int = Field(100, gt=0)

    # ── Data ─────────────────────────────────────────────────────────────────
    datasets: list[DatasetSourceConfig] = Field(
        ..., min_length=1, description="One or more HuggingFace training datasets"
    )
    val_dataset: DatasetSourceConfig = Field(
        ..., description="HuggingFace validation dataset"
    )

    # ── Curriculum ───────────────────────────────────────────────────────────
    phases: list[PhaseConfig] = Field(default_factory=list)

    # ── Augmentation ─────────────────────────────────────────────────────────
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    # ── Derived (computed, not set in YAML) ──────────────────────────────────
    num_class: int = 0  # set after build_character_set()

    @model_validator(mode="after")
    def tf_end_lte_tf_start(self) -> TrainingConfig:
        if self.tf_end > self.tf_start:
            raise ValueError(
                f"tf_end ({self.tf_end}) must be <= tf_start ({self.tf_start})"
            )
        return self

    @model_validator(mode="after")
    def phases_cover_full_range(self) -> TrainingConfig:
        """If phases are provided, they must be non-overlapping and contiguous."""
        if not self.phases:
            return self
        sorted_phases = sorted(self.phases, key=lambda p: p.from_iter)
        for i in range(1, len(sorted_phases)):
            prev, curr = sorted_phases[i - 1], sorted_phases[i]
            if curr.from_iter != prev.to_iter:
                raise ValueError(
                    f"Gap or overlap between phase '{prev.name}' (to_iter={prev.to_iter}) "
                    f"and '{curr.name}' (from_iter={curr.from_iter})"
                )
        return self

    def build_character_set(self) -> str:
        """
        Returns the full character set with GO and EOS tokens prepended.
        Also sets self.num_class.

        The converter reserves index 0 = [GO], 1 = [EOS], 2..N = characters.
        """
        chars = list(dict.fromkeys(self.character))  # deduplicate, preserve order
        self.num_class = len(chars) + 2  # +2 for GO and EOS
        return "".join(chars)

    def teacher_forcing_ratio(self, current_iter: int) -> float:
        """Linear decay schedule for teacher forcing ratio τ."""
        progress = min(current_iter / self.tf_decay_iters, 1.0)
        return self.tf_start - (self.tf_start - self.tf_end) * progress

    def active_phase(self, current_iter: int) -> PhaseConfig | None:
        """Return the currently active curriculum phase, or None if no phases."""
        for phase in self.phases:
            if phase.from_iter <= current_iter < phase.to_iter:
                return phase
        return None

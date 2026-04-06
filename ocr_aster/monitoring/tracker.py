"""
MLflow experiment tracker.

Wraps mlflow so the training loop never imports mlflow directly.
All side-effects (run creation, metric logging, artifact upload) are
contained here and can be swapped or silenced in tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from ocr_aster.train.metrics import ValidationResult


class ExperimentTracker:
    """
    Thin wrapper around an active MLflow run.

    Usage::

        tracker = ExperimentTracker.start(
            experiment_name="aster_v2_iiit5k",
            run_name="baseline",
            tracking_uri="http://localhost:5000",
            params={"lr": 1e-4, "batch_size": 64},
        )
        ...
        tracker.log_train_step(iteration=100, loss=2.4, tf_ratio=0.9)
        tracker.log_validation(result)
        tracker.log_artifact(Path("checkpoints/best.pth"))
        tracker.finish()
    """

    def __init__(self, run: mlflow.ActiveRun) -> None:
        self._run = run

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> "ExperimentTracker":
        """
        Create or resume an MLflow run and return a tracker bound to it.

        Args:
            experiment_name: MLflow experiment name (created if absent)
            run_name:        human-readable run label
            tracking_uri:    MLflow server URI; falls back to env / local ./mlruns
            params:          hyper-parameters logged once at run start
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)

        if params:
            mlflow.log_params(params)

        return cls(run)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_train_step(
        self,
        iteration: int,
        loss: float,
        tf_ratio: float,
    ) -> None:
        """Log scalars for a single training iteration."""
        mlflow.log_metrics(
            {
                "train/loss": loss,
                "train/teacher_forcing_ratio": tf_ratio,
            },
            step=iteration,
        )

    def log_validation(self, result: ValidationResult) -> None:
        """Log all metrics from a ValidationResult."""
        metrics: dict[str, float] = {
            "val/accuracy": result.accuracy,
            "val/cer": result.cer,
            "val/norm_edit_distance": result.norm_edit_distance,
            "val/loss": result.val_loss,
            "val/calibration_gap": result.calibration_gap,
        }

        for group, acc in result.accuracy_by_length.items():
            key = group.replace("+", "plus")  # MLflow dislikes '+' in keys
            metrics[f"val/accuracy_by_length/{key}"] = acc

        mlflow.log_metrics(metrics, step=result.iteration)

    def log_artifact(self, path: Path) -> None:
        """Upload a local file or directory as an MLflow artifact."""
        if path.is_dir():
            mlflow.log_artifacts(str(path))
        else:
            mlflow.log_artifact(str(path))

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()

    # Context-manager support so callers can use `with ExperimentTracker.start(...)`
    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()

    # ------------------------------------------------------------------
    # Introspection (useful in tests)
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

"""
Tests for ExperimentTracker — all MLflow calls are mocked so no server is needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from ocr_aster.monitoring.tracker import ExperimentTracker
from ocr_aster.train.metrics import ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(**kwargs) -> ValidationResult:
    defaults = dict(
        iteration=1000,
        num_samples=500,
        accuracy=0.42,
        cer=0.19,
        norm_edit_distance=0.84,
        val_loss=1.48,
        accuracy_by_length={"1-5": 0.68, "6-10": 0.38, "11-20": 0.12, "21+": 0.0},
        counts_by_length={"1-5": 300, "6-10": 150, "11-20": 45, "21+": 5},
        top_confusions=[("a", "o", 30)],
        avg_conf_correct=0.72,
        avg_conf_incorrect=0.18,
    )
    defaults.update(kwargs)
    return ValidationResult(**defaults)


# ---------------------------------------------------------------------------
# Fixture — patch the entire mlflow module used inside tracker
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mlflow():
    """Patch mlflow in the tracker module and yield the mock."""
    with patch("ocr_aster.monitoring.tracker.mlflow") as m:
        # start_run returns a context-manager-like object with .info.run_id
        fake_run = MagicMock()
        fake_run.info.run_id = "test-run-id-123"
        m.start_run.return_value = fake_run
        yield m


# ---------------------------------------------------------------------------
# ExperimentTracker.start
# ---------------------------------------------------------------------------

class TestTrackerStart:
    def test_sets_experiment(self, mock_mlflow):
        ExperimentTracker.start("my-exp")
        mock_mlflow.set_experiment.assert_called_once_with("my-exp")

    def test_starts_run_with_name(self, mock_mlflow):
        ExperimentTracker.start("my-exp", run_name="baseline")
        mock_mlflow.start_run.assert_called_once_with(run_name="baseline")

    def test_sets_tracking_uri_when_given(self, mock_mlflow):
        ExperimentTracker.start("my-exp", tracking_uri="http://localhost:5000")
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_no_tracking_uri_call_when_omitted(self, mock_mlflow):
        ExperimentTracker.start("my-exp")
        mock_mlflow.set_tracking_uri.assert_not_called()

    def test_logs_params_when_given(self, mock_mlflow):
        ExperimentTracker.start("my-exp", params={"lr": 1e-4, "batch_size": 32})
        mock_mlflow.log_params.assert_called_once_with({"lr": 1e-4, "batch_size": 32})

    def test_no_log_params_when_omitted(self, mock_mlflow):
        ExperimentTracker.start("my-exp")
        mock_mlflow.log_params.assert_not_called()

    def test_run_id_accessible(self, mock_mlflow):
        tracker = ExperimentTracker.start("my-exp")
        assert tracker.run_id == "test-run-id-123"


# ---------------------------------------------------------------------------
# log_train_step
# ---------------------------------------------------------------------------

class TestLogTrainStep:
    def test_logs_correct_keys(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        tracker.log_train_step(iteration=500, loss=2.3, tf_ratio=0.85)
        mock_mlflow.log_metrics.assert_called_once_with(
            {"train/loss": 2.3, "train/teacher_forcing_ratio": 0.85},
            step=500,
        )

    def test_step_matches_iteration(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        tracker.log_train_step(iteration=12345, loss=1.0, tf_ratio=0.5)
        _, kwargs = mock_mlflow.log_metrics.call_args
        assert kwargs["step"] == 12345


# ---------------------------------------------------------------------------
# log_validation
# ---------------------------------------------------------------------------

class TestLogValidation:
    def test_logs_core_metrics(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        result = _make_result()
        tracker.log_validation(result)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "val/accuracy" in logged
        assert "val/cer" in logged
        assert "val/norm_edit_distance" in logged
        assert "val/loss" in logged
        assert "val/calibration_gap" in logged

    def test_logs_accuracy_by_length(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        result = _make_result()
        tracker.log_validation(result)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert "val/accuracy_by_length/1-5" in logged
        assert "val/accuracy_by_length/6-10" in logged
        assert "val/accuracy_by_length/11-20" in logged
        # "21+" → "21plus" to avoid '+' in MLflow key
        assert "val/accuracy_by_length/21plus" in logged

    def test_plus_sign_sanitized(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        result = _make_result()
        tracker.log_validation(result)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        for key in logged:
            assert "+" not in key, f"'+' found in MLflow key: {key!r}"

    def test_step_matches_iteration(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        result = _make_result(iteration=25000)
        tracker.log_validation(result)

        _, kwargs = mock_mlflow.log_metrics.call_args
        assert kwargs["step"] == 25000

    def test_metric_values_correct(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        result = _make_result(accuracy=0.55, cer=0.10)
        tracker.log_validation(result)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert logged["val/accuracy"] == pytest.approx(0.55)
        assert logged["val/cer"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# log_artifact
# ---------------------------------------------------------------------------

class TestLogArtifact:
    def test_file_uses_log_artifact(self, mock_mlflow, tmp_path):
        f = tmp_path / "model.pth"
        f.write_bytes(b"fake")
        tracker = ExperimentTracker.start("exp")
        tracker.log_artifact(f)
        mock_mlflow.log_artifact.assert_called_once_with(str(f))

    def test_directory_uses_log_artifacts(self, mock_mlflow, tmp_path):
        d = tmp_path / "checkpoints"
        d.mkdir()
        tracker = ExperimentTracker.start("exp")
        tracker.log_artifact(d)
        mock_mlflow.log_artifacts.assert_called_once_with(str(d))


# ---------------------------------------------------------------------------
# set_tag
# ---------------------------------------------------------------------------

class TestSetTag:
    def test_delegates_to_mlflow(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        tracker.set_tag("best_iter", "50000")
        mock_mlflow.set_tag.assert_called_once_with("best_iter", "50000")


# ---------------------------------------------------------------------------
# Lifecycle — finish / context manager
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_finish_ends_run(self, mock_mlflow):
        tracker = ExperimentTracker.start("exp")
        tracker.finish()
        mock_mlflow.end_run.assert_called_once()

    def test_context_manager_calls_finish(self, mock_mlflow):
        with ExperimentTracker.start("exp"):
            pass
        mock_mlflow.end_run.assert_called_once()

    def test_context_manager_returns_tracker(self, mock_mlflow):
        with ExperimentTracker.start("exp") as t:
            assert isinstance(t, ExperimentTracker)

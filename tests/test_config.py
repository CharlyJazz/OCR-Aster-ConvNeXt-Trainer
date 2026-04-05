"""
Tests for config schema (Pydantic v2) and YAML loader.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from ocr_aster.config.schema import (
    AugmentationConfig,
    DatasetSourceConfig,
    MLflowConfig,
    PhaseConfig,
    TrainingConfig,
)
from ocr_aster.config.loader import load_config


# ---------------------------------------------------------------------------
# Minimal valid config dict (reused across tests)
# ---------------------------------------------------------------------------

def _minimal() -> dict:
    return {
        "experiment_name": "test-run",
        "character": "abc123",
        "datasets": [
            {
                "repo_id": "user/test-dataset",
                "split": "train",
                "image_column": "image",
                "label_column": "label",
            }
        ],
        "val_dataset": {
            "repo_id": "user/test-dataset",
            "split": "test",
            "image_column": "image",
            "label_column": "label",
        },
    }


# ---------------------------------------------------------------------------
# TrainingConfig — valid cases
# ---------------------------------------------------------------------------

class TestTrainingConfigValid:
    def test_minimal_config_loads(self):
        cfg = TrainingConfig(**_minimal())
        assert cfg.experiment_name == "test-run"

    def test_defaults_are_applied(self):
        cfg = TrainingConfig(**_minimal())
        assert cfg.imgH == 120
        assert cfg.imgW == 280
        assert cfg.output_channel == 512
        assert cfg.hidden_size == 512
        assert cfg.batch_size == 32
        assert cfg.tf_start == 1.0
        assert cfg.tf_end == 0.3

    def test_augmentation_defaults(self):
        cfg = TrainingConfig(**_minimal())
        assert cfg.augmentation.enabled is True
        assert cfg.augmentation.level == "medium"

    def test_mlflow_defaults(self):
        cfg = TrainingConfig(**_minimal())
        assert cfg.mlflow.enabled is True
        assert cfg.mlflow.tracking_uri == "mlruns/"

    def test_empty_phases_ok(self):
        cfg = TrainingConfig(**_minimal())
        assert cfg.phases == []

    def test_override_fields(self):
        d = _minimal()
        d["batch_size"] = 64
        d["lr"] = 0.001
        cfg = TrainingConfig(**d)
        assert cfg.batch_size == 64
        assert cfg.lr == 0.001


# ---------------------------------------------------------------------------
# TrainingConfig — invalid cases
# ---------------------------------------------------------------------------

class TestTrainingConfigInvalid:
    def test_unknown_field_raises(self):
        d = _minimal()
        d["totally_unknown_field"] = "oops"
        with pytest.raises(ValidationError, match="Extra inputs"):
            TrainingConfig(**d)

    def test_tf_end_gt_tf_start_raises(self):
        d = _minimal()
        d["tf_start"] = 0.3
        d["tf_end"] = 0.9
        with pytest.raises(ValidationError, match="tf_end"):
            TrainingConfig(**d)

    def test_negative_batch_size_raises(self):
        d = _minimal()
        d["batch_size"] = -1
        with pytest.raises(ValidationError):
            TrainingConfig(**d)

    def test_empty_character_set_raises(self):
        d = _minimal()
        d["character"] = ""
        with pytest.raises(ValidationError):
            TrainingConfig(**d)

    def test_missing_datasets_raises(self):
        d = _minimal()
        del d["datasets"]
        with pytest.raises(ValidationError):
            TrainingConfig(**d)

    def test_empty_datasets_list_raises(self):
        d = _minimal()
        d["datasets"] = []
        with pytest.raises(ValidationError):
            TrainingConfig(**d)

    def test_missing_val_dataset_raises(self):
        d = _minimal()
        del d["val_dataset"]
        with pytest.raises(ValidationError):
            TrainingConfig(**d)


# ---------------------------------------------------------------------------
# PhaseConfig
# ---------------------------------------------------------------------------

class TestPhaseConfig:
    def test_valid_phase(self):
        p = PhaseConfig(name="p1", from_iter=0, to_iter=50000,
                        batch_size=32, lr=1e-4)
        assert p.name == "p1"

    def test_to_iter_before_from_iter_raises(self):
        with pytest.raises(ValidationError, match="to_iter"):
            PhaseConfig(name="bad", from_iter=50000, to_iter=1000,
                        batch_size=32, lr=1e-4)

    def test_equal_iters_raises(self):
        with pytest.raises(ValidationError):
            PhaseConfig(name="bad", from_iter=100, to_iter=100,
                        batch_size=32, lr=1e-4)

    def test_invalid_augmentation_level_raises(self):
        with pytest.raises(ValidationError):
            PhaseConfig(name="p", from_iter=0, to_iter=100,
                        batch_size=32, lr=1e-4, augmentation_level="extreme")


# ---------------------------------------------------------------------------
# Phases contiguity validation
# ---------------------------------------------------------------------------

class TestPhasesContiguity:
    def _phases(self, ranges: list[tuple[int, int]]) -> list[dict]:
        return [
            {"name": f"p{i}", "from_iter": a, "to_iter": b, "batch_size": 32, "lr": 1e-4}
            for i, (a, b) in enumerate(ranges)
        ]

    def test_contiguous_phases_ok(self):
        d = _minimal()
        d["phases"] = self._phases([(0, 100), (100, 200), (200, 500)])
        cfg = TrainingConfig(**d)
        assert len(cfg.phases) == 3

    def test_gap_between_phases_raises(self):
        d = _minimal()
        d["phases"] = self._phases([(0, 100), (150, 200)])  # gap at 100-150
        with pytest.raises(ValidationError, match="Gap or overlap"):
            TrainingConfig(**d)

    def test_overlap_between_phases_raises(self):
        d = _minimal()
        d["phases"] = self._phases([(0, 150), (100, 200)])  # overlap 100-150
        with pytest.raises(ValidationError, match="Gap or overlap"):
            TrainingConfig(**d)


# ---------------------------------------------------------------------------
# Derived methods
# ---------------------------------------------------------------------------

class TestDerivedMethods:
    def test_build_character_set_deduplicates(self):
        d = _minimal()
        d["character"] = "aaabbbccc"
        cfg = TrainingConfig(**d)
        chars = cfg.build_character_set()
        assert chars == "abc"
        assert cfg.num_class == 5  # 3 chars + GO + EOS

    def test_build_character_set_sets_num_class(self):
        cfg = TrainingConfig(**_minimal())
        cfg.build_character_set()
        assert cfg.num_class == len(set("abc123")) + 2

    def test_teacher_forcing_at_iter_0(self):
        cfg = TrainingConfig(**_minimal())
        assert cfg.teacher_forcing_ratio(0) == pytest.approx(1.0)

    def test_teacher_forcing_at_end(self):
        cfg = TrainingConfig(**_minimal())
        ratio = cfg.teacher_forcing_ratio(cfg.tf_decay_iters)
        assert ratio == pytest.approx(cfg.tf_end)

    def test_teacher_forcing_clamped_after_decay(self):
        cfg = TrainingConfig(**_minimal())
        ratio = cfg.teacher_forcing_ratio(cfg.tf_decay_iters * 10)
        assert ratio == pytest.approx(cfg.tf_end)

    def test_teacher_forcing_midpoint(self):
        cfg = TrainingConfig(**_minimal())
        mid = cfg.tf_decay_iters // 2
        ratio = cfg.teacher_forcing_ratio(mid)
        expected = (cfg.tf_start + cfg.tf_end) / 2
        assert ratio == pytest.approx(expected, abs=1e-3)

    def test_active_phase_no_phases(self):
        cfg = TrainingConfig(**_minimal())
        assert cfg.active_phase(0) is None

    def test_active_phase_correct(self):
        d = _minimal()
        d["phases"] = [
            {"name": "p1", "from_iter": 0, "to_iter": 100, "batch_size": 32, "lr": 1e-4},
            {"name": "p2", "from_iter": 100, "to_iter": 200, "batch_size": 64, "lr": 5e-5},
        ]
        cfg = TrainingConfig(**d)
        assert cfg.active_phase(0).name == "p1"
        assert cfg.active_phase(50).name == "p1"
        assert cfg.active_phase(100).name == "p2"
        assert cfg.active_phase(199).name == "p2"
        assert cfg.active_phase(200) is None


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

class TestLoader:
    def test_load_valid_yaml(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""
            experiment_name: yaml-test
            character: "abc"
            datasets:
              - repo_id: user/ds
                split: train
                image_column: image
                label_column: label
            val_dataset:
              repo_id: user/ds
              split: test
              image_column: image
              label_column: label
        """)
        f = tmp_path / "config.yaml"
        f.write_text(yaml_content)
        cfg = load_config(f)
        assert cfg.experiment_name == "yaml-test"

    def test_load_nonexistent_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_env_var_substitution(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TEST_REPO", "myuser/myrepo")
        yaml_content = textwrap.dedent("""
            experiment_name: env-test
            character: "abc"
            datasets:
              - repo_id: "${TEST_REPO}"
                split: train
                image_column: image
                label_column: label
            val_dataset:
              repo_id: "${TEST_REPO}"
              split: test
              image_column: image
              label_column: label
        """)
        f = tmp_path / "config.yaml"
        f.write_text(yaml_content)
        cfg = load_config(f)
        assert cfg.datasets[0].repo_id == "myuser/myrepo"

    def test_missing_env_var_raises(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        yaml_content = textwrap.dedent("""
            experiment_name: env-test
            character: "abc"
            datasets:
              - repo_id: "${MISSING_VAR}"
                split: train
                image_column: image
                label_column: label
            val_dataset:
              repo_id: "user/ds"
              split: test
              image_column: image
              label_column: label
        """)
        f = tmp_path / "config.yaml"
        f.write_text(yaml_content)
        with pytest.raises(EnvironmentError, match="MISSING_VAR"):
            load_config(f)

    def test_load_example_iiit5k_config(self):
        """The bundled example config must load without errors."""
        cfg = load_config("configs/training/aster_v2_iiit5k.yaml")
        assert cfg.experiment_name == "aster-v2-iiit5k-baseline"
        assert len(cfg.datasets) == 1
        assert cfg.phases == []

    def test_load_example_curriculum_config(self):
        cfg = load_config("configs/training/aster_v2_curriculum.yaml")
        assert len(cfg.phases) == 3

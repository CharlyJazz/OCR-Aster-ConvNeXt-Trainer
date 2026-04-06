"""
Tests for all metric classes and ValidationResult.
"""

from __future__ import annotations

import pytest

from ocr_aster.train.metrics import (
    AccuracyByLength,
    CharacterErrorRate,
    ConfidenceCalibration,
    ExactMatchAccuracy,
    NormEditDistance,
    TopKCharacterConfusions,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# ExactMatchAccuracy
# ---------------------------------------------------------------------------

class TestExactMatchAccuracy:
    def test_all_correct(self):
        m = ExactMatchAccuracy()
        m.update(["hello", "world"], ["hello", "world"])
        assert m.value == pytest.approx(1.0)

    def test_none_correct(self):
        m = ExactMatchAccuracy()
        m.update(["abc", "xyz"], ["def", "uvw"])
        assert m.value == pytest.approx(0.0)

    def test_partial(self):
        m = ExactMatchAccuracy()
        m.update(["a", "b", "c", "d"], ["a", "x", "c", "x"])
        assert m.value == pytest.approx(0.5)

    def test_case_sensitive(self):
        m = ExactMatchAccuracy()
        m.update(["Hello"], ["hello"])
        assert m.value == pytest.approx(0.0)

    def test_empty_no_crash(self):
        m = ExactMatchAccuracy()
        assert m.value == 0.0

    def test_reset(self):
        m = ExactMatchAccuracy()
        m.update(["a"], ["a"])
        m.reset()
        assert m.value == 0.0

    def test_accumulates_across_updates(self):
        m = ExactMatchAccuracy()
        m.update(["a"], ["a"])   # 1/1
        m.update(["b"], ["x"])   # 0/1
        assert m.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CharacterErrorRate
# ---------------------------------------------------------------------------

class TestCER:
    def test_perfect_prediction(self):
        m = CharacterErrorRate()
        m.update(["hello"], ["hello"])
        assert m.value == pytest.approx(0.0)

    def test_completely_wrong(self):
        # edit_dist("abc", "xyz") = 3, len("xyz") = 3 → CER = 1.0
        m = CharacterErrorRate()
        m.update(["abc"], ["xyz"])
        assert m.value == pytest.approx(1.0)

    def test_one_substitution(self):
        # edit_dist("hxllo", "hello") = 1, len("hello") = 5 → CER = 0.2
        m = CharacterErrorRate()
        m.update(["hxllo"], ["hello"])
        assert m.value == pytest.approx(0.2)

    def test_empty_prediction(self):
        # edit_dist("", "abc") = 3, len("abc") = 3 → CER = 1.0
        m = CharacterErrorRate()
        m.update([""], ["abc"])
        assert m.value == pytest.approx(1.0)

    def test_empty_gt_no_crash(self):
        m = CharacterErrorRate()
        m.update(["abc"], [""])
        # max(len(""), 1) = 1, edit_dist("abc","") = 3 → CER = 3.0 (clipped by convention)
        assert m.value >= 0.0

    def test_reset(self):
        m = CharacterErrorRate()
        m.update(["abc"], ["xyz"])
        m.reset()
        assert m.value == 0.0


# ---------------------------------------------------------------------------
# NormEditDistance
# ---------------------------------------------------------------------------

class TestNED:
    def test_perfect(self):
        m = NormEditDistance()
        m.update(["hello"], ["hello"])
        assert m.value == pytest.approx(1.0)

    def test_completely_wrong(self):
        # edit_dist("abc","xyz")=3, max(3,3)=3 → NED = 1 - 1 = 0.0
        m = NormEditDistance()
        m.update(["abc"], ["xyz"])
        assert m.value == pytest.approx(0.0)

    def test_partial(self):
        # edit_dist("hxllo","hello")=1, max=5 → NED = 1 - 0.2 = 0.8
        m = NormEditDistance()
        m.update(["hxllo"], ["hello"])
        assert m.value == pytest.approx(0.8)

    def test_reset(self):
        m = NormEditDistance()
        m.update(["abc"], ["xyz"])
        m.reset()
        assert m.value == 0.0


# ---------------------------------------------------------------------------
# AccuracyByLength
# ---------------------------------------------------------------------------

class TestAccuracyByLength:
    def test_short_group(self):
        m = AccuracyByLength()
        m.update(["ab", "cd"], ["ab", "xy"])
        vals = m.value
        assert vals["1-5"] == pytest.approx(0.5)

    def test_medium_group(self):
        m = AccuracyByLength()
        m.update(["abcdefg"], ["abcdefg"])  # len=7 → "6-10"
        vals = m.value
        assert vals["6-10"] == pytest.approx(1.0)

    def test_long_group(self):
        m = AccuracyByLength()
        word = "abcdefghijk"   # len=11 → "11-20"
        m.update([word], [word])
        assert m.value["11-20"] == pytest.approx(1.0)

    def test_very_long_group(self):
        m = AccuracyByLength()
        word = "a" * 25  # len=25 → "21+"
        m.update([word], [word])
        assert m.value["21+"] == pytest.approx(1.0)

    def test_zero_for_empty_group(self):
        m = AccuracyByLength()
        m.update(["ab"], ["ab"])  # only "1-5" populated
        assert m.value["6-10"] == 0.0
        assert m.counts["6-10"] == 0

    def test_counts_correct(self):
        m = AccuracyByLength()
        m.update(["ab", "cd", "ef"], ["ab", "cd", "ef"])
        assert m.counts["1-5"] == 3

    def test_reset(self):
        m = AccuracyByLength()
        m.update(["ab"], ["ab"])
        m.reset()
        assert m.value["1-5"] == 0.0


# ---------------------------------------------------------------------------
# TopKCharacterConfusions
# ---------------------------------------------------------------------------

class TestTopKConfusions:
    def test_counts_substitutions(self):
        m = TopKCharacterConfusions(k=5)
        m.update(["hxllo", "hxllo"], ["hello", "hello"])
        confusions = m.value
        # 'e' → 'x' appears twice
        pairs = {(g, p): c for g, p, c in confusions}
        assert pairs[("e", "x")] == 2

    def test_perfect_predictions_no_confusions(self):
        m = TopKCharacterConfusions()
        m.update(["hello", "world"], ["hello", "world"])
        assert m.value == []

    def test_top_k_limit(self):
        m = TopKCharacterConfusions(k=3)
        # Create many different confusions
        preds = [chr(ord('a') + i) for i in range(10)]
        gts = [chr(ord('z') - i) for i in range(10)]
        m.update(preds, gts)
        assert len(m.value) <= 3

    def test_reset(self):
        m = TopKCharacterConfusions()
        m.update(["ax"], ["ay"])
        m.reset()
        assert m.value == []


# ---------------------------------------------------------------------------
# ConfidenceCalibration
# ---------------------------------------------------------------------------

class TestConfidenceCalibration:
    def test_high_confidence_correct(self):
        m = ConfidenceCalibration()
        m.update(["a", "b"], ["a", "b"], [0.9, 0.8])
        assert m.avg_correct == pytest.approx(0.85)
        assert m.avg_incorrect == 0.0

    def test_low_confidence_incorrect(self):
        m = ConfidenceCalibration()
        m.update(["x", "y"], ["a", "b"], [0.2, 0.3])
        assert m.avg_incorrect == pytest.approx(0.25)
        assert m.avg_correct == 0.0

    def test_calibration_gap(self):
        m = ConfidenceCalibration()
        m.update(["a", "x"], ["a", "b"], [0.9, 0.2])
        assert m.calibration_gap == pytest.approx(0.7)

    def test_reset(self):
        m = ConfidenceCalibration()
        m.update(["a"], ["a"], [0.9])
        m.reset()
        assert m.avg_correct == 0.0


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_summary_runs(self):
        r = ValidationResult(
            iteration=1000,
            num_samples=500,
            accuracy=0.42,
            cer=0.19,
            norm_edit_distance=0.84,
            val_loss=1.48,
            accuracy_by_length={"1-5": 0.68, "6-10": 0.38, "11-20": 0.12, "21+": 0.0},
            counts_by_length={"1-5": 300, "6-10": 150, "11-20": 45, "21+": 5},
            top_confusions=[("a", "o", 30), ("l", "1", 20)],
            avg_conf_correct=0.72,
            avg_conf_incorrect=0.18,
        )
        summary = r.summary()
        assert "42.00%" in summary
        assert "0.1900" in summary
        assert "calibration" in summary.lower()

    def test_calibration_gap(self):
        r = ValidationResult(iteration=0, num_samples=10,
                             avg_conf_correct=0.8, avg_conf_incorrect=0.2)
        assert r.calibration_gap == pytest.approx(0.6)

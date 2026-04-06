"""
Tests for AugmentationPipeline — no GPU, no HuggingFace needed.
"""

from __future__ import annotations

import pytest
from PIL import Image

from ocr_aster.data.augmentation import AugmentationPipeline


def _grey_image(w: int = 128, h: int = 32) -> Image.Image:
    return Image.new("L", (w, h), color=128)


def _noisy_image(w: int = 128, h: int = 32) -> Image.Image:
    import random
    img = Image.new("L", (w, h))
    img.putdata([random.randint(0, 255) for _ in range(w * h)])
    return img


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_all_levels_construct(self):
        for level in ("off", "low", "medium", "high", "all"):
            p = AugmentationPipeline(level=level)
            assert p.level == level

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation level"):
            AugmentationPipeline(level="extreme")

    def test_off_has_no_transforms(self):
        p = AugmentationPipeline(level="off")
        assert len(p._transforms) == 0

    def test_higher_levels_have_more_transforms(self):
        counts = {
            level: len(AugmentationPipeline(level=level)._transforms)
            for level in ("off", "low", "medium", "high", "all")
        }
        assert counts["off"] <= counts["low"] <= counts["medium"]
        assert counts["medium"] <= counts["high"] <= counts["all"]

    def test_repr_contains_level(self):
        p = AugmentationPipeline(level="medium")
        assert "medium" in repr(p)


# ---------------------------------------------------------------------------
# Output shape & type
# ---------------------------------------------------------------------------

class TestOutputShape:
    @pytest.mark.parametrize("level", ["off", "low", "medium", "high", "all"])
    def test_output_is_pil_image(self, level):
        p = AugmentationPipeline(level=level)
        img = _grey_image()
        out = p(img)
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("level", ["off", "low", "medium", "high", "all"])
    def test_output_size_unchanged(self, level):
        p = AugmentationPipeline(level=level)
        img = _grey_image(w=128, h=32)
        out = p(img)
        assert out.size == (128, 32)

    @pytest.mark.parametrize("level", ["off", "low", "medium", "high", "all"])
    def test_output_mode_is_L(self, level):
        p = AugmentationPipeline(level=level)
        img = _grey_image()
        out = p(img)
        assert out.mode == "L"


# ---------------------------------------------------------------------------
# Determinism check — off level must be identity
# ---------------------------------------------------------------------------

class TestOffLevel:
    def test_off_returns_identical_pixels(self):
        import numpy as np
        p = AugmentationPipeline(level="off")
        img = _noisy_image()
        out = p(img)
        assert np.array_equal(np.array(img), np.array(out))


# ---------------------------------------------------------------------------
# No-crash on edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_tiny_image(self):
        p = AugmentationPipeline(level="high")
        img = _grey_image(w=8, h=8)
        out = p(img)
        assert isinstance(out, Image.Image)

    def test_wide_image(self):
        p = AugmentationPipeline(level="medium")
        img = _grey_image(w=512, h=32)
        out = p(img)
        assert out.size == (512, 32)

    def test_black_image_no_crash(self):
        p = AugmentationPipeline(level="all")
        img = Image.new("L", (64, 32), color=0)
        assert isinstance(p(img), Image.Image)

    def test_white_image_no_crash(self):
        p = AugmentationPipeline(level="all")
        img = Image.new("L", (64, 32), color=255)
        assert isinstance(p(img), Image.Image)

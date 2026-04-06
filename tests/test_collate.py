"""
Tests for AlignCollate — batch image processing and tensor conversion.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from ocr_aster.data.collate import AlignCollate


def _img(w: int, h: int, color: int = 128) -> Image.Image:
    return Image.new("L", (w, h), color=color)


# ---------------------------------------------------------------------------
# Basic output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_single_image_shape(self):
        collate = AlignCollate(imgH=32, imgW=128)
        batch = [(_img(64, 16), "hello")]
        tensors, labels = collate(batch)
        assert tensors.shape == (1, 1, 32, 128)

    def test_batch_shape(self):
        collate = AlignCollate(imgH=32, imgW=128)
        batch = [(_img(64, 16), f"word{i}") for i in range(4)]
        tensors, labels = collate(batch)
        assert tensors.shape == (4, 1, 32, 128)

    def test_labels_preserved(self):
        collate = AlignCollate(imgH=32, imgW=128)
        words = ["hello", "world", "ocr"]
        batch = [(_img(64, 16), w) for w in words]
        _, labels = collate(batch)
        assert labels == words


# ---------------------------------------------------------------------------
# Pixel range
# ---------------------------------------------------------------------------

class TestPixelRange:
    def test_values_in_0_1(self):
        collate = AlignCollate(imgH=32, imgW=128)
        batch = [(_img(64, 16, color=200), "a")]
        tensors, _ = collate(batch)
        assert tensors.min() >= 0.0
        assert tensors.max() <= 1.0

    def test_dtype_float32(self):
        collate = AlignCollate(imgH=32, imgW=128)
        batch = [(_img(64, 16), "a")]
        tensors, _ = collate(batch)
        assert tensors.dtype == torch.float32

    def test_black_image_near_zero(self):
        collate = AlignCollate(imgH=32, imgW=128)
        batch = [(Image.new("L", (64, 16), color=0), "a")]
        tensors, _ = collate(batch)
        assert tensors.max() < 0.01

    def test_white_image_near_one(self):
        collate = AlignCollate(imgH=32, imgW=128)
        batch = [(Image.new("L", (64, 16), color=255), "a")]
        tensors, _ = collate(batch)
        assert tensors.min() > 0.99


# ---------------------------------------------------------------------------
# RGB input is converted to grayscale
# ---------------------------------------------------------------------------

class TestGrayscaleConversion:
    def test_rgb_input_produces_L_tensor(self):
        collate = AlignCollate(imgH=32, imgW=128)
        rgb = Image.new("RGB", (64, 16), color=(100, 150, 200))
        tensors, _ = collate([(rgb, "test")])
        assert tensors.shape == (1, 1, 32, 128)


# ---------------------------------------------------------------------------
# keep_ratio mode
# ---------------------------------------------------------------------------

class TestKeepRatio:
    def test_output_shape_correct(self):
        collate = AlignCollate(imgH=32, imgW=128, keep_ratio=True)
        batch = [(_img(200, 50), "a")]
        tensors, _ = collate(batch)
        assert tensors.shape == (1, 1, 32, 128)

    def test_narrow_image_padded(self):
        # Very narrow image — most of width should be padding (≈ 0.5 / grey)
        collate = AlignCollate(imgH=32, imgW=128, keep_ratio=True)
        narrow = Image.new("L", (10, 32), color=0)
        tensors, _ = collate([(narrow, "x")])
        # At least half the pixels should be the grey pad value (127/255 ≈ 0.498)
        flat = tensors.flatten().tolist()
        grey_count = sum(0.45 < v < 0.55 for v in flat)
        assert grey_count > len(flat) * 0.3


# ---------------------------------------------------------------------------
# adjust_contrast flag (smoke test — just shouldn't crash)
# ---------------------------------------------------------------------------

class TestAdjustContrast:
    def test_no_crash(self):
        collate = AlignCollate(imgH=32, imgW=128, adjust_contrast=True)
        batch = [(_img(64, 16), "hi")]
        tensors, _ = collate(batch)
        assert tensors.shape == (1, 1, 32, 128)

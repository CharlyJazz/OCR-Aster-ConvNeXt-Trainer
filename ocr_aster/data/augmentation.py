"""
Image augmentation pipeline for OCR training.

Uses the real STRAug library and Albumentations with the same configuration
that was battle-tested in the private chord-OCR training runs.

Key design points (ported from private repo):
  1. STRAug transforms are grouped — only ONE per group is applied per image
     (pattern / blur / camera / geometry / etc.). This prevents stacking too
     many transforms of the same family and destroying legibility.
  2. 3-retry loop for STRAug: if a transform produces a near-black image,
     a different random transform from the same groups is tried up to 3 times.
  3. Albumentations pipeline is applied BEFORE STRAug.
  4. Parameters are deliberately conservative (soft mag, low prob) to avoid
     the black-image artefacts that plagued earlier training runs.

Two construction modes:

    AugmentationPipeline(level="medium")
        Level-based preset — used when no per-phase transform lists are given.

    AugmentationPipeline.from_lists(
        level="high",
        straug_augs=["Grid", "VGrid", "MotionBlur", "JpegCompression"],
        albumentations_augs=["PixelDropout", "OpticalDistortion"],
    )
        Explicit named lists — mirrors the per-phase YAML format.

All transforms operate on PIL Images (mode "L").
"""

from __future__ import annotations

import logging
import random
from functools import partial
from typing import Callable, List, Optional

import albumentations as A
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# STRAug imports — lazy try/except so unit tests still run if somehow absent
# ---------------------------------------------------------------------------
try:
    from straug.blur import DefocusBlur, GaussianBlur, GlassBlur, MotionBlur, ZoomBlur
    from straug.camera import Brightness, Contrast, JpegCompression, Pixelate
    from straug.geometry import Perspective as StraugPerspective
    from straug.geometry import Rotate, Shrink
    from straug.noise import GaussianNoise, ImpulseNoise, ShotNoise, SpeckleNoise
    from straug.pattern import EllipseGrid, Grid, HGrid, RectGrid, VGrid
    from straug.process import (
        AutoContrast, Color, Equalize, Invert, Posterize, Sharpness, Solarize
    )
    from straug.warp import Curve, Distort, Stretch
    from straug.weather import Fog, Frost, Rain, Shadow, Snow
    _HAVE_STRAUG = True
except ImportError:
    _HAVE_STRAUG = False
    logging.warning("straug not installed — STRAug transforms will be skipped")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-image detector (prevents augmentations from destroying images)
# ---------------------------------------------------------------------------

def is_image_mostly_black(
    img: Image.Image,
    std_threshold: float = 0.02,
    darkness_threshold: float = 0.95,
) -> bool:
    """
    Returns True if the image is nearly uniform AND dark.

    A near-black image (uniform dark pixels) indicates a failed augmentation.
    Threshold values are conservative — only truly degenerate images are flagged.

    Args:
        std_threshold:    below this std the image is considered uniform
        darkness_threshold: if mean < this value it is considered dark
                           (0.95 = catch everything except near-white images)
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    std_dev = float(np.std(arr))
    mean_val = float(np.mean(arr))
    return std_dev < std_threshold and mean_val < darkness_threshold


# ---------------------------------------------------------------------------
# STRAug transform registry
# Soft parameters (mag=0 or mag=1, low prob) to preserve legibility.
# ---------------------------------------------------------------------------

_STRAUG_MAP: dict[str, Callable] = {}

if _HAVE_STRAUG:
    _STRAUG_MAP = {
        # Pattern (grid artefacts simulating LCD/screen noise)
        "Grid":           lambda: partial(Grid(),           mag=0, prob=0.2),
        "VGrid":          lambda: partial(VGrid(),          mag=0, prob=0.2, max_width=1),
        "HGrid":          lambda: partial(HGrid(),          mag=0, prob=0.2, max_width=1),
        "RectGrid":       lambda: partial(RectGrid(),       mag=0, prob=0.2),
        "EllipseGrid":    lambda: partial(EllipseGrid(),    mag=0, prob=0.2),
        # Blur
        "GaussianBlur":   lambda: partial(GaussianBlur(),   mag=1, prob=0.4),
        "DefocusBlur":    lambda: partial(DefocusBlur(),    mag=1, prob=0.4),
        "MotionBlur":     lambda: partial(MotionBlur(),     mag=1, prob=0.4),
        "GlassBlur":      lambda: partial(GlassBlur(),      mag=1, prob=0.3),
        "ZoomBlur":       lambda: partial(ZoomBlur(),       mag=1, prob=0.3),
        # Noise
        "GaussianNoise":  lambda: partial(GaussianNoise(),  mag=1, prob=0.4),
        "ShotNoise":      lambda: partial(ShotNoise(),      mag=1, prob=0.4),
        "ImpulseNoise":   lambda: partial(ImpulseNoise(),   mag=1, prob=0.3),
        "SpeckleNoise":   lambda: partial(SpeckleNoise(),   mag=1, prob=0.4),
        # Weather
        "Fog":            lambda: partial(Fog(),            mag=1, prob=0.3),
        "Snow":           lambda: partial(Snow(),           mag=1, prob=0.3),
        "Frost":          lambda: partial(Frost(),          mag=1, prob=0.3),
        "Rain":           lambda: partial(Rain(),           mag=1, prob=0.3),
        "Shadow":         lambda: partial(Shadow(),         mag=1, prob=0.4),
        # Camera
        "Contrast":       lambda: partial(Contrast(),       mag=1, prob=0.4),
        "Brightness":     lambda: partial(Brightness(),     mag=1, prob=0.4),
        "JpegCompression": lambda: partial(JpegCompression(), mag=1, prob=0.6),
        "Pixelate":       lambda: partial(Pixelate(),       mag=1, prob=0.4),
        # Warp
        "Curve":          lambda: partial(Curve(),          mag=1, prob=0.3),
        "Distort":        lambda: partial(Distort(),        mag=1, prob=0.3),
        "Stretch":        lambda: partial(Stretch(),        mag=1, prob=0.3),
        # Geometry
        "StraugPerspective": lambda: partial(StraugPerspective(), mag=1, prob=0.3),
        "Rotate":         lambda: partial(Rotate(),         mag=1, prob=0.3),
        "Shrink":         lambda: partial(Shrink(),         mag=1, prob=0.3),
        # Process
        "Posterize":      lambda: partial(Posterize(),      mag=1, prob=0.3),
        "Solarize":       lambda: partial(Solarize(),       mag=1, prob=0.3),
        "Invert":         lambda: partial(Invert(),         mag=1, prob=0.2),
        "Equalize":       lambda: partial(Equalize(),       mag=1, prob=0.3),
        "AutoContrast":   lambda: partial(AutoContrast(),   mag=1, prob=0.3),
        "Sharpness":      lambda: partial(Sharpness(),      mag=1, prob=0.4),
        "Color":          lambda: partial(Color(),          mag=1, prob=0.4),
    }

# Groups: only ONE transform per group is applied per image.
_STRAUG_GROUPS: dict[str, list[str]] = {
    "pattern":  ["Grid", "VGrid", "HGrid", "RectGrid", "EllipseGrid"],
    "blur":     ["GaussianBlur", "DefocusBlur", "MotionBlur", "GlassBlur", "ZoomBlur"],
    "noise":    ["GaussianNoise", "ShotNoise", "ImpulseNoise", "SpeckleNoise"],
    "weather":  ["Fog", "Snow", "Frost", "Rain", "Shadow"],
    "camera":   ["Contrast", "Brightness", "JpegCompression", "Pixelate"],
    "warp":     ["Curve", "Distort", "Stretch"],
    "geometry": ["StraugPerspective", "Rotate", "Shrink"],
    "process":  ["Posterize", "Solarize", "Invert", "Equalize",
                 "AutoContrast", "Sharpness", "Color"],
}


# ---------------------------------------------------------------------------
# Albumentations registry — soft parameters
# ---------------------------------------------------------------------------

_ALBUMENTATIONS_MAP: dict[str, Callable[[], A.BasicTransform]] = {
    "ImageCompression": lambda: A.ImageCompression(
        quality_range=(70, 95), compression_type="jpeg", p=0.6
    ),
    "Perspective": lambda: A.Perspective(scale=(0.01, 0.05), p=0.5),
    "MotionBlur":  lambda: A.MotionBlur(blur_limit=(3, 7), p=0.4),
    "Defocus":     lambda: A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=0.3),
    "OpticalDistortion": lambda: A.OpticalDistortion(distort_limit=0.2, p=0.3),
    "PixelDropout": lambda: A.PixelDropout(
        dropout_prob=0.01, per_channel=False, drop_value=0, p=0.3
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_straug_per_image(names: list[str]) -> list:
    """
    From a list of STRAug transform names, randomly pick ONE per group
    and return instantiated transform callables.

    Called per-image so each image gets a different random selection.
    """
    if not _HAVE_STRAUG:
        return []

    # Bucket requested names into their groups
    groups_present: dict[str, list[str]] = {}
    for name in names:
        for group, members in _STRAUG_GROUPS.items():
            if name in members:
                groups_present.setdefault(group, []).append(name)
                break

    # Pick one per group and instantiate
    transforms = []
    for group, members in groups_present.items():
        chosen = random.choice(members)
        if chosen in _STRAUG_MAP:
            try:
                transforms.append(_STRAUG_MAP[chosen]())
            except Exception as e:
                logger.warning(f"STRAug init error for '{chosen}': {e}")

    return transforms


def _build_albumentations_pipeline(names: list[str]) -> A.Compose | None:
    """Build an Albumentations Compose from a list of names."""
    transforms = []
    for name in names:
        if name in _ALBUMENTATIONS_MAP:
            try:
                transforms.append(_ALBUMENTATIONS_MAP[name]())
            except Exception as e:
                logger.warning(f"Albumentations init error for '{name}': {e}")
        else:
            logger.warning(f"Unknown albumentations aug '{name}'")
    return A.Compose(transforms) if transforms else None


def _apply_albumentations(
    img: Image.Image, pipeline: A.Compose
) -> Image.Image:
    """Apply an Albumentations pipeline to a grayscale PIL Image."""
    arr = np.array(img)  # (H, W) uint8
    # Albumentations expects 3-channel input
    arr3 = np.stack([arr, arr, arr], axis=-1)
    result = pipeline(image=arr3)["image"]
    # Convert back to grayscale
    grey = np.mean(result, axis=-1).astype(np.uint8)
    return Image.fromarray(grey, mode="L")


def _apply_straug(img: Image.Image, transforms: list) -> Image.Image:
    """Apply a list of STRAug transforms to a grayscale PIL Image."""
    for t in transforms:
        img = t(img)
    # STRAug may return RGB — normalise back to L
    if img.mode != "L":
        img = img.convert("L")
    return img


# ---------------------------------------------------------------------------
# Level-based presets (fallback when no explicit transform lists are given)
# ---------------------------------------------------------------------------

# For level-based mode we use a curated set of STRAug names per level.
_LEVEL_STRAUG: dict[str, list[str]] = {
    "off":    [],
    "low":    ["Grid", "JpegCompression", "Brightness"],
    "medium": ["Grid", "VGrid", "JpegCompression", "StraugPerspective",
               "MotionBlur", "Brightness"],
    "high":   ["Grid", "VGrid", "HGrid", "RectGrid", "JpegCompression",
               "StraugPerspective", "MotionBlur", "DefocusBlur", "Pixelate"],
    "all":    ["Grid", "VGrid", "HGrid", "RectGrid", "JpegCompression",
               "StraugPerspective", "MotionBlur", "DefocusBlur", "Pixelate",
               "GaussianBlur", "Brightness", "Contrast"],
}

_LEVEL_ALBU: dict[str, list[str]] = {
    "off":    [],
    "low":    [],
    "medium": ["PixelDropout"],
    "high":   ["PixelDropout", "OpticalDistortion"],
    "all":    ["PixelDropout", "OpticalDistortion"],
}


# ---------------------------------------------------------------------------
# AugmentationPipeline — public class
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    Probabilistic OCR augmentation pipeline.

    Applies (in order):
      1. Albumentations pipeline (once per image)
      2. STRAug — one transform per group, chosen randomly per image
         with up to 3 retries if output is near-black

    Construct via:
        AugmentationPipeline(level="medium")         # level preset
        AugmentationPipeline.from_lists(             # explicit named transforms
            level="high",
            straug_augs=["Grid", "MotionBlur"],
            albumentations_augs=["PixelDropout"],
        )
    """

    def __init__(self, level: str = "medium") -> None:
        valid = set(_LEVEL_STRAUG)
        if level not in valid:
            raise ValueError(
                f"Unknown augmentation level {level!r}. Choose from {sorted(valid)}"
            )
        self.level = level
        self._straug_names: list[str] = _LEVEL_STRAUG[level]
        self._albu_pipeline: A.Compose | None = _build_albumentations_pipeline(
            _LEVEL_ALBU[level]
        )

    @classmethod
    def from_lists(
        cls,
        level: str = "medium",
        straug_augs: list[str] | None = None,
        albumentations_augs: list[str] | None = None,
    ) -> "AugmentationPipeline":
        """
        Build from explicit named transform lists (per-phase YAML format).

        Falls back to level preset if both lists are empty.
        """
        straug_augs = straug_augs or []
        albumentations_augs = albumentations_augs or []

        if not straug_augs and not albumentations_augs:
            return cls(level=level)

        obj = object.__new__(cls)
        obj.level = level
        obj._straug_names = list(straug_augs)

        # Validate STRAug names
        all_straug = {n for members in _STRAUG_GROUPS.values() for n in members}
        for name in straug_augs:
            if name not in all_straug:
                raise ValueError(
                    f"Unknown straug_aug {name!r}. Valid: {sorted(all_straug)}"
                )

        obj._albu_pipeline = _build_albumentations_pipeline(albumentations_augs)
        return obj

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the full augmentation pipeline to a PIL Image (mode "L").

        Returns a PIL Image (mode "L") of the same size.
        """
        img = img.convert("L")

        # Step 1: Albumentations (once, no retry needed — params are soft)
        if self._albu_pipeline is not None:
            img_before = img.copy()
            try:
                candidate = _apply_albumentations(img, self._albu_pipeline)
                if not is_image_mostly_black(candidate):
                    img = candidate
                else:
                    logger.debug("Albumentations produced near-black image, skipping")
                    img = img_before
            except Exception as e:
                logger.debug(f"Albumentations error: {e}")

        # Step 2: STRAug — one per group, 3-retry anti-black loop
        if self._straug_names and _HAVE_STRAUG:
            img_before_straug = img.copy()
            applied = False
            for attempt in range(3):
                transforms = _select_straug_per_image(self._straug_names)
                if not transforms:
                    break
                try:
                    candidate = _apply_straug(img_before_straug.copy(), transforms)
                    if not is_image_mostly_black(candidate):
                        img = candidate
                        applied = True
                        break
                    else:
                        if attempt < 2:
                            logger.debug(
                                f"STRAug attempt {attempt + 1}/3 near-black, retrying"
                            )
                        else:
                            logger.debug("STRAug: all 3 attempts near-black, keeping pre-STRAug image")
                except Exception as e:
                    logger.debug(f"STRAug attempt {attempt + 1} error: {e}")

            if not applied:
                img = img_before_straug

        return img

    def __repr__(self) -> str:
        return (
            f"AugmentationPipeline(level={self.level!r}, "
            f"straug={self._straug_names}, "
            f"albu={'yes' if self._albu_pipeline else 'none'})"
        )

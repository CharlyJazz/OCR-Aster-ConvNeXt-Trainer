"""
Image augmentation pipeline for OCR training.

Augmentation is applied in publisher threads before images reach the GPU.
The level string ("off" / "low" / "medium" / "high" / "all") controls which
transforms are active and at what probability.

All transforms operate on a PIL Image (grayscale, mode "L") and return a PIL Image.
The final conversion to a normalised float tensor is handled by the collate step.
"""

from __future__ import annotations

import random
from typing import Callable

from PIL import Image, ImageFilter, ImageOps

# albumentations is imported lazily so the module loads even without GPU env
try:
    import albumentations as A
    import numpy as np
    _HAVE_ALBUMENTATIONS = True
except ImportError:
    _HAVE_ALBUMENTATIONS = False


# ---------------------------------------------------------------------------
# Individual transforms (PIL-based, no extra deps)
# ---------------------------------------------------------------------------

def _random_blur(p: float) -> Callable[[Image.Image], Image.Image]:
    def transform(img: Image.Image) -> Image.Image:
        if random.random() < p:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
    return transform


def _random_motion_blur(p: float) -> Callable[[Image.Image], Image.Image]:
    def transform(img: Image.Image) -> Image.Image:
        if random.random() < p:
            # Approximate motion blur with a directional box filter
            size = random.choice([3, 5])
            img = img.filter(ImageFilter.BoxBlur(size))
        return img
    return transform


def _random_noise(p: float, amount: float = 0.02) -> Callable[[Image.Image], Image.Image]:
    def transform(img: Image.Image) -> Image.Image:
        if random.random() < p:
            import numpy as np
            arr = np.array(img, dtype=np.int16)
            noise = np.random.normal(0, amount * 255, arr.shape).astype(np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L")
        return img
    return transform


def _random_perspective(p: float, distortion: float = 0.1) -> Callable[[Image.Image], Image.Image]:
    def transform(img: Image.Image) -> Image.Image:
        if random.random() < p:
            w, h = img.size
            d = int(distortion * min(w, h))
            coeffs = _find_perspective_coeffs(
                [(0, 0), (w, 0), (w, h), (0, h)],
                [
                    (random.randint(0, d), random.randint(0, d)),
                    (w - random.randint(0, d), random.randint(0, d)),
                    (w - random.randint(0, d), h - random.randint(0, d)),
                    (random.randint(0, d), h - random.randint(0, d)),
                ],
            )
            img = img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        return img
    return transform


def _find_perspective_coeffs(
    src: list[tuple[int, int]], dst: list[tuple[int, int]]
) -> list[float]:
    """Compute 8 coefficients for PIL perspective transform."""
    import numpy as np
    matrix = []
    for p1, p2 in zip(dst, src):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A_mat = np.array(matrix, dtype=float)
    B_vec = np.array([x for pair in src for x in pair], dtype=float).reshape(8)
    res = np.linalg.solve(A_mat, B_vec)
    return list(res.flatten())


def _random_threshold(p: float) -> Callable[[Image.Image], Image.Image]:
    def transform(img: Image.Image) -> Image.Image:
        if random.random() < p:
            threshold = random.randint(100, 180)
            img = img.point(lambda x: 255 if x > threshold else 0)
        return img
    return transform


def _random_jpeg_quality(p: float, min_q: int = 40) -> Callable[[Image.Image], Image.Image]:
    """Simulate JPEG compression artefacts."""
    def transform(img: Image.Image) -> Image.Image:
        if random.random() < p:
            import io
            q = random.randint(min_q, 85)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("L")
        return img
    return transform


# ---------------------------------------------------------------------------
# Albumentations-based transforms (optional, require numpy)
# ---------------------------------------------------------------------------

def _albu_optical_distortion(p: float) -> Callable[[Image.Image], Image.Image] | None:
    if not _HAVE_ALBUMENTATIONS:
        return None

    transform = A.OpticalDistortion(p=1.0)

    def apply(img: Image.Image) -> Image.Image:
        if random.random() < p:
            arr = np.array(img)
            arr = transform(image=arr)["image"]
            img = Image.fromarray(arr)
        return img

    return apply


def _albu_pixel_dropout(p: float) -> Callable[[Image.Image], Image.Image] | None:
    if not _HAVE_ALBUMENTATIONS:
        return None

    transform = A.PixelDropout(dropout_prob=0.02, p=1.0)

    def apply(img: Image.Image) -> Image.Image:
        if random.random() < p:
            arr = np.array(img)
            arr = transform(image=arr)["image"]
            img = Image.fromarray(arr)
        return img

    return apply


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

# Level → (blur_p, motion_p, noise_p, perspective_p, threshold_p, jpeg_p,
#           optical_p, dropout_p)
_LEVEL_PARAMS: dict[str, tuple[float, ...]] = {
    "off":    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "low":    (0.1, 0.0, 0.1, 0.05, 0.1, 0.1, 0.0, 0.0),
    "medium": (0.3, 0.2, 0.3, 0.2,  0.2, 0.2, 0.1, 0.1),
    "high":   (0.5, 0.4, 0.5, 0.4,  0.4, 0.3, 0.3, 0.2),
    "all":    (0.6, 0.5, 0.6, 0.5,  0.5, 0.4, 0.4, 0.3),
}


class AugmentationPipeline:
    """
    Composes a list of probabilistic PIL image transforms.

    Args:
        level: one of "off", "low", "medium", "high", "all"
    """

    def __init__(self, level: str = "medium") -> None:
        if level not in _LEVEL_PARAMS:
            raise ValueError(f"Unknown augmentation level {level!r}. "
                             f"Choose from {list(_LEVEL_PARAMS)}")
        self.level = level
        (blur_p, motion_p, noise_p, persp_p,
         thresh_p, jpeg_p, optical_p, drop_p) = _LEVEL_PARAMS[level]

        self._transforms: list[Callable[[Image.Image], Image.Image]] = []

        if blur_p > 0:
            self._transforms.append(_random_blur(blur_p))
        if motion_p > 0:
            self._transforms.append(_random_motion_blur(motion_p))
        if noise_p > 0:
            self._transforms.append(_random_noise(noise_p))
        if persp_p > 0:
            self._transforms.append(_random_perspective(persp_p))
        if thresh_p > 0:
            self._transforms.append(_random_threshold(thresh_p))
        if jpeg_p > 0:
            self._transforms.append(_random_jpeg_quality(jpeg_p))

        # Optional albumentations transforms
        if optical_p > 0:
            t = _albu_optical_distortion(optical_p)
            if t is not None:
                self._transforms.append(t)
        if drop_p > 0:
            t = _albu_pixel_dropout(drop_p)
            if t is not None:
                self._transforms.append(t)

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply pipeline to a PIL Image (mode "L"). Returns PIL Image."""
        for t in self._transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        return f"AugmentationPipeline(level={self.level!r}, transforms={len(self._transforms)})"

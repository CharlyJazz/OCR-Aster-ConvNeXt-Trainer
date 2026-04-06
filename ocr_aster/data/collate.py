"""
AlignCollate — batch collation for OCR images.

Resizes every image to a fixed (imgH, imgW), converts to a normalised
float tensor in [0, 1], and stacks into a batch tensor (B, 1, H, W).

Optionally applies contrast normalisation (histogram equalization) before
conversion, which can help with very dark or washed-out scans.
"""

from __future__ import annotations

import torch
from PIL import Image, ImageOps
from torch import Tensor


class AlignCollate:
    """
    Callable that turns a list of (PIL Image, label_str) pairs into a
    padded batch ready for the model.

    Args:
        imgH:            target image height (default 120)
        imgW:            target image width  (default 280)
        keep_ratio:      if True, pad to keep aspect ratio (adds grey border);
                         if False (default), stretch to fill (imgH, imgW)
        adjust_contrast: apply CLAHE-style auto-contrast before normalisation
    """

    def __init__(
        self,
        imgH: int = 120,
        imgW: int = 280,
        keep_ratio: bool = False,
        adjust_contrast: bool = False,
    ) -> None:
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.adjust_contrast = adjust_contrast

    def __call__(
        self, batch: list[tuple[Image.Image, str]]
    ) -> tuple[Tensor, list[str]]:
        """
        Args:
            batch: list of (PIL Image, label_string) pairs

        Returns:
            images:  FloatTensor (B, 1, imgH, imgW), values in [0, 1]
            labels:  list[str], same order as input
        """
        images, labels = zip(*batch)
        tensors = [self._process(img) for img in images]
        return torch.stack(tensors, dim=0), list(labels)

    def _process(self, img: Image.Image) -> Tensor:
        # Ensure grayscale
        img = img.convert("L")

        if self.adjust_contrast:
            img = ImageOps.autocontrast(img)

        if self.keep_ratio:
            img = self._resize_keep_ratio(img)
        else:
            img = img.resize((self.imgW, self.imgH), Image.BICUBIC)

        # PIL "L" → float32 tensor, shape (1, H, W), range [0, 1]
        import torchvision.transforms.functional as TF
        return TF.to_tensor(img)  # already (1, H, W) and normalised

    def _resize_keep_ratio(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ratio = min(self.imgW / w, self.imgH / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Pad to target size (grey background = 127)
        pad_w = self.imgW - new_w
        pad_h = self.imgH - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        result = Image.new("L", (self.imgW, self.imgH), 127)
        result.paste(img, (pad_left, pad_top))
        return result

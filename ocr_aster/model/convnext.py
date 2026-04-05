"""
ConvNeXt backbone in OCR mode.

Key property: asymmetric downsampling — height is aggressively compressed
while width is preserved, yielding T=35 temporal positions from a 280px-wide
input. This gives the BiLSTM enough horizontal resolution to separate
individual characters.

No TIMM dependency. Custom LayerNorm and DropPath throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """
    LayerNorm with optional channel-first layout (C, H, W).

    PyTorch's built-in LayerNorm expects channel-last (H, W, C).
    This wrapper handles both so ConvNeXt blocks can use it directly
    on feature maps without permuting every time.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) — channel-first
        # Permute → normalize → permute back
        x = x.permute(0, 2, 3, 1)          # (B, H, W, C)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)        # (B, C, H, W)


class DropPath(nn.Module):
    """
    Stochastic depth / drop path regularization (Huang et al., 2016).
    Drops entire residual branches with probability `drop_prob` during training.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * noise.div_(keep)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# ConvNeXt Block
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """
    Standard ConvNeXt block:
        depthwise 7×7 conv → LayerNorm → 1×1 (expand 4×) → GELU → 1×1 (project back)

    The large depthwise kernel (7×7) captures long-range spatial context
    efficiently — similar to self-attention but at a fraction of the cost.
    """

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + self.drop_path(x)


# ---------------------------------------------------------------------------
# Downsampling layer
# ---------------------------------------------------------------------------

class DownsampleLayer(nn.Module):
    """
    Spatial downsampling between ConvNeXt stages.

    OCR mode: downsamples HEIGHT only (stride on H, stride=1 on W).
    This preserves horizontal resolution — characters are laid out
    horizontally and we must not lose that information.
    """

    def __init__(self, in_channels: int, out_channels: int, height_only: bool = True) -> None:
        super().__init__()
        if height_only:
            # stride (2, 1): halve H, keep W
            stride = (2, 1)
            kernel = (2, 1)
        else:
            # standard 2× downsampling on both axes (stem only)
            stride = (4, 4)
            kernel = (4, 4)
        self.norm = LayerNorm(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.norm(x))


# ---------------------------------------------------------------------------
# ConvNeXt OCR backbone
# ---------------------------------------------------------------------------

class ConvNeXtOCR(nn.Module):
    """
    ConvNeXt backbone tuned for OCR sequence extraction.

    Architecture
    ============
    Stem    : 4×4 conv stride-4 (both axes) → C0 channels
    Stage 1 : depth_1 ConvNeXt blocks
    Down 1  : height-only stride-2 → 2*C0 channels
    Stage 2 : depth_2 blocks
    Down 2  : height-only stride-2 → 4*C0 channels
    Stage 3 : depth_3 blocks
    AdaptiveAvgPool: collapse H → 1, keep W (= T temporal positions)

    With default config (C0=128, imgH=120, imgW=280):
        After stem    : (B, 128, 30, 70)
        After down-1  : (B, 256, 15, 70)
        After down-2  : (B, 512,  7, 70)
        After pool    : (B, 512,  1, 35)   ← squeeze → (T=35, B, 512)

    The output is a sequence of T=35 visual feature vectors, one per
    horizontal position, ready for the BiLSTM encoder.

    Args:
        input_channel: number of input channels (1 for grayscale)
        output_channel: channels in the final feature sequence (= BiLSTM input size)
        depths: blocks per stage, default (3, 3, 9) following ConvNeXt-Tiny proportions
        drop_path_rate: stochastic depth rate (linearly scaled per block)
    """

    def __init__(
        self,
        input_channel: int = 1,
        output_channel: int = 512,
        depths: tuple[int, int, int] = (3, 3, 9),
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # Channel progression: C0 → 2*C0 → 4*C0 = output_channel
        # So C0 = output_channel // 4
        c0 = output_channel // 4          # 128  with output_channel=512
        c1 = output_channel // 2          # 256
        c2 = output_channel               # 512

        # Drop path rates — linearly increase across all blocks
        total_blocks = sum(depths)
        dp_rates = [r.item() for r in torch.linspace(0, drop_path_rate, total_blocks)]
        idx = 0

        # ── Stem ──────────────────────────────────────────────────────────
        # 4×4 stride-4 on both axes: (B, 1, 120, 280) → (B, c0, 30, 70)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channel, c0, kernel_size=4, stride=4),
            LayerNorm(c0),
        )

        # ── Stage 1 ───────────────────────────────────────────────────────
        self.stage1 = nn.Sequential(
            *[ConvNeXtBlock(c0, drop_path=dp_rates[idx + i]) for i in range(depths[0])]
        )
        idx += depths[0]

        # ── Downsample 1 (height only) ────────────────────────────────────
        # (B, c0, 30, 70) → (B, c1, 15, 70)
        self.down1 = DownsampleLayer(c0, c1, height_only=True)

        # ── Stage 2 ───────────────────────────────────────────────────────
        self.stage2 = nn.Sequential(
            *[ConvNeXtBlock(c1, drop_path=dp_rates[idx + i]) for i in range(depths[1])]
        )
        idx += depths[1]

        # ── Downsample 2 (height only) ────────────────────────────────────
        # (B, c1, 15, 70) → (B, c2, 7, 70)
        self.down2 = DownsampleLayer(c1, c2, height_only=True)

        # ── Stage 3 ───────────────────────────────────────────────────────
        self.stage3 = nn.Sequential(
            *[ConvNeXtBlock(c2, drop_path=dp_rates[idx + i]) for i in range(depths[2])]
        )

        # ── Adaptive pool → collapse H, halve W ───────────────────────────
        # (B, c2, 7, 70) → (B, c2, 1, 35)
        self.pool = nn.AdaptiveAvgPool2d((1, 35))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, input_channel, H, W) — typically (B, 1, 120, 280)

        Returns:
            sequence: (T, B, output_channel) — e.g. (35, B, 512)
                      ready to feed into BiLSTM encoder
        """
        x = self.stem(x)       # (B, c0, 30, 70)
        x = self.stage1(x)     # (B, c0, 30, 70)
        x = self.down1(x)      # (B, c1, 15, 70)
        x = self.stage2(x)     # (B, c1, 15, 70)
        x = self.down2(x)      # (B, c2,  7, 70)
        x = self.stage3(x)     # (B, c2,  7, 70)
        x = self.pool(x)       # (B, c2,  1, 35)
        x = x.squeeze(2)       # (B, c2, 35)
        x = x.permute(2, 0, 1) # (35, B, c2) = (T, B, output_channel)
        return x

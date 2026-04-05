"""
Tests for ConvNeXtOCR backbone.

We verify shapes, temporal resolution, and that no TIMM import
exists anywhere in the module.
"""

import importlib
import inspect
import sys

import pytest
import torch

from ocr_aster.model.convnext import ConvNeXtOCR, ConvNeXtBlock, LayerNorm, DropPath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model() -> ConvNeXtOCR:
    return ConvNeXtOCR(input_channel=1, output_channel=512)


@pytest.fixture
def batch() -> torch.Tensor:
    """Standard OCR input: batch=4, grayscale, 120×280."""
    return torch.randn(4, 1, 120, 280)


# ---------------------------------------------------------------------------
# No external vision library dependency
# ---------------------------------------------------------------------------

def test_no_timm_dependency():
    """The module must be self-contained — no TIMM or torchvision models."""
    import ocr_aster.model.convnext as mod
    source = inspect.getsource(mod)
    assert "timm" not in source, "timm import found — remove it"
    assert "torchvision.models" not in source, "torchvision.models found — use custom impl"


# ---------------------------------------------------------------------------
# Output shape — the critical property
# ---------------------------------------------------------------------------

def test_output_shape(model, batch):
    """
    ConvNeXtOCR must output (T=35, B, 512).
    T=35 temporal positions is the core OCR-mode property.
    """
    out = model(batch)
    assert out.shape == (35, 4, 512), (
        f"Expected (35, 4, 512), got {out.shape}. "
        "Check asymmetric downsampling config."
    )


def test_output_temporal_dim_is_35(model, batch):
    """Temporal dimension must be exactly 35 — not 8, not 70, not something else."""
    out = model(batch)
    T = out.shape[0]
    assert T == 35, f"Temporal positions T={T}, expected 35"


def test_output_channel_matches_config():
    """output_channel controls the feature dimension of the sequence."""
    for out_ch in [256, 512]:
        m = ConvNeXtOCR(input_channel=1, output_channel=out_ch)
        x = torch.randn(2, 1, 120, 280)
        out = m(x)
        assert out.shape[2] == out_ch, f"Expected channel={out_ch}, got {out.shape[2]}"


def test_batch_size_independence(model):
    """Output T and C must be the same regardless of batch size."""
    for B in [1, 2, 8]:
        x = torch.randn(B, 1, 120, 280)
        out = model(x)
        assert out.shape == (35, B, 512), f"Failed for B={B}: {out.shape}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow(model, batch):
    """All parameters must receive gradients after a backward pass."""
    out = model(batch)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def test_layer_norm_channel_first():
    """LayerNorm must handle (B, C, H, W) input correctly."""
    ln = LayerNorm(64)
    x = torch.randn(2, 64, 8, 8)
    out = ln(x)
    assert out.shape == x.shape


def test_drop_path_inference():
    """DropPath must be identity at eval time."""
    dp = DropPath(drop_prob=0.5)
    dp.eval()
    x = torch.randn(4, 32)
    assert torch.allclose(dp(x), x)


def test_drop_path_training():
    """DropPath must drop paths (some outputs zero) during training."""
    torch.manual_seed(42)
    dp = DropPath(drop_prob=0.9)
    dp.train()
    x = torch.ones(100, 1)
    out = dp(x)
    # With 90% drop rate, most outputs should be 0
    zeros = (out == 0).sum().item()
    assert zeros > 50, f"Expected most outputs to be dropped, got {zeros}/100 zeros"


def test_convnext_block_residual():
    """ConvNeXtBlock must be a residual: output shape == input shape."""
    block = ConvNeXtBlock(dim=64)
    x = torch.randn(2, 64, 8, 8)
    out = block(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Eval mode (no randomness)
# ---------------------------------------------------------------------------

def test_eval_mode_deterministic(model, batch):
    """Same input must produce same output in eval mode."""
    model.eval()
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    assert torch.allclose(out1, out2)

"""
Tests for BiLSTMEncoder.
"""

import pytest
import torch

from ocr_aster.model.encoder import BiLSTMEncoder, BidirectionalLSTM


T, B, C = 35, 4, 512  # standard dims


@pytest.fixture
def encoder():
    return BiLSTMEncoder(input_size=C, hidden_size=C)


@pytest.fixture
def sequence():
    return torch.randn(T, B, C)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

def test_encoded_sequence_shape(encoder, sequence):
    encoded, bridge = encoder(sequence)
    assert encoded.shape == (T, B, C), f"Expected ({T}, {B}, {C}), got {encoded.shape}"


def test_bridge_shape(encoder, sequence):
    """Bridge must be (B, hidden_size) — used to init decoder GRU."""
    encoded, bridge = encoder(sequence)
    assert bridge.shape == (B, C), f"Expected ({B}, {C}), got {bridge.shape}"


def test_batch_size_independence(encoder):
    for b in [1, 2, 8]:
        x = torch.randn(T, b, C)
        encoded, bridge = encoder(x)
        assert encoded.shape == (T, b, C)
        assert bridge.shape == (b, C)


# ---------------------------------------------------------------------------
# Bridge is not zeros
# ---------------------------------------------------------------------------

def test_bridge_is_not_zeros(encoder, sequence):
    """Bridge must carry real encoder information, not be a zero tensor."""
    _, bridge = encoder(sequence)
    assert not torch.allclose(bridge, torch.zeros_like(bridge)), \
        "Bridge is all zeros — encoder not learning anything"


def test_bridge_differs_per_sample(encoder):
    """Different inputs must produce different bridge vectors."""
    x1 = torch.randn(T, B, C)
    x2 = torch.randn(T, B, C)
    _, bridge1 = encoder(x1)
    _, bridge2 = encoder(x2)
    assert not torch.allclose(bridge1, bridge2), \
        "Bridge is identical for different inputs"


# ---------------------------------------------------------------------------
# LayerNorm gate (v2 property)
# ---------------------------------------------------------------------------

def test_layernorm_exists_before_lstm():
    """v2 requires LayerNorm before the BiLSTM — verify it's there."""
    import torch.nn as nn
    enc = BiLSTMEncoder()
    children = list(enc.children())
    # First child must be a LayerNorm
    assert isinstance(children[0], nn.LayerNorm), \
        "First module must be LayerNorm (v2 property). Found: " + str(type(children[0]))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow(encoder, sequence):
    sequence.requires_grad_(True)
    encoded, bridge = encoder(sequence)
    loss = encoded.sum() + bridge.sum()
    loss.backward()
    assert sequence.grad is not None
    assert not torch.isnan(sequence.grad).any()


def test_all_params_get_gradients(encoder, sequence):
    encoded, bridge = encoder(sequence)
    (encoded.sum() + bridge.sum()).backward()
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# BidirectionalLSTM unit
# ---------------------------------------------------------------------------

def test_bilstm_output_shape():
    lstm = BidirectionalLSTM(input_size=64, hidden_size=128, output_size=64)
    x = torch.randn(10, 3, 64)
    out, (h_n, _) = lstm(x)
    assert out.shape == (10, 3, 64)
    assert h_n.shape == (2, 3, 128)  # 2 directions


def test_bilstm_projects_to_output_size():
    """Linear projection must map 2*hidden → output_size correctly."""
    lstm = BidirectionalLSTM(input_size=32, hidden_size=64, output_size=16)
    x = torch.randn(5, 2, 32)
    out, _ = lstm(x)
    assert out.shape == (5, 2, 16)

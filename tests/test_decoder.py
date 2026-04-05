"""
Tests for AdditiveAttention and ASTERDecoder.
"""

import pytest
import torch

from ocr_aster.model.attention import AdditiveAttention
from ocr_aster.model.decoder import ASTERDecoder


T, B, H = 35, 4, 512
NUM_CLASSES = 70
MAX_LEN = 28


@pytest.fixture
def encoded():
    return torch.randn(T, B, H)


@pytest.fixture
def bridge():
    return torch.randn(B, H)


@pytest.fixture
def decoder():
    return ASTERDecoder(num_classes=NUM_CLASSES, hidden_size=H, embed_dim=256)


@pytest.fixture
def targets():
    # GO token (0) at position 0, then random chars
    t = torch.randint(2, NUM_CLASSES, (B, MAX_LEN + 1))
    t[:, 0] = 0  # GO_TOKEN
    return t


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAdditiveAttention:
    def test_context_shape(self):
        attn = AdditiveAttention(H)
        hidden = torch.randn(B, H)
        enc = torch.randn(T, B, H)
        enc_proj = attn.project_encoder(enc)
        context, alpha = attn(hidden, enc, enc_proj)
        assert context.shape == (B, H)

    def test_alpha_shape(self):
        attn = AdditiveAttention(H)
        hidden = torch.randn(B, H)
        enc = torch.randn(T, B, H)
        enc_proj = attn.project_encoder(enc)
        _, alpha = attn(hidden, enc, enc_proj)
        assert alpha.shape == (B, T)

    def test_alpha_sums_to_one(self):
        """Attention weights must sum to 1 over positions."""
        attn = AdditiveAttention(H)
        hidden = torch.randn(B, H)
        enc = torch.randn(T, B, H)
        enc_proj = attn.project_encoder(enc)
        _, alpha = attn(hidden, enc, enc_proj)
        sums = alpha.sum(dim=1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
            f"Alpha sums: {sums}"

    def test_encoder_projection_cached_equals_computed(self):
        """project_encoder must equal W_e applied to each position."""
        attn = AdditiveAttention(H)
        enc = torch.randn(T, B, H)
        cached = attn.project_encoder(enc)
        # Manually compute: W_e applied to each (B, H) slice
        manual = torch.stack([attn.W_e(enc[t]) for t in range(T)], dim=0)
        assert torch.allclose(cached, manual, atol=1e-5)


# ---------------------------------------------------------------------------
# Decoder: bridge init
# ---------------------------------------------------------------------------

class TestDecoderBridge:
    def test_init_hidden_not_zeros(self, decoder, bridge):
        """Bridge must produce non-zero initial hidden state."""
        h = decoder.init_hidden(bridge)
        assert not torch.allclose(h, torch.zeros_like(h)), \
            "init_hidden returned zeros — bridge not working"

    def test_init_hidden_shape(self, decoder, bridge):
        h = decoder.init_hidden(bridge)
        assert h.shape == (B, H)

    def test_bridge_differs_for_different_encoders(self, decoder):
        """Two different encoder bridges must produce different hidden inits."""
        b1 = torch.randn(B, H)
        b2 = torch.randn(B, H)
        h1 = decoder.init_hidden(b1)
        h2 = decoder.init_hidden(b2)
        assert not torch.allclose(h1, h2)


# ---------------------------------------------------------------------------
# Decoder: output shapes
# ---------------------------------------------------------------------------

class TestDecoderShapes:
    def test_training_mode_output_shape(self, decoder, encoded, bridge, targets):
        logits = decoder(encoded, bridge, targets=targets, max_length=MAX_LEN,
                         teacher_forcing_ratio=1.0)
        assert logits.shape == (B, MAX_LEN, NUM_CLASSES), \
            f"Expected ({B}, {MAX_LEN}, {NUM_CLASSES}), got {logits.shape}"

    def test_inference_mode_output_shape(self, decoder, encoded, bridge):
        with torch.no_grad():
            logits = decoder(encoded, bridge, targets=None, max_length=MAX_LEN)
        assert logits.shape == (B, MAX_LEN, NUM_CLASSES)

    def test_batch_size_independence(self, decoder, encoded, bridge):
        for b in [1, 2, 8]:
            enc = torch.randn(T, b, H)
            brg = torch.randn(b, H)
            logits = decoder(enc, brg, targets=None, max_length=MAX_LEN)
            assert logits.shape == (b, MAX_LEN, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Decoder: teacher forcing
# ---------------------------------------------------------------------------

class TestTeacherForcing:
    def test_teacher_forcing_1_uses_targets(self, decoder, encoded, bridge, targets):
        """With τ=1.0, output must differ from τ=0.0 (different inputs at each step)."""
        torch.manual_seed(0)
        logits_tf = decoder(encoded, bridge, targets=targets,
                            max_length=MAX_LEN, teacher_forcing_ratio=1.0)
        torch.manual_seed(0)
        logits_free = decoder(encoded, bridge, targets=targets,
                              max_length=MAX_LEN, teacher_forcing_ratio=0.0)
        # They won't be identical because free-running feeds back different chars
        # (unless the model happens to predict exactly the targets, which is unlikely at init)
        assert not torch.allclose(logits_tf, logits_free), \
            "τ=1.0 and τ=0.0 produced identical outputs — teacher forcing not applied"

    def test_teacher_forcing_deterministic_with_seed(self, decoder, encoded, bridge, targets):
        """Same seed + τ=1.0 must produce same output (no randomness from TF itself)."""
        torch.manual_seed(42)
        logits1 = decoder(encoded, bridge, targets=targets,
                          max_length=MAX_LEN, teacher_forcing_ratio=1.0)
        torch.manual_seed(42)
        logits2 = decoder(encoded, bridge, targets=targets,
                          max_length=MAX_LEN, teacher_forcing_ratio=1.0)
        assert torch.allclose(logits1, logits2)


# ---------------------------------------------------------------------------
# Decoder: gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow_through_decoder(decoder, encoded, bridge, targets):
    encoded.requires_grad_(True)
    bridge.requires_grad_(True)
    logits = decoder(encoded, bridge, targets=targets, max_length=MAX_LEN)
    loss = logits.sum()
    loss.backward()
    assert encoded.grad is not None, "No gradient for encoded sequence"
    assert bridge.grad is not None, "No gradient for encoder bridge"
    assert not torch.isnan(encoded.grad).any()

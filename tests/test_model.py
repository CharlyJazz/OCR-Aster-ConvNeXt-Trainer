"""
Integration tests for AsterConvNeXt — full end-to-end forward pass.
"""

import pytest
import torch

from ocr_aster.model.model import AsterConvNeXt


NUM_CLASSES = 70
MAX_LEN = 28
B = 4


@pytest.fixture
def model():
    return AsterConvNeXt(num_classes=NUM_CLASSES)


@pytest.fixture
def images():
    return torch.randn(B, 1, 120, 280)


@pytest.fixture
def targets():
    t = torch.randint(2, NUM_CLASSES, (B, MAX_LEN + 1))
    t[:, 0] = 0  # GO token
    return t


# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------

def test_training_forward_shape(model, images, targets):
    logits = model(images, targets=targets, max_length=MAX_LEN)
    assert logits.shape == (B, MAX_LEN, NUM_CLASSES)


def test_training_loss_computes(model, images, targets):
    """Cross-entropy loss must compute without error."""
    logits = model(images, targets=targets, max_length=MAX_LEN)
    # logits: (B, L, C) → reshape for CE
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, NUM_CLASSES),
        targets[:, 1:].reshape(-1),  # skip GO token
    )
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_backward_pass(model, images, targets):
    """Gradients must flow all the way back to the input."""
    images.requires_grad_(True)
    logits = model(images, targets=targets, max_length=MAX_LEN)
    loss = logits.sum()
    loss.backward()
    assert images.grad is not None
    assert not torch.isnan(images.grad).any()


def test_all_parameters_get_gradients(model, images, targets):
    logits = model(images, targets=targets, max_length=MAX_LEN)
    logits.sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters without gradient: {no_grad}"


# ---------------------------------------------------------------------------
# Inference mode
# ---------------------------------------------------------------------------

def test_inference_forward_shape(model, images):
    with torch.no_grad():
        logits = model(images, targets=None, max_length=MAX_LEN)
    assert logits.shape == (B, MAX_LEN, NUM_CLASSES)


def test_generate_shape(model, images):
    preds = model.generate(images, max_length=MAX_LEN)
    assert preds.shape == (B, MAX_LEN)


def test_generate_returns_valid_class_indices(model, images):
    preds = model.generate(images, max_length=MAX_LEN)
    assert preds.min() >= 0
    assert preds.max() < NUM_CLASSES


def test_generate_deterministic(model, images):
    """Same input must give same output in eval mode."""
    model.eval()
    with torch.no_grad():
        p1 = model.generate(images)
        p2 = model.generate(images)
    assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# Batch size independence
# ---------------------------------------------------------------------------

def test_batch_size_independence(model):
    for b in [1, 2, 8]:
        imgs = torch.randn(b, 1, 120, 280)
        with torch.no_grad():
            logits = model(imgs, targets=None, max_length=MAX_LEN)
        assert logits.shape == (b, MAX_LEN, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Sanity: model has reasonable parameter count
# ---------------------------------------------------------------------------

def test_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    # Should be in the range of 20M–100M for this config
    assert 10_000_000 < total < 200_000_000, \
        f"Unusual parameter count: {total:,}. Check architecture."
    print(f"\nTotal parameters: {total:,}")

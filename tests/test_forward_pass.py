"""
Tests for forward_pass() — AMP forward + attention CE loss.
"""

import pytest
import torch

from ocr_aster.model.model import AsterConvNeXt
from ocr_aster.train.forward_pass import forward_pass
from ocr_aster.train.utils import AttnLabelConverter


CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
MAX_LEN = 25
B = 4
DEVICE = torch.device("cpu")


@pytest.fixture
def converter():
    return AttnLabelConverter(CHARS)


@pytest.fixture
def model(converter):
    return AsterConvNeXt(num_classes=converter.num_class)


@pytest.fixture
def batch(converter):
    images = torch.randn(B, 1, 32, 128)
    texts = ["hello", "world", "abc", "test"]
    pred, loss_t, lengths = converter.encode(texts, MAX_LEN)
    return images, pred, loss_t, lengths


def test_loss_is_scalar(model, batch, converter):
    images, pred, loss_t, lengths = batch
    loss, _ = forward_pass(model, images, pred, loss_t, lengths,
                           converter, 1.0, DEVICE)
    assert loss.shape == ()
    assert loss.item() > 0


def test_loss_is_finite(model, batch, converter):
    images, pred, loss_t, lengths = batch
    loss, _ = forward_pass(model, images, pred, loss_t, lengths,
                           converter, 1.0, DEVICE)
    assert torch.isfinite(loss)


def test_preds_shape(model, batch, converter):
    images, pred, loss_t, lengths = batch
    _, preds = forward_pass(model, images, pred, loss_t, lengths,
                            converter, 1.0, DEVICE)
    assert preds.shape == (B, MAX_LEN)


def test_preds_valid_indices(model, batch, converter):
    images, pred, loss_t, lengths = batch
    _, preds = forward_pass(model, images, pred, loss_t, lengths,
                            converter, 1.0, DEVICE)
    assert preds.min() >= 0
    assert preds.max() < converter.num_class


def test_backward_computes(model, batch, converter):
    images, pred, loss_t, lengths = batch
    loss, _ = forward_pass(model, images, pred, loss_t, lengths,
                           converter, 1.0, DEVICE)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_loss_changes_with_tf_ratio(model, batch, converter):
    """Different teacher forcing ratios should produce different losses."""
    images, pred, loss_t, lengths = batch
    torch.manual_seed(0)
    loss1, _ = forward_pass(model, images, pred, loss_t, lengths,
                            converter, 1.0, DEVICE)
    torch.manual_seed(0)
    loss2, _ = forward_pass(model, images, pred, loss_t, lengths,
                            converter, 0.0, DEVICE)
    # With TF=0.0 the model feeds its own (random) predictions — loss will differ
    assert not torch.isclose(loss1, loss2), "Loss identical for TF=1.0 and TF=0.0"


def test_label_smoothing_changes_loss(model, batch, converter):
    """Label smoothing > 0 should produce a different loss than 0."""
    images, pred, loss_t, lengths = batch
    torch.manual_seed(0)
    loss_no_smooth, _ = forward_pass(model, images, pred, loss_t, lengths,
                                     converter, 1.0, DEVICE, label_smoothing=0.0)
    torch.manual_seed(0)
    loss_smooth, _ = forward_pass(model, images, pred, loss_t, lengths,
                                  converter, 1.0, DEVICE, label_smoothing=0.1)
    assert not torch.isclose(loss_no_smooth, loss_smooth), \
        "Loss unchanged with label_smoothing=0.1"


def test_label_smoothing_zero_matches_default(model, batch, converter):
    """label_smoothing=0.0 must be identical to the default (no arg)."""
    images, pred, loss_t, lengths = batch
    torch.manual_seed(0)
    loss_default, _ = forward_pass(model, images, pred, loss_t, lengths,
                                   converter, 1.0, DEVICE)
    torch.manual_seed(0)
    loss_explicit, _ = forward_pass(model, images, pred, loss_t, lengths,
                                    converter, 1.0, DEVICE, label_smoothing=0.0)
    assert torch.isclose(loss_default, loss_explicit)

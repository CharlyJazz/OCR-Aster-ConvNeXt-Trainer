"""
Tests for AttnLabelConverter and Averager.
"""

import pytest
import torch

from ocr_aster.train.utils import AttnLabelConverter, Averager


CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
MAX_LEN = 25


@pytest.fixture
def conv():
    return AttnLabelConverter(CHARS)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TestVocab:
    def test_num_class(self, conv):
        # len(CHARS) + GO + EOS
        assert conv.num_class == len(CHARS) + 2

    def test_go_at_index_0(self, conv):
        assert conv.char2idx["[GO]"] == 0

    def test_eos_at_index_1(self, conv):
        assert conv.char2idx["[EOS]"] == 1

    def test_first_char_at_index_2(self, conv):
        assert conv.char2idx["0"] == 2

    def test_deduplication(self):
        c = AttnLabelConverter("aaabbb")
        assert c.num_class == 4  # GO + EOS + a + b

    def test_vocab_length(self, conv):
        assert len(conv.vocab) == conv.num_class


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncode:
    def test_output_shapes(self, conv):
        texts = ["hello", "world", "ai"]
        pred, loss, lengths = conv.encode(texts, MAX_LEN)
        assert pred.shape == (3, MAX_LEN + 1)
        assert loss.shape == (3, MAX_LEN + 1)
        assert lengths.shape == (3,)

    def test_pred_starts_with_go(self, conv):
        pred, _, _ = conv.encode(["abc"], MAX_LEN)
        assert pred[0, 0].item() == conv.char2idx["[GO]"]

    def test_loss_ends_with_eos(self, conv):
        _, loss, _ = conv.encode(["abc"], MAX_LEN)
        eos = conv.char2idx["[EOS]"]
        assert loss[0, 3].item() == eos  # position after 'abc'

    def test_lengths_correct(self, conv):
        texts = ["ab", "hello", "x"]
        _, _, lengths = conv.encode(texts, MAX_LEN)
        assert lengths.tolist() == [2, 5, 1]

    def test_truncation_at_max_length(self, conv):
        long_text = "a" * 100
        pred, loss, lengths = conv.encode([long_text], MAX_LEN)
        assert lengths[0].item() == MAX_LEN

    def test_empty_string(self, conv):
        pred, loss, lengths = conv.encode([""], MAX_LEN)
        assert lengths[0].item() == 0
        assert pred[0, 0].item() == conv.char2idx["[GO]"]
        assert loss[0, 0].item() == conv.char2idx["[EOS]"]

    def test_unknown_char_maps_to_eos(self, conv):
        # '?' is not in CHARS
        pred, _, _ = conv.encode(["?"], MAX_LEN)
        assert pred[0, 1].item() == conv.char2idx["[EOS]"]


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class TestDecode:
    def test_round_trip(self, conv):
        texts = ["hello", "world"]
        pred, _, _ = conv.encode(texts, MAX_LEN)
        # pred[:, 1:] skips the GO token — these are the char indices
        decoded = conv.decode(pred[:, 1:])
        assert decoded[0] == "hello"
        assert decoded[1] == "world"

    def test_stops_at_eos(self, conv):
        eos = conv.char2idx["[EOS]"]
        a_idx = conv.char2idx["a"]
        b_idx = conv.char2idx["b"]
        indices = torch.tensor([[a_idx, b_idx, eos, a_idx, a_idx]])
        result = conv.decode(indices)
        assert result[0] == "ab"

    def test_empty_output_when_eos_first(self, conv):
        eos = conv.char2idx["[EOS]"]
        indices = torch.tensor([[eos, conv.char2idx["a"]]])
        result = conv.decode(indices)
        assert result[0] == ""

    def test_batch_decode(self, conv):
        texts = ["cat", "dog", "fish"]
        pred, _, _ = conv.encode(texts, MAX_LEN)
        decoded = conv.decode(pred[:, 1:])
        assert decoded == texts


# ---------------------------------------------------------------------------
# Averager
# ---------------------------------------------------------------------------

class TestAverager:
    def test_initial_val_is_zero(self):
        avg = Averager()
        assert avg.val == 0.0

    def test_add_scalars(self):
        avg = Averager()
        avg.add(1.0)
        avg.add(3.0)
        assert avg.val == pytest.approx(2.0)

    def test_add_tensor(self):
        avg = Averager()
        avg.add(torch.tensor(4.0))
        assert avg.val == pytest.approx(4.0)

    def test_add_with_count(self):
        avg = Averager()
        avg.add(2.0, count=3)   # sum += 6, count += 3
        avg.add(0.0, count=3)   # sum += 0, count += 3
        assert avg.val == pytest.approx(1.0)

    def test_reset(self):
        avg = Averager()
        avg.add(99.0)
        avg.reset()
        assert avg.val == 0.0

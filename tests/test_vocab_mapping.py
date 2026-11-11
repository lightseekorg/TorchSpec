"""Test generate_vocab_mapping and _count_token_frequencies from torchspec.data.preprocessing.

Verifies that:
  - Token counting via packed_loss_mask matches a reference implementation.
  - Only tokens where loss_mask == 1 are counted.
  - generate_vocab_mapping produces correct d2t/t2d shapes and values.
"""

from collections import Counter

import pytest
import torch

from torchspec.data.dataset import _load_pretokenized_dataset
from torchspec.data.preprocessing import (
    _count_token_frequencies,
    generate_vocab_mapping,
    process_token_dict_to_mappings,
)
from torchspec.data.utils import pack_loss_mask, serialize_packed_loss_mask


def _make_prompts(input_ids_list, loss_mask_list):
    """Build prompts in the same format as load_conversation_dataset."""
    prompts = []
    for ids, mask in zip(input_ids_list, loss_mask_list):
        packed = pack_loss_mask(torch.tensor(mask))
        prompts.append(
            {
                "input_ids": torch.tensor(ids),
                "packed_loss_mask": serialize_packed_loss_mask(packed),
            }
        )
    return prompts


def test_generate_vocab_mapping_accepts_a_single_token_pretokenized_row():
    prompts = _load_pretokenized_dataset(
        [
            {"input_ids": [5, 6, 7], "loss_mask": [0, 1, 1]},
            {"input_ids": [3], "loss_mask": [1]},
        ],
        max_length=16,
    )
    d2t, t2d = generate_vocab_mapping(prompts, target_vocab_size=32, draft_vocab_size=3)
    assert torch.where(t2d)[0].tolist() == [3, 6, 7]
    assert (d2t + torch.arange(3)).tolist() == [3, 6, 7]


@pytest.fixture
def sample_data():
    input_ids_list = [
        [10, 20, 30, 40, 50],
        [10, 20, 60, 70, 80],
        [30, 40, 50, 90, 100],
    ]
    loss_mask_list = [
        [0, 0, 1, 1, 1],
        [0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1],
    ]
    return input_ids_list, loss_mask_list


@pytest.fixture
def multi_turn_data():
    """Simulate multi-turn conversations with alternating prompt/response segments."""
    input_ids_list = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ]
    loss_mask_list = [
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    ]
    return input_ids_list, loss_mask_list


def _count_expected(input_ids_list, loss_mask_list):
    """Reference implementation: straightforward per-token counting."""
    counts = Counter()
    for ids, mask in zip(input_ids_list, loss_mask_list):
        for token_id, m in zip(ids, mask):
            if m == 1:
                counts[token_id] += 1
    return counts


class TestCountTokenFrequencies:
    """Tests for _count_token_frequencies."""

    def test_packed_loss_mask_format(self, sample_data):
        """Counting with packed_loss_mask should match the reference."""
        input_ids_list, loss_mask_list = sample_data
        prompts = _make_prompts(input_ids_list, loss_mask_list)

        result = _count_token_frequencies(prompts)
        expected = _count_expected(input_ids_list, loss_mask_list)

        assert result == expected

    def test_multi_turn_packed_loss_mask(self, multi_turn_data):
        """Multi-turn conversations with alternating segments should count correctly."""
        input_ids_list, loss_mask_list = multi_turn_data
        prompts = _make_prompts(input_ids_list, loss_mask_list)

        result = _count_token_frequencies(prompts)
        expected = _count_expected(input_ids_list, loss_mask_list)

        assert result == expected

    def test_only_masked_tokens_counted(self, sample_data):
        """Tokens where loss_mask == 0 must not appear in the result."""
        input_ids_list, loss_mask_list = sample_data
        prompts = _make_prompts(input_ids_list, loss_mask_list)

        result = _count_token_frequencies(prompts)

        unmasked_ids = set()
        for ids, mask in zip(input_ids_list, loss_mask_list):
            for token_id, m in zip(ids, mask):
                if m == 0:
                    unmasked_ids.add(token_id)
        masked_ids = set()
        for ids, mask in zip(input_ids_list, loss_mask_list):
            for token_id, m in zip(ids, mask):
                if m == 1:
                    masked_ids.add(token_id)

        only_unmasked = unmasked_ids - masked_ids
        for token_id in only_unmasked:
            assert token_id not in result, f"Token {token_id} should not be counted (loss_mask=0)"

    def test_all_zeros_loss_mask(self):
        """All-zero loss_mask should produce empty counts."""
        prompts = _make_prompts([[10, 20, 30]], [[0, 0, 0]])
        result = _count_token_frequencies(prompts)
        assert len(result) == 0

    def test_all_ones_loss_mask(self):
        """All-ones loss_mask should count every token."""
        prompts = _make_prompts([[10, 20, 10]], [[1, 1, 1]])
        result = _count_token_frequencies(prompts)
        assert result == Counter({10: 2, 20: 1})

    def test_input_ids_as_list(self, sample_data):
        """input_ids stored as plain lists (not tensors) should work."""
        input_ids_list, loss_mask_list = sample_data
        prompts = []
        for ids, mask in zip(input_ids_list, loss_mask_list):
            packed = pack_loss_mask(torch.tensor(mask))
            prompts.append(
                {
                    "input_ids": ids,
                    "packed_loss_mask": serialize_packed_loss_mask(packed),
                }
            )

        result = _count_token_frequencies(prompts)
        expected = _count_expected(input_ids_list, loss_mask_list)

        assert result == expected

    def test_no_packed_loss_mask_defaults_to_all_ones(self):
        """When packed_loss_mask is absent, count all tokens."""
        prompts = [{"input_ids": torch.tensor([10, 20, 10])}]
        result = _count_token_frequencies(prompts)
        assert result == Counter({10: 2, 20: 1})


class TestGenerateVocabMapping:
    """Tests for generate_vocab_mapping end-to-end."""

    def test_output_shapes(self, sample_data):
        input_ids_list, loss_mask_list = sample_data
        prompts = _make_prompts(input_ids_list, loss_mask_list)

        target_vocab_size = 200
        draft_vocab_size = 5

        d2t, t2d = generate_vocab_mapping(
            prompts=prompts,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )

        assert d2t.shape == (draft_vocab_size,)
        assert t2d.shape == (target_vocab_size,)

    def test_d2t_maps_correctly(self, sample_data):
        """d2t[i] + i should give the target token id for draft token i."""
        input_ids_list, loss_mask_list = sample_data
        prompts = _make_prompts(input_ids_list, loss_mask_list)

        target_vocab_size = 200
        draft_vocab_size = 5

        d2t, t2d = generate_vocab_mapping(
            prompts=prompts,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )

        target_ids = torch.arange(draft_vocab_size) + d2t
        assert torch.all(target_ids[1:] > target_ids[:-1])
        assert torch.all(target_ids >= 0)
        assert torch.all(target_ids < target_vocab_size)

    def test_t2d_consistency_with_d2t(self, sample_data):
        """t2d should mark exactly the target ids that d2t maps to."""
        input_ids_list, loss_mask_list = sample_data
        prompts = _make_prompts(input_ids_list, loss_mask_list)

        target_vocab_size = 200
        draft_vocab_size = 5

        d2t, t2d = generate_vocab_mapping(
            prompts=prompts,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )

        target_ids = torch.arange(draft_vocab_size) + d2t
        assert t2d.sum() == draft_vocab_size
        for tid in target_ids:
            assert t2d[tid], f"t2d[{tid}] should be True"


class TestProcessTokenDictToMappings:
    """Tests for process_token_dict_to_mappings."""

    def test_basic(self):
        token_dict = Counter({10: 5, 20: 3, 30: 1})
        d2t, t2d = process_token_dict_to_mappings(
            token_dict, draft_vocab_size=3, target_vocab_size=50
        )

        assert d2t.shape == (3,)
        assert t2d.shape == (50,)

        target_ids = torch.arange(3) + d2t
        assert set(target_ids.tolist()) == {10, 20, 30}

    def test_fills_missing_when_fewer_unique_tokens(self):
        token_dict = Counter({100: 5})
        d2t, t2d = process_token_dict_to_mappings(
            token_dict, draft_vocab_size=4, target_vocab_size=200
        )

        assert d2t.shape == (4,)
        assert t2d.sum() == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

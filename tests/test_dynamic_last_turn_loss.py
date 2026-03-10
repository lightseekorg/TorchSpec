"""Tests for dynamic per-sample last_turn_loss_only based on thinking content detection."""

import torch

from torchspec.data.parse import has_thinking_content
from torchspec.data.utils import DataCollatorWithPadding


# ── has_thinking_content detection ────────────────────────────────────


class TestHasThinkingContent:
    def test_think_tag_with_content(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<think>reasoning here</think>Hello!"},
        ]
        assert has_thinking_content(conv) is True

    def test_empty_think_tag(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<think></think>Hello!"},
        ]
        assert has_thinking_content(conv) is False

    def test_think_tag_whitespace_only(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<think>   \n\t  </think>Hello!"},
        ]
        assert has_thinking_content(conv) is False

    def test_think_tag_with_single_char(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<think> x</think>answer"},
        ]
        assert has_thinking_content(conv) is True

    def test_no_think_tag(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
        ]
        assert has_thinking_content(conv) is False

    def test_thinking_field(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "thinking": "some reasoning"},
        ]
        assert has_thinking_content(conv) is True

    def test_thinking_content_field(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "thinking_content": "reasoning"},
        ]
        assert has_thinking_content(conv) is True

    def test_empty_thinking_field(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "thinking": ""},
        ]
        assert has_thinking_content(conv) is False

    def test_none_thinking_field(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "thinking": None},
        ]
        assert has_thinking_content(conv) is False

    def test_user_message_with_think_tag_ignored(self):
        conv = [
            {"role": "user", "content": "<think>user put this here</think>Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert has_thinking_content(conv) is False

    def test_multi_turn_thinking_in_earlier_turn(self):
        conv = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>thought</think>A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "<think></think>A2"},
        ]
        assert has_thinking_content(conv) is True

    def test_multi_turn_no_thinking(self):
        conv = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        assert has_thinking_content(conv) is False

    def test_empty_conversation(self):
        assert has_thinking_content([]) is False

    def test_non_dict_messages(self):
        assert has_thinking_content(["not a dict"]) is False

    def test_multimodal_content_no_thinking(self):
        conv = [
            {"role": "user", "content": [{"type": "text", "text": "Describe"}]},
            {"role": "assistant", "content": "A cat."},
        ]
        assert has_thinking_content(conv) is False

    def test_system_message_ignored(self):
        conv = [
            {"role": "system", "content": "<think>system thinking</think>"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert has_thinking_content(conv) is False


# ── Collator reads pre-computed loss_mask from MooncakeDataset ────────


class TestCollatorPrecomputedMask:
    def _make_collator(self):
        return DataCollatorWithPadding()

    def test_precomputed_loss_mask_used(self):
        collator = self._make_collator()
        mask_tensor = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]], dtype=torch.long)
        item = {
            "input_ids": torch.zeros(1, 12, dtype=torch.long),
            "loss_mask": mask_tensor,
        }
        result = collator._get_loss_mask(item)
        assert torch.equal(result, mask_tensor)

    def test_missing_loss_mask_raises(self):
        import pytest

        collator = self._make_collator()
        item = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        with pytest.raises(KeyError):
            collator._get_loss_mask(item)


# ── MooncakeDataset._compute_loss_mask (single source of truth) ──────


class TestComputeLossMask:
    """Test that _compute_loss_mask computes, stores, and skips correctly."""

    def _make_dataset(self, dynamic=True, last_turn_loss_only=False):
        from torchspec.training.data_fetcher import MooncakeDataset

        ds = MooncakeDataset.__new__(MooncakeDataset)
        ds.assistant_header_ids = [10, 20]
        ds.end_token_ids = [30, 40]
        ds.dynamic_loss_mask = dynamic
        ds.last_turn_loss_only = last_turn_loss_only
        return ds

    def test_packed_loss_mask_nonzero(self):
        ds = self._make_dataset(dynamic=False)
        data = {"packed_loss_mask": "2,3,2"}
        mask = ds._compute_loss_mask(data)
        assert mask is not None
        assert mask.tolist() == [0, 0, 1, 1, 1, 0, 0]
        assert "loss_mask" in data

    def test_packed_loss_mask_zero(self):
        ds = self._make_dataset(dynamic=False)
        data = {"packed_loss_mask": "10"}
        assert ds._compute_loss_mask(data) is None

    def test_dynamic_mask_nonzero(self):
        ds = self._make_dataset(dynamic=True)
        data = {"input_ids": torch.tensor([10, 20, 1, 2, 30, 40], dtype=torch.long)}
        mask = ds._compute_loss_mask(data)
        assert mask is not None
        assert mask.tolist() == [0, 0, 1, 1, 0, 0]
        assert "loss_mask" in data

    def test_dynamic_mask_zero(self):
        ds = self._make_dataset(dynamic=True)
        data = {"input_ids": torch.tensor([5, 6, 7, 8], dtype=torch.long)}
        assert ds._compute_loss_mask(data) is None

    def test_dynamic_mask_per_sample_last_turn_only(self):
        ds = self._make_dataset(dynamic=True, last_turn_loss_only=False)
        ids = [10, 20, 1, 30, 40, 10, 20, 2, 30, 40]
        data = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "last_turn_loss_only": True,
        }
        mask = ds._compute_loss_mask(data)
        assert mask is not None
        assert mask.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        assert "loss_mask" in data

    def test_dynamic_mask_per_sample_all_turns(self):
        ds = self._make_dataset(dynamic=True, last_turn_loss_only=True)
        ids = [10, 20, 1, 2, 30, 40, 10, 20, 3, 4, 30, 40]
        data = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "last_turn_loss_only": False,
        }
        mask = ds._compute_loss_mask(data)
        assert mask is not None
        assert mask.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

    def test_no_params_defaults_nonzero(self):
        ds = self._make_dataset(dynamic=False)
        data = {"input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
        assert ds._compute_loss_mask(data) is not None

    def test_empty_packed_loss_mask(self):
        ds = self._make_dataset(dynamic=False)
        data = {"packed_loss_mask": ""}
        assert ds._compute_loss_mask(data) is None

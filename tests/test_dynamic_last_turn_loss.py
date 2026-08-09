"""Tests for dynamic per-sample last_turn_loss_only based on thinking content detection."""

import torch

from torchspec.data.parse import has_thinking_content
from torchspec.data.utils import DataCollatorWithPadding, resolve_loss_mask

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

    def test_reasoning_content_field(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "Let me think..."},
        ]
        assert has_thinking_content(conv) is True

    def test_reasoning_field(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning": "step by step"},
        ]
        assert has_thinking_content(conv) is True

    def test_empty_reasoning_content_field(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": ""},
        ]
        assert has_thinking_content(conv) is False

    def test_none_reasoning_content_field(self):
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": None},
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


# ── resolve_loss_mask (single source of truth in data/utils.py) ───────

_HEADER = [10, 20]
_END = [30, 40]


class TestResolveLossMask:
    """Test that resolve_loss_mask computes, stores, and skips correctly."""

    def test_packed_loss_mask_nonzero(self):
        data = {"packed_loss_mask": "2,3,2"}
        mask = resolve_loss_mask(data)
        assert mask is not None
        assert mask.tolist() == [0, 0, 1, 1, 1, 0, 0]
        assert "loss_mask" in data

    def test_packed_loss_mask_zero(self):
        data = {"packed_loss_mask": "10"}
        assert resolve_loss_mask(data) is None

    def test_dynamic_mask_nonzero(self):
        data = {"input_ids": torch.tensor([10, 20, 1, 2, 30, 40], dtype=torch.long)}
        mask = resolve_loss_mask(
            data,
            dynamic_loss_mask=True,
            assistant_header_ids=_HEADER,
            end_token_ids=_END,
        )
        assert mask is not None
        assert mask.tolist() == [0, 0, 1, 1, 0, 0]
        assert "loss_mask" in data

    def test_dynamic_mask_zero(self):
        data = {"input_ids": torch.tensor([5, 6, 7, 8], dtype=torch.long)}
        assert (
            resolve_loss_mask(
                data,
                dynamic_loss_mask=True,
                assistant_header_ids=_HEADER,
                end_token_ids=_END,
            )
            is None
        )

    def test_dynamic_mask_per_sample_last_turn_only(self):
        ids = [10, 20, 1, 30, 40, 10, 20, 2, 30, 40]
        data = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "last_turn_loss_only": True,
        }
        mask = resolve_loss_mask(
            data,
            dynamic_loss_mask=True,
            assistant_header_ids=_HEADER,
            end_token_ids=_END,
            last_turn_loss_only=False,
        )
        assert mask is not None
        assert mask.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        assert "loss_mask" in data

    def test_dynamic_mask_per_sample_all_turns(self):
        ids = [10, 20, 1, 2, 30, 40, 10, 20, 3, 4, 30, 40]
        data = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "last_turn_loss_only": False,
        }
        mask = resolve_loss_mask(
            data,
            dynamic_loss_mask=True,
            assistant_header_ids=_HEADER,
            end_token_ids=_END,
            last_turn_loss_only=True,
        )
        assert mask is not None
        assert mask.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

    def test_unresolved_auto_global_supervises_all_turns(self):
        ids = [10, 20, 1, 2, 30, 40, 10, 20, 3, 4, 30, 40]
        data = {"input_ids": torch.tensor(ids, dtype=torch.long)}
        mask = resolve_loss_mask(
            data,
            dynamic_loss_mask=True,
            assistant_header_ids=_HEADER,
            end_token_ids=_END,
            last_turn_loss_only="auto",
        )
        assert mask is not None
        assert mask.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

    def test_per_sample_decision_overrides_auto_global(self):
        ids = [10, 20, 1, 2, 30, 40, 10, 20, 3, 4, 30, 40]
        data = {"input_ids": torch.tensor(ids, dtype=torch.long), "last_turn_loss_only": True}
        mask = resolve_loss_mask(
            data,
            dynamic_loss_mask=True,
            assistant_header_ids=_HEADER,
            end_token_ids=_END,
            last_turn_loss_only="auto",
        )
        assert mask is not None
        assert mask.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    def test_no_params_defaults_nonzero(self):
        data = {"input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
        assert resolve_loss_mask(data) is not None

    def test_empty_packed_loss_mask(self):
        data = {"packed_loss_mask": ""}
        assert resolve_loss_mask(data) is None


# ── auto decisions survive the multimodal mask drop ───────────────────


_THINKING_CONV = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "<think>reasoning</think>Hello!"},
]
_PLAIN_CONV = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]


class _StubRenderer:
    """Minimal ConversationRenderer stand-in returning a fixed token stream."""

    def render(self, messages, tools, max_seq_length, last_turn_only, generation_config):
        return [1, 2, 3, 4], [0, 1, 1, 0]


class TestTokenizeWorkerAutoMetadata:
    def _run(self, conversation, last_turn_loss_only):
        from torchspec.data import dataset as dataset_mod

        saved = dict(dataset_mod._worker_state)
        dataset_mod._worker_state.clear()
        dataset_mod._worker_state.update(
            {
                "renderer": _StubRenderer(),
                "last_turn_loss_only": last_turn_loss_only,
                "min_loss_tokens": 0,
            }
        )
        try:
            return dataset_mod._tokenize_single((conversation, None, None, 128, False))
        finally:
            dataset_mod._worker_state.clear()
            dataset_mod._worker_state.update(saved)

    def test_auto_records_thinking_decision(self):
        assert self._run(_THINKING_CONV, "auto")["has_thinking"] is True

    def test_auto_records_non_thinking_decision(self):
        assert self._run(_PLAIN_CONV, "auto")["has_thinking"] is False

    def test_explicit_flag_records_nothing(self):
        assert "has_thinking" not in self._run(_THINKING_CONV, True)


class TestDropStaleMultimodalMasks:
    def _prompt(self, metadata=None):
        return {
            "data_id": "s0",
            "multimodal_inputs": [{"type": "image", "url": "x"}],
            "packed_loss_mask": "1,2,1",
            "metadata": metadata or {},
        }

    def test_mask_dropped_when_decision_recorded(self):
        from torchspec.data.dataset import _drop_stale_multimodal_masks

        prompts = [self._prompt({"has_thinking": False})]
        _drop_stale_multimodal_masks(prompts, True, last_turn_loss_only="auto")
        assert "packed_loss_mask" not in prompts[0]

    def test_missing_decision_under_auto_raises(self):
        import pytest

        from torchspec.data.dataset import _drop_stale_multimodal_masks

        with pytest.raises(ValueError, match="last_turn_loss_only"):
            _drop_stale_multimodal_masks([self._prompt()], True, last_turn_loss_only="auto")

    def test_missing_decision_under_explicit_flag_allowed(self):
        from torchspec.data.dataset import _drop_stale_multimodal_masks

        prompts = [self._prompt()]
        _drop_stale_multimodal_masks(prompts, True, last_turn_loss_only=False)
        assert "packed_loss_mask" not in prompts[0]

    def test_text_rows_keep_their_mask(self):
        from torchspec.data.dataset import _drop_stale_multimodal_masks

        prompts = [{"data_id": "s0", "multimodal_inputs": None, "packed_loss_mask": "1,2,1"}]
        _drop_stale_multimodal_masks(prompts, True, last_turn_loss_only="auto")
        assert prompts[0]["packed_loss_mask"] == "1,2,1"

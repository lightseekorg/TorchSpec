"""Tests for KimiK25Parser and kimi-k25-vlm template."""

from unittest.mock import MagicMock

import pytest
import torch

from torchspec.data.parse import KimiK25Parser
from torchspec.data.template import TEMPLATE_REGISTRY


class MockKimiTokenizer:
    """Mock tokenizer that behaves like Kimi-K2.5 tokenizer for testing."""

    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 0
        self._vocab = {
            "<|im_user|>": 163600,
            "<|im_assistant|>": 163601,
            "<|media_begin|>": 163602,
            "<|media_content|>": 163603,
            "<|media_end|>": 163604,
            "<|media_pad|>": 163605,
            "<think>": 163606,
            "</think>": 163607,
            "<|im_middle|>": 163608,
            "<|im_end|>": 163609,
            "<|im_system|>": 163610,
            "user": 1000,
            "assistant": 1001,
            "system": 1002,
            "image": 1003,
            "\n": 1004,
        }
        self._next_id = 2000

    def _get_token_id(self, token: str) -> int:
        if token in self._vocab:
            return self._vocab[token]
        self._vocab[token] = self._next_id
        self._next_id += 1
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for special in sorted(self._vocab.keys(), key=len, reverse=True):
                if text[i:].startswith(special):
                    tokens.append(self._vocab[special])
                    i += len(special)
                    matched = True
                    break
            if not matched:
                word_end = i + 1
                while word_end < len(text) and text[word_end] not in " \n<>":
                    word_end += 1
                word = text[i:word_end]
                tokens.append(self._get_token_id(word))
                i = word_end
        return tokens

    def __call__(
        self,
        text: str,
        max_length: int = None,
        truncation: bool = False,
        return_tensors: str = None,
        add_special_tokens: bool = True,
    ):
        tokens = self.encode(text, add_special_tokens)
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        result = MagicMock()
        if return_tensors == "pt":
            result.input_ids = torch.tensor([tokens])
        else:
            result.input_ids = [tokens]
        return result

    def decode(self, token_ids: list) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return "".join(id_to_token.get(tid, f"[{tid}]") for tid in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, self.unk_token_id)


@pytest.fixture
def mock_tokenizer():
    return MockKimiTokenizer()


@pytest.fixture
def kimi_template():
    return TEMPLATE_REGISTRY.get("kimi-k25-vlm")


class TestKimiK25TemplateRegistration:
    """Tests for kimi-k25-vlm template registration."""

    def test_template_registered(self):
        assert "kimi-k25-vlm" in TEMPLATE_REGISTRY.get_all_template_names()

    def test_template_attributes(self):
        template = TEMPLATE_REGISTRY.get("kimi-k25-vlm")
        assert template.assistant_header == "<|im_assistant|>assistant<|im_middle|>"
        assert template.user_header == "<|im_user|>user<|im_middle|>"
        assert template.end_of_turn_token == "<|im_end|>"
        assert template.parser_type == "kimi-k25"
        assert template.system_prompt is None


class TestKimiK25ParserBasic:
    """Basic tests for KimiK25Parser."""

    def test_parser_initialization(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        assert (
            parser.MEDIA_TOKEN == "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        )
        assert parser.USER_HEADER == "<|im_user|>user<|im_middle|>"
        assert parser.ASSISTANT_HEADER == "<|im_assistant|>assistant<|im_middle|>"
        assert parser.END_TOKEN == "<|im_end|>"

    def test_single_turn_conversation(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(loss_mask, torch.Tensor)
        assert len(input_ids) == len(loss_mask)
        assert loss_mask.sum() > 0

    def test_multi_turn_conversation(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine."},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert isinstance(input_ids, torch.Tensor)
        assert loss_mask.sum() > 0


class TestKimiK25ParserImageHandling:
    """Tests for image placeholder handling."""

    def test_image_placeholder_replacement(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "<|image|> What is this?"},
            {"role": "assistant", "content": "A cat."},
        ]

        formatted = parser.format(conversation)

        assert "<|image|>" not in formatted
        assert "<|media_begin|>" in formatted
        assert "<|media_content|>" in formatted
        assert "<|media_end|>" in formatted
        assert "<|media_pad|>" in formatted

    def test_multiple_images(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {
                "role": "user",
                "content": "<|image|> First image. <|image|> Second image. Compare them.",
            },
            {"role": "assistant", "content": "They are different."},
        ]

        formatted = parser.format(conversation)

        assert formatted.count("<|media_begin|>") == 2
        assert formatted.count("<|media_end|>") == 2

    def test_multimodal_content_format(self, mock_tokenizer, kimi_template):
        """Test content as list of dicts (OpenAI multimodal format)."""
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                    {"type": "text", "text": "What is this?"},
                ],
            },
            {"role": "assistant", "content": "A cat."},
        ]

        formatted = parser.format(conversation)

        assert "<|media_begin|>" in formatted
        assert "What is this?" in formatted


class TestKimiK25ParserMediaTokenPassthrough:
    """Tests for expand_media_tokens=False (sglang inference passthrough)."""

    def test_string_placeholder_kept(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "<|image|> What is this?"},
            {"role": "assistant", "content": "A cat."},
        ]
        formatted = parser.format(conversation, expand_media_tokens=False)

        assert "<|image|>" in formatted
        assert "<|media_begin|>" not in formatted
        assert "<|media_pad|>" not in formatted

    def test_multiple_placeholders_kept(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "<|image|> First. <|image|> Second."},
            {"role": "assistant", "content": "Two images."},
        ]
        formatted = parser.format(conversation, expand_media_tokens=False)

        assert formatted.count("<|image|>") == 2
        assert "<|media_begin|>" not in formatted

    def test_list_content_placeholder_kept(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                    {"type": "text", "text": "Describe this."},
                ],
            },
            {"role": "assistant", "content": "A photo."},
        ]
        formatted = parser.format(conversation, expand_media_tokens=False)

        assert "<|image|>" in formatted
        assert "<|media_begin|>" not in formatted
        assert "Describe this." in formatted

    def test_default_expands_media_tokens(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "<|image|> What is this?"},
            {"role": "assistant", "content": "A cat."},
        ]
        formatted_default = parser.format(conversation)
        formatted_explicit = parser.format(conversation, expand_media_tokens=True)

        assert "<|media_begin|>" in formatted_default
        assert formatted_default == formatted_explicit

    def test_no_images_unaffected(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]
        fmt_expand = parser.format(conversation, expand_media_tokens=True)
        fmt_passthrough = parser.format(conversation, expand_media_tokens=False)

        assert fmt_expand == fmt_passthrough


class TestKimiK25ParserThinkingBlocks:
    """Tests for thinking block handling."""

    def test_thinking_block_preserved(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>Let me calculate...</think>The answer is 4."},
        ]

        formatted = parser.format(conversation)

        assert "<think>Let me calculate...</think>" in formatted

    def test_thinking_block_in_loss_mask(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<think>thinking</think>answer"},
        ]

        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert loss_mask.sum() > 0

    def test_multi_turn_with_thinking_strips_history(self, mock_tokenizer, kimi_template):
        """Only the last assistant turn keeps its thinking block."""
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "<think>thinking1</think>answer1"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "<think>thinking2</think>answer2"},
        ]

        formatted = parser.format(conversation)
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert "<think>thinking1</think>" not in formatted
        assert "answer1" in formatted
        assert "<think>thinking2</think>" in formatted
        assert loss_mask.sum() > 0

    def test_single_turn_thinking_preserved(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>Let me calculate...</think>The answer is 4."},
        ]

        formatted = parser.format(conversation)
        assert "<think>Let me calculate...</think>" in formatted

    def test_three_turns_only_last_thinking_kept(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>t1</think>A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "<think>t2</think>A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "<think>t3</think>A3"},
        ]

        formatted = parser.format(conversation)
        assert "<think>t1</think>" not in formatted
        assert "<think>t2</think>" not in formatted
        assert "<think>t3</think>" in formatted
        assert "A1" in formatted
        assert "A2" in formatted
        assert "A3" in formatted


class TestKimiK25ParserLossMask:
    """Tests for loss mask generation."""

    def test_loss_mask_only_on_assistant(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        formatted = parser.format(conversation)
        assert loss_mask.sum() > 0
        user_part_end = formatted.find("<|im_end|>")
        assert user_part_end > 0

    def test_loss_mask_multi_turn(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]

        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert loss_mask.sum() > 0


class TestKimiK25ParserSystemMessage:
    """Tests for system message handling."""

    def test_with_system_message(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        formatted = parser.format(conversation)

        assert "<|im_system|>" in formatted
        assert "You are helpful." in formatted


class TestKimiK25ParserTruncation:
    """Tests for max_length truncation."""

    def test_truncation(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "A" * 1000},
            {"role": "assistant", "content": "B" * 1000},
        ]

        input_ids, loss_mask = parser.parse(conversation, max_length=100)

        assert len(input_ids) <= 100
        assert len(loss_mask) <= 100


class TestKimiK25ParserPreformatted:
    """Tests for preformatted text parsing."""

    def test_preformatted_text(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        preformatted = (
            "<|im_user|>user<|im_middle|>Hello<|im_end|>"
            "<|im_assistant|>assistant<|im_middle|>Hi there!<|im_end|>"
        )

        input_ids, loss_mask = parser.parse(preformatted, max_length=512, preformatted=True)

        assert isinstance(input_ids, torch.Tensor)
        assert loss_mask.sum() > 0


class TestKimiK25ParserToolCalls:
    """Tests for tool/function call handling."""

    def test_tool_role_formatted(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }
                ],
            },
            {"role": "tool", "content": '{"temp": 72}', "tool_call_id": "call_1"},
            {"role": "assistant", "content": "It is 72 degrees."},
        ]

        formatted = parser.format(conversation)

        assert "<|im_system|>tool<|im_middle|>" in formatted
        assert "## Return of call_1" in formatted
        assert '{"temp": 72}' in formatted
        assert "<|tool_calls_section_begin|>" in formatted
        assert "<|tool_call_begin|>call_1" in formatted
        assert '<|tool_call_argument_begin|>{"city": "NYC"}' in formatted
        assert "<|tool_call_end|>" in formatted
        assert "<|tool_calls_section_end|>" in formatted

    def test_tool_role_no_loss_mask(self, mock_tokenizer, kimi_template):
        """Tool messages should not have loss — only assistant turns do."""
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Check weather"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": '{"temp": 72}', "tool_call_id": "call_1"},
            {"role": "assistant", "content": "72 degrees."},
        ]

        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert loss_mask.sum() > 0

    def test_tool_without_call_id(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "Ok."},
            {"role": "tool", "content": '{"result": "done"}'},
            {"role": "assistant", "content": "Done."},
        ]

        formatted = parser.format(conversation)

        assert "<|im_system|>tool<|im_middle|>" in formatted
        assert '{"result": "done"}' in formatted
        assert "## Return of" not in formatted

    def test_multiple_tool_calls(self, mock_tokenizer, kimi_template):
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Compare weather"},
            {
                "role": "assistant",
                "content": "Checking both.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "LA"}'},
                    },
                ],
            },
            {"role": "tool", "content": '{"temp": 72}', "tool_call_id": "call_1"},
            {"role": "tool", "content": '{"temp": 85}', "tool_call_id": "call_2"},
            {"role": "assistant", "content": "NYC is 72, LA is 85."},
        ]

        formatted = parser.format(conversation)

        assert formatted.count("<|tool_call_begin|>") == 2
        assert formatted.count("<|tool_call_end|>") == 2
        assert formatted.count("<|im_system|>tool<|im_middle|>") == 2
        assert "## Return of call_1" in formatted
        assert "## Return of call_2" in formatted

    def test_assistant_with_tool_calls_gets_loss(self, mock_tokenizer, kimi_template):
        """Assistant turns that include tool_calls should still have loss computed."""
        parser = KimiK25Parser(mock_tokenizer, kimi_template)
        conversation = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": '{"temp": 72}', "tool_call_id": "call_1"},
            {"role": "assistant", "content": "72 degrees."},
        ]

        input_ids, loss_mask = parser.parse(conversation, max_length=1024)

        assert loss_mask.sum() > 0
        assert isinstance(input_ids, torch.Tensor)

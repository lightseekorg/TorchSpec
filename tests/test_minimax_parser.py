"""Tests for MiniMaxParser and minimax-m2 template."""

from unittest.mock import MagicMock

import pytest
import torch

from torchspec.data.parse import MiniMaxParser
from torchspec.data.template import TEMPLATE_REGISTRY


class MockMiniMaxTokenizer:
    """Mock tokenizer that behaves like MiniMax-M2 tokenizer for testing."""

    def __init__(self):
        self.pad_token_id = 200000
        self.unk_token_id = 200021
        self._vocab = {
            "]~!b[": 200034,
            "]~b]": 200019,
            "[e~[": 200020,
            "]!d~[": 200021,
            "]!p~[": 200000,
            "<think>": 200050,
            "</think>": 200051,
            "<minimax:tool_call>": 200052,
            "</minimax:tool_call>": 200053,
            "]<]start of image[>[": 200029,
            "]<]end of image[>[": 200030,
            "]<]vision pad[>[": 200033,
            "\n": 1000,
            "system": 1001,
            "user": 1002,
            "ai": 1003,
            "tool": 1004,
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
                while word_end < len(text) and text[word_end] not in " \n<>[]":
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

    def decode(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        id_to_token = {v: k for k, v in self._vocab.items()}
        return "".join(id_to_token.get(tid, f"[{tid}]") for tid in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, self.unk_token_id)


@pytest.fixture
def mock_tokenizer():
    return MockMiniMaxTokenizer()


@pytest.fixture
def minimax_template():
    return TEMPLATE_REGISTRY.get("minimax-m2")


class TestMiniMaxTemplateRegistration:
    def test_template_registered(self):
        assert "minimax-m2" in TEMPLATE_REGISTRY.get_all_template_names()

    def test_template_attributes(self):
        template = TEMPLATE_REGISTRY.get("minimax-m2")
        assert template.assistant_header == "]~b]ai\n"
        assert template.user_header == "]~b]user\n"
        assert template.end_of_turn_token == "[e~["
        assert template.parser_type == "minimax-m2"
        assert template.image_placeholder == "<image>"
        assert (
            template.system_prompt
            == "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax."
        )


class TestMiniMaxParserBasic:
    def test_parser_initialization(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        assert parser.ASSISTANT_HEADER == "]~b]ai\n"
        assert parser.USER_HEADER == "]~b]user\n"
        assert parser.END_TOKEN == "[e~["
        assert parser.SYSTEM_HEADER == "]~!b[]~b]system\n"

    def test_single_turn_conversation(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(loss_mask, torch.Tensor)
        assert len(input_ids) == len(loss_mask)
        assert loss_mask.sum() > 0

    def test_multi_turn_conversation(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine."},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert isinstance(input_ids, torch.Tensor)
        assert loss_mask.sum() > 0

    def test_format_structure(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        formatted = parser.format(conversation)

        assert formatted.startswith("]~!b[]~b]system\n")
        assert "]~b]user\n" in formatted
        assert "]~b]ai\n" in formatted
        assert formatted.count("[e~[") == 3  # system + user + assistant


class TestMiniMaxParserSystemMessage:
    def test_explicit_system_message(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "system", "content": "You are a custom assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        formatted = parser.format(conversation)

        assert "You are a custom assistant." in formatted
        assert "MiniMax-M2.5" not in formatted

    def test_default_system_prompt(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        formatted = parser.format(conversation)

        assert "MiniMax-M2.5" in formatted

    def test_generation_prompt(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello"},
        ]
        formatted = parser.format(conversation, add_generation_prompt=True)

        assert formatted.endswith("]~b]ai\n<think>\n")


class TestMiniMaxParserThinkingBlocks:
    def test_thinking_preserved_single_turn(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>\nLet me calculate.\n</think>\n\nThe answer is 4.",
            },
        ]
        formatted = parser.format(conversation)

        assert "<think>" in formatted
        assert "Let me calculate." in formatted
        assert "The answer is 4." in formatted

    def test_thinking_stripped_from_earlier_turns(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>\nt1\n</think>\n\nA1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "<think>\nt2\n</think>\n\nA2"},
        ]
        formatted = parser.format(conversation)

        assert "t1" not in formatted
        assert "A1" in formatted
        assert "<think>" in formatted
        assert "t2" in formatted
        assert "A2" in formatted

    def test_three_turns_only_last_thinking_kept(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "<think>\nt1\n</think>\n\nA1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "<think>\nt2\n</think>\n\nA2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "<think>\nt3\n</think>\n\nA3"},
        ]
        formatted = parser.format(conversation)

        assert "t1" not in formatted
        assert "t2" not in formatted
        assert "t3" in formatted
        assert "A1" in formatted
        assert "A2" in formatted
        assert "A3" in formatted

    def test_reasoning_content_field(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "The answer.", "reasoning_content": "Let me think."},
        ]
        formatted = parser.format(conversation)

        assert "<think>\nLet me think.\n</think>" in formatted
        assert "The answer." in formatted

    def test_thinking_in_loss_mask(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<think>\nthinking\n</think>\n\nanswer"},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert loss_mask.sum() > 0


class TestMiniMaxParserImageHandling:
    def test_image_placeholder_replacement(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "<image>\nWhat is this?"},
            {"role": "assistant", "content": "A cat."},
        ]
        formatted = parser.format(conversation)

        assert "<image>" not in formatted
        assert "]<]start of image[>[" in formatted
        assert "]<]vision pad[>[" in formatted
        assert "]<]end of image[>[" in formatted

    def test_multiple_images(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "<image>\n<image>\nCompare these."},
            {"role": "assistant", "content": "They differ."},
        ]
        formatted = parser.format(conversation)

        assert formatted.count("]<]start of image[>[") == 2
        assert formatted.count("]<]end of image[>[") == 2

    def test_image_not_replaced_in_assistant(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Use <image> tag."},
        ]
        formatted = parser.format(conversation)

        assert "]<]start of image[>[" not in formatted

    def test_multimodal_list_content(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is this?"},
                ],
            },
            {"role": "assistant", "content": "A dog."},
        ]
        formatted = parser.format(conversation)

        assert "]<]start of image[>[" in formatted
        assert "What is this?" in formatted

    def test_multimodal_image_url_content(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe:"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                ],
            },
            {"role": "assistant", "content": "A landscape."},
        ]
        formatted = parser.format(conversation)

        assert "]<]start of image[>[" in formatted
        assert "Describe:" in formatted


class TestMiniMaxParserMediaTokenPassthrough:
    """Tests for expand_media_tokens=False (sglang inference passthrough)."""

    def test_string_placeholder_kept(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "<image>\nWhat is this?"},
            {"role": "assistant", "content": "A cat."},
        ]
        formatted = parser.format(conversation, expand_media_tokens=False)

        assert "<image>" in formatted
        assert "]<]start of image[>[" not in formatted
        assert "]<]vision pad[>[" not in formatted

    def test_multiple_placeholders_kept(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "<image>\n<image>\nCompare."},
            {"role": "assistant", "content": "They differ."},
        ]
        formatted = parser.format(conversation, expand_media_tokens=False)

        assert formatted.count("<image>") == 2
        assert "]<]start of image[>[" not in formatted

    def test_list_content_placeholder_kept(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this."},
                ],
            },
            {"role": "assistant", "content": "A photo."},
        ]
        formatted = parser.format(conversation, expand_media_tokens=False)

        assert "<image>" in formatted
        assert "]<]start of image[>[" not in formatted
        assert "Describe this." in formatted

    def test_default_expands_media_tokens(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "<image>\nWhat is this?"},
            {"role": "assistant", "content": "A cat."},
        ]
        formatted_default = parser.format(conversation)
        formatted_explicit = parser.format(conversation, expand_media_tokens=True)

        assert "]<]start of image[>[" in formatted_default
        assert formatted_default == formatted_explicit

    def test_no_images_unaffected(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]
        fmt_expand = parser.format(conversation, expand_media_tokens=True)
        fmt_passthrough = parser.format(conversation, expand_media_tokens=False)

        assert fmt_expand == fmt_passthrough


class TestMiniMaxParserToolCalls:
    def test_tool_call_formatted(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
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
            {"role": "tool", "content": '{"temp": 72}'},
            {"role": "assistant", "content": "It is 72 degrees."},
        ]
        formatted = parser.format(conversation)

        assert "<minimax:tool_call>" in formatted
        assert "</minimax:tool_call>" in formatted
        assert '<invoke name="get_weather">' in formatted
        assert '<parameter name="city">NYC</parameter>' in formatted
        assert "<response>" in formatted
        assert '{"temp": 72}' in formatted

    def test_multiple_tool_calls(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
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
            {"role": "tool", "content": '{"temp": 72}'},
            {"role": "tool", "content": '{"temp": 85}'},
            {"role": "assistant", "content": "NYC 72, LA 85."},
        ]
        formatted = parser.format(conversation)

        assert formatted.count('<invoke name="get_weather">') == 2
        assert formatted.count("<response>") == 2
        # Consecutive tool messages share one header
        assert formatted.count("]~b]tool") == 1

    def test_tool_call_with_dict_arguments(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "Ok.",
                "tool_calls": [
                    {
                        "function": {"name": "my_func", "arguments": {"x": 1, "y": "hello"}},
                    }
                ],
            },
            {"role": "tool", "content": "done"},
            {"role": "assistant", "content": "Done."},
        ]
        formatted = parser.format(conversation)

        assert '<parameter name="x">1</parameter>' in formatted
        assert '<parameter name="y">hello</parameter>' in formatted

    def test_tool_no_loss_mask(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Check"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {"function": {"name": "check", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Done."},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert loss_mask.sum() > 0

    def test_consecutive_tool_messages_grouped(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Do stuff"},
            {
                "role": "assistant",
                "content": "Ok.",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": "{}"}},
                    {"function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "result_a"},
            {"role": "tool", "content": "result_b"},
            {"role": "assistant", "content": "All done."},
        ]
        formatted = parser.format(conversation)

        # Only one tool header for the group
        tool_header_count = formatted.count("]~b]tool")
        assert tool_header_count == 1
        assert "<response>result_a</response>" in formatted
        assert "<response>result_b</response>" in formatted


class TestMiniMaxParserLossMask:
    def test_loss_mask_only_on_assistant(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert loss_mask.sum() > 0

    def test_loss_mask_multi_turn(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=512)

        assert loss_mask.sum() > 0

    def test_last_turn_only(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        _, mask_all = parser.parse(conversation, max_length=512)
        _, mask_last = parser.parse(conversation, max_length=512, last_turn_only=True)

        assert mask_last.sum() > 0
        assert mask_last.sum() <= mask_all.sum()


class TestMiniMaxParserTruncation:
    def test_truncation(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "A" * 1000},
            {"role": "assistant", "content": "B" * 1000},
        ]
        input_ids, loss_mask = parser.parse(conversation, max_length=100)

        assert len(input_ids) <= 100
        assert len(loss_mask) <= 100


class TestMiniMaxParserPreformatted:
    def test_preformatted_text(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        preformatted = (
            "]~!b[]~b]system\nYou are helpful.[e~[\n]~b]user\nHello[e~[\n]~b]ai\nHi there![e~[\n"
        )
        input_ids, loss_mask = parser.parse(preformatted, max_length=512, preformatted=True)

        assert isinstance(input_ids, torch.Tensor)
        assert loss_mask.sum() > 0


class TestMiniMaxParserNullContent:
    def test_null_content_with_tool_calls(self, mock_tokenizer, minimax_template):
        parser = MiniMaxParser(mock_tokenizer, minimax_template)
        conversation = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "action", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "done"},
            {"role": "assistant", "content": "Finished."},
        ]
        formatted = parser.format(conversation)

        assert "<minimax:tool_call>" in formatted
        assert "None" not in formatted.split("]~b]ai\n")[1].split("[e~[")[0]

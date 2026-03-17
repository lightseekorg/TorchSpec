"""Integration tests for Kimi-K2.5 dataset parsing end-to-end flow."""

import json
import os
import tempfile

import pytest
import torch

from torchspec.data.parse import KimiK25Parser
from torchspec.data.preprocessing import preprocess_conversations
from torchspec.data.template import TEMPLATE_REGISTRY
from torchspec.data.utils import (
    DataCollatorWithPadding,
    deserialize_packed_loss_mask,
    unpack_loss_mask,
)


@pytest.fixture
def kimi_template():
    return TEMPLATE_REGISTRY.get("kimi-k25-vlm")


@pytest.fixture
def sample_converted_data():
    """Sample data in the format produced by convert_multimodal_dataset.py"""
    return {
        "id": "test_sample_001",
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
                        },
                    },
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
            {
                "role": "assistant",
                "content": "<think>I can see a furry animal with whiskers and pointed ears. It appears to be a domestic cat.</think>This image shows a cat.",
            },
        ],
        "image_urls": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
        ],
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_multi_turn_data():
    """Multi-turn conversation sample."""
    return {
        "id": "test_multi_turn",
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"
                        },
                    },
                    {"type": "text", "text": "What is this?"},
                ],
            },
            {"role": "assistant", "content": "<think>Analyzing the image...</think>A cat."},
            {"role": "user", "content": "What color is it?"},
            {
                "role": "assistant",
                "content": "<think>Looking at the fur color...</think>Orange and white.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
                        },
                    },
                    {"type": "text", "text": "And this one?"},
                ],
            },
            {
                "role": "assistant",
                "content": "<think>This appears to be a different animal...</think>A dog.",
            },
        ],
        "image_urls": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
        ],
    }


class MockKimiTokenizer:
    """Mock tokenizer for testing without real model."""

    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 0
        self._vocab = {}
        self._next_id = 1000

        special_tokens = [
            "<|im_user|>",
            "<|im_assistant|>",
            "<|im_system|>",
            "<|im_middle|>",
            "<|im_end|>",
            "<|media_begin|>",
            "<|media_content|>",
            "<|media_end|>",
            "<|media_pad|>",
            "<think>",
            "</think>",
            "user",
            "assistant",
            "system",
            "image",
            "\n",
        ]
        for i, tok in enumerate(special_tokens):
            self._vocab[tok] = i + 100

    def _get_token_id(self, token: str) -> int:
        if token not in self._vocab:
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
                tokens.append(self._get_token_id(text[i:word_end]))
                i = word_end
        return tokens

    def __call__(
        self, text, max_length=None, truncation=False, return_tensors=None, add_special_tokens=True
    ):
        tokens = self.encode(text, add_special_tokens)
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        class Result:
            pass

        result = Result()
        if return_tensors == "pt":
            result.input_ids = torch.tensor([tokens])
        else:
            result.input_ids = [tokens]
        return result

    def decode(self, token_ids: list) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return "".join(id_to_token.get(tid, f"[{tid}]") for tid in token_ids)


@pytest.fixture
def mock_tokenizer():
    return MockKimiTokenizer()


class TestEndToEndParsing:
    """End-to-end parsing tests."""

    def test_converted_data_to_parser(self, mock_tokenizer, kimi_template, sample_converted_data):
        """Test parsing converted data with KimiK25Parser."""
        parser = KimiK25Parser(mock_tokenizer, kimi_template)

        input_ids, loss_mask = parser.parse(sample_converted_data["conversations"], max_length=2048)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(loss_mask, torch.Tensor)
        assert len(input_ids) == len(loss_mask)
        assert loss_mask.sum() > 0

        decoded = mock_tokenizer.decode(input_ids.tolist())
        assert "<|media_begin|>" in decoded
        assert "<think>" in decoded
        assert "This image shows a cat." in decoded

    def test_multi_turn_parsing(self, mock_tokenizer, kimi_template, sample_multi_turn_data):
        """Test multi-turn conversation parsing strips historical thinking."""
        parser = KimiK25Parser(mock_tokenizer, kimi_template)

        formatted = parser.format(sample_multi_turn_data["conversations"])
        assert "<think>Analyzing the image...</think>" not in formatted
        assert "<think>Looking at the fur color...</think>" not in formatted
        assert "<think>This appears to be a different animal...</think>" in formatted
        assert "A cat." in formatted
        assert "Orange and white." in formatted
        assert "A dog." in formatted

        input_ids, loss_mask = parser.parse(
            sample_multi_turn_data["conversations"], max_length=4096
        )

        assert loss_mask.sum() > 0

        decoded = mock_tokenizer.decode(input_ids.tolist())
        assert decoded.count("<|im_assistant|>") == 3


class TestPreprocessConversationsIntegration:
    """Integration tests for preprocess_conversations with kimi-k25 template."""

    def test_preprocess_single_sample(self, mock_tokenizer, kimi_template, sample_converted_data):
        """Test preprocess_conversations with single sample."""
        result = preprocess_conversations(
            mock_tokenizer,
            [sample_converted_data["conversations"]],
            kimi_template,
            max_length=2048,
            use_packed_loss_mask=True,
        )

        assert "input_ids" in result
        assert "packed_loss_mask" in result
        assert len(result["input_ids"]) == 1
        assert len(result["packed_loss_mask"]) == 1

        packed = deserialize_packed_loss_mask(result["packed_loss_mask"][0])
        loss_mask = unpack_loss_mask(packed)
        assert loss_mask.sum() > 0

    def test_preprocess_batch(
        self, mock_tokenizer, kimi_template, sample_converted_data, sample_multi_turn_data
    ):
        """Test preprocess_conversations with batch of samples."""
        conversations = [
            sample_converted_data["conversations"],
            sample_multi_turn_data["conversations"],
        ]

        result = preprocess_conversations(
            mock_tokenizer,
            conversations,
            kimi_template,
            max_length=4096,
            use_packed_loss_mask=True,
        )

        assert len(result["input_ids"]) == 2
        assert len(result["packed_loss_mask"]) == 2

    def test_preprocess_to_collator(self, mock_tokenizer, kimi_template, sample_converted_data):
        """Test full flow: preprocess -> collator."""
        result = preprocess_conversations(
            mock_tokenizer,
            [sample_converted_data["conversations"]],
            kimi_template,
            max_length=2048,
            use_packed_loss_mask=True,
            include_attention_mask=True,
        )

        features = [
            {
                "input_ids": result["input_ids"][0],
                "attention_mask": result["attention_mask"][0],
                "loss_mask": unpack_loss_mask(result["packed_loss_mask"][0])[None, :],
            }
        ]

        collator = DataCollatorWithPadding()
        batch = collator(features)

        assert "input_ids" in batch
        assert "loss_mask" in batch
        assert "attention_mask" in batch


class TestReturnFormattedText:
    """Tests for return_formatted_text and add_generation_prompt parameters."""

    def test_return_formatted_text(self, mock_tokenizer, kimi_template, sample_converted_data):
        result = preprocess_conversations(
            mock_tokenizer,
            [sample_converted_data["conversations"]],
            kimi_template,
            max_length=2048,
            use_packed_loss_mask=True,
            return_formatted_text=True,
        )

        assert "formatted_text" in result
        assert len(result["formatted_text"]) == 1
        assert isinstance(result["formatted_text"][0], str)
        assert "<|im_assistant|>" in result["formatted_text"][0]

    def test_no_formatted_text_by_default(
        self, mock_tokenizer, kimi_template, sample_converted_data
    ):
        result = preprocess_conversations(
            mock_tokenizer,
            [sample_converted_data["conversations"]],
            kimi_template,
            max_length=2048,
            use_packed_loss_mask=True,
        )

        assert "formatted_text" not in result

    def test_formatted_text_batch_alignment(
        self, mock_tokenizer, kimi_template, sample_converted_data, sample_multi_turn_data
    ):
        conversations = [
            sample_converted_data["conversations"],
            sample_multi_turn_data["conversations"],
        ]

        result = preprocess_conversations(
            mock_tokenizer,
            conversations,
            kimi_template,
            max_length=4096,
            use_packed_loss_mask=True,
            return_formatted_text=True,
        )

        assert len(result["formatted_text"]) == len(result["input_ids"])


class TestJsonlFileIntegration:
    """Test reading from JSONL files (simulating converted dataset)."""

    def test_read_and_parse_jsonl(self, mock_tokenizer, kimi_template, sample_converted_data):
        """Test reading converted JSONL and parsing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_converted_data) + "\n")
            temp_path = f.name

        try:
            with open(temp_path) as f:
                loaded = json.loads(f.readline())

            parser = KimiK25Parser(mock_tokenizer, kimi_template)
            input_ids, loss_mask = parser.parse(loaded["conversations"], max_length=2048)

            assert len(input_ids) > 0
            assert loss_mask.sum() > 0
        finally:
            os.unlink(temp_path)

    def test_batch_jsonl_processing(
        self, mock_tokenizer, kimi_template, sample_converted_data, sample_multi_turn_data
    ):
        """Test batch processing multiple JSONL lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_converted_data) + "\n")
            f.write(json.dumps(sample_multi_turn_data) + "\n")
            temp_path = f.name

        try:
            conversations = []
            with open(temp_path) as f:
                for line in f:
                    data = json.loads(line)
                    conversations.append(data["conversations"])

            result = preprocess_conversations(
                mock_tokenizer,
                conversations,
                kimi_template,
                max_length=4096,
                use_packed_loss_mask=True,
            )

            assert len(result["input_ids"]) == 2
        finally:
            os.unlink(temp_path)


class TestRealTokenizerIntegration:
    """Integration tests with real Kimi tokenizer."""

    @pytest.fixture
    def real_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained("moonshotai/Kimi-K2.5", trust_remote_code=True)
        except Exception:
            pytest.skip("Kimi-K2.5 tokenizer not available")

    def test_full_pipeline_real_tokenizer(
        self, real_tokenizer, kimi_template, sample_converted_data
    ):
        """Test full pipeline with real tokenizer."""
        result = preprocess_conversations(
            real_tokenizer,
            [sample_converted_data["conversations"]],
            kimi_template,
            max_length=2048,
            use_packed_loss_mask=True,
        )

        assert len(result["input_ids"]) == 1

        input_ids = result["input_ids"][0].squeeze()
        decoded = real_tokenizer.decode(input_ids.tolist())

        assert "<|im_user|>" in decoded
        assert "<|im_assistant|>" in decoded
        assert "<|media_begin|>" in decoded
        assert "<think>" in decoded

    def test_media_tokens_present(self, real_tokenizer, kimi_template, sample_converted_data):
        """Verify media tokens are properly encoded."""
        parser = KimiK25Parser(real_tokenizer, kimi_template)
        input_ids, loss_mask = parser.parse(sample_converted_data["conversations"], max_length=2048)

        media_begin_id = real_tokenizer.convert_tokens_to_ids("<|media_begin|>")
        media_end_id = real_tokenizer.convert_tokens_to_ids("<|media_end|>")
        think_id = real_tokenizer.convert_tokens_to_ids("<think>")

        ids_list = input_ids.tolist()
        assert media_begin_id in ids_list
        assert media_end_id in ids_list
        assert think_id in ids_list

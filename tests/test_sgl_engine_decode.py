from types import SimpleNamespace

import pytest

from torchspec.inference.engine.sgl_engine_decode import SglDecodeEngineMixin


class FakeSglEngine:
    def __init__(self, completion_tokens: int = 3, output_ids=None, prompt_lengths=None):
        self.completion_tokens = completion_tokens
        self.output_ids = output_ids if output_ids is not None else [11, 12, 13]
        self.generate_kwargs = None
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(
                encode=lambda prompt: [0] * (prompt_lengths or {}).get(prompt, 1)
            )
        )

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if "prompt" in kwargs:
            batch_size = len(kwargs["prompt"])
        else:
            batch_size = len(kwargs["input_ids"])
        return [
            {
                "meta_info": {
                    "spec_training_mooncake_store_keys": [f"sample-key-{i}"],
                    "prompt_tokens": 5,
                    "completion_tokens": self.completion_tokens,
                },
                "output_ids": self.output_ids,
            }
            for i in range(batch_size)
        ]


class FakeDecodeEngine(SglDecodeEngineMixin):
    def __init__(
        self,
        args,
        completion_tokens: int = 3,
        output_ids=None,
        prompt_lengths=None,
    ):
        self.args = args
        self.rank = 0
        self._engine = FakeSglEngine(
            completion_tokens=completion_tokens,
            output_ids=output_ids,
            prompt_lengths=prompt_lengths,
        )

    def _extract_image_data(self, multimodal_inputs):
        return None

    def _get_tensor_shapes(self, seq_len):
        return {
            "input_ids": (seq_len,),
            "hidden_states": (seq_len, 16),
            "last_hidden_states": (seq_len, 16),
        }

    def _get_tensor_dtypes(self):
        return {
            "input_ids": "torch.int64",
            "hidden_states": "torch.bfloat16",
            "last_hidden_states": "torch.bfloat16",
        }


def make_args(**overrides):
    values = {
        "decode_max_new_tokens": 16,
        "decode_min_new_tokens": 2,
        "decode_stop_token_ids": None,
        "decode_temperature": 1.0,
        "decode_top_p": 1.0,
        "decode_top_k": -1,
        "attention_backend": "flex_attention",
        "max_seq_length": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_generate_with_decode_passes_min_new_tokens():
    engine = FakeDecodeEngine(make_args(decode_min_new_tokens=4))

    outputs = engine.generate_with_decode(data_id=["row-1"], formatted_prompts=["prompt"])

    assert outputs[0]["packed_loss_mask"] == "5,2"
    assert engine._engine.generate_kwargs["sampling_params"]["max_new_tokens"] == 16
    assert engine._engine.generate_kwargs["sampling_params"]["min_new_tokens"] == 4


def test_generate_with_decode_passes_stop_token_ids():
    engine = FakeDecodeEngine(make_args(decode_stop_token_ids=[163586]))

    engine.generate_with_decode(data_id=["row-1"], formatted_prompts=["prompt"])

    assert engine._engine.generate_kwargs["sampling_params"]["stop_token_ids"] == [163586]


def test_generate_with_decode_rejects_invalid_min_new_tokens():
    engine = FakeDecodeEngine(make_args(decode_max_new_tokens=2, decode_min_new_tokens=3))

    with pytest.raises(ValueError, match="cannot exceed"):
        engine.generate_with_decode(data_id=["row-1"], formatted_prompts=["prompt"])


def test_generate_with_decode_drops_zero_loss_completions():
    engine = FakeDecodeEngine(make_args(), completion_tokens=1)

    outputs = engine.generate_with_decode(data_id=["row-1"], formatted_prompts=["prompt"])

    assert outputs == [None]


def test_generate_with_decode_drops_leading_stop_token_completions():
    engine = FakeDecodeEngine(
        make_args(decode_stop_token_ids=[163586]),
        completion_tokens=3,
        output_ids=[163586, 11, 12],
    )

    outputs = engine.generate_with_decode(data_id=["row-1"], formatted_prompts=["prompt"])

    assert outputs == [None]


def test_generate_with_decode_skips_prompts_without_min_new_token_room():
    engine = FakeDecodeEngine(
        make_args(max_seq_length=4, decode_min_new_tokens=2),
        prompt_lengths={"too-long": 3},
    )

    outputs = engine.generate_with_decode(
        data_id=["row-1"],
        formatted_prompts=["too-long"],
    )

    assert outputs == [None]
    assert engine._engine.generate_kwargs is None


def test_generate_with_decode_preserves_batch_positions_when_skipping_prompts():
    engine = FakeDecodeEngine(
        make_args(max_seq_length=4, decode_min_new_tokens=2),
        prompt_lengths={"too-long": 3, "ok": 2},
    )

    outputs = engine.generate_with_decode(
        data_id=["row-1", "row-2"],
        formatted_prompts=["too-long", "ok"],
    )

    assert outputs[0] is None
    assert outputs[1]["mooncake_key"] == "sample-key-0"
    assert engine._engine.generate_kwargs["prompt"] == ["ok"]
    assert engine._engine.generate_kwargs["spec_training_data_id"] == ["row-2"]

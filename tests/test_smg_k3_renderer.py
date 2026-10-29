"""Policy/negative-path tests; real checkpoint parity is a separate required gate."""

import json
from types import SimpleNamespace

import pytest

from torchspec.data.renderers.smg import effective_renderer
from torchspec.data.renderers.smg_k3 import SmgK3Renderer, _K3Tokenizer


@pytest.mark.parametrize("flag", ["defer_tokenization", "train_with_decode"])
def test_k3_opt_in_rejects_unsupported_training_modes(flag):
    with pytest.raises(ValueError, match="requires"):
        effective_renderer(
            SimpleNamespace(rendering_backend="smg", renderer="kimi-k3", **{flag: True})
        )


def test_k3_returns_native_ids_without_prefix_rewriting():
    class Native:
        def render(self, messages, *, tools=None, template_kwargs=None):
            assert template_kwargs == {"thinking": True}
            return {
                "segmentwise": True,
                "input_ids": [99] if messages else [1],
                "text": "bad" if messages else "prefix",
            }

    tokenizer = object.__new__(_K3Tokenizer)
    tokenizer.native = Native()
    assert tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "answer"}],
        tokenize=True,
        add_generation_prompt=False,
        preserve_thinking=True,
        padding=False,
        truncation=False,
        return_tensors=None,
        return_dict=False,
    ) == [99]


@pytest.mark.parametrize(
    "config", [{}, {"thinking": True}, {"thinking": True, "thinking_effort": None}]
)
def test_k3_null_or_missing_effort_uses_smg_default(config):
    original = dict(config)
    assert SmgK3Renderer._resolve_generation_config(config) == (True, "max")
    assert config == original


@pytest.mark.parametrize("value", [[1, 2], {"x": "y"}, 1e-7])
@pytest.mark.parametrize("as_json", [False, True])
def test_k3_passes_nested_and_float_tool_values_to_native_render(value, as_json):
    tokenizer = object.__new__(_K3Tokenizer)
    arguments = {"value": value}
    if as_json:
        arguments = json.dumps(arguments)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "lookup", "arguments": arguments}}],
        }
    ]

    class Native:
        def render(self, actual, **kwargs):
            assert actual == messages
            assert kwargs["template_kwargs"] == {"thinking": True, "thinking_effort": None}
            return {"segmentwise": True, "input_ids": [99]}

    tokenizer.native = Native()
    assert tokenizer.apply_chat_template(
        messages,
        thinking_effort=None,
        tokenize=True,
        add_generation_prompt=False,
        preserve_thinking=True,
        padding=False,
        truncation=False,
        return_tensors=None,
        return_dict=False,
    ) == [99]


def test_k3_refuses_flat_encoding_and_nontext_inputs():
    class Native:
        def render(self, *args, **kwargs):
            return {"segmentwise": False}

    tokenizer = object.__new__(_K3Tokenizer)
    tokenizer.native = Native()
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=False,
        preserve_thinking=True,
        padding=False,
        truncation=False,
        return_tensors=None,
        return_dict=False,
    )
    with pytest.raises(ValueError, match="segmented"):
        tokenizer.apply_chat_template([{"role": "assistant", "content": "answer"}], **kwargs)
    with pytest.raises(ValueError, match="text-only"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "image_url"}]}], **kwargs
        )

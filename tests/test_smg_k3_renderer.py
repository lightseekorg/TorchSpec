"""Policy/negative-path tests; real checkpoint parity is a separate required gate."""

import json
from types import SimpleNamespace

import pytest

from torchspec.data.renderers.smg import effective_renderer
from torchspec.data.renderers.smg_k3 import SmgK3Renderer, _K3Tokenizer


@pytest.mark.parametrize("flag", ["defer_tokenization", "train_with_decode"])
@pytest.mark.parametrize("backend", ["auto", "huggingface", "smg"])
@pytest.mark.parametrize("identification", ["renderer", "model_name", "local_config"])
def test_k3_rejects_unsupported_training_modes(flag, backend, identification, tmp_path):
    args = SimpleNamespace(rendering_backend=backend, **{flag: True})
    if identification == "renderer":
        args.renderer = "kimi-k3"
    elif identification == "model_name":
        args.target_model_path = "org/Kimi-K3"
    else:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "kimi_k3"}))
        args.target_model_path = str(tmp_path)
    if backend == "smg":
        with pytest.raises(ValueError, match="K3 rendering requires.*including huggingface"):
            effective_renderer(args)
    else:
        # Legacy K3 owns its mask policy; never fall back to the generic parser.
        assert effective_renderer(args) == "kimi-k3"


@pytest.mark.parametrize("flag", ["defer_tokenization", "train_with_decode"])
@pytest.mark.parametrize("backend", ["auto", "huggingface", "smg"])
def test_k3_dataset_rejects_unsupported_modes(flag, backend, monkeypatch):
    import torchspec.data.dataset as dataset_module

    args = SimpleNamespace(
        target_model_path="org/Kimi-K3",
        rendering_backend=backend,
        max_seq_length=128,
        **{flag: True},
    )
    if flag == "defer_tokenization" or backend == "smg":
        with pytest.raises(ValueError, match="requires|require"):
            dataset_module.load_conversation_dataset(args)
    else:
        assert effective_renderer(args) == "kimi-k3"
        # The worker rejects decode before invoking either K3 renderer.
        monkeypatch.setattr(dataset_module, "_worker_state", {"renderer": object()})
        with pytest.raises(ValueError, match="do not support train_with_decode=True"):
            dataset_module._tokenize_single(([], None, None, 128, True))


@pytest.mark.parametrize("backend", ["auto", "huggingface", "smg"])
@pytest.mark.parametrize("mode", ["preformatted", "generation_prefix", "builder"])
def test_k3_public_apis_reject_unsupported_modes(backend, mode):
    from datasets import Dataset

    from torchspec.data.preprocessing import build_eagle3_dataset, preprocess_conversations

    tokenizer = SimpleNamespace(name_or_path="org/Kimi-K3")
    with pytest.raises(ValueError, match="complete, unformatted conversations"):
        if mode == "builder":
            build_eagle3_dataset(
                Dataset.from_dict({"text": ["preformatted"]}),
                tokenizer,
                chat_template="qwen",
                is_preformatted=True,
                rendering_backend=backend,
            )
        else:
            preprocess_conversations(
                tokenizer,
                ["preformatted"]
                if mode == "preformatted"
                else [[{"role": "user", "content": "hi"}]],
                "qwen",
                is_preformatted=mode == "preformatted",
                add_generation_prompt=mode == "generation_prefix",
                rendering_backend=backend,
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

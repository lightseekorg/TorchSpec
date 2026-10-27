"""Policy/negative-path tests; real checkpoint parity is a separate required gate."""

from types import SimpleNamespace

import pytest

from torchspec.data.renderers.smg import effective_renderer
from torchspec.data.renderers.smg_k3 import _K3Tokenizer


@pytest.mark.parametrize("flag", ["defer_tokenization", "train_with_decode"])
def test_k3_opt_in_rejects_unsupported_training_modes(flag):
    with pytest.raises(ValueError, match="requires"):
        effective_renderer(
            SimpleNamespace(
                rendering_backend="smg", renderer="kimi-k3", smg_exclude_k3=False, **{flag: True}
            )
        )


def test_k3_prefix_correction_fails_when_native_layout_changes():
    class Native:
        def render(self, messages, *, tools=None, template_kwargs=None):
            return {
                "segmentwise": True,
                "input_ids": [99] if messages else [1],
                "text": "bad" if messages else "prefix",
            }

    tokenizer = object.__new__(_K3Tokenizer)
    tokenizer.native = Native()
    with pytest.raises(ValueError, match="prefix changed"):
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": "answer"}],
            tokenize=True,
            add_generation_prompt=False,
            preserve_thinking=True,
            padding=False,
            truncation=False,
            return_tensors=None,
            return_dict=False,
        )


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

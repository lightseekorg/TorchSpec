"""SMG's segmented K3 encoding with the existing TorchSpec training policy."""

import json
from functools import lru_cache

from torchspec.data.renderers.kimi_k3 import K3Renderer


class _K3Tokenizer:
    def __init__(self, source):
        from smg_tokenizer import Tokenizer

        self.native = Tokenizer(source)
        self.special_tokens = {
            token: self.native.token_to_id(token)
            for token in ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")
        }

    @lru_cache(maxsize=32)
    def _implicit_effort_prefix(self, tools_json):
        tools = json.loads(tools_json)
        declarations = self.native.render([], tools=tools, template_kwargs={"thinking": False})
        effort = self.native.render([], template_kwargs={"thinking": True})
        if not declarations["segmentwise"] or not effort["segmentwise"]:
            raise ValueError("Expected the native segmented K3 renderer")
        return declarations, effort

    def apply_chat_template(
        self, messages, *, tools=None, thinking=True, thinking_effort=None, **kwargs
    ):
        expected = {
            "tokenize": True,
            "add_generation_prompt": False,
            "preserve_thinking": True,
            "padding": False,
            "truncation": False,
            "return_tensors": None,
            "return_dict": False,
        }
        if kwargs != expected:
            raise ValueError("SMG K3 requires the complete-conversation training contract")
        for message in messages:
            if message.get("content") is not None and not isinstance(message["content"], str):
                raise ValueError("SMG K3 training currently accepts text-only string content")
        result = self.native.render(
            messages,
            tools=tools,
            template_kwargs={"thinking": thinking, "thinking_effort": thinking_effort},
        )
        if not result["segmentwise"]:
            raise ValueError("Expected the native segmented K3 renderer")
        ids = result["input_ids"]
        if thinking and thinking_effort is None:
            # The checkpoint distinguishes an absent effort (defaults to max)
            # from explicit None (no directive). SMG's serving API conflates
            # them. Remove only its generated directive, after tool declarations.
            # Both pieces are rendered/encoded by SMG; never flatten/re-encode.
            declarations, effort = self._implicit_effort_prefix(json.dumps(tools))
            prefix = declarations["input_ids"] + effort["input_ids"]
            text_prefix = declarations["text"] + effort["text"]
            if (
                not effort["input_ids"]
                or ids[: len(prefix)] != prefix
                or not result["text"].startswith(text_prefix)
            ):
                raise ValueError(
                    "SMG K3 implicit-effort prefix changed; refusing to alter training IDs"
                )
            ids = declarations["input_ids"] + ids[len(prefix) :]
        return ids


class SmgK3Renderer(K3Renderer):
    # Reuse the established thinking validation, token-ID matcher, EOM policy,
    # last-turn selection and truncation without duplicating training semantics.
    CACHE_VERSION = "smg-k3-explicit-null-v1-" + K3Renderer.CACHE_VERSION

    def __init__(self, source):
        super().__init__(_K3Tokenizer(source))

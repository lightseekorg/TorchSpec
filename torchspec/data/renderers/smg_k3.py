"""SMG's segmented K3 encoding with checkpoint-default generation settings."""

from torchspec.data.renderers.kimi_k3 import K3Renderer


class _K3Tokenizer:
    def __init__(self, source):
        from torchspec_smg import Tokenizer

        self.native = Tokenizer(source)
        self.special_tokens = {
            token: self.native.token_to_id(token)
            for token in ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")
        }

    def apply_chat_template(self, messages, *, tools=None, thinking=True, **kwargs):
        template_kwargs = {"thinking": thinking}
        if "thinking_effort" in kwargs:
            effort = kwargs.pop("thinking_effort")
            # SMG currently maps explicit null to max, unlike the checkpoint.
            # Reject this unsupported mode instead of rewriting rendered IDs.
            if thinking and effort is None:
                raise ValueError(
                    "SMG K3 does not support explicit null effort with thinking enabled"
                )
            template_kwargs["thinking_effort"] = effort
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
                raise ValueError(
                    "SMG K3 training currently accepts text-only string content; "
                    "set dataset.rendering_backend=huggingface for the legacy path"
                )
        result = self.native.render(
            messages,
            tools=tools,
            template_kwargs=template_kwargs,
        )
        if not result["segmentwise"]:
            raise ValueError("Expected the native segmented K3 renderer")
        return result["input_ids"]


class SmgK3Renderer(K3Renderer):
    # Reuse the established thinking validation, token-ID matcher, EOM policy,
    # last-turn selection and truncation without duplicating training semantics.
    CACHE_VERSION = "smg-k3-native-v2-" + K3Renderer.CACHE_VERSION

    def __init__(self, source):
        super().__init__(_K3Tokenizer(source))

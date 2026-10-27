# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SMG's segmented K3 encoding with checkpoint-default generation settings."""

import json

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
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    arguments = call.get("function", call).get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            # Both renderers preserve invalid JSON as a raw block.
                            continue
                    if isinstance(arguments, dict) and any(
                        isinstance(value, (dict, list, float)) for value in arguments.values()
                    ):
                        raise ValueError(
                            "SMG K3 tool arguments containing objects, arrays or floats "
                            "are not checkpoint-equivalent; set "
                            "dataset.rendering_backend=huggingface"
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
    CACHE_VERSION = "smg-k3-native-v3-" + K3Renderer.CACHE_VERSION

    def __init__(self, source):
        super().__init__(_K3Tokenizer(source))

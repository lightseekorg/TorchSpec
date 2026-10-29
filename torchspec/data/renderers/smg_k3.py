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

from torchspec.data.renderers.kimi_k3 import K3Renderer


class _K3Tokenizer:
    """Adapt SMG's native segmented encoding to the K3 checkpoint interface.

    Attributes:
        native: Bundled SMG tokenizer used for rendering and encoding.
        special_tokens: K3 message-boundary token strings mapped to their IDs.
    """

    def __init__(self, source):
        from torchspec_smg import Tokenizer

        self.native = Tokenizer(source)
        self.special_tokens = {
            token: self.native.token_to_id(token)
            for token in ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")
        }

    def apply_chat_template(self, messages, *, tools=None, thinking=True, **kwargs):
        """Return native segmented K3 IDs for a complete text conversation.

        Callers must supply tokenize=True, add_generation_prompt=False,
        preserve_thinking=True, padding=False, truncation=False,
        return_tensors=None and a boolean return_dict, with no extra kwargs except
        thinking_effort. Omitted effort retains SMG's checkpoint default;
        null effort follows SMG's default when thinking is enabled.

        Tool arguments are serialized by SMG, including nested values and floats.
        Returned IDs are neither re-encoded nor truncated here. return_dict=True
        additionally returns native text for callers requesting formatted output.
        """
        template_kwargs = {"thinking": thinking}
        if "thinking_effort" in kwargs:
            effort = kwargs.pop("thinking_effort")
            template_kwargs["thinking_effort"] = effort
        return_dict = kwargs.pop("return_dict", False)
        expected = {
            "tokenize": True,
            "add_generation_prompt": False,
            "preserve_thinking": True,
            "padding": False,
            "truncation": False,
            "return_tensors": None,
        }
        if kwargs != expected or not isinstance(return_dict, bool):
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
        return result if return_dict else result["input_ids"]


class SmgK3Renderer(K3Renderer):
    """Render Kimi-K3 text using SMG's native segmented encoding.

    Construct with a local checkpoint source path. The parent renderer supplies
    assistant-mask construction, end-of-message exclusion and last-turn selection.
    IDs and masks are truncated together after native rendering.
    """

    # Reuse the established thinking validation, token-ID matcher, EOM policy,
    # last-turn selection. SMG owns tool-argument serialization.
    CACHE_VERSION = "smg-k3-native-v4-" + K3Renderer.CACHE_VERSION

    def __init__(self, source):
        super().__init__(_K3Tokenizer(source))

    @classmethod
    def _resolve_generation_config(cls, generation_config):
        # SMG treats missing/null effort as its default max. Keep the shared
        # validation for invalid effort names and contradictory row metadata.
        if isinstance(generation_config, dict):
            generation_config = dict(generation_config)
            generation_config.setdefault("thinking", True)
            if generation_config["thinking"] and generation_config.get("thinking_effort") is None:
                generation_config["thinking_effort"] = "max"
        return super()._resolve_generation_config(generation_config)

    def render(
        self,
        messages,
        tools=None,
        *,
        max_seq_length,
        last_turn_only=False,
        generation_config=None,
        return_formatted_text=False,
    ):
        """Mask native SMG IDs; optionally return the same render's full text."""
        if (
            not isinstance(max_seq_length, int)
            or isinstance(max_seq_length, bool)
            or max_seq_length <= 0
        ):
            raise ValueError("max_seq_length must be a positive integer")
        thinking, effort = self._resolve_generation_config(generation_config)
        if not thinking:
            self._validate_disabled_thinking_messages(messages)
        result = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            thinking=thinking,
            thinking_effort=effort,
            tokenize=True,
            add_generation_prompt=False,
            preserve_thinking=True,
            padding=False,
            truncation=False,
            return_tensors=None,
            return_dict=True,
        )
        ids = self._normalize_token_ids(result["input_ids"])
        mask = self.compute_loss_mask(ids, last_turn_only=last_turn_only)
        output = (ids[:max_seq_length], mask[:max_seq_length])
        return (*output, result["text"]) if return_formatted_text else output

"""In-process SMG tokenizer. No gateway, GPU, or HTTP service is required."""

import json

from torchspec_smg._native import BINDING_VERSION, TOKENIZERS_V1
from torchspec_smg._native import Tokenizer as _Tokenizer

__version__ = BINDING_VERSION
build_features = "tokenizers-v1" if TOKENIZERS_V1 else "default"


class Tokenizer:
    """Load a local, pinned checkpoint directory using SMG's tokenizer factory."""

    def __init__(self, path, *, chat_template_path=None):
        self._inner = _Tokenizer(str(path), chat_template_path)

    def encode(self, text, *, add_special_tokens=False):
        """Return token IDs for text, optionally adding the encoder's special tokens."""
        return self._inner.encode(text, add_special_tokens)

    def token_to_id(self, token):
        """Return the vocabulary ID for a token, or None if it is unknown."""
        return self._inner.token_to_id(token)

    @property
    def encode_path(self):
        """Return the name of the selected native encoder path."""
        return self._inner.encode_path

    @property
    def raw_tool_call_arguments(self):
        """Return whether the renderer accepts raw JSON strings as tool-call arguments."""
        return self._inner.raw_tool_call_arguments

    def render(self, messages, *, tools=None, template_kwargs=None, now=None):
        """Render a complete conversation with exact token IDs and encoding metadata.

        This entry point renders a complete conversation without a generation
        header. It never flattens and re-encodes a native renderer's segments.
        Offsets are absent if the exact encoding cannot be mapped to flat text.

        Args:
            messages: JSON-serializable message dictionaries in conversation order.
            tools: Optional list of tool definitions passed to the renderer.
            template_kwargs: Optional renderer settings. These cannot override
                messages, tools, add_generation_prompt, or continue_final_message.
            now: Optional RFC3339 timestamp string for clock-dependent templates;
                None leaves the clock selection to SMG.

        Returns:
            A dictionary with text (the full rendered string), input_ids (exact
            token IDs), offsets (per-token UTF-8 byte ranges, or None), and
            segmentwise (whether a native deferred encoding job produced the IDs).
            When segmentwise is True, offsets is always None and re-encoding the
            flat text is not guaranteed to reproduce the returned IDs.
        """
        text, ids, offsets, segmentwise = self._inner.render_json(
            json.dumps(
                {
                    "messages": messages,
                    "tools": tools,
                    "template_kwargs": template_kwargs or {},
                    "now": now,
                },
                ensure_ascii=False,
            )
        )
        return {"text": text, "input_ids": ids, "offsets": offsets, "segmentwise": segmentwise}

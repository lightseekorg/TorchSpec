"""Independent K3 supervision oracle using checkpoint segment provenance.

Never calls a TorchSpec renderer/matcher or scans token IDs for headers. The
checkpoint's assistant-body callback labels segment objects before encoding.
The outer closing tag belongs to supervision; outer headers and EOM do not.
Only use with the hash-pinned checkpoint verified by k3_tokenizer_contract.
"""

import sys
from dataclasses import dataclass
from unittest.mock import patch


@dataclass
class MaskOracle:
    input_ids: list[int]
    owners: list[int | None]
    assistant_count: int

    def mask(self, *, last_turn_only=False, limit=None):
        return [
            int(owner is not None and (not last_turn_only or owner == self.assistant_count - 1))
            for owner in self.owners[:limit]
        ]


def checkpoint_mask_oracle(tokenizer, messages, tools=None, generation_config=None):
    """Observe semantic rendering callbacks, then encode each labelled segment.

    Object identity, rather than text/value equality, distinguishes literal
    control strings and repeated identical user/assistant/tool content. Keeping
    the original segments alive also prevents Python object-ID reuse.
    """
    build = tokenizer.apply_chat_template.__func__.__globals__["build_chat_segments"]
    module = sys.modules[build.__module__]
    render_assistant = module._render_assistant_segments
    close_tag = module._close_tag
    bodies = []
    pending = None

    def assistant(*args, **kwargs):
        nonlocal pending
        assert pending is None, "Unclosed checkpoint assistant body"
        segments = render_assistant(*args, **kwargs)
        pending = len(bodies)
        bodies.append(list(segments))
        return segments

    def close(tag):
        nonlocal pending
        segments = close_tag(tag)
        if tag == "message" and pending is not None:
            bodies[pending].extend(segments)
            pending = None
        return segments

    kwargs = dict(tokenize=True, add_generation_prompt=False, preserve_thinking=True)
    if generation_config is not None:
        kwargs.update(generation_config)
    # Capture the exact build invoked by the public API, including its default
    # effort and tool-result normalization, without reimplementing those rules.
    captured = []

    def capture(*args, **kwargs):
        segments = build(*args, **kwargs)
        captured.extend(segments)
        return segments

    with (
        patch.object(module, "_render_assistant_segments", assistant),
        patch.object(module, "_close_tag", close),
        patch.dict(tokenizer.apply_chat_template.__func__.__globals__, build_chat_segments=capture),
    ):
        public_ids = tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
    assert pending is None
    ownership = {id(segment): turn for turn, body in enumerate(bodies) for segment in body}
    ids, owners = [], []
    for segment in captured:
        encoded = tokenizer._encode_text_piece(
            segment.text, allow_special_tokens=segment.allow_special
        )
        ids.extend(encoded)
        owners.extend([ownership.get(id(segment))] * len(encoded))
    assert ids == public_ids, "Oracle segment encoding differs from checkpoint public API"
    assert len(ids) == len(owners)
    return MaskOracle(ids, owners, len(bodies))

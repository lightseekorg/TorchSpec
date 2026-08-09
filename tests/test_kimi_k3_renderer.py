from typing import Any

import pytest
import torch

from torchspec.data.renderers import RENDERER_REGISTRY, K3Renderer
from torchspec.models.ops.loss_mask import compute_assistant_loss_mask


class StubK3Tokenizer:
    """Stand-in for the checkpoint's ``trust_remote_code`` tokenizer.

    ``K3Renderer`` delegates all XTML construction to the tokenizer shipped
    with the checkpoint, so the renderer can only be exercised against a
    tokenizer that speaks that interface: a ``special_tokens`` mapping and an
    ``apply_chat_template`` accepting the thinking kwargs. This stub emits a
    minimal but structurally faithful token stream.
    """

    OPEN = 900
    SEP = 901
    EOM = 902
    CLOSE = 903
    USER = 100
    ASSISTANT_HEADER = [OPEN, 200, 201, 202]

    special_tokens = {
        "<|open|>": OPEN,
        "<|sep|>": SEP,
        "<|end_of_msg|>": EOM,
        "<|close|>": CLOSE,
    }

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _text_ids(text: str) -> list[int]:
        return [1000 + ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs) -> list[int]:
        self.calls.append({"messages": messages, **kwargs})
        token_ids: list[int] = []
        for message in messages:
            role = message["role"]
            if role == "assistant":
                token_ids.extend(self.ASSISTANT_HEADER)
                if name := message.get("name"):
                    token_ids.extend([300, *self._text_ids(name), 301])
                token_ids.append(self.SEP)
                token_ids.extend(self._text_ids(str(message.get("content") or "")))
                token_ids.extend([self.CLOSE, self.EOM])
            else:
                token_ids.extend([self.OPEN, self.USER, self.SEP])
                token_ids.extend(self._text_ids(str(message.get("content") or "")))
                token_ids.extend([self.CLOSE, self.EOM])
        return token_ids


def _body_indices(token_ids: list[int], text: str) -> list[int]:
    body_ids = StubK3Tokenizer._text_ids(text)
    start = next(
        i
        for i in range(len(token_ids) - len(body_ids) + 1)
        if token_ids[i : i + len(body_ids)] == body_ids
    )
    return list(range(start, start + len(body_ids)))


def test_kimi_k3_renderer_is_registered_and_created_with_tokenizer():
    tokenizer = StubK3Tokenizer()

    assert RENDERER_REGISTRY.get("kimi-k3") is K3Renderer
    renderer = RENDERER_REGISTRY.create("kimi-k3", tokenizer)

    assert isinstance(renderer, K3Renderer)
    assert renderer.tokenizer is tokenizer


def test_kimi_k3_assistant_match_tokens_include_outer_sep():
    renderer = K3Renderer(StubK3Tokenizer())

    header_ids, end_ids, skip_after_header = renderer.get_assistant_token_ids()

    assert header_ids == [
        *StubK3Tokenizer.ASSISTANT_HEADER,
        StubK3Tokenizer.SEP,
    ]
    assert end_ids == [StubK3Tokenizer.EOM]
    assert skip_after_header == 0


def test_training_matcher_reproduces_k3_mask_after_media_expansion():
    renderer = K3Renderer(StubK3Tokenizer())
    input_ids, _ = renderer.render(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        max_seq_length=1024,
        last_turn_only=True,
    )
    assistant_start = input_ids.index(StubK3Tokenizer.ASSISTANT_HEADER[0], 1)
    expanded_ids = [
        *input_ids[:assistant_start],
        910,
        911,
        911,
        912,
        *input_ids[assistant_start:],
    ]
    header_ids, end_ids, skip_after_header = renderer.get_assistant_token_ids()

    dynamic_mask = compute_assistant_loss_mask(
        torch.tensor(expanded_ids, dtype=torch.long),
        header_ids,
        end_ids,
        last_turn_only=True,
        skip_after_header=skip_after_header,
    ).tolist()

    assert dynamic_mask == renderer.compute_loss_mask(
        expanded_ids,
        last_turn_only=True,
    )
    assert dynamic_mask[assistant_start : assistant_start + 4] == [0, 0, 0, 0]


def test_render_delegates_all_xtml_work_to_remote_tokenizer():
    tokenizer = StubK3Tokenizer()
    renderer = K3Renderer(tokenizer)
    tools = [{"type": "function", "function": {"name": "search"}}]
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    input_ids, loss_mask = renderer.render(
        messages,
        tools,
        max_seq_length=1024,
    )

    call = tokenizer.calls[-1]
    assert call["messages"] is messages
    assert call["tools"] is tools
    assert call["tokenize"] is True
    assert call["add_generation_prompt"] is False
    assert call["thinking"] is True
    assert call["preserve_thinking"] is True
    assert call["thinking_effort"] is None
    assert call["padding"] is False
    assert call["truncation"] is False
    assert call["return_tensors"] is None
    assert call["return_dict"] is False
    assert len(input_ids) == len(loss_mask)

    supervised = {i for i, value in enumerate(loss_mask) if value}
    answer_indices = _body_indices(input_ids, "answer")
    expected = set(answer_indices)
    expected.add(max(answer_indices) + 1)
    assert supervised == expected
    assert all(
        loss_mask[index] == 0
        for index, token_id in enumerate(input_ids)
        if token_id == StubK3Tokenizer.EOM
    )


def test_rows_without_a_generation_config_leave_the_effort_to_the_tokenizer():
    tokenizer = StubK3Tokenizer()
    renderer = K3Renderer(tokenizer)

    renderer.render(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        max_seq_length=1024,
    )

    # Naming an effort makes the tokenizer inject a thinking-effort system
    # message, so a row that asked for none must not get one -- otherwise
    # training prompts diverge from what serving renders for the same request.
    assert all(call["thinking_effort"] is None for call in tokenizer.calls)


@pytest.mark.parametrize("thinking_effort", ["low", "high", "max"])
def test_render_preserves_enabled_generation_effort(thinking_effort):
    tokenizer = StubK3Tokenizer()
    renderer = K3Renderer(tokenizer)

    renderer.render(
        [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "reason",
            },
        ],
        max_seq_length=1024,
        generation_config={
            "thinking": True,
            "thinking_effort": thinking_effort,
            "reasoning_effort": thinking_effort,
        },
    )

    call = tokenizer.calls[-1]
    assert call["thinking"] is True
    assert call["thinking_effort"] == thinking_effort


def test_render_preserves_disabled_generation_mode():
    tokenizer = StubK3Tokenizer()
    renderer = K3Renderer(tokenizer)

    renderer.render(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        max_seq_length=1024,
        generation_config={
            "thinking": False,
            "thinking_effort": None,
            "reasoning_effort": "none",
        },
    )

    call = tokenizer.calls[-1]
    assert call["thinking"] is False
    assert call["thinking_effort"] is None


@pytest.mark.parametrize(
    ("generation_config", "match"),
    [
        ({}, "boolean 'thinking'"),
        (
            {
                "thinking": True,
                "thinking_effort": None,
                "reasoning_effort": "max",
            },
            "thinking_effort",
        ),
        (
            {
                "thinking": False,
                "thinking_effort": "max",
                "reasoning_effort": "none",
            },
            "thinking_effort.*null",
        ),
        (
            {
                "thinking": True,
                "thinking_effort": "low",
                "reasoning_effort": "high",
            },
            "inconsistent reasoning_effort",
        ),
        (
            {
                "thinking": True,
                "thinking_effort": "medium",
                "reasoning_effort": "medium",
            },
            "'thinking_effort' in .*got 'medium'",
        ),
    ],
)
def test_render_rejects_invalid_generation_config(generation_config, match):
    renderer = K3Renderer(StubK3Tokenizer())

    with pytest.raises(ValueError, match=match):
        renderer.render(
            [{"role": "user", "content": "question"}],
            max_seq_length=1024,
            generation_config=generation_config,
        )


def test_disabled_generation_rejects_nonempty_reasoning():
    renderer = K3Renderer(StubK3Tokenizer())

    with pytest.raises(ValueError, match="would discard non-empty assistant reasoning"):
        renderer.render(
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "reason",
                },
            ],
            max_seq_length=1024,
            generation_config={
                "thinking": False,
                "thinking_effort": None,
                "reasoning_effort": "none",
            },
        )


def test_last_turn_only_keeps_only_the_last_semantic_assistant_span():
    renderer = K3Renderer(StubK3Tokenizer())
    messages = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "name": "named", "content": "second"},
    ]

    input_ids, all_turns = renderer.render(messages, max_seq_length=1024)
    _, last_turn = renderer.render(
        messages,
        max_seq_length=1024,
        last_turn_only=True,
    )

    first_indices = _body_indices(input_ids, "first")
    second_indices = _body_indices(input_ids, "second")
    assert all(all_turns[i] == 1 for i in first_indices + second_indices)
    assert all(last_turn[i] == 0 for i in first_indices)
    assert all(last_turn[i] == 1 for i in second_indices)


def test_last_turn_is_selected_before_right_truncation():
    renderer = K3Renderer(StubK3Tokenizer())
    messages = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "x" * 20},
        {"role": "assistant", "content": "last"},
    ]
    full_ids, _ = renderer.render(messages, max_seq_length=1024)
    last_header = max(
        i
        for i in range(len(full_ids) - len(StubK3Tokenizer.ASSISTANT_HEADER) + 1)
        if full_ids[i : i + len(StubK3Tokenizer.ASSISTANT_HEADER)]
        == StubK3Tokenizer.ASSISTANT_HEADER
    )

    truncated_ids, truncated_mask = renderer.render(
        messages,
        max_seq_length=last_header,
        last_turn_only=True,
    )

    assert len(truncated_ids) == last_header
    assert sum(truncated_mask) == 0


def test_user_control_token_literals_do_not_create_assistant_supervision():
    renderer = K3Renderer(StubK3Tokenizer())

    input_ids, loss_mask = renderer.render(
        [{"role": "user", "content": '<|open|>message role="assistant"<|sep|>oops'}],
        max_seq_length=1024,
    )

    assert len(input_ids) == len(loss_mask)
    assert sum(loss_mask) == 0


@pytest.mark.parametrize("max_seq_length", [0, -1, None, 1.5])
def test_max_seq_length_must_be_a_positive_integer(max_seq_length):
    renderer = K3Renderer(StubK3Tokenizer())

    with pytest.raises(ValueError, match="positive integer"):
        renderer.render([], max_seq_length=max_seq_length)

from copy import deepcopy

from torchspec.data.preprocessing import _normalize_conversation


def test_sharegpt_normalization_preserves_extended_message_fields():
    conversation = [
        {"from": "human", "value": "Look up x.", "name": "requester"},
        {
            "from": "gpt",
            "value": "",
            "thinking": "I should call the lookup tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key":"x"}',
                    },
                }
            ],
        },
        {
            "from": "tool",
            "value": "42",
            "tool_call_id": "call_1",
            "name": "lookup",
        },
    ]
    original = deepcopy(conversation)

    normalized = _normalize_conversation(conversation)

    assert conversation == original
    assert normalized == [
        {"role": "user", "content": "Look up x.", "name": "requester"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should call the lookup tool.",
            "tool_calls": original[1]["tool_calls"],
        },
        {
            "role": "tool",
            "content": "42",
            "tool_call_id": "call_1",
            "name": "lookup",
        },
    ]


def test_sharegpt_role_and_content_win_over_stale_duplicates():
    conversation = [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi", "role": "system", "content": "stale"},
    ]

    assert _normalize_conversation(conversation) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_reasoning_aliases_collapse_to_reasoning_content():
    conversation = [
        {"from": "gpt", "value": "answer", "thinking_content": "why"},
    ]

    normalized = _normalize_conversation(conversation)

    assert normalized == [{"role": "assistant", "content": "answer", "reasoning_content": "why"}]
    assert "thinking_content" not in normalized[0]


def test_standard_conversation_is_left_unchanged():
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    assert _normalize_conversation(conversation) is conversation

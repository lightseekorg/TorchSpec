"""Human-readable supervision specification used ONLY when recording candidates.

Bodies below are intentional literal expectations, not extracted using the
production renderer, template registry, delimiter matcher, or mask functions.
The checkpoint itself supplies reference rendering and tokenization.
"""

PROFILES = {
    "qwen3": {
        "model": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "template": "qwen3-thinking",
        "default_encode_path": "native",
        "end": "<|im_end|>",
        "header": "<|im_start|>assistant\n",
    },
    "llama3": {
        "model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "revision": "53346005fb0ef11d3b6a83b12c895cca40156b6c",
        "template": "llama3",
        "default_encode_path": "tokenizers",
        "end": "<|eot_id|>",
        "header": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "minimax25": {
        "model": "MiniMaxAI/MiniMax-M2.5",
        "revision": "2515a3ea0afec1dd22d0c7794b3a91557804d195",
        "template": "minimax-m2",
        "default_encode_path": "native",
        "end": "[e~[",
        "header": "]~b]ai\n",
    },
}

REFERENCE_VERSIONS = {"transformers": "5.20.2", "tokenizers": "0.23.3"}
REQUIRED_CASES = ["ascii", "unicode", "whitespace", "multi_turn", "empty", "reasoning", "tools"]


def cases(profile):
    """Messages and explicitly specified complete assistant bodies, in order."""
    prefix = [{"role": "system", "content": "System."}, {"role": "user", "content": "Question?"}]
    think = "<think>\n\n</think>\n\n" if profile == "qwen3" else ""
    for name, content in [
        ("ascii", "Answer one."),
        ("unicode", "你好 café 🧠 e\u0301 \u200d\tend"),
        ("whitespace", "\n\nAnswer one.\n"),
        ("empty", ""),
    ]:
        body = content
        if profile == "llama3":
            body = content.strip()  # This pinned Llama template trims content.
        elif profile == "qwen3":
            body = think + content.lstrip("\n")
        yield {
            "name": name,
            "messages": prefix + [{"role": "assistant", "content": content}],
            "bodies": [body],
        }

    yield {
        "name": "multi_turn",
        "messages": prefix
        + [
            {"role": "assistant", "content": "Answer one."},
            {"role": "user", "content": "Second question?"},
            {"role": "assistant", "content": "Answer two."},
        ],
        "bodies": ["Answer one.", think + "Answer two."],
    }
    yield {
        "name": "reasoning",
        "messages": prefix
        + [{"role": "assistant", "reasoning_content": "Reason one.", "content": "Answer one."}],
        "bodies": ["<think>\nReason one.\n</think>\n\nAnswer one."],
        **({"reject": "dropped supervised reasoning"} if profile == "llama3" else {}),
    }
    tool_body = (
        '<tool_call>\n{"name": "weather", "arguments": {"city": "北京"}}\n</tool_call>'
        if profile == "qwen3"
        else '\n<minimax:tool_call>\n<invoke name="weather">\n<parameter name="city">北京</parameter>\n</invoke>\n</minimax:tool_call>'
    )
    yield {
        "name": "tools",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "messages": prefix
        + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": {"city": "北京"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call1", "name": "weather", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ],
        "bodies": [tool_body, think + "It is sunny."],
        **({"reject": "dropped a tool call"} if profile == "llama3" else {}),
    }

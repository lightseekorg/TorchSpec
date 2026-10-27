"""Parity against the pinned K3 tokenizer, including its absent-effort default."""

import argparse
import hashlib
import importlib.metadata
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from torchspec.data.renderers.kimi_k3 import K3Renderer
from torchspec.data.renderers.smg_k3 import SmgK3Renderer

ASSETS = {
    "config.json": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
    "tokenization_kimi.py": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    "encoding_k3.py": "b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333",
    "tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
}


def verify_assets(path):
    for name, expected in ASSETS.items():
        actual = hashlib.sha256((Path(path) / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unqualified K3 checkpoint asset: {name}: {actual}")


def cases():
    def pair(user, answer, **extra):
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer, **extra},
        ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find a value",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    yield "ascii", pair("Question?", "Answer."), None
    yield "unicode", pair("中文 café e\u0301 👩‍💻", "العربية 日本語 🧑🏽‍🔬"), None
    yield "whitespace", pair(" \n\tquery\r\n", "\n\n answer\t \r\n"), None
    yield "empty", pair("Question", ""), None
    yield (
        "literal_controls",
        pair("<|open|><|close|><|sep|><|end_of_msg|>", "literal <|open|> is data"),
        None,
    )
    yield "reasoning", pair("Think", "Answer", reasoning_content="Reason 中文"), None
    yield "reasoning_alias", pair("Think", "Answer", reasoning="Alternate reason"), None
    yield "multi_turn", pair("first", "first answer") + pair("second", "second answer"), None
    yield "named_assistant", pair("Hi", "Answer", name="named café"), None
    yield "system", [{"role": "system", "content": "Follow the user."}] + pair("Hi", "Answer"), None
    for arguments in [{"query": "中文 <|open|>"}, '{"query": "中文 <|open|>"}']:
        call = {
            "id": "call_a",
            "type": "function",
            "function": {"name": "lookup", "arguments": arguments},
        }
        messages = pair("Find something", "", tool_calls=[call]) + [
            {"role": "tool", "tool_call_id": "call_a", "content": "Found"},
            {"role": "assistant", "content": "Here it is"},
        ]
        yield "tools_" + type(arguments).__name__, messages, tools
    yield "unused_tools", pair("Hi", "Hello"), tools


def check_dataset(path, legacy):
    from torchspec.data.dataset import load_conversation_dataset
    from torchspec.data.preprocessing import _normalize_conversation
    from torchspec.data.utils import unpack_loss_mask

    # Exercise the real loader, warmed-parent spawn and cache reload rather than
    # inferring integration from renderer-only equality.
    with tempfile.TemporaryDirectory(prefix="k3-tokenizer-contract-") as tmp:
        root = Path(tmp)
        selected = list(cases())
        raw = root / "samples.jsonl"
        rows = [
            {"id": name, "messages": messages, "tools": tools} for name, messages, tools in selected
        ]
        raw.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
        for workers in (1, 2):
            args = SimpleNamespace(
                renderer="kimi-k3",
                target_model_path=str(Path(path).resolve()),
                train_data_path=str(raw),
                prompt_key="messages",
                max_seq_length=512,
                num_proc=workers,
                cache_dir=str(root / f"cache-{workers}"),
                last_turn_loss_only=True,
            )
            actual = load_conversation_dataset(args)
            cached = load_conversation_dataset(args)
            assert len(actual) == len(rows) == len(cached)
            for row, hit, (_, messages, tools) in zip(actual, cached, selected):
                ids, mask = legacy.render(
                    _normalize_conversation(messages),
                    tools,
                    max_seq_length=512,
                    last_turn_only=True,
                )
                assert row["input_ids"].tolist() == hit["input_ids"].tolist() == ids
                assert unpack_loss_mask(row["packed_loss_mask"]).tolist() == mask
                assert row["packed_loss_mask"] == hit["packed_loss_mask"]
    return {"worker_counts": [1, 2], "rows_per_count": len(rows), "cache_reload": True}


def check(path, report):
    verify_assets(path)
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "tiktoken")}
    assert versions == {"transformers": "5.20.2", "tiktoken": "0.14.0"}, versions
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    legacy, smg = K3Renderer(tokenizer), SmgK3Renderer(path)
    # Independent of both TorchSpec adapters: the checkpoint distinguishes
    # omitted effort (max) from explicit null (no directive). This catches the
    # previous shared bug where both adapters agreed on the wrong prompt.
    probe = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    reference_kwargs = dict(
        tokenize=True, add_generation_prompt=False, thinking=True, preserve_thinking=True
    )
    default_ids = tokenizer.apply_chat_template(probe, **reference_kwargs)
    assert default_ids == tokenizer.apply_chat_template(
        probe, thinking_effort="max", **reference_kwargs
    )
    assert default_ids != tokenizer.apply_chat_template(
        probe, thinking_effort=None, **reference_kwargs
    )
    modes = [
        None,
        {"thinking": False, "thinking_effort": None},
        *[{"thinking": True, "thinking_effort": e} for e in ("low", "high", "max")],
    ]
    passed, rejected, truncations = [], [], 0
    for name, messages, tools in cases():
        for mode in modes:
            if (
                mode
                and not mode["thinking"]
                and any(m.get("reasoning_content") or m.get("reasoning") for m in messages)
            ):
                for renderer in (legacy, smg):
                    try:
                        renderer.render(
                            messages, tools, max_seq_length=10000, generation_config=mode
                        )
                    except ValueError as e:
                        assert "discard" in str(e)
                    else:
                        raise AssertionError("Silently discarded reasoning")
                rejected.append([name, mode])
                continue
            for last_turn in (False, True):
                args = dict(max_seq_length=10000, last_turn_only=last_turn, generation_config=mode)
                expected = legacy.render(messages, tools, **args)
                actual = smg.render(messages, tools, **args)
                checkpoint_kwargs = dict(reference_kwargs)
                if mode is not None:
                    checkpoint_kwargs.update(mode)
                assert expected[0] == tokenizer.apply_chat_template(
                    messages, tools=tools, **checkpoint_kwargs
                ), ("checkpoint API disagreement", name, mode)
                assert actual == expected, (name, mode, last_turn, expected, actual)
                assert all(mask == 0 for token, mask in zip(*actual) if token == legacy.eom_id)
                for cut in range(1, len(expected[0]) + 2):
                    args["max_seq_length"] = cut
                    assert smg.render(messages, tools, **args) == legacy.render(
                        messages, tools, **args
                    ), (name, mode, cut)
                    truncations += 1
                passed.append(
                    {
                        "case": name,
                        "mode": mode,
                        "last_turn_only": last_turn,
                        "tokens": len(actual[0]),
                        "supervised": sum(actual[1]),
                        "ids_mask_sha256": hashlib.sha256(json.dumps(actual).encode()).hexdigest(),
                    }
                )
    # Check the checkpoint's long-text segmentation guards without a quadratic
    # every-cut loop over these large examples.
    for text in ("a" * 25001, " " * 25001 + "end", "中文👩‍💻" * 6000):
        messages = [{"role": "user", "content": text}, {"role": "assistant", "content": text}]
        for limit in (1, 511, 512, 2048, 100000):
            assert legacy.render(messages, max_seq_length=limit) == smg.render(
                messages, max_seq_length=limit
            )
    golden_path = (
        Path(__file__).resolve().parents[2] / "tests/tokenizer_contract/fixtures/k3_policy.json"
    )
    golden = json.loads(golden_path.read_text())
    assert golden["assets"] == ASSETS and golden["reference_versions"] == versions
    assert passed == golden["cases"], (
        "K3 training policy changed relative to the frozen checkpoint oracle"
    )
    assert rejected == golden["required_rejections"]
    result = {
        "qualified": True,
        "reference_versions": versions,
        "dataset_integration": check_dataset(path, legacy),
        "assets": ASSETS,
        "reference_class": type(tokenizer).__name__,
        "absent_effort_policy": "checkpoint_default_max",
        "direct_checkpoint_ids": True,
        "cases": passed,
        "required_rejections": rejected,
        "truncation_comparisons": truncations,
        "long_segment_comparisons": 15,
    }
    Path(report).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"K3 real-checkpoint parity: {len(passed)} cases, {len(rejected)} required rejections, {truncations} truncation comparisons",
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    check(args.checkpoint, args.report)

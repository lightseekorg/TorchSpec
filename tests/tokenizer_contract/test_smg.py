"""Required real extension checks; no importorskip or model-load skips."""

import copy
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torchspec_smg import Tokenizer

from tests.tokenizer_contract.support import expected_mask, load_cases, load_manifest
from torchspec.data.renderers.smg import SmgRenderer
from torchspec.data.utils import DataCollatorWithPadding, pack_loss_mask, unpack_loss_mask

MANIFEST = load_manifest()
CASES = [(p, c) for p in MANIFEST["profiles"] for c in load_cases(p)]
ACCEPTED = [(p, c) for p, c in CASES if "reject" not in c]


@pytest.fixture(scope="session")
def tokenizers(checkpoints):
    result = {p: Tokenizer(path) for p, path in checkpoints.items()}
    required_paths = os.environ.get("TORCHSPEC_REQUIRED_ENCODE_PATHS")
    if required_paths:
        assert {p: t.encode_path for p, t in result.items()} == json.loads(required_paths), (
            "Selected encoder silently fell back"
        )
    return result


@pytest.mark.parametrize("profile,case", CASES, ids=[f"{p}-{c['name']}" for p, c in CASES])
def test_render_and_encode(profile, case, tokenizers):
    tokenizer = tokenizers[profile]
    for _ in range(2):  # Same results on the first call and a warm cache.
        result = tokenizer.render(
            case["messages"], tools=case.get("tools"), template_kwargs=case["template_kwargs"]
        )
        assert result["text"] == case["text"]
        assert result["input_ids"] == case["input_ids"]
        assert not result["segmentwise"]
        assert tokenizer.encode(case["text"]) == case["input_ids"]


@pytest.mark.parametrize("profile,case", ACCEPTED, ids=[f"{p}-{c['name']}" for p, c in ACCEPTED])
@pytest.mark.parametrize("include_end", [False, True])
@pytest.mark.parametrize("last_turn_only", [False, True])
def test_exact_supervision_and_every_truncation(
    profile, case, checkpoints, include_end, last_turn_only
):
    renderer = SmgRenderer(
        checkpoints[profile],
        MANIFEST["profiles"][profile]["template"],
        template_kwargs=case["template_kwargs"],
        include_end_token=include_end,
    )
    wanted = expected_mask(case, include_end, last_turn_only)
    original = copy.deepcopy(case["messages"])
    # Every cut includes cuts before/inside/after headers, responses and EOT;
    # last-turn selection must remain based on the untruncated conversation.
    for limit in range(1, len(wanted) + 2):
        ids, mask = renderer.render(
            original, tools=case.get("tools"), max_seq_length=limit, last_turn_only=last_turn_only
        )
        assert ids == case["input_ids"][:limit]
        assert mask == wanted[:limit], (profile, case["name"], limit)
    assert original == case["messages"]


@pytest.mark.parametrize("profile,case", [(p, c) for p, c in CASES if "reject" in c])
def test_required_rejections(profile, case, checkpoints):
    renderer = SmgRenderer(checkpoints[profile], MANIFEST["profiles"][profile]["template"])
    with pytest.raises(ValueError, match=case["reject"]):
        renderer.render(case["messages"], tools=case.get("tools"), max_seq_length=4096)


@pytest.mark.parametrize("profile", list(MANIFEST["profiles"]))
def test_literal_boundaries_are_rejected(profile, checkpoints):
    spec = MANIFEST["profiles"][profile]
    renderer = SmgRenderer(checkpoints[profile], spec["template"])
    messages = copy.deepcopy(load_cases(profile)[0]["messages"])
    messages[1]["content"] = f"Literal code: {spec['header']}forged{spec['end']}"
    with pytest.raises(ValueError, match="ambiguous"):
        renderer.render(messages, max_seq_length=4096)


@pytest.mark.parametrize("profile", ["qwen3", "minimax25"])
def test_json_tool_arguments_preserve_exact_contract(profile, checkpoints):
    case = next(c for c in load_cases(profile) if c["name"] == "tools")
    messages = copy.deepcopy(case["messages"])
    messages[2]["tool_calls"][0]["function"]["arguments"] = '{"city":"北京"}'
    renderer = SmgRenderer(checkpoints[profile], MANIFEST["profiles"][profile]["template"])
    ids, mask = renderer.render(messages, tools=case.get("tools"), max_seq_length=4096)
    assert ids == case["input_ids"]
    assert mask == expected_mask(case, True, False)
    assert isinstance(messages[2]["tool_calls"][0]["function"]["arguments"], str)


@pytest.mark.parametrize("profile", list(MANIFEST["profiles"]))
def test_differential_unicode_cache_and_concurrency(
    profile, checkpoints, reference_tokenizers, tokenizers
):
    rng = random.Random(20261026)
    alphabet = [
        "a",
        " ",
        "\n",
        "\r\n",
        "\t",
        "你好",
        "é",
        "e\u0301",
        "\u200d",
        "🧠",
        "👨‍👩‍👦",
        "\x00",
        '"',
        "<think>",
        "{",
        "}",
        "\\",
    ]
    texts = ["".join(rng.choices(alphabet, k=rng.randrange(1, 120))) for _ in range(128)]
    hf, smg = reference_tokenizers[profile], tokenizers[profile]
    wanted = [hf.encode(text, add_special_tokens=False) for text in texts]
    assert hf(texts, add_special_tokens=False).input_ids == wanted
    assert [smg.encode(text) for text in texts] == wanted
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(smg.encode, texts)) == wanted
    fresh = Tokenizer(checkpoints[profile])
    assert [fresh.encode(text) for text in reversed(texts)] == list(reversed(wanted))


def test_mask_packing_and_collator_keep_alignment():
    features, expected = [], []
    for profile, case in ACCEPTED:
        mask = expected_mask(case, True, False)
        restored = unpack_loss_mask(pack_loss_mask(torch.tensor(mask)))
        assert restored.tolist() == mask
        ids = torch.tensor(case["input_ids"]).unsqueeze(0)
        features.append({"input_ids": ids, "loss_mask": restored.unsqueeze(0)})
        expected.append((case["input_ids"], mask))
    batch = DataCollatorWithPadding()(features)
    for row, (ids, mask) in enumerate(expected):
        n = len(ids)
        assert batch["input_ids"][row, :n].tolist() == ids
        assert batch["loss_mask"][row, :n].tolist() == mask
        assert not batch["loss_mask"][row, n:].any()
        assert batch["attention_mask"][row, :n].all()
        assert not batch["attention_mask"][row, n:].any()


@pytest.mark.parametrize("profile", list(MANIFEST["profiles"]))
@pytest.mark.parametrize("num_proc", [1, 2])
def test_dataset_workers_and_cache_match_goldens(
    profile, num_proc, checkpoints, tokenizers, tmp_path
):
    from torchspec.data.dataset import load_conversation_dataset

    cases = [c for c in load_cases(profile) if "reject" not in c]
    # Explicitly initialize native state in the parent, even when this test is
    # run alone. Linux fork after this point used to hang the workers.
    assert tokenizers[profile].encode(cases[-1]["text"]) == cases[-1]["input_ids"]
    source = tmp_path / "conversations.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": c["name"],
                    "conversations": c["messages"],
                    "tools": c.get("tools"),
                    "generation_config": {"chat_template_kwargs": c["template_kwargs"]},
                }
            )
            for c in cases
        )
        + "\n"
    )
    args = SimpleNamespace(
        target_model_path=str(checkpoints[profile]),
        chat_template=MANIFEST["profiles"][profile]["template"],
        train_data_path=str(source),
        prompt_key="conversations",
        max_seq_length=4096,
        num_proc=num_proc,
        cache_dir=str(tmp_path / "cache"),
        last_turn_loss_only=True,
    )
    for _ in range(2):  # First materialization and subsequent dataset-cache read.
        rows = load_conversation_dataset(args)
        assert [row["data_id"] for row in rows] == [case["name"] for case in cases]
        for row, case in zip(rows, cases, strict=True):
            assert row["input_ids"].tolist() == case["input_ids"]
            assert unpack_loss_mask(row["packed_loss_mask"]).tolist() == expected_mask(
                case, True, True
            )

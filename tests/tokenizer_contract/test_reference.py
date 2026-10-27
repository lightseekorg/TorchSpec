"""Immutable checkpoint rendering/IDs and reviewed training annotations."""

import hashlib
import json

import pytest

from tests.tokenizer_contract.specification import PROFILES, REQUIRED_CASES
from tests.tokenizer_contract.support import (
    FIXTURES,
    expected_mask,
    load_cases,
    load_manifest,
    resolve_checkpoint,
)

CASES = [(profile, case) for profile in load_manifest()["profiles"] for case in load_cases(profile)]


@pytest.mark.parametrize("profile,case", CASES, ids=[f"{p}-{c['name']}" for p, c in CASES])
def test_checkpoint_reference(profile, case, reference_tokenizers):
    tokenizer = reference_tokenizers[profile]
    text = tokenizer.apply_chat_template(
        case["messages"],
        tools=case.get("tools"),
        tokenize=False,
        add_generation_prompt=False,
        **case["template_kwargs"],
    )
    assert text == case["text"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded.input_ids == case["input_ids"]
    for annotation, positions in zip(
        case.get("assistant_annotations", []), case.get("assistant_tokens", []), strict=True
    ):
        start, body_end, end = annotation["byte_span"]
        raw = text.encode()
        assert raw[start:body_end].decode() == annotation["body"]
        assert raw[body_end:end].decode() == annotation["end"]
        for key, stop in (("body", body_end), ("with_end", end)):
            # The golden positions are frozen. Recheck annotation consistency
            # against the independent HF offsets, never a production matcher.
            actual = []
            for i, (a, b) in enumerate(encoded.offset_mapping):
                lo, hi = len(text[:a].encode()), len(text[:b].encode())
                if start <= lo < hi <= stop:
                    actual.append(i)
            assert actual == positions[key]


def test_required_coverage_is_present():
    manifest = load_manifest()
    assert set(manifest["profiles"]) == set(PROFILES)
    for name, profile in manifest["profiles"].items():
        assert profile["required_cases"] == REQUIRED_CASES
        assert profile["revision"] == PROFILES[name]["revision"]
        assert len(profile["revision"]) == 40
        for case in load_cases(name):
            if "reject" in case:
                assert name == "llama3" and case["name"] in ("reasoning", "tools")
                continue
            assert len(case["assistant_annotations"]) == sum(
                m["role"] == "assistant" for m in case["messages"]
            )
            assert len(case["assistant_tokens"]) == len(case["assistant_annotations"])
            assert len(expected_mask(case, True, False)) == len(case["input_ids"])
            assert sum(expected_mask(case, True, False)) > 0
    assert {p.stem for p in FIXTURES.glob("*.json")} == {*PROFILES, "manifest", "k3_policy"}


def test_changed_checkpoint_is_rejected(tmp_path, monkeypatch):
    profile = load_manifest()["profiles"]["qwen3"]
    mapping = tmp_path / "paths.json"
    mapping.write_text(json.dumps({profile["model"]: str(tmp_path)}))
    # First expected asset deliberately has the wrong content.
    name = next(iter(profile["assets"]))
    (tmp_path / name).write_text("{}")
    assert hashlib.sha256(b"{}").hexdigest() != profile["assets"][name]
    monkeypatch.setenv("TORCHSPEC_TOKENIZER_PATHS", str(mapping))
    with pytest.raises(ValueError, match="artifact mismatch"):
        resolve_checkpoint("qwen3")

"""Fixture loading and checkpoint integrity checks. No production mask imports."""

import hashlib
import json
import os
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def load_manifest():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported tokenizer contract schema")
    return manifest


def load_cases(profile):
    entry = load_manifest()["profiles"][profile]
    path = FIXTURES / f"{profile}.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["fixture_sha256"]:
        raise ValueError(f"Fixture digest mismatch: {profile}; review fixture updates explicitly")
    cases = json.loads(path.read_text())
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names) or sorted(names) != sorted(entry["required_cases"]):
        raise ValueError(f"Missing, duplicated or unexpected fixture cases: {profile}")
    return cases


def resolve_checkpoint(profile):
    entry = load_manifest()["profiles"][profile]
    mapping = os.environ.get("TORCHSPEC_TOKENIZER_PATHS")
    if mapping:
        # An explicit mapping is exhaustive: never hide a missing local asset
        # by downloading another checkpoint or falling back to a model alias.
        path = Path(json.loads(Path(mapping).read_text())[entry["model"]])
    else:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                entry["model"], revision=entry["revision"], allow_patterns=list(entry["assets"])
            )
        )
    for name, digest in entry["assets"].items():
        actual = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Checkpoint artifact mismatch: {profile}/{name}")
    # Extra rendering assets can override the pinned config even when all
    # expected files match. Reject them, while allowing unrelated model weights.
    relevant = {
        p.relative_to(path).as_posix()
        for p in path.rglob("*")
        if p.is_file()
        and (
            p.suffix in (".jinja", ".model", ".tiktoken")
            or p.name
            in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "config.json",
                "vocab.json",
                "vocab.txt",
                "merges.txt",
            )
        )
    }
    if relevant != set(entry["assets"]):
        raise ValueError(
            f"Unexpected or missing tokenizer artifacts: {profile}: {relevant ^ set(entry['assets'])}"
        )
    return path


def expected_mask(case, include_end, last_turn_only, length=None):
    mask = [0] * len(case["input_ids"])
    turns = case["assistant_tokens"][-1:] if last_turn_only else case["assistant_tokens"]
    for turn in turns:
        for i in turn["with_end" if include_end else "body"]:
            mask[i] = 1
    return mask if length is None else mask[:length]

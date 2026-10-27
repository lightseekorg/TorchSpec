"""Record reference candidates, never overwrite the reviewed fixture directory.

Run from the repo root with PYTHONPATH=. and --paths mapping.json --output DIR.
Review the rendered text, byte annotations and token indices before copying the
candidate JSON files into tests/tokenizer_contract/fixtures/.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer

from tests.tokenizer_contract.specification import (
    PROFILES,
    REFERENCE_VERSIONS,
    REQUIRED_CASES,
    cases,
)
from tests.tokenizer_contract.support import FIXTURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output == FIXTURES.resolve() or FIXTURES.resolve() in output.parents:
        raise ValueError("Write candidates elsewhere; reviewed fixtures cannot be auto-blessed")
    output.mkdir(parents=True, exist_ok=False)
    for package, version in REFERENCE_VERSIONS.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Reference generator requires {package}=={version}")
    paths = json.loads(Path(args.paths).read_text())
    manifest = {
        "schema_version": 1,
        "reference_versions": REFERENCE_VERSIONS,
        "policy": {
            "mode": "complete_text_conversation",
            "header_loss": False,
            "reasoning_loss": True,
            "tool_call_loss": True,
            "boundary_straddling_token_loss": False,
            "last_turn_selection": "before_right_truncation",
            "end_policy": "end_marker_only_excluding_following_formatting_newline",
            "unsupported": [
                "K3 native segments",
                "multimodal",
                "generation/continuation",
                "literal role delimiters",
            ],
        },
        "profiles": {},
    }
    for name, profile in PROFILES.items():
        path = Path(paths[profile["model"]])
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        fixtures = []
        for spec in cases(name):
            spec = dict(spec)
            bodies = spec.pop("bodies")
            kwargs = {"preserve_thinking": True, "enable_thinking": True}
            text = tokenizer.apply_chat_template(
                spec["messages"],
                tools=spec.get("tools"),
                tokenize=False,
                add_generation_prompt=False,
                **kwargs,
            )
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            spec.update(template_kwargs=kwargs, text=text, input_ids=encoded.input_ids)
            if "reject" not in spec:
                annotations, tokens, cursor = [], [], 0
                for body in bodies:
                    # Exact authored body + its header/end must occur exactly
                    # once after the previous turn. No production delimiter scan.
                    literal = profile["header"] + body + profile["end"]
                    if text.count(literal, cursor) != 1:
                        raise ValueError(
                            f"{name}/{spec['name']}: authored body does not uniquely match {body!r}"
                        )
                    start = text.index(literal, cursor) + len(profile["header"])
                    body_end, end = start + len(body), start + len(body) + len(profile["end"])
                    annotations.append(
                        {
                            "body": body,
                            "end": profile["end"],
                            "byte_span": [len(text[:p].encode()) for p in (start, body_end, end)],
                        }
                    )
                    tokens.append(
                        {
                            "body": [
                                i
                                for i, (a, b) in enumerate(encoded.offset_mapping)
                                if start <= a < b <= body_end
                            ],
                            "with_end": [
                                i
                                for i, (a, b) in enumerate(encoded.offset_mapping)
                                if start <= a < b <= end
                            ],
                        }
                    )
                    cursor = end
                spec.update(assistant_annotations=annotations, assistant_tokens=tokens)
            fixtures.append(spec)
        dest = output / f"{name}.json"
        dest.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n")
        assets = {
            p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*"))
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
        manifest["profiles"][name] = {
            **profile,
            "assets": assets,
            "required_cases": REQUIRED_CASES,
            "fixture_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

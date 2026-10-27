"""Select a deterministic disjoint corpus prefix and require exact K3 parity."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from tools.qualification.k3_tokenizer_contract import verify_assets
from torchspec.data.preprocessing import _normalize_conversation
from torchspec.data.renderers.kimi_k3 import K3Renderer
from torchspec.data.renderers.smg_k3 import SmgK3Renderer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    verify_assets(a.model)
    r = K3Renderer(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True))
    s = SmgK3Renderer(a.model)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    seen = set()
    rows = []
    rejected = {}
    scanned = 0
    with open(a.source) as f:
        for line_no, line in enumerate(f):
            scanned += 1
            x = json.loads(line)
            messages = _normalize_conversation(x["conversations"])
            if any(
                m.get("content") is not None and not isinstance(m["content"], str) for m in messages
            ):
                rejected["nontext"] = rejected.get("nontext", 0) + 1
                continue
            user = json.dumps(
                [m for m in messages if m.get("role") == "user"], sort_keys=True, ensure_ascii=False
            )
            if user in seen:
                continue
            ids, mask = r.render(
                messages,
                x.get("tools"),
                max_seq_length=512,
                generation_config=x.get("generation_config"),
            )
            if sum(mask) < 32:
                rejected["short_supervision"] = rejected.get("short_supervision", 0) + 1
                continue
            assert (ids, mask) == s.render(
                messages,
                x.get("tools"),
                max_seq_length=512,
                generation_config=x.get("generation_config"),
            )
            seen.add(user)
            rows.append(
                {
                    "id": x.get("data_id", str(line_no)),
                    "messages": messages,
                    "tools": x.get("tools"),
                    "generation_config": x.get("generation_config"),
                    "input_ids": ids,
                    "loss_mask": mask,
                    "source_line": line_no,
                    "source_row_sha256": hashlib.sha256(line.encode()).hexdigest(),
                    "split": "train" if len(rows) < 256 else "eval",
                }
            )
            if len(rows) == 320:
                break
    assert len(rows) == 320
    (out / "samples.json").write_text(json.dumps(rows, ensure_ascii=False))
    (out / "preparation.json").write_text(
        json.dumps(
            {
                "source": a.source,
                "source_prefix_lines_scanned": scanned,
                "selected": len(rows),
                "rejections": rejected,
                "max_length": 512,
                "train_samples": 256,
                "eval_samples": 64,
                "parity": True,
                "samples_sha256": hashlib.sha256((out / "samples.json").read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    print("selected", len(rows), "scanned", scanned, "rejected", rejected, flush=True)


if __name__ == "__main__":
    main()

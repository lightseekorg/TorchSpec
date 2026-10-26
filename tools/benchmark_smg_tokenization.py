"""Compare exact IDs and warm CPU latency on pinned local tokenizers.

Run from the repository with PYTHONPATH=. and --models pointing to a JSON list
of {name, path, chat_template}. No weights or GPU are loaded. Results describe
synthetic warm-cache workloads, not training throughput.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer

from torchspec.data.renderers.smg import SmgRenderer


def timed(function, samples):
    for sample in samples[:2]:
        function(sample)
    times = []
    for sample in samples:
        start = time.perf_counter()
        function(sample)
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(times), "samples_ms": times}


class HFReference:
    """Same renderer/mask contract, with HF producing the text and offsets."""

    raw_tool_call_arguments = False

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def render(self, messages, *, tools=None, template_kwargs=None, now=None):
        if now is not None:
            raise ValueError("This benchmark does not cover clock-dependent templates")
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **(template_kwargs or {}),
        )
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        byte_positions = [0]
        for char in text:
            byte_positions.append(byte_positions[-1] + len(char.encode()))
        return {
            "text": text,
            "input_ids": encoded.input_ids,
            "segmentwise": False,
            "offsets": [(byte_positions[a], byte_positions[b]) for a, b in encoded.offset_mapping],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--gigatoken", action="store_true")
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("--rounds must be at least 3")
    import smg_tokenizer

    report = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "smg_features": smg_tokenizer.build_features,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("transformers", "tokenizers", "smg-tokenizer")
            },
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        },
        "notes": [
            "Warm single-caller synthetic inputs with repeated vocabulary and changing nonce.",
            "Raw encoders return Python lists, including the Gigatoken HF compatibility API.",
            "HF end-to-end uses the same supervision policy as SMG, not TorchSpec's legacy parser.",
            "SMG end-to-end includes reference offset computation for accelerated encoders.",
            "No GPU or training-throughput measurement.",
        ],
        "models": [],
    }
    for spec in json.loads(Path(args.models).read_text()):
        hf = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=False)
        smg = SmgRenderer(spec["path"], spec["chat_template"])
        reference = SmgRenderer(spec["path"], spec["chat_template"])
        reference.tokenizer = HFReference(hf)
        encoders = {
            "hf": lambda text: hf.encode(text, add_special_tokens=False),
            "smg": smg.tokenizer.encode,
        }
        if args.gigatoken:
            import gigatoken

            giga = gigatoken.Tokenizer(hf).as_hf()
            encoders["gigatoken_hf"] = lambda text: giga.encode(text, add_special_tokens=False)
            report["environment"]["versions"]["gigatoken"] = importlib.metadata.version("gigatoken")
        model = {**spec, "encode_path": smg.tokenizer.encode_path, "cases": []}
        for language, phrase in (
            ("english", "The tokenizer must preserve exact tokens, tools, and reasoning. "),
            ("cjk", "训练分词必须保留完整语义、工具调用与推理过程。"),
        ):
            for target in (2048, 50000):
                repeat = max(1, target // len(hf.encode(phrase, add_special_tokens=False)))
                samples = [f"Sample {i}: " + phrase * repeat for i in range(args.rounds)]
                rows = [
                    [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "A concise answer. 你好"},
                    ]
                    for text in samples
                ]
                raw_parity = {
                    name: all(list(encode(text)) == encoders["hf"](text) for text in samples)
                    for name, encode in encoders.items()
                }
                if not all(raw_parity.values()):
                    raise AssertionError(f"{spec['name']} raw ID mismatch: {raw_parity}")
                for messages in rows:
                    actual = smg.render(messages, max_seq_length=100000)
                    expected = reference.render(messages, max_seq_length=100000)
                    if actual != expected:
                        raise AssertionError(f"{spec['name']} full token/mask mismatch")
                result = {
                    "language": language,
                    "target_tokens": target,
                    "actual_tokens": len(encoders["hf"](samples[0])),
                    "raw_id_parity": raw_parity,
                    "full_id_and_mask_parity": True,
                    "raw_encode": {
                        name: timed(encode, samples) for name, encode in encoders.items()
                    },
                    "render_encode_mask": {
                        "hf": timed(lambda m: reference.render(m, max_seq_length=100000), rows),
                        "smg": timed(lambda m: smg.render(m, max_seq_length=100000), rows),
                    },
                }
                model["cases"].append(result)
                print(spec["name"], language, target, "parity PASS", flush=True)
        report["models"].append(model)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

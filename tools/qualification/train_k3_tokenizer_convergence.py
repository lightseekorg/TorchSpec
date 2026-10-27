"""Matched 1,000-step K3 legacy/SMG training on real cached K3 hidden states.

Requires exact IDs/masks and per-layer input-ID/hash checks before sharing target
states. The target checkpoint, dataset selection and extraction manifest remain
external artifacts; no synthetic targets or automatic oracle updates are used.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import LlamaConfig

from tools.qualification.k3_tokenizer_contract import ASSETS, verify_assets
from tools.qualification.train_tokenizer_convergence import (
    digest_file,
    qualify,
    run_backend,
    trainer_args,
)
from torchspec.data.dataset import load_conversation_dataset
from torchspec.data.renderers.smg import effective_renderer
from torchspec.data.utils import unpack_loss_mask
from torchspec.utils.distributed import init_gloo_group


def prepare(args):
    source = args.extraction.parent / "data" / "samples.json"
    samples = json.loads(source.read_text())
    preparation = json.loads((source.parent / "preparation.json").read_text())
    assert digest_file(source) == preparation["samples_sha256"]
    assert len(samples) == 320
    manifest = json.loads((args.extraction / "manifest.json").read_text())
    assert [r["id"] for r in manifest] == [r["id"] for r in samples]
    data, cache = {}, {}
    for split in ("train", "eval"):
        selected = [r for r in samples if r["split"] == split]
        raw = args.output / f"{split}.jsonl"
        raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n")
        data[split] = {}
        for backend in ("hf", "smg"):
            cfg = SimpleNamespace(
                target_model_path=str(args.assets / "model"),
                train_data_path=str(raw),
                prompt_key="messages",
                renderer="kimi-k3",
                rendering_backend="huggingface" if backend == "hf" else "auto",
                smg_exclude_k3=False,
                max_seq_length=args.max_length,
                defer_tokenization=False,
                train_with_decode=False,
                last_turn_loss_only=False,
                dynamic_loss_mask=False,
                min_loss_tokens=32,
                num_proc=2,
                cache_dir=str(args.output / "dataset-cache"),
            )
            actual = load_conversation_dataset(cfg)
            assert effective_renderer(cfg) == ("kimi-k3" if backend == "hf" else "smg-kimi-k3")
            assert len(actual) == len(selected)
            rows = []
            for got, expected in zip(actual, selected):
                ids = (
                    got["input_ids"].tolist()
                    if isinstance(got["input_ids"], torch.Tensor)
                    else got["input_ids"]
                )
                mask = unpack_loss_mask(got["packed_loss_mask"]).tolist()
                assert ids == expected["input_ids"] and mask == expected["loss_mask"]
                rows.append({"id": expected["id"], "input_ids": ids, "loss_mask": mask})
            data[split][backend] = rows
        assert data[split]["hf"] == data[split]["smg"]
        cache[split] = []
        for row in (r for r in manifest if r["split"] == split):
            tensors = {}
            expected = selected[len(cache[split])]
            for layer in row["layers"]:
                path = Path(layer["path"])
                assert digest_file(path) == layer["sha256"]
                item = load_file(path)
                assert item["input_ids"].reshape(-1).tolist() == expected["input_ids"]
                assert item["hidden_states"].shape == (len(expected["input_ids"]), 7168)
                assert torch.isfinite(item["hidden_states"]).all()
                tensors[layer["layer_id"]] = item["hidden_states"]
            assert set(tensors) == {2, 46, 90, 93}
            cache[split].append(
                {
                    "input_ids": torch.tensor(expected["input_ids"]).unsqueeze(0),
                    "hidden_states": torch.cat([tensors[i] for i in (2, 46, 90)], dim=-1).unsqueeze(
                        0
                    ),
                    "last_hidden_states": tensors[93].unsqueeze(0),
                }
            )
    return data, cache, preparation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="Directory containing model/ pointing to the real K3 checkpoint",
    )
    p.add_argument("--extraction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.steps <= 0 or (args.steps != 1000 and not args.smoke):
        p.error("Require 1000 steps, or explicit --smoke")
    args.max_length = 512
    args.batch_size = 1
    args.output.mkdir(parents=True, exist_ok=False)
    verify_assets(args.assets / "model")
    assert torch.cuda.is_available()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.compiler.set_stance("force_eager")
    config = LlamaConfig(
        vocab_size=163840,
        hidden_size=7168,
        intermediate_size=4096,
        num_hidden_layers=1,
        num_attention_heads=56,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
        rope_theta=1000000,
        attention_dropout=0.0,
        pad_token_id=163839,
        tie_word_embeddings=False,
        draft_vocab_size=163840,
        num_aux_hidden_states=3,
        fc_norm=True,
    )
    data, cache, preparation = prepare(args)
    options = {}
    for backend in ("hf", "smg"):
        cfg = trainer_args(args, backend)
        cfg.embedding_key = "language_model.model.embed_tokens.weight"
        cfg.lm_head_key = "language_model.lm_head.weight"
        cfg.norm_key = "language_model.model.norm.weight"
        cfg.last_hidden_states_prenorm = True
        cfg.trust_remote_code = True
        options[backend] = cfg
    metadata = {
        "checkpoint_assets": ASSETS,
        "torchspec_source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): digest_file(path)
            for path in sorted(Path(__file__).resolve().parents[2].glob("torchspec/**/*.py"))
            if not path.name.startswith(".")
        },
        "qualification_source_sha256": {
            path.name: digest_file(path)
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
        "weight_shard_sha256": digest_file(args.assets / "model/model-00094-of-000096.safetensors"),
        "data": preparation,
        "extraction_manifest_sha256": digest_file(args.extraction / "manifest.json"),
        "draft_config": config.to_dict(),
        "training": vars(options["hf"]),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "scope": "single-rank Eagle3Trainer/BF16Optimizer with real pretrained K3 TP8/PP2 cached targets, one-layer Llama draft; not a distributed queue or live serving qualification",
        "acceptance_criteria": {
            "steps_each": 1000,
            "max_relative_step_loss_difference": 0.01,
            "max_eval_loss_ratio": 0.9,
            "max_late_early_train_loss_ratio": 0.9,
            "min_accuracy_gain": 0.01,
        },
    }
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="k3-contract-dist-") as temp:
        dist.init_process_group("nccl", init_method=f"file://{temp}/init", rank=0, world_size=1)
        init_gloo_group()
        try:
            schedule = np.random.default_rng(42).integers(
                0, len(cache["train"]), size=(args.steps, args.batch_size)
            )
            runs = {}
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                for backend in ("hf", "smg"):
                    runs[backend] = run_backend(
                        args,
                        backend,
                        config,
                        data,
                        cache,
                        schedule,
                        expected_initial_digest=runs.get("hf", {}).get("initial_trainable_sha256"),
                        trainer_options=options[backend],
                        target_reference={
                            "K3_checkpoint_assets": ASSETS,
                            "weight_shard_sha256": metadata["weight_shard_sha256"],
                        },
                    )
            result = qualify(runs, args.steps)
            result["initial_trainable_equal"] = (
                runs["hf"]["initial_trainable_sha256"] == runs["smg"]["initial_trainable_sha256"]
            )
            result["final_trainable_equal"] = (
                runs["hf"]["final_trainable_sha256"] == runs["smg"]["final_trainable_sha256"]
            )
            (args.output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            import torchspec.utils.distributed as utils

            utils.GLOO_GROUP = None
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Matched 1,000-step HF-reference / SMG Eagle3 convergence qualification.

Pretrained Qwen3-0.6B, pinned UltraChat train/test shards, actual Eagle3Trainer
initialization/train steps and BF16Optimizer. Single-rank REPLICATE, cached real
HFRunner hidden states, no Ray queue or Mooncake transport. The HF control uses
the explicit one-turn supervision policy below, not the legacy parser.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, LlamaConfig

from tests.tokenizer_contract.support import load_manifest
from torchspec.data.dataset import load_conversation_dataset
from torchspec.data.utils import (
    DataCollatorWithPadding,
    pack_loss_mask,
    serialize_packed_loss_mask,
    unpack_loss_mask,
)
from torchspec.utils.distributed import init_gloo_group

MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
DATASET = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
SHARDS = {
    "train": "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
    "eval": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
}
SHARD_HASHES = {
    "train": "afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727",
    "eval": "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70",
}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def reference_sample(tokenizer, messages, max_length):
    """Independent Qwen3 single-turn specification, derived from message content.

    No SmgRenderer, production template registry, prefix re-encoding, or production
    mask helper participates in these expected IDs/masks.
    """
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Convergence reference requires exactly one system/user/assistant turn")
    if any(
        messages[-1].get(key)
        for key in ("reasoning", "reasoning_content", "thinking", "tool_calls")
    ):
        raise ValueError("Convergence reference accepts plain assistant content only")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        preserve_thinking=True,
    )
    body = "<think>\n\n</think>\n\n" + messages[-1]["content"].lstrip("\n")
    header, end = "<|im_start|>assistant\n", "<|im_end|>"
    tail = header + body + end + "\n"
    if not text.endswith(tail):
        raise ValueError("Checkpoint rendering does not satisfy the declared one-turn policy")
    start = len(text) - len(tail) + len(header)
    stop = start + len(body) + len(end)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(start <= a < b <= stop) for a, b in encoded.offset_mapping]
    return {"input_ids": encoded.input_ids[:max_length], "loss_mask": mask[:max_length]}


def prepare_data(args, tokenizer):
    data = {}
    seen_prompts = set()
    provenance = {}
    for split, wanted in (("train", args.train_samples), ("eval", args.eval_samples)):
        selected, reference, rejected = [], [], {}
        path = args.assets / "data" / f"{split}.parquet"
        assert digest_file(path) == SHARD_HASHES[split], f"Wrong pinned {split} shard"
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, use_threads=False):
            for row in batch.to_pylist():
                messages = row.get("messages") or []
                reason = None
                if len(messages) < 2 or [m.get("role") for m in messages[:2]] != [
                    "user",
                    "assistant",
                ]:
                    reason = "not_user_assistant_pair"
                else:
                    user, answer = (m.get("content") for m in messages[:2])
                    if (
                        not isinstance(user, str)
                        or not isinstance(answer, str)
                        or not answer.strip()
                    ):
                        reason = "empty_or_nontext"
                    elif len(user) + len(answer) > 20000:
                        reason = "source_character_limit"
                    elif any(
                        marker in user + answer
                        for marker in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
                    ):
                        reason = "unsupported_literal_markers"
                    elif digest_json(user) in seen_prompts:
                        reason = "duplicate_prompt_across_splits"
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    *messages[:2],
                ]
                expected = reference_sample(tokenizer, conversation, args.max_length)
                if sum(expected["loss_mask"]) < 32:
                    rejected["fewer_than_32_supervised_tokens"] = (
                        rejected.get("fewer_than_32_supervised_tokens", 0) + 1
                    )
                    continue
                seen_prompts.add(digest_json(user))
                sample_id = f"{split}-{len(selected):05d}"
                selected.append({"id": sample_id, "conversations": conversation})
                reference.append({"id": sample_id, **expected})
                if len(selected) == wanted:
                    break
            if len(selected) == wanted:
                break
        if len(selected) != wanted:
            raise ValueError(f"Insufficient qualified {split} samples: {len(selected)} / {wanted}")
        source = args.output / f"{split}.jsonl"
        source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n")
        smg_args = SimpleNamespace(
            rendering_backend="smg",
            target_model_path=str(args.assets / "model"),
            chat_template="qwen3-thinking",
            smg_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
            smg_include_end_token=True,
            train_data_path=str(source),
            prompt_key="conversations",
            max_seq_length=args.max_length,
            num_proc=2,
            cache_dir=str(args.output / "dataset-cache"),
            last_turn_loss_only=False,
        )
        actual = load_conversation_dataset(smg_args)
        normalized = [
            {
                "id": row["data_id"],
                "input_ids": row["input_ids"].tolist(),
                "loss_mask": unpack_loss_mask(row["packed_loss_mask"]).tolist(),
            }
            for row in actual
        ]
        if normalized != reference:
            for i, (a, b) in enumerate(zip(normalized, reference)):
                if a != b:
                    (args.output / "preprocessing-mismatch.json").write_text(
                        json.dumps({"split": split, "index": i, "hf": b, "smg": a}, indent=2)
                    )
                    break
            raise ValueError(f"{split}: HF / SMG preprocessing differs; refusing training")
        data[split] = {"hf": reference, "smg": normalized}
        provenance[split] = {
            "shard": SHARDS[split],
            "shard_sha256": digest_file(path),
            "selected": len(selected),
            "selected_source_sha256": digest_file(source),
            "hf_token_mask_sha256": digest_json(reference),
            "smg_token_mask_sha256": digest_json(normalized),
            "rejected_before_backend_comparison": rejected,
            "tokens": sum(len(row["input_ids"]) for row in reference),
            "supervised_tokens": sum(sum(row["loss_mask"]) for row in reference),
        }
    (args.output / "preprocessing.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return data, provenance


def materialize_hidden_states(args, data):
    from torchspec.inference.engine.hf_runner import HFRunner

    runner = HFRunner.from_pretrained(
        str(args.assets / "model"),
        torch_dtype=torch.bfloat16,
        max_seq_length=args.max_length,
        aux_hidden_states_layers=[1, 14, 26],
    )
    cache, observed = {}, []
    handle = runner.target_model.model.get_input_embeddings().register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[0].detach().cpu().squeeze(0).tolist())
    )
    try:
        for split in ("train", "eval"):
            cache[split] = []
            for i, row in enumerate(data[split]["hf"]):
                packed = serialize_packed_loss_mask(pack_loss_mask(torch.tensor(row["loss_mask"])))
                output = runner.generate(row["id"], [torch.tensor(row["input_ids"])], [packed])[0]
                sample = {
                    k: v.detach().cpu()
                    for k, v in output["tensors"].items()
                    if isinstance(v, torch.Tensor)
                }
                assert observed.pop() == row["input_ids"]
                assert sample["input_ids"].squeeze(0).tolist() == row["input_ids"]
                cache[split].append(sample)
                if (i + 1) % 32 == 0:
                    print(f"cached {split}: {i + 1}/{len(data[split]['hf'])}", flush=True)
    finally:
        handle.remove()
        runner.shutdown()
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def trainer_args(args, backend):
    return argparse.Namespace(
        target_model_path=str(args.assets / "model"),
        chat_template=None,
        attention_backend="sdpa",
        ttt_length=2,
        gradient_checkpointing=False,
        freeze_lm_head=True,
        load_lm_head_from_target=True,
        compute_logits_in_trainer=True,
        last_hidden_states_prenorm=False,
        trust_remote_code=False,
        embedding_key="model.embed_tokens.weight",
        lm_head_key="model.embed_tokens.weight",
        norm_key="model.norm.weight",
        max_seq_length=args.max_length,
        seed=42,
        learning_rate=1e-4,
        weight_decay=0.0,
        max_grad_norm=0.5,
        warmup_ratio=0.015,
        lr_total_steps=args.steps,
        lr_decay_style="cosine",
        min_lr=1e-5,
        fsdp_cpu_offload=False,
        fsdp_strategy="REPLICATE",
        fsdp_reduce_dtype="float32",
        fp16=False,
        load_path=None,
        checkpoint_dir=str(args.output / backend),
        use_pytorch_profiler=False,
        record_memory_history=False,
        profile_target=[],
        dynamic_loss_mask=False,
        last_turn_loss_only=False,
        enable_perf_metrics=False,
        save_debug_train_data=None,
        max_dump_steps=0,
        micro_batch_size=args.batch_size,
    )


def trainable_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def batch_for(indices, backend, split, data, cache, collator):
    features = []
    for i in indices:
        row = data[split][backend][i]
        item = dict(cache[split][i])
        # The backend-specific inputs and masks are used by the real trainer;
        # hidden-state reuse is permitted only after the exact parity gate.
        item["input_ids"] = torch.tensor(row["input_ids"]).unsqueeze(0)
        item["loss_mask"] = torch.tensor(row["loss_mask"]).unsqueeze(0)
        features.append(item)
    return collator(features)


def evaluate(trainer, backend, data, cache, collator, batch_size, step):
    trainer.model.eval()
    weighted_loss, correct, counts = (torch.zeros(2, device="cuda") for _ in range(3))
    with torch.no_grad():
        for start in range(0, len(cache["eval"]), batch_size):
            batch = batch_for(
                range(start, min(start + batch_size, len(cache["eval"]))),
                backend,
                "eval",
                data,
                cache,
                collator,
            )
            values = trainer.eval_forward(batch)
            counts += values["acc_counts"]
            weighted_loss += values["vlosses"] * values["acc_counts"]
            correct += values["acces"] * values["acc_counts"]
    trainer.model.train()
    losses, accuracy = weighted_loss / counts, correct / counts
    return {
        "step": step,
        "loss": float((losses[0] + 0.8 * losses[1]) / 1.8),
        "loss_by_depth": losses.tolist(),
        "accuracy_by_depth": accuracy.tolist(),
        "valid_tokens_by_depth": counts.tolist(),
        "simulated_acceptance_length": float(accuracy[0] + accuracy[0] * accuracy[1]),
    }


def run_backend(args, backend, config, data, cache, schedule, expected_initial_digest=None):
    from torchspec.training.eagle3_trainer import Eagle3Trainer

    folder = args.output / backend
    folder.mkdir()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    trainer = Eagle3Trainer(trainer_args(args, backend))
    try:
        trainer.init_model(config, str(args.assets / "model"))
        initial_digest = trainable_digest(trainer.draft_model)
        if expected_initial_digest is not None:
            assert initial_digest == expected_initial_digest, (
                "Different initialization before training"
            )
        collator = DataCollatorWithPadding()
        evaluations = [evaluate(trainer, backend, data, cache, collator, args.batch_size, 0)]
        trajectory = []
        started = time.monotonic()
        with (folder / "steps.jsonl").open("w") as stream:
            for step, indices in enumerate(schedule, start=1):
                batch = batch_for(indices, backend, "train", data, cache, collator)
                values = trainer._train_step(
                    batch, accumulation_steps=1, step=step - 1, batch_idx=0, num_batches=1
                )
                grad_norm = trainer.optimizer.step()
                loss = float(values["total_loss"] / 1.8)
                if not math.isfinite(loss) or not torch.isfinite(grad_norm):
                    raise ValueError(f"{backend} step {step}: nonfinite loss/gradient")
                assert (values["acc_counts"] > 0).all()
                item = {
                    "step": step,
                    "loss": loss,
                    "grad_norm": float(grad_norm),
                    "accuracy_by_depth": values["acces"].tolist(),
                    "lr": trainer.optimizer.get_learning_rate(),
                    "sample_indices": list(map(int, indices)),
                }
                trajectory.append(item)
                stream.write(json.dumps(item) + "\n")
                trainer.global_step = step
                if step % 100 == 0 or step == args.steps:
                    stream.flush()
                    evaluations.append(
                        evaluate(trainer, backend, data, cache, collator, args.batch_size, step)
                    )
                    (folder / "eval.json").write_text(json.dumps(evaluations, indent=2) + "\n")
                    print(
                        f"{backend} step={step} train_loss={loss:.5f} eval={evaluations[-1]} elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
        final_digest = trainable_digest(trainer.draft_model)
        assert final_digest != initial_digest
        torch.save(
            {
                "step": args.steps,
                "draft_config": config.to_dict(),
                "target_revision": load_manifest()["profiles"]["qwen3"]["revision"],
                "frozen_keys_from_target": ["embed_tokens.weight", "lm_head.weight"],
                "trainable_state_dict": {
                    name: p.detach().cpu()
                    for name, p in trainer.draft_model.named_parameters()
                    if p.requires_grad
                },
            },
            folder / "trainable-checkpoint.pt",
        )
        return {
            "initial_trainable_sha256": initial_digest,
            "final_trainable_sha256": final_digest,
            "elapsed_seconds": time.monotonic() - started,
            "steps": trajectory,
            "eval": evaluations,
        }
    finally:
        trainer.close()


def qualify(runs, steps):
    assert steps > 0
    for run in runs.values():
        assert [row["step"] for row in run["steps"]] == list(range(1, steps + 1))
        assert all(math.isfinite(row["loss"]) for row in run["steps"])
        assert run["eval"][0]["step"] == 0 and run["eval"][-1]["step"] == steps
        assert all(
            math.isfinite(row["loss"]) and all(math.isfinite(v) for v in row["accuracy_by_depth"])
            for row in run["eval"]
        )
    assert runs["hf"]["initial_trainable_sha256"] == runs["smg"]["initial_trainable_sha256"]
    diffs = [
        abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), abs(b["loss"]), 1e-8)
        for a, b in zip(runs["hf"]["steps"], runs["smg"]["steps"], strict=True)
    ]
    result = {
        "required_steps_per_backend": 1000,
        "actual_steps_per_backend": steps,
        "max_relative_step_loss_difference": max(diffs),
        "max_allowed_relative_difference": 0.01,
        "backends": {},
    }
    assert max(diffs) <= 0.01, result
    for name, run in runs.items():
        assert len(run["steps"]) == steps
        initial, final = run["eval"][0], run["eval"][-1]
        early = float(np.mean([x["loss"] for x in run["steps"][:100]]))
        late = float(np.mean([x["loss"] for x in run["steps"][-100:]]))
        result["backends"][name] = {
            "initial_eval_loss": initial["loss"],
            "final_eval_loss": final["loss"],
            "eval_loss_ratio": final["loss"] / initial["loss"],
            "early_100_train_loss": early,
            "late_100_train_loss": late,
            "train_loss_ratio": late / early,
            "initial_accuracy": initial["accuracy_by_depth"][0],
            "final_accuracy": final["accuracy_by_depth"][0],
        }
        if steps == 1000:
            assert final["loss"] <= 0.9 * initial["loss"], result
            assert late <= 0.9 * early, result
            assert final["accuracy_by_depth"][0] >= initial["accuracy_by_depth"][0] + 0.01, result
    result["qualified"] = steps == 1000
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a short debug run; never marks convergence qualified",
    )
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if (
        min(args.steps, args.train_samples, args.eval_samples, args.max_length, args.batch_size)
        <= 0
    ):
        parser.error("Steps, sample counts, maximum length and batch size must be positive")
    if args.steps != 1000 and not args.smoke:
        parser.error("Qualification requires exactly 1000 steps per backend")
    args.output.mkdir(parents=True, exist_ok=False)
    model = args.assets / "model"
    profile = load_manifest()["profiles"]["qwen3"]
    for name, digest in profile["assets"].items():
        assert digest_file(model / name) == digest, name
    assert digest_file(model / "model.safetensors") == MODEL_SHA256
    assert torch.cuda.is_available()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.compiler.set_stance("force_eager")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    data, provenance = prepare_data(args, tokenizer)
    config = LlamaConfig(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=4096,
        rope_theta=1000000,
        attention_dropout=0.0,
        pad_token_id=0,
        tie_word_embeddings=False,
        draft_vocab_size=151936,
        num_aux_hidden_states=3,
    )
    metadata = {
        "model": profile["model"],
        "model_revision": profile["revision"],
        "weights_sha256": MODEL_SHA256,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "preprocessing": provenance,
        "draft_config": config.to_dict(),
        "training": vars(trainer_args(args, "hf")),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "policy": "complete one-turn Qwen3, assistant body and end marker, no formatting newline, full-sequence offsets",
        "scope": "single rank real pretrained target and Eagle3Trainer/BF16Optimizer; direct cached tensors; no Ray/RDMA or live speculative decoding",
        "acceptance_criteria": {
            "steps_each": 1000,
            "max_relative_step_loss_difference": 0.01,
            "maximum_final_initial_eval_loss_ratio": 0.9,
            "maximum_late_early_train_loss_ratio": 0.9,
            "minimum_eval_accuracy_gain": 0.01,
        },
    }
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="torchspec-contract-dist-") as rendezvous:
        dist.init_process_group(
            "nccl", init_method=f"file://{rendezvous}/init", rank=0, world_size=1
        )
        init_gloo_group()
        try:
            cache = materialize_hidden_states(args, data)
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
                        runs.get("hf", {}).get("initial_trainable_sha256"),
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
            result = qualify(runs, args.steps)
            (args.output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
        finally:
            import torchspec.utils.distributed as distributed_utils

            distributed_utils.GLOO_GROUP = None
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

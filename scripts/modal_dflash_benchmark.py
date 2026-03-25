"""
DFlash inference benchmark on Modal GPU platform — z-lab methodology.

Uses the same datasets, prompt formatting, and evaluation protocol as
z-lab/dflash (https://github.com/z-lab/dflash):
  GSM8K, MATH-500, AIME24, AIME25, HumanEval, MBPP, LiveCodeBench,
  SWE-Bench, MT-Bench, Alpaca.

Setup (one-time — same secrets as modal_dflash_train.py):
    modal token set --token-id <id> --token-secret <secret>
    modal secret create huggingface-secret HF_TOKEN=hf_...

Usage:
    # Run all 10 z-lab benchmarks (default)
    modal run scripts/modal_dflash_benchmark.py

    # Run specific datasets
    modal run scripts/modal_dflash_benchmark.py --datasets gsm8k,math500

    # Custom draft model
    modal run scripts/modal_dflash_benchmark.py \
        --draft-repo Xingh3/dflash-qwen3-8b-3epoch

    # Custom block size and temperature
    modal run scripts/modal_dflash_benchmark.py --block-size 8 --temperature 0.6

    # Skip baseline (only measure DFlash)
    modal run scripts/modal_dflash_benchmark.py --skip-baseline
"""

from __future__ import annotations

from typing import Optional

import modal

# =============================================================================
# Constants
# =============================================================================

TORCHSPEC_REPO = "https://github.com/zhubohao911/TorchSpec.git"
TORCHSPEC_BRANCH = "feature/dflash-training"

REPO_DIR = "/workspace/TorchSpec"
HF_CACHE_DIR = "/root/.cache/huggingface"

BENCHMARK_GPU = "H100:1"

ZLAB_BENCHMARKS = {
    "gsm8k": 128,
    "math500": 128,
    "aime24": 30,
    "aime25": 30,
    "humaneval": 164,
    "mbpp": 128,
    "livecodebench": 128,
    "swe-bench": 128,
    "mt-bench": 80,
    "alpaca": 128,
}

# =============================================================================
# Modal app + volumes
# =============================================================================

app = modal.App("torchspec-dflash-benchmark")

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

# =============================================================================
# Container image
# =============================================================================

benchmark_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        f"git clone {TORCHSPEC_REPO} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {TORCHSPEC_BRANCH}",
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
        "transformers==4.57.1",
        "datasets",
        "accelerate",
        "safetensors",
    )
    .run_commands(f"cd {REPO_DIR} && pip install -e '.[dev]'")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": HF_CACHE_DIR,
        }
    )
)


# =============================================================================
# Dataset loading — mirrors z-lab/dflash model/utils.py exactly
# =============================================================================

def load_benchmark_dataset(data_name: str, max_samples: int | None = None):
    """Load and format datasets using the same logic as z-lab/dflash."""
    from datasets import load_dataset, Features, Sequence, Value

    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {
            "formatted_input": (
                f"{x['instruction']}\n\nInput:\n{x['input']}" if x["input"] else x["instruction"]
            )
        })
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = (
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            "```python\n{prompt}\n```"
        )
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = [f"test{s}.jsonl" for s in ["", "2", "3", "4", "5", "6"]]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features,
        )

    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=0).select(range(max_samples))

    return dataset


# =============================================================================
# Benchmark runner
# =============================================================================

@app.function(
    image=benchmark_image,
    gpu=BENCHMARK_GPU,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    timeout=4 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_benchmark(
    target_model: str,
    draft_repo: str,
    dataset_name: str,
    max_samples: int,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    skip_baseline: bool,
):
    import json
    import math
    import random
    import time
    from typing import List, Tuple

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from torchspec.models.draft.dflash import (
        DFlashConfig,
        DFlashDraftModel,
        build_target_layer_ids,
    )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = "cuda"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {mem_gb:.1f} GB")

    # ── Load target model ──
    print(f"\nLoading target model: {target_model}...")
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    target.eval()
    target_params = sum(p.numel() for p in target.parameters()) / 1e9
    print(f"Target model loaded ({target_params:.1f}B params)")

    # ── Load draft model from HuggingFace ──
    print(f"\nLoading DFlash draft model: {draft_repo}...")
    draft = DFlashDraftModel.from_pretrained(
        draft_repo,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    draft.eval()

    target_layer_ids = draft.target_layer_ids
    context_proj = draft.context_proj
    context_norm = draft.context_norm

    draft_params = sum(p.numel() for p in draft.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in draft.parameters() if p.requires_grad) / 1e6
    print(f"Draft model loaded ({draft_params:.1f}M total, {trainable_params:.1f}M trainable)")
    print(f"Target layer IDs: {target_layer_ids}")
    print(f"Block size: {block_size}")

    # ── Load dataset ──
    print(f"\nLoading dataset: {dataset_name} (max {max_samples} samples)...")
    dataset = load_benchmark_dataset(dataset_name, max_samples)
    print(f"Loaded {len(dataset)} samples")

    # ── Helpers ──

    def extract_context_feature(
        hidden_states: list[torch.Tensor],
        layer_ids: list[int],
    ) -> torch.Tensor:
        offset = 1
        selected = [hidden_states[lid + offset] for lid in layer_ids]
        return torch.cat(selected, dim=-1)

    def sample(logits: torch.Tensor, temp: float = 0.0) -> torch.Tensor:
        if temp < 1e-5:
            return torch.argmax(logits, dim=-1)
        logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        if logits.dim() == 3:
            bsz, seq_len, vocab = logits.shape
            return torch.multinomial(
                probs.view(-1, vocab), num_samples=1
            ).view(bsz, seq_len)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def has_stop_token(token_ids: torch.Tensor) -> bool:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            return False
        return (token_ids == eos_id).any().item()

    @torch.inference_mode()
    def generate_baseline(
        input_ids: torch.LongTensor,
        max_new: int,
    ) -> Tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        past_key_values = DynamicCache()
        generated = input_ids.clone()

        out = target(input_ids, past_key_values=past_key_values, use_cache=True)
        next_token = sample(out.logits[:, -1:, :], temperature)
        generated = torch.cat([generated, next_token], dim=1)

        for _ in range(max_new - 1):
            out = target(next_token, past_key_values=past_key_values, use_cache=True)
            next_token = sample(out.logits[:, -1:, :], temperature)
            generated = torch.cat([generated, next_token], dim=1)
            if has_stop_token(next_token):
                break

        elapsed = time.perf_counter() - t0
        return generated, elapsed

    @torch.inference_mode()
    def generate_dflash_spec(
        input_ids: torch.LongTensor,
        max_new: int,
    ) -> Tuple[torch.Tensor, float, List[int]]:
        num_input = input_ids.shape[1]
        max_length = num_input + max_new
        mask_token_id = draft.mask_token_id

        output_ids = torch.full(
            (1, max_length + block_size),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        output_ids[:, :num_input] = input_ids

        t0 = time.perf_counter()

        past_kv_target = DynamicCache()
        out = target(
            input_ids,
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )

        first_token = sample(out.logits[:, -1:, :], temperature)
        output_ids[:, num_input] = first_token.squeeze()

        ctx_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
        ctx_feature = context_norm(context_proj(ctx_hidden.to(context_proj.weight.dtype)))

        acceptance_lengths = []
        start = num_input

        while start < max_length:
            block_ids = output_ids[:, start : start + block_size].clone()

            draft_pos = torch.arange(
                start, start + block_size, device=device
            ).unsqueeze(0)
            ctx_pos = torch.arange(start, device=device).unsqueeze(0)

            draft_hidden = draft(
                draft_input_ids=block_ids,
                context_feature=ctx_feature[:, :start, :],
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
                block_mask=None,
            )

            draft_logits = target.lm_head(draft_hidden[:, 1:, :])
            draft_tokens = sample(draft_logits, temperature)

            block_ids[:, 1:] = draft_tokens

            out_verify = target(
                block_ids,
                position_ids=draft_pos,
                past_key_values=past_kv_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(out_verify.logits, temperature)

            match = (block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1)
            acc_len = match.sum(dim=1)[0].item()

            output_ids[:, start : start + acc_len + 1] = block_ids[:, : acc_len + 1]
            output_ids[:, start + acc_len + 1] = posterior[:, acc_len]

            start += acc_len + 1
            past_kv_target.crop(start)

            ctx_hidden_new = extract_context_feature(
                out_verify.hidden_states, target_layer_ids
            )
            new_feat = context_norm(
                context_proj(ctx_hidden_new.to(context_proj.weight.dtype))
            )
            ctx_feature = torch.cat(
                [ctx_feature[:, :start - acc_len - 1, :], new_feat[:, : acc_len + 1, :]],
                dim=1,
            )

            acceptance_lengths.append(acc_len + 1)

            if has_stop_token(output_ids[:, start - 1 : start]):
                break

        elapsed = time.perf_counter() - t0

        output_ids = output_ids[:, :max_length]
        valid_mask = output_ids[0] != mask_token_id
        output_ids = output_ids[:, valid_mask]

        return output_ids, elapsed, acceptance_lengths

    # ── Format prompts using chat template (z-lab methodology) ──
    def format_prompt(turns: list[str]) -> str:
        """Apply chat template to (potentially multi-turn) prompts."""
        messages = [{"role": "user", "content": turns[0]}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    print(f"\n{'='*70}")
    print(f"Benchmark: {dataset_name} | {len(dataset)} samples | max_new_tokens={max_new_tokens}")
    print(f"Draft model: {draft_repo}")
    print(f"Target model: {target_model}")
    print(f"Temperature: {temperature}, Block size: {block_size}")
    print(f"{'='*70}\n")

    # ── Warmup ──
    warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)
    generate_dflash_spec(warmup_ids, 16)
    if not skip_baseline:
        generate_baseline(warmup_ids, 16)
    torch.cuda.synchronize()

    # ── Multi-turn handling for MT-Bench ──
    is_multiturn = dataset_name == "mt-bench"

    # ── Run benchmark ──
    baseline_times = []
    baseline_tokens = []
    dflash_times = []
    dflash_tokens = []
    all_acc_lens = []

    for idx in range(len(dataset)):
        instance = dataset[idx]
        turns = instance["turns"]

        messages = []
        for turn_index, user_content in enumerate(turns):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            # Only benchmark on the last turn for multi-turn (collect metrics)
            # For single-turn datasets this is always the only turn
            is_last_turn = (turn_index == len(turns) - 1)

            # Baseline
            if not skip_baseline and is_last_turn:
                torch.cuda.synchronize()
                output, elapsed = generate_baseline(input_ids, max_new_tokens)
                torch.cuda.synchronize()
                num_new = output.shape[1] - input_ids.shape[1]
                baseline_times.append(elapsed)
                baseline_tokens.append(num_new)

            # DFlash
            torch.cuda.synchronize()
            output, elapsed, acc_lens = generate_dflash_spec(input_ids, max_new_tokens)
            torch.cuda.synchronize()
            num_new = output.shape[1] - input_ids.shape[1]

            if is_last_turn:
                dflash_times.append(elapsed)
                dflash_tokens.append(num_new)
                all_acc_lens.extend(acc_lens)

            # Decode response and add to conversation for multi-turn
            if is_multiturn and not is_last_turn:
                generated_ids = output[0, input_ids.shape[1]:]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": output_text})

        if (idx + 1) % max(1, len(dataset) // 10) == 0 or idx == len(dataset) - 1:
            avg_tau_so_far = sum(all_acc_lens) / max(len(all_acc_lens), 1)
            print(f"  [{idx+1}/{len(dataset)}] τ={avg_tau_so_far:.2f}")

    # ── Compute metrics ──
    avg_tau = sum(all_acc_lens) / max(len(all_acc_lens), 1)

    avg_dflash_time = sum(dflash_times) / len(dflash_times) if dflash_times else 0
    avg_dflash_tokens = sum(dflash_tokens) / len(dflash_tokens) if dflash_tokens else 0
    avg_dflash_tps = (
        sum(t / e for t, e in zip(dflash_tokens, dflash_times) if e > 0) / len(dflash_times)
        if dflash_times else 0
    )

    # ── τ distribution ──
    print(f"\n{'='*70}")
    print(f"τ DISTRIBUTION ({dataset_name})")
    print(f"{'='*70}")
    if all_acc_lens:
        tau_counts = {}
        for a in all_acc_lens:
            tau_counts[a] = tau_counts.get(a, 0) + 1
        for k in sorted(tau_counts.keys()):
            bar = "█" * int(tau_counts[k] / max(tau_counts.values()) * 40)
            pct = tau_counts[k] / len(all_acc_lens) * 100
            print(f"  τ={k:2d}: {tau_counts[k]:4d} ({pct:5.1f}%) {bar}")
        median_tau = sorted(all_acc_lens)[len(all_acc_lens) // 2]
        print(f"  Total cycles: {len(all_acc_lens)}, mean τ={avg_tau:.2f}, median τ={median_tau}")

    # ── Summary ──
    result = {
        "dataset": dataset_name,
        "num_samples": len(dataset),
        "draft_model": draft_repo,
        "target_model": target_model,
        "gpu": torch.cuda.get_device_name(0),
        "max_new_tokens": max_new_tokens,
        "block_size": block_size,
        "temperature": temperature,
        "avg_acceptance_length": round(avg_tau, 2),
        "avg_dflash_tps": round(avg_dflash_tps, 1),
    }

    print(f"\n{'='*70}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*70}")
    print(f"  Draft model:           {draft_repo}")
    print(f"  Target model:          {target_model}")
    print(f"  GPU:                   {torch.cuda.get_device_name(0)}")
    print(f"  Samples:               {len(dataset)}")
    print(f"  Max new tokens:        {max_new_tokens}")
    print(f"  Block size:            {block_size}")
    print(f"  Temperature:           {temperature}")
    print(f"  Acceptance length (τ): {avg_tau:.2f}")
    print(f"  DFlash throughput:     {avg_dflash_tps:.1f} tok/s")

    if not skip_baseline and baseline_times:
        avg_baseline_tps = (
            sum(t / e for t, e in zip(baseline_tokens, baseline_times) if e > 0)
            / len(baseline_times)
        )
        speedup = avg_baseline_tps / avg_dflash_tps if avg_dflash_tps > 0 else 0

        # z-lab method: compute speedup from time-per-token ratio
        t1_per_tok = np.mean([e / t for e, t in zip(baseline_times, baseline_tokens) if t > 0])
        tb_per_tok = np.mean([e / t for e, t in zip(dflash_times, dflash_tokens) if t > 0])
        zlab_speedup = t1_per_tok / tb_per_tok if tb_per_tok > 0 else 0

        print(f"  Baseline throughput:   {avg_baseline_tps:.1f} tok/s")
        print(f"  Decoding speedup:      {zlab_speedup:.2f}x")
        result["avg_baseline_tps"] = round(avg_baseline_tps, 1)
        result["decoding_speedup"] = round(zlab_speedup, 2)

    print(f"{'='*70}")

    # ── z-lab reference comparison ──
    zlab_tau_ref = {
        "gsm8k": 3.38, "math500": 4.61, "aime24": 4.12, "aime25": 4.07,
    }
    zlab_speedup_ref = {
        "gsm8k": 5.20, "math500": 6.17, "aime24": 5.91, "aime25": 5.85,
        "humaneval": 5.20, "mbpp": 4.75, "livecodebench": 5.43,
        "swe-bench": 2.92, "mt-bench": 2.79, "alpaca": 2.27,
    }
    if dataset_name in zlab_tau_ref:
        ref_tau = zlab_tau_ref[dataset_name]
        print(f"\n  z-lab reference τ:     {ref_tau:.2f} (ours: {avg_tau:.2f}, "
              f"gap: {ref_tau - avg_tau:+.2f})")
    if dataset_name in zlab_speedup_ref:
        ref_spd = zlab_speedup_ref[dataset_name]
        print(f"  z-lab reference speed: {ref_spd:.2f}x")

    return result


# =============================================================================
# CLI entry point
# =============================================================================

@app.local_entrypoint()
def main(
    target_model: str = "Qwen/Qwen3-8B",
    draft_repo: str = "Xingh3/dflash-qwen3-8b-3epoch",
    datasets: str = "gsm8k,math500",
    max_new_tokens: int = 2048,
    block_size: int = 16,
    temperature: float = 0.0,
    skip_baseline: bool = False,
    all_datasets: bool = False,
):
    import numpy as np

    if all_datasets:
        ds_list = list(ZLAB_BENCHMARKS.keys())
    else:
        ds_list = [d.strip() for d in datasets.split(",")]

    print("=" * 70)
    print("  DFlash Inference Benchmark on Modal (z-lab methodology)")
    print("=" * 70)
    print(f"  Target model:   {target_model}")
    print(f"  Draft model:    {draft_repo}")
    print(f"  GPU:            {BENCHMARK_GPU}")
    print(f"  Datasets:       {', '.join(ds_list)}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Block size:     {block_size}")
    print(f"  Temperature:    {temperature}")
    print(f"  Skip baseline:  {skip_baseline}")
    print("=" * 70)

    all_results = []

    for ds_name in ds_list:
        max_samples = ZLAB_BENCHMARKS.get(ds_name, 128)
        print(f"\n>>> Running: {ds_name} ({max_samples} samples) <<<\n")

        result = run_benchmark.remote(
            target_model=target_model,
            draft_repo=draft_repo,
            dataset_name=ds_name,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            skip_baseline=skip_baseline,
        )
        all_results.append(result)

    # ── Cross-dataset summary ──
    print(f"\n\n{'='*70}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'τ':>6} {'tok/s':>8} {'Speedup':>8} {'z-lab τ':>8} {'z-lab Spd':>10}")
    print("-" * 70)

    zlab_tau_ref = {
        "gsm8k": 3.38, "math500": 4.61, "aime24": 4.12, "aime25": 4.07,
    }
    zlab_speedup_ref = {
        "gsm8k": 5.20, "math500": 6.17, "aime24": 5.91, "aime25": 5.85,
        "humaneval": 5.20, "mbpp": 4.75, "livecodebench": 5.43,
        "swe-bench": 2.92, "mt-bench": 2.79, "alpaca": 2.27,
    }

    for r in all_results:
        ds = r["dataset"]
        tau = r.get("avg_acceptance_length", 0)
        tps = r.get("avg_dflash_tps", 0)
        spd = r.get("decoding_speedup", 0)
        ref_tau = zlab_tau_ref.get(ds, "-")
        ref_spd = zlab_speedup_ref.get(ds, "-")
        ref_tau_str = f"{ref_tau:.2f}" if isinstance(ref_tau, float) else ref_tau
        ref_spd_str = f"{ref_spd:.2f}x" if isinstance(ref_spd, float) else ref_spd
        spd_str = f"{spd:.2f}x" if spd else "-"
        print(f"{ds:<15} {tau:>6.2f} {tps:>8.1f} {spd_str:>8} {ref_tau_str:>8} {ref_spd_str:>10}")

    print(f"{'='*70}")

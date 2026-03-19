#!/usr/bin/env python3
"""DFlash inference benchmark — measures speculative decoding performance.

Benchmarks:
  1. Target-only baseline (autoregressive)
  2. DFlash speculative decoding (block-parallel draft)

Metrics:
  - Acceptance length (τ): avg tokens accepted per draft cycle
  - Wall-clock speedup: target_time / dflash_time
  - Tokens/sec throughput

Usage:
  python scripts/benchmark_dflash_inference.py \
    --target_model Qwen/Qwen3-8B \
    --draft_checkpoint outputs/dflash/model.pt \
    --num_prompts 20 --max_new_tokens 128
"""

import argparse
import json
import math
import os
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from torchspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDraftModel,
    build_target_layer_ids,
)


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    target_layer_ids: list[int],
) -> torch.Tensor:
    """Extract and concatenate hidden states from specified target layers."""
    # hidden_states includes embedding output at index 0, so add offset
    offset = 1
    selected = [hidden_states[lid + offset] for lid in target_layer_ids]
    return torch.cat(selected, dim=-1)


@torch.inference_mode()
def generate_baseline(
    target: nn.Module,
    tokenizer,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    """Target-only autoregressive generation (baseline)."""
    t0 = time.perf_counter()

    past_key_values = DynamicCache()
    num_input = input_ids.shape[1]
    generated = input_ids.clone()

    # Prefill
    out = target(
        input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    next_token = _sample(out.logits[:, -1:, :], temperature)
    generated = torch.cat([generated, next_token], dim=1)

    # Decode
    for _ in range(max_new_tokens - 1):
        out = target(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = _sample(out.logits[:, -1:, :], temperature)
        generated = torch.cat([generated, next_token], dim=1)

        if _has_stop_token(next_token, tokenizer):
            break

    elapsed = time.perf_counter() - t0
    return generated, elapsed


@torch.inference_mode()
def generate_dflash_spec(
    target: nn.Module,
    draft_model: DFlashDraftModel,
    context_proj: nn.Linear,
    context_norm: nn.Module,
    tokenizer,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    block_size: int = 16,
    target_layer_ids: list[int] = None,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, float, List[int]]:
    """DFlash speculative decoding.

    Returns:
        generated: output token IDs
        elapsed: wall-clock time
        acceptance_lengths: list of accepted tokens per cycle
    """
    device = input_ids.device
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens

    mask_token_id = draft_model.mask_token_id

    # Pre-allocate output buffer
    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    output_ids[:, :num_input] = input_ids

    t0 = time.perf_counter()

    # Prefill target (single call)
    past_kv_target = DynamicCache()
    out = target(
        input_ids,
        past_key_values=past_kv_target,
        use_cache=True,
        output_hidden_states=True,
    )

    # First token from target
    first_token = _sample(out.logits[:, -1:, :], temperature)
    output_ids[:, num_input] = first_token.squeeze()

    # Extract context feature for draft
    ctx_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
    ctx_feature = context_norm(context_proj(ctx_hidden.to(context_proj.weight.dtype)))

    acceptance_lengths = []
    start = num_input

    while start < max_length:
        # Build draft block: [anchor_token, MASK, MASK, ..., MASK]
        block_ids = output_ids[:, start : start + block_size].clone()

        # Position IDs for draft block and context
        draft_pos = torch.arange(
            start, start + block_size, device=device
        ).unsqueeze(0)
        ctx_pos = torch.arange(start, device=device).unsqueeze(0)

        # Draft forward (no KV cache — recompute each cycle)
        draft_hidden = draft_model(
            draft_input_ids=block_ids,
            context_feature=ctx_feature[:, :start, :],
            draft_position_ids=draft_pos,
            context_position_ids=ctx_pos,
            block_mask=None,  # No FlexAttention at inference — bidirectional SDPA
        )

        # Get logits from target's LM head — skip anchor (pos 0), predict pos 1..block_size-1
        draft_logits = target.lm_head(draft_hidden[:, 1:, :])
        draft_tokens = _sample(draft_logits, temperature)

        # Fill in draft predictions
        block_ids[:, 1:] = draft_tokens

        # Verify with target
        out_verify = target(
            block_ids,
            position_ids=draft_pos,
            past_key_values=past_kv_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = _sample(out_verify.logits, temperature)

        # Count accepted tokens
        match = (block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1)
        acc_len = match.sum(dim=1)[0].item()

        # Accept tokens
        output_ids[:, start : start + acc_len + 1] = block_ids[:, : acc_len + 1]
        output_ids[:, start + acc_len + 1] = posterior[:, acc_len]

        start += acc_len + 1
        past_kv_target.crop(start)

        # Update context feature
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

        if _has_stop_token(output_ids[:, start - 1 : start], tokenizer):
            break

    elapsed = time.perf_counter() - t0

    # Trim output
    output_ids = output_ids[:, :max_length]
    valid_mask = output_ids[0] != mask_token_id
    output_ids = output_ids[:, valid_mask]

    return output_ids, elapsed, acceptance_lengths


def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Sample from logits. Greedy if temperature < 1e-5."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if logits.dim() == 3:
        bsz, seq_len, vocab = logits.shape
        return torch.multinomial(
            probs.view(-1, vocab), num_samples=1
        ).view(bsz, seq_len)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _has_stop_token(token_ids: torch.Tensor, tokenizer) -> bool:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        return False
    return (token_ids == eos_id).any().item()


def _generate_training_sequences(
    target: nn.Module,
    tokenizer,
    prompts: list[str],
    gen_len: int = 128,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Generate long sequences from target model for training data.

    Each prompt is extended to gen_len new tokens via greedy autoregressive decoding,
    ensuring training sequences are long enough for block_size=16.
    """
    sequences = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = target.generate(
                input_ids,
                max_new_tokens=gen_len,
                do_sample=False,
                use_cache=True,
            )
        sequences.append(out[0])  # [seq_len]
    return sequences


def train_draft_quick(
    target: nn.Module,
    draft_config: DFlashConfig,
    tokenizer,
    num_steps: int = 100,
    device: str = "cuda",
) -> Tuple[DFlashDraftModel, nn.Linear, nn.Module]:
    """Quick-train a DFlash draft model for benchmarking.

    Generates long sequences from target, pre-computes hidden states, then trains
    the draft model using cached data for fast iteration.

    Returns the trained draft model, context_proj, and context_norm.
    """
    from torchspec.models.dflash import DFlashModel

    draft = DFlashDraftModel(draft_config).to(device)

    # Load embedding from target
    draft.embed_tokens.weight.data.copy_(target.model.embed_tokens.weight.data)
    draft.freeze_embedding()
    draft = draft.to(torch.bfloat16)

    target_layer_ids = draft.target_layer_ids
    lm_head_weight = target.lm_head.weight.detach()

    dflash = DFlashModel(
        draft_model=draft,
        block_size=16,
        num_anchors=256,
        loss_decay_gamma=7.0,
        gradient_checkpointing=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01
    )
    # Cosine schedule with 10% warmup
    warmup_steps = max(1, num_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load diverse training sequences from wikitext dataset (much more diverse than generated text)
    seq_len_train = 256
    num_train_seqs = 100
    print(f"  Loading {num_train_seqs} training sequences ({seq_len_train} tokens each)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        all_text = " ".join([t for t in ds["text"] if len(t.strip()) > 50])
        tokens = tokenizer(all_text, return_tensors="pt", truncation=False)["input_ids"][0]
        train_seqs = []
        for i in range(0, len(tokens) - seq_len_train, seq_len_train):
            train_seqs.append(tokens[i : i + seq_len_train].to(device))
            if len(train_seqs) >= num_train_seqs:
                break
        print(f"  Got {len(train_seqs)} sequences of {seq_len_train} tokens from wikitext")
    except Exception as e:
        print(f"  Wikitext failed ({e}), falling back to target-generated sequences...")
        prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What is the capital of France and why is it important?",
            "Describe the process of photosynthesis.",
            "How does a neural network learn?",
        ]
        train_seqs_raw = _generate_training_sequences(
            target, tokenizer, prompts, gen_len=256, device=device,
        )
        train_seqs = train_seqs_raw

    # Pre-compute target hidden states for all sequences
    print(f"  Pre-computing target hidden states for {len(train_seqs)} sequences...")
    cached_data = []
    for i, seq in enumerate(train_seqs):
        input_ids = seq.unsqueeze(0) if seq.dim() == 1 else seq
        with torch.no_grad():
            out = target(input_ids, output_hidden_states=True)
        hidden_list = [out.hidden_states[lid + 1].detach() for lid in target_layer_ids]
        cached_data.append((input_ids, hidden_list))
        if (i + 1) % 20 == 0:
            print(f"    Cached {i+1}/{len(train_seqs)} sequences")
    print(f"  Cached {len(cached_data)} sequences with hidden states")

    print(f"Quick-training DFlash draft for {num_steps} steps...")
    draft.train()

    for step in range(num_steps):
        input_ids, hidden_list = cached_data[step % len(cached_data)]
        loss_mask = torch.ones(1, input_ids.shape[1], dtype=torch.float32, device=device)

        loss, acc = dflash(
            input_ids=input_ids,
            hidden_states_list=hidden_list,
            loss_mask=loss_mask,
            lm_head_weight=lm_head_weight,
        )

        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if (step + 1) % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Step {step+1}/{num_steps}: loss={loss.item():.4f}, acc={acc.item():.4f}, lr={lr_now:.6f}")

    draft.eval()
    return draft, draft.context_proj, draft.context_norm


def main():
    parser = argparse.ArgumentParser(description="DFlash Inference Benchmark")
    parser.add_argument("--target_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft_checkpoint", default=None,
                        help="Path to trained DFlash draft checkpoint")
    parser.add_argument("--draft_config", default="torchspec/config/dflash_draft_config.json")
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--quick_train_steps", type=int, default=1000,
                        help="If no checkpoint, quick-train for this many steps")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load target model
    print(f"Loading target model: {args.target_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    target.eval()
    print(f"Target model loaded ({sum(p.numel() for p in target.parameters()) / 1e9:.1f}B params)")

    # Load or train draft model
    with open(args.draft_config) as f:
        config_dict = json.load(f)
    draft_config = DFlashConfig(**config_dict)

    if args.draft_checkpoint and os.path.exists(args.draft_checkpoint):
        print(f"Loading DFlash checkpoint: {args.draft_checkpoint}")
        draft = DFlashDraftModel(draft_config).to(device).to(torch.bfloat16)
        state = torch.load(args.draft_checkpoint, map_location=device)
        draft.load_state_dict(state, strict=False)
        draft.eval()
        context_proj = draft.context_proj
        context_norm = draft.context_norm
    else:
        print("No checkpoint found, quick-training draft model...")
        draft, context_proj, context_norm = train_draft_quick(
            target, draft_config, tokenizer,
            num_steps=args.quick_train_steps, device=device,
        )

    target_layer_ids = draft.target_layer_ids
    print(f"Target layer IDs: {target_layer_ids}")
    print(f"Draft model: {sum(p.numel() for p in draft.parameters() if p.requires_grad) / 1e6:.1f}M trainable params")

    # Benchmark prompts
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function that implements binary search.",
        "What are the main causes of climate change?",
        "Describe how a transformer neural network works.",
        "What is the difference between TCP and UDP?",
        "Explain the process of making bread from scratch.",
        "How do vaccines work to protect against diseases?",
        "What are the key principles of object-oriented programming?",
        "Describe the solar system and its planets.",
        "What is blockchain technology and how does it work?",
        "Explain how a compiler works step by step.",
        "What are the benefits and risks of artificial intelligence?",
        "How does the human immune system fight infections?",
        "Describe the process of machine learning model training.",
        "What is quantum entanglement?",
    ][:args.num_prompts]

    print(f"\n{'='*60}")
    print(f"Benchmark: {len(prompts)} prompts, max_new_tokens={args.max_new_tokens}")
    print(f"{'='*60}\n")

    # ── Baseline: target-only ──
    baseline_times = []
    baseline_tokens = []
    if not args.skip_baseline:
        print("Running BASELINE (target-only autoregressive)...")
        # Warmup
        warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)
        generate_baseline(target, tokenizer, warmup_ids, 16, args.temperature)
        torch.cuda.synchronize()

        for i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            torch.cuda.synchronize()
            output, elapsed = generate_baseline(
                target, tokenizer, input_ids, args.max_new_tokens, args.temperature
            )
            torch.cuda.synchronize()
            num_new = output.shape[1] - input_ids.shape[1]
            baseline_times.append(elapsed)
            baseline_tokens.append(num_new)
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(prompts)}] {num_new} tokens in {elapsed:.2f}s "
                      f"({num_new/elapsed:.1f} tok/s)")

        avg_baseline_time = sum(baseline_times) / len(baseline_times)
        avg_baseline_tokens = sum(baseline_tokens) / len(baseline_tokens)
        avg_baseline_tps = sum(t/e for t, e in zip(baseline_tokens, baseline_times)) / len(prompts)
        print(f"\n  Baseline avg: {avg_baseline_tokens:.0f} tokens, "
              f"{avg_baseline_time:.2f}s, {avg_baseline_tps:.1f} tok/s\n")

    # ── DFlash speculative decoding ──
    print("Running DFLASH speculative decoding...")
    # Warmup
    warmup_ids = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)
    generate_dflash_spec(
        target, draft, context_proj, context_norm, tokenizer,
        warmup_ids, 16, args.block_size, target_layer_ids, args.temperature
    )
    torch.cuda.synchronize()

    dflash_times = []
    dflash_tokens = []
    all_acc_lens = []

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        torch.cuda.synchronize()
        output, elapsed, acc_lens = generate_dflash_spec(
            target, draft, context_proj, context_norm, tokenizer,
            input_ids, args.max_new_tokens, args.block_size,
            target_layer_ids, args.temperature
        )
        torch.cuda.synchronize()
        num_new = output.shape[1] - input_ids.shape[1]
        dflash_times.append(elapsed)
        dflash_tokens.append(num_new)
        all_acc_lens.extend(acc_lens)
        avg_tau = sum(acc_lens) / max(len(acc_lens), 1)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(prompts)}] {num_new} tokens in {elapsed:.2f}s "
                  f"({num_new/elapsed:.1f} tok/s, τ={avg_tau:.2f})")

    avg_dflash_time = sum(dflash_times) / len(dflash_times)
    avg_dflash_tokens = sum(dflash_tokens) / len(dflash_tokens)
    avg_dflash_tps = sum(t/e for t, e in zip(dflash_tokens, dflash_times)) / len(prompts)
    avg_tau = sum(all_acc_lens) / max(len(all_acc_lens), 1)

    print(f"\n  DFlash avg: {avg_dflash_tokens:.0f} tokens, "
          f"{avg_dflash_time:.2f}s, {avg_dflash_tps:.1f} tok/s, τ={avg_tau:.2f}\n")

    # ── Summary ──
    print(f"{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    if not args.skip_baseline:
        speedup = avg_baseline_time / avg_dflash_time if avg_dflash_time > 0 else 0
        print(f"  Baseline:  {avg_baseline_tps:.1f} tok/s")
        print(f"  DFlash:    {avg_dflash_tps:.1f} tok/s (τ={avg_tau:.2f})")
        print(f"  Speedup:   {speedup:.2f}x")
    else:
        print(f"  DFlash:    {avg_dflash_tps:.1f} tok/s (τ={avg_tau:.2f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

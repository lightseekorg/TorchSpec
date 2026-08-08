#!/usr/bin/env python3
"""Benchmark Eagle3 end-to-end training (forward + backward + optimizer step).

Measures:
  1. Memory efficiency: max sequence length via binary search
  2. Speed: full training step time at various sequence lengths

Gradient checkpointing is enabled by default. Use --no-grad-ckpt to disable.

Usage:
    python tools/benchmark_eagle3.py [--hidden-size 4096] [--vocab-size 151936]
"""

import argparse
import gc
import time
from typing import Tuple

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from torchspec.models.eagle3 import (
    Eagle3Model,
    compute_lazy_target_padded,
    compute_target_p_padded,
)
from torchspec.training.optimizer import BF16Optimizer


# ---------------------------------------------------------------------------
# Helpers (aligned with max_seq_search.py)
# ---------------------------------------------------------------------------
def clear_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def make_config(H, V, num_heads=32, num_kv_heads=8):
    config = LlamaConfig(
        hidden_size=H,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=H * 4,
        max_position_embeddings=131072,
        vocab_size=V,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_scaling=None,
        pretraining_tp=1,
        pad_token_id=0,
    )
    config.draft_vocab_size = V
    return config


def make_eagle3_model(
    config,
    length=7,
    gradient_checkpointing=True,
    attention_backend="flex_attention",
    device="cuda",
):
    draft_model = LlamaForCausalLMEagle3(config, attention_backend=attention_backend)
    draft_model = draft_model.to(device=device, dtype=torch.bfloat16)
    draft_model.freeze_embedding()
    model = Eagle3Model(
        draft_model,
        length=length,
        attention_backend=attention_backend,
        gradient_checkpointing=gradient_checkpointing,
    )
    model.train()
    return model


def make_optimizer(model):
    return BF16Optimizer(model.draft_model, lr=1e-4)


def reset_optimizer_state(optimizer: BF16Optimizer):
    """Reset optimizer state to free memory from momentum/variance buffers."""
    optimizer.optimizer.state.clear()


def create_synthetic_batch(B, T, H, V, device="cuda"):
    """Create synthetic training batch matching Eagle3 expected format."""
    input_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    loss_mask = (torch.rand(B, T, device=device) > 0.5).float()
    hidden_states = torch.randn(B, T, H * 3, device=device, dtype=torch.bfloat16)
    target_hidden_states = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target_hidden_states": target_hidden_states,
    }


def cleanup_batch(batch: dict):
    """Delete batch tensors and free memory."""
    for k in list(batch.keys()):
        del batch[k]


def run_eagle3_forward(model, batch):
    eagle3 = model.module if hasattr(model, "module") else model
    draft_model = eagle3.draft_model
    _, lm_head_weight, _ = draft_model.get_lm_head_params()
    t2d = draft_model.t2d if eagle3.vocab_pruning else None

    if t2d is not None:
        target = compute_target_p_padded(
            target_hidden_states=batch["target_hidden_states"],
            target_lm_head_weight=lm_head_weight,
            t2d=t2d,
            loss_mask=batch["loss_mask"],
            length=eagle3.length,
        )
    else:
        target = compute_lazy_target_padded(
            batch["target_hidden_states"],
            lm_head_weight,
            eagle3.length,
        )

    plosses, _, acces, _, _ = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target=target,
        loss_mask=batch["loss_mask"],
        hidden_states=batch["hidden_states"],
    )
    return plosses


def run_eagle3_backward_and_update(plosses, optimizer):
    """Run backward pass and optimizer step (matches eagle3_trainer.py)."""
    ploss_weight = [0.8**i for i in range(len(plosses))]
    ploss = sum(ploss_weight[i] * plosses[i] for i in range(len(plosses)))
    ploss.backward()
    optimizer.step()
    return ploss


def run_train_step(model, optimizer, B, T, H, V):
    """Full training step: forward + backward + optimizer step."""
    batch = create_synthetic_batch(B, T, H, V)
    plosses = run_eagle3_forward(model, batch)
    loss = run_eagle3_backward_and_update(plosses, optimizer)
    cleanup_batch(batch)
    return loss


# ---------------------------------------------------------------------------
# 1. Memory benchmark: binary search for max sequence length
# ---------------------------------------------------------------------------
def try_seq_len(model, optimizer, T, B, H, V, warmup_iters=2, test_iters=3) -> Tuple[bool, float]:
    reset_optimizer_state(optimizer)
    clear_memory()
    torch.cuda.reset_peak_memory_stats()

    try:
        for i in range(warmup_iters + test_iters):
            run_train_step(model, optimizer, B, T, H, V)
            if i == warmup_iters - 1:
                clear_memory()

        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        clear_memory()
        return True, peak_mem
    except torch.cuda.OutOfMemoryError:
        reset_optimizer_state(optimizer)
        clear_memory()
        return False, 0.0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            reset_optimizer_state(optimizer)
            clear_memory()
            return False, 0.0
        raise


def binary_search_max_seq(
    model, optimizer, B, H, V, lo=256, hi=131072, step=256, verbose=True
) -> Tuple[int, float]:
    lo = lo // step * step
    hi = hi // step * step
    best = lo
    best_mem = 0.0

    if verbose:
        print(f"  {'T':>8s}  {'status':>6s}  {'peak_mem':>10s}")

    while lo <= hi:
        mid = (lo + hi) // 2
        mid = mid // step * step
        if mid == 0:
            mid = step

        ok, peak_mem = try_seq_len(model, optimizer, mid, B, H, V)
        if verbose:
            status = "OK" if ok else "OOM"
            mem_str = f"{peak_mem:.2f} GB" if peak_mem > 0 else "N/A"
            print(f"  {mid:>8d}  {status:>6s}  {mem_str:>10s}")
        if ok:
            best = mid
            best_mem = peak_mem
            lo = mid + step
        else:
            hi = mid - step

    return best, best_mem


def benchmark_memory(
    B,
    H,
    V,
    config,
    length=7,
    step=256,
    gradient_checkpointing=True,
    attention_backend="flex_attention",
):
    ckpt_str = "ON" if gradient_checkpointing else "OFF"
    print("=" * 70)
    print("MEMORY BENCHMARK: max sequence length (binary search)")
    print(f"  B={B}, H={H}, V={V}, length={length}, step={step}")
    print(f"  gradient_checkpointing={ckpt_str}")
    print("=" * 70)

    print()
    clear_memory()
    model = make_eagle3_model(
        config,
        length=length,
        gradient_checkpointing=gradient_checkpointing,
        attention_backend=attention_backend,
    )
    optimizer = make_optimizer(model)
    max_seq, peak_mem = binary_search_max_seq(model, optimizer, B, H, V, step=step)
    del model, optimizer
    clear_memory()

    print()
    print(f"  Result: max_seq = {max_seq},  peak = {peak_mem:.2f} GB")
    print()
    return max_seq, peak_mem


# ---------------------------------------------------------------------------
# 2. Speed benchmark: full training step time
# ---------------------------------------------------------------------------
def time_eagle3(model, optimizer, B, T, H, V, warmup=3, repeats=10) -> Tuple[float, float]:
    reset_optimizer_state(optimizer)
    clear_memory()

    for _ in range(warmup):
        run_train_step(model, optimizer, B, T, H, V)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_train_step(model, optimizer, B, T, H, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    return sum(times) / len(times), peak_mem


def _try_time(model, optimizer, B, T, H, V, **kwargs):
    try:
        return time_eagle3(model, optimizer, B, T, H, V, **kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        reset_optimizer_state(optimizer)
        clear_memory()
        return float("inf"), 0.0


def benchmark_speed(
    B,
    H,
    V,
    config,
    seq_lens,
    length=7,
    gradient_checkpointing=True,
    attention_backend="flex_attention",
):
    ckpt_str = "ON" if gradient_checkpointing else "OFF"
    print("=" * 70)
    print("SPEED BENCHMARK: full training step (fwd + bwd + optimizer) (ms)")
    print(f"  B={B}, H={H}, V={V}, length={length}")
    print(f"  gradient_checkpointing={ckpt_str}")
    print("=" * 70)

    results = []

    for T in seq_lens:
        clear_memory()
        model = make_eagle3_model(
            config,
            length=length,
            gradient_checkpointing=gradient_checkpointing,
            attention_backend=attention_backend,
        )
        optimizer = make_optimizer(model)
        ms, mem = _try_time(model, optimizer, B, T, H, V)
        del model, optimizer
        clear_memory()

        results.append({"T": T, "ms": ms, "mem": mem})

        def _fmt(ms, mem):
            ms_s = f"{ms:.1f}" if ms != float("inf") else "OOM"
            mem_s = f"{mem:.1f}" if mem > 0 else "N/A"
            return f"{ms_s:>8s} ms {mem_s:>5s} GB"

        print(f"  T={T:>6d}: {_fmt(ms, mem)}")

    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark Eagle3 end-to-end training")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=7, help="Number of draft steps")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--mem-step", type=int, default=1024, help="Step size for memory binary search"
    )
    parser.add_argument(
        "--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["sdpa", "fa", "flex_attention"],
    )
    args = parser.parse_args()

    B = args.batch_size
    H = args.hidden_size
    V = args.vocab_size
    grad_ckpt = not args.no_grad_ckpt
    attention_backend = args.attention_backend

    config = make_config(H, V, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads)

    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {total / 1024**3:.1f} GB total, {free / 1024**3:.1f} GB free")
    print(f"Config: B={B}, H={H}, V={V}, length={args.length}")
    print(f"        num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}")
    print(f"        gradient_checkpointing={'ON' if grad_ckpt else 'OFF'}")
    print(f"        attention_backend={attention_backend}")
    print()

    # 1. Memory benchmark
    max_seq, peak_mem = benchmark_memory(
        B,
        H,
        V,
        config,
        length=args.length,
        step=args.mem_step,
        gradient_checkpointing=grad_ckpt,
        attention_backend=attention_backend,
    )

    # 2. Speed benchmark
    speed_lens = [s for s in [512, 1024, 2048, 4096, 8192, 16384, 32768] if s <= max_seq]
    if not speed_lens:
        speed_lens = [256, 512]
    speed_results = benchmark_speed(
        B,
        H,
        V,
        config,
        speed_lens,
        length=args.length,
        gradient_checkpointing=grad_ckpt,
        attention_backend=attention_backend,
    )

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Config: B={B}, H={H}, V={V}, length={args.length}")
    print(f"  Gradient checkpointing: {'ON' if grad_ckpt else 'OFF'}")
    print(f"  Max sequence length: {max_seq}")
    print(f"  Peak memory at max seq: {peak_mem:.2f} GB")
    print()
    print(f"  {'T':>8s}  {'step_ms':>10s}  {'peak_mem':>10s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}")
    for r in speed_results:
        ms_s = f"{r['ms']:.1f}" if r["ms"] != float("inf") else "OOM"
        mem_s = f"{r['mem']:.1f} GB" if r["mem"] > 0 else "N/A"
        print(f"  {r['T']:>8d}  {ms_s:>8s} ms  {mem_s:>10s}")
    print()


if __name__ == "__main__":
    main()

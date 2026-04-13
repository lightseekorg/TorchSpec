"""Micro-benchmarks for DFlash training optimizations.

Tests on a single GPU to measure kernel-level impact of:
1. QKV fusion (q/k/v_proj → single qkv_proj)
2. Gate-Up fusion (gate_proj + up_proj → single gate_up_proj)
3. micro_batch_size scaling (1 vs 2 vs 4)
4. torch.compile fullgraph mode
5. Different attention backends

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_dflash_opts.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dflash_config():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        hidden_size=4096,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=152064,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=32768,
        head_dim=128,
        layer_types=["full_attention"] * 5,
    )
    config.block_size = 16
    config.num_target_layers = 5
    config.dflash_config = {"mask_token_id": 151669, "target_layer_ids": None}
    return config


def bench_forward_backward(model, inputs, name, warmup=5, iters=20):
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        out = model(**inputs)
        if isinstance(out, tuple):
            out[0].sum().backward()
        else:
            out.sum().backward()
        model.zero_grad()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        out = model(**inputs)
        if isinstance(out, tuple):
            out[0].sum().backward()
        else:
            out.sum().backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters * 1000
    print(f"  {name}: {elapsed:.2f}ms/iter")
    return elapsed


def bench_qkv_fusion():
    """Compare separate Q/K/V projections vs fused QKV."""
    print("\n=== QKV Fusion ===")
    H = 4096
    KV_H = 1024  # 8 kv heads * 128 head_dim
    bsz, seq_len = 1, 2048

    # Separate (current DFlash)
    q_proj = nn.Linear(H, H, bias=False).cuda().bfloat16()
    k_proj = nn.Linear(H, KV_H, bias=False).cuda().bfloat16()
    v_proj = nn.Linear(H, KV_H, bias=False).cuda().bfloat16()

    # Fused
    qkv_proj = nn.Linear(H, H + 2 * KV_H, bias=False).cuda().bfloat16()

    x = torch.randn(bsz, seq_len, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Separate
    torch.cuda.synchronize()
    for _ in range(5):
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        (q.sum() + k.sum() + v.sum()).backward()
        x.grad = None
        q_proj.zero_grad()
        k_proj.zero_grad()
        v_proj.zero_grad()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        (q.sum() + k.sum() + v.sum()).backward()
        x.grad = None
        q_proj.zero_grad()
        k_proj.zero_grad()
        v_proj.zero_grad()
    torch.cuda.synchronize()
    sep_ms = (time.time() - t0) / 50 * 1000

    # Fused
    x2 = x.detach().requires_grad_(True)
    for _ in range(5):
        qkv = qkv_proj(x2)
        qkv.sum().backward()
        x2.grad = None
        qkv_proj.zero_grad()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        qkv = qkv_proj(x2)
        qkv.sum().backward()
        x2.grad = None
        qkv_proj.zero_grad()
    torch.cuda.synchronize()
    fused_ms = (time.time() - t0) / 50 * 1000

    print(f"  Separate Q/K/V: {sep_ms:.2f}ms")
    print(f"  Fused QKV:      {fused_ms:.2f}ms")
    print(f"  Speedup:        {sep_ms / fused_ms:.2f}x")


def bench_gate_up_fusion():
    """Compare separate gate/up projections vs fused."""
    print("\n=== Gate-Up Fusion ===")
    H = 4096
    INTER = 14336
    bsz, seq_len = 1, 2048

    gate = nn.Linear(H, INTER, bias=False).cuda().bfloat16()
    up = nn.Linear(H, INTER, bias=False).cuda().bfloat16()
    gate_up = nn.Linear(H, 2 * INTER, bias=False).cuda().bfloat16()

    x = torch.randn(bsz, seq_len, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Separate
    for _ in range(5):
        g = gate(x)
        u = up(x)
        (F.silu(g) * u).sum().backward()
        x.grad = None
        gate.zero_grad()
        up.zero_grad()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        g = gate(x)
        u = up(x)
        (F.silu(g) * u).sum().backward()
        x.grad = None
        gate.zero_grad()
        up.zero_grad()
    torch.cuda.synchronize()
    sep_ms = (time.time() - t0) / 50 * 1000

    # Fused
    x2 = x.detach().requires_grad_(True)
    for _ in range(5):
        gu = gate_up(x2)
        g2, u2 = gu.chunk(2, dim=-1)
        (F.silu(g2) * u2).sum().backward()
        x2.grad = None
        gate_up.zero_grad()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        gu = gate_up(x2)
        g2, u2 = gu.chunk(2, dim=-1)
        (F.silu(g2) * u2).sum().backward()
        x2.grad = None
        gate_up.zero_grad()
    torch.cuda.synchronize()
    fused_ms = (time.time() - t0) / 50 * 1000

    print(f"  Separate Gate+Up: {sep_ms:.2f}ms")
    print(f"  Fused Gate-Up:    {fused_ms:.2f}ms")
    print(f"  Speedup:          {sep_ms / fused_ms:.2f}x")


def bench_batch_size_scaling():
    """Test how DFlash forward scales with micro_batch_size."""
    print("\n=== Batch Size Scaling ===")
    from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel

    config = DFlashConfig(
        hidden_size=4096,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=152064,
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        num_target_layers=5,
        target_hidden_size=4096,
        target_num_hidden_layers=36,
    )
    model = DFlashDraftModel(config).cuda().bfloat16()
    seq_len = 2048
    block_size = 16
    n_blocks = 32

    for bsz in [1, 2, 4]:
        noise = torch.randn(bsz, n_blocks * block_size, 4096, device="cuda", dtype=torch.bfloat16)
        ctx = torch.randn(bsz, seq_len, 4096, device="cuda", dtype=torch.bfloat16)
        draft_pos = torch.arange(n_blocks * block_size, device="cuda").unsqueeze(0).expand(bsz, -1)
        ctx_pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

        draft_ids = torch.randint(0, 152064, (bsz, n_blocks * block_size), device="cuda")

        # Warmup
        for _ in range(3):
            out = model(
                context_feature=ctx,
                draft_input_ids=draft_ids,
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
                noise_embedding=noise,
            )
            out.sum().backward()
            model.zero_grad()

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            out = model(
                context_feature=ctx,
                draft_input_ids=draft_ids,
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
                noise_embedding=noise,
            )
            out.sum().backward()
            model.zero_grad()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 10 * 1000
        print(f"  batch_size={bsz}: {ms:.1f}ms total, {ms / bsz:.1f}ms/sample")


def bench_compile_modes():
    """Test torch.compile with different modes on DFlash draft model."""
    print("\n=== Compile Modes ===")
    from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel

    config = DFlashConfig(
        hidden_size=4096,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=152064,
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        num_target_layers=5,
        target_hidden_size=4096,
        target_num_hidden_layers=36,
    )
    seq_len = 2048
    block_size = 16
    n_blocks = 32
    bsz = 1

    noise = torch.randn(bsz, n_blocks * block_size, 4096, device="cuda", dtype=torch.bfloat16)
    ctx = torch.randn(bsz, seq_len, 4096, device="cuda", dtype=torch.bfloat16)
    draft_pos = torch.arange(n_blocks * block_size, device="cuda").unsqueeze(0).expand(bsz, -1)
    ctx_pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

    for mode_name, compile_fn in [
        ("no compile", lambda m: m),
        ("default", lambda m: torch.compile(m)),
        ("reduce-overhead", lambda m: torch.compile(m, mode="reduce-overhead")),
        ("max-autotune", lambda m: torch.compile(m, mode="max-autotune")),
    ]:
        model = DFlashDraftModel(config).cuda().bfloat16()
        model = compile_fn(model)

        draft_ids = torch.randint(0, 152064, (bsz, n_blocks * block_size), device="cuda")

        # Warmup (extra for compile)
        for _ in range(5):
            out = model(
                context_feature=ctx,
                draft_input_ids=draft_ids,
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
                noise_embedding=noise,
            )
            out.sum().backward()
            model.zero_grad()

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            out = model(
                context_feature=ctx,
                draft_input_ids=draft_ids,
                draft_position_ids=draft_pos,
                context_position_ids=ctx_pos,
                noise_embedding=noise,
            )
            out.sum().backward()
            model.zero_grad()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 20 * 1000
        print(f"  {mode_name:20s}: {ms:.2f}ms")
        del model


if __name__ == "__main__":
    print("DFlash Training Optimization Benchmarks")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    bench_qkv_fusion()
    bench_gate_up_fusion()
    bench_batch_size_scaling()
    bench_compile_modes()

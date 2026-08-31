#!/usr/bin/env python3
"""Compare masking strategies for anchored Eagle3 attention.

    python tools/bench_anchored_attention.py --seq 384 4096 16384 --anchors 128 256

Anchored attention is small (Q = num_anchors), so the usual block-sparse machinery can cost
more than the density it removes. This measures that trade-off directly.
"""

import argparse
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from torchspec.models.anchored_eagle3 import anchored_bool_mask
from torchspec.models.ops.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
)

_compiled_create_block_mask = None


def _compiled_builder():
    global _compiled_create_block_mask
    if _compiled_create_block_mask is None:
        _compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)
    return _compiled_create_block_mask


DEPTHS, HEADS, KV_HEADS, HEAD_DIM = 7, 32, 8, 128  # DEPTHS overridden by --depths
QK_DIM, V_DIM = HEAD_DIM, HEAD_DIM  # MLA splits these: 192 for Q/K, 128 for V
B16 = torch.bfloat16


def _mask_mod(anchors, keep, ctx_len, num_anchors):
    """BlockMask form of the anchored predicate, for the flex comparison only.

    The shipped model uses anchored_bool_mask; this exists so the benchmark can measure what
    the BlockMask path would cost.
    """

    def mask_mod(b, h, q_idx, kv_idx):
        anchor = anchors[b, q_idx]
        context = (kv_idx < ctx_len) & (kv_idx <= anchor)
        chain = (kv_idx >= ctx_len) & (((kv_idx - ctx_len) % num_anchors) == q_idx)
        return (context | chain) & keep[b, q_idx]

    return mask_mod


def _step(kind, anchors, keep, seq, num_anchors, query, cache, device):
    """One TTT unroll: an attention call per depth over a KV cache that grows by num_anchors."""
    mask_mod = _mask_mod(anchors, keep, seq, num_anchors)
    for key, value in cache:
        kv_len = key.shape[2]
        if kind == "sdpa":
            mask = anchored_bool_mask(anchors, keep, seq, num_anchors, kv_len)
            F.scaled_dot_product_attention(query, key, value, attn_mask=mask, enable_gqa=True)
        else:
            block = compile_friendly_create_block_mask(mask_mod, 1, 1, num_anchors, kv_len, device)
            compile_friendly_flex_attention(
                query=query, key=key, value=value, block_mask=block, enable_gqa=True
            )


def _dense_step(seq, query, cache, device):
    """Dense TTT attention: every position is a query, KV grows by seq each depth."""
    from torchspec.models.ops.flex_attention import eagle3_block_mask

    for depth, (key, value) in enumerate(cache):
        block = eagle3_block_mask(
            Q_LEN=seq, KV_LEN=key.shape[2], B=1, H=1, device=device, lck=depth
        )
        compile_friendly_flex_attention(
            query=query, key=key, value=value, block_mask=block, enable_gqa=True
        )


def measure(kind, seq, num_anchors, device, reps, warmup, layers=1):
    if kind == "dense-ttt":
        query = torch.randn(1, HEADS, seq, QK_DIM, device=device, dtype=torch.bfloat16)
        cache = [
            (
                torch.randn(1, KV_HEADS, seq * (d + 1), QK_DIM, device=device, dtype=B16),
                torch.randn(1, KV_HEADS, seq * (d + 1), V_DIM, device=device, dtype=B16),
            )
            for d in range(DEPTHS)
        ]
        run = lambda: [_dense_step(seq, query, cache, device) for _ in range(layers)]  # noqa: E731
    else:
        anchors = torch.sort(torch.randperm(seq - DEPTHS, device=device)[:num_anchors]).values
        anchors = anchors.unsqueeze(0)
        keep = torch.ones(1, num_anchors, dtype=torch.bool, device=device)
        query = torch.randn(1, HEADS, num_anchors, QK_DIM, device=device, dtype=torch.bfloat16)
        cache = [
            (
                torch.randn(1, KV_HEADS, seq + d * num_anchors, QK_DIM, device=device, dtype=B16),
                torch.randn(1, KV_HEADS, seq + d * num_anchors, V_DIM, device=device, dtype=B16),
            )
            for d in range(DEPTHS)
        ]
        args = (anchors, keep, seq, num_anchors, query, cache, device)
        # With more than one layer the depth-0 context K/V are no longer pointwise: layer L
        # needs layer L-1 outputs at every position, so depth 0 becomes a dense pass per layer.
        dense_q = torch.randn(1, HEADS, seq, QK_DIM, device=device, dtype=torch.bfloat16)
        dense_kv = cache[0]

        def run():
            for _ in range(layers):
                if layers > 1:
                    _dense_step(seq, dense_q, [dense_kv], device)
                _step(kind, *args)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(reps):
        run()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / reps * 1000
    return elapsed, torch.cuda.max_memory_allocated() / 1e9


KINDS = ["dense-ttt", "sdpa", "flex+blockmask"]


def main():
    global DEPTHS, KV_HEADS, QK_DIM, V_DIM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, nargs="+", default=[384, 4096, 16384])
    parser.add_argument("--anchors", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--depths", type=int, default=7, help="ttt_length")
    parser.add_argument("--layers", type=int, default=1, help="draft decoder layers")
    parser.add_argument(
        "--mla",
        action="store_true",
        help="MLA head shape: keys carry qk_nope+qk_rope (192) and values v_head_dim "
        "(128), across all heads rather than a smaller KV group",
    )
    parser.add_argument(
        "--kinds", nargs="+", default=KINDS, choices=KINDS, help="strategies to measure"
    )
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--kind",
        choices=KINDS,
        help="Measure one strategy and print 'seq anchors kind ms'. Without it, each strategy "
        "is measured in its own subprocess: compiling one warms inductor state the others "
        "would otherwise inherit, which silently equalises them.",
    )
    args = parser.parse_args()
    if args.mla:
        KV_HEADS, QK_DIM, V_DIM = HEADS, 192, 128
    DEPTHS = args.depths

    if args.kind:
        for seq in args.seq:
            for num_anchors in args.anchors:
                if num_anchors > seq - DEPTHS:
                    continue
                ms, peak = measure(
                    args.kind, seq, num_anchors, "cuda", args.reps, args.warmup, args.layers
                )
                print(f"RESULT {seq} {num_anchors} {args.kind} {ms:.3f} {peak:.3f}")
        return

    results = {}
    for kind in args.kinds:
        out = subprocess.run(
            [
                sys.executable,
                __file__,
                "--kind",
                kind,
                "--depths",
                str(args.depths),
                "--layers",
                str(args.layers),
                "--reps",
                str(args.reps),
                "--warmup",
                str(args.warmup),
                "--seq",
                *map(str, args.seq),
                "--anchors",
                *map(str, args.anchors),
                *(["--mla"] if args.mla else []),
            ],
            capture_output=True,
            text=True,
        )
        for line in out.stdout.splitlines():
            if line.startswith("RESULT"):
                _, seq, anch, k, ms, peak = line.split()
                results[(int(seq), int(anch), k)] = (float(ms), float(peak))

    header = f"{'seq':>7} {'anch':>5}  " + "".join(f"{k:>26}" for k in args.kinds)
    print(header)
    print(f"{'':>7} {'':>5}  " + "".join(f"{'ms / peak GB':>26}" for _ in args.kinds))
    for seq in args.seq:
        for num_anchors in args.anchors:
            row = {k: results.get((seq, num_anchors, k)) for k in args.kinds}
            if not any(row.values()):
                continue
            cells = "".join(
                f"{v[0]:>14.2f} /{v[1]:>8.2f}" if v else f"{'n/a':>26}" for v in row.values()
            )
            print(f"{seq:>7} {num_anchors:>5}  {cells}")


if __name__ == "__main__":
    main()

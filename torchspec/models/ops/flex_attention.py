# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
    or_masks,
)
from transformers.utils import is_torchdynamo_compiling

# DFlash's block-causal mask generates different mask_mod closures per step
# (varying anchor positions), causing frequent recompilation. Raise the limit
# to avoid constant re-tracing.
try:
    dynamo.config.recompile_limit = 128
except AttributeError:
    dynamo.config.cache_size_limit = 128

# Without ATEN fallback, inductor's GEMM autotuner can fail with
# NoValidChoicesError during FlexAttention backward (Issue 10).
if "ATEN" not in getattr(inductor_config, "max_autotune_gemm_backends", ""):
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"


# Reference Implementation https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py
class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled:
            self._compiled_flex_attention = torch.compile(
                flex_attention,
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    flex_attention_compiled = (
        WrappedFlexAttention()() if not is_torchdynamo_compiling() else flex_attention
    )
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


def compile_friendly_create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
):
    """Create block mask directly (no compilation wrapper).

    Matches SpecForge behavior — create_block_mask is fast enough without
    torch.compile, and compiling it adds overhead with torch 2.9.1.
    """
    return create_block_mask(
        mask_mod,
        B,
        H,
        Q_LEN,
        KV_LEN,
        device,
    )


def generate_eagle3_mask(seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0):
    """Return a mask_mod for the Eagle3 causal+suffix pattern."""
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def suffix_mask(b, h, q_idx, kv_idx):
        return (kv_idx >= Q_LEN) & ((kv_idx - q_idx) % Q_LEN == 0)

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod


def build_eagle3_block_mask(
    Q_LEN: int,
    KV_LEN: int,
    B: int = 1,
    H: int = 1,
    device: torch.device = "cuda",
    BLOCK_SIZE: int = 128,
) -> "BlockMask":
    """Build Eagle3 BlockMask analytically -- O(num_blocks) memory.

    create_block_mask materialises the full (Q_LEN, KV_LEN) boolean grid
    internally (~112 GB at Q=49K, KV=245K).  This function constructs the
    sparse kv_indices and q_indices tensors directly from the known Eagle3
    mask structure (causal first round + diagonal suffix rounds), reducing
    peak memory to a few MB.

    Requires Q_LEN, KV_LEN to be multiples of BLOCK_SIZE and KV_LEN to be
    an integer number of Q_LEN-sized rounds.  Use ``eagle3_block_mask``
    for the dispatching wrapper that falls back to create_block_mask when
    those preconditions don't hold.
    """
    assert Q_LEN % BLOCK_SIZE == 0 and KV_LEN % BLOCK_SIZE == 0
    assert KV_LEN % Q_LEN == 0, (
        "build_eagle3_block_mask requires KV_LEN to be a multiple of Q_LEN; "
        f"got Q_LEN={Q_LEN}, KV_LEN={KV_LEN}"
    )
    n_q = Q_LEN // BLOCK_SIZE
    n_kv = KV_LEN // BLOCK_SIZE
    n_rounds = KV_LEN // Q_LEN

    # KV blocks per Q row
    max_kv_per_row = n_q + (n_rounds - 1)
    kv_num = torch.zeros(B, H, n_q, dtype=torch.int32, device=device)
    kv_idx = torch.zeros(B, H, n_q, max_kv_per_row, dtype=torch.int32, device=device)

    for qi in range(n_q):
        col = 0
        for ki in range(qi + 1):
            kv_idx[:, :, qi, col] = ki
            col += 1
        for r in range(1, n_rounds):
            kv_idx[:, :, qi, col] = r * n_q + qi
            col += 1
        kv_num[:, :, qi] = col

    # Q blocks per KV column (transpose)
    max_q_per_col = n_q + 1
    q_num = torch.zeros(B, H, n_kv, dtype=torch.int32, device=device)
    q_idx = torch.zeros(B, H, n_kv, max_q_per_col, dtype=torch.int32, device=device)

    for ki in range(n_kv):
        r = ki // n_q
        pos = ki % n_q
        col = 0
        if r == 0:
            for qi in range(pos, n_q):
                q_idx[:, :, ki, col] = qi
                col += 1
        else:
            q_idx[:, :, ki, col] = pos
            col += 1
        q_num[:, :, ki] = col

    def mask_mod(b, h, q, kv):
        causal = (kv < Q_LEN) & (q >= kv)
        suffix = (kv >= Q_LEN) & ((kv - q) % Q_LEN == 0)
        return causal | suffix

    return BlockMask(
        seq_lengths=(Q_LEN, KV_LEN),
        kv_num_blocks=kv_num,
        kv_indices=kv_idx,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=q_num,
        q_indices=q_idx,
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE),
        mask_mod=mask_mod,
    )


def eagle3_block_mask(
    Q_LEN: int,
    KV_LEN: int,
    *,
    B: int = 1,
    H: int = 1,
    device: torch.device = "cuda",
    BLOCK_SIZE: int = 128,
    seq_lengths: torch.Tensor = None,
    lck: int = 0,
) -> "BlockMask":
    """Eagle3 block-mask dispatcher -- analytical when possible, fallback otherwise.

    Eagle3 training appends one full Q_LEN-sized round per step, so in normal
    training the analytical builder's preconditions
    ``(Q_LEN % BLOCK_SIZE == 0 and KV_LEN % Q_LEN == 0)`` always hold.  The
    create_block_mask fallback only triggers for tests/edge cases (tiny
    sequence lengths, non-aligned shapes), where its O(Q*KV) memory cost is
    irrelevant.

    Args:
        Q_LEN: query length (current round).
        KV_LEN: total KV length (cached + current).
        B: batch size for the BlockMask (broadcast-friendly when 1).
        H: head count for the BlockMask (broadcast-friendly when 1).
        device: target device.
        BLOCK_SIZE: flex_attention block size; defaults to 128.
        seq_lengths: per-batch sequence lengths.  Currently unused by the
            Eagle3 mask closure but accepted to mirror the legacy call site
            signature, and reserved for future variable-length variants.
        lck: number of completed rounds; only used to name the fallback
            mask_mod for debug clarity.

    Returns:
        A flex_attention BlockMask implementing the Eagle3 causal+suffix
        pattern.
    """
    use_analytical = (
        Q_LEN % BLOCK_SIZE == 0
        and KV_LEN % BLOCK_SIZE == 0
        and KV_LEN % Q_LEN == 0
    )
    if use_analytical:
        return build_eagle3_block_mask(
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            B=B,
            H=H,
            device=device,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Fallback for non-aligned shapes (typically only seen in tests).
    # generate_eagle3_mask's closure does not consume seq_lengths today, so
    # synthesise a reasonable default when the caller didn't supply one.
    # TODO: Remove the usage of uncompiled create_block_mask after
    # https://github.com/pytorch/pytorch/issues/160018
    creator = create_block_mask if Q_LEN <= 128 else compile_friendly_create_block_mask
    if seq_lengths is None:
        seq_lengths = torch.full((B,), KV_LEN, dtype=torch.int32, device=device)
    return creator(
        mask_mod=generate_eagle3_mask(
            seq_lengths=seq_lengths,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            lck=lck,
        ),
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )

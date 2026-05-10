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


def generate_eagle3_mask(Q_LEN: int, KV_LEN: int, lck: int = 0):
    """Eagle3 causal+suffix mask_mod.

    Note: to support packed sequences (multiple variable-length samples
    concatenated into one row), seq_lengths must be passed in here so the
    mask can clamp causal and suffix clauses to per-sample boundaries; for
    the current single-sample-per-row case the legacy seq_lengths clauses
    are tautological on every valid q row and are omitted.
    """

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def suffix_mask(b, h, q_idx, kv_idx):
        return (kv_idx >= Q_LEN) & ((kv_idx - q_idx) % Q_LEN == 0)

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod


def _build_eagle3_block_mask_tensors(
    Q_LEN: int,
    KV_LEN: int,
    B: int,
    H: int,
    BLOCK_SIZE: int,
    device: torch.device,
):
    """Return (kv_num, kv_idx, q_num, q_idx) for the Eagle3 BlockMask.

    Split out from BlockMask wrapping so it can be torch.compile'd: BlockMask
    carries Python-side closures and cannot be a compiled-graph return value.
    """
    n_q = Q_LEN // BLOCK_SIZE
    n_kv = KV_LEN // BLOCK_SIZE
    n_rounds = KV_LEN // Q_LEN

    # Row qi attends to causal cols 0..qi and one diagonal col per suffix round
    # at (col-qi)*n_q + qi; total qi + n_rounds entries per row.
    max_kv_per_row = n_q + n_rounds - 1
    qi = torch.arange(n_q, device=device, dtype=torch.int32)
    col = torch.arange(max_kv_per_row, device=device, dtype=torch.int32)
    qi_b = qi.unsqueeze(1)
    col_b = col.unsqueeze(0)
    is_causal = col_b <= qi_b
    causal_kv = col_b.expand(n_q, max_kv_per_row)
    suffix_kv = (col_b - qi_b) * n_q + qi_b
    kv_idx_2d = torch.where(is_causal, causal_kv, suffix_kv)
    valid = col_b < (qi_b + n_rounds)
    kv_idx_2d = torch.where(valid, kv_idx_2d, torch.zeros_like(kv_idx_2d))
    kv_num_1d = (qi + n_rounds).to(torch.int32)

    # Column ki: r=ki//n_q, pos=ki%n_q. r==0 -> q in [pos, n_q); r>=1 -> q==pos.
    max_q_per_col = n_q
    ki = torch.arange(n_kv, device=device, dtype=torch.int32)
    col_q = torch.arange(max_q_per_col, device=device, dtype=torch.int32)
    r = ki // n_q
    pos = ki % n_q
    r_b = r.unsqueeze(1)
    pos_b = pos.unsqueeze(1)
    col_q_b = col_q.unsqueeze(0)
    q_idx_2d = torch.where(r_b == 0, pos_b + col_q_b, pos_b.expand(n_kv, max_q_per_col))
    q_num_1d = torch.where(r == 0, n_q - pos, torch.ones_like(pos)).to(torch.int32)
    valid_q = col_q_b < q_num_1d.unsqueeze(1)
    q_idx_2d = torch.where(valid_q, q_idx_2d, torch.zeros_like(q_idx_2d))

    # flex_attention iterates these directly; force contiguous storage.
    kv_num = kv_num_1d.unsqueeze(0).unsqueeze(0).expand(B, H, n_q).contiguous()
    kv_idx = kv_idx_2d.unsqueeze(0).unsqueeze(0).expand(B, H, n_q, max_kv_per_row).contiguous()
    q_num = q_num_1d.unsqueeze(0).unsqueeze(0).expand(B, H, n_kv).contiguous()
    q_idx = q_idx_2d.unsqueeze(0).unsqueeze(0).expand(B, H, n_kv, max_q_per_col).contiguous()
    return kv_num, kv_idx, q_num, q_idx


# dynamic=True so KV_LEN growing per TTT step doesn't recompile; inductor's
# persistent cache amortises the one-off compile across runs.
_compiled_build_tensors = None


@torch.compiler.disable(recursive=False)
def _get_compiled_build_tensors():
    global _compiled_build_tensors
    if _compiled_build_tensors is None:
        _compiled_build_tensors = torch.compile(
            _build_eagle3_block_mask_tensors,
            dynamic=True,
            fullgraph=True,
        )
    return _compiled_build_tensors


def build_eagle3_block_mask(
    Q_LEN: int,
    KV_LEN: int,
    B: int = 1,
    H: int = 1,
    device: torch.device = "cuda",
    BLOCK_SIZE: int = 128,
) -> "BlockMask":
    """Build Eagle3 BlockMask analytically -- O(num_blocks) memory and time.

    create_block_mask materialises the full (Q_LEN, KV_LEN) boolean grid
    (~112 GB at Q=49K, KV=245K). This builds the sparse kv/q indices
    directly from the known Eagle3 structure (causal first round + diagonal
    suffix rounds), so peak memory drops to a few MB.

    Requires Q_LEN, KV_LEN multiples of BLOCK_SIZE and KV_LEN a multiple of
    Q_LEN. Use ``eagle3_block_mask`` for the dispatching wrapper that falls
    back to create_block_mask otherwise.
    """
    assert Q_LEN % BLOCK_SIZE == 0 and KV_LEN % BLOCK_SIZE == 0
    assert KV_LEN % Q_LEN == 0, (
        "build_eagle3_block_mask requires KV_LEN to be a multiple of Q_LEN; "
        f"got Q_LEN={Q_LEN}, KV_LEN={KV_LEN}"
    )

    # Skip the compiled path when nested inside another torch.compile graph.
    builder = (
        _build_eagle3_block_mask_tensors
        if is_torchdynamo_compiling()
        else _get_compiled_build_tensors()
    )
    kv_num, kv_idx, q_num, q_idx = builder(Q_LEN, KV_LEN, B, H, BLOCK_SIZE, device)

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
        lck: number of completed rounds; only used to name the fallback
            mask_mod for debug clarity.

    Returns:
        A flex_attention BlockMask implementing the Eagle3 causal+suffix
        pattern.
    """
    use_analytical = Q_LEN % BLOCK_SIZE == 0 and KV_LEN % BLOCK_SIZE == 0 and KV_LEN % Q_LEN == 0
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
    # TODO: Remove the usage of uncompiled create_block_mask after
    # https://github.com/pytorch/pytorch/issues/160018
    creator = create_block_mask if Q_LEN <= 128 else compile_friendly_create_block_mask
    return creator(
        mask_mod=generate_eagle3_mask(Q_LEN=Q_LEN, KV_LEN=KV_LEN, lck=lck),
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )

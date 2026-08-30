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

"""Anchor sampling shared by the anchored draft training paths."""

from typing import Optional, Tuple

import torch

from torchspec.utils.logging import logger


def _candidates(seq_len: int, loss_mask: torch.Tensor, block_size: int, device: torch.device):
    """Positions usable as anchors, as (num_candidates, valid, valid_counts, masked_indices).

    An anchor is only usable if its own position and the position right after it are both
    supervised: the block's first prediction target is ``anchor + 1``, so an isolated
    supervised token yields no gradient.
    """
    bsz = loss_mask.shape[0]
    max_anchor = max(seq_len - block_size, 0)
    num_candidates = min(max_anchor + 1, seq_len - 1)

    valid = (loss_mask[:, :num_candidates] > 0.5) & (loss_mask[:, 1 : num_candidates + 1] > 0.5)
    indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
    return num_candidates, valid, valid.sum(dim=1), torch.where(valid, indices, seq_len + 1)


def _empty(bsz: int, num_anchors: int, seq_len: int, block_size: int, device: torch.device):
    logger.warning(
        f"Sequence too short for anchor sampling (seq_len={seq_len}, "
        f"block_size={block_size}). Returning dummy anchors so loss is zero."
    )
    return (
        torch.zeros(bsz, num_anchors, dtype=torch.long, device=device),
        torch.zeros(bsz, num_anchors, dtype=torch.bool, device=device),
    )


def sample_anchor_positions(
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    block_size: int,
    device: torch.device,
    max_gap: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample anchor positions per sample; returns (anchors, keep_mask).

    Always returns exactly ``num_anchors`` anchor slots so that
    ``Q_LEN = num_anchors * block_size`` is constant across steps, preventing
    FlexAttention recompilation from shape changes. Samples with fewer valid positions
    leave the excess slots masked off.

    ``max_gap`` is how many supervised positions may sit *between* neighbouring anchors,
    so ``max_gap=0`` puts them side by side (5, 6, 7, ...). It matters because anchor ``a``
    supervises token ``a + d`` at unroll depth ``d``: a token is seen at depth ``d`` only if
    the anchor ``d`` positions behind it was also picked. Anchors with a gap of ``g`` leave
    each token ``block_size / (g + 1)`` of its depths, so once ``g`` reaches ``block_size``
    every token is pinned to a single depth.

        max_gap=0     neighbours adjacent; every token in the covered span is supervised
                      at all ``block_size`` depths, as in the dense unroll
        max_gap=k     every (k+1)-th supervised position, over a (k+1)x wider span
        max_gap=None  defaults to ``block_size``: one depth per token

    Gaps count supervised positions, not raw tokens, so slots are never spent on
    unsupervised regions and anchors stay token-adjacent inside a supervised span.

    Args:
        seq_len: sequence length
        loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
        device: torch device
        max_gap: supervised positions permitted between neighbours; None means block_size

    Returns:
        anchors: [B, num_anchors] — sampled anchor positions (sorted)
        keep_mask: [B, num_anchors] — True for valid sampled anchors
    """
    bsz = loss_mask.shape[0]
    if max(seq_len - block_size, 0) == 0:
        return _empty(bsz, num_anchors, seq_len, block_size, device)

    if max_gap is None:
        max_gap = block_size

    num_candidates, _, valid_counts, masked_indices = _candidates(
        seq_len, loss_mask, block_size, device
    )
    # Supervised positions in order; sentinels sort to the back and are masked off below.
    ordered = masked_indices.sort(dim=1).values

    # Spread the budget over the whole sample when it reaches, otherwise pack as tightly
    # as max_gap allows. The window wraps and its phase is drawn over every candidate, so
    # each one is picked with probability num_anchors / valid_count exactly as uniform
    # sampling would. Drawing the phase over the leftover slack instead would pick the
    # middle of a sample far more often than its ends.
    counts = valid_counts.unsqueeze(1).clamp(min=1)
    stride = (valid_counts // num_anchors).clamp(min=1, max=max_gap + 1).unsqueeze(1)
    offset = (torch.rand(bsz, 1, device=device) * counts).long()

    slot = torch.arange(num_anchors, device=device).unsqueeze(0)
    # num_anchors * stride <= valid_count by construction, so the wrap never collides.
    rank = (offset + slot * stride) % counts
    keep_mask = slot < valid_counts.unsqueeze(1)
    anchors = ordered.gather(1, rank.clamp(max=num_candidates - 1))
    return torch.where(keep_mask, anchors, 0), keep_mask

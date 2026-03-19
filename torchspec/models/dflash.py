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

"""DFlash training model: wraps the DFlash draft model with training-specific logic.

Handles anchor sampling, block-causal mask generation, noise input construction,
and cross-entropy loss with exponential decay weighting.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask
from torchspec.utils.logging import logger

_DFLASH_DEBUG_COUNTER = 0


def _sample_anchor_positions(
    loss_mask: torch.Tensor,
    num_anchors: int,
    block_size: int,
) -> torch.Tensor:
    """Sample random anchor positions within valid regions.

    For each batch element, samples `num_anchors` positions where:
      1. The position is in a valid region (loss_mask == 1)
      2. There is room for a full block after the anchor

    Falls back to uniformly sampling across all valid-end positions if no
    loss_mask positions satisfy the constraint.

    Args:
        loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
        num_anchors: number of anchors to sample per batch element
        block_size: size of each draft block

    Returns:
        anchor_positions: [B, num_anchors] — sampled anchor positions (sorted)
    """
    B, seq_len = loss_mask.shape
    device = loss_mask.device
    anchor_positions = torch.zeros(B, num_anchors, dtype=torch.long, device=device)

    for b in range(B):
        valid = loss_mask[b].bool()
        valid_end = seq_len - block_size
        if valid_end <= 0:
            continue

        valid_positions = torch.where(valid[:valid_end])[0]
        if len(valid_positions) == 0:
            # Fallback: sample uniformly from all positions that leave room for a block
            valid_positions = torch.arange(valid_end, device=device)
            if len(valid_positions) == 0:
                continue
            logger.debug(
                f"[DFlash] batch {b}: loss_mask has no valid anchors before position {valid_end}, "
                f"falling back to uniform sampling over {valid_end} positions"
            )

        if len(valid_positions) <= num_anchors:
            indices = torch.arange(len(valid_positions), device=device)
            indices = indices.repeat((num_anchors // len(valid_positions)) + 1)[:num_anchors]
        else:
            indices = torch.randperm(len(valid_positions), device=device)[:num_anchors]
        selected = valid_positions[indices]
        selected, _ = selected.sort()
        anchor_positions[b, : len(selected)] = selected

    return anchor_positions


def _create_position_ids(
    anchor_positions: torch.Tensor,
    block_size: int,
    ctx_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create position IDs for context and draft tokens.

    Context position IDs: [0, 1, 2, ..., ctx_len-1]
    Draft position IDs: each block starts from its anchor position
      Block i: [anchor_i, anchor_i+1, ..., anchor_i+block_size-1]

    Args:
        anchor_positions: [B, num_anchors]
        block_size: tokens per block
        ctx_len: length of context sequence

    Returns:
        context_position_ids: [B, ctx_len]
        draft_position_ids: [B, num_anchors * block_size]
    """
    B, num_anchors = anchor_positions.shape
    device = anchor_positions.device

    context_position_ids = torch.arange(ctx_len, dtype=torch.long, device=device)
    context_position_ids = context_position_ids.unsqueeze(0).expand(B, -1)

    offsets = torch.arange(block_size, dtype=torch.long, device=device)
    # [B, num_anchors, 1] + [block_size] → [B, num_anchors, block_size]
    draft_position_ids = anchor_positions.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)
    draft_position_ids = draft_position_ids.reshape(B, num_anchors * block_size)

    return context_position_ids, draft_position_ids


def _prepare_noise_input(
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct draft input: anchor token + (block_size-1) MASK tokens per block.

    Also constructs the target labels for loss computation.

    Args:
        input_ids: [B, seq_len] — full sequence token IDs
        anchor_positions: [B, num_anchors]
        block_size: tokens per block
        mask_token_id: ID of the MASK token

    Returns:
        draft_input_ids: [B, num_anchors * block_size] — anchor + MASKs
        draft_labels: [B, num_anchors * block_size] — ground truth tokens (-100 for anchor)
    """
    B, seq_len = input_ids.shape
    num_anchors = anchor_positions.shape[1]
    device = input_ids.device
    total_draft_len = num_anchors * block_size

    offsets = torch.arange(block_size, dtype=torch.long, device=device)

    # [B, num_anchors, block_size] — absolute positions for every token in every block
    all_positions = anchor_positions.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)
    all_positions_flat = all_positions.reshape(B, total_draft_len)

    in_bounds = all_positions_flat < seq_len
    gather_idx = all_positions_flat.clamp(max=seq_len - 1)
    all_tokens = torch.gather(input_ids, 1, gather_idx)

    pos_in_block = torch.arange(total_draft_len, device=device) % block_size
    is_anchor = (pos_in_block == 0).unsqueeze(0).expand(B, -1)

    draft_input_ids = torch.where(is_anchor, all_tokens, torch.tensor(mask_token_id, device=device))
    draft_labels = torch.where(~is_anchor & in_bounds, all_tokens, torch.tensor(-100, device=device))

    return draft_input_ids, draft_labels


def _create_dflash_mask_mod(
    num_anchors: int,
    block_size: int,
    anchor_positions: torch.Tensor,
    ctx_len: int,
):
    """Create a mask_mod function for DFlash block-causal attention.

    The mask defines which (Q, KV) pairs are visible:
      - KV indices [0, ctx_len) are context tokens
      - KV indices [ctx_len, ctx_len + num_anchors*block_size) are draft tokens

    Rules:
      1. Block i can see context tokens before its anchor position
      2. Block i has bidirectional attention within itself
      3. Blocks cannot see each other

    Args:
        num_anchors: number of anchor blocks
        block_size: tokens per block
        anchor_positions: [B, num_anchors] — anchor positions in context
        ctx_len: length of context sequence
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        # Which block does q belong to?
        q_block = q_idx // block_size

        # Is kv in context region?
        is_context = kv_idx < ctx_len

        # Context visibility: block q_block can see context up to (but not including)
        # its anchor position
        anchor_pos = anchor_positions[b, q_block]
        context_visible = is_context & (kv_idx < anchor_pos)

        # Draft region
        is_draft = kv_idx >= ctx_len
        kv_draft_idx = kv_idx - ctx_len
        kv_block = kv_draft_idx // block_size

        # Block-internal: bidirectional (same block)
        same_block = q_block == kv_block
        draft_visible = is_draft & same_block

        return context_visible | draft_visible

    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}"
    return dflash_mask_mod


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with exponential decay weighting
    """

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.gradient_checkpointing = gradient_checkpointing

        # Pre-compute loss decay weights: w(k) = exp(-(k-1)/gamma) for k=1..block_size-1
        decay_weights = torch.zeros(block_size)
        decay_weights[0] = 0.0  # anchor position: no loss
        for k in range(1, block_size):
            decay_weights[k] = math.exp(-(k - 1) / loss_decay_gamma)
        self.register_buffer("decay_weights", decay_weights, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        norm_weight: Optional[torch.Tensor] = None,
        norm_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full DFlash training forward pass.

        Args:
            input_ids: [B, seq_len] — full sequence token IDs
            hidden_states_list: list of [B, seq_len, D] from target model layers
            loss_mask: [B, seq_len] — 1 for valid positions
            lm_head_weight: [vocab_size, D] — frozen LM head weight
            norm_weight: [D] — final norm weight (if None, skip norm)
            norm_eps: epsilon for RMSNorm

        Returns:
            loss: scalar loss
            accuracy: scalar accuracy
        """
        global _DFLASH_DEBUG_COUNTER
        _DFLASH_DEBUG_COUNTER += 1
        _debug = _DFLASH_DEBUG_COUNTER <= 5

        B, seq_len = input_ids.shape
        device = input_ids.device

        if _debug:
            logger.info(
                f"[DFlash DBG step={_DFLASH_DEBUG_COUNTER}] "
                f"input_ids={input_ids.shape}, "
                f"hidden_states_list=[{', '.join(str(h.shape) for h in hidden_states_list)}], "
                f"loss_mask={loss_mask.shape} sum={loss_mask.sum().item():.0f}, "
                f"lm_head_weight={lm_head_weight.shape}"
            )

        # 1. Extract context features from target hidden states
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)

        # 2. Sample anchor positions
        actual_anchors = min(self.num_anchors, (seq_len - self.block_size))
        if actual_anchors <= 0:
            if _debug:
                logger.warning(
                    f"[DFlash DBG] EARLY RETURN: actual_anchors={actual_anchors}, "
                    f"seq_len={seq_len}, block_size={self.block_size}"
                )
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, torch.tensor(0.0, device=device)

        anchor_positions = _sample_anchor_positions(
            loss_mask=loss_mask,
            num_anchors=actual_anchors,
            block_size=self.block_size,
        )

        if _debug:
            nonzero_anchors = (anchor_positions > 0).sum().item()
            logger.info(
                f"[DFlash DBG] actual_anchors={actual_anchors}, "
                f"anchor_positions nonzero={nonzero_anchors}, "
                f"min={anchor_positions.min().item()}, max={anchor_positions.max().item()}"
            )

        # 3. Create position IDs
        context_position_ids, draft_position_ids = _create_position_ids(
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            ctx_len=seq_len,
        )

        # 4. Prepare noise input (anchor + MASK)
        draft_input_ids, draft_labels = _prepare_noise_input(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            mask_token_id=self.draft_model.mask_token_id,
        )

        if _debug:
            valid_labels = (draft_labels != -100).sum().item()
            logger.info(
                f"[DFlash DBG] draft_input_ids={draft_input_ids.shape}, "
                f"draft_labels={draft_labels.shape}, "
                f"valid_labels={valid_labels}/{draft_labels.numel()}, "
                f"mask_token_id={self.draft_model.mask_token_id}"
            )

        # 5. Create block-causal attention mask (FlexAttention requires CUDA)
        draft_len = actual_anchors * self.block_size
        kv_len = seq_len + draft_len

        block_mask = None
        if device.type == "cuda":
            mask_mod = _create_dflash_mask_mod(
                num_anchors=actual_anchors,
                block_size=self.block_size,
                anchor_positions=anchor_positions,
                ctx_len=seq_len,
            )

            block_mask = compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=B,
                H=1,
                Q_LEN=draft_len,
                KV_LEN=kv_len,
                device=device,
            )

        # 6. Draft model forward
        draft_hidden = self.draft_model(
            draft_input_ids=draft_input_ids,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
        )

        if _debug:
            logger.info(
                f"[DFlash DBG] draft_hidden={draft_hidden.shape} dtype={draft_hidden.dtype} "
                f"range=[{draft_hidden.min().item():.4f}, {draft_hidden.max().item():.4f}] "
                f"has_nan={draft_hidden.isnan().any().item()}"
            )

        # 7. Compute logits via frozen LM head
        logits = F.linear(draft_hidden, lm_head_weight)  # [B, draft_len, vocab]

        # 8. Compute weighted cross-entropy loss
        loss, accuracy = self._compute_loss(logits, draft_labels)

        return loss, accuracy

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy loss with exponential decay weighting.

        Args:
            logits: [B, draft_len, vocab_size]
            labels: [B, draft_len] — -100 for positions to ignore

        Returns:
            loss: weighted average loss
            accuracy: accuracy on valid positions
        """
        B, draft_len, vocab_size = logits.shape
        device = logits.device

        _debug = _DFLASH_DEBUG_COUNTER <= 5

        valid_mask = labels != -100
        if _debug:
            logger.info(
                f"[DFlash DBG _compute_loss] logits={logits.shape} "
                f"labels={labels.shape} valid={valid_mask.sum().item()}/{labels.numel()} "
                f"logits_range=[{logits.min().item():.4f}, {logits.max().item():.4f}] "
                f"logits_has_nan={logits.isnan().any().item()} "
                f"labels_unique_sample={labels[0, :20].tolist()}"
            )
        if not valid_mask.any():
            if _debug:
                logger.warning("[DFlash DBG] ALL labels are -100 → returning zero loss")
            zero = logits.sum() * 0.0
            return zero, torch.tensor(0.0, device=device)

        # Per-position CE loss
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=-100)
        ce_loss = ce_loss.reshape(B, draft_len)

        # Apply decay weights per block position
        block_pos_weights = self.decay_weights.to(device)
        # Tile the weights across all blocks: [block_size] → [num_blocks * block_size]
        num_blocks = draft_len // self.block_size
        weights = block_pos_weights.repeat(num_blocks)  # [draft_len]
        weights = weights.unsqueeze(0).expand(B, -1)  # [B, draft_len]

        # Weighted loss (only on valid positions)
        weighted_loss = (ce_loss * weights * valid_mask.float()).sum()
        weight_sum = (weights * valid_mask.float()).sum()

        if weight_sum > 0:
            loss = weighted_loss / weight_sum
        else:
            if _debug:
                logger.warning("[DFlash DBG] weight_sum=0 → returning zero loss")
            loss = logits.sum() * 0.0

        # Accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == labels) & valid_mask
            accuracy = correct.float().sum() / valid_mask.float().sum().clamp(min=1)

        if _debug:
            logger.info(
                f"[DFlash DBG] loss={loss.item():.6f}, accuracy={accuracy.item():.6f}, "
                f"weight_sum={weight_sum.item():.4f}, "
                f"ce_loss_valid_mean={ce_loss[valid_mask].mean().item():.4f}"
            )

        return loss, accuracy



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

Matches SpecForge's OnlineDFlashModel (specforge/core/dflash.py).
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask
from torchspec.utils.logging import logger

_VALID_DFLASH_LOSS_OBJECTIVES = {"decay", "dpace", "dspark"}


def _dpace_position_weights(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute detached D-PACE weights from per-position draft confidences."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights.to(dtype=confidences.dtype)


def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention is bidirectional (per SpecForge PR #427)
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing
    """
    num_anchors = anchor_positions.shape[1]

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < ctx_len
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}"
    return dflash_mask_mod


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with configurable position weighting
      - Per-position loss_mask application
    """

    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        loss_decay_gamma: float = 7.0,
        dspark_ce_alpha: float = 0.1,
        dspark_l1_alpha: float = 0.9,
        dspark_confidence_alpha: float = 1.0,
    ):
        super().__init__()
        loss_objective = loss_objective.lower()
        if loss_objective not in _VALID_DFLASH_LOSS_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_DFLASH_LOSS_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash loss objective {loss_objective!r}; expected one of {valid}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_objective = loss_objective
        self.dpace_alpha = dpace_alpha
        self.loss_decay_gamma = loss_decay_gamma
        self.dspark_ce_alpha = dspark_ce_alpha
        self.dspark_l1_alpha = dspark_l1_alpha
        self.dspark_confidence_alpha = dspark_confidence_alpha
        # Detached DSpark component losses from the last forward (best-effort logging).
        self._dspark_components: dict = {}

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchor positions per sample; returns (anchors, keep_mask).

        Always returns exactly ``self.num_anchors`` anchor slots so that
        ``Q_LEN = num_anchors * block_size`` is constant across steps,
        preventing FlexAttention recompilation from shape changes.  Samples
        with fewer valid positions use ``block_keep_mask=False`` for the
        excess slots (those blocks are skipped by the block-sparse kernel).

        Args:
            seq_len: sequence length
            loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
            device: torch device

        Returns:
            anchors: [B, num_anchors] — sampled anchor positions (sorted)
            keep_mask: [B, num_anchors] — True for valid sampled anchors
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        max_n = self.num_anchors

        if max_anchor == 0:
            logger.warning(
                f"Sequence too short for anchor sampling (seq_len={seq_len}, "
                f"block_size={bs}). Returning dummy anchors so loss is zero."
            )
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, seq_len + 1)

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        # Take up to num_anchors slots; pad with zeros if fewer valid positions
        take_n = min(max_n, gathered.shape[1])
        selected = gathered[:, :take_n].sort(dim=1).values
        if take_n < max_n:
            pad = torch.zeros(bsz, max_n - take_n, dtype=torch.long, device=device)
            selected = torch.cat([selected, pad], dim=1)
        anchors = selected

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(
            1
        ).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, 0)

        return anchors, keep_mask

    def _sample_anchor_positions_dspark(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DeepSpec-faithful anchor sampling for the dspark objective.

        Candidates require both the anchor token and its first target token to be valid
        (loss_mask[i] and loss_mask[i+1]), over positions [0, seq_len-2]. This avoids
        wasting anchor slots on blocks whose first prediction is masked — which the
        eval-mask cumprod would otherwise void entirely. Mirrors DeepSpec
        common.py:sample_anchor_positions. The DFlash (decay/dpace) path keeps the
        original anchor-only sampler above.
        """
        bsz = loss_mask.shape[0]
        max_n = self.num_anchors
        num_candidates = max(seq_len - 1, 0)
        if num_candidates == 0:
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep

        anchor_valid = loss_mask[:, :num_candidates] > 0.5
        first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
        valid = anchor_valid & first_target_valid
        valid_counts = valid.sum(dim=1)

        indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, torch.full_like(indices, seq_len + 1))
        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        if num_candidates < max_n:
            pad = torch.full(
                (bsz, max_n - num_candidates), seq_len + 1, dtype=gathered.dtype, device=device
            )
            gathered = torch.cat([gathered, pad], dim=1)
        anchors = gathered[:, :max_n].sort(dim=1).values
        keep = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(
            max=max_n
        )
        anchors = torch.where(keep, anchors, torch.zeros_like(anchors))
        return anchors, keep

    def _create_position_ids(
        self, anchor_positions: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position IDs for context and draft tokens."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        draft_position_ids = draft_position_ids.view(bsz, -1)

        return context_position_ids, draft_position_ids

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise embeddings: anchor token at block starts, MASK elsewhere.

        Matches SpecForge's OnlineDFlashModel._create_noise_embed().
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )

        return self.draft_model.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full DFlash training forward pass.

        Matches SpecForge's OnlineDFlashModel.forward().

        Returns:
            loss: scalar training loss (objective-weighted)
            accuracy: scalar accuracy (binary mask, no decay)
            loss_per_position: [block_size] mean loss at each within-block position
                (index 0 is the anchor slot and always 0; indices 1..B-1 are the
                predicted tokens at 1..B-1 steps past the anchor)
            acc_per_position: [block_size] mean accuracy at each within-block position
            count_per_position: [block_size] valid label count at each within-block
                position before loss decay is applied
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Extract context features from target hidden states
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)

        # 2. Sample anchor positions (DSpark uses a DeepSpec-faithful sampler; see
        # _sample_anchor_positions_dspark)
        if self.loss_objective == "dspark":
            anchor_positions, block_keep_mask = self._sample_anchor_positions_dspark(
                seq_len, loss_mask, device
            )
        else:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
        n_blocks = anchor_positions.shape[1]

        # 3. Create noise embeddings (anchor token + MASK tokens)
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        # 4. Create position IDs
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )

        # 5. Create block-causal attention mask
        draft_len = n_blocks * self.block_size
        kv_len = seq_len + draft_len

        block_mask = None
        if device.type == "cuda":
            mask_mod = _create_dflash_mask_mod(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.block_size,
            )
            block_mask = compile_friendly_create_block_mask(
                mask_mod=mask_mod,
                B=bsz,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=kv_len,
                device=device,
            )

        # 6. Draft model forward — pass embeddings directly
        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )

        # 7. Compute logits via frozen LM head
        logits = (
            self.draft_model.lm_head(draft_hidden)
            if hasattr(self.draft_model, "lm_head")
            else F.linear(draft_hidden, lm_head_weight)
        )

        # 8. Compute labels and weight mask.
        is_dspark = self.loss_objective == "dspark"
        if is_dspark:
            # DeepSpec next-token convention: slot j predicts anchor+j+1, all slots
            # supervised; eval_mask = contiguous supervised prefix per block (cumprod).
            label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
            label_indices = anchor_positions.unsqueeze(-1) + label_offsets
            valid_label_mask = label_indices < seq_len
            safe_label_indices = label_indices.clamp(max=seq_len - 1)
            safe_label_indices = torch.where(
                block_keep_mask.unsqueeze(-1),
                safe_label_indices,
                torch.zeros_like(safe_label_indices),
            )
            target_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
            )
            target_loss_mask_gathered = torch.gather(
                loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
            )
            eval_bool = (
                block_keep_mask.unsqueeze(-1) & valid_label_mask & (target_loss_mask_gathered > 0.5)
            )
            eval_bool = eval_bool.to(torch.int32).cumprod(dim=-1).bool()
            weight_mask = eval_bool.float()
        else:
            # DFlash same-position / infill convention (position k predicts anchor+k);
            # slot 0 is the anchor token and carries no loss.
            label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
            label_indices = anchor_positions.unsqueeze(-1) + label_offsets
            valid_label_mask = label_indices < seq_len
            safe_label_indices = label_indices.clamp(max=seq_len - 1)

            target_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
                2,
                safe_label_indices,
            )  # [B, n_blocks, block_size]

            # Weight mask: block validity × bounds × exclude anchor (pos 0) × loss_mask
            weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
            weight_mask = weight_mask * valid_label_mask.float()

            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

            # Gather original loss_mask at label positions
            original_loss_mask_gathered = torch.gather(
                loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
                2,
                safe_label_indices,
            )
            weight_mask = weight_mask * original_loss_mask_gathered

        # Capture binary mask BEFORE applying objective weights. Accuracy measures
        # "did we predict correctly?" uniformly across positions, while weighting
        # only shapes gradient contribution.
        binary_eval_mask = weight_mask.view(-1)

        # 8b. Teacher-forced previous-token ids for the Markov head: DSpark seeds slot 0
        # with the anchor token, DFlash with the same-position token (see DFlashMarkovHead).
        markov_head = getattr(self.draft_model, "markov_head", None)
        if is_dspark:
            anchor_token_ids = torch.gather(input_ids, 1, anchor_positions.clamp(0, seq_len - 1))
            prev_token_ids = torch.cat(
                [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
            )
        else:
            prev_token_ids = torch.cat([target_ids[:, :, :1], target_ids[:, :, :-1]], dim=-1)
        if markov_head is not None:
            logits_by_block = logits.view(bsz, n_blocks, self.block_size, -1)
            logits_by_block = markov_head.apply_block_logits(logits_by_block, prev_token_ids)
            logits = logits_by_block.reshape(bsz, n_blocks * self.block_size, -1)

        # 9. Loss
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss_per_token_by_position = loss_per_token.view(bsz, n_blocks, self.block_size)

        # Position decay weights: DSpark uses exp(-k/γ) (slot 0 = first prediction);
        # DFlash decay/dpace use exp(-(k-1)/γ) since their slot 0 is the masked anchor.
        decay_weights = None
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            if is_dspark:
                decay_weights = torch.exp(-k.float() / self.loss_decay_gamma)
            else:
                decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)

        if self.loss_objective == "dspark":
            loss = self._dspark_loss(
                flat_logits=flat_logits,
                loss_per_token=loss_per_token,
                draft_hidden=draft_hidden,
                weight_mask=weight_mask,
                decay_weights=decay_weights,
                label_indices=label_indices,
                target_last_hidden_states=target_last_hidden_states,
                lm_head_weight=lm_head_weight,
                bsz=bsz,
                n_blocks=n_blocks,
                prev_token_ids=prev_token_ids,
            )
        else:
            objective_weights = weight_mask
            if self.loss_objective == "decay" and decay_weights is not None:
                objective_weights = weight_mask * decay_weights
            elif self.loss_objective == "dpace":
                dpace_weights = torch.ones_like(weight_mask)
                if self.block_size > 1:
                    with torch.no_grad():
                        target_confidences = torch.exp(-loss_per_token_by_position[..., 1:].float())
                        dpace_pred_weights = _dpace_position_weights(
                            target_confidences,
                            self.dpace_alpha,
                        ).to(dtype=weight_mask.dtype)
                    dpace_weights[..., 1:] = dpace_pred_weights
                objective_weights = weight_mask * dpace_weights

            flat_weights = objective_weights.view(-1)
            valid_token_count = flat_weights.sum().clamp(min=1e-6)
            loss = (loss_per_token * flat_weights).sum() / valid_token_count

        # 10. Accuracy (using binary mask without decay)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count

            # Per-position-within-block metrics (index 0 = anchor, masked out;
            # indices 1..block_size-1 correspond to 1..B-1 tokens past the anchor).
            # Matches Eagle3's per-TTT-position breakdown semantically.
            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size)
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)

            loss_per_position = (
                loss_per_token.view(bsz, n_blocks, self.block_size) * binary_weights
            ).sum(dim=(0, 1)) / count_per_pos
            acc_per_position = (correct.view(bsz, n_blocks, self.block_size).float()).sum(
                dim=(0, 1)
            ) / count_per_pos

        return loss, accuracy, loss_per_position, acc_per_position, count_per_position

    def _dspark_loss(
        self,
        *,
        flat_logits: torch.Tensor,
        loss_per_token: torch.Tensor,
        draft_hidden: torch.Tensor,
        weight_mask: torch.Tensor,
        decay_weights: Optional[torch.Tensor],
        label_indices: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor],
        lm_head_weight: torch.Tensor,
        bsz: int,
        n_blocks: int,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DSpark objective: CE + TV/L1 distribution matching (+ optional confidence BCE).

        Faithful to DeepSpec (deepspec/modeling/dspark/loss.py): the three terms share a
        single token-pooled denominator (eval_mask x decay). The backward loss divides the
        summed numerators by the global denominator (all-reduced across ranks) and scales
        by world_size to cancel FSDP's mean gradient reduction — a true global token mean
        rather than a mean-of-per-rank-means; identical to a local mean at world_size==1.
        Target probabilities are built lazily from the target model's final hidden state
        and the frozen LM head, on supervised rows only (off-by-one: the target dist over
        the token at label position j comes from the target hidden at j-1). Logged
        per-component values use the local denominator.
        """
        if target_last_hidden_states is None:
            raise ValueError(
                "loss_objective='dspark' requires target_last_hidden_states (the target "
                "model's final pre-LM-head hidden state). Enable store_last_hidden_states "
                "on the inference engine and forward batch['last_hidden_states']"
            )

        block_size = self.block_size
        draft_len = n_blocks * block_size
        hidden = draft_hidden.size(-1)
        seq_len = target_last_hidden_states.size(1)

        # Single token-pooled weight = eval mask x optional decay (DeepSpec loss_weight_mask).
        weight = weight_mask if decay_weights is None else weight_mask * decay_weights
        flat_weight = weight.reshape(-1)
        local_den = flat_weight.sum()

        # CE numerator over all supervised positions.
        ce_num = (loss_per_token * flat_weight).sum()

        zero = flat_logits.new_zeros(())
        l1_num = zero
        conf_num = zero
        valid_idx = (flat_weight > 0).nonzero(as_tuple=True)[0]
        if valid_idx.numel() > 0:
            weight_valid = flat_weight[valid_idx]
            draft_logits_valid = flat_logits[valid_idx]  # [Nv, V] (markov-corrected)

            # Lazy target logits: gather target hidden at (label_index - 1), frozen LM head,
            # supervised rows only — never a full-vocab x all-slots tensor.
            target_pred_indices = (label_indices - 1).clamp(min=0, max=seq_len - 1)
            gather_idx = (
                target_pred_indices.reshape(bsz, draft_len)
                .unsqueeze(-1)
                .expand(bsz, draft_len, hidden)
            )
            aligned_target_hidden = torch.gather(target_last_hidden_states, 1, gather_idx)
            aligned_target_hidden = aligned_target_hidden.reshape(-1, hidden)[valid_idx]
            target_logits_valid = F.linear(
                aligned_target_hidden.to(lm_head_weight.dtype), lm_head_weight
            )

            draft_probs = torch.softmax(draft_logits_valid.float(), dim=-1)
            target_probs = torch.softmax(target_logits_valid.float(), dim=-1)
            l1_per_valid = (draft_probs - target_probs).abs().sum(dim=-1)
            l1_num = (l1_per_valid * weight_valid).sum()

            # Optional confidence head: BCE against the TV acceptance estimate (detached).
            confidence_head = getattr(self.draft_model, "confidence_head", None)
            if confidence_head is not None:
                accept_rate = (1.0 - 0.5 * l1_per_valid).clamp(0.0, 1.0).detach()
                draft_hidden_valid = draft_hidden.reshape(-1, hidden)[valid_idx]
                with_markov = bool(getattr(self.draft_model, "confidence_head_with_markov", False))
                if with_markov and prev_token_ids is not None:
                    prev_emb = self.draft_model.markov_head.get_prev_embeddings(prev_token_ids)
                    prev_emb = prev_emb.reshape(-1, prev_emb.size(-1))[valid_idx].to(
                        draft_hidden_valid.dtype
                    )
                    conf_features = torch.cat([draft_hidden_valid, prev_emb], dim=-1)
                else:
                    conf_features = draft_hidden_valid
                conf_logits_valid = confidence_head(conf_features).reshape(-1).float()
                bce = F.binary_cross_entropy_with_logits(
                    conf_logits_valid, accept_rate, reduction="none"
                )
                conf_num = (bce * weight_valid).sum()

        # Logged per-component values (local token means).
        den_eps = local_den + 1e-6
        self._dspark_components = {
            "dspark_ce": (ce_num / den_eps).detach(),
            "dspark_l1": (l1_num / den_eps).detach(),
            "dspark_conf": (conf_num / den_eps).detach(),
        }

        # Backward loss: global token-pooled mean x world_size (DeepSpec _build_loss).
        global_den = local_den.detach().clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
        global_den = global_den + 1e-6
        total = (
            (
                self.dspark_ce_alpha * ce_num
                + self.dspark_l1_alpha * l1_num
                + self.dspark_confidence_alpha * conf_num
            )
            / global_den
            * world_size
        )
        return total

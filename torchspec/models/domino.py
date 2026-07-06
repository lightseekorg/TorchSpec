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

"""Domino training wrapper.

Extends the DFlash training wrapper with the Domino causal-correction head and the
base-anchored training curriculum (arXiv:2605.29707, Sec. 4):

    L = (1 - lambda) * L_final + lambda * L_base

where L_base is the cross entropy on the parallel backbone's base logits and
L_final is the cross entropy on (base + causal correction). Lambda is annealed from
1 -> 0 over training so the backbone is strengthened first and the Domino head
gradually takes over the residual correction. The causal encoder is teacher
forced on ground-truth token embeddings (Sec. 4.1). Both losses reuse DFlash's
configurable position weighting.

Reuses DFlash's anchor sampling, block-causal mask, and label/weight construction
verbatim, so Domino's data contract is identical to DFlash.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from torchspec.models.dflash import DFlashModel, _create_dflash_mask_mod, _dpace_position_weights
from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask


class DominoModel(DFlashModel):
    """DFlash training wrapper + Domino causal correction + curriculum loss.

    ``curriculum_lambda`` is set by the trainer before each step (1 -> 0 schedule).
    At lambda=1 the loss is pure base (backbone); at lambda=0 pure final (with correction).
    """

    def __init__(self, *args, curriculum_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Mutable attribute: the trainer updates it each step (avoids threading kwargs
        # through the FSDP-wrapped forward).
        self.curriculum_lambda = curriculum_lambda

    def _weighted_objective_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        objective_weights: torch.Tensor,
        safe_label_indices: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Objective-weighted CE/L1 objective (fp32 for stability)."""
        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_targets = target_ids.reshape(-1)
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss_per_token = self.ce_loss_alpha * ce_per_token
        if self.l1_loss_alpha > 0:
            if last_hidden_states is None:
                raise ValueError(
                    "Domino L1 distillation (l1_loss_alpha > 0) requires target "
                    "last_hidden_states; set inference.store_last_hidden_states=true in the "
                    "run config."
                )
            bsz = target_ids.shape[0]
            tgt_idx = (safe_label_indices - 1).clamp(min=0)
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
            target_logits = F.linear(aligned_hidden, lm_head_weight).reshape(
                -1, lm_head_weight.size(0)
            )
            target_probs = torch.softmax(target_logits.float(), dim=-1)
            draft_probs = torch.softmax(flat_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
            loss_per_token = loss_per_token + self.l1_loss_alpha * l1_per_token
        w = objective_weights.reshape(-1)
        return (loss_per_token * w).sum() / w.sum().clamp(min=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Steps 1-6: identical to DFlash (reuse parent helpers).
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )

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

        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )

        # Base logits via the frozen target LM head (the DFlash backbone).
        base_logits = F.linear(draft_hidden, lm_head_weight)

        # Labels and binary weight mask are identical to DFlash.
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )

        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered
        binary_eval_mask = weight_mask.view(bsz, n_blocks, self.block_size)

        # Domino causal correction (teacher-forced on ground-truth tokens).
        hidden_dim = draft_hidden.shape[-1]
        h_blocked = draft_hidden.reshape(bsz * n_blocks, self.block_size, hidden_dim)
        gt_token_embeds = self.draft_model.embed_tokens(target_ids).reshape(
            bsz * n_blocks, self.block_size, hidden_dim
        )
        delta = self.draft_model.domino_correction(h_blocked, gt_token_embeds)
        delta = delta.reshape(bsz, n_blocks, self.block_size, -1).reshape(bsz, draft_len, -1)
        final_logits = base_logits.float() + delta.float()

        # Objective weighting mirrors DFlash. For D-PACE, final logits drive the
        # detached dynamic weights because they are the deployed Domino outputs.
        objective_weights = weight_mask
        if (
            self.loss_objective == "decay"
            and self.loss_decay_gamma is not None
            and self.loss_decay_gamma > 0
        ):
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            dpace_weights = torch.ones_like(weight_mask)
            if self.block_size > 1:
                with torch.no_grad():
                    final_loss_per_position = F.cross_entropy(
                        final_logits.reshape(-1, final_logits.size(-1)).float(),
                        target_ids.reshape(-1),
                        reduction="none",
                    ).view(bsz, n_blocks, self.block_size)
                    target_confidences = torch.exp(-final_loss_per_position[..., 1:].float())
                    dpace_pred_weights = _dpace_position_weights(
                        target_confidences,
                        self.dpace_alpha,
                    ).to(dtype=weight_mask.dtype)
                dpace_weights[..., 1:] = dpace_pred_weights
            objective_weights = weight_mask * dpace_weights

        # Base-anchored curriculum loss.
        loss_base = self._weighted_objective_loss(
            base_logits,
            target_ids,
            objective_weights,
            safe_label_indices,
            last_hidden_states,
            lm_head_weight,
        )
        loss_final = self._weighted_objective_loss(
            final_logits,
            target_ids,
            objective_weights,
            safe_label_indices,
            last_hidden_states,
            lm_head_weight,
        )
        lam = float(self.curriculum_lambda)
        loss = (1.0 - lam) * loss_final + lam * loss_base

        with torch.no_grad():
            correction_norm = delta.float().pow(2).mean().sqrt()
            correction_abs_mean = delta.float().abs().mean()

        # Metrics from the final logits (what gets deployed).
        with torch.no_grad():
            flat_final = final_logits.reshape(-1, final_logits.size(-1))
            flat_targets = target_ids.reshape(-1)
            loss_per_token = F.cross_entropy(flat_final.float(), flat_targets, reduction="none")
            bem = binary_eval_mask.reshape(-1)
            pred_ids = torch.argmax(flat_final, dim=-1)
            correct = (pred_ids == flat_targets) & (bem > 0.5)
            accuracy = correct.sum().float() / bem.sum().clamp(min=1e-6)

            bw = bem.view(bsz, n_blocks, self.block_size)
            count_per_position = bw.sum(dim=(0, 1))
            count_pp = count_per_position.clamp(min=1.0)
            loss_per_position = (loss_per_token.view(bsz, n_blocks, self.block_size) * bw).sum(
                dim=(0, 1)
            ) / count_pp
            acc_per_position = (
                correct.view(bsz, n_blocks, self.block_size).float().sum(dim=(0, 1)) / count_pp
            )

        aux_metrics = {
            "base_loss": loss_base.detach(),
            "final_loss": loss_final.detach(),
            "correction_norm": correction_norm.detach(),
            "correction_abs_mean": correction_abs_mean.detach(),
        }

        return loss, accuracy, loss_per_position, acc_per_position, count_per_position, aux_metrics

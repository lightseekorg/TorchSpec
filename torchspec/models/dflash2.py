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

"""DFlash2 training wrapper."""

import math

import torch
import torch.nn.functional as F

from torchspec.models.dflash import DFlashModel


class DFlash2Model(DFlashModel):
    def __init__(self, *args, selector_loss_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        selector_loss_alpha = float(selector_loss_alpha)
        if not math.isfinite(selector_loss_alpha) or selector_loss_alpha <= 0:
            raise ValueError(
                f"dflash2_selector_loss_alpha must be positive, got {selector_loss_alpha}"
            )
        self.selector_loss_alpha = selector_loss_alpha

        config = self.draft_model.config
        layer_types = list(getattr(config, "layer_types", []) or [])
        if layer_types and len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                "DFlash2 layer_types must contain one entry per draft layer, got "
                f"{len(layer_types)} for {config.num_hidden_layers} layers"
            )
        attention_types = set(layer_types or ["full_attention"])
        if not attention_types <= {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported DFlash2 layer types: {sorted(attention_types)}")
        explicit_causality = getattr(config, "is_causal", None)
        configured_window = getattr(config, "sliding_window", None)
        if "sliding_attention" in attention_types:
            if configured_window is None:
                raise ValueError(
                    "DFlash2 sliding_attention layers require an explicit positive sliding_window"
                )
            configured_window = int(configured_window)
            if configured_window < 1:
                raise ValueError(f"sliding_window must be positive, got {configured_window}")

        normalized_layer_types = layer_types or ["full_attention"] * config.num_hidden_layers
        self.layer_block_mask_options = []
        for layer_type in normalized_layer_types:
            uses_sliding_window = layer_type == "sliding_attention"
            self.layer_block_mask_options.append(
                {
                    "is_causal": (
                        uses_sliding_window
                        if explicit_causality is None
                        else bool(explicit_causality)
                    ),
                    "sliding_window": configured_window if uses_sliding_window else None,
                }
            )

        # Retain the uniform-policy attributes for callers that inspect them.
        self.attention_is_causal = self.layer_block_mask_options[0]["is_causal"]
        self.sliding_window = self.layer_block_mask_options[0]["sliding_window"]

    def _block_mask_options(self) -> dict:
        return self.layer_block_mask_options[0]

    def _block_mask_options_for_layer(self, layer_id: int) -> dict:
        return self.layer_block_mask_options[layer_id]

    def _compute_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        logits = super()._compute_logits(draft_hidden, lm_head_weight)
        logits = logits * self.draft_model.config.output_multiplier
        softcap = self.draft_model.config.final_logit_softcapping
        if softcap is not None and softcap > 0:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _extra_training_loss(
        self,
        draft_hidden: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        objective_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        batch, num_blocks, block_size = target_ids.shape
        hidden = draft_hidden.reshape(batch, num_blocks, block_size, -1)[..., 1:, :]
        unary_logits = logits.reshape(batch, num_blocks, block_size, -1)[..., 1:, :]
        predecessor_ids = target_ids[..., :-1]
        successor_ids = target_ids[..., 1:]

        scores, candidate_ids = self.draft_model.candidate_selector.score_candidates(
            hidden,
            unary_logits,
            predecessor_ids,
            training_successor_ids=successor_ids,
        )
        matches = candidate_ids == successor_ids.unsqueeze(-1)
        eligible_weights = objective_weights[..., 1:]
        eligible_weights = eligible_weights * (eligible_weights > 0).cumprod(dim=-1)
        target_indices = matches.to(torch.int64).argmax(dim=-1)
        selector_ce = F.cross_entropy(
            scores.reshape(-1, scores.shape[-1]),
            target_indices.reshape(-1),
            reduction="none",
        ).reshape_as(eligible_weights)
        selector_num = (selector_ce * eligible_weights).sum()
        selector_den = eligible_weights.sum().detach()
        return self.selector_loss_alpha * selector_num, {
            "selector_loss": (selector_num.detach(), selector_den),
        }

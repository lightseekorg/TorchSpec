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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from torchspec.models.ops.loss import (
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,
)
from torchspec.utils.tensor import padding


@dataclass
class PrecomputedTarget:
    """Pre-computed target probabilities (used with vocab pruning)."""

    target_p_padded: torch.Tensor  # (B, T + length, V_draft)
    position_mask: Optional[torch.Tensor] = None  # (B, T)


@dataclass
class LazyTarget:
    """Deferred target computation to avoid materializing (B, T, V_full)."""

    hidden_states_padded: torch.Tensor  # (B, T + length, D)
    lm_head_weight: torch.Tensor  # (V_full, D)


class Eagle3Model(nn.Module):
    def __init__(
        self,
        draft_model,
        length: int = 7,
        attention_backend="sdpa",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_pruning = draft_model.vocab_size != draft_model.target_vocab_size

    def _calculate_loss(
        self,
        hidden_states: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget],
        mask: torch.Tensor,
        idx: int,
        seq_length: int,
        norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        norm_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute forward-KL loss and accuracy for one TTT step.

        Both paths pass full (B*T, ...) flat views + valid_idx into the
        compiled function so torch.compile can fuse index_select with
        subsequent ops, avoiding separate (N_valid, V) copies outside.

        - PrecomputedTarget (vocab pruning): compiled_forward_kl_loss
          with pre-computed target probs.
        - LazyTarget (no pruning): compiled_forward_kl_loss_from_hs
          computes target softmax inside the compiled graph.
        """
        valid_idx = mask.flatten().nonzero().squeeze(-1)
        if valid_idx.numel() == 0:
            # FSDP requires every trainable param to participate in gradient
            # all-reduce/reduce-scatter.
            total = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad)
            zero = total * 0.0
            return zero, zero.detach()
        # Important as it prevents recompilation.
        torch._dynamo.maybe_mark_dynamic(valid_idx, 0)
        hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

        if isinstance(target, PrecomputedTarget):
            target_p_step = target.target_p_padded[:, idx : idx + seq_length, :]
            tp_flat = target_p_step.reshape(-1, target_p_step.shape[-1])
            args = (hs_flat, tp_flat, valid_idx, norm_weight, lm_head_weight, norm_eps)
            if self.gradient_checkpointing and self.training:
                return torch_checkpoint(
                    compiled_forward_kl_loss,
                    *args,
                    use_reentrant=False,
                )
            return compiled_forward_kl_loss(*args)
        else:
            # lazy
            ths_flat = target.hidden_states_padded[:, idx : idx + seq_length, :].reshape(
                -1, target.lm_head_weight.shape[-1]
            )
            args = (
                hs_flat,
                ths_flat,
                valid_idx,
                norm_weight,
                lm_head_weight,
                target.lm_head_weight,
                norm_eps,
            )
            if self.gradient_checkpointing and self.training:
                return torch_checkpoint(
                    compiled_forward_kl_loss_from_hs,
                    *args,
                    use_reentrant=False,
                )
            return compiled_forward_kl_loss_from_hs(*args)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget],
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        norm_weight, lm_head_weight, norm_eps = self.draft_model.get_lm_head_params()

        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # position_mask (vocab pruning) is a subset of loss_mask that further
        # filters to tokens whose target argmax falls in the draft vocab.
        if isinstance(target, PrecomputedTarget) and target.position_mask is not None:
            mask = target.position_mask
        else:
            mask = loss_mask

        plosses = []
        vlosses = []
        acces = []
        cache_keys = None
        cache_values = None

        # Clamp multimodal placeholder IDs (hash-based pad values from SGLang)
        # to valid vocab range before embedding lookup.
        input_ids = input_ids.clamp(min=0, max=self.draft_model.target_vocab_size - 1)

        for idx in range(self.length):
            is_last = idx == self.length - 1

            inputs_embeds = self.draft_model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            if self.gradient_checkpointing and self.training:
                hidden_states_out, cache_keys, cache_values = torch_checkpoint(
                    self.draft_model.backbone,
                    inputs_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cache_keys,
                    cache_values,
                    True,  # use_cache
                    use_reentrant=False,
                )
            else:
                hidden_states_out, cache_keys, cache_values = self.draft_model.backbone(
                    input_embeds=inputs_embeds,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_keys=cache_keys,
                    cache_values=cache_values,
                    use_cache=True,
                )

            hidden_states = hidden_states_out

            loss, acc = self._calculate_loss(
                hidden_states=hidden_states,
                target=target,
                mask=mask,
                idx=idx,
                seq_length=seq_length,
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_eps=norm_eps,
            )
            plosses.append(loss)
            acces.append(acc)

            if not is_last:
                input_ids = padding(input_ids, left=False)
                mask = padding(mask, left=False)
        return plosses, vlosses, acces


@torch.no_grad()
def compute_target_p_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    t2d: torch.Tensor,
    loss_mask: torch.Tensor,
    length: int,
    chunk_size: int = 4096,
) -> PrecomputedTarget:
    target_lm_head_weight = target_lm_head_weight.detach()
    pruned_weight = target_lm_head_weight[t2d]  # (V_draft, D)

    B, T, _D = target_hidden_states.shape
    loss_mask_bool = loss_mask.bool()

    valid_flat_idx = loss_mask_bool.reshape(-1).nonzero(as_tuple=True)[0]
    valid_hs = target_hidden_states.reshape(-1, _D)[valid_flat_idx]  # (N_valid, D)

    position_mask_flat = torch.zeros(B * T, device=target_hidden_states.device, dtype=torch.float)
    for i in range(0, valid_hs.shape[0], chunk_size):
        chunk_hs = valid_hs[i : i + chunk_size]
        chunk_argmax = F.linear(chunk_hs, target_lm_head_weight).argmax(-1)
        in_draft = t2d[chunk_argmax]
        position_mask_flat[valid_flat_idx[i : i + chunk_size]] = in_draft.float()
    position_mask = position_mask_flat.reshape(B, T)

    target_logits_pruned = F.linear(target_hidden_states, pruned_weight)
    target_p = F.softmax(target_logits_pruned.float(), dim=-1)
    target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)

    return PrecomputedTarget(target_p_padded, position_mask)


def compute_lazy_target_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    length: int,
) -> LazyTarget:
    """Build a LazyTarget that defers softmax to the forward loop.

    Used for non-pruning cases to avoid materializing the full
    (B, T, V_full) target probability tensor.
    """
    return LazyTarget(
        hidden_states_padded=F.pad(target_hidden_states, (0, 0, 0, length), value=0.0),
        lm_head_weight=target_lm_head_weight.detach(),
    )

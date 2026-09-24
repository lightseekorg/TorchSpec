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

"""Batch-first PyTorch wrapper around a native Megatron MTP layer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SequenceFirstEmbedding(nn.Module):
    """Adapt an ``nn.Embedding`` to the keyword, sequence-first call Megatron makes."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, *, input_ids: torch.Tensor, position_ids=None) -> torch.Tensor:
        return self.embedding(input_ids).transpose(0, 1).contiguous()


class MegatronMTPModel(nn.Module):
    """One Megatron ``MultiTokenPredictionLayer`` with the target's frozen embedding and head.

    Returns MTP hidden states; the caller computes the loss with
    ``compute_logits`` and its own objective.
    """

    def __init__(
        self,
        *,
        mtp_layer: nn.Module,
        embedding: nn.Embedding,
        lm_head: nn.Linear,
        rotary_embedding: nn.Module,
    ) -> None:
        super().__init__()
        self.mtp_layer = mtp_layer
        self.embedding = SequenceFirstEmbedding(embedding).requires_grad_(False)
        self.lm_head = lm_head.requires_grad_(False)
        self.rotary_embedding = rotary_embedding

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map unpadded ``[B, S]`` tokens and ``[B, S, H]`` target states to ``[B, S, H]``.

        Row ``t`` of the output combines ``last_hidden_states[:, t]`` with the
        embedding of ``input_ids[:, t + 1]``; the last row sees a wrapped token.
        """

        if input_ids.shape != last_hidden_states.shape[:2]:
            raise ValueError("input_ids and last_hidden_states must share [B, S]")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(
                input_ids
            )
        hidden_states, _, _, _ = self.mtp_layer(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=last_hidden_states.transpose(0, 1).contiguous(),
            attention_mask=None,  # Megatron's causal mask
            rotary_pos_emb=self.rotary_embedding(input_ids.shape[1]),
            embedding=self.embedding,
        )
        return hidden_states.transpose(0, 1).contiguous()

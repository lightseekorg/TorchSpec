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

"""Domino draft model: DFlash parallel backbone + a lightweight causal-correction head.

Domino (arXiv:2605.29707) decouples autoregressive modelling from parallel
draft execution. It keeps the cheap block-parallel DFlash backbone (which produces
base logits via the frozen target LM head) and adds a small "Domino head" that
injects token-dependence:

    logits_i = base_logits_i + Delta_i,  Delta_i = g(z_i, s_{i-1})

where s_{i-1} = GRU(E(t_{<i})) summarizes previously drafted tokens. Applying the
correction in logit space - rather than feeding it back through the expensive
backbone - keeps the expensive backbone computation parallel and restricts the
causal branch to a cheap low-rank update.

This module defines the model; training-time logic (anchor sampling, masks, the
base-anchored curriculum loss) lives in torchspec/models/domino.py (DominoModel).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel


class DominoConfig(DFlashConfig):
    """Domino config = DFlash config + Domino head dimensions."""

    model_type = "domino"

    def __init__(
        self,
        gru_hidden_size: int = 1024,
        correction_rank: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gru_hidden_size = gru_hidden_size
        self.correction_rank = correction_rank


class DominoDraftModel(DFlashDraftModel):
    """DFlash backbone + Domino causal-correction head.

    Adds (trainable) on top of the frozen-embedding DFlash backbone:
      causal_gru: GRU causal encoder over draft-token embeddings (teacher-forced)
      correction_w1 / correction_w2: low-rank (rank r) logit-space residual head
    """

    config_class = DominoConfig

    def __init__(self, config: DominoConfig):
        super().__init__(config)
        self.gru_hidden_size: int = getattr(config, "gru_hidden_size", 1024)
        self.correction_rank: int = getattr(config, "correction_rank", 256)

        # Causal encoder: summarizes preceding draft-token embeddings.
        self.causal_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_size,
            batch_first=True,
        )

        # Low-rank correction head: [H || S_gru] -> r -> vocab (logit-space residual).
        self.correction_w1 = nn.Linear(
            config.hidden_size + self.gru_hidden_size,
            self.correction_rank,
            bias=False,
        )
        self.correction_w2 = nn.Linear(
            self.correction_rank,
            config.vocab_size,
            bias=False,
        )

    def domino_correction(
        self,
        draft_hidden_blocked: torch.Tensor,
        gt_token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Domino logit-space residual correction for one or more blocks.

        Args:
            draft_hidden_blocked: [N, block, D] - backbone hidden states
            gt_token_embeds: [N, block, D] - embeddings of the ground-truth tokens
                at block positions (teacher forcing)

        Returns:
            delta_logits: [N, block, vocab]. Delta at position 0 uses s_{-1}=0.
        """
        # CUDA RNNs are happiest in fp32; run the (small) GRU there, then cast back.
        gru_dtype = self.causal_gru.weight_ih_l0.dtype
        states, _ = self.causal_gru(gt_token_embeds.to(gru_dtype))

        # Delta_i = g(z_i, s_{i-1}); position 0 sees no tokens before it.
        s_prev = torch.cat([torch.zeros_like(states[:, :1]), states[:, :-1]], dim=1)
        s_prev = s_prev.to(draft_hidden_blocked.dtype)
        x = torch.cat([draft_hidden_blocked, s_prev], dim=-1)
        return self.correction_w2(F.silu(self.correction_w1(x)))

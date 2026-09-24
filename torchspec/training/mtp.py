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

"""Hard-label cross entropy for an MTP layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MTPObjective:
    loss: torch.Tensor
    num_targets: int
    logits: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None


def compute_mtp_objective(
    model,
    hidden_states: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    normalizer: Optional[float] = None,
    keep_logits: bool = False,
) -> MTPObjective:
    """Score MTP row ``t`` (inputs ``h_t`` and ``x[t+1]``) against the hard label ``x[t+2]``.

    Only targets with ``loss_mask`` set count; the last two rows have no target.
    The loss is the mean CE over valid targets, or the summed CE divided by
    ``normalizer`` (e.g. the global batch's valid-target count when
    accumulating gradients over micro-batches).
    """

    valid = loss_mask[:, 2:].bool()
    targets = input_ids[:, 2:][valid]
    if targets.numel() == 0:
        return MTPObjective(loss=hidden_states.sum() * 0.0, num_targets=0)
    logits = model.compute_logits(hidden_states[:, :-2][valid])
    if normalizer is None:
        loss = F.cross_entropy(logits.float(), targets)
    else:
        loss = F.cross_entropy(logits.float(), targets, reduction="sum") / normalizer
    if not keep_logits:
        return MTPObjective(loss=loss, num_targets=targets.numel())
    return MTPObjective(
        loss=loss, num_targets=targets.numel(), logits=logits.detach().float(), targets=targets
    )

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

"""Expert-Parallel gradient utilities: EP-aware gradient sync and grad-norm clip.

Why this is needed: with EP, routed-expert parameters are UNIQUE per EP rank
(rank r owns experts ``[r*E/ep : (r+1)*E/ep]``), whereas every other parameter is
data-parallel REPLICATED. So they need different treatment:

  * non-expert grads: all-reduce(AVG) over the data-parallel group (each rank saw
    a different micro-batch). After this they are identical on every rank.
  * expert grads: NOT averaged across the EP group — each expert lives on exactly
    one rank (for ep_size == world_size, i.e. ep_fsdp == 1), and that rank already
    received, via the all-to-all dispatch, every token globally routed to it. The
    gradient is therefore already complete and correct locally.

For the global grad NORM, the expert contributions live on different ranks, so
their sum-of-squares must be all-reduced(SUM) over the EP group; the (replicated)
non-expert contribution is counted once.

Scope: this module implements and is validated for ``ep_size == world_size``
(ep_fsdp == 1). The general ``ep_fsdp > 1`` case (expert replicas across a second
mesh dim) additionally averages expert grads over the ep_fsdp group; that hook is
marked below.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


def is_ep_param(p: torch.Tensor) -> bool:
    return getattr(p, "_is_ep", False)


def split_ep_params(params: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Partition params into (expert_params, non_expert_params) by the ``_is_ep`` tag."""
    ep = [p for p in params if is_ep_param(p)]
    non_ep = [p for p in params if not is_ep_param(p)]
    return ep, non_ep


def sync_gradients_ep(
    params: List[torch.Tensor],
    dp_group=None,
    ep_fsdp_group=None,
) -> None:
    """All-reduce(AVG) gradients with EP awareness (call after backward, before step).

    * non-expert grads -> averaged over ``dp_group`` (the full data-parallel world).
    * expert grads     -> averaged over ``ep_fsdp_group`` only (no-op when that group
                          has size 1, i.e. ep_size == world_size).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        from torch.distributed.tensor import DTensor
    except Exception:  # pragma: no cover
        DTensor = ()  # type: ignore
    ep_params, non_ep_params = split_ep_params(params)

    for p in non_ep_params:
        # Skip FSDP-managed (DTensor) grads — FSDP already reduce-scatters them. This
        # also correctly skips FSDP-sharded experts whose ``_is_ep`` tag was dropped
        # when FSDP converted them to DTensors (they'd otherwise look "non-expert").
        if p.grad is not None and not isinstance(p.grad, DTensor):
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=dp_group)

    if ep_fsdp_group is not None and dist.get_world_size(ep_fsdp_group) > 1:
        for p in ep_params:
            if p.grad is not None and not isinstance(p.grad, DTensor):
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=ep_fsdp_group)


def clip_grad_norm_ep(
    params: List[torch.Tensor],
    max_norm: float,
    ep_group=None,
    ep_mask: Optional[List[bool]] = None,
    norm_type: float = 2.0,
    ep_extra_group=None,
) -> torch.Tensor:
    """EP-aware global grad-norm clipping (returns the pre-clip global grad norm).

    The total norm is ``sqrt(||non_ep||^2 + sum_over_EP_ranks ||ep_local||^2)``: the
    replicated non-expert grads are counted once (identical on all ranks), and the
    expert sum-of-squares is all-reduced(SUM) over ``ep_group`` because each rank
    holds a distinct slice of experts. When experts are additionally FSDP-sharded
    over an ``ep_fsdp`` mesh (ep_size < world_size), pass that group as
    ``ep_extra_group`` so the expert sum-of-squares is also reduced across the
    shards. DTensor grads (FSDP-managed) are reduced over their local shard.
    """
    try:
        from torch.distributed.tensor import DTensor
    except Exception:  # pragma: no cover
        DTensor = ()  # type: ignore

    if ep_mask is None:
        ep_mask = [is_ep_param(p) for p in params]

    grads = [(p.grad, m) for p, m in zip(params, ep_mask) if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    device = grads[0][0].to_local().device if isinstance(grads[0][0], DTensor) else grads[0][0].device
    non_ep_sq = torch.zeros((), device=device, dtype=torch.float32)
    ep_sq = torch.zeros((), device=device, dtype=torch.float32)
    for g, is_ep in grads:
        gl = g.detach()
        if isinstance(gl, DTensor):
            gl = gl.to_local()  # sum over this rank's shard; cross-shard handled by reductions below
        gl = gl.float()
        sq = gl.pow(2).sum() if norm_type == 2.0 else gl.abs().pow(norm_type).sum()
        if is_ep:
            ep_sq = ep_sq + sq
        else:
            non_ep_sq = non_ep_sq + sq

    if dist.is_available() and dist.is_initialized():
        if ep_group is not None and dist.get_world_size(ep_group) > 1:
            dist.all_reduce(ep_sq, op=dist.ReduceOp.SUM, group=ep_group)
        if ep_extra_group is not None and dist.get_world_size(ep_extra_group) > 1:
            dist.all_reduce(ep_sq, op=dist.ReduceOp.SUM, group=ep_extra_group)

    total_norm = (non_ep_sq + ep_sq).pow(1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g, _ in grads:
            g.mul_(clip_coef.to(g.dtype))
    return total_norm


def shard_experts_fsdp(model: torch.nn.Module, ep_fsdp_mesh) -> torch.nn.Module:
    """FSDP-shard routed-expert modules across the ``ep_fsdp`` mesh (memory sharding).

    Used when ``ep_size < world_size``: the experts are data-parallel REPLICATED
    across the ``ep_fsdp`` mesh dim, so ``fully_shard`` shards each expert param
    ``[E/ep, H, ...]`` on dim 1 (hidden) across those replicas, all-gathering during
    forward and reduce-scattering grads across ep_fsdp. No-op when ep_fsdp size <= 1
    (ep_size == world_size: experts are unique per rank, nothing to shard/replicate).
    """
    if ep_fsdp_mesh is None or ep_fsdp_mesh.size() <= 1:
        return model
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard

    from torchspec.models.draft.moe import MoEExperts

    def _shard_on_hidden(_param):
        return Shard(1)  # dim 0 is the (already EP-split) expert dim; shard hidden instead

    for module in model.modules():
        if isinstance(module, MoEExperts) and getattr(module, "ep_group", None) is not None:
            fully_shard(module, mesh=ep_fsdp_mesh, shard_placement_fn=_shard_on_hidden)
    return model


def gather_ep_full_state_dict(model: torch.nn.Module, ep_group=None) -> dict:
    """Reconstruct a FULL (un-sharded) state dict from an EP model for checkpointing.

    Routed-expert params hold only this rank's ``E/ep_size`` experts; this all-gathers
    them across ``ep_group`` and concatenates on the expert dim (dim 0) so the saved
    checkpoint matches the single-GPU / HuggingFace expert layout ``[E, ...]``. All
    other params are returned as-is (they are replicated). No-op when ep_size == 1.
    """
    sd = model.state_dict()
    if ep_group is None or not (dist.is_available() and dist.is_initialized()):
        return sd
    ep_size = dist.get_world_size(ep_group)
    if ep_size == 1:
        return sd

    ep_names = {n for n, p in model.named_parameters() if is_ep_param(p)}
    full = {}
    for name, tensor in sd.items():
        if name in ep_names:
            tensor = tensor.contiguous()
            gathered = [torch.empty_like(tensor) for _ in range(ep_size)]
            dist.all_gather(gathered, tensor, group=ep_group)
            full[name] = torch.cat(gathered, dim=0)  # concat experts along dim 0
        else:
            full[name] = tensor
    return full

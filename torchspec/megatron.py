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

"""Megatron-Core process groups and expert-parallel gradient helpers.

TorchSpec keeps its own training loop, loss and optimizer; this module only
sets up Megatron model-parallel state and handles gradients whose ownership
differs between replicated and expert-sharded parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection


def require_megatron_version(expected: str) -> None:
    from megatron.core.package_info import __version__

    if __version__ != expected:
        raise RuntimeError(f"Megatron-Core {expected} is required, found {__version__}")


def initialize_model_parallel(
    *, expert_parallel_size: int, seed: int = 1234
) -> "ProcessGroupCollection":
    """Create TP1/PP1 Megatron groups with the given EP size and seed Megatron's CUDA RNG."""

    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if dist.get_world_size() % expert_parallel_size:
        raise ValueError(
            f"world size {dist.get_world_size()} is not divisible by EP {expert_parallel_size}"
        )
    if parallel_state.model_parallel_is_initialized():
        actual = parallel_state.get_expert_model_parallel_world_size()
        if actual != expert_parallel_size:
            raise RuntimeError(f"Megatron is already initialized with EP={actual}")
    else:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=expert_parallel_size,
            order="tp-cp-ep-dp-pp",
        )
        model_parallel_cuda_manual_seed(seed)

    groups = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "cp", "ep", "dp", "expt_tp", "expt_dp", "tp_ep", "tp_cp", "tp_dp_cp"]
    )
    # synchronize_gradients relies on the dense DP group spanning EP x expert-DP.
    sizes = [dist.get_world_size(g) for g in (groups.dp, groups.ep, groups.expt_dp)]
    if sizes[0] != sizes[1] * sizes[2]:
        raise RuntimeError(
            f"unexpected Megatron groups: dp={sizes[0]}, ep={sizes[1]}, expt_dp={sizes[2]}"
        )
    return groups


def is_expert_parameter(parameter: torch.nn.Parameter) -> bool:
    return getattr(parameter, "allreduce", True) is False


def synchronize_gradients(module: torch.nn.Module, groups: "ProcessGroupCollection") -> None:
    """Average gradients across data-parallel replicas, following Megatron DDP.

    All ranks of an EP group see the same batch, so a routed expert's gradient
    already sums one copy of every token per EP rank. Replicated gradients are
    summed over dense DP (which spans EP), expert gradients over expert DP, and
    both are divided by the dense DP size.
    """

    dp_size = dist.get_world_size(groups.dp)
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        group = groups.expt_dp if is_expert_parameter(parameter) else groups.dp
        if dist.get_world_size(group) > 1:
            dist.all_reduce(parameter.grad, group=group)
        if dp_size > 1:
            parameter.grad.div_(dp_size)


def clip_grad_norm_(
    module: torch.nn.Module, max_norm: float, groups: "ProcessGroupCollection"
) -> torch.Tensor:
    """Clip by the global norm, counting each expert shard once and replicated params once."""

    grads = [p for p in module.parameters() if p.grad is not None]
    replicated_sq = torch.zeros((), device=grads[0].device)
    expert_sq = torch.zeros((), device=grads[0].device)
    for parameter in grads:
        squared = parameter.grad.detach().float().pow(2).sum()
        (expert_sq if is_expert_parameter(parameter) else replicated_sq).add_(squared)
    if dist.get_world_size(groups.ep) > 1:
        dist.all_reduce(expert_sq, group=groups.ep)
    total_norm = (replicated_sq + expert_sq).sqrt()
    scale = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for parameter in grads:
        parameter.grad.mul_(scale.to(parameter.grad.dtype))
    return total_norm

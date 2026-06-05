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

"""Expert-Parallel (EP) all-to-all token dispatch for the MoE draft block.

Single-file, DeepEP-free port of XoRL's all-to-all expert parallelism
(``xorl-internal/src/xorl/distributed/moe/{comm,alltoall,utils}.py``). Experts
are sharded across ``ep_size`` ranks (each owns ``num_experts // ep_size`` local
experts). Tokens are dispatched to the rank owning their chosen expert, computed
locally with ``torch._grouped_mm``, and combined back.

Data flow for one MoE layer:
  router topk -> per-expert counts -> all_gather counts -> local permute (group
  by destination rank) -> all_to_all -> regroup by local expert -> grouped_mm
  over local experts (weights applied after down-proj) -> all_to_all back
  (splits swapped) -> scatter-add unpermute into original [N, hidden] order.

The autograd-aware ``_AllToAll`` makes gradients flow (backward re-runs the
all-to-all with input/output split sizes swapped).
"""

import dataclasses
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Autograd-aware all-to-all primitive
# ─────────────────────────────────────────────────────────────────────────────
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(
                size=(sum(output_split_sizes), input.size(1)),
                dtype=input.dtype,
                device=input.device,
            )
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        # Re-run the all-to-all on the gradient with the split sizes swapped.
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(group, input, output_split_size=None, input_split_size=None):
    return _AllToAll.apply(group, input, output_split_size, input_split_size)


# ─────────────────────────────────────────────────────────────────────────────
# Permute / unpermute / chunk-reorder helpers
# ─────────────────────────────────────────────────────────────────────────────
def permute(tokens: torch.Tensor, routing_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder tokens into expert-major (row-major over [num_experts, num_tokens]) order.

    ``routing_map`` is ``[num_experts, num_tokens]`` (bool). Returns the permuted
    tokens and the ``sorted_indices`` (original token id per permuted row).
    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def unpermute(
    tokens: torch.Tensor, hidden_states_shape: torch.Size, permutation_mapping: torch.Tensor
) -> torch.Tensor:
    """Scatter-add permuted rows back into original token order (sums top_k, fp32 accum)."""
    hidden_dim = hidden_states_shape[-1]
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_fp32 = torch.zeros(hidden_states_shape, device=tokens.device, dtype=torch.float32)
    unpermuted_fp32.scatter_add_(0, expanded_mapping, tokens.float())
    return unpermuted_fp32.to(tokens.dtype)


def generate_weights_idx(
    routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts: int
) -> torch.Tensor:
    num_tokens, _ = routing_weights.shape
    weights_idx = torch.zeros(
        (num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def permuted_weights(
    routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Routing weights in the same expert-major order as ``permute``."""
    dense_weights = generate_weights_idx(routing_weights, selected_experts, num_experts)
    routing_map = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0).sum(dim=1)
    return dense_weights.T.contiguous().masked_select(routing_map.bool())


def sort_chunks_by_idxs(
    input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: List[int]
) -> torch.Tensor:
    """Split row-wise into chunks of ``split_sizes`` then concat in ``sorted_idxs`` order."""
    chunks = torch.split(input, split_sizes.tolist(), dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)


def _expert_chunk_permute_order(num_experts: int, ep_size: int) -> List[int]:
    """Reorder chunks from source-rank-major to local-expert-major (dispatch direction)."""
    num_local_experts = num_experts // ep_size
    return torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()


def _expert_chunk_unpermute_order(num_experts: int, ep_size: int) -> List[int]:
    """Inverse of ``_expert_chunk_permute_order`` (combine direction)."""
    num_local_experts = num_experts // ep_size
    return torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch / combine
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class AllToAllDispatchContext:
    input_splits: List[int]
    output_splits: List[int]
    num_tokens_per_expert: torch.Tensor  # [ep_size, num_local_experts] (CPU)
    routing_map: torch.Tensor  # [num_experts, num_tokens]
    perm_mapping: torch.Tensor  # permute() sorted_indices
    expert_scores: torch.Tensor  # dispatched routing weights, aligned to permuted tokens
    orig_shape: torch.Size  # [N, hidden]
    num_experts: int


def preprocess(expert_mask: torch.Tensor, num_experts: int, ep_group):
    """Compute per-rank send/recv token splits via an all_gather of per-expert counts.

    Returns (input_splits, output_splits, num_global_tokens_per_local_expert[CPU],
    sum_tokens_per_local_expert[device]).
    """
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)

    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))  # [num_experts]
    input_splits = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    )

    num_global_tokens_per_expert = torch.zeros(
        ep_size,
        num_local_tokens_per_expert.size(0),
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(
        num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group
    )

    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        :, start_idx:end_idx
    ].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    sum_tokens = num_global_tokens_per_local_expert.sum(dim=0)  # [num_local_experts]
    return input_splits, output_splits, num_global_tokens_per_local_expert.cpu(), sum_tokens


def token_pre_all2all(
    hidden_states,
    expert_mask,
    num_experts,
    input_splits,
    output_splits,
    num_global_tokens_per_local_expert,
    ep_group,
):
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)  # [num_experts, num_tokens]

    local_permuted, perm_mapping = permute(hidden_states, routing_map)
    global_permuted = all_to_all(ep_group, local_permuted, output_splits, input_splits)
    global_permuted = sort_chunks_by_idxs(
        global_permuted,
        num_global_tokens_per_local_expert.ravel(),
        _expert_chunk_permute_order(num_experts, ep_group.size()),
    )
    return global_permuted, routing_map, perm_mapping, org_hidden_states_shape


def score_pre_all2all(
    routing_weights,
    selected_experts,
    num_experts,
    input_splits,
    output_splits,
    num_global_tokens_per_local_expert,
    ep_group,
):
    local_scores = permuted_weights(routing_weights, selected_experts, num_experts).unsqueeze(-1)
    global_scores = all_to_all(ep_group, local_scores, output_splits, input_splits)
    global_scores = sort_chunks_by_idxs(
        global_scores,
        num_global_tokens_per_local_expert.ravel(),
        _expert_chunk_permute_order(num_experts, ep_group.size()),
    )
    return global_scores.squeeze(-1)


def tokens_post_all2all(
    expert_outputs,
    num_experts,
    input_splits,
    output_splits,
    num_global_tokens_per_local_expert,
    perm_mapping,
    org_shape,
    ep_group,
):
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        _expert_chunk_unpermute_order(num_experts, ep_group.size()),
    )
    # Reverse all-to-all: input/output splits swapped vs dispatch.
    unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)
    return unpermute(unpermute_outputs, org_shape, perm_mapping)


def alltoall_pre_dispatch(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    ep_group,
) -> Tuple[torch.Tensor, torch.Tensor, AllToAllDispatchContext]:
    """Dispatch tokens + routing weights to expert-owning ranks.

    Returns (permuted_tokens[M, hidden] grouped by local expert, cumsum[num_local]
    per-local-expert cumulative token boundary, ctx).
    """
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(
        2, 1, 0
    )  # [E, top_k, N]

    input_splits, output_splits, num_tokens_per_expert, sum_tokens = preprocess(
        expert_mask, num_experts, ep_group
    )

    permuted_tokens, routing_map, perm_mapping, orig_shape = token_pre_all2all(
        hidden_states,
        expert_mask,
        num_experts,
        input_splits,
        output_splits,
        num_tokens_per_expert,
        ep_group,
    )
    cumsum = torch.cumsum(sum_tokens, dim=0).to(permuted_tokens.device)
    expert_scores = score_pre_all2all(
        routing_weights,
        selected_experts,
        num_experts,
        input_splits,
        output_splits,
        num_tokens_per_expert,
        ep_group,
    ).to(permuted_tokens.dtype)

    ctx = AllToAllDispatchContext(
        input_splits=input_splits,
        output_splits=output_splits,
        num_tokens_per_expert=num_tokens_per_expert,
        routing_map=routing_map,
        perm_mapping=perm_mapping,
        expert_scores=expert_scores,
        orig_shape=orig_shape,
        num_experts=num_experts,
    )
    return permuted_tokens, cumsum, ctx


def alltoall_post_combine(
    expert_output: torch.Tensor, ctx: AllToAllDispatchContext, ep_group
) -> torch.Tensor:
    return tokens_post_all2all(
        expert_output,
        ctx.num_experts,
        ctx.input_splits,
        ctx.output_splits,
        ctx.num_tokens_per_expert,
        ctx.perm_mapping,
        ctx.orig_shape,
        ep_group,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EP expert compute (dispatch -> local grouped_mm -> combine)
# ─────────────────────────────────────────────────────────────────────────────
def _cumsum_to_counts(cumsum: torch.Tensor, num_local: int) -> torch.Tensor:
    return torch.diff(cumsum, prepend=cumsum.new_zeros(1))


def _grouped_ffn_by_counts(
    tokens: torch.Tensor,
    counts: torch.Tensor,
    gate_up_local: torch.Tensor,
    down_local: torch.Tensor,
) -> torch.Tensor:
    """Pad local-expert token blocks to 8-alignment, run grouped_mm, unpad."""
    # Lazy import to avoid an import cycle (moe.py does not import moe_ep at module load).
    from torchspec.models.draft.moe import (
        _TOKEN_GROUP_ALIGN,
        _compute_pad_indices,
        _run_experts_grouped_mm,
    )

    total = tokens.shape[0]
    hidden_dim = tokens.shape[-1]
    padded_counts = torch.clamp_min(counts, _TOKEN_GROUP_ALIGN)
    padded_counts = (
        (padded_counts + _TOKEN_GROUP_ALIGN - 1) // _TOKEN_GROUP_ALIGN * _TOKEN_GROUP_ALIGN
    ).to(torch.int32)
    pad_dst = _compute_pad_indices(counts.to(torch.int32), padded_counts, total, tokens.device)
    total_padded = int(padded_counts.sum().item())
    padded = tokens.new_zeros(total_padded, hidden_dim)
    padded[pad_dst] = tokens
    out_padded = _run_experts_grouped_mm(padded, gate_up_local, down_local, padded_counts)
    return out_padded[pad_dst]


def ep_moe_experts_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_up_local: torch.Tensor,
    down_local: torch.Tensor,
    num_experts: int,
    ep_group,
) -> torch.Tensor:
    """Expert-parallel routed-expert compute. ``gate_up_local``/``down_local`` hold this
    rank's ``num_experts // ep_size`` experts. Returns ``[*, hidden]`` like the input."""
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    flat = hidden_states.reshape(-1, hidden_dim)

    permuted_tokens, cumsum, ctx = alltoall_pre_dispatch(
        flat, routing_weights, selected_experts, num_experts, ep_group
    )
    num_local = gate_up_local.shape[0]
    counts = _cumsum_to_counts(cumsum, num_local)

    expert_out = _grouped_ffn_by_counts(permuted_tokens, counts, gate_up_local, down_local)
    # Apply routing weights after the down-projection (combine is a pure scatter-add).
    expert_out = expert_out * ctx.expert_scores.to(expert_out.dtype).unsqueeze(-1)

    combined = alltoall_post_combine(expert_out, ctx, ep_group)
    return combined.view(orig_shape)

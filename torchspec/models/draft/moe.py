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

"""DeepSeek-V3 / Kimi-style Mixture-of-Experts block for the Eagle3 draft model.

This is the single-GPU (no Expert-Parallel) MoE implementation. The routed
experts are computed with PyTorch's native grouped matmul (``torch._grouped_mm``),
which performs E independent GEMMs over contiguous, expert-grouped token blocks.

Ported from the XoRL training framework
(``xorl-internal/src/xorl/models/transformers/deepseek_v3`` and
``.../models/layers/moe/backend/native.py``). EP / FSDP-on-experts is layered
on top of this in a later phase; this module is intentionally self-contained.

Key facts about the expert weight layout (``(G, K, N)`` = ``(experts, in, out)``):
  - ``experts.gate_up_proj``: ``[E, hidden, 2 * moe_intermediate]`` (gate and up FUSED)
  - ``experts.down_proj``:    ``[E, moe_intermediate, hidden]``
This is exactly what ``torch._grouped_mm(x[T, K], w[E, K, N], offs=...)`` expects.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

# torch._grouped_mm requires each group's row count to be a multiple of this
# (a bf16/fp16 alignment constraint); we pad per-expert token counts up to it.
_TOKEN_GROUP_ALIGN = 8


# ─────────────────────────────────────────────────────────────────────────────
# Native single-GPU grouped-matmul expert compute
# ─────────────────────────────────────────────────────────────────────────────
def _compute_pad_indices(
    counts: torch.Tensor,
    padded_counts: torch.Tensor,
    total_sorted: int,
    device: torch.device,
) -> torch.Tensor:
    """Map each expert-sorted token to its row in the padded (8-aligned) layout.

    ``counts``/``padded_counts`` are per-expert real / 8-aligned token counts
    ``[E]``. Returns ``pad_dst[total_sorted]`` such that
    ``padded[pad_dst] = sorted_tokens`` scatters real tokens into the padded
    buffer (leaving the pad rows as zeros), and the same index gathers them back.
    """
    counts64 = counts.to(torch.int64)
    padded64 = padded_counts.to(torch.int64)
    sorted_starts = torch.cumsum(counts64, 0) - counts64  # [E] start of expert e in sorted array
    padded_starts = torch.cumsum(padded64, 0) - padded64  # [E] start of expert e in padded array
    boundaries = torch.cumsum(counts64, 0)  # [E] exclusive-end of expert e in sorted array

    pos = torch.arange(total_sorted, device=device, dtype=torch.int64)
    expert_of = torch.searchsorted(
        boundaries, pos, right=True
    )  # which expert each sorted row belongs to
    within = pos - sorted_starts[expert_of]  # offset within that expert's block
    return padded_starts[expert_of] + within


def _run_experts_grouped_mm(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    padded_counts: torch.Tensor,
) -> torch.Tensor:
    """Fused-gate_up SwiGLU expert FFN via two ``torch._grouped_mm`` calls.

    ``x`` is the padded, expert-grouped token buffer ``[total_padded, hidden]``.
    ``offs`` is the cumulative (exclusive-end) per-expert boundary, int32.
    """
    offsets = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
    compute_dtype = torch.bfloat16

    # gate_up_proj: [E, hidden, 2*I] -> grouped GEMM -> [total_padded, 2*I]
    gate_up = torch._grouped_mm(x.to(compute_dtype), gate_up_proj.to(compute_dtype), offs=offsets)
    intermediate_size = gate_up.shape[-1] // 2
    gate = gate_up[..., :intermediate_size]
    up = gate_up[..., intermediate_size:]
    h = F.silu(gate) * up

    # down_proj: [E, I, hidden] -> [total_padded, hidden]
    out = torch._grouped_mm(h, down_proj.to(compute_dtype), offs=offsets)
    return out.to(x.dtype)


def native_moe_experts_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute routed-expert output for flattened tokens, single GPU.

    Args:
        hidden_states:   ``[N, hidden]`` flattened tokens.
        routing_weights: ``[N, top_k]`` per-(token, slot) gating weight.
        selected_experts:``[N, top_k]`` chosen expert id per (token, slot).
    Returns:
        ``[N, hidden]`` combined expert output.
    """
    num_tokens, top_k = selected_experts.shape
    hidden_dim = hidden_states.shape[-1]
    device = hidden_states.device

    # 1. Flatten top-k assignments and sort tokens by expert (stable keeps order).
    flat_experts = selected_experts.reshape(-1)  # [N*top_k]
    flat_weights = routing_weights.reshape(-1)  # [N*top_k]
    sorted_order = torch.argsort(flat_experts, stable=True)
    token_ids = sorted_order // top_k  # original token index per sorted slot
    sorted_hidden = hidden_states[token_ids]  # [N*top_k, hidden]
    sorted_weights = flat_weights[sorted_order]

    # 2. Per-expert token counts (exact integer counts).
    counts = torch.bincount(flat_experts, minlength=num_experts).to(torch.int32)

    # 3. Pad counts up to a multiple of _TOKEN_GROUP_ALIGN (min one group each).
    padded_counts = torch.clamp_min(counts, _TOKEN_GROUP_ALIGN)
    padded_counts = (
        (padded_counts + _TOKEN_GROUP_ALIGN - 1) // _TOKEN_GROUP_ALIGN * _TOKEN_GROUP_ALIGN
    ).to(torch.int32)

    # 4. Scatter sorted tokens into the padded, expert-contiguous buffer.
    total_sorted = num_tokens * top_k
    pad_dst = _compute_pad_indices(counts, padded_counts, total_sorted, device)
    total_padded = int(padded_counts.sum().item())
    sorted_hidden_padded = sorted_hidden.new_zeros(total_padded, hidden_dim)
    sorted_hidden_padded[pad_dst] = sorted_hidden

    # 5. Grouped expert FFN.
    expert_out_padded = _run_experts_grouped_mm(
        sorted_hidden_padded, gate_up_proj, down_proj, padded_counts
    )

    # 6. Gather back to sorted order, apply gating weights.
    expert_out = expert_out_padded[pad_dst]
    expert_out = expert_out * sorted_weights.to(expert_out.dtype).unsqueeze(-1)

    # 7. Un-sort to (token, slot) order and sum the top_k contributions per token.
    inv_sorted = torch.empty_like(sorted_order)
    inv_sorted[sorted_order] = torch.arange(sorted_order.size(0), device=device)
    expert_out = expert_out[inv_sorted]
    return expert_out.view(num_tokens, top_k, hidden_dim).sum(dim=1)


def global_load_balancing_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    dp_group=None,
) -> torch.Tensor:
    """Switch/DeepSeek-style MoE load-balancing auxiliary loss.

    ``router_logits``: ``[num_tokens, num_experts]``. The per-expert token
    fraction is (optionally) averaged across ``dp_group`` so the balance term
    reflects GLOBAL routing, while the per-expert router probability is kept
    local so gradients still flow to this rank's gate.
    """
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    _, selected = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected, num_experts).float()  # [N, top_k, E]
    tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # [E] fraction of tokens per expert
    if dp_group is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.AVG, group=dp_group)
    router_prob_per_expert = routing_weights.mean(dim=0)  # [E] (local: keeps gate grads)
    return (tokens_per_expert * router_prob_per_expert).sum() * num_experts


# ─────────────────────────────────────────────────────────────────────────────
# Modules
# ─────────────────────────────────────────────────────────────────────────────
class MoEExperts(nn.Module):
    """Container for the stacked routed-expert weights ``(E, in, out)``.

    When ``ep_group`` is given (Expert Parallel), the weights hold only this
    rank's ``num_experts // ep_size`` local experts and ``forward`` dispatches
    tokens across ranks via all-to-all. With ``ep_group=None`` (or size 1) the
    full expert set lives on one device and a single-GPU grouped_mm is used.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_size: int,
        init_std: float = 0.02,
        ep_group=None,
    ):
        super().__init__()
        self.num_global_experts = num_experts
        self.ep_group = ep_group
        ep_size = ep_group.size() if (ep_group is not None and ep_group.size() > 1) else 1
        assert num_experts % ep_size == 0, (
            f"num_experts {num_experts} not divisible by ep_size {ep_size}"
        )
        self.num_local_experts = num_experts // ep_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_dim, 2 * intermediate_size)
        )
        self.gate_up_proj._fused_gate_up = True  # marker for later EP/checkpoint handling
        self.down_proj = nn.Parameter(
            torch.empty(self.num_local_experts, intermediate_size, hidden_dim)
        )
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=init_std)
        nn.init.normal_(self.down_proj, mean=0.0, std=init_std)

        if ep_size > 1:
            # Experts are unique per EP rank: tag so the trainer's gradient sync /
            # grad-norm clip treats them differently from replicated (DP) params.
            self.gate_up_proj._is_ep = True
            self.down_proj._is_ep = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_group is not None and self.ep_group.size() > 1:
            from torchspec.models.draft.moe_ep import ep_moe_experts_forward

            return ep_moe_experts_forward(
                hidden_states,
                routing_weights,
                selected_experts,
                self.gate_up_proj,
                self.down_proj,
                self.num_global_experts,
                self.ep_group,
            )
        return native_moe_experts_forward(
            hidden_states,
            routing_weights,
            selected_experts,
            self.gate_up_proj,
            self.down_proj,
            self.num_local_experts,
        )


class DeepseekV3TopkRouter(nn.Module):
    """DeepSeek-V3 routing gate: a bare weight + an aux-loss-free correction bias.

    NOTE: this differs from a plain softmax-topk gate. Scoring is ``sigmoid``; an
    ``e_score_correction_bias`` shifts WHICH experts are picked (selection) while
    the routing WEIGHTS use the un-biased sigmoid scores. Selection is also
    group-limited (``n_group`` / ``topk_group``). See ``DeepseekV3MoEBlock._route``.
    """

    def __init__(self, config, init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # fp32 gate matmul for routing stability.
        return F.linear(hidden_states.float(), self.weight.float())


class SwiGLUMLP(nn.Module):
    """Plain SwiGLU MLP used for the shared expert (explicit intermediate size)."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV3MoEBlock(nn.Module):
    """Drop-in replacement for the draft layer's dense MLP: routed experts + shared expert.

    forward(hidden_states[B, S, H]) -> output[B, S, H], so it slots directly into
    ``DeepSeekDecoderLayer`` where ``self.mlp(hidden_states)`` is used.
    The last router logits are cached on ``self.router_logits`` for the MoE
    load-balancing auxiliary loss (wired in a later phase).
    """

    def __init__(self, config, ep_group=None):
        super().__init__()
        self.config = config
        # ep_group may be passed explicitly or attached to the config by the trainer.
        ep_group = ep_group if ep_group is not None else getattr(config, "ep_group", None)
        self.ep_group = ep_group
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        # Draft is trained from scratch, so by default let the gate learn through
        # the expert path (train_router=True). DeepSeek-V3's own default is False
        # (gate trained only via aux loss + bias); make it configurable.
        self.train_router = getattr(config, "train_router", True)

        init_std = getattr(config, "initializer_range", 0.02)
        self.gate = DeepseekV3TopkRouter(config, init_std=init_std)
        self.experts = MoEExperts(
            self.num_experts,
            self.hidden_size,
            config.moe_intermediate_size,
            init_std=init_std,
            ep_group=ep_group,
        )

        n_shared = getattr(config, "n_shared_experts", 0) or 0
        if n_shared > 0:
            self.shared_experts = SwiGLUMLP(
                self.hidden_size,
                config.moe_intermediate_size * n_shared,
                getattr(config, "hidden_act", "silu"),
            )
        else:
            self.shared_experts = None

        self.router_logits = None  # cached per-forward for aux loss (later phase)

    def _route(self, router_logits: torch.Tensor, input_dtype: torch.dtype):
        """Sigmoid scoring + group-limited selection + aux-loss-free bias.

        Returns (routing_weights[N, top_k], selected_experts[N, top_k]).
        """
        router_scores = router_logits.sigmoid()  # un-biased scores -> used for WEIGHTS
        choice_scores = (
            router_scores + self.gate.e_score_correction_bias.float()
        )  # biased -> used for SELECTION

        experts_per_group = self.num_experts // self.n_group
        group_topk = min(2, experts_per_group)
        group_scores = (
            choice_scores.view(-1, self.n_group, experts_per_group)
            .topk(group_topk, dim=-1)[0]
            .sum(dim=-1)
        )  # [N, n_group]
        group_idx = torch.topk(
            group_scores, k=min(self.topk_group, self.n_group), dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, experts_per_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = choice_scores.masked_fill(~score_mask.bool(), 0.0)
        selected_experts = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

        routing_weights = router_scores.gather(1, selected_experts)  # gather from UN-biased scores
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights.to(input_dtype), selected_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(flat_hidden)
        self.router_logits = router_logits
        routing_weights, selected_experts = self._route(router_logits, flat_hidden.dtype)
        if not self.train_router:
            routing_weights = routing_weights.detach()

        expert_output = self.experts(flat_hidden, routing_weights, selected_experts)
        expert_output = expert_output.view(bsz, seq_len, hidden_dim)

        if self.shared_experts is not None:
            expert_output = expert_output + self.shared_experts(hidden_states)
        return expert_output

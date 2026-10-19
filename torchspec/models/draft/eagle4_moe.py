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

"""EAGLE-4 MoE draft MLPs, in the weight layout SGLang's EAGLE-4 loader reads."""

import re
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

LATENT_MOE = "nemotron_latent_relu2"
_PACKED_LATENT = re.compile(r"(.*\.experts)\.(up_proj|down_proj)")
_PER_EXPERT_LATENT = re.compile(r"(.*\.experts)\.(\d+)\.(up_proj|down_proj)\.weight")


class Eagle4Router(nn.Module):
    """Sigmoid router whose expert-selection bias balances load without a loss."""

    def __init__(self, config):
        super().__init__()
        n = config.n_routed_experts
        self.weight = nn.Parameter(
            nn.init.normal_(torch.empty(n, config.hidden_size), std=config.initializer_range)
        )
        self.register_buffer("e_score_correction_bias", torch.zeros(n))
        self.register_buffer("load", torch.zeros(n, dtype=torch.long), persistent=False)
        self.update_rate = getattr(
            config,
            "expert_bias_update_rate",
            getattr(config, "moe_router_bias_update_rate", 1e-3),
        )

    def _apply(self, fn, recurse=True):
        # The bias stays fp32 through model.to(bf16): bf16 cannot take 1e-3 steps past |0.25|.
        super()._apply(fn, recurse)
        self.e_score_correction_bias = self.e_score_correction_bias.float()
        return self

    @torch.no_grad()
    def update_bias(self) -> float:
        """Called once per optimizer step; returns max / mean expert load."""
        load = self.load.float()
        if dist.is_initialized():
            dist.all_reduce(load)
        self.load.zero_()

        # Under-loaded experts step up, over-loaded ones down.
        self.e_score_correction_bias += self.update_rate * (load.mean() - load).sign()
        return (load.max() / load.mean()).item()


class SharedExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn, gated):
        super().__init__()
        if gated:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x):
        h = self.up_proj(x)
        gated = hasattr(self, "gate_proj")
        return self.down_proj(self.act_fn(self.gate_proj(x)) * h if gated else self.act_fn(h))


class Eagle4MoE(nn.Module):
    """``sparse_swiglu`` (default) or ``nemotron_latent_relu2`` MoE replacing the dense MLP."""

    def __init__(self, config):
        super().__init__()
        self.latent = getattr(config, "moe_architecture", "sparse_swiglu") == LATENT_MOE
        self.top_k = config.num_experts_per_tok
        self.n_group = getattr(config, "n_group", None) or 1
        self.topk_group = getattr(config, "topk_group", None) or self.n_group
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.act_fn = ACT2FN[config.hidden_act]
        n, hidden, inter = config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
        width = config.moe_latent_size if self.latent else hidden

        self.gate = Eagle4Router(config)

        # Packed experts: SwiGLU as w13 (gate | up) and w2, latent ReLU^2 as up and down.
        # Zero down projections make a new routed bank a no-op, so an upcycled draft starts
        # from its dense output.
        up = torch.empty(n, inter if self.latent else 2 * inter, width)
        up = nn.Parameter(nn.init.normal_(up, std=config.initializer_range))
        down = nn.Parameter(torch.zeros(n, width, inter))
        self.expert_names = ("up_proj", "down_proj") if self.latent else ("w13_weight", "w2_weight")
        self.experts = nn.ParameterDict(dict(zip(self.expert_names, (up, down))))

        # Latent experts run between these projections.
        if self.latent:
            self.fc1_latent_proj = nn.Linear(hidden, width, bias=False)
            self.fc2_latent_proj = nn.Linear(width, hidden, bias=False)

        # n_shared_experts are folded into one full-width expert.
        n_shared = getattr(config, "n_shared_experts", None) or 0
        shared_inter = config.moe_shared_expert_intermediate_size if self.latent else inter
        self.shared_experts = (
            SharedExpert(hidden, shared_inter * n_shared, self.act_fn, gated=not self.latent)
            if n_shared
            else None
        )

    def route(self, x):
        """DeepSeek-V3 routing: top-k experts per token and their combine weights."""
        scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        biased = scores + self.gate.e_score_correction_bias

        # Group-limited routing keeps the topk_group groups with the best top-2 biased scores.
        if self.n_group > 1:
            groups = biased.unflatten(-1, (self.n_group, -1))
            best = groups.topk(2, dim=-1).values.sum(-1).topk(self.topk_group, dim=-1).indices
            dropped = torch.ones_like(groups[..., 0], dtype=torch.bool).scatter_(1, best, False)
            biased = groups.masked_fill(dropped[..., None], float("-inf")).flatten(1)

        # The bias only selects experts; the unbiased scores weight them.
        idx = biased.topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, idx)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        return idx, weights * self.routed_scaling_factor

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        idx, weights = self.route(x)
        counts = torch.bincount(idx.flatten(), minlength=self.gate.weight.shape[0])
        if self.training:
            self.gate.load += counts

        # (token, slot) pairs sorted by expert feed one grouped GEMM per projection.
        order = idx.flatten().argsort()
        tokens = order // self.top_k
        offs = counts.cumsum(0).int()
        h = self.fc1_latent_proj(x) if self.latent else x
        up, down = (self.experts[name] for name in self.expert_names)
        inner = F.grouped_mm(h[tokens], up.transpose(1, 2), offs=offs)
        if self.latent:
            inner = self.act_fn(inner)
        else:
            gate, value = inner.chunk(2, dim=-1)
            inner = self.act_fn(gate) * value
        out = F.grouped_mm(inner, down.transpose(1, 2), offs=offs)

        # Weight each expert output and sum it back onto its token.
        out = out * weights.flatten()[order, None].to(out.dtype)
        out = h.new_zeros(h.shape).index_add_(0, tokens, out)
        if self.latent:
            out = self.fc2_latent_proj(out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out.view_as(hidden_states)


def split_latent_experts(tensors: dict) -> dict:
    """Packed latent experts to the one-key-per-expert layout SGLang serves."""
    out = {}
    for key, value in tensors.items():
        m = _PACKED_LATENT.fullmatch(key)
        out.update(
            {f"{m[1]}.{i}.{m[2]}.weight": w.clone() for i, w in enumerate(value)}
            if m
            else {key: value}
        )
    return out


def stack_latent_experts(tensors: dict) -> dict:
    """Inverse of ``split_latent_experts``."""
    out, groups = {}, defaultdict(dict)
    for key, value in tensors.items():
        m = _PER_EXPERT_LATENT.fullmatch(key)
        if m:
            groups[f"{m[1]}.{m[3]}"][int(m[2])] = value
        else:
            out[key] = value
    out.update({k: torch.stack([g[i] for i in sorted(g)]) for k, g in groups.items()})
    return out

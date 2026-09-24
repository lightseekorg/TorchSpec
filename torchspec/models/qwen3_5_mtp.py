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

"""Qwen3.5 MoE MTP on Megatron-Core: construction and Hugging Face checkpoint import."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from torchspec.megatron import require_megatron_version
from torchspec.models.mtp import MegatronMTPModel

MCORE_VERSION = "0.18.0"
_EXPERT_PREFIX = "mtp.layers.0.mlp.experts."


@dataclass(frozen=True)
class Qwen35MTPConfig:
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    attention_output_gate: bool

    @classmethod
    def from_dict(cls, raw_config: Mapping[str, Any]) -> "Qwen35MTPConfig":
        config = raw_config.get("text_config", raw_config)
        rope = config.get("rope_parameters", {})
        result = cls(
            hidden_size=int(config["hidden_size"]),
            vocab_size=int(config["vocab_size"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            num_experts=int(config["num_experts"]),
            num_experts_per_tok=int(config["num_experts_per_tok"]),
            moe_intermediate_size=int(config["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(config["shared_expert_intermediate_size"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(rope.get("rope_theta", 10_000.0)),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
            attention_output_gate=bool(config.get("attn_output_gate", False)),
        )
        if result.num_attention_heads % result.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        return result

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "Qwen35MTPConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def build_qwen35_mtp_model(
    config: Qwen35MTPConfig, groups, *, dtype: torch.dtype = torch.float32
) -> MegatronMTPModel:
    """Build the Qwen3.5 MTP layer from native Megatron modules (local spec)."""

    require_megatron_version(MCORE_VERSION)
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.multi_token_prediction import get_mtp_layer_spec
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    ep_size = dist.get_world_size(groups.ep)
    if config.num_experts % ep_size:
        raise ValueError(f"num_experts={config.num_experts} is not divisible by EP={ep_size}")
    core_config = TransformerConfig(
        num_layers=1,
        mtp_num_layers=1,
        mtp_use_repeated_layer=True,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
        kv_channels=config.head_dim,
        ffn_hidden_size=config.moe_intermediate_size,
        moe_ffn_hidden_size=config.moe_intermediate_size,
        num_moe_experts=config.num_experts,
        moe_router_topk=config.num_experts_per_tok,
        # Softmax over the top-k logits equals Qwen's softmax -> top-k -> renormalize.
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_dtype="fp32",
        # Train on hard-label CE only.
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_shared_expert_intermediate_size=config.shared_expert_intermediate_size,
        moe_shared_expert_gate=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        normalization="RMSNorm",
        layernorm_epsilon=config.rms_norm_eps,
        qk_layernorm=True,
        attention_output_gate=config.attention_output_gate,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=ep_size,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        use_cpu_initialization=not torch.cuda.is_available(),
    )
    layer_spec = get_gpt_layer_local_spec(
        num_experts=config.num_experts, qk_layernorm=True, normalization="RMSNorm"
    )
    mtp_layer = build_module(
        get_mtp_layer_spec(layer_spec, use_transformer_engine=False),
        config=core_config,
        layer_number=1,
        pg_collection=groups,
    )
    # Standalone construction does not propagate the layer number to the router.
    mtp_layer.mtp_model_layer.mlp.set_layer_number(1)

    model = MegatronMTPModel(
        mtp_layer=mtp_layer,
        embedding=torch.nn.Embedding(config.vocab_size, config.hidden_size),
        lm_head=torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        rotary_embedding=RotaryEmbedding(
            kv_channels=config.head_dim,
            rotary_percent=config.partial_rotary_factor,
            rotary_interleaved=False,
            rotary_base=config.rope_theta,
            use_cpu_initialization=not torch.cuda.is_available(),
            cp_group=groups.cp,
        ),
    )
    if torch.cuda.is_available():
        model = model.cuda()
    return model.to(dtype=dtype)


def pack_qwen35_qkv(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reorder Qwen's per-head ``[query, gate]`` rows plus K/V into Megatron's
    per-KV-group ``[queries, gates, key, value]`` layout."""

    hidden_size = q_proj.shape[1]
    if q_proj.shape[0] != 2 * num_attention_heads * head_dim:
        raise ValueError(f"unexpected q_proj shape {tuple(q_proj.shape)}")
    if k_proj.shape[0] != num_key_value_heads * head_dim or k_proj.shape != v_proj.shape:
        raise ValueError(f"unexpected k/v_proj shapes {tuple(k_proj.shape)}, {tuple(v_proj.shape)}")
    q_and_gate = q_proj.view(num_attention_heads, 2, head_dim, hidden_size)
    key = k_proj.view(num_key_value_heads, head_dim, hidden_size)
    value = v_proj.view(num_key_value_heads, head_dim, hidden_size)
    per_group = num_attention_heads // num_key_value_heads
    blocks = []
    for group in range(num_key_value_heads):
        heads = slice(group * per_group, (group + 1) * per_group)
        blocks += [
            q_and_gate[heads, 0],
            q_and_gate[heads, 1],
            key[group : group + 1],
            value[group : group + 1],
        ]
    return torch.cat(blocks).reshape(-1, hidden_size)


@torch.no_grad()
def load_qwen35_mtp_weights(
    model: MegatronMTPModel, config: Qwen35MTPConfig, checkpoint_dir: Union[str, Path]
) -> dict[str, Any]:
    """Load the frozen embedding/head and this rank's MTP shard from an HF checkpoint.

    Parameters are NaN-filled first, so anything left unloaded is caught.
    Returns a per-tensor mapping report and raises if a parameter is not loaded
    or a checkpoint MTP key is left unconsumed (other than experts owned by
    another EP rank).
    """

    root = Path(checkpoint_dir)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, Any] = {}

    def read(key: str) -> torch.Tensor:
        filename = weight_map[key]
        if filename not in handles:
            handles[filename] = safe_open(root / filename, framework="pt", device="cpu")
        return handles[filename].get_tensor(key)

    layer = model.mtp_layer
    transformer = layer.mtp_model_layer
    attention = transformer.self_attention
    moe = transformer.mlp
    names = {id(p): name for name, p in model.named_parameters()}
    for parameter in model.parameters():
        parameter.fill_(float("nan"))

    rank = dist.get_rank()
    ep_rank = dist.get_rank(moe.ep_group)
    entries: list[dict[str, Any]] = []

    def load(parameter, keys, transform="identity"):
        tensors = [read(key) for key in keys]
        if transform == "identity":
            value = tensors[0]
        elif transform == "rmsnorm_one_plus_delta":  # Qwen RMSNorm stores gamma - 1
            value = tensors[0].float() + 1.0
        elif transform == "concat_gate_up":
            value = torch.cat(tensors)
        else:
            value = pack_qwen35_qkv(
                *tensors,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
            )
        if parameter.shape != value.shape:
            raise ValueError(f"{keys}: shape {tuple(value.shape)} != {tuple(parameter.shape)}")
        parameter.copy_(value)
        expected = value.to(device=parameter.device, dtype=parameter.dtype)
        entries.append(
            {
                "source": keys,
                "destination": names[id(parameter)],
                "source_shape": [list(t.shape) for t in tensors],
                "source_dtype": [str(t.dtype) for t in tensors],
                "destination_shape": list(parameter.shape),
                "destination_dtype": str(parameter.dtype),
                "transform": transform,
                "owning_rank": rank,
                "ep_rank": ep_rank,
                "expert_parallel": getattr(parameter, "allreduce", True) is False,
                "max_abs_diff_vs_transformed_source": float((parameter - expected).abs().max()),
                "result": "loaded",
            }
        )

    norm = "rmsnorm_one_plus_delta"
    attn = "mtp.layers.0.self_attn."
    load(model.embedding.embedding.weight, ["model.language_model.embed_tokens.weight"])
    load(model.lm_head.weight, ["lm_head.weight"])
    load(layer.enorm.weight, ["mtp.pre_fc_norm_embedding.weight"], norm)
    load(layer.hnorm.weight, ["mtp.pre_fc_norm_hidden.weight"], norm)
    load(layer.eh_proj.weight, ["mtp.fc.weight"])
    load(transformer.input_layernorm.weight, ["mtp.layers.0.input_layernorm.weight"], norm)
    load(attention.q_layernorm.weight, [attn + "q_norm.weight"], norm)
    load(attention.k_layernorm.weight, [attn + "k_norm.weight"], norm)
    load(attention.linear_qkv.weight, [attn + f"{p}_proj.weight" for p in "qkv"], "grouped_qkv")
    load(attention.linear_proj.weight, [attn + "o_proj.weight"])
    load(
        transformer.pre_mlp_layernorm.weight, ["mtp.layers.0.post_attention_layernorm.weight"], norm
    )
    load(moe.router.weight, ["mtp.layers.0.mlp.gate.weight"])
    for local_index, expert_index in enumerate(moe.local_expert_indices):
        expert = moe.experts.local_experts[local_index]
        prefix = f"{_EXPERT_PREFIX}{expert_index}."
        load(
            expert.linear_fc1.weight,
            [prefix + "gate_proj.weight", prefix + "up_proj.weight"],
            "concat_gate_up",
        )
        load(expert.linear_fc2.weight, [prefix + "down_proj.weight"])
    shared = moe.shared_experts
    prefix = "mtp.layers.0.mlp.shared_expert."
    load(
        shared.linear_fc1.weight,
        [prefix + "gate_proj.weight", prefix + "up_proj.weight"],
        "concat_gate_up",
    )
    load(shared.linear_fc2.weight, [prefix + "down_proj.weight"])
    load(shared.gate_weight, ["mtp.layers.0.mlp.shared_expert_gate.weight"])
    load(layer.final_layernorm.weight, ["mtp.norm.weight"], norm)

    local_experts = {int(i) for i in moe.local_expert_indices}
    consumed = {key for entry in entries for key in entry["source"]}
    required = {k for k in weight_map if k.startswith("mtp.")}
    required |= {"model.language_model.embed_tokens.weight", "lm_head.weight"}
    other_rank, unexpected = [], []
    for key in sorted(required - consumed):
        is_expert = key.startswith(_EXPERT_PREFIX)
        if is_expert and int(key[len(_EXPERT_PREFIX) :].split(".")[0]) not in local_experts:
            other_rank.append(key)
        else:
            unexpected.append(key)
    missing = sorted(set(names.values()) - {entry["destination"] for entry in entries})
    nonfinite = sorted(name for name, p in model.named_parameters() if not p.isfinite().all())
    if unexpected or missing or nonfinite:
        raise RuntimeError(
            f"MTP import failed: unexpected={unexpected}, missing={missing}, nonfinite={nonfinite}"
        )
    return {
        "checkpoint_root": str(root),
        "owning_rank": rank,
        "ep_rank": ep_rank,
        "local_expert_indices": sorted(local_experts),
        "entries": entries,
        "consumed_keys": sorted(consumed),
        "keys_owned_by_other_ep_ranks": other_rank,
        "unexpected_unconsumed_keys": unexpected,
        "missing_destinations": missing,
    }

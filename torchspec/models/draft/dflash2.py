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

"""DFlash2 draft model adapted from z-lab/dflash at revision 07ebd93.

Source: https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDecoderLayer,
    DFlashDraftModel,
    build_target_layer_ids,
)

_SERVING_KEY_REMAP = (
    ("context_proj.", "fc."),
    ("context_norm.", "hidden_norm."),
    ("final_norm.", "norm."),
)

_NESTED_CONFIG_FIELDS = (
    "attention_mode",
    "block_size",
    "conv_group_size",
    "conv_kernel_size",
    "final_logit_softcapping",
    "input_embedding_scale",
    "mask_token_id",
    "output_multiplier",
    "selector_rank",
    "selector_top_k",
    "target_layer_ids",
)


def dflash2_config_for_serving(raw_config: dict, *, export_for_vllm: bool = False) -> dict:
    config = json.loads(json.dumps(raw_config))
    dflash_config = dict(config.get("dflash_config") or {})
    for key in _NESTED_CONFIG_FIELDS:
        if key in config and key not in dflash_config:
            dflash_config[key] = config[key]
        config.pop(key, None)
    config["dflash_config"] = dflash_config
    target_num_hidden_layers = config.pop("target_num_hidden_layers", None)
    if target_num_hidden_layers is not None:
        config["num_target_layers"] = target_num_hidden_layers
    if not export_for_vllm and float(dflash_config.get("input_embedding_scale", 1.0)) != 1.0:
        raise ValueError("SGLang DFlash2 export requires input_embedding_scale=1.0")
    config.pop("target_hidden_size", None)
    config["model_type"] = "qwen3"
    return config


def dflash2_state_dict_for_serving(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key == "embed_tokens.weight":
            continue
        new_key = key
        for old_prefix, new_prefix in _SERVING_KEY_REMAP:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                break
        remapped[new_key] = value
    return remapped


class DFlash2Config(DFlashConfig):
    model_type = "qwen3_dflash2"

    def __init__(
        self,
        block_size: int = 16,
        conv_kernel_size: int = 2,
        conv_group_size: int = 16,
        selector_rank: int = 256,
        selector_top_k: int = 16,
        attention_mode: str = "gqa",
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        mla_use_output_gate: bool = False,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        input_embedding_scale: float = 1.0,
        output_multiplier: float = 1.0,
        final_logit_softcapping: float | None = None,
        dflash_config: dict | None = None,
        **kwargs,
    ):
        nested = dict(dflash_config or {})
        model_type = kwargs.get("model_type")
        if model_type not in (None, "qwen3", "qwen3_dflash2"):
            raise ValueError(
                f"DFlash2 model_type must be 'qwen3' or 'qwen3_dflash2', got {model_type!r}"
            )

        block_size = int(nested.get("block_size", block_size))
        conv_kernel_size = int(nested.get("conv_kernel_size", conv_kernel_size))
        conv_group_size = int(nested.get("conv_group_size", conv_group_size))
        selector_rank = int(nested.get("selector_rank", selector_rank))
        selector_top_k = int(nested.get("selector_top_k", selector_top_k))
        attention_mode = str(nested.get("attention_mode", attention_mode)).lower()
        input_embedding_scale = float(nested.get("input_embedding_scale", input_embedding_scale))
        output_multiplier = float(nested.get("output_multiplier", output_multiplier))
        final_logit_softcapping = nested.get("final_logit_softcapping", final_logit_softcapping)
        if final_logit_softcapping is not None:
            final_logit_softcapping = float(final_logit_softcapping)
            if final_logit_softcapping == 0.0:
                final_logit_softcapping = None

        nested_target_layer_ids = nested.get("target_layer_ids")
        target_layer_ids = (
            nested_target_layer_ids
            if nested_target_layer_ids is not None
            else kwargs.get("target_layer_ids")
        )
        if target_layer_ids is None:
            target_depth = int(
                kwargs.get("target_num_hidden_layers", kwargs.get("num_target_layers", 36))
            )
            draft_depth = int(kwargs.get("num_hidden_layers", 5))
            target_layer_ids = build_target_layer_ids(draft_depth, target_depth)
            kwargs["target_num_hidden_layers"] = target_depth
            kwargs["num_target_layers"] = draft_depth
            kwargs["target_layer_ids"] = target_layer_ids
        else:
            target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
            target_depth = kwargs.get("num_target_layers")
            if (
                kwargs.get("target_num_hidden_layers") is None
                and target_depth is not None
                and (
                    nested_target_layer_ids is not None
                    or int(target_depth) != len(target_layer_ids)
                )
            ):
                kwargs["target_num_hidden_layers"] = int(target_depth)
            kwargs["num_target_layers"] = len(target_layer_ids)
            kwargs["target_layer_ids"] = target_layer_ids

        if not target_layer_ids:
            raise ValueError("target_layer_ids must contain at least one layer")
        target_depth = int(kwargs.get("target_num_hidden_layers", 36))
        invalid_layer_ids = [
            layer_id for layer_id in target_layer_ids if not 0 <= layer_id < target_depth
        ]
        if invalid_layer_ids:
            raise ValueError(
                f"target_layer_ids must be in [0, {target_depth}), got {invalid_layer_ids}"
            )

        hidden_size = int(kwargs.get("hidden_size", 4096))
        target_hidden_size = kwargs.get("target_hidden_size")
        if target_hidden_size is None:
            target_hidden_size = hidden_size
        if target_hidden_size != hidden_size:
            raise ValueError(
                "DFlash2 target_hidden_size must equal hidden_size, "
                f"got {target_hidden_size} and {hidden_size}"
            )
        kwargs["target_hidden_size"] = hidden_size
        kwargs["mask_token_id"] = int(
            nested.get("mask_token_id", kwargs.get("mask_token_id", 151669))
        )
        rope_parameters = rope_scaling if rope_scaling is not None else rope_parameters
        rope_parameters = dict(rope_parameters) if rope_parameters is not None else None
        rope_parameters_for_validation = rope_parameters or {}
        kwargs.setdefault("rope_theta", rope_parameters_for_validation.get("rope_theta", 10000.0))
        kwargs.pop("model_type", None)

        hidden_act = kwargs.get("hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(f"DFlash2 training requires hidden_act='silu', got {hidden_act!r}")
        if kwargs.get("attention_bias", False):
            raise ValueError("DFlash2 training does not support attention_bias=True")
        if float(kwargs.get("attention_dropout", 0.0)) != 0.0:
            raise ValueError("DFlash2 training requires attention_dropout=0")
        if kwargs.get("fc_norm", False):
            raise ValueError("DFlash2 training does not support fc_norm=True")
        if attention_mode not in {"gqa", "mla"}:
            raise ValueError(
                f"DFlash2 attention_mode must be 'gqa' or 'mla', got {attention_mode!r}"
            )
        if attention_mode == "gqa":
            if rope_scaling is not None:
                raise ValueError("GQA DFlash2 training does not support rope_scaling")
            unsupported_rope_keys = set(rope_parameters_for_validation) - {
                "rope_theta",
                "rope_type",
            }
            if (
                rope_parameters_for_validation.get("rope_type", "default") != "default"
                or unsupported_rope_keys
            ):
                raise ValueError("GQA DFlash2 training supports only default rope_parameters")

        num_hidden_layers = int(kwargs.get("num_hidden_layers", 5))
        use_sliding_window = bool(kwargs.get("use_sliding_window", False))
        sliding_window = kwargs.get("sliding_window")
        layer_types = kwargs.get("layer_types")
        if layer_types is None:
            if use_sliding_window and sliding_window is None:
                raise ValueError(
                    "DFlash2 use_sliding_window requires an explicit positive sliding_window"
                )
            max_window_layers = int(kwargs.get("max_window_layers", 28))
            layer_types = [
                "sliding_attention"
                if use_sliding_window and layer_id >= max_window_layers
                else "full_attention"
                for layer_id in range(num_hidden_layers)
            ]
        else:
            layer_types = list(layer_types)
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                "DFlash2 layer_types must contain one entry per draft layer, got "
                f"{len(layer_types)} for {num_hidden_layers} layers"
            )
        attention_types = set(layer_types)
        if not attention_types <= {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported DFlash2 layer types: {sorted(attention_types)}")
        if "sliding_attention" in attention_types:
            if sliding_window is None:
                raise ValueError(
                    "DFlash2 sliding_attention layers require an explicit positive sliding_window"
                )
            sliding_window = int(sliding_window)
            if sliding_window < 1:
                raise ValueError(f"sliding_window must be positive, got {sliding_window}")
        else:
            sliding_window = None
        kwargs["layer_types"] = layer_types
        kwargs["sliding_window"] = sliding_window

        if block_size < 2:
            raise ValueError(f"block_size must be at least 2, got {block_size}")
        if not 1 <= conv_kernel_size <= block_size:
            raise ValueError(
                f"conv_kernel_size must be in [1, block_size={block_size}], got {conv_kernel_size}"
            )
        if conv_group_size < 1 or hidden_size % conv_group_size:
            raise ValueError(
                f"conv_group_size={conv_group_size} must divide hidden_size={hidden_size}"
            )
        vocab_size = int(kwargs.get("vocab_size", 152064))
        if not 0 <= kwargs["mask_token_id"] < vocab_size:
            raise ValueError(
                f"mask_token_id must be in [0, {vocab_size}), got {kwargs['mask_token_id']}"
            )
        draft_vocab_size = kwargs.get("draft_vocab_size")
        if draft_vocab_size is not None and int(draft_vocab_size) != vocab_size:
            raise ValueError("DFlash2 does not support draft_vocab_size different from vocab_size")
        if selector_rank < 1:
            raise ValueError(f"selector_rank must be positive, got {selector_rank}")
        if not 2 <= selector_top_k <= vocab_size:
            raise ValueError(f"selector_top_k must be in [2, {vocab_size}], got {selector_top_k}")
        if not math.isfinite(input_embedding_scale) or input_embedding_scale <= 0:
            raise ValueError(f"input_embedding_scale must be positive, got {input_embedding_scale}")
        if not math.isfinite(output_multiplier) or output_multiplier <= 0:
            raise ValueError(f"output_multiplier must be positive, got {output_multiplier}")
        if final_logit_softcapping is not None and (
            not math.isfinite(final_logit_softcapping) or final_logit_softcapping < 0
        ):
            raise ValueError(
                f"final_logit_softcapping must be positive, got {final_logit_softcapping}"
            )

        super().__init__(**kwargs)
        self.attention_mode = attention_mode
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.mla_use_output_gate = bool(mla_use_output_gate)
        self.rope_parameters = rope_parameters
        if self.attention_mode == "mla":
            # DeepSeekMLAAttention's shared rotary builder reads the legacy
            # ``rope_scaling`` name.  Keep it populated for Transformers
            # versions where it is not an alias of ``rope_parameters``.
            self.rope_scaling = rope_parameters
        if rope_parameters is not None:
            nested_rope_theta = rope_parameters.get("rope_theta")
            if nested_rope_theta is not None:
                self.rope_theta = float(nested_rope_theta)
            if rope_parameters.get("rope_type", rope_parameters.get("type")) == "yarn":
                for key, default in {
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "mscale": 1.0,
                    "mscale_all_dim": 0.0,
                }.items():
                    rope_parameters.setdefault(key, default)
        self.block_size = block_size
        self.conv_kernel_size = conv_kernel_size
        self.conv_group_size = conv_group_size
        self.selector_rank = selector_rank
        self.selector_top_k = selector_top_k
        self.input_embedding_scale = input_embedding_scale
        self.output_multiplier = output_multiplier
        self.final_logit_softcapping = final_logit_softcapping

        nested.update(
            {
                "attention_mode": self.attention_mode,
                "block_size": self.block_size,
                "conv_kernel_size": self.conv_kernel_size,
                "conv_group_size": self.conv_group_size,
                "selector_rank": self.selector_rank,
                "selector_top_k": self.selector_top_k,
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": self.target_layer_ids,
            }
        )
        if self.input_embedding_scale != 1.0:
            nested["input_embedding_scale"] = self.input_embedding_scale
        if self.output_multiplier != 1.0:
            nested["output_multiplier"] = self.output_multiplier
        if self.final_logit_softcapping is not None:
            nested["final_logit_softcapping"] = self.final_logit_softcapping
        self.dflash_config = nested


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        kernel_size: int,
        group_size: int,
    ):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"group_size={group_size} must divide hidden_size={hidden_size}")
        self.block_size = int(block_size)
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size

        base_kernel = torch.zeros(2, self.kernel_size, hidden_size)
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * self.kernel_size * self.num_groups,
            bias=False,
        )
        nn.init.zeros_(self.kernel_projection.weight)

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        dynamic_kernel: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        batch, length, hidden_size = hidden_states.shape
        if length % self.block_size:
            raise ValueError(
                f"draft length {length} must be divisible by block_size={self.block_size}"
            )

        blocks = hidden_states.reshape(-1, self.block_size, self.num_groups, self.group_size)
        dynamic = dynamic_kernel.reshape(-1, self.block_size, self.kernel_size, self.num_groups)
        base = self.base_kernel[side].reshape(
            1, 1, self.kernel_size, self.num_groups, self.group_size
        )
        coefficients = base.to(hidden_states.dtype) + dynamic.unsqueeze(-1)

        output = torch.zeros_like(blocks)
        for offset in range(self.kernel_size):
            values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
            output = output + coefficients[:, :, offset] * values
        return output.reshape(batch, length, hidden_size)

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1],
            2,
            self.kernel_size,
            self.num_groups,
        )
        return self._convolve(hidden_states, dynamic[..., 0, :, :], 0), dynamic[..., 1, :, :]

    def finish(
        self,
        hidden_states: torch.Tensor,
        dynamic_kernel: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, dynamic_kernel, 1)


class CandidateSelector(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int):
        super().__init__()
        if not 2 <= top_k <= vocab_size:
            raise ValueError(f"top_k must be in [2, {vocab_size}], got {top_k}")
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        nn.init.normal_(self.predecessor_codebook, std=0.02)
        nn.init.normal_(self.successor_codebook, std=0.02)

    def score_candidates(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        predecessor_ids: torch.Tensor,
        *,
        training_successor_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unary_logits, candidate_ids = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        if training_successor_ids is not None:
            matches = candidate_ids == training_successor_ids.unsqueeze(-1)
            missing = ~matches.any(dim=-1)
            replace_at = unary_logits.argmin(dim=-1, keepdim=True)
            current_ids = candidate_ids.gather(-1, replace_at)
            current_logits = unary_logits.gather(-1, replace_at)
            replacement_ids = torch.where(
                missing.unsqueeze(-1), training_successor_ids.unsqueeze(-1), current_ids
            )
            replacement_logits = torch.where(
                missing.unsqueeze(-1),
                logits.gather(-1, training_successor_ids.unsqueeze(-1)),
                current_logits,
            )
            candidate_ids = candidate_ids.scatter(-1, replace_at, replacement_ids)
            unary_logits = unary_logits.scatter(-1, replace_at, replacement_logits)
        hidden = self.hidden_projection(hidden_states)
        predecessor = self.predecessor_codebook[predecessor_ids] * hidden
        successor = self.successor_codebook[candidate_ids]
        scores = unary_logits + torch.einsum("...r,...kr->...k", predecessor, successor)
        return scores, candidate_ids


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlash2Config):
        if config.attention_mode == "mla":
            from torchspec.models.draft.dspark import K3DSparkMLAAttention

            self.attention_class = K3DSparkMLAAttention
        super().__init__(config)
        conv_args = {
            "hidden_size": config.hidden_size,
            "block_size": config.block_size,
            "kernel_size": config.conv_kernel_size,
            "group_size": config.conv_group_size,
        }
        self.attention_conv = DFlashGroupedConv(**conv_args)
        self.mlp_conv = DFlashGroupedConv(**conv_args)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        residual = draft_hidden
        draft_hidden = self.input_layernorm(draft_hidden)
        draft_hidden, attention_kernel = self.attention_conv.prepare(draft_hidden)
        draft_hidden = self.self_attn(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
        )
        draft_hidden = self.attention_conv.finish(draft_hidden, attention_kernel)
        draft_hidden = residual + draft_hidden

        residual = draft_hidden
        draft_hidden = self.post_attention_layernorm(draft_hidden)
        draft_hidden, mlp_kernel = self.mlp_conv.prepare(draft_hidden)
        draft_hidden = self.mlp(draft_hidden)
        draft_hidden = self.mlp_conv.finish(draft_hidden, mlp_kernel)
        return residual + draft_hidden


class DFlash2DraftModel(DFlashDraftModel):
    config_class = DFlash2Config
    decoder_layer_class = DFlash2DecoderLayer

    def __init__(self, config: DFlash2Config):
        super().__init__(config)
        self.candidate_selector = CandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            rank=config.selector_rank,
            top_k=config.selector_top_k,
        )

    def config_for_serving(self) -> dict:
        return dflash2_config_for_serving(self.config.to_dict())

    def state_dict_for_serving(
        self,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if state_dict is None:
            state_dict = self.state_dict()
        return dflash2_state_dict_for_serving(state_dict)

    def forward(
        self,
        draft_input_ids: torch.Tensor | None,
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise_embedding is None:
            noise_embedding = self.embed_tokens(draft_input_ids)
            draft_input_ids = None
        noise_embedding = noise_embedding * self.config.input_embedding_scale
        return super().forward(
            draft_input_ids=draft_input_ids,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )

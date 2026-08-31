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

"""Shared modules for Eagle3 draft models.

Extracted common components used across different architecture-specific
draft implementations (Llama, DeepSeek, etc.) to reduce code duplication
and simplify maintenance.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class EagleRMSNorm(nn.Module):
    """RMS normalization layer.

    Equivalent to T5LayerNorm. Used across multiple draft architectures
    (Llama, DeepSeek, etc.).

    Args:
        hidden_size: Dimension of the input tensor.
        eps: Epsilon value for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @torch.compile(dynamic=True)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class EagleMLP(nn.Module):
    """Multi-layer perceptron with gated activation.

    Standard transformer feed-forward network using gate-up-down projection
    pattern with configurable activation function.

    Args:
        config: Model configuration object with attributes:
            - hidden_size: Input/output dimension
            - intermediate_size: Hidden dimension of MLP
            - hidden_act: Activation function name (e.g., "silu", "gelu")
            - pretraining_tp: Tensor parallelism degree (default: 1)
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.pretraining_tp > 1:
            slice_size = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_size, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_size, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_size, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice_size, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def eagle_decoder_layer_forward(
    self,
    input_emb: torch.Tensor,
    hidden_states: torch.Tensor,
    cache_keys: Optional[torch.Tensor] = None,
    cache_values: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Generic forward pass for Eagle3 decoder layers.

    This function implements the standard Eagle3 decoder layer pattern:
    1. Pre-normalization of hidden states and input embeddings
    2. Concatenation of input embedding with hidden states
    3. Self-attention with optional caching
    4. Residual connection
    5. Post-attention normalization and MLP
    6. Final residual connection

    Used by both LlamaDecoderLayer and DeepSeekDecoderLayer to avoid
    code duplication.

    Args:
        self: Decoder layer instance with attributes:
            - hidden_norm: Normalization for hidden_states
            - input_layernorm: Normalization for input_emb
            - self_attn: Attention module
            - post_attention_layernorm: Post-attention normalization
            - mlp: MLP module
        input_emb: Input embeddings [batch, seq_len, hidden_size]
        hidden_states: Current hidden states [batch, seq_len, hidden_size]
        cache_keys: Cached key tensors for attention (optional)
        cache_values: Cached value tensors for attention (optional)
        attention_mask: Attention mask tensor (optional)
        position_ids: Position IDs for RoPE (optional)
        use_cache: Whether to use KV caching

    Returns:
        Tuple of (hidden_states, cache_keys, cache_values)
    """
    residual = hidden_states

    hidden_states = self.hidden_norm(hidden_states)
    input_emb = self.input_layernorm(input_emb)

    # Eagle3: concatenate input embedding and hidden states
    hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

    # Self Attention
    hidden_states, cache_keys, cache_values = self.self_attn(
        hidden_states=hidden_states,
        cache_keys=cache_keys,
        cache_values=cache_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # MLP
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, cache_keys, cache_values


def eagle_embed_input_ids(
    self, input_ids: torch.Tensor
) -> torch.Tensor:
    """Embed input IDs using the model's token embedding layer.

    Generic implementation used across all Eagle3 draft models.

    Args:
        self: Model instance with embed_tokens attribute
        input_ids: Input token IDs [batch, seq_len]

    Returns:
        Embedded tokens [batch, seq_len, hidden_size]
    """
    return self.embed_tokens(input_ids)


def eagle_project_hidden_states(
    self, hidden_states: torch.Tensor, use_fp32_proj: bool = True
) -> torch.Tensor:
    """Project target hidden states to draft model dimension.

    Generic projection logic with optional FC normalization and FP32
    computation for numerical stability.

    Args:
        self: Model instance with attributes:
            - fc: Projection linear layer
            - fc_norm: Optional list of normalization layers (one per aux state)
            - num_aux_hidden_states: Number of auxiliary hidden state chunks
        hidden_states: Target model hidden states to project
        use_fp32_proj: Whether to perform projection in FP32 (default: True)

    Returns:
        Projected hidden states matching draft model dimension

    Raises:
        ValueError: If input hidden size doesn't match expected size
    """
    expected_size = self.fc.in_features
    if hidden_states.size(-1) != expected_size:
        raise ValueError(
            f"Target hidden states size mismatch: {hidden_states.size(-1)} != expected: {expected_size}"
        )
    if self.fc_norm is not None:
        chunks = hidden_states.chunk(self.num_aux_hidden_states, dim=-1)
        hidden_states = torch.cat(
            [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
            dim=-1,
        )
    if use_fp32_proj and os.environ.get("TORCHSPEC_EAGLE3_PROJ_FP32", "1") in {"0", "false", "False"}:
        use_fp32_proj = False

    if use_fp32_proj:
        proj = F.linear(
            hidden_states.to(torch.float32),
            self.fc.weight.to(torch.float32),
            None if self.fc.bias is None else self.fc.bias.to(torch.float32),
        )
        return proj.to(self.fc.weight.dtype)
    return self.fc(hidden_states.to(self.fc.weight.dtype))


def eagle_compute_logits(
    self, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Compute logits from hidden states.

    Generic implementation: apply final normalization then LM head.

    Args:
        self: Model instance with attributes:
            - norm: Final layer normalization
            - lm_head: Language model head (linear or factorized)

    Returns:
        Logits [batch, seq_len, vocab_size]
    """
    norm_hidden_states = self.norm(hidden_states)
    return self.lm_head(norm_hidden_states)


def eagle_backbone(
    self,
    input_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_keys: Optional[torch.Tensor] = None,
    cache_values: Optional[torch.Tensor] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Forward pass through the draft model backbone (single midlayer).

    Generic wrapper that delegates to the model's midlayer module.

    Args:
        self: Model instance with midlayer attribute
        input_embeds: Input embeddings
        hidden_states: Projected hidden states from target model
        attention_mask: Attention mask
        position_ids: Position IDs
        cache_keys: Cached keys (optional)
        cache_values: Cached values (optional)
        use_cache: Whether to use caching

    Returns:
        Tuple of (output_hidden_states, cache_keys, cache_values)
    """
    return self.midlayer(
        input_emb=input_embeds,
        hidden_states=hidden_states,
        cache_keys=cache_keys,
        cache_values=cache_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=use_cache,
    )

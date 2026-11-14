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

"""DFlash draft model: block-diffusion drafter with dual-source KV injection.

Architecture overview:
  - W_proj projects concatenated multi-layer target hidden states into context features
  - Each decoder layer uses dual-source KV: context KV (from target) + draft KV (from draft)
  - Bidirectional attention within each block; no inter-block attention
  - Shared embedding and LM head from target model (frozen)
"""

import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

from torchspec.config.utils import _normalize_rope_scaling
from torchspec.models.draft.llama3_eagle import build_rotary_embedding


class DFlashConfig(PretrainedConfig):
    """Configuration for DFlash draft model."""

    model_type = "dflash"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        vocab_size: int = 152064,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        num_target_layers: int = 5,
        target_hidden_size: int = 4096,
        target_num_hidden_layers: int = 36,
        target_layer_ids: Optional[List[int]] = None,
        fc_norm: bool = False,
        mask_token_id: int = 151669,
        tie_word_embeddings: bool = False,
        rope_scaling: Optional[dict] = None,
        rope_parameters: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_target_layers = num_target_layers
        self.target_hidden_size = target_hidden_size
        self.target_num_hidden_layers = target_num_hidden_layers
        self.target_layer_ids = target_layer_ids
        self.fc_norm = bool(fc_norm)
        self.mask_token_id = mask_token_id

        # transformers 5.x aliases rope_scaling to rope_parameters (a property
        # on PretrainedConfig), so both names must resolve to one normalized
        # dict. It is kept out of PretrainedConfig.__init__, whose rope
        # validation reads fields set above and wants rope_theta nested. An
        # explicit legacy rope_scaling wins over rope_parameters; a nested
        # rope_theta stays in the dict for serving-config round trips but is
        # also lifted to the top-level attribute the rotary embedding reads.
        params = _normalize_rope_scaling(
            rope_scaling if rope_scaling is not None else rope_parameters
        )
        if params is not None and params.get("rope_theta") is not None:
            self.rope_theta = float(params["rope_theta"])
        self.rope_parameters = params
        # Transformers 4.x stores the legacy field independently.
        if not hasattr(self, "rope_scaling"):
            self.rope_scaling = params


class DFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotary_position_span(position_ids: torch.Tensor) -> int:
    """Table length needed to index these positions, including any offsets."""
    return int(position_ids.max()) + 1 if position_ids.numel() else 0


# NOTE: Not called in DFlashAttention.forward() (RoPE is applied inline there),
# but kept as a standard utility matching SpecForge's apply_rotary_pos_emb.
def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class DFlashAttention(nn.Module):
    """Dual-source KV attention for DFlash.

    K/V come from two sources concatenated along the sequence dimension:
      1. Context KV: projected from target model's context features (via shared W_k/W_v)
      2. Draft KV: projected from draft model's own hidden states (via same W_k/W_v)

    Q comes only from the draft model's hidden states.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Q-norm and K-norm (Qwen3 architecture requirement)
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = build_rotary_embedding(
            config, self.head_dim, self.max_position_embeddings, supports_mrope=False
        )

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        *,
        position_span: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass with dual-source KV.

        Args:
            draft_hidden: [B, draft_len, D] — hidden states of draft tokens
            context_hidden: [B, ctx_len, D] — context features from target
            draft_position_ids: [B, draft_len] — position IDs for draft tokens
            context_position_ids: [B, ctx_len] — position IDs for context tokens
            block_mask: FlexAttention BlockMask for block-causal attention
            position_span: Precomputed table length for the supplied position IDs
        """
        bsz, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]

        # Q only from draft
        q = self.q_proj(draft_hidden)
        q = q.view(bsz, draft_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, num_heads, draft_len, head_dim]

        # K/V from both context and draft (shared projections)
        k_ctx = self.k_proj(context_hidden)
        v_ctx = self.v_proj(context_hidden)
        k_draft = self.k_proj(draft_hidden)
        v_draft = self.v_proj(draft_hidden)

        # Concatenate K and V along sequence dimension BEFORE normalization
        # This matches SpecForge: concat → K-norm → RoPE
        k = torch.cat([k_ctx, k_draft], dim=1)  # [B, ctx+draft, kv_dim]
        v = torch.cat([v_ctx, v_draft], dim=1)

        total_len = ctx_len + draft_len
        k = k.view(bsz, total_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # [B, num_kv_heads, ctx+draft, head_dim]
        v = v.view(bsz, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE with concatenated position IDs (context + draft)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        if position_span is None:
            position_span = rotary_position_span(full_position_ids)
        cos, sin = self.rotary_emb(q, seq_len=position_span)
        cos = cos.to(q.device)
        sin = sin.to(q.device)

        # Apply RoPE to Q (using draft positions only — last draft_len of full_position_ids)
        cos_q = cos.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
        sin_q = sin.squeeze(1).squeeze(0)[draft_position_ids].unsqueeze(1)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)

        # Apply RoPE to K (using full concatenated positions)
        cos_k = cos.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(1)
        sin_k = sin.squeeze(1).squeeze(0)[full_position_ids].unsqueeze(1)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

        if block_mask is not None:
            from torchspec.models.ops.flex_attention import compile_friendly_flex_attention

            # Use enable_gqa=True to let FlexAttention handle GQA internally
            # instead of materializing expanded KV via _repeat_kv
            attn_output = compile_friendly_flex_attention(
                query=q,
                key=k,
                value=v,
                block_mask=block_mask,
                enable_gqa=True,
            )
        else:
            # Fallback: bidirectional attention (no mask) — expand KV for SDPA
            k = _repeat_kv(k, self.num_kv_groups)
            v = _repeat_kv(v, self.num_kv_groups)
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, draft_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    """Single transformer decoder layer for DFlash draft model."""

    # Subclasses (e.g. K3 MLA) swap the attention implementation only.
    attention_class = DFlashAttention

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.self_attn = self.attention_class(config)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        *,
        position_span: Optional[int] = None,
    ) -> torch.Tensor:
        residual = draft_hidden
        draft_hidden = self.input_layernorm(draft_hidden)

        draft_hidden = self.self_attn(
            draft_hidden=draft_hidden,
            context_hidden=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            position_span=position_span,
        )
        draft_hidden = residual + draft_hidden

        residual = draft_hidden
        draft_hidden = self.post_attention_layernorm(draft_hidden)
        draft_hidden = self.mlp(draft_hidden)
        draft_hidden = residual + draft_hidden

        return draft_hidden


def build_target_layer_ids(num_target_layers: int, num_hidden_layers: int) -> List[int]:
    """Compute uniformly spaced layer IDs from the target model.

    Matches SpecForge's build_target_layer_ids() exactly:
      start = 1, end = num_hidden_layers - 3, span = end - start
      For num_target_layers=5 and num_hidden_layers=36:
        start=1, end=33, span=32
        → [1, 9, 17, 25, 33]

    Note: SpecForge convention — num_target_layers here is the number of layers
    to capture (called num_draft_layers in SpecForge). num_hidden_layers is the
    total number of target model decoder layers.
    """
    if num_target_layers == 1:
        return [num_hidden_layers // 2]
    start = 1
    end = num_hidden_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_target_layers - 1))) for i in range(num_target_layers)
    ]


class DFlashDraftModel(PreTrainedModel):
    """DFlash draft model with dual-source KV injection.

    Trainable parameters:
      - W_proj: Linear(num_target_layers * target_hidden_size, hidden_size)
      - proj_norm: RMSNorm after projection
      - N decoder layers (each with attention + FFN)

    Frozen (from target):
      - embed_tokens: token embedding
      - LM head is external (loaded separately in trainer)
    """

    config_class = DFlashConfig
    decoder_layer_class = DFlashDecoderLayer
    # See Eagle3DraftModel.all_tied_weights_keys.
    all_tied_weights_keys = {}

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        num_target_layers = getattr(config, "num_target_layers", 5)
        self.num_target_layers = num_target_layers
        self.mask_token_id = getattr(config, "mask_token_id", 151669)

        # Target layer IDs for hidden state extraction
        target_num_hidden = getattr(config, "target_num_hidden_layers", 36)
        self.target_layer_ids = getattr(config, "target_layer_ids", None)
        if self.target_layer_ids is None:
            self.target_layer_ids = build_target_layer_ids(num_target_layers, target_num_hidden)

        # Context feature projection: concat(multi-layer hidden) → hidden_size
        proj_input_dim = num_target_layers * target_hidden_size
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [
                    DFlashRMSNorm(target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(num_target_layers)
                ]
            )
        else:
            self.fc_norm = None
        self.context_proj = nn.Linear(proj_input_dim, self.hidden_size, bias=False)
        self.context_norm = DFlashRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Token embedding (shared from target, frozen)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList(
            [self.decoder_layer_class(config) for _ in range(self.num_layers)]
        )

        # Final norm
        self.final_norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def extract_context_feature(self, all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Extract and project context features from target hidden states.

        Args:
            all_hidden_states: list of [B, seq_len, D] tensors from target layers

        Returns:
            context_feature: [B, seq_len, hidden_size]
        """
        if self.fc_norm is not None:
            all_hidden_states = [
                norm(hidden_states)
                for norm, hidden_states in zip(self.fc_norm, all_hidden_states, strict=True)
            ]
        concatenated = torch.cat(all_hidden_states, dim=-1).to(self.context_proj.weight.dtype)
        projected = self.context_proj(concatenated)
        return self.context_norm(projected)

    def forward(
        self,
        draft_input_ids: Optional[torch.Tensor],
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: Optional[torch.Tensor] = None,
        *,
        position_span: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass through draft model.

        Args:
            draft_input_ids: [B, draft_len] — token IDs (anchor + MASK tokens).
                Ignored if noise_embedding is provided.
            context_feature: [B, ctx_len, D] — projected context from target
            draft_position_ids: [B, draft_len]
            context_position_ids: [B, ctx_len]
            block_mask: FlexAttention BlockMask, or a per-layer sequence of
                BlockMasks for mixed attention schedules
            noise_embedding: [B, draft_len, D] — pre-computed embeddings (from training wrapper)
            position_span: Precomputed table length for the supplied position IDs

        Returns:
            hidden_states: [B, draft_len, D] — pre-norm hidden states
        """
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_feature.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(context_feature.dtype)

        if position_span is None:
            position_span = rotary_position_span(
                torch.cat([context_position_ids, draft_position_ids], dim=1)
            )

        for layer_id, layer in enumerate(self.layers):
            layer_block_mask = block_mask
            if isinstance(block_mask, (list, tuple)):
                layer_block_mask = block_mask[layer_id]
            draft_hidden = layer(
                draft_hidden=draft_hidden,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=layer_block_mask,
                position_span=position_span,
            )

        return self.final_norm(draft_hidden)

    def freeze_embedding(self) -> None:
        self.embed_tokens.weight.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """Load embedding weights from target model checkpoint."""
        if os.path.exists(model_path):
            glob_path = os.path.join(model_path, "*.index.json")
            import glob as glob_mod

            index_json_path = glob_mod.glob(glob_path)

            if len(index_json_path) == 0:
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    with safe_open(safetensors_path, framework="pt") as f:
                        self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
                    return
                pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    state_dict = torch.load(
                        pytorch_model_path, map_location="cpu", weights_only=True
                    )
                    self.embed_tokens.weight.copy_(state_dict[embedding_key])
                    return
                raise FileNotFoundError(
                    f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
                )
            index_json_path = index_json_path[0]
            with open(index_json_path, "r") as f:
                index_json = json.load(f)
            ckpt_file = index_json["weight_map"][embedding_key]
            if ckpt_file.endswith(".safetensors"):
                with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
                    self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
            else:
                state_dict = torch.load(
                    os.path.join(model_path, ckpt_file), map_location="cpu", weights_only=True
                )
                self.embed_tokens.weight.copy_(state_dict[embedding_key])
        else:
            local_cache_path = snapshot_download(repo_id=model_path)
            self.load_embedding(local_cache_path, embedding_key)

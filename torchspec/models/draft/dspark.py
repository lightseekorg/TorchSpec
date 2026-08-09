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

"""
DSpark draft model: DFlash backbone + EAGLE-style Markov and confidence heads.

DSpark shares DFlash's block-diffusion drafter (dual-source KV injection, anchor
sampling, MASK-token noise stream) and adds two heads on top:

  - Markov head: a low-rank learned bigram bias added to the draft logits,
    conditioned on the (teacher-forced) previous token. Improves the per-token
    distribution without touching the backbone.
  - Confidence head (AcceptRatePredictor): predicts a per-draft-position
    acceptance probability, trained against the empirical draft-vs-target
    accept rate (used at inference time for adaptive block length).

Markov / confidence modeling code is adapted from DeepSeek's DeepSpec
(deepspec/modeling/dspark/{markov_head,common}.py, MIT License).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.draft.deepseek_eagle import (
    DeepSeekMLAAttention,
    _rotate_half_interleaved,
)
from torchspec.models.draft.dflash import (
    DFlashConfig,
    DFlashDecoderLayer,
    DFlashDraftModel,
    DFlashRMSNorm,
)
from torchspec.models.ops.flex_attention import compile_friendly_flex_attention


class DSparkConfig(DFlashConfig):
    """
    Configuration for the DSpark draft model. Extends :class:`DFlashConfig`.
    """

    model_type = "qwen3_dspark"

    def __init__(
        self,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov


class VanillaMarkov(nn.Module):
    """
    Adapted from DeepSpec's ``deepspec/modeling/dspark/markov_head.py``.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        assert self.markov_rank > 0, (
            f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        # TODO: markow_w2 out_features should match "draft_vocab_size" if pruning is used.
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids)


class AcceptRatePredictor(nn.Module):
    """
    Adapted from DeepSpec's ``deepspec/modeling/dspark/common.py``.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None

    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=config.vocab_size, markov_rank=markov_rank)
    raise NotImplementedError(
        f"markov_head_type={markov_head_type!r} is not supported yet; only 'vanilla' "
        "is implemented in TorchSpec as it is recommended by the authors."
    )


class DSparkDraftModel(DFlashDraftModel):
    config_class = DSparkConfig

    def __init__(self, config: DSparkConfig):
        super().__init__(config)

        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )

        self.markov_head = build_markov_head(config)

        self.confidence_head: Optional[nn.Module] = None
        if getattr(config, "enable_confidence_head", False):
            conf_input_dim = self.hidden_size
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "confidence_head_with_markov=True requires a Markov head (markov_rank > 0)."
                    )
                conf_input_dim += self.markov_rank
            self.confidence_head = AcceptRatePredictor(conf_input_dim)


class K3DSparkConfig(DSparkConfig):
    model_type = "k3_dspark"

    def __init__(
        self,
        q_lora_rank: Optional[int] = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        mla_use_output_gate: bool = False,
        rope_scaling: Optional[dict] = None,
        rope_parameters: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_output_gate = bool(mla_use_output_gate)

        # transformers 5.x aliases rope_scaling to rope_parameters (a property
        # on PretrainedConfig), so both names must resolve to one normalized
        # dict. An explicit legacy rope_scaling wins over rope_parameters;
        # rope_theta stays nested in the dict for serving-config round trips
        # but is also lifted to the top-level attribute _init_rope reads.
        params = rope_scaling if rope_scaling is not None else rope_parameters
        if params is not None:
            params = dict(params)
            rope_theta = params.get("rope_theta")
            if rope_theta is not None:
                self.rope_theta = float(rope_theta)
            if params.get("rope_type", params.get("type")) == "yarn":
                yarn_defaults = {
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "mscale": 1.0,
                    "mscale_all_dim": 0.0,
                }
                for key, default in yarn_defaults.items():
                    if params.get(key) is None:
                        params[key] = default

        self.rope_parameters = params


# Deliberately not @torch.compile'd: like DFlashAttention's inline RoPE, this
# runs eagerly and is captured by the model-level compile when enabled.
def _apply_rope_interleaved(x, cos, sin, position_ids, unsqueeze_dim=1):
    """Interleaved-pair RoPE for one tensor, with its own position ids.

    Unlike deepseek_eagle's paired helper, Q and K use different positions here
    (draft-only vs context+draft), so each side is rotated independently.
    """
    cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(unsqueeze_dim)
    half = cos.shape[-1] // 2
    cos = cos[..., :half].repeat_interleave(2, dim=-1)
    sin = sin[..., :half].repeat_interleave(2, dim=-1)
    return (x * cos) + (_rotate_half_interleaved(x) * sin)


class K3DSparkMLAAttention(nn.Module):
    """DeepSeek MLA attention with DFlash dual-source KV for the K3 draft.

    - Q is projected from the draft hidden states only.
    - K/V are projected from cat(context, draft) with the same weights.
    - RoPE rotates only the qk_rope_head_dim side dims: Q with
      draft_position_ids, K with cat(context, draft) position ids.
    - V is never rotated; attention is bidirectional under the block mask.
    """

    _init_rope = DeepSeekMLAAttention._init_rope
    _compute_softmax_scale = DeepSeekMLAAttention._compute_softmax_scale

    def __init__(self, config: K3DSparkConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if getattr(config, "mla_use_output_gate", False):
            raise NotImplementedError(
                "mla_use_output_gate=True is not supported; the published "
                "Kimi-K3-DSpark checkpoint carries no gate weights."
            )
        # K3 declares a ~1M-token max context; pre-building cos/sin caches at
        # that length would cost ~0.5 GB per layer. Start small — the rotary
        # cache grows on demand and yarn inv_freq is independent of its length.
        self.max_position_embeddings = min(int(config.max_position_embeddings), 32768)

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = DFlashRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = DFlashRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        self.rotary_emb = None
        self._init_rope()
        self.softmax_scale = self._compute_softmax_scale()

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
    ) -> torch.Tensor:
        bsz, draft_len, _ = draft_hidden.shape
        ctx_len = context_hidden.shape[1]
        total_len = ctx_len + draft_len

        # Q from draft hidden states only
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(draft_hidden)))
        else:
            q = self.q_proj(draft_hidden)
        q = q.view(bsz, draft_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # K/V from both sources through the shared MLA projections
        kv_input = torch.cat([context_hidden, draft_hidden], dim=1)
        kv_combined = self.kv_a_proj_with_mqa(kv_input)
        kv_compressed, k_rope = torch.split(
            kv_combined, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
        kv = kv.view(bsz, total_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value = value.transpose(1, 2)

        # Single shared rope head (MQA-style): [B, 1, ctx+draft, rope_dim]
        k_rope = k_rope.unsqueeze(1)

        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        cos, sin = self.rotary_emb(q_rope, seq_len=total_len)
        cos = cos.to(draft_hidden.device)
        sin = sin.to(draft_hidden.device)
        q_rope = _apply_rope_interleaved(q_rope, cos, sin, draft_position_ids)
        k_rope = _apply_rope_interleaved(k_rope, cos, sin, full_position_ids)

        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], dim=-1)

        if block_mask is not None:
            attn_output = compile_friendly_flex_attention(
                query=query_states,
                key=key_states,
                value=value.contiguous(),
                block_mask=block_mask,
                scale=self.softmax_scale,
            )
        else:
            # Fallback: bidirectional attention (no mask)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value,
                is_causal=False,
                dropout_p=0.0,
                scale=self.softmax_scale,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, draft_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


class K3DSparkDecoderLayer(DFlashDecoderLayer):
    attention_class = K3DSparkMLAAttention


class K3DSparkModel(DSparkDraftModel):
    """K3 DSpark draft model: DFlash skeleton with MLA attention layers."""

    config_class = K3DSparkConfig
    decoder_layer_class = K3DSparkDecoderLayer

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers.modeling_utils import PreTrainedModel


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class LowRankHead(nn.Module):
    """Factorized output projection of rank ``hidden_size // divisor``.

    ``down.weight`` is ``[rank, hidden]`` and ``up.weight`` is ``[vocab, rank]``; the dense
    product is never materialized.
    """

    def __init__(self, hidden_size: int, vocab_size: int, divisor: int) -> None:
        super().__init__()
        if divisor < 2:
            raise ValueError(f"lm_head_rank_divisor must be at least 2, got {divisor}")
        rank = hidden_size // divisor
        if rank < 1:
            raise ValueError(
                f"lm_head_rank_divisor={divisor} leaves no rank for hidden_size={hidden_size}"
            )
        self.rank = rank
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden_states))


def build_lm_head(config, hidden_size: int, vocab_size: int, target_vocab_size: int) -> nn.Module:
    """Dense head, or a ``LowRankHead`` when the config sets ``lm_head_rank_divisor``."""
    divisor = getattr(config, "lm_head_rank_divisor", None)
    if not divisor:
        return nn.Linear(hidden_size, vocab_size, bias=False)
    if vocab_size != target_vocab_size:
        raise ValueError(
            f"lm_head_rank_divisor is not supported with a pruned draft vocabulary "
            f"(draft_vocab_size={vocab_size} < vocab_size={target_vocab_size})."
        )
    return LowRankHead(hidden_size, vocab_size, divisor)


class Eagle3DraftModel(PreTrainedModel, ABC):
    """
    This is the base class for the Eagle3 draft model implementation. The child class needs to implement
    the abstract methods to support training with TTT.
    """

    # Drafts tie nothing and never call post_init(), where Transformers 5.x would set this.
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)

        self.num_aux_hidden_states = getattr(config, "num_aux_hidden_states", None)
        if self.num_aux_hidden_states is None:
            eagle_config = getattr(config, "eagle_config", None) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            self.num_aux_hidden_states = len(layer_ids) if layer_ids else 3

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed the input ids.
        """

    @abstractmethod
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
        """

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits of the draft model.
        """

    def prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_length: int,
        past_key_values_length: int,
    ) -> torch.Tensor:
        """
        Prepare the attention mask of the draft model.
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(
                (batch_size, seq_length),
                hidden_states.dtype,
                device=hidden_states.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, hidden_states.dtype, tgt_len=seq_length
            ).to(hidden_states.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    @abstractmethod
    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        The backbone of the draft model.
        """

    @property
    def is_lm_head_factorized(self) -> bool:
        return isinstance(self.lm_head, LowRankHead)

    def get_lm_head_params(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return (norm_weight, projection_weight, norm_eps) for fused loss computation.

        For a factorized head the projection is the down weight; pair it with
        ``get_lm_head_up_weight()``. Override for a different normalization scheme.
        """
        weight = self.lm_head.down.weight if self.is_lm_head_factorized else self.lm_head.weight
        return self.norm.weight, weight, self.norm.variance_epsilon

    def get_lm_head_up_weight(self) -> Optional[torch.Tensor]:
        """Second factor ``[vocab, rank]`` of a factorized head, or None when it is dense."""
        return self.lm_head.up.weight if self.is_lm_head_factorized else None

    @property
    def has_vocab_pruning(self) -> bool:
        return hasattr(self, "t2d") and not self.t2d.all()

    @torch.no_grad()
    def set_vocab_buffers(self, d2t: torch.Tensor, t2d: torch.Tensor) -> None:
        """Set the t2d/d2t vocab mapping buffers directly from tensors.

        Refuses to replace a mapping that arrived with loaded weights. ``lm_head`` row j holds the
        j-th target token selected by the mapping, and the target side is pruned with
        ``target_lm_head_weight[t2d]`` in the same order, so swapping in a mapping that selects a
        different token set repoints every row without touching the rows themselves.
        """
        assert hasattr(self, "t2d") and hasattr(self, "d2t"), (
            "t2d and d2t buffers are not found in the draft model"
        )
        if self.has_vocab_pruning and not (
            torch.equal(self.t2d, t2d.to(self.t2d.device))
            and torch.equal(self.d2t, d2t.to(self.d2t.device))
        ):
            raise ValueError(
                "The draft was seeded from weights whose vocabulary mapping selects a different "
                "token set than the one computed for this run. The loaded lm_head rows are "
                "ordered by the old mapping, so replacing the buffers alone would silently "
                "mistrain every row. Either set model.keep_initial_vocab_mapping=true to train "
                "against the mapping the weights were built with, or drop "
                "model.initial_draft_model_path / training.load_path and train the head from "
                "scratch for the new mapping."
            )
        self.t2d.copy_(t2d)
        self.d2t.copy_(d2t)

    def freeze_embedding(self) -> None:
        """
        Freeze the embeddings of the draft model so that they are not updated during training.
        """
        self.embed_tokens.weight.requires_grad = False

    def freeze_lm_head(self) -> None:
        """
        Freeze the lm_head of the draft model so that it is not updated during training.
        """
        for param in self.lm_head.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """
        Load the embedding of the draft model.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
            repository ID or a local directory path containing the model files.
        """
        self.embed_tokens.weight.copy_(load_tensor_from_pretrained(model_path, embedding_key))

    @torch.no_grad()
    def load_lm_head(self, model_path: str, lm_head_key: str = "lm_head.weight") -> None:
        """
        Seed the draft lm_head from the target model's lm_head.

        Only valid without vocabulary pruning. A pruned head's rows are ordered by ``t2d``, which
        is not populated until after model construction (see ``set_vocab_buffers``), so the target
        rows cannot be selected in draft order at this point.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
            repository ID or a local directory path containing the model files.
        """
        if hasattr(self, "t2d"):
            raise ValueError(
                "load_lm_head does not support a pruned draft vocabulary: the head's rows are "
                "ordered by a t2d mapping that is only computed after model init. Seed the head "
                "from model.initial_draft_model_path or training.load_path instead."
            )
        if self.is_lm_head_factorized:
            raise ValueError(
                "load_lm_head does not support a factorized lm_head (lm_head_rank_divisor); "
                "the target head is dense and has no factors to seed."
            )
        weight = load_tensor_from_pretrained(model_path, lm_head_key)
        if weight.shape != self.lm_head.weight.shape:
            raise ValueError(
                f"Target lm_head shape {tuple(weight.shape)} does not match the draft lm_head "
                f"{tuple(self.lm_head.weight.shape)}. Seeding the draft head requires the draft "
                "and target to share hidden size and vocabulary."
            )
        self.lm_head.weight.copy_(weight)


def load_tensor_from_pretrained(model_path: str, key: str) -> torch.Tensor:
    """Read a single tensor by key from an HF checkpoint directory or hub repository id."""
    if not os.path.exists(model_path):
        # model_path is a huggingface repository, so locate its local cache first
        return load_tensor_from_pretrained(snapshot_download(repo_id=model_path), key)

    # check if there is file ending with index.json
    glob_path = os.path.join(model_path, "*.index.json")
    index_json_path = glob.glob(glob_path)

    if len(index_json_path) == 0:
        # No index.json found, look for single model file
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt") as f:
                return f.get_tensor(key)

        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            return state_dict[key]

        raise FileNotFoundError(
            f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
        )
    if len(index_json_path) > 1:
        raise FileNotFoundError(f"Multiple index.json files found in {model_path}")
    index_json_path = index_json_path[0]

    with open(index_json_path, "r") as f:
        index_json = json.load(f)
    ckpt_file = index_json["weight_map"][key]

    if ckpt_file.endswith(".safetensors"):
        with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
            return f.get_tensor(key)
    state_dict = torch.load(os.path.join(model_path, ckpt_file))
    return state_dict[key]

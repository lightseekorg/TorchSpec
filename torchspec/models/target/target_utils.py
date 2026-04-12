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

import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig


class TargetLMHead(nn.Module):
    """
    Efficiently loads only the lm_head from a pretrained model.
    Used for computing logits from last_hidden_states in the trainer.

    When ``load_norm=True``, also loads the final RMSNorm weights so the
    trainer can normalise pre-norm hidden states before the lm_head projection.
    """

    def __init__(self, config):
        super().__init__()
        self.config = getattr(config, "text_config", config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.norm: nn.Module | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        lm_head_key: str = "lm_head.weight",
        norm_key: str = "model.norm.weight",
        load_norm: bool = False,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetLMHead":
        config = AutoConfig.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
        instance = cls(config)

        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(repo_id=model_path, cache_dir=cache_dir)
            except Exception:
                pass

        instance._load_lm_head(local_model_path, lm_head_key)

        if load_norm:
            instance._init_and_load_norm(local_model_path, norm_key)

        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_lm_head(self, model_path: str, lm_head_key: str):
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            if lm_head_key in weight_map:
                file_path = os.path.join(model_path, weight_map[lm_head_key])
                self._load_key_from_file(file_path, lm_head_key)
            else:
                raise KeyError(
                    f"lm_head_key '{lm_head_key}' not found in weight_map. "
                    f"Available keys: {list(weight_map.keys())[:10]}..."
                )
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)
            if target_file:
                self._load_key_from_file(target_file, lm_head_key)
            else:
                raise FileNotFoundError(f"No checkpoint file found in {model_path}")

    def _load_key_from_file(self, file_path: str, key: str):
        tensor = None
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                if key in f.keys():
                    tensor = f.get_tensor(key)
        else:
            state_dict = torch.load(file_path, map_location="cpu")
            if key in state_dict:
                tensor = state_dict[key]
                del state_dict

        if tensor is not None:
            self.lm_head.weight.data.copy_(tensor)
        else:
            raise KeyError(f"Key {key} not found in {file_path}")

    def _init_norm_structure(self) -> None:
        """Create the norm module structure (no weights loaded).

        Used by non-rank-0 processes so that ``parameters()`` yields the
        same count as rank 0 before the broadcast sync.
        """
        import logging

        _log = logging.getLogger(__name__)

        try:
            norm_module = self._extract_norm_from_architecture()
            if norm_module is None:
                return
            self.norm = norm_module.to_empty(device="cpu")
            torch.nn.init.ones_(self.norm.weight)
        except Exception as e:
            _log.warning(f"Failed to create verifier norm structure: {e}")
            self.norm = None

    def _init_and_load_norm(self, model_path: str, norm_key: str) -> None:
        """Extract the final norm module from the target model architecture and load its weight.

        Falls back to no-op if the architecture has no final norm or the
        weight cannot be loaded — the trainer checks ``self.norm is not None``
        before applying it.
        """
        import logging

        _log = logging.getLogger(__name__)

        try:
            norm_module = self._extract_norm_from_architecture()
            if norm_module is None:
                _log.warning(
                    "No final norm found in model architecture "
                    f"(model_type={getattr(self.config, 'model_type', 'unknown')}). "
                    "last_hidden_states will be used without normalization."
                )
                return

            self.norm = norm_module.to_empty(device="cpu")
            self._load_key_into(model_path, norm_key, self.norm.weight)

        except Exception as e:
            _log.warning(
                f"Failed to load verifier norm: {e}. "
                "last_hidden_states will be used without normalization."
            )
            self.norm = None

    def _extract_norm_from_architecture(self) -> "nn.Module | None":
        """Instantiate the model on meta device and return the final norm module."""
        from transformers import AutoModelForCausalLM

        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(
                self.config,
                trust_remote_code=True,
                attn_implementation="eager",
            )

        inner = skeleton
        for attr in ("model", "language_model", "model"):
            inner = getattr(inner, attr, inner)
        norm_module = None
        for name in ("norm", "ln_f", "final_layer_norm"):
            norm_module = getattr(inner, name, None)
            if norm_module is not None:
                break

        del skeleton
        return norm_module

    def _load_key_into(self, model_path: str, key: str, param: torch.nn.Parameter) -> None:
        """Load a single key from safetensors/bin files into a parameter."""
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            if key in weight_map:
                file_path = os.path.join(model_path, weight_map[key])
            else:
                raise KeyError(f"Key '{key}' not found in weight_map")
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            file_path = safetensors[0] if safetensors else (bins[0] if bins else None)
            if file_path is None:
                raise FileNotFoundError(f"No checkpoint file found in {model_path}")

        tensor = None
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                if key in f.keys():
                    tensor = f.get_tensor(key)
        else:
            state_dict = torch.load(file_path, map_location="cpu")
            if key in state_dict:
                tensor = state_dict[key]
                del state_dict

        if tensor is not None:
            param.data.copy_(tensor)
        else:
            raise KeyError(f"Key {key} not found in {file_path}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states."""
        return self.lm_head(hidden_states)


class TargetEmbeddingsAndHead(nn.Module):
    """Loads frozen embed_tokens and lm_head from a pretrained target model.

    Used by DFlash training which needs target embeddings for noise input
    construction and the LM head for computing draft logits.

    Handles weight tying (``tie_word_embeddings``) and multimodal models
    with a ``text_config`` sub-config.
    """

    def __init__(self, config):
        super().__init__()
        self.config = getattr(config, "text_config", config)
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=getattr(self.config, "pad_token_id", None),
        )
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: str = "model.embed_tokens.weight",
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":
        config = AutoConfig.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
        instance = cls(config)

        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(repo_id=model_path, cache_dir=cache_dir)
            except Exception:
                pass

        tie_weights = getattr(config, "tie_word_embeddings", False)
        instance._load_weights(local_model_path, embed_key, lm_head_key, tie_weights)

        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(
        self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool
    ) -> None:
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        weight_map: dict = {}
        files_to_load: dict = {}

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            if embed_key not in weight_map:
                raise KeyError(
                    f"Embedding key '{embed_key}' not found in weight_map. "
                    f"Available keys: {list(weight_map.keys())[:10]}..."
                )
            files_to_load[embed_key] = weight_map[embed_key]

            if not tie_weights:
                if lm_head_key in weight_map:
                    files_to_load[lm_head_key] = weight_map[lm_head_key]
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)
            if not target_file:
                raise FileNotFoundError(f"No checkpoint file found in {model_path}")
            files_to_load[embed_key] = os.path.basename(target_file)
            if not tie_weights:
                files_to_load[lm_head_key] = os.path.basename(target_file)

        # Group keys by file for efficient loading
        file_to_keys: dict = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            file_to_keys.setdefault(full_path, []).append(key)

        loaded_keys: set = set()
        for file_path, keys in file_to_keys.items():
            tensors = self._extract_tensors(file_path, keys)
            for k, tensor in tensors.items():
                if k == embed_key:
                    self.embed_tokens.weight.data.copy_(tensor)
                elif k == lm_head_key:
                    self.lm_head.weight.data.copy_(tensor)
                loaded_keys.add(k)

        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        if embed_key not in loaded_keys:
            raise RuntimeError(f"Failed to load embedding weights for key '{embed_key}'")

    @staticmethod
    def _extract_tensors(file_path: str, keys: list) -> dict:
        result = {}
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys:
                    if k in f.keys():
                        result[k] = f.get_tensor(k)
        else:
            state_dict = torch.load(file_path, map_location="cpu")
            for k in keys:
                if k in state_dict:
                    result[k] = state_dict[k]
            del state_dict
        return result

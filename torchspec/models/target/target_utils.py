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
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class _TargetRMSNorm(nn.Module):
    """Weight-only RMSNorm used when remote model construction is unavailable."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TargetLMHead(nn.Module):
    """
    Efficiently loads only the lm_head from a pretrained model.
    Used for computing logits from last_hidden_states in the trainer.

    When ``load_norm=True``, also loads the final RMSNorm weights so the
    trainer can normalise pre-norm hidden states before the lm_head projection.
    That load fails closed: an unavailable norm is an error rather than a
    warning, since silently skipping it trains against the wrong targets.
    """

    def __init__(self, config):
        super().__init__()
        # Both configs are kept: ``model_config`` is what ``AutoModelForCausalLM``
        # needs to rebuild multimodal architectures (whose remote code expects the
        # outer wrapper), while ``config`` is the text sub-config carrying the
        # hidden/vocab sizes this module actually allocates against.
        self.model_config = config
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
        same count as rank 0 before the broadcast sync. A rank that silently
        ends up with fewer parameters than rank 0 deadlocks that broadcast
        loop, so a missing norm is fatal here rather than a warning.
        """
        norm_module = self._create_norm_module()
        if norm_module is None:
            raise RuntimeError(
                "No final norm structure is available for "
                f"model_type={getattr(self.config, 'model_type', 'unknown')}"
            )
        self.norm = norm_module.to_empty(device="cpu")
        torch.nn.init.ones_(self.norm.weight)

    def _init_and_load_norm(self, model_path: str, norm_key: str) -> None:
        """Create the final norm module and load its checkpoint weight.

        ``load_norm=True`` is a correctness requirement for pre-norm hidden
        states — training against unnormalised ones silently learns against the
        wrong targets — so any construction or weight-loading failure is fatal.
        """
        try:
            norm_module = self._create_norm_module()
            if norm_module is None:
                raise RuntimeError(
                    "No final norm found for "
                    f"model_type={getattr(self.config, 'model_type', 'unknown')}"
                )

            self.norm = norm_module.to_empty(device="cpu")
            self._load_key_into(model_path, norm_key, self.norm.weight)

        except Exception as e:
            self.norm = None
            raise RuntimeError(
                f"Failed to load verifier norm key {norm_key!r} from {model_path!r}"
            ) from e

    def _create_norm_module(self) -> "nn.Module | None":
        """Create the model final norm, with a config-driven RMSNorm fallback.

        Architecture introspection needs the full model to be constructible,
        which fails for targets whose remote code cannot be instantiated from
        config alone. Every such architecture seen so far still ends in a plain
        RMSNorm, so rebuild it from ``hidden_size``/``rms_norm_eps`` instead of
        dropping normalisation entirely.
        """
        architecture_error = None
        try:
            norm_module = self._extract_norm_from_architecture()
        except Exception as exc:
            architecture_error = exc
            norm_module = None

        if norm_module is not None:
            return norm_module

        hidden_size = getattr(self.config, "hidden_size", None)
        rms_norm_eps = getattr(self.config, "rms_norm_eps", None)
        if hidden_size is not None and rms_norm_eps is not None:
            if architecture_error is not None:
                logger.warning(
                    "Final-norm architecture extraction failed; using a "
                    "config-driven RMSNorm fallback (hidden_size=%s, eps=%s): %s",
                    hidden_size,
                    rms_norm_eps,
                    architecture_error,
                )
            return _TargetRMSNorm(hidden_size=hidden_size, eps=rms_norm_eps)

        if architecture_error is not None:
            raise RuntimeError(
                "Failed to construct the target model final norm"
            ) from architecture_error

        return None

    def _extract_norm_from_architecture(self) -> "nn.Module | None":
        """Instantiate the model on meta device and return the final norm module."""
        from transformers import AutoModelForCausalLM

        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(
                self.model_config,
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


def load_synced_target_lm_head(
    model_path: str,
    *,
    load_norm: bool,
    lm_head_key: str = "lm_head.weight",
    norm_key: str = "model.norm.weight",
    trust_remote_code: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> "TargetLMHead":
    """Build a frozen ``TargetLMHead`` on every rank, reading the checkpoint once.

    Rank 0 loads the weights and broadcasts them; the other ranks allocate a
    matching structure to receive them. ``load_norm`` therefore has to resolve the
    same way on all of them: each rank either ends up with the final norm or raises,
    which is what keeps the parameter lists aligned across the broadcast.
    """
    rank = dist.get_rank()

    head: Optional[TargetLMHead] = None
    local_error: Optional[Exception] = None
    try:
        if rank == 0:
            head = TargetLMHead.from_pretrained(
                model_path=model_path,
                lm_head_key=lm_head_key,
                norm_key=norm_key,
                load_norm=load_norm,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            head = TargetLMHead(config)
            if load_norm:
                head._init_norm_structure()
            head.to(device=device, dtype=dtype)
            head.eval()
            head.requires_grad_(False)

        if load_norm and head.norm is None:
            raise RuntimeError(
                "last_hidden_states_prenorm=True requires a loaded verifier norm, "
                f"which is unavailable for {model_path!r}"
            )
    except Exception as e:
        local_error = e

    # Raising straight out of the block above would strand the other ranks in the
    # parameter broadcast below until the process-group timeout, so agree on failure
    # first and let every rank report it.
    initialized = torch.tensor([local_error is None], dtype=torch.int32, device=device)
    dist.all_reduce(initialized, op=dist.ReduceOp.MIN)
    if not initialized.item():
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            f"[Rank {rank}] TargetLMHead initialization failed on another rank; "
            "see that rank's traceback for the root cause"
        )

    for param in head.parameters():
        dist.broadcast(param.data, src=0)

    logger.info(
        "[Rank %s] TargetLMHead initialized and synced from %s (verifier norm: %s)",
        rank,
        model_path,
        "loaded" if head.norm is not None else "not requested",
    )
    return head

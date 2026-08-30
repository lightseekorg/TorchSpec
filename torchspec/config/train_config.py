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

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from torchspec.config.inference_config import InferenceConfig, _validate_inference_batch_config
from torchspec.data.utils import is_local_data_path
from torchspec.utils.logging import logger


@dataclass
class DatasetConfig:
    chat_template: Optional[str] = "llama3"
    defer_tokenization: bool = False
    eval_data_path: Optional[str] = None
    eval_interval: int = 50
    eval_micro_batch_size: Optional[int] = None
    eval_prompt_key: Optional[str] = None
    last_turn_loss_only: Any = "auto"  # bool or "auto"
    length_group_size: int = 32
    min_loss_tokens: int = 0  # DFlash: skip sequences with < N supervised tokens (use 2*block_size)
    prompt_key: str = "conversations"
    renderer: Optional[str] = None  # registered ConversationRenderer; overrides chat_template
    shuffle_dataset: bool = True
    train_data_path: str = ""


@dataclass
class DebugConfig:
    enable_perf_metrics: bool = True
    max_dump_steps: int = 5
    memory_recorder: str = "torch"
    memory_snapshot_dir: str = "."
    memory_snapshot_num_steps: Optional[int] = None
    memory_snapshot_path: str = ""
    profile_dir_name: Optional[str] = "/tmp/torchspec_profiles"
    profile_step_end: int = 0
    profile_step_start: int = 0
    profile_target: list = field(default_factory=lambda: ["train_overall"])
    record_memory_history: bool = False
    save_debug_train_data: Optional[str] = None
    use_pytorch_profiler: bool = False


@dataclass
class LoggingConfig:
    report_to: str = "none"
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_dir: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_host: Optional[str] = None
    wandb_key: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_random_suffix: bool = True
    wandb_run_id: Optional[str] = None
    wandb_team: Optional[str] = None


@dataclass
class ModelConfig:
    draft_model_config: Optional[str] = None
    initial_draft_model_path: Optional[str] = None
    keep_initial_vocab_mapping: bool = False
    embedding_key: str = "model.embed_tokens.weight"
    freeze_lm_head: bool = False
    lm_head_key: str = "lm_head.weight"
    norm_key: str = "model.norm.weight"
    target_model_backend: str = "sglang"
    target_model_path: str = ""
    trust_remote_code: bool = False


@dataclass
class TrainingConfig:
    attention_backend: str = "sdpa"
    colocate: bool = False
    continual_training: bool = False
    distributed_backend: str = "nccl"
    distributed_timeout_minutes: int = 10
    draft_accumulation_steps: int = 1
    eagle3_num_anchors: int = 0
    eagle3_anchor_max_gap: Optional[int] = None
    fsdp_reduce_dtype: str = "float32"  # "float32" or "bfloat16"
    fsdp_strategy: str = "REPLICATE"
    # Controls which workload claims head-node GPUs first under PACK strategy.
    # "training_first" (default), "inference_first", or "custom".
    placement_strategy: str = "training_first"
    training_node_ips: Optional[list[str]] = None
    training_node_selectors: Optional[list[dict[str, str]]] = None
    compile_model: bool = False  # torch.compile the full training model
    sp_ring_size: int = 1
    sp_ulysses_size: int = 1

    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    lk_eta: float = 3.0
    load_path: Optional[str] = None
    loss_type: str = "forward_kl"  # "forward_kl", "lk_alpha", or "lk_lambda" (Eagle3 only)
    lr_decay_style: str = "cosine"
    lr_wsd_decay_ratio: float = 0.2
    lr_wsd_decay_style: str = "cosine"
    lr_total_steps: Optional[int] = None
    max_concurrent_batches: int = 1
    max_grad_norm: float = 0.5
    max_seq_length: int = 8192
    min_lr: float = 0.0
    weight_decay: float = 0.0
    num_epochs: int = 10
    num_train_steps: Optional[int] = None
    micro_batch_size: int = 2
    prefetch_depth: int = 2  # 0 = disabled, >0 = async pre-fetch N batches ahead
    save_interval: int = 5000
    save_per_epoch: bool = False
    max_checkpoints: int = 0  # 0 = keep all, N > 0 = rotate and keep only N most recent
    seed: int = 0
    train_backend: str = "fsdp"
    train_env_vars: str = "{}"
    train_with_decode: bool = False
    training_num_gpus_per_node: int = 1
    training_num_nodes: int = 1
    ttt_length: int = 7
    # Per-position TTT loss weights. If unset, defaults to [0.8**i for i in range(ttt_length)].
    # Length must equal ttt_length when supplied.
    ploss_weights: Optional[list[float]] = None
    warmup_ratio: float = 0.015

    # WSD LR schedule parameters (used by DFlash trainer only)
    wsd_decay_ratio: float = 0.2
    wsd_decay_style: Optional[str] = None

    # DFlash-specific parameters (ignored for Eagle3 training)
    dflash_block_size: int = 16
    dflash_dpace_alpha: float = 0.5
    dflash_loss_decay_gamma: float = 7.0
    dflash_loss_objective: str = "decay"  # "decay" or "dpace"
    dflash_ce_loss_alpha: float = 1.0
    dflash_l1_loss_alpha: float = 0.0
    dflash_num_anchors: int = 512
    dflash_num_target_layers: int = 5

    # DFlash2-specific parameters (used by DFlash2 trainer only)
    dflash2_selector_loss_alpha: float = 1.0

    # DSpark-specific parameters (used by DSpark trainer only)
    dspark_num_anchors: int = 512
    dspark_num_target_layers: int = 5
    dspark_loss_decay_gamma: float = 4.0
    dspark_ce_loss_alpha: float = 0.1
    dspark_l1_loss_alpha: float = 0.9
    dspark_confidence_head_alpha: float = 1.0


@dataclass
class DecodeConfig:
    """Config for train-with-decode mode (speculative decoding during training)."""

    cuda_graph_max_bs: Optional[int] = None
    max_new_tokens: int = 512
    min_new_tokens: int = 2
    stop_token_ids: Optional[list[int]] = None
    max_running_requests: Optional[int] = None
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_num_steps: Optional[int] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    weight_sync_enabled: bool = False
    weight_sync_interval: int = 500


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mooncake: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cache_dir: str = "./cache"
    cache_key: Optional[str] = None
    model_download_dir: Optional[str] = None
    output_dir: str = ""


_ALWAYS_LOCAL_PATH_KEYS = (
    "output_dir",
    "cache_dir",
    "model_download_dir",
    "inference.offline.data_path",
    "model.initial_draft_model_path",
)
_DATA_PATH_KEYS = ("dataset.train_data_path", "dataset.eval_data_path")


def _resolve_relative_paths(
    config: DictConfig,
    base_dir: str,
    *,
    skip_keys: frozenset[str] = frozenset(),
) -> None:
    """Resolve local relative paths in *config* against *base_dir* (in-place).

    Always-local keys (output_dir, cache_dir, …) are absolutized unconditionally.
    Data-path keys are only absolutized when ``is_local_data_path`` says they look
    like filesystem paths (as opposed to HF Hub dataset IDs).

    Keys listed in *skip_keys* are left untouched (useful for deferring
    CWD-relative keys when resolving a file-level config).
    """
    for dotted_key in (*_ALWAYS_LOCAL_PATH_KEYS, *_DATA_PATH_KEYS):
        if dotted_key in skip_keys:
            continue
        val = OmegaConf.select(config, dotted_key, default=None)
        if not (isinstance(val, str) and val):
            continue

        expanded = os.path.expanduser(val)
        if os.path.isabs(expanded):
            if expanded != val:
                OmegaConf.update(config, dotted_key, expanded)
            continue

        if dotted_key in _ALWAYS_LOCAL_PATH_KEYS or is_local_data_path(expanded, base_dir=base_dir):
            OmegaConf.update(config, dotted_key, os.path.abspath(os.path.join(base_dir, expanded)))


def _validate_vllm_config(config: DictConfig) -> None:
    """Raise if the vllm backend is selected with unsupported feature flags."""
    if config.model.target_model_backend != "vllm":
        return
    unsupported_flags = {
        "inference.vllm.enable_multimodal": "enable_multimodal",
        "training.train_with_decode": "train_with_decode",
    }
    for key, label in unsupported_flags.items():
        if OmegaConf.select(config, key):
            raise NotImplementedError(f"{label} is not yet supported with the vllm backend!")


def _validate_vocab_mapping_config(config: DictConfig) -> None:
    """Reusing a loaded vocabulary mapping requires weights that carry one."""
    if not OmegaConf.select(config, "model.keep_initial_vocab_mapping"):
        return
    if not (
        OmegaConf.select(config, "model.initial_draft_model_path")
        or OmegaConf.select(config, "training.load_path")
    ):
        raise ValueError(
            "model.keep_initial_vocab_mapping requires model.initial_draft_model_path or "
            "training.load_path — without loaded weights there is no mapping to keep, and the "
            "draft lm_head would be trained against an all-pass t2d."
        )


def _validate_offline_config(config: DictConfig) -> None:
    if config.inference.inference_engine_type == "offline":
        if not config.inference.offline.data_path:
            raise ValueError(
                "inference.offline.data_path is required when "
                "inference.inference_engine_type=offline"
            )
        if config.training.train_with_decode:
            raise ValueError("training.train_with_decode is not supported in offline mode")
        if config.training.attention_backend == "usp":
            raise ValueError("training.attention_backend=usp is not supported offline")
        if config.inference.offline.num_engines <= 0:
            raise ValueError("inference.offline.num_engines must be positive")


def _validate_training_batch_config(config: DictConfig) -> None:
    if config.training.micro_batch_size < 1:
        raise ValueError(
            "training.micro_batch_size must be positive (>= 1): a value of 0 yields "
            "dispatch_batch_size=0, so try_dispatch_batch no-op-dispatches and every rank "
            "blocks on the data queue, surfacing as an NCCL all-gather timeout"
        )


def _validate_training_numeric_config(config: DictConfig) -> None:
    """Reject non-positive values that would otherwise fail silently or crash late.

    These fields share the fail-closed-at-load principle of #171: a misconfig that
    produces flat loss, sign-flipped gradients, or an opaque post-init crash is
    preferable to surfacing after expensive Ray/mooncake init.
    """
    if config.training.draft_accumulation_steps <= 0:
        raise ValueError(
            f"draft_accumulation_steps must be > 0 (got {config.training.draft_accumulation_steps}); "
            f"<=0 propagates into global_batch_size/lr_total_steps and crashes post-init "
            f"(ZeroDivisionError or bare total_steps assert)"
        )
    if config.training.learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be > 0 (got {config.training.learning_rate}); "
            f"0 yields silent flat loss (untrained checkpoint), <0 hits a late assert"
        )
    if config.training.max_grad_norm <= 0:
        raise ValueError(
            f"max_grad_norm must be > 0 (got {config.training.max_grad_norm}); "
            f"0 zeroes all grads (silent flat loss), <0 sign-flips grads (silent gradient ascent)"
        )


def _save_config_snapshot(config: DictConfig) -> None:
    """Save the resolved config to output_dir/config.yaml if output_dir is set."""
    output_dir = OmegaConf.select(config, "output_dir", default=None)
    if not output_dir:
        return
    dest = Path(output_dir) / "config.yaml"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, str(dest))
        logger.info(f"Saved resolved config to {dest}")
    except OSError as e:
        logger.warning(f"Failed to save config to {dest}: {e}")


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[list] = None,
    base_config: Optional[DictConfig] = None,
    save_snapshot: bool = False,
) -> DictConfig:
    schema = OmegaConf.structured(Config)

    configs_to_merge = [schema]

    if base_config is not None:
        configs_to_merge.append(base_config)

    if config_path is not None:
        file_config = OmegaConf.load(config_path)
        _resolve_relative_paths(
            file_config,
            os.path.dirname(os.path.abspath(config_path)),
            skip_keys=frozenset(_ALWAYS_LOCAL_PATH_KEYS),
        )
        configs_to_merge.append(file_config)

    if cli_args:
        cli_config = OmegaConf.from_dotlist(cli_args)
        configs_to_merge.append(cli_config)

    config = OmegaConf.merge(*configs_to_merge)
    _resolve_relative_paths(config, os.getcwd())

    _validate_vllm_config(config)
    _validate_offline_config(config)
    _validate_vocab_mapping_config(config)
    _validate_training_batch_config(config)
    _validate_training_numeric_config(config)
    _validate_inference_batch_config(config)

    if save_snapshot:
        _save_config_snapshot(config)

    return config


# Sub-sections whose fields receive a name prefix when flattened.
_PREFIXED_SECTIONS = {
    "decode": "decode_",
    "mooncake": "mooncake_",
    "offline": "offline_",
    "sglang": "sglang_",
    "vllm": "vllm_",
    "trtllm": "trtllm_",
    "tokenspeed": "tokenspeed_",
}


def config_to_flat_args(config: DictConfig) -> argparse.Namespace:
    flat: dict[str, Any] = {}

    def _add(key: str, val: Any, origin: str) -> None:
        if key in flat:
            raise ValueError(f"Duplicate config key '{key}' (from '{origin}')")
        flat[key] = val

    for section_name, section in config.items():
        if not isinstance(section, DictConfig):
            _add(section_name, section, section_name)
            continue

        prefix = _PREFIXED_SECTIONS.get(section_name, "")
        for key, val in section.items():
            # Nested sub-config (e.g. inference.sglang) — flatten with its
            # own prefix so consumers keep seeing ``sglang_tp_size`` etc.
            if isinstance(val, DictConfig) and key in _PREFIXED_SECTIONS:
                sub_prefix = _PREFIXED_SECTIONS[key]
                for sub_key, sub_val in val.items():
                    _add(
                        f"{sub_prefix}{sub_key}",
                        sub_val,
                        f"{section_name}.{key}.{sub_key}",
                    )
            else:
                _add(f"{prefix}{key}", val, f"{section_name}.{key}")

    # --- Computed / alias fields ---
    flat["world_size"] = flat["training_num_nodes"] * flat["training_num_gpus_per_node"]
    flat["rank"] = 0
    if flat.get("inference_engine_type") == "offline":
        # Replay records are already tokenized and carry their loss masks.
        flat["defer_tokenization"] = False
    # Deferred tokenization always needs the training-time matcher, since only
    # the engine knows the final token IDs. Renderers need it too: a renderer
    # emits its mask before the engine expands media placeholders, so any
    # multimodal sample's mask has to be recomputed after expansion.
    flat["dynamic_loss_mask"] = (
        flat.get("inference_engine_type") != "offline"
        and (flat["defer_tokenization"] or bool(flat.get("renderer")))
        and not flat["train_with_decode"]
    )
    flat["use_wandb"] = flat.get("use_wandb", False) or flat.get("report_to") == "wandb"
    flat["use_tensorboard"] = (
        flat.get("use_tensorboard", False) or flat.get("report_to") == "tensorboard"
    )
    flat["checkpoint_dir"] = (
        str(Path(flat["output_dir"]) / "checkpoints") if flat.get("output_dir") else None
    )
    if flat.get("continual_training") and not flat.get("load_path"):
        logger.warning("continual_training=True but no training.load_path was provided")

    if (
        "last_hidden_states_prenorm" not in flat or flat["last_hidden_states_prenorm"] is None
    ) and flat.get("inference_engine_type") != "offline":
        flat["last_hidden_states_prenorm"] = flat.get("inference_engine_type") == "vllm"

    return argparse.Namespace(**flat)


def save_config(config: DictConfig, path: str) -> None:
    OmegaConf.save(config, path)


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

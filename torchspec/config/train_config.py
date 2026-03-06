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

from torchspec.config.inference_config import InferenceConfig
from torchspec.data.utils import is_local_data_path
from torchspec.utils.logging import logger


@dataclass
class DatasetConfig:
    chat_template: str = "llama3"
    defer_tokenization: bool = False
    eval_data_path: Optional[str] = None
    eval_interval: int = 50
    eval_micro_batch_size: Optional[int] = None
    eval_prompt_key: Optional[str] = None
    last_turn_loss_only: bool = False
    prompt_key: str = "conversations"
    train_data_path: str = ""


@dataclass
class DebugConfig:
    debug_inference_only: bool = False
    debug_train_only: bool = False
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
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    target_model_backend: str = "sglang"
    target_model_path: str = ""
    trust_remote_code: bool = False


@dataclass
class TrainingConfig:
    attention_backend: str = "sdpa"
    colocate: bool = False
    distributed_backend: str = "nccl"
    distributed_timeout_minutes: int = 10
    draft_accumulation_steps: int = 1
    fsdp_strategy: str = "REPLICATE"

    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    load_path: Optional[str] = None
    lr_decay_style: str = "cosine"
    lr_total_steps: Optional[int] = None
    max_concurrent_batches: int = 1
    max_grad_norm: float = 0.5
    max_seq_length: int = 8192
    no_save_optim: bool = False
    num_epochs: int = 10
    num_train_steps: Optional[int] = None
    micro_batch_size: int = 2
    prefetch_depth: int = 4
    save_interval: int = 5000
    save_per_epoch: bool = False
    seed: int = 0
    train_backend: str = "fsdp"
    train_env_vars: str = "{}"
    training_num_gpus_per_node: int = 1
    training_num_nodes: int = 1
    ttt_length: int = 7
    warmup_ratio: float = 0.015


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mooncake: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cache_dir: str = "./cache"
    cache_key: Optional[str] = None
    model_download_dir: Optional[str] = None
    output_dir: str = ""


_ALWAYS_LOCAL_PATH_KEYS = ("output_dir", "cache_dir", "model_download_dir")
_DATA_PATH_KEYS = ("dataset.train_data_path", "dataset.eval_data_path")


def _resolve_relative_paths(config: DictConfig, base_dir: str) -> None:
    """Resolve local relative paths in *config* against *base_dir* (in-place).

    Always-local keys (output_dir, cache_dir, …) are absolutized unconditionally.
    Data-path keys are only absolutized when ``is_local_data_path`` says they look
    like filesystem paths (as opposed to HF Hub dataset IDs).
    """
    for dotted_key in (*_ALWAYS_LOCAL_PATH_KEYS, *_DATA_PATH_KEYS):
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
        _resolve_relative_paths(file_config, os.path.dirname(os.path.abspath(config_path)))
        configs_to_merge.append(file_config)

    if cli_args:
        cli_config = OmegaConf.from_dotlist(cli_args)
        configs_to_merge.append(cli_config)

    config = OmegaConf.merge(*configs_to_merge)
    _resolve_relative_paths(config, os.getcwd())
    if save_snapshot:
        _save_config_snapshot(config)

    return config


# Sub-sections whose fields receive a name prefix when flattened.
_PREFIXED_SECTIONS = {
    "mooncake": "mooncake_",
    "sglang": "sglang_",
    "vllm": "vllm_",
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
    flat["dynamic_loss_mask"] = flat["defer_tokenization"]
    flat["use_wandb"] = flat.get("use_wandb", False) or flat.get("report_to") == "wandb"
    flat["use_tensorboard"] = (
        flat.get("use_tensorboard", False) or flat.get("report_to") == "tensorboard"
    )
    flat["checkpoint_dir"] = (
        str(Path(flat["output_dir"]) / "checkpoints") if flat.get("output_dir") else None
    )

    return argparse.Namespace(**flat)


def save_config(config: DictConfig, path: str) -> None:
    OmegaConf.save(config, path)


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

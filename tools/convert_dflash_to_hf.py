"""
Convert DFlash FSDP checkpoint to HuggingFace safetensors format.

Usage:
    python tools/convert_dflash_to_hf.py --input-dir <checkpoint_dir>

    python tools/convert_dflash_to_hf.py --input-dir <checkpoint_dir> \
        --config configs/draft_models/qwen3_8b_dflash.json

Options:
    --input-dir     Path to FSDP checkpoint directory (required)
    --output-dir    Output directory (default: {input_dir}_hf)
    --config        Path to draft model config.json (default: {input_dir}/config.json)
    --dtype         Output dtype (float16, bfloat16, float32)
    -f, --force     Overwrite output directory if exists
"""

import argparse
import json
import logging
import os
import pickle
from typing import Optional

import torch
import torch.distributed.checkpoint as dist_cp
from safetensors.torch import save_file
from typing_extensions import override

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "version.txt")


def _get_torchspec_version() -> str:
    try:
        with open(_VERSION_FILE) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


class _UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class _WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = _UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class _EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _detect_model_dir(input_dir: str) -> str:
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=_WrappedStorageReader(input_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _extract_draft_model_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract DFlash draft model weights from FSDP checkpoint."""
    model_state = {}
    skipped_keys = []

    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "draft_model." not in k:
            skipped_keys.append(k)
            continue
        new_key = k.split("draft_model.")[-1]
        model_state[new_key] = v

    logger.info(
        "Extracted %d model weight keys (skipped %d non-draft keys)",
        len(model_state),
        len(skipped_keys),
    )
    return model_state


def convert(
    input_dir: str,
    output_dir: str,
    config_path: str,
    dtype: Optional[str] = None,
) -> None:
    model_dir = _detect_model_dir(input_dir)
    logger.info("Loading FSDP checkpoint from %s", model_dir)

    state_dict = _load_fsdp_state_dict(model_dir)
    model_weights = _extract_draft_model_weights(state_dict)
    del state_dict

    # Load config
    with open(config_path) as f:
        raw_config = json.load(f)

    # Instantiate model to validate weight shapes
    from torchspec.models.draft.auto import AutoDraftModelConfig
    from torchspec.models.draft.dflash import DFlashDraftModel

    draft_config = AutoDraftModelConfig.from_dict(raw_config)
    model = DFlashDraftModel(draft_config)

    # Load weights into model
    missing, unexpected = model.load_state_dict(model_weights, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    # Cast dtype if requested
    if dtype:
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        model = model.to(dtype=dtype_map[dtype])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tensors = model.state_dict()

    version = _get_torchspec_version()
    save_file(
        tensors,
        os.path.join(output_dir, "model.safetensors"),
        metadata={"torchspec_version": version},
    )

    export_config = json.loads(json.dumps(raw_config))
    export_config["_torchspec_version"] = version
    actual_dtype = next(iter(tensors.values())).dtype
    export_config["torch_dtype"] = str(actual_dtype).replace("torch.", "")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(export_config, f, indent=2)

    logger.info("DFlash model saved to %s (%d tensors)", output_dir, len(tensors))


def main():
    parser = argparse.ArgumentParser(description="Convert DFlash FSDP checkpoint to HF format")
    parser.add_argument("--input-dir", required=True, help="FSDP checkpoint directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: {input}_hf)")
    parser.add_argument("--config", default=None, help="Draft model config.json path")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output if exists")
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.input_dir}_hf"
    if os.path.exists(output_dir) and not args.force:
        logger.error("%s already exists. Use -f to overwrite.", output_dir)
        return

    config_path = args.config
    if config_path is None:
        config_path = os.path.join(args.input_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found in {args.input_dir}. Provide --config.")

    convert(args.input_dir, output_dir, config_path, args.dtype)


if __name__ == "__main__":
    main()

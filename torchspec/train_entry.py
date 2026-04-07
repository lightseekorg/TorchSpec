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

"""Training entry point for Eagle3 speculative decoding."""

import argparse
import os
import sys
import time

# Fix PyTorch 2.9+ TorchInductor GEMM backend regression: without this,
# FlexAttention backward pass hits NoValidChoicesError and training is 3x slower.
# See Phase E in docs/inference/dflash/training_results.md.
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON")
from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Generator

import ray
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from torchspec import AutoDraftModelConfig
from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.config.utils import generate_draft_model_config
from torchspec.controller import (
    AsyncTrainingController,
    auto_calculate_training_steps,
    build_mooncake_config,
    run_training_loop,
    setup_async_training_with_engines,
)
from torchspec.inference.factory import prepare_inference_engines
from torchspec.ray.placement_group import (
    allocate_train_group,
    create_placement_groups,
)
from torchspec.training.trainer_actor import TrainerActor
from torchspec.transfer.mooncake.utils import launch_mooncake_master
from torchspec.utils.env import get_torchspec_env_vars
from torchspec.utils.logging import init_tracking, logger

_Phase = namedtuple("_Phase", ["name", "duration", "is_async", "blocked"])


class _InitTimer:
    """Lightweight segmented timer for initialization phases."""

    def __init__(self) -> None:
        self._t0 = time.time()
        self._phases: list[_Phase] = []
        self._pending: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Time a synchronous phase."""
        start = time.time()
        yield
        self._phases.append(_Phase(name, time.time() - start, is_async=False, blocked=0.0))

    def begin_async(self, name: str) -> None:
        """Mark the start of an async operation (e.g., ray.remote dispatch)."""
        self._pending[name] = time.time()

    def wait(self, name: str, refs) -> Any:
        """Wrap ray.get for async phases. Returns the result."""
        if name not in self._pending:
            raise ValueError(f"No async phase '{name}' was started via begin_async()")
        t_before = time.time()
        result = ray.get(refs)
        t_after = time.time()
        dispatch_time = self._pending.pop(name)
        total = t_after - dispatch_time
        blocked = t_after - t_before
        self._phases.append(_Phase(name, total, is_async=True, blocked=blocked))
        return result

    def log_summary(self) -> None:
        total = time.time() - self._t0
        lines = ["Initialization timing:"]
        for p in self._phases:
            suffix = f"  (blocked {p.blocked:.2f}s)" if p.is_async else ""
            lines.append(f"  {p.name:<48s} {p.duration:>8.2f}s{suffix}")
        lines.append(f"  {'─' * 57}")
        lines.append(f"  {'Total':<48s} {total:>8.2f}s")
        logger.info("\n".join(lines))


def parse_config():
    """Parse YAML config and convert to flat args.

    Supports configs with sections matching the Config dataclass:
    model, dataset, training, debug, inference, logging, mooncake, decode.

    The config is flattened via config_to_flat_args(), with prefixed sections:
    mooncake_*, sglang_*, decode_*.
    """

    parser = argparse.ArgumentParser(description="Eagle3 speculative decoding training")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--print-config-only", action="store_true", help="Print resolved config and exit"
    )

    args, unknown = parser.parse_known_args()

    config = load_config(
        config_path=args.config, cli_args=unknown if unknown else None, save_snapshot=True
    )

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config))

    if args.print_config_only:
        sys.exit(0)

    flat_args = config_to_flat_args(config)

    flat_args.rank = 0
    flat_args.world_size = flat_args.training_num_nodes * flat_args.training_num_gpus_per_node

    defaults = {
        "colocate": False,
        "debug_train_only": False,
        "debug_inference_only": False,
        "dp_size": None,
        "save_debug_train_data": None,
    }
    for key, value in defaults.items():
        if not hasattr(flat_args, key) or getattr(flat_args, key) is None:
            setattr(flat_args, key, value)

    _resolve_batch_size(flat_args)

    return flat_args


def _maybe_create_scratch_draft(args, train_group):
    """Auto-create scratch draft checkpoint for inference engine if not provided."""
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "decode_speculative_algorithm", None)
        and getattr(args, "decode_speculative_draft_model_path", None) is None
    ):
        scratch_dir = os.path.join(getattr(args, "output_dir", "./outputs"), "scratch_draft_model")
        os.makedirs(scratch_dir, exist_ok=True)
        logger.info(f"Auto-creating scratch draft checkpoint at {scratch_dir}")
        train_group.save_draft_model_for_serving(scratch_dir)
        args.decode_speculative_draft_model_path = scratch_dir
        logger.info(f"Set decode_speculative_draft_model_path = {scratch_dir}")


def _resolve_batch_size(args):
    """Derive dp_size, per_dp_rank_batch_size, dispatch_batch_size, and global_batch_size."""
    dp_size = (
        getattr(args, "dp_size", None) or args.training_num_nodes * args.training_num_gpus_per_node
    )
    args.dp_size = dp_size
    sp_size = getattr(args, "sp_size", None)
    if sp_size is not None and sp_size != 1:
        raise NotImplementedError(f"Sequence parallel is not yet supported (got sp_size={sp_size})")
    sp_size = sp_size or 1
    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    args.per_dp_rank_batch_size = args.micro_batch_size * sp_size
    args.global_batch_size = args.per_dp_rank_batch_size * dp_size * accumulation_steps


def _get_draft_model_config(args):
    """Resolve draft model config from args or auto-generate from target model."""

    draft_config_path = getattr(args, "draft_model_config", None)
    if draft_config_path is not None:
        return AutoDraftModelConfig.from_file(draft_config_path)

    config_dict = generate_draft_model_config(
        target_model_path=args.target_model_path,
        cache_dir=getattr(args, "model_download_dir", None),
    )
    return AutoDraftModelConfig.from_dict(config_dict)


def train_async_no_generation(args):
    """Entry point for Eagle3 online training.

    Supports prefill-only mode (default) and decode mode (train_with_decode=True)
    with speculative decoding. Uses distributed Ray actors with placement groups.
    Engines store tensors in mooncake and return keys to AsyncInferenceManager.
    """
    if (
        getattr(args, "train_with_decode", False)
        and getattr(args, "inference_engine_type", "sgl") != "sgl"
    ):
        raise ValueError("train_with_decode=True requires inference_engine_type=sgl")

    init_tracking(args)
    timer = _InitTimer()

    # [1] Create controller early (lightweight: only needs args + dp_size)
    with timer.phase("Create controller"):
        driver_node_id = ray.get_runtime_context().get_node_id()
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": get_torchspec_env_vars()},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, args.dp_size)

    # [2] Kick off dataset loading on controller (async — runs on actor while driver continues)
    timer.begin_async("Dataset loading")
    dataset_size_ref = controller.load_dataset.remote(args)
    eval_dataset_size_ref = controller.load_eval_dataset.remote(args)

    # [3] Do initialization that doesn't depend on dataset in parallel
    with timer.phase("Driver-side init"):
        draft_model_config = _get_draft_model_config(args)
        args.draft_model_config_obj = draft_model_config

        # Auto-set aux layer IDs for DFlash (5 layers) if not explicitly provided
        from torchspec.models.draft.dflash import DFlashConfig

        if isinstance(draft_model_config, DFlashConfig) and not getattr(
            args, "aux_hidden_states_layers", None
        ):
            from torchspec.models.draft.dflash import build_target_layer_ids

            target_layer_ids = getattr(draft_model_config, "target_layer_ids", None)
            if target_layer_ids is None:
                num_target = getattr(draft_model_config, "num_target_layers", 5)
                target_num_hidden = getattr(draft_model_config, "target_num_hidden_layers", 36)
                target_layer_ids = build_target_layer_ids(num_target, target_num_hidden)
            args.aux_hidden_states_layers = target_layer_ids
            logger.info(f"DFlash: set aux_hidden_states_layers = {target_layer_ids}")

        pgs = create_placement_groups(args)
        launch_mooncake_master(args)
        mooncake_config = build_mooncake_config(args)

    # [4] Wait for dataset sizes (small ints, unlike the old ray.put of the full dataset)
    dataset_size, eval_dataset_size = timer.wait(
        "Dataset loading", [dataset_size_ref, eval_dataset_size_ref]
    )
    logger.info(f"Dataset loaded on controller: {dataset_size} train, {eval_dataset_size} eval")

    # [5] Auto-calculate training steps (needs dataset_size)
    with timer.phase("Auto-calculate training steps"):
        auto_calculate_training_steps(args, dataset_size)

    # [6] Generate vocab mapping on controller if vocab pruning is enabled
    vocab_mapping = None
    draft_vocab_size = getattr(draft_model_config, "draft_vocab_size", None)
    vocab_size = draft_model_config.vocab_size
    if draft_vocab_size is not None and draft_vocab_size != vocab_size:
        with timer.phase("Vocab mapping"):
            logger.info(
                f"Computing vocab mapping on controller "
                f"(target={vocab_size}, draft={draft_vocab_size})..."
            )
            vocab_mapping = ray.get(
                controller.compute_vocab_mapping.remote(vocab_size, draft_vocab_size)
            )
            logger.info(
                f"Generated vocab mapping: "
                f"d2t={vocab_mapping[0].shape}, t2d={vocab_mapping[1].shape}"
            )

    # [7] Create training actors + inference engines (args now has num_train_steps)
    timer.begin_async("Actor initialization")
    with timer.phase("Allocate actors + dispatch init"):
        train_group = allocate_train_group(
            args=args,
            num_nodes=args.training_num_nodes,
            num_gpus_per_node=args.training_num_gpus_per_node,
            pg=pgs["training"],
            training_class=TrainerActor,
        )
        train_init_refs = train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )

        # Decode mode: create scratch draft checkpoint before inference engines
        # are prepared, since they need decode_speculative_draft_model_path on args.
        # This blocks on train actor init (FSDP gather), so inference engines are
        # dispatched after to maximize parallelism with the wait below.
        _maybe_create_scratch_draft(args, train_group)

        inference_engines, engine_init_refs = prepare_inference_engines(
            args, pgs["inference"], mooncake_config
        )

    # [8] Wait for all actor init to complete concurrently
    n_train = len(train_init_refs)
    logger.info(
        f"Waiting for {n_train} training actors and {len(engine_init_refs)} "
        f"inference engines to initialize in parallel..."
    )
    all_results = timer.wait("Actor initialization", train_init_refs + engine_init_refs)

    train_results = all_results[:n_train]
    assert len(set(train_results)) == 1
    logger.info(
        f"All {n_train} training actors and {len(engine_init_refs)} inference engines initialized"
    )

    if vocab_mapping is not None:
        train_group.set_vocab_buffers(*vocab_mapping)
        logger.info("Loaded vocab mapping into training actors")

    # [9] Setup async training with pre-created controller
    with timer.phase("Setup async training"):
        controller, inference_manager = setup_async_training_with_engines(
            args, train_group, mooncake_config, inference_engines, controller=controller
        )

    timer.log_summary()

    # [10] Run training loop (no ray.put needed — dataset lives on controller)
    run_training_loop(
        args,
        controller,
        inference_manager,
        train_group,
        inference_engines=inference_engines,
        dataset_size=dataset_size,
        eval_dataset_size=eval_dataset_size,
    )


if __name__ == "__main__":
    args = parse_config()
    train_async_no_generation(args)

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
from torchspec.colocate import is_mps_colocate, validate_colocate_config
from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.config.utils import generate_draft_model_config
from torchspec.controller import (
    AsyncTrainingController,
    auto_calculate_training_steps,
    build_mooncake_config,
    run_training_loop,
    setup_async_training_with_engines,
    setup_colocate_training_with_engines,
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
    _validate_usp_args(flat_args)
    validate_colocate_config(flat_args)

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
    world_size = args.training_num_nodes * args.training_num_gpus_per_node
    if getattr(args, "attention_backend", None) == "usp":
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
        if sp_size <= 0:
            raise ValueError(f"USP requires positive sp_size, got {sp_size}")
        if world_size % sp_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by USP sp_size ({sp_size})"
            )
        dp_size = getattr(args, "dp_size", None) or (world_size // sp_size)
        if dp_size * sp_size != world_size:
            raise ValueError(
                f"dp_size ({dp_size}) * sp_size ({sp_size}) must equal world_size ({world_size})"
            )
        args.dp_size = dp_size
        args.sp_size = sp_size
        args.per_dp_rank_batch_size = 1
    else:
        dp_size = getattr(args, "dp_size", None) or world_size
        args.dp_size = dp_size
        sp_size = getattr(args, "sp_size", None)
        if sp_size is not None and sp_size != 1:
            raise NotImplementedError(
                f"Sequence parallel is not yet supported (got sp_size={sp_size})"
            )
        sp_size = sp_size or 1
        args.per_dp_rank_batch_size = args.micro_batch_size * sp_size

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    args.global_batch_size = args.per_dp_rank_batch_size * dp_size * accumulation_steps


def _validate_usp_args(args) -> None:
    if getattr(args, "attention_backend", None) != "usp":
        return

    sp_size = getattr(args, "sp_size", None)
    if sp_size is None:
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
    if sp_size <= 1:
        raise NotImplementedError(f"USP requires sp_size > 1, got {sp_size}")

    inference_engine_type = getattr(args, "inference_engine_type", "sgl")
    if inference_engine_type != "sgl":
        raise ValueError(
            f"USP currently only supports inference_engine_type=sgl, got {inference_engine_type}"
        )

    fsdp_strategy = getattr(args, "fsdp_strategy", "REPLICATE").upper()
    if fsdp_strategy != "REPLICATE":
        raise NotImplementedError(
            f"USP currently only supports fsdp_strategy=REPLICATE, got {fsdp_strategy}"
        )

    micro_batch_size = getattr(args, "micro_batch_size", 1)
    if micro_batch_size != 1:
        raise NotImplementedError(
            f"USP currently only supports micro_batch_size=1, got {micro_batch_size}"
        )


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


def _validate_and_configure_dflash(args, draft_model_config) -> None:
    """Validate DFlash-specific config and auto-set aux layer IDs.

    Called before dataset loading to fail fast on misconfigurations.
    """
    from torchspec.models.draft.dflash import DFlashConfig

    if not isinstance(draft_model_config, DFlashConfig):
        return

    if getattr(args, "inference_engine_type", "hf") != "sgl":
        raise NotImplementedError("DFlash currently supports only inference_engine_type='sgl'.")
    if getattr(args, "defer_tokenization", False):
        raise NotImplementedError("DFlash does not support defer_tokenization=True.")
    block_size = getattr(args, "dflash_block_size", 16)
    min_loss = getattr(args, "min_loss_tokens", 0)
    if min_loss < 2 * block_size:
        raise ValueError(
            f"DFlash requires dataset.min_loss_tokens >= 2 * training.dflash_block_size "
            f"({min_loss} < {2 * block_size}). Set dataset.min_loss_tokens={2 * block_size}."
        )

    # Auto-set aux layer IDs from draft config if not explicitly provided
    if not getattr(args, "aux_hidden_states_layers", None):
        from torchspec.models.draft.dflash import build_target_layer_ids

        target_layer_ids = getattr(draft_model_config, "target_layer_ids", None)
        if target_layer_ids is None:
            num_target = getattr(draft_model_config, "num_target_layers", 5)
            target_num_hidden = getattr(draft_model_config, "target_num_hidden_layers", 36)
            target_layer_ids = build_target_layer_ids(num_target, target_num_hidden)
        args.aux_hidden_states_layers = target_layer_ids
        logger.info(f"DFlash: set aux_hidden_states_layers = {target_layer_ids}")


def _maybe_resolve_colocate_aux_layers(args) -> None:
    """Auto-resolve aux_hidden_states_layers for Eagle3 colocate runs.

    The colocate training loop sizes the NCCL hidden-states transfer
    buffer up front, so it needs aux_hidden_states_layers on `args`
    before the loop starts — unlike the disagg path there's no engine
    round-trip to discover it. DFlash configs are already handled by
    _validate_and_configure_dflash; this covers Eagle3, using the same
    default the engine falls back to (sgl_engine resolves the identical
    function when args.aux_hidden_states_layers is None) so both sides
    agree on the tensor's last-dim.
    """
    if not is_mps_colocate(args):
        return
    if getattr(args, "aux_hidden_states_layers", None):
        return
    from torchspec.utils.misc import get_default_eagle3_aux_layer_ids

    args.aux_hidden_states_layers = get_default_eagle3_aux_layer_ids(args.target_model_path)
    logger.info(f"Colocate: auto-set aux_hidden_states_layers = {args.aux_hidden_states_layers}")


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

    # [0] Pre-Ray MPS bring-up (Phase 1): once the MPS control daemon is
    # running on a node, the *node* enters MPS client mode — every CUDA
    # context on that node has to register with MPS by setting
    # CUDA_MPS_PIPE_DIRECTORY (otherwise CUDA calls fail with
    # error 805, "MPS client failed to connect"). Ray spawns its
    # gcs/worker processes inheriting `os.environ`; if we start MPS
    # *after* Ray is up, those workers come up with no MPS env and
    # any later `torch.cuda.*` call in any actor blows up. Start
    # the daemon first AND export the client env into our own
    # process so every actor (including ones whose runtime_env we
    # don't directly own, e.g. AsyncTrainingController) inherits it.
    if is_mps_colocate(args):
        from torchspec.colocate.mps import setup_for_colocate as _early_setup_mps

        _mps_handle, _mps_env = _early_setup_mps()
        if _mps_handle is None:
            # MPS is unavailable in this environment (e.g. Modal sandbox
            # without --ipc=host). Continue with fractional GPU sharing
            # but no MPS — see setup_for_colocate docstring for the
            # tradeoff. Mark the args so downstream code knows not to
            # inject CUDA_MPS_PIPE_DIRECTORY into actor runtime_envs.
            args.colocate_mps_unavailable = True
            logger.warning(
                "MPS unavailable on this host; running colocate without "
                "kernel concurrency (fractional GPU sharing only)."
            )
        else:
            args.colocate_mps_unavailable = False
            os.environ.update(_mps_env)
            logger.info(
                "MPS daemon ready (pre-Ray start, started_by_us=%s, pipe_dir=%s)",
                _mps_handle.started_by_us,
                _mps_handle.pipe_dir,
            )

    # [1] Create controller early (lightweight: only needs args + dp_size)
    with timer.phase("Create controller"):
        driver_node_id = ray.get_runtime_context().get_node_id()
        controller_env = get_torchspec_env_vars()
        # Ray inherits os.environ for in-cluster workers, but the
        # controller's runtime_env override is layered separately —
        # explicitly include MPS pipe so the controller process
        # joins the same MPS client world as the trainer/engine
        # actors created later. Without this, the first
        # `torch.cuda.is_available()` inside the controller (e.g.
        # via tokenizer/dataset code that does `torch.cuda.*`)
        # crashes the whole run.
        if is_mps_colocate(args) and not getattr(args, "colocate_mps_unavailable", False):
            from torchspec.colocate.mps import mps_client_env as _mps_env_fn

            controller_env.update(_mps_env_fn())
        controller = AsyncTrainingController.options(
            runtime_env={"env_vars": controller_env},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
        ).remote(args, args.dp_size)

    # [1.5] Parse draft config + DFlash validation (before any async work)
    with timer.phase("Parse draft model config"):
        draft_model_config = _get_draft_model_config(args)
        args.draft_model_config_obj = draft_model_config

        _validate_and_configure_dflash(args, draft_model_config)
        _maybe_resolve_colocate_aux_layers(args)

    # [2] Kick off dataset loading on controller (async — runs on actor while driver continues)
    timer.begin_async("Dataset loading")
    dataset_size_ref = controller.load_dataset.remote(args)
    eval_dataset_size_ref = controller.load_eval_dataset.remote(args)

    # [3] Do initialization that doesn't depend on dataset in parallel
    with timer.phase("Driver-side init"):
        # NOTE: under colocate the MPS daemon was already started
        # in step [0] above so the controller (started in step [1])
        # could come up with the matching CUDA_MPS_PIPE_DIRECTORY.
        # `setup_for_colocate` is idempotent so callers expecting a
        # handle here still get one, but we intentionally don't
        # re-start the daemon.
        #
        # Multi-node colocate: step [0]'s pre-Ray MPS bring-up only
        # covered the driver's own node. Bootstrap the daemon on every
        # other node before the trainer/engine actors are placed there.
        # No-op for single-node (the driver node's daemon is already up)
        # so the validated single-node path is untouched.
        if (
            is_mps_colocate(args)
            and not getattr(args, "colocate_mps_unavailable", False)
            and int(getattr(args, "training_num_nodes", 1) or 1) > 1
        ):
            from torchspec.colocate.mps import ensure_mps_on_all_nodes

            ensure_mps_on_all_nodes()
        pgs = create_placement_groups(args)
        # Phase 5: in colocate (NCCL transfer) mode the entire Mooncake
        # plumbing is unused. Skip both the master daemon and the
        # config build. Downstream code (Trainer / SglEngine) treats
        # `mooncake_config=None` as "not on the Mooncake path".
        if is_mps_colocate(args):
            mooncake_config = None
        else:
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

        # Phase 4/5: Driver-computed colocate union-world rendezvous params.
        # The trainer rank-0 already self-discovered its master_addr/port
        # via setup_master in its constructor — we read them off the
        # train_group, derive the union-world endpoint (port + 5000), and
        # inject the env contract into BOTH the driver process (so trainer
        # actors created below see it via Ray's child env propagation) and
        # the engine actors' runtime_env (so they see it before they
        # spawn the sglang TP scheduler subprocess).
        engine_extra_env: dict[str, str] = {}
        if is_mps_colocate(args):
            n_per_role = args.training_num_nodes * args.training_num_gpus_per_node
            union_master_addr = train_group.master_addr
            union_master_port = int(train_group.master_port) + 5000
            union_timeout_min = int(getattr(args, "distributed_timeout_minutes", 30))
            union_env = {
                "TORCHSPEC_COLOCATE_TRANSFER_MODE": "nccl",
                "TORCHSPEC_COLOCATE_UNION_MASTER_ADDR": str(union_master_addr),
                "TORCHSPEC_COLOCATE_UNION_MASTER_PORT": str(union_master_port),
                "TORCHSPEC_COLOCATE_UNION_WORLD_SIZE": str(2 * n_per_role),
                "TORCHSPEC_COLOCATE_UNION_N_PER_ROLE": str(n_per_role),
                "TORCHSPEC_COLOCATE_UNION_TIMEOUT_MIN": str(union_timeout_min),
                # engine_tp_size for the colocate rank math. Currently
                # always 1 (validator invariant D); part of the contract
                # so the multi-TP data-plane work doesn't have to touch
                # the env wiring later.
                "TORCHSPEC_COLOCATE_ENGINE_TP_SIZE": str(
                    int(getattr(args, "inference_num_gpus_per_engine", 1) or 1)
                ),
            }
            # Re-publish any explicit CUDA IPC override through the same
            # env contract so the trainer-side fetcher and the engine-side
            # connector make an identical transport decision (a one-sided
            # choice would desync the wire protocol). CUDA IPC is the
            # default transport; when the var is unset both sides default
            # to it independently, so only an explicit value needs to be
            # forwarded (typically TORCHSPEC_COLOCATE_IPC=0 to force gloo).
            _ipc_opt = os.environ.get("TORCHSPEC_COLOCATE_IPC")
            if _ipc_opt is not None:
                union_env["TORCHSPEC_COLOCATE_IPC"] = _ipc_opt
            for k, v in union_env.items():
                os.environ[k] = v
            engine_extra_env = union_env
            logger.info(
                "[colocate] Driver-computed union rendezvous: %s:%d "
                "(world_size=2*%d=%d, timeout=%dmin). Injecting into engine "
                "runtime_env so the patched sglang sees it before init.",
                union_master_addr,
                union_master_port,
                n_per_role,
                2 * n_per_role,
                union_timeout_min,
            )

        train_init_refs = train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )

        # Decode mode: create scratch draft checkpoint before inference engines
        # are prepared, since they need decode_speculative_draft_model_path on args.
        # This blocks on train actor init (FSDP gather), so inference engines are
        # dispatched after to maximize parallelism with the wait below.
        _maybe_create_scratch_draft(args, train_group)

        # NOTE: the previous "init-order fence" that awaited trainer init
        # before kicking off engines is incompatible with the colocate
        # union-world rendezvous, which is COLLECTIVE across all 2N ranks.
        # If we waited on trainer init here, every trainer's
        # init_process_group(world_size=2N) would block forever waiting
        # for engines that hadn't been spawned. Instead we let trainer
        # init and engine init run in parallel; both block on the
        # rendezvous, both unblock together. Memory contention under
        # MPS is handled by `expandable_segments:True` + the
        # train_frac/infer_frac budget split (no double-allocation
        # because both sides start tiny and grow into their share).

        inference_engines, engine_init_refs = prepare_inference_engines(
            args,
            pgs["inference"],
            mooncake_config,
            extra_env_vars=engine_extra_env if is_mps_colocate(args) else None,
        )

    # [8] Wait for all actor init to complete concurrently. Under
    # colocate mode this is also where the union-world rendezvous
    # collectively unblocks — every trainer + engine rank is sitting
    # inside dist.init_process_group(world_size=2N) until ALL of them
    # call it. Awaiting both sets of refs together is what allows
    # progress.
    n_train = len(train_init_refs)
    logger.info(
        f"Waiting for {n_train} training actors and {len(engine_init_refs)} "
        f"inference engines to initialize in parallel..."
    )
    all_results = timer.wait("Actor initialization", train_init_refs + engine_init_refs)

    if n_train > 0:
        train_results = all_results[:n_train]
        assert len(set(train_results)) == 1
    logger.info(
        f"All {n_train} training actors and {len(engine_init_refs)} inference engines initialized"
    )

    if vocab_mapping is not None:
        train_group.set_vocab_buffers(*vocab_mapping)
        logger.info("Loaded vocab mapping into training actors")

    # [9] Setup training with pre-created controller. Colocate (NCCL)
    # mode skips the AsyncInferenceManager entirely — see
    # setup_colocate_training_with_engines for what's left out.
    with timer.phase("Setup training"):
        if is_mps_colocate(args):
            controller, inference_manager = setup_colocate_training_with_engines(
                args, train_group, inference_engines, controller=controller
            )
        else:
            controller, inference_manager = setup_async_training_with_engines(
                args, train_group, mooncake_config, inference_engines, controller=controller
            )

    timer.log_summary()

    if is_mps_colocate(args):
        from torchspec.controller.colocate_loop import run_colocate_training_loop

        run_colocate_training_loop(
            args,
            controller,
            train_group,
            inference_engines=inference_engines,
            dataset_size=dataset_size,
            eval_dataset_size=eval_dataset_size,
        )
        return

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

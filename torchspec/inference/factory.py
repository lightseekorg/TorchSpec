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

"""Inference engine creation and initialization with Ray placement groups."""

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from torchspec.utils.env import get_torchspec_env_vars
from torchspec.utils.logging import logger

# Multi-node TP worker engines must stay alive to participate in NCCL
# operations but are never called directly. Store refs here to prevent GC.
_alive_worker_engines: list = []


def create_inference_engines(args, inference_pg, mooncake_config, engine_group: int = 0):
    """Create inference engines based on configured engine type (blocking).

    Supports "hf", "sgl", and "vllm" engine types via inference_engine_type config.

    Returns:
        List of head engines used for dispatching requests. Multi-node TP
        worker engines (if any) are kept alive internally but not returned.
    """
    engine_type = getattr(args, "inference_engine_type", "hf")

    if engine_type not in ("hf", "sgl", "vllm"):
        raise ValueError(f"Unknown inference_engine_type: {engine_type}")

    logger.info(f"Using {engine_type} engine for inference")

    engines = init_engines(
        args,
        inference_pg,
        engine_type=engine_type,
        mooncake_config=mooncake_config,
        engine_group=engine_group,
    )

    logger.info(
        f"Created {len(engines)} distributed {engine_type} engines from {args.target_model_path}"
    )
    return engines


def prepare_inference_engines(args, inference_pg, mooncake_config, engine_group: int = 0):
    """Create inference engines and fire init calls without waiting.

    Use this to parallelize engine initialization with other setup work
    (e.g., training actor initialization). Call ray.get() on the returned
    init_refs before using the engines.

    Returns:
        Tuple of (head_engines, init_refs) where head_engines are the engines
        for dispatching requests, and init_refs are ObjectRefs to wait on.
    """
    engine_type = getattr(args, "inference_engine_type", "hf")

    if engine_type not in ("hf", "sgl", "vllm"):
        raise ValueError(f"Unknown inference_engine_type: {engine_type}")

    logger.info(f"Preparing {engine_type} inference engines...")

    if engine_type == "hf":
        engines, init_refs = _prepare_hf_engines(args, inference_pg, mooncake_config, engine_group)
    elif engine_type == "sgl":
        engines, init_refs = _prepare_sgl_engines(args, inference_pg, mooncake_config, engine_group)
    else:
        engines, init_refs = _prepare_vllm_engines(
            args, inference_pg, mooncake_config, engine_group
        )

    return engines, init_refs


def init_engines(args, pg, engine_type: str, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize inference engines with Ray placement groups.

    Args:
        args: Configuration arguments.
        pg: Placement group tuple (pg, reordered_bundle_indices, reordered_gpu_ids).
        engine_type: Engine type ("hf", "sgl", or "vllm").
        mooncake_config: MooncakeConfig object.

    Returns:
        List of head engines for dispatching requests.
    """
    if engine_type == "hf":
        return _init_hf_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "sgl":
        return _init_sgl_engines(args, pg, mooncake_config, engine_group)
    elif engine_type == "vllm":
        return _init_vllm_engines(args, pg, mooncake_config, engine_group)
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}")


def _prepare_hf_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> tuple[list, list]:
    """Create HF engine actors and fire init calls without waiting.

    Returns:
        Tuple of (engines, init_handles).
    """
    num_gpus_total = getattr(args, "inference_num_gpus", 1)
    num_gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
    num_gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)

    num_gpus_per_engine = min(num_gpus_per_engine, num_gpus_per_node)
    num_engines = num_gpus_total // num_gpus_per_engine

    logger.info(f"Initializing {num_engines} HF engines ({num_gpus_per_engine} GPU(s) each)")

    from torchspec.inference.engine.hf_engine import HFEngine

    HFRayActor = ray.remote(HFEngine)
    return _create_and_init_actors(
        args,
        pg,
        num_engines,
        num_gpus_per_engine,
        HFRayActor,
        mooncake_config,
        engine_group=engine_group,
    )


def _init_hf_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize HF engines with Ray placement groups."""
    engines, init_handles = _prepare_hf_engines(args, pg, mooncake_config, engine_group)
    _wait_for_init(init_handles, "HF", timeout=1200)
    return engines


def _prepare_sgl_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create SGL engine actors and fire init calls without waiting.

    Handles three cases:
      - Single-node, multiple engines: one engine per group of GPUs
      - Multi-node, single replica: one engine per node, all forming one TP group
      - Multi-node, multiple replicas: N independent TP groups, each spanning nnodes

    For multi-node, worker engines are stored in a module-level list to prevent GC.

    Returns:
        Tuple of (head_engines, init_handles). head_engines are the engines that
        accept generate() calls. init_handles are ObjectRefs for ALL engines
        (head + worker) that must be waited on before use.
    """

    nnodes = getattr(args, "sglang_nnodes", 1)
    num_gpus_total = getattr(args, "inference_num_gpus", 1)

    if nnodes > 1:
        gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)
        gpus_per_replica = nnodes * gpus_per_node
        num_replicas = num_gpus_total // gpus_per_replica
        num_engines = num_replicas * nnodes
        gpus_per_engine = gpus_per_node
    else:
        gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
        num_replicas = num_gpus_total // gpus_per_engine
        num_engines = num_replicas

    logger.info(
        f"Initializing {num_engines} Sgl engines "
        f"({gpus_per_engine} GPU(s) each, nnodes={nnodes}, replicas={num_replicas})"
    )

    from torchspec.inference.engine.sgl_engine import SglEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    SglRayActor = ray.remote(SglEngine)
    env_vars = get_torchspec_env_vars()

    # Step 1: Create all engine actors (without calling init yet)
    engines = []
    for i in range(num_engines):
        node_rank = i % nnodes if nnodes > 1 else 0

        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        engine = SglRayActor.options(
            num_cpus=0.2,
            num_gpus=0.2,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=node_rank,
            engine_group=engine_group,
        )
        engines.append(engine)

    # Step 2: Auto-negotiate dist_init_addr for each multi-node replica
    # Uses RayActor.get_node_ip() and .find_free_port() on the node_rank=0 engine
    dist_init_addrs: dict[int, str] = {}
    if nnodes > 1:
        configured_addr = getattr(args, "sglang_dist_init_addr", None)
        for replica_idx in range(num_replicas):
            if configured_addr and num_replicas == 1:
                dist_init_addrs[replica_idx] = configured_addr
                logger.info(
                    f"Replica {replica_idx}: using configured dist_init_addr: {configured_addr}"
                )
            else:
                head_engine = engines[replica_idx * nnodes]
                ip, port = ray.get(
                    [head_engine.get_node_ip.remote(), head_engine.find_free_port.remote()],
                    timeout=120,
                )
                addr = f"{ip}:{port}"
                dist_init_addrs[replica_idx] = addr
                logger.info(f"Replica {replica_idx}: auto-negotiated dist_init_addr: {addr}")

    # Step 2.5: Pre-allocate ports to avoid TOCTOU races between parallel engines.
    # Each engine needs 2 consecutive ports; allocate sequentially so engines
    # on the same node never collide.
    pre_allocated_ports: dict[int, int] = {}
    next_start = 10000
    for i in range(num_engines):
        port = ray.get(
            engines[i].find_free_port.remote(start_port=next_start, consecutive=2),
            timeout=120,
        )
        pre_allocated_ports[i] = port
        next_start = port + 2
        logger.info(f"Engine {i}: pre-allocated ports {port}, {port + 1}")

    # Step 3: Fire init() on all engines (non-blocking)
    init_handles = []
    for i, engine in enumerate(engines):
        replica_idx = i // nnodes if nnodes > 1 else i
        init_handles.append(
            engine.init.remote(
                mooncake_config=mooncake_config,
                dist_init_addr=dist_init_addrs.get(replica_idx),
                pre_allocated_port=pre_allocated_ports.get(i),
            )
        )

    # Separate head/worker engines for multi-node
    if nnodes > 1:
        head_engines = [engines[i] for i in range(num_engines) if i % nnodes == 0]
        worker_engines = [engines[i] for i in range(num_engines) if i % nnodes != 0]
        # Keep worker engines alive (they participate in NCCL TP) via module-level ref
        _alive_worker_engines.extend(worker_engines)
        logger.info(
            f"Prepared multi-node Sgl engines: {len(head_engines)} heads + "
            f"{len(worker_engines)} workers ({num_replicas} replicas)"
        )
        return head_engines, init_handles

    return engines, init_handles


def _init_sgl_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize SGLang engines with Ray placement groups (blocking)."""
    head_engines, init_handles = _prepare_sgl_engines(args, pg, mooncake_config, engine_group)
    nnodes = getattr(args, "sglang_nnodes", 1)
    init_timeout = getattr(args, "sglang_init_timeout", 300 if nnodes == 1 else 600)
    _wait_for_init(init_handles, "Sgl", timeout=init_timeout)
    return head_engines


def _prepare_vllm_engines(
    args, pg, mooncake_config=None, engine_group: int = 0
) -> tuple[list, list]:
    """Create vLLM engine actors and fire init calls without waiting.

    Handles three cases:
      - Single-node, multiple engines: one engine per group of GPUs
      - Multi-node, single replica: one engine per node, all forming one TP group
      - Multi-node, multiple replicas: N independent TP groups, each spanning nnodes

    For multi-node, worker engines are stored in a module-level list to prevent GC.

    Returns:
        Tuple of (head_engines, init_handles). head_engines are the engines that
        accept generate() calls. init_handles are ObjectRefs for ALL engines
        (head + worker) that must be waited on before use.
    """
    nnodes = getattr(args, "vllm_nnodes", 1)
    num_gpus_total = getattr(args, "inference_num_gpus", 1)

    if nnodes > 1:
        gpus_per_node = getattr(args, "inference_num_gpus_per_node", 8)
        gpus_per_replica = nnodes * gpus_per_node
        num_replicas = num_gpus_total // gpus_per_replica
        num_engines = num_replicas * nnodes
        gpus_per_engine = gpus_per_node
    else:
        gpus_per_engine = getattr(args, "inference_num_gpus_per_engine", 1)
        num_replicas = num_gpus_total // gpus_per_engine
        num_engines = num_replicas

    logger.info(
        f"Initializing {num_engines} vLLM engines "
        f"({gpus_per_engine} GPU(s) each, nnodes={nnodes}, replicas={num_replicas})"
    )

    from torchspec.inference.engine.vllm_engine import VllmEngine

    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    VllmRayActor = ray.remote(VllmEngine)
    env_vars = get_torchspec_env_vars()

    engines = []
    for i in range(num_engines):
        node_rank = i % nnodes if nnodes > 1 else 0

        bundle_offset = i * gpus_per_engine
        base_gpu_id = int(reordered_gpu_ids[bundle_offset])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[bundle_offset],
        )

        engine = VllmRayActor.options(
            num_cpus=0.2,
            num_gpus=0.2,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(
            args=args,
            rank=i,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=gpus_per_engine,
            node_rank=node_rank,
            engine_group=engine_group,
        )
        engines.append(engine)

    # Pre-allocate ports to avoid TOCTOU races between parallel engines.
    # Each engine needs 2 consecutive ports (dist_init + nccl).
    pre_allocated_ports: dict[int, int] = {}
    next_start = 10000
    for i in range(num_engines):
        port = ray.get(
            engines[i].find_free_port.remote(start_port=next_start, consecutive=2),
            timeout=30,
        )
        pre_allocated_ports[i] = port
        next_start = port + 2
        logger.info(f"vLLM Engine {i}: pre-allocated ports {port}, {port + 1}")

    dist_init_addrs: dict[int, str] = {}
    if nnodes > 1:
        configured_addr = getattr(args, "vllm_dist_init_addr", None)
        for replica_idx in range(num_replicas):
            if configured_addr and num_replicas == 1:
                dist_init_addrs[replica_idx] = configured_addr
                logger.info(
                    f"Replica {replica_idx}: using configured dist_init_addr: {configured_addr}"
                )
            else:
                head_idx = replica_idx * nnodes
                head_engine = engines[head_idx]
                ip = ray.get(head_engine.get_node_ip.remote(), timeout=30)
                addr = f"{ip}:{pre_allocated_ports[head_idx]}"
                dist_init_addrs[replica_idx] = addr
                logger.info(
                    f"Replica {replica_idx}: dist_init_addr from pre-allocated port: {addr}"
                )

    init_handles = []
    for i, engine in enumerate(engines):
        replica_idx = i // nnodes if nnodes > 1 else i
        init_handles.append(
            engine.init.remote(
                mooncake_config=mooncake_config,
                dist_init_addr=dist_init_addrs.get(replica_idx),
                pre_allocated_port=pre_allocated_ports.get(i),
            )
        )

    if nnodes > 1:
        head_engines = [engines[i] for i in range(num_engines) if i % nnodes == 0]
        worker_engines = [engines[i] for i in range(num_engines) if i % nnodes != 0]
        _alive_worker_engines.extend(worker_engines)
        logger.info(
            f"Prepared multi-node vLLM engines: {len(head_engines)} heads + "
            f"{len(worker_engines)} workers ({num_replicas} replicas)"
        )
        return head_engines, init_handles

    return engines, init_handles


def _init_vllm_engines(args, pg, mooncake_config=None, engine_group: int = 0) -> list:
    """Initialize vLLM engines with Ray placement groups (blocking)."""
    head_engines, init_handles = _prepare_vllm_engines(args, pg, mooncake_config, engine_group)
    nnodes = getattr(args, "vllm_nnodes", 1)
    init_timeout = getattr(args, "vllm_init_timeout", 300 if nnodes == 1 else 600)
    _wait_for_init(init_handles, "Vllm", timeout=init_timeout)
    return head_engines


def _create_and_init_actors(
    args,
    pg,
    num_engines,
    num_gpus_per_engine,
    ray_actor_cls,
    mooncake_config,
    extra_kwargs=None,
    engine_group: int = 0,
) -> tuple[list, list]:
    """Create Ray actors and start their init calls.

    Args:
        args: Configuration arguments.
        pg: Placement group tuple.
        num_engines: Number of engines to create.
        num_gpus_per_engine: GPUs per engine.
        ray_actor_cls: The ray.remote-wrapped actor class.
        mooncake_config: Mooncake configuration dict.
        extra_kwargs: Extra kwargs to pass to actor constructor (e.g. num_gpus_per_engine for Sgl).
        engine_group: Group index for log file disambiguation.

    Returns:
        Tuple of (engines list, init_handles list).
    """
    pg_obj, reordered_bundle_indices, reordered_gpu_ids = pg

    engines = []
    init_handles = []

    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        base_gpu_id = int(reordered_gpu_ids[i * num_gpus_per_engine])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpus_per_engine],
        )

        env_vars = get_torchspec_env_vars()

        constructor_kwargs = {
            "args": args,
            "rank": i,
            "base_gpu_id": base_gpu_id,
            "engine_group": engine_group,
        }
        if extra_kwargs:
            constructor_kwargs.update(extra_kwargs)

        engine = ray_actor_cls.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={"env_vars": env_vars},
        ).remote(**constructor_kwargs)

        engines.append(engine)
        init_handles.append(engine.init.remote(mooncake_config=mooncake_config))

    return engines, init_handles


def _wait_for_init(init_handles: list, engine_label: str, timeout: int = 300):
    """Wait for engine init calls to complete with error handling.

    Args:
        init_handles: List of init() call ObjectRefs.
        engine_label: Label for log messages (e.g. "HF", "Sgl").
        timeout: Timeout in seconds.
    """
    try:
        logger.info(
            f"Waiting for {len(init_handles)} {engine_label} engines to initialize "
            f"(timeout={timeout}s)..."
        )
        ray.get(init_handles, timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        logger.error(f"Timeout waiting for {engine_label} engines to initialize after {timeout}s")
        ready, not_ready = ray.wait(init_handles, num_returns=len(init_handles), timeout=0)
        logger.error(f"Initialized: {len(ready)}/{len(init_handles)}, Failed: {len(not_ready)}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize {engine_label} engines: {e}")
        raise

    logger.info(f"Successfully initialized {len(init_handles)} {engine_label} engines")

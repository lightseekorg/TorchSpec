# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 1 — Placement & MPS smoke test.

This test runs **on Modal** via
``modal run scripts/modal/modal_colocate_smoke.py::phase1_placement``. It
requires:

- A real Ray cluster (the in-actor head will be auto-started).
- 4 GPUs on a single node with NVIDIA MPS available
  (``nvidia-cuda-mps-control`` in PATH).

The test deliberately does **not** load a model. It only verifies the
placement / lifecycle invariants from
``docs/colocate/implementation.md`` §Phase 1:

1. Spawn placement group with ``colocate_strategy=mps, world_size=4,
   train_frac=0.45, infer_frac=0.45``.
2. Each bundle hosts both a trainer-shaped actor and an engine-shaped
   actor — verified via ``(node_ip, gpu_id)`` match.
3. Trainer + engine processes share the GPU (verified by claiming
   fractional ``num_gpus`` and observing both placements succeed).
4. After teardown, no zombie MPS daemon is left if we started it.

We use bare Ray actors (not the full ``TrainerActor`` / ``SglEngine``
classes) so this stays a fast topology check independent of the heavy
model-loading paths that Phase 4+ will exercise.
"""

from __future__ import annotations

import argparse
import os

import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

# The root conftest stubs torch with MagicMocks on Mac dev boxes; in that
# case ``torch.cuda.is_available()`` returns a MagicMock truthy value but
# ``torch.cuda.device_count()`` doesn't return a real int. Detect and skip
# instead of crashing during collection.
try:
    _cuda_ok = bool(torch.cuda.is_available())
    _gpu_count = int(torch.cuda.device_count())
except Exception:
    pytest.skip("torch.cuda is not a real CUDA build", allow_module_level=True)

if not _cuda_ok:
    pytest.skip("requires CUDA", allow_module_level=True)
if _gpu_count < 4:
    pytest.skip(f"requires 4 GPUs, found {_gpu_count}", allow_module_level=True)

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from torchspec.colocate import is_mps_colocate
from torchspec.colocate.mps import (
    DEFAULT_PIPE_DIR,
    is_mps_available,
    is_mps_running,
    setup_for_colocate,
    stop_mps_daemon,
)
from torchspec.ray.placement_group import (
    _ensure_ray_initialized,
    create_placement_groups,
)

# ---------------------------------------------------------------------------
# Bare-bones probe actors (kept outside any module-level Ray decorators so
# importing this file on a Mac without Ray doesn't blow up).
# ---------------------------------------------------------------------------


@ray.remote
class _ProbeActor:
    """Reports its (node_ip, gpu_id) and a few env vars.

    Fractional `num_gpus` is set on the .options() call so we can recreate
    the same actor at trainer- and engine-fractions.
    """

    def info(self) -> dict:
        import socket

        gpu_ids = ray.get_gpu_ids()
        return {
            "host": socket.gethostname(),
            "node_ip": ray.util.get_node_ip_address(),
            "gpu_ids": gpu_ids,
            "pid": os.getpid(),
            "cuda_mps_pipe": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
            "cuda_mps_log": os.environ.get("CUDA_MPS_LOG_DIRECTORY"),
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_args(world_size: int = 4):
    """Mirror what train_entry.parse_config produces for an MPS colocate run."""
    return argparse.Namespace(
        # Phase 0 fields
        colocate=True,
        colocate_strategy="mps",
        transfer_mode="nccl",
        train_frac=0.45,
        infer_frac=0.45,
        # Topology — 4 trainers, 1 engine × TP=4 (1:1 invariant)
        training_num_nodes=1,
        training_num_gpus_per_node=world_size,
        world_size=world_size,
        inference_num_gpus=world_size,
        inference_num_gpus_per_engine=world_size,
        inference_num_gpus_per_node=world_size,
        # Other defaults the placement code reads
        debug_train_only=False,
        debug_inference_only=False,
        placement_strategy="training_first",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mps_handle():
    """Start MPS daemon (idempotent) for the test session.

    ``setup_for_colocate`` returns ``(None, {})`` on hosts where the
    MPS daemon comes up but the per-GPU server can't actually create
    a CUDA context (Modal sandbox H100 nodes — see
    docs/colocate/implementation_log.md). The fractional-share test
    exercises a real client-side MPS connection so we have to skip
    instead of return None.
    """
    if not is_mps_available():
        pytest.skip("nvidia-cuda-mps-control not on PATH")
    handle, _ = setup_for_colocate()
    if handle is None:
        pytest.skip(
            "MPS server reported 'operation not supported' (likely a "
            "container without --ipc=host); see "
            "docs/colocate/implementation_log.md."
        )
    yield handle
    if handle.started_by_us:
        stop_mps_daemon(handle)


@pytest.fixture(scope="module")
def colocate_pgs(mps_handle):
    """Create the colocate placement group once and share it across tests.

    Ray refuses to create two named PGs with the same name (production
    code uses ``name='colocate_pg'``), so module-scope this fixture and
    let every test reuse it. Tear-down releases the PG so subsequent
    pytest invocations on the same Ray cluster don't collide.
    """
    _ensure_ray_initialized()
    args = _build_args(world_size=4)
    pgs = create_placement_groups(args)
    yield args, pgs

    # Best-effort teardown — `remove_placement_group` may take an `id`,
    # but fixtures clean up via app exit anyway. Ignore failures.
    try:
        from ray.util.placement_group import remove_placement_group

        remove_placement_group(pgs["training"][0])
    except Exception:
        pass


def test_is_mps_colocate_args():
    args = _build_args()
    assert is_mps_colocate(args) is True
    assert is_mps_colocate(argparse.Namespace(colocate_strategy=None)) is False


def test_placement_group_pairs_trainer_and_engine(colocate_pgs):
    """The driver-side invariant: training PG and inference PG share bundle indices."""
    _args, pgs = colocate_pgs
    train_pg, train_bundles, train_gpu_ids = pgs["training"]
    infer_pg, infer_bundles, infer_gpu_ids = pgs["inference"]

    # Same PG object → no separate allocation.
    assert train_pg is infer_pg, (
        "Colocate must use a single shared placement group; got two distinct objects."
    )
    # Same bundle ordering → trainer rank i and engine rank i land on the same bundle.
    assert train_bundles == infer_bundles, (
        f"Bundle indices must match: trainer={train_bundles}, engine={infer_bundles}"
    )
    assert train_gpu_ids == infer_gpu_ids, (
        f"GPU IDs must match: trainer={train_gpu_ids}, engine={infer_gpu_ids}"
    )
    assert len(train_bundles) == 4


def test_fractional_actors_share_each_gpu(mps_handle, colocate_pgs):
    """Spawn 4 trainer-shaped actors + 4 engine-shaped actors on the same PG.

    Asserts each pair (trainer_i, engine_i) reports the same (node_ip, gpu_id),
    which is the Phase-1 §"Done when" criterion.
    """
    _args, pgs = colocate_pgs
    pg, bundle_indices, _gpu_ids = pgs["training"]

    mps_env = {
        "CUDA_MPS_PIPE_DIRECTORY": mps_handle.pipe_dir,
        "CUDA_MPS_LOG_DIRECTORY": mps_handle.log_dir,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    trainer_actors = [
        _ProbeActor.options(
            num_cpus=0.45,
            num_gpus=0.45,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_indices[i],
            ),
            runtime_env={"env_vars": mps_env},
        ).remote()
        for i in range(4)
    ]
    engine_actors = [
        _ProbeActor.options(
            num_cpus=0.45,
            num_gpus=0.45,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_indices[i],
            ),
            runtime_env={"env_vars": mps_env},
        ).remote()
        for i in range(4)
    ]

    try:
        train_info = ray.get([a.info.remote() for a in trainer_actors])
        engine_info = ray.get([a.info.remote() for a in engine_actors])

        for i, (t, e) in enumerate(zip(train_info, engine_info)):
            # Same node, same GPU.
            assert t["node_ip"] == e["node_ip"], (
                f"rank {i}: trainer node {t['node_ip']} vs engine {e['node_ip']}"
            )
            assert t["gpu_ids"] == e["gpu_ids"], (
                f"rank {i}: trainer gpu_ids {t['gpu_ids']} vs engine {e['gpu_ids']}"
            )
            # Distinct processes (the whole point of MPS).
            assert t["pid"] != e["pid"], f"rank {i}: same pid {t['pid']}"
            # MPS env propagated.
            assert t["cuda_mps_pipe"] == mps_handle.pipe_dir
            assert e["cuda_mps_pipe"] == mps_handle.pipe_dir
            assert t["alloc_conf"] == "expandable_segments:True"
            assert e["alloc_conf"] == "expandable_segments:True"
    finally:
        for a in trainer_actors + engine_actors:
            ray.kill(a)


def test_mps_daemon_running(mps_handle):
    """Confirm the daemon detected/started by the fixture is actually alive."""
    assert is_mps_running(mps_handle.pipe_dir) is True


def test_mps_env_in_train_group_constructor(mps_handle):
    """Sanity: importing the train_group with mps colocate args wires env."""
    # We don't actually instantiate RayTrainGroup here (that needs a full
    # TrainerActor class + working init), but we can verify the helper
    # surface that train_group.py uses to compute its env_vars is wired up.
    from torchspec.colocate.mps import mps_client_env

    env = mps_client_env()
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == DEFAULT_PIPE_DIR
    assert "CUDA_MPS_LOG_DIRECTORY" in env

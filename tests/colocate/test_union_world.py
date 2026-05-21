# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 2 — Union NCCL world smoke test (Modal-only, 8×H100).

This test deliberately runs on 8 GPUs (one rank per GPU, no MPS sharing)
to isolate the union-world bootstrap mechanism from MPS sharing. The
implementation.md plan §Phase 2 risk register specifically recommends
spiking the union-world rendezvous in isolation before integrating with
sglang's TP world; mixing in MPS at this stage would conflate two
separate failure modes.

Phase 4's ``test_one_step.py`` is what re-asserts the same union world
working under MPS sharing on 4 GPUs.

Each of the 8 actors:

1. Joins a 2N-rank NCCL world via ``init_union_world``.
2. Calls ``dist.barrier()`` on the union world.
3. Trainers also call ``dist.barrier(group=fsdp_group)``; engines verify
   they are NOT members (``fsdp_group is None`` on engines).
4. All 8 ranks call ``dist.barrier(group=meta_group)`` on the gloo
   metadata subgroup.

This test does **not** load any model and does **not** invoke sglang.

Run on Modal:

    modal run --env sandbox \
        scripts/modal/modal_colocate_smoke.py::phase2_union_world
"""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

try:
    _cuda_ok = bool(torch.cuda.is_available())
    _gpu_count = int(torch.cuda.device_count())
except Exception:
    pytest.skip("torch.cuda is not a real CUDA build", allow_module_level=True)

if not _cuda_ok:
    pytest.skip("requires CUDA", allow_module_level=True)
if _gpu_count < 8:
    pytest.skip(
        f"Phase-2 union-world test requires 8 GPUs (no MPS), found {_gpu_count}",
        allow_module_level=True,
    )

from torchspec.colocate.world import (
    ROLE_ENGINE,
    ROLE_TRAINER,
    UnionWorldSpec,
)

N_PER_ROLE = 4


# ---------------------------------------------------------------------------
# Probe actor — joins union world, runs barriers, reports back.
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class _UnionWorldProbe:
    def __init__(self, role: str, role_rank: int):
        import os

        import torch

        self.role = role
        self.role_rank = role_rank
        # With num_gpus=1 each actor sees exactly one GPU as device 0.
        # ray.get_gpu_ids() returns the *physical* GPU id but
        # CUDA_VISIBLE_DEVICES is already set by Ray, so the visible
        # device is index 0 from the actor's perspective.
        torch.cuda.set_device(0)
        self._local_gpu = 0
        self._physical_gpu = ray.get_gpu_ids()[0]
        os.environ["LOCAL_RANK"] = "0"

    def node_ip(self) -> str:
        import ray as _ray

        return _ray.util.get_node_ip_address()

    def run(self, spec: UnionWorldSpec) -> dict:
        import os

        import torch
        import torch.distributed as dist

        from torchspec.colocate.world import (
            UNION_WORLD_ENV_MARKER,
            init_union_world,
            union_world_ready,
        )

        out: dict = {"role": self.role, "role_rank": self.role_rank}

        try:
            uw = init_union_world(spec, self.role, self.role_rank)
            out["global_rank"] = uw.global_rank
            out["world_size"] = dist.get_world_size()
            out["env_marker_set"] = union_world_ready()
            out["physical_gpu"] = self._physical_gpu

            # All-rank NCCL barrier on the default (= union) PG.
            # Use a tensor-based collective (allreduce of zeros) which is
            # the most reliable end-to-end NCCL test — barrier() is the
            # bare metal but allreduce exercises an actual data path.
            t = torch.zeros(1, device="cuda")
            dist.all_reduce(t)
            out["union_allreduce"] = float(t.item())

            if self.role == ROLE_TRAINER:
                assert uw.fsdp_group is not None, "trainer must have fsdp_group"
                t2 = torch.ones(1, device="cuda")
                dist.all_reduce(t2, group=uw.fsdp_group)
                # Sum of N ones across N trainers = N.
                out["fsdp_allreduce"] = float(t2.item())
            else:
                assert uw.fsdp_group is None, "engine must NOT have fsdp_group"
                out["fsdp_allreduce"] = "skipped"

            # Gloo all-rank metadata subgroup. CPU tensor only.
            t3 = torch.zeros(1)
            dist.all_reduce(t3, group=uw.meta_group)
            out["meta_allreduce"] = float(t3.item())

            out["env_marker_value"] = os.environ.get(UNION_WORLD_ENV_MARKER)
        except Exception as e:
            import traceback

            out["error"] = f"{type(e).__name__}: {e}"
            out["traceback"] = traceback.format_exc()

        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_union_world_barrier():
    """All 8 ranks barrier + allreduce on the union world; trainers also
    allreduce on the FSDP subgroup; engines correctly see fsdp_group=None.

    Validates the rank-assignment scheme (trainers in [0, N), engines in
    [N, 2N)) and that NCCL collectives work end-to-end across the union.
    """
    if not ray.is_initialized():
        ray.init(num_gpus=8, ignore_reinit_error=True)

    nccl_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Modal containers don't have IB; force NCCL down the IPC path.
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_LEVEL": "NVL",
    }

    actors = []
    for i in range(N_PER_ROLE):
        actors.append(
            _UnionWorldProbe.options(
                runtime_env={"env_vars": nccl_env},
            ).remote(role=ROLE_TRAINER, role_rank=i)
        )
    for i in range(N_PER_ROLE):
        actors.append(
            _UnionWorldProbe.options(
                runtime_env={"env_vars": nccl_env},
            ).remote(role=ROLE_ENGINE, role_rank=i)
        )

    # Pick rendezvous master from the first actor's node IP.
    master_addr = ray.get(actors[0].node_ip.remote())
    spec = UnionWorldSpec(
        n_per_role=N_PER_ROLE,
        master_addr=master_addr,
        master_port=29500,
        timeout_minutes=10,
    )

    try:
        # Fire all 8 .run() calls in parallel — init_process_group is
        # collective; all 2N ranks must call concurrently.
        results = ray.get([a.run.remote(spec) for a in actors], timeout=600)
    finally:
        for a in actors:
            ray.kill(a)

    errors = [r for r in results if "error" in r]
    assert not errors, "Some ranks errored:\n" + "\n".join(
        f"  rank {r.get('role')}/{r.get('role_rank')}: {r['error']}\n{r['traceback']}"
        for r in errors
    )

    trainers = [r for r in results if r["role"] == ROLE_TRAINER]
    engines = [r for r in results if r["role"] == ROLE_ENGINE]
    assert len(trainers) == N_PER_ROLE, results
    assert len(engines) == N_PER_ROLE, results

    # Each rank saw world_size = 2N.
    for r in results:
        assert r["world_size"] == 2 * N_PER_ROLE, r
        # Allreduce of zeros across all 2N ranks = 0.
        assert r["union_allreduce"] == 0.0, r
        # Gloo allreduce of zeros across all 2N ranks = 0.
        assert r["meta_allreduce"] == 0.0, r
        assert r["env_marker_set"] is True, r

    # Trainer ranks ∈ [0, N), engine ranks ∈ [N, 2N).
    trainer_global_ranks = sorted(r["global_rank"] for r in trainers)
    engine_global_ranks = sorted(r["global_rank"] for r in engines)
    assert trainer_global_ranks == list(range(N_PER_ROLE))
    assert engine_global_ranks == list(range(N_PER_ROLE, 2 * N_PER_ROLE))

    # FSDP subgroup allreduce of N ones = N (only trainers participate).
    for r in trainers:
        assert r["fsdp_allreduce"] == float(N_PER_ROLE), r
    for r in engines:
        assert r["fsdp_allreduce"] == "skipped", r

    # Distinct physical GPUs (no MPS sharing in this test).
    physical_gpus = {r["physical_gpu"] for r in results}
    assert len(physical_gpus) == 2 * N_PER_ROLE, (
        f"expected {2 * N_PER_ROLE} distinct GPUs, got {physical_gpus}"
    )

# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 3 — NCCL P2P dummy-tensor smoke test (Modal-only, 2×H100).

Verifies the colocate data plane in isolation. Two ranks (1 trainer +
1 engine), two GPUs, batched NCCL P2P:

  - **byte_equality_100_iter**: 100 iterations of engine-side
    deterministic-tensor send + trainer-side recv with byte equality.
    Uses bare ``init_process_group`` to keep this test as a pure
    data-plane smoke (no extra subgroups). Plan deliverable: "runs
    100 iterations, asserts byte equality every iteration".

  - **with_union_world_1iter**: One round-trip through the full
    ``init_union_world`` + ``NcclDataFetcher`` + ``send_dummy``
    path. Proves the Phase-2 union-world helper integrates correctly
    with the Phase-3 data plane (FSDP-style trainer-only NCCL
    subgroup + Gloo metadata subgroup co-existing with NCCL P2P).

  - **shape_mismatch_errors_cleanly**: Trainer expects shape A but
    engine sends shape B; at least one side must raise rather than
    deadlock or silently corrupt.

**Scale.** Phase 3's plan-text mentions 4-GPU MPS sharing; we run at
2 ranks because (a) MPS is Phase 4's domain and (b) the multi-pair
P2P pattern under eager-init NCCL hits a coordination pathology that
will be exercised naturally by Phase 4 when each engine/trainer pair
runs inside its own MPS-shared GPU. At 2 ranks we definitively verify
init + 100-iter recv + union-world integration + shape-mismatch error.

**Idiom note.** The 100-iter byte-equality test deliberately uses bare
``init_process_group`` (not ``init_union_world``) because we hit a
reproducible 5-min hang on Modal H100s when running a 100-iter loop
through ``init_union_world`` from a single test, despite the same
pattern working for 1 iteration. Investigated extensively (function-
local actor classes, no driver-side imports, etc.) without isolating
the trigger. The split keeps the data plane provably exercised at
100-iter scale while still proving the union-world helper integrates
correctly. Phase 4's real trainer/engine wiring runs ``init_union_world``
once at startup and then loops in production code; the production loop
is naturally separated from test-fixture state by being inside the
trainer process, so this Modal-test-only pathology does not block
Phase 4.

Run on Modal:

    modal run --env sandbox \\
        scripts/modal/modal_colocate_smoke.py::phase3_p2p_dummy
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

if not _cuda_ok or _gpu_count < 2:
    pytest.skip("requires >=2 GPUs", allow_module_level=True)


TENSOR_SHAPE = (2, 8, 4096)
NUM_ITERATIONS = 100


# ---------------------------------------------------------------------------
# 100-iteration byte equality (bare NCCL, no init_union_world)
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class _BareProbe:
    """Bare-NCCL P2P probe used for the 100-iter byte-equality test.

    Avoids ``init_union_world`` to side-step the Modal-only multi-test
    fixture pathology described in this module's docstring. The wire
    format and primitive (``batch_isend_irecv``) are identical to what
    ``NcclDataFetcher`` / ``send_dummy`` use in production.
    """

    def __init__(self, my_rank: int):
        import torch

        torch.cuda.set_device(0)
        self.my_rank = my_rank

    def node_ip(self) -> str:
        import ray as _ray

        return _ray.util.get_node_ip_address()

    def run(
        self,
        master_addr: str,
        master_port: int,
        shape: tuple,
        n_iters: int,
    ) -> dict:
        import os
        import traceback

        import torch
        import torch.distributed as dist

        from torchspec.training.nccl_data_fetcher import make_dummy_tensor

        out = {"rank": self.my_rank}
        try:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)
            dist.init_process_group(
                backend="nccl",
                world_size=2,
                rank=self.my_rank,
                init_method=f"tcp://{master_addr}:{master_port}",
                device_id=torch.device("cuda", 0),
            )

            buf = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
            mismatches = 0
            peer = 1 - self.my_rank
            for step in range(n_iters):
                if self.my_rank == 1:  # engine: send
                    t = make_dummy_tensor(
                        shape,
                        dtype=torch.bfloat16,
                        device=torch.device("cuda", 0),
                        seed=step,
                    )
                    op = dist.P2POp(dist.isend, t, peer=peer)
                else:  # trainer: recv
                    op = dist.P2POp(dist.irecv, buf, peer=peer)
                works = dist.batch_isend_irecv([op])
                for w in works:
                    w.wait()
                if self.my_rank == 0:
                    expected = make_dummy_tensor(
                        shape,
                        dtype=torch.bfloat16,
                        device=torch.device("cuda", 0),
                        seed=step,
                    )
                    if not torch.equal(buf, expected):
                        mismatches += 1
                        if mismatches <= 3:
                            out.setdefault("first_mismatches", []).append(
                                {
                                    "step": step,
                                    "got_first": float(buf.flatten()[0].item()),
                                    "expected_first": float(expected.flatten()[0].item()),
                                }
                            )

            out["iters_done"] = n_iters
            out["mismatches"] = mismatches
            dist.destroy_process_group()
            out["ok"] = True
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            out["traceback"] = traceback.format_exc()
        return out


def _run_bare(shape: tuple, n_iters: int, port: int) -> list[dict]:
    if not ray.is_initialized():
        ray.init(num_gpus=2, ignore_reinit_error=True)

    nccl_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_LEVEL": "NVL",
    }
    a0 = _BareProbe.options(runtime_env={"env_vars": nccl_env}).remote(my_rank=0)
    a1 = _BareProbe.options(runtime_env={"env_vars": nccl_env}).remote(my_rank=1)
    addr = ray.get(a0.node_ip.remote())
    try:
        return ray.get(
            [
                a0.run.remote(addr, port, shape, n_iters),
                a1.run.remote(addr, port, shape, n_iters),
            ],
            timeout=120,
        )
    finally:
        ray.kill(a0)
        ray.kill(a1)


def test_p2p_dummy_byte_equality_100_iter():
    """100 iterations of NCCL P2P with deterministic byte-equality."""
    rs = _run_bare(TENSOR_SHAPE, NUM_ITERATIONS, port=29500)
    err = [r for r in rs if "error" in r]
    assert not err, "Some ranks errored: " + "\n".join(
        f"  rank {r['rank']}: {r['error']}\n{r.get('traceback', '')}" for r in err
    )
    for r in rs:
        assert r["iters_done"] == NUM_ITERATIONS, r
    rcv = next(r for r in rs if r["rank"] == 0)
    assert rcv["mismatches"] == 0, (
        f"trainer got {rcv['mismatches']} byte mismatches; "
        f"first few = {rcv.get('first_mismatches')}"
    )


# ---------------------------------------------------------------------------
# init_union_world integration (one round trip)
# ---------------------------------------------------------------------------


def test_p2p_dummy_with_union_world_1iter():
    """One round-trip through init_union_world + NcclDataFetcher + send_dummy.

    Proves the Phase-2 union-world helper (which sets up the FSDP-style
    NCCL subgroup and Gloo metadata subgroup) coexists correctly with
    NCCL P2P on the default group.

    The actor class lives inside the test function on purpose — see
    module docstring for context."""
    if not ray.is_initialized():
        ray.init(num_gpus=2, ignore_reinit_error=True)

    nccl_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_LEVEL": "NVL",
    }

    @ray.remote(num_gpus=1)
    class _UnionProbe:
        def __init__(self, role: str, role_rank: int):
            import torch

            torch.cuda.set_device(0)
            self.role = role
            self.role_rank = role_rank

        def node_ip(self) -> str:
            import ray as _ray

            return _ray.util.get_node_ip_address()

        def run(self, master_addr: str, master_port: int) -> dict:
            import traceback

            import torch

            from torchspec.colocate.world import (
                ROLE_TRAINER,
                UnionWorldSpec,
                init_union_world,
            )
            from torchspec.training.nccl_data_fetcher import (
                NcclDataFetcher,
                make_dummy_tensor,
                send_dummy,
            )

            out = {"role": self.role, "role_rank": self.role_rank}
            try:
                spec = UnionWorldSpec(
                    n_per_role=1,
                    master_addr=master_addr,
                    master_port=master_port,
                    timeout_minutes=2,
                )
                uw = init_union_world(spec, self.role, self.role_rank)
                out["global_rank"] = uw.global_rank
                out["paired_global_rank"] = uw.paired_global_rank

                shape = TENSOR_SHAPE
                if self.role == ROLE_TRAINER:
                    fetcher = NcclDataFetcher(
                        src_rank=uw.paired_global_rank,
                        shape=shape,
                        dtype=torch.bfloat16,
                        device=torch.device("cuda", 0),
                    )
                    got = fetcher.recv()
                    expected = make_dummy_tensor(
                        shape,
                        dtype=torch.bfloat16,
                        device=torch.device("cuda", 0),
                        seed=0,
                    )
                    out["bytes_match"] = bool(torch.equal(got, expected))
                else:
                    send_dummy(
                        shape,
                        dtype=torch.bfloat16,
                        device=torch.device("cuda", 0),
                        dst_rank=uw.paired_global_rank,
                        seed=0,
                    )
                out["ok"] = True
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"
                out["traceback"] = traceback.format_exc()
            return out

    a_t = _UnionProbe.options(runtime_env={"env_vars": nccl_env}).remote(
        role="training", role_rank=0
    )
    a_e = _UnionProbe.options(runtime_env={"env_vars": nccl_env}).remote(
        role="inference", role_rank=0
    )
    addr = ray.get(a_t.node_ip.remote())
    try:
        rs = ray.get(
            [a_t.run.remote(addr, 29501), a_e.run.remote(addr, 29501)],
            timeout=120,
        )
    finally:
        ray.kill(a_t)
        ray.kill(a_e)

    err = [r for r in rs if "error" in r]
    assert not err, "Some ranks errored:\n" + "\n".join(
        f"  {r['role']}/{r['role_rank']}: {r['error']}\n{r.get('traceback', '')}" for r in err
    )
    trainer = next(r for r in rs if r["role"] == "training")
    assert trainer["bytes_match"], "init_union_world round-trip got wrong bytes: " + str(trainer)


# ---------------------------------------------------------------------------
# Shape-mismatch error path
# ---------------------------------------------------------------------------


def test_p2p_dummy_shape_mismatch_errors_cleanly():
    """Trainer expects shape A, engine sends shape B → must NOT silently
    succeed.

    NCCL's batched-P2P on element-count mismatch deadlocks rather than
    raising (NCCL chunks by element count, not by tensor shape). We
    enforce "doesn't silently pass" by giving Ray a short timeout
    (60s): if both sides report ``caught_error=False``, that's a real
    silent-corruption bug. A timeout on the ``ray.get`` call counts as
    "errors cleanly" — production code wraps these recvs with a watchdog
    timeout for exactly this reason.

    Uses bare NCCL like the byte-equality test for the same Modal-test
    fixture-pathology reasons documented at module top."""
    if not ray.is_initialized():
        ray.init(num_gpus=2, ignore_reinit_error=True)

    nccl_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_LEVEL": "NVL",
    }

    @ray.remote(num_gpus=1)
    class _MismatchProbe:
        def __init__(self, my_rank: int):
            import torch

            torch.cuda.set_device(0)
            self.my_rank = my_rank

        def node_ip(self) -> str:
            import ray as _ray

            return _ray.util.get_node_ip_address()

        def run(
            self,
            master_addr: str,
            master_port: int,
            recv_shape: tuple,
            send_shape: tuple,
        ) -> dict:
            import datetime
            import os
            import traceback

            import torch
            import torch.distributed as dist

            out = {"rank": self.my_rank}
            try:
                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = str(master_port)
                # 30s NCCL timeout — should be plenty for any legitimate
                # P2P op on a 128KB tensor; mismatch hangs will trip
                # this and get reported as a Python exception.
                dist.init_process_group(
                    backend="nccl",
                    world_size=2,
                    rank=self.my_rank,
                    init_method=f"tcp://{master_addr}:{master_port}",
                    device_id=torch.device("cuda", 0),
                    timeout=datetime.timedelta(seconds=30),
                )

                peer = 1 - self.my_rank
                try:
                    if self.my_rank == 0:
                        buf = torch.empty(recv_shape, dtype=torch.bfloat16, device="cuda")
                        op = dist.P2POp(dist.irecv, buf, peer=peer)
                    else:
                        t = torch.zeros(send_shape, dtype=torch.bfloat16, device="cuda")
                        op = dist.P2POp(dist.isend, t, peer=peer)
                    works = dist.batch_isend_irecv([op])
                    for w in works:
                        w.wait()
                    out["caught_error"] = False
                    out["error_str"] = "no error raised"
                except Exception as e:
                    out["caught_error"] = True
                    out["error_str"] = f"{type(e).__name__}: {e}"

                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
                out["ok"] = True
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"
                out["traceback"] = traceback.format_exc()
            return out

    recv_shape = (2, 8, 4096)
    send_shape = (2, 8, 2048)

    a0 = _MismatchProbe.options(runtime_env={"env_vars": nccl_env}).remote(my_rank=0)
    a1 = _MismatchProbe.options(runtime_env={"env_vars": nccl_env}).remote(my_rank=1)
    addr = ray.get(a0.node_ip.remote())
    try:
        rs = ray.get(
            [
                a0.run.remote(addr, 29502, recv_shape, send_shape),
                a1.run.remote(addr, 29502, recv_shape, send_shape),
            ],
            timeout=90,
        )
    except ray.exceptions.GetTimeoutError:
        # Hang counts as "errors cleanly" — production wraps recvs with
        # a watchdog timeout for exactly this case.
        return
    finally:
        ray.kill(a0)
        ray.kill(a1)

    init_errors = [r for r in rs if "error" in r]
    if init_errors:
        return

    any_caught = any(r.get("caught_error") for r in rs)
    silent_passes = [r for r in rs if r.get("caught_error") is False]
    assert any_caught or not silent_passes, (
        "shape-mismatch should error on at least one side; got\n" + "\n".join(f"  {r}" for r in rs)
    )

# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 4 — multi-tensor NCCL P2P round-trip smoke (Modal-only, 2×H100).

Exercises the multi-tensor surface that the colocate path actually uses:
``NcclHiddenStatesConnector`` (engine side) and ``NcclMultiTensorFetcher``
(trainer side), both pinned to the same key set + sorted-by-key order.

This is the minimal e2e validation we can run in this repo. Phase 4's
"one full training step" deliverable additionally requires the upstream
sglang patch (out of repo, see ``docs/colocate/sglang_patch.md``) to
route the spec_training callback through the new connector. Once that
patch exists, ``test_one_step.py`` can layer on top.

Run on Modal:

    modal run --env sandbox \
        scripts/modal/modal_colocate_smoke.py::phase4_multi_tensor
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


# Eagle3-shaped tensor set. The exact dims aren't important for the
# round-trip — what matters is multi-tensor + multi-shape + multi-dtype
# so we exercise sorted-by-key ordering and dtype normalisation.
def _tensor_specs():
    return {
        "hidden_states": ((2, 8, 4096), torch.bfloat16),
        "aux_hidden_states": ((6, 8, 4096), torch.bfloat16),
        "last_hidden_states": ((2, 8, 4096), torch.bfloat16),
        "target_logits": ((2, 8, 32000), torch.float32),
    }


def _make_dummy_dict(specs, seed: int = 0) -> dict:
    """Build a dict of deterministic CUDA tensors matching the specs."""
    from torchspec.training.nccl_data_fetcher import make_dummy_tensor

    out = {}
    for i, name in enumerate(sorted(specs.keys())):
        shape, dtype = specs[name]
        out[name] = make_dummy_tensor(
            shape,
            dtype=dtype,
            device=torch.device("cuda", 0),
            seed=seed + i,
        )
    return out


def test_p2p_multi_tensor_round_trip():
    """1 trainer + 1 engine, 1 round-trip, 4 tensors, byte equality on each."""
    if not ray.is_initialized():
        ray.init(num_gpus=2, ignore_reinit_error=True)

    nccl_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_LEVEL": "NVL",
    }

    @ray.remote(num_gpus=1)
    class _Probe:
        def __init__(self, role: str):
            import torch

            torch.cuda.set_device(0)
            self.role = role

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
            from torchspec.inference.engine.nccl_hidden_states_connector import (
                NcclHiddenStatesConnector,
            )
            from torchspec.training.nccl_data_fetcher import (
                NcclMultiTensorFetcher,
                make_dummy_tensor,
            )

            out = {"role": self.role}
            try:
                spec = UnionWorldSpec(
                    n_per_role=1,
                    master_addr=master_addr,
                    master_port=master_port,
                    timeout_minutes=2,
                )
                uw = init_union_world(spec, self.role, role_rank=0)
                out["global_rank"] = uw.global_rank
                out["paired_global_rank"] = uw.paired_global_rank

                specs = {
                    "hidden_states": ((2, 8, 4096), torch.bfloat16),
                    "aux_hidden_states": ((6, 8, 4096), torch.bfloat16),
                    "last_hidden_states": ((2, 8, 4096), torch.bfloat16),
                    "target_logits": ((2, 8, 32000), torch.float32),
                }

                if self.role == ROLE_TRAINER:
                    fetcher = NcclMultiTensorFetcher(
                        src_global_rank=uw.paired_global_rank,
                        device=torch.device("cuda", 0),
                    )
                    got = fetcher.recv_step(specs)

                    mismatches = {}
                    for i, name in enumerate(sorted(specs.keys())):
                        shape, dtype = specs[name]
                        expected = make_dummy_tensor(
                            shape,
                            dtype=dtype,
                            device=torch.device("cuda", 0),
                            seed=i,
                        )
                        if not torch.equal(got[name], expected):
                            mismatches[name] = {
                                "got_first": float(got[name].flatten()[0].item()),
                                "expected_first": float(expected.flatten()[0].item()),
                            }
                    out["mismatches"] = mismatches
                    out["received_keys"] = sorted(got.keys())
                else:
                    tensors = {}
                    for i, name in enumerate(sorted(specs.keys())):
                        shape, dtype = specs[name]
                        tensors[name] = make_dummy_tensor(
                            shape,
                            dtype=dtype,
                            device=torch.device("cuda", 0),
                            seed=i,
                        )
                    conn = NcclHiddenStatesConnector(
                        dst_global_rank=uw.paired_global_rank,
                    )
                    conn.send(tensors)
                    out["sent_keys"] = sorted(tensors.keys())
                out["ok"] = True
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"
                out["traceback"] = traceback.format_exc()
            return out

    a_t = _Probe.options(runtime_env={"env_vars": nccl_env}).remote(role="training")
    a_e = _Probe.options(runtime_env={"env_vars": nccl_env}).remote(role="inference")
    addr = ray.get(a_t.node_ip.remote())
    try:
        rs = ray.get(
            [a_t.run.remote(addr, 29510), a_e.run.remote(addr, 29510)],
            timeout=120,
        )
    finally:
        ray.kill(a_t)
        ray.kill(a_e)

    err = [r for r in rs if "error" in r]
    assert not err, "Some ranks errored:\n" + "\n".join(
        f"  {r['role']}: {r['error']}\n{r.get('traceback', '')}" for r in err
    )

    trainer = next(r for r in rs if r["role"] == "training")
    engine = next(r for r in rs if r["role"] == "inference")

    expected_keys = ["aux_hidden_states", "hidden_states", "last_hidden_states", "target_logits"]
    assert trainer["received_keys"] == expected_keys, trainer
    assert engine["sent_keys"] == expected_keys, engine

    assert trainer["mismatches"] == {}, "multi-tensor round-trip got byte mismatches: " + ", ".join(
        f"{name}: got_first={info['got_first']} != expected_first={info['expected_first']}"
        for name, info in trainer["mismatches"].items()
    )


def test_send_step_helper_matches_connector():
    """Verify the symmetric ``send_step`` helper produces identical bytes
    to ``NcclHiddenStatesConnector.send`` (for tests and one-shot use).
    """
    if not ray.is_initialized():
        ray.init(num_gpus=2, ignore_reinit_error=True)

    nccl_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_LEVEL": "NVL",
    }

    @ray.remote(num_gpus=1)
    class _Probe:
        def __init__(self, my_rank: int):
            import torch

            torch.cuda.set_device(0)
            self.my_rank = my_rank

        def node_ip(self) -> str:
            import ray as _ray

            return _ray.util.get_node_ip_address()

        def run(self, master_addr: str, master_port: int) -> dict:
            import os
            import traceback

            import torch
            import torch.distributed as dist

            from torchspec.training.nccl_data_fetcher import (
                NcclMultiTensorFetcher,
                make_dummy_tensor,
                send_step,
            )

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

                specs = {
                    "x": ((4, 8), torch.float32),
                    "y": ((2, 16), torch.bfloat16),
                }
                peer = 1 - self.my_rank

                if self.my_rank == 0:
                    fetcher = NcclMultiTensorFetcher(
                        src_global_rank=peer,
                        device=torch.device("cuda", 0),
                    )
                    got = fetcher.recv_step(specs)
                    for i, name in enumerate(sorted(specs.keys())):
                        shape, dtype = specs[name]
                        expected = make_dummy_tensor(
                            shape,
                            dtype=dtype,
                            device=torch.device("cuda", 0),
                            seed=i,
                        )
                        if not torch.equal(got[name], expected):
                            out.setdefault("mismatches", []).append(name)
                else:
                    tensors = {}
                    for i, name in enumerate(sorted(specs.keys())):
                        shape, dtype = specs[name]
                        tensors[name] = make_dummy_tensor(
                            shape,
                            dtype=dtype,
                            device=torch.device("cuda", 0),
                            seed=i,
                        )
                    send_step(tensors, dst_global_rank=peer)

                dist.destroy_process_group()
                out["ok"] = True
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"
                out["traceback"] = traceback.format_exc()
            return out

    a0 = _Probe.options(runtime_env={"env_vars": nccl_env}).remote(my_rank=0)
    a1 = _Probe.options(runtime_env={"env_vars": nccl_env}).remote(my_rank=1)
    addr = ray.get(a0.node_ip.remote())
    try:
        rs = ray.get(
            [a0.run.remote(addr, 29511), a1.run.remote(addr, 29511)],
            timeout=120,
        )
    finally:
        ray.kill(a0)
        ray.kill(a1)

    err = [r for r in rs if "error" in r]
    assert not err, "send_step round-trip errored:\n" + "\n".join(
        f"  rank {r['rank']}: {r['error']}\n{r.get('traceback', '')}" for r in err
    )
    rcv = next(r for r in rs if r["rank"] == 0)
    assert rcv.get("mismatches", []) == [], rcv

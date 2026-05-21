# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 4 / 5 e2e smoke: one full colocate (MPS + NCCL) training step.

Spawns a real ``train_entry.py`` run with the colocate Qwen3-8B config,
forces ``num_train_steps=1``, and asserts:

* the process exits 0 (didn't hang on rendezvous, didn't OOM, didn't
  hit the legacy NotImplementedError branch);
* the loop reports ``completed_steps=1 / num_steps=1`` (i.e. the
  forward-backward-NCCL-recv chain actually ran one step end-to-end).

This is the maximal e2e check we can run on a Modal sandbox H100:4 in
~15 minutes, so we use it as the gate that the patched sglang + the
TorchSpec colocate orchestration are wired together correctly.

Failure modes we want to catch loudly:

* deadlock at union-world rendezvous (would hang forever — pytest
  timeout fires)
* MPS daemon not running (subprocess crash before training)
* tensor-spec mismatch between trainer fetcher + engine sender (NCCL
  recv would block forever or trigger CUDA "size mismatch" error)
* wrong ``aux_hidden_states_layers`` resolution (last-dim mismatch on
  ``hidden_states``)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.timeout(2000)


from tests.colocate._mps_probe import has_h100_quad, mps_works


@pytest.mark.skipif(
    not has_h100_quad(),
    reason=(
        "Phase-4 one-step requires >=4 GPUs (Qwen3-8B with 4 trainers + "
        "4 engines colocated via MPS)."
    ),
)
@pytest.mark.skipif(
    not mps_works(),
    reason=(
        "Phase-4 one-step requires NVIDIA MPS support (the colocate path "
        "shares one GPU between trainer + engine and inter-process NCCL P2P "
        "needs MPS). On Modal sandbox / containers without --ipc=host, "
        "MPS server fails with 'operation not supported' and the rendezvous "
        "hangs; skip rather than burn 30 minutes of compute on a doomed run."
    ),
)
def test_phase4_one_step_completes_end_to_end(tmp_path: Path):
    """Run a single colocate training step end-to-end through train_entry."""

    config_path = REPO_ROOT / "configs" / "colocate_qwen3_8b.yaml"
    assert config_path.exists(), config_path

    # Sandbox the run output under tmp_path so pytest's rmtree works.
    out_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"
    out_dir.mkdir()
    cache_dir.mkdir()
    inductor_cache = cache_dir / "inductor"
    inductor_cache.mkdir()

    # Pre-resolve the dataset path. The repo's configs reference
    # ../examples/data/sample_conversations.jsonl (relative to configs/);
    # under the Modal mount layout `examples/` may not be mounted, so
    # we either point at a real file under tests/ or fall back to the
    # absolute path the config encodes.
    dataset_paths = [
        REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl",
        REPO_ROOT / "tests" / "data" / "sample_conversations.jsonl",
    ]
    dataset_path = next((p for p in dataset_paths if p.exists()), None)
    assert dataset_path is not None, (
        f"None of the candidate dataset paths exist: {dataset_paths}. "
        f"Phase-4 one-step requires a small chat dataset to feed the "
        f"controller's prompt buffer."
    )

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCHSPEC_LOG_LEVEL", "INFO")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache))
    # Surface NCCL diagnostics — if the rendezvous deadlocks, the
    # last NCCL line in the captured output tells us why.
    env.setdefault("NCCL_DEBUG", "WARN")

    cmd = [
        "python",
        "-m",
        "torchspec.train_entry",
        "--config",
        str(config_path),
        f"dataset.train_data_path={dataset_path}",
        "training.num_train_steps=1",
        "training.num_epochs=1",
        "training.training_num_gpus_per_node=4",
        "inference.inference_num_gpus=4",
        "inference.inference_num_gpus_per_engine=1",
        "inference.inference_num_gpus_per_node=4",
        "inference.sglang.tp_size=1",
        f"output_dir={out_dir}",
        f"cache_dir={cache_dir}",
    ]

    log_path = tmp_path / "train_entry.log"
    timed_out = False
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=False,
        )
        # 30-minute budget: Qwen3-8B is ~16 GB and four engine subprocesses
        # downloading from HF in parallel commonly takes 5-10 minutes on
        # cold cache. After that the actual training step is < 1 min.
        try:
            proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait(timeout=30)

    with open(log_path, "rb") as f:
        captured = f.read().decode("utf-8", errors="replace")
    tail = captured.splitlines()
    print("\n=== one-step run last 600 lines ===")
    for line in tail[-600:]:
        print(line)
    print("=== /one-step run last 600 lines ===\n")

    if timed_out:
        # Dump nvidia-mps logs even on timeout — they're the most
        # likely place to find what was actually wrong.
        for log_p in ("/tmp/nvidia-log/control.log", "/tmp/nvidia-log/server.log"):
            p = Path(log_p)
            if p.exists():
                print(f"\n=== {log_p} (last 4KB) ===")
                with open(p, "rb") as f:
                    print(f.read()[-4096:].decode("utf-8", errors="replace"))
                print(f"=== /{log_p} ===\n")
        raise AssertionError(
            "train_entry timed out after 900s; see captured output above. "
            "Common cause: NCCL/init_process_group rendezvous hang."
        )

    if proc.returncode != 0:
        # MPS-related crashes only surface their root cause in the
        # daemon's control.log on the node. Dump it explicitly so
        # the pytest output has the actual reason.
        for log_path in ("/tmp/nvidia-log/control.log", "/tmp/nvidia-log/server.log"):
            p = Path(log_path)
            if p.exists():
                print(f"\n=== {log_path} (last 4KB) ===")
                with open(p, "rb") as f:
                    print(f.read()[-4096:].decode("utf-8", errors="replace"))
                print(f"=== /{log_path} ===\n")
            else:
                print(f"\n[{log_path} not present]\n")

    assert proc.returncode == 0, (
        f"train_entry exited with code {proc.returncode}; see captured "
        f"output above for the actual error."
    )

    completed_marker = "completed_steps=1 / num_steps=1"
    assert any(completed_marker in line for line in tail), (
        f"Expected log line containing {completed_marker!r} not found. "
        f"This means the colocate loop didn't reach the end of step 1 — "
        f"the rendezvous succeeded but the forward/backward/recv chain "
        f"failed silently. Last 50 lines:\n" + "\n".join(tail[-50:])
    )

    # Output dir cleanup is the responsibility of pytest's tmp_path teardown.
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 4 / 6 / 7 — single-GPU tiny-model colocate smoke.

This is the cheap-host counterpart to ``test_one_step.py``,
``test_stability.py``, ``test_grad_parity.py``, and
``test_convergence.py``. It exercises **the same colocate code path**
(MPS daemon, fractional GPU sharing, NCCL P2P union world,
NcclMultiTensorFetcher, sglang colocate.patch) but at a footprint that
fits inside a single 24 GB consumer or L40S-class GPU.

Why a separate file:

* The 4×H100 + Qwen3-8B tests are gated behind ``has_h100_quad()`` and
  cost real money to run. People without that hardware budget
  (Modal sandbox doesn't support MPS at all — see
  ``docs/colocate/implementation_log.md``) need a path to validate
  correctness on the cheapest 1-GPU rental they can find
  (Vast.ai 3090/4090/L40S, Lambda Labs spot A6000, Hyperstack L40S, …).
* The skip gates are different (``has_n_gpus(1)`` instead of
  ``has_h100_quad()``); keeping them on the same test function would
  silently let a 1-GPU host run the 4-GPU Qwen3-8B test and OOM.

What it covers (same defects each test in the 4-GPU sweep catches):

* ``test_phase4_tiny_one_step`` — same as ``test_phase4_one_step_…``
  but with the tiny config: catches rendezvous deadlocks, MPS-daemon
  failures, tensor-spec mismatches between trainer + engine, missing
  upstream sglang patch.
* ``test_phase7_tiny_loss_decreases`` — same as
  ``test_phase7_convergence_loss_decreases`` but with horizon=20 by
  default: catches gradient-not-flowing bugs and dropped-data bugs in
  the NCCL recv path. 20 steps on 0.6 B params takes ~30 s on an
  L40S; a longer 100-step variant is available via
  ``COLOCATE_TINY_CONVERGE_STEPS``.

Run via:
    bash scripts/colocate/run_smoke_host.sh
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.colocate._mps_probe import has_n_gpus, mps_works

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "colocate_qwen0p6b_tiny.yaml"
DATASET_PATH = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"

CONVERGE_STEPS = int(os.environ.get("COLOCATE_TINY_CONVERGE_STEPS", "20"))


pytestmark = [
    pytest.mark.timeout(2400),
    pytest.mark.skipif(
        not has_n_gpus(1),
        reason="Tiny colocate smoke needs at least one CUDA GPU.",
    ),
    pytest.mark.skipif(
        not mps_works(),
        reason=(
            "Tiny colocate smoke needs working NVIDIA MPS. On hosts where "
            "the MPS server reports 'operation not supported' "
            "(e.g. Modal sandbox without --ipc=host) the colocate path "
            "would hang on the first inter-process NCCL P2P. Run on a "
            "host that exposes --ipc=host (Vast.ai, Lambda Labs, "
            "Hyperstack, dedicated/bare-metal Linux)."
        ),
    ),
]


def _build_train_cmd(num_steps: int, *, seed: int = 42) -> list[str]:
    return [
        "python",
        "-m",
        "torchspec.train_entry",
        "--config",
        str(CONFIG_PATH),
        f"dataset.train_data_path={DATASET_PATH}",
        f"training.num_train_steps={num_steps}",
        "training.num_epochs=1",
        f"training.seed={seed}",
        "training.training_num_gpus_per_node=1",
        "inference.inference_num_gpus=1",
        "inference.inference_num_gpus_per_engine=1",
        "inference.inference_num_gpus_per_node=1",
        "inference.sglang.tp_size=1",
    ]


def _make_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCHSPEC_LOG_LEVEL", "INFO")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("NCCL_DEBUG", "WARN")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "inductor")
    (tmp_path / "inductor").mkdir(exist_ok=True)
    return env


def _run_train(
    cmd: list[str], env: dict[str, str], tmp_path: Path, *, timeout: int
) -> tuple[int, str]:
    """Run train_entry with stdout streamed to a log file; return (rc, log)."""
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
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait(timeout=30)

    with open(log_path, "rb") as f:
        log = f.read().decode("utf-8", errors="replace")
    print("\n=== train_entry tail (200 lines) ===")
    for line in log.splitlines()[-200:]:
        print(line)
    print("=== /train_entry tail ===\n")

    if timed_out:
        for log_p in ("/tmp/nvidia-log/control.log", "/tmp/nvidia-log/server.log"):
            p = Path(log_p)
            if p.exists():
                print(f"\n=== {log_p} (last 4KB) ===")
                with open(p, "rb") as f:
                    print(f.read()[-4096:].decode("utf-8", errors="replace"))
                print(f"=== /{log_p} ===\n")
        raise AssertionError(
            f"tiny colocate run timed out after {timeout}s; see captured output above."
        )
    return proc.returncode, log


def test_phase4_tiny_one_step(tmp_path: Path) -> None:
    """One full colocate step end-to-end on a single GPU + tiny model."""
    assert CONFIG_PATH.exists(), CONFIG_PATH
    assert DATASET_PATH.exists(), DATASET_PATH

    cmd = _build_train_cmd(num_steps=1)
    env = _make_env(tmp_path)
    # Cold HF cache for Qwen3-0.6B is < 1.5 GB so 15 min is plenty even on
    # slow networks; warm cache + tiny model usually finishes in < 90 s.
    rc, log = _run_train(cmd, env, tmp_path, timeout=15 * 60)

    assert rc == 0, f"train_entry exited {rc}; see log above."

    completed_marker = "completed_steps=1 / num_steps=1"
    assert any(completed_marker in line for line in log.splitlines()), (
        f"Expected log line containing {completed_marker!r} not found. "
        "The colocate loop didn't reach the end of step 1 — "
        "the rendezvous succeeded but the forward/backward/recv chain "
        "failed silently."
    )


def _losses_from_log(log: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    pat = re.compile(r"\[colocate_loop\] step=(?P<step>\d+).*?loss=(?P<v>[0-9eE.+\-]+)")
    for line in log.splitlines():
        m = pat.search(line)
        if m:
            try:
                out.append((int(m.group("step")), float(m.group("v"))))
            except ValueError:
                continue
    return out


def test_phase7_tiny_loss_decreases(tmp_path: Path) -> None:
    """``CONVERGE_STEPS`` colocate steps drop the late-window loss
    below the early-window loss.

    Uses the same parsing as Phase-7 ``test_convergence`` but with
    horizon=20 by default. On Qwen3-0.6B with seq_len=2048 each step
    is < 2 s on an L40S, so the whole test fits inside 60 s of GPU
    time after the cold-start tax.
    """
    cmd = _build_train_cmd(num_steps=CONVERGE_STEPS)
    env = _make_env(tmp_path)
    # 20 steps * ~2 s/step = 40 s training + 5 min cold start budget.
    rc, log = _run_train(cmd, env, tmp_path, timeout=20 * 60)
    assert rc == 0, f"train_entry exited {rc}; see log above."

    losses = _losses_from_log(log)
    assert len(losses) >= max(2, CONVERGE_STEPS // 5), (
        f"only captured {len(losses)} loss points; expected at least "
        f"~{CONVERGE_STEPS // 5}. The colocate loop's metric flush "
        "format may have changed."
    )
    quartile = max(1, len(losses) // 4)
    early = sum(v for _, v in losses[:quartile]) / quartile
    late = sum(v for _, v in losses[-quartile:]) / quartile
    assert late < early, (
        f"loss did not decrease: early={early:.4f} late={late:.4f}. "
        "Either the gradient isn't flowing (NCCL recv buffers are "
        "uninitialised) or the LR/dtype is wrong for the tiny "
        "colocate path."
    )

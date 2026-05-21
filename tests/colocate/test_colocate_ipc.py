# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Colocate CUDA IPC transport — end-to-end.

CUDA IPC is the **default** colocate hidden-state transport: the engine
exports a CUDA IPC handle per tensor and the trainer maps the memory
directly (one on-device D->D copy, no host round-trip). The fallback is
the gloo CPU-staged transport (engine D->H, gloo ship, trainer H->D),
selected with ``TORCHSPEC_COLOCATE_IPC=0``.

This test runs the colocate tiny config with ``TORCHSPEC_COLOCATE_IPC=1``
(explicit, though it is also the default) and asserts the run completes
with a sane, decreasing loss. Because the IPC path is fail-fast (the
connector/fetcher raise at construction if ``probe_ipc_capability`` says
IPC is unusable — never a silent fallback to gloo), a successful
completion means the IPC transport actually carried every step's hidden
states.

When IPC is on the colocate path skips the ``expandable_segments``
allocator config (IPC's classic capability-free handle path needs plain
``cudaMalloc`` memory — see ``torchspec/colocate/cuda_ipc.py``), so this
test deliberately does **not** export it.

Needs 1 GPU + working MPS.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.colocate._mps_probe import has_n_gpus, mps_works

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.timeout(50 * 60),
    pytest.mark.skipif(not has_n_gpus(1), reason="colocate IPC test needs >=1 GPU."),
    pytest.mark.skipif(not mps_works(), reason="colocate IPC test needs working NVIDIA MPS."),
]

_NUM_STEPS = 5


def _losses(log: str) -> list[float]:
    """Parse the per-step losses from the colocate-loop output."""
    out: list[float] = []
    pat = re.compile(r"\[colocate_loop\] step=\d+.*?loss=(?P<v>[0-9eE.+\-]+)")
    for line in log.splitlines():
        m = pat.search(line)
        if m:
            try:
                out.append(float(m.group("v")))
            except ValueError:
                pass
    return out


def test_colocate_ipc_transport_end_to_end():
    """A colocate run with TORCHSPEC_COLOCATE_IPC=1 completes via CUDA IPC."""
    config_path = REPO_ROOT / "configs" / "colocate_qwen0p6b_tiny.yaml"
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"
    out_dir = Path(tempfile.mkdtemp(prefix="coloipc-"))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Select the CUDA IPC transport explicitly (it is also the default).
    # Deliberately do NOT set expandable_segments — the colocate path
    # drops it for IPC mode so the classic capability-free handle path
    # is used.
    env["TORCHSPEC_COLOCATE_IPC"] = "1"
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTORCH_ALLOC_CONF", None)

    proc = subprocess.run(
        [
            "python",
            "-m",
            "torchspec.train_entry",
            "--config",
            str(config_path),
            f"dataset.train_data_path={dataset}",
            f"training.num_train_steps={_NUM_STEPS}",
            "training.num_epochs=1",
            f"output_dir={out_dir}",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=45 * 60,
    )
    log = proc.stdout + proc.stderr
    print("\n=== colocate-IPC run tail ===")
    for line in log.splitlines()[-60:]:
        print(line)
    print("=== /colocate-IPC run tail ===\n")

    assert proc.returncode == 0, f"colocate IPC run exited {proc.returncode}; see log above."
    assert f"completed_steps={_NUM_STEPS}" in log, (
        f"colocate IPC run did not finish all {_NUM_STEPS} steps."
    )

    losses = _losses(log)
    assert len(losses) >= _NUM_STEPS, f"expected >={_NUM_STEPS} loss points, got {losses}"
    for i, v in enumerate(losses):
        assert v == v and 0.0 < abs(v) < 1e6, (
            f"colocate IPC loss at step {i + 1} is suspect: {v!r} "
            f"(NaN/inf or 0/huge => IPC transport delivered bad data)"
        )
    # Loss should trend down — proves real hidden states crossed the IPC
    # plane and gradients flowed.
    assert losses[-1] < losses[0], (
        f"colocate IPC loss did not decrease ({losses[0]:.3f} -> "
        f"{losses[-1]:.3f}); the IPC transport may be delivering stale data."
    )
    print(f"[colocate-ipc] OK: {len(losses)} steps, loss {losses[0]:.3f} -> {losses[-1]:.3f}")

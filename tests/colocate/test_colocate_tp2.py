# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Multi-engine TP (`engine_tp_size > 1`) — end-to-end.

The colocate path normally runs one engine per trainer
(`engine_tp_size == 1`). This test exercises the multi-engine TP data
plane: a single inference engine running tensor-parallel across 2 GPUs
(`tp_size=2`), paired with 2 trainer ranks.

Per step, the engine's `generate()` carries a 2-prompt batch; both TP
scheduler subprocesses process it, and TP rank ``t`` NCCL-sends batch
item ``t`` to trainer ``t`` (the ``_send_hidden_states_to_nccl``
batch-index gate in ``colocate.patch``). If the per-TP-rank dispatch is
wrong, the run hangs on the first recv or a trainer trains on the wrong
hidden states.

This runs the colocate tp2 tiny config and asserts the loop completes
all steps with a finite, decreasing loss. Needs 2 GPUs + working MPS.
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
    pytest.mark.skipif(
        not has_n_gpus(2),
        reason="multi-engine TP test needs >=2 GPUs (engine_tp_size=2).",
    ),
    pytest.mark.skipif(not mps_works(), reason="multi-engine TP test needs working NVIDIA MPS."),
]

_NUM_STEPS = 5


def _losses(log: str) -> list[float]:
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


def test_colocate_engine_tp2_end_to_end():
    """A colocate run with engine_tp_size=2 completes with sane loss."""
    config_path = REPO_ROOT / "configs" / "colocate_qwen0p6b_tp2_tiny.yaml"
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"
    out_dir = Path(tempfile.mkdtemp(prefix="colotp2-"))

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env["CUDA_VISIBLE_DEVICES"] = "0,1"

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
    print("\n=== colocate-tp2 run tail ===")
    for line in log.splitlines()[-60:]:
        print(line)
    print("=== /colocate-tp2 run tail ===\n")

    assert proc.returncode == 0, (
        f"colocate engine_tp_size=2 run exited {proc.returncode}; see log above."
    )
    assert f"completed_steps={_NUM_STEPS}" in log, (
        f"colocate tp2 run did not finish all {_NUM_STEPS} steps "
        f"(hang in the per-TP-rank hidden-state dispatch?)."
    )

    losses = _losses(log)
    assert len(losses) >= _NUM_STEPS, f"expected >={_NUM_STEPS} loss points, got {losses}"
    for i, v in enumerate(losses):
        assert v == v and 0.0 < abs(v) < 1e6, (
            f"colocate tp2 loss at step {i + 1} is suspect: {v!r} "
            f"(a TP rank may be sending/receiving the wrong batch item)."
        )
    assert losses[-1] < losses[0], (
        f"colocate tp2 loss did not decrease ({losses[0]:.3f} -> {losses[-1]:.3f})."
    )
    print(f"[colocate-tp2] OK: {len(losses)} steps, loss {losses[0]:.3f} -> {losses[-1]:.3f}")

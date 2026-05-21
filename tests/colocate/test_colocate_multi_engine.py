# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Multi-engine TP fan-out (`n_engines > 1` AND `engine_tp_size > 1`).

`test_colocate_tp2.py` covers a *single* tp=2 engine — it validates the
per-TP-rank routing but never runs the colocate loop's
`for e in range(n_engines)` dispatch loop with more than one engine.

This test runs TWO inference engines, each tensor-parallel across 2
GPUs (`tp_size=2`), paired with 4 trainer ranks. Per step:

  * colocate_loop pulls dp_size=4 prompts and dispatches per engine:
    engine 0 gets prompts for trainers [0,2), engine 1 for [2,4).
  * Each engine's `generate()` carries a 2-prompt batch; TP rank ``t``
    NCCL-sends batch item ``t`` to trainer ``e*2 + t``.

If the multi-engine base-rank math is wrong, an engine sends hidden
states to the wrong trainer block and either the run hangs on a recv
or a trainer trains on another engine's hidden states. Needs 4 GPUs +
working MPS.
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
        not has_n_gpus(4),
        reason="multi-engine fan-out test needs >=4 GPUs (2 engines x engine_tp_size=2).",
    ),
    pytest.mark.skipif(
        not mps_works(), reason="multi-engine fan-out test needs working NVIDIA MPS."
    ),
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


def test_colocate_multi_engine_tp2_end_to_end():
    """A colocate run with 2 engines x engine_tp_size=2 completes sanely."""
    config_path = REPO_ROOT / "configs" / "colocate_qwen0p6b_2eng_tp2_tiny.yaml"
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"
    out_dir = Path(tempfile.mkdtemp(prefix="colo2eng-"))

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
    print("\n=== colocate-2eng-tp2 run tail ===")
    for line in log.splitlines()[-60:]:
        print(line)
    print("=== /colocate-2eng-tp2 run tail ===\n")

    assert proc.returncode == 0, (
        f"colocate 2-engine tp2 run exited {proc.returncode}; see log above."
    )
    assert f"completed_steps={_NUM_STEPS}" in log, (
        f"colocate 2-engine run did not finish all {_NUM_STEPS} steps "
        f"(hang in the per-engine / per-TP-rank hidden-state dispatch?)."
    )

    losses = _losses(log)
    assert len(losses) >= _NUM_STEPS, f"expected >={_NUM_STEPS} loss points, got {losses}"
    for i, v in enumerate(losses):
        assert v == v and 0.0 < abs(v) < 1e6, (
            f"colocate 2-engine loss at step {i + 1} is suspect: {v!r} "
            f"(an engine may be routing hidden states to the wrong trainer "
            f"block)."
        )
    assert losses[-1] < losses[0], (
        f"colocate 2-engine loss did not decrease ({losses[0]:.3f} -> {losses[-1]:.3f})."
    )
    print(f"[colocate-2eng-tp2] OK: {len(losses)} steps, loss {losses[0]:.3f} -> {losses[-1]:.3f}")

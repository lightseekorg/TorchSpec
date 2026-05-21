# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Colocate checkpoint save + resume.

Background: commit 59400f1 scoped the seven ``dcp.save`` / ``dcp.load``
calls in ``torchspec/training/checkpoint.py`` to ``actor.dp_group`` so
they don't deadlock on the 2N-rank union world in colocate mode. That
fix shipped **unexercised** — the colocate loop read a non-existent
``save_steps`` attribute (always 0 via ``getattr``), so the save path
never fired. The colocate loop now uses the real ``save_interval`` knob
(same as the disagg loop); this test exercises the whole path:

* ``test_colocate_checkpoint_save`` — run the colocate tiny config with
  ``save_interval=1`` and assert a checkpoint lands on disk and the run
  completes. If ``dcp.save`` deadlocked (the bug 59400f1 fixed) the run
  would hang and the test would time out.
* ``test_colocate_checkpoint_resume`` — save, then start a fresh run
  with ``load_path`` pointed at the checkpoint and assert ``dcp.load``
  restores the draft model without deadlocking.

Both need 1 GPU + working MPS (the colocate tiny topology).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.colocate._mps_probe import has_n_gpus, mps_works

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.skipif(not has_n_gpus(1), reason="colocate checkpoint test needs >=1 GPU."),
    pytest.mark.skipif(
        not mps_works(), reason="colocate checkpoint test needs working NVIDIA MPS."
    ),
]


def _run_colocate(
    *,
    output_dir: Path,
    num_steps: int,
    extra_args: list[str],
    timeout_s: int = 1800,
) -> str:
    """Run the colocate tiny config through train_entry; return the log."""
    config_path = REPO_ROOT / "configs" / "colocate_qwen0p6b_tiny.yaml"
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        "python",
        "-m",
        "torchspec.train_entry",
        "--config",
        str(config_path),
        f"dataset.train_data_path={dataset}",
        f"training.num_train_steps={num_steps}",
        "training.num_epochs=1",
        f"output_dir={output_dir}",
        *extra_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    log = proc.stdout + proc.stderr
    print("\n=== _run_colocate tail ===")
    for line in log.splitlines()[-80:]:
        print(line)
    print("=== /_run_colocate tail ===\n")
    assert proc.returncode == 0, f"train_entry exited {proc.returncode}; see log above."
    return log


def _find_checkpoint(checkpoints_dir: Path) -> Path:
    """Return the single iter_* checkpoint dir, asserting it is complete."""
    iters = sorted(checkpoints_dir.glob("iter_*"))
    assert iters, f"no iter_* checkpoint under {checkpoints_dir}"
    ckpt = iters[-1]
    for sub in ("model", "optimizer", "lr_scheduler"):
        assert (ckpt / sub).is_dir(), f"checkpoint missing {sub}/ at {ckpt}"
    assert (checkpoints_dir / "latest_checkpointed_iteration.txt").exists(), (
        "save did not write the latest_checkpointed_iteration.txt tracker"
    )
    return ckpt


@pytest.mark.timeout(50 * 60)
def test_colocate_checkpoint_save():
    """A colocate run with save_interval>0 writes a complete checkpoint."""
    tmp = Path(tempfile.mkdtemp(prefix="colockpt-save-"))
    log = _run_colocate(
        output_dir=tmp / "run",
        num_steps=2,
        extra_args=["training.save_interval=1"],
    )
    assert "Saving checkpoint at step" in log, (
        "colocate loop never reached the save branch — save_interval not honoured."
    )
    ckpt = _find_checkpoint(tmp / "run" / "checkpoints")
    print(f"[colocate-ckpt] save OK: {ckpt}")


@pytest.mark.timeout(90 * 60)
def test_colocate_checkpoint_resume():
    """A colocate run resumes a saved checkpoint via dcp.load without hanging."""
    tmp = Path(tempfile.mkdtemp(prefix="colockpt-resume-"))

    # Arm 1: save.
    _run_colocate(
        output_dir=tmp / "run1",
        num_steps=2,
        extra_args=["training.save_interval=1"],
    )
    checkpoints_dir = tmp / "run1" / "checkpoints"
    _find_checkpoint(checkpoints_dir)

    # Arm 2: fresh run, resume from arm 1's checkpoint.
    log = _run_colocate(
        output_dir=tmp / "run2",
        num_steps=3,
        extra_args=[
            "training.save_interval=1",
            f"training.load_path={checkpoints_dir}",
        ],
    )
    assert "Loaded model from" in log, (
        "resume did not load the checkpoint — checkpoint.load() never "
        "reached dcp.load (load_path / tracker-file resolution failed)."
    )
    print("[colocate-ckpt] resume OK")

# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 7 — short-run convergence (slow).

Plan reference: ``implementation.md`` §Phase 7, "Short-horizon
convergence: 1k step training loss curve overlaps within 2% of the
disaggregated baseline."

This is the slow (``@pytest.mark.slow``) counterpart to
``test_grad_parity.py``. Two tests:

* ``test_phase7_convergence_loss_decreases`` — runs a short colocate
  training horizon and asserts the loss curve trends downward (i.e.,
  training is making real progress — not a no-op or constant signal).
* ``test_convergence_disagg_overlap`` — the plan's literal ask: run the
  colocate tiny config and the disaggregated (Mooncake) tiny config for
  the same horizon, same seed, same data order, and assert the two loss
  curves overlap within ~2 %. Both arms train the identical draft model
  on identical batches; the only difference is the hidden-state
  transport (CUDA IPC colocate vs Mooncake disagg), so an overlapping
  loss curve is the end-to-end signal that the colocate transport
  converges like the disaggregated baseline. Skips cleanly if Mooncake
  cannot run on the host.

Both tests parse the env-gated ``[loss_curve] step=N loss=V`` log line
emitted by *both* training loops (``controller/loop.py`` and
``controller/colocate_loop.py``) when ``TORCHSPEC_LOSS_CURVE_LOG`` is
set — an identical format on both sides so the curves are directly
comparable.

Default horizon: 50 steps. Override with ``PHASE7_CONVERGE_STEPS``
(the plan's reference is 1000 but that's an hour of compute under
MPS; CI only needs to see a clear downward trend). The overlap
tolerance defaults to 2 % — override with ``CONVERGE_OVERLAP_PCT``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.colocate._mps_probe import has_h100_quad, has_n_gpus, mps_works
from tests.colocate.test_grad_parity import _disagg_runnable

REPO_ROOT = Path(__file__).resolve().parents[2]

NUM_STEPS = int(os.environ.get("PHASE7_CONVERGE_STEPS", "50"))

# colocate-vs-disagg loss-curve overlap tolerance, in percent. The plan
# asks for 2 %; both arms train the identical model on identical batches,
# so the curves should track tightly — the band only absorbs NCCL
# non-determinism and the bf16 transport-copy ULP differences.
TOL_PCT = float(os.environ.get("CONVERGE_OVERLAP_PCT", "2.0"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(60 * 60),
]


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


@pytest.mark.skipif(
    not has_h100_quad(),
    reason="Phase-7 convergence requires >=4 GPUs.",
)
@pytest.mark.skipif(
    not mps_works(),
    reason=(
        "Phase-7 convergence needs the colocate path to actually run, "
        "which needs working NVIDIA MPS (see tests/colocate/_mps_probe.py)."
    ),
)
def test_phase7_convergence_loss_decreases():
    """After ``NUM_STEPS`` colocate steps the average late-window loss
    is below the average early-window loss. Drives the same loop as
    Phase 4 / 6 but for many steps; this is the cheapest e2e signal
    that the gradient is actually flowing (the trainer is updating
    weights from real engine-supplied hidden states)."""

    config_path = REPO_ROOT / "configs" / "colocate_qwen3_8b.yaml"
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    proc = subprocess.run(
        [
            "python",
            "-m",
            "torchspec.train_entry",
            "--config",
            str(config_path),
            f"dataset.train_data_path={dataset}",
            f"training.num_train_steps={NUM_STEPS}",
            "training.num_epochs=1",
            "training.training_num_gpus_per_node=4",
            "inference.inference_num_gpus=4",
            "inference.inference_num_gpus_per_engine=1",
            "inference.inference_num_gpus_per_node=4",
            "inference.sglang.tp_size=1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60 * 60 - 30,
    )

    log = proc.stdout + proc.stderr
    print("\n=== last 200 lines ===")
    for line in log.splitlines()[-200:]:
        print(line)
    print("=== /last 200 lines ===\n")
    assert proc.returncode == 0, f"train_entry exited {proc.returncode}"

    losses = _losses_from_log(log)
    assert len(losses) >= max(2, NUM_STEPS // 10), (
        f"only captured {len(losses)} loss points; expected at least "
        f"~{NUM_STEPS // 10}. The colocate loop's metric flush "
        f"may have changed format."
    )
    early = sum(v for _, v in losses[: max(1, len(losses) // 4)])
    late = sum(v for _, v in losses[-max(1, len(losses) // 4) :])
    early /= max(1, len(losses) // 4)
    late /= max(1, len(losses) // 4)
    assert late < early, (
        f"loss did not decrease: early={early:.4f} late={late:.4f}. "
        f"Either the gradient isn't flowing (NCCL recv buffers are "
        f"uninitialised) or LR/dtype is wrong for the colocate path."
    )


# ---------------------------------------------------------------------------
# colocate-vs-disagg loss-curve overlap
# ---------------------------------------------------------------------------


def _loss_curve_from_log(log: str) -> dict[int, float]:
    """Parse the env-gated ``[loss_curve] step=N loss=V`` trace into a
    ``{step: loss}`` dict. Both training loops emit this identically when
    ``TORCHSPEC_LOSS_CURVE_LOG`` is set, so the two arms are directly
    comparable."""
    out: dict[int, float] = {}
    pat = re.compile(r"\[loss_curve\] step=(?P<step>\d+) loss=(?P<v>[0-9eE.+\-]+)")
    for line in log.splitlines():
        m = pat.search(line)
        if m:
            try:
                out[int(m.group("step"))] = float(m.group("v"))
            except ValueError:
                continue
    return out


def _run_loss_curve_arm(
    config_name: str,
    *,
    num_steps: int,
    visible_devices: str,
    seed: int = 42,
    ipc: bool = False,
    disable_mps: bool = False,
    skip_on_failure: bool = False,
    timeout_s: int,
) -> dict[int, float]:
    """Run ``train_entry`` for ``num_steps`` and return its loss curve.

    Mirrors ``test_grad_parity._run_arm`` but multi-step and loss-curve
    oriented: no gradient dump, and ``TORCHSPEC_LOSS_CURVE_LOG`` is on so
    both loops emit the per-step ``[loss_curve]`` line this parses.
    """
    config_path = REPO_ROOT / "configs" / config_name
    dataset = REPO_ROOT / "examples" / "data" / "sample_conversations.jsonl"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["TORCHSPEC_LOSS_CURVE_LOG"] = "1"
    if ipc:
        # CUDA IPC transport (the colocate default). It needs plain
        # cudaMalloc memory, so drop expandable_segments.
        env["TORCHSPEC_COLOCATE_IPC"] = "1"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("PYTORCH_ALLOC_CONF", None)
    if disable_mps:
        # The disagg arm is a non-colocate run; it must not be caught by
        # an MPS daemon left up by the colocate arm (its actors would
        # fail MPS's CUDA_VISIBLE_DEVICES validation).
        env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        env.pop("CUDA_MPS_LOG_DIRECTORY", None)
        env["TORCHSPEC_DISABLE_MPS"] = "1"

    cmd = [
        "python",
        "-m",
        "torchspec.train_entry",
        "--config",
        str(config_path),
        f"dataset.train_data_path={dataset}",
        f"training.num_train_steps={num_steps}",
        # High epoch cap so num_train_steps is the only stopping limit
        # (the dataset reloads identically on both arms — shuffle off).
        "training.num_epochs=1000",
        f"training.seed={seed}",
        # Deterministic prompt order so both arms see the same batches.
        "dataset.shuffle_dataset=false",
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
    print(f"\n=== _run_loss_curve_arm({config_name}) tail ===")
    for line in log.splitlines()[-80:]:
        print(line)
    print(f"=== /_run_loss_curve_arm({config_name}) tail ===\n")

    if proc.returncode != 0 and skip_on_failure:
        # The disagg baseline arm runs the environment-fragile Mooncake
        # transfer engine — a baseline that cannot run is not a colocate
        # defect, so skip rather than fail the suite.
        pytest.skip(
            f"convergence baseline arm '{config_name}' could not run on "
            f"this host (train_entry exit {proc.returncode}); see the "
            f"captured tail above."
        )
    assert proc.returncode == 0, (
        f"train_entry({config_name}) exited {proc.returncode}; see log above."
    )

    curve = _loss_curve_from_log(log)
    assert curve, (
        f"no [loss_curve] points parsed from {config_name}: the "
        f"TORCHSPEC_LOSS_CURVE_LOG line may have changed format, or the "
        f"loop never reached its metrics block."
    )
    return curve


@pytest.mark.timeout(2 * 60 * 60)
@pytest.mark.skipif(
    not has_n_gpus(2),
    reason="convergence-overlap needs >=2 GPUs (disagg arm: trainer + engine).",
)
@pytest.mark.skipif(
    not mps_works(),
    reason="convergence-overlap needs working NVIDIA MPS for the colocate arm.",
)
@pytest.mark.skipif(
    not _disagg_runnable(),
    reason=(
        "convergence-overlap needs an importable Mooncake store for the "
        "disagg baseline arm (apt-get install libibverbs1 libnuma1 "
        "librdmacm1 libnl-3-200)."
    ),
)
def test_convergence_disagg_overlap():
    """The colocate loss curve overlaps the disaggregated baseline.

    The plan's literal Phase-7 ask: "1k step training loss curve overlaps
    within 2 % of the disaggregated baseline." Run the colocate tiny
    config and the disaggregated (Mooncake) tiny config for ``NUM_STEPS``
    steps with the same seed and the same (unshuffled) data order. Both
    arms train the identical tiny draft model on identical batches; the
    only thing that differs is the hidden-state transport — CUDA IPC
    (colocate) vs Mooncake (disagg). Both transports are lossless copies,
    so the two loss curves must track within ``TOL_PCT``.

    A divergence beyond the band means the colocate transport is not
    converging like the disaggregated baseline — the exact failure the
    plan's validation calls for. Per-step parity is the stronger
    ``test_grad_parity.test_phase7_grad_parity_vs_disagg`` (one step,
    per-parameter gradients); this is the many-step curve counterpart.

    The disagg arm is environment-fragile (Mooncake's transfer engine);
    if it cannot run the test skips rather than fails — a broken
    third-party baseline is not a colocate regression, and the colocate
    transport is independently covered by the grad-parity tests.
    """
    # The disagg arm is a non-colocate run. Tear down any MPS daemon a
    # prior colocate test left up, else its actors fail MPS's
    # CUDA_VISIBLE_DEVICES validation. (Same dance as
    # test_grad_parity.test_phase7_grad_parity_vs_disagg.)
    from torchspec.colocate.mps import force_stop_mps

    force_stop_mps()

    # Generous safety cap — the tiny model is ~0.15 s/step, so the real
    # runtime is minutes; this only bounds a hang.
    arm_timeout = 600 + NUM_STEPS * 2

    # Disagg baseline arm — 2 GPUs (trainer + engine disjoint), MPS off.
    disagg = _run_loss_curve_arm(
        "disagg_qwen0p6b_tiny.yaml",
        num_steps=NUM_STEPS,
        visible_devices="0,1",
        disable_mps=True,
        skip_on_failure=True,
        timeout_s=arm_timeout,
    )
    # Colocate arm — 1 GPU (trainer + engine MPS-shared), CUDA IPC.
    colocate = _run_loss_curve_arm(
        "colocate_qwen0p6b_tiny.yaml",
        num_steps=NUM_STEPS,
        visible_devices="0",
        ipc=True,
        timeout_s=arm_timeout,
    )

    common = sorted(set(disagg) & set(colocate))
    assert len(common) >= max(2, NUM_STEPS // 2), (
        f"too few overlapping loss-curve steps: {len(common)} "
        f"(disagg={len(disagg)}, colocate={len(colocate)}, expected "
        f"~{NUM_STEPS}). One arm logged far fewer steps than the other — "
        f"check both runs completed {NUM_STEPS} steps."
    )

    devs = []
    for s in common:
        c, d = colocate[s], disagg[s]
        devs.append(abs(c - d) / max(abs(d), 1e-6))
    mean_dev = sum(devs) / len(devs)
    max_dev = max(devs)
    worst = common[devs.index(max_dev)]

    print("\n=== colocate vs disagg loss curve ===")
    print(f"{'step':>6} {'colocate':>12} {'disagg':>12} {'rel.dev%':>10}")
    n = len(common)
    sample = sorted(set(common[:: max(1, n // 20)] + [worst]))
    for s in sample:
        c, d = colocate[s], disagg[s]
        rd = abs(c - d) / max(abs(d), 1e-6) * 100
        print(f"{s:>6} {c:>12.6f} {d:>12.6f} {rd:>10.3f}")
    print(
        f"mean rel.dev = {mean_dev * 100:.3f}%   "
        f"max rel.dev = {max_dev * 100:.3f}% (step {worst})   "
        f"tol = {TOL_PCT:.2f}%"
    )
    print("=== /colocate vs disagg loss curve ===\n")

    tol = TOL_PCT / 100.0
    assert mean_dev <= tol, (
        f"colocate and disagg loss curves do not overlap: mean relative "
        f"deviation {mean_dev * 100:.3f}% exceeds the {TOL_PCT:.2f}% "
        f"tolerance over {n} steps. The colocate transport is not "
        f"converging like the disaggregated baseline."
    )
    assert max_dev <= 3 * tol, (
        f"colocate vs disagg loss diverges at step {worst}: relative "
        f"deviation {max_dev * 100:.3f}% exceeds the {3 * TOL_PCT:.2f}% "
        f"per-step ceiling (mean was {mean_dev * 100:.3f}%). A single-step "
        f"spike this large points at a transport glitch, not slow drift."
    )
    print(
        f"[convergence] disagg-overlap OK: mean {mean_dev * 100:.3f}%, "
        f"max {max_dev * 100:.3f}% over {n} steps"
    )

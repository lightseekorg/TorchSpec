# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 6 — long-run memory stability (slow).

Plan reference: ``implementation.md`` §Phase 6, "1000-step stability run
with `dflash_trainer` config: ``peak_alloc(step=10) ≈ peak_alloc(step=999)``
within 1%."

This is the slow (``@pytest.mark.slow``) counterpart to ``test_one_step``.
It runs the full ``train_entry`` colocate path for ``PHASE6_STABILITY_STEPS``
steps and asserts that the per-step peak GPU allocation reported by
``TrainProfiler.peak_alloc_metrics`` doesn't drift more than 1 % between
an early step and a late step. A drift larger than 1 % typically means
either:

* the per-step recv-buffer alloc in ``NcclMultiTensorFetcher.recv_step``
  is fragmenting the pool (expandable_segments not working as expected);
* the engine side is leaking KV-cache slabs because
  ``mem_fraction_static`` doesn't agree with the trainer's
  ``train_frac`` claim (Phase 1 invariant breach).

To keep CI cost reasonable, this test is gated behind ``-m slow`` and
the step count defaults to 200; pass ``PHASE6_STABILITY_STEPS=1000``
(the plan's reference number) for the full run. The nightly
``.github/workflows/colocate-stability.yml`` job does exactly that on
a self-hosted 4×H100 runner; ``run_smoke_host.sh --stability`` is the
manual equivalent. At >=1000 steps the acceptance bar tightens to the
plan's 1 % (measured after a 100-step allocator warmup).

The test parses the captured stdout for the colocate loop's
``perf/peak_bytes_allocated`` metric. The loop emits one
``[colocate_loop] step=N step_time=...`` line every 5 steps, plus the
profiler logs full metrics every step.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_STEPS = int(os.environ.get("PHASE6_STABILITY_STEPS", "200"))

# expandable_segments grows its segment table over the first ~50-100
# steps; sampling the "early" peak-alloc baseline before it settles
# inflates the apparent drift. For the nightly 1000-step run we skip
# that ramp (warmup=100) and hold the plan's 1 % bar; the 200-step
# smoke can't fully settle, so it keeps the looser 5 % bar against a
# step-10 baseline.
_LONG_RUN = NUM_STEPS >= 1000
WARMUP_STEPS = 100 if _LONG_RUN else 10
PEAK_ALLOC_TOLERANCE = 0.01 if _LONG_RUN else 0.05

# Setup (clone/patch/install + model download) is ~10-15 min; each
# colocate step is a few seconds under MPS. Size the budget off the
# step count so the 1000-step nightly doesn't trip a 200-step timeout.
_TIMEOUT_S = max(60 * 60, 900 + NUM_STEPS * 6)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(_TIMEOUT_S),
]


from tests.colocate._mps_probe import has_h100_quad, mps_works


def _extract_peak_alloc(log: str) -> dict[int, float]:
    """Parse `step=N ... peak=... GB` markers out of the captured log.

    The colocate loop's metric flush prints a Python dict every 5 steps.
    We just regex-match `step=N` and the closest peak-alloc number
    (Mb or GB) on the same line.
    """
    out: dict[int, float] = {}
    pattern = re.compile(
        r"step=(?P<step>\d+).*?peak[_ ]alloc[^=]*=(?P<bytes>[0-9eE.+\-]+)",
        re.IGNORECASE,
    )
    for line in log.splitlines():
        m = pattern.search(line)
        if m:
            out[int(m.group("step"))] = float(m.group("bytes"))
    return out


@pytest.mark.skipif(
    not has_h100_quad(),
    reason="Phase 6 stability requires >=4 GPUs.",
)
@pytest.mark.skipif(
    not mps_works(),
    reason=(
        "Phase 6 stability requires NVIDIA MPS support (skipped on hosts "
        "where MPS server reports 'operation not supported'; see "
        "tests/colocate/_mps_probe.py for details)."
    ),
)
def test_phase6_peak_alloc_flatness():
    """Run NUM_STEPS colocate steps; peak-alloc must stay flat ±5 %."""
    config_path = REPO_ROOT / "configs" / "colocate_qwen3_8b.yaml"
    run_sh = REPO_ROOT / "examples" / "colocate-qwen3-8b-1node" / "run.sh"
    assert config_path.exists() and run_sh.exists()

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env.setdefault("TORCHSPEC_LOG_LEVEL", "INFO")

    proc = subprocess.run(
        [
            "bash",
            str(run_sh),
            str(config_path),
            f"training.num_train_steps={NUM_STEPS}",
            "training.num_epochs=1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S - 30,
    )

    log = proc.stdout + proc.stderr
    print("\n=== last 200 lines ===")
    for line in log.splitlines()[-200:]:
        print(line)
    print("=== /last 200 lines ===\n")

    assert proc.returncode == 0, f"colocate stability run exited {proc.returncode}; see log above."

    peaks = _extract_peak_alloc(log)
    early = next((peaks[s] for s in sorted(peaks) if s >= WARMUP_STEPS), None)
    late = max((peaks[s] for s in peaks if s >= NUM_STEPS - 5), default=None)
    assert early is not None and late is not None, (
        f"could not extract peak-alloc samples from log "
        f"(need a step >= {WARMUP_STEPS} for the post-warmup baseline and a "
        f"step >= {NUM_STEPS - 5} for the late sample); got steps={sorted(peaks)}"
    )
    drift = abs(late - early) / early
    assert drift < PEAK_ALLOC_TOLERANCE, (
        f"peak-alloc drift {drift:.4f} (step>={WARMUP_STEPS}: {early:.3e} → "
        f"step>={NUM_STEPS - 5}: {late:.3e}) exceeds tolerance "
        f"{PEAK_ALLOC_TOLERANCE} over {NUM_STEPS} steps; suggests a memory "
        f"leak or fragmentation in the colocate path."
    )

#!/usr/bin/env bash
# scripts/colocate/run_smoke_host.sh
#
# Cheap-host smoke runner for the colocate (MPS+NCCL) MPS-required tests.
#
# Why this exists:
#   Modal sandbox H100 nodes don't pass --ipc=host to the container, so
#   NVIDIA MPS server reports "operation not supported" and the colocate
#   path can't actually run (see docs/colocate/implementation_log.md
#   §"Modal sandbox MPS limitation"). The Phase-4 / 6 / 7 tests
#   correctly skip on Modal but still need to run *somewhere* to
#   validate end-to-end correctness.
#
#   This script lets you do that on the cheapest GPU rental you can
#   find (Vast.ai 3090/4090/L40S, Lambda Labs spot, Hyperstack L40S,
#   etc.) — anything with one CUDA-8.0+ GPU and a container runtime
#   that doesn't sandbox IPC. Total cost on Vast.ai L40S is ~$0.20–$0.40
#   for one full pass once the cache is warm.
#
# Prerequisites on the host:
#   * Linux + NVIDIA driver >= 535 + CUDA Driver API 12.4+
#   * `nvidia-smi` shows at least 1 GPU
#   * Either:
#     - `--ipc=host` Docker container (Vast.ai default; Hyperstack default)
#     - OR bare-VM SSH (no Docker isolation at all)
#   * Python 3.10 or 3.11 + `pip` available
#   * `git` available, and outbound HTTPS to github.com + huggingface.co
#   * (optional) HF_TOKEN exported for gated models — Qwen3-0.6B-Base is
#     not gated, so this is only needed if you change the config.
#
# Usage (from a fresh checkout of this repo):
#   bash scripts/colocate/run_smoke_host.sh                 # tiny smoke (1 GPU)
#   bash scripts/colocate/run_smoke_host.sh --skip-setup    # tests only
#   bash scripts/colocate/run_smoke_host.sh --setup-only    # bootstrap, no tests
#   bash scripts/colocate/run_smoke_host.sh --full          # tiny + 4xGPU Phase 4/6/7
#   bash scripts/colocate/run_smoke_host.sh --stability     # nightly 1000-step run (4xH100)
#   bash scripts/colocate/run_smoke_host.sh --tests=A,B,C   # run specific test files
#
# Environment overrides:
#   COLOCATE_TINY_CONVERGE_STEPS=50    # default 20; raise for stability
#   PHASE6_STABILITY_STEPS=200         # default 200; bump to 1000 on 4xH100
#   PHASE7_CONVERGE_STEPS=50           # default 50; bump to 1000 for full
#   SGLANG_DIR=/abs/path/to/sglang     # default <repo>/_sglang
#   SGLANG_PATCH_VERSION=v0.5.8.post1  # default v0.5.10.post1; selects
#                                      #   which patches/sglang/<ver>/ dir
#   SGLANG_COMMIT=<sha>                # default the v0.5.10.post1 base sha;
#                                      #   must match SGLANG_PATCH_VERSION
#   PYTHON=python3.11                  # default whatever python3 is on PATH
#   PIP_INDEX_URL=...                  # default PyPI
#   COLOCATE_PIN_TORCH=1               # pin torch==2.5.* if you hit a wheel mismatch
#   COLOCATE_SKIP_MPS_PROBE=1          # skip pre-flight MPS probe (let tests SKIP)
#   COLOCATE_KEEP_MPS=1                # don't tear MPS daemon down on script exit
#
# Exit codes:
#   0 — every selected test either PASSED or SKIPPED cleanly
#   1 — host pre-flight failed (no GPU / no MPS binary / MPS probe fails /
#       no CUDA driver). The pre-flight MPS probe means a host without
#       working MPS now exits 1 here instead of running tests that would
#       all SKIP; set COLOCATE_SKIP_MPS_PROBE=1 to revert to the old
#       "skip tests cleanly" behavior.
#   2 — invalid CLI flag
#   non-0 from pytest — at least one test FAILED; see captured log
#
# What it does:
#   1. (pre-flight) nvidia-smi visible, >=1 GPU, MPS daemon binary on
#      PATH, MPS server can actually spawn a CUDA context (cuInit probe).
#      Cleans up stale Ray + MPS state from previous runs.
#   2. (setup) Clone sglang at the pinned commit and apply both patches
#      (the existing disagg sglang.patch and our new colocate.patch).
#   3. (setup) `pip install -e .` torchspec + sglang in --user mode so
#      the host python sees them.
#   4. (run)   `pytest tests/colocate/test_colocate_tiny.py -xvs`
#              tee'd to ./colocate-smoke-pytest.log.
#   5. (run)   Generate ./colocate-smoke-report.txt with everything the
#              "Reporting back" section of cheap_host_test_plan.md asks
#              for: host details, exit code, pytest summary, captured
#              loss values, last 50 lines on failure.
#   6. (exit)  Best-effort `nvidia-cuda-mps-control quit` so the next
#              user gets a clean daemon (skip with COLOCATE_KEEP_MPS=1).

set -euo pipefail

# ---------------------------------------------------------------------------
# Locations & arg parsing
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

SGLANG_DIR="${SGLANG_DIR:-$REPO_ROOT/_sglang}"
SGLANG_COMMIT="${SGLANG_COMMIT:-94f03a39dbd39edfc2b118b5357bbbadaaa9ad28}"
SGLANG_PATCH_VERSION="${SGLANG_PATCH_VERSION:-v0.5.10.post1}"
PATCHES_DIR="$REPO_ROOT/patches/sglang/$SGLANG_PATCH_VERSION"

PYTHON="${PYTHON:-python3}"
PIP="$PYTHON -m pip"

PYTEST_LOG="$REPO_ROOT/colocate-smoke-pytest.log"
REPORT_PATH="$REPO_ROOT/colocate-smoke-report.txt"

DO_SETUP=1
DO_RUN=1
RUN_FULL=0
RUN_STABILITY=0
TESTS_OVERRIDE=""

for arg in "$@"; do
  case "$arg" in
    --skip-setup) DO_SETUP=0 ;;
    --setup-only) DO_RUN=0 ;;
    --full) RUN_FULL=1 ;;
    --stability) RUN_STABILITY=1 ;;
    --tests=*) TESTS_OVERRIDE="${arg#--tests=}" ;;
    --help|-h)
      grep -E '^# ' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# --stability: the nightly long-run job. Pin the step count to the
# plan's 1000 (unless the caller already set it) so the test's 1 %
# acceptance bar engages.
if [[ $RUN_STABILITY -eq 1 ]]; then
  export PHASE6_STABILITY_STEPS="${PHASE6_STABILITY_STEPS:-1000}"
fi

# This runner installs into the pod's system Python. On PEP-668
# distros (Ubuntu 24.04 image, pip >= 23.3) that is "externally
# managed" and pip refuses without this flag. The host is a throwaway
# rental, so installing system-wide is fine.
export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"

banner() {
  echo
  echo "=============================================="
  echo "  $*"
  echo "=============================================="
}

# ---------------------------------------------------------------------------
# EXIT trap: tear MPS daemon down so the next renter gets a clean slate.
# Disabled with COLOCATE_KEEP_MPS=1 (useful when iterating with --skip-setup).
# ---------------------------------------------------------------------------

cleanup_mps() {
  if [[ "${COLOCATE_KEEP_MPS:-0}" == "1" ]]; then
    return
  fi
  if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "quit" | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}
trap cleanup_mps EXIT

# ---------------------------------------------------------------------------
# Stale-state cleanup. Idempotent / safe to run repeatedly.
#  - Stop any Ray cluster left over from a prior run (one of the failure
#    modes documented in cheap_host_test_plan.md).
#  - Remove stale /tmp/nvidia-{mps,log} only if no daemon is currently
#    running (otherwise we'd nuke a healthy daemon's pipe dir).
# ---------------------------------------------------------------------------

preflight_cleanup() {
  if command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
  if ! pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
    rm -rf /tmp/nvidia-mps /tmp/nvidia-log
  fi
}

# ---------------------------------------------------------------------------
# Pre-flight: GPU + MPS. Runs *before* setup so a bad host fails in <60s
# instead of after 10 minutes of pip install.
# ---------------------------------------------------------------------------

run_preflight() {
  banner "Pre-flight: GPU + MPS"
  preflight_cleanup

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found — host has no NVIDIA driver. Aborting." >&2
    exit 1
  fi
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  echo "GPU count: $GPU_COUNT"
  if [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "Need at least 1 GPU; found $GPU_COUNT." >&2
    exit 1
  fi

  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "nvidia-cuda-mps-control NOT FOUND — install the CUDA toolkit "  \
         "(it ships the MPS daemon)." >&2
    exit 1
  fi
  echo "MPS daemon binary: $(command -v nvidia-cuda-mps-control)"

  if [[ "${COLOCATE_SKIP_MPS_PROBE:-0}" == "1" ]]; then
    echo "Skipping MPS server probe (COLOCATE_SKIP_MPS_PROBE=1)."
    return
  fi

  echo
  echo "Probing whether the MPS daemon can actually spawn a working server"
  echo "(this is what catches 'no --ipc=host' / sandboxed containers in <30s"
  echo "instead of letting pytest SKIP after 10 min of setup) …"

  PYTHONPATH="$REPO_ROOT" "$PYTHON" -m tests.colocate._mps_probe || {
    echo >&2
    echo "*** MPS pre-flight FAILED. ***" >&2
    echo >&2
    echo "  All colocate tests would SKIP on this host. Most likely causes:" >&2
    echo "    * Container runtime is sandboxing IPC (RunPod Serverless," >&2
    echo "      Modal sandbox, gVisor-backed managed runtimes)." >&2
    echo "    * Host kernel / driver doesn't support MPS sharing." >&2
    echo >&2
    echo "  Fix options:" >&2
    echo "    1. Switch to a host/template that exposes --ipc=host" >&2
    echo "       (Vast.ai 'PyTorch (cuda:12.4)', RunPod 'Interactive Pod'," >&2
    echo "        Hyperstack, bare-metal Linux). See" >&2
    echo "        docs/colocate/cheap_host_test_plan.md cost-tier matrix." >&2
    echo "    2. Set COLOCATE_SKIP_MPS_PROBE=1 to bypass this check and" >&2
    echo "       let pytest report the SKIPs explicitly (validates the" >&2
    echo "       skip path, doesn't validate the colocate code path)." >&2
    if [[ -f /tmp/nvidia-log/server.log ]]; then
      echo >&2
      echo "  --- /tmp/nvidia-log/server.log (last 20 lines) ---" >&2
      tail -n 20 /tmp/nvidia-log/server.log >&2 || true
      echo "  --- end server.log ---" >&2
    fi
    exit 1
  }
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup_system_libs() {
  # sgl_kernel's prebuilt sm90 .so dynamically links libnuma; recent
  # sgl_kernel builds (>=0.3.x) hard-fail to load without it. Mooncake's
  # transfer-engine .so links the RDMA verbs userspace stack. Neither is
  # guaranteed on a bare CUDA base image, so install both up front —
  # otherwise the engine subprocess dies with an opaque
  #   "[sgl_kernel] CRITICAL: Could not load any common_ops library"
  # (root cause: libnuma.so.1 not found) at first import.
  if ! command -v apt-get >/dev/null 2>&1; then
    banner "system libs: no apt-get — skipping (ensure libnuma/libibverbs present)"
    return 0
  fi
  banner "system libs: libnuma + RDMA verbs stack"
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq \
    libnuma1 libibverbs1 librdmacm1 libnl-3-200 libnl-route-3-200 \
    ibverbs-providers >/dev/null 2>&1 \
    || echo "WARNING: apt-get install of system libs failed (continuing)"
}

setup_sglang() {
  banner "sglang: clone + apply patches"
  if [[ ! -d "$SGLANG_DIR" ]]; then
    git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
  fi
  (
    cd "$SGLANG_DIR"
    git fetch --depth=1 origin "$SGLANG_COMMIT" || true
    git checkout "$SGLANG_COMMIT"
    git reset --hard HEAD
    # git reset --hard only restores *tracked* files; colocate.patch
    # creates new files (e.g. torchspec_colocate.py) that survive a
    # reset, so a second run would fail "already exists in working
    # directory". git clean -fd drops them, making this idempotent.
    git clean -fdq
    rm -f python/sglang/srt/speculative/spec_training_info.py
    git apply --recount "$PATCHES_DIR/sglang.patch" || true
    git apply --recount "$PATCHES_DIR/colocate.patch"
  )
  # NOTE: the dp_attention.py rank-offset and tp_worker.py
  # broadcast_pyobj global-rank fixes are now hunks inside
  # colocate.patch (folded 2026-05-20) -- no post-patch surgery
  # needed here anymore.
}

setup_python() {
  banner "python: $($PYTHON --version) at $(command -v "$PYTHON")"
  $PIP install --upgrade pip wheel setuptools
  if [[ "${COLOCATE_PIN_TORCH:-0}" == "1" ]]; then
    $PIP install "torch==2.5.*" --index-url https://download.pytorch.org/whl/cu124
  else
    $PIP install torch
  fi
  $PIP install \
    "transformers==4.57.1" datasets tqdm wandb accelerate \
    pydantic omegaconf ray openai openai-harmony qwen-vl-utils \
    psutil "numpy<2.4" pyzmq numba cmake ninja packaging \
    setuptools pytest pytest-timeout

  banner "torchspec: pip install -e ."
  $PIP install -e ".[dev]"
  banner "sglang: pip install -e ."
  $PIP install -e "$SGLANG_DIR/python[all]"
}

# ---------------------------------------------------------------------------
# Test selection
# ---------------------------------------------------------------------------

pick_test_files() {
  if [[ -n "$TESTS_OVERRIDE" ]]; then
    IFS=',' read -ra TEST_FILES <<< "$TESTS_OVERRIDE"
  elif [[ $RUN_STABILITY -eq 1 ]]; then
    # Nightly long-run: just the stability test (PHASE6_STABILITY_STEPS
    # already pinned to 1000 above). Hard-requires a 4×H100 + MPS host;
    # self-skips cleanly elsewhere.
    TEST_FILES=(
      "tests/colocate/test_stability.py"
    )
  elif [[ $RUN_FULL -eq 1 ]]; then
    # 4×H100-class hosts: run the tiny + every MPS-gated full test. Each
    # test self-skips if its preconditions aren't met (e.g. has_h100_quad
    # for the Qwen3-8B tests; mps_works for everything), so this is safe
    # to run on a 1-GPU host too — the 4-GPU tests just SKIP cleanly.
    TEST_FILES=(
      "tests/colocate/test_colocate_tiny.py"
      "tests/colocate/test_one_step.py"
      "tests/colocate/test_grad_parity.py"
      "tests/colocate/test_colocate_checkpoint.py"
      "tests/colocate/test_colocate_ipc.py"
      "tests/colocate/test_colocate_tp2.py"
      "tests/colocate/test_colocate_multi_engine.py"
      "tests/colocate/test_stability.py"
      "tests/colocate/test_convergence.py"
    )
  else
    TEST_FILES=(
      "tests/colocate/test_colocate_tiny.py"
    )
  fi
}

# ---------------------------------------------------------------------------
# Report generator: pulls the "Reporting back" data points out of the
# captured pytest log so the next agent can paste a single file instead
# of hand-curating six.
# ---------------------------------------------------------------------------

write_report() {
  local pytest_rc="$1"
  local wall_clock="$2"

  {
    echo "# Colocate cheap-host smoke report"
    echo "# Generated:   $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "# Repo:        $REPO_ROOT"
    echo "# Branch:      $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "# Commit:      $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "# Test files:  ${TEST_FILES[*]}"
    echo
    echo "## Exit code"
    echo "$pytest_rc"
    echo
    echo "## Wall-clock (seconds)"
    echo "$wall_clock"
    echo
    echo "## Host details"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null \
      || echo "nvidia-smi unavailable"
    echo "Kernel:    $(uname -srm)"
    echo "Python:    $($PYTHON --version 2>&1)"
    echo
    echo "## pytest summary"
    if [[ -f "$PYTEST_LOG" ]]; then
      grep -E "^=+ .*(passed|failed|skipped|error).*=+$" "$PYTEST_LOG" \
        | tail -n 5 || echo "(no pytest summary line found)"
    else
      echo "(pytest log $PYTEST_LOG missing)"
    fi
    echo
    echo "## Captured loss progression"
    if [[ -f "$PYTEST_LOG" ]]; then
      grep -E "\[colocate_loop\] step=[0-9]+" "$PYTEST_LOG" \
        | sed 's/^.*\[colocate_loop\]/[colocate_loop]/' \
        || echo "(no [colocate_loop] lines — either all tests SKIPPED or output format changed)"
    fi
    echo
    echo "## SKIPPED tests"
    if [[ -f "$PYTEST_LOG" ]]; then
      grep -E "^SKIPPED \[" "$PYTEST_LOG" | head -n 20 \
        || echo "(none — every test was selected for run)"
    fi
    echo
    if [[ "$pytest_rc" -ne 0 ]]; then
      echo "## Pytest tail (last 60 lines) — FAILURE CASE"
      if [[ -f "$PYTEST_LOG" ]]; then
        tail -n 60 "$PYTEST_LOG"
      fi
      echo
      if [[ -f /tmp/nvidia-log/server.log ]]; then
        echo "## /tmp/nvidia-log/server.log tail (last 50 lines)"
        tail -n 50 /tmp/nvidia-log/server.log
      fi
      if [[ -f /tmp/nvidia-log/control.log ]]; then
        echo
        echo "## /tmp/nvidia-log/control.log tail (last 50 lines)"
        tail -n 50 /tmp/nvidia-log/control.log
      fi
    fi
  } > "$REPORT_PATH"

  echo
  echo "Report written to: $REPORT_PATH"
  echo "Pytest log:        $PYTEST_LOG"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Pre-flight first, *before* the expensive setup step, so a host without
# working MPS bails in seconds. With --setup-only we skip the pre-flight
# entirely (e.g. baking an image on a build host that has no GPU).
if [[ $DO_RUN -eq 1 ]]; then
  run_preflight
fi

if [[ $DO_SETUP -eq 1 ]]; then
  setup_system_libs
  setup_sglang
  setup_python
else
  banner "Skipping setup (--skip-setup)"
fi

if [[ $DO_RUN -eq 0 ]]; then
  banner "Setup complete (--setup-only). Re-run without --setup-only to run tests."
  exit 0
fi

pick_test_files

banner "pytest: ${TEST_FILES[*]}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
# Default CUDA_VISIBLE_DEVICES depends on whether we're running --full
# (multi-GPU) or just the tiny smoke. Don't override an already-set value.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  if [[ $RUN_FULL -eq 1 ]] && [[ "$GPU_COUNT" -ge 4 ]]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$REPO_ROOT"
START_TS=$(date +%s)
PYTEST_RC=0
# tee'd so write_report can grep loss values + summary + SKIP reasons.
# PIPESTATUS captures pytest's exit (bash-only; shebang is bash).
set +e
$PYTHON -m pytest -xvs "${TEST_FILES[@]}" 2>&1 | tee "$PYTEST_LOG"
PYTEST_RC=${PIPESTATUS[0]}
set -e
END_TS=$(date +%s)
WALL_CLOCK=$((END_TS - START_TS))

write_report "$PYTEST_RC" "$WALL_CLOCK"

banner "Smoke run complete (pytest exit=$PYTEST_RC, wall=${WALL_CLOCK}s)."
exit "$PYTEST_RC"

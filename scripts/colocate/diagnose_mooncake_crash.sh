#!/usr/bin/env bash
# scripts/colocate/diagnose_mooncake_crash.sh
#
# Capture the real stack trace of the Mooncake-disagg SIGSEGV.
#
# Why this exists:
#   The disaggregated grad-parity baseline arm SIGSEGVs inside the
#   Mooncake transfer engine's Go runtime on rental hosts (see
#   docs/colocate/implementation_log.md §"GPU validation" Session B).
#   `colocate.patch` replaces exactly this fragility — but to *fix* the
#   disagg arm (or pick a host where it doesn't crash) we need the
#   actual crash signature, not "it SIGSEGVs somewhere".
#
#   Mooncake already defaults to protocol=tcp (see
#   torchspec/config/mooncake_config.py), so this is NOT an RDMA /
#   verbs problem — it is an environment problem (container seccomp,
#   kernel, glibc, or core Mooncake bug). This script fingerprints the
#   host and runs the disagg path under full crash instrumentation so
#   the next run knows exactly which host trait to require.
#
# Prerequisites on the host (same as run_smoke_host.sh):
#   * `torchspec` and `mooncake.store` importable — run
#       bash scripts/colocate/run_smoke_host.sh --setup-only
#     first on a fresh pod, then run this script.
#   * `gdb` is optional but recommended (apt-get install -y gdb) — it
#     turns a core dump into a C/C++ backtrace.
#
# Usage (from the repo root):
#   bash scripts/colocate/diagnose_mooncake_crash.sh
#
# Output:
#   mooncake-crash-report.txt — host fingerprint + Go traceback + dmesg
#   segfault line + gdb backtrace (if a core was produced). Paste this
#   back; it is the whole deliverable.
#
# Exit codes:
#   0 — the disagg run completed WITHOUT crashing (this host is a
#       candidate for the real grad-parity run — surprising; double
#       check the report)
#   2 — the disagg run crashed; the report has the captured signature
#   1 — could not even start (deps missing / config missing)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
CONFIG="configs/disagg_qwen0p6b_tiny.yaml"
REPORT="$REPO_ROOT/mooncake-crash-report.txt"
RUN_LOG="$(mktemp /tmp/mooncake-disagg-run.XXXXXX.log)"
STEPS="${MOONCAKE_DIAG_STEPS:-2}"
RUN_TIMEOUT="${MOONCAKE_DIAG_TIMEOUT:-1800}"   # 30 min hard cap

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
: > "$REPORT"
section() { printf '\n===== %s =====\n' "$1" | tee -a "$REPORT"; }
log()     { printf '%s\n' "$*" | tee -a "$REPORT"; }
# Run a command, capture stdout+stderr into the report, never abort the script.
cap()     { log "\$ $*"; { "$@" 2>&1 || log "(command failed: rc=$?)"; } | tee -a "$REPORT"; }

log "Mooncake-disagg crash diagnosis — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "repo: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 0. Preconditions
# ---------------------------------------------------------------------------
if [[ ! -f "$CONFIG" ]]; then
  log "FATAL: $CONFIG not found."
  exit 1
fi
if ! "$PYTHON" -c 'import torchspec' >/dev/null 2>&1; then
  log "FATAL: 'import torchspec' failed. Run:"
  log "  bash scripts/colocate/run_smoke_host.sh --setup-only"
  log "first, then re-run this script."
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Host fingerprint — the point of "diagnose first": this tells us which
#    host traits correlate with the crash so the next host can dodge it.
# ---------------------------------------------------------------------------
section "HOST / OS"
cap uname -a
cap cat /etc/os-release
cap systemd-detect-virt
# gVisor (Modal-style) and other sandboxes show up here:
log "--- kernel identity (gVisor/sandbox tell) ---"
cap cat /proc/version
cap cat /proc/sys/kernel/osrelease

section "GLIBC / TOOLCHAIN"
cap ldd --version
cap "$PYTHON" --version
command -v go >/dev/null 2>&1 && cap go version || log "go: not on PATH (Mooncake ships its own runtime)"

section "CONTAINER ISOLATION (the prime suspect — Mooncake is on TCP, not RDMA)"
# seccomp mode 2 = filtered: a blocked syscall is the classic Go-runtime SIGSEGV cause.
cap grep -E 'Seccomp|CapEff|NoNewPrivs' /proc/self/status
command -v capsh >/dev/null 2>&1 && cap capsh --print || log "capsh: not installed (apt-get install -y libcap2-bin)"
log "--- cgroup (container vs bare VM) ---"
cap cat /proc/1/cgroup
log "--- ulimits (core dump size must be non-zero to get a core) ---"
cap bash -c 'ulimit -a'
log "--- shared memory (Mooncake transfer engine uses /dev/shm) ---"
cap df -h /dev/shm

section "RDMA SURFACE (should be irrelevant at protocol=tcp — recorded for completeness)"
command -v ibv_devices >/dev/null 2>&1 && cap ibv_devices || log "ibv_devices: not installed"
cap ls -l /dev/infiniband

section "GPU"
cap nvidia-smi

section "MOONCAKE BUILD"
cap bash -c "pip show mooncake-transfer-engine 2>/dev/null || pip show mooncake 2>/dev/null || echo 'mooncake: pip metadata not found'"
MC_SO="$("$PYTHON" -c 'import mooncake.store as m; print(m.__file__)' 2>/dev/null)"
if [[ -n "$MC_SO" ]]; then
  log "mooncake.store module: $MC_SO"
  cap file "$MC_SO"
  # ldd on the native .so reveals which RDMA/Go deps it actually links.
  NATIVE_SO="$(find "$(dirname "$MC_SO")" -maxdepth 2 -name '*.so' 2>/dev/null | head -3)"
  for so in $NATIVE_SO; do cap ldd "$so"; done
else
  log "mooncake.store: NOT importable — disagg path cannot run here."
fi
MC_MASTER="$("$PYTHON" -c 'from torchspec.transfer.mooncake.utils import resolve_mooncake_master_bin as r; print(r())' 2>/dev/null)"
log "mooncake_master binary: ${MC_MASTER:-<unresolved>}"
[[ -n "${MC_MASTER:-}" && -f "$MC_MASTER" ]] && cap file "$MC_MASTER"

# ---------------------------------------------------------------------------
# 2. Crash-capture environment
# ---------------------------------------------------------------------------
section "CRASH-CAPTURE SETUP"
# Core dumps: try to get one. In a container without CAP_SYS_ADMIN we may
# not be able to set core_pattern — record whether it worked.
ulimit -c unlimited 2>/dev/null && log "ulimit -c: unlimited (OK)" || log "ulimit -c: could NOT raise (no core dump expected)"
CORE_DIR="$REPO_ROOT/cores"
mkdir -p "$CORE_DIR"
if echo "$CORE_DIR/core.%e.%p" > /proc/sys/kernel/core_pattern 2>/dev/null; then
  log "core_pattern -> $CORE_DIR/core.%e.%p (OK)"
else
  log "core_pattern: read-only (container) — relying on Go traceback + dmesg instead"
  log "current core_pattern: $(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo '<unreadable>')"
fi

# GOTRACEBACK=crash: on a Go runtime fault, dump ALL goroutine stacks +
# register state, then re-raise the signal so a core is produced. This is
# the single most useful knob — it turns "SIGSEGV" into a real stack.
export GOTRACEBACK=crash
export GODEBUG=cgocheck=1
# Make TorchSpec / Mooncake as loud as possible.
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-DEBUG}"
export MC_LOG_LEVEL="${MC_LOG_LEVEL:-INFO}"
export GLOG_v="${GLOG_v:-1}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
log "GOTRACEBACK=crash  GODEBUG=cgocheck=1  PYTHONFAULTHANDLER=1"

# Snapshot dmesg position so we only report NEW segfault lines.
DMESG_BEFORE="$(dmesg 2>/dev/null | wc -l || echo 0)"

# ---------------------------------------------------------------------------
# 3. Run the disagg path
# ---------------------------------------------------------------------------
section "DISAGG RUN ($CONFIG, $STEPS steps, ${RUN_TIMEOUT}s cap)"
log "run log: $RUN_LOG"
set -x
timeout --signal=SIGTERM "$RUN_TIMEOUT" \
  "$PYTHON" -m torchspec.train_entry \
    --config "$CONFIG" \
    "training.num_train_steps=$STEPS" \
    "training.num_epochs=1" \
    > "$RUN_LOG" 2>&1
RUN_RC=$?
set +x
log "disagg run exit code: $RUN_RC"

# ---------------------------------------------------------------------------
# 4. Post-mortem
# ---------------------------------------------------------------------------
section "RUN LOG TAIL (last 80 lines)"
tail -n 80 "$RUN_LOG" | tee -a "$REPORT"

section "GO RUNTIME TRACEBACK (GOTRACEBACK=crash output)"
# The Go panic block: 'fatal error' / 'panic' / 'signal SIGSEGV' followed
# by 'goroutine N [...]:' stacks. Print a generous window around it.
if grep -nE 'SIGSEGV|fatal error|runtime\.|goroutine [0-9]+ |signal arrived|cgocheck' "$RUN_LOG" >/dev/null 2>&1; then
  grep -nE -A2 -B2 'SIGSEGV|fatal error|^panic|goroutine [0-9]+ \[|signal arrived|^runtime\.|cgocheck|created by ' "$RUN_LOG" \
    | head -200 | tee -a "$REPORT"
else
  log "No Go-runtime crash markers in the run log."
fi

section "PYTHON FAULTHANDLER / TRACEBACK"
grep -nE -A3 -B1 'Fatal Python error|Current thread|Traceback \(most recent' "$RUN_LOG" \
  | head -80 | tee -a "$REPORT" || log "(none)"

section "KERNEL dmesg — new segfault lines"
DMESG_NOW="$(dmesg 2>/dev/null | wc -l || echo 0)"
if [[ "$DMESG_NOW" -gt "$DMESG_BEFORE" ]]; then
  dmesg 2>/dev/null | tail -n $((DMESG_NOW - DMESG_BEFORE)) \
    | grep -iE 'segfault|general protection|traps|oom|killed process' \
    | tee -a "$REPORT" || log "(no segfault/oom lines in new dmesg)"
else
  log "dmesg: unreadable or no new lines (common in unprivileged containers)."
fi

section "CORE DUMP -> BACKTRACE"
CORE_FILE="$(ls -t "$CORE_DIR"/core.* 2>/dev/null | head -1)"
[[ -z "$CORE_FILE" ]] && CORE_FILE="$(ls -t "$REPO_ROOT"/core* /tmp/core* 2>/dev/null | head -1)"
if [[ -n "${CORE_FILE:-}" && -f "$CORE_FILE" ]]; then
  log "core file: $CORE_FILE ($(du -h "$CORE_FILE" | cut -f1))"
  if command -v gdb >/dev/null 2>&1; then
    PYBIN="$("$PYTHON" -c 'import sys; print(sys.executable)')"
    cap gdb --batch -nx \
      -ex 'thread apply all bt' \
      -ex 'info sharedlibrary' \
      "$PYBIN" "$CORE_FILE"
  else
    log "gdb not installed — apt-get install -y gdb, then:"
    log "  gdb --batch -ex 'thread apply all bt' \$(which $PYTHON) $CORE_FILE"
  fi
else
  log "No core file produced (core_pattern likely read-only in this container)."
  log "The GOTRACEBACK=crash block above is the primary signature in that case."
fi

# ---------------------------------------------------------------------------
# 5. Verdict
# ---------------------------------------------------------------------------
section "VERDICT"
if [[ "$RUN_RC" -eq 0 ]]; then
  log "Disagg run COMPLETED WITHOUT CRASHING on this host."
  log "-> This host is a candidate for the real Mooncake-disagg grad-parity run."
  log "-> Record its fingerprint above as a known-good environment."
  exit 0
elif [[ "$RUN_RC" -eq 124 ]]; then
  log "Disagg run HUNG (timeout after ${RUN_TIMEOUT}s) — not a clean SIGSEGV."
  log "-> Check the Go traceback section: a deadlock looks different from a crash."
  exit 2
else
  log "Disagg run FAILED (rc=$RUN_RC)."
  log "-> The captured signature above identifies the host trait to require/avoid."
  log "-> Full run log preserved at: $RUN_LOG"
  exit 2
fi

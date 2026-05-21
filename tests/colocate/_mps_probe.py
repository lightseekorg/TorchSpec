# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Shared helpers for the colocate phase tests.

Centralised here because every Phase-4+ test needs the same two
preconditions (>=4 GPUs *and* a working MPS daemon), and the MPS
probe is a 50-line subprocess dance we don't want to copy four times.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def has_n_gpus(n: int) -> bool:
    """Return True iff at least ``n`` CUDA GPUs are visible to nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return len([g for g in out.splitlines() if g.strip()]) >= n


def has_h100_quad() -> bool:
    """Detect whether we're on a Modal H100:4 (or any 4+ GPU box).

    Thin wrapper over ``has_n_gpus(4)`` for backwards compat with
    existing Phase-4/6/7 ``pytest.mark.skipif`` calls; the cheap-host
    1-GPU tiny tests use ``has_n_gpus(1)`` directly.
    """
    return has_n_gpus(4)


def mps_works_verbose() -> tuple[bool, str]:
    """Like :func:`mps_works` but returns ``(ok, reason)``.

    ``reason`` is a single-line human-readable string suitable for
    logging or printing to stderr. On failure it tries to extract the
    most diagnostic line from ``/tmp/nvidia-log/server.log`` (e.g.
    ``"operation not supported"``) so callers can tell ``no --ipc=host``
    apart from e.g. ``CUDA driver too old``.

    Implementation mirrors
    ``torchspec.colocate.mps._probe_mps_server_works`` but is kept here
    so test files (and ``scripts/colocate/run_smoke_host.sh``) don't
    need to import torchspec just to gate their pytest ``skipif``.
    """
    if not shutil.which("nvidia-cuda-mps-control"):
        return False, "nvidia-cuda-mps-control not on PATH (install CUDA toolkit)"
    pipe_dir = "/tmp/nvidia-mps"
    log_dir = "/tmp/nvidia-log"
    try:
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        env = {
            **os.environ,
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": log_dir,
        }
        if not os.path.exists(os.path.join(pipe_dir, "control")):
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                env=env,
                timeout=10,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        probe_code = (
            "import ctypes, sys\n"
            "cuda = ctypes.CDLL('libcuda.so.1')\n"
            "rc = cuda.cuInit(0)\n"
            "if rc != 0:\n    sys.exit(rc)\n"
            "cnt = ctypes.c_int(0)\n"
            "rc = cuda.cuDeviceGetCount(ctypes.byref(cnt))\n"
            "sys.exit(rc)\n"
        )
        proc = subprocess.run(
            ["python3", "-c", probe_code],
            env=env,
            timeout=20,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            return True, "ok"

        server_log = os.path.join(log_dir, "server.log")
        detail = ""
        if os.path.exists(server_log):
            with open(server_log, "rb") as f:
                tail = f.read()[-2048:].decode("utf-8", errors="replace")
            if "operation not supported" in tail:
                detail = (
                    " — MPS server reports 'operation not supported' "
                    "(container likely lacks --ipc=host; switch host/template)"
                )
            elif tail.strip():
                detail = f" (server.log tail: {tail.strip().splitlines()[-1]!r})"
        return False, (f"cuInit/cuDeviceGetCount returned rc={proc.returncode}{detail}")
    except Exception as e:
        return False, f"unexpected exception during MPS probe: {e!r}"


def mps_works() -> bool:
    """True iff nvidia-cuda-mps-control is on PATH and the per-GPU
    server can actually start a CUDA context. False on hosts where
    the MPS server reports 'operation not supported' (e.g. Modal
    sandbox H100 nodes without --ipc=host); see
    docs/colocate/implementation_log.md for the full story.

    Thin wrapper over :func:`mps_works_verbose` for the common case of
    a pytest ``skipif`` predicate that only needs a bool.
    """
    return mps_works_verbose()[0]


if __name__ == "__main__":
    # CLI: print the verbose reason and exit 0/1. Used by
    # ``scripts/colocate/run_smoke_host.sh`` for the pre-flight gate
    # and by humans following the doc's "Quick MPS sanity check".
    import sys

    ok, reason = mps_works_verbose()
    print(f"mps_works: {ok} — {reason}")
    sys.exit(0 if ok else 1)

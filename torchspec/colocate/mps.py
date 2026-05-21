# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""NVIDIA MPS (Multi-Process Service) lifecycle helper (Phase 1).

The colocate plan puts a trainer process and an inference engine process on
the same physical GPU. By default CUDA serialises kernels from different
processes, which makes context-switch overhead dominate. MPS reroutes both
processes' commands to a single per-GPU server so the GPU sees them as
threads of one client and can run kernels concurrently.

What this module does:

    1. Detect whether `nvidia-cuda-mps-control` is already running on this
       node (idempotent — multiple drivers must coexist safely).
    2. If not, start it with `nvidia-cuda-mps-control -d` (daemon mode).
    3. Return the env-var dict that client processes (TrainerActor and
       SglEngine actors) need to merge into their Ray ``runtime_env``.
    4. Provide a best-effort cleanup hook (`stop_mps_daemon`) called at
       shutdown.

What this module does NOT do:

    - Manage `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`. That's an optional Phase-6
      knob; off by default.
    - Spawn one daemon per GPU. A single MPS control daemon services all
      GPUs visible to the calling user.
    - Touch CUDA — it's pure subprocess + filesystem, so it's safely
      importable from the Ray driver on a headless box.

The module is split out so that:

    - Unit tests can verify env-var construction and idempotency without
      requiring NVIDIA drivers (subprocess is mocked).
    - The Ray driver doesn't import torch just to set up MPS.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("torchspec.colocate.mps")

# Default control-pipe and log directories. MPS clients identify the daemon
# by these env vars, so trainer and engine processes must agree on them
# (and so must the daemon process). These are the documented NVIDIA
# defaults; we expose them as constants so tests can match them.
DEFAULT_PIPE_DIR = "/tmp/nvidia-mps"
DEFAULT_LOG_DIR = "/tmp/nvidia-log"

_MPS_CONTROL_BIN = "nvidia-cuda-mps-control"
_MPS_SERVER_BIN = "nvidia-cuda-mps-server"


@dataclass
class MpsHandle:
    """Information about a started (or detected) MPS daemon."""

    pipe_dir: str
    log_dir: str
    started_by_us: bool
    """True if *this* call launched the daemon. False if it was already
    running, in which case ``stop_mps_daemon`` becomes a best-effort no-op."""


def mps_client_env(
    pipe_dir: str = DEFAULT_PIPE_DIR, log_dir: str = DEFAULT_LOG_DIR
) -> dict[str, str]:
    """Env vars that MPS clients (trainer + engine) need.

    Both must point at the same control pipe directory; otherwise they'd
    talk to different MPS servers (or none), defeating the colocate goal.
    Documented at https://docs.nvidia.com/deploy/mps/index.html#environment-variables.
    """
    return {
        "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
        "CUDA_MPS_LOG_DIRECTORY": log_dir,
    }


def is_mps_available() -> bool:
    """True iff ``nvidia-cuda-mps-control`` is in PATH.

    Used as a precondition for callers that want to fall back gracefully on
    boxes without MPS (e.g. local dev, CPU-only CI).
    """
    return shutil.which(_MPS_CONTROL_BIN) is not None


def is_mps_running(pipe_dir: str = DEFAULT_PIPE_DIR) -> bool:
    """True iff an MPS control daemon appears to be running on this node.

    We check two signals because either alone is unreliable:

    - The control pipe directory exists *and* contains the named pipe
      ``control`` (created by the daemon at startup).
    - ``ps`` shows an `nvidia-cuda-mps-control` process.

    Either match is good enough; we only need one to avoid double-starting.
    """
    pipe_file = os.path.join(pipe_dir, "control")
    if os.path.exists(pipe_file):
        return True

    if not shutil.which("pgrep"):
        # On an unusual base image without pgrep — fall back to "no daemon".
        # We'd rather double-start (the second instance fails fast with
        # `daemon already running`) than skip startup on a fresh box.
        return False
    try:
        rc = subprocess.run(
            ["pgrep", "-f", _MPS_CONTROL_BIN],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).returncode
    except subprocess.TimeoutExpired:
        return False
    return rc == 0


def start_mps_daemon(
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    *,
    skip_if_running: bool = True,
) -> MpsHandle:
    """Start the MPS control daemon (idempotent).

    Args:
        pipe_dir: ``CUDA_MPS_PIPE_DIRECTORY`` to use. Defaults to NVIDIA's
            documented ``/tmp/nvidia-mps`` so a daemon started by
            ``nvidia-cuda-mps-control -d`` (no env vars) works out of the
            box.
        log_dir: ``CUDA_MPS_LOG_DIRECTORY`` to use.
        skip_if_running: If True (default), return without starting if a
            daemon is already up. Set to False for tests that want to force
            a fresh start.

    Returns:
        An ``MpsHandle`` capturing the directories and whether *we* started
        the daemon.

    Raises:
        FileNotFoundError: ``nvidia-cuda-mps-control`` not in PATH.
        RuntimeError: the start command failed (e.g. permission error,
            previous orphaned daemon, etc.).
    """
    if not is_mps_available():
        raise FileNotFoundError(
            f"{_MPS_CONTROL_BIN} not found on PATH. MPS ships with the CUDA "
            "toolkit; ensure CUDA development tools are installed in the "
            "container image."
        )

    if skip_if_running and is_mps_running(pipe_dir):
        logger.info("MPS daemon already running; not starting another.")
        return MpsHandle(pipe_dir=pipe_dir, log_dir=log_dir, started_by_us=False)

    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = {**os.environ, **mps_client_env(pipe_dir=pipe_dir, log_dir=log_dir)}
    logger.info("Starting MPS control daemon (pipe_dir=%s, log_dir=%s)", pipe_dir, log_dir)
    try:
        # `-d` runs in daemon mode; the binary backgrounds itself and exits
        # 0 if it spawned successfully.
        subprocess.run(
            [_MPS_CONTROL_BIN, "-d"],
            env=env,
            check=True,
            timeout=30,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        # If the daemon was already running, a second `-d` call is harmless
        # but exits non-zero with a recognisable message. Treat as success.
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        if "already running" in stderr.lower():
            logger.info("MPS daemon already running (race-detected at start time).")
            return MpsHandle(pipe_dir=pipe_dir, log_dir=log_dir, started_by_us=False)
        raise RuntimeError(
            f"Failed to start MPS daemon (exit {e.returncode}): {stderr.strip()}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timed out starting MPS daemon: {e}") from e

    # The daemon's `-d` mode forks and returns immediately. The control
    # pipe under `pipe_dir/control` is only created once the daemon's
    # init completes. If we return here without polling, downstream
    # actors that call `torch.cuda.set_device(...)` race with the
    # daemon's startup and CUDA reports error 805 ("MPS client failed
    # to connect to the MPS control daemon or the MPS server"). Poll
    # for the pipe file so this race is impossible.
    import time

    deadline = time.time() + 10.0
    pipe_file = os.path.join(pipe_dir, "control")
    while time.time() < deadline:
        if os.path.exists(pipe_file):
            break
        time.sleep(0.1)
    else:
        # Daemon failed to come up cleanly. Try to surface a helpful
        # error rather than the obscure CUDA error 805 that downstream
        # actors would otherwise hit.
        raise RuntimeError(
            f"MPS daemon did not produce {pipe_file!r} within 10s. "
            f"Check {log_dir}/control.log on the host for daemon logs. "
            f"Common causes: stale {pipe_dir} from a previous run "
            f"(rm -rf and retry), incompatible CUDA driver, or container "
            f"missing /dev/shm + /run mounts."
        )
    logger.info("MPS daemon ready (control pipe %s exists)", pipe_file)

    return MpsHandle(pipe_dir=pipe_dir, log_dir=log_dir, started_by_us=True)


def stop_mps_daemon(handle: Optional[MpsHandle] = None) -> bool:
    """Best-effort shutdown. Returns True iff we actually told a daemon to quit.

    The driver's atexit / Ray shutdown hook calls this. We deliberately
    swallow errors — leaving an orphan MPS daemon costs only a small idle
    process, whereas raising during cleanup would mask the real exception
    that triggered shutdown.
    """
    if not is_mps_available():
        return False

    pipe_dir = handle.pipe_dir if handle else DEFAULT_PIPE_DIR
    log_dir = handle.log_dir if handle else DEFAULT_LOG_DIR

    if not is_mps_running(pipe_dir):
        return False

    env = {**os.environ, **mps_client_env(pipe_dir=pipe_dir, log_dir=log_dir)}
    try:
        subprocess.run(
            [_MPS_CONTROL_BIN],
            input=b"quit\n",
            env=env,
            timeout=15,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        logger.info("Sent 'quit' to MPS control daemon.")
        return True
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("Best-effort MPS shutdown failed: %s", e)
        return False


def force_stop_mps(
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
) -> None:
    """Forcefully tear MPS down: kill the daemon + server, remove the dirs.

    :func:`stop_mps_daemon` sends a graceful ``quit``, which a CUDA
    client still attached to the MPS server can block indefinitely
    ("Server was unable to shutdown due to N active clients"), leaving
    the daemon stuck half-shutdown and rejecting new clients with CUDA
    error 805. This always succeeds: SIGKILL the ``nvidia-cuda-mps``
    processes and delete the pipe/log dirs so the node is cleanly
    no-MPS again.

    Use it to guarantee a no-MPS environment — e.g. a disaggregated run
    on a node where a colocate run left MPS up — or to recover from a
    stuck daemon. A subsequent :func:`setup_for_colocate` starts fresh.
    """
    import time

    try:
        subprocess.run(
            ["pkill", "-9", "-f", "nvidia-cuda-mps"],
            check=False,
            timeout=10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as e:  # pragma: no cover
        logger.warning("force_stop_mps: pkill failed: %s", e)
    time.sleep(1.0)
    shutil.rmtree(pipe_dir, ignore_errors=True)
    shutil.rmtree(log_dir, ignore_errors=True)
    logger.info(
        "force_stop_mps: killed nvidia-cuda-mps processes, removed %s + %s",
        pipe_dir,
        log_dir,
    )


def _probe_mps_server_works(
    pipe_dir: str, log_dir: str, *, timeout_s: float = 30.0
) -> tuple[bool, str]:
    """Force the MPS daemon to spawn a server and report whether it succeeded.

    The daemon launches the per-GPU server process *lazily* on the first
    client connect, so a healthy ``-d`` start tells us nothing about
    whether the server can actually create a CUDA context. On
    container hosts (Modal sandbox H100s, in particular) the daemon
    starts cleanly but the server fails immediately with
    ``Failed to start : operation not supported``, leaving every
    real CUDA client to crash with ``Error 805``.

    The most reliable probe is to spawn a tiny CUDA client (a
    subprocess that imports torch and does ``torch.cuda.device_count()``)
    with the MPS env vars set: if it succeeds, MPS works; if it
    raises with error 805 (or its CUDA equivalent), MPS is broken
    and we should fall back. We do this in an isolated subprocess
    so the *driver's* CUDA state isn't polluted by a failed init.

    Returns ``(ok, reason)`` so the caller can log a useful message.
    """
    env = {**os.environ, **mps_client_env(pipe_dir=pipe_dir, log_dir=log_dir)}

    probe_code = (
        "import os, sys, ctypes\n"
        "try:\n"
        "    cuda = ctypes.CDLL('libcuda.so.1')\n"
        "    rc = cuda.cuInit(0)\n"
        "    if rc != 0:\n"
        "        sys.exit(rc)\n"
        "    cnt = ctypes.c_int(0)\n"
        "    rc = cuda.cuDeviceGetCount(ctypes.byref(cnt))\n"
        "    sys.exit(rc)\n"
        "except OSError as e:\n"
        "    sys.stderr.write(str(e))\n"
        "    sys.exit(255)\n"
    )
    try:
        proc = subprocess.run(
            ["python3", "-c", probe_code],
            env=env,
            timeout=timeout_s,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return False, f"MPS probe timed out after {timeout_s}s: {e}"

    if proc.returncode == 0:
        return True, "ok"

    # Check the server log too — the daemon writes its own diagnostic
    # there which is much more readable than the bare cuInit return
    # code.
    server_log = os.path.join(log_dir, "server.log")
    detail = ""
    if os.path.exists(server_log):
        with open(server_log, "rb") as f:
            tail = f.read()[-2048:].decode("utf-8", errors="replace")
        if "operation not supported" in tail:
            detail = " (MPS server reported 'operation not supported' — common in containers without --ipc=host)"
        elif tail.strip():
            detail = f" (server.log tail: {tail.strip().splitlines()[-1]!r})"
    return False, (
        f"MPS probe failed with cuInit/cuDeviceGetCount rc={proc.returncode}"
        f"{detail}. Falling back to fractional GPU sharing without MPS."
    )


def setup_for_colocate(
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    *,
    register_atexit: bool = True,
    probe_server: bool = True,
) -> tuple[Optional[MpsHandle], dict[str, str]]:
    """One-shot: start daemon (if needed), return handle + client env.

    Convenience entry point for the Ray driver — mirrors the
    ``setup_for_colocate(...)`` signature the placement-group code will
    import in the next sub-task of Phase 1.

    Phase 6 hygiene: when ``register_atexit`` is true (default) and we
    actually started the daemon, register an ``atexit`` hook to
    ``stop_mps_daemon`` so a clean driver shutdown doesn't leak the
    daemon process. SIGKILL / OOM-kills bypass ``atexit`` of course;
    that's by design — the next driver run's ``start_mps_daemon`` is
    idempotent and will reuse a still-running daemon.

    When ``probe_server`` (default) is true we eagerly spawn an MPS
    server to detect environments where the daemon comes up but the
    server can't create a CUDA context (Modal sandbox H100s, some
    Docker hosts without --ipc=host). On detection we tear the
    daemon back down and return ``(None, {})``: the caller still gets
    a working colocate path (fractional GPU claim, no MPS env) — the
    only loss is concurrent trainer/engine kernel execution.

    Set ``TORCHSPEC_DISABLE_MPS=1`` to skip MPS bring-up entirely
    (useful for local / CI environments where MPS is known broken).
    """
    if os.environ.get("TORCHSPEC_DISABLE_MPS", "") in ("1", "true", "True"):
        logger.info(
            "TORCHSPEC_DISABLE_MPS set; skipping MPS daemon. Trainer "
            "and engine will share each GPU but kernels will serialise."
        )
        return None, {}

    handle = start_mps_daemon(pipe_dir=pipe_dir, log_dir=log_dir)

    if probe_server:
        ok, reason = _probe_mps_server_works(pipe_dir=pipe_dir, log_dir=log_dir)
        if not ok:
            logger.warning("MPS server probe failed: %s", reason)
            # Best-effort tear down so a future driver run doesn't
            # find a stale (broken) daemon and skip restart.
            try:
                stop_mps_daemon(handle)
            except Exception:
                logger.exception("Failed to stop broken MPS daemon")
            return None, {}

    if register_atexit and handle.started_by_us:
        import atexit

        atexit.register(stop_mps_daemon, handle)
    return handle, mps_client_env(pipe_dir=pipe_dir, log_dir=log_dir)


def ensure_mps_on_all_nodes(
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
) -> dict[str, bool]:
    """Start the MPS control daemon on every node of the Ray cluster.

    The driver-side :func:`setup_for_colocate` only brings MPS up on the
    *driver's own* node. For multi-node colocate, every node that will
    host a trainer or engine actor needs its own daemon — once a node's
    daemon is up, every CUDA process there must register with it (else
    CUDA error 805). This schedules one idempotent bootstrap task per
    live node via Ray node-affinity scheduling.

    Must be called after ``ray.init()`` and before any colocate actor is
    created. Idempotent — a node whose daemon is already running (e.g.
    the driver node) is a no-op. Single-node clusters therefore make
    this a no-op superset of the pre-Ray bring-up.

    The per-node tasks pass ``probe_server=False`` (the driver node was
    already probed pre-Ray; non-driver nodes are assumed validated by
    the operator) and ``register_atexit=False`` (a short-lived Ray task
    is not the daemon's owner — the daemon persists like the driver
    node's, and is reaped by node teardown).

    Returns ``{node_id: started_ok}``. Failures are logged, not raised:
    a node that fails here will surface a clear CUDA error 805 when its
    first actor starts.

    NOTE: the multi-node colocate path is implemented but has only been
    exercised single-node — see docs/colocate/usage.md.
    """
    import ray

    try:
        from ray.util.scheduling_strategies import (
            NodeAffinitySchedulingStrategy,
        )
    except Exception:  # pragma: no cover - very old ray
        logger.warning(
            "ray.util.scheduling_strategies unavailable; cannot pin "
            "per-node MPS bootstrap. Multi-node colocate needs MPS started "
            "on each node out-of-band."
        )
        return {}

    nodes = [n for n in ray.nodes() if n.get("Alive")]

    @ray.remote(num_cpus=0)
    def _bootstrap_mps_on_node() -> bool:
        handle, _env = setup_for_colocate(
            pipe_dir=pipe_dir,
            log_dir=log_dir,
            register_atexit=False,
            probe_server=False,
        )
        return handle is not None

    pending = {}
    for n in nodes:
        node_id = n["NodeID"]
        strategy = NodeAffinitySchedulingStrategy(node_id, soft=False)
        pending[node_id] = _bootstrap_mps_on_node.options(scheduling_strategy=strategy).remote()

    results: dict[str, bool] = {}
    for node_id, ref in pending.items():
        try:
            results[node_id] = bool(ray.get(ref))
        except Exception:
            logger.exception("MPS bootstrap failed on node %s", node_id)
            results[node_id] = False
    logger.info(
        "[colocate] per-node MPS bootstrap: %d/%d nodes ready",
        sum(results.values()),
        len(results),
    )
    return results

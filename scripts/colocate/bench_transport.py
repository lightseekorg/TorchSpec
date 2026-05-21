#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Benchmark: colocate hidden-state transports — gloo, CUDA IPC, and the
protocol-level IPC optimizations (send-buffer pool, ack pipelining).

The colocate hidden-state plane moves engine→trainer tensors between two
processes that share one physical GPU. This script A/B-tests four
transport arms:

  * **gloo** — engine ``D->H`` copy, gloo ship, trainer ``H->D`` copy.
    Two PCIe-class copies + a host memcpy per tensor.
  * **ipc** — the current production CUDA IPC path
    (``torchspec.colocate.cuda_ipc``): engine exports a fresh CUDA IPC
    handle every step, trainer maps it and does one on-device ``D->D``
    copy, then a blocking gloo ack.
  * **ipc-pool** — *Opt 1*: the engine copies hidden states into a
    **persistent** send buffer whose IPC handle never changes, so the
    trainer opens the handle (``cudaIpcOpenMemHandle``) **once** and
    caches the mapping for every later step.
  * **ipc-pipe** — *Opt 2*: ipc-pool plus **ack pipelining** — the
    engine defers the ack wait by one step (non-blocking ``isend`` +
    double-buffered pool), so the ~1 ms ack round-trip leaves the
    engine's critical path.

See ``docs/colocate/transport_optimization.md`` for the design and
``docs/colocate/transport_benchmark.md`` for the original gloo-vs-ipc
measurement this extends.

It spins up **two processes on GPU 0** (the colocate topology), forms a
2-rank gloo group, and for each arm reports end-to-end latency, the
engine/trainer own-call split, a **cold vs warm** breakdown (cold = the
first iteration, which pays one-time IPC setup; warm = steady state),
and a per-stage anatomy. Then it writes a Markdown report.

It needs **1 GPU**. Do **not** export
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments`` — CUDA IPC needs plain
``cudaMalloc`` memory (the script warns if it is set).

Run on a GPU host (after ``run_smoke_host.sh`` has installed torchspec,
or any env with torch + this repo importable)::

    python scripts/colocate/bench_transport.py
    python scripts/colocate/bench_transport.py --arms ipc,ipc-pool,ipc-pipe
    python scripts/colocate/bench_transport.py --iters 50 --sizes-mb 1,16,256
    python scripts/colocate/bench_transport.py --engine-step-ms 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import platform
import sys
import time
from pathlib import Path

# Make the repo importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load torchspec/colocate/cuda_ipc.py directly by file path. cuda_ipc.py
# has no torchspec-internal imports, so loading it standalone avoids
# triggering torchspec's package __init__ chain (which pulls heavy model
# deps). The benchmark then runs on a bare torch install — no
# `pip install -e .` needed on the GPU host.
import importlib.util as _ilu

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

_CUDA_IPC_PATH = _REPO_ROOT / "torchspec" / "colocate" / "cuda_ipc.py"
_spec = _ilu.spec_from_file_location("colocate_cuda_ipc", _CUDA_IPC_PATH)
_cuda_ipc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cuda_ipc)
ipc_send, ipc_recv = _cuda_ipc.ipc_send, _cuda_ipc.ipc_recv

TRAINER_RANK = 0
ENGINE_RANK = 1

ALL_ARMS = ("gloo", "ipc", "ipc-pool", "ipc-pipe")

# IPC handshake tags — kept distinct from cuda_ipc.py's (7001-7003) and
# from the gloo per-tensor tags (0,1,2,…) so nothing collides.
_BREAKDOWN_TAG = 9100
_POOL_LEN_TAG = 9200
_POOL_DATA_TAG = 9201
_POOL_ACK_TAG = 9202


# ---------------------------------------------------------------------------
# Small wire helpers (length-framed pickled blob over gloo)
# ---------------------------------------------------------------------------
# Mirrors cuda_ipc.py's framing: send_object_list / recv_object_list were
# observed to deadlock on this group, so we pickle + frame ourselves.


def _send_blob(obj, dst, group, len_tag, data_tag) -> None:
    blob = bytearray(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    buf = torch.frombuffer(blob, dtype=torch.uint8)
    dist.send(torch.tensor([buf.numel()], dtype=torch.long), dst=dst, group=group, tag=len_tag)
    dist.send(buf, dst=dst, group=group, tag=data_tag)


def _recv_blob(src, group, len_tag, data_tag):
    length = torch.empty(1, dtype=torch.long)
    dist.recv(length, src=src, group=group, tag=len_tag)
    buf = torch.empty(int(length.item()), dtype=torch.uint8)
    dist.recv(buf, src=src, group=group, tag=data_tag)
    return pickle.loads(buf.numpy().tobytes())


# ---------------------------------------------------------------------------
# Transport implementations
# ---------------------------------------------------------------------------
# Each arm is a stateful object: a fresh instance per (arm, payload) bench
# pass. The gloo path mirrors the gloo branch of NcclHiddenStatesConnector
# / NcclMultiTensorFetcher; the `ipc` arm calls the real production
# torchspec.colocate.cuda_ipc API; `ipc-pool` / `ipc-pipe` are the
# prototype optimizations from transport_optimization.md.


class Transport:
    """Base transport. `stages` accumulates per-stage seconds for the
    anatomy table (cleared after warmup so it reflects steady state)."""

    name = "base"

    def __init__(self):
        self.stages: dict = {}

    def _stage(self, key, dt):
        self.stages.setdefault(key, []).append(dt)

    def engine_send(self, payload: dict, dst: int, group) -> None:
        raise NotImplementedError

    def trainer_recv(self, spec: dict, src: int, device, group) -> dict:
        raise NotImplementedError

    def flush(self, peer: int, group, is_engine: bool) -> None:
        """Drain any deferred in-flight state at end of run. Default: none."""


class GlooTransport(Transport):
    """gloo CPU-staged: D->H copy, gloo ship, H->D copy."""

    name = "gloo"

    def engine_send(self, payload, dst, group):
        t = time.perf_counter()
        cpu = [payload[n].detach().to("cpu", copy=True).contiguous() for n in sorted(payload)]
        self._stage("gloo.engine D->H copy", time.perf_counter() - t)
        t = time.perf_counter()
        for tag, ct in enumerate(cpu):
            dist.send(ct, dst=dst, group=group, tag=tag)
        self._stage("gloo.engine gloo ship", time.perf_counter() - t)

    def trainer_recv(self, spec, src, device, group):
        t = time.perf_counter()
        bufs = []
        for tag, name in enumerate(sorted(spec)):
            shape, dtype = spec[name]
            b = torch.empty(tuple(shape), dtype=dtype, device="cpu")
            dist.recv(b, src=src, group=group, tag=tag)
            bufs.append((name, b))
        self._stage("gloo.trainer gloo recv", time.perf_counter() - t)
        t = time.perf_counter()
        out = {name: b.to(device) for name, b in bufs}
        torch.cuda.synchronize()
        self._stage("gloo.trainer H->D copy", time.perf_counter() - t)
        return out


class IpcTransport(Transport):
    """The current production CUDA IPC path (torchspec.colocate.cuda_ipc).

    Calls the real `ipc_send` / `ipc_recv` — this is the A/B baseline, so
    it must be the unmodified production code. It is treated as a black
    box (no internal `stages`); the gloo+ipc anatomy comes from
    `_breakdown` instead."""

    name = "ipc"

    def engine_send(self, payload, dst, group):
        ipc_send(payload, dst, group)

    def trainer_recv(self, spec, src, device, group):
        return ipc_recv(spec, src, device, group)


class IpcPoolTransport(Transport):
    """Opt 1 — persistent send-buffer pool + trainer mapping cache.

    The engine keeps `n_slots` persistent CUDA buffers (one per tensor,
    per slot) whose IPC handles never change. Each step it copies the
    hidden states into a slot buffer and ships the handle args **only
    the first time a slot is used**; afterwards it ships `None`. The
    trainer rebuilds + caches the mapping on first sight of a slot and
    reuses it forever, so `cudaIpcOpenMemHandle` is a one-time cost.

    Cross-process ordering: the engine `torch.cuda.synchronize()`s after
    the pool copy, before signalling — so when the trainer reads the
    buffer the engine's copy is GPU-complete (replaces the per-step IPC
    event sync that a fresh `reduce_tensor` would carry)."""

    name = "ipc-pool"
    n_slots = 1

    def __init__(self):
        super().__init__()
        self._pool = None  # engine: {name: [buf] * n_slots}
        self._pool_args = None  # engine: {name: [reduce_args] * n_slots}
        self._step = 0
        self._mapping = {}  # trainer: {(name, slot): alias tensor}

    # -- engine ------------------------------------------------------------

    def _ensure_pool(self, payload):
        if self._pool is not None:
            return
        self._pool, self._pool_args = {}, {}
        for name, t in payload.items():
            bufs = [torch.empty_like(t.detach().contiguous()) for _ in range(self.n_slots)]
            self._pool[name] = bufs
            # reduce_tensor once per persistent buffer — the IPC handle
            # is stable for the buffer's lifetime, so cache the args.
            self._pool_args[name] = [reduce_tensor(b)[1] for b in bufs]

    def engine_send(self, payload, dst, group):
        slot = self._step % self.n_slots
        first_use = self._step < self.n_slots

        t = time.perf_counter()
        self._ensure_pool(payload)
        for name in sorted(payload):
            self._pool[name][slot].copy_(payload[name])
        torch.cuda.synchronize()
        self._stage(f"{self.name}.engine pool copy", time.perf_counter() - t)

        t = time.perf_counter()
        msg = [
            (name, slot, (self._pool_args[name][slot] if first_use else None))
            for name in sorted(payload)
        ]
        _send_blob(msg, dst, group, _POOL_LEN_TAG, _POOL_DATA_TAG)
        self._stage(f"{self.name}.engine ship", time.perf_counter() - t)

        self._wait_ack(dst, group)
        self._step += 1

    def _wait_ack(self, dst, group):
        """ipc-pool waits for the ack inline (blocking). ipc-pipe overrides."""
        t = time.perf_counter()
        ack = torch.zeros(1, dtype=torch.uint8)
        dist.recv(ack, src=dst, group=group, tag=_POOL_ACK_TAG)
        self._stage(f"{self.name}.engine ack wait", time.perf_counter() - t)

    # -- trainer -----------------------------------------------------------

    def trainer_recv(self, spec, src, device, group):
        t = time.perf_counter()
        msg = _recv_blob(src, group, _POOL_LEN_TAG, _POOL_DATA_TAG)
        self._stage(f"{self.name}.trainer recv msg", time.perf_counter() - t)

        t = time.perf_counter()
        for name, slot, args in msg:
            key = (name, slot)
            if key not in self._mapping:
                if args is None:
                    raise RuntimeError(f"{self.name}: no IPC handle for uncached slot {key}")
                self._mapping[key] = rebuild_cuda_tensor(*args)
        self._stage(f"{self.name}.trainer handle open", time.perf_counter() - t)

        t = time.perf_counter()
        out = {name: self._mapping[(name, slot)].to(device, copy=True) for name, slot, _a in msg}
        torch.cuda.synchronize()
        self._stage(f"{self.name}.trainer D->D copy", time.perf_counter() - t)

        self._send_ack(src, group)
        return out

    def _send_ack(self, src, group):
        """ipc-pool acks synchronously. ipc-pipe overrides with isend."""
        dist.send(torch.ones(1, dtype=torch.uint8), dst=src, group=group, tag=_POOL_ACK_TAG)


class IpcPipeTransport(IpcPoolTransport):
    """Opt 2 — ipc-pool plus one-step ack pipelining.

    `n_slots = 2` (double-buffered). The trainer acks with a non-blocking
    `isend`; the engine collects the **previous** step's ack instead of
    this step's, so the ~1 ms ack round-trip overlaps the engine's next
    step instead of stalling its `send()`. Slot s reuse is safe because
    the engine collects ack(s-2) before step s overwrites slot s%2."""

    name = "ipc-pipe"
    n_slots = 2

    def __init__(self):
        super().__init__()
        self._pending = False  # engine: an ack is outstanding
        self._ack_req = None  # trainer: in-flight isend handle
        self._ack_buf = None  # trainer: tensor kept alive for isend

    def _wait_ack(self, dst, group):
        # Deferred: collect the *previous* step's ack, not this one.
        t = time.perf_counter()
        if self._pending:
            ack = torch.zeros(1, dtype=torch.uint8)
            dist.recv(ack, src=dst, group=group, tag=_POOL_ACK_TAG)
        self._stage(f"{self.name}.engine ack wait (deferred)", time.perf_counter() - t)
        self._pending = True

    def _send_ack(self, src, group):
        # Non-blocking: the engine picks this up on its *next* step.
        if self._ack_req is not None:
            self._ack_req.wait()  # previous isend must be consumed first
        self._ack_buf = torch.ones(1, dtype=torch.uint8)
        self._ack_req = dist.isend(self._ack_buf, dst=src, group=group, tag=_POOL_ACK_TAG)

    def flush(self, peer, group, is_engine):
        if is_engine:
            if self._pending:
                ack = torch.zeros(1, dtype=torch.uint8)
                dist.recv(ack, src=peer, group=group, tag=_POOL_ACK_TAG)
                self._pending = False
        else:
            if self._ack_req is not None:
                self._ack_req.wait()
                self._ack_req = None


def _make_transport(arm: str) -> Transport:
    return {
        "gloo": GlooTransport,
        "ipc": IpcTransport,
        "ipc-pool": IpcPoolTransport,
        "ipc-pipe": IpcPipeTransport,
    }[arm]()


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def _single_tensor_spec(mb: float) -> dict:
    """One 2-D bf16 tensor of approximately ``mb`` megabytes."""
    cols = 4096
    rows = max(1, int(mb * 1024 * 1024) // (cols * 2))
    return {"hidden_states": ((rows, cols), torch.bfloat16)}


def _eagle3_spec(tokens: int, hidden: int) -> dict:
    """Realistic Eagle3-shaped multi-tensor hidden-state set."""
    return {
        "hidden_states": ((tokens, hidden), torch.bfloat16),
        "aux_hidden_states": ((tokens, 3 * hidden), torch.bfloat16),
        "last_hidden_states": ((tokens, hidden), torch.bfloat16),
    }


def _spec_bytes(spec: dict) -> int:
    total = 0
    for shape, dtype in spec.values():
        n = 1
        for d in shape:
            n *= d
        total += n * torch.empty(0, dtype=dtype).element_size()
    return total


def _make_payload(spec: dict, device, seed: int, deterministic: bool = False) -> dict:
    """Fresh CUDA tensors — a new allocation every iteration, like real
    hidden states (so the plain `ipc` arm pays a real cudaIpcOpenMemHandle
    every step). ``deterministic`` uses a reproducible arange payload so
    both ranks can recompute it for the byte-equality correctness gate."""
    out = {}
    if deterministic:
        for name, (shape, dtype) in spec.items():
            n = 1
            for d in shape:
                n *= d
            flat = torch.arange(n, device=device, dtype=torch.float32) + float(seed)
            out[name] = flat.reshape(tuple(shape)).to(dtype)
    else:
        g = torch.Generator(device=device).manual_seed(seed)
        for name, (shape, dtype) in spec.items():
            out[name] = torch.randn(tuple(shape), generator=g, device=device).to(dtype)
    torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _stats(samples_s: list) -> dict:
    """mean / p50 / p99 / min in milliseconds from a list of seconds."""
    ms = sorted(s * 1e3 for s in samples_s)
    n = len(ms)
    return {
        "mean": sum(ms) / n,
        "p50": ms[int(0.50 * (n - 1))],
        "p99": ms[int(0.99 * (n - 1))],
        "min": ms[0],
        "n": n,
    }


def _bench_transport(
    transport, spec, *, iters, warmup, rank, device, group, engine_step_ms
) -> dict:
    """Time one transport arm on one payload.

    Returns per-iteration ``span`` (barrier-to-barrier end-to-end) and
    ``own`` (this rank's own send/recv call duration), plus the warm
    per-stage ``stages``. Iteration 0 uses a deterministic payload and is
    byte-equality checked on the trainer; it is also the ``cold`` sample
    (it pays one-time IPC setup). ``warm`` stats use ``[warmup:]``.
    """
    spans, own = [], []
    total = warmup + iters
    for i in range(total):
        # Stages from warmup iterations are dropped — keep only steady state.
        if i == warmup:
            transport.stages.clear()

        deterministic = i == 0
        payload = (
            _make_payload(spec, device, seed=i, deterministic=deterministic)
            if rank == ENGINE_RANK
            else None
        )
        torch.cuda.synchronize()
        dist.barrier(group)

        t0 = time.perf_counter()
        if rank == ENGINE_RANK:
            ts = time.perf_counter()
            transport.engine_send(payload, TRAINER_RANK, group)
            te = time.perf_counter()
        else:
            ts = time.perf_counter()
            got = transport.trainer_recv(spec, ENGINE_RANK, device, group)
            te = time.perf_counter()
            if i == 0:  # byte-equality correctness gate
                ref = _make_payload(spec, device, seed=0, deterministic=True)
                for name in spec:
                    if not torch.equal(got[name], ref[name]):
                        raise RuntimeError(
                            f"{transport.name}: byte mismatch on '{name}' "
                            f"— transport is incorrect, timings void"
                        )
        dist.barrier(group)
        t1 = time.perf_counter()

        spans.append(t1 - t0)
        own.append(te - ts)

        # Inter-step engine pacing (stand-in for the next generate()) —
        # outside the measured window; lets a deferred ack land naturally.
        if rank == ENGINE_RANK and engine_step_ms > 0:
            time.sleep(engine_step_ms / 1e3)

    peer = TRAINER_RANK if rank == ENGINE_RANK else ENGINE_RANK
    transport.flush(peer, group, is_engine=(rank == ENGINE_RANK))
    dist.barrier(group)

    stages = {k: sum(v) / len(v) * 1e3 for k, v in transport.stages.items()}
    return {"transport": transport.name, "span": spans, "own": own, "stages": stages}


# ---------------------------------------------------------------------------
# Per-stage breakdown — gloo + ipc baseline (instrumented replicas)
# ---------------------------------------------------------------------------


def _breakdown(spec, *, iters, rank, device, group) -> dict:
    """Stage-by-stage anatomy of the plain gloo + ipc transports.

    Replicas of NcclHiddenStatesConnector / cuda_ipc.py so each stage can
    be timed individually. The pool/pipe arms self-instrument via their
    own `stages` dict, so they are not replicated here.
    """
    names = sorted(spec)
    acc: dict = {}

    def add(stage, dt):
        acc.setdefault(stage, []).append(dt)

    for i in range(iters):
        payload = _make_payload(spec, device, seed=1000 + i) if rank == ENGINE_RANK else None
        torch.cuda.synchronize()

        # ---- gloo ----
        dist.barrier(group)
        if rank == ENGINE_RANK:
            cpu_tensors = []
            t = time.perf_counter()
            for name in names:
                cpu_tensors.append(payload[name].detach().to("cpu", copy=True).contiguous())
            add("gloo.engine D->H copy", time.perf_counter() - t)
            t = time.perf_counter()
            for tag, ct in enumerate(cpu_tensors):
                dist.send(ct, dst=TRAINER_RANK, group=group, tag=tag)
            add("gloo.engine gloo ship", time.perf_counter() - t)
        else:
            bufs = []
            t = time.perf_counter()
            for tag, name in enumerate(names):
                shape, dtype = spec[name]
                b = torch.empty(tuple(shape), dtype=dtype, device="cpu")
                dist.recv(b, src=ENGINE_RANK, group=group, tag=tag)
                bufs.append(b)
            add("gloo.trainer gloo recv", time.perf_counter() - t)
            t = time.perf_counter()
            dev = [b.to(device) for b in bufs]
            torch.cuda.synchronize()
            add("gloo.trainer H->D copy", time.perf_counter() - t)
            del dev

        # ---- CUDA IPC ----
        dist.barrier(group)
        if rank == ENGINE_RANK:
            t = time.perf_counter()
            args_list = []
            for name in names:
                tt = payload[name].detach()
                if not tt.is_contiguous():
                    tt = tt.contiguous()
                _fn, args = reduce_tensor(tt)
                args_list.append((name, args))
            add("ipc.engine handle export", time.perf_counter() - t)
            t = time.perf_counter()
            blob = pickle.dumps(args_list, protocol=pickle.HIGHEST_PROTOCOL)
            buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
            dist.send(
                torch.tensor([buf.numel()], dtype=torch.long),
                dst=TRAINER_RANK,
                group=group,
                tag=_BREAKDOWN_TAG,
            )
            dist.send(buf, dst=TRAINER_RANK, group=group, tag=_BREAKDOWN_TAG + 1)
            add("ipc.engine ship handles", time.perf_counter() - t)
            t = time.perf_counter()
            ack = torch.zeros(1, dtype=torch.uint8)
            dist.recv(ack, src=TRAINER_RANK, group=group, tag=_BREAKDOWN_TAG + 2)
            add("ipc.engine wait for ack", time.perf_counter() - t)
        else:
            length = torch.empty(1, dtype=torch.long)
            dist.recv(length, src=ENGINE_RANK, group=group, tag=_BREAKDOWN_TAG)
            rbuf = torch.empty(int(length.item()), dtype=torch.uint8)
            dist.recv(rbuf, src=ENGINE_RANK, group=group, tag=_BREAKDOWN_TAG + 1)
            payloads = pickle.loads(rbuf.numpy().tobytes())
            t = time.perf_counter()
            aliases = [rebuild_cuda_tensor(*args) for _name, args in payloads]
            add("ipc.trainer handle open", time.perf_counter() - t)
            t = time.perf_counter()
            cloned = [a.to(device, copy=True) for a in aliases]
            torch.cuda.synchronize()
            add("ipc.trainer D->D copy", time.perf_counter() - t)
            del aliases, cloned
            dist.send(
                torch.ones(1, dtype=torch.uint8),
                dst=ENGINE_RANK,
                group=group,
                tag=_BREAKDOWN_TAG + 2,
            )

    return {stage: sum(v) / len(v) * 1e3 for stage, v in acc.items()}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _worker(rank, world_size, port, argsd, result_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    iters, warmup = argsd["iters"], argsd["warmup"]
    arms = argsd["arms"]
    engine_step_ms = argsd["engine_step_ms"]
    payloads = list(argsd["payloads"])  # [(label, spec)]

    results = []
    for label, spec in payloads:
        per_arm = {}
        for arm in arms:
            transport = _make_transport(arm)
            res = _bench_transport(
                transport,
                spec,
                iters=iters,
                warmup=warmup,
                rank=rank,
                device=device,
                group=group,
                engine_step_ms=engine_step_ms,
            )
            gathered = [None, None]
            dist.all_gather_object(gathered, res)
            per_arm[arm] = gathered
        if rank == TRAINER_RANK:
            results.append((label, spec, per_arm))

    # gloo + ipc stage anatomy on the largest payload (the clearest).
    big_label, big_spec = max(payloads, key=lambda ls: _spec_bytes(ls[1]))
    bd = _breakdown(big_spec, iters=max(8, warmup), rank=rank, device=device, group=group)
    bd_gathered = [None, None]
    dist.all_gather_object(bd_gathered, bd)

    if rank == TRAINER_RANK:
        merged_bd = {}
        for d in bd_gathered:
            merged_bd.update(d)
        report = _build_report(
            results,
            (big_label, big_spec, merged_bd),
            arms=arms,
            iters=iters,
            warmup=warmup,
            engine_step_ms=engine_step_ms,
        )
        Path(result_path).write_text(report)
        print(report)

    dist.barrier(group)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_mb(nbytes: int) -> str:
    mb = nbytes / (1024 * 1024)
    return f"{mb:.2f} MB" if mb < 1 else f"{mb:.0f} MB"


def _warm(vals_s: list, warmup: int) -> dict:
    """Steady-state stats (ms) from the measured (post-warmup) iterations."""
    return _stats(vals_s[warmup:])


def _build_report(results, breakdown, *, arms, iters, warmup, engine_step_ms) -> str:
    try:
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = "unknown"
    mps = "yes" if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") else "no"

    L = []
    L.append("# Colocate transport optimization benchmark — IPC variants")
    L.append("")
    L.append(f"- GPU: **{gpu}**  ·  torch {torch.__version__}  ·  CUDA {torch.version.cuda}")
    L.append(f"- Host: {platform.platform()}  ·  MPS active: {mps}")
    L.append(
        f"- Method: 2 processes on GPU 0, 2-rank gloo group; "
        f"{warmup} warmup + {iters} measured iters; fresh payload "
        f"allocated every iter."
    )
    L.append(f"- Arms: {', '.join(arms)}  ·  inter-step engine pacing: {engine_step_ms} ms")
    L.append(
        "- **cold** = iteration 0 (pays one-time `cudaIpcOpenMemHandle`); "
        "**warm** = mean of the measured iterations (steady state)."
    )
    L.append("- Every arm passed an iteration-0 byte-equality check (else the run aborts).")
    L.append("")

    # -- Table A: end-to-end span ------------------------------------------
    L.append("## End-to-end transfer latency (warm mean, ms)")
    L.append("")
    L.append(
        "Barrier-to-barrier; both ranks see the same window. "
        "Ack pipelining does **not** shrink this number (the barrier "
        "forces the full round-trip into the window) — its win shows "
        "in the engine-send table below."
    )
    L.append("")
    L.append("| Payload | Size | " + " | ".join(arms) + " |")
    L.append("|---|--:|" + "--:|" * len(arms))
    for label, spec, per_arm in results:
        nbytes = _spec_bytes(spec)
        cells = []
        for arm in arms:
            span = per_arm[arm][TRAINER_RANK]["span"]
            cells.append(f"{_warm(span, warmup)['mean']:.3f}")
        L.append(f"| {label} | {_fmt_mb(nbytes)} | " + " | ".join(cells) + " |")
    L.append("")

    # -- Table B: engine send own-call ------------------------------------
    L.append("## Engine `send()` own-call duration (warm mean, ms)")
    L.append("")
    L.append(
        "The colocate-loop stall: how long the engine is blocked "
        "inside the transfer before it can resume. This is the metric "
        "ack pipelining targets."
    )
    L.append("")
    has_ab = "ipc" in arms and "ipc-pipe" in arms
    hdr = "| Payload | " + " | ".join(arms) + " |"
    if has_ab:
        hdr += " ipc→ipc-pipe |"
    L.append(hdr)
    L.append("|---|" + "--:|" * len(arms) + ("--:|" if has_ab else ""))
    for label, spec, per_arm in results:
        cells = []
        warm_by_arm = {}
        for arm in arms:
            own = per_arm[arm][ENGINE_RANK]["own"]
            w = _warm(own, warmup)["mean"]
            warm_by_arm[arm] = w
            cells.append(f"{w:.3f}")
        row = f"| {label} | " + " | ".join(cells) + " |"
        if has_ab:
            spd = warm_by_arm["ipc"] / max(warm_by_arm["ipc-pipe"], 1e-9)
            row += f" **{spd:.1f}×** |"
        L.append(row)
    L.append("")

    # -- Table C: trainer recv own-call -----------------------------------
    L.append("## Trainer `recv()` own-call duration (warm mean, ms)")
    L.append("")
    L.append("| Payload | " + " | ".join(arms) + " |")
    L.append("|---|" + "--:|" * len(arms))
    for label, spec, per_arm in results:
        cells = []
        for arm in arms:
            own = per_arm[arm][TRAINER_RANK]["own"]
            cells.append(f"{_warm(own, warmup)['mean']:.3f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    # -- Table D: cold vs warm on the realistic payload -------------------
    eagle = next((r for r in results if r[0].startswith("Eagle3")), None)
    if eagle is None:
        eagle = max(results, key=lambda r: _spec_bytes(r[1]))
    elabel, espec, eper = eagle
    L.append(f"## Cold vs warm — {elabel} ({_fmt_mb(_spec_bytes(espec))})")
    L.append("")
    L.append(
        "Cold is iteration 0. A large cold→warm drop means the arm "
        "amortizes a one-time cost (the `cudaIpcOpenMemHandle` the "
        "pool/cache arms pay once); a flat arm re-pays it every step."
    )
    L.append("")
    L.append("| Arm | engine cold | engine warm | trainer cold | trainer warm |")
    L.append("|---|--:|--:|--:|--:|")
    for arm in arms:
        eng = eper[arm][ENGINE_RANK]["own"]
        tr = eper[arm][TRAINER_RANK]["own"]
        L.append(
            f"| {arm} | {eng[0] * 1e3:.3f} ms "
            f"| {_warm(eng, warmup)['mean']:.3f} ms "
            f"| {tr[0] * 1e3:.3f} ms "
            f"| {_warm(tr, warmup)['mean']:.3f} ms |"
        )
    L.append("")

    # -- Table E: gloo + ipc stage anatomy --------------------------------
    big_label, big_spec, bd = breakdown
    L.append(
        f"## Stage anatomy — gloo + ipc baseline — {big_label} "
        f"({_fmt_mb(_spec_bytes(big_spec))}, mean ms)"
    )
    L.append("")
    L.append("| Stage | Time |")
    L.append("|---|--:|")
    for stage in sorted(bd):
        L.append(f"| `{stage}` | {bd[stage]:.3f} ms |")
    L.append("")

    # -- Table F: pool / pipe stage anatomy (warm, self-instrumented) -----
    opt_arms = [a for a in arms if a in ("gloo", "ipc-pool", "ipc-pipe")]
    if opt_arms:
        L.append(f"## Stage anatomy — optimization arms — {elabel} (warm mean ms)")
        L.append("")
        L.append("| Arm | Stage | Time |")
        L.append("|---|---|--:|")
        for arm in opt_arms:
            merged = {}
            for rk in (ENGINE_RANK, TRAINER_RANK):
                merged.update(eper[arm][rk].get("stages", {}))
            for stage in sorted(merged):
                L.append(f"| {arm} | `{stage}` | {merged[stage]:.3f} ms |")
        L.append("")

    L.append(
        "> See `docs/colocate/transport_optimization.md` for the "
        "design of each arm and how to read these tables."
    )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--arms",
        default=",".join(ALL_ARMS),
        help=f"comma-separated transport arms ({', '.join(ALL_ARMS)})",
    )
    ap.add_argument("--iters", type=int, default=30, help="measured iterations per arm per payload")
    ap.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="warmup iterations (>=2 so ipc-pipe primes both slots)",
    )
    ap.add_argument(
        "--sizes-mb",
        default="0.25,1,4,16,64,256",
        help="comma-separated single-tensor payload sizes in MB",
    )
    ap.add_argument(
        "--tokens", type=int, default=4096, help="Eagle3 multi-tensor case: number of tokens (B*S)"
    )
    ap.add_argument("--hidden", type=int, default=4096, help="Eagle3 multi-tensor case: hidden dim")
    ap.add_argument(
        "--engine-step-ms",
        type=float,
        default=0.0,
        help="inter-step engine pacing (stand-in for generate()); outside the measured window",
    )
    ap.add_argument("--port", type=int, default=29555, help="rendezvous port")
    ap.add_argument(
        "--out",
        default=str(_REPO_ROOT / "colocate-transport-bench.md"),
        help="Markdown report output path",
    )
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ALL_ARMS]
    if bad:
        print(f"ERROR: unknown arm(s) {bad}; valid: {', '.join(ALL_ARMS)}", file=sys.stderr)
        return 2
    if args.warmup < 2 and "ipc-pipe" in arms:
        print(
            "ERROR: --warmup must be >=2 when ipc-pipe is selected (it primes 2 pool slots).",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print("ERROR: this benchmark needs a CUDA GPU.", file=sys.stderr)
        return 2
    for ev in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if "expandable" in os.environ.get(ev, ""):
            print(
                f"WARNING: {ev}={os.environ[ev]!r} — CUDA IPC needs plain "
                f"cudaMalloc memory and will fail. Unset it.",
                file=sys.stderr,
            )

    payloads = [
        (f"single {s.strip()} MB", _single_tensor_spec(float(s)))
        for s in args.sizes_mb.split(",")
        if s.strip()
    ]
    payloads.append(
        (
            f"Eagle3 ({args.tokens}t × {args.hidden}h, 3 tensors)",
            _eagle3_spec(args.tokens, args.hidden),
        )
    )

    argsd = {
        "iters": args.iters,
        "warmup": args.warmup,
        "payloads": payloads,
        "arms": arms,
        "engine_step_ms": args.engine_step_ms,
    }
    print(
        f"Benchmarking arms [{', '.join(arms)}] over {len(payloads)} "
        f"payloads, {args.warmup}+{args.iters} iters each, on "
        f"{torch.cuda.get_device_name(0)} …\n"
    )
    mp.spawn(_worker, args=(2, args.port, argsd, args.out), nprocs=2, join=True)
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""CUDA IPC zero-copy hidden-state transport for colocate mode.

This is the **default** colocate hidden-state transport. The engine
exports a CUDA IPC handle for each hidden-state tensor (via PyTorch's
``torch.multiprocessing`` reduction machinery), ships the small handle
blobs over the gloo channel, and the trainer maps the engine's GPU
memory directly and does a single on-device D->D copy into its own
buffer. No host round-trip.

The fallback is the gloo CPU-staged transport: the engine does a D->H
copy, ships the bytes over the gloo ``meta_group``, and the trainer
does an H->D copy — two PCIe-class copies per tensor per step. Both
processes share the *same physical GPU* under MPS, so that host
round-trip is pure overhead (the data never needs to leave the device)
— which is exactly what this IPC path eliminates.

Default & opt-out
-----------------
CUDA IPC is **on by default**, layered on top of ``transfer_mode=nccl``
(it replaces only the gloo transport, not the union-world bootstrap).
Set ``TORCHSPEC_COLOCATE_IPC=0`` to fall back to the gloo CPU-staged
transport. Both the engine connector and the trainer fetcher read the
*same* env var, so the two sides always agree on the transport without
a runtime negotiation message.

The ``expandable_segments`` conflict
------------------------------------
CUDA IPC has two memory-sharing paths. For plain ``cudaMalloc`` memory
it uses the classic ``cudaIpcGetMemHandle`` / ``cudaIpcOpenMemHandle``
handles, which work in any container. For the virtual-memory segments
produced by ``expandable_segments:True`` it instead passes the backing
fd between processes via the ``pidfd_getfd`` syscall — and that needs
``CAP_SYS_PTRACE``, which typical containers (RunPod, most Docker
hosts) do not grant, so ``rebuild_cuda_tensor`` fails with
``pidfd_getfd: Operation not permitted``.

Resolution: when ``TORCHSPEC_COLOCATE_IPC`` is opted in, the colocate
path (``ray/train_group.py``, ``inference/factory.py``) **does not**
inject ``expandable_segments`` into the trainer/engine actors, so IPC
stays on the capability-free classic-handle path. (IPC already avoids
the H<->D staging churn that ``expandable_segments`` was mitigating.)

:func:`probe_ipc_capability` runs a **non-destructive** capability check
at construction (it does *not* share a CUDA tensor — a ``reduce_tensor``
smoke test wedges CUDA under MPS; see that function's docstring). The
connector/fetcher **fail fast** with an actionable message if IPC is
unavailable, rather than silently falling back (a one-sided fallback
would desync the wire protocol).

Wire protocol
-------------
Per step, engine -> trainer over the gloo group:

  1. engine: ``send_object_list([[(name, ipc_args), ...]])`` — the
     pickled IPC handle blobs, in ``sorted(name)`` order.
  2. trainer: ``recv_object_list`` -> rebuild each tensor as an alias of
     the engine's memory -> ``.clone()`` into a trainer-owned buffer ->
     ``cuda.synchronize()``.
  3. trainer: send a 1-byte ack back.
  4. engine: block on the ack before returning from ``send`` — this
     keeps the engine's (sglang-owned) hidden-state tensors alive until
     the trainer has finished copying, exactly like the blocking gloo
     ``send`` it replaces.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

_IPC_ENV = "TORCHSPEC_COLOCATE_IPC"

# Env values that disable IPC and fall back to the gloo transport.
_IPC_DISABLE_VALUES = ("0", "false", "no", "off")

# Opt-in flag for the pipelined transport (send-buffer pool + one-step
# ack pipelining — see :class:`IpcPipelineTransport`). Layered on top of
# CUDA IPC; default off, so the plain per-step ipc_send / ipc_recv path
# is unchanged unless this is explicitly set.
_IPC_PIPELINE_ENV = "TORCHSPEC_COLOCATE_IPC_PIPELINE"
_IPC_PIPELINE_ENABLE_VALUES = ("1", "true", "yes", "on")

# Cached (ok, reason) from the one-time capability probe.
_probe_cache: Optional[Tuple[bool, str]] = None


def ipc_enabled() -> bool:
    """True iff the CUDA IPC zero-copy transport is selected.

    CUDA IPC is the **default** colocate hidden-state transport. Any
    value of ``TORCHSPEC_COLOCATE_IPC`` other than an explicit disable
    token (``0`` / ``false`` / ``no`` / ``off``) — including the var
    being unset — selects it. Set one of those tokens to fall back to
    the gloo CPU-staged transport.
    """
    return os.environ.get(_IPC_ENV, "").strip().lower() not in _IPC_DISABLE_VALUES


def ipc_pipeline_enabled() -> bool:
    """True iff the pipelined CUDA IPC transport is selected.

    Opt-in via ``TORCHSPEC_COLOCATE_IPC_PIPELINE`` (``1`` / ``true`` /
    ``yes`` / ``on``). The pipelined path (:class:`IpcPipelineTransport`)
    is layered *on top of* CUDA IPC — a send-buffer pool plus one-step ack
    deferral — so it is only active when IPC itself is enabled
    (:func:`ipc_enabled`). Default off: with the flag unset, the engine
    connector and trainer fetcher use the plain per-step :func:`ipc_send`
    / :func:`ipc_recv` path, unchanged.

    See ``docs/colocate/transport_optimization.md`` (Opt 1 + Opt 2): the
    pool gives the engine a stable IPC handle so the trainer skips the
    per-step ``cudaIpcOpenMemHandle``, and the one-step ack deferral
    lifts the ~1 ms ack round-trip off the engine's critical path
    (MPS-measured 3.9x on the realistic Eagle3 engine-``send()`` stall).
    """
    if not ipc_enabled():
        return False
    return os.environ.get(_IPC_PIPELINE_ENV, "").strip().lower() in _IPC_PIPELINE_ENABLE_VALUES


def probe_ipc_capability() -> Tuple[bool, str]:
    """Probe whether CUDA IPC can be used on this process.

    Returns ``(ok, reason)``. Cached after the first call.

    This is a **non-destructive** check. It deliberately does *not* run a
    ``reduce_tensor`` smoke test: sharing a CUDA tensor via IPC and then
    immediately discarding it (no consumer ever maps it) leaves PyTorch's
    CUDA-IPC producer-side machinery in a state that wedges subsequent
    CUDA work **under MPS** -- the engine's sglang forward hangs.
    (Diagnosed 2026-05-21 on 1xH100: the probe, not the transport, caused
    the colocate IPC hang; skipping it makes the full IPC path pass.)

    The only capability that matters for the classic, container-friendly
    CUDA IPC handle path is that memory is **not** ``expandable_segments``
    (those force the ``pidfd_getfd`` path, which needs ``CAP_SYS_PTRACE``).
    The colocate path already guarantees this -- ``inference/factory.py``
    and ``ray/train_group.py`` skip the ``expandable_segments`` allocator
    config whenever IPC is the transport -- so a config check suffices.
    """
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache
    try:
        import torch

        if not torch.cuda.is_available():
            _probe_cache = (False, "CUDA not available")
            return _probe_cache
        for _ev in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            if "expandable_segments:true" in os.environ.get(_ev, "").lower():
                _probe_cache = (
                    False,
                    (
                        _ev + " enables expandable_segments, which forces CUDA "
                        "IPC onto the pidfd_getfd path (needs CAP_SYS_PTRACE). "
                        "Drop expandable_segments, or set TORCHSPEC_COLOCATE_IPC=0 "
                        "for the gloo CPU-staged transport."
                    ),
                )
                return _probe_cache
        _probe_cache = (True, "ok")
    except Exception as e:  # pragma: no cover - needs a real GPU
        _probe_cache = (False, repr(e))
    return _probe_cache


def ensure_ipc_usable() -> None:
    """Raise a clear error if IPC (the default transport) is not usable.

    Called once at connector/fetcher construction. Both sides run the
    same check on the same platform, so they fail (or pass) together.
    """
    ok, reason = probe_ipc_capability()
    if not ok:
        raise RuntimeError(
            f"CUDA IPC is the default colocate hidden-state transport but "
            f"is not usable on this host: {reason}. Set "
            f"TORCHSPEC_COLOCATE_IPC=0 to fall back to the gloo CPU-staged "
            f"transport."
        )


def _reset_probe_cache_for_test() -> None:
    """Test hook: clear the cached probe result."""
    global _probe_cache
    _probe_cache = None


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

# Distinct tags for the three point-to-point messages of one transfer.
# The payload is shipped as plain dist.send/recv of byte tensors — the
# same primitive the gloo CPU-staged path uses (proven on the union
# meta_group). The send_object_list / recv_object_list helpers were
# observed to deadlock on this group, so we pickle + frame the blob
# ourselves.
_IPC_LEN_TAG = 7001
_IPC_DATA_TAG = 7002
_IPC_ACK_TAG = 7003


def ipc_send(
    tensors: Dict[str, "torch.Tensor"],  # noqa: F821
    dst: int,
    group,
) -> None:
    """Engine side: ship hidden-state tensors to ``dst`` via CUDA IPC.

    Blocks until the trainer acks (i.e. has cloned the data), so the
    caller's tensors stay valid for the whole transfer — same contract
    as the blocking gloo ``dist.send`` this replaces.
    """
    import pickle

    import torch
    import torch.distributed as dist
    from torch.multiprocessing.reductions import reduce_tensor

    names = sorted(tensors.keys())
    keepalive = []  # hold contiguous copies alive until the ack
    payloads = []
    for name in names:
        t = tensors[name].detach()
        if t.device.type != "cuda":
            raise ValueError(f"cuda_ipc.ipc_send requires CUDA tensors; '{name}' is on {t.device}")
        if not t.is_contiguous():
            t = t.contiguous()
        keepalive.append(t)
        # reduce_tensor returns (rebuild_cuda_tensor, args); only the
        # args tuple needs to travel — the receiver knows the rebuild fn.
        _rebuild_fn, args = reduce_tensor(t)
        payloads.append((name, args))

    # Pickle the IPC-handle payloads and ship as a length-framed byte
    # tensor via plain dist.send (the gloo path's proven primitive).
    blob = bytearray(pickle.dumps(payloads, protocol=pickle.HIGHEST_PROTOCOL))
    buf = torch.frombuffer(blob, dtype=torch.uint8)
    length = torch.tensor([buf.numel()], dtype=torch.long)
    dist.send(length, dst=dst, group=group, tag=_IPC_LEN_TAG)
    dist.send(buf, dst=dst, group=group, tag=_IPC_DATA_TAG)

    # Block until the trainer has cloned the data out of our memory.
    ack = torch.zeros(1, dtype=torch.uint8)
    dist.recv(ack, src=dst, group=group, tag=_IPC_ACK_TAG)
    del keepalive, blob


def ipc_recv(
    tensor_specs: Dict[str, Tuple],
    src: int,
    device: "torch.device",  # noqa: F821
    group,
) -> Dict[str, "torch.Tensor"]:  # noqa: F821
    """Trainer side: receive hidden-state tensors from ``src`` via CUDA IPC.

    Maps the engine's GPU memory, copies (D->D, on-device) into
    trainer-owned buffers, then acks. ``tensor_specs`` is used only to
    validate the received key set — the shapes/dtypes ride along inside
    the IPC payload.
    """
    import pickle

    import torch
    import torch.distributed as dist
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    # Receive the length-framed pickled payload (mirrors ipc_send).
    length = torch.empty(1, dtype=torch.long)
    dist.recv(length, src=src, group=group, tag=_IPC_LEN_TAG)
    buf = torch.empty(int(length.item()), dtype=torch.uint8)
    dist.recv(buf, src=src, group=group, tag=_IPC_DATA_TAG)
    payloads = pickle.loads(buf.numpy().tobytes())
    if not isinstance(payloads, list):
        raise RuntimeError(f"cuda_ipc.ipc_recv: expected a list payload, got {type(payloads)}")

    out: Dict[str, torch.Tensor] = {}
    aliases = []  # keep IPC aliases alive until the post-clone sync
    for name, args in payloads:
        alias = rebuild_cuda_tensor(*args)
        aliases.append(alias)
        # D->D copy into trainer-owned (normal) memory on `device`.
        out[name] = alias.to(device, copy=True)

    # The clones above are async on the current stream; finish them
    # before we drop the aliases and ack (after which the engine may
    # free its memory).
    torch.cuda.synchronize()
    del aliases

    expected = set(tensor_specs.keys())
    got = set(out.keys())
    if expected != got:
        raise RuntimeError(
            f"cuda_ipc.ipc_recv: key mismatch — expected {sorted(expected)}, got {sorted(got)}"
        )

    ack = torch.ones(1, dtype=torch.uint8)
    dist.send(ack, dst=src, group=group, tag=_IPC_ACK_TAG)
    return out


# ---------------------------------------------------------------------------
# Pipelined transport — send-buffer pool + one-step ack pipelining
# ---------------------------------------------------------------------------
#
# This is the optimized counterpart to the plain ipc_send / ipc_recv pair
# above, selected by `TORCHSPEC_COLOCATE_IPC_PIPELINE=1`
# (:func:`ipc_pipeline_enabled`). Unlike the stateless functions, it must
# carry state across steps (the pool, the trainer's handle cache, the
# deferred ack), so it is a class — one long-lived instance per connector
# (engine role) / fetcher (trainer role).
#
# Wire tags are kept distinct from the plain path's 7001-7003 so the two
# protocols can never collide if both happen to be linked into a process.
_PIPE_LEN_TAG = 7011
_PIPE_DATA_TAG = 7012
_PIPE_ACK_TAG = 7013

# Double-buffered: slot s is reused every _PIPELINE_SLOTS steps. K=2 is
# the minimum that lets the engine defer one ack — step N writes slot
# N % 2 while step N-1's ack (slot (N-1) % 2) is still in flight.
_PIPELINE_SLOTS = 2


def _send_pickle(obj, dst, group, len_tag: int, data_tag: int) -> None:
    """Ship a picklable object as a length-framed byte tensor over gloo.

    Mirrors :func:`ipc_send`'s framing — ``send_object_list`` was observed
    to deadlock on the colocate gloo group, so we pickle + frame by hand.
    """
    import pickle

    import torch
    import torch.distributed as dist

    blob = bytearray(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    buf = torch.frombuffer(blob, dtype=torch.uint8)
    dist.send(torch.tensor([buf.numel()], dtype=torch.long), dst=dst, group=group, tag=len_tag)
    dist.send(buf, dst=dst, group=group, tag=data_tag)


def _recv_pickle(src, group, len_tag: int, data_tag: int):
    """Inverse of :func:`_send_pickle`."""
    import pickle

    import torch
    import torch.distributed as dist

    length = torch.empty(1, dtype=torch.long)
    dist.recv(length, src=src, group=group, tag=len_tag)
    buf = torch.empty(int(length.item()), dtype=torch.uint8)
    dist.recv(buf, src=src, group=group, tag=data_tag)
    return pickle.loads(buf.numpy().tobytes())


class IpcPipelineTransport:
    """Stateful pipelined CUDA IPC transport — pool + one-step ack deferral.

    The plain :func:`ipc_send` / :func:`ipc_recv` pair is stateless: every
    step exports a fresh IPC handle and the engine blocks on the trainer's
    ack inside ``send()``. This class is the optimized alternative
    (``TORCHSPEC_COLOCATE_IPC_PIPELINE=1``) — it carries state across
    steps and implements both protocol-level optimizations from
    ``docs/colocate/transport_optimization.md``:

    * **Send-buffer pool (Opt 1).** The engine owns ``K = 2`` persistent
      CUDA buffers per tensor name. Each step it copies the engine's
      transient hidden states into ``pool[step % K]`` (one D->D copy) and
      exports that *pooled* buffer's IPC handle. Pool buffers have stable
      device pointers, so their handle args are computed **once** and the
      trainer opens each handle (``cudaIpcOpenMemHandle``) **once**,
      caching the mapping for every later step.

    * **Ack pipelining (Opt 2).** The trainer acks with a non-blocking
      ``isend``; the engine collects the *previous* step's ack instead of
      this step's, so the ~1 ms ack round-trip overlaps the engine's next
      forward instead of stalling ``send()``. ``K = 2`` guarantees step N
      never lands in the slot whose step N-1 ack is still outstanding.

    Variable ``seq_len`` is handled by sizing each pool buffer to the
    largest payload seen so far (grow-to-fit, sized *exactly* — no x2
    overshoot, which on a memory-tight config stacks unaffordably with
    sglang's KV cache). A resize re-exports that slot's handle (the
    trainer re-opens it once) and *retires* the old buffer; the retired
    buffer is freed one step later, the moment the trainer acks the
    resize step — by then it has re-opened the new handle and dropped its
    mapping of the old one, so the free can never race a live mapping and
    a variable-``seq_len`` run does not accumulate dead pool buffers.

    **Teardown is drain-safe without an explicit flush.** The engine never
    blocks on the final ack: an un-collected ack would only matter to
    guard a step N+2 that never happens, and the trainer has already
    ``cuda.synchronize()``-d its copy before sending it, so the engine
    freeing its pool on exit cannot corrupt anything. The trainer waits
    its previous ``isend`` before each new one, so at most one 1-byte ack
    is ever in flight. :meth:`flush` waits that last ``isend`` for a tidy
    teardown; skipping it is harmless.

    One instance per :class:`NcclHiddenStatesConnector` (``role="engine"``)
    or :class:`NcclMultiTensorFetcher` (``role="trainer"``). The class has
    no torchspec-internal imports so the transport benchmark
    (``scripts/colocate/bench_transport.py``) can load this module
    standalone.
    """

    def __init__(self, role: str):
        if role not in ("engine", "trainer"):
            raise ValueError(
                f"IpcPipelineTransport role must be 'engine' or 'trainer', got {role!r}"
            )
        self.role = role
        self._step = 0
        # -- engine-role state --------------------------------------------
        self._pool: Dict[str, list] = {}  # name -> [K] flat CUDA buffers
        self._pool_args: Dict[str, list] = {}  # name -> [K] reduce_tensor args
        self._shipped: set = set()  # (name, slot) handles shipped
        self._retired: list = []  # [(step, buf)] awaiting free
        self._pending_ack = False  # a deferred ack is outstanding
        # -- trainer-role state -------------------------------------------
        self._mapping: Dict[tuple, "torch.Tensor"] = {}  # noqa: F821
        self._ack_req = None  # in-flight ack isend handle
        self._ack_buf = None  # tensor kept alive for the isend

    # -- engine ------------------------------------------------------------

    def _ensure_slot(self, name: str, slot: int, numel: int, dtype, reduce_tensor) -> None:
        """Make ``pool[name][slot]`` exactly big enough for ``numel`` elements.

        Allocates on first use; on overflow reallocates to exactly
        ``numel`` (grow-to-fit, no overshoot) and retires the old buffer
        tagged with the current step — :meth:`engine_send` frees it once
        the trainer acks that step. A (re)allocation drops the slot from
        ``_shipped`` so the next send re-exports the handle.
        """
        import torch

        bufs = self._pool.get(name)
        if bufs is None:
            bufs = [None] * _PIPELINE_SLOTS
            self._pool[name] = bufs
            self._pool_args[name] = [None] * _PIPELINE_SLOTS
        buf = bufs[slot]
        if buf is not None and buf.numel() >= numel and buf.dtype == dtype:
            return
        if buf is not None:
            # Retire (tagged with the current step) rather than free now:
            # the trainer may still hold an IPC mapping of the old buffer
            # until it processes this step's re-ship. engine_send frees it
            # once the trainer acks this step (CUDA IPC UB otherwise).
            self._retired.append((self._step, buf))
        # Exact size — no x2 overshoot. The overshoot is unaffordable on a
        # memory-tight config (it stacks with sglang's KV cache); grow-to-
        # fit still holds, we only reallocate on a genuine new seq_len high.
        new_buf = torch.empty(numel, dtype=dtype, device="cuda")
        bufs[slot] = new_buf
        self._pool_args[name][slot] = reduce_tensor(new_buf)[1]
        self._shipped.discard((name, slot))

    def engine_send(
        self,
        tensors: Dict[str, "torch.Tensor"],  # noqa: F821
        dst: int,
        group,
    ) -> None:
        """Engine side: ship hidden-state tensors to ``dst`` (pipelined).

        Returns as soon as the handle message is on the wire — the ack of
        *this* step is collected at the start of the *next* call (or by
        :meth:`flush`). Same lifetime contract as :func:`ipc_send`: the
        caller's tensors are fully consumed (copied into the pool) before
        this returns, so sglang is free to reuse them immediately.
        """
        import torch
        import torch.distributed as dist
        from torch.multiprocessing.reductions import reduce_tensor

        if self.role != "engine":
            raise RuntimeError("engine_send called on a trainer-role transport")
        if not tensors:
            raise ValueError("IpcPipelineTransport.engine_send requires at least one tensor")

        slot = self._step % _PIPELINE_SLOTS
        msg = []
        for name in sorted(tensors.keys()):
            t = tensors[name].detach()
            if t.device.type != "cuda":
                raise ValueError(
                    f"IpcPipelineTransport requires CUDA tensors; '{name}' is on {t.device}"
                )
            flat = t.reshape(-1)
            numel = flat.numel()
            self._ensure_slot(name, slot, numel, t.dtype, reduce_tensor)
            self._pool[name][slot][:numel].copy_(flat)
            key = (name, slot)
            if key in self._shipped:
                ship_args = None
            else:
                ship_args = self._pool_args[name][slot]
                self._shipped.add(key)
            msg.append((name, slot, tuple(t.shape), numel, ship_args))

        # The trainer reads pool[slot] on its own stream; make the copy
        # device-complete before we signal so the bytes are settled.
        torch.cuda.synchronize()
        _send_pickle(msg, dst, group, _PIPE_LEN_TAG, _PIPE_DATA_TAG)

        # Ack pipelining: collect the *previous* step's ack, not this one.
        if self._pending_ack:
            ack = torch.zeros(1, dtype=torch.uint8)
            dist.recv(ack, src=dst, group=group, tag=_PIPE_ACK_TAG)
            # ack(self._step-1) is in hand: the trainer has finished that
            # step, including re-opening any handle resized at or before
            # it and dropping its old IPC alias. Free pool buffers retired
            # then so a variable-seq_len run does not accumulate dead ones.
            acked = self._step - 1
            self._retired = [(s, b) for (s, b) in self._retired if s > acked]
        self._pending_ack = True
        self._step += 1

    # -- trainer -----------------------------------------------------------

    def trainer_recv(
        self, tensor_specs: Dict[str, Tuple], src: int, device, group
    ) -> Dict[str, "torch.Tensor"]:  # noqa: F821
        """Trainer side: receive one step's tensors from ``src`` (pipelined).

        Opens each pooled IPC handle only on the first step that uses its
        slot (or after an engine-side resize); every other step reuses the
        cached mapping and just does the per-step D->D copy. Acks with a
        non-blocking ``isend`` the engine collects on its next step.
        """
        import torch
        import torch.distributed as dist
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        if self.role != "trainer":
            raise RuntimeError("trainer_recv called on an engine-role transport")

        msg = _recv_pickle(src, group, _PIPE_LEN_TAG, _PIPE_DATA_TAG)
        if not isinstance(msg, list):
            raise RuntimeError(
                f"IpcPipelineTransport.trainer_recv: expected a list payload, got {type(msg)}"
            )

        out: Dict[str, torch.Tensor] = {}
        for name, slot, shape, numel, ship_args in msg:
            key = (name, slot)
            if ship_args is not None:
                # First use of this slot, or the engine resized it — open
                # the handle and (re)cache the mapping. The old alias, if
                # any, is dropped here; its engine buffer is retired (not
                # freed) so this is safe.
                self._mapping[key] = rebuild_cuda_tensor(*ship_args)
            elif key not in self._mapping:
                raise RuntimeError(
                    f"IpcPipelineTransport.trainer_recv: no cached IPC "
                    f"mapping for {key} and the engine shipped no handle"
                )
            flat = self._mapping[key]
            out[name] = flat[:numel].view(shape).to(device, copy=True)

        # Finish the D->D copies before we ack — after the ack the engine
        # may reuse this slot.
        torch.cuda.synchronize()

        expected = set(tensor_specs.keys())
        got = set(out.keys())
        if expected != got:
            raise RuntimeError(
                f"IpcPipelineTransport.trainer_recv: key mismatch — expected "
                f"{sorted(expected)}, got {sorted(got)}"
            )

        # Non-blocking ack — the engine picks it up on its next step. Wait
        # the previous isend first so at most one is ever in flight.
        if self._ack_req is not None:
            self._ack_req.wait()
        self._ack_buf = torch.ones(1, dtype=torch.uint8)
        self._ack_req = dist.isend(self._ack_buf, dst=src, group=group, tag=_PIPE_ACK_TAG)
        self._step += 1
        return out

    # -- teardown ----------------------------------------------------------

    def flush(self) -> None:
        """Drain in-flight pipelined state for a tidy teardown.

        Trainer: wait the last outstanding ack ``isend``. Engine: drop any
        buffers still on the retired list (their final ack is never
        collected — see the class docstring on teardown-safety). Idempotent;
        safe to call any number of times, or not at all.
        """
        if self.role == "trainer" and self._ack_req is not None:
            self._ack_req.wait()
            self._ack_req = None
            self._ack_buf = None
        if self.role == "engine":
            self._retired.clear()

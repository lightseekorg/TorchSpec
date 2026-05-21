# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""NCCL P2P data fetcher for colocate mode (Phases 3 & 4).

This is the trainer-side counterpart to the engine's hidden-state writer.
Whereas the disaggregated path goes engine → Mooncake store → trainer
(``MooncakeDataFetcher``), the colocate path is engine → NCCL P2P send →
trainer recv into a pre-allocated buffer on the same physical GPU.

Phase 3 ships the minimal single-tensor primitive:

    NcclDataFetcher(
        src_rank=engine_rank,
        shape=(B_eng_per_tp, S, H),
        dtype=torch.bfloat16,
        device=torch.device('cuda'),
    )
    tensor = fetcher.recv()

Phase 4 ships the generalised multi-tensor receiver,
:class:`NcclMultiTensorFetcher`, which assembles a Mooncake-shaped
batch dict (``hidden_states``, ``aux_hidden_states``,
``last_hidden_states``, ``target_logits`` … the exact key set is
draft-model-dependent) and pulls per-step CPU-side metadata
(``input_ids``, ``packed_loss_mask``) from a Ray queue. The trainer's
``_train_step`` consumes batches identically whether they came from the
Mooncake or NCCL fetcher.

Wire protocol
-------------

The engine and trainer agree on the per-step ``Dict[str, Tensor]`` key
set via the metadata channel (a Ray queue carrying
:class:`torchspec.training.data_fetcher.ColocateTrainSample`). Both sides
send/recv tensors in **sorted-by-key** order (see
``NcclHiddenStatesConnector.sorted_tensor_names``). All tensor ops for
one step happen in a single ``dist.batch_isend_irecv`` to avoid the
lazy 2-rank sub-communicator pathology that bit Phase 3.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from torchspec.colocate.cuda_ipc import (
    IpcPipelineTransport,
    ensure_ipc_usable,
    ipc_enabled,
    ipc_pipeline_enabled,
    ipc_recv,
)

logger = logging.getLogger("torchspec.training.nccl_data_fetcher")


class NcclDataFetcher:
    """Single-tensor NCCL P2P receiver with a pre-allocated buffer.

    Args:
        src_rank: Global rank to receive from (the paired engine rank in
            the union world).
        shape: Tensor shape to allocate. Must match exactly what the
            sender sends or NCCL will silently corrupt / hang.
        dtype: Tensor dtype.
        device: CUDA device to allocate on. Must be a real CUDA device
            because NCCL refuses CPU tensors.
        group: Optional ``ProcessGroup`` to use; defaults to the world
            (default PG). Tests pass a subgroup; production passes the
            union world's default PG.
        clone_on_return: If ``True`` (default), ``recv()`` returns a
            ``buffer.clone()`` so the caller can mutate freely. If
            ``False``, returns the buffer itself; the caller must finish
            using it before the next ``recv()`` call.
    """

    def __init__(
        self,
        src_rank: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
        clone_on_return: bool = True,
    ):
        if device.type != "cuda":
            raise ValueError(f"NcclDataFetcher requires a CUDA device; got device={device}")

        self._src_rank = int(src_rank)
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = device
        self._group = group
        self._clone = bool(clone_on_return)

        # Pre-allocate the recv buffer. Phase 6 will verify that this
        # allocation lives in expandable_segments territory so it
        # doesn't fragment the pool.
        self._buffer = torch.empty(self._shape, dtype=self._dtype, device=self._device)

        logger.debug(
            "NcclDataFetcher initialised: src_rank=%d shape=%s dtype=%s device=%s "
            "clone_on_return=%s",
            self._src_rank,
            self._shape,
            self._dtype,
            self._device,
            self._clone,
        )

    @property
    def buffer_shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def src_rank(self) -> int:
        return self._src_rank

    def recv(self) -> torch.Tensor:
        """Block on a single P2P recv from ``src_rank``.

        Uses ``dist.batch_isend_irecv`` rather than ``dist.recv`` because
        unbatched send/recv on a large parent group serialises through
        NCCL's lazy 2-rank sub-communicator init, which can deadlock
        across multiple pairs (PyTorch warns
        ``ProcessGroupNCCL.cpp:4004``). Batched P2P is its own primitive
        class and always handled correctly by NCCL.

        Returns:
            The received tensor (a clone by default; the underlying
            buffer if ``clone_on_return=False``).
        """
        op = dist.P2POp(dist.irecv, self._buffer, peer=self._src_rank, group=self._group)
        works = dist.batch_isend_irecv([op])
        for work in works:
            work.wait()
        return self._buffer.clone() if self._clone else self._buffer


def make_dummy_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic dummy tensor used as the Phase 3 send payload.

    Uses ``torch.arange`` rather than ``torch.rand`` so byte-equality is
    well-defined (no RNG state to coordinate). The optional ``seed``
    offsets every element so successive iterations send distinct payloads
    — that catches a class of bugs where the receiver "passes" simply
    because the buffer didn't change between iterations.
    """
    n = 1
    for d in shape:
        n *= d
    flat = torch.arange(n, device=device, dtype=torch.float32) + float(seed)
    return flat.reshape(shape).to(dtype)


def send_dummy(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    dst_rank: int,
    *,
    seed: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Engine-side helper that builds a deterministic tensor and sends it.

    Mirrors ``NcclDataFetcher.recv``: uses batched P2P to side-step the
    lazy-init pathology of unbatched send on large parent groups.

    Returns the tensor it sent (so a caller can keep it alive until the
    receive completes if they care to verify locally).
    """
    tensor = make_dummy_tensor(shape, dtype=dtype, device=device, seed=seed)
    op = dist.P2POp(dist.isend, tensor, peer=dst_rank, group=group)
    works = dist.batch_isend_irecv([op])
    for work in works:
        work.wait()
    return tensor


# ----------------------------------------------------------------------
# Phase 4: multi-tensor receiver + iterator over Ray queue of metadata.
# ----------------------------------------------------------------------


# Public type alias for what a per-tensor specification looks like on the
# wire. The metadata channel carries one of these per tensor name; both
# engine and trainer use it to know shape/dtype before the P2P call.
TensorSpec = Tuple[Tuple[int, ...], torch.dtype]


def _sorted_tensor_names(specs: Dict[str, TensorSpec]) -> List[str]:
    """Canonical send/recv ordering: sorted by key.

    Mirrored in ``torchspec.inference.engine.nccl_hidden_states_connector``.
    The two sides never exchange the order explicitly; agreeing on
    ``sorted(keys)`` removes a class of bugs where a dict-ordering
    difference between Python versions / HF model configs would cause
    silent data corruption.
    """
    return sorted(specs.keys())


def _normalise_dtype(dtype: Any) -> torch.dtype:
    """Accept either a ``torch.dtype`` or a string from the metadata channel.

    The metadata channel runs over Ray queues, which serialise via
    cloudpickle. ``torch.dtype`` survives cloudpickle but
    ``Mooncake``-shaped metadata sometimes carries dtypes as strings
    (``"bfloat16"``, ``"torch.bfloat16"``); we accept both for symmetry
    with :class:`MooncakeDataFetcher`.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype.replace("torch.", ""))
    raise TypeError(f"unsupported tensor dtype representation: {dtype!r} (type={type(dtype)})")


def _group_is_gloo(group: Optional[dist.ProcessGroup]) -> bool:
    """True iff ``group`` (or the default PG) uses the gloo backend.

    The colocate path runs the transfer over a gloo group: trainer and
    engine share one physical GPU, and NCCL refuses to form a
    communicator with two ranks on the same device ("Duplicate GPU
    detected"). gloo stages through host memory, so colocate uses it
    for the engine→trainer P2P.
    """
    try:
        return str(dist.get_backend(group)).lower() == "gloo"
    except Exception:
        return False


class NcclMultiTensorFetcher:
    """Trainer-side multi-tensor receiver for the colocate path.

    One fetcher per trainer rank (= one per paired engine TP rank). The
    fetcher exposes a single method, :meth:`recv_step`, that:

      1. Receives the per-step ``Dict[str, Tensor]`` from the paired
         engine via a single ``batch_isend_irecv``.
      2. Returns a Mooncake-shaped batch dict, with optional CPU-side
         metadata (loss mask, input_ids) merged in by the caller.

    The tensor list and shapes change every step (variable seq_len), so
    we don't pre-allocate buffers. Phase 6 will revisit this if memory
    churn shows up in the stability test.

    Args:
        src_global_rank: Global rank to receive from (the paired engine
            in the union world).
        device: CUDA device to allocate recv buffers on.
        group: Process group; defaults to the default (union world).

    Raises:
        RuntimeError: torch.distributed not initialised.
        ValueError: ``device`` is not a CUDA device.
    """

    def __init__(
        self,
        src_global_rank: int,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
    ):
        if not dist.is_initialized():
            raise RuntimeError(
                "NcclMultiTensorFetcher requires torch.distributed to be "
                "initialised (call init_union_world first)."
            )
        if device.type != "cuda":
            raise ValueError(f"NcclMultiTensorFetcher requires a CUDA device; got {device}")
        self._src = int(src_global_rank)
        self._device = device
        self._group = group
        # CUDA IPC transport — the default; must match the engine
        # connector. Fail fast at construction if it is unusable.
        self._use_ipc = ipc_enabled() and _group_is_gloo(self._group)
        # Pipelined transport (pool + ack pipelining): opt-in, stateful.
        # Must agree with the engine connector — both read the same env.
        self._pipeline: Optional[IpcPipelineTransport] = None
        if self._use_ipc:
            ensure_ipc_usable()
            if ipc_pipeline_enabled():
                self._pipeline = IpcPipelineTransport(role="trainer")

    @property
    def src_global_rank(self) -> int:
        return self._src

    def flush(self) -> None:
        """Drain the pipelined transport at loop teardown.

        No-op unless the pipelined transport is active. Waits the last
        outstanding ack ``isend``; the pipeline is drain-safe without it
        (see :class:`IpcPipelineTransport`), so this is a tidiness call.
        """
        if self._pipeline is not None:
            self._pipeline.flush()

    def recv_step(self, tensor_specs: Dict[str, TensorSpec]) -> Dict[str, torch.Tensor]:
        """Receive one step's worth of tensors and return them as a dict.

        Args:
            tensor_specs: dict of name → (shape, dtype). Must match
                exactly what the engine sends. Both sides walk
                ``sorted(tensor_specs.keys())``.

        Returns:
            ``Dict[str, Tensor]`` with the same keys as ``tensor_specs``.
            Tensors live on ``self._device``. Buffers are freshly
            allocated each step (variable seq_len).

        Raises:
            ValueError: empty tensor_specs (likely caller bug).
        """
        if not tensor_specs:
            raise ValueError("recv_step requires at least one tensor spec")

        names = _sorted_tensor_names(tensor_specs)

        if self._use_ipc:
            # Zero-copy: map the engine's GPU memory via CUDA IPC and
            # copy on-device into trainer-owned buffers. No host
            # round-trip.
            if self._pipeline is not None:
                # Pipelined: reuse the cached pool-buffer mapping, ack
                # with a non-blocking isend.
                logger.debug(
                    "NcclMultiTensorFetcher.recv_step (cuda-ipc-pipeline): src=%d names=%s",
                    self._src,
                    names,
                )
                return self._pipeline.trainer_recv(
                    tensor_specs, self._src, self._device, self._group
                )
            logger.debug(
                "NcclMultiTensorFetcher.recv_step (cuda-ipc): src=%d names=%s",
                self._src,
                names,
            )
            return ipc_recv(tensor_specs, self._src, self._device, self._group)

        if _group_is_gloo(self._group):
            # Colocate transport: receive into host buffers over the
            # gloo union group (NCCL can't span two ranks on one GPU),
            # then copy up to the device. tag=index matches the
            # sender's per-tensor tag.
            logger.debug(
                "NcclMultiTensorFetcher.recv_step (gloo): src=%d names=%s",
                self._src,
                names,
            )
            out: Dict[str, torch.Tensor] = {}
            for tag, name in enumerate(names):
                shape, dtype_raw = tensor_specs[name]
                dtype = _normalise_dtype(dtype_raw)
                cpu_buf = torch.empty(tuple(shape), dtype=dtype, device="cpu")
                dist.recv(cpu_buf, src=self._src, group=self._group, tag=tag)
                out[name] = cpu_buf.to(self._device)
            return out

        buffers: Dict[str, torch.Tensor] = {}
        ops = []
        for name in names:
            shape, dtype_raw = tensor_specs[name]
            dtype = _normalise_dtype(dtype_raw)
            buf = torch.empty(tuple(shape), dtype=dtype, device=self._device)
            buffers[name] = buf
            ops.append(dist.P2POp(dist.irecv, buf, peer=self._src, group=self._group))

        logger.debug(
            "NcclMultiTensorFetcher.recv_step: src=%d names=%s",
            self._src,
            names,
        )
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
        return buffers


def send_step(
    tensors: Dict[str, torch.Tensor],
    dst_global_rank: int,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Convenience symmetric helper for tests / engine-side library calls.

    Equivalent to constructing a one-shot
    :class:`torchspec.inference.engine.nccl_hidden_states_connector.NcclHiddenStatesConnector`
    and calling ``.send(tensors)``. We expose it here to keep the test
    surface minimal and avoid an inference-engine import from the
    trainer test path.
    """
    if not tensors:
        raise ValueError("send_step requires at least one tensor")

    names = sorted(tensors.keys())
    ops = []
    for name in names:
        t = tensors[name]
        if not t.is_contiguous():
            raise ValueError(f"send_step requires contiguous tensors; got non-contiguous '{name}'")
        if t.device.type != "cuda":
            raise ValueError(f"send_step requires CUDA tensors; got '{name}' on {t.device}")
        ops.append(dist.P2POp(dist.isend, t, peer=int(dst_global_rank), group=group))

    works = dist.batch_isend_irecv(ops)
    for work in works:
        work.wait()

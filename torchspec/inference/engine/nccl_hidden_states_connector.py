# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Engine-side multi-tensor NCCL P2P sender for colocate mode (Phase 4).

This is the engine-side counterpart to ``NcclDataFetcher`` /
``NcclMultiTensorFetcher`` on the trainer. It mirrors what the disaggregated
``MooncakeHiddenStatesConnector`` does (write hidden states to a shared
Mooncake store keyed by ``mooncake_key``), but the wire is a single NCCL
``batch_isend_irecv`` to the paired trainer rank instead of a TCP write
to a remote Mooncake server.

Wire protocol
-------------

Per training step, the engine produces a per-request ``Dict[str, Tensor]``.
The exact key set depends on the draft model:

- Eagle3 with last_hidden_states + target_logits:
  ``{"hidden_states", "aux_hidden_states", "last_hidden_states",
     "target_logits"}``
- Eagle3 without last_hidden_states (older configs):
  ``{"hidden_states", "aux_hidden_states", "target_logits"}``
- DFlash variants: as defined by the draft trainer.

The connector sends the tensors in **sorted-by-key** order via a single
``dist.batch_isend_irecv`` call. The receiver
(:class:`torchspec.training.nccl_data_fetcher.NcclMultiTensorFetcher`)
must agree on this ordering — it does, because it uses the same sort.

Pairing
-------

Each engine rank ``i`` (in ``[0, N)`` of the engine role, i.e. global rank
``N+i`` in the union world) is paired with trainer rank ``i`` (global rank
``i``). The connector therefore needs only its own engine role rank and
the union-world ``UnionWorld`` handle to pick the destination:

    dst_global_rank = paired_global_rank  # held on UnionWorld

Within an engine TP group, the engine's TP rank-0 worker is the canonical
sender (sglang's spec_training callback runs there). For TP > 1 the
local-shard split happens **upstream** of this connector (the sglang patch
slices the global-batch hidden states by TP rank before invoking the
callback). This connector is intentionally TP-unaware.

Layering
--------

This module **does not** depend on sglang. It's a pure
``torch.distributed`` library function that the upstream sglang patch
calls. The patch lives outside this repo (see
``docs/colocate/sglang_patch.md`` for the patch surface). When the
``transfer_mode == 'nccl'`` flag is set on ``SglEngine``, sgl_engine.py
exports an env marker (:data:`TRANSFER_MODE_ENV`) and a destination-rank
table; the patch reads them and instantiates this connector.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist

from torchspec.colocate.cuda_ipc import (
    IpcPipelineTransport,
    ensure_ipc_usable,
    ipc_enabled,
    ipc_pipeline_enabled,
    ipc_send,
)

logger = logging.getLogger("torchspec.inference.engine.nccl_hidden_states_connector")

# Env marker the engine sets when colocate NCCL transfer is selected. The
# upstream sglang patch checks this to decide between Mooncake-write and
# NCCL-send paths in its spec_training callback.
TRANSFER_MODE_ENV = "TORCHSPEC_COLOCATE_TRANSFER_MODE"

# Env variable carrying the paired trainer global rank. The engine sets
# this once at init; the patch reads it on each callback invocation.
PAIRED_TRAINER_RANK_ENV = "TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK"


def _group_is_gloo(group: Optional[dist.ProcessGroup]) -> bool:
    """True iff ``group`` (or the default PG) uses the gloo backend.

    The colocate path runs the transfer over a gloo group: trainer and
    engine share one physical GPU, and NCCL refuses to form a
    communicator with two ranks on the same device ("Duplicate GPU
    detected"). gloo has no such restriction — it stages through host
    memory — so colocate uses it for the engine→trainer P2P.
    """
    try:
        return str(dist.get_backend(group)).lower() == "gloo"
    except Exception:
        return False


def sorted_tensor_names(tensors: Dict[str, torch.Tensor]) -> list[str]:
    """Canonical send/recv ordering: sorted by key.

    Both the sender (this module) and the receiver
    (:class:`NcclMultiTensorFetcher`) use this to pick the order of P2P
    ops in a single batched call. Using sorted-by-key lets the two sides
    agree without a separate handshake message — the metadata channel
    (gloo group) already carries the dict's key set as part of
    ``ColocateTrainSample.tensor_specs``.
    """
    return sorted(tensors.keys())


class NcclHiddenStatesConnector:
    """Engine-side sender for the colocate hidden-state plane.

    One connector per engine TP rank. The connector holds:

    - the destination global rank (paired trainer in the union world),
    - the union-world default process group (for the actual send).

    By default the connector is **stateless across calls** — it holds no
    per-tensor buffers and sends the caller's tensors directly. The sglang
    patch manages the lifetime of those tensors (the callback owns them
    for the duration of the send, then sglang frees them afterwards).

    The exception is the **pipelined transport**
    (``TORCHSPEC_COLOCATE_IPC_PIPELINE=1``): there the connector holds a
    persistent :class:`IpcPipelineTransport` (a send-buffer pool), but the
    lifetime contract to the caller is unchanged — ``send`` still fully
    consumes the caller's tensors (copies them into the pool) before it
    returns. Call :meth:`flush` at loop teardown for a tidy shutdown
    (optional — the pipeline is drain-safe without it).

    Args:
        dst_global_rank: Global rank to send to. For engine role rank
            ``i`` in a union world of size ``2N`` this is ``i`` (the
            paired trainer).
        group: Process group to send on. Defaults to the world default
            (the union world). Tests can pass a subgroup.

    Raises:
        RuntimeError: if torch.distributed is not initialised.
    """

    def __init__(
        self,
        dst_global_rank: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        if not dist.is_initialized():
            raise RuntimeError(
                "NcclHiddenStatesConnector requires torch.distributed to be "
                "initialised (call init_union_world first)."
            )
        self._dst = int(dst_global_rank)
        self._group = group
        # CUDA IPC transport (the default) replaces the gloo CPU-staged
        # path. Fail fast at construction if the platform can't do it
        # (e.g. expandable_segments active) so the engine and trainer
        # never disagree on the wire format.
        self._use_ipc = ipc_enabled() and _group_is_gloo(self._group)
        # Pipelined transport (pool + ack pipelining): an opt-in, stateful
        # alternative to the plain ipc_send. None unless explicitly on.
        self._pipeline: Optional[IpcPipelineTransport] = None
        if self._use_ipc:
            ensure_ipc_usable()
            if ipc_pipeline_enabled():
                self._pipeline = IpcPipelineTransport(role="engine")

    @property
    def dst_global_rank(self) -> int:
        return self._dst

    def flush(self) -> None:
        """Drain the pipelined transport at loop teardown.

        No-op unless the pipelined transport is active. The pipeline is
        drain-safe without this (see :class:`IpcPipelineTransport`), so a
        caller that cannot reach a teardown hook may skip it.
        """
        if self._pipeline is not None:
            self._pipeline.flush()

    def send(self, tensors: Dict[str, torch.Tensor]) -> None:
        """Send a named-tensor dict to the paired trainer rank.

        The send is synchronous on the calling thread: this function
        returns only after every P2P op has reported completion. Using a
        single ``batch_isend_irecv`` issues all ops to NCCL at once,
        which avoids the lazy 2-rank sub-communicator init pathology of
        unbatched send/recv on a large parent group (Phase 3 lessons).

        Args:
            tensors: dict of name → tensor. Every tensor must:
                - Live on a CUDA device matching the union world's
                  ``device_id`` for this rank (typically the only GPU
                  visible under Ray's ``CUDA_VISIBLE_DEVICES`` isolation).
                - Be contiguous (NCCL P2P requires contiguous memory).
                - Have a shape and dtype that match what the receiver
                  pre-allocated, in the same key order this side sends.

        Raises:
            ValueError: empty tensor dict (the metadata channel does not
                announce zero-tensor steps; this is always a bug).
            RuntimeError: NCCL error from the underlying send.
        """
        if not tensors:
            raise ValueError("NcclHiddenStatesConnector.send requires at least one tensor")

        names = sorted_tensor_names(tensors)

        if self._use_ipc:
            # Zero-copy: ship CUDA IPC handles over gloo, trainer maps
            # our memory and does an on-device D->D copy. No host
            # round-trip.
            if self._pipeline is not None:
                # Pipelined: copy into the send-buffer pool, ship the
                # pooled handle, defer this step's ack by one step.
                logger.debug(
                    "NcclHiddenStatesConnector.send (cuda-ipc-pipeline): dst=%d names=%s",
                    self._dst,
                    names,
                )
                self._pipeline.engine_send(tensors, self._dst, self._group)
                return
            # Plain CUDA IPC: fresh handle per step, blocks until the
            # trainer acks.
            logger.debug(
                "NcclHiddenStatesConnector.send (cuda-ipc): dst=%d names=%s",
                self._dst,
                names,
            )
            ipc_send(tensors, self._dst, self._group)
            return

        if _group_is_gloo(self._group):
            # Colocate transport: trainer + engine share one physical
            # GPU, so NCCL refuses a communicator spanning both ranks.
            # Stage each tensor through host memory and send over the
            # gloo union group. The blocking .cpu() copy synchronises
            # the producing CUDA stream, so the bytes on the wire are
            # the finished hidden states. tag=index pairs each send
            # with the receiver's matching recv unambiguously.
            logger.debug(
                "NcclHiddenStatesConnector.send (gloo): dst=%d names=%s",
                self._dst,
                names,
            )
            for tag, name in enumerate(names):
                cpu_t = tensors[name].detach().to("cpu", copy=True).contiguous()
                dist.send(cpu_t, dst=self._dst, group=self._group, tag=tag)
            return

        ops = []
        for name in names:
            t = tensors[name]
            if not t.is_contiguous():
                # We could `t = t.contiguous()` silently, but that hides
                # an upstream allocator inefficiency that the user
                # probably wants to see. Fail loud at the boundary.
                raise ValueError(
                    f"NcclHiddenStatesConnector requires contiguous tensors; "
                    f"got non-contiguous '{name}' (shape={tuple(t.shape)})"
                )
            if t.device.type != "cuda":
                raise ValueError(
                    f"NcclHiddenStatesConnector requires CUDA tensors; "
                    f"got '{name}' on device {t.device}"
                )
            ops.append(dist.P2POp(dist.isend, t, peer=self._dst, group=self._group))

        logger.debug(
            "NcclHiddenStatesConnector.send: dst=%d names=%s",
            self._dst,
            names,
        )
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()


def export_transfer_mode_env(transfer_mode: str, paired_trainer_rank: int) -> None:
    """Engine-side helper: surface transfer_mode + pairing to sglang patch.

    The sglang patch (out-of-tree) reads these to decide its
    spec_training callback path. We set both regardless of mode so the
    patch can fail loudly if the env is missing — that's how upstream
    detects "TorchSpec wired me wrong" vs "TorchSpec is genuinely on
    Mooncake".
    """
    import os

    os.environ[TRANSFER_MODE_ENV] = str(transfer_mode)
    os.environ[PAIRED_TRAINER_RANK_ENV] = str(int(paired_trainer_rank))


def read_transfer_mode_env() -> Optional[str]:
    """Inverse of :func:`export_transfer_mode_env`. Returns None if unset."""
    import os

    return os.environ.get(TRANSFER_MODE_ENV)


def read_paired_trainer_rank_env() -> Optional[int]:
    """Read the paired trainer global rank, or None if unset."""
    import os

    val = os.environ.get(PAIRED_TRAINER_RANK_ENV)
    return int(val) if val is not None else None

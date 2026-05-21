# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Union NCCL world bootstrap for colocate mode (Phase 2).

The colocate plan puts trainer and engine processes on the same physical
GPUs. To send hidden states from the engine to the trainer over NCCL P2P,
both sides must be members of one NCCL world of size ``2 * N`` (N =
training_world_size). This module provides:

- A small ``UnionWorldSpec`` dataclass capturing rendezvous params.
- ``rank_for_role(world_size, role, role_rank)`` — the canonical
  rank-assignment scheme from ``implementation.md`` §Phase 2: trainer ranks
  ``0..N-1``, engine ranks ``N..2N-1``.
- ``init_union_world(spec)`` — initialises the **default** torch.distributed
  PG for the calling process so it sees a 2N-rank world, plus exposes the
  FSDP-only subgroup ``ranks=[0..N-1]`` and a gloo CPU subgroup spanning
  all ranks (for step-metadata broadcast).

**Important**: the trainer side is the easy half. The engine side has a
known wrinkle — sglang internally calls ``dist.init_process_group`` for
its own TP group, and PyTorch only allows one *default* PG per process.
``init_union_world`` writes a small marker into the env so a later
sglang-patch hook can:

  - Skip its own ``init_process_group`` call when our union world is
    already the default (``TORCHSPEC_UNION_WORLD_INITIALIZED=1``), or
  - Reconstruct sglang's TP via ``dist.new_group`` against our union world
    using the rank list it would have used otherwise.

That patch lives in ``patches/_sglang/`` (Phase 2 sub-task 5) and is
exercised by the Phase 2 Modal smoke test.

For Phase 2 we ship:

  1. This helper, fully unit-tested against torch.distributed semantics.
  2. A trainer-side init path that uses it.
  3. A standalone NCCL barrier test: 4 trainer-shape + 4 engine-shape
     processes (no sglang), all join the union world, all
     ``dist.barrier()``.

Phase 2 *does not* require sglang to use the union world for its own TP
yet — that's Phase 4's hidden-state hook. We just need the mechanism to
exist and the 8-rank barrier to succeed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

logger = logging.getLogger("torchspec.colocate.world")

# Roles for the union-world rank-assignment helper. Names match the
# ``role`` argument passed to ``RayTrainGroup.async_init`` /
# ``SglEngine.init`` so the call sites read naturally.
ROLE_TRAINER = "training"
ROLE_ENGINE = "inference"

# Marker we set in os.environ once the union world is up. Read by the
# sglang patch (or any other downstream code) to know the default PG is
# already a 2N-rank world and not a vanilla per-process one.
UNION_WORLD_ENV_MARKER = "TORCHSPEC_COLOCATE_UNION_WORLD"


@dataclass(frozen=True)
class UnionWorldSpec:
    """Parameters needed to bootstrap the union NCCL world on every rank.

    The driver computes this once and broadcasts it to all 2N actors via
    Ray. Ranks join collectively.
    """

    n_per_role: int
    """Number of ranks per role (trainer count == engine count == N)."""

    master_addr: str
    """IP/hostname of the rendezvous master (any 1 actor's IP works)."""

    master_port: int
    """Free TCP port on master_addr; pre-checked by the driver."""

    timeout_minutes: int = 30
    """init_process_group timeout. NCCL default is 10 min, which is too
    short for cold starts where one side might be slower to boot."""

    @property
    def world_size(self) -> int:
        return 2 * self.n_per_role

    @property
    def init_method(self) -> str:
        return f"tcp://{self.master_addr}:{self.master_port}"


def rank_for_role(spec: UnionWorldSpec, role: str, role_rank: int) -> int:
    """Map (role, role_rank) → global rank in the union world.

    Trainers occupy ranks ``[0, N)``, engines occupy ``[N, 2N)``.

    Raises:
        ValueError: unknown role, or role_rank out of range.
    """
    if role == ROLE_TRAINER:
        if not 0 <= role_rank < spec.n_per_role:
            raise ValueError(f"trainer role_rank {role_rank} out of range [0, {spec.n_per_role})")
        return role_rank
    if role == ROLE_ENGINE:
        if not 0 <= role_rank < spec.n_per_role:
            raise ValueError(f"engine role_rank {role_rank} out of range [0, {spec.n_per_role})")
        return spec.n_per_role + role_rank
    raise ValueError(f"unknown role {role!r}; expected {ROLE_TRAINER!r} or {ROLE_ENGINE!r}")


def trainer_global_ranks(spec: UnionWorldSpec) -> list[int]:
    """Convenience: union-world ranks held by trainers (= [0..N))."""
    return list(range(spec.n_per_role))


def engine_global_ranks(spec: UnionWorldSpec) -> list[int]:
    """Convenience: union-world ranks held by engines (= [N..2N))."""
    return list(range(spec.n_per_role, 2 * spec.n_per_role))


@dataclass
class UnionWorld:
    """Live handle to the initialised union world for one rank.

    Returned by ``init_union_world``. Holds references to the subgroups so
    callers can pass them to FSDP / collective ops without re-deriving.
    """

    spec: UnionWorldSpec
    role: str
    role_rank: int
    global_rank: int
    paired_global_rank: int
    """The opposite-role rank paired with this one. Trainer rank ``i``
    is paired with engine rank ``N+i`` and vice versa. Use for the
    ``dst``/``src`` arg of ``dist.send`` / ``dist.recv`` /
    ``dist.batch_isend_irecv`` ops on the union world."""
    fsdp_group: object  # torch.distributed.ProcessGroup
    """Subgroup of just trainer ranks; pass to FSDP DeviceMesh.

    On engine ranks this is set to ``None`` because the engine is not in
    the FSDP group; calling collectives on it from an engine would hang."""
    meta_group: object  # torch.distributed.ProcessGroup
    """Gloo subgroup spanning all 2N ranks. Used for CPU-side step
    metadata broadcast (cheap dict broadcast, no GPU needed)."""
    trainer_gloo_group: object  # torch.distributed.ProcessGroup
    """Gloo subgroup of just trainer ranks ``[0, N)``. Bound to
    :data:`torchspec.utils.distributed.GLOO_GROUP` in trainer_actor so
    that ``dist.barrier(group=get_gloo_group())`` calls (e.g.
    eagle3_trainer.py line 82, dflash_trainer.py line 113) sync only
    the trainer half of the union world. Using ``meta_group`` here
    would block on the engine, which never enters trainer-side
    barriers. Set to ``None`` on engine ranks (engines don't use it).
    For 1-trainer runs this is a 1-rank gloo group — gloo handles
    1-rank groups cleanly, unlike NCCL."""


def init_union_world(
    spec: UnionWorldSpec,
    role: str,
    role_rank: int,
    *,
    device_id: Optional[int] = None,
) -> UnionWorld:
    """Collective: initialise the union world from this process.

    All 2N ranks must call this with consistent ``spec`` (same master_addr,
    master_port, n_per_role) and the right ``role`` / ``role_rank``.

    Args:
        device_id: Local CUDA device index this rank uses. Defaults to
            ``torch.cuda.current_device()`` (typically ``0`` under
            Ray's ``CUDA_VISIBLE_DEVICES`` isolation since the actor
            sees only one GPU). **Must be passed correctly** — without
            it, NCCL guesses device by global rank, which under Ray
            isolation maps to a non-existent local GPU and silently
            deadlocks P2P send/recv.

    Side-effects:
        - Calls ``dist.init_process_group(backend='nccl', world_size=2N, …)``.
          The default PG of this process becomes the union world.
        - Calls ``dist.new_group`` twice (collective on all 2N ranks):
          once for the trainer-only NCCL subgroup, once for the gloo
          all-rank metadata subgroup.
        - Sets ``TORCHSPEC_COLOCATE_UNION_WORLD`` env marker so downstream
          code (e.g. sglang patches) can detect the union-world setup.

    P2P transfers (engine→trainer hidden states) should use
    ``dist.batch_isend_irecv`` on the default union world; this is faster
    and avoids the lazy 2-rank sub-communicator pathology of unbatched
    ``send``/``recv`` on a large parent group.

    Returns:
        UnionWorld handle with the subgroup references.

    Raises:
        RuntimeError: if a default PG is already initialised. This is the
            integration-with-sglang risk flagged in implementation.md
            §Phase 2 risk register.
    """
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed default group is already initialised. The colocate "
            "union world must be the default group; call init_union_world *before* "
            "any other framework (FSDP, sglang, etc.) initialises its own world. "
            "Set role=engine and patch sglang to skip its own init_process_group "
            "when TORCHSPEC_COLOCATE_UNION_WORLD=1."
        )

    global_rank = rank_for_role(spec, role, role_rank)
    paired_global_rank = (
        rank_for_role(spec, ROLE_ENGINE, role_rank)
        if role == ROLE_TRAINER
        else rank_for_role(spec, ROLE_TRAINER, role_rank)
    )

    if device_id is None:
        device_id = torch.cuda.current_device()
    device = torch.device("cuda", int(device_id))

    logger.info(
        "Initialising union world: role=%s role_rank=%d global_rank=%d "
        "paired_global_rank=%d world_size=%d init_method=%s device=%s",
        role,
        role_rank,
        global_rank,
        paired_global_rank,
        spec.world_size,
        spec.init_method,
        device,
    )

    # NB: deliberately *do not* pass ``device_id=`` here. Passing it
    # turns init_process_group into "eager init" mode where every rank
    # must reach init_process_group before NCCL's socketPollConnect
    # backoff exhausts itself (35 retries — single-digit seconds in
    # practice). Trainers are ready in tens of seconds; engines
    # sometimes need minutes for sglang scheduler subprocess startup
    # and HF model download. The lazy default is what we want — the
    # NCCL handshake happens on the first collective op (the broadcast
    # the trainer issues right after init_process_group), and that
    # collective inherits the 10-minute ``timeout`` we passed below
    # so the slowest engine has plenty of slack to catch up.
    dist.init_process_group(
        backend="nccl",
        world_size=spec.world_size,
        rank=global_rank,
        init_method=spec.init_method,
        timeout=timedelta(minutes=spec.timeout_minutes),
    )

    # Subgroups are collective: every rank must call new_group with the
    # same args, even ranks not in the resulting subgroup.
    all_world_ranks = list(range(spec.world_size))

    # sglang's `init_distributed_environment` -> `init_world_group` ->
    # `GroupCoordinator.__init__` creates a (nccl, gloo) pair of world-
    # spanning subgroups for its `_WORLD` GroupCoordinator. Those calls
    # are collective on the world group, so this rank must call the
    # matching new_groups in the same order — otherwise the engine TP
    # scheduler subprocess hangs forever in `init_distributed_environment`
    # waiting for the trainer half of the rendezvous (validated on
    # RunPod H100 SXM, see implementation_log.md §RunPod validation
    # session). We discard the resulting handles since this side
    # doesn't actually use sglang's world group, but the new_group
    # collective bookkeeping must match.
    #
    # `use_local_synchronization=True` is required for symmetry with
    # the engine side: the colocate sglang patch installs a
    # dist.new_group monkey-patch that defaults the flag to True for
    # every call inside the engine TP scheduler subprocess. If the two
    # sides disagree on the flag, c10d's rendezvous semantics don't
    # match up and the call deadlocks. For ranks covering the full
    # world (all 2N ranks are members) the True/False distinction is
    # otherwise equivalent — every rank participates either way — so
    # this just keeps both sides honest.
    # Ordering invariant: the three *shared* (all-world) new_groups —
    # sglang-paired nccl, sglang-paired gloo, meta_group — must be
    # created BEFORE any role-restricted group (fsdp, trainer-only
    # gloo). With use_local_synchronization=True, c10d derives each
    # group's name from a hash that includes the per-process new_group
    # counter; a shared group only rendezvouses if every member creates
    # it at the same counter value. The engine side issues exactly
    # three all-world new_groups (sglang init_world_group's nccl+gloo,
    # then the patch's meta_group). If the trainer slips a trainer-only
    # new_group (fsdp) in between, its counter runs ahead and the
    # meta_group hash no longer matches the engine's — a hard
    # rendezvous deadlock. Invisible at N=1 (fsdp is skipped); fatal at
    # N>=2. So: all shared groups first, role-restricted groups after.
    logger.info(
        "[colocate] %s rank %d: world.py new_group #1 sglang-paired nccl (all %d ranks)",
        role,
        role_rank,
        spec.world_size,
    )
    _ = dist.new_group(
        ranks=all_world_ranks,
        backend="nccl",
        use_local_synchronization=True,
    )
    logger.info(
        "[colocate] %s rank %d: world.py new_group #2 sglang-paired gloo (all %d ranks)",
        role,
        role_rank,
        spec.world_size,
    )
    _ = dist.new_group(
        ranks=all_world_ranks,
        backend="gloo",
        use_local_synchronization=True,
    )
    logger.info(
        "[colocate] %s rank %d: world.py new_group #3 meta_group gloo (all %d ranks)",
        role,
        role_rank,
        spec.world_size,
    )
    meta_group = dist.new_group(
        ranks=all_world_ranks,
        backend="gloo",
        use_local_synchronization=True,
    )

    # Role-restricted groups — created AFTER all shared groups so the
    # shared-group counter stays in lockstep with the engine side.
    fsdp_ranks = trainer_global_ranks(spec)
    if len(fsdp_ranks) >= 2:
        # NCCL 1-rank groups can hang under eager-init / `device_id`;
        # skip when there's only one trainer (e.g. tests at minimal
        # scale). FSDP itself doesn't need a group at world_size 1.
        logger.info(
            "[colocate] %s rank %d: world.py new_group #4 fsdp nccl (trainer ranks %s)",
            role,
            role_rank,
            fsdp_ranks,
        )
        fsdp_group = dist.new_group(
            ranks=fsdp_ranks,
            backend="nccl",
            use_local_synchronization=True,
        )
        if role != ROLE_TRAINER:
            # Engines aren't in the FSDP group; expose None so calling
            # FSDP collectives on this is a clear error rather than a hang.
            fsdp_group_for_role: Optional[object] = None
        else:
            fsdp_group_for_role = fsdp_group
    else:
        fsdp_group_for_role = None

    # Trainer-only gloo group for trainer-side barriers. Engine ranks
    # don't need to participate; we pass use_local_synchronization=True
    # so they skip the call entirely. On engine ranks the local handle
    # is discarded (set to None on the returned UnionWorld). For
    # 1-trainer runs this is a 1-rank gloo group — gloo handles
    # 1-rank groups cleanly (unlike NCCL where 1-rank groups can hang
    # at eager init).
    logger.info(
        "[colocate] %s rank %d: world.py new_group #5 trainer-only gloo (trainer ranks %s)",
        role,
        role_rank,
        trainer_global_ranks(spec),
    )
    trainer_only_gloo = dist.new_group(
        ranks=trainer_global_ranks(spec),
        backend="gloo",
        use_local_synchronization=True,
    )
    trainer_gloo_for_role: Optional[object]
    if role == ROLE_TRAINER:
        trainer_gloo_for_role = trainer_only_gloo
    else:
        trainer_gloo_for_role = None

    logger.info(
        "[colocate] %s rank %d: world.py all new_groups complete",
        role,
        role_rank,
    )

    os.environ[UNION_WORLD_ENV_MARKER] = "1"

    return UnionWorld(
        spec=spec,
        role=role,
        role_rank=role_rank,
        global_rank=global_rank,
        paired_global_rank=paired_global_rank,
        fsdp_group=fsdp_group_for_role,
        meta_group=meta_group,
        trainer_gloo_group=trainer_gloo_for_role,
    )


def union_world_ready() -> bool:
    """Cheap query for downstream code (e.g. the sglang patch hook)."""
    return os.environ.get(UNION_WORLD_ENV_MARKER) == "1"

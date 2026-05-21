# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Colocate configuration validation (Phase 0).

Kept in its own module so the unit tests can import the validator without
pulling in Ray, sglang, or torch (the project's root ``conftest.py`` stubs
those for Mac dev boxes, but importing ``train_entry`` triggers eager Ray
imports we want to avoid in fast unit tests).
"""

from __future__ import annotations

from typing import Any


class ColocateConfigError(ValueError):
    """Raised when the colocate flag combination is unsupported.

    Subclassing ``ValueError`` keeps callers (and tests) compatible with the
    pre-existing ``raise ValueError(...)`` patterns elsewhere in
    ``train_entry.py``.
    """


# The only two combinations the implementation currently supports. See
# docs/colocate/implementation.md §"Configuration model".
SUPPORTED_COMBINATIONS: tuple[tuple[str | None, str], ...] = (
    (None, "mooncake"),
    ("mps", "nccl"),
)

# Headroom we reserve on every GPU for CUDA context, allocator caches, and
# other overhead that neither the trainer nor the engine accounts for in its
# own ``mem_fraction``. Phase 1 invariant (`train_frac + infer_frac + 0.10
# <= 1.0`).
_HEADROOM_FRAC = 0.10


def _get(args: Any, name: str, default: Any = None) -> Any:
    """Mirror ``train_entry.py``'s ``getattr(args, ..., default)`` style.

    ``args`` here is whatever ``parse_config()`` produced (either a flat
    ``argparse.Namespace`` post-``config_to_flat_args`` or, in the test
    harness, a small stand-in object).
    """
    return getattr(args, name, default)


def is_colocate_enabled(args: Any) -> bool:
    """Return True iff colocate mode is requested.

    We treat ``colocate=True`` _or_ ``colocate_strategy`` set as the trigger,
    so the existing partial colocate path (which only sets the bool) keeps
    working.
    """
    return bool(_get(args, "colocate", False)) or _get(args, "colocate_strategy") is not None


def is_mps_colocate(args: Any) -> bool:
    """Return True iff the *new* MPS-strategy colocate path is selected.

    Distinguishes the new (Phase 1+) code path from the legacy
    ``colocate=True`` boolean which still routes through the old shared-PG
    branch. Used by placement / actor wiring to decide whether to apply
    fractional GPU claims and inject MPS env vars.
    """
    return _get(args, "colocate_strategy") == "mps"


def _resolve_engine_count(args: Any) -> int:
    """Number of inference engines the controller will spawn.

    Mirrors ``factory._prepare_sgl_engines`` for single-node:

        num_engines = inference_num_gpus // inference_num_gpus_per_engine

    For multi-node we fall back to ``inference_num_gpus`` since each engine
    spans a full node — the ``engine_count × engine_tp_size == world_size``
    invariant only needs to match _logical_ engines, not physical ones.
    """
    inf_gpus = int(_get(args, "inference_num_gpus", 0) or 0)
    gpus_per_engine = int(_get(args, "inference_num_gpus_per_engine", 1) or 1)
    if gpus_per_engine <= 0:
        gpus_per_engine = 1
    return max(1, inf_gpus // gpus_per_engine)


def _resolve_engine_tp_size(args: Any) -> int:
    gpus_per_engine = int(_get(args, "inference_num_gpus_per_engine", 1) or 1)
    return max(1, gpus_per_engine)


def validate_colocate_config(args: Any) -> None:
    """Validate the colocate flag combination on a parsed config.

    Called from ``train_entry.parse_config`` after ``config_to_flat_args``.
    No-op unless colocate is enabled.

    Raises:
        ColocateConfigError: if any invariant is violated. The error message
            states which invariant failed and suggests a fix.
    """
    if not is_colocate_enabled(args):
        # Disaggregated default: nothing to validate. We do, however, want to
        # warn the user if they set strategy/frac fields by mistake without
        # turning colocate on, since otherwise those fields silently no-op.
        for stray in ("colocate_strategy", "train_frac", "infer_frac"):
            if _get(args, stray) is not None:
                raise ColocateConfigError(
                    f"training.{stray} was set but training.colocate=False. "
                    f"Either set training.colocate=true (or "
                    f"training.colocate_strategy=mps) or remove training.{stray}."
                )
        return

    strategy = _get(args, "colocate_strategy")
    transfer_mode = _get(args, "transfer_mode", "mooncake") or "mooncake"

    # Invariant A: only the two (strategy, transfer_mode) combinations from
    # implementation.md §Configuration model are accepted.
    combo = (strategy, transfer_mode)
    if combo not in SUPPORTED_COMBINATIONS:
        supported_str = ", ".join(
            f"(colocate_strategy={s!r}, transfer_mode={t!r})" for s, t in SUPPORTED_COMBINATIONS
        )
        raise ColocateConfigError(
            f"Unsupported colocate combination: colocate_strategy={strategy!r}, "
            f"transfer_mode={transfer_mode!r}. Supported: {supported_str}. "
            f"In particular, colocate_strategy='mps' requires transfer_mode='nccl' "
            f"— Mooncake-with-colocate provides no benefit and is intentionally "
            f"unsupported."
        )

    if strategy != "mps":
        # The implicit (None, mooncake) case is allowed even when
        # ``colocate=True`` for backwards compatibility with the existing
        # partial colocate path; nothing else to validate.
        return

    # Invariant B: train_frac + infer_frac + headroom <= 1.0
    train_frac = _get(args, "train_frac")
    infer_frac = _get(args, "infer_frac")
    if train_frac is None or infer_frac is None:
        raise ColocateConfigError(
            "training.train_frac and training.infer_frac are required when "
            "training.colocate_strategy='mps'. Pick values that leave at "
            f"least {_HEADROOM_FRAC:.0%} headroom (e.g. train_frac=0.45, "
            "infer_frac=0.45)."
        )

    train_frac = float(train_frac)
    infer_frac = float(infer_frac)
    if not (0.0 < train_frac < 1.0):
        raise ColocateConfigError(f"training.train_frac must be in (0, 1); got {train_frac}.")
    if not (0.0 < infer_frac < 1.0):
        raise ColocateConfigError(f"training.infer_frac must be in (0, 1); got {infer_frac}.")
    total = train_frac + infer_frac + _HEADROOM_FRAC
    if total > 1.0 + 1e-9:
        raise ColocateConfigError(
            f"train_frac ({train_frac}) + infer_frac ({infer_frac}) + "
            f"headroom ({_HEADROOM_FRAC}) = {total:.3f} > 1.0. Lower one or "
            f"both fractions so the sum (plus headroom) fits on a single GPU."
        )

    # Invariant C: engine_count × engine_tp_size == training_world_size. The
    # MPS strategy lays out one engine rank per trainer rank on the same Ray
    # bundle; if those counts don't match we'd either leave bundles empty or
    # try to stack two engine ranks on the same GPU.
    world_size = int(_get(args, "world_size") or 0)
    if world_size <= 0:
        # parse_config sets ``world_size = num_nodes * num_gpus_per_node``
        # before validation runs; if it's still 0 we have a bigger problem
        # than colocate.
        world_size = int(_get(args, "training_num_nodes", 1) or 1) * int(
            _get(args, "training_num_gpus_per_node", 1) or 1
        )

    engine_count = _resolve_engine_count(args)
    engine_tp_size = _resolve_engine_tp_size(args)
    if engine_count * engine_tp_size != world_size:
        raise ColocateConfigError(
            f"engine_count ({engine_count}) × engine_tp_size "
            f"({engine_tp_size}) = {engine_count * engine_tp_size} != "
            f"training_world_size ({world_size}). Colocate (mps) requires a "
            f"1:1 trainer↔engine-rank pairing. Adjust "
            f"inference.inference_num_gpus / "
            f"inference.inference_num_gpus_per_engine or "
            f"training.training_num_gpus_per_node."
        )

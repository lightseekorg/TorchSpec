# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Multi-engine-TP union-world rank math.

``torchspec_colocate.ColocateEnv.engine_global_rank`` and
``build_engine_tp_ranks`` (shipped inside ``colocate.patch``) map an
engine's TP ranks onto the union world. They were originally scoped to
``engine_tp_size == 1``; they now return the contiguous
``[N + base, N + base + engine_tp_size)`` block for any TP size, where
``base = engine_index * engine_tp_size == paired_trainer_rank``.

This test imports the patched sglang module and checks both the
``engine_tp_size == 1`` backward-compatible case and the ``> 1`` block.
It self-skips when sglang isn't installed (Mac dev box) — the module
lives inside ``colocate.patch``, so it's only importable on a host that
has applied the patch.
"""

from __future__ import annotations

import dataclasses

import pytest

# The module lives inside colocate.patch, so it's only importable on a
# host that has applied the patch. conftest *mocks* sglang on a Mac dev
# box, so importorskip would not skip — instead require ColocateEnv to be
# a real dataclass (a MagicMock stub is not).
try:
    from sglang.srt.distributed import torchspec_colocate as tsc

    _HAVE_REAL = dataclasses.is_dataclass(getattr(tsc, "ColocateEnv", None))
except Exception:  # pragma: no cover
    tsc = None
    _HAVE_REAL = False

pytestmark = pytest.mark.skipif(
    not _HAVE_REAL,
    reason="patched sglang (colocate.patch) not installed on this host",
)


def _env(paired_trainer_rank: int, n_per_role: int, engine_tp_size: int):
    return tsc.ColocateEnv(
        paired_trainer_rank=paired_trainer_rank,
        master_addr="127.0.0.1",
        master_port=29500,
        world_size=2 * n_per_role,
        n_per_role=n_per_role,
        timeout_minutes=30,
        engine_tp_size=engine_tp_size,
    )


def test_engine_tp_size_field_defaults_to_one():
    e = tsc.ColocateEnv(
        paired_trainer_rank=0,
        master_addr="x",
        master_port=1,
        world_size=2,
        n_per_role=1,
        timeout_minutes=30,
    )
    assert e.engine_tp_size == 1


def test_tp1_backward_compatible():
    """At engine_tp_size==1 the rank math is byte-identical to before."""
    e = _env(paired_trainer_rank=2, n_per_role=4, engine_tp_size=1)
    assert e.engine_global_rank(0) == 6  # N(4) + base(2) + t(0)
    assert tsc.build_engine_tp_ranks(e) == [6]


@pytest.mark.parametrize(
    "engine_index,n_per_role,tp,expected",
    [
        (0, 4, 2, [4, 5]),  # engine 0, base 0 -> [N+0, N+1]
        (1, 4, 2, [6, 7]),  # engine 1, base 2 -> [N+2, N+3]
        (0, 8, 4, [8, 9, 10, 11]),
        (1, 8, 4, [12, 13, 14, 15]),
    ],
)
def test_tp_gt_1_contiguous_block(engine_index, n_per_role, tp, expected):
    base = engine_index * tp
    e = _env(paired_trainer_rank=base, n_per_role=n_per_role, engine_tp_size=tp)
    assert tsc.build_engine_tp_ranks(e) == expected
    for t in range(tp):
        assert e.engine_global_rank(t) == expected[t]


def test_engine_global_rank_rejects_out_of_range_tp_rank():
    e = _env(paired_trainer_rank=0, n_per_role=4, engine_tp_size=2)
    with pytest.raises(ValueError):
        e.engine_global_rank(2)  # tp_rank must be in [0, engine_tp_size)

# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Phase 2 — UnionWorldSpec / rank-assignment unit tests.

The actual ``init_union_world`` requires torch.distributed (and 8 ranks).
That's exercised by the Phase 2 Modal smoke test
``tests/colocate/test_union_world.py``. Here we just unit-test the pure
helpers.
"""

from __future__ import annotations

import pytest

from torchspec.colocate.world import (
    ROLE_ENGINE,
    ROLE_TRAINER,
    UNION_WORLD_ENV_MARKER,
    UnionWorldSpec,
    engine_global_ranks,
    rank_for_role,
    trainer_global_ranks,
    union_world_ready,
)


def _spec(n: int = 4) -> UnionWorldSpec:
    return UnionWorldSpec(
        n_per_role=n,
        master_addr="10.0.0.1",
        master_port=29500,
    )


def test_world_size_and_init_method():
    s = _spec(4)
    assert s.world_size == 8
    assert s.init_method == "tcp://10.0.0.1:29500"


def test_rank_assignment_trainer():
    s = _spec(4)
    for r in range(4):
        assert rank_for_role(s, ROLE_TRAINER, r) == r


def test_rank_assignment_engine_offset():
    s = _spec(4)
    for r in range(4):
        assert rank_for_role(s, ROLE_ENGINE, r) == 4 + r


def test_unknown_role_rejected():
    s = _spec(4)
    with pytest.raises(ValueError, match="unknown role"):
        rank_for_role(s, "evaluator", 0)


@pytest.mark.parametrize("role", [ROLE_TRAINER, ROLE_ENGINE])
def test_rank_out_of_range_rejected(role):
    s = _spec(4)
    with pytest.raises(ValueError, match="out of range"):
        rank_for_role(s, role, 4)
    with pytest.raises(ValueError, match="out of range"):
        rank_for_role(s, role, -1)


def test_global_rank_lists_disjoint_and_cover():
    s = _spec(4)
    t = trainer_global_ranks(s)
    e = engine_global_ranks(s)
    assert t == [0, 1, 2, 3]
    assert e == [4, 5, 6, 7]
    assert set(t).isdisjoint(set(e))
    assert set(t) | set(e) == set(range(s.world_size))


def test_union_world_ready_off_by_default(monkeypatch):
    monkeypatch.delenv(UNION_WORLD_ENV_MARKER, raising=False)
    assert union_world_ready() is False


def test_union_world_ready_on_when_set(monkeypatch):
    monkeypatch.setenv(UNION_WORLD_ENV_MARKER, "1")
    assert union_world_ready() is True


def test_union_world_ready_off_when_other_value(monkeypatch):
    monkeypatch.setenv(UNION_WORLD_ENV_MARKER, "0")
    assert union_world_ready() is False

import importlib
import sys
from dataclasses import dataclass
from unittest.mock import patch

import torch

from torchspec.utils.types import InferenceOutput


@dataclass
class MockControllerArgs:
    per_dp_rank_batch_size: int = 2
    max_sample_pool_size: int = 0


def _create_controller_class():
    module_name = "torchspec.controller.training_controller"
    if module_name in sys.modules:
        del sys.modules[module_name]
    with patch("ray.remote", lambda cls: cls):
        module = importlib.import_module(module_name)
        return module.AsyncTrainingController


def _make_output(data_id: str, seq_len: int) -> InferenceOutput:
    return InferenceOutput(
        data_id=data_id,
        mooncake_key=f"key-{data_id}",
        tensor_shapes={"input_ids": (1, seq_len), "hidden_states": (1, seq_len, 4096)},
        tensor_dtypes={"input_ids": torch.int64, "hidden_states": torch.bfloat16},
    )


def _make_controller(dp_size: int, per_dp_rank_batch_size: int):
    AsyncTrainingController = _create_controller_class()
    args = MockControllerArgs(per_dp_rank_batch_size=per_dp_rank_batch_size)
    return AsyncTrainingController(args, dp_size=dp_size)


class TestPartitionFallback:
    """When at most one sample per rank, partition is round-robin."""

    def test_single_dp_rank_keeps_all_samples_together(self):
        controller = _make_controller(dp_size=1, per_dp_rank_batch_size=4)
        results = [_make_output(f"s{i}", seq_len=100 + i) for i in range(4)]

        partitions = controller._partition_results(results)

        assert len(partitions) == 1
        assert [r.data_id for r in partitions[0]] == ["s0", "s1", "s2", "s3"]

    def test_one_sample_per_rank_uses_round_robin(self):
        controller = _make_controller(dp_size=4, per_dp_rank_batch_size=1)
        results = [_make_output(f"s{i}", seq_len=1000 - 100 * i) for i in range(4)]

        partitions = controller._partition_results(results)

        assert [p[0].data_id for p in partitions] == ["s0", "s1", "s2", "s3"]

    def test_empty_results_returns_empty_partitions(self):
        controller = _make_controller(dp_size=4, per_dp_rank_batch_size=2)

        partitions = controller._partition_results([])

        assert partitions == [[], [], [], []]

    def test_non_divisible_batch_falls_back_to_round_robin(self):
        # 5 results over 2 ranks: capacity would floor to 2 and the
        # greedy generator would empty out on the 5th item. Fall back
        # to round-robin instead of crashing.
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [_make_output(f"s{i}", seq_len=100 + i) for i in range(5)]

        partitions = controller._partition_results(results)

        assert [r.data_id for r in partitions[0]] == ["s0", "s2", "s4"]
        assert [r.data_id for r in partitions[1]] == ["s1", "s3"]


class TestPartitionBinPacking:
    """When per-rank capacity > 1, partition balances total sequence load."""

    def test_capacity_is_exactly_results_per_rank(self):
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [_make_output(f"s{i}", seq_len=100) for i in range(4)]

        partitions = controller._partition_results(results)

        assert len(partitions) == 2
        assert len(partitions[0]) == 2
        assert len(partitions[1]) == 2

    def test_longest_first_balances_load_across_ranks(self):
        # Lengths 1000, 800, 200, 100 across dp=2 mbs=2:
        # Greedy LPT pairs 1000+100 and 800+200 (loads 1100 and 1000),
        # which is more balanced than round-robin's (1000+200, 800+100) = (1200, 900).
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [
            _make_output("a", 1000),
            _make_output("b", 800),
            _make_output("c", 200),
            _make_output("d", 100),
        ]

        partitions = controller._partition_results(results)

        loads = [sum(r.tensor_shapes["input_ids"][-1] for r in p) for p in partitions]
        assert sorted(loads) == [1000, 1100]

    def test_each_rank_receives_exactly_capacity_samples(self):
        # Stress test: skewed lengths must not violate per-rank capacity.
        controller = _make_controller(dp_size=4, per_dp_rank_batch_size=3)
        lengths = [2000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        results = [_make_output(f"s{i}", L) for i, L in enumerate(lengths)]

        partitions = controller._partition_results(results)

        assert len(partitions) == 4
        for p in partitions:
            assert len(p) == 3
        # Every sample is assigned exactly once.
        assigned = sorted(r.data_id for p in partitions for r in p)
        assert assigned == sorted(r.data_id for r in results)

    def test_outlier_does_not_starve_other_ranks(self):
        # One huge sample plus many small ones — the rank holding the
        # outlier should still receive `capacity` samples, not all of them.
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=4)
        results = [_make_output("big", 5000)] + [_make_output(f"s{i}", 100) for i in range(7)]

        partitions = controller._partition_results(results)

        assert len(partitions[0]) == 4
        assert len(partitions[1]) == 4

    def test_partition_is_deterministic_for_fixed_input(self):
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [_make_output(f"s{i}", L) for i, L in enumerate([300, 200, 400, 100])]

        p1 = controller._partition_results(results)
        p2 = controller._partition_results(results)

        ids1 = [[r.data_id for r in part] for part in p1]
        ids2 = [[r.data_id for r in part] for part in p2]
        assert ids1 == ids2


class TestPartitionDefensiveFallback:
    """`_partition_results` should not crash when `input_ids` shape is missing."""

    def test_missing_input_ids_shape_treated_as_zero_length(self):
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [
            InferenceOutput(
                data_id=f"s{i}",
                mooncake_key=f"k{i}",
                tensor_shapes={"hidden_states": (1, 100, 4096)},  # no "input_ids"
                tensor_dtypes={"hidden_states": torch.bfloat16},
            )
            for i in range(4)
        ]

        partitions = controller._partition_results(results)

        assert len(partitions) == 2
        assert len(partitions[0]) == 2
        assert len(partitions[1]) == 2

    def test_none_tensor_shapes_treated_as_zero_length(self):
        controller = _make_controller(dp_size=2, per_dp_rank_batch_size=2)
        results = [
            InferenceOutput(data_id=f"s{i}", mooncake_key=f"k{i}", tensor_shapes=None)
            for i in range(4)
        ]

        partitions = controller._partition_results(results)

        assert sum(len(p) for p in partitions) == 4

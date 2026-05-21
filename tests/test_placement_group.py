from argparse import Namespace
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


repo_root = Path(__file__).resolve().parents[1]
torchspec_pkg = sys.modules.get("torchspec")
if torchspec_pkg is None and importlib.util.find_spec("torch") is None:
    torchspec_pkg = types.ModuleType("torchspec")
    torchspec_pkg.__path__ = [str(repo_root / "torchspec")]
    torchspec_pkg.__package__ = "torchspec"
    sys.modules["torchspec"] = torchspec_pkg

ray_stub = types.ModuleType("ray")
ray_util_stub = types.ModuleType("ray.util")
ray_pg_stub = types.ModuleType("ray.util.placement_group")
ray_sched_stub = types.ModuleType("ray.util.scheduling_strategies")


def _remote(*args, **_kwargs):
    if args and len(args) == 1 and callable(args[0]):
        return args[0]

    def _decorator(obj):
        return obj

    return _decorator


def _placement_group(*_args, **_kwargs):
    return MagicMock(name="placement_group")


class _PlacementGroup:
    pass


class _PlacementGroupSchedulingStrategy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NodeAffinitySchedulingStrategy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


ray_stub.remote = _remote
ray_stub.ObjectRef = object
ray_stub.util = ray_util_stub
ray_pg_stub.PlacementGroup = _PlacementGroup
ray_pg_stub.placement_group = _placement_group
ray_sched_stub.PlacementGroupSchedulingStrategy = _PlacementGroupSchedulingStrategy
ray_sched_stub.NodeAffinitySchedulingStrategy = _NodeAffinitySchedulingStrategy
sys.modules["ray"] = ray_stub
sys.modules["ray.util"] = ray_util_stub
sys.modules["ray.util.placement_group"] = ray_pg_stub
sys.modules["ray.util.scheduling_strategies"] = ray_sched_stub

train_group_stub = types.ModuleType("torchspec.ray.train_group")
train_group_stub.RayTrainGroup = object
sys.modules["torchspec.ray.train_group"] = train_group_stub

from torchspec.ray.placement_group import (  # noqa: E402
    _NodeConstraint,
    _build_custom_bundles,
    _sort_probed_bundle_infos,
    create_placement_groups,
)


def _make_args(**overrides):
    defaults = dict(
        placement_strategy="training_first",
        debug_train_only=False,
        debug_inference_only=False,
        colocate=False,
        training_num_nodes=1,
        training_num_gpus_per_node=2,
        inference_num_gpus=2,
        inference_num_gpus_per_node=2,
        training_node_ips=None,
        inference_node_ips=None,
        training_node_selectors=None,
        inference_node_selectors=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_custom_ip_bundles_add_node_resources():
    bundles, selectors, node_groups = _build_custom_bundles(
        "training",
        [_NodeConstraint(ip="10.0.0.1"), _NodeConstraint(ip="10.0.0.2")],
        total_gpus=3,
        gpus_per_node=2,
    )

    assert bundles == [
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
        {"GPU": 1, "CPU": 1, "node:10.0.0.2": 0.001},
    ]
    assert selectors == [{}, {}, {}]
    assert node_groups == [0, 0, 1]


def test_custom_label_bundles_add_bundle_label_selectors():
    bundles, selectors, node_groups = _build_custom_bundles(
        "inference",
        [
            _NodeConstraint(label_selector=(("torchspec/node", "infer-0"),)),
            _NodeConstraint(label_selector=(("torchspec/node", "infer-1"),)),
        ],
        total_gpus=3,
        gpus_per_node=2,
    )

    assert bundles == [
        {"GPU": 1, "CPU": 1},
        {"GPU": 1, "CPU": 1},
        {"GPU": 1, "CPU": 1},
    ]
    assert selectors == [
        {"torchspec/node": "infer-0"},
        {"torchspec/node": "infer-0"},
        {"torchspec/node": "infer-1"},
    ]
    assert node_groups == [0, 0, 1]


def test_custom_bundles_validate_node_count():
    with pytest.raises(ValueError, match="expected 2 node"):
        _build_custom_bundles(
            "training",
            [_NodeConstraint(ip="10.0.0.1")],
            total_gpus=3,
            gpus_per_node=2,
        )


def test_custom_bundle_sort_preserves_user_node_order_then_gpu_id():
    gpu_ids = [
        ("10.0.0.2", 7),
        ("10.0.0.1", 1),
        ("10.0.0.2", 0),
        ("10.0.0.1", 0),
    ]

    sorted_infos = _sort_probed_bundle_infos(gpu_ids, node_group_indices=[1, 0, 1, 0])

    assert [info[0] for info in sorted_infos] == [3, 1, 2, 0]


def test_create_placement_groups_requires_custom_strategy_for_custom_fields():
    args = _make_args(training_node_ips=["10.0.0.1"])

    with (
        patch("torchspec.ray.placement_group._ensure_ray_initialized"),
        patch("torchspec.ray.placement_group._wait_for_gpu_resources") as wait_for_gpus,
        pytest.raises(ValueError, match="placement_strategy=custom"),
    ):
        create_placement_groups(args)

    wait_for_gpus.assert_not_called()


def test_create_placement_groups_validates_custom_constraints_before_waiting_for_gpus():
    args = _make_args(
        placement_strategy="custom",
        training_num_nodes=2,
        training_num_gpus_per_node=2,
        training_node_ips=["10.0.0.1"],
        inference_node_ips=["10.0.0.2"],
    )

    with (
        patch("torchspec.ray.placement_group._ensure_ray_initialized"),
        patch("torchspec.ray.placement_group._wait_for_gpu_resources") as wait_for_gpus,
        pytest.raises(ValueError, match="training custom placement expected 2 node"),
    ):
        create_placement_groups(args)

    wait_for_gpus.assert_not_called()


def test_create_placement_groups_custom_unified_uses_role_node_order():
    args = _make_args(
        placement_strategy="custom",
        training_num_nodes=1,
        training_num_gpus_per_node=2,
        inference_num_gpus=2,
        inference_num_gpus_per_node=2,
        training_node_ips=["10.0.0.1"],
        inference_node_ips=["10.0.0.2"],
    )
    fake_pg = MagicMock(name="pg")

    with (
        patch("torchspec.ray.placement_group._ensure_ray_initialized"),
        patch("torchspec.ray.placement_group._wait_for_gpu_resources"),
        patch(
            "torchspec.ray.placement_group._create_placement_group",
            return_value=(fake_pg, [0, 1, 2, 3], [0, 1, 0, 1]),
        ) as create_pg,
    ):
        result = create_placement_groups(args)

    assert create_pg.call_count == 1
    kwargs = create_pg.call_args.kwargs
    assert kwargs["bundles"][0]["node:10.0.0.1"] == 0.001
    assert kwargs["bundles"][2]["node:10.0.0.2"] == 0.001
    assert kwargs["node_group_indices"] == [0, 0, 1, 1]
    assert result["training"] == (fake_pg, [0, 1], [0, 1])
    assert result["inference"] == (fake_pg, [2, 3], [0, 1])


def test_create_placement_groups_custom_unified_allows_zero_inference_gpus():
    args = _make_args(
        placement_strategy="custom",
        training_num_nodes=1,
        training_num_gpus_per_node=2,
        inference_num_gpus=0,
        inference_num_gpus_per_node=2,
        training_node_ips=["10.0.0.1"],
        inference_node_ips=None,
    )
    fake_pg = MagicMock(name="pg")

    with (
        patch("torchspec.ray.placement_group._ensure_ray_initialized"),
        patch("torchspec.ray.placement_group._wait_for_gpu_resources"),
        patch(
            "torchspec.ray.placement_group._create_placement_group",
            return_value=(fake_pg, [0, 1], [0, 1]),
        ) as create_pg,
    ):
        result = create_placement_groups(args)

    kwargs = create_pg.call_args.kwargs
    assert kwargs["bundles"] == [
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
    ]
    assert kwargs["node_group_indices"] == [0, 0]
    assert result["training"] == (fake_pg, [0, 1], [0, 1])
    assert result["inference"] == (fake_pg, [], [])


def test_custom_colocate_uses_training_topology_for_inference_constraints():
    args = _make_args(
        placement_strategy="custom",
        colocate=True,
        training_num_nodes=2,
        training_num_gpus_per_node=4,
        inference_num_gpus_per_node=8,
        training_node_ips=None,
        inference_node_ips=["10.0.0.1", "10.0.0.2"],
    )
    fake_pg = MagicMock(name="pg")

    with (
        patch("torchspec.ray.placement_group._ensure_ray_initialized"),
        patch("torchspec.ray.placement_group._wait_for_gpu_resources"),
        patch(
            "torchspec.ray.placement_group._create_placement_group",
            return_value=(fake_pg, list(range(8)), list(range(8))),
        ) as create_pg,
    ):
        result = create_placement_groups(args)

    kwargs = create_pg.call_args.kwargs
    assert len(kwargs["bundles"]) == 8
    assert kwargs["bundles"][0]["node:10.0.0.1"] == 0.001
    assert kwargs["bundles"][4]["node:10.0.0.2"] == 0.001
    assert kwargs["node_group_indices"] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert result["training"] == (fake_pg, list(range(8)), list(range(8)))
    assert result["inference"] == (fake_pg, list(range(8)), list(range(8)))

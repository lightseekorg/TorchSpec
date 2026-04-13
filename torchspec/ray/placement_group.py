# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import socket
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from torchspec.ray.train_group import RayTrainGroup
from torchspec.utils.logging import logger


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus, strategy="PACK", name=None):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy=strategy, name=name)
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def _ensure_ray_initialized():
    """Connect to an existing Ray cluster, or start a local instance as fallback."""
    if ray.is_initialized():
        return

    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        logger.info(f"Connected to Ray cluster at {ray_address}")
    except ConnectionError:
        logger.warning("No existing Ray cluster found, starting a local instance")
        ray.init(ignore_reinit_error=True)


def _get_expected_gpu_count(args) -> int:
    training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    inference_gpus = getattr(args, "inference_num_gpus", 0)
    if (
        getattr(args, "colocate", False)
        or getattr(args, "debug_train_only", False)
        or getattr(args, "debug_inference_only", False)
    ):
        return max(training_gpus, inference_gpus)
    return training_gpus + inference_gpus


def _wait_for_gpu_resources(expected_gpus: int, timeout: int = 300, poll_interval: int = 5):
    """Block until the Ray cluster has at least ``expected_gpus`` GPUs."""
    available = int(ray.cluster_resources().get("GPU", 0))
    if available >= expected_gpus:
        logger.info(f"Ray cluster has {available} GPUs (need {expected_gpus})")
        return

    logger.info(f"Waiting for {expected_gpus} GPUs (currently {available}), timeout={timeout}s...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        available = int(ray.cluster_resources().get("GPU", 0))
        logger.info(f"Ray cluster GPUs: {available}/{expected_gpus}")
        if available >= expected_gpus:
            logger.info(f"All {expected_gpus} GPUs available")
            return

    raise RuntimeError(
        f"Timed out waiting for GPUs: {available}/{expected_gpus} after {timeout}s. "
        f"Check that all Ray worker nodes have joined the cluster."
    )


def create_placement_groups(args):
    """Initialize Ray, wait for GPU resources, and create placement groups.

    This is the single entry point for all GPU placement setup.
    """
    _ensure_ray_initialized()
    _wait_for_gpu_resources(_get_expected_gpu_count(args))

    if args.debug_train_only:
        num_training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
        logger.info(f"Creating training placement group with {num_training_gpus} GPUs...")
        training_pg, training_bundle_indices, training_gpu_ids = _create_placement_group(
            num_training_gpus, strategy="PACK", name="training_pg"
        )
        return {
            "training": (training_pg, training_bundle_indices, training_gpu_ids),
            "inference": (training_pg, [], []),
        }

    if args.debug_inference_only:
        num_inference_gpus = args.inference_num_gpus
        logger.info(f"Creating inference placement group with {num_inference_gpus} GPUs...")
        inference_pg, inference_bundle_indices, inference_gpu_ids = _create_placement_group(
            num_inference_gpus, strategy="PACK", name="inference_pg"
        )
        return {
            "training": (inference_pg, [], []),
            "inference": (inference_pg, inference_bundle_indices, inference_gpu_ids),
        }

    if args.colocate:
        num_gpus = args.training_num_nodes * args.training_num_gpus_per_node
        logger.info(f"Creating colocated placement group with {num_gpus} GPUs...")
        pg, bundle_indices, gpu_ids = _create_placement_group(
            num_gpus, strategy="PACK", name="colocate_pg"
        )
        return {
            "training": (pg, bundle_indices, gpu_ids),
            "inference": (pg, bundle_indices, gpu_ids),
        }

    num_training_gpus = args.training_num_nodes * args.training_num_gpus_per_node
    num_inference_gpus = args.inference_num_gpus
    total_gpus = num_training_gpus + num_inference_gpus

    # Single PG ensures deterministic node-to-role assignment across restarts,
    # avoiding kernel/weight cache misses from random GPU shuffling.
    logger.info(
        f"Creating unified placement group with {total_gpus} GPUs "
        f"({num_training_gpus} training + {num_inference_gpus} inference)..."
    )

    pg, sorted_bundle_indices, sorted_gpu_ids = _create_placement_group(
        total_gpus, strategy="PACK", name="unified_pg"
    )

    placement_strategy = getattr(args, "placement_strategy", "training_first")

    if placement_strategy == "training_first":
        training_bundle_indices = sorted_bundle_indices[:num_training_gpus]
        training_gpu_ids = sorted_gpu_ids[:num_training_gpus]
        inference_bundle_indices = sorted_bundle_indices[num_training_gpus:]
        inference_gpu_ids = sorted_gpu_ids[num_training_gpus:]
    else:
        inference_bundle_indices = sorted_bundle_indices[:num_inference_gpus]
        inference_gpu_ids = sorted_gpu_ids[:num_inference_gpus]
        training_bundle_indices = sorted_bundle_indices[num_inference_gpus:]
        training_gpu_ids = sorted_gpu_ids[num_inference_gpus:]

    logger.info(
        f"Placement (strategy={placement_strategy}): "
        f"training bundles={training_bundle_indices}, "
        f"inference bundles={inference_bundle_indices}"
    )

    return {
        "training": (pg, training_bundle_indices, training_gpu_ids),
        "inference": (pg, inference_bundle_indices, inference_gpu_ids),
    }


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, training_class=None):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        training_class=training_class,
    )


def create_train_group(args, training_pg, training_class=None, mooncake_config=None):
    train_group = allocate_train_group(
        args=args,
        num_nodes=args.training_num_nodes,
        num_gpus_per_node=args.training_num_gpus_per_node,
        pg=training_pg,
        training_class=training_class,
    )

    some_ids = ray.get(
        train_group.async_init(
            args, role="training", mooncake_config=mooncake_config, with_ref=False
        )
    )

    assert len(set(some_ids)) == 1

    return train_group

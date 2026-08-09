"""Queue consumer that saves inference outputs instead of training on them."""

from __future__ import annotations

import dataclasses

import ray
import torch

from torchspec.offline.dataset import OfflineDataset
from torchspec.ray.ray_actor import RayActor
from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore


@ray.remote(num_cpus=1, num_gpus=0)
class OfflineSavingActor:
    """Consume the normal training queues and persist their tensors."""

    def __init__(self, output_dir, mooncake_config) -> None:
        self.dataset = OfflineDataset(output_dir)
        config = dataclasses.replace(
            mooncake_config,
            local_hostname=RayActor.get_node_ip(),
            global_segment_size=0,
            async_put_pool_size=0,
            enable_gpu_direct=False,
        )
        self.store = EagleMooncakeStore(config)
        self.store.setup(device=torch.device("cpu"))
        self.queues = {}

    def set_queues(self, train_queue, eval_queue) -> None:
        self.queues = {"train": train_queue, "eval": eval_queue}

    def save_from_queue(self, split: str, count: int = 1) -> int:
        """Save exactly ``count`` items and return the number newly written."""
        written = 0
        for _ in range(count):
            sample = self.queues[split].get()
            if not sample.data_id:
                raise ValueError("Offline saving requires TrainSample.data_id")
            if (sample.metadata or {}).get("vllm_pp_layer_manifest") is not None:
                # Under vLLM pipeline parallelism each stage writes its own
                # per-layer fragments and the base key is never written, so a
                # plain get() here would miss the data and then "clean up" a
                # key that does not exist while every fragment leaks.
                raise NotImplementedError(
                    "Offline saving does not support vLLM pipeline parallelism yet; "
                    f"sample {sample.data_id} carries a per-layer PP manifest. "
                    "Generate offline data with vllm_pp_size=1."
                )
            dtypes = {
                key: getattr(torch, value.replace("torch.", ""))
                if isinstance(value, str)
                else value
                for key, value in (sample.tensor_dtypes or {}).items()
            }
            output = self.store.get(
                key=sample.mooncake_key,
                shapes=sample.tensor_shapes,
                dtypes=dtypes,
                device=torch.device("cpu"),
            )
            try:
                written += self.dataset.append(
                    split,
                    data_id=sample.data_id,
                    tensors=output.to_tensor_dict(),
                    packed_loss_mask=sample.packed_loss_mask,
                    metadata=sample.metadata,
                )
            finally:
                self.store.remove_eagle3_tensors(
                    sample.mooncake_key,
                    has_last_hidden_states="last_hidden_states" in sample.tensor_shapes,
                    has_target="target" in sample.tensor_shapes,
                )
        return written

    def counts(self) -> dict[str, int]:
        return {split: self.dataset.count(split) for split in ("train", "eval")}

    def close(self) -> None:
        self.store.close()

"""Tests for MooncakeDataFetcher and create_mooncake_dataloader."""

import queue
import time
from typing import Dict, List, Tuple

import torch

from torchspec.data.utils import pack_loss_mask, serialize_packed_loss_mask
from torchspec.training.data_fetcher import (
    MooncakeDataFetcher,
    MooncakeDataset,
    TrainSample,
    create_mooncake_dataloader,
)


class MockRayQueue:
    """Mock Ray Queue using stdlib queue."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def put(self, item):
        self._q.put(item)

    def get(self, block=True, timeout=None):
        return self._q.get(block=block, timeout=timeout)


class MockTargetOutput:
    """Wraps a dict of tensors with a to_tensor_dict() method, like Eagle3TargetOutput."""

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self._tensors)


class MockMooncakeStore:
    """Mock mooncake store that stores tensors or returns random ones."""

    def __init__(self, latency: float = 0.0):
        self._data: Dict[str, Dict[str, torch.Tensor]] = {}
        self._key_counter = 0
        self.latency = latency
        self.call_count = 0
        self.call_times: List[float] = []

    def put(self, tensors: Dict[str, torch.Tensor]) -> str:
        """Store tensors and return a generated key."""
        key = f"mc_{self._key_counter}"
        self._key_counter += 1
        self._data[key] = tensors
        return key

    def put_tensors(self, key: str, tensors: Dict[str, torch.Tensor]):
        """Store tensors with a provided key."""
        self._data[key] = tensors

    def _create_random_tensor(
        self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Create a random tensor, handling both float and integer dtypes."""
        if dtype in (torch.long, torch.int, torch.int32, torch.int64, torch.int16, torch.int8):
            return torch.randint(0, 100, shape, dtype=dtype, device=device)
        return torch.randn(shape, dtype=dtype, device=device)

    def get(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ) -> "MockTargetOutput":
        self.call_count += 1
        self.call_times.append(time.time())
        if self.latency > 0:
            time.sleep(self.latency)

        if key in self._data:
            tensors = {k: v.to(device) for k, v in self._data[key].items()}
            return MockTargetOutput(tensors)

        tensors = {
            name: self._create_random_tensor(shape, dtypes.get(name, torch.float32), device)
            for name, shape in shapes.items()
        }
        return MockTargetOutput(tensors)

    def remove_eagle3_tensors(
        self, key: str, has_last_hidden_states: bool = False, has_target: bool = True
    ):
        """Remove tensors from store (no-op for mock)."""
        self._data.pop(key, None)


def simple_collator(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack samples into batched tensors."""
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


def make_sample(idx: int) -> TrainSample:
    return TrainSample(
        mooncake_key=f"sample_{idx}",
        tensor_shapes={"input_ids": (128,), "labels": (128,)},
        tensor_dtypes={"input_ids": torch.long, "labels": torch.long},
    )


class TestMooncakeDataset:
    def test_iterates_samples(self):
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(3):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataset = MooncakeDataset(ray_queue, store, device, prefetch_factor=2)
        samples = list(dataset)

        assert len(samples) == 3
        assert store.call_count == 3
        for s in samples:
            assert "input_ids" in s
            assert "labels" in s
            assert s["input_ids"].shape == (1, 128)

    def test_stops_on_none_sentinel(self):
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        ray_queue.put(make_sample(0))
        ray_queue.put(None)

        dataset = MooncakeDataset(ray_queue, store, device, prefetch_factor=2)
        samples = list(dataset)

        assert len(samples) == 1

    def test_usp_sharded_keeps_local_zero_loss_shard_when_global_loss_exists(self):
        """A local all-zero USP shard must not be skipped independently.

        With SP=2, the first rank can own only prompt tokens while the second
        rank owns loss-bearing response tokens.  If rank 0 skips locally and
        rank 1 trains, subsequent USP collectives are ordered differently.
        """
        full_loss_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        packed_loss_mask = serialize_packed_loss_mask(pack_loss_mask(full_loss_mask))
        sample = TrainSample(
            mooncake_key="sample",
            tensor_shapes={
                "input_ids": (1, 8),
                "hidden_states": (1, 8, 2),
                "target": (1, 8, 2),
            },
            tensor_dtypes={
                "input_ids": torch.long,
                "hidden_states": torch.bfloat16,
                "target": torch.bfloat16,
            },
            packed_loss_mask=packed_loss_mask,
            metadata={"usp_sharded": True},
        )

        outputs = []
        for sp_rank in (0, 1):
            ray_queue = MockRayQueue()
            ray_queue.put(sample)
            ray_queue.put(None)

            store = MockMooncakeStore()
            store.put_tensors(
                f"sample_usp{sp_rank}",
                {
                    "input_ids": torch.arange(sp_rank * 4, sp_rank * 4 + 4).view(1, 4),
                    "hidden_states": torch.zeros(1, 4, 2, dtype=torch.bfloat16),
                    "target": torch.zeros(1, 4, 2, dtype=torch.bfloat16),
                },
            )

            dataset = MooncakeDataset(
                ray_queue,
                store,
                torch.device("cpu"),
                usp_enabled=True,
                ttt_length=0,
            )
            dataset._sp_world_size = 2
            dataset._sp_rank = sp_rank
            dataset._sp_ring_size = 1

            tensors, skipped = dataset._usp_get_sharded_item(skip_count=0)
            outputs.append((tensors, skipped))

        rank0_tensors, rank0_skipped = outputs[0]
        rank1_tensors, rank1_skipped = outputs[1]

        assert rank0_tensors is not None
        assert rank1_tensors is not None
        assert rank0_skipped == 0
        assert rank1_skipped == 0
        assert not rank0_tensors["loss_mask"].any()
        assert rank1_tensors["loss_mask"].any()


class TestCreateMooncakeDataloader:
    def test_default_batch_size_is_one(self):
        """Default batch_size=1 yields one sample at a time."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(4):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataloader = create_mooncake_dataloader(
            ray_queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            prefetch_factor=2,
        )

        batches = list(dataloader)
        assert len(batches) == 4
        for batch in batches:
            assert batch["input_ids"].shape == (1, 1, 128)
            assert batch["labels"].shape == (1, 1, 128)

    def test_batch_size_batches_samples_together(self):
        """batch_size > 1 batches multiple samples together (with padding)."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(4):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataloader = create_mooncake_dataloader(
            ray_queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=4,
        )

        batches = list(dataloader)
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape == (4, 1, 128)
        assert batches[0]["labels"].shape == (4, 1, 128)

    def test_handles_incomplete_final_batch(self):
        """Incomplete final batch still yields remaining samples."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(3):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataloader = create_mooncake_dataloader(
            ray_queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=2,
        )

        batches = list(dataloader)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape == (2, 1, 128)
        assert batches[1]["input_ids"].shape == (1, 1, 128)


class TestMooncakeDataFetcher:
    def test_default_batch_size_one(self):
        """Default batch_size=1 yields one sample at a time."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(3):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        fetcher = MooncakeDataFetcher(
            queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
        )

        batches = list(fetcher)
        assert len(batches) == 3
        assert fetcher.batch_size == 1

    def test_batch_size_parameter(self):
        """batch_size parameter controls batching (= per_dp_rank_batch_size)."""
        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(4):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        fetcher = MooncakeDataFetcher(
            queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=4,
        )

        batches = list(fetcher)
        assert len(batches) == 1
        assert fetcher.batch_size == 4


class TestSynchronousFetching:
    def test_fetches_samples_synchronously(self):
        """Verify samples are fetched one at a time synchronously."""
        ray_queue = MockRayQueue()
        num_samples = 4
        store = MockMooncakeStore(latency=0.01)
        device = torch.device("cpu")

        for i in range(num_samples):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        dataloader = create_mooncake_dataloader(
            ray_queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            prefetch_factor=4,
        )

        batch_count = 0
        for batch in dataloader:
            batch_count += 1

        assert batch_count == num_samples
        assert store.call_count == num_samples


class TestCacheEvalSamples:
    """Tests that cache_eval_samples drains individual samples into a flat cache."""

    def test_cache_eval_samples_drains_count(self):
        """Drain 3 samples in one call, verify _eval_cache has 3 entries."""
        import itertools

        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(3):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        fetcher = MooncakeDataFetcher(
            queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=1,
        )

        eval_cache: list[dict] = []

        def cache_eval_samples(count):
            for sample in itertools.islice(fetcher, count):
                cpu_sample = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
                }
                eval_cache.append(cpu_sample)
            return len(eval_cache)

        assert cache_eval_samples(3) == 3
        assert len(eval_cache) == 3

        for sample in eval_cache:
            assert "input_ids" in sample
            assert "labels" in sample

    def test_cache_eval_samples_incremental(self):
        """Drain samples incrementally across multiple calls."""
        import itertools

        ray_queue = MockRayQueue()
        store = MockMooncakeStore()
        device = torch.device("cpu")

        for i in range(5):
            ray_queue.put(make_sample(i))
        ray_queue.put(None)

        fetcher = MooncakeDataFetcher(
            queue=ray_queue,
            mooncake_store=store,
            collator=simple_collator,
            device=device,
            batch_size=1,
        )

        eval_cache: list[dict] = []

        def cache_eval_samples(count):
            for sample in itertools.islice(fetcher, count):
                cpu_sample = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
                }
                eval_cache.append(cpu_sample)
            return len(eval_cache)

        assert cache_eval_samples(2) == 2
        assert cache_eval_samples(3) == 5
        assert len(eval_cache) == 5

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

"""Queue-based data fetching with mooncake store.
Data flow:
  TrainActor -> MooncakeDataFetcher -> MooncakeDataset -> MooncakeStore -> Collator
                     |                      |                  |               |
                iter(fetcher)          queue.get()      store.get(key)     pad & batch
"""

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from ray.util.queue import Queue as RayQueue
from torch.utils.data import DataLoader, IterableDataset

from torchspec.data.utils import resolve_loss_mask
from torchspec.utils.logging import logger


@dataclass
class TrainSample:
    mooncake_key: str
    tensor_shapes: Dict[str, Tuple[int, ...]]
    tensor_dtypes: Optional[Dict[str, torch.dtype]] = None
    packed_loss_mask: Optional[str] = None
    last_turn_loss_only: Optional[bool] = None


class MooncakeDataset(IterableDataset):
    """IterableDataset that loads from mooncake via queue.

    Each DP rank waits on its queue for TrainSample items sent by the
    centralized controller. Data is loaded from mooncake.
    """

    def __init__(
        self,
        ray_queue: RayQueue,
        mooncake_store,
        device: torch.device,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        batch_size: int = 1,
        min_loss_tokens: int = 0,
    ):
        self.ray_queue = ray_queue
        self.mooncake_store = mooncake_store
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.assistant_header_ids = assistant_header_ids
        self.end_token_ids = end_token_ids
        self.dynamic_loss_mask = dynamic_loss_mask
        self.last_turn_loss_only = last_turn_loss_only
        self.skip_after_header = skip_after_header
        self._batch_size = batch_size
        self._min_loss_tokens = min_loss_tokens

    def _load_from_mooncake(self, sample: TrainSample) -> Dict[str, Any]:
        """Load tensors from mooncake key into device memory."""
        dtypes_raw = sample.tensor_dtypes or {}

        # Convert string dtypes to torch.dtype objects
        dtypes = {}
        for key, dtype_val in dtypes_raw.items():
            if isinstance(dtype_val, str):
                # Handle "bfloat16" or "torch.bfloat16" format
                dtype_str = dtype_val.replace("torch.", "")
                dtypes[key] = getattr(torch, dtype_str)
            else:
                dtypes[key] = dtype_val

        # DEBUG: Print the shapes we're requesting
        logger.debug(
            f"_load_from_mooncake: key={sample.mooncake_key}, "
            f"requesting shapes={sample.tensor_shapes}"
        )

        tensors = self.mooncake_store.get(
            key=sample.mooncake_key,
            shapes=sample.tensor_shapes,
            dtypes=dtypes,
            device=self.device,
        )

        tensor_dict = tensors.to_tensor_dict()
        if self._batch_size > 1:
            # Clone to prevent use-after-free: collator holds sample N while
            # fetching N+1, but cleanup frees the Mooncake buffer (Issue 31).
            # Note: clone() converts pinned → unpinned, breaking non_blocking
            # H2D transfers. Only do this when actually needed.
            result = {k: v.clone() for k, v in tensor_dict.items()}
        else:
            # batch_size=1: safe to use pinned views — consumed immediately.
            # Preserves pinned memory for async H2D via non_blocking=True.
            result = dict(tensor_dict)

        self._cleanup_mooncake_data(sample)
        if sample.packed_loss_mask is not None:
            result["packed_loss_mask"] = sample.packed_loss_mask
        if sample.last_turn_loss_only is not None:
            result["last_turn_loss_only"] = sample.last_turn_loss_only
        return result

    def _cleanup_mooncake_data(self, sample: TrainSample) -> None:
        """Remove data from mooncake store to release buffer space."""
        shapes = sample.tensor_shapes or {}
        has_lhs = "last_hidden_states" in shapes
        has_target = "target" in shapes

        self.mooncake_store.remove_eagle3_tensors(
            sample.mooncake_key,
            has_last_hidden_states=has_lhs,
            has_target=has_target,
        )

    def _compute_loss_mask(self, data: Dict[str, Any]) -> torch.Tensor | None:
        return resolve_loss_mask(
            data,
            dynamic_loss_mask=self.dynamic_loss_mask,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples synchronously.

        Blocks waiting for each item from the queue and loads from mooncake.
        Skips samples whose loss mask is all zeros to avoid wasted compute.
        """
        yield_count = 0
        skip_count = 0
        while True:
            logger.debug(f"__iter__: waiting for item from ray_queue (yield_count={yield_count})")
            try:
                item = self.ray_queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"__iter__: Exception waiting for data: {e}, timeout={self.timeout}")
                break

            if item is None:
                logger.debug("__iter__: received None sentinel, stopping iteration")
                break

            logger.debug(f"__iter__: got item, mooncake_key={item.mooncake_key}")
            data = self._load_from_mooncake(item)

            mask = self._compute_loss_mask(data)
            if mask is None:
                skip_count += 1
                logger.warning(
                    f"Skipping sample with all-zero loss mask "
                    f"(mooncake_key={item.mooncake_key}, total_skipped={skip_count})"
                )
                continue

            if self._min_loss_tokens > 0 and isinstance(mask, torch.Tensor) and mask.sum() < self._min_loss_tokens:
                skip_count += 1
                logger.warning(
                    f"Skipping sample with too few loss-masked tokens "
                    f"({int(mask.sum())} < {self._min_loss_tokens}, "
                    f"mooncake_key={item.mooncake_key}, total_skipped={skip_count})"
                )
                continue

            # Note: target is computed in the collator from last_hidden_states for sglang mode

            # Add batch dimension if missing (sglang stores without batch dim)
            for key, tensor in data.items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    # Check if tensor is missing batch dimension
                    # 1D tensors (loss_mask, input_ids) should be 2D: (1, seq_len)
                    # 2D tensors (hidden_states, last_hidden_states) should be 3D: (1, seq_len, dim)
                    if tensor.dim() == 1:
                        data[key] = tensor.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
                    elif tensor.dim() == 2 and key in [
                        "hidden_states",
                        "last_hidden_states",
                        "target",
                    ]:
                        data[key] = tensor.unsqueeze(0)  # (seq_len, dim) -> (1, seq_len, dim)

            # Debug: log all tensor shapes after adding batch dim
            if data:
                shapes_str = {
                    k: v.shape if hasattr(v, "shape") else type(v) for k, v in data.items()
                }
                logger.debug(f"final shapes (with batch dim): {shapes_str}")
            yield_count += 1
            logger.debug(f"__iter__: yielding batch {yield_count}, keys={list(data.keys())}")
            yield data


def create_mooncake_dataloader(
    ray_queue: RayQueue,
    mooncake_store,
    collator: Callable[[List[Dict]], Dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int = 1,
    prefetch_factor: int = 2,
    timeout: Optional[float] = None,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    dynamic_loss_mask: bool = False,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
    min_loss_tokens: int = 0,
) -> DataLoader:
    """Create a DataLoader that fetches from mooncake via queue.

    Data flow:
      Controller (dispatches dispatch_batch_size samples) ->
      Ray Queue (per_dp_rank_batch_size samples per rank) ->
      DataLoader (batches per_dp_rank_batch_size samples together with padding) ->
      Training loop (one iteration per step)

    The collator pads sequences within the batch to the same length.

    Args:
        ray_queue: Ray Queue to receive TrainSample from controller.
        mooncake_store: Mooncake store client for loading tensors.
        collator: Collator for padding and batching samples.
        device: Target device for tensors.
        batch_size: Number of samples per batch (= per_dp_rank_batch_size).
        prefetch_factor: Unused, kept for API compatibility.
        timeout: Timeout in seconds for waiting on queue. None means wait forever.
        assistant_header_ids: Token IDs for assistant header (for loss mask skip check).
        end_token_ids: Token IDs for end of turn (for loss mask skip check).
        dynamic_loss_mask: Whether loss mask is computed dynamically from input_ids.
        last_turn_loss_only: Global fallback for last-turn-only loss masking.

    Returns:
        DataLoader instance.
    """
    dataset = MooncakeDataset(
        ray_queue,
        mooncake_store,
        device,
        prefetch_factor,
        timeout,
        assistant_header_ids=assistant_header_ids,
        end_token_ids=end_token_ids,
        dynamic_loss_mask=dynamic_loss_mask,
        last_turn_loss_only=last_turn_loss_only,
        skip_after_header=skip_after_header,
        batch_size=batch_size,
        min_loss_tokens=min_loss_tokens,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
    )


class MooncakeDataFetcher:
    """Queue-based data fetcher for mooncake with DataLoader backend.

    Provides iteration over training samples that are pushed to a Ray queue
    by the AsyncTrainingController and loaded from mooncake.

    Batch size design:
      - micro_batch_size: Samples per GPU per training step (user config)
      - per_dp_rank_batch_size = micro_batch_size * sp_size (derived)
      - dispatch_batch_size = per_dp_rank_batch_size * dp_size (derived)
      - DataLoader batch_size = per_dp_rank_batch_size (all samples batched together)
      - Training loop does ONE iteration per step

    The collator pads sequences within the batch to the max length.
    """

    def __init__(
        self,
        queue: RayQueue,
        mooncake_store,
        collator: Callable[[List[Dict]], Dict[str, torch.Tensor]],
        device: torch.device,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        min_loss_tokens: int = 0,
    ):
        self.batch_size = batch_size
        self._dataloader = create_mooncake_dataloader(
            ray_queue=queue,
            mooncake_store=mooncake_store,
            collator=collator,
            device=device,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            timeout=timeout,
            assistant_header_ids=assistant_header_ids,
            end_token_ids=end_token_ids,
            dynamic_loss_mask=dynamic_loss_mask,
            last_turn_loss_only=last_turn_loss_only,
            skip_after_header=skip_after_header,
            min_loss_tokens=min_loss_tokens,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self._dataloader)


class PrefetchedDataFetcher:
    """Wraps MooncakeDataFetcher with async pre-fetching.

    A background thread continuously fetches batches from the underlying
    MooncakeDataFetcher (which blocks on Mooncake TCP), staging them in a
    thread-safe queue.  The training loop reads from this queue, overlapping
    data transfer with GPU compute.

    Without prefetch: [data] → [compute] → [data] → [compute]  (sequential)
    With prefetch:    [compute] → [compute] → [compute]         (overlapped)
                      [data]      [data]      [data]

    The background thread starts lazily on the first ``__iter__`` call and
    keeps running across multiple ``itertools.islice`` invocations (one per
    training step).  The training loop simply reads from the shared queue.
    """

    _SENTINEL = object()

    def __init__(
        self,
        inner: MooncakeDataFetcher,
        prefetch_depth: int = 2,
        target_device: Optional[torch.device] = None,
    ):
        self.inner = inner
        self.prefetch_depth = prefetch_depth
        self.target_device = target_device
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._error: Optional[BaseException] = None

    def _prefetch_loop(self) -> None:
        try:
            for batch in self.inner:
                self._queue.put(batch)
        except Exception as e:
            self._error = e
        finally:
            self._queue.put(self._SENTINEL)

    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._thread.start()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._ensure_started()
        return self

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch of tensors to the target device (GPU)."""
        if self.target_device is None:
            return batch
        return {
            k: v.to(self.target_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._error is not None:
            raise self._error
        item = self._queue.get()
        if item is self._SENTINEL:
            if self._error is not None:
                raise self._error
            raise StopIteration
        return self._to_device(item)

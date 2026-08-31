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

"""KV Connector that writes hidden states directly to Mooncake.

vLLM discovers this connector via ``kv_connector_module_path`` in the
``kv_transfer_config`` dict -- no registration in vLLM's factory needed.

Architecture note: vLLM creates separate connector instances for the scheduler
process and each worker process.  Scheduler-side methods (``build_connector_meta``,
``request_finished``) run on one instance; worker-side methods (``save_kv_layer``,
``wait_for_save``) run on another.  They do NOT share state.  Metadata returned
by ``request_finished`` must therefore be pre-computed on the scheduler side.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)

HIDDEN_STATES_DTYPE_STR = "bfloat16"


def _sanitize_mooncake_key(key: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    if sanitized and sanitized[0].isdigit():
        sanitized = "k" + sanitized
    return sanitized


def _extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Extract data from an LBNHC hidden-state cache view.

    ``kv_cache`` has shape ``(num_blocks, num_heads, block_size, head_size)``.
    This matches latest vLLM's standardized hidden-state connector layout.
    """
    block_size = kv_cache.shape[2]
    return kv_cache[
        slot_mapping // block_size,
        :,
        slot_mapping % block_size,
    ][:num_tokens]


def _slot_mapping_from_block_ids(
    block_ids: list[int],
    page_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute a complete slot mapping from finished-request block IDs."""
    block_ids_gpu = torch.tensor(block_ids, dtype=torch.int64, device=device)
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (block_ids_gpu.unsqueeze(1) * page_size + offsets).flatten()[:num_tokens]


@dataclass
class _PendingSave:
    req_id: str
    token_ids: torch.Tensor
    block_ids: list[int]


@dataclass
class MooncakeConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[_PendingSave] = field(default_factory=list)


class MooncakeHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """KV Connector that stores extracted hidden states directly to Mooncake.

    Must be used with vLLM's ``extract_hidden_states`` speculative method.
    Mooncake connection parameters are read from environment variables
    (exported by TorchSpec's VllmEngine before creating the LLM instance).
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._cache_layer_group_id: int = self._find_cache_layer_group_id(kv_cache_config)
        self._block_size = self._get_cache_block_size(
            vllm_config,
            kv_cache_config,
            self._cache_layer_group_id,
        )
        self.cache_layers: list[str] = []

        assert self._vllm_config.speculative_config is not None, (
            "MooncakeHiddenStatesConnector requires 'extract_hidden_states' speculative method"
        )
        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self._layer_ids = list(getattr(spec_config, "eagle_aux_hidden_state_layer_ids", []))
        self.num_hidden_states = len(self._layer_ids)
        self._hidden_size = vllm_config.model_config.get_hidden_size()

        # The last aux layer is the model's final layer (appended by
        # VllmEngine for last_hidden_states capture).  Training hidden
        # states use the remaining layers.
        self._num_training_layers = max(self.num_hidden_states - 1, 1)

        # Scheduler-side state: track requests and pre-computed metadata
        self._request_token_ids: dict[str, list[int]] = {}
        self._req_metadata: dict[str, dict[str, Any]] = {}
        self._pending_saves: dict[str, _PendingSave] = {}

        self._mooncake_store = None
        self._mooncake_setup_done = False
        self._tp_rank: int | None = None
        self._pp_rank: int | None = None
        self._pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self._num_target_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self._kv_cache: torch.Tensor | None = None
        self._check_layer_layout()
        self._check_mooncake_env()

    def _check_layer_layout(self) -> None:
        if not self._layer_ids:
            raise ValueError(
                "MooncakeHiddenStatesConnector requires a non-empty "
                "eagle_aux_hidden_state_layer_ids on the draft model config"
            )
        if any(a >= b for a, b in zip(self._layer_ids, self._layer_ids[1:])):
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be strictly ascending, "
                f"got {self._layer_ids}"
            )
        if self._layer_ids[-1] != self._num_target_layers:
            raise ValueError(
                "Final aux layer id must be the target model's post-last-layer "
                f"slot: got eagle_aux_hidden_state_layer_ids={self._layer_ids} "
                f"with vLLM num_hidden_layers={self._num_target_layers}. The "
                "HF config VllmEngine read and vLLM's own layer count disagree."
            )

    @staticmethod
    def _find_cache_layer_group_id(kv_cache_config) -> int:
        """Find the isolated ``HiddenStateCacheSpec`` group.

        Latest vLLM places the hidden-state cache in group zero on the
        scheduler side and exposes the concrete group specs on workers.
        """
        if kv_cache_config is None:
            return 0

        from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

        groups = kv_cache_config.kv_cache_groups
        group_ids = [
            gid
            for gid, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        ]
        if len(group_ids) == 1:
            return group_ids[0]
        if not group_ids and len(groups) == 1:
            return 0
        raise ValueError(
            "Could not uniquely identify the hidden-state KV cache group "
            f"among {len(groups)} groups: {group_ids}"
        )

    @staticmethod
    def _get_cache_block_size(vllm_config, kv_cache_config, group_id: int) -> int:
        if kv_cache_config is None:
            return vllm_config.cache_config.block_size
        return kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size

    def _get_tp_rank(self) -> int:
        if self._tp_rank is None:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                self._tp_rank = get_tensor_model_parallel_rank()
            except Exception:
                self._tp_rank = 0
        return self._tp_rank

    def _get_pp_rank(self) -> int:
        if self._pp_rank is None:
            try:
                from vllm.distributed import get_pp_group

                self._pp_rank = get_pp_group().rank_in_group
            except Exception:
                self._pp_rank = 0
        return self._pp_rank

    def _local_layer_positions(self) -> list[int]:
        if self._pp_size == 1:
            return list(range(self.num_hidden_states))

        from vllm.distributed.utils import get_pp_indices

        start_layer, end_layer = get_pp_indices(
            self._num_target_layers,
            self._get_pp_rank(),
            self._pp_size,
        )
        return [
            index
            for index, layer_id in enumerate(self._layer_ids)
            if (layer_id == 0 and self._get_pp_rank() == 0) or (start_layer < layer_id <= end_layer)
        ]

    @staticmethod
    def _check_mooncake_env() -> None:
        if not os.environ.get("MOONCAKE_MASTER_SERVER") and not os.environ.get(
            "MOONCAKE_MASTER_HOST"
        ):
            raise RuntimeError(
                "MooncakeHiddenStatesConnector requires MOONCAKE_MASTER_SERVER or "
                "MOONCAKE_MASTER_HOST to be set; without a store no hidden states "
                "are published and training would silently receive nothing."
            )

    def _ensure_mooncake_store(self) -> bool:
        if self._mooncake_setup_done:
            return self._mooncake_store is not None

        if self._get_tp_rank() != 0:
            self._mooncake_setup_done = True
            return False

        from torchspec.config.mooncake_config import MooncakeConfig
        from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

        config = MooncakeConfig.from_env()
        store = EagleMooncakeStore(config)

        device: torch.device | None = None
        if torch.cuda.is_initialized():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        store.setup(device=device)

        self._mooncake_store = store
        self._mooncake_setup_done = True
        logger.info(
            "MooncakeHiddenStatesConnector: store initialized "
            f"(master={config.master_server_address})"
        )
        return True

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        if self._mooncake_store is not None:
            # Do not serialize the engine step behind the RDMA write.  Readers
            # wait for the published keys before moving bytes; here we only
            # surface failures from puts that have already completed.
            self._mooncake_store.check_async_errors()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )
        self._kv_cache = kv_caches[self.cache_layers[0]]
        if self._block_size != self._kv_cache.shape[2]:
            raise ValueError(
                "Hidden-state block-size mismatch: "
                f"derived {self._block_size}, cache view has {self._kv_cache.shape[2]}"
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # Latest vLLM extracts completed hidden-state caches from get_finished.
        # Per-layer metadata is not guaranteed to contain a complete request.
        pass

    def _pending_save_tensors(
        self,
        pending: _PendingSave,
        local_positions: list[int],
    ) -> list[tuple[str, torch.Tensor]]:
        assert self._kv_cache is not None

        num_tokens = pending.token_ids.shape[0]
        slot_mapping = _slot_mapping_from_block_ids(
            pending.block_ids,
            self._block_size,
            num_tokens,
            device=self._kv_cache.device,
        )
        if slot_mapping.shape[0] < num_tokens:
            raise RuntimeError(
                f"Completed request {pending.req_id} has only "
                f"{slot_mapping.shape[0]} hidden-state slots for {num_tokens} tokens"
            )

        hidden_states_3d = _extract_from_kv_cache(
            self._kv_cache,
            slot_mapping,
            num_tokens,
        )
        if hidden_states_3d.dtype != torch.bfloat16:
            hidden_states_3d = hidden_states_3d.to(torch.bfloat16)
        input_ids = pending.token_ids.to(hidden_states_3d.device)
        mooncake_key = _sanitize_mooncake_key(pending.req_id)

        if self._pp_size == 1:
            all_hidden = hidden_states_3d.reshape(num_tokens, -1)
            split_at = self._num_training_layers * self._hidden_size
            return [
                (f"{mooncake_key}_hs", all_hidden[:, :split_at]),
                (f"{mooncake_key}_ids", input_ids),
                (f"{mooncake_key}_lhs", all_hidden[:, -self._hidden_size :]),
            ]

        tensors: list[tuple[str, torch.Tensor]] = []
        for position in local_positions:
            layer_key = f"{mooncake_key}_layer{self._layer_ids[position]}"
            tensors.extend(
                [
                    (f"{layer_key}_hs", hidden_states_3d[:, position, :]),
                    (f"{layer_key}_ids", input_ids),
                ]
            )
        return tensors

    def _publish_pending_saves(self, pending_saves: list[_PendingSave]) -> None:
        local_positions = self._local_layer_positions()
        if self._pp_size > 1 and not local_positions:
            return
        if not self._ensure_mooncake_store():
            return
        assert self._kv_cache is not None
        capacity = int(self._mooncake_store.config.host_buffer_size)
        keys: list[str] = []
        tensors: list[torch.Tensor] = []
        used_bytes = 0

        def submit() -> None:
            nonlocal keys, tensors, used_bytes
            if keys:
                self._mooncake_store.put_raw_tensors(keys, tensors)
            keys = []
            tensors = []
            used_bytes = 0

        for pending in pending_saves:
            for key, tensor in self._pending_save_tensors(pending, local_positions):
                tensor_bytes = tensor.numel() * tensor.element_size()
                if tensor_bytes > capacity:
                    raise RuntimeError(
                        f"Mooncake tensor {key} needs {tensor_bytes} bytes, "
                        f"host buffer has {capacity}"
                    )
                if keys and used_bytes + tensor_bytes > capacity:
                    submit()
                keys.append(key)
                tensors.append(tensor)
                used_bytes += tensor_bytes
        submit()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        pending_saves: list[_PendingSave] = []
        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if isinstance(metadata, MooncakeConnectorMetadata):
                pending_saves = metadata.pending_saves

        if not pending_saves:
            return None, None

        local_error: BaseException | None = None
        try:
            self._publish_pending_saves(pending_saves)
            if self._mooncake_store is not None:
                # The cache gather above creates independent GPU tensors. The
                # store records them on its DtoH stream, so hidden-cache blocks
                # can be released after scheduling while PUTs finish in the
                # background. Surface any already-completed failure without
                # draining the async manager; readers wait for every key.
                self._mooncake_store.check_async_errors()
        except BaseException as exc:
            local_error = exc

        if self._pp_size > 1:
            from vllm.distributed import get_pp_group

            # All stages reach the collective even if a local PUT failed, so a
            # single-stage exception cannot strand its peer before propagation.
            get_pp_group().barrier()

        if local_error is not None:
            raise local_error

        return {pending.req_id for pending in pending_saves}, None

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        meta.pending_saves = list(self._pending_saves.values())
        self._pending_saves.clear()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            self._request_token_ids[new_req.req_id] = list(token_ids)

            seq_len = len(token_ids)
            training_hidden_size = self._num_training_layers * self._hidden_size
            mooncake_key = _sanitize_mooncake_key(new_req.req_id)
            self._req_metadata[new_req.req_id] = {
                "mooncake_key": mooncake_key,
                "tensor_shapes": {
                    "hidden_states": (seq_len, training_hidden_size),
                    "input_ids": (seq_len,),
                    "last_hidden_states": (seq_len, self._hidden_size),
                },
                "tensor_dtypes": {
                    "hidden_states": HIDDEN_STATES_DTYPE_STR,
                    "input_ids": "int64",
                    "last_hidden_states": HIDDEN_STATES_DTYPE_STR,
                },
                "num_layers": self.num_hidden_states,
                "input_ids_list": token_ids,
            }
            if self._pp_size > 1:
                self._req_metadata[new_req.req_id]["pp_layer_manifest"] = [
                    {
                        "layer_id": layer_id,
                        "mooncake_key": f"{mooncake_key}_layer{layer_id}",
                        "role": (
                            "last_hidden_states"
                            if layer_id == self._num_target_layers
                            else "hidden_states"
                        ),
                    }
                    for layer_id in self._layer_ids
                ]

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        token_ids = self._request_token_ids.pop(
            req_id,
            list(request.prompt_token_ids or []),
        )
        mooncake_meta = self._req_metadata.pop(req_id, None)
        self._pending_saves[req_id] = _PendingSave(
            req_id=req_id,
            token_ids=torch.tensor(token_ids, dtype=torch.long),
            block_ids=list(block_ids),
        )
        # Delay freeing the hidden-state cache blocks until get_finished has
        # completed every local Mooncake PUT.
        return True, mooncake_meta

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        cache_group_ids = block_ids[self._cache_layer_group_id] if block_ids else []
        return self.request_finished(request, cache_group_ids)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "LBNHC"

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
    """Extract data from KV cache.

    Assumes kv_cache shape: (num_pages, page_size, num_heads, head_size)
    """
    padded_kv = kv_cache.flatten(0, 1)[slot_mapping]
    return padded_kv[:num_tokens]


@dataclass
class _ReqMeta:
    req_id: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    new_req: bool

    @staticmethod
    def make(
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        new_req: bool,
    ) -> _ReqMeta:
        token_ids_tensor = torch.tensor(token_ids)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()
        return _ReqMeta(
            req_id=req_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            new_req=new_req,
        )


@dataclass
class MooncakeConnectorMetadata(KVConnectorMetadata):
    requests: list[_ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        new_req: bool = True,
    ) -> None:
        self.requests.append(_ReqMeta.make(req_id, token_ids, block_ids, block_size, new_req))


class MooncakeHiddenStatesConnector(KVConnectorBase_V1):
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
        self._block_size = vllm_config.cache_config.block_size
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
        self._active_requests: dict[str, Any] = {}
        self._req_blocks: dict[str, list[int]] = {}
        self._req_metadata: dict[str, dict[str, Any]] = {}

        # Worker-side state: Mooncake store (lazy init)
        self._mooncake_store = None
        self._mooncake_setup_done = False

    def _ensure_mooncake_store(self) -> bool:
        if self._mooncake_setup_done:
            return self._mooncake_store is not None

        if not os.environ.get("MOONCAKE_MASTER_SERVER") and not os.environ.get(
            "MOONCAKE_MASTER_HOST"
        ):
            logger.warning(
                "MooncakeHiddenStatesConnector: no MOONCAKE_MASTER_SERVER env var; "
                "hidden states will NOT be stored."
            )
            self._mooncake_setup_done = True
            return False

        try:
            from torchspec.config.mooncake_config import MooncakeConfig
            from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

            config = MooncakeConfig.from_env()
            self._mooncake_store = EagleMooncakeStore(config)

            device: torch.device | None = None
            if torch.cuda.is_initialized():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._mooncake_store.setup(device=device)
            self._mooncake_setup_done = True
            logger.info(
                "MooncakeHiddenStatesConnector: store initialized "
                f"(master={config.master_server_address})"
            )
            return True
        except Exception:
            logger.exception("MooncakeHiddenStatesConnector: failed to init store")
            self._mooncake_setup_done = True
            return False

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        if self._mooncake_store is not None:
            self._mooncake_store.flush()

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

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, MooncakeConnectorMetadata)

        if not self._ensure_mooncake_store():
            logger.warning("save_kv_layer: Mooncake store not available, skipping")
            return

        for request in connector_metadata.requests:
            num_tokens = request.token_ids.shape[0]
            num_slots = request.slot_mapping.shape[0]

            # With chunked prefill, save_kv_layer is called per chunk.
            # Mooncake keys are write-once (can't overwrite), so we skip
            # partial chunks and only write when all blocks are allocated.
            if num_slots < num_tokens:
                continue

            hidden_states_3d = _extract_from_kv_cache(kv_layer, request.slot_mapping, num_tokens)

            all_hidden = hidden_states_3d.reshape(num_tokens, -1)

            # Split: first N-1 aux layers → draft model input,
            # last aux layer (final model layer) → target logit computation
            split_at = self._num_training_layers * self._hidden_size
            hidden_states = all_hidden[:, :split_at]
            last_hidden_states = all_hidden[:, -self._hidden_size :]

            input_ids = request.token_ids.to(hidden_states.device)

            mooncake_key = _sanitize_mooncake_key(request.req_id)

            try:
                self._mooncake_store.put(
                    key=mooncake_key,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    last_hidden_states=last_hidden_states,
                    target=None,
                )
            except Exception:
                logger.exception(f"save_kv_layer: failed to store to Mooncake for {request.req_id}")

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
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            meta.add_request(
                new_req.req_id,
                token_ids=token_ids,
                block_ids=new_req.block_ids[0],
                block_size=self._block_size,
            )
            self._active_requests[new_req.req_id] = new_req
            self._req_blocks[new_req.req_id] = list(new_req.block_ids[0])

            # Pre-compute metadata that request_finished will return.
            # The mooncake key and shapes are deterministic from the request.
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

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue

            new_block_ids = cached_reqs.new_block_ids[i]
            if new_block_ids is None:
                continue

            cached_req = self._active_requests[req_id]
            req_block_ids = self._req_blocks[req_id]

            block_ids = new_block_ids[0]
            req_block_ids.extend(block_ids)

            meta.add_request(
                req_id=req_id,
                token_ids=cached_req.prompt_token_ids or [],
                block_ids=req_block_ids,
                block_size=self._block_size,
                new_req=False,
            )

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        _ = self._active_requests.pop(req_id, None)
        _ = self._req_blocks.pop(req_id, None)

        mooncake_meta = self._req_metadata.pop(req_id, None)
        return False, mooncake_meta

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "NHD"

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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
VLLM Ray actor engine for distributed deployment.

Uses Worker Extension mode with MultiprocExecutor for reliable hidden states
extraction via model.forward patching in worker processes.
"""

import socket

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from torchspec.inference.engine.base import InferenceEngine
from torchspec.ray.ray_actor import RayActor
from torchspec.utils.logging import logger, setup_file_logging
from torchspec.utils.misc import get_default_eagle3_aux_layer_ids

_PROTECTION_ENGINE_KEYS = frozenset(
    {
        "model",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "nnodes",
        "node_rank",
        "distributed_backend",
    }
)


class VllmEngine(InferenceEngine, RayActor):
    """Ray actor wrapper for vLLM LLM engine with distributed deployment support.

    Uses Worker Extension mode with MultiprocExecutor and VllmWorkerExtension
    for reliable hidden states extraction by patching model.forward in worker processes.
    """

    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int = 1,
        node_rank: int = 0,
        engine_group: int = 0,
    ):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine
        self.node_rank = node_rank
        self._engine = None
        self._mooncake_config = None
        self._mooncake_store = None
        self._hidden_size = None
        self.local_gpu_id = None

        setup_file_logging("inference", self.rank, group=engine_group)

    def init(self, mooncake_config=None, dist_init_addr: str | None = None) -> None:
        if self.base_gpu_id is not None:
            self.local_gpu_id = self.setup_gpu(self.base_gpu_id)
            logger.info(
                f"VllmEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, "
                f"using local GPU {self.local_gpu_id}"
            )

        self._mooncake_config = mooncake_config

        if mooncake_config is not None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "localhost"
                logger.warning(
                    f"VllmEngine rank {self.rank}: failed to get local IP, using localhost"
                )

            mooncake_config.local_hostname = local_ip
            mooncake_config.export_env()

            from torchspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address,
                mooncake_config.metadata_server,
            )

        mem_fraction = getattr(self.args, "vllm_mem_fraction_static", 0.8)
        pp_size = getattr(self.args, "vllm_pp_size", 1)

        if self.args.aux_hidden_states_layers is not None:
            self.aux_hidden_state_layer_ids = self.args.aux_hidden_states_layers
        else:
            self.aux_hidden_state_layer_ids = get_default_eagle3_aux_layer_ids(
                self.args.target_model_path
            )
            if self.rank == 0:
                logger.info(
                    f"Using default aux hidden state layer ids: {self.aux_hidden_state_layer_ids}"
                )

        nnodes = getattr(self.args, "vllm_nnodes", 1)
        tp_size = nnodes * self.num_gpus_per_engine

        logger.info(
            f"VllmEngine rank {self.rank}: BEFORE init - "
            f"base_gpu_id={self.base_gpu_id}, num_gpus={self.num_gpus_per_engine}, "
            f"tp_size={tp_size}, pp_size={pp_size}, nnodes={nnodes}, node_rank={self.node_rank}, "
            f"aux_hidden_state_layer_ids={self.aux_hidden_state_layer_ids}"
        )

        self._init_engine(tp_size, pp_size, nnodes, mem_fraction, dist_init_addr)

        self._hidden_size = self._get_hidden_size_from_engine()

        if self._mooncake_config is not None:
            self._init_mooncake_store()

        logger.info(
            f"VllmEngine rank {self.rank}: initialized from {self.args.target_model_path} "
            f"(tp_size={tp_size}, aux_layers={self.aux_hidden_state_layer_ids}, hidden_size={self._hidden_size})"
        )

    def _init_engine(
        self,
        tp_size: int,
        pp_size: int,
        nnodes: int,
        mem_fraction: float,
        dist_init_addr: str | None,
    ) -> None:
        """Initialize LLM with worker extension enabled."""
        from vllm import LLM

        engine_kwargs = {
            "model": self.args.target_model_path,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": mem_fraction,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
            "distributed_executor_backend": "mp",
            "disable_custom_all_reduce": True,
            "worker_extension_cls": (
                "torchspec.inference.engine.vllm_worker_extension.VllmWorkerExtension"
            ),
        }

        extra_args = getattr(self.args, "vllm_extra_args", None)
        if extra_args:
            if isinstance(extra_args, DictConfig):
                extra = OmegaConf.to_container(extra_args, resolve=True)
            else:
                extra = dict(extra_args) if not isinstance(extra_args, dict) else extra_args
            blocked = extra.keys() & _PROTECTION_ENGINE_KEYS
            if blocked:
                logger.warning(
                    f"vllm extra_args contains protected keys that will be ignored: "
                    f"{sorted(blocked)}. These are managed internally by TorchSpec."
                )
                extra = {k: v for k, v in extra.items() if k not in _PROTECTION_ENGINE_KEYS}
            engine_kwargs.update(extra)

        inference_batch_size = getattr(self.args, "inference_batch_size", None)
        if inference_batch_size is not None:
            comp_cfg = engine_kwargs.get("compilation_config", {})
            if isinstance(comp_cfg, dict) and "max_cudagraph_capture_size" not in comp_cfg:
                comp_cfg["max_cudagraph_capture_size"] = inference_batch_size
                engine_kwargs["compilation_config"] = comp_cfg
                logger.info(
                    f"VllmEngine rank {self.rank}: defaulting "
                    f"max_cudagraph_capture_size={inference_batch_size} from inference_batch_size"
                )

        # Disable prefix caching and chunked prefill
        engine_kwargs["enable_prefix_caching"] = False
        engine_kwargs["enable_chunked_prefill"] = False

        max_seq_length = getattr(self.args, "max_seq_length", None)
        if max_seq_length:
            engine_kwargs["max_model_len"] = max_seq_length

        if nnodes > 1:
            engine_kwargs["nnodes"] = nnodes
            engine_kwargs["node_rank"] = self.node_rank
            if dist_init_addr:
                engine_kwargs["distributed_backend"] = "nccl"
                engine_kwargs["distributed_init_address"] = dist_init_addr

        self._engine = LLM(**engine_kwargs)
        self._setup_rpc_hidden_states_capture()
        logger.info(
            f"VllmEngine rank {self.rank}: initialized worker extension mode "
            f"with layers={self.aux_hidden_state_layer_ids}"
        )

    def _setup_rpc_hidden_states_capture(self) -> None:
        """Initialize worker-side hidden-state capture hooks."""
        if self._engine is None:
            raise RuntimeError("VllmEngine not initialized. Call init() first.")
        if not hasattr(self._engine, "collective_rpc"):
            raise RuntimeError("vLLM LLM.collective_rpc is required for worker extension mode")

        # Set environment variables so workers can connect to Mooncake
        if self._mooncake_config is not None:
            import os

            os.environ["TORCHSPEC_MOONCAKE_MASTER_ADDR"] = (
                self._mooncake_config.master_server_address
            )
            os.environ["TORCHSPEC_MOONCAKE_METADATA_PORT"] = str(
                self._mooncake_config.metadata_server.split(":")[-1].replace("/metadata", "")
            )
            os.environ["TORCHSPEC_MOONCAKE_LOCAL_HOSTNAME"] = self._mooncake_config.local_hostname
            os.environ["TORCHSPEC_MOONCAKE_PROTOCOL"] = self._mooncake_config.protocol
            if self._mooncake_config.device_name:
                os.environ["TORCHSPEC_MOONCAKE_DEVICE_NAME"] = self._mooncake_config.device_name
            logger.info(
                f"VllmEngine rank {self.rank}: Set Mooncake env vars for workers: "
                f"master={self._mooncake_config.master_server_address}"
            )

        layer_ids = list(self.aux_hidden_state_layer_ids)
        results = self._engine.collective_rpc(
            "_setup_hidden_states_capture",
            args=(layer_ids,),
        )
        logger.info(f"VllmEngine rank {self.rank}: worker capture setup replies={results}")

    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict]:
        """Generate hidden states for training data using Worker Extension mode."""
        if self._engine is None:
            raise RuntimeError("VllmEngine not initialized. Call init() first.")

        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        use_prompts = formatted_prompts is not None
        input_ids_list: list[torch.Tensor] | None = None

        if use_prompts:
            batch_size = len(formatted_prompts)
            prompts = formatted_prompts
        else:
            if isinstance(input_ids_ref, ray.ObjectRef):
                input_ids_list = ray.get(input_ids_ref)
            else:
                input_ids_list = input_ids_ref
            if input_ids_list is None:
                raise ValueError("input_ids_ref resolved to None")
            batch_size = len(input_ids_list)
            prompts = self._format_input_ids_for_vllm(input_ids_list)

        if isinstance(data_id, str):
            data_ids = [f"{data_id}_{i}" for i in range(batch_size)]
        elif len(data_id) == batch_size:
            data_ids = data_id
        else:
            raise ValueError(
                f"data_id length {len(data_id)} does not match batch size {batch_size}"
            )

        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=1, temperature=0)
        request_metadata = {}
        if input_ids_list is not None:
            for i, ids in enumerate(input_ids_list):
                request_metadata[data_ids[i]] = int(self._normalize_input_ids(ids).numel())

        # Build packed_loss_mask_map for workers
        packed_loss_mask_map = {}
        if packed_loss_mask_list is not None:
            for i, data_id in enumerate(data_ids):
                if i < len(packed_loss_mask_list):
                    packed_loss_mask_map[data_id] = packed_loss_mask_list[i]

        # Build input_ids_map for workers (pass real input_ids via RPC)
        input_ids_map = {}
        if input_ids_list is not None:
            for i, data_id in enumerate(data_ids):
                if i < len(input_ids_list):
                    ids = self._normalize_input_ids(input_ids_list[i])
                    input_ids_map[data_id] = ids.cpu().tolist()

        try:
            self._engine.collective_rpc("_reset_capture")
            if request_metadata:
                self._engine.collective_rpc(
                    "_set_request_metadata",
                    args=(request_metadata, packed_loss_mask_map, input_ids_map),
                )
        except Exception as e:
            logger.warning(f"Could not reset capture via worker extension: {e}")

        outputs = self._engine.generate(prompts, sampling_params, use_tqdm=False)

        # outputs are sorted by int(request_id), matching submission order.
        # Build mapping from vLLM's internal worker IDs ("{request_id}-{uuid}")
        # to our external data_ids.
        internal_to_external = {}
        for i, output in enumerate(outputs):
            internal_to_external[output.request_id] = data_ids[i]

        # For the formatted_prompts path, request_metadata and input_ids_map
        # were not set before generation (no input_ids available).  Build them
        # from the outputs so the worker can map captured states to requests.
        if use_prompts and not request_metadata:
            for i, output in enumerate(outputs):
                did = data_ids[i]
                request_metadata[did] = len(output.prompt_token_ids)
                input_ids_map[did] = list(output.prompt_token_ids)
            try:
                self._engine.collective_rpc(
                    "_set_request_metadata",
                    args=(request_metadata, packed_loss_mask_map, input_ids_map),
                )
            except Exception as e:
                logger.warning(
                    f"VllmEngine rank {self.rank}: Could not set post-generation "
                    f"request metadata: {e}"
                )

        # Get metadata from workers (tensors are already stored in Mooncake by workers)
        metadata_by_request: dict[str, dict] = {}
        try:
            metadata_list = self._engine.collective_rpc(
                "_store_and_get_metadata", args=(internal_to_external,)
            )
            if isinstance(metadata_list, list):
                for metadata in metadata_list:
                    if isinstance(metadata, dict):
                        metadata_by_request.update(metadata)
            elif isinstance(metadata_list, dict):
                metadata_by_request = metadata_list
        except Exception as e:
            logger.warning(f"Could not get metadata from worker extension: {e}")

        if not metadata_by_request:
            logger.error(
                f"VllmEngine rank {self.rank}: metadata_by_request is EMPTY for "
                f"data_ids={data_ids}. Worker returned metadata_list={metadata_list!r}. "
                f"use_prompts={use_prompts}, request_metadata_keys={list(request_metadata.keys())}, "
                f"internal_to_external={internal_to_external}"
            )

        results = []
        for i, output in enumerate(outputs):
            seq_len = len(output.prompt_token_ids)
            data_id = data_ids[i]

            # Get metadata for this request
            metadata = metadata_by_request.get(data_id)
            if metadata is None:
                logger.error(
                    f"VllmEngine rank {self.rank}: No metadata for data_id={data_id}. "
                    f"metadata_by_request has keys={list(metadata_by_request.keys())}. "
                    f"Training may be corrupted."
                )
                continue

            # Extract info from metadata (tensors are already in Mooncake)
            mooncake_key = metadata.get("mooncake_key", data_id)
            tensor_shapes = metadata.get("tensor_shapes", {})
            tensor_dtypes = metadata.get("tensor_dtypes", {})

            result = {
                "mooncake_key": mooncake_key,
                "tensor_shapes": tensor_shapes,
                "tensor_dtypes": tensor_dtypes,
                "data_id": data_id,
                "seq_len": seq_len,
            }
            # Get packed_loss_mask from metadata (returned by worker)
            packed_loss_mask = metadata.get("packed_loss_mask")
            if packed_loss_mask is not None:
                result["packed_loss_mask"] = packed_loss_mask
            # Get input_ids_list from metadata (returned by worker via RPC)
            input_ids_list = metadata.get("input_ids_list")
            if input_ids_list is not None:
                result["input_ids_list"] = input_ids_list
            results.append(result)

        # No need to flush here - workers already flushed after storing

        logger.debug(
            f"VllmEngine rank {self.rank}: generated {len(results)} mooncake results "
            f"for data_ids={data_ids}"
        )
        return results

    def _init_mooncake_store(self) -> None:
        if self._mooncake_store is not None or self._mooncake_config is None:
            return
        from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

        self._mooncake_store = EagleMooncakeStore(self._mooncake_config)
        if torch.cuda.is_available():
            self._mooncake_store.setup(device=torch.cuda.current_device())
        else:
            self._mooncake_store.setup()

    def _normalize_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            return input_ids.squeeze(0)
        if input_ids.dim() == 1:
            return input_ids
        raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")

    def _format_input_ids_for_vllm(
        self, input_ids_list: list[torch.Tensor]
    ) -> list[dict[str, list[int]]]:
        prompts = []
        for ids in input_ids_list:
            prompts.append({"prompt_token_ids": self._normalize_input_ids(ids).tolist()})
        return prompts

    def health_check(self, timeout: float = 5.0) -> bool:
        return self._engine is not None

    def shutdown(self) -> None:
        if self._mooncake_store is not None:
            try:
                self._mooncake_store.close()
            except Exception as e:
                logger.warning(f"VllmEngine rank {self.rank}: Error closing mooncake store: {e}")
            self._mooncake_store = None

        if self._engine is not None:
            del self._engine
            self._engine = None

        logger.info(f"VllmEngine rank {self.rank}: shutdown complete")

    def get_status(self) -> dict:
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
            "hidden_size": self._hidden_size,
        }

    def _get_hidden_size_from_engine(self) -> int:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not determine hidden_size from model config: {self.args.target_model_path}"
            )
        return hidden_size

    def _get_tensor_shapes(self, seq_len: int) -> dict:
        aux_hidden_state_layer_ids = self.aux_hidden_state_layer_ids
        num_aux_layers = len(aux_hidden_state_layer_ids)
        if self._hidden_size is None:
            raise ValueError(
                f"VllmEngine rank {self.rank}: hidden_size not initialized. Call init() first."
            )
        hidden_size = self._hidden_size

        concat_hidden_size = num_aux_layers * hidden_size

        return {
            "hidden_states": (seq_len, concat_hidden_size),
            "input_ids": (seq_len,),
            "last_hidden_states": (seq_len, hidden_size),
        }

    def _get_tensor_dtypes(self) -> dict:
        return {
            "hidden_states": torch.bfloat16,
            "input_ids": torch.long,
            "last_hidden_states": torch.bfloat16,
        }

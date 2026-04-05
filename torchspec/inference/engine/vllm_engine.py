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

Uses vLLM's ``extract_hidden_states`` speculative decoding method with a
custom ``MooncakeHiddenStatesConnector`` KV Connector to capture intermediate
hidden states and store them directly to Mooncake via RDMA.

This replaces the previous worker-extension approach that monkey-patched
``model.forward``.  The new approach uses only public vLLM APIs
(``speculative_config`` + ``kv_transfer_config``) and is compatible with
MRV2, CUDA graphs, and ``torch.compile``.
"""

import socket
from typing import Any

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from torchspec.inference.engine.base import InferenceEngine
from torchspec.ray.ray_actor import RayActor
from torchspec.transfer.mooncake.eagle_store import HIDDEN_STATES_STORAGE_DTYPE
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
        "speculative_config",
        "kv_transfer_config",
    }
)


class VllmEngine(InferenceEngine, RayActor):
    """Ray actor wrapper for vLLM LLM engine with distributed deployment support.

    Uses vLLM's ``extract_hidden_states`` speculative method with a
    ``MooncakeHiddenStatesConnector`` to capture hidden states from selected
    model layers and write them directly to Mooncake.
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
            # Export env vars so worker processes (and the connector) can
            # initialize their own Mooncake stores via MooncakeConfig.from_env().
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
            self.aux_hidden_state_layer_ids = list(self.args.aux_hidden_states_layers)
        else:
            self.aux_hidden_state_layer_ids = get_default_eagle3_aux_layer_ids(
                self.args.target_model_path
            )
            if self.rank == 0:
                logger.info(
                    f"Using default aux hidden state layer ids: {self.aux_hidden_state_layer_ids}"
                )

        # The connector can only access aux layer outputs from the KV cache,
        # so we append the model's final layer to capture last_hidden_states
        # (pre-norm) for target logit computation on the training side.
        from transformers import AutoConfig as _AC

        _cfg = _AC.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        _cfg = getattr(_cfg, "text_config", _cfg)
        final_layer_id = _cfg.num_hidden_layers - 1
        if final_layer_id not in self.aux_hidden_state_layer_ids:
            self.aux_hidden_state_layer_ids.append(final_layer_id)
            if self.rank == 0:
                logger.info(
                    f"Appended final layer {final_layer_id} to aux layers for "
                    f"last_hidden_states: {self.aux_hidden_state_layer_ids}"
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
        """Initialize LLM with extract_hidden_states speculative config."""
        from vllm import LLM

        engine_kwargs = {
            "model": self.args.target_model_path,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": mem_fraction,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
            "distributed_executor_backend": "mp",
            "disable_custom_all_reduce": True,
            "speculative_config": {
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": list(self.aux_hidden_state_layer_ids)
                    }
                },
            },
            "kv_transfer_config": {
                "kv_connector": "MooncakeHiddenStatesConnector",
                "kv_connector_module_path": (
                    "torchspec.inference.engine.mooncake_hidden_states_connector"
                ),
                "kv_role": "kv_producer",
            },
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

        engine_kwargs["enable_prefix_caching"] = False

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
        logger.info(
            f"VllmEngine rank {self.rank}: initialized extract_hidden_states mode "
            f"with layers={self.aux_hidden_state_layer_ids}"
        )

    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str | None] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict]:
        """Generate hidden states for training data.

        Hidden states are captured by vLLM's ``extract_hidden_states``
        speculative method and stored to Mooncake by the
        ``MooncakeHiddenStatesConnector``.  Metadata comes back in
        ``output.kv_transfer_params``.
        """
        if self._engine is None:
            raise RuntimeError("VllmEngine not initialized. Call init() first.")

        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        use_prompts = formatted_prompts is not None
        input_ids_list: list[torch.Tensor] | None = None

        if use_prompts:
            batch_size = len(formatted_prompts)
        else:
            if isinstance(input_ids_ref, ray.ObjectRef):
                input_ids_list = ray.get(input_ids_ref)
            else:
                input_ids_list = input_ids_ref
            if input_ids_list is None:
                raise ValueError("input_ids_ref resolved to None")
            batch_size = len(input_ids_list)

        prompts = self._build_prompts(
            formatted_prompts=formatted_prompts if use_prompts else None,
            input_ids_list=input_ids_list,
            multimodal_inputs=multimodal_inputs,
            batch_size=batch_size,
        )

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

        # Build packed_loss_mask_map for result assembly
        packed_loss_mask_map: dict[str, str | None] = {}
        if packed_loss_mask_list is not None:
            for i, did in enumerate(data_ids):
                if i < len(packed_loss_mask_list):
                    packed_loss_mask_map[did] = packed_loss_mask_list[i]

        outputs = self._engine.generate(prompts, sampling_params, use_tqdm=False)

        results = []
        for i, output in enumerate(outputs):
            seq_len = len(output.prompt_token_ids)
            did = data_ids[i]

            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is None:
                logger.error(
                    f"VllmEngine rank {self.rank}: No kv_transfer_params for data_id={did}. "
                    f"The MooncakeHiddenStatesConnector may not have stored this request."
                )
                continue

            mooncake_key = kv_params.get("mooncake_key", did)
            tensor_shapes = kv_params.get("tensor_shapes", {})
            tensor_dtypes = kv_params.get("tensor_dtypes", {})

            result: dict[str, Any] = {
                "mooncake_key": mooncake_key,
                "tensor_shapes": tensor_shapes,
                "tensor_dtypes": tensor_dtypes,
                "data_id": did,
                "seq_len": seq_len,
            }

            packed_loss_mask = packed_loss_mask_map.get(did)
            if packed_loss_mask is not None:
                result["packed_loss_mask"] = packed_loss_mask

            input_ids_from_kv = kv_params.get("input_ids_list")
            if input_ids_from_kv is not None:
                result["input_ids_list"] = input_ids_from_kv
            else:
                result["input_ids_list"] = list(output.prompt_token_ids)

            results.append(result)

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

    @staticmethod
    def _resolve_media(items: list, fetch_fn) -> list:
        """Resolve URL strings to loaded objects via *fetch_fn*, drop None entries."""
        resolved = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, str):
                resolved.append(fetch_fn(item))
            else:
                resolved.append(item)
        return resolved

    @staticmethod
    def _to_vllm_multi_modal_data(mm_input: dict | None) -> dict | None:
        """Convert TorchSpec multimodal payload to vLLM ``multi_modal_data``.

        TorchSpec shape:  ``{"images": [...], "videos": [...]}``
        vLLM shape:       ``{"image": <PIL.Image|list>, "video": ...}``

        URL strings are resolved to PIL Images / video objects via
        ``vllm.multimodal.utils.fetch_image`` / ``fetch_video`` so that
        ``LLM.generate()`` receives the data types it expects.
        None entries (from incomplete media blocks) are filtered out.
        """
        if not mm_input:
            return None
        mm_data: dict = {}
        images = mm_input.get("images")
        if images:
            from vllm.multimodal.utils import fetch_image

            loaded = VllmEngine._resolve_media(images, fetch_image)
            if loaded:
                mm_data["image"] = loaded[0] if len(loaded) == 1 else loaded
        videos = mm_input.get("videos")
        if videos:
            try:
                from vllm.multimodal.utils import fetch_video

                loaded = VllmEngine._resolve_media(videos, fetch_video)
            except ImportError:
                loaded = [v for v in videos if v is not None]
            if loaded:
                mm_data["video"] = loaded[0] if len(loaded) == 1 else loaded
        return mm_data or None

    def _build_prompts(
        self,
        formatted_prompts: list[str] | None,
        input_ids_list: list[torch.Tensor] | None,
        multimodal_inputs: list[dict | None] | None,
        batch_size: int,
    ) -> list:
        """Assemble per-request vLLM prompt dicts, attaching multimodal data when present."""
        if multimodal_inputs is not None and len(multimodal_inputs) != batch_size:
            raise ValueError(
                f"multimodal_inputs length {len(multimodal_inputs)} "
                f"does not match batch size {batch_size}"
            )

        prompts: list = []
        for i in range(batch_size):
            if formatted_prompts is not None:
                prompt_dict: dict = {"prompt": formatted_prompts[i]}
            else:
                prompt_dict = {
                    "prompt_token_ids": self._normalize_input_ids(input_ids_list[i]).tolist()
                }

            if multimodal_inputs is not None:
                mm_data = self._to_vllm_multi_modal_data(multimodal_inputs[i])
                if mm_data is not None:
                    prompt_dict["multi_modal_data"] = mm_data

            prompts.append(prompt_dict)
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
            try:
                if hasattr(self._engine, "close"):
                    self._engine.close()
                elif hasattr(self._engine, "llm_engine"):
                    llm_engine = self._engine.llm_engine
                    if hasattr(llm_engine, "shutdown"):
                        llm_engine.shutdown()
            except Exception as e:
                logger.warning(f"VllmEngine rank {self.rank}: Error during engine shutdown: {e}")
            finally:
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
            "hidden_states": HIDDEN_STATES_STORAGE_DTYPE,
            "input_ids": torch.long,
            "last_hidden_states": HIDDEN_STATES_STORAGE_DTYPE,
        }

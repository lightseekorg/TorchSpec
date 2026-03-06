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

"""vLLM Worker Extension for Hidden States Capture.

This module provides a TorchSpec-style worker extension for vLLM that enables
reliable hidden states extraction during inference. It patches the model's
forward method in each worker process to capture intermediate layer activations
and store them directly to Mooncake to avoid RPC serialization issues.

Based on the vllm-speculators approach but integrated into TorchSpec's
architecture with Ray Actors and Mooncake storage.
"""

import logging
import os
import re
import types
from collections import defaultdict
from itertools import islice
from typing import Any, Dict, List, Optional

import torch
from vllm.distributed import get_pp_group, get_tp_group
from vllm.sequence import IntermediateTensors

logger = logging.getLogger(__name__)


def _sanitize_mooncake_key(key: str) -> str:
    """Sanitize a key for use with Mooncake store.

    Mooncake keys should only contain alphanumeric characters, hyphens, and underscores.
    This function replaces invalid characters with underscores.

    Args:
        key: The original key (e.g., vLLM req_id)

    Returns:
        A sanitized key safe for Mooncake operations
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    if sanitized and sanitized[0].isdigit():
        sanitized = "k" + sanitized
    return sanitized


def _patched_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: Optional[IntermediateTensors] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Any:
    """Patched forward pass that captures hidden states from specified layers.

    This function is dynamically bound to base_model instances via types.MethodType.
    It expects base_model to have an _extension attribute pointing to the
    VllmWorkerExtension instance.

    Args:
        input_ids: Input token IDs
        positions: Position IDs
        intermediate_tensors: For pipeline parallelism
        inputs_embeds: Pre-computed input embeddings (for multimodal)
        **kwargs: Additional arguments

    Returns:
        Hidden states or IntermediateTensors (for PP)
    """
    # Get extension reference
    extension = self._extension  # noqa: SLF001

    # Handle pipeline parallelism - first rank does embedding
    if get_pp_group().is_first_rank:
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        )
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Track auxiliary hidden states for capture
    aux_hidden_states: List[torch.Tensor] = []

    # Only capture on TP rank 0 to avoid duplicates
    should_capture = get_tp_group().rank_in_group == 0
    target_layers = extension._layer_ids if should_capture else frozenset()  # noqa: SLF001

    # Capture input_ids only on first call (prefill phase) to avoid including generated tokens
    if should_capture and get_pp_group().is_first_rank and extension._captured_input_ids is None:
        # input_ids shape: (batch_size, seq_len) or (seq_len,)
        if input_ids.dim() == 2:
            # Flatten batch dimension
            extension._captured_input_ids = input_ids.view(-1).clone()
        else:
            extension._captured_input_ids = input_ids.clone()

    # Process each layer
    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(
            hidden_states=hidden_states,
            positions=positions,
            residual=residual,
        )
        absolute_layer_idx = self.start_layer + idx

        # Capture intermediate layers (not the last) before normalization
        if absolute_layer_idx in target_layers:
            # Add residual before capturing (matching speculators pattern)
            captured = (
                (hidden_states + residual).clone()
                if residual is not None
                else hidden_states.clone()
            )
            aux_hidden_states.append(captured)

    # Handle pipeline parallelism - return intermediate tensors if not last rank
    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    # Final normalization (only on last PP rank)
    hidden_states, _ = self.norm(hidden_states, residual)

    # Store captured states (only on last PP rank and TP rank 0)
    if should_capture and aux_hidden_states:
        extension._store_captured_states(aux_hidden_states)  # noqa: SLF001

    return hidden_states


class VllmWorkerExtension:
    """Worker extension that adds hidden states capture functionality to vLLM.

    This extension hooks into vLLM's Worker by being specified in the worker
    initialization. It patches the model's forward pass to intercept and capture
    intermediate layer hidden states during inference.

    Key behaviors:
    - Only captures on tensor parallel (TP) rank 0 to avoid duplicate data when
      using tensor parallelism. All TP ranks compute the same hidden states, so
      capturing from rank 0 is sufficient.
    - Stores captured states in GPU memory during batch processing, then writes
      directly to Mooncake to avoid RPC serialization issues.
    - Supports pipeline parallelism by handling IntermediateTensors correctly.
    - Tracks request metadata to map captured states back to original requests
      across chunked prefill iterations.

    Attributes:
        _layer_ids: Frozenset of layer indices for O(1) lookup during capture
        _captured_states: Accumulated hidden states per layer (GPU tensors)
        _request_metadata: Metadata tracking tokens per request per iteration
        _mooncake_store: EagleMooncakeStore instance for direct storage
        model_runner: Reference to the vLLM model runner
    """

    def __init__(self):
        """Initialize the worker extension with Mooncake store support."""
        self._layer_ids: frozenset = frozenset()
        self._captured_states: Optional[List[List[torch.Tensor]]] = None
        self._request_metadata: List[Dict[str, int]] = []
        self._current_request_metadata: Optional[Dict[str, int]] = None
        self._mooncake_store: Optional[Any] = None
        self._store_initialized: bool = False
        self._store_setup_complete: bool = False
        self._init_retry_count: int = 0
        self._max_init_retries: int = 3
        self.model_runner: Optional[Any] = None

    def _get_cuda_device_safe(self) -> torch.device:
        """Safely get CUDA device, handling uninitialized context (V1 compatibility).

        In vLLM V1, CUDA context may not be initialized when this method is called.
        This method safely handles both initialized and uninitialized contexts.

        Returns:
            torch.device: The CUDA device to use. Falls back to cuda:0 if context
                         is not yet initialized (common in V1 engine).
        """
        try:
            if torch.cuda.is_initialized():
                current_device = torch.cuda.current_device()
                logger.debug(f"CUDA initialized, using device cuda:{current_device}")
                return torch.device(f"cuda:{current_device}")
            else:
                # CUDA not initialized yet (V1), use device 0 as fallback
                # V1 will initialize context during model loading
                logger.debug("CUDA not initialized yet (V1), falling back to cuda:0")
                return torch.device("cuda:0")
        except RuntimeError as e:
            # CUDA context not available
            logger.warning(f"Failed to get CUDA device: {e}, falling back to cuda:0")
            return torch.device("cuda:0")

    def _init_mooncake_store(self) -> bool:
        """Initialize Mooncake store connection in the worker.

        Uses environment variables set by the main process to connect to
        the Mooncake master and metadata servers.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._store_initialized:
            return True

        # Only initialize on TP rank 0 - other ranks don't capture hidden states
        try:
            if get_tp_group().rank_in_group != 0:
                logger.debug("Skipping Mooncake store init on non-zero TP rank")
                return False
        except Exception:
            # If we can't get TP group info, proceed anyway (for backward compatibility)
            pass

        try:
            # Import here to avoid circular dependencies
            from torchspec.config.mooncake_config import MooncakeConfig
            from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

            # Get connection info from environment (set by main process)
            # Try TORCHSPEC_MOONCAKE_* first, then fall back to MOONCAKE_*
            master_addr = os.environ.get("TORCHSPEC_MOONCAKE_MASTER_ADDR") or os.environ.get(
                "MOONCAKE_MASTER_SERVER"
            )
            metadata_server = os.environ.get(
                "TORCHSPEC_MOONCAKE_METADATA_SERVER"
            ) or os.environ.get("MOONCAKE_METADATA_SERVER")
            local_hostname = os.environ.get("TORCHSPEC_MOONCAKE_LOCAL_HOSTNAME") or os.environ.get(
                "MOONCAKE_LOCAL_HOSTNAME", "localhost"
            )
            protocol = os.environ.get("TORCHSPEC_MOONCAKE_PROTOCOL") or os.environ.get(
                "MOONCAKE_PROTOCOL", "tcp"
            )

            if not master_addr:
                logger.warning(
                    "Mooncake master address not available in worker environment. "
                    "Set TORCHSPEC_MOONCAKE_MASTER_ADDR or MOONCAKE_MASTER_SERVER environment variable."
                )
                return False

            # Parse metadata_server to get port if not explicitly set
            if metadata_server:
                # Extract port from URL like "http://host:port/metadata"
                try:
                    metadata_port = metadata_server.split(":")[-1].replace("/metadata", "")
                except Exception:
                    metadata_port = "8090"
            else:
                metadata_port = "8090"
                metadata_server = f"http://{master_addr.split(':')[0]}:{metadata_port}/metadata"

            # Read buffer sizes from environment (set by main process via export_env)
            host_buffer_size_env = os.environ.get(
                "TORCHSPEC_MOONCAKE_HOST_BUFFER_SIZE"
            ) or os.environ.get("MOONCAKE_HOST_BUFFER_SIZE")

            # Build config kwargs with optional overrides from environment
            config_kwargs = {
                "master_server_address": master_addr,
                "metadata_server": metadata_server,
                "local_hostname": local_hostname,
                "protocol": protocol,
                "device_name": os.environ.get("TORCHSPEC_MOONCAKE_DEVICE_NAME")
                or os.environ.get("MOONCAKE_DEVICE_NAME", ""),
                "async_put_pool_size": int(
                    os.environ.get("TORCHSPEC_MOONCAKE_ASYNC_POOL_SIZE")
                    or os.environ.get("MOONCAKE_ASYNC_PUT_POOL_SIZE", "2")
                ),
                "enable_gpu_direct": (
                    os.environ.get("TORCHSPEC_MOONCAKE_GPU_DIRECT")
                    or os.environ.get("MOONCAKE_ENABLE_GPU_DIRECT", "0")
                ).lower()
                in ("true", "1", "yes"),
            }

            # Only override defaults if environment variables are set
            if host_buffer_size_env:
                config_kwargs["host_buffer_size"] = int(host_buffer_size_env)

            global_segment_size_env = os.environ.get(
                "TORCHSPEC_MOONCAKE_GLOBAL_SEGMENT_SIZE"
            ) or os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE")
            if global_segment_size_env:
                config_kwargs["global_segment_size"] = int(global_segment_size_env)

            local_buffer_size_env = os.environ.get(
                "TORCHSPEC_MOONCAKE_LOCAL_BUFFER_SIZE"
            ) or os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE")
            if local_buffer_size_env:
                config_kwargs["local_buffer_size"] = int(local_buffer_size_env)

            # Create config for worker
            config = MooncakeConfig(**config_kwargs)

            # Create store object but don't call setup() yet
            # setup() will be called lazily when CUDA context is ready
            self._mooncake_store = EagleMooncakeStore(config)
            # Mark as initialized but not yet setup
            # setup() will be called on first put() when CUDA context is ready
            self._store_initialized = True
            self._store_setup_complete = False

            logger.info(
                f"Worker initialized Mooncake store (setup deferred): master={master_addr}, protocol={protocol}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Mooncake store in worker: {e}", exc_info=True)
            self._mooncake_store = None
            self._store_initialized = False
            return False

    def _ensure_mooncake_store(self) -> bool:
        """Ensure Mooncake store is initialized and setup, with retry logic.

        This method handles lazy initialization for vLLM V1 compatibility.
        In V1, CUDA context may not be ready during initial Worker initialization,
        so we defer the actual setup() call until first use.

        Returns:
            True if store is ready for use, False otherwise.
        """
        # Ensure attributes exist (for vLLM V1 compatibility where __init__ may not be called)
        if not hasattr(self, "_store_initialized"):
            self._store_initialized = False
        if not hasattr(self, "_store_setup_complete"):
            self._store_setup_complete = False
        if not hasattr(self, "_init_retry_count"):
            self._init_retry_count = 0
        if not hasattr(self, "_max_init_retries"):
            self._max_init_retries = 3
        if not hasattr(self, "_mooncake_store"):
            self._mooncake_store = None

        # Already fully initialized and setup
        if self._store_initialized and self._store_setup_complete:
            return True

        # Check retry limit
        if self._init_retry_count >= self._max_init_retries:
            logger.error(
                f"Max retries ({self._max_init_retries}) exceeded for Mooncake store initialization"
            )
            return False

        try:
            # Initialize store if not already done
            if not self._store_initialized:
                if not self._init_mooncake_store():
                    self._init_retry_count += 1
                    logger.warning(
                        f"Mooncake store init failed (attempt {self._init_retry_count}/{self._max_init_retries})"
                    )
                    return False

            # Setup store if not already done
            if not self._store_setup_complete and self._mooncake_store is not None:
                try:
                    # Use safe CUDA device detection for V1 compatibility
                    device = self._get_cuda_device_safe()
                    logger.info(f"Setting up Mooncake store on device {device}")
                    self._mooncake_store.setup(device=device)

                    try:
                        logger.info("Warming up Mooncake RDMA path...")
                        self._mooncake_store.warmup_rdma()
                        logger.info("Mooncake RDMA warmup completed successfully")
                    except Exception as warmup_error:
                        logger.warning(f"Mooncake RDMA warmup failed: {warmup_error}")

                    self._store_setup_complete = True
                    logger.info("Mooncake store setup completed successfully")
                    return True
                except Exception as e:
                    self._init_retry_count += 1
                    # Check if this is a CUDA context error (common in V1)
                    error_msg = str(e).lower()
                    if "cuda" in error_msg or "device" in error_msg:
                        logger.warning(
                            f"CUDA context not ready (attempt {self._init_retry_count}/{self._max_init_retries}): {e}. "
                            f"Will retry on next put."
                        )
                    else:
                        logger.error(f"Mooncake store setup failed: {e}")
                    return False

            return True

        except Exception as e:
            self._init_retry_count += 1
            logger.error(f"Unexpected error in _ensure_mooncake_store: {e}", exc_info=True)
            return False

    def _store_captured_states(self, aux_hidden_states: List[torch.Tensor]) -> None:
        """Store captured hidden states from a forward pass.

        Args:
            aux_hidden_states: List of tensors, one per target layer
        """
        if getattr(self, "_captured_states", None) is None:
            # First capture - initialize lists for each layer
            self._captured_states = [[h] for h in aux_hidden_states]
        else:
            # Append to existing lists
            for i, h in enumerate(aux_hidden_states):
                self._captured_states[i].append(h)

        # Track how many tokens were captured in this step
        # Get from input_batch if available, otherwise use metadata
        model_runner = getattr(self, "model_runner", None)
        input_batch = getattr(model_runner, "input_batch", None)
        if input_batch is not None and hasattr(input_batch, "req_ids"):
            # Track by internal request IDs - will map to external IDs later
            step_tokens = {}
            for req_id in input_batch.req_ids:
                num_tokens = 0
                req_idx = getattr(input_batch, "req_id_to_index", {}).get(req_id)
                if req_idx is not None:
                    num_computed = getattr(
                        input_batch, "num_computed_tokens", [0] * len(input_batch.req_ids)
                    )[req_idx]
                    num_total = getattr(input_batch, "num_tokens", [0] * len(input_batch.req_ids))[
                        req_idx
                    ]
                    num_tokens = num_total - num_computed
                step_tokens[req_id] = num_tokens
            self._request_metadata.append(step_tokens)
        else:
            # Fallback: assume all requests in one step
            self._request_metadata.append({})

    def _store_input_ids(self, input_ids: torch.Tensor) -> None:
        """Store input_ids from a forward pass.

        Args:
            input_ids: Input token IDs tensor (batch_size, seq_len) or (seq_len,)
        """
        # Flatten if needed and store
        if input_ids.dim() == 2:
            # (batch_size, seq_len) - flatten to (batch_size * seq_len,)
            input_ids = input_ids.view(-1)
        if getattr(self, "_captured_input_ids", None) is None:
            self._captured_input_ids = input_ids.clone()
        else:
            self._captured_input_ids = torch.cat([self._captured_input_ids, input_ids], dim=0)

    def _setup_hidden_states_capture(self, layer_ids: List[int]) -> None:
        """Setup model to capture auxiliary hidden states from specific layers.

        This method patches the model's forward method to intercept hidden states
        during the forward pass.

        Args:
            layer_ids: List of layer indices to capture from
        """
        self._layer_ids = frozenset(layer_ids)
        self._captured_states = None
        self._request_metadata = []
        self._current_request_metadata = None
        self._packed_loss_mask_map: Dict[str, str] = {}
        self._store_initialized = False
        self._store_setup_complete = False
        self._init_retry_count = 0
        self._mooncake_store = None

        model_runner = getattr(self, "model_runner", None)
        if model_runner is None and hasattr(self, "model"):
            model_runner = self
        if model_runner is None:
            raise AttributeError("Could not find model_runner for worker extension setup")

        self.model_runner = model_runner
        model = self.model_runner.model  # type: ignore[attr-defined]

        # Handle vision-language models (e.g., Qwen-VL)
        if hasattr(model, "get_language_model"):
            base_model = model.get_language_model().model
        # Handle standard text models
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            base_model = model.model
        else:
            # Try to find model with layers attribute
            attrs = [a for a in dir(model) if not a.startswith("_")]
            raise AttributeError(
                f"Could not find base model with 'layers' attribute. "
                f"Model type: {type(model).__name__}, "
                f"Available attributes: {attrs}"
            )

        # Attach extension reference and patch forward method
        base_model._extension = self  # noqa: SLF001
        base_model.forward = types.MethodType(_patched_forward, base_model)

        logger.info(f"Hidden states capture setup complete for layers {layer_ids}")

    def _set_request_metadata(
        self,
        request_metadata: Dict[str, int],
        packed_loss_mask_map: Optional[Dict[str, str]] = None,
        input_ids_map: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """Set request metadata for the next forward pass.

        This is called before each scheduler iteration to track which tokens
        belong to which request.

        Args:
            request_metadata: Dict mapping request_id -> num_prefill_tokens
            packed_loss_mask_map: Optional dict mapping request_id -> packed_loss_mask string
            input_ids_map: Optional dict mapping request_id -> input_ids list (passed via RPC)
        """
        self._current_request_metadata = request_metadata
        self._packed_loss_mask_map = packed_loss_mask_map or {}
        self._input_ids_map = input_ids_map or {}

    def _reset_capture(self) -> None:
        """Reset captured states before starting a new batch.

        Must be called before processing a new batch of requests.
        """
        if not hasattr(self, "_layer_ids") or len(self._layer_ids) == 0:
            raise RuntimeError("Must call _setup_hidden_states_capture before capturing states")
        self._captured_states = None
        self._captured_input_ids: Optional[torch.Tensor] = None
        self._request_metadata = []
        self._current_request_metadata = None
        self._packed_loss_mask_map = {}
        self._input_ids_map = {}

    def _store_and_get_metadata(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Store captured hidden states to Mooncake and return metadata.

        This method stores tensors directly to Mooncake from the worker process,
        avoiding RPC serialization issues. It returns only lightweight metadata
        that can be safely serialized and returned via collective_rpc.

        Returns:
            Dict mapping request_id to metadata dict with keys:
                - 'mooncake_key': str, the base key used for storage
                - 'tensor_shapes': dict of tensor shapes
                - 'tensor_dtypes': dict of dtype names
                - 'num_layers': int, number of captured layers
            or None if no states captured or not on TP rank 0.
        """
        # Only TP rank 0 has captured data
        if get_tp_group().rank_in_group != 0:
            return None
        if self._captured_states is None:
            return None

        # Ensure Mooncake store is initialized and setup (with retry for V1 compatibility)
        if not self._ensure_mooncake_store():
            logger.error(
                "Failed to initialize/setup Mooncake store, cannot store hidden states. "
                "This may be due to CUDA context not being ready in V1 engine."
            )
            return None

        # Concatenate captured states from all scheduler iterations
        concatenated_layers = [
            torch.cat(layer_tensors, dim=0) for layer_tensors in self._captured_states
        ]
        total_captured_tokens = concatenated_layers[0].shape[0]

        # Slice and group by request using external IDs
        external_ids = (
            list(self._current_request_metadata.keys()) if self._current_request_metadata else []
        )
        token_counts = (
            list(self._current_request_metadata.values()) if self._current_request_metadata else []
        )
        total_expected_tokens = sum(token_counts) if token_counts else 0

        request_chunks: defaultdict[str, List[List[torch.Tensor]]] = defaultdict(
            lambda: [[] for _ in range(len(concatenated_layers))]
        )
        current_idx = 0

        # Handle multi-step scheduling:按比例分配实际捕获的tokens
        if total_expected_tokens > 0 and total_captured_tokens > 0:
            ratio = total_captured_tokens / total_expected_tokens
            for req_idx, (external_id, expected_tokens) in enumerate(
                zip(external_ids, token_counts)
            ):
                # 按比例分配实际捕获的token数
                actual_tokens = int(expected_tokens * ratio)
            for req_idx, (external_id, expected_tokens) in enumerate(
                zip(external_ids, token_counts)
            ):
                # 按比例分配实际捕获的token数
                actual_tokens = int(expected_tokens * ratio)
                actual_tokens = min(actual_tokens, total_captured_tokens - current_idx)
                if actual_tokens > 0:
                    for layer_idx, layer_tensor in enumerate(concatenated_layers):
                        chunk = (
                            layer_tensor[current_idx : current_idx + actual_tokens].clone().cpu()
                        )
                        request_chunks[external_id][layer_idx].append(chunk)
                    current_idx += actual_tokens
        else:
            # Fallback: simple sequential slicing
            for req_idx, (external_id, num_tokens) in enumerate(zip(external_ids, token_counts)):
                if current_idx < total_captured_tokens:
                    for layer_idx, layer_tensor in enumerate(concatenated_layers):
                        chunk = layer_tensor[current_idx : current_idx + num_tokens].clone().cpu()
                        request_chunks[external_id][layer_idx].append(chunk)
                    current_idx += num_tokens

        # Store to Mooncake and collect metadata
        result: Dict[str, Dict[str, Any]] = {}
        for req_id, layer_chunks in request_chunks.items():
            mooncake_key = _sanitize_mooncake_key(req_id)
            if mooncake_key != req_id:
                logger.debug(f"Sanitized key '{req_id}' -> '{mooncake_key}'")

            # Concatenate all layer chunks for this request (keep on GPU)
            layer_tensors = [torch.cat(chunks, dim=0) for chunks in layer_chunks]

            # Concatenate all layers along hidden dimension
            if len(layer_tensors) > 1:
                hidden_states = torch.cat(layer_tensors, dim=-1)
            else:
                hidden_states = layer_tensors[0]

            last_hidden_states = layer_tensors[-1]

            # Use real input_ids from RPC, otherwise create dummy
            if req_id in self._input_ids_map:
                input_ids_list = self._input_ids_map[req_id]
                input_ids = torch.tensor(
                    input_ids_list, dtype=torch.long, device=hidden_states.device
                )
            else:
                seq_len = hidden_states.shape[0]
                input_ids = torch.zeros(seq_len, dtype=torch.long, device=hidden_states.device)

            # Skip empty tensors
            if hidden_states.numel() == 0:
                logger.error(f"Request {req_id}: hidden_states is empty! Skipping.")
                continue

            try:
                logger.debug(
                    f"Storing to Mooncake: key={mooncake_key}, "
                    f"hidden_states_shape={hidden_states.shape}"
                )

                # Store to Mooncake
                tensor_shapes = self._mooncake_store.put(
                    key=mooncake_key,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    last_hidden_states=last_hidden_states,
                    target=None,
                )

                logger.debug(f"Successfully stored to Mooncake: key={mooncake_key}")

                # Convert dtype to string for RPC serialization
                # Include input_ids as a list for reconstruction (avoids Mooncake storage issues)
                result[req_id] = {
                    "mooncake_key": mooncake_key,
                    "tensor_shapes": tensor_shapes,
                    "tensor_dtypes": {
                        "hidden_states": str(hidden_states.dtype).replace("torch.", ""),
                        "input_ids": str(input_ids.dtype).replace("torch.", ""),
                        "last_hidden_states": str(last_hidden_states.dtype).replace("torch.", ""),
                    },
                    "num_layers": len(layer_tensors),
                    "packed_loss_mask": self._packed_loss_mask_map.get(req_id),
                    "input_ids_list": input_ids.cpu().tolist(),  # Serialize via RPC instead of Mooncake
                }
            except Exception as e:
                logger.error(
                    f"Failed to store tensors to Mooncake for {req_id} (key={mooncake_key}): {e}"
                )
                # Continue with other requests even if one fails
                continue

        # Flush to ensure all writes are complete before returning
        if self._mooncake_store is not None:
            self._mooncake_store.flush()

        # Clear intermediate storage to free memory
        self._captured_states = None
        self._captured_input_ids = None
        self._request_metadata = []
        self._input_ids_map = {}

        return result if result else None

    def _get_captured_states(self) -> Optional[Dict[str, List[torch.Tensor]]]:
        """Legacy method - now delegates to _store_and_get_metadata.

        This method is kept for backward compatibility but should not be used
        in production due to RPC serialization issues. Use _store_and_get_metadata
        instead which stores tensors directly to Mooncake.

        Returns:
            Dict mapping request_id to list of tensors (one per layer),
            or None if no states captured.
        """
        # If Mooncake store is available, use the new method
        if self._store_initialized or self._init_mooncake_store():
            metadata = self._store_and_get_metadata()
            if metadata is None:
                return None
            # Return empty dict to signal success - actual data is in Mooncake
            return {}

        # Fallback to old behavior if Mooncake not available
        if self._captured_states is None:
            return None

        # Concatenate captured states from all scheduler iterations
        concatenated_layers = [
            torch.cat(layer_tensors, dim=0) for layer_tensors in self._captured_states
        ]

        # Slice and group by request
        request_chunks: defaultdict[str, List[List[torch.Tensor]]] = defaultdict(
            lambda: [[] for _ in range(len(concatenated_layers))]
        )
        current_idx = 0

        # Use external IDs for slicing
        external_ids = (
            list(self._current_request_metadata.keys()) if self._current_request_metadata else []
        )
        token_counts = (
            list(self._current_request_metadata.values()) if self._current_request_metadata else []
        )

        req_idx = 0
        for step_metadata in self._request_metadata:
            step_tokens = sum(step_metadata.values()) if step_metadata else 0
            if step_tokens == 0 and req_idx < len(token_counts):
                step_tokens = token_counts[req_idx]

            if req_idx < len(external_ids):
                external_id = external_ids[req_idx]
                for layer_idx, layer_tensor in enumerate(concatenated_layers):
                    chunk = layer_tensor[current_idx : current_idx + step_tokens].clone().cpu()
                    request_chunks[external_id][layer_idx].append(chunk)
                current_idx += step_tokens
                req_idx += 1

        # Concatenate chunks for each request across iterations
        result: Dict[str, List[torch.Tensor]] = {
            req_id: [torch.cat(chunks, dim=0) for chunks in layer_chunks]
            for req_id, layer_chunks in request_chunks.items()
        }

        # Clear intermediate storage to free memory
        self._captured_states = None
        self._request_metadata = []

        return result

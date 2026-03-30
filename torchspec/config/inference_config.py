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

"""Inference configuration classes.

SGLangConfig exposes only the fields that TorchSpec actually references.
Power users can pass arbitrary extra kwargs to sgl.Engine via ``extra_args``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torchspec.config.mooncake_config import MooncakeConfig


@dataclass
class SGLangConfig:
    """Essential SGLang engine configuration.

    Only fields that TorchSpec explicitly uses are listed here.
    Any additional sgl.Engine kwargs can be supplied via ``extra_args``
    and will be forwarded as-is.
    """

    # Parallelism
    tp_size: int = 8
    pp_size: int = 1
    nnodes: int = 1

    # Memory
    mem_fraction_static: float = 0.8

    # Observability (read by TorchSpec's wandb integration)
    enable_metrics: bool = False

    # Multimodal
    enable_multimodal: bool = False

    # Networking (port is auto-selected by SglEngine via get_free_port)
    dist_init_addr: Optional[str] = None
    dist_timeout: int = 60
    init_timeout: int = 300

    # Passthrough: forwarded as-is to sgl.Engine.
    # Use this for any sgl.Engine kwarg that TorchSpec doesn't need to
    # inspect (e.g. quantization, context_length, attention_backend,
    # log_level, ...).
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VllmConfig:
    """Essential vLLM engine configuration.

    Only fields that TorchSpec explicitly uses are listed here.
    Any additional vLLM engine kwargs can be supplied via ``extra_args``
    and will be forwarded as-is.

    Uses vLLM's ``extract_hidden_states`` speculative method with a
    ``MooncakeHiddenStatesConnector`` KV Connector for hidden states
    retrieval (requires vLLM >= 0.18.0).
    """

    # Parallelism
    tp_size: int = 8
    pp_size: int = 1
    nnodes: int = 1

    # Memory
    mem_fraction_static: float = 0.8

    # Observability
    enable_metrics: bool = False

    # Multimodal
    enable_multimodal: bool = False

    # Networking (port is auto-selected by VllmEngine)
    dist_init_addr: Optional[str] = None
    dist_timeout: int = 60
    init_timeout: int = 300

    # Passthrough: forwarded as-is to vLLM LLM.
    # Use this for any vLLM kwarg that TorchSpec doesn't need to
    # inspect (e.g. quantization, max_model_len, trust_remote_code, ...).
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    aux_hidden_states_layers: Optional[list] = None
    inference_batch_size: int = 1
    inference_buffer_threshold: int = 32
    inference_engine_type: str = "hf"
    inference_fetch_batch: int = 1
    inference_num_gpus: Optional[int] = None
    inference_num_gpus_per_engine: int = 1
    inference_num_gpus_per_node: int = 8
    last_hidden_states_prenorm: Optional[bool] = None
    max_sample_pool_size: int = 0
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)

    def resolve_last_hidden_states_prenorm(self) -> bool:
        """Whether last_hidden_states from the engine are pre-norm.

        vLLM's extract_hidden_states connector can only capture raw layer
        outputs (pre-norm), while sglang and hf provide post-norm outputs.
        """
        if self.last_hidden_states_prenorm is not None:
            return self.last_hidden_states_prenorm
        return self.inference_engine_type == "vllm"


@dataclass
class HFInferenceConfig:
    """Configuration for HFRunner."""

    model_path: str
    max_seq_length: int = 8192
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    aux_hidden_states_layers: Optional[list[int]] = None
    mooncake_config: Optional[MooncakeConfig] = None

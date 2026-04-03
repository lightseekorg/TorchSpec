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

import json
import os
from typing import Union

from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import LlamaConfig, PretrainedConfig, modeling_utils
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from torchspec.models.draft.deepseek_eagle import DeepSeekForCausalLMEagle3
from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from torchspec.utils.logging import logger


class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
        DeepseekV3Config: DeepSeekForCausalLMEagle3,
    }

    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        _model_cls = cls._model_mapping[type(config)]
        model = _model_cls(config, **config_kwargs)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ):
        original_warn = modeling_utils.logger.warning

        def filtered_warning(msg):
            if "embed_tokens.weight" in str(msg) and "initialized" in str(msg):
                return
            original_warn(msg)

        modeling_utils.logger.warning = filtered_warning

        try:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        finally:
            modeling_utils.logger.warning = original_warn

        return model


class AutoDraftModelConfig:
    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
        "DeepSeekForCausalLMEagle3": DeepseekV3Config,
        "Eagle3DeepseekV2ForCausalLM": DeepseekV3Config,
    }

    @classmethod
    def from_dict(cls, config: dict):
        config = dict(config)

        if "tie_word_embeddings" in config:
            logger.info("Set draft model tie_word_embeddings to False")
            config["tie_word_embeddings"] = False

        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        if "draft_vocab_size" not in config or config["draft_vocab_size"] is None:
            config["draft_vocab_size"] = config.get("vocab_size", None)

        return cls._config_mapping[architecture].from_dict(config)

    @classmethod
    def from_file(cls, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls.from_dict(config)

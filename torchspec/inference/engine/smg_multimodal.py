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

"""Narrow adapter from SMG Qwen3-VL pixels to vLLM's prepared input contract."""

import numpy as np
import torch


def prepared_vllm_input(ids, payload, model_config):
    """Validate a prepared payload and bypass vLLM's HF media processor entirely."""
    from transformers import BatchFeature
    from vllm.inputs import mm_input
    from vllm.multimodal.inputs import (
        MultiModalFieldConfig,
        MultiModalKwargsItems,
        PlaceholderRange,
    )

    if payload.get("format") != "torchspec-smg-qwen3-vl-v1":
        raise ValueError("Unknown prepared multimodal payload")
    if getattr(model_config.hf_config, "model_type", None) != "qwen3_vl":
        raise ValueError("Prepared SMG images require a Qwen3-VL engine")
    if model_config.hf_config.image_token_id != payload["image_token_id"]:
        raise ValueError("Prepared image token ID does not match the engine checkpoint")
    mm_config = model_config.multimodal_config
    if mm_config.mm_processor_kwargs or mm_config.media_io_kwargs:
        raise ValueError(
            "Prepared SMG images do not support engine processor/image-loader overrides"
        )
    grid = np.asarray(payload["image_grid_thw"])
    pixels = np.asarray(payload["pixel_values"])
    ranges, hashes = payload["image_ranges"], payload["image_hashes"]
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.dtype.kind not in "iu" or np.any(grid <= 0):
        raise ValueError("Invalid image_grid_thw")
    if len(grid) != len(ranges) or len(grid) != len(hashes):
        raise ValueError("Image metadata counts disagree")
    sizes = grid.prod(axis=1)
    if (
        pixels.ndim != 2
        or pixels.shape != (int(sizes.sum()), 1536)
        or not np.isfinite(pixels).all()
    ):
        raise ValueError("Invalid Qwen3-VL patch tensor")
    end = 0
    for (start, length), size in zip(ranges, sizes):
        if start < end or length != size // 4 or size % 4 or start + length > len(ids):
            raise ValueError("Image ranges do not match the grids or final IDs")
        if any(token != payload["image_token_id"] for token in ids[start : start + length]):
            raise ValueError("Image range does not contain the prepared image token IDs")
        end = start + length
    if sum(token == payload["image_token_id"] for token in ids) != sum(n for _, n in ranges):
        raise ValueError("Image tokens exist outside the declared ranges")
    fields = {
        "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", torch.tensor(sizes)),
        "image_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
    }
    values = BatchFeature(
        {
            "pixel_values": torch.tensor(pixels, dtype=model_config.dtype),
            "image_grid_thw": torch.tensor(grid, dtype=torch.long),
        }
    )
    return mm_input(
        prompt_token_ids=ids,
        mm_kwargs=MultiModalKwargsItems.from_hf_inputs(values, fields),
        mm_hashes={"image": hashes},
        mm_placeholders={
            "image": [PlaceholderRange(offset=start, length=n) for start, n in ranges]
        },
    )


def generate_prepared(llm, prompts, sampling_params):
    """Submit already-processed inputs without invoking LLM.generate's renderer.

    This adapter uses the vLLM 0.26 prepared-input submission API. Mixed batches
    retain ordinary vLLM rendering for unprepared samples. Unsupported versions
    fail explicitly instead of silently reprocessing the SMG inputs.
    """
    from vllm.outputs import RequestOutput

    if not hasattr(llm, "_render_and_run_requests") or not hasattr(llm, "_preprocess_cmpl"):
        raise RuntimeError(
            "SMG image inputs require vLLM's prepared EngineInput API (qualified on 0.26)"
        )
    inputs = (
        prompt if prompt.get("type") == "multimodal" else llm._preprocess_cmpl([prompt])[0]
        for prompt in prompts
    )
    return llm._render_and_run_requests(
        inputs, [sampling_params.clone() for _ in prompts], RequestOutput, use_tqdm=False
    )

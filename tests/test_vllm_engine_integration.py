"""Integration test for vLLM extract_hidden_states with Qwen2.5-VL.

Uses a single VLM engine for all tests:
  1. Text-only via prompt_token_ids
  2. Text-only formatted_prompts (defer tokenization)
  3. Multimodal samples from sample_kimi_k25_conversations.jsonl
     with real image fetching, multi_modal_data, vision token expansion,
     and loss mask verification

Requires: 2+ GPUs, vLLM >= 0.18 (with PR #38987 fix for VLM configs).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python tests/test_vllm_engine_integration.py
"""

import json
import os
import re
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoProcessor, AutoTokenizer


def main():
    from vllm import LLM, SamplingParams

    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    tp_size = 2
    aux_layer_ids = [2, 4, 6]
    hidden_size = 3584

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"{'=' * 60}")
    print("vLLM extract_hidden_states Integration Test (VLM)")
    print(f"{'=' * 60}")
    print(f"Model:      {model_path}")
    print(f"TP size:    {tp_size}")
    print(f"Aux layers: {aux_layer_ids}")
    print(f"Hidden:     {hidden_size}")
    print(f"{'=' * 60}", flush=True)

    print("\n[init] Creating LLM with extract_hidden_states...", flush=True)
    engine = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        disable_custom_all_reduce=True,
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 4},
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": aux_layer_ids,
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_connector_module_path": (
                "torchspec.inference.engine.mooncake_hidden_states_connector"
            ),
            "kv_role": "kv_producer",
        },
    )
    print("      Engine created.", flush=True)

    sampling_params = SamplingParams(max_tokens=1, temperature=0)

    # =========================================================================
    # Test 1: Text-only via prompt_token_ids
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Text-only via prompt_token_ids")
    print("=" * 60)

    prompts = [
        {"prompt_token_ids": [1, 2345, 6789, 101, 202]},
        {"prompt_token_ids": [100, 200, 300, 400]},
    ]
    outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
    assert len(outputs) == 2

    for i, output in enumerate(outputs):
        kv = getattr(output, "kv_transfer_params", None)
        print(
            f"  Output {i}: {len(output.prompt_token_ids)} tokens, "
            f"kv_params={'present' if kv else 'None'}"
        )
        if kv:
            print(f"    mooncake_key={kv.get('mooncake_key')}")
            print(f"    input_ids_list[:5]={kv.get('input_ids_list', [])[:5]}")
    print("✓ Test 1 passed")

    # =========================================================================
    # Test 2: Text-only formatted_prompts (defer tokenization)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Text-only formatted_prompts (defer tokenization)")
    print("=" * 60)

    text_prompts = ["Hello, world!", "The quick brown fox jumps over the lazy dog."]
    outputs = engine.generate(text_prompts, sampling_params, use_tqdm=False)
    assert len(outputs) == 2

    for i, output in enumerate(outputs):
        kv = getattr(output, "kv_transfer_params", None)
        token_ids = list(output.prompt_token_ids)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  Output {i}: {len(token_ids)} tokens, decoded='{decoded[:60]}...'")
        if kv and kv.get("input_ids_list"):
            assert kv["input_ids_list"] == token_ids, (
                "input_ids_list from kv_params doesn't match prompt_token_ids"
            )
            print("    input_ids_list matches prompt_token_ids ✓")
    print("✓ Test 2 passed")

    # =========================================================================
    # Test 3: Multimodal samples with real images
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Multimodal (real images + defer tokenization + loss mask)")
    print("=" * 60)

    from vllm.multimodal.utils import fetch_image

    from torchspec.data.template import TEMPLATE_REGISTRY
    from torchspec.data.utils import extract_media_urls
    from torchspec.models.ops.loss_mask import compute_assistant_loss_mask

    kimi_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "data",
        "sample_kimi_k25_conversations.jsonl",
    )
    samples = []
    with open(kimi_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples")

    # -- 3a: Build prompt dicts with multi_modal_data --
    mm_data_ids = []
    mm_prompts = []
    mm_multimodal_inputs = []

    for sample in samples:
        sid = sample["id"]
        messages = list(sample["conversations"])
        multimodal_inputs = extract_media_urls(messages)

        formatted = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        prompt_dict = {"prompt": formatted}
        if multimodal_inputs and multimodal_inputs.get("images"):
            image_urls = multimodal_inputs["images"]
            loaded_images = [fetch_image(url) for url in image_urls]
            prompt_dict["multi_modal_data"] = {
                "image": loaded_images[0] if len(loaded_images) == 1 else loaded_images
            }

        mm_data_ids.append(sid)
        mm_prompts.append(prompt_dict)
        mm_multimodal_inputs.append(multimodal_inputs)

        has_mm = "multi_modal_data" in prompt_dict
        print(f"  {sid}: multimodal={has_mm}")

    assert "multi_modal_data" not in mm_prompts[0], "text-only should have no mm"
    assert "multi_modal_data" in mm_prompts[3], "kimi_mm_001 should have mm"

    # -- 3b: Run through engine --
    print("\n  Running engine.generate()...", flush=True)
    outputs = engine.generate(mm_prompts, sampling_params, use_tqdm=False)
    assert len(outputs) == len(samples)

    sample_by_id = {s["id"]: s for s in samples}

    for i, output in enumerate(outputs):
        did = mm_data_ids[i]
        token_ids = list(output.prompt_token_ids)
        kv = getattr(output, "kv_transfer_params", None)
        has_mm = mm_multimodal_inputs[i] is not None
        print(
            f"  {did}: {len(token_ids)} tokens, multimodal={has_mm}, "
            f"kv_params={'present' if kv else 'None'}"
        )

    # -- 3c: Verify vision tokens in multimodal outputs --
    vision_start_text = "<|vision_start|>"
    print("\n  Checking for vision placeholders...")

    for i, (did, output) in enumerate(zip(mm_data_ids, outputs)):
        mm_input = mm_multimodal_inputs[i]
        if mm_input is None:
            continue
        num_images = len(mm_input.get("images") or [])
        token_ids = list(output.prompt_token_ids)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        count = decoded.count(vision_start_text)
        assert count == num_images, f"{did}: expected {num_images} vision block(s), found {count}"
        # Vision tokens expand the sequence beyond the text template
        text_template_len = len(tokenizer.encode(mm_prompts[i]["prompt"], add_special_tokens=False))
        assert len(token_ids) > text_template_len, (
            f"{did}: vision-expanded ({len(token_ids)}) should exceed "
            f"text template ({text_template_len})"
        )
        print(
            f"  {did}: {count} vision block(s), "
            f"{len(token_ids)} tokens (template={text_template_len}) ✓"
        )

    # -- 3d: Verify loss mask recovers last-turn assistant response --
    def _normalize(s: str) -> str:
        s = re.sub(r"(</?think>)", r" \1 ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _visible_content(s: str) -> str:
        return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

    qwen_template = TEMPLATE_REGISTRY.get("qwen2-vl")
    header_text = qwen_template.assistant_header.rstrip("\n")
    full_header_ids = tokenizer.encode(qwen_template.assistant_header, add_special_tokens=False)
    stripped_header_ids = tokenizer.encode(header_text, add_special_tokens=False)
    skip_after_header = len(full_header_ids) - len(stripped_header_ids)
    end_token_ids = tokenizer.encode(qwen_template.end_of_turn_token, add_special_tokens=False)
    print(
        f"\n  Loss mask: header={stripped_header_ids}, end={end_token_ids}, "
        f"skip_after={skip_after_header}"
    )

    for i, (did, output) in enumerate(zip(mm_data_ids, outputs)):
        token_ids_tensor = torch.tensor(list(output.prompt_token_ids))
        mask = compute_assistant_loss_mask(
            token_ids_tensor,
            stripped_header_ids,
            end_token_ids,
            last_turn_only=True,
            skip_after_header=skip_after_header,
        )
        masked_ids = token_ids_tensor[mask.bool()].tolist()
        recovered_text = tokenizer.decode(masked_ids, skip_special_tokens=False)

        conv = sample_by_id[did]["conversations"]
        last_assistant_content = None
        for msg in conv:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_assistant_content = content

        assert mask.sum().item() > 0, f"{did}: loss mask is all zeros"

        recovered_norm = _normalize(recovered_text)
        visible = _visible_content(last_assistant_content)
        assert visible, f"{did}: last assistant turn has no visible content"
        visible_norm = _normalize(visible)
        assert visible_norm in recovered_norm, (
            f"{did}: last assistant content not found in loss-masked region.\n"
            f"  Expected: {visible_norm[:100]}...\n"
            f"  Recovered: {recovered_norm[:200]}..."
        )

        all_assistant = [
            msg.get("content", "")
            for msg in conv
            if msg.get("role") == "assistant" and isinstance(msg.get("content", ""), str)
        ]
        if len(all_assistant) > 1:
            earlier_visible = _visible_content(all_assistant[0])
            if earlier_visible:
                earlier_norm = _normalize(earlier_visible)
                assert earlier_norm not in recovered_norm, (
                    f"{did}: earlier turn should NOT be in loss mask: {earlier_norm[:80]}"
                )
            print(f"  {did}: {mask.sum().item()} masked tokens, last-turn only ✓")
        else:
            print(f"  {did}: {mask.sum().item()} masked tokens, content matched ✓")

    # -- 3e: Sequence length checks --
    lens = {mm_data_ids[i]: len(outputs[i].prompt_token_ids) for i in range(len(outputs))}
    assert lens["kimi_mm_003"] > lens["kimi_mm_001"], (
        f"Two-image ({lens['kimi_mm_003']}) should exceed single-image ({lens['kimi_mm_001']})"
    )
    print(f"\n  kimi_mm_001={lens['kimi_mm_001']} < kimi_mm_003={lens['kimi_mm_003']} ✓")

    print("\n✓ Test 3 passed")

    # =========================================================================
    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print(f"{'=' * 60}")
    del engine
    sys.exit(0)


if __name__ == "__main__":
    main()

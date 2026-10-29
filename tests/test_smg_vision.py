"""Small regression cases for the opt-in prepared-image training path."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests.test_smg_renderer import checkpoint  # noqa: F401
from torchspec.data.dataset import load_conversation_dataset
from torchspec.data.renderers.smg import effective_renderer
from torchspec.data.renderers.smg_vision import SmgVisionRenderer
from torchspec.data.utils import unpack_loss_mask


@pytest.fixture
def vision_checkpoint(checkpoint):  # noqa: F811
    from torchspec_smg import build_features

    if "multimodal" not in build_features:
        pytest.skip("requires TORCHSPEC_SMG_FEATURES=multimodal")
    path, hf = checkpoint
    hf.add_special_tokens(
        {"additional_special_tokens": ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]}
    )
    hf.chat_template = (
        "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' }}"
        "{% if m.content is string %}{{ m.content }}{% else %}"
        "{% for p in m.content %}{% if p.type == 'image_url' %}"
        "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        "{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}"
        "{{ '<|im_end|>\\n' }}{% endfor %}"
    )
    hf.save_pretrained(path)
    (path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_vl", "image_token_id": hf.convert_tokens_to_ids("<|image_pad|>")}
        )
    )
    (path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "size": {"shortest_edge": 1024, "longest_edge": 65536},
                "image_mean": [0.5] * 3,
                "image_std": [0.5] * 3,
            }
        )
    )
    Image.new("RGB", (32, 64), "red").save(path / "image.png")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": str(path / "image.png")}},
                {"type": "text", "text": "Describe"},
            ],
        },
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "Again"},
        {"role": "assistant", "content": "second answer"},
    ]
    return path, hf, messages


@pytest.mark.parametrize("last_turn_only", [False, True])
def test_final_ids_masks_and_image_truncation(vision_checkpoint, last_turn_only):
    path, hf, messages = vision_checkpoint
    renderer = SmgVisionRenderer(path)
    with pytest.raises(ValueError, match="Use render_prepared"):
        renderer.render(messages, max_seq_length=4096)
    ids, mask, payload = renderer.render_prepared(
        messages, max_seq_length=4096, last_turn_only=last_turn_only
    )
    text = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    start, length = payload["image_ranges"][0]
    text = text.replace("<|image_pad|>", "<|image_pad|>" * length)
    reference = hf(text, add_special_tokens=False, return_offsets_mapping=True)
    assert ids == reference.input_ids
    bodies = ["second answer"] if last_turn_only else ["first answer", "second answer"]
    spans = [(text.index(body), text.index(body) + len(body + "<|im_end|>")) for body in bodies]
    expected = [
        int(a < b and any(lo <= a and b <= hi for lo, hi in spans))
        for a, b in reference.offset_mapping
    ]
    assert mask == expected
    assert not any(mask[start : start + length])
    with pytest.raises(ValueError, match="Truncation cuts through an image"):
        renderer.render_prepared(messages, max_seq_length=start + 1)
    cut_ids, cut_mask, dropped = renderer.render_prepared(messages, max_seq_length=start - 1)
    assert cut_ids == ids[: start - 1] and not any(cut_mask) and dropped is None
    cut_ids, cut_mask, kept = renderer.render_prepared(
        messages, max_seq_length=len(ids) - 2, last_turn_only=last_turn_only
    )
    assert cut_ids == ids[:-2] and cut_mask == mask[:-2]
    np.testing.assert_array_equal(kept["pixel_values"], payload["pixel_values"])


@pytest.mark.parametrize("num_proc", [1, 2])
def test_loader_preserves_prepared_mask_and_does_not_cache_mutable_media(
    vision_checkpoint, num_proc
):
    path, _, messages = vision_checkpoint
    source = path / "data.jsonl"
    source.write_text(json.dumps({"conversations": messages}) + "\n")
    args = SimpleNamespace(
        renderer="smg-qwen3-vl",
        target_model_path=str(path),
        train_data_path=str(source),
        prompt_key="conversations",
        max_seq_length=4096,
        num_proc=num_proc,
        cache_dir=str(path / "cache"),
        dynamic_loss_mask=True,
    )
    first = load_conversation_dataset(args)[0]
    assert len(unpack_loss_mask(first["packed_loss_mask"])) == len(first["input_ids"])
    assert first["formatted_prompt"] is None
    Image.new("RGB", (32, 64), "blue").save(path / "image.png")
    second = load_conversation_dataset(args)[0]
    assert first["multimodal_inputs"]["image_hashes"] != second["multimodal_inputs"]["image_hashes"]
    assert not list((path / "cache").rglob("*.pt"))


@pytest.mark.parametrize("bad", ["<|image_pad|>", "<|vision_start|>"])
def test_literal_vision_tokens_cannot_forge_images(vision_checkpoint, bad):
    path, _, messages = vision_checkpoint
    messages[-1]["content"] = bad
    with pytest.raises(ValueError, match="Literal vision delimiters"):
        SmgVisionRenderer(path).render_prepared(messages, max_seq_length=4096)


def test_vision_route_is_explicit_and_vllm_only():
    assert (
        effective_renderer(SimpleNamespace(target_model_path="Qwen/Qwen3-VL-2B-Instruct")) is None
    )
    assert effective_renderer(SimpleNamespace(renderer="smg-qwen3-vl")) == "smg-qwen3-vl"
    for settings in (
        {"inference_engine_type": "sglang"},
        {"defer_tokenization": True},
        {"train_with_decode": True},
        {"rendering_backend": "huggingface"},
    ):
        with pytest.raises(ValueError):
            effective_renderer(SimpleNamespace(renderer="smg-qwen3-vl", **settings))


def test_multiple_images_keep_order_and_drop_truncated_tensors(vision_checkpoint):
    import copy

    path, _, messages = vision_checkpoint
    messages[2]["content"] = copy.deepcopy(messages[0]["content"])
    renderer = SmgVisionRenderer(path)
    ids, mask, payload = renderer.render_prepared(messages, max_seq_length=4096)
    assert len(payload["image_ranges"]) == 2
    cut = payload["image_ranges"][1][0] - 1
    ci, cm, cp = renderer.render_prepared(messages, max_seq_length=cut)
    assert ci == ids[:cut] and cm == mask[:cut]
    assert len(cp["image_ranges"]) == 1 and cp["image_grid_thw"].shape == (1, 3)
    assert cp["pixel_values"].shape[0] == cp["image_grid_thw"].prod()


@pytest.mark.parametrize("fault", [None, "ids", "range", "pixels", "model", "override"])
def test_prepared_vllm_adapter_checks_alignment(vision_checkpoint, fault):
    pytest.importorskip("vllm")
    import torch

    from torchspec.inference.engine.smg_multimodal import prepared_vllm_input

    path, _, messages = vision_checkpoint
    ids, _, payload = SmgVisionRenderer(path).render_prepared(messages, max_seq_length=4096)
    cfg = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(model_type="qwen3_vl", image_token_id=payload["image_token_id"]),
        multimodal_config=SimpleNamespace(mm_processor_kwargs=None, media_io_kwargs=None),
    )
    if fault == "ids":
        ids[payload["image_ranges"][0][0]] = 0
    elif fault == "range":
        start, length = payload["image_ranges"][0]
        payload["image_ranges"][0] = (start + 1, length)
    elif fault == "pixels":
        payload["pixel_values"] = payload["pixel_values"][:-1]
    elif fault == "model":
        cfg.hf_config.model_type = "qwen3"
    elif fault == "override":
        cfg.multimodal_config.mm_processor_kwargs = {"max_pixels": 4096}
    if fault:
        with pytest.raises(ValueError):
            prepared_vllm_input(ids, payload, cfg)
    else:
        prepared = prepared_vllm_input(ids, payload, cfg)
        assert prepared["prompt_token_ids"] == ids
        assert prepared["mm_placeholders"]["image"][0].offset == payload["image_ranges"][0][0]


def test_prepared_generation_does_not_render_image_inputs():
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    from torchspec.inference.engine.smg_multimodal import generate_prepared

    class Engine:
        def _preprocess_cmpl(self, prompts):
            assert prompts == [{"prompt_token_ids": [1, 2]}]
            return [{"type": "token", "prompt_token_ids": [1, 2]}]

        def _render_and_run_requests(self, inputs, params, output_type, use_tqdm):
            values = list(inputs)
            assert len(params) == 2 and params[0] is not params[1]
            return values

    prepared = {"type": "multimodal", "prompt_token_ids": [3, 4]}
    values = generate_prepared(
        Engine(), [prepared, {"prompt_token_ids": [1, 2]}], SamplingParams(max_tokens=1)
    )
    assert values[0] is prepared
    assert values[1]["prompt_token_ids"] == [1, 2]

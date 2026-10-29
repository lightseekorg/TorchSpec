# SMG image preprocessing (opt-in)

The first supported path is **Qwen3-VL images with vLLM**. It uses SMG's pinned
`llm-multimodal` processor inside Python workers, alongside the existing SMG chat
renderer. This path prepares final IDs, image patches, image grids, image ranges,
and assistant loss masks before inference. The vLLM adapter submits prepared
`EngineInput` objects, bypassing its Hugging Face media processor.

Build the optional native feature:

```bash
TORCHSPEC_SMG_FEATURES=multimodal pip install -e '.[vllm]'
```

Use `TORCHSPEC_SMG_FEATURES=tokenizers-v1,multimodal` to combine build features.
A text-only build remains supported and reports an actionable error if this
renderer is selected. The pinned `nightly-e9d1398d9...` vLLM Dockerfile builds
with `multimodal` and checks the installed native feature. Other prebuilt images
need a wheel built with `multimodal`.

Set these fields in an otherwise complete training config:

```yaml
model:
  target_model_path: Qwen/Qwen3-VL-2B-Instruct

dataset:
  renderer: smg-qwen3-vl
  rendering_backend: smg
  defer_tokenization: false

training:
  train_with_decode: false

inference:
  inference_engine_type: vllm
```

The complete two-GPU recipe is
[`configs/vllm_qwen3_vl_2b_smg.yaml`](../configs/vllm_qwen3_vl_2b_smg.yaml).
From the repository root, supply your image-conversation datasets:

```bash
SMG_MM_THREADS=1 python -m torchspec.train_entry \
  --config configs/vllm_qwen3_vl_2b_smg.yaml \
  dataset.train_data_path=/absolute/path/train.jsonl \
  dataset.eval_data_path=/absolute/path/validation.jsonl
```

The recipe specifies the target's tied embedding/head keys, final norm key,
three auxiliary layers, and a matching one-layer Eagle3 draft configuration.

This explicit renderer is not selected by `auto` detection. Its prepared-input
adapter targets vLLM's 0.26 API; older versions without that interface fail instead
of reprocessing the images. Configure vLLM's image-per-prompt and encoder budgets
for the dataset as usual.

Inputs are complete conversations. User/tool messages may contain OpenAI-style
`text` and `image_url` parts; image URLs may be local paths, `file:` URLs,
HTTP(S) URLs, or base64 image data URLs. Only `detail: auto` is supported.
Assistant images, video, audio, and generation-prefix/decode training are not
supported by this first path. Images use vLLM's default white alpha background;
custom engine image-loader or processor overrides are not part of this contract.
When using several dataset workers, set `SMG_MM_THREADS=1` to avoid creating a
full image-processing thread pool inside every worker.

SMG handles decoding, EXIF orientation, resizing, normalization, patch layout,
and image-token replacements. TorchSpec keeps the existing text supervision
policy and shifts the mask with each replacement; image positions carry zero
loss. Qwen3-VL uses the checkpoint's system-message behavior rather than injecting
the text-Qwen registry's default system prompt.

Right truncation is allowed through text. A cut through an image span or its
vision wrapper raises an error; fully truncated images are removed from the
pixel/grid payload. Final prompt IDs and connector-reported IDs are checked
against the prepared IDs before their loss masks are accepted.

For this opt-in path, dataset disk caching is disabled: changing an image at the
same URL/path must not reuse a cache keyed only by JSONL metadata. Image tensors
are currently held with dataset samples, so this implementation is a correctness
milestone, not a memory-optimized large-corpus preprocessing service. A future
content-addressed cache or worker-side placement can reuse the same input contract.

`defer_tokenization` remains available for other models/backends. Removing it
universally requires equivalent processor and engine-adapter validation for each
supported path. Exhaustive checkpoint comparisons and GPU harnesses stay local;
focused regressions are in `tests/test_smg_vision.py`.

The initial qualification used Qwen3-VL-2B-Instruct on GB300 with vLLM
`0.26.1rc1.dev1102+ge9d1398d9`. On 64 ChartQA training images and 16 held-out
images, HF and SMG produced identical final IDs, pixels, grids, masks, and four
captured target-state layers. Paired 1,000-step Eagle3 runs produced identical
step metrics and final weights; held-out depth-0 loss fell from 11.46 to 2.29.
A separate two-GPU run exercised the real Ray/controller/Mooncake TCP path for
16 steps, including evaluation and checkpoint saving. These are bounded
correctness and convergence checks, not large-corpus performance measurements.

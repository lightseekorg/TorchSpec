# SMG rendering and tokenization

The optional SMG backend uses the standalone `smg-tokenizer` Python binding to
SMG's Rust `llm-tokenizer` crate. It requires no gateway, HTTP service, or GPU.
The default Hugging Face path is unchanged. This is a first integration for
complete, teacher-forced **text** conversations, not universal model support.

## Installation

TorchSpec owns the thin PyO3 bridge under `bindings/smg-tokenizer`. Cargo builds
it against unmodified upstream SMG at commit
`df17c6c24b7be9e6eae50d23373f5d459e7f722b`; `Cargo.lock` pins transitive
dependencies. No SMG fork, companion PR, separate checkout or gateway is needed.
Rust executes in each Python process and releases the GIL during rendering and
encoding. The default HF installation does not require Rust.

From the TorchSpec checkout, in the Python environment used by dataset workers:

```bash
python -m pip install maturin
rustup toolchain install 1.98.0 --profile minimal
RUSTUP_TOOLCHAIN=1.98.0 maturin develop --release --locked --manifest-path bindings/smg-tokenizer/Cargo.toml
# Optional faster encoder (build-time opt-in):
RUSTUP_TOOLCHAIN=1.98.0 maturin develop --release --locked --manifest-path bindings/smg-tokenizer/Cargo.toml --features tokenizers-v1
```

For deployment, use `maturin build --release --locked` with the same manifest
and install the resulting `smg_tokenizer` wheel in each worker environment.
The first build fetches the pinned upstream Rust source and crates; an installed
wheel needs neither Rust nor an SMG checkout. Build separately for each
OS/architecture; macOS wheels cannot run on Linux Grace hosts. The bridge is
optional and its wheel is not currently published to PyPI.

The library exposes `Tokenizer.encode_path` (`native`, `v1`, `tokenizers`, or
`tiktoken`) so qualification verifies the actual selected path. When changing the
Rust revision or bridge behavior, update the lockfile and bridge package version
(which participates in TorchSpec's cache key), then rerun the contract gates.

## Configuration

```yaml
dataset:
  rendering_backend: smg
  smg_exclude_k3: true
  chat_template: qwen3-thinking
  defer_tokenization: false
  smg_revision: null  # Prefer an immutable HF checkpoint revision or local snapshot.
  smg_template_kwargs:
    preserve_thinking: true
    enable_thinking: true
  smg_include_end_token: true
  smg_render_instant: null  # Pin an RFC3339 timestamp for templates using strftime_now.
  last_turn_loss_only: auto
training:
  train_with_decode: false
```

`chat_template` selects TorchSpec's supervision delimiters and optional default
system message. The checkpoint's SMG renderer owns the actual template and
encoding; TorchSpec's legacy model-specific formatting code is not called.
Per-row `generation_config.chat_template_kwargs` and explicit thinking/effort
fields override dataset defaults; sampling fields such as temperature are not
passed into rendering. Generation headers and continuation mode cannot be
overridden for this complete-conversation API.
Clock-dependent templates require `smg_render_instant` so dataset workers and
cache rebuilds render the same timestamp; that value is also in the cache key.

When the selected renderer is `kimi-k3`, the model path names Kimi-K3, or a local
config declares `model_type: kimi_k3`, `smg_exclude_k3: true` keeps the existing
checkpoint-Python renderer without importing SMG. Set `smg_exclude_k3: false`
to opt into the SMG K3 text path. Keep `renderer: kimi-k3` explicit for an aliased
or Hub checkpoint whose name hides its family. Other custom renderers are not
silently replaced.

### K3 training policy

```yaml
dataset:
  rendering_backend: smg
  renderer: kimi-k3
  smg_exclude_k3: false
  defer_tokenization: false
training:
  train_with_decode: false
```

`SmgK3Renderer` uses SMG's native segmented encoding and inherits the existing
`K3Renderer` training policy: assistant bodies and their internal XTML structure
are supervised, outer headers and `<|end_of_msg|>` are excluded, last-turn
selection precedes truncation. It does not use the generic text-offset adapter
or `smg_include_end_token`; K3's established EOM policy is preserved.

Generation settings retain the existing per-row `generation_config` contract:
no config means thinking enabled with **explicit null effort**; explicit enabled
configs require `low`, `high` or `max`, and disabled thinking rejects nonempty
reasoning. Dataset template kwargs and render-instant overrides are rejected.

The pinned SMG serving entry point conflates null effort with omitted effort,
adding a `max` directive. The TorchSpec adapter removes only that auto-generated
prefix after tool declarations. Both prefix pieces are rendered and encoded by
SMG; exact text and token-prefix checks must pass before removal. It never
flattens and re-encodes K3 text or searches arbitrary payloads for a directive.
Literal control spellings remain ordinary text, as in the checkpoint tokenizer.

K3 remains opt-in and text-only. Its qualification uses locally supplied real
checkpoint assets with pinned hashes, not the existing unit-test tokenizer stub:

```bash
PYTHONPATH=. python tools/qualification/k3_tokenizer_contract.py \
  --checkpoint /path/to/Kimi-K3 --report k3-contract.json
PYTHONPATH=. python tools/qualification/prepare_k3_convergence_data.py \
  --model /path/to/Kimi-K3 --source /path/to/training.jsonl --output /path/to/run/data
```

The corpus selector takes 256 training and 64 held-out conversations with disjoint
user prompts from an ordered source, preserves per-row configuration, requires
at least 32 supervised tokens after truncation to 512, and checks exact legacy/SMG
IDs and masks before selection. It records source line and row hashes.

For convergence, extract real K3 hidden states for those exact IDs at layer
boundaries 2, 46, 90 and 93 (the last is pre-norm). `extraction/manifest.json`
contains one entry per selected row, in order: `id`, `split`, and `layers`, each
with `layer_id`, `path` and the safetensors file's `sha256`. Each file contains
`input_ids` and finite `[sequence_length, 7168]` `hidden_states`. Extraction must
not split an individual 512-token prompt across prefill chunks.

```bash
PYTHONPATH=. python tools/qualification/train_k3_tokenizer_convergence.py \
  --assets /path/to/run/assets --extraction /path/to/run/extraction \
  --output /path/to/run/comparison
```

`assets/model` points to the pinned full K3 checkpoint. The comparison reruns
both actual dataset loaders, verifies every cached layer's IDs and hash, and
runs 1,000 optimizer steps per arm using the same initial one-layer Eagle3 draft
and schedule. Target embedding/head/norm weights come from K3. This isolates
preprocessing using one-rank training and cached real targets; it does not
reproduce a prior multi-node training job or qualify media/queue/RDMA paths.
The loss, accuracy and trajectory acceptance thresholds match the Qwen gate;
`--smoke` never qualifies convergence.

Measured on 2026-10-26 with the pinned checkpoint and a 320-row subset of the
existing K3 training corpus: both arms completed 1,000 steps, with identical
per-step metrics, all 11 evaluations, and byte-identical trainable checkpoints.
Held-out loss fell from 14.5035 to 5.0557; depth-0 accuracy rose from 0% to 44.03%;
early/late-100 mean training loss fell from 8.6286 to 1.3983. The one-layer draft
had 425,772,032 trainable parameters; target embedding/head were frozen. Real
targets were extracted with TP8/PP2 on 16 GB200 GPUs, then both training arms ran
on one GB200 using PyTorch 2.14.1+cu130. Final checkpoint SHA256:
`bd523ce828b54f34f1c415416369812f4b8675753d433c4906d67bb699674c37`.

The real-checkpoint tokenizer gate passed 126 cases, two required rejections,
14,802 truncation comparisons and 15 long-segment checks on macOS and
Linux/aarch64, including one/two-worker dataset loading and cache reloads.
`tests/tokenizer_contract/fixtures/k3_policy.json` freezes the expected token/mask
digests so a shared change to legacy and SMG masking cannot silently redefine
this policy. These measurements qualify the tested text-training setup, not
multimodal or live speculative-decoding performance.

## Supervision and compatibility

The adapter obtains exact SMG IDs and locates assistant bodies in the rendered
conversation. It requires explicit assistant/end delimiters, checks the number
of assistant spans, rejects raw message/tool content that can forge delimiters,
and rejects missing/crossing boundaries. The selected assistant spans must not
silently lose separately supplied reasoning or tool calls.

HF byte offsets map spans onto the **complete** token stream, including Unicode.
Tokens crossing a supervision boundary are conservatively masked out. End-of-turn
tokens are included by default; set `smg_include_end_token: false` to exclude
them. This supervises the end marker itself, never the following formatting
newline. The `text-spans-v2-eot-marker-only` cache version invalidates masks
from the initial prototype, which included that newline for Qwen but not
MiniMax. Last-turn selection happens before synchronized right truncation. SMG
rows always carry `packed_loss_mask`; they are not re-masked by the trainer.

The exact-mask policy is explicit and can differ from the legacy parser's
encode-prefix approximation or custom thinking formatting. Validate a training
corpus before reusing old caches or treating a migration as the same experiment.
The cache key includes the tokenizer/template file contents, binding version,
build feature, adapter version, supervision configuration, and template kwargs.
Existing pretokenized datasets bypass rendering and do not require the binding.

Flat tiktoken encodings without offsets are accepted only when the encoded
prefixes at supervision boundaries exactly match the full sequence. Native
segment-wise encodings are rejected by the generic adapter until they expose
training annotations; the dedicated K3 path uses its structural token IDs. Their IDs are preserved by the binding and never silently
replaced with flat-text encoding.

Unsupported in this first version: multimodal content/engine-side media expansion,
`defer_tokenization`, `train_with_decode`, templates without explicit delimiters,
and arbitrary model families whose emitted delimiters differ from the selected
TorchSpec supervision definition. These cases fail rather than falling back.

## Validation and performance

### Required correctness gate

`tests/tokenizer_contract/fixtures/manifest.json` pins the exact model revisions,
tokenizer/template artifact SHA256s, reference library versions, required case
inventory and fixture digests. Initial profiles are Qwen3-0.6B, the NousResearch
Llama-3-8B-Instruct checkpoint, and MiniMax-M2.5. These are exact checkpoints;
passing one does not qualify its entire model family.

Every accepted fixture contains checkpoint-rendered bytes and reference IDs,
plus explicitly authored assistant bodies, byte annotations and frozen token
positions for both end-marker policies. The oracle does not import the SMG
renderer, TorchSpec template registry or production loss-mask implementation.
Llama's unsupported reasoning/tool cases are required rejection tests.
Empty bodies can intentionally have zero loss when end-marker loss is disabled.
SMG dataset workers use the `spawn` multiprocessing context on Linux as well as
macOS: forking a parent that has already used native tokenizers can inherit
unusable thread-pool/lock state. The worker tests explicitly warm the parent
first. Required cases have a 180-second timeout so a regression fails the gate.

From the repository root, in an environment with TorchSpec and the binding:

```bash
python -m pip install -r tools/qualification/tokenizer-requirements.txt
PYTHONPATH=. python tools/qualification/tokenizer_contract.py \
  --backend smg --encode-path v1 --report tokenizer-contract.json
```

The runner fails on skipped, xfailed or deselected tests, missing assets, hash
mismatches, or encoder fallback. It downloads only pinned tokenizer assets;
model weights are unnecessary. To use existing local snapshots, set
`TORCHSPEC_TOKENIZER_PATHS` to a JSON file mapping each manifest model ID to its
directory. All listed artifacts are still verified, and an incomplete mapping
fails. `--backend hf` runs the independent reference/annotation gate without
requiring the extension; it does not qualify SMG.

For a binding built without the v1 feature, use `--encode-path default`. This
requires the manifest's explicit paths: native for Qwen/MiniMax and `tokenizers`
for this Llama checkpoint. It does not assume every model uses the native path.

Coverage includes exact renders and IDs, masks at every truncation position,
last-turn selection before truncation, rejected delimiter injection, JSON tool
arguments, packed-mask/padding roundtrips, fresh/warm encoders, concurrent calls,
and deterministic Unicode differential fuzzing. Raw fuzzing checks IDs only;
reviewed conversation fixtures establish supervision semantics.

The existing on-demand CPU workflow requires the pinned HF gate. The separate
`SMG tokenizer contract` workflow builds this repository's binding against its
pinned upstream dependency and requires both default-build and v1 gates.
It has no external source-selection inputs.
Making a check branch-protection-required is a repository setting, not established
by this source change.

Fixture changes are intentional specification changes. Record candidates with:

```bash
PYTHONPATH=. python tools/qualification/record_tokenizer_candidates.py \
  --paths model-paths.json --output /tmp/tokenizer-candidates-new
```

The recorder refuses to overwrite the reviewed fixture directory. Review the
rendered bytes, authored bodies, token positions and dependency/revision changes
before copying candidate files. Do not regenerate expectations to accommodate an
unexplained backend mismatch. SMG's Bellwether is the wider rendering/encoding
conformance suite; these fixtures add TorchSpec-specific training semantics and
do not claim to replace or run Bellwether's entire corpus.

### Actual training alignment smoke

In a CUDA environment with TorchSpec's HFRunner dependencies installed:

```bash
PYTHONPATH=. python tools/qualification/check_training_alignment.py \
  --report training-alignment.json
```

This executes real HFRunner inference, mask resolution, padding, and
Eagle3Trainer forward/backward using tiny random target/draft networks and full
checkpoint token vocabularies. Hooks inspect actual target/draft embedding
inputs, returned hidden states, supervision masks and three speculative depths.
The smoke freezes the existing Eagle3 convention: initial draft inputs and
target hidden states shift left once, while the initial loss mask stays in its
original coordinates; each subsequent depth shifts the inputs and mask once
more. It does not redefine this convention as ordinary next-token SFT labels.
It bypasses FSDP setup and queue/RDMA transport. It does not qualify pretrained
model numerics, vLLM/SGLang, multimodal expansion or distributed execution.

### Required 1,000-step convergence qualification

Tokenizer/unit gates are prerequisites. The training acceptance gate is
`tools/qualification/train_tokenizer_convergence.py`: **1,000 optimizer steps
per backend**, using pretrained Qwen3-0.6B and a pinned UltraChat train/test
split. It initializes a real one-layer Eagle3 draft through `Eagle3Trainer` and
uses TorchSpec's `BF16Optimizer` in single-rank REPLICATE mode. The target's
embedding/head are loaded and frozen; the remaining draft parameters train.

```bash
PYTHONPATH=. python tools/qualification/download_convergence_assets.py --output /path/to/assets
PYTHONPATH=. python tools/qualification/train_tokenizer_convergence.py \
  --assets /path/to/assets --output /path/to/new-run
```

Defaults: 256 unique training prompts, 64 disjoint held-out prompts, length 512,
batch size 2, two Eagle3 depths, seed 42, learning rate 1e-4, cosine decay and
15 warmup steps. Each sample uses the first user/assistant pair from UltraChat;
short supervision, oversized sources and unsupported literal markers are
counted and excluded before comparing backends. Selected raw conversations and
all artifact/token/mask hashes are retained.

The HF control uses the checkpoint renderer and an independently specified
single-turn assistant-body mask with the same explicit end-marker policy. It
does not claim equivalence to the legacy parser. The SMG arm uses the actual
dataset loader with spawned workers. Any token/mask mismatch aborts before
training. Exact-ID parity permits both arms to reuse the same real HFRunner
hidden-state cache; hooks check IDs consumed by the pretrained target.

Both arms use identical initialization and sample order. Passing requires:

- Exactly 1,000 steps each, with finite loss and gradient norm at every step.
- At least 10% reduction in held-out loss and in late-100 versus early-100
  training loss, plus at least one percentage point of held-out depth-0 token
  accuracy improvement, for each arm.
- At most 1% relative difference between the two per-step loss trajectories.

Criteria are recorded before training. Every step is logged; held-out evaluation
runs at step 0 and every 100 steps. A short `--smoke --steps 2` debug run is never
marked convergence-qualified. Checkpoints contain trainable draft parameters,
configuration and the pinned target reference for the frozen weights.
Teacher-forced accuracy and simulated acceptance length are diagnostic proxies,
not measurements from a live speculative-decoding server. Queue/RDMA and
multi-rank qualification remain separate.

### Performance measurements

```bash
PYTHONPATH=. pytest -q tests/test_smg_renderer.py tests/test_renderer_registry.py tests/test_kimi_k3_renderer.py
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 PYTHONPATH=. \
  python tools/benchmark_smg_tokenization.py --models models.json --output results.json --gigatoken
```

`models.json` is a list of `{ "name": "model label", "path": "/pinned/local/snapshot",
"chat_template": "qwen3-thinking" }`. Omit `--gigatoken` when it is not installed.
The harness compares IDs and masks before timing 2K/50K English/CJK cases.
Its HF comparison shares the supervision algorithm, so mask agreement there is
only a consistency check. Use the independent correctness gate above first.
It reports actual backend selection, environment, individual samples and medians.
Its synthetic repeated vocabulary warms word caches; it does not predict cold,
unique-corpus throughput or GPU training throughput.

SMG's accelerated paths may return only IDs. The current binding also runs the
reference HF pipeline to obtain safe offsets and checks exact ID equality. That
extra pass can dominate render/tokenize/mask latency, even when raw encoding is
much faster. Native training span annotations are the next optimization; do
not describe raw-encode speedups as end-to-end training speedups.

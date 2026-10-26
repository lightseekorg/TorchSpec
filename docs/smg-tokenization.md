# SMG rendering and tokenization

The optional SMG backend uses the standalone `smg-tokenizer` Python binding to
SMG's Rust `llm-tokenizer` crate. It requires no gateway, HTTP service, or GPU.
The default Hugging Face path is unchanged. This is a first integration for
complete, teacher-forced **text** conversations, not universal model support.

## Installation

The standalone binding is part of the companion SMG change under
`bindings/tokenizer-python`; installing the existing `smg` gateway package is
not sufficient. From that SMG checkout, in the Python environment used by all
dataset worker processes:

```bash
python -m pip install maturin
maturin develop --release --manifest-path bindings/tokenizer-python/Cargo.toml
# Optional faster encoder (build-time opt-in):
maturin develop --release --manifest-path bindings/tokenizer-python/Cargo.toml --features tokenizers-v1
```

Use SMG's pinned Rust toolchain. A wheel built with `maturin build --release`
can be installed on other machines of the same OS/architecture; macOS wheels
cannot be installed on the Linux Grace hosts. The library exposes
`Tokenizer.encode_path` (`native`, `v1`, `tokenizers`, or `tiktoken`) so a
benchmark can verify the actual selected path instead of assuming it.

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
config declares `model_type: kimi_k3`, `smg_exclude_k3: true` uses the existing
K3 renderer without importing SMG. Keep `renderer: kimi-k3` explicit for an
aliased or Hub checkpoint whose name hides its model family. Setting the flag
false currently raises an error: K3 is not yet qualified for this training API.
Other explicit custom renderers are not silently replaced.

## Supervision and compatibility

The adapter obtains exact SMG IDs and locates assistant bodies in the rendered
conversation. It requires explicit assistant/end delimiters, checks the number
of assistant spans, rejects raw message/tool content that can forge delimiters,
and rejects missing/crossing boundaries. The selected assistant spans must not
silently lose separately supplied reasoning or tool calls.

HF byte offsets map spans onto the **complete** token stream, including Unicode.
Tokens crossing a supervision boundary are conservatively masked out. End-of-turn
tokens are included by default; set `smg_include_end_token: false` to exclude
them. Last-turn selection happens before synchronized right truncation. SMG
rows always carry `packed_loss_mask`; they are not re-masked by the trainer.

The exact-mask policy is explicit and can differ from the legacy parser's
encode-prefix approximation or custom thinking formatting. Validate a training
corpus before reusing old caches or treating a migration as the same experiment.
The cache key includes the tokenizer/template file contents, binding version,
build feature, adapter version, supervision configuration, and template kwargs.
Existing pretokenized datasets bypass rendering and do not require the binding.

Flat tiktoken encodings without offsets are accepted only when the encoded
prefixes at supervision boundaries exactly match the full sequence. Native
segment-wise encodings are rejected by the training adapter until they expose
training annotations. Their IDs are preserved by the binding and never silently
replaced with flat-text encoding.

Unsupported in this first version: multimodal content/engine-side media expansion,
`defer_tokenization`, `train_with_decode`, templates without explicit delimiters,
and arbitrary model families whose emitted delimiters differ from the selected
TorchSpec supervision definition. These cases fail rather than falling back.

## Validation and performance

```bash
PYTHONPATH=. pytest -q tests/test_smg_renderer.py tests/test_renderer_registry.py tests/test_kimi_k3_renderer.py
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 PYTHONPATH=. \
  python tools/benchmark_smg_tokenization.py --models models.json --output results.json --gigatoken
```

`models.json` is a list of `{ "name": "model label", "path": "/pinned/local/snapshot",
"chat_template": "qwen3-thinking" }`. Omit `--gigatoken` when it is not installed.
The harness verifies exact IDs and masks before timing 2K/50K English/CJK cases.
It reports actual backend selection, environment, individual samples and medians.
Its synthetic repeated vocabulary warms word caches; it does not predict cold,
unique-corpus throughput or GPU training throughput.

SMG's accelerated paths may return only IDs. The current binding also runs the
reference HF pipeline to obtain safe offsets and checks exact ID equality. That
extra pass can dominate render/tokenize/mask latency, even when raw encoding is
much faster. Native training span annotations are the next optimization; do
not describe raw-encode speedups as end-to-end training speedups.

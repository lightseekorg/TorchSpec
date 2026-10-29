# SMG training tokenization

TorchSpec calls SMG's Rust `llm-tokenizer` library **inside each Python dataset
worker** through a small PyO3 extension. Loading, rendering and encoding release
the GIL. The native `torchspec_smg._native` module and its Python wrapper
ship inside the **TorchSpec wheel**. There is no separate tokenizer distribution,
server or subprocess, and no SMG fork is needed.

SMG owns chat formatting and encoding. TorchSpec owns assistant supervision,
last-turn selection and truncation. Rendering always consumes a complete
conversation: no generation header or continuation of the final message.

The pinned SMG API does not expose assistant spans or masks. Its Jinja renderer
treats `{% generation %}` blocks as plain blocks without retaining their bounds.
TorchSpec therefore still needs template delimiter metadata for generic text
masking and a K3 adapter for native segmented IDs. Token offsets alone do not
identify which message owns a token. These small span/mask adapters are owned by TorchSpec; training does not
require an upstream SMG span API.

## Selection and fallback

`dataset.rendering_backend: auto` is the default:

| Input/configuration | Route |
| --- | --- |
| Text with `llama3`, `qwen`, `qwen3-thinking`, `minimax-m2` | SMG |
| Kimi-K3 text, identified by renderer/model name/local config | SMG K3 |
| Deferred tokenization, decode training, other templates, custom renderers | Existing path |
| Pretokenized datasets with masks | No re-rendering |

To select the Python implementation explicitly:

```yaml
dataset:
  rendering_backend: huggingface
```

The same override works on the CLI: `dataset.rendering_backend=huggingface`.
`smg` forces the SMG route; `auto` selects it only for supported configurations.
The same backend selector applies to K3; there is no separate K3 exclusion flag.
Missing bindings, unsupported media and rendering errors are actionable failures,
never silent backend swaps.
Both backends retain their existing mode restrictions. Routes are logged and
backend/tokenizer/policy changes produce distinct dataset cache keys.

The public `preprocess_conversations` and `build_eagle3_dataset` APIs use the same
`rendering_backend="auto"` policy. Pass `rendering_backend="huggingface"` for the
legacy implementation. SMG loads artifacts from `tokenizer.name_or_path`, or an
explicit `target_model_path`; `smg_revision` pins remote artifacts. `smg_options`
accepts `template_kwargs`, `include_end_token` and `render_instant`. Per-row `tools`
and `generation_config` are forwarded. Native renderers are initialized inside
map workers, and explicit dataset cache names include backend/assets and mask
policy. Preformatted text and generation-prefix requests keep their legacy routes;
forcing SMG for those modes raises an error.

For generic SMG text profiles, `smg_template_kwargs` configures rendering and
per-row `generation_config` overrides it. `smg_include_end_token` defaults to true;
the formatting newline after the end marker is not supervised. Clock-dependent
templates require a pinned RFC3339 `smg_render_instant`. Ambiguous assistant
boundaries and discarded supervised reasoning/tool calls are rejected.
For structured reasoning and tool calls, substring presence alone is insufficient:
ordinary content can repeat the same text. The adapter renders a copy with a unique
field marker and requires it in the corresponding assistant body. These probes
preserve call/result structure, add preprocessing work only for structured fields,
and never supply the IDs used for training. A cache-version bump prevents old
cached rows from bypassing this validation.

## Installation

Install from this checkout with the normal command:

```bash
rustup toolchain install 1.98.0 --profile minimal  # source builds only
python -m pip install -e .
```

The declared setuptools-rust build dependency compiles the extension automatically
in release mode. Both editable installs and normal wheels include it; no second
`smg-tokenizer` installation is needed. Source builds require Rust 1.98.0 and a
C/C++ toolchain. Installing a prebuilt TorchSpec wheel needs neither. The existing
HF runtime fallback remains available; it does not disable native compilation
when building TorchSpec from source.

The vLLM, SGLang and TensorRT-LLM Dockerfiles build this same **TorchSpec wheel**
in a separate stage using the same base image/architecture, then install it and
resolve Python dependencies in the runtime image. The compiler stays in the build
stage. The checkout remains in `/root/torchspec` for configs/tools and Python development.
The native package uses a separate source directory so the checkout does not
shadow the installed extension.

To produce a wheel and source archive:

```bash
python -m pip install build
python -m build --outdir wheelhouse
```

`Cargo.toml` pins SMG at `df17c6c24b7be9e6eae50d23373f5d459e7f722b`, and builds
use `Cargo.lock` with `--locked`. The source archive includes the Rust sources,
manifest, lockfile and toolchain declaration. Set `TORCHSPEC_SMG_FEATURES=tokenizers-v1`
at build time for accelerated encoders. `Tokenizer.encode_path` and `build_features`
expose the actual selection; K3 uses tiktoken in either build. Generic offset
validation performs an extra reference encode, so encoder speedups do not imply
the same end-to-end training speedup.

## K3 policy

Use `renderer: kimi-k3` explicitly when a model alias hides its family. SMG keeps
K3's native segmented IDs; TorchSpec never flattens/re-encodes or strips generated
prefixes. Assistant bodies/internal XTML are supervised; outer headers and EOM
are excluded. Last-turn selection happens before truncation.

SMG owns tool argument serialization, including nested objects/arrays and floats.
Binding >=0.1.3 enables exact JSON float round-tripping; decimal spelling may
change, but the parsed numeric value must not. Older bindings are rejected.
Its compact JSON and exponent spellings may differ from HF; formatting equality
with HF is not an acceptance requirement. TorchSpec preserves SMG's native IDs
and derives the mask from those same IDs. No prompt/ID rewriting or silent
backend fallback is performed. The K3 cache version is bumped for this policy.

Absent `generation_config`, an empty config, or thinking-enabled missing/null
effort uses SMG's `max` default. Explicit effort names remain `low`, `high` or
`max`; contradictory reasoning metadata is rejected. Thinking-off rejects
nonempty assistant reasoning. The explicit HF fallback retains its existing
validation and does not gain SMG's null-effort semantics. K3 continues to reject
dataset-level template kwargs and clock overrides.

The earlier default-effort correction fixed TorchSpec missing-value handling that passed explicit null and
suppressed the default directive. All 273 single-turn examples in the audited
sample had generation prompt-token counts matching default/max; explicit null
was 67 tokens shorter. This is count evidence, not original-request byte equality.
Both SMG and HF fallback use the corrected default. Rebuild existing pretokenized
data and target-state caches to adopt it; changing a backend flag cannot repair
already-tokenized rows.

## Correctness gates and measured training

Frozen fixtures record text, IDs and assistant masks. The K3 parity gate covers the shared subset with the
real checkpoint API directly, checks missing-versus-null behavior, every
truncation boundary and long segmentation, and exercises worker spawning and
cache reloads.

```bash
PYTHONPATH=. python tools/qualification/tokenizer_contract.py \
  --backend smg --encode-path default --report tokenizer-contract.json
PYTHONPATH=. python tools/qualification/k3_tokenizer_contract.py \
  --checkpoint /path/to/Kimi-K3 --report k3-contract.json
```

Additional local qualification used an independent checkpoint segment-ownership
oracle and CPU probes through dataset loading, cache reload, batching, three
Eagle3 depths and loss/backpropagation. The local harness is outside this PR.
It used synthetic hidden states and a tiny draft, and preserved the trainer's
existing anchor-position convention; it does not qualify GPU/distributed training.

Prior paired 1,000-step tests used real target states with matched initialization and
sample order. Both pairs produced identical metrics, 11 evaluations and checkpoint
bytes: Qwen3-0.6B/UltraChat held-out loss 12.973 → 5.399; corrected K3 loss
14.508 → 4.865 and depth-0 accuracy 0% → 44.69%. K3 used 256 training/64 held-out
conversations, length 512, fresh TP8/PP2 targets, and one-rank Eagle3 training on
GB200. This does not qualify multimodal, distributed queues/RDMA or live acceptance.

The experimental runners and full provenance remain available at the
[qualified source snapshot](https://github.com/lightseekorg/TorchSpec/tree/2d47940ea1844327553e26a21890467c60d5d42c/tools/qualification).
They are kept out of this integration's final diff. Earlier K3 explicit-null
convergence established old-path equivalence only; the results above use the
corrected checkpoint default.

The newly accepted K3 nested/float arguments and null-effort inputs require
SMG-native qualification, not HF byte equality. Their local oracle adapts only
JSON spelling/default effort in the checkpoint's segment renderer, independently
labels assistant segments before encoding, and compares the resulting IDs/masks
with SMG. The prior 1,000-step comparisons did not exercise these newly accepted
inputs.

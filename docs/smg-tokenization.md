# SMG training tokenization

TorchSpec calls SMG's Rust `llm-tokenizer` library **inside each Python dataset
worker** through a small PyO3 extension. Loading, rendering and encoding release
the GIL. There is no tokenizer server or subprocess. `smg-tokenizer` is the wheel
containing this bridge, owned and built in this repository; it does not require
a separately published package or an SMG fork.

SMG owns chat formatting and encoding. TorchSpec owns assistant supervision,
last-turn selection and truncation. Rendering always consumes a complete
conversation: no generation header or continuation of the final message.

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
`smg_exclude_k3: true` remains a K3-only fallback. Missing bindings, unsupported
media and rendering errors are actionable failures, never silent backend swaps.
Both backends retain their existing mode restrictions. Routes are logged and
backend/tokenizer/policy changes produce distinct dataset cache keys.

For generic SMG text profiles, `smg_template_kwargs` configures rendering and
per-row `generation_config` overrides it. `smg_include_end_token` defaults to true;
the formatting newline after the end marker is not supervised. Clock-dependent
templates require a pinned RFC3339 `smg_render_instant`. Ambiguous assistant
boundaries and discarded supervised reasoning/tool calls are rejected.

## Installation

The repository's vLLM, SGLang and TensorRT-LLM Dockerfiles build the wheel in a
separate stage using the **same base image and target architecture**, then install
it into the final image. Rust, Cargo and maturin are build dependencies only.
Each worker image includes the extension; training needs no Rust installation.
The image build verifies that Python can load the compiled extension.

Outside these images, build once for the Python environment/OS/architecture:

```bash
python -m pip install 'maturin==1.15.0'
rustup toolchain install 1.98.0 --profile minimal
RUSTUP_TOOLCHAIN=1.98.0 maturin build --release --locked \
  --manifest-path bindings/smg-tokenizer/Cargo.toml --out wheelhouse
python -m pip install --no-deps wheelhouse/smg_tokenizer-*.whl
```

The bridge pins upstream SMG at `df17c6c24b7be9e6eae50d23373f5d459e7f722b`;
`Cargo.lock` pins transitive dependencies. The wheel is not published to PyPI.
HF fallback does not import it. The optional `--features tokenizers-v1` build
selects accelerated encoders; `Tokenizer.encode_path` and `build_features` expose
the actual selection. K3 uses tiktoken regardless of this feature. Generic offset
validation performs an extra reference encode, so encoder speedups do not imply
the same end-to-end training speedup.

## K3 policy

Use `renderer: kimi-k3` explicitly when a model alias hides its family. SMG keeps
K3's native segmented IDs; TorchSpec never flattens/re-encodes or strips generated
prefixes. Assistant bodies/internal XTML are supervised; outer headers and EOM
are excluded. Last-turn selection happens before truncation.

Absent `generation_config` leaves effort **omitted**, allowing the checkpoint's
`max` default. Explicit thinking-enabled configs require `low`, `high` or `max`;
thinking-off rejects nonempty assistant reasoning. Thinking-enabled explicit
null is unsupported because the checkpoint and SMG interpret it differently.
K3 rejects dataset-level template kwargs and clock overrides.

This fixes old TorchSpec missing-value handling that passed explicit null and
suppressed the default directive. All 273 single-turn examples in the audited
sample had generation prompt-token counts matching default/max; explicit null
was 67 tokens shorter. This is count evidence, not original-request byte equality.
Both SMG and HF fallback use the corrected default. Rebuild existing pretokenized
data and target-state caches to adopt it; changing a backend flag cannot repair
already-tokenized rows.

## Correctness gates and measured training

Frozen checkpoint fixtures independently specify text, IDs and assistant masks.
The K3 gate additionally calls the real checkpoint API directly, checks
missing-versus-null behavior, every truncation boundary and long segmentation,
and exercises worker spawning and cache reloads.

```bash
PYTHONPATH=. python tools/qualification/tokenizer_contract.py \
  --backend smg --encode-path default --report tokenizer-contract.json
PYTHONPATH=. python tools/qualification/k3_tokenizer_contract.py \
  --checkpoint /path/to/Kimi-K3 --report k3-contract.json
```

Paired 1,000-step tests used real target states with matched initialization and
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

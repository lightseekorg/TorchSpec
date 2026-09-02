# Configs

Training configuration files for TorchSpec Eagle3 distillation.

## Config hierarchy

```
default.yaml          Base defaults (all fields with sane values)
  └── specific.yaml   Model/setup-specific overrides
        └── CLI args   Runtime overrides (highest priority)
```

Any value in a config file can be overridden via CLI:

```bash
python -m torchspec.train_entry --config configs/sglang_qwen3_8b.yaml training.learning_rate=5e-5
```

## Config files

| Config | Backend | Model | Use case |
|--------|---------|-------|----------|
| `default.yaml` | — | — | Base template with all defaults |
| `hf_qwen3_8b.yaml` | HuggingFace | Qwen3-8B | Single-node, 3 GPUs |
| `sglang_qwen3_8b.yaml` | SGLang | Qwen3-8B | Single-node, 4+ GPUs |
| `sglang_kimi_k25_2node.yaml` | SGLang | Kimi-K2.5 | 2-node H200, 16 GPUs |
| `sglang_kimi_k25_3node.yaml` | SGLang | Kimi-K2.5 | 3-node H100, 24 GPUs |
| `vllm_kimi_k3_dflash2_mla_swa_gb300.yaml` | vLLM | Kimi-K3 | DFlash2 MLA+SWA, six-node GB300 recipe; replace placeholders |
| `vllm_kimi_k3_eagle3.yaml` | vLLM | Kimi-K3 | EAGLE3 MLA training recipe; replace placeholders |
| `ci/vllm_qwen3_8_27b_eagle3_pp_convergence.yaml` | vLLM | Qwen3.8-27B | GB200 TP2/PP2 convergence gate; patched vLLM required |

### vLLM pipeline-parallel recipes

Configs with `inference.vllm.pp_size > 1` require TorchSpec's patched vLLM
runtime. The standard vLLM environment installed by the general setup
instructions does not provide the `SupportsPP` declaration and per-stage
auxiliary hidden-state returns required by these recipes.

For the Qwen3.8 convergence recipe, use the image built from
`docker/vllm/nightly-e9d1398d9edfd90fcc1cf783805240e3effec013/Dockerfile`.
It pins the upstream vLLM image by tag and digest and applies
`patches/vllm/nightly-e9d1398d9edfd90fcc1cf783805240e3effec013/series`.
GPU CI additionally checks the in-container vLLM revision, base-image tag, V2
runner setting, and extraction patch markers before training starts.

## Key sections

| Section | Key fields | Description |
|---------|-----------|-------------|
| `model` | `target_model_path`, `draft_model_config` | Target model and optional draft architecture |
| `dataset` | `train_data_path`, `chat_template` | Training data and tokenization |
| `training` | `learning_rate`, `micro_batch_size`, `ttt_length` | Training hyperparameters |
| `inference` | `inference_engine_type`, `inference_num_gpus` | Inference backend configuration |
| `inference.sglang` | `tp_size`, `mem_fraction_static`, `extra_args` | SGLang engine settings (nested under inference) |
| `inference.offline` | `data_path`, `num_engines` | Offline training settings selected by `inference_engine_type: offline` |
| `mooncake` | `protocol`, `device_name` | Mooncake transfer engine settings |

## Custom online Ray placement

Use `training.placement_strategy: custom` when training and inference must run
on explicitly chosen Ray nodes. This is useful when the default `PACK` placement
would put actors on nodes with the wrong network locality, cache state, or GPU
partition. Offline training and materialization are single-role workflows and
use plain `PACK` placement.

IP-based placement uses Ray's built-in `node:<ip>` resource and does not require
custom Ray labels:

```yaml
training:
  placement_strategy: custom
  training_num_nodes: 1
  training_num_gpus_per_node: 8
  training_node_ips:
    - 10.0.0.1

inference:
  inference_num_gpus: 16
  inference_num_gpus_per_node: 8
  inference_node_ips:
    - 10.0.0.2
    - 10.0.0.3
```

Ray label selectors are also supported when your Ray version supports placement
group `bundle_label_selector`:

```yaml
training:
  placement_strategy: custom
  training_node_selectors:
    - {"torchspec/node": "trainer-0"}

inference:
  inference_node_selectors:
    - {"torchspec/node": "infer-0"}
    - {"torchspec/node": "infer-1"}
```

For each role, set either `*_node_ips` or `*_node_selectors`, not both. The
configured node order is preserved; for multi-node inference it determines the
engine actor order and therefore the `node_rank` passed to SGLang or vLLM.

## SGLang engine configuration

SGLang settings live under `inference.sglang` and are split into two tiers:

### Essential fields

Only fields that TorchSpec reads for its own logic live here. Everything else goes in `extra_args`.

```yaml
inference:
  sglang:
    tp_size: 8
    mem_fraction_static: 0.6
    nnodes: 2
    dist_timeout: 600
    enable_metrics: true
```

| Field | Default | Why it's here |
|-------|---------|---------------|
| `tp_size` | 8 | Factory validates it against `nnodes × gpus_per_engine`; engine uses it to compute the total TP degree |
| `pp_size` | 1 | Engine asserts `pp_size == 1` (only prefill, no pipeline parallelism) |
| `nnodes` | 1 | Factory uses it to decide single-node vs multi-node topology, replica count, and `dist_init_addr` auto-negotiation |
| `mem_fraction_static` | 0.8 | Passed to `sgl.Engine` as a protected key — TorchSpec manages it so `extra_args` cannot accidentally override it |
| `enable_metrics` | false | Read by TorchSpec's wandb integration to decide whether to scrape SGLang's metrics endpoint |
| `dist_init_addr` | null | Factory auto-negotiates it via Ray when null; multi-node engines receive it at `init()` time |
| `dist_timeout` | 60 | Engine sets it in `engine_kwargs` only for multi-node TP (`nnodes > 1`) |
| `init_timeout` | 300 | Factory uses it as the `ray.get()` timeout when waiting for engine initialization |

### `extra_args`: sgl.Engine passthrough

Any `sgl.Engine` keyword argument that TorchSpec doesn't need to inspect can be passed via `extra_args`. These are forwarded as-is to the engine constructor. A few overridable defaults are applied before `extra_args` (e.g. `log_level` defaults to `"warning"`), so you can override them here.

```yaml
inference:
  sglang:
    tp_size: 8
    enable_multimodal: true
    extra_args:
      attention_backend: flashinfer
      quantization: fp8
      disable_flashinfer_autotune: true
      watchdog_timeout: 1800
      enable_torch_compile: true
      log_level: info            # override the "warning" default
```

Or via CLI:

```bash
python -m torchspec.train_entry --config my.yaml inference.sglang.extra_args.watchdog_timeout=1800
```

### Protected keys

Some `sgl.Engine` parameters are always set by TorchSpec after `extra_args` is applied, so they cannot be overridden. If you put one of the commonly confused keys in `extra_args` (`model_path`, `tp_size`, `mem_fraction_static`, `nnodes`, `port`, `nccl_port`, `dist_init_addr`, `dist_timeout`), a warning is logged and the value is dropped. Other internally managed keys (e.g. `disable_radix_cache`, `enable_return_hidden_states`, `base_gpu_id`) are silently overwritten by the same mechanism without a warning since they're unlikely to appear in user configs.

## Draft model configs

Architecture configs for Eagle3 and DFlash2 draft models are in [`draft_models/`](draft_models/). These define the draft model architecture (hidden size, layers, attention heads, etc.) and are referenced via the `model.draft_model_config` field. K3 recipes use placeholder paths so private checkpoints and datasets are not committed.

## Creating a custom config

1. Copy the closest existing config
2. Update `model.target_model_path` to your model
3. Adjust GPU counts in `training` and `inference` sections
4. If your model needs a custom draft architecture, add a config in `draft_models/`

# TorchSpec

TorchSpec is a torch-native speculative decoding training framework. We introduce a disaggregated way of training speculative decoding draft models where inference and training are fully decoupled and stream hidden states directly from inference engine groups to distributed training workers via Mooncake (RDMA/TCP) store, allowing each side to scale independently.

<p align="center">
  <img src="docs/torchspec_architecture.png" alt="TorchSpec Architecture" width="100%">
</p>

## Setup

### Choose Your Backend

TorchSpec supports two inference backends:

| Backend | Best For | Installation |
|---------|----------|--------------|
| **vLLM** | Flexibility, easier deployment | `./tools/build_conda.sh 1 vllm` |
| **SGLang** | Production workloads, high throughput | `./tools/build_conda.sh 1 sglang` |
| **Both** | Development, comparison testing | `./tools/build_conda.sh 1 both` |

### Quick Setup

```bash
# Install with vLLM
./tools/build_conda.sh 1 vllm
micromamba activate torchspec

# Or install with SGLang
./tools/build_conda.sh
micromamba activate torchspec
```

To install into your current environment instead:
```bash
./tools/build_conda.sh current sglang  # or 'vllm' or 'both'
```

Optional — install Flash Attention:

```bash
pip install -e ".[fa]"
```

### Backend-Specific Usage

**vLLM:**
```bash
./examples/qwen3-8b-single-node/run.sh --config configs/vllm_qwen3_8b.yaml
```

**SGLang:**
```bash
./examples/qwen3-8b-single-node/run.sh
```

TorchSpec uses vLLM's **Worker Extension** mechanism to hook into the model's forward pass and capture hidden states directly in the worker processes. This avoids RPC serialization issues and enables reliable hidden states extraction.

## Quick Start

Train an Eagle3 draft model for Qwen3-8B using inference engine (4 GPUs: 2 training + 2 inference):

```bash
./examples/qwen3-8b-single-node/run.sh
```

Override any config value via CLI:

```bash
./examples/qwen3-8b-single-node/run.sh training.learning_rate=5e-5 training.num_train_steps=500
```

## Resume Vs Continual Training

TorchSpec supports two different ways to start from an existing checkpoint. They both use
`training.load_path`, but they do different things:

| Goal | `training.load_path` | `training.continual_training` | What gets restored |
|------|----------------------|-------------------------------|--------------------|
| Resume an interrupted run | Required | `false` (default) | Model, optimizer, LR scheduler, RNG, and step metadata |
| Start a new run from existing weights | Required | `true` | Model weights only |

### 1. Resume training from an existing checkpoint

Use this when a previous run was interrupted and you want to continue as if it never stopped.

```yaml
training:
  load_path: /path/to/old_run/checkpoints
  continual_training: false   # default; can be omitted

output_dir: /path/to/old_run
```

Behavior:

- TorchSpec reads the latest checkpoint under `training.load_path` unless you explicitly set `ckpt_step`.
- It restores model weights, optimizer state, LR scheduler state, RNG state, and step counters.
- Training resumes from the saved global step, so the old learning-rate schedule continues instead of restarting.

Use resume mode when:

- A job crashed or was preempted.
- You want to keep the same optimizer momentum and LR schedule.
- You want the run to continue writing checkpoints for the same training trajectory.

### 2. Continual training from the beginning using an existing checkpoint

Use this when you want to initialize from an existing model checkpoint but start a new training run with new optimization settings.

```yaml
training:
  load_path: /path/to/old_run/checkpoints
  continual_training: true
  learning_rate: 1e-5
  warmup_ratio: 0.01
  num_epochs: 1

output_dir: /path/to/new_run
```

Behavior:

- TorchSpec loads model weights from `training.load_path`.
- It does **not** restore optimizer state, LR scheduler state, RNG state, or saved step metadata.
- The new run starts from step 0 and uses the new config values in your YAML, such as a lower `learning_rate`, different `warmup_ratio`, or a new dataset.
- For BF16 training, TorchSpec also refreshes the optimizer's fp32 master weights from the checkpointed model so the first optimizer step does not overwrite the loaded weights.

Use continual training when:

- You want to adapt an existing draft model to new data.
- You want a fresh optimizer and fresh LR schedule.
- You want the new run to save into a different `output_dir` while reusing old weights.

### Choosing the right mode

If you want to continue the same run, use normal resume mode and leave `continual_training` unset or `false`.
If you want to start a new run from pretrained draft weights, set `continual_training: true` and point `output_dir` to a new directory.

## Examples

| Example | Backend | Model |
|---------|---------|-------|
| [hf-quickstart](examples/hf-quickstart/) | HuggingFace | Qwen3-8B |
| [qwen3-8b-single-node](examples/qwen3-8b-single-node/) | Inference Engine | Qwen3-8B |
| [kimi-k25-2node-h200](examples/kimi-k25-2node-h200/) | Inference Engine | Kimi-K2.5 |
| [kimi-k25-3node-h100](examples/kimi-k25-3node-h100/) | Inference Engine | Kimi-K2.5 |

See [examples/README.md](examples/README.md) for details.

## Checkpoint Conversion

Convert an FSDP checkpoint to HuggingFace format:

```bash
python tools/convert_to_hf.py --input-dir ./outputs/my_experiment/iter_0010000/
```

Vocabulary pruning — reducing the draft model's `lm_head` to a smaller token set and emitting `d2t`/`t2d` mappings — can be applied either **during training** (pre-pruning) or **at conversion time** (post-pruning):

- **Pre-pruning**: set `draft_vocab_size` in your training config. The checkpoint already contains the pruned `lm_head` and `d2t`/`t2d` buffers. Use the basic conversion command above — no extra flags needed.

- **Post-pruning**: train with the full vocabulary, then prune at conversion time by passing `--prune-vocab` along with a representative dataset to compute token frequencies:

```bash
python tools/convert_to_hf.py \
    --input-dir ./outputs/my_experiment/iter_0010000/ \
    --prune-vocab \
    --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
    --draft-vocab-size 32000 \
    --tokenizer Qwen/Qwen3-8B \
    --chat-template qwen \
    --prompt-key conversations
```

Pass `--cache-dir ./cache` to reuse the tokenized dataset cache from training.

## Metrics Reporting

W&B logging is disabled by default (`report_to: none`). To enable it, set `report_to: wandb` in your config and supply your API key.

## Troubleshooting

Set `TORCHSPEC_LOG_LEVEL=DEBUG` for verbose logging when diagnosing issues:

```bash
TORCHSPEC_LOG_LEVEL=DEBUG ./examples/qwen3-8b-single-node/run.sh
```

### Per-Rank File Logging

Set `TORCHSPEC_LOG_DIR` to an <b> absolute path </b> on a shared filesystem (NFS) to enable per-rank log files for every Ray actor (training and inference):

```bash
export TORCHSPEC_LOG_DIR=/my_project/running_logs
```

This creates a structured directory with one file per actor, organized by role and node:

```
running_logs/
  training/
    10.0.0.1/
      training_g0_rank0_20260301_080012.log
      training_g0_rank1_20260301_080012.log
    10.0.0.2/
      training_g0_rank2_20260301_080013.log
  inference/
    10.0.0.1/
      inference_g0_rank0_20260301_080014.log
    10.0.0.2/
      inference_g0_rank1_20260301_080015.log
```

The path must be an absolute path on a shared filesystem (NFS) accessible from all nodes. If `TORCHSPEC_LOG_DIR` is not set or the path is not writable, per-rank file logging is disabled and only Ray's default stdout/stderr capture is used.

| Issue | Reference |
|-------|-----------|
| Stuck or failing distributed runs, Ray actor errors | [docs/debugging_ray_jobs.md](docs/debugging_ray_jobs.md) |
| Ray cluster setup, actor hierarchy, placement groups | [docs/ray.md](docs/ray.md) |
| Pipeline bottlenecks, slow steps, throughput analysis | [docs/performance_metrics.md](docs/performance_metrics.md) |

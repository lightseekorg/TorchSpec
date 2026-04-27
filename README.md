# TorchSpec

TorchSpec is a torch-native speculative decoding training framework. We introduce a disaggregated way of training speculative decoding draft models where inference and training are fully decoupled and stream hidden states directly from inference engine groups to distributed training workers via [Mooncake](https://github.com/kvcache-ai/Mooncake) store, allowing each side to scale independently.

TorchSpec currently includes training flows and examples for:

- Kimi-K2.5
- MiniMax-M2.5
- Qwen3-Coder-Next

## 🚀 Blogs

- PyTorch blog: [TorchSpec: Speculative Decoding Training at Scale](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/)
- Release blog: [TorchSpec: Speculative Decoding Training at Scale](https://lightseek.org/blog/torchspec-speculative-decoding-training-at-scale.html)
- Released draft model: [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3)

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Examples](#examples)
- [Training Modes](#training-modes)
- [Checkpoint Conversion](#checkpoint-conversion)
- [Metrics Reporting](#metrics-reporting)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

<p align="center">
  <img src="docs/torchspec_architecture.png" alt="TorchSpec Architecture" width="100%">
</p>

TorchSpec is built around a disaggregated training pipeline:

- **Inference engines** generate target-model hidden states with either vLLM or SGLang.
- **Mooncake store** transfers tensors between inference and training without materializing them on disk.
- **Training workers** consume streamed hidden states to train speculative decoding draft models.

This separation keeps the training side focused on optimization while letting the inference side scale for hidden-state generation throughput.

## Quick Start

Train an Eagle3 draft model for Qwen3-8B on a single node with 4 GPUs (2 for training and 2 for inference):

```bash
./examples/qwen3-8b-single-node/run.sh
```

Override config values directly from the CLI:

```bash
./examples/qwen3-8b-single-node/run.sh training.learning_rate=5e-5 training.num_train_steps=500
```

## Setup

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

Optional: install Flash Attention support:

```bash
pip install -e ".[fa]"
```

### Backend-Specific Usage

**vLLM**

```bash
./examples/qwen3-8b-single-node/run.sh --config configs/vllm_qwen3_8b.yaml
```

**SGLang**

```bash
./examples/qwen3-8b-single-node/run.sh
```

TorchSpec uses vLLM's **Worker Extension** mechanism to hook into the model forward pass and capture hidden states directly inside worker processes, which avoids RPC serialization overhead during extraction. For SGLang, TorchSpec applies a patch to the existing codebase to enable hidden-state extraction.

## Examples

| Example | Backend | Model |
|---------|---------|-------|
| [hf-quickstart](examples/hf-quickstart/) | HuggingFace | Qwen3-8B |
| [qwen3-8b-single-node](examples/qwen3-8b-single-node/) | Inference engine | Qwen3-8B |
| [kimi-k25-2node-h200](examples/kimi-k25-2node-h200/) | Inference engine | Kimi-K2.5 |
| [kimi-k25-3node-h100](examples/kimi-k25-3node-h100/) | Inference engine | Kimi-K2.5 |
| [minimax-m25-5node-h200](examples/minimax-m25-5node-h200/) | Inference engine | MiniMax-M2.5 |

See [examples/README.md](examples/README.md) for more details about each example.

## Training Modes

### Resume vs. Continual Training

Both modes use `training.load_path`, but they restore different states:

| Goal | `training.load_path` | `training.continual_training` | What gets restored |
|------|----------------------|-------------------------------|--------------------|
| Resume an interrupted run | Required | `false` (default) | Model, optimizer, LR scheduler, RNG, and step metadata |
| Start a new run from existing weights | Required | `true` | Model weights only |

Resume the same run:

```yaml
training:
  load_path: /path/to/old_run/checkpoints

output_dir: /path/to/old_run
```

Start a new run from existing weights:

```yaml
training:
  load_path: /path/to/old_run/checkpoints
  continual_training: true
  learning_rate: 1e-5
  warmup_ratio: 0.01
  num_epochs: 1

output_dir: /path/to/new_run
```

## Checkpoint Conversion

Convert an FSDP checkpoint to HuggingFace format:

```bash
python tools/convert_to_hf.py --input-dir ./outputs/my_experiment/iter_0010000/
```

Vocabulary pruning, which reduces the draft model `lm_head` to a smaller token set and emits `d2t` and `t2d` mappings, can be applied either during training or at conversion time.

- **Pre-pruning**: set `draft_vocab_size` in your training config. The checkpoint already contains the pruned `lm_head` and `d2t`/`t2d` buffers, so the basic conversion command is enough.
- **Post-pruning**: train with the full vocabulary, then pass `--prune-vocab` at conversion time together with a representative dataset to compute token frequencies.

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

W&B logging is disabled by default with `report_to: none`. To enable it, set `report_to: wandb` in your config and provide your API key.

## Troubleshooting

Set `TORCHSPEC_LOG_LEVEL=DEBUG` for more verbose logging when diagnosing issues:

```bash
TORCHSPEC_LOG_LEVEL=DEBUG ./examples/qwen3-8b-single-node/run.sh
```

## Mooncake SEGFAULT

Current Mooncake version has a bug with TCP-only hosts causing a SEGFAULT error. Set `MC_STORE_MEMCPY=0` until the [upstream issue](https://github.com/kvcache-ai/Mooncake/issues/1986) is fixed.

### Per-Rank File Logging

Set `TORCHSPEC_LOG_DIR` to an absolute path on a shared filesystem (NFS) to enable per-rank log files for every Ray actor on both training and inference:

```bash
export TORCHSPEC_LOG_DIR=/my_project/running_logs
```

This creates a structured directory with one file per actor, organized by role and node:

```text
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

The path must be absolute and writable from all nodes. If `TORCHSPEC_LOG_DIR` is unset or not writable, per-rank file logging stays disabled and Ray falls back to stdout/stderr capture.

| Issue | Reference |
|-------|-----------|
| Stuck or failing distributed runs, Ray actor errors | [docs/debugging_ray_jobs.md](docs/debugging_ray_jobs.md) |
| Ray cluster setup, actor hierarchy, placement groups | [docs/ray.md](docs/ray.md) |
| Pipeline bottlenecks, slow steps, throughput analysis | [docs/performance_metrics.md](docs/performance_metrics.md) |

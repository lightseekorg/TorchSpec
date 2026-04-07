# DFlash

DFlash is a block-diffusion draft model for speculative decoding. Unlike Eagle3's autoregressive approach, DFlash predicts 16-token blocks in parallel using dual-source KV attention from the target model's hidden states.

## Architecture

| | DFlash | Eagle3 |
|---|--------|--------|
| Loss | CE (cross-entropy) | KL (forward KL) |
| Prediction | Block-parallel (block_size=16) | Autoregressive (ttt_length=7) |
| Target layers | 5 layers | 3 layers |
| Mask | Block-causal (FlexAttention) | Causal |
| Context | Dual-source KV (W_proj) | Input fusion (fc layer) |
| Forward passes per step | 1 | 7 (sequential) |

### Key Design Decisions

- **Standalone model**: DFlash does not inherit Eagle3DraftModel — the interfaces are fundamentally different (dual-source KV vs input fusion, block-parallel vs autoregressive).
- **FlexAttention for block-causal mask**: Reuses TorchSpec's `compile_friendly_flex_attention` and `compile_friendly_create_block_mask` singletons.
- **CE loss**: Cross-entropy against ground truth tokens (not KL divergence against target distribution like Eagle3).
- **Config-based trainer dispatch**: `TrainerActor.init()` resolves the draft config and uses `isinstance(config, DFlashConfig)` to select the trainer.
- **Shared target model**: Reuses `Eagle3TargetModel` with generalized N-layer support. The same hook-based mechanism works for both 3 and 5 layers.

---

## Prerequisites

### Hardware

| Setup | GPUs | VRAM | Notes |
|-------|------|------|-------|
| **Multi-GPU (recommended)** | 4x H100 80GB | 320 GB total | 1 inference + 3 training (FSDP) |

- **Disk**: ~100 GB free (model weights ~16 GB, dataset ~1 GB, checkpoints ~30 GB)
- **CUDA**: 12.4+
- **Python**: 3.11+

### System Libraries

Required for Mooncake transfer engine (RDMA support):

```bash
# Ubuntu/Debian
sudo apt-get install -y libibverbs-dev librdmacm-dev libnuma-dev

# RHEL/CentOS
sudo yum install -y libibverbs-devel librdmacm-devel numactl-devel
```

---

## Installation

### Step 1: Clone and checkout

```bash
git clone https://github.com/torchspec-project/TorchSpec.git
cd TorchSpec
```

### Step 2: Install SGLang 0.5.9

SGLang provides the inference engine and pulls in PyTorch 2.9.1:

```bash
pip install "sglang[all]==0.5.9" --find-links https://flashinfer.ai/whl/cu124/torch2.9/flashinfer-python
```

### Step 3: Apply SGLang patch

The patch adds speculative training hooks (`enable_aux_hidden_states`, etc.):

```bash
SGLANG_DIR=$(python -c "import sglang; print(sglang.__path__[0])")
cd "$(dirname "$SGLANG_DIR")"
git apply /path/to/TorchSpec/patches/sglang/v0.5.8.post1/sglang.patch
cd /path/to/TorchSpec
```

### Step 4: Install TorchSpec

```bash
pip install -e ".[dev]"
```

### Step 5: Set environment variables

These are **required** for correct training behavior:

```bash
export HF_HOME=/path/to/your/huggingface/cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
```

### Step 6: Verify installation

```bash
python -c "
import torchspec
import torch
print(f'TorchSpec installed')
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```

---

## Data Preparation

DFlash expects a JSONL file with conversation data:

```json
{"id": "conv_001", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Recommended dataset** — PerfectBlend 50K:

```bash
python scripts/tools/prepare_perfectblend.py \
  --output /path/to/data/perfectblend_50k.jsonl \
  --max_samples 50000
```

Or bring your own JSONL in the same format. Sequences shorter than 32 tokens of supervised content are automatically filtered.

---

## Training

Uses SGLang for inference on 1 GPU, FSDP training on 3 GPUs:

```bash
python -m torchspec.train_entry \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  dataset.train_data_path=/path/to/data/perfectblend_50k.jsonl \
  output_dir=./outputs/qwen3-8b-dflash
```

Key hyperparameters in `configs/sglang_qwen3_8b_dflash.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `micro_batch_size` | 1 | Per-GPU batch size |
| `draft_accumulation_steps` | 4 | Gradient accumulation |
| `global_batch_size` | 12 | 1 x 3 GPUs x 4 accum |
| `max_seq_length` | 2048 | Max tokens per sequence |
| `learning_rate` | 6e-4 | AdamW learning rate |
| `warmup_ratio` | 0.04 | Warmup fraction |
| `num_epochs` | 3 | Training epochs |
| `fsdp_strategy` | FULL_SHARD | Shards params + optimizer |
| `dflash_block_size` | 16 | Block-parallel prediction size |
| `dflash_num_anchors` | 512 | Anchor positions per sequence |
| `save_interval` | 1000 | Checkpoint every N steps |

Expected throughput: **~5 step/s** (~15 samples/sec) on 4x H100.

### Resume from Checkpoint

```bash
python -m torchspec.train_entry \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  dataset.train_data_path=/path/to/data/perfectblend_50k.jsonl \
  training.load_path=./outputs/qwen3-8b-dflash/checkpoints
```

The trainer reads `latest_checkpointed_iteration.txt` and resumes from that step, restoring model weights, optimizer state, LR scheduler, and RNG state.

---

## Monitoring

| Metric | Healthy | Problem |
|--------|---------|---------|
| step time | ~1.0 s/step | >2 s/step |
| loss | Decreasing (10 -> 3 over epoch 1) | Flat or increasing |
| accuracy | Increasing (0 -> 0.25 over epoch 1) | Stuck at 0 |
| data_time | <0.7 s | >1 s consistently |

W&B logging is supported out of the box. Set `WANDB_API_KEY` and `WANDB_PROJECT` before launching.

---

## After Training

### 1. Extract Checkpoint

Convert the FSDP distributed checkpoint to a single file:

```bash
python scripts/tools/extract_dflash_checkpoint.py \
  --checkpoint_dir outputs/qwen3-8b-dflash/checkpoints/iter_NNNNNNN \
  --output /tmp/dflash_draft.pt
```

### 2. Upload Checkpoint (optional)

```bash
huggingface-cli upload YOUR_ORG/dflash-qwen3-8b \
  outputs/qwen3-8b-dflash/checkpoints/iter_NNNNNNN/ \
  --repo-type model
```

---

## Validation Results

### Best Model: P2-WSD

Trained on 800K PerfectBlend for 3 epochs with WSD (Warmup-Stable-Decay) LR schedule on 8x H100.

| Dataset | Our τ | z-lab τ | E2E Speedup |
|---------|-------|---------|-------------|
| gsm8k | 3.89 | 3.38 | 2.47x |
| math500 | 4.19 | 4.61 | 2.80x |
| aime24 | 3.98 | 4.12 | 2.60x |
| aime25 | 3.69 | 4.07 | 2.42x |
| humaneval | 4.30 | — | 2.68x |
| mbpp | 3.47 | — | 2.14x |
| livecodebench | 4.72 | — | 2.96x |
| swe-bench | 2.63 | — | 1.70x |
| mt-bench | 2.76 | — | 1.56x |
| alpaca | 2.47 | — | 1.55x |

**Domain averages**: Math τ=3.94 (2.7% gap to z-lab 4.05), Code τ=4.16, General τ=2.62.

### Training Progression

| Model | Dataset | Epochs | Optimizer Steps | Math τ | Gap to z-lab |
|-------|---------|--------|-----------------|--------|--------------|
| Phase H | 760K PerfectBlend, cosine LR | 3 | 189K | 3.79 | 6.4% |
| P2-accum1 | 800K PerfectBlend, cosine LR, accum=1 | 3 | 378K | 3.83 | 5.4% |
| **P2-WSD** | **800K PerfectBlend, WSD LR** | **3** | **189K** | **3.94** | **2.7%** |

**Key insight**: The WSD schedule outperforms cosine decay. Its stable LR phase (76% of training at peak LR) lets the model explore the loss landscape more thoroughly before settling, preventing the premature convergence seen with cosine decay.

### Best Known Config

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 6e-4 |
| `min_lr` | 6e-5 |
| `lr_decay_style` | WSD |
| `wsd_decay_ratio` | 0.2 |
| `wsd_decay_style` | cosine |
| `warmup_ratio` | 0.04 |
| `weight_decay` | 0.01 |
| `draft_accumulation_steps` | 2 |
| `dflash_loss_decay_gamma` | 7.0 |
| `dflash_num_anchors` | 512 |

---

## Configuration Reference

### DFlash-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dflash_block_size` | 16 | Tokens predicted per block |
| `dflash_num_anchors` | 512 | Anchor positions per sequence (biggest speed lever) |
| `dflash_loss_decay_gamma` | 7.0 | Exponential decay rate for position-wise loss weighting |
| `dflash_num_target_layers` | 5 | Number of target model layers to project from |

### Draft Model Architecture

Defined in `torchspec/config/dflash_draft_config.json`:

- 5-layer transformer (~1.05B trainable parameters)
- Frozen embedding + LM head shared from Qwen3-8B (~622M frozen)
- Hidden size: 4096, 32 attention heads, 8 KV heads (GQA)
- Target layer IDs: [1, 9, 17, 25, 33] (evenly spaced across Qwen3-8B's 36 layers)

---

## Troubleshooting

| Issue | Symptom | Fix |
|-------|---------|-----|
| Missing RDMA libs | `ImportError: libibverbs.so.1` | Install system libs (see Prerequisites) |
| SGLang patch not applied | `unexpected keyword argument 'enable_aux_hidden_states'` | Re-apply the SGLang patch (Step 3) |
| Slow training (3x expected) | step time >3 s | Set `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` |
| OOM on A100 40GB | CUDA out of memory | Use H100/A100 80GB, or reduce `max_seq_length` / `dflash_num_anchors` |
| Loss stuck at ~10 | No convergence after 500 steps | Check data format, ensure conversations have assistant turns |
| `No module named 'sglang'` | Import error | Install SGLang (Step 2) |
| HF cache miss in Ray workers | Model re-downloads on each step | Set `HF_HOME` env var before launching |

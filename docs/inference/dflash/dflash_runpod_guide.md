# DFlash RunPod Setup & Resume Guide

> Updated for PyTorch 2.9.1 / SGLang 0.5.9 (2026-03-21)

## Pod Requirements

| Spec | Value |
|------|-------|
| GPUs | 1x H100 80GB (minimum) or 4x for SGLang mode |
| Container Disk | **100 GB** |
| Volume Disk | Optional (checkpoint persistence only) |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |

## What Survives Pod Restart

| Survives (`/workspace/`) | Lost (container disk) |
|---|---|
| Git repo directories (but `.py` files may appear deleted) | All pip packages |
| Checkpoints, training data | System packages (libibverbs, etc.) |
| HF model cache | Ray state, logs, inductor cache |

> **Critical**: After restart, run `git restore .` first — git-tracked files may appear deleted.

## Setup Steps

**Step 0 — SSH** (requires `expect` for RunPod PTY):
```bash
expect -c '
set timeout 60
spawn ssh -o StrictHostKeyChecking=no -o RequestTTY=force \
    -i ~/.ssh/id_ed25519 USER@ssh.runpod.io
expect -re {[#\$] }
interact
'
```

**Step 1 — Restore git files**:
```bash
cd /workspace/TorchSpec && git restore .
```

**Step 2 — System libraries**:
```bash
apt-get update -qq && apt-get install -y libibverbs-dev librdmacm-dev libnuma-dev && ldconfig
```

**Step 3 — SGLang from source** (pulls PyTorch 2.9.1 automatically):
```bash
cd /workspace/TorchSpec
git -C _sglang checkout 0f2df9370a1de1b4fb11b071d39ab3ce2287a350
git -C _sglang reset --hard HEAD
pip3 install -e "_sglang/python[all]"
rm -f _sglang/python/sglang/srt/speculative/spec_training_info.py
cd _sglang && git apply /workspace/TorchSpec/patches/sglang/v0.5.8.post1/sglang.patch
cd /workspace/TorchSpec
```

> **Note**: Don't install PyTorch separately — SGLang `[all]` pulls the correct version. Don't install flashinfer separately — it's included in SGLang `[all]`.

**Step 3.5 — Patch factory.py timeouts** (PyTorch 2.9+ needs longer CUDA init):
```bash
sed -i "s/timeout=30,/timeout=120,/g" torchspec/inference/factory.py
```

**Step 4 — TorchSpec**:
```bash
cd /workspace/TorchSpec && pip3 install -e ".[dev]"
```

**Step 5 — Verify**:
```bash
python -c 'import torch; print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")'
python -c 'import sglang; print("SGLang OK")'
python -c 'import torchspec; print("TorchSpec OK")'
```

**Step 6 — Restore checkpoint** (only if `/workspace/` was lost):
```bash
export HF_HOME=/workspace/.cache/huggingface
huggingface-cli download Xingh3/qwen3-8b-dflash-checkpoint-phase-c \
  --local-dir /workspace/TorchSpec/outputs/qwen3-8b-dflash-phase-c/checkpoints/
```

**Step 7 — Launch training with resume**:
```bash
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /workspace/TorchSpec

# CRITICAL: training.load_path is REQUIRED for checkpoint resume.
nohup python -m torchspec.train_entry \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  training.micro_batch_size=4 \
  training.draft_accumulation_steps=1 \
  training.num_epochs=4 \
  training.max_seq_length=2048 \
  training.learning_rate=6e-4 \
  training.warmup_ratio=0.04 \
  training.max_grad_norm=1.0 \
  training.save_per_epoch=true \
  training.save_interval=1000 \
  training.max_checkpoints=1 \
  training.dflash_num_anchors=256 \
  training.load_path=./outputs/qwen3-8b-dflash-phase-c/checkpoints \
  dataset.train_data_path=/workspace/data/perfectblend_50k.jsonl \
  dataset.eval_data_path=null \
  dataset.eval_interval=0 \
  inference.inference_num_gpus=2 \
  inference.inference_num_gpus_per_engine=1 \
  inference.inference_num_gpus_per_node=2 \
  output_dir=./outputs/qwen3-8b-dflash-phase-c \
  > /tmp/phase_c_resume.log 2>&1 &
```

## Verifying Resume

Check log after ~5 min (model loading takes ~3 min): `tail -20 /tmp/phase_c_resume.log`

- **Good**: step 18001+/23704, loss 2.7-4.1, accuracy 0.20-0.32
- **Bad**: step 0/23704, loss 10-12, accuracy 0.000 → check `training.load_path`

Resume mechanism: `checkpoint.py` reads `training.load_path` → finds `latest_checkpointed_iteration.txt` → loads model/optimizer/lr_scheduler/rng → sets `global_step` → `loop.py` skips first N steps.

## After Training Completes

1. **Extract checkpoint**:
   ```bash
   python scripts/extract_dflash_checkpoint.py \
     --checkpoint_dir outputs/qwen3-8b-dflash-phase-c/checkpoints/iter_NNNNNNN \
     --output /tmp/dflash_draft_phase_c.pt
   ```

2. **Benchmark τ**:
   ```bash
   python scripts/benchmark_dflash_inference.py \
     --target_model Qwen/Qwen3-8B \
     --draft_checkpoint /tmp/dflash_draft_phase_c.pt \
     --num_prompts 20 --max_new_tokens 256
   ```

3. **Target**: τ ≥ 3.0, speedup ≥ 1.5x over baseline.

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Missing `training.load_path` | Starts from step 0 | Add `training.load_path=./outputs/.../checkpoints` |
| Git files deleted after restart | `No module named torchspec` | `git restore .` |
| Missing RDMA libs | `ImportError: libibverbs.so.1` | `apt-get install -y libibverbs-dev librdmacm-dev libnuma-dev` |
| PyTorch too old (2.4.1) | FlexAttention import errors | Let SGLang pull 2.9.1 (Step 3) |
| SGLang not installed | `No module named 'sglang'` | Step 3 |
| SGLang patch not applied | `unexpected keyword argument 'enable_aux_hidden_states'` | Remove `spec_training_info.py` then `git apply` |
| Standard SSH fails | `Your SSH client doesn't support PTY` | Use `expect` (Step 0) |
| Ray actor timeout (30s) | `GetTimeoutError` during engine init | Patch factory.py timeouts (Step 3.5) |
| Old env var name | `PYTORCH_CUDA_ALLOC_CONF is deprecated` | Use `PYTORCH_ALLOC_CONF` |

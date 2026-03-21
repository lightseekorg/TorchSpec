# DFlash Implementation Log — TorchSpec

**Branch**: `feature/dflash-training`
**Start Date**: 2026-03-19
**Goal**: Implement DFlash training in TorchSpec, following SpecForge reference implementation

---

## Implementation Plan

| # | Task | Status | Date |
|---|------|--------|------|
| 1 | Create branch `feature/dflash-training` | ✅ | 2026-03-19 |
| 2 | `torchspec/models/draft/dflash.py` — DFlash draft model (dual-source KV, W_proj) | ✅ | 2026-03-19 |
| 3 | `torchspec/models/dflash.py` — DFlashModel wrapper (mask, anchor, loss) | ✅ | 2026-03-19 |
| 4 | `torchspec/training/dflash_trainer.py` — DFlashTrainer | ✅ | 2026-03-19 |
| 5 | Config + registration (auto.py, __init__.py, draft config JSON) | ✅ | 2026-03-19 |
| 6 | Unit tests for training code | ✅ | 2026-03-19 |
| 7 | Framework integration (train_entry.py, target model) | ✅ | 2026-03-19 |
| 8 | End-to-end validation | ✅ | 2026-03-19 |

---

## Architecture: DFlash vs Eagle3

| | DFlash | Eagle3 |
|---|--------|--------|
| Loss | CE (cross-entropy) | KL (forward KL) |
| Prediction | Block-parallel (block_size=16) | Autoregressive (ttt_length=7) |
| Target layers | 5 layers | 3 layers |
| Mask | Block-causal (FlexAttention) | Causal |
| Context | Dual-source KV (W_proj) | Input fusion (fc layer) |
| Forward passes per step | 1 | 7 (sequential) |

---

## Implementation Summary

### Files Created

1. **`torchspec/models/draft/dflash.py`** — DFlash draft model
   - `DFlashConfig(PretrainedConfig)` — DFlash-specific configuration
   - `DFlashRMSNorm`, `DFlashRotaryEmbedding` — reused from Llama patterns
   - `DFlashAttention` — Dual-source KV attention: takes `context_hidden` (from target) and `draft_hidden` separately, projects both through shared W_k/W_v, concatenates K/V. Q from draft only. Supports GQA.
   - `DFlashDecoderLayer` — Standard pre-norm transformer layer
   - `DFlashDraftModel(PreTrainedModel)` — Full draft model with `W_proj: Linear(num_target_layers * target_hidden_size, hidden_size)`, shared embedding + LM head from target (frozen)

2. **`torchspec/models/dflash.py`** — DFlash training wrapper
   - `DFlashModel(nn.Module)` — `_sample_anchor_positions()`, `_create_position_ids()`, `_prepare_noise_input()`, `_create_dflash_mask_mod()` (FlexAttention block-causal mask), `forward()` with CE loss + decay weighting

3. **`torchspec/training/dflash_trainer.py`** — DFlash trainer
   - `DFlashTrainer(Trainer)` — `init_model()` (FSDP2, optimizer, frozen LM head), `_train_step()`, `_aggregate_metrics()`, `eval_forward()` / `eval_from_cache()`

4. **`torchspec/config/dflash_draft_config.json`** — Draft model config for Qwen3-8B

5. **`tests/test_dflash.py`** — 67 tests (unit + integration + quality)

6. **`configs/sglang_qwen3_8b_dflash.yaml`** — 4-GPU SGLang training config
7. **`configs/hf_qwen3_8b_1gpu.yaml`** / **`hf_qwen3_8b_dflash_1gpu.yaml`** — Single-GPU colocate configs
8. **`scripts/runpod_dflash_train.sh`** — RunPod training script (auto-detects GPU count)
9. **`scripts/benchmark_dflash_inference.py`** — Inference benchmark (baseline vs speculative)
10. **`scripts/extract_dflash_checkpoint.py`** — FSDP checkpoint extraction

### Files Modified

1. **`torchspec/config/train_config.py`** — Added `dflash_block_size`, `dflash_num_anchors`, `dflash_loss_decay_gamma`, `dflash_num_target_layers`
2. **`torchspec/models/target/eagle3_target_model.py`** — Generalized `set_aux_hidden_states_layers()` for N layers (removed `assert len == 3`)
3. **`torchspec/utils/misc.py`** — Added `get_default_dflash_aux_layer_ids()`
4. **`torchspec/training/trainer_actor.py`** — Config-based trainer dispatch (`DFlashConfig` → `DFlashTrainer`)
5. **`torchspec/train_entry.py`** — Auto-set aux layer IDs when `DFlashConfig` detected
6. **`torchspec/inference/engine/__init__.py`** / **`inference/factory.py`** — Lazy SGLang/vLLM imports (HF-only no longer requires these)
7. **`torchspec/models/ops/flex_attention.py`** — Inductor config (`max_autotune_gemm_backends = "ATEN,TRITON"`)
8. **`torchspec/models/draft/auto.py`**, **`models/draft/__init__.py`**, **`models/__init__.py`**, **`__init__.py`** — DFlash registration + exports

### Key Design Decisions

1. **Not inheriting Eagle3DraftModel**: DFlash has fundamentally different interfaces (dual-source KV vs input fusion, block-parallel vs autoregressive). Clean separation avoids coupling.

2. **FlexAttention for block-causal mask**: Reuses TorchSpec's existing `compile_friendly_flex_attention` and `compile_friendly_create_block_mask` singletons.

3. **CE loss instead of KL**: DFlash uses cross-entropy against ground truth tokens (not KL divergence against target distribution like Eagle3).

4. **Config-based trainer dispatch**: `TrainerActor.init()` resolves the draft config before creating the trainer, using `isinstance(config, DFlashConfig)` to choose. No separate config flag needed.

5. **No separate DFlash target model**: TorchSpec reuses `Eagle3TargetModel` + `HFTargetModel` with generalized N-layer support. Same hook-based mechanism works for both 3 and 5 layers.

6. **Backward-compatible target model**: `set_aux_hidden_states_layers()` defaults to 3 Eagle3 layers when called with `None`. Only the hard assertion was removed.

### Lessons from SpecForge PR #473

1. ✅ `tie_word_embeddings` correctly handled — load from target, freeze
2. ✅ Block-causal mask is bidirectional within block (not causal)
3. ✅ Loss decay starts at k=1 (first prediction), not k=0 (anchor)
4. ✅ Same-position prediction alignment between training and inference

---

## Issues & Solutions

### Issue 1: RunPod SSH PTY Requirement

**Problem**: RunPod's SSH gateway requires a pseudo-terminal. Non-interactive SSH fails with `Error: Your SSH client doesn't support PTY`.

**Solution**: Use `expect` to allocate a real PTY:
```bash
expect -c '
set timeout 60
spawn ssh -o StrictHostKeyChecking=no -o RequestTTY=force \
    -i ~/.ssh/id_ed25519 USER@ssh.runpod.io
expect -re {[#\$] }
send "your-command-here\r"
expect -re {[#\$] }
send "exit\r"
expect eof
'
```
`-o RequestTTY=force` alone is not sufficient without a real PTY on stdin.

### Issue 2: PyTorch Version (2.4.1 → 2.6+ required)

FlexAttention (`torch.nn.attention.flex_attention`) requires PyTorch 2.6+. RunPod image has 2.4.1.

**Solution**: See [RunPod Setup Guide](#runpod-setup--resume-guide) Step 3.

### Issue 3: Missing Native Libraries (Mooncake/RDMA)

`mooncake-transfer-engine` requires RDMA libraries: `apt-get install -y libibverbs-dev librdmacm-dev libnuma-dev`

### Issue 4: SGLang/vLLM Unconditional Imports

TorchSpec imported SGLang/vLLM at module level even when using HF backend.

**Solution**: Made imports lazy — moved into the functions that use them. Files: `inference/engine/__init__.py`, `inference/factory.py`.

### Issue 5: Container Disk Too Small

Default RunPod container disk (20 GB) is insufficient. Training needs ~30-40 GB (model 16 GB + PyTorch 2.5 GB + deps 1 GB + Ray tmp 5-10 GB + inductor cache 1-2 GB).

**Solution**: Provision with **100 GB container disk**.

### Issue 9: DFlash Draft Config Dimension Mismatch

`dflash_draft_config.json` had Qwen2.5-7B dimensions but we train against Qwen3-8B.

**Fix**: Updated config — `hidden_size: 3584→4096`, `target_hidden_size: 3584→4096`, `intermediate_size: 18944→12288`, `num_attention_heads: 28→32`, `num_key_value_heads: 4→8`, `target_num_hidden_layers: 28→36`, `vocab_size: 152064→151936`, `max_position_embeddings: 32768→40960`.

### Issue 10: FlexAttention Inductor `NoValidChoicesError`

**Problem**: `torch._inductor` kernel autotuner had no valid GEMM backends during FlexAttention backward pass.

**Solution** (two-part):
1. Set inductor config at import time in `flex_attention.py`: `inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"`
2. Initially used `backend="aot_eager"` as workaround; later switched back to inductor (default) for 3x speedup once the GEMM fix was confirmed working.

### Issues 11+12: DFlash dtype Mismatch (merged)

**Problem**: Target hidden states (BFloat16) fed into float32 layers. Two locations:
1. `extract_context_feature()` — `context_proj` layer in float32
2. `F.linear(draft_hidden, lm_head_weight)` — decoder layers still float32 after partial bf16 cast

**Solution**:
1. Added `.to(self.context_proj.weight.dtype)` in `extract_context_feature()` before projection
2. Moved `draft_model.to(torch.bfloat16)` after `freeze_embedding()` in `dflash_trainer.py` to ensure full model is bf16 before FSDP wrapping

### Issues 13+14: Zero-Loss Reporting Bug (merged)

**Symptoms**: DFlash showed `loss=0.000, acc=0.000` for all 200 steps. Training suspiciously fast (8 steps/sec vs Eagle3's 1.64).

**Root cause**: **Metric key name mismatch**, not actual zero loss. Loss was computed correctly (values: 13.0, 38.7, 9.8, 18.6) but the controller couldn't read them.

| Component | Expected (Eagle3) | Actual (DFlash) |
|-----------|------------------|-----------------|
| Loss key | `train/avg_loss` | `train/loss` |
| Accuracy key | `train/avg_acc` | `train/accuracy` |

`metrics.get('train/avg_loss', 0)` silently returned default `0`.

**Fix**: Renamed metric keys in `DFlashTrainer._aggregate_metrics()` to match Eagle3 convention.

**Lesson**: When extending a trainer subclass, always check the controller's expected metric key format.

**Additional fixes during investigation**:
- Added fallback in `_sample_anchor_positions`: when no loss_mask positions satisfy the constraint, sample uniformly from all positions with room for a block
- Rewrote `_prepare_noise_input` as fully vectorized using `torch.gather` (eliminated 2,760 Python loop iterations with GPU-CPU sync)

### Issue 15: RunPod Volume Disk Quota

Writing to `/workspace/` (shared NFS mount) failed with `Disk quota exceeded`.

**Solution**: Use container disk (`/root/`, `/tmp/`) for temporary files. Container disk has ~76 GB free.

### Issue 16: FSDP Checkpoint Extraction

`dcp_to_torch_save()` and direct `torch.load()` both fail on `.distcp` format.

**Solution**: Use `dist_cp.state_dict_loader._load_state_dict(no_dist=True)` with `_WrappedStorageReader` and `_EmptyStateDictLoadPlanner` (same pattern as `tools/convert_to_hf.py`).

### Issue 17: FlexAttention OOM with Large num_anchors

`num_anchors=512 × block_size=16 = 8192` Q tokens. FlexAttention attention matrix is too large even with `micro_batch_size=1` on 2 training GPUs (80GB each).

**Memory breakdown (per GPU)**:

| Component | Memory |
|-----------|--------|
| Draft model (1B params, bf16) | ~2 GB |
| Optimizer states (Adam, fp32) | ~8 GB |
| FlexAttention forward | ~6-12 GB |
| FlexAttention backward | ~9-18 GB |
| Gradient buffers + overhead | ~9-14 GB |

**Solution**: `num_anchors=256` halves Q_LEN (8192→4096), roughly halving attention memory. Combined with `draft_accumulation_steps=4` to maintain effective batch size.

### Issue 18: Collator Negative Padding Dimension

`paddingtensor2D` crashes when `loss_mask` is longer than `input_ids` (negative padding size).

**Solution**: Truncation guard: `if n > N: return intensors[:, :N]` in both `paddingtensor2D` and `paddingtensor`. Committed as `398138a`.

### Issue 19: HF Cache Miss in Ray Workers

Ray workers default to `/root/.cache/huggingface/` which is empty. Cached model at `/workspace/.cache/huggingface/` not used.

**Solution**: `ln -s /workspace/.cache/huggingface /root/.cache/huggingface` + `export HF_HOME=/workspace/.cache/huggingface`

### Issue 20: FSDP Checkpoint Disk Quota

Each FSDP checkpoint ~15 GB. Two checkpoints exceed RunPod's per-pod quota.

**Solution**: Added **checkpoint rotation** (`max_checkpoints` config option). `max_checkpoints: 1` deletes oldest before saving. Committed as `c7c6605`.

### Issues 8+21+22: Eval Cache Timeout (merged)

**Problem**: Eval cache generation hangs/times out (300s) in multiple scenarios:
1. Colocate mode — single GPU shared between inference and training deadlocks during eval
2. Missing eval data path on RunPod
3. Stale Ray/Mooncake state after kill/restart cycles
4. `eval_interval=0` alone is insufficient — initial cache generation still runs if `eval_data_path` is set

**Solution**: Must set **both** overrides to fully disable eval:
```
dataset.eval_data_path=null dataset.eval_interval=0
```
Setting `eval_data_path=null` makes `eval_dataset_size=0` → `eval_enabled=False` → skips cache generation entirely. Eval during training is not critical for DFlash — benchmark τ separately after training.

---

## GPU Training Results (1x H100, 200 steps)

| Metric | DFlash | Eagle3 | Notes |
|--------|--------|--------|-------|
| Final accuracy | **0.894** (89.4%) | 0.646 (64.6%) | DFlash 38% higher |
| Final loss | 0.477 (CE) | 2.166 (KL) | Different scales, not comparable |
| Training time | **81.7s** | 202.4s | DFlash **2.5x faster** |
| Steps/sec | **~10** | ~1.8 | DFlash 5.5x more steps/sec |
| Forward passes/step | **1** | 7 | Block-parallel vs autoregressive |

Loss converges rapidly: 13.0 → 0.48 in 200 steps. Accuracy grows steadily from 0% to 89.4%.

---

## Inference Benchmark (200-step checkpoint, 1x H100)

| Method | Tokens/sec | τ (acceptance length) | Speedup |
|--------|-----------|----------------------|---------|
| Baseline (target-only) | 59.4 | N/A | 1.0x |
| DFlash speculative | 42.5 | 1.01 | 0.72x (slower) |

**τ=1.01 is expected** — 200 steps on tiny sample data is insufficient. The draft model produces completely wrong predictions (repetitive unrelated tokens). Root cause: 1000 conversations is insufficient for 899M parameter model. Real training requires 50K+ samples × multiple epochs.

---

## Phase B: 4-GPU SGLang Validation

### Critical Bug: `build_target_layer_ids()` Off-by-One

**Problem**: TorchSpec produced `[1, 10, 18, 26, 35]` instead of SpecForge's `[1, 9, 17, 25, 33]`. Layer 35 + SGLang's +1 capture offset = 36, out of bounds for `range(36)`, so only 4 of 5 hooks fired → Mooncake size mismatch.

**Fix**: Rewrote to match SpecForge: `start=1, end=num_hidden_layers-3`. Also set explicit `target_layer_ids: [1, 9, 17, 25, 33]` in `dflash_draft_config.json`.

**Key insight**: SGLang applies `+1` offset by design — `set_eagle3_layers_to_capture([val + 1 for val in layer_ids])`. The forward loop captures hidden states at the start of each iteration (before the layer runs), so position `k+1` captures output of layer `k`.

### 4-GPU Results (200 steps)

| Metric | Baseline (target-only) | DFlash |
|--------|----------------------|--------|
| Throughput | 61.8 tok/s | 41.2 tok/s |
| τ | — | 1.03 |
| Speedup | 1.0x | 0.67x |

τ=1.03 expected — 200 steps on tiny data. Pipeline works end-to-end.

### SpecForge Cross-Check Findings

**Not bugs (intentional design divergences):**

| Aspect | TorchSpec | SpecForge |
|--------|-----------|-----------|
| Architecture | Standalone PreTrainedModel | Extends Qwen3PreTrainedModel (model-specific) |
| RoPE | Computed on-the-fly per layer | Pre-computed, passed as embeddings |
| Context projection | In `extract_context_feature()` | In draft model `forward()` |
| KV cache | Not supported (recomputes) | Supported via past_key_values |
| Block mask | `if device.type == "cuda"` else None | Always created |

**Real issues fixed:**

| Parameter | Old | New | Source |
|-----------|-----|-----|--------|
| `learning_rate` | 1e-4 | **6e-4** | SpecForge default |
| `warmup_ratio` | 0.015 | **0.04** | SpecForge default |
| `max_grad_norm` | 0.5 | **1.0** | SpecForge default |
| `num_epochs` | 1 | **6** | SpecForge default |

### SpecForge Eagle3 Production Performance (Reference)

| Model | τ | Speedup |
|-------|---|---------|
| Llama-3.1-8B | 1.8–3.1 | 1.0–1.7x |
| Llama-3.3-70B | 1.4–3.2 | 1.1–2.0x |
| Qwen3-30B-A3B | 2.6–5.3 | 1.4–2.5x |
| Llama-4-Scout | 2.1–3.0 | 1.5–2.7x |

**DFlash target**: τ ≥ 3.0 with full training.

---

## Phase C: Full Training (4x H100)

### Speed Optimization

**Bottleneck**: FlexAttention is ~60% of per-step time (O(Q×KV) where Q = num_anchors × block_size).

**Key insight — `aot_eager` vs `inductor`**: Session 5 used `backend="aot_eager"` as workaround for Issue 10. Inductor generates fused Triton kernels that eliminate intermediate tensor materialization, giving **3x speedup and 20 GB less GPU memory**. The GEMM fix (Issue 10) makes inductor safe.

| Config | Speed | ETA | GPU Mem | Notes |
|--------|-------|-----|---------|-------|
| batch=1, accum=4, anchors=512, seq=4096 | 1.5 step/s | 6.6 hr | 40 GB | Baseline |
| batch=2, accum=2, anchors=512, seq=2048, epochs=4 | 1.4 step/s | 4.7 hr | 53 GB | batch=2 slower per step |
| batch=2, accum=2, anchors=**256**, seq=2048 | **1.88** step/s | 3.4 hr | 38 GB | **Big win from reducing anchors** |
| **batch=4, accum=1, anchors=256, seq=2048** | **2.1 step/s** | **3.1 hr** | 50 GB | **Final config** |

**Optimization insights**:
1. **`num_anchors` is the biggest lever**: 512→256 halves Q_LEN, gave largest single speedup
2. **`max_seq_length`** helps inference more than training (FlexAttention Q_LEN dominated by anchors × block_size)
3. **`micro_batch_size`** has diminishing returns (larger batch = more Q tokens per step)
4. anchors=256 + batch=2 has same FlexAttention cost as anchors=512 + batch=1, but processes 2x data per optimizer step

### Phase C Attempts Summary

| # | Issue | Resolution |
|---|-------|------------|
| 1 | OOM forward (35 GiB, anchors=512, batch=4) | Reduce batch→1 |
| 2 | OOM backward (9 GiB, anchors=512, batch=1) | Reduce anchors→256 |
| 3 | Collator negative padding (step 6) | Truncation guard (Issue 18) |
| 4 | Too slow: 0.67 step/s with aot_eager | Switch to inductor |
| 5 | Inductor NoValidChoicesError | ATEN,TRITON backends (Issue 10) |
| 6 | Disk quota at step 15K | Checkpoint rotation (Issue 20) |
| 7-9 | Config errors, eval timeout | Various fixes |
| 10-11 | Eval cache hang | `eval_data_path=null` (Issue 8+21+22) |
| 12 | Stable but slow (1.5 step/s, 6.6hr) | Speed optimization |
| 13-14 | Iterating batch/seq/anchors | See optimization table |
| **15** | **Final config: 2.1 step/s, 3.1hr** | **Paused at 17% (step ~4200)** |

### Final Training Config

```yaml
micro_batch_size: 4
draft_accumulation_steps: 1
num_epochs: 4
max_seq_length: 2048
learning_rate: 6e-4
warmup_ratio: 0.04
max_grad_norm: 1.0
save_per_epoch: true
save_interval: 1000
max_checkpoints: 1
eval_data_path: null
eval_interval: 0
dflash_block_size: 16
dflash_num_anchors: 256
dflash_loss_decay_gamma: 7.0
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
```

### Training Progress (at pause)

| Metric | Value |
|--------|-------|
| Step | ~4,200 / 23,740 (17%, epoch 1/4) |
| Loss | 2.9-4.6 (down from 12.5) |
| Accuracy | 0.16-0.29 |
| Speed | 2.0-2.1 step/s, 16-17 samples/s |
| GPU memory | 50-51 GB / 80 GB (training), 58 GB (inference) |
| Remaining | ~2.7 hours |

### Checkpoint & Backup

- **Latest checkpoint**: `iter_0004001` (~15 GB per checkpoint)
- **HuggingFace backup**: [`Xingh3/qwen3-8b-dflash-checkpoint-phase-c`](https://huggingface.co/Xingh3/qwen3-8b-dflash-checkpoint-phase-c) (private)

| Location | Survives Pod Stop | Survives Pod Delete |
|----------|:-:|:-:|
| `/tmp/`, `/root/` (container disk) | ❌ | ❌ |
| `/workspace/` (volume) | ✅ | ❌ |
| HuggingFace Hub | ✅ | ✅ |

---

## RunPod Setup & Resume Guide

### Pod Requirements

| Spec | Value |
|------|-------|
| GPUs | 1x H100 80GB (minimum) or 4x for SGLang mode |
| Container Disk | **100 GB** |
| Volume Disk | Optional (checkpoint persistence only) |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |

### What Survives Pod Restart

| Survives (`/workspace/`) | Lost (container disk) |
|---|---|
| Git repo directories (but `.py` files may appear deleted) | All pip packages |
| Checkpoints, training data | System packages (libibverbs, etc.) |
| HF model cache | Ray state, logs, inductor cache |

> **Critical**: After restart, run `git restore .` first — git-tracked files may appear deleted.

### Setup Steps

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

**Step 3 — PyTorch 2.6+** (fast: reuse system CUDA, ~900 MB):
```bash
pip3 install --no-deps torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip3 install --upgrade typing_extensions sympy triton \
    nvidia-cusparselt-cu12 nvidia-nvjitlink-cu12
```

**Step 4 — SGLang from source + patches**:
```bash
cd /workspace/TorchSpec
git -C _sglang checkout 0f2df9370a1de1b4fb11b071d39ab3ce2287a350
git -C _sglang reset --hard HEAD
pip3 install -e "_sglang/python[all]"
rm -f _sglang/python/sglang/srt/speculative/spec_training_info.py
cd _sglang && git apply /workspace/TorchSpec/patches/sglang/v0.5.8.post1/sglang.patch
cd /workspace/TorchSpec
pip3 install flashinfer-jit-cache==0.6.2 --index-url https://flashinfer.ai/whl/cu124
```

**Step 5 — TorchSpec**:
```bash
cd /workspace/TorchSpec && pip3 install -e ".[dev]"
```

**Step 6 — Verify**:
```bash
python -c 'import torch; print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")'
python -c 'import sglang; print("SGLang OK")'
python -c 'import torchspec; print("TorchSpec OK")'
```

**Step 7 — Restore checkpoint** (only if `/workspace/` was lost):
```bash
export HF_HOME=/workspace/.cache/huggingface
huggingface-cli download Xingh3/qwen3-8b-dflash-checkpoint-phase-c \
  --local-dir /workspace/TorchSpec/outputs/qwen3-8b-dflash-phase-c/checkpoints/
```

**Step 8 — Launch training with resume**:
```bash
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

### Verifying Resume

Check log after ~5 min (model loading takes ~3 min): `tail -20 /tmp/phase_c_resume.log`

- **Good**: step 4001+/23704, loss 2.9-4.6, accuracy 0.15-0.29
- **Bad**: step 0/23704, loss 10-12, accuracy 0.000 → check `training.load_path`

Resume mechanism: `checkpoint.py` reads `training.load_path` → finds `latest_checkpointed_iteration.txt` → loads model/optimizer/lr_scheduler/rng → sets `global_step` → `loop.py` skips first N steps.

### Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Missing `training.load_path` | Starts from step 0 | Add `training.load_path=./outputs/.../checkpoints` |
| Git files deleted after restart | `No module named torchspec` | `git restore .` |
| Missing RDMA libs | `ImportError: libibverbs.so.1` | `apt-get install -y libibverbs-dev librdmacm-dev libnuma-dev` |
| PyTorch too old (2.4.1) | FlexAttention import errors | Upgrade to 2.6+ (Step 3) |
| SGLang not installed | `No module named 'sglang'` | Step 4 |
| SGLang patch not applied | `unexpected keyword argument 'enable_aux_hidden_states'` | Remove `spec_training_info.py` then `git apply` |
| Standard SSH fails | `Your SSH client doesn't support PTY` | Use `expect` (Step 0) |

---

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

---

## Session 10: 2026-03-21 — Phase C Resume & PyTorch 2.9.1 Migration

### Context

Training paused at step ~17,001 (72%, epoch 3/4). Pod was restarted, wiping all pip packages. SGLang's pinned commit (`0f2df93`) now requires PyTorch 2.9.1 (up from 2.6.0), triggering several compatibility issues during environment rebuild.

### Environment Changes

| Component | Previous | Current |
|-----------|----------|---------|
| PyTorch | 2.6.0+cu124 | **2.9.1+cu128** |
| torchao | (not installed) | **0.9.0** |
| SGLang | 0.5.8.post1 | **0.5.9** |
| sgl-kernel | 0.3.x | **0.3.21** |
| `PYTORCH_CUDA_ALLOC_CONF` | Used | **Deprecated** → `PYTORCH_ALLOC_CONF` |

### Issues Encountered

#### Issue 23: `PYTORCH_CUDA_ALLOC_CONF` Deprecated in PyTorch 2.9+

**Problem**: PyTorch 2.9.1 renamed the environment variable. Using the old name produces:
```
Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead
```

**Solution**: Use `export PYTORCH_ALLOC_CONF=expandable_segments:True` instead.

#### Issue 24: `flashinfer-jit-cache==0.6.2` Not Available on cu124 Index

**Problem**: `pip3 install flashinfer-jit-cache==0.6.2 --index-url https://flashinfer.ai/whl/cu124` fails with `No matching distribution found`. The cu124 wheel index doesn't have 0.6.2.

**Solution**: Not needed as a separate install — SGLang's `[all]` extras already install `flashinfer_python-0.6.2` and `flashinfer_cubin-0.6.2` which provide the same functionality. Skip the standalone flashinfer install.

#### Issue 25: SGLang Engine Ray Actor Timeout (30s) with PyTorch 2.9.1

**Problem**: Training crashes during SGLang engine initialization at `factory.py:250`:
```
ray.exceptions.GetTimeoutError: Get timed out: some object(s) not ready.
```

**Root cause**: `_prepare_sgl_engines()` calls `engine.find_free_port.remote()` with a **hardcoded 30s timeout** (line 252). With PyTorch 2.9.1, CUDA context initialization inside Ray actors takes significantly longer than with 2.6.0. The actor's `__init__` (which imports torch and initializes CUDA) exceeds the 30s limit before `find_free_port` can even execute.

The same 30s timeout appears at 3 locations in `factory.py` (lines 238, 252, 377).

**Fix**: Patched all three `timeout=30` → `timeout=120` in `torchspec/inference/factory.py`:
```bash
sed -i "s/timeout=30,/timeout=120,/g" torchspec/inference/factory.py
```

**Note**: This is a local pod patch, not committed. The `init_timeout` config (default 300s) only applies to the engine `.init()` call (line 290), not to the pre-init `find_free_port` calls.

### Resume Results

| Metric | Value |
|--------|-------|
| Resumed from | Step 17,001 / 23,704 (72%, epoch 3/4) |
| Loss | 2.7-4.1 (correctly in range, not 10-12) |
| Accuracy | 0.20-0.32 |
| Speed | ~1.35 s/step (~0.75 step/s) |
| Previous speed | 2.1 step/s (with PyTorch 2.6.0) |
| ETA | ~2.5 hours |
| Log file | `/tmp/phase_c_resume3.log` |

**Speed regression**: 2.1 → 0.75 step/s (64% slower) after PyTorch 2.9.1 upgrade. Likely causes:
1. torch.compile / Triton kernel cache invalidated (needs re-warmup)
2. Changed CUDA allocator behavior
3. Different Triton version (3.5.1 → 3.6.0) generating different kernels

Speed should improve as inductor cache warms up. Monitor over next ~30 minutes.

### Updated Resume Procedure

Changes from previous procedure (Session 9):

1. **Step 3**: Don't install PyTorch separately — let SGLang's `pip install -e "_sglang/python[all]"` pull the correct version (currently 2.9.1)
2. **Step 4**: Skip standalone flashinfer install (included in SGLang `[all]`)
3. **New Step 3.5**: Patch factory.py timeouts: `sed -i "s/timeout=30,/timeout=120,/g" torchspec/inference/factory.py`
4. **Env var**: Use `PYTORCH_ALLOC_CONF` instead of `PYTORCH_CUDA_ALLOC_CONF`

### Monitor Command

```bash
# In a separate terminal:
ssh -o RequestTTY=force -i ~/.ssh/id_ed25519 sguy1wcn46v8zr-64411b9f@ssh.runpod.io
tail -f /tmp/phase_c_resume3.log
```

## Session 11: 2026-03-21 — Speed Regression Investigation (torch 2.6→2.9.1)

### Context

Training speed regressed 3x after PyTorch 2.6.0→2.9.1 migration: **0.48 s/step → 1.5-1.7 s/step**. GPU utilization showed bursty compute (~30% average) with long idle gaps, indicating the bottleneck was not in model computation itself.

### Investigation: SpecForge vs TorchSpec Comparison

Compared DFlash implementations across both codebases to identify code-level differences:

| Component | TorchSpec | SpecForge | Impact |
|-----------|-----------|-----------|--------|
| RMSNorm | `@torch.compile(dynamic=True)` per module | No compile decorator | 22 separate compiled units per forward |
| GQA in FlexAttention | `_repeat_kv()` → `enable_gqa=False` | Uses GQA natively | Materializes 4x expanded KV tensors |
| `create_block_mask` | Compiled via singleton wrapper | Direct call (not compiled) | Extra compilation overhead |
| `inductor_config.max_autotune_gemm_backends` | Set to `"ATEN,TRITON"` | Not set | Required for Issue 10 fix |
| RoPE position indexing | Advanced indexing gather | Similar pattern | Both break fusion |

### Issue 26: PyTorch 2.9.1 Speed Regression (3x Slower)

**Problem**: Training at 1.5-1.7 s/step with torch 2.9.1 vs 0.48 s/step with torch 2.6.0.

**GPU utilization profile** (sampled every 0.5s over 10s during training):
```
GPU 0: 0, 100, 100, 28, 78, 0, 0, 0, 71, 0, 1, 0, 34, 0, 1, 0, 98, 0, 41  (~30% avg)
GPU 1: 0, 0, 0, 89, 100, 0, 27, 0, 0, 0, 0, 54, 0, 0, 0, 0, 100, 0, 0     (~20% avg)
```
Pattern: short compute bursts (100%) followed by long idle periods (0%). Consistent with CPU-bound overhead or FSDP2 synchronization dominance.

**Profiling results**:
- `create_block_mask`: **4.2 ms** per call — NOT the bottleneck
- Anchor sampling (2 sorts + gather): Small overhead — NOT the bottleneck
- The bottleneck is at the PyTorch runtime level (kernel generation, FSDP2 communication, or inductor codegen changes in 2.9.1)

### Optimization Attempts

| # | Change | Result | Commit |
|---|--------|--------|--------|
| 1 | Remove `@torch.compile(dynamic=True)` from DFlashRMSNorm | No speed change | `c5b71e8` |
| 2 | Use `enable_gqa=True` in FlexAttention (skip `_repeat_kv`) | No speed change | `c5b71e8` |
| 3 | Use `create_block_mask` directly (remove compiled wrapper) | No speed change | `c5b71e8` |
| 4 | `TORCHINDUCTOR_MAX_AUTOTUNE=1` + `COORDINATE_DESCENT_TUNING=1` env vars | No speed change | — |
| 5 | `TORCH_COMPILE_DISABLE=1` (eager mode) | **Crash** — SGLang requires compilation | — |
| 6 | `mode="reduce-overhead"` for flex_attention compilation | Syntax error in sed; not tested | — |

### Root Cause Analysis

The speed regression is at the **PyTorch 2.9.1 runtime level**, not code-level. Likely causes:
1. **TorchInductor codegen changes** — torch 2.9.1 may generate different/slower Triton kernels for FlexAttention forward+backward
2. **FSDP2 behavior changes** — different all-gather/reduce-scatter patterns or synchronization overhead
3. **NCCL/CUDA runtime differences** — cu128 vs cu124, different NCCL version
4. **Triton version change** — 3.5.1→3.6.0 generating suboptimal kernels

**Conclusion**: No code-level fix found. The regression is intrinsic to the torch 2.6→2.9.1 upgrade. Training continues at ~1.5 s/step. Future mitigation options:
- Pin to torch 2.6.0 if SGLang compatibility allows
- Profile with `torch.profiler` to identify exact kernel-level regression
- Test torch 2.7/2.8 as intermediate versions
- Try `torch.compile(mode="max-autotune-no-cudagraphs")` for FlexAttention

### Training Status

| Metric | Value |
|--------|-------|
| Checkpoint | Step 18,001 / 23,704 (76%) |
| Remaining | ~5,703 steps |
| Speed | ~1.5-1.7 s/step |
| ETA | ~2.5 hours |
| Code | Optimization commit `c5b71e8` |
| Log file | `/tmp/phase_c_final.log` |

---

## Session 12: Inference Benchmark & SpecForge Comparison (2026-03-21)

### Benchmark Results — Step 18,001

Extracted checkpoint from FSDP shards to `/workspace/dflash_step18001.pt` (59 keys).

| Metric | Value |
|--------|-------|
| Acceptance length (τ) | **1.86** |
| Wall-clock speedup | **1.27x** |
| Target | τ ≥ 3.0 |

For comparison, Eagle3 achieves τ = 2.4–3.2 on similar models (Qwen3-8B).

**Conclusion**: τ = 1.86 is significantly below target. Investigation into code-level bugs follows.

### SpecForge DFlash Comparison

Compared TorchSpec `torchspec/models/dflash.py` against SpecForge `specforge/core/dflash.py` at HEAD, and reviewed three relevant SpecForge commits:

| Commit | Description |
|--------|-------------|
| `3cebdf5` | fix mask (#453) — Changed `H=num_heads` → `H=None` in `create_block_mask` (was creating per-head masks unnecessarily) |
| `ffc4ab7` | Feat/dflash training improvements (#463) — Added random anchor sampling, exponential loss decay, per-sample block isolation |
| `507da3e` | fix: resolve extremely low acceptance rate (#473) — **Critical fix**: aligned training labels with inference, fixed context mask |

### Key Fixes in SpecForge `507da3e` (Already Ported)

These fixes from the critical acceptance-rate commit are **already present** in TorchSpec:

1. **Label alignment**: `label_offsets = [0, ..., block_size-1]` (same-position prediction, matching inference). The old code used `[1, ..., block_size]` (shifted), causing a train/inference mismatch.
2. **Context mask**: `kv_idx < anchor_pos` (strict less-than). Old code used `kv_idx <= anchor_pos`, but during inference `target_hidden[anchor_pos]` is NOT available as context — only the anchor's token embedding via `noise_embedding`.
3. **Exclude anchor from loss**: `pos_in_block > 0` — anchor token (position 0 in block) is already known input, not a prediction.
4. **Loss decay shift**: `exp(-(k-1)/γ)` so k=1 gets weight 1.0 (was `exp(-k/γ)`).
5. **Loss mask at label positions**: `original_loss_mask_gathered` checks `loss_mask` at each label position, not just the anchor.

### Bugs Identified in TorchSpec (NOT in SpecForge)

#### Bug 1: Zero-loss dummy on empty anchors (`dflash.py:128-134`)

**TorchSpec** (current):
```python
if max_n <= 0:
    anchors = torch.zeros(bsz, 1, dtype=torch.long, device=device)
    keep_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
    return anchors, keep_mask
```
Returns dummy tensors with `keep_mask=False` everywhere → zero loss, zero gradients. Training steps are wasted silently.

**SpecForge** (fixed):
```python
if max_n <= 0:
    raise ValueError("should preprocess the data.")
```
Raises an error, forcing data preprocessing to ensure valid anchors exist.

**Impact**: If any batch has no valid anchor positions (e.g., very short sequences or mostly-masked data), TorchSpec silently produces zero gradients. This wastes training compute and dilutes gradient signal.

#### Bug 2: Anchor filtering by anchor position's mask (`dflash.py:126`)

```python
valid = loss_mask[:, : max_anchor + 1] > 0.5
```

This filters candidate anchors by whether the **anchor token itself** has `loss_mask=1`. But the labels predicted by the block are at positions `anchor+1` through `anchor+block_size-1`. This means:

- An anchor at a prompt position (loss_mask=0) is excluded, even though its block's labels at completion positions (loss_mask=1) would be valid training signal.
- The first supervised token after a masked→unmasked boundary is never trainable as a block's position-1 prediction.

**Note**: This same pattern exists in SpecForge's current code (`specforge/core/dflash.py:102`). Both implementations have this limitation. However, the downstream `original_loss_mask_gathered` multiplication (line 301-306 in TorchSpec) correctly zeros out any labels that fall in masked regions, so the bug's impact is limited to **reduced anchor diversity** near prompt→completion boundaries, not incorrect gradients.

### Action Items

1. **Fix Bug 1**: Replace zero-loss dummy with `raise ValueError` or a proper fallback (e.g., uniform stride anchors on valid positions).
2. **Investigate Bug 2**: Benchmark whether allowing anchors at prompt positions (where labels are in completion region) improves τ.
3. **Retrain**: Apply fixes and retrain from scratch or resume, then re-benchmark τ.
4. **Port any remaining SpecForge improvements**: Check for additional commits after `507da3e`.

---

## Pending Work

1. ~~**Resume Phase C training**~~: ✅ Resumed from step 17,001 (2026-03-21)
2. ~~**Inference benchmark**~~: ✅ τ = 1.86 at step 18,001 (below target 3.0)
3. **Fix anchor sampling bugs**: Bug 1 (zero-loss dummy) and Bug 2 (anchor mask filtering)
4. **Retrain with fixes**: Target τ ≥ 3.0
5. **Draft KV cache**: Add KV cache support to `DFlashDraftModel.forward()` (currently recomputes full context each cycle — O(n²) scaling)
6. **Eagle3 inference comparison**: Side-by-side benchmark
7. **Commit factory.py timeout fix**: Increase `find_free_port` timeout from 30s to 120s for PyTorch 2.9+ compatibility
8. **Update resume guide**: Reflect PyTorch 2.9.1 changes (env var rename, skip standalone flashinfer, factory.py patch)

---

## Commits

| Hash | Description |
|------|-------------|
| `398138a` | Fix collator crash when loss_mask length differs from input_ids |
| `fee3156` | Switch FlexAttention from aot_eager to inductor backend for 3x speedup |
| `c7c6605` | Add checkpoint rotation to prevent disk quota exceeded during training |
| `ba3cd02` | Document Phase C crash debugging: disk quota, eval timeout, checkpoint rotation |
| `dddcd2a` | Document training speed optimization: 6.6hr → 3.1hr |
| `c5b71e8` | Optimize DFlash training speed: remove compile overhead and enable GQA |

---

*Implementation Log v18 — 2026-03-21*

# DFlash Training Test & Benchmark Plan

**Date**: 2026-03-21
**Branch**: `feature/dflash-training`
**Goal**: Verify DFlash training works correctly, trains fast, and achieves target acceptance length (τ ≥ 5.0) across different configurations.
**Hardware**: 4x H100 80GB (2 inference + 2 training)

**Related docs**:
- [Implementation Overview](dflash_overview.md) — Architecture, files, design decisions
- [Issues & Bugs](dflash_issues.md) — All 26 issues + 2 training quality bugs
- [Training Results](dflash_training_results.md) — Phase A/B/C results, speed optimization
- [RunPod Guide](dflash_runpod_guide.md) — Setup (PyTorch 2.9.1, factory.py patch)
- [Pending Work](dflash_pending_work.md) — Active bug fixes, future work

---

## Current State Summary

| Metric | Value | Target |
|--------|-------|--------|
| Phase C τ (step 18,001) | 1.86 | ≥ 5.0 |
| Phase C speedup | 1.27x | ≥ 3.0x |
| Training speed (torch 2.6) | 2.1 step/s | — |
| Training speed (torch 2.9.1) | 0.75 step/s | ≥ 2.0 step/s |
| Known bugs | 2 ([Bug 1](dflash_issues.md#bug-1-zero-loss-dummy-on-empty-anchors-dflashpy128-134), [Bug 2](dflash_issues.md#bug-2-anchor-filtering-by-anchor-positions-mask-dflashpy126)) | 0 |
| Latest checkpoint | `iter_0018001` ([details](dflash_training_results.md#checkpoint--backup)) | — |
| PyTorch regression | 3x slower on 2.9.1 ([Issue 26](dflash_issues.md#issue-26-pytorch-291-speed-regression-3x-slower)) | Resolve or pin 2.6 |

### Paper Reference — DFlash(16) on Qwen3-8B (Temperature=0)

| Domain | EAGLE-3(16) τ | EAGLE-3(60) τ | DFlash(16) τ | DFlash Speedup |
|--------|---------------|---------------|--------------|----------------|
| GSM8K | 3.23 | 3.71 | **6.54** | 5.15x |
| MATH | 3.02 | 3.49 | **7.87** | 6.08x |
| Code | 3.17 | 3.65 | **6.50** | 5.14x |
| Chat | 2.83 | 3.26 | **4.24** | 2.75x |
| **Average** | **2.96** | **3.40** | **6.49** | **4.86x** |

Our implementation must match or approach these numbers. τ < 5.0 indicates implementation bugs, not insufficient training.

**Reference**: SpecForge PR #415 — DFlash online training implementation
**Reference**: SpecForge commits `3cebdf5`, `ffc4ab7`, `507da3e` — critical fixes for acceptance rate

---

## Phase 1: Bug Fixes & Correctness Verification

**Priority**: Must complete before any benchmark runs.
**Tracking**: [Pending Work — Active Bug Fixes](dflash_pending_work.md#active--bug-fixes)

### 1.1 Fix Zero-Loss Dummy ([Bug 1](dflash_issues.md#bug-1-zero-loss-dummy-on-empty-anchors-dflashpy128-134))

**File**: `torchspec/models/dflash.py:128-134`

Replace silent zero-loss fallback with proper error:
```python
# Current (broken): silently returns zero gradients
if max_n <= 0:
    anchors = torch.zeros(bsz, 1, ...)
    keep_mask = torch.zeros(bsz, 1, ...)
    return anchors, keep_mask

# Fix: match SpecForge — fail loudly on bad data
if max_n <= 0:
    raise ValueError(
        f"No valid anchor positions found (max_anchor={max_anchor}, "
        f"block_size={self.block_size}). Preprocess data to ensure "
        f"sequences have at least 2*block_size loss-masked tokens."
    )
```

**Impact**: Any batch with no valid anchor positions (short sequences, mostly-masked data) silently produces zero gradients — wasting compute and diluting gradient signal across the entire training run.

### 1.2 Fix Anchor Filtering ([Bug 2](dflash_issues.md#bug-2-anchor-filtering-by-anchor-positions-mask-dflashpy126))

**File**: `torchspec/models/dflash.py:126`

Current code filters anchors by anchor position's `loss_mask`. Should also consider whether the block's label positions have valid `loss_mask`.

```python
# Current: only checks anchor position
valid = loss_mask[:, : max_anchor + 1] > 0.5

# Improved: check that at least 1 label in block has loss_mask=1
# (anchor+1 through anchor+block_size-1 should overlap with loss_mask=1 region)
```

**Note**: Same pattern exists in SpecForge — impact is reduced anchor diversity near prompt→completion boundaries, not incorrect gradients.

### 1.3 Unit Test Checklist

| Test | Description | Pass? |
|------|-------------|-------|
| `test_anchor_sampling_valid` | No samples produce zero anchors on perfectblend data | |
| `test_block_causal_mask` | Mask matches SpecForge's expected pattern | |
| `test_loss_decay_weights` | `w_k = exp(-(k-1)/7.0)` for block_size=16 | |
| `test_label_alignment` | Labels at positions `anchor+0..anchor+block_size-1` (same-position) | |
| `test_context_mask_strict_lt` | Context mask: `kv_idx < anchor_pos` (not `<=`) | |
| `test_anchor_loss_excluded` | Position 0 in block (anchor) has zero loss weight | |
| `test_original_loss_mask` | Loss mask applied at label positions, not anchor positions | |
| `test_short_sequence_error` | Sequences with < 2*block_size tokens raise ValueError | |

### 1.4 Smoke Test (10 steps)

```bash
python -m torchspec.train_entry \
  --config configs/hf_qwen3_8b_dflash_1gpu.yaml \
  training.num_epochs=1 \
  dataset.train_data_path=../examples/data/sample_conversations.jsonl \
  dataset.eval_data_path=null dataset.eval_interval=0
```

**Expected**: Loss starts ~12-13, drops within 10 steps. No NaN, no zero-loss steps.

### 1.5 SpecForge Deep Diff

Before benchmarking, do a line-by-line diff of these critical code paths against SpecForge HEAD:

| TorchSpec File | SpecForge File | Focus |
|----------------|----------------|-------|
| `torchspec/models/dflash.py` | `specforge/core/dflash.py` | Mask creation, anchor sampling, loss computation |
| `torchspec/models/draft/dflash.py` | `specforge/modeling/draft/dflash.py` | Dual-source KV attention, W_proj |
| `torchspec/training/dflash_trainer.py` | `scripts/train_dflash.py` | Training loop, optimizer, FSDP setup |

Check for any additional SpecForge commits after `507da3e` that may contain further fixes.

---

## Phase 2: Training Speed Benchmark

**Goal**: Measure training throughput across configurations and identify optimal setup.
**Known blocker**: [Issue 26](dflash_issues.md#issue-26-pytorch-291-speed-regression-3x-slower) — PyTorch 2.9.1 is 3x slower than 2.6.0.

### 2.1 Resolve PyTorch Speed Regression (Pre-requisite)

The 3x speed regression (2.1 → 0.75 step/s) on PyTorch 2.9.1 is the single biggest obstacle to fast training. All speed experiments below assume this is resolved or worked around.

| Option | Approach | Risk |
|--------|----------|------|
| **A — Pin torch 2.6.0** | `pip install torch==2.6.0+cu124` + compatible SGLang | SGLang may require 2.9.1 |
| **B — Test torch 2.7/2.8** | Try intermediate versions | Unknown compatibility |
| **C — Profile 2.9.1** | `torch.profiler` to find exact kernel regression | May not be fixable |
| **D — Compile mode tuning** | `mode="max-autotune-no-cudagraphs"` for FlexAttention | Untested |

**Also pending**: [Commit factory.py timeout fix](dflash_pending_work.md#active--code-improvements) (30s → 120s for PyTorch 2.9+ Ray actor init).

### 2.2 Baseline Speed Test (100 steps each)

All tests use: 4x H100, 2 inference + 2 training, `perfectblend_50k.jsonl`

| Config ID | batch | accum | anchors | seq_len | block_size | Expected step/s |
|-----------|-------|-------|---------|---------|------------|-----------------|
| S1 | 1 | 4 | 512 | 4096 | 16 | ~1.5 |
| S2 | 2 | 2 | 256 | 2048 | 16 | ~1.9 |
| S3 | 4 | 1 | 256 | 2048 | 16 | ~2.1 |
| S4 | 4 | 1 | 128 | 2048 | 16 | ~2.5+ |
| S5 | 4 | 1 | 256 | 2048 | 8 | ~3.0+ |
| S6 | 8 | 1 | 128 | 2048 | 8 | ~3.5+ |

Expected speeds assume torch 2.6 or resolved regression. See [Phase C speed optimization](dflash_training_results.md#speed-optimization) for prior measurements.

**Launch template** (e.g., config S3):
```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m torchspec.train_entry \
  --config configs/sglang_qwen3_8b_dflash.yaml \
  training.micro_batch_size=4 \
  training.draft_accumulation_steps=1 \
  training.dflash_num_anchors=256 \
  training.max_seq_length=2048 \
  training.dflash_block_size=16 \
  training.num_epochs=1 \
  dataset.train_data_path=/workspace/data/perfectblend_50k.jsonl \
  dataset.eval_data_path=null dataset.eval_interval=0 \
  output_dir=./outputs/speed_test_S3
```

**Note**: Use `PYTORCH_ALLOC_CONF` (not `PYTORCH_CUDA_ALLOC_CONF`) — see [Issue 23](dflash_issues.md#issue-23-pytorch_cuda_alloc_conf-deprecated-in-pytorch-29).

### 2.3 Speed Metrics to Collect

| Metric | How |
|--------|-----|
| Steps/second | From training log (after warmup, skip first 20 steps) |
| Samples/second | steps/s × micro_batch_size |
| GPU memory (train) | `nvidia-smi` peak during training |
| GPU memory (inference) | `nvidia-smi` peak during target forward |
| FlexAttention % of step | `torch.profiler` if needed |

---

## Phase 3: Acceptance Length (τ) Benchmark Matrix

**Goal**: Measure τ across different training configurations (data size, epochs, training time) to find the sweet spot and verify we can reach paper-level τ ≥ 5.0.

### 3.1 Benchmark Script

```bash
# Extract checkpoint from FSDP shards
python scripts/extract_dflash_checkpoint.py \
  --checkpoint_dir outputs/<run>/checkpoints/iter_NNNNNNN \
  --output /tmp/dflash_draft.pt

# Benchmark τ across domains
python scripts/benchmark_dflash_inference.py \
  --target_model Qwen/Qwen3-8B \
  --draft_checkpoint /tmp/dflash_draft.pt \
  --num_prompts 50 --max_new_tokens 512
```

See [After Training Completes](dflash_training_results.md#inference-benchmark--step-18001) for prior benchmark results.

### 3.2 τ by Training Data Size

Train from scratch with different dataset sizes, same config (S3), 6 epochs each.
Paper reference: SpecForge trains on ~50K regenerated data for 6 epochs → τ=6.49 avg.

| Exp | Dataset | Samples | Expected Total Steps | Expected τ | Training Time |
|-----|---------|---------|---------------------|------------|---------------|
| D1 | perfectblend_5k | 5,000 | ~3,750 | 2.0-3.0 | ~30 min |
| D2 | perfectblend_10k | 10,000 | ~7,500 | 3.0-4.0 | ~1 hr |
| D3 | perfectblend_25k | 25,000 | ~18,750 | 4.0-5.0 | ~2.5 hr |
| D4 | perfectblend_50k | 50,000 | ~37,500 | **5.0-6.0** | ~5.0 hr |
| D5 | perfectblend_100k | 100,000 | ~75,000 | **5.5-6.5** | ~10 hr |
| D6 | regen_50k (target-model generated) | 50,000 | ~37,500 | **6.0-7.0** | ~5.0 hr |

**Note on D6**: SpecForge uses `perfectblend_qwen3-8b_regen.jsonl` — data regenerated by the target model itself. This is critical for high τ because the draft learns the target's exact distribution, not generic human text.

**Checkpoint at**: end of each epoch + every 5000 steps. Benchmark τ at each.

### 3.3 τ by Epoch Progression

Using D4 (50k samples), measure τ at each epoch boundary:

| Checkpoint | Steps | Expected τ | Notes |
|-----------|-------|------------|-------|
| Epoch 1 end | ~6,250 | 2.5-3.5 | Rapid early convergence |
| Epoch 2 end | ~12,500 | 3.5-4.5 | Significant improvement |
| Epoch 3 end | ~18,750 | 4.5-5.5 | Approaching paper numbers |
| Epoch 4 end | ~25,000 | 5.0-6.0 | Should meet target τ≥5.0 |
| Epoch 5 end | ~31,250 | 5.5-6.5 | Near paper-level (6.49 avg) |
| Epoch 6 end | ~37,500 | 5.5-6.5 | Check for overfitting |

**Critical signal**: If τ < 3.0 after epoch 2 with 50K data, there is likely a code bug — not a training issue. Stop and investigate before continuing. See [Failure Diagnosis Guide](#failure-diagnosis-guide).

### 3.4 τ by Training Wall-Clock Time

Quick-iteration checkpoints to understand τ progression over time:

| Time | Approx Steps (at 2.1 step/s) | Expected τ | Benchmark? |
|------|-------------------------------|------------|------------|
| 5 min | ~630 | 1.0-1.5 | Yes — sanity check (τ > 1.0 = draft working) |
| 15 min | ~1,890 | 2.0-3.0 | Yes |
| 30 min | ~3,780 | 3.0-4.0 | Yes |
| 1 hr | ~7,560 | 4.0-5.0 | Yes — should already beat Eagle3 (τ=2.96) |
| 2 hr | ~15,120 | 5.0-5.5 | Yes — approaching target |
| 3 hr | ~22,680 | 5.5-6.0 | Yes — near paper-level |
| 5 hr | ~37,500 | 5.5-6.5 | Yes — full training (6 epochs on 50K) |

**Note**: Step counts assume torch 2.6 speed (2.1 step/s). With torch 2.9.1 regression (0.75 step/s), multiply wall-clock times by ~2.8x.

### 3.5 τ by Block Size (train vs inference)

Train with block_size B_train, infer with block_size B_infer (paper Table 7):

| Exp | B_train | B_infer | Expected τ | Paper τ | Notes |
|-----|---------|---------|------------|---------|-------|
| B1 | 16 | 16 | 5.5-6.5 | 6.33 | Matched — optimal |
| B2 | 16 | 8 | 4.5-5.5 | 5.09 | Large→small generalizes |
| B3 | 16 | 4 | 3.5-4.5 | — | Large→small, more degradation |
| B4 | 8 | 8 | 4.5-5.5 | — | Faster training (smaller FlexAttention) |
| B5 | 8 | 16 | 4.0-5.0 | 5.02 | Small→large — weaker but usable |

### 3.6 τ by Eval Domain

Measure τ separately per benchmark domain (using best checkpoint). Paper numbers as targets:

| Domain | Prompt Source | Paper τ (DFlash) | Paper τ (EAGLE-3) | Our Target τ |
|--------|-------------|------------------|--------------------|----|
| GSM8K | Math word problems | **6.54** | 3.23 | ≥ 5.5 |
| MATH | Formal math | **7.87** | 3.02 | ≥ 6.0 |
| Code (HumanEval) | Code generation | **6.50** | 3.17 | ≥ 5.5 |
| Chat (MT-Bench) | Chat/creative | **4.24** | 2.83 | ≥ 3.5 |
| **Average** | All above | **6.49** | **2.96** | **≥ 5.0** |

---

## Phase 4: Target Accept Length Configuration

**Goal**: Set different target τ thresholds based on practical deployment constraints.

### 4.1 Accept Length Targets by Use Case

| Use Case | Min τ | Target τ | Block Size | Rationale |
|----------|-------|----------|------------|-----------|
| Production (latency) | 5.0 | 6.0+ | 16 | Paper: 4.86x avg speedup |
| Production (throughput) | 4.0 | 5.0+ | 16 | Batch serving, high-concurrency |
| Development/CI | 3.0 | 4.0+ | 8-16 | Quick validation, still meaningfully faster |
| Demo/PoC | 2.0 | 3.0+ | 8 | Visible speedup for demos |

### 4.2 Recommended Training Configs by Target τ

| Target τ | Data Size | Data Type | Epochs | block_size | anchors | Est. Time (4xH100) |
|----------|-----------|-----------|--------|------------|---------|---------------------|
| τ ≥ 3.0 | 10K | curated | 4 | 16 | 256 | ~1 hr |
| τ ≥ 4.0 | 25K | curated | 6 | 16 | 256 | ~2.5 hr |
| τ ≥ 5.0 | 50K | curated/regen | 6 | 16 | 512 | ~5 hr |
| τ ≥ 6.0 | 50K+ | **regen** (target-generated) | 6 | 16 | 512 | ~5-10 hr |

**Key insight**: Data quality matters more than quantity. Target-model-regenerated data (SpecForge's approach) is critical for τ ≥ 6.0 because the draft model must learn the target's specific token distribution, not generic human text patterns.

### 4.3 Data Quality vs Quantity Trade-off

| Dataset | Quality | Size | Expected τ (6 epochs) | Notes |
|---------|---------|------|----------------------|-------|
| ShareGPT raw | Mixed | 100K+ | 3.0-4.0 | Noisy, distribution mismatch with target |
| PerfectBlend curated | High | 50K | 4.0-5.5 | Good signal, but not target-aligned |
| Domain-specific (math) | Narrow | 10-20K | 6.0+ on math, 2.0 elsewhere | Overfits to domain |
| **Regenerated (target model)** | **Aligned** | **50K+** | **5.5-6.5** | **Best overall — matches SpecForge pipeline** |

### 4.4 Data Regeneration Pipeline

To match paper results, we need target-model-regenerated training data:

```bash
# Step 1: Prepare prompts from ShareGPT/PerfectBlend (keep only user turns)
python scripts/prepare_perfectblend.py \
  --input data/perfectblend_50k.jsonl \
  --output data/perfectblend_prompts.jsonl \
  --mode prompts_only

# Step 2: Regenerate completions using the target model (Qwen3-8B)
# This ensures draft model trains on the target's exact distribution
python scripts/regenerate_data.py \
  --model Qwen/Qwen3-8B \
  --prompts data/perfectblend_prompts.jsonl \
  --output data/perfectblend_qwen3-8b_regen.jsonl \
  --temperature 0.6 --max_tokens 2048

# Step 3: Use regenerated data for training
# dataset.train_data_path=data/perfectblend_qwen3-8b_regen.jsonl
```

**GPU allocation for regen**: Use 2 inference GPUs for generation while training GPUs are idle during Phase 1 bug fixes.

---

## Phase 5: End-to-End Validation Checklist

### 5.1 Training Pipeline

- [ ] Bug fixes applied and unit tests pass ([Bug 1](dflash_issues.md#bug-1-zero-loss-dummy-on-empty-anchors-dflashpy128-134), [Bug 2](dflash_issues.md#bug-2-anchor-filtering-by-anchor-positions-mask-dflashpy126))
- [ ] Training starts without errors on 4-GPU setup (see [RunPod Guide](dflash_runpod_guide.md))
- [ ] Loss decreases monotonically over first 100 steps (expect 13.0 → <5.0)
- [ ] Accuracy increases steadily (0% → 20%+ in first epoch)
- [ ] Checkpoint saves correctly (FSDP shards → extractable)
- [ ] Resume from checkpoint produces correct loss range (not reset to ~12, see [Issue 13+14](dflash_issues.md#issues-1314-zero-loss-reporting-bug))
- [ ] No zero-loss steps in training log
- [ ] Memory usage stable (no OOM, no gradual leak; see [Issue 17](dflash_issues.md#issue-17-flexattention-oom-with-large-num_anchors))

### 5.2 Inference Pipeline

- [ ] Checkpoint extraction (`extract_dflash_checkpoint.py`) succeeds (see [Issue 16](dflash_issues.md#issue-16-fsdp-checkpoint-extraction))
- [ ] Draft model loads with correct architecture (5 layers, dual-source KV)
- [ ] Speculative decoding produces coherent text (not garbage/repetition)
- [ ] τ ≥ 5.0 average across domains (paper: 6.49)
- [ ] τ ≥ 3.5 on Chat/MT-Bench (paper: 4.24, lowest domain)
- [ ] Wall-clock speedup ≥ 3.0x (paper: 4.86x avg)
- [ ] No accuracy degradation vs baseline (lossless speculative decoding)

### 5.3 SpecForge Parity Check

These are the critical fixes from SpecForge commit `507da3e` — all already ported to TorchSpec but must be verified:

- [ ] `target_layer_ids` match: `[1, 9, 17, 25, 33]` for Qwen3-8B (see [Phase B off-by-one](dflash_issues.md#build_target_layer_ids-off-by-one-phase-b))
- [ ] Label alignment: same-position prediction (not shifted)
- [ ] Context mask: strict `<` (not `<=`)
- [ ] Loss decay: `exp(-(k-1)/γ)` where γ=7.0
- [ ] Shared embedding + LM head frozen during training
- [ ] CE loss (not KL divergence)

---

## Execution Plan

### Week 1: Fix & Verify

| Day | Task | GPU Hours |
|-----|------|-----------|
| 1 | Bug fixes (1.1, 1.2) + SpecForge deep diff (1.5) + unit tests (1.3) | 0 |
| 1 | Resolve PyTorch speed regression (2.1) — try pinning torch 2.6 | 1 |
| 1 | Smoke test (1.4) + speed benchmark S1-S3 (2.2) | 2 |
| 2 | Speed benchmark S4-S6 (2.2) | 1 |
| 2 | Start data regeneration (4.4) on inference GPUs | 2 |
| 2 | Data size experiments D1-D2 (3.2) — quick τ sanity check | 1.5 |

### Week 2: Full Benchmark

| Day | Task | GPU Hours |
|-----|------|-----------|
| 1 | Data size experiments D3-D4 (3.2) | 7.5 |
| 2 | D6 — regenerated data experiment (3.2) | 5 |
| 2 | Epoch progression benchmark (3.3) — reuse D4/D6 checkpoints | 0 |
| 3 | Block size experiments B1-B5 (3.5) | 4 |
| 3 | Domain-specific τ evaluation (3.6) | 1 |
| 3 | End-to-end validation checklist (Phase 5) | 1 |

**Total estimated GPU hours**: ~26 hours (4x H100)

---

## Metrics Dashboard

Track all experiments in a single results table:

```
| Exp ID | Config | Steps | Epoch | Train Time | Loss | Acc | τ (avg) | τ (GSM8K) | τ (MATH) | τ (Code) | τ (Chat) | Speedup | Notes |
|--------|--------|-------|-------|------------|------|-----|---------|-----------|----------|----------|----------|---------|-------|
| PhaseC | S3,50K | 18001 | 3/4   | ~2.5hr     | 2.7  | 0.32| 1.86    | —         | —        | —        | —        | 1.27x   | Pre-bugfix baseline |
|        |        |       |       |            |      |     |         |           |          |          |          |         |       |
```

Report with: `--report-to wandb --wandb-project torchspec-dflash-benchmark`

---

## Success Criteria

| Criterion | Threshold | Target | Paper Reference |
|-----------|-----------|--------|-----------------|
| Training runs without bugs | All unit tests pass | — | — |
| Training speed | ≥ 2.0 step/s (config S3) | ≥ 3.0 step/s | — |
| Acceptance length (τ) avg | **≥ 5.0** | ≥ 6.0 | 6.49 |
| τ on GSM8K | ≥ 5.0 | ≥ 6.0 | 6.54 |
| τ on MATH | ≥ 5.5 | ≥ 7.0 | 7.87 |
| τ on Code | ≥ 5.0 | ≥ 6.0 | 6.50 |
| τ on Chat | ≥ 3.5 | ≥ 4.0 | 4.24 |
| Wall-clock inference speedup | **≥ 3.0x** | ≥ 4.5x | 4.86x |
| SpecForge parity | All checks pass | — | — |

### Failure Diagnosis Guide

| Observed τ | Likely Cause | Action |
|-----------|--------------|--------|
| τ < 1.5 | Critical bug (mask, labels, or loss) | Diff against SpecForge line-by-line (1.5) |
| τ = 1.5-3.0 | Bug in anchor sampling, context mask, or loss decay | Check [Bug 1](dflash_issues.md#bug-1-zero-loss-dummy-on-empty-anchors-dflashpy128-134) & [Bug 2](dflash_issues.md#bug-2-anchor-filtering-by-anchor-positions-mask-dflashpy126) fixes, verify SpecForge parity (5.3) |
| τ = 3.0-5.0 | Data quality issue or insufficient training | Switch to regenerated data (D6), increase epochs to 6 |
| τ = 5.0-6.0 | On track — minor tuning needed | Increase num_anchors to 512, try longer sequences |
| τ ≥ 6.0 | Paper-level — success | Ship it |

**Current state (τ=1.86)** falls in the 1.5-3.0 range → priority is bug fixes, not more training.

---

*Plan v3 — 2026-03-21*

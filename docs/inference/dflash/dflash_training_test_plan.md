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
| Training speed (torch 2.9.1, post-fix) | ~5 step/s (colocate 1GPU) | ≥ 2.0 step/s |
| Known bugs | 0 (Bug 1 fixed, Bug 2 matches SpecForge) | 0 |
| Latest checkpoint | `iter_0018001` ([details](dflash_training_results.md#checkpoint--backup)) | — |
| PyTorch regression | **Resolved** — `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` fixes it | ✓ |
| Environment | torch 2.9.1 + sglang 0.5.9 + 4x H100 | — |

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

### 1.1 Fix Zero-Loss Dummy ([Bug 1](dflash_issues.md#bug-1-zero-loss-dummy-on-empty-anchors-dflashpy128-134)) — DONE

**File**: `torchspec/models/dflash.py:130-134`

**Fix applied** (commit `f3311e4`): Replaced silent zero-loss fallback with `raise ValueError`, matching SpecForge exactly.

**Additional fix**: Plumbed `min_loss_tokens` config parameter through the data pipeline (`DatasetConfig` → `load_conversation_dataset` → `_init_tokenize_worker` → `preprocess_conversations`) so sequences with < 2×block_size supervised tokens are filtered at data loading time. Added `min_loss_tokens: 32` to both DFlash YAML configs. This is the data-level guard that SpecForge uses (line 217-224 of `scripts/train_dflash.py`).

### 1.2 Fix Anchor Filtering ([Bug 2](dflash_issues.md#bug-2-anchor-filtering-by-anchor-positions-mask-dflashpy126)) — SKIPPED (matches SpecForge)

Same pattern exists in SpecForge (`specforge/core/dflash.py:102`). Impact is reduced anchor diversity near prompt→completion boundaries, not incorrect gradients. Priority is SpecForge parity first; can revisit after reaching τ ≥ 5.0.

### 1.3 Unit Test Checklist — ALL PASS (54/54 on pod, 2026-03-22)

| Test | Description | Pass? |
|------|-------------|-------|
| `test_anchor_sampling_valid` | No samples produce zero anchors on perfectblend data | ✅ (`test_basic_sampling`, `test_respects_loss_mask`) |
| `test_block_causal_mask` | Mask matches SpecForge's expected pattern | ✅ (`test_block_internal_visibility`) |
| `test_loss_decay_weights` | `w_k = exp(-(k-1)/7.0)` for block_size=16 | ✅ (new, commit `f3311e4`) |
| `test_label_alignment` | Labels at positions `anchor+0..anchor+block_size-1` (same-position) | ✅ (`test_label_alignment_same_position`, new) |
| `test_context_mask_strict_lt` | Context mask: `kv_idx < anchor_pos` (not `<=`) | ✅ (`test_context_causal`) |
| `test_anchor_loss_excluded` | Position 0 in block (anchor) has zero loss weight | ✅ (new, commit `f3311e4`) |
| `test_original_loss_mask` | Loss mask applied at label positions, not anchor positions | ✅ (`test_loss_mask_at_label_positions`) |
| `test_short_sequence_error` | Sequences with < 2*block_size tokens raise ValueError | ✅ (updated, commit `f3311e4`) |

### 1.4 Smoke Test (30 steps) — PASS (2026-03-22)

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
python -m torchspec.train_entry \
  --config configs/hf_qwen3_8b_dflash_1gpu.yaml \
  training.num_epochs=1 \
  dataset.train_data_path=/workspace/data/perfectblend_50k.jsonl \
  dataset.eval_data_path=null dataset.eval_interval=0
```

**Results** (torch 2.9.1, sglang 0.5.9, 1x H100 colocate):

| Step | Loss | Accuracy | Speed |
|------|------|----------|-------|
| 1 | 12.70 | 0.000 | warmup |
| 10 | 11.08 | 0.000 | ~1.2 step/s |
| 20 | 9.30 | 0.091 | ~3.6 step/s |
| 30 | 8.55 | 0.065 | ~5.0 step/s |

- ✅ Loss starts ~12.7, drops monotonically
- ✅ No NaN, no zero-loss steps (Bug 1 fix working)
- ✅ Accuracy starting to increase by step 20
- ✅ Speed ~5 step/s after warmup (exceeds target)

### 1.5 SpecForge Deep Diff — DONE (2026-03-22)

Line-by-line comparison of all three critical file pairs completed:

| TorchSpec File | SpecForge File | Focus | Result |
|----------------|----------------|-------|--------|
| `torchspec/models/dflash.py` | `specforge/core/dflash.py` | Mask creation, anchor sampling, loss computation | ✅ Match |
| `torchspec/models/draft/dflash.py` | `specforge/modeling/draft/dflash.py` | Dual-source KV attention, W_proj | ✅ Match |
| `torchspec/training/dflash_trainer.py` | `scripts/train_dflash.py` | Training loop, optimizer, FSDP setup | ✅ Match (FSDP2 vs FSDP1 is intentional) |

**Verified**: No SpecForge commits after `507da3e` modified dflash files. All critical code paths match: mask creation, anchor sampling, label alignment (same-position), loss computation (CE + decay), context projection, dual-source KV attention, `build_target_layer_ids`, RoPE application.

**Only gap found**: `min_loss_tokens` data filtering — SpecForge filters at data load time but TorchSpec wasn't plumbing the parameter. Fixed in commit `f3311e4`.

---

## Phase 2: Training Speed Benchmark

**Goal**: Measure training throughput across configurations and identify optimal setup.

### 2.1 Resolve PyTorch Speed Regression — RESOLVED (2026-03-22)

The 3x speed regression (2.1 → 0.75 step/s) on PyTorch 2.9.1 was caused by TorchInductor GEMM backend selection. Fixed with:

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
```

This resolves both the speed regression AND the `NoValidChoicesError` in FlexAttention backward pass. Training speed is now ~5 step/s in colocate 1-GPU mode (exceeds torch 2.6 baseline of 2.1 step/s).

**Environment**: torch 2.9.1 + sglang 0.5.9 (latest, matching SpecForge commit `961ca7c`).

### 2.2 Baseline Speed Test — DONE (2026-03-22)

**Setup**: 4x H100 80GB, 2 training + 1 inference + 1 idle, SGLang + Mooncake, `perfectblend_50k.jsonl`, block_size=16, seq=2048, 50 steps each.

| Config | batch | accum | anchors | step/s | GPU Mem (train) | GPU Mem (infer) | Notes |
|--------|-------|-------|---------|--------|-----------------|-----------------|-------|
| S1 | 1 | 4 | 512 | **1.05** | 13.4 GB | 57.3 GB | Baseline |
| S2 | 2 | 2 | 512 | **1.0** | ~13 GB | 57.3 GB | Same speed |
| S3 | 4 | 1 | 512 | **OOM** | — | — | Backward alloc 9.27 GB failed |
| S3b | 2 | 2 | 256 | **1.0** | ~13 GB | 57.3 GB | Anchors don't matter |

**Key finding: All configs converge to ~1.0 step/s.** Instrumented timing breakdown (steps 10-30):
- **GPU compute: ~0.68s (70%)** — forward+backward+optimizer (FlexAttention + FSDP allreduce)
- **Data pipeline: ~0.40s (30%)** — Mooncake KV queue fetch (overlaps partially with compute)
- **Ray dispatch: ~0.07s (7%)** — negligible
- Reducing anchors 512→256 yields identical step/s — FlexAttention is not the dominant compute cost

**Comparison**:
- 1-GPU colocate (Phase D smoke test): **~5 step/s** — no Mooncake + no multi-GPU FSDP overhead
- 4-GPU SGLang mode: **~1.0 step/s** — compute and data pipeline both contribute
- Training time estimate for 50K×6 epochs: **~10 hours** at 1.0 step/s (37,500 optimizer steps)

**Detailed analysis**: See [Speed Investigation Summary](dflash_training_results.md#phase-e-speed-benchmark) for raw timing data.

### 2.3 Speed Improvement Recommendations

| Priority | Approach | Expected Impact | Rationale |
|----------|----------|-----------------|-----------|
| **P0** | Profile compute breakdown (forward vs backward vs optimizer vs FSDP allreduce) | 1.5-2x | Compute is 70% — biggest lever |
| **P0** | Bypass Mooncake for same-node transfer (use NCCL/shared memory) | 1.3-1.5x | data_time is 30% of step |
| **P1** | Increase `max_concurrent_batches` to better overlap inference+training | Up to 1.5x | Pipeline already partially overlaps |
| **P1** | FSDP CPU offload to fit batch=4+anchors=512 | May help if compute scales sublinearly | Needs testing |
| **P2** | Colocate mode if memory allows | 3-5x (eliminates data_time + FSDP overhead) | OOM risk with 8B target + draft on same GPU |

### 2.4 Speed Analysis Summary (for project owner)

**Setup**: 4x H100 80GB, 2 training + 1 inference (SGLang + Mooncake), torch 2.9.1 + sglang 0.5.9. DFlash config: block_size=16, anchors=512, seq_len=2048, batch=2, accum=2.

**Result**: All configurations converge to **~1.0 step/s**. Full training (50K × 6 epochs) = **~10 hours**.

**Instrumented timing breakdown (averaged over steps 10-30):**

| Component | Time (s) | % of Step |
|-----------|----------|-----------|
| **GPU compute** (forward+backward+optimizer+FSDP allreduce) | **0.68** | **~70%** |
| **Data pipeline** (Mooncake KV queue fetch) | **0.40** | **~30%** |
| **Ray dispatch** | **0.07** | **~7%** |
| **Total step time** | **~0.95** | — |

Note: data and compute partially overlap (async pipeline), so percentages sum to >100%.

**Key observations:**
1. **Compute dominates at 70%** — reducing anchors 512→256 doesn't change speed, so FlexAttention is NOT the bottleneck within compute. FSDP allreduce or optimizer step may dominate.
2. **Data pipeline is 30%** — Mooncake KV transfer ~400ms/step. For same-node setups, NCCL/shared memory should be <10ms.
3. **1-GPU colocate is 5x faster** (~5 step/s) — eliminates both Mooncake and multi-GPU FSDP overhead.
4. **batch=4 + anchors=512 OOMs** — backward needs 9.27 GB extra.

**Questions for discussion:**
1. Why is compute 0.68s for a ~1B draft model? Is FSDP allreduce the dominant cost? Can we use REPLICATE for a model this small?
2. Is Mooncake needed for single-node? NCCL p2p should be <10ms for this data volume.
3. Can `fsdp_cpu_offload` or reduced FSDP sharding help fit batch=4+anchors=512?

### 2.5 Speed Optimization Tests (Phase 2 continued)

#### 2.5.1 Compute Sub-Breakdown (P0) — COMPLETED

Instrumented `dflash_trainer._train_step()` and `trainer._train_core_from_queue()` with `torch.cuda.Event` timing. Config: batch=2, accum=2, anchors=512, seq=2048, 30 steps.

**Result**: Compute is now ~0.26s (was 0.68s in earlier run — likely torch.compile warmup effect).

| Component | Time (ms) | % of Compute | Notes |
|-----------|----------|--------------|-------|
| **Backward** | **~140** | **54%** | Dominates — includes gradient allreduce |
| **Forward** | **~82** | **31%** | FlexAttention + CE loss |
| **Optimizer** | **~41** | **16%** | FSDP optimizer step, very consistent |

**Raw TIMING data (steps 2-30):**

| Step | step (s) | data (s) | compute (s) | fwd (s) | bwd (s) | opt (s) |
|------|----------|----------|-------------|---------|---------|---------|
| 2 | 0.391 | 0.159 | 0.292 | 0.102 | 0.148 | 0.041 |
| 5 | 0.422 | 0.199 | 0.279 | 0.105 | 0.132 | 0.041 |
| 10 | 0.412 | 0.212 | 0.261 | 0.079 | 0.141 | 0.041 |
| 15 | 0.359 | 0.172 | 0.261 | 0.066 | 0.154 | 0.041 |
| 20 | 0.400 | 0.235 | 0.233 | 0.073 | 0.120 | 0.041 |
| 25 | 0.413 | 0.221 | 0.267 | 0.068 | 0.158 | 0.041 |
| 30 | 0.348 | 0.149 | 0.258 | 0.085 | 0.132 | 0.041 |

**Speed**: ~2.5 step/s steady-state (step_time ~0.4s). This is 2.5x faster than the 1.0 step/s measured in Section 2.2.

**Key insight**: Backward (54%) is the biggest lever within compute. Optimizer is small and constant (41ms). Forward is moderate. FSDP gradient allreduce happens during backward — this likely explains why backward > forward for a ~1B model.

#### 2.5.1b Extended Steady-State (200 steps)

Verified speed stability over 200 steps (2 epochs). **Result: stable ~2.5 step/s** with no degradation.

| Step Range | Avg step (s) | Avg data (s) | Avg compute (s) | Avg fwd (s) | Avg bwd (s) | Avg opt (s) |
|-----------|-------------|-------------|----------------|------------|------------|------------|
| 2-30 | 0.370 | 0.160 | 0.263 | 0.085 | 0.138 | 0.041 |
| 30-100 | 0.360 | 0.162 | 0.255 | 0.077 | 0.137 | 0.041 |
| 100-200 | 0.356 | 0.162 | 0.255 | 0.076 | 0.138 | 0.041 |

Training converged well: loss 12.35→0.12, accuracy 0%→96.2% in 200 steps.

#### 2.5.2 Bypass Mooncake for Same-Node (P0) — NOT FEASIBLE (config-only)

**Finding**: No built-in option to bypass Mooncake for same-node transfers. Code investigation shows:
- `EagleMooncakeStore` always uses Mooncake protocol (TCP or RDMA) regardless of node topology
- `enable_gpu_direct=True` tested → CUDA assert error (RunPod lacks RDMA hardware)
- Implementing bypass requires code changes: shared memory or NCCL p2p transport

**Available alternatives**:
- `mooncake.protocol="rdma"` — needs RDMA NIC (not available on this pod)
- Colocate mode (`training.colocate=True`) — eliminates transfer entirely but OOMs with 8B model

**Recommendation**: Data pipeline (~160ms) is already <50% of step_time. Focus optimization on compute (backward dominates at 54% of compute) or training config (batch size, FSDP sharding).

#### 2.5.3 max_concurrent_batches > 1 (P1) — NO IMPROVEMENT

Tested `training.max_concurrent_batches=2`. Result: **same ~2.5 step/s** as baseline.

| Config | step/s | step_time (s) | data (s) | compute (s) |
|--------|--------|--------------|----------|-------------|
| concurrent=1 (baseline) | ~2.5 | 0.37 | 0.16 | 0.26 |
| concurrent=2 | ~2.5 | 0.37 | 0.16 | 0.26 |

Inference throughput increased (I=48-52 vs I=30-33 samples/s), but training was not bottlenecked on inference supply — data fetcher always has samples ready.

#### 2.5.4 Compute Speed Optimizations (no_sync + compile + bf16 reduce)

Tested three optimizations to reduce compute time:

| Optimization | Result |
|-------------|--------|
| **no_sync() + bf16 reduce** | ~2.7 step/s (**+8%** vs 2.5 baseline) |
| **Full model torch.compile** | **Worse** — recompilation overhead from dynamic shapes (variable anchor counts, seq lengths). Steps 2-30 at 1-3s/step. |

**no_sync + bf16 reduce detailed comparison (steps 5-50):**

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| step_time | 0.370s | 0.353s | -5% |
| backward | 138ms | 133ms | -4% |
| forward | 78ms | 81ms | +4% |
| optimizer | 41ms | 41ms | 0% |
| data | 162ms | 147ms | -9% |

**Conclusion**: no_sync saves ~5ms/step on backward (skips 1 allreduce per step with accum=2). bf16 reduce gives minor comm savings. Full model compile is not viable due to dynamic tensor shapes in DFlash (anchor sampling, variable-length sequences). Total improvement is modest (+8%) — the pipeline is already well-optimized.

**Updated training time**: 37,500 steps / 2.7 step/s = **~3.9 hours** (was 4.2h).

#### 2.5.5 FSDP Strategy: FULL_SHARD vs REPLICATE

Tested FULL_SHARD (parameter sharding) vs REPLICATE (DDP-like gradient allreduce) with bf16 reduce enabled.

**Test 1 — FULL_SHARD, 2 training GPUs, bf16 reduce (200 steps):**

| Metric | REPLICATE (baseline) | FULL_SHARD | Change |
|--------|---------------------|------------|--------|
| step_time | 0.370s | **0.350s** | -5% |
| step/s | 2.5 | **2.9** | **+16%** |
| forward | ~82ms | ~100ms | +22% |
| backward | ~138ms | ~134ms | -3% |
| optimizer | 41ms | **22ms** | **-46%** |
| data | 162ms | 145ms | -10% |
| 200-step wall time | 244.5s | **220.5s** | **-10%** |
| Final accuracy | 96.2% | **97.1%** | +0.9% |
| Final loss | 0.12 | **0.101** | -16% |

**Key insight**: FULL_SHARD halves optimizer time (41ms→22ms) because each GPU only updates its local shard. Forward is slightly slower (+22%) due to parameter gather, but the optimizer savings more than compensate.

#### 2.5.6 Scaling: 3 Training GPUs + 1 Inference GPU

**Test 2 — REPLICATE, 3 training GPUs, bf16 reduce (200 steps):**

| Metric | 2-GPU REPLICATE | 3-GPU REPLICATE | Change |
|--------|----------------|-----------------|--------|
| step_time | 0.370s | **0.450s** | +22% |
| step/s | 2.5 | **2.2** | -12% |
| forward | ~82ms | ~85ms | +4% |
| backward | ~138ms | ~140ms | +1% |
| optimizer | 41ms | 41ms | 0% |
| data | 162ms | **240ms** | **+48%** |
| 200-step wall time | 244.5s | **285.4s** | +17% |
| Final accuracy | 96.2% | **96.3%** | 0% |
| Final loss | 0.12 | **0.126** | +5% |

**Key finding**: 3-GPU is slower per-step due to higher data_time (3-way Mooncake fetch: 240ms vs 160ms for 2-way). However, 3-way DP processes 50% more samples per optimizer step, so **fewer total steps** are needed (37,500 / 1.5 = 25,000 steps). Net training time: 25,000 / 2.2 = **~3.2 hours** vs 37,500 / 2.5 = **~4.2 hours** for 2-GPU.

#### 2.5.7 Combined: FULL_SHARD + 3 Training GPUs

**Test 3 — FULL_SHARD, 3 training GPUs, max_concurrent_batches=3, bf16 reduce (200 steps):**

| Metric | Value | vs Test 1 (FULL_SHARD 2GPU) | vs Test 2 (3GPU REPLICATE) |
|--------|-------|-----------------------------|----------------------------|
| step_time | **0.440s** | +26% | -2% |
| step/s | **2.3** | -21% | +5% |
| forward | ~115ms | +15% | +35% |
| backward | ~136ms | +1% | -3% |
| optimizer | **16-22ms** | -27% | **-61%** |
| data | ~220ms | +52% | -8% |
| 200-step wall time | **255.4s** | +16% | **-11%** |

**Key finding**: FULL_SHARD + 3 GPU gives the lowest optimizer time (16-22ms) but data pipeline remains the bottleneck. Effective training time: 25,000 / 2.3 = **~3.0 hours**.

#### 2.5.8 Configuration Comparison Summary

| Config | GPUs | FSDP | step/s | step_time | opt_time | data_time | Eff. Steps | Est. Train Time | Acc |
|--------|------|------|--------|-----------|----------|-----------|-----------|-----------------|-----|
| Baseline | 2 train | REPLICATE | 2.5 | 0.370s | 41ms | 162ms | 37,500 | 4.2 hr | 96.2% |
| + no_sync + bf16 | 2 train | REPLICATE | 2.7 | 0.353s | 41ms | 147ms | 37,500 | 3.9 hr | ~96% |
| **Test 1: FULL_SHARD** | 2 train | FULL_SHARD | **2.9** | 0.350s | **22ms** | 145ms | 37,500 | **3.6 hr** | **97.1%** |
| Test 2: 3 GPU | 3 train | REPLICATE | 2.2 | 0.450s | 41ms | 240ms | 25,000 | 3.2 hr | 96.3% |
| Test 3: FULL_SHARD+3GPU | 3 train | FULL_SHARD | 2.3 | 0.440s | 16-22ms | 220ms | 25,000 | **3.0 hr** | ~96% |

#### 2.5.9 Scaling Inference: 2 Training + 2 Inference GPUs

**Test 4 — FULL_SHARD, 2 train GPU, 2 inference GPU, max_concurrent_batches=2, bf16 reduce (200 steps):**

| Metric | Test 1 (2T+1I) | Test 4 (2T+2I) | Change |
|--------|---------------|----------------|--------|
| step/s | **2.9** | **1.1** | **-62%** |
| step_time | 0.350s | **0.85s** | +143% |
| forward | ~100ms | **~230ms** | +130% |
| backward | ~134ms | **~410ms** | +206% |
| optimizer | 22ms | 22ms | 0% |
| data | 145ms | **~400ms** | +176% |

**Raw TIMING data (selected steps):**

| Step | step (s) | data (s) | compute (s) | fwd (s) | bwd (s) | opt (s) |
|------|----------|----------|-------------|---------|---------|---------|
| 2 | 0.699 | 0.265 | 0.520 | 0.204 | 0.295 | 0.022 |
| 10 | 0.858 | 0.304 | 0.677 | 0.246 | 0.409 | 0.022 |
| 50 | 1.083 | 0.654 | 0.657 | 0.236 | 0.399 | 0.022 |
| 100 | 1.348 | 0.935 | 0.688 | 0.250 | 0.416 | 0.022 |
| 150 | 0.891 | 0.495 | 0.604 | 0.172 | 0.410 | 0.022 |
| 200 | 0.902 | 0.479 | 0.650 | 0.236 | 0.391 | 0.022 |

**Root cause**: 2 inference engines push hidden states to the same 2 training GPUs via Mooncake TCP, creating **PCIe bandwidth contention**. Mooncake transfers compete with GPU compute for bus bandwidth, roughly doubling both compute and data times. The sample pool was constantly full (72/64), confirming inference was never the bottleneck.

**Verdict**: **Worse than 1-inference.** Adding inference GPUs creates contention without benefit. The pipeline was never inference-bound.

#### 2.5.10 Configuration Comparison Summary (Final)

| Config | Train | Infer | FSDP | step/s | step_time | opt (ms) | data (ms) | Eff. Steps | Est. Train | Acc |
|--------|-------|-------|------|--------|-----------|----------|-----------|-----------|------------|-----|
| Baseline | 2 | 1 | REPLICATE | 2.5 | 0.370s | 41 | 162 | 37,500 | 4.2 hr | 96.2% |
| + no_sync + bf16 | 2 | 1 | REPLICATE | 2.7 | 0.353s | 41 | 147 | 37,500 | 3.9 hr | ~96% |
| **Test 1: FULL_SHARD** | **2** | **1** | **FULL_SHARD** | **2.9** | **0.350s** | **22** | **145** | **37,500** | **3.6 hr** | **97.1%** |
| Test 2: 3 GPU | 3 | 1 | REPLICATE | 2.2 | 0.450s | 41 | 240 | 25,000 | 3.2 hr | 96.3% |
| Test 3: FS+3GPU | 3 | 1 | FULL_SHARD | 2.3 | 0.440s | 16-22 | 220 | 25,000 | 3.0 hr | ~96% |
| Test 4: 2T+2I | 2 | 2 | FULL_SHARD | **1.1** | 0.850s | 22 | 400 | 37,500 | **9.5 hr** | — |

**Recommendation**:
- **Best per-step throughput**: Test 1 (FULL_SHARD, 2 train + 1 infer) — 2.9 step/s, simplest setup
- **Fastest wall-clock**: Test 3 (FULL_SHARD + 3 train + 1 infer) — ~3.0 hr total, uses more GPU-hours
- **Avoid**: 2+ inference GPUs — creates Mooncake/PCIe contention, strictly worse

#### 2.5.11 Hardware Topology Analysis

**GPU Interconnect**: All 4x H100 80GB are on the **same node** with full-mesh **NV18 NVLink** (18 links × 26.562 GB/s = **478 GB/s per GPU bidirectional**). Single NUMA node, Intel Xeon Platinum 8468, 1.5 TiB RAM.

```
nvidia-smi topo -m (GPU-GPU only):
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV18    NV18    NV18
GPU1    NV18     X      NV18    NV18
GPU2    NV18    NV18     X      NV18
GPU3    NV18    NV18    NV18     X
```

**No InfiniBand**: `ibstat` not available. 8x mlx5 NICs visible in topology but not exposed as RDMA interfaces. `mooncake.protocol="rdma"` and `enable_gpu_direct=True` both fail (Issue 28).

**The Mooncake overhead problem**:

Mooncake transfer engine only supports two protocols: **TCP** and **RDMA**. On this pod:
- **TCP**: Data path is `GPU → cudaMemcpy → CPU RAM → TCP loopback → CPU RAM → cudaMemcpy → GPU`. ~145ms per transfer for ~40MB hidden states.
- **RDMA**: Requires InfiniBand/RoCE hardware — not available on this RunPod.
- **No NVLink/NCCL transport**: Mooncake has no option to use NVLink p2p, NCCL, or CUDA IPC for same-node GPU-to-GPU transfers.

**What the hardware can do vs what Mooncake uses**:

| Transport | Bandwidth | Latency for 40MB | Used? |
|-----------|-----------|-------------------|-------|
| NVLink p2p (NV18) | **478 GB/s** | **<0.1 ms** | No — Mooncake doesn't support it |
| NCCL (NVLink backend) | ~450 GB/s | **<0.1 ms** | No — Mooncake doesn't support it |
| CUDA IPC / shared memory | ~400 GB/s | **<0.1 ms** | No — Mooncake doesn't support it |
| **Mooncake TCP (loopback)** | **~0.3 GB/s** | **~145 ms** | **Yes — only option** |
| Mooncake RDMA | 25-100 GB/s | ~1-5 ms | No — no IB hardware |

**Impact**: Mooncake TCP adds **145-240ms per step** (~40-60% of step time) on hardware that could do the same transfer in <0.1ms. This is the single biggest performance bottleneck, wasting **1500x available bandwidth**.

**Why more training GPUs don't help proportionally**: Adding training GPUs increases the number of Mooncake TCP consumers sharing the same loopback network. Each additional GPU adds ~80ms to data_time because TCP transfers are serialized through the CPU.

**Why 2 inference GPUs made things worse**: 2 inference engines double the Mooncake TCP traffic to each training GPU (receiving from 2 engines), saturating the loopback path in both directions simultaneously.

**Solution — same-node NVLink bypass**: Replace Mooncake TCP with NCCL p2p or `torch.distributed.send/recv` for same-node transfers. This would:
1. Reduce data_time from ~145ms to <1ms
2. Enable all 4 GPUs to communicate at NVLink speed
3. Allow 3T+1I config to achieve ~4+ step/s (currently limited by data pipeline)
4. Requires code changes to `data_fetcher.py` and `eagle_store.py` — add a `local` transport backend that uses NCCL or CUDA IPC when all GPUs are on the same node.

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

**Note**: Step counts assume ~2.1 step/s in 4-GPU SGLang mode. Actual speed with torch 2.9.1 + env var fix TBD — re-benchmark in Phase 2.2.

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

**Current state (τ=1.86)** falls in the 1.5-3.0 range. Bug 1 is now fixed; SpecForge deep diff confirms code parity. Next step: retrain from scratch with bug fixes applied and measure τ progression (Phase 3).

---

*Plan v5 — 2026-03-22 (Phase 1 complete, Phase 2 complete — FULL_SHARD recommended)*

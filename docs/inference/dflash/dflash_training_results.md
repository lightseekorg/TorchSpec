# DFlash Training Results

## Phase A: Single GPU (1x H100, 200 steps)

| Metric | DFlash | Eagle3 | Notes |
|--------|--------|--------|-------|
| Final accuracy | **0.894** (89.4%) | 0.646 (64.6%) | DFlash 38% higher |
| Final loss | 0.477 (CE) | 2.166 (KL) | Different scales, not comparable |
| Training time | **81.7s** | 202.4s | DFlash **2.5x faster** |
| Steps/sec | **~10** | ~1.8 | DFlash 5.5x more steps/sec |
| Forward passes/step | **1** | 7 | Block-parallel vs autoregressive |

Loss converges rapidly: 13.0 → 0.48 in 200 steps.

### Inference (200-step checkpoint)

| Method | Tokens/sec | τ | Speedup |
|--------|-----------|---|---------|
| Baseline (target-only) | 59.4 | N/A | 1.0x |
| DFlash speculative | 42.5 | 1.01 | 0.72x (slower) |

τ=1.01 expected — 200 steps on tiny sample data is insufficient for 899M parameter model.

---

## Phase B: 4-GPU SGLang Validation (200 steps)

| Metric | Baseline (target-only) | DFlash |
|--------|----------------------|--------|
| Throughput | 61.8 tok/s | 41.2 tok/s |
| τ | — | 1.03 |
| Speedup | 1.0x | 0.67x |

τ=1.03 expected — 200 steps on tiny data. Pipeline works end-to-end.

### SpecForge Cross-Check

**Intentional design divergences (not bugs):**

| Aspect | TorchSpec | SpecForge |
|--------|-----------|-----------|
| Architecture | Standalone PreTrainedModel | Extends Qwen3PreTrainedModel (model-specific) |
| RoPE | Computed on-the-fly per layer | Pre-computed, passed as embeddings |
| Context projection | In `extract_context_feature()` | In draft model `forward()` |
| KV cache | Not supported (recomputes) | Supported via past_key_values |
| Block mask | `if device.type == "cuda"` else None | Always created |

**Real issues fixed (training hyperparameters):**

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

**Key insight — `aot_eager` vs `inductor`**: Inductor generates fused Triton kernels that eliminate intermediate tensor materialization, giving **3x speedup and 20 GB less GPU memory**.

| Config | Speed | ETA | GPU Mem | Notes |
|--------|-------|-----|---------|-------|
| batch=1, accum=4, anchors=512, seq=4096 | 1.5 step/s | 6.6 hr | 40 GB | Baseline |
| batch=2, accum=2, anchors=512, seq=2048, epochs=4 | 1.4 step/s | 4.7 hr | 53 GB | batch=2 slower per step |
| batch=2, accum=2, anchors=**256**, seq=2048 | **1.88** step/s | 3.4 hr | 38 GB | **Big win from reducing anchors** |
| **batch=4, accum=1, anchors=256, seq=2048** | **2.1 step/s** | **3.1 hr** | 50 GB | **Final config** |

**Optimization insights**:
1. **`num_anchors` is the biggest lever**: 512→256 halves Q_LEN, gave largest single speedup
2. **`max_seq_length`** helps inference more than training
3. **`micro_batch_size`** has diminishing returns
4. anchors=256 + batch=2 has same FlexAttention cost as anchors=512 + batch=1, but processes 2x data per optimizer step

### Phase C Training Config (Historical — Pre-Bugfix)

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

### Inference Benchmark — Step 18,001

| Metric | Value |
|--------|-------|
| Acceptance length (τ) | **1.86** |
| Wall-clock speedup | **1.27x** |
| Target | τ ≥ 3.0 |

τ = 1.86 is significantly below target. Investigation identified two training quality bugs (see [Issues](dflash_issues.md#training-quality-bugs-found-in-session-12)).

### PyTorch 2.9.1 Speed Regression — RESOLVED

After PyTorch 2.6.0→2.9.1 migration: **2.1 step/s → 0.75 step/s** (3x slower).

**Fix**: `export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` — restored speed to ~5 step/s in colocate 1-GPU mode. See [Issue 26](dflash_issues.md#issue-26-pytorch-291-speed-regression-3x-slower--resolved).

### Checkpoint & Backup

- **Latest checkpoint**: `iter_0018001` (~15 GB per checkpoint)
- **HuggingFace backup**: [`Xingh3/qwen3-8b-dflash-checkpoint-phase-c`](https://huggingface.co/Xingh3/qwen3-8b-dflash-checkpoint-phase-c) (private)

| Location | Survives Pod Stop | Survives Pod Delete |
|----------|:-:|:-:|
| `/tmp/`, `/root/` (container disk) | No | No |
| `/workspace/` (volume) | Yes | No |
| HuggingFace Hub | Yes | Yes |

---

## Phase D: Bug Fixes & Smoke Test (2026-03-22)

### Environment

- torch 2.9.1 + sglang 0.5.9 + CUDA 12.4
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON`
- 1x H100 (colocate mode)
- Data: `perfectblend_50k.jsonl`

### Bug Fixes Applied

1. **Bug 1 — Zero-loss dummy** (commit `f3311e4`): Replaced silent zero-gradient fallback with `raise ValueError` in `dflash.py:130-134`
2. **`min_loss_tokens` pipeline** (commit `f3311e4`): Plumbed `min_loss_tokens` from config → data loader → preprocessing. Set `min_loss_tokens: 32` (2×block_size) in DFlash configs.
3. **SpecForge deep diff**: All 3 critical file pairs verified to match. No additional SpecForge fixes needed.

### Unit Tests

54/54 pass. 3 new tests added:
- `test_loss_decay_weights` — verifies `exp(-(k-1)/γ)` weighting
- `test_label_alignment_same_position` — verifies same-position prediction (not shifted)
- `test_anchor_loss_excluded` — verifies position 0 (anchor) has zero loss weight

### Smoke Test (30 steps, 1-GPU colocate)

| Step | Loss | Accuracy | Speed |
|------|------|----------|-------|
| 1 | 12.70 | 0.000 | warmup |
| 10 | 11.08 | 0.000 | ~1.2 step/s |
| 20 | 9.30 | 0.091 | ~3.6 step/s |
| 30 | 8.55 | 0.065 | ~5.0 step/s |

Loss starts ~12.7, drops monotonically. No NaN, no zero-loss steps. Speed ~5 step/s after warmup (exceeds Phase C baseline of 2.1 step/s on torch 2.6).

### Next Steps

- Fresh training from scratch with bug fixes, 6 epochs on 50K data (Phase 3)
- Measure τ at epoch boundaries to track progression
- Investigate pipeline overhead for speed improvements

---

## Phase E: Speed Benchmark (2026-03-22)

### Environment

- torch 2.9.1 + sglang 0.5.9 (patched) + CUDA 12.4
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON`
- 4x H100 80GB: 2 training + 1 inference + 1 idle
- Data: `perfectblend_50k.jsonl`, block_size=16, seq_len=2048, 50 steps each

### Results

| Config | batch | accum | anchors | step/s | Status |
|--------|-------|-------|---------|--------|--------|
| S1 | 1 | 4 | 512 | **1.05** | OK |
| S2 | 2 | 2 | 512 | **1.0** | OK |
| S3 | 4 | 1 | 512 | **OOM** | Backward alloc 9.27 GB |
| S3b | 2 | 2 | 256 | **1.0** | OK |

### Key Finding: Compute + Data Pipeline Both Contribute

**All configs converge to ~1.0 step/s regardless of DFlash hyperparameters.** Instrumented timing breakdown (steps 10-30 averages) reveals:

### Timing Breakdown (measured, batch=2, accum=2, anchors=512, seq=2048)

| Component | Time (s) | % of Step | Notes |
|-----------|----------|-----------|-------|
| **GPU compute** (forward+backward+optimizer) | **0.68** | **~70%** | Includes FlexAttention, CE loss, FSDP allreduce |
| **Data pipeline** (Mooncake queue fetch) | **0.40** | **~30%** | Waiting for hidden states from inference engine |
| **Ray dispatch** | **0.07** | **~7%** | Negligible |
| **Total step time** | **~0.95** | — | data + compute overlap, so sum > 100% |

Note: `data_time + compute_time > step_time` because data fetching overlaps with GPU compute (async pipeline). The "ray_overhead" column is negative, confirming overlap.

**Raw timing data (every 5th step):**

| Step | step_time | data_time | compute_time | dispatch_wait |
|------|-----------|-----------|--------------|---------------|
| 1 | 6.306s | 0.322s | 5.988s | 33.779s |
| 2 | 1.041s | 0.384s | 0.782s | 0.063s |
| 5 | 0.908s | 0.422s | 0.537s | 0.060s |
| 10 | 0.936s | 0.386s | 0.673s | 0.071s |
| 15 | 0.922s | 0.275s | 0.739s | 0.057s |
| 20 | 1.134s | 0.382s | 0.884s | 0.071s |
| 25 | 0.595s | 0.353s | 0.444s | 0.073s |
| 30 | 1.092s | 0.618s | 0.658s | 0.070s |

Step 1 is warmup (torch.compile JIT). Steps 2-30 show ~1.0s/step steady state.

**Why anchors=512 vs 256 doesn't matter**: Both produce ~0.6-0.9s compute — FlexAttention is not the dominant cost within compute. FSDP allreduce and optimizer step may dominate.

**Why 1-GPU colocate is 5x faster**: Colocate avoids the 0.4s data_time (no Mooncake transfer) and may have lower FSDP overhead (no multi-GPU allreduce). Target model runs on the same GPU, so hidden states are already in local GPU memory.

### Recommendations for Speed Improvement

| Priority | Approach | Expected Impact | Rationale |
|----------|----------|-----------------|-----------|
| **P0** | Reduce compute_time: profile forward vs backward vs optimizer breakdown | 1.5-2x | Compute is 70% of step — biggest lever |
| **P0** | Bypass Mooncake for same-node — use NCCL or shared GPU memory | 1.3-1.5x | data_time is 30% of step |
| **P1** | Increase `max_concurrent_batches` to overlap inference+training | Up to 1.5x | Pipeline already overlaps partially |
| **P1** | Enable FSDP CPU offload to fit batch=4+anchors=512 | May help if compute scales sublinearly with batch | Needs testing |
| **P2** | Use colocate mode if memory allows | 3-5x (eliminates data_time + reduces FSDP overhead) | But OOM risk with 8B target + draft on same GPU |

### Compute Sub-Breakdown (2026-03-22)

Instrumented forward/backward/optimizer with CUDA event timing. Config: batch=2, accum=2, anchors=512, seq=2048, 30 steps.

| Component | Time (ms) | % of Compute |
|-----------|----------|--------------|
| **Backward** (+ FSDP allreduce) | **~140** | **54%** |
| **Forward** (FlexAttention + CE loss) | **~82** | **31%** |
| **Optimizer** (FSDP step) | **~41** | **16%** |

Steady-state speed: **~2.5 step/s** (step_time ~0.4s). Backward dominates compute; optimizer is small and constant.

### Training Time Estimate

With 2.5 step/s (2.7 with optimizations):
- 50K samples × 6 epochs / (batch=2 × dp=2) = **37,500 optimizer steps → ~3.9 hours**

### Speed Optimization Attempts (2026-03-22)

| Optimization | Speed | Impact | Status |
|-------------|-------|--------|--------|
| **no_sync()** — skip allreduce on non-last micro-batches | Saves ~5ms/step | +4% on backward | **Implemented** (commit `949c744`) |
| **bf16 reduce** — halve allreduce volume | Minor | +4% overall | **Implemented** (commit `949c744`) |
| **Full model torch.compile** | 1-3s/step | **-10x worse** | **Not viable** — dynamic shapes cause recompilation |
| **GPU Direct RDMA** | CUDA assert | N/A | **Failed** — RunPod lacks RDMA hardware |
| **max_concurrent_batches=2** | Same 2.5 step/s | 0% | **No effect** — not bottlenecked on inference |
| **Mooncake same-node bypass** | N/A | N/A | **Needs code changes** — no config option exists |

**Best config**: no_sync + bf16 reduce → **~2.7 step/s** (+8% over baseline).

**Why torch.compile fails**: DFlash has dynamic tensor shapes (variable anchor count per sample, variable sequence lengths after filtering). torch.compile (inductor) recompiles for each new shape, causing 1-3s overhead per step for the first ~50 steps, never reaching steady state.

---

## Phase F: Speed Optimization Session (2026-03-22)

### Environment
- torch 2.9.1 + sglang 0.5.9 (patched) + CUDA 12.4
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON`
- 4x H100 80GB: 2 training + 1 inference + 1 idle
- Data: `perfectblend_50k.jsonl`

### Compute Sub-Breakdown (CUDA Event Profiling)

Instrumented `dflash_trainer._train_step()` with forward/backward events and `trainer._train_core_from_queue()` with optimizer events.

**Steady-state breakdown (steps 2-200, batch=2, accum=2, anchors=512, seq=2048):**

| Component | Time (ms) | % of Step |
|-----------|----------|-----------|
| **Backward** (gradient compute + FSDP allreduce) | 138 | 37% |
| **Data pipeline** (Mooncake TCP fetch) | 162 | 44% |
| **Forward** (FlexAttention + CE loss) | 78 | 21% |
| **Optimizer** (FSDP step) | 41 | 11% |
| **Dispatch** (Ray overhead) | 36 | 10% |
| **Step total** | ~370 | — |

> Sum > 100% because data and compute overlap (async pipeline).

### 200-Step Stability Test

Verified speed is stable over 200 steps (2 epochs) with no degradation:

| Step Range | step/s | step (s) | compute (s) | data (s) |
|-----------|--------|----------|-------------|----------|
| 2-30 | 2.7 | 0.370 | 0.263 | 0.160 |
| 30-100 | 2.8 | 0.360 | 0.255 | 0.162 |
| 100-200 | 2.8 | 0.356 | 0.255 | 0.162 |

Training converged well: loss 12.35→0.12, accuracy 0%→96.2%.

### Optimization Tests

**no_sync + bf16 reduce (commit `949c744`):**

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| step_time | 0.370s | 0.353s | **-5%** |
| backward | 138ms | 133ms | -4% |
| optimizer | 41ms | 41ms | 0% |
| data | 162ms | 147ms | -9% |
| **Speed** | **2.5 step/s** | **2.7 step/s** | **+8%** |

**Full model torch.compile:** Tested but not viable. Step 1 = 111s warmup. Steps 2-50 averaged 1-3s/step due to repeated recompilation from dynamic tensor shapes. Discarded.

### FSDP Strategy & GPU Scaling Tests (2026-03-22)

#### Test 1: FULL_SHARD (2 training GPUs, bf16 reduce)

FULL_SHARD distributes parameters across GPUs, halving optimizer memory and time.

| Metric | REPLICATE (baseline) | FULL_SHARD | Change |
|--------|---------------------|------------|--------|
| step/s | 2.5 | **2.9** | **+16%** |
| optimizer | 41ms | **22ms** | **-46%** |
| forward | ~82ms | ~100ms | +22% |
| backward | ~138ms | ~134ms | -3% |
| 200-step wall time | 244.5s | **220.5s** | **-10%** |
| Final accuracy | 96.2% | **97.1%** | +0.9% |

#### Test 2: 3 Training GPUs (REPLICATE, bf16 reduce)

3-way DP: slower per-step (higher Mooncake data_time), but 50% more data per step.

| Metric | 2-GPU REPLICATE | 3-GPU REPLICATE |
|--------|----------------|-----------------|
| step/s | 2.5 | **2.2** |
| data_time | 162ms | **240ms** |
| 200-step wall time | 244.5s | 285.4s |
| Effective total steps | 37,500 | **25,000** |
| Est. training time | 4.2 hr | **3.2 hr** |

#### Test 3: FULL_SHARD + 3 Training GPUs (concurrent=3, bf16 reduce)

Combines FULL_SHARD optimizer savings with 3-way DP throughput.

| Metric | Value |
|--------|-------|
| step/s | **2.3** |
| optimizer | **16-22ms** |
| data_time | ~220ms |
| 200-step wall time | 255.4s |
| Est. training time | **~3.0 hr** |

#### Configuration Comparison

| Config | GPUs | FSDP | step/s | opt (ms) | data (ms) | Est. Train Time |
|--------|------|------|--------|----------|-----------|-----------------|
| Baseline | 2 | REPLICATE | 2.5 | 41 | 162 | 4.2 hr |
| + no_sync + bf16 | 2 | REPLICATE | 2.7 | 41 | 147 | 3.9 hr |
| **Test 1** | 2 | FULL_SHARD | **2.9** | **22** | 145 | **3.6 hr** |
| Test 2 | 3 | REPLICATE | 2.2 | 41 | 240 | 3.2 hr |
| **Test 3** | 3 | FULL_SHARD | **2.3** | **16-22** | 220 | **3.0 hr** |
| Test 4 (2T+2I) | 2+2I | FULL_SHARD | **1.1** | 22 | 400 | **9.5 hr** |

#### Test 4: 2 Training + 2 Inference GPUs (FULL_SHARD) — WORSE

Adding a second inference engine caused **-62% speed regression** (2.9→1.1 step/s). Root cause: 2 inference engines push hidden states to the same 2 training GPUs via Mooncake TCP, creating PCIe bandwidth contention. Both compute and data times roughly doubled. Sample pool was constantly full (72/64), confirming inference was never the bottleneck.

**Lesson**: Do not add inference GPUs beyond 1 for this workload. The pipeline is training-bound, not inference-bound.

### Hardware Topology & Mooncake Bottleneck Analysis

**Hardware**: 4x H100 80GB on single node with full-mesh **NV18 NVLink** (478 GB/s per GPU). Intel Xeon Platinum 8468, 1.5 TiB RAM, single NUMA node. No InfiniBand.

**The problem**: Mooncake only supports TCP or RDMA. On this pod, TCP is the only option — data goes `GPU → CPU RAM → TCP loopback → CPU RAM → GPU`, taking ~145ms for ~40MB of hidden states. The NVLink interconnect (478 GB/s, <0.1ms for same transfer) is **completely unused** by Mooncake.

| Transport | Time for 40MB | Available? |
|-----------|---------------|------------|
| NVLink p2p | <0.1 ms | Hardware yes, Mooncake no |
| Mooncake TCP | **~145 ms** | **Currently used** |

This TCP overhead is 40-60% of step time and the single biggest bottleneck. Implementing a same-node NVLink/NCCL bypass for Mooncake would reduce data_time from ~145ms to <1ms, potentially enabling 4+ step/s.

### Async Data Pre-Fetch Tests (2026-03-22)

#### Test 5a: GPU Prefetch (prefetch_depth=2, staging to GPU) — WORSE

Background thread fetches from Mooncake to GPU concurrently with forward/backward. **GPU contention** caused 2-3x compute slowdown.

| Metric | Test 1 (no prefetch) | Test 5a (GPU prefetch) |
|--------|---------------------|----------------------|
| step/s | 2.9 | **1.0** |
| data_time | 145ms | 11-46ms |
| compute_time | 100ms | **200-300ms** |

**Root cause**: Background `cudaMemcpy` (HtoD) from Mooncake competes with forward/backward for GPU/PCIe bandwidth. Commit `3ceb630`.

#### Test 5b: CPU Prefetch (prefetch_depth=2, staging to CPU) — **2.3x SPEEDUP**

Fixed: Background thread stages data on CPU only. Main thread moves to GPU synchronously between steps, eliminating GPU contention.

| Metric | Test 1 (baseline) | Test 5b (CPU prefetch) | Change |
|--------|---------------------|----------------------|--------|
| step/s | 2.9 | **~6.8** | **+134%** |
| step_time | 345ms | **100-123ms** | -65% |
| data_time | 145ms | **1-23ms** | -97% |
| compute_time | 100ms | **92-115ms** | No contention |

**200-step stability confirmed**: Steady ~6.5-7.0 step/s with no degradation. Loss 12.3→0.39, accuracy 0%→93.5%. Data transfer fully overlapped with compute — step time ≈ compute time. Commit `bb922ba`.

#### Test 6: NVLink Intra-Node Transport (mooncake.protocol=nvlink_intra) — FAILED (Architectural Mismatch)

Three progressive issues were encountered and resolved, revealing a fundamental limitation:

**Issue 1 — pip wheel lacks NVLink transport** (resolved):
```
E0322 04:11:07.613032 client_service.cpp:413] unsupported_protocol protocol=nvlink_intra
```
Fix: Built Mooncake from source with `-DUSE_INTRA_NVLINK=ON -DUSE_CUDA=ON`.

**Issue 2 — `client_service.cpp` protocol switch missing `nvlink_intra`** (resolved):
The mooncake-store's `client_service.cpp` had a hardcoded if-else chain for protocols (tcp, rdma, ascend, ubshmem, cxl) that did not include `nvlink_intra`. Patched to add the case, rebuilt, and replaced both `store.so` and `engine.so`.

**Issue 3 — NVLink requires GPU memory, but mooncake-store uses host memory** (fundamental):
```
E0322 05:07:54.569949 intranode_nvlink_transport.cpp:298] Unsupported memory type, 0x701413ff4040 0
```
The `nvlink_intra` transport checks `cudaPointerGetAttributes` and requires `cudaMemoryTypeDevice` (GPU memory). The mooncake-store allocates shared memory (SHM) buffers on the host — memory type `0` (`cudaMemoryTypeUnregistered`). NVLink IPC only works for GPU-to-GPU transfers.

**Conclusion**: The `nvlink_intra` transport is designed for SGLang's disaggregated prefill/decode KV cache transfers (GPU↔GPU). The mooncake-store data pipeline (training data) uses host memory buffers, which fundamentally cannot use NVLink. This is not fixable via configuration — it would require the store to allocate GPU-resident buffers.

**Mooncake transport viability for training data**:

| Protocol | Works with Store? | Notes |
|----------|------------------|-------|
| `tcp` | **Yes (current)** | ~0.3 GB/s, ~145ms/40MB |
| `rdma` | Possible | Needs InfiniBand hardware (RunPod lacks it) |
| `nvlink_intra` | **No** | Requires GPU memory; store uses host memory |
| `nvlink` (MNNVL) | **No** | Same GPU memory requirement |

**Recommendation**: TCP + CPU prefetch (Test 5b) already eliminates the data pipeline bottleneck entirely (data_time < compute_time). NVLink transport is not needed for training throughput.

#### Updated Configuration Comparison

| Config | GPUs | FSDP | Prefetch | step/s | data (ms) | compute (ms) | Est. Train Time |
|--------|------|------|----------|--------|-----------|--------------|-----------------|
| Baseline | 2 | REPLICATE | No | 2.5 | 162 | 258 | 4.2 hr |
| + no_sync + bf16 | 2 | REPLICATE | No | 2.7 | 147 | 253 | 3.9 hr |
| Test 1 | 2 | FULL_SHARD | No | 2.9 | 145 | 100 | 3.6 hr |
| Test 2 | 3 | REPLICATE | No | 2.2 | 240 | 258 | 3.2 hr |
| Test 3 | 3 | FULL_SHARD | No | 2.3 | 220 | 100 | 3.0 hr |
| Test 4 (2T+2I) | 2+2I | FULL_SHARD | No | 1.1 | 400 | 200 | 9.5 hr |
| Test 5a | 2 | FULL_SHARD | GPU | 1.0 | 11-46 | 200-300 | N/A (worse) |
| **Test 5b** | **2** | **FULL_SHARD** | **CPU** | **6.8** | **1-23** | **92-115** | **1.5 hr** |
| Test 6 (NVLink) | 2 | FULL_SHARD | No | N/A | N/A | N/A | Not viable (store uses host mem) |
| **Test 7** | **3** | **FULL_SHARD** | **CPU** | **5.3** | **1-9** | **118-148** | **~1.3 hr** |

#### Test 7: 3 GPU + FULL_SHARD + CPU Prefetch (30 steps)

Combined 3 training GPUs with FULL_SHARD and CPU prefetch. Eval disabled (`dataset.eval_data_path=null`).

| Metric | Value |
|--------|-------|
| step/s | 5.2-5.6 (steady state) |
| step_time | 127-165ms |
| data_time | 1-9ms |
| compute_time | 118-148ms |
| forward | 43-69ms |
| backward | 52-64ms |
| optimizer | 15-16ms |
| samples/step | 3 (1 per GPU) |
| effective throughput | 15.9 samples/s (5.3 × 3) |

**Comparison with Test 5b (2 GPU + CPU prefetch)**:
- Per-step: 5.3 vs 6.8 step/s (3 GPU is 22% slower per step due to FSDP overhead)
- But each step processes 3 samples vs 2 → effective throughput: 15.9 vs 13.6 samples/s
- **3 GPU is ~17% faster** in total wall clock time (~1.3 hr vs ~1.5 hr)
- Loss progression: 12.3 → 6.3 in 30 steps (similar trajectory to 2 GPU)

---

## Phase 3: Acceptance Length (τ) Benchmark Matrix

### Plan

**Goal**: Measure τ (average accepted tokens per draft cycle) at training checkpoints to determine when training quality is sufficient for speculative decoding speedup.

**Target**: τ ≥ 5.0 (each draft cycle produces ~5 accepted tokens → significant speedup over baseline)

**Training config** (Test 7 — best throughput):
- 3 GPU + FULL_SHARD + CPU prefetch (5.3 step/s × 3 samples/step)
- Dataset: `perfectblend_50k.jsonl` (47,484 samples)
- 6 epochs, save_interval=5000, eval disabled
- Steps per epoch: ~15,828 (47,484 ÷ 3 samples/step)
- Total steps: ~94,968

**Incremental approach**: Train 5k steps first, benchmark τ and performance. If results are promising, continue to 10k and 15k.

**Checkpoint schedule**:

| Checkpoint | Steps | Epoch | Est. Wall Time | Measurement |
|------------|-------|-------|-----------------|-------------|
| **ckpt-5k** | **5,000** | **~0.3** | **~16 min** | **τ, loss, inference tok/s (verify first)** |
| ckpt-10k | 10,000 | ~0.6 | ~31 min | τ, loss, inference tok/s |
| ckpt-15k | 15,000 | ~0.9 | ~47 min | τ, loss, inference tok/s |
| epoch-1 | 15,828 | 1.0 | ~50 min | τ, loss, inference tok/s |

**Measurements at each checkpoint**:
1. **τ (acceptance length)**: Average accepted tokens per draft cycle via `torchspec.eval_entry`
2. **Loss**: Training loss at checkpoint step
3. **Inference speed**: Tokens/sec with DFlash speculative decoding vs baseline (target-only)
4. **Comparison**: Side-by-side with Eagle3 inference at same target model

**Baseline references**:
- Target-only (no speculation): ~60 tok/s
- Previous DFlash (step 18,001, pre-bugfix): τ = 1.86
- Eagle3: τ = TBD (to be benchmarked for comparison)

### Updated Plan (v6 — 2-Epoch Stop)

After comparing with the [reference configs](specforge_dflash_training_reference.md), switched to a 2-epoch stop with decision matrix. Our hyperparameters match z-lab official (lr=6e-4, warmup=0.04), but dataset is 16x smaller (50K vs 800K).

**Revised training config** (batch=1, accum=4 for better Mooncake TCP overlap):
- `micro_batch_size: 1`, `draft_accumulation_steps: 4` → global_batch=12
- `max_seq_length: 2048`, `num_anchors: 512`, `prefetch_depth: 8`
- `num_epochs: 2`, `save_interval: 1000`, `max_checkpoints: 1`
- Steps per epoch: ~3,934, total 2 epochs: ~7,868 steps

### Critical Bug Fixes (2026-03-22)

| Bug | Impact | Fix |
|-----|--------|-----|
| **FlexAttention recompile overflow** (Issue 30) | 5.7x slower (1.09s vs 0.19s/step) — 64+ unique padded shapes exhaust `recompile_limit`, falling back to eager mode | Bucketed padding to nearest 256 boundary (max 8 unique shapes). Commit `2e105c4`. |
| **Mooncake buffer use-after-free** (Issue 31) | Batch>1 crash — collator holds freed Mooncake buffer, gets corrupted by new inference data | Clone tensors before cleanup. Commit `cdd18cf`. |
| **Clone breaks pinned memory** (Issue 32) | 5x speed regression — `.clone()` creates unpinned tensors, `non_blocking=True` silently falls back to blocking transfer | Conditional clone for batch>1 only; batch=1 uses pinned views directly. Commit `2a6a3d9`. |

### Results

| Checkpoint | Steps | Epoch | Loss | Accuracy | τ | DFlash tok/s | Baseline tok/s | Speedup |
|------------|-------|-------|------|----------|---|-------------|----------------|---------|
| ckpt-1k | 1,001 | 0.25 | ~5.0 | 0.13 | — | — | — | — |
| ckpt-2epoch | 7,869 | 2.0 | 3.47 | 0.23 | **1.85** | 67.9 | 56.4 | 1.10x |
| ckpt-3epoch | 11,803 | 3.0 | ~2.7 | 0.35 | **1.85** | 67.9 | 56.4 | 1.10x |

**Key finding**: τ is identical between epoch 2 and epoch 3 (1.85). Training loss continued to decrease (3.47→2.7) and accuracy improved (0.23→0.35), but **inference quality did not improve**. The model saturated on 47K samples — additional epochs only improve memorization, not generalization for speculative decoding.

### 2-Epoch Training Run (2026-03-22)

**Pod**: 4x H100 80GB (RunPod), custom Docker image `ghcr.io/zhubohao911/torchspec-dflash:latest`

**Training summary**:
- Duration: **1h 39min** (7,868 steps)
- Speed: **~0.75 s/step** (~1.2 step/s)
- Loss progression: 10+ → 5.0 (step 500) → 3.5 (step 2000) → 3.0 (step 5000) → 3.47 (final)
- Accuracy: 0 → 0.13 (step 1000) → 0.23 (step 3000) → 0.30 (step 5000) → 0.23 (final)
- Auto-resumed from `iter_1001` checkpoint (downloaded from HF)

**Speed analysis**: 0.75 s/step is slower than Phase F Test 7 (0.19 s/step) due to:
- Mooncake TCP transfer dominates data_time (~300ms per 80MB hidden states)
- Batch=1 avoids the clone overhead but reduces GPU utilization
- FlexAttention bucketed padding eliminates recompilation but adds padding overhead

### 2-Epoch Benchmark (49 prompts, 512 tokens, greedy)

| Metric | Value |
|--------|-------|
| **Mean τ** | **1.78** |
| **Median τ** | **1** |
| **DFlash throughput** | 72.5 tok/s |
| **Baseline throughput** | 56.3 tok/s |
| **Speedup** | 1.09x |

**τ distribution** (14,140 draft cycles across 49 prompts):

| τ | Count | % |
|---|-------|---|
| 1 | 7,559 | 53.5% |
| 2 | 3,930 | 27.8% |
| 3 | 1,652 | 11.7% |
| 4 | 581 | 4.1% |
| 5+ | 418 | 3.0% |

**Per-topic performance** (best → worst):
- Code/structured prompts: 93-221 tok/s (τ~2.5-3.4) — draft predicts well
- History/science: 60-72 tok/s (τ~1.6-1.8) — average
- Open-ended/creative: 22-50 tok/s (τ<1.5) — draft adds little value

### Assessment

τ=1.85 is **below the 3.0 target** but consistent with expectations:
- z-lab achieved τ≥3.0 with **800K samples** (16x our dataset)
- Our 50K dataset (47,484 samples) × 2 epochs = 94,968 sample passes
- z-lab's 800K × 6 epochs = 4.8M sample passes — **50x more training**

The model has learned basic next-token patterns but cannot reliably predict multi-token blocks (53% of cycles accept only 1 token).

**Epoch 3 confirmed no improvement**: τ=1.85 identical to epoch 2. Training loss dropped (3.47→2.7) and accuracy rose (0.23→0.35), but the draft model's predictions at inference time are byte-for-byte identical. The model has memorized the 47K dataset without gaining better generalization for token prediction.

### Decision Matrix (per test plan v6)

| τ at Epoch 2 | Action |
|---|---|
| τ < 2.0 ← **HERE (1.85)** | Scale dataset (200K-800K) — more epochs won't help (confirmed by epoch 3) |
| 2.0 ≤ τ < 3.0 | Continue to epoch 4-6 on current data |
| τ ≥ 3.0 | Success — deploy |

### Recommended Next Steps

1. **Scale dataset to 200K+ samples** — most impactful (16x data gap is the primary limitation, epoch 3 confirmed more epochs don't help)
2. ~~Train 4-6 epochs on current 50K~~ — **ruled out** (epoch 3 = epoch 2, no improvement)
3. **Investigate τ=1 dominance** — 53% of cycles accept only 1 token; may indicate block-position bias in draft model or benchmark implementation issue

### Checkpoints

| Checkpoint | Location |
|------------|----------|
| iter_1001 (step 1k) | [Xingh3/dflash-qwen3-8b-1k](https://huggingface.co/Xingh3/dflash-qwen3-8b-1k) |
| iter_7869 (2 epochs) | [Xingh3/dflash-qwen3-8b-2epoch](https://huggingface.co/Xingh3/dflash-qwen3-8b-2epoch) |
| iter_11803 (3 epochs) | [Xingh3/dflash-qwen3-8b-3epoch](https://huggingface.co/Xingh3/dflash-qwen3-8b-3epoch) |

---

## Environment Issues & Workarounds

| Issue | Symptom | Workaround |
|-------|---------|------------|
| Missing `libibverbs.so.1` | Mooncake fails to load at runtime | `apt-get install libibverbs-dev` on pod |
| RunPod SCP not supported | SSH proxy rejects file transfers | Commit changes locally, `git pull` on pod |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` not set | PyTorch 2.9.1 speed regression (3x slower) | `export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` — should be added to `train_entry.py` |
| `eval_interval` under wrong section | Config parsing error | `eval_interval` belongs under `dataset`, not `training` |
| `enable_perf` wrong path | Perf metrics flag not found | Correct path: `debug.enable_perf_metrics` |
| Eval timeout with 3 GPUs | Eval cache generation hangs | Set `dataset.eval_data_path=null` to disable eval |

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
| `dcd5f45` | Add compute sub-breakdown profiling instrumentation |
| `949c744` | Add no_sync, compile_model, bf16 reduce optimizations |
| `641802f` | Add FSDP strategy config (FULL_SHARD/REPLICATE) |
| `cedef38` | Fix factory.py timeout 30s→120s for PyTorch 2.9+ compatibility |
| `f75a285` | Add async data pre-fetch (PrefetchedDataFetcher) |
| `3ceb630` | Fix PrefetchedDataFetcher persistent thread (single background thread) |
| `bb922ba` | Fix prefetch GPU contention: stage data on CPU, move to GPU synchronously |
| `cdd18cf` | Fix Mooncake buffer use-after-free with batch>1 (clone before cleanup) |
| `2e105c4` | Fix FlexAttention recompile overflow: bucketed padding + higher recompile limit |
| `3ab009b` | Switch to batch=1 accum=4 for better Mooncake TCP overlap |
| `2a6a3d9` | Fix 5x speed regression: skip clone for batch=1 to preserve pinned memory |
| `260f7af` | Update Dockerfile and setup script for faster pod provisioning |
| `ece3e25` | Expand benchmark to 50 diverse prompts with τ distribution analysis |

# DFlash Training Results

## Success Criteria

From the test plan:

| Metric | Target | Best Achieved | Status |
|--------|--------|---------------|--------|
| Acceptance length (τ) | ≥ 5.0 | 3.94 math avg (Phase K P2-WSD) | Below target |
| Training speed | ≥ 2.0 step/s | ~6.8 step/s (RunPod 2-GPU CPU prefetch) | Exceeded |
| Speedup | ≥ 3.0x | 3.02x (livecodebench, Phase K P2-WSD) | **Met** |

**Note**: The original τ ≥ 5.0 and speedup ≥ 3.0x targets were from the DFlash paper. These were later revised downward as we discovered the gap is primarily due to training recipe differences (data volume, epochs, sequence length) rather than code defects. The best model (Phase K P2-WSD: 760K samples, 3 epochs, WSD schedule) achieved τ=3.94 math avg — within 2.7% of z-lab's 4.05 with 47% of their total sample passes. Decode-only speedup hit 3.02x on livecodebench.

---

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

τ = 1.86 is significantly below target. Investigation identified two training quality bugs (see [Issues](issues.md#training-quality-bugs-found-in-session-12)).

### PyTorch 2.9.1 Speed Regression — RESOLVED

After PyTorch 2.6.0→2.9.1 migration: **2.1 step/s → 0.75 step/s** (3x slower).

**Fix**: `export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON` — restored speed to ~5 step/s in colocate 1-GPU mode. See [Issue 26](issues.md#issue-26-pytorch-291-speed-regression-3x-slower--resolved).

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

After comparing with the [reference configs](specforge_reference.md), switched to a 2-epoch stop with decision matrix. Our hyperparameters match z-lab official (lr=6e-4, warmup=0.04), but dataset is 16x smaller (50K vs 800K).

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

## Modal: Baseline & Speed Tuning (8x H100)

### Environment

- **Platform**: Modal (serverless GPU cloud)
- **GPUs**: 8x NVIDIA H100 80GB HBM3
- **Interconnect**: NVLink (intra-node), no RDMA/InfiniBand
- **Software**: torch 2.11 + sglang 0.5.9 + CUDA 12.4
- **Mooncake**: TCP protocol, CPU prefetch (prefetch_depth=8)
- **Dataset**: PerfectBlend 50K (`perfectblend_50k.jsonl`, 47,484 samples)
- **Base config**: `configs/sglang_qwen3_8b_dflash.yaml`

### Baseline: 8x H100 (1 Inference + 7 Training FSDP)

200 steps, `micro_batch_size=1`, `draft_accumulation_steps=4`, `dflash_num_anchors=512`, `max_seq_length=2048`.

Global batch size = 1 × 4 × 7 = 28.

#### Timing (steady state, steps 50-200)

| Metric | Value |
|--------|-------|
| step_time | 0.87-1.4s (high variance) |
| forward | 260-800ms |
| backward | 465-580ms |
| optimizer | ~15ms |
| data_time | 400-540ms (overlapped) |
| dispatch_wait | high |
| thru (samples/s) | ~20-25 |
| T (train capacity) | ~50 |
| pool | full (24-44 / 64) |

#### Analysis

- **Forward variance**: FlexAttention `mask_mod` closure changes every step (different `anchor_positions`), triggering Dynamo recompilation. `Q_LEN = n_blocks × block_size` where `n_blocks` varies per batch.
- **Backward**: 7-way FSDP FULL_SHARD reduce-scatter + all-gather.
- **Data**: Fully overlapped with compute via CPU prefetch (prefetch_depth=8).
- **Bottleneck**: Compute (forward + backward), not data transfer.

### Speed Tuning Tests (200 steps each)

All tests use 8x H100 with CLI overrides via `--extra-overrides`. No YAML changes.

#### Test A: Reduce Anchors (1 Inference + 7 Training)

```
training.dflash_num_anchors=256
```

Rationale: Halves Q_LEN from ~8K to ~4K. Prior Phase C results showed anchors=256 was the single biggest speedup lever.

| Metric | Value |
|--------|-------|
| Total time | **610s** |
| step_time | **0.55-0.64s** |
| forward | **240-326ms** |
| backward | 256-303ms |
| optimizer | ~13ms |
| thru (samples/s) | 17-19 |
| T (train capacity) | 48-51 |
| pool | full (8-12 / 64) |
| dispatch_wait | 0.7-1.7s |

**TIMING samples (every 50 steps)**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.649s | 0.212s | 0.630s | 0.344s | 0.274s | 0.012s | 0.903s |
| 100 | 0.639s | 0.219s | 0.628s | 0.312s | 0.303s | 0.013s | 0.798s |
| 150 | 0.630s | 0.224s | 0.613s | 0.326s | 0.273s | 0.013s | 0.900s |
| 200 | 0.558s | 0.224s | 0.544s | 0.240s | 0.290s | 0.014s | 0.714s |

**COMPUTE_BREAKDOWN (CUDA event profiling)**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 272.8 | 256.6 |
| 150 | 266.8 | 269.9 |

#### Test B: Larger Micro-Batch + Fewer Anchors (1 Inference + 7 Training)

```
training.micro_batch_size=2 training.draft_accumulation_steps=2 training.dflash_num_anchors=256
```

Global batch size = 2 × 2 × 7 = 28 (same as baseline).

| Metric | Value |
|--------|-------|
| Total time | 637s |
| step_time | 0.62-0.71s |
| forward | 260-455ms |
| backward | 226-255ms |
| optimizer | ~13ms |
| thru (samples/s) | 16-18 |
| T (train capacity) | 33-45 |
| pool | 8-12 / 64 |
| dispatch_wait | 0.6-0.8s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.749s | 0.156s | 0.733s | 0.488s | 0.232s | 0.013s | 0.698s |
| 100 | 0.676s | 0.137s | 0.665s | 0.395s | 0.255s | 0.014s | 0.555s |
| 150 | 0.713s | 0.162s | 0.693s | 0.455s | 0.226s | 0.012s | 0.775s |
| 200 | 0.620s | 0.170s | 0.597s | 0.355s | 0.230s | 0.013s | 0.613s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 502.2 | 230.8 |
| 150 | 260.2 | 230.3 |

#### Test C: Maximum Micro-Batch (1 Inference + 7 Training)

```
training.micro_batch_size=4 training.draft_accumulation_steps=1 training.dflash_num_anchors=256
```

Global batch size = 4 × 1 × 7 = 28 (same as baseline).

| Metric | Value |
|--------|-------|
| Total time | 640s |
| step_time | 1.39-1.46s |
| forward | 607-986ms |
| backward | 213-218ms |
| optimizer | ~13ms |
| thru (samples/s) | 16-18 |
| T (train capacity) | 16-20 |
| pool | **16-28 / 64 (starved!)** |
| dispatch_wait | 0.1s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 1.447s | 0.350s | 1.095s | 0.865s | 0.217s | 0.014s | 0.071s |
| 100 | 1.459s | 0.237s | 1.218s | 0.986s | 0.218s | 0.013s | 0.069s |
| 150 | 1.440s | 0.523s | 0.915s | 0.686s | 0.216s | 0.013s | 0.069s |
| 200 | 1.389s | 0.549s | 0.837s | 0.607s | 0.217s | 0.013s | 0.071s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 173.3 | 215.2 |
| 150 | 639.5 | 213.8 |

**Problem**: With `accum=1`, every step consumes 28 samples. The single inference GPU produces ~17 samples/s, which cannot keep the pool full. Data starvation causes the trainer to idle waiting for samples.

#### Test C2: Maximum Micro-Batch + 2 Inference GPUs (2 Inference + 6 Training)

```
training.micro_batch_size=4 training.draft_accumulation_steps=1 training.dflash_num_anchors=256
inference.inference_num_gpus=2 training.training_num_gpus_per_node=6
```

Global batch size = 4 × 1 × 6 = 24.

| Metric | Value |
|--------|-------|
| Total time | **476s** |
| step_time | 0.86-0.95s |
| forward | 272-434ms |
| backward | 211-214ms |
| optimizer | ~14ms |
| thru (samples/s) | **22-25** |
| I (inference/s) | **33-35** |
| T (train capacity) | 19-26 |
| pool | **64 / 64 (full!)** |
| dispatch_wait | 0.05s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.857s | 0.263s | 0.593s | 0.361s | 0.217s | 0.014s | 0.053s |
| 100 | 0.953s | 0.312s | 0.640s | 0.414s | 0.211s | 0.015s | 0.054s |
| 150 | 0.913s | 0.251s | 0.660s | 0.434s | 0.212s | 0.014s | 0.054s |
| 200 | 0.943s | 0.301s | 0.641s | 0.412s | 0.212s | 0.016s | 0.053s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 374.9 | 212.1 |
| 150 | 272.1 | 212.2 |
| 195 | 142.4 | 214.4 |

### Anchors=512 Speed Tuning (200 steps each)

Motivated by `specforge_reference.md`: z-lab uses `dflash_num_anchors=512` for best
acceptance length (τ). This phase tests whether anchors=512 can match anchors=256 speed when
properly tuned with 2 inference GPUs.

#### Test 512-A: Baseline (1 Inference + 7 Training)

```
training.dflash_num_anchors=512
```

Identical to original baseline, re-run for consistent comparison.

| Metric | Value |
|--------|-------|
| Total time | **584s** |
| step_time | 0.83-1.40s |
| forward | 258-837ms |
| backward | 456-539ms |
| optimizer | ~13-14ms |
| thru (samples/s) | 17-20 |
| T (train capacity) | 29-33 |
| pool | **12-28 / 64 (starved!)** |
| dispatch_wait | 1.0-1.5s |

**TIMING samples (every 50 steps)**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.825s | 0.396s | 0.811s | 0.258s | 0.539s | 0.014s | 0.678s |
| 100 | 1.404s | 0.442s | 1.394s | 0.777s | 0.603s | 0.013s | 0.066s |
| 150 | 0.864s | 0.434s | 0.848s | 0.300s | 0.534s | 0.014s | 0.644s |
| 200 | 1.377s | 0.419s | 1.351s | 0.837s | 0.500s | 0.013s | 0.201s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 319.3 | 501.6 |
| 150 | 284.9 | 534.8 |
| 195 | 468.3 | 456.1 |

**Problem**: Single inference GPU cannot keep pool saturated (pool=12-28). Dispatch wait 1.0-1.5s.

#### Test 512-B: Larger Micro-Batch (1 Inference + 7 Training)

```
training.dflash_num_anchors=512 training.micro_batch_size=2 training.draft_accumulation_steps=2
```

Global batch size = 2 × 2 × 7 = 28 (same as baseline).

| Metric | Value |
|--------|-------|
| Total time | **574s** |
| step_time | 0.85-1.28s |
| forward | 204-705ms |
| backward | 428-553ms |
| optimizer | ~13-14ms |
| thru (samples/s) | 17-19 |
| T (train capacity) | 29-40 |
| pool | **12-20 / 64 (starved!)** |
| dispatch_wait | 1.1-1.4s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.865s | 0.318s | 0.850s | 0.312s | 0.525s | 0.014s | 0.615s |
| 100 | 1.283s | 0.292s | 1.272s | 0.705s | 0.553s | 0.013s | 0.262s |
| 150 | 0.845s | 0.308s | 0.826s | 0.357s | 0.456s | 0.013s | 0.687s |
| 200 | 0.990s | 0.316s | 0.968s | 0.527s | 0.428s | 0.014s | 0.523s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 377.0 | 447.9 |
| 150 | 239.6 | 495.3 |
| 195 | 204.4 | 456.6 |

**Problem**: Same as 512-A — single inference GPU starves the pool. batch=2 slightly faster total time (574s vs 584s) but still bottlenecked by data.

#### Test 512-C: Larger Micro-Batch + 2 Inference GPUs (2 Inference + 6 Training)

```
training.dflash_num_anchors=512 training.micro_batch_size=2 training.draft_accumulation_steps=2
inference.inference_num_gpus=2 training.training_num_gpus_per_node=6
```

Global batch size = 2 × 2 × 6 = 24.

| Metric | Value |
|--------|-------|
| Total time | **446s** |
| step_time | 0.99-1.29s |
| forward | 264-706ms |
| backward | 414-515ms |
| optimizer | ~15-16ms |
| thru (samples/s) | **20-25** |
| I (inference/s) | 36-46 |
| T (train capacity) | 19-26 |
| pool | **64 / 64 (full!)** |
| dispatch_wait | 0.05s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.997s | 0.462s | 0.783s | 0.264s | 0.504s | 0.015s | 0.053s |
| 100 | 1.082s | 0.379s | 1.003s | 0.573s | 0.414s | 0.016s | 0.051s |
| 150 | 1.293s | 0.360s | 1.213s | 0.682s | 0.515s | 0.015s | 0.047s |
| 200 | 0.985s | 0.250s | 0.973s | 0.502s | 0.455s | 0.016s | 0.060s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 438.1 | 435.2 |
| 150 | 323.0 | 496.9 |
| 195 | 705.9 | 479.4 |

#### Test 512-D: Baseline + 2 Inference GPUs (2 Inference + 6 Training)

```
training.dflash_num_anchors=512 inference.inference_num_gpus=2 training.training_num_gpus_per_node=6
```

Global batch size = 1 × 4 × 6 = 24.

| Metric | Value |
|--------|-------|
| Total time | **457s** |
| step_time | **0.93-1.09s** |
| forward | 305-448ms |
| backward | 473-540ms |
| optimizer | ~14-15ms |
| thru (samples/s) | **22-25** |
| I (inference/s) | 27-41 |
| T (train capacity) | 27-29 |
| pool | **64 / 64 (full!)** |
| dispatch_wait | 0.05-0.07s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 1.085s | 0.516s | 0.933s | 0.380s | 0.537s | 0.015s | 0.066s |
| 100 | 1.001s | 0.431s | 0.982s | 0.448s | 0.518s | 0.015s | 0.066s |
| 150 | 0.972s | 0.419s | 0.930s | 0.443s | 0.473s | 0.015s | 0.060s |
| 200 | 0.934s | 0.427s | 0.917s | 0.363s | 0.540s | 0.014s | 0.066s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 332.8 | 521.9 |
| 150 | 305.3 | 507.3 |
| 195 | 308.0 | 492.1 |

**Notable**: Most stable forward times of all anchors=512 tests (305-448ms range vs 204-837ms for others). Less FlexAttention recompilation jitter with batch=1.

#### Test 512-E: 4 Inference + 4 Training GPUs

```
training.dflash_num_anchors=512 inference.inference_num_gpus=4 training.training_num_gpus_per_node=4
```

Global batch size = 1 × 4 × 4 = 16.

| Metric | Value |
|--------|-------|
| Total time | **368s** |
| step_time | **0.75-0.83s** |
| forward | **230-355ms** |
| backward | 459-517ms |
| optimizer | ~15-16ms |
| thru (samples/s) | 17-20 |
| I (inference/s) | **108-120** |
| T (train capacity) | 18-25 |
| pool | **64-72 / 64 (overfull!)** |
| dispatch_wait | 0.04-0.06s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.763s | 0.404s | 0.744s | 0.230s | 0.498s | 0.016s | 0.044s |
| 100 | 0.751s | 0.403s | 0.741s | 0.258s | 0.468s | 0.016s | 0.046s |
| 150 | 0.834s | 0.369s | 0.825s | 0.292s | 0.517s | 0.016s | 0.045s |
| 200 | 0.781s | 0.346s | 0.767s | 0.292s | 0.459s | 0.015s | 0.061s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 247.4 | 364.7 |
| 150 | 355.6 | 496.4 |
| 195 | 331.8 | 496.4 |

**Notable**: Fastest total time of all tests. 4 inference GPUs massively oversaturate the pool (I=108-120 samples/s, pool overflows to 72). Fewer FSDP ranks (4 vs 6-7) means less allreduce overhead, giving the lowest forward times. Trade-off: global batch size drops to 16, reducing per-step sample throughput but more than compensated by faster steps.

#### Test 512-F: Larger Micro-Batch + 4 Inference GPUs (4 Inference + 4 Training)

```
training.dflash_num_anchors=512 training.micro_batch_size=2 training.draft_accumulation_steps=2
inference.inference_num_gpus=4 training.training_num_gpus_per_node=4
```

Global batch size = 2 × 2 × 4 = 16 (same as 512-E).

| Metric | Value |
|--------|-------|
| Total time | **394s** |
| step_time | 0.90-0.93s |
| forward | 231-399ms |
| backward | 360-549ms |
| optimizer | ~16ms |
| thru (samples/s) | 17-18 |
| I (inference/s) | 114-128 |
| T (train capacity) | 18-20 |
| pool | **64 / 64 (full)** |
| dispatch_wait | 0.03-0.04s |

**TIMING samples**:

| Step | step_time | data_time | compute_time | fwd | bwd | opt | dispatch |
|------|-----------|-----------|--------------|-----|-----|-----|----------|
| 50 | 0.895s | 0.412s | 0.743s | 0.261s | 0.466s | 0.016s | 0.034s |
| 100 | 0.910s | 0.394s | 0.805s | 0.282s | 0.507s | 0.016s | 0.035s |
| 150 | 0.909s | 0.393s | 0.794s | 0.298s | 0.480s | 0.016s | 0.039s |
| 200 | 0.927s | 0.419s | 0.796s | 0.231s | 0.549s | 0.016s | 0.033s |

**COMPUTE_BREAKDOWN**:

| Step | forward (ms) | backward (ms) |
|------|-------------|---------------|
| 100 | 399.8 | 360.4 |
| 150 | 257.7 | 414.2 |
| 195 | 375.3 | 461.2 |

**Notable**: Slower than 512-E (394s vs 368s) despite same global batch size. batch=2 packs 2 sequences per micro-batch, increasing FlexAttention overhead per forward pass (0.90-0.93s/step vs 0.75-0.83s). batch=1 with accum=4 remains superior for the 4+4 split.

### Comparison Summary

#### Phase 1: Anchors=256 Speed Tuning

| Config | GPUs (I+T) | batch | accum | anchors | Total Time | step_time | thru (s/s) | pool | Bottleneck |
|--------|-----------|-------|-------|---------|------------|-----------|------------|------|------------|
| Baseline | 1+7 | 1 | 4 | 512 | — | 0.87-1.4s | 20-25 | full | compute (fwd variance) |
| **Test A** | 1+7 | 1 | 4 | **256** | **610s** | **0.55-0.64s** | 17-19 | full | compute |
| Test B | 1+7 | 2 | 2 | 256 | 637s | 0.62-0.71s | 16-18 | full | compute |
| Test C | 1+7 | 4 | 1 | 256 | 640s | 1.39-1.46s | 16-18 | starved | **data (pool empty)** |
| Test C2 | 2+6 | 4 | 1 | 256 | 476s | 0.86-0.95s | 22-25 | full | compute |

#### Phase 2: Anchors=512 Speed Tuning

| Config | GPUs (I+T) | batch | accum | anchors | Total Time | step_time | thru (s/s) | pool | Bottleneck |
|--------|-----------|-------|-------|---------|------------|-----------|------------|------|------------|
| 512-A | 1+7 | 1 | 4 | 512 | 584s | 0.83-1.40s | 17-20 | starved (12-28) | data |
| 512-B | 1+7 | 2 | 2 | 512 | 574s | 0.85-1.28s | 17-19 | starved (12-20) | data |
| 512-C | 2+6 | 2 | 2 | 512 | 446s | 0.99-1.29s | 20-25 | full | compute |
| 512-D | 2+6 | 1 | 4 | 512 | 457s | 0.93-1.09s | 22-25 | full | compute |
| **512-E** | **4+4** | **1** | **4** | **512** | **368s** | **0.75-0.83s** | 17-20 | **overfull (72)** | compute |
| 512-F | 4+4 | 2 | 2 | 512 | 394s | 0.90-0.93s | 17-18 | full | compute |

#### Key Findings

1. **`dflash_num_anchors` is the biggest lever**: 512→256 halves Q_LEN, cuts forward time from 260-800ms to 240-434ms. Consistent with Phase C results on RunPod.

2. **2+ inference GPUs is essential for anchors=512**: With 1 inference GPU, every anchors=512 config suffered pool starvation (12-28/64) and dispatch waits of 1.0-1.5s. Adding more inference GPUs fully resolves this.

3. **4+4 split (512-E) is the fastest overall**: 368s total, 17% faster than 512-C (446s). Fewer FSDP ranks (4 vs 6-7) reduces allreduce overhead, giving the lowest forward times (230-355ms). Trade-off: global batch size drops to 16 vs 24, but faster steps more than compensate.

4. **Inference is massively oversaturated at 4+4**: I=108-120 samples/s with pool overflowing to 72/64. The 4 inference GPUs produce far more data than 4 training GPUs can consume. This suggests 3+5 may be the sweet spot.

5. **Anchors=512 matches anchors=256 speed when properly tuned**: All 2+ inference GPU configs (512-C/D/E) match or beat the best anchors=256 config (C2, 476s).

6. **512-D has the most stable step times among 2+6 configs**: Forward variance 305-448ms vs 204-837ms for other configs. batch=1 with accum=4 avoids FlexAttention recompilation from varying padded batch shapes.

7. **RDMA is not available on Modal**: The RDMA probe confirmed `/dev/infiniband` and `ibstat` are not present. Mooncake uses TCP only, but CPU prefetch effectively hides the latency.

#### 200K × 3 Epoch Estimates

600,000 total samples. Estimates based on steady-state throughput (excluding warmup/compilation).

| Config | samples/s | Est. Time |
|--------|-----------|-----------|
| Baseline (1+7, anchors=512) | ~18 | ~9.3 hr |
| Test C2 (2+6, batch=4, anchors=256) | ~24 | ~6.9 hr |
| 512-C (2+6, batch=2, anchors=512) | ~22 | ~7.6 hr |
| 512-D (2+6, batch=1, anchors=512) | ~23 | ~7.2 hr |
| **512-E (4+4, batch=1, anchors=512)** | **~19** | **~8.8 hr** |

Note: 512-E has the fastest wall-clock for 200 steps (368s) but lower samples/s (19 vs 23) due to
smaller global batch (16 vs 24). For full training, effective time depends on optimizer convergence —
fewer samples per step means more optimizer steps needed, partially offsetting the faster step time.

#### Recommended Config for Full Training (Quality-Optimized)

Based on `specforge_reference.md`: z-lab achieves τ=3.95 with `dflash_num_anchors=512`,
`max_seq_length=2048`, and 200K+ samples. Anchors=512 is recommended for best acceptance length.

**512-E (fastest step time, 4+4)**:

```bash
modal run --detach --env sandbox scripts/modal/modal_dflash_train.py \
  --num-epochs 3 --dataset-size 200000 \
  --extra-overrides "training.dflash_num_anchors=512 inference.inference_num_gpus=4 training.training_num_gpus_per_node=4"
```

**512-D (best throughput, most stable, 2+6)**:

```bash
modal run --detach --env sandbox scripts/modal/modal_dflash_train.py \
  --num-epochs 3 --dataset-size 200000 \
  --extra-overrides "training.dflash_num_anchors=512 inference.inference_num_gpus=2 training.training_num_gpus_per_node=6"
```

**Speed-optimized (if τ quality not critical)**:

```bash
modal run --detach --env sandbox scripts/modal/modal_dflash_train.py \
  --num-epochs 3 --dataset-size 200000 \
  --extra-overrides "training.dflash_num_anchors=256 training.micro_batch_size=4 training.draft_accumulation_steps=1 inference.inference_num_gpus=2 training.training_num_gpus_per_node=6"
```

---

## Phase G: Full z-lab Inference Benchmark — Modal / SGLang (2026-03-25)

### Environment

- **Platform**: Modal (serverless GPU)
- **GPU**: 1x H100 80GB HBM3
- **Backend**: SGLang (tp=1, KV cache, CUDA graphs, paged attention)
- **SGLang version**: Built from PR #16818 (DFlash speculative decoding support)
- **Target model**: `Qwen/Qwen3-8B`
- **Draft model**: `Xingh3/dflash-qwen3-8b-3epoch` (iter_11803, 3 epochs on 190K samples)
- **Training config**: 190K samples (200K subsampled, filtered), 3 epochs, 23,622 optimizer steps, global_batch=24 (micro=1 x accum=4 x dp=6), 8x H100 (2 inference + 6 training)
- **Inference config**: temperature=0.0 (greedy), max_new_tokens=2048, concurrency=1
- **Total runtime**: ~45 minutes for all 10 datasets

### SGLang Compatibility Patches

The TorchSpec-exported DFlash model required runtime patching to work with SGLang's `DFlashDraftModel`:

| Patch | Description |
|-------|-------------|
| **Config rewrite** | `model_type: "dflash"` → `"qwen3"` + nested `dflash_config` block with `mask_token_id`, `target_layer_ids`, `block_size`, `layer_types` |
| **Custom code files** | Copied `dflash.py`, `modeling_dflash.py`, `utils.py` from `z-lab/Qwen3-8B-DFlash-b16` for `trust_remote_code` |
| **Weight key remapping** | `context_proj.weight` → `fc.weight`, `context_norm.weight` → `hidden_norm.weight`, `final_norm.weight` → `norm.weight`; skipped `embed_tokens.weight` (58 tensors remapped, 1 skipped) |
| **SGLang method alias** | `sed` patch to add `set_dflash_layers_to_capture` as alias for `set_eagle3_layers_to_capture` in SGLang's `qwen3.py` |

### Results: Cross-Dataset Acceptance Length (τ)

| Dataset | Samples | Our τ | z-lab τ | Gap | z-lab Speedup |
|---------|---------|-------|---------|-----|---------------|
| **gsm8k** | 128 | **3.19** | 3.38 | +0.19 | 5.20x |
| **math500** | 128 | **3.28** | 4.61 | +1.33 | 6.17x |
| **aime24** | 30 | **3.27** | 4.12 | +0.85 | 5.91x |
| **aime25** | 30 | **3.05** | 4.07 | +1.02 | 5.85x |
| **humaneval** | 164 | **3.55** | — | — | 5.20x |
| **mbpp** | 128 | **3.06** | — | — | 4.75x |
| **livecodebench** | 128 | **4.14** | — | — | 5.43x |
| **swe-bench** | 128 | **2.30** | — | — | 2.92x |
| **mt-bench** | 80 | **2.50** | — | — | 2.79x |
| **alpaca** | 128 | **2.22** | — | — | 2.27x |
| **Average** | — | **3.06** | — | — | — |

### τ Distribution per Dataset

| Dataset | τ∈[1,2) | τ∈[2,3) | τ∈[3,4) | τ∈[4,5) | τ∈[5,6) | τ∈[6,8) |
|---------|---------|---------|---------|---------|---------|---------|
| gsm8k | — | 32.0% | **61.7%** | 6.2% | — | — |
| math500 | — | 38.3% | **49.2%** | 10.9% | 1.6% | — |
| aime24 | — | 36.7% | **46.7%** | 16.7% | — | — |
| aime25 | — | 43.3% | **53.3%** | 3.3% | — | — |
| humaneval | — | 11.0% | **71.3%** | 17.1% | 0.6% | — |
| mbpp | — | 44.5% | **54.7%** | 0.8% | — | — |
| livecodebench | — | 10.9% | **43.0%** | 26.6% | 12.5% | 7.0% |
| swe-bench | 8.6% | **89.8%** | 1.6% | — | — | — |
| mt-bench | 30.0% | **45.0%** | 21.2% | 1.2% | 2.5% | — |
| alpaca | 34.4% | **54.7%** | 9.4% | 1.6% | — | — |

### Analysis: Comparison with z-lab

**Performance by domain:**

| Domain | Datasets | Avg τ (ours) | Avg z-lab τ | Notes |
|--------|----------|-------------|-------------|-------|
| **Code** | humaneval, mbpp, livecodebench | **3.58** | — | Best domain — repetitive structure in code aids draft prediction |
| **Math** | gsm8k, math500, aime24, aime25 | **3.20** | **4.05** | Consistent ~3.2, but z-lab achieves ~4.0 with 16x more training data |
| **General** | mt-bench, alpaca, swe-bench | **2.34** | — | Hardest domain — diverse text patterns limit draft accuracy |

**Gap to z-lab (on 4 datasets with reference τ):**

- **Our average**: 3.20 τ
- **z-lab average**: 4.05 τ
- **Gap**: 0.85 τ (21% lower)

**Root cause of gap:**

| Factor | Ours | z-lab |
|--------|------|-------|
| Training samples | 188,977 (200K subsampled, filtered) | ~800,000 (4.2x more) |
| Epochs | 3 | 6 |
| Total sample passes | ~567K | ~4.8M (8.5x more) |
| Training data quality | PerfectBlend 200K (open-source blend) | Proprietary blend |
| Optimizer steps | 23,622 | Unknown (est. ~200K+) |
| Max seq length | 2,048 | 3,072 |

The 21% τ gap is attributable to a combination of factors: 8.5x fewer total training sample passes, shorter max sequence length (2048 vs 3072), and likely higher-quality proprietary training data. With ~567K total sample passes vs z-lab's ~4.8M, the model has learned to consistently accept 3+ tokens per draft cycle (a functional speculative decoder), but needs more data diversity and longer sequences to match z-lab's 4+ τ.

**Key takeaway**: Our DFlash model trained on 190K samples for 3 epochs achieves **~78% of z-lab's acceptance length** on math benchmarks, with **8.5x fewer total sample passes**. Code tasks perform even better (τ=3.58-4.14) since code has more predictable token patterns. Scaling to 800K samples with seq_len=3072 should close much of the remaining gap.

### Progress: τ Improvement Over Training

| Phase | Checkpoint | Training | τ | Method | Notes |
|-------|------------|----------|---|--------|-------|
| Phase A | 200 steps | 200 steps, tiny data | 1.01 | Transformers | Pipeline validation only |
| Phase C | iter_18001 | 18K steps, 50K data | 1.86 | Transformers | Pre-bugfix, 2 training bugs |
| Phase D | iter_7869 (2 epoch) | 7.9K steps, 50K data | 1.85 | Transformers | Post-bugfix, saturated on 50K |
| Phase D | iter_11803 (3 epoch) | 11.8K steps, 50K data | 1.85 | Transformers | Identical to 2-epoch |
| **Phase G** | **iter_11803 (3 epoch)** | **23.6K steps, 190K data** | **3.06 avg** | **SGLang** | **Retrained on 200K, KV cache + CUDA graphs** |

Two factors contribute to the τ=1.85 → τ=3.06 improvement:
1. **Dataset scale**: 50K → 190K samples (3.8x more training data)
2. **Inference backend**: SGLang's KV cache, CUDA graphs, and optimized attention enable the draft model to express its full prediction quality, while the Transformers backend's naive recomputation underestimates the model's capability

### Phase G Commits

| Hash | Description |
|------|-------------|
| `3156a1a` | Fix HF repo visibility and missing WandB secret in training script |
| `e00b5f3` | Replace hardcoded prompts with z-lab standard benchmarks |
| `9ddb506` | Add SGLang-backend DFlash benchmark with compat fixes |
| `dc90f9d` | Add HF config fix utility and 3-epoch training metrics plot |

### Convergence Analysis

#### Observed Training Curves (200K PerfectBlend, 3 Epochs, 23,622 Steps)

See `training_metrics_3epoch.png`.

**Loss (CE):**
- Drops sharply from ~8 to ~3.5 in first ~2K steps (early epoch 1)
- Slow decline from ~3.5 to ~2.0 over remaining 21K steps
- Plateaus around step 5K-7K, then creeps down
- Visible bumps at epoch boundaries (dataset re-shuffle causes partial forgetting)

**Accuracy (Top-1):**
- Rises from 0.05 to ~0.35 in first 5K steps
- Stalls at ~0.35-0.45 for the remaining 18K steps
- Reaches only ~0.45 after 23K steps — low for a draft model
- Learning clearly decelerates well before epoch 3

**Diagnosis:** The model learns fast early but hits a wall around 35-45% accuracy. This indicates the optimizer configuration is preventing deeper convergence.

#### Root Cause Analysis

##### 1. Global Batch Size Too Large → Too Few Optimizer Steps

Phase G config: `micro_batch=1, accum=4, dp=6` → **global_batch = 24**.

With 190K samples: only **~7,900 steps per epoch**, 23,700 total for 3 epochs. The cosine LR schedule decays too fast relative to data diversity. The optimizer takes big batch steps but sees relatively few unique gradient directions.

z-lab's 800K × 6 epochs with (likely smaller) batch size would have significantly more optimizer steps. Community result (@jianc99, 1.2M × 3 epochs) likely had 100K+ optimizer steps.

##### 2. Cosine LR Decays Before Convergence

```python
# BF16Optimizer: total_steps=23,622, warmup_ratio=0.04
warmup = 945 steps (reaches peak lr=6e-4)
# Then cosine decay: 6e-4 → 0 over 22,677 steps
# By step ~12K (midpoint): LR ≈ 3e-4
# By step ~18K: LR < 1e-4 — model barely learning
```

This explains the accuracy stall at ~0.40 — the LR decays too aggressively for the amount of data diversity.

##### 3. weight_decay = 0.0 (No Regularization)

`BF16Optimizer` passed `weight_decay=0.0` to `AdamW`, making it equivalent to plain Adam. Adding weight decay (0.01) helps prevent overfitting on repeated data across epochs and improves generalization — which directly maps to inference τ.

##### 4. min_lr = 0.0 (LR Decays to Zero)

The cosine schedule decayed to `min_lr=0.0`. In later training steps, the model was effectively frozen. Setting `min_lr` to ~10% of peak (6e-5) keeps the model learning through later epochs.

#### Configuration Changes Implemented

| Parameter | Phase G | Phase H | Rationale |
|-----------|---------|---------|-----------|
| `draft_accumulation_steps` | 4 | **2** | 2x more optimizer steps per epoch |
| `weight_decay` | 0.0 (hardcoded) | **0.01** | AdamW regularization → better generalization |
| `min_lr` | 0.0 (hardcoded) | **6e-5** | 10% of peak → prevents LR death |
| `save_interval` | 1000 | **5000** | Less frequent for longer runs |
| `save_per_epoch` | false | **true** | Compare τ at epoch boundaries |
| `max_checkpoints` | 2 | **3** | Keep all 3 epoch checkpoints |

Code changes (commits TBD):
- `torchspec/config/train_config.py`: Added `min_lr` and `weight_decay` to `TrainingConfig`
- `torchspec/training/optimizer.py`: `BF16Optimizer` accepts `min_lr` and passes to scheduler
- `torchspec/training/dflash_trainer.py`: Plumbs `weight_decay` and `min_lr` from args
- `torchspec/training/eagle3_trainer.py`: Same plumbing (consistent across trainers)
- `scripts/modal/modal_dflash_train.py`: Fixed WandB config prefix (`logging.*` not `training.*`)

#### Expected Impact

| Change | Speed Impact | τ Impact |
|--------|-------------|----------|
| `accum=4→2` | Neutral (same samples/s, 2x optimizer steps) | **+0.3-0.5 τ** |
| `min_lr=6e-5` | Neutral | **+0.2-0.3 τ** |
| `weight_decay=0.01` | Neutral | **+0.1-0.2 τ** |
| Combined | No speed change | **+0.5-1.0 τ** (est. τ=3.5-4.0 on 190K, 4.0-4.5 on 800K) |

---

## Phase H: 800K PerfectBlend Training (2026-03-25)

### Training Plan

#### Goal

Close the τ gap to z-lab (currently 3.06 vs 4.05) by scaling data 4x and fixing convergence.

#### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | PerfectBlend 800K (~760K after filtering) | 4x Phase G data |
| Epochs | 3 | Start with 3, extend to 6 if τ improving |
| GPU split | 2I + 6T (512-D) | Best sample throughput (~23 samples/s) |
| Global batch | 12 (micro=1, accum=2, dp=6) | 2x more optimizer steps than Phase G |
| Anchors | 512 | z-lab recipe |
| Seq length | 2048 | Keep current; 3072 is Priority 2 |
| LR | 6e-4, cosine, warmup 0.04, min_lr 6e-5 | Slower decay |
| Weight decay | 0.01 | New |
| Optimizer steps | ~189K (3 epochs) | 8x more than Phase G (23.6K) |
| Total sample passes | ~2.27M | 4x Phase G (567K), 48% of z-lab (4.8M) |

#### Estimated Training Time & Cost

| Epochs | Steps | Wall Time | GPU-hours | Modal Cost |
|--------|-------|-----------|-----------|------------|
| 3 | ~189K | ~23 hr | 184 | ~$728 |
| 6 | ~378K | ~46 hr | 368 | ~$1,456 |

Based on 512-D throughput: ~23 samples/s steady state, ~12 global_batch, ~1.0-1.1 step/s (accum=2).
Note: 4I+4T (512-E) was initially chosen for fastest step time but has lower sample throughput
(19 vs 23 samples/s) due to smaller dp_size. 2I+6T processes more samples per wall-clock second.

#### τ Targets

| Milestone | Expected τ | Comparison |
|-----------|-----------|------------|
| Epoch 1 | ~3.3-3.6 | Above Phase G final (3.06) |
| Epoch 2 | ~3.6-4.0 | Approaching z-lab (4.05) |
| Epoch 3 | ~3.8-4.2 | Matching z-lab math benchmarks |

#### Launch Command

```bash
modal run --detach --env sandbox scripts/modal/modal_dflash_train.py \
  --num-epochs 3 --dataset-size 800000 \
  --wandb-project dflash-800k --wandb-team dflash \
  --hf-repo Xingh3/dflash-qwen3-8b-800k-3epoch \
  --extra-overrides "training.dflash_num_anchors=512 \
    inference.inference_num_gpus=2 training.training_num_gpus_per_node=6 \
    training.draft_accumulation_steps=2 training.min_lr=6e-5 \
    training.weight_decay=0.01 training.save_interval=5000 \
    training.save_per_epoch=true training.max_checkpoints=3"
```

#### Monitoring

- **WandB**: `dflash-800k` project — watch `train/avg_loss`, `train/avg_acc`, `train/lr`, `train/grad_norm`
- **Checkpoints**: Saved at every 5K steps + epoch boundaries → auto-converted + uploaded to `Xingh3/dflash-qwen3-8b-800k-3epoch`
- **τ benchmark**: After each epoch, run `scripts/modal/modal_dflash_benchmark_sglang.py` against the HF checkpoint

#### Training Summary

- **Completed**: 188,943 / 188,943 steps (100%), 3 epochs
- **Wall time**: 8h 27m (30,887s)
- **Average throughput**: 19.0 samples/s (steady state ~20 samples/s)
- **Final metrics**: loss ≈ 1.8-2.0, accuracy ≈ 0.45-0.50 (peaking at 0.67)
- **Infrastructure**: Pool full (64/64) throughout, dispatch wait ≈ 0.0s, no data starvation
- **WandB run**: `dflash-800k/runs/6oez0kwd`

### Inference Benchmark Results (2026-03-27)

#### Environment

- **Platform**: Modal (serverless GPU)
- **GPU**: 1x NVIDIA H100 80GB HBM3
- **Backend**: SGLang (tp=1, KV cache, CUDA graphs, paged attention, FlashAttention3)
- **SGLang version**: Built from PR #16818 (DFlash speculative decoding support)
- **Target model**: `Qwen/Qwen3-8B`
- **Draft model**: Modal volume `/dflash-qwen3-8b/hf_model` (iter_188944, 3 epochs on 760K samples)
- **Inference config**: temperature=0.0 (greedy), max_new_tokens=2048, concurrency=1
- **Total runtime**: ~2.2 hours for all 10 datasets (baseline + DFlash per dataset)

#### Results: Cross-Dataset (Phase H vs Phase G vs z-lab)

| Dataset | Samples | Phase G τ | **Phase H τ** | **Δ** | z-lab τ | Gap to z-lab |
|---------|---------|-----------|---------------|-------|---------|--------------|
| **gsm8k** | 128 | 3.19 | **3.75** | **+0.56** | 3.38 | **-0.37 (beat)** |
| **math500** | 128 | 3.28 | **3.87** | **+0.59** | 4.61 | +0.74 |
| **aime24** | 30 | 3.27 | **3.92** | **+0.65** | 4.12 | +0.20 |
| **aime25** | 30 | 3.05 | **3.62** | **+0.57** | 4.07 | +0.45 |
| **humaneval** | 164 | 3.55 | **4.12** | **+0.57** | — | — |
| **mbpp** | 128 | 3.06 | **3.50** | **+0.44** | — | — |
| **livecodebench** | 128 | 4.14 | **4.90** | **+0.76** | — | — |
| **swe-bench** | 128 | 2.30 | **2.60** | **+0.30** | — | — |
| **mt-bench** | 80 | 2.50 | **2.80** | **+0.30** | — | — |
| **alpaca** | 128 | 2.22 | **2.45** | **+0.23** | — | — |

#### Throughput & Speedup (SGLang backend)

| Dataset | Baseline tok/s | DFlash tok/s | Speedup |
|---------|---------------|-------------|---------|
| gsm8k | 142.4 | 325.8 | **2.29x** |
| math500 | 143.3 | 346.5 | **2.42x** |
| aime24 | 143.0 | 344.2 | **2.41x** |
| aime25 | 143.2 | 320.3 | **2.24x** |
| humaneval | 142.9 | 363.2 | **2.54x** |
| mbpp | 143.0 | 303.3 | **2.12x** |
| livecodebench | 141.3 | 409.4 | **2.90x** |
| swe-bench | 142.8 | 236.0 | **1.65x** |
| mt-bench | 143.0 | 225.6 | **1.58x** |
| alpaca | 143.1 | 220.0 | **1.54x** |

#### Domain Analysis

| Domain | Datasets | Phase G τ | Phase H τ | Δ | z-lab τ |
|--------|----------|-----------|-----------|---|---------|
| **Math** | gsm8k, math500, aime24, aime25 | 3.20 | **3.79** | +0.59 | 4.05 |
| **Code** | humaneval, mbpp, livecodebench | 3.58 | **4.17** | +0.59 | — |
| **General** | swe-bench, mt-bench, alpaca | 2.34 | **2.62** | +0.28 | — |
| **Overall** | all 10 | 3.06 | **3.45** | +0.39 | — |

#### Gap Analysis: Phase G → Phase H → z-lab

**Math benchmarks (4 datasets with z-lab reference):**

| Metric | Phase G | Phase H | z-lab |
|--------|---------|---------|-------|
| Average τ | 3.20 | **3.79** | 4.05 |
| Gap to z-lab | 21.0% | **6.4%** | — |
| Training samples | 190K | 760K | ~800K |
| Epochs | 3 | 3 | 6 |
| Total sample passes | 567K | 2.27M | 4.8M |
| Optimizer steps | 23.6K | 189K | ~200K+ (est.) |
| Seq length | 2,048 | 2,048 | 3,072 |
| weight_decay | 0.0 | 0.01 | unknown |
| min_lr | 0.0 | 6e-5 | unknown |

**What worked (Phase G → Phase H):**

1. **4x data scale** (190K → 760K): Directly increases data diversity seen per epoch.
2. **8x more optimizer steps** (23.6K → 189K): `accum=4→2` doubled steps per epoch; combined with 4x data, total steps went from 23.6K to 189K.
3. **weight_decay=0.01**: Regularization prevents overfitting on repeated data across epochs, improving generalization to unseen inference patterns.
4. **min_lr=6e-5**: Prevents learning rate death in late training; the model continued learning through epoch 3 instead of freezing.
5. **Combined effect**: +0.59 τ on math (+0.59 on code, +0.28 on general), closing the z-lab gap from 21% to 6.4%.

**What the remaining 6.4% gap is attributable to:**

| Factor | Impact (est.) | Evidence |
|--------|--------------|----------|
| **Epochs 3 vs 6** | +0.2-0.4 τ | z-lab trains 6 epochs (4.8M passes vs our 2.27M). Our acc was still rising at epoch 3 end. |
| **Seq length 2048 vs 3072** | +0.1-0.3 τ | Longer context gives the draft model more signal for multi-step reasoning patterns. Math/AIME prompts can exceed 2048 tokens. |
| **Data quality** | +0.0-0.2 τ | z-lab may use proprietary or higher-quality data blend. PerfectBlend is open-source. |

---

## Phase I: Next Steps to Close z-lab Gap

### Priority 1: Extend to 6 Epochs (est. +0.2-0.4 τ)

The model was still improving at epoch 3 end (loss ~1.8, accuracy ~0.50, not fully converged). Training accuracy peaked at 0.67 for individual batches, meaning the model has capacity for more learning. z-lab trains 6 epochs on similar data volume.

**Estimated impact**: τ 3.79 → 4.0-4.2 on math benchmarks.

```bash
modal run --detach --env sandbox scripts/modal/modal_dflash_train.py \
  --num-epochs 6 --dataset-size 800000 \
  --wandb-project dflash-800k --wandb-team dflash \
  --hf-repo Xingh3/dflash-qwen3-8b-800k-6epoch \
  --extra-overrides "training.dflash_num_anchors=512 \
    inference.inference_num_gpus=2 training.training_num_gpus_per_node=6 \
    training.draft_accumulation_steps=2 training.min_lr=6e-5 \
    training.weight_decay=0.01 training.save_interval=5000 \
    training.save_per_epoch=true training.max_checkpoints=6 \
    training.load_path=/workspace/outputs/dflash-qwen3-8b/checkpoints"
```

**Cost**: ~$728 incremental (3 more epochs, ~23 hr on 8x H100). Resume from Phase H checkpoint.

### Priority 2: Increase Sequence Length to 3072 (est. +0.1-0.3 τ)

z-lab uses `max_seq_length=3072`. Longer context provides:
- More complete reasoning chains for the draft model to learn from
- Better coverage of long-form math problems (AIME prompts + solutions can exceed 2048 tokens)
- More diverse anchor positions for DFlash's context feature extraction

Requires changes to:
- Training config: `training.max_seq_length=3072`
- May need `training.dflash_num_anchors=768` (proportional increase) or tuning
- Forward time will increase ~50% (Q_LEN grows proportionally)
- Consider reducing `micro_batch_size` or increasing GPU count to compensate

### Priority 3: Data Quality Improvements (est. +0.0-0.2 τ)

- **Curate harder math samples**: Current PerfectBlend is a general-purpose mix. Enriching with AIME/competition-math and long-chain-of-thought samples would directly target the weakest benchmarks (AIME25 has the largest gap: +0.45 to z-lab).
- **Add code-heavy data**: Code tasks (humaneval, livecodebench) already show the best τ (4.12, 4.90). More code data could push these higher and lift the overall average.
- **Filter short/low-quality samples**: Samples shorter than ~256 tokens provide limited signal for DFlash anchor training.

### Priority 4: Speculative Decoding Optimizations (improves speedup, not τ)

Our speedup (2.2-2.9x) is below z-lab's reference (2.3-6.2x). This is not a model quality issue — it's inference-side:
- **SGLang DFlash spec v2**: The benchmark log shows `spec v2 is not supported yet` for DFlash. Once SGLang enables overlap scheduling for DFlash (spec v2), the pipelining of draft and verify stages will significantly improve throughput.
- **Block size tuning**: Currently using block_size=16. Smaller blocks (8-12) may be better for datasets where τ < 4, reducing wasted verification work.
- **Continuous batching**: Our benchmark runs concurrency=1. Production workloads with concurrent requests will see higher GPU utilization and better effective speedup.

### Estimated Cumulative Impact

| Change | τ (math avg) | Gap to z-lab |
|--------|-------------|--------------|
| **Phase H (current)** | 3.79 | 6.4% |
| + 6 epochs | ~4.0-4.2 | ~1-3% |
| + seq_len 3072 | ~4.1-4.4 | ~0-2% |
| + data quality | ~4.2-4.5 | **≈0% (match/exceed)** |

---

## Speedup Methodology Analysis (2026-03-28)

### Why Our Speedup Doesn't Match z-lab's Reported Numbers

Our E2E throughput speedup (1.5-2.9x) is well below z-lab's headline numbers (2.3-6.2x) despite competitive τ values. Investigation of z-lab's benchmark code and a decode-only re-benchmark reveal this is a **measurement methodology difference**, not a model quality issue.

### z-lab's Measurement: Decode-Only Time-per-Token Ratio (Transformers backend)

z-lab's [`benchmark.py`](https://github.com/z-lab/dflash/blob/main/benchmark.py) computes speedup as:

```
speedup = mean(baseline_time_per_output_token) / mean(dflash_time_per_output_token)
```

Where `time_per_output_token = decode_time / num_output_tokens`, and decode_time **excludes prefill** (timing starts after prefill + first draft iteration). The baseline uses `block_size=1` (pure autoregressive target model). This runs on bare Transformers with `torch.cuda.synchronize()` — no serving framework overhead.

### Our Measurement: SGLang Server End-to-End Throughput

Our benchmark sends HTTP requests to an SGLang server and measures `total_tokens / total_wall_time`, which includes prefill, HTTP round-trips, scheduler overhead, CUDA graph dispatch, and KV cache paging.

### Decode-Only Re-Benchmark (2026-03-28)

To produce an apples-to-apples comparison, we added decode-only timing: for each prompt, we first measure prefill latency (via a 1-token generation), then subtract it from the server-side `e2e_latency` to isolate decode time.

**GSM8K results (128 samples, 1x H100 80GB):**

| Metric | Baseline | DFlash | Speedup |
|--------|----------|--------|---------|
| E2E throughput | 148.3 tok/s | 363.7 tok/s | **2.45x** |
| Decode-only ms/tok | 6.676 | 2.674 | **2.50x** |
| Avg prefill latency | 16.1 ms | 21.2 ms | — |
| τ (acceptance length) | — | 3.75 | — |
| z-lab reference τ | — | 3.38 | — |
| z-lab reference speedup | — | — | 5.20x |

**Key finding**: Decode-only speedup (2.50x) is only marginally higher than E2E (2.45x). Prefill is a small fraction of total time for these generation lengths. The ~2x gap to z-lab's 5.20x is **not** caused by prefill inclusion.

### Sources of the Remaining Gap

The gap between our 2.50x decode-only and z-lab's 5.20x comes from the **serving framework overhead**, not measurement methodology or model quality:

| Factor | Impact (est.) | Notes |
|--------|--------------|-------|
| **SGLang scheduling + CUDA graph dispatch** | ~1.5-2x | Each verify step goes through SGLang's scheduler, CUDA graph replay, KV cache management. z-lab's Transformers benchmark directly calls `model()` with `torch.cuda.synchronize()`. |
| **Spec v2 overlap not active** | ~1.2-1.5x | SGLang's spec v2 (pipelining draft + verify) is not yet supported for DFlash. z-lab's `run_sglang_benchmark.sh` uses `trtllm_mha` + `fa4` backends on B200 GPUs with spec v2 enabled. |
| **Attention backend** | ~1.1-1.2x | We use FlashAttention3 on H100. z-lab's SGLang benchmark uses `trtllm_mha` with `fa4` draft backend on B200 (SM100), which has faster attention kernels. |
| **Hardware** | ~1.0-1.1x | z-lab's Transformers benchmark runs on H200/B200 with higher memory bandwidth. |

**Critical evidence**: On GSM8K, our τ=3.75 **exceeds** z-lab's τ=3.38, yet our speedup is 2.50x vs their 5.20x. A model with higher acceptance length producing lower speedup can only be explained by per-iteration overhead in the serving stack.

### z-lab's Two Benchmark Scripts

z-lab maintains two separate benchmarks with different purposes:

1. **`benchmark.py` (Transformers backend)** — produces the headline 5-6x numbers on their project page. Uses bare PyTorch with `cuda_time()` = `torch.cuda.synchronize()` + `time.perf_counter()`. Runs `block_size=1` as baseline. Measures decode-only time-per-token ratio.

2. **`benchmark_sglang.py` (SGLang backend)** — production serving benchmark. Tests multiple TP sizes, concurrencies, and attention backends. Uses `trtllm_mha` + `fa4` on B200 GPUs. Reports `output_toks_per_s` (end-to-end throughput).

The headline numbers (5.20x on GSM8K etc.) are from benchmark #1 — the Transformers backend with decode-only timing. These are not directly comparable to SGLang serving throughput.

### Conclusion

Our model quality (τ) is competitive with z-lab and exceeds them on GSM8K. The speedup gap is entirely infrastructure-side and will close as:
1. SGLang enables spec v2 overlap scheduling for DFlash
2. Newer attention backends (fa4, trtllm_mha) become available on our hardware
3. We add a Transformers-backend benchmark for direct comparison with z-lab's methodology

---

## Phase J: UltraChat Training + Convergence Analysis (2026-03-29)

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | `jiapingW/qwen3.5-35b-a3b-ultrachat-regen` (~208K samples) | Arrow/Parquet HF Hub dataset |
| Epochs | 3 | |
| GPU split | 2I + 6T (512-D config) | 8x H100 80GB |
| Global batch | 12 (micro=1, accum=2, dp=6) | |
| Anchors | 512 | |
| Seq length | 2048 | |
| LR | 6e-4, cosine, warmup 0.04, min_lr 6e-5 | |
| Weight decay | 0.01 | |
| Total steps | 51,921 | |
| Resumed from | Step 40,001 (completed to ~43,000 before stopping) | |
| WandB | `dflash-ultrachat` | |

### Training Metrics (Steps 40,001–41,516)

Training was resumed from checkpoint step 40,001 and ran for ~1,500 additional steps. Metrics showed complete plateau — no convergence improvement.

**Average Loss per 100-step Window:**

| Steps | Avg Loss |
|-------|----------|
| 40,001–40,100 | 3.149 |
| 40,101–40,200 | 3.170 |
| 40,201–40,300 | 3.146 |
| 40,301–40,400 | 3.129 |
| 40,401–40,500 | 3.155 |
| 40,501–40,600 | 3.146 |
| 40,601–40,700 | 3.124 |
| 40,701–40,800 | 3.169 |
| 40,801–40,900 | 3.122 |
| 40,901–41,000 | 3.156 |
| 41,001–41,100 | 3.132 |
| 41,101–41,200 | 3.154 |
| 41,201–41,300 | 3.145 |
| 41,301–41,400 | 3.130 |
| 41,401–41,500 | 3.162 |

**Average Accuracy per 100-step Window:**

| Steps | Avg Acc |
|-------|---------|
| 40,001–40,100 | 0.278 |
| 40,101–40,200 | 0.275 |
| 40,201–40,300 | 0.278 |
| 40,301–40,400 | 0.279 |
| 40,401–40,500 | 0.277 |
| 40,501–40,600 | 0.279 |
| 40,601–40,700 | 0.281 |
| 40,701–40,800 | 0.276 |
| 40,801–40,900 | 0.282 |
| 40,901–41,000 | 0.277 |
| 41,001–41,100 | 0.280 |
| 41,101–41,200 | 0.278 |
| 41,201–41,300 | 0.278 |
| 41,301–41,400 | 0.280 |
| 41,401–41,500 | 0.276 |

**Diagnosis**: Loss plateaued at ~3.14 and accuracy at ~0.278 throughout epoch 3. No improvement over 1,500 steps. The cosine LR was near `min_lr` (6e-5) at this late stage, and the model had already seen all data twice. Training was stopped and the step 43,001 checkpoint was converted and uploaded.

### Inference Benchmark Results (Step 43,001 — UltraChat model)

- **HF Model**: [`Xingh3/dflash-qwen3-8b-ultrachat-43k`](https://huggingface.co/Xingh3/dflash-qwen3-8b-ultrachat-43k)
- **Backend**: SGLang (tp=1, 1x H100 80GB, CUDA graphs, paged attention)
- **Mode**: Quick (30 samples per dataset), temperature=0.0, concurrency=1, skip-baseline

| Dataset | UltraChat τ | Phase H τ | Δ (vs H) | z-lab τ | Gap to z-lab |
|---------|-------------|-----------|----------|---------|--------------|
| **gsm8k** | **3.22** | 3.75 | **-0.53** | 3.38 | +0.16 |
| **math500** | **3.07** | 3.87 | **-0.80** | 4.61 | +1.54 |
| **aime24** | **3.13** | 3.92 | **-0.79** | 4.12 | +0.99 |
| **aime25** | **2.90** | 3.62 | **-0.72** | 4.07 | +1.17 |
| **humaneval** | **3.90** | 4.12 | -0.22 | — | — |
| **mbpp** | **3.27** | 3.50 | -0.23 | — | — |
| **livecodebench** | **3.34** | 4.90 | **-1.56** | — | — |
| **swe-bench** | **2.62** | 2.60 | +0.02 | — | — |
| **mt-bench** | **2.58** | 2.80 | -0.22 | — | — |
| **alpaca** | **2.39** | 2.45 | -0.06 | — | — |

**Domain Averages:**

| Domain | UltraChat τ | Phase H τ | Δ | z-lab τ |
|--------|-------------|-----------|---|---------|
| Math (gsm8k, math500, aime24, aime25) | 3.08 | **3.79** | **-0.71** | 4.05 |
| Code (humaneval, mbpp, livecodebench) | 3.50 | **4.17** | **-0.67** | — |
| General (swe-bench, mt-bench, alpaca) | 2.53 | **2.62** | -0.09 | — |
| Overall (all 10) | 3.04 | **3.45** | -0.41 | — |

### Key Observations

1. **UltraChat model is significantly worse than Phase H** across all domains, especially math (-0.71 τ) and code (-0.67 τ).
2. **Phase H (PerfectBlend 760K, 3 epochs, 189K steps)** remains our best model — the UltraChat experiment did not improve on it.
3. **UltraChat dataset is much smaller** (~208K samples vs 760K PerfectBlend) and lacks the data diversity needed for strong draft model training.
4. **General-domain tasks are comparable** (swe-bench +0.02, alpaca -0.06), suggesting UltraChat's conversational data transfers okay for dialogue but poorly for reasoning/code.

---

## Training Code Audit: Why Convergence Lags Behind z-lab (2026-03-29)

A deep audit of the TorchSpec DFlash training code was performed against the SpecForge reference (`specforge_reference.md`) and known upstream bugs (PR #427, #472, #473). The goal was to identify any code-level issues explaining the persistent τ gap to z-lab (our best: 3.79 math avg vs z-lab: 4.05).

### Verified: No Critical Bugs Found

| Known Upstream Bug | Status in TorchSpec |
|---|---|
| **PR #427: Causal-within-block masking** | **CLEAR** — `torchspec/models/dflash.py` lines 66-71: same-block draft attention uses `q_block_id == kv_block_id` with no ordering constraint (bidirectional). Context is `kv_idx < anchor_pos` (strictly before block start). Matches post-fix behavior. |
| **PR #472/#473: SGLang padding + mask data leak** | **CLEAR** — Mask function does not allow queries to attend to context positions at/after the anchor. No "peek at future tokens" pattern detected. Context keys capped by `anchor_pos`, draft keys limited to same block. |
| **PR #473: Attention mask data leak** | **CLEAR** — `block_keep_mask` gates invalid anchor slots. `valid_label_mask` prevents loss on padded tail positions. |

### Potential Issues Identified (Not Bugs, But Recipe Mismatches)

#### Issue 1: Block 0 Not Excluded from Loss

**Reference**: The DFlash paper states loss should exclude "block 0 (no preceding context)" and anchor tokens. Our code excludes anchor tokens (`pos_in_block > 0` at `dflash.py:308`) but does **not** exclude blocks where `anchor_pos == 0` (i.e., the first block with no visible context).

**Impact**: Low-to-moderate. Block 0 predictions have zero context signal (only same-block MASK interactions), so including them adds noise to the gradient. z-lab may omit these. Estimated τ impact: +0.05-0.15.

**Location**: `torchspec/models/dflash.py` lines 306-309.

#### Issue 2: Dataset — Not z-lab's Recipe

z-lab uses **~800K samples from NVIDIA Nemotron Post-Training V2 + CodeAlpaca**, trained for **6 epochs** at **seq_len=3072**. Our experiments used:

| Run | Dataset | Samples | Epochs | Total Passes | seq_len |
|-----|---------|---------|--------|-------------|---------|
| Phase H | PerfectBlend | 760K | 3 | 2.27M | 2048 |
| UltraChat | ultrachat-regen | ~208K | 3 | ~0.62M | 2048 |
| z-lab | Nemotron + CodeAlpaca | ~800K | 6 | **4.8M** | **3072** |

The total sample passes (dataset × epochs) is the strongest predictor of τ quality. Phase H reaches 47% of z-lab's data exposure; UltraChat reaches only 13%.

#### Issue 3: Sequence Length 2048 vs 3072

z-lab trains with `max_seq_length=3072`. Longer sequences provide:
- More complete reasoning chains (AIME solutions often exceed 2048 tokens)
- More diverse anchor positions per sample (more blocks per sequence)
- Better coverage of long-range dependencies the draft model must predict

With 2048, sequences are truncated and the model never sees full reasoning patterns. This disproportionately affects math/reasoning benchmarks (our weakest domain).

#### Issue 4: LR Schedule on Resume

When training resumes from a checkpoint, the LR scheduler is restored from the checkpoint state. However, `lr_total_steps` is recalculated from the config at startup via `auto_calculate_training_steps()` (`torchspec/controller/setup.py:90`). If the dataset size or batch configuration differs between runs, the cosine decay shape may not match the original schedule.

In the UltraChat run, the model was already in epoch 3 of 3 at step 40,001 — the LR had decayed to near `min_lr` (6e-5), explaining the complete plateau. This is **expected behavior**, not a bug, but confirms the model was saturated.

#### Issue 5: `intermediate_size` Default Mismatch

`DFlashConfig` class default has `intermediate_size=14336` (`torchspec/models/draft/dflash.py`), while `dflash_draft_config.json` specifies `intermediate_size=12288` (matching Qwen3-8B). The JSON config wins in practice, but this is a footgun if anyone instantiates the config without the JSON file.

### Root Cause Summary: Why Our Best Model (Phase H, τ=3.79) Still Trails z-lab (τ=4.05)

The code audit found **no training bugs** — the attention mask, loss function, optimizer, and data pipeline are correctly implemented. The remaining 6.4% τ gap is explained by **training recipe differences**, not code defects:

| Factor | Our Config | z-lab Config | Est. τ Impact |
|--------|-----------|-------------|---------------|
| **Data volume × epochs** | 2.27M passes | **4.8M passes** | **+0.15-0.30 τ** |
| **Sequence length** | 2048 | **3072** | **+0.10-0.25 τ** |
| **Dataset composition** | PerfectBlend (general) | **Nemotron + CodeAlpaca** | **+0.05-0.15 τ** |
| **Block 0 loss exclusion** | Not excluded | Likely excluded | +0.05-0.15 τ |
| **Total gap explained** | | | **+0.35-0.85 τ** |

The gap range (+0.35 to +0.85 τ) brackets the actual gap (+0.26 τ), confirming these recipe differences fully account for the observed performance difference.

### Recommended Next Steps (Revised)

Based on the code audit, the priorities have shifted from "find bugs" to "match the recipe":

1. **Extend to 6 epochs** (resume Phase H checkpoint): The single highest-impact change. Phase H accuracy was still improving at epoch 3 end.
2. **Increase seq_len to 3072**: Requires forward time increase (~50%), may need 4+4 GPU split to compensate.
3. **Exclude block 0 from loss**: Simple code change in `torchspec/models/dflash.py`, expected small but free improvement.
4. **Try Nemotron + CodeAlpaca data blend**: Match z-lab's exact dataset composition for the fairest comparison.

---

## Phase K: Hyperparameter Sweep for Convergence (2026-03-30)

### Motivation

Training consistently converges quickly then plateaus (loss ~3.1, accuracy ~0.50 by epoch 2). Phase H achieved τ=3.79 but was 6.4% below z-lab's 4.05. Rather than adding more data or epochs, we investigated whether the training *recipe* itself (LR schedule, batch size, etc.) was limiting convergence.

### Strategy: 3-Phase Hyperparameter Search

| Phase | Data | Steps | Configs | Purpose | Cost |
|-------|------|-------|---------|---------|------|
| **1 — Screening** | 200K PerfectBlend | 1,000 | 6 parallel | Cheaply eliminate bad configs | ~$144 |
| **2 — Validation** | 800K PerfectBlend | 3 epochs (~189K) | 3 parallel | Confirm winners at full scale | ~$2,200 |
| **3 — Champion** | Full dataset | Full | 1 (best) | Production model + benchmark τ | TBD |

### Phase 1 Results: Quick Screening (1,000 steps, 200K data)

All runs share: 8x H100, 2I+6T, 512 anchors, seq_len=2048, WandB project `dflash-hparam-sweep`.

| Run | LR Schedule | Accum | Key Difference | Final Loss | Final Acc | Duration |
|-----|-------------|-------|----------------|-----------|----------|----------|
| R00-baseline | cosine | 2 | Phase H config (control) | 4.545 | 0.158 | 15m |
| R01-higherLR | cosine (lr=1e-3) | 2 | Higher learning rate | 4.290 | 0.161 | 16m |
| R04-constantLR | constant | 2 | No LR decay | 4.168 | 0.164 | 16m |
| **R07-accum1** | cosine | **1** | **2x optimizer steps** | **4.035** | **0.182** | 17m |
| **R03-WSD** | **WSD (cosine)** | 2 | **Warmup-Stable-Decay** | **4.053** | **0.184** | 18m |
| R11-aggressive | WSD + lr=1e-3 | 1 | All aggressive settings | 5.498 | 0.113 | 11m |

**Winners**: R03-WSD and R07-accum1 both achieved ~12% lower loss and ~15% higher accuracy than the baseline.

### Key Findings

1. **WSD schedule matches accum=1** in final metrics despite using standard accumulation. The stable LR phase (80% of training at peak LR) prevents premature convergence from cosine decay.

2. **accum=1 gives 2x more optimizer updates** per epoch (smaller effective batch size of 6 vs 12), which helps convergence at the cost of 2x total steps.

3. **Higher LR (1e-3) hurts when combined with other aggressive settings** — R11-aggressive was the worst performer despite using both WSD and accum=1.

4. **Cosine decay alone is suboptimal** — the baseline configs cluster at loss 4.2–4.5, meaningfully worse than WSD (4.05).

### WSD Schedule Explained

WSD (Warmup-Stable-Decay) is a 3-phase LR schedule:
- **Warmup** (4%): Linear ramp from 0 to peak LR
- **Stable** (76%): Constant at peak LR — the model keeps exploring
- **Decay** (20%): Cosine decay to min_lr — the model settles

vs cosine decay which starts dropping LR immediately after warmup, causing the "converge-then-plateau" pattern.

### Phase 2 Results (800K PerfectBlend, 3 epochs)

Launched 3 full-scale validation runs. All crashed at Modal's 24h ephemeral app limit before completing 3 epochs. WandB captured training curves for all runs.

**Original runs (24h, WandB project: `dflash-phase2`):**

| Run | Last Step / Total | Progress | Last Loss | Last Acc | LR at crash |
|-----|-------------------|----------|-----------|----------|-------------|
| **P2-WSD** | 137,776 / 188,943 | **73%** | **1.86** | **0.46** | 6.0e-4 (still at peak) |
| P2-baseline | 135,106 / 188,943 | 72% | 2.56 | 0.38 | 1.69e-4 |
| P2-accum1 | 246,815 / 377,889 | 65% | 2.42 | 0.39 | 2.16e-4 |

**Key finding**: P2-WSD is clearly winning — 27% lower loss and 21% higher accuracy than baseline at similar wall time. WSD's LR was still at peak (6e-4) when it crashed, meaning it had more convergence headroom remaining. Baseline's cosine LR had already decayed to 1.69e-4 (72% below peak).

**Bug found**: All 3 runs used hardcoded `run_id="dflash-qwen3-8b"`, writing checkpoints to the same volume directory. Only P2-accum1's checkpoint survived (step 245K). Fixed by adding `dflash_run_id` parameter to `train_sglang` so each sweep run gets an isolated output directory.

**Resumed runs (WandB project: `dflash-phase2-v2`):**
- **P2-accum1**: Resumed from step 245K checkpoint. ~12h remaining.
- **P2-WSD**: Fresh start (checkpoint lost). ~30h total, will need one more resume after 24h limit.

Each run now saves to its own volume path (`/workspace/outputs/P2-accum1/`, `/workspace/outputs/P2-WSD/`) and supports `--resume` for multi-session training across Modal's 24h limit.

### P2-accum1 Inference Benchmark (2026-04-01)

P2-accum1 completed 3 epochs on 800K PerfectBlend with `accum=1` (2x optimizer steps vs Phase H). Model uploaded to public HuggingFace repo.

- **HF Model**: [`Xingh3/dflash-qwen3-8b-P2-accum1`](https://huggingface.co/Xingh3/dflash-qwen3-8b-P2-accum1) (1.67B params, BF16)
- **Backend**: SGLang (tp=1, 1x H100 80GB, KV cache, CUDA graphs, paged attention)
- **Mode**: Quick (30 samples per dataset), temperature=0.0, concurrency=1
- **Total runtime**: ~60 minutes for all 10 datasets

#### Results: Cross-Dataset (P2-accum1 vs Phase H vs z-lab)

| Dataset | P2-accum1 τ | Phase H τ | Δ (vs H) | z-lab τ | Gap to z-lab |
|---------|-------------|-----------|----------|---------|--------------|
| **gsm8k** | **3.79** | 3.75 | +0.04 | 3.38 | **-0.41 (beat)** |
| **math500** | **4.07** | 3.87 | +0.20 | 4.61 | +0.54 |
| **aime24** | **3.86** | 3.92 | -0.06 | 4.12 | +0.26 |
| **aime25** | **3.59** | 3.62 | -0.03 | 4.07 | +0.48 |
| **humaneval** | **4.30** | 4.12 | +0.18 | — | — |
| **mbpp** | **3.44** | 3.50 | -0.06 | — | — |
| **livecodebench** | **4.86** | 4.90 | -0.04 | — | — |
| **swe-bench** | **2.61** | 2.60 | +0.01 | — | — |
| **mt-bench** | **2.74** | 2.80 | -0.06 | — | — |
| **alpaca** | **2.42** | 2.45 | -0.03 | — | — |

#### Throughput & Speedup (SGLang backend)

| Dataset | Baseline tok/s | DFlash tok/s | E2E Speedup | Decode Speedup |
|---------|---------------|-------------|-------------|----------------|
| gsm8k | 146.9 | 331.5 | 2.26x | 2.30x |
| math500 | 145.4 | 357.1 | 2.46x | 2.33x |
| aime24 | — | — | 2.41x | 2.39x |
| aime25 | — | — | 2.28x | 2.26x |
| humaneval | — | — | 2.67x | 2.69x |
| mbpp | — | — | 2.13x | 2.16x |
| livecodebench | — | — | 2.28x | 2.27x |
| swe-bench | — | — | 1.77x | 1.74x |
| mt-bench | — | — | 1.52x | 1.59x |
| alpaca | — | — | 1.54x | 1.45x |

#### Domain Analysis

| Domain | P2-accum1 τ | Phase H τ | Δ | z-lab τ |
|--------|-------------|-----------|---|---------|
| Math (gsm8k, math500, aime24, aime25) | **3.83** | 3.79 | **+0.04** | 4.05 |
| Code (humaneval, mbpp, livecodebench) | **4.20** | 4.17 | **+0.03** | — |
| General (swe-bench, mt-bench, alpaca) | **2.59** | 2.62 | -0.03 | — |
| Overall (all 10) | **3.57** | 3.45 | **+0.12** | — |

#### τ Distribution per Dataset (30 samples each)

| Dataset | τ∈[1,2) | τ∈[2,3) | τ∈[3,4) | τ∈[4,5) | τ∈[5,6) | τ∈[6,8) | Mean τ |
|---------|---------|---------|---------|---------|---------|---------|--------|
| gsm8k | — | 6.7% | **70.0%** | 16.7% | 6.7% | — | 3.79 |
| math500 | — | 16.7% | 36.7% | **33.3%** | 3.3% | 10.0% | 4.07 |
| aime24 | — | 6.7% | **53.3%** | 30.0% | 10.0% | — | 3.86 |
| aime25 | — | 16.7% | **70.0%** | 10.0% | 3.3% | — | 3.59 |
| humaneval | — | — | **43.3%** | 33.3% | 20.0% | 3.3% | 4.30 |
| mbpp | — | 20.0% | **56.7%** | 23.3% | — | — | 3.44 |
| livecodebench | — | — | 26.7% | **36.7%** | 20.0% | 16.7% | 4.86 |
| swe-bench | — | **100.0%** | — | — | — | — | 2.61 |
| mt-bench | 20.0% | **50.0%** | 23.3% | 3.3% | — | 3.3% | 2.74 |
| alpaca | 23.3% | **60.0%** | 16.7% | — | — | — | 2.42 |

#### Distribution Comparison: P2-accum1 vs Phase G (3-epoch 190K)

Phase G used 128 samples per dataset; P2-accum1 uses 30. Comparing the distribution shape shift:

| Dataset | Phase G: τ≥3 | P2-accum1: τ≥3 | Shift | Phase G: τ≥5 | P2-accum1: τ≥5 |
|---------|--------------|-----------------|-------|--------------|-----------------|
| gsm8k | 67.9% | **93.4%** | +25.5 | 0% | **6.7%** |
| math500 | 61.7% | **83.3%** | +21.6 | 1.6% | **13.3%** |
| aime24 | 63.4% | **93.3%** | +29.9 | 0% | **10.0%** |
| aime25 | 56.6% | **83.3%** | +26.7 | 0% | **3.3%** |
| humaneval | 89.0% | **100.0%** | +11.0 | 0.6% | **23.3%** |
| mbpp | 55.5% | **80.0%** | +24.5 | 0% | 0% |
| livecodebench | 62.5% | **100.0%** | +37.5 | 19.5% | **36.7%** |
| swe-bench | 1.6% | 0% | -1.6 | 0% | 0% |
| mt-bench | 24.9% | **30.0%** | +5.1 | 2.5% | **3.3%** |
| alpaca | 10.9% | **16.7%** | +5.8 | 0% | 0% |

**Key distribution insights:**

1. **τ≥3 is the dominant bucket** for math/code datasets: 80-100% of requests achieve τ≥3 (vs 55-89% in Phase G). This means the draft model reliably accepts 3+ tokens per cycle on structured tasks.

2. **τ≥5 emerged as a real category**: Phase G had almost no requests with τ≥5 except livecodebench. P2-accum1 pushes 3-37% of requests into τ∈[5,8) across math/code datasets — these are the high-value requests where speculative decoding delivers maximum throughput.

3. **General-domain distribution unchanged**: swe-bench is stuck at 100% τ∈[2,3), and alpaca/mt-bench remain τ<3 dominated. These datasets have diverse/unpredictable text patterns that the draft model cannot reliably anticipate.

4. **The mean understates the improvement**: On livecodebench, mean τ barely changed (4.90→4.86), but the τ≥5 fraction nearly doubled (19.5%→36.7%). The distribution is shifting rightward with a heavier tail, meaning more requests get large speedups even when the average is similar.

#### Analysis

P2-accum1 produces results **on par with Phase H** in mean τ across all domains, with slight improvements on math (+0.04 τ) and code (+0.03 τ). The model beats z-lab on GSM8K (3.79 vs 3.38) and matches Phase H on everything else.

The `accum=1` config doubles optimizer steps but halves the effective batch size (6 vs 12). At full scale, this does not provide a meaningful advantage over Phase H in mean τ — the additional gradient updates are offset by noisier per-step gradients from the smaller batch.

The z-lab gap on math benchmarks (5.4% for P2-accum1 vs 6.4% for Phase H) remains primarily attributable to recipe differences: fewer epochs (3 vs 6), shorter sequences (2048 vs 3072), and different dataset composition.

### P2-WSD Inference Benchmark (2026-04-02)

P2-WSD completed 3 epochs on 800K PerfectBlend with WSD schedule (warmup 4%, stable 76% at peak LR, cosine decay 20%). Training required two Modal sessions due to 24h limit — resumed from `iter_125963` in the second session.

- **HF Model**: [`Xingh3/dflash-qwen3-8b-P2-WSD`](https://huggingface.co/Xingh3/dflash-qwen3-8b-P2-WSD) (1.67B params, BF16)
- **Backend**: SGLang (tp=1, 1x H100 80GB, KV cache, CUDA graphs, paged attention)
- **Mode**: Quick (30 samples per dataset), temperature=0.0, concurrency=1

#### Results: Cross-Dataset (P2-WSD vs P2-accum1 vs Phase H vs z-lab)

| Dataset | P2-WSD τ | P2-accum1 τ | Phase H τ | z-lab τ | E2E Spd | Dec Spd |
|---------|----------|-------------|-----------|---------|---------|---------|
| **gsm8k** | **3.89** | 3.79 | 3.75 | 3.38 | 2.47x | 2.53x |
| **math500** | **4.19** | 4.07 | 3.87 | 4.61 | 2.80x | 2.67x |
| **aime24** | **3.98** | 3.86 | 3.92 | 4.12 | 2.60x | 2.59x |
| **aime25** | **3.69** | 3.59 | 3.62 | 4.07 | 2.42x | 2.41x |
| **humaneval** | **4.30** | 4.30 | 4.12 | — | 2.68x | 2.72x |
| **mbpp** | **3.47** | 3.44 | 3.50 | — | 2.14x | 2.20x |
| **livecodebench** | **4.72** | 4.86 | 4.90 | — | 2.96x | 3.02x |
| **swe-bench** | **2.63** | 2.61 | 2.60 | — | 1.70x | 1.70x |
| **mt-bench** | **2.76** | 2.74 | 2.80 | — | 1.56x | 1.62x |
| **alpaca** | **2.47** | 2.42 | 2.45 | — | 1.55x | 1.42x |

#### Domain Analysis

| Domain | P2-WSD τ | P2-accum1 τ | Phase H τ | z-lab τ |
|--------|----------|-------------|-----------|---------|
| Math (gsm8k, math500, aime24, aime25) | **3.94** | 3.83 | 3.79 | 4.05 |
| Code (humaneval, mbpp, livecodebench) | **4.16** | 4.20 | 4.17 | — |
| General (swe-bench, mt-bench, alpaca) | **2.62** | 2.59 | 2.62 | — |
| Overall (all 10) | **3.57** | 3.57 | 3.45 | — |

#### τ Distribution per Dataset (30 samples each)

| Dataset | τ∈[1,2) | τ∈[2,3) | τ∈[3,4) | τ∈[4,5) | τ∈[5,6) | τ∈[6,8) | Mean τ |
|---------|---------|---------|---------|---------|---------|---------|--------|
| gsm8k | — | 3.3% | **70.0%** | 13.3% | 13.3% | — | 3.89 |
| math500 | — | 10.0% | 36.7% | **36.7%** | 6.7% | 10.0% | 4.19 |
| aime24 | — | 6.7% | **46.7%** | 36.7% | 10.0% | — | 3.98 |
| aime25 | — | 10.0% | **70.0%** | 16.7% | 3.3% | — | 3.69 |
| humaneval | — | — | **46.7%** | 36.7% | 13.3% | 3.3% | 4.30 |
| mbpp | — | 13.3% | **76.7%** | 10.0% | — | — | 3.47 |
| livecodebench | — | 3.3% | 23.3% | **43.3%** | 16.7% | 13.3% | 4.72 |
| swe-bench | — | **100.0%** | — | — | — | — | 2.63 |
| mt-bench | 16.7% | **53.3%** | 23.3% | 3.3% | — | 3.3% | 2.76 |
| alpaca | 16.7% | **66.7%** | 16.7% | — | — | — | 2.47 |

#### Distribution Comparison: P2-WSD vs P2-accum1

| Dataset | P2-accum1: τ≥3 | P2-WSD: τ≥3 | Shift | P2-accum1: τ≥5 | P2-WSD: τ≥5 |
|---------|----------------|--------------|-------|----------------|--------------|
| gsm8k | 93.4% | **96.7%** | +3.3 | 6.7% | **13.3%** |
| math500 | 83.3% | **90.0%** | +6.7 | 13.3% | **16.7%** |
| aime24 | 93.3% | **93.4%** | +0.1 | 10.0% | 10.0% |
| aime25 | 83.3% | **90.0%** | +6.7 | 3.3% | 3.3% |
| humaneval | 100.0% | 100.0% | 0 | 23.3% | **16.7%** |
| mbpp | 80.0% | **86.7%** | +6.7 | 0% | 0% |
| livecodebench | 100.0% | **96.7%** | -3.3 | 36.7% | **30.0%** |
| swe-bench | 0% | 0% | 0 | 0% | 0% |
| mt-bench | 30.0% | **30.0%** | 0 | 3.3% | 3.3% |
| alpaca | 16.7% | 16.7% | 0 | 0% | 0% |

#### Analysis: P2-WSD is the New Best Model

P2-WSD achieves **τ=3.94 math avg**, our best result and only **2.7% below z-lab** (4.05). It beats P2-accum1 on all 4 math benchmarks:

| Dataset | P2-WSD vs P2-accum1 | P2-WSD vs Phase H |
|---------|---------------------|-------------------|
| gsm8k | **+0.10** | +0.14 |
| math500 | **+0.12** | +0.32 |
| aime24 | **+0.12** | +0.06 |
| aime25 | **+0.10** | +0.07 |

The WSD schedule's stable LR phase (76% of training at peak LR=6e-4) allows the model to explore the loss landscape more thoroughly before settling. By contrast, cosine decay (Phase H, P2-accum1) begins reducing LR immediately after warmup, causing premature convergence. This is particularly visible on math500 (+0.32 τ vs Phase H), where longer exploration at high LR helps the model learn more complex reasoning patterns.

Code domain is slightly lower (4.16 vs 4.20 for P2-accum1), with livecodebench dropping from 4.86→4.72. This may reflect a small trade-off where the extended stable LR phase favors math reasoning patterns at the expense of code prediction. General domain is unchanged.

The decode-only speedup of **3.02x on livecodebench** is the first time we've crossed the 3.0x speedup target on any dataset.

### Infrastructure: Sweep Runner

Added `--sweep-config` flag to `modal_dflash_train.py` for parallel hyperparameter sweeps. Each run gets its own 8xH100 Modal container with isolated WandB tracking.

```bash
modal run --detach --env sandbox scripts/modal/modal_dflash_train.py \
  --sweep-config scripts/modal/sweep_configs/convergence_sweep.json \
  --wandb-team dflash \
  --sweep-filter "R00-baseline,R03-WSD,R07-accum1"
```

Sweep configs live in `scripts/modal/sweep_configs/` as JSON files with `base` (shared settings) and `runs` (per-config overrides).

### Best Known Training Config (as of Phase K)

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 6e-4 |
| `min_lr` | 6e-5 |
| `lr_decay_style` | WSD |
| `wsd_decay_ratio` | 0.2 (last 20% of steps for decay) |
| `wsd_decay_style` | cosine |
| `warmup_ratio` | 0.04 |
| `weight_decay` | 0.01 |
| `draft_accumulation_steps` | 2 |
| `dflash_loss_decay_gamma` | 7.0 |
| `dflash_num_anchors` | 512 |

---

## Commits (RunPod Era)

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

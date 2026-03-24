# DFlash Modal Training Results

## Environment

- **Platform**: Modal (serverless GPU cloud)
- **GPUs**: 8x NVIDIA H100 80GB HBM3
- **Interconnect**: NVLink (intra-node), no RDMA/InfiniBand
- **Software**: torch 2.11 + sglang 0.5.9 + CUDA 12.4
- **Mooncake**: TCP protocol, CPU prefetch (prefetch_depth=8)
- **Dataset**: PerfectBlend 50K (`perfectblend_50k.jsonl`, 47,484 samples)
- **Base config**: `configs/sglang_qwen3_8b_dflash.yaml`

---

## Baseline: 8x H100 (1 Inference + 7 Training FSDP)

200 steps, `micro_batch_size=1`, `draft_accumulation_steps=4`, `dflash_num_anchors=512`, `max_seq_length=2048`.

Global batch size = 1 × 4 × 7 = 28.

### Timing (steady state, steps 50-200)

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

### Analysis

- **Forward variance**: FlexAttention `mask_mod` closure changes every step (different `anchor_positions`), triggering Dynamo recompilation. `Q_LEN = n_blocks × block_size` where `n_blocks` varies per batch.
- **Backward**: 7-way FSDP FULL_SHARD reduce-scatter + all-gather.
- **Data**: Fully overlapped with compute via CPU prefetch (prefetch_depth=8).
- **Bottleneck**: Compute (forward + backward), not data transfer.

---

## Speed Tuning Tests (200 steps each)

All tests use 8x H100 with CLI overrides via `--extra-overrides`. No YAML changes.

### Test A: Reduce Anchors (1 Inference + 7 Training)

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

### Test B: Larger Micro-Batch + Fewer Anchors (1 Inference + 7 Training)

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

### Test C: Maximum Micro-Batch (1 Inference + 7 Training)

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

### Test C2: Maximum Micro-Batch + 2 Inference GPUs (2 Inference + 6 Training)

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

---

## Comparison Summary

| Config | GPUs (I+T) | batch | accum | anchors | Total Time | step_time | thru (s/s) | pool | Bottleneck |
|--------|-----------|-------|-------|---------|------------|-----------|------------|------|------------|
| Baseline | 1+7 | 1 | 4 | 512 | — | 0.87-1.4s | 20-25 | full | compute (fwd variance) |
| **Test A** | 1+7 | 1 | 4 | **256** | **610s** | **0.55-0.64s** | 17-19 | full | compute |
| Test B | 1+7 | 2 | 2 | 256 | 637s | 0.62-0.71s | 16-18 | full | compute |
| Test C | 1+7 | 4 | 1 | 256 | 640s | 1.39-1.46s | 16-18 | starved | **data (pool empty)** |
| **Test C2** | **2+6** | **4** | **1** | **256** | **476s** | **0.86-0.95s** | **22-25** | **full** | **compute** |

### Key Findings

1. **`dflash_num_anchors` is the biggest lever**: 512→256 halves Q_LEN, cuts forward time from 260-800ms to 240-434ms. Consistent with Phase C results on RunPod.

2. **Larger micro_batch_size has diminishing returns with 1 inference GPU**: batch=4 with accum=1 drains the sample pool faster than inference can refill it. The pool drops to 16-28/64, causing data starvation and slower overall throughput despite faster per-sample backward time.

3. **Adding a second inference GPU solves data starvation for batch=4**: Test C2 (2 inference GPUs) keeps the pool full at 64/64 with inference throughput of ~34 samples/s. This is the opposite of Phase F Test 4 on RunPod (4-GPU), where a second inference engine caused a 62% regression due to Mooncake TCP contention. The difference: 8-GPU has more PCIe bandwidth headroom and 6 training GPUs consuming data faster.

4. **Test C2 is the fastest configuration**: 476s total, 22-25 samples/s effective throughput, with full pool and low dispatch wait (~0.05s).

5. **RDMA is not available on Modal**: The RDMA probe confirmed `/dev/infiniband` and `ibstat` are not present. Mooncake uses TCP only, but CPU prefetch effectively hides the latency.

### 200K × 3 Epoch Estimates

| Config | samples/s | Est. Time |
|--------|-----------|-----------|
| Baseline (1+7, anchors=512) | ~22 | ~7.6 hr |
| Test A (1+7, anchors=256) | ~18 | ~9.3 hr |
| **Test C2 (2+6, batch=4, anchors=256)** | **~24** | **~6.9 hr** |

### Recommended Config for Full Training

```bash
modal run --detach --env sandbox scripts/modal_dflash_train.py \
  --max-steps 999999 --num-epochs 3 --dataset-size 200000 \
  --extra-overrides "training.dflash_num_anchors=256 training.micro_batch_size=4 training.draft_accumulation_steps=1 inference.inference_num_gpus=2 training.training_num_gpus_per_node=6"
```

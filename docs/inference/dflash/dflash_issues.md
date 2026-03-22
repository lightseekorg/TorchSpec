# DFlash Issues & Solutions

All issues encountered during DFlash implementation and training, consolidated from the implementation log.

---

## Environment & Setup Issues

### Issue 1: RunPod SSH PTY Requirement

RunPod's SSH gateway requires a pseudo-terminal. Non-interactive SSH fails with `Error: Your SSH client doesn't support PTY`.

**Solution**: Use `expect` to allocate a real PTY:
```bash
expect -c '
set timeout 60
spawn ssh -o StrictHostKeyChecking=no -o RequestTTY=force \
    -i ~/.ssh/id_ed25519 USER@ssh.runpod.io
expect -re {[#\$] }
interact
'
```

### Issue 2: PyTorch Version (2.4.1 → 2.6+ required)

FlexAttention (`torch.nn.attention.flex_attention`) requires PyTorch 2.6+. RunPod image has 2.4.1.

**Solution**: See [RunPod Setup Guide](dflash_runpod_guide.md) Step 3.

### Issue 3: Missing Native Libraries (Mooncake/RDMA)

`mooncake-transfer-engine` requires RDMA libraries.

**Solution**: `apt-get install -y libibverbs-dev librdmacm-dev libnuma-dev`

### Issue 5: Container Disk Too Small

Default RunPod container disk (20 GB) is insufficient. Training needs ~30-40 GB.

**Solution**: Provision with **100 GB container disk**.

### Issue 19: HF Cache Miss in Ray Workers

Ray workers default to `/root/.cache/huggingface/` which is empty. Cached model at `/workspace/.cache/huggingface/` not used.

**Solution**: `ln -s /workspace/.cache/huggingface /root/.cache/huggingface` + `export HF_HOME=/workspace/.cache/huggingface`

### Issue 23: `PYTORCH_CUDA_ALLOC_CONF` Deprecated in PyTorch 2.9+

PyTorch 2.9.1 renamed the environment variable.

**Solution**: Use `export PYTORCH_ALLOC_CONF=expandable_segments:True` instead.

### Issue 24: `flashinfer-jit-cache==0.6.2` Not Available on cu124

**Solution**: Not needed as a separate install — SGLang's `[all]` extras already install `flashinfer_python-0.6.2` and `flashinfer_cubin-0.6.2`. Skip the standalone flashinfer install.

---

## Code & Configuration Issues

### Issue 4: SGLang/vLLM Unconditional Imports

TorchSpec imported SGLang/vLLM at module level even when using HF backend.

**Solution**: Made imports lazy — moved into the functions that use them. Files: `inference/engine/__init__.py`, `inference/factory.py`.

### Issue 9: DFlash Draft Config Dimension Mismatch

`dflash_draft_config.json` had Qwen2.5-7B dimensions but we train against Qwen3-8B.

**Fix**: Updated config — `hidden_size: 3584→4096`, `target_hidden_size: 3584→4096`, `intermediate_size: 18944→12288`, `num_attention_heads: 28→32`, `num_key_value_heads: 4→8`, `target_num_hidden_layers: 28→36`, `vocab_size: 152064→151936`, `max_position_embeddings: 32768→40960`.

### Issues 13+14: Zero-Loss Reporting Bug

**Symptoms**: DFlash showed `loss=0.000, acc=0.000` for all 200 steps. Training suspiciously fast.

**Root cause**: **Metric key name mismatch**, not actual zero loss. `metrics.get('train/avg_loss', 0)` silently returned `0` because DFlash used `train/loss` instead of `train/avg_loss`.

**Fix**: Renamed metric keys in `DFlashTrainer._aggregate_metrics()` to match Eagle3 convention.

**Lesson**: When extending a trainer subclass, always check the controller's expected metric key format.

**Additional fixes during investigation**:
- Added fallback in `_sample_anchor_positions`: when no valid positions exist, sample uniformly
- Rewrote `_prepare_noise_input` as fully vectorized using `torch.gather` (eliminated 2,760 Python loop iterations with GPU-CPU sync)

### Issues 11+12: DFlash dtype Mismatch

**Problem**: Target hidden states (BFloat16) fed into float32 layers at two locations.

**Solution**:
1. Added `.to(self.context_proj.weight.dtype)` in `extract_context_feature()` before projection
2. Moved `draft_model.to(torch.bfloat16)` after `freeze_embedding()` in `dflash_trainer.py`

### Issue 18: Collator Negative Padding Dimension

`paddingtensor2D` crashes when `loss_mask` is longer than `input_ids`.

**Solution**: Truncation guard: `if n > N: return intensors[:, :N]` in both `paddingtensor2D` and `paddingtensor`. Commit: `398138a`.

### `build_target_layer_ids()` Off-by-One (Phase B)

TorchSpec produced `[1, 10, 18, 26, 35]` instead of SpecForge's `[1, 9, 17, 25, 33]`. Layer 35 + SGLang's +1 capture offset = 36, out of bounds for `range(36)`.

**Fix**: Rewrote to match SpecForge: `start=1, end=num_hidden_layers-3`. Also set explicit `target_layer_ids: [1, 9, 17, 25, 33]` in `dflash_draft_config.json`.

**Key insight**: SGLang applies `+1` offset by design — `set_eagle3_layers_to_capture([val + 1 for val in layer_ids])`. The forward loop captures hidden states at the start of each iteration, so position `k+1` captures output of layer `k`.

---

## Training Runtime Issues

### Issue 10: FlexAttention Inductor `NoValidChoicesError`

`torch._inductor` kernel autotuner had no valid GEMM backends during FlexAttention backward pass.

**Solution** (two-part):
1. Set inductor config at import time in `flex_attention.py`: `inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"`
2. Initially used `backend="aot_eager"` as workaround; later switched back to inductor for 3x speedup.

### Issue 15: RunPod Volume Disk Quota

Writing to `/workspace/` (shared NFS mount) failed with `Disk quota exceeded`.

**Solution**: Use container disk (`/root/`, `/tmp/`) for temporary files. Container disk has ~76 GB free.

### Issue 16: FSDP Checkpoint Extraction

`dcp_to_torch_save()` and direct `torch.load()` both fail on `.distcp` format.

**Solution**: Use `dist_cp.state_dict_loader._load_state_dict(no_dist=True)` with `_WrappedStorageReader` and `_EmptyStateDictLoadPlanner`.

### Issue 17: FlexAttention OOM with Large num_anchors

`num_anchors=512 × block_size=16 = 8192` Q tokens. FlexAttention attention matrix too large on 80GB GPUs.

**Memory breakdown (per GPU)**:

| Component | Memory |
|-----------|--------|
| Draft model (1B params, bf16) | ~2 GB |
| Optimizer states (Adam, fp32) | ~8 GB |
| FlexAttention forward | ~6-12 GB |
| FlexAttention backward | ~9-18 GB |
| Gradient buffers + overhead | ~9-14 GB |

**Solution**: `num_anchors=256` halves Q_LEN (8192→4096). Combined with `draft_accumulation_steps=4` to maintain effective batch size.

### Issue 20: FSDP Checkpoint Disk Quota

Each FSDP checkpoint ~15 GB. Two checkpoints exceed RunPod's per-pod quota.

**Solution**: Added **checkpoint rotation** (`max_checkpoints` config option). `max_checkpoints: 1` deletes oldest before saving. Commit: `c7c6605`.

### Issues 8+21+22: Eval Cache Timeout

**Problem**: Eval cache generation hangs/times out (300s) in multiple scenarios:
1. Colocate mode — single GPU shared between inference and training deadlocks
2. Missing eval data path on RunPod
3. Stale Ray/Mooncake state after kill/restart cycles
4. `eval_interval=0` alone is insufficient — initial cache generation still runs if `eval_data_path` is set

**Solution**: Must set **both** overrides:
```
dataset.eval_data_path=null dataset.eval_interval=0
```

### Issue 25: SGLang Engine Ray Actor Timeout (30s) with PyTorch 2.9.1

CUDA context initialization inside Ray actors takes longer with PyTorch 2.9.1, exceeding the 30s limit for `find_free_port`.

**Fix**: Patched all three `timeout=30` → `timeout=120` in `torchspec/inference/factory.py`.

**Note**: Local pod patch, not committed. The `init_timeout` config (default 300s) only applies to the engine `.init()` call, not the pre-init `find_free_port` calls.

### Issue 26: PyTorch 2.9.1 Speed Regression (3x Slower) — RESOLVED

Training at 1.5-1.7 s/step with torch 2.9.1 vs 0.48 s/step with torch 2.6.0. GPU utilization ~20-30% average with bursty compute and long idle periods.

**Root cause**: TorchInductor GEMM backend selection. PyTorch 2.9.1's autotuner fails to select valid backends for FlexAttention kernels.

**Solution**:
```bash
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
```

This single env var fixes both:
1. The 3x speed regression (restored to ~5 step/s in colocate 1-GPU mode)
2. The `NoValidChoicesError` in FlexAttention backward pass ([Issue 10](dflash_issues.md#issue-10-flexattention-inductor-novalidchoiceserror))

**Previous unsuccessful attempts** (before finding the fix):
1. Remove `@torch.compile(dynamic=True)` from DFlashRMSNorm — no change
2. Use `enable_gqa=True` in FlexAttention — no change
3. Use `create_block_mask` directly (remove compiled wrapper) — no change
4. `TORCHINDUCTOR_MAX_AUTOTUNE=1` + `COORDINATE_DESCENT_TUNING=1` — no change
5. `TORCH_COMPILE_DISABLE=1` (eager mode) — crash (SGLang requires compilation)

---

## Training Quality Bugs (Found in Session 12)

### Bug 1: Zero-loss dummy on empty anchors (`dflash.py:128-134`)

**TorchSpec**: Returns dummy tensors with `keep_mask=False` everywhere → zero loss, zero gradients. Training steps wasted silently.

**SpecForge**: Raises `ValueError("should preprocess the data.")`.

**Impact**: If any batch has no valid anchor positions (short sequences or mostly-masked data), TorchSpec silently produces zero gradients, wasting compute and diluting gradient signal.

**Status**: **Fixed** (commit `f3311e4`). Replaced silent zero-loss return with `raise ValueError`. Added `min_loss_tokens` config parameter to filter short sequences at data loading time.

### Bug 2: Anchor filtering by anchor position's mask (`dflash.py:126`)

```python
valid = loss_mask[:, : max_anchor + 1] > 0.5
```

Filters candidate anchors by whether the **anchor token itself** has `loss_mask=1`, but the labels predicted by the block are at positions `anchor+1` through `anchor+block_size-1`. This excludes anchors at prompt positions even though their block's labels at completion positions would be valid training signal.

**Note**: Same pattern exists in SpecForge's code. Impact is limited to **reduced anchor diversity** near prompt→completion boundaries (downstream `original_loss_mask_gathered` correctly zeros out labels in masked regions).

**Status**: Needs investigation — benchmark whether allowing anchors at prompt positions improves τ.

---

## Speed Optimization Issues (Phase E/F)

### Issue 27: torch.compile on full DFlash model causes recompilation storm

Applying `torch.compile(dflash_model)` causes 1-3s/step overhead for 50+ steps (vs 0.37s baseline). Step 1 warmup = 111s.

**Root cause**: DFlash has dynamic tensor shapes — anchor count varies per sample (depends on `loss_mask`), sequence lengths vary after `min_loss_tokens` filtering. Inductor recompiles for each new shape combination.

**Status**: **Not viable**. `compile_model=true` config option added but should NOT be used. Individual op-level compiles (FlexAttention, RoPE, loss) already cover the hot paths.

### Issue 28: GPU Direct RDMA fails on RunPod

Setting `mooncake.enable_gpu_direct=True` causes `torch.AcceleratorError: CUDA error: device-side assert triggered` during training.

**Root cause**: RunPod H100 pods use PCIe interconnect without RDMA NIC. Mooncake GPU Direct requires InfiniBand or RoCE for RDMA registration.

**Status**: **Expected on RunPod**. GPU Direct only works on RDMA-capable clusters.

### Issue 29: No same-node Mooncake bypass — **Critical Bottleneck**

`EagleMooncakeStore` always uses Mooncake TCP protocol even when inference and training are on the same node. No config option to use NVLink, NCCL p2p, or shared memory.

**Hardware context**: Pod has 4x H100 with full-mesh **NV18 NVLink** (478 GB/s per GPU). Mooncake uses TCP over loopback (~0.3 GB/s effective), wasting **1500x available bandwidth**. Transfer of ~40MB hidden states takes ~145ms via TCP vs <0.1ms via NVLink.

**Impact**: data_time is 40-60% of step time across all configs. This is the **single biggest performance bottleneck**.

| Transport | Bandwidth | 40MB latency | Supported by Mooncake? |
|-----------|-----------|-------------|----------------------|
| NV18 NVLink | 478 GB/s | <0.1 ms | No |
| NCCL (NVLink) | ~450 GB/s | <0.1 ms | No |
| Mooncake TCP | ~0.3 GB/s | **~145 ms** | **Yes (only option)** |
| Mooncake RDMA | 25-100 GB/s | ~1-5 ms | No (no IB on RunPod) |

**Status**: **Needs code changes**. Would require new `local` transport backend in `data_fetcher.py` and `eagle_store.py` using `torch.distributed.send/recv` (NCCL) or CUDA IPC. Colocate mode is the only workaround (but OOMs with 8B target).

### Issue 30: FlexAttention recompile overflow from variable padding — **Critical Performance Bug**

With `batch_size>1`, `DataCollatorWithPadding` pads each batch to the max sequence length within that batch. Every batch gets a unique padded length, producing unique `(Q_LEN, KV_LEN)` dimensions for `flex_attention`.

**Root cause**: `flex_attention` is compiled via `torch.compile` (singleton `WrappedFlexAttention`). Each new `(Q_LEN, KV_LEN)` shape triggers a recompilation. After `dynamo.config.recompile_limit=64` unique shapes, dynamo silently falls back to **eager mode** — losing all FlexAttention kernel optimization.

**Symptoms** (observed at step 615, Phase 3):
- `fwd=266ms` (vs 78ms Phase F steady-state) — 3.4x slower
- `bwd=393ms` (vs 138ms Phase F) — 2.8x slower
- `data=670ms` — prefetch queue drains because compute is slow
- `step=1.09s` (vs 0.19s target) — **5.7x slower**
- Phase F benchmarks appeared fast because they were short runs that stayed within the 64-shape limit

**Solution**: Bucketed padding in `DataCollatorWithPadding` — round `max_length` up to the nearest 256 boundary. With `max_seq_length=2048`, this limits unique shapes to 8 (`256, 512, 768, ..., 2048`), well within the recompile limit.

```python
# torchspec/data/utils.py — DataCollatorWithPadding.__call__()
_BUCKET = 256
max_length = ((max_length + _BUCKET - 1) // _BUCKET) * _BUCKET
```

Also increased `dynamo.config.recompile_limit` from 64 to 128 as safety margin.

**Impact**: Restores steady-state speed of ~5 step/s (matching Phase F Test 7).

**Status**: **RESOLVED** (2026-03-22).

### Issue 31: Mooncake buffer use-after-free with batch_size>1

With `batch_size>1`, the DataLoader collator holds sample N while waiting for sample N+1. `_load_from_mooncake()` called `_cleanup_mooncake_data()` (which frees the Mooncake buffer) before `to_tensor_dict()` returned copies. The freed buffer gets reused by new inference data, corrupting held tensors with garbage values.

**Symptoms**: `input_ids` contained random int64 values like `-4953751034053673866`, causing `embed_tokens` to crash with CUDA index-out-of-bounds.

**Root cause**: `to_tensor_dict()` returns **views** into Mooncake buffers, not copies. With batch_size=1, the single sample is consumed immediately, so the race doesn't manifest.

**Solution**: Clone tensors before cleanup:
```python
result = {k: v.clone() for k, v in tensors.to_tensor_dict().items()}
self._cleanup_mooncake_data(sample)
```

**Status**: **RESOLVED** (2026-03-22). See Issue 32 for performance regression caused by this fix.

### Issue 32: clone() breaks pinned memory async transfers — **5x speed regression**

The `.clone()` fix for Issue 31 inadvertently breaks the async CPU→GPU transfer path for ALL batch sizes, causing a 5x speed regression (1 step/s vs 5.3 step/s).

**Root cause**: Mooncake host buffers are allocated with `pin_memory=True`. `torch.frombuffer()` creates views preserving pinned status. `.clone()` creates **unpinned** tensors. With unpinned memory, `tensor.to(device, non_blocking=True)` is **silently ignored** — PyTorch falls back to blocking synchronous transfer (~300ms+ per batch for 80MB hidden states).

**Symptoms** (Phase 3, step 100+):
- `data_time=300-1100ms` (vs Phase F's 1-9ms)
- `step_time=1.0-1.5s` (vs Phase F's 0.19s)
- Prefetch queue drains because blocking H2D stalls the main thread

**Solution**: Make clone conditional on batch_size — only needed for batch>1 (Issue 31). With batch=1, use pinned views directly:
```python
if self._batch_size > 1:
    result = {k: v.clone() for k, v in tensor_dict.items()}
else:
    result = dict(tensor_dict)  # preserves pinned memory
```

**Status**: **RESOLVED** (2026-03-22).

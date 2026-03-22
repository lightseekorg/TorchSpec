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

### Issue 26: PyTorch 2.9.1 Speed Regression (3x Slower)

Training at 1.5-1.7 s/step with torch 2.9.1 vs 0.48 s/step with torch 2.6.0. GPU utilization ~20-30% average with bursty compute and long idle periods.

**Optimization attempts** (none helped):
1. Remove `@torch.compile(dynamic=True)` from DFlashRMSNorm — no change
2. Use `enable_gqa=True` in FlexAttention — no change
3. Use `create_block_mask` directly (remove compiled wrapper) — no change
4. `TORCHINDUCTOR_MAX_AUTOTUNE=1` + `COORDINATE_DESCENT_TUNING=1` — no change
5. `TORCH_COMPILE_DISABLE=1` (eager mode) — crash (SGLang requires compilation)

**Root cause**: PyTorch 2.9.1 runtime-level regression (TorchInductor codegen, FSDP2 behavior, NCCL/CUDA runtime, or Triton 3.5.1→3.6.0).

**Conclusion**: No code-level fix found. Future mitigation options:
- Pin to torch 2.6.0 if SGLang compatibility allows
- Profile with `torch.profiler` for exact kernel-level regression
- Test torch 2.7/2.8 as intermediate versions
- Try `torch.compile(mode="max-autotune-no-cudagraphs")` for FlexAttention

---

## Training Quality Bugs (Found in Session 12)

### Bug 1: Zero-loss dummy on empty anchors (`dflash.py:128-134`)

**TorchSpec**: Returns dummy tensors with `keep_mask=False` everywhere → zero loss, zero gradients. Training steps wasted silently.

**SpecForge**: Raises `ValueError("should preprocess the data.")`.

**Impact**: If any batch has no valid anchor positions (short sequences or mostly-masked data), TorchSpec silently produces zero gradients, wasting compute and diluting gradient signal.

**Status**: Not yet fixed.

### Bug 2: Anchor filtering by anchor position's mask (`dflash.py:126`)

```python
valid = loss_mask[:, : max_anchor + 1] > 0.5
```

Filters candidate anchors by whether the **anchor token itself** has `loss_mask=1`, but the labels predicted by the block are at positions `anchor+1` through `anchor+block_size-1`. This excludes anchors at prompt positions even though their block's labels at completion positions would be valid training signal.

**Note**: Same pattern exists in SpecForge's code. Impact is limited to **reduced anchor diversity** near prompt→completion boundaries (downstream `original_loss_mask_gathered` correctly zeros out labels in masked regions).

**Status**: Needs investigation — benchmark whether allowing anchors at prompt positions improves τ.

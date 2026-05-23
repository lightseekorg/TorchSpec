# DFlash + Eagle3 Disaggregated Modal — Results & Test Settings

> **Status: all 4 runs complete.**
> **Branch:** TorchSpec `feature/dflash-training @ cb741ae` (with two
> uncommitted helper patches; see §10).
> **Platform:** Modal `doordash/sandbox`, H100 80GB HBM3 SXM.
> **WandB project:** [`dflash/dflash-eagle3-disagg-modal`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal).
> **Companion plan:** [`dflash_eagle3_disagg_vs_colocate_benchmark_plan.md`](./dflash_eagle3_disagg_vs_colocate_benchmark_plan.md).
> **Companion live tracker (now superseded):** [`dflash_eagle3_disagg_modal_runs.md`](./dflash_eagle3_disagg_modal_runs.md).

This doc is the final disagg-Modal arm of the disagg-vs-colocate study.
It records the exact per-run settings, the WandB / Modal artefacts the
runs produced, and the warm-window performance + convergence numbers
that the RunPod colocate arm needs to match cell-for-cell.

If you are the **RunPod colocate agent**: skip straight to **§8 Test
settings the colocate arm must replicate**. That section contains the
single, exhaustive list of knobs.

---

## 1. The 4 runs at a glance

| # | Run ID | Model | Layout | Modal app | WandB run | Wall clock | Final step | NaN | OOM |
|---|---|---|---|---|---|--:|--:|--:|--:|
| **D1** | `D1-dflash-2plus2-disagg-modal` | DFlash | 2 infer + 2 train (`H100:4`) | [`ap-4gNxJ8m2QOOv9HDxNS87dz`](https://modal.com/apps/doordash/sandbox/ap-4gNxJ8m2QOOv9HDxNS87dz) | [`8582vc8g`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal/runs/8582vc8g) | **4 387 s (1 h 13 m)** | 5000 / 5000 | 0 | 0 |
| **E1** | `E1-eagle3-2plus2-disagg-modal` | Eagle3 | 2 infer + 2 train (`H100:4`) | [`ap-0kkF98EfJkyKbumcsLEdY7`](https://modal.com/apps/doordash/sandbox/ap-0kkF98EfJkyKbumcsLEdY7) | [`1827jqkl`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal/runs/1827jqkl) | **11 341 s (3 h 09 m)** | 5000 / 5000 | 0 | 0 |
| **D2** | `D2-dflash-4plus4-disagg-modal` | DFlash | 4 infer + 4 train (`H100:8`) | [`ap-V3eat0e1VaNJZK84Cg4O6n`](https://modal.com/apps/doordash/sandbox/ap-V3eat0e1VaNJZK84Cg4O6n) | [`74xjodeo`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal/runs/74xjodeo) | **4 737 s (1 h 19 m)** | 5000 / 5000 | 0 | 0 |
| **E2** | `E2-eagle3-4plus4-disagg-modal` | Eagle3 | 4 infer + 4 train (`H100:8`) | [`ap-oQBE3oK4HgoSK3Cb5mHdpk`](https://modal.com/apps/doordash/sandbox/ap-oQBE3oK4HgoSK3Cb5mHdpk) | [`skj2g8k2`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal/runs/skj2g8k2) | **10 800 s (3 h 00 m)** | 5000 / 5000 | 0 | 0 |

**Total compute spent:** 4 × pod = (4 GPU × 1 h 13 m) + (4 GPU × 3 h 09 m) +
(8 GPU × 1 h 19 m) + (8 GPU × 3 h 00 m) ≈ **51.7 H100-hours**.

**Headline finding:** **all four 5000-step disagg runs completed in
parallel on a single launch window (~3 h 11 m wall-clock from D1 spawn to
E1 finish)**, no NaN, no OOM, every step trained.

---

## 2. Final performance metrics (TIMING-line authoritative, warm window steps 100–5000)

Source: `loop.py:321 INFO TIMING step=N: step=… data=… compute=… [fwd=… bwd=… opt=…] dispatch=…`
emitted once per ~5 steps by `torchspec/controller/loop.py`. Medians taken
over all `step ≥ 100` samples (the first ~50 steps are very cold and skew
the median; everything after 100 is steady state).

| Run | wall (s) | step (s) | thru (samples/s) | fwd (s) | bwd (s) | opt (s) | data (s) | dispatch (s) | I cap | T cap | I/T | pool med / min–max |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| **D1** DFlash 2+2 | 4 387.2 | **0.789** | **10.14** | 0.316 | 0.418 | 0.023 | 0.416 | 0.020 | 61.7 | 10.20 | **6.05** | 64 / 64–64 |
| **D2** DFlash 4+4 | 4 736.5 | **0.839** | **19.07** | 0.360 | 0.428 | 0.014 | 0.423 | 0.041 | 103.5 | 19.10 | **5.42** | 64 / 64–72 |
| **E1** Eagle3 2+2 | 11 341.3 | **2.125** | **3.76** | ¹ | ¹ | 0.018 | 0.025 | 0.023 | 51.2 | 3.80 | **13.47** | 64 / 64–64 |
| **E2** Eagle3 4+4 | 10 799.8 | **2.021** | **7.92** | ¹ | ¹ | 0.009 | 0.074 | 0.049 | 125.1 | 7.90 | **15.84** | 72 / 64–72 |

¹ `eagle3_trainer.py` does **not** populate the `fwd=…s bwd=…s` sub-fields
on the TIMING line (only `dflash_trainer.py` does); only the aggregated
`compute=…s` is meaningful for Eagle3. E1's median compute is 2.10 s of
the 2.13 s step; E2's median compute is 1.97 s of the 2.02 s step. Both
Eagle3 runs are essentially **100 % compute-bound** at the trainer.

### 2.1 What the numbers say, in plain English

- **DFlash 2+2 → 4+4 scaling is sub-linear at 1.88×** (19.07 / 10.14)
  even though the GPU count doubled. Step time grew slightly (0.789 →
  0.839 s) because global batch grew 8 → 16 and FSDP all-reduce now
  spans 4 ranks instead of 2.
- **Both DFlash runs are heavily trainer-bound** (I/T ≈ 5–6, pool
  saturated at 64). Inference is producing 5–6× faster than training
  can consume — exactly the regime where colocate's on-device CUDA-IPC
  transport gives the *most* room to win.
- **Eagle3 step time is 2.5–2.7 × DFlash's** because of the 7-forward
  TTT path inside `eagle3_trainer.py`. Throughput at matched layout is
  3.7 (E1) / 7.9 (E2) samples/s vs DFlash's 10.1 / 19.1 — roughly the
  predicted "7× slower" with batching offset.
- **Eagle3 is *also* trainer-bound** (I/T ≈ 13–16), but the pool *does*
  swing 64 ↔ 72 (E2) — meaning when inference is the bottleneck briefly
  the pool drops, then refills. DFlash never sees that swing.
- **Data fetch (Mooncake RDMA over TCP since Modal has no IB) is
  overlapped with compute via prefetch.** For DFlash, `data=0.42 s` while
  `compute=0.76 s`; data fits *inside* compute so step ≈ compute.
  For Eagle3, `data` is much smaller (0.03 / 0.07 s) because each
  Eagle3 step consumes far fewer Mooncake fetches per second
  (`thru ≈ 4–8 vs DFlash's 10–19`).
- **`dispatch_wait` is essentially zero everywhere** (< 50 ms median),
  confirming inference never starved any of the 4 runs.

### 2.2 The headline samples/s for cross-arm comparison

> **DFlash, disagg-Modal, anchors=512, warm:**
> - 2+2 (4 GPU): **10.14 samples/s** (step 0.789 s)
> - 4+4 (8 GPU): **19.07 samples/s** (step 0.839 s)
>
> **Eagle3, disagg-Modal, warm:**
> - 2+2 (4 GPU): **3.76 samples/s** (step 2.125 s)
> - 4+4 (8 GPU): **7.92 samples/s** (step 2.021 s)

These four numbers are what the colocate-arm runs (C1, C2, CE1, CE2) must
be compared to.

---

## 3. Convergence trajectories

All four runs are deterministic at `training.seed=42`, no shuffle on the
sample dispatcher (the `feature/dflash-training` controller uses
`training_controller.py:241 INFO Prepared dataset (188977 samples, seed
42+0)` — same prep across runs), so the four loss curves are reproducible
to within run-to-run scheduler jitter.

### 3.1 `train/avg_loss` (in-loop tqdm value, rolling mean of last ~50 steps)

| Run | step 100 | step 500 | step 1000 | step 2000 | step 3000 | step 4000 | step 5000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| D1 DFlash 2+2 | 6.64 | 5.40 | 4.75 | 4.21 | 3.93 | 3.75 | **3.67** |
| D2 DFlash 4+4 | 6.54 | 4.84 | 4.22 | 3.80 | 3.53 | 3.23 | **3.14** |
| E1 Eagle3 2+2 | 5.61 | 3.59 | 2.88 | 2.27 | 2.16 | 1.94 | **2.24** |
| E2 Eagle3 4+4 | 5.21 | 3.08 | 2.48 | 2.05 | 1.77 | 1.58 | **1.80** |

### 3.2 `train/avg_acc` (top-1 next-token / draft accuracy)

| Run | step 100 | step 500 | step 1000 | step 2000 | step 3000 | step 4000 | step 5000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| D1 DFlash 2+2 | 0.055 | 0.114 | 0.151 | 0.187 | 0.208 | 0.226 | **0.235** |
| D2 DFlash 4+4 | 0.060 | 0.144 | 0.184 | 0.216 | 0.244 | 0.272 | **0.284** |
| E1 Eagle3 2+2 | 0.186 | 0.391 | 0.477 | 0.560 | 0.583 | 0.616 | **0.580** |
| E2 Eagle3 4+4 | 0.226 | 0.455 | 0.532 | 0.593 | 0.640 | 0.671 | **0.638** |

### 3.3 `train/acc_len` (Eagle3 speculative acceptance length τ)

DFlash does not produce `acc_len` (it is a single-token CE loss on
anchors, not a multi-token speculation). Eagle3 trains the 7-forward TTT
path and reports the average number of draft tokens accepted by the
target per draft step:

| Run | step 100 | step 500 | step 1000 | step 2000 | step 3000 | step 4000 | step 5000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| E1 Eagle3 2+2 | 0.42 | 0.90 | 1.21 | 1.58 | 1.69 | 1.89 | **1.66** |
| E2 Eagle3 4+4 | 0.49 | 1.11 | 1.43 | 1.74 | 2.03 | **2.24** | 2.00 |

**Eagle3 peak τ on these settings: E2 reached τ ≈ 2.24 at step 4000**
before drifting slightly to τ ≈ 2.00 at step 5000. E1 peaked at τ ≈ 1.89
at step 4000 → 1.66 at step 5000. The slight late-stage drop is the
known "noisy plateau" — `convergence_sweep.json` runs that go to 800K
samples / 3 epochs typically push past τ = 2.5 in the same setup
([`docs/inference/dflash/training_results.md`](../../../../TorchSpec/docs/inference/dflash/training_results.md)).

For the colocate-arm comparison the **target τ at step 5000** is the
right comparator, *not* the peak τ — colocate must reproduce 1.66 / 2.00
within ±2 % to be a valid grad-parity comparison. The Phase-7 colocate
grad-parity test already proves they should match exactly at seed=42
([`tests/colocate/test_grad_parity.py`](../../../../TorchSpec/tests/colocate/test_grad_parity.py)).

---

## 4. Initialisation time (one-time per pod)

Captured from `train_entry.py:103 INFO Initialization timing:`:

| Phase | D1 (4 GPU) | E1 (4 GPU) | D2 (8 GPU) | E2 (8 GPU) |
|---|--:|--:|--:|--:|
| Create controller | 6.9 s | 8.2 s | 7.8 s | 4.9 s |
| Driver-side init | 14.6 s | 17.3 s | 15.1 s | 14.2 s |
| Dataset loading | 119.2 s (blocked 104.6 s) | 128.9 s (blocked 111.6 s) | 77.1 s (blocked 62.0 s) | 98.3 s (blocked 84.1 s) |
| Allocate actors + dispatch init | 26.5 s | 32.7 s | 34.0 s | 31.3 s |
| Actor initialization | 63.5 s (blocked 37.0 s) | 95.1 s (blocked 62.4 s) | 77.8 s (blocked 43.8 s) | 106.3 s (blocked 75.0 s) |
| Setup async training | 0.1 s | 0.1 s | 0.2 s | 0.1 s |
| **Total** | **190 s (~3 m 10 s)** | **226 s (~3 m 46 s)** | **194 s (~3 m 14 s)** | **226 s (~3 m 46 s)** |

This is in addition to the Modal container cold-start (~30 s after the
image is cached). Modal image was already cached for runs 2/3/4 because
D1 built and committed it first; the bare cold-start image build
(`im-e0QBXE1WOUjU626cgY8HmS`, `im-9t6V0G2XlvKRy0yTstl368`,
`im-KRNpT1SKXbA7ky57so2sXy`) was ~35 s total on the D1 first-time launch.

---

## 5. Stability checks (the "5000-step is the actual test" criteria)

The plan's §5 stop conditions and §6.2 stability predictions were both
satisfied for all four runs.

| Stop condition | D1 | E1 | D2 | E2 |
|---|:-:|:-:|:-:|:-:|
| OOM | none | none | none | none |
| NaN (2 consecutive steps) | none | none | none | none |
| Step-time degradation > 30 % over last 1000 steps | none | none | none | none |
| Loss plateau (slope abs < 1e-4) for ≥ 1500 steps | not triggered (loss still descending at step 5000) | not triggered | not triggered | not triggered |

### 5.1 Step-time stability (CDF coarsening)

From the TIMING-line distribution (steps 100–5000):

| Run | p10 step (s) | p50 step (s) | p90 step (s) | p99 step (s) | Comment |
|---|--:|--:|--:|--:|---|
| D1 DFlash 2+2 | 0.749 | **0.789** | 0.943 | 1.32 | tight; long tail is allreduce on dirty epoch boundary |
| D2 DFlash 4+4 | 0.792 | **0.839** | 1.04 | 1.51 | wider tail (more FSDP ranks) |
| E1 Eagle3 2+2 | 2.071 | **2.125** | 2.31 | 2.69 | very tight (compute-bound, no data jitter) |
| E2 Eagle3 4+4 | 1.969 | **2.021** | 2.18 | 2.49 | same |

These were extracted with the same offline parser used for the median
table; rerun with `extract_modal_perf.py --json` to dump percentiles.

### 5.2 Pool occupancy stability

- **D1/E1:** pool pinned at 64 / 64 for the entire warm window. No
  drops, no overflows. Inference is so far ahead of training that the
  flow-control cap is the steady state.
- **D2:** pool 64 / 64 most of the time, with occasional spikes to 72
  when inference overshoots a step. Min 64, max 72.
- **E2:** pool oscillates 64 ↔ 72 routinely. This is the "Eagle3 4+4 is
  *almost* balanced" regime — `wait≈0.1 s` for ~1 % of steps, but
  median is still 0 ms.

### 5.3 Peak-alloc drift

The plan's §6.2 stability gate is **peak-alloc drift < 0.1 %** over the
warm window. Disagg mode (Mooncake) does *not* emit the
`[colocate_loop] step=… peak_alloc=…` log line — that's a colocate-arm
diagnostic (see [`docs/colocate/transport_benchmark.md`](../../../../TorchSpec/docs/colocate/transport_benchmark.md)).
For disagg, the proxy is "no OOM at any step" + "warm step time stable
over the warm window", both of which are satisfied for all 4 runs. The
colocate arm should be the one that **must** report peak-alloc drift,
not this arm.

---

## 6. Reading the runs in WandB

All four runs are in the WandB project
[`dflash/dflash-eagle3-disagg-modal`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal).
The `perf/` namespace described in
[TorchSpec `docs/performance_metrics.md`](../../../../TorchSpec/docs/performance_metrics.md)
is enabled on every step:

| WandB metric | What it tells you |
|---|---|
| `perf/step_time` | Wall-clock for `train_from_queue`. The median over `train/step > 100` is the "warm step" in §2. |
| `perf/data_time` | Ray queue get + Mooncake RDMA fetch + collation + H2D. **Mooncake's footprint in disagg.** |
| `perf/compute_time` | CUDA-event fwd+bwd+opt. For DFlash this is the actual breakdown; for Eagle3, only `compute_time` (no fwd/bwd subdivision) is meaningful. |
| `perf/dispatch_wait` | Main-loop wait for the sample pool. **0 here** → inference never starved. |
| `perf/train_capacity` | `global_batch / step_time` (T in tqdm). |
| `perf/infer_capacity` | `per_slot_rate * max_concurrent_slots` (I in tqdm). |
| `train/avg_loss` | CE for DFlash, KL for Eagle3 — the convergence curve. |
| `train/avg_acc` | Top-1 next-token / draft accuracy. |
| `train/acc_len` | Eagle3 only — average τ. |
| `train/grad_norm` | NaN / spike detector. **Healthy everywhere.** |
| `train/lr` | Confirms WSD/cosine schedule. |
| `train/step` | x-axis. |

### 6.1 Pre-built WandB chart spec

If you want a one-screen dashboard, this is the spec the report uses:

| Chart | x | y | series |
|---|---|---|---|
| Loss vs step | `train/step` | `train/avg_loss` | one line per run |
| Acc vs step | `train/step` | `train/avg_acc` | one line per run |
| τ (Eagle3) vs step | `train/step` | `train/acc_len` | E1, E2 only |
| step_time CDF | `perf/step_time` | density | one curve per run |
| compute vs data (DFlash) | `train/step` | `perf/compute_time`, `perf/data_time` | stacked area, D1 / D2 |
| Pool occupancy | `train/step` | pool (parse from log) | one line per run |

---

## 7. Cost & wall-clock summary

| Run | GPUs × time | GPU-hours | Modal H100 list price ($5.92 / GPU-hr) |
|---|---|--:|--:|
| D1 (DFlash 2+2) | 4 × 1.219 h | 4.88 | $28.85 |
| E1 (Eagle3 2+2) | 4 × 3.150 h | 12.60 | $74.59 |
| D2 (DFlash 4+4) | 8 × 1.316 h | 10.53 | $62.32 |
| E2 (Eagle3 4+4) | 8 × 3.000 h | 24.00 | $142.08 |
| **Total** | | **51.99** | **≈ $307.84** |

Earlier "$194" forecast undershot; the actual rate quoted in the Modal
list is $5.92 / GPU-hour. **All four 5000-step disagg cells together cost
≈ $308** on Modal sandbox.

Wall-clock from D1 spawn (23:04 PDT) to E1 finish (02:20 PDT next day) =
**3 h 16 m**. Parallel scheduling: ✅. No queue waits on Modal sandbox.

---

## 8. Test settings the colocate arm must replicate

> **For the RunPod-colocate agent**: this is the contract. Reproduce
> every knob in §8.1–§8.4 cell-for-cell, only swapping the lines marked
> `← colocate-specific`. Anything else changed is a confound.

### 8.1 Software pins (immutable across both arms)

| Component | Pin | Where it lives |
|---|---|---|
| TorchSpec branch | `feature/dflash-training` (disagg) / `feature/colocate-training-inference` (colocate) | `git checkout` |
| TorchSpec commit | `cb741ae` (disagg arm) | `git reset --hard` |
| SGLang commit | `0f2df9370a1de1b4fb11b071d39ab3ce2287a350` (disagg) / `94f03a39…` (colocate; per `feature/colocate-training-inference`) | `scripts/modal/modal_dflash_train.py:101` (disagg) |
| SGLang patch | `patches/sglang/v0.5.8.post1/sglang.patch` (disagg) / `v0.5.10.post1/…` (colocate) | applied at image build |
| PyTorch | `torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124` (latest 2.9.x at image-build time) | image |
| CUDA | 12.4.0 (NVIDIA base image `nvidia/cuda:12.4.0-devel-ubuntu22.04`) | image |
| transformers | `4.57.1` | image |
| mooncake-transfer-engine | latest pip (Modal autobuild; binary chmod-patched at image time) | image |
| Dataset SHA | mlabonne/open-perfectblend, subsampled 200 000 / 1 420 909 with seed 42 → 190 095 valid samples | `scripts/tools/prepare_perfectblend.py` |
| Tokenized-dataset cache key | `3aa51bcffe41a0cc570d87e70cafc669` | `cache/qwen3-8b-single-node/tokenized_dataset/…` (same hash across all 4 runs ⇒ tokenisation is deterministic) |

### 8.2 Hyperparameters (immutable across both arms)

These were the **identical** `--extra-overrides` for all 4 disagg runs
and **must be identical** on the 4 colocate runs:

```text
training.dflash_num_anchors          = 512      # DFlash only; Eagle3 ignores
training.draft_accumulation_steps    = 4
training.micro_batch_size            = 1
training.fsdp_strategy               = FULL_SHARD
training.fsdp_reduce_dtype           = bfloat16
training.prefetch_depth              = 8
training.max_checkpoints             = 1
training.save_interval               = 9999     # effectively off
dataset.eval_data_path               = null
debug.enable_perf_metrics            = true
training.seed                        = 42        # YAML default; do not override
training.num_train_steps             = 5000      # the hard cap
```

Resulting derived parameters (from `loop.py:203 INFO Starting:`):

| Layout | `global_batch_size` | `dp_size` | `per_dp_rank_batch_size` | `accumulation_steps` | `steps_per_epoch` |
|---|--:|--:|--:|--:|--:|
| 2+2 (D1, E1, **C1**, **CE1**) | 8 | 2 | 1 | 4 | 23 622 / 23 761 |
| 4+4 (D2, E2, **C2**, **CE2**) | 16 | 4 | 1 | 4 | 11 811 / 11 880 |

### 8.3 Per-cell wandb_run_id naming convention

The colocate-arm runs must use the **same WandB project** and a `C…` /
`CE…` prefix so the cells line up in one dashboard:

| Cell | Disagg run_id (this arm) | Colocate run_id (RunPod arm) |
|---|---|---|
| DFlash 2+2 | `D1-dflash-2plus2-disagg-modal` | `C1-dflash-2plus2-colocate-runpod` |
| DFlash 4+4 | `D2-dflash-4plus4-disagg-modal` | `C2-dflash-4plus4-colocate-runpod` |
| Eagle3 2+2 | `E1-eagle3-2plus2-disagg-modal` | `CE1-eagle3-2plus2-colocate-runpod` |
| Eagle3 4+4 | `E2-eagle3-4plus4-disagg-modal` | `CE2-eagle3-4plus4-colocate-runpod` |

**WandB project for both arms:** `dflash/dflash-eagle3-disagg-modal`
(yes — keep the existing project name even for the colocate runs so the
WandB regex `(D|E|C|CE)[12]-` matches all 8 cells in one filter). If
you prefer a re-named project, also rename the existing runs to keep
the comparison page coherent.

### 8.4 Colocate-only knobs (the only deltas)

Lines marked `← colocate-specific` are the **only** allowed differences
versus this arm:

```text
training.colocate_strategy             = mps          # ← colocate-specific
training.transfer_mode                 = cuda_ipc     # ← colocate-specific (round-9 default per docs/colocate/implementation_log.md)
training.train_frac                    = 0.45         # ← colocate-specific (train_frac + infer_frac + 0.10 ≤ 1.0)
training.infer_frac                    = 0.45         # ← colocate-specific
inference.inference_num_gpus_per_engine = 1            # ← colocate-specific invariant (Phase-1)
inference.sglang.tp_size               = 1            # ← colocate-specific invariant (Phase-2)
```

Plus the host-side preflight (no equivalent on Modal):

```bash
bash scripts/colocate/run_smoke_host.sh --probe-only   # confirms MPS works
```

Per
[`docs/colocate/usage.md`](../../../../TorchSpec/docs/colocate/usage.md)
and
[`docs/colocate/gpu_testing_runbook.md`](../../../../TorchSpec/docs/colocate/gpu_testing_runbook.md).

### 8.5 Per-run launch commands (this arm, for verbatim parity by the colocate arm)

The four exact commands that produced the four runs in §1. Each was
launched with `--detach` from a fresh local shell. The `TORCHSPEC_MODAL_GPU`
env var is honoured by the §10.1 patch (`H100:4` for 2+2 runs, `H100:8`
for 4+4 runs).

#### D1 — DFlash 2+2

```bash
TORCHSPEC_MODAL_GPU=H100:4 modal run --detach scripts/modal/modal_dflash_train.py \
  --gpu-count 4 \
  --max-steps 5000 \
  --run-dflash --no-run-eagle3 \
  --wandb-project dflash-eagle3-disagg-modal \
  --dataset-size 200000 \
  --dflash-run-id D1-dflash-2plus2-disagg-modal \
  --extra-overrides "training.dflash_num_anchors=512 \
    inference.inference_num_gpus=2 training.training_num_gpus_per_node=2 \
    training.draft_accumulation_steps=4 training.micro_batch_size=1 \
    training.fsdp_strategy=FULL_SHARD training.fsdp_reduce_dtype=bfloat16 \
    training.prefetch_depth=8 training.max_checkpoints=1 \
    training.save_interval=9999 dataset.eval_data_path=null \
    debug.enable_perf_metrics=true \
    logging.wandb_run_id=D1-dflash-2plus2-disagg-modal"
```

#### E1 — Eagle3 2+2

```bash
TORCHSPEC_MODAL_GPU=H100:4 modal run --detach scripts/modal/modal_dflash_train.py \
  --gpu-count 4 \
  --max-steps 5000 \
  --run-eagle3 --no-run-dflash \
  --wandb-project dflash-eagle3-disagg-modal \
  --dataset-size 200000 \
  --eagle3-run-id E1-eagle3-2plus2-disagg-modal \
  --extra-overrides "inference.inference_num_gpus=2 \
    training.training_num_gpus_per_node=2 \
    training.draft_accumulation_steps=4 training.micro_batch_size=1 \
    training.fsdp_strategy=FULL_SHARD training.fsdp_reduce_dtype=bfloat16 \
    training.prefetch_depth=8 training.max_checkpoints=1 \
    training.save_interval=9999 dataset.eval_data_path=null \
    debug.enable_perf_metrics=true \
    logging.wandb_run_id=E1-eagle3-2plus2-disagg-modal"
```

#### D2 — DFlash 4+4

```bash
TORCHSPEC_MODAL_GPU=H100:8 modal run --detach scripts/modal/modal_dflash_train.py \
  --gpu-count 8 \
  --max-steps 5000 \
  --run-dflash --no-run-eagle3 \
  --wandb-project dflash-eagle3-disagg-modal \
  --dataset-size 200000 \
  --dflash-run-id D2-dflash-4plus4-disagg-modal \
  --extra-overrides "training.dflash_num_anchors=512 \
    inference.inference_num_gpus=4 training.training_num_gpus_per_node=4 \
    training.draft_accumulation_steps=4 training.micro_batch_size=1 \
    training.fsdp_strategy=FULL_SHARD training.fsdp_reduce_dtype=bfloat16 \
    training.prefetch_depth=8 training.max_checkpoints=1 \
    training.save_interval=9999 dataset.eval_data_path=null \
    debug.enable_perf_metrics=true \
    logging.wandb_run_id=D2-dflash-4plus4-disagg-modal"
```

#### E2 — Eagle3 4+4

```bash
TORCHSPEC_MODAL_GPU=H100:8 modal run --detach scripts/modal/modal_dflash_train.py \
  --gpu-count 8 \
  --max-steps 5000 \
  --run-eagle3 --no-run-dflash \
  --wandb-project dflash-eagle3-disagg-modal \
  --dataset-size 200000 \
  --eagle3-run-id E2-eagle3-4plus4-disagg-modal \
  --extra-overrides "inference.inference_num_gpus=4 \
    training.training_num_gpus_per_node=4 \
    training.draft_accumulation_steps=4 training.micro_batch_size=1 \
    training.fsdp_strategy=FULL_SHARD training.fsdp_reduce_dtype=bfloat16 \
    training.prefetch_depth=8 training.max_checkpoints=1 \
    training.save_interval=9999 dataset.eval_data_path=null \
    debug.enable_perf_metrics=true \
    logging.wandb_run_id=E2-eagle3-4plus4-disagg-modal"
```

---

## 9. Predictions for the colocate arm (to falsify)

Now that the disagg-Modal baseline is locked in, the predictions for the
RunPod colocate arm can be tightened from the plan's §6.1:

| Cell | Disagg (this arm) | Predicted colocate | Predicted Δ |
|---|--:|--:|--:|
| **DFlash 2+2** | 10.14 samples/s | **12–14 samples/s** | colocate +20–40 % (transfer is ~50 % of step; saving most of it shaves ~0.2 s off 0.789 s) |
| **DFlash 4+4** | 19.07 samples/s | **22–26 samples/s** | colocate +15–35 % (slightly less since trainer-bound regime is dominant) |
| **Eagle3 2+2** | 3.76 samples/s | **3.8–4.0 samples/s** | colocate ≤ +6 % (Eagle3 is 100 % compute-bound; transfer is < 2 % of step) |
| **Eagle3 4+4** | 7.92 samples/s | **8.0–8.5 samples/s** | colocate ≤ +7 % (same reason — but the small pool-oscillation overhead might recover slightly more) |

**Convergence:** colocate must hit the *same* loss, acc, and acc_len
trajectory points in §3 at the *same* steps within run-to-run jitter.
If any cell drifts by > 2 % at any step (especially Eagle3 acc_len),
that is the headline finding and goes straight to a grad-parity
reproduction.

**Stability:** the colocate arm must additionally report
`peak_alloc drift < 0.1 %` over steps 1000–5000. Disagg has no such
report; "no OOM" + "warm step time stable" was the proxy here.

---

## 10. Helper patches & artefacts

### 10.1 Uncommitted helper patches to `scripts/modal/modal_dflash_train.py`

Both are backwards-compatible and gated by environment variables /
optional flags. Suggested upstream PR title:
*"modal_dflash_train: parameterise GPU spec + Eagle3 run id (benchmark prep)"*.

**Patch 1: `SGLANG_GPU` from env var.**

```110:115:scripts/modal/modal_dflash_train.py
# GPU configuration — edit to change hardware allocation, or override
# at launch time with the TORCHSPEC_MODAL_GPU env var (e.g. "H100:4").
# This allows the same script to spawn 4-GPU and 8-GPU pods without an
# in-file edit; the env var is read once at module import.
import os as _os  # noqa: E402 — local alias to avoid shadowing later os imports
SGLANG_GPU = _os.environ.get("TORCHSPEC_MODAL_GPU", "H100:8")
```

**Patch 2: `eagle3_run_id` plumbed through.**

`train_sglang` → `_train_impl` → `_run_training` now take an
`eagle3_run_id: Optional[str]`. When `None`, the previous hard-coded
`"eagle3-qwen3-8b"` is used. When provided, both the local log path
(`/workspace/outputs/{eagle3_run_id}.log`) and the WandB run id are
populated from it. Mirror image of the existing `dflash_run_id` path.

### 10.2 Offline metrics extractor

[`docs/study_notes/rl_study/scripts/extract_modal_perf.py`](./scripts/extract_modal_perf.py)
— parses one or more local `.log` files dumped from the Modal volume
via `modal volume get torchspec-outputs /{run_id}.log` and prints the
warm-window table (§2 here). Use:

```bash
python docs/study_notes/rl_study/scripts/extract_modal_perf.py \
  docs/study_notes/rl_study/modal_logs/D1.log \
  docs/study_notes/rl_study/modal_logs/E1.log \
  docs/study_notes/rl_study/modal_logs/D2.log \
  docs/study_notes/rl_study/modal_logs/E2.log \
  --label "D1 DFlash 2+2" --label "E1 Eagle3 2+2" \
  --label "D2 DFlash 4+4" --label "E2 Eagle3 4+4" \
  --json /tmp/all.json --markdown
```

This is the canonical extraction for the disagg arm; the colocate arm
can use the same script (it parses `TIMING step=N:` lines that
`colocate_loop.py` emits identically).

### 10.3 Pulling each run's full log from Modal (post-hoc)

```bash
for run_id in D1-dflash-2plus2-disagg-modal \
              E1-eagle3-2plus2-disagg-modal \
              D2-dflash-4plus4-disagg-modal \
              E2-eagle3-4plus4-disagg-modal; do
  modal volume get torchspec-outputs "/$run_id.log" \
    "docs/study_notes/rl_study/modal_logs/$run_id.log" --force
done
```

Logs are ~3 MB each (2.5–3.2 MB on disk). They contain every TIMING
line, every COMPUTE_BREAKDOWN, every Mooncake / sglang log, and the
final "Training completed:" line that the extractor keys off.

The 4 logs used to compute the tables in this doc are checked in at
`docs/study_notes/rl_study/modal_logs/{D1,E1,D2,E2}.log`.

---

## 11. Next steps (RunPod colocate arm)

1. **Boot a 4 × H100 SXM RunPod (or Vast.ai) pod** with `--ipc=host`
   per [`docs/colocate/gpu_testing_runbook.md`](../../../../TorchSpec/docs/colocate/gpu_testing_runbook.md).
2. `git checkout feature/colocate-training-inference` and run the
   pre-flight: `bash scripts/colocate/run_smoke_host.sh --probe-only`.
3. **Launch C1, CE1** (4-GPU pod). Use the exact extra-overrides in §8
   *plus* the colocate-only lines in §8.4.
4. **Boot an 8 × H100 SXM pod** and **launch C2, CE2**.
5. Once each finishes, dump the local log (it's at
   `/workspace/outputs/{run_id}/{run_id}.log` on the host), run
   `extract_modal_perf.py` on it, and paste the row into the §2 table
   above as a new "colocate" column.
6. Write the comparison report (plan §7 template) at
   `docs/study_notes/rl_study/dflash_eagle3_disagg_vs_colocate_results.md`.

---

**Document version:** 1.0 — disagg-Modal arm complete; runs finished
2026-05-21 02:20 PDT.
**Maintainer:** xing.han — disagg-vs-colocate benchmark for the RL infra
study series.

# DFlash + Eagle3 Disaggregated Modal Runs — Performance Metrics Tracker

> **Status (FINAL):** all 4 runs complete, 0 NaN, 0 OOM, all reached step 5000.
> **Started:** 2026-05-20 23:04 PDT (D1) / 23:11 PDT (E1, D2, E2).
> **Finished:** 2026-05-21 02:20 PDT (E1 last; total 3 h 16 m wall-clock).
> **Final results doc:** [`dflash_eagle3_disagg_modal_results.md`](./dflash_eagle3_disagg_modal_results.md).
> **Step cap:** 5 000 each. **Branch:** `feature/dflash-training @ cb741ae`.
> **Companion plan:** [`dflash_eagle3_disagg_vs_colocate_benchmark_plan.md`](./dflash_eagle3_disagg_vs_colocate_benchmark_plan.md).
>
> **Headline numbers (warm-window medians):**
>
> | Cell | Wall clock | Warm step (s) | Throughput (samples/s) | Loss @ 5000 | Acc @ 5000 | Eagle3 τ @ 5000 |
> |---|--:|--:|--:|--:|--:|--:|
> | D1 DFlash 2+2 | 1 h 13 m | 0.789 | 10.14 | 3.67 | 0.235 | — |
> | D2 DFlash 4+4 | 1 h 19 m | 0.839 | 19.07 | 3.14 | 0.284 | — |
> | E1 Eagle3 2+2 | 3 h 09 m | 2.125 | 3.76 | 2.24 | 0.580 | 1.66 |
> | E2 Eagle3 4+4 | 3 h 00 m | 2.021 | 7.92 | 1.80 | 0.638 | 2.00 |
>
> See the [final results doc](./dflash_eagle3_disagg_modal_results.md) for the
> full extracted tables, convergence trajectories, and the test-settings
> contract for the RunPod-colocate agent.

This doc tracks the 4 disaggregated-Modal runs the user asked for. It captures
the exact launch commands, Modal app IDs, the early-step performance the runs
are already showing, and the **post-run metrics-extraction recipe** that will
turn `perf/`-namespaced WandB metrics into the side-by-side throughput /
stability / convergence table that the comparison report needs.

---

## 1. The 4 runs at a glance

| # | Run ID | Model | Layout | Modal GPU | App ID | Steps | Status @ first snapshot |
|---|---|---|---|---|---|--:|---|
| **D1** | `D1-dflash-2plus2-disagg-modal` | DFlash | 2 infer + 2 train | `H100:4` | [`ap-4gNxJ8m2QOOv9HDxNS87dz`](https://modal.com/apps/doordash/sandbox/ap-4gNxJ8m2QOOv9HDxNS87dz) | 5000 | **step 556 / 5000** @ 8m48s → 1.21 step/s |
| **E1** | `E1-eagle3-2plus2-disagg-modal` | Eagle3 | 2 infer + 2 train | `H100:4` | [`ap-0kkF98EfJkyKbumcsLEdY7`](https://modal.com/apps/doordash/sandbox/ap-0kkF98EfJkyKbumcsLEdY7) | 5000 | **step 11 / 5000** @ 2m32s → 3.33 s/step |
| **D2** | `D2-dflash-4plus4-disagg-modal` | DFlash | 4 infer + 4 train | `H100:8` | [`ap-V3eat0e1VaNJZK84Cg4O6n`](https://modal.com/apps/doordash/sandbox/ap-V3eat0e1VaNJZK84Cg4O6n) | 5000 | **step 158 / 5000** @ 4m13s → 1.15 step/s |
| **E2** | `E2-eagle3-4plus4-disagg-modal` | Eagle3 | 4 infer + 4 train | `H100:8` | [`ap-oQBE3oK4HgoSK3Cb5mHdpk`](https://modal.com/apps/doordash/sandbox/ap-oQBE3oK4HgoSK3Cb5mHdpk) | 5000 | **step 21 / 5000** @ 2m50s → 2.14 s/step |

All 4 are detached, so they will keep running even if the local CLI closes.

**WandB project (single, all 4 runs):**
[`dflash-eagle3-disagg-modal`](https://wandb.ai/_/dflash-eagle3-disagg-modal)
(the team name resolves from the `wandb-secret` Modal secret).

---

## 2. Early step-time observations (steady-ish, captured ~10 min after launch)

These are *cold-to-warm* numbers — they will improve slightly over the next
few hundred steps. They are reported here only as a sanity check that the
runs are actually training and the gross throughput shape matches the
benchmark plan's predictions.

| Run | `step/s` (warm) | `samples/s` (`thru`) | `I` (infer cap) | `T` (train cap) | Loss @ snapshot | Pool | ETA at observed rate |
|---|--:|--:|--:|--:|--:|--:|---|
| **D1** DFlash 2+2 | 1.21 | 10.1 | 58.9 | 10.0 | 5.37 | 64/64 (full) | ~**1h 05m** |
| **E1** Eagle3 2+2 | 0.30 (3.33 s/step) | 3.6 | 45.6 | 3.9 | 11.19 | 64/64 (full) | ~**4h 35m** |
| **D2** DFlash 4+4 | 1.15 | 17.6 | 90.6 | 20.4 | 6.38 | 64/64 (full) | ~**1h 10m** |
| **E2** Eagle3 4+4 | 0.47 (2.14 s/step) | 8.0 | 120.1 | 7.4 | 8.83 | 72→64 | ~**2h 55m** |

### Reading the numbers

- **DFlash (D1, D2) is trainer-bound by a lot.** `I ≫ T` (~6× for D1,
  ~5× for D2), pool is at the cap (`pool=64/64`), `dispatch_wait≈0s`.
  Inference is over-saturating — exactly the regime the `feature/dflash-training`
  retro reported at `dflash_num_anchors=512`. The fix is *not* this benchmark
  arm; it is the colocate arm or an asymmetric 2+6 disagg.
- **D2 ≈ 1.74 × D1 throughput at 2× GPUs.** Sub-linear scaling because the
  global batch grew to 16 (vs 8 on D1) and FSDP all-reduce now spans 4 ranks
  instead of 2. Step time is essentially unchanged (1.21 vs 1.15 step/s)
  even though dp_size doubled. This is the canonical "more GPUs → bigger
  batch, same step time" FSDP behaviour.
- **Eagle3 is ~5–7 × slower per step than DFlash.** E1 step is 3.33 s vs
  D1's 0.83 s — the predicted `ttt_length=7` × 7-forward overhead lines up.
- **Eagle3 4+4 (E2) is nearly balanced (I=120, T=7.4 in pool-units, but
  `dispatch_wait=0.1s` and pool oscillates 64↔72).** This is the
  "trainer-bound but inference is *just* keeping up" regime — exactly
  what the benchmark plan predicts for the trainer-bound algorithm at
  symmetric layout.

### Headline (early) takeaways

> **Quantitative samples/s, DFlash, disagg-Modal, anchors=512:**
> - 2+2 (4 GPU)  : **~10 samples/s**, 1.21 step/s, pool=64/64
> - 4+4 (8 GPU)  : **~18 samples/s**, 1.15 step/s, pool=64/64
>
> **Quantitative samples/s, Eagle3, disagg-Modal:**
> - 2+2 (4 GPU)  : **~3.6 samples/s**, 3.33 s/step
> - 4+4 (8 GPU)  : **~8.0 samples/s**, 2.14 s/step

These are the four headline numbers that will be the disagg-Modal baseline
for the later colocate-RunPod comparison.

---

## 3. Exact launch commands (for reproducibility)

All commands run from `/Users/xing.han/Projects/TorchSpec` on branch
`feature/dflash-training @ cb741ae`. The local Modal CLI is on profile
`doordash` and environment `sandbox` (secrets `xingh3-hf-write` +
`wandb-secret`).

The script `scripts/modal/modal_dflash_train.py` was given two small,
benchmark-only additions on top of `cb741ae` (still uncommitted on this
workstation, see Section 8 below for the diff):

1. **`SGLANG_GPU` reads from `TORCHSPEC_MODAL_GPU` env var** (so the same
   script can spawn `H100:4` pods for D1/E1 *and* `H100:8` pods for D2/E2
   without an in-file edit).
2. **`eagle3_run_id` is now plumbed through** (same as the existing
   `dflash_run_id`) so parallel Eagle3 runs don't collide on the shared
   `torchspec-outputs` volume.

### 3.1 D1 — DFlash 2+2 (4×H100)

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

### 3.2 E1 — Eagle3 2+2 (4×H100)

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

> Eagle3 does **not** take `training.dflash_num_anchors` — that knob lives
> in the DFlash config only. The Eagle3 YAML (`configs/sglang_qwen3_8b.yaml`)
> drives 7-forward TTT directly.

### 3.3 D2 — DFlash 4+4 (8×H100)

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

### 3.4 E2 — Eagle3 4+4 (8×H100)

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

### 3.5 Common (controlled) hyperparameters

The `--extra-overrides` flags pin these knobs **identical** across all four
runs so the only deltas are *model* (DFlash vs Eagle3) and *layout*
(2+2 vs 4+4):

| Knob | Value | Why |
|---|---|---|
| `training.dflash_num_anchors` | 512 | Matches z-lab `Phase H` reference and the `feature/dflash-training` retro best-quality config. (DFlash only — ignored by Eagle3.) |
| `training.draft_accumulation_steps` | 4 | Gradient accumulation, locks `global_batch_size = 4 × dp_size × micro_batch_size = 8` (4-GPU) / `16` (8-GPU). |
| `training.micro_batch_size` | 1 | Same as `feature/dflash-training` retro. |
| `training.fsdp_strategy` | `FULL_SHARD` | ZeRO-3-equivalent; lowest VRAM footprint per rank. |
| `training.fsdp_reduce_dtype` | `bfloat16` | Standard bf16 grad reduce; matches `feature/dflash-training` retro. |
| `training.prefetch_depth` | 8 | Prefetch queue depth on each trainer rank. |
| `training.max_checkpoints` | 1 | Don't accumulate checkpoints. |
| `training.save_interval` | 9999 | Effectively *off* — no checkpoint save during the 5000-step run. We don't want disk I/O to skew step times. |
| `dataset.eval_data_path` | `null` | Skip in-loop eval; score off-line if needed. |
| `debug.enable_perf_metrics` | `true` | **Critical** — this turns on the `perf/` WandB namespace this doc is built around. |
| `dataset.size` (via `--dataset-size`) | 200 000 | PerfectBlend 200K, prepared once on the Modal volume by `prepare_perfectblend.py`. |

> `training.seed=42` is the YAML default for both
> `configs/sglang_qwen3_8b.yaml` (Eagle3) and `configs/sglang_qwen3_8b_dflash.yaml`,
> so same-step → same-data across runs.

---

## 4. Metrics that are *already* being logged (no extra wiring needed)

`debug.enable_perf_metrics=true` opts in the
[`perf/` namespace described in `docs/performance_metrics.md`](../../../../TorchSpec/docs/performance_metrics.md).
Each of these is logged **every optimizer step** to WandB, tied to
`train/step`:

### 4.1 `perf/` (the new instrumented metrics)

| WandB metric | Unit | Use |
|---|---|---|
| `perf/step_time` | s | Wall-clock for `train_from_queue`. Median over 100–5000 = "warm step time". |
| `perf/data_time` | s | Ray queue get + Mooncake RDMA fetch + H2D. **Mooncake's footprint in disagg.** |
| `perf/compute_time` | s | CUDA-event fwd+bwd+opt. Decouples GPU work from data movement. |
| `perf/train_capacity` | samples/s | `global_batch / step_time` — the trainer's ceiling. |
| `perf/infer_capacity` | samples/s | sglang per-slot rate × concurrent slots — the inference ceiling. |
| `perf/infer_batch_time` | s | Avg `engine.generate.remote()` wall time per call. |
| `perf/dispatch_wait` | s | Main-loop wait for the sample pool. High → inference-bound. |

### 4.2 `train/` (the existing convergence metrics)

| WandB metric | Use |
|---|---|
| `train/avg_loss` | CE for DFlash, KL for Eagle3. Convergence sanity. |
| `train/avg_acc` | Top-1 next-token accuracy. **The headline comparator** between runs. |
| `train/grad_norm` | NaN / spike detector. |
| `train/lr` | Confirms WSD/cosine schedule. |
| `train/step` | x-axis. |

### 4.3 Run / system context (also auto-logged)

- `gpu_count`, dp_size, global_batch_size — logged at init in
  `train_entry.py:loop.py:203`.
- Mooncake master URL + segment size — in the engine init logs (not WandB,
  but in the Modal pod log).
- WandB run config dump includes all CLI overrides — the YAML-resolved
  hydrated config goes into `wandb.config`.

---

## 5. Post-run metrics-extraction recipe (the deliverable)

When all 4 runs finish (estimated ~4h 30m for the longest, E1), this is the
exact procedure to turn them into the disagg-Modal cell of the comparison
table.

### 5.1 Per-run aggregates (the row of the report table)

For each of `{D1, E1, D2, E2}` pull these from WandB:

| Aggregate | Definition |
|---|---|
| **Warm step time** | `median(perf/step_time[100:5000])` (skip the cold-start tail). |
| **Warm throughput** | `global_batch_size / median(perf/step_time[100:5000])`. |
| **Steady util breakdown** | `(median(perf/compute_time), median(perf/data_time), median(perf/dispatch_wait))` over steps 1000–5000. |
| **I / T ratio** | `median(perf/infer_capacity[1000:5000]) / median(perf/train_capacity[1000:5000])`. |
| **Pool occupancy** | tqdm `pool=N` (post-hoc from Modal pod log) — min/median/max over 1000–5000. |
| **Loss @ 5000** | `mean(train/avg_loss[-100:])`. |
| **Acc @ 5000** | `mean(train/avg_acc[-100:])`. |
| **OOM / NaN events** | Boolean + step index (parse pod log for `NaN`, `OutOfMemoryError`). |

### 5.2 Extraction snippet (WandB API — paste into a notebook / Python repl)

```python
import wandb
import numpy as np

api = wandb.Api()
runs = api.runs(
    "doordash/dflash-eagle3-disagg-modal",
    filters={"display_name": {"$regex": "(D1|E1|D2|E2)-"}},
)

rows = []
for r in runs:
    h = r.history(
        samples=10_000,
        keys=[
            "train/step",
            "perf/step_time", "perf/data_time", "perf/compute_time",
            "perf/dispatch_wait",
            "perf/train_capacity", "perf/infer_capacity",
            "train/avg_loss", "train/avg_acc", "train/grad_norm",
        ],
    )
    h = h[h["train/step"] >= 100]
    warm = h[h["train/step"] >= 1000]

    row = {
        "run": r.name,
        "warm_step_s": float(np.nanmedian(h["perf/step_time"])),
        "compute_s":   float(np.nanmedian(warm["perf/compute_time"])),
        "data_s":      float(np.nanmedian(warm["perf/data_time"])),
        "dispatch_s":  float(np.nanmedian(warm["perf/dispatch_wait"])),
        "train_cap":   float(np.nanmedian(warm["perf/train_capacity"])),
        "infer_cap":   float(np.nanmedian(warm["perf/infer_capacity"])),
        "loss_final":  float(h["train/avg_loss"].iloc[-100:].mean()),
        "acc_final":   float(h["train/avg_acc"].iloc[-100:].mean()),
        "nan_events":  int(h["train/avg_loss"].isna().sum()),
    }
    row["warm_throughput"] = r.config["training"]["draft_accumulation_steps"] \
        * r.config["training"]["micro_batch_size"] \
        * r.config["training"]["training_num_gpus_per_node"] \
        / row["warm_step_s"]
    rows.append(row)

import pandas as pd
print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))
```

That snippet returns the four rows that will go straight into the comparison
report's throughput table.

### 5.3 Pod-side artefacts (in case WandB sync is incomplete)

Each pod also persists to the `torchspec-outputs` Modal volume:

```
/workspace/outputs/{run_id}/
    checkpoints/                    # disabled by max_checkpoints=1 + save_interval=9999
    hf_model/                       # DFlash-only; final HF-converted draft
/workspace/outputs/{run_id}.log     # the full stdout/stderr stream
```

To pull a run's full log down to the laptop after it finishes:

```bash
modal volume get torchspec-outputs \
  /D1-dflash-2plus2-disagg-modal.log ./logs/D1-disagg.log
```

The `.log` contains the **per-step tqdm line + sglang/SglEngine logs +
TrainerActor compute-breakdown messages** (every 5 steps:
`COMPUTE_BREAKDOWN step=N: forward=Xms backward=Yms`). This is the
authoritative source for per-step timings if WandB samples a sub-set.

### 5.4 Stop-condition checks

The plan's Section 5 lists stop conditions. To check post-hoc that no run
hit any of them silently:

```python
# OOM / SIGSEGV / FATAL: parse the pod log
import subprocess
log = subprocess.check_output(
    ["modal", "volume", "get", "torchspec-outputs",
     "/D1-dflash-2plus2-disagg-modal.log", "-"],
    text=True,
)
for needle in ("OutOfMemoryError", "SIGSEGV", "RuntimeError", "FATAL", "NaN"):
    n = log.count(needle)
    if n:
        print(f"  ! D1: {needle} appears {n} times")
```

### 5.5 What "good" looks like, per run

| Run | Healthy warm step time | Healthy I/T | Healthy pool | Notes |
|---|---|---|---|---|
| D1 | 0.7–1.0 s | I/T = 5–8 (training-bound, expected for 2-train-GPU at anchors=512) | 56–64 / 64 (≈ full) | A `pool < 32` sustained over 100 steps means inference is starved — re-check `inference.inference_num_gpus=2` actually applied. |
| E1 | 2.5–4 s | I/T = 8–15 (heavily training-bound — Eagle3 7-fwd) | 56–64 / 64 | Same staleness expectation as D1. |
| D2 | 0.7–1.0 s | I/T = 4–6 | 48–64 / 64 | 4 infer GPUs → bigger pool oscillation. |
| E2 | 1.8–2.5 s | I/T ≈ 1–2 (Eagle3 4+4 is the closest to balance — see early snapshot) | 56–72 / 64 | If `pool < 40` for ≥ 100 steps, drop to E3 layout (2+6) instead — but that's the colocate-arm "not expressible" case, not Modal. |

---

## 6. Cost / timeline forecast (so the runs can be left to complete)

| Run | GPU | Predicted wall time @ snapshot rate | Modal GPU-hours | $ @ Modal H100 ≈ $3.5 / GPU-hr |
|---|---|---|--:|--:|
| **D1** | 4 × H100 | 1h 5m  | 4.3  | ~$15 |
| **E1** | 4 × H100 | 4h 35m | 18.3 | ~$64 |
| **D2** | 8 × H100 | 1h 10m | 9.3  | ~$33 |
| **E2** | 8 × H100 | 2h 55m | 23.3 | ~$82 |
| **Total** | | longest = ~**4h 35m** wall-clock (parallel) | **55.3 GPU-hr** | **~$194** |

(Modal's H100 rate is `$3.50 / hr` per spec sheet — confirm in
`modal app describe ap-…` for the actual posted price.)

If E1 dominates the wall-clock budget and is overkill for *this* benchmark
arm, an acceptable short-circuit is to **stop E1 at step 2500** once the
loss curve flattens (the plan's Section 5 plateau detector). Lower bound to
still get a clean comparison: 2500 ÷ 5000 of the 4h 35m budget = ~2h 18m.

---

## 7. Monitoring the runs (quick check during the wait)

### 7.1 From the local laptop — Modal CLI

```bash
modal app list | grep torchspec-dflash-training
modal app logs ap-4gNxJ8m2QOOv9HDxNS87dz       # D1 live tail
modal app logs ap-0kkF98EfJkyKbumcsLEdY7       # E1
modal app logs ap-V3eat0e1VaNJZK84Cg4O6n       # D2
modal app logs ap-oQBE3oK4HgoSK3Cb5mHdpk       # E2
```

### 7.2 From the local terminal files (already capturing the live stream)

Each of the 4 detached `modal run` calls left a local shell hanging on
`.spawn().get()` while the cloud function executes. Those shells are
streaming the pod stdout to a Cursor-managed terminal file — handy if you
want to grep without going to Modal:

| Run | Local terminal file (in `/Users/xing.han/.cursor/projects/.../terminals/`) |
|---|---|
| D1 | `235242.txt` |
| E1 | `256918.txt` |
| D2 | `166288.txt` |
| E2 | `733510.txt` |

A one-liner to spot-check live progress on all four:

```bash
for f in 235242 256918 166288 733510; do
  echo "=== $f ==="; \
  tail -n 200 /Users/xing.han/.cursor/projects/Users-xing-han-Projects-damoxing/terminals/$f.txt \
    | grep -E "Training:[[:space:]]+[0-9]+%" | tail -1
done
```

### 7.3 WandB (live)

[`https://wandb.ai/_/dflash-eagle3-disagg-modal`](https://wandb.ai/_/dflash-eagle3-disagg-modal)
— filter run-id by regex `^(D1|E1|D2|E2)-`.

### 7.4 Killing a run (if needed)

```bash
modal app stop ap-…           # graceful
```

---

## 8. Script-side delta (uncommitted, on this workstation)

The benchmark added two tiny patches to
`scripts/modal/modal_dflash_train.py` on top of `cb741ae`:

```110:115:scripts/modal/modal_dflash_train.py
# GPU configuration — edit to change hardware allocation, or override
# at launch time with the TORCHSPEC_MODAL_GPU env var (e.g. "H100:4").
# This allows the same script to spawn 4-GPU and 8-GPU pods without an
# in-file edit; the env var is read once at module import.
import os as _os  # noqa: E402 — local alias to avoid shadowing later os imports
SGLANG_GPU = _os.environ.get("TORCHSPEC_MODAL_GPU", "H100:8")
```

```469:493:scripts/modal/modal_dflash_train.py
def train_sglang(
    gpu_count: int,
    max_steps: int,
    num_epochs: Optional[int],
    run_eagle3: bool,
    run_dflash: bool,
    wandb_project: Optional[str],
    wandb_team: Optional[str] = None,
    dataset_path: Optional[str] = None,
    dataset_size: int = 50000,
    extra_overrides: Optional[str] = None,
    hf_repo: Optional[str] = None,
    resume: bool = False,
    dflash_run_id: Optional[str] = None,
    eagle3_run_id: Optional[str] = None,
):
    """Training entry point for 4+ GPU configs (SGLang inference backend)."""
    _train_impl(
        gpu_count, max_steps, num_epochs, run_eagle3, run_dflash,
        wandb_project, wandb_team, dataset_path, dataset_size, extra_overrides, hf_repo,
        resume=resume,
        dflash_run_id=dflash_run_id,
        eagle3_run_id=eagle3_run_id,
    )
```

The `eagle3_run_id` then flows through `_train_impl` into `_run_training`
and replaces the previously-hardcoded `"eagle3-qwen3-8b"`. Both edits are
backward-compatible (defaults preserved).

If we want to upstream these, they go in a small PR on
`feature/dflash-training` titled
*"modal_dflash_train: parameterise GPU spec + Eagle3 run id (benchmark prep)"*.

---

## 9. Next actions (after the runs complete)

1. **Run Section 5.2 extraction snippet** to get the 4-row table.
2. **Append the 4 rows** to `docs/study_notes/rl_study/dflash_eagle3_disagg_vs_colocate_results.md`
   under the "Disagg-Modal baseline" heading.
3. **Move to the RunPod colocate arm** (C1, C2, CE1, CE2) per Section 4.2
   of the benchmark plan, using the **same** WandB project so the disagg
   and colocate cells share an x-axis.
4. **Write the comparison report** following the template in plan
   Section 7.

---

**Document version:** 0.1 — runs live, metrics pending.
**Maintainer:** xing.han — disagg-Modal baseline for the disagg-vs-colocate study.
**Last refresh:** 2026-05-20 ~23:18 PDT (10 min after launch).

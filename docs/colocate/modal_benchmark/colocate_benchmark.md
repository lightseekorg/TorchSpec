# Colocate Benchmark — DFlash + Eagle3 (colocate arm)

> **Status: CE1 + C1 done.** This is the colocate arm of the
> disagg-vs-colocate study. **CE1** (Eagle3 2+2 colocate) and **C1**
> (DFlash 2+2 colocate) both completed matched 20000-step / 40k-sample
> runs (rc=0). C1 first hit two stacked colocate hangs — both
> root-caused & fixed (#1 init, GPU-verified; #2 step-0 CUDA-IPC
> handshake, pinned by a faulthandler dump) — then ran clean. See the
> **Re-analysis** + **§C1** sections for what the data does and does not
> establish. Headline (vs the same-SGLang `main`-branch disagg rerun,
> [`…rerun_on_main.md`](./dflash_eagle3_disagg_modal_rerun_on_main.md)):
> colocate's real gain is **≈2× less GPU-h for Eagle3, ≈1.5× for
> DFlash** — entirely from reclaiming the idle disagg inference GPUs.
> CE1's old 6.6× was a trainer-impl confound, now **proven** by the
> rerun (disagg Eagle3 jumped ×3.4 on FA4 + post-norm alone). CE2 / C2
> remain unrun.
> **Baseline to match (disagg arm):** [`dflash_eagle3_disagg_modal_results.md`](./dflash_eagle3_disagg_modal_results.md)
> — that doc's **§8** is the immutable test contract; **§9** is its
> (to-be-falsified) predictions.
> **Branch:** TorchSpec `feature/colocate-training-inference`.
> **Transport evidence:** [`transport_benchmark.md`](../../../../TorchSpec/docs/colocate/transport_benchmark.md),
> [`transport_optimization.md`](../../../../TorchSpec/docs/colocate/transport_optimization.md).

---

## Re-analysis — what the CE1 data actually shows (2026-05-21)

> Based on CE1's matched 20000-step / 40k-sample run (plus an earlier
> 5000-step run and an uncapped soak for stability), the disagg raw
> metrics in [`modal_logs/all_runs.json`](./modal_logs/all_runs.json),
> and `scripts/extract_modal_perf.py` (warm window = step ≥ 100, median —
> the *same* method applied to both arms). This section **supersedes**
> the pre-run predictions in the "Disagg vs Colocate" section below.
> (The `dflash_eagle3_disagg_vs_colocate_benchmark_plan.md` was not
> relied on.)

### The measurement

| Quantity | Disagg E1 | Colocate CE1 | Ratio |
|---|--:|--:|--:|
| total GPUs | 4 (2 infer + 2 train) | 2 (shared via MPS) | 0.5× |
| training ranks (dp_size) | 2 | 2 | 1× |
| samples seen | 40,000 | 40,000 | matched |
| optimizer-step time (warm median) | 2.125 s | 0.151 s | 14.1× |
| └ of which trainer GPU compute | 2.108 s | ≤ 0.151 s¹ | — |
| global batch (accum) | 8 (accum 4) | 2 (accum 1) | — |
| **samples/s** | **3.76** | **~13.25** | **3.5×** |
| samples/s per *total* GPU | 0.94 | 6.63 | 7.0× |
| per-sample trainer compute | 0.264 s | ≤ 0.076 s | ~3.5× |

¹ the `[colocate_loop]` log line carries no compute/data split — 0.151 s
is the whole loop iteration, so trainer compute is *at most* 0.151 s.
Disagg's `compute=2.108s` is CUDA-event measured (`TIMING` line); disagg
E1 is 99% compute-bound (data 0.025 s, dispatch 0.023 s).

### Decomposition — where the 3.5× comes from

Per **total GPU** the gap is **7.0×**. It decomposes cleanly into two
independent factors:

- **×2.0 — colocate folds inference onto the training GPUs.** Disagg
  dedicates 2 of its 4 GPUs to inference; colocate runs inference on the
  *same* 2 GPUs as training via MPS. **This is the one genuinely
  *architectural* colocate saving in the data.**
- **×3.5 — the trainer itself is faster on the colocate branch.** A
  per-sample *trainer GPU-compute* gap (0.264 s → ≤ 0.076 s). The
  trainer does the same draft-model math regardless of where inference
  runs — so this is **not** a colocate-vs-disaggregated effect.

### Why the 3.5× is a confound, not a verdict

The two arms differ in **six** ways at once — the 3.5× is the net of all
of them:

1. **Branch / trainer code** — disagg `feature/dflash-training @
   cb741ae`; colocate `feature/colocate-training-inference @ b82d64b`.
   These are **divergent sibling branches** (merge-base `7fa10c22`;
   cb741ae = +123 commits, colocate HEAD = +52). The colocate branch's
   Eagle3 trainer + draft-model attention got substantial development
   `cb741ae` never received — see "Trainer-implementation gap" below.
2. **torch version** — colocate pulled latest via `uv`; disagg pinned at
   image-build time. Both land on ~torch 2.9.x — **not** a material
   confound.
3. **gradient accumulation** — the colocate loop **rejects `accum > 1`**
   (`NotImplementedError ... Multi-step accumulation is parked`), so its
   global batch is 2 vs disagg's 8.
4. **platform** — Modal H100 vs RunPod H100 SXM.
5. **transport** — Mooncake RDMA-over-TCP vs NCCL P2P + CUDA IPC.
6. **GPU count** — 4 vs 2.

Only #6 (and folding inference in) is the colocate *architecture*.
**#1 — divergent trainer code — is the dominant cause of the ×3.5
trainer-speed gap** and has nothing to do with colocate vs
disaggregated. **As run, this CE1-vs-old-E1 benchmark cannot answer "is
colocate mode faster than disaggregated mode."** The earlier pre-run
prediction (colocate ≈ parity, +0–5%) assumed *identical trainer code* —
that assumption is false here, which is why the measured gap is so large.

> **Update (2026-05-21) — confound #1 now *proven* and quantified.**
> The disagg arm was re-run on `origin/main @ 068f253` with the colocate
> arm's SGLang
> ([`dflash_eagle3_disagg_modal_rerun_on_main.md`](./dflash_eagle3_disagg_modal_rerun_on_main.md)):
> disagg Eagle3 jumped **3.76 → 12.72 samples/s (×3.4)** from the
> `main`-branch FA4 (#96) + post-norm (#97) work *alone* — almost
> exactly the hypothesized ×3.3. Against that fast, same-SGLang baseline
> the Eagle3 colocate win is **~2.1×**, not 6.6×; the DFlash C1/D1 pair
> (disagg branch-stable at ~10 samples/s) lands at **~1.5×**. See §C1
> "Disagg D1 vs colocate C1" for the resolved comparison.

### Trainer-implementation gap — verified (corrects an earlier claim)

> **Correction.** Doc v0.4–v0.5 attributed the trainer-speed gap to
> "FSDP1 → FSDP2". **That is wrong — both arms use FSDP2.** Verified
> with `git` on the TorchSpec repo: `cb741ae:torchspec/training/fsdp.py`
> and the colocate branch's both use `fully_shard` / `apply_fsdp2` /
> `fsdp2_load_full_state_dict`, and both `eagle3_trainer.py` log
> "Eagle3 model initialized with FSDP2".

What the trainer-speed gap actually is — from a branch comparison:

| Aspect | Disagg `cb741ae` | Colocate `b82d64b` |
|---|---|---|
| FSDP | FSDP2 (`fully_shard`) | FSDP2 (`fully_shard`) — **same** |
| torch | ~2.9.x | ~2.9.x — **same** |
| `eagle3_trainer.py` commits since merge-base `7fa10c22` | **1** | **8** |
| FA4 block-sparse attn + cutlass-dsl 4.4.2 (PR #65) | **absent** | **present** |
| USP sequence-parallel attn for Eagle3 (PR #93) | **absent** | **present** |
| draft-model block-sparse path (`models/draft/llama3_eagle.py`) | older | reworked by #65 |

Both runs used `attention_backend: flex_attention` (neither used the
explicit `fa4` backend), but on H100 / SM90 that path goes through the
block-sparse attention kernels in `llama3_eagle.py` — **and that code
was substantially reworked on the colocate branch (PR #65) and is not
in `cb741ae`**. Attention is a large fraction of the Eagle3 7-forward
TTT compute, so a reworked block-sparse kernel is the **most plausible
single cause** of the ~3.4× per-sample compute gap — but **this is a
hypothesis, not a proven attribution**: offline `git` inspection rules
*out* FSDP and torch and points *at* the attention/trainer development,
but only the same-branch controlled run can prove and quantify it.

**For future benchmarks:**
1. **Pin and log exact versions for both arms** — TorchSpec commit,
   sglang commit, *and* the resolved `torch` / `flashinfer` /
   `sgl-kernel` versions. The disagg arm pinned well (its doc §8.1);
   the colocate arm used `uv`-latest without recording the resolved
   versions — fix that.
2. **Run both arms from the same commit.** A cross-branch A/B is
   uninterpretable: 123 + 52 divergent commits dwarf the mode change.
3. **Record `attention_backend` and the resolved attention kernel** —
   `flex_attention` dispatches to different kernels across branches and
   SM versions; that dispatch, not the config string, is what matters.

### What IS validly established

1. ✅ **Colocate runs real Qwen3-8B Eagle3 training end-to-end** — the
   matched 20000-step run plus an earlier 5000-step run and an uncapped
   soak (~25000 steps observed in total), zero NaN / OOM / hang.
   Previously only tiny Qwen3-0.6B and CI configs had been validated
   (per the transport docs); the production 8B Eagle3 path under MPS was
   unproven.
2. ✅ **2+2 colocate = 2 physical GPUs** — the hardware-halving is real
   and measured.
3. ✅ **Step time is stable** — warm median 0.151 s over the matched
   20000-step run, early-third → late-third drift **−0.7%**; consistent
   with the earlier runs; no degradation.
4. ✅ **Convergence matched** — at the same 40,000 samples seen and with
   the WSD LR annealed to 0 at that mark on both arms, CE1's final loss
   ≈ 2.09 (rolling mean of the last ~1000 steps) vs disagg E1's
   2.24 / 1.98. Colocate does genuine, equivalent Eagle3 training per
   sample — the speed is a real wall-clock win, not skipped work.
5. ✅ **peak_alloc** oscillates 16–34 GB with variable seq_len, no
   upward trend over the 20000 steps — no leak.

### What is NOT established

- ❌ Whether colocate *mode* beats disaggregated *mode* at equal trainer
  code — the original study question.
- ❌ The §8 cell-for-cell contract — `accum` cannot be 4 under colocate.
- ❌ Eagle3 τ (acceptance length) for colocate — not on the log line.

### The controlled experiment that would answer the question

Run **both modes on the same branch + commit**
(`feature/colocate-training-inference`), same torch, same host class,
differing *only* in the colocate knobs:

- **Arm A** — `colocate_strategy=mps, transfer_mode=nccl` (colocate).
- **Arm B** — the disaggregated path on the *same* branch
  (`colocate_strategy` unset → `transfer_mode=mooncake`, separate
  infer/train GPUs).

Same `accum`, same global batch, same dataset + seed. Then the delta is
purely architectural. Until that runs, read CE1's 3.5× as *"the colocate
branch trains Eagle3 fast"*, **not** a mode comparison.

---

## Disagg vs Colocate — Comparison & Verdict (pre-run predictions — superseded by the Re-analysis above)

> ⚠️ **These were predictions made before any colocate run.** The
> headline prediction below — "expect rough parity; colocate unlikely to
> win on throughput" — was **falsified**: CE1 measured ~3.5× disagg E1's
> samples/s (a branch-confounded result — see the Re-analysis). This
> section is kept only as a record of prior reasoning; do **not** cite
> §1–§4 as findings. §5 "Tests to run" is still actionable.

### Bottom line up front

**You cannot conclude colocate is better or worse yet — the comparison
hasn't been run.** The disagg doc contains 4 full, matched 5000-step
Qwen3-8B training runs. The two colocate docs contain **transport
micro-benchmarks and CI smoke tests** — *none* of them is the colocate
arm (C1/C2/CE1/CE2) the disagg doc §8 was written to be compared
against. So there is no apples-to-apples number to compare.

What the colocate docs *do* let you predict: colocate is **very unlikely
to win meaningfully on per-step throughput**, and the disagg doc's §9
prediction of "+20–40%" for DFlash is over-optimistic and internally
inconsistent. The real place colocate can win is **GPU-hours / cost**,
and that is exactly what the unrun C1–CE2 cells must measure.

### 1. What each document actually measures

| | Disagg doc | transport_benchmark.md | transport_optimization.md |
|---|---|---|---|
| What ran | 4× **full 5000-step training** (D1/D2/E1/E2) | Transport mechanism in **isolation** (gloo vs CUDA IPC) + CI smoke tests | Kernel investigation + ipc-pipe A/B + 3000-step soak |
| Model / config | Qwen3-8B, real DFlash `anchors=512` / Eagle3 7-forward | 0.25 MB–256 MB synthetic payloads; CI configs (phase6/7, `colocate_tiny`) | Qwen3-0.6B tiny + "25.8 GB-class" CI |
| Headline metric | **samples/s, step time, loss/acc/τ @ 5000** | transfer latency (ms), test pass/fail | engine `send()` stall (ms), leak check |
| Matches §8 contract? | **Yes — it _is_ the contract** | No | No |

The disagg arm produced the four numbers it explicitly says must be
matched (§2.2):

| Cell | Disagg samples/s | Step (s) | Loss @5000 | τ @5000 |
|---|--:|--:|--:|--:|
| D1 DFlash 2+2 | 10.14 | 0.789 | 3.67 | — |
| D2 DFlash 4+4 | 19.07 | 0.839 | 3.14 | — |
| E1 Eagle3 2+2 | 3.76 | 2.125 | 2.24 | 1.66 |
| E2 Eagle3 4+4 | 7.92 | 2.021 | 1.80 | 2.00 |

The colocate side has **no equivalent row** — only `~0.18 s` step times
from CI tests on *different* workloads.

### 2. The step-time trap — do not compare these naively

The colocate CI tests show `~0.177–0.18 s/step` (`test_phase7_convergence`,
`test_phase6_peak_alloc_flatness`); disagg DFlash shows `0.789 s/step`.
**That is not "colocate is 4× faster."**

- Disagg DFlash D1 **compute alone** is `fwd 0.316 + bwd 0.418 + opt
  0.023 = 0.757 s`. Compute is workload-bound — colocate cannot shrink
  it.
- The `~0.18 s` colocate tests are a **lighter trainer config** (loss
  `12.13 → 3.27` in 50 steps ≠ DFlash's `6.64 → 3.67` over 5000). They
  are not the `anchors=512` DFlash or 7-forward Eagle3 trainer at all.

Those configs measure colocate *stability and correctness*, not
*throughput parity*. Treat them as a green stability light, nothing
more.

### 3. What the transport docs *do* let you predict

The transport docs settle one thing decisively: **transport is not a
step-time factor for colocate.** CUDA IPC moves the 160 MB Eagle3
payload in `~1.9 ms` — `~1 %` of a colocate step.

But here is the key point the disagg doc itself half-misses: **disagg's
transport is also already hidden.** §2.1 states the Mooncake fetch
(`data 0.42 s`) is *fully overlapped inside compute* (`0.76 s`) via
`prefetch_depth=8`. The unhidden transport cost in a disagg DFlash step
is only `step − compute = 0.789 − 0.757 ≈ 0.03 s`.

So:

- **The disagg §9 prediction is wrong.** It claims DFlash colocate gains
  "+20–40%" because "transfer is ~50% of step." That treats `data`
  (0.42 s) as removable critical-path time — but §2.1 says it is
  overlapped. The most colocate can recover is the `~0.03 s` unhidden
  slice → **~+4%, not +20–40%.** This contradicts §2.1 and both
  transport docs ("transport is not a colocate step-time bottleneck").
  §9's DFlash rows should be revised down.
- **Eagle3 is honestly forecast.** Both Eagle3 runs are ~100%
  compute-bound; §9's "≤ +6–7%" is realistic. Colocate cannot speed up
  the 7-forward TTT compute.

**Conclusion on throughput: expect rough parity (±~5–7%), not a colocate
win.** At a fixed layout (`dp_size` pinned by §8), colocate cannot
out-throughput a trainer-bound disagg run — the training compute is the
wall for both.

### 4. Is colocate better?

**On per-step throughput: almost certainly not meaningfully — expect a
tie.** Both arms hide transport; compute dominates.

**On GPU-hours / cost: this is the only place colocate can win — and it
can win big.** The disagg §11 launch plan ("Launch C1, CE1 on a *4-GPU*
pod"; "C2, CE2 on an *8-GPU* pod") implies the colocate cells run the
same logical 2+2 / 4+4 layout on **half the GPUs** (2 and 4) via MPS
sharing. If true:

- At parity step-time + half the GPUs → colocate is **~2× cheaper**
  (disagg D1 = 4.88 GPU-h; a 2-GPU colocate C1 at similar wall-clock
  ≈ 2.4 GPU-h).
- The **break-even is "colocate step < 2× disagg step."** Colocate can
  be up to ~2× slower per step and still tie disagg on cost.

**The decisive unknown** — only C1–CE2 can answer it — is how much the
colocate execution model inflates the step:

- transport_benchmark describes the colocate handoff as a **serial**
  stall (engine produces → transfer → trainer trains). If the loop is
  truly serial, `generate()` is *added* to the critical path (in disagg
  it runs on separate GPUs, hidden) → colocate step > disagg step.
- If engine/train overlap under MPS, then MPS SM-sharing slows training
  compute instead.
- Either way colocate per-step ≥ disagg per-step; the question is by how
  much, and whether GPU-halving covers it.

### 5. Tests to run for the colocate arm

The disagg doc **§8 is the contract** — run exactly **C1, C2, CE1, CE2**
with §8.1–8.3 knobs immutable and only the §8.4 colocate lines changed.
Beyond that:

1. **Run the 4 matched cells** — they don't exist yet. C1/C2/CE1/CE2 are
   also the *first* run of the real `anchors=512` DFlash and 7-forward
   Eagle3 trainer configs under colocate (the CI tests use different
   configs; `test_grad_parity` covers only gradient equality, not a
   5000-step run).
2. **Report cost-normalized metrics, not just samples/s.** Add
   **samples/s per GPU** and **GPU-hours to 5000 steps** to the results
   table. Raw samples/s will look like a colocate loss; per-GPU /
   per-dollar is where colocate's case lives.
3. **Confirm and record the colocate GPU count per cell** (2 for 2+2?
   4 for 4+4?). This is the entire value proposition — make it explicit.
4. **`peak_alloc` drift < 0.1%** over steps 1000–5000 (§9 colocate-only
   gate; parse `[colocate_loop] step=… peak_alloc=…`).
5. **Convergence parity** — loss/acc/τ within ±2% of disagg §3 at
   matched steps, especially Eagle3 `acc_len` (1.66 / 2.00 @5000).
6. **`ipc-pipe` flag A/B (optional, low priority).** §8.4 pins
   `transfer_mode=cuda_ipc`, but `TORCHSPEC_COLOCATE_IPC_PIPELINE` is a
   *separate opt-in flag, default off*. Run the baseline with it **off**.
7. **Follow-up: `train_frac` / `infer_frac` sweep (not the baseline).**
   The disagg I/T ratios show inference is *massively* overprovisioned
   (DFlash I/T ≈ 5–6, Eagle3 ≈ 13–16). The §8.4 `0.45 / 0.45` split
   likely over-feeds idle inference and starves training. Run the
   contract `0.45/0.45` first, then sweep `train_frac` up as a separate
   study.
8. **Note the platform confound.** Disagg ran on Modal
   (Mooncake-over-TCP); colocate runs on RunPod. §8.1 pins the software,
   but Modal-vs-RunPod hardware/network is itself a variable — flag it.

---

## Colocate run tracker

Colocate uses MPS GPU-sharing, so a logical N+N layout runs on **N
physical GPUs** (each GPU hosts one trainer rank + one engine rank) —
confirmed in `examples/colocate-qwen3-8b-1node/run.sh`. This halves the
GPU count vs the disagg arm.

| Cell | Run ID | Model | Layout | GPUs | Status | samples/s | step (s) | loss (40k smpl) | τ |
|---|---|---|---|--:|---|--:|--:|--:|--:|
| **CE1** | `CE1-eagle3-2plus2-colocate` | Eagle3 | 2 infer + 2 train | 2 | ✅ 20000 steps = 40k samples (2026-05-21) | ~13.25 | 0.151 | ~2.09¹ | n/c² |
| **C1** | `C1-dflash-2plus2-colocate` | DFlash | 2 infer + 2 train | 2 | ✅ 20000 steps = 40k samples, rc=0 (2026-05-22) — see §C1 | 7.51 | 0.266 | ~3.81⁵ | n/c² |
| **CE2** | `CE2-eagle3-4plus4-colocate` | Eagle3 | 4 infer + 4 train | 4 | pending | — | — | — | — |
| **C2** | `C2-dflash-4plus4-colocate` | DFlash | 4 infer + 4 train | 4 | pending | — | — | — | — |

Comparison rows (filled per cell as runs complete):

Disagg baselines below use the **`main`-branch rerun**
([`dflash_eagle3_disagg_modal_rerun_on_main.md`](./dflash_eagle3_disagg_modal_rerun_on_main.md))
— same SGLang as the colocate arm, so the comparison is no longer
cross-branch-confounded.

| Cell | Disagg samples/s | Colocate samples/s | Δ raw | Disagg GPU-h | Colocate GPU-h | Cost Δ |
|---|--:|--:|--:|--:|--:|--:|
| Eagle3 2+2 (E1 / CE1) | 12.72³ | ~13.25 | ≈ even | 3.49 | **1.68**⁴ | **~2.1× less GPU-h**³ |
| DFlash 2+2 (D1 / C1) | 10.00⁶ | 7.51 | **−25%** | 4.44⁷ | **2.96**⁷ | **1.50× less GPU-h** |
| Eagle3 4+4 (E2 / CE2) | 7.92 | — | — | 24.00 | — | — |
| DFlash 4+4 (D2 / C2) | 19.07 | — | — | 10.53 | — | — |

¹ rolling mean of CE1's last ~1000 steps (batch-of-2 per-step loss is
very noisy). Matches disagg E1 at the same 40k samples (E1: avg_loss
2.24, JSON final-mean 1.98). ² τ (acc_len) is not emitted on the
`[colocate_loop]` log line — not captured. ³ **corrected baseline.**
Disagg Eagle3 re-run on `main @ 068f253` (same SGLang as colocate) =
**12.72 samples/s**, vs **3.76** on the old `cb741ae` branch — a ×3.4
jump from FA4 (#96) + post-norm (#97) *alone*. This **proves** the
earlier "6.6× less GPU-h" headline was a trainer-impl confound: against
the correct fast baseline the Eagle3 colocate win is **~2.1×**, i.e.
the pure ×2.0 half-the-GPUs architecture effect (colocate raw
throughput ≈ disagg here). ⁴ matched 40k-sample point, warm rate:
CE1 colocate = 40000 / 13.25 on 2 GPUs = 1.68 GPU-h; disagg E1-rerun =
40000 / 12.72 on 4 GPUs = 3.49 GPU-h.
⁵ window-mean of C1's last 2000 steps (per-step batch-of-2 loss is
noisy; raw range 3–6). ⁶ disagg DFlash re-run on `main @ 068f253` =
**10.00 samples/s**, statistically identical to **10.14** on old
`cb741ae` (±1.4 %) — DFlash disagg throughput is **branch-stable**, so
C1/D1 is *not* confounded. ⁷ **matched 40k-sample point**, warm rate:
disagg D1 = 40000 / 10.00 on 4 GPUs = 4.44 GPU-h; colocate C1 =
40000 / 7.51 on 2 GPUs = 2.96 GPU-h. On *actual* training wall the
ratio is 1.66× (D1 4467 s = 4.96 GPU-h; C1 5384 s loop-wall =
2.99 GPU-h).

---

## CE1 — Eagle3 2+2 colocate (2026-05-21) — first colocate result

**Status: ✅ matched 20000-step run completed** (`rc=0`, no NaN, no OOM)
— 20000 steps × global-batch 2 = **40,000 samples**, the same data
exposure as disagg E1 (5000 × 8), with the WSD LR annealed to 0 at that
mark for a fair endpoint. An earlier 5000-step run (10k samples) and an
uncapped soak are folded into the stability evidence below.

**Setup:** 2×H100 80GB SXM RunPod pod, branch
`feature/colocate-training-inference @ b82d64b`, sglang `94f03a39` +
colocate patch, `transfer_mode=nccl` (CUDA IPC default sub-transport).
The `uv`-based launcher built the whole environment — clone + deps +
sglang editable build + Qwen3-8B + perfectblend download — in **~100 s**
(vs the first attempt's `pip` path, which had not finished setup after
~50 min). `uv` + backgrounding the model/dataset downloads is the win.

### Result — matched 20000-step run (40k samples; warm window step ≥ 100, 3981 pts)

| Metric | CE1 colocate | Disagg E1 | Note |
|---|--:|--:|---|
| samples seen | **40,000** (20000 × gb 2) | **40,000** (5000 × gb 8) | matched data exposure |
| GPUs | 2 (MPS-shared) | 4 (2 infer + 2 train) | colocate folds inference in |
| step_time median | 0.151 s | 2.125 s | per-optimizer-step; not the same work/step |
| step_time p10 / p90 / p99 | 0.133 / 0.222 / 0.358 s | 2.07 / 2.31 / 2.69 s | colocate step tight + stable |
| step_time drift (early→late third) | **−0.7%** over 20000 steps | — | flat; no degradation |
| **samples/s** | **~13.25** | 3.76 | **3.5×** — throughput comparator |
| **wall clock (40k samples)** | **~57 min** (loop 55.8 min) | **3 h 09 m** (11341 s) | **3.3× faster** |
| **GPU-hours (40k samples)** | **~1.9** (2 GPU) | **12.60** (4 GPU) | **6.6× less** |
| peak_alloc (warm) | 16.0–34.3 GB | n/a | oscillates with seq_len; no leak trend |
| final loss (rolling, @40k samples) | **~2.09** | 2.24 / 1.98 | **convergence matched** |

> **⚠ Superseded — this table compares CE1 to the *slow-branch* disagg
> E1 (3.76 samples/s, `cb741ae`).** The disagg arm was later re-run on
> `main` with the colocate arm's SGLang
> ([`…rerun_on_main.md`](./dflash_eagle3_disagg_modal_rerun_on_main.md)):
> Eagle3 disagg is **12.72 samples/s**, not 3.76. The "3.5× / 3.3× /
> 6.6×" figures below are therefore confound-inflated. **The corrected
> Eagle3 colocate advantage is ~2.1× less GPU-h** (1.68 vs 3.49 GPU-h /
> 40k samples) — see §C1 "Disagg D1 vs colocate C1". The CE1 numbers in
> the *colocate* column (13.25 samples/s, 0.151 s, ~2.09 loss) are
> unaffected and remain valid.

### Reading the numbers

**Throughput & cost.** CE1 sustains ~13.25 samples/s vs disagg E1's
3.76 — a raw **3.5×**. For the *same 40,000 samples*: CE1 finished in
~57 min on 2 GPUs (~1.9 GPU-h); disagg E1 took 3 h 09 m on 4 GPUs
(12.60 GPU-h) → **3.3× faster wall-clock, 6.6× less GPU-hours**. The
per-micro-iteration compute (0.151 s vs 0.531 s = 2.125 / 4, each the
fwd+bwd of one sample/rank through the Eagle3 7-forward TTT path) shows
the same ~3.5×, so it is not a batch-count artifact.

**Convergence is matched.** The 20000-step run was sized so CE1 sees the
*same* 40,000 samples as disagg E1, with the WSD LR annealed to 0 at
that mark — a fair endpoint. CE1's final loss (rolling mean of the last
~1000 steps — the batch-of-2 per-step loss is very noisy) is ≈ **2.09**,
vs disagg E1's **2.24 / 1.98**. Equal data, equal LR phase, equal
convergence: colocate is doing genuine, equivalent Eagle3 training — the
speed is a real wall-clock win, not skipped work.

**This is NOT yet a clean colocate-vs-disagg verdict.** The ~3.5× is
largely a **branch / trainer-implementation difference**, not the
colocate vs disaggregated architecture:
- The two arms are on divergent sibling branches; the colocate branch's
  Eagle3 trainer + block-sparse attention got development `cb741ae`
  lacks (PR #65 FA4 block-sparse, #93 USP). **Both use FSDP2 and
  ~torch 2.9.x** — see the Re-analysis "Trainer-implementation gap" for
  the verified breakdown (an earlier "FSDP1→FSDP2" claim was wrong).
- Different sglang pin.
- `accum` could not be matched — the colocate loop **rejects
  `draft_accumulation_steps > 1`** (`NotImplementedError: colocate loop
  currently requires draft_accumulation_steps=1 ... Multi-step
  accumulation is parked`), so global batch is 2 vs disagg's 8.

A clean §8-contract comparison needs both arms on the same trainer
code. Until then, read the 3.5× as *"the colocate branch trains Eagle3
~3.5× faster than the disagg branch"*, **not** *"colocate mode beats
disaggregated mode"*.

### Limitations found
1. **No gradient accumulation** — the colocate loop is `accum=1` only;
   multi-step accumulation is "parked". This breaks cell-for-cell
   parity with the disagg §8 contract (global batch 2 vs 8).
2. **τ (acc_len) not captured** — the `[colocate_loop]` log line carries
   step_time / loss / lr / peak_alloc but not Eagle3 acceptance length;
   comparing τ vs disagg needs wandb or a trainer-side log.
3. **peak_alloc** swings 16–34 GB with variable seq_len (expected, not
   a leak — step-time drift was −0.7% over the full 20000 steps); a
   strict flatness gate needs the fixed-workload phase-6 test.

### Outcome of the first attempt (for the record)
An earlier 2×H100 pod (`pptwzmrl4e777f`) was provisioned and stopped by
the user during env setup — no steps ran. Spend ≈ $6.36. The re-run
below used the `uv`-accelerated launcher.

### Launch recipe (ready to re-run)

Colocate cannot run on Modal (gVisor blocks NVIDIA MPS); it needs a
RunPod / Vast.ai host with `--ipc=host`. Recipe worked out for CE1:

1. **Provision** a 2×H100 SXM pod (`runpodctl pod create --gpu-id
   "NVIDIA H100 80GB HBM3" --gpu-count 2 --template-id runpod-torch-v240
   --container-disk-in-gb 200 --ports 22/tcp --terminate-after +3h`).
2. **Setup:** `git clone -b feature/colocate-training-inference`, then
   `bash scripts/colocate/run_smoke_host.sh --setup-only` (sglang
   clone + patch + build, pip install torchspec + deps), then the MPS
   probe `python -m tests.colocate._mps_probe`.
3. **Dataset:** `python scripts/tools/prepare_perfectblend.py --output
   data/perfectblend_200k.jsonl --sample-size 200000 --seed 42`
   (matches the disagg 200k subsample).
4. **Train:** `CUDA_VISIBLE_DEVICES=0,1
   ./examples/colocate-qwen3-8b-1node/run.sh configs/colocate_qwen3_8b.yaml`
   with overrides: `dataset.train_data_path=data/perfectblend_200k.jsonl
   dataset.eval_data_path=null training.num_train_steps=5000
   training.draft_accumulation_steps=1 training.micro_batch_size=1
   training.fsdp_strategy=FULL_SHARD training.fsdp_reduce_dtype=bfloat16
   training.prefetch_depth=8 training.max_checkpoints=1
   training.save_interval=9999 training.seed=42
   training.train_frac=0.45 training.infer_frac=0.45`.
   **Note:** `draft_accumulation_steps` must be `1` — the colocate loop
   raises `NotImplementedError` for any value > 1.

**Resolved discrepancy — `transfer_mode`:** the disagg doc §8.4 lists
`transfer_mode=cuda_ipc`, but `cuda_ipc` is **not a valid
`transfer_mode`** value (`validate_colocate_config` only accepts
`(mps, nccl)`). The colocate path uses `transfer_mode=nccl`; CUDA IPC
vs gloo is a *sub-transport* selected by the `TORCHSPEC_COLOCATE_IPC`
env var (CUDA IPC is the round-9 default). So the matched setting is
`transfer_mode=nccl` + default IPC — the §8.4 line should be corrected.

---

## C1 — DFlash 2+2 colocate (2026-05-22) — ✅ completed, 20000 steps, rc=0

**Status: DONE.** Two distinct, sequential hangs were root-caused &
fixed; the production run then completed cleanly. Hang #1
(`_init_target_lm_head`) — bare collectives on the union PG; fixed in
`dflash_trainer.py` and GPU-verified. Hang #2 (step-0 hidden-state
transfer) — a **CUDA-IPC handshake deadlock from a 3-vs-2 tensor-count
mismatch**; pinned by a `PYTHONFAULTHANDLER` stack dump and fixed in
`colocate_loop.py`. With both fixes, **C1 ran 20000 steps to completion
(rc=0, 40000 samples, zero hang / NaN / OOM)** on 2026-05-22.

### Result — matched 20000-step run (40k samples)

2×H100 80GB HBM3 SXM RunPod pod, the *same* colocate setup as CE1 (`uv`
launcher; `colocate_strategy=mps`, `transfer_mode=nccl`,
`train_frac/infer_frac=0.45`, `accum=1`), base config
`sglang_qwen3_8b_dflash.yaml` (DFlash draft `dflash_draft_config.json`,
`dflash_num_anchors=512`, `dflash_block_size=16`, 5 aux layers
`[1,9,17,25,33]`). Global batch = 2 (dp_size 2 × micro 1 × accum 1).

| Metric | C1 (DFlash 2+2 colocate) |
|---|--:|
| Steps / samples | 20000 / 40000 |
| Training-loop wall | 5384 s (1 h 29 m 44 s) |
| Warm step-time (step ≥ 1000) | **0.266 s** mean / 0.262 s median |
| Warm throughput | **7.51 samples/s** (2 GPUs, global batch 2) |
| Per-step compute (fwd+bwd) | ~180 ms (fwd ~75 ms + bwd ~108 ms) |
| Per-step non-compute overhead | ~85 ms (engine-forward wait + IPC transfer + loop) |
| Loss (window mean) | 6.19 (step 0–2k) → **3.81** (step 18–20k) |
| Peak GPU alloc | ~30 GB / 80 GB |
| GPU-h (loop wall, 2 GPUs) | **2.99 GPU-h** |

Loss converged cleanly and near-monotonically by 2k-step window: 6.19 → 5.07
→ 4.56 → 4.40 → 4.27 → 4.21 → 4.06 → 3.96 → 4.01 → 3.81. The first
~1000 steps ran slow (~0.39 s/step, compile/warmup) then locked to a
flat ~0.262–0.270 s for the remaining 19000.

### Disagg D1 vs colocate C1 — comparison

Baseline: the **disagg rerun on `main`**
([`dflash_eagle3_disagg_modal_rerun_on_main.md`](./dflash_eagle3_disagg_modal_rerun_on_main.md),
2026-05-21) — D1 re-run on `origin/main @ 068f253` with **the same
SGLang** (`94f03a39` + `v0.5.10.post1`) that C1 colocate used. This
retires the SGLang confound and lets the two arms be compared directly.

| | Disagg D1 (rerun-on-main) | Colocate C1 | Ratio |
|---|--:|--:|--:|
| Physical GPUs | 4 (2 infer + 2 train) | **2** (MPS-shared) | ½ |
| Step time / global batch | 800 ms / gb 8 | 266 ms / gb 2 | — |
| Per-sample compute | ~97 ms | ~90 ms | ≈ |
| Raw throughput (samples/s) | 10.00 | 7.51 | colocate **0.75×** |
| GPU-h for 40k samples (warm rate) | 4.44 | **2.96** | colocate **1.50× less** |
| GPU-h for 40k samples (actual wall) | 4.96 | 2.99 | colocate **1.66× less** |

The GPU-h win decomposes exactly: `2.0` (half the GPU count) × `0.75`
(colocate's lower raw throughput) = **1.50× less GPU-h**. Colocate
trades ~25 % raw throughput — the cost of two roles MPS-sharing each
GPU — for halving the GPU count. (On *actual* training wall-clock the
edge is 1.66×: D1's 4467 s run carried more init/checkpoint overhead
than C1's 5384 s; the warm-rate row isolates steady state.)

**The DFlash disagg number is branch-stable — so this comparison is
solid, not confounded.** The rerun's headline finding: D1 disagg is
**10.00 samples/s** on `main @ 068f253` vs **10.14** on the old
`feature/dflash-training @ cb741ae` — identical within ±1.4 %. FA4 (#96)
and post-norm (#97) did not move DFlash disagg throughput. So the
cross-branch caveat that earlier versions of this doc attached to C1/D1
is **empirically negligible for DFlash** — the ≈1.5× GPU-h win stands.

**The same rerun *proves* CE1's 6.6× for Eagle3 was a confound, not a
colocate benefit.** It shows Eagle3 disagg E1 jumping **3.76 → 12.72
samples/s (×3.4)** purely from the `main`-branch FA4 + post-norm work —
exactly the "×3.3 trainer-impl confound" hypothesized in the
Re-analysis. Re-comparing CE1 colocate Eagle3 (13.25) against the
*correct* fast disagg baseline (E1-rerun 12.72) collapses the Eagle3
colocate win:

| Eagle3 2+2 | colocate CE1 | disagg (E1-rerun, fast) | disagg (old E1, slow) |
|---|--:|--:|--:|
| samples/s | 13.25 | 12.72 | 3.76 |
| GPU-h / 40k | 1.68 | 3.49 | 11.82 |
| colocate GPU-h win | — | **~2.1×** | 6.6× (confounded) |

**Unified conclusion.** Against same-SGLang, current-code disagg
baselines, colocate's real advantage is **≈2× less GPU-h for Eagle3,
≈1.5× for DFlash** — and it comes entirely from *reclaiming the idle
disagg inference GPUs* (disagg runs inference-saturated, I/T ≈ 8–9×),
not from any trainer speedup. DFlash's win is the smaller of the two
because its heavier trainer leaves less GPU headroom for the colocated
engine → more MPS contention → a steeper (0.75× vs ~1.0×) raw-throughput
penalty.

**Convergence.** C1 colocate final loss **3.81** (40k samples) sits in
the same band as both disagg D1 runs — original D1 3.67, D1-rerun 4.89.
The rerun flags a **DFlash loss regression on `main`** (3.67 → 4.89,
attributed to FA4/post-norm); C1 colocate does *not* show it. Loss is
too noisy and the runs too differently configured (gb 2 vs 8, 20k vs 5k
steps, disagg's `avg_loss` is a last-step rolling mean) for a finer
claim — acceptance-length τ would be the better convergence metric, but
the colocate loop does not log it.

This C1 result is the colocate baseline to compare against the upcoming
**Modal** DFlash test. Per-step metrics (step-time, loss, lr — 4004
rows, every 5 steps) are archived at
[`logs/c1_dflash_colocate_metrics.tsv`](./logs/c1_dflash_colocate_metrics.tsv).

### How the two hangs were found and fixed

*(Debug history — kept for the record. The successful run above used the
same setup once both fixes below were in place.)*

**Symptom (first attempts):** init completed on both sides (both `SglEngine` ranks +
both DFlash `TrainerActor` ranks — DFlash draft 1.05 B trainable,
`TargetLMHead`, FSDP2). Then the log froze right after
`dflash_trainer.py:220 [Rank 0] TargetLMHead loaded`, GPU **0 %**, no
`[colocate_loop] step=` ever — a silent ~13-min deadlock, no traceback.

### Root cause (found by offline `git` analysis — no debug pod needed)

`DFlashTrainer._init_target_lm_head` calls **bare** collectives —
`dist.barrier()` and `dist.broadcast(param.data, src=0)` with **no
`group=`**. In colocate mode the *default* process group is the **union
NCCL world** (trainer ranks `[0,N)` + engine ranks `[N,2N)`). Only
trainer ranks execute `_init_target_lm_head`; the engine ranks are in
sglang and never reach it → the barrier waits for all `2N` ranks, only
`N` arrive → **deadlock**, exactly at the observed freeze point (the log
line immediately precedes `dist.barrier()`).

**This is *not* the transport** (the v0.8 hypothesis was wrong).
`Eagle3Trainer._init_target_lm_head` already carries this exact fix and
even documents it: *"Without the explicit group they default to the
union-world PG in colocate mode, and the engine never enters this code
path, so the trainer hangs."* The DFlash trainer — comment-labelled
"same as Eagle3Trainer" — never received it. CE1 (Eagle3) ran fine on
the identical setup precisely because Eagle3 scopes these collectives to
`get_gloo_group()` (the trainer-only group).

**Five bare collectives** in `dflash_trainer.py` default to the union
PG and hang in colocate: `dist.barrier()` + `dist.broadcast()` in
`_init_target_lm_head` (the C1 hang), and three `dist.all_reduce()` in
the per-position metric reduction (would hang at step 1).

### Fix applied (`torchspec/training/dflash_trainer.py`)

Scoped all five collectives to `get_gloo_group()` — already imported,
already used correctly elsewhere in the same file (`init_model`). Mirrors
`eagle3_trainer.py` exactly. **Safe for disagg too**: there
`get_gloo_group()` *is* the whole trainer PG, so the change is a no-op
outside colocate (which is why disagg D1 was unaffected). **GPU-verified
2026-05-21** — the patched re-run (C1-v2) reached "TargetLMHead
initialized and synced", the colocate data-fetcher init, and the
`Colocate Training` loop — i.e. it cleared hang #1.

### Hang #2 — CUDA-IPC handshake deadlock at step 0 (pinned & fixed)

With hang #1 fixed, the re-run reached the training loop and the
`Colocate Training: 0/20000` bar, then **deadlocked at step 0** — log
frozen, GPU **0 %** on both, no `[colocate_loop] step=`, no error.

**Debug rounds.** Three instrumented re-runs were needed; the first two
gave a *wrong* intermediate conclusion that the third overturned:

- **C1-v3/v4** added `[HANG2]` phase markers to `colocate_loop.py` and
  `[HANG2-DEBUG]` prints to the connector/fetcher. The loop markers
  showed it blocking at `ray.get(engine_refs)`; the transfer markers
  appeared not to print, which was read as *"the hang is before the
  transfer, inside the engine `generate()`"* and *"transport ruled
  out."* **That conclusion was wrong** — the markers simply weren't
  captured before the freeze, and `ray.get(engine_refs)` blocks
  whenever the engine's *send* (deep inside `generate()`) blocks.
- **C1-v5** ran with `PYTHONFAULTHANDLER=1` and `kill -ABRT <pid>` on
  the hung processes to dump every thread's Python stack — no ptrace
  needed (py-spy was blocked by the container). This **pinned it
  exactly.**

**Root cause — a 3-vs-2 tensor-count mismatch in the CUDA-IPC
handshake.** The faulthandler dump put the engine at:

```
cuda_ipc.py:250  ipc_send  (blocked in dist.recv — waiting for an ack)
  ← nccl_hidden_states_connector.py:242  NcclHiddenStatesConnector.send
  ← colocate.patch  _send_hidden_states_to_nccl
```

and the `[HANG2-DEBUG]` payload prints showed the mismatch:

| side | tensors declared / sent | count |
|---|---|--:|
| **engine** `connector.send` | `hidden_states (388,20480)`, `input_ids (388,)`, `last_hidden_states (388,4096)` | **3** |
| **trainer** `recv_step` specs | `hidden_states (388,20480)`, `input_ids (388,)` | **2** |

CUDA-IPC transfer is a per-tensor handshake: `ipc_send` ships one IPC
handle per tensor (walking `sorted(keys)`) and **blocks on `dist.recv`
for one ack per tensor**; the trainer's `recv_step` walks the same
`sorted(keys)`, maps each handle, and sends one ack each. The engine
sent 3 handles and waited for 3 acks; the trainer declared only 2 specs,
mapped 2, acked 2 → the engine's **3rd `dist.recv` blocked forever**.

**Why the trainer declared only 2.** `colocate_loop._build_tensor_specs`
gated `last_hidden_states` behind `store_last_hidden_states`, and
DFlash's config (`sglang_qwen3_8b_dflash.yaml`) sets
`store_last_hidden_states: false` → the spec was omitted. **But the
colocate engine always sends `last_hidden_states`**: `sgl_engine.py`
sets `enable_return_hidden_states=True` *unconditionally*, so
`logits_output.last_hidden_states` is always populated, and the sglang
`colocate.patch`'s `_send_hidden_states_to_nccl` ships it whenever it is
non-`None` — it does **not** consult `store_last_hidden_states` (that
flag only gates the disagg Mooncake metadata path, `_get_tensor_shapes`).

**Why CE1 (Eagle3) never hit it.** Eagle3's config has
`store_last_hidden_states: true`, so its `_build_tensor_specs` already
declared all 3 tensors → trainer and engine agreed. The bug is specific
to draft configs with `store_last_hidden_states: false` (DFlash).

### Fix applied (`torchspec/controller/colocate_loop.py`)

`_build_tensor_specs` now **always declares `last_hidden_states`** — the
`if store_last_hidden_states:` gate (and the now-unused parameter) were
removed. The trainer must declare every tensor the engine sends or the
IPC handshake deadlocks; the colocate engine always sends all 3. Draft
trainers that don't consume `last_hidden_states` (DFlash reads only
`input_ids` + `hidden_states`) simply ignore the extra dict key — the
cost is one unused `(seq_len, 4096)` bf16 buffer per step (~3 MB),
negligible. Both hang fixes verified — the C1 production run then
completed all 20000 steps clean (rc=0); see the **Result** section above.

### Secondary finding — colocate loop should fail-fast

`colocate_loop.py` guards `accum>1` and `per_dp_rank>1` with
`NotImplementedError`, but nothing catches a union-vs-trainer PG mismatch
or a transfer-spec mismatch — both can only deadlock silently. A
follow-up guard and/or a first-step watchdog would have turned both
hangs into immediate, legible errors instead of multi-pod debug rounds.

### Lesson for future benchmarks — the spec is the engine's, not a config flag

The trainer-side `tensor_specs` is a **contract that must mirror what
the engine actually sends**, not what a training-side config says it
*should* want. `store_last_hidden_states` is a training-side preference;
the wire payload is decided by the engine + sglang patch. Any future
draft variant must derive its `_build_tensor_specs` from the engine's
real output set (or, better, have the engine announce its key set on the
metadata channel) — never from a local flag.

**C1 spend:** ≈ $16 across 5 debug pods (v1–v5) + ≈ $11 for the v6
production pod (2×H100, ~100 min) — all torn down.

---

**Document version:** 1.5 — **re-based the disagg comparison onto the
`main`-branch rerun** ([`dflash_eagle3_disagg_modal_rerun_on_main.md`](./dflash_eagle3_disagg_modal_rerun_on_main.md))
— D1 + E1 re-run on `origin/main @ 068f253` with the **same SGLang**
(`94f03a39` + `v0.5.10.post1`) as the colocate arm, retiring the
cross-branch confound. Findings: (a) DFlash disagg is **branch-stable**
(10.00 vs 10.14 samples/s, ±1.4 %) → **C1/D1 is solid: colocate ≈1.50×
less GPU-h** (2.96 vs 4.44 / 40k samples; 1.66× on actual wall). (b) The
rerun **proves** CE1's 6.6× was a trainer-impl confound — disagg Eagle3
jumped 3.76 → 12.72 samples/s (×3.4) from FA4 + post-norm alone; against
that correct baseline the Eagle3 colocate win is **~2.1×**, not 6.6×.
Unified: colocate's real edge is **≈2× (Eagle3) / ≈1.5× (DFlash) less
GPU-h**, from reclaiming idle disagg inference GPUs — not a trainer
speedup. v1.4 — **C1 DFlash colocate run COMPLETE: 20000
steps, rc=0, 40k samples, zero hang/NaN/OOM.** Warm step-time 0.266 s →
7.51 samples/s on 2 GPUs; loss 6.19 → 3.81. v1.3 — **C1 DFlash: both colocate hangs root-caused
& fixed; production run launched.** Hang #1: `DFlashTrainer.
_init_target_lm_head` ran `dist.barrier()`/`broadcast()` (+3
`all_reduce`) on the union PG → only trainer ranks reach it → deadlock;
**fixed** in `dflash_trainer.py` (5 collectives → `group=get_gloo_group()`,
mirroring Eagle3) and **GPU-verified**. Hang #2: a `PYTHONFAULTHANDLER`
stack dump (C1-v5, `kill -ABRT`) pinned a **CUDA-IPC handshake deadlock**
— the engine sends 3 tensors (`hidden_states`, `input_ids`,
`last_hidden_states`) but the trainer's `_build_tensor_specs` declared
only 2 (it gated `last_hidden_states` on `store_last_hidden_states`,
which DFlash sets `false`); the per-tensor IPC ack handshake left the
engine's 3rd `dist.recv` blocked forever. The colocate engine *always*
sends `last_hidden_states` (`enable_return_hidden_states=True` is
unconditional). **Fixed** in `colocate_loop.py` — `_build_tensor_specs`
now always declares `last_hidden_states`. (This corrects the v1.2 claim
that hang #2 was "before the transfer / inside engine `generate()`,
transport ruled out" — the v3/v4 marker reads were inconclusive; v5's
faulthandler dump showed the hang *is* in the transport's `ipc_send`.)
v0.7: refreshed the Re-analysis
to CE1's matched
20000-step / 40k-sample numbers; **verified both arms use FSDP2 and
~torch 2.9.x** (the v0.4–v0.5 "FSDP1→FSDP2" attribution was wrong). The
×3.3 trainer-speed gap is divergent branch code — most plausibly the
reworked block-sparse attention (PR #65) absent from disagg `cb741ae` —
see "Trainer-implementation gap". Headline: CE1 = 6.6× less GPU-h than
disagg E1 at matched 40k samples = ×2.0 colocate-architecture × ×3.3
trainer-impl confound; a same-branch controlled run is needed to isolate
the pure mode effect.
**Maintainer:** xing.han — disagg-vs-colocate benchmark, RL infra study.

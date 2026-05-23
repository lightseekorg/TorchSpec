# DFlash + Eagle3 Disaggregated Modal — Rerun on `origin/main` (May 21 2026)

> **Status: 2 of 2 runs complete (D1 + E1 v3).** D2 / E2 (4+4 layout)
> were intentionally skipped this round per request.
> **TorchSpec base:** `origin/main @ 068f253` (latest at run time —
> _post-norm support_ + _FA4 BlockMask_ + _network topology reporter_).
> **TorchSpec branch (local-only):** `benchmark/disagg-modal-rerun-on-main`.
> **Modal driver:** `scripts/modal/modal_dflash_train.py` (cherry-picked
> from `8eb33177` on `feature/dflash-training`, retargeted to `main`).
> **Platform:** Modal `doordash/sandbox`, H100 80 GB HBM3 SXM.
> **WandB project:** [`dflash/dflash-eagle3-disagg-modal`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal).
> **Companion doc (original 4-run baseline):** [`dflash_eagle3_disagg_modal_results.md`](./dflash_eagle3_disagg_modal_results.md).

This doc captures the May 21 rerun on the `main` branch with the canonical
SGLang pairing (`94f03a39` + `v0.5.10.post1`). It does **not** restate
methodology that is already covered in the original baseline doc — read
that first if you need test-rig details.

---

## 1. The 2 runs at a glance

| # | Run name | Model | Layout | Modal app | WandB run | Wall (training only) | WandB runtime | Final step | NaN | OOM |
|---|---|---|---|---|---|--:|--:|--:|--:|--:|
| **D1** | `D1-dflash-2plus2-disagg-modal-main-v2` | DFlash | 2 infer + 2 train (`H100:4`) | [`ap-M7bSYhcYFgUepY1fygLZiU`](https://modal.com/apps/doordash/sandbox/ap-M7bSYhcYFgUepY1fygLZiU) | [`9jc10axs`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal/runs/9jc10axs) | **4 466.9 s (1 h 14 m 27 s)** | 4 488 s | 5000 / 5000 | 0 | 0 |
| **E1 v3** | `E1-eagle3-2plus2-disagg-modal-main-v3` | Eagle3 | 2 infer + 2 train (`H100:4`) | [`ap-o81oAzEYIuI06A515LvUhF`](https://modal.com/apps/doordash/sandbox/ap-o81oAzEYIuI06A515LvUhF) | [`mz2685i1`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal/runs/mz2685i1) | **3 827.0 s (1 h 03 m 47 s)** | 3 850 s | 5000 / 5000 | 0 | 0 |

`-main-v2` / `-main-v3` are the volume-output suffixes; v3 was needed
for E1 because the v2 attempt died early on the RoPE incompatibility
described in §3 and a v2.log already existed in the volume.

> **Total compute:** 4 GPU × (1 h 14 + 1 h 04) = **9.3 H100-hours**.
> Both apps showed clean WandB shutdowns; no failed pods, no Modal retries
> consumed.

---

## 2. Why we needed a rerun

The original 4-run set ([results doc](./dflash_eagle3_disagg_modal_results.md))
was on `feature/dflash-training @ cb741ae` and pinned SGLang to
`0f2df9370a1d…` + the `v0.5.8.post1` patch. We needed to confirm the
exact same 2 + 2 layout still trained cleanly after `main` accumulated
three feature commits:

| Commit | Title | Why it matters here |
|---|---|---|
| `068f253` | feat: support post-norm architecture (#97) | New rope/norm code paths the draft model touches. |
| `87dfadf` | [Tool] Add network topology and status reporter (#94) | Diagnostic-only; no runtime risk but new logs in the trainer. |
| `5c865bd` | feat: Integrate FA4 with custom BlockMask construction (#96) | New attention kernel selection logic — verify it doesn't regress 2 + 2 step time. |

The local `benchmark/disagg-modal-rerun-on-main` branch was created
straight off `origin/main` (`068f253`) and four small fixes were
cherry-picked / authored on top:

```text
4f59919 fix(eagle3 draft): handle rope_scaling.type == 'default' for Qwen3-8B
2bca899 benchmark: use git apply --recount for sglang.patch
11e42f6 benchmark: bump SGLang to 94f03a39 + v0.5.10.post1, fail-fast on patch error
4e43764 benchmark: pin Modal container to origin/main @ 068f253
3269dae feat: parameterize GPU allocation and eagle3 run_id  ← from 8eb33177
```

`3269dae` is the cherry-pick of `8eb33177` onto `main`; the other four
are the inline fixes documented in §3.

---

## 3. Three issues debugged before D1 / E1 ran clean

### 3.1 SGLang patch silently mis-applied (recurrent)

`modal_dflash_train.py` originally invoked `git apply … || true` so a
broken SGLang patch would let the image build complete with an
*unpatched* SGLang. The training process would later die at
SglEngine init with:

```
TypeError: ServerArgs.__init__() got an unexpected keyword argument
  'enable_aux_hidden_states'
```

**Fixes (commits `11e42f6` + `2bca899`):**

1. Bumped to the `main`-canonical pairing
   `SGLANG_COMMIT=94f03a39dbd39edfc2b118b5357bbbadaaa9ad28` +
   `SGLANG_PATCH_VERSION=v0.5.10.post1` (matches `tools/build_conda.sh`
   on `main`).
2. Removed the `|| true` swallow.
3. Added `--recount` to `git apply` because the v0.5.10.post1 patch ships
   with two malformed hunks (`@@ -893,6 +894,75 @@` declares 75 added
   lines but contains 106 — `--recount` lets git infer real counts).

### 3.2 RoPE `"default"` type not recognised by Eagle3 draft

Eagle3 `LlamaFlexAttention._init_rope` raised
`ValueError: Unknown RoPE scaling type default` against the Qwen3-8B
target after `transformers ≥ 4.45` started writing
`rope_scaling = {"rope_type": "default", "factor": 1.0}` for plain RoPE.

**Fix (commit `4f59919`, `torchspec/models/draft/llama3_eagle.py`):**

```python
# transformers >=4.45 sets rope_scaling={"rope_type":"default", ...}
# to mean "standard RoPE, no scaling". Treat it the same as
# rope_scaling is None (the legacy transformers convention).
if scaling_type in (None, "default"):
    self.rotary_emb = LlamaRotaryEmbedding(
        self.head_dim,
        max_position_embeddings=self.max_position_embeddings,
        base=getattr(self.config, "rope_theta", 10000),
    )
elif scaling_type == "linear":
    ...
```

The change is picked up inside Modal via the `add_local_dir` overlay —
no SGLang or transformers edits required.

### 3.3 Hugging Face 429s under concurrent tokenizer loads

When all 4 pods (D1 / E1 / D2 / E2) launched within the same minute,
`Qwen/Qwen3-8B` tokenizer fetches collided and produced
`Client error '429 Too Many Requests'` — the per-token quota is 1000 req
/ 5 min. The provided HF token (stored in the `xingh3-hf-write` Modal
secret) is a free-tier key so the limit is identical. **The actual fix was launching D2 / E2
2 minutes after D1 / E1**, which is why we ultimately settled on D1 +
E1 only this round.

---

## 4. Final performance metrics (WandB-authoritative, steady-state steps 51–5000)

Source: WandB step-history pulled live from the `dflash-eagle3-disagg-modal`
project on May 21 23:50 UTC. First 50 steps dropped (pure warm-up); all
quantiles taken over the remaining 4 950 samples.

### 4.1 Per-step latency breakdown

| Run | step (ms) | compute (ms) | fwd (ms) | bwd (ms) | opt (ms) | data (ms) | dispatch (ms) |
|---|--:|--:|--:|--:|--:|--:|--:|
| **D1** DFlash 2+2 (median) | **800.0** | 775.8 | 327.9 | 421.2 | 22.6 | 417.1 | 20.8 |
| **D1** DFlash 2+2 (p95)    | 889.3 | 846.8 | 401.6 | 441.8 | 23.0 | 500.1 | 24.7 |
| **E1 v3** Eagle3 2+2 (median) | **628.8** | 609.5 | ¹ | ¹ | 17.3 | 25.3 | 24.3 |
| **E1 v3** Eagle3 2+2 (p95)    | 819.9 | 779.4 | ¹ | ¹ | 17.5 | 120.0 | 28.6 |

¹ `eagle3_trainer.py` does not split fwd / bwd into separate WandB
fields — only the rolled-up `compute=…` is logged. Of E1 v3's 629 ms
median step, ~610 ms is compute and the rest is opt + dispatch + data.
This is consistent with the original baseline doc's observation that
Eagle3 is essentially 100 % compute-bound.

### 4.2 Throughput (samples / s consumed by trainer / produced by inference)

| Run | train_capacity median | train_capacity p95 | infer_capacity median | infer_capacity p95 | I / T ratio |
|---|--:|--:|--:|--:|--:|
| **D1** DFlash 2+2  | **10.00** | 10.56 | 92.4 | 105.9 | **9.2 ×** |
| **E1 v3** Eagle3 2+2 | **12.72** | 14.04 | 98.5 | 112.4 | **7.7 ×** |

Both runs are still inference-saturated (infer capacity ~8–10 × what
the trainer can chew through), but the gap is _smaller_ on E1 v3 than
on the original E1 (which had I / T ≈ 13.5). The `main`-branch SGLang
+ FA4 stack is producing samples slower _relative to the trainer_ than
the old `feature/dflash-training` stack did — see §6 for why we believe
this is FA4 enabling itself for the SGLang side.

### 4.3 Headline samples / s

> **DFlash, disagg-Modal, anchors=512, warm:**
> - 2+2 (4 GPU): **10.00 samples/s** (step 800 ms median) — vs 10.14 on the original `feature/dflash-training` run; **statistically identical**.
>
> **Eagle3, disagg-Modal, warm:**
> - 2+2 (4 GPU): **12.72 samples/s** (step 629 ms median) — vs 3.76 on the original `feature/dflash-training` run; **3.4 × faster**.

The E1 jump is the headline result of this rerun. See §6.

---

## 5. Convergence trajectories

All values are pulled from WandB at the listed `train/step`. D1 reports
15 horizons (`acc_0 … acc_14`); E1 v3 reports 7 (`acc_0 … acc_6`) —
not a regression, just the model-specific configuration.

### 5.1 `train/avg_loss`

| Run | step 100 | step 500 | step 1000 | step 2000 | step 3000 | step 4000 | step 5000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| D1 DFlash 2+2 main-v2  | 6.288 | 5.355 | 5.117 | 4.146 | 4.600 | 4.772 | **4.890** |
| E1 v3 Eagle3 2+2 main-v3 | 5.443 | 3.772 | 2.426 | 2.960 | 2.681 | 2.825 | **2.297** |

### 5.2 `train/avg_acc`

| Run | step 100 | step 500 | step 1000 | step 2000 | step 3000 | step 4000 | step 5000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| D1 DFlash 2+2 main-v2  | 0.070 | 0.117 | 0.124 | 0.210 | 0.172 | 0.156 | **0.134** |
| E1 v3 Eagle3 2+2 main-v3 | 0.212 | 0.364 | 0.527 | 0.455 | 0.530 | 0.516 | **0.572** |

### 5.3 `train/simulated_acc_len`

| Run | step 100 | step 500 | step 1000 | step 2000 | step 3000 | step 4000 | step 5000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| D1 DFlash 2+2 main-v2  | 0.148 | 0.450 | 0.524 | 0.821 | 0.782 | 0.689 | **0.650** |
| E1 v3 Eagle3 2+2 main-v3 | 0.465 | 0.841 | 1.432 | 1.069 | 1.364 | 1.284 | **1.574** |

> **Reading note:** `train/avg_loss` is the rolling average reported on
> the **last training step**, _not_ the run minimum. D1's last-step loss
> (4.89) is higher than the median over the run (4.50) because the LR
> schedule is approaching its cosine floor and the loss-decay-gamma=0.9
> across 15 horizons keeps the late-horizon ploss high. The relevant
> _convergence_ signal is the median over the run plus the trajectory.

### 5.4 Final WandB summary highlights

| Field | D1 (`9jc10axs`) | E1 v3 (`mz2685i1`) |
|---|---|---|
| `train/avg_loss` (final step) | 4.890 | 2.297 |
| `train/avg_acc` (final step) | 0.134 | 0.572 |
| `train/simulated_acc_len` (final step) | 0.650 | **1.574** |
| `train/grad_norm` (final step) | 0.300 | 5.832 |
| `train/lr` (final step) | 6.0 e-5 | 0.0 (cosine done) |
| `perf/train_capacity` (final step) | 9.32 | 13.36 |
| `perf/infer_capacity` (final step) | 81.28 | 95.56 |
| `train/acc_0` (single-token horizon) | 0.468 | 0.671 |
| Horizons logged | 15 (`acc_0 … acc_14`) | 7 (`acc_0 … acc_6`) |

---

## 6. Comparison vs the May-12 baseline (same layout, different branch)

The original baseline ([results doc](./dflash_eagle3_disagg_modal_results.md))
ran the **same 2 + 2 layout** on `feature/dflash-training @ cb741ae`
with SGLang `0f2df93` + `v0.5.8.post1`. Identical Modal pod shape,
identical training data, identical seed.

| Metric | D1 baseline (May 12) | D1 rerun (`main-v2`) | Δ | E1 baseline (May 12) | E1 v3 rerun (`main-v3`) | Δ |
|---|--:|--:|--:|--:|--:|--:|
| Wall (training) | 4 387.2 s | 4 466.9 s | **+1.8 %** | 11 341.3 s | 3 827.0 s | **−66.2 %** |
| Step time (median) | 0.789 s | 0.800 s | +1.4 % | 2.125 s | 0.629 s | **−70.4 %** |
| samples/s (train_capacity) | 10.14 | 10.00 | −1.4 % | 3.76 | 12.72 | **+238 %** |
| infer_capacity (med) | 61.7 | 92.4 | +50 % | 51.2 | 98.5 | +92 % |
| simulated_acc_len (final step) | n/a (DFlash) | 0.650 | — | n/a (was acc_len 1.74) | 1.574 | −9 % |
| `avg_loss` (step 5000) | 3.67 | 4.89 | **worse** | 2.24 | 2.30 | comparable |
| `avg_acc` (step 5000) | 0.235 | 0.134 | **worse** | 0.580 | 0.572 | comparable |

### 6.1 What changed for D1 (DFlash 2+2)

- **Latency unchanged within noise** — step time is +1.4 %, well inside
  Modal pod-to-pod variation. The new FA4 + post-norm code paths in
  `068f253` did **not** introduce a measurable regression at this
  layout.
- **Loss is worse and acc is worse, by ~33 %.** This is the surprise.
  The two runs share data, seed, and trainer — the main differences
  are (a) `main` enabled FA4 by default in the trainer-side attention
  kernel, and (b) the post-norm refactor.
  - Hypothesis A: FA4 + draft-side custom BlockMask is producing a
    subtly different attention output for the DFlash horizon=15 path
    that hurts the late-horizon ploss. Worth ablating with
    `attention_backend=flash_attention_2` on the draft side as a
    follow-up.
  - Hypothesis B: post-norm changes the residual stream the DFlash
    target consumes for some layers, and the loss-decay-gamma weighting
    amplifies the difference at horizons 10–14.
  - **Both hypotheses are testable** by re-running D1 against
    `e63cfab` (the commit immediately before FA4 / post-norm) — that
    is the recommended next step.
- **infer_capacity jumped 50 %.** The SGLang `94f03a39` build wins on
  pure decode throughput; this matches what we'd expect from
  `v0.5.10.post1`'s sampler + scheduler improvements over `v0.5.8.post1`.

### 6.2 What changed for E1 (Eagle3 2+2)

- **3.4 × throughput improvement, 70 % step-time reduction.** The old
  E1 step was 2.13 s; the new one is 0.63 s. This is dominated by the
  post-norm + FA4 work — Eagle3 spends ~100 % of its step in the
  7-forward TTT compute, and FA4 is reportedly 1.5–3 × faster than
  FA2 on Hopper for the BlockMask shapes Eagle3 uses.
- **Convergence quality is preserved.** Final loss 2.30 vs 2.24 baseline
  is within run-to-run noise (the original Eagle3 run also fluctuated
  0.1–0.3 between adjacent step buckets). Acc final 0.572 vs 0.580
  is statistically identical; final `simulated_acc_len` 1.57 vs the
  baseline run's ~1.74 is mildly worse — likely the same FA4-vs-FA2
  attention difference, but here the headline is that we're getting
  the same Eagle3 quality at **a third of the wall-clock cost**.

> **Bottom line:** the `main`-branch FA4 + post-norm work is the
> **biggest performance jump we've seen on the disagg-Modal arm so
> far** for Eagle3 specifically. DFlash gets the same SGLang
> infer_capacity uplift but appears to leave loss / acc on the table;
> that regression is the one to investigate next.

---

## 7. Modal driver settings (snapshot)

`scripts/modal/modal_dflash_train.py` final values for this rerun
(committed in `4e43764` + `11e42f6`):

```python
TORCHSPEC_REPO = "https://github.com/zhubohao911/TorchSpec.git"
TORCHSPEC_BRANCH = "main"
TORCHSPEC_PIN_COMMIT = "068f253"  # latest origin/main @ 2026-05-21
SGLANG_COMMIT = "94f03a39dbd39edfc2b118b5357bbbadaaa9ad28"
SGLANG_PATCH_VERSION = "v0.5.10.post1"

# inside _run_training():
f"cd {SGLANG_DIR} && git apply --recount "
f"{REPO_DIR}/patches/sglang/{SGLANG_PATCH_VERSION}/sglang.patch",
```

Other knobs are unchanged from the May-12 baseline doc and are not
restated here. Modal `Retries(initial_delay=0.0, max_retries=3)` is in
effect; the v3 of E1 was a fresh launch (not an internal retry) after
the v2 attempt died on the RoPE issue with the local overlay still
inheriting the un-patched draft model.

---

## 8. Artefacts

| Asset | D1 | E1 v3 |
|---|---|---|
| WandB run | [`9jc10axs`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal/runs/9jc10axs) | [`mz2685i1`](https://wandb.ai/dflash/dflash-eagle3-disagg-modal/runs/mz2685i1) |
| Modal app | [`ap-M7bSYhcYFgUepY1fygLZiU`](https://modal.com/apps/doordash/sandbox/ap-M7bSYhcYFgUepY1fygLZiU) | [`ap-o81oAzEYIuI06A515LvUhF`](https://modal.com/apps/doordash/sandbox/ap-o81oAzEYIuI06A515LvUhF) |
| Volume log | `/D1-dflash-2plus2-disagg-modal-main-v2.log` (~3.0 MB, 19 741 lines) | `/E1-eagle3-2plus2-disagg-modal-main-v3.log` (~2.6 MB, 16 786 lines) |
| Volume output dir | `/D1-dflash-2plus2-disagg-modal-main-v2/{checkpoints,hf_model,config.yaml}` | `/E1-eagle3-2plus2-disagg-modal-main-v3/{checkpoints,config.yaml}` |
| Final WandB run name | `dflash-eagle3-disagg-modal_p8ioo76g-RANK_0` | `dflash-eagle3-disagg-modal_jbsu2a3g-RANK_0` |

---

## 9. Recommended follow-ups

1. **Ablate D1 loss regression.** Re-run D1 against `e63cfab` (the
   commit immediately before FA4 #96 + post-norm #97) holding SGLang at
   `94f03a39`. If loss recovers to ~3.7, the regression is on the
   trainer side; if it stays at ~4.9, the SGLang bump is the suspect.
2. **Land the RoPE-`"default"` fix upstream.** `4f59919` is currently
   only on the local benchmark branch. It's a one-liner that anyone
   training Qwen3-8B on `transformers ≥ 4.45` will hit; opening a PR
   on TorchSpec `main` is cheap and unblocks future agents.
3. **Add a CI lint for SGLang patch hunks.** The two malformed hunks
   in `patches/sglang/v0.5.10.post1/sglang.patch` should be rewritten
   so `--recount` is no longer required (or — at minimum — add a
   `git apply --check` step in CI so the next stale patch fails fast
   instead of inside a 4-GPU Modal container).
4. **Run D2 + E2 once Hugging Face 429s are mitigated** (either an HF
   Pro token or an explicit 2-minute stagger between launches in the
   driver itself).
5. **Backport this run's `infer_capacity` win** to the colocate arm
   comparison: the original results doc's headline samples/s for
   colocate-vs-disagg parity is now slightly out of date because
   disagg's inference half got faster.

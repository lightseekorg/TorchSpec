# Colocate (PR #92) — full detail & iteration history

> The complete, detailed write-up of the colocate PR: every phase,
> follow-up round, N>1 bug fix, architectural correction, and
> rented-GPU validation run. The **PR #92 description itself is kept
> concise** and links here for the depth.
>
> Source of truth for *what actually happened* is
> [`implementation_log.md`](implementation_log.md) (rounds 1–10); this
> doc is the PR-narrative companion, preserved so the concise PR body
> can drop the accumulated detail without losing it.

---

Tracking work on [#81](https://github.com/lightseekorg/TorchSpec/issues/81) — co-locate training and inference on the same GPUs via CUDA MPS + gloo (CPU-staged) hidden-state transfer.

Every phase is gated behind `colocate_strategy=mps` + `transfer_mode=nccl` so the disaggregated baseline keeps working throughout.

## Status

- [x] Phase 0 — config flags & validation
- [x] Phase 1 — placement: 1:1 bundle pairing + MPS env
- [x] Phase 2 — union NCCL world bootstrap
- [x] Phase 3 — P2P data plane (smoke test)
- [x] Phase 4 — sglang hidden-state hook
- [x] Phase 5 — controller / sync training loop — **DFlash-config tensor-spec contract amended round 12 (`last_hidden_states` always declared)**
- [x] Phase 6 — memory caps & stability — **`test_phase6_peak_alloc_flatness` PASSED (200 steps; 1000-step `--stability` GREEN, round 11)**
- [x] Phase 7 — numeric parity & convergence — **`test_phase7_convergence_loss_decreases` PASSED (50 steps), `test_phase7_grad_parity_smoke` PASSED, `test_convergence_disagg_overlap` GREEN (1000 steps vs Mooncake, round 11)**
- [x] Phase 8 — docs & example config
- [x] **Production-scale GPU validation** — Eagle3 (CE1, round 11) **and** DFlash (C1, round 12) each ran 20000 steps / 40k samples clean on 2×H100, rc=0; same-SGLang disagg baseline rerun on `main` shows **≈2.1× less GPU-h for Eagle3, ≈1.5× for DFlash**

## Test results — full suite GREEN on 4×H100

Independently verified twice (4×H100 SXM first, then a clean re-run on 4×H100 NVL — same outcome):

```
test_phase4_tiny_one_step                  PASSED  (1 step end-to-end on 1×GPU)
test_phase7_tiny_loss_decreases            PASSED  (loss 12.02 → 9.74 over 20 steps)
test_phase4_one_step_completes_end_to_end  PASSED  (1 step end-to-end on 4×GPU)
test_phase7_grad_parity_smoke              PASSED
test_phase6_peak_alloc_flatness            PASSED  (200 steps, peak-alloc flat)
test_phase7_convergence_loss_decreases     PASSED  (50 steps, loss decreases)
============== 6 passed in 734.59s ==============   pytest exit=0
```

The full colocate path is exercised end-to-end on the 4×H100 box: MPS daemon → 8-rank union world → patched sglang × 4 engines (engine-only `_WORLD`, union-default PG, `dp_attention` rank offset) → 4 concurrent engine→trainer gloo-staged hidden-state pairs → `NcclMultiTensorFetcher` × 4 → Eagle3 draft fwd/bwd across 4-trainer FSDP NCCL subgroup → optimizer step. Loss decreases monotonically, peak GPU alloc stays flat for 200 steps.

## Iteration chain — 4 N>1 bug fixes found by `--full`

Every `--full` run before run #7 hit a bug that the 1-GPU tiny smoke had no way to surface — code paths that only the `dp_size==1` case ever exercised had latent ≥2-rank bugs. The pattern was consistent enough that I audited the whole codebase for it after run #7 went green; one more bug of the same shape was found and fixed proactively.

| Run | Fix | What surfaced |
|---|---|---|
| #1-#2 | [`33b7e26`](https://github.com/lightseekorg/TorchSpec/pull/92/commits/33b7e26) | All 4 engines computed their union rank as `N + tp_rank` (always 0) → 8-rank rendezvous deadlock. `tp_rank` is rank *within* the engine's own size-1 TP group; the engine's union rank is `N + paired_trainer_rank`. |
| #3 | [`a5a0288`](https://github.com/lightseekorg/TorchSpec/pull/92/commits/a5a0288) | `fsdp_group` `new_group` interleaved between the two sglang-paired shared groups and the meta_group — bumped the trainer's per-process new_group counter by 1, so the meta_group's hashed name mismatched the engine's → all-world rendezvous deadlock. Reordered: all shared groups before role-restricted ones. |
| #4 | [`058871d`](https://github.com/lightseekorg/TorchSpec/pull/92/commits/058871d) | `dp_attention` rank-offset surgery shifted by `n_per_role` (=N) instead of the engine's own union rank → all engines computed their attn_tp group as `[N]`; only engine 0 passed the `GroupCoordinator` membership check. Offset by `N + paired_trainer_rank` instead. |
| #5-#6 | (no code change — pod was stopped mid-run twice; on restart the disk persists, so each relaunch just re-clones + re-runs) | — |
| #7 | [`bdc30ae`](https://github.com/lightseekorg/TorchSpec/pull/92/commits/bdc30ae) | All 4 trainers hung in `set_model_state_dict(broadcast_from_rank0=True)`. PyTorch's `_broadcast_state_dict` hard-codes `group=None`, so the broadcast landed on the 2N-rank union world; engines never enter this path. Temporarily install the trainer-only FSDP mesh group as the default PG (`_default_pg_override`) for the duration of the call. |
| audit | [`59400f1`](https://github.com/lightseekorg/TorchSpec/pull/92/commits/59400f1) | **Found preemptively, not from a failure:** the same shape as bdc30ae in `checkpoint.py` — 7 `dcp.save` / `dcp.load` calls with no `process_group=` argument would deadlock the same way at any dp_size in colocate. The green suite doesn't exercise this path (`save_steps==0` in every test config), but a real training run with periodic checkpointing would. Pass `process_group=actor.dp_group` to all 7 calls. |

## Key architectural corrections found during validation

- **NCCL cannot do same-GPU P2P.** A union-world NCCL communicator with two ranks on one physical GPU is hard-rejected (`ncclInvalidUsage`, "Duplicate GPU detected", and there is no env-var override) — exactly the colocate topology. **This invalidates issue #81's original "direct NCCL send/recv, same device" data-plane design.** The hidden-state plane was rerouted over the all-rank **gloo** `meta_group` with CPU staging (`aad72e2`), and a zero-copy **CUDA IPC** transport (engine exports a CUDA IPC handle, trainer maps it and does one on-device D→D copy) is **now the default** (`ea618a2`; `TORCHSPEC_COLOCATE_IPC=0` opts back to gloo) — a 1×H100 benchmark measured it **~170× faster** than gloo on realistic payloads (see round 7). So #81's zero-copy *intent* is met, just not via NCCL. The NCCL batched path is retained only for the separate-GPU Phase-3 dummy tests.
- **Unscoped `dist.*` collectives deadlock** on the 2N union default PG (trainer and engine run different code paths). All trainer-side collectives are now scoped to a trainer-only gloo group, FSDP broadcasts to the mesh group, and sglang's `_WORLD` is rebuilt as engine-only `[N, 2N)`.
- **External lib calls with hardcoded `group=None` are landmines** in colocate. Two confirmed (`set_model_state_dict`, `dcp.save`/`dcp.load`); both fixed by either swapping the default PG temporarily (`_default_pg_override`) or passing `process_group=` explicitly.
- **`transfer_mode=nccl` is genuinely Mooncake-free** — the top-level `mooncake.store` import was made lazy so the colocate path no longer needs libibverbs/libnuma.

## Environment constraint

The bundled `sgl_kernel` wheel ships **sm90+ kernels only** (no Ampere sm80/sm86, no Ada sm89). Real GPU testing is effectively limited to H100 / H200 / B200.

## Test cost

- Tiny smoke (`test_colocate_tiny.py`): ~3 min on 1×H100, ~$0.15 — pre-merge gate candidate.
- Full suite (`run_smoke_host.sh --full`): ~12 min on 4×H100, ~$2.5 — on-demand or label-gated.

## PR-review follow-ups — implemented & validated (2026-05-20)

A review of this PR against [#81](https://github.com/lightseekorg/TorchSpec/issues/81)'s
validation plan identified seven follow-ups. Rather than file them as
separate post-merge issues, **all seven were implemented on this branch**
and validated across rented-GPU sessions.

| # | Follow-up | Status |
|---|-----------|--------|
| P3 | Fold the `dp_attention` / `tp_worker` post-patch `sed` surgery into `colocate.patch` | ✅ `colocate.patch` self-contained (7 files); `apply_sglang_patch.sh --colocate` |
| P0 | Per-parameter grad parity + deterministic-seed plumbing | ✅ `test_grad_parity_determinism` + `test_grad_parity_full` (reframed — see below) + `torchspec/colocate/determinism.py` |
| P1 | Colocate checkpoint save/resume test | ✅ `test_colocate_checkpoint.py` — also fixed an **unreachable save path** (loop read a non-existent `save_steps`; now uses the real `save_interval`) |
| P1 | CUDA IPC zero-copy hidden-state plane | ✅ `torchspec/colocate/cuda_ipc.py` + `test_colocate_ipc.py`; **now the default transport** (opt out with `TORCHSPEC_COLOCATE_IPC=0`) — see round 7 |
| P2 | Multi-engine TP (`engine_tp_size > 1`) | ✅ rank math **and** data plane complete — colocate-loop per-engine dispatch, base paired-rank, `build_hidden_states_writer(tp_rank)`, per-request `_send_hidden_states_to_nccl` gate. No-op at tp=1. **Live `engine_tp_size=2` run validated on RunPod 2×H100 (2026-05-20).** |
| P2 | Multi-node colocate | 🟡 code complete (`ensure_mps_on_all_nodes`, `configs/colocate_qwen3_8b_2node.yaml`); a true 2-node run is a tracked follow-up (untested at scale, by agreed scope) |
| P2 | 1000-step nightly stability | ✅ `run_smoke_host.sh --stability` + `.github/workflows/colocate-stability.yml` (nightly cron + label-gated) |

### Follow-up round 2 (2026-05-20)

* **`grad_parity_full` reframed** — it was a colocate-vs-Mooncake-disagg comparison that skipped on every rental host (the disagg arm SIGSEGVs in Mooncake's Go runtime). It is now a **gloo-vs-CUDA-IPC transport parity** test: same seed, same everything except the hidden-state transport, assert per-parameter gradients match. Needs no Mooncake, runs anywhere the colocate path runs, no longer skips. **GPU-validated (RunPod 2×H100):** PASSED — "13 gradients match across gloo + CUDA IPC transports".
* **Multi-engine TP data plane** — completed (was rank-math-only): per-engine dispatch in the colocate loop, per-TP-rank connector `dst`, and the `_send_hidden_states_to_nccl` batch-index gate. **GPU-validated (RunPod 2×H100):** `test_colocate_engine_tp2_end_to_end` PASSED — 5 steps, loss 12.037 → 11.369. The first run surfaced a real bug: `initialize_model_parallel` rejected `engine_tp_size=2` because the colocate MoE-group guard only passed at tp=1 — fixed in `6e74ffc` (guard now rejects only real expert parallelism; `_MOE_EP` built as a per-rank singleton from `tp_world_ranks`).
* **Tracked follow-ups:** multi-node 2-node run; the same multi-TP changes ported to `v0.5.10.post1/colocate.patch`; the literal Mooncake-disagg grad parity.

### Follow-up round 3 (2026-05-20)

Three of the round-2 tracked follow-ups were picked up:

* **`v0.5.10.post1/colocate.patch` — forward-ported** (`af68196`). Regenerated onto sglang v0.5.10.post1; v0.5.10 restructured `initialize_model_parallel` (new `_ATTN_CP` / `_ATTN_TP` / MoE-DP groups), so `parallel_state.py` now uses a uniform engine-logical-world + offset-shift remap across all 8 group sites and the `dp_attention.py` hunk is dropped (v0.5.10 folded that group in). **GPU-tested (RunPod 1×H100):** `test_colocate_tiny.py` 2/2 with `SGLANG_PATCH_VERSION=v0.5.10.post1` at tp=1. **Still open:** the multi-TP `build_hidden_states_writer` changes are not yet ported into the v0.5.10 patch.
* **Multi-engine fan-out test** (`444903e`). `test_colocate_tp2` only covers a single tp=2 engine; added `configs/colocate_qwen0p6b_2eng_tp2_tiny.yaml` (2 engines × tp=2, dp_size=4, union world 2N=8 on 4 MPS GPUs) + `tests/colocate/test_colocate_multi_engine.py` exercising the colocate loop's `for e in range(n_engines)` dispatch. Wired into `--full`, self-skips below 4 GPUs. **GPU-validated in round 4 — see below.**
* **Mooncake-disagg crash diagnostic harness** (`a7d4436`). Restores `configs/disagg_qwen0p6b_tiny.yaml` and adds `scripts/colocate/diagnose_mooncake_crash.sh` — fingerprints the host and post-mortems the Mooncake SIGSEGV (Go traceback + dmesg + gdb) into a crash report. **Ran in round 4 — see below.**

### Follow-up round 4 (2026-05-20) — GPU validation on RunPod 4×H100

One 4×H100 pod ran both remaining round-3 GPU items.

* **Multi-engine fan-out — VALIDATED.** `test_colocate_multi_engine_tp2_end_to_end` **PASSED** (1 passed in 120.67s) — 2 engines × `engine_tp_size=2`, dp_size=4, union world 2N=8 across 4 MPS-shared H100s. The colocate loop's per-engine dispatch and per-engine base-paired-rank routing are confirmed correct at `n_engines > 1`. Also fixed a `run_smoke_host.sh` gap (`d6431d2`): `sgl_kernel` ≥ 0.3.x hard-fails to load without `libnuma.so.1`; setup now apt-installs `libnuma` + the RDMA verbs stack.
* **Mooncake-disagg crash — diagnosed; it is not a host problem.** `diagnose_mooncake_crash.sh` caught the `TrainerActor` SIGSEGV inside Go's `runtime.sigfwd` (signal-forwarding trampoline). That Go runtime is **`go1.25.9` bundled in `libetcd_wrapper.so`**, which `mooncake/engine.so` dlopens unconditionally; loading it into a process that already has PyTorch/CUDA collides the two sets of signal handlers. Mooncake's data transfers all **succeeded** before the crash. Host fingerprint is unremarkable (stock Ubuntu 22.04 Docker, glibc 2.35, default seccomp, `protocol=tcp`) — the conflict is **process-internal, not host-fixable**, so no host choice helps; this corrects the round-3 "container/seccomp" guess. `GODEBUG=asyncpreemptoff=1` was tried and does **not** fix it. Remaining avenues are version-pinning `mooncake-transfer-engine` (older Go toolchain) or import-order control — both process-internal. The reframed gloo-vs-CUDA-IPC `grad_parity_full` already covers per-parameter parity host-independently.

**Tracked follow-ups after round 4:** multi-node 2-node run; literal Mooncake-disagg grad parity (blocked on the third-party Go/CGO signal bug above, not on host availability).

### Follow-up round 5 (2026-05-21) — v0.5.10.post1 multi-TP + RoPE fix

Closes the round-4 follow-up "v0.5.10 patch multi-TP port".

* **`v0.5.10.post1/colocate.patch` regenerated** (`af68196`) from the current `v0.5.8.post1/colocate.patch`, so it now includes the `engine_tp_size>1` MoE-EP changes (`6e74ffc`). The forward-port's uniform offset-shift remap covers multi-TP group construction with no v0.5.10-specific extra work. **GPU-validated (RunPod 2×H100):** `test_colocate_tp2.py` (`engine_tp_size=2`) PASSED — 2 engine TP ranks, loss 12.04 → 11.37 over 5 steps. v0.5.10 is now validated at both tp=1 and `engine_tp_size=2`; `pp_size>1` and the Qwen3-8B 4×H100 `--full` matrix remain unexercised on v0.5.10.
* **RoPE `_init_rope` fix** (`be399a0`). `LlamaFlexAttention._init_rope` in the Eagle3 draft model handled `rope_scaling=None` and the named scaling types but raised `ValueError` on `rope_scaling={"rope_type": "default"}` — how transformers ≥4.x normalises "no scaling". This blocked every colocate test on a current-transformers environment; fixed by treating `"default"` as standard RoPE.

Note: `v0.5.10.post1/colocate.patch` is a *derived forward-port* of the v0.5.8 patch — the v0.5.8 patch remains the maintained source, so v0.5.10 must be re-derived whenever it changes. Once v0.5.10 passes full validation and nothing else pins v0.5.8, v0.5.10 should become the sole maintained patch.

### Follow-up round 6 (2026-05-21) — Mooncake-disagg crash FIXED

The round-4 Mooncake SIGSEGV is fixed. Inspecting the Go toolchain of each Mooncake wheel's `libetcd_wrapper.so` (`strings | grep go1.`):

| Mooncake version | Go toolchain |
|---|---|
| **0.3.10.post2** (was installed — crashes) | **go1.25.9** |
| 0.3.10.post1 | go1.24.13 |
| 0.3.10 / 0.3.9 / 0.3.8.post1 | go1.24.x |

`0.3.10.post2` is the **only** build using Go 1.25 — and `post1` is the *same Mooncake release* rebuilt, which isolates the regression to the **Go 1.25 toolchain**. `pyproject.toml` is pinned `mooncake-transfer-engine==0.3.10.post1` (`dfbb823`) — an exact pin, not a `>=` ceiling, since newer wheels will likely also ship on go1.25. The rationale is documented at both the pin (`pyproject.toml`) and the Mooncake load site (`torchspec/transfer/mooncake/store.py`) so a future dependency bump can't miss it (`327f2ef`).

**GPU-confirmed (RunPod 2×H100):** with `post1` (go1.24.13) the disagg path (`disagg_qwen0p6b_tiny.yaml`, 2 steps) **completes cleanly** — `Training: 100% 2/2`, loss 12.073 → 11.604, checkpoint saved, **no `runtime.sigfwd` SIGSEGV**. The same run on `post2` dies before step 1. This unblocks the literal vs-Mooncake-disagg grad-parity comparison; rebuilding that comparison test (removed in the `grad_parity_full` reframe) is the remaining piece.

### Follow-up round 7 (2026-05-21) — CUDA IPC made the default transport

The colocate hidden-state transport was flipped: **CUDA IPC is now the default**, gloo CPU-staging is the explicit opt-out.

* **The change** (`ea618a2`). `TORCHSPEC_COLOCATE_IPC` went opt-in (`=1`) → opt-out: unset selects CUDA IPC; `0`/`false`/`no`/`off` falls back to gloo. `cuda_ipc.ipc_requested()` → `ipc_enabled()` with the default inverted; `factory.py` / `train_group.py` skip `expandable_segments` by default (CUDA IPC needs plain `cudaMalloc` memory — only the gloo fallback injects it). 10 files; engine and trainer read the same env var so they always agree on the transport. `test_cuda_ipc.py` 13/13 on the dev box.
* **Benchmark** (`de5e930`). New `scripts/colocate/bench_transport.py` — two processes on one GPU, 2-rank gloo group, both transports timed across a payload sweep + a realistic Eagle3 multi-tensor case. **GPU-measured (RunPod 1×H100 80GB SXM):**

  | Payload | gloo | CUDA IPC | speedup |
  |---|--:|--:|--:|
  | 16 MB | 14.98 ms | 1.53 ms | 9.8× |
  | 64 MB | 154 ms | 0.77 ms | 200× |
  | 256 MB | 497 ms | 0.82 ms | 605× |
  | Eagle3 160 MB (realistic) | 319 ms | 1.9 ms | **171×** |

  gloo is capped at ~0.5 GB/s by its own TCP `dist.send`/`recv` ship; CUDA IPC is near-constant ~1 ms (the D→D copy is 0.26 ms for 256 MB, the rest a fixed `cudaIpcOpenMemHandle` + ack handshake). Full tables + per-stage breakdown in [`docs/colocate/transport_benchmark.md`](https://github.com/lightseekorg/TorchSpec/blob/feature/colocate-training-inference/docs/colocate/transport_benchmark.md).

**Outcome → see round 9:** the IPC-default `--full` run was attempted and **hung** — root-caused to the IPC capability probe and fixed (`e166c21`).

### Follow-up round 8 (2026-05-21) — v0.5.10.post1 full matrix + cutover

The full `run_smoke_host.sh --full` matrix was run against `v0.5.10.post1/colocate.patch` on a RunPod 4×H100 — **all 13 tests across 9 files pass** (`SGLANG_PATCH_VERSION=v0.5.10.post1`): tp=1, engine_tp_size=2, 4-engine Qwen3-8B end-to-end, grad parity (smoke/determinism/full), checkpoint save+resume, CUDA IPC, multi-engine fan-out, 200-step stability, convergence. (Two runs — the first hit an HF Hub `429` on the unauthenticated Qwen3-8B metadata fetch, an environment rate-limit and not a patch bug; the second set `HF_TOKEN`.)

With v0.5.10 fully validated, the colocate default was **cut over** off v0.5.8.post1 (`092b68f`): `run_smoke_host.sh`, `apply_sglang_patch.sh --colocate`, and `modal_colocate_smoke.py` now default to v0.5.10.post1. v0.5.8.post1 stays selectable via `SGLANG_PATCH_VERSION=v0.5.8.post1` but is no longer maintained — future colocate patch work lands in v0.5.10.post1 directly, ending the forward-port treadmill. A TorchSpec-side `_init_rope` fix for transformers' `rope_type="default"` (`be399a0`) was also needed for the matrix to run on a current-transformers environment.

### Follow-up round 9 (2026-05-21) — CUDA IPC default hang: diagnosed & fixed

Round 7's pending item — the IPC-default `--full` run — was attempted on 4×H100 and **hung** at colocate training-loop step 0 (every actor finished init, then froze before the first hidden-state transfer). Isolated on a 1×H100:

| Config | Result |
|---|---|
| gloo ± `expandable_segments` | PASS both ways → **`expandable_segments` ruled out** |
| CUDA IPC, probe runs | **HANG** at step 0 |
| CUDA IPC, probe skipped | PASS — `loss=12.02` |
| CUDA IPC, non-destructive probe (the fix) | PASS — `loss=12.02` |

**Root cause:** `probe_ipc_capability()` ran a `reduce_tensor()` smoke test on a scratch CUDA tensor at connector/fetcher construction — sharing it via CUDA IPC, then discarding it with no consumer ever mapping it. That leaves PyTorch's CUDA-IPC producer-side machinery in a state that wedges subsequent CUDA work **under MPS**, hanging the engine's next sglang `generate()` forward. The transport itself is innocent — connector/fetcher instrumentation confirmed `ipc_send`/`ipc_recv` are never reached; once the probe is skipped they carry the step correctly (IPC loss bit-identical to gloo, `12.021415908336417`).

**Fix** (`e166c21`): `probe_ipc_capability()` no longer calls `reduce_tensor()`. The only capability that matters for the classic container-friendly handle path — memory must not be `expandable_segments` — is now checked from `PYTORCH_CUDA_ALLOC_CONF`/`PYTORCH_ALLOC_CONF`, a non-destructive config check. `ensure_ipc_usable()` still fails fast. `test_cuda_ipc.py` 13/13; GPU-verified — IPC-default colocate tiny passes with the real fixed probe.

**Second bug, found by the `--full` re-run** (`e62c941`): `test_colocate_tiny.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the `train_entry` driver env; the engine actor **inherits** it, and CUDA IPC genuinely cannot use expandable_segments memory on a no-`CAP_SYS_PTRACE` container. The probe correctly rejected it — but `factory.py`/`train_group.py` only *skipped adding* expandable_segments for IPC actors, not *overriding* the inherited value. Both now actively set `expandable_segments:False` for IPC actors.

**`--full` re-validation — GREEN (4×H100, 2026-05-21).** With both fixes, **13 colocate tests pass under CUDA IPC default**: tiny one-step + loss-decrease, `test_phase4_one_step` (4-GPU/4-engine Qwen3-8B), grad parity (determinism/full/vs-disagg), checkpoint save+resume, `test_colocate_ipc`, `test_colocate_tp2`, `test_colocate_multi_engine`, `test_phase6_peak_alloc_flatness` (200 steps), `test_phase7_convergence` (50 steps, loss 12.13 → 3.27). The one non-pass — `grad_parity_smoke` (Qwen3-8B) — was an HF-Hub `429` rate-limit (environment, not a colocate defect). Real-workload CUDA IPC perf: warm colocate step ~0.18 s with the transfer ~1 % of it; `peak_alloc` flat to 0.014 % over 200 steps — see [`docs/colocate/transport_benchmark.md`](https://github.com/lightseekorg/TorchSpec/blob/feature/colocate-training-inference/docs/colocate/transport_benchmark.md).

### GPU validation — 12 sessions

| Session | Result |
|---------|--------|
| 1×H100 | patch apply + `test_colocate_tiny` + `test_engine_tp_rank_math` + grad-parity determinism + checkpoint save/resume — all PASS |
| 2×H100 | grad-parity determinism re-confirmed |
| 4×H200 | `run_smoke_host.sh --full` — **10 passed, 1 skipped, exit 0** (24m56s) |
| 2×H100 (round 2) | `test_phase7_grad_parity_full` (reframed) + `test_colocate_engine_tp2_end_to_end` — **both PASS** (the latter after the `6e74ffc` MoE-group fix) |
| 1×H100 (round 3) | `v0.5.10.post1/colocate.patch` — `test_colocate_tiny.py` **2/2 PASS** at tp=1 |
| 4×H100 (round 4) | `test_colocate_multi_engine_tp2_end_to_end` (2 engines × tp=2) **PASS**; Mooncake-disagg crash diagnosed (Go/CGO `sigfwd` conflict — not host-fixable) |
| 1×H100 + 2×H100 (round 5) | `v0.5.10.post1/colocate.patch` — `test_colocate_tiny.py` 2/2 (tp=1) + `test_colocate_tp2.py` (`engine_tp_size=2`) **PASS** |
| 2×H100 (round 6) | Mooncake-disagg with `mooncake==0.3.10.post1` (go1.24.13) — disagg run **completes 2 steps, no crash** (post2/go1.25 dies before step 1) |
| 1×H100 (round 7) | `bench_transport.py` — gloo-vs-CUDA-IPC transport benchmark; CUDA IPC **~170× faster** on the realistic 160 MB Eagle3 payload |
| 4×H100 (round 8) | `v0.5.10.post1/colocate.patch` — full `run_smoke_host.sh --full` matrix **13/13 PASS**; colocate default cut over to v0.5.10.post1 |
| 1×H100 (round 9) | CUDA IPC default hang isolated (gloo passes, IPC hangs) → root-caused to the `probe_ipc_capability` `reduce_tensor` smoke test; fixed (`e166c21`) — IPC-default colocate tiny **PASS** (`loss=12.02`) |
| 4×H100 (round 9 — `--full` re-validation) | `run_smoke_host.sh --full` under **CUDA IPC default** + the probe/expandable fixes — **13 colocate tests PASS**; warm step ~0.18 s, `peak_alloc` flat over 200 steps (`grad_parity_smoke` non-pass = HF-Hub 429, environment) |

```
test_phase4_tiny_one_step                  PASSED
test_phase7_tiny_loss_decreases            PASSED   (loss 12.02 → 9.74)
test_phase4_one_step_completes_end_to_end  PASSED   (4-GPU, Qwen3-8B)
test_phase7_grad_parity_smoke              PASSED   (4-GPU)
test_phase7_grad_parity_determinism        PASSED   (13 gradients bit-identical)
test_phase7_grad_parity_full               SKIPPED  (Mooncake disagg baseline unavailable)
test_colocate_checkpoint_save              PASSED
test_colocate_checkpoint_resume            PASSED
test_colocate_ipc_transport_end_to_end     PASSED   (5 steps, loss 12.02 → 11.38)
test_phase6_peak_alloc_flatness            PASSED   (200 steps, peak-alloc flat)
test_phase7_convergence_loss_decreases     PASSED   (50 steps, loss 12.13 → 3.28)
============ 10 passed, 1 skipped in 1496.03s ============
```

The one skip above — `test_phase7_grad_parity_full` — was the original
vs-disagg comparison, environment-gated because the disaggregated
baseline arm SIGSEGVs inside the Mooncake transfer engine's Go runtime
on rental hosts (the exact third-party fragility colocate replaces — not
a colocate defect). **Round 2 reframed this test** as a gloo-vs-CUDA-IPC
transport parity check (see [Follow-up round 2](#follow-up-round-2-2026-05-20)),
which needs no Mooncake and **PASSED on the round-2 2×H100 session** —
so it no longer skips.

### Bugs found & fixed during validation

| Commit | Fix |
|--------|-----|
| `edfdceb` | `run_smoke_host.sh`: PEP-668 pip + non-idempotent `setup_sglang` (`git clean -fd`) |
| `4e4ddc6` | grad-parity: `shuffle_dataset` is a `dataset.*` key, not `training.*` |
| `880b11a` / `fb4c7d0` | disagg grad-parity arm caught by the MPS daemon — added `force_stop_mps()` |
| `aebacda` | CUDA IPC handshake deadlocked on `send_object_list` — rewrote to plain `dist.send`/`recv` of pickled bytes |
| `f7a5aef` | CUDA IPC + `expandable_segments` needs `CAP_SYS_PTRACE` (`pidfd_getfd`) — IPC opt-in now skips `expandable_segments` to use the capability-free classic-handle path |
| `a0d71cf` | grad-parity-full skips (not fails) when the Mooncake baseline can't run |

### CUDA IPC capability finding

torch 2.9's CUDA IPC supports `expandable_segments` memory but shares
the backing fd via the `pidfd_getfd` syscall, which needs
`CAP_SYS_PTRACE` — not granted in typical containers. Plain `cudaMalloc`
memory uses the classic capability-free `cudaIpc*` handles, so
`TORCHSPEC_COLOCATE_IPC=1` makes the colocate path skip the
`expandable_segments` injection; IPC then works in any container
(validated end-to-end, loss decreasing).

## Transport optimization (round 10)

After round 9 made CUDA IPC usable as the default, the transport was
investigated for further headroom — full write-up in
[`transport_optimization.md`](docs/colocate/transport_optimization.md).

- **No hand-written C++/CUDA or Triton kernel is needed.** The CUDA IPC
  path has no GPU compute kernel — `cudaIpcGetMemHandle` (driver API) →
  a small handle blob over gloo → `cudaIpcOpenMemHandle` → one D→D
  `cudaMemcpyAsync`. That copy already runs at HBM bandwidth (~1 TB/s);
  no custom kernel can beat a bandwidth-bound copy. The only headroom is
  protocol-level.
- **`ipc-pipe` — protocol-level optimization, MPS-validated.** Two
  optimization arms (`ipc-pool`, `ipc-pipe`) were prototyped in
  `bench_transport.py`. `ipc-pipe` (persistent send-buffer pool +
  one-step ack pipelining) cuts the engine `send()` stall **3.9×** on
  the realistic Eagle3 payload, A/B-measured under MPS. It is
  **low-priority** — the transport is only ~1 % of a colocate step, so
  it is not a step-time bottleneck.
- **`ipc-pool` alone is NOT worth shipping.** The A/B (see
  `transport_optimization.md` Part 4 finding 3 + Part 5) showed
  `ipc-pool` standalone is **break-even** at most payloads and a **net
  regression at 256 MB** (engine `send()` 1.71 → 2.68 ms — copying a
  256 MB tensor into the pool costs more than the handle-open it
  avoids). The pool's value is **solely as the enabler** for
  `ipc-pipe`'s double-buffered ack deferral; the
  `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag deliberately enables pool +
  pipe together, never the pool by itself.
- **3000-step 4-GPU stability soak** — `colocate_qwen0p6b_2eng_tp2_tiny`
  (2 engines × tp2, 4×H100 MPS-shared), CUDA IPC default: 3000/3000
  steps, no hang, step time and `peak_alloc` flat throughout.

## Production-scale colocate runs (rounds 11 + 12)

Until round 11 the colocate path had been GPU-validated only against
the `--full` CI matrix (Qwen3-0.6B tiny + a 4-engine Qwen3-8B
one-step) and a 3000-step Qwen0.6B soak. **Production-scale (20000-step
/ 40k-sample) Qwen3-8B colocate runs were unproven** until the two
benchmark cells below — one per draft model family — completed
end-to-end. Both were matched against the **same-SGLang disagg rerun
on `origin/main @ 068f253`** (see
`docs/colocate/modal_benchmark/dflash_eagle3_disagg_modal_rerun_on_main.md`),
which retires the cross-branch confound earlier versions of the
benchmark carried.

| Cell | Steps | Samples | Throughput | GPU-h | Disagg baseline | Win |
|---|--:|--:|--:|--:|--:|---|
| **CE1** — Eagle3 2+2 colocate (round 11, 2026-05-21) | 20000 | 40000 | ~13.25 samples/s | **1.68** / 40k (2 GPU) | E1-rerun = 12.72 samples/s, **3.49** / 40k (4 GPU) | **~2.1× less GPU-h** |
| **C1** — DFlash 2+2 colocate (round 12, 2026-05-22) | 20000 | 40000 | 7.51 samples/s | **2.96** / 40k (2 GPU) | D1-rerun = 10.00 samples/s, **4.44** / 40k (4 GPU) | **~1.5× less GPU-h** |
| CE2 — Eagle3 4+4 colocate | — | — | — | — | — | pending |
| C2 — DFlash 4+4 colocate | — | — | — | — | — | pending |

Both wins decompose cleanly as `2.0 ×` (half the GPU count via MPS
sharing) `× r` (colocate's raw-throughput ratio): Eagle3 r ≈ 1.0
(colocate ≈ even with disagg), DFlash r ≈ 0.75 (heavier trainer →
more MPS contention → ~25 % raw-throughput hit). **The architectural
saving is reclaiming the idle disagg inference GPUs**; both trainers
do the same draft-model math regardless of where inference runs. Full
analysis: `docs/colocate/modal_benchmark/colocate_benchmark.md`.

**Convergence holds for both cells.** CE1's final rolling loss
(~2.09 at 40k samples) matches disagg E1's (2.24 / 1.98) — equal data,
equal LR phase, equal convergence. C1's final rolling loss (~3.81 at
40k samples) sits inside the disagg D1 noise band (D1 orig 3.67,
D1-rerun-on-`main` 4.89). Notably, the disagg D1 rerun on `main`
flagged a **DFlash loss regression** (3.67 → 4.89, attributed by the
rerun doc to FA4 #96 / post-norm #97 changes on the trainer side);
**C1 colocate does not show that regression** — its loss lands
between the two disagg points, so the colocate path is producing
genuine, on-trend DFlash training, not a degraded variant.

## One-pod batch validation (round 11)

Round 11 productionized `ipc-pipe`, added the convergence-vs-Mooncake
test, and GPU-validated four issue-#81 follow-ups in one 4×H100
secure-cloud pod session (~1.6 h, ~$21). Full results in
[`handoff_followups.md`](docs/colocate/handoff_followups.md).

- **`ipc-pipe` is now production-wired.** `IpcPipelineTransport` in
  `cuda_ipc.py` (send-buffer pool + handle cache + one-step ack deferral)
  is wired into the connector + fetcher behind the opt-in
  `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag (default off). Teardown-safe
  without a flush, so no sglang-patch change was needed.
- **One bug found and fixed.** `--full` with the flag passed 12/13;
  `test_phase6_peak_alloc_flatness` OOM'd the memory-tight Qwen3-8B
  config — the pool's variable-`seq_len` resize retired buffers without
  freeing them, and the ×2 grow overshoot stacked on sglang's KV cache.
  Fixed: exact-size grow + retired buffers freed one step after the
  trainer acks the resize. Re-test GREEN (peak-alloc flat ~25.75 GB).
- **#3 `--stability` 1000-step** — GREEN (exit 0). **#6
  `grad_parity_smoke`** (Qwen3-8B) — GREEN in `--full` with `HF_TOKEN`.
- **#4 convergence vs Mooncake** — `test_convergence_disagg_overlap`
  ran 1000 steps each arm: colocate vs disagg loss curves overlap at
  **mean 0.006 % / max 0.219 %** deviation — the colocate transport
  converges identically to the disaggregated baseline.

## DFlash colocate two deadlocks (round 12, 2026-05-22)

Round 11 GPU-validated Eagle3 at production scale; **the DFlash
colocate path had not been exercised at production scale until round
12**. Bringing DFlash up surfaced two distinct, sequential
DFlash-only deadlocks. Both are now fixed (`f28dc73`) and the C1
20000-step run completed cleanly (above).

| # | Hang | Root cause | Fix |
|---|------|-----------|-----|
| 1 | `DFlashTrainer._init_target_lm_head` froze right after `[Rank 0] TargetLMHead loaded` | `dist.barrier()` / `dist.broadcast()` + 3 `dist.all_reduce()` ran with no `group=` → in colocate the default PG is the **union world** (trainer `[0,N)` + engine `[N,2N)`); only trainer ranks execute the method, the engine ranks never arrive → deadlock. **Same shape as the round-7 `set_model_state_dict` / `dcp.save` / `dcp.load` bugs.** `Eagle3Trainer` already carried the fix; `DFlashTrainer` (comment-labelled "same as Eagle3Trainer") had never received it. | All 5 collectives scoped to `get_gloo_group()`. No-op for disagg. |
| 2 | `colocate_loop` froze at step 0; faulthandler dump put the engine in `cuda_ipc.py:250 ipc_send` `dist.recv` waiting for an ack | CUDA-IPC handshake is **per-tensor**: engine ships one IPC handle per tensor and blocks for one ack each. Colocate engine *always* sends `last_hidden_states` (`enable_return_hidden_states=True` is unconditional); trainer's `_build_tensor_specs` gated it on `store_last_hidden_states` (= `false` in DFlash's config) → **3 sent, 2 declared → 3rd `dist.recv` blocked forever**. CE1 (Eagle3, `store_last_hidden_states: true`) declared all 3, so it never hit the bug. | `_build_tensor_specs` now **always** declares `last_hidden_states`. Trainers that don't consume it (DFlash) ignore the extra ~3 MB / step bf16 buffer. The `store_last_hidden_states` parameter was removed. |

**How hang #2 was pinned.** Three debug rounds were needed; the first
two used `[HANG2]` phase markers + `[HANG2-DEBUG]` payload prints and
gave a *wrong* intermediate conclusion that the hang was upstream of
the transfer (the markers simply weren't captured before the freeze).
The third ran with `PYTHONFAULTHANDLER=1` and `kill -ABRT <pid>` on the
hung processes — **no ptrace needed** (`py-spy` is blocked by the
container's missing `CAP_SYS_PTRACE`, but `SIGABRT` is allowed to the
process owner), and dumped every Python thread's stack to stderr.
That pinned it exactly.

### Lessons captured

1. **Trainer-only collectives must scope `group=` in colocate.** Any
   bare `dist.barrier` / `broadcast` / `all_reduce` on a trainer path
   will hang the union default PG. The five `dflash_trainer.py` sites
   are the same shape as the round-7 fixes. Worth converting into a
   runtime check inside `Trainer.__init__` (assert the default PG is
   the trainer-only group, not the union world) or a colocate-mode
   lint that flags `dist.*(group=None)` in trainer files.
2. **The tensor spec is the engine's, not a config flag.** The
   trainer-side `tensor_specs` is a contract that must mirror what the
   engine actually sends, not what a training-side config says it
   should want. `store_last_hidden_states` is a training-side
   preference; the wire payload is decided by the engine + sglang
   patch. Any future draft variant must derive its
   `_build_tensor_specs` from the engine's real output set (or, better,
   have the engine announce its key set on the metadata channel) —
   never from a local flag. Captured in the new `_build_tensor_specs`
   docstring (`colocate_loop.py:71-95`).

### Companion cleanup

`a2ed921` drops a duplicated `_COLOCATE_UNION_WORLD_PORT_OFFSET`
constant in `trainer_actor.py` (cherry-pick artefact, no functional
change).

## Open follow-ups (tracked, not blocking this PR)

| Follow-up | Why it's open |
|-----------|---------------|
| Multi-node 2-node colocate run | code-complete (`ensure_mps_on_all_nodes`, 2-node config) but untested at scale — needs a 2-node rented cluster with cross-node networking |
| Large `engine_tp_size` (8-GPU TP per engine) | rank math + data plane handle any TP size but are only GPU-tested at `engine_tp_size=2`; issue-#81 scale-out wants 1 engine × 8-GPU TP — needs an 8-GPU config + run |
| v0.5.10 `pp_size>1` | `v0.5.10.post1/colocate.patch` passed the full 4×H100 `--full` matrix and is now the default; only `pp_size>1` (pipeline parallelism) is unexercised — blocked by an explicit guard, out of scope for the current colocate plan |
| CE2 / C2 benchmark cells (4+4 colocate) | the disagg-vs-colocate study (`colocate_benchmark.md`) has CE1 + C1 done at 2+2; CE2 (Eagle3 4+4) and C2 (DFlash 4+4) are the outstanding cells, matched against the existing disagg E2 / D2 rerun-on-`main` baselines. Code-ready, unrun — needs one 4×H100 pod and a matched 40k-sample run per cell. **Next productive item that does not need new hardware beyond a 4-GPU pod.** |
| `draft_accumulation_steps > 1` in `colocate_loop.py` | guarded with `NotImplementedError("Multi-step accumulation is parked")`; CE1/C1 ran at `accum=1` / global-batch 2 so they cannot match the disagg §8 contract (`accum=4` / global-batch 8). Out of scope unless the benchmark needs the §8 cell-for-cell parity. |
| Colocate fail-fast for spec / default-PG mismatches | round 12 found two distinct silent-deadlock failure modes (bare collective → union default PG; tensor-spec count mismatch). Both could be turned into immediate, legible errors with: (a) a runtime check in `Trainer.__init__` that asserts the default PG is **not** the union world, and/or a colocate-mode lint that flags `dist.*(group=None)` in trainer files; (b) a step-0 watchdog in `colocate_loop.py` that times out the first `engine_refs` `ray.get` and dumps both sides' tensor specs on mismatch. Small code change, high value — converts the next deadlock of either shape into an immediate error instead of a multi-pod debug round. |
| ~~Literal Mooncake-disagg parity~~ | ✅ **Done.** Per-parameter gradient parity vs the disagg baseline is covered by `test_phase7_grad_parity_vs_disagg` (1-step), and the 1k-step convergence-curve comparison by `test_convergence_disagg_overlap` — GPU-validated round 11 (loss curves overlap mean 0.006 % over 1000 steps). The Mooncake crash that blocked this was fixed in round 6 (`mooncake-transfer-engine==0.3.10.post1`). |
| ~~`--full` re-run with CUDA IPC as default~~ | ✅ **Done (round 9).** 4×H100 `run_smoke_host.sh --full` under CUDA IPC default — 13 colocate tests pass after the `e166c21` probe fix + `e62c941` expandable-segments fix. |
| ~~Productionize `ipc-pipe` (ack pipelining)~~ | ✅ **Done (round 11).** Folded into `cuda_ipc.py` as `IpcPipelineTransport` behind the opt-in `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag; GPU-validated on 4×H100 (one OOM bug on the 8B config found + fixed). Opt-in and low-priority — the transport is ~1 % of a colocate step. |
| ~~Eagle3 production-scale colocate run~~ | ✅ **Done (round 11, CE1).** Qwen3-8B Eagle3 2+2 colocate, 20000 steps / 40k samples, `rc=0`. ~13.25 samples/s, **~1.68 GPU-h** for 40k samples on 2 GPU vs **3.49 GPU-h** on 4 GPU for same-SGLang disagg E1-rerun → **~2.1× less GPU-h**. |
| ~~DFlash production-scale colocate run~~ | ✅ **Done (round 12, C1).** Qwen3-8B DFlash 2+2 colocate, 20000 steps / 40k samples, `rc=0`. Two latent DFlash-only deadlocks fixed in `f28dc73` (see "DFlash colocate two deadlocks (round 12)" above). 7.51 samples/s, **~2.96 GPU-h** for 40k samples on 2 GPU vs **4.44 GPU-h** on 4 GPU for same-SGLang disagg D1-rerun → **~1.5× less GPU-h**. |

## Full debug log

[`docs/colocate/implementation_log.md`](https://github.com/lightseekorg/TorchSpec/blob/feature/colocate-training-inference/docs/colocate/implementation_log.md) — RunPod sessions #1-#3 (1×H100 / tiny green) + Vast sessions #4-#5 (4×H100 / full green) + follow-up rounds 1-12 (grad parity, CUDA IPC, multi-engine TP + fan-out, v0.5.10 port + multi-TP validation, RoPE fix, Mooncake crash diagnosis + fix, CUDA-IPC-default switch + transport benchmark, v0.5.10 full-matrix cutover, CUDA-IPC-default hang diagnosis + probe fix, transport optimization investigation + MPS re-benchmark, ipc-pipe productionization + one-pod GPU validation of issue-#81 follow-ups, CE1 production-scale Eagle3 20000-step run, round 12 DFlash two deadlocks + C1 production-scale DFlash 20000-step run). Transport benchmark detail: [`docs/colocate/transport_benchmark.md`](https://github.com/lightseekorg/TorchSpec/blob/feature/colocate-training-inference/docs/colocate/transport_benchmark.md). Disagg-vs-colocate study: [`docs/colocate/modal_benchmark/colocate_benchmark.md`](https://github.com/lightseekorg/TorchSpec/blob/feature/colocate-training-inference/docs/colocate/modal_benchmark/colocate_benchmark.md).

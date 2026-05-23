# Colocate (PR #92 / issue #81) — leftover follow-ups (handoff)

> Handoff summary as of **2026-05-22**. Self-contained — an agent picking
> this up should not need prior conversation context.

## Current state (grounding)

- **Branch:** `feature/colocate-training-inference` is the PR branch;
  `feature/colocate-training-inference-development` is the local
  development branch and carries the latest two DFlash fixes
  (`f28dc73`, `a2ed921`, 2026-05-22) on top of everything the PR branch
  has. **PR #92** is still `[WIP]` DRAFT on
  `github.com/lightseekorg/TorchSpec`. Repo fork remote:
  `zhubohao911/TorchSpec`. The PR description was kept concise — full
  detail is preserved in `docs/colocate/pr92_detail.md`.
- **Transport:** CUDA IPC zero-copy is the **default**;
  `TORCHSPEC_COLOCATE_IPC=0` opts back to gloo CPU-staging. Three pieces:
  `e166c21` (non-destructive IPC capability probe — the old
  `reduce_tensor` probe wedged CUDA under MPS), `e62c941`
  (factory/train_group actively clear `expandable_segments` for IPC
  actors), and **round 10/11** (transport optimization — no
  C++/CUDA/Triton kernel needed; `ipc-pipe` ack pipelining is a
  low-priority protocol-level 3.9× win, productionized in `cuda_ipc.py`
  behind the opt-in `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag, GPU-validated
  2026-05-21).
- **Eagle3 validation:** `run_smoke_host.sh --full` is **green on 4×H100
  under IPC default** — 13 colocate tests pass (single-node). A
  3000-step 4-GPU multi-engine soak (round 10) ran clean. CE1 (Eagle3
  2+2 colocate) ran **20000 steps / 40k samples** clean (`rc=0`) on
  2×H100 at ~13.25 samples/s — **~2.1× less GPU-h** than the same-SGLang
  disagg E1 rerun.
- **DFlash validation (new, 2026-05-22):** C1 (DFlash 2+2 colocate) ran
  **20000 steps / 40k samples** clean (`rc=0`) on 2×H100 at 7.51
  samples/s — **~1.5× less GPU-h** than the same-SGLang disagg D1
  rerun. Two latent DFlash-only deadlocks were root-caused and fixed
  here (round 12) — see "What changed since 2026-05-21" below.
- **sglang patch:** `v0.5.10.post1` is the default
  (`v0.5.8.post1` still selectable via `SGLANG_PATCH_VERSION`).
- **Docs of record:** `docs/colocate/implementation_log.md` (rounds
  1–12), `docs/colocate/transport_benchmark.md`,
  `docs/colocate/transport_optimization.md` (transport
  kernel-vs-protocol investigation + MPS-validated A/B),
  `docs/colocate/pr92_detail.md` (full PR narrative),
  `docs/colocate/modal_benchmark/colocate_benchmark.md` (the
  disagg-vs-colocate study — CE1 + C1 done, CE2/C2 pending).
- **GPU access:** `runpodctl` is configured; SSH key
  `~/.runpod/ssh/runpodctl-ssh-key`; recipe = clone the branch +
  `bash scripts/colocate/run_smoke_host.sh --full`. Colocate cannot
  run on Modal (gVisor blocks NVIDIA MPS) — use RunPod / Vast.ai with
  `--ipc=host`.

## What changed since the 2026-05-21 handoff (round 12)

Two distinct, sequential **DFlash-only** colocate deadlocks were found
and fixed (`f28dc73`, 2026-05-22). With both fixes a 20000-step DFlash
2+2 colocate run completed cleanly (`rc=0`, zero hang / NaN / OOM).

| # | Hang | Where | Fix |
|---|------|-------|-----|
| 1 | `DFlashTrainer._init_target_lm_head` froze right after `[Rank 0] TargetLMHead loaded`. `dist.barrier()` / `dist.broadcast()` + 3 `dist.all_reduce()` ran with **no `group=`** → default PG in colocate is the **union world** (trainer `[0,N)` + engine `[N,2N)`); only trainer ranks execute this method, the engine ranks never arrive → deadlock. | `torchspec/training/dflash_trainer.py` | All 5 collectives scoped to `get_gloo_group()` (the trainer-only group). Mirrors the same fix `Eagle3Trainer` already carries. No-op for disagg (where `get_gloo_group()` is the whole trainer world). |
| 2 | `colocate_loop` froze at step 0 — engine wedged in `ipc_send` `dist.recv` waiting for an ack. CUDA-IPC handshake is **per-tensor**: engine ships one IPC handle per tensor, blocks for one ack each. The colocate engine *always* sends `last_hidden_states` (`enable_return_hidden_states=True` is unconditional), but the trainer's `_build_tensor_specs` gated it on `store_last_hidden_states` — `false` for DFlash. **3 sent, 2 declared → 3rd `dist.recv` blocked forever.** | `torchspec/controller/colocate_loop.py` | `_build_tensor_specs` now **always** declares `last_hidden_states`. Trainers that don't consume it (DFlash) ignore the extra `(seq_len, hidden_size)` bf16 buffer per step (~3 MB, negligible). The `store_last_hidden_states` parameter is removed. |

Why CE1 (Eagle3) didn't hit either bug: (a) Eagle3 sets
`store_last_hidden_states: true`, so the third tensor was already
declared; and (b) `Eagle3Trainer._init_target_lm_head` already scopes
its collectives to `get_gloo_group()` — `DFlashTrainer`, comment-labelled
"same as Eagle3Trainer", had never received that fix. The bugs were
DFlash-config-specific, not transport-specific.

Hang #2 was pinned by a `PYTHONFAULTHANDLER=1` + `kill -ABRT <pid>`
stack dump (py-spy is blocked on RunPod's no-`CAP_SYS_PTRACE`
containers). The fault dump cleanly showed the engine blocked in
`cuda_ipc.py:250 ipc_send` waiting for the missing ack, plus the
3-tensor vs 2-spec mismatch in the `[HANG2-DEBUG]` payload prints
intermediate rounds had instrumented.

**Lesson — the tensor spec is the engine's, not a config flag.** The
trainer-side `tensor_specs` is a **contract that must mirror what the
engine actually sends**, not what a training-side config says it should
want. `store_last_hidden_states` is a training-side preference; the
wire payload is decided by the engine + sglang patch. Any future draft
variant must derive its `_build_tensor_specs` from the engine's real
output set (or, better, have the engine announce its key set on the
metadata channel) — never from a local flag. Captured inline in the
new `_build_tensor_specs` docstring (`colocate_loop.py:71-95`).

Also folded in: `a2ed921` dropped a duplicated
`_COLOCATE_UNION_WORLD_PORT_OFFSET` constant in `trainer_actor.py`
(cherry-pick artefact, no functional change).

## Leftover items

| # | Item | Status | What "done" needs |
|---|---|---|---|
| 1 | **Multi-node 2-node run** | code-complete, untested | Run colocate on 2 nodes × 8 GPU. Code: `ensure_mps_on_all_nodes` (`torchspec/colocate/mps.py`), config `configs/colocate_qwen3_8b_2node.yaml`. Needs a 2-node rented cluster with cross-node networking. |
| 2 | **Large `engine_tp_size` (8-GPU TP per engine)** | validated only at `engine_tp_size=2` | Issue #81 scale-out wants 1 engine × 8-GPU TP. Rank math (`engine_global_rank`, `build_engine_tp_ranks`) + data plane (`colocate_loop.py` dispatch, `build_hidden_states_writer(tp_rank)`, `_send_hidden_states_to_nccl` in `colocate.patch`) handle any TP size but are only GPU-tested at tp=2 (`test_colocate_tp2.py`) + 2-engine fan-out (`test_colocate_multi_engine.py`). Needs an 8-GPU config + run. |
| 3 | **`pp_size > 1`** | open, **out-of-scope by agreement** | Pipeline parallelism — blocked by an explicit guard in `colocate.patch`. Listed for completeness; not planned. |
| 4 | **CE2 / C2 benchmark cells (4+4 layouts)** | code-ready, unrun | The disagg-vs-colocate study's outstanding cells (`docs/colocate/modal_benchmark/colocate_benchmark.md`): CE2 = Eagle3 4+4 colocate on 4 GPUs, C2 = DFlash 4+4 colocate on 4 GPUs, both matched against the existing disagg E2 / D2 rerun-on-`main` baselines. CE1 + C1 are done. Needs one 4×H100 pod and a matched 40k-sample run per cell. |
| 5 | **`draft_accumulation_steps > 1`** | open, parked | `colocate_loop.py` guards with `NotImplementedError("Multi-step accumulation is parked")`. CE1/C1 ran at `accum=1` / global-batch 2, which breaks cell-for-cell parity with the disagg `accum=4` / global-batch 8 contract in the benchmark's §8. Unblocking gradient accumulation would let the benchmark pin its global batch and remove a confound. |
| 6 | **Colocate fail-fast for spec/PG mismatches** | not started, recommended | Round 12 found two distinct silent-deadlock failure modes (bare collective → default PG; tensor-spec count mismatch). Both could be turned into immediate, legible errors with: (a) a colocate-mode lint that flags `dist.*(group=None)` in trainer files, or a runtime check that the default PG is **not** the union world inside trainer-only paths; and (b) a step-0 watchdog in `colocate_loop.py` that times out the first `engine_refs` `ray.get` and dumps both sides' tensor specs on mismatch. |

Items **#1 and #2 are the only remaining issue-#81 "Scale-out" work** —
both need different hardware (2 nodes / 8 GPUs), not code. **#4 (CE2
/ C2)** is the next *productive* item that does not require new
hardware beyond a 4×H100 pod, and is what the benchmark study still
needs to close. **#5 (accum)** and **#6 (fail-fast)** are
quality-of-life follow-ups surfaced by round 12.

The `--stability`, convergence-vs-Mooncake, Qwen3-8B grad-parity, and
`ipc-pipe`-productionization follow-ups were **GPU-validated 2026-05-21**
on a 4×H100 pod — see `implementation_log.md` round 11. The DFlash
colocate path was **GPU-validated 2026-05-22** on a 2×H100 pod — see
round 12.

## What is the next item for this issue?

Ranked by ROI given current state and hardware constraints:

1. **CE2 + C2 (4+4 colocate cells)** — closes the disagg-vs-colocate
   benchmark study (`colocate_benchmark.md`) and gives the first real
   `dp_size > 2` colocate signal under MPS contention. Needs one 4×H100
   pod, ~$25 per run, ~2 h each. No code change required beyond
   matched-step launch overrides.
2. **Multi-node 2-node run (#1)** — the largest open issue-#81 item;
   code-complete but untested at scale. Needs a 2-node rented cluster.
3. **`engine_tp_size=8` (#2)** — the other open issue-#81 scale-out
   item; needs an 8-GPU pod and an 8-GPU config.
4. **Round-12 lessons — fail-fast guards (#6)** — small code change;
   high value because it converts the next deadlock of either shape
   into an immediate error instead of a multi-pod debug round.

`pp_size>1` (#3) is out of scope by agreement and not on this list.
`draft_accumulation_steps>1` (#5) is also out of scope until and
unless the benchmark needs `accum=4` for the §8 contract.

## PR #92 description

Kept concise — the full phase / round / bug detail lives in
`docs/colocate/pr92_detail.md` and the PR body links there. The body's
"Open follow-ups" line should match the leftover-items table above:
2-node (#1), 8-GPU-TP (#2), out-of-scope `pp_size>1` (#3), CE2/C2
benchmark cells (#4), accum (#5), fail-fast guards (#6).

## Environment gotchas for the GPU work

- HF-Hub **429 rate-limits** unauthenticated Qwen3-8B fetches mid-`--full`;
  set `HF_TOKEN`, or pre-cache models + `HF_HUB_OFFLINE=1`.
- RunPod **community-cloud H100s are usually unavailable** — secure cloud
  (~$3.29/GPU/hr) works.
- This container type **blocks `py-spy`/ptrace**; for hung-process
  diagnosis use `faulthandler.dump_traceback_later` via a
  `sitecustomize.py`, **or** `PYTHONFAULTHANDLER=1` + `kill -ABRT <pid>`
  on the hung processes — the round-12 hang #2 was pinned with the
  latter. Not a SIGUSR1 handler.
- `uv` + backgrounding the Qwen3-8B / perfectblend downloads cuts env
  setup from ≥50 min (`pip`) to ~100 s. CE1 + C1 both used the `uv`
  launcher.
- The 8B runs leave large checkpoints in `outputs/` — `rm -rf
  outputs/* /tmp/ray/*` between phases or the 200 GB pod disk fills
  and Ray fails to acquire GPUs (round-11 finding).

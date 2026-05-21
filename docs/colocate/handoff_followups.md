# Colocate (PR #92 / issue #81) — leftover follow-ups (handoff)

> Handoff summary as of 2026-05-21. Self-contained — an agent picking this
> up should not need prior conversation context.

## Current state (grounding)

- **Branch:** `feature/colocate-training-inference`;
  **PR #92** (still `[WIP]` DRAFT) on `github.com/lightseekorg/TorchSpec`.
  Repo fork remote: `zhubohao911/TorchSpec`. The round-10
  transport-optimization work is merged in (`8905c55`); the PR
  description was rewritten concise — full detail preserved in
  `docs/colocate/pr92_detail.md`.
- **Transport:** CUDA IPC zero-copy is the **default**;
  `TORCHSPEC_COLOCATE_IPC=0` opts back to gloo CPU-staging. Three pieces:
  `e166c21` (non-destructive IPC capability probe — the old
  `reduce_tensor` probe wedged CUDA under MPS), `e62c941`
  (factory/train_group actively clear `expandable_segments` for IPC
  actors), and **round 10** (transport optimization investigated — no
  C++/CUDA/Triton kernel needed; `ipc-pipe` ack pipelining is a
  low-priority protocol-level 3.9× win, now wired into `cuda_ipc.py`
  behind the opt-in `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag, GPU-validated
  2026-05-21 — see `implementation_log.md` round 11).
- **Validated:** `run_smoke_host.sh --full` matrix is **green on 4×H100
  under IPC default** — 13 colocate tests pass (single-node). A
  3000-step 4-GPU multi-engine soak (round 10) ran clean. sglang patch:
  `v0.5.10.post1` is the default (`v0.5.8.post1` still selectable via
  `SGLANG_PATCH_VERSION`).
- **Docs of record:** `docs/colocate/implementation_log.md` (rounds
  1–11), `docs/colocate/transport_benchmark.md`,
  `docs/colocate/transport_optimization.md` (transport
  kernel-vs-protocol investigation + MPS-validated A/B),
  `docs/colocate/pr92_detail.md` (full PR narrative).
- **GPU access:** `runpodctl` is configured; SSH key
  `~/.runpod/ssh/runpodctl-ssh-key`; recipe = clone the branch +
  `bash scripts/colocate/run_smoke_host.sh --full`.

## Leftover items

| # | Item | Status | What "done" needs |
|---|---|---|---|
| 1 | **Multi-node 2-node run** | code-complete, untested | Run colocate on 2 nodes × 8 GPU. Code: `ensure_mps_on_all_nodes` (`torchspec/colocate/mps.py`), config `configs/colocate_qwen3_8b_2node.yaml`. Needs a 2-node rented cluster with cross-node networking. |
| 2 | **Large `engine_tp_size` (8-GPU TP per engine)** | validated only at `engine_tp_size=2` | Issue #81 scale-out wants 1 engine × 8-GPU TP. Rank math (`engine_global_rank`, `build_engine_tp_ranks`) + data plane (`colocate_loop.py` dispatch, `build_hidden_states_writer(tp_rank)`, `_send_hidden_states_to_nccl` in `colocate.patch`) handle any TP size but are only GPU-tested at tp=2 (`test_colocate_tp2.py`) + 2-engine fan-out (`test_colocate_multi_engine.py`). Needs an 8-GPU config + run. |
| 3 | **`pp_size > 1`** | open, **out-of-scope by agreement** | Pipeline parallelism — blocked by an explicit guard in `colocate.patch`. Listed for completeness; not planned. |

Items **#1 and #2 are the only remaining issue-#81 "Scale-out" work** —
both need different hardware (2 nodes / 8 GPUs), not code. The
`--stability`, convergence-vs-Mooncake, Qwen3-8B grad-parity, and
`ipc-pipe`-productionization follow-ups were **GPU-validated 2026-05-21**
on a 4×H100 pod — see `implementation_log.md` round 11 for the results.

## PR #92 description

Kept concise — the full phase / round / bug detail lives in
`docs/colocate/pr92_detail.md` and the PR body links there. The body's
"Open follow-ups" line matches the leftover-items table above: 2-node
(#1), 8-GPU-TP (#2), and out-of-scope `pp_size>1` (#3).

## Environment gotchas for the GPU work

- HF-Hub **429 rate-limits** unauthenticated Qwen3-8B fetches mid-`--full`;
  set `HF_TOKEN`, or pre-cache models + `HF_HUB_OFFLINE=1`.
- RunPod **community-cloud H100s are usually unavailable** — secure cloud
  (~$3.29/GPU/hr) works.
- This container type **blocks `py-spy`/ptrace**; for hung-process
  diagnosis use `faulthandler.dump_traceback_later` via a
  `sitecustomize.py`, not a SIGUSR1 handler.

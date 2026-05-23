# Colocate Mode — Usage Guide

> Run a TorchSpec spec-decoding training job where the trainer and the
> sglang inference engine share the same physical GPUs via NVIDIA MPS,
> with hidden states crossing the boundary on-device (no Mooncake).
>
> **Status:** the TorchSpec side of the path lands in this PR; the
> end-to-end run also requires an upstream sglang patch — see
> [`sglang_patch.md`](sglang_patch.md). Without that patch, init succeeds
> but the first step hangs (the engine never sends).
>

> ⚠️ **Transport — updated 2026-05-21.** This guide originally said hidden
> states cross "over NCCL P2P". That turned out to be impossible: NCCL
> hard-rejects a communicator with two ranks on one physical GPU
> (`ncclInvalidUsage`, "Duplicate GPU detected"). The hidden-state plane
> is now **CUDA IPC zero-copy by default** (gloo CPU-staged is the
> opt-out fallback) — see *Hidden-state transport* below, plus
> [`transport_benchmark.md`](transport_benchmark.md) and
> [`implementation_log.md`](implementation_log.md) rounds 1/7/9. Older
> "NCCL P2P" wording elsewhere in this file is kept for history and
> flagged inline.
>
> Background reading:
> - [`knowledge/knowledge.zh-en.md`](knowledge/knowledge.zh-en.md) —
>   bilingual conceptual background: what MPS / NCCL / fractional Ray
>   bundles / the union world actually do here, plus the two colocate
>   contracts (default PG and wire-payload).
> - [`implementation.md`](implementation.md) — the phased build plan.
> - [`implementation_log.md`](implementation_log.md) — what is actually
>   wired up so far + Modal verification status.

## When to use colocate mode

Use colocate (`colocate_strategy=mps`, `transfer_mode=nccl`) when **all**
of these are true:

- Single-node training (1 host).
- Inference engine is **sglang** (not vLLM).
- You want to halve GPU count by running trainer + engine on the same
  GPUs.
- Spec-training is the workload (Eagle3-style aux-hidden-state pipe).

Use the default disaggregated path (separate trainer GPUs + engine GPUs +
Mooncake transport) when:

- Multi-node setup, **or**
- Multiple engine replicas / async pipelining, **or**
- vLLM engine.

## Hardware & software prerequisites

- 1 node, **N ≥ 2** GPUs (we test on 4×H100 80GB; 2-GPU smoke runs in
  CI).
- NVIDIA driver supporting MPS (anything ≥ R535).
- `nvidia-cuda-mps-control` binary in `$PATH` — ships with the CUDA
  toolkit. The driver auto-starts the daemon via
  `torchspec/colocate/mps.py:setup_for_colocate` when the first trainer
  actor comes up; you should not start it manually.
- `expandable_segments:True` for the PyTorch CUDA allocator (set via
  `PYTORCH_CUDA_ALLOC_CONF`). The example `run.sh` does this for you.
  ⚠️ *Update (2026-05-21): only the **gloo** fallback transport wants
  `expandable_segments`. With the default CUDA IPC transport the colocate
  path actively disables it (IPC needs plain `cudaMalloc` memory) — see*
  Hidden-state transport *below.*
- `torch ≥ 2.4`, `sglang` with the colocate patch from
  [`sglang_patch.md`](sglang_patch.md).

## GPU layout invariants

Colocate mode pins the layout to **1:1 trainer↔engine pairs**:

```
training_num_gpus_per_node = N
inference_num_gpus         = N
inference_num_gpus_per_engine = 1     # always 1 in colocate
inference.sglang.tp_size  = 1         # always 1 in colocate
```

Each GPU `i` ∈ `[0, N)` runs both:

- Trainer rank `i`     — global rank `i`     in the union NCCL world.
- Engine rank `i` (TP=1) — global rank `N+i` in the union NCCL world.

The Phase-2 `init_union_world` helper builds this `2N`-rank world; FSDP
collectives go on the `[0, N)` subgroup; metadata broadcasts go on a
gloo `[0, 2N)` subgroup. Hidden states cross between `i` and `N+i` over
that gloo `meta_group` — by default as a **CUDA IPC** zero-copy handoff,
with gloo CPU-staging as the opt-out fallback. (⚠️ *Update 2026-05-21:
earlier drafts said "via P2P on the union NCCL default group" — that is
wrong; NCCL cannot form a communicator with two ranks on one physical
GPU. See* Hidden-state transport *below.*)

If you violate the invariant (e.g. `tp_size>1`), Phase-0 validation in
`train_entry.parse_config()` errors out with the offending product.

## Per-GPU memory split

Each GPU's memory is split between trainer and engine:

```
train_frac + infer_frac + 0.10 ≤ 1.0
```

- `train_frac` is propagated to `torch.cuda.set_per_process_memory_fraction(train_frac)`
  inside the trainer actor.
- `infer_frac` overrides sglang's `mem_fraction_static` inside
  `SglEngine.init`. Anything you set in `inference.sglang.mem_fraction_static`
  is overridden — in colocate mode the budget lives on `infer_frac`.
- The `0.10` slack is reserved for NCCL workspace, Python, and the
  CUDA driver. Do not lower it.

Default values (when both are unset under colocate) are `0.45 / 0.45`,
which is a safe starting point on H100 80GB for Qwen3-8B. Tune empirically
once Phase-6 stability runs land.

## Quickstart: 1-node 4×H100 Qwen3-8B

The shipped example mirrors `examples/qwen3-8b-single-node/` but pins
the colocate layout. Both the config and the run script are deliberately
diffable against the disaggregated example to make the colocate-only
changes obvious.

```bash
# default 4-GPU layout
./examples/colocate-qwen3-8b-1node/run.sh

# explicit GPU pinning
CUDA_VISIBLE_DEVICES=0,1,2,3 ./examples/colocate-qwen3-8b-1node/run.sh

# override config from CLI (Phase-0 flat-args parser)
./examples/colocate-qwen3-8b-1node/run.sh \
    configs/colocate_qwen3_8b.yaml \
    training.train_frac=0.50 \
    training.infer_frac=0.40
```

Inputs the example pulls together:

- [`configs/colocate_qwen3_8b.yaml`](../../configs/colocate_qwen3_8b.yaml)
  — colocate-specific config; only the four colocate fields differ from
  `configs/sglang_qwen3_8b.yaml`.
- [`examples/colocate-qwen3-8b-1node/run.sh`](../../examples/colocate-qwen3-8b-1node/run.sh)
  — sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `CUDA_VISIBLE_DEVICES=0,1,2,3` by default, pins
  `inference_num_gpus_per_engine=1` and `tp_size=1`, then calls
  `python -m torchspec.train_entry`.

## Configuration reference

The four colocate-specific fields (Phase 0):

| Field | Default | Required when colocate | Description |
|---|---|---|---|
| `training.colocate_strategy` | `null` | yes (`"mps"`) | Set to `"mps"` to enable MPS-based colocate. |
| `training.transfer_mode` | `"mooncake"` | yes (`"nccl"`) | Set to `"nccl"` for the colocate union-world data plane. ⚠️ The `"nccl"` value name is historical — the actual hidden-state transport is CUDA IPC (default) or gloo CPU-staging, not NCCL P2P; see *Hidden-state transport*. |
| `training.train_frac` | `null` | yes | Trainer per-process memory fraction, `(0, 1)`. |
| `training.infer_frac` | `null` | yes | Engine `mem_fraction_static`, `(0, 1)`. |

Validation rules (enforced by `torchspec.colocate.config.validate_colocate_config`,
called from `train_entry.parse_config`):

1. Only two combinations are accepted:
   - `colocate_strategy=null` + `transfer_mode="mooncake"` (default disaggregated path).
   - `colocate_strategy="mps"` + `transfer_mode="nccl"` (this guide).
2. `train_frac, infer_frac ∈ (0, 1)` and `train_frac + infer_frac + 0.10 ≤ 1.0`.
3. `engine_count × engine_tp_size == training_world_size`. With the
   colocate layout that means `inference_num_gpus == training_num_gpus_per_node`
   and `inference_num_gpus_per_engine == 1`.

Stray-field guard: setting `train_frac` / `infer_frac` without
`colocate_strategy=mps` errors out rather than silently no-oping.

## What changes inside the run

Compared to the disaggregated path:

1. **Placement** — both trainer and engine actor groups bind to the
   *same* Ray placement group; bundle `i` is the (trainer rank `i`,
   engine rank `i`) pair on a single physical GPU. Each actor claims
   `num_gpus = train_frac` (resp. `infer_frac`) instead of `1.0`.
2. **MPS daemon** — driver-side `setup_for_colocate` starts
   `nvidia-cuda-mps-control -d` if it isn't running, exports
   `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY` into both actor
   groups' `runtime_env`, and registers an `atexit` hook to `quit` the
   daemon on driver shutdown (Phase 6).
3. **Distributed init** — `TrainerActor.init` calls `init_union_world`
   on `master_port + 5000` (offset to avoid colliding with FSDP's own
   range) instead of `dist.init_process_group`. The trainer's
   `world_size` / `rank` views are remapped to the trainer-only
   `[0, N)` subgroup; FSDP arithmetic stays in that space. The handle
   is forwarded to `Trainer` via `set_union_world`.
4. **Data fetcher** — `Trainer.set_train_queue` constructs a
   `ColocateDataFetcher` (backed by `NcclMultiTensorFetcher`) instead
   of `MooncakeDataFetcher`. The struct shape downstream of the fetcher
   is identical, so `Eagle3Trainer._train_step` is unchanged.
5. **Engine init** — `SglEngine.init` exports
   `TORCHSPEC_COLOCATE_TRANSFER_MODE=nccl` and the paired trainer
   global rank into the engine-process env, sets
   `enable_spec_training_mooncake=False`, and overrides
   `mem_fraction_static := infer_frac`. The upstream sglang patch reads
   these env vars and re-routes its spec-training callback to
   `NcclHiddenStatesConnector` instead of the Mooncake KV connector.
6. **Controller** — `setup_colocate_training_with_engines` is used in
   place of `setup_async_training_with_engines`. The
   `AsyncInferenceManager` and Mooncake master are not started; the
   step loop is strictly serialised (engine forwards → hidden-state
   transfer → trainer recv → fwd/bwd). The synchronous loop body is the
   one piece that's gated on the upstream sglang patch — see
   [Known limitations](#known-limitations) below.

## Validation hooks

While the upstream sglang patch is in flight, the TorchSpec side is
exercised by these Modal smoke tests (`scripts/modal/modal_colocate_smoke.py`,
`--env sandbox`):

| Phase | Modal entrypoint | What it proves |
|---|---|---|
| 0 | `pytest tests/colocate/test_phase0_validation.py` (local, no GPU) | flag combinations + memory math |
| 1 | `phase1_placement` (4×H100) | both actor groups land on the same GPUs, MPS env propagates |
| 2 | `phase2_union_world` (8×H100) | `2N`-rank NCCL bootstrap + FSDP/gloo subgroups |
| 3 | `phase3_p2p_dummy` (2×H100) | 100-iter byte-equal P2P + clean shape-mismatch error |
| 4 | `phase4_multi_tensor` (2×H100) | full Mooncake-shaped 4-tensor round-trip |
| 4 | `phase4_one_step` (4×H100) | **placeholder** — runs only with the upstream sglang patch |
| 6 | `phase6_stability` (4×H100, slow) | placeholder — 1k-step VRAM flatness |
| 7 | `phase7_grad_parity` (4×H100) | placeholder — disagg vs colocate per-param grads |

Anything green in `implementation_log.md` runs without the upstream
patch. Anything still ⬜ in that doc is gated on it.

> ⚠️ *Update (2026-05-21): this Modal-smoke table is the early
> "patch-in-flight" era. The upstream patch landed; the colocate path is
> now GPU-validated end-to-end across ~12 rented-GPU sessions — see
> [`implementation_log.md`](implementation_log.md) rounds 1-9. The
> `run_smoke_host.sh --full` matrix is green under the CUDA IPC default.
> The `phase7_grad_parity` "placeholder" row is done — `test_grad_parity.py`
> covers determinism, gloo-vs-IPC parity, and colocate-vs-disagg parity.*

## Known limitations

- **Multi-node is implemented but untested at scale.** The union-world
  rank math and gloo transport are global-world-size based, and
  `mps.ensure_mps_on_all_nodes()` bootstraps the MPS daemon on every
  Ray node; `configs/colocate_qwen3_8b_2node.yaml` is the 2-node
  example. A true 2-node run has not been validated — single-node is
  the only exercised path.
- ~~**Engine `tp_size > 1` is partial.**~~ ✅ *Resolved (2026-05-21).*
  The union-world rank math (`engine_global_rank`, `build_engine_tp_ranks`)
  **and** the data plane — partitioning each step's requests across an
  engine's TP ranks — are complete and GPU-validated (`engine_tp_size=2`
  and 2-engine fan-out both pass; implementation_log rounds 2-5).
  `inference_num_gpus_per_engine=1` is no longer required.
- **sglang only.** No vLLM colocate path; nothing in
  `mooncake_hidden_states_connector.py` (vLLM KV connector) is
  affected.
- **No async pipelining.** The colocate step loop is strictly
  synchronous. Async + colocate is explicitly Phase ∞ in
  [`implementation.md`](implementation.md).
- **No `eval` parity yet.** `set_eval_queue` reuses the colocate fetcher
  but the eval step driver is still in flight.
- **`USP` (unified sequence parallel) is not supported under colocate.**
  Combining USP with the union-world FSDP subgroup is left as future
  work; `TrainerActor.init` errors out fast if both flags are set.

### Hidden-state transport (CUDA IPC default, gloo opt-out)

The engine→trainer hidden-state plane defaults to the **CUDA IPC**
zero-copy transport: the engine exports a CUDA IPC handle per tensor
and the trainer maps that memory directly, doing a single on-device
D→D copy with no host round-trip. (NCCL cannot be used here at all — it
refuses a communicator with two ranks on one physical GPU.)

Set **`TORCHSPEC_COLOCATE_IPC=0`** to fall back to the **gloo
CPU-staged** transport (engine D→H copy, gloo ship, trainer H→D copy —
two PCIe-class copies per tensor per step).

CUDA IPC needs plain `cudaMalloc` memory and **fails on
`expandable_segments:True`**, so while IPC is on (the default) colocate
does **not** inject `expandable_segments`; only the gloo fallback does.
On a host where IPC is genuinely unusable the connector fails fast at
construction with an actionable message — set `TORCHSPEC_COLOCATE_IPC=0`
to use the gloo transport.

## Troubleshooting

**Trainer comes up but the first step hangs.**
The most common cause is a missing/stale upstream sglang patch — the
engine never reaches `NcclHiddenStatesConnector.send`, so the trainer's
`recv_step` blocks on `dist.batch_isend_irecv`. Verify that
`TORCHSPEC_COLOCATE_TRANSFER_MODE` and
`TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK` are visible inside the engine
subprocess (`ps eww` on the engine PID, or log them from inside the
patched callback). If they're set but the patch didn't fire, re-check
the patch contract in [`sglang_patch.md`](sglang_patch.md).

**OOM on first step.**
`train_frac + infer_frac` is too aggressive. Drop both to `0.40 / 0.40`
and re-run. The `+ 0.10` headroom is for NCCL workspace +
driver/runtime + Python; don't try to squeeze it.

**`nvidia-smi` shows two unrelated PIDs per GPU but no MPS context.**
The MPS daemon didn't start (or didn't propagate its env vars). Check
the driver-side log line `setup_for_colocate: started MPS daemon …`;
if it's missing, look for `nvidia-cuda-mps-control` in `$PATH`.

**`P2P/CUMEM` channels show as `via PCIe` instead of on-device.**
That means NCCL didn't pick the on-device transport. Confirm
`device_id=` is being passed to `init_process_group` inside
`init_union_world` (it is by default — Phase 3 lesson). If you
wrap-init from outside the helper, you need to pass it yourself.

**MPS daemon left behind after a crash.**
Run `nvidia-cuda-mps-control` interactively and type `quit`. The
driver-side `atexit` hook (Phase 6) handles the clean-shutdown case;
crashes naturally bypass it.

## Where the code lives (quick map)

| Concern | File |
|---|---|
| Config + validation | [`torchspec/colocate/config.py`](../../torchspec/colocate/config.py) |
| MPS daemon lifecycle | [`torchspec/colocate/mps.py`](../../torchspec/colocate/mps.py) |
| Union NCCL world bootstrap | [`torchspec/colocate/world.py`](../../torchspec/colocate/world.py) |
| Placement (1:1 pairing) | [`torchspec/ray/placement_group.py`](../../torchspec/ray/placement_group.py) |
| Trainer-side P2P fetcher | [`torchspec/training/nccl_data_fetcher.py`](../../torchspec/training/nccl_data_fetcher.py) |
| Trainer DataFetcher swap | [`torchspec/training/data_fetcher.py`](../../torchspec/training/data_fetcher.py) (`ColocateDataFetcher`) |
| Engine-side P2P sender | [`torchspec/inference/engine/nccl_hidden_states_connector.py`](../../torchspec/inference/engine/nccl_hidden_states_connector.py) |
| TrainerActor wiring | [`torchspec/training/trainer_actor.py`](../../torchspec/training/trainer_actor.py) |
| Engine wiring | [`torchspec/inference/engine/sgl_engine.py`](../../torchspec/inference/engine/sgl_engine.py) |
| Controller setup | [`torchspec/controller/setup.py`](../../torchspec/controller/setup.py) (`setup_colocate_training_with_engines`) |
| Driver branch | [`torchspec/train_entry.py`](../../torchspec/train_entry.py) |
| Tests | [`tests/colocate/`](../../tests/colocate/) |
| Modal smoke | [`scripts/modal/modal_colocate_smoke.py`](../../scripts/modal/modal_colocate_smoke.py) |
| Example config | [`configs/colocate_qwen3_8b.yaml`](../../configs/colocate_qwen3_8b.yaml) |
| Example run script | [`examples/colocate-qwen3-8b-1node/run.sh`](../../examples/colocate-qwen3-8b-1node/run.sh) |

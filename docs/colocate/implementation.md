# Colocate Mode — Implementation Plan

> Scope: implement the colocate (training + inference on the same GPU) mode
> described in [Issue #81](https://github.com/lightseekorg/TorchSpec/issues/81).
>
> Prerequisite: read [`knowledge.md`](knowledge.md) first. This doc assumes
> you already understand MPS, fractional Ray bundles, NCCL union worlds, and
> how the disaggregated baseline works today.

> ⚠️ **This is the original plan — superseded in places. Read with
> [`implementation_log.md`](implementation_log.md).** Cross-check, updated
> 2026-05-21:
> - **Phase 3's "NCCL P2P data plane" is not what shipped.** NCCL hard-rejects
>   a communicator with two ranks on one physical GPU ("Duplicate GPU
>   detected"), so same-GPU NCCL P2P is impossible. The shipped hidden-state
>   transport is **CUDA IPC zero-copy (default)** with **gloo CPU-staging**
>   as the fallback, both over a gloo `meta_group`. See implementation_log
>   rounds 1 (the NCCL correction), 7 (CUDA IPC made default), 9 (the IPC
>   probe fix), and [`transport_benchmark.md`](transport_benchmark.md).
> - **`expandable_segments`** is wanted only by the gloo fallback; the CUDA
>   IPC default actively disables it (IPC needs plain `cudaMalloc` memory).
> - The phase plan completed (Phases 0-8) plus follow-up rounds 1-10; the
>   `--full` matrix is GPU-green. `implementation_log.md` is the source of
>   truth for what actually happened. Original text below is kept for the
>   design rationale and flagged inline.
> - **Transport optimization** was investigated separately —
>   [`transport_optimization.md`](transport_optimization.md): no
>   hand-written C++/CUDA/Triton kernel is needed (the path is a
>   bandwidth-bound D→D copy plus driver-API calls); the worthwhile
>   headroom is protocol-level (`ipc-pipe` ack pipelining — 3.9× on the
>   engine-`send()` stall) and **low-priority**, since the transport is
>   only ~1 % of a colocate step. Round 10 in the log.

The plan is **phased**: each phase is independently runnable and testable. Do
not skip ahead — Phase 3 (the data plane) is far easier to debug if Phases 1
and 2 have been validated standalone first.

---

## Guiding principles

1. **Ship the baseline behaviour unchanged.** Every change must be gated behind
   a new flag (`colocate_strategy=mps` + `transfer_mode=nccl`). The default
   path stays on Mooncake; existing examples and CI keep passing.
2. **One concept per phase.** Each phase introduces exactly one new mechanism
   (placement, union world, NCCL transfer, controller trim). When a bug shows
   up, you know which mechanism owns it.
3. **No async, no buffering.** Strictly serialised step. Async + colocate is
   a Phase ∞ optimisation; do not let it leak into the baseline.
4. **sglang only.** vLLM colocate is out of scope (issue says so explicitly).
   Mooncake's `vllm_engine.py` and `mooncake_hidden_states_connector.py` are
   untouched.

---

## Configuration model (introduced in Phase 0, used throughout)

We add two new flat args (consumed via `getattr(args, ..., default)` like the
rest of the codebase):

| Arg | Default | Values | Meaning |
|---|---|---|---|
| `colocate_strategy` | `null` | `null`, `"mps"` | Whether to colocate trainer + engine. `null` = today's behaviour. |
| `transfer_mode` | `"mooncake"` | `"mooncake"`, `"nccl"` | How hidden states cross the engine→trainer boundary. |
| `train_frac` | `null` | float in `(0, 1)` | Trainer's `set_per_process_memory_fraction` value. Required when colocate. |
| `infer_frac` | `null` | float in `(0, 1)` | Engine's `mem_fraction_static`. Required when colocate. |

**Validation** (added to `train_entry.py`):

- If `colocate_strategy=mps` then `transfer_mode` must be `nccl`. (Mooncake
  with colocate is supported by the existing partial code path but provides
  no benefit; we won't bother.)
- `train_frac + infer_frac + 0.10 <= 1.0`.
- `engine_count × engine_tp_size == training_world_size`.

These are the only two combinations we support:

| `colocate_strategy` | `transfer_mode` | What it does |
|---|---|---|
| `null` (default) | `mooncake` | Today's disaggregated path. |
| `mps` | `nccl` | New colocate path. |

Other combinations: error at startup.

---

## Phase 0 — Configuration plumbing & feature flag

**Goal.** Make the new flags exist, parse them, validate them. No behaviour
change.

**Files**

- `torchspec/config/train_config.py` — add the four new fields.
- `torchspec/train_entry.py` — add the validation block.

**Done when**

- `python -m torchspec.train_entry --config <existing config>` still runs.
- A test config with `colocate_strategy=mps, transfer_mode=mooncake` errors
  out with a clear message.
- A test config with `train_frac=0.6, infer_frac=0.5` errors out (sum > 1).

**Test plan**

- Unit test for the validation function (no Ray, no GPUs needed).

---

## Phase 1 — Placement: 1:1 bundle pairing + MPS env

**Goal.** When `colocate_strategy=mps`, every (trainer rank, engine rank) pair
lands on the **same** Ray bundle, and both processes are launched with MPS
client env vars set.

**Sub-tasks**

1. **MPS daemon lifecycle.** Add a small driver-side helper (e.g.
   `torchspec/colocate/mps.py`) that:
   - Checks if `nvidia-cuda-mps-control` is already running on each node (via
     a per-node `InfoActor`-style probe).
   - If not, runs `nvidia-cuda-mps-control -d`.
   - Records cleanup hook to `quit` it at shutdown (best-effort).
   - Returns the env vars that clients need:
     ```python
     {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
      "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-log"}
     ```

2. **Placement group invariant.** In
   [`torchspec/ray/placement_group.py`](../../torchspec/ray/placement_group.py)
   extend the existing `if args.colocate:` branch:
   - Size = `N = world_size`.
   - Both `pgs["training"]` and `pgs["inference"]` keys point at the same PG.
   - Bundle ordering preserved (the existing IP+GPU sort already does this) so
     bundle index `i` ↔ trainer rank `i` ↔ engine rank `i`.

3. **Fractional GPU claim.**
   - In `RayTrainGroup._allocate_gpus_for_training`
     ([torchspec/ray/train_group.py](../../torchspec/ray/train_group.py)):
     change `num_gpus_per_actor` from `1` to `train_frac` when colocate.
   - In `_prepare_sgl_engines`
     ([torchspec/inference/factory.py](../../torchspec/inference/factory.py)):
     change the engine's `num_gpus=0.2` placeholder to `infer_frac` when
     colocate.

4. **Env var injection.** Both `RayTrainGroup` and `_prepare_sgl_engines`
   should merge the MPS env vars + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   into their actor `runtime_env`.

**Files**

- `torchspec/ray/placement_group.py` — extend colocate branch with strategy=mps.
- `torchspec/ray/train_group.py` — fractional `num_gpus_per_actor`, MPS env.
- `torchspec/inference/factory.py` — fractional `num_gpus`, MPS env, same bundle index.
- `torchspec/colocate/mps.py` (new) — MPS lifecycle helper.
- `torchspec/colocate/__init__.py` (new).

**Done when**

- On a 1-node 4-GPU box with `colocate_strategy=mps`, you can spawn 4 trainer
  actors + 4 engine actors and `nvidia-smi` shows two processes per GPU sharing
  it.
- `ray.get(trainer_i.get_node_ip.remote())` and the corresponding engine return
  the same node + GPU.
- Existing disaggregated path still works (regression test on
  `examples/qwen3-8b-single-node`).

**Test plan**

- New integration test `tests/colocate/test_placement.py`:
  - Spawn placement group with `colocate_strategy=mps, world_size=4,
    train_frac=0.45, infer_frac=0.45`.
  - Assert each bundle has both a trainer and an engine actor.
  - Assert both report the same `(node_ip, gpu_id)`.
  - Tear down, assert no zombie MPS processes.

---

## Phase 2 — Union NCCL world (no actual transfer yet)

**Goal.** Both trainer and engine processes join one `2*N`-rank NCCL world.
The trainer also constructs the FSDP-only subgroup. **No data flows yet** —
this is just bootstrap.

**Sub-tasks**

1. **Rendezvous.** Driver picks one node + one port and broadcasts to all
   `2*N` actors via Ray. Existing trainer logic already does this for the
   training-only world; generalise it.

2. **Rank assignment.** Trainers get ranks `0..N-1`, engines get `N..2N-1`.
   Add this to `TrainerActor.init` and to a new init method on `SglEngine`.

3. **`init_process_group`.** Both sides call:
   ```python
   dist.init_process_group(
       backend="nccl",
       world_size=2*N,
       rank=my_rank,
       init_method=f"tcp://{master_addr}:{master_port}",
   )
   ```
   on the engine side this is a **new** code path — today sglang manages its
   own intra-engine TP NCCL world, but we need an *additional* world for
   trainer↔engine. (Implementation note: see "sglang patch surface" below.)

4. **Subgroups.**
   - `fsdp_dp_group = dist.new_group(ranks=list(range(N)), backend="nccl")`
     — called on **all** `2*N` ranks (collective).
   - `meta_group = dist.new_group(ranks=list(range(2*N)), backend="gloo")`
     — for CPU-side step metadata broadcast.

5. **FSDP rewires.** `Trainer._setup_device_mesh` currently uses the global
   world. In colocate mode, build the device mesh off `fsdp_dp_group` instead.

**Files**

- `torchspec/training/trainer_actor.py` — colocate-aware `init`.
- `torchspec/training/trainer.py` — colocate-aware `_setup_device_mesh`.
- `torchspec/inference/engine/sgl_engine.py` — colocate-aware init that
  creates the second NCCL world.
- `torchspec/colocate/world.py` (new) — union-world bootstrap helper shared
  by both sides.

**sglang patch surface.** sglang internally calls
`dist.init_process_group` on its own world. We need to either (a) ensure that
call uses a dedicated subgroup tag, or (b) initialise *our* union world before
sglang and pass sglang an explicit `init_method` that doesn't conflict. Both
are doable but require a small patch in `patches/_sglang/`. Investigate this
in the first hour of Phase 2 — it may pull the schedule.

**Done when**

- A 1-node 4-GPU smoke test: spawn 4 trainers + 4 engines, all ranks call
  `dist.barrier()` on the union world successfully. FSDP-side
  `dist.barrier(group=fsdp_dp_group)` also passes.
- Engine still serves a `generate()` call (sglang's own NCCL world is
  untouched).

**Test plan**

- `tests/colocate/test_union_world.py`:
  - Spawn 4+4 actors. Each actor calls `dist.barrier()` and reports back.
  - Trainer actor calls `dist.barrier(group=fsdp_dp_group)` — should pass with
    only 4 ranks blocking.
  - Engine actor calls `dist.barrier(group=fsdp_dp_group)` — should
    immediately return (engine is not in the group).
  - Engine calls `engine.generate(prompt)` — should still produce output.

---

## Phase 3 — NCCL P2P data plane (smoke test on dummy tensors)

> ⚠️ **Superseded (see top banner).** Same-GPU NCCL P2P is impossible —
> NCCL rejects two ranks on one physical GPU. The shipped data plane is
> **CUDA IPC (default)** / **gloo CPU-staging (fallback)** over a gloo
> `meta_group`, *not* NCCL `send`/`recv` on the union world. The
> `nccl_data_fetcher.py` / `nccl_hidden_states_connector.py` module names
> below are historical; the NCCL batched path they still contain is used
> only by the separate-GPU Phase-3 dummy test. See implementation_log
> rounds 1, 7, 9 and `transport_benchmark.md`.

**Goal.** Engine sends a fixed dummy tensor, trainer receives it, contents
match. No model code involved.

**Sub-tasks**

1. **Trainer side.** New module `torchspec/training/nccl_data_fetcher.py`:
   - Pre-allocates a recv buffer sized for `[B_eng/TP, S, H]`, dtype bf16, on
     the local GPU.
   - Each step: `dist.recv(buffer, src=engine_rank)`, optionally on a
     dedicated transfer CUDA stream.
   - Yields the buffer (or a clone if downstream consumers may stomp it).

2. **Engine side.** Add a method `SglEngine.transfer_dummy(shape)`:
   - Allocates a deterministic tensor on its GPU
     (`torch.arange(...).reshape(shape).to(bf16)`).
   - Calls `dist.send(tensor, dst=trainer_rank)`.

3. **Driver test loop.**
   - Pick a fixed shape `[2, 8, 4096]`.
   - For 100 iterations: each engine calls `transfer_dummy(shape)`, each
     trainer pulls one buffer from its fetcher and asserts byte equality with
     the deterministic source.

**Files**

- `torchspec/training/nccl_data_fetcher.py` (new).
- `torchspec/inference/engine/sgl_engine.py` — `transfer_dummy` method.
- `torchspec/training/trainer.py` — colocate-mode `set_train_queue` shortcut
  that wires up `NcclDataFetcher` instead of `MooncakeDataFetcher`.

**Done when**

- `tests/colocate/test_p2p_dummy.py` runs 100 iterations, asserts byte
  equality every iteration, with `train_frac=0.45, infer_frac=0.45` on a
  4-GPU box.
- `nvidia-smi` shows zero PCIe / NVLink traffic during the test (NCCL chose
  the on-device path).

**Test plan**

- See above. Add a deliberate corruption test: engine sends shape A, trainer
  expects shape B → must error cleanly, not deadlock.

---

## Phase 4 — Real hidden-state hook in sglang

**Goal.** Replace `transfer_dummy` with the actual post-target-forward hidden
state, sent from inside sglang's spec-training mode.

**Sub-tasks**

1. **sglang patch.** Inside `patches/_sglang/`, find the spec-training hidden
   state callback (where today it writes to Mooncake via
   `mooncake_hidden_states_connector`). Add a sibling callback path
   `nccl_hidden_states_connector.py` that:
   - Receives `hidden_states ∈ [B_eng, S, H]`.
   - Local-chunks: `shard_i = hidden_states[i*B_eng/TP : (i+1)*B_eng/TP]`
     where `i = engine.tp_rank`.
   - `dist.send(shard_i, dst=trainer_rank_i)` on the union world.

2. **Aux layers + last_hidden_states.** Eagle3 needs more than just the final
   hidden state; the connector emits a list of tensors. Send each in sequence
   on the same group, with consistent ordering.

3. **Trainer recv side.** Update `NcclDataFetcher` to receive the matching
   list of tensors and assemble them into the existing batch dict shape
   (matching what `MooncakeDataFetcher` produces) so downstream
   `Eagle3Trainer._train_step` doesn't have to know which fetcher it's using.

4. **Connector selection.** In sglang's engine init, select Mooncake or NCCL
   connector based on the `transfer_mode` arg.

**Files**

- `patches/_sglang/.../nccl_hidden_states_connector.py` (new) — mirror of the
  Mooncake one.
- `torchspec/inference/engine/sgl_engine.py` — propagate `transfer_mode` and
  trainer-rank table into sglang at init.
- `torchspec/training/nccl_data_fetcher.py` — generalise to multi-tensor.

**Done when**

- A 1-node 4-GPU run: 1 engine × TP=4 + 4 trainer ranks. One training step
  end-to-end. Loss is finite and non-zero.

**Test plan**

- `tests/colocate/test_one_step.py`: drive one training step, assert loss is
  finite, assert no Mooncake calls happened (mock the Mooncake store and
  fail the test if it gets touched).

---

## Phase 5 — Controller trim & loop integration

**Goal.** When `transfer_mode=nccl`, drop the Mooncake-specific plumbing in
the controller. The controller still owns prompt dispatch and step
sequencing, but doesn't push tensor metadata.

**Sub-tasks**

1. **`TrainSample` slim variant.** In
   [`torchspec/training/data_fetcher.py`](../../torchspec/training/data_fetcher.py):
   `TrainSample(mooncake_key, tensor_shapes, tensor_dtypes, ...)` becomes
   `TrainSample(step_id, seq_len, loss_mask, input_ids)` in the colocate
   branch. The struct already exists; add a sibling `ColocateSample` or use a
   union type.

2. **No `SamplePool`.** `AsyncInferenceManager`'s backpressure machinery
   isn't needed (engine is rate-limited by trainer's recv). Don't instantiate
   it in colocate mode.

3. **No `Mooncake master`.** In `train_entry.py`, skip
   `launch_mooncake_master` and `build_mooncake_config` when
   `transfer_mode=nccl`.

4. **Loop simplification.** `controller/loop.py` already orchestrates per-step
   dispatch. In colocate mode, the loop is:
   ```
   for step in steps:
       controller.broadcast_meta(step)  # via gloo group
       engines.generate_one_step()      # blocks until P2P send completes
       trainers.train_one_step()        # blocks until P2P recv + fwd/bwd
   ```
   Most of this exists; the change is removing the
   `try_dispatch_batch` + `SamplePool` indirection.

**Files**

- `torchspec/controller/training_controller.py` — colocate branch.
- `torchspec/controller/inference_manager.py` — skip in colocate mode.
- `torchspec/controller/loop.py` — synchronous step loop variant.
- `torchspec/controller/setup.py` — `setup_colocate_training_with_engines`
  alongside the existing `setup_async_training_with_engines`.
- `torchspec/train_entry.py` — branch on `transfer_mode`.
- `torchspec/training/data_fetcher.py` — `TrainSample` variants.

**Done when**

- A clean colocate run leaves no Mooncake processes alive (`pgrep
  mooncake_master` returns nothing).
- The async ramp-up (prompt buffer warming) is gone; first training step
  starts within seconds of init.

**Test plan**

- Modify `tests/colocate/test_one_step.py` to assert no Mooncake imports were
  hit (use `sys.modules` introspection or a guard module).

---

## Phase 6 — Memory caps, MPS hygiene, stability

**Goal.** Run 1000 steps without VRAM growth, with both processes capped.

**Sub-tasks**

1. **Trainer init order.** Make sure trainer's actor init runs and warms its
   allocator (one dummy fwd/bwd) **before** sglang starts. Currently
   `_prepare_sgl_engines` and `RayTrainGroup` run roughly in parallel; in
   colocate mode, gate the engine's `init` on the trainer's
   `set_per_process_memory_fraction` having been applied.

2. **`expandable_segments`** propagated to both sides via runtime_env (already
   in Phase 1, double-check here).

3. **MPS thread percentage knob.** Optional: if there's contention, expose
   `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` per role. Off by default.

4. **`torch.cuda.memory_stats()` in profiler.** Add peak alloc to the perf
   metrics dump.

**Files**

- `torchspec/colocate/world.py` — init ordering fence.
- `torchspec/training/trainer_actor.py` — pre-warm hook.
- `torchspec/utils/profiling.py` — peak alloc metric.

**Done when**

- 1000-step stability run with `dflash_trainer` config:
  `peak_alloc(step=10) ≈ peak_alloc(step=999)` within 1%.
- No process-side OOM. No system-side hang.

**Test plan**

- New `tests/colocate/test_stability.py` (slow, marked `@pytest.mark.slow`):
  1000 steps, log `memory_stats` every 100 steps, assert flat.

---

## Phase 7 — Numeric parity & convergence

**Goal.** Confirm the colocate path is bit-comparable to the disaggregated
baseline.

**Sub-tasks**

1. **Per-layer gradient parity.** Same prompts, same seed:
   - Run one step on disaggregated mode → dump `extract_gradients(model)`.
   - Run one step on colocate mode → dump same.
   - `torch.allclose(g_disagg, g_colocate, atol=1e-6, rtol=0)` per parameter.
     (NCCL is bit-deterministic given identical reduction order; we expect
     exact match modulo floating-point reduce ordering, which we don't
     change.)

2. **Convergence curve.** 1k steps on `qwen3-8b-single-node` with both modes,
   plot loss curves. They should overlap to within 1–2% per-step.

3. **Eval stability.** Cached eval batches → eval loss should match between
   modes within tokenizer-deterministic noise.

**Files** (new tests only)

- `tests/colocate/test_grad_parity.py`.
- `tests/colocate/test_convergence.py` (slow).

**Done when**

- Both tests green.
- Plot of loss curves in PR description.

---

## Phase 8 — Documentation & examples

- Update [`docs/ray.md`](../ray.md) with a colocate placement table row.
- New `docs/colocate/usage.md` with a runnable config example.
- New `examples/colocate-qwen3-8b-1node/` mirroring the qwen3-8b example with
  `colocate_strategy=mps` set.

---

## Out-of-scope (don't let scope creep in)

- vLLM colocate path. We touch only sglang. Mooncake's
  `vllm_engine.py` and `mooncake_hidden_states_connector.py` are untouched.
- Async pipelining / double buffering between engine and trainer. Strictly
  step-serialised handoff.
- Mixed colocate + disaggregated in the same job.
- Reduce-scatter optimisation (skipping engine's TP all-reduce, fusing with
  scatter). Future work; documented as a follow-up issue.

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| sglang patch is more invasive than expected (Phase 2/4) | High | Spike on this on day 1. If it requires upstream-PR-grade changes, we may want to fork the spec-training callback path. |
| Allocator fragmentation under MPS exceeds `expandable_segments` mitigation | Medium | Phase 6 stability test will catch this. Fallback: tune `train_frac` lower. |
| FSDP all-gather and our P2P serialise (no overlap) | Low | Dedicated transfer CUDA stream (Phase 3). Worst case: small throughput hit, not a correctness issue. |
| Straggler engine blocks paired trainer on `dist.recv` | Low | Already FSDP-bottlenecked. Add timeout-skip policy if it becomes an issue in practice. |
| MPS scheduling fairness under load | Low | Expose `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (Phase 6); off by default. |
| MPS daemon zombie processes after crashes | Low | Best-effort `quit` on driver shutdown + per-node health check on next startup. |

---

## Milestones (suggested ordering for PRs)

| PR | Phases | Reviewable size |
|---|---|---|
| `colocate-1: config + flag` | Phase 0 | ~100 LOC |
| `colocate-2: placement + MPS` | Phase 1 | ~300 LOC |
| `colocate-3: union NCCL world` | Phase 2 | ~200 LOC + sglang patch |
| `colocate-4: P2P smoke test` | Phase 3 | ~250 LOC + tests |
| `colocate-5: real hidden-state hook` | Phase 4 | ~400 LOC (most of the sglang patch) |
| `colocate-6: controller trim` | Phase 5 | ~300 LOC |
| `colocate-7: stability + parity` | Phase 6 + 7 | mostly tests |
| `colocate-8: docs + example` | Phase 8 | docs only |

Each phase is independently mergeable behind the feature flag, so we can land
them as separate PRs without breaking main.

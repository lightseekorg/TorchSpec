# Upstream sglang patch surface for the colocate (NCCL) path

> Phase 4 of [`implementation.md`](implementation.md) requires a small
> set of changes inside sglang itself. This doc enumerates the exact
> patch surface so a human submitter can drive the upstream PR (or, in
> the meantime, maintain a fork).
>
> **The patch lives in this repo under
> `patches/sglang/<version>/colocate.patch`; `v0.5.10.post1` is the
> current default.** It is applied on top of the existing `sglang.patch`
> (the disagg patch). The Modal smoke image
> (`scripts/modal/modal_colocate_smoke.py`) applies both in order; for a
> local checkout, `./tools/apply_sglang_patch.sh --colocate <sglang-repo>`
> does the same. The pseudocode in the rest of this document still
> describes what the patch does and serves as the upstream-PR spec —
> see `colocate.patch` for the actual diff.

> **Version status.** `patches/sglang/v0.5.10.post1/colocate.patch` is
> the **default, fully GPU-validated** colocate patch — as of the
> 2026-05-21 cutover, `apply_sglang_patch.sh --colocate`,
> `run_smoke_host.sh`, and the Modal smoke all default to it.
> `patches/sglang/v0.5.8.post1/colocate.patch` is retained as a
> fallback (`SGLANG_PATCH_VERSION=v0.5.8.post1` selects it) but is no
> longer the maintained target.
>
> The v0.5.10 forward-port reworked `parallel_state.py` — v0.5.10
> restructured `initialize_model_parallel` (new `_ATTN_CP` / `_ATTN_TP`
> / MoE-DP groups), so the per-site rank branches became a uniform
> engine-logical-world + offset-shift remap; the `dp_attention.py` hunk
> is dropped because v0.5.10 moved that group into
> `initialize_model_parallel`.
>
> **GPU validation (2026-05-21, RunPod H100).** The full
> `run_smoke_host.sh --full` matrix — all 13 tests across 9 files —
> **passes on 4×H100** with `SGLANG_PATCH_VERSION=v0.5.10.post1`:
> tp_size=1, engine_tp_size=2, 4-engine Qwen3-8B end-to-end, grad
> parity (smoke / determinism / full), checkpoint save+resume, CUDA
> IPC, multi-engine fan-out, 200-step stability, and convergence.
> Still unexercised: pipeline parallelism (`pp_size>1`, blocked by an
> explicit guard). One TorchSpec-side fix outside this patch was needed
> for the matrix — the `_init_rope` handling of transformers'
> `rope_type="default"` (committed separately). See
> [Testing the v0.5.10.post1 forward-port](#testing-the-v0510post1-forward-port).

## Testing the v0.5.10.post1 forward-port

> **Modal cannot run this.** The colocate path needs NVIDIA MPS, and
> Modal sandbox runs containers under gVisor, whose nvproxy
> [does not implement MPS multiplexing](https://github.com/google/gvisor/blob/master/g3doc/proposals/nvidia_driver_proxy.md).
> On Modal the MPS-dependent tests (`phase4_one_step`, `phase6`,
> `phase7`) `pytest.skip` instead of running — see
> [`implementation_log.md`](implementation_log.md)
> §"Cheap-host workflow for MPS-required validation". The patch must be
> tested on a host that passes `--ipc=host` to its container: Vast.ai,
> RunPod *Interactive* Pod, Lambda, Hyperstack, or bare-metal.

**Cheap-host recipe (~$2, ~25 min).** Rent a **1×H100** instance (sm90
— L40S / A6000 / 4090 are rejected by the bundled `sgl_kernel` wheel,
see [`cheap_host_test_plan.md`](cheap_host_test_plan.md)) with
`--ipc=host`, then:

```bash
git clone https://github.com/zhubohao911/TorchSpec.git
cd TorchSpec
git checkout feature/colocate-training-inference

# Point the smoke runner at the v0.5.10.post1 patch dir + base commit.
SGLANG_PATCH_VERSION=v0.5.10.post1 \
SGLANG_COMMIT=94f03a39dbd39edfc2b118b5357bbbadaaa9ad28 \
    bash scripts/colocate/run_smoke_host.sh
```

`run_smoke_host.sh` clones sglang at `SGLANG_COMMIT`, applies
`patches/sglang/v0.5.10.post1/{sglang,colocate}.patch`, installs
torchspec + sglang, and runs `tests/colocate/test_colocate_tiny.py`
(Qwen3-0.6B; 1 GPU shared by 1 trainer + 1 engine over MPS) — which
exercises the full colocate sync loop including the sglang patch's
hidden-state hook.

**Success:** the script exits `0`, the pytest summary shows the tiny
test `PASSED` (not `SKIPPED`), and `colocate-smoke-report.txt` has a
decreasing `[colocate_loop] step=…` loss progression. **Failure
signature:** a wrong distributed-wiring patch **hangs on the first P2P
recv** (see [§Verification](#verification)); the report's pytest tail
captures the hang.

For the full 4-GPU suite (Phase 4 / 6 / 7, Qwen3-8B) use a 4×H100
`--ipc=host` host and add `--full` — same two env vars.

## Motivation

In disaggregated mode, sglang's spec_training callback writes hidden
states to a Mooncake KV store keyed by a UUID, then the trainer reads
from Mooncake. In colocate mode (`transfer_mode=nccl`) the trainer +
engine ranks share one **union NCCL world** of size `2N` (N trainers
+ N engine TP workers, paired by rank). The engine writes hidden states
**directly** to its paired trainer rank via `dist.batch_isend_irecv` on
that union world — no shared store, no serialisation overhead.

The TorchSpec side of the wire is already in this repo:

- Engine-side sender:
  [`torchspec/inference/engine/nccl_hidden_states_connector.py`](../../torchspec/inference/engine/nccl_hidden_states_connector.py)
  — `NcclHiddenStatesConnector(dst_global_rank).send(tensors)`.
- Trainer-side receiver:
  [`torchspec/training/nccl_data_fetcher.py`](../../torchspec/training/nccl_data_fetcher.py)
  — `NcclMultiTensorFetcher(src_global_rank, device).recv_step(specs)`.
- Union-world bootstrap:
  [`torchspec/colocate/world.py`](../../torchspec/colocate/world.py).

What's missing is the **engine-process side of the bootstrap**: sglang
itself must (a) skip its own `dist.init_process_group` when our union
world is already up, or (b) join the union world and re-derive its TP
group from a slice of it; and (c) route the spec_training callback to
the new `NcclHiddenStatesConnector` instead of the Mooncake writer.

## Env-var contract

The TorchSpec driver exports the following env vars before launching
sglang. Read them from inside sglang's TP scheduler subprocess:

| env var | meaning |
|---|---|
| `TORCHSPEC_COLOCATE_TRANSFER_MODE` | Set to `"nccl"` when colocate is on. Set the spec_training callback path accordingly. Empty / unset means stay on the legacy Mooncake path. |
| `TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK` | Global rank in the union world to send hidden states to. |
| `TORCHSPEC_COLOCATE_UNION_MASTER_ADDR` | Rendezvous host for `init_process_group`. |
| `TORCHSPEC_COLOCATE_UNION_MASTER_PORT` | Rendezvous port. |
| `TORCHSPEC_COLOCATE_UNION_WORLD_SIZE` | `2N` — total ranks in the union world. |
| `TORCHSPEC_COLOCATE_UNION_N_PER_ROLE` | `N` — number of trainer / engine ranks. The engine TP scheduler is at union global rank `N + sglang_tp_rank`. |
| `TORCHSPEC_COLOCATE_UNION_TIMEOUT_MIN` | `init_process_group` timeout in minutes. Use this exact value — the trainer side already booted the rendezvous and will wait this long. |
| `TORCHSPEC_COLOCATE_UNION_WORLD` | Set to `"1"` once the union world is initialised. The patch can use this as a "torch.dist already brought up" sentinel. |

## Patch points

The patch spans a handful of sglang files (see
[`colocate.patch`](../../patches/sglang/v0.5.8.post1/colocate.patch) for
the actual diff). Pseudo-paths are shown for the layout that's been
stable in sglang since ~mid-2024; they may shift slightly if the
upstream refactor changes.

### 1. Distributed init: `sglang/srt/distributed/parallel_state.py` (or equivalent)

When the scheduler subprocess boots, it normally calls
`torch.distributed.init_process_group` to bring up its TP world. In
colocate mode, the union world is the default PG; sglang should join it
instead of creating a new default.

Pseudocode:

```python
import os
import torch.distributed as dist
from datetime import timedelta

def _maybe_join_torchspec_union_world():
    if os.environ.get("TORCHSPEC_COLOCATE_TRANSFER_MODE") != "nccl":
        return False  # disaggregated path — no-op

    if dist.is_initialized():
        # Trainer's init_union_world already ran in this process —
        # nothing to do. (This branch fires when the engine and
        # trainer happen to share a Python process; not the common
        # case but possible in tests.)
        return True

    addr = os.environ["TORCHSPEC_COLOCATE_UNION_MASTER_ADDR"]
    port = int(os.environ["TORCHSPEC_COLOCATE_UNION_MASTER_PORT"])
    world_size = int(os.environ["TORCHSPEC_COLOCATE_UNION_WORLD_SIZE"])
    n_per_role = int(os.environ["TORCHSPEC_COLOCATE_UNION_N_PER_ROLE"])
    timeout = int(os.environ.get("TORCHSPEC_COLOCATE_UNION_TIMEOUT_MIN", "30"))

    # Engines occupy ranks [N, 2N). The current TP rank determines our
    # offset within the engine block.
    tp_rank = int(os.environ.get("TP_RANK", os.environ.get("RANK", "0")))
    global_rank = n_per_role + tp_rank

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        init_method=f"tcp://{addr}:{port}",
        timeout=timedelta(minutes=timeout),
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )

    # The TP group sglang would normally create with new_group is now a
    # subgroup of the 2N-rank default PG; the rank list is contiguous.
    tp_world_ranks = list(range(n_per_role, 2 * n_per_role))
    tp_group = dist.new_group(ranks=tp_world_ranks, backend="nccl")
    return True, tp_group
```

The exact integration pattern depends on how sglang's distributed init
is structured. The key invariants:

- Default PG must be the 2N-rank union world after this runs.
- sglang's TP group is `dist.new_group(ranks=range(N, 2N))` — a
  contiguous slice of the engine half of the union world.
- All trainer ranks have already joined the rendezvous via
  `init_union_world` (TorchSpec side); the engine joining is what
  unblocks them.

### 2. spec_training callback: `sglang/srt/managers/scheduler.py` (or wherever `enable_spec_training_mooncake` is consumed)

The callback today writes to `EagleMooncakeStore` keyed by `mooncake_key`.
In colocate mode, route to the NCCL connector instead. Pseudo-code:

```python
import os

def _build_hidden_states_writer():
    transfer_mode = os.environ.get("TORCHSPEC_COLOCATE_TRANSFER_MODE", "")
    if transfer_mode == "nccl":
        from torchspec.inference.engine.nccl_hidden_states_connector import (
            NcclHiddenStatesConnector,
        )
        dst = int(os.environ["TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK"])
        return NcclHiddenStatesConnector(dst_global_rank=dst)
    else:
        return _build_mooncake_writer()  # existing path
```

In the callback itself:

```python
def on_spec_training_step(hidden_states, aux_hidden_states, last_hidden_states, target_logits):
    if isinstance(writer, NcclHiddenStatesConnector):
        writer.send({
            "hidden_states": hidden_states,
            "aux_hidden_states": aux_hidden_states,
            "last_hidden_states": last_hidden_states,
            "target_logits": target_logits,
        })
    else:
        writer.put(mooncake_key, ...)  # existing Mooncake path
```

The **dict key set** must match what TorchSpec's controller ships in
`ColocateTrainSample.tensor_specs` — see
[`torchspec/training/data_fetcher.py`](../../torchspec/training/data_fetcher.py)
`class ColocateTrainSample`. Both sides walk `sorted(keys)` so insertion
order doesn't matter.

The tensors **must be contiguous and on CUDA**. The connector raises
`ValueError` otherwise.

The callback runs **only on TP rank 0** today (it's the rank that
coordinates the Mooncake write). For colocate, every TP rank participates
in the P2P send because the trainer side has one fetcher per trainer
rank (paired 1:1 with engine TP ranks). Either:

  - Move the callback to fire on every TP rank, OR
  - Do an all-gather on TP rank 0 first and then send the shards out.

The former is simpler and matches the way the trainer expects to
receive (one shard per trainer rank). The Phase-4 plan in
`implementation.md` §"sglang patch" §1 makes this explicit:
*"Local-chunks: shard_i = hidden_states[i*B_eng/TP : (i+1)*B_eng/TP]
where i = engine.tp_rank."*

### 3. (Optional) Skip the Mooncake setup completely

When `enable_spec_training_mooncake=False`, sglang's existing flag flow
already skips the Mooncake bootstrap. TorchSpec sets the flag from
[`torchspec/inference/engine/sgl_engine.py`](../../torchspec/inference/engine/sgl_engine.py)
based on `transfer_mode`. No extra patch needed here as long as the flag
is honoured.

### 4. Engine rank-offset fixes (`dp_attention.py`, `tp_worker.py`)

Two callsites in sglang assume the engine owns the whole `dist` world
(global rank == tp-local rank). Under colocate the engine sits at global
ranks `[N, 2N)`, so both need a global-rank correction. These were
prototyped as post-patch `sed`-style surgery in `run_smoke_host.sh`
during validation and are now **folded into `colocate.patch` as proper
hunks** (2026-05-20) — no out-of-band surgery step remains.

- **`layers/dp_attention.py`** — `_ATTN_TP_GROUP`'s rank list is computed
  as `range(head, head + _ATTN_TP_SIZE)`, landing in `[0, tp_size)`. For
  a `tp_size=1` engine that is `[0]`, so only engine 0 passes
  `GroupCoordinator`'s `self.rank in ranks` membership check and every
  other engine trips `assert self.cpu_group is not None`. The hunk adds a
  `_ts_offset` (this engine's own union rank via `engine_global_rank()`,
  `0` when colocate is inactive) and shifts the range by it.
- **`managers/tp_worker.py`** — the random-seed `broadcast_pyobj` call
  passes `tp_size * pp_rank + tp_rank` as the *global* rank argument.
  That equals the global rank only when the engine owns the whole world;
  under colocate the engine's tp-local rank is `0` but its global rank is
  `N`, so it wrongly takes the receiver path and `IndexError`s on the
  empty result. The hunk passes `world_group.rank` (already the global
  rank) instead — correct for both colocate and standalone.

Both files are untouched by `sglang.patch` and the other colocate hunks,
so the diffs apply cleanly stacked on either.

## Verification

After the patch lands, run the colocate smoke on an `--ipc=host` GPU
host — **not** Modal; see
[Testing the v0.5.10.post1 forward-port](#testing-the-v0510post1-forward-port)
for why and the exact command. The Phase-4 end-to-end test
(`tests/colocate/test_one_step.py`) runs on a 4×H100 box: 1 engine ×
TP=4 + 4 trainers × FSDP=4, all sharing GPUs via MPS, hidden states
moving over the union world. The plan's §Phase 4 done-criterion
("loss is finite and non-zero") is checked there.

Without the patch, that test will **hang on the first P2P recv** because
the engine's spec_training callback is still writing to a (now disabled)
Mooncake store and the trainer's `NcclMultiTensorFetcher.recv_step` is
waiting for tensors that never arrive. This hang is the diagnostic — if
you see it, the patch isn't being picked up.

## Test surface available without the patch

`tests/colocate/test_p2p_multi_tensor.py` exercises the connector +
fetcher + union-world integration **without** sglang involvement
(both sides are Ray actors that call the connector directly). Modal
entrypoint: `phase4_multi_tensor`. This is the maximal e2e check that
runs in this repo today.

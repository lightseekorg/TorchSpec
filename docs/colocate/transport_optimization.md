# Colocate hidden-state transport — kernel investigation & optimization plan

> Companion to [`transport_benchmark.md`](transport_benchmark.md) (the
> measured gloo-vs-IPC numbers) and the round-7 entry in
> [`implementation_log.md`](implementation_log.md). This doc answers two
> questions:
>
> 1. **Should we replace the PyTorch CUDA IPC transport with hand-written
>    C++/CUDA or Triton?** — investigated below; the answer is **no**.
> 2. **What are the real optimizations, and how do we benchmark them?** —
>    a concrete design + A/B benchmark plan for the pure-Python /
>    protocol-level wins.
>
> Source under discussion: [`torchspec/colocate/cuda_ipc.py`](../../torchspec/colocate/cuda_ipc.py),
> exercised by [`scripts/colocate/bench_transport.py`](../../scripts/colocate/bench_transport.py).

---

## Part 1 — Do we need C++/CUDA or Triton?

**Short answer: no.** The PyTorch implementation is already at the
hardware ceiling. There is no GPU compute kernel anywhere in this path
for a CUDA/Triton kernel to replace, and the costs that *do* dominate
are CUDA-driver-API and network-control costs that hand-written C++
cannot speed up.

### There is no kernel in the path

The entire CUDA IPC transport ([`cuda_ipc.py`](../../torchspec/colocate/cuda_ipc.py)
`ipc_send` / `ipc_recv`) is four things, **none of which is GPU device
code**:

1. `reduce_tensor` → `cudaIpcGetMemHandle` — a CUDA *driver API* call (host-side).
2. `pickle` + `dist.send/recv` of a small handle blob over gloo — a control message.
3. `rebuild_cuda_tensor` → `cudaIpcOpenMemHandle` — a CUDA *driver API* call (host-side).
4. `alias.to(device, copy=True)` — a single D→D `cudaMemcpyAsync`.

Grepping the connector ([`nccl_hidden_states_connector.py`](../../torchspec/inference/engine/nccl_hidden_states_connector.py))
and fetcher ([`nccl_data_fetcher.py`](../../torchspec/training/nccl_data_fetcher.py))
for compute (`matmul`, `cast`, `reshape`, elementwise) returns nothing.
Hidden states are shipped bf16/contiguous and consumed as-is by the
draft model. **There is nothing to fuse and nothing to compute.** A
CUDA or Triton kernel could only ever replace item 4 — the copy.

### Where the time actually goes

Per-stage breakdown from [`transport_benchmark.md`](transport_benchmark.md),
**256 MB** payload:

| Stage | Time | Is it a GPU kernel? |
|---|--:|---|
| `ipc.engine handle export` | 0.20 ms | No — `cudaIpcGetMemHandle` + pickle |
| `ipc.engine ship handles` | 0.29 ms | No — gloo TCP |
| `ipc.engine wait for ack` | 1.26 ms | No — gloo round-trip |
| `ipc.trainer handle open` | 0.52 ms | No — `cudaIpcOpenMemHandle` |
| **`ipc.trainer D->D copy`** | **0.26 ms** | **Yes — the only kernel** |

### Why the copy can't be improved

The D→D copy moves 256 MB in 0.26 ms ≈ **~1 TB/s effective**, i.e. H100
HBM3 bandwidth. The copy is purely memory-bandwidth-bound and already
saturated.

- A **custom CUDA kernel** for a contiguous copy lowers to the same
  `LDG`/`STG` stream `cudaMemcpyAsync` already uses — it cannot beat a
  bandwidth-bound copy.
- **Triton** is built for *fused* elementwise/reduction work; for a pure
  copy it emits `tl.load`/`tl.store` and lands, at best, equal — more
  likely slightly *worse* (launch + masking overhead).

The copy is 0.26 ms out of a ~1.9 ms transfer out of a training step
measured in **tens-to-hundreds of ms**. Even a zero-cost copy saves
nothing observable.

### Why C++ can't help the rest either

The biggest line item — the **ack round-trip (1.26 ms)** — is gloo TCP
latency on localhost. `cudaIpcOpenMemHandle` (0.52 ms) is a fixed CUDA
driver cost. Neither is GPU device code. You *could* write a C++ host
extension that calls `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`
directly to shave Python/pickle/storage-bookkeeping overhead — but:

- That is a **host-side driver wrapper, not a CUDA kernel or Triton**.
- The realistic saving is ~0.1–0.3 ms on a path that is already a
  non-bottleneck.
- It adds a compiled-extension build dependency (toolchain, ABI, wheels)
  to a repo where the benchmark deliberately "runs on a bare torch
  install with no `pip install`".
- Negative ROI.

### When you *would* reach for a kernel — and why colocate isn't it

A custom kernel pays off when you can **fuse** transport with compute:
copy + dtype cast, copy + layout transform, or gather/scatter. The
colocate path has none — hidden states cross the wire and enter the
draft model unchanged. The one mandatory copy (`alias.to(copy=True)`)
exists purely for **lifetime safety** (the engine reuses/frees its
sglang-owned buffers each step); removing it needs a deeper lifetime
contract, not a faster kernel — and even a fused copy+cast stays
bandwidth-bound.

### Verdict

Do not write C++/CUDA or Triton for this transport. The benchmark
already settles the performance question — CUDA IPC is **171×** faster
than gloo on the realistic 160 MB Eagle3 payload and "removes the
hidden-state transfer as a step-time factor entirely". The remaining
headroom is **protocol-level, not kernel-level** — and that is Part 2.

---

## Part 2 — The real optimizations (pure Python / protocol-level)

All wins below are protocol changes to [`cuda_ipc.py`](../../torchspec/colocate/cuda_ipc.py).
None needs a compiled extension, a CUDA kernel, or Triton.

### Cost model (the target)

For the realistic **Eagle3 160 MB** case the transfer is ~1.9 ms
end-to-end, of which the engine-visible stall (`ipc engine send`) is
~1.58 ms. Breaking the 256 MB anatomy into "fixed handshake" vs "real
work":

| Bucket | Stages | ~Time | Attackable? |
|---|---|--:|---|
| Fixed handshake | export + ship + ack-wait + open | ~2.3 ms | **yes — protocol** |
| Real data movement | D→D copy | ~0.26 ms | no — at HBM bandwidth |

Every optimization below shrinks the **fixed handshake**, which is
~90 % of the transfer and 100 % protocol overhead.

### Opt 1 — Persistent send-buffer pool + trainer mapping cache

**Attacks:** `handle export` (0.20 ms) + `handle open` (0.52 ms) —
the per-step `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` pair.

**Why they are paid every step today.** The engine's hidden states are
freshly allocated inside sglang's forward each step. With variable
`seq_len` the allocation size changes, so the caching allocator hands
back a different underlying block → a different device pointer → a
different IPC handle. The trainer sees a new handle every step → it
must call `cudaIpcOpenMemHandle` every step. PyTorch's own IPC cache
(`torch.multiprocessing.reductions.shared_cache`) holds opened storages
only by *weakref*, and `ipc_recv` does `del aliases` each step — so even
a repeated handle would miss.

**The fix is two cooperating halves:**

- **Engine side — a send-buffer pool.** Allocate a small ring of `K`
  persistent buffers (`K = 2` is enough; see Opt 2), each sized to the
  *maximum* expected `[seq_len, hidden]`. Each step the engine copies
  sglang's transient hidden states into `pool[step % K]` (a D→D copy)
  and exports the handle for that *pooled* buffer. Pool buffers have
  stable device pointers for the life of the run → their IPC handles
  never change → `reduce_tensor` args can be computed **once at startup**
  and reused. `handle export` → ~0 in steady state.

- **Trainer side — a keep-alive mapping cache.** Keep an LRU of opened
  IPC storages keyed by handle bytes, so PyTorch's `shared_cache`
  weakrefs stay alive across steps. On a repeated handle (which the pool
  now guarantees) `rebuild_cuda_tensor` skips `cudaIpcOpenMemHandle` and
  reuses the existing mapping — only the per-step view + D→D copy remain.
  `handle open` → ~0 in steady state.

**Cost it adds:** one extra D→D copy on the engine side (~0.26 ms for
256 MB) to move sglang's tensor into the pooled buffer. Net steady-state
swing: `−0.20 − 0.52 + 0.26 ≈ −0.46 ms`, *and* the IPC handshake
becomes a one-time startup cost instead of a per-step cost.

**Why the pool, not just luck:** for a *fixed* `seq_len` the caching
allocator may already reuse the same block and hand you stable handles
for free — but seq_len is variable, so this is non-deterministic. The
pool makes handle stability deterministic and, critically, is the
prerequisite for Opt 2.

**Sketch:**

```python
# engine, once at startup
pool = [torch.empty(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for _ in range(K)]
pool_args = [reduce_tensor(b)[1] for b in pool]   # IPC handles, computed once

# engine, per step
slot = step % K
dst = pool[slot][:seq_len]          # view; same base pointer every step
dst.copy_(hidden_states)            # the one added D->D copy
ship(slot, seq_len, dtype)          # tiny message: which slot + shape

# trainer, per step
storage = mapping_cache.get(handle) # cudaIpcOpenMemHandle only on miss
alias = view(storage, seq_len, dtype)
out = alias.to(device, copy=True)
```

### Opt 2 — Ack pipelining (one-step deferral) + double buffering

**Attacks:** `wait for ack` (1.26 ms) — the single largest line item,
and a pure engine stall.

**Why the ack exists.** The engine must not overwrite/free the memory
the trainer is reading from until the trainer's D→D copy has finished.
Today the engine *blocks* on that ack inside `send()`.

**The fix.** Defer the wait by one step. With Opt 1's pool sized at
`K ≥ 2`, the engine ping-pongs between two slots. At step *N*:

1. Engine copies hidden states into `pool[N % 2]`.
2. Engine ships the handle/slot message for `pool[N % 2]`.
3. Engine waits for the ack of step **N−1** (`pool[(N−1) % 2]`) — which
   the trainer almost certainly already sent while the engine was busy
   with step *N*'s forward.
4. Engine returns from `send()` immediately. Step *N*'s ack is collected
   at the *start* of step *N+1*.

The 1.26 ms round-trip is now overlapped with the engine's next-step
generate (tens of ms) instead of stalling the colocate loop. The
engine-visible `send()` duration drops by ~1.26 ms — from ~1.58 ms to
~0.3 ms for the Eagle3 case.

**Correctness notes to encode in the implementation:**
- `K ≥ 2` so step *N* never lands in the slot whose step *N−1* ack is
  still outstanding.
- The final step must **drain** the last outstanding ack before
  teardown (a `flush()` call at loop exit).
- If `seq_len` grows past `MAX_TOKENS`, the pool buffer is reallocated —
  that one step pays a fresh `cudaIpcOpenMemHandle` (cache miss) and
  must not be in flight; size `MAX_TOKENS` generously to make this rare.

### Opt 3 — IPC-event ack instead of a gloo-byte ack *(optional)*

**Attacks:** the *nature* of the ack rather than its placement.

Instead of the trainer sending a 1-byte gloo message, the trainer
records a CUDA event after its D→D copy; the engine waits on that event.
Cross-process events need `cudaIpcGetEventHandle` exchanged **once** at
startup. This replaces a gloo TCP round-trip with a much cheaper
device-side `cudaEventSynchronize` / stream wait.

**Relationship to Opt 2:** Opt 2 *hides* the ack; Opt 3 *shrinks* it.
They are largely **alternatives** — if Opt 2 ships, the ack is already
off the critical path and Opt 3 adds little. Opt 3 is the fallback if
double-buffering's lifetime bookkeeping is judged too complex. Keep it
in the benchmark as a separate arm; promote it only if Opt 2 is dropped.

### Opt 4 — Static metadata fast path *(minor)*

Today every step pickles `(name, shape, dtype, ipc_args)` and ships a
length-framed blob. Once Opt 1's pool fixes the handles and dtype, the
only per-step variable is `seq_len`. The per-step message can collapse
to a fixed-size header — `(slot:int, seq_len:int)` — shipped as a tiny
int tensor, skipping `pickle` entirely. Saves a slice of `handle export`
+ `ship handles` (~0.1–0.2 ms). Small; bundle it with Opt 1.

### Projected combined effect

Estimates for the **Eagle3 160 MB** case — to be confirmed by the
benchmark in Part 3 (numbers are projections, not measurements):

| Configuration | engine `send()` | end-to-end | vs current IPC |
|---|--:|--:|--:|
| current IPC (baseline) | ~1.58 ms | ~1.9 ms | 1.0× |
| + Opt 1 (pool + cache) | ~1.3 ms | ~1.4 ms | ~1.4× |
| + Opt 1 + Opt 2 (pipelining) | **~0.3 ms** | ~0.5 ms (engine-visible) | **~5×** |

The headline is Opt 2: it removes the largest cost from the engine's
critical path. Opt 1 is its prerequisite and a modest win on its own.

> **Measured 2026-05-21 — see Parts 4 & 5.** `ipc-pipe` delivered
> **3.2×** non-MPS and **3.9×** under MPS on the Eagle3 engine-`send()`
> stall. `ipc-pool` *alone* did **not** — break-even, a net regression
> at 256 MB — so Opt 1 ships only bundled inside Opt 2, never standalone.

---

## Part 3 — Benchmark plan: optimized vs. current CUDA IPC

Goal: an apples-to-apples A/B of each optimization against today's IPC
path, on the same hardware and payloads as
[`transport_benchmark.md`](transport_benchmark.md), so results drop
straight into a comparison table.

### Where it runs

Extend [`scripts/colocate/bench_transport.py`](../../scripts/colocate/bench_transport.py).
It already: spawns two processes on one GPU (the colocate topology),
forms a 2-rank gloo group, sweeps payload sizes + a realistic Eagle3
multi-tensor case, and produces a per-stage breakdown. Keep all of that;
add new transport arms and two new knobs.

### Transport arms to register

| Arm | Description |
|---|---|
| `gloo` | existing CPU-staged baseline (kept for context) |
| `ipc` | **current** implementation — the A/B baseline |
| `ipc-pool` | Opt 1: persistent send-buffer pool + trainer mapping cache |
| `ipc-pipe` | Opt 1 + Opt 2: pool + one-step ack deferral (double-buffered) |
| `ipc-event` | Opt 1 + Opt 3: pool + IPC-event ack |
| `ipc-all` | Opt 1 + Opt 2 + Opt 4 (the recommended production stack) |

### How to implement the arms without forking the benchmark

Prototype each variant **inside the benchmark first** (the benchmark
already inlines replicas of `ipc_send`/`ipc_recv` in `_breakdown`). Once
an arm wins, fold it into [`cuda_ipc.py`](../../torchspec/colocate/cuda_ipc.py)
behind env flags so production and the benchmark share one code path:

- `TORCHSPEC_COLOCATE_IPC_POOL=1` — enable Opt 1
- `TORCHSPEC_COLOCATE_IPC_PIPELINE=1` — enable Opt 2 (implies pool)

Independent flags keep each optimization individually A/B-testable and
individually revertable.

### New knobs

- `--reuse-buffers` / cold-vs-warm reporting. The current benchmark
  "allocates a fresh payload every iteration" — this is the realistic
  worst case that *defeats* any cache, and it is exactly what the `ipc`
  baseline should keep doing. The pool arms inherently reuse their own
  buffers. So instead of a flag, **report cold vs warm per arm**: the
  first measured iteration (cold — pays the one-time `cudaIpcOpen*`)
  separate from the mean of the rest (warm — steady state). The `ipc`
  baseline will show no cold/warm gap (it pays the handshake every
  iter); pool arms will show a large gap. That gap *is* the Opt 1 win.

- `--engine-step-ms N` (default ~20). Inserts a dummy CUDA kernel /
  `time.sleep` of `N` ms between transfers, standing in for the engine's
  next-step `generate()`. Without this, ack pipelining has nothing to
  overlap against and its benefit is invisible. With it, `ipc-pipe`'s
  engine `send()` duration drops by ~1.26 ms because the deferred ack
  wait overlaps the dummy compute.

### Metrics to report, per arm, per payload

1. **end-to-end** barrier-to-barrier mean / p99 (existing).
2. **engine `send()` own-call** mean — the number that matters for the
   colocate loop (the engine stall). This is where Opt 2 shows up.
3. **trainer `recv()` own-call** mean.
4. **per-stage breakdown** — export / ship / ack-wait / open / copy,
   plus the new `engine pool copy` stage for the pool arms.
5. **cold vs warm** split (see knob above) — isolates Opt 1.

### Correctness gate (must pass before any timing is trusted)

The benchmark already builds deterministic payloads. For every arm,
assert **byte-equality** of every received tensor against the sent
tensor (`torch.equal`), every iteration. A faster arm that corrupts
data is a fail, not a win. Pipelining especially: verify the trainer
reads slot *N* before the engine overwrites it at step *N+2*.

### Expected output

A comparison table appended to [`transport_benchmark.md`](transport_benchmark.md)
(or a new "optimized transport" section), in the same shape as the
existing end-to-end table:

```
| Payload | ipc e2e | ipc-pool e2e | ipc-pipe engine-send | speedup vs ipc |
|---------|---------|--------------|----------------------|----------------|
| Eagle3 160 MB | (fill) | (fill) | (fill) | (fill) |
```

Plus a regression assertion in the benchmark: each optimized arm must
be **≥ the `ipc` baseline** on engine `send()` for payloads > 4 MB
(below the ~3–4 MB crossover none of this matters — colocate hidden
states are hundreds of MB, so that regime never applies).

### Reproduce (once the arms land)

```bash
# all arms, full sweep + Eagle3 case + breakdown + cold/warm split
python scripts/colocate/bench_transport.py --arms gloo,ipc,ipc-pool,ipc-pipe,ipc-all

# isolate the ack-pipelining win: needs a non-trivial engine step to overlap
python scripts/colocate/bench_transport.py --arms ipc,ipc-pipe --engine-step-ms 20
```

---

## Part 4 — Measured results, non-MPS baseline (2026-05-21, H100 SXM)

> **✅ Re-validated under MPS — see Part 5.** The non-MPS caveat that
> stood here is resolved. The step-0 MPS hang was a probe bug —
> `probe_ipc_capability()` ran a `reduce_tensor()` IPC smoke test that
> poisoned the MPS context (fixed in `e166c21`); it was never the
> transport. The A/B was repeated under MPS and the numbers hold
> (Eagle3 `ipc-pipe` 3.2× → **3.9×**). The tables in this Part 4 are the
> original **non-MPS** run, kept for comparison; Part 5 has the
> MPS-validated numbers.

The four arms (`gloo`, `ipc`, `ipc-pool`, `ipc-pipe`) were run on a
RunPod **1×H100 80GB SXM** (torch 2.4.1 + CUDA 12.4, no MPS), 5 warmup +
30 measured iterations, a fresh payload allocated every iteration. All
four arms passed the iteration-0 byte-equality gate.

### Engine `send()` stall — the colocate-loop metric (warm mean, ms)

How long the engine is blocked inside the transfer before it can resume
its next step — the number that matters for the colocate loop.

| Payload | `ipc` (baseline) | `ipc-pool` | `ipc-pipe` | ipc → ipc-pipe |
|---|--:|--:|--:|--:|
| single 4 MB | 1.466 | 1.742 | 0.670 | **2.2×** |
| single 16 MB | 1.524 | 1.239 | 0.780 | **2.0×** |
| single 64 MB | 1.725 | 1.310 | 0.670 | **2.6×** |
| single 256 MB | 1.707 | 2.681 | 1.387 | **1.2×** |
| **Eagle3 160 MB (realistic)** | **2.646** | **2.368** | **0.817** | **3.2×** |

End-to-end (barrier-to-barrier) on the Eagle3 case also improved — `ipc`
3.55 ms → `ipc-pipe` 1.53 ms (2.3×): with the ack deferred, the current
step's round-trip is not inside the measured window at all.

### Stage anatomy — both mechanisms confirmed

| Stage | `ipc` baseline | `ipc-pool`/`ipc-pipe` (warm) | verdict |
|---|--:|--:|---|
| `cudaIpcOpenMemHandle` (handle open) | 0.630 ms / step | **0.011 ms** | mapping cache eliminates it |
| ack wait | 1.933 ms / step | **0.138 ms** (deferred) | pipelining lifts it off the critical path |

### Findings

1. **Opt 2 (ack pipelining) — decisive, ship it.** `ipc-pipe` cut the
   engine `send()` stall on the realistic Eagle3 payload from 2.65 ms to
   0.82 ms (**3.2×**), and 2.0–2.6× across the rest of the
   colocate-relevant range. The stage anatomy proves the mechanism: the
   ack wait collapses from 1.93 ms to 0.14 ms.

2. **The handle cache works as designed.** `cudaIpcOpenMemHandle` drops
   from 0.630 ms *every step* to 0.011 ms warm — a persistent buffer +
   trainer-side mapping cache makes it a one-time cost.

3. **Opt 1 (pool + cache) ALONE is not worth shipping.** Standalone
   `ipc-pool` was break-even — the ~0.6 ms the handle cache saves is
   eaten by the extra engine-side D→D pool copy and its sync. At 256 MB
   it is a **net regression** (engine `send()` 1.71 → 2.68 ms: copying a
   256 MB tensor into the pool costs more than the handle-open it
   avoids). Opt 1's value is **solely as the enabler** for Opt 2 — the
   double-buffered pool that pipelining requires.

4. **Caveat — very large single tensors.** At 256 MB single, `ipc-pipe`
   is only 1.2× (the extra pool copy erodes the win). Real colocate
   hidden states are the Eagle3 multi-tensor shape (160 MB across three
   tensors), where `ipc-pipe` delivers the full 3.2×.

5. **Absolute scale, in perspective.** The win is ~1.8 ms/step lifted
   off the engine's critical path. Against a colocate step measured in
   tens of ms that is real but small — consistent with Part 1: the
   transport is not currently a step-time bottleneck.

> Run with the committed benchmark: `python scripts/colocate/bench_transport.py`
> (worktree branch `feature/colocate-transport-opt`). `--engine-step-ms`
> was 0 for this run; the engine-`send()` and stage-anatomy tables
> already isolate each win, so the pacing knob was not needed.

---

## Part 5 — MPS re-validation (2026-05-21)

The Part 4 A/B was re-run **under CUDA MPS** after the step-0 hang was
root-caused and fixed.

### The hang was a probe bug, not the transport

`probe_ipc_capability()` ran a `reduce_tensor()` "smoke test" — it
shared a scratch CUDA tensor over IPC with no consumer ever mapping it,
leaving PyTorch's CUDA-IPC producer-side machinery in a state that
wedged the engine's next `generate()` under MPS. The transport
(`ipc_send` / `ipc_recv`) was never the cause. Fixed in **`e166c21`**:
`probe_ipc_capability()` no longer calls `reduce_tensor()` — it does a
non-destructive `PYTORCH_*ALLOC_CONF` config check instead.

### Standalone bench — under MPS (`MPS active: yes`, all arms byte-correct)

H100 80GB SXM, torch 2.9.1, 5 warmup + 30 iters. Engine `send()` stall,
warm mean (ms):

| Payload | `ipc` | `ipc-pool` | `ipc-pipe` | ipc → ipc-pipe |
|---|--:|--:|--:|--:|
| single 16 MB | 1.607 | 1.206 | 0.439 | **3.7×** |
| single 64 MB | 1.695 | 1.198 | 1.557 | **1.1×** |
| single 256 MB | 1.614 | 1.998 | 1.384 | **1.2×** |
| **Eagle3 160 MB** | **3.006** | **1.953** | **0.780** | **3.9×** |

Stage anatomy confirms both mechanisms under MPS: `cudaIpcOpenMemHandle`
0.668 ms → **0.008 ms** (handle cache), ack wait 2.066 ms → **0.117 ms**
(pipelining). The non-MPS conclusions hold — MPS does not change the
transport story; `ipc-pipe` is, if anything, slightly better under MPS
(3.2× → 3.9×).

### Colocate-loop A/B — real `train_entry`, 50 steps, MPS active

`train_entry` colocate-tiny (Qwen3-0.6B), IPC-default vs gloo
(`TORCHSPEC_COLOCATE_IPC=0`):

| Arm | Result | Warm step | Loss |
|---|---|--:|---|
| IPC (default) | 50 steps, no hang | 0.142 s | 12.02 → 7.75 |
| gloo | 50 steps | 0.142 s | 12.02 → 7.75 |

IPC-default runs clean in the real colocate loop under MPS — the fix is
validated end-to-end. The two transports are **indistinguishable
in-loop on the tiny model**: the few-MB tiny-model payload is noise
against a ~142 ms step. The transport only moves the needle at the
large-payload (Eagle3 160 MB) scale the standalone bench measures —
consistent with Part 1: the transport is not a colocate step-time
bottleneck.

### Long-run stability — 4-GPU multi-engine, 3000 steps

A soak on the real multi-GPU topology: `train_entry` with
`colocate_qwen0p6b_2eng_tp2_tiny.yaml` (2 engines × `engine_tp_size=2`,
`dp_size=4`, union world 2N=8 on **4×H100** MPS-shared), CUDA IPC
default, **3000 steps**.

* **3000/3000 steps completed** — no hang, no crash, no NaN/OOM.
* **step time flat** — ~0.16–0.18 s from step 80 through step 2980 (no
  drift: step 80 = 0.177 s, step 2980 = 0.183 s).
* **peak alloc flat** — ~5.2–5.45 GB throughout, oscillating with
  variable `seq_len` but with **no upward trend** over 3000 steps (no
  leak; the Phase-6 flatness criterion holds).

The IPC-default colocate path is stable over a long multi-engine
multi-GPU run — a clean stability signal on top of the correctness
result above.

### Follow-up — `test_colocate_tiny` env conflict (resolved)

At round 9 (`1f62aaf`) the `e166c21` probe fix exposed a conflict:
`test_colocate_tiny._make_env` still forced
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, incompatible with
IPC-default, so the test fail-fasted at engine init. **Resolved by
`7aeaa80`** (the `--full` IPC re-validation) — `test_colocate_tiny`
passes again (verified green on the 3000-step pod's build).

---

## Recommendation & sequencing

1. **Do not** write C++/CUDA or Triton — the transport has no kernel to
   optimize and the copy is bandwidth-saturated (Part 1). The GPU A/B
   (Part 4) confirms the only headroom was protocol-level.
2. **First**, re-run `run_smoke_host.sh --full` on 4×H100 with IPC as the
   new default — the open item from round 7; it settles *stability*
   (the benchmark already settled *performance*).
3. **`ipc-pipe` (pool + ack pipelining) is now MPS-validated — 3.9× on
   the Eagle3 engine-`send()` stall** (Part 5). The step-0 hang was a
   probe bug (`e166c21`), not the transport. **`ipc-pipe` is now folded
   into production** — `cuda_ipc.py` carries `IpcPipelineTransport`
   (pool + cache + one-step ack deferral + variable-`seq_len`
   grow-to-fit resize), wired into `NcclHiddenStatesConnector` and
   `NcclMultiTensorFetcher` behind the **opt-in**
   `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag (default off; the plain
   `ipc_send`/`ipc_recv` path is unchanged). The design is teardown-safe
   without an explicit flush (the engine never blocks on the final ack;
   the trainer keeps ≤1 ack `isend` in flight), so no sglang-patch
   change was needed — `flush()` exists for tidiness only. **GPU-validated
   2026-05-21** on a 4×H100 pod (`run_smoke_host.sh --full` with the flag):
   12/13 colocate tests passed first time; `test_phase6_peak_alloc_flatness`
   OOM'd the memory-tight Qwen3-8B config — the variable-`seq_len` resize
   *retired pool buffers without ever freeing them*, and the ×2 grow
   overshoot stacked on sglang's KV cache. **Fixed**: the resize is now
   exact-size (no overshoot) and a retired buffer is freed one step later,
   the moment the trainer acks the resize step. Re-test passed (peak-alloc
   flat ~25.75 GB, no OOM). **Still low-priority.** Do **not** ship
   `ipc-pool` alone (break-even, regresses at 256 MB) — the flag
   deliberately enables the full pool+pipe stack, never the pool by itself.
4. **Opt 3 / Opt 4 — skip.** Opt 2 already takes the ack to 0.14 ms, so
   the IPC-event ack (Opt 3) has nothing left to win; Opt 4 (static
   metadata) is in the noise.
5. **Priority: low.** ~1.8 ms/step against a tens-of-ms step —
   worthwhile, not urgent. Do it when colocate step-time optimization
   comes up, not before.

**Bottom line:** no C++/CUDA/Triton (Part 1, MPS-independent). The
`ipc-pipe` optimization is real and **MPS-validated** — 3.9× on the
Eagle3 engine-`send()` stall (Part 5) — but low-priority: the transport
is not a colocate step-time bottleneck. The step-0 MPS hang was a probe
bug, fixed in `e166c21`; CUDA IPC as the default transport is correct.

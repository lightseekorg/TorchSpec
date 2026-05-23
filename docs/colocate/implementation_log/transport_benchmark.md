# Colocate hidden-state transport benchmark — gloo CPU-staging vs CUDA IPC

Measured **2026-05-21** on a RunPod **1×H100 80GB HBM3** (SXM), torch
2.4.1 + CUDA 12.4, with [`scripts/colocate/bench_transport.py`](../../scripts/colocate/bench_transport.py).

> **See also:** [`transport_optimization.md`](transport_optimization.md) —
> whether to hand-write a C++/CUDA or Triton kernel for this transport
> (no — the only kernel in the path is a bandwidth-saturated D→D copy),
> plus the protocol-level optimization design (send-buffer pool + handle
> cache, ack pipelining) and its GPU A/B — **validated under MPS**:
> `ipc-pipe` cuts the engine `send()` stall **3.9×** on the realistic
> Eagle3 case, and CUDA IPC runs clean in the real colocate loop (the
> step-0 MPS hang was a probe bug, fixed in `e166c21`) — see that doc's
> Part 5.

## TL;DR

For realistic colocate hidden-state payloads, **CUDA IPC is ~170× faster
than gloo CPU-staging** — the Eagle3-shaped 160 MB case transfers in
**1.9 ms** over CUDA IPC vs **319 ms** over gloo. The speedup widens with
payload size: gloo's CPU-staged path is bottlenecked at ~0.5 GB/s, while
CUDA IPC stays ≈1 ms almost flat because the only real data movement is a
single on-device D→D copy. This is the measured justification for making
CUDA IPC the default transport.

The one exception is **tiny payloads (<~3 MB)**, where IPC is marginally
slower (0.5–0.8×) — its fixed handshake + `cudaIpcOpenMemHandle` cost
(~1 ms) dominates. Colocate hidden states are tens-to-hundreds of MB, so
that regime never applies in practice.

## End-to-end transfer latency

Barrier-to-barrier end-to-end transfer (engine send + trainer recv/copy);
8 warmup + 40 measured iterations; a fresh payload allocated every
iteration (so CUDA IPC pays a real `cudaIpcOpenMemHandle` each time).

| Payload | Size | gloo mean | gloo p99 | IPC mean | IPC p99 | gloo GB/s | IPC GB/s | **IPC speedup** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| single 0.25 MB | 0.25 MB | 0.335 ms | 0.531 ms | 0.639 ms | 0.774 ms | 0.8 | 0.4 | **0.5×** |
| single 1 MB | 1 MB | 0.800 ms | 0.964 ms | 0.948 ms | 1.002 ms | 1.3 | 1.1 | **0.8×** |
| single 4 MB | 4 MB | 2.937 ms | 5.446 ms | 1.124 ms | 1.192 ms | 1.4 | 3.7 | **2.6×** |
| single 16 MB | 16 MB | 14.979 ms | 24.695 ms | 1.533 ms | 1.609 ms | 1.1 | 10.9 | **9.8×** |
| single 64 MB | 64 MB | 154.399 ms | 186.129 ms | 0.773 ms | 0.959 ms | 0.4 | 86.8 | **199.7×** |
| single 256 MB | 256 MB | 497.434 ms | 564.811 ms | 0.822 ms | 0.991 ms | 0.5 | 326.6 | **605.1×** |
| **Eagle3 (4096t × 4096h, 3 tensors)** | **160 MB** | **319.076 ms** | 389.803 ms | **1.870 ms** | 1.949 ms | 0.5 | 89.7 | **170.6×** |

## Engine / trainer split (own-call duration, mean)

| Payload | gloo engine send | gloo trainer recv | IPC engine send | IPC trainer recv |
|---|--:|--:|--:|--:|
| single 0.25 MB | 0.154 ms | 0.254 ms | 0.445 ms | 0.555 ms |
| single 1 MB | 0.459 ms | 0.725 ms | 0.663 ms | 0.836 ms |
| single 4 MB | 1.493 ms | 2.831 ms | 0.798 ms | 1.005 ms |
| single 16 MB | 9.145 ms | 14.711 ms | 1.073 ms | 1.374 ms |
| single 64 MB | 129.110 ms | 154.097 ms | 0.555 ms | 0.672 ms |
| single 256 MB | 455.701 ms | 497.103 ms | 0.631 ms | 0.733 ms |
| Eagle3 (160 MB) | 297.242 ms | 318.804 ms | 1.583 ms | 1.740 ms |

## Per-stage breakdown — single 256 MB

| Stage | Time |
|---|--:|
| `gloo.engine D->H copy` | 176.791 ms |
| `gloo.engine gloo ship` | 272.904 ms |
| `gloo.trainer gloo recv` | 459.425 ms |
| `gloo.trainer H->D copy` | 34.673 ms |
| `ipc.engine handle export` | 0.203 ms |
| `ipc.engine ship handles` | 0.293 ms |
| `ipc.engine wait for ack` | 1.259 ms |
| `ipc.trainer handle open` | 0.518 ms |
| `ipc.trainer D->D copy` | 0.264 ms |

## Interpretation

- **gloo is bottlenecked by its own transport, not by PCIe.** The 256 MB
  breakdown shows the gloo ship (`dist.send`/`recv` over gloo's TCP
  transport on localhost) at ~270–460 ms — only ~0.5–0.9 GB/s. Even the
  engine's pageable D→H copy is slow (~177 ms ≈ 1.4 GB/s). gloo is built
  for small control-plane collectives, not bulk tensor transfer; the
  colocate gloo path inherits that ceiling.
- **CUDA IPC is near-constant-time.** 64 MB → 0.77 ms, 256 MB → 0.82 ms.
  The actual D→D copy is **0.26 ms for 256 MB** (~1 TB/s effective). The
  dominant IPC cost is the fixed handshake — `cudaIpcOpenMemHandle`
  (~0.5 ms) plus the ack round-trip — so IPC latency is essentially
  payload-size-independent across the whole colocate range.
- **Crossover is ~3–4 MB.** Below it, IPC's fixed overhead loses to gloo;
  above it IPC wins by a widening margin. Real Eagle3 hidden states (the
  160 MB case) sit deep in IPC-favorable territory → **170×**.
- **Per-step impact.** In the serial colocate loop (engine produces →
  transfer → trainer trains) the transfer is pure stall. Replacing a
  ~300 ms gloo stall with a ~2 ms IPC stall removes the hidden-state
  transfer as a step-time factor entirely.

### Caveats

- The tables above were measured **without MPS** (two plain processes);
  the benchmark was later **re-run under MPS** —
  [`transport_optimization.md`](transport_optimization.md) Part 5 — and
  the ratios held (CUDA IPC stays flat, ~85–170× over gloo on the Eagle3
  payload), so this caveat is **resolved**. The step-0 MPS hang that
  briefly blocked IPC-default was a separate probe bug (`e166c21`), not
  a transport issue. The rest of this note still holds: MPS changes
  kernel-scheduling concurrency, not the transport mechanism — and the
  transfer is serial (engine sends while trainer waits), so there is
  little kernel overlap
  to gain. The headline ratio holds.
- The gloo arm uses pageable host memory (`.to("cpu")`), matching the
  current `NcclHiddenStatesConnector`. Pinned host memory would speed
  gloo's copies somewhat but not its TCP ship, which is the dominant term.
- IPC re-pays `cudaIpcOpenMemHandle` every step because the engine
  reallocates hidden states each step. A handle cache keyed by device
  pointer is a possible future optimization, but at ~0.5 ms it is not
  currently a bottleneck. See [`transport_optimization.md`](transport_optimization.md)
  for the full protocol-level optimization plan (send-buffer pool +
  handle cache, ack pipelining) and how to A/B it against this baseline.

## Reproduce

```bash
# Any 1-GPU host; no `pip install` needed — bench_transport.py loads
# cuda_ipc.py directly and runs on a bare torch install.
python scripts/colocate/bench_transport.py
python scripts/colocate/bench_transport.py --iters 40 --warmup 8 --sizes-mb 1,16,256
```

## Real-workload validation — `--full` run under CUDA IPC default (2026-05-21)

The benchmark above measures the transport **in isolation** (and without
MPS). This section records CUDA IPC's performance in the **real colocate
loop** — the `run_smoke_host.sh --full` matrix on a RunPod 4×H100, with
CUDA IPC as the default transport (after the round-9 probe fix). 13
colocate tests passed; per-step metrics are the driver's
`[colocate_loop] step=… step_time=… loss=… peak_alloc=…` log.

### Per-step time (CUDA IPC default)

| Test | Config | Step 1 (cold) | Warm steps | Warm throughput |
|---|---|--:|--:|--:|
| `test_phase7_convergence` | ~25.8 GB-class, 50 steps | 42.8 s | **~0.18 s** (0.16–0.20 s) | ~5.3–5.5 step/s |
| `test_phase6_peak_alloc_flatness` | ~25.8 GB-class, 200 steps | — | **~0.177 s** (0.16–0.19 s) | ~5.5 step/s |
| `test_colocate_tiny` (Qwen3-0.6B) | ~5.2 GB, tiny | 13–50 s | **~0.10–0.16 s** | — |

Step 1 is a one-time cold start (sglang engine warm-up + first
`generate()` + MPS/CUDA init) — **not** transport-related and identical
under gloo. Warm steps are the steady state.

### Two findings

- **The transport is invisible in the step budget.** A warm colocate
  step is ~0.18 s; the round-7 benchmark puts a CUDA IPC hidden-state
  transfer at ~1–2 ms — i.e. **~1 % of the step**. CUDA IPC is not a
  step-time factor. (gloo, at ~300 ms for the same payload, *would* be —
  it would more than double the step.)
- **No memory leak from the per-step IPC handles.** Over the 200-step
  stability test, `peak_alloc` stayed within
  25 754 027 520 – 25 757 765 120 B — a **0.014 %** spread on a 25.75 GB
  footprint. CUDA IPC exports/opens a fresh handle every step; that
  churn does not accumulate.

Convergence is correct under CUDA IPC — loss decreases monotonically
(`test_phase7_convergence`: 12.13 → 3.27 over 50 steps; the tiny test:
12.02 → 9.74 over 20) — confirming the transport delivers correct
hidden states in the real loop, not just in the byte-equality benchmark.

> Round 9 found and fixed the bug that made CUDA IPC unusable as the
> default (a destructive capability probe wedged CUDA under MPS — see
> `implementation_log.md`). These numbers are from the post-fix
> re-validation.

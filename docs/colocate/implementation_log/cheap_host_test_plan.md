# Colocate Cheap-Host Test Plan

> Self-contained agent handoff for validating the colocate (MPS+NCCL)
> training mode on a non-Modal host. Modal sandbox blocks NVIDIA MPS at
> the gVisor runtime layer (see `implementation_log.md` §"Modal sandbox
> MPS limitation"), so the Phase-4/6/7 tests that need MPS auto-skip
> there. This doc tells you how to actually *run* them on the cheapest
> GPU rental that supports MPS.
>
> Branch: `feature/colocate-training-inference` (TorchSpec)
> Last verified Modal sandbox baseline: 2026-05-13.

---

## TL;DR

```bash
# On any cheap GPU host with --ipc=host (RunPod, Vast.ai, Lambda, etc.):
git clone https://github.com/zhubohao911/TorchSpec.git
cd TorchSpec
git checkout feature/colocate-training-inference
bash scripts/colocate/run_smoke_host.sh        # 1-GPU tiny smoke (~25 min)
# OR for 4×H100 hosts:
bash scripts/colocate/run_smoke_host.sh --full # full Phase-4/6/7 (~90 min)
```

Exit code `0` = every selected test PASSED or SKIPPED cleanly. Anything
else is a real failure; the captured pytest output names the test that
failed.

---

## What you're validating

The MPS-required colocate code path exercises:

- `torchspec/colocate/mps.py` — NVIDIA MPS daemon lifecycle + the
  `_probe_mps_server_works` cuInit/cuDeviceGetCount probe.
- `torchspec/colocate/world.py` — the `UnionWorldSpec` rendezvous and
  lazy-init NCCL `init_process_group` (no `device_id=` so slow engines
  get the full timeout).
- `torchspec/training/nccl_data_fetcher.py` — multi-tensor receive
  with deterministic key ordering.
- `torchspec/inference/engine/nccl_hidden_states_connector.py` — the
  engine-side P2P send.
- `torchspec/controller/colocate_loop.py` — the synchronous
  trainer↔engine loop (Phase 5 body).
- The sglang `colocate.patch` (see `patches/sglang/v0.5.8.post1/`)
  and its three patch points: `init_union_default_pg`, the spec-training
  callback (`_send_hidden_states_to_nccl`), and the scheduler init
  (`Scheduler.__init__`).

A single working colocate step on **any** GPU exercises all of the
above. The 4-GPU + Qwen3-8B tests stress the same code under realistic
sharding (FSDP world=4, TP=4, true 1:1 trainer↔engine bundle pairing
under MPS sharing). The 1-GPU tiny variant is the cheapest credible
correctness check.

---

## Cost-tier matrix

Pick the cheapest tier that satisfies your validation goal.

**GPU compatibility requirement: SM89 or newer (Ada / Hopper / Blackwell).**
The pre-built `sgl_kernel 0.3.21` wheel that the runner installs only
ships `sm90` (Hopper) and `sm100` (Blackwell) binaries — Ada (sm89) and
Ampere (sm80/sm86) variants are missing. Practical implication: **A100,
A6000, RTX 3090, RTX A5000, RTX 4090, L40S, and RTX 6000 Ada will NOT
load `sgl_kernel.common_ops` at engine startup.** This was originally
covered in the test plan as "RTX A6000 (Recommended)" — that line is now
struck through. Confirmed empirically on RunPod 2026-05-13; see
`docs/colocate/implementation_log.md` §"RunPod validation session"
for the wheel layout. Workaround is to build `sgl_kernel` from source on
the host (~20-min compile, needs CUDA toolkit), or use a sm90+ GPU.

| Goal | Recommended host | $/hr | One pass | Tests run |
|---|---|---|---|---|
| Tiny correctness only | 1×H100 PCIe 80 GB on **RunPod** SECURE | ~$2.39 | ~30 min | tiny one-step + tiny convergence |
| Tiny correctness only | 1×H100 PCIe 80 GB on **RunPod** community (if available) | ~$2.50 | ~30 min | same |
| Tiny correctness only | 1×H100 SXM5 80 GB on **RunPod** SECURE | ~$2.99 | ~30 min | same |
| Tiny correctness only | 1×H100 80 GB on **Vast.ai** spot | ~$2.00 | ~25 min | same (with room for full Qwen3-8B) |
| Full Phase-4/6/7 | 4×H100 80 GB on **Hyperstack** | ~$8/hr | ~90 min | all five test files |
| Full Phase-4/6/7 | 4×H100 on **Lambda Labs** spot | ~$10/hr | ~90 min | all five test files |
| Full Phase-4/6/7 | 4×H100 SXM on **RunPod** community | ~$10–12/hr | ~90 min | all five test files |

~~Tiny correctness only | 1×L40S 48 GB on Vast.ai | ~$0.50~~ — sm89 not supported by bundled sgl_kernel wheel.
~~Tiny correctness only | 1×A6000 48 GB / 1×4090 24 GB on Vast.ai | ~$0.40~~ — sm80/sm86 not supported either.

**Important: pick a Vast.ai or RunPod template that has Docker support
with `--ipc=host` enabled.** Most "PyTorch" templates default to this;
look for "shared IPC" or "interactive" mode in the rental UI. On RunPod
the `runpod-torch-v240` template is confirmed working.

**Runner orchestration tip:** drive provisioning with `runpodctl`
(brew-installed; `runpodctl doctor` for auth setup) rather than the web
UI. Each step is a discrete API call so the loop is
`pod create → ssh -i ... 'bash -s' < bootstrap.sh → scp report → pod delete`.
The H100 PCIe `gpu-id` is the literal string `'NVIDIA H100 PCIe'` (NOT
`'NVIDIA H100 80GB HBM3'` which is the SXM variant). When `pod create`
hits "no instances available", DO NOT retry in a tight loop without
sleep — partial successful responses can race and you'll get multiple
charged pods. Always confirm with `runpodctl pod list` immediately.

---

## Pre-flight requirements (any host)

The runner script aborts with exit code 1 if any of these are missing:

1. `nvidia-smi` reports at least 1 GPU with CUDA capability ≥ **9.0**
   (Hopper / Blackwell). The bundled `sgl_kernel 0.3.21` wheel doesn't
   ship Ada (sm89) or Ampere (sm80/sm86) variants, so realistically
   only H100/H200/B200 GPUs work without a source build. 80 GB VRAM is
   plenty for the tiny config; minimum 24 GB if you happen to find a
   sm90+ card with less RAM.
2. `nvidia-cuda-mps-control` is on `$PATH` (ships with the CUDA
   toolkit; almost always pre-installed on rental images).
3. **`libnuma.so.1` available system-wide** for `sgl_kernel`'s native
   `common_ops.abi3.so` to dlopen at engine startup. RunPod's stock
   `runpod-torch-v240` image does *not* ship this; the runner's
   bootstrap installs it via `apt-get install -y libnuma1`. If you
   roll your own bootstrap on a fresh image, do the same — without
   it, `sgl.Engine(...)` will crash with
   `ImportError: libnuma.so.1: cannot open shared object file`.
   (You no longer need `libibverbs1` / `librdmacm1` / `libnl-3-200`
   for the colocate path — commit `3f7e708` made the Mooncake
   imports lazy, so only the disagg path needs the RDMA verbs stack.)
3. Container runtime passes `--ipc=host` (or you're on a bare VM).
   On Vast.ai this is the default for "On-Demand" instances; on RunPod
   it's the default for "Pods" but **not** for "Serverless" endpoints.
4. Outbound HTTPS to `github.com` and `huggingface.co` (for sglang
   clone + Qwen3-0.6B-Base download — model is **not gated**).

**Quick MPS sanity check** (run on the host before committing time). The
runner does this automatically in pre-flight, but it's also useful as a
standalone 30-second smoke test from a fresh checkout:

```bash
PYTHONPATH=. python -m tests.colocate._mps_probe
# Prints e.g.   mps_works: True  — ok
# Or            mps_works: False — cuInit/cuDeviceGetCount returned rc=805 (operation not supported)
# Exit 0 if MPS works on this host; 1 if it doesn't.
```

If you don't have torchspec checked out yet and just want to test the
MPS plumbing manually:

```bash
nvidia-cuda-mps-control -d                 # start daemon
echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control
# Expect: a number like "100.0"; if you get
#   "Failed to talk to MPS control daemon"
#   "operation not supported"
# the host doesn't actually support MPS — try a different rental.
echo "quit" | nvidia-cuda-mps-control      # cleanup
```

---

## RunPod-specific setup

RunPod is the platform the user named, so here's the explicit recipe.

1. **Choose a Pod template**: pick "PyTorch 2.4" or "RunPod CUDA 12.4"
   on a community-cloud GPU. Avoid "Serverless" — those run with
   restricted IPC.
2. **GPU**: 1×H100 PCIe (~$2.50/hr) for the tiny smoke or 4×H100 SXM
   (~$12/hr) for the `--full` matrix.
3. **Volume**: attach a 50 GB workspace volume mounted at `/workspace`
   (the model + sglang clone fit in ~10 GB; 50 GB leaves headroom for
   future runs).
4. **Network**: enable "Public IP" + "Start SSH" so you can SSH in.
5. **Once the pod is running**, SSH in and:

   ```bash
   cd /workspace
   git clone https://github.com/zhubohao911/TorchSpec.git
   cd TorchSpec
   git checkout feature/colocate-training-inference

   # Tiny smoke (1×H100 host):
   bash scripts/colocate/run_smoke_host.sh

   # OR full matrix (4×H100 host):
   bash scripts/colocate/run_smoke_host.sh --full
   ```

6. **Watch for the success markers** in the pytest output (see below).
7. **Stop the Pod** as soon as the run completes — RunPod charges
   per-second whether it's busy or not.

If you see `MPS server reports 'operation not supported'` in the
pre-flight, the Pod template doesn't have shared IPC. Stop it, pick
the "Interactive" PyTorch template (or any template with "Direct
Network Mode" in the description), and try again.

---

## Vast.ai alternative (cheapest)

1. Search for "1x L40S" or "1x RTX 4090" with at least 24 GB VRAM,
   "Reliable" trust score, "Direct" net type. Filter by `--ipc=host`
   support: in the template list, pick "PyTorch (cuda:12.4)" or
   similar — both default to shared IPC.
2. Click **Rent**, then SSH in via the connection string.
3. Same git-clone + script invocation as the RunPod recipe above.
4. Vast.ai's typical 1×L40S spot price is around **$0.40–0.60/hr**;
   one tiny smoke pass is ~$0.20.

---

## What "passing" looks like

### Tiny smoke (`bash scripts/colocate/run_smoke_host.sh`)

Expected pytest output (excerpt) on a working MPS host:

```
tests/colocate/test_colocate_tiny.py::test_phase4_tiny_one_step PASSED
tests/colocate/test_colocate_tiny.py::test_phase7_tiny_loss_decreases PASSED

================ 2 passed in ~700s ================
```

Plus, in the captured stdout from each test, you should see:

```
[colocate_loop] step=1 loss=<float>
...
completed_steps=1 / num_steps=1     # for test_phase4_tiny_one_step
[colocate_loop] step=20 loss=<float>  # for test_phase7_tiny_loss_decreases
```

The runner exits `0` on success.

### Full matrix (`--full` on 4×H100)

```
tests/colocate/test_colocate_tiny.py::test_phase4_tiny_one_step      PASSED
tests/colocate/test_colocate_tiny.py::test_phase7_tiny_loss_decreases PASSED
tests/colocate/test_one_step.py::test_phase4_one_step_completes_end_to_end PASSED
tests/colocate/test_grad_parity.py::test_phase7_grad_parity_smoke    PASSED
tests/colocate/test_stability.py::test_phase6_peak_alloc_flatness    PASSED
tests/colocate/test_convergence.py::test_phase7_convergence_loss_decreases PASSED
```

(`test_stability` and `test_convergence` are `@pytest.mark.slow`; if
they don't run, pass `-m slow` via `--tests=...` or set
`PHASE6_STABILITY_STEPS` / `PHASE7_CONVERGE_STEPS` to non-default
values.)

### Pre-flight MPS probe failure (exit 1)

As of commit `0a1e153`+ the runner probes MPS *before* the expensive
setup step. On a host where the MPS daemon starts but the server can't
spawn a CUDA context (the most common cheap-host failure), pre-flight
fails in ~30 s with:

```
*** MPS pre-flight FAILED. ***

  All colocate tests would SKIP on this host. Most likely causes:
    * Container runtime is sandboxing IPC ...
    * Host kernel / driver doesn't support MPS sharing.
```

…and exit code `1`. **This is by design** — it saves you the 5–10
minutes of `pip install` that would otherwise precede an all-SKIP
pytest run. Switch host/template and re-run.

If you specifically want to validate the SKIP path (e.g. you're
verifying on Modal sandbox that the skip gate fires), set
`COLOCATE_SKIP_MPS_PROBE=1` to bypass the pre-flight gate. You'll then
see:

```
SKIPPED [1] tests/colocate/test_colocate_tiny.py:64: Tiny colocate
smoke needs working NVIDIA MPS. On hosts where the MPS server reports
'operation not supported' ...
```

…and exit code `0` (skip ≠ fail).

---

## Failure modes & how to diagnose

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi: command not found` | No NVIDIA driver | Wrong host / image. Use a CUDA-enabled template. |
| `nvidia-cuda-mps-control: command not found` | CUDA toolkit not installed | `apt-get install cuda-toolkit-12-4` or use a `nvidia/cuda:*-devel-*` image. |
| Pre-flight: `Need at least 1 GPU; found 0` | GPU not visible to the container | Re-launch with `--gpus all` (Docker) or pick a template with GPU passthrough enabled. |
| Test SKIP with `'operation not supported'` in MPS server log | No `--ipc=host` (gVisor / Modal-style sandbox) | Switch host or pick the "Interactive" template. |
| Test FAILS with `MPS daemon did not produce ... within 10s` | Stale state from a previous run | The runner's pre-flight now does `rm -rf /tmp/nvidia-mps /tmp/nvidia-log` automatically when no daemon is running. If this still fires, the daemon *is* running but is wedged — `echo quit \| nvidia-cuda-mps-control` then re-run. |
| Test FAILS with `socketPollConnect ... Connection refused` | Stale Ray cluster | The runner's pre-flight now runs `ray stop -f` automatically. If you still see this, a non-`ray`-managed actor is bound to the port — `pkill -f raylet` is the bigger hammer. |
| Test HANGS at `init_union_world` | sglang colocate.patch wasn't applied | Re-run with `--skip-setup` removed; the script's setup phase re-clones + re-patches sglang. |
| Test FAILS with `OutOfMemoryError` on the **tiny** config | GPU smaller than 24 GB | The tiny config needs at least 24 GB VRAM. Try a bigger GPU. |
| Test FAILS with `OutOfMemoryError` on the **full** config | Trying to run Qwen3-8B on <80 GB GPU | Stop trying to run `--full` on non-H100 / non-A100-80 hardware. |
| Cold start `pip install -e .` takes >10 min | Network throttling | Patience; the deps are large (~3 GB). On RunPod community-cloud the bandwidth is usually fine. |

When in doubt, the runner prints:

- `nvidia-smi --query-gpu=index,name,memory.total --format=csv` (host
  capabilities)
- `nvidia-cuda-mps-control` location and pre-flight result
- pytest's `-xvs` output streamed live (no buffering)

The `_run_train` helper inside the test files also dumps the last
4 KB of `/tmp/nvidia-log/control.log` and `/tmp/nvidia-log/server.log`
on any timeout.

---

## Reporting back

The runner writes a pre-baked report at `colocate-smoke-report.txt`
inside the repo root when pytest exits. Paste that file in your
report-back — it contains everything below already filled in:

1. **Host details**: cloud + GPU model + count + memory + driver
   version (auto-captured from `nvidia-smi`).
2. **Exit code** of `run_smoke_host.sh`.
3. **pytest summary line** (e.g. `2 passed in 712.34s`).
4. For each test that PASSED: the captured `loss=<float>` values from
   the `[colocate_loop]` lines (auto-grepped from the pytest log so
   we can sanity-check whether training is making sane progress).
5. For each test that FAILED: the last ~60 lines of pytest output
   plus the tail of `/tmp/nvidia-log/server.log` and `control.log`.
6. Total wall-clock seconds (you'll have to back-of-envelope the cost
   from the host's $/hr — the script doesn't know what tier you rented).

The full pytest output is also kept at `colocate-smoke-pytest.log`
in case the report's grep heuristics miss something interesting.

If exit code is non-zero **and** the failure isn't covered in the
table above, file a comment on the colocate-training-inference branch
or back-channel the agent who handed off this plan.

---

## Optional: longer stability runs

The default test horizons are sized for a fast cheap-host smoke.
For higher-confidence runs:

```bash
PHASE6_STABILITY_STEPS=1000 PHASE7_CONVERGE_STEPS=500 \
  bash scripts/colocate/run_smoke_host.sh --full
```

Wall-clock on 4×H100 SXM:

- `PHASE6_STABILITY_STEPS=1000` ≈ 30–40 min
- `PHASE7_CONVERGE_STEPS=500` ≈ 15–20 min

Both are still gated on `has_h100_quad() AND mps_works()`, so if the
host doesn't qualify they SKIP cleanly.

---

## Cleanup

Before stopping the host:

```bash
# (optional) Tear the MPS daemon down cleanly so the next user gets
# a clean slate. The runner's atexit hook does this automatically on
# normal exit; this is the manual incantation if pytest crashed:
echo "quit" | nvidia-cuda-mps-control || true
rm -rf /tmp/nvidia-mps /tmp/nvidia-log

# (optional) Delete the HF cache so the volume snapshot is small:
rm -rf ~/.cache/huggingface
```

Then stop the Pod / instance from the cloud console. **Don't forget**
— a 4×H100 instance left running for an hour costs ~$10.

---

## Where things live in the repo (for the next agent)

- `configs/colocate_qwen0p6b_tiny.yaml` — tiny config (1-GPU,
  Qwen3-0.6B-Base, mem fractions 0.45/0.45)
- `configs/colocate_qwen3_8b.yaml` — full config (4-GPU, Qwen3-8B)
- `tests/colocate/test_colocate_tiny.py` — tiny smoke (1+ GPU)
- `tests/colocate/test_one_step.py` — Phase-4 one-step (4+ GPU)
- `tests/colocate/test_grad_parity.py` — Phase-7 grad parity (4+ GPU)
- `tests/colocate/test_stability.py` — Phase-6 stability (4+ GPU, slow)
- `tests/colocate/test_convergence.py` — Phase-7 convergence (4+ GPU, slow)
- `tests/colocate/_mps_probe.py` — `has_n_gpus(n)` + `mps_works()`
  shared skip helpers
- `scripts/colocate/run_smoke_host.sh` — the runner (this doc's main
  artifact). Writes `colocate-smoke-report.txt` +
  `colocate-smoke-pytest.log` at repo root on exit.
- `scripts/modal/modal_colocate_smoke.py::phase_tiny` — same tiny
  test, runnable on Modal as a SKIP sanity check
- `patches/sglang/v0.5.8.post1/colocate.patch` — the upstream sglang
  patch that the runner's setup phase applies for you
- `docs/colocate/implementation_log.md` — the full phase-by-phase log;
  §"Cheap-host workflow for MPS-required validation" links back here
- `docs/colocate/sglang_patch.md` — patch surface contract

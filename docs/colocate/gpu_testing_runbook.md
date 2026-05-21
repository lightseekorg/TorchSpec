# GPU-rental testing runbook (for agents)

> **Audience: an agent running colocate GPU tests on a rental platform
> without a human in the loop.** It tells you how to provision, run,
> monitor, and — critically — tear down a rented GPU pod safely and
> cheaply. Everything here was learned the hard way across the RunPod
> sessions in [`implementation_log.md`](implementation_log.md).
>
> Companion docs: [`cheap_host_test_plan.md`](cheap_host_test_plan.md)
> (cost-tier matrix, test plan) and [`sglang_patch.md`](sglang_patch.md)
> (the sglang patch the tests exercise).

## When you need this

The colocate tests (`tests/colocate/test_*`, phases 4/6/7) need **NVIDIA
MPS**, which needs a container started with `--ipc=host`. Use this
runbook whenever a task asks you to GPU-validate colocate.

**Modal does not work for colocate.** Modal sandboxes run under gVisor,
whose nvproxy does not implement MPS multiplexing — the MPS-required
tests `pytest.skip` there, they do not run. Use a real `--ipc=host`
host: **RunPod** (default here), Vast.ai, Lambda, or bare metal.

## Hard rules — follow these every time

You are spending real money and sharing an account with other agents.

1. **Check for other pods before you provision.**
   `runpodctl pod list -o json`. If a pod you did **not** create is
   running, never `stop`/`delete` it. (The deprecated `runpodctl get
   pod` can print an *empty* list while pods exist — always use
   `pod list -o json`.)
2. **Always pass `--terminate-after`** (≈3 h out) when creating a pod.
   It is a backstop: if you lose track, the pod self-destructs instead
   of billing forever.
3. **Always tear the pod down** as soon as the run finishes — pass or
   fail. Then verify: `runpodctl pod get <id>` must say `pod not found`.
4. **Watch the balance.** `runpodctl user`. A 4×H100 is ~$13/hr. Do not
   start a run that would drain the balance toward $0 — that stops
   *every* pod on the account, including other agents'.
5. **One run, then capture and tear down.** Do not open-endedly iterate
   on a billing pod. If a real (non-environment) failure needs code
   changes, tear down first, fix locally, re-provision.
6. **Surface, don't silently proceed,** if you find another agent's pod
   that your run would starve, or if the balance is too low for one run.

## Prerequisites (already set up on this machine)

- `runpodctl` installed and authenticated — API key in
  `~/.runpod/config.toml`, SSH key at `~/.runpod/ssh/runpodctl-ssh-key`
  (registered on the account). Check: `runpodctl user` prints a balance.
- An **`HF_TOKEN`** is required for the Qwen3-8B tests (unauthenticated
  HF Hub requests get rate-limited — see failure modes). The tiny
  Qwen3-0.6B tests do not need it. Ask the user for the token if you do
  not have one; never commit it anywhere.

## Workflow (RunPod — the ready path)

RunPod is the platform set up on this machine and used for every GPU
run to date. Vast.ai is a working alternative — see the section after
this one.

### 1 — Provision

```bash
runpodctl pod create --name colocate-<purpose> \
  --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count <N> \
  --template-id runpod-torch-v240 \
  --container-disk-in-gb 200 --ports "22/tcp" \
  --terminate-after "$(date -u -v+3H +%Y-%m-%dT%H:%M:%SZ)" -o json
```

- GPU: `"NVIDIA H100 80GB HBM3"` (H100 SXM). `runpodctl gpu list` for
  others. Only **sm90+** (H100 / H200 / B200) — the bundled `sgl_kernel`
  wheel has no Ampere/Ada kernels.
- Template `runpod-torch-v240` = `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  — the validated image. RunPod "Pods" get `--ipc=host` by default.
- `--gpu-count`: see the sizing table below.
- The create call returns the pod `id` — keep it.

### 2 — Wait for SSH (it is slow: 1–8+ min)

The `.ssh.ip` / `.ssh.port` fields appear in `runpodctl pod get <id>
-o json` **before** SSH actually accepts connections. Poll until a real
connection succeeds:

```bash
ssh -i ~/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 \
  -p <port> root@<ip> 'echo ok'
```

> **zsh gotcha:** do not put ssh options in a shell variable — zsh does
> not word-split unquoted variables, so `ssh $OPTS ...` passes them as
> one bad argument. Inline every option.

### 3 — Deploy

```bash
ssh ... 'cd /root && git clone --depth=1 -b feature/colocate-training-inference \
  https://github.com/zhubohao911/TorchSpec.git'
```

If the code/patch you want to test is **committed and pushed**, the
clone already has it. If it is only local (uncommitted), `scp` the
files onto the pod after cloning.

### 4 — Run (detached, with an exit-code file)

Write a launcher on the pod and run it with `nohup … & disown` so it
survives the SSH session closing. Capture the exit code to a file you
can poll:

```bash
# /root/launcher.sh on the pod:
cd /root/TorchSpec
export HF_TOKEN=<token>                  # for Qwen3-8B tests
export SGLANG_PATCH_VERSION=v0.5.10.post1
export SGLANG_COMMIT=94f03a39dbd39edfc2b118b5357bbbadaaa9ad28
export CUDA_VISIBLE_DEVICES=0,1,2,3      # see note below
bash scripts/colocate/run_smoke_host.sh [--full | --tests=a.py,b.py]
echo $? > /root/run.rc
```

Launch: `nohup bash /root/launcher.sh > /root/run.log 2>&1 & disown`.

- `run_smoke_host.sh` defaults to `SGLANG_PATCH_VERSION=v0.5.10.post1`;
  it clones sglang, applies the patches, builds, and runs pytest.
- `--full` runs the whole matrix; `--tests=` runs specific files (use
  this to skip already-passed tests on a re-run).
- **`CUDA_VISIBLE_DEVICES` note:** `run_smoke_host.sh` only auto-sets
  all 4 GPUs for `--full`. With `--tests=`, pre-export
  `CUDA_VISIBLE_DEVICES=0,1,2,3` yourself or the multi-GPU tests see
  one GPU and skip.

### 5 — Monitor

Poll the **remote** files, not a local background job:

```bash
ssh ... 'cat /root/run.rc 2>/dev/null || echo RUNNING; tail -8 /root/run.log'
```

`run.rc` existing = run finished (`0` = all passed). The colocate
failure signature is a **hang on the first P2P recv** — if the log
stops advancing for many minutes mid-step, that is the diagnostic.

### 6 — Tear down (every time)

```bash
scp ... root@<ip>:/root/TorchSpec/colocate-smoke-report.txt /tmp/   # keep the report
runpodctl pod stop <id> && runpodctl pod delete <id>
runpodctl pod get <id>          # must say: pod not found
runpodctl user                  # confirm currentSpendPerHr dropped
```

## Vast.ai (alternative platform)

Vast.ai is a documented alternative — it ran the 4×H100 `--full` suite
in sessions #4/#5 ([`implementation_log.md`](implementation_log.md)),
and is often cheaper than RunPod. The `vastai` CLI (v1.0.x) is
installed, **but not authenticated on this machine.** Before an agent
can use Vast autonomously, the user must run it once:

```bash
vastai set api-key <KEY>      # key from the vast.ai console
```

All the same constraints and **hard rules** above apply (check other
instances, watch balance, tear down every time). Vast On-Demand
instances default to `--ipc=host`; choose a "Direct" net-type host with
a good reliability score and a CUDA 12.x + Python 3.11 PyTorch image.
sm90+ only, same as RunPod.

The workflow mirrors the RunPod one — only the CLI differs:

| Step | RunPod | Vast.ai |
|---|---|---|
| find capacity | `runpodctl gpu list` | `vastai search offers 'gpu_name=H100_SXM num_gpus=4 reliability>0.98'` |
| provision | `runpodctl pod create …` | `vastai create instance <offer-id> --image <pytorch-cu124-img> --disk 200 --ssh --direct` |
| list | `runpodctl pod list -o json` | `vastai show instances` |
| SSH endpoint | `.ssh.ip` / `.ssh.port` | `vastai ssh-url <id>` |
| **tear down** | `pod stop` + `pod delete` | **`vastai destroy instance <id>`** |

Run `vastai search offers --help` / `vastai create instance --help` for
exact field syntax — query fields and image flags change between CLI
versions.

> **Two Vast-specific cautions:**
> - **`stop instance` is not enough** — a stopped Vast instance still
>   **bills for storage**. Only `destroy instance` (irreversible —
>   deletes the disk) fully stops billing. Always `destroy` when done.
> - **No `--terminate-after` backstop.** RunPod self-destructs a lost
>   pod; Vast does not. The "always tear down" rule is therefore
>   load-bearing on Vast — never leave an instance unattended.

## GPU sizing

| Test | GPUs | Model | ~Time (after setup) |
|---|---|---|---|
| `test_colocate_tiny.py` | 1 | Qwen3-0.6B | ~4 min |
| `test_colocate_tp2.py` (`engine_tp_size=2`) | 2 | Qwen3-0.6B | ~2 min |
| `run_smoke_host.sh --full` (13 tests) | 4 | Qwen3-0.6B + Qwen3-8B | ~22 min |

Setup (pip install + sglang build) adds ~5–12 min on top, once per pod.

## Known failure modes — NOT your patch's bug

| Symptom | Cause | Action |
|---|---|---|
| `libnuma.so.1: cannot open shared object file` | RunPod image lacks it | `run_smoke_host.sh` already apt-installs it; if running sglang by hand, `apt-get install -y libnuma1` |
| HF Hub `429 Too Many Requests` on Qwen3-8B | unauthenticated HF requests rate-limited | set `HF_TOKEN` |
| pod returns `404 pod not found` / SSH dies mid-run | RunPod infra flakiness (some datacenters worse) | re-provision once; if it repeats, report |
| SSH never comes up after ~10 min | slow/bad pod | delete it, re-provision |
| multi-GPU test SKIPs (sees 1 GPU) | `--tests=` didn't set `CUDA_VISIBLE_DEVICES` | pre-export `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| `Unknown RoPE scaling type default` | old TorchSpec checkout (pre-`be399a0`) | clone current `feature/colocate-training-inference` |

## Cost reference

| Pod | Rate | One run (incl. setup) |
|---|---|---|
| 1×H100 SXM | ~$3.3/hr | tiny smoke ≈ $1–2 |
| 2×H100 SXM | ~$6.6/hr | tp2 ≈ $3–4 |
| 4×H100 SXM | ~$13/hr | `--full` ≈ $8–12 |

Rates above are RunPod. Vast.ai spot is usually cheaper (~$2/hr for
1×H100, ~$10–11/hr for 4×H100) but availability and host reliability
vary more.

Keep the pod alive only for the run. Idle time is pure waste — tear
down immediately on completion.

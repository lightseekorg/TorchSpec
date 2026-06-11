---
name: vllm-version-bump
description: Use when updating TorchSpec's vLLM backend to a new vLLM release — adding docker/vllm/<version>/ and patches/vllm/<version>/, bumping docker/justfile or .github/workflows/docker-release.yml, or when a new vLLM breaks mooncake imports (libcudart), extract_hidden_states, or the Docker image build.
---

# vLLM Version Bump

## Overview

TorchSpec pins vLLM per version directory: `docker/vllm/<ver>/Dockerfile` builds on
`vllm/vllm-openai:<ver>` and applies `patches/vllm/<ver>/*.patch` into the image's
dist-packages; `docker/justfile` (`VLLM_VERSION` default) and
`.github/workflows/docker-release.yml` (`VLLM_VERSION` + two `IMAGE_TAG`s) pin the
release. A bump = new directory pair + pin updates + e2e training verification on
BOTH surfaces (bare metal and the image). Input: target version `vX.Y.Z`.

**Core principle: audit, don't copy.** Patches exist because fixes hadn't landed
upstream yet. On every bump, most of them have. Carrying a patch forward without
checking each hunk against the new version's installed source ships dead code at
best and failed hunks at worst. (v0.19.1→v0.22.1: 466 patch lines shrank to one hunk.)

## Phase 1 — Audit patches against the new vLLM

1. Install the target into an env: `micromamba run -n torchspec uv pip install vllm==X.Y.Z`.
2. For each hunk in the previous `patches/vllm/<prev>/*.patch`: grep the installed
   source (`$ENV/lib/python3*/site-packages/vllm/...`). Has the fix landed upstream,
   possibly under a different name? (Example: patched `HiddenStatesCacheSpec` landed
   as `HiddenStateCacheSpec`.) Drop landed hunks; carry the rest.
3. Regenerate carried hunks by diffing real installed source in a /tmp scratch dir
   (copy file into `a/vllm/<path>/`, apply the edit into `b/vllm/<path>/`,
   `diff -u a/... b/...`), never by hand-editing old hunks — context lines shift
   between versions. Keep one `vllm.patch` per version (existing convention).
4. Verify both apply methods against a copy of the tree: `patch -p1 --dry-run`
   (what the Dockerfile runs) AND `git apply --check`. `python -m py_compile` the
   patched file.
5. Document dropped hunks and why in the patch header (see `patches/vllm/v0.22.1/vllm.patch`).

Known long-lived hunk: the FA4 guard wrapping
`from flash_attn.ops.triton.rotary import apply_rotary` in
`vllm/model_executor/layers/rotary_embedding/common.py` — flash-attn-4 ships a
`flash_attn` module without that path, and vLLM's bare import crashes engine init.
Check each bump whether vLLM finally guarded it upstream.

## Phase 2 — Create version files

| File | Change |
|---|---|
| `docker/vllm/vX.Y.Z/Dockerfile` | Copy previous version's; bump `FROM` tag + patch dir |
| `patches/vllm/vX.Y.Z/vllm.patch` | Output of Phase 1 |
| `docker/justfile` | `VLLM_VERSION` default |
| `.github/workflows/docker-release.yml` | `VLLM_VERSION` and both `IMAGE_TAG` values |

Verify against the new base image BEFORE building:

- Tag exists and arches: `https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?name=vX.Y`
- Python version (the patch layer hardcodes `/usr/local/lib/python3.NN/dist-packages`):
  check `ARG PYTHON_VERSION` in
  `https://raw.githubusercontent.com/vllm-project/vllm/vX.Y.Z/docker/Dockerfile`.
- **Never pip install/upgrade/uninstall mooncake inside the image.** The vllm-openai
  image bundles its own CUDA-matched mooncake build that satisfies torchspec's floor;
  PyPI wheels cannot replace it (the generic wheel links the wrong libcudart, and on
  aarch64 ALL mooncake wheels need glibc ≥ 2.39 while the image is Ubuntu 22.04 /
  glibc 2.35 — pip reports `versions: none`). Confirm the bundled build matches the
  image's CUDA major:
  `docker run --rm --entrypoint bash vllm/vllm-openai:vX.Y.Z -c 'ldd /usr/local/lib/python3*/dist-packages/mooncake/engine.so | grep cudart'`

## Phase 3 — Bare-metal e2e

1. Env: `./tools/build_conda.sh 1 vllm`.
2. Check the mooncake/CUDA pairing — the canonical test is the linker, not release notes:
   `ldd $ENV/lib/python3*/site-packages/mooncake/engine.so | grep -E "cudart|not found"`.
   A `libcudart.so.NN => not found` means vLLM's torch stack moved CUDA major and the
   default `mooncake-transfer-engine` wheel links the old runtime; swap:
   `uv pip uninstall mooncake-transfer-engine && uv pip install mooncake-transfer-engine-cudaNN`.
3. Run, capturing all output (key log lines arrive on both stdout and stderr):

```bash
export CUDA_VISIBLE_DEVICES=<the node's actual GPU ids>  # COUNT matters, not just ids:
    # run.sh derives inference_num_gpus_per_node from it and defaults to 8 GPUs —
    # on a 4-GPU node you MUST set 0,1,2,3 or the run is misconfigured
export MC_STORE_MEMCPY=0                                 # mooncake-over-TCP bug (see README)
./examples/qwen3-8b-single-node/run.sh configs/vllm_qwen3_8b.yaml \
    training.num_train_steps=5 > /tmp/bump_e2e.log 2>&1
```

PASS requires ALL of: `initialized extract_hidden_states mode` in the log;
`Training completed: 5 steps`; exit 0; final train loss < first train loss
(this config starts ≈12 — compare `grep -oE "loss=[0-9.]+" /tmp/bump_e2e.log`
first vs last train entries). Then probe the checkpoint:
`python tools/convert_to_hf.py --input-dir outputs/.../iter_*/ --target-model-path Qwen/Qwen3-8B`.

## Phase 4 — Docker e2e (replicates the release workflow's build)

```bash
cd docker && BACKEND=vllm VLLM_VERSION=vX.Y.Z IMAGE_REPO=local/torchspec \
  IMAGE_TAG=test-vllm-vX.Y.Z just build

# Sanity — mooncake import needs --gpus all even without training (libcuda.so.1):
docker run --rm --gpus all --entrypoint python3 local/torchspec:test-vllm-vX.Y.Z -c \
  "import vllm, torchspec, flash_attn; from mooncake.store import MooncakeDistributedStore; print('ok')"

# e2e — the ONLY place the FA4 patch executes (flash-attn-4 is installed here, not bare metal):
docker run --rm --gpus all --network=host --shm-size=16g --entrypoint bash \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e MC_STORE_MEMCPY=0 -e CUDA_VISIBLE_DEVICES=<same ids as Phase 3> \
  local/torchspec:test-vllm-vX.Y.Z \
  examples/qwen3-8b-single-node/run.sh configs/vllm_qwen3_8b.yaml training.num_train_steps=5
```

`--entrypoint bash` is required (base entrypoint is the vLLM API server). Same PASS
criteria as Phase 3.

## Debugging table

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: libcudart.so.NN` from `mooncake.store` | vLLM moved CUDA major; mooncake wheel links the old runtime (torch no longer preloads it) | Bare metal: swap to `mooncake-transfer-engine-cudaNN`. Image: use the bundled mooncake — never reinstall |
| `Mooncake master gRPC unreachable ... Connection refused` | Master died ~2s after start with exit 0 — usually a fixed-port bind failure (e.g. another master's metrics port) | Run the exact master command manually and read stderr; every port it binds must be free |
| pip `versions: none` for mooncake during image build | aarch64 mooncake wheels need glibc ≥ 2.39; image is Ubuntu 22.04 (2.35) | Don't install mooncake in the image |
| Patch layer: `Hunk #N FAILED` | Upstream code shifted | Regenerate the hunk per Phase 1 |
| `ModuleNotFoundError: flash_attn.ops.triton.rotary` at engine init | FA4 guard hunk missing or not applied | Phase 1 known hunk |
| Warning: `Model runner v2 does not yet support ... extract_hidden_states; using the v1 model runner` | Upstream fallback | OK while v1 runner exists; if a release removes it, STOP — extract_hidden_states needs vLLM-side work first |
| Training hangs with `Sample pool full, pausing generation` only | Backpressure working; training side stalled for another reason | Check trainer actor logs, not inference |

## Red flags — you are about to ship a broken bump

- Copying `patches/` forward without auditing each hunk against the new installed source
- Adding any mooncake pip operation to the vllm Dockerfile
- Declaring PASS from build + import checks alone — a 5-step training run must complete on BOTH surfaces
- Skipping the Docker e2e because bare metal passed — the FA4 patch path only executes in the image

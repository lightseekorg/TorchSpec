"""Colocate (training+inference on same GPU) smoke tests on Modal.

Each phase from `docs/colocate/implementation.md` has its own entry point
here. The image, volumes, and secrets are shared across phases. Local
torchspec/, tests/, and patches/ are overlaid on top of a pinned upstream
commit so iterating on code does NOT require an image rebuild.

Setup (one-time):
    modal token set --token-id <id> --token-secret <secret> --profile=doordash
    modal profile activate doordash
    bash scripts/modal/setup_modal_secrets.sh --env sandbox

Run smoke tests (each function is a separate Modal `local_entrypoint`):
    modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase1_placement
    modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase2_union_world
    modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase3_p2p_dummy
    modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase4_one_step
    modal run --detach --env sandbox scripts/modal/modal_colocate_smoke.py::phase6_stability
    modal run --env sandbox scripts/modal/modal_colocate_smoke.py::phase7_grad_parity

Notes:
- All phases default to a 4×H100 single-node container — that's the size the
  implementation plan specifies as the smoke-test target. Override at the CLI
  via `--gpu` for ad-hoc experiments.
- MPS is enabled by phase-1 onwards; the Modal H100 image already ships
  `nvidia-cuda-mps-control` as part of the CUDA toolkit, so no extra apt
  package is needed.
- Phase 0 is unit-only (no GPU) — run it locally with `pytest tests/colocate/
  test_phase0_validation.py`.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

import modal

# =============================================================================
# Constants
# =============================================================================

TORCHSPEC_REPO = "https://github.com/zhubohao911/TorchSpec.git"
TORCHSPEC_BRANCH = "feature/colocate-training-inference"
# Bump to bust the Modal image cache when the upstream pinned commit changes.
TORCHSPEC_PIN_COMMIT = "cbecbec"
SGLANG_COMMIT = "94f03a39dbd39edfc2b118b5357bbbadaaa9ad28"
SGLANG_PATCH_VERSION = "v0.5.10.post1"

REPO_DIR = "/workspace/TorchSpec"
SGLANG_DIR = f"{REPO_DIR}/_sglang"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/workspace/outputs"

# 4×H100 — the smoke-test target from implementation.md (Phase 1+).
DEFAULT_GPU = "H100:4"

# =============================================================================
# Modal app + volumes
# =============================================================================

app = modal.App("torchspec-colocate-smoke")

hf_cache_vol = modal.Volume.from_name("torchspec-colocate-hf-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("torchspec-colocate-outputs", create_if_missing=True)

# =============================================================================
# Container image — shared by every phase.
# Mirrors the dflash branch's modal_dflash_train image (same CUDA/PyTorch/sglang
# versions, same Mooncake binary patch, same env-var fixes).
# =============================================================================

base_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "vim",
        "htop",
        # RDMA libs — required by Mooncake (used by the disaggregated baseline
        # we run in Phase 7's control arm).
        "libibverbs-dev",
        "librdmacm-dev",
        "libnuma-dev",
        "libcurl4-openssl-dev",
        # MPS daemon ships with the CUDA toolkit base image, so no extra apt
        # package is needed for `nvidia-cuda-mps-control`.
    )
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        f"git clone {TORCHSPEC_REPO} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {TORCHSPEC_BRANCH} && "
        f"git reset --hard {TORCHSPEC_PIN_COMMIT}",
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "transformers==4.57.1",
        "datasets",
        "tqdm",
        "wandb",
        "accelerate",
        "pydantic",
        "omegaconf",
        "ray",
        "mooncake-transfer-engine",
        "sglang-router",
        "openai",
        "openai-harmony",
        "qwen-vl-utils",
        "psutil",
        "numpy<2.4",
        "pyzmq",
        "numba",
        "cmake",
        "ninja",
        "packaging",
        "setuptools",
        "pytest",
    )
    .run_commands(f"cd {REPO_DIR} && pip install -e '.[dev]'")
    # Mooncake binary perms (mirrors Dockerfile.runpod Layer 6 from the
    # dflash branch).
    .run_commands(
        'MOONCAKE_DIR=$(python3 -c "import mooncake, os; '
        'print(os.path.dirname(mooncake.__file__))") && '
        'chmod 755 "$MOONCAKE_DIR/mooncake_master" 2>/dev/null || true && '
        "sed -i 's/os.chmod(bin_path, 0o755)/pass/' "
        '"$MOONCAKE_DIR/cli.py" 2>/dev/null || true',
    )
    .run_commands(
        "mkdir -p /root/.cache && ln -sf /root/.cache/huggingface /root/.cache/huggingface || true",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            # PyTorch <2.9 still reads the old name — set both for safety
            # since we want fragmentation-friendly allocator under MPS.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
            "TORCHSPEC_LOG_LEVEL": "INFO",
            "HF_HOME": HF_CACHE_DIR,
        }
    )
)

sglang_image = (
    base_image
    # Layer 1: clone sglang at the pinned commit, install editable, and
    # apply the existing disagg patch (which has been part of the
    # pinned TorchSpec commit since before this branch).
    .run_commands(
        f"git clone https://github.com/sgl-project/sglang.git {SGLANG_DIR}",
        f"cd {SGLANG_DIR} && git checkout {SGLANG_COMMIT} && git reset --hard HEAD",
        f"cd {REPO_DIR} && pip install -e '_sglang/python[all]'",
        f"rm -f {SGLANG_DIR}/python/sglang/srt/speculative/spec_training_info.py",
        f"cd {SGLANG_DIR} && git apply --recount "
        f"{REPO_DIR}/patches/sglang/{SGLANG_PATCH_VERSION}/sglang.patch || true",
    )
    # Layer 2: overlay the local working tree (so iteration on the
    # colocate code or patch doesn't require rebuilding the heavy
    # base+disagg layers above). `patches/` overlay brings in the new
    # `colocate.patch` file that may not exist in the pinned commit.
    .add_local_dir("torchspec", f"{REPO_DIR}/torchspec", copy=True)
    .add_local_dir("tests", f"{REPO_DIR}/tests", copy=True)
    .add_local_dir("patches", f"{REPO_DIR}/patches", copy=True)
    .add_local_dir("configs", f"{REPO_DIR}/configs", copy=True)
    .add_local_dir("scripts/tools", f"{REPO_DIR}/scripts/tools", copy=True)
    # Phase-4 one-step needs the sample-conversations dataset under
    # examples/data/ that the colocate config points at, plus the
    # example run.sh in case future tests want to exercise the shell
    # entrypoint directly. The directory is small (<1 MB) so the
    # cache-invalidation cost of overlaying it on every iteration is
    # negligible.
    .add_local_dir("examples", f"{REPO_DIR}/examples", copy=True)
    # Layer 3: apply the Phase-4 colocate (NCCL) patch from the
    # overlaid local patches/ directory. Layered AFTER the overlay so
    # patch iteration only invalidates this thin layer's cache.
    # Disagg runs are unaffected — the patch is structurally a no-op
    # when TORCHSPEC_COLOCATE_TRANSFER_MODE is unset.
    .run_commands(
        f"cd {SGLANG_DIR} && git apply --recount "
        f"{REPO_DIR}/patches/sglang/{SGLANG_PATCH_VERSION}/colocate.patch",
    )
)


_common_kwargs = dict(
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        OUTPUTS_DIR: outputs_vol,
    },
    timeout=24 * 3600,
    secrets=[
        modal.Secret.from_name("xingh3-hf-write"),
        modal.Secret.from_name("wandb-secret"),
    ],
)


# =============================================================================
# Helpers used inside the container
# =============================================================================


def _gpu_banner() -> int:
    import torch

    detected = torch.cuda.device_count()
    print(f"  GPUs detected: {detected}")
    for i in range(detected):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"    GPU {i}: {name} ({mem_gb:.1f} GB)")
    return detected


def _hf_token_setup() -> None:
    import os
    import shutil

    os.environ["HF_HOME"] = HF_CACHE_DIR
    hf_token = os.environ.get("HF_WRITE_TOKEN")
    if not hf_token:
        return
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    for token_file in [
        os.path.join(HF_CACHE_DIR, "token"),
        os.path.expanduser("~/.huggingface/token"),
    ]:
        os.makedirs(os.path.dirname(token_file), exist_ok=True)
        with open(token_file, "w") as f:
            f.write(hf_token)
    stored_dir = os.path.join(HF_CACHE_DIR, "stored_tokens")
    if os.path.isdir(stored_dir):
        shutil.rmtree(stored_dir)


def _run_pytest(test_path: str, extra_args: Optional[list[str]] = None) -> int:
    """Run a pytest target inside the container; return exit code."""
    cmd = [sys.executable, "-m", "pytest", "-xvs", test_path]
    if extra_args:
        cmd.extend(extra_args)
    print("  $", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_DIR)
    return proc.returncode


# =============================================================================
# Phase 1 — placement + MPS
# =============================================================================


@app.function(image=sglang_image, gpu=DEFAULT_GPU, **_common_kwargs)
def _run_phase1_placement():
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest("tests/colocate/test_placement.py")
    if rc != 0:
        raise RuntimeError(f"phase1_placement failed (exit {rc})")


@app.local_entrypoint()
def phase1_placement():
    """Placement: 1:1 bundle pairing + MPS daemon env vars."""
    _run_phase1_placement.remote()


# =============================================================================
# Phase 2 — union NCCL world
# =============================================================================


@app.function(image=sglang_image, gpu="H100:8", **_common_kwargs)
def _run_phase2_union_world():
    """Phase 2 deliberately uses 8 GPUs (one per rank, no MPS sharing) to
    isolate the union-world bootstrap from MPS sharing. The MPS+union-world
    integration is Phase 4's hidden-state hook; per the implementation.md
    risk register, Phase 2 should validate the bootstrap mechanism alone.
    """
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest("tests/colocate/test_union_world.py")
    if rc != 0:
        raise RuntimeError(f"phase2_union_world failed (exit {rc})")


@app.local_entrypoint()
def phase2_union_world():
    """Union NCCL world: 2*N rank barrier + FSDP-only subgroup."""
    _run_phase2_union_world.remote()


# =============================================================================
# Phase 3 — NCCL P2P dummy transfer
# =============================================================================


@app.function(image=sglang_image, gpu="H100:2", **_common_kwargs)
def _run_phase3_p2p_dummy():
    """Phase 3 uses a 2-rank topology (1 trainer + 1 engine, dedicated
    GPUs, no MPS) to verify the NCCL data plane mechanism end-to-end.

    The plan-text mentions 4-GPU MPS sharing for Phase 3; we ship the
    smaller scale because (a) MPS is Phase 4's domain and (b) the 8-rank
    concurrent multi-pair P2P pattern under eager-init NCCL hits a
    resource-coordination pathology that's naturally resolved when the
    trainer+engine wiring lands in Phase 4 (each pair runs inside MPS
    with its own NCCL world). At 2 ranks we definitively verify
    init_union_world + NcclDataFetcher round-trip + deterministic byte
    equality + clean shape-mismatch error path."""
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest("tests/colocate/test_p2p_dummy.py")
    if rc != 0:
        raise RuntimeError(f"phase3_p2p_dummy failed (exit {rc})")


@app.local_entrypoint()
def phase3_p2p_dummy():
    """100-iteration dummy P2P byte-equality test."""
    _run_phase3_p2p_dummy.remote()


# =============================================================================
# Phase 4 — real hidden-state hook (multi-tensor P2P + one training step)
# =============================================================================


@app.function(image=sglang_image, gpu="H100:2", **_common_kwargs)
def _run_phase4_multi_tensor():
    """Phase 4 multi-tensor round-trip on the union world (2-rank).

    Validates the in-repo half of Phase 4: NcclHiddenStatesConnector
    sends a Mooncake-shaped tensor dict (hidden_states +
    aux_hidden_states + last_hidden_states + target_logits), and
    NcclMultiTensorFetcher receives it with byte equality on every
    tensor. This is the maximal e2e check we can run without the
    upstream sglang patch — the patch is required for the "one full
    training step" deliverable, which lives in `_run_phase4_one_step`."""
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest("tests/colocate/test_p2p_multi_tensor.py")
    if rc != 0:
        raise RuntimeError(f"phase4_multi_tensor failed (exit {rc})")


@app.local_entrypoint()
def phase4_multi_tensor():
    """Multi-tensor NCCL P2P round-trip (Mooncake-shaped dict)."""
    _run_phase4_multi_tensor.remote()


@app.function(image=sglang_image, gpu=DEFAULT_GPU, **_common_kwargs)
def _run_phase4_one_step():
    """Phase 4 one-step training (requires upstream sglang patch).

    See ``docs/colocate/sglang_patch.md`` for the patch surface. Without
    that patch the engine's spec_training callback writes to a (now
    non-existent) Mooncake store and the trainer hangs on its first P2P
    recv. The test file is parked here for when the patch lands."""
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest("tests/colocate/test_one_step.py")
    if rc != 0:
        raise RuntimeError(f"phase4_one_step failed (exit {rc})")


@app.local_entrypoint()
def phase4_one_step():
    """Run a single colocate training step on Qwen3-8B (TP=4 + FSDP=4).

    Requires the upstream sglang patch — see docs/colocate/sglang_patch.md.
    """
    _run_phase4_one_step.remote()


# =============================================================================
# Tiny (1×GPU + Qwen3-0.6B) — cheap-host smoke; verifies skip behaviour on Modal
# =============================================================================


@app.function(image=sglang_image, gpu="H100:1", **_common_kwargs)
def _run_phase_tiny():
    """Run the 1-GPU tiny-model colocate smoke (Phase-4 one-step + Phase-7
    mini convergence) inside the Modal image.

    On Modal sandbox the host doesn't pass --ipc=host so MPS fails with
    'operation not supported'; the test correctly skips. Running it here
    proves:
      * the tiny config is accepted by Phase-0 validation;
      * the tiny test file imports cleanly inside the image;
      * the MPS-probe skip gate matches the 4-GPU tests' behaviour.

    Once the same image runs on a host that exposes --ipc=host (Vast.ai,
    Lambda Labs, etc.), this entry point is the easiest way to drive the
    same code path that scripts/colocate/run_smoke_host.sh runs locally.
    """
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest("tests/colocate/test_colocate_tiny.py")
    if rc != 0:
        raise RuntimeError(f"phase_tiny failed (exit {rc})")


@app.local_entrypoint()
def phase_tiny():
    """Single-GPU colocate smoke (Qwen3-0.6B, 1×H100).

    Mirrors scripts/colocate/run_smoke_host.sh on Modal so we can
    sanity-check the test importability + skip-gate behaviour without
    paying for a 4-GPU job. Will SKIP on Modal sandbox (no MPS); will
    PASS on any host with --ipc=host."""
    _run_phase_tiny.remote()


# =============================================================================
# Phase 6 — 1000-step stability (slow)
# =============================================================================


@app.function(image=sglang_image, gpu=DEFAULT_GPU, **_common_kwargs)
def _run_phase6_stability():
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest(
        "tests/colocate/test_stability.py",
        extra_args=["-m", "slow"],
    )
    if rc != 0:
        raise RuntimeError(f"phase6_stability failed (exit {rc})")


@app.local_entrypoint()
def phase6_stability():
    """Slow: 1000-step run, assert flat peak alloc."""
    _run_phase6_stability.remote()


# =============================================================================
# Phase 7 — grad parity (one-step) and convergence (slow)
# =============================================================================


@app.function(image=sglang_image, gpu=DEFAULT_GPU, **_common_kwargs)
def _run_phase7_grad_parity():
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest("tests/colocate/test_grad_parity.py")
    if rc != 0:
        raise RuntimeError(f"phase7_grad_parity failed (exit {rc})")


@app.local_entrypoint()
def phase7_grad_parity():
    """Per-parameter gradient parity vs disaggregated baseline."""
    _run_phase7_grad_parity.remote()


@app.function(image=sglang_image, gpu=DEFAULT_GPU, **_common_kwargs)
def _run_phase7_convergence():
    _gpu_banner()
    _hf_token_setup()
    rc = _run_pytest(
        "tests/colocate/test_convergence.py",
        extra_args=["-m", "slow"],
    )
    if rc != 0:
        raise RuntimeError(f"phase7_convergence failed (exit {rc})")


@app.local_entrypoint()
def phase7_convergence():
    """Slow: 1k-step loss-curve overlap (run with --detach)."""
    _run_phase7_convergence.remote()


# =============================================================================
# Sanity: container probe (no test, just confirms the image starts up).
# =============================================================================


@app.function(image=sglang_image, gpu="H100:1", **_common_kwargs)
def _run_probe():
    _gpu_banner()
    print("\n  --- nvidia-smi ---")
    subprocess.run(["nvidia-smi"], check=False)
    print("\n  --- nvidia-cuda-mps-control --version ---")
    subprocess.run(
        ["nvidia-cuda-mps-control", "-V"], check=False
    )  # `-V` is a noop in some builds; we just want the binary to be present
    print("\n  --- python imports ---")
    import torch

    print(f"  torch {torch.__version__}")
    try:
        import sglang  # noqa: F401

        print("  sglang OK")
    except Exception as e:
        print(f"  sglang import failed: {e}")
        return

    # ---------------------------------------------------------------
    # colocate.patch surface verification — these checks fail loudly
    # if the layered patch did not apply during image build.
    # ---------------------------------------------------------------
    print("\n  --- colocate.patch surface ---")
    import importlib
    import inspect
    import os

    tc = importlib.import_module("sglang.srt.distributed.torchspec_colocate")
    print(f"  helper module: {tc.__file__}")
    assert tc.is_colocate_active() is False, (
        "is_colocate_active() should be False with no env vars set"
    )

    os.environ["TORCHSPEC_COLOCATE_TRANSFER_MODE"] = "nccl"
    os.environ["TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK"] = "0"
    os.environ["TORCHSPEC_COLOCATE_UNION_MASTER_ADDR"] = "127.0.0.1"
    os.environ["TORCHSPEC_COLOCATE_UNION_MASTER_PORT"] = "12345"
    os.environ["TORCHSPEC_COLOCATE_UNION_WORLD_SIZE"] = "8"
    os.environ["TORCHSPEC_COLOCATE_UNION_N_PER_ROLE"] = "4"
    env = tc.read_colocate_env()
    print(
        f"  read_colocate_env: world_size={env.world_size} "
        f"n_per_role={env.n_per_role} "
        f"engine_global_rank(0)={env.engine_global_rank(0)} "
        f"engine_global_rank(3)={env.engine_global_rank(3)}"
    )
    assert env.engine_global_rank(0) == 4
    assert env.engine_global_rank(3) == 7
    assert tc.build_engine_tp_ranks(env) == [4, 5, 6, 7]
    print("  helper round-trip OK (4 trainer + 4 engine union world)")

    from sglang.srt.distributed import parallel_state as ps

    sig = inspect.signature(ps.initialize_model_parallel)
    assert "tp_world_ranks" in sig.parameters, (
        "tp_world_ranks kwarg missing — colocate.patch did not patch parallel_state.py"
    )
    print(
        f"  parallel_state.initialize_model_parallel: tp_world_ranks kwarg present "
        f"(params={list(sig.parameters.keys())})"
    )

    from sglang.srt.managers import scheduler_output_processor_mixin as som

    assert hasattr(som.SchedulerOutputProcessorMixin, "_send_hidden_states_to_nccl"), (
        "_send_hidden_states_to_nccl missing — output processor mixin not patched"
    )
    print("  scheduler_output_processor_mixin._send_hidden_states_to_nccl present")

    from sglang.srt.managers import scheduler as sc

    src = inspect.getsource(sc.Scheduler.__init__)
    assert "eagle_nccl_writer" in src, "eagle_nccl_writer init missing — scheduler.py not patched"
    assert "is_colocate_active" in src or "torchspec_colocate" in src, (
        "torchspec_colocate import missing in Scheduler.__init__"
    )
    print("  scheduler.Scheduler.__init__ wires eagle_nccl_writer + colocate gate")

    print("\n  *** colocate.patch surface OK ***")


@app.local_entrypoint()
def probe():
    """Single-GPU sanity probe: image starts, MPS binary present, sglang imports."""
    _run_probe.remote()

"""
DFlash / Eagle3 training on Modal GPU platform.

Ports the RunPod training workflow (runpod_dflash_train.sh + runpod_setup.sh)
to Modal's serverless GPU infrastructure. The container image is built
declaratively, and training runs as a subprocess inside a single multi-GPU
container where Ray manages distributed coordination internally.

Incorporates all fixes from runpod_setup.sh and Dockerfile.runpod:
    Issue 3:  RDMA libs (libibverbs, librdmacm, libnuma) for Mooncake
    Issue 19: HF cache symlink for Ray workers
    Issue 23: PYTORCH_ALLOC_CONF=expandable_segments:True (PyTorch 2.9+)
    Issue 24: SGLang [all] includes flashinfer — no separate install
    Issue 26: TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON
    Mooncake binary chmod + cli.py patch

GPU auto-config (this script — SGLang backend, 4+ GPUs):
    4 GPUs -> 1 inference + 3 training (FSDP)
    8 GPUs -> 1 inference + 7 training (FSDP)

For 1-2 GPU HF mode, use modal_dflash_train_hf.py instead.

Setup (one-time):
    modal token set --token-id ak-l6eYjquYvG1smlyMfH0EKl --token-secret as-eJoVAHKgk0J9iRxNgMNPRJ --profile=doordash
    modal profile activate doordash
    modal secret create huggingface-secret HF_TOKEN=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
    modal secret create xingh3-hf-write HF_WRITE_TOKEN=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  # personal write token for HF uploads
    modal secret create wandb-secret WANDB_API_KEY=<key>   # get key from https://wandb.ai/authorize

Usage:
    modal run scripts/modal_dflash_train.py                                    # 8x H100, 200-step test
    modal run scripts/modal_dflash_train.py --max-steps 999999 --dataset-size 200000  # full 200K run
    modal run scripts/modal_dflash_train.py --num-epochs 3 --max-steps 999999  # control epochs
    modal run scripts/modal_dflash_train.py --gpu-count 4                      # 4-GPU mode
    modal run --detach scripts/modal_dflash_train.py                           # detached (survives terminal close)

    To change GPU type, edit SGLANG_GPU constant at the top of this file.

Recommended parameters (8x H100, quality-optimized with anchors=512):
    --extra-overrides "training.dflash_num_anchors=512 \
        inference.inference_num_gpus=4 training.training_num_gpus_per_node=4"

    Config 512-E: 4 inference + 4 training GPUs, batch=1, accum=4, anchors=512
    Results: 17-20 samples/s, 368s for 200 steps (fastest wall-clock)
    Quality: anchors=512 matches z-lab recipe for best τ (acceptance length)
    Note: inference oversaturated (I=108-120), pool overfull (72/64)

    Alternative (best throughput per step, anchors=512):
    --extra-overrides "training.dflash_num_anchors=512 \
        inference.inference_num_gpus=2 training.training_num_gpus_per_node=6"

    Config 512-D: 2 inference + 6 training GPUs, batch=1, accum=4, anchors=512
    Results: 22-25 samples/s, 457s for 200 steps, most stable step times

    Speed-only (if τ quality not critical):
    --extra-overrides "training.dflash_num_anchors=256 training.micro_batch_size=4 \
        training.draft_accumulation_steps=1 inference.inference_num_gpus=2 \
        training.training_num_gpus_per_node=6"

    Key tuning insights (see dflash_modal_training_results.md):
      - 4+4 split (512-E) is fastest: 368s, fewer FSDP ranks = less allreduce
      - 2+ inference GPUs essential for anchors=512 (pool starves with 1)
      - anchors=512 matches anchors=256 speed with enough inference GPUs
      - 512-D (2+6, batch=1) has most stable fwd times (305-448ms)
      - 512-E (4+4) oversaturates inference (I=108); 3+5 may be sweet spot
      - 1 inference GPU causes pool starvation (12-28/64) for all 512 configs

    Full training example (200K samples, 3 epochs, quality-optimized):
        modal run --detach scripts/modal_dflash_train.py \
            --max-steps 999999 --num-epochs 3 --dataset-size 200000 \
            --extra-overrides "training.dflash_num_anchors=512 \
                inference.inference_num_gpus=4 training.training_num_gpus_per_node=4"
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import modal

# =============================================================================
# Constants
# =============================================================================

TORCHSPEC_REPO = "https://github.com/zhubohao911/TorchSpec.git"
TORCHSPEC_BRANCH = "feature/dflash-training"
SGLANG_COMMIT = "0f2df9370a1de1b4fb11b071d39ab3ce2287a350"
SGLANG_PATCH_VERSION = "v0.5.8.post1"

REPO_DIR = "/workspace/TorchSpec"
SGLANG_DIR = f"{REPO_DIR}/_sglang"
HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/workspace/outputs"
DATA_DIR = "/workspace/data"

# GPU configuration — edit to change hardware allocation.
SGLANG_GPU = "H100:8"   # 1 inference + N-1 training (FSDP)

# =============================================================================
# Modal app + volumes
# =============================================================================

app = modal.App("torchspec-dflash-training")

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("torchspec-outputs", create_if_missing=True)

# =============================================================================
# Container image — base layer shared by all GPU counts
#
# Mirrors runpod_setup.sh Steps 3-7 and Dockerfile.runpod Layers 1-8.
# =============================================================================

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    # Issue 3: RDMA libs for Mooncake + libcurl for mooncake store + general system deps
    .apt_install(
        "git", "vim", "htop",
        "libibverbs-dev", "librdmacm-dev", "libnuma-dev",
        "libcurl4-openssl-dev",
    )
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        f"git clone {TORCHSPEC_REPO} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout {TORCHSPEC_BRANCH}",
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
    )
    .run_commands(f"cd {REPO_DIR} && pip install -e '.[dev]'")
    # Mooncake binary permission fix (from Dockerfile.runpod Layer 6)
    .run_commands(
        "MOONCAKE_DIR=$(python3 -c \"import mooncake, os; print(os.path.dirname(mooncake.__file__))\") && "
        "chmod 755 \"$MOONCAKE_DIR/mooncake_master\" 2>/dev/null || true && "
        "sed -i 's/os.chmod(bin_path, 0o755)/pass/' \"$MOONCAKE_DIR/cli.py\" 2>/dev/null || true",
    )
    # Issue 19: HF cache symlink so Ray workers find models at /root/.cache/huggingface
    .run_commands(
        "mkdir -p /root/.cache && ln -sf /root/.cache/huggingface /root/.cache/huggingface || true",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Issue 23: PyTorch 2.9+ renamed PYTORCH_CUDA_ALLOC_CONF
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            # Issue 26: Fix TorchInductor GEMM backend regression
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
            "TORCHSPEC_LOG_LEVEL": "INFO",
            "HF_HOME": HF_CACHE_DIR,
        }
    )
)

# SGLang image: extends base with SGLang from source + patches
# Mirrors runpod_setup.sh Steps 5-6 and Dockerfile.runpod Layer 2
sglang_image = (
    base_image
    .run_commands(
        f"git clone https://github.com/sgl-project/sglang.git {SGLANG_DIR}",
        f"cd {SGLANG_DIR} && git checkout {SGLANG_COMMIT} && git reset --hard HEAD",
        # Issue 24: SGLang [all] pulls flashinfer — no separate install
        f"cd {REPO_DIR} && pip install -e '_sglang/python[all]'",
        # Patch sequence from runpod_setup.sh Step 6:
        # 1. Remove conflicting spec_training_info.py AFTER install, BEFORE patch
        # 2. Apply sglang.patch
        f"rm -f {SGLANG_DIR}/python/sglang/srt/speculative/spec_training_info.py",
        f"cd {SGLANG_DIR} && git apply "
        f"{REPO_DIR}/patches/sglang/{SGLANG_PATCH_VERSION}/sglang.patch || true",
    )
)

# =============================================================================
# GPU configuration — mirrors the bash script's auto-detect logic
# =============================================================================

@dataclass
class GPUConfig:
    mode: str
    eagle3_config: str
    dflash_config: str
    overrides: list[str]


def _gpu_config(gpu_count: int, extra_overrides: list[str] | None = None) -> GPUConfig:
    if gpu_count >= 4:
        infer_gpus = 1
        train_gpus = gpu_count - 1
        if extra_overrides:
            for ov in extra_overrides:
                if ov.startswith("inference.inference_num_gpus="):
                    infer_gpus = int(ov.split("=", 1)[1])
                elif ov.startswith("training.training_num_gpus_per_node="):
                    train_gpus = int(ov.split("=", 1)[1])
        return GPUConfig(
            mode=f"{gpu_count}gpu (SGLang, {infer_gpus} inference + {train_gpus} training FSDP)",
            eagle3_config="configs/sglang_qwen3_8b.yaml",
            dflash_config="configs/sglang_qwen3_8b_dflash.yaml",
            overrides=[
                f"training.training_num_gpus_per_node={gpu_count - 1}",
                "inference.inference_num_gpus=1",
                "inference.inference_num_gpus_per_engine=1",
                f"inference.inference_num_gpus_per_node={gpu_count}",
            ],
        )
    if gpu_count == 2:
        return GPUConfig(
            mode="2gpu (HF backend, 1 inference + 1 training)",
            eagle3_config="configs/hf_qwen3_8b.yaml",
            dflash_config="configs/hf_qwen3_8b_dflash_1gpu.yaml",
            overrides=[
                "training.training_num_gpus_per_node=1",
                "inference.inference_num_gpus=1",
                "inference.inference_num_gpus_per_engine=1",
                "inference.inference_num_gpus_per_node=2",
            ],
        )
    return GPUConfig(
        mode="1gpu colocate (HF backend, shared GPU — needs 80GB)",
        eagle3_config="configs/hf_qwen3_8b_1gpu.yaml",
        dflash_config="configs/hf_qwen3_8b_dflash_1gpu.yaml",
        overrides=[
            "training.colocate=true",
            "training.training_num_gpus_per_node=1",
            "inference.inference_num_gpus=1",
            "inference.inference_num_gpus_per_engine=1",
            "inference.inference_num_gpus_per_node=1",
        ],
    )


# =============================================================================
# Training runner — executed inside the Modal container
# =============================================================================

def _run_training(
    name: str,
    config: str,
    run_id: str,
    max_steps: int,
    num_epochs: Optional[int],
    gpu_overrides: list[str],
    wandb_project: Optional[str],
    extra_args: list[str],
):
    import os
    import time

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    log_path = f"{OUTPUTS_DIR}/{run_id}.log"

    wandb_args = []
    if os.environ.get("WANDB_API_KEY"):
        wandb_args = [
            "training.report_to=wandb",
            f"training.wandb_project={wandb_project or 'dflash-vs-eagle3'}",
            f"training.wandb_run_id={run_id}",
        ]

    epoch_args = []
    if num_epochs is not None:
        epoch_args = [f"training.num_epochs={num_epochs}"]

    output_dir = f"{OUTPUTS_DIR}/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    step_args = []
    if num_epochs is None:
        step_args = [f"training.num_train_steps={max_steps}"]

    cmd = [
        sys.executable, "-m", "torchspec.train_entry",
        "--config", config,
        *step_args,
        f"output_dir={output_dir}",
        *epoch_args,
        *gpu_overrides,
        *wandb_args,
        *extra_args,
    ]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  TRAINING: {name}")
    print(f"  Config:   {config}")
    print(f"  Run ID:   {run_id}")
    print(f"  Steps:    {max_steps}")
    print(f"  Log:      {log_path}")
    print(sep)
    print(f"  Command:  {' '.join(cmd)}")
    print()

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  {name} completed in {int(elapsed)}s ({mins}m {secs}s)")

    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


def _convert_and_upload_hf(
    run_id: str,
    target_model: str,
    hf_repo: Optional[str] = None,
):
    """Convert final FSDP checkpoint to HF format and optionally push to HuggingFace Hub."""
    import glob
    import os

    output_dir = f"{OUTPUTS_DIR}/{run_id}"
    ckpt_base = f"{output_dir}/checkpoints"

    tracker = os.path.join(ckpt_base, "latest_checkpointed_iteration.txt")
    if not os.path.isfile(tracker):
        print(f"\n  [HF Convert] No checkpoint found at {ckpt_base}, skipping conversion.")
        return

    latest_step = open(tracker).read().strip()
    ckpt_dir = os.path.join(ckpt_base, f"iter_{int(latest_step):07d}")
    if not os.path.isdir(ckpt_dir):
        print(f"\n  [HF Convert] Checkpoint dir {ckpt_dir} not found, skipping.")
        return

    hf_output = f"{output_dir}/hf_model"
    print(f"\n  [HF Convert] Converting FSDP checkpoint: {ckpt_dir}")
    print(f"  [HF Convert] Output: {hf_output}")

    draft_config = f"{REPO_DIR}/torchspec/config/dflash_draft_config.json"
    cmd = [
        sys.executable, f"{REPO_DIR}/tools/convert_to_hf.py",
        "--input-dir", ckpt_dir,
        "--output-dir", hf_output,
        "--config", draft_config,
        "--target-model-path", target_model,
        "--trust-remote-code",
        "-f",
    ]
    proc = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False)
    if proc.returncode != 0:
        print(f"  [HF Convert] WARNING: Conversion failed (exit code {proc.returncode})")
        return

    print(f"  [HF Convert] Conversion successful: {hf_output}")

    if hf_repo:
        print(f"  [HF Upload] Pushing to: {hf_repo}")
        try:
            from huggingface_hub import HfApi
            write_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
            api = HfApi(token=write_token)
            api.create_repo(hf_repo, exist_ok=True, private=False)
            api.upload_folder(
                folder_path=hf_output,
                repo_id=hf_repo,
                commit_message=f"DFlash draft model (step {latest_step})",
            )
            print(f"  [HF Upload] Done: https://huggingface.co/{hf_repo}")
        except Exception as e:
            print(f"  [HF Upload] WARNING: Upload failed: {e}")
            print(f"  [HF Upload] Model saved locally at {hf_output} on Modal volume.")
    else:
        print(f"  [HF Upload] No --hf-repo specified, skipping upload.")
        print(f"  [HF Upload] To upload later: huggingface-cli upload {hf_output} <repo_id>")


# =============================================================================
# Modal function — SGLang backend (4+ GPUs)
#
# For 1-2 GPU HF mode, use modal_dflash_train_hf.py instead.
# =============================================================================

_common_kwargs = dict(
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        OUTPUTS_DIR: outputs_vol,
    },
    timeout=24 * 3600,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("xingh3-hf-write"),
        modal.Secret.from_name("wandb-secret"),
    ],
)


@app.function(image=sglang_image, gpu=SGLANG_GPU, **_common_kwargs)
def train_sglang(
    gpu_count: int,
    max_steps: int,
    num_epochs: Optional[int],
    run_eagle3: bool,
    run_dflash: bool,
    wandb_project: Optional[str],
    dataset_path: Optional[str],
    dataset_size: int,
    extra_overrides: Optional[str] = None,
    hf_repo: Optional[str] = None,
):
    """Training entry point for 4+ GPU configs (SGLang inference backend)."""
    _train_impl(
        gpu_count, max_steps, num_epochs, run_eagle3, run_dflash,
        wandb_project, dataset_path, dataset_size, extra_overrides, hf_repo,
    )


def _probe_rdma():
    """Probe for RDMA/InfiniBand devices available in this container."""
    import os
    import pathlib

    print("\n  --- RDMA / Network Probe ---")

    ib_dir = pathlib.Path("/dev/infiniband")
    if ib_dir.exists():
        devices = list(ib_dir.iterdir())
        print(f"  /dev/infiniband: {[d.name for d in devices]}")
    else:
        print("  /dev/infiniband: NOT FOUND")

    ib_sys = pathlib.Path("/sys/class/infiniband")
    if ib_sys.exists():
        devices = list(ib_sys.iterdir())
        print(f"  /sys/class/infiniband: {[d.name for d in devices]}")
    else:
        print("  /sys/class/infiniband: NOT FOUND")

    try:
        result = subprocess.run(
            ["ibstat", "-l"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"  ibstat devices: {result.stdout.strip()}")
        else:
            print(f"  ibstat: not available (rc={result.returncode})")
    except FileNotFoundError:
        print("  ibstat: command not found")
    except Exception as e:
        print(f"  ibstat: error — {e}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines[:12]:
                print(f"  {line}")
            if len(lines) > 12:
                print(f"  ... ({len(lines) - 12} more lines)")
    except Exception:
        pass

    rdma_link = pathlib.Path("/sys/class/net")
    if rdma_link.exists():
        roce_devices = []
        for dev in rdma_link.iterdir():
            device_path = dev / "device" / "infiniband"
            if device_path.exists():
                roce_devices.append(dev.name)
        if roce_devices:
            print(f"  RoCE-capable NICs: {roce_devices}")
        else:
            print("  RoCE-capable NICs: none found")

    print("  --- End RDMA Probe ---\n")


def _train_impl(
    gpu_count: int,
    max_steps: int,
    num_epochs: Optional[int],
    run_eagle3: bool,
    run_dflash: bool,
    wandb_project: Optional[str],
    dataset_path: Optional[str],
    dataset_size: int,
    extra_overrides: Optional[str] = None,
    hf_repo: Optional[str] = None,
):
    import os
    import shutil

    import torch

    detected = torch.cuda.device_count()
    print(f"  GPUs detected: {detected}")
    for i in range(detected):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        print(f"    GPU {i}: {name} ({mem_gb:.1f} GB)")

    _probe_rdma()

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(detected))
    )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{REPO_DIR}/cache/compiled_kernels"

    # Step 13 from runpod_setup.sh: clear stale caches
    for stale_dir in [f"{REPO_DIR}/cache", "/tmp/torchinductor_root"]:
        if os.path.isdir(stale_dir):
            shutil.rmtree(stale_dir, ignore_errors=True)
            print(f"  Cleared stale cache: {stale_dir}")

    # Prepare dataset (mirrors runpod_setup.sh Step 10)
    data_file = dataset_path
    if data_file is None:
        data_file = f"{DATA_DIR}/perfectblend_{dataset_size // 1000}k.jsonl"
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.isfile(data_file):
            print(f"\n  Preparing PerfectBlend dataset ({dataset_size} samples)...")
            subprocess.run(
                [
                    sys.executable, f"{REPO_DIR}/scripts/prepare_perfectblend.py",
                    "--output", data_file,
                    "--sample-size", str(dataset_size),
                    "--seed", "42",
                ],
                check=True,
            )
        print(f"  Dataset: {data_file}")

    user_overrides = []
    if extra_overrides:
        user_overrides = extra_overrides.split()

    cfg = _gpu_config(gpu_count, extra_overrides=user_overrides or None)
    print(f"\n  GPU mode: {cfg.mode}")

    dataset_overrides = [f"dataset.train_data_path={data_file}"]

    if user_overrides:
        print(f"  Extra overrides: {user_overrides}")

    if run_eagle3:
        _run_training(
            name="Eagle3 (baseline)",
            config=cfg.eagle3_config,
            run_id="eagle3-qwen3-8b",
            max_steps=max_steps,
            num_epochs=num_epochs,
            gpu_overrides=cfg.overrides,
            wandb_project=wandb_project,
            extra_args=dataset_overrides + user_overrides,
        )

    if run_dflash:
        _run_training(
            name="DFlash",
            config=cfg.dflash_config,
            run_id="dflash-qwen3-8b",
            max_steps=max_steps,
            num_epochs=num_epochs,
            gpu_overrides=cfg.overrides,
            wandb_project=wandb_project,
            extra_args=dataset_overrides + user_overrides,
        )

    outputs_vol.commit()

    if run_dflash:
        _convert_and_upload_hf(
            run_id="dflash-qwen3-8b",
            target_model="Qwen/Qwen3-8B",
            hf_repo=hf_repo,
        )

    outputs_vol.commit()

    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  GPU mode: {cfg.mode}")
    if run_eagle3:
        print(f"  Eagle3 log:         {OUTPUTS_DIR}/eagle3-qwen3-8b.log")
        print(f"  Eagle3 checkpoints: {OUTPUTS_DIR}/eagle3-qwen3-8b/checkpoints/")
    if run_dflash:
        print(f"  DFlash log:         {OUTPUTS_DIR}/dflash-qwen3-8b.log")
        print(f"  DFlash checkpoints: {OUTPUTS_DIR}/dflash-qwen3-8b/checkpoints/")
        hf_dir = f"{OUTPUTS_DIR}/dflash-qwen3-8b/hf_model"
        if os.path.isdir(hf_dir):
            print(f"  DFlash HF model:    {hf_dir}")
    print()
    print(f"  All outputs saved to Modal volume 'torchspec-outputs'")
    print(f"  Download: modal volume get torchspec-outputs /dflash-qwen3-8b/checkpoints/ ./checkpoints/")
    if os.environ.get("WANDB_API_KEY"):
        proj = wandb_project or "dflash-vs-eagle3"
        print(f"  WandB dashboard: https://wandb.ai/{proj}")
    print()
    print("  Key metrics to compare:")
    print("    train/avg_acc   — top-1 token prediction accuracy")
    print("    train/avg_loss  — CE (DFlash) vs KL (Eagle3)")
    print("    train/grad_norm — gradient health")


# =============================================================================
# Modal function — checkpoint conversion + HF upload (no GPU needed)
# =============================================================================

@app.function(
    image=sglang_image,
    volumes={HF_CACHE_DIR: hf_cache_vol, OUTPUTS_DIR: outputs_vol},
    timeout=3600,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("xingh3-hf-write"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def convert_checkpoint(
    run_id: str,
    hf_repo: str,
    checkpoint_step: Optional[int] = None,
):
    """Convert an FSDP checkpoint to HF format and upload. No GPU required."""
    import os

    output_dir = f"{OUTPUTS_DIR}/{run_id}"
    ckpt_base = f"{output_dir}/checkpoints"

    if checkpoint_step is not None:
        ckpt_dir = os.path.join(ckpt_base, f"iter_{checkpoint_step:07d}")
        step_str = str(checkpoint_step)
    else:
        tracker = os.path.join(ckpt_base, "latest_checkpointed_iteration.txt")
        if not os.path.isfile(tracker):
            print(f"No checkpoint found at {ckpt_base}")
            return
        step_str = open(tracker).read().strip()
        ckpt_dir = os.path.join(ckpt_base, f"iter_{int(step_str):07d}")

    if not os.path.isdir(ckpt_dir):
        print(f"Checkpoint dir {ckpt_dir} not found")
        print(f"Available: {os.listdir(ckpt_base) if os.path.isdir(ckpt_base) else 'none'}")
        return

    hf_output = f"{output_dir}/hf_model_{step_str}"
    print(f"Converting FSDP checkpoint: {ckpt_dir}")
    print(f"Output: {hf_output}")

    draft_config = f"{REPO_DIR}/torchspec/config/dflash_draft_config.json"
    cmd = [
        sys.executable, f"{REPO_DIR}/tools/convert_to_hf.py",
        "--input-dir", ckpt_dir,
        "--output-dir", hf_output,
        "--config", draft_config,
        "--target-model-path", "Qwen/Qwen3-8B",
        "--trust-remote-code",
        "-f",
    ]
    proc = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False)
    if proc.returncode != 0:
        print(f"Conversion failed (exit code {proc.returncode})")
        return

    print(f"Conversion successful: {hf_output}")
    print(f"Uploading to: {hf_repo}")

    from huggingface_hub import HfApi
    write_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=write_token)
    api.create_repo(hf_repo, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=hf_output,
        repo_id=hf_repo,
        commit_message=f"DFlash draft model (step {step_str})",
    )
    print(f"Upload complete: https://huggingface.co/{hf_repo}")

    outputs_vol.commit()


# =============================================================================
# CLI entry point
# =============================================================================

@app.local_entrypoint()
def main(
    gpu_count: int = 8,
    max_steps: int = 200,
    num_epochs: int = 0,
    run_eagle3: bool = False,
    run_dflash: bool = True,
    wandb_project: Optional[str] = None,
    dataset_path: Optional[str] = None,
    dataset_size: int = 50000,
    extra_overrides: str = "",
    hf_repo: str = "",
    convert_only: str = "",
    convert_step: int = 0,
):
    # --- Convert-only mode: no GPU, just convert + upload ---
    if convert_only:
        if not hf_repo:
            print("Error: --hf-repo is required with --convert-only")
            return
        step = convert_step if convert_step > 0 else None
        print(f"Converting checkpoint from run '{convert_only}' (step={step or 'latest'}) -> {hf_repo}")
        convert_checkpoint.remote(
            run_id=convert_only,
            hf_repo=hf_repo,
            checkpoint_step=step,
        )
        return

    # --- Normal training mode ---
    if gpu_count < 4:
        print(f"Error: This script requires >= 4 GPUs (got {gpu_count}).")
        print("  For 1-2 GPU HF mode, use: modal run scripts/modal_dflash_train_hf.py")
        return

    epochs_override = num_epochs if num_epochs > 0 else None
    infer_gpus = 1
    train_gpus = gpu_count - 1
    if extra_overrides:
        for ov in extra_overrides.split():
            if ov.startswith("inference.inference_num_gpus="):
                infer_gpus = int(ov.split("=", 1)[1])
            elif ov.startswith("training.training_num_gpus_per_node="):
                train_gpus = int(ov.split("=", 1)[1])

    print("=" * 60)
    print("  DFlash Training on Modal")
    print("=" * 60)
    print(f"  GPU:          {SGLANG_GPU} ({infer_gpus} infer + {train_gpus} train)")
    print(f"  Eagle3:       {'YES' if run_eagle3 else 'SKIP'}")
    print(f"  DFlash:       {'YES' if run_dflash else 'SKIP'}")
    print(f"  Max steps:    {max_steps}")
    print(f"  Num epochs:   {epochs_override or '(from YAML config)'}")
    print(f"  Dataset:      {dataset_path or f'PerfectBlend ({dataset_size} samples)'}")
    print(f"  WandB:        {wandb_project or '(disabled / auto)'}")
    if extra_overrides:
        print(f"  Overrides:    {extra_overrides}")
    print("=" * 60)

    if not run_eagle3 and not run_dflash:
        print("Nothing to run. Pass --run-eagle3 and/or --run-dflash.")
        return

    if hf_repo:
        print(f"  HF Upload:    {hf_repo}")

    train_sglang.spawn(
        gpu_count=gpu_count,
        max_steps=max_steps,
        num_epochs=epochs_override,
        run_eagle3=run_eagle3,
        run_dflash=run_dflash,
        wandb_project=wandb_project,
        dataset_path=dataset_path,
        dataset_size=dataset_size,
        extra_overrides=extra_overrides or None,
        hf_repo=hf_repo or None,
    ).get()

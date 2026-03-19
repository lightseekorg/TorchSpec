#!/usr/bin/env bash
# RunPod DFlash vs Eagle3 training comparison.
#
# Auto-detects GPU count and picks the right config:
#   1 GPU  → HF backend + colocate mode (inference & training share GPU)
#   2 GPUs → HF backend (1 inference + 1 training)
#   4 GPUs → SGLang backend (2 inference + 2 training)
#
# Usage (on RunPod pod):
#   bash scripts/runpod_dflash_train.sh
#
#   SKIP_SETUP=1 bash scripts/runpod_dflash_train.sh
#   RUN_EAGLE3=1 RUN_DFLASH=0 bash scripts/runpod_dflash_train.sh
#   RUN_EAGLE3=0 RUN_DFLASH=1 bash scripts/runpod_dflash_train.sh
#   MAX_STEPS=50 bash scripts/runpod_dflash_train.sh
set -euo pipefail

WORKSPACE="${RUNPOD_VOLUME_PATH:-/workspace}"
VENV_DIR="$WORKSPACE/.venv-torchspec"
REQUIRED_PY_MINOR=11
SGLANG_VERSION="v0.5.8.post1"
SGLANG_COMMIT="0f2df9370a1de1b4fb11b071d39ab3ce2287a350"
SKIP_SETUP="${SKIP_SETUP:-0}"
RUN_EAGLE3="${RUN_EAGLE3:-1}"
RUN_DFLASH="${RUN_DFLASH:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-dflash-vs-eagle3}"

echo "============================================================"
echo "  DFlash vs Eagle3 — GPU Training Comparison"
echo "============================================================"
echo "  Workspace:   $WORKSPACE"
echo "  Eagle3:      $([ "$RUN_EAGLE3" = "1" ] && echo "YES" || echo "SKIP")"
echo "  DFlash:      $([ "$RUN_DFLASH" = "1" ] && echo "YES" || echo "SKIP")"
echo "  Max steps:   $MAX_STEPS"
echo "  WandB proj:  $WANDB_PROJECT"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────────
# Section 1: Environment Setup
# ─────────────────────────────────────────────────────────────────

if [ "$SKIP_SETUP" = "0" ]; then

echo "━━━ [1/6] Python version check ━━━"
cd "$WORKSPACE"

get_python_version() {
    "$1" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0"
}
get_python_minor() {
    "$1" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0"
}

PYTHON=""
for candidate in python3.12 python3.11 python3.13 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        minor=$(get_python_minor "$candidate")
        if [ "$minor" -ge "$REQUIRED_PY_MINOR" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  Installing Python 3.11..."
    apt-get update -qq
    apt-get install -y -qq software-properties-common > /dev/null 2>&1
    add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev > /dev/null 2>&1
    PYTHON="python3.11"
fi
echo "  Python: $(get_python_version "$PYTHON")"

echo ""
echo "━━━ [2/6] Python venv ━━━"
if [ -d "$VENV_DIR" ]; then
    VENV_PY_MINOR=$(get_python_minor "$VENV_DIR/bin/python" 2>/dev/null || echo "0")
    if [ "$VENV_PY_MINOR" -ge "$REQUIRED_PY_MINOR" ]; then
        echo "  Reusing existing venv"
    else
        rm -rf "$VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR"
    fi
else
    "$PYTHON" -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi

export PATH="$VENV_DIR/bin:$PATH"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"
$PIP install --upgrade pip --quiet 2>&1 | tail -1

echo ""
echo "━━━ [3/6] System deps ━━━"
apt-get update -qq 2>/dev/null && apt-get install -y -qq git vim htop tmux > /dev/null 2>&1
echo "  Done."

echo ""
echo "━━━ [4/6] Clone / update TorchSpec ━━━"
TORCHSPEC_REPO="${TORCHSPEC_REPO:-https://github.com/zhubohao911/TorchSpec.git}"
if [ ! -d "$WORKSPACE/TorchSpec" ]; then
    git clone "$TORCHSPEC_REPO" "$WORKSPACE/TorchSpec"
    cd "$WORKSPACE/TorchSpec"
    git checkout feature/dflash-training
else
    cd "$WORKSPACE/TorchSpec"
    git fetch --all
    git checkout feature/dflash-training
    git pull || true
fi
echo "  Branch: $(git branch --show-current)"
echo "  Commit: $(git log -1 --format='%h %s')"

echo ""
echo "━━━ [5/6] Install deps + TorchSpec ━━━"
cd "$WORKSPACE/TorchSpec"

# PyTorch 2.6+ required for FlexAttention (torch.nn.attention.flex_attention)
TORCH_OK=$($PY -c "
import torch
v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
ok = torch.cuda.is_available() and v >= (2, 6)
print('1' if ok else '0')
" 2>/dev/null || echo "0")

if [ "$TORCH_OK" = "0" ]; then
    echo "  Installing PyTorch 2.6+ with CUDA 12.4..."
    $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet 2>&1 | tail -3
    $PY -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
else
    $PY -c "import torch; print(f'  PyTorch {torch.__version__} OK (>= 2.6, CUDA available)')"
fi

# Detect GPU count to decide whether SGLang is needed
NUM_GPUS=$($PY -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$NUM_GPUS" -ge 4 ]; then
    SGLANG_DIR="$WORKSPACE/TorchSpec/_sglang"
    if [ ! -d "$SGLANG_DIR" ]; then
        echo "  Cloning SGLang (4+ GPUs detected)..."
        git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
    fi
    cd "$SGLANG_DIR"
    git checkout "$SGLANG_COMMIT"
    git reset --hard HEAD
    cd "$WORKSPACE/TorchSpec"

    echo "  Installing SGLang..."
    $PIP install -e "_sglang/python[all]" --quiet 2>&1 | tail -3

    echo "  Applying SGLang patch..."
    cd "$SGLANG_DIR"
    git apply "$WORKSPACE/TorchSpec/patches/sglang/$SGLANG_VERSION/sglang.patch" 2>/dev/null || {
        echo "  Patch already applied or not needed."
    }
    cd "$WORKSPACE/TorchSpec"
else
    echo "  ${NUM_GPUS} GPU(s) detected — using HF backend (skipping SGLang install)"
fi

echo "  Installing TorchSpec..."
$PIP install -e ".[dev]" --quiet 2>&1 | tail -3

echo ""
echo "━━━ [6/6] Verify environment ━━━"
$PY -c "
import sys, torch

print(f'  Python:  {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda or \"N/A\"}')
print(f'  GPUs:    {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')

from torchspec import DFlashModel, DFlashConfig, DFlashDraftModel
print('  TorchSpec DFlash imports: OK')
"

else
    # SKIP_SETUP=1: just activate venv
    export PATH="$VENV_DIR/bin:$PATH"
    PY="$VENV_DIR/bin/python"
    cd "$WORKSPACE/TorchSpec"
    echo "Skipping setup (SKIP_SETUP=1). Using existing env."
fi

# ─────────────────────────────────────────────────────────────────
# Section 2: Auto-detect GPU config
# ─────────────────────────────────────────────────────────────────

PY="${PY:-$VENV_DIR/bin/python}"
NUM_GPUS=$($PY -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

echo ""
echo "━━━ GPU Configuration ━━━"
echo "  GPUs available: $NUM_GPUS"

if [ "$NUM_GPUS" -ge 4 ]; then
    GPU_MODE="4gpu"
    EAGLE3_CONFIG="configs/sglang_qwen3_8b.yaml"
    DFLASH_CONFIG="configs/sglang_qwen3_8b_dflash.yaml"
    TRAIN_GPUS=2
    INFERENCE_GPUS=2
    GPU_OVERRIDES=(
        training.training_num_gpus_per_node="$TRAIN_GPUS"
        inference.inference_num_gpus="$INFERENCE_GPUS"
        inference.inference_num_gpus_per_engine=2
        inference.inference_num_gpus_per_node=4
    )
    echo "  Mode: 4-GPU (SGLang, 2 inference + 2 training)"
elif [ "$NUM_GPUS" -ge 2 ]; then
    GPU_MODE="2gpu"
    EAGLE3_CONFIG="configs/hf_qwen3_8b.yaml"
    DFLASH_CONFIG="configs/hf_qwen3_8b_dflash_1gpu.yaml"
    TRAIN_GPUS=1
    INFERENCE_GPUS=1
    GPU_OVERRIDES=(
        training.training_num_gpus_per_node=1
        inference.inference_num_gpus=1
        inference.inference_num_gpus_per_engine=1
        inference.inference_num_gpus_per_node=2
    )
    echo "  Mode: 2-GPU (HF backend, 1 inference + 1 training)"
elif [ "$NUM_GPUS" -ge 1 ]; then
    GPU_MODE="1gpu"
    EAGLE3_CONFIG="configs/hf_qwen3_8b_1gpu.yaml"
    DFLASH_CONFIG="configs/hf_qwen3_8b_dflash_1gpu.yaml"
    TRAIN_GPUS=1
    INFERENCE_GPUS=1
    GPU_OVERRIDES=(
        training.colocate=true
        training.training_num_gpus_per_node=1
        inference.inference_num_gpus=1
        inference.inference_num_gpus_per_engine=1
        inference.inference_num_gpus_per_node=1
    )
    echo "  Mode: 1-GPU colocate (HF backend, shared GPU)"
    echo "  WARNING: Memory-intensive! Needs H100/A100 80GB."
    echo "  Using max_seq_length=4096 (reduced from 16384)"
else
    echo "  ERROR: No GPUs detected!"
    exit 1
fi

echo "  Eagle3 config: $EAGLE3_CONFIG"
echo "  DFlash config: $DFLASH_CONFIG"

# ─────────────────────────────────────────────────────────────────
# Section 3: WandB Setup
# ─────────────────────────────────────────────────────────────────

echo ""
echo "━━━ WandB Configuration ━━━"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "  WANDB_API_KEY not set."
    echo "  Continuing without WandB (metrics in stdout/logs only)."
    WANDB_ARGS=()
else
    WANDB_ARGS=(training.report_to=wandb training.wandb_project="$WANDB_PROJECT")
    echo "  WandB enabled: project=$WANDB_PROJECT"
fi

# ─────────────────────────────────────────────────────────────────
# Section 4: Model cache
# ─────────────────────────────────────────────────────────────────

export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo ""
echo "  HF cache: $HF_HOME"

echo "  Pre-downloading Qwen/Qwen3-8B (if not cached)..."
$PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', cache_dir='$HF_HOME')
print('  Model ready.')
" 2>/dev/null || echo "  Download skipped (will download on first run)."

# ─────────────────────────────────────────────────────────────────
# Section 5: Training Runs
# ─────────────────────────────────────────────────────────────────

cd "$WORKSPACE/TorchSpec"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
export TORCHINDUCTOR_CACHE_DIR="$WORKSPACE/TorchSpec/cache/compiled_kernels"
export TORCHSPEC_LOG_LEVEL=INFO

run_training() {
    local NAME="$1"
    local CONFIG="$2"
    local RUN_ID="$3"
    shift 3

    echo ""
    echo "============================================================"
    echo "  TRAINING: $NAME"
    echo "  Config:   $CONFIG"
    echo "  GPU mode: $GPU_MODE ($NUM_GPUS GPUs)"
    echo "  Run ID:   $RUN_ID"
    echo "  Steps:    $MAX_STEPS"
    echo "============================================================"

    local START_TIME
    START_TIME=$(date +%s)

    $PY -m torchspec.train_entry \
        --config "$CONFIG" \
        training.num_train_steps="$MAX_STEPS" \
        "${GPU_OVERRIDES[@]}" \
        "${WANDB_ARGS[@]}" \
        training.wandb_run_id="$RUN_ID" \
        "$@" \
        2>&1 | tee "$WORKSPACE/TorchSpec/outputs/${RUN_ID}.log"

    local END_TIME
    END_TIME=$(date +%s)
    local ELAPSED=$(( END_TIME - START_TIME ))
    echo ""
    echo "  $NAME completed in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
}

mkdir -p "$WORKSPACE/TorchSpec/outputs"

# ── Run 1: Eagle3 baseline ──────────────────────────────────────

if [ "$RUN_EAGLE3" = "1" ]; then
    run_training \
        "Eagle3 (baseline)" \
        "$EAGLE3_CONFIG" \
        "eagle3-qwen3-8b"
fi

# ── Run 2: DFlash ───────────────────────────────────────────────

if [ "$RUN_DFLASH" = "1" ]; then
    run_training \
        "DFlash" \
        "$DFLASH_CONFIG" \
        "dflash-qwen3-8b"
fi

# ─────────────────────────────────────────────────────────────────
# Section 6: Summary
# ─────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Training Complete!"
echo "============================================================"
echo ""
echo "  GPU mode: $GPU_MODE ($NUM_GPUS GPUs)"
echo ""
echo "  Logs:"
[ "$RUN_EAGLE3" = "1" ] && echo "    Eagle3: outputs/eagle3-qwen3-8b.log"
[ "$RUN_DFLASH" = "1" ] && echo "    DFlash: outputs/dflash-qwen3-8b.log"
echo ""
if [ "${#WANDB_ARGS[@]}" -gt 0 ]; then
    echo "  WandB dashboard: https://wandb.ai/$WANDB_PROJECT"
    echo "  Compare runs: eagle3-qwen3-8b vs dflash-qwen3-8b"
    echo ""
fi
echo "  Key metrics to compare:"
echo "    train/avg_acc   — top-1 token prediction accuracy (directly comparable)"
echo "    train/avg_loss  — CE (DFlash) vs KL (Eagle3), compare trends not absolutes"
echo "    train/grad_norm — gradient health"
echo ""
echo "  To re-run DFlash only:"
echo "    SKIP_SETUP=1 RUN_EAGLE3=0 RUN_DFLASH=1 bash scripts/runpod_dflash_train.sh"
echo ""

#!/bin/bash
# Train Qwen3-8B with the colocate (MPS + NCCL) path on a single
# 4×H100 node. This is the colocate sibling of
# `examples/qwen3-8b-single-node/run.sh`; it pins the GPU layout so
# `engine_count × engine_tp_size == training_world_size == 4`,
# which is what the Phase-2 union NCCL world is shaped for.
#
# Usage:
#   ./examples/colocate-qwen3-8b-1node/run.sh                  # default 4 GPUs
#   ./examples/colocate-qwen3-8b-1node/run.sh CONFIG.yaml      # custom config
#   ./examples/colocate-qwen3-8b-1node/run.sh CONFIG.yaml training.num_train_steps=10
#
# Prerequisites:
#   * NVIDIA MPS daemon binary in $PATH (`nvidia-cuda-mps-control`); the
#     CUDA toolkit ships it. The driver auto-starts it via setup_for_colocate.
#   * Hugging Face credentials for Qwen/Qwen3-8B (via HF_TOKEN or `huggingface-cli login`).
#   * The upstream sglang colocate patch — see docs/colocate/sglang_patch.md.
#     Without it the run will hang on the first NCCL recv (the trainer
#     side comes up fine; the engine side never sends).

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export TORCHSPEC_LOG_LEVEL=INFO

# expandable_segments matters under MPS — both trainer and engine
# sit in the same allocator pool, so non-fragmenting growth is what
# keeps the long stability run flat.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONFIG_FILE="${1:-$ROOT_DIR/configs/colocate_qwen3_8b.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
elif [[ -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
    CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/colocate_qwen3_8b.yaml"
fi

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

# Colocate (MPS) layout: every GPU runs both a trainer rank and an
# engine rank. So training_num_gpus_per_node == TOTAL_GPUS and
# inference_num_gpus == TOTAL_GPUS too. The placement-group code
# (Phase 1) puts the 1:1 paired actors on the same Ray bundle.
TRAIN_GPUS="$TOTAL_GPUS"
INFERENCE_GPUS="$TOTAL_GPUS"

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

echo "=============================================="
echo "Train Qwen3-8B (colocate: MPS + NCCL)"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Total GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Trainer ranks: $TRAIN_GPUS  (FSDP, ranks 0..N-1 in union world)"
echo "  - Engine ranks:  $INFERENCE_GPUS  (TP=1 per engine, ranks N..2N-1)"
echo "  - GPUs are SHARED via NVIDIA MPS"
echo "Local IP: $LOCAL_IP"
echo "Extra args: $*"
echo "=============================================="

python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_engine=1 \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    inference.sglang.tp_size=1 \
    "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="

#!/bin/bash
# DFlash training with SGLang async inference (single-node)
#
# GPU allocation (default: 4 GPUs):
#   - 2 GPUs for inference (SGLang tp=2)
#   - 2 GPUs for training (FSDP)
#
# Usage:
#   ./examples/dflash-qwen3-8b-single-node/run.sh [CONFIG_FILE] [EXTRA_ARGS...]
#
# Examples:
#   ./examples/dflash-qwen3-8b-single-node/run.sh
#   ./examples/dflash-qwen3-8b-single-node/run.sh configs/dflash_qwen3_8b_e2e.yaml training.num_train_steps=5

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export TORCHSPEC_LOG_LEVEL=INFO

CONFIG_FILE="${1:-$ROOT_DIR/configs/dflash_qwen3_8b_e2e.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
elif [[ -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
    CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/dflash_qwen3_8b_e2e.yaml"
fi

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

TRAIN_GPUS=2
INFERENCE_GPUS=2

echo "=============================================="
echo "DFlash Training with async inference"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Total GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training GPUs: $TRAIN_GPUS"
echo "  - Inference GPUs: $INFERENCE_GPUS"
echo "Extra args: $*"
echo "=============================================="

python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_engine=2 \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    inference.sglang.tp_size=2 \
    "$@"

echo "=============================================="
echo "DFlash training completed!"
echo "=============================================="

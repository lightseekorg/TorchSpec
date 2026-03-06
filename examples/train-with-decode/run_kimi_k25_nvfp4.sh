#!/bin/bash
# Launch torchspec.train_entry for Kimi-K2.5 NVFP4 Eagle3 decode-mode training
#
# GPU allocation (default: 6 GPUs on single node):
#   - 4 GPUs for inference (SglEngine TP=4, target + draft model with NVFP4 quantization)
#   - 2 GPUs for training (FSDP/DP: draft model sharded across 2 GPUs)
#
# Usage:
#   ./examples/train-with-decode/run_kimi_k25_nvfp4.sh [extra_overrides...]
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES — GPUs to use (default: 0,1,2,3,4,5)
#   TRAIN_GPUS           — GPUs for training (default: 2)
#   INFERENCE_GPUS       — GPUs for inference (default: 4)
#   CONFIG_FILE          — Override config file path

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export TORCHSPEC_LOG_LEVEL=INFO

TRAIN_GPUS="${TRAIN_GPUS:-2}"
INFERENCE_GPUS="${INFERENCE_GPUS:-4}"

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/train_with_decode/sglang_kimi_k25_nvfp4.yaml}"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/kimi25_nvfp4_decode_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "=============================================="
echo "Kimi-K2.5 NVFP4 Decode-Mode Training"
echo "=============================================="
echo "Config:          $CONFIG_FILE"
echo "Total GPUs:      $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Training GPUs:  $TRAIN_GPUS (FSDP/DP)"
echo "  Inference GPUs: $INFERENCE_GPUS (SglEngine TP=$INFERENCE_GPUS, NVFP4 quantization)"
echo "Extra args: $*"
echo "=============================================="

python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_engine_type="sgl" \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_engine="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    inference.sglang.tp_size="$INFERENCE_GPUS" \
    model.draft_model_config="$ROOT_DIR/configs/draft_models/kimi_k25_eagle3.json" \
    "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="

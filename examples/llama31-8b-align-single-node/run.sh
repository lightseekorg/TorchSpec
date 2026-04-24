#!/bin/bash
# Launch TorchSpec alignment training for Meta-Llama-3.1-8B-Instruct on a single node.
#
# This recipe is tuned to match the main hyperparameters in
# /nfs/ofs-llab-volume/users/fengyu/SpecForge/train_offline.sh:
#   - target model / draft config / train data
#   - lr=1e-4, epochs=10, max_seq_length=4096, ttt_length=7
#   - effective global batch size = 8

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HOME="${HF_HOME:-/nfs/ofs-llab-volume/users/fengyu/hf_cache}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/sglang_llama31_8b_align.yaml}"

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/llama31_8b_align_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS="${#GPU_ARRAY[@]}"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
INFERENCE_GPUS="${INFERENCE_GPUS:-1}"

echo "=============================================="
echo "TorchSpec Llama-3.1-8B alignment run"
echo "=============================================="
echo "Config:                $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES"
echo "Visible GPUs:          $TOTAL_GPUS"
echo "Training GPUs:         $TRAIN_GPUS"
echo "Inference GPUs:        $INFERENCE_GPUS (colocate=true)"
echo "Effective global batch: $((TRAIN_GPUS * 1 * 1))"
echo "Log file:              $LOG_FILE"
echo "Extra args:            $*"
echo "=============================================="

python3 -m torchspec.train_entry \
  --config "$CONFIG_FILE" \
  training.training_num_gpus_per_node="$TRAIN_GPUS" \
  inference.inference_num_gpus="$INFERENCE_GPUS" \
  training.num_epochs=10 \
  "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="

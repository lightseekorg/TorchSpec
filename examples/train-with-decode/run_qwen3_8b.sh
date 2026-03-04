#!/bin/bash
# Train with SglEngine in decode mode (speculative decoding)
#
# GPU allocation (default: 3 GPUs total):
#   - 1 GPU  for inference (SglEngine with TP=1, loads target + draft model)
#   - 2 GPUs for training (FSDP/DP: draft model sharded across 2 GPUs)
#
# Usage:
#   ./examples/train-with-decode/run_qwen3_8b.sh [CONFIG_FILE] [EXTRA_ARGS...]
#
# Examples:
#   # Run with default decode config
#   ./examples/train-with-decode/run_qwen3_8b.sh
#
#   # Run with custom config
#   ./examples/train-with-decode/run_qwen3_8b.sh configs/train_with_decode/sglang_qwen3_8b.yaml
#
#   # Run with extra args
#   ./examples/train-with-decode/run_qwen3_8b.sh configs/train_with_decode/sglang_qwen3_8b.yaml training.num_train_steps=10

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export TORCHSPEC_LOG_LEVEL=INFO

CONFIG_FILE="${1:-$ROOT_DIR/configs/train_with_decode/sglang_qwen3_8b.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
elif [[ -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
    CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/train_with_decode/sglang_qwen3_8b.yaml"
fi

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

TRAIN_GPUS=2
INFERENCE_GPUS=1

echo "=============================================="
echo "Train with SglEngine decode (speculative decoding)"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Total GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training GPUs: $TRAIN_GPUS (FSDP/DP - model sharded)"
echo "  - Inference GPUs: $INFERENCE_GPUS (SglEngine TP=1, target + draft model)"
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
    "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="

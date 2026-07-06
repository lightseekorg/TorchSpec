#!/bin/bash
# Qwen3-8B Domino 8-GPU verification run.
#
# Default GPU allocation:
#   - 4 GPUs for SGLang inference, one full model copy per engine
#   - 4 GPUs for Domino training with FSDP FULL_SHARD
#
# Usage:
#   TRAIN_DATA_PATH=/path/to/perfectblend_10k.jsonl \
#   OUTPUT_ROOT=/path/to/durable/output \
#   ./examples/qwen3-8b-domino-8h100/run.sh
#
# Optional:
#   ./examples/qwen3-8b-domino-8h100/run.sh configs/sglang_qwen3_8b_domino_2gpu.yaml \
#     training.num_train_steps=20

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS:-ATEN,TRITON}"
export MC_STORE_MEMCPY="${MC_STORE_MEMCPY:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$ROOT_DIR"

CONFIG_FILE="${1:-$ROOT_DIR/configs/sglang_qwen3_8b_domino_2gpu.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
  shift 1 || true
elif [[ -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
  CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
  shift 1 || true
else
  CONFIG_FILE="$ROOT_DIR/configs/sglang_qwen3_8b_domino_2gpu.yaml"
fi

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS="${#GPU_ARRAY[@]}"
if [[ "$TOTAL_GPUS" -lt 8 ]]; then
  echo "Expected at least 8 visible GPUs, got ${TOTAL_GPUS}: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-qwen3-8b-domino-8h100-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$ROOT_DIR/data/perfectblend_10k.jsonl}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-1}"

TRAIN_GPUS="${TRAIN_GPUS:-4}"
INFERENCE_GPUS="${INFERENCE_GPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
DRAFT_ACCUMULATION_STEPS="${DRAFT_ACCUMULATION_STEPS:-4}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-8}"

export HF_HOME="${HF_HOME:-$ROOT_DIR/hf-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}"
export TORCHSPEC_LOG_DIR="${TORCHSPEC_LOG_DIR:-$OUTPUT_DIR/rank-logs}"

if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
  echo "Training data not found: ${TRAIN_DATA_PATH}" >&2
  echo "Set TRAIN_DATA_PATH to the JSONL dataset used for the verification run." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$TORCHSPEC_LOG_DIR" "$ROOT_DIR/cache"

LOG_FILE="$OUTPUT_DIR/launcher.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "Qwen3-8B Domino 8-GPU verification"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Run name: $RUN_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Training data: $TRAIN_DATA_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Training GPUs: $TRAIN_GPUS"
echo "Inference GPUs: $INFERENCE_GPUS"
echo "Inference TP size: $TP_SIZE"
echo "Steps: $NUM_TRAIN_STEPS"
echo "Save interval: $SAVE_INTERVAL"
echo "Max checkpoints: $MAX_CHECKPOINTS"
echo "Extra args: $*"
echo "=============================================="

python3 -m torchspec.train_entry \
  --config "$CONFIG_FILE" \
  dataset.train_data_path="$TRAIN_DATA_PATH" \
  training.num_train_steps="$NUM_TRAIN_STEPS" \
  training.save_interval="$SAVE_INTERVAL" \
  training.max_checkpoints="$MAX_CHECKPOINTS" \
  training.training_num_nodes=1 \
  training.training_num_gpus_per_node="$TRAIN_GPUS" \
  training.fsdp_strategy=FULL_SHARD \
  training.draft_accumulation_steps="$DRAFT_ACCUMULATION_STEPS" \
  training.prefetch_depth="$PREFETCH_DEPTH" \
  inference.inference_num_gpus="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_engine="$TP_SIZE" \
  inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
  inference.sglang.tp_size="$TP_SIZE" \
  debug.enable_perf_metrics=true \
  output_dir="$OUTPUT_DIR" \
  cache_dir="$ROOT_DIR/cache/$RUN_NAME" \
  "$@"

echo "=============================================="
echo "Training completed. Checkpoints: $OUTPUT_DIR/checkpoints"
echo "=============================================="

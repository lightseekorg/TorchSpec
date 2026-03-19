#!/bin/bash
# MiniMax-M2.5 Eagle3 training
#
# GPU allocation (40 GPUs across 5 nodes):
#   - Node 0-2: 24 GPUs for inference (6 engines x TP=4)
#   - Node 3-4: 16 GPUs for training (FSDP across 16 GPUs)
#
# Usage:
#   ./examples/minimax-m25-5node-h200/run.sh [CONFIG_FILE] [EXTRA_ARGS...]
#
# Examples:
#   ./examples/minimax-m25-5node-h200/run.sh
#   ./examples/minimax-m25-5node-h200/run.sh configs/sglang_minimax_m25_5node.yaml training.num_train_steps=10

set -euo pipefail
set -x
export SGLANG_DISABLE_CUDNN_CHECK=1
# export SGLANG_VLM_CACHE_SIZE_MB=16384
export TORCHSPEC_LOG_LEVEL=INFO

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

export TORCHSPEC_LOG_DIR="$ROOT_DIR/running_logs/minimax_m25_train"
CONFIG_FILE="${1:-configs/sglang_minimax_m25_5node.yaml}"

if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
elif [[ -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
    CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/sglang_minimax_m25_5node.yaml"
fi


TRAIN_NODES="${TRAIN_NODES:-2}"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
INFERENCE_GPUS="${INFERENCE_GPUS:-24}"
INFERENCE_GPUS_PER_NODE="${INFERENCE_GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-4}"

CHECKPOINT_DIR=$(python3 - "$CONFIG_FILE" "$@" <<'PY'
import sys

from torchspec.config.train_config import config_to_flat_args, load_config

config_file = sys.argv[1]
cli_args = sys.argv[2:]

config = load_config(config_file, cli_args)
args = config_to_flat_args(config)
print(args.checkpoint_dir or "")
PY
)

RESUME_ARGS=()
if [ -n "$CHECKPOINT_DIR" ]; then
  RESUME_ARGS=(training.load_path="$CHECKPOINT_DIR")
  if [ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]; then
    LAST_STEP=$(<"$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")
    echo "Resuming from checkpoint: $CHECKPOINT_DIR/iter_$(printf '%07d' "$LAST_STEP")"
  else
    echo "No checkpoint tracker found at: $CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
  fi
fi

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/minimax_m25_train_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "=============================================="
echo "MiniMax-M2.5 Eagle3 Training"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "  Training GPUs:  $((TRAIN_NODES * TRAIN_GPUS)) ($TRAIN_NODES nodes x $TRAIN_GPUS GPUs)"
echo "  Inference GPUs: $INFERENCE_GPUS ($((INFERENCE_GPUS / INFERENCE_GPUS_PER_NODE)) nodes x $INFERENCE_GPUS_PER_NODE GPUs, TP=$TP_SIZE per engine)"
echo "Extra args: $*"
echo "=============================================="

python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_nodes="$TRAIN_NODES" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_engine="$TP_SIZE" \
    inference.inference_num_gpus_per_node="$INFERENCE_GPUS_PER_NODE" \
    inference.sglang.tp_size="$TP_SIZE" \
    model.draft_model_config="$ROOT_DIR/configs/draft_models/minimax_m25_eagle3.json" \
    "${RESUME_ARGS[@]}" \
    "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="

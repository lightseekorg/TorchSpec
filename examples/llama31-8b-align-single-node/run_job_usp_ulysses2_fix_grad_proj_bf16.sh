#!/bin/bash
set -xeo pipefail

echo "DISTRIBUTED_NODE_COUNT: ${DISTRIBUTED_NODE_COUNT:-1}"
echo "DISTRIBUTED_NODE_RANK: ${DISTRIBUTED_NODE_RANK:-0}"
echo "DISTRIBUTED_MASTER_HOSTS: ${DISTRIBUTED_MASTER_HOSTS:-}"
echo "PET_MASTER_PORT: ${PET_MASTER_PORT:-6379}"
echo "NP: ${NP:-8}"

set -euo pipefail
set -x

export PATH="/nfs/ofs-fengyu/env/conda/envs/torchspec/bin:/nfs/ofs-fengyu/env/conda/condabin:/nfs/ofs-fengyu/env/conda/bin/:$PATH"
export MAMBA_EXE="/nfs/ofs-fengyu/env/conda/bin/micromamba"
export MAMBA_ROOT_PREFIX="/nfs/ofs-fengyu/env/conda"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate torchspec

WORKING_DIR="${WORKING_DIR:-/nfs/ofs-llab-volume/users/fengyu/base_torchspec}"
cd "$WORKING_DIR"
export PYTHONPATH="$WORKING_DIR${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/nfs/ofs-llab-volume/users/fengyu/hf_cache}"
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-/nfs/ofs-fengyu/cache/deep_gemm}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
mkdir -p "$SGLANG_DG_CACHE_DIR"

RAY_TMP_ROOT="${RAY_TMP_ROOT:-/tmp}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-$RAY_TMP_ROOT/ray_ts_l31_usp_u2_fix_grad_proj_bf16_$(id -u)}"
export RAY_TMPDIR="$RAY_TEMP_DIR"
export RAY_TEMP_DIR="$RAY_TMPDIR"
mkdir -p "$RAY_TEMP_DIR"

LOCAL_IP="$(ip -4 addr show scope global | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1)"
LOG_SUFFIX="$(printf "%s" "$LOCAL_IP" | tr '.' '-')"
DATE_TAG="$(date +%m%d_%H%M)"
RAY_LOG_ROOT="$WORKING_DIR/tmp/ray_usp_u2_fix_grad_proj_bf16_${DATE_TAG}_${LOG_SUFFIX}"
RAY_LOG_DEST="$RAY_LOG_ROOT/node_${DISTRIBUTED_NODE_RANK:-0}"
mkdir -p "$RAY_LOG_DEST"

sync_ray_logs() {
    if [ -d "$RAY_TEMP_DIR" ]; then
        cp -a "$RAY_TEMP_DIR" "$RAY_LOG_DEST/" 2>/dev/null || true
    fi
}

sync_ray_logs_loop() {
    while true; do
        sync_ray_logs
        sleep 30
    done
}

sync_ray_logs_loop >/dev/null 2>&1 &
RAY_LOG_SYNC_PID=$!
trap 'kill "$RAY_LOG_SYNC_PID" 2>/dev/null || true; sync_ray_logs; ray stop --force >/dev/null 2>&1 || true' EXIT

ray stop --force || true
export RAY_DISABLE_DASHBOARD=1
export RAY_DISABLE_METRICS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/sglang_llama31_8b_align.yaml}"
LOG_DIR="$ROOT_DIR/running_logs"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/llama31_8b_align_usp_ulysses2_fix_grad_proj_bf16}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/cache/llama31_8b_align_usp_ulysses2_fix_grad_proj_bf16}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-$OUTPUT_DIR/tensorboard}"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$TENSORBOARD_LOGDIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/llama31_8b_align_usp_ulysses2_fix_grad_proj_bf16_${TIMESTAMP}.log"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS="${#GPU_ARRAY[@]}"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
INFERENCE_GPUS="${INFERENCE_GPUS:-1}"
SP_ULYSSES_SIZE="${SP_ULYSSES_SIZE:-2}"
SP_RING_SIZE="${SP_RING_SIZE:-1}"
SP_SIZE="$((SP_ULYSSES_SIZE * SP_RING_SIZE))"

if (( SP_SIZE <= 0 )); then
    echo "Invalid USP topology: ulysses=${SP_ULYSSES_SIZE}, ring=${SP_RING_SIZE}" >&2
    exit 1
fi
if (( TRAIN_GPUS % SP_SIZE != 0 )); then
    echo "TRAIN_GPUS=${TRAIN_GPUS} must be divisible by SP_SIZE=${SP_SIZE}" >&2
    exit 1
fi

# Keep optimizer-step global batch aligned with the baseline recipe.
# baseline: global_batch = micro_batch_size * training_num_gpus_per_node = 1 * 8
# USP:      global_batch = 1 * (TRAIN_GPUS / SP_SIZE) * draft_accumulation_steps
# => draft_accumulation_steps = SP_SIZE when micro_batch_size=1
DRAFT_ACCUMULATION_STEPS="${DRAFT_ACCUMULATION_STEPS:-$SP_SIZE}"

# Work around Mooncake memcpy worker crashes in TrainerActor initialization.
export MC_STORE_MEMCPY="${MC_STORE_MEMCPY:-0}"
export TORCHSPEC_EAGLE3_PROJ_FP32="${TORCHSPEC_EAGLE3_PROJ_FP32:-0}"

if [ "${DISTRIBUTED_NODE_RANK:-0}" -eq 0 ]; then
    ray start \
      --head \
      --port "${PET_MASTER_PORT:-6379}" \
      --node-ip-address "$LOCAL_IP" \
      --num-gpus "${NUM_GPUS:-8}" \
      --temp-dir "$RAY_TEMP_DIR" \
      --disable-usage-stats \
      --include-dashboard=false

    export RAY_ADDRESS="${LOCAL_IP}:${PET_MASTER_PORT:-6379}"

    exec > >(tee -a "$LOG_FILE") 2>&1

    echo "=============================================="
    echo "TorchSpec Llama-3.1-8B USP alignment run (fix_grad_proj_bf16)"
    echo "=============================================="
    echo "Config:                 $CONFIG_FILE"
    echo "Working dir:            $WORKING_DIR"
    echo "CUDA_VISIBLE_DEVICES:   $CUDA_VISIBLE_DEVICES"
    echo "Visible GPUs:           $TOTAL_GPUS"
    echo "Training GPUs:          $TRAIN_GPUS"
    echo "Inference GPUs:         $INFERENCE_GPUS (colocate=true)"
    echo "USP topology:           ulysses=$SP_ULYSSES_SIZE ring=$SP_RING_SIZE"
    echo "Draft accumulation:     $DRAFT_ACCUMULATION_STEPS"
    echo "MC_STORE_MEMCPY:        $MC_STORE_MEMCPY"
    echo "Eagle3 proj fp32:       $TORCHSPEC_EAGLE3_PROJ_FP32"
    echo "RAY_ADDRESS:            $RAY_ADDRESS"
    echo "Output dir:             $OUTPUT_DIR"
    echo "Cache dir:              $CACHE_DIR"
    echo "TensorBoard logdir:     $TENSORBOARD_LOGDIR"
    echo "Effective global batch: $((TRAIN_GPUS * 1 * 1))"
    echo "Log file:               $LOG_FILE"
    echo "Extra args:             $*"
    echo "=============================================="

    python3 -m torchspec.train_entry \
      --config "$CONFIG_FILE" \
      training.training_num_gpus_per_node="$TRAIN_GPUS" \
      inference.inference_num_gpus="$INFERENCE_GPUS" \
      training.num_epochs=10 \
      training.attention_backend=usp \
      training.sp_ulysses_size="$SP_ULYSSES_SIZE" \
      training.sp_ring_size="$SP_RING_SIZE" \
      training.draft_accumulation_steps="$DRAFT_ACCUMULATION_STEPS" \
      training.train_env_vars="'{\"TORCHSPEC_EAGLE3_PROJ_FP32\":\"$TORCHSPEC_EAGLE3_PROJ_FP32\"}'" \
      output_dir="$OUTPUT_DIR" \
      cache_dir="$CACHE_DIR" \
      logging.use_tensorboard=true \
      logging.tensorboard_dir="$TENSORBOARD_LOGDIR" \
      "$@"

    echo "=============================================="
    echo "USP fix_grad_proj_bf16 training completed!"
    echo "=============================================="
else
    echo "This recipe is single-node only; non-zero ranks will stay idle."
    sleep infinity
fi

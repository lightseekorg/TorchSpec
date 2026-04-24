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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/nfs/ofs-llab-volume/users/fengyu/hf_cache}"
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-/nfs/ofs-fengyu/cache/deep_gemm}"
mkdir -p "$SGLANG_DG_CACHE_DIR"

RAY_TMP_ROOT="${RAY_TMP_ROOT:-/tmp}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-$RAY_TMP_ROOT/ray_ts_l31_$(id -u)}"
export RAY_TMPDIR="$RAY_TEMP_DIR"
export RAY_TEMP_DIR="$RAY_TMPDIR"
mkdir -p "$RAY_TEMP_DIR"

LOCAL_IP="$(ip -4 addr show scope global | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1)"
LOG_SUFFIX="$(printf "%s" "$LOCAL_IP" | tr '.' '-')"
DATE_TAG="$(date +%m%d_%H%M)"
RAY_LOG_ROOT="$WORKING_DIR/tmp/ray_${DATE_TAG}_${LOG_SUFFIX}"
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
trap "kill $RAY_LOG_SYNC_PID 2>/dev/null || true; sync_ray_logs; ray stop --force >/dev/null 2>&1 || true" EXIT

ray stop --force || true
export RAY_DISABLE_DASHBOARD=1
export RAY_DISABLE_METRICS=1
export RAY_ADDRESS="auto"

if [ "${DISTRIBUTED_NODE_RANK:-0}" -eq 0 ]; then
    ray start \
      --head \
      --port "${PET_MASTER_PORT:-6379}" \
      --node-ip-address "$LOCAL_IP" \
      --num-gpus "${NUM_GPUS:-8}" \
      --temp-dir "$RAY_TEMP_DIR" \
      --disable-usage-stats \
      --include-dashboard=false

    bash examples/llama31-8b-align-single-node/run.sh "$@"
else
    echo "This recipe is single-node only; non-zero ranks will stay idle."
    sleep infinity
fi

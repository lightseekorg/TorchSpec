#!/bin/bash
# Test all Mooncake transfer path combinations for the force delete refactoring.
#
# Paths tested:
#   1. TCP  + host buffer async (default path)
#   2. RDMA + host buffer async
#   3. RDMA + GPU Direct sync
#   4. TCP  + GPU Direct sync (GDR over TCP, uncommon but valid)
#
# Each path runs 3 training steps to verify put/get/delete work end-to-end.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export TORCHSPEC_LOG_LEVEL=INFO

CONFIG="$ROOT_DIR/configs/sglang_qwen3_8b.yaml"
STEPS=${1:-30}
PASS=0
FAIL=0
RESULTS=()

run_test() {
    local name="$1"
    shift
    local logfile="/tmp/ts_path_test_${name}.log"

    echo ""
    echo "========================================"
    echo "TEST: $name"
    echo "  Args: $*"
    echo "========================================"

    # Stop any leftover Ray
    ray stop --force 2>/dev/null || true
    sleep 2

    set +e
    python3 -m torchspec.train_entry \
        --config "$CONFIG" \
        training.training_num_gpus_per_node=2 \
        inference.inference_num_gpus=2 \
        inference.inference_num_gpus_per_engine=2 \
        inference.inference_num_gpus_per_node=4 \
        inference.sglang.tp_size=2 \
        training.num_train_steps=$STEPS \
        "$@" \
        > "$logfile" 2>&1
    local rc=$?
    set -e

    # Check for training completion in Ray worker logs
    local step_count
    step_count=$(grep -c "step.*${STEPS}/${STEPS}" /tmp/ray/session_latest/logs/worker*.err 2>/dev/null || echo 0)
    local delete_errors
    delete_errors=$(grep -c "force delete abandoned" /tmp/ray/session_latest/logs/worker*.err 2>/dev/null || echo 0)
    local put_errors
    put_errors=$(grep -c "batch_put_from failed" /tmp/ray/session_latest/logs/worker*.err 2>/dev/null || echo 0)

    if [ "$step_count" -ge 1 ] && [ "$delete_errors" -eq 0 ] && [ "$put_errors" -eq 0 ]; then
        echo "  RESULT: PASS (${STEPS} steps completed, 0 delete errors, 0 put errors)"
        PASS=$((PASS + 1))
        RESULTS+=("PASS: $name")
    else
        echo "  RESULT: FAIL (steps=$step_count, delete_errors=$delete_errors, put_errors=$put_errors, exit=$rc)"
        echo "  Log: $logfile"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL: $name (steps=$step_count, del_err=$delete_errors, put_err=$put_errors)")
    fi
}

echo "========================================"
echo "Mooncake Transfer Path Tests"
echo "========================================"

# Test 1: TCP + host buffer async (default)
run_test "tcp_host_async" \
    mooncake.protocol=tcp \
    mooncake.enable_gpu_direct=false

# Test 2: RDMA + host buffer async
run_test "rdma_host_async" \
    mooncake.protocol=rdma \
    mooncake.device_name=mlx5_0 \
    mooncake.enable_gpu_direct=false

# Test 3: RDMA + GPU Direct
run_test "rdma_gpu_direct" \
    mooncake.protocol=rdma \
    mooncake.device_name=mlx5_0 \
    mooncake.enable_gpu_direct=true

# Test 4: TCP + GPU Direct
run_test "tcp_gpu_direct" \
    mooncake.protocol=tcp \
    mooncake.enable_gpu_direct=true

# Summary
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "Total: $PASS passed, $FAIL failed"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi

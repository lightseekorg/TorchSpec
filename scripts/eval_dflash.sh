#!/bin/bash
# Evaluate a trained DFlash draft model on standard benchmarks.
#
# Prerequisites:
#   - Trained DFlash checkpoint converted to HF format (via extract_dflash_checkpoint.py)
#   - SGLang with DFlash support installed
#   - SpecForge benchmarks available at $SPECFORGE_DIR
#
# Usage:
#   # With Qwen3-8B target model
#   bash scripts/eval_dflash.sh <draft_model_path> [target_model] [tp_size]
#
# Example:
#   bash scripts/eval_dflash.sh ./outputs/dflash_repro/draft_model Qwen/Qwen3-8B 2
#
# Benchmarks: gsm8k, math500, aime (2024+2025), humaneval, livecodebench, mtbench

set -euo pipefail

DRAFT_MODEL="${1:?Usage: $0 <draft_model_path> [target_model] [tp_size]}"
TARGET_MODEL="${2:-Qwen/Qwen3-8B}"
TP_SIZE="${3:-1}"
PORT="${PORT:-30000}"
SPECFORGE_DIR="${SPECFORGE_DIR:-$(dirname $(dirname $0))/../SpecForge}"

if [ ! -d "$SPECFORGE_DIR/benchmarks" ]; then
    echo "Error: SpecForge benchmarks not found at $SPECFORGE_DIR/benchmarks"
    echo "Set SPECFORGE_DIR to the SpecForge repository root."
    exit 1
fi

echo "=============================================="
echo "DFlash Evaluation"
echo "=============================================="
echo "Target model: $TARGET_MODEL"
echo "Draft model:  $DRAFT_MODEL"
echo "TP size:      $TP_SIZE"
echo "Port:         $PORT"
echo "=============================================="

# Step 1: Launch SGLang server with DFlash speculative decoding
echo "[1/3] Launching SGLang server..."

python3 -m sglang.launch_server \
    --model "$TARGET_MODEL" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 1 \
    --tp "$TP_SIZE" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 &

SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:${PORT}', timeout=600)
print('Server ready!')
"

echo "[2/3] Running benchmarks..."

cd "$SPECFORGE_DIR/benchmarks"

# Run all benchmarks from PR #64 validation table
# config-list: batch_size,steps,topk,num_draft_tokens
#   1,0,0,0 = baseline (no speculation)
#   1,0,0,16 = DFlash with block_size=16
python3 bench_eagle3.py \
    --model "$TARGET_MODEL" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --port "$PORT" \
    --skip-launch-server \
    --config-list 1,0,0,0 1,0,0,16 \
    --benchmark-list \
        gsm8k:200 \
        math500:200 \
        aime:200 \
        humaneval:200 \
        livecodebench:200 \
        mtbench:80 \
    --output-dir ./results/dflash_eval \
    --name dflash_repro \
    --dtype bfloat16

echo "[3/3] Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $SPECFORGE_DIR/benchmarks/results/dflash_eval/"
echo "=============================================="

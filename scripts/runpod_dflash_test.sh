#!/usr/bin/env bash
# RunPod DFlash test setup script.
#
# Usage (on RunPod H100 pod):
#   bash scripts/runpod_dflash_test.sh
#
# Works with any RunPod image — auto-installs Python 3.12 if the
# system Python is too old (TorchSpec requires >= 3.12).
#
# Prerequisites:
#   - RunPod pod with H100 GPU (80GB recommended)
#   - Network volume mounted at /workspace (optional, for persistence)
set -euo pipefail

WORKSPACE="${RUNPOD_VOLUME_PATH:-/workspace}"
REQUIRED_PY_MAJOR=3
REQUIRED_PY_MINOR=11
VENV_DIR="$WORKSPACE/.venv-torchspec"

cd "$WORKSPACE"

echo "============================================"
echo "  DFlash Test Setup — RunPod H100"
echo "============================================"

# ── 1. Python version check + install ────────────────────────────────────────
echo ""
echo "[1/7] Checking Python version..."

get_python_version() {
    "$1" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0"
}

get_python_minor() {
    "$1" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0"
}

# Try to find a suitable Python (prefer python3.12, then python3, then python)
PYTHON=""
for candidate in python3.11 python3.12 python3.13 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        minor=$(get_python_minor "$candidate")
        if [ "$minor" -ge "$REQUIRED_PY_MINOR" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -n "$PYTHON" ]; then
    echo "  Found: $PYTHON ($(get_python_version "$PYTHON"))"
else
    echo "  System Python is $(get_python_version python3) — need >= $REQUIRED_PY_MAJOR.$REQUIRED_PY_MINOR"
    echo "  Installing Python $REQUIRED_PY_MAJOR.$REQUIRED_PY_MINOR..."

    apt-get update -qq
    apt-get install -y -qq software-properties-common > /dev/null 2>&1
    add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev > /dev/null 2>&1

    PYTHON="python3.11"
    if ! command -v "$PYTHON" &>/dev/null; then
        echo "  ERROR: Failed to install Python 3.11+."
        echo "  Recommended: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
        exit 1
    fi
    echo "  Installed: $PYTHON ($(get_python_version "$PYTHON"))"
fi

# ── 2. Create/reuse venv ─────────────────────────────────────────────────────
echo ""
echo "[2/7] Setting up Python venv..."

if [ -d "$VENV_DIR" ]; then
    VENV_PY_VER=$(get_python_version "$VENV_DIR/bin/python" 2>/dev/null || echo "0.0")
    VENV_PY_MINOR=$(echo "$VENV_PY_VER" | cut -d. -f2)
    if [ "$VENV_PY_MINOR" -ge "$REQUIRED_PY_MINOR" ]; then
        echo "  Reusing existing venv ($VENV_PY_VER)"
    else
        echo "  Existing venv is Python $VENV_PY_VER — recreating..."
        rm -rf "$VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR"
        echo "  Created fresh venv"
    fi
else
    "$PYTHON" -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi

# Activate venv for the rest of the script
export PATH="$VENV_DIR/bin:$PATH"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

$PIP install --upgrade pip --quiet 2>&1 | tail -1

echo "  Python: $($PY --version)"
echo "  Pip:    $($PIP --version | cut -d' ' -f1-2)"

# ── 3. System deps ───────────────────────────────────────────────────────────
echo ""
echo "[3/7] System dependencies..."
apt-get update -qq 2>/dev/null && apt-get install -y -qq git vim htop tmux > /dev/null 2>&1
echo "  Done."

# ── 4. Clone / update TorchSpec ──────────────────────────────────────────────
echo ""
echo "[4/7] Setting up TorchSpec..."
if [ ! -d "$WORKSPACE/TorchSpec" ]; then
    git clone https://github.com/torchspec-project/TorchSpec.git "$WORKSPACE/TorchSpec"
    cd "$WORKSPACE/TorchSpec"
    git checkout feature/dflash-training
else
    cd "$WORKSPACE/TorchSpec"
    git fetch origin
    git checkout feature/dflash-training
    git pull origin feature/dflash-training || true
fi
echo "  Branch: $(git branch --show-current)"
echo "  Commit: $(git log -1 --format='%h %s')"

# ── 5. Install TorchSpec + deps ──────────────────────────────────────────────
echo ""
echo "[5/7] Installing TorchSpec + dependencies..."
cd "$WORKSPACE/TorchSpec"

# Install PyTorch with CUDA first (skip if already present)
$PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "  Installing PyTorch with CUDA..."
    $PIP install torch --quiet 2>&1 | tail -2
}

$PIP install -e ".[dev]" --quiet 2>&1 | tail -3
echo "  Done."

# ── 6. Verify environment ───────────────────────────────────────────────────
echo ""
echo "[6/7] Verifying environment..."
$PY -c "
import sys
import torch

print(f'  Python:  {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda or \"N/A\"}')
print(f'  GPUs:    {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')

if not torch.cuda.is_available():
    print()
    print('  WARNING: CUDA not available! GPU tests will be skipped.')

from torchspec import DFlashModel, DFlashConfig, DFlashDraftModel
print()
print('  TorchSpec imports OK')
"

# ── 7. Run tests ─────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Running DFlash tests..."
echo ""
echo "── Unit tests ──────────────────────────────────────────"
cd "$WORKSPACE/TorchSpec"
$PY -m pytest tests/test_dflash.py -v --tb=short 2>&1

echo ""
echo "── CUDA smoke test (FlexAttention + training loop) ────"
$PY -c "
import torch

if not torch.cuda.is_available():
    print('  SKIPPED — no CUDA available')
    exit(0)

from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from torchspec.models.dflash import DFlashModel

torch.manual_seed(42)
device = 'cuda'
H, V = 128, 1024
num_target_layers = 2

config = DFlashConfig(
    hidden_size=H, intermediate_size=512,
    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    vocab_size=V, rms_norm_eps=1e-6, max_position_embeddings=2048,
    num_target_layers=num_target_layers, target_hidden_size=H,
    target_num_hidden_layers=12, mask_token_id=V-1,
)

draft_model = DFlashDraftModel(config).to(device=device, dtype=torch.bfloat16)
draft_model.freeze_embedding()
model = DFlashModel(
    draft_model=draft_model, block_size=8, num_anchors=16, loss_decay_gamma=7.0,
).to(device)

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

B, seq_len = 2, 256
input_ids = torch.randint(0, V, (B, seq_len), device=device)
hs_list = [torch.randn(B, seq_len, H, device=device, dtype=torch.bfloat16) for _ in range(num_target_layers)]
loss_mask = torch.ones(B, seq_len, device=device)
lm_head_weight = torch.randn(V, H, device=device, dtype=torch.bfloat16)

model.train()
for step in range(5):
    optimizer.zero_grad()
    loss, acc = model(input_ids=input_ids, hidden_states_list=hs_list, loss_mask=loss_mask, lm_head_weight=lm_head_weight)
    loss.backward()
    optimizer.step()
    print(f'  Step {step}: loss={loss.item():.4f}, acc={acc.item():.4f}')

print()
print('  CUDA smoke test PASSED')
print(f'  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
"

echo ""
echo "============================================"
echo "  All tests passed!"
echo "============================================"
echo ""
echo "To re-run tests later:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $WORKSPACE/TorchSpec"
echo "  pytest tests/test_dflash.py -v"

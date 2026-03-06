#!/bin/bash

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Parse command line arguments
# Usage: ./build_conda.sh [MODE] [BACKEND]
#   MODE:
#     1       - Create new micromamba env and install (default)
#     current - Install into current environment
#     0       - Skip env creation and installation
#   BACKEND:
#     sglang  - Install SGLang only (default)
#     vllm    - Install vLLM only
#     both    - Install both backends

MODE="${1:-1}"
BACKEND="${2:-sglang}"

# Validate backend
if [[ ! "$BACKEND" =~ ^(sglang|vllm|both)$ ]]; then
    echo "Error: Invalid backend '$BACKEND'"
    echo "Usage: $0 [MODE] [BACKEND]"
    echo "  BACKEND options: sglang (default), vllm, both"
    exit 1
fi

echo "=========================================="
echo "TorchSpec Installation"
echo "Backend: $BACKEND"
echo "=========================================="

if [ "$MODE" = "1" ]; then
    if ! command -v micromamba &> /dev/null; then
        echo "Error: micromamba is not installed."
        echo "Please install it first: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
        exit 1
    fi

    # Initialize micromamba for this script
    export MAMBA_EXE="${MAMBA_EXE:-$(command -v micromamba)}"
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "$("$MAMBA_EXE" shell hook --shell bash)"

    micromamba create -n torchspec python=3.12 uv -c conda-forge -y
elif [ "$MODE" = "current" ]; then
    echo "Using current environment: $(python3 --version), $(which python3)"
else
    echo "Skipping micromamba setup (mode=0)"
fi

# Install SGLang if requested
if [ "$BACKEND" = "sglang" ] || [ "$BACKEND" = "both" ]; then
    echo "=========================================="
    echo "Installing SGLang..."
    echo "=========================================="

    SGLANG_VERSION="${SGLANG_VERSION:-v0.5.8.post1}"
    SGLANG_COMMIT=0f2df9370a1de1b4fb11b071d39ab3ce2287a350
    SGLANG_FOLDER_NAME="_sglang"

    # Install sglang inside the conda environment
    if [ ! -d "$PROJECT_ROOT/$SGLANG_FOLDER_NAME" ]; then
        git clone https://github.com/sgl-project/sglang.git "$PROJECT_ROOT/$SGLANG_FOLDER_NAME"
    fi

    # Avoid pythonpath conflict, because we are using the offline engine.
    cd "$PROJECT_ROOT/$SGLANG_FOLDER_NAME"
    git checkout $SGLANG_COMMIT
    git reset --hard HEAD

    cd "$PROJECT_ROOT"

    if [ "$MODE" = "1" ]; then
        micromamba run -n torchspec pip install -e "${SGLANG_FOLDER_NAME}/python[all]"
    elif [ "$MODE" = "current" ]; then
        pip install -e "${SGLANG_FOLDER_NAME}/python[all]"
    fi

    cd "$PROJECT_ROOT/$SGLANG_FOLDER_NAME"

    # Apply sglang patch (matches Docker build behavior)
    git apply "$PROJECT_ROOT/patches/sglang/$SGLANG_VERSION/sglang.patch"

    cd "$PROJECT_ROOT"
fi

# Install vLLM if requested
if [ "$BACKEND" = "vllm" ] || [ "$BACKEND" = "both" ]; then
    echo "=========================================="
    echo "Installing vLLM..."
    echo "=========================================="

    if [ "$MODE" = "1" ]; then
        micromamba run -n torchspec uv pip install "vllm>=0.16.0"
    elif [ "$MODE" = "current" ]; then
        pip install "vllm>=0.16.0"
    fi
fi

# Install torchspec with appropriate extras
if [ "$MODE" = "1" ]; then
    echo "=========================================="
    echo "Installing TorchSpec..."
    echo "=========================================="

    EXTRAS="dev"
    if [ "$BACKEND" = "vllm" ]; then
        EXTRAS="dev,vllm"
    elif [ "$BACKEND" = "both" ]; then
        EXTRAS="dev,vllm"
    fi

    micromamba run -n torchspec uv pip install -e ".[$EXTRAS]"

    echo ""
    echo "=========================================="
    echo "✓ TorchSpec environment setup complete!"
    echo "=========================================="
    echo "Activate with: micromamba activate torchspec"
    echo ""
    if [ "$BACKEND" = "sglang" ]; then
        echo "Backend: SGLang"
        echo "Run: ./examples/qwen3-8b-single-node/run.sh"
    elif [ "$BACKEND" = "vllm" ]; then
        echo "Backend: vLLM"
        echo "Run: ./examples/qwen3-8b-single-node/run.sh --config configs/vllm_qwen3_8b.yaml"
    elif [ "$BACKEND" = "both" ]; then
        echo "Backends: SGLang + vLLM"
        echo "SGLang: ./examples/qwen3-8b-single-node/run.sh"
        echo "vLLM:   ./examples/qwen3-8b-single-node/run.sh --config configs/vllm_qwen3_8b.yaml"
    fi
elif [ "$MODE" = "current" ]; then
    EXTRAS="dev"
    if [ "$BACKEND" = "vllm" ]; then
        EXTRAS="dev,vllm"
    elif [ "$BACKEND" = "both" ]; then
        EXTRAS="dev,vllm"
    fi

    pip install -e ".[$EXTRAS]"

    echo ""
    echo "=========================================="
    echo "✓ TorchSpec installed into current environment!"
    echo "=========================================="
else
    echo ""
    echo "Skipping package installation (mode=0)"
    echo "Please install packages manually:"
    if [ "$BACKEND" = "sglang" ]; then
        echo "  pip install -e \"${SGLANG_FOLDER_NAME}/python[all]\""
        echo "  pip install -e \".[dev]\""
    elif [ "$BACKEND" = "vllm" ]; then
        echo "  pip install vllm>=0.16.0"
        echo "  pip install -e \".[dev,vllm]\""
    elif [ "$BACKEND" = "both" ]; then
        echo "  pip install -e \"${SGLANG_FOLDER_NAME}/python[all]\""
        echo "  pip install vllm>=0.16.0"
        echo "  pip install -e \".[dev,vllm]\""
    fi
fi

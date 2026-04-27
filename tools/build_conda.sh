#!/bin/bash

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Parse command line arguments
# Usage: ./build_conda.sh [MODE] [BACKEND]
#   MODE:
#     1       - Create a new micromamba/conda env and install (default)
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

ENV_MANAGER=""
ENV_CREATE_CMD=()
ENV_RUN_CMD=()
ACTIVATE_HINT=""

if command -v micromamba &> /dev/null; then
    ENV_MANAGER="micromamba"
    export MAMBA_EXE="${MAMBA_EXE:-$(command -v micromamba)}"
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    ENV_CREATE_CMD=("$MAMBA_EXE" create -n torchspec python=3.12 uv -c conda-forge -y)
    ENV_RUN_CMD=("$MAMBA_EXE" run -n torchspec)
    ACTIVATE_HINT="micromamba activate torchspec"
elif command -v conda &> /dev/null; then
    ENV_MANAGER="conda"
    ENV_CREATE_CMD=(conda create -n torchspec python=3.12 uv -c conda-forge -y)
    ENV_RUN_CMD=(conda run -n torchspec)
    ACTIVATE_HINT="conda activate torchspec"
fi

if [ "$MODE" = "1" ]; then
    if [ -z "$ENV_MANAGER" ]; then
        echo "Error: neither micromamba nor conda is installed."
        echo "Please install one of them first:"
        echo "  micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
        echo "  conda:      https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html"
        exit 1
    fi

    "${ENV_CREATE_CMD[@]}"
elif [ "$MODE" = "current" ]; then
    echo "Using current environment: $(python3 --version), $(which python3)"
else
    echo "Skipping environment setup (mode=0)"
fi

# Install SGLang if requested
if [ "$BACKEND" = "sglang" ] || [ "$BACKEND" = "both" ]; then
    echo "=========================================="
    echo "Installing SGLang..."
    echo "=========================================="

    SGLANG_VERSION="${SGLANG_VERSION:-v0.5.10.post1}"
    SGLANG_COMMIT=94f03a39dbd39edfc2b118b5357bbbadaaa9ad28
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
        "${ENV_RUN_CMD[@]}" pip install -e "${SGLANG_FOLDER_NAME}/python[all]"
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
        "${ENV_RUN_CMD[@]}" uv pip install "vllm>=0.16.0"
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

    "${ENV_RUN_CMD[@]}" uv pip install -e ".[$EXTRAS]"

    echo ""
    echo "=========================================="
    echo "✓ TorchSpec environment setup complete!"
    echo "=========================================="
    echo "Activate with: $ACTIVATE_HINT"
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

#!/bin/bash
#SBATCH --job-name=torchspec-gpu-ci
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=06:00:00

set -euo pipefail

: "${TORCHSPEC_CI_IMAGE:?TORCHSPEC_CI_IMAGE is required}"
: "${TORCHSPEC_CI_REPO_DIR:?TORCHSPEC_CI_REPO_DIR is required}"
: "${TORCHSPEC_CI_ARTIFACT_DIR:?TORCHSPEC_CI_ARTIFACT_DIR is required}"
: "${TORCHSPEC_CI_TMP_DIR:?TORCHSPEC_CI_TMP_DIR is required}"
: "${TORCHSPEC_CI_MODEL_CACHE_HOST:?TORCHSPEC_CI_MODEL_CACHE_HOST is required}"
: "${TORCHSPEC_CI_MODEL_REVISION:?TORCHSPEC_CI_MODEL_REVISION is required}"

container_mounts="${TORCHSPEC_CI_REPO_DIR}:/workspace:ro,${TORCHSPEC_CI_ARTIFACT_DIR}:/artifacts,${TORCHSPEC_CI_TMP_DIR}:/ci-tmp,${TORCHSPEC_CI_MODEL_CACHE_HOST}:/model-cache:ro"

srun --gpus=2 \
  --container-image="${TORCHSPEC_CI_IMAGE}" \
  --container-mounts="${container_mounts}" \
  bash -lc '
    set -euo pipefail
    export RUNNER_TEMP=/ci-tmp
    export HF_HOME=/ci-tmp/huggingface
    export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export TORCHSPEC_CI_ARTIFACT_DIR=/artifacts
    export TORCHSPEC_CI_MODEL_PATH="/model-cache/snapshots/${TORCHSPEC_CI_MODEL_REVISION}"
    export TORCHSPEC_CI_MODEL_CACHE=/ci-tmp/huggingface
    mkdir -p "${RUNNER_TEMP}" "${HF_HOME}" "${TORCHSPEC_CI_ARTIFACT_DIR}"

    nvidia-smi | tee "${TORCHSPEC_CI_ARTIFACT_DIR}/nvidia-smi.txt"
    python3 - <<'PY' | tee "${TORCHSPEC_CI_ARTIFACT_DIR}/runtime.txt"
import platform

import torch
import vllm

print(f"platform={platform.platform()}")
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
print(f"cuda_devices={torch.cuda.device_count()}")
if torch.cuda.device_count() != 2:
    raise SystemExit("Expected exactly two visible CUDA devices")
PY

    cd /workspace
    tools/ci/run_2gpu_training_smoke.sh
  '

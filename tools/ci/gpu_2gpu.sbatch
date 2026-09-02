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
: "${TORCHSPEC_CI_EXPECTED_VLLM_COMMIT:?TORCHSPEC_CI_EXPECTED_VLLM_COMMIT is required}"
: "${TORCHSPEC_CI_EXPECTED_VLLM_IMAGE_TAG:?TORCHSPEC_CI_EXPECTED_VLLM_IMAGE_TAG is required}"

container_mounts="${TORCHSPEC_CI_REPO_DIR}:/workspace:ro,${TORCHSPEC_CI_ARTIFACT_DIR}:/artifacts,${TORCHSPEC_CI_TMP_DIR}:/ci-tmp,${TORCHSPEC_CI_MODEL_CACHE_HOST}:/model-cache:ro"

gpu_count="${TORCHSPEC_CI_GPU_COUNT:-2}"

srun --gpus="${gpu_count}" \
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
    export TORCHSPEC_CI_GPU_COUNT="${TORCHSPEC_CI_GPU_COUNT:-2}"
    run_script="${TORCHSPEC_CI_RUN_SCRIPT:-tools/ci/run_2gpu_training_smoke.sh}"
    mkdir -p "${RUNNER_TEMP}" "${HF_HOME}" "${TORCHSPEC_CI_ARTIFACT_DIR}"

    if [[ "${TORCHSPEC_CI_MODE:-standard}" == "pp_convergence" ]]; then
      /workspace/tools/ci/apply_pp_runtime_overlays.sh \
        | tee "${TORCHSPEC_CI_ARTIFACT_DIR}/pp-runtime-overlays.txt"
    fi

    nvidia-smi | tee "${TORCHSPEC_CI_ARTIFACT_DIR}/nvidia-smi.txt"
    python3 - <<'PY' | tee "${TORCHSPEC_CI_ARTIFACT_DIR}/runtime.txt"
import os
import platform
from pathlib import Path

import torch
import vllm

print(f"platform={platform.platform()}")
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
print(f"cuda_devices={torch.cuda.device_count()}")
expected_commit = os.environ["TORCHSPEC_CI_EXPECTED_VLLM_COMMIT"]
actual_commit = os.environ.get("VLLM_BUILD_COMMIT", "")
print(f"vllm_build_commit={actual_commit}")
if actual_commit != expected_commit:
    actual_commit_display = actual_commit or "<missing>"
    raise SystemExit(
        f"Unexpected vLLM build commit: expected {expected_commit}, got {actual_commit_display}"
    )
expected_image = os.environ["TORCHSPEC_CI_EXPECTED_VLLM_IMAGE_TAG"]
actual_image = os.environ.get("VLLM_IMAGE_TAG", "")
print(f"vllm_image_tag={actual_image}")
if actual_image != expected_image:
    actual_image_display = actual_image or "<missing>"
    raise SystemExit(
        f"Unexpected vLLM image: expected {expected_image}, got {actual_image_display}"
    )
if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") not in {"1", "true", "True"}:
    raise SystemExit("Expected VLLM_USE_V2_MODEL_RUNNER=1")
runtime_root = Path(vllm.__file__).resolve().parent.parent
runner_source = (runtime_root / "vllm/v1/worker/gpu/model_runner.py").read_text()
for marker in ("self.extract_hidden_states", "uses_extract_hidden_states"):
    if marker not in runner_source:
        raise SystemExit(f"Patched vLLM runtime marker is missing: {marker}")
print("vllm_pp_extract_hidden_states_patch=present")
expected = int(os.environ["TORCHSPEC_CI_GPU_COUNT"])
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Expected exactly {expected} visible CUDA devices")
if os.environ.get("TORCHSPEC_CI_MODE") == "pp_convergence":
    model_source = (runtime_root / "vllm/model_executor/models/qwen3_next.py").read_text()
    for marker in (
        "if self.start_layer == 0:",
        "return intermediate_tensors, aux_hidden_states",
    ):
        if marker not in model_source:
            raise SystemExit(f"Qwen3-Next PP runtime marker is missing: {marker}")
    print("qwen3_next_pp_aux_hidden_states_patch=present")
PY

    cd /workspace
    "${run_script}"
  '

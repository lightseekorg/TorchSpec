#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="${TORCHSPEC_CI_CONFIG:-${repo_root}/configs/ci/vllm_qwen3_8b_2gpu_smoke.yaml}"
fixture="${TORCHSPEC_CI_FIXTURE:-${repo_root}/tests/fixtures/ci_training_smoke.jsonl}"
artifact_dir="${TORCHSPEC_CI_ARTIFACT_DIR:-${RUNNER_TEMP:-/tmp}/torchspec-2gpu-training}"
model="${TORCHSPEC_CI_MODEL:-Qwen/Qwen3-8B}"
model_cache="${TORCHSPEC_CI_MODEL_CACHE:-${HF_HOME:-${artifact_dir}/huggingface}}"
compile_cache="${TORCHSPEC_CI_COMPILE_CACHE:-${RUNNER_TEMP:-/tmp}/torchspec-torchinductor}"
train_log="${artifact_dir}/training.log"

mkdir -p "${artifact_dir}" "${model_cache}" "${compile_cache}" "${artifact_dir}/actor-logs"
export HF_HOME="${model_cache}"
export TORCHSPEC_LOG_DIR="${artifact_dir}/actor-logs"
export TORCHSPEC_LOG_LEVEL=INFO
node_ip="$(hostname -i | awk '{print $1}')"
export TORCHSPEC_PIN_NODE_IP="${node_ip}"
export TORCHINDUCTOR_CACHE_DIR="${compile_cache}"

python3 - <<'PY'
import torch

count = torch.cuda.device_count()
print(f"CI_CUDA_DEVICE_COUNT={count}")
if count != 2:
    raise SystemExit(f"Expected exactly 2 visible CUDA devices, got {count}")
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    print(f"CI_GPU index={index} name={props.name} memory={props.total_memory}")
PY

python3 - "${fixture}" "${artifact_dir}/sample.json" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[0])
messages = record["conversations"]
sample = {
    "id": record["id"],
    "input": messages[0]["content"],
    "target_output": messages[1]["content"],
}
Path(sys.argv[2]).write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
print(f"CI_SAMPLE_INPUT={sample['input']}")
print(f"CI_SAMPLE_TARGET_OUTPUT={sample['target_output']}")
PY

cd "${repo_root}"
python3 -m torchspec.train_entry \
  --config "${config}" \
  model.target_model_path="${model}" \
  model_download_dir="${model_cache}" \
  cache_dir="${artifact_dir}/cache" \
  2>&1 | tee "${train_log}"

python3 - "${train_log}" "${artifact_dir}/step-losses.json" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

pattern = re.compile(r"TRAIN_STEP step=(\d+) loss=([^ ]+)")
losses = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = pattern.search(line)
    if match:
        losses.append({"step": int(match.group(1)), "loss": float(match.group(2))})

expected_steps = [1, 2, 3]
if [item["step"] for item in losses] != expected_steps:
    raise SystemExit(f"Expected losses for steps {expected_steps}, got {losses}")
if not all(math.isfinite(item["loss"]) and item["loss"] > 0 for item in losses):
    raise SystemExit(f"Losses must be finite and positive: {losses}")

Path(sys.argv[2]).write_text(json.dumps(losses, indent=2) + "\n", encoding="utf-8")
print(f"CI_STEP_LOSSES={json.dumps(losses, separators=(',', ':'))}")
PY

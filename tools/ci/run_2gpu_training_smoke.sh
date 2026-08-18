#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eagle3_config="${TORCHSPEC_CI_EAGLE3_CONFIG:-${repo_root}/configs/ci/vllm_qwen3_8_27b_eagle3_2gpu_smoke.yaml}"
dspark_config="${TORCHSPEC_CI_DSPARK_CONFIG:-${repo_root}/configs/ci/vllm_qwen3_8_27b_dspark_2gpu_smoke.yaml}"
fixture="${TORCHSPEC_CI_FIXTURE:-${repo_root}/tests/fixtures/ci_training_smoke.jsonl}"
artifact_dir="${TORCHSPEC_CI_ARTIFACT_DIR:-${RUNNER_TEMP:-/tmp}/torchspec-2gpu-training}"
model="${TORCHSPEC_CI_MODEL:-Qwen/Qwen3.8-27B}"
model_revision="${TORCHSPEC_CI_MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
model_cache="${TORCHSPEC_CI_MODEL_CACHE:-${HF_HOME:-${artifact_dir}/huggingface}}"
compile_cache="${TORCHSPEC_CI_COMPILE_CACHE:-${RUNNER_TEMP:-/tmp}/torchspec-torchinductor}"

mkdir -p "${artifact_dir}" "${model_cache}" "${compile_cache}"
export HF_HOME="${model_cache}"
export TORCHSPEC_LOG_LEVEL=INFO
export TORCHINDUCTOR_CACHE_DIR="${compile_cache}"
node_ip="$(hostname -i | awk '{print $1}')"
export TORCHSPEC_PIN_NODE_IP="${node_ip}"

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

if [[ -n "${TORCHSPEC_CI_MODEL_PATH:-}" ]]; then
  model_snapshot="${TORCHSPEC_CI_MODEL_PATH}"
else
  python3 - "${model}" "${model_revision}" "${model_cache}" "${artifact_dir}/model-snapshot-path.txt" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    cache_dir=sys.argv[3],
)
Path(sys.argv[4]).write_text(model_path + "\n", encoding="utf-8")
print(f"CI_MODEL_SNAPSHOT={model_path}")
PY
  model_snapshot="$(tail -n 1 "${artifact_dir}/model-snapshot-path.txt")"
fi

if [[ ! -d "${model_snapshot}" ]]; then
  echo "Pinned model snapshot does not exist: ${model_snapshot}" >&2
  exit 1
fi
echo "CI_MODEL=${model}"
echo "CI_MODEL_REVISION=${model_revision}"
echo "CI_MODEL_SNAPSHOT=${model_snapshot}"

run_lane() {
  local lane="$1"
  local config="$2"
  local lane_dir="${artifact_dir}/${lane}"
  local train_log="${lane_dir}/training.log"

  mkdir -p "${lane_dir}/actor-logs"
  export TORCHSPEC_LOG_DIR="${lane_dir}/actor-logs"

  echo "CI_LANE_START=${lane}"
  cd "${repo_root}"
  python3 -m torchspec.train_entry \
    --config "${config}" \
    model.target_model_path="${model_snapshot}" \
    model_download_dir="${model_cache}" \
    cache_dir="${lane_dir}/cache" \
    2>&1 | tee "${train_log}"

  python3 - "${lane}" "${train_log}" "${lane_dir}/step-losses.json" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

lane = sys.argv[1]
pattern = re.compile(r"TRAIN_STEP step=(\d+) loss=([^ ]+)")
losses = []
for line in Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = pattern.search(line)
    if match:
        losses.append({"step": int(match.group(1)), "loss": float(match.group(2))})

expected_steps = [1, 2]
if [item["step"] for item in losses] != expected_steps:
    raise SystemExit(f"{lane}: expected losses for steps {expected_steps}, got {losses}")
if not all(math.isfinite(item["loss"]) and item["loss"] > 0 for item in losses):
    raise SystemExit(f"{lane}: losses must be finite and positive: {losses}")

Path(sys.argv[3]).write_text(json.dumps(losses, indent=2) + "\n", encoding="utf-8")
print(f"CI_STEP_LOSSES lane={lane} values={json.dumps(losses, separators=(',', ':'))}")
PY
  echo "CI_LANE_COMPLETE=${lane}"
}

run_lane eagle3 "${eagle3_config}"
run_lane dspark "${dspark_config}"

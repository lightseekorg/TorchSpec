#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eagle3_config="${TORCHSPEC_CI_EAGLE3_CONFIG:-${repo_root}/configs/ci/vllm_qwen3_8_27b_eagle3_2gpu_smoke.yaml}"
pp_convergence_config="${TORCHSPEC_CI_PP_CONVERGENCE_CONFIG:-${repo_root}/configs/ci/vllm_qwen3_8_27b_eagle3_pp_convergence.yaml}"
dspark_config="${TORCHSPEC_CI_DSPARK_CONFIG:-${repo_root}/configs/ci/vllm_qwen3_8_27b_dspark_2gpu_smoke.yaml}"
dflash2_config="${TORCHSPEC_CI_DFLASH2_CONFIG:-${repo_root}/configs/ci/vllm_qwen3_8_27b_dflash2_2gpu_smoke.yaml}"
fixture="${TORCHSPEC_CI_FIXTURE:-${repo_root}/examples/data/sample_conversations.jsonl}"
artifact_dir="${TORCHSPEC_CI_ARTIFACT_DIR:-${RUNNER_TEMP:-/tmp}/torchspec-2gpu-training}"
ci_mode="${TORCHSPEC_CI_MODE:-standard}"
expected_gpu_count="${TORCHSPEC_CI_GPU_COUNT:-2}"
model="${TORCHSPEC_CI_MODEL:-Qwen/Qwen3.8-27B}"
model_revision="${TORCHSPEC_CI_MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
long_context_dataset_id="${TORCHSPEC_CI_LONG_CONTEXT_DATASET_ID:-long_chunked_prefill_test}"
model_cache="${TORCHSPEC_CI_MODEL_CACHE:-${HF_HOME:-${artifact_dir}/huggingface}}"
compile_cache="${TORCHSPEC_CI_COMPILE_CACHE:-${RUNNER_TEMP:-/tmp}/torchspec-torchinductor}"
profile_num_steps="${TORCHSPEC_CI_PROFILE_NUM_STEPS:-}"
profile_step_start="${TORCHSPEC_CI_PROFILE_STEP_START:-}"
profile_step_end="${TORCHSPEC_CI_PROFILE_STEP_END:-}"

mkdir -p "${artifact_dir}" "${model_cache}" "${compile_cache}"
export HF_HOME="${model_cache}"
export TORCHSPEC_LOG_LEVEL=INFO
export TORCHINDUCTOR_CACHE_DIR="${compile_cache}"
node_ip="$(hostname -i | awk '{print $1}')"
export TORCHSPEC_PIN_NODE_IP="${node_ip}"

python3 - "${expected_gpu_count}" <<'PY'
import sys
import torch

count = torch.cuda.device_count()
print(f"CI_CUDA_DEVICE_COUNT={count}")
expected = int(sys.argv[1])
if count != expected:
    raise SystemExit(f"Expected exactly {expected} visible CUDA devices, got {count}")
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

fixture_records="$(python3 - "${fixture}" "${long_context_dataset_id}" <<'PY'
import json
import sys
from pathlib import Path

records = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not records:
    raise SystemExit("CI fixture must contain at least one training record")
long_context_id = sys.argv[2]
matching = [record for record in records if record.get("id") == long_context_id]
if len(matching) != 1:
    raise SystemExit(
        f"Expected exactly one long-context record with id={long_context_id!r}, got {len(matching)}"
    )
print(len(records))
PY
)"
echo "CI_LONG_CONTEXT_DATASET_ID=${long_context_dataset_id}"
echo "CI_FIXTURE_RECORDS=${fixture_records}"

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
  local expected_trainer="${3:-}"
  local lane_dir="${artifact_dir}/${lane}"
  local train_log="${lane_dir}/training.log"
  if (( $# >= 3 )); then
    shift 3
  else
    shift $#
  fi
  local -a extra_args=("$@")

  mkdir -p "${lane_dir}/actor-logs"
  export TORCHSPEC_LOG_DIR="${lane_dir}/actor-logs"

  if [[ -n "${profile_num_steps}" ]]; then
    [[ "${profile_num_steps}" =~ ^[1-9][0-9]*$ ]] || {
      echo "TORCHSPEC_CI_PROFILE_NUM_STEPS must be a positive integer" >&2
      exit 2
    }
    extra_args+=("training.num_train_steps=${profile_num_steps}")
  fi
  if [[ -n "${profile_step_start}" || -n "${profile_step_end}" ]]; then
    [[ "${profile_step_start}" =~ ^[0-9]+$ && "${profile_step_end}" =~ ^[1-9][0-9]*$ ]] || {
      echo "Both profile step bounds must be non-negative integers" >&2
      exit 2
    }
    ((profile_step_end > profile_step_start)) || {
      echo "Profile step end must be greater than profile step start" >&2
      exit 2
    }
    mkdir -p "${lane_dir}/profiles"
    extra_args+=(
      "debug.enable_perf_metrics=true"
      "debug.use_pytorch_profiler=true"
      "debug.profile_target=[train_overall]"
      "debug.profile_step_start=${profile_step_start}"
      "debug.profile_step_end=${profile_step_end}"
      "debug.profile_dir_name=${lane_dir}/profiles"
    )
  fi

  echo "CI_LANE_START=${lane}"
  cd "${repo_root}"
  if [[ -n "${expected_trainer}" ]]; then
    python3 - "${config}" "${expected_trainer}" <<'PY'
import sys

from torchspec import AutoDraftModelConfig
from torchspec.config import load_config
from torchspec.training.trainer_actor import _trainer_class_for_config

config = load_config(sys.argv[1])
draft_config = AutoDraftModelConfig.from_file(config.model.draft_model_config)
trainer_name = _trainer_class_for_config(draft_config).__name__
if trainer_name != sys.argv[2]:
    raise SystemExit(f"Expected trainer {sys.argv[2]}, got {trainer_name}")
print(f"CI_TRAINER lane={trainer_name.removesuffix('Trainer').lower()} class={trainer_name}")
PY
  fi
  python3 -m torchspec.train_entry \
    --config "${config}" \
    model.target_model_path="${model_snapshot}" \
    model_download_dir="${model_cache}" \
    cache_dir="${lane_dir}/cache" \
    "${extra_args[@]}" \
    2>&1 | tee "${train_log}"

  python3 - "${lane}" "${train_log}" "${lane_dir}/step-losses.json" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

lane = sys.argv[1]
lines = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace").splitlines()
step_count_pattern = re.compile(r"num_train_steps=(\d+)")
step_counts = [int(match.group(1)) for line in lines if (match := step_count_pattern.search(line))]
if not step_counts:
    raise SystemExit(f"{lane}: training log did not report num_train_steps")
expected_step_count = step_counts[-1]
if expected_step_count < 1:
    raise SystemExit(f"{lane}: expected at least one optimizer step, got {expected_step_count}")

loss_pattern = re.compile(r"TRAIN_STEP step=(\d+) loss=([^ ]+)")
losses = []
for line in lines:
    match = loss_pattern.search(line)
    if match:
        losses.append({"step": int(match.group(1)), "loss": float(match.group(2))})

expected_steps = list(range(1, expected_step_count + 1))
if [item["step"] for item in losses] != expected_steps:
    raise SystemExit(f"{lane}: expected losses for steps {expected_steps}, got {losses}")
if not all(math.isfinite(item["loss"]) and item["loss"] > 0 for item in losses):
    raise SystemExit(f"{lane}: losses must be finite and positive: {losses}")

Path(sys.argv[3]).write_text(json.dumps(losses, indent=2) + "\n", encoding="utf-8")
print(f"CI_EPOCH_OPTIMIZER_STEPS lane={lane} count={expected_step_count}")
print(f"CI_STEP_LOSSES lane={lane} values={json.dumps(losses, separators=(',', ':'))}")
PY
  echo "CI_LANE_COMPLETE=${lane}"
}

if [[ "${ci_mode}" == "pp_convergence" ]]; then
  run_lane tp "${pp_convergence_config}" "" \
    inference.inference_num_gpus=2 \
    inference.inference_num_gpus_per_engine=2 \
    inference.inference_num_gpus_per_node=4 \
    inference.vllm.pp_size=1
  run_lane pp "${pp_convergence_config}" "" \
    inference.inference_num_gpus=2 \
    inference.inference_num_gpus_per_engine=2 \
    inference.inference_num_gpus_per_node=4 \
    inference.vllm.pp_size=2

  max_relative_diff="${TORCHSPEC_CI_PP_MAX_REL_LOSS_DIFF:-0.01}"
  python3 "${repo_root}/tools/ci/compare_eagle3_convergence.py" \
    "${artifact_dir}/tp/step-losses.json" \
    "${artifact_dir}/pp/step-losses.json" \
    "${artifact_dir}/pp-convergence.json" \
    --max-relative-diff "${max_relative_diff}"
  exit 0
fi

run_lane eagle3 "${eagle3_config}"
run_lane dspark "${dspark_config}"
run_lane dflash2 "${dflash2_config}" DFlash2Trainer

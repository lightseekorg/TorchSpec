#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SOURCE_DIR REPORT_DIR" >&2
  exit 2
fi

source_dir="$(cd "$1" && pwd)"
report_dir="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TORCHSPEC_SLURM_ROOT:?TORCHSPEC_SLURM_ROOT is required}"
: "${TORCHSPEC_CI_IMAGE:?TORCHSPEC_CI_IMAGE is required}"
: "${TORCHSPEC_CI_MODEL_CACHE_HOST:?TORCHSPEC_CI_MODEL_CACHE_HOST is required}"
shared_root="${TORCHSPEC_SLURM_ROOT}"
run_id="${GITHUB_RUN_ID:-manual}"
run_attempt="${GITHUB_RUN_ATTEMPT:-1}"
run_key="${run_id}-${run_attempt}"
run_root="${shared_root}/runs/${run_key}"
repo_dir="${run_root}/repo"
artifact_dir="${run_root}/artifacts"
tmp_dir="${run_root}/tmp"
image="${TORCHSPEC_CI_IMAGE}"
model_cache_host="${TORCHSPEC_CI_MODEL_CACHE_HOST}"
model="${TORCHSPEC_CI_MODEL:-Qwen/Qwen3.8-27B}"
model_revision="${TORCHSPEC_CI_MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
expected_vllm_commit="${TORCHSPEC_CI_EXPECTED_VLLM_COMMIT:-e9d1398d9edfd90fcc1cf783805240e3effec013}"
expected_vllm_image_tag="${TORCHSPEC_CI_EXPECTED_VLLM_IMAGE_TAG:-vllm/vllm-openai:nightly-e9d1398d9edfd90fcc1cf783805240e3effec013}"

case "${run_key}" in
  *[!0-9-]*) echo "Unsafe run key: ${run_key}" >&2; exit 2 ;;
esac
case "${run_root}" in
  "${shared_root}"/runs/*) ;;
  *) echo "Unsafe run root: ${run_root}" >&2; exit 2 ;;
esac

[[ -f "${source_dir}/tools/ci/run_2gpu_training_smoke.sh" ]] || {
  echo "TorchSpec CI launcher is missing from ${source_dir}" >&2
  exit 2
}
[[ -f "${script_dir}/gpu_2gpu.sbatch" ]] || {
  echo "Trusted Slurm batch script is missing" >&2
  exit 2
}
[[ -f "${image}" ]] || {
  echo "TorchSpec container image is missing: ${image}" >&2
  exit 2
}
[[ -d "${model_cache_host}/snapshots/${model_revision}" ]] || {
  echo "Pinned Qwen3.8 snapshot is missing: ${model_cache_host}/snapshots/${model_revision}" >&2
  exit 2
}

mkdir -p "${shared_root}/runs" "${report_dir}"
if [[ -e "${run_root}" ]]; then
  echo "Refusing to reuse existing Slurm run directory: ${run_root}" >&2
  exit 2
fi
mkdir -p "${repo_dir}" "${artifact_dir}" "${tmp_dir}"
rsync -a --exclude=.git -- "${source_dir}/" "${repo_dir}/"

export_spec="TORCHSPEC_CI_IMAGE=${image},TORCHSPEC_CI_REPO_DIR=${repo_dir},TORCHSPEC_CI_ARTIFACT_DIR=${artifact_dir},TORCHSPEC_CI_TMP_DIR=${tmp_dir},TORCHSPEC_CI_MODEL_CACHE_HOST=${model_cache_host},TORCHSPEC_CI_MODEL=${model},TORCHSPEC_CI_MODEL_REVISION=${model_revision},TORCHSPEC_CI_EXPECTED_VLLM_COMMIT=${expected_vllm_commit},TORCHSPEC_CI_EXPECTED_VLLM_IMAGE_TAG=${expected_vllm_image_tag}"
submitted="$(sbatch \
  --parsable \
  --output="${run_root}/slurm-%j.log" \
  --export="${export_spec}" \
  "${script_dir}/gpu_2gpu.sbatch")"
job_id="${submitted%%;*}"
log_path="${run_root}/slurm-${job_id}.log"
echo "Submitted GPU cluster job"

cancel_job() {
  if squeue -h -j "${job_id}" | grep -q .; then
    echo "Cancelling Slurm job ${job_id} after dispatcher interruption" >&2
    scancel "${job_id}"
  fi
}
trap cancel_job INT TERM

while true; do
  state="$(squeue -h -j "${job_id}" -o '%T' | tr -d ' ')"
  [[ -n "${state}" ]] || break
  elapsed="$(squeue -h -j "${job_id}" -o '%M' | tr -d ' ')"
  printf 'SLURM_PROGRESS state=%s elapsed=%s\n' "${state}" "${elapsed}"
  sleep 60
done
trap - INT TERM

for _ in 1 2 3 4 5 6; do
  final_state="$(sacct -j "${job_id}" -X -n -P -o State | head -n 1 | cut -d'|' -f1)"
  [[ -n "${final_state}" ]] && break
  sleep 5
done
final_state="${final_state:-UNKNOWN}"

sacct -j "${job_id}" -X --format=State,ExitCode,Elapsed,AllocTRES -P \
  | tee "${report_dir}/sacct.txt"
[[ -f "${log_path}" ]] && cp "${log_path}" "${report_dir}/slurm.log"
[[ -d "${artifact_dir}" ]] && rsync -a "${artifact_dir}/" "${report_dir}/artifacts/"

{
  echo "## TorchSpec GPU integration"
  echo
  echo "- Final state: \`${final_state}\`"
  echo "- Source SHA: \`${TORCHSPEC_CI_SOURCE_SHA:-${GITHUB_SHA:-unknown}}\`"
  echo "- Model revision: \`${model_revision}\`"
  echo "- Expected vLLM commit: \`${expected_vllm_commit}\`"
  echo "- Expected vLLM image: \`${expected_vllm_image_tag}\`"
} | tee "${report_dir}/summary.md"

if [[ "${final_state}" != COMPLETED ]]; then
  [[ -f "${log_path}" ]] && tail -n 240 "${log_path}"
  exit 1
fi

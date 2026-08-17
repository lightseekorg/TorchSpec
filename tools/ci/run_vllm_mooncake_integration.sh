#!/usr/bin/env bash
set -euo pipefail

model="${TORCHSPEC_CI_MODEL:-Qwen/Qwen3-8B}"
artifact_dir="${TORCHSPEC_CI_ARTIFACT_DIR:-${RUNNER_TEMP:-/tmp}/torchspec-gpu-integration}"
master_port="${MOONCAKE_MASTER_PORT:-51135}"
metadata_port="${MOONCAKE_METADATA_PORT:-8763}"

mkdir -p "${artifact_dir}/tensor-dumps"

export MOONCAKE_MASTER_HOST=127.0.0.1
export MOONCAKE_MASTER_PORT="${master_port}"
export MOONCAKE_METADATA_PORT="${metadata_port}"
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_MASTER_SERVER="127.0.0.1:${master_port}"
export MOONCAKE_METADATA_SERVER="http://127.0.0.1:${metadata_port}/metadata"

master_bin="$(python3 - <<'PY'
from torchspec.transfer.mooncake.utils import resolve_mooncake_master_bin

print(resolve_mooncake_master_bin())
PY
)"

if [[ ! -x "${master_bin}" ]]; then
  echo "mooncake_master is not executable: ${master_bin}" >&2
  exit 1
fi

master_pid=""
cleanup() {
  if [[ -n "${master_pid}" ]] && kill -0 "${master_pid}" 2>/dev/null; then
    kill "${master_pid}" 2>/dev/null || true
    wait "${master_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${master_bin}" \
  --port="${master_port}" \
  --http_metadata_server_port="${metadata_port}" \
  --http_metadata_server_host=0.0.0.0 \
  --enable_http_metadata_server=true \
  >"${artifact_dir}/mooncake-master.log" 2>&1 &
master_pid=$!

python3 - "${master_port}" "${metadata_port}" <<'PY'
import socket
import sys
import time

ports = [int(value) for value in sys.argv[1:]]
deadline = time.monotonic() + 30
for port in ports:
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            if time.monotonic() >= deadline:
                raise SystemExit(f"Mooncake service did not open port {port}")
            time.sleep(0.5)
PY

python3 tests/test_vllm_engine_integration.py \
  --model "${model}" \
  --tp 1 \
  --load-format auto \
  --enforce-eager \
  --max-model-len 4096 \
  --dump-dir "${artifact_dir}/tensor-dumps" \
  2>&1 | tee "${artifact_dir}/integration.log"

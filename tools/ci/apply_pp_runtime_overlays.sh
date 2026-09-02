#!/usr/bin/env bash
set -euo pipefail

runtime_root="${VLLM_RUNTIME_ROOT:-}"
if [[ -z "${runtime_root}" ]]; then
  runtime_root="$(python3 - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"
fi

python3 - "${runtime_root}" <<'PY'
import sys
from pathlib import Path

runtime_root = Path(sys.argv[1])
model_path = runtime_root / "vllm/model_executor/models/qwen3_next.py"
runner_path = runtime_root / "vllm/v1/worker/gpu_model_runner.py"
model_source = model_path.read_text()
runner_source = runner_path.read_text()


def replace_once(source: str, old: str, new: str, description: str) -> tuple[str, bool]:
    count = source.count(old)
    if count > 1:
        raise SystemExit(f"Expected one {description}, found {count}")
    if count == 1:
        return source.replace(old, new), True
    return source, False


model_replacements = (
    (
        "    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:\n",
        "    ) -> (\n"
        "        torch.Tensor\n"
        "        | IntermediateTensors\n"
        "        | tuple[torch.Tensor | IntermediateTensors, list[torch.Tensor]]\n"
        "    ):\n",
        "model return annotation",
    ),
    (
        "        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)\n",
        "        # Capture ids are global layer positions. Only PP0 owns the embedding\n"
        "        # output at id 0; later stages must not duplicate their stage input.\n"
        "        aux_hidden_states: list[torch.Tensor] = []\n"
        "        if self.start_layer == 0:\n"
        "            self._maybe_add_hidden_state(\n"
        "                aux_hidden_states, 0, hidden_states, residual\n"
        "            )\n",
        "stage-local auxiliary capture",
    ),
    (
        "            return IntermediateTensors(\n"
        '                {"hidden_states": hidden_states, "residual": residual}\n'
        "            )\n",
        "            intermediate_tensors = IntermediateTensors(\n"
        '                {"hidden_states": hidden_states, "residual": residual}\n'
        "            )\n"
        "            if self.aux_hidden_state_layers:\n"
        "                return intermediate_tensors, aux_hidden_states\n"
        "            return intermediate_tensors\n",
        "per-stage auxiliary return",
    ),
)

model_changed = False
for old, new, description in model_replacements:
    model_source, changed = replace_once(model_source, old, new, description)
    model_changed = model_changed or changed

model_markers = (
    "tuple[torch.Tensor | IntermediateTensors, list[torch.Tensor]]",
    "if self.start_layer == 0:",
    "return intermediate_tensors, aux_hidden_states",
)
for marker in model_markers:
    if marker not in model_source:
        raise SystemExit(f"Missing model runtime change: {marker}")

barrier_blocks = (
    "            if extract_hidden_states and get_pp_group().world_size > 1:\n"
    "                get_pp_group().barrier()\n",
    "            if get_pp_group().world_size > 1:\n"
    "                get_pp_group().barrier()\n",
    "            if (\n"
    "                spec_config.uses_extract_hidden_states()\n"
    "                and get_pp_group().world_size > 1\n"
    "            ):\n"
    "                get_pp_group().barrier()\n\n",
)

removed = 0
for block in barrier_blocks:
    count = runner_source.count(block)
    if count > 1:
        raise SystemExit(f"Expected at most one PP barrier block, found {count}")
    if count == 1:
        runner_source = runner_source.replace(block, "")
        removed += 1

for forbidden in (
    "if extract_hidden_states and get_pp_group().world_size > 1:\n"
    "                get_pp_group().barrier()",
    "if get_pp_group().world_size > 1:\n                get_pp_group().barrier()",
    "spec_config.uses_extract_hidden_states()\n"
    "                and get_pp_group().world_size > 1",
):
    if forbidden in runner_source:
        raise SystemExit(f"Unexpected pipeline barrier remains: {forbidden}")

if model_changed:
    model_path.write_text(model_source)
if removed:
    runner_path.write_text(runner_source)
print(f"PP runtime changes verified: model_changed={model_changed}, barriers_removed={removed}")
PY

python3 -m compileall -q \
  "${runtime_root}/vllm/model_executor/models/qwen3_next.py" \
  "${runtime_root}/vllm/v1/worker/gpu_model_runner.py"
echo "PP runtime overlays verified"

# scripts/colocate/

Cheap-host runner for the colocate (MPS+NCCL) MPS-required tests.

Modal sandbox can't run these tests because gVisor blocks NVIDIA MPS;
this runner targets any other GPU host that supports `--ipc=host`
(RunPod, Vast.ai, Lambda, Hyperstack, bare-metal, …).

## Quick start

```bash
# On the cheap host, after `git clone` + `git checkout
# feature/colocate-training-inference`:
bash scripts/colocate/run_smoke_host.sh         # 1-GPU tiny smoke
bash scripts/colocate/run_smoke_host.sh --full  # 4-GPU full Phase-4/6/7
```

Exit code `0` = every selected test PASSED or SKIPPED cleanly.

## Full handoff doc

See **[`docs/colocate/cheap_host_test_plan.md`](../../docs/colocate/cheap_host_test_plan.md)**
for the self-contained agent-handoff plan: cost-tier matrix, RunPod /
Vast.ai setup recipes, expected output, failure-mode table, and the
report-back checklist.

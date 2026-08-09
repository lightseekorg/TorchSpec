"""Acceptance test: vLLM hidden-state export is invariant to pipeline parallelism.

Under PP each stage writes only the aux layers it owns, and the training sample
is reassembled from those fragments.  The property that protects the training
signal is that the reassembled tensors match what a single-stage run produces
for the same prompts, so this test runs both arms and compares them.

Both arms go through production code: the ``MooncakeHiddenStatesConnector``
publishes, and ``MooncakeDataset._load_from_mooncake`` reads -- which is the
only place that knows the per-layer fragment layout.

Requires a vLLM built with ``patches/vllm/<image-tag>/vllm_pp_hidden_states.patch``;
stock vLLM captures aux states on the last stage only.

The arms must share a batch composition, which is why one request per
``generate()`` call is the default.  Batching the same three prompts together
changes the GEMM shapes and moves bf16 results by up to one or two ULP -- on
Qwen3-8B that is an absolute 128 on a ~1e4 massive-activation element.  That is
not a pipeline effect: holding ``pp=2`` fixed and varying only batch
composition reproduces the identical deviation.  With composition held fixed,
pp=1 and pp=2 agree bitwise, which is what this test asserts.

Usage:
    mooncake_master --rpc_port 51135 \
        --enable_http_metadata_server=true --http_metadata_server_port=8763 &

    for phase in baseline candidate; do
        python tests/test_vllm_pp_equivalence.py --model Qwen/Qwen3-8B --pp 2 --phase $phase
    done
    python tests/test_vllm_pp_equivalence.py --model Qwen/Qwen3-8B --pp 2 --phase compare
"""

import argparse
import json
import os
import socket
from pathlib import Path

import torch
from transformers import AutoConfig

# Mooncake env has to be set before the connector builds its store.
try:
    _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _s.connect(("8.8.8.8", 80))
    LOCAL_IP = _s.getsockname()[0]
    _s.close()
except Exception:
    LOCAL_IP = "localhost"

os.environ.setdefault("MOONCAKE_MASTER_HOST", LOCAL_IP)
os.environ.setdefault("MOONCAKE_MASTER_PORT", "51135")
os.environ.setdefault("MOONCAKE_METADATA_PORT", "8763")
os.environ.setdefault("MOONCAKE_LOCAL_HOSTNAME", LOCAL_IP)
os.environ.setdefault("MOONCAKE_MASTER_SERVER", f"{LOCAL_IP}:51135")

PROMPTS: dict[str, list[int]] = {
    "short": [1, 2345, 6789],
    "medium": list(range(1000, 1200)),
    "long": list(range(2000, 2600)),
}


def resolve_aux_layer_ids(model_path: str) -> tuple[list[int], int, int]:
    """Mirror ``VllmEngine.init``'s aux-id resolution.

    vLLM captures at index ``layer_idx + 1`` after each layer, so the valid
    range is ``[0, num_hidden_layers]`` and ``num_hidden_layers`` is the
    pre-``norm`` slot used for ``last_hidden_states``.
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg = getattr(cfg, "text_config", cfg)
    num_layers = cfg.num_hidden_layers
    aux_ids = [lid + 1 for lid in (1, num_layers // 2 - 1, num_layers - 4)]
    if num_layers not in aux_ids:
        aux_ids.append(num_layers)
    return aux_ids, cfg.hidden_size, num_layers


def describe_ownership(aux_ids: list[int], num_layers: int, pp_size: int) -> dict[int, list[int]]:
    """Map pipeline rank -> aux ids it owns, using vLLM's own partitioning."""
    from vllm.distributed.utils import get_pp_indices

    ownership: dict[int, list[int]] = {}
    for pp_rank in range(pp_size):
        start_layer, end_layer = get_pp_indices(num_layers, pp_rank, pp_size)
        ownership[pp_rank] = [
            layer_id
            for layer_id in aux_ids
            if (layer_id == 0 and pp_rank == 0) or (start_layer < layer_id <= end_layer)
        ]
    return ownership


def build_engine(model_path: str, tp_size: int, pp_size: int, aux_ids: list[int], max_len: int):
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        disable_custom_all_reduce=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
        max_model_len=max_len,
        enforce_eager=True,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": list(aux_ids)}
            },
        },
        kv_transfer_config={
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_connector_module_path": (
                "torchspec.inference.engine.mooncake_hidden_states_connector"
            ),
            "kv_role": "kv_producer",
        },
        compilation_config={"cudagraph_mode": "NONE"},
    )


def read_sample(dataset, kv_params: dict) -> dict[str, torch.Tensor]:
    """Reassemble one request through the production consumer path."""
    from torchspec.training.data_fetcher import TrainSample

    metadata = None
    manifest = kv_params.get("pp_layer_manifest")
    if manifest is not None:
        metadata = {"vllm_pp_complete": True, "vllm_pp_layer_manifest": manifest}

    sample = TrainSample(
        mooncake_key=kv_params["mooncake_key"],
        tensor_shapes=kv_params["tensor_shapes"],
        tensor_dtypes=kv_params["tensor_dtypes"],
        metadata=metadata,
    )
    loaded = dataset._load_from_mooncake(sample)
    # Copy out eagerly: at batch_size=1 the PP=1 path hands back views into a
    # Mooncake host buffer that the next request is free to reuse.
    return {
        key: value.detach().to("cpu", copy=True)
        for key, value in loaded.items()
        if torch.is_tensor(value)
    }


def produce(args, aux_ids: list[int]) -> dict[str, dict[str, torch.Tensor]]:
    from vllm import SamplingParams

    from torchspec.training.data_fetcher import MooncakeDataset
    from torchspec.transfer.mooncake import EagleMooncakeStore, MooncakeConfig

    engine = build_engine(args.model, args.tp, args.pp, aux_ids, args.max_model_len)

    store = EagleMooncakeStore(MooncakeConfig.from_env())
    store.setup(device=torch.device("cuda"))
    dataset = MooncakeDataset(None, store, torch.device("cuda"))

    names = list(PROMPTS)
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    outputs = []
    for start in range(0, len(names), args.requests_per_call):
        batch = names[start : start + args.requests_per_call]
        outputs.extend(
            engine.generate(
                [{"prompt_token_ids": PROMPTS[name]} for name in batch],
                sampling_params,
                use_tqdm=False,
            )
        )

    results: dict[str, dict[str, torch.Tensor]] = {}
    for name, output in zip(names, outputs):
        sampled = sum(len(completion.token_ids) for completion in output.outputs)
        if sampled != 0:
            raise AssertionError(f"{name}: extract-only prefill must emit no tokens, got {sampled}")

        kv_params = getattr(output, "kv_transfer_params", None)
        if kv_params is None:
            raise AssertionError(f"{name}: connector published no kv_transfer_params")

        has_manifest = kv_params.get("pp_layer_manifest") is not None
        if has_manifest != (args.pp > 1):
            raise AssertionError(
                f"{name}: pp_layer_manifest present={has_manifest} at pp_size={args.pp}"
            )

        tensors = read_sample(dataset, kv_params)
        results[name] = tensors
        print(
            f"  {name}: seq_len={len(output.prompt_token_ids)} "
            f"hidden_states={tuple(tensors['hidden_states'].shape)} "
            f"last_hidden_states={tuple(tensors['last_hidden_states'].shape)}"
        )

    store.close()
    return results


def compare(
    baseline: dict[str, dict[str, torch.Tensor]],
    candidate: dict[str, dict[str, torch.Tensor]],
    tolerant: bool,
    rel_l2_tol: float,
    cos_tol: float,
) -> bool:
    ok = True
    for name in sorted(baseline):
        if name not in candidate:
            print(f"  {name}: MISSING from the pipeline-parallel arm")
            ok = False
            continue

        base_tensors, cand_tensors = baseline[name], candidate[name]
        if not torch.equal(base_tensors["input_ids"], cand_tensors["input_ids"]):
            print(f"  {name}: input_ids differ")
            ok = False

        for key in ("hidden_states", "last_hidden_states"):
            base, cand = base_tensors[key].float(), cand_tensors[key].float()
            if base.shape != cand.shape:
                print(f"  {name}/{key}: shape {tuple(base.shape)} != {tuple(cand.shape)}")
                ok = False
                continue

            bitwise = torch.equal(base_tensors[key], cand_tensors[key])
            # float64 throughout: these tensors carry Qwen-style massive
            # activations (~1e4) next to ~1e-2 values, and an fp32 dot product
            # over millions of such elements loses enough precision to report
            # a cosine above 1.
            base64, cand64 = base.double(), cand.double()
            delta = cand64 - base64
            rel_l2 = (delta.norm() / base64.norm().clamp_min(1e-12)).item()
            cosine = torch.nn.functional.cosine_similarity(
                base64.flatten(), cand64.flatten(), dim=0
            ).item()
            max_abs = delta.abs().max().item()
            passed = bitwise or (tolerant and rel_l2 <= rel_l2_tol and cosine >= cos_tol)
            ok = ok and passed
            print(
                f"  {name}/{key}: {'PASS' if passed else 'FAIL'} bitwise={bitwise} "
                f"rel_l2={rel_l2:.3e} cos={cosine:.8f} max_abs={max_abs:.3e}"
            )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=2, help="pipeline size of the candidate arm")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dump-dir", default="./pp_equivalence_dumps")
    parser.add_argument(
        "--phase",
        choices=("baseline", "candidate", "compare"),
        required=True,
        help="each arm runs in its own process so vLLM fully releases its workers",
    )
    parser.add_argument(
        "--requests-per-call",
        type=int,
        default=1,
        help="prompts per generate() call; 1 keeps batch composition identical across arms",
    )
    parser.add_argument(
        "--tolerant",
        action="store_true",
        help="accept close-but-not-bitwise results (use only when the arms cannot "
        "be given identical batch composition)",
    )
    parser.add_argument("--rel-l2-tol", type=float, default=5e-2)
    parser.add_argument("--cos-tol", type=float, default=1.0 - 1e-4)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    aux_ids, hidden_size, num_layers = resolve_aux_layer_ids(args.model)

    if args.phase == "compare":
        baseline = torch.load(dump_dir / "pp1.pt", weights_only=False)
        candidate = torch.load(dump_dir / f"pp{args.pp}.pt", weights_only=False)
        print(f"Comparing pp=1 against pp={args.pp}")
        if not compare(baseline, candidate, args.tolerant, args.rel_l2_tol, args.cos_tol):
            raise SystemExit("PP equivalence FAILED")
        print("PP equivalence PASSED")
        return

    if args.phase == "baseline":
        args.pp = 1

    ownership = describe_ownership(aux_ids, num_layers, args.pp)
    print(f"model={args.model} tp={args.tp} pp={args.pp} aux_ids={aux_ids} hidden={hidden_size}")
    print(f"stage ownership: {json.dumps({str(k): v for k, v in ownership.items()})}")
    if args.pp > 1 and sum(1 for owned in ownership.values() if owned) < 2:
        raise SystemExit(
            "Refusing to run: every aux layer lands on one stage, so this arm would "
            "not exercise cross-stage reassembly. Choose different aux layers."
        )

    results = produce(args, aux_ids)
    out_path = dump_dir / f"pp{args.pp}.pt"
    torch.save(results, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

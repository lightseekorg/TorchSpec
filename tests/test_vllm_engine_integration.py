"""Integration test for vLLM extract_hidden_states + MooncakeHiddenStatesConnector.

Uses the same engine setup as VllmEngine: speculative_config with
extract_hidden_states method, and kv_transfer_config pointing to
MooncakeHiddenStatesConnector.

Dumps all tensors to local .pt files for cross-engine comparison (e.g. vs sglang).

Tests:
  1. Short sequences via prompt_token_ids
  2. Longer sequences via prompt_token_ids
  3. Text prompts (defer tokenization path)

Usage:
  # Start mooncake master first:
  #   mooncake_master --port 50051 &
  #   etcd --listen-client-urls http://0.0.0.0:8090 --advertise-client-urls http://localhost:8090 &
  #
  python tests/test_vllm_engine_integration.py [--model MODEL] [--tp TP] [--dump-dir DIR]
"""

import argparse
import os
import socket
from pathlib import Path

import torch
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Mooncake env setup (must happen before any vLLM import)
# ---------------------------------------------------------------------------
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    LOCAL_IP = s.getsockname()[0]
    s.close()
except Exception:
    LOCAL_IP = "localhost"

os.environ.setdefault("MOONCAKE_MASTER_HOST", LOCAL_IP)
os.environ.setdefault("MOONCAKE_MASTER_PORT", "51135")
os.environ.setdefault("MOONCAKE_METADATA_PORT", "8763")
os.environ.setdefault("MOONCAKE_LOCAL_HOSTNAME", LOCAL_IP)
os.environ.setdefault("MOONCAKE_MASTER_SERVER", f"{LOCAL_IP}:51135")


def get_aux_layer_ids(model_path: str) -> list[int]:
    """Replicate VllmEngine's aux layer resolution: default Eagle3 layers + final layer."""
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg = getattr(cfg, "text_config", cfg)
    num_layers = cfg.num_hidden_layers
    # Default Eagle3 aux layers
    aux_ids = [1, num_layers // 2 - 1, num_layers - 4]
    # VllmEngine appends the final layer for last_hidden_states capture
    final_layer = num_layers - 1
    if final_layer not in aux_ids:
        aux_ids.append(final_layer)
    return aux_ids, cfg.hidden_size, num_layers


def create_engine(
    model_path: str,
    tp_size: int,
    aux_layer_ids: list[int],
    max_num_batched_tokens: int | None = None,
):
    from vllm import LLM

    extra_args = {}
    if max_num_batched_tokens is not None:
        extra_args["max_num_batched_tokens"] = max_num_batched_tokens
        extra_args["enable_chunked_prefill"] = True

    engine = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        disable_custom_all_reduce=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": list(aux_layer_ids),
                }
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
        **extra_args,
    )
    return engine


def fetch_and_dump(
    mooncake_store,
    key: str,
    seq_len: int,
    hidden_dim: int,
    last_hidden_dim: int,
    dump_dir: Path,
    label: str,
) -> dict[str, torch.Tensor]:
    """Retrieve tensors from mooncake, verify shapes, save to disk."""
    shapes = {
        "hidden_states": (seq_len, hidden_dim),
        "input_ids": (seq_len,),
        "last_hidden_states": (seq_len, last_hidden_dim),
    }
    dtypes = {
        "hidden_states": torch.bfloat16,
        "input_ids": torch.long,
        "last_hidden_states": torch.bfloat16,
    }
    data = mooncake_store.get(key, shapes=shapes, dtypes=dtypes, device="cuda")

    tensors = {
        "hidden_states": data.hidden_states.cpu(),
        "input_ids": data.input_ids.cpu(),
        "last_hidden_states": data.last_hidden_states.cpu(),
    }

    assert tensors["hidden_states"].shape == (seq_len, hidden_dim), (
        f"hidden_states shape {tensors['hidden_states'].shape} != expected {(seq_len, hidden_dim)}"
    )
    assert tensors["input_ids"].shape == (seq_len,)
    assert tensors["last_hidden_states"].shape == (seq_len, last_hidden_dim)

    dump_path = dump_dir / f"vllm_{label}.pt"
    torch.save(tensors, dump_path)
    print(f"  Saved: {dump_path}")
    print(
        f"    hidden_states:      {tensors['hidden_states'].shape}, dtype={tensors['hidden_states'].dtype}"
    )
    print(
        f"    input_ids:          {tensors['input_ids'].shape}, first_10={tensors['input_ids'][:10].tolist()}"
    )
    print(
        f"    last_hidden_states: {tensors['last_hidden_states'].shape}, dtype={tensors['last_hidden_states'].dtype}"
    )

    hs = tensors["hidden_states"].float()
    lhs = tensors["last_hidden_states"].float()
    print(f"    hidden_states      norm={hs.norm():.4f}, mean={hs.mean():.6f}, std={hs.std():.6f}")
    print(
        f"    last_hidden_states norm={lhs.norm():.4f}, mean={lhs.mean():.6f}, std={lhs.std():.6f}"
    )

    return tensors


def run_test(
    engine,
    mooncake_store,
    prompts,
    data_ids: list[str],
    expected_seq_lens: list[int] | None,
    hidden_dim: int,
    last_hidden_dim: int,
    dump_dir: Path,
    test_name: str,
):
    from vllm import SamplingParams

    sampling_params = SamplingParams(max_tokens=1, temperature=0)

    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 60}")

    outputs = engine.generate(prompts, sampling_params, use_tqdm=False)

    for i, output in enumerate(outputs):
        kv_params = getattr(output, "kv_transfer_params", None)
        seq_len = len(output.prompt_token_ids)

        if expected_seq_lens is not None:
            assert seq_len == expected_seq_lens[i], (
                f"seq_len mismatch: got {seq_len}, expected {expected_seq_lens[i]}"
            )

        if kv_params is None:
            print(f"  WARNING: Request {data_ids[i]}: no kv_transfer_params!")
            continue

        mooncake_key = kv_params.get("mooncake_key", data_ids[i])
        print(f"\n  Request {data_ids[i]}: seq_len={seq_len}, mooncake_key={mooncake_key}")

        label = f"{test_name}_{data_ids[i]}"
        fetch_and_dump(
            mooncake_store,
            mooncake_key,
            seq_len,
            hidden_dim,
            last_hidden_dim,
            dump_dir,
            label,
        )

    print(f"\n✓ {test_name} passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dump-dir", default="./tensor_dumps")
    parser.add_argument(
        "--aux-layers",
        type=int,
        nargs="*",
        default=None,
        help="Override aux layer IDs (without final layer; it is auto-appended)",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Resolve aux layers
    auto_aux_ids, hidden_size, num_layers = get_aux_layer_ids(args.model)
    if args.aux_layers is not None:
        aux_layer_ids = list(args.aux_layers)
        final_layer = num_layers - 1
        if final_layer not in aux_layer_ids:
            aux_layer_ids.append(final_layer)
    else:
        aux_layer_ids = auto_aux_ids

    num_training_layers = len(aux_layer_ids) - 1  # last is final-layer for last_hidden_states
    hidden_dim = num_training_layers * hidden_size
    last_hidden_dim = hidden_size

    print(f"Model:           {args.model}")
    print(f"TP size:         {args.tp}")
    print(f"Aux layer IDs:   {aux_layer_ids}")
    print(f"  training layers: {aux_layer_ids[:-1]} -> hidden_dim={hidden_dim}")
    print(f"  final layer:     {aux_layer_ids[-1]} -> last_hidden_dim={last_hidden_dim}")
    print(f"Hidden size:     {hidden_size}")
    print(f"Num layers:      {num_layers}")
    print(f"Dump dir:        {dump_dir}")

    # Save test metadata for comparison script
    meta = {
        "engine": "vllm",
        "model": args.model,
        "aux_layer_ids": aux_layer_ids,
        "num_training_layers": num_training_layers,
        "hidden_size": hidden_size,
        "last_hidden_dim": last_hidden_dim,
    }
    torch.save(meta, dump_dir / "vllm_meta.pt")

    engine = create_engine(args.model, args.tp, aux_layer_ids)

    from torchspec.transfer.mooncake import EagleMooncakeStore, MooncakeConfig

    mooncake_config = MooncakeConfig.from_env()
    mooncake_store = EagleMooncakeStore(mooncake_config)
    mooncake_store.setup(device="cuda")

    # ── Test 1: Short sequences (raw token IDs) ──────────────────────────
    input_ids_list = [
        [1, 2345, 6789],
        [100, 200, 300, 400],
        [500, 600],
    ]
    data_ids = ["short_0", "short_1", "short_2"]
    prompts = [{"prompt_token_ids": ids} for ids in input_ids_list]
    seq_lens = [len(ids) for ids in input_ids_list]

    run_test(
        engine,
        mooncake_store,
        prompts,
        data_ids,
        seq_lens,
        hidden_dim,
        last_hidden_dim,
        dump_dir,
        "short_seqs",
    )

    # ── Test 2: Longer sequences (raw token IDs) ─────────────────────────
    long_input_ids = [
        list(range(1, 101)),
        list(range(200, 351)),
        list(range(400, 465)),
    ]
    long_data_ids = ["long_0", "long_1", "long_2"]
    prompts = [{"prompt_token_ids": ids} for ids in long_input_ids]
    long_seq_lens = [len(ids) for ids in long_input_ids]

    run_test(
        engine,
        mooncake_store,
        prompts,
        long_data_ids,
        long_seq_lens,
        hidden_dim,
        last_hidden_dim,
        dump_dir,
        "long_seqs",
    )

    # ── Test 3: Text prompts (defer tokenization) ────────────────────────
    text_prompts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time in a land far away, there lived a brave knight.",
    ]
    prompt_data_ids = ["prompt_0", "prompt_1", "prompt_2"]

    run_test(
        engine,
        mooncake_store,
        text_prompts,
        prompt_data_ids,
        None,
        hidden_dim,
        last_hidden_dim,
        dump_dir,
        "text_prompts",
    )

    # ── Test 4: Chunked prefill ─────────────────────────────────────────
    # Destroy the first engine and create one with a small token budget
    # so sequences are split across multiple scheduler steps.
    del engine

    print(f"\n{'=' * 60}")
    print("Creating chunked-prefill engine (max_num_batched_tokens=128)")
    print(f"{'=' * 60}")

    chunked_engine = create_engine(
        args.model,
        args.tp,
        aux_layer_ids,
        max_num_batched_tokens=128,
    )

    # Sequences longer than 128 tokens will require multiple prefill chunks.
    chunked_input_ids = [
        list(range(1, 201)),  # 200 tokens → 2 chunks
        list(range(300, 700)),  # 400 tokens → 4 chunks
    ]
    chunked_data_ids = ["chunked_0", "chunked_1"]
    prompts = [{"prompt_token_ids": ids} for ids in chunked_input_ids]
    chunked_seq_lens = [len(ids) for ids in chunked_input_ids]

    run_test(
        chunked_engine,
        mooncake_store,
        prompts,
        chunked_data_ids,
        chunked_seq_lens,
        hidden_dim,
        last_hidden_dim,
        dump_dir,
        "chunked_prefill",
    )

    del chunked_engine

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print(f"Tensor dumps saved to: {dump_dir}/")
    print(f"{'=' * 60}")

    pt_files = sorted(dump_dir.glob("vllm_*.pt"))
    for f in pt_files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

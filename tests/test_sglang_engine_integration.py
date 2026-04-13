"""Integration test for sglang extract_hidden_states + MooncakeHiddenStatesConnector.

Uses the same engine setup as SglEngine: enable_aux_hidden_states with
aux_hidden_state_layer_ids, and enable_spec_training_mooncake for Mooncake
hidden states transfer.

Dumps all tensors to local .pt files for cross-engine comparison (e.g. vs vllm).

Tests:
  1. Short sequences via input_ids
  2. Longer sequences via input_ids
  3. Text prompts (defer tokenization path)

Usage:
  # Start mooncake master first:
  #   mooncake_master --port 50051 &
  #   etcd --listen-client-urls http://0.0.0.0:8090 --advertise-client-urls http://localhost:8090 &
  #
  python tests/test_sglang_engine_integration.py [--model MODEL] [--tp TP] [--dump-dir DIR]
"""

import argparse
import os
import socket
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so the editable install of torchspec
# isn't shadowed by /root/torchspec (a second repo clone in the home dir).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Mooncake env setup (must happen before any sglang import)
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
_MC_PORT = os.environ["MOONCAKE_MASTER_PORT"]
os.environ.setdefault("MOONCAKE_MASTER_SERVER", f"{LOCAL_IP}:{_MC_PORT}")


def get_aux_layer_ids(model_path: str) -> tuple[list[int], int, int]:
    """Replicate SglEngine's aux layer resolution: default Eagle3 layers (no final layer).

    Unlike vllm, sglang captures last_hidden_states automatically from the
    model's final layer output, so aux_hidden_state_layer_ids should NOT
    include the final layer.
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg = getattr(cfg, "text_config", cfg)
    num_layers = cfg.num_hidden_layers
    aux_ids = [1, num_layers // 2 - 1, num_layers - 4]
    return aux_ids, cfg.hidden_size, num_layers


def create_engine(model_path: str, tp_size: int, aux_layer_ids: list[int]):
    import sglang as sgl

    engine = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        mem_fraction_static=0.7,
        trust_remote_code=True,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        enable_return_hidden_states=True,
        enable_aux_hidden_states=True,
        aux_hidden_state_layer_ids=list(aux_layer_ids),
        enable_spec_training_mooncake=True,
        chunked_prefill_size=-1,
        log_level="warning",
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

    dump_path = dump_dir / f"sglang_{label}.pt"
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


def run_test_input_ids(
    engine,
    mooncake_store,
    input_ids_list: list[list[int]],
    data_ids: list[str],
    hidden_dim: int,
    last_hidden_dim: int,
    dump_dir: Path,
    test_name: str,
):
    """Run test with pre-tokenized input_ids."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 60}")

    results = engine.generate(
        input_ids=input_ids_list,
        spec_training_data_id=data_ids,
        sampling_params={"max_new_tokens": 1},
        return_hidden_states=True,
    )

    for i, result in enumerate(results):
        meta = result["meta_info"]
        store_keys = meta.get("spec_training_mooncake_store_keys", [])
        seq_len = len(input_ids_list[i])

        assert meta.get("hidden_states") is None, "hidden_states should be None when using mooncake"
        assert len(store_keys) > 0, f"Request {data_ids[i]}: no mooncake store keys returned"

        key = store_keys[0]
        print(f"\n  Request {data_ids[i]}: seq_len={seq_len}, mooncake_key={key}")

        label = f"{test_name}_{data_ids[i]}"
        fetch_and_dump(
            mooncake_store,
            key,
            seq_len,
            hidden_dim,
            last_hidden_dim,
            dump_dir,
            label,
        )

    print(f"\n✓ {test_name} passed")


def run_test_text_prompts(
    engine,
    mooncake_store,
    text_prompts: list[str],
    data_ids: list[str],
    hidden_dim: int,
    last_hidden_dim: int,
    dump_dir: Path,
    test_name: str,
):
    """Run test with text prompts (defer tokenization)."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 60}")

    results = engine.generate(
        prompt=text_prompts,
        spec_training_data_id=data_ids,
        sampling_params={"max_new_tokens": 1},
        return_hidden_states=True,
    )

    for i, result in enumerate(results):
        meta = result["meta_info"]
        store_keys = meta.get("spec_training_mooncake_store_keys", [])
        seq_len = meta.get("prompt_tokens")

        assert meta.get("hidden_states") is None, "hidden_states should be None when using mooncake"
        assert len(store_keys) > 0, f"Request {data_ids[i]}: no mooncake store keys returned"
        assert seq_len is not None, f"Request {data_ids[i]}: prompt_tokens missing from meta_info"

        key = store_keys[0]
        print(f"\n  Request {data_ids[i]}: seq_len={seq_len}, mooncake_key={key}")

        label = f"{test_name}_{data_ids[i]}"
        fetch_and_dump(
            mooncake_store,
            key,
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
        help="Override aux layer IDs (training layers only; final layer is automatic)",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    auto_aux_ids, hidden_size, num_layers = get_aux_layer_ids(args.model)
    if args.aux_layers is not None:
        aux_layer_ids = list(args.aux_layers)
    else:
        aux_layer_ids = auto_aux_ids

    num_training_layers = len(aux_layer_ids)
    hidden_dim = num_training_layers * hidden_size
    last_hidden_dim = hidden_size

    print(f"Model:           {args.model}")
    print(f"TP size:         {args.tp}")
    print(f"Aux layer IDs:   {aux_layer_ids}  (sglang captures last_hidden_states automatically)")
    print(f"  training layers: {aux_layer_ids} -> hidden_dim={hidden_dim}")
    print(f"  last_hidden_states from final model layer -> last_hidden_dim={last_hidden_dim}")
    print(f"Hidden size:     {hidden_size}")
    print(f"Num layers:      {num_layers}")
    print(f"Dump dir:        {dump_dir}")

    meta = {
        "engine": "sglang",
        "model": args.model,
        "aux_layer_ids": aux_layer_ids,
        "num_training_layers": num_training_layers,
        "hidden_size": hidden_size,
        "hidden_dim": hidden_dim,
        "last_hidden_dim": last_hidden_dim,
    }
    torch.save(meta, dump_dir / "sglang_meta.pt")

    # Import mooncake before creating sglang engine — sglang's subprocess
    # forking can interfere with the import chain through torchspec.config.__init__
    from torchspec.config.mooncake_config import MooncakeConfig
    from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

    engine = create_engine(args.model, args.tp, aux_layer_ids)

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

    run_test_input_ids(
        engine,
        mooncake_store,
        input_ids_list,
        data_ids,
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

    run_test_input_ids(
        engine,
        mooncake_store,
        long_input_ids,
        long_data_ids,
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

    run_test_text_prompts(
        engine,
        mooncake_store,
        text_prompts,
        prompt_data_ids,
        hidden_dim,
        last_hidden_dim,
        dump_dir,
        "text_prompts",
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print(f"Tensor dumps saved to: {dump_dir}/")
    print(f"{'=' * 60}")

    pt_files = sorted(dump_dir.glob("sglang_*.pt"))
    for f in pt_files:
        print(f"  {f.name}")

    engine.shutdown()


if __name__ == "__main__":
    main()

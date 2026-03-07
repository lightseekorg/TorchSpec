"""Standalone integration script that tests vLLM Worker Extension hidden states collection behavior.

Tests:
  1. Short sequences via input_ids (basic capture)
  2. Longer sequences via input_ids
  3. formatted_prompts path (defer tokenization mode)
"""

import os
import socket

import torch
from transformers import AutoTokenizer

from torchspec.transfer.mooncake import EagleMooncakeStore, MooncakeConfig

# Detect local IP for Mooncake connections
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    LOCAL_IP = s.getsockname()[0]
    s.close()
except Exception:
    LOCAL_IP = "localhost"

# Mooncake env vars read by MooncakeConfig.from_env() on both sides
os.environ["MOONCAKE_MASTER_HOST"] = LOCAL_IP
os.environ["MOONCAKE_MASTER_PORT"] = "50051"
os.environ["MOONCAKE_METADATA_PORT"] = "8090"
os.environ["MOONCAKE_LOCAL_HOSTNAME"] = LOCAL_IP
os.environ["MOONCAKE_MASTER_SERVER"] = f"{LOCAL_IP}:50051"


def collect_metadata(engine, internal_to_external=None):
    """Call _store_and_get_metadata and merge results from all TP ranks."""
    args = (internal_to_external,) if internal_to_external else ()
    metadata_list = engine.collective_rpc("_store_and_get_metadata", args=args)
    merged = {}
    if isinstance(metadata_list, list):
        for m in metadata_list:
            if isinstance(m, dict):
                merged.update(m)
    elif isinstance(metadata_list, dict):
        merged = metadata_list
    return merged


def verify_from_mooncake(mooncake_store, keys, seq_lens, hidden_dim, last_hidden_dim):
    """Fetch tensors from Mooncake and verify shapes."""
    for i, key in enumerate(keys):
        seq_len = seq_lens[i]
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
        print(f"\n  Key: {key}")
        print(
            f"    hidden_states: shape={data.hidden_states.shape}, dtype={data.hidden_states.dtype}"
        )
        print(f"    input_ids: {data.input_ids.tolist()[:10]}{'...' if seq_len > 10 else ''}")
        print(f"    last_hidden_states: shape={data.last_hidden_states.shape}")

        assert data.hidden_states.shape == (seq_len, hidden_dim), (
            f"hidden_states shape {data.hidden_states.shape} != expected {(seq_len, hidden_dim)}"
        )
        assert data.input_ids.shape == (seq_len,)
        assert data.last_hidden_states.shape == (seq_len, last_hidden_dim)


if __name__ == "__main__":
    model_path = "Qwen/Qwen3-8B"
    aux_layer_ids = [2, 4, 6]
    tp_size = 4
    hidden_size = 4096
    num_aux_layers = len(aux_layer_ids)
    hidden_dim = num_aux_layers * hidden_size
    last_hidden_dim = hidden_size

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        disable_custom_all_reduce=True,
        disable_log_stats=True,
        worker_extension_cls="torchspec.inference.engine.vllm_worker_extension.VllmWorkerExtension",
        max_model_len=2048,
        enable_chunked_prefill=False,
    )

    engine.collective_rpc("_setup_hidden_states_capture", args=(aux_layer_ids,))

    mooncake_config = MooncakeConfig.from_env()
    mooncake_store = EagleMooncakeStore(mooncake_config)
    mooncake_store.setup(device="cuda")

    sampling_params = SamplingParams(max_tokens=1, temperature=0)

    # =========================================================================
    # Test 1: Short sequences
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Short sequences")
    print("=" * 60)

    input_ids_list = [
        [1, 2345, 6789],
        [100, 200, 300, 400],
        [500, 600],
    ]
    data_ids = ["short_0", "short_1", "short_2"]

    prompts = [{"prompt_token_ids": ids} for ids in input_ids_list]
    request_metadata = {data_ids[i]: len(ids) for i, ids in enumerate(input_ids_list)}
    input_ids_map = {data_ids[i]: ids for i, ids in enumerate(input_ids_list)}

    engine.collective_rpc("_reset_capture")
    engine.collective_rpc("_set_request_metadata", args=(request_metadata, {}, input_ids_map))

    outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
    for i, output in enumerate(outputs):
        print(f"  Request {i}: {len(output.prompt_token_ids)} prompt tokens")

    metadata = collect_metadata(engine)
    all_keys = [metadata[did]["mooncake_key"] for did in data_ids]
    seq_lens = [request_metadata[did] for did in data_ids]
    assert len(metadata) == len(data_ids), f"Expected {len(data_ids)} results, got {len(metadata)}"

    verify_from_mooncake(mooncake_store, all_keys, seq_lens, hidden_dim, last_hidden_dim)
    print("\n✓ Test 1 passed")

    # =========================================================================
    # Test 2: Longer sequences
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Longer sequences")
    print("=" * 60)

    long_input_ids_list = [
        list(range(1, 101)),
        list(range(200, 351)),
        list(range(400, 465)),
    ]
    long_data_ids = ["long_0", "long_1", "long_2"]

    prompts = [{"prompt_token_ids": ids} for ids in long_input_ids_list]
    request_metadata = {long_data_ids[i]: len(ids) for i, ids in enumerate(long_input_ids_list)}
    input_ids_map = {long_data_ids[i]: ids for i, ids in enumerate(long_input_ids_list)}

    engine.collective_rpc("_reset_capture")
    engine.collective_rpc("_set_request_metadata", args=(request_metadata, {}, input_ids_map))

    outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
    for i, output in enumerate(outputs):
        print(f"  Request {i}: {len(output.prompt_token_ids)} prompt tokens")

    metadata = collect_metadata(engine)
    all_keys = [metadata[did]["mooncake_key"] for did in long_data_ids]
    seq_lens = [request_metadata[did] for did in long_data_ids]
    assert len(metadata) == len(long_data_ids), (
        f"Expected {len(long_data_ids)} results, got {len(metadata)}"
    )

    verify_from_mooncake(mooncake_store, all_keys, seq_lens, hidden_dim, last_hidden_dim)
    print("\n✓ Test 2 passed")

    # =========================================================================
    # Test 3: formatted_prompts path (defer tokenization)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: formatted_prompts path (defer tokenization)")
    print("=" * 60)

    text_prompts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time",
    ]
    prompt_data_ids = ["prompt_0", "prompt_1", "prompt_2"]

    engine.collective_rpc("_reset_capture")

    outputs = engine.generate(text_prompts, sampling_params, use_tqdm=False)

    # Build authoritative metadata from outputs and set on workers,
    # mirroring VllmEngine.generate()'s unconditional post-generation path.
    request_metadata = {}
    input_ids_map = {}
    internal_to_external = {}
    for i, output in enumerate(outputs):
        did = prompt_data_ids[i]
        request_metadata[did] = len(output.prompt_token_ids)
        input_ids_map[did] = list(output.prompt_token_ids)
        internal_to_external[output.request_id] = did
        print(f'  Request {i}: "{text_prompts[i]}" -> {len(output.prompt_token_ids)} tokens')
    engine.collective_rpc("_set_request_metadata", args=(request_metadata, {}, input_ids_map))

    metadata = collect_metadata(engine, internal_to_external=internal_to_external)
    all_keys = [metadata[did]["mooncake_key"] for did in prompt_data_ids]
    seq_lens = [request_metadata[did] for did in prompt_data_ids]
    assert len(metadata) == len(prompt_data_ids), (
        f"Expected {len(prompt_data_ids)} results, got {len(metadata)}"
    )

    verify_from_mooncake(mooncake_store, all_keys, seq_lens, hidden_dim, last_hidden_dim)
    print("\n✓ Test 3 passed")

    # =========================================================================
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    del engine

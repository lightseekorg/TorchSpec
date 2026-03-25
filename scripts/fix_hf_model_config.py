#!/usr/bin/env python3
"""Fix the HuggingFace model config to be compatible with z-lab / SGLang DFlash inference.

Our uploaded model uses model_type="dflash" (custom, not recognized by Transformers).
z-lab's format uses model_type="qwen3" with a nested dflash_config, plus auto_map
pointing to custom modeling code (dflash.py, utils.py).

This script:
1. Uploads a corrected config.json
2. Uploads dflash.py, modeling_dflash.py, utils.py (from z-lab) for trust_remote_code
"""

import json
import tempfile
import os
from pathlib import Path

REPO_ID = "Xingh3/dflash-qwen3-8b-3epoch"

NEW_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoModel": "dflash.DFlashDraftModel"
    },
    "block_size": 16,
    "bos_token_id": 151643,
    "dflash_config": {
        "mask_token_id": 151669,
        "target_layer_ids": [1, 9, 17, 25, 33]
    },
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "layer_types": [
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention"
    ],
    "max_position_embeddings": 40960,
    "max_window_layers": 5,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 36,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "4.57.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936
}


def main():
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(NEW_CONFIG, f, indent=2)
        
        print(f"Uploading corrected config.json to {REPO_ID}...")
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=REPO_ID,
            commit_message="Fix config.json to z-lab format for SGLang DFlash inference",
        )
        print("  config.json uploaded.")
    
    zlab_repo = "z-lab/Qwen3-8B-DFlash-b16"
    for filename in ["dflash.py", "modeling_dflash.py", "utils.py"]:
        print(f"Copying {filename} from {zlab_repo}...")
        local_path = api.hf_hub_download(repo_id=zlab_repo, filename=filename)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=REPO_ID,
            commit_message=f"Add {filename} from z-lab for trust_remote_code SGLang inference",
        )
        print(f"  {filename} uploaded.")
    
    print(f"\nDone! {REPO_ID} now has z-lab-compatible config and modeling code.")
    print("SGLang should now be able to load the model with --trust-remote-code.")


if __name__ == "__main__":
    main()

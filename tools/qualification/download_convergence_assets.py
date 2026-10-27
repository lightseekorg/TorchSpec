"""Download immutable public assets for train_tokenizer_convergence.py on a CPU host."""

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from tests.tokenizer_contract.support import load_manifest


def copy_asset(repo, revision, filename, destination, digest, repo_type="model"):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        source = hf_hub_download(repo, filename, revision=revision, repo_type=repo_type)
        # Copy contents only; HF cache files may be readonly and remote shared
        # filesystems may reject chmod/chown when copying their metadata.
        shutil.copyfile(source, destination)
    actual = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            actual.update(chunk)
    if actual.hexdigest() != digest:
        raise ValueError(f"Asset digest mismatch: {destination}")
    print(destination, actual.hexdigest(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = load_manifest()["profiles"]["qwen3"]
    files = {
        **profile["assets"],
        "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    }
    for name, digest in files.items():
        copy_asset(
            profile["model"], profile["revision"], name, args.output / "model" / name, digest
        )
    for split, filename, digest in [
        (
            "train",
            "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
            "afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727",
        ),
        (
            "eval",
            "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
            "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70",
        ),
    ]:
        copy_asset(
            "HuggingFaceH4/ultrachat_200k",
            "8049631c405ae6576f93f445c6b8166f76f5505a",
            filename,
            args.output / "data" / f"{split}.parquet",
            digest,
            "dataset",
        )


if __name__ == "__main__":
    main()

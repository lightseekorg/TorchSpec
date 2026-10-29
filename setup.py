"""Build TorchSpec and its in-process SMG extension as one distribution."""

import os

from setuptools import setup
from setuptools_rust import Binding, RustExtension

features = [name for name in os.environ.get("TORCHSPEC_SMG_FEATURES", "").split(",") if name]
if set(features) - {"tokenizers-v1", "multimodal"}:
    raise ValueError("TORCHSPEC_SMG_FEATURES accepts comma-separated tokenizers-v1,multimodal")

setup(
    rust_extensions=[
        RustExtension(
            "torchspec_smg._native",
            path="bindings/smg-tokenizer/Cargo.toml",
            binding=Binding.PyO3,
            cargo_manifest_args=["--locked"],
            features=features,
            debug=False,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)

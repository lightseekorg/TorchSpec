"""Build TorchSpec and its in-process SMG extension as one distribution."""

import os

from setuptools import setup
from setuptools_rust import Binding, RustExtension

features = os.environ.get("TORCHSPEC_SMG_FEATURES", "")
if features not in ("", "tokenizers-v1"):
    raise ValueError("TORCHSPEC_SMG_FEATURES must be empty or 'tokenizers-v1'")

setup(
    rust_extensions=[
        RustExtension(
            "torchspec_smg._native",
            path="bindings/smg-tokenizer/Cargo.toml",
            binding=Binding.PyO3,
            cargo_manifest_args=["--locked"],
            features=[features] if features else [],
            debug=False,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)

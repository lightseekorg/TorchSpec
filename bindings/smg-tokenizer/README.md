# TorchSpec SMG tokenizer bridge

This Rust crate is compiled by the root setuptools-rust build into
`torchspec_smg._native`. Its Python wrapper is `python/torchspec_smg/__init__.py`. Both ship in the main
TorchSpec wheel; `pip install -e .` builds them automatically. This directory is
not an independently installable Python package.

The Dockerfiles build the same wheel in a compiler stage and install it into each
training image. SMG comes from the immutable revision in `Cargo.toml`, with
transitive dependencies pinned in `Cargo.lock`. Rust executes directly inside
Python dataset workers through PyO3, releasing the GIL during heavy work.

See [installation and qualification](../../docs/smg-tokenization.md) for build
commands, supported inputs, supervision policy, and the convergence gate.

`Tokenizer.render` handles complete conversations and returns rendered text,
exact token IDs, optional UTF-8 byte offsets, and a segmented-encoding indicator.
It preserves native deferred encoding jobs. Flat offsets require an extra
reference encode with identical IDs; segmented encodings have no flat offsets.
TorchSpec defines the training loss mask, not this bridge. Media processing is
outside its scope.

Run binding tests after installing the wheel:

```bash
python -m pytest -q bindings/smg-tokenizer/tests
```

The internal Rust binding version is exposed for dataset cache invalidation. Version
0.1.2 exposes special-token lookup for TorchSpec's K3 token-ID mask policy.
Rendering and encoding continue to use the pinned upstream Rust implementation.

0.1.3 enables serde_json float round-tripping so Python tool-argument floats
retain their exact binary value across the native JSON boundary.

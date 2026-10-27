# TorchSpec SMG tokenizer bridge

This native extension runs SMG's Rust tokenizer inside Python dataset workers
using PyO3. The wheel is an installation artifact, not a separate service or
independently maintained project. TorchSpec owns the bridge; its Dockerfiles
build the wheel in a separate stage and install it in each training image.
The Rust crate comes from the immutable upstream
revision in `Cargo.toml`, with transitive dependencies pinned in `Cargo.lock`.
It needs no SMG fork, gateway or separate source checkout.

See [installation and qualification](../../docs/smg-tokenization.md) for build
commands, supported inputs, supervision policy and the convergence gate.

`Tokenizer.render` handles complete conversations and returns rendered text,
exact token IDs, optional UTF-8 byte offsets and a segmented-encoding indicator.
It preserves native deferred encoding jobs. Flat offsets require an extra
reference encode with identical IDs; segmented encodings have no flat offsets.
TorchSpec defines the training loss mask, not this bridge. Media processing is
outside its scope.

Run binding tests after installing the wheel:

```bash
python -m pytest -q bindings/smg-tokenizer/tests
```

The binding is versioned independently for dataset cache invalidation. Version
0.1.2 exposes special-token lookup for TorchSpec's K3 token-ID mask policy.
Rendering and encoding continue to use the pinned upstream Rust implementation.

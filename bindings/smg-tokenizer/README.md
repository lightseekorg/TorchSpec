# TorchSpec SMG tokenizer bridge

This optional package runs SMG's Rust tokenizer inside Python processes using
PyO3. TorchSpec owns the bridge; the Rust crate comes from the immutable upstream
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

The binding was moved from the prototype companion SMG change without changing
its Rust implementation. Version 0.1.1 distinguishes this owned, pinned build
from that prototype for dataset cache invalidation.

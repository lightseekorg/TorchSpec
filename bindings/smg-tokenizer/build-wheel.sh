#!/usr/bin/env bash
# Used only in Docker's torchspec-builder stage; the runtime installs one TorchSpec wheel.
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates curl git pkg-config libssl-dev
rm -rf /var/lib/apt/lists/*

export CARGO_HOME=/opt/smg-build/cargo
export RUSTUP_HOME=/opt/smg-build/rustup
export RUSTUP_TOOLCHAIN=1.98.0
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
export PATH="$CARGO_HOME/bin:$PATH"
curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/smg-rustup-init.sh
bash /tmp/smg-rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain "$RUSTUP_TOOLCHAIN"

cd /tmp/torchspec
python3 -m pip wheel --no-cache-dir --no-deps . --wheel-dir /opt/torchspec-wheels

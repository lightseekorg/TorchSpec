#!/usr/bin/env bash
# Used only in Docker's smg-builder stage; the runtime copies the wheel alone.
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
python3 -m pip install --no-cache-dir maturin==1.15.0

cd /tmp/smg-tokenizer
maturin build --release --locked --out /opt/smg-wheels

#!/bin/bash

# To force apply a patch, run: ./tools/apply_sglang_patch.sh <absolute-path-to-sglang-repo>
# Please note that this will overwrite all the local changes.

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

SGLANG_VERSION="${SGLANG_VERSION:-v0.5.8.post1}"
SGLANG_DIR="$PROJECT_ROOT/docker/sglang/$SGLANG_VERSION"

if [ ! -d "$SGLANG_DIR" ]; then
    echo "Error: sglang version directory not found: $SGLANG_DIR"
    exit 1
fi

SGLANG_COMMIT=$(grep "^ARG SGLANG_COMMIT=" "$SGLANG_DIR/Dockerfile" | cut -d= -f2)

if [ -z "$SGLANG_COMMIT" ]; then
    echo "Error: Could not find SGLANG_COMMIT in $SGLANG_DIR/Dockerfile"
    exit 1
fi

SGLANG_PATH="${1:?Usage: $0 <path-to-sglang-repo>}"

PATCH_FILE="$PROJECT_ROOT/patches/sglang/$SGLANG_VERSION/sglang.patch"
if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: Patch file not found: $PATCH_FILE"
    exit 1
fi

echo "SGLANG_VERSION: $SGLANG_VERSION"
echo "SGLANG_COMMIT: $SGLANG_COMMIT"
echo "SGLANG_PATH:   $SGLANG_PATH"
echo "PATCH_FILE:    $PATCH_FILE"
echo ""

if [ ! -d "$SGLANG_PATH" ]; then
    echo "Error: $SGLANG_PATH directory not found"
    exit 1
fi

cd "$SGLANG_PATH"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: $SGLANG_PATH is not a git repository"
    exit 1
fi

if ! git rev-parse "$SGLANG_COMMIT" > /dev/null 2>&1; then
    echo "Error: Commit $SGLANG_COMMIT not found in $SGLANG_PATH repository"
    exit 1
fi

echo "Resetting to base commit $SGLANG_COMMIT..."
git reset --hard "$SGLANG_COMMIT"
git clean -fd

echo ""
echo "Applying patch..."
git apply "$PATCH_FILE"

echo ""
echo "✓ Patch applied successfully."
echo ""
echo "Files modified:"
git status --short

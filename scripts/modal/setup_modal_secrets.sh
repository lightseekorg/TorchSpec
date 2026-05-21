#!/usr/bin/env bash
# Setup Modal secrets for TorchSpec colocate smoke tests (sandbox env).
#
# Usage:
#   bash scripts/modal/setup_modal_secrets.sh                  # defaults to sandbox env
#   bash scripts/modal/setup_modal_secrets.sh --env <name>     # target a different env
#
# Tokens can be provided via environment variables or interactively:
#   HF_WRITE_TOKEN  — HuggingFace write token (https://huggingface.co/settings/tokens)
#                     Needed to download Qwen3-8B for Phase 4+ smoke tests.
#   WANDB_API_KEY   — Weights & Biases API key (https://wandb.ai/authorize)
#                     Optional — used by Phase 6 / Phase 7 long runs.
#
# This script mirrors scripts/modal/setup_modal_secrets.sh from the
# feature/dflash-training branch but creates the same secret names so that
# both the dflash training script and the colocate smoke script can share
# them inside the sandbox env.

set -euo pipefail

ENV="sandbox"
SKIP_WANDB="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV="$2"; shift 2 ;;
        --skip-wandb) SKIP_WANDB="1"; shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Modal Secret Setup (env: $ENV) ==="
echo

if [[ -z "${HF_WRITE_TOKEN:-}" ]]; then
    read -rp "HF_WRITE_TOKEN (from https://huggingface.co/settings/tokens): " HF_WRITE_TOKEN
fi
if [[ ${#HF_WRITE_TOKEN} -lt 10 ]]; then
    echo "ERROR: HF_WRITE_TOKEN looks too short (${#HF_WRITE_TOKEN} chars)"; exit 1
fi
echo "  Creating xingh3-hf-write ..."
modal secret create xingh3-hf-write "HF_WRITE_TOKEN=${HF_WRITE_TOKEN}" --env "$ENV" --force
echo

if [[ "$SKIP_WANDB" != "1" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        read -rp "WANDB_API_KEY (from https://wandb.ai/authorize, blank to skip): " WANDB_API_KEY || true
    fi
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        if [[ ${#WANDB_API_KEY} -lt 40 ]]; then
            echo "ERROR: WANDB_API_KEY looks too short (${#WANDB_API_KEY} chars, need 40+)"; exit 1
        fi
        echo "  Creating wandb-secret ..."
        modal secret create wandb-secret "WANDB_API_KEY=${WANDB_API_KEY}" --env "$ENV" --force
    else
        echo "  (skipping WandB secret — long-running phase 6/7 metrics will be local-only)"
    fi
fi
echo

echo "=== Done. Secrets created in env '$ENV' ==="
modal secret list --env "$ENV" 2>&1 | grep -E 'xingh3-hf-write|wandb-secret' || true

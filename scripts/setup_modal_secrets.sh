#!/usr/bin/env bash
# Setup Modal secrets for DFlash training.
# Usage:
#   bash scripts/setup_modal_secrets.sh                  # defaults to sandbox env
#   bash scripts/setup_modal_secrets.sh --env main       # target a different env
#
# Tokens can be provided via environment variables or interactively:
#   HF_WRITE_TOKEN  — HuggingFace write token (https://huggingface.co/settings/tokens)
#   WANDB_API_KEY   — Weights & Biases API key (https://wandb.ai/authorize)

set -euo pipefail

ENV="sandbox"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Modal Secret Setup (env: $ENV) ==="
echo

# --- HuggingFace write token ---
if [[ -z "${HF_WRITE_TOKEN:-}" ]]; then
    read -rp "HF_WRITE_TOKEN (from https://huggingface.co/settings/tokens): " HF_WRITE_TOKEN
fi
if [[ ${#HF_WRITE_TOKEN} -lt 10 ]]; then
    echo "ERROR: HF_WRITE_TOKEN looks too short (${#HF_WRITE_TOKEN} chars)"; exit 1
fi
echo "  Creating xingh3-hf-write ..."
modal secret create xingh3-hf-write "HF_WRITE_TOKEN=${HF_WRITE_TOKEN}" --env "$ENV" --force
echo

# --- WandB API key ---
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    read -rp "WANDB_API_KEY (from https://wandb.ai/authorize): " WANDB_API_KEY
fi
if [[ ${#WANDB_API_KEY} -lt 40 ]]; then
    echo "ERROR: WANDB_API_KEY looks too short (${#WANDB_API_KEY} chars, need 40+)"; exit 1
fi
echo "  Creating wandb-secret ..."
modal secret create wandb-secret "WANDB_API_KEY=${WANDB_API_KEY}" --env "$ENV" --force
echo

echo "=== Done. Secrets created in env '$ENV' ==="
modal secret list --env "$ENV" 2>&1 | grep -E 'xingh3-hf-write|wandb-secret'

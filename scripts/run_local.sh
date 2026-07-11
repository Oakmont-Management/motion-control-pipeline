#!/usr/bin/env bash
# scripts/run_local.sh — launch the EXISTING local ComfyUI (Documents/ComfyUI) with
# the portable model paths + local flags, then drive it by hand in the browser.
set -e
: "${COMFY_ROOT:=/c/Users/$USERNAME/Documents/ComfyUI}"
export MODELS_ROOT="$COMFY_ROOT/models"
source "$(dirname "$0")/../configs/env.local.sh"
cp "$(dirname "$0")/../configs/extra_model_paths.local.yaml" "$COMFY_ROOT/extra_model_paths.yaml"
cd "$COMFY_ROOT"
# On Windows use the venv python: .venv/Scripts/python.exe main.py $COMFY_LAUNCH_FLAGS --port 8000
# shellcheck disable=SC2086
python main.py $COMFY_LAUNCH_FLAGS --listen 127.0.0.1 --port 8000

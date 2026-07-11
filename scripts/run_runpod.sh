#!/usr/bin/env bash
# scripts/run_runpod.sh — launch ComfyUI on a RunPod pod (network volume), same graph.
set -e
: "${COMFY_ROOT:=/runpod-volume/ComfyUI}"
export MODELS_ROOT="$COMFY_ROOT/models"
source "$(dirname "$0")/../configs/env.runpod.sh"
cp "$(dirname "$0")/../configs/extra_model_paths.runpod.yaml" "$COMFY_ROOT/extra_model_paths.yaml"
cd "$COMFY_ROOT"
# --listen 0.0.0.0 exposes the UI through RunPod's proxy port.
# shellcheck disable=SC2086
python main.py $COMFY_LAUNCH_FLAGS --listen 0.0.0.0 --port 8188

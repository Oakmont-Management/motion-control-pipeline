#!/usr/bin/env bash
# scripts/run_runpod.sh — launch ComfyUI on a RunPod pod, same graph.
# Model paths follow COMFY_ROOT so it works whether the volume mounts at
# /workspace (Pods) or /runpod-volume (serverless) — no empty model dropdowns.
set -e

# Resolve COMFY_ROOT: honor an explicit export, else pick whichever mount exists.
if [ -z "${COMFY_ROOT:-}" ]; then
  for c in /workspace/ComfyUI /runpod-volume/ComfyUI; do
    if [ -d "$c" ]; then COMFY_ROOT="$c"; break; fi
  done
  : "${COMFY_ROOT:=/workspace/ComfyUI}"
fi
export COMFY_ROOT
export MODELS_ROOT="$COMFY_ROOT/models"
echo "COMFY_ROOT=$COMFY_ROOT"

source "$(dirname "$0")/../configs/env.runpod.sh"

# Write extra_model_paths with base_path pinned to the ACTUAL root (not the
# committed default), so ComfyUI always finds the models this repo downloaded.
DEST="$COMFY_ROOT/extra_model_paths.yaml"
cp "$(dirname "$0")/../configs/extra_model_paths.runpod.yaml" "$DEST"
sed -i "s#^\([[:space:]]*base_path:\).*#\1 $COMFY_ROOT/#" "$DEST"
echo "wrote $DEST (base_path: $COMFY_ROOT/)"

cd "$COMFY_ROOT"
# --listen 0.0.0.0 exposes the UI through RunPod's proxy port.
# shellcheck disable=SC2086
python main.py $COMFY_LAUNCH_FLAGS --listen 0.0.0.0 --port 8188

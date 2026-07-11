#!/usr/bin/env bash
# scripts/install_nodes.sh — clone the custom nodes the autoMask workflow needs,
# then install the vendored auto-mask node. Identical set local + RunPod.
set -e
: "${COMFY_ROOT:?set COMFY_ROOT to your ComfyUI dir (e.g. export COMFY_ROOT=/path/to/ComfyUI)}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$COMFY_ROOT/custom_nodes"
cd "$COMFY_ROOT/custom_nodes"

clone() { [ -d "$(basename "$1" .git)" ] || git clone "$1"; }

clone https://github.com/ltdrdata/ComfyUI-Manager
clone https://github.com/city96/ComfyUI-GGUF                    # GGUF loader
clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite   # VHS load/save video
clone https://github.com/kijai/ComfyUI-KJNodes                  # utils (PointsEditor etc.)
clone https://github.com/Fannovel16/comfyui_controlnet_aux      # DWPose
clone https://github.com/kijai/ComfyUI-WanVideoWrapper          # Wan Animate nodes (THIS graph uses these)
clone https://github.com/kijai/ComfyUI-segment-anything-2       # SAM2 subject mask
clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler     # SeedVR2 video upscale (finishing pass)

# --- vendored auto-mask node (correction #2: the doc omits this) ---
# DWPoseToSam2Points seeds SAM2 from DWPose keypoints so replacement-mode masks
# the subject automatically instead of a hand-placed PointsEditor click.
mkdir -p "$COMFY_ROOT/custom_nodes/dwpose_sam2_seed"
cp "$HERE/custom_nodes/dwpose_sam2_seed/__init__.py" \
   "$COMFY_ROOT/custom_nodes/dwpose_sam2_seed/__init__.py"

cd "$COMFY_ROOT"
for d in custom_nodes/*/requirements.txt; do pip install -r "$d" || true; done
echo "Nodes installed (incl. vendored dwpose_sam2_seed)."

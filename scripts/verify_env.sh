#!/usr/bin/env bash
# scripts/verify_env.sh — assert GPU/torch/nodes/models before spending GPU time.
# Run with bash (uses compgen): bash scripts/verify_env.sh
set -euo pipefail
: "${COMFY_ROOT:?set COMFY_ROOT}"; : "${MODELS_ROOT:=$COMFY_ROOT/models}"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "== Torch =="
python -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0),torch.version.cuda)"

echo "== Nodes =="
for n in ComfyUI-GGUF ComfyUI-WanVideoWrapper comfyui_controlnet_aux ComfyUI-segment-anything-2 ComfyUI-KJNodes ComfyUI-VideoHelperSuite ComfyUI-SeedVR2_VideoUpscaler dwpose_sam2_seed; do
  test -d "$COMFY_ROOT/custom_nodes/$n" && echo "ok $n" || { echo "MISSING $n"; exit 1; }
done

echo "== Models (Kijai set) =="
declare -A NEED=(
  ["text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors"]=1
  ["vae/Wan2_1_VAE_bf16.safetensors"]=1
  ["clip_vision/clip_vision_h.safetensors"]=1
  ["loras/WanAnimate_relight_lora_fp16.safetensors"]=1
  ["loras/lightx2v_I2V_14B_480p_distill_rank64_bf16.safetensors"]=1
)
for f in "${!NEED[@]}"; do
  test -f "$MODELS_ROOT/$f" && echo "ok $f" || { echo "MISSING $f"; exit 1; }
done
# At least one Animate GGUF present (Q4 local or Q8 runpod)
if compgen -G "$MODELS_ROOT/unet/Wan2.2-Animate-14B-*.gguf" > /dev/null; then
  echo "ok unet/Wan2.2-Animate-14B-*.gguf"
else echo "MISSING unet/Wan2.2-Animate-14B-*.gguf"; exit 1; fi

echo "== Models (SeedVR2 finishing pass) =="
for f in "SEEDVR2/seedvr2_ema_3b_fp16.safetensors" "SEEDVR2/ema_vae_fp16.safetensors"; do
  test -f "$MODELS_ROOT/$f" && echo "ok $f" || { echo "MISSING $f"; exit 1; }
done

echo "ALL CHECKS PASSED"

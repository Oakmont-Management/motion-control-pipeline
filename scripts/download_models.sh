#!/usr/bin/env bash
# scripts/download_models.sh — hydrate the Kijai-wrapper model set (correction #3).
# Usage: TIER=local ./scripts/download_models.sh   (or TIER=runpod)
# Filenames match the working local install; confirm live on HF if a download 404s
# (QuantStack/Kijai occasionally re-version). Uses the modern `hf` CLI.
set -euo pipefail
: "${MODELS_ROOT:?set MODELS_ROOT to your ComfyUI/models dir}"
: "${TIER:=local}"
export HF_HUB_DISABLE_XET=1          # correction #4: xet hangs on this network
pip install -q "huggingface_hub>=0.34"

dl() { hf download "$1" "$2" --local-dir "$3"; }

ANIMATE_REPO="QuantStack/Wan2.2-Animate-14B-GGUF"
KIJAI_REPO="Kijai/WanVideo_comfy"

if [ "$TIER" = "local" ]; then
  UNET="Wan2.2-Animate-14B-Q4_K_S.gguf"      # 8 GB local
else
  UNET="Wan2.2-Animate-14B-Q8_0.gguf"        # 24 GB+ RunPod
fi

# Diffusion model (Kijai WanVideoModelLoader reads models/unet in this install)
dl "$ANIMATE_REPO" "$UNET" "$MODELS_ROOT/unet"

# Text encoder — Kijai NON-scaled fp8 (the scaled/native one is rejected by the wrapper)
dl "$KIJAI_REPO" "umt5-xxl-enc-fp8_e4m3fn.safetensors" "$MODELS_ROOT/text_encoders"
# Optional RunPod max-quality encoder:
[ "$TIER" = "runpod" ] && dl "$KIJAI_REPO" "umt5-xxl-enc-bf16.safetensors" "$MODELS_ROOT/text_encoders" || true

# VAE (Wan 2.1 VAE for the 14B family) + CLIP vision (reference identity)
dl "$KIJAI_REPO" "Wan2_1_VAE_bf16.safetensors" "$MODELS_ROOT/vae"
dl "$KIJAI_REPO" "clip_vision_h.safetensors"    "$MODELS_ROOT/clip_vision"

# LoRAs used by the graph
dl "$KIJAI_REPO" "WanAnimate_relight_lora_fp16.safetensors"            "$MODELS_ROOT/loras"
dl "$KIJAI_REPO" "lightx2v_I2V_14B_480p_distill_rank64_bf16.safetensors" "$MODELS_ROOT/loras"

# SeedVR2 finishing pass (SeedVR2_Upscale_Finish.json). 3B DiT + its VAE.
# The node also auto-downloads these to models/SEEDVR2 on first use; we pre-stage
# so verify_env can assert them and the first pod render doesn't stall on a fetch.
SEEDVR2_REPO="numz/SeedVR2_comfyUI"
dl "$SEEDVR2_REPO" "seedvr2_ema_3b_fp16.safetensors" "$MODELS_ROOT/SEEDVR2"
dl "$SEEDVR2_REPO" "ema_vae_fp16.safetensors"        "$MODELS_ROOT/SEEDVR2"

echo "Models hydrated for TIER=$TIER at $MODELS_ROOT (unet=$UNET, +SeedVR2 3B)."
echo "SAM2 (sam2_hiera_base_plus) + DWPose (dw-ll_ucoco, yolox_l) auto-download on first queue."

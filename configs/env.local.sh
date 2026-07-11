# configs/env.local.sh — Blackwell 8 GB + WDDM local launch.
# Kijai wrapper does VRAM management via the workflow's block-swap (=40), so we
# don't pass --lowvram here; keep the async-offload guard for sm_120 stability.
export COMFY_LAUNCH_FLAGS="--disable-async-offload"
export TIER=local
# Match the working local install exactly.
export ANIMATE_UNET="Wan2.2-Animate-14B-Q4_K_S.gguf"
export TEXT_ENCODER="umt5-xxl-enc-fp8_e4m3fn.safetensors"
export HF_HUB_DISABLE_XET=1

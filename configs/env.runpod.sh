# configs/env.runpod.sh — clean CUDA, 24 GB+ VRAM.
# No lowvram / async-offload guard needed. Higher quant + (optionally) bf16 encoder.
export COMFY_LAUNCH_FLAGS=""
export TIER=runpod
export ANIMATE_UNET="Wan2.2-Animate-14B-Q8_0.gguf"
# fp8 matches local 1:1; switch to umt5-xxl-enc-bf16.safetensors for max quality (~11 GB).
export TEXT_ENCODER="umt5-xxl-enc-fp8_e4m3fn.safetensors"
export HF_HUB_DISABLE_XET=1
# On the pod, raise the workflow's block-swap toward 0 and bump resolution to 720p.

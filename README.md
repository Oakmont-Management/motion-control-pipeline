# motion-control-pipeline — portable Wan 2.2 Animate (local → RunPod)

Portable wrapper around the **already-working** local Wan 2.2 Animate motion-control setup, so
the *same* graph runs on a rented 24 GB+ RunPod GPU at 720p / Q8 / full speed instead of the slow
8 GB RAM-offload path. **The model and workflow are unchanged** — this repo just isolates the three
things that differ between hosts (model paths, quant, launch flags) into config.

Full design rationale + phase-by-phase notes live in a local design doc (kept outside this repo).
This repo already applies that doc's **four corrections**:

1. **Wraps the existing workflow** `workflows/Wan22_Animate_8GB_GGUF_autoMask.json` (not the QuantStack example).
2. **Ships the auto-mask node** `custom_nodes/dwpose_sam2_seed/` — `install_nodes.sh` copies it in.
3. **Kijai-wrapper model set** — encoder `umt5-xxl-enc-fp8_e4m3fn.safetensors` (non-scaled), VAE
   `Wan2_1_VAE_bf16.safetensors`, unet `Wan2.2-Animate-14B-Q4_K_S.gguf` (local) → `-Q8_0.gguf` (RunPod).
4. **`HF_HUB_DISABLE_XET=1`** on every download.

The graph uses the **Kijai WanVideoWrapper** nodes (`WanVideoModelLoader`, `WanVideoTextEncodeCached`,
`WanVideoVAELoader`) — *not* the native ComfyUI GGUF/CLIP loaders the setup.md's inline scripts assume.

## What differs local ↔ RunPod (the entire portability surface)
| | Local (8 GB) | RunPod (24 GB+) |
|---|---|---|
| unet quant | `Wan2.2-Animate-14B-Q4_K_S.gguf` | `Wan2.2-Animate-14B-Q8_0.gguf` |
| flags | `--disable-async-offload`, block-swap 40 | none, block-swap → 0 |
| resolution | 480–576p | 720p, longer clips |
| paths | `configs/extra_model_paths.local.yaml` | `...runpod.yaml` |

Everything else (workflow JSON, custom nodes, VAE/encoder/LoRA/clip-vision files) is identical.

## Local — nothing to rebuild
Animate already runs in your local `ComfyUI` install. To drive it via this repo's config:
```bash
export COMFY_ROOT=/c/Users/<you>/Documents/ComfyUI   # your local ComfyUI path
bash scripts/run_local.sh          # or: .venv/Scripts/python.exe main.py --disable-async-offload --port 8000
# browser → load Wan22_Animate_8GB_GGUF_autoMask.json → set driving video + ref → Queue
```

## RunPod (the payoff)
```bash
# [MANUAL] RunPod: Network Volume >= 60 GB, deploy RTX 4090 (24 GB) + ComfyUI/torch template, mount /runpod-volume
git clone <this-repo-url> && cd motion-control-pipeline
export COMFY_ROOT=/runpod-volume/ComfyUI MODELS_ROOT=$COMFY_ROOT/models
bash scripts/install_nodes.sh
TIER=runpod bash scripts/download_models.sh     # pulls Q8_0 + Kijai set
bash scripts/verify_env.sh
bash scripts/run_runpod.sh
# In the browser: load workflows/Wan22_Animate_RunPod_720p_Q8.json (the 720p/Q8 base render),
# then finish with workflows/SeedVR2_Upscale_Finish.json. See "Realism path" below.
# [MANUAL] Stop the pod when idle — models persist on the network volume.
```

## Realism path — the actual Kling-Motion-Control replacement
Two workflows, run as separate passes (Animate 14B fully unloads before SeedVR2 loads → no VRAM clash;
and post stays a *finishing touch*, not a crutch — the base render must already be acceptable).

**Inputs:** a **4K hero image** (the desired character) + a **driving/reference video**. Output =
the hero performs the driving motion **inside the driving video's scene** (replacement mode: SAM2 masks
the person in the clip and swaps your character in, keeping its background/camera/lighting).

**Pass 1 — base render · `workflows/Wan22_Animate_RunPod_720p_Q8.json`**
Cloned from the working 8 GB autoMask graph, patched for a 24 GB+ pod:
| node | local (8 GB) | this 720p/Q8 variant |
|---|---|---|
| `WanVideoModelLoader` unet | `Wan2.2-Animate-14B-Q4_K_S.gguf` | **`-Q8_0.gguf`**, `load_device=main_device` |
| `WanVideoBlockSwap` blocks | 40 | **0** (model stays resident) |
| resolution (`INTConstant` Width/Height) | 576×768 | **720×960** (same 3:4 aspect, scaled up) |

- **Set resolution to match your driving clip.** Edit the two `INTConstant` nodes (titled *Width* / *Height*)
  — they are the single source of truth (they drive the resize + embeds). Keep both a multiple of 16.
  e.g. 720×1280 for 9:16, 1280×720 for 16:9. On 24 GB you can push higher (832×1088) as a realism knob.
- **Base-quality knobs** (left at proven defaults; dial on the first pod render if needed):
  `WanVideoSampler` **steps** 6 → try 8–10 for sharper detail · `WanVideoLoraSelectMulti` **lightx2v distill
  weight** 1.2 → lower it if motion looks soft/over-smoothed · `WanVideoClipVisionEncode` **strength** →
  lower toward the reference if the look drifts from the hero, raise for stronger scene coherence.
- **Guardrail:** the **MASK CHECK** preview (`DrawMaskOnImage`) must show the subject blacked out before
  you trust the swap — an empty mask fails silently and passes the driving video straight through.

**Pass 2 — finish · `workflows/SeedVR2_Upscale_Finish.json`**
Load Pass 1's saved mp4 → **SeedVR2 3B** video upscale (→ ~1440 short side, `batch_size 33` for temporal
consistency) → recombine with the **original audio + fps** (both carried through automatically). Weights
live in `models/SEEDVR2/` (`seedvr2_ema_3b_fp16` + `ema_vae_fp16`; `download_models.sh` pre-stages them,
and the node also auto-downloads on first use).

### INPUT PREP (hero image)
Frame the hero so it **roughly matches the driving subject's first frame** (scale, crop, orientation) —
Animate anchors identity/appearance from it. The standardized 4K hero is downscaled to the render
resolution; its extra detail feeds CLIP-vision cleanly, so keep the face sharp and well-lit.

### Scope note
This repo is **only** the motion-control engine (motion slice + SeedVR2 finish). The broader
"AI Content Engine" plan (a separate local design doc — LoRA training, voice,
ledger, orchestrator) is **parked / out of scope** here by design; identity comes from the hero image,
exactly like Kling.

## Notes
- `models/`, `inputs/`, `outputs/` are git-ignored; the repo is the portable unit, models re-hydrate per host.
- Git repo initialized (`master`); `.gitattributes` forces LF so the bash scripts run on the Linux pod.
- SAM2 (`sam2_hiera_base_plus`) + DWPose (`dw-ll_ucoco`, `yolox_l`) auto-download on first queue.
- Auto-mask guardrail: the workflow's "MASK CHECK" preview must show the subject blacked out before
  trusting replacement-mode output (an empty mask fails silently — see the local setup memory).

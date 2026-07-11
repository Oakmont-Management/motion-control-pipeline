# RUNPOD.md — run the motion-control engine on a rented GPU

**Mental model:** a workflow is just a recipe. To run it, three things must sit together on the pod —
(1) ComfyUI + custom nodes, (2) the model files, (3) the workflow JSON. This repo carries (1)'s installer
and (3); the scripts fetch (2) from HuggingFace. You drive the pod's ComfyUI through a browser URL — your
laptop is not involved in the render.

Result: a **4K hero image** + a **driving video** → the hero performs the driving motion **inside the
driving video's scene** (replacement mode), at 720p/Q8, finished with a SeedVR2 upscale.

---

## Step 0 — Deploy the pod
- RunPod → **Pods** → **RTX 4090 (24 GB)** (or L40/A5000 for more headroom).
- Pick a **ComfyUI template** (one with ComfyUI-Manager + a recent ComfyUI core).
- Attach a **Network Volume ≥ 60 GB** so models + this repo survive pod restarts.
- Note where the template installs ComfyUI — usually **`/workspace/ComfyUI`** on Pods.

## Step 1 — Set up (pod web terminal)
```bash
git clone https://github.com/Oakmont-Management/motion-control-pipeline
cd motion-control-pipeline

# Point at wherever the template put ComfyUI. Pods: /workspace/ComfyUI. Serverless: /runpod-volume/ComfyUI.
export COMFY_ROOT=/workspace/ComfyUI
export MODELS_ROOT=$COMFY_ROOT/models

bash scripts/install_nodes.sh                 # (1) custom nodes: WanVideoWrapper, SAM2, SeedVR2, ...
TIER=runpod bash scripts/download_models.sh   # (2) ~35-40 GB: Q8 Animate, encoder, VAE, clip, LoRAs, SeedVR2
bash scripts/verify_env.sh                     # confirms every node + weight is present
```
`verify_env.sh` must print `ALL CHECKS PASSED` before you spend render time.

## Step 2 — Launch ComfyUI + open the UI
```bash
bash scripts/run_runpod.sh
```
Then open the pod's **port 8188** HTTP service (RunPod → your pod → "Connect" → the `:8188` proxy URL).
`run_runpod.sh` auto-pins the model paths to `COMFY_ROOT`, so the dropdowns will be populated. It also
copies this repo's workflows into `ComfyUI/user/default/workflows/`, so both appear in the **Workflows**
sidebar (left panel) — just click to open, no local file needed.

## Step 3 — Base render
1. Open **`Wan22_Animate_RunPod_720p_Q8.json`** from the **Workflows** sidebar (or drag the file onto the canvas).
2. **`Load Image`** node (in the *Reference Image* group) → upload your **4K hero**.
3. **`Load Video (Upload)`** node → upload your **driving video**.
4. (If your clip isn't 3:4) set the two **`INTConstant`** nodes (*Width* / *Height*) to match its aspect,
   both multiples of 16 — e.g. `720×1280` for 9:16, `1280×720` for 16:9.
5. **Confirm the MASK CHECK preview** (`DrawMaskOnImage`) shows the subject blacked out. An empty mask
   fails silently and passes the driving video through unchanged.
6. **Queue.** The saved mp4 lands in `ComfyUI/output/`.

## Step 4 — Finish (SeedVR2 upscale)
1. Open **`SeedVR2_Upscale_Finish.json`** from the **Workflows** sidebar (or drag the file onto the canvas).
2. Point its **`Load Video`** node at the Step-3 output.
3. **Queue.** Output preserves the original audio + fps, upscaled to ~1440.

## Step 5 — Collect + stop
- Download the finished mp4.
- **Stop the pod.** Models + this repo persist on the network volume, so the next session skips Steps 1's
  downloads (re-run `install_nodes.sh` only if the template resets custom_nodes).

---

## Base-quality knobs (dial on the first render if the raw clip isn't yet *acceptable*)
The base render should look good **before** SeedVR2 — post is a finishing touch, not the fix. In the base
workflow:
- `WanVideoSampler` **steps** 6 → 8–10 (sharper detail, slower).
- `WanVideoLoraSelectMulti` **lightx2v distill weight** 1.2 → lower if motion looks soft/over-smoothed.
- `WanVideoClipVisionEncode` **strength** → lower toward the reference if the look drifts from the hero;
  raise for stronger scene coherence.
- Resolution → push above 720p on 24 GB if you want (e.g. `832×1088`).

## Troubleshooting
- **Empty model dropdowns / "model not found":** `COMFY_ROOT` doesn't match where models were downloaded.
  Re-export the correct `COMFY_ROOT`, re-run `download_models.sh` (or move the models), relaunch.
- **MASK CHECK preview is empty/black everywhere:** the subject wasn't segmented — check the driving video
  loaded and DWPose/SAM2 downloaded (they auto-fetch on first queue).
- **OOM at 720p:** lower the resolution `INTConstant`s, or set `WanVideoBlockSwap` blocks > 0 to trade
  speed for VRAM.

## Cost note
RTX 4090 ≈ $0.34–0.69/hr. First-time model download (~35–40 GB) is ~15–30 min of paid time; the network
volume means you pay that once (+ ~$0.07/GB/month to keep the volume).

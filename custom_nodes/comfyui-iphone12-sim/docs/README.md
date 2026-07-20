# comfyui-iphone12-sim

Makes AI-generated video (Wan 2.2 Animate → SeedVR2) look like it was shot on an iPhone 12
and posted to Instagram. It does this by simulating the whole physical capture chain, not by
adding a grain filter — AI video reads as fake because it never went through a real camera and
a real codec.

```
optics → sensor noise → ISP → motion → codec (×2 real H.264 generations)
```

The single biggest realism contributor is the codec stage running two **real** libx264
generations (not a DCT approximation).

## Nodes (`CATEGORY = iPhone12Sim`)
- **iPhone 12 Capture Sim** (`IPhone12CaptureSim`) — the whole stack, one node, locked preset.
- **iPhone 12 Preflight Check** — validates fps/resolution/codec-backend/scene-ISO before a long render.
- **iPhone 12 Preset Info** — dumps a preset's values and notes.

Presets: `iphone12_day` (bright, tell = tone curve + sharpen + codec, near-clean noise) and
`iphone12_night` (low light, noisy + soft, sharpens *less*). `auto` picks by scene ISO.

## Where it sits in this pipeline
This package plugs into the tail of the SeedVR2 finishing pass. The real workflow uses
ComfyUI's native video nodes, not VHS:

```
LoadVideo → GetVideoComponents → SeedVR2VideoUpscaler
  → IPhone12PreflightCheck → IPhone12CaptureSim → CreateVideo → SaveVideo
```

Integrated via a copy of `SeedVR2_Upscale_Finish.json` → `SeedVR2_iPhone12_Finish.json`
(original untouched). Both new nodes' `fps` input is wired to the same `GetVideoComponents`
fps output `CreateVideo` already uses, so they can't drift out of sync with the container's
actual framerate.

## Why the order is fixed (spec 9)
1. **Sim AFTER the upscaler.** A camera's noise/lens live at capture resolution. Sim-then-upscale
   magnifies grain into mush and destroys codec artifacts. Capture last.
2. **Turn SeedVR2 restoration DOWN** — in restoration mode it re-adds synthetic skin texture (the
   glassiness you're removing) and fights the downstream degradation. Its only job here is resolution.
3. **fps ≥ 24.** iPhone 12 never shot below 24; the cadence tells regardless of everything else.
   (This pipeline gets fps from the driving video, so no RIFE stage — unlike the generic spec graph.)
4. If output still reads as AI, fix upstream: CFG 4.0–5.5, negatives for `plastic skin, airbrushed,
   CGI, waxy`. The Wan VAE structurally low-passes skin texture — which is *why* this node replaces
   lost texture with synthetic texture of correct statistics (what a real ISP does anyway).

## Determinism & safety
Every stochastic stage takes a seed → same seed, bit-identical output. The top-level node
catches any exception and passes frames through unmodified, so a failure never kills a long render.

## Status
Build complete per the staged order (spec 11) and `iphone effect node implementation.md`:
all 8 stages implemented, both test suites fully green, zero skips (`tests/test_validation.py`
75 assertions, `tests/test_presets.py` 32 assertions — spec 8 named 66/45 as targets; the
delivered suites cover the same checklist with denser or additional assertions per group,
not a 1:1 count match). Integrated into `SeedVR2_iPhone12_Finish.json`; independently
audited (fresh-eyes review, zero correctness bugs found) and validated live against a
running ComfyUI's `/object_info` and `validate_workflow` (0 errors, 0 warnings). Not yet
smoke-tested against a real SeedVR2 render end-to-end — the SeedVR2 node pack itself isn't
installed on the machine this was built on (unrelated, pre-existing gap).

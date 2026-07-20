# Calibration (from real iPhone 12 footage)

The default constants are grounded in sensor physics (spec 2.4) except the Smart HDR tone
curve (spec 2.6), which is the least rigorously grounded constant — calibrate it first when
real footage arrives.

## Shoot this, not that
- **DO** shoot a **brightness gradient** (e.g. a wall with a soft light falloff, or a grey card
  under a gradient). Separating read noise from shot noise needs variance measured against
  multiple signal *levels*.
- **DON'T** shoot a flat evenly-lit wall — a single signal level gives no line, and least-squares
  returns a confident wrong answer (all variance → read term, shot = 0, absurd ISO).
- Shoot in **video mode**, not photo. iPhone 12 Night Mode does **not** apply to video; video just
  gets noisy and soft. Calibrating from photo-mode stills gives the wrong answer.

## What the harness extracts
Noise model (photon-transfer curve on flat patches via temporal diff), chroma ratio, spatial
correlation, temporal persistence, tone curve, sharpening overshoot → `*_calibrated.json`.

## Two gotchas the harness handles
- **Degenerate noise fit:** if signal spread is too low it refuses and tells you to shoot a gradient.
- **4:2:0 chroma understatement:** any phone file is H.264 4:2:0; chroma planes are half-res and
  average away chroma noise before you can measure it. The harness reports raw AND a
  subsampling-compensated estimate (×1.6, conservative vs the theoretical 2.0).

ISO is backed out of the **shot** term (`gain = (shot/0.00135)²`, `ISO = 32·gain`), the
photon-limited component that tracks sensor gain — not the read term.

## Usage
```
python calibration/calibrate_from_footage.py path/to/clip.mov [out.json]
```

## A scale caveat worth knowing before you read the output

The harness's ISO back-out (`gain = (shot/0.00135)^2`) expects `shot` to be the temporal
shot-noise sigma measured in a **real, post-ISP, post-H.264 clip** — noise that's already
been through in-camera NR and codec compression, so it's small. `iphone_core.py`'s own
`apply_sensor_noise` intentionally simulates the opposite end of the pipeline (raw
*pre-ISP* sensor noise, deliberately prominent because the ISP/codec stages downstream
still need to tame it). Feeding the harness synthetic frames built from
`iphone_core.apply_sensor_noise` directly will *not* round-trip to the same nominal ISO —
that's not a bug, it's two different noise domains. Validated instead at noise magnitudes
representative of real compressed footage (temporal sigma ~0.003–0.02 in linear [0,1]):
the fit is monotonic and clamps correctly across that whole realistic band.

The chroma-ratio compensation, by contrast, **is** scale-invariant (it's a ratio, not an
absolute sigma) and was round-tripped end-to-end: injected a known `chroma_ratio=2.1`,
simulated real 4:2:0 subsampling loss (2×2 chroma average + upsample) on top, and the
harness recovered raw≈1.33 / compensated≈2.13 — matching this doc's stated round-trip
target almost exactly.

_Status: implemented (Stage 8). Degenerate-fit refusal and the 4:2:0 compensation are both
exercised by a synthetic self-test (no real footage needed to sanity-check the math); the
tone-curve and sharpen-overshoot extractions are single-clip approximations per the note
above and should be the first things re-validated once real footage arrives._

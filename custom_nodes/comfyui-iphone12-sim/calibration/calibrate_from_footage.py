"""calibrate_from_footage.py - extract an iPhone 12 model from real .mov footage (spec 6).

Numpy/scipy allowed here (calibration harness only - iphone_core.py stays pure torch).
Writes *_calibrated.json in the same five-section shape as presets/*.json (see docs/CALIBRATION.md).

Extracts: noise model (photon-transfer curve via temporal diff on flat patches),
chroma ratio, spatial corr, temporal persistence, an approximate tone curve, and a
sharpening-overshoot estimate.

SHOOTING INSTRUCTIONS (read before capturing):
  Shoot a BRIGHTNESS GRADIENT, not a flat evenly-lit wall. Separating read noise from shot
  noise needs variance-vs-signal-LEVEL; a single signal level gives no line and the fit
  returns a confident wrong answer (all variance -> read, shot=0, absurd ISO).
  Shoot in VIDEO mode, not photo - iPhone 12 Night Mode does not apply to video.

Two subtleties the harness handles (spec 6):
  - Degenerate noise fit: detects low signal spread and REFUSES, telling the user to shoot
    a gradient, instead of returning a confident wrong answer.
  - 4:2:0 chroma understatement: any phone file is H.264 4:2:0 - chroma planes are half-res
    and average away chroma noise before you can measure it. Reports raw AND a subsampling-
    compensated estimate (x1.6, conservative vs the theoretical 2.0 because noise is already
    spatially correlated). Round-trip target: recover ~2.1 from a synthetically known 2.1.

Back out ISO from the SHOT term: gain = (shot/0.00135)^2, ISO = 32*gain - shot (not read)
is the photon-limited component that actually tracks sensor gain.

The tone-curve and sharpen-overshoot extractions below are honest approximations from a
single gradient clip (no calibrated step-wedge chart), consistent with CALIBRATION.md's
note that the Smart HDR curve is the least rigorously grounded constant in this package.
"""
import json
import os
import sys

import numpy as np


# ----------------------------------------------------------------------------
# Colour helpers (numpy mirror of iphone_core's torch versions - kept independent
# per spec 0: iphone_core.py stays pure torch, this harness stays pure numpy/scipy)
# ----------------------------------------------------------------------------
def _srgb_to_linear(x):
    a = 0.055
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)


def _rgb_to_ycbcr(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack((y, cb, cr), axis=-1)


# ----------------------------------------------------------------------------
# Frame loading (I/O boundary - kept separate from the numpy-only extraction logic
# below so the extraction can be unit-tested with synthetic frames, no .mov needed)
# ----------------------------------------------------------------------------
def _load_frames(mov_path):
    """Decode a video file to an (N,H,W,3) float64 [0,1] sRGB array via PyAV."""
    try:
        import av
    except ImportError:
        raise RuntimeError("calibrate_from_footage: PyAV required (pip install av)")
    container = av.open(mov_path)
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24").astype(np.float64) / 255.0
        frames.append(arr)
    container.close()
    if len(frames) < 8:
        raise RuntimeError(
            f"calibrate_from_footage: only {len(frames)} frames decoded - need a real clip, "
            f"not a single still (temporal-diff noise measurement needs consecutive frames)."
        )
    return np.stack(frames, axis=0)


# ----------------------------------------------------------------------------
# Flat-patch photon-transfer curve (PTC)
# ----------------------------------------------------------------------------
def _patch_grid(h, w, patch, stride):
    ys = list(range(0, max(h - patch, 1), stride)) or [0]
    xs = list(range(0, max(w - patch, 1), stride)) or [0]
    return ys, xs


def _flat_patch_stats(luma, patch=24, stride=32, flatness_thresh=0.02):
    """Per spatially-flat patch: (signal_level, temporal_noise_var).

    var = 0.5*mean((frame_t - frame_{t+1})^2) over the patch - this cancels any static
    spatial structure and isolates TEMPORAL (sensor) noise, so it's safe to run over a
    whole gradient frame rather than needing a perfectly flat wall for this step; the
    "flat patch" filter here is about local flatness (no real detail to alias into the
    diff), not about the frame overall being flat (which is the wrong shot - see module
    docstring).
    """
    n, h, w = luma.shape
    py = min(patch, h)
    px = min(patch, w)
    ys, xs = _patch_grid(h, w, py, stride)
    signals, variances = [], []
    for y in ys:
        for x in xs:
            block = luma[:, y:y + py, x:x + px]
            if block.shape[1] < 2 or block.shape[2] < 2:
                continue
            spatial_std = block[0].std()
            if spatial_std > flatness_thresh:
                continue
            diff = block[:-1] - block[1:]
            variances.append(0.5 * float(np.mean(diff ** 2)))
            signals.append(float(block.mean()))
    return np.asarray(signals), np.asarray(variances)


def _fit_ptc(signals, variances):
    """var ~= read_var + shot_var_coeff * signal (matches iphone_core's additive noise
    model). Returns (sigma_read, sigma_shot). Raises on a degenerate (too-flat) fit."""
    if len(signals) < 6:
        raise RuntimeError(
            "calibrate_from_footage: not enough flat patches found to fit a noise curve. "
            "Shoot a BRIGHTNESS GRADIENT (soft light falloff on a wall, or a grey card under "
            "a gradient) - a single signal level gives no line to fit."
        )
    spread = float(signals.max() - signals.min())
    if spread < 0.05:
        raise RuntimeError(
            f"calibrate_from_footage: signal spread too low ({spread:.4f} over "
            f"[{signals.min():.3f}, {signals.max():.3f}]) to separate read from shot noise - "
            f"this looks like a flat evenly-lit wall. Shoot a BRIGHTNESS GRADIENT instead "
            f"(see module docstring): least-squares would otherwise return a confident wrong "
            f"answer (all variance -> read term, shot=0, absurd ISO)."
        )
    slope, intercept = np.polyfit(signals, variances, 1)
    sigma_read = float(np.sqrt(max(intercept, 0.0)))
    sigma_shot = float(np.sqrt(max(slope, 0.0)))
    return sigma_read, sigma_shot


# ----------------------------------------------------------------------------
# Chroma ratio, spatial correlation, temporal persistence
# ----------------------------------------------------------------------------
def _lag_corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def _chroma_ratio(lin, luma, patch=24, stride=32, flatness_thresh=0.02):
    """Raw chroma/luma temporal-noise ratio over flat patches, plus the 4:2:0
    subsampling-compensated estimate (x1.6, conservative vs the theoretical 2.0)."""
    ycc = _rgb_to_ycbcr(lin)
    n, h, w = luma.shape
    py = min(patch, h)
    px = min(patch, w)
    ys, xs = _patch_grid(h, w, py, stride)
    luma_vars, chroma_vars = [], []
    for y in ys:
        for x in xs:
            yblock = luma[:, y:y + py, x:x + px]
            if yblock.shape[1] < 2 or yblock.shape[2] < 2 or yblock[0].std() > flatness_thresh:
                continue
            cblock = ycc[:, y:y + py, x:x + px, 1:3]
            ydiff = yblock[:-1] - yblock[1:]
            cdiff = cblock[:-1] - cblock[1:]
            luma_vars.append(0.5 * float(np.mean(ydiff ** 2)))
            chroma_vars.append(0.5 * float(np.mean(cdiff ** 2)))
    if not luma_vars or sum(luma_vars) < 1e-12:
        return None, None
    luma_sigma = float(np.sqrt(np.mean(luma_vars)))
    chroma_sigma = float(np.sqrt(np.mean(chroma_vars)))
    raw = chroma_sigma / luma_sigma if luma_sigma > 1e-12 else 0.0
    return raw, raw * 1.6


def _spatial_corr(luma, patch=48, flatness_thresh=0.02):
    """Lag-1 horizontal correlation of the temporal-diff noise field in the flattest
    available patch - estimates demosaic spatial correlation (spec 2.4.1)."""
    n, h, w = luma.shape
    py, px = min(patch, h), min(patch, w)
    ys, xs = _patch_grid(h, w, py, py)
    best_std, best = None, None
    for y in ys:
        for x in xs:
            block = luma[:, y:y + py, x:x + px]
            if block.shape[1] < 2 or block.shape[2] < 3:
                continue
            s = block[0].std()
            if best_std is None or s < best_std:
                best_std, best = s, block
    if best is None:
        return 0.0
    diff = best[:-1] - best[1:]
    return _lag_corr(diff[:, :, :-1].flatten(), diff[:, :, 1:].flatten())


def _temporal_persistence(luma, patch=48, flatness_thresh=0.02):
    """Lag-1 frame-to-frame correlation of the temporal noise field."""
    n, h, w = luma.shape
    py, px = min(patch, h), min(patch, w)
    ys, xs = _patch_grid(h, w, py, py)
    best_std, best = None, None
    for y in ys:
        for x in xs:
            block = luma[:, y:y + py, x:x + px]
            s = block[0].std()
            if best_std is None or s < best_std:
                best_std, best = s, block
    if best is None or best.shape[0] < 3:
        return 0.0
    mean_per_frame = best.mean(axis=(1, 2), keepdims=True)
    noise = best - mean_per_frame
    return _lag_corr(noise[:-1].flatten(), noise[1:].flatten())


# ----------------------------------------------------------------------------
# Tone curve (approximate - needs a step-wedge chart for a rigorous result) and
# sharpen-overshoot estimate
# ----------------------------------------------------------------------------
def _tone_curve_samples(first_frame_luma, n_points=10):
    """Approximate response curve from the brightness gradient itself: assumes the
    gradient's spatial axis (whichever axis has the larger range) varies linearly in
    SCENE luminance, then samples the measured sRGB output at even steps along it.
    This is a rough diagnostic, not a lab measurement - a real calibration should use
    a step-wedge / grey-card chart with known reflectance values."""
    row_range = float(first_frame_luma.mean(axis=1).max() - first_frame_luma.mean(axis=1).min())
    col_range = float(first_frame_luma.mean(axis=0).max() - first_frame_luma.mean(axis=0).min())
    profile = first_frame_luma.mean(axis=1) if row_range >= col_range else first_frame_luma.mean(axis=0)
    idx = np.linspace(0, len(profile) - 1, n_points).astype(int)
    xs = np.linspace(0.0, 1.0, n_points).tolist()
    ys = [float(profile[i]) for i in idx]
    lo, hi = min(ys), max(ys)
    if hi - lo > 1e-6:
        ys = [(v - lo) / (hi - lo) for v in ys]
    return [{"x": x, "y": y} for x, y in zip(xs, ys)]


def _sharpen_overshoot(luma_frame):
    """Ringing-ratio estimate at the strongest edge: overshoot peak above the plateau,
    relative to the edge's step height, along whichever 1D profile has the steepest
    single-step gradient."""
    gy = np.abs(np.diff(luma_frame, axis=0))
    gx = np.abs(np.diff(luma_frame, axis=1))
    if gy.max() >= gx.max():
        r, c = np.unravel_index(np.argmax(gy), gy.shape)
        profile = luma_frame[:, c]
    else:
        r, c = np.unravel_index(np.argmax(gx), gx.shape)
        profile = luma_frame[r, :]
    edge = int(np.argmax(np.abs(np.diff(profile))))
    lo_win = profile[max(0, edge - 8):edge]
    hi_win = profile[edge + 1:edge + 9]
    if len(lo_win) < 2 or len(hi_win) < 2:
        return 0.0
    step = abs(float(hi_win.mean()) - float(lo_win.mean()))
    if step < 1e-6:
        return 0.0
    overshoot = float(max(hi_win.max() - hi_win.mean(), lo_win.mean() - lo_win.min()))
    return overshoot / step


# ----------------------------------------------------------------------------
# Core extraction (pure numpy - unit-testable with synthetic frames, no .mov needed)
# ----------------------------------------------------------------------------
def extract_model(frames_srgb):
    """frames_srgb: (N,H,W,3) float64 [0,1] sRGB -> calibrated model dict.
    Raises RuntimeError on a degenerate (flat-wall) shot."""
    lin = _srgb_to_linear(frames_srgb)
    luma = lin.mean(axis=-1)

    signals, variances = _flat_patch_stats(luma)
    sigma_read, sigma_shot = _fit_ptc(signals, variances)

    gain = (sigma_shot / 0.00135) ** 2 if sigma_shot > 0 else 0.0
    iso = float(np.clip(32.0 * gain, 32.0, 1600.0))

    chroma_raw, chroma_compensated = _chroma_ratio(lin, luma)
    spatial_corr = _spatial_corr(luma)
    temporal_persistence = _temporal_persistence(luma)
    tone_curve = _tone_curve_samples(luma[0])
    sharpen_overshoot = _sharpen_overshoot(luma[0])

    return {
        "name": "calibrated",
        "notes": [
            f"Auto-extracted from {frames_srgb.shape[0]} frames "
            f"({frames_srgb.shape[1]}x{frames_srgb.shape[2]}).",
            "Tone curve and sharpen_overshoot are rough single-clip estimates, not a "
            "lab measurement - see docs/CALIBRATION.md.",
            f"Chroma ratio: raw={chroma_raw} understates true chroma noise (4:2:0 "
            f"subsampling averages it away before capture); compensated x1.6 is reported "
            f"alongside it and is the value to actually use.",
        ],
        "sensor": {
            "iso": round(iso, 1),
            "sigma_read": sigma_read,
            "sigma_shot": sigma_shot,
            "chroma_ratio_raw": chroma_raw,
            "chroma_ratio_compensated": chroma_compensated,
            "spatial_corr": spatial_corr,
            "temporal_persistence": temporal_persistence,
        },
        "isp": {
            "tone_curve_samples": tone_curve,
            "sharpen_overshoot_estimate": sharpen_overshoot,
        },
    }


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def calibrate(mov_path, out_json=None):
    frames = _load_frames(mov_path)
    model = extract_model(frames)
    model["notes"].insert(0, f"Source: {os.path.basename(mov_path)}")

    if out_json is None:
        base = os.path.splitext(os.path.basename(mov_path))[0]
        out_json = os.path.join(os.path.dirname(os.path.abspath(mov_path)), f"{base}_calibrated.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"[calibrate] wrote {out_json}")
    comp = model["sensor"]["chroma_ratio_compensated"]
    chroma_disp = f"{comp:.2f}" if comp is not None else "n/a (no flat-patch chroma signal found)"
    print(f"[calibrate] ISO~{model['sensor']['iso']:.0f}  "
          f"chroma_ratio~{chroma_disp} (compensated)  "
          f"spatial_corr~{model['sensor']['spatial_corr']:.2f}  "
          f"temporal_persistence~{model['sensor']['temporal_persistence']:.2f}")
    return model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: calibrate_from_footage.py <clip.mov> [out.json]")
        print("  shoot a BRIGHTNESS GRADIENT, not a flat wall (see module docstring).")
        print("  shoot in VIDEO mode - iPhone 12 Night Mode does not apply to video.")
        sys.exit(2)
    try:
        calibrate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    except RuntimeError as e:
        print(f"[calibrate] REFUSED: {e}")
        sys.exit(1)

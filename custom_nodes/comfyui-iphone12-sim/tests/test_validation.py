"""test_validation.py - physics/signal-property assertions on iphone_core.py + iphone_codec.py
(spec 8's "66 assertions" target; this suite landed at 75 `assert` statements once every
group was filled in - some spec checks needed more than one assertion to pin down).

Properties, NOT golden images (golden images break on any torch bump and prove nothing).
sys.exit(1) on any failure so this can gate a deploy. Grouped by the spec-8 checklist;
every group below is implemented (none raise NotImplementedError).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import iphone_core as core

torch.manual_seed(0)


def group_colour():          # Stage 1 - round-trips identity; linearisation darkens midtones
    x = torch.rand(2, 16, 24, 3)

    # sRGB <-> linear round-trips to identity
    assert torch.allclose(core.linear_to_srgb(core.srgb_to_linear(x)), x, atol=1e-5), "srgb->lin->srgb identity"
    assert torch.allclose(core.srgb_to_linear(core.linear_to_srgb(x)), x, atol=1e-5), "lin->srgb->lin identity"

    # RGB <-> YCbCr round-trips to identity
    assert torch.allclose(core.ycbcr_to_rgb(core.rgb_to_ycbcr(x)), x, atol=1e-5), "rgb->ycbcr->rgb identity"
    assert torch.allclose(core.rgb_to_ycbcr(core.ycbcr_to_rgb(x)), x, atol=1e-5), "ycbcr->rgb->ycbcr identity"

    # Linearisation darkens midtones (sRGB is perceptual; linear 0.5 grey ~= 0.214)
    mid = torch.tensor([0.5])
    assert core.srgb_to_linear(mid).item() < 0.5, "linearise darkens midtones"
    assert core.linear_to_srgb(mid).item() > 0.5, "encode brightens midtones (inverse)"

    # Endpoints are fixed points (black->black, white->white)
    ends = torch.tensor([0.0, 1.0])
    assert torch.allclose(core.srgb_to_linear(ends), ends, atol=1e-6), "srgb_to_linear endpoints"
    assert torch.allclose(core.linear_to_srgb(ends), ends, atol=1e-6), "linear_to_srgb endpoints"

    # YCbCr luma weighting: pure green is the brightest primary (0.587)
    prim = torch.tensor([[[[1., 0, 0]], [[0, 1., 0]], [[0, 0, 1.]]]])  # R,G,B as (1,3,1,3)
    y = core.rgb_to_ycbcr(prim)[..., 0].flatten()
    assert y[1] > y[0] > y[2], "BT.601 luma: G > R > B"

    # gaussian_blur: sigma<=0 is a bit-identical no-op; blur preserves mean, reduces variance; shape/dtype kept
    assert torch.equal(core.gaussian_blur(x, 0.0), x), "gaussian_blur sigma=0 is bit-identical"
    blurred = core.gaussian_blur(x, 1.5)
    assert blurred.shape == x.shape and blurred.dtype == x.dtype, "gaussian_blur preserves shape/dtype"
    assert blurred.var().item() < x.var().item(), "gaussian_blur reduces variance"
    # A flat field is a fixed point (reflect-pad preserves constants exactly).
    flat = torch.full((1, 12, 12, 3), 0.37)
    assert torch.allclose(core.gaussian_blur(flat, 2.0), flat, atol=1e-6), "gaussian_blur preserves a flat field"

def _flat(v, b=8, h=64, w=64):
    return torch.full((b, h, w, 3), float(v))


def _lin_noise_sigma(level, iso, **kw):  # noise std measured in LINEAR light
    img = _flat(level)
    kw.setdefault("spatial_corr", 0.0); kw.setdefault("fpn_amount", 0.0)
    kw.setdefault("temporal_persistence", 0.0); kw.setdefault("seed", 1)
    out = core.apply_sensor_noise(img, iso=iso, **kw)
    return (core.srgb_to_linear(out) - core.srgb_to_linear(img)).std().item()


def _srgb_noise_sigma(level, iso, **kw):  # noise std measured in sRGB (what the eye sees)
    img = _flat(level)
    kw.setdefault("spatial_corr", 0.0); kw.setdefault("fpn_amount", 0.0)
    kw.setdefault("temporal_persistence", 0.0); kw.setdefault("seed", 1)
    out = core.apply_sensor_noise(img, iso=iso, **kw)
    return (out - img).std().item()


def _lag_corr(field):  # Pearson correlation of a 1D pair, mean-removed
    a, b = field
    a = a - a.mean(); b = b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def group_noise_physics():   # Stage 2 - shot noise rises with signal in LINEAR light; SNR up with exposure;
                             #           visible sRGB noise higher in shadows (emergent); monotonic with ISO;
                             #           ISO32 near-clean (<0.006 sRGB sigma)
    # Shot noise rises with signal (measured in LINEAR light - sRGB would be a category error)
    lo = _lin_noise_sigma(0.1, 400)
    hi = _lin_noise_sigma(0.8, 400)
    assert hi > lo, "shot noise rises with signal in linear light"

    # SNR improves with exposure: relative noise (sigma/signal) falls as signal rises
    assert (hi / core.srgb_to_linear(torch.tensor(0.8)).item()) \
         < (lo / core.srgb_to_linear(torch.tensor(0.1)).item()), "SNR improves with exposure"

    # Visible sRGB noise is HIGHER in shadows (emergent from the sRGB encode slope, no special-casing)
    assert _srgb_noise_sigma(0.05, 400) > _srgb_noise_sigma(0.8, 400), "sRGB noise higher in shadows"

    # Noise increases monotonically with ISO
    s100, s400, s1600 = (_srgb_noise_sigma(0.3, iso) for iso in (100, 400, 1600))
    assert s100 < s400 < s1600, "noise monotonic with ISO"

    # ISO 32 near-clean (bright signal, where the sRGB slope is shallowest)
    assert _srgb_noise_sigma(0.9, 32) < 0.006, "ISO 32 near-clean (<0.006 sRGB sigma)"


def group_rgb_gain():        # Stage 2 - RGB sigma matches physical constant within 10% across
                             #           chroma_ratio in {1.0,1.6,2.4}  (locks Bug #2)
    import math
    # The gain-compensation constant itself: ~1.72x even at chroma_ratio=1 (spec predicts 1.722).
    g_r = math.sqrt(1 + (1.402 * 1.0) ** 2)
    g_g = math.sqrt(1 + (0.344136 * 1.0) ** 2 + (0.714136 * 1.0) ** 2)
    g_b = math.sqrt(1 + (1.772 * 1.0) ** 2)
    rgb_gain = math.sqrt((g_r ** 2 + g_g ** 2 + g_b ** 2) / 3)
    assert abs(rgb_gain - 1.722) < 0.02, "rgb_gain ~= 1.722 at chroma_ratio=1 (Bug #2 constant)"

    # After compensation the measured RGB sigma equals the physical sigma_map, independent of
    # chroma_ratio - that is the whole point of Bug #2.
    level, iso = 0.4, 400
    sig = core.srgb_to_linear(torch.tensor(level)).item()
    gain = iso / 32.0
    sr = (core.READ_NOISE_E / core.FULL_WELL_E) * math.sqrt(gain)
    ss = (1.0 / math.sqrt(core.FULL_WELL_E)) * math.sqrt(gain)
    predicted = math.sqrt(sr ** 2 + ss ** 2 * sig)

    measured = [_lin_noise_sigma(level, iso, chroma_ratio=cr, seed=2) for cr in (1.0, 1.6, 2.4)]
    for cr, m in zip((1.0, 1.6, 2.4), measured):
        assert abs(m - predicted) / predicted < 0.10, f"RGB sigma matches physical constant (cr={cr})"
    # ... and is stable across chroma_ratio (no 1.7-3x blow-up)
    assert (max(measured) - min(measured)) / predicted < 0.10, "RGB sigma stable across chroma_ratio"


def group_noise_structure(): # Stage 2 - chroma>luma; spatial_corr raises neighbour corr;
                             #           temporal_persistence correlates frames; FPN frame-invariant
    img = _flat(0.4, b=12)

    # Chroma noise > luma noise (measured in the linear YCbCr where it is injected)
    out = core.apply_sensor_noise(img, iso=800, chroma_ratio=2.4, spatial_corr=0.0,
                                  fpn_amount=0.0, temporal_persistence=0.0, seed=3)
    dyc = core.rgb_to_ycbcr(core.srgb_to_linear(out)) - core.rgb_to_ycbcr(core.srgb_to_linear(img))
    assert dyc[..., 1:3].std().item() > dyc[..., 0].std().item(), "chroma noise > luma noise"

    # spatial_corr raises neighbour (horizontal) correlation
    def neighbour_corr(sc):
        o = core.apply_sensor_noise(img, iso=800, spatial_corr=sc, chroma_ratio=1.0,
                                    fpn_amount=0.0, temporal_persistence=0.0, seed=4)
        n = (core.srgb_to_linear(o) - core.srgb_to_linear(img))[..., 0]
        return _lag_corr((n[:, :, :-1].flatten(), n[:, :, 1:].flatten()))
    assert neighbour_corr(0.8) > neighbour_corr(0.0) + 0.1, "spatial_corr raises neighbour correlation"

    # temporal_persistence correlates adjacent frames
    def frame_corr(tp):
        o = core.apply_sensor_noise(img, iso=800, spatial_corr=0.0, chroma_ratio=1.0,
                                    fpn_amount=0.0, temporal_persistence=tp, seed=5)
        n = (core.srgb_to_linear(o) - core.srgb_to_linear(img))[..., 0]
        return _lag_corr((n[:-1].flatten(), n[1:].flatten()))
    assert frame_corr(0.85) > 0.5, "temporal_persistence correlates frames (r>0.5 @ 0.85)"
    assert frame_corr(0.85) > frame_corr(0.0) + 0.3, "persistence raises frame correlation"

    # FPN frame-invariant (fpn_amount=1 -> the whole noise field is the fixed pattern)
    o = core.apply_sensor_noise(img, iso=800, spatial_corr=0.0, chroma_ratio=1.0,
                                fpn_amount=1.0, temporal_persistence=0.0, seed=6)
    assert o.std(dim=0).max().item() < 1e-6, "FPN frame-invariant across frames (Bug #3)"


def group_tiling():          # Stage 2 - magnitude within 0.1%; deterministic; survives persistence (r>0.5);
                             #           small inputs bit-identical  (locks Bug #5)
    # Small input: auto mode must NOT tile -> bit-identical to the untiled impl
    small = torch.rand(4, 32, 32, 3)
    a = core.apply_sensor_noise(small, iso=400, seed=7)
    b = core._apply_sensor_noise_impl(small, 400, 0.0, 1.9, 0.55, 0.15, 7, 1.0)
    assert torch.equal(a, b), "small input bit-identical to untiled path (Bug #5)"

    # Tiled path is deterministic (same seed -> identical)
    big = torch.rand(4, 300, 300, 3)
    assert torch.equal(core.apply_sensor_noise(big, iso=400, seed=8, tile=128),
                       core.apply_sensor_noise(big, iso=400, seed=8, tile=128)), "tiled path deterministic"

    # Tiling preserves overall noise magnitude
    med = _flat(0.4, b=6, h=256, w=256)
    su = _lin_noise_sigma_of(med, tile=0)
    st = _lin_noise_sigma_of(med, tile=128)
    assert abs(st - su) / su < 0.001, "tiling preserves noise magnitude (within 0.1%)"

    # Temporal persistence survives tiling (each tile is a full-length clip)
    seq = _flat(0.4, b=8, h=256, w=256)
    o = core.apply_sensor_noise(seq, iso=800, seed=10, tile=128, temporal_persistence=0.85,
                                spatial_corr=0.0, fpn_amount=0.0)
    n = (core.srgb_to_linear(o) - core.srgb_to_linear(seq))[..., 0]
    assert _lag_corr((n[:-1].flatten(), n[1:].flatten())) > 0.5, "persistence survives tiling (r>0.5)"


def _lin_noise_sigma_of(img, **kw):
    kw.setdefault("iso", 800); kw.setdefault("seed", 9)
    kw.setdefault("spatial_corr", 0.0); kw.setdefault("fpn_amount", 0.0)
    out = core.apply_sensor_noise(img, **kw)
    return (core.srgb_to_linear(out) - core.srgb_to_linear(img)).std().item()

def group_optics():          # Stage 3 - optics/vignette/corner-softness behave; strength=0 no-op (Bug #1)
    img = torch.rand(2, 48, 64, 3)

    # strength=0 is a bit-identical no-op (an identity built via grid_sample with the wrong
    # align_corners would NOT be - this guards the resample convention, Bug #1)
    assert torch.equal(core.apply_lens_geometry(img, strength=0.0), img), "optics strength=0 no-op"
    warped = core.apply_lens_geometry(img, strength=1.0)
    assert warped.shape == img.shape, "optics preserves shape"
    assert not torch.equal(warped, img), "optics strength=1 warps pixels"

    # Chromatic aberration displaces channels differently: ca_px=0 vs ca_px large must differ
    no_ca = core.apply_lens_geometry(img, ca_px=0.0, strength=1.0)
    big_ca = core.apply_lens_geometry(img, ca_px=3.0, strength=1.0)
    assert not torch.allclose(no_ca, big_ca, atol=1e-4), "chromatic aberration shifts channels"

    # Vignette darkens corners in linear light, not the centre; amount=0 no-op
    flat = torch.full((1, 48, 64, 3), 0.5)
    assert torch.equal(core.apply_vignette(flat, amount=0.0), flat), "vignette amount=0 no-op"
    vig = core.apply_vignette(flat, amount=0.5)
    assert vig[0, 24, 32].mean().item() > vig[0, 0, 0].mean().item() + 0.05, "vignette darkens corners"

    # Corner softness blurs the corners (variance drops), leaves the centre sharp; no-op at 0
    tex = torch.rand(1, 48, 64, 3)
    assert torch.equal(core.apply_corner_softness(tex, max_sigma=0.0), tex), "corner softness 0 no-op"
    soft = core.apply_corner_softness(tex, max_sigma=2.0)
    assert soft[0, :8, :8].var().item() < tex[0, :8, :8].var().item(), "corner softness blurs corners"
    assert torch.allclose(soft[0, 20:28, 28:36], tex[0, 20:28, 28:36], atol=1e-3), "corner softness keeps centre sharp"


def group_motion():          # Stage 3 - full OIS bit-identical (torch.equal, Bug #4);
                             #           rolling-shutter bit-identical on static (torch.equal, Bug #5b);
                             #           rolling-shutter still engages on a real pan
    img = torch.rand(6, 48, 64, 3)

    # Bug #4: full OIS correction (resid=0) passes through BIT-IDENTICAL
    assert torch.equal(core.apply_handheld_jitter(img, ois_correction=1.0), img), "full OIS bit-identical (Bug #4)"
    j1 = core.apply_handheld_jitter(img, ois_correction=0.5, seed=1)
    j2 = core.apply_handheld_jitter(img, ois_correction=0.5, seed=1)
    assert j1.shape == img.shape and not torch.equal(j1, img), "partial OIS jitters"
    assert torch.equal(j1, j2), "jitter deterministic with seed"

    # Bug #5b: rolling shutter on a STATIC clip is bit-identical (no pan -> no resample)
    static = torch.rand(1, 48, 64, 3).repeat(5, 1, 1, 1)
    assert torch.equal(core.apply_rolling_shutter(static), static), "rolling shutter static bit-identical (Bug #5b)"

    # ... but it must still engage on a real horizontal pan (a bright bar sweeping across)
    pan = torch.zeros(5, 48, 96, 3)
    for t in range(5):
        pan[t, :, 10 + t * 12: 18 + t * 12, :] = 1.0
    rs = core.apply_rolling_shutter(pan, amount=1.0)
    assert rs.shape == pan.shape and not torch.equal(rs, pan), "rolling shutter engages on a real pan"

    # Motion blur: no-op at 0; otherwise blends each frame toward its neighbours
    assert torch.equal(core.apply_motion_blur_shutter(img, amount=0.0), img), "motion blur 0 no-op"
    mb = core.apply_motion_blur_shutter(img, amount=1.0)
    assert mb.shape == img.shape and not torch.equal(mb, img), "motion blur blends frames"

def group_isp():             # Stage 4 - tone curve lifts shadows/rolls highlights, monotonic, in-range,
                             #           no-op at 0; sharpen bright-side overshoot, no-op at 0
    # --- Smart HDR curve ---
    ramp = torch.linspace(0.0, 1.0, 501).view(1, 1, 501, 1).repeat(1, 1, 1, 3)
    assert torch.equal(core.apply_smart_hdr_curve(ramp, amount=0.0), ramp), "HDR curve amount=0 no-op"

    out = core.apply_smart_hdr_curve(ramp, amount=1.0)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0, "HDR curve stays in [0,1]"
    diffs = out[0, 0, 1:, 0] - out[0, 0, :-1, 0]
    assert diffs.min().item() >= -1e-6, "HDR curve monotonic non-decreasing"

    lo = core.apply_smart_hdr_curve(torch.full((1, 1, 1, 3), 0.1), amount=1.0)
    assert lo[0, 0, 0, 0].item() > 0.1, "HDR curve lifts shadows"

    # Highlight rolloff: increments compress near white relative to the midtones
    def slope(v, d=0.01):
        hi = core.apply_smart_hdr_curve(torch.full((1, 1, 1, 3), v + d), amount=1.0)
        lo_ = core.apply_smart_hdr_curve(torch.full((1, 1, 1, 3), v - d), amount=1.0)
        return (hi[0, 0, 0, 0].item() - lo_[0, 0, 0, 0].item()) / (2 * d)
    assert slope(0.98) < slope(0.5), "HDR curve rolls off highlights (compressed slope near white)"

    # --- ISP sharpen: asymmetric bright-side overshoot ---
    edge = torch.full((1, 8, 16, 3), 0.3)
    edge[:, :, 8:] = 0.7
    assert torch.equal(core.apply_isp_sharpen(edge, amount=0.0), edge), "sharpen amount=0 no-op"
    sharp = core.apply_isp_sharpen(edge, amount=0.85, radius=0.9, overshoot=1.35)
    symmetric = core.apply_isp_sharpen(edge, amount=0.85, radius=0.9, overshoot=1.0)
    assert sharp.max().item() > symmetric.max().item(), "sharpen bright-side overshoot exceeds symmetric USM"
    assert abs(sharp.min().item() - symmetric.min().item()) < 1e-6, "sharpen dark-side left alone (overshoot doesn't touch it)"

    # --- Local tone mapping: luma-only (needs >= ~2*radius pixels for reflect-pad at sigma~18) ---
    img = torch.rand(1, 140, 140, 3)
    assert torch.equal(core.apply_local_tone_mapping(img, amount=0.0), img), "local tone map amount=0 no-op"
    ltm = core.apply_local_tone_mapping(img, amount=0.35)
    ycc_before, ycc_after = core.rgb_to_ycbcr(img), core.rgb_to_ycbcr(ltm)
    chroma_delta = (ycc_after[..., 1:3] - ycc_before[..., 1:3]).abs().mean().item()
    luma_delta = (ycc_after[..., 0] - ycc_before[..., 0]).abs().mean().item()
    # Cb/Cr are untouched by construction; any drift here is only RGB gamut-clipping fallout
    # from the luma shift, so it must stay a small fraction of the luma change itself.
    assert chroma_delta < luma_delta * 0.2, "local tone map leaves chroma essentially untouched"
    assert luma_delta > 1e-4, "local tone map changes luma"

    # --- White balance drift ---
    flat = torch.full((1, 8, 8, 3), 0.5)
    assert torch.equal(core.apply_white_balance_drift(flat, 0.0, 0.0), flat), "WB drift no-op at 0,0"
    warm = core.apply_white_balance_drift(flat, warmth=0.5, tint=0.0)
    assert warm[..., 0].mean().item() > flat[..., 0].mean().item(), "positive warmth raises red"
    assert warm[..., 2].mean().item() < flat[..., 2].mean().item(), "positive warmth lowers blue"


def group_auto_iso():        # Stage 4 - dark -> higher ISO, in range
    dark = torch.full((1, 16, 16, 3), 0.03)
    bright = torch.full((1, 16, 16, 3), 0.6)
    iso_dark = core.estimate_scene_iso(dark)
    iso_bright = core.estimate_scene_iso(bright)
    assert iso_dark > iso_bright, "dark scene -> higher ISO"
    assert 32.0 <= iso_bright <= 1600.0, "ISO in range (bright)"
    assert 32.0 <= iso_dark <= 1600.0, "ISO in range (dark)"

    black = torch.zeros(1, 16, 16, 3)
    assert core.estimate_scene_iso(black) == 1600.0, "near-black clamps to max ISO"
    white = torch.ones(1, 16, 16, 3)
    assert core.estimate_scene_iso(white) == 32.0, "bright-white clamps to min ISO"

def group_codec():           # Stage 5 - shape/dtype across odd dims; codec degrades; lower bitrate worse;
                             #           2nd generation compounds loss; rescale works
    import iphone_codec as codec

    assert codec.codec_backend() in ("pyav", "ffmpeg"), "a real H.264 backend is available on this machine"

    # Realistic COMPRESSIBLE content (gradient + a moving bar), not noise - noise is
    # incompressible and would measure ~0.177 error even at 20Mbps (spec 3 warning).
    b, h, w = 8, 96, 128
    xs = torch.linspace(0, 1, w).view(1, w, 1)
    ys = torch.linspace(0, 1, h).view(h, 1, 1)
    base = xs * 0.6 + ys * 0.4
    img = base.expand(b, h, w, 3).clone()
    for t in range(b):
        bar = 10 + t * 8
        img[t, :, bar:bar + 12, :] = 1.0

    def psnr(a, b_):
        mse = torch.mean((a - b_) ** 2).item()
        return 10.0 * math.log10(1.0 / max(mse, 1e-12))

    out = codec.h264_roundtrip(img, fps=30.0, bitrate_kbps=20000, gop=60)
    assert out.shape == img.shape and out.dtype == img.dtype, "codec preserves shape/dtype"
    p_hi = psnr(out, img)
    assert p_hi > 40.0, "near-transparent at 20Mbps on realistic content (~46.7dB expected)"

    # Lower bitrate degrades more
    out_lo = codec.h264_roundtrip(img, fps=30.0, bitrate_kbps=500, gop=60)
    p_lo = psnr(out_lo, img)
    assert p_lo < p_hi, "lower bitrate degrades more than high bitrate"

    # Second generation compounds loss
    gen2 = codec.h264_roundtrip(out_lo, fps=30.0, bitrate_kbps=500, gop=60)
    p_gen2 = psnr(gen2, img)
    assert p_gen2 < p_lo, "second H.264 generation compounds loss over the first"

    # Rescale works (platform pass downscales)
    scaled = codec.h264_roundtrip(img, fps=30.0, bitrate_kbps=3500, scale_w=64, scale_h=48)
    assert scaled.shape == (b, 48, 64, 3), "rescale produces the requested output resolution"

    # Odd dims survive: cropped to even, batch count preserved
    odd = torch.rand(5, 51, 77, 3)
    out_odd = codec.h264_roundtrip(odd, fps=24.0, bitrate_kbps=4000)
    assert out_odd.shape[0] == 5, "odd-dim roundtrip preserves frame count (pad/trim reconciliation)"
    assert out_odd.shape[1] % 2 == 0 and out_odd.shape[2] % 2 == 0, "odd H/W cropped to even for yuv420p"

    # Does not mutate the caller's input tensor
    img_before = img.clone()
    codec.h264_roundtrip(img, fps=30.0, bitrate_kbps=8000)
    assert torch.equal(img, img_before), "h264_roundtrip does not mutate its input tensor"

    assert out.min().item() >= 0.0 and out.max().item() <= 1.0, "decoded output stays in [0,1]"


ALL_GROUPS = [group_colour, group_noise_physics, group_rgb_gain, group_noise_structure,
              group_tiling, group_optics, group_motion, group_isp, group_auto_iso, group_codec]

if __name__ == "__main__":
    failed = 0
    for g in ALL_GROUPS:
        try:
            g()
        except NotImplementedError as e:
            print(f"SKIP (scaffold) {g.__name__}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {g.__name__}: {e}")
    sys.exit(1 if failed else 0)

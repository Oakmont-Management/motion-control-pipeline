"""iphone_core.py - iPhone 12 capture-chain signal processing.

PURE TORCH. No ComfyUI imports in this module (keeps it independently testable -
do not break this rule; the test suites import this file directly).

Tensor convention EVERYWHERE (spec 0):
    float32, shape (B, H, W, C), range [0, 1], RGB  ==  ComfyUI's native IMAGE.
    Convert to NCHW internally where needed, convert back before returning.
    B is the time axis for video.

Determinism (spec 0): every stochastic function takes `seed`; same seed -> bit-identical.

Build complete per the staged order (spec 11): colour helpers, sensor noise + spatial
tiling (Bugs #2, #5), optics + motion (Bugs #1, #4, #5b), ISP + auto-ISO. See
iphone_codec.py for the codec stage. tests/test_validation.py is fully green (75 assertions).
"""

import math
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 2.1 Constants (real values from spec - do NOT re-tune by eye)
# ----------------------------------------------------------------------------
IPHONE12_WIDE = {
    "focal_equiv_mm": 26.0, "f_number": 1.6, "sensor_diag_mm": 7.06,
    "pixel_pitch_um": 1.4, "readout_ms_1080p30": 16.4,
    "native_iso": 32, "max_video_iso_day": 64, "max_video_iso_night": 1600,
}
DISTORTION_K1 = 0.0125    # residual pincushion AFTER the ISP's own correction
DISTORTION_K2 = -0.0042   # NOT raw barrel distortion - you never see that in an iPhone file

# Sensor physics (spec 2.4 step 2) - linear-light RGB sigmas, intentionally tiny.
FULL_WELL_E = 6000.0      # IMX503-class 1.4um pixel
READ_NOISE_E = 2.5        # modern stacked CMOS


# ============================================================================
# 2.2 Colour space helpers  (Stage 1 - implement + unit-test FIRST)
# ============================================================================
def srgb_to_linear(x):
    """Piecewise sRGB EOTF -> linear (IEC 61966-2-1). Round-trips to identity <1e-5.

    threshold 0.04045; below: x/12.92; above: ((x+0.055)/1.055)^2.4.
    clamp the pow base at 0 so tiny negatives (from upstream noise) don't NaN.
    """
    a = 0.055
    hi = ((torch.clamp(x, min=0.0) + a) / (1.0 + a)) ** 2.4
    return torch.where(x <= 0.04045, x / 12.92, hi)


def linear_to_srgb(x):
    """Linear -> piecewise sRGB (1.055*x^(1/2.4) - 0.055). Round-trips to identity <1e-5.

    threshold 0.0031308; below: x*12.92; above: 1.055*x^(1/2.4) - 0.055.
    """
    a = 0.055
    hi = (1.0 + a) * torch.clamp(x, min=0.0) ** (1.0 / 2.4) - a
    return torch.where(x <= 0.0031308, x * 12.92, hi)


def rgb_to_ycbcr(x):
    """BT.601 full-range RGB->YCbCr on (B,H,W,C=3). Round-trips <1e-5.
    (601 deliberate - spec 2.2 note: chroma-subsampling artifacts are matrix-agnostic
    and 601 keeps luma weighting aligned with the noise model.)"""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack((y, cb, cr), dim=-1)


def ycbcr_to_rgb(x):
    """BT.601 full-range YCbCr->RGB on (B,H,W,C=3). Round-trips <1e-5."""
    y, cb, cr = x[..., 0], x[..., 1] - 0.5, x[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack((r, g, b), dim=-1)


def gaussian_blur(img, sigma):
    """Separable, reflect-padded gaussian on (B,H,W,C). No-op if sigma <= 1e-6.

    Per-axis radius is clamped to dim-1 (reflect-pad requires pad < dim) so this never
    crashes on inputs smaller than the nominal 3-sigma support - e.g. a chunked batch,
    a small preview crop, or apply_local_tone_mapping's sigma~18 on a <108px frame.
    Only pathologically small inputs are affected; the clamp is a no-op for real frames.
    """
    if sigma <= 1e-6:
        return img
    b, h, w, c = img.shape
    radius = max(1, int(math.ceil(3.0 * sigma)))

    def _kernel(r):
        coords = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
        k = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        return k / k.sum()

    r_w = min(radius, max(0, w - 1))
    r_h = min(radius, max(0, h - 1))
    x = img.permute(0, 3, 1, 2)  # B,C,H,W for conv
    if r_w > 0:
        kh = _kernel(r_w).view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        x = F.conv2d(F.pad(x, (r_w, r_w, 0, 0), mode="reflect"), kh, groups=c)
    if r_h > 0:
        kv = _kernel(r_h).view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        x = F.conv2d(F.pad(x, (0, 0, r_h, r_h), mode="reflect"), kv, groups=c)
    return x.permute(0, 2, 3, 1).contiguous()


# ============================================================================
# 2.3 Stage 1 - Optics  (Stage 3 in build order - Bug #1: align_corners=True)
# ============================================================================
def _norm_grid(h, w, device, dtype):
    """Base sampling grid on linspace(-1,1,N) (the align_corners=True convention).
    Returns gy, gx (both (h,w)) and r2 = radial distance^2 normalised so r=1 at the corner
    regardless of aspect ratio (grid_sample coords are always [-1,1] per axis -> corner at sqrt2)."""
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gy, gx, (gx * gx + gy * gy) / 2.0


def apply_lens_geometry(img, k1=DISTORTION_K1, k2=DISTORTION_K2, ca_px=0.6, strength=1.0):
    """Residual Brown-Conrady distortion + lateral chromatic aberration in ONE grid_sample.

    CONSTRAINT (Bug #1): grids built with linspace(-1,1,N) -> MUST use align_corners=True.
    strength=0 -> return input unchanged (no resample).
    """
    if strength <= 1e-6:
        return img
    b, h, w, c = img.shape
    device, dtype = img.device, img.dtype
    gy, gx, r2 = _norm_grid(h, w, device, dtype)
    warp = 1.0 + strength * (k1 * r2 + k2 * r2 * r2)             # 1 + k1 r^2 + k2 r^4

    # Lateral CA as per-channel radial scale: R and B pushed apart by ca_px at the corner, G ref.
    half_diag = math.sqrt(((w - 1) / 2.0) ** 2 + ((h - 1) / 2.0) ** 2)
    ca = strength * ca_px / max(half_diag, 1e-6)                 # ca_px -> normalised units
    scales = (1.0 + ca, 1.0, 1.0 - ca)                          # R, G, B

    # Fold the 3 channels into the batch so per-channel grids resample in a SINGLE grid_sample.
    grid = torch.stack([torch.stack((gx * warp * s, gy * warp * s), dim=-1) for s in scales], dim=0)
    grid = grid.repeat(b, 1, 1, 1)                              # (3b,h,w,2), ordered (b,c)
    src = img.permute(0, 3, 1, 2).reshape(b * c, 1, h, w)      # (b*3,1,h,w), index b*3+c
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(b, c, h, w).permute(0, 2, 3, 1).contiguous()


def apply_corner_softness(img, max_sigma=0.75):
    """Radially-weighted (r^4) blend sharp<->blurred: cheap spatially-varying PSF."""
    if max_sigma <= 1e-6:
        return img
    b, h, w, c = img.shape
    _, _, r2 = _norm_grid(h, w, img.device, img.dtype)
    wgt = (r2 * r2).clamp(0.0, 1.0).view(1, h, w, 1)            # r^4: ~0 across central ~60%
    return img * (1.0 - wgt) + gaussian_blur(img, max_sigma) * wgt


def apply_vignette(img, amount=0.18):
    """cos^4 falloff IN LINEAR LIGHT (linearise -> multiply -> re-encode). amount=0 no-op."""
    if amount <= 1e-6:
        return img
    b, h, w, c = img.shape
    _, _, r2 = _norm_grid(h, w, img.device, img.dtype)
    falloff = torch.cos(torch.sqrt(r2) * (math.pi / 2.0)).clamp(min=0.0) ** 4   # 1 centre -> 0 corner
    mask = (1.0 - amount * (1.0 - falloff)).view(1, h, w, 1)
    return torch.clamp(linear_to_srgb(srgb_to_linear(img) * mask), 0.0, 1.0)


# ============================================================================
# 2.4 Stage 2 - Sensor noise  (Stage 2 in build order - Bugs #2, #5)
# ============================================================================
def apply_sensor_noise(img, iso=64, temporal_persistence=0.0, chroma_ratio=1.9,
                       spatial_corr=0.55, fpn_amount=0.15, seed=0, strength=1.0, tile=0):
    """Memory-bounded wrapper (spec 2.5 spatial tiling) around _apply_sensor_noise_impl.

    Bug #5 (ship-blocker): tile SPATIALLY, never temporally. tile=0 -> auto-select.
    Small inputs must be bit-identical to the untiled path. strength<=0 is a bit-identical
    no-op (consistent with every other stage in this file) - without this, realism=0 would
    still pay for a full srgb->linear->ycbcr->ycbcr->srgb round trip to inject exactly zero.
    """
    if strength <= 1e-6:
        return img
    b, h, w, c = img.shape

    if tile == 0:
        # Auto-select: budget ~1.5GB working set; the impl allocates ~10.6x the input.
        px_budget = 1.5e9 / 4 / 10.6 / max(b, 1) / 3
        if h * w <= px_budget:
            tile_size = 0                         # small enough -> single pass
        else:
            side = int(px_budget ** 0.5)
            tile_size = max(192, (side // 64) * 64)  # never below 192px (PSD window)
    else:
        tile_size = int(tile)

    # Single-pass path (also the bit-identical reference for small inputs).
    if tile_size == 0 or (h <= tile_size and w <= tile_size):
        return _apply_sensor_noise_impl(img, iso, temporal_persistence, chroma_ratio,
                                        spatial_corr, fpn_amount, seed, strength)

    # Spatial tiling: each tile is a FULL-LENGTH clip of a small region (never split time).
    out = torch.empty_like(img)
    idx = 0
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            sub = img[:, y0:y1, x0:x1, :]
            out[:, y0:y1, x0:x1, :] = _apply_sensor_noise_impl(
                sub, iso, temporal_persistence, chroma_ratio,
                spatial_corr, fpn_amount, seed + idx * 7919, strength)
            idx += 1
    return out


def _ar1(n, a):
    """In-place AR(1) along the batch/time axis: n[t] = a*n[t-1] + sqrt(1-a^2)*n[t]."""
    if a <= 1e-6:
        return n
    coeff = math.sqrt(1.0 - a * a)
    for t in range(1, n.shape[0]):
        n[t] = a * n[t - 1] + coeff * n[t]
    return n


def _apply_sensor_noise_impl(img, iso, temporal_persistence, chroma_ratio,
                             spatial_corr, fpn_amount, seed, strength):
    """The real noise work (spec 2.4 steps 1-9).

    Bug #2: YCbCr->RGB gain compensation - divide sigma by rgb_gain (~1.72x) or the
    model injects 1.7-3x too much noise. Bug #3: FPN frame-invariant on all 3 channels.
    """
    b, h, w, c = img.shape
    device, dtype = img.device, img.dtype

    # 1. Work in linear light - noise happens before the transfer function.
    lin = srgb_to_linear(img)

    # 2. Noise magnitude from sensor physics (linear-light RGB sigmas, tiny by design).
    gain = iso / 32.0
    sigma_read = (READ_NOISE_E / FULL_WELL_E) * math.sqrt(gain) * strength
    sigma_shot = (1.0 / math.sqrt(FULL_WELL_E)) * math.sqrt(gain) * strength

    # 3. Signal-dependent sigma (shot noise ~ sqrt(photon count)); signal = per-pixel linear mean.
    signal = lin.mean(dim=-1)                                   # (b,h,w)
    sigma_map = torch.sqrt(sigma_read ** 2 + sigma_shot ** 2 * signal)
    del signal

    # 4. Three correlated noise fields; chroma is coarser (spatial_corr * 2.2).
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n_y = _correlated_noise(b, h, w, spatial_corr, gen, device, dtype)
    n_cb = _correlated_noise(b, h, w, spatial_corr * 2.2, gen, device, dtype)
    n_cr = _correlated_noise(b, h, w, spatial_corr * 2.2, gen, device, dtype)

    # 5. Fixed-pattern noise (Bug #3): frame-invariant, all three channels, seeded off seed^0x5EED.
    if fpn_amount > 1e-6:
        fgen = torch.Generator(device=device).manual_seed(int(seed) ^ 0x5EED)
        a = fpn_amount
        fy = torch.randn(1, h, w, generator=fgen, device=device, dtype=dtype)
        fcb = torch.randn(1, h, w, generator=fgen, device=device, dtype=dtype)
        fcr = torch.randn(1, h, w, generator=fgen, device=device, dtype=dtype)
        n_y = n_y * (1 - a) + fy * a          # fy is (1,h,w) -> frame-invariant broadcast
        n_cb = n_cb * (1 - a) + fcb * a
        n_cr = n_cr * (1 - a) + fcr * a

    # 6. Temporal correlation: AR(1) across the batch axis (time for video).
    n_y = _ar1(n_y, temporal_persistence)
    n_cb = _ar1(n_cb, temporal_persistence)
    n_cr = _ar1(n_cr, temporal_persistence)

    # 7. YCbCr->RGB gain compensation (Bug #2): variances add through the inverse matrix.
    g_r = math.sqrt(1 + (1.402 * chroma_ratio) ** 2)
    g_g = math.sqrt(1 + (0.344136 * chroma_ratio) ** 2 + (0.714136 * chroma_ratio) ** 2)
    g_b = math.sqrt(1 + (1.772 * chroma_ratio) ** 2)
    rgb_gain = math.sqrt((g_r ** 2 + g_g ** 2 + g_b ** 2) / 3)
    s = sigma_map / rgb_gain
    del sigma_map

    # 8. Inject in YCbCr, invert, re-encode, clamp.
    ycc = rgb_to_ycbcr(lin)
    del lin
    y = ycc[..., 0] + n_y * s
    cb = ycc[..., 1] + n_cb * s * chroma_ratio
    cr = ycc[..., 2] + n_cr * s * chroma_ratio
    del ycc, n_y, n_cb, n_cr, s
    rgb_lin = ycbcr_to_rgb(torch.stack((y, cb, cr), dim=-1))
    del y, cb, cr
    return torch.clamp(linear_to_srgb(rgb_lin), 0.0, 1.0)


def _correlated_noise(b, h, w, corr, generator, device, dtype):
    """White randn -> rFFT * radial low-pass 1/(1+(f*corr*12)^2) -> irFFT -> unit variance.
    corr <= 1e-6 -> return white noise. (spec 2.4.1)"""
    white = torch.randn(b, h, w, generator=generator, device=device, dtype=dtype)
    if corr <= 1e-6:
        return white
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(h, 1)
    fx = torch.fft.rfftfreq(w, device=device, dtype=dtype).view(1, w // 2 + 1)
    f = torch.sqrt(fy * fy + fx * fx)                      # radial frequency, (h, w//2+1)
    psd = 1.0 / (1.0 + (f * corr * 12.0) ** 2)             # demosaic-like low-pass
    spec = torch.fft.rfft2(white) * psd                    # broadcast over batch
    out = torch.fft.irfft2(spec, s=(h, w))
    return out / (out.std(dim=(-2, -1), keepdim=True) + 1e-12)   # per-frame unit variance


def apply_chroma_smear(img, amount=0.0):
    """Blur Cb/Cr only (sigma 0.5 + amount*3.5), leave Y intact. Low-light chroma-NR tell."""
    if amount <= 1e-6:
        return img
    sigma = 0.5 + amount * 3.5
    ycc = rgb_to_ycbcr(img)
    chroma = gaussian_blur(ycc[..., 1:3], sigma)          # blur Cb/Cr only
    ycc = torch.cat((ycc[..., 0:1], chroma), dim=-1)
    return torch.clamp(ycbcr_to_rgb(ycc), 0.0, 1.0)


def apply_temporal_nr_ghosting(img, amount=0.0):
    """Small asymmetric leak from previous frame (a = amount*0.35). Motion trails."""
    if amount <= 1e-6:
        return img
    a = amount * 0.35
    out = img.clone()
    for t in range(1, out.shape[0]):
        out[t] = (1 - a) * img[t] + a * out[t - 1]        # IIR leak from the past only
    return out


# ============================================================================
# 2.6 Stage 3 - ISP  (Stage 4 in build order) - operate in GAMMA space
# ============================================================================
def apply_smart_hdr_curve(img, amount=1.0):
    """THE biggest 'looks like an iPhone' tell. Highlight rolloff + shadow lift + midtone S.
    MUST be monotonic non-decreasing, stay in [0,1], no-op at amount=0. (spec 2.6)

    hl = asymptotic highlight rolloff, sh = shadow-lift power curve, blended by x itself
    so shadows lean on sh and highlights lean on hl; a small smoothstep nudge adds the
    "gentle midtone S". The x-weighted blend isn't a fixed convex combination (the
    weights vary with x), so monotonicity here isn't a one-line algebraic proof - it's
    verified numerically (test_validation.py group_isp checks a 501-point ramp for any
    negative step) rather than assumed.
    """
    if amount <= 1e-6:
        return img
    x = torch.clamp(img, 0.0, 1.0)
    hl = (1.0 - torch.exp(-x * 1.55)) / (1.0 - math.exp(-1.55))
    sh = x ** 0.88
    curve = (1.0 - x) * sh + x * hl               # shadow-lift -> highlight-rolloff blend
    smooth = 3.0 * curve ** 2 - 2.0 * curve ** 3   # smoothstep, monotonic on [0,1]
    s = 0.85 * curve + 0.15 * smooth               # gentle midtone S
    out = (1.0 - amount) * x + amount * s
    return torch.clamp(out, 0.0, 1.0)


def apply_isp_sharpen(img, amount=0.85, radius=0.9, overshoot=1.35):
    """Unsharp mask with ASYMMETRIC overshoot: light side amplified by `overshoot`,
    dark side left alone. The asymmetric bright fringe is the real signature."""
    if amount <= 1e-6:
        return img
    blur = gaussian_blur(img, radius)
    detail = img - blur
    pos = torch.clamp(detail, min=0.0) * overshoot   # bright-side fringe amplified
    neg = torch.clamp(detail, max=0.0)                # dark side untouched (no extra gain)
    out = img + amount * (pos + neg)
    return torch.clamp(out, 0.0, 1.0)


def apply_local_tone_mapping(img, amount=0.35):
    """Large-radius (sigma ~18) unsharp on LUMA ONLY. Luma-only or it oversaturates."""
    if amount <= 1e-6:
        return img
    ycc = rgb_to_ycbcr(img)
    y = ycc[..., 0:1]
    blur = gaussian_blur(y, 18.0)
    y2 = torch.clamp(y + amount * (y - blur), 0.0, 1.0)
    ycc = torch.cat((y2, ycc[..., 1:3]), dim=-1)
    return torch.clamp(ycbcr_to_rgb(ycc), 0.0, 1.0)


def apply_white_balance_drift(img, warmth=0.0, tint=0.0):
    """Per-channel gain in linear light: [1+warmth*0.06, 1+tint*0.03, 1-warmth*0.05]."""
    if abs(warmth) <= 1e-6 and abs(tint) <= 1e-6:
        return img
    gains = torch.tensor(
        [1.0 + warmth * 0.06, 1.0 + tint * 0.03, 1.0 - warmth * 0.05],
        device=img.device, dtype=img.dtype,
    )
    lin = srgb_to_linear(img) * gains
    return torch.clamp(linear_to_srgb(lin), 0.0, 1.0)


# ============================================================================
# 2.7 Stage 4 - Motion  (Stage 3 in build order - Bugs #4, #5b)
# ============================================================================
def _ou_process(n, theta, sigma, generator, device, dtype):
    """Ornstein-Uhlenbeck mean-reverting process (handheld motion is mean-reverting)."""
    eps = torch.randn(n, generator=generator, device=device, dtype=dtype)
    x = torch.zeros(n, device=device, dtype=dtype)
    for t in range(1, n):
        x[t] = x[t - 1] - theta * x[t - 1] + sigma * eps[t]   # pull toward mean + kick
    return x


def apply_handheld_jitter(img, amount=1.0, ois_correction=0.72, seed=0):
    """OU sway + high-freq tremor + slow OU roll. resid = (1-ois_correction)*amount.
    CONSTRAINT (Bug #4): resid <= 1e-6 -> early-return input BIT-IDENTICAL (torch.equal)."""
    resid = (1.0 - ois_correction) * amount
    if resid <= 1e-6:
        return img                                             # OIS 1.0 == tripod, no resample
    b, h, w, c = img.shape
    device, dtype = img.device, img.dtype
    g = torch.Generator(device=device).manual_seed(int(seed))
    tremor = lambda: torch.randn(b, generator=g, device=device, dtype=dtype) * 0.0004
    dx = (_ou_process(b, 0.08, 0.0022, g, device, dtype) + tremor()) * resid
    dy = (_ou_process(b, 0.08, 0.0022, g, device, dtype) + tremor()) * resid
    roll = _ou_process(b, 0.05, 0.0009, g, device, dtype) * resid

    gy, gx, _ = _norm_grid(h, w, device, dtype)
    cos, sin = torch.cos(roll).view(b, 1, 1), torch.sin(roll).view(b, 1, 1)
    gx_, gy_ = gx.view(1, h, w), gy.view(1, h, w)
    xr = cos * gx_ - sin * gy_ + dx.view(b, 1, 1)              # rotate (roll) then translate (sway)
    yr = sin * gx_ + cos * gy_ + dy.view(b, 1, 1)
    grid = torch.stack((xr, yr), dim=-1)
    out = F.grid_sample(img.permute(0, 3, 1, 2), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.permute(0, 2, 3, 1).contiguous()


def apply_rolling_shutter(img, amount=1.0, readout_ms=16.4, fps=30.0):
    """Skew rows by estimated per-frame horizontal pan velocity.
    CONSTRAINT (Bug #5b): max|vel| < 1e-6 -> early-return BIT-IDENTICAL. Must still engage on a real pan."""
    if amount <= 1e-6:
        return img
    b, h, w, c = img.shape
    device, dtype = img.device, img.dtype

    # Estimate horizontal pan velocity from the luma column-centroid shift between frames.
    luma = rgb_to_ycbcr(img)[..., 0]                           # (b,h,w)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    col = luma.sum(dim=1)                                       # (b,w) column energy
    centroid = (col * xs.view(1, w)).sum(dim=1, keepdim=True) / (col.sum(dim=1, keepdim=True) + 1e-12)
    vel = torch.zeros(b, 1, device=device, dtype=dtype)
    vel[1:] = (centroid[1:] - centroid[:-1]) * amount          # per-frame pan
    if vel.abs().max().item() < 1e-6:
        return img                                             # locked-off shot: no resample

    frame_ms = 1000.0 / fps
    gy, gx, _ = _norm_grid(h, w, device, dtype)
    row_delay = (gy + 1.0) / 2.0 * min(readout_ms / frame_ms, 1.0)    # 0 at top row -> full at bottom
    xr = gx.view(1, h, w) + vel.view(b, 1, 1) * row_delay.view(1, h, w)
    yr = gy.view(1, h, w).expand(b, h, w)
    grid = torch.stack((xr, yr), dim=-1)
    out = F.grid_sample(img.permute(0, 3, 1, 2), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.permute(0, 2, 3, 1).contiguous()


def apply_motion_blur_shutter(img, amount=0.0):
    """Blend each frame toward neighbours (a = amount*0.25). Finite shutter angle.

    No-op below 3 frames even if amount>0 - there's no interior frame to blend two
    neighbours into. Not a real-world limit (video batches are always well over 3).
    """
    if amount <= 1e-6 or img.shape[0] < 3:
        return img
    a = amount * 0.25
    out = img.clone()
    out[1:-1] = (1.0 - a) * img[1:-1] + (a / 2.0) * img[:-2] + (a / 2.0) * img[2:]
    return out


# ============================================================================
# 2.8 Auto scene analysis
# ============================================================================
def estimate_scene_iso(img):
    """Inverse-AE from linear luminance: iso = 32*(0.18/mean_lum)^1.15, clamp [32, 1600].
    Dark scene (low mean_lum) -> higher ISO."""
    lin = srgb_to_linear(img)
    mean_lum = torch.clamp(lin.mean(), min=1e-6)
    iso = 32.0 * (0.18 / mean_lum) ** 1.15
    return float(torch.clamp(iso, 32.0, 1600.0))

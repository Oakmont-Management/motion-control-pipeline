"""iphone_codec.py - in-memory H.264 round-trips (spec 3).

Real libx264 encode+decode, TWICE - actual generation loss, not a DCT approximation.
This is the single biggest realism contributor (spec 1). Real H.264 artifacts come from
motion compensation, in-loop deblocking, rate-control panic, GOP structure and B-frame
prediction; none of that is fakeable, so we run the real encoder.

No temp files - everything through in-memory BytesIO (PyAV) or piped stdin/stdout (ffmpeg).
"""

import io
import shutil
import subprocess

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Backend detection
# ----------------------------------------------------------------------------
def _pyav_available():
    try:
        import av
        av.codec.Codec("libx264", "w")
        return True
    except Exception:
        return False


def _find_ffmpeg():
    """Locate an ffmpeg binary. Order (machine-adapted, see plan):
      1. VideoHelperSuite bundled path  (absent on this machine)
      2. shutil.which('ffmpeg')          (not on PATH on this machine)
      3. imageio_ffmpeg.get_ffmpeg_exe() (present in the ComfyUI venv - the working fallback)
    Returns a path string or None. PyAV is preferred over any of these anyway.
    """
    try:
        from videohelpersuite import ffmpeg_path  # spec's preferred VHS-bundled path
        if ffmpeg_path:
            return ffmpeg_path
    except Exception:
        pass

    which = shutil.which("ffmpeg")
    if which:
        return which

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def codec_backend():
    """Return 'pyav' | 'ffmpeg' | 'none'. Prefer PyAV (in-memory BytesIO)."""
    if _pyav_available():
        return "pyav"
    if _find_ffmpeg():
        return "ffmpeg"
    return "none"


# ----------------------------------------------------------------------------
# Frame prep / reconciliation (shared by both backends)
# ----------------------------------------------------------------------------
def _even(n):
    return n - (n % 2)


def _to_uint8_frames(img):
    """(B,H,W,C) float [0,1], any device/dtype -> (B,H,W,3) uint8 numpy, contiguous."""
    x = torch.clamp(img, 0.0, 1.0).detach().to("cpu", torch.float32)
    x = (x * 255.0 + 0.5).floor().clamp(0, 255).to(torch.uint8)
    return np.ascontiguousarray(x.numpy())


def _resize_frames(frames, h, w):
    """(B,H0,W0,3) uint8 -> (B,h,w,3) uint8, bilinear."""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    t = t.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return np.ascontiguousarray(t.numpy())


def _prep_frames(img, scale_w, scale_h):
    """Crop to even dims (yuv420p requires it); optionally resize to an even (scale_w,scale_h)."""
    frames = _to_uint8_frames(img)
    b, h0, w0, _ = frames.shape
    h, w = _even(h0), _even(w0)
    frames = frames[:, :h, :w, :]
    if scale_w and scale_h:
        sh, sw = _even(int(round(scale_h))), _even(int(round(scale_w)))
        if (sh, sw) != (h, w):
            frames = _resize_frames(frames, sh, sw)
        h, w = sh, sw
    return frames, h, w


def _reconcile_count(frames, b_in, fallback):
    """Pad (repeat last frame) or trim to match the input batch size exactly.

    If the decoder returned nothing at all (total decode failure), there's no frame to
    repeat - fall back to the pre-encode source frames so the caller gets `b_in` valid
    frames back instead of np.stack crashing on an empty list.
    """
    frames = list(frames) if frames else list(fallback)
    if len(frames) > b_in:
        return frames[:b_in]
    while len(frames) < b_in and frames:
        frames.append(frames[-1].copy())
    return frames


def _rc_options(bitrate_kbps, crf, preset, profile):
    """Shared x264 option set. Deblock loop filter ON (+loop) - why compressed video
    looks soft, not blocky. Constrained VBR when using bitrate: fixed maxrate/bufsize
    makes rate control panic on motion, which is the artifact we want."""
    opts = {"preset": preset, "profile": profile, "flags": "+loop"}
    if crf is not None:
        opts["crf"] = str(crf)
    return opts


# ----------------------------------------------------------------------------
# PyAV backend
# ----------------------------------------------------------------------------
def _encode_pyav(frames, fps, w, h, bitrate_kbps, crf, gop, preset, profile, bframes):
    import av

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    cc = stream.codec_context
    cc.gop_size = gop
    cc.max_b_frames = bframes

    opts = _rc_options(bitrate_kbps, crf, preset, profile)
    if crf is None:
        bps = int(bitrate_kbps * 1000)
        cc.bit_rate = bps
        opts["maxrate"] = str(int(1.45 * bps))
        opts["bufsize"] = str(int(2 * bps))
    cc.options = opts

    for f in frames:
        vf = av.VideoFrame.from_ndarray(f, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)
    for packet in stream.encode(None):    # flush
        container.mux(packet)
    container.close()
    return buf.getvalue()


def _decode_pyav(data):
    import av

    buf = io.BytesIO(data)
    container = av.open(buf, mode="r")
    out = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return out


# ----------------------------------------------------------------------------
# ffmpeg subprocess fallback (raw video piped through stdin/stdout, no temp files)
# ----------------------------------------------------------------------------
def _encode_ffmpeg(ffmpeg, frames, fps, w, h, bitrate_kbps, crf, gop, preset, profile, bframes):
    opts = _rc_options(bitrate_kbps, crf, preset, profile)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", opts["preset"], "-profile:v", opts["profile"],
        "-bf", str(bframes), "-g", str(gop), "-flags", "+loop",
    ]
    if crf is not None:
        cmd += ["-crf", opts["crf"]]
    else:
        bps = int(bitrate_kbps * 1000)
        cmd += ["-b:v", str(bps), "-maxrate", str(int(1.45 * bps)), "-bufsize", str(int(2 * bps))]
    cmd += ["-f", "matroska", "-"]

    raw = frames.tobytes()
    proc = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode(errors='replace')[-2000:]}")
    return proc.stdout


def _decode_ffmpeg(ffmpeg, data, w, h):
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[-2000:]}")
    raw = proc.stdout
    frame_bytes = h * w * 3
    n = len(raw) // frame_bytes
    arr = np.frombuffer(raw[:n * frame_bytes], dtype=np.uint8).reshape(n, h, w, 3)
    return [arr[i] for i in range(n)]


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def h264_roundtrip(img, fps=30.0, bitrate_kbps=17000, crf=None, gop=60,
                   preset="medium", profile="high", bframes=2,
                   scale_w=None, scale_h=None):
    """Encode img (B,H,W,C float [0,1] RGB) with libx264 then decode back to the same
    tensor convention on the input's device/dtype.

    - yuv420p requires EVEN dims: crop source, round output dims to even.
    - Deblocking filter ON (flags=+loop) - why compressed video looks soft not blocky.
    - Constrained VBR when using bitrate: maxrate = 1.45*bitrate, bufsize = 2*bitrate.
    - Reconciles frame count defensively (pad/trim to input B).
    - PyAV failure -> fall back to ffmpeg silently. backend 'none' -> raise 'pip install av'.
    """
    device, dtype = img.device, img.dtype
    b_in = img.shape[0]
    frames, h, w = _prep_frames(img, scale_w, scale_h)

    out_frames = None
    if _pyav_available():
        try:
            data = _encode_pyav(frames, fps, w, h, bitrate_kbps, crf, gop, preset, profile, bframes)
            out_frames = _decode_pyav(data)
        except Exception:
            out_frames = None   # fall through to ffmpeg rather than failing the queue

    if out_frames is None:
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError(
                "iphone12-sim: no H.264 backend available - pip install av "
                "(or ensure ffmpeg is reachable)"
            )
        data = _encode_ffmpeg(ffmpeg, frames, fps, w, h, bitrate_kbps, crf, gop, preset, profile, bframes)
        out_frames = _decode_ffmpeg(ffmpeg, data, w, h)

    out_frames = _reconcile_count(out_frames, b_in, frames)
    out = np.stack(out_frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(out).to(device=device, dtype=dtype)

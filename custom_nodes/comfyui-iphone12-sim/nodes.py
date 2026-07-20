"""nodes.py - ComfyUI node definitions (spec 5).

Three nodes, CATEGORY = "iPhone12Sim":
  IPhone12CaptureSim   - the whole stack in one node, locked preset
  IPhone12PreflightCheck - validates fps/res/codec-backend/scene-ISO before a long render
  IPhone12PresetInfo   - dumps a preset's values and notes
"""

import os
import json
import glob

import torch

from . import iphone_core as core
from . import iphone_codec as codec

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")

# Chunk auto-sizing (spec 5: chunk_frames=0 -> auto by pixel budget). Spatial stages
# (optics, ISP) are strictly per-frame - no cross-frame coupling - so splitting the
# batch at ANY chunk size is bit-identical to not chunking; this budget only bounds
# peak memory, it never changes the result.
_AUTO_CHUNK_PX_BUDGET = 64 * 1024 * 1024


def _preset_names():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(PRESET_DIR, "*.json")))


def load_preset(name):
    with open(os.path.join(PRESET_DIR, name + ".json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _auto_chunk_frames(h, w):
    return max(1, _AUTO_CHUNK_PX_BUDGET // max(h * w, 1))


def _chunked_spatial(x, chunk_frames, fn):
    """Run a per-frame fn over x, splitting the batch into chunks. Bit-identical to
    fn(x) regardless of chunk size (spatial stages don't couple across frames)."""
    b, h, w, _ = x.shape
    cf = chunk_frames if chunk_frames and chunk_frames > 0 else _auto_chunk_frames(h, w)
    cf = max(1, min(int(cf), b))
    if cf >= b:
        return fn(x)
    return torch.cat([fn(x[i:i + cf]) for i in range(0, b, cf)], dim=0)


def _resolve_preset(preset, images):
    if preset == "auto":
        iso = core.estimate_scene_iso(images)
        return "iphone12_night" if iso >= 400.0 else "iphone12_day"
    return preset


class IPhone12CaptureSim:
    """Full optics -> sensor -> ISP -> motion -> codec gen1 -> codec gen2 stack."""
    CATEGORY = "iPhone12Sim"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (["auto"] + _preset_names(),),
                "fps": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "realism": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.01}),
                "enable_codec": ("BOOLEAN", {"default": True}),
                "enable_platform_pass": ("BOOLEAN", {"default": True}),
                "chunk_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
        }

    def run(self, images, preset, fps, seed, realism=1.0, enable_codec=True,
            enable_platform_pass=True, chunk_frames=0):
        try:
            return self._run(images, preset, fps, seed, realism, enable_codec,
                             enable_platform_pass, chunk_frames)
        except Exception as e:
            print(f"[iphone12-sim] IPhone12CaptureSim FAILED ({e!r}) - "
                  f"passing {images.shape[0]} frames through unmodified.")
            return (images,)

    def _run(self, images, preset, fps, seed, realism, enable_codec,
             enable_platform_pass, chunk_frames):
        if fps < 24.0:
            print(f"[iphone12-sim] WARNING: fps={fps} < 24 - iPhone 12 never shot below 24; "
                  f"the cadence tells regardless of everything else.")

        preset_name = _resolve_preset(preset, images)
        p = load_preset(preset_name)   # bad preset name -> FileNotFoundError -> caught by run()
        opt, sens, isp, mot, cod = p["optics"], p["sensor"], p["isp"], p["motion"], p["codec"]
        r = float(realism)

        x = images

        def optics_stage(chunk):
            chunk = core.apply_lens_geometry(chunk, k1=opt["k1"], k2=opt["k2"],
                                             ca_px=opt["ca_px"], strength=opt["strength"] * r)
            chunk = core.apply_corner_softness(chunk, max_sigma=opt["corner_softness_max_sigma"] * r)
            chunk = core.apply_vignette(chunk, amount=opt["vignette_amount"] * r)
            return chunk
        x = _chunked_spatial(x, chunk_frames, optics_stage)

        # Sensor + motion are temporal stages (spec 5): always run on the full sequence,
        # never chunked - chunking them would seam the output at chunk boundaries.
        x = core.apply_sensor_noise(x, iso=sens["iso"], temporal_persistence=sens["temporal_persistence"],
                                    chroma_ratio=sens["chroma_ratio"], spatial_corr=sens["spatial_corr"],
                                    fpn_amount=sens["fpn_amount"], seed=seed, strength=sens["strength"] * r)
        x = core.apply_chroma_smear(x, sens["chroma_smear"] * r)
        x = core.apply_temporal_nr_ghosting(x, sens["temporal_nr_ghosting"] * r)

        def isp_stage(chunk):
            chunk = core.apply_smart_hdr_curve(chunk, amount=isp["smart_hdr"] * r)
            chunk = core.apply_isp_sharpen(chunk, amount=isp["sharpen"] * r,
                                           radius=isp["sharpen_radius"], overshoot=isp["sharpen_overshoot"])
            chunk = core.apply_local_tone_mapping(chunk, amount=isp["local_tone_mapping"] * r)
            chunk = core.apply_white_balance_drift(chunk, warmth=isp["warmth"] * r, tint=isp["tint"] * r)
            return chunk
        x = _chunked_spatial(x, chunk_frames, isp_stage)

        x = core.apply_handheld_jitter(x, amount=mot["jitter_amount"] * r,
                                       ois_correction=mot["ois_correction"], seed=seed)
        x = core.apply_rolling_shutter(x, amount=mot["rolling_shutter"] * r,
                                       readout_ms=mot["readout_ms"], fps=fps)
        x = core.apply_motion_blur_shutter(x, amount=mot["motion_blur"] * r)

        if enable_codec:
            backend = codec.codec_backend()
            if backend == "none":
                print("[iphone12-sim] WARNING: no H.264 backend (pip install av) - skipping codec stage.")
            else:
                x = codec.h264_roundtrip(x, fps=fps, bitrate_kbps=cod["gen1_bitrate_kbps"],
                                         crf=cod["crf"], gop=cod["gop"], preset=cod["preset"],
                                         profile=cod["profile"], bframes=cod["bframes"])
                if enable_platform_pass:
                    h, w = x.shape[1], x.shape[2]
                    max_w = cod["platform_max_width"]
                    scale_w, scale_h = (None, None)
                    if w > max_w:
                        scale_w = max_w
                        scale_h = int(round(h * (max_w / w)))
                    x = codec.h264_roundtrip(x, fps=fps, bitrate_kbps=cod["platform_bitrate_kbps"],
                                             crf=cod["crf"], gop=cod["gop"], preset=cod["preset"],
                                             profile=cod["profile"], bframes=cod["bframes"],
                                             scale_w=scale_w, scale_h=scale_h)

        return (x,)


class IPhone12PreflightCheck:
    """Validate fps/resolution/codec-backend/scene-ISO; passes images through + STRING report."""
    CATEGORY = "iPhone12Sim"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",),
                             "fps": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 120.0})}}

    def run(self, images, fps):
        b, h, w, c = images.shape
        backend = codec.codec_backend()
        iso = core.estimate_scene_iso(images)
        lines = [
            f"frames: {b}  resolution: {w}x{h}  fps: {fps}",
            f"codec backend: {backend}" + ("  !! none - pip install av" if backend == "none" else ""),
            f"estimated scene ISO: {iso:.0f}" +
            ("  (low light -> auto picks night)" if iso >= 400.0 else "  (bright -> auto picks day)"),
        ]
        if fps < 24.0:
            lines.append(f"WARNING: fps={fps} < 24 - iPhone 12 never shot below 24; "
                        f"the cadence tells regardless of everything else.")
        report = "\n".join(lines)
        print(f"[iphone12-sim] preflight:\n{report}")
        return (images, report)


class IPhone12PresetInfo:
    """Dump a preset's values and notes as a STRING."""
    CATEGORY = "iPhone12Sim"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"preset": (_preset_names(),)}}

    def run(self, preset):
        p = load_preset(preset)
        return (json.dumps(p, indent=2),)


NODE_CLASS_MAPPINGS = {
    "IPhone12CaptureSim": IPhone12CaptureSim,
    "IPhone12PreflightCheck": IPhone12PreflightCheck,
    "IPhone12PresetInfo": IPhone12PresetInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IPhone12CaptureSim": "iPhone 12 Capture Sim",
    "IPhone12PreflightCheck": "iPhone 12 Preflight Check",
    "IPhone12PresetInfo": "iPhone 12 Preset Info",
}

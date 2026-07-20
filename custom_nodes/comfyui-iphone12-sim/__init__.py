"""comfyui-iphone12-sim - make Wan 2.2 Animate -> SeedVR2 video look like iPhone 12 footage.

Exposes NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS and prints a load banner that
reports the detected codec backend (warns if 'none'). Do NOT set WEB_DIRECTORY - no JS shipped
(pointing at an empty dir makes ComfyUI log a spurious warning every start, spec 5).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from .iphone_codec import codec_backend
    _backend = codec_backend()
except Exception:
    _backend = "unknown (scaffold)"

_warn = "  !! no H.264 backend - `pip install av` for the codec stage" if _backend == "none" else ""
print(f"[iphone12-sim] loaded {len(NODE_CLASS_MAPPINGS)} nodes | codec backend: {_backend}{_warn}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

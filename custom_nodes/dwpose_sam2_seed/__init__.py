"""
DWPose -> SAM2 seed points (Auto)

Converts the DWPose POSE_KEYPOINT that the Wan-Animate graph already computes
into a SAM2 `coordinates_positive` STRING, so the subject mask is seeded
automatically from the detected body instead of a hand-placed PointsEditor click.

Output format matches what Sam2Segmentation expects:  '[{"x": int, "y": int}, ...]'
(see ComfyUI-segment-anything-2/nodes.py: coordinates parsed as [(c['x'], c['y']) ...]).

No model downloads, negligible VRAM -- it only reads keypoints already in the graph.
"""

import json
import numpy as np

# OpenPose (COCO-18) body keypoint indices used as anchors.
BODY = {"nose": 0, "neck": 1, "rsho": 2, "lsho": 5, "rhip": 8, "lhip": 11}


def _kps_array(flat):
    """Flat [x,y,c, x,y,c, ...] -> np.array [N,3]. Returns empty array if unusable."""
    if flat is None or len(flat) < 3:
        return np.zeros((0, 3), dtype=np.float32)
    a = np.array(flat, dtype=np.float32)
    a = a[: (len(a) // 3) * 3].reshape(-1, 3)
    return a


def _valid(a):
    return a[a[:, 2] > 0.0] if len(a) else a


class DWPoseToSam2Points:
    """Auto-derive SAM2 positive seed point(s) from DWPose keypoints."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
                "width": ("INT", {"default": 576, "min": 16, "max": 8192,
                                  "tooltip": "Target SAM2 image width (wire to your output Width)"}),
                "height": ("INT", {"default": 768, "min": 16, "max": 8192,
                                   "tooltip": "Target SAM2 image height (wire to your output Height)"}),
                "person_select": (["largest", "index"], {"default": "largest"}),
                "person_index": ("INT", {"default": 0, "min": 0, "max": 50}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coordinates_positive",)
    FUNCTION = "convert"
    CATEGORY = "WanAnimate/Auto"

    def convert(self, pose_kps, width, height, person_select="largest", person_index=0):
        frames = pose_kps if isinstance(pose_kps, list) else [pose_kps]

        def people_of(fr):
            return fr.get("people", []) if isinstance(fr, dict) else []

        # SAM2 (video) is seeded on the first frame: pick the first frame that has a person.
        seed_frame = next((fr for fr in frames if people_of(fr)), None)
        if seed_frame is None:
            print("[DWPoseToSam2Points] WARNING: no people detected in pose; "
                  "using center fallback point (check your driving video / DWPose).")
            return (json.dumps([{"x": int(width // 2), "y": int(height // 2)}]),)

        cw = float(seed_frame.get("canvas_width", width)) or float(width)
        ch = float(seed_frame.get("canvas_height", height)) or float(height)
        people = people_of(seed_frame)

        def body_of(p):
            return _kps_array(p.get("pose_keypoints_2d", []))

        # Choose which person to mask.
        if person_select == "index":
            idx = person_index if person_index < len(people) else 0
        else:
            idx, best_area = 0, -1.0
            for i, p in enumerate(people):
                v = _valid(body_of(p))
                if not len(v):
                    continue
                area = (v[:, 0].max() - v[:, 0].min()) * (v[:, 1].max() - v[:, 1].min())
                if area > best_area:
                    best_area, idx = area, i

        person = people[idx]
        a = body_of(person)
        v = _valid(a)
        normalized = len(v) > 0 and bool(np.all(np.abs(v[:, :2]) <= 1.5))

        def to_target(x, y):
            nx, ny = (x, y) if normalized else (x / cw, y / ch)
            return (int(round(float(np.clip(nx, 0.0, 1.0)) * width)),
                    int(round(float(np.clip(ny, 0.0, 1.0)) * height)))

        def anchor(*names):
            pts = [a[BODY[nm], :2] for nm in names
                   if BODY[nm] < len(a) and a[BODY[nm], 2] > 0]
            if not pts:
                return None
            m = np.mean(np.stack(pts, 0), axis=0)
            return to_target(float(m[0]), float(m[1]))

        # Seed a few points spread down the torso so SAM2 grabs the whole figure.
        coords = []
        for spec in (("neck", "rsho", "lsho"),
                     ("rsho", "lsho", "rhip", "lhip"),
                     ("rhip", "lhip")):
            pt = anchor(*spec)
            if pt is not None:
                coords.append({"x": pt[0], "y": pt[1]})

        # Drop near-duplicate points.
        uniq = []
        for c in coords:
            if all(abs(c["x"] - u["x"]) > 4 or abs(c["y"] - u["y"]) > 4 for u in uniq):
                uniq.append(c)
        coords = uniq

        # Fallbacks if the body anchors were missing.
        if not coords:
            if len(v):
                m = np.mean(v[:, :2], axis=0)
                p = to_target(float(m[0]), float(m[1]))
                coords = [{"x": p[0], "y": p[1]}]
            else:
                fv = _valid(_kps_array(person.get("face_keypoints_2d", [])))
                if len(fv):
                    m = np.mean(fv[:, :2], axis=0)
                    p = to_target(float(m[0]), float(m[1]))
                    coords = [{"x": p[0], "y": p[1]}]
                else:
                    print("[DWPoseToSam2Points] WARNING: chosen person had no valid keypoints; center fallback.")
                    coords = [{"x": int(width // 2), "y": int(height // 2)}]

        print(f"[DWPoseToSam2Points] person {idx + 1}/{len(people)} -> {len(coords)} seed point(s): "
              f"{coords}  (canvas {int(cw)}x{int(ch)} -> target {width}x{height}, normalized={normalized})")
        return (json.dumps(coords),)


NODE_CLASS_MAPPINGS = {"DWPoseToSam2Points": DWPoseToSam2Points}
NODE_DISPLAY_NAME_MAPPINGS = {"DWPoseToSam2Points": "DWPose → SAM2 Seed Points (Auto)"}

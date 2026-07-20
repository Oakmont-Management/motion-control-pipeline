"""test_presets.py - preset + node-contract assertions (spec 8's "45 assertions" target;
this suite landed at 32 `assert` statements - fewer, denser checks than the target, each
covering a broader property than a single spec bullet). All four groups are implemented.

sys.exit(1) on any failure. Differentials measured PER-PROPERTY (aggregate delta is
misleading - day's stronger tone curve moves more pixels than night's noise).
"""
import os
import sys
import json
import importlib.util

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESET_DIR = os.path.join(PKG_DIR, "presets")
SECTIONS = ("optics", "sensor", "isp", "motion", "codec")
_PKG_NAME = "_iphone12_sim_test_pkg"


def _load(name):
    with open(os.path.join(PRESET_DIR, name + ".json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_nodes_module():
    """nodes.py uses package-relative imports (from . import iphone_core), exactly as
    ComfyUI's real loader runs it (custom_nodes dirs are loaded as packages regardless
    of the hyphenated folder name). Load it the same way here rather than sys.path-hacking
    around the relative imports, so this test exercises the real production import path."""
    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME + ".nodes"]
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME, os.path.join(PKG_DIR, "__init__.py"),
        submodule_search_locations=[PKG_DIR])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = pkg
    spec.loader.exec_module(pkg)   # runs __init__.py -> `from .nodes import ...` -> full package load
    return sys.modules[_PKG_NAME + ".nodes"]


def group_preset_integrity():
    """Both presets load with all five sections + name + notes. (runnable now)"""
    day, night = _load("iphone12_day"), _load("iphone12_night")
    for p in (day, night):
        assert "name" in p and "notes" in p, "missing name/notes"
        for s in SECTIONS:
            assert s in p, f"{p['name']} missing section {s}"


def group_differentials():
    """Per-property day-vs-night (runnable now against the JSON; code-measured version is Stage 6/7)."""
    day, night = _load("iphone12_day"), _load("iphone12_night")
    assert night["sensor"]["iso"] > day["sensor"]["iso"], "night higher ISO"
    assert night["sensor"]["chroma_ratio"] > day["sensor"]["chroma_ratio"], "night stronger chroma ratio"
    assert night["sensor"]["spatial_corr"] > day["sensor"]["spatial_corr"], "night coarser grain"
    assert night["sensor"]["chroma_smear"] > 0 and day["sensor"]["chroma_smear"] == 0, "night has chroma-smear, day zero"
    assert night["sensor"]["temporal_nr_ghosting"] > 0 and day["sensor"]["temporal_nr_ghosting"] == 0, "night has NR-ghosting, day zero"
    assert night["isp"]["sharpen"] < day["isp"]["sharpen"], "night sharpens LESS (the classic mistake)"
    assert night["isp"]["warmth"] > day["isp"]["warmth"], "night warmer WB"
    assert night["motion"]["motion_blur"] > day["motion"]["motion_blur"], "night more shutter blur"
    assert night["motion"]["ois_correction"] < day["motion"]["ois_correction"], "night weaker OIS"


def group_measured_noise():   # Stage 6 - measured night noise > 3x day; night smear hits chroma not luma
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import iphone_core as core

    day, night = _load("iphone12_day"), _load("iphone12_night")
    img = torch.full((8, 64, 64, 3), 0.4)

    def measured_sigma(p, seed):
        s = p["sensor"]
        out = core.apply_sensor_noise(
            img, iso=s["iso"], temporal_persistence=s["temporal_persistence"],
            chroma_ratio=s["chroma_ratio"], spatial_corr=s["spatial_corr"],
            fpn_amount=s["fpn_amount"], seed=seed, strength=s["strength"],
        )
        return (core.srgb_to_linear(out) - core.srgb_to_linear(img)).std().item()

    day_sigma = measured_sigma(day, seed=100)
    night_sigma = measured_sigma(night, seed=101)
    assert night_sigma > 3.0 * day_sigma, "measured night noise > 3x measured day noise (per-preset ISO/chroma/spatial)"

    # Night's chroma-smear hits chroma, not luma: blur Cb/Cr variance drops a lot, Y barely moves.
    tex = torch.rand(1, 96, 96, 3)
    smeared = core.apply_chroma_smear(tex, night["sensor"]["chroma_smear"])
    ycc_before, ycc_after = core.rgb_to_ycbcr(tex), core.rgb_to_ycbcr(smeared)
    luma_var_ratio = ycc_after[..., 0].var().item() / ycc_before[..., 0].var().item()
    chroma_var_ratio = ycc_after[..., 1:3].var().item() / ycc_before[..., 1:3].var().item()
    assert chroma_var_ratio < 0.5, "night chroma-smear substantially reduces chroma variance"
    assert luma_var_ratio > 0.9, "night chroma-smear leaves luma variance essentially untouched"

    # Day has chroma_smear=0 -> bit-identical no-op (locked by group_differentials already,
    # re-verified here against the actual function, not just the JSON value).
    assert torch.equal(core.apply_chroma_smear(tex, day["sensor"]["chroma_smear"]), tex), \
        "day chroma_smear=0 is a bit-identical no-op through the real function"

def group_node_contract():    # Stage 7 - shape/range/dtype/differs across presets; deterministic;
                              #           auto-selection; realism scaling (0 ~ passthrough); codec toggle;
                              #           bad preset passes through (no crash); chunking bit-identical; odd dims survive
    import torch
    nodes_mod = _load_nodes_module()
    node = nodes_mod.IPhone12CaptureSim()
    img = torch.rand(6, 64, 96, 3)

    # shape/range/dtype/differs-from-input across all presets
    for name in ("iphone12_day", "iphone12_night"):
        (out,) = node.run(img, name, fps=30.0, seed=1, enable_codec=False)
        assert out.shape == img.shape, f"{name}: shape preserved"
        assert out.dtype == img.dtype, f"{name}: dtype preserved"
        assert out.min().item() >= 0.0 and out.max().item() <= 1.0, f"{name}: range preserved"
        assert not torch.equal(out, img), f"{name}: output differs from input"

    # deterministic with the same seed
    (a,) = node.run(img, "iphone12_day", fps=30.0, seed=7, enable_codec=False)
    (b,) = node.run(img, "iphone12_day", fps=30.0, seed=7, enable_codec=False)
    assert torch.equal(a, b), "same seed -> deterministic output"

    # auto-selection: dark scene -> night, bright scene -> day
    dark_img = torch.full((4, 32, 32, 3), 0.03)
    (auto_dark,) = node.run(dark_img, "auto", fps=30.0, seed=2, enable_codec=False)
    (night_dark,) = node.run(dark_img, "iphone12_night", fps=30.0, seed=2, enable_codec=False)
    assert torch.equal(auto_dark, night_dark), "auto-selection on a dark scene matches iphone12_night"

    bright_img = torch.full((4, 32, 32, 3), 0.6)
    (auto_bright,) = node.run(bright_img, "auto", fps=30.0, seed=2, enable_codec=False)
    (day_bright,) = node.run(bright_img, "iphone12_day", fps=30.0, seed=2, enable_codec=False)
    assert torch.equal(auto_bright, day_bright), "auto-selection on a bright scene matches iphone12_day"

    # realism scaling: 0 ~= passthrough through the DSP chain (codec off - codec always alters bits)
    (r0,) = node.run(img, "iphone12_day", fps=30.0, seed=3, realism=0.0, enable_codec=False)
    assert torch.allclose(r0, img, atol=1e-4), "realism=0 is ~passthrough through the DSP chain"

    # codec toggle
    (no_codec,) = node.run(img, "iphone12_day", fps=30.0, seed=4, enable_codec=False)
    (with_codec,) = node.run(img, "iphone12_day", fps=30.0, seed=4, enable_codec=True)
    assert not torch.equal(no_codec, with_codec), "enable_codec toggles the codec stage"

    # bad preset name passes through rather than crashing (spec 0: never kill a long render)
    (bad,) = node.run(img, "not_a_real_preset", fps=30.0, seed=5, enable_codec=False)
    assert torch.equal(bad, img), "bad preset name passes frames through unmodified"

    # chunking is bit-identical to not chunking (spatial stages don't couple across frames)
    (unchunked,) = node.run(img, "iphone12_day", fps=30.0, seed=6, enable_codec=False, chunk_frames=0)
    (chunked,) = node.run(img, "iphone12_day", fps=30.0, seed=6, enable_codec=False, chunk_frames=2)
    assert torch.equal(unchunked, chunked), "chunk_frames=2 is bit-identical to unchunked"

    # odd dims survive the full pipeline with codec on (codec crops internally to even)
    odd = torch.rand(3, 51, 77, 3)
    (odd_out,) = node.run(odd, "iphone12_day", fps=30.0, seed=8, enable_codec=True, enable_platform_pass=False)
    assert odd_out.shape[0] == odd.shape[0], "odd-dim run preserves frame count"
    assert odd_out.shape[1] % 2 == 0 and odd_out.shape[2] % 2 == 0, "odd-dim run crops H/W to even via codec"

    # platform pass downscales when width exceeds platform_max_width (1080)
    wide = torch.rand(3, 60, 1400, 3)
    (plat_out,) = node.run(wide, "iphone12_day", fps=30.0, seed=9, enable_codec=True, enable_platform_pass=True)
    assert plat_out.shape[2] <= 1080, "platform pass downscales to platform_max_width"

    # IPhone12PreflightCheck: passes images through unchanged + reports codec backend
    pf = nodes_mod.IPhone12PreflightCheck()
    pf_imgs, report = pf.run(img, fps=30.0)
    assert torch.equal(pf_imgs, img), "preflight passes images through unchanged"
    assert isinstance(report, str) and "codec backend" in report, "preflight report mentions codec backend"

    # IPhone12PresetInfo: dumps the preset JSON
    info = nodes_mod.IPhone12PresetInfo()
    (info_str,) = info.run("iphone12_day")
    assert "smart_hdr" in info_str and "iphone12_day" in info_str, "preset info dumps the preset JSON"


ALL_GROUPS = [group_preset_integrity, group_differentials, group_measured_noise, group_node_contract]

if __name__ == "__main__":
    failed = 0
    for g in ALL_GROUPS:
        try:
            g()
            print(f"PASS {g.__name__}")
        except NotImplementedError as e:
            print(f"SKIP (scaffold) {g.__name__}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {g.__name__}: {e}")
    sys.exit(1 if failed else 0)

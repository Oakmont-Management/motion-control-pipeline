#!/usr/bin/env bash
# Idempotent installer (spec 10). Primarily for RunPod; locally the ComfyUI venv already
# has av/numpy/scipy so this only verifies + runs the test suites.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[install] python deps (soft: av; torch/numpy/scipy assumed present)"
python -m pip install --break-system-packages -r "$HERE/requirements.txt" || true

echo "[install] checking ffmpeg + libx264"
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
  && echo "  ok: libx264" || echo "  WARN: libx264 not found (PyAV in-memory path still works)"

# NOTE: this pipeline does NOT use RIFE - Frame-Interpolation intentionally omitted (see plan).

echo "[install] running test suites (must pass: test_validation.py + test_presets.py, both exit 0)"
python "$HERE/tests/test_validation.py"
python "$HERE/tests/test_presets.py"

echo "[install] codec backend:"
python -c "from iphone_codec import codec_backend; print(' ', codec_backend())" 2>/dev/null || true
echo "[install] done"

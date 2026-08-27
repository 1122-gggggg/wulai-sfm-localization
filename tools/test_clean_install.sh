#!/usr/bin/env bash
# Rebuild the Python 3.10 environment in a temporary directory and run preflight.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE_VIDEO="${SFM_SMOKE_VIDEO:-$ROOT/模擬器/測試影片/P1190119.MP4}"
TEMP_ROOT="$(mktemp -d -t sfm-clean-install-XXXXXX)"
cleanup() {
  if [[ -d "$TEMP_ROOT" ]]; then
    rm -r -- "$TEMP_ROOT"
  fi
}
trap cleanup EXIT
if [[ ! -s "$SMOKE_VIDEO" ]]; then
  echo "[clean-install] test video is missing or empty: $SMOKE_VIDEO" >&2
  exit 2
fi
VENV_DIR="$TEMP_ROOT/venv"
SFM_VENV_DIR="$VENV_DIR" SFM_INSTALL_TEST_DEPS=1 \
  bash "$ROOT/tools/install_runtime.sh" --test-deps
SFM_UI_PYTHON="$VENV_DIR/bin/python" \
SFM_LOCALIZER_PYTHON="$VENV_DIR/bin/python" \
SFM_WORKSPACE_ROOT="$ROOT" \
SFM_TORCH_HUB_CACHE="$ROOT/執行環境/torch_hub_cache" \
  "$VENV_DIR/bin/python" "$ROOT/tools/simulator_preflight.py" \
    --workspace-root "$ROOT" \
    --site-profile "$ROOT/地圖檔/場域/river_site/site_profile.json" \
    --video "$SMOKE_VIDEO" \
    --check-runtime \
    --full-runtime \
    --json
"$VENV_DIR/bin/python" -m pytest -q --timeout=300 --cov \
  --cov-config="$ROOT/pyproject.toml" --cov-report=term-missing \
  --cov-fail-under=0 "$ROOT/tools" "$ROOT/定位演算法/validation/tests"

#!/usr/bin/env bash
# Rebuild the Python 3.10 environment in a temporary directory and run preflight.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMP_ROOT="$(mktemp -d -t sfm-clean-install-XXXXXX)"
cleanup() {
  if [[ -d "$TEMP_ROOT" ]]; then
    rm -r -- "$TEMP_ROOT"
  fi
}
trap cleanup EXIT
VENV_DIR="$TEMP_ROOT/venv"
SFM_VENV_DIR="$VENV_DIR" SFM_INSTALL_TEST_DEPS=1 \
  bash "$ROOT/tools/install_runtime.sh" --test-deps
SFM_UI_PYTHON="$VENV_DIR/bin/python" \
SFM_LOCALIZER_PYTHON="$VENV_DIR/bin/python" \
SFM_WORKSPACE_ROOT="$ROOT" \
SFM_TORCH_HUB_CACHE="$ROOT/執行環境/torch_hub_cache" \
  "$VENV_DIR/bin/python" "$ROOT/tools/simulator_preflight.py" \
    --workspace-root "$ROOT" \
    --site-profile "$ROOT/控制介面程式/site_profiles/river_site_edm.json" \
    --video "$ROOT/模擬器/測試影片/河濱_P1180118_first_2s.mp4" \
    --check-runtime \
    --full-runtime \
    --json
"$VENV_DIR/bin/python" -m pytest -q --timeout=300 --cov \
  --cov-config="$ROOT/pyproject.toml" --cov-report=term-missing \
  --cov-fail-under=0 "$ROOT/tools" "$ROOT/定位演算法/validation/tests"

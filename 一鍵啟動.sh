#!/usr/bin/env bash
# Unified operator entrypoint. Default: real aircraft. VIDEO=<file> switches to the
# simulated stream (no aircraft link, no takeoff). Opening the UI never sends a takeoff command.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${SFM_PYTHON:-python3.10}"
venv_dir="${SFM_VENV_DIR:-$root_dir/.venv}"
runtime_python="$venv_dir/bin/python"
portable_mode=0
portable_identity=""

for argument in "$@"; do
  if [[ "$argument" == "--site-profile" || "$argument" == --site-profile=* ]]; then
    echo "[一鍵啟動] 真機入口的 site profile 由 mission selection 原子產生；不可直接覆寫 --site-profile" >&2
    exit 2
  fi
done
if [[ -n "${SFM_SITE_PROFILE:-}" ]]; then
  echo "[一鍵啟動] 拒絕既有 SFM_SITE_PROFILE；請改用 SFM_MISSION_SELECTION" >&2
  exit 2
fi

if [[ -f "$root_dir/PORTABLE_PACKAGE.json" ]]; then
  portable_mode=1
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[一鍵啟動] 找不到 $python_bin；需要 Linux x86_64 的 CPython 3.10" >&2
    exit 1
  fi
  "$python_bin" "$root_dir/tools/package_manifest.py" verify --root "$root_dir"
  portable_identity="$("$python_bin" - "$root_dir" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
for name in ("PORTABLE_PACKAGE.json", "MANIFEST.tsv"):
    digest.update(name.encode("utf-8") + b"\0")
    digest.update((root / name).read_bytes())
print(digest.hexdigest())
PY
)"
elif [[ ! -e "$root_dir/.git" ]]; then  # a worktree has a .git file
  echo "[一鍵啟動] 非 Git 原始碼目錄必須包含 PORTABLE_PACKAGE.json；拒絕未封裝內容" >&2
  exit 1
fi

external_python=0
if [[ "${SFM_PORTABLE_ALLOW_EXTERNAL_PYTHON:-0}" == "1" ]] \
   && [[ -n "${SFM_UI_PYTHON:-}" ]]; then
  runtime_python="$SFM_UI_PYTHON"
  external_python=1
fi
if [[ "$portable_mode" == "1" && "$external_python" == "0" ]] \
   && [[ -x "$runtime_python" ]]; then
  marker="$venv_dir/.sfm-portable-runtime"
  if [[ -L "$venv_dir" || ! -f "$marker" ]] \
     || [[ "$(<"$marker")" != "$portable_identity" ]]; then
    echo "[一鍵啟動] 拒絕未由目前 portable package 建立的既有 .venv：$venv_dir" >&2
    echo "[一鍵啟動] 請移除該目錄後重新執行，由離線 wheelhouse 建立環境" >&2
    exit 1
  fi
fi
if [[ ! -x "$runtime_python" ]]; then
  echo "[一鍵啟動] 首次執行：從包內 wheelhouse 建立隔離環境"
  SFM_PYTHON="$python_bin" SFM_VENV_DIR="$venv_dir" \
    bash "$root_dir/tools/install_runtime.sh" --offline
  runtime_python="$venv_dir/bin/python"
fi

export SFM_WORKSPACE_ROOT="$root_dir"
export SFM_UI_PYTHON="$runtime_python"
export SFM_LOCALIZER_PYTHON="${SFM_LOCALIZER_PYTHON:-$runtime_python}"

# Map choice precedence: explicit SFM_MISSION_SELECTION > last opened map > river default.
if [[ -z "${SFM_MISSION_SELECTION:-}" ]]; then
  remembered_selection="$("$runtime_python" "$root_dir/控制介面程式/last_mission_memory.py" show || true)"
  if [[ -n "$remembered_selection" ]]; then
    echo "[一鍵啟動] 沿用上次開啟的地圖：$remembered_selection" >&2
  fi
fi
default_selection="$root_dir/控制介面程式/mission_selections/river_gluemap_all8_direct_localization.json"
export SFM_MISSION_SELECTION="${SFM_MISSION_SELECTION:-${remembered_selection:-$default_selection}}"

if [[ -n "${VIDEO:-}" ]]; then
  # Simulation: same mission selection, resolved to its read-only site profile snapshot.
  remember_arg=()
  [[ "${SFM_LAUNCH_DRY_RUN:-0}" == "1" ]] || remember_arg=(--remember)
  if ! mission_report="$("$runtime_python" "$root_dir/控制介面程式/launch_mission.py" \
    "$SFM_MISSION_SELECTION" --check-only "${remember_arg[@]}")"; then
    echo "[一鍵啟動] mission selection 驗證失敗" >&2
    exit 2
  fi
  SFM_SITE_PROFILE="$("$runtime_python" -c \
    'import json,sys; print(json.load(sys.stdin)["site_profile"])' <<<"$mission_report")"
  export SFM_SITE_PROFILE
  echo "[一鍵啟動] 模擬模式（VIDEO=$VIDEO）：不連接真機、不會起飛" >&2
  exec bash "$root_dir/控制介面程式/影片模擬串流/啟動.sh" "$@"
fi

echo "[一鍵啟動] 真機模式（預設）；若要模擬請設定 VIDEO=<影片路徑>" >&2
exec bash "$root_dir/控制介面程式/真機串流/啟動.sh" "$@"

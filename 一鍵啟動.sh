#!/usr/bin/env bash
# Portable live operator entrypoint. Opening the UI never sends a takeoff command.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${SFM_PYTHON:-python3.10}"
venv_dir="${SFM_VENV_DIR:-$root_dir/.venv}"
runtime_python="$venv_dir/bin/python"
portable_mode=0
portable_identity=""

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
elif [[ ! -d "$root_dir/.git" ]]; then
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

default_profile="$root_dir/控制介面程式/site_profiles/river_site_edm.json"
has_profile=0
for argument in "$@"; do
  if [[ "$argument" == "--site-profile" || "$argument" == --site-profile=* ]]; then
    has_profile=1
    break
  fi
done
if [[ "$has_profile" == "0" && -z "${SFM_SITE_PROFILE:-}" ]]; then
  export SFM_SITE_PROFILE="$default_profile"
fi

export SFM_WORKSPACE_ROOT="$root_dir"
export SFM_UI_PYTHON="$runtime_python"
export SFM_LOCALIZER_PYTHON="${SFM_LOCALIZER_PYTHON:-$runtime_python}"

exec bash "$root_dir/控制介面程式/真機串流/啟動.sh" "$@"

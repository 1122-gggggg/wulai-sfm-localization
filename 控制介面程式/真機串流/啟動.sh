#!/usr/bin/env bash
# 接口 A：真機串流 — SkyController / 無人機即時畫面 + 飛控 UI
# 預設不起飛；起飛僅操作員在 UI 按「起飛」。
set -euo pipefail
CTRL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$CTRL_DIR/.." && pwd)"
OI="$CTRL_DIR/operator_interface"
PACKAGE_ROOT="$ROOT"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -z "${SFM_UI_PYTHON:-}" && -x "$ROOT/.venv/bin/python" ]]; then
  export SFM_UI_PYTHON="$ROOT/.venv/bin/python"
fi
export LOCAL_TOPK="${LOCAL_TOPK:-0}"
export IP="${IP:-192.168.53.1}"
export CTRL="${CTRL:-skycontroller3}"

verify_portable_package() {
  if [[ ! -f "$PACKAGE_ROOT/PORTABLE_PACKAGE.json" ]]; then
    if [[ ! -d "$PACKAGE_ROOT/.git" ]]; then
      echo "[真機串流] ERROR: non-Git runtime is missing PORTABLE_PACKAGE.json" >&2
      exit 2
    fi
    return 0
  fi
  local verifier
  verifier="$(command -v python3.10 || command -v python3 || true)"
  if [[ -z "$verifier" ]]; then
    echo "[真機串流] ERROR: no Python interpreter available for portable manifest verification" >&2
    exit 2
  fi
  if [[ ! -f "$PACKAGE_ROOT/tools/package_manifest.py" ]]; then
    echo "[真機串流] ERROR: portable package is missing tools/package_manifest.py" >&2
    exit 2
  fi
  echo "[真機串流] verifying portable package manifest ..."
  if ! "$verifier" "$PACKAGE_ROOT/tools/package_manifest.py" verify --root "$PACKAGE_ROOT"; then
    echo "[真機串流] ERROR: portable package manifest verification failed" >&2
    exit 1
  fi
}

verify_portable_package

source "$OI/resolve_display.sh"
configure_operator_display "${SFM_LAUNCH_DRY_RUN:-0}"

# 真機任務只接受獨立 component mission selection。site profile 是 resolver
# 產生的唯讀相容 snapshot，不能再由另一個入口提供不同的飛行核准答案。
for a in "$@"; do
  if [[ "$a" == "--site-profile" || "$a" == --site-profile=* ]]; then
    echo "[真機串流] ERROR: site profile 由 mission selection 產生；拒絕直接 --site-profile" >&2
    exit 2
  fi
done
if [[ -n "${SFM_SITE_PROFILE:-}" ]]; then
  echo "[真機串流] ERROR: 拒絕既有 SFM_SITE_PROFILE；請只設定 SFM_MISSION_SELECTION" >&2
  exit 2
fi
if [[ -z "${SFM_MISSION_SELECTION:-}" ]]; then
  echo "[真機串流] ERROR: 必須用 SFM_MISSION_SELECTION 明確選擇並驗證任務" >&2
  exit 2
fi

MISSION_PYTHON="${SFM_UI_PYTHON:-}"
if [[ -z "$MISSION_PYTHON" && -x "$ROOT/.venv/bin/python" ]]; then
  MISSION_PYTHON="$ROOT/.venv/bin/python"
fi
if [[ -z "$MISSION_PYTHON" ]]; then
  echo "[真機串流] ERROR: 需要 .venv/bin/python 或 SFM_UI_PYTHON，拒絕使用 PATH python3" >&2
  exit 2
fi
if [[ -f "$PACKAGE_ROOT/PORTABLE_PACKAGE.json" ]]; then
  if [[ -L "$PACKAGE_ROOT/.venv" ]]; then
    echo "[真機串流] ERROR: portable package-local .venv must not be a symlink" >&2
    exit 2
  fi
  case "$MISSION_PYTHON" in
    "$PACKAGE_ROOT/.venv/"*)
      portable_marker="$PACKAGE_ROOT/.venv/.sfm-portable-runtime"
      identity_python="$(command -v python3.10 || command -v python3 || true)"
      if [[ -z "$identity_python" ]]; then
        echo "[真機串流] ERROR: no Python interpreter available for portable runtime binding" >&2
        exit 2
      fi
      portable_identity="$($identity_python - "$PACKAGE_ROOT" <<'PY'
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
      if [[ ! -f "$portable_marker" ]] \
         || [[ "$(<"$portable_marker")" != "$portable_identity" ]]; then
        echo "[真機串流] ERROR: package-local .venv is not bound to this portable manifest" >&2
        exit 2
      fi
      ;;
  esac
fi
if ! MISSION_REPORT="$("$MISSION_PYTHON" "$CTRL_DIR/launch_mission.py" \
  "$SFM_MISSION_SELECTION" --check-only)"; then
  echo "[真機串流] ERROR: mission selection 驗證失敗" >&2
  exit 2
fi
export SFM_SITE_PROFILE="$("$MISSION_PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin)["site_profile"])' \
  <<<"$MISSION_REPORT")"
export SFM_MISSION_SELECTION="$("$MISSION_PYTHON" -c \
  'import os,sys; print(os.path.realpath(sys.argv[1]))' \
  "$SFM_MISSION_SELECTION")"
echo "[真機串流] mission-selection: $SFM_MISSION_SELECTION"
echo "[真機串流] $MISSION_REPORT"

echo "[真機串流] IP=$IP CTRL=$CTRL LOCAL_TOPK=$LOCAL_TOPK"
echo "[真機串流] 起飛僅 UI 人工按鍵；動搖桿強制交回搖桿"
exec "$OI/start_anafi_live.sh" "$@"

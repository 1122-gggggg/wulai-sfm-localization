#!/usr/bin/env bash
# 互動式模擬介面：先選完整地圖對應的 site profile，再選影片，最後開啟 UI。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${SFM_UI_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "[選擇介面] Python 不可用: $PY" >&2
  exit 1
fi
exec "$PY" "$ROOT/控制介面程式/影片模擬串流/選擇啟動.py" "$@"

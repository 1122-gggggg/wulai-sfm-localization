#!/usr/bin/env bash
# 測自動飛行：一鍵驗收畫好的場域路線（Blender 存檔 → 自動飛行演算法 dry-run）。
# 用法： ./測自動飛行.sh <flight_path.json> [--steps N] [--min-progress F]
# 流程：驗證路線文件 → 用生產版 RouteAutoController 跑 toy-dynamics 閉環 →
#       印 progress/state → 達標 exit 0。只讀測試，不碰真機、不寫 site_profile。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROUTE="${1:-}"
if [[ -z "${ROUTE}" || "${ROUTE}" == -* ]]; then
  echo "用法： $0 <flight_path.json> [--steps N] [--min-progress F]" >&2
  echo "例：   $0 地圖檔/場域/river_site/routes/flight_path.json" >&2
  exit 1
fi
shift
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then PY="python3"; fi
exec "${PY}" "${ROOT}/tools/test_route_autoflight.py" --route "${ROUTE}" "$@"

#!/usr/bin/env bash
# 接口 A：真機串流 — SkyController / 無人機即時畫面 + 飛控 UI
# 預設不起飛；起飛僅操作員在 UI 按「起飛」。
set -euo pipefail
CTRL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$CTRL_DIR/.." && pwd)"
OI="$CTRL_DIR/operator_interface"

export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export SFM_UI_PYTHON="${SFM_UI_PYTHON:-$ROOT/.venv/bin/python}"
export LOCAL_TOPK="${LOCAL_TOPK:-0}"
export IP="${IP:-192.168.53.1}"
export CTRL="${CTRL:-skycontroller3}"

# 真機必須明確選場域；不可因舊預設而把河濱地圖帶到別的場地。
HAS_PROFILE=0
for a in "$@"; do
  if [[ "$a" == "--site-profile" || "$a" == --site-profile=* ]]; then
    HAS_PROFILE=1
    break
  fi
done
if [[ "$HAS_PROFILE" -eq 0 && -z "${SFM_SITE_PROFILE:-}" ]]; then
  echo "[真機串流] ERROR: 必須用 --site-profile 或 SFM_SITE_PROFILE 明確選擇場域" >&2
  echo "[真機串流] 請選擇含同座標地圖、定位 bundle 與已驗證安全航線的 site profile（urai_edm 目前 route=null，只允許離線 replay）" >&2
  exit 2
fi

echo "[真機串流] IP=$IP CTRL=$CTRL LOCAL_TOPK=$LOCAL_TOPK"
echo "[真機串流] 起飛僅 UI 人工按鍵；動搖桿強制交回搖桿"
cd "$OI"
exec ./start_anafi_live.sh "$@"

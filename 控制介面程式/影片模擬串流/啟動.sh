#!/usr/bin/env bash
# 接口 B：影片模擬串流 — 預錄影片當 720p 串流（無真機、無起飛）
#
# 預設「飛行管線」（與真機定位狀態機一致）：
#   1) 第一幀凍住 → MegaLoc 跑一次，EDM 候選鎖定
#   2) 鎖定後進 TRACK 持續定位
#   3) 中途 LOST → 暫停串流，MegaLoc 跑一次，之後 EDM recovery
#   4) 找回後解除凍結，恢復 TRACK
#
# 可選 LOC_BENCH_TRACK=1：跳過 MegaLoc，每幀純 TRACK 測速（非飛行管線）。
set -euo pipefail
CTRL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$CTRL_DIR/.." && pwd)"
OI="$CTRL_DIR/operator_interface"
PY="${SFM_UI_PYTHON:-$ROOT/.venv/bin/python}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT/模擬器/測試影片}"
SITE_PROFILE="${SFM_SITE_PROFILE:-$CTRL_DIR/site_profiles/example_site_edm.json}"
# Target EDM verified balance: TRACK=1, LOW/WEAK=3.
LOCAL_TOPK="${LOCAL_TOPK:-1}"
STREAM_FPS="${STREAM_FPS:-30}"
VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
# BOOT_INIT hold budget (ms). Released early on first successful MegaLoc lock.
BOOT_LOCK_MS="${BOOT_LOCK_MS:-20000}"
LOST_HOLD_MAX="${LOST_HOLD_MAX:-8}"
LOST_HOLD_TIMEOUT_MS="${LOST_HOLD_TIMEOUT_MS:-15000}"
POSE_STABILIZE="${POSE_STABILIZE:-0}"

usage() {
  cat <<EOF
用法:
  $0 <影片路徑.mp4> [額外 flight_operator_app 參數...]
  VIDEO=/path/to.mp4 $0 [額外參數...]

飛行管線（預設）:
  首幀 MegaLoc 一次 → TRACK；LOW/WEAK 提高 EDM top-k；LOST 才凍幀並再跑 MegaLoc 一次

串流 30 FPS / 定位 ~15 FPS:
  不堆積舊幀；worker busy 時只保留最新一幀 (latest-frame coalesce)。
  畫面可 30fps 刷新；pose 以定位完成率更新。

常用環境變數:
  VIDEO / 第一個參數     影片路徑
  LOCAL_TOPK             TRACK 參考數（預設 1；LOW/WEAK 自動升為 3）
  STREAM_FPS             餵入串流 FPS（預設 30）
  BOOT_LOCK_MS           首幀 MegaLoc 最長凍幀（預設 20000；鎖定後立刻放行）
  LOST_HOLD_MAX          LOST 凍幀重試次數（預設 8）
  LOST_HOLD_TIMEOUT_MS   LOST 凍幀逾時放行（預設 15000）
  LOC_BENCH_TRACK=1      改純 TRACK 測速（跳過 MegaLoc / 不走飛行管線）
  POSE_STABILIZE=1       發布三幀一致性過濾姿態（原始 PnP 仍寫入遺測）
  SFM_SITE_PROFILE       site profile（預設 example_site_edm，請改成你的場域設定）

可用測試片（\$VIDEO_DIR）:
EOF
  if [[ -d "$VIDEO_DIR" ]]; then
    found=0
    while IFS= read -r -d '' f; do
      echo "  $f"
      found=1
    done < <(find "$VIDEO_DIR" -maxdepth 1 -type f \
      \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) -print0 2>/dev/null)
    if [[ "$found" -eq 0 ]]; then
      echo "  （目錄空）"
    fi
  else
    echo "  （無 $VIDEO_DIR）"
  fi
}

VIDEO_PATH="${VIDEO:-}"
EXTRA=()
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  VIDEO_PATH="$1"
  shift
fi
EXTRA+=("$@")

if [[ -z "$VIDEO_PATH" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$VIDEO_PATH" ]]; then
  if [[ -f "$VIDEO_DIR/$VIDEO_PATH" ]]; then
    VIDEO_PATH="$VIDEO_DIR/$VIDEO_PATH"
  else
    echo "[影片模擬串流] 找不到影片: $VIDEO_PATH" >&2
    usage >&2
    exit 1
  fi
fi
VIDEO_PATH="$(cd "$(dirname "$VIDEO_PATH")" && pwd)/$(basename "$VIDEO_PATH")"

if [[ ! -x "$PY" && ! -f "$PY" ]]; then
  echo "[影片模擬串流] Python 不可用: $PY" >&2
  exit 1
fi
if [[ ! -f "$SITE_PROFILE" ]]; then
  echo "[影片模擬串流] site profile 不存在: $SITE_PROFILE" >&2
  exit 1
fi

ARGS=(
  -u flight_operator_app.py
  --site-profile "$SITE_PROFILE"
  --video "$VIDEO_PATH"
  --video-stride "$VIDEO_STRIDE"
  --stream-fps "$STREAM_FPS"
  --live-localize
  --no-live-detect
  --local-topk "$LOCAL_TOPK"
  --auto-inspect
)

if [[ "$POSE_STABILIZE" == "1" ]]; then
  ARGS+=(--pose-stabilize)
fi

if [[ "${LOC_BENCH_TRACK:-0}" == "1" ]]; then
  # Pure TRACK microbench: no MegaLoc BOOT/LOST pipeline.
  ARGS+=(--boot-lock-ms 0 --loc-force-track-bench --no-lost-hold)
  echo "[影片模擬串流] LOC_BENCH_TRACK=1：純 TRACK 測速（跳過 MegaLoc / 不凍幀）"
else
  # Flight-like pipeline (default).
  ARGS+=(
    --boot-lock-ms "$BOOT_LOCK_MS"
    --lost-hold
    --lost-hold-max-attempts "$LOST_HOLD_MAX"
    --lost-hold-timeout-ms "$LOST_HOLD_TIMEOUT_MS"
  )
  echo "[影片模擬串流] 飛行管線：BOOT MegaLoc 一次 → TRACK=1；LOW/WEAK=3；LOST MegaLoc 一次 → EDM recovery"
  echo "[影片模擬串流] boot_lock_ms=$BOOT_LOCK_MS lost_hold_max=$LOST_HOLD_MAX lost_timeout_ms=$LOST_HOLD_TIMEOUT_MS"
fi

export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

echo "[影片模擬串流] video=$VIDEO_PATH"
echo "[影片模擬串流] profile=$SITE_PROFILE topk=$LOCAL_TOPK stream_fps=$STREAM_FPS"
echo "[影片模擬串流] 無真機連線、無起飛指令"
cd "$OI"
exec "$PY" "${ARGS[@]}" "${EXTRA[@]}"

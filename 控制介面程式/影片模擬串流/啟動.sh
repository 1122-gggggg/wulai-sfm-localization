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
SITE_PROFILE="${SFM_SITE_PROFILE:-$CTRL_DIR/site_profiles/urai_edm.json}"
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
  SFM_SITE_PROFILE       site profile（預設 烏來 EDM v1；換場域改這個變數即可）
  ANAFI_LINK_SIM=1       模擬 ANAFI 實機無線串流畫質（白皮書 v1.4 §5.2）
                         720p H264 main profile 5 Mb/s、45 slices×16px、intra-refresh
                         錄影檔碼率比實機高 5-12 倍，不開這個會高估定位表現
  ANAFI_LINK_KBPS        串流碼率 kbps（預設 5000，即白皮書的 up to 5 Mb/s）
  ANAFI_LINK_LATENCY_MS  端到端延遲 ms（白皮書 280；純 replay 只影響時間戳語意）
  ANAFI_LINK_LOSS_PCT    slice 丟包率 %（模擬 Wi-Fi 遺失 + error concealment）
  SFM_HOLD_ON_LOW_CONF=1 精度優先：連續低信心即暫停串流（等同懸停），
                         held frame 改走 LOST recovery（高 top-k + MegaLoc）
  SFM_LOW_CONF_HOLD_RESULTS  連續幾次低信心才懸停（預設 2）

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
if [[ "${SFM_HOLD_ON_LOW_CONF:-0}" == "1" ]]; then
  echo "[影片模擬串流] 精度優先：連續 ${SFM_LOW_CONF_HOLD_RESULTS:-2} 次低信心即懸停（串流暫停）並升級為 LOST recovery"
fi
if [[ -n "${ANAFI_LINK_LATENCY_MS:-}" || -n "${ANAFI_LINK_LOSS_PCT:-}" ]]; then
  echo "[影片模擬串流] 鏈路劣化：延遲 ${ANAFI_LINK_LATENCY_MS:-0} ms、丟包 ${ANAFI_LINK_LOSS_PCT:-0} %"
fi
if [[ "${ANAFI_LINK_SIM:-0}" == "1" ]]; then
  echo "[影片模擬串流] ANAFI 串流模擬：H264 main ${ANAFI_LINK_KBPS:-5000} kbps + intra-refresh（貼近實機畫質）"
else
  echo "[影片模擬串流] 未開串流模擬：畫質優於實機（ANAFI_LINK_SIM=1 可開啟）"
fi
echo "[影片模擬串流] 無真機連線、無起飛指令"
cd "$OI"
exec "$PY" "${ARGS[@]}" "${EXTRA[@]}"

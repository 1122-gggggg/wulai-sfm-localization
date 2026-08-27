#!/usr/bin/env bash
# 接口 B：影片模擬串流 — 預錄影片當 720p 串流（無真機、無起飛）
#
#   1) 第一幀凍住 → MegaLoc 跑一次，EDM 候選鎖定
#   2) 鎖定後進 TRACK 持續定位
#   3) 連續低信心或 LOST → 凍結目前影格，模擬真機懸停
#   4) 用 LOST EDM top-k 5，再依 profile 排程 MegaLoc top-k 10/20
#   5) 找回後恢復 TRACK；同幀無解時有限次重試後改用新影格
#
# 可選 LOC_BENCH_TRACK=1：跳過 MegaLoc，每幀純 TRACK 測速（非飛行管線）。
set -euo pipefail
CTRL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$CTRL_DIR/.." && pwd)"
OI="$CTRL_DIR/operator_interface"
PY="${SFM_UI_PYTHON:-$ROOT/.venv/bin/python}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT/模擬器/測試影片}"
SITE_PROFILE="${SFM_SITE_PROFILE:-$ROOT/地圖檔/場域/river_site/site_profile.json}"
EXPECTED_TORCH_HUB_CACHE="$ROOT/執行環境/torch_hub_cache"

canonical_path() {
  readlink -f -- "$1" 2>/dev/null || realpath -m -- "$1"
}

normalized_path() {
  realpath -s -m -- "$1"
}

check_fixed_environment() {
  local expected actual
  expected="$(canonical_path "$ROOT")"
  if [[ -n "${SFM_WORKSPACE_ROOT:-}" ]]; then
    actual="$(canonical_path "$SFM_WORKSPACE_ROOT")"
    if [[ "$actual" != "$expected" ]]; then
      echo "[影片模擬串流] SFM_WORKSPACE_ROOT 必須指向目前工作區: $expected" >&2
      exit 2
    fi
  fi
  expected="$(canonical_path "$EXPECTED_TORCH_HUB_CACHE")"
  if [[ -n "${SFM_TORCH_HUB_CACHE:-}" ]]; then
    actual="$(canonical_path "$SFM_TORCH_HUB_CACHE")"
    if [[ "$actual" != "$expected" ]]; then
      echo "[影片模擬串流] SFM_TORCH_HUB_CACHE 必須指向工作區固定 cache: $expected" >&2
      exit 2
    fi
  fi
  expected="$(normalized_path "$PY")"
  if [[ -n "${SFM_LOCALIZER_PYTHON:-}" ]]; then
    actual="$(normalized_path "$SFM_LOCALIZER_PYTHON")"
    if [[ "$actual" != "$expected" ]]; then
      echo "[影片模擬串流] SFM_LOCALIZER_PYTHON 必須與 SFM_UI_PYTHON 使用同一 Python: $expected" >&2
      exit 2
    fi
  fi
  export SFM_WORKSPACE_ROOT="$ROOT"
  export SFM_TORCH_HUB_CACHE="$EXPECTED_TORCH_HUB_CACHE"
  export SFM_UI_PYTHON="$PY"
  export SFM_LOCALIZER_PYTHON="$PY"
  export SFM_SITE_PROFILE="$SITE_PROFILE"
}

check_fixed_environment
P119_SHA256="600bbf70227311cab079d77fcb896f97e6d3e55f6bc40b5bef01d74b65f7826c"
LOCAL_TOPK="${LOCAL_TOPK:-2}"
STREAM_FPS="${STREAM_FPS:-30}"
VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
# BOOT_INIT hold budget (ms). Released early on first successful MegaLoc lock.
BOOT_LOCK_MS="${BOOT_LOCK_MS:-20000}"
LOST_HOLD_MAX="${LOST_HOLD_MAX:-8}"
LOST_HOLD_TIMEOUT_MS="${LOST_HOLD_TIMEOUT_MS:-3000}"
SFM_HOLD_ON_LOW_CONF="${SFM_HOLD_ON_LOW_CONF:-1}"
POSE_STABILIZE="${POSE_STABILIZE:-1}"
DRY_RUN="${SFM_LAUNCH_DRY_RUN:-0}"
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "[影片模擬串流] SFM_LAUNCH_DRY_RUN 必須是 0 或 1" >&2
  exit 2
fi
if [[ "$SFM_HOLD_ON_LOW_CONF" != "0" && "$SFM_HOLD_ON_LOW_CONF" != "1" ]]; then
  echo "[影片模擬串流] SFM_HOLD_ON_LOW_CONF 必須是 0 或 1" >&2
  exit 2
fi
if [[ "$DRY_RUN" == "0" && -n "${SFM_UI_PYTHON:-}" \
      && "$(normalized_path "$PY")" != "$(normalized_path "$ROOT/.venv/bin/python")" \
      && "${SFM_PORTABLE_ALLOW_EXTERNAL_PYTHON:-0}" != "1" ]]; then
  echo "[影片模擬串流] portable mode 要求使用 $ROOT/.venv/bin/python" >&2
  echo "[影片模擬串流] 若已完成獨立環境驗證，才可設定 SFM_PORTABLE_ALLOW_EXTERNAL_PYTHON=1" >&2
  exit 2
fi

LINK_PRESET="${ANAFI_LINK_PRESET:-nominal}"
case "$LINK_PRESET" in
  nominal) PRESET_LOSS=0 ;;
  loss-1) PRESET_LOSS=1 ;;
  loss-3) PRESET_LOSS=3 ;;
  loss-5) PRESET_LOSS=5 ;;
  *)
    echo "[影片模擬串流] ANAFI_LINK_PRESET 只支援 nominal/loss-1/loss-3/loss-5" >&2
    exit 2
    ;;
esac
export ANAFI_LINK_SIM=1
export ANAFI_LINK_KBPS="${ANAFI_LINK_KBPS:-5000}"
export ANAFI_LINK_PROFILE="${ANAFI_LINK_PROFILE:-main}"
export ANAFI_LINK_LATENCY_MS="${ANAFI_LINK_LATENCY_MS:-280}"
export ANAFI_LINK_LOSS_PCT="${ANAFI_LINK_LOSS_PCT:-$PRESET_LOSS}"

usage() {
  cat <<EOF
用法:
  $0 [影片路徑.mp4]
  VIDEO=/path/to.mp4 $0

若沒有指定 VIDEO，啟動器會先找測試影片目錄中的
P1190119.MP4；否則只有一部影片時自動使用它。多部影片時請指定 VIDEO 或第一個參數。

飛行管線（預設）:
  MegaLoc top-k 10，失敗升到 20 並完整 EDM 驗證；仍失敗才進入後續恢復／旋轉搜尋

串流 30 FPS / 定位 ~15 FPS:
  不堆積舊幀；worker busy 時只保留最新一幀 (latest-frame coalesce)。
  畫面可 30fps 刷新；pose 以定位完成率更新。

常用環境變數:
  VIDEO / 第一個參數     影片路徑
  LOCAL_TOPK             TRACK 參考數（預設 2；WEAK 3、LOST 附近 5）
  STREAM_FPS             餵入串流 FPS（預設 30）
  BOOT_LOCK_MS           首幀 MegaLoc 最長凍幀（預設 20000；鎖定後立刻放行）
  LOST_HOLD_MAX          同一影格高精度重定位上限（預設 8 次）
  LOST_HOLD_TIMEOUT_MS   凍幀硬逾時（預設 3000 ms；仍受次數限制）
  LOC_BENCH_TRACK=1      改純 TRACK 測速（跳過 MegaLoc / 不走飛行管線）
  POSE_STABILIZE=1       因果發布濾波（過去+當前 3 點中位數＋0.15s 低通；實機可用）
  SFM_SITE_PROFILE       site profile（預設 河濱 EDM；換場域改這個變數即可）
  ANAFI_LINK_PRESET      nominal/loss-1/loss-3/loss-5（預設 nominal）
  ANAFI_LINK_SIM=1       模擬 ANAFI 實機無線串流畫質（白皮書 v1.4 §5.2）
                         720p H264 main profile 5 Mb/s、45 slices×16px、intra-refresh
                         錄影檔碼率比實機高 5-12 倍，不開這個會高估定位表現
  ANAFI_LINK_KBPS        串流碼率 kbps（預設 5000，即白皮書的 up to 5 Mb/s）
  ANAFI_LINK_LATENCY_MS  端到端延遲 ms（白皮書 280；以解碼後固定影格 backlog 模擬）
  ANAFI_LINK_LOSS_PCT    slice 丟包率 %（模擬 Wi-Fi 遺失 + error concealment）
  SFM_HOLD_ON_LOW_CONF   連續低信心時凍幀並升級 MegaLoc（預設 1）
  SFM_LOW_CONF_HOLD_RESULTS  升級門檻（預設連續 2 筆）

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
for arg in "${EXTRA[@]}"; do
  case "$arg" in
    --live|--interface|--interface=*|--video|--video=*|--site-profile|--site-profile=*)
      echo "[影片模擬串流] 拒絕跨接口參數: $arg" >&2
      echo "[影片模擬串流] 此入口固定為 simulated-stream，不會連接實機" >&2
      exit 2
      ;;
  esac
done
if [[ "${#EXTRA[@]}" -gt 0 ]]; then
  echo "[影片模擬串流] portable 入口不接受額外 flight_operator_app 參數" >&2
  echo "[影片模擬串流] 請使用已驗證的選擇介面與啟動器環境變數" >&2
  exit 2
fi

if [[ -z "$VIDEO_PATH" ]]; then
  if [[ -f "$VIDEO_DIR/P1190119.MP4" ]]; then
    VIDEO_PATH="$VIDEO_DIR/P1190119.MP4"
  else
    mapfile -t video_candidates < <(find "$VIDEO_DIR" -maxdepth 1 -type f \
      \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) -print 2>/dev/null | sort)
    if [[ "${#video_candidates[@]}" -eq 1 ]]; then
      VIDEO_PATH="${video_candidates[0]}"
    elif [[ "${#video_candidates[@]}" -eq 0 ]]; then
      echo "[影片模擬串流] 測試影片目錄沒有可用影片: $VIDEO_DIR" >&2
      usage >&2
      exit 1
    else
      echo "[影片模擬串流] 發現多部影片，請指定 VIDEO 或第一個參數:" >&2
      printf '  %s\n' "${video_candidates[@]}" >&2
      exit 2
    fi
  fi
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

if [[ "$(basename "$VIDEO_PATH")" == "P1190119.MP4" ]]; then
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "[影片模擬串流] 無法驗證預設 P119：sha256sum 不可用" >&2
    exit 1
  fi
  ACTUAL_VIDEO_SHA="$(sha256sum "$VIDEO_PATH" | awk '{print $1}')"
  if [[ "$ACTUAL_VIDEO_SHA" != "$P119_SHA256" ]]; then
    echo "[影片模擬串流] 預設 P119 SHA-256 不符" >&2
    echo "  expected=$P119_SHA256" >&2
    echo "  actual=$ACTUAL_VIDEO_SHA" >&2
    echo "  請明確傳入另一個影片，不會靜默替換預設檔" >&2
    exit 1
  fi
  export SFM_SOURCE_INTEGRITY="KNOWN_INCOMPLETE"
  export SFM_SOURCE_DECLARED_FRAMES=2935
  export SFM_SOURCE_DECODED_FRAMES=2934
fi

if [[ ! -x "$PY" ]]; then
  echo "[影片模擬串流] Python 不可用: $PY" >&2
  exit 1
fi
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
  echo "[影片模擬串流] 需要可執行的 CPython 3.10: $PY" >&2
  exit 1
fi
if [[ ! -f "$SITE_PROFILE" ]]; then
  echo "[影片模擬串流] site profile 不存在: $SITE_PROFILE" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "0" ]]; then
  "$PY" "$ROOT/tools/simulator_preflight.py" \
    --workspace-root "$ROOT" \
    --site-profile "$SITE_PROFILE" \
    --video "$VIDEO_PATH" \
    --check-runtime \
    --full-runtime
fi

ARGS=(
  -u "$OI/flight_operator_app.py"
  --interface simulated-stream
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
  # Flight-like pipeline (default): hover/freeze, then staged EDM + MegaLoc recovery.
  export SFM_HOLD_ON_LOW_CONF
  export SFM_EDM_REF_FEATURE_CACHE="${SFM_EDM_REF_FEATURE_CACHE:-32}"
  ARGS+=(
    --boot-lock-ms "$BOOT_LOCK_MS"
    --lost-hold
    --lost-hold-max-attempts "$LOST_HOLD_MAX"
    --lost-hold-timeout-ms "$LOST_HOLD_TIMEOUT_MS"
  )
  if [[ "$SFM_HOLD_ON_LOW_CONF" == "1" ]]; then
    ARGS+=(--hold-on-low-confidence)
  else
    ARGS+=(--no-hold-on-low-confidence)
  fi
  echo "[影片模擬串流] 飛行管線：低信心/LOST 凍幀；LOST EDM 5 張，再依 profile 排程 MegaLoc 10/20"
  echo "[影片模擬串流] boot_lock_ms=$BOOT_LOCK_MS lost_hold=on max=$LOST_HOLD_MAX timeout_ms=$LOST_HOLD_TIMEOUT_MS"
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
source "$OI/resolve_display.sh"
configure_operator_display "$DRY_RUN"

echo "[影片模擬串流] video=$VIDEO_PATH"
echo "[影片模擬串流] profile=$SITE_PROFILE topk=$LOCAL_TOPK stream_fps=$STREAM_FPS"
echo "[影片模擬串流] 低信心 recovery：凍幀=${SFM_HOLD_ON_LOW_CONF}，連續 ${SFM_LOW_CONF_HOLD_RESULTS:-2} 筆觸發"
echo "[影片模擬串流] 鏈路 preset=$LINK_PRESET：延遲 $ANAFI_LINK_LATENCY_MS ms、丟包 $ANAFI_LINK_LOSS_PCT %"
echo "[影片模擬串流] ANAFI 串流模擬：H264 $ANAFI_LINK_PROFILE $ANAFI_LINK_KBPS kbps + intra-refresh"
if [[ "${SFM_SOURCE_INTEGRITY:-}" == "KNOWN_INCOMPLETE" ]]; then
  echo "[影片模擬串流] P119 KNOWN_INCOMPLETE：container=2935 decoded=2934，EOF 將停在最後完整幀"
fi
echo "[影片模擬串流] 無真機連線、無起飛指令"
CMD=("$PY" "${ARGS[@]}" "${EXTRA[@]}")
if [[ "$DRY_RUN" == "1" ]]; then
  printf '[影片模擬串流] dry-run command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi
exec "${CMD[@]}"

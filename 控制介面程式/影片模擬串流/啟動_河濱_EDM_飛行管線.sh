#!/usr/bin/env bash
# 接口 B：影片模擬串流 — 河濱 EDM 飛行管線（local_topk=1）
# 首幀 MegaLoc(BOOT) → TRACK → LOST 凍幀 MegaLoc → 找回後 TRACK
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CTRL_DIR="$(cd "$HERE/.." && pwd)"
export SFM_SITE_PROFILE="${SFM_SITE_PROFILE:-$CTRL_DIR/site_profiles/river_site_edm.json}"
export LOCAL_TOPK="${LOCAL_TOPK:-1}"
export STREAM_FPS="${STREAM_FPS:-30}"
export BOOT_LOCK_MS="${BOOT_LOCK_MS:-20000}"
export LOST_HOLD_MAX="${LOST_HOLD_MAX:-8}"
export LOST_HOLD_TIMEOUT_MS="${LOST_HOLD_TIMEOUT_MS:-15000}"
# Explicitly off: this is NOT pure TRACK microbench.
unset LOC_BENCH_TRACK || true
export LOC_BENCH_TRACK=0
echo "[影片模擬串流/飛行管線] 河濱 EDM topk=1 · 無真機 · 模擬起飛 MegaLoc + TRACK + LOST 凍幀"
exec "$HERE/啟動.sh" "$@"

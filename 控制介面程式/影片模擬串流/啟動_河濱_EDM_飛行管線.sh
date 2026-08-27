#!/usr/bin/env bash
# 接口 B：影片模擬串流 — 河濱 EDM 飛行管線（local_topk=2, weak=5, 逐張 PnP）
# 首幀 MegaLoc(BOOT) → TRACK → LOST 先附近 EDM，再依 profile 週期重試 MegaLoc
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CTRL_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$CTRL_DIR/.." && pwd)"
export SFM_SITE_PROFILE="${SFM_SITE_PROFILE:-$ROOT/地圖檔/場域/river_site/site_profile.json}"
export LOCAL_TOPK="${LOCAL_TOPK:-2}"
export STREAM_FPS="${STREAM_FPS:-30}"
export BOOT_LOCK_MS="${BOOT_LOCK_MS:-20000}"
export POSE_STABILIZE="${POSE_STABILIZE:-1}"
unset LOC_BENCH_TRACK || true
export LOC_BENCH_TRACK=0
echo "[影片模擬串流/飛行管線] 河濱 EDM topk=2 · 無凍幀 · 因果 pose stabilize"
exec "$HERE/啟動.sh" "$@"

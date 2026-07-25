#!/usr/bin/env bash
# 接口 B：影片模擬串流 — 純 TRACK 測速（跳過 MegaLoc，非飛行管線）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export LOC_BENCH_TRACK=1
export LOCAL_TOPK="${LOCAL_TOPK:-2}"
export STREAM_FPS="${STREAM_FPS:-30}"
echo "[影片模擬串流/純TRACK] 每幀 force TRACK（測 wall_ms / loc FPS；不模擬起飛/LOST）"
exec "$HERE/啟動.sh" "$@"

#!/usr/bin/env bash
# Launch the fixed map/video selector through the real GUI path, then stop it.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
log_file="$(mktemp -t sfm-ui-smoke-XXXXXX.log)"
process_id=""
video_path="${SFM_SMOKE_VIDEO:-$root_dir/模擬器/測試影片/P1190119.MP4}"
if [[ ! -s "$video_path" ]]; then
  echo "[ui-smoke] test video is missing or empty: $video_path" >&2
  echo "[ui-smoke] set SFM_SMOKE_VIDEO to an explicit replay video" >&2
  exit 2
fi

cleanup() {
  if [[ -n "$process_id" ]] && kill -0 "$process_id" 2>/dev/null; then
    kill -TERM -- "-$process_id" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      if ! kill -0 "$process_id" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$process_id" 2>/dev/null; then
      kill -KILL -- "-$process_id" 2>/dev/null || true
    fi
  fi
  if [[ -n "$process_id" ]]; then
    wait "$process_id" 2>/dev/null || true
  fi
  if [[ -f "$log_file" ]]; then
    rm -- "$log_file"
  fi
}
trap cleanup EXIT

SFM_WORKSPACE_ROOT="$root_dir" setsid \
  "$root_dir/控制介面程式/影片模擬串流/選擇啟動.sh" \
  --map "$root_dir/地圖檔/場域/river_site/releases/river_site_b0_p116_p117_20260818/map/map.ply" \
  --video "$video_path" \
  >"$log_file" 2>&1 &
process_id=$!

feed_ready=0
pose_ready=0
session_dir=""
for _ in $(seq 1 120); do
  if grep -Fq "[operator] auto-inspect: started localization feed" "$log_file"; then
    feed_ready=1
  fi
  if [[ -z "$session_dir" ]]; then
    session_dir="$(sed -n 's/^\[session\] \(.*\) | disk=.*/\1/p' "$log_file" | tail -n 1)"
  fi
  if [[ -n "$session_dir" ]] && [[ -f "$session_dir/localization.jsonl" ]] && \
      grep -Eq '"event": "pose_result".*"success": true.*"pose": \{' \
        "$session_dir/localization.jsonl"; then
    pose_ready=1
  fi
  if [[ "$feed_ready" == "1" && "$pose_ready" == "1" ]]; then
    break
  fi
  if ! kill -0 "$process_id" 2>/dev/null; then
    break
  fi
  sleep 1
done

cat "$log_file"
if [[ "$feed_ready" != "1" ]]; then
  echo "[ui-smoke] GUI did not reach the simulated localization feed" >&2
  exit 1
fi
if [[ "$pose_ready" != "1" ]]; then
  echo "[ui-smoke] simulated GUI did not produce a valid localization pose" >&2
  exit 1
fi
echo "[ui-smoke] PASS: simulated GUI produced a valid localization pose; stopping test session"

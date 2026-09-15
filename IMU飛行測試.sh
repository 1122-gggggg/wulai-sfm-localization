#!/usr/bin/env bash
# IMU 飛行測試：手動搖桿飛行 + 開定位，全程把 IMU、搖桿指令與畫面錄成一組
# 可離線重播的資料，用來判斷「把 IMU 接進定位系統」這條路走不走得通。
#
# 用法：
#   ./IMU飛行測試.sh                     # 用預設 mission selection
#   ./IMU飛行測試.sh --auto-inspect      # 開介面就自動開始定位（仍然不起飛）
#   SFM_MISSION_SELECTION=<path> ./IMU飛行測試.sh
#
# 這個腳本只做兩件事：打開真機操作介面，並把錄製開起來。
# 它不會起飛、不會下任何飛行指令；起飛與「開始定位」都由操作員自己動手。
#
# 錄到哪裡（outputs/flight_logs/session_<UTC>_real-flight_<id>/）：
#   telemetry.jsonl        fused_odometry：姿態 roll/pitch/yaw + NED 速度 + 高度 + GPS（~8 Hz）
#                          stick_axes：搖桿各軸原始值，動了就記、沒動 1 Hz 一筆
#   localization.jsonl     每幀定位結果，並附上「這一幀送出去時掛的那個 IMU 樣本」
#                          與 pose_status / imu_bridge / imu_bridge_reason 診斷
#   imu_test/frames/       定位真正吃到的那幾張畫面（JPEG）
#   imu_test/frames.jsonl  每張畫面的 capture stamp 與配對的 IMU 樣本
#
# 為什麼要另外存 imu_test/frames：機上錄影跟主機 telemetry 是兩個時鐘，
# 舊 ESEKF runbook 的手動影片對齊流程已退役。
# 這裡存的是定位當下那一張圖，跟 IMU 同一個 monotonic 時鐘，不用再對齊。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${ROOT}/.venv/bin/python"
SESSION_ROOT="${ROOT}/outputs/flight_logs"

export SFM_IMU_FLIGHT_TEST=1
# 這趟是「量測」不是「導航」。預設 operator acceptance 已開放原本 AUTO 按鈕，
# 所以這裡必須明確設 SFM_EVALUATION_ONLY=1，把本場次鎖成定位量測、拒絕 AUTO。
# 該旗標只放行「地圖未驗證」這一條 localization blocker；digest 不符、
# camera profile 不符、收據缺漏照擋。flight.approved 在 evaluation-only
# 下仍是 false。手動搖桿飛行不經過那條合約。
export SFM_EVALUATION_ONLY=1
# 影格上限與品質。用河濱實拍量到的：1280x720 JPEG q90 平均 222 KiB，
# 2 GiB 約 9400 張；定位實測約 8 Hz，也就是大概 20 分鐘（一顆電池夠用）。
# 要飛更久就把 SFM_IMU_TEST_EVERY_N 調成 2，不要降畫質 —— EDM 是稠密比對，
# JPEG artifact 會直接汙染要評估的東西。
# 除了這兩個上限，錄製器還會盯著剩餘空間，掉到 disk_policy 的 critical 門檻
# （5 GiB / 5%）就自己停，不會去吃掉飛行安全與 session log 需要的餘裕。
export SFM_IMU_TEST_JPEG_QUALITY="${SFM_IMU_TEST_JPEG_QUALITY:-90}"
export SFM_IMU_TEST_MAX_FRAMES="${SFM_IMU_TEST_MAX_FRAMES:-20000}"
export SFM_IMU_TEST_MAX_MB="${SFM_IMU_TEST_MAX_MB:-2048}"
export SFM_IMU_TEST_EVERY_N="${SFM_IMU_TEST_EVERY_N:-1}"

cat <<'SOP'
============================================================
IMU 飛行測試 — 現場步驟
============================================================
 0. 這是 EVALUATION-ONLY 場次：地圖未經獨立驗證，定位只能當量測用，
    不可拿來導航。自動飛行仍然被飛行合約擋著（這是對的，不要繞過）。
    全程手動搖桿飛行。
 1. 介面開起來後，先確認影像有畫面、telemetry 有在跳。
 2. 按「開始定位」。等它從 BOOT 進到 TRACK。
 3. 用搖桿起飛，手動飛。這段要讓定位走過所有狀態：
      穩定 TRACK → 掃過難定位區觸發 WEAK_TRACK / LOST → 回到已知區域重新定位
    這些資料僅用於 fused-yaw 與視覺追蹤分析；目前沒有 ESEKF。
 4. 想要更高解析度的備份，可以另外在介面上開機上錄影；
    離線分析用不到它，imu_test/frames 已經是定位當下吃的那批畫面。
 5. 落地後正常關窗（不要直接 kill），錄製才會收尾寫出 summary.json。
============================================================
SOP

STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
STAMP_FILE="$(mktemp)"
trap 'rm -f "$STAMP_FILE"' EXIT
touch "$STAMP_FILE"

STATUS=0
bash "${ROOT}/一鍵啟動.sh" "$@" || STATUS=$?

# 找出這次跑出來的 session（比腳本啟動還新的那個 real-flight 目錄）。
SESSION=""
if [[ -d "$SESSION_ROOT" ]]; then
  SESSION="$(find "$SESSION_ROOT" -maxdepth 1 -type d -name 'session_*_real-flight_*' \
    -newer "$STAMP_FILE" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
fi

echo
echo "[IMU飛行測試] 開始時間（UTC）：$STARTED_AT  介面結束碼：$STATUS"
if [[ -z "$SESSION" ]]; then
  echo "[IMU飛行測試] 找不到這次的 session 目錄；請確認介面有真的連上真機並開始記錄。" >&2
  exit "$STATUS"
fi

echo "[IMU飛行測試] session：$SESSION"
if [[ -x "$PY" ]]; then
  "$PY" "${ROOT}/tools/imu_flight_test_report.py" --session "$SESSION" || true
else
  echo "[IMU飛行測試] 找不到 $PY；請自己跑："
  echo "  .venv/bin/python tools/imu_flight_test_report.py --session \"$SESSION\""
fi

cat <<EOF

[IMU飛行測試] 離線檢查（不連接飛機）：
  .venv/bin/python tools/imu_flight_test_report.py --session "$SESSION" --json
  現行系統只有 fused-yaw bridge，沒有 ESEKF on/off replay。
  原始 gyro/accel、尺度與外參不足時不能宣稱完成視覺慣性融合驗收。
  判讀方式見 docs/direct_offline_validation.md。
EOF

exit "$STATUS"

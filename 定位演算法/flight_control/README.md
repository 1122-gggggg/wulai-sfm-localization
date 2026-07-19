# ANAFI 自動飛行安全操作

真機唯一入口為 `path_follow_flight.py --fly`。起飛前必須完成 `--selftest`、`--dry-run`、Sphinx 模擬與拆槳實驗，並由人類安全員全程監看。

## Firmware 高度／距離上限

`--fly` 必須明確提供 `--max-altitude-m` 與 `--max-distance-m`；兩者會先和
firmware 公布的 min/max 比對，再逐項送出並回讀確認。距離圍欄預設以
`--distance-geofence` 開啟；起飛前另外要求電池至少 30% 且 GPS 已 fix。
任一狀態缺失或不一致都拒絕起飛。

```bash
python path_follow_flight.py --fly \
  --max-altitude-m <METERS> --max-distance-m <METERS> \
  --distance-geofence
```

距離圍欄只阻止飛越上限，不會自動返航。`NavigateHome` 是獨立的 RTH
操作；本程式不會因碰到 geofence 而自行呼叫它。

## 明確授權 AUTO

SafetySwitch 預設 fail closed：

- `SFM_SAFETY_FILE` 缺失時會建立內容為 `hover` 的檔案，不會自動寫入 `auto`。
- 既有的 `hover`、`manual`、`land` 或 `emergency` 不會被覆寫。
- 每次啟動 `--fly` 後，operator 確認場地、航線、串流與人工接管均就緒，必須在 30 秒內重新寫入一次 `auto`。前一次任務留下的 `auto` 視為過期，程式不會連線或起飛。

```bash
printf 'auto\n' > "${SFM_SAFETY_FILE:-/tmp/sfm_drone_safety.cmd}"
```

推論中切換 HOVER/MANUAL/LAND/EMERGENCY、串流中斷或終止信號時，每一筆自主 PCMD（包括零 PCMD）都會先經過同一把 authority lock 重新檢查。MANUAL 與 EMERGENCY 會完全停止 PCMD；`SIGINT`、`SIGTERM` 與 `SIGHUP` 由獨立安全監視線程執行停止與降落。終止動作只允許 `NONE → LAND → EMERGENCY` 單向升級；即使指令檔隨後變回 AUTO/HOVER/MANUAL，也不會取消已鎖存的動作。Landing／Emergency callback 若丟出例外或明確回傳 `False`，監視線程會限頻重試；只有 callback 成功才標記已執行，Emergency 永遠維持最高優先。

感知迴路只更新最新一筆已授權的 desired PCMD；`SafetyMonitor` 另以
`SFM_PCMD_CONTROL_HZ`（預設 20 Hz）送出，不會因同步定位推論而停止刷新。
若串流失效、切換 HOVER、停止信號或 watchdog 超時，desired command 會先
清成零；之後即使狀態恢復 AUTO，也不會重新送出舊的非零命令。指令 JSONL
同時保存 desired 更新與實際 PCMD 呼叫的 `monotonic_ns` 時間戳。

## BOOT 定位鎖定

起飛後只會懸停定位，必須在 25 秒內同時滿足：

- 連續 3 個 fresh Pose。Pose 時戳使用 PDRAW `ntp_raw_timestamp` 的 source-clock 變化對映至主機 monotonic clock，絕不使用推論完成時間。metadata 缺失時的 callback-receipt fallback 只供診斷／離線模式；真飛會標記 degraded、拒絕該 frame 並懸停。
- 每個 Pose 距航線起點不超過 `1.5` map-units（可用 `SFM_BOOT_START_MAX_U` 在實測尺度後調整）。
- 連續 fix 之間不可超過 pose-jump gate。

任一條件失敗會重置計數；逾時或 safety/stream 不健康時不進入 AUTO，並降落。

## 巡檢完成條件

無人機到達巡檢 waypoint 後會停住等待，不會因為「進入半徑」就標記完成。只有下列條件全部成功才會 acknowledgement：

1. 在 authority lock 下先送零 PCMD，並在整個阻塞式擷取期間持續保持零輸出。
2. 目前這一幀必須有可用的 Olympe body-yaw telemetry 與 map-frame 校正；機身 yaw 對準電桿水平 bearing，誤差不超過 6 度。
3. 依相機與電桿目標的相對高度計算 gimbal absolute pitch；target 在實際 bounds 內，且回讀的 `pitch_absolute` 與 target 相差不超過 3 度。
4. gimbal 確認後記錄 source NTP baseline，同時等待 host receipt 與 source clock 都前進至少 `SFM_INSPECTION_PIPELINE_DRAIN_S`（預設 1 秒；280 ms 只是白皮書下限），排除確認前曝光但延遲抵達的 frame；最後 frame 仍須不超過 0.5 秒。
5. JPEG 實際寫入 `sfm_system/定位/outputs/flight_inspections/`。

任一步失敗時 waypoint 保持 pending。對準或擷取超過 15 秒，以及抵達終點仍有 pending 巡檢，都會 fail closed 並降落；短暫抖出巡檢半徑不會重置 deadline。目前對應 15 點航線的有效巡檢標籤為 9、10、11、15。

低信心但仍有 fresh pose 時使用 `SFM_WEAK_HOVER_LAND_S`；完全沒有 fresh pose 時使用 `LOST_LAND_S`。WEAK 與 LOST 交替仍屬同一段 uncertainty，採期間遇過的較短降落門檻，不會重置計時。兩者都只會懸停並嘗試重新定位，不會拿不可靠姿態驅動飛行。

## 驗證

```bash
pytest -q sfm_system/定位/mission/flight_control/test_flight_safety_gates.py
python sfm_system/定位/mission/flight_control/path_follow_flight.py --selftest
python sfm_system/定位/mission/flight_control/path_follow_flight.py --dry-run
sfm_system/定位/sync_mirror_check.sh
```

## 電腦微移控制（手動方向鍵）

真機短脈衝 PCMD（非 AUTO 航線）使用 `manual_nudge_pilot.py`：

```bash
python manual_nudge_pilot.py --selftest
python manual_nudge_pilot.py --dry-run
# LIVE：先停掉其他 Olympe 連線；安全員握住搖桿
python manual_nudge_pilot.py --ip 192.168.53.1 --controller skycontroller3
```

安全行為：Ctrl-C / 關終端 / 關視窗 → 零 PCMD + Landing + 還原 SkyController 搖桿；
Esc / 手動 → 立即交還搖桿（電腦靜默）。驗證步驟見腳本頂部 docstring 與 operator_interface/README。

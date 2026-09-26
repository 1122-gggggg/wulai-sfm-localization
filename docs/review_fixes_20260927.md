# 系統檢查後的修正（2026-09-27）

對應 2026-09-26 的全面檢查。沒有連接飛機或發送飛行指令；結果來自離線測試與
模擬器。模擬通過不代表真機精度或抗風能力。

## 真機傳輸改回 Olympe

操作員決定繼續使用 Olympe。ARSDK-NG／RTSP 後端完整保留在分支
`backup/arsdk-ng-20260927`（commit `de27693`），主線移除 `arsdk_ng*.py`、
`rtsp_frame_source.py`、`tools/build_arsdk_ng.sh` 與 native libraries。
改回的主因是 ARSDK-NG 版未實飛過，且直連時斷線偵測失效：
`_probe_host_link_ok` 呼叫的 `connection_state()` 在真正的 client 上不存在，
例外被吞掉後改用不會因斷線而失敗的 `get_state()`，所以連線永遠顯示正常；
測試的假物件有這個方法，因此測試沒有抓到。

09-22～09-26 與傳輸無關的修正都保留：

- 控制用遙測歷史改由 Olympe `drone.subscribe()` 收集
  （`backend_telemetry.MotionEventHistory`，最多 512 筆），時間取 Olympe 事件
  物件的收件時間換算成主機 monotonic；未來、超過 5 秒或重複的樣本不收。
- GPS、衛星、訊號、雲台與變焦的唯讀輪詢留在 `backend_telemetry.py`；
  MAP_ITEM 狀態依 `gimbal_id`／`cam_id` 取主相機。
- 經 SkyController 時 RSSI 取自有效的機體清單項目，直連仍讀 `WifiSignalChanged`。
- 舊版 `autoflight`／`cruise_geofence` 的真機指令維持永久上鎖。
- 任務選擇檔的車輛元件雜湊改回 Olympe 版（`adapter: olympe`）。

## 航線模擬 600 秒飛不完

根因不是只有 REJOIN。接近航點的減速斜坡長度是航線長度的 25%，短航線的每一段
都在斜坡內，推力始終無法到滿；最後幾十公分以約 6 秒的時間常數爬行，期間被風帶離
航線，又觸發 REJOIN。`SLOWDOWN_ROUTE_FRACTION` 改為 0.075
（`real_path_follow_controller.py`）。

| 比例 | 預設航線 26 個起點 | 其他航線測試（陣風、延遲雜訊、視覺中斷） |
|---:|---:|---:|
| 0.25（原） | 13 通過 | 6 失敗 |
| 0.075 | 21 通過 | 1 失敗 |

也試過讓 REJOIN 的觸發寬度不小於 0.05u。它加快了完成時間，但追蹤 p95 變差，
和「p95 不超過 2 倍到站半徑」的驗收契約衝突，所以沒有採用。

仍未通過：09-23 航線（到站半徑 0.02u）五個起點都能在 600 秒內飛完（原本全部
逾時），但追蹤 p95 0.043–0.061u 仍略高於 0.04u 上限；0.2 m 陣風恢復仍超過
5 秒。這條航線的半徑小於實測重定位跳動（p50 0.03–0.05u），是否放寬由操作員決定。

## 其他

- UI：地圖／影像畫布最低高度 140 → 96px（`VIEWPORT_MIN_H`），路線面板間距縮小；
  1180×768 最小視窗下所有控制元件都能顯示，5 個版面測試通過。
- 品質門檻：ruff、format、mypy、維護性檢查全部通過；已結案的
  `執行環境/models/vista_slam/` 加入 `.gitignore`。
- 啟動腳本以 `-e .git` 判斷原始碼目錄，git worktree 不再被當成未封裝的 portable 內容。

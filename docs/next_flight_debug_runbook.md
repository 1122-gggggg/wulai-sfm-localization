# 下一次飛行：狀態、風漂移與定位除錯

2026-09-14。適用目前桌面介面的 direct 定位與 AUTO 巡航。修改只經離線測試，沒有代操作員起飛。驗收數字與版本見 [本次驗證報告](../outputs/analysis/next_flight_debug/report.md)。

## 螢幕上的狀態

「飛行狀態與航點事件」固定放在分頁上方，切換分頁仍能看到。第一行顯示飛機狀態、控制權與定位狀態；第二行顯示目前航點、動作、路徑誤差和已抵達航點。下方保留附時間的事件歷史，可以捲動查看。

動作包括：初始定位等待、修正朝向、轉向前減速、朝向確認、修正高度、前往第 n 個航點、抵達確認、已抵達、回到航段、近點位置／高度調整、等待可靠地圖定位、限速懸停、暫停／繼續、人工接管、失敗、終點確認和降落。AUTO 結束後保留最後結果。定位失效會取代先前快追蹤狀態，避免顯示過期成功訊息。

AUTO 一律從第 1 航點開始，依序往後飛；原有返程沿線回到第 1 點。轉向可搭配升降，兩者與水平平移分開。接近航點時速度會隨剩餘距離縮小（極限式減速），回線則保持完整修正權限。高度調整有進入／退出門檻，避免小幅量測雜訊反覆切換；水平風漂移變大時優先修復水平偏差。這是位置回授，沒有宣稱量測出風速。

## 自動留下的資料

正常啟動介面會建立 `outputs/flight_logs/session_*`。沿用既有四個 JSONL 檔，不需額外開啟 debug 開關。

| 檔案／事件 | 分析用途與主要內容 |
|---|---|
| `localization.jsonl` / `pose_result` | 原始視覺姿態、相機中心／朝向、定位模式、成功／拒絕原因、地圖與 VO 內點、重投影誤差、影像覆蓋率、影格與提交／回傳時間、處理延遲 |
| 同檔 / KLT 欄位 | 輸入點數、前向／反向有效點數、留下點數、影像內點數、前後向誤差 p50／p95、像素位移中位數、重播種、影格間隔 |
| 同檔 / IMU 欄位 | 是否使用橋接、使用／拒絕原因、當前與參考姿態訊號時間、訊號年齡、補償 yaw 增量 |
| 同檔 / `pnp_observation_sample` | 最多 128 組已接受的 2D 像素／3D 地圖點、點 ID、實際 K、影像大小、原始 world-to-camera 矩陣、取樣時間、總內點數 |
| 同檔 / `auto_route_plan` | 每次 AUTO 的 `auto_run_id`、航點座標、抵達半徑、地圖基底、路線 SHA、座標框架、實際控制設定與時限 |
| 同檔 / `auto_route_tick` | 每個控制 tick 的目標／引導點、位置與姿態、定位年齡與地圖確認、沿線與目標誤差、機體座標誤差、朝向目標／誤差／融合偏移、階段、確認次數、高度切換狀態、速度保護與無進度時間、要求與授權 PCMD |
| `telemetry.jsonl` / `fused_odometry` | 約每 0.1 秒輪詢記錄 firmware fused Euler、NED 速度、相對高度／AGL、GPS 與精度、飛行狀態、控制權、風警示、雲台角度／變焦，以及各來源的獨立時間 |
| 同檔 / `pcmd_dispatch` | 每次 SDK PCMD 呼叫的實際整數值 `[roll,pitch,yaw,gaz]`、`command_mono_ns`、序號。它代表 SDK 呼叫，不代表韌體已執行或飛機已產生該運動 |
| 同檔 / `autonomy_event` | 狀態轉換、抵達、暫停／繼續、失敗等事件，包含 AUTO run ID 與事件時間 |
| `commands.jsonl` / `incidents.jsonl` | 操作員指令、控制權、既有安全事件、worker／串流／紀錄異常 |
| `session_manifest.json` / `session_summary.json` | 執行環境、既有地圖／設定雜湊、來源程式雜湊、欄位單位契約、事件數與 deferred debug 丟棄數 |

PnP 對應點平常最多每秒取一組；重新定位時允許間隔至少 0.5 秒的額外樣本。這些是已接受內點的有界抽樣，可重算殘差、研究幾何條件與觀測雜訊，不能代表所有匹配或完整影像重播。要追查錯誤匹配／模糊，仍需保留原始影片及同次 session。

`body_velocity_mps` 記錄這次控制運算實際使用的 `[forward,right]` 速度，`body_velocity_stamp_mono` 為其來源時間。近點／回線的位置增益為水平命令上限除以 `2 × 抵達半徑`，速度阻尼為 `10 PCMD/(m/s)`；正常遠距前進維持原本的命令幅度，避免阻尼把整條路線拖慢。

SDK debug 先放進 512 筆有界佇列，由 telemetry 輪詢批次落盤；飛控發送執行緒不執行這項磁碟寫入。正常關閉時清空佇列，`debug_deferred_dropped` 報告容量丟棄。非有限值寫成 JSON `null`。異常斷電仍可能失去尚未落盤資料，不能把缺少紀錄當成沒有送過命令。

## KLT 與 IMU 修正

- KLT 用有已知位移的實際 OpenCV 影像追蹤測試核對方向與誤差；光流 API 沒有結果時拒絕該批追蹤，避免 worker 崩潰。
- IMU yaw 增量繞地圖 up 軸旋轉。PnP 的矩陣是 `world-to-camera`，修正其乘法側與方向，並保持相機中心不動；以不對齊地圖軸的相機姿態測試。
- 姿態採來源事件時間，不把持續輪詢的同一筆快取刷新為新資料。相同時間的訊號不能重複當成新量測；陳舊訊號會留下拒絕原因。
- IMU 橋接仍屬 WEAK，只補短時朝向。純 VO／橋接不單獨推進航點或累積抵達降落確認，長時間無法取得可靠地圖定位仍會停住。

## 為未來 Kalman 準備的解讀規則

這次沒有在 direct 路徑新啟用 Kalman 預測。現有朝向融合和短時視覺／IMU 橋接各自保留來源標記，之後才能做離線 A/B 比較。

1. `*_mono_ns` 是主機 monotonic 奈秒；`*_mono`、`pose_stamp`、控制 tick 的 `t` 是秒。先用影像擷取時間及來源時間對齊，再看提交／處理／落盤延遲。`t_mono_ns` 的寫入時間不能取代緩衝命令的 `command_mono_ns`。
2. 地圖位置／誤差用 u；NED 速度用 m/s；firmware yaw 是從北向順時針的弧度；地圖 heading 沿已記錄地圖基底逆時針。不得直接將 NED m/s 加到地圖座標。
3. `attitude_stamp_source` 區分事件時間與快取值變化的觀測時間。重複 sample、遺失及負時間差需單獨處理，不能用輪詢頻率當成感測器頻率。
4. 現有輸入是韌體已融合姿態與速度，沒有 raw gyro、raw accelerometer 或 FC covariance；不能憑空填成零雜訊，也不能視為完全獨立感測器。
5. 目前 river 地圖仍缺實測公尺尺度與核准的相機／機體外參。K、地圖 up、來源 hash 與缺失狀態已可追查；平移 IMU 融合前仍需實測這些標定。雲台／變焦變化也要納入觀測模型。
6. 視覺連續幀、VO 延伸點、IMU_BRIDGE 與先前狀態有相關性；橋接結果不是新的絕對地圖量測。未來估計 Q/R 需區分 FAST_TRACK、RELOC_SEED、VO_ONLY、DEAD_RECKON、IMU_BRIDGE 和無定位，保留拒絕樣本的原因。

## 飛行後產生報告

確認落地後，正常關閉並重新啟動介面才會載入這次程式。飛行後保留完整 session 目錄；在專案根目錄執行：

```bash
.venv/bin/python tools/flight_debug_bundle.py outputs/flight_logs/<該次session目錄>
```

會輸出資料完整度與轉向診斷，並建立 `debug_bundle.json`。同一次 session 若啟動多次 AUTO，請分別看 `auto_runs`，各航點另列在 `targets`；不要把人工接管的空檔算成巡航時間，或將正常換航段的方向改變當成定位漂移。終點抵達與修正高度／回線也會分開標記。GPS 位移本身不能證明風漂移，仍須對照控制權、命令、地圖軌跡與定位品質。

重跑三條預設路線與風漂移測試：

```bash
.venv/bin/python tools/verify_preset_routes.py \
  --flight-logs outputs/analysis/preset_route_acceptance/inputs \
  --out outputs/analysis/next_flight_debug/rerun
```
`--quick` 只驗證每個航點附近起飛但一律從第 1 點開始的基本情境；完整模式另外產生 `wind_results.json`。陣風測試以間隔性的指定公尺位移模擬擾動，並非已辨識的真機空氣動力模型。

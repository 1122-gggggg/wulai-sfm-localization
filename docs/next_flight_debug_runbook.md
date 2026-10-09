# 下一次飛行：狀態、風漂移與定位除錯

2026-10-03 文件整理：同步影格已在一般真機定位場次預設錄製，資料內容與可選的
人工 evaluation-only 流程統一見 [EKF 資料錄製與收集清單](next_flight_ekf_capture_20260922.md)。
收集不啟用實驗性 EKF／滑動視窗控制，也不更改既有 AUTO 授權。

2026-09-14。適用目前桌面介面的 direct 定位與 AUTO 巡航。修改只經離線測試，沒有代操作員起飛。驗收數字與版本見 [本次驗證報告](../outputs/analysis/next_flight_debug/report.md)。

## 下次飛行：場域尺度預設自動估計

落地後關閉舊介面，再用專案根目錄的 `./一鍵啟動.sh` 啟動。連接真機串流並開啟
定位後，`SessionMapScale` 自動累積證據，不需要另跑 `IMU飛行測試.sh` 或啟用
實驗性 EKF；操作員手動飛行也能估計，不必先按 AUTO。起飛與 AUTO 仍由操作員授權。
目前預設 mission selection 已綁定 7 點河濱路線；同場域、同座標系的有效路線
可從介面選用 AUTO，不需逐條靜態核准。仍須通過定位授權、路線驗證與原有
preflight；更改設定後須在確認落地後重新啟動介面。

公尺尺度來自帶事件時間的韌體融合 NED 速度積分，姿態用來對齊地圖方向，再與
strong 視覺地圖位移比較；不是 IMU 姿態本身提供公尺尺度，也不是 raw accel 積分。
所有地圖座標使用同一個 `m/u`，只新增公尺換算，不改寫原始地圖或航線。
HUD 顯示「全場尺度候選」時不可視為可信；「全場尺度 …（可用）」才是通過門檻：
至少 8 個獨立區間、累積至少 10 m，每段至少 1 s／1 m／0.02 map units、
最長 4.5 s，方向差不超過 20°，比例分散與前後漂移各不超過 20%。
weak／IMU_BRIDGE／reseed／過期或缺失遙測不教尺度；30 s 沒有新合格證據就撤回
可用狀態，換地圖或新 session 重新估計。不要為了湊尺度距離而改變原有安全飛行計畫。

每筆 `localization.jsonl` 的 `map_scale` 記錄候選、可信狀態、區間數、距離與拒絕原因；
只有可信時才提供 `metric_pose_m`。離線介面路徑驗證使用目前預設任務的地圖基底，
合成 10 m/u 軌跡收斂為 10 m/u；弱定位不收斂、30 s 到期不再換算、新 session
不繼承尺度，結果見
[`scale_startup_smoke.json`](../outputs/sim_autoflight/anafi_fidelity_20261003/scale_startup_smoke.json)。
這不是實飛精度認證，仍須用下次真機紀錄核對收斂與偏差。

2026-10-04 實飛核對：17 個合格區間、累積 17.14 m，最終候選
`7.616409478 m/u`，相對 IQR `42.98%`，仍因 `scale_iqr_unstable` 不可靠，之後
因沒有新合格證據而 `scale_stale`。`velocity_not_new` 是異步速度事件的時戳
去重，不應把每次略過都當成速度封包遺失；也不能複製舊速度戳製造新證據。
路線編輯器可讀取獨立、綁定 map hash/frame 的 `map_distance_scale.json`
做兩點及各段公尺**估算**；這不繼承或放寬線上可信尺度，不啟用飛控公尺判定。
詳見 [量距操作與限制](../控制介面程式/README.md)。
本場次的 `telemetry.jsonl` 與 `imu_test/frames.jsonl` 沒有實測加速度或陀螺儀
樣本；`session_manifest.json` 亦明列 raw accelerometer／gyro 不可用。尺度已納入
事件時戳地速的梯形積分，不把速度差分加速度當成另一個獨立感測器。要加入
真正 IMU 加速度，還需原始樣本、單位／軸向、姿態與重力扣除、bias 校正及
可對齊的感測時戳；資料缺失時不產生假融合結果。

## 2026-10-04 飛行紀錄驅動修正

- 速度限制器改為加速漸增之後的最後水平包絡；不再把預測超速所需煞車縮回
  低速阻尼，受限命令回饋漸增狀態。真實請求／速度的固定輸入重播中，違反
  預測煞車包絡的 31 筆命令降為 0；這不是該場實飛速度的反事實預測。
- HUD 快取每列／字體的最新文字寬度與固定字體高度，略過垂直空間不可能容納
  的版面。實際 Tk 的受控完成事件驗證，配對 UI polling p95 約 57.27 → 15.70 ms；
  最終整合 smoke p95 7.70 ms、最大 139.37 ms，仍保留首繪與排程長尾。
  原實飛 383.32 ms 的單一等待不能僅靠區間 p95 完整歸因於 HUD。
- 尺度以曝光時刻的姿態配對保存每個端點的 map／NED offset。弱、缺失、bridge、
  reseed 或視覺缺口切斷校準區間；合格但矛盾的比例仍納入 IQR／漂移，不能
  只剔除不符合候選的比例來製造可信度。8 段／10 m／20%／30 s 原門檻不放寬。
  事件批次重播的新候選約 8.5928 m/u，僅 6 段／6.04 m、IQR 32.36%、
  漂移 30.28%，仍不可用。此重播與前述逐列融合分析的攝取方法不同，不能將
  7.6164 → 8.5928 當成精度改善；編輯器的歷史估算側檔沒有被新候選覆寫。
- 新增／編輯路線的地圖右上角有＋／－。兩點量距球可單擊選 A／B、`G` 開始
  移動、`X／Y／Z` 鎖軸、左鍵／Enter 確認、Esc 取消；確認與取消都不改正式航點。
  實際 Tk、河濱地圖的鍵鼠操作已驗證，量距隨預覽移動更新。
- 最新七點去回航線的同 seed 配對，0.3 m/s 設定下完成率 5/6 → 6/6，
  六案地速 p95 皆下降；但橫向半徑驗收仍為 4/6，高延遲陣風與慢響應未通過。
  陣風失鎖案耗時 272 → 774.05 秒，高延遲陣風峰值 0.856 → 0.944 m/s，
  不能據此宣稱整體飛行更快、所有峰值下降或抗風已驗收。
- 修正舊 headless fixture 的測量移動狀態，移除跨弱定位洞仍應產生可信尺度的
  過時案例，改由弱定位切斷區間的安全回歸涵蓋。相關編輯器、量距與尺度
  回歸共 146 項通過，Ruff 通過；實際 Tk 的縮放、三軸移動與撤銷亦通過。

離線證據：
[`scale_alignment.json`](../outputs/flight_optimization_20261004/scale_alignment.json)、
[`operator_latency.json`](../outputs/flight_optimization_20261004/operator_latency.json)、
[`measurement_move_smoke.json`](../outputs/flight_optimization_20261004/measurement_move_smoke.json)、
[`route_response_comparison.json`](../outputs/flight_optimization_20261004/route_response_comparison.json)、
[`measurement 畫面`](../outputs/flight_optimization_20261004/route_measurement_move.png)。
控制敏感度與未通過條件另見 [抵風控制文件](../定位演算法/flight_control/WIND_CONTROL.md)；
離線軟體修正不保證真機抗風、瞬時 0.3 m/s 硬上限或獨立公尺標定。

### 最新場次 `075845Z`：四項離線優化

本段使用 `session_20261004T075845Z_real-flight_9c1c0c03` 的六點去回航線與實際 JPG，
不同於上節七點航線。場域 profile、0.3 m/s 設定、到站與幾何門檻均未改。

- **速度**：正向 PCMD 額度再受既有 30 PCMD/(m/s) 阻尼乘上剩餘預測地速空間限制，
  避免 50% 權限把微小速度差放大成大幅加速。最後速度包絡、既有更強煞車、
  側向抵風、yaw／gaz 與缺失速度的處理保留。無風、同 seed 的三種完整航線
  模擬均完成；典型 τ 1.4 s／延遲 0.4 s 的地速 P95 0.457 → 0.183 m/s、
  超速時間 36.25 → 2.20 s，但耗時 182.8 → 217.4 s。
  τ 2 s／延遲 0.5 s 的峰值反而 1.823 → 1.868 m/s，三案的航線半徑 P95
  均未驗收通過。另一個限制新側向增長的候選在 1200 s 未完成，已撤回。
  模擬 TTL 是 1 s，不是正式 profile 的 150 ms；不是機體動態或抗風認證。
- **重定位**：同一 capture 內重用每張參考的 lift 證據，仍在合併後依原規則去重。
  強 PnP 但無足量稀疏 seed 不再提前停止原有 top-2+2 搜尋；最終門檻不放寬。
  34 張實拍前後的定位判定／姿態／seed 逐位元一致；失敗求解的配對 median
  約 407 → 383 ms。返程仍 31/32 `ABSTAINED`，另一次 strong 仍無可用 seed。
  兩張失敗影格另查排名 25–64 仍無可用錨；不能宣稱缺少反向影像或全圖無解。
- **尺度**：無效姿態／速度事件切斷插值、位移與預測連續性；曝光端點 FIFO 配對，
  不能跳過無法包夾的舊端點而把兩側接成合格區間。已知 12.7 m/u、
  非同步遙測及連續轉向的 smoke 正確收斂；無效來源不再製造 reliable。
  實際 3728 筆姿態收件重播仍為候選 9.143 m/u、IQR 59.67%、reliable 0 筆，
  沒有替實飛資料捏造可信公尺尺度。
- **畫面**：視角與尺寸不變時沿用點雲照片，航線與軌跡用有界保留式 Canvas；
  一般地圖更新最短 50 ms，視角、弱估計失效與 gimbal 變化不等待此節拍。
  移除影片／地圖互斥輪替，影片縮小先整數 box reduction，再 bilinear。
  相鄰實際 Tk 配對的繪製 P95 5.66 → 3.77 ms、影片約 19.8 → 29.6 Hz。
  兩邊共用新的完整內容／視角 dirty key，不能把此收益算入上述配對。

受影響離線回歸 2628 通過、1 跳過；型別補強後的受影響子集另有 324 通過，
Ruff 與既有 21 檔 mypy 邊界通過。實拍 native 重定位／40 幀快追蹤、尺度重播、
完整航線模擬與實際 Tk 的縮放／平移／告警亦已執行；未連接或操作飛機。
證據：[整合收據](../outputs/analysis/flight_optimization_latest_20261004/final_verification.json)、
[速度配對](../outputs/analysis/flight_optimization_latest_20261004/speed/isolated_summary.json)、
[尺度重播](../outputs/analysis/flight_optimization_latest_20261004/scale/final_smoke.json)、
[UI 配對](../outputs/analysis/flight_optimization_latest_20261004/ui/comparison.json)。




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
5. river 地圖仍缺獨立實測公尺標定與核准的相機／機體外參。線上韌體融合速度與
   視覺位移可被動估計尺度，但候選值或一次收斂不取代獨立標定。K、地圖 up、
   來源 hash 與缺失狀態已可追查；raw IMU 平移融合仍需標定，雲台／變焦變化也要
   納入觀測模型。
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

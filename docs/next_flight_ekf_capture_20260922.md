# 下次一般飛行的 EKF 資料錄製

2026-09-22 依操作員要求，一般 `real-flight` 場次預設開啟同步資料錄製。
按原本方式開啟介面並啟動定位即可，AUTO／人工飛行均可收集；不會為了錄製
更改任務授權或起降控制，也不會啟用 EKF。

2026-10-03 文件整理：本頁合併原 2026-09-21 人工收集清單與已退役 ESEKF
操作頁中的仍適用流程；目前資料收集與報告以本頁為單一入口，歷史實驗結論仍保留在
[能力稽核](imu_odometry_capability_audit.md)與[優化總帳](verified_localization_optimization_ledger.md)。

## 相機參考點與內參

- 依操作員約定，以相機中心代表無人機位置，平移偏移近似為零。
  這個假設會寫入場次快照，不冒充實測外參。
- 相機光學座標、機身 NED／FRD 姿態仍分開記錄；相機朝向不直接等同機身朝向。
  新增雲台絕對／相對 pitch、yaw、roll 及其來源時間戳，供後續驗證旋轉關係。
- [intrinsics.json](../intrinsics.json) 的 1280×720 尺寸與 K，已與目前河濱
  PINHOLE 設定核對一致。完整檔案、五個畸變係數及 SHA-256 都複製進每次紀錄。
- 主機目前做影像縮放，未套用這份畸變係數；PINHOLE 管線假定輸入已校正。
  2026-09-22 操作員確認標定來源為未去畸變的 ANAFI 原始串流；
  `intrinsics_provenance.json` 將此確認綁定到內參 SHA。新場次的
  `distortion_domain_verified=true` 只代表來源已確認，
  `host_rectification_applied=false`、`pinhole_image_domain_verified=false` 仍明示
  正式管線尚未校正。去畸變候選先以離線 capture 比較，避免誤用到其他相機或重複校正。
- 已修正 `NavigationPoseTransformer` 將公尺外參偏移混入地圖單位後再縮放的
  問題，並用 0.2、2.5、7.0 倍尺度回歸測試；本次沒有建立假的核准外參。

## 自動保存的內容

輸出在 `outputs/flight_logs/session_<UTC>_real-flight_<id>/`。

| 檔案 | 內容 |
|---|---|
| `imu_test/frames/*.jpg` | 成功提交給定位器的影格之 JPEG 副本 |
| `imu_test/frames.jsonl` | 影格序號、capture 時間、NTP 原始時間與主機映射、接收／前處理時間、該次提交的遙測快照、尺寸、檔案雜湊及原始 RGB 雜湊 |
| `imu_test/intrinsics.json` | 操作員提供的完整相機標定原檔 |
| `imu_test/site_profile.json`、`localizer_profile.json`、`map_align.json` | 當次相機模型、定位設定及地圖重力對齊快照 |
| `imu_test/capture_contract.json` | 標定雜湊、相機中心近似、座標和單位、內參匹配結果、未測尺度／真值及未啟用 EKF 的明確標記 |
| `imu_test/summary.json` | 已保存數量、丟幀、容量停止、編碼／寫入／關閉錯誤 |
| `imu_test/report.json`、`report.md` | 正常關閉後自動產生的資料完整性及重播準備報告；執行訊息在 `report.log` |
| `trajectory.jsonl` | 永久保存原始／處理後姿態、相機軸、地圖／VO 支持、品質、有限 PnP 對應樣本、各階段時間、飛控遙測、搖桿、實際 PCMD 下發及 AUTO 軌跡 |

原本的 `localization.jsonl`、`telemetry.jsonl`、`commands.jsonl` 等仍照常記錄。
診斷檔輪替後，報告會從 `trajectory.jsonl` 讀回必要資料，避免留下影格卻遺失
速度、姿態及時間戳。來源版本雜湊也已納入估計器、航向融合、影像來源及錄製程式。

每次提交保留速度、姿態、高度各自的來源時間；不會用新的姿態時間刷新舊速度。
實際變焦回讀 `camera_zoom_level` 與 UI 設定 `zoom` 分開，便於辨識遙控器造成的
相機變化。SDK 不提供的欄位維持 null；輪詢次數不當成獨立感測樣本數。

## 錄製範圍與收尾

錄製採背景編碼／寫入、四格有界佇列，預設 JPEG 品質 90、20,000 張與 2,048 MiB
上限；磁碟餘量不足時停止錄製。佇列滿時丟錄影影格並記錄數量，不阻塞飛行 UI。
JPEG 是有損副本，離線 A/B 必須共用同一份輸入，不能宣稱與當時 RGB 像素完全相同。

保持正常相機設定，依正常操作飛行與關閉介面即可。沒有啟動定位時，不會產生
定位提交影格；報告會明確指出沒有可用定位資料。正常關閉會先收尾錄製，再啟動
獨立的離線報告程序；異常斷電／強制終止可能需要手動重建報告。

若環境曾設定 `SFM_IMU_FLIGHT_TEST=0`，會明確停用錄製；未設定時真機預設啟用。
`SFM_IMU_FLIGHT_TEST=1` 可讓其他模式也錄製。這個旗標不改變飛行授權。

手動重建報告（唯讀原始資料，不連飛機）：

```bash
.venv/bin/python tools/imu_flight_test_report.py \
  --session outputs/flight_logs/session_實際場次 \
  --out outputs/flight_logs/session_實際場次/imu_test/report.md \
  --json-out outputs/flight_logs/session_實際場次/imu_test/report.json
```

報告的 `UNUSABLE` 代表缺少關鍵資料；`USABLE WITH GAPS` 必須保留缺口與
丟幀資訊，只用可配對部分。`USABLE` 只表示資料契約可用，不代表 EKF 通過驗收。

## 可選的人工 evaluation-only 收集

操作員也可在專案根目錄執行 `./IMU飛行測試.sh`。此入口設定
`SFM_IMU_FLIGHT_TEST=1`、`SFM_EVALUATION_ONLY=1`，並拒絕 AUTO；不要移除
evaluation-only 來做這次收集。這不是讓實驗性 EKF 或滑動視窗接管飛機。
AI 不得執行真機入口或代為起飛；操作員依正常 preflight 親自在 UI 起飛，
起降與接管以 [SAFETY.md](../控制介面程式/SAFETY.md)及
[AGENTS.md](../定位演算法/AGENTS.md)為準。

- [ ] 起飛前確認影像、定位、遙測持續更新，錄製沒有磁碟／佇列錯誤。
- [ ] 維持正常相機設定，盡量固定雲台角度與變焦，並記錄設定。
- [ ] 場地及電量允許時，人工保留短暫懸停、平緩直線移動與停住、另一方向的
  平移及平緩轉向，供區分雜訊、延遲、尺度與方向對齊。
- [ ] 自然遇到弱定位或失鎖時保留紀錄，依正常操作回到可定位區；不得為製造
  LOST 而冒險飛出可控範圍。
- [ ] 若能獨立量測地標距離，記下量測值、兩端地標與對應時段供尺度交叉檢查；
  不要把 profile 的 `map_scale=2.3444` 當公尺換算。
- [ ] 正常降落後關閉介面，讓錄製完成寫檔；保留完整 session 目錄，包括
  `session_manifest.json`、`session_summary.json`、`commands.jsonl`、
  `incidents.jsonl`、原始定位／遙測及 `imu_test/`。機上 MP4 可作備份，
  不能替代帶主機 capture timestamp 的同步影格。

## 下次回來可以做什麼

先看報告的 `pose_telemetry_replay_ready`、`image_replay_ready`、配對影格數、各來源
獨立時間戳、來源時間差及丟幀。資料足夠時即可開始鬆耦合 EKF 離線整合與 A/B，
包含短暫掉點、旋轉、重定位跳變、資料延遲和拒絕錯誤量測。

**自動錄製不能補出獨立真值或實測尺度。** 仍需核對地圖與公尺尺度、NED→map
旋轉、高度零點及相機觀測模型，並以未參與調參的量測／場次評估誤差。
相機中心近似省略平移外參補償，不會自動解決這些問題。

EKF 還未運行，因此目前沒有 EKF 的協方差或創新量；報告將它和「資料漏錄」分開。
後續加入濾波器時再記錄狀態、不確定度、創新量、接受／拒絕原因、重置與輸出年齡。
正式控制仍需離線 A/B、故障注入、旁路時序驗證及操作員控制下的整合測試，
`formal_control_ready` 不會因收集完成就變成 true。

## 本次驗證

- 錄製、完整性報告、內參快照、事件配對、日誌輪替、雲台／變焦、外參單位、
  介面切換／關閉及飛行安全相關測試：636 passed、1 skipped（缺少測試場域對齊檔）。
- 端到端測試實際啟動離線報告子程序，核對 JSON 與 Markdown 生成，並驗證
  診斷日誌清理後仍可由永久紀錄配對影格。
- 本機合成 1280×720、20 Hz 錄製測試：40／40 張寫入，零佇列／編碼丟幀，
  capture 呼叫 p95 約 0.93 ms。此為單次合成負載，不是真機全系統延遲保證。
- 原始資料、測試紀錄與標定核對輸出保存在
  `outputs/analysis/ekf_capture_preparation_20260922/`。未連線或操作飛機。

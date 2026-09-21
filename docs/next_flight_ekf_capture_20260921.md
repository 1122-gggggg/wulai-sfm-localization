# 下次飛行：EKF 同步資料收集清單

日期：2026-09-21。用途是補足視覺＋飛控遙測融合的驗證資料。
**這是人工飛行的資料收集，不是讓實驗性 EKF 或滑動視窗接管飛機。**
目前正式 DIRECT 沒有完整 EKF；新原型只在離線影片上執行。

## 現場怎麼做

由操作員在專案目錄執行：

```bash
cd /home/allen/localization
./IMU飛行測試.sh
```

此入口會開啟同步影格錄製，設定 `SFM_IMU_FLIGHT_TEST=1`、
`SFM_EVALUATION_ONLY=1`，並拒絕 AUTO。不要移除 evaluation-only 來做這次收集。
AI 不得執行這個真機入口或代為起飛；操作員依正常 preflight，親自在 UI 起飛。
舊 launcher 的文字若提到搖桿起飛，以目前
[SAFETY.md](../控制介面程式/SAFETY.md)／[AGENTS.md](../定位演算法/AGENTS.md) 為準。

- [ ] 起飛前確認影像、定位與遙測有持續更新，錄製沒有磁碟／佇列錯誤。
- [ ] 使用正常相機設定，這次盡量維持相同雲台角度與變焦；記錄其設定。
- [ ] 場地及電量允許時，人工保留：短暫懸停、平緩直線移動與停住、另一個方向的
  平移、平緩轉向。這些片段用於區分雜訊、時間延遲、尺度與方向對齊。
- [ ] 自然遇到弱定位或失鎖時保留紀錄，再依正常操作回到可定位區；不用為製造
  LOST 而冒險飛出可控範圍。
- [ ] 若場地有可獨立量測的地標距離，記下量測值、兩端地標及對應時段，供尺度
  交叉檢查；不要把 profile 的 `map_scale=2.3444` 當公尺換算。
- [ ] 正常降落後關閉介面，讓錄製器完成寫檔；保留整個 session 目錄。

## 必須保留的檔案

輸出位置：`outputs/flight_logs/session_<UTC>_real-flight_<id>/`。

| 檔案 | 用途 |
| --- | --- |
| `imu_test/frames/`、`imu_test/frames.jsonl` | 定位當下的影像、capture timestamp、當幀掛載的 fused sample；重播的主要來源 |
| `imu_test/summary.json` | 總影格數、佇列丟幀、編碼失敗、容量上限與停止原因 |
| `localization.jsonl`、`trajectory.jsonl` | 原始視覺姿態、map／VO 支持、重投影品質、capture／完成時間、控制軌跡 |
| `telemetry.jsonl` | NED 速度、姿態、高度、各來源獨立時間戳、GPS 狀態、雲台／變焦、搖桿紀錄 |
| `session_manifest.json`、`session_summary.json`、`commands.jsonl`、`incidents.jsonl` | 場域及程式版本、資料完整性與接管／異常事件 |

錄製預設 JPEG 品質 90、每筆提交影格、上限 20,000 張／2,048 MiB。
碰到容量或磁碟餘量門檻會停止錄影，實際長度以 summary 為準。
機上 MP4 可另外保留作備份，但不能替代帶主機 capture timestamp 的同步影格。
JPEG 仍是有損影像，離線比對須讓所有候選使用同一份輸入。

## 落地後立即檢查

正常關窗後 launcher 會自動產生資料可用性報告。也可手動執行以下唯讀命令，
將路徑換成該次 session：

```bash
.venv/bin/python tools/imu_flight_test_report.py \
  --session outputs/flight_logs/session_該次場次 --json
```

- `UNUSABLE`：缺少關鍵資料，先看明確原因，不能拿來宣稱融合有效。
- `USABLE WITH GAPS`：保留缺口與丟幀資訊，離線實驗只使用可配對部分。
- `USABLE`：資料契約可用，**不等於 EKF 已通過**。

## 回來要驗證的 EKF 部分

1. 逐影格配對曝光時間及遙測來源時間。`frames.jsonl` 的共同
   `fused_telemetry_mono` 不足以證明速度新鮮；要另外核對 `telemetry.jsonl` 的
   `ground_speed_mono_ns`、`attitude_mono_ns`、`altitude_mono_ns`，不能刷新舊資料時間。
2. 用實際移動片段驗證公尺／地圖尺度、NED-to-map 方向、相機與機身參考點，
   同時檢查雲台／變焦變化。不能只憑一個短片段擬合出的比例當作校準完成。
3. 建立鬆耦合「PnP＋firmware fused velocity／attitude」的離線 A/B；
   不把現有遙測當成 raw accelerometer／gyroscope，也不估計不存在的 IMU bias 量測。
4. 比較弱定位及短時失鎖的恢復、重定位跳變、延遲與不確定度；純預測幀單獨計數。
   確認改善後才另行規劃旁路現場觀察，正式控制來源仍維持既有流程。

目前仍沒有獨立公尺位置真值。上述資料可以驗證時序與相對一致性；若要宣稱位置
精度提高，仍需獨立量測。原始 gyro／accel 與核准外參也不會因開啟錄製而自動出現。

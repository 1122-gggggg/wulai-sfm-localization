# 2026-09-22：錄製、同步、延遲與終點收斂修正

依操作員「修正 1、2、4、6」的要求實作。保留工作區原有修改，所有驗證均為
離線單元測試、合成事件迴圈或模擬，沒有操作飛行設備。

## 錄製完整性

`frames.jsonl` 由背景 writer 唯一負責關閉。`close()` 等待逾時後不再提早關閉
仍被使用的索引，也不等待被 writer 佔用的 summary lock。摘要保留 incomplete、
writer 存活／逾時及錯誤資訊；writer 稍後完成時更新摘要並保留逾時證據。
資料可用性報告將此類場次標記為 USABLE WITH GAPS，舊版摘要仍可讀取。

錄製影像改為持有獨立副本，避免 contiguous RGB 輸入被上游重用後改掉內容；
每筆錄製也保留 attitude、speed、altitude 各自的來源時間與時間來源標記。

## 獨立遙測時間戳

新增 SFM5 訊息格式，分別傳送影格、姿態、速度與 GNSS 來源時間；worker 繼續
支援既有 SFM3／SFM4。不同但各自有效的姿態與速度時間戳都會保留，超出原有
0.15 秒影格同步範圍、非有限或位於未來的訊號分別剔除。未取得獨立速度時間戳
的 client 資料不再借用姿態時間。GNSS 維持獨立的新鮮度規則，GPS 高度不是氣壓高度。

FusedTelemetry 保留既有 stamp 欄位供目前 yaw bridge 使用，另外提供
attitude_stamp 與 speed_stamp；沒有新增 EKF 或改變弱定位續飛行為。
回歸涵蓋姿態過期但速度有效、不同來源時間、舊格式及連續兩組 header/body 傳輸。

## 終點控制

正常航線最後一點通過位置確認、但地速尚未穩定時，原本將 LAND 改成 HOVER，
這也重置了位置控制器累積的抵風修正。現在沿用 FINAL_HOLD，保持 yaw 鎖定及
既有位置控制，直到原有地圖位置、新鮮地速 <= 0.10 m/s、至少 3 個速度樣本與
1 秒確認均通過。日誌記錄實際 FINAL_HOLD 狀態，不將等待過程記作 LAND。
人工 HOVER／LAND／EMERGENCY、失鎖處理、命令上限與接管流程不變。
FINAL_HOLD 也會檢查既有 AUTO wait budget；逾時歸零並退出 AUTO，由操作員
處理，不能靠長期缺失的速度資料繞過等待限制。

新增 `test_terminal_wind_settling.py` 以兩個機頭方向、正負 0.3 m/s 固定擾動及
無擾動驗證終點控制。修改前 4 failed、2 passed；修改後 6 passed。
模型使用 15 m/map-unit、0.4 秒一階響應及 0.018 m/s/PCMD，屬合成反例，
不是實測場域尺度或經辨識的 ANAFI 動態。

完整航線使用 `outputs/analysis/review_20260921/candidate_drift03_600/summary.json`
保存的全部模擬參數，正式控制核心、seed 7、600 秒預算。基準為本輪開始時的
工作區，已包含先前的抵風積分修正。

| 情境 | 基準 | 本次修正 |
| --- | --- | --- |
| 固定水平擾動 0.3 m/s | 本輪重現：600 秒未完成，6/7 點，路徑 113.69 m | 224.4 秒完成，7/7 點，路徑 58.97 m |
| 無擾動 | 9/21 保存結果：217.8 秒完成，7/7 點，路徑約 53.2 m | 本輪量測：226.2 秒完成，7/7 點，路徑 55.56 m |

擾動案例完成時：真值地速 0.0667 m/s，供控制器判斷的延遲遙測為 0.0496 m/s，
6 個速度樣本涵蓋 1 秒。無擾動情境略慢且路徑較長，不能宣稱所有指標都改善。
模擬完成不等於實飛驗證。

## UI 結果通知

定位 worker 的 readable-fd 通知仍只排程一次結果處理，改用 `after(0)`，避免
持續的 timer／render 事件餓死 `after_idle`。不在 fd callback 內重入更新姿態；
若等待期间 localizer 已替換或移除，丟棄該次舊通知。

使用真正的 Tcl 事件迴圈、30 個連續排程且各耗時約 4 ms 的合成忙碌 callback，
同一結果通知的等待時間：修改前 123.48 ms、修改後 4.32 ms。
這是排程反例的前後量測，不是影片繪圖、GPU 重定位或實飛端到端效能。
UI Skills 的效能指引用於要求先重現與量測，再做最小排程修改。

## 重定位交接與耗時

worker、影格 ring 與離線 ScheduledRelocalizer 共同保留 capture timestamp、
ordinal 及 reset 世代。交接時核對來源、ring 首尾與時間單調性，拒絕跨 reset、
倒退或不一致的結果。保留原本 48 幀記憶體上限，沒有任意增加秒數硬閘。
capture age 與結果來源資訊一路傳入 adapter、worker 輸出和 localization 日誌。

離線 capture 的每筆 reloc job 現在保存 provider 已產生的 stage_ms，方便區分
檢索、匹配／lift 與 PnP 成本。依 GPU 效能指引做暖機與 CUDA 同步後，以 P172
第 238 幀單次 probe 得到 wall 254.854 ms，其中 match/lift 212.946 ms、檢索
13.104 ms、PnP 28.107 ms。原來的 1613 ms 未重現，負載和重播條件不同，原因
未確認；本輪沒有宣稱 GPU／EDM 加速，也沒有改 batch、cache 或凍結 profile。
可重跑命令與當次結果見 [stage probe](direct_reloc_stage_probe_20260922.md)。

## 驗證範圍

最終整組回歸包括 `tests/localization/deploy`、飛控安全閘門、終點／持續擾動、
航段／高度控制、桌面 AUTO、worker lifecycle、遙測指標、錄製器及報告、UI
排程、關閉流程與離線模擬：916 passed、1 skipped（49.64 秒）。跳過項目缺少
URAI alignment 資產，並非硬體驗證。修改檔案的 Ruff 及 `git diff --check` 通過。
未重跑全專案維護性閘門；前次檢視已確認存在既有複雜度／檔案長度超標。

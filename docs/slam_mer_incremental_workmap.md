# SLAM-MER 概念的增量工作圖原型

這次實作採用 SLAM-MER 的增量圖優化概念，在既有 EDM／KLT／PnP 系統內提供
可選的 iSAM2 後端。它保留原本的雙視角三角化；目前沒有接入 MASt3R 深度、ALIKED
地標描述子或論文的 3D cell 查詢，也不宣稱重現整套 SLAM-MER 的效能。

## 已實作的行為

- `online_workmap.optimizer` 可選 `local_ba` 或 `isam2`，省略時保留原 local BA。
  iSAM2 持續保留同一 session／source epoch 的關鍵幀與真實影像觀測，新增因子時
  不重複計入既有觀測。相機內參固定，Pose3 座標轉換明確區分 camera-to-world 與
  world-to-camera。
- 舊圖 3D 點固定；具有既有 strong-inlier 數量的舊圖錨定關鍵幀保持原姿態。
  缺乏尺度支持時，保留第一、第二個具非零基線的規範姿態。固定相機與固定舊點
  之間的常數殘差只參與驗證，省去無效的求解工作。
- iSAM2 使用 QR 分解。P199 的實際快照在 Cholesky 下出現數值求解失敗；同一快照
  使用 QR 可完成求解。它不是額外的位置或深度量測。
- `online_workmap.retain_valid_tracks=true` 時，強舊圖姿態下仍符合既有三角化
  重投影門檻的新點可繼續 KLT 追蹤，為後續關鍵幀提供觀測。這不改變當幀的
  舊圖 PnP 姿態。僅保留的點與實際參與 PnP 的點分開記錄，避免虛增定位使用次數。
- 候選提交仍檢查 session、epoch、pose／landmark revision、正深度、舊圖支持與
  受影響的其他關鍵幀。幾何更新只有被後續 PnP 使用，才記錄 `ba_geometry_used`。
- 工作圖快照改為一次複製 NumPy 陣列；提交時成批比對各影格的舊點，避免每個
  地標反覆搜尋所有關鍵幀。對照組保留既有解算與保護規則。

三角化座標只作新變數初值，沒有再當成獨立的 3D 深度因子。已有的 session
影像檢索、共視候選與新舊圖 Sim(3) 重疊修正仍由原模組處理。

## 啟用與重現

在新的 SHA 綁定 profile 內設定：

```json
"online_workmap": {
  "enabled": true,
  "optimizer": "isam2",
  "retain_valid_tracks": true
}
```

新 release 的建置器支援 `--online-workmap --workmap-optimizer isam2
--workmap-retain-tracks`；不改動既有 release，也不授予定位品質或飛行核准。
GTSAM 必須先依 [`選配 lock`](../requirements/optional-workmap-graph-lock.txt)
安裝。缺少套件會在初始化時明確失敗，不會假裝已使用增量圖。

本工作樹沒有自己的 `.venv`，實測使用 `/home/allen/localization/.venv/bin/python`，
並以 `PYTHONPATH` 載入本次隔離的 GTSAM 相依。runtime、測試、模型與場域
資產都沒有因安裝這個選配層而被覆寫。

所有影片皆由 `定位演算法/validation/online_workmap_eval.py` 的 `_run_one` 經正式
worker 完整解碼回放，未連接飛機。這次是開發比較，每組完整影片一次；不是該驗收器
要求的全模式三次重複矩陣。重跑時應使用新的 `run_id` 與 `output_root`，避免覆寫
既有紀錄。session JSON／JSONL 在量測與關閉排空完成後才壓縮，
`session_archive.json` 保存解壓內容的 SHA-256；原始影片維持唯讀。

## 評估範圍

素材為 `/home/allen/下載/P1980198.MP4`、`P1990199.MP4`、`P2000200.MP4`。
操作員確認均在河濱拍攝，可能包含新區，但尚無獨立真值或已確認的新區／重訪區段。
query camera 沿用既有標定並依解析度縮放，未重新校正這批影片。

P198 的初步回放沒有建立新地標或觸發 BA，不能用它證明兩種優化器的差異。
主要比較 P199、P200 的原 local BA、保留新點的 local BA，以及保留新點的 iSAM2。
FPS 包含解碼、worker、consumer 與關閉排空，不含模型載入與完整資產雜湊；
端到端延遲使用 capture-to-consumer 時戳。判斷時另外查看新鮮視覺位姿、真實 PnP
使用事件、拒絕原因與地圖更新排隊。

詳細結果位於 `outputs/analysis/slam_mer_incremental_dev/`。v1／v3／v4 為開發過程；
v2 使用記憶體檔案系統，不能代表 SSD 寫入表現。因空間不足中斷的回放另標記
`incomplete.json`，不列入完成結果。最終 QR 比較使用 SSD 並記錄可用空間。

## 最終比較與決策

以下兩個 local BA 對照取自 v4，QR 增量圖取自 v6。最終保留的六個相關 runtime
檔案與 v6 實測版本逐檔 SHA-256 相同；沒有新增觀測時跳過求解的 v7 嘗試未改善
延遲，已撤回。測試設定在工作樹搬移後改成自帶合成 profile。

| 影片 | 方法 | 端到端 FPS | 延遲 p95 / p99 (ms) | 新鮮視覺位姿幀 | BA 幾何使用事件 |
|---|---|---:|---:|---:|---:|
| P199，4,422 幀 | 原 local BA | 29.30 | 22.61 / 33.14 | 3,561 | 0 |
| P199 | 保留新點＋local BA | 29.37 | 21.96 / 30.69 | 3,558 | 0 |
| P199 | 保留新點＋iSAM2 QR | 28.72 | 103.58 / 226.67 | 3,483 | 74 |
| P200，2,997 幀 | 原 local BA | 29.44 | 145.46 / 315.07 | 2,613 | 0 |
| P200 | 保留新點＋local BA | 29.45 | 17.87 / 24.61 | 2,612 | 0 |
| P200 | 保留新點＋iSAM2 QR | 29.31 | 121.17 / 161.70 | 2,565 | 26 |

QR 組的 P199／P200 分別建立 1,228／726 個新地標，其中 930／452 個具有至少
三個關鍵幀觀測。回放期間接受 84／26 次 BA，並有 74／26 次後續 PnP 使用更新
幾何的事件。事件次數不是精度量測；少量點被多次使用也可能產生多次事件。

保留新點的 local BA 雖然能累積觀測，P199 有 368 次候選因窗外關鍵幀重投影變差
而被拒絕，最終沒有提交 BA。增量圖能把這些跨視角資料納入求解，這是此次實作
驗證到的能力差異。

**目前不建議將 iSAM2 設為預設替代方案。** 它達到 15 FPS 吞吐門檻，卻未通過
capture-to-consumer p95 ≤ 100 ms、p99 ≤ 200 ms 的共同延遲門檻，這兩片的新鮮
位姿數也沒有增加。原 local BA 在 P200 的一次回放同樣超過延遲門檻。這是單次
開發比較，尚不能推論統計上的優劣；正式全模式三次驗收與真正新區重訪仍未完成。

目前先保留為選配原型，維持既有預設。MASt3R／3D cell 查詢尚未整合或進行本機
推論效能測試；後續應先處理地圖發布與索引對即時路徑的成本，再評估增加深度模型。

原始數據與逐檔核對結果：

- [`final_comparison.json`](../outputs/analysis/slam_mer_incremental_dev/final_comparison.json)
- [`final_source_verification.json`](../outputs/analysis/slam_mer_incremental_dev/final_source_verification.json)
- [`cleanup.json`](../outputs/analysis/slam_mer_incremental_dev/cleanup.json)
- 新地標點雲：[P199](../outputs/analysis/slam_mer_incremental_dev/p199_qr_new_points.ply)、
  [P200](../outputs/analysis/slam_mer_incremental_dev/p200_qr_new_points.ply)。座標為原地圖單位，
  供檢視與分析，不是可直接選入 mission 的完整 release。

最終定位部署測試、release 建置與遙測測試以 `-m 'not hardware'` 執行，共
**310 passed**；10 個警告來自既有 pycolmap `Camera.create()` 棄用提示。

沒有獨立真值，不能以點數、位姿覆蓋率或 BA 次數宣稱公尺精度提升。
目前成果供離線評估，未切換現有 mission 或飛行核准。

## 來源

- [SLAM-MER 論文與補充材料](https://www.merl.com/publications/docs/TR2026-056.pdf)
- [作者架構文件](https://slam-mer.github.io/slam-mer/code_structure.html)
- [作者目前的深度模組整合範圍](https://slam-mer.github.io/slam-mer/depth_inference.html)

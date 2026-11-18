# 場域地圖定位系統
```text
<workspace-root>/                   ← workspace root，同時是 git repo
├── 定位演算法/       ★唯一一份定位演算法（EDM runtime、飛控、安全邏輯、驗證與建圖工具）
├── 控制介面程式/     site profile、操作 UI、兩個串流接口、任務工具
├── 地圖檔/場域/      ★每個場域一包（maps / bundles / routes / reports），不納入 Git
├── 模擬器/           測試影片、Sphinx 實驗
├── 執行環境/         runtime artifacts、wheelhouse 與本機 cache
├── requirements/     runtime、測試與品質工具的 hash locks
├── outputs/          flight_logs、benchmark 產物、實驗決策鏈
├── tools/            驗證實作、測試與唯讀工作區稽核
└── .venv/            Python 3.10 執行環境
```
```bash
python tools/workspace_audit.py --strict-output-names

```

## 烏來 all9 地圖

已匯入 `地圖檔/場域/wulai_all9_direct_20261019/`。開啟操作介面後，在
「地圖與 AUTO 航線 → ① 地圖」選擇「烏來 all9 DIRECT」；既有河濱與福和橋地圖保留，
不變更上次選擇或預設啟動場域。

- 定位模型：2,043 張關鍵影格、4,270,546 個 EDM 重三角化點。
- 依現有正式 BoQ 模型重建 2,019 × 16,384 檢索庫，不混用來源的 MegaLoc 描述子。
- 顯示點雲為定位模型的 147,261 點抽樣；原始模型與相機標定未修改。
- 主地圖與航線編輯器顯示 `map/map_dense.ply`：MVSFormer++ 稠密化點雲（與定位模型同一 EDM
  座標系，恆等轉換；來源 1.048 億點）抽樣為 3,000,000 點，綁定本場域座標系與 `map.ply`
  SHA-256，僅供顯示與點選，定位不讀取。以 `tools/verify_dense_alignment.py` 用 60 張關鍵影格
  的觀測深度驗證（一致率 0.62；2026-10-27 的舊稠密檔只有 0.02，因不在定位座標系而移除）。
  來源雜湊與建置參數在 `provenance/receipts/dense_display_20261028.json`；
  原始壓縮包 `wulai_all9_mvsformerpp_20261028.tar.zst`（6.5 GB）未複製進工作區。
- 保留來源收據、原始雜湊清單與轉接資訊於場域的 `provenance/`。
- 地圖維持原生 map 單位，無公尺尺度、無航線、未獨立驗證，不授權 AUTO。

離線驗證使用建圖參考影像重播，確認介面顯示與正式定位 worker 可產生有效位姿，
不代表獨立影片精度或實飛驗收。報告與截圖位於
[`outputs/analysis/wulai_all9_import_20261026/`](outputs/analysis/wulai_all9_import_20261026/)。

## 檔案整理原則

- 場域資產集中於 `地圖檔/場域/<site_id>/`，驗證報告集中於 `outputs/analysis/`。
- `__pycache__/`、`.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/` 是可重建快取，
  不作為地圖、模型或驗證證據保存。
- 保留原始碼、`MANIFEST.tsv`／`SHA256SUMS`、虛擬環境、離線 wheelhouse、模型權重、
  地圖版本、航線、任務快照與 `outputs/flight_logs/`；不可只因體積大就刪除。
- 三類電力設備偵測模型位於
  `執行環境/models/power_equipment_sample50_3class_yolo26n_640_aug_best.pt`；
  UI、偵測 worker 與離線啟動器由工作區根目錄解析，不依賴下載資料夾。
- 福和橋與烏來 all9 原始壓縮包保存在各自場域的 `provenance/`；
  烏來 `.tar.zst` 與同名 `.sha256` 一起保存。這些原始包不是可重建快取，不可清除。

## 注意事項
起飛、降落與即時飛行控制只能由現場操作員在桌面 UI 執行。任何 agent 或自動化
都不得代為起飛。請先完成離線影片、地面、模擬與拆槳測試。

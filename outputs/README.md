# outputs 治理

`outputs/` 只保存本機產生的驗證、實驗與操作證據。Git 忽略其內容，唯一進版控的是本 README。

長期結論寫在 [`docs/verified_localization_optimization_ledger.md`](../docs/verified_localization_optimization_ledger.md)。舊的 replay JSON、operator log、P157/P167/P168 原始基準已刪；不要從垃圾桶外的副本重做那些已否決的實驗。

## 目前保留

- `analysis/`：飛行紀錄複查、離線反例與修正驗證，保留各次來源雜湊及限制。
- `sim_autoflight/`：路線控制模擬結果，不代表實飛或定位精度驗收。
- `validation_receipts/`
  - 全地面系統驗證（`tools/system_validation.py`）之 JSON receipt（`validation_<timestamp>.json`）與步驟記錄（`validation_<timestamp>_logs/`）。
  - 生命週期：支援 `--smoke` 與完整（full）兩層 tier 驗證；內建自動輪替政策，嚴格保留最新 10 份 receipt 與對應 log 目錄，超出者自動清理，防止無界增長。
- `flight_logs/`
  - 飛行與操作日誌目錄，新的操作 session 使用該目錄。

## 歷史註記（已清理 1.19 GB 十目錄）

下列十個實驗與驗證目錄（共約 1.19 GB）已自本機磁碟清理，其數值結論、反例與定讞機制已記載於 [`docs/verified_localization_optimization_ledger.md`](../docs/verified_localization_optimization_ledger.md) 與 [`docs/direct_backend_ledger.md`](../docs/direct_backend_ledger.md)，不再留存本機暫存檔案：

| 路徑 | 內容 |
|---|---|
| `river_gluemap_all8_direct_20260831_import/` | 現行河濱圖單幀 smoke（1/1 TRACK、120 inliers） |
| `river_map_localizability/` | FIF+LWL 弱點：原圖 70 球，修圖後 62 球 |
| `river_map_existing_data_repair/` | 既有影像修圖：793,643→428,028 點；p90 6.79→1.65 px |
| `security/` | SBOM 與 pip-audit |
| `gates_20260905/` | 2026-09-05 單段 gate（boot60、ANISO、async multi-seed） |
| `corpus_20260905/` | **七段 720p 語料庫**：`baseline`（async 舊預設）、`sync`（現行預設）、`async_bounded`、`sync_relaxed66`、`p116_matrix`。數值結論在總帳「2026-09-05 全影片 720p 語料庫」 |
| `augment_20260905/` | 失敗球 reference 補強實驗（**NOT GO**）：候選、單幀 smoke、七段語料庫對照。機制與數值見總帳「2026-09-05 失敗球 reference 補強」 |
| `baseline_recheck_20260905/` | 上述實驗的 P168 4K seq700 對照基準 |
| `klt_lost_20260905/` | KLT LOST 預測實驗（見 `docs/klt_lost_prediction_experiment.md`） |
| `veto_gates_20260906/` | 隨機化配對閘門證據（`定位演算法/validation/randomized_ab_gate.py`）。否決器：`gate_r3c_report.json`（R3 軌跡 21/21 PASS）、`gate_p3_report.json`（P3 重力 21/21 PASS）、`gate_both_report.json`（雙旗標＋另一組 seed）、`injection.json`（故障注入檢出率）、`gate_r3_firstversion_FAIL.json`（**保留反例**：距離語意版 P167 −35、河濱_P117 −30）。profile 旗標定讞：`gate_qgraph_report.json`（query CUDA graph）、`gate_tmf_report.json`（track_map_first）、`gate_asm_report.json`（acquire_stage_mode）、`gate_prb_report.json`（pnp_ranked_batches） |
## 已濃縮的實驗結論

- 保留：query feature reuse、GPU cache 192、full-token MegaLoc TensorRT、persistent feature bank、packed D2H、bundle mmap。
- 不預設：token-reduced MegaLoc、EDM 640 輸入、BoQ VPR、native EDM TensorRT、P168 PnP/batch 候選、EDM/PnP overlap。
- 現行地圖未獨立 holdout；resolver 維持 fail closed。

`tools/workspace_audit.py --strict-output-names` 仍適用。`flight_logs/` 已清空；新的操作 session 仍用該目錄。

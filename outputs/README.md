# outputs 治理

`outputs/` 只保存本機產生的驗證、實驗與操作證據。Git 忽略其內容，唯一進版控的是本 README。

長期結論寫在 [`docs/verified_localization_optimization_ledger.md`](../docs/verified_localization_optimization_ledger.md)。舊的 replay JSON、operator log、P157/P167/P168 原始基準已刪；不要從垃圾桶外的副本重做那些已否決的實驗。

## 目前保留

- `validation_receipts/`
  - 全地面系統驗證（`tools/system_validation.py`）之 JSON receipt（`validation_<timestamp>.json`）與步驟記錄（`validation_<timestamp>_logs/`）。
  - 生命週期：支援 `--smoke` 與完整（full）兩層 tier 驗證；每次只輪替指定 receipt 目錄直屬的 `validation_*.json` 與對應 log，保留最新 10 份，不遞迴清理歷史子目錄。
  - 保留 2026-10-02 的 release 與 development smoke 收據，以及 `offline_completion_20260915/` 的完成收據、`p0_20260915_final/` 的 release 拒絕證據；這些 smoke 不代表完整驗證或硬體許可。
  - 2026-10-03 清理 `p0_20260915_baseline/`（7 檔、222,086 bytes）：舊格式／型別／工作區診斷已由上述同一基底 commit 的完成收據取代，六項步驟均通過；保留後續完成與 release 來源紀錄。
- `route_workflow_audit_20260917/`：21/21 路線離線稽核結果與來源 SHA manifest。這是模擬 plant 的歷史證據，物理校準未驗證，不代表目前程式或實飛通過。
- `river_gluemap_all8_direct_20260831_import/`：保留總帳引用的原始地圖匯入收據與 persistent store smoke；不代表獨立 holdout 或目前地圖版本驗收。
- `security/`：依賴安全 gate 使用的 pip-audit 與 CycloneDX SBOM，保留最近產物。
- `deps/`：選配相依（如 `deps/workmap_graph` 的 gtsam）以 `PYTHONPATH` 載入的本機安裝目標；不是證據，也不納入 manifest，可由對應鎖檔重建。
- `flight_logs/`
  - 飛行與操作日誌目錄，新的操作 session 使用該目錄。
  - `session_*/trajectory.jsonl` 保留規劃路線快照、AUTO 控制軌跡、定位來源與
    AUTO 起訖事件，不隨診斷日誌輪替清除。降落後使用
    [`tools/flight_trajectory_review.py`](../tools/flight_trajectory_review.py)
    匯出離線回放頁、CSV 與 JSON；操作方式見 [`tools/README.md`](../tools/README.md)。
  - 診斷日誌依 30 天／20 GiB 政策輪替，略過目前 session；不刪 `trajectory.jsonl` 或整個 session 目錄。真實飛行資料不得僅因沒有程式引用就刪除。

## 歷史已清理實驗

下列實驗目錄先前已自本機磁碟清理，其數值結論、反例與定讞機制已記載於 [`docs/verified_localization_optimization_ledger.md`](../docs/verified_localization_optimization_ledger.md) 與 [`docs/direct_backend_ledger.md`](../docs/direct_backend_ledger.md)，不再留存本機暫存檔案。地圖匯入與安全 gate 收據仍保留，見上列清單。

| 路徑 | 內容 |
|---|---|
| `river_map_localizability/` | FIF+LWL 弱點：原圖 70 球，修圖後 62 球 |
| `river_map_existing_data_repair/` | 既有影像修圖：793,643→428,028 點；p90 6.79→1.65 px |
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

`tools/workspace_audit.py --strict-output-names` 仍適用。新的操作 session 仍使用 `flight_logs/`；清理前須重新確認 session 已關閉、資料用途與現存內容，不得整目錄清空。

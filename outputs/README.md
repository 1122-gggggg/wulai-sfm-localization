# 測試輸出索引

最後整理：2026-08-03。輸出不搬動既有路徑，避免舊報告內的引用失效。

所有成功、失敗、淘汰與中斷實驗都列在 `EXPERIMENT_LEDGER.md`；新實驗使用
`EXPERIMENT_RECORD_TEMPLATE.json`；尚未執行項目列在 `EXPERIMENT_BACKLOG.md`。
結果證據不因未採用而刪除。

本輪 production 結論與測試數字見 `PROJECT_REVIEW_20260719.md`。

| 路徑 | 用途 | 保留策略 |
|---|---|---|
| `flight_logs/` | 每次 UI session、命令、incident、telemetry、定位與延遲 | 安全／命令紀錄永久；可再生效能紀錄依 30 天或 20 GB retention |
| `validation_receipts/` | 單一驗證入口的 JSON receipt 與逐步 log | 保留，正式驗收證據，不由自動 retention 刪除 |
| `validation/` | P119 integrity、品質 replay、PnP A/B 與稽核產物 | 保留摘要與基準；大型可再生中間檔另行人工封存 |
| `production_stream_bench/` | production 串流 benchmark | 保留與 release／設定變更直接相關的結果 |
| `edm_target_site_p024_bench/` | P024 全片、LOO 精度代理與 top-k A/B | 保留，現行參數依據 |
| `edm_random_corpus_20260719/` | 固定種子 20260719 隨機三片、1,800 幀 RTX 5060 驗證 | 保留，現行 Target EDM 證據 |
| `edm_coarse_topk_ab_20260719/` | coarse top-k 2304/3225 短 corpus + LOO A/B | 保留，3225 retain 依據 |
| `edm_weak_topk_ab_20260719/` | WEAK top-k 2/3 完整 reverse A/B | 保留，no-retain 依據 |
| `edm_lost_sweep_20260719/` | LOST local 3/5、scan 1/2/3、grace 6/12 完整矩陣 | 保留，no-retain 依據 |
| `edm_full_video_corpus_20260719/` | 六片本機影片全部 39,020 幀與逐片結果 | 保留，場域混合與 policy 證據 |
| `edm_target_site_stress_20260719/` | 反向跳變影片 20 分鐘 UI/硬體壓測 | 保留，熱與尾延遲證據 |
| `edm_onnx_bench/`、`edm_trt_ab/` | TensorRT/ONNX 淘汰依據 | 保留小型 JSON/Markdown/log；engine cache 已清理 |
| `reverse_topk_ab_20260715/` | TRACK top-k=1/2 延遲與內點 A/B | 保留，參數決策依據 |
| `regression_20260710/baseline/`、`after/` | 舊回歸摘要 | 保留 |
| `regression_20260710/frames/` | stride JPEG 中間幀 | P024/P121 原片存在，已清理；P123/P124 本機無原片，暫留 |
| `exact_latency_20260713/`、`optimization_trials_20260713/` | 歷史最佳參數與延遲拆解 | 保留 |
| `onnx_flow_sweep_20260713/`、`video720_optim_20260713/` | 歷史淘汰方案證據 | 保留摘要，後續可另行封存 |

## 新輸出命名

- 正式驗證：`validation/` 或 `validation_receipts/`。
- 操作 session：只能由 UI 建立在 `flight_logs/session_<UTC>_<mode>_<id>/`。
- 新實驗：`<backend>_<目的>_<YYYYMMDD>/`，並同步寫入 `EXPERIMENT_LEDGER.md`。
- 臨時中間檔不得直接散落在 workspace root；放在對應實驗目錄內。
- 無法歸入上述分類的新頂層項目會被 `tools/workspace_audit.py --strict-output-names`
  標記，先補用途再保留。

`flight_logs` 的自動 retention 只處理 localization／performance／video metrics 與
舊 `loc_metrics_*`；commands、incidents、session manifest／summary、硬體與影片清單
不會自動刪除。其他實驗證據與模擬器產物也不會由 UI 自動清理。

現行 production 是 PyTorch CUDA FP16 + channels-last；TensorRT FP16 失準，
TensorRT FP32 比 PyTorch FP16 慢，因此操作介面不提供 ONNX/TensorRT 選項。

人工清理先送桌面資源回收筒。UI retention 只會直接刪除上面列出的可再生效能 log，
並寫入 `retention_audit.jsonl`；原始影片、RGB PLY、定位 bundle、建圖資料、flight
safety／command／incident logs、失敗／中斷實驗與仍無原片可重建的中間幀不列入
自動清理。

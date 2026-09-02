# outputs 治理

`outputs/` 只保存本機產生的驗證、實驗與操作證據。Git 忽略其內容，唯一進版控的是本 README。

長期結論寫在 [`docs/verified_localization_optimization_ledger.md`](../docs/verified_localization_optimization_ledger.md)。舊的 replay JSON、operator log、P157/P167/P168 原始基準已刪；不要從垃圾桶外的副本重做那些已否決的實驗。

## 目前保留

| 路徑 | 內容 |
|---|---|
| `river_gluemap_all8_direct_20260831_import/` | 現行河濱圖單幀 smoke（1/1 TRACK、120 inliers） |
| `river_map_localizability/` | FIF+LWL 弱點：原圖 70 球，修圖後 62 球 |
| `river_map_existing_data_repair/` | 既有影像修圖：793,643→428,028 點；p90 6.79→1.65 px |
| `security/` | SBOM 與 pip-audit |

## 已濃縮的實驗結論

- 保留：query feature reuse、GPU cache 192、full-token MegaLoc TensorRT、persistent feature bank、packed D2H、bundle mmap。
- 不預設：token-reduced MegaLoc、EDM 640 輸入、BoQ VPR、native EDM TensorRT、P168 PnP/batch 候選、EDM/PnP overlap。
- 現行地圖未獨立 holdout；resolver 維持 fail closed。

`tools/workspace_audit.py --strict-output-names` 仍適用。`flight_logs/` 已清空；新的操作 session 仍用該目錄。

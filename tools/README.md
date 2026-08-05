# 工作區工具

| 檔案 | 用途 |
|---|---|
| `system_validation.py` | 編排主系統與 `parrot_stimulate` 的完整純地面驗證 |
| `workspace_audit.py` | 唯讀檢查目錄、必要檔案、symlink、output 命名與容量 |
| `simulator_preflight.py` | 模擬介面啟動前檢查 Python、CUDA、模型、場域資產與影片 |
| `install_runtime.sh` | 用 `requirements-lock.txt` 的 transitive pins/hashes 建立乾淨 CPython 3.10 venv |
| `export_simulator_package.py` | 匯出固定程式/runtime，排除 venv、outputs、地圖與影片並產生 manifest |
| `package_manifest.py` | 驗證目前可攜式發布包的 MANIFEST.tsv / SHA256SUMS |
| `test_clean_install.sh` | 在暫存目錄重建 CPython 3.10 venv 並執行 runtime preflight |
| `simulated_ui_smoke.sh` | 實際啟動地圖/影片選擇路徑，等 GUI 產生有效 pose 後安全終止 |
| `test_portable_runtime.sh` | 在 actual portable 暫時匯入固定場域，乾淨安裝並要求 UI 產生有效 pose，後恢復發布邊界 |
| `test_system_validation.py` | 驗證編排步驟和 Python 環境隔離 |
| `test_workspace_audit.py` | 驗證工作區結構契約 |

正常使用仍從根目錄執行 `./驗證系統.sh`；不需要直接呼叫
`system_validation.py`。正式轉移時以
`./驗證系統.sh --portable-package /path/to/portable_localization` 讓收據同時綁定
實際輸出包的 `PORTABLE_PACKAGE.json`、`MANIFEST.tsv` 與 `SHA256SUMS`。
加上 `--clean-install --ui-smoke` 時，還會在 actual portable 中暫時匯入固定場域，
以其 lock 檔建立乾淨環境，並經 selector/UI 產生有效 pose 後自動清理。只要重查
目錄時可執行：

```bash
python tools/workspace_audit.py --strict-output-names
```

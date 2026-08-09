# 工作區工具

| 檔案 | 用途 |
|---|---|
| `system_validation.py` | 編排主系統與 `parrot_stimulate` 的完整純地面驗證 |
| `workspace_audit.py` | 唯讀檢查目錄、必要檔案、symlink、output 命名與容量 |
| `check_maintainability.py` | 以分區 C901 預算阻止複雜度熱點數量或最壞值回升 |
| `simulator_preflight.py` | 模擬介面啟動前檢查 Python、CUDA、模型、場域資產與影片 |
| `install_runtime.sh` | 用 `requirements-lock.txt` 的 transitive pins/hashes 建立乾淨 CPython 3.10 venv |
| `export_simulator_package.py` | 匯出固定程式/runtime，排除 venv、執行輸出、地圖與影片，僅保留 outputs 治理 README，並產生 manifest |
| `package_manifest.py` | 驗證目前可攜式發布包的 MANIFEST.tsv / SHA256SUMS |
| `release_activation.py` | 驗證 commit/version-bound package，原子 stage/activate/rollback |
| `test_clean_install.sh` | 在暫存目錄重建 CPython 3.10 venv 並執行 runtime preflight |
| `simulated_ui_smoke.sh` | 實際啟動地圖/影片選擇路徑，等 GUI 產生有效 pose 後安全終止 |
| `test_portable_runtime.sh` | 在 actual portable 暫時匯入固定場域，乾淨安裝並要求 UI 產生有效 pose，後恢復發布邊界 |
| `test_system_validation.py` | 驗證編排步驟和 Python 環境隔離 |
| `test_workspace_audit.py` | 驗證工作區結構契約 |

`requirements-lock.txt` 是 runtime 的唯一 hash lock；`requirements-test-lock.txt`
另外固定 pytest、ruff、coverage 與 timeout plugin。乾淨驗證可用
`bash tools/install_runtime.sh --test-deps`，安裝器會拒絕
`include-system-site-packages=true` 的既有 venv。`simulator_preflight.py --json` 的
receipt 會記錄稀疏點雲 collision monitor 是否能在 clean lock-only 環境取得；
scipy 已固定於 runtime lock，正常狀態為 `available_non_production`。它不是
production safety，`collision_protection_claim` 永遠為 false。若未來將其納入
production，先完成 safety wiring 審查，再使用 `--require-collision-monitor`，
否則 preflight fail closed。

`system_validation.py` 的 receipt 也會在 `pytest.steps` 彙整每個 pytest step 的
passed/failed/skipped/xfailed/xpassed/errors/warnings，並在 `pytest.skipped` 列出
所有條件式 skip；因此「未執行」不會被誤報成通過。

同一個 validation entry 會以 `monitor_hardware.py --output - --samples 1` 讀取硬體
狀態；JSONL 只進該 step 的 receipt log，不建立額外的永久硬體輸出檔。

Portable manifest 的 scope 由 `tools/package_manifest.py` 唯一決定；editor files、
cache、workspace 執行輸出、audit review artifacts 與其他 `.gitignore` 對應的非發布
資料不會因為偶然存在於 workspace 就進入 MANIFEST.tsv。唯一例外是受版控的
`outputs/README.md`，用來把輸出分類與保留規則帶到新電腦。

發布收據與 `PORTABLE_PACKAGE.json` 必須同時帶 manifest/SHA256SUMS、commit、version
與 dirty 狀態。`release_activation.py` 先驗證完整 package，再以 temporary directory
與 atomic symlink replacement 建立 `current`／`previous`；dirty source 預設拒絕，只有
明確傳入 `--allow-dirty`（development）才可 stage 或 activate，rollback 也會重新驗證
目標 package。

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

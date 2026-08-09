# 工作區工具

| 檔案 | 用途 |
|---|---|
| `system_validation.py` | 編排主系統與 `parrot_stimulate` 的完整純地面驗證 |
| `workspace_audit.py` | 唯讀檢查目錄、必要檔案、symlink、output 命名與容量 |
| `check_maintainability.py` | 以分區 C901 預算阻止複雜度熱點數量或最壞值回升 |
| `simulator_preflight.py` | 模擬介面啟動前檢查 Python、CUDA、模型、場域資產與影片 |
| `offline_wheelhouse.py` | 建立／驗證與三份 lockfile 及每個 wheel SHA-256 綁定的離線安裝庫 |
| `install_runtime.sh` | 用 hash lock 建立乾淨 CPython 3.10 venv；`--offline` 強制 `--no-index` |
| `export_simulator_package.py` | 依 runtime allowlist 匯出固定程式；可用 `--site-profile` 收錄完整 hash-bound 場域資產，排除 venv、執行輸出與影片 |
| `package_manifest.py` | 驗證目前可攜式發布包的 MANIFEST.tsv / SHA256SUMS |
| `release_activation.py` | 驗證 commit/version-bound package，原子 stage/activate/rollback |
| `test_clean_install.sh` | 在暫存目錄重建 CPython 3.10 venv 並執行 runtime preflight |
| `simulated_ui_smoke.sh` | 實際啟動地圖/影片選擇路徑，等 GUI 產生有效 pose 後安全終止 |
| `test_portable_runtime.sh` | 在 actual portable 暫時匯入固定場域，乾淨安裝並要求 UI 產生有效 pose，後恢復發布邊界 |
| `test_system_validation.py` | 驗證編排步驟和 Python 環境隔離 |
| `test_workspace_audit.py` | 驗證工作區結構契約 |
| `security_dependency_gate.py` | 執行 pip-audit、驗證到期中的安全例外，並輸出 CycloneDX SBOM |

`requirements-lock.txt` 是 runtime 的唯一 hash lock；`requirements-test-lock.txt`
另外固定 pytest、ruff、coverage 與 timeout plugin。正式 portable 先在來源機建立
wheelhouse：

```bash
.venv/bin/python tools/offline_wheelhouse.py build \
  --output /path/to/approved-wheelhouse \
  --requirements requirements-lock.txt \
  --requirements requirements-test-lock.txt \
  --requirements requirements-quality-lock.txt
```

乾淨測試可用 `bash tools/install_runtime.sh --offline --test-deps`；完整 system
validation 使用 `bash tools/install_runtime.sh --offline --test-deps --quality-deps`。
offline 模式會先核對 `WHEELHOUSE.json`、三份 lockfile 與所有 wheel，再以
`--no-index --find-links` 安裝。安裝器會拒絕
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
目標 package。即使 manifest 正確，`offline_install.complete` 不是字面 `true` 或
wheelhouse 與 lock/digest 不符也不得 activation。

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

## 外部 runtime artifacts

`RUNTIME_ARTIFACTS.json` 是 Git source 內的 runtime artifact registry，不是下載器。
它只列出 portable runtime 真正需要的相對路徑、大小與 SHA-256；模型 cache 的其他
repo、checkpoint 與暫存檔不會被 exporter 掃描或複製。來源工作區的
`MANIFEST.tsv` / `SHA256SUMS` 是 source-only manifest，外部 artifact 則由 registry
獨立綁定；portable 輸出包的 manifest 會同時包含已解析的 allowlist 檔案。
來源 checkout 可用 `python tools/package_manifest.py verify --source-only` 檢查；
portable package 則在其根目錄執行一般的 `verify`。

乾淨 checkout 沒有外部 artifact 時，exporter 會 fail closed，列出缺少的相對路徑與
digest，不會嘗試網路下載。請從受核准的離線 artifact bundle 建立與 registry 相同的
目錄樹，再指定 seed root：

```bash
python tools/export_simulator_package.py /path/to/portable_localization \
  --artifact-root /path/to/seed-root \
  --wheelhouse-root /path/to/approved-wheelhouse \
  --site-profile 控制介面程式/site_profiles/river_site_edm.json
# 或：SFM_RUNTIME_ARTIFACT_ROOT=/path/to/seed-root python tools/export_simulator_package.py ...
```

`--artifact-root` 下的檔案必須以 repository-relative path 放置；resolver 會先驗證
大小，再驗證 SHA-256。沒有 registry 或 digest 不符時不會產生部分可信的 runtime 包。
沒有 `--wheelhouse-root` 的一般匯出不會假裝成離線包；其
`PORTABLE_PACKAGE.json` 會明確標示 `offline_install.complete=false`，不得當作正式
activation 輸入。
有 `--site-profile` 時，`PORTABLE_SITE_ASSETS.json` 另外綁定 profile 與每個場域檔案；
reference index 不能只帶 manifest，signed hardware approval 也不能缺 receipt、
signature 或 trust store sidecar。

## 工程品質 gates

`requirements-quality.txt` / `requirements-quality-lock.txt` 是獨立的品質工具
清單，不會改寫 runtime 或 test lock。CPython 3.10 可由安裝器加入現有驗證環境：

```bash
bash tools/install_runtime.sh --quality-deps
python -m mypy
ruff check . --select S102,S105,S106,S107,S602,S604,S605,S608 \
  --exclude '**/test*.py' --exclude '**/tests/**' \
  --exclude '定位演算法/deploy_code/runtime/**'
python tools/security_dependency_gate.py
```

`mypy` 只對 `pyproject.toml` 的明確 typed boundary 清單設 fail gate，並不宣稱
整個 Tk/Olympe/Torch 應用程式已完成型別化。安全 gate 的 vulnerability exception 在
`pyproject.toml` 以 package、advisory ID、理由和 ISO 到期日保存；到期或未列出的
漏洞都會 fail。pip-audit 的 `skip_reason` 不能以一般 vulnerability exception
略過，必須在 `[[tool.security_dependency_gate.skip_exceptions]]` 以精確的 package、
version、`skip_reason`、理由和到期日列出。版本、原因改變或例外過期都會 fail，避免
把新的未稽核套件默認視為安全。`security_dependency_gate.py` 會在
`outputs/security/` 產生 `pip-audit.json` 和 `sbom.cyclonedx.json`；非 PyPI 的
CUDA wheel 即使被 pip-audit skip，也會以 `audited=false` 與原始 `skip_reason`
保留在 CycloneDX JSON SBOM 中。

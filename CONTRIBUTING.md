# 開發與驗證

主專案使用 CPython 3.10；`模擬器/parrot_stimulate` 使用獨立的 Python 3.11 環境。
相依來源、固定版本與離線安裝契約見 [`requirements/README.md`](requirements/README.md)。
新環境由 `bash tools/install_runtime.sh --test-deps --quality-deps` 建立；已有核准的
離線 wheelhouse 時加上 `--offline`。不要把個人環境或模型權重提交到 Git。

## 日常修改

1. 先讀該模組、相關測試及最近的安全規範。目錄責任見
   [`文件/ARCHITECTURE.md`](文件/ARCHITECTURE.md)。
2. 行為修正先保留可重現案例。控制器、定位協定、worker 關閉與發布完整性優先使用
   回歸測試；文件修改檢查連結與差異即可。
3. 使用下面的快速檢查，再執行受影響測試。較大的修改執行完整離線測試與 coverage。
4. 完成後更新 source manifest，檢查 diff，再提交原始碼、測試與文件。

```bash
# 開發模式容許尚未提交的修改；產生的 receipt 不代表正式發布核准。
./驗證系統.sh --smoke --allow-dirty

# 所有預設收集的離線案例，包含 Sphinx contract tests；不連接飛機。
.venv/bin/python -m pytest -q --timeout=300 -m 'not hardware' \
  --cov --cov-config="$PWD/pyproject.toml" --cov-report=term

# 獨立 Python 3.11 子專案，在其目錄執行。
cd 模擬器/parrot_stimulate
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

## 品質門檻

| 檢查 | 契約 |
|---|---|
| Ruff `E9,F` | 第一方 Python 的語法、未定義名稱及未使用匯入 |
| Ruff format | `tools/system_validation.py:ROOT_FORMAT_SCOPE` 定義的共同範圍 |
| 複雜度 | `tools/check_maintainability.py` 的分區預算與大型模組行數上限 |
| mypy | `pyproject.toml` 列出的 typed boundaries，非全專案型別化聲明 |
| coverage | 第一方核心整體 coverage（含分支）的 50% 下限 |
| 安全與依賴 | 指定 Ruff security 規則、pip-audit、具期限的例外及 CycloneDX SBOM |

CI 與本機共用這些門檻。功能新增不應靠放寬複雜度或降低 coverage 門檻通過。
第三方 EDM 與 direct vendor 不做全面格式化；修改 vendor 必須更新來源與 patch 雜湊紀錄。

大型 Tk/Olympe 檔案未加入全域 formatter，避免不必要的全檔變動。新增邏輯優先放到
已有責任模組；抽取函式後要保留輸入驗證、呼叫順序、例外與空值行為。

## 測試與飛行邊界

新增測試位置依 [`tests/README.md`](tests/README.md)。用合成資料、fake backend 與
虛擬時鐘測試控制流程，讓時間戳、速度與位置共享同一時間尺度。場域資料依賴必須
明確標註，缺少資產的 skip 不可算成通過。私有航線數量增加時，測試的 wall-time
預算需反映案例數；飛行任務時間則沿用正式控制器的契約。

起飛、降落、關閉時降落、按住微移及 Esc 交接受
[`控制介面程式/SAFETY.md`](控制介面程式/SAFETY.md) 與
[`定位演算法/AGENTS.md`](定位演算法/AGENTS.md) 保護。任何自動化測試都不得啟動真機馬達。
離線模擬通過不構成真機、定位精度或物理速度保證。

## 提交與發布

回到 repository 根目錄，待程式、測試及文件定稿後執行：

```bash
.venv/bin/python tools/package_manifest.py generate --source-only
.venv/bin/python tools/package_manifest.py verify --source-only
git diff --check
git status --short
```

`MANIFEST.tsv` 與 `SHA256SUMS` 一起提交。Git 保存原始碼與可追溯設定；地圖、影片、
session、模型、wheelhouse 由各自的 artifact 契約提供。不要使用 `git add -f` 把本機
資料強制加入。正式 portable 安裝及 GPU/GUI 驗證依根目錄 README 執行，不能用
`--allow-dirty` 的開發收據取代正式交付驗證。

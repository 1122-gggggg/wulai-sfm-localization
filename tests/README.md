# 第一方測試

本目錄集中主系統原本與 production source 混放的 pytest，並鏡像來源責任：

```text
tests/
├── control_interface/       操作介面、場域與 mission pipeline
├── localization/            flight-control 與 deploy boundary
├── runtime_tools/           執行環境工具的 contract tests
└── tools/                   發布、安裝、manifest 與 validation 工具
```

以下子專案有獨立測試邊界：

- `模擬器/parrot_stimulate/tests/` 使用獨立的 Python 3.11 環境。
- `模擬器/sphinx_anafi_path_convergence/tests/` 由根目錄 pytest 收集。

根目錄執行 `.venv/bin/python -m pytest` 會依 `pytest.ini` 收集 `tests/` 與 Sphinx 測試。
CI 的根目錄 pytest 亦採同一收集邊界，Sphinx 不另跑一次；
這是收集設定，不是本輪完整測試已通過的聲明。
標記為 `slow` 的案例預設不執行（`pytest.ini` 的 `-m "not slow"`），需要時以 `-m slow`
手動執行；手動指定 `-m` 排除硬體與慢測時須寫成 `-m 'not hardware and not slow'`，
因為 pytest 只採用最後一個 `-m`。
Parrot simulator 固定由自己的 Python 3.11 環境執行（CI 的 `parrot-simulator` job）。
定位演算法的第一方測試已集中在 `tests/localization/`；測試不得再放回 production
source 目錄。

擴圖版本（`定位演算法/deploy_code_expand/`）與原始版本有同名模組，不能在同一個 pytest
程序共存，所以另跑（預設 `pytest` 不收集 `tests/localization/deploy_expand/`）。
`pytest-expand.ini` 收集兩類測試：

- `tests/localization/deploy/` 中與版本無關的共用測試（於 ini 的 `testpaths` 逐檔列出），
  由 `pythonpath` 讓擴圖版模組遮蔽原版，同一份測試檔直接跑在擴圖版模組上，不再複製；
- `tests/localization/deploy_expand/` 中擴圖版專屬或斷言與原版不同的測試。

`test_direct_motion_timing.py` 與 `test_direct_retrieval_policy.py` 也已收斂為
`tests/localization/deploy/` 的共用單份檔案，於 `pytest-expand.ini` 明列收集；
不再另留擴圖版同名副本。清單是逐檔指定，不是把整個 `deploy/` 加入擴圖版程序。
`testpaths` 必須維持 `deploy_expand/` 在共用檔案之前：共用 pose 實驗測試
收集時會把 `定位演算法/validation` 插入 `sys.path`，影響後續匯入。
擴圖版專屬測試先收集，加上 ini 的 expand-first `pythonpath`，才能維持
同名模組的擴圖版優先順序。

新增與版本無關的共用測試時，放在 `deploy/` 並加入 `pytest-expand.ini` 的清單：

```bash
.venv/bin/python -m pytest -c pytest-expand.ini -q --timeout=300 -m 'not hardware'
```

常用離線指令：

```bash
.venv/bin/python -m pytest -q --timeout=300 -m 'not hardware and not slow' \
  --cov --cov-config="$PWD/pyproject.toml" --cov-report=term
```

完整航線測試使用正式控制器和虛擬時鐘；測試不可透過縮短安全確認視窗來加速。
需要私有場域資產的案例會在資產缺失時略過，略過數必須與通過數分開報告。

`定位演算法/EDM工具包/` 目前沒有部署程式或獨立測試。部署測試只收集
`tests/localization/deploy/`，避免同一批測試被重複執行。

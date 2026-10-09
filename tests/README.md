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
Parrot simulator 固定由自己的 Python 3.11 環境及 system validation step 執行。
定位演算法的第一方測試已集中在 `tests/localization/`；測試不得再放回 production
source 目錄。

擴圖版本（`定位演算法/deploy_code_expand/`）與原始版本有同名模組，不能在同一個 pytest
程序共存，其測試在 `tests/localization/deploy_expand/`，預設 `pytest` 不收集，另跑：

```bash
.venv/bin/python -m pytest -c pytest-expand.ini -q --timeout=300 -m 'not hardware'
```

完整開發流程見 [`CONTRIBUTING.md`](../CONTRIBUTING.md)。常用離線指令：

```bash
./驗證系統.sh --smoke --allow-dirty
.venv/bin/python -m pytest -q --timeout=300 -m 'not hardware' \
  --cov --cov-config="$PWD/pyproject.toml" --cov-report=term
```

完整航線測試使用正式控制器和虛擬時鐘；測試不可透過縮短安全確認視窗來加速。
需要私有場域資產的案例會在資產缺失時略過，略過數必須與通過數分開報告。

`定位演算法/EDM工具包/` 目前沒有部署程式或獨立測試。部署測試只收集
`tests/localization/deploy/`，避免同一批測試被重複執行。

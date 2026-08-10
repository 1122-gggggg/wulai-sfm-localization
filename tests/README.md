# 第一方測試

本目錄集中主系統原本與 production source 混放的 pytest，並鏡像來源責任：

```text
tests/
├── control_interface/       操作介面、場域與 mission pipeline
├── localization/            flight-control 與 deploy boundary
├── runtime_tools/           執行環境工具的 contract tests
└── tools/                   發布、安裝、manifest 與 validation 工具
```

以下子專案已經有獨立測試邊界與執行環境，因此保留原位：

- `定位演算法/EDM工具包/tests/`
- `定位演算法/validation/tests/`
- `模擬器/parrot_stimulate/tests/`
- `模擬器/sphinx_anafi_path_convergence/tests/`

根目錄執行 `python -m pytest` 會依 `pytest.ini` 收集第一方、EDM 與 Sphinx 測試；
Parrot simulator 固定由自己的 Python 3.11 環境及 system validation step 執行。
測試不得再放回 production source 目錄。

`定位演算法/EDM工具包/deploy` 是指向正式 deploy code 的相容 symlink；pytest 只收集
`tests/localization/deploy/`，避免同一批 deploy tests 被重複執行。

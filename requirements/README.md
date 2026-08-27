# 主專案 Python 相依

本目錄是操作介面、定位 runtime、測試與品質工具的單一相依入口：

| 用途 | 直接相依 | 可重現安裝 lock |
|---|---|---|
| 正式 runtime | `runtime.txt` | `runtime-lock.txt` |
| 測試 | `test.txt` | `test-lock.txt` |
| 品質與安全檢查 | `quality.txt` | `quality-lock.txt` |
| 選配 RePoseD/MoGe | `optional-reposed.txt` | `optional-reposed-lock.txt` |

安裝正式環境請使用 `tools/install_runtime.sh`。lockfiles 固定 transitive
版本與 SHA-256；不要直接用未鎖定的 `.txt` 建立正式或 portable 環境。

以下相依屬於不同的執行環境，因此保留在原子專案：

- `定位演算法/EDM工具包/requirements.txt`
- `定位演算法/deploy_code/runtime/EDM/deploy/requirements_deploy.txt`
- `模擬器/parrot_stimulate/pyproject.toml`

更新 lockfile 時，從工作區根目錄依各 lockfile 首行記錄的 `uv pip compile`
命令重新產生，並執行完整 system validation。

`runtime-lock.txt` 是 B0 基礎環境；它不安裝 RePoseD/MoGe、PoseLib、utils3d
或 `huggingface-hub`，因此乾淨 base 不會因為非 PyPI source wheel 而失敗。
這些套件只存在於 `optional-reposed-lock.txt`，且不能以同名 PyPI 套件替代。

選配完整環境必須先取得核准的 optional wheelhouse，包含 lock 中四個精確 wheel
以及 MoGe wheel metadata 宣告的所有依賴，包括
`utils3d@3fab839f0be9931dac7c8488eb0e1600c236e183` 與
`pipeline@866f059d2a05cde05e4a52211ec5051fd5f276d6` 這兩個 source pin。目前
repository 的 `執行環境/wheels` 只有三個 source wheel，未構成可交付的完整 optional
wheelhouse；缺任何一項都必須停止，不得啟用 RePoseD：

```bash
python -m pip install --require-hashes --no-deps --no-index \
  --find-links /path/to/approved-optional-wheelhouse \
  -r requirements/optional-reposed-lock.txt
python -m pip check
python -c 'from moge.model.v2 import MoGeModel; import poselib, utils3d'
```

這個 optional lock 只鎖四個 direct source wheel，不宣稱已涵蓋 MoGe metadata
的 transitive/VCS 依賴；approved bundle 必須另外提供完整 hash lock 並先安裝那些
依賴。`--no-deps` 刻意避免 pip 自動解析未核准的 VCS/PyPI 版本，故 `pip check`
或 import smoke 失敗時必須 fail-closed。

三個命令全部成功後才可把 profile 的 RePoseD mode 設為 enabled；任何安裝、`pip
check` 或 import 失敗都代表 optional runtime 未驗證，B0 base 仍是唯一可用契約。

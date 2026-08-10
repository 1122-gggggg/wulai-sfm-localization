# 主專案 Python 相依

本目錄是操作介面、定位 runtime、測試與品質工具的單一相依入口：

| 用途 | 直接相依 | 可重現安裝 lock |
|---|---|---|
| 正式 runtime | `runtime.txt` | `runtime-lock.txt` |
| 測試 | `test.txt` | `test-lock.txt` |
| 品質與安全檢查 | `quality.txt` | `quality-lock.txt` |

安裝正式環境請使用 `tools/install_runtime.sh`。lockfiles 固定 transitive
版本與 SHA-256；不要直接用未鎖定的 `.txt` 建立正式或 portable 環境。

以下相依屬於不同的執行環境，因此保留在原子專案：

- `定位演算法/EDM工具包/requirements.txt`
- `定位演算法/deploy_code/runtime/EDM/deploy/requirements_deploy.txt`
- `模擬器/parrot_stimulate/pyproject.toml`

更新 lockfile 時，從工作區根目錄依各 lockfile 首行記錄的 `uv pip compile`
命令重新產生，並執行完整 system validation。

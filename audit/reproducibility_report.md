# audit/reproducibility_report.md — 相依套件與部署可重現性

稽核日期：2026-08-07
稽核範圍：`/home/allen/localization`
Git branch：`agent/localization-runtime-optimizations`
基準 commit：`d0b2250`（`Replace scrolling controls with responsive tabs`）
稽核時工作區狀態：**dirty** — 63 個變更檔案（52 files changed, 8155 insertions, 1229 deletions vs HEAD）

> 本報告只評估「能不能在另一台機器上重建並重跑」，不評估定位精度或建圖品質。
> 稽核期間未連線真機、未起飛、未送出任何飛行命令。

---

## 1. 支援平台

| 項目 | 要求 | 證據 |
|---|---|---|
| OS | Linux x86_64（manylinux_2_31） | `requirements-lock.txt:2` 的 uv compile 指令 |
| Python | **CPython 3.10.x**（實測 3.10.12） | `執行環境/requirements_runtime.txt:1`；`tools/install_runtime.sh:44-49` 硬性檢查 `sys.version_info[:2] != (3,10)` 就終止 |
| GPU | NVIDIA RTX 5060，CUDA 12.8 capable driver | `執行環境/requirements_runtime.txt:2-3` |
| 系統套件 | `ffmpeg`、`python3-tk`（tkinter） | `執行環境/requirements_runtime.txt:22`（註解形式，非自動安裝） |
| 模擬器子環境 | Python **3.11**（獨立 venv） | `模擬器/parrot_stimulate/.python-version`、`模擬器/parrot_stimulate/.venv/lib/python3.11/` |

實測本機：Python 3.10.12、torch 2.11.0+cu128、parrot-olympe 8.4.0、scipy 1.15.3。

---

## 2. 必要版本（實際 pin 狀況）

**結論：pinning 品質高於一般研究專案。**

- `requirements-lock.txt`（651 行）由 `uv pip compile --generate-hashes` 產生，**每個套件都有 SHA256 hash**，含 transitive 相依。這是可重現安裝的最強形式。
- `執行環境/requirements_runtime.txt` 為人類可讀的直接 pin，全部使用 `==`：

| 套件 | 版本 | 備註 |
|---|---|---|
| torch | 2.11.0+cu128 | 官方 PyTorch index，非 PyPI |
| torchvision | 0.26.0+cu128 | 同上 |
| numpy | 2.2.6 | |
| opencv-python | 4.13.0.92 | |
| pillow | 12.3.0 | |
| pycolmap | 4.0.4 | |
| **protobuf** | **3.19.4** | 檔案內註明「Olympe 的 wire schema 綁定此版本」 |
| parrot-olympe | 8.4.0 | |
| safetensors / kornia | 0.8.0 / 0.8.2 | XFeat / LighterGlue / MegaLoc stack |
| einops / joblib / loguru / yacs | 0.8.2 / 1.5.3 / 0.7.3 / 0.1.8 | EDM matcher 直接 import |

- 測試環境另 pin `pytest==9.1.1`、`ruff==0.16.1`（`requirements-test.txt`）。

### 2.1 已知安全漏洞（pip-audit，對 `.venv` 實跑）

```
protobuf   3.19.4  PYSEC-2026-899, PYSEC-2026-1806, PYSEC-2026-1805
setuptools 59.6.0  PYSEC-2022-43012, PYSEC-2025-49, PYSEC-2026-1918, PYSEC-2026-3447
```

無法稽核（不在 PyPI）：`torch 2.11.0+cu128`、`torchvision 0.26.0+cu128`、`lingbot-map 0.1.0`。

**判定：protobuf 不建議盲目升級。** 該版本是被 Olympe 8.4.0 的 wire schema 綁定的，且程式碼已明確註記此約束。升級 protobuf 等於改變與飛控的通訊層，風險遠高於該 CVE 在本系統（離線、區域網路、無外部輸入）的實際暴露面。**建議做法：維持 pin，於 `docs/` 記錄為「已知並接受的風險」，待 Olympe 升版時一併處理。**
`setuptools 59.6.0` 僅為 venv bootstrap 套件，非執行期相依，可安全升級，屬 P3。

### 2.2 未 pin 的相依 — **本報告最重要的可重現性缺陷**

`執行環境/requirements_runtime.txt:24-26`：

```
# Optional: scipy enables the sparse-cloud collision monitor in
# real_path_follow_controller (guarded import; absent = monitor unavailable)
# scipy==1.17.1
```

`scipy` 被**註解掉**，因此不在 lock 內。實際行為（`定位演算法/flight_control/real_path_follow_controller.py:59-62, 660, 676`）：

```python
try:
    from scipy.spatial import cKDTree
except Exception:  # optional; collision monitor becomes unavailable
    cKDTree = None
...
self.tree = cKDTree(self.xyz) if (cKDTree is not None and len(self.xyz)) else None

def update(self, pos):
    if self.tree is None:
        return {"status": "OFF", "distance": None, "point": None, "severity": 0.0}
```

**後果**：依照文件化流程（`tools/install_runtime.sh` + `requirements-lock.txt`）做乾淨安裝，`scipy` **不會被安裝**，碰撞監控會靜默變成 `status="OFF"`、`severity=0.0` —— 與「附近沒有障礙物」在資料上無法區分。本機目前有 scipy 1.15.3 是**歷史殘留**，而非安裝流程的結果；且與註解中的 1.17.1 不符。

這是 fail-open。詳見 `findings.md` F-07。緩解因素：該監控在程式碼自述中即為「operator warning layer」，且位於目前硬鎖的自主飛行路徑上。

---

## 3. 安裝指令

```bash
# 1. 建立 runtime（會強制 CPython 3.10、拒絕 include-system-site-packages）
bash tools/install_runtime.sh
#    可用環境變數：SFM_PYTHON=python3.10  SFM_VENV_DIR=/tmp/sfm-venv

# 2. 系統驗證（不連線真機）
./驗證系統.sh
```

`tools/install_runtime.sh` 的正面設計：
- `set -euo pipefail`
- 找不到 `python3.10` 直接失敗，並提示 Ubuntu 需 `python3.10-venv`
- **拒絕** `include-system-site-packages = true` 的 venv（`:35-39`）——避免系統套件污染
- 以 `requirements-lock.txt`（含 hash）安裝

---

## 4. 啟動指令

| 介面 | 指令 | 模式 pin |
|---|---|---|
| 影片模擬 | `./控制介面程式/影片模擬串流/啟動.sh` | 固定 `simulated-stream` |
| 影片模擬（選檔） | `./控制介面程式/影片模擬串流/選擇啟動.sh` | 同上 |
| **真機** | `./控制介面程式/真機串流/啟動.sh --site-profile <profile.json>` | 固定 `real-flight` |

專案根目錄的 `開啟介面的終端機代碼` 記錄了操作員實際使用的真機指令（含 `river_site_edm.json`）。

兩個介面互斥，且會拒絕跨介面參數；真機介面不會回退到影片檔（`控制介面程式/README.md:1-12`，由 `backend_contract.InterfaceMode` enum 強制）。

---

## 5. 測試指令

```bash
.venv/bin/python -m pytest -q          # 全套件
.venv/bin/python 定位演算法/validation/check_runtime_mirrors.py
```

**實測基準（稽核起點，乾淨執行）**：

```
1070 passed, 1 skipped, 17 warnings in 30.18s
```

`pytest.ini` 已正確設定 `testpaths` / `pythonpath` / `norecursedirs`（排除 `.venv`、`outputs`、`地圖檔`、`parrot_stimulate`）。

30 秒跑完 1070 個測試，代表**幾乎全部是 mock/unit 層級**；此觀察對測試可信度的意涵見 `test_matrix.md`。

---

## 6. 已知環境限制

1. **需要 X display**：`flight_operator_app.py` 是 Tk 應用；`控制介面程式/operator_interface/resolve_display.sh` 專門處理 display 解析。無頭環境無法啟動 UI（測試本身可無頭執行）。
2. **GPU 綁定**：production profile 以 RTX 5060 + CUDA 12.8 為契約。無 GPU 時定位 worker 行為未在本次稽核中驗證。
3. **模擬器需要獨立 Python 3.11 venv**，與主環境不同版本；`tools/simulator_preflight.py` 有對應檢查（`tools/test_system_validation.py::test_parrot_simulator_validation_stays_in_its_python311_environment`）。
4. **torch_hub_cache 為 package-local**：XFeat / LighterGlue / MegaLoc 以本地 torch.hub repo 提供（`執行環境/torch_hub_cache/`），不從網路抓取——這是刻意的離線設計，正面。
5. **離線強制**：`runtime_safety.configure_offline_environment()` 設定 `HF_HUB_OFFLINE=1` 等；`install_network_guard()` 以 monkey-patch `socket` 阻擋非白名單外連。副作用：任何期望對外連線的除錯工具在此程序內會失敗，屬預期行為。
6. **`os.execv` 重啟路徑**：切換 site profile 會 `os.closerange(3, 4096)` 後 `execv`（`flight_operator_app.py:7785-7796`）。註解說明這是為了規避 Olympe pomp loop 無法釋放 fd 的問題。此路徑在無真機時無法完整驗證。

---

## 7. 不可重現因素

| # | 因素 | 嚴重度 | 說明 |
|---|---|---|---|
| R1 | **`scipy` 未 pin** | **高** | 乾淨安裝會缺少碰撞監控且靜默降級。見 §2.2 / `findings.md` F-07 |
| R2 | 工作區 dirty | 高 | 63 個未提交變更、8155 行新增。目前 `MANIFEST.tsv`／`SHA256SUMS` 與工作區不一致，無法由 commit 唯一決定執行內容 |
| R3 | 系統套件非自動安裝 | 中 | `ffmpeg`、`python3-tk` 僅在註解中提及，安裝腳本不檢查、不安裝。缺少時失敗點在執行期而非安裝期 |
| R4 | `lingbot-map 0.1.0` | 中 | 不在 PyPI，來源與版本未記錄於 lock 的可取得位置，無法稽核亦難重建 |
| R5 | scipy 版本漂移 | 低 | 註解寫 1.17.1，實際安裝 1.15.3 |
| R6 | GPU/驅動版本未鎖 | 低 | 只記錄「CUDA 12.8 capable」，未記錄實測驅動版本號 |

**正面確認**：runtime 程式碼與腳本中**沒有**機器特定的絕對路徑。全庫僅一處出現 `/home/allen`，且是 `定位演算法/flight_control/path_follow_flight.py:89` 的一行註解，說明前次稽核已移除硬編碼路徑。這是明顯優於一般研究專案的表現。

環境變數方面，執行期共有約 50 個 `SFM_*` 變數。全部有預設值，**不存在「缺少某環境變數就無法啟動」的隱性相依**。但其中數個可放寬安全上限，屬設定管理問題而非可重現性問題（見 `findings.md` F-04）。

---

## 8. 建議改善方式

| 優先級 | 動作 | 驗收條件 |
|---|---|---|
| **飛行前** | 把 `scipy` 正式加入 `執行環境/requirements_runtime.txt` 並重新產生 lock；同時讓 `CollisionMonitor` 在 `cKDTree is None` 時回報 `status="UNAVAILABLE"`（而非 `"OFF"`），並在啟動 preflight 明確記錄一行 | 乾淨環境安裝後 `update()` 不再回 `OFF`；preflight log 出現 collision monitor 狀態 |
| **飛行前** | 提交或明確捨棄目前 63 個變更，重新產生 `MANIFEST.tsv` / `SHA256SUMS` | `git status --porcelain` 為空；manifest 驗證通過 |
| 近期 | `install_runtime.sh` 增加 `ffmpeg` / `tkinter` 存在性檢查，缺少時明確失敗 | 在無 ffmpeg 的容器中安裝會於安裝期失敗並給出可執行的修正指令 |
| 近期 | 記錄 `lingbot-map` 的來源（git URL + commit 或 wheel 位置） | 新機器可依文件取得同一版本 |
| 近期 | 於啟動日誌記錄 runtime identity（`collect_runtime_identity()` 已存在且會收集 python/platform/packages/cuda/driver）並寫入 session manifest | 每個 session 目錄的 `session_manifest.json` 可還原當時環境 |
| P3 | 升級 `setuptools`；`protobuf` 維持 pin 並文件化為已接受風險 | pip-audit 僅剩 protobuf 且有書面理由 |
| P3 | 補 scipy 版本註解與實際一致 | — |

---

## 9. 只能靜態驗證的項目（無真機）

以下在本次稽核中**未經真機驗證**，僅做程式碼與離線測試層級確認：

- `os.execv` + `closerange` 的 site profile 熱重啟是否真能釋放 SkyController socket
- Olympe 連線、firmware limit pin 與 readback 的實際行為
- PDRAW 影像串流中斷後的復原
- RTH / lost-link policy 在真實 GPS 不可靠場域的行為
- 碰撞監控在真實點雲上的判定（本次只驗證 scipy 缺席時的降級路徑）

正式真機測試前仍需完成的驗證，列於 `final_validation.md`。

# 場域地圖定位系統

一套可插拔定位演算法 + 一個操作介面。無人機、場域、地圖版本、定位器版本、
路徑與校正由獨立 manifest 組成；既有 site profile 保留為啟動相容層。
元件設計與更換流程見 [`文件/MISSION_COMPONENTS.md`](文件/MISSION_COMPONENTS.md)。

Git 不收錄實際場域點雲、localization bundle、影片與飛行紀錄；本機 workspace
在被忽略的目錄保存它們。固定模型由 `RUNTIME_ARTIFACTS.json` 綁定，既有選配
MoGe checkpoint 另有 Git LFS pointer；一般 checkout 不等於完整 runtime 資產包。系統架構與目錄所有權
規則見 [`文件/ARCHITECTURE.md`](文件/ARCHITECTURE.md)。

最後整理：2026-09-22。歷史設計決策與安全需求見 [`文件/SYSTEM_SPEC.md`](文件/SYSTEM_SPEC.md)；
目前可執行的場域與發布契約以本 README、site profile schema 與 preflight 為準。結構／容量
檢查請直接執行 `tools/workspace_audit.py`。

```text
<workspace-root>/                   ← workspace root，同時是 git repo
├── 定位演算法/       ★唯一一份定位演算法（EDM runtime、飛控、安全邏輯、驗證與建圖工具）
├── 控制介面程式/     site profile、操作 UI、兩個串流接口、任務工具
├── 地圖檔/場域/      ★每個場域一包（maps / bundles / routes / reports），不納入 Git
├── 模擬器/           測試影片、Sphinx 實驗
├── 執行環境/         runtime artifacts、wheelhouse 與本機 cache
├── requirements/     runtime、測試與品質工具的 hash locks
├── outputs/          flight_logs、benchmark 產物、實驗決策鏈
├── 文件/             系統規格、架構邊界、工作區稽核
├── tools/            驗證實作、測試與唯讀工作區稽核
├── 驗證系統.sh       唯一的純地面驗證入口
├── IMU飛行測試.sh    手動飛行 + 定位，錄 IMU／搖桿／畫面（見 docs/esekf_live_eval_runbook.md）
└── .venv/            Python 3.10 執行環境
```

目前唯一註冊的正式定位後端為 `direct`，由 CPU KLT/PnP 快迴路與 GPU MegaLoc/EDM
背景重定位組成。`sfm_glomap_deploy` 保留共用 contract 與 factory，不代表仍支援舊
EDM `.pt` 後端。歷史參數見 [`舊 EDM 後端紀錄`](docs/legacy_edm_backend.md)。

開發安裝、測試、品質門檻與提交流程見 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
本次檢查的範圍與量測見 [`專案品質檢查`](docs/project_quality_review_20260922.md)。

Git 只追蹤程式碼、設定與必要說明；場域資產、影片、權重與 outputs 不進版控。

工作區整理後可用同一個唯讀入口重查，不會刪除或搬動資料：

```bash
python tools/workspace_audit.py --strict-output-names
```

## 執行接口

| 接口 | 唯一入口 | 影像來源 | 飛控 backend |
|---|---|---|---|
| 模擬串流 | `控制介面程式/影片模擬串流/選擇啟動.sh` | 本機影片 / FFmpeg | 模擬，永不載入 Olympe |
| 實機操作（人工／preflight 後 AUTO） | `控制介面程式/真機串流/啟動.sh` | ANAFI PDRAW | Olympe，操作員 UI 控制；只接受 mission selection |
| IMU 飛行測試（手動飛行資料採集） | `IMU飛行測試.sh` | ANAFI PDRAW | 同上，只是額外錄 IMU／搖桿／定位當幀畫面 |
| 任務 authoring／檢查 | `控制介面程式/mission_pipeline.py` | 本機資產 | 不直接取代真機操作入口 |

兩個 runtime 入口互斥。模擬入口拒絕 `--live` / `real-flight`；實機入口拒絕
`--video`，且 PDRAW 不可用時不會拿錄影檔冒充實機畫面。真機入口只接受
`sfm-mission-selection/v1`；resolver 驗證所有 component SHA、定位品質、必要校正與
route 契約後產生唯讀 site-profile snapshot。AUTO 仍須由操作員在當次 UI 完成四步
preflight 並親自按下按鈕。

AUTO 巡航水平輸出受短 TTL 與 PCMD 百分比上限（`nudge_pct`／AUTO 自身 cap）
保護；fail-closed 地速閘門與轉向守門已拿掉，地速缺失、過期或達閾值不再送零。
這不是物理硬速度保證。路線完成降落仍要求新鮮地速不高於 0.10 m/s。AUTO
worker 若 2 秒沒有 heartbeat、發生例外或連續三次發送失敗，會進入
`AUTO_FAILED` 並保持零輸出，不會自動續行。

## 換地圖

新架構優先以 `sfm-mission-selection/v1` 選擇獨立元件，resolver 驗證完成後再產生
既有啟動器可讀的 site profile 快照。本次河濱 official69 map_v000 已使用此格式，
且已綁定同一重建座標系內唯一的新繪 route。

目前建圖端須輸出 `direct_bundle.json`、`direct_localizer_profile.json`、COLMAP model、
檢索 bank 與參考影像，以及同一座標系的顯示點雲與 site profile。交付清單與雜湊規格見
[`建圖端輸出規格`](控制介面程式/site_profiles/建圖端輸出規格.md)，後端分工見
[`direct README`](定位演算法/deploy_code/sfm_direct_deploy/README.md)。

換地圖等於換座標系，航線須重新建立並重新驗證。真機使用 mission selection
解析後的 snapshot，不能用 PLY 或任意 site profile 繞過 resolver。定位品質收據與
operator acceptance 的目前語意以 [`SAFETY.md`](控制介面程式/SAFETY.md) 為準。

```bash
# 影片目錄中只有一部影片時自動選用它（P119 會先驗證 SHA）
./控制介面程式/影片模擬串流/啟動.sh

# 多部影片或影片在外部路徑時明確指定
SFM_SITE_PROFILE=/absolute/path/to/site.json \
VIDEO=/absolute/path/to/video.mp4 \
  ./控制介面程式/影片模擬串流/啟動.sh
```

選擇器會要求操作員明確選擇場域 profile 和影片；直接呼叫上述 CLI
啟動器時，未指定 profile 會使用發布包內已驗證的河濱 profile。舊的
per-asset 參數只能在明確加上 `--allow-legacy-assets` 的遷移作業中使用。

## 轉到另一台電腦

先在來源電腦建立固定程式/runtime 發布包。它會包含 UI、EDM 程式、固定模型與
authoritative controller，排除 `.venv`、執行輸出與影片。指定 `--site-profile`
時，exporter 會另外收錄該 profile 的所有 digest-bound 地圖、route、bundle、
reference index 與核准 sidecar，並產生 `PORTABLE_SITE_ASSETS.json`：

```bash
# 來源電腦一次性建立與三份 hash lock 完全綁定的 Python wheelhouse
.venv/bin/python tools/offline_wheelhouse.py build \
  --output /path/to/approved-wheelhouse \
  --requirements requirements/runtime-lock.txt \
  --requirements requirements/test-lock.txt \
  --requirements requirements/quality-lock.txt

python tools/export_simulator_package.py /path/to/portable_localization \
  --artifact-root /path/to/approved-runtime-artifact-seed \
  --wheelhouse-root /path/to/approved-wheelhouse \
  --site-profile 地圖檔/場域/river_site/site_profile.json
cd /path/to/portable_localization
python tools/package_manifest.py verify

# 回到來源工作區，讓發布收據綁定實際輸出包的 manifest/hash
cd /path/to/source/localization
./驗證系統.sh --portable-package /path/to/portable_localization

# 正式交付再加入乾淨重建與實際 GUI smoke 證據
./驗證系統.sh --portable-package /path/to/portable_localization \
  --clean-install --ui-smoke

# 在新電腦建立乾淨 CPython 3.10 虛擬環境；強制不查詢任何 package index
bash tools/install_runtime.sh --offline

# 驗證／測試所需套件也從 hash lock 安裝（不使用 system site packages）
bash tools/install_runtime.sh --offline --test-deps

# 完整 system validation 另安裝 bounded mypy、dependency audit 與 SBOM 工具
bash tools/install_runtime.sh --offline --test-deps --quality-deps

# 匯入完整地圖場域包到地圖檔/場域/<site>/，再由選擇介面挑選地圖與影片
./控制介面程式/影片模擬串流/選擇啟動.sh
```

`RUNTIME_ARTIFACTS.json` 是 Git source 內的固定 runtime artifact allowlist；exporter
只會複製其中已驗證大小與 SHA-256 的檔案，不會無條件搬整個 model cache。clean
checkout 若沒有外部 runtime artifacts，請從受核准的離線 bundle 依相對路徑 seed
後再指定 `--artifact-root`（或 `SFM_RUNTIME_ARTIFACT_ROOT`）；exporter 只提供缺檔／
digest 與 seed 指引，不會假造下載 URL 或自行連網。

`WHEELHOUSE.json` 是 Python 套件的第二個 artifact registry：它同時綁定
runtime、test、quality 三份 lockfile 與每個 wheel 的大小／SHA-256。exporter 只在
`--wheelhouse-root` 驗證完整後才寫入 `offline_install.complete=true`；正式
clean-install 強制 `--no-index`，缺檔、多檔、symlink、lock 變動或 digest 不符都
fail closed。

正式驗證會將固定場域與短影片暫時匯入 actual portable，使用該包的
lock 檔建立乾淨環境，從其 selector/UI 產生有效 pose 後自動清理；
因此 receipt 同時證明雜湊及實際輸出包可執行。

模擬選擇介面保留直接選取現有有效 site profile 的相容入口，但 PLY 不能單獨定位；
真機入口不接受 profile 直接覆寫 mission resolver 的核准結果。
舊 EDM bundle 匯入接口要求 profile、PLY、EDM bundle、runtime profile 與
reference poses 五項；目前 direct release 應依建圖端輸出規格使用專屬契約。`query_camera` 與 `coordinate_frame` 必須寫在
profile，四個資產都必須有 SHA-256。`route_json` 與 `poles_json` 仍是獨立的
overlay／任務選配。完整格式見
[`控制介面程式/site_profiles/建圖端輸出規格.md`](控制介面程式/site_profiles/建圖端輸出規格.md)。
`requirements/` 是主專案的單一 Python 相依目錄；正式、測試與品質工具各有
人類可讀的 direct pins，以及帶 transitive pins 與 hashes 的對應 lockfile。
安裝器預設使用 `requirements/runtime-lock.txt`；
`tools/simulator_preflight.py --full-runtime` 會在 GUI 前實際載入
地圖 bundle、EDM CUDA model、BoQ VPR、GUI 與 worker。正式發布目前只支援已驗證的
NVIDIA RTX 5060 + CUDA 12.8；其他 GPU 必須重新做 CUDA、模型與效能驗證。系統層仍需
CPython 3.10 的 `venv/ensurepip`（Ubuntu/Debian 通常是 `python3.10-venv`）、
`ffmpeg`、`python3-tk`、X11/XWayland 與 NVIDIA CUDA driver。這些 OS/driver 先決條件不在
Python wheelhouse 內，必須由目標電腦的離線 OS 安裝媒體預先供應。正式 portable
內已收錄 PyTorch/CUDA 及所有固定 Python wheels；目標電腦的
`install_runtime.sh --offline` 不需要 Internet。

`scipy` 已固定在 runtime hash lock，供診斷與 deploy 程式使用，不是碰撞保護。
桌面操作介面不再有稀疏點雲近接懸停互鎖；路徑淨空由操作員目視與搖桿接管。
稀疏 SfM 點雲會漏掉動態、細小、無紋理及未建圖障礙物，不得宣稱具備
collision protection。

## 場域與航線

Git checkout 不帶私有場域資產。`控制介面程式/mission_selections/` 保存任務選擇，
實際地圖與航線放在 `地圖檔/場域/<site>/`。選定場域是否完整、定位品質狀態及
AUTO readiness 必須由當次 resolver/preflight 確認，不以文件中的靜態表格授權。

新增場域時：

1. 依建圖端輸出規格產生完整 direct release，驗證 profile、影像、內參及資產 SHA。
2. 航線與定位使用同一次重建的座標系。新的 route 與 selection 要重新綁定雜湊。
3. 先完成離線影片與模擬驗證，再由操作員依安全規範進行現場驗收。

### 一包一座標系

**同一個實體場地的不同次 SfM 重建，座標系不通用**（scale-free 也 gauge-free）。
本系統不推測跨重建 Sim3，換地圖時直接重畫 route。`map_ply`、`localization_bundle`、`map_reference_poses` 與
`route_json` 必須全部來自同一次重建。

混用的症狀很隱蔽：定位數值完全正常（inliers 95–108）但軌跡畫在錯誤的位置。
驗證法是比對 bundle 的 `ref_centers` 與該次重建 `final_model` 的相機中心，
逐張距離應在 1e-7 量級。

## 演算法只有一份

| 路徑 | 角色 |
|---|---|
| `定位演算法/deploy_code/sfm_direct_deploy/` | direct 地圖、tracker、provider 與 adapter |
| `定位演算法/deploy_code/sfm_glomap_deploy/` | 共用 pose/integrity、registry 與 factory |
| `定位演算法/flight_control/` | 控制器、Olympe frame source、安全監控與人工工具的 owner |
| `定位演算法/EDM工具包/deploy` `runtime` | symlink → `deploy_code/` |

模型權重在 `定位演算法/deploy_code/runtime/EDM/weights/`（不進版控）。
兩個 runtime 目錄沒有同名 Python 實作；需要共用的模組各自只有一個 owner。
`定位演算法/validation/check_runtime_mirrors.py` 保存權威所有權清單，並在 CI
拒絕缺少 owner、重新出現重複副本或未分類的同名 runtime 檔案。

## 驗證

```bash
./驗證系統.sh

# 選配：完整解碼 P119，並明確接受已核准的 2935/2934 已知例外
./驗證系統.sh --p119-integrity --accept-p119-known-incomplete

# 選配：將 receipt 與逐步 log 直接寫到已掛載的外接檔案系統
./驗證系統.sh --receipt-dir /media/VALIDATION_RECEIPTS
```

入口使用主 Python 3.10 與獨立 Python 3.11 `parrot_stimulate` 環境，執行
pytest、ruff、module ownership、flight selftest、dependency、profile/SHA、CUDA 與離線模型檢查，
並寫入 `outputs/validation_receipts/`。任一必要項失敗時整體 exit code 非零。
CI 與 `tools/test_clean_install.sh` 由 `requirements/test-lock.txt` 提供固定的
pytest-timeout、coverage 與 pytest-cov；測試使用每測試 300 秒上限並收集 coverage，
目前對第一方核心模組設 `50.00%` 最低門檻。validation receipt 會彙整 pytest
各 step 的 conditional skip，未執行的測試不會被當成通過。
`--receipt-dir` 的目標必須允許建立檔案與原子 rename；驗證器使用排他建立的隨機
暫存檔，不會跟隨可預測的 `.json.tmp` symlink。真正的 WORM 媒體應在 receipt 完成後
再封存，因為執行期間會持續更新 `running` 狀態與各 step 結果。

`parrot_stimulate` 明確要求 Python 3.11，因此使用自己的 `.venv` 驗證，
不由根目錄的 Python 3.10 pytest 跨版本收集。實際通過數以當次輸出為準。

## 注意事項

「地圖」不只是 PLY 點雲。定位還需要同場域建置的 localization bundle，以及和
影片／相機一致的內參；固定 EDM checkpoint 與 BoQ weights 已包在 portable
runtime，換場域不需也不得另行取得或替換。

起飛、降落與即時飛行控制只能由現場操作員在桌面 UI 執行。任何 agent 或自動化
都不得代為起飛。請先完成離線影片、地面、模擬與拆槳測試。

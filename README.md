# 場域地圖定位系統

一套 EDM 定位演算法 + 一個操作介面，場域可隨時更換。演算法只有一份實體，
每個場域是一個獨立資產包，換場域＝換一個 site profile。

Git 版控不包含實際場域點雲、localization bundle、影片、飛行紀錄或模型權重；
本機 workspace 會在被忽略的資料目錄保存它們。系統架構、目錄所有權與相容鏡像
規則見 [`文件/ARCHITECTURE.md`](文件/ARCHITECTURE.md)。

最後整理：2026-08-07。歷史設計決策與安全需求見 [`文件/SYSTEM_SPEC.md`](文件/SYSTEM_SPEC.md)；
目前可執行的場域與發布契約以本 README、site profile schema 與 preflight 為準。最近一次
結構／容量與優化稽核見 [`文件/WORKSPACE_AUDIT.md`](文件/WORKSPACE_AUDIT.md)。

```text
<workspace-root>/                   ← workspace root，同時是 git repo
├── 定位演算法/       ★唯一一份定位演算法（EDM runtime、飛控、安全邏輯、驗證與建圖工具）
├── 控制介面程式/     site profile、操作 UI、兩個串流接口、任務工具
├── 地圖檔/場域/      ★每個場域一包（maps / bundles / routes / reports），不納入 Git
├── 模擬器/           測試影片、Sphinx 實驗
├── 執行環境/         torch_hub_cache、requirements、舊 package git
├── outputs/          flight_logs、benchmark 產物、實驗決策鏈
├── 文件/             系統規格、架構邊界、工作區稽核
├── tools/            驗證實作、測試與唯讀工作區稽核
├── 驗證系統.sh       唯一的純地面驗證入口
└── .venv/            Python 3.10 執行環境
```

git 只追蹤程式碼、設定與說明；場域資產、影片、權重與 outputs 不進版控，
但各資料目錄的 `README.md` 例外保留，以固定用途與保留規則。

工作區整理後可用同一個唯讀入口重查，不會刪除或搬動資料：

```bash
python tools/workspace_audit.py --strict-output-names
```

## 兩個執行接口

| 接口 | 唯一入口 | 影像來源 | 飛控 backend |
|---|---|---|---|
| 模擬串流 | `控制介面程式/影片模擬串流/選擇啟動.sh` | 本機影片 / FFmpeg | 模擬，永不載入 Olympe |
| 實機飛行 | `控制介面程式/真機串流/啟動.sh` | ANAFI PDRAW | Olympe，僅操作員可按 UI 起飛 |

兩個入口互斥。模擬入口拒絕 `--live` / `real-flight`；實機入口拒絕
`--video`，且 PDRAW 不可用時不會拿錄影檔冒充實機畫面。

## 換地圖

在操作介面的「場域資產」區選取一個完整建圖資料夾即可原子化匯入。建圖端必須
一起輸出 `site_profile.json`、顯示點雲 `.ply`、EDM 定位 bundle `.pt`、
EDM runtime profile `.json` 與參考影像位姿 `.json`；缺少任何一項或 SHA-256、
相機、座標系不一致時，整包都不會匯入。PLY 的 XYZ 已包含座標值，但軸向語意必須
寫在 `site_profile.json.coordinate_frame`，不需要額外的軸向檔案。

預畫航線與電桿／目標物不屬於基本定位包，分別由另外兩個接口選配匯入。匯入完成
後，換場域只需切換 site profile，演算法與 UI 不需修改：

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
authoritative controller，但排除 `.venv`、outputs、地圖與影片：

```bash
python tools/export_simulator_package.py /path/to/portable_localization
cd /path/to/portable_localization
python tools/package_manifest.py verify

# 回到來源工作區，讓發布收據綁定實際輸出包的 manifest/hash
cd /path/to/source/localization
./驗證系統.sh --portable-package /path/to/portable_localization

# 正式交付再加入乾淨重建與實際 GUI smoke 證據
./驗證系統.sh --portable-package /path/to/portable_localization \
  --clean-install --ui-smoke

# 在新電腦建立乾淨 CPython 3.10 虛擬環境並安裝固定套件
bash tools/install_runtime.sh

# 驗證／測試所需套件也從 hash lock 安裝（不使用 system site packages）
bash tools/install_runtime.sh --test-deps

# 匯入完整地圖場域包到地圖檔/場域/<site>/，再由選擇介面挑選地圖與影片
./控制介面程式/影片模擬串流/選擇啟動.sh
```

正式驗證會將固定場域與短影片暫時匯入 actual portable，使用該包的
lock 檔建立乾淨環境，從其 selector/UI 產生有效 pose 後自動清理；
因此 receipt 同時證明雜湊及實際輸出包可執行。

選擇介面也保留直接選取現有有效 site profile 的相容入口，但 PLY 不能單獨定位。
對新建圖端的一鍵匯入契約，profile、PLY、EDM bundle、runtime profile 與
reference poses 五項都是必需；`query_camera` 與 `coordinate_frame` 必須寫在
profile，四個資產都必須有 SHA-256。`route_json` 與 `poles_json` 仍是獨立的
overlay／任務選配。完整格式見
[`控制介面程式/site_profiles/建圖端輸出規格.md`](控制介面程式/site_profiles/建圖端輸出規格.md)。
`requirements.txt` 是根目錄的單一 Python
相依來源，安裝器會使用帶 transitive pins 與 hashes 的 `requirements-lock.txt`；
`tools/simulator_preflight.py --full-runtime` 會在 GUI 前實際載入
地圖 bundle、EDM CUDA model、MegaLoc、GUI 與 worker。正式發布目前只支援已驗證的
NVIDIA RTX 5060 + CUDA 12.8；其他 GPU 必須重新做 CUDA、模型與效能驗證。系統層仍需
CPython 3.10 的 `venv/ensurepip`（Ubuntu/Debian 通常是 `python3.10-venv`）、
`ffmpeg`、`python3-tk`、X11/XWayland 與 NVIDIA CUDA driver。
首次執行安裝器需要網路連線，並需預留數 GB 空間下載 PyTorch/CUDA 與其他固定
Python wheels；安裝完成後，模擬介面可在模型與場域資產都已備妥時離線執行。

`scipy` 目前刻意不在 runtime hash lock。`SparseCloudCollisionMonitor` 的 guarded
import 只提供非 production 的稀疏點雲警告，且沒有接入 autonomous safety；因此
`tools/simulator_preflight.py --json` 會在
`runtime.collision_monitor` 明確記錄 `status=unavailable`（即使某個開發環境碰巧
能 import scipy）、`production_safety=false` 與
`collision_protection_claim=false`。需要此 monitor 作為 production safety 時，必須
先完成 production safety wiring 的審查，以及 scipy 版本／平台 wheel hash lock，
再以 `--require-collision-monitor` 執行 fail-closed preflight；沒有這些證據不得
宣稱具備 collision protection。

## 現有場域

| site profile | 資產包 | runtime profile | 航線 | 狀態 |
|---|---|---|---|---|
| `urai_edm.json` | `地圖檔/場域/urai/` | 共用 | **無** | 地面定位可用；自主飛行未核准 |
| `river_site_edm.json` | `地圖檔/場域/river_site/` | `edm_profiles/river_site.json` | 顯示用 overlay | 地面定位可用；overlay 不可飛 |
| `football_field_edm.json` | `地圖檔/場域/football_field/` | `edm_profiles/football_field.json` | **無** | 待 replay；自主飛行未核准 |
| `example_site_edm.json` | — | — | — | 新場域範本 |

全部使用 EDM。XFeat / LighterGlue 的地圖、bundle 與設定已於 2026-07-26 移除
（程式碼保留）。`urai`（烏來）就是交付包代號 `target_site` 的實體場域。
目前三個實際場域的 `flight.approved` 全是 `false`，因此沒有任何一個場域能
通過自主飛行入口；這是刻意的 fail-closed 狀態。

## 新增場域

1. 建圖端建立單一資料夾，放入 `site_profile.json`、PLY、EDM bundle、
   EDM runtime profile 與 reference poses 五個必需檔案。詳細 schema 與固定入口
   見 [`建圖端輸出規格`](控制介面程式/site_profiles/建圖端輸出規格.md)。
2. 由操作介面的「① 場域建圖資料夾」選取該資料夾；介面完整驗證後才會原子化
   複製到 `地圖檔/場域/<site_id>/`。`localizer_deploy_dir` 仍指向
   `定位演算法/`，不要在場域包裡放演算法副本。
3. 固定 EDM checkpoint 與 MegaLoc weights 已由 portable runtime 提供；換場域時
   不得替換或再放一份場域副本。
4. 沒有 EDM bundle 的場域要先建：EDM 是 detector-free，無法沿用 XFeat bundle，
   必須用 `定位演算法/EDM工具包/build/build_reloc_map_edm.py` 對同一組 COLMAP
   位姿做固定位姿重三角化。需要原始影像與 COLMAP model。
   `EDM工具包` 是來源工作區的建置工具，不包含在 portable runtime；請先完成場域包，
   再由五檔資料夾接口匯入。route 與巡檢目標依任務需求使用各自接口匯入。
5. 以統一任務入口做編修與地面驗證：

   ```bash
   python 控制介面程式/mission_pipeline.py \
     --site-profile /absolute/path/to/site.json --mode dry-run
   ```

6. 資產定稿後，對 PLY、bundle、reference poses 與 runtime profile 填入
   SHA-256；另外匯入的 route 或巡檢目標由接口更新其路徑與 SHA-256。
   自主飛行還必須另外完成座標系 ID、航線淨空核准、無尺度控制參數、
   真機 PCMD 響應及速度回授測試；
   完整契約見 [`控制介面程式/site_profiles/README.md`](控制介面程式/site_profiles/README.md)。

### 一包一座標系

**同一個實體場地的不同次 SfM 重建，座標系不通用**（scale-free 也 gauge-free，
要對齊得解 Sim3）。`map_ply`、`localization_bundle`、`map_reference_poses` 與
`route_json` 必須全部來自同一次重建。

混用的症狀很隱蔽：定位數值完全正常（inliers 95–108）但軌跡畫在錯誤的位置。
驗證法是比對 bundle 的 `ref_centers` 與該次重建 `final_model` 的相機中心，
逐張距離應在 1e-7 量級。

## 固定的 EDM 正式參數

`定位演算法/configs/edm_production_profile.json` 是共用的正式設定：1024×576、
PyTorch CUDA FP16、coarse top-k 3225、confidence 0.2、reference tensor cache 32、
TRACK/WEAK/LOST top-k 1/3/5、BOOT MegaLoc top-k 10（先驗證前 2 張，不足才展開）、
batch size 2、LOST grace 12、recovery bank/scan 192/2、correspondence 上限 900、
inliers 80/50/30。MegaLoc 只在 BOOT 與每個 LOST episode 各執行一次，
temporal reference 關閉，PnP acquire/track/RANSAC gate 固定為 5/6/5，
capture-time 預測上限為 0.25 秒。

### coarse tail 融合（不改任何參數）

matcher 的 kernel 實作換過，**參數一個都沒動**。上游 coarse head 會實體化一個
9216×9216 的 fp32 confidence matrix（每個 batch element 324 MiB），分成 exp、兩次
L1 normalize、相乘四個 kernel，之後再讀一次做 row max。`edm_matcher.py` 在
`_import_edm()` 內把這段換成一個 `torch.compile` 融合的 reduction，矩陣不再落地。

RTX 5060 Laptop、真實 720p 影格實測：b=1（TRACK，一張 reference）41.52 → 25.57 ms，
峰值記憶體 1457 → 494 MiB；b=2（WEAK/LOST 的 reference 配對）89.50 → 57.36 ms。
選出的 match **集合完全相同**（3095/3095 共同、0 只在單邊；逐列 argmax 0/3225 不一致；
mconf 最大差 4.8e-07），只有 `torch.topk` 在 ~5e-7 等值處的 tie-break 順序不同。

`SFM_EDM_FUSED_COARSE=0` 可退回上游實作。編譯產物存在 `執行環境/inductor_cache/`
（可重建，冷啟約 5 秒，已排除版控）；`SFM_EDM_COARSE_WARMUP_BATCHES`（預設 `1,2`）
控制啟動時預熱哪些 batch size — 一次只進一張 query，所以 batch 維度是 reference 數，
上限就是 `match_batch_size`。

**但有五個欄位是地圖尺度相依的**，不能跨場域共用：`radius`、`max_jump`、
`adaptive_jump_floor` / `_bootstrap` / `_ceiling`，係數分別是
0.16 / 0.40 / 0.0006 / 0.004 / 0.0016 乘上該場域的
`S = 2·p95(‖center − componentwise_median‖)`。共用檔的值是照 urai 的尺度定的
（S = 5.007236），其他場域沒重算就會鬆掉：

| 場域 | refs | S |
|---|---:|---:|
| urai | 1383 | 5.007236 |
| river_site | 454 | 1.900843 |
| football_field | 505 | 1.840396 |

需要校正時在 `定位演算法/configs/edm_profiles/<site>.json` 建一份場域專屬 profile。

## 模擬實機串流

錄影檔的碼率是實機無線鏈路的 5–12 倍，直接 replay 會高估定位表現。
`ANAFI_LINK_SIM=1` 會依 ANAFI v1.4 白皮書 §5.2 的串流契約重新編碼
（720p、H264 main profile、5 Mb/s、45 slices × 16 px、periodic intra-refresh）：

```bash
ANAFI_LINK_SIM=1 ANAFI_LINK_LATENCY_MS=280 ANAFI_LINK_LOSS_PCT=1.0 \
  ./控制介面程式/影片模擬串流/啟動.sh <video>
```

`SFM_HOLD_ON_LOW_CONF=1` 為精度優先模式：連續低信心即暫停串流（等同懸停），
held frame 改走 LOST recovery（提高 local top-k + 每個 episode 一次 MegaLoc）。
預設開啟。實機端對應的是 `SFM_GATE_WEAK`（預設開啟，WEAK fix 直接 hover）。

## 共享記憶體傳幀

畫面透過共享記憶體送給 localizer worker（`SFM_SHARED_FRAMES=0` 可改用 pipe，較慢）。

worker 只是**附加**這塊由操作介面建立的記憶體，所以 `attach_frame_shm()` 會把它從
CPython 的 `resource_tracker` 取消註冊。不這樣做的話，任何被殺掉的 worker 會在退出時
unlink 掉這塊記憶體 —— 而 client 只在 `__init__` 建立一次、重啟不重建，於是之後每個
worker 都會 `FileNotFoundError: /psm_*` 而死，定位完全停擺（`candidate_mode` 全 null）。
由建立者負責 unlink，worker 只負責 detach。

worker 啟動要載入數百 MB 的 bundle 與模型，可能超過 stall timeout 而被重啟；
`SFM_WORKER_WARMUP_S`（預設 20 秒）可以放寬啟動寬限，大型 bundle 建議 90。

worker 自己的錯誤在 `/tmp/sfm_live_localizer_worker.log`，操作介面的 stdout
只會顯示 `restarted after stall/exit`，看不出原因。

## 演算法只有一份

| 路徑 | 角色 |
|---|---|
| `定位演算法/deploy_code/sfm_glomap_deploy/` | 唯一實體，UI worker 與 site profile 都指這裡 |
| `定位演算法/flight_control/` | 飛控鏡像，改一邊要同步 |
| `定位演算法/EDM工具包/deploy` `runtime` | symlink → `deploy_code/` |

模型權重在 `定位演算法/deploy_code/runtime/EDM/weights/`（不進版控）。
鏡像配對的權威清單在 `定位演算法/validation/check_runtime_mirrors.py`——
同名檔案不代表是鏡像，只有列在那裡的才是。

## 驗證

```bash
./驗證系統.sh

# 選配：完整解碼 P119，並明確接受已核准的 2935/2934 已知例外
./驗證系統.sh --p119-integrity --accept-p119-known-incomplete
```

入口使用主 Python 3.10 與獨立 Python 3.11 `parrot_stimulate` 環境，執行
pytest、ruff、mirror、flight selftest、dependency、profile/SHA、CUDA 與離線模型檢查，
並寫入 `outputs/validation_receipts/`。任一必要項失敗時整體 exit code 非零。
CI 與 `tools/test_clean_install.sh` 由 `requirements-test-lock.txt` 提供固定的
pytest-timeout、coverage 與 pytest-cov；測試使用每測試 300 秒上限並收集 coverage，
但不設不切實際的 coverage release threshold。validation receipt 會彙整 pytest
各 step 的 conditional skip，未執行的測試不會被當成通過。

`parrot_stimulate` 明確要求 Python 3.11，因此使用自己的 `.venv` 驗證，
不由根目錄的 Python 3.10 pytest 跨版本收集。實際通過數以當次輸出為準。

## 真機飛行前的外部必要條件

程式端已改成缺任一條件就拒絕自主飛行，但下列資料必須由現場量測／安全審查產生，
不能由程式自動猜測：

- 永久不建立 `map_units_per_meter`。路徑只提供 map-space 方向與相對進度；
  實際速度由真機新鮮遙測保護，初始上限 0.30 m/s。
- `urai` 與 `football_field` 的 `route_json` 是 `null`，真機入口會拒絕啟動。
  `river_site` 的既有路線只有顯示用途，也會被拒絕。飛行 route 必須使用
  `sfm-flight-route/v1`，綁定同一 `site_id`、`coordinate_frame_id` 與資產 digest。
- **航線淨空核准。** 稀疏點雲不是可靠障礙物模型；必須由現場審查核准整條走廊，
  才能設定 `route_clearance_approved=true`。
- props-off `--yaw-sign` bench test 尚未執行（列為強制）。
- EDM 從未在真機上跑過，所有驗證都是離線 replay。
- 尚無 SHA 固定的 ANAFI 4K、SkyController 3、firmware 與 Olympe 核准 receipt。
- 自主路徑所有入口目前無條件 `LOCKED`；即使手寫 `approved=true` 也不會連機。

舊公尺換算路徑控制器已從可執行入口移除。新的無尺度方向／飛機速度保護核心
只供模擬與單元驗證，未經外部核准前不會連接真機自主飛行。

ONNX 匯出／簡化依賴與 Olympe 的 protobuf 版本不相容，不可裝在同一個 runtime
virtualenv。主 `.venv` 可保留不衝突的推論用 ONNX Runtime；匯出工具使用
`定位演算法/deploy_code/runtime/EDM/deploy/requirements_deploy.txt` 建立獨立環境。

## 注意事項

「地圖」不只是 PLY 點雲。定位還需要同場域建置的 localization bundle，以及和
影片／相機一致的內參；固定 EDM checkpoint 與 MegaLoc weights 已包在 portable
runtime，換場域不需也不得另行取得或替換。

起飛、降落與即時飛行控制只能由現場操作員在桌面 UI 執行。任何 agent 或自動化
都不得代為起飛。請先完成離線影片、地面、模擬與拆槳測試。

真機介面提供 ANAFI 與 SkyController 3 各自獨立的韌體羅盤校正按鈕。只有操作員
在飛控明確回報 `landed` 時才能啟動；校正只要求使用者手持旋轉設備，不會啟動
馬達或自行起飛。校正狀態未知、必須校正、失敗或進行中時，起飛與自主入口會
保持鎖定。

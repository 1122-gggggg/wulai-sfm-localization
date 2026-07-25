# 場域地圖定位系統

一套 EDM 定位演算法 + 一個操作介面，場域可隨時更換。演算法只有一份實體，
每個場域是一個獨立資產包，換場域＝換一個 site profile。

最後整理：2026-07-26。

```
/home/allen/localization/            ← workspace root，同時是 git repo
├── 定位演算法/       ★唯一一份定位演算法（EDM / XFeat / 飛控鏡像 / 建圖驗證工具）
├── 控制介面程式/     操作 UI、兩個串流接口、site_profiles/（每個場域一個 json）
├── 地圖檔/場域/      ★每個場域一包（maps / bundles / routes / reports）
├── 模擬器/           測試影片、Sphinx 實驗
├── 候補定位方法/     NeuFlow、projection-guided、ONNX/TensorRT 的實作索引
├── 執行環境/         torch_hub_cache、requirements、舊 package git
├── outputs/          flight_logs、benchmark 產物、實驗決策鏈
├── 封存/             歷史交付包程式碼與整理前快照
└── .venv/            Python 3.10 執行環境
```

git 只追蹤程式碼、設定與說明；場域資產、影片、權重、outputs 與封存都不進版控。

## 換地圖

換場域只需要換 site profile，演算法、參數、UI 都不動：

```bash
# 離線影片驗證
SFM_SITE_PROFILE=控制介面程式/site_profiles/river_site_edm.json \
  ./控制介面程式/影片模擬串流/啟動.sh 模擬器/測試影片/河濱_P1180118.MP4

# 不設 SFM_SITE_PROFILE 時預設 urai_edm.json
./控制介面程式/影片模擬串流/啟動.sh 模擬器/測試影片/反向跳變最嚴重.MP4
```

真機入口沒有預設場域，一定要顯式指定 profile，且該 profile 必須有與地圖同座標系、
已由操作人員驗證過的安全航線（`route_json` 為 `null` 時只能離線 replay）。

## 現有場域

| site profile | 資產包 | runtime profile | 航線 | 狀態 |
|---|---|---|---|---|
| `urai_edm.json` | `地圖檔/場域/urai/` | 共用 | 無（`route_json: null`） | 正式，2026-07-19 驗證 |
| `river_site_edm.json` | `地圖檔/場域/river_site/` | `edm_profiles/river_site.json` | `river_site_safezone` | 可用 |
| `football_field_edm.json` | `地圖檔/場域/football_field/` | `edm_profiles/football_field.json` | 無 | 2026-07-25 整合，待 replay 驗證 |
| `example_site_edm.json` | — | — | — | 新場域範本 |

全部使用 EDM。XFeat / LighterGlue 的地圖、bundle 與設定已於 2026-07-26 移除
（程式碼保留在 `定位演算法/`）；`urai` 就是交付包代號 `target_site` 的實體場域。

## 新增場域

1. 建 `地圖檔/場域/<site>/{maps,bundles,routes,reports}/`，放進該場域的點雲、
   定位 bundle、參考位姿與安全航線。
2. 複製 `控制介面程式/site_profiles/example_site_edm.json`，改成你的相對路徑與相機內參。
   `localizer_deploy_dir` 與 `localizer_profile` **保持指向 `定位演算法/`**，
   不要在場域包裡放演算法副本。
3. 沒有 EDM bundle 的場域要先建：EDM 是 detector-free，無法沿用 XFeat bundle，
   必須用 `定位演算法/EDM工具包/build/build_reloc_map_edm.py` 對同一組 COLMAP
   位姿做固定位姿重三角化。需要原始影像與 COLMAP model。
4. 先用離線影片與地面測試驗證，再談真機。

## 固定的 EDM 正式參數

`定位演算法/configs/edm_production_profile.json` 是所有場域共用的正式設定：
1024×576、PyTorch CUDA FP16、coarse top-k 3225、confidence 0.2、reference tensor
cache 32、TRACK/WEAK/LOST top-k 1/3/5、BOOT MegaLoc top-k 10（先驗證前 2 張，
不足才展開）、batch size 2、LOST grace 12、recovery bank/scan 192/2、
correspondence 上限 900、inliers 80/50/30。MegaLoc 只在 BOOT 與每個 LOST episode
各執行一次，temporal reference 關閉，PnP acquire/track/RANSAC gate 固定為 5/6/5，
capture-time 預測上限為 0.25 秒。

實測驗證數據留在 `定位演算法/configs/edm_profiles/`（RTX 5060/5090 語料結果）。

**但有五個欄位是地圖尺度相依的**，不能跨場域共用：`radius`、`max_jump`、
`adaptive_jump_floor` / `_bootstrap` / `_ceiling`，係數分別是
0.16 / 0.40 / 0.0006 / 0.004 / 0.0016 乘上該場域的
`S = 2·p95(‖center − componentwise_median‖)`。共用檔裡的值是照 target_site
的尺度 5.0 定的；新場域沒重算就會掉回這些預設。需要校正時在
`定位演算法/configs/edm_profiles/<site>.json` 建一份場域專屬 profile，
site profile 的 `localizer_profile` 指過去（例：`football_field.json`，
S = 1.840396，未校正時鬆 2.72 倍）。

## 演算法只有一份

| 路徑 | 角色 |
|---|---|
| `定位演算法/deploy_code/sfm_glomap_deploy/` | 唯一實體，UI worker 與 site profile 都指這裡 |
| `定位演算法/flight_control/` | 飛控鏡像，改一邊要同步（`sync_mirror_check.sh`） |
| `定位演算法/EDM工具包/deploy` | symlink → `deploy_code/sfm_glomap_deploy` |
| `定位演算法/EDM工具包/runtime` | symlink → `deploy_code/runtime` |

模型權重在 `定位演算法/deploy_code/runtime/EDM/weights/`（不進版控）。

## 路徑解析

程式透過 `控制介面程式/workspace_layout.py` 偵測 workspace 根目錄
（同層需有 `控制介面程式` + `定位演算法` + `地圖檔`）。
可設環境變數：`export SFM_WORKSPACE_ROOT=/home/allen/localization`

## 輸出與保留策略

測試輸出索引與保留項目見 `outputs/README.md`；成功、失敗、淘汰與中斷結果的
決策鏈見 `outputs/EXPERIMENT_LEDGER.md`。CodeGraph 只索引 source/profile/test，
排除 `.venv`、`env` 與 generated outputs。

## 注意事項

「地圖」在本系統中不只是 PLY 點雲。定位還需要由相同場域建置出的 localization
bundle，以及和影片/相機一致的內參。EDM 執行也需要另行取得模型權重。

起飛、降落與即時飛行控制僅能由現場操作人員在 UI 執行。請先使用離線影片與地面
測試驗證新場域設定。

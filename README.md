# 場域地圖定位系統

一套 EDM 定位演算法 + 一個操作介面，場域可隨時更換。演算法只有一份實體，
每個場域是一個獨立資產包，換場域＝換一個 site profile。

本倉庫不含任何實際場域的點雲、localization bundle、影片、飛行紀錄或模型權重。
系統架構、目錄所有權與相容鏡像規則見 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

最後整理：2026-07-26。

```text
/home/allen/localization/            ← workspace root，同時是 git repo
├── 定位演算法/       ★唯一一份定位演算法（EDM runtime、飛控、安全邏輯、驗證與建圖工具）
├── 控制介面程式/     site profile、操作 UI、兩個串流接口、任務工具
├── 地圖檔/場域/      ★每個場域一包（maps / bundles / routes / reports），不納入 Git
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

正式任務模式不會猜測預設場域。未指定 profile 時 fail closed；舊的 per-asset 參數
只能在明確加上 `--allow-legacy-assets` 的遷移作業中使用。

## 現有場域

| site profile | 資產包 | runtime profile | 航線 | 狀態 |
|---|---|---|---|---|
| `urai_edm.json` | `地圖檔/場域/urai/` | 共用 | **無** | 正式，2026-07-19 驗證 |
| `river_site_edm.json` | `地圖檔/場域/river_site/` | `edm_profiles/river_site.json` | `river_site_safezone` | 可用 |
| `football_field_edm.json` | `地圖檔/場域/football_field/` | `edm_profiles/football_field.json` | **無** | 2026-07-25 整合，待 replay 驗證 |
| `example_site_edm.json` | — | — | — | 新場域範本 |

全部使用 EDM。XFeat / LighterGlue 的地圖、bundle 與設定已於 2026-07-26 移除
（程式碼保留）。`urai`（烏來）就是交付包代號 `target_site` 的實體場域。

## 新增場域

1. 建 `地圖檔/場域/<site>/{maps,bundles,routes,reports}/`，放進該場域的點雲、
   定位 bundle、參考位姿與安全航線。
2. 複製 `控制介面程式/site_profiles/example_site_edm.json`，改成你的相對路徑與相機內參。
   `localizer_deploy_dir` 與 `localizer_profile` **保持指向 `定位演算法/`**，
   不要在場域包裡放演算法副本。
3. 將 EDM 權重放到 `定位演算法/deploy_code/runtime/EDM/weights/`。
4. 沒有 EDM bundle 的場域要先建：EDM 是 detector-free，無法沿用 XFeat bundle，
   必須用 `定位演算法/EDM工具包/build/build_reloc_map_edm.py` 對同一組 COLMAP
   位姿做固定位姿重三角化。需要原始影像與 COLMAP model。
5. 以統一任務入口做編修與地面驗證：

   ```bash
   python 控制介面程式/mission_pipeline.py \
     --site-profile /absolute/path/to/site.json --mode dry-run
   ```

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
預設關閉。實機端對應的是 `SFM_GATE_WEAK`（預設開啟，WEAK fix 直接 hover）。

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
pytest -q
python 定位演算法/validation/check_runtime_mirrors.py
python 控制介面程式/mission_pipeline.py --mode flight-selftest
```

## 真機飛行前的未完成項

離線定位已驗證，但**尚不足以進行自動飛行**：

- **地圖沒有公制尺度。** 這是 scale-free 單目 SfM 地圖，一個 map unit 沒有公尺
  對應值。航線是以 map unit 畫的，而 geofence（`--max-altitude-m`、
  `--max-distance-m`）是公尺。必須先在現場量測建立 scale anchor。
- `urai` 與 `football_field` 的 `route_json` 是 `null`，真機入口會拒絕啟動。
  舊的 glomap 座標系航線搬不過去（`T_align.json` 只是 Y-up→Z-up 軸交換，非 Sim3）。
- props-off `--yaw-sign` bench test 尚未執行（列為強制）。
- EDM 從未在真機上跑過，所有驗證都是離線 replay。

## 注意事項

「地圖」不只是 PLY 點雲。定位還需要同場域建置的 localization bundle，以及和
影片／相機一致的內參；EDM 也需要另行取得模型權重。

起飛、降落與即時飛行控制只能由現場操作員在桌面 UI 執行。任何 agent 或自動化
都不得代為起飛。請先完成離線影片、地面、模擬與拆槳測試。

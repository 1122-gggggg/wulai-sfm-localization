# 可替換任務元件設計

系統以 `sfm-mission-selection/v1` 選擇一次任務所需的元件。選擇器只組合元件，
不靠檔名推測地圖或定位演算法；每份 manifest 與實際資產都由 SHA-256 綁定。
既有啟動器暫時仍讀取 `site_profile`，resolver 會從已驗證的 selection 產生不可變
相容快照，因此遷移期間只有一套真實元件身分。

## 元件與更換邊界

| 元件 | manifest | 更換時必須重做 | 不必跟著更換 |
|---|---|---|---|
| 無人機 | `sfm-vehicle/v1` | 機型能力、串流相機 pipeline、限制、明確宣告的校正 receipt | 地圖、路徑 |
| 場域 | `sfm-site/v1` | 穩定的場域 ID 與 site frame | 定位演算法 |
| 地圖版本 | `sfm-map-revision/v1` | PLY、reference poses、gravity alignment、地圖 frame ID | 無人機 adapter |
| 定位器版本 | `sfm-localizer-variant/v1` | bundle、runtime profile、相機相容清單、quality receipt | 路徑 |
| 路徑 | `sfm-route-package/v1` | 路徑檔、frame ID、操作員現場淨空確認 | 定位器程式 |
| 舊場域對齊 | `sfm-site-alignment/v1` + `sfm-calibration-receipt/v1` | raw-map AUTO 不使用；僅保留歷史相容 | 全部作用中元件 |
| 舊相機／機身外參 | `sfm-camera-body-extrinsic/v1` + `sfm-calibration-receipt/v1` | raw-map AUTO 不使用；相機光心直接作導航中心 | 地圖、路徑 |
| IMU | `sfm-calibration-receipt/v1` | 只有 vehicle manifest 明確要求時才檢查；目前 ANAFI 不要求靜態 receipt | 場域、地圖 |
| 舊任務核准 metadata | `sfm-mission-approval/v1` | 選配稽核資料；不參與 readiness | 全部作用中元件 |

`mission_resolver.py` 會檢查地圖與場域、定位器與地圖版本、pose contract、相機
pipeline、無人機能力、manifest 明確要求的 calibration／quality receipt、路徑 frame，
以及所有選定檔案的 SHA。舊 mission approval 欄位仍可解析，但不影響 readiness。

## Raw-map 6DoF 導航（AUTO）

導航位置直接採 localizer 的相機中心 `C_M`，route waypoint 也必須在相同 map
revision/frame。runtime 不再組合 `map→site` 或 camera-body transformation，亦不使用
尺度。相機 optical forward 投影到 map manifest 的 gravity horizontal plane 後，作為
機頭水平朝向；相機近乎垂直時 AUTO 沒有有效 heading，保持懸停。

AUTO 任務 resolver 仍要求 vehicle manifest 明確指定的 calibration receipt、current
`localizer_quality` receipt 與相符的 route package。路線現場淨空由操作員在當次
preflight 第三步確認，不要求靜態 mission approval。`site_alignment` 與
`camera_body_extrinsic` 若存在只作舊格式驗證，不再是缺少時的飛行錯誤。

## 目前河濱 selection

作用中的預設選擇檔是：

```text
控制介面程式/mission_selections/river_gluemap_all8_direct_localization.json
```

它選用全八段 GLUEMAP（1,045 refs，frame
`river_gluemap_all8_direct_20260831_d2b8a5304eff`）與 Parrot ANAFI 720p
vehicle slot。來源 receipt 明列 `validation: NONE`；目前只有同資料集單幀 production
runtime 煙霧成功，尚未驗證實際 ANAFI camera pipeline。現況：

- 地面定位：resolver fail closed，等候獨立 holdout 與 camera-pipeline 品質 receipt。
- AUTO：另缺新座標系 route，因此 `flight_ready=false`。

舊 release 保留供人工回復，但不再是預設 selection。


檢查指令：

```bash
.venv/bin/python tools/resolve_mission.py \
  控制介面程式/mission_selections/river_gluemap_all8_direct_localization.json \
  --require valid
```

從 selection 啟動既有操作介面可使用相容 launcher；它只建立快照與啟動 UI，
不會發出起飛或控制命令：

```bash
.venv/bin/python 控制介面程式/launch_mission.py \
  控制介面程式/mission_selections/river_site_official69_localization.json \
  --check-only
```

移除 `--check-only` 並在 `--` 後放既有 operator 參數即可啟動 UI。resolver 的
`flight_ready=true` 只允許 UI 顯示 AUTO 流程，不會自動起飛；操作員仍須完成四步
preflight 並親自按下「自動飛行」。

更換任一元件後，不必手動計算 selection 內的 manifest hash，可建立一份新 selection：

```bash
.venv/bin/python tools/pin_mission_selection.py \
  --output 控制介面程式/mission_selections/<new-selection>.json \
  --selection-id <new-selection> \
  --vehicle <vehicle-manifest.json> \
  --site <site-manifest.json> \
  --map <map-manifest.json> \
  --localizer <localizer-manifest.json> \
  --calibration <localizer-quality-receipt.json>
```

工具不覆寫舊 selection，且新組合未達定位相容時不會建立輸出。route 完成後，
再用相同方式建立下一份 selection，保留可追溯的版本歷史；`--calibration` 可重複
指定各 receipt。`--approval` 只保留舊檔稽核相容性，不影響 readiness。

## 從另一台電腦移入新地圖

最安全的交付單位是原始 `.tar.gz` 與它的 `.sha256`。本次兩檔已放在：

```text
地圖檔/場域/river_site/releases/
├── river_site_official69_map_v000_20260811.tar.gz
└── river_site_official69_map_v000_20260811.tar.gz.sha256
```

定位所需內容至少包含：

```text
map/map.ply
localization/localization_bundle.pt
localization/reference_poses.json
compat/T_align_gravity.json
compat/edm_runtime_profile.json
compat/map_manifest.json
compat/localizer_edm_manifest.json
compat/localizer_quality_receipt.json
```

此外還要帶入或在目標系統選定 `site.json`、`vehicle.json` 與 mission selection。
同時帶入 `map_manifest.json`、`localizer_edm_manifest.json` 與 vehicle/site
manifests；若要 selection-based AUTO，還要帶入 route package。route 對
localization 可為 optional，但 AUTO
必須選定並通過 clearance。移入後先驗證 archive checksum，再執行 resolver；不要
只複製 PLY，因為 PLY 沒有定位 bundle、相機契約或 route frame identity。

新電腦的最小交付清單：

1. 原始 map archive（`.tar.gz`）與對應 checksum（`.sha256`）。
2. map/localizer assets：`map/map.ply`、localization bundle、reference poses、
   gravity alignment、runtime profile，以及兩份 localizer/map manifests 與 quality
   receipt。
3. `site.json`、`map_manifest.json`、`localizer_edm_manifest.json`、vehicle
   manifest 與要使用的 mission selection。
4. 僅在準備 selection-based AUTO 時：route package，以及 vehicle manifest 明確要求的
   校正 receipt；現場淨空由操作員在當次 preflight 確認。

## 更換流程

### 換地圖

1. 新增 map revision manifest，對 PLY、reference poses、map alignment 填入 hash。
2. 建立對應該地圖的 localizer variant；舊 bundle 不得跨重建沿用。
3. 更新 selection 的 map 與 localizer reference/hash。
4. 舊 route 不得跨 map revision 使用；直接在新地圖重畫並綁定新的 frame ID/hash。
5. resolver 通過 localization 後才啟動定位；AUTO 前由操作員在當次 preflight 確認
   新 route 與現場淨空。

本次 official69 地圖是 gauge-free、非公尺尺度，因此已在新地圖重畫唯一 route，
不製造推測比例或 map→site Sim3。

### 換無人機

1. 新增 vehicle manifest，列出實際 adapter、能力、serial allowlist、限制與相機
   pipeline ID。
2. 相機縮放／裁切或內參不同時，必須選擇列出該 pipeline ID 的 localizer variant，
   或重新驗證／建置一份 variant。
3. 依新 vehicle manifest 的 `required_calibrations` 產生必要 receipt；空陣列表示
   不要求靜態校正收據，但真機 firmware calibration gate 仍會執行。
4. 確認相機沒有相對機身的水平 yaw 偏移；raw-map controller 把相機水平朝向直接
   當作機頭朝向，gimbal pitch 近乎垂直時會拒絕 AUTO pose。
5. 更新 selection；連機 preflight 仍須比對實際機身 serial、firmware 與 adapter。
6. 重新執行 resolver，並在下一個真機 session 重新完成四步 preflight。

### 換定位演算法

1. 實作 `LocalizerProvider`，輸出相同的 pose contract；provider API 與 pose contract
   目前都固定為 v1。
2. 在 deployment registry 註冊 provider，不得由 bundle 檔名猜 backend。
3. 建立 localizer variant manifest，明確宣告所需資產、地圖版本、frame ID、支援的
   camera pipeline 與無人機能力。
4. 用 replay／現場資料完成品質 gate，產生 `localizer_quality` receipt。
5. selection 只換 localizer reference/hash。resolver 通過後，既有 UI 與控制器仍只
   接收同一份 pose contract。

### 換路徑

1. route 必須是 `sfm-flight-route/v1`，至少兩點，且 `site_id`、frame ID、units
   必須與 route package 一致。
2. 目前相容飛行 adapter 使用 map-frame route；新地圖可直接用介面重畫。
3. 建立 route package，更新 selection，完成整條路線的現場淨空。
4. 下一個真機 session 必須重新完成路線雜湊與現場淨空確認。

## 維護規則

- 程式碼、vehicle/site/selection 小型 manifest 可進版控；大型地圖與 bundle 留在
  `地圖檔/場域/<site>/releases/`。
- 元件 manifest 使用嚴格 schema，未知欄位直接拒絕；新增欄位要升 schema 版本。
- runtime 只使用 resolver 產生的 immutable snapshot；任務進行中不得熱換元件。
- `site_profile` 是遷移相容層，不再是元件資料的主要來源。
- 地圖、定位器、路徑、必要校正或無人機任何一項變更都會形成新的 immutable snapshot，
  並要求操作員在下一個真機 session 重新完成四步 preflight。

# Site profile

一份 profile 把單一場域的點雲、localization bundle、相機內參與選用的安全航線
綁在一起，避免不同場域的資產被混用。**換場域＝換一個 profile**，演算法與參數不動。

請複製 `example_site_edm.json` 後填實際路徑。路徑用相對於 profile 所在目錄的形式，
不要寫個人電腦的絕對路徑。

## 現有 profile

| 檔案 | 場域包 |
|---|---|
| `urai_edm.json` | `地圖檔/場域/urai/`（烏來，交付包代號 target_site） |
| `river_site_edm.json` | `地圖檔/場域/river_site/` |
| `football_field_edm.json` | `地圖檔/場域/football_field/` |
| `example_site_edm.json` | 範本，指向不存在的 `your_site` |

全部是 EDM；`localizer` 欄位仍支援 `xfeat`，但已無 XFeat 場域資產。

## 欄位

| 欄位 | 說明 |
|---|---|
| `localizer` | `edm` |
| `localizer_deploy_dir` | **一律** `../../定位演算法/deploy_code/sfm_glomap_deploy` |
| `localizer_profile` | 共用 `../../定位演算法/configs/edm_production_profile.json`，或該場域在 `configs/edm_profiles/` 的尺度校正版 |
| `map_ply` / `localization_bundle` | 同場域、同座標系的地圖與定位 bundle |
| `map_reference_poses` | 參考位姿 |
| `query_camera` | 查詢影像的相機模型與內參 |
| `route_json` | 安全航線；`null` 時禁止 LIVE，只能離線 replay |

`localizer_deploy_dir` 與 `localizer_profile` 不要指進場域包內。交付包常常自帶
一份 `deploy/` 與 `profiles/`，指過去等於「換地圖也換了演算法版本與參數」，
這正是這次整理要消除的問題。包內那些副本請存進場域包的 `reports/` 供對照。

## 尺度相依參數（新場域必看）

EDM profile 的 tracker 欄位分兩類。多數是尺度無關的（top-k、inliers 門檻、
reproj error 像素、時間秒數），可以直接共用；但這五個是**地圖單位**，
換場域一定要重算，否則會掉回 dataclass 預設（那是照 target_site 的尺度 5.0 定的）：

| 欄位 | 係數 |
|---|---|
| `radius` | 0.16·S |
| `max_jump` | 0.40·S |
| `adaptive_jump_floor` | 0.0006·S |
| `adaptive_jump_bootstrap` | 0.004·S |
| `adaptive_jump_ceiling` | 0.0016·S |

`S = 2·p95(‖center − componentwise_median‖)`，對該場域 bundle 的所有 reference
中心計算。算好後在 `定位演算法/configs/edm_profiles/<site>.json` 建一份，
site profile 的 `localizer_profile` 指過去。範例見 `football_field.json`
（S = 1.840396，未校正時鬆 2.72 倍）。

## 使用限制

用 `--site-profile` 時不可再混用 `--map-ply`、`--route-json`、`--bundle`、
`--megaloc-cache`、`--localizer-backend`、`--localizer-deploy-dir` 或
`--localizer-profile`，避免跨場域資產與參數錯配（程式會直接報錯）。

## 同一次重建

`map_ply`、`localization_bundle`、`map_reference_poses` 與 `route_json` 必須來自
**同一次 SfM 重建**。不同次重建即使拍同一個場地，座標系也不通用（scale-free +
gauge-free）。混用的症狀很隱蔽：定位數值完全正常，但軌跡畫在錯誤的位置。
驗證法是比對 bundle 的 `ref_centers` 與該重建 `final_model` 的相機中心，
逐張距離應在 1e-7 量級。

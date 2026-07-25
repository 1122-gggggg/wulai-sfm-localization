# 場域包總覽

每個子目錄是一個完整場域資產包。內容不進版控（見根目錄 `.gitignore`），
只有這份總覽會被追蹤。全部使用 EDM，XFeat/LighterGlue 資產已於 2026-07-26 移除。

| 場域 | site profile | runtime profile | EDM bundle | 安全航線 | 大小 |
|---|---|---|---|---|---|
| `urai`（烏來/目標場域） | `urai_edm.json` | 共用 `edm_production_profile.json` | 1383 refs | 無（`route_json: null`） | 767M |
| `river_site`（河濱） | `river_site_edm.json` | `edm_profiles/river_site.json` | 454 refs | `river_site_safezone` | 217M |
| `football_field`（足球場） | `football_field_edm.json` | `edm_profiles/football_field.json` | 505 refs | 無 | 199M |

## 各包狀態

- **urai** — 現行 EDM v1，2026-07-19 在 RTX 5060 完成 replay 驗證。交付包代號是
  `target_site`，檔名保留該前綴。舊的 glomap/XFeat 重建資產已移除，只留航線
  （在舊座標系，不能直接用於 EDM 地圖）。
- **river_site** — EDM bundle 可用，有安全航線。
- **football_field** — 2026-07-25 整合 EDM 交付包（`ff_a1`）。**尚未跑過影片 replay**。

## 地圖尺度

`radius` / `max_jump` / `adaptive_jump_*` 五個 tracker 欄位是地圖單位，換場域必須重算
`S = 2·p95(‖center − componentwise_median‖)`：

| 場域 | refs | S |
|---|---:|---:|
| urai | 1383 | 5.007236 |
| river_site | 454 | 1.900843 |
| football_field | 505 | 1.840396 |

共用 profile 的預設值是照 urai 的尺度定的，另外兩個場域各有專屬校正檔。

## 一包一座標系

**同一個實體場地的不同次 SfM 重建，座標系不通用。** 單目 SfM 是 scale-free 也
gauge-free，兩次重建的尺度、旋轉、平移都不同，要對齊得解 Sim3。所以 `map_ply`、
`localization_bundle`、`map_reference_poses`、`route_json` 必須全部來自**同一次**
重建——這點吃過虧：足球場曾把 `ff_a1` 的 bundle 配上 `gluemap_dense` 的點雲，
定位數值完全正常（inliers 95–108）但軌跡畫在完全錯誤的位置。

驗證方法：拿 bundle 的 `ref_centers` 與該次重建 `final_model` 的相機中心逐張比對，
距離應該是 1e-7 量級。

## 測試影片

影片不放在場域包內，統一在 `模擬器/測試影片/`，非 urai 的以場域名為前綴。

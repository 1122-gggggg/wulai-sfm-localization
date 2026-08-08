# Site profile 與飛行契約

給建圖端與第一次使用介面的人員：完整五檔輸出、座標軸、manifest、航線與目標物
格式請先看 [建圖端輸出規格](建圖端輸出規格.md)。UI 分成三個明確接口：

1. 必需的場域建圖資料夾。
2. 有航線時才匯入的 route JSON。
3. 有巡檢點時才匯入的電桿／目標物 JSON。

資料匯入本身不會起飛，也不會自動核准自主飛行。

Site profile 原子化綁定地圖、定位 bundle、參考位姿、相機與 runtime profile。
所有現有與未來場域一律必須設定 `localizer="edm"`；schema 會直接拒絕 XFeat 或其他 backend。
地面定位可在 `flight.approved=false` 下使用；自主飛行只有在所有必要欄位與檔案
完整、驗證通過，且 `flight.approved=true` 時才可進入。河濱場域已由操作員核准，
其他隨附場域仍維持未核准。

## 換場域資產分界

| 資產 / 欄位 | 地面定位與介面需求 | 換場域動作 |
|---|---|---|
| site profile JSON | 啟動契約必要 | 新增或替換 |
| `assets.localization_bundle` | EDM 定位核心必要 | 替換為新場域 bundle |
| `query_camera` | portable full-runtime 必要 | 填入與影片解析度/裁切流程一致的內參 |
| `assets.map_ply` | schema、selector 與 UI 必要；不參與 EDM 姿態估計 | 替換顯示點雲 |
| `localizer_profile` | 可使用固定共用預設 | 只有場域特調且重新驗證時替換 |
| `map_reference_poses` | 純定位可省略 | UI 點雲範圍過濾與飛行 readiness 才需要 |
| `route_json` | 純定位可為 `null` | 航線 overlay/飛行才替換 |
| `poles_json` | 純定位可為 `null` | 有 `inspect_waypoints` 才需要 |
| `megaloc_cache` | EDM 不使用 | 保持 `null`；MegaLoc descriptors 已在 bundle `ref_global` |
| `track_landmarks` | XFeat/projection legacy | 正式 EDM 保持 `null` |
| `flight` | 真機自主飛行核准 | 地面模擬維持未核准 |
| `hardware_approval` | 選配的硬體版本收據 | 不作為自主入口必要欄位 |

每個有提供的場域資產都要同步更新 profile 內路徑與 SHA-256。
換場域不要替換固定 EDM checkpoint、MegaLoc weights、deploy code、UI 或
Parrot 模擬器。

## 自主飛行必要條件

- `flight.approved=true`，且 `approval_note` 記錄核准依據。
- `coordinate_frame_id` 唯一識別這一次 SfM 重建；route 必須使用同一 ID。
- schema v2 永久不接受 `map_units_per_meter`，也不從相機或路徑猜測公尺尺度。
- `route_clearance_approved=true` 表示整條航線已由現場安全審查確認淨空。
- `asset_sha256` 必須包含 localization bundle、route、reference poses，
  EDM 場域另包含 localizer profile；啟用巡檢時也必須包含 `poles_json`。
- `flight.controller` 是選配的相容 metadata，不是自主入口必要欄位。production
  runner 使用同一份全域無尺度方向控制設定；場域只提供 route 幾何與 route 自帶的
  `arrive_radius_map_units`，不提供 map-to-metre 比例。
- 有巡檢 waypoint 時仍必須提供同場域的 `poles_json` 與 SHA-256；目前核准的河濱
  route 不啟用巡檢 waypoint。

## 飛行 route 格式

route JSON 必須是開放 polyline，至少兩個互異且有限的三維 waypoint，並包含：

```json
{
  "schema": "sfm-flight-route/v1",
  "site_id": "same-as-site-profile",
  "coordinate_frame_id": "measured-reconstruction-id",
  "frame": "glomap",
  "units": "map",
  "purpose": "flight",
  "closed": false,
  "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
}
```

`frame` 也可為 `aligned`，但必須明確填寫。範例座標只說明結構，不是可飛航線。
任何 route 或資產內容更動後，都必須重新計算 profile 中的 SHA-256 並重新審核。

目前只有 `river_site_edm.json` 已由操作員核准；其他實際場域與範本均維持
`approved=false`。

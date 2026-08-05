# Site profile 與飛行契約

給建圖端與第一次使用介面的人員：完整五檔輸出、座標軸、manifest、航線與目標物
格式請先看 [建圖端輸出規格](建圖端輸出規格.md)。UI 分成三個明確接口：

1. 必需的場域建圖資料夾。
2. 有航線時才匯入的 route JSON。
3. 有巡檢點時才匯入的電桿／目標物 JSON。

資料匯入本身不會起飛，也不會解除自主飛行鎖定。

Site profile 原子化綁定地圖、定位 bundle、參考位姿、相機與 runtime profile。
所有現有與未來場域一律必須設定 `localizer="edm"`；schema 會直接拒絕 XFeat 或其他 backend。
地面定位可在 `flight.approved=false` 下使用；自主飛行只有在所有必要欄位與檔案
完整且驗證通過時才可能解除外部鎖定。目前所有自主路徑入口均為 `LOCKED`。

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
| `flight`、`hardware_approval` | 真機自主飛行專用 | 地面模擬維持未核准 |

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
- `hardware_approval` 必須指向一份 SHA-256 固定的 ANAFI 4K／SkyController 3／
  firmware／Olympe 測試 receipt。
- `flight.controller` 的路徑幾何門檻只使用場域 map units；實際速度只由新鮮的
  飛機端速度遙測保護，不做 map-unit-to-metre 換算。

`flight.controller` 必須完整包含：

```text
model = scale_free_direction_speed_guard_v1
speed_limit_mps
pose_max_age_ms
speed_max_age_ms
command_ttl_ms
yaw_tolerance_deg
horizontal_axes
vertical_axis
camera_to_body_yaw_deg
body_right_sign
lookahead_map_units
rejoin_tolerance_map_units
arrival_tolerance_map_units
inspect_radius_map_units
inspect_resume_margin_map_units
max_pose_jump_map_units
max_route_deviation_map_units
progress_jump_slack_map_units
max_progress_regression_map_units
segment_window
progress_speed_factor
inspect_waypoints
```

`speed_limit_mps` 初始為 0.30；修改只允許在確認落地時進行，且會使舊核准失效。
pose／speed freshness 最多 500 ms，command TTL 最多 250 ms。不要把另一場域的
map-unit 門檻複製過來。`inspect_waypoints` 使用 1-based waypoint 編號；
只要非空，就還必須提供同場域的 `poles_json`。

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

目前隨附的所有實際場域 profile 都刻意維持 `approved=false`。

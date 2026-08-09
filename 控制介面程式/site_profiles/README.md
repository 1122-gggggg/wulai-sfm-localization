# Site profile 與飛行契約

給建圖端與第一次使用介面的人員：完整五檔輸出、座標軸、manifest、航線與目標物
格式請先看 [建圖端輸出規格](建圖端輸出規格.md)。UI 分成三個明確接口：

1. 必需的場域建圖資料夾。
2. 有航線時才匯入的 route JSON。
3. 有巡檢點時才匯入的電桿／目標物 JSON。

資料匯入本身不會起飛，也不會自動核准自主飛行。硬體 receipt 若存在，仍可與
場域、座標系、profile、route、bundle 綁定並解析記錄；但 receipt 不再是 AUTO
readiness 的必要 gate，manual 收據也不會升級核准狀態。

Site profile 原子化綁定地圖、定位 bundle、參考位姿、相機與 runtime profile。
`localizer` 必須是 deployment registry 已註冊的 backend。目前提供 `edm` 與 `xfeat`；
schema 會拒絕未知 backend，並依 provider capabilities 驗證 backend-specific assets。
EDM 需要場域 `localizer_profile`；XFeat 不接受 EDM profile，且兩者都以同一套
bundle SHA 綁定。切換 backend 不會自動升級真機 AUTO approval。

Profile 內所有資產與 `localizer_deploy_dir` 都相對於 JSON 所在位置解析，但必須
留在自動發現的 workspace root 內；`../` 或 workspace 外的絕對路徑，以及含有
symlink 的檔案或父目錄，一律拒絕。若要選擇不同 workspace，請明確設定既有的
`SFM_WORKSPACE_ROOT`，不要依賴隱含的開發環境路徑。
地面定位可在 `flight.approved=false` 下使用；自主飛行只有在所有必要欄位與檔案
完整、驗證通過，且 `flight.approved=true` 時才可進入。目前隨附場域由 profile
各自記錄核准狀態，河濱場域已核准 AUTO，其餘場域仍維持未核准。

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
| `reference_index` | backend-neutral 的大型 reference retrieval index，可選 | EDM/XFeat 可指向 index 的 `SHA256SUMS.json`；同時更新 `asset_sha256.reference_index` |
| `track_landmarks` | XFeat/projection legacy | 正式 EDM 保持 `null` |
| `flight` | 真機自主飛行核准 | 地面模擬維持未核准 |
| `hardware_approval` | 綁定場域與硬體身份的 signed v2 收據 | 可選；若存在則保留 receipt、簽章與 trust store 的解析／記錄 |

每個有提供的場域資產都要同步更新 profile 內路徑與 SHA-256。
換場域不要替換固定 EDM checkpoint、MegaLoc weights、deploy code、UI 或
Parrot 模擬器。

## 自主飛行必要條件

- `flight.approved=true`，且 `approval_note` 記錄核准依據。
- `hardware_approval` 若存在，會保留 `receipt`/`sha256`、detached
  `signature`/`signature_sha256` 與 operator-provisioned `trust_store`/
  `trust_store_sha256` 的解析與記錄；它不是 AUTO readiness 的必要條件。
- `profile_sha256` 是排除 `hardware_approval` 物件後的 canonical profile
  SHA-256，避免 receipt reference 形成循環，同時仍綁定其餘場域與飛行政策。
- `approved_envelope.gps_required=false`，保留現有 GPS 不作為 AUTO readiness 必要條件的政策。
- `coordinate_frame_id` 唯一識別這一次 SfM 重建；route 必須使用同一 ID。
- schema v2 永久不接受 `map_units_per_meter`，也不從相機或路徑猜測公尺尺度。
- `route_clearance_approved=true` 表示整條航線已由現場安全審查確認淨空。
- `asset_sha256` 必須包含 localization bundle、route、reference poses，
  EDM 場域另包含 localizer profile；啟用巡檢時也必須包含 `poles_json`。
  若 profile 提供 `assets.reference_index`，必須指向 index 目錄內的
  `SHA256SUMS.json`，並提供 `asset_sha256.reference_index`。runtime 會再驗證
  manifest 內所有 index 檔案；它是 retrieval 加速資產，不取代 bundle identity
  或 AUTO approval，也不能和 legacy `megaloc_cache` 同時設定。
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

目前河濱 profile 具備 AUTO 核准；烏來的 v2 收據仍只記錄 2026-08-06
手動方向測試，且其他未核准場域維持 `approved=false`。

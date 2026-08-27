# Site profile 與飛行契約

給建圖端與第一次使用介面的人員：完整五檔輸出、座標軸、manifest、航線與目標物
格式請先看 [建圖端輸出規格](建圖端輸出規格.md)。UI 分成三個明確接口：

1. 必需的場域建圖資料夾。
2. 有航線時才匯入的 route JSON。
3. 有巡檢點時才匯入的電桿／目標物 JSON。

資料匯入本身不會起飛，也不會自動核准自主飛行。AUTO 的 route 與 localizer pose
必須直接使用同一次重建的 raw map frame；相機光心直接作為導航中心，不建立
`map→site` 對齊，也不從場域動作猜尺度。舊的 `site_alignment` 與
`camera_body_extrinsic` 檔案仍可被 schema 解析以保留歷史相容性，但 runtime 不載入
或套用它們。另有的 signed `hardware_approval` receipt 若存在，仍可與場域、座標系、
profile、route、bundle 綁定並解析記錄。

Site profile 原子化綁定地圖、定位 bundle、參考位姿、相機與 runtime profile。
`localizer` 必須是 deployment registry 已註冊的 backend。正式隨附場域與 portable
只使用 `edm`；registry 內的 `xfeat` 僅保留研究／舊包遷移，不在 production artifact
allowlist。schema 會拒絕未知 backend，並依 provider capabilities 驗證
backend-specific assets。EDM 需要場域 `localizer_profile`；XFeat 不接受 EDM
profile，且兩者都以 bundle SHA 綁定。切換 backend 會形成新 snapshot，當次 preflight
必須重新完成。

## Raw-map 6DoF 導航契約（AUTO）

- `camera_center_map` 與 route waypoint 都是同一個 `coordinate_frame_id` 的 raw map
  座標；兩者不做旋轉、平移或尺度轉換。
- 相機 6DoF 的 optical forward 會投影到由 `T_align_gravity.json` 定義的水平面；投影
  太短時沒有可靠方位，AUTO 保持懸停。
- 相機水平朝向視為機頭朝向。每一段先原地轉向至目標射線，容許誤差 7.5°；確認
  穩定後只輸出 roll/pitch/gaz。若途中 yaw 誤差超過 30°，立即停止平移、只用 yaw
  重新對準，回到 7.5° 且穩定後才續飛。
- 平移期間每一筆新 pose 都重畫「目前相機中心→目標 waypoint」並重算 PCMD 分量；
  抵達下一點後歸零、重設轉向階段，再飛下一段。
- 上下分量只取目標位移在 map gravity `up` 上的投影；瞬時 camera pitch/roll 不參與
  vertical PCMD，避免機身前傾時把前方 waypoint 誤判成需要爬升。
- vehicle manifest 明確宣告的校正 receipt、current `localizer_quality`、相符的 route
  package 與 component hashes 仍是任務 resolver 的獨立 gate；路線淨空由操作員在
  當次 preflight 確認，不要求靜態 mission approval。

Profile 內所有資產與 `localizer_deploy_dir` 都相對於 JSON 所在位置解析，但必須
留在自動發現的 workspace root 內；`../` 或 workspace 外的絕對路徑，以及含有
symlink 的檔案或父目錄，一律拒絕。若要選擇不同 workspace，請明確設定既有的
`SFM_WORKSPACE_ROOT`，不要依賴隱含的開發環境路徑。
地面定位可在 `flight.approved=false` 下使用；自主飛行只有在所有必要欄位與檔案
完整、驗證通過，且 resolver 判定 mission selection `flight_ready=true` 時才可進入。
目前河濱預設是 B0+P116/P117 fringe 地圖，舊 official69 航線已刪除，尚未重畫
新座標系 route，因此不是 flight-ready。ANAFI 羅盤改由真機韌體即時回讀，校正
完成且狀態有效後自動通過 preflight 第一步；其餘步驟仍由操作員當次確認。

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
| legacy `pose_chain.*` | 可保留歷史檔案；raw-map AUTO 不讀取 | 不需建立或更新 |
| calibration receipt | 只有 vehicle manifest 明確要求時才是 AUTO gate；目前 ANAFI 不要求 | 換 vehicle revision 時依新 manifest 處理 |
| `megaloc_cache` | EDM 不使用 | 保持 `null`；MegaLoc descriptors 已在 bundle `ref_global` |
| `reference_index` | backend-neutral 的大型 reference retrieval index，可選 | EDM/XFeat 可指向 index 的 `SHA256SUMS.json`；同時更新 `asset_sha256.reference_index` |
| `track_landmarks` | XFeat/projection legacy | 正式 EDM 保持 `null` |
| `flight` | 真機自主飛行核准 | route/map 契約與淨空確認後更新 |
| `hardware_approval` | 綁定場域與硬體身份的 signed v2 收據 | 可選；若存在則保留 receipt、簽章與 trust store 的解析／記錄 |

每個有提供的場域資產都要同步更新 profile 內路徑與 SHA-256。
換場域不要替換固定 EDM checkpoint、MegaLoc weights、deploy code、UI 或
Parrot 模擬器。

### 現場 raw-map 起航流程

1. 完成四步起飛前檢查，按「開始定位」，確認定位持續輸出相機中心與 6DoF。
2. 手動或 AUTO 起飛後先穩定懸停；AUTO 會找最近 waypoint，第一個目標是它的下一點。
3. 系統在原地判斷目標射線位於相機水平朝向的左側或右側，只送 yaw PCMD，7.5°
   內且轉速穩定後才解鎖平移。
4. 平移中只重算 roll/pitch/gaz；yaw 誤差超過 30° 才停止平移並重新對準。抵達下一
   點後先歸零再重新轉向。
5. 航線位置以相機光心為準；航線淨空評估仍必須把整個機體尺寸算入。
6. 實體搖桿動作永遠優先，會立即終止電腦非零 PCMD 並取回人工控制。

## 自主飛行必要條件

- selection resolver 必須判定 `flight_ready=true`；相容 profile 的
  `flight.approved=true` 由此結果產生。
- route、localizer pose 與 profile `coordinate_frame_id` 完全相同，且提供有效的
  `map_align` 重力方向；不需要 `site_alignment` 或 camera-body artifact。
- vehicle manifest 明確要求的校正 receipt、route package 與 component hashes 均已通過；
  route clearance 由操作員在當次 preflight 確認。
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
- 有巡檢 waypoint 時仍必須提供同場域的 `poles_json` 與 SHA-256；目前河濱
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

目前河濱 profile 已啟用新地圖定位、唯一的新繪 route 與 raw-map 6DoF 導航；不需
七點座標校正。到現場仍必須通過四步檢查、確認相機水平朝向等同機頭朝向並逐段低速
試飛。烏來與其他場域仍維持 `approved=false`。

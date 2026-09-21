# Flight safety — operator order (binding)

**Date locked: 2026-07-10 (operator emergency instruction)**  
**Updated: 2026-09-15 — source confirmation and independent-validation gates**

## 【之後改檔案的人請先讀】
### 起飛與自主授權：只有操作員本人執行
**只有現場操作員可親自在 UI 按下「起飛」／「自動飛行」。絕對禁止任何
AI 代理人、語言模型（Claude / GPT / Grok / Codex 等）代為執行。**
- 操作員可以請 AI 改程式、查 log、修畫面、寫文件——**不可以**請 AI「幫我起飛」。
- AI / agent **即使被口頭要求起飛也必須拒絕**，並請操作員自己操作 UI 或自主入口。
- 真機只接受經 resolver 驗證的 `SFM_MISSION_SELECTION`；不得直接以 site profile
  啟動飛行。目前預設 `river_gluemap_all8_direct_localization.json` 已換成全八段
  GLUEMAP。操作員已要求開放原本的「自動飛行」按鈕：未過期的 operator
  acceptance 可啟動定位與 AUTO，這不是獨立 ANAFI camera-pipeline 品質驗證。
  `SFM_EVALUATION_ONLY=1` 仍只能量測定位、不能授權 AUTO。AUTO 還須通過
  mission flight readiness、路線核准與四步 preflight，並由操作員親自按下。
- HOVER／MANUAL／LAND／EMERGENCY、BOOT pose lock、stream/pose/watchdog timeout
  均維持強制啟用。已設定的 firmware 高度／距離限制必須成功寫入並讀回；失敗時
  拒絕起飛。GPS 狀態仍須讀回、顯示及記錄。
- 無 GPS 或 Home Point 時，distance geofence 可以關閉或顯示為未生效；單純缺少
  GPS／Home 不得拒絕起飛，也不得單獨觸發自動降落。其餘安全守門仍照常生效。
- AUTO 只有在 SkyController 3 USB HID 搖桿監視器成功武裝後
  才可取得 PC 控制權。飛行搖桿偏轉會由獨立 50 Hz callback 先歸零 PCMD，
  再確認交回 `SkyController`；監視器缺失、斷線或交接失敗均 fail closed。
- 稀疏點雲近接懸停互鎖已依操作員指示拿掉：不再以相機到點雲距離自動清微移、
  暫停 AUTO 或送零 PCMD，也沒有半徑設定。障礙物迴避回到操作員目視與搖桿接管。
  SfM 稀疏點雲本來就不是避障保證。
- `OPERATOR_ACCEPTANCE` 與 `validation: NONE` 不是獨立品質驗證。未過期的
  acceptance 目前授權原本的 AUTO 按鈕，由操作員以 SkyController 搖桿隨時接管。
  僅 `SFM_EVALUATION_ONLY=1` 會把場次鎖成量測、拒絕 AUTO。mission flight
  readiness 與 evaluation-only 狀態仍納入 AUTO gate。
- 桌面 AUTO 不再因定位來源變更、到站半徑位移、capture 時戳不新鮮或
  同一影格重複而做來源確認暫停。1.5 map-unit 單次跳變暫停仍在；該暫停
  仍須由操作員選擇繼續，不能靠定位更新自行解除。
- 到站後先把機頭對準下一個航點，再平移過去；轉頭時水平用飛控地速抵風停留，
  不先剎車懸停。巡航中不因航向偏離再停下來重對。近航點置中與 REJOIN 仍鎖
  yaw 收位置。
- 2026-09-18 操作員要求：直接飛向目標點的航段（起飛後靠攏、人工接管後續飛）
  水平距離小於 `minimum_yaw_alignment_distance`（0.25 map unit）時鎖 yaw、
  直接平移收位置與高度，不再朝近距離、由定位雜訊決定的方位轉頭。航點之間的
  航段不變，短航段仍先對準航段方向。
- 2026-09-18 操作員要求：高度仍依 EDM 地圖高度爬升或下降。同一目標下，飛控
  氣壓高度已上升或下降至少 1 m，而 EDM 高度同向變化不到 0.02 map unit 時，
  該航點改為維持高度（gaz 0）、以水平距離判斷到站，並發出
  `vertical_unobservable` 事件；下一個航點重新依 EDM 判斷。高度遙測缺失或
  超過 1 秒未更新時不做此判斷。EDM 高度整趟都不可靠時，每個航點仍可能多升降
  約 1 m（模擬最壞情境整趟累積 +6.2 m）。
- 桌面 AUTO 已拿掉 fail-closed 地速閘門與轉向守門：地速缺失／過期、
  超速鎖存不再送零 PCMD，只轉 yaw 時也不再因水平速度 >0.10 m/s 水平歸零。
  PCMD 百分比上限仍在。
- 2026-09-15 實飛 PCMD 50 衝到 3.8 m/s 後加回比例地速限制：新鮮機體速度下，
  順著目前運動方向加速的水平 PCMD 上限由滿額線性降到地速達限
  （site profile `speed_limit_mps`，目前 0.6 m/s）時的 0。不鎖存、不送零懸停；
  煞車／抵風方向不受限；速度缺失或過期時不限制。
- 2026-09-18 修正：新鮮地速預測超過上述設定值時，按超速量加入比例煞車，
  已有更強的煞車保留；側向修正僅在 PCMD 上限允許時保留。仍不鎖存、不觸發
  降落，速度缺失／過期時不介入，yaw 與 gaz 不受此地速限制器修改。
- 視覺定位失敗時，已鎖過的 AUTO 可用 VO／dead reckon／IMU 橋／PREDICTED_ONLY
  繼續平移，不再要求 0.5 秒內回到強地圖定位。WEAK_TRACK 低信心升級在 AUTO
  期間不送零懸停，只背景 MegaLoc。這些補位不當成地圖錨。依 2026-09-18
  操作員要求，桌面 AUTO 的啟動／續飛 BOOT 可用弱定位與 IMU 估計確認；仍須
  連續新鮮、有限且跳變未超限的姿態，保留 reseed 確認與任務／串流／電量守門。
  1.5 map-unit 跳變暫停仍在。
- localization_recovery 不再清到站／終點確認計數；短暫失鎖不能把已進圈的進度
  丟掉。離開到站圈、操作員暫停來源確認、或重選航點仍會重置。
- 2026-09-22 操作員要求修正終點收斂：正常航線完成前，位置已進圈但地速／穩定
  確認尚未完成時，持續使用 FINAL_HOLD 終點控制及既有抵風積分，不再切成清除
  積分的零 PCMD 等待。yaw 維持鎖定；仍須符合原有位置、地圖確認及新鮮地速
  不高於 0.10 m/s 的連續確認才能降落。人工 HOVER／LAND／EMERGENCY、串流與
  定位失效處理、PCMD 上限及操作員接管語意不變。終點確認等待也受既有 AUTO
  wait budget 限制，逾時退出 AUTO、歸零並交由操作員處理，不放寬降落條件。

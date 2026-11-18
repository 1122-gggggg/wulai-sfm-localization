# Flight Operator Interface

Purpose: one operator UI behind two explicit, mutually exclusive interfaces:
`--interface simulated-stream` and `--interface real-flight`.

## 單一介面與演算法切換

操作介面唯一來源為 `/home/allen/localization/控制介面程式`，只有一套 UI。
既有 `site_profile.localizer_deploy_dir`（worker 的 `--deploy-dir`）選擇演算法，
選定目錄在 query-camera 驗證與 factory 匯入前生效：
原始版本載入 `定位演算法/deploy_code/sfm_direct_deploy`，擴圖版本載入
`定位演算法/deploy_code_expand/sfm_direct_deploy`；其 `direct_localizer_profile.json`
以 `online_workmap.enabled` 標示定位＋擴圖。
BOOT／LOST、起降、微移與 Esc 邏輯兩個版本都使用本介面。
影片模擬串流的 preflight 保留固定工作區檢查，deploy 目錄只允許上述兩份
（擴圖那份只限 direct profile）；其他目錄與外部 symlink 不因此取得許可。

工作圖點雲沿用原始 RGB，缺色時採固定分類色；trajectory backfill 只修正畫面紀錄，
不回寫 `live_pose` 或控制歷史。session 換代會清除前一張工作圖、顏色與回填軌跡。
workmap 健康／狀態／merge／圖求解欄位沿既有 worker IPC 與 session logs 傳遞。
worker 的正常回覆在發布結果前清除 fatal restart 預算與錯誤，避免 UI 收到舊生命週期狀態。

## MVS 絕對深度檢視

影片右鍵開啟唯讀檢視器（影片左上只有「深度估計」切換鈕）。
「更新配對影格」選取最近一筆同來源影格、且該影格自己算出位姿與完整相機姿態的結果
（弱定位也可）；主串流仍繼續播放，檢視器明確顯示配對影格名稱與更新時的擷取時間差。
不把延遲姿態套到目前較新的畫面，也不以 yaw-only 或保留舊姿態提供深度。
使用未平滑的 `pose_raw` 中心與 `camera_axes_world` 的 right/down/forward 列，
不是 UI 重力對齊後的顯示座標。

點擊配對影像查該像素的公尺 Z 深度；「產生深度圖」顯示最高 640 像素寬的
0–30 m 色階（紅近藍遠），黑色是未知。mesh 載入與深度渲染在獨立 CPU thread 執行，
不加入定位快迴路或 GPU reloc。取消定位、場域切換及關窗會清除檢視狀態。
公尺比例只接受配對影格時可信的 session 尺度，或有校正來源的 manifest 尺度，
不將 map unit 當公尺。查詢只代表靜態地圖中的表面，不代表即時測距或避障。
建圖、相依與純離線深度輸出見 [tools/README.md](../../tools/README.md#離線-mvs-與公尺深度查詢)。

### 主畫面深度估計切換

影片左上「深度估計」切換整個影片窗格：按下後顯示最近一筆配對影格的
深度色圖（紅近藍遠、0–30 m、黑色未知），再按「返回影像」回到原串流。
畫面分成 3×3 格，每格顯示該格最小的公尺 Z 深度，並以星號標出該像素；
全畫面最近的格以紅星標示。深度只在有新的配對結果時重算（單一 CPU thread、
最新優先），上方一行顯示配對影格名稱與擷取距今秒數，超過 2 秒轉琥珀色。
尚無可配對位姿或影格比例與相機校正不符時，只顯示原因，不猜測。尺度尚未可信（現場尺度
未收斂且 `dense_surface.json` 的 `meters_per_map_unit` 為 null）時照樣顯示深度，但單位為
地圖單位 `u`（不是公尺）、色階改為該影格的 0–95 百分位，並以琥珀色註明「尺度未校正」。
切換期間 HUD 與橫幅照常顯示；偵測框與分割只屬於即時影格，深度視圖中不繪製。
此為靜態地圖估計，不是即時測距或避障，也不接入任何飛控或 AUTO 判斷。


## HARD SAFETY (operator order 2026-07-10)

- **起飛只准操作員本人親手按「起飛」或「自動飛行」。**
  **絕對禁止**請 AI 代理人／任何語言模型（Claude / GPT / Grok / Codex 等）代為起飛；
  即使對話中被要求，agent 也必須拒絕。只有這兩個人類 UI 動作可送出 `TakeOff`。
- See `../SAFETY.md`. Automated acceptance is **ground-only** by default
  (`live_non_map_acceptance.py`).
- **【之後改檔案的人】禁止動飛行按鍵／按鈕指令：** 起飛、原地降落、關窗強制降落、
  按住微移/放開懸停、Esc 凍結。**尤其起飛／降落／強制降落**改壞會造成意外。
  細節表在 `../SAFETY.md`。

Recommended for real flight: use the single-process desktop app, not the browser
server. It avoids localhost HTTP/browser state and keeps safety commands inside
one Python process.

Layout:

- The production window constants are `UI_STANDARD_SIZE` = **1440×900** and
  `UI_MIN_SIZE` = **1180×720**. The desktop app and `--layout-selftest` use the
  same values; no lower minimum is supported.

- Left: point-cloud map, planned route, and localization trail. The planned route is always shown when loaded. Mouse controls: left drag 360-degree rotate, double left click sets the rotation pivot like CloudCompare, middle drag roll, right drag pan, wheel zoom. Use `重設地圖` to restore the default upright view.
- Right: ANAFI-like drone video stream. Input stream is resized to `1280x720`
  and paced as 720p30 by default.
- Bottom: persistent flight actions, followed by camera/recording controls and preflight confirmation. A single 操作工作區 shows flight limits, virtual sticks, compass calibration, and site/route assets together; flight status and recent waypoint events sit below the flight settings.

### 操作檢查與飛後數據

- 羅盤校正以韌體 `started`、目前軸、失敗與完成狀態為準。開始／取消只等待
  對應的 StartedChanged 並確認讀回，不額外等待下一次 AxisChanged。依畫面軸向
  持續旋轉至韌體切軸，不假設每軸固定轉三圈；失敗會清除舊的完成軸標記。
  `started` 未知不等於完成。校正不啟動馬達，實體旋轉仍由操作員執行。
- 起飛前確認只有三步（羅盤、地圖、路線）；原第四步「串流與系統資訊」已依操作員
  2026-10-09 要求移除，影像是否出現由操作員自行目視。電量、硬體 inventory、
  影像新鮮度仍由 backend 起飛閘門檢查。連線初期尚未收到 `landed` 時失敗的
  硬體／失聯策略讀回，會在確認落地後由單一背景工作重查；
  不在空中、校正中、斷線或關閉期間重試設定。證據就緒後仍須操作員確認，
  不會自動起飛。
- KLT／IMU 的弱定位相機圖形使用該筆估計中心與方向，不借用停住的控制姿態。
  KLT 有完整方向時顯示綠色相機平面；IMU 只有航向時顯示黃色 heading-only
  箭頭，不捏造俯仰／滾轉。缺少方向或資料過期會明確標示。這只是顯示，
  不提升定位可信度、不更換飛控姿態、不解除 AUTO 門檻。
- 每次開始的 AUTO 結束後，地圖下緣保留上一趟的平均、RMS、最大偏差與樣本數，
  單位是地圖單位；完整結果及 P95、水平偏差寫入永久紀錄。使用本趟不可變的
  規劃折線，計算新鮮、有限、獨立觀測位置到最近 3D 航段的距離。
  排除到達第一航點前的靠攏、重複／過期位置、預測／holdover 與暫停資料。
  沒有合格樣本會顯示無法計算，不填零；每趟只有一筆終結摘要。
  這是定位估計相對規劃的誤差，不是獨立量測的實際飛行真值；原始地圖與航線仍用
  map units，不因顯示公尺換算而改寫。`localization_metrics.py` 的 `SessionMapScale`
  在定位開啟期間**預設自動**累積比例證據，手動與 AUTO 都適用；正常以
  `./一鍵啟動.sh` 開啟介面即可，不需要 IMU 測試腳本、額外環境變數或 AUTO 開關。
  只使用唯讀、帶來源時間戳的韌體融合姿態／NED 速度與 strong、帶來源時間戳的
  定位姿態，不是原始／純 IMU 積分。啟用估計不等於尺度已收斂，也不會自動起飛。
  UI 是唯一姿態取得者；`LoopHooks.control_state` 與
  `DesktopRouteAutonomy.control_state` 可注入同一個 `ControlStateEstimator`，
  相關讀寫以分組 `RLock` transaction 保持一致。同地圖的控制器 reset 保留累積比例
  證據，換地圖／新 session 才清除。
  全場景只用一個 `m/map unit` 比例；`metric_xyz` 是相對地圖原點的公尺座標，
  不是 GPS 座標或高度計讀數。HUD 區分 candidate 與 reliable，候選值不是可信公尺尺度。
  比例累積所有來源合格的獨立區間，不是最近 12 段的 rolling window；凍結門檻為
  至少 8 段、累積至少 10 m，每段至少 1 s／1 m／0.02 map units、最長 4.5 s，
  方向差不超過 20°、比例 spread／drift 各不超過 20%，新鮮度為 30 s。
  weak／bridge／reseed／缺失或超過 0.5 s 的視覺間隔切斷待校準區間，
  不跨品質洞教比例；保留先前合格證據，但過期或被拒絕的 trust 不得驅動預測。
  端點等到曝光時刻的姿態可插值後才校準，保存當時 map／NED offset，
  所有中間端點也須通過方向一致性。來源合格但比例矛盾的區間仍納入
  spread／drift，不能只刪掉不符合舊候選的比例來維持可靠標記。
  無效姿態／速度來源也切斷插值、位移與預測連續性，不以兩側有效樣本填平品質洞。
  待曝光配對端點按 FIFO 處理；不能跳過無法包夾的舊端點而連成新的校準區間。
  更長距離可能穩定估計，但有偏或不一致的幾何不保證收斂；即使曾短暫 reliable，
  後續方向不一致也會撤回信任。錄影重播沒有獨立公尺真值，不能把候選比例當成
  已驗證的實際尺度。
- 永久 `trajectory.jsonl` 保留路線／座標系／SHA／AUTO run ID、逐拍原始觀測與
  控制位置、控制指令、來源時間、KLT／IMU／相機方向資料、融合遙測與終結摘要。
  `pcmd_dispatch` 記錄實際 SDK 呼叫的開始／返回時間、向量設定時間、reason、
  run ID 與 PCMD；這是 SDK enqueue 證據，不是韌體執行或馬達反應確認。
  地速／姿態是韌體融合量，不是原始加速度計／陀螺儀，也沒有感測器 covariance。
  指令／incident 每筆同步，bulk stream 每 16 筆同步並在關閉時完成 fsync；
  突然斷電仍可能損失尚未同步的 bulk 尾端，不能承諾零遺失。
  佇列溢位、寫入或最終同步失敗會留下 incident／summary health，
  關閉時仍未停止的 AUTO worker 另記 `auto_worker_unresolved_at_close`，
  明示可能缺少該趟終結摘要。分析前檢查 `session_summary.json` 的
  `recording_complete`、`write_failures`、`debug_deferred_dropped` 與 `unresolved`。
- 延遲優化限於已確認的熱路徑：不變 HUD 文字不重送 Tcl、完整唯讀 RGB bytes
  直接供 optional detector 使用；可變或局部 buffer 仍複製。沒有 localizer 時
  不以 5 ms 空輪詢，有 worker 時仍保留 5 ms／filehandler。沒有縮短安全等待、
  跳過 SHA、CUDA／模型或 preflight 檢查。離線 GUI 量測工具與飛後分析入口見
  [`tools/README.md`](../../tools/README.md)；合成影片測試不代表定位／GPU 就緒。
- 地圖用一張視角／尺寸相關的點雲照片，加上有界保留式 Canvas 航線與軌跡；
  一般刷新最短 50 ms，視角、gimbal 與估計失效變化立即刷新。此節拍只限制繪圖，
  不限制定位收件、軌跡紀錄或控制迴路。影片不再與地圖互斥輪替；大幅縮小的
  顯示影像先做整數 box reduction，再 bilinear，不改定位 worker 的輸入。
  Dirty key 使用完整座標內容及投影中心／pivot，中間點原地改動也會重繪。
  最新實拍及實際 Tk 的離線配對與限制見
  [`飛前除錯手冊`](../../docs/next_flight_debug_runbook.md#最新場次-075845z四項離線優化)。


### UI 與可替換接口邊界

- `site_assets_panel.py`：只負責場域資產區的 Tk 排版、檔案選擇器與狀態文字。
- `add_map_dialog.py`：「新增地圖」視窗（資料夾瀏覽＋預檢），只提出資料夾，不複製、不雜湊、不啟用。
- `operator_ui_style.py`：全系統共用的米黃色主題（深咖啡標題列、卡片、扁平按鈕、深色地圖疊層）；
  主視窗、航線編輯器、新增地圖與深度／對位視窗都只從這裡取色與字型。
- `operator_actions.py`：按鍵規格與操作流程；不依賴 Tk，也不直接讀寫 Olympe。
- `site_asset_interfaces.py`：場域包、航線、巡檢目標三個 replaceable Protocol。
- `local_site_assets.py`：目前的本機資料夾／JSON 實作與格式驗證。
- `route_editor_model.py`：航線 schema、場域綁定與 GLOMAP ↔ Z-up 座標轉換。
- `route_editor_controller.py`：航點、復原／重做、G 與軸向鎖定；不依賴 Tk。
- `route_editor_window.py`：自製點雲編輯畫面（預設視窗化，最小 1040×640，F11 切換全螢幕，
  Esc 只取消移動、不影響全螢幕）；沒有 backend 或飛行指令接口。
- `route_editor_rendering.py`：完整點雲的 CUDA／NumPy 深度緩衝渲染；不依賴 Tk。
- `simulated_route_test.py`：把正式 route controller 的 PCMD 接到記憶體內運動模型；
  只供模擬介面測試，不連 Olympe。
- `simulated_backend.py`：離線模擬 backend（`DroneBackend`），只在記憶體內移動，
  不連 Olympe；與 `olympe_live_backend.py` 同一組方法介面。
- `flight_operator_app.py`：composition root，接上既有地圖、串流、飛控與安全清理；
  不在 UI view 內實作資產格式。

因此日後可替換本機地圖來源或航線來源，而不需要改動飛行按鍵與 Olympe backend。
建圖端五檔交付契約見 `../site_profiles/建圖端輸出規格.md`。

「場域資產」區先選地圖、再選該地圖的 AUTO 航線：「① 地圖」下拉選單列出已匯入的地圖，
選定即走既有的場域切換交易（需已落地／無 AUTO／無飛行指令處理中；失敗或取消保留原地圖）。
「② 此地圖的 AUTO 航線」是固定五列、可捲動的清單，下方常駐 `新增路線`、`編輯選取`、
`刪除選取`；沒有航線時清單內顯示「此地圖尚無可用航線」，不保留上一張地圖的路線。
未啟用的地圖、長地圖名稱與長錯誤訊息都在自己的區域內捲動／換行，不撐開控制區。
地圖切換尚未完成前，清單中的航線選取、編輯、刪除保持禁用，避免把新地圖路線交給舊 runtime。
只有核准紀錄有效的航線可綁為 AUTO 候選；沒有核准紀錄的歷史航線僅切換顯示並提示
「開啟編輯器並儲存此路線即可自動核准」。清單同步不授權 AUTO。

「＋ 新增地圖」開啟非強制回應（不鎖住滑鼠與鍵盤，飛行列仍可點）的資料夾選擇視窗。
直接點選標有 ◆ 的場域資料夾（含 `site_profile.json`），或輸入完整路徑後按 Enter。
預檢通過便自動匯入，不需另外確認；背景驗證及複製完整定位包至 `地圖檔/場域/<site_id>/`，
成功後切換到該地圖。驗證或切換失敗時保留目前地圖。
開啟、匯入都受同一個已落地／無 AUTO／無飛行指令處理中守門，不送出任何飛行指令、不核准 AUTO。


### 自製航線編輯器

「場域資產」的航線區提供 `新增路線`（已移除 `匯入航線 JSON` 按鈕）、`編輯選取` 與 `刪除選取`。
編輯器標題列顯示場域與地圖，頂端只有一個 `導入地圖資料夾`；儲存／模擬／離開固定在底部，
不因側欄捲動或長訊息而被推出視窗，側欄五個分頁（航點、物體、量測、說明、電線）在
1040×640 以上都可到達。系統遞迴偵測 `.ply`；只有一個時
自動載入，多個時列出相對路徑與檔案大小供使用者選擇。選定後計算 PLY 的
SHA-256，若唯一匹配到已匯入場域的 `asset_sha256.map_ply`，就自動綁定該場域並
開放正式航線匯入。沒有匹配或同時匹配多個場域時，只能標注及另存 preview
JSON，正式航線匯入按鈕會鎖住。

新航線分兩階段。第一階段以俯視模式標路徑點；「標路徑點模式」預設關閉，開啟後
左鍵單擊才會新增路徑點。左鍵拖曳旋轉、左鍵雙擊將最近點雲設為新地圖中心，雙擊不會
同時新增路徑點；右鍵拖曳平移、滾輪縮放。Enter 後進入第二階段，標點模式會自動停用；
選取現有路徑點後按 G，再按 X、Y 或 Z 鎖定軸向，左鍵／Enter 確認，Esc 取消。編輯器使用
Z-up 顯示座標並輸出 `frame: aligned`；匯入端再轉回 GLOMAP 的 X/Z 水平、-Y
向上座標。

「物體長方體」提供與標路徑點、量距互斥的「物體框選模式」。開啟後左鍵拖曳
粗框物體；「新增長方體（拖曳框選）」可再建立其他框。初始長方體包住矩形內
所有深度的點雲，背景點也可能納入；沒有點雲時不建立。零厚度方向會補上
地圖視角半徑的 1%，可再調整。點框線或中心選取，選取框顯示黃色。

- `G`：整個長方體隨游標平移；接 `X/Y/Z` 限制單軸，Z 正向永遠向上。
- `S`：以中心等比例縮放；接 `X/Y/Z` 只改長 X／寬 Y／高 Z。滑鼠沿起點遠離
  框中心的方向移動放大、往中心方向移動縮小，每 250 px 約變 e 倍（≈2.7×），可無限縮小；
  從框中心起手時右／上放大、左／下縮小。按住 Shift 移動為 1/10 細調，切換
  Shift 不會讓框跳動。縮放過程中滾輪仍可放大縮小地圖，不影響目前的變換量。
  也可輸入正縮放倍數，例如 `S → X → 2` 將 X 長度加倍。
- 鎖軸後可輸入有正負號的位移，單位為地圖單位，例如 `G → Z → 1` 上移 1 u，
  `G → Z → -0.5` 下移 0.5 u。重按同一軸解除限制，Backspace 修改輸入。
- `Ctrl+C` 複製選取的長方體、`Ctrl+V` 在同一位置貼上完全相同的新框並選取它
  （可直接按 `G` 移開）；貼上可用 Ctrl+Z 復原，變換進行中不可複製／貼上。
- 物體模式中雙擊點雲同樣可切換地圖旋轉中心，不會新增或選取長方體。
- 左鍵或 Enter 確認，Esc 或右鍵還原本次變換；未變換時右鍵仍平移視角。
  新增框選時 Esc 取消粗框；閒置時 Esc 離開物體模式，不刪除標註。
- Delete 或「刪除選取長方體」只刪物體框；物體模式內的 Ctrl+Z／Ctrl+Shift+Z
  只復原／重做長方體，不影響航點。文字輸入欄不觸發地圖快捷鍵。

中心及長寬高顯示於側欄，均使用地圖單位 u。儲存路線或預覽 JSON 時，標註保存為
可選 `object_boxes: [{"center": [x,y,z], "size": [長X,寬Y,高Z]}]`；座標固定為
編輯器的 aligned Z-up，跟隨文件的 `align_source`，不是外層 `frame: glomap` 的
航點座標。重新編輯正式路線會載入標註，換地圖會清空舊框。legacy 框轉到 measured
座標時以八個角點重算包圍長方體，範圍可能擴大，並非保留旋轉。未指定航段朝向
時只作物體標註；指定後提供機頭朝向目標，不是避障或飛行安全區，亦不修改航點
與起降控制。只畫框仍不能取代至少兩個航點的既有路線儲存條件。

「電線」分頁可補畫重建時遺漏的電線，與標路徑點、物體框選、量距模式互斥：

- 開啟「電線繪製模式」，依序單擊點雲或既有航點選 A、B，建立直線並選取。
  要畫下一條時按「新增電線（兩點）」。端點必須不同；附近沒有可吸附點時不新增。
- 每條電線依目前清單順序自動分配不同色相，不使用短循環配色；端點及線號沿用線色。
  選取時保留原色並加白色外框。相同排序重開配色一致，刪線後後續線色依新順序重排。
- 主系統地圖也顯示當次載入／選取航線的電線，沿用相同線色及 `W1／W2…` 線號；
  航點路線預設隱藏時仍可看見電線。編輯器成功儲存回主視窗後立即更新，
  切換航線同步替換電線，沒有電線或刪除所顯示航線時清除舊線；換地圖不沿用其他場域的線。
  曲線依原本的 legacy／measured 對齊座標轉回地圖，隨主地圖旋轉、平移與縮放。
  圖層只從符合當次航線 SHA-256 及場域／地圖身份的檔案載入；不符時不顯示並寫入紀錄。
- 單擊電線端點 A／B 選取，選取端點顯示白圈，側欄顯示其 X／Y／Z 座標。
  按 `G` 平移，再按 `X／Y／Z` 鎖定編輯器單軸（Z 正向為上）；
  重按同一軸解除限制。可移動滑鼠，或鎖軸後輸入正負位移，例如
  `G → Z → 0.1 → Enter` 上移 0.1 u、`G → X → -0.05 → Enter` 沿 X 負向微調。
  只改所選端點，另一端、既有垂度及其他電線不變；曲線即時隨端點更新。
  點線身只選電線，不會讓 `G` 誤移整條線。
- 點線選取後按 `S`，滑鼠下移增加垂度、上移減少，最小為零（直線）。
  也可直接輸入絕對垂度，例如 `S → 0.5 → Enter`，單位為地圖單位 u；
  垂度是跨中相對兩端直線的下垂量，不是最低點的絕對高度。
- 左鍵／Enter 確認；Esc／右鍵還原本次端點移動或垂度調整。
  新增中 Esc／右鍵取消未完成端點；閒置時 Esc 離開模式，保留已完成電線。
  輸入未完成、數字位移未選軸、負垂度或兩端重合時不能確認。
- Delete／「刪除選取電線」刪線，Ctrl+Z／Ctrl+Shift+Z 只復原／重做電線，
  不修改航點；左鍵拖曳旋轉、雙擊設中心、右鍵平移、滾輪縮放沿用既有視角操作。
- 移動端點、調垂度或只選了第一端時，必須先確認／取消才能儲存、換模式或換地圖。

電線使用拋物線視覺標註，不是懸鏈線力學模擬，不推估真實電線位置或安全淨空。
儲存路線／預覽 JSON 時保留可選 `wires` 清單，每條包含 aligned 座標 `a`、`b`、
非負 `sag` 及單位向上軸 `up`。預設沿編輯器 Z 軸下垂；legacy 轉 measured 時
端點及 `up` 一起轉換，保留原有曲線。重新編輯正式路線會載入，換圖會清空。
比照物體框，仍需至少兩個航點才能儲存；不是全場域共用圖層，不會寫回原始 PLY，
也不啟用自動避障或改變飛控。手畫電線不可作為實際淨空或飛行安全保證。

「航段機頭朝向物體」可選起點、終點及長方體，再按「套用所選區間朝向」。
例如航點 2 → 4 包含 2→3 與 3→4；可對多個不連續區間逐一設定，不同區間也可
選不同物體。重複套用只覆蓋所選航段；「清除所選區間朝向」恢復該區間原本航向。
設定航段顯示青色與 `B1` 等物體編號，側欄列出目前區間；未設定航段保持原本控制。

每次控制更新依目前控制姿態估計與長方體中心計算水平 yaw，指定航段的進入轉向
改為先朝向物體，再沿原路線平移；移動、回到航段及近點置中時持續修正朝向，
並在既有 PCMD 上限內提高移動目標的轉向響應。僅去程適用，返程維持一般航向；
到點轉向／懸停／鏡頭動作與巡檢仍優先，起點靠攏、終點降落置中、搖桿接管及
既有姿態／串流／速度守門不變。只控制機頭水平朝向，不自動調整鏡頭俯仰；
水平位置與物體中心重合時沒有唯一方向，保留最近有效朝向，不突然旋轉。

JSON 可選 `segment_look_at` 長度為航點數減一，每項為 `object_boxes` 的零起算
索引或 `null`（原本航向）。例如 `[null, 0, 0]` 對應 2→3、3→4 朝向第一個框。
存檔／重開及 SHA 綁定的 mission snapshot 均保留設定；移動長方體後，目標跟隨
新中心，需儲存後才用於新任務。正在被航段引用的框必須先清除引用才可刪除；
刪除其他框不會把朝向誤指到別的物體。復原跨越物體建立／刪除時，若引用的框
已不存在，明確清除該朝向並提示，不自動指向另一框。刪除區間中間航點時，
新接起航段僅在原本兩側朝向相同時延續，否則恢復原本航向，可再指定。

本功能已做離線桌面存檔／重開與正式閉迴路模擬，並非真機朝向精度保證；
物體中心與航線仍由操作員確認，真機起飛／AUTO 只能由操作員親自執行。

第二階段會顯示可拖曳調整的半透明綠色到達球；拖曳滑桿會自動開啟對應圖形並即時縮放。
航線 JSON 以 `arrive_radius_map_units` 保存到達半徑。每一拍都從目前定位位置重畫往
目前航點的直線，偏航由機身平移持續修正，沒有安全管／重返執法。

右側「航點動作」可替選取航點加入依序執行的動作：左轉／右轉 N 度（1–180，
相對到點時的機頭方向）、懸停 N 秒（最多 60）、鏡頭俯仰 -90～+90°。
鏡頭採相對地平線的**絕對角度**，負向下、正向上、0 水平，不是機體俯仰。
每點最多 8 個動作（去程、回程各自計算），可復原／重做；畫面以 `R30`、`L45`、`H3s`、`C-45`
標在航點編號旁，回程動作以 `ret[...]` 標示。動作面板的「去程／回程」選項決定新增與
「刪除最後一個動作」作用在哪一趟。航線 JSON 以 `waypoint_actions`（去程，每個航點一個清單，
`{"type": "yaw", "degrees": 30}` 正值為右轉、
`{"type": "hover", "seconds": 3}`、
`{"type": "gimbal_pitch", "pitch_deg": -45}`）與 `return_waypoint_actions`（回程，同格式、
同樣每個航點一個清單，舊航線沒有此欄位即無回程動作）保存，儲存與 AUTO 載入都用
`route_domain` 驗證。去程動作在去程到點後執行；回程動作在返程飛回該航點時執行，
維持在航點位置、轉向時鎖高度，完成後才轉向下一點。第一個航點是降落點、最後一個航點
是只飛一次的折返點，因此兩者不能設定回程動作（折返點的動作屬於去程）；儲存與載入都會
拒絕。轉向超過既有對頭時限未收斂會略過該點其餘動作並寫入操作紀錄。
鏡頭每步只送一次既有 camera command；需兩個不同、送出後的新鮮 gimbal 回讀
（≤1 秒、誤差≤2°）才確認。10 秒有效控制等待逾時或命令／回讀失敗時，記錄
未完成並略過該點其餘動作。暫停不重送、不累計等待；同一航線人工接管續飛
只繼承已確認的鏡頭步驟，且仍須重新到站，落地後的新任務重新執行。
鏡頭保持最後目標，直到下一個相機動作或人工調整，不自動復位。
最後降落點仍不可附加動作，起降與手動接管守門不變。

2026-10-06 離線桌面驗證已實際保存／重開鏡頭動作，並經正式 simulated backend
回讀執行 -45° → 0° → +30°；場域清單已驗證待套用、無航線、套用與 rollback。
定位 deploy、飛控與操作介面回歸共 2651 項通過，1 項因場域 alignment 缺失跳過。
未連硬體、未起飛。實際畫面：
[航點鏡頭編輯器](../../outputs/analysis/operator_workflow_20261006/route_editor_camera_pitch.png)、
[所選場域航線清單](../../outputs/analysis/operator_workflow_20261006/selected_site_routes_pending.png)。

若場域地圖旁有 `map/map_dense.ply`（用
`tools/build_dense_display_ply.py --release <release>` 由完整稀疏模型產生；或加
`--mvs-fused <fused.ply>` 改用 MVS 彩色融合點雲，河濱目前是這個，3.00M 點，
舊稀疏版不留備份，不加 `--mvs-fused` 並以 `--out <path>` 即可重建；福和橋同樣是
MVS 版，2.20M 點；烏來 all9 是 MVSFormer++ 稠密化點雲，由 1.048 億點串流抽樣為
3.00M 點，與定位模型同一座標系，另須通過 `tools/verify_dense_alignment.py` 的關鍵影格深度驗證），且其標頭的
座標系與 `map.ply` SHA-256 都符合目前場域，編輯器改顯示這份較密的點雲
（原 `map.ply` 為 1/17 取樣）；不符時沿用 `map.ply`。旋轉、平移與縮放每幀均使用
全部已載入的點，不再互動抽樣。背景單一 worker 只保留最新視角，不累積滑鼠事件的
繪圖工作；CUDA 可用時將座標與顏色常駐 GPU，以逐像素深度緩衝取代全點雲排序。
無 CUDA 或 GPU 資源不足時改用完整點雲的 NumPy 渲染，介面事件仍不等待重算。
航點點選使用實際已顯示的視角，避免背景尚未追上最新旋轉時選錯位置；點選查詢在
render worker 內對已常駐 GPU 的點雲執行（3.00M 點約 4 ms，原本在 Tk 執行緒用 NumPy
投影全部點約 95–140 ms）。單擊新增航點仍保留 280 ms 雙擊判斷等待。
這份檔案只供顯示與點選航點，定位不讀取。

2026-10-03 離線桌面驗證：河濱 PLY 的 **2,437,743** 點均保留。RTX 5060 Laptop、
1200×800 同視角，舊完整繪製中位數 426 ms、舊互動抽樣 54 ms，新完整 CUDA
繪製 5.1 ms。實際 1620×1124 編輯畫面旋轉的顯示間隔中位數 22.8 ms；
數值依 GPU 負載、視窗尺寸與點雲而變，CPU 模式不保證相同速度。

真機模式只有明確回讀為 `landed` 且沒有飛行命令處理中才可開啟；狀態改變時
編輯器會自動關閉。「儲存路線」會驗證場域、座標系與航點，並以編輯器開啟時的地圖
`map_ply_sha256` 為該次儲存蓋章；只有儲存成功且取得可驗證的核准結果才算完成，
失敗或被拒絕不會把航線標為已儲存。成功時原子更新 managed profile 的路線 SHA-256 與
`flight.authored_routes` 核准紀錄（`path` 相對路徑、`sha256` 路線雜湊、`map_sha256`
地圖雜湊，並綁定座標系），其他航線的紀錄保留，編輯會取代該航線的紀錄，刪除會移除。
選取航線時以「地圖未變、路線 SHA 相符、座標系相符」驗證該紀錄；任一不符、遭竄改或外部匯入
（`匯入正式 JSON` 不再自動核准）都維持鎖定。介面內成功儲存的有效已綁定航線即視為核准，
不需另外申請，可直接供自動飛行選用；舊版以 `flight.approved` 與編輯器核准註記留下、
且路線 SHA 吻合的現行航線自動沿用，其他沒有可信紀錄的歷史航線只需在編輯器開啟並成功儲存一次。
若目前使用 system profile，會在編輯器關閉後套用更新過的 managed profile。
選定航線後，AUTO 準備度＝該 mission 的非航線條件（定位、校正、相機）全部通過，加上
所選航線的核准／快照一致；非航線證據未知時一律鎖定，不從 `False` 推導成 `True`，
評量模式（`SFM_EVALUATION_ONLY`）永遠不授權 AUTO。這不會立即起飛；真機仍須通過三項驗證與既有任務、
定位及後端檢查，由操作員親自按「自動飛行」。AUTO 請求送出後，航線選擇鎖定；HOVER、MANUAL 與重新
定位不會解除，只有後端拒絕該次請求或確認降落完成後才可選下一條航線。

「新增路線」保留既有航線；每條航線右上角的「編輯」只開啟並覆蓋點選的那條，
不會先更換 AUTO 選取。儲存結果成為目前 AUTO 航線並同步 profile、route component
與 selection SHA，不建立草稿。`×` 經確認後刪除該航線，連帶清除引用它的 profile、
component 與 selection；刪除目前綁定者會清空畫面路線、AUTO 綁定及接續紀錄，
不自動選替代航線。真機須落地、沒有飛行命令／AUTO 處理中，且編輯器未開啟；
資產作業處理中也不能編輯或刪除。未選取的舊航線可單獨刪除，其他綁定不受影響。
只有未綁定場域的 PLY 可另存 `preview_only` 預覽航線，該檔不能進入 AUTO。

模擬介面另顯示「儲存並開始模擬航線」。操作員按下後會先完成相同的原子儲存、
SHA 驗證與航線綁定，再用 production `RouteAutoController` 和記憶體內運動模型執行
閉迴路測試；到達最後一點並通過地速確認後結束。這條流程不會連接真機或送出
Olympe 指令；真機介面不顯示此按鈕，仍須完成三步 preflight 並親自按「自動飛行」。

新的 AUTO 任務以連續穩定的定位姿態判定起飛位置，不要求位於第一個航點附近；先飛往第 1 航點，再依序往第 2、3 點移動。同一趟飛行、同一路線在搖桿接管後再次按 AUTO，會重新交接控制權與確認定位，接續尚未完成的航點，包含返程進度；落地、更換路線或更換 backend 後清除接續紀錄。開放航線不循環，最後回到返程起點降落。起飛位置仍須在任一航點的
`SFM_BOOT_START_MAX_U` 範圍內（預設 `1.5` map units），避免從航線外直接橫切進入。

AUTO 尊重路線設定的全域或逐點到站半徑，不把已設定的小半徑放大到 0.15 map units。
高度誤差仍取自定位姿態與航點的三維差；高度計不參與這個位置控制迴路。

## Single-Process Desktop App

### REAL-FLIGHT interface (Olympe via SkyController)

**`--interface real-flight` connects the controls below to a real Olympe backend
and accepts only ANAFI PDRAW video. It never falls back to a video file. The old
`--live` flag remains only as a compatibility alias. Only a human UI click on
`起飛` or `自動飛行` may initiate takeoff.**

介面進入後會顯示一條不會遮住降落／緊急控制的「起飛前依序確認」導引，依序檢查：
① 飛機羅盤校正狀態（連線後自動沿用機上可用校正）；② 目前匯入的地圖與場域；
③ 顯示後逐點檢查的航線（包含匯入／修改結果）。可用證據會依序自動確認，操作員仍須
檢查現場條件。2026-10-09 操作員要求拿掉原第 ④ 步「串流與系統資訊」，改由操作員
自行目視影像；它原本預先顯示的項目不再當作起飛前步驟，但 backend 起飛閘門仍各自強制
檢查：電量低於門檻或讀不到（同一份 `takeoff_battery_blocker`）、硬體 inventory 未就緒、
有影像串流時影像超過 1 秒未更新，都會拒絕起飛；高度／距離限制未設定時確實不限制，但
**已設定**者若寫入或回讀失敗也會拒絕起飛；GPS 只有在已設定距離圍欄且要求 GPS 時才阻擋。
前一步未確認時不能跳到後一步；任何已確認的狀態或資產後來改變，該步驟與其後步驟會失效。
全部完成後，介面才顯示可以由操作員按「起飛」並啟用按鈕；這不會自動起飛，按下後後端仍會
重新執行完整 fail-closed preflight。

```bash
cd .../localization/控制介面程式/operator_interface
# stop any other Olympe session first (single connection)
python3 flight_operator_app.py --interface real-flight \
  --ip 192.168.53.1 --controller skycontroller3 \
  --max-altitude-m <METERS> --max-distance-m <METERS> \
  --distance-geofence \
  --no-live-detect
```

`MaxAltitude` and `MaxDistance` are adjustable firmware settings. The backend
still attempts to configure and read them back, but missing values, rejected
writes, and readback mismatches are advisory and do not block takeoff. Equivalent environment variables are `SFM_MAX_ALTITUDE_M` and
`SFM_MAX_DISTANCE_M`; do not choose flight limits by copying the simulated HUD
numbers.

`--distance-geofence` sends `NoFlyOverMaxDistance(1)` and is **off by default**.
It does not make GPS fix a takeoff requirement. Turn it on with the flag above, or with
`SFM_DISTANCE_GEOFENCE=1`, which is the way to enable it through
`start_anafi_live.sh` — that launcher deliberately does not pin the flag, so the
environment variable governs. The firmware flag prevents outward piloting beyond the configured radius and
does **not** itself start Return-To-Home. The host safety monitor separately
requests RTH at 95% of the confirmed readback limit when Home is reachable,
otherwise it requests in-place Landing. RTH depends on valid GPS/home state. The takeoff
preflight uses a 20% battery floor (operator decision 2026-10-11, was 30%) and rejects
any configured floor below 20%. AUTO keeps its separate 10% in-flight battery reserve.
When GPS is unavailable, distance geofence may be disabled or kept advisory-only
according to the backend/firmware policy; that state is shown and logged. Missing
GPS is an advisory and does not block manual or autonomous takeoff.

All safety-relevant CLI/environment values pass through `live_safety_config.py`
before the UI or backend can use them. The effective normalized values and their
SHA-256 are printed at startup. Overrides cannot exceed the measured operator
envelope (20° tilt, 2 m/s vertical, 20°/s yaw, 10 s stream-loss grace); altitude
and distance must be finite, positive, supplied together, and are still confirmed
against firmware bounds and readback.

The launcher keeps the UI and localization interpreters explicit:

```bash
SFM_UI_PYTHON=/path/to/olympe-python \
SFM_LOCALIZER_PYTHON=/path/to/torch-python \
IP=192.168.53.1 CTRL=skycontroller3 ./start_anafi_live.sh
```

The standard launcher starts with `MaxAltitude=50 m`, `MaxDistance=100 m`, and
the distance geofence enabled when GPS/home support is available. If GPS is not
available, the backend may disable the distance geofence or keep it advisory-only,
with the state shown and logged. Override these conservative startup values with
`SFM_MAX_ALTITUDE_M` and `SFM_MAX_DISTANCE_M`, or edit and apply them from the
ANAFI panel while the aircraft is confirmed landed. The panel always shows the
aircraft readback separately; applying limits while airborne is rejected.
Crossing either firmware limit prevents continued flight outward; it does not
automatically invoke RTH. AUTO translation is capped by the AUTO PCMD-strength
limit, and a ground-speed limit shrinks horizontal PCMD that adds speed along
the current motion. It acts on speed extrapolated 0.25 s past the telemetry
sample's age at the measured acceleration, tapering quadratically from the full
cap at standstill to zero at `autonomous_speed_limit_mps` (site profile
`speed_limit_mps`). It never latches
or sends a zero-PCMD hover, and braking commands pass untouched. The fail-closed
interlock and yaw-only turn brake are gone: missing or stale ground speed leaves
PCMD unlimited. This is not a physical
hard-speed guarantee; PCMD response, telemetry latency, wind, and braking
distance still require field validation. Route completion still waits for
fresh ground speed at or below 0.10 m/s before requesting Landing.

On connection the backend records the actual ANAFI model/serial/firmware,
SkyController 3 model/serial/software, Olympe version, transport, Home Point,
lost-link/RTH policy, and firmware limit readbacks. Ground diagnostics remain
available when these are incomplete. Firmware/Olympe versions and an optional
signed hardware receipt are diagnostic/audit records; they are not takeoff or
AUTO readiness gates. The selected profile, immutable route snapshot, asset
digests, three preflight steps, and post-takeoff localization gates remain
enforced.

UI Python resolution is `SFM_UI_PYTHON` > legacy `VENV` executable > the
nearest package/workspace `.venv/bin/python` (up to two parent levels) >
`python3` on `PATH`. The launcher validates an inherited display, then probes
active X/XWayland sockets when `DISPLAY` changed after reboot; no usable desktop
fails clearly. The default target is always SkyController 3 at `192.168.53.1`;
the launcher does not probe target reachability or automatically switch to direct
drone Wi-Fi. Supplying only a known ANAFI IP or controller derives its safe
counterpart; inconsistent known pairs fail before launch.

The real-flight UI opens disconnected, even when the target is offline. Press
**「真機串流」** in the UI to start the background connection. If it fails,
press the same button again to retry without restarting the terminal. Opening
the UI or connecting does not request takeoff; flight controls remain operator-only.
The backend keeps a single connection-owner worker alive for the command and
video loops, including video retries: Olympe stops its loops when their creating
thread exits. The owner is released only after the existing cleanup confirms
touchdown and disconnects; an unconfirmed landing retains it for cleanup retry.
For parser/command QA, `SFM_LAUNCH_DRY_RUN=1` prints the command without network
operations or starting the UI.

The launcher defaults to `SFM_MAX_PERFORMANCE=1`: GameMode, the performance
power profile, disabled automatic low-battery power saver, and the verified
four-thread sustained CPU budget. The profile and GNOME setting are restored
when the UI exits; no service, clock, power limit, or thermal protection is
changed permanently. This applies on AC and battery, but a failed/unavailable
setting is reported rather than bypassed with `sudo`. Use
`SFM_MAX_PERFORMANCE=0` to skip all tuning, or `SFM_CPU_THREADS=N` only for a
separately validated sustained-load experiment.

For a read-only thermal/throttle trace during a ground or replay benchmark:

```bash
python3 ../../validation/monitor_hardware.py \
  --output ../../outputs/hardware.jsonl --interval 1
```

The monitor logs Linux CPU temperature/frequency/throttle counters and NVIDIA
temperature/clock/power/slowdown reasons. It emits rate-limited warnings when
CPU throttle counters increase or GPU thermal slowdown is active; it never
changes hardware state. Use `--warning-interval 0` only when stderr warnings
would interfere with an automated capture.

This wires the UI buttons to real Olympe:

| UI | Real action |
|---|---|
| 起飛 | TakeOff → hover；不檢查定位；起飛成功後固定開始機載錄影 |
| 自動飛行 | TakeOff → 原地零 PCMD 懸停 → 連續可靠定位後先飛往第 1 航點再依序執行已鎖定路線；定位失敗時持續懸停，等待恢復或操作者手動接管／降落；暫停後按「繼續自動飛行」沿原路線恢復 |
| 原地降落 | 停止錄影並盡力下載 → Landing + restore sticks |
| 關窗 / Ctrl+C | **強制原地降落**（即使曾 Esc；已落地則跳過） |
| 錄影 | 介面啟動時固定武裝；起飛成功後自動錄機載 SD；降落時存檔。下拉選 FHD 1080p30 或 4K UHD 30（只改 SD 編碼，live stream 仍是 720p） |
| 懸停 / Space | zero PCMD；AUTO 執行中改為暫停 AUTO 並保留原路線狀態 |
| 手動 / Esc | zero PCMD + `setPilotingSource(SkyController)` |
| 動搖桿（SC USB） | HID 偵測偏轉 → 強制交回搖桿（即使當下是 PC 控機） |
| 微移 8 方向 | hold-to-move PCMD; release returns to hover |
| 俯仰滑桿 | gimbal set_target |
| 開始／取消定位 | 第一次按下啟動 localization feed，按鈕變成「取消定位」；再按一次停止送入影格並忽略待處理結果。全程不取回 PC 控制、不起飛、不執行航線 |
| 恢復電腦控制 | 明確切換 piloting source 至 PC；仍不會啟動自主航線 |

2026-10-04 關閉流程修正：

- Ctrl+C／SIGTERM／SIGHUP 的訊號處理器只記錄退出請求，不在被打斷的主執行緒
  取得控制、記錄或清理鎖；Tk 每 50 ms 接收請求，沿用關窗的背景清理流程。
- 背景清理以佇列回報結果，只有 Tk 主執行緒排程與銷毀視窗。清理期間的重複
  關窗／訊號會合併，不會啟動第二份清理或在失敗後自行重試。
- 後端清理成功後，才關閉定位／偵測 worker 與模擬影片串流，再關閉工作階段記錄
  和視窗；不再因 `_shutdown_completed` 跳過 worker 清理而留下子程序。
- 真機仍必須確認落地才能退出；未確認時保留介面、連線與 worker，讓操作員處理後
  再次關窗或按 Ctrl+C。不加入強制退出捷徑，不弱化 PCMD 歸零、強制降落與搖桿交還。

Each run writes an immutable manifest plus command, localization, telemetry,
incident, and summary streams under `outputs/flight_logs/session_*`. Logging or
critical disk failure zeros PCMD and hands control to manual without auto-resume.

### Latest-frame and latency pipeline

The live path is deliberately lossy: Pdraw's raw callback only records
`time.monotonic_ns()`, calls `ref()`, replaces one pending YUV slot, releases the
replaced frame, and returns. `VideoFrame.info()`, `as_ndarray()`, YUV-to-RGB,
resize, localization, detection, display, and logging never run in that
callback. Pdraw keeps two decoder buffers for decoder progress, while the
application has exactly one pending frame; those are different layers.

The UI copies the newest contiguous RGB array into the inactive slot of a
two-slot shared-memory buffer; the localization worker receives only a one-byte
slot index and reads that frame zero-copy. Completed results wake Tk through a
file descriptor, with a low-rate poll retained as fallback. Optional YOLO keeps
the compatible pipe/fixed-`bytearray` path. TRACK runs as fast as the worker
becomes free, and busy periods retain only the newest frame.

`loc_metrics_*.jsonl` records frame receipt, preprocessing, IPC, worker,
localization, response, and UI timestamps in host monotonic nanoseconds, plus
derived `frame_receipt_to_*_ms` values and mapped receipt age. The HUD shows core
latency, receipt-to-localization/UI latency, stream backlog age, and the
PCMD-to-next-telemetry-poll interval. The latter is a host readback marker, not
proof that aircraft motion had already changed.

**2026-08-03, operator decision:** the fixed summary bar above the map and the
permanent top SIM/REAL identity strip were removed as duplicates. Those rates now
appear in the **定位資訊** tab, and SIM/REAL identity only in the video HUD, so
neither is visible while another tab is selected or when there is no frame to
draw. `format_pipeline_metrics_summary()` and
`OperatorApp.pipeline_metrics_summary()` are kept (test-only now) so the bar can
be reattached to `_build_ui`. The counting semantics below are unchanged and still
describe what the定位儀表 numbers mean: localization results arriving at the UI per
second (5 s rolling), current stream FPS (5 s rolling), current submit-to-UI
latency, its recent 5 s p95, and frame age. In
SIM this is explicitly labelled as real-link simulation: frames first pass the
720p30/H.264 Main/5 Mb/s encoder-decoder, fixed 280 ms frame backlog, and
optional slice loss, then EDM, and are counted only when results reach Tk. REAL
counts the corresponding PDRAW-to-EDM-to-UI results. Missing or expired
measurements are `N/A`; core model throughput is kept separate and is never
presented as end-to-end FPS.

串流底部 HUD 顯示飛控融合姿態、相對起飛點高度、AGL、NED 速度、GPS／RTH
與連接品質；這些不依賴視覺定位。2026-10-03 新增三組唯讀裝置資訊，每秒讀取
一次 Olympe 的本機事件 cache，不發送相機設定、雲台校正或飛行指令：

- 電池：電壓、健康度、循環次數、詳細警告，以及 SkyController 電量／充電狀態。
  電壓原始單位為 mV，HUD 轉為 V；遙控器電量 255 代表充電中，不顯示為 255%。
- 相機：有效快門／ISO、曝光模式、白平衡、相機模式、機載錄影狀態、
  錄影解析度／幀率／設定碼率及無損變焦上限。錄影設定碼率不是串流實測吞吐量；
  非 custom 白平衡不顯示無效的色溫，碼率 0 顯示未知。
- 雲台：既有絕對／相對 R/P/Y，加上校正狀態、穩定狀態、鎖軸與錯誤。
  校正狀態只讀回，不啟動校正；相機與雲台只選 ID 0 的資料。

新資訊依實際可用寬高，以量測後的欄寬與字體調整 HUD，並保留定位警示的間距。
HUD 資訊背景完全透明，不再用黑色底板遮住串流；一般尺寸採 18 px 資訊字、
20 px 影像狀態字，白字加 1 px 黑色陰影，亮色畫面仍可辨識。
短面板使用緊湊排版；高度在物理上無法容納全部資訊時，優先保留來源／新鮮度與
比例信心，次要裝置列可省略，並非承諾任意視窗大小皆顯示全部列。
未回傳或不可讀的事件顯示「未知」，不視為正常；失聯清除新增 cache。
`telemetry.jsonl` 的 `readback` 每秒記錄
`battery_info`、`camera_info`、`gimbal_info`，保留 SDK 原始單位及枚舉名稱。
`device_info_read_mono_ns` 是本機 cache 讀取時間，不代表感測器的新樣本時間。
原版 ANAFI 不承諾 ANAFI Ai 專用的電池溫度、單電芯電壓或 mAh 容量；
這次也沒有接入影格 `vmeta()` 或原始 IMU／氣壓。正在執行的 UI 不會熱載入，
需操作員在落地後重啟介面並重新連接，才會讀取新增欄位。

Production does not force a CUDA synchronization for timing. Set
`SFM_LOC_PROFILE_GPU=1` only for a profiling run; it enables CUDA Events and the
required event synchronization, so its FPS is not a production FPS result.

### Render and layout budget

Everything below runs on the Tk main thread, which is also the thread that hands
frames to the localizer, so per-tick waste there is not free.

- The always-visible header identifies the active aircraft model as REAL/SIM,
  separately from link, control owner, flight state, battery, localization and GPS
  status chips. Link or flight-state changes do not change the aircraft identity.
  GPS NO FIX explicitly says that manual takeoff remains available and AUTO will hover first.
  Per-result localization updates show only health in the header; inliers stay in
  the video HUD. Numeric inliers and success/failure counters must not expand the
  header chip and shift or resize the map/video when confidence changes.
- Flight actions (手動/搖桿, 恢復電腦控制, 懸停, 原地降落) stay **above** the
  shared 操作工作區. 飛行、校正、場域資產 appear together on that page, so
  preflight navigation never hides another section. Camera and recording controls
  use their own compact row. Warm ivory surfaces, fields, buttons and virtual sticks
  share the same palette with enlarged interface text; flight actions retain their distinct safety colours.
  Workspace sizing uses the notebook's measured content height, bounded by the
  map/video minimum, rather than only a window-height percentage. The route list
  yields height before the asset selectors, action buttons and status footer;
  calibration and flight-setting overflow remains locally scrollable.
  The map/video sash defaults to 35/65 and remains operator-adjustable; the
  supported client-area constants above are not a guarantee that every HUD row
  fits at every operator-selected pane height.
- Localization alerts (LOCALIZATION LOST, LOW CONFIDENCE, LOST hold) live only in
  the video panel (`render_video`) and in `loc_health_label`. **2026-08-03 operator
  decision:** the top-level banner was removed as duplicated by the middle of the
  interface; the in-video one is clipped to that panel and gated by the video dirty
  key. `incident_banner` (CONTROL LINK LOST, stream loss, disk pressure) is packed
  only while an incident is active — there is no idle 「安全狀態：正常」 row.
- `VideoCanvasPresenter` retains the immutable fitted video source and its
  `ImageTk.PhotoImage`, together with native canvas overlay items. A new frame or
  changed fit may resize/upload the video image; a HUD-only update does neither
  and updates retained overlays instead of compositing a whole PIL frame.
  PhotoImage and canvas work stays on the Tk thread.
- The point-cloud base is a depth argsort plus a scatter over the whole cloud:
  ~16 ms at 250k points. While the operator drags or zooms it drops to
  `SFM_MAP_INTERACTIVE_POINTS` (default 60k, ~4.9 ms) and settles back to
  `SFM_MAP_STATIC_POINTS` (default 250k) once the view stops moving. The
  decimation level is part of the base-image cache key.
- 場域旁有綁定目前座標系與 `map.ply` SHA-256 的 `map/map_dense.ply` 時（編輯器同一套檢查），
  主地圖改顯示這份稠密點雲，載入上限 `SFM_MAP_DENSE_POINTS`（預設 1M，經參考位姿
  範圍濾除後約 0.7M，烏來 all9）。點雲光柵在背景執行緒，只在操作員旋轉／縮放時
  重畫，靜止時不耗每拍成本；烏來實測 140k 點 15 ms → 705k 點 66 ms，Tk 執行緒只多
  一次約 5 ms 的陣列複製。拖曳仍降到 `SFM_MAP_INTERACTIVE_POINTS`。不符或沒有稠密檔時
  維持原 `map.ply`。僅供顯示，定位不讀取。
- The planned-route polyline still uses every point; only its dots are thinned to
  `SFM_ROUTE_DOT_MAX` (default 200). At a 2000-point route that is 9.5 ms → 1.1 ms.
- All HUD `StringVar`s are `DedupStringVar`: Tk repaints a label on every `set()`
  even when the text is unchanged, and the tick runs at 100 Hz while these values
  move at most at the localization rate. The comparison reads the live Tcl value,
  so text an operator typed into a bound Entry is still overwritten correctly.
  The rolling pipeline summary and the age readout are additionally throttled to
  10 Hz.

Safety pilot must hold the sticks. After the four preflight steps, AUTO takes off
and hovers while the post-takeoff localization gates converge; route translation
starts only after those gates pass. Manual takeover cancels AUTO without automatic
resume, and UI/terminal shutdown stops AUTO before requesting landing.

Self-test without opening a GUI:

```bash
cd .../localization/控制介面程式/operator_interface
python3 flight_operator_app.py --selftest --max-points 30000
```

Run the desktop app:

```bash
cd .../localization/控制介面程式/operator_interface
python3 flight_operator_app.py --max-points 90000
```

Use `--interface simulated-stream` with an explicit `--video` for an ANAFI-like
720p30 replay.
The displayed frame is localized by a worker using the selected localization
runtime. Direct app invocation keeps detection off unless `--live-detect` is
passed. The production `./一鍵啟動.sh` / `start_anafi_live.sh` entrypoint loads
the three-class power-equipment detector by default (pass `--no-live-detect` to
skip it), but frames are fed to it only after the operator presses
「開始物件偵測」 (toggles to 「停止物件偵測」) or starts AUTO, which turns on
localization and detection together. Detection is independent of localization,
sends no flight command, and a detector failure does not stop AUTO. The button
is disabled when no detector was loaded.

2026-10-09 端到端 FPS：偵測開著（偵測框模式、有載入偵測模型）時，畫面與 HUD 的
「定位 FPS」改顯示「端到端 FPS(定位+偵測)」＝每秒有幾張**同一影格的定位結果與偵測結果都已
回來**（5 秒滾動）。定位在跑時，送給定位的每一張影格，只要偵測 worker 空閒就同一張一起送給
偵測（不再依 `--detect-every-n-frames` 間隔）；偵測忙碌時略過該張，**不等待也不延後定位**，
該張因此不計入。這個速率受較慢的一端限制；偵測關閉或分割模式時仍顯示原本的定位結果率。
定位 worker 自己的結果率仍記錄在 `loc_fps`（metrics JSONL），只有顯示改變。偵測每張都跑會增加
GPU 負載，AUTO 啟動時偵測也會一併開啟，現場請留意定位 FPS 與延遲是否因此變差，
必要時按「停止物件偵測」。沒有開定位、只開偵測時，偵測維持原本每 N 幀、不配對。

```bash
python3 flight_operator_app.py \
  --interface simulated-stream \
  --video /path/to/input.MP4 \
  --boot-lock-ms 2500
```

Startup behavior matches the real flight sequence: wait for the localization
worker's model-ready handshake, then hold the first 1280x720 frame while
`BOOT_INIT` / MegaLoc / local matching / PnP locks the start pose. The left map
path is produced from the same frame shown on the right; replay JSON is not used
unless `--no-live-localize` is passed.

MegaLoc has three automatic production triggers: one `BOOT_INIT` retrieval
during the landed/takeoff-initialization phase, one recovery after two
consecutive low-confidence EDM results, and one retrieval on entry to each
actual `LOST` episode. A single WEAK result only raises the EDM local reference
count. The operator UI has no manual global-retrieval or benchmark-state
buttons. On sustained LOW, replay freezes its frame; real flight sends zero
PCMD and hovers before requesting MegaLoc. AUTO does not force a manual handoff
or landing; recovery continues with EDM and requires consecutive good fixes
before motion can resume, while any SkyController stick movement still takes
control immediately.

The route contract defaults to the same `outputs/current_safezone/flight_path.json`
used by the production runner. It remains loaded, verified, and always visible on
the map. The app prints its resolved path and SHA-256 at startup. A draft route is used only when
selected explicitly with `--route-json` or `SFM_FLIGHT_PATH_JSON`. Startup fails
closed if the route is empty, malformed, not three-dimensional, or contains a
non-finite coordinate.

## Live Object Detection

YOLO runs in `object_detector_worker.py`, separate from the localization worker.
This keeps the UI and localization loop responsive while detection runs every
few frames.

Default detector settings:

- Enabled: yes through the production real-flight launcher; direct app invocation is opt-in
- Model: `執行環境/models/power_equipment_sample50_3class_yolo26n_640_aug_best.pt` under the workspace root
- Task/classes: detection; `high voltage insulator`, `transformer box`, `utility pole`
- Python: production launcher uses `.venv-yolo/bin/python`; `--detector-python`
  or `SFM_DETECTOR_PYTHON` can select an explicitly prepared alternate environment
- Input stream: same `1280x720` RGB frame shown in the right video panel
- Inference size: `640`
- Detection period: every `3` frames by default
- Confidence: `0.25`
- IoU: `0.7`

The local checkpoint and YOLO environment must exist before launch; neither is
downloaded automatically. Detection only supplies video overlays, never
localization measurements or flight-control commands. Both workers share the
GPU; enabling detection does not guarantee unchanged localization throughput.

已在目標 GPU 匯出的 TensorRT `.engine` 可用既有 `--detector-model` 或
`SFM_DETECTOR_MODEL` 明確指定；不在 UI 啟動時編譯，也不自動改換模型。
重建指令見 [`export_yolo_engine.py`](../../tools/README.md#yolo-tensorrt-fp16-引擎重建)。
FP16 batch-one 引擎維持相同矩形 640 推論尺度，回傳的框／分割輪廓仍是原始
影格座標。引擎依 GPU／TensorRT 版本而定，搬機需重建；加速量測使用正式
定位與偵測 worker 的同幀完整結果，不只計算 YOLO inference FPS。


Runtime controls:

```bash
python3 flight_operator_app.py \
  --live-detect --detector-python /path/to/.venv-yolo/bin/python \
  --video /path/to/input.MP4 \
  --detect-every-n-frames 5

python3 flight_operator_app.py --no-live-detect
```

#### Optional pretrained instance segmentation

The simulated-video launcher supports `SFM_YOLO_SEG=1`. It selects the existing
localization pipeline plus `--live-detect --detector-task segment --detector-half`,
`yolo26n-seg.pt` and the separately prepared `.venv-yolo` Python. This optional
segmentation mode is separate from production three-class detection.
The worker waits for model warmup before accepting frames, converts RGB input to
Ultralytics BGR, and returns original-input-pixel mask polygons with its boxes.

Segmentation displays its exact retained submitted source frame, with colored
stippled masks, outlines and labels. It never paints an older mask over a newer
unrelated frame or changes the source/localizer scheduling. Stop/restart/site
changes clear retained frames and session-scoped same-source pair metrics.
The HUD counts a unique successful pair only after localization and segmentation
are both presented. Source-to-presentation and earliest-serialization-to-dual-
presentation E2E latency are distinct from source FPS. The delay-equivalent FPS
is `1000 / mean submit-to-presentation ms`, not unlimited compute throughput.
See [simulated-video instructions](../影片模擬串流/README.md#同時定位與-yolo26n-seg-分割)
for commands, model/environment prerequisites and exercised GUI evidence.


If `--live-detect` is requested and the model is absent, startup fails with a
clear error instead of launching a worker that cannot become ready.
Localization and detection worker input/output are time-bounded; an exited,
non-reading, or partial-response worker is reaped and restarted without blocking
the UI shutdown path. A `success` response is accepted only with finite XYZ.

The right video panel overlays the latest detection boxes. Its bottom-left HUD
combines image FPS with the selected engineering telemetry: AUTO PCMD command cap,
localization FPS/wall/core/e2e/inliers, RTH/GPS, flight-controller altitude/AGL,
link quality, fused attitude and three-axis velocity. There is no UI log tab;
persistent session/audit logs remain on disk. The latest detection JSON is also
written to:

```text
/tmp/sfm_flight_operator_detection_status.json
```

## Parrot ANAFI Simulation Profile

The desktop simulator uses the ANAFI white-paper values as hard limits and HUD
metadata:

- Video stream: `1280x720 @ 30 fps`, simulated H264/RTP, `5 Mb/s`, `280 ms` link latency.
- Camera: video HFOV `69 deg`, digital zoom `1.0x` to `3.0x`.
- Gimbal: controllable pitch `-90 deg` to `+90 deg`, max control speed `180 deg/s`.
- Flight envelope: max horizontal speed `15 m/s`, vertical speed `4 m/s`, yaw rate `200 deg/s`.
- Flight behavior: takeoff targets `1.0 m` hover before AUTO; `hover`, `manual`, and `land` remain highest-priority safety commands.
- Battery model: linear `25 min` flight-time estimate for simulator HUD only.

Use `--video-stride N` only for debugging. The default `--video-stride 1` keeps
the 720p30 ANAFI-like stream; values above 1 intentionally drop source frames
before the UI and no longer represent the real drone stream.

Offline replay mode is still available for comparison:

```bash
python3 flight_operator_app.py --no-live-localize \
  --replay-json /path/to/localization_replay.json
```

Offline replay uses the simulator backend. `--interface real-flight` selects
`OlympeLiveBackend`; cross-interface video/flag combinations fail before either
backend is created.

Keyboard safety shortcuts:

- `起飛` and `自動飛行` remain keyboard-focusable; `Return` invokes the focused
  button just like a local click.
- `Space`: full-direction hover only, even when an action button has focus; it
  never invokes the focused button.
- `Esc`: manual

### Firmware magnetometer calibration

Panel **飛機羅盤校正** starts the ANAFI's firmware magnetometer calibration and
displays its requested X/Y/Z rotation and result. It changes the aircraft's
compass calibration, unlike the removed passive attitude check. FreeFlight 6
remains a manual fallback; the SkyController compass panel is not exposed here.

- Calibration can start only after the flight state is explicitly read back as
  `landed`, from a local operator button click, with a healthy live link.
- Starting clears pending PC movement and sends zero PCMD only when the PC already
  owns control. It never sends takeoff, motor, or movement commands.
- The aircraft reports whether calibration is required or recommended, the current
  X/Y/Z axis, completion flags, and failure state. SkyController 3 reports its own
  NotCalibrated/CalibratingX/Y/Z/Calibrated state.
- Required, unknown, failed, or in-progress calibration blocks takeoff and
  autonomous start. Firmware `recommended` status produces a warning but does not
  block operator takeoff.
- The operator must lift and rotate the powered aircraft or controller by hand as
  directed by the displayed axis. Keep propellers stopped and move away from metal,
  reinforced concrete, vehicles, magnets, and strong current-carrying cables.

On connection, the UI automatically reads and reuses calibration stored onboard.
Both valid and firmware-recommended calibration advance the compass preflight
step automatically; recommended remains visible as an advisory. Only the operator's
**重新校正** button starts a new calibration. Required, unknown, failed, or
in-progress readbacks remain blocking; old logs never override live firmware.

### Passive gravity / attitude check

The passive **姿態／重力檢查** panel has been removed from the operator interface.
It only sampled firmware attitude and saved a PASS/FAIL report; the report never
changed the aircraft, localization, map frame, PCMD, or preflight decisions.
The UI no longer loads or writes `gravity_cal_*.json`; existing reports are kept.
The standalone diagnostic helper remains available, with an offline self-test
from the workspace root:

```bash
.venv/bin/python 定位演算法/flight_control/gravity_calibration.py --selftest
```

This removal does not affect the site's measured `map_align` /
`T_align_gravity.json`. AUTO uses that map basis to split position errors into
horizontal and vertical control and to interpret heading. The map's up axis must
come from the active site's measurement, not a universal GLOMAP `-Y` assumption.

Micro-move buttons / keys use the selected backend:

- Buttons: 右上前 / 左上前 / 右下前 / 左下前 / 右上後 / 左上後 (+ 下後 補齊)
- Keys: 左搖桿為 `A/D` 左右旋、`W/S` 上下；右搖桿為 `J/L` 左右、`I/K` 前後。

- With `--interface simulated-stream`, they move only the simulated pose.
- With `--interface real-flight`, they use the protected hold-to-move PCMD path: press sends the
  bounded nudge and release returns to hover. Do not change these bindings or
  release semantics without explicit pilot review.

Safety for the live operator is defined in `../SAFETY.md`; Esc / 手動 returns
control to the safety pilot. Agents must never invoke takeoff or live movement.

## Browser Fallback

This is useful for remote monitoring or quick browser review, but is not the
preferred real-flight control surface.

```bash
cd .../localization/控制介面程式/operator_interface
python3 serve_flight_interface.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

Backend API expected by the UI:

- `GET /api/state`
  - Returns mode, localization status, pose, inliers, reprojection error, tracker state.
- `POST /api/control`
  - Body: `{"command": "hover|land|manual|auto|takeoff|boot_lock|start_auto|pause|resume|gimbal_pitch|zoom", "payload": {...}}`
  - This is the hook that should call Olympe / ANAFI commands.
- `GET /api/video.jpg` or future MJPEG/WebRTC endpoint
  - Drone live stream.
- `GET /api/map_points`
  - Downsampled point cloud for the left map panel.

Safety rule for real integration:

- `hover`, `land`, and `manual` must bypass AUTO mission logic and be handled at the highest priority in the Olympe backend.
- Losing the 720p stream should force `STREAM_LOST_HOVER` before localization or path-follow commands run.
- Losing localization should force hover before any path-follow command is sent.

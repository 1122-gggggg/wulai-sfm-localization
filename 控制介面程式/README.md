# 控制介面程式

執行期只有兩個互斥接口：

1. `影片模擬串流/啟動.sh`：固定為 `simulated-stream`，只讀影片，使用模擬 backend。
   `影片模擬串流/選擇啟動.sh` 會先用檔案選擇器挑選地圖 PLY/profile 與影片。
2. `真機串流/啟動.sh`：固定為 `real-flight`，只讀 ANAFI Olympe PDRAW，且只接受
   `SFM_MISSION_SELECTION`；site profile 由 resolver 產生。

兩個啟動器都拒絕跨接口參數；實機接口不會回退到影片檔。兩者共用同一套
定位 worker 與場域 profile，但只有實機接口載入 Olympe。起飛仍只允許操作員
本人在 UI 親手按下按鈕。

真機 AUTO 在落地或已在空中重新啟動時都必須先完成同一套三步 preflight。新鮮地速
用於比例加速限制與超速煞車；缺少或過期地速不會單獨觸發零輸出懸停。這不是物理
硬速度上限。完整操作語意以 [`SAFETY.md`](SAFETY.md) 為準。
AUTO worker 例外、2 秒 heartbeat timeout 或連續三次 PCMD 發送失敗
會鎖存 `AUTO_FAILED`、清零並停止路線。

模擬入口不帶參數時使用唯一的河濱 profile `../地圖檔/場域/river_site/site_profile.json`；影片會從
`模擬器/測試影片/` 自動選擇：P1190119.MP4 優先，否則只有一部影片時使用該影片，多部時必須明確傳入。P119 會先驗證 SHA-256，再模擬 720p30、5 Mb/s、
280 ms 無線鏈路；影片結束後停在最後完整幀。直接真機入口無預設場域；根目錄
`一鍵啟動.sh` 會明確提供隨附的定位用 mission selection，但它目前不是飛行核准。
真機 UI 先開啟、不連線；預設固定 SkyController 3（`192.168.53.1`），啟動器不做
網路探測或自動切換直連 Wi-Fi，也不因目標離線而退出。UI 開啟後按「連接真機串流」
才開始背景連線；失敗可再次按下重試，不必重開終端機。
關窗／Ctrl-C 時，控制連線正常仍須原地降落並確認落地；未確認就保留介面供重試。
飛機斷電或控制連線已失聯時（包含 USB 仍連著 SkyController、但飛機已斷線），
會停止本機控制與背景資源、關閉紀錄和視窗，不等待不可能收到的落地回報。
`cleanup_done` 此時記為 `CONTROL_LINK_LOST`、`touchdown_confirmed=false`，不是落地證據；
失聯無法區分關機與無線中斷，機體若仍在空中，只能依既有機載失聯策略處理。
連線後讀取並記錄飛機、SkyController、firmware、Olympe、Home 與 RTH。
hardware approval receipt 是可選的診斷與追溯資料；沒有 receipt 時仍依 profile、資產
hash、路線與其他真機安全條件判定是否可起飛。
介面也分別回讀 ANAFI 與 SkyController 3 的韌體羅盤校正狀態；狀態未知、必須
校正、校正失敗或進行中時會封鎖起飛與自主入口。校正只能由操作員在確認
`landed` 後按下對應按鈕，不會啟動馬達、起飛或自動移動。

選擇介面只接受能對應現有有效 site profile 的 PLY，或直接選 profile JSON；PLY
本身不包含 EDM bundle、reference poses、相機與 runtime profile，因此不會猜測或
自動拼湊不完整場域。啟動前會執行完整 preflight，失敗就不建立 GUI/worker。

兩個介面共用 `operator_interface/backend_contract.py` 內的 typed Python contract，
執行中不可熱切換模式或地圖。AUTO 是否解鎖只由當次 mission selection 的完整
flight readiness 決定；缺少校正或核准時 fail closed。

介面內的「場域資產」區把資料來源拆成三個可替換接口：完整建圖資料夾、預畫
航線 JSON、巡檢目標 JSON。完整建圖端交付清單與格式見
[`site_profiles/建圖端輸出規格.md`](site_profiles/建圖端輸出規格.md)。真機只有在
飛控明確回讀 `landed` 時才能安全重啟套用新場域；匯入和套用都不送出起飛指令。

「新增路線」／「編輯路線」的地圖右上角提供「＋／－」縮放按鈕，與滑鼠滾輪
共用縮放範圍；只改變視角，不修改航點、到站半徑或量距結果。

右側新增「物體框選模式」：左鍵拖曳先框住物體點雲，點框線／中心選取後，
用 `G` 平移、`S` 縮放，接 `X/Y/Z` 限制單軸；Z 正向為上。
例如 `G → Z → 1` 上移 1 地圖單位、`S → X → 2` 將長度加倍。
左鍵／Enter 確認，Esc／右鍵取消；框選隨路線保存，不提供避障。
「航段機頭朝向物體」可指定航點起終區間與長方體，移動時持續讓機頭朝向框中心，
例如 2→4 套用到 2→3、3→4，僅去程朝向物體，返程維持一般航向；可清除區間恢復原航向。
此設定只控制水平 yaw，不改鏡頭俯仰；到點動作、接管與起降守門不變。
詳細操作見 [航線編輯器](operator_interface/README.md)。

「新增路線」／「編輯路線」右側可開啟「兩點量距模式」，依序單擊點雲或既有
航點選 A、B，顯示三維直線距離；量距不新增、移動或儲存航點。單擊測量球選取
A／B，再按 `G` 移動，`X／Y／Z` 鎖定編輯器座標軸（Z 為高度），左鍵或 Enter
確認；距離隨移動即時更新。移動中 Esc 還原該球，保留量距模式；未移動時 Esc
清除並離開。點其他點雲／航點開始新一組；「清除測量」清空測量球。
拖曳與雙擊仍只操作視角，測量球與真正航點分開保存。
「完整路線航段」列出每個相鄰航點距離與單程總長，編輯位置、刪點、復原／重做
後同步更新；不包含返程、起飛靠攏或實際飛行繞行。側欄與航段清單均可捲動。
換圖會清除測量並重新核對尺度，未綁定或缺尺度時只顯示地圖單位，不假定 1 u = 1 m。

公尺換算讀取 canonical `profile.map_ply` 旁的 `map_distance_scale.json`
（`sfm_map_distance_scale_v1`），必須同時匹配地圖 SHA-256 與 frame ID，不使用
dense 顯示點雲的 hash。此檔只供編輯器顯示，不改地圖、航線座標或飛控門檻。
目前河濱尺度為 2026-10-04 實飛候選 **7.6164 m/u**，17 段累積 17.14 m，
`IQR／中位數 = 43%`，未通過可靠門檻，因此公尺長度標示「估算（非實測標定）」；
沒有獨立標定或可信誤差界限，不能把顯示的小數位當成精度保證。

- `operator_interface/`：操作 UI、即時定位 worker 與 ANAFI 後端。
- `site_profiles/`：每個場域的一組原子化定位資產設定。
- `影片模擬串流/`：以預錄影片進行離線驗證的入口。
- `真機串流/`：真機操作入口，必須明確指定已驗證的 mission selection。

目前只註冊 `direct` 後端。新增場域請依建圖端輸出規格建立 direct release 與 mission
selection；舊 `example_site_edm.json` 只保留歷史格式參考。真機啟動器不會預設選取任何場域。

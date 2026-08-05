# 控制介面程式

執行期只有兩個互斥接口：

1. `影片模擬串流/啟動.sh`：固定為 `simulated-stream`，只讀影片，使用模擬 backend。
   `影片模擬串流/選擇啟動.sh` 會先用檔案選擇器挑選地圖 PLY/profile 與影片。
2. `真機串流/啟動.sh`：固定為 `real-flight`，只讀 ANAFI Olympe PDRAW。

兩個啟動器都拒絕跨接口參數；實機接口不會回退到影片檔。兩者共用同一套
定位 worker 與場域 profile，但只有實機接口載入 Olympe。起飛仍只允許操作員
本人在 UI 親手按下按鈕。

模擬入口不帶參數時使用 `river_site_edm.json`；影片會從
`模擬器/測試影片/` 自動選擇：P1190119.MP4 優先，否則只有一部影片時使用該影片，多部時必須明確傳入。P119 會先驗證 SHA-256，再模擬 720p30、5 Mb/s、
280 ms 無線鏈路；影片結束後停在最後完整幀。真機入口無預設場域，
連線後讀取並記錄飛機、SkyController、firmware、Olympe、Home 與 RTH。
沒有 SHA 固定的核准 receipt 時可做地面診斷，但起飛 fail closed。
介面也分別回讀 ANAFI 與 SkyController 3 的韌體羅盤校正狀態；狀態未知、必須
校正、校正失敗或進行中時會封鎖起飛與自主入口。校正只能由操作員在確認
`landed` 後按下對應按鈕，不會啟動馬達、起飛或自動移動。

選擇介面只接受能對應現有有效 site profile 的 PLY，或直接選 profile JSON；PLY
本身不包含 EDM bundle、reference poses、相機與 runtime profile，因此不會猜測或
自動拼湊不完整場域。啟動前會執行完整 preflight，失敗就不建立 GUI/worker。

兩個介面共用 `operator_interface/backend_contract.py` 內的 typed Python contract，
執行中不可熱切換模式或地圖。所有自主路徑入口目前皆 `LOCKED`。

介面內的「場域資產」區把資料來源拆成三個可替換接口：完整建圖資料夾、預畫
航線 JSON、巡檢目標 JSON。完整建圖端交付清單與格式見
[`site_profiles/建圖端輸出規格.md`](site_profiles/建圖端輸出規格.md)。真機只有在
飛控明確回讀 `landed` 時才能安全重啟套用新場域；匯入和套用都不送出起飛指令。

- `operator_interface/`：操作 UI、即時定位 worker 與 ANAFI 後端。
- `site_profiles/`：每個場域的一組原子化定位資產設定。
- `影片模擬串流/`：以預錄影片進行離線驗證的入口。
- `真機串流/`：真機操作入口，必須明確指定已驗證的 site profile。

先從 `site_profiles/example_site_edm.json` 建立自己的設定。真機啟動器不會預設選取任何場域。

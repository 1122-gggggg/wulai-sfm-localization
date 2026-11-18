# 接口 A：真機串流

從 SkyController / 無人機讀即時 720p 串流。一般入口是 `啟動.sh`，且必須明確
指定 hash-pinned mission selection。resolver 會驗證 vehicle、site、map、localizer、
route、calibration 與 approval，再產生介面使用的唯讀 site-profile snapshot。

- 有 Olympe 連線、真機畫面與 `stick_override`（動搖桿交回）
- 固定傳入 `--interface real-flight`，只接受 ANAFI PDRAW；`--video` 會直接拒絕
- 禁止 AI 或腳本代為起飛；起飛僅操作員按 UI
- 正式飛行只允許 ANAFI 4K + SkyController 3；direct Wi-Fi 只供地面診斷
- 連線後會讀取並記錄機型、serial、firmware、Olympe、Home、RTH 與 firmware limits
- 沒有 profile 內 SHA 固定的核准 receipt 時可顯示地面診斷，但起飛 fail closed
- 未達 mission flight readiness 時 AUTO 鎖定；地面定位與診斷仍可使用
- 落地與空中重新啟動 AUTO 都必須完成同一套三步 preflight
- 水平 AUTO 需要 0.5 秒內的新鮮地速；達閾值或資料過期即清零懸停，閾值是
  fail-closed 安全閘門而非物理硬速度保證
- UI／終端機關閉會先永久鎖掉晚到 PCMD，再以獨立於 AUTO worker 的路徑原地降落；
  未確認 touchdown 時不關 UI、不斷線
- 高度／距離預設 50 m／100 m，只能在確認落地時修改；高度 49 m 或距離
  95 m 時，Home reachable 則 RTH 並降落，否則原地降落

```bash
SFM_MISSION_SELECTION=/absolute/path/to/selection.json ./啟動.sh
```

UI 先開啟、不連線；啟動器不探測目標網路、不因真機離線而退出，也不自動選擇
直連 Wi-Fi。預設目標固定為 SkyController 3（`192.168.53.1`）；顯式 `IP`／`CTRL`
覆寫可推導已知配對的另一值，不一致組合仍會拒絕啟動。
UI 開啟後按「真機串流」才開始背景連線；失敗可再次按下重試，不必重開終端機。
開 UI 與連線不會起飛，起飛仍由操作員本人親自按下。
`SFM_LAUNCH_DRY_RUN=1` 只印出啟動指令，不啟動 UI、不操作網路。

正式入口預設同時啟動 MegaLoc＋EDM 定位與獨立三類電力設備偵測 worker。
偵測權重為工作區內的 `執行環境/models/power_equipment_sample50_3class_yolo26n_640_aug_best.pt`，
使用工作區 `.venv-yolo/bin/python`；可用 `SFM_DETECTOR_PYTHON` 覆寫已備妥的環境。
權重與環境必須事先存在，不自動下載。偵測框／類別／信心值只顯示在影片窗格，
不參與定位或飛控；兩個 worker 共用 GPU，不能視為即時效能保證。
要明確停用偵測，可執行 `./啟動.sh --no-live-detect`。

啟動器會在重開機後重新探測有效的 X/XWayland `DISPLAY`；找不到可見桌面時拒絕啟動。
執行中不能切到 SIM 或更換地圖，必須安全關閉後以另一啟動器重開。

舊的河濱 XFeat TRACK-lock 測速入口已退出工作區；現有研究元件由
`../../定位演算法/validation/` 管理，不是真機操作入口。

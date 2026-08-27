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
- 落地與空中重新啟動 AUTO 都必須完成同一套四步 preflight
- 水平 AUTO 需要 0.5 秒內的新鮮地速；達閾值或資料過期即清零懸停，閾值是
  fail-closed 安全閘門而非物理硬速度保證
- UI／終端機關閉會先永久鎖掉晚到 PCMD，再以獨立於 AUTO worker 的路徑原地降落；
  未確認 touchdown 時不關 UI、不斷線
- 高度／距離預設 50 m／100 m，只能在確認落地時修改；高度 49 m 或距離
  95 m 時，Home reachable 則 RTH 並降落，否則原地降落

```bash
SFM_MISSION_SELECTION=/absolute/path/to/selection.json ./啟動.sh
```

啟動器會在重開機後重新探測有效的 X/XWayland `DISPLAY`；找不到可見桌面時拒絕啟動。
執行中不能切到 SIM 或更換地圖，必須安全關閉後以另一啟動器重開。

舊的河濱 XFeat TRACK-lock 測速入口已退出工作區；現有研究元件由
`../../定位演算法/validation/` 管理，不是真機操作入口。

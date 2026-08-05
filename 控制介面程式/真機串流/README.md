# 接口 A：真機串流

從 SkyController / 無人機讀即時 720p 串流。一般入口是 `啟動.sh`，且必須明確
指定包含同座標地圖、定位 bundle、reference poses 與相機內參的 site profile。

- 有 Olympe 連線、真機畫面與 `stick_override`（動搖桿交回）
- 固定傳入 `--interface real-flight`，只接受 ANAFI PDRAW；`--video` 會直接拒絕
- 禁止 AI 或腳本代為起飛；起飛僅操作員按 UI
- 正式飛行只允許 ANAFI 4K + SkyController 3；direct Wi-Fi 只供地面診斷
- 連線後會讀取並記錄機型、serial、firmware、Olympe、Home、RTH 與 firmware limits
- 沒有 profile 內 SHA 固定的核准 receipt 時可顯示地面診斷，但起飛 fail closed
- 自主路徑所有入口目前無條件 `LOCKED`；只保留即時定位與人工控制
- 高度／距離預設 50 m／100 m，只能在確認落地時修改；高度 49 m 或距離
  95 m 時，Home reachable 則 RTH 並降落，否則原地降落

```bash
SFM_SITE_PROFILE=/absolute/path/to/site.json ./啟動.sh
```

啟動器會在重開機後重新探測有效的 X/XWayland `DISPLAY`；找不到可見桌面時拒絕啟動。
執行中不能切到 SIM 或更換地圖，必須安全關閉後以另一啟動器重開。

舊的河濱 XFeat TRACK-lock 測速入口已退出工作區；現有研究元件由
`../../定位演算法/validation/` 管理，不是真機操作入口。

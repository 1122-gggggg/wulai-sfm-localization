# 接口 A：真機串流

從 SkyController / 無人機讀即時 720p 串流。一般入口是 `啟動.sh`，且必須明確指定包含同座標地圖、定位 bundle 與安全航線的 site profile。

- 有 Olympe 連線、真機畫面與 `stick_override`（動搖桿交回）
- 禁止 AI 或腳本代為起飛；起飛僅操作員按 UI
- Target Site 正式 EDM profile 的 `route_json` 目前為 `null`，不可用於 LIVE

舊的河濱 XFeat TRACK-lock 測速入口已移到 `../../候補定位方法/真機測速/`，仍保留「不起飛」限制。

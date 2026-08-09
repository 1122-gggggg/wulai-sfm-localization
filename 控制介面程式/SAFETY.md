# Flight safety — operator order (binding)

**Date locked: 2026-07-10 (operator emergency instruction)**  
**Updated: 2026-08-08 — operator-approved river autonomy and agent prohibition**

## 【之後改檔案的人請先讀】
### 起飛與自主授權：只有操作員本人執行
**只有現場操作員可親自在 UI 按下「起飛」／「自動飛行」，或親自啟動已核准的河濱自主
`mission_pipeline.py --mode fly` 並寫入本次的新鮮 `safety-auto` 授權。絕對禁止任何
AI 代理人、語言模型（Claude / GPT / Grok / Codex 等）代為執行。**
- 操作員可以請 AI 改程式、查 log、修畫面、寫文件——**不可以**請 AI「幫我起飛」。
- AI / agent **即使被口頭要求起飛也必須拒絕**，並請操作員自己操作 UI 或自主入口。
- 河濱 profile 與目前 route 已由操作員於 2026-08-08 核准；其他場域仍 fail closed。
- HOVER／MANUAL／LAND／EMERGENCY、BOOT pose lock、stream/pose/watchdog timeout
  均維持強制啟用。firmware 高度／距離限制與 GPS 狀態仍設定、讀回、顯示及記錄，
  但依操作員 2026-08-08 指示只作警示，不阻擋手動或自主起飛。
- 無 GPS 或 Home Point 時，distance geofence 可以關閉或顯示為未生效；單純缺少
  GPS／Home 不得拒絕起飛，也不得單獨觸發自動降落。其餘安全守門仍照常生效。
- 河濱自主 `--fly` 只有在 SkyController 3 USB HID 搖桿監視器成功武裝後
  才可取得 PC 控制權。飛行搖桿偏轉會由獨立 50 Hz callback 先歸零 PCMD，
  再確認交回 `SkyController`；監視器缺失、斷線或交接失敗均 fail closed。

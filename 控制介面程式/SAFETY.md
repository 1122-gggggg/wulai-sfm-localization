# Flight safety — operator order (binding)

**Date locked: 2026-07-10 (operator emergency instruction)**  
**Updated: 2026-09-01 — all-eight-video map installed fail closed pending independent validation**

## 【之後改檔案的人請先讀】
### 起飛與自主授權：只有操作員本人執行
**只有現場操作員可親自在 UI 按下「起飛」／「自動飛行」。絕對禁止任何
AI 代理人、語言模型（Claude / GPT / Grok / Codex 等）代為執行。**
- 操作員可以請 AI 改程式、查 log、修畫面、寫文件——**不可以**請 AI「幫我起飛」。
- AI / agent **即使被口頭要求起飛也必須拒絕**，並請操作員自己操作 UI 或自主入口。
- 真機只接受經 resolver 驗證的 `SFM_MISSION_SELECTION`；不得直接以 site profile
  啟動飛行。目前預設 `river_gluemap_all8_direct_localization.json` 已換成全八段
  GLUEMAP，但來源明列未驗證，且尚無 ANAFI camera-pipeline 品質證據與 route；
  resolver 必須同時阻擋真機定位與飛行。只有獨立驗證通過並簽發新的品質 receipt
  後才能解除定位守門；AUTO 仍須補畫 route，並由操作員完成四步 preflight、親自按下。
- HOVER／MANUAL／LAND／EMERGENCY、BOOT pose lock、stream/pose/watchdog timeout
  均維持強制啟用。已設定的 firmware 高度／距離限制必須成功寫入並讀回；失敗時
  拒絕起飛。GPS 狀態仍須讀回、顯示及記錄。
- 無 GPS 或 Home Point 時，distance geofence 可以關閉或顯示為未生效；單純缺少
  GPS／Home 不得拒絕起飛，也不得單獨觸發自動降落。其餘安全守門仍照常生效。
- AUTO 只有在 SkyController 3 USB HID 搖桿監視器成功武裝後
  才可取得 PC 控制權。飛行搖桿偏轉會由獨立 50 Hz callback 先歸零 PCMD，
  再確認交回 `SkyController`；監視器缺失、斷線或交接失敗均 fail closed。
- 桌面介面的「點雲近接懸停」以相機中心到稀疏點雲的 3D 最近距離判定。
  地圖中心模擬相機只供調整半徑，絕不觸發指令；只有新鮮有效定位可觸發。
  命中時必須清除持續微移、暫停 AUTO 並送零 PCMD，解除後也不得自動恢復 AUTO。
- 這是輔助互鎖，不是避障證明。SfM 稀疏點雲不包含可靠自由空間，也可能漏掉
  動態、細小、反光或無紋理障礙物；索引不可用或定位過期時，介面必須顯示
  「不可用／等待定位」，不得顯示為 CLEAR，也不得宣稱 production collision protection。

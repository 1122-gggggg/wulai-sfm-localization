# 工作區完整稽核與優化清單

初始稽核日期：2026-08-03；更新日期：2026-08-08。範圍：
`/home/allen/localization`。2026-08-05 經操作員明確授權，已永久刪除 35 組
`__pycache__`、pytest／ruff cache、`執行環境/inductor_cache/`，以及原本位於
`模擬器/封存/parrot_stimulate_standalone_20260803/` 的 7.5 GiB 舊獨立工作樹。
該封存原本由外層 Git 忽略，目前不能由此 repository 回復。地圖、bundle、正式
模型、測試影片、飛行紀錄、整合版模擬器與飛控程式均未刪除。

## 結論

正式 EDM runtime、兩個串流入口、site profile、場域資產 SHA、離線模型與安全驗證
已有明確邊界。2026-08-08 已移除 flight/deploy 之間的 Python runtime 副本，改由
11 個明確 owner 與 module-ownership gate 防止重複回流；安全命令檔也統一到
owner-private 路徑。P119 固定 SHA 全片重播通過既有品質 gate，第一方 coverage
門檻提升為 50.00%。

下列容量數字、測試數與優化清單保留 2026-08-07 的盤點口徑；它們是歷史證據，
不是即時狀態。現行結果以 `./驗證系統.sh` receipt 與本次維護稽核為準。

刪除後檔案系統約有 18.2 GiB／13.0% 可用空間；主要目錄大小如下：

| 類型 | 約略大小 | 處置 |
|---|---:|---|
| `模擬器/` | 11.53 GiB | 測試影片、整合版與 convergence 歷史證據；舊封存已刪除 |
| `.venv/` | 7.78 GiB | 可重建但目前正式 runtime 需要；不刪 |
| `outputs/` | 2.05 GiB | 依 operations／validation／experiment evidence 分類 |
| `地圖檔/` | 1.29 GiB | 場域正式資產；禁止自動清理 |
| `執行環境/` | 1.06 GiB | 主要是完全離線 MegaLoc/XFeat cache；保留 |
| `定位演算法/` | 0.42 GiB | 程式、EDM runtime、驗證與權重 |

## 已驗證正常

- 模擬、真機與單一驗證入口各自唯一，SIM／REAL 禁止熱切換。
- 三個實際 site profile 均能解析且 `flight.approved=false`。
- 沒有 broken symlink；EDM 工具包使用有效 symlink 指向唯一實作。研究候補方法
  直接由 `定位演算法/` 管理，不再建立頂層重複索引。
- runtime 共用模組各有唯一 owner；檢查會拒絕缺檔、重複副本與未分類同名檔。
- 主測試 820 passed、1 skipped；Parrot Python 3.11 測試 86 passed，且本機固定
  firmware manifest/SHA preflight 通過。
- CUDA fail-closed、EDM checkpoint／bundle／profile SHA 與離線載入正常。

## 優化優先序

### P1：先處理

1. **磁碟門檻已解除。** 操作員授權刪除可重建 cache 與舊模擬器封存後，餘量由
   低於 5% 提升到約 18.2 GiB／13.0%，目前高於實機 preflight 的 fail-closed 門檻。
   地圖、影片與飛安紀錄仍不納入自動清理。
2. **P119 品質 gate 尚未全綠。** 成功定位由 1686 提升至 1956 幀，p95 延遲改善，
   但 inliers p50 690 低於既有 693。不得為了綠燈直接降低 gate。
3. **EDM 真實吞吐。** P119 原生 23.984 FPS 下，UI 端到端中位約 14.9 FPS；主要
   成本是 TRACK matcher 約 44.7 ms，加上 PnP、worker queue 與 UI 回傳。先前約
   20 FPS 的 session 含 30 FPS 補出的重複影格，不是同口徑的真實來源吞吐。

### P2：分階段重構

1. `flight_operator_app.py` 約 7058 行。場域資產面板與自製航線編輯器已拆出
   model／controller／window；剩餘內容仍宜依「影片 source／worker client／metrics／
   rendering」分階段拆模組。飛行按鍵、Esc、關窗、PCMD 與起降語意必須保持不動並先補
   characterization tests。
2. `olympe_live_backend.py` 約 2768 行，宜先抽出純 read-only telemetry formatting
   與 inventory，再考慮 command dispatch；安全命令不可順手重構。
3. 8 組 compatibility mirror 應長期改為 packaging 時生成；目前不能直接刪，因為
   可搬移包與飛控仍依賴兩種目錄形狀。
4. XFeat、ONNX/TensorRT、NeuFlow 與 projection-guided 程式仍在 `定位演算法/` 留作研究重現。
   production 已鎖 EDM；長期應由 package build 排除候補 backend，而非現在搬動
   import 路徑。
5. `workspace_layout.py` 的頂層 `地圖檔/maps|bundles|mission_routes` 僅為 legacy
   fallback；正式 site profile 已使用 `地圖檔/場域/<site>/`。待 legacy CLI 移除後
   再刪 fallback。

### P3：可維護性

- 模擬影片改為工作區內可携式自動選擇：P119 固定 SHA 優先，只有一部影片時自動選用，多部時要求 `VIDEO=` 明確指定。
- 操作 UI 已改用自製航線編輯器，但 `控制介面程式/authoring/` 的舊 Blender／獨立繪圖工具
  仍被 `mission_pipeline.py` 與 `mission_configs/mission_defaults.json` 引用，且部分預設路徑是舊機器的
  `/media/...`。這些是明確的 legacy 遷移項目，在 mission pipeline 移除對應指令前不可直接刪除。
- `path_follow_flight.py` 尚有 `/home/allen/足球場` legacy fallback；自主飛行鎖定解除
  前應移除，但目前不可在缺少現場核准時順便改飛控。
- CodeGraph、pytest、ruff cache 都是可重建資料；只在磁碟緊急且沒有程序使用時清理。

## 本輪整理

- 新增 `tools/workspace_audit.py`，唯讀檢查目錄、入口、symlink、output 分類、容量與磁碟。
- 場域資產 UI 已分為 panel／actions／interfaces／local provider，航線編輯器另分為
  model／controller／window，不持有 Olympe backend 或飛行指令接口。
- 新增根目錄 `requirements.txt`、`tools/install_runtime.sh` 與 `tools/simulator_preflight.py`，統一搬移環境、資產與影片啟動前檢查。
- `outputs/README.md` 補上 session、validation receipt、P119 與命名／retention 規則。
- 建立 `文件/` 作為架構、Spec 與稽核的唯一索引；驗證實作與測試收入 `tools/`。
- 舊獨立 `parrot_stimulate` 封存已於 2026-08-05 經操作員明確授權永久刪除；
  已整合且測試較完整的 `模擬器/parrot_stimulate/` 保持不變。
- 移除已不存在的頂層候補／封存目錄要求，候補實作由 `定位演算法/` 直接索引。
- 更新根 README、架構文件與 Spec 的 As-Is，移除過時路徑。

重跑：

```bash
python tools/workspace_audit.py --strict-output-names
./驗證系統.sh
```

## 2026-08-07 執行契約補充

本文件上方的 2026-08-05 容量數字是當次稽核的歷史觀測，不代表目前主機容量。
現在由 `tools/workspace_audit.py` 以唯讀方式重新計算：`audit_YYYYMMDD` 目錄歸類
為 governance／稽核證據；workspace 總量超過 20 GiB 或檔案系統可用空間低於 15%
時，各自輸出明確 `WARNING`，不自動刪除資料，也不把 warning 誤報成通過 release。
低容量時應停止建立大型驗證產物並由操作員處理；請以該次 audit receipt 的
`workspace_size_bytes`、`storage_warnings` 與可用空間觀測為準，不沿用上述歷史數字。

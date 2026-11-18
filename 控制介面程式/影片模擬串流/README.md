# 接口 B：影片模擬串流

把預錄影片當 **720p 串流** 餵進同一套定位 worker。入口固定傳入
`--interface simulated-stream`；`--live` 或 `--interface real-flight` 會直接
拒絕。**不載入 Olympe、不連真機、無真機 TakeOff**。UI 的起飛、降落、
懸停與微移只更新模擬狀態。

不帶參數時預設使用 `../../地圖檔/場域/river_site/site_profile.json`。影片從
`模擬器/測試影片/` 自動選擇：P1190119.MP4 優先，或只有一部影片時使用它；多部影片必須用 `VIDEO=` 或第一個參數指定。P119 啟動前核對固定 SHA-256。串流預設為 720p30、
H.264 Main、5 Mb/s、280 ms、0% 丟包；`ANAFI_LINK_PRESET` 可選
`nominal`、`loss-1`、`loss-3` 或 `loss-5`。P119 容器宣告 2,935 幀、實際
可解碼 2,934 幀，只對這個 SHA 標記 `KNOWN_INCOMPLETE`。播完不循環，
介面停在最後完整幀。

需要檔案選擇器時使用：

```bash
./選擇啟動.sh
```

它先選地圖 PLY 或 site profile JSON，再選 MP4/MOV/MKV；PLY 必須能唯一對應
`控制介面程式/site_profiles/` 中的有效 profile，否則會拒絕啟動。

選 PLY 時會跳出「選擇定位模式」視窗（也可用 `--mode localize|expand` 直接指定）：

- **純定位（原始版本）**：使用該 PLY 對應的原 profile。
- **定位＋擴圖（擴圖版本）**：換成 `地圖檔/場域/<場域>/releases/` 內、
  地圖 PLY SHA-256 相同且 `online_workmap.enabled=true` 的場域包。找不到或有多個時
  此選項停用並顯示原因，不會靜默退回純定位。

直接選 profile JSON（或 `--site-profile`）時不跳視窗，模式由該 profile 決定；
與 `--mode` 衝突時拒絕啟動。啟動後使用同一個介面、同一套 BOOT／LOST／凍幀與飛行
規則；模式只決定 worker 載入哪份定位演算法。選擇發生在啟動前，視窗內的「切換場域」
不能切換模式：它以主程式的 loader 匯入場域包，無法驗證擴圖 profile。
擴圖場域包 `flight.approved=false`，不授予 AUTO 或真機核准；真機串流仍只接受
`SFM_MISSION_SELECTION`，本選項不改變該路徑。

## 預設：飛行管線（與真機狀態機一致）

```
第一幀 ──凍幀──► MegaLoc 定位啟動組合 (BOOT_INIT)
                    │ 鎖定成功
                    ▼
                 TRACK 持續定位
                    │ 中途無法定位 (LOST)
                    ▼
              暫停串流 · 凍幀 · MegaLoc 重抓
                    │ 找回
                    ▼
                 恢復 TRACK · 串流繼續
```

MegaLoc 在 `BOOT_INIT` 起飛初始化執行一次、連續 2 筆低信心 EDM 結果時
升級執行一次，以及每次真正進入 `LOST` 時執行一次。單筆 WEAK 只提高 EDM
候選數；低信心升級時先凍結模擬影格。正式操作介面沒有手動強制 MegaLoc
的按鈕。

| 檔案 | 用途 |
|------|------|
| `啟動.sh` | 離線定位入口；以 `SFM_SITE_PROFILE` 指定場域設定 |

## 範例

```bash
# 預設河濱 + P119
./啟動.sh

# 明確覆寫場域與影片；不連真機、不會真機起飛。
SFM_SITE_PROFILE=/absolute/path/to/your_site.json \\
  ./啟動.sh /absolute/path/to/video.mp4
```

### 福和橋新場域與 P1130113

福和橋場域已匯入 `地圖檔/場域/fuhe_bridge/`，沿用工作區的 direct 定位器，
不覆蓋河濱 profile、預設場域或真機 mission selection。在工作區根目錄執行：

```bash
SFM_SITE_PROFILE="$PWD/地圖檔/場域/fuhe_bridge/site_profile.json" \
VIDEO="$PWD/模擬器/測試影片/P1130113.MP4" \
./控制介面程式/影片模擬串流/啟動.sh
```

也可從 `選擇啟動.sh` 選福和橋的 profile 或
`地圖檔/場域/fuhe_bridge/releases/fuhe_edm_transfer_20261006/fuhe_edm_release/map/map.ply`，
再選影片。首次顯示「這個場域還沒有航線」時選 **No**，只重播定位，不建立航線。

原始 ZIP 位於工作區的 `地圖檔/場域/fuhe_bridge/provenance/fuhe_edm_transfer_20261006.zip`；
解壓的交付包與內部 `SHA256SUMS` 保留在場域的
`releases/fuhe_edm_transfer_20261006/`。上層 `site_profile.json` 只調整接入路徑，
地圖、bundle、定位 profile、參考位姿及其 SHA 不變。

P113 原檔為 3840×2160、約 62.1 秒，輸入轉為 1280×720，不裁切或再去畸變；
960×360 是影片預覽拼圖，不是輸入尺寸。串流上限設為 30 FPS，原片約 24 FPS
不補幀；模擬採 H.264 Main、5 Mb/s、280 ms 與 0% 丟包。播放完停在最後完整幀，不循環。

此場域沒有航線、重力對齊、實測尺度或真機驗收，`flight.approved=false`，
AUTO 維持鎖定。P113 也是建圖來源影片之一，重播成功不代表獨立定位精度驗證。

本機驗證：ZIP 校驗相符、內部 883 筆 SHA 通過、離線 full-runtime preflight
通過，模型資產與 preflight 回歸測試 21 項通過。原片容器宣告 1,488 幀，
`ffprobe` 可解碼 1,487 幀；同一 FFmpeg 串流類別完整輸出 1,487 幀，
encoder／decoder 正常結束。實際操作介面亦到達 `EOF_HOLD`。

完整介面重播有 1,199 筆 TRACK、139 筆 WEAK_TRACK、11 筆 LOST；片尾剩
48 個地圖 inliers、319 個 VO inliers，屬低信心補位，不是強地圖定位。
飛控 command log 為空。驗證收據與截圖在
[`outputs/analysis/fuhe_edm_replay_20261006/`](../../outputs/analysis/fuhe_edm_replay_20261006/)。

### 同時定位與三類電力設備偵測

福和橋場域與指定錄影可用下列正式介面入口，保持離線、不啟動 AUTO：

```bash
SFM_SITE_PROFILE="$PWD/地圖檔/場域/fuhe_bridge/site_profile.json" \
VIDEO="$PWD/模擬器/測試影片/P1130113.MP4" \
SFM_LIVE_DETECT=1 \
./控制介面程式/影片模擬串流/啟動.sh
```

權重預設為工作區內的 `執行環境/models/power_equipment_sample50_3class_yolo26n_640_aug_best.pt`，
可用 `SFM_DETECTOR_MODEL` 指定已存在的其他權重。使用 `.venv-yolo/bin/python`，
不下載模型。偵測框、類別與信心值只疊加在影片窗格，不參與 MegaLoc＋EDM 定位
或飛控；原片轉為 1280×720，沿用 H264 Main 5 Mb/s、280 ms 模擬鏈路。
沒有航線時選 No，不建立或啟動航線。

2026-10-06 的 100 秒實際介面驗證：兩個 GPU worker 均就緒，定位與偵測均回傳
成功結果，偵測輸出包含高壓絕緣子、變壓器箱、電線桿三類。以 100 ms 觀測抽樣
記錄 555 組定位結果與 437 組偵測結果；這些不是全部影格計數或同幀配對 FPS。
抽樣定位包含 507 筆 TRACK、44 筆 WEAK_TRACK、4 筆 LOST；低信心凍幀保持原樣，
來源停在第 1,332 幀（約 55.5 秒），未把此次驗證當成全片強定位或辨識準確率驗收。
收據與畫面見 [`fuhe_power_detector_20261006`](../../outputs/analysis/fuhe_power_detector_20261006/)。

### 同時定位與 YOLO26n-seg 分割

在工作區根目錄執行：

```bash
SFM_YOLO_SEG=1 \
VIDEO="$PWD/模擬器/測試影片/P1980198.MP4" \
./控制介面程式/影片模擬串流/啟動.sh
```

使用官方 COCO 預訓練 `yolo26n-seg.pt`、640 letterbox、FP16；
不訓練、不微調、不下載。定位仍是原本的 MegaLoc＋EDM 背景重定位及 KLT/PnP
前景追蹤，分割不參與定位或飛控。`SFM_YOLO_SEG=1` 優先於 `SFM_LIVE_DETECT`；
兩者均未啟用時維持僅定位的行為。

分割畫面保留該次提交的原始來源幀，疊上彩色半透明點陣 mask、外框、類別與
信心值，不把舊 mask 畫在較新的不同影格。來源與定位仍獨立繼續處理。
取消／重開定位或換場域時清除舊分割畫面與配對統計。

HUD 分開顯示來源 FPS、成功且唯一的同幀雙路交付 FPS、來源到雙路呈現的
E2E 延遲，以及最早序列化／提交到雙路呈現的 E2E 延遲。
「延遲等效 FPS」是 `1000 / 平均提交到呈現毫秒`，不是不限速的吞吐上限；
同幀交付率受來源、定位、分割與地圖／影片呈現節拍影響，不把它當成串流 FPS。
BOOT／LOST 的同幀重試不重複計數，失敗另列。

2026-10-05 實際 1440×900、50 秒 SIMULATED-STREAM 驗證有定位與分割 GPU
worker、1160 次分割畫面呈現、595 組同幀雙路呈現；最後 5 秒交付約 14.03 FPS、
來源到呈現平均 63.17 ms、提交到呈現平均 61.96 ms。這不是 30 FPS
整體即時保證，也不等於先前無 GUI 的不限速運算吞吐。
截圖與收據在 `outputs/analysis/localization_yolo26n_seg_ui_20261005/`。

本機 `.venv-yolo` 是另外備妥的選配環境，不寫入正式 `.venv` 的依賴；其 `.pth`
使用本機絕對路徑，不能當成可攜式發布環境。COCO 是一般物件類別，mask
不代表草地／道路或電力設備的專用辨識，也不是避障保證。


## 環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `LOCAL_TOPK` | 1 | TRACK 候選數；WEAK/LOST 依 profile 和 tracker 設定調整 |
| `STREAM_FPS` | 30 | 餵入 FPS |
| `BOOT_LOCK_MS` | 20000 | 首幀 MegaLoc 最長等待；**鎖定成功立刻放行** |
| `LOST_HOLD_MAX` | 8 | LOST 凍幀重試次數 |
| `LOST_HOLD_TIMEOUT_MS` | 15000 | LOST 凍幀逾時後放行串流 |
| `LOC_BENCH_TRACK=1` | off | 改純 TRACK 測速 |
| `POSE_STABILIZE=1` | off | 3 幀一致性＋0.15 s 低通輸出；遙測同時保留原始 PnP |
| `SFM_LIVE_DETECT` | 0 | 1 同時啟動三類電力設備偵測 |
| `SFM_DETECTOR_MODEL` | `執行環境/models/power_equipment_sample50_3class_yolo26n_640_aug_best.pt` | 工作區內必須事先存在的偵測權重 |
| `SFM_YOLO_SEG` | 0 | 1 啟用預訓練分割與 mask 顯示；不改定位／飛控 |
| `SFM_YOLO_SEG_MODEL` | `執行環境/models/yolo26n-seg.pt` | 必須事先存在的分割 `.pt` 權重 |
| `SFM_DETECTOR_PYTHON` | `.venv-yolo/bin/python`（偵測／分割模式） | 已備妥 Ultralytics 與 CUDA Torch 的選配 Python |

## 與真機接口

| | 真機串流 | 影片模擬串流 |
|--|----------|----------------|
| 畫面 | Olympe PDRAW | FFmpeg 讀檔 |
| BOOT MegaLoc | 真機串流首幀 | 凍第一幀模擬 |
| LOST | 飛機實際懸停 | **暫停影片串流** 凍幀重抓 |
| 飛控 | 可（人類起飛） | 無 |

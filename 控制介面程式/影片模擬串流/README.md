# 接口 B：影片模擬串流

把預錄影片當 **720p 串流** 餵進同一套定位 worker。入口固定傳入
`--interface simulated-stream`；`--live` 或 `--interface real-flight` 會直接
拒絕。**不載入 Olympe、不連真機、無真機 TakeOff**。UI 的起飛、降落、
懸停與微移只更新模擬狀態。

不帶參數時預設使用 `../site_profiles/river_site_edm.json`。影片從
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

## 與真機接口

| | 真機串流 | 影片模擬串流 |
|--|----------|----------------|
| 畫面 | Olympe PDRAW | FFmpeg 讀檔 |
| BOOT MegaLoc | 真機串流首幀 | 凍第一幀模擬 |
| LOST | 飛機實際懸停 | **暫停影片串流** 凍幀重抓 |
| 飛控 | 可（人類起飛） | 無 |

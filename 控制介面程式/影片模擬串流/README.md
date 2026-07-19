# 接口 B：影片模擬串流

把預錄影片當 **720p 串流** 餵進同一套定位 worker。**不連真機、無 TakeOff**。

## 預設：飛行管線（與真機狀態機一致）

```
第一幀 ──凍幀──► MegaLoc 起飛組合 (BOOT_INIT)
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

| 檔案 | 用途 |
|------|------|
| `啟動.sh` | 離線定位入口；以 `SFM_SITE_PROFILE` 指定場域設定 |

## 範例

```bash
# 使用你的場域 profile；不連真機、不會起飛。
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
| `POSE_STABILIZE=1` | off | 3 幀一致性＋0.15 s 低通輸出；遺測同時保留原始 PnP |

## 與真機接口

| | 真機串流 | 影片模擬串流 |
|--|----------|----------------|
| 畫面 | Olympe PDRAW | FFmpeg 讀檔 |
| BOOT MegaLoc | 真起飛後首幀 | 凍第一幀模擬 |
| LOST | 飛機實際懸停 | **暫停影片串流** 凍幀重抓 |
| 飛控 | 可（人類起飛） | 無 |

# TRACK 全影片視覺參考軌跡比較 (2026-07-14)

> **2026-07-14 更新：本文的 production 選擇已被推翻。** 操作者依準度決定**移除 mutual-NN 快路**，
> TRACK/WEAK 改為每幀跑 XFeat + LighterGlue adaptive 3→5（即本文的「LG reference」路徑本身）。
> temporal cache 只在 NN 分支被查詢，故現已失效。詳見現行 production 設定與驗證紀錄。
> 本文以下的 `deep_nn_then_lg_3_5` 相關結論僅作為歷史量測保留。


## 結論

精度優先的 production 組合維持 `deep_nn_then_lg_3_5`：XFeat top-K 1700、cosine mutual-NN 快路、adaptive PyTorch LighterGlue 3 -> 5 fallback、temporal cache 開啟、光流關閉。

- `deep` 是四支影片中最保守、一致的 production 選擇。相對全量 XFeat+LighterGlue 參考軌跡，困難影片 P121 的位置差 P95 為 0.130 m，明顯優於 NeuFlow 100/100 的 0.441 m。
- `NeuFlow` 在容易區段有很大速度優勢，但困難影片的大偏差與抖動不可接受，不應取代 production deep 快路。
- `projection-guided` 目前實作的尾延遲太高，四支影片整體只有 9.49-18.80 FPS，現階段不採用。
- 目前沒有任何方案能在四支影片中同時證明「不降精度」且正常 TRACK 穩定超過 23 FPS。deep 只在 P024/P072 超過 23 FPS，P017/P121 分別為 19.17/15.82 FPS。

## 絕對重要的限制

依照要求，本次沒有讀取、解析、融合或新增任何 GPS/GNSS 資訊。所有結果只來自影像、地圖特徵、共視關係與 PnP 輸出。

「全量 XFeat+LighterGlue」只是 image-only pseudo-ground-truth，不是實測絕對 ground truth。它與候選方法共用 XFeat、地圖、相機參數與 PnP，因此可用來比較軌跡一致性、幀間運動差和抖動，但無法測出兩者共有的絕對偏差。若要得到公分級絕對精度，需要全站儀控制點、動作捕捉或已測量的 AprilTag 軌跡等非衛星參考。

## 測試設定

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU。
- CPU: Intel Core i7-13620H, 16 logical CPUs。
- 電源模式：`performance`。
- 全部輸入送入定位器前為 1280x720；P017/P072 的原片解碼與縮放成本計入「整體實際 FPS」。
- 相機：ANAFI 720p `FULL_OPENCV`，並且與 production calibration 逐參數相等。
- 地圖 bundle SHA-256: `8227e3bd37d4d99966ae1fb307060f9bc8e38e6e5269ab918f64b87957c6ccab`。
- 所有影片 stride=1，每幀執行，沒有用抽幀結果代替全影片結果。
- 「正常 TRACK」只統計 `TRACK` 與 `NEUFLOW_TRACK`，排除 BOOT_INIT/LOST/WEAK_TRACK；FPS 由中位 `wall_ms + frame load_ms` 計算。
- 「整體實際 FPS」是全程 wall time，包含影片 I/O、解碼、縮放、BOOT/LOST/WEAK 與結果收集；它最接近此離線串流程的實際 throughput。

### 共同 production 參數

| 區塊 | 參數 |
|---|---|
| TRACK 候選 | local top-5, weak local top-8, near pool 24, radius 0.8, yaw <=90 deg |
| 特徵 | TRACK XFeat top-K 1700, acquire top-K 2048 |
| NN 快路 | mutual cosine NN, score >=0.85 |
| 強接受 | >=100 unique inliers, RMS <=3.5 px |
| LighterGlue fallback | PyTorch LighterGlue，先 3 refs，不足再到 5 refs |
| 接受門 | acquire/track/weak min inliers = 80/50/30；acquire/track max RMS = 5/6 px |
| PnP | RANSAC max error 5 px, max jump 2 m |
| temporal cache | on, anchors 80-2048, max age 2, min score 0.85, seed >=150 inliers and RMS <=3.5 px |
| BOOT/LOST | MegaLoc top-30 + XFeat 2048 + LighterGlue，流程沒有改變 |

### 候選方法

| 名稱 | 設定 |
|---|---|
| LG reference | 每幀 XFeat 1700 + `matcher_mode=lighterglue` + adaptive 3 -> 5；temporal cache 雖保持設定，此路徑的 cache-used/accept 皆為 0 幀 |
| deep | mutual-NN 快路 + PyTorch LighterGlue 3 -> 5 fallback，temporal cache on |
| NeuFlow standard | deep 為 refresh/fallback；NeuFlow-v2 512x288, refresh=3, seed/track=100/100, inliers >=50, ratio >=0.25, RMS <=4 px |
| NeuFlow loose | 僅 P121 敏感度測試；seed/track=80/60，其餘同 standard |
| projection | 3D landmarks 投影，15 -> 25 -> 40 px，cosine >=0.60，ratio <=0.95，不足回退 deep |

## 完整速度與成功率

| 影片 | 方法 | 定位成功率 | 整體實際 FPS | 正常 TRACK FPS | TRACK wall P90 (ms) |
|---|---|---:|---:|---:|---:|
| P017 | LG reference | 92.44% | 12.23 | 20.70 | 71.48 |
| P017 | deep | 93.06% | 12.41 | 19.17 | 79.74 |
| P017 | NeuFlow 100/100 | 94.03% | 15.46 | 42.57 | 64.02 |
| P017 | projection | 92.97% | 9.49 | 16.49 | 159.29 |
| P024 | LG reference | 99.91% | 24.55 | 25.41 | 45.36 |
| P024 | deep | 100.00% | 33.56 | 37.07 | 42.05 |
| P024 | NeuFlow 100/100 | 100.00% | 42.38 | 51.24 | 34.39 |
| P024 | projection | 100.00% | 18.80 | 20.03 | 64.75 |
| P072 | LG reference | 99.25% | 21.85 | 24.40 | 49.54 |
| P072 | deep | 99.29% | 25.94 | 32.45 | 54.07 |
| P072 | NeuFlow 100/100 | 99.31% | 32.05 | 46.24 | 45.85 |
| P072 | projection | 99.28% | 17.28 | 19.50 | 65.97 |
| P121 | LG reference | 98.28% | 12.68 | 16.65 | 106.07 |
| P121 | deep | 98.19% | 12.46 | 15.82 | 121.68 |
| P121 | NeuFlow 100/100 | 98.36% | 16.69 | 22.43 | 104.85 |
| P121 | NeuFlow 80/60 | 99.09% | 20.18 | 49.91 | 98.46 |
| P121 | projection | 98.39% | 9.90 | 15.09 | 231.34 |

成功率上升不代表軌跡更準。例如 P121 NeuFlow 80/60 成功率最高，但它與參考軌跡的大幅偏差也最多。

## 相對 LG reference 的軌跡差

| 影片 | 候選 | 絕對位置差 P95 (m) | yaw 差 P95 (deg) | 幀間位移差 P95 (m) | 二階位置差 P95 (m) | >0.5 m | >1.0 m |
|---|---|---:|---:|---:|---:|---:|---:|
| P017 | deep | 0.109 | 1.459 | 0.101 | 0.181 | 68 | 0 |
| P017 | NeuFlow 100/100 | 0.298 | 2.784 | 0.267 | 0.445 | 88 | 3 |
| P017 | projection | 0.289 | 2.691 | 0.267 | 0.425 | 67 | 0 |
| P024 | deep | 0.173 | 2.807 | 0.149 | 0.253 | 0 | 0 |
| P024 | NeuFlow 100/100 | 0.177 | 4.286 | 0.134 | 0.222 | 0 | 0 |
| P024 | projection | 0.146 | 3.346 | 0.145 | 0.238 | 0 | 0 |
| P072 | deep | 0.298 | 1.759 | 0.197 | 0.344 | 121 | 0 |
| P072 | NeuFlow 100/100 | 0.313 | 1.478 | 0.171 | 0.278 | 134 | 0 |
| P072 | projection | 0.266 | 1.411 | 0.173 | 0.277 | 8 | 0 |
| P121 | deep | 0.130 | 1.571 | 0.126 | 0.211 | 32 | 2 |
| P121 | NeuFlow 100/100 | 0.441 | 4.258 | 0.450 | 0.739 | 217 | 21 |
| P121 | NeuFlow 80/60 | 0.473 | 4.814 | 0.466 | 0.721 | 266 | 18 |
| P121 | projection | 0.413 | 4.864 | 0.422 | 0.680 | 180 | 15 |

「二階位置差」是此次的抖動 proxy，數值越大表示候選軌跡的幀間加速變化與參考軌跡越不一致。NeuFlow 在 P024/P072 可以降低這個數值，但在困難 P121 上從 deep 的 0.211 m 惡化到 0.739 m。因此光流有機會讓容易區段更快或更平滑，但本次結果不支持它可同時解決跳動並不降精度。

## 熱降頻與功率觀察

全組測試共記錄 3,726 筆、7,450 秒：

- CPU 最高 98 degC；最熱核心的 thermal-throttle counter time 增加 87.1 s，package counter time 最大增加 106.1 s。
- GPU 最高 71 degC，0 筆 NVIDIA thermal-slowdown active；但 power-limit flag 有 2,162 筆 active，表示 GPU 常受筆電功率上限約束，不是 GPU 熱降頻。
- 另外的 P121 NeuFlow 100/100 標準化重跑中，CPU/GPU 最高為 96/69 degC，CPU 核心/package throttle time 約增加 9.7/13.1 s。
- 因此上表是長時間負載下的實際結果，不是短時冷機 peak FPS。CPU 散熱是現有 throughput 的實際限制之一。

## Production 決策

1. 預設保留 `deep_nn_then_lg_3_5`，不啟用 NeuFlow、LK 或 projection-guided 直接發布位姿。
2. NeuFlow 可保留為實驗性 performance mode，但要求獨立 pose cross-check 並在不一致時回退 deep；本次數據不允許它成為 precision-preserving 預設路徑。
3. 若目標是困難場景穩定 >23 FPS，下一個優先項應是降低 deep fallback/PnP 的尾延遲與 CPU 熱降頻，而不是放鬆光流門檻。
4. 其他方案在沒有非衛星絕對 ground truth 前，只能聲明「與全量 LG 參考一致」，不能聲明絕對精度提升。

## 可重現資料

- 原始每幀 benchmark JSON、camera PLY、比較 JSON 與溫度 JSONL：`report_20260714/`
- 參考比較程式：`compare_pose_reference.py`
- 串流 benchmark 入口：`benchmark_production_stream.py`
- 測試：validation test suite 74 passed；新增標準化重跑後的資料格式不需要改動程式。

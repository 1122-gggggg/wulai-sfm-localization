# 定位效能與品質驗證（2026-07-13）

> **2026-07-14 更新：本文的 production 選擇已被推翻。** 操作者依準度決定**移除 mutual-NN 快路**，
> TRACK/WEAK 改為每幀跑 XFeat + LighterGlue adaptive 3→5（即本文的「LG reference」路徑本身）。
> temporal cache 只在 NN 分支被查詢，故現已失效。詳見 `OPTIMIZATION_CHANGES_LOG.md` §十一。
> 本文以下的 `deep_nn_then_lg_3_5` 相關結論僅作為歷史量測保留。


## 結論

正式設定應使用 **XFeat TRACK top-K 1700 + 官方 PyTorch adaptive LighterGlue + NN fast accept + temporal cache + 光流關閉**。使用這份地圖與驗證序列時，提案的 TRACK top-K 1300 不是最佳值；固定隨機種子的 P124 中，1700 為 444/723，1300 為 426/723。舊的四序列探索性資料也是 1700 多 54 個成功幀。

光流可保留為預設關閉的實驗路徑，但現在不應納入 production default。TensorRT FP16 的 matcher 很快，但目前也未通過逐序列 no-loss 品質門。

在不改變 matcher/PnP 數值路徑的前提下，本次採用提前載入 LighterGlue、快取 LighterGlue `image_size` GPU tensor、合併 match-index D2H，以及延後 query-keypoint D2H 到確實有有效 3D correspondence 時才執行。搭配四執行緒的持續效能預算，固定 seed 四序列 1,961 幀的成功、狀態、inliers、reprojection 與 pose 逐幀完全一致；前一輪合併 actual FPS 由 7.826 提升到 8.181（+4.54%）。最後一項 lazy D2H 又以 P123 前 20 幀逐幀比對，所有 success/state/inliers/reprojection/pose 皆無差異。

## 測試條件與限制

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU。
- 輸入：P024/P121/P123/P124，共 1,961 幀，實際模型輸入縮放為 1280x720。
- `actual FPS` 包含從圖片目錄讀取與完整定位流程，不是只測單一 matcher kernel。
- 熱測同步記錄 Linux thermal zones、CPU frequency、CPU core/package throttle counters、GPU clock/power 與 NVIDIA clock event reasons；累積 throttle counter 以首末樣本差值判定。
- 資料包沒有 GT pose，因此品質只能用成功幀、inliers、reprojection RMS 與和 baseline 的成對姿態差來把關。
- 沒有原始影片 timestamp，無法從這批資料推導 capture-to-pose latency。
- 舊的四序列候選實驗未固定 pycolmap RANSAC seed，只能當探索性證據。本次已加入 `--pnp-ransac-random-seed` 與 `--random-seed`；P123 重跑在排除計時欄位後為 bit-exact，但 FPS 有約 1.5% 執行波動。

## 目前 production baseline

這是本次實驗前的 top-K 1700、PyTorch adaptive matcher、光流關閉數據。

| 序列 | 成功幀 | actual FPS | reproj RMS median | inliers median |
|---|---:|---:|---:|---:|
| P024 | 328/328 | 19.997 | 2.720 px | 478 |
| P121 | 638/641 | 14.036 | 3.015 px | 147 |
| P123 | 236/269 | 10.233 | 3.115 px | 138 |
| P124 | 476/723 | 5.018 | 2.954 px | 95 |
| 合併 | 1,678/1,961 (85.569%) | 8.436 | - | - |

固定 seed=0 的現行組合結果為 P024 19.668 FPS、P121 13.815 FPS、P123 9.533 FPS、P124 4.539 FPS。P124 成功幀為 444，和舊次數 476 差異很大，顯示此序列對 RANSAC/狀態軌跡敏感；候選方案必須使用相同 seed 比較。

## FP32 exact runtime 與持續效能結果

下表的 baseline 與 candidate 都是固定 seed=0。Candidate 保持 PyTorch FP32/Flash LighterGlue，不啟用 AMP、ONNX、TensorRT 或光流。每個序列的 frame-equivalence gate 均為 0 mismatch，pose/reprojection 最大差異均為 0。

| 序列 | 成功幀 | baseline actual FPS | candidate actual FPS | wall p50 改善 | wall p90 改善 |
|---|---:|---:|---:|---:|---:|
| P024 | 328/328 | 19.668 | 18.850 | 2.76% | 0.56% |
| P121 | 639/641 | 13.815 | 13.815 | 4.58% | 4.38% |
| P123 | 236/269 | 9.533 | 10.037 | 7.26% | 9.77% |
| P124 | 444/723 | 4.539 | 4.849 | 6.65% | 8.29% |
| 合併 | 1,647/1,961 | 7.826 | 8.181 | - | - |

P024 的 core p50/p90 都改善，但 `actual FPS` 被圖片 I/O 與啟動波動蓋過；因此不能用單一序列的含 I/O FPS 否定已成對改善的 core latency。四序列合計 wall time 為 250.586 -> 239.702 秒。

P124 的拆分 A/B：

- 只加入 image-size cache / deferred D2H：wall p50 96.19 -> 93.25 ms（+3.1%），p90 448.56 -> 444.72 ms（+0.9%）。
- 再加入四執行緒預算：wall p50 89.80 ms，p90 411.38 ms，actual FPS 4.849。
- 二執行緒：actual FPS 4.858，只有 +0.18% 噪聲；p50 反而慢 0.4%，CPU package throttle time 也比四執行緒高，因此不採用。

P124 持續熱測：

| CPU 執行緒 | actual FPS | CPU 最高溫 | package throttle time 增量 | GPU thermal slowdown |
|---:|---:|---:|---:|---:|
| 原系統預設 | 4.544 | 96°C | 6,192 ms | 0 |
| 4（採用） | 4.849 | 96°C | 1,736 ms | 0 |
| 2（不採用） | 4.858 | 96°C | 2,881 ms | 0 |

四執行緒不是省電模式；系統仍維持 performance profile 與 GameMode。它是這台 i7-13620H/RTX 5060 Laptop 的最高「可持續」組合：相較原本 10/16-thread library defaults，FPS +6.7%、CPU thermal-throttle time -72%，同時讓更多共享功熱預算留給 GPU。OpenCV 與 PyTorch 官方都提供 thread-count 控制，因此 production launcher 設 `OPENCV_FOR_THREADS_NUM/OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS=4`，可用 `SFM_CPU_THREADS` 覆寫。

原始資料位於 `outputs/exact_latency_20260713/`，其中包含四序列 JSON、成對 compare verdict 與逐秒 hardware JSONL。

## 真實 ANAFI 地面串流

以 direct Wi-Fi 720p30、YOLO 關閉、無起飛的 live UI 做三種狀態測速。畫面不在這份烏來地圖場域，因此所有幀皆為 `success=false`；下表只證明真實相機、Pdraw、UI 與 worker 串接後的速度，不是定位精度證據。為避免無線串流偶發停頓扭曲 A/B，舊版與新版使用相同的前段樣本數比較。

| 模式 | 樣本 | 舊 observed FPS | 新 observed FPS | FPS 變化 | core p50 舊 -> 新 |
|---|---:|---:|---:|---:|---:|
| BOOT_INIT / global path | 66 | 2.420 | 2.719 | +12.37% | 393.16 -> 357.72 ms（-9.01%） |
| WEAK_TRACK | 120 | 10.243 | 11.334 | +10.66% | 80.88 -> 76.52 ms（-5.39%） |
| TRACK | 174 | 14.480 | 16.730 | +15.54% | 52.32 -> 51.12 ms（-2.30%） |

TRACK 另跑 621 秒熱穩態，和硬體監控結束時刻切齊 7,728 幀：全區間 observed 15.085 FPS，core p50/p95 為 52.16/61.03 ms；最初 174 幀為 16.730 FPS、51.12/53.59 ms，最後 174 幀為 14.839 FPS、53.09/60.14 ms。CPU 平均/p90/最高 79.2/89/98°C，package throttle time 增加 1,064 ms；GPU 平均/最高 68.0/72°C，622 個樣本皆無 thermal slowdown。這表示 CPU 有短暫降頻但不是持續主要瓶頸，GPU 沒有熱降頻。

測試中曾有一段 source age 約 300--360 ms、source gap 尾端超過 1 秒的 Pdraw/transport backlog；該段 core latency 仍約 53--60 ms，CPU/GPU 也沒有同步熱降頻，不能把端到端 FPS 降低誤判為模型變慢。現在 raw callback 已只做 timestamp、`ref()` 與 capacity-1 latest-frame 置換，YUV->RGB/resize 移到專用 worker；覆蓋、flush、stop 與 late callback 的 `unref()` 競態測試為 8/8 通過。結構已完成，但仍需一段沒有 transport stale 的同場景串流才能做乾淨 A/B，所以目前只宣稱排隊結構改善，不虛報真實串流 FPS 增益。

改動後另做一次 348-frame 地面重啟。最後連續模式樣本為 TRACK 25 幀、16.221 FPS、core p50/p95 55.96/56.85 ms；WEAK 9 幀、10.665 FPS、core p50/p95 86.95/89.17 ms。BOOT/global 路徑期間 Pdraw 仍有間歇性 stale，整段 observed FPS 被 transport gap 拉低，因此這次不能構成乾淨的舊/新 A/B，也不把它寫成 worker FPS 增益。安全紀錄仍是 TakeOff 0、非零 PCMD 0；cleanup 只記錄 `already_landed`，沒有實際飛行。

再把 Pdraw 與完整 stack 分開診斷後，純串流 30 秒為 865 received / 865 enqueued / 865 converted，28.8 FPS；queue、flush、timestamp、decode drop 全為 0，YUV->RGB/resize 平均 0.210 ms、最大 1.870 ms。隨後完整 UI + BOOT/global 定位跑 45 秒，98 幀 observed 2.744 FPS，core p50/p95 357.27/363.81 ms，submit-to-UI p50/p95 368.75/378.80 ms，`stream_lost=0`。因此前一輪 stale 未能重現，不能歸因於新 frame worker；目前 BOOT 的主要成本仍是 MegaLoc + 30-reference LighterGlue，不是 Pdraw 轉換。該輪同樣 TakeOff 0、非零 PCMD 0，結束時狀態為 `already_landed`。

加入可調限制面板後再以同一台 landed ANAFI 重開 production UI：韌體實際
讀回 MaxAltitude=30 m、MaxDistance=100 m、NoFlyOverMaxDistance=ON；150 個
BOOT/global 樣本為 2.735 FPS，core p50/p95 358.44/364.98 ms，submit-to-UI
p50/p95 368.62/378.51 ms。介面保持開啟供操作員切換 BOOT/WEAK/TRACK；
稽核當下 TakeOff 0、非零 PCMD 0，且只存在單一主視窗，沒有獨立 renderer。

安全稽核為 TakeOff 0、非零 PCMD 0、實際 Landing 0；所有週期 PCMD 都是 `[0,0,0,0]`。只有 `SfM Flight Operator - LIVE Olympe` 主視窗，沒有獨立 Olympe/Pdraw renderer。原始資料為 `outputs/flight_logs/loc_metrics_20260713_143411.jsonl` 與 `outputs/exact_latency_20260713/hardware_live_optimized_modes.jsonl`。

## 完整候選結果

下表為舊的四序列探索性跑法。合併 FPS 以 `1961 / Σ actual_wall_s` 計算。

| 候選 | 成功幀 | 成功率 | 合併 FPS | 逐序列 no-loss |
|---|---:|---:|---:|---|
| current, top-K 1700 | 1678 | 85.569% | 8.436 | baseline |
| top-K 1300 | 1624 | 82.815% | 7.946 | FAIL，P124 -54 |
| flow refresh=2 | 1674 | 85.365% | 8.857 | FAIL，0/-1/-2/-1 |
| flow refresh=6 | 1673 | 85.314% | 8.860 | FAIL，0/+1/-1/-5 |
| TensorRT FP16, full matcher | 1670 | 85.161% | 12.756 | FAIL，0/+1/0/-9 |
| TensorRT FP16, acquire-only | 1678 | 85.569% | 9.323 | FAIL，0/0/+2/-2 |
| MegaLoc FP16 | 1640 | 83.631% | 7.508 | FAIL |

固定 seed=0 的 P124 關鍵對照：

| 候選 | 成功幀 | actual FPS | 結果 |
|---|---:|---:|---|
| current top-K 1700 | 444/723 | 4.539 | baseline |
| top-K 1300 | 426/723 | 4.567 | 品質 FAIL |
| flow refresh=2 | 444/723 | 4.759 | 成功幀 PASS，姿態尾端 FAIL |
| TensorRT FP16 acquire-only | 442/723 | 5.760 | 品質 FAIL |

### 本輪補測：光流、TRACK-only TensorRT 與 MegaLoc FP16

以下都使用 current code、seed=0、四執行緒與同一個 P124 723-frame 輸入；
fresh FP32 baseline 為 444/723、actual FPS 4.85，且和既有 exact baseline
逐幀 0 mismatch。

| 候選 | 成功幀 | actual FPS | 關鍵差異 | 判定 |
|---|---:|---:|---|---|
| Flow r1 / FB 1.0 | 444 | 5.06 | pose max 30.81 cm / 6.51° | FAIL |
| Flow r1 / FB 0.5 | 444 | 5.09 | pose max 30.81 cm / 6.51° | FAIL |
| Flow r2 / FB 0.5 | 444 | 5.18 | pose max 30.72 cm / 6.49° | FAIL |
| Flow strict r1 / FB 0.35 | 444 | 5.05 | pose max 30.81 cm / 6.51° | FAIL |
| MegaLoc FP16 | 445 | 5.04 | 776 mismatch，max pose component 0.672 m | FAIL |
| TRT FP16 TRACK-only | 441 | 4.90 | 1,150 mismatch，LOST +1，p90 慢 1.3% | FAIL |

光流四組的 success mask 都和 baseline 相同，且可增加約 4--7% actual FPS；
但更嚴格的 FB/inlier/RMS/tracked-point gate 仍無法濾掉 coherent drift，故維持
production OFF。MegaLoc FP16 把 global VPR median 30.19 -> 19.13 ms
（快 36.6%），但改變 top-30 排序與後續狀態軌跡。TRACK-only TensorRT
雖讓 wall p50 快 8.6%，卻少 3 個成功幀且 wall p90 變慢，因此也維持 OFF。

20 分鐘、2,400 筆 0.5 秒硬體取樣顯示 CPU p50/p95/max 為
84/90/97°C，package throttle time 增加 7,689 ms（約佔觀測時間 0.64%）；
GPU p50/p95/max 為 65/69/72°C，thermal slowdown 0。這些 A/B 使用交錯且
固定執行緒，但高 CPU 溫度仍是實機長跑應持續監控的限制。

原始 JSON、compare verdict、PLY 與 hardware JSONL 位於
`outputs/optimization_trials_20260713/`。

## LightGlue / ONNX / TensorRT

上游 [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) 的標準匯出路徑主要支援 SuperPoint/DISK，不是這個地圖的 XFeat；而且這裡的 query/reference 特徵數不一定相同。本次新增了 XFeat 專用、可用不同 M/N shape 的實驗 harness，並使用和 [XFeat](https://github.com/verlab/accelerated_features) / Kornia 一致的 isotropic keypoint normalization。上游 LightGlue-ONNX 原始測試為 1 passed。

| Matcher core | 1300 | 1700 | 2048 | 品質摘要 |
|---|---:|---:|---:|---|
| 現行 PyTorch adaptive | 7.03 ms | 7.76 ms | 8.73 ms | production reference |
| ORT CUDA FP32 static | 10.10 ms | 11.58 ms | 13.26 ms | 比 adaptive PyTorch 慢 |
| ORT graph FP16 static | 7.51 ms | - | - | 沒有足夠端到端優勢 |
| TensorRT FP16 static | 2.803 ms | 3.543 ms | 3.675 ms | 快，但完整定位會少成功幀 |
| TensorRT FP8 | 2.548 ms | - | - | match Jaccard 約 0.043，不可用 |
| TensorRT INT8 | 3.449 ms | - | - | Jaccard 約 0.687，且更慢 |
| TensorRT INT4 | - | - | - | TensorRT 10.16/10.9 因 scale mode 無法建 engine |

TensorRT FP16 是「單一 matcher kernel 最快」，不是「最佳定位組合」。現行 [LightGlue](https://github.com/cvg/LightGlue) adaptive depth/width pruning 在真實資料上讓 PyTorch 比 ORT FP32 static 更快，而 TensorRT static 的少量 match 變化會經狀態機放大。

## 光流判斷

現有 KLT 光流不需 FP16，CPU 中位數約 1.23 ms（150 點雙向 LK）。本次已修正實驗路徑的安全性：

- 只能在 `TRACK` 使用，進入 WEAK/LOST 會清除 cache。
- 種子點先做 unique 2D/3D，避免重複 correspondence 膨脹 inlier。
- 最低 seed 改為 100，和 tracked-point gate 一致。
- 光流失敗、低信心或跳變會在同一幀回退 deep localization，不浪費一幀。

舊一輪固定 seed=0 的 P124 refresh=2 和 baseline 成功 mask 完全相同，actual FPS 提升 4.86%。但 444 個共同成功幀的差異為：

- 平移 p90/p95/max：4.20/6.69/21.43 cm。
- yaw p90/p95/max：0.379/0.836/5.557°。
- 146 個直接 flow 幀的 inlier median 由 265.5 降到 134。

因為沒有 GT，無法證明這些尾端誤差是變好或變差。只用 4.86% 收益不足以承擔 21 cm/5.6° 的安全不確定性，所以預設仍關閉。

## FP16 與 tensor 加速建議

- bundle metadata 標示原始 reference descriptors 為 FP16-derived，但現行 loader 與 LighterGlue 會以 FP32 運算；不可把 matcher descriptor 常駐精度直接降成 FP16，除非重新通過逐序列品質門。
- XFeat autocast FP16：4.343 -> 4.323 ms，無實質收益，且 keypoint 已有小差異；不啟用。
- MegaLoc FP16 純模型測試 28.55 -> 16.64 ms，descriptor cosine 0.999959；但 fresh fixed-seed P124 完整狀態機已確認 444 -> 445 且 776 個逐幀不等價欄位，因此現以 `SFM_MEGALOC_FP16=1` 留作預設關閉實驗開關。
- LighterGlue TensorRT FP16：保留在 validation harness，不改 production default。
- cosine NN、PnP/RANSAC、reprojection gate 維持 FP32/CPU；這些是品質邊界，轉 FP16 得不償失。

最後也實測了「快取每個 reference 的 normalized descriptor」。五個 reference 的 synthetic NN 核心 p50 從 1.120 降到 1.028 ms，只省 0.092 ms；P123 269 幀逐幀輸出完全一致，但完整 wall p50 反而受噪聲影響由 43.88 變 44.43 ms，p90 僅由 347.83 變 342.24 ms。若和 700-reference GPU LRU 綁定，最壞還會增加約 367 MB VRAM。收益不足以抵銷 OOM headroom，production 已回退此候選。

## 2602.08430v2 的適用性

[arXiv 2602.08430v2](https://arxiv.org/pdf/2602.08430v2) 是特徵偵測/訓練研究，不是現有 XFeat bundle 的 drop-in runtime backend。論文最有潛力的 zero-shot 組合是 XFeat keypoints + ALIKED descriptors + ALIKED LightGlue，但現有 bundle 只有 XFeat reference descriptors/3D anchors，參考圖像也不在轉移包內。要測這個組合必須取回 reference images，重算 descriptors/對應並重建地圖 cache；不可把不同 descriptor 直接塞進現有 XFeat 地圖。

論文 Table 2 的 zero-shot 數字支持將此列為未來「重建地圖」分支，但不足以取代本次已在真實序列上驗證的 production 組合。

## Kalman filter 判斷

目前 tracker 已使用最近兩個成功 center 做常速度外推，預測值只用來選 local/covis refs 並執行 2 m jump gate；因此 Kalman 不會加速 XFeat/LighterGlue 本身。只把輸出 pose 做低通/Kalman 平滑會改姿態並加入相位延遲，現在不應直接進控制輸出。

真實 TRACK 串流被定位器取用的 capture 間隔不是固定一步：穩定區間約為 p50 66.7 ms、p95 100 ms，且 transport backlog 時會更長。因此第一個實驗不必直接上完整 Kalman；先用兩個 pose 的 capture timestamp 算速度，再按目前幀的 `dt` 做 constant-velocity extrapolation，已能處理目前固定一步外推的主要缺點。

若之後測 Kalman/alpha-beta filter，範圍應限定為「使用 capture timestamp 的 prediction，只取代 candidate center prediction」；原始 PnP pose、inlier/reprojection gate、max-jump gate與狀態轉移保持不變。它可能在掉幀或不等間隔時減少錯選 refs、間接減少掉進 LOST/MegaLoc 的次數，但不會降低單次 XFeat/LighterGlue core latency；同時它會改候選集合，屬於 accuracy A/B，不是 exact runtime 優化。現有離線資料沒有 GT/IMU，尚不足以證明它優於目前簡單外推，因此不進 production default。

## 實作與驗證

- `validation/benchmark_xfeat_lg_onnx.py`：XFeat LighterGlue ONNX 匯出、ORT/TensorRT benchmark、ModelOpt 量化與 experimental drop-in matcher。
- `validation/benchmark_production_stream.py`：固定 seed、ONNX provider 實驗參數，並從本次修正後開始將 flow/MegaLoc runtime env 寫入 JSON。
- production tracker 的 mission/deploy 鏡像已同步，top-K 正式改為 1700，光流仍預設關閉。
- launcher 預設在執行期間使用 performance profile、GameMode、關閉低電量自動 power-saver，並套用驗證過的四執行緒持續效能預算；結束或收到 INT/TERM/HUP 且 UI 完成安全清理後會還原 session 設定。不鎖 GPU 時脈、不提高功耗限制、不停用熱保護。
- `validation/monitor_hardware.py`：CPU/GPU 溫度、頻率、功耗與 thermal-throttle counter 的 JSONL/CSV logger，並對新 CPU throttle counter 與 NVIDIA thermal slowdown 發出節流警告。
- live UI 預設 poll tick 16 -> 10 ms；latest-frame/drop-busy 語意不變，預期平均 phase wait 約少 3 ms。
- Olympe 指令新增控制權回讀、nudge heartbeat deadman、TakeOff/land race epoch、先 Landing/landed 後媒體清理，以及 firmware 高度/距離/GPS/電池 preflight；全程只以 fake drone 驗證，未由 agent 發出真機飛行命令。
- 完整 pytest、鏡像檢查、shell/python 語法檢查及四序列 frame-equivalence gate 均通過。
- CodeGraph 主程式索引：81 files / 2,633 nodes / 5,614 edges，狀態 up to date。

## 本輪指定實驗：FP32 batching、top-2、PnP cap 與 LK refresh=1

本輪使用兩支原始影片的 1280x720 序列解碼全版：P024 為 3,277 幀，
P121 為 6,407 幀。RANSAC 與 NumPy/PyTorch 的 seed 固定為 0。Baseline
為 TRACK top-K 1700、adaptive first top-3/full top-5、NN fast accept、
temporal cache on、correspondence cap 0/0、LK off。

### 最終完整影片 A/B

LK 候選使用 refresh=1、FB 0.5 px、MNN 最少 80 unique-query inliers、
MNN RMS 上限 3.5 px，並要求 LK/MNN 兩個 PnP pose 差異不超過 0.10 m /
3°。MNN 不一致後不得再接受同一個 NN pose，而是在同幀強制走 FP32
LighterGlue；成功 fallback 後會立即重建下一幀 LK 錨點。

| 序列 | 組合 | 成功幀 | actual FPS | wall p50 / p90 | inliers p50 | RMS p50 | 狀態摘要 |
|---|---|---:|---:|---:|---:|---:|---|
| P024 | baseline | 3,277/3,277 | 55.324 | 14.64 / 23.38 ms | 520 | 2.717 px | TRACK 3,275，WEAK 1 |
| P024 | LK + MNN cross-check | 3,277/3,277 | 52.490 | 15.99 / 24.17 ms | 327 | 2.705 px | FLOW 1,580 |
| P121 | baseline | 6,402/6,407 | 17.092 | 55.03 / 94.19 ms | 169 | 3.019 px | WEAK 226，LOST 4 |
| P121 | LK + MNN cross-check | 6,402/6,407 | 16.862 | 55.61 / 92.83 ms | 153 | 3.013 px | FLOW 671，WEAK/LOST 不變 |

兩個序列的 success mask 都和 baseline 完全一致，但 P024 實際吞吐下降
5.12%、p50 延遲增加 9.27%；P121 吞吐下降 1.35%、p50 延遲增加
1.05%。P024/P121 的 inliers median 分別從 520 降至 327，以及從 169
降至 153。P121 的 p90 延遲雖改善 1.44%，仍無法抵銷吞吐與 inlier 下降。
因此 LK 繼續預設關閉。

和 baseline 成對的姿態差異（不是 GT error）為：P024 平移 p50/p95/max
0.66/3.73/22.49 cm，yaw 0.046/1.574/14.738°；P121 平移 p50/p95/max
0/8.58/32.73 cm，yaw 0/1.243/24.153°。這些影片沒有 GT pose，因此
不能用這個差異宣稱 LK 變準或變差。

### 其他三項指定實驗

- **FP32 LighterGlue top-3/top-5 batching**：真正 B>1 在現行 Kornia
  adaptive width pruning 路徑觸發 batch 維度錯誤。改用 CUDA streams
  將各 reference 保持獨立推論時，輸出 bit-equal，但 2/3/5 refs 都沒有
  快於序列執行；production 不加入較慢路徑。
- **adaptive first top-3 -> top-2**：P121 前 200 幀的 actual FPS
  13.497 -> 11.864，wall p50 57.30 -> 81.24 ms；first-stage accept
  124 -> 24，大量幀被迫進 full top-5 retry。正式值回復 top-3。
- **quality + spatial correspondence cap**：metadata 對齊與 used_refs
  殘留問題已修正，同一組 keep indices 會同步套用到 2D/3D、score、ref id
  與 ref keypoint id。但 300/ref、800 total 在 P024 前 100 幀將 median
  inliers 715.5 降至 504.5；P121 前 200 幀 FPS 13.497 -> 13.124。
  功能保留，production 仍使用 0/0（無上限）。

refresh=1 的意思是 deep 定位建立錨點後，最多發布一幀 LK，形成
deep/LK/deep/LK 交替。FB 是 forward-backward consistency：將點從上幀
追到本幀，再追回上幀，往返誤差超過 0.5 px 的點不參與 PnP。

## NeuFlow-v2 refresh=3 完整 P121 實驗

後續將 LK 替換為官方
[NeuFlow-v2](https://github.com/neufieldrobotics/NeuFlow_v2) PyTorch 權重進行另一條
validation-only 實驗。實驗使用 repo commit
`204b5e3744461d90303b9ff82caa7a1bb56a2ca2`，`neuflow_mixed.pth` SHA-256
`76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f`，
FP16 512x288、官方 1 次 s16 + 8 次 s8 refinement。

流程為一幀原本 XFeat/NN/LighterGlue/PnP deep refresh，接著最多兩幀
NeuFlow 傳遞已經 PnP/RANSAC 驗證的 2D<->3D anchors，每幀仍重跑
PnP/RANSAC。第四幀回到 deep refresh，因此 refresh=3 是每三幀一次
deep，不是連續三幀光流。錨點 seed/tracked 門為 80/60，PnP 最少
50 inliers，inlier ratio >=0.25，flow RMS <=4.0 px，並保留原本
5 px RANSAC、6 px TRACK gate 與 2 m prediction jump gate。光流或品質門失敗時
在同一幀回退 deep path。BOOT/LOST 完全不改。

| P121 6,407 幀 | Production baseline | NeuFlow hybrid | 變化 |
|---|---:|---:|---:|
| 成功幀 | 6,402 | 6,407 | +5 |
| actual FPS（含解碼/I/O） | 17.092 | 26.427 | +54.6% |
| wall p50 / p90 | 55.03 / 94.19 ms | 16.29 / 86.23 ms | -70.4% / -8.4% |
| inliers p50 | 169 | 119 | -29.6% |
| RMS p50 | 3.019 px | 2.927 px | -3.0% |
| XFeat extracts / frame | 1.000 | 0.431 | -56.9% |
| LighterGlue calls / frame | 2.909 | 1.514 | -48.0% |

NeuFlow 直接發布 3,644 幀（56.9%），這些幀的 wall p50/p90 為
15.02/17.32 ms，其中 dense-flow GPU p50 7.95 ms、PnP p50 1.91 ms。
本次沒有一幀已啟動的 NeuFlow attempt 觸發 deep fallback；其餘 2,763 幀
為定期 refresh、BOOT 或無足夠 unique anchors 而直接走 deep。全程觀察 GPU
約 61--69 C、SM 約 2.05--2.15 GHz，未觀察到 thermal clock collapse。

這支影片沒有 GT pose。以 baseline 當參考而非當真值，兩者成對 pose
的平移差 p50/p95/max 為 3.65/18.25/106.24 cm，yaw 為
0.222/2.511/21.724 deg。NeuFlow 的 temporal second-difference p50 從
6.74 cm 降為 3.36 cm，yaw 從 0.485 降為 0.263 deg，但「較平滑」
不能在無 GT 時直接解釋為「較準」。median inliers 下降是因光流只保留
已驗證且去重的 anchors，仍是不能忽略的幾何冗餘度下降。

結論：離線證據支持它取代「TRACK 每幀都跑 deep matcher」，但不支持
刪除原本方法。原本 XFeat -> NN/LighterGlue -> PnP 仍必須作為每三幀
refresh、同幀 fallback 與 BOOT/LOST。在沒有 GT/實機靜止與往返軌跡測試前，
維持 validation-only，不直接改 production default。

### 資料品質限制

- 兩支影片都無 GT pose，只能做相對 no-regression 與定位健康代理指標。
- 影片 benchmark 使用 SIMPLE_RADIAL 1280x720
  [934.139423, 640, 360, 0.001061]，和真實 ANAFI production FULL_OPENCV
  校正不同；不可把離線 FPS 當成真實 Wi-Fi capture-to-pose FPS。
- 本輪記錄跑前/跑後 GPU 溫度 45--53°C，但沒有為兩支完整影片建立連續
  CPU/GPU thermal trace；不以這些資料宣稱「無熱降頻」。

## 檔案清理稽核

沒有將「可重建」自動視為「絕對不會再用」。以下為精確分級：

| 類別 | 大小 | 建議 | 理由 |
|---|---:|---|---|
| outputs/regression_20260710/frames/ | 2,722,870,884 B（2.54 GiB） | 可刪，先確認 | 四支 MP4 的 stride-10 JPEG 中間檔；原片仍在，baseline/after JSON 可保留 |
| Sphinx outputs 下 3 份 ticks.jsonl | 269,270,361 B（256.8 MiB） | 可歸檔或刪 | 高密度 simulation tick；summary/trials/RESULTS 可獨立保留 |
| 舊的 720p short/不安全 LK 候選結果 | 28,268,108 B（27.0 MiB） | 報告完成後可刪 | 最終 baseline 與 MNN80 candidate 已另存 |
| ONNX/TensorRT 模型與 engine 21 份 | 100,028,696 B（95.4 MiB） | 若不再跑 ONNX 可刪 | 都是可重建實驗產物；57 份 JSON 數據可保留 |
| .experiment_deps/lightglue_onnx/ | 653,477,696 B（623.2 MiB） | 若實驗結案可刪 | 只含 ONNX/ORT/Polygraphy 實驗依賴，production 虛擬環境不依賴 |
| 外層 localization/.codegraph/codegraph.db | 1,049,452,544 B（1000.8 MiB） | 建議重建而非原樣保留 | 誤將 .venv 等 20,892 檔納入且落後 3,741 檔；定位子專案的 81-file CodeGraph 健康 |

不建議刪除 .venv 7.15 GiB、torch_hub_cache 1.08 GiB、bundles 1.33 GiB、
maps 0.92 GiB 或 flight logs 141 MiB。前四者是目前離線 production/驗證
的執行依賴；flight logs 是真實串流、飛安與延遲稽核證據。

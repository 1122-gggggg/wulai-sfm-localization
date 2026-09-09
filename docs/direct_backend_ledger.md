# direct 後端總帳

最後更新：2026-09-09。硬體基準：NVIDIA GeForce RTX 5060 Laptop GPU，repo venv
（torch 2.11.0+cu128、opencv 4.13.0、pycolmap 4.0.4）。

## 這份文件為什麼存在

`docs/verified_localization_optimization_ledger.md` 與
`docs/localization_optimization_runbook.md` 記錄的是 `sfm_glomap_deploy`（EDM `.pt`
bundle）那條路徑。那條路徑已經整條從 worktree 移除，`localizer_registry` 現在只註冊
`direct` 一個 backend。**舊總帳裡的旗標、閾值與「不要重做」條目對現行程式碼一律不適用**
—— KLT bridge、async tracker、`acquire_relaxed_min_inliers`、CUDA graphs、
`match_batch_size`、兩個否決器，這些旋鈕在 direct 後端都不存在。

新路徑的凍結組態與禁令在
`地圖檔/場域/river_site/releases/*/provenance/P174_AND_NEXT.md`，本檔只記錄
**在 5060 上實測、且已落地**的項目。

准入規則沿用舊總帳：任何項目在通過對應驗證之前，不得寫進本檔、不得改凍結 profile。

## 現行架構的實測輪廓（2026-09-09，改動前）

證據：`outputs/flight_logs/session_20260909T033332Z_simulated-stream_f8d6eec4/`，
P157（未進地圖、未用於選旋鈕的影片），802 幀 / 48.4 s，真機 UI 的模擬串流路徑。

| 指標 | 值 |
|---|---|
| `success` | 797 / 802（99.38%） |
| 狀態分佈 | FAST_TRACK 654、RELOC_SEED 143、NO_POSE 5 |
| `loc_fps` p50 | **16.63**（串流 30 fps；`rejected_submits` 656 / 約 1458） |
| fast loop `total_ms` | p50 6.77 / p90 34.43 / max 144.09 |
| `core_wall_ms` | p50 7.68 / p90 37.28 / p95 48.54 |
| `e2e_submit_to_ui_ms` | p50 36.34 / p95 95.62 / **mean 41.48**（= 1000/41.48 ≈ 24 fps 上限） |
| reloc 週期（RELOC_SEED 間距） | p50 **338 ms**（profile `period_s` = 300 ms，即 worker 永不待機） |
| `vo_inliers` | **0 / 802**（VO 一次都沒有貢獻，見下「目前是惰性的子系統」） |
| `map_inliers` p50 | 462（`live_points` p50 462，`inlier_ratio` p50 0.996） |

`loc_fps` 由 e2e **平均**延遲決定，而平均被尾巴拉高，不是被 p50 決定。

## 1. reloc 的 `lift_reference_matches` 向量化（2026-09-09 落地）

**機制。** fast loop 的停頓與 reloc 週期相位鎖定：

```
距上次 handover   FAST_TRACK total_ms p50
  +  0-100 ms       4.3 - 4.9 ms      （乾淨）
  +150-250 ms      20.6 - 23.5 ms     （5x 停頓）
  +250 ms 之後      5.5 - 7.5 ms      （但 p90 拖到 46-97 ms）
```

停頓發生在 worker 自己的 `TwoRateTracker.step()` 內（`core_wall_ms` 同步惡化），
所以與 UI／IPC 無關。同機微基準指認機制為 **GIL**：

| 主執行緒跑 fast loop，背景執行緒為 | fast loop p50 |
|---|---|
| 無 | 2.24 ms |
| 一條佔住 GIL 的純 Python 執行緒 | **23.06 ms（10.3x）** ← 對上實測的 20-23 ms |
| 一條吃滿 CPU 的 BLAS 執行緒 | 9.96 ms（4.45x） |
| 一條 torch CUDA 執行緒 | 2.36 ms（1.05x） |

另外實測 **pycolmap 與 cv2 都會釋放 GIL**，torch CUDA 亦然，所以持有 GIL 的只可能是
純 Python。cProfile 指認 `vendor/river_map_quality/official_edm_adapter.py`
的 `lift_reference_matches` 為 tottime 第一名：每次 lift 對約 20,000 個觀測建 Python
dict 空間格，再對每個 EDM match 跑 Python 迴圈 + 逐點 numpy 微呼叫，單次 reloc
約 **478,720 次 `math.floor`**。

**改動。** 改為一次算完所有 match 的向量化鄰域搜尋
（`_nearest_observation_within` / `_dense_rank`）。半徑（cell 邊長 = 搜尋半徑，
所以 3×3 鄰域必定完整覆蓋圓）、admission 規則、`(distance, Point3D ID)` 併發序
全部不變。cell key 改用 dense rank 編碼，上界 `len(observations)²`，不會溢位。

**等價性。** 四組真實 reloc 輸入逐欄位比對，`point3d_id` 與 `lift_distance_px`
完全相等（atol=0）：

| obs | matches | 原版 lifted | 向量化 lifted |
|---|---|---|---|
| 19912 | 845 | 717 | 717 |
| 19493 | 476 | 390 | 390 |
| 19912 | 915 | 618 | 618 |
| 19493 | 595 | 398 | 398 |

**驗收（真實 provider，26 幀 P157）：**

```
scalar oracle       localize_array p50 = 254.1 ms   STRONG 26/26
landed vectorised   localize_array p50 = 174.1 ms   STRONG 26/26
逐欄位相同（status / inliers / inlier_ratio / reproj_p90 /
            anchor point3d_id 全序列 / cam_from_world 原始位元組）
-> -80.0 ms / -31.5%
```

**格子快取被否決。** 把每個 reference 的空間格快取起來，單段再快 2.5 倍
（1.58 → 0.63 ms），但端到端只多買到 **3.1 ms（79.2 中的 3.1）**。為了 4% 的收益
把可變狀態塞進 vendor 邊界不划算，且會引入陳舊快取的正確性風險。**不要再排。**

**回歸。** `tests/localization/deploy/test_lift_reference_matches.py` 把改版前的純量
實作留作 oracle 做差分比對（38 tests）。變異測試：拿掉 point_id 併發序 → 9 失敗、
`<=` 改 `<` → 1 失敗、floor 改截斷 → 18 失敗、鄰域縮成 3 格 → 25 失敗。
stable sort 改 quicksort **抓不到**，這是預期的：cell 內順序只在距離與 point_id
同時打平時才用得到，而那時輸出本來就相同，所以不可觀測。

**未還的債。** 這是 vendored 檔的本地 patch，記在
`VENDOR_PROVENANCE.json.local_patches`（含 `upstream_sha256` / `patched_sha256`）。
下次 re-vendor 是整包換，沒有先回上游就會被蓋掉。

## 2. BoQ 檢索改吃彩色（2026-09-09 落地）

`live_provider.localize_array` 一直有 `color_bgr` 參數，docstring 也寫明凍結的 BoQ
參考庫是從**彩色** keyframe 抽的；但唯一的生產呼叫點傳的是灰階，被複製成三通道。
彩色幀在 `TwoRateTracker.step()` 裡本來就在手上。

實測 40 幀：

| | ok | STRONG | inliers p50 | reproj_p90 中位 |
|---|---|---|---|---|
| 灰階（改動前） | 40/40 | 40 | 1976 | 1.181 |
| 彩色（現行） | 40/40 | 40 | **2017** | 1.175 |

top-1 參考有 3/40 不同，top-2 集合有 **13/40（32.5%）** 不同。檢索確實被改變，
inliers 一致小幅上升約 2%，但**在這段覆蓋良好的影片上不改變成敗**。
定位為「免費、方向正確、但不是救命」的修正；價值在檢索邊緣的場景。

交接時對彩色幀取快照（`gray` 一定是新配置的 `cvtColor` 輸出，彩色幀在不需要
resize 時卻是呼叫端的陣列），回歸見 `tests/localization/deploy/test_reloc_worker_handover.py`。

## 3. dead-reckon 的 yaw 守衛改用重力對齊基底（2026-09-09 落地）

守衛的用途是：DR 用最近步長外插（假設局部等速），單幀航向大跳就把步長砍半。
但它原本用 `atan2(forward[1], forward[0])` —— 而本場域的重力沿 **+Y**
（`T_align_gravity.json`：`gravity_glomap ≈ [-0.011, 0.997, 0.082]`），水平面是
X-Z。那個角度因此橫跨一個**垂直**平面。用場域自己的重力實測，兩個方向都錯：

| 情境 | 真實航向變化 | 舊慣例回報 | 結果 |
|---|---|---|---|
| 俯角 0-10°（正常巡檢姿態）轉彎 30° | 30° | **1.0 - 3.9°** | 低於 5° 門檻，**漏報** |
| 俯角 5° → 10°，航向不變 | 0° | **5.3°** | 超過門檻，**誤報** |
| 俯角 15° → 45°，航向不變 | 0° | 31.6° | 誤報 |

它量的實質上是雲台俯角。改為 `azimuth_turn_exceeds()`，用 `MapFrame` 的實測水平
基底；沒有 `MapFrame` 時**不開火**（沒有可辯護的水平基底時，用錯的量開火比不開火差）。
`map_frame` 由 `DirectTrackerAdapter` 的 property 轉發到 tracker，因為
`live_localizer_worker` 會在建構後重新指派它，全系統只有一份。

**影響是潛伏的，不是現行損害**：P174 與 2026-09-09 的 P157 session，`DEAD_RECKON`
發生次數都是 **0**。回歸見 `tests/localization/deploy/test_dead_reckon_yaw_guard.py`。

發布出去的 `pose.yaw` 一直是對的：adapter 在 `map_frame` 存在時用
`map_frame.heading()` 覆蓋，而 `map_align` 確實有接上。缺陷只在守衛內部。

## 4. `reloc.period_s` 實測後維持 0.3（2026-09-09，**不改**）

`P174_AND_NEXT.md` 的「准做 #2」要求在本機用實測 reloc median 設 `RELOC_PERIOD_S`
（`period >= 1.0 * median_s`）。改動前這條沒有滿足：worker 實際週期 338 ms > 300 ms，
`period_s` 等於失效、worker 永不待機。

**但第 1 項已經把前提解掉了。** 620 幀 P157、30 fps 步調、真實 two-rate tracker：

| `period_s` | reloc_ms p50 | reloc_ms p90 | 實際週期 p50 | worker busy | loop p50 | loop p90 | 覆蓋 | handover 掉包 |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 150.7 | 191.8 | 200 ms | 78.1% | 7.28 | 24.62 | 100% | 0 |
| 0.25 | 151.1 | 176.3 | 267 ms | 65.2% | 6.53 | 25.57 | 100% | 0 |
| **0.30（現行）** | **161.4** | 194.1 | 333 ms | **54.8%** | 3.69 | 23.36 | 100% | 0 |
| 0.35 | 174.9 | 196.8 | 367 ms | 51.4% | 4.25 | 33.12 | 100% | 0 |
| 0.45 | 177.0 | 199.4 | 467 ms | 41.0% | 3.93 | 32.53 | 100% | 0 |

reloc median 已降到 **150-177 ms**，現行 0.3 對 median 有 1.9 倍、對 p90 有 1.5 倍
餘裕，worker duty cycle 從約 100% 降到 **54.8%**。P174 的規則現在**不改就已滿足**。

**決定：維持 0.3，不動 SHA 鏈。** 往下調（0.25 / 0.2）能換到更密的地圖修正
（每秒 3.7 / 5.0 次 vs 3.0 次），但：覆蓋率在五個設定上都是 100%，量不出差別；
loop p50/p90 的差異來自同一行程內依序跑的五個臂，落在順序／熱噪聲內，不可信；
而「為了更密的修正而調整」是**準度**取捨，`P174_AND_NEXT.md:93` 明文禁止拿
P167/P173/P174 這類已經反覆量測過的影片來做。要往下調，證據必須來自一條全新航線
的驗收飛行。

**不要再排**：拿 P157 掃 `period_s` 求覆蓋率最佳值。覆蓋率在此已飽和，掃不出東西。

## 目前是惰性的子系統（不是死碼，但這條路上沒開火）

- **VO**：`vo_inliers` 在 802 幀上全為 0。`vo.min_live = 350`，但 handover 每
  333 ms 就把 live 點整組換成約 480 個地圖點，所以 live 點從來沒有掉到 350 以下，
  VO 從來沒有種子過。P174 上 VO 撐了 230 幀，所以它是地圖點變稀時的 fallback。
- **dead reckoning**：P174 與 P157 session 都是 0 次。

兩條 fallback 的正確性在這兩段影片上完全未被行使。任何依賴它們的結論都缺乏證據。

## 已知缺口（不是優化，是護欄沒蓋到）

- `tools/check_maintainability.py` 的 `SCAN_PATHS` 只有
  `定位演算法/deploy_code/sfm_glomap_deploy`，**沒有 `sfm_direct_deploy`**。整條現行
  production 樹（fast loop、provider、vendor）不在複雜度 ratchet 之內，而 ratchet
  守的是已經被刪掉的那棵樹。納入需要先重新定基準。
- `river_gluemap_all8_direct_20260908` 的 quality receipt 是 `passed: false`、
  `validation: NONE`、`absolute_ground_truth: NONE`。本檔所有數字都是**相對**比較
  （同輸入前後對照），沒有一項是絕對精度。

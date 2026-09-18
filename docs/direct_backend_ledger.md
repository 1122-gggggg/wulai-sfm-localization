# direct 後端總帳

最後更新：2026-09-17。硬體基準：NVIDIA GeForce RTX 5060 Laptop GPU，repo venv
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

## 測試影片身分（先看這個，再看任何數字）

`localization/boq_references_sideview.names.json` 的 session id 就是**來源影片
SHA-256 的前 16 碼**，所以影片是否在地圖裡可以直接驗，不必靠記憶：

| 影片 | session id | 在地圖裡？ |
|---|---|---|
| P157（`模擬器/測試影片/P1570157.MP4`） | `vid_931611d6ff4f0361` | **是**，159 張關鍵幀（地圖第二大 session） |
| P167 | `vid_349c83c4bf56a785` | 是，100 張 |
| **P173**（`/home/allen/下載/P1730173.MP4`） | `vid_15425ad23062e87f` | **否 — 真 holdout** |
| **P174**（`/home/allen/下載/P1740174.MP4`） | `vid_16f53f8d52f9e357` | **否 — 真 holdout** |

地圖 `river_gluemap_all8` 共 8 個 session，941 張參考影像。

**准入規則（2026-09-09 新增，由一次判斷錯誤換來）**：本檔第一版把 P157 當成
holdout 來報數字。它在地圖裡，所以它對**檢索品質**與**飢餓 fallback**（VO、dead
reckoning）都沒有鑑別力 —— 地圖含有該場景幾何，供點永遠充足，觸發條件不會出現。
任何關於精度、檢索、`min_live` / `track_cap` / handover 門檻的結論，
**必須在 P173 / P174 上取得**。P157 只能用來量與資料無關的東西（例如逐位元等價性、
kernel 延遲）。

## In-sample 的實測輪廓（2026-09-09，改動前）

證據：`outputs/flight_logs/session_20260909T033332Z_simulated-stream_f8d6eec4/`，
**P157（在地圖裡，見上表）**，802 幀 / 48.4 s，真機 UI 的模擬串流路徑。
下面的高分是系統在讀自己的答案，**不是泛化能力**；保留它是為了記錄主機側延遲鏈與
GIL 停頓的機制，那部分與 in/out-of-sample 無關。

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

**Holdout 驗收（2026-09-09，兩部 holdout 各兩輪交錯 B,A,B,A）。** 上面的 254→174 是在
P157（in-sample）上量的；延遲與是否 in-sample 無關，但既然有 holdout 就用 holdout 報：

| 影片 | before reloc p50 | after reloc p50 | 改善 |
|---|---|---|---|
| P174 輪 1 | 207.0 ms | 184.1 ms | −11.1% |
| P174 輪 2 | 192.2 ms | 175.6 ms | −8.6% |
| P173 輪 1 | 226.9 ms | 190.1 ms | −16.2% |
| P173 輪 2 | 229.4 ms | 179.7 ms | −21.7% |

**4 組配對全部改善。** 覆蓋率在同一批跑次上是噪聲主導（P173 同臂自身在 96.90% 與
100.00% 之間擺盪），本項不宣稱任何精度效果 —— 它逐位元相同，**不可能**有精度效果。

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

## 2. BoQ 檢索改吃彩色 —— **NOT GO，已退回**（2026-09-09）

`live_provider.localize_array` 有 `color_bgr` 參數，docstring 也寫明凍結的 BoQ 參考庫
是從**彩色** keyframe 抽的，而生產呼叫點傳的是灰階複製三通道。這個分佈不匹配是真的，
彩色幀在 `TwoRateTracker.step()` 裡也本來就在手上 —— 但**交過去量不到好處**。

P173（真 holdout）五輪配對，兩臂都用向量化 lift，只差彩色：

| 指標 | grey | colour | 配對差值 |
|---|---|---|---|
| 覆蓋率 | 97.68% ± 1.35 | 96.23% ± 1.38 | **−1.46%**（5 組中 4 組為負） |
| 地圖確認幀（FAST_TRACK+RELOC_SEED） | 69.76% ± 2.23 | 69.89% ± 2.53 | +0.13% |

配對 t 檢定：覆蓋率 t=−1.63 / p≈0.18，地圖確認幀 p≈0.94。**都不顯著**，但覆蓋率方向
一致為負，grey 拿到唯一一次 100%，colour 五輪從未超過 97.0%。colour 的
`DEAD_RECKON` 平均 73.0（grey 53.8）、`NO_POSE` 平均 109.4（grey 67.2）。

P174 上兩臂覆蓋率都是 100%，天花板效應，無訊號。

**曾經支持它的證據已作廢**：最早的「40 幀 inliers p50 1976→2017（+2%）」是在 **P157**
上量的，而 P157 在地圖裡。

**決定：退回灰階。** 程式碼在 submit 處留了註解記錄這次量測。
**不要再排**：沒有一次夠力的 gate（多輪配對 + holdout）之前，不要再把彩色接上去。
分佈不匹配仍然存在，它只是還沒被證明有害或有益。

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

**但第 1 項已經把前提解掉了。** 620 幀 **P157（in-sample）**、30 fps 步調、真實
two-rate tracker。注意覆蓋率欄在此**沒有份量**：in-sample 上它必然飽和在 100%，
所以這張表只能用來讀 reloc 延遲與 duty cycle。

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

**注意上表是 P157 的數字。** 驗收影片上 reloc 較重：P174 reloc p50 175 - 207 ms、
P173 190 - 229 ms（皆為 2026-09-09 兩輪交錯的實測）。0.3 對 P173 的上界仍有約 1.3 倍
餘裕，但沒有 P157 上那麼寬。若日後 reloc 變慢，這裡是第一個要重量的地方。

## 7. `reloc.period_s` 0.3 → 1.0（2026-09-11 落地；§4 的重衡量，機器換了）

§4 留的觸發條件成立了：搬到 RTX 5060 筆電本機後，驗收影片 reloc 變慢到
**P173 p50 445 ms、P174 p50 182 ms**（live headless，30 fps 步調，
`outputs/benchmarks/localization_1346/live_cur_defaults`、
`live_p174_defaults`）。0.3 s 週期 < 0.445 s 中位數，worker 在 P173 上
100% busy、reloc 永久排隊；排隊的 CPU/GPU 負載經 GIL 反壓快迴路
（track p95 34 ms）與 Tk tick（ui_poll p95 97 ms），整機定位顯示只剩 ~12 fps。
`P174_AND_NEXT.md`「准做 #2」(`period >= 1.0 * median_s`) 要求用本機實測重設，
1.0 s 對兩條驗收路線的中位數都有 2 倍以上餘裕。觸發器（`min_points`、
`NO_POSE`）不受週期閘門影響，弱段救援速率不變；只放慢健康態的定期 reseed。

| 路線 | `period_s` | decode p50/p95 | loop p95 | reloc p50 | NO_POSE | FAST_TRACK | 地圖約束比 |
|---|---|---|---|---|---|---|---|
| P173 live | 0.3 | 28.6 / 73.8 | 49.5 | 445 | 108 | 1704 | 61.5% |
| P173 live | **1.0** | **10.7 / 32.1** | **19.8** | **243** | **45** | **1977** | **69.1%** |
| P174 live（holdout） | 0.3 | 7.4 / 44.1 | 35.6 | 182 | 12 | 1504 | 87.2% |
| P174 live（holdout） | **1.0** | **5.6 / 28.7** | **20.9** | **191** | **15** | **1562** | **87.0%** |

單位 ms；地圖約束比 = (FAST_TRACK + RELOC_SEED) / 全幀。P174 兩臂聯合 ok 幀
位姿差 p50 0.019、p95 0.093 地圖單位（map_scale 2.34）—— 無回退。
P173 改善幅度含定時相位運氣（兩臂 reloc 投遞時刻不同），P174 持平才是保守讀法：
**不退化 + 全幀 < 50 ms**。冷啟動不受影響（`min_points` 觸發不走週期閘門；
worker 暖機中本來就投不進去）。

GUI 端到端（simulated-stream，P173，正式 release profile）：10 s 窗格
distinct 定位結果 **20.0–23.2/s**（改前同條件 ~12/s），e2e submit→UI
p50 22 / p95 45 ms，tick 27.3/s。顯示層另有兩筆不碰精度的修改：
HUD 描邊文字改純文字加底條（`draw_video_hud` 43.7→8.6 ms/幀），
雙面板髒汙同 tick 時地圖每 3 幀畫一次（`_render_if_dirty`）。
無加速（reloc p50 445→446 ms），維持預設關閉，程式與單元測試保留。

## VO 與 dead reckoning：P157 上惰性，驗收影片上是主力

**先記一次判斷錯誤。** 本檔第一版根據 P157（802 幀）寫下「VO 與 dead-reckon 一次都沒
開火、正確性未被行使」。那是**用錯影片得到的結論**：P157 的地圖覆蓋太好，
`live_points` 從來沒掉到 `vo.min_live = 350` 以下，PnP 也從來沒失敗過，所以兩個子
系統的觸發條件根本沒出現。它們對 P157 沒有鑑別力。

在操作員指定的驗收影片上，同樣的程式碼完全不是這樣（2026-09-09，兩輪交錯）：

| | P157 | P174（1829 幀） | P173（2929 幀） |
|---|---|---|---|
| `VO_ONLY` 幀 | **0** | 105 - 180 | **670 - 821** |
| VO 有貢獻 inlier 的幀 | **0 / 802** | 777 - 885 | 1190 - 1541 |
| `DEAD_RECKON` 幀 | **0** | 0 | **19 - 101** |
| `vo_candidates` 峰值 | 0 | 約 1700 | 約 1700 |

P173 上這兩條 fallback 扛了四分之一以上的幀。**機制**：兩者都是飢餓 fallback ——
VO 只在 `len(live_xy) < vo.min_live` 且撞上 keyframe tick 時播種，DR 只在 PnP 解不出
`>= pnp.min_inliers` 且已有 `last_pose` 時接手。地圖供點夠快時它們必然是死的。

**教訓（准入規則補充）**：任何關於 VO、dead reckoning、或 `min_live` / `track_cap` /
handover 門檻的結論，**不得只用 P157 這類全程高覆蓋的影片得出**。要用 P173。

## `azimuth_turn_exceeds` 目前沒有可量測的效果

第 3 項修正的是一個真實的量測錯誤（守衛量錯平面），但在 P173/P174 上
`_dead_reckon_guarded` **四輪全部為 0** —— 舊版與新版都一次沒有觸發，即使 P173 的
`DEAD_RECKON` 開火了 19 - 101 次。所以這項改動目前是**正確性修正，不是已證實的改善**。
要證明它有用，需要一段真的出現急轉彎且同時 PnP 餓死的錄影。

## 5. 快迴路 PnP RANSAC seed 固定為 0（2026-09-09 落地，量測方法）

pycolmap 預設 `RANSACOptions.random_seed = -1`（非確定），vendor reloc
（`_solve` 經 `_estimate_pnp`）早已用 seed=0，快迴路 `TwoRateTracker` 卻沒設。
合成場景實測：同一組 80 點（含 25% 粗差）跑 5 次，seed=-1 得 3 種不同
inlier 集合（59/59/60/60/59），seed=0 五次逐位元相同。這就是 P173 覆蓋率
同臂擺盪 3-4pp 的機制：邊緣幀在 ≥12 inlier 門檻上被 RNG 翻來翻去。
改動只是一行 `estimation.ransac.random_seed = 0`，不是準度旋鈕（P174 禁令不適用），
與 vendor 已有慣例對齊。回歸見
`tests/localization/deploy/test_pnp_ransac_seed.py`（3 tests：seed=0 確定性、
seed=-1 非確定性、tracker 原始碼釘住）。

驗證（非調參）：P173 全片單跑（seed 已固定），2899 評分幀覆蓋 **97.03%**
（FAST_TRACK 1753、RELOC_SEED 173、VO_ONLY 811、DEAD_RECKON 76、NO_POSE 86），
落在既有 grey 基線 97.68% ± 1.35 的 1σ 內 —— 無退化，但單跑也不能宣稱提升。
端到端完全確定性尚未驗證：VO 的 cv2 RANSAC（`findEssentialMat`）仍吃全域 RNG，
沒有釘。要把 A/B 降到單跑可比，還得處理那一處。

## 6. P173 失敗幀空間分佈（2026-09-09，純分析，未動任何旋鈕）

同一單跑的逐幀記錄：86 個 NO_POSE **100% 落在 ordinal 2614–2725**（112 幀窗口，
即已知 151 幀洞 2614–2764 的子集）。洞內 `map_inliers` p50 = **0**（洞外 176），
洞內狀態只有 NO_POSE 86 + DEAD_RECKON 26，一個地圖約束幀都沒有；
洞外 2787 幀 **0 個 NO_POSE**。VO 在洞前扛了最長 514 幀（2090–2603），到洞口才斷。
括弧定位顯示相機過洞位移僅約 0.01 地圖單位（慢速段）。

結論：這是**地圖覆蓋洞，不是 tracker 調參問題**。槓桿是往該路段補參考影像，
不是調 `min_live` / handover / PnP（且 P174 禁令本來就禁止拿 P173 調這些）。
後續精度實驗的准入維持：新航線凍結組態跑一次；P173 只做驗證，不做選擇。

## 已知缺口（不是優化，是護欄沒蓋到）

- ~~`check_maintainability.py` 沒有涵蓋 `sfm_direct_deploy`~~ **已修（2026-09-09）**。
  `SCAN_PATHS` 加入現行 localizer，vendor 子樹由 pyproject 的 ruff `exclude` 擋掉。
  重新定基準後 `deploy` 從舊的 26/57 收緊到實測 **2/23**，`validation` 從 8/19 收緊到
  **0/0**（唯一的違規者是凍結的 P174 replay harness，改用 per-file-ignores 具名豁免，
  而不是把整組天花板拉到它的 115）。`tools` 與 `control` 則因遷移帶進來的發布工具而
  放寬，那是待還的債。
- `river_gluemap_all8_direct_20260908` 的 quality receipt 自 2026-09-09 起是
  **OPERATOR_ACCEPTANCE（passed=true，2026-10-09 到期），validation: NONE、
  ground truth: NONE** —— 放行的是有人監督試飛的定位，不是驗證。
  本檔所有數字仍是**相對**比較（同輸入前後對照），沒有一項是絕對精度。

## 8. 幾何預計算、載入與 handover 補救（2026-09-17）

目前河濱 release 已加入 `localization/direct_geometry.npz`，由 bundle 的
`files[]` 綁定大小與 SHA-256，archive 另檢查三份 COLMAP model 的 digest。
未宣告的 archive 不會被使用；舊 release 繼續走原始解析。建圖發布工具會產生
這份資料，內容不包含絕對影像路徑，搬機後仍從已驗證的 keyframe index 取路徑。
幾何計算方式變更時必須更新 archive schema 並重建，不能沿用舊預計算資料。

本機 CPU 量測，證據在 `outputs/benchmarks/direct_1245_20260917/`：

| 項目 | 原本 | 修改後 | 驗證範圍 |
|---|---:|---:|---|
| 河濱幾何載入 | 25.207 s | 0.332 / 0.340 / 0.341 s | 原始一次、archive 三次；所有位姿、觀測、ID、XYZ 與統計逐項相同 |
| 500 萬點 binary PLY 讀取 | 41.2 ms | 13.5 ms | 同一全域 stride，輸出逐位元相同 |
| 上述 PLY 子行程峰值 RSS | 104524 KiB | 33080 KiB | 暫存讀取限制為 65536 筆，輸出上限 12 萬點 |

archive 約 348 MB。量測未清除 OS page cache；幾何時間不包含 GPU 模型。
另一次實際 production factory + CUDA 模型載入與預熱為 3.548 s，不含 Python
行程與前置 import，沒有 GUI 或飛行連線。原始三份 model、profile 參數與品質
收據的 `validation: NONE` 均維持原意；此次只更新 bundle 相依 digest，額外記錄
`provenance/geometry_preparation_20260917.json`，不宣稱取得新的品質或實飛驗證。

handover 保留 stride-2。單跳存活點不足既有 reseed 門檻時，使用中間影格重試；
只有保留更多通過原 FB／邊界檢查的點才採用，成功後該次 handover 改逐幀走完。
合成 960×540 紋理、500 點、五幀平移測試：每幀 16 px 時存活點 35→433，
24 px 時 10→269；兩種情境耗時約 17–19→37–38 ms。每幀 2/8 px 的點集不變，
仍走隔幀快路徑。這是困難 handover 的計算成本與存活率取捨，不能宣稱實飛
精度提升或每一幀更快。`handover_retry_hops` 記錄自 tracker reset 起的重試跳數。

相關回歸涵蓋真實 CPU KLT 的已知平移誤差、失敗補救不替換較好的隔幀結果、
預計算／原始幾何等價、搬移路徑與資產竄改拒絕。航向錨點只接受已觀測的地圖
定位，BOOT 也直接檢查 pose metadata 與 predicted/reseed hooks；弱定位維持
既有補位移動語意，起降及操作員接管語意未變。

此次相關離線測試 534 passed、1 skipped（烏來 alignment 資產未提供）；Ruff、
diff whitespace 與兩份 mission selection 的相依驗證通過。全專案 mypy 仍有兩項
既有錯誤（`_predicted_speed`、`_make_autonomy_pose`），複雜度／檔案長度總閘門也仍
超標；已確認相關既有程式與 HEAD 相同，未因這次工作修改它們。

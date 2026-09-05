# 定位演算法優化 Runbook

最後更新：2026-09-05
分支：`agent/localization-runtime-optimizations`
硬體基準：NVIDIA GeForce RTX 5060 Laptop GPU（8151 MiB，sm_120）

本文件是「還能優化什麼、每項要怎麼驗、目前做到哪」的操作清單。已驗證且保留的優化寫在
`docs/verified_localization_optimization_ledger.md`（總帳），本文件只負責把待辦與進行中的實驗
排序、標註 gate、追蹤狀態。**任何項目在通過對應 gate 之前，不得寫進總帳、不得升級為 flight
profile、不得移除安全 gate。**

---

## 0. 狀態（2026-09-05，跑在 repo venv + RTX 5060）

**本輪唯一的結構性結論：單段 gate 不足以決定預設。** 七段 720p 語料庫（§5a）第一次全跑，
2026-09-04 靠 P117/P168 兩段升級的 `SFM_EDM_ASYNC_TRACKER` 預設在其餘五段大幅退化，已改回預設 0。
數值、機制與三項證據見總帳「2026-09-05 全影片 720p 語料庫（七段）＋ async 預設複查」。

| 2026-09-05 項目 | 結果 |
|---|---|
| **七段 720p 語料庫**（新標準回歸集） | 建立。`模擬器/測試影片/720p/`，production-path、stride 3、seed 0、全片。**報告 successes 與 verified 兩個數字**（verified = 當幀真的重新對上地圖）。 |
| **`SFM_EDM_ASYNC_TRACKER` 預設 1 → 0** | 七段 async successes 62.9% / verified 7.5%；sync 全部 success 都是 verified。async 另有連續 710 幀（約 89 秒）純光流推算回報成功、同輸入兩次跑出 0/292 與 135/273 的不可重現性。詳見總帳。 |
| **`SyncCarry` 無界推算修正**（`SFM_EDM_ASYNC_MAX_ANCHORLESS`，預設 30） | `anchorless_frames` 硬上限原本寫在它要保護的軟上限分支內，永遠到不了。改為獨立判斷。這是 correctness fix，async 逃生閥才有意義。 |
| **操作介面相機朝向**（三個缺陷） | async adapter 從不發佈 `camera_axes_world`（也讓 `_autonomy_pose` 永遠回 None）、`_update_live_heading` 只在 AUTO 時採用相機朝向、平面符號把雲台俯角混進方位。全部已修＋回歸測試。 |
| **`acquire_relaxed_min_inliers` 預設 0 → 66** | 09-04 的 NOT GO 是 sequential／2–4 段／floor 70 的結論。七段 production-path 重測：**+231 幀（80.6%→83.6%）**，P116 +102、P119 +96、P157 +28、P167 +25、P117/P118 平手、P168 −21（seed 0/1/2 為 −21/−36/−21，唯一付代價的段）。off-map hard negative off/on 都 0/241。品質 inliers p05 不動、p50 −3，reproj p50 +0.09 px。**升級為 code default**（不 pin 進 profile，SHA 鏈不動）。 |
| **降 720p 不傷精度** | P116 前 300 幀 sequential：720p 252/300 vs 原始 4K 211/300。720p 語料庫可當標準集。 |

### 前一輪狀態（2026-09-04）

全套 pytest `2448 passed / 3 skipped`（該日收尾數）。GPU gate 明細在總帳「2026-09-02 GPU gate 結果」+ 項 12 / 13 +
2026-09-04 的四個區塊（LOST 近失 acceptance gate、starvation retry、async anchor 供給、conf-only KLT）。

| 2026-09-04 新增項 | 結果 |
|---|---|
| **LOST 近失 acceptance gate**（`acquire_relaxed_min_inliers`，預設 0）<br>**（2026-09-05 已推翻，升級為預設 66，見上表）** | 當時判定：實作完成、量到最大單一成功率槓桿但 **NOT GO as default**：floor 66 在 P119 +100（seed 1 +93）、P157 +26、P117 0，但 **P168 −53（seed 1 −18）且品質退化**（inliers p50 82→68、p95 154→111、step p95 2x）。floor 70 是零退化版（+26/+3/0）但淨 +29 落在 ±30 seed 帶內。off-map hard negative（河濱影片 vs 烏來地圖 700 幀）off/on 都 0/700 逐項一致，off-map inlier 上限 17 << 66。詳見總帳。 |
| **starvation-triggered global retry**（`lost_starved_global_frames`，預設 0） | **REJECTED**：P119 k=2 觸發 48 次只換到 +2（45 次仍死在 acquire 地板），P157 觸發 0 次，與放寬地板疊加無增益，p50 +0.8 ms。瓶頸在 acceptance 側不在 reference 供給。 |
| **async fast/slow 升級為預設**（`SFM_EDM_ASYNC_TRACKER` 預設 1）<br>**（2026-09-05 已推翻，改回預設 0，見上表）** | 當時判定：**正式轉為 code default 開**。fast 門檻定錨為 `fast50`（inliers 50、reproj 2.5）：P168 605/700（與 Sync 605 相同）、P117 375/416（Sync 358，+17 幀），全幀誤差 vs Sync 為 p50 4–8 mm / p95 2.8–3.6 cm / max 11.5 cm；延遲 p50 1.58–1.80 ms（**15–17x 加速**）。off-map hard negative 0/700 全 NO_ANCHOR 零假鎖。逃生閥：`SFM_EDM_ASYNC_TRACKER=0` 可無損回退同步追蹤。 |
| **TRACK 失手同幀擴 ref**（`track_miss_widen_topk`，預設 0） | 中性（+6/+3/−2/0，淨 +7）：同 pool 多 refs 救不了幾何不成立。不開。 |
| **BOOT 兩幀確認 + 放寬地板**（`boot_relaxed_min_inliers`，預設 0） | 小幅正向：P116 +14、P168 +1，品質不變。等 seed 1 確認。 |
| **relaxed-accept 試用期**（`acquire_relaxed_probation_frames`，預設 0） | REJECTED：N=5 與 prob0 完全相同（P168 801）。損失機制不是 TRACK 撐不住。 |
| **p95 尾巴**（`SFM_EDM_LOST_LOCAL_TOPK`） | 定位完成：尾巴 = 5-ref LOST 的 EDM forward（~19 ms/ref）。k=3 在 P168 −62 → NOT GO；P119 中性但 p95 −36 ms。 |
| **async sequential 簡併 + inline-sync floor**（本日稽核實測） | async 預設開後，sequential replay（tight loop 快過 slow thread）**0/200 全 NO_ANCHOR**：slow 首個 keyframe 永遠追不上。修法：fast 無可發佈（VC/KLT_BRIDGED 以外，含帶 stale pose 的 NEED_REANCHOR/LOST）時當幀 inline 跑同步 keyframe（`SFM_EDM_ASYNC_INLINE_SYNC=0` 可關），slow 錯誤計數不殺 thread（`slow_errors`/`slow_alive`/`inline_syncs` 進 `_last_info`）。sequential+async 回到 151/200（sync 165/200，剩 −7pp 是 sparse 慢驅動的 stale-prior 差距，見 §3 async 列）。 |
| **factory 雙建構浪費** | async 開時同時建 sync + async 兩個 adapter（含兩個 stateful tracker，async 的還起 slow thread），sync 那個只借 pose_guided 就丟掉。改為先判 async_mode 只建一個。sequential 200f pin-sync 跑兩次：165/200 ×2 且 trace SHA 與改動前 sync baseline **逐位元相同**（`4d5fe1…`），零行為差異。 |
| **sequential gate 確定性** | threaded async 非 bit-deterministic，exact trace SHA gate 會被污染。`benchmark_edm_site_replay.py` sequential 分支在 env 未顯式指定時 pin `SFM_EDM_ASYNC_TRACKER=0`；production-path 不動。receipt 新增 `async_tracker` / `async_inline_sync` 並納入 fail-closed identity（舊 baseline 會報 missing key，屬預期失效，需重跑）。 |
| **production-path P117 複驗**（本日樹，async+inline vs sync） | common 396 幀：**async 370 vs sync 352（+18；async-only-ok 22 / sync-only-ok 4）**，pose 一致 p50 5mm / p95 3.9cm，p50 1.72 vs 22.7 ms，p95 同 ~85 ms，coalesce drops 21 vs 14。預設開的決策在本日樹上依然成立。證據 `outputs/audit_20260904/audit_pp_p117_async.json` / `outputs/audit_20260904/audit_pp_p117_sync.json`。**同日再加 slow-prior 回寫後：398/405＋rep2 394/404（common 390 vs 372，+18；19/1），p95 85→11 ms，見下行。** |
| **精度最高預設審計＋slow-prior 回寫**（本日第二輪） | 審計：async/inline/fast50/interval3/quality0.5/temporal-true/hysteresis 等**已全是預設，無需改**（項 11 過時標題已修）。回寫：已發佈 fast pose 只刷新 slow 先驗四件（center/yaw/velocity/stamp），worker 播種過 mirror 就 dormant。pp P117 **398/405＋394/404，兩次同向超 threading 帶**；bridged inliers p50 52→275，失敗 26→7。sequential 中性（150 vs 151，valley 無新資訊可給）。保留常開。詳見總帳。 |
| **追問再優化**（inline stand-down＋TF32） | stand-down（連敗 K 次改 fail-fast）sequential 151→111、LOST 7→72，機制證偽（失敗 keyframe 仍推進 slow 狀態機），**revert 零殘留**；TF32（`high`）非 exact 且 match_ms 無加速，**NOT GO**；`match_batch_size=2` 確認量過最優不動。詳見總帳。 |
| **SDPA fusion 實測**（`SFM_EDM_SDPA=1`，sequential P168 200f） | succ 164 vs 165（−1 雜訊），但 103/200 幀 inliers 不同（非 exact），match_ms 中位數只 −0.8 ms（LOST 幀 −7.7 ms）。按本 repo exact 規則（bitwise identity + p50/p95 雙改善）**NOT GO**，維持預設關。 |
| **測試/靜態門** | 修 width 脆弱的 `--help` 斷言（pin `COLUMNS=200` + 反向斷言無 `%%` 殘留）；清 ruff 12 項（async adapter/test  dead imports、`_PoseAttempt` F821、`axis` F841）；修 mypy `localization_contract.py` 7 項（stale ignore＋補 `keys`/`__iter__` 註解）。 |

### 前一輪狀態（2026-09-03）

GPU gate 明細在總帳「2026-09-02 GPU gate 結果」+ 項 12 / 13。

| 項目 | 結果 |
|---|---|
| **KLT bridge `SFM_EDM_KLT_BRIDGE_INTERVAL` 預設 `0`（關閉）** | 2026-09-03 曾開成 `3`,同日 production-path 重測後**改回 `0`**（總帳項 13）。重測:P168 700f 同設定 4 次,bridge-off 穩定 575/685、bridge-on 516–528 → **−47~−59 succ**（原記錄 −18 是單次量測）。sequential 七段仍淨 +89 succ / −121 LOST,所以功能保留,平順延遲敏感場景可設 `=3`（P117 wall p50 3.3x 改善、succ 平手）。**replay baseline 回到 bridge-off 語意。** |
| **Tier 4 — 1045-ref 地圖相依參數重驗** | 2026-09-02 完成（總帳「Tier 4」）。S=2.3346915 重算 exact 相符、radius=0.40·S 為 P168 sweep 峰值、cache 192 在 P168/P117 與 cache-64 trace byte-identical。無參數變動 → SHA chain 不動。 |
| **ESEKF / KLT 3D-aware live gate** | `SFM_EDM_ESEKF_DISABLE=1` toggle + `benchmark_esekf_live_replay.py` + `docs/esekf_live_eval_runbook.md`（一次手動飛行 + 機上錄影 → 離線 on/off 相對指標）。純 replay 仍評不到（無 live velocity）。 |
| **`lost_global_retrieval_interval` 15→3** | **已升級進 flight profile**（總帳項 12）。P168 536→608（+72），P117 366→366（0）。profile SHA `93e0c2…`→`a65f78ca…`,manifest chain 已同步。 |
| `use_temporal_reference` true→false | REJECTED — P168 +20 但 P117 −20 / LOST +12 / p50 +12。候選 profile 已刪。 |
| `lost_prior_strategy` full_global / score_fusion | REJECTED — 同型（P168 +65，P117 −17）。 |
| `--no-track-map-first` | REJECTED — +37 但 p50 22.75→41.56 ms。 |
| `acquire_stage_mode` / `pnp_ranked_batches` / `local_topk` | 無效（P168 successes 不變）。 |
| EDM neck `repeat→expand` | exact 已驗證（700 幀 0 diff），但無 TRACK 加速。保留 guarded + `SFM_EDM_NECK_NO_EXPAND` 逃生閥。 |
| ESEKF / KLT 3D-aware | 在 `__init__` 無條件建但 replay 休眠（無 live velocity）；replay gate 評不到。 |
| `_track_klt_prior` `raise StopIteration` | 移除,行為等價。 |
| cache 8GiB 預算對 CLI override 失效 | 已修（`_apply_reference_feature_cache_overrides` 先 validate 再 mutate）+ 回歸測試。 |

雙份同步邊界（`edm_matcher.py` 等,見總帳「實作同步邊界」）本輪未動,兩棵樹 0 行差異。

---

## 0b. 主機側延遲 —— 一直沒被量過的另外一半（2026-09-05 新增）

Tier 1-3 都在壓 worker 裡的 GPU 時間。這一節第一次把**結果算完之後**的時間量出來，
結論是：**送出到 UI 拿到位姿的時間裡，推論只佔一半。**

量法：模擬串流跑 132 秒 P1680168、1676 幀 `pose_result`，直接讀 session
`localization.jsonl` 已有的 timing 欄位（都是既有欄位，不是為這次加的）。

| 階段 | p50 | p95 | p99 |
|---|---|---|---|
| `ui_serialize_ms`（UI 打包影格） | 0.72 | 1.22 | 1.57 |
| `client_queue_wait_ms` | 0.38 | 14.85 | 50.31 |
| `client_pipe_write_ms` | 0.14 | 0.94 | 8.89 |
| `submit_to_worker_read_ms` | 0.60 | 15.78 | 54.27 |
| **`core_wall_ms`（worker 推論）** | **26.30** | **109.47** | **388.07** |
| `worker_done_to_client_ms` | 3.07 | 11.31 | 19.02 |
| **`ui_poll_delay_ms`（結果躺著等 Tk 事件圈）** | **24.99** | **45.58** | **61.41** |
| **`e2e_submit_to_ui_ms`** | **59.55** | **152.28** | **392.31** |

三個可以直接拿去做的觀察：

1. **`ui_poll_delay_ms` p50 25 ms，跟整段 GPU 推論同一個量級。** 這段純粹是結果已經
   在 client queue 裡、等 Tk 事件圈回頭拿。喚醒機制沒有問題 —— `createfilehandler`
   有掛上（Tk 8.6.12 有這個 API，`_attach_localizer_file_handler` 成功時 fallback poll
   放寬到 100 ms），所以 25 ms 是**事件圈本身在忙**，不是沒被叫醒。
2. **worker duty cycle 只有 61.7%**（132.4 s 內 `sum(core_wall)` = 81.7 s）。
   結果間隔 p50 66 ms，其中推論 26 ms —— 也就是 GPU 有近四成時間沒事做，
   而擋在中間的最大單項就是上面那 25 ms。把它壓下去等於免費拉高定位率，
   而且是**降低**位姿年齡，不是拿延遲換吞吐。
3. 單 tick 的算圖成本（RTX 5060、900x700 圖面、20 萬點雲）：
   `render_map` 疊圖 **2.10 ms**（底圖有 cache）、`prepare_video_frame` **2.52 ms**、
   底圖重畫 **11.14 ms**（只在縮放／平移／旋轉改變時）。單看都不夠解釋 25 ms。

**限制（重要）：以上是模擬串流路徑。** 模擬路徑的 ffmpeg 讀影格是在 Tk 執行緒上
inline 做的，真機路徑的影格來自 Olympe callback 執行緒 —— 兩者的事件圈負載不同，
**真機的數字必須另外量，不能沿用**。

真機要怎麼量：這些欄位在真機 session 一樣會寫進 `localization.jsonl`，所以
`./IMU飛行測試.sh` 飛一趟就有：

```bash
.venv/bin/python - <<'EOF'
import json
rows=[json.loads(l) for l in open("<session>/localization.jsonl")]
rows=[r for r in rows if r.get("event")=="pose_result"]
def p(k,q):
    v=sorted(float(r[k]) for r in rows if isinstance(r.get(k),(int,float)))
    return v[min(len(v)-1,max(0,round(q*len(v))-1))] if v else None
for k in ("core_wall_ms","ui_poll_delay_ms","e2e_submit_to_ui_ms"):
    print(k, p(k,.5), p(k,.95))
EOF
```

**先量再改。** 如果真機的 `ui_poll_delay_ms` 也在 20 ms 以上，候選手段依序是：
(a) 把 render 從結果處理的同一個 callback 裡拆開（結果先套用、畫面下一 tick 再畫）；
(b) 只在 pose 真的變了才重畫地圖疊圖（`_map_dirty_key` 已有骨架）；
(c) 影片面板降到 UI 需要的張數，而不是每個 stream 影格都 fit 一次。
三者都不碰定位數值，所以 gate 是延遲百分位 + 既有 pytest，不需要七段語料庫。

### 0b-1. 三個候選的現況（2026-09-05 覆核）

覆核程式碼後，上面三項有兩項其實**已經做掉了**，所以不要再把它們當待辦：

- **(b) 已實作。** `operator_tick._render_if_dirty` 用 `_map_dirty_key` 比對
  pose / 相機軸 / 防撞圈 / 路線 / 面板尺寸，不同才重畫並換 PhotoImage。
- **(c) 已實作。** 同一個函式用 `_video_dirty_key` 擋住影片面板，來源影格沒換就
  跳過 720p resize + PhotoImage。
- **(a) 不是問題所在。** `_on_localizer_result_ready`（Tk `createfilehandler`
  回呼）只做 `drain_result_notifications()` + `update_live_results()`，本來就沒有
  render。真正擋住事件圈的是**週期性的 tick 本身**（真機 30 Hz，`tick_ms=33`），
  結果躺在 queue 裡等 tick 跑完。

也就是說 doc 裡列的手段都不能解釋那 25 ms，而 §0b 自己量到的單 tick 成本
（疊圖 2.10 ms + 影片 2.52 ms）也對不起來。**下一步不是再猜一個手段，是先知道
是哪一個 stage 佔住事件圈。**

`run_tick` 現在會逐 stage 計時（`_TickProfile`，沿用既有的 `stage` 名稱，
失敗回報的語意完全不變），每 5 秒把 p50/p95 寫一筆 `ui_tick_profile` 進 session
的 `telemetry.jsonl`。所以**真機數字不需要另外做一次分析** ——
`./IMU飛行測試.sh` 飛一趟，`tools/imu_flight_test_report.py` 就會直接印出
「最貴的階段」與每個 stage 的 p50/max。拿到那個名字之後才值得動手改。

### 0b-1-答：兇手是 tick 自我追趕，已修（2026-09-05 同日）

儀表第一次實測（模擬串流預設影片 P119、session `outputs/flight_logs/session_20260905T153849Z_simulated-stream_eb232b58`，130.6 s）回答了上面的問題，而且答案不是任何單一 stage：

- BOOT 期的第一個 window：99 Hz tick、render 3.7 ms —— 便宜。
- 之後 **`render_if_dirty` 每個 render tick 44–64 ms**（map overlay + 720p 影片 + HUD
  一起畫；舊估「疊圖 2.10 + 影片 2.52 ms」量不到 in-situ 這個價），8 ms 週期永遠追不上，
  `next_tick_deadline` 對逾期 tick 以 1 ms 自排程 → **back-to-back tick 把事件圈吃滿**
  （實測 13–23 Hz、duty ~100%）。§0b-1 的「真機 30 Hz，tick_ms=33」假設不成立：
  真機啟動鏈（`真機串流/啟動.sh` → `start_anafi_live.sh`）不帶 `--tick-ms`，
  `min(8, 33)=8`，真機原本同樣跑 125 Hz。
- Tk 每次迴圈先服務 window、再 timer、最後 file event，所以結果通知的
  filehandler 只能在 tick 之間的縫隙被消費。`ui_poll_delay_ms` 分佈證實機制：
  21.9% <5 ms（縫隙裡消費）、46% 落在 20–40 ms 平台；**TRACK 結果（render 多）p50 26 ms、
  LOST/WEAK 結果（render 少）p50 7.5 ms** —— render 越多、事件圈越堵、結果越遲，
  自我強化。

**修復（已上樹）**：`OperatorApp.__init__` 把 tick 週期夾到 8–33 ms、CLI `--tick-ms`
預設 8→33（help 註明理由）。夾兩側都有依據：更慢會讓 live nudge deadman TTL
（下限 100 ms）在 tick 之間過期；更快就是上面量到的餓死機制。舊行為保留為
`--tick-ms 8` 明確 opt-out。測試 `test_ui_tick_defaults_to_30_hz_and_clamps_both_sides`
取代 `test_ui_refresh_targets_125_hz`（125 Hz 是隨 `97cb0ab` 整倉遷入的預設，無量測依據）。

**A/B（同 launcher、同影片；candidate session `session_20260905T161535Z_simulated-stream_ad941f58`，191.5 s）**：

| 指標 | baseline 8 ms | candidate 33 ms |
|---|---|---|
| `ui_poll_delay_ms` p50 / p95 | 24.69 / 48.56 | **3.51 / 17.33** |
| `e2e_submit_to_ui_ms` p50 / p95 | 58.67 / 140.93 | **37.75 / 122.21** |
| tick 實際節奏 | 13–23 Hz、duty ~100%（body 44–64 ms） | 15.2 Hz、duty ~55% |
| worker duty / pose_result 成功 | 54.8% / 1626/1938（83.9%） | 55.7% / 2246/2707（83.0%） |

worker duty 與成功率不動（submit 是 per-new-frame，與 tick 頻率無關）——收益全部是
**位姿送達年齡**：p50 −21 ms、p95 −18 ms。incidents 兩 session 都只有模擬器固有的
`sparse_cloud_collision_hover`。Gate：control-interface 1234 passed、全套
**2539 passed / 3 skipped**、ruff 乾淨。

**新浮出的下一個槓桿：`render_if_dirty` 本身（兩邊都 p50 34–54 ms，佔 tick body ~95%）。**
tick 33 ms 之後它就是唯一的大項；把它拆薄（候選：靜態點雲層 per view cache、
map/video 交錯渲染）才能把影片面板推回 30 fps（現在 33 ms tick 上限 ~15 fps，
baseline 飽和時 ~15–20 Hz，兩邊都沒到 30）。真機確認仍欠：飛一趟讀
`ui_tick_profile` / `ui_poll_delay_ms`（§0b 的腳本），預期同機制。

---

## 0b-2. worker 內那 27% 已經拆完了 —— 沒有東西可撿（2026-09-05）

§0b 量到 `vpr+match+pnp` 只佔 `core_wall` 的 73%,剩下 27% 沒人知道是什麼。
猜測是 `prepare_query`（每幀 query backbone forward,確實在 `match_ms` 之外）。
**猜錯了。** 加了四個 stage timer 之後量出來:

P168 700f、stride 3、seed 0、sequential、RTX 5060（p50 / p95,單位 ms）:

| stage | 全部 700 幀 | TRACK 556 | LOST+WEAK 137 | 佔 p50 total |
|---|---|---|---|---|
| `match_ms` | **18.21** / 87.23 | 17.53 / 39.47 | **57.93** / 107.41 | **72.7%** |
| `stage_select_ms`（含 `vpr_ms`） | 2.27 / 8.95 | 1.80 / 5.78 | 2.41 / 13.49 | 9.1% |
| `pnp_ms` | 1.76 / 4.59 | 1.74 / 3.68 | 1.98 / 6.58 | 7.0% |
| `stage_gray_ms` | 0.71 / 1.33 | 0.75 / 1.36 | 0.68 / 1.03 | 2.8% |
| `stage_query_ms` | **0.60** / 0.82 | 0.61 / 0.83 | 0.59 / 0.72 | **2.4%** |
| `stage_bridge_ms`（bridge 預設關） | 0.00 | 0.00 | 0.00 | 0.0% |
| localize 內殘差 | 1.48 | 1.26 | −0.48 | 5.9% |
| `_localization_info` 之後的尾巴 | 1.45 | 1.46 | 0.63 | 5.5% of wall |
| **`total_ms` / `wall_ms`** | 25.03 / **26.48** | 23.69 / 25.15 | 63.11 / 63.74 | — |

**結論:那 27% 是五個 0.6–2.3 ms 的零碎項,不是一個藏起來的區塊,沒有單一槓桿。**
`prepare_query` 只有 **0.60 ms** —— 正因為總帳項 2「每幀 query backbone feature 只算一次」
已經做掉了,重的部分本來就留在 per-reference 的 match 裡。設計是對的。
LOST/WEAK 段 match 佔到 **91.8%**,再次確認瓶頸就是多 ref 的 EDM forward,
跟 Tier 1–3 的結論一致。**這條線到此為止,不要再花時間。**

儀表本身保留:已證明 **exact**（見下),而且真機 session 也會寫進 `localization.jsonl`,
以後不必再猜。

**exact 證明（照 §5b 准入規則）:** 同一條指令跑 timer-on / timer-off 兩輪,
`successes` 557/557、`state_counts` 兩邊皆 `{BOOT_INIT 6, TRACK 557, WEAK_TRACK 65, LOST 72}`,
700 幀逐幀比對 `success` / `mode` / `inliers` / `reproj_rms` / `n_corr` / `refs` / `pose_xyz` /
`candidate_mode` / `selected_ref` / `rejected` / `limited_jump` / `reference_count`
**零差異**。改動是純加法（三個 hunk,只加不刪,無新分支)。

> **順帶發現的 baseline 漂移,待查:** 上面兩輪(含未加儀表的那輪)在
> `模擬器/測試影片/P1680168.MP4` 700f 都是 **557/700**,不是 §0/§5b 到處引用的 **608/700**。
> 兩輪都是本樹,所以與這次改動無關。最可能是 2026-09-05 把
> `acquire_relaxed_min_inliers` 升成預設 66（總帳記該項在 P168 為 −21,但那是七段 720p
> production-path 的數字,不是這條 4K sequential 指令)。**在拿 608 當任何 gate 的門檻之前,
> 先重新確立這條指令在本樹的 baseline。**
>
> **2026-09-05 已重新確立（本項結案）:** 本樹重跑同一條指令
> （`benchmark_edm_site_replay.py --site-profile 地圖檔/場域/river_site/site_profile.json
> --video 模擬器/測試影片/P1680168.MP4 --stride 3 --max-frames 700 --pnp-random-seed 0
> --require-cuda --gpu-span`,sequential 預設 pin-sync）=
> **557/700（BOOT 6 / TRACK 557 / WEAK 65 / LOST 72）**,與上述兩輪一致。
> 證據 `outputs/baseline_recheck_20260905/p168_4k_seq700.json`。
> **這條 4K sequential 指令的本樹 baseline 就是 557,不是 608**;608 是
> `acquire_relaxed_min_inliers` 升預設前的 2026-09-04 樹。之後任何用這條指令的
> exact/policy gate 都以 557 為 baseline;七段 720p production-path 的 baseline 以
> 總帳「sync_relaxed66」欄（5256/6287）為準。

---

## 0c. 還沒被碰過、但比執行期更有槓桿的兩件事

- **成功率的瓶頸不在執行期，在地圖覆蓋。** 總帳「失敗是空間集中的」量到七段語料庫
  1,206 個失敗幀裡 **95% 落在既有的 14 顆 `red_intrinsic` 球內**。再擠 matcher 的
  毫秒數改不動這一塊；補那 14 個區域的參考影像會。這是建圖工作，不是演算法工作。
  **2026-09-05 已試過「從既有影片抽幀補 reference」：NOT GO。** 工具
  `validation/augment_failure_sphere_references.py` 完整跑通（63 顆新 reference、13/14 球
  涵蓋、單幀煙霧 1/1 TRACK），但七段語料庫 gate 退步 −0.9pp（P116 −5.4pp）且 p50 +3.8 ms：
  新 reference 的 LUT 天然稀疏（單幀 inlier ~1e2 vs 原版 SfM 數千觀測），進了候選池卻永遠
  選不中，純粹擠佔 top-k 名額（aug run 全 1764 幀 `aug_` 出現 0 次）。完整數值與機制見
  總帳「2026-09-05 失敗球 reference 補強」。**補 reference 要嘛正規重飛重建（含三角化），
  要嘛先解決稀疏性機制，不要直接重跑本實驗。**
- **真機入口目前是關著的，這擋住所有真機驗證。**
  `控制介面程式/mission_selections/river_gluemap_all8_direct_localization.json`
  跑 `launch_mission.py --check-only` 會被擋在兩個獨立的 gate：
  1. `localizer_quality` receipt `passed: false`
     （`releases/river_gluemap_all8_direct_20260831/compat/localizer_quality_receipt.json`，
     `validation: NONE`，就是 Tier 4 那個缺口）。
  2. ~~vehicle 的 `camera.pipeline_id` 不在 localizer manifest 的 `camera_profiles`~~
     **2026-09-05 已修**：manifest 的 `camera_profiles` 殘留 official69 時代的
     `river_official69_1280x720_pinhole_v1`（其內參 931.2/931.2/640/360 與本圖完全不同），
     是官方69模板一路複製的 authoring bug。已改成
     `river_b0_p116_p117_map_scaled_1280x720_pinhole_v1`（內參 960.49/958.20/670.82/358.72，
     與 site profile `query_camera` 逐位元一致），selection 的 `localizer.sha256`
     同步輪轉（`eaff3d6f…` → `963fc6db…`）。`validate_mission_selections.py` 的
     camera pipeline 錯誤歸零，resolver 測試與 `package_manifest.py verify` 全過。
  resolver 沒有 override 旗標，**這是刻意的**：兩個都是安全 gate，不要為了出去飛而繞過。
  (2) 已修；(1) 需要真的補獨立 holdout。**現狀：真機入口仍被 (1) 擋住（刻意）。**

---

## 1. Tier 1 — 已無升級標的

`lost_global_retrieval_interval 15→3` 是本輪唯一過 gate 的改動,已進 flight profile（總帳項 12）。
`use_temporal_reference` / `lost_prior_strategy` 全部 REJECTED（見 §0 + 總帳）。`reference_quality_weight 0.5`
已是 code default 且生效中,若要 pin 進 profile 是純文件化動作,下次發版再做。

**Profile 升級流程**（項 12 已照做一次,供未來參考）:改兩份 `edm_runtime_profile.json` → 重算 SHA →
同步 `site_profile.json`（頂+release）`asset_sha256`、`compat/localizer_edm_manifest.json`
`artifacts.profile.sha256`、`控制介面程式/mission_selections/river_gluemap_all8_direct_localization.json`
`localizer.sha256`（= 新 manifest SHA）、`MANIFEST.tsv`/`SHA256SUMS`、總帳 SHA 行 → 跑 `pytest` +
`tools/package_manifest.py verify` + `tools/system_validation.py`。

---

## 2. Tier 2 — 已做完 reference-policy A/B（2026-09-02）

一次掃 11 個單槓桿 + P117 交叉驗證,完整表在總帳「Tier 2 — reference-policy A/B sweep」。**結論:唯一
過 gate 的是 `lost_global_retrieval_interval 15→3`,已升級。** 其餘全 REJECTED,不要重做。

state-conditional top-k / min_inliers 早已實作（`production_edm_tracker.py:1748-1751`、`:1770`）。
`max_corr_total` / `pnp_workers` 沒 state-conditional 但在 PnP 路徑（p50 1.34 ms）不值得。

下一輪若要再壓 P168 LOST（現行 608/700,LOST 65）:候選是「WEAK/LOST 專用的 reference 選擇」而非
全域調參 —— 見 §2b。

### 2b. EDM detector-free 的單幀成本 ∝ candidate references

`production_edm_tracker.py` 開頭 docstring 自述：XFeat 的 temporal anchor cache（比對「上一幀」並帶其 3D）
在 EDM 沒有等價物，「留到證明有需要再做」。在 WEAK/LOST（跑 3-8 refs）這正是瓶頸段。

- 實驗：WEAK/LOST 時把「上一次成功 PnP 幀」當一個額外 reference，帶著它的 inlier 3D。
- gate：§5，另外要確認不會把漂移的上一幀鎖進 recovery。

### 2c. KLT 快路徑 / EDM 慢路徑 解耦

原 `async_pipeline.py` stub 已於 2026-09-02 移除（見 §3）。**in-tracker 版本 2026-09-03 開成 code default
（`SFM_EDM_KLT_BRIDGE_INTERVAL`，現行預設 `0`=關閉，見總帳項 13）：** steady TRACK 從強 EDM anchor 起，
中間 2 幀用 KLT 帶 pose、第 3 幀跑 EDM re-anchor；strong-anchor gate + drift guard。sequential 七段
holdout 淨 +89 succ / −121 LOST、mean −7~−13 ms/段。**已知成本:production-path P168 700f −18 succ**
（sequential 同段 −4，全片 +12 靠平順後半）—— 依使用者決策接受。`_MAX_CONSEC=1` 與 `--adaptive-submit`
都試過且 REJECTED（見總帳「不要重做」表）。**2026-09-04 更新：**（a）把上限放寬成 confidence-only
（`=100`）後，production-path 的結構性劣化消失（P117 +4 / P168 −5，p50 4.6–5.0x），多 seed Δ 全在 ±30 內，
但 p95 變差故不改預設；（b）真解耦（`SFM_EDM_ASYNC_TRACKER`）已接線並實測到 −2.8pp / p50 1.5 ms，
四個 anchor 供給缺口已修完，見 §3 與總帳。exact per-frame 比對的 gate 要設 `SFM_EDM_KLT_BRIDGE_INTERVAL=0`。

---

## 3. Tier 3 — 本分支進行中的實驗

**仍在樹上：**

| 實驗 | 檔案 | 狀態 |
|---|---|---|
| ESEKF 15 維融合（IMU/NED 速度 + visual pose） | `esekf.py`；接線於 `production_edm_tracker.py`（`_init_esekf`、`localize`、`_predict_center` / `_search_yaw` / `_on_miss` PREDICTED_ONLY 分支） | `_init_esekf` 建 `ESEKF(EKFConfig())`（`SFM_EDM_ESEKF_DISABLE=1` 關）。`predict` 需 `observe_fused_state` 餵 `_latest_velocity_ned`；純 replay 無此來源 → `prediction_allowed()` 維持 False → 分支不觸發。608/700 基準已含這條休眠路徑。**live-telemetry gate：`定位演算法/validation/benchmark_esekf_live_replay.py` + `docs/esekf_live_eval_runbook.md`（一次手動飛行 + 機上錄影 → 離線 on/off 相對指標）。** |
| KLT 3D-aware init（`OPTFLOW_USE_INITIAL_FLOW` + covariance window 15→41） | `production_edm_tracker.py::_track_klt_prior` | 只有 `self.esekf.prediction_allowed()` 才啟用 → replay 也休眠,走 fallback LK。`raise StopIteration` 已移除。`test_klt_miss_prior.py` 通過。同 ESEKF live gate（`SFM_EDM_ESEKF_DISABLE` 一起關）。 |
| EDM neck `repeat→expand` | `runtime/EDM/src/edm/neck/neck.py`（`SFM_EDM_NECK_NO_EXPAND=1` 逃生閥） | **exact 已驗證**（700 幀 0 diff），**但 B=2 無加速**（match_ms 17.89 vs 17.55）。保留 guarded,不列已驗證優化。 |
| `megaloc_token_reduction.py` | 接線於 `reloc_localizer_edm.py`（預設 off） | 總帳已否決 L2/EViT token reduction。工具留著,不預設開。 |
| `fine_matching.py` bi-directional `m_bids` stable-sort | `runtime/EDM/src/edm/head/fine_matching.py` | 修 `bs>1` 時 `m_bids` 非單調 → `_split_match_outputs` 邊界錯誤。像 correctness fix,未單獨 exact 驗證。 |
| KLT bridge — **預設關閉，見總帳項 13 + §2c** | `production_edm_tracker.py::_try_klt_bridge`（`SFM_EDM_KLT_BRIDGE_INTERVAL` 預設 0，`=3` 或 `=100` 開），`test_klt_bridge.py` | sequential 淨賺、production-path interval-3 P168 −47~−59（2026-09-03 重測，4 次重跑）。**2026-09-04：conf-only（`=100`）在 production-path 只有 +4/−5，多 seed 在雜訊內，但 p95 變差 → 仍不改預設。** |
| **async fast/slow 真解耦**（2026-09-05 起 **code default 關**） | `async_localizer.py`（`AnchorSupply` / `SyncCarry` / `FastPath` / `SlowPath` / `Scheduler`）、`edm_localizer_adapter.py::AsyncEDMTrackerAdapter`（`SFM_EDM_ASYNC_TRACKER=1` 開；`SFM_EDM_ASYNC_INLINE_SYNC=0` 關 inline 地板），`test_async_localizer.py`（32）、`test_async_adapter.py`（13） | 2026-09-04 已轉預設開（見 §0）＋同日 inline-sync floor（slow 來不及答時當幀跑同步 keyframe，async 永不差於 sync）、drive-lock（inline 與 background keyframe 互斥＋drain 舊排程，stateful tracker 保序）、sequential replay 預設 pin sync（bit-deterministic gate；production-path 照常用 async）。pp P117：async 370 vs sync 352（common 396 幀，pose p50 5mm）。下一步已做完：slow-prior 回寫（`_apply_fast_prior_feedback`，常開）pp P117 再 +18（398/405＋394/404），sequential 中性。剩餘 7 幀 NEED_REANCHOR 是 fast/slow 皆無幾何可用的真 valley，不是先驗問題；再往下要動 matcher 單-ref 速度（超出本次範圍）或重開已被否決的 reference 政策（需新證據）。 |
| **LOST 近失 acceptance gate** | `production_edm_tracker.py::_relaxed_acquire_gate` / `_count_relaxed_agreement`、`edm_pose_selection.py::count_agreeing_refs`（`acquire_relaxed_min_inliers` **2026-09-05 起預設 66**） | 當時（floor 66，sequential）：P119 +100/+93（雙 seed）、P157 +26、P117 0、**P168 −53/−18 且品質退化** → NOT GO as default。floor 70 零退化但 +29 在雜訊內。要升級需 4 段 × 多 seed，且最好先有 on-map 錯位 corpus。 |
| **starvation-triggered global retry** | `production_edm_tracker.py::_observe_lost_starvation` / `_lost_starvation_armed`（`lost_starved_global_frames` 預設 0） | REJECTED（P119 +2、P157 0）。保留為換地圖時的候選旗標；不要再調 k / corr_max。 |

**2026-09-02 已移除（0 引用、未接線的 WIP 空殼；此處即簡潔紀錄）：**
`async_pipeline.py`（KLT/EDM 解耦 stub，行 62/103 placeholder，從未 wire）、`klt_tracker.py`（inline KLT
抽出，未接）、`velocity_estimator.py`（只被 async_pipeline 引用）、`local_map_manager.py`、
`sim3_alignment.py`、`telemetry_sync.py`、`replay_system.py`、`evaluation.py`。要重做時從 git
歷史（commit 之前的分支狀態）取回。KLT/EDM 快慢路徑解耦（原 §2c）已於 2026-09-03/04 以 `async_localizer.py`
重做並實測（見上表），不再是未實作的開放題。

---

## 4. Tier 4 — 前置阻擋項（不是演算法，但擋著升級）

- **無獨立 ANAFI camera-pipeline holdout。** flight release `validation: NONE`；總帳每一項都標同資料集單幀 smoke。任何 flight 核准前必須補。**hard-negative 現況（2026-09-04 更新）：off-map corpus 已有可用配方 —— 河濱影片 vs 烏來地圖（`控制介面程式/site_profiles/urai_edm.json`，700 幀）穩定 0/700、全程 BOOT_INIT、最佳 PnP inliers 上限 17，可當 acceptance-gate 改動的 off-map 迴歸。仍缺兩種：(1) 開機在圖上、中途離圖的序列（能真正走到 LOST 路徑的 off-map corpus），(2) 同地圖錯位 ground truth（P119 植被假鎖那一類）。**
- ~~**地圖 1045 refs 的地圖相依參數未依 B/C 重算**~~ **已補（2026-09-02，總帳「Tier 4 — 1045-ref 地圖相依參數重驗」）：** `S=2.3346915` 從 1045 ref centers 重算 exact 相符；`radius=max_jump=0.40·S=0.9338766` 在 P168 700f 掃描（0.20–0.60·S）為峰值，較小傷 recovery、較大中性且略慢；`reference_feature_cache_size=192` 在 P168/P117 cache 64/192 trace byte-identical（working set 75 / 36 distinct refs），full-1045 在 8 GiB GPU 不可行但 cache miss 只是重抓不影響結果。**無參數變動 → SHA chain 不動。** 仍缺的是上一項的獨立 holdout + hard-negative。
- **`docs/imu_odometry_capability_audit.md`** 是 ESEKF/IMU 相關的既有稽核，接 ESEKF 前先讀。ESEKF/KLT-3D 的 live gate 見 `docs/esekf_live_eval_runbook.md`。

---

## 5. 驗證 gate 配方（改編自總帳「地圖更換重做流程 D」）

### 5a. 七段 720p 語料庫（2026-09-05 起的標準回歸集）

單段 gate 只證明單段。async 預設就是靠 P117/P168 兩段過關，卻在 P116/P119/P157/P167 崩掉
（見總帳 2026-09-05）。**任何要改預設的政策改動，先跑七段。**

```bash
# 一次性：把測試影片降到 720p（ANAFI 實機串流的解析度）
mkdir -p 模擬器/測試影片/720p
for f in P1160116 P1180118 P1190119 P1570157 P1670167 P1680168 河濱_P1170117; do
  ffmpeg -nostdin -v error -stats -y -i "模擬器/測試影片/$f.MP4" \
    -vf "scale=1280:720:flags=lanczos" -an -c:v libx264 -preset medium -crf 18 \
    -pix_fmt yuv420p -x264-params keyint=48:min-keyint=48:scenecut=0 \
    "模擬器/測試影片/720p/${f}_720p.MP4"
done

# 每個變體跑一輪（約 12 分鐘 / 輪，RTX 5060）
PY=/home/allen/localization/.venv/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
for v in 河濱_P1170117 P1160116 P1180118 P1190119 P1570157 P1670167 P1680168; do
  $PY 定位演算法/validation/benchmark_edm_site_replay.py \
    --site-profile 地圖檔/場域/river_site/site_profile.json \
    --video "模擬器/測試影片/720p/${v}_720p.MP4" \
    --stride 3 --pnp-random-seed 0 --require-cuda --gpu-span \
    --worker-mode production-path --out "outputs/<tag>/$v.json"
done
```

**兩個必看數字，不能只看 successes：**

- `summary.successes` —— 但 KLT-bridged 幀也算成功。
- **verified** = `rows[]` 中 `success` 且 `inliers > 0` 且 `candidate_mode` 不是 `klt_fast` /
  `klt_bridge` / `None` 的幀數。這才是「這一幀真的重新對上地圖」。async 預設開時七段
  successes 62.9% 但 verified 只有 7.5%，其中 P168 有一段連續 710 幀（stride 3 約 89 秒）
  完全靠光流推算卻回報成功。**兩個都要報。**

兩個數字連同失敗幀的 inlier 分桶由這支工具算（`--failures` 出分桶表）：

```bash
$PY 定位演算法/validation/summarize_corpus_runs.py \
  outputs/<baseline-tag> outputs/<candidate-tag> --failures
```

### 5b. 單段 gate 配方（原有）

```bash
PY=/home/allen/localization/.venv/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

# baseline = 現行 flight profile
$PY 定位演算法/validation/benchmark_edm_site_replay.py \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --video 模擬器/測試影片/P1680168.MP4 \
  --stride 3 --max-frames 700 --pnp-random-seed 0 --require-cuda --gpu-span \
  --out outputs/<name>_baseline.json

# candidate（CLI override 或 env；例：temporal off）
$PY 定位演算法/validation/benchmark_edm_site_replay.py \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --video 模擬器/測試影片/P1680168.MP4 \
  --stride 3 --max-frames 700 --pnp-random-seed 0 --require-cuda --gpu-span \
  --no-use-temporal-reference \
  --out outputs/<name>_candidate.json

# 重跑一次 P117：--video 模擬器/測試影片/河濱_P1170117.MP4（不加 --max-frames）
```

注意：`--quality-baseline` 需要帶 `thresholds` 物件的 baseline 檔,不能直接餵 `--out` 產生的 JSON
（會 `ValueError: quality baseline must contain a thresholds object`）。用兩個 `--out` 檔手動比對
`summary.successes` / `state_counts` / `wall_ms` / `inliers` / `reproj_rms` 即可,方法論與總帳一致。
每幀 exact 比對讀 `rows[]`（欄位 `success` / `mode` / `inliers` / `reproj_rms` / `n_corr` / `refs` /
`pose_xyz`）。

准入：

- **map-independent exact 優化**（neck expand、fine_matching sort、matcher 內部）：`m_bids` / `mkpts0_f` /
  `mkpts1_f` / `mconf` 逐元素相等；`reference_trace_sha256`、`frame_schedule_sha256`、state、inliers、pose
  完全相同；synchronized GPU timing 與 wall p50/p95 都改善才宣稱更快。
- **map-dependent policy**（temporal off、quality weight、recovery topk、ESEKF predictor）：完整 replay
  不得降低已核准的 recovery / quality gate；hard-negative replay 不得新增誤鎖；success 不得下降。
- production-path 另查 `source-frame age`、`coalesce drops`、`ready/first-result latency`，不能只看 inference wall time。
- PnP RANSAC seed 固定 0；來源影格排程固定。

---

## 6. 執行環境

- 本機 **有完整 stack**：repo venv `/home/allen/localization/.venv`（Python 3.10.12，torch 2.11.0+cu128，
  CUDA on RTX 5060 Laptop 8151 MiB，pycolmap 4.0.4，cv2 4.13.0），1045-ref bundle、feature-bank
  shards、P168/P117 測試影片都在。**跑測試 / gate 一律用 `/home/allen/localization/.venv/bin/python`,
  不要用系統 `python3`（那個沒有 torch）。**
- 全套 pytest：`.venv/bin/python -m pytest -q`（2026-09-02：`2301 passed / 3 skipped / 0 failed`）。
- gate 指令見 §5；原始 JSON 放 `outputs/`（不進版控），數值結論寫進總帳。

# KLT 延伸進 LOST — 實驗記錄

日期：2026-09-05
分支：`agent/localization-runtime-optimizations`
旗標：`SFM_EDM_KLT_LOST_PREDICT`（預設 0）、`SFM_EDM_KLT_MAX_AGE_S` / `SFM_EDM_KLT_MAX_FRAMES`、
`SFM_EDM_KLT_SHADOW_EVAL`（僅離線評估用）

## 起點：現況比想像的更空

`ProductionEDMTracker` 在一幀視覺定位失敗時有兩條預測分支，**兩條都排除 LOST**：

| 分支 | 條件 | `prediction_mode` |
|---|---|---|
| ESEKF | `state in {TRACK, WEAK_TRACK}` 且 `prediction_allowed()` | `esekf` |
| KLT | `not esekf_done` 且 `state != "LOST"` | `klt_pnp` |

所以 **LOST 期間完全沒有任何位置輸出** —— 不是「只剩慣性」，是什麼都沒有。
操作介面上看到的紅點（`pose_status == "PREDICTED_ONLY"`）其實來自 TRACK / WEAK_TRACK
的失敗幀，不是 LOST。

## 為什麼 KLT 在這裡有機會

`_track_klt_prior` 不是航位推算。它對**固定的 3D 錨點集**（上一次成功 PnP 的 inliers）
重新解 PnP，只有 2D 觀測隨光流前進。所以誤差不像 IMU 積分那樣累積，而是受
特徵點流失與既有品質閘（`weak_min_inliers`、`max_reproj_error_track`、`max_jump`）自然限制。

這代表現行的 6 幀 / 0.50 秒上限**可能是保守的**，值得實測而不是假設。

## E1 — 天花板：LOST 幀有多少落在時窗內

免費計算，直接讀既有七段語料庫（`outputs/corpus_20260905/sync`，815 個 LOST 幀）。
影格間距實測 p50 = 0.1251 s，正好 8 Hz，與真機定位速率一致 —— **0.5 秒只有 4 幀**。

| 時窗 | 涵蓋 LOST 幀 | 比例 | ≈幀數 |
|---|---|---|---|
| 0.5 s（現行上限） | 37 | **4.5%** | 4 |
| 1.0 s | 179 | 22.0% | 8 |
| 2.0 s | 383 | 47.0% | 16 |
| 4.0 s | 611 | 75.0% | 32 |
| 8.0 s | 733 | 89.9% | 64 |

LOST 幀距上次成功定位：p50 = 2.13 s、p90 = 8.01 s、max = 17.6 s。

**結論：照現行上限把 KLT 打開，最多只能影響 4.5% 的 LOST 幀。**
這個想法的價值幾乎完全取決於能不能把時窗拉長，而那要看漂移。

## E2 — 精度：漂移 vs 時窗

`SFM_EDM_KLT_SHADOW_EVAL=1` 讓 KLT 鏈在**每一個成功幀**上跑一次但不重新播種，
於是它的年齡就像走過一段長 LOST，同時 EDM 持續提供同一幀的真值。
（在真正失敗的幀上沒有真值可比 —— 這是唯一能量測的地方。）
生產快取在 shadow 步驟前後完整存回，所以開這個旗標不可能影響追蹤。

七段語料庫、production-path、stride 3、seed 0，比照 runbook §5a。4504 次 shadow
步進，其中 **92.8% 鏈仍存活**。

| 時窗上限 | 樣本 | p50 | p90 | max | p90/到達容差 | p90/max_jump |
|---|---|---|---|---|---|---|
| 0.25 s | 211 | 0.0082 | 0.0321 | 0.1725 | 0.10 | 0.03 |
| **0.50 s（現行）** | 331 | 0.0137 | 0.0484 | 0.2039 | **0.15** | 0.05 |
| 1.00 s | 461 | 0.0199 | 0.0752 | 0.2634 | 0.24 | 0.08 |
| 2.00 s | 565 | 0.0284 | 0.0956 | 0.2792 | 0.30 | 0.10 |
| 4.00 s | 739 | 0.0410 | 0.1382 | 0.3396 | 0.44 | 0.15 |
| 8.00 s | 719 | 0.0459 | 0.1039 | 0.3361 | 0.33 | 0.11 |
| 無限 | 1153 | 0.0258 | 0.0981 | 0.2279 | 0.31 | 0.11 |

**漂移是平的，不是發散的。** p90 從 0.5 秒的 0.048 只長到 4 秒的 0.138，之後不再增加
（8 秒與無限反而更低）。這與機制一致：PnP 每幀對固定 3D 錨點集重解，誤差不積分；
鏈要嘛活著而且準，要嘛被既有品質閘殺掉。

**存活者偏差（無法消除）**：shadow 只能在 EDM 成功的幀上量，因為只有那裡有真值。
真正的 LOST 之所以發生就是因為那些畫面更難，所以上表天生偏樂觀，8 秒/無限那兩列
尤其是「能撐那麼久的鏈本來就在紋理好的場景」。此偏差用離線資料消不掉。

## E3 — 回歸：打開 LOST 預測會不會動到追蹤

預測分支只寫入回報用的 info dict，不改 `self.st`，所以**理論上**成功率不該變。
唯一的副作用是 `_track_klt_prior` 會推進 / 清掉 KLT 快取，而該快取在 bridge 關閉時
（預設）沒有其他消費者。E3 就是用來證實這一點，而不是相信它。

### E3-a：LOST 預測實際產生了幾幀 —— **0**

| arm | PREDICTED_ONLY | 其中 `klt_pnp` | 其中 LOST 狀態的 `klt_pnp` |
|---|---|---|---|
| base | 364 | 14 | **0** |
| lost | 404 | 20 | **0** |

`SFM_EDM_KLT_LOST_PREDICT=1` 在整個 5,794 幀語料庫上**一幀 LOST 預測都沒產生**。

**機制原因（決定性）**：`_on_miss` 在 WEAK_TRACK → LOST 的轉換當下就呼叫
`_clear_klt_cache()`（`production_edm_tracker.py`，與 `_clear_visual_motion_cache()`
並列）。所以進到 LOST 時 `_klt_2d / _klt_3d / _klt_gray` 全是 None，
`_klt_prior_allowed` 永遠回 False —— 這個開關沒有東西可以放行。

要讓它真的動起來，必須**同時**停止在進入 LOST 時清快取。那是在動一個刻意的
陳舊狀態清理決定，而 E1 已經說明報酬上限只有 4.5%。

### E3-b：LOST 其實早就有預測了

先前記錄「LOST 期間沒有任何位置輸出」是**錯的**。`pose_guided` 控制器
（`_on_miss` 裡 `controller.predicted_only(...)`，條件只是
`pose_status != VISUALLY_CONFIRMED`）在 LOST 照樣出手：

| prediction_mode | base 的 `LOST->LOST` 幀 | lost arm |
|---|---|---|
| `visual_velocity` | **136** | 155 |
| `esekf` | 0 | 0 |
| `klt_pnp` | 0 | 0 |

也就是說 LOST 的預測缺口早就被 `pose_guided` 的 `visual_velocity` 補著，
KLT 進來能加的是 0。

### E3-c：回歸 —— 在雜訊內，但這個 harness 量不了逐位元

`base/P1160116` 該次跑壞（0/266 成功、p50 wall 348 ms vs 正常 25 ms），
是被同時執行的操作介面搶 GPU 造成的，**已從統計排除**。其餘六段：

| | base | lost | 差 |
|---|---|---|---|
| verified | 4551 / 5528 (82.3%) | 4575 / 5551 (82.4%) | **+0.09 pp** |

**逐位元對照失敗，但原因不是這個改動**：七段的 `frame_schedule_sha256` 全部不同，
而且兩 arm 解出的影格數本來就不一樣（如 P168 是 1762 vs 1764）。兩者只差一個
布林旗標，而該旗標證實沒有產生任何預測，所以**這個 replay harness 在此設定下
不是逐幀確定性的**。這代表總帳的「相同 reference trace」准入條件無法直接套用，
單段成功率的雜訊底線約 ±30 幀（P1180118 −19、P1190119 +29）。

## 判讀尺度

漂移要對照場域校準值（`地圖檔/場域/river_site/site_profile.json`）：

- 到達容差 `arrival_tolerance_map_units` = 0.3175 u
- 單幀允許跳動 `max_jump` = 0.9339 u
- 路線編輯器的到達球預設 0.02 u（範圍 0.005–0.05）

## 重跑方式

```bash
bash outputs/klt_lost_20260905/run_arms.sh          # 三組 arm，約 60 分鐘
.venv/bin/python 定位演算法/validation/analyze_klt_lost_prediction.py \
  --shadow outputs/klt_lost_20260905/shadow \
  --base   outputs/klt_lost_20260905/base \
  --lost   outputs/klt_lost_20260905/lost
```


## 結論：NO-GO

1. **開關無效**：進入 LOST 時 KLT 快取被清掉，實測 0 幀 LOST 預測。
2. **缺口不存在**：LOST 已由 `pose_guided` 的 `visual_velocity` 覆蓋（136 幀）。
3. **上限太低**：即使拆掉清快取那行，E1 算出現行時窗只能碰到 4.5% 的 LOST 幀。

旗標保留在關閉狀態，作為已量測的否決記錄；shadow 評估器保留，它量出的
「KLT 漂移不隨時間發散」對未來任何光流外推提案都是可重用的證據。

**不要在沒有新機制的情況下重跑本實驗。** 若要再碰這塊，先回答的是
「進入 LOST 時該不該保留視覺快取」，而不是「要不要讓 KLT 在 LOST 跑」。

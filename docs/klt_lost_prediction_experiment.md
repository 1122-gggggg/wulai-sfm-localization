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

七段語料庫、production-path、stride 3、seed 0，比照 runbook §5a。

<!-- E2_RESULTS -->

## E3 — 回歸：打開 LOST 預測會不會動到追蹤

預測分支只寫入回報用的 info dict，不改 `self.st`，所以**理論上**成功率不該變。
唯一的副作用是 `_track_klt_prior` 會推進 / 清掉 KLT 快取，而該快取在 bridge 關閉時
（預設）沒有其他消費者。E3 就是用來證實這一點，而不是相信它。

<!-- E3_RESULTS -->

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

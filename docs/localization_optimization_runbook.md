# 定位演算法優化 Runbook

最後更新：2026-09-02
分支：`agent/localization-runtime-optimizations`
硬體基準：NVIDIA GeForce RTX 5060 Laptop GPU（8151 MiB，sm_120）

本文件是「還能優化什麼、每項要怎麼驗、目前做到哪」的操作清單。已驗證且保留的優化寫在
`docs/verified_localization_optimization_ledger.md`（總帳），本文件只負責把待辦與進行中的實驗
排序、標註 gate、追蹤狀態。**任何項目在通過對應 gate 之前，不得寫進總帳、不得升級為 flight
profile、不得移除安全 gate。**

---

## 0. 狀態（2026-09-02，跑在 repo venv + RTX 5060）

全套 pytest `2301 passed / 3 skipped`。GPU gate 明細在總帳「2026-09-02 GPU gate 結果」+ 項 12。

| 項目 | 結果 |
|---|---|
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

### 2c. KLT 快路徑 / EDM 慢路徑 解耦（開放題，尚未實作）

原 `async_pipeline.py` stub 已於 2026-09-02 移除（見 §3）。這條方向仍值得做:EDM 低頻非阻塞、
KLT/velocity/PnP 走快路徑。與總帳已否決的「EDM/PnP overlap pipeline（42.67→42.69 ms 無收益）」
不同（那是 PnP 重疊）。是大改,需獨立設計 + 完整 replay + production-path source-frame age /
coalesce drops 檢查。

---

## 3. Tier 3 — 本分支進行中的實驗

**仍在樹上：**

| 實驗 | 檔案 | 狀態 |
|---|---|---|
| ESEKF 15 維融合（IMU/NED 速度 + visual pose） | `esekf.py`；接線於 `production_edm_tracker.py`（`__init__`、`localize`、`_predict_center` / `_search_yaw` / `_on_miss` PREDICTED_ONLY 分支） | `__init__` 無條件建 `ESEKF(EKFConfig())`。`predict` 需 `observe_fused_state` 餵 `_latest_velocity_ned`；純 replay 無此來源 → `prediction_allowed()` 維持 False → 分支不觸發。**replay gate 評不到**,608/700 基準已含這條休眠路徑,效果要 live telemetry。 |
| KLT 3D-aware init（`OPTFLOW_USE_INITIAL_FLOW` + covariance window 15→41） | `production_edm_tracker.py::_track_klt_prior` | 只有 `self.esekf.prediction_allowed()` 才啟用 → replay 也休眠,走 fallback LK。`raise StopIteration` 已移除。`test_klt_miss_prior.py` 通過。 |
| EDM neck `repeat→expand` | `runtime/EDM/src/edm/neck/neck.py`（`SFM_EDM_NECK_NO_EXPAND=1` 逃生閥） | **exact 已驗證**（700 幀 0 diff），**但 B=2 無加速**（match_ms 17.89 vs 17.55）。保留 guarded,不列已驗證優化。 |
| `megaloc_token_reduction.py` | 接線於 `reloc_localizer_edm.py`（預設 off） | 總帳已否決 L2/EViT token reduction。工具留著,不預設開。 |
| `fine_matching.py` bi-directional `m_bids` stable-sort | `runtime/EDM/src/edm/head/fine_matching.py` | 修 `bs>1` 時 `m_bids` 非單調 → `_split_match_outputs` 邊界錯誤。像 correctness fix,未單獨 exact 驗證。 |

**2026-09-02 已移除（0 引用、未接線的 WIP 空殼；此處即簡潔紀錄）：**
`async_pipeline.py`（KLT/EDM 解耦 stub，行 62/103 placeholder，從未 wire）、`klt_tracker.py`（inline KLT
抽出，未接）、`velocity_estimator.py`（只被 async_pipeline 引用）、`local_map_manager.py`、
`sim3_alignment.py`、`telemetry_sync.py`、`replay_system.py`、`evaluation.py`。要重做時從 git
歷史（commit 之前的分支狀態）取回。KLT/EDM 快慢路徑解耦（原 §2c）仍是未實作的開放題。

---

## 4. Tier 4 — 前置阻擋項（不是演算法，但擋著升級）

- **無獨立 ANAFI camera-pipeline holdout。** flight release `validation: NONE`；總帳每一項都標同資料集單幀 smoke。任何 flight 核准前必須補。
- **地圖已是 1045 refs**（總帳頂部），但 `S=2.3347`、`radius=max_jump=0.40·S=0.9339`、feature-cache size（192 是在 340-ref 地圖調的）都還沒依總帳「地圖更換重做流程 B/C」重算。P168/P117 兩段 holdout 顯示 working set ~152 distinct refs、cache-192 與 cache-1045 trace exact-equal，所以 192 對「這兩段影片」夠用，但不等於全地圖已驗。
- **`docs/imu_odometry_capability_audit.md`** 是 ESEKF/IMU 相關的既有稽核，接 ESEKF 前先讀。

---

## 5. 驗證 gate 配方（改編自總帳「地圖更換重做流程 D」）

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

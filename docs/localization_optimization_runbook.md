# 定位演算法優化 Runbook

最後更新：2026-09-02
分支：`agent/localization-runtime-optimizations`
硬體基準：NVIDIA GeForce RTX 5060 Laptop GPU（8151 MiB，sm_120）

本文件是「還能優化什麼、每項要怎麼驗、目前做到哪」的操作清單。已驗證且保留的優化寫在
`docs/verified_localization_optimization_ledger.md`（總帳），本文件只負責把待辦與進行中的實驗
排序、標註 gate、追蹤狀態。**任何項目在通過對應 gate 之前，不得寫進總帳、不得升級為 flight
profile、不得移除安全 gate。**

---

## 0. 本輪（2026-09-02）已完成的安全子集

| 項目 | 檔案 | 狀態 |
|---|---|---|
| `_track_klt_prior` 去掉 `raise StopIteration` 控制流 | `定位演算法/deploy_code/sfm_glomap_deploy/production_edm_tracker.py`（`_track_klt_prior`） | 完成，行為等價（3D-aware LK 成功時 `nxt/stf` 已綁定，fallback 段 `try: nxt except NameError` 直接重用；失敗時 `except Exception` 走 fallback）。`py_compile` 通過。 |
| 1045-ref temporal-off + quality-0.5 候選 profile | `定位演算法/configs/edm_profiles/river_gluemap_all8_direct_20260831_temporal_off_quality_candidate.json` | 完成，`json.load` 通過；schema key 全部落在 `edm_profile.py` 的 required∪optional 內。標記 candidate / NOT flight approved。 |
| Reference feature cache 8GiB 預算對 CLI override 失效（SUMMARY.md s1_cache1045） | `定位演算法/validation/benchmark_edm_site_replay.py::_apply_reference_feature_cache_overrides`（行 376-411） | 已在工作樹修好：override 先跑 `_validate_matcher_cache_budget` 再 mutate `reference_feature_cache_size` / `_feature_cache_capacity["map"]`，另有 fallback 內嵌檢查。本輪補上回歸測試。 |
| 回歸測試 | `定位演算法/validation/tests/test_validation_benchmark_helpers.py`（`test_reference_feature_cache_override_*`、`test_host_reference_feature_cache_override_*`） | 完成。用 repo venv（`/home/allen/localization/.venv`，torch 2.11.0+cu128）跑全套 `2301 passed / 3 skipped / 0 failed`。 |
| `SFM_EDM_NECK_NO_EXPAND` env toggle | `定位演算法/deploy_code/runtime/EDM/src/edm/neck/neck.py` | 新增，強制回舊的 2*B unary-CNN 路徑，供 exact A/B。預設不設 = 行為不變。用於 2026-09-02 的 neck expand exact 驗證（見總帳）。 |
| GPU gate（P168 700f + P117 全段） | `outputs/tier1_gate_20260902/`（本機） | 完成，見總帳「2026-09-02 GPU gate 結果」。 |
| EDMConfig code default 同步 | `production_edm_tracker.py:173`、`:234` | 無需改動：`use_temporal_reference` 已是 `False`、`reference_quality_weight` 已是 `0.5`。 |
| EDM 工具包 / deploy 雙份同步 | `edm_matcher.py`、`reloc_localizer_edm.py`、`production_localizer_factory.py`、`edm_pose_selection.py`、`edm_profile.py` | 本輪未改動這些雙份檔案；`diff` 兩棵樹目前 0 行差異。若後續改到其中任一，兩份必須一起改並跑對應測試（總帳「實作同步邊界」）。 |

---

## 1. Tier 1 — 已有證據、只差版本發布時同步進 flight profile（風險最低）

活躍 river flight profile：
`地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260831/{localization,compat}/edm_runtime_profile.json`
（SHA-256 `93e0c2d1…`，兩份內容相同；`site_profile.json`（頂層 + release）`asset_sha256.localizer_profile`
記同一 hash）。

### 1a. `use_temporal_reference: true -> false` — 2026-09-02 gate REJECTED

- 2026-09-02 在現行 committed 樹跑了完整 gate（P168 700f + P117 全段，baseline = 現行 flight profile）：
  - P168：536 → 556（+20），p95 107.78 → 79.36 ms，p50 23.33 → 35.47（+12）
  - P117：366 → **346（−20）**，LOST 42 → **54（+12）**，p50 23.75 → 34.86（+11），p95 平
- 分裂結果，P117 退化 5 個百分點。違反准入規則。**flight profile 維持 `true`，候選 profile 標記 REJECTED。**
- 總帳項 11 的 `s3_no_temporal` +34 建立在 2026-09-01 舊樹（matcher rework 前，s0=486），現行基準已是 536，那個結論不再有效。
- 原始 JSON：`outputs/tier1_gate_20260902/`（本機，`outputs/` 不進版控）。

### 1b. `reference_quality_weight` 0.5

- 已是 EDMConfig code default，flight profile 未寫此 key → 生效值就是 0.5，2026-09-02 baseline 與 candidate 都在跑。
- 這個權重本身沒有被 gate 否定；被否定的是與它綁在同一支候選 profile 的 temporal-off。
- 若日後要「pin against code-default 漂移」，可在下次發版時把 `reference_quality_weight: 0.5` 明寫進 flight profile（`EDM_OPTIONAL_TRACKER_KEYS` 允許），這是純文件化、不改行為。

### 1c. 下一步

- Tier 1 目前沒有可升級的東西。若 Tier 2（recovery 路徑）改善後,可重跑 1a 的 gate 看 P117 是否還退化。

### 發布流程（總帳「E. 升級與記錄」的子集；目前 Tier 1 無升級標的，保留供未來 within-map profile 修正用）

1. 改 `releases/river_gluemap_all8_direct_20260831/localization/edm_runtime_profile.json` 與 `.../compat/edm_runtime_profile.json`（兩份保持相同內容）。
2. `sha256sum` 兩份 → 應仍相同 → 更新：
   - `地圖檔/場域/river_site/site_profile.json` 的 `asset_sha256.localizer_profile`
   - `地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260831/site_profile.json` 的 `asset_sha256.localizer_profile`
   - 根目錄 `SHA256SUMS`
   - `MANIFEST.tsv`
   - 總帳項 10、11 的 SHA 與「現況」段
3. 跑 `tools/system_validation.py`、package manifest 檢查、固定 replay、UI smoke。
4. 候選 profile `river_gluemap_all8_direct_20260831_temporal_off_quality_candidate.json` 的 tracker 區塊就是目標內容，可直接對照。

---

## 2. Tier 2 — 結構性、證據指向但尚未解掉

### 2a. recovery ↔ latency 前緣

現行 committed 樹基準（2026-09-02）：P168 536/700 @ p50 23.33 ms、P117 366/416 @ p50 23.75 ms。
temporal-off gate 顯示「多拿 P168 successes」會連帶 P117 退化 + p50 +12 ms,所以單純調 recovery 力道
不是免費的。

**已經是 state-conditional 的**（不用再做）：top-k（`local_topk=1` TRACK / `weak_local_topk=3` /
`lost_local_topk=5`,`production_edm_tracker.py:1748-1751`）、min_inliers 門檻（`:1770`）。

**還沒 state-conditional 但價值低**：`max_corr_total`、`pnp_workers` — 兩者在 PnP 路徑,`pnp p50 1.34 ms`,
不是瓶頸。

**真正的槓桿是 reference 選擇 policy**（`acquire_stage_mode`、temporal、quality weight、
`lost_global_retrieval_interval`）—— 這正是 temporal-off gate 測的東西,分裂結果。要有進展得找到
「P168 recovery 上升但 P117 不退」的組合,靠 §5 gate A/B,不是寫新功能。

### 2b. EDM detector-free 的單幀成本 ∝ candidate references

`production_edm_tracker.py` 開頭 docstring 自述：XFeat 的 temporal anchor cache（比對「上一幀」並帶其 3D）
在 EDM 沒有等價物，「留到證明有需要再做」。在 WEAK/LOST（跑 3-8 refs）這正是瓶頸段。

- 實驗：WEAK/LOST 時把「上一次成功 PnP 幀」當一個額外 reference，帶著它的 inlier 3D。
- gate：§5，另外要確認不會把漂移的上一幀鎖進 recovery。

### 2c. `async_pipeline.py` 仍是空殼

`定位演算法/deploy_code/sfm_glomap_deploy/async_pipeline.py` 行 62 / 103 是 `placeholder`，
`_edm_loop` / `_klt_loop` 沒有呼叫 `self.edm/klt/pnp/vel/fusion`，只 push `time.monotonic()` 的 dict。
它 docstring 描述的「EDM 低頻非阻塞、KLT/velocity/PnP 快路徑」尚未實作。

- 注意：與總帳已否決的「EDM/PnP overlap pipeline（42.67→42.69 ms 無收益）」不同，那是 PnP 重疊。
- 這裡是 KLT 快路徑 / EDM 慢路徑 的解耦，未被量測過。
- 這是大改，需獨立設計 + 完整 replay + production-path source-frame age / coalesce drops 檢查。

---

## 3. Tier 3 — 本分支進行中的實驗（未跑 gate、未進總帳）

以下多數程式碼已寫在分支上（部分 untracked）。每項都要：預設 off / 有 guard、單獨 commit、
過 §5 gate（exact 類要 matcher 陣列 + reference trace + inlier + pose 完全相同；policy 類要完整
replay 不降 recovery/quality gate）才可宣稱「已驗證」。

| 實驗 | 檔案 | 目前狀態 / guard | gate 狀態（2026-09-02） |
|---|---|---|---|
| ESEKF 15 維融合（IMU/NED 速度 + visual pose） | `esekf.py`；接線於 `production_edm_tracker.py`（`__init__` ~859-887、`localize` ~3327-3340、`_predict_center` / `_search_yaw` / `_on_miss` PREDICTED_ONLY 分支） | **在 `__init__` 無條件實例化**（import 成功即建 `ESEKF(EKFConfig())`）。`predict` 需 `observe_fused_state` 餵 `_latest_velocity_ned`，純 replay 沒這來源 → `prediction_allowed()` 維持 False → ESEKF 分支不觸發 | **replay gate 評不到**：536/700 基準已含這條休眠路徑。實際效果要 live telemetry。不能宣稱已驗證,也沒有 replay 退化證據。 |
| KLT 3D-aware init（`OPTFLOW_USE_INITIAL_FLOW` + 依 covariance trace 放大 window 15→41） | `production_edm_tracker.py::_track_klt_prior` | 只有 `self.esekf` 存在且 `prediction_allowed()` 才啟用。因 ESEKF 在 replay 休眠 → 3D-aware 分支在 replay 也不觸發，走 fallback LK。`raise StopIteration` 控制流已移除 | `tests/localization/deploy/test_klt_miss_prior.py` 通過（在 2301 全綠內）。3D-aware 路徑本身要 live telemetry 才測得到。 |
| EDM neck `repeat -> expand` | `定位演算法/deploy_code/runtime/EDM/src/edm/neck/neck.py`（`CIM.forward`）、`.../edm/edm.py`、`.../edm/head/fine_matching.py` | shape-driven `duplicated`（B>1）分流；新增 `SFM_EDM_NECK_NO_EXPAND=1` 逃生閥 | **exact 已驗證**：P168 700 幀 A/B（expand ON vs `SFM_EDM_NECK_NO_EXPAND=1`）0 個欄位差,536 pose `max|Δ|=0`。**但 `match_ms` p50 17.89 vs 17.55 — expand 在 B=2 略慢,不是 TRACK 加速。** 數值安全可保留,不列為已驗證優化。詳見總帳。 |
| `async_pipeline.py` | 見 §2c | 空殼、無接線 | 見 §2c |
| `megaloc_token_reduction.py` | untracked | 無接線、無測試 | 總帳已否決 MegaLoc L2/EViT token reduction（ranking / 端到端 replay 退化）。要重新做必須端到端不退化。 |
| `local_map_manager.py` / `velocity_estimator.py` / `gnss_prior.py` / `sim3_alignment.py` / `telemetry_sync.py` / `replay_system.py` / `evaluation.py` | untracked | 接線 / 用途 / 測試狀態不明 | 逐一釐清是否被 runtime 路徑引用；未引用者不列入 runtime 優化，先歸類為工具/實驗。 |
| `fine_matching.py` bi-directional 遮罩後 `m_bids` 重新 stable-sort | `定位演算法/deploy_code/runtime/EDM/src/edm/head/fine_matching.py`（已改未提交） | 修正 `bs>1` 時 `m_bids` 非單調導致 `EDMMatcher._split_match_outputs` 邊界錯誤 | 這像是 correctness fix 不是 perf。要 exact test：多 reference batch 下 split 後每 ref 的 match 集合與未排序版一致（只重排不增減）。 |

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

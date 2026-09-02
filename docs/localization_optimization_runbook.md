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
| 回歸測試 | `定位演算法/validation/tests/test_validation_benchmark_helpers.py`（`test_reference_feature_cache_override_*`、`test_host_reference_feature_cache_override_*`） | 完成，`py_compile` 通過。**需 torch 環境才能執行**（`benchmark_edm_site_replay.py` 於 module import 就 `import torch`），本機無 torch 未實跑；與該檔既有 17 個測試同一限制。 |
| EDMConfig code default 同步 | `production_edm_tracker.py:173`、`:234` | 無需改動：`use_temporal_reference` 已是 `False`、`reference_quality_weight` 已是 `0.5`。 |
| EDM 工具包 / deploy 雙份同步 | `edm_matcher.py`、`reloc_localizer_edm.py`、`production_localizer_factory.py`、`edm_pose_selection.py`、`edm_profile.py` | 本輪未改動這些雙份檔案；`diff` 兩棵樹目前 0 行差異。若後續改到其中任一，兩份必須一起改並跑對應測試（總帳「實作同步邊界」）。 |

---

## 1. Tier 1 — 已有證據、只差版本發布時同步進 flight profile（風險最低）

活躍 river flight profile：
`地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260831/{localization,compat}/edm_runtime_profile.json`
（SHA-256 `93e0c2d1…`，兩份內容相同；`site_profile.json`（頂層 + release）`asset_sha256.localizer_profile`
記同一 hash）。

### 1a. `use_temporal_reference: true -> false`

- 證據（總帳項 11）：`s3_no_temporal` 520/700（+34 vs s0 486），LOST 168→113，wall p95 109.56→91.32 ms。
- 現況：EDMConfig code default 已是 `False`，但 flight profile JSON 仍明寫 `true`，override 掉 code default。實際飛行走的是較慢、LOST 較多的 temporal-on 路徑。
- 待辦：下次版本發布時，把兩份 `edm_runtime_profile.json` 的 `use_temporal_reference` 改 `false`。

### 1b. `reference_quality_weight` 明寫 `0.5`

- 證據（總帳項 10）：`a2_quality` 508/700（+22 vs s0），wall p95 89.87 ms（全 run 最低尾延遲），LOST 168→134。
- 現況：flight profile 未寫此 key → 生效值已是 code default 0.5。功能上「已上線」，但沒 pin，未來 code default 漂移就會靜默改變飛行行為。
- 待辦：發布時把 `reference_quality_weight: 0.5` 明寫進 profile（`EDM_OPTIONAL_TRACKER_KEYS` 允許）。

### 1c. 兩項疊加（未測）

- `s3_no_temporal` 與 `a2_quality` 各自對 s0 測過，**兩項一起開的合併 profile 沒跑過任何 holdout。**
- 待辦：用 §5 gate 跑 `河濱 P168`（700 frame）+ `P117` 全段，baseline = 現行 flight profile。

### 發布流程（總帳「E. 升級與記錄」的子集，這裡是 within-map profile 修正，不是換地圖）

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

現行 flight-ish 預設 486/700 @ wall p50 25 ms；recovery-strong 變體（`conservative` 521、`s3_no_temporal` 520、`a2_quality` 508）把 LOST 從 24% 壓到 16-19%，代價是 p50 +10~17 ms。

目標：**只有進 WEAK/LOST 才加 reference / 拉高 top-k，TRACK 維持 topk=1。** 現行機制已有 `weak_local_topk` /
`lost_local_topk` / `acquire_stage_mode=progressive`，A/B 顯示還沒到位。可做的實驗：

- state-conditional `max_corr_total` / `pnp_workers`（TRACK 低、WEAK/LOST 高）。
- LOST 專用的 `lost_global_retrieval_interval` 更積極，但 TRACK 不受影響。
- gate：§5，重點看 p50 不退、LOST 下降、hard-negative 不新增誤鎖。

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

| 實驗 | 檔案 | 目前狀態 / guard | 所需 gate |
|---|---|---|---|
| ESEKF 15 維融合（IMU/NED 速度 + visual pose） | `esekf.py`（untracked, 1211 行）；接線於 `production_edm_tracker.py`（`__init__` ~859-887、`localize` ~3327-3340、`_predict_center` / `_search_yaw` / `_on_miss` PREDICTED_ONLY 分支） | `from esekf import ...` 有 `try/except -> None` fallback；`ESEKF` 未經 `production_localizer_factory` 接線；無專屬測試、無 replay 證據 | 完整 P168 + P117 holdout；predictor 改動屬高風險（`a1_pose_guided` 已硬回退 307/700）。要證明 predicted-only 分支不會把錯誤 pose 餵給 controller。 |
| KLT 3D-aware init（`OPTFLOW_USE_INITIAL_FLOW` + 依 covariance trace 放大 window 15→41） | `production_edm_tracker.py::_track_klt_prior` | 只有 `self.esekf` 存在且 `prediction_allowed()` 才啟用；整段 `except Exception` 包住。`raise StopIteration` 控制流本輪已移除 | `tests/localization/deploy/test_klt_miss_prior.py`（需 pycolmap）＋ P168/P117 replay，確認 3D-aware 路徑不比 fallback LK 差。 |
| EDM neck `repeat -> expand`（query unary CNN 只算一次，省 B-1 次 conv + `edm.py` 的 9.4MB `torch.cat`） | `定位演算法/deploy_code/runtime/EDM/src/edm/neck/neck.py`（`CIM.forward`）、`.../edm/edm.py`、`.../edm/head/fine_matching.py`（已改未提交） | 靠 `duplicated`（B>1 且 shape 對稱）heuristic 分流，B==1 走原路徑 | **總帳「已驗證為無效或尚未驗證」明列：「query tensor repeat 改 expand：尚未做完整 EDM exact + synchronized benchmark，不做」。** 必須先給 `mkpts0/mkpts1/mconf` 逐元素相等 + synchronized GPU timing + 固定 300-frame replay trace hash 相同，才可 land。這是單幀 GPU 最大潛在節省（match p50 ~18-30 ms 是主成本，copy 4.6 / pnp 1.3）。 |
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
# baseline = 現行 flight profile
python 定位演算法/validation/benchmark_edm_site_replay.py \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --video 模擬器/測試影片/P1680168.MP4 \
  --stride 3 --max-frames 700 --pnp-random-seed 0 --require-cuda --gpu-span \
  --out outputs/<name>_baseline.json

# candidate（用 CLI override 或指向候選 profile）
python 定位演算法/validation/benchmark_edm_site_replay.py \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --video 模擬器/測試影片/P1680168.MP4 \
  --stride 3 --max-frames 700 --pnp-random-seed 0 --require-cuda --gpu-span \
  --quality-baseline outputs/<name>_baseline.json \
  --out outputs/<name>_candidate.json
# 重跑一次 P117：模擬器/測試影片/河濱_P1170117.MP4（不加 --max-frames）
```

准入：

- **map-independent exact 優化**（neck expand、fine_matching sort、matcher 內部）：`m_bids` / `mkpts0_f` /
  `mkpts1_f` / `mconf` 逐元素相等；`reference_trace_sha256`、`frame_schedule_sha256`、state、inliers、pose
  完全相同；synchronized GPU timing 與 wall p50/p95 都改善才宣稱更快。
- **map-dependent policy**（temporal off、quality weight、recovery topk、ESEKF predictor）：完整 replay
  不得降低已核准的 recovery / quality gate；hard-negative replay 不得新增誤鎖；success 不得下降。
- production-path 另查 `source-frame age`、`coalesce drops`、`ready/first-result latency`，不能只看 inference wall time。
- PnP RANSAC seed 固定 0；來源影格排程固定。

---

## 6. 已知環境限制

- 本開發機無 `torch` / `pycolmap` / GPU / 1045-ref bundle / 測試影片。`定位演算法/validation/` 與
  `tests/localization/deploy/` 內凡 import `benchmark_edm_site_replay` 或 `production_edm_tracker`
  的測試都需在 CI（有 torch/pycolmap）或 GPU 機執行。
- 因此本輪所有改動只做到 `py_compile` + `json.load` + 靜態 schema 對照；Tier 1c / Tier 3 的
  數值宣稱都需要 GPU 機跑 §5。

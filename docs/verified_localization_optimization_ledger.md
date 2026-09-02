# 已驗證定位優化總帳

最後更新：2026-09-01  
適用硬體：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB VRAM，sm_120

## 用途與准入規則

本文件只列入已保留、已有行為或效能證據的改動。更換地圖時依「地圖更換重做流程」處理；不要直接複製舊場域參數。

准入條件：固定來源影格排程、固定 PnP seed 0；宣稱數值完全相同的改動還必須有相同 reference trace、inlier、pose 與 matcher 輸出。只有微基準加速、但整段回放退化或改變安全結果者，不升級為預設值。

目前 river active-map 身分：

- site profile：`地圖檔/場域/river_site/site_profile.json`
- source archive SHA-256：`dcf3d87ef1e448b95a66ab61e36e8f3a8e94a8fe2132c5128570a64eaf78b879`
- localization bundle SHA-256：`603dc7c473a88f05c968144d584c47e6807218c7fa87f1d7effeb016a2d58dcb`
- EDM runtime profile SHA-256：`93e0c2d166f0378f6bf6af992023174baa86339ee6e91ce0c9292030380dac03`
- persistent feature bank：`執行環境/models/edm_reference_features/603dc7c473a88f05c968_dc24014f8f9e8d05d2f6.pt`
- feature bank SHA-256：`fd86ed9618c2f435f3327e2aa8fd3e323ab7ba4835b2023b17a1d1fdfd98e2fe`
- feature bank 大小：8,937,437,001 bytes
- map frame：`river_gluemap_all8_direct_20260831_d2b8a5304eff`，1,045 references
- 驗證狀態：來源明列 `validation: NONE`；只完成同資料集單幀 smoke，resolver 維持 fail closed。

`outputs/` 是本機驗證證據且預設不進版控；本文件才是長期總帳。刪除 `outputs/` 前要先保留本文件列出的原始 JSON 或把它們移到受治理的驗證收據位置。

## 已驗證且保留的優化

### 1. 場域相機內參與地圖尺度調校

**新地圖現行暫定設定**

- query camera：`PINHOLE 1280x720`
- params：`[960.4853099760471, 958.1961747147875, 670.8167651412149, 358.7191813450141]`
- 場域尺度：`S = 2.3346915245056152`
- 定義：`2·p95(||reference_center - componentwise_median||)`，目前 1,045 張 EDM references
- `radius = max_jump = 0.40·S = 0.9338766098022462`

**目前證據與限制**

- 匯入收據：`outputs/river_gluemap_all8_direct_20260831_import/import_summary.json`
- production runtime 單幀 smoke：1/1 TRACK、120 inliers、1.3312 px reprojection RMS
- 原始證據：`outputs/river_gluemap_all8_direct_20260831_import/persistent_store_smoke.json`
- 這是來源 reference 的同資料集煙霧，不是獨立 holdout，也沒有實際 ANAFI camera-pipeline 證據。
- 舊 340-reference map 的 P167 82.4418% 結論不移植到此地圖。

**地圖相依性：完全相依。** 新 reconstruction、reference bank、座標尺度或相機來源任一改變，都必須重算 `S`、相機內參、`radius`、`max_jump`，並重跑整段回放。現行值在獨立驗證前不得作為品質核准依據。

### 2. 每幀 query backbone feature 只算一次

**改動**

同一 query 對多批 references 時，先 `prepare_query`，之後重用 query 的四層 backbone features；reference 端仍依 cache/store 取得。正式預設 `query_feature_reuse=true`。

**證據**

- 固定 top-20 微基準：463.77 ms → 380.81 ms，1.218x
- `mkpts0`、`mkpts1`、`mconf` 最大絕對差：0
- 固定 300-frame replay：兩邊皆 297/300；0 semantic differences；frame/reference trace hash 相同
- match p50：46.55 ms → 43.84 ms
- 原始證據：
  - `outputs/edm_exact_optimizations/query_reuse_off_300.json`
  - `outputs/edm_exact_optimizations/query_reuse_on_300.json`
  - `outputs/optimization_20260831_report.md`
- 防回歸測試：`定位演算法/validation/tests/test_reference_feature_cache.py::test_prepared_query_features_are_reused_without_changing_matches`

**地圖相依性：不相依。** 新地圖只需重跑 exact matcher test 與一段固定 replay，不需重新調參。

### 3. GPU reference feature cache 192、host spill 0

**改動**

正式預設：

- `reference_feature_cache_size=192`
- `host_reference_feature_cache_size=0`
- temporal cache 維持獨立、有界

**證據：固定 P168 stride-3、700-frame max-precision replay**

| GPU entries | Host entries | 成功 | Wall p50 | Wall p95 | Misses |
|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 520/700 | 49.58 ms | 534.24 ms | 316 |
| 192 | 0 | 520/700 | 46.63 ms | 495.61 ms | 199 |
| 340 | 0 | 520/700 | 46.96 ms | 496.37 ms | 199 |

三組 frame schedule、reference trace、state、inliers、pose 完全相同；192 已涵蓋量測 working set，340 沒有額外收益。

- 原始證據：`outputs/edm_exact_optimizations/cache64_stride3_700.json`、`cache192_stride3_700.json`、`cache340_stride3_700.json`
- 預設值測試：`定位演算法/validation/tests/test_reference_feature_cache.py::test_validated_production_feature_cache_defaults`

**地圖相依性：容量相依。** 192 只對目前 340-reference 地圖的量測 working set 有效。新地圖必須重跑至少 `64 / 192 / 全 reference 數` 三點；只在輸出完全相同時選最低 p50/p95 的最小容量。

### 4. Full-token MegaLoc TensorRT 作為 VPR 預設

**改動**

保留 full-token MegaLoc TensorRT；不使用 token-reduced PyTorch/TensorRT、BoQ 替代或 EDM native TensorRT。

**證據**

- 現有 full-token TensorRT query 約 6.3–8.7 ms。
- L2 keep-0.5 TensorRT 雖把 75-frame VPR p50 由 8.91 ms 降到 5.32 ms，固定 stride-3、700-frame tracker replay 卻由 46.63/495.61 ms 退化為 53.61/546.91 ms wall p50/p95。
- L2 keep-0.9 的 EDM+PnP 成功由 62/75 降為 60/75。
- 原始證據：`outputs/megaloc_token_reduction_trt/`、`outputs/optimization_20260831_report.md`

**地圖相依性：模型執行不相依，reference descriptor bank 相依。** 更換地圖要重建/驗證 bundle 內 `ref_global`，但不要因地圖更換自行切換 query encoder 或 engine。GPU、TensorRT、CUDA 或 model weight 改變時才重建 engine 並重做端到端 gate。

### 5. 持久化 immutable reference backbone feature bank

**改動**

每張地圖 reference 的四層 EDM backbone features 只抽取一次並原子寫入 feature bank。小於等於 3 GiB 的 bank 使用 ZIP+mmap；更大的 bank 改用 PyTorch legacy stream 並以非 mmap 載入，避免 ZIP 32-bit 邊界產生不可讀檔案。兩種格式都使用 `weights_only=True`，bank 同時綁定：
- localization bundle SHA-256
- EDM checkpoint/config/input/fp16/torch 組成的 model key
- reference 名稱順序
- 每張 reference image SHA-256

任一身分不符即拒絕使用，不會錯套另一張地圖。

**證據**

- 單張 persistent restore：4.8435 ms → 1.0267 ms，4.72x
- restored matcher output exact equal
- 程式：`定位演算法/EDM工具包/deploy/edm_matcher.py` 與 portable copy `定位演算法/deploy_code/sfm_glomap_deploy/edm_matcher.py`
- factory binding：兩份 `production_localizer_factory.py`
- 防回歸測試：`定位演算法/validation/tests/test_reference_feature_cache.py::test_persistent_reference_features_restore_exact_matches`
- 大型 stream 載入測試：`定位演算法/validation/tests/test_reference_feature_cache.py::test_legacy_stream_reference_feature_store_loads`
- 新地圖冷啟動煙霧：startup 8.73 s 後 1/1 TRACK、120 inliers；證據 `outputs/river_gluemap_all8_direct_20260831_import/persistent_store_smoke.json`

**地圖相依性：bank 完全相依，機制不相依。** 新 bundle 會得到不同檔名，舊 bank 不可改名冒用。以 `SFM_EDM_BUILD_REFERENCE_FEATURE_STORE=1` 啟動一次正式 factory 建置；完成後關閉該旗標並確認 store 可再次載入。只有驗證新 bank 後才可清理舊 bank。

### 6. Matcher 輸出一次 GPU→CPU packed transfer

**改動**

`m_bids`、`mkpts0_f`、`mkpts1_f`、`mconf` 先在 GPU 打包，執行一次 D2H，再切回 NumPy views，避免四次同步/傳輸。

**證據**

- 同資料 D2H：0.0767 ms → 0.0580 ms
- values 與 batch ids 完全相同
- 防回歸測試：`定位演算法/validation/tests/test_reference_feature_cache.py::test_match_output_pack_preserves_values_and_batch_ids`

**地圖相依性：不相依。** 新地圖只需跑 exact test。

### 7. Bundle mmap 與 reference JPEG 平行解碼

**改動**

- bundle 使用 `torch.load(..., weights_only=True, mmap=True)`
- reference JPEG 以最多 8 個 CPU workers 平行 `cv2.imdecode`
- 載入前仍保留檔案大小、SHA-256、schema、tensor/image budget 與 JPEG 維度檢查

**證據**

- 目前 bundle map load：1323 ms → 773 ms
- decoded images 與 bundle schema 不變
- 程式：`定位演算法/EDM工具包/deploy/reloc_localizer_edm.py::EDMRelocMap.load` 及 portable copy

**地圖相依性：機制不相依，收益隨 reference 數與 JPEG 大小改變。** 新地圖要記錄 cold-load p50/p95；workers 上限除非新證據更快，維持 8。

### 8. Latest-frame shared-memory coalescing與自適應 busy-submit cadence

**改動**

- inference 中只保留一個最新 coalesced frame，不累積多幀 queue
- shared-memory frame 在 coalesce 時不重複 copy，僅 promote 時 copy
- result notification fd 立即喚醒 UI，5 ms polling 僅為 fallback
- busy-worker overwrite cadence 預設自適應為最新 localization latency 的一半，限制 20–100 ms；`SFM_ADAPTIVE_LOC_SUBMIT` 預設開啟

**證據**

- 行為測試：
  - `tests/control_interface/operator_interface/test_worker_lifecycle.py::test_shared_frame_worker_coalesces_latest_slot_and_notifies`
  - `tests/control_interface/operator_interface/test_worker_lifecycle.py::test_shared_frame_coalesce_copies_only_on_promote`
  - `tests/control_interface/operator_interface/test_worker_lifecycle.py::test_adaptive_localization_submit_tracks_half_the_latest_latency`
- 這項證據證明有界 queue、最新影格優先與 cadence 邊界；目前沒有獨立的毫秒加速宣稱。

**地圖相依性：演算法不相依，cadence 會自動跟隨新地圖實際 latency。** 新地圖只需檢查 source-frame age、coalesce drops、submit busy attempts，不能只看 inference wall time。

### 9. Sparse localization JSON telemetry

**改動**

`build_localization_metric_record` 只複製 stable contract 中實際存在的 worker 欄位；缺少欄位不再輸出大量 `null`。顯式 `null` 仍保留，非有限數值轉成 `null`，不改變事件語意。

**證據**

- 代表性 localization record：3871 bytes → 406 bytes
- 防回歸測試：
  - `tests/control_interface/operator_interface/test_localization_metrics.py::test_metric_record_keeps_explicit_null_but_omits_absent_result_fields`
  - `tests/control_interface/operator_interface/test_localization_metrics.py::test_metric_record_replaces_non_finite_values_before_json_output`

**地圖相依性：不相依。** 不需因地圖更換重做。

### 10. Reference quality weight 0.5（P95 尾延遲優化，a2_quality）

**改動**

- `EDMConfig.reference_quality_weight` 預設由 `0.0` 改為 `0.5`（code default，profile 可省略）。
- `SFM_EDM_REFERENCE_QUALITY_WEIGHT` env 仍可覆蓋（`production_edm_tracker.py:767`），已驗證的 river compat profile `edm_runtime_profile.json`（SHA `93e0c2...`）維持不改以避免 SHA 破裂，透過 env 或新 code default 生效。
- `edm_profile.py` 註記此權重為已驗證預設，保持 `EDM_OPTIONAL_TRACKER_KEYS` 可選以相容舊 profile。

**證據：P168 700-frame holdout，stride 3，RTX 5060，2026-09-01**

- `s0_current`（baseline, 權重 0.0）：486/700，wall p50 25.00 ms，p95 109.56 ms，match p50 19.93 ms，inliers p50 84.5，LOST 168，trace `068754ee…`
- `a2_quality`（`SFM_EDM_REFERENCE_QUALITY_WEIGHT=0.5`）：508/700（+22 vs s0），wall p50 24.25 ms，**p95 89.87 ms（全部 run 中最低尾延遲）**，match p50 17.63 ms，inliers p50 80.5，reproj p95 3.137，LOST 168→134，WEAK 40→52，trace `defc3bd4…`，frame_schedule `3bdfc0e1…` 相同
- 原始證據：`outputs/p168_all8_baseline_20260901/a2_quality.json`（及 `.log` / `.exit`）、`outputs/p168_all8_baseline_20260901/SUMMARY.md` 表格
- 對比失敗案例：`a1_pose_guided` 307/700 p95 227 ms 明顯回退，不升級；`a3_composite_rank` 485/700 無收益
- 預設生效驗證：無 env 時 `EDMConfig().reference_quality_weight == 0.5`，`SFM_EDM_REFERENCE_QUALITY_WEIGHT=0.0` 可退回；`ProductionEDMTracker` 初始化時讀 env 並 `validate()` 後覆蓋

**地圖相依性：參考品質分佈相依，但 tail 優化在 1,045-ref river 地圖上已獨立驗證。** 新地圖需重跑 `a2_quality` 對照（0.0 vs 0.5）並比較 p95 與成功率；若新地圖品質訊號不同，可調權重或退回 0.0。

- 2026-09-02：候選 profile `定位演算法/configs/edm_profiles/river_gluemap_all8_direct_20260831_temporal_off_quality_candidate.json` 已把此權重明寫 `0.5`（pin against code-default 漂移）。flight profile 尚未改，等版本發布依「E. 升級與記錄」同步 SHA。

### 11. Temporal reference 預設關閉（s3_no_temporal，+34 gain）

**改動**

- `EDMConfig.use_temporal_reference` 維持 `False` 預設（不再考慮改回 `True`），已加註解指向 `s3_no_temporal` 證據。
- River compat profile 仍為 `"use_temporal_reference": true`，為保持 `site_profile.json` 中 `localizer_profile` SHA `93e0c2...` 不破裂，暫不改 JSON；code default `False` 對未指定 profile 的預設 tracker 生效，新發版再同步 profile。
- 註解位置：`production_edm_tracker.py:162`（`use_temporal_reference` 定義處）

**證據：同 P168 700-frame holdout**

- `s3_no_temporal`（`--no-use-temporal-reference`）：520/700（**+34 vs s0 486/700**），wall p50 35.18 ms，p95 91.32 ms，match p50 29.80 ms，inliers p50 83.0，reproj p95 2.628，LOST 168→113，WEAK 40→61，trace `fee9ab93…`
- 原始證據：`outputs/p168_all8_baseline_20260901/s3_no_temporal.json` 及 `SUMMARY.md`
- 失敗對照：s0 的 temporal on 在此 holdout 較慢且 LOST 更多；保存 `False` 預設避免無收益的 temporal 成本

**地圖相依性：機制不相依但收益隨 reference 密度改變。** 新地圖需重跑 temporal on/off 對照；若新地圖 temporal 確有增益，可於新 compat profile 明確設 `true`。

- 2026-09-02：候選 profile `定位演算法/configs/edm_profiles/river_gluemap_all8_direct_20260831_temporal_off_quality_candidate.json` 已把 `use_temporal_reference` 設 `false`。flight profile（SHA `93e0c2...`）仍為 `true`，等版本發布時同步。**注意：temporal-off 與 quality-0.5 兩項疊加的合併 profile 尚未跑過任何 holdout（見 `docs/localization_optimization_runbook.md` §1c）。**

## 待辦與進行中優化（不在本總帳，另見 runbook）

排序、gate、狀態追蹤在 `docs/localization_optimization_runbook.md`。摘要：

- Tier 1：temporal-off、quality-0.5 已有證據，等版本發布同步進 flight profile（候選 profile 已備）。
- Tier 2：recovery↔latency 前緣（state-conditional reference budget）、EDM 上一幀 anchor、`async_pipeline.py` 空殼。
- Tier 3（本分支進行中、未過 gate、不得宣稱已驗證）：ESEKF 15 維融合、KLT 3D-aware init、EDM neck `repeat→expand`（總帳明列尚未驗證，需 exact + synchronized benchmark）、`fine_matching.py` bi-directional `m_bids` 重排（疑似 correctness fix）。
- 2026-09-02 已做的安全子集：`_track_klt_prior` 移除 `raise StopIteration` 控制流（行為等價）；`benchmark_edm_site_replay._apply_reference_feature_cache_overrides` 的 8GiB 預算對 CLI override 失效（SUMMARY.md s1_cache1045）已修並補回歸測試 `定位演算法/validation/tests/test_validation_benchmark_helpers.py::test_reference_feature_cache_override_*`。

## 現行 profile 中仍需場域重驗的組合設定

下列設定目前存在於 river compat profile，但沒有可攜到新地圖的「單項 exact 加速」結論：

- `query_cuda_graph=true`
- `acquire_stage_mode=progressive`
- `pnp_ranked_batches=true`
- `track_map_first=true`
- `match_batch_size=2`
- `global_retrieval_policy=boot_and_lost_once`
- 場域 inlier/reprojection/jump/yaw/stale-LOST gates

P157/P167 latency候選的確顯著降低 p50，但 reference trace 與狀態路徑有變；P168 的完整 recovery corpus 又拒絕所有 PnP/batch 候選。證據：

- `outputs/edm_latency_optimizations/p157_baseline.json`：wall p50 56.34 ms，2481/3082
- `outputs/edm_latency_optimizations/p157_candidate.json`：wall p50 33.07 ms，2505/3082
- `outputs/edm_latency_optimizations/p167_baseline.json`：wall p50 52.41 ms，1854/2318
- `outputs/edm_latency_optimizations/p167_candidate.json`：wall p50 28.82 ms，1892/2318
- `outputs/edm_p168_20260831/p168_edm_pnp_optimization_20260831/summary.json`：所有候選 rejected；固定 4 frames / 80 refs 的 prepared-query 結果 exact、快 9.5%，但未通過完整 replay gate

因此換地圖時可以把這些設定當候選，不得當成已證明的通用優化直接升級。安全 gate 也不能為了追求成功率或 latency 被移除。

## 已驗證為無效或尚未驗證：不要重做／不要預設開啟

| 項目 | 結論 |
|---|---|
| Temporal feature promotion | TRACK 44.97 → 45.40 ms，較慢；預設 off |
| Inclusive pinned-host cache | cache-192 GPU-only 已覆蓋 working set；預設 off |
| EDM/PnP overlap pipeline | 42.67 → 42.69 ms，無收益；預設 off |
| MegaLoc L2/EViT token reduction | ranking 或端到端 replay 退化；不升級 |
| EDM 640x384 / 640x480 | 雖快 3.39x / 1.70x，但成功 26/30 → 20/30；維持 1024x576 |
| BoQ-ResNet50 VPR replacement | 75-frame final PnP success 降低；不替換 |
| Native EDM TensorRT FP32 | PyTorch FP16 的 0.486x/0.560x，且不 identical；不使用 |
| P168 PnP/batch/cross-stage candidates | recovery gate 退化；已回退 |
| 四層 persistent features 合併成單次 H2D | 尚未實作/同步 benchmark，不做 |
| query tensor `repeat` 改 `expand` | 尚未做完整 EDM exact + synchronized benchmark，不做 |
| reference feature async prefetch | 尚未證明端到端收益，不做 |
| cold-start map/model overlap | 尚未證明 I/O contention 下更快，不做 |
| approved GNSS-map calibration persistence | 是功能/治理工作，不是已驗證效能優化；目前不做 |
| 進一步移除 JSON 欄位或改 schema | 尚未完成 consumer 相容性驗證，不做 |

原始負面實驗詳情：`outputs/optimization_20260831_report.md`、`outputs/edm_p168_20260831/`、`outputs/edm_resolution/`、`outputs/megaloc_token_reduction_*`、`outputs/megaloc_boq_resnet50/`、`outputs/edm_tensorrt/`。

## 地圖更換重做流程

### A. 先固定新場域身分

1. 產生新 `localization_bundle.pt`、reference poses、map alignment、runtime profile、site profile。
2. 寫入並核對每個 asset SHA-256；coordinate-frame id 必須換成新 reconstruction 身分。
3. 記錄 reference 數、reference image SHA、MegaLoc descriptor dimension、EDM input/grid。
4. 相機來源若不同，重新校正 query intrinsics；不要沿用 river 內參。

### B. 重算所有地圖相依參數

1. 從新地圖 reference centers 重算 `S = 2·p95(||center - median||)`。
2. 以 `0.40·S` 作為 `radius/max_jump` 起始候選，不是自動核准值。
3. 重新驗證 local/weak/lost top-k、near pool、covis、inlier/reprojection、jump/yaw 與 stale-LOST gates。
4. 使用新 bundle 建立新的 persistent reference feature bank：

```bash
SFM_EDM_BUILD_REFERENCE_FEATURE_STORE=1 \
python 定位演算法/validation/benchmark_edm_site_replay.py \
  --site-profile "$NEW_SITE_PROFILE" \
  --video "$VALIDATION_VIDEO" \
  --out outputs/edm_feature_store_build_smoke.json \
  --max-frames 1 --pnp-random-seed 0 --require-cuda
```

5. 關閉 build flag 後重跑同一影格，確認 bank identity、persistent hit 與 exact matcher output。舊 bank 保留到新 bank 驗證完成。

### C. 重新量測 cache 與冷啟動

對固定 replay 分別測 `--reference-feature-cache-size 64`、`192`、新地圖全 reference 數，host cache 都先設 0。只有 frame/reference trace、state、inliers、pose 完全相同才比較 wall p50/p95、misses、VRAM。

同時記錄：bundle load、JPEG decode、model load、feature-bank mmap、worker ready latency、first-result latency。不要把 warm page cache 當 cold-start 結果。

### D. 固定回放 gate

至少準備：

- 兩段完整飛行 replay，涵蓋 BOOT、TRACK、WEAK、LOST、reacquire
- 一段 hard-negative / off-map replay
- 固定 video SHA-256、source-frame schedule、camera、PnP seed 0

基準命令：

```bash
python 定位演算法/validation/benchmark_edm_site_replay.py \
  --site-profile "$NEW_SITE_PROFILE" \
  --video "$VIDEO" \
  --out "$BASELINE_JSON" \
  --pnp-random-seed 0 --require-cuda

python 定位演算法/validation/benchmark_edm_site_replay.py \
  --site-profile "$NEW_SITE_PROFILE" \
  --video "$VIDEO" \
  --out "$CANDIDATE_JSON" \
  --quality-baseline "$BASELINE_JSON" \
  --pnp-random-seed 0 --require-cuda
```

准入要求：

- map-independent exact 優化：matcher arrays、reference trace、state、inliers、pose 必須完全相同
- map-dependent policy：完整 replay 不得降低核准的 recovery/quality gate
- hard-negative 不得新增錯誤 lock
- synchronized GPU timing 與端到端 wall p50/p95 都改善才宣稱更快
- production-path 另查 source-frame age、coalesce drops、ready/first-result latency

### E. 升級與記錄

1. 只修改新 release 的 compat runtime profile，不覆寫舊 release。
2. 更新 site profile 中所有 SHA-256 與 coordinate-frame id。
3. 把 baseline、candidate、quality gate、frame/reference hashes 放到受治理的 validation receipt 目錄。
4. 在本文件新增日期、硬體、設定、數值、證據路徑、決策與 rollback 值。
5. 完成 package manifest、system validation、固定 replay、UI smoke 後才刪除舊 feature bank 或舊 map release。

## 實作同步邊界

EDM runtime 目前有 workspace 與 portable/deploy 兩份同步副本。任何已驗證優化若修改下列檔案，必須一起更新並跑對應測試：

- `定位演算法/EDM工具包/deploy/edm_matcher.py`
- `定位演算法/deploy_code/sfm_glomap_deploy/edm_matcher.py`
- `定位演算法/EDM工具包/deploy/reloc_localizer_edm.py`
- `定位演算法/deploy_code/sfm_glomap_deploy/reloc_localizer_edm.py`
- 兩份 `production_localizer_factory.py`

不可只改 benchmark copy 或只改 portable copy後宣稱正式 runtime 已優化。

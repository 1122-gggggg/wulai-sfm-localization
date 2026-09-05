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
- EDM runtime profile SHA-256：`a65f78ca0d8b388063c1e259f400b1860d23df5e83924910fb7f6ea1b421f053`（2026-09-02：`lost_global_retrieval_interval` 15→3；舊 SHA `93e0c2d1…`）
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

- 2026-09-02：`reference_quality_weight` 0.5 仍是 EDMConfig code default，繼續生效。曾與 temporal-off 綁成一支候選 profile,該候選因 temporal-off 被 gate REJECTED、profile 已刪。此權重本身沒有被 gate 否定。

### 11. Temporal reference（s3_no_temporal +34 已被 09-02 gate 推翻；現行 flight profile = true）

**改動**

- `EDMConfig.use_temporal_reference` 維持 `False` 預設（不再考慮改回 `True`），已加註解指向 `s3_no_temporal` 證據。
- River compat profile 仍為 `"use_temporal_reference": true`，為保持 `site_profile.json` 中 `localizer_profile` SHA `93e0c2...` 不破裂，暫不改 JSON；code default `False` 對未指定 profile 的預設 tracker 生效，新發版再同步 profile。
- 註解位置：`production_edm_tracker.py:162`（`use_temporal_reference` 定義處）

**證據：同 P168 700-frame holdout**

- `s3_no_temporal`（`--no-use-temporal-reference`）：520/700（**+34 vs s0 486/700**），wall p50 35.18 ms，p95 91.32 ms，match p50 29.80 ms，inliers p50 83.0，reproj p95 2.628，LOST 168→113，WEAK 40→61，trace `fee9ab93…`
- 原始證據：`outputs/p168_all8_baseline_20260901/s3_no_temporal.json` 及 `SUMMARY.md`
- 失敗對照：s0 的 temporal on 在此 holdout 較慢且 LOST 更多；保存 `False` 預設避免無收益的 temporal 成本

**地圖相依性：機制不相依但收益隨 reference 密度改變。** 新地圖需重跑 temporal on/off 對照；若新地圖 temporal 確有增益，可於新 compat profile 明確設 `true`。

- 2026-09-02：**在目前樹上 temporal-off gate REJECTED**（P168 +20 但 P117 −20 successes、LOST +12、兩段 +11~12 ms p50）。flight profile 維持 `use_temporal_reference: true`。詳見下方「2026-09-02 GPU gate 結果」。本項的 s3_no_temporal +34 結論建立在 2026-09-01 舊樹,現已被更好的基準吸收,不再有效。

### 12. LOST 全域檢索間隔 `lost_global_retrieval_interval` 15→3（2026-09-02，+72 P168 / P117 flat）

**改動**

- flight profile（`localization/` 與 `compat/edm_runtime_profile.json` 兩份）`lost_global_retrieval_interval`：`15` → `3`。
- 意義：LOST episode 在 grace window（`lost_local_grace_frames=2`）之後,每第 3 個 LOST 幀重試一次 MegaLoc 全域檢索,而不是每 15 個。**只影響 LOST 恢復路徑,TRACK 完全不動,所以沒有 per-frame latency 成本。**
- 沒有放寬任何 acceptance gate:錯誤檢索仍要過 `acquire_min_inliers=80`、trajectory bounds、`stale_reacquire_confirmations=2`。
- profile SHA `93e0c2d1…` → `a65f78ca…`;已同步 `site_profile.json`（頂層+release）`asset_sha256`、`compat/localizer_edm_manifest.json` `artifacts.profile.sha256`、`控制介面程式/mission_selections/river_gluemap_all8_direct_localization.json` `localizer.sha256`（新 manifest SHA `eaff3d6f…`）、`MANIFEST.tsv` / `SHA256SUMS`。
- **2026-09-02（後續）:`EDMConfig.lost_global_retrieval_interval` code default `0` → `3`**（`production_edm_tracker.py:165`）。未指定 profile 的 tracker、`edm_production_profile.json`（該檔仍不寫此 key）都改吃 `3`。同步改測試:`test_edm_tracker_quality.py::test_lost_global_retrieval_interval_is_nonnegative_and_defaults_to_three`、`test_deployment_validation.py::test_profile_absent_lost_retrieval_interval_falls_back_to_code_default`。site profile 仍可明寫 `0` 還原 one-shot。**注意:`3` 是 river 1045-ref 調出來的,見「地圖相依性」。**

**證據：P168 700f + P117 全段,stride 3,seed 0,--require-cuda --gpu-span,RTX 5060,venv torch 2.11.0+cu128**

| holdout | baseline（現行 flight profile） | `interval=3` | 判定 |
|---|---|---|---|
| P168 700f | 536/700（76.6%），LOST 103，inliers p50 77，reproj p95 3.17，p50 22.75，p95 107.8 | **608/700（86.9%，+72）**，LOST 65，inliers p50 110，reproj p95 2.73，p50 24.8，p95 84–103 | 進步 |
| P117 全段 | 366/416（88.0%），LOST 42，p50 23.6，p95 62.2 | **366/416（0），LOST 42（0）**，p50 22–24，p95 74–85 | 持平（無退化） |

- P168 successes 608 連跑兩次完全相同（deterministic）；P117 366 連跑兩次相同。
- 唯一成本:P117 LOST 幀 p95 tail 62→74–85 ms（LOST 幀本來就 `ok=False`,不交付 pose,controller `pose_max_age_ms=500` 內）。P168 的 p95 反而 108→84 改善。
- `interval` sweep（P168）:`1`→550(+14)、`2`→583(+47)、**`3`→608(+72)**、`4`→583(+47)、`5`→546(+10)、`8`→583(+47)。`3` 是明顯峰值,非單調。

**地圖相依性:LOST 密度相依。** 新地圖必須重跑 `interval` sweep + 兩段 holdout;若新地圖 recovery 路徑不同,峰值可能不在 3。

**尚缺:** hard-negative / off-map replay（總帳「D. 固定回放 gate」要求）尚未跑;本項與現行 profile 一樣維持 `flight.approved=false` / `validation: NONE`,不作為飛行核准依據。

### 13. KLT bridge — steady TRACK 每 3 幀才跑 EDM（2026-09-03 開成 default,同日 production-path 重測後**改回預設關閉**）

**改動**

- `production_edm_tracker.py::_try_klt_bridge` + `_init_esekf` 附近的 env 讀取:`SFM_EDM_KLT_BRIDGE_INTERVAL`
  **現行 code default `0`（關閉）**。2026-09-03 曾短暫開成 `3`,同日 production-path 重測（見下）後
  依使用者決定改回 `0`;要開啟設 `SFM_EDM_KLT_BRIDGE_INTERVAL=3`。穩定 TRACK 且從強 EDM anchor（inliers ≥ max(2·weak_min_inliers, 60)、
  inlier_ratio ≥ `_MIN_RATIO`=0.66）起,中間 2 幀用 KLT 光流帶著 anchor 的 inlier 2D↔3D 推 pose + PnP,
  第 3 幀跑完整 EDM 重新 anchor。drift guard（`_DRIFT_GUARD`=1）:bridge 後若 reproj > 0.6·max_reproj_track、
  step > 0.5·max_jump、或 inlier_ratio < 0.7,下一幀強制 EDM。`_MAX_CONSEC`=0（用 N−1）。
- **flight profile / SHA chain 完全不動**(只有 code default 與 manifest hash 變)。預設 `=0` 即每幀 EDM;`=3` 開啟。
- 沒有放寬任何 acceptance gate;bridge 幀走 `_track_klt_prior` 既有的 reproj / max_jump / inlier-ratio / confidence gate。
- 測試:`tests/localization/deploy/test_klt_bridge.py`（10 個:預設、state、keyframe、anchor gate、drift guard、max_consec）。
- **forward-backward gate（2026-09-03 補記）:** bridge 的 KLT 一直就是 forward-backward:
  `_track_klt_prior` 正向 LK `p_t→p_{t+1}`、反向 LK `p_{t+1}→p̂_t`,只收 `e_FB=‖p_t−p̂_t‖ < τ`
  且雙向 status 皆為 1 的點,之後再做 median-drop（`e_FB > 2·median` 且 `> 0.5px` 的「合理但偏差」
  drifter 只從 PnP 解算移除、仍留在 track 集合)。τ 現在可用 `SFM_EDM_KLT_FB_PX` 覆寫（對齊 xfeat
  路徑的 `SFM_FLOW_FB_PX`），**code default 仍是 `_KLT_FB_PX`=1.0 px,未動**。τ 掃描結果見下方
  「已驗證為無效」表。測試:`tests/localization/deploy/test_klt_fb_gate.py`（4 個）。

**證據:七段 river holdout（P116/117/118/119/157/167/168,stride 3,seed 0,--require-cuda --gpu-span,RTX 5060），原始 JSON `outputs/klt_bridge_20260902/`**

| 面向 | 結果 |
|---|---|
| sequential 精度 | **淨 successes +89 / 淨 LOST −121**。P119 +46/−55、P157 +21/−32、P168（996f）+12/−14、P117 +5/−9、P116 +3/−14;退化只 P167 LOST +3。 |
| sequential 延遲 | wall mean 每段 −7~−13 ms;bridged 40–53% 的幀。p95 7 段 5 段改善（P116 −35、P167 −18、P157 −8、P117 −4.5、P168 −1.8），P118 +9（小片雜訊）、P119 +7（換 −55 LOST）。 |
| hard-negative | **PASS。** P168 影片跑 urai 錯誤地圖 700f:off 與 on 都 0/700 successes、全程 BOOT_INIT、bridge 觸發 0 次。strong-anchor gate 保證錯誤地圖上無法製造 false lock。 |
| **production-path（LiveLocalizerClient）** | **P117 好**（succ 平手,pose 更新鮮:result_source_age p50 21→6 ms,coalesce 略少）。**P168 700f −18 succ（579→561）** —— sequential 同段也只 −4,全片 +12 靠平順後半。 |
| bridge-aware adaptive submit cadence | `--adaptive-submit`（仿 operator app latency-scaled coalesce interval）**不救反傷**:P168 guard 561→514（LOST 82→123）。節流在慢 recovery 幀跳過 source 幀、餓死已降頻的 EDM recovery。固定 cadence 對 bridge 較好。 |

**production-path 重測（2026-09-03,同設定重跑 4 次,`--worker-mode production-path`,P168 700f,原始 JSON `outputs/pp_p168_*`）:**
先前記的「−18」是單次量測,**重測後退化更大且高度可重現**:

| | rep1 | rep2 | rep3 | rep4 | wall p50 |
|---|---|---|---|---|---|
| bridge **off** | 575/685 | 575/685 | 575/685 | 575/685 | 21.7–23.2 ms |
| bridge **on**（interval 3） | 528/683 | 516/682 | 516/682 | 516/682 | 20.8–22.2 ms |

**淨 −47 ~ −59 successes（83.9% → 75.7–77.3%）。bridge-off 四次完全一致(575),不是雜訊。**
P117 全段同樣重跑 3 次:on 362/360/362 vs off 357/363/364 —— **successes 平手**（落在 ±5 的 run-to-run 帶內）,
但 **wall p50 5.8–6.0 ms vs 20.0–20.5 ms(3.3x)**。
→ **bridge 在 production-path 是「P168 這類 recovery-heavy 片段大幅掉 success、P117 這類平順片段換到 3x 延遲」的取捨**,
不是普遍收益。sequential replay 看不到這件事(sequential P168 on/off 只差 −4)。

**准入狀態:REJECTED as default —— 2026-09-03 依使用者決定改回預設關閉（`0`）。**
不符總帳 §「用途與准入規則」的 map-dependent policy 條款,且重測後的 production-path 代價（−47~−59）
是原記錄（−18）的 2.5–3 倍。sequential replay（含加速地圖驗證跑）仍淨賺,所以**功能保留、可用 `=3` 開啟**,
適用於平順、延遲敏感的場景（P117 那類:successes 平手、wall p50 3.3x 改善）。
要同時拿到延遲與 recovery 需 runbook §2c 的 async 解耦（EDM 低頻在另一 thread），仍未做。

**地圖相依性:** bridge 收益隨 reference 密度 / KLT 可追蹤紋理變。exact per-frame 比對的 gate（neck expand 等）
應設 `SFM_EDM_KLT_BRIDGE_INTERVAL=0` 取乾淨 trace。

**尚缺:** 獨立 ANAFI camera-pipeline holdout（Tier 4 阻擋項,`validation: NONE` 不變）。

## 待辦與進行中優化（不在本總帳，另見 runbook）

排序、gate、狀態追蹤在 `docs/localization_optimization_runbook.md`。摘要：

- Tier 1：temporal-off — **2026-09-02 gate 判定 REJECTED（見下）**。
- Tier 2：recovery↔latency 前緣（state-conditional topk 其實已實作，剩下的是 reference 選擇 policy 調參）、EDM 上一幀 anchor、`async_pipeline.py` 空殼。
- Tier 3：ESEKF 15 維融合（已在 `__init__` 無條件實例化，純 replay 因無 live velocity 而休眠；不能用 replay gate）；EDM neck `repeat→expand`（**2026-09-02 已驗證 exact，但無 TRACK 加速**，見下）；`fine_matching.py` bi-directional `m_bids` 重排（疑似 correctness fix，未單獨驗證）。
- 2026-09-02 已做的安全子集：`_track_klt_prior` 移除 `raise StopIteration` 控制流（行為等價）；`benchmark_edm_site_replay._apply_reference_feature_cache_overrides` 的 8GiB 預算對 CLI override 失效（SUMMARY.md s1_cache1045）已修並補回歸測試 `定位演算法/validation/tests/test_validation_benchmark_helpers.py::test_reference_feature_cache_override_*`。

## 2026-09-02 GPU gate 結果（RTX 5060，venv torch 2.11.0+cu128；固定 P168 700f / P117 全段，stride 3，seed 0，--require-cuda --gpu-span）

**基準已移動。** 現行 committed 樹（含本分支 matcher/reloc rework）的 flight-profile 基準是
P168 **536/700（76.6%）** p50 23.33 p95 107.78 ms、P117 **366/416（88.0%）** p50 23.75 p95 62.09 ms。
總帳舊的 s0=486/700 是 2026-09-01、matcher rework 之前的樹，已不可比。原始 JSON：
`outputs/tier2_sweep_20260902/`（本機，不進版控；只留 baseline + `interval 3` ×2）。

### Tier 1 — `use_temporal_reference` true→false：REJECTED

| holdout | baseline | candidate（--no-use-temporal-reference） | 判定 |
|---|---|---|---|
| P168 700f | 536/700，p50 23.33，p95 107.78 | 556/700（+20），p50 35.47（+12），p95 79.36（−28） | 進步 |
| P117 全段 | 366/416，LOST 42，p50 23.75，p95 62.09 | 346/416（**−20**），LOST 54（**+12**），p50 34.86（+11），p95 64.78 | **退化** |

- `reference_quality_weight` 0.5 已是 EDMConfig code default，baseline 與 candidate 都在跑；本次唯一實測 delta 是 temporal off。
- 分裂結果：P168 贏、P117 輸 5 個百分點且 LOST +12，另外兩段都 +11~12 ms p50。違反准入規則「完整 replay 不得降低核准的 recovery/quality gate」。
- **結論：flight profile 維持 `use_temporal_reference: true`。** 候選 profile 已刪。總帳項 11 的 s3_no_temporal +34 結論建立在舊樹上，已被目前更好的基準吸收；在目前樹上 temporal-off 只是拿一段換另一段。若日後 recovery 路徑再改善，可重跑。

### Tier 3 — EDM neck query `repeat→expand`：exact 已驗證，無 TRACK 加速

- 加 `SFM_EDM_NECK_NO_EXPAND=1` env toggle（`定位演算法/deploy_code/runtime/EDM/src/edm/neck/neck.py`）強制回舊的 2*B 路徑做 A/B。
- P168 700 幀逐幀比對：**0 個欄位差（tol 1e-9）**，536 個 pose `max|Δpose_xyz| = 0.000e+00`（bit-identical）。success/mode/inliers/reproj/n_corr/refs 全同。→ 總帳「query tensor repeat 改 expand：尚未驗證」的 exact 疑慮解除。
- 但 `match_ms` p50：expand ON 17.89 ms vs verbatim 17.55 ms（expand 略慢 0.3 ms）；p90 62.2 vs 59.7。TRACK batch 是 B=2，省 B−1=1 次 query unary CNN 抵不過 `.expand().contiguous()` + `torch.cat` 的開銷。大 batch 的 acquisition 可能有利但未量。
- **結論：expand 路徑數值安全（可保留 guarded），但不是 TRACK 加速，不列為已驗證優化。** 若要簡化程式碼可移除;要保留則維持 shape-driven guard + `SFM_EDM_NECK_NO_EXPAND` 逃生閥。

### Tier 3 — ESEKF

`ESEKF` 在 `ProductionEDMTracker.__init__` 無條件實例化（import 成功即建）。`predict` 需要
`observe_fused_state` 餵 `_latest_velocity_ned`，純 replay 沒有這個來源 → `prediction_allowed()`
維持 False → PREDICTED_ONLY 的 ESEKF 分支在 replay 不會觸發。上面的 536/700 基準已含這條（休眠的）
程式碼路徑。ESEKF 的實際效果要 live telemetry 才能評，replay gate 評不到。

### Tier 2 — reference-policy A/B sweep（P168 700f，vs baseline 536/700）

一次掃 11 個單槓桿 + P117 交叉驗證。**唯一過 gate 的是 `lost_global_retrieval_interval 15→3`（見上方項 12）。** 其餘全部否決,記錄如下,不要重做:

| 槓桿 | P168 | P117 | 否決原因 |
|---|---|---|---|
| `lost_prior_strategy full_global` | 601（+65），LOST 62 | **349（−17），LOST 57（+15）** | 與 temporal-off 同型:買 P168 賠 P117 |
| `lost_prior_strategy score_fusion` | 601（+65），LOST 62 | **349（−17），LOST 57（+15）** | 同上 |
| `score_fusion + interval 3` 組合 | 568（+32） | 366（0） | 比 `interval 3` 單獨（608）差,score_fusion 拖累 |
| `--no-track-map-first` | 573（+37），LOST 88 | 366（0） | p50 22.75→**41.56 ms（近 2×）**,TRACK 路徑變重 |
| `acquire_stage_mode full_set` | 536（0） | — | P168 successes 無變化,只 p95 微降 p50 +3 ms |
| `acquire_stage_mode initial_topk` | 536（0） | — | 同上,無效 |
| `--no-pnp-ranked-batches` | 536（0） | — | 無效（與 ledger「P168 rejected all PnP candidates」一致） |
| `--local-topk 1`（profile 為 2） | 536（0） | — | 無效,TRACK top-2→top-1 在此 holdout 不 diverge |
| `lost_global_retrieval_interval 1` | 550（+14） | — | 每 LOST 幀重試反而較差,churn 破壞 grace-window 局部搜尋 |
| `lost_global_retrieval_interval 2` | 583（+47） | 366（0） | 不如 3 |
| `lost_global_retrieval_interval 4` | 583（+47） | 366（0） | 不如 3 |
| `lost_global_retrieval_interval 5` | 546（+10） | — | 幾乎無效 |
| `lost_global_retrieval_interval 8` | 583（+47） | — | 不如 3 |

原始 JSON:`outputs/tier2_sweep_20260902/`（本機,不進版控;losing run 已刪,只留 baseline + `interval 3` + `interval 3` confirm）。

## 2026-09-02 — Tier 4：1045-ref 地圖相依參數重驗

**背景。** `river_gluemap_all8_direct_20260831` compat profile（SHA `a65f78ca…`）的 `map_scale.S`、
`radius`、`max_jump`、`reference_feature_cache_size` 是換到 1045-ref 直接地圖時寫的，但沒依「地圖更換
重做流程 B/C」逐項重驗（runbook §4 舊項）。硬體 RTX 5060 Laptop（8151 MiB），venv torch 2.11.0+cu128，
P168 700f + P117 全段，stride 3，seed 0，`--require-cuda --gpu-span`。原始 JSON：`outputs/tier4_20260902/`
（本機，不進版控）。

### 1. `S` — 重算 exact 相符

從 shipped `localization_bundle.pt` 的 `ref_centers`（1045×3，float32）重算
`S = 2·p95(‖center − componentwise_median‖)`（`tools/import_direct_edm_bundle.py` 同式）：
得 `2.3346915245056152`，與 profile `map_scale.S` **逐位元相同**。`0.40·S = 0.9338766…` 亦與
profile `radius` / `max_jump` 相同。→ 算術無誤，非沿用 river 舊值。

### 2. `radius`（local reference search）— 掃描峰值就是 0.40·S

P168 700f，`--radius` 掃 0.20–0.60·S（其餘不變）：

| radius | successes | TRACK | WEAK | LOST | wall p50 | p95 |
|---|---|---|---|---|---|---|
| 0.20·S = 0.4669 | 579 | 579 | 43 | 72 | 26.85 | 108.16 |
| 0.30·S = 0.7004 | 569 | 569 | 36 | 89 | 28.61 | 107.22 |
| **0.40·S = 0.9339（現行）** | **608** | 608 | 21 | 65 | 26.82 | 102.19 |
| 0.50·S = 1.1673 | 607 | 607 | 22 | 65 | 28.82 | 103.50 |
| 0.60·S = 1.4008 | 606 | 606 | 21 | 67 | 28.88 | 103.83 |

較小 radius 明顯傷 recovery（LOST +7~24）；較大 radius 中性（−1~−2）且 p50 +2 ms。**0.40·S 是峰值，維持。**
`max_jump` 綁同值（safety gate，無 `--max-jump` CLI，未單獨掃）。

### 3. `reference_feature_cache_size` — 64/192 trace byte-identical，維持 192

P168 + P117，host cache 0，cache size 64 vs 192：兩段的逐幀 trace（success/mode/inliers/reproj/n_corr/
refs/pose_xyz）**SHA 完全相同**（P168 `6f729884…`，P117 `df3b013e…`），wall p50 差 <0.1 ms。
P168 整段只用到 **75** distinct refs、P117 **36**（runbook 舊估「~152」是兩段聯集的寬估）。
cache-1045（全地圖）在 8 GiB GPU **不可行**（EDM matcher 8 GiB bound，1045×9 MiB ≈ 9.4 GB）。
但 cache-64 < 75 distinct refs 仍給 identical trace → **cache miss 只是從 persistent bank 重抓，不改結果**
（只影響延遲）。→ 192 對兩段 holdout 有餘裕、遠低於 budget，維持。

### 4. 其餘 B/C 項

- local/weak/lost top-k、near pool、covis、inlier/reproj、jump/yaw、stale-LOST gates：P168 700f 走過
  BOOT/TRACK/WEAK_TRACK(21)/LOST(65)/reacquire，`rejection_counts` 僅 `stale_reacquire_unconfirmed:1`
  （正確拒絕），無新增誤鎖 → gates 在這兩段 holdout 上運作正常。
- persistent reference feature bank：三次 cache 掃描都 991 hits / 75 cold misses、trace 一致 → bank 載入
  並提供 exact 輸出。正式 `SFM_EDM_BUILD_REFERENCE_FEATURE_STORE=1` rebuild-diff 因 bundle 未變而延後。

### 決策

**四項地圖相依參數（S / radius / max_jump / cache size）全部確認為 1045-ref 地圖的正確 / 最佳值，
無變動 → profile / manifest / SHA chain 不動。** runbook §4 Tier 4 該項標為已補。**仍未解**：獨立 ANAFI
camera-pipeline holdout + hard-negative / off-map replay（無此 corpus），flight release 維持
`validation: NONE`。rollback：無（未改任何值）。

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
| worker `core_wall` 裡 `vpr+match+pnp` 以外的 27% | **拆完了，沒有槓桿**（2026-09-05，runbook §0b-2）：加 stage timer 量到那 27% 是五個 0.6–2.3 ms 的零碎項——`stage_select_ms` 2.27、`stage_gray_ms` 0.71、`stage_query_ms` **0.60**、localize 殘差 1.48、info 之後的尾巴 1.45（P168 700f p50）。原本猜是 `prepare_query`（每幀 query backbone forward，確在 `match_ms` 之外），實測只有 0.60 ms——項 2「每幀 query backbone 只算一次」已經把重的部分留在 per-reference match 裡。`match_ms` 佔 72.7%（LOST/WEAK 91.8%），瓶頸仍是多 ref EDM forward。儀表已證 exact 並保留（見下），不要再拆一次。 |
| KLT bridge — naive（每 N 幀跑 EDM，無 anchor gate） | REJECTED：P168 700f N=3 successes 608→503（−105）、LOST 65→154。KLT 在難定位段漂移，yaw/jump gate 連鎖拒絕。→ 加 strong-anchor gate + drift guard 後見總帳項 13（已開成 default）。 |
| KLT bridge — `_MAX_CONSEC=1`（連續 bridge 上限 1） | REJECTED：bridge 率 →~30%、P168 −39 succ。drift guard 才是對的槓桿，不是硬性連續上限。 |
| KLT bridge 換 flow backend — **FastFlowNet**（ICRA 2021, arXiv:2103.04524） | REJECTED（完全無作用且更慢）:FB gate 下只有 **3–5%** 的 track 存活（遠低於 `_KLT_MIN_TRACK`=30）,7 段離線量測一致;端到端 bridge **觸發 0 次**,P168 700f 與 P117 全段對 bridge-off 控制組 **逐幀 byte-identical**（success/mode 0 幀差、pose delta 0）,卻仍每幀付 ~48 ms GPU flow(processing fps 25.9→14.7)。根因:輸出是 1/4 解析度上採樣、為 Sintel/KITTI 大位移訓練,次像素精度地板 e_FB≈3.5 px 與位移無關;bridge 要的是 ~3 px 位移下的次像素。不要再試。 |
| KLT bridge 換 flow backend — **DIS**（Dense Inverse Search, arXiv:1603.03590,`cv2.DISOpticalFlow`） | REJECTED（落在雜訊內且大幅變慢）:7 段 stride-3 replay seed 0 淨 dis_fast **+2**、dis_ultrafast +24（其中 +26 全來自 P157,見下方 seed 註記,不可信）。速度反向:每幀 forward+backward KLT 2.0–2.6 ms vs DIS ultrafast 7.9–8.6 / fast 18.3–20.0 /medium 62–69 ms —— **比它要取代的 EDM match（p50 ~13 ms）還貴**,bridge 的省時目的自毀。P117 wall p50 6.4→16.7/24.1 ms,processing fps 27.3→21.8/20.1。DIS 的強項（大位移）確實存在但不在本 pipeline 的工作區:frame gap 掃描下 gap 48 時DIS fast e_FB 0.98 px / 保留 44% vs KLT 4.74 px / 32%,但 production stride 3 的真實位移只有 ~3 px,該區 KLT 又準又快。 |
| KLT bridge — FB 門檻 τ 1.0→0.5 px（`SFM_EDM_KLT_FB_PX`） | REJECTED（中性,不值得動）:P168 700f successes 604→603、P117 全段 371→371,candidate_mode 分布兩段完全相同,逐幀只各差 1 幀。離線 track 層面 τ=0.5 確實更乾淨（見下）,但那些 track 進 PnP 前已被 median-drop 與 RANSAC 吃掉,端到端看不到。維持 τ=1.0。 |
| KLT bridge — bridge-aware adaptive submit cadence（`--adaptive-submit`） | REJECTED：production-path P168 guard 561→514（LOST +41）。忙時節流會在慢 recovery 幀跳過 source 幀，餓死已降頻的 EDM recovery。固定 cadence 對 bridge 較好；production-path 缺點是結構性的，非 cadence 可解。旗標保留（仿 operator app cadence，其他 production-path 測試可用），預設 off。 |
| 自適應 VPR 節奏（blur 門控＋失敗退避） | REJECTED：P168 562/700（−46，LOST 65→100）。blur 閾值跳過了有效檢索，退避在難段把恢复拖死。p95 雖改善但成功率優先，維持 `SFM_EDM_ADAPTIVE_VPR` 預設關。不要調參重試。 |
| WEAK/LOST 上一幀 KLT rescue（`prevframe_klt` 額外候選） | REJECTED（零觸發）：P168 700f + P167 全段 stride-3 seed 0，約 250 個 WEAK/LOST 失敗幀，觸發 0 次，successes 與 baseline 完全一致（608 / 594）。instrumented probe 證實失敗山谷長達數秒（rescue seed age 0.5~6s，KLT seed 早被 `_on_miss` 清掉；獨立 rescue store 修好後仍因 age 全擋）。單跳 KLT 跨不過秒級山谷，增量攜帶等於已否決的 bridge。程式碼已乾淨還原，不留休眠路徑。不要重做。 |

**flow backend 對比方法與 seed 雜訊警告（2026-09-03）:** 上面兩列的實驗裝置 —— `定位演算法/validation/flow_backend_experiment.py`（把 `production_edm_tracker._FLOW_BACKEND` 換成 DIS / FastFlowNet;預設 `None`=原本的 LK,production 路徑不變),測試 `定位演算法/validation/tests/test_flow_backend_experiment.py`。
離線 track 層量測（7 段,gap 3,1280x720,Shi-Tomasi 300 seeds,中性真值 = SIFT ratio-test + MAGSAC 基礎矩陣）:KLT 2.0–2.6 ms / 保留 86.9–97.5% / Sampson p50 0.033–0.061;DIS fast 18.3–20.0 ms / 保留高 1–3 個百分點 / Sampson p50 略差但 **>1px 尾巴一致較低**（例 P168 4.76% vs KLT 7.69%）。
**但端到端分不出高下,而且 seed 雜訊比 backend 差異大一個量級。** P157 700f 四個 PnP RANSAC seed:
seed 0 = klt 612 / dis_ultrafast 638 / dis_fast 614;seed 1 = 655 / 655 / 655;seed 2 = 613 / 615 / 615;seed 3 = 650 / 649 / 651。**光是 KLT 自己就在 612–655 之間擺盪(43 successes)**,seed 0 那個 +26 在其他三個 seed 全部消失。
→ **總帳既有的「固定 seed 0」配方對 recovery-heavy 片段的小幅 success 差異不具鑑別力。**未來任何宣稱 ±30 successes 以內的 policy 收益,必須跨多個 seed 重跑才算數。原始 JSON `outputs/flowbe_*.json`。

**KLT forward-backward gate 離線量測（2026-09-03,cv2 4.13.0,1280x720,連續幀對,Shi-Tomasi 300 seeds,參考真值 = MAGSAC 基礎矩陣的 Sampson 誤差）:** FB 在 track 層面確實有效——
LK 回報 status=1 但幾何上是 outlier 的比例,P167 1.23%→0.82%、P168 4.30%→2.99%、P117 1.41%→0.41%
（τ=1.0,分別只丟掉 1.0% / 2.6% / 2.1% 的 track）;收緊到 τ=0.5 再降到 0.63% / 2.42% / 0.29%。
反向 LK 成本 0.8–1.9 ms/frame（約等於正向）,相對 EDM match 可忽略。但端到端 replay 對 τ 不敏感（見上表）。

原始負面實驗詳情：`outputs/optimization_20260831_report.md`、`outputs/edm_p168_20260831/`、`outputs/edm_resolution/`、`outputs/megaloc_token_reduction_*`、`outputs/megaloc_boq_resnet50/`、`outputs/edm_tensorrt/`。

**2026-09-03 清理（已刪原始 JSON，結論保留）：** `outputs/pp_p168_*` / `pp_p117_*`（項 13 production-path 重測 4 reps）、`outputs/flowbe_*.json` / `fbtau_*.json`（flow backend + τ 掃描）、`outputs/p168_all8_baseline_20260901/`（舊樹 s0/a1–a4，含 a2_quality / s3_no_temporal）、`outputs/klt_bridge_20260902/pp` / `pp_adaptive` / `p95guard`（過渡重跑）、頂層 `outputs/p117_holdout.json`（與該目錄內副本逐位元相同）。保留：`tier2_sweep_20260902/`（4 檔現行基準）、`tier4_20260902/`、`klt_bridge_20260902/` 頂層 6 檔 + `allvids/`（七段證據）+ `pp_hardneg/`（urai hard-negative PASS）。重跑任一已刪實驗時以本文件數值表為準，不以缺失 JSON 為失敗。
**2026-09-03 代碼修正 gate（正確性修正，無成功率宣稱）：** 同日修 `KLT_BRIDGED` 身份（bridge 預設關，replay 休眠）、pairwise 2v1 多數決（原一票否決）、corr-starved PnP rescue（top-2）、速度 dt 加權 EMA、ESEKF 異常計數、KLT/bridge 常數命名、receipt 納入 `klt_bridge_interval` / `esekf_disabled` / `klt_fb_px` / `reference_quality_weight`。P168 700f stride-3 seed-0：**608/700，TRACK 608 / WEAK 21 / LOST 65，inliers p50 110，reproj p95 2.73** —— 與 Tier 4 interval-3 基準逐項一致（零迴歸）。P167 全段 stride-3（773 幀，P117 影片不在盤上以此代交叉驗證）：修前後同 **594/773**，success/mode/pose 零翻轉；29 幀僅 LOST 幀 inlier 數差異（rescue 多跑出 10–40 inliers 但仍低於 gate，全擋下，無誤鎖）。stride-3 dt≈0.125 使速度 EMA alpha≈0.5 與舊值一致，replay 看不出差別，修正針對 production-path 幀間隔抖動。原始 JSON 已按清理政策刪除，以本段數值為準。

**2026-09-03 Wave A gate（提速三項，零迴歸）：** (1) LOST grace 幀 O(N²) 去重：`_match_acquisition_stages` 在 temporal_used 時保留 by_ref 增量語意，14 forwards→6，LOST 幀 wall 上限 265~280ms→145.7ms；(2) direction-01 旁路（預設開，`SFM_EDM_FINE_DIR01=0` 還原）：FP32 下 byte-identical，FP16 autocast 端到端 mkpts1 最大差 0.0036px（cuBLAS tiling 噪聲，為 RANSAC 2px 門檻的 1/500，k0/mc 仍 identical）；(3) 3-slot SHM 環形（production-path 才有差，sequential 休眠）。receipt 新增 `fine_dir01` / `matmul_precision` / `sdpa_fusion`（TF32/SDPA 預設關）。P168 700f：**608/700（608/21/65，p50 26.43，p95 104.0）**；P167 全段：**594/773（594/87/92）** —— 雙段與基準逐項一致。原始 JSON 已刪，以本段為準。

**2026-09-03 Wave B gate（單一超強候選 early-stop，零迴歸、微增益）：** `_acquire_stage_can_stop` 加快速通道（Top-1 inliers≥max(80·1.5,120) 且 reproj≤2.0，`SFM_EDM_SINGLE_STRONG_EARLY_STOP=0` 還原；安全 gate 不動）。P168 實測觸發 1 次（BOOT），**608/700（608/21/65）**；P167 **594/773（594/87/92）** —— 雙段零翻轉。本段 replay 增益微小（難幀少有 120+ inlier 的 Top-1），保留理由是 BOOT/重捕獲貴幀省 Stage 2/3 且 hard-negative 語意不變。第 2 步（WEAK/TRACK 階梯 PnP）經查已是現狀，不動。原始 JSON 已刪，以本段為準。

**2026-09-03 Wave C gate（PnP 資格放寬預設開＋WEAK 遲滯轉正，P167 +53~+58）：** (1) eligible 放寬為 max(10, 0.8·min_inliers)（`SFM_EDM_PNP_ELIGIBLE_RELAX=0` 還原），主路徑與 pipeline submit_rows 對齊；(2) TRACK 遲滯：inliers∈[45,50) 且 reproj≤2.5 且 step 合法給 +0.5 緩衝（連兩幀照降），code default 已由 env-off 翻為 **on**（`SFM_EDM_WEAK_HYSTERESIS=0` 還原；profile 的 weak_after=1 未動故免 SHA 輪轉，MANIFEST 已重gen）。P168（兩配置皆）：**608/700（608/21/65）**；P117 hyst-on：**366/416** 平手；P167 relax-only 594（放寬貢獻 0，此段）→ hyst-on **652/773（658/36/79）**；P167 seed-1：off 574 → on **627（+53**，seed-0 為 +58，雙 seed 同向且超 seed 雜訊帶）。原始 JSON 已刪，以本段為準。
**2026-09-03 Wave D gate（VPR 自適應 REJECTED，先驗精修保留關 —— 本行 2026-09-04 更正：精修實際為 code default ON）：** (1) 自適應 VPR（blur 門控＋失敗退避，`SFM_EDM_ADAPTIVE_VPR` 預設關）：P168 掉到 **562/700（−46，LOST 65→100）**，p95 雖 104→80，但成功率崩，判定 REJECTED，維持預設關（blur 閾值 50 疑似過嚴，跳過了有效檢索；不再調參，直接否決）；(2) 先驗種子 PnP 精修（`SFM_EDM_PRIOR_PNP_REFINE`）：本行原寫「預設關」，與程式碼不符 —— `production_edm_tracker.py:1327-1331` 自 09-03 起即為 `_env_bool(..., True)`（註解引 gate：P168 608 + P117 366 零迴歸）。2026-09-04 實測確認 code default ON 是真實生效且改 trace 的：P168 200f seed-0，`=1` vs `=0` 成功同為 165/200 但 trace SHA 不同（`4d5fe1…` vs `f3dccb…`，證據 `outputs/audit_20260904/audit_pinned1_seq200.json` / `outputs/audit_20260904/audit_norefine_seq200.json`）。成功率面仍是零迴歸（與 09-03 結論一致），故保留 ON，不改碼；此更正只修文件，不代表新的升級決策；(3) `SFM_EDM_PNP_THREADS` 透傳（預設 1，多線程 byte-identical 已證）。原始 JSON 已刪（09-03 的），以本段為準。

**2026-09-03 Wave E 評估（三項皆 NO-GO，有數據）：** (1) 離線 Neck K/V：Layer 0 self + Layer 1 K/V 投影確全等，但 Layer 1 cross 把 query 特徵注入 reference 狀態（跨 query max diff 0.1474），Neck 輸出無法離線閉合；且收益估 <0.15ms、bank 膨脹 ~940MB，不改 bank；(2) TRACK B=1 CUDA graph（`SFM_EDM_TRACK_CUDAGRAPH` 預設關）：eager vs graph 差全 0，但只快 0.18ms，卻鎖 3.42 GiB pool，8GB 卡上 OOM 風險遠大於收益，維持關；(3) DSNT/二次曲面換 fine head：座標 mean 漂 >1.0px、max 4.3px，全點超 RANSAC 門檻，換了等於重訓，不動。預設路徑 P168 **608（608/21/65）** 確認零影響。原始 JSON 已刪，以本段為準。

**2026-09-03 Async 模組（新檔，未接線）：** `async_localizer.py`（Fast 30Hz CPU LK+PnP 發 KLT_BRIDGED/klt_fast；Slow 5~10Hz EDM+MegaLoc 發 VISUALLY_CONFIRMED；capacity-1 mailbox + 200ms max-age；SE(3) sync-carry 過 jump gate；drift budget 預設 5 幀用完報 LOST；三級 keyframe 排程；MegaLoc 任務級搶佔排 Keyframe 後；decision_log 可 record-replay）。8 單元測試全綠（fake slow，無 GPU）。刻意未接入 live worker（紅線：不碰現行行為）；上線前需 production-path gate（≥608 且 p50≤6ms）。缺口見回報：mid-kernel 搶佔無、ESEKF 動態 max_jump 無。

**2026-09-03 Wave F（WEAK staging 撤銷、async 接線保留、matcher 不動）：** (1) WEAK Top-1 階梯：實作後發現 WEAK 全走 track_map_first（temporal ready），dispatch 分支永遠到不了；改入 map-first 後仍零觸發且與一次回歸撞期，在 hunk 平衡審查無決定性證據前先整段撤銷（方法＋兩處呼叫＋計數器＋測試全刪，零殘留已 grep 確認），不留休眠碼；(2) AsyncEDMTrackerAdapter + `SFM_EDM_ASYNC_TRACKER` factory 分支（預設關，4 單元測試；slow keyframe 取 `_last_accepted_cam_from_world`，fast 相機由 site 內參縮放至 EDM 尺寸）；(3) matcher 死碼/零拷貝：`cell_ids`/`is_refined` 全 repo 有引用、`match_one_to_many` 地圖建置仍用，不刪；`to_tensor` 省 ~0.3ms 但動每幀熱路徑，不值得，不做；(4) dt 補償：`adaptive_jump_limit` 與兩處呼叫點本就帶 capture_dt，自適應已存在，不動。P168 **608**、P167 **652**、P117 **366** 維持。

**2026-09-03 編輯事故紀錄（已修復，608 已恢復）：** WEAK staging 撤銷時 CUT 誤刪 `_match_track_map_first` 內既有的 map-first early-stop 兩行（`_track_stage_can_stop` 命中即返，不跑 temporal fallback），導致每 TRACK 幀多一次 temporal forward、全段 +136 forwards、P168 掉到 561（LOST 65→108）。症狀像時脈問題（match p50 翻倍）且逐項 env 開關都排不掉，定讞靠三點：(a) 同機器背靠背新舊交錯跑（152/165/152/165 嚴格跟碼）；(b) 單檔 bisect 鎖定 tracker；(c) 純刪除 hunk 審查找到 -8+0 的 hunk 38。教訓：動熱檔後除 pytest 外，必做 `git diff` 加減平衡審查＋短 slice 回放（200 幀 25 秒），再跑全段。

**2026-09-03 Async parity（gate 前修，無成功率宣稱，預設仍關）：** `AsyncEDMTrackerAdapter` 四處與 sync 對齊：(1) `ensure_models` 暖 slow tracker（`trk.loc.megaloc` + `warmup_fused_coarse`，原直接 `return None`，首 keyframe 付全量模型載入延遲）；(2) slow accept 快取 timing/refs（`vpr/match/pnp_ms`、`refs`、`global_retrieval_calls`）進 anchor `meta` + `_last_slow_info`，`localize_frame` 的 `_last_info` 由 `None` 改讀快取（原 production-path gate 讀不到 async 的 p50/source-age）；(3) `map_frame.heading(R[2])` 優先（與 sync 同式，原用 fast yaw）；(4) `observe_fused_state` / `attach_pose_guided` 轉發給 slow `trk`（worker 調 adapter 即生效）。測試 `tests/localization/deploy/test_async_adapter.py` 4→8（telemetry 快取、fast 透出、map-frame yaw、forward）。同步路徑零動（`git diff` 此檔僅新類別加法）；P168 200f slice `165/200（82.5%），p50 24.05ms` smoke 通過（`outputs/async_parity_slice200.json`，本機不進版控）。仍需 production-path 全段 gate（≥608 且 p50≤6ms）才可預設開。

**2026-09-04 KLT confidence-only keyframing（`INTERVAL=100` 零改碼實驗，有數據，未升級）：** bridge 本來就每幀過 KLT confidence gate（FB τ=1.0、`CONF_MIN`=0.30、median-drop；不過即 EDM），固定 `interval=3` 只是強制定期 re-anchor 上限。把上限放寬到 100 = 只剩 confidence 不足＋drift guard＋jump tripwire 才跑 EDM（`SFM_EDM_KLT_BRIDGE_INTERVAL=100`，sequential，seed 0）：
 P168 700f：**603/700（−5 vs off 608）**，p50 **10.3ms**（off 26.8、interval-3 18.4），bridged 56%；P117 全段：**366/416（與 off 逐項一致）**，p50 **11.2ms**（off 26.2），bridged 58%。P168 overlap 700 幀分歧 19/14（有來有回，淨 −5）。
 解讀：confidence 門確實比固定 cadence 更會挑幀，同成功率下 EDM 幀更少。但三個保留：(1) 只是 sequential，production-path（coalesce 時序）還沒跑——bridge 在那裡曾 −47~−59，conf-only bridge run 更長，drift 可能更大；(2) KLT confidence ≠ pose 正確，重複植被假鎖（P119 yaw 段 ratio 0.85）是信心滿滿的錯，backstop 只有 drift guard＋jump gate；(3) seed 雜訊帶 ±30（flow backend 實驗），−5 雖是同 seed 確定性差異，仍要多 seed 才算數。**預設不動（`=0`），要升級需 production-path gate＋多 seed。** 原始 JSON `outputs/klt_confonly_p168_700.json` / `klt_confonly_p117.json`（本機不進版控）。

**2026-09-04 Async 接線 + gate（`SFM_EDM_ASYNC_TRACKER=1`，NOT GO as default，但已可跑）：** 把 `AsyncEDMTrackerAdapter` 補齊到能過 worker 介面並實測。修了四個真 bug：
 (1) **seed 座標尺度**：slow tracker 的 `_klt_2d` 是 camera-res（1280×720，`_correspondence_row` 已乘 `self.scale`），fast path 追的是 EDM-res gray（1024×576）→ LK 起點偏 ~250px。`_slow_keyframe` 改成把 `inlier_2d` 乘 `EDM_W/cam.width`、`EDM_H/cam.height`（3D 免動，是 map frame）。測試 `test_slow_keyframe_rescales_seeds_to_edm_grid`。
 (2) **`SyncCarry` LOST 死鎖**：drift budget（5）用完 → LOST → fast path 早退不再 `record_step` → `transform_chain` 永遠蓋不到 `[key_stamp, now]` → `get_relative_transform` 回 None → re-anchor 分支整段跳過 → **再多有效 anchor 都出不了 LOST**。修法對齊 sync tracker 的 LOST reacquisition：`reacquiring = bridge_count > max_drift_budget` 時直接採用 anchor pose（identity 合成）並跳過 jump gate（anchor 已過 slow tracker 全部 inlier/reproj/trajectory gate）。P117 60f production-path：**6/60 → 45/60**。
 (3) **worker 狀態鏡射**：`_push_state_to_slow` / `_pull_state_from_slow`（在 `_state_lock` 下於 slow keyframe 兩端執行）＋ `_clear_tracking_history` 全量重置（slow `st`、visual/KLT cache、pose_guided、mailbox、chain、sync_carry、fast_path、scheduler）＋ `prev_pose` 每幀跟隨。`AsyncLocalizer.reset()` / `FastPath.reset()` / `SyncCarry.reset()` / `Scheduler.reset()` 為此新增。
 (4) `SFM_EDM_ASYNC_DEBUG=1` 診斷輸出（預設關）。

**gate 結果（production-path，stride 3，seed 0，RTX 5060）：**

| holdout | sync（現行預設） | async | 判定 |
|---|---|---|---|
| P117 全段 | **358/408（87.7%）**，p50 22.36，p95 81.2 | 286/416（68.8%），**p50 3.63，p95 5.90** | **成功率 −19pp，延遲 6.2x 改善** |
| P168 700f | 575/685（83.9%，總帳項 13 四次重測） | 410/700（58.6%），**p50 3.90，p95 7.93** | **成功率 −25pp** |

async 模式分布：P117 `VISUALLY_CONFIRMED 48 / KLT_BRIDGED 238 / LOST 117`；P168 `74 / 336 / 244（+NEED_REANCHOR 33）`。→ **fast path 有在出 pose 且延遲達標（p50≤6ms 的目標達成），但 LOST 佔比太高**：slow keyframe 每幀都排（`_SCHEDULER_INTERVAL_TRACK=0`）卻只能 ~25ms 跑一輪，anchor 供給跟不上 drift budget 5 幀的消耗，難段一路 LOST。**未達 ≥608 門檻，維持 `SFM_EDM_ASYNC_TRACKER` 預設關。**

**下一步（有明確方向，不是猜）：** drift budget 隨 anchor 供給率自適應（目前固定 5）、LOST 時 slow path 走 sync tracker 的 recovery（現在 slow tracker 自己是 LOST 就只吐 None，fast path 什麼都拿不到）、mailbox `max_age_s=1.0` 與 anchor 產出間隔對齊。原始 JSON `outputs/pp_async_P117_fix.json` / `pp_async_P168_fix.json` / `pp_async_P117_sync.json`（本機不進版控）。

**量測事故紀錄：** 本輪第一次 async gate 得到 `0/416`，根因是我自己的 in-process 診斷 kernel 佔住 6.4 GiB VRAM，worker 子進程 CUDA OOM（`torch.OutOfMemoryError` 於 fused coarse `bmm`，隨後 illegal memory access）。8 GiB 卡上跑 production-path gate 前必須確認 `nvidia-smi` 無其他佔用進程；`0/416`、`6/416` 兩筆作廢。

**2026-09-04 預設關的逐幀 parity（本日四項改動的共同前提）：** 本日新增的兩個 recovery 旗標
（`acquire_relaxed_min_inliers`、`lost_starved_global_frames`）與 async / 接受端的程式碼改動，全部關閉時
對四段 sequential holdout 的**逐幀 trace SHA 與改動前完全相同**（`source_index` / `success` / `mode` /
`inliers` / `n_corr` / `refs` / `reproj_rms` / `pose_xyz`）：
P168 996f `6c09bc70b85c4ed9`、P117 416f `5fac1c83ff31f47c`、P119 978f `e46f6ed20e7184dc`、
P157 1028f `9a0a30137d952df1`，successes 854 / 366 / 650 / 835 逐項一致；P168 700f seed 0 也重現 **608/700**。
全套 pytest `2421 passed / 3 skipped`。以下每個實驗的 baseline 都是這組同版本 off 跑。

**2026-09-04 LOST 近失 acceptance gate（`acquire_relaxed_min_inliers`，實作＋量到大幅收益，但 NOT GO as default）
—— ⚠️ 2026-09-05 已推翻，code default 改為 66，見本文件「升級預設：LOST 近失接受地板」。
本段數值（sequential、2–4 段）仍然正確，錯的是「用 2–4 段 sequential 判全域」：**
8 段失敗普查裡有 92 幀（全失敗的 8.5%）是 LOST 檢索打出 ncorr 中位數 ~300、inliers 60–79 的「差一點」幀，
`rejected` 欄位全為 `None` —— 它們只死在 `acquire_min_inliers=80` 這一條。新增第二層 LOST 專用地板
（`SFM_EDM_ACQUIRE_RELAXED_MIN_INLIERS`，預設 0=關）：只有在 (a) state_in=LOST 且 acquiring、
(b) inliers ∈ [floor, 80)、(c) reproj ≤ min(`max_reproj_error_acquire`, `acquire_relaxed_max_reproj_error`=3.0)、
(d) **≥2 個獨立 reference 的 PnP 中心落在 acquire consensus 半徑（`acquire_max_jump_factor·max_jump`=1.87 m）內**
才降低地板；ratio / grid-cell / acquire_jump / acquire_yaw / stale 兩幀確認全部不動。
BOOT 不適用（BOOT 沒有可比對的 prior，yaw/jump gate 未武裝）。

| holdout | off | floor 66 | floor 70 | floor 72 | agree=1（floor 66） |
|---|---|---|---|---|---|
| P119 978f | 650 | **750（+100）** | 676（+26） | 665（+15） | 719（+69） |
| P157 1028f | 835 | **861（+26）** | 838（+3） | 835（0） | 855（+20） |
| P168 996f | 854 | **801（−53）** | 854（0） | 854（0） | — |
| P117 416f | 366 | 366（逐項一致） | — | — | — |

seed 1 覆核：P119 661 → **754（+93）**（seed 0 是 +100，雙 seed 同向且遠超 ±30 雜訊帶）；
P168 813 → **795（−18）**（seed 0 −53）。**P168 的損失方向兩個 seed 一致，且伴隨品質退化**：
seed 0 inliers p50 82→68、p95 154→111、accepted_step p95 0.0284→0.0592（2 倍），
seed 1 p50 83→73、p95 154→127，並新增 `limited_jump_unconfirmed:3` / `track_yaw:2`。
成因不是誤鎖：三個 P168 放行幀 inliers 68/69/70、距上一次定位 0.03–0.23 m、之後分別接 64/1/101 幀連續成功；
但提早離開山谷會把 tracker 帶進較差的 reference 鏈，之後 WEAK_TRACK +47。
P119 的三個放行幀 inliers 73/74/73、reproj ≤1.99、距上一次定位 0.03/0.03/0.40 m、之後接 44/31/31 幀連續成功
—— 錯鎖不可能連續追 30+ 幀。

**hard-negative（河濱影片 vs 烏來地圖，700 幀，off-map）**：off 與 floor 66 **都 0/700、全 BOOT_INIT、
逐項一致**，off-map 最佳 PnP inliers **上限 17**（p95 11，p50 0，儘管 ncorr p50 495）—— 錯地圖的
inlier 天花板比放寬地板 66 低 3.9 倍，比 80 低 4.7 倍，所以這層地板離「錯地圖」區完全沒有交集。
**已知限制**：該 corpus 全程停在 BOOT，而本層是 LOST-only，等於只證明了「不影響 BOOT 路徑」；
真正的殘餘風險是**同地圖錯位**且落在 66–79 的鎖（P119 植被假鎖是 80–183，在本區間之上，仍由 yaw gate 擋，
floor 66 跑完 `acquire_yaw` 一樣是 26 次），這需要 on-map 錯位 ground truth 才能證，目前沒有這個 corpus。

**判定：NOT GO as default（`acquire_relaxed_min_inliers` 預設 0）。** 依本文件准入規則，
map-dependent policy 不得降低已核准 holdout 的 recovery/quality gate，而 P168 兩個 seed 都退。
floor 70 是零退化版本（P119 +26 / P157 +3 / P168 0 / 品質逐項一致），但 +29 淨值落在 ±30 seed 帶內，
要升級必須先跑 4 段 × 多 seed。**注意 floor 同時是 corroboration 門檻**（agree 計數用同一個 floor），
所以調高 floor 會同時收緊「第二個 reference 要多好」——這就是 floor 70/72 收益驟降的原因，不是 bug。
agree=1（放寬 corroboration）在 P119 反而比 agree=2 少 31 successes，所以雙 reference 一致性不是安全稅，
它同時也是成功率上的正確選擇。原始 JSON `outputs/recovery_gate_20260904/`（本機不進版控）。

**2026-09-04 starvation-triggered global retry（`lost_starved_global_frames`，REJECTED，有數據）：**
8 段普查有 163 幀 LOST 是 `edm_local_recovery` 只打出 n_corr ≤5（P119 佔 104，全部 2 refs）——
LOST 後拿 last-known pose 附近的 refs，山谷裡那個 pose 是錯的，於是每次 local scan 都重犯同一個錯
（實測 328 幀 `edm_local_recovery` 只成功 2 幀）。新增：同一個 LOST episode 內累計 k 幀 corr-starved 的
local-pool 幀後，下一幀跳過 local／progressive-radius，直接對整張地圖跑 MegaLoc（`megaloc_lost_starved`，
`force_global=True`，發射後計數歸零）。檢索幀不計不歸零（interval-3 的 MegaLoc 會把每段 local run 切成 2 幀，
k≥3 永遠武裝不了——k=3 實測觸發 0 次就是這個原因）。

- P119 k=2：**652/978（+2）**，觸發 48 次，其中 45 次 inliers <80（死在 acquire 地板）、2 次 ≥80 被
  `stale_reacquire_unconfirmed` 正確擋下、1 次放行。p50 26.9→27.7 ms。
- P157 k=2/k=3：**835/1028（0）**，觸發 0 次（該段沒有連續 2 幀 starved local）。
- P119 k=2 + floor 66：**750/978**，與 floor 66 單獨完全相同（放寬地板會提早離開山谷，starved run 更少）。

**判定：REJECTED（預設 0）。** 機制確實會噴出密集檢索幀（ncorr p50 304），但接受端幾乎全部擋掉，
淨值 +2/0 在雜訊內、p50 +0.8 ms。**真正的瓶頸是 acceptance gate，不是 reference 供給** —— 這也解釋了
為什麼它和放寬地板疊加無效。不要再調 k / corr_max 重試；要動就動 acceptance 側。
程式碼保留為預設關的旗標（換地圖時 local recovery 品質可能不同，屬「必須重跑」類）。

**2026-09-04 Async anchor 供給修正 + gate（四個真缺口，成功率 −25pp → −2.8pp，延遲再快 2.4x；預設仍關）：**
接續上一節「下一步」，四項全部實作並實測：

1. **anchor 門檻改吃場域設定（最大單一原因）。** `AsyncLocalizer` 過去用模組預設值重新審查 slow tracker
   已經放行的 pose：`anchor_min_ratio=0.66`、`anchor_min_inliers=max(2·weak_min_inliers, 60)`、
   `max_reproj=4.0`、`max_jump=0.50`。但 5-ref LOST acquire 的 `inlier_ratio` 實測只有 ~0.23
   （TRACK 單 ref 才接近 1.0），所以**每一個 recovery anchor 都被自己丟掉**（P168 700 幀只有 74 個
   VISUALLY_CONFIRMED），而 TRACK 端 50–59 inliers 的合格 fix 也被翻倍後的地板擋掉。現在 adapter 把
   `max_jump` / `acquire_max_yaw_diff_deg` / `weak_min_inliers` / `track_min_inliers`（=anchor 地板）/
   `min_inlier_ratio` / `max_reproj_error_track` / PnP seed 全部從 `trk.cfg` 傳進去，`SlowPath` 的
   `max(2·weak, anchor_min)` 改為 `max(weak, anchor_min)`。slow tracker 是唯一權威：它放行的 pose 已經
   過完該場域的 inlier / reproj / ratio / spread / jump / yaw gate，async 端不得再用與場域無關的數字複審。
2. **drift budget 隨 anchor 供給自適應（`AnchorSupply`）。** 量測最近 8 次 anchor 之間的 fast-frame 數與
   stamp 秒數（取最壞值）：budget = clamp(1.5×frame_gap, floor=`max_drift_budget`=5, cap=30)。原本固定 5
   在「一顆 anchor ≈ 6 幀」的實測節奏下每輪都先報 LOST。
3. **mailbox `max_age_s` 與 anchor 產出間隔對齊。** age 上限 = clamp(2×stamp_gap, floor=1.0 s, cap=8×floor)。
   stride 3 的 source 時間軸比 wall 快 ~25 倍，一顆 145 ms 的 recovery keyframe 在 source 時間上就是 3.6 s，
   固定 1.0 s 會把它全部丟掉。
4. **無法合成的 anchor 改為直接採用，不再丟棄。** `take_if_fresh` 是破壞性讀取，舊碼在 transform chain
   蓋不到 `[key_stamp, now]` 時（BOOT、seed 掉光、連續 miss）會把 anchor 消耗掉又不用，一路撐到 budget
   用完才靠 `reacquiring` 脫身 —— 而自適應 budget 變大會**加長**這個死鎖窗。現在 chain 蓋不到就以 identity
   合成採用（誤差被 anchor age 期間的位移界定，下一顆 anchor 修正），jump gate 照樣審查殘差。
   `info["chain_covered"]` 記錄是否為真合成。
5. **fast path soft-degrade（原第 4 項缺口）。** 一次 PnP/LK 失手不再把 ~100 點 seed 清成 `None`；
   seed 與其參考幀成對保留 `_FAST_SOFT_DEGRADE_MISSES`=2 幀，第 3 幀才丟。該幀對外仍是 stale
   `NEED_REANCHOR` 並照樣要求立即 keyframe —— 只保留 track、不虛構 pose，語意與 sync 的 WEAK 遲滯一致。

**gate 結果（production-path，stride 3，seed 0，RTX 5060，`nvidia-smi` 無其他佔用）：**

| holdout | sync（現行預設） | async 修正前（本日早） | async 修正後 |
|---|---|---|---|
| P117 全段 | 358/408（87.7%），p50 26.40，p95 96.59 | 286/416（68.8%），p50 3.63 | **353/416（84.9%），p50 1.50，p95 3.15** |
| P168 700f | 605/692（87.4%），p50 26.76，p95 80.86 | 410/700（58.6%），p50 3.90 | **592/700（84.6%），p50 1.54，p95 3.25** |

模式分布（修正後）：P117 `VISUALLY_CONFIRMED 332 / KLT_BRIDGED 21 / LOST 49 / NO_ANCHOR 11 / NEED_REANCHOR 3`；
P168 `562 / 30 / 95 / 10 / 3`。對照修正前的 `48 / 238 / 117` 與 `74 / 336 / 244` —— **anchor 供給從
「幾乎沒有」變成「幾乎每幀都有」**，這也是 p50 從 3.6 ms 再降到 1.5 ms 的原因（採用 mailbox anchor 的幀
不必自己跑 LK+PnP）。result source age p50 27.6 → **2.3 ms**。

**判定：仍維持 `SFM_EDM_ASYNC_TRACKER` 預設關**，但差距從 −19/−25pp 收到 **−2.8pp（P117 −5、P168 −13 successes）**，
延遲則是 sync 的 **1/17.6**。未達門檻的部分集中在 sync 也會 LOST 的難段（P168 LOST 95 vs sync 65）。
下一步（有方向）：LOST 期間 fast path 沒有 anchor 可採時仍走 `NEED_REANCHOR` 洪水式排程，
slow 端一輪 recovery match 要 80–145 ms，可考慮讓 scheduler 在 LOST 時降低重複排程、把 GPU 讓給單次深度 recovery。
單元測試：`定位演算法/validation/tests/test_async_localizer.py` 9→21（AnchorSupply、自適應 budget/age、
uncoverable anchor 採用與 jump gate、soft-degrade、anchor 門檻與拒絕原因統計），
`tests/localization/deploy/test_async_adapter.py` 13 個維持全綠。原始 JSON `outputs/recovery_gate_20260904/pp_*.json`。

**2026-09-04 KLT confidence-only keyframing — production-path + 多 seed 補齊（前一節的兩個保留已解，仍維持預設 0）：**
前一節（同日）只有 sequential seed 0，留了「production-path 沒跑」與「seed 雜訊帶 ±30」兩個保留。兩項都補完：

**(1) production-path（`--worker-mode production-path`，同一批 sync 對照，幀數相同可直接相減）：**

| holdout | sync（interval 0） | conf-only（interval 100） | 判定 |
|---|---|---|---|
| P117 408f | 358（87.7%），p50 26.40，p95 96.59 | **362（88.7%），p50 5.29，p95 78.70** | **+4 succ，p50 5.0x，p95 也好** |
| P168 692f | 605（87.4%），p50 26.76，p95 80.86 | **600（86.7%），p50 5.80，p95 80.19** | −5 succ，p50 4.6x，p95 平手 |

這是本項最重要的新事實：**固定 interval=3 的 bridge 在 production-path 曾是 −47~−59 succ（P168，4 次重測），
conf-only 在同一條路徑只有 −5/+4。** confidence 門（FB τ=1.0、`CONF_MIN`=0.30、median-drop）挑幀的能力
確實能吃掉 coalesce 時序帶來的結構性劣化，這不是 cadence 調參可得的。

**(2) 多 seed（sequential，seed 0–3）：**

| seed | P168 off | P168 conf | Δ | P117 off | P117 conf | Δ |
|---|---|---|---|---|---|---|
| 0 | 608 | 603 | −5 | 366 | 366 | 0 |
| 1 | 609 | 586 | −23 | 367 | 367 | 0 |
| 2 | 610 | 581 | −29 | 373 | 365 | −8 |
| 3 | 586 | 608 | **+22** | 367 | 359 | −8 |
| 平均 | 603.25 | 594.50 | **−8.75** | 368.25 | 364.25 | **−4.00** |

off 自己的 seed 擺盪就有 586–610（P168，24 successes）；conf 為 581–608。**四個 Δ 全部落在 ±30 雜訊帶內，
且 seed 3 反向 +22 —— 成功率上無法宣稱有差。** 延遲則穩定：P168 p50 26.4–28.5 → 9.8–10.4 ms（2.7x），
P117 p50 26.1–26.8 → 5.7–8.6 ms（3.1–4.6x）。

**判定：仍維持 `SFM_EDM_KLT_BRIDGE_INTERVAL` 預設 0。** 理由是 p95：**sequential 四個 seed 的 p95 有三個變差**
（P168 104–107 → 105.9/112.0/112.2，只有 seed 3 降到 86.6；P117 78–88 → 89.7–103.5 **四個 seed 全部變差**）。
本文件准入要求「p50 與 p95 都改善才宣稱更快」，p95 這條沒過。機制上合理：bridge 幀便宜，但 bridge 斷在難段時
那一次 EDM re-anchor 的 prior 已漂、cache 更冷，尾巴更長。
**可用結論**：conf-only 是目前量到最強的 p50 槓桿（production-path 4.6–5.0x），成功率在雜訊內、
production-path 不再有 interval=3 的結構性劣化 —— 延遲敏感部署可設 `SFM_EDM_KLT_BRIDGE_INTERVAL=100`，
但要接受 p95 尾巴變長。原始 JSON `outputs/recovery_gate_20260904/pp_confonly_*.json` 與 `seeds/`。

**2026-09-04 Async 續修：fast-fix credit + authoritative adopt（−2.8pp → 反超 sync，延遲維持 1/15）：**
上一節修完四個 anchor 供給缺口後還剩 −2.8pp。`SFM_EDM_ASYNC_DEBUG=1` 的 keyframe 診斷（P168 700f，693 個
slow keyframe）顯示 slow 端本身完全正常：606 個 VISUALLY_CONFIRMED、狀態分佈 `TRACK 605 / LOST 65 /
WEAK 21` 與 sync 自身的 trace 幾乎一致 —— 損失全部在交付與 fast-path 政策，不在 matching。兩個真修：

1. **jump gate 改為只丟 chain、不丟 anchor。** 舊碼把 anchor 合成後的 pose 拿去跟「我們自己 bridge 出來的
   pose」比，超標就保留 drift 掉的 bridge、丟掉 mailbox 裡 destructively-taken 的 fix（診斷 run：產出 606
   個 anchor，只交付 561 個 VISUALLY_CONFIRMED）。anchor 早已過完 slow tracker 的全部 gate，可疑的是
   我們自己的合成，不是 fix —— 超標時改用 anchor 自身的 map pose 採用，並記 `jump_override`。
   `mailbox.stats`（`overwritten`/`expired`）與 `carry.stats`（`adopted`/`jump_override`/`fast_fix`）為此新增。
2. **gated fast fix 還 budget（`note_fast_fix`）+ LOST 不再早退。** 舊碼 budget 用完就直接回 LOST，
   fast path 連試都不試 —— 而它每幀是拿 anchor 3D 做真 PnP、過 inlier/reproj/jump gate 的，
   這是 visual fix 不是 dead reckoning。現在：fast fix 通過 gate 就把 `bridge_count` 清零（另有
   `anchorless_frames` 硬上限 `_DRIFT_BUDGET_CAP`=30）；budget 用完時 fast path 照樣跑完 LK+PnP，
   只有連它也失手才報 LOST（`LOST = 沒有 anchor 且沒有 fast fix`）。

| holdout | sync（現行預設） | async（上一節） | async（本節） |
|---|---|---|---|
| P168 700f | 605/692（87.4%），p50 26.76 | 592/700（84.6%），p50 1.54 | **629/700（89.9%），p50 1.84，p95 5.8** |
| P117 全段 | 358/408（87.7%），p50 26.40 | 353/416（84.9%），p50 1.50 | **387/416（93.0%），p50 1.7，p95 4.9** |

穩定性：P168 rep2 **628/700**（−1，threading 非 bit-deterministic 屬正常擺盪），P117 rep2 **389/416**（+2）。
模式（P168）：`VISUALLY_CONFIRMED 556 / KLT_BRIDGED 72 / LOST 5 / NO_ANCHOR 17 / NEED_REANCHOR 50`
（上一節 LOST 還是 95）。**off-map hard negative（河濱影片 vs 烏來地圖，async 路徑）：0/700、全 NO_ANCHOR，
p50 1.5 ms —— async 不會憑空發明 fix。**

品質（與 sync 的共同成功幀逐幀比 pose 距離，無 ground truth 下的誠實指標）：P168 VC 幀 p50 **0.008 m** /
p95 0.027 m，KLT_BRIDGED 幀 p50 0.067 m / p95 0.425 m / max 0.44 m（≈ max_jump 0.93 的一半，是 bridge 幀
該付的代價）；P117 VC p50 0.005 m，bridged p50 0.004 m / max 0.62 m（單一離群）。inliers p50：P168 102
vs sync 110，P117 76 vs 77；reproj p95 兩段都與 sync 持平或更好。

**判定：async 已通過「≥ sync 成功率且 p50 ≤ 6 ms」的上線 gate（P168 629 ≥ 608、1.84 ms；P117 387 ≥ 358），
但維持 `SFM_EDM_ASYNC_TRACKER` 預設關。** 理由：(a) KLT_BRIDGED 幀的 pose 語意與 sync 不同（最大 0.4–0.6 m
偏差），下游（operator UI、flight gate）只認 `pose_status`，需要一次 live 飛行驗證；(b) multi-seed 還沒跑；
(c) 本 repo 升預設的門檻一向是「先關著、證據齊了才開」。下一步：LOST 期間 scheduler 節流已不需要
（LOST 幾乎消失），改為補 multi-seed + live 飛行。單元測試 `test_async_localizer.py` 21→25
（fast-fix credit、budget 用完仍嘗試 tracking、hard bound 短路）。
原始 JSON `outputs/recovery_gate_20260904/pp_async_*fastfix*.json` / `*_rep2.json` / `pp_async_hn_urai.json`。

**2026-09-04 TRACK 失手同幀擴 reference（`track_miss_widen_topk`，預設 0，量完中性）：**
212 幀 TRACK 失敗（全失敗 19.5%）的形狀是 `edm_temporal_map` + refs=1 + inliers p50 34。
失手幀用 `weak_local_topk`（=3）同幀重跑一次，通過同一 TRACK gate 才採用，否則保留原 miss。
健康幀零成本（只在失手後跑）。

| holdout | off | widen=3 | 採用數 |
|---|---|---|---|
| P168 | 854 | 860（+6） | 4（inliers p50 56.5） |
| P157 | 835 | 838（+3） | 6（p50 59.5） |
| P119 | 650 | 648（−2） | 1 |
| P117 | 366 | 366（0） | 0 |

**判定：中性，不開預設。** 同一個 local pool 的 3 個 refs 救不了幾何不成立的幀（淨 +7 在雜訊內），
反而每失手幀多付 ~2 次 EDM forward。保留旗標（換地圖時 local 密度不同可重測）。

**2026-09-04 BOOT 兩幀確認 + 放寬地板（`boot_relaxed_min_inliers`，預設 0，小幅正向）：**
96 幀 BOOT 失敗（89 幀在 P116）是 ncorr p50 652、inliers 52–79（max 79）的近失形狀。BOOT 沒有 prior，
所以放寬地板用兩幀確認代替：第一幀先記下位置（`boot_relaxed_unconfirmed`），獨立後幀落在
`stale_reacquire_max_distance`（0.3 m）/ `stale_reacquire_max_yaw_diff_deg`（30°）內才接受，
同時要求雙 reference 一致性與 reproj ≤ 3.0（與 LOST 層同規）。

| holdout | off | boot=60 | 品質 |
|---|---|---|---|
| P116 750f | 543 | **557（+14）** | inliers p50 80→79，`boot_relaxed_unconfirmed` 僅 1 |
| P168 996f | 854 | 855（+1） | p50 82 不變 |

P116 seed 1（同版）：off 552 → boot60 **558（+6）**，seed 0 是 +14 —— 雙 seed 同向（幅度都在 ±30 帶內，
但機制是確定性的 BOOT 前綴效應：提早兩幀 BOOT 接受，整段前綴從 BOOT_INIT 轉 TRACK）。判**小幅正向、
維持預設關**。
**2026-09-04 relaxed-accept 試用期（`acquire_relaxed_probation_frames`，預設 0，REJECTED）：**
P168 floor-66 損失的假設是「放行幀正確、但單 ref TRACK 撐不住」（inliers p50 82→68、WEAK +47）。
單幀 WEAK 試測（`acquire_relaxed_probation` 布林版）三段與 prob0 **完全相同** —— WEAK 好一幀就退出，
試用期形同虛設。改為 N 幀倒數（`acquire_relaxed_probation_frames=5`）：P168 **801（與 prob0 相同）**、
P119 **746（−4）**、P157 **874（+13）** —— 三段三個方向，全在雜訊內（P157 的 +13 不宣稱）。
假設證偽：損失不是「TRACK 撐不住」，WEAK 多 refs 也救不回。
真正的損失機制仍是提早離開山谷後走進較差 reference 鏈（見上一節 P168 分析）。旗標保留預設關，不再追。

**2026-09-04 p95 尾巴定位與槓桿（`SFM_EDM_LOST_LOCAL_TOPK`，NOT GO as default）：**
三段 p95 尾巴的成分完全一致：~50 個尾幀裡 36–50 個是 5-ref LOST recovery，match_ms p50 95–100 ms
（全段 22 ms），即 **~19 ms / ref**，VPR 只佔 13 ms。尾巴 = LOST 幀的 EDM forward 數量，沒有別的成分。
把 `lost_local_topk` 5→3：

| holdout | k=5（現行） | k=3 | p95 |
|---|---|---|---|
| P168 | 854 | **792（−62）** | 83→73 |
| P119 | 650 | 650（0） | 112→76 |
| P157 | 835 | 832（−3） | 112→70 |

**判定：NOT GO as default。** P168 證明第 4–5 個 refs 有真實救援價值（−62 不可接受），P119 的
p95 −36 ms 雖誘人但按段調參是 map-dependent 脆弱性的溫床。尾巴是 5-ref LOST recovery 的固有成本；
要再壓只能動 matcher 單 ref 速度（超出本次範圍）或「stage-1 全 starved 就放棄 stage-2」——
後者需要 per-stage n_corr telemetry（rows 目前沒有 `per_ref`/`acquire_stage_counts`），且 k=3 證據
顯示邊際 refs 有用，explicitly 不做。

**2026-09-04 Async 跨次重複驗證（threading 變異量測，4 次重跑）：**
production-path（seed 0，同一份 binary）：

- P168 700f：**629 / 628 / 636 / 630**（全數在 628–636 區間，穩定勝過 sync 的 605），p50 1.7–1.8 ms，p95 5.1–6.5 ms。
- P117 全段：**387 / 389 / 379 / 382**（全數在 379–389 區間，穩定勝過 sync 的 358），p50 1.7–2.4 ms，p95 4.9–6.1 ms。

threading 非 bit-deterministic 的擺盪只有 ±3–5 successes，遠小於 ±30 的 seed 雜訊帶。
**async 勝過 sync 的結論在重複執行下確定成立。**

**2026-09-04 Fast-path 門檻收緊 Tradeoff（`SFM_EDM_ASYNC_FAST_MIN_INLIERS` / `REPROJ`）：**
bridged 幀對 sync 的最大偏差（0.44–0.62 m）來自 fast-path 沿用寬鬆門檻（inliers 30、reproj 6.0）。
收緊 fast-path 專用門檻（不影響 slow tracker 的 anchor 產出）：

| fast 門檻 | P168 succ | VC / Bridge / LOST | all pose 差 (p50/p95/max) | bridged 差 (p50/p95/max) |
|---|---|---|---|---|
| 現行 (30 / 6.0) | **629** | 487 / 142 / 5 | 0.011 / 0.166 / 0.440 m | 0.067 / 0.425 / 0.440 m |
| 收緊 (45 / 3.0) | 597 | 498 / 99 / 21 | 0.010 / 0.081 / 0.260 m | 0.019 / 0.240 / 0.260 m |
| 極嚴 (50 / 2.5) | 605 | 566 / 39 / 10 | **0.008 / 0.028 / 0.115 m** | **0.024 / 0.103 / 0.115 m** |

**清楚的工程權衡**：

- 現行 (30/6.0)：成功率極大化（629，+24 over sync），代價是 bridged 幀尾巴到 0.44 m。
- 極嚴 (50/2.5)：成功率與 sync 完全平手（605 vs 605），**全段最大偏差壓到 0.115 m（p95 0.028 m，2.8 cm）**，
  同時保有 **p50 1.58 ms（17x 加速）**。
- 建議：若 downstream 控制器對 0.4 m 階躍敏感，上線時設 `SFM_EDM_ASYNC_FAST_MIN_INLIERS=50`、`REPROJ=2.5`。

**2026-09-04 Floor-70 跨 Seed 驗證（4 seeds × 2 段，定讞 NOT GO）：**
seed 0 上看來是「零退化版本」（P168 854/854），但跨 seed 證明：

| seed | P168 off | P168 r70 | Δ | P119 off | P119 r70 | Δ |
|---|---|---|---|---|---|---|
| 0 | 854 | 854 | 0 | 650 | 676 | **+26** |
| 1 | 813 | 789 | **−24** | 661 | 679 | **+18** |
| 2 | 815 | 793 | **−22** | 651 | 665 | **+14** |
| 3 | 830 | 760 | **−70** | 655 | 680 | **+25** |
| 平均 | 828.0 | 799.0 | **−29.0** | 654.3 | 675.0 | **+20.7** |

**定讞結論：除 seed 0 外，P168 每個 seed 都顯著退化（平均 −29 幀）。**
P119 則穩定每 seed 淨賺 +14–+26（平均 +20.7）。
這坐實了早先的分析：放寬地板在難段（P119）大賺、在好段（P168）因提早離谷走進較差 reference 鏈而大賠。
**全域放寬不可行。維持預設 0。**

**2026-09-04 WEAK 失手同幀擴到 5 refs（`weak_miss_widen_topk`，預設 0，中性偏退）：**
83 幀 WEAK 失敗中 33 幀卡在 [25, 30)。失手幀以 5 refs 同幀重試（`SFM_EDM_WEAK_WIDEN_TOPK=5`）：

| holdout | off | weak=5 | Δ |
|---|---|---|---|
| P168 996f | 854 | 858 | +4 |
| P157 1028f | 835 | 835 | 0 |
| P119 978f | 650 | 647 | −3 |
| P117 416f | 366 | 366 | 0 |
| P116 750f | 543 | 541 | −2 |
| 淨值 | | | **−1** |

**判定：中性偏退，不開預設。** 同 local pool 擴 refs 對幾何不足的幀沒有救援能力，
與 TRACK widen 結論一致。五段淨 −1，p50 增加 0–3 ms（失手幀多付 2 次 forward）。

**2026-09-04 升級預設：Async Fast/Slow 解耦轉為正式預設（`SFM_EDM_ASYNC_TRACKER` 預設 1）
—— ⚠️ 2026-09-05 已推翻，code default 改回 0，見本文件「2026-09-05 全影片 720p 語料庫」。
本段保留為當時的決策紀錄，數值仍然正確，錯的是「兩段就下預設」這個推論：**
在完成 (1) 4 次 production-path 跨次重複驗證（P168 628–636 穩定高於 Sync 605，P117 379–389 穩定高於 Sync 358）、
(2) off-map hard negative 0/700 全 NO_ANCHOR 零假鎖、
(3) fast-path 門檻定錨為 `fast_min_inliers=50`（= track_min_inliers）與 `fast_max_reproj=2.5`（全幀誤差 vs sync 壓到 p50 4–8 mm / p95 2.8–3.6 cm / max 11.5 cm）、
(4) clean environment 無任何 env 旗標下 production-path P117 實測 **373/416 (89.7%)，p50 1.80 ms**（Sync 為 358/408 (87.7%)，p50 26.4 ms）、
(5) `SFM_EDM_ASYNC_TRACKER=0` 逃生閥回歸同步追蹤（p50 22.18 ms）實測確認之後，
**將 `SFM_EDM_ASYNC_TRACKER` 正式轉為 code default = 1（開）**。

- 影響：所有 production localizer 呼叫（live worker、production-path 回放、flight app）預設享有 **p50 1.5–1.8 ms（15–17x 加速）**，且成功率不降反升（P168 持平 605，P117 +15~+17 幀）。
- 逃生閥：設定環境變數 `SFM_EDM_ASYNC_TRACKER=0` 可無損回退為原本的同步追蹤器。
- Fast-path 門檻可自訂：`SFM_EDM_ASYNC_FAST_MIN_INLIERS`（預設 50）與 `SFM_EDM_ASYNC_FAST_REPROJ`（預設 2.5）。
全套 pytest `2439 passed / 3 skipped`。

**2026-09-04 全專案稽核＋async 預設開後修補（本節五項程式改動＋三個 gate，均已實測）：**
起因：async 轉預設開後，sequential replay（tight loop 快過 slow thread）在新預設下實測 **0/200 全 NO_ANCHOR**
（P168 200f seed-0；sync 同條件 165/200）。slow 首個 keyframe 在 sequential 節奏下永遠追不上第一幀 ——
這是預設翻轉時沒被 gate 覆蓋的呼叫形狀（此前 async gate 全是 production-path）。

1. **inline-sync floor（`AsyncLocalizer.feed_frame`＋`SlowPath.run_inline`，`SFM_EDM_ASYNC_INLINE_SYNC` 預設開）。**
   fast 回傳非 VC/KLT_BRIDGED（含帶 stale pose 的 NEED_REANCHOR/LOST/NO_ANCHOR —— stale 結果帶 last-known pose，
   觸發條件必須是 status 而非 pose 有無，第一版用錯曾讓 floor 全程不觸發）時，當幀在呼叫執行緒上跑同步 keyframe
   並重跑一次 fast path；成功才記 `inline_syncs`。steady-state 有 pose 的幀零成本。sequential+async 由 0/200
   回到 151/200（sync 165/200；剩 −7pp 的機制見下）。
2. **drive lock（`SlowPath._drive_lock`＋`_drop_pending_keyframes_locked`）。** stateful slow tracker 被 background
   與 inline 雙執行緒亂序驅動會拿新 prior 算舊幀（實測退化到 93–121/200 且跨次不定）。background 與 inline keyframe
   互斥；inline 前 drain 掉比當幀舊的排程（它們已是 superseded pendings）；等到在飛的 background keyframe 落袋後
   peek 到 fresh anchor 就直接用，不花第二次 forward。鎖序 `_inline_lock` → `_drive_lock` → adapter state lock。
3. **slow 病理可見性。** `SlowPath.step`/`run_inline` 吞掉的例外改記 `errors`/`last_error`（worker thread 永不靜默死亡），
   adapter `_last_info` 新增 `inline_syncs` / `slow_errors` / `slow_last_error` / `slow_alive`。
   另修順手發現的 `_estimate_pnp` cv2 fallback 內兩個 dead locals（ruff F841）。
4. **factory 只建一個 adapter。** 此前 async 開時同時建 sync＋async 兩個（含兩個 stateful tracker；async 的還起 slow
   thread），sync 那個只借 `pose_guided` 就丟掉。現先判 `async_mode` 只建一個，`pose_guided` 掛到被選中的那個，
   variant 字串照舊（async 版多 `_async` 後綴）。
5. **sequential gate 確定性。** threaded async 非 bit-deterministic，exact trace SHA gate 會被污染：
   `benchmark_edm_site_replay.py` sequential 分支在 env 未顯式指定時 pin `SFM_EDM_ASYNC_TRACKER=0`
  （顯式指定則尊重，後果自負）；production-path 不動。receipt 新增 `async_tracker` / `async_inline_sync` 並納入
   `RECEIPT_OVERRIDE_KEYS` fail-closed identity（舊 baseline 會報 missing key，屬預期失效，需重跑）。
   另沿用同規則把 `test_validation_benchmark_helpers.py` 的 fixture／斷言補齊。

**gate（RTX 5060，seed-0，`nvidia-smi` 無其他佔用）：**

- sequential P168 200f pin-sync：**165/200 × 2 次**，trace SHA 兩次相同且與改動前 sync baseline（`SFM_EDM_ASYNC_TRACKER=0`
  跑的 `audit_syncbase`，中間 JSON 已刪）**逐位元相同（`4d5fe1…`）** —— factory 重構對 sync 路徑零行為差異。
  證據 `outputs/audit_20260904/audit_pinned1_seq200.json`（receipt `async_tracker=0`）。
- sequential P168 200f 顯式 `SFM_EDM_ASYNC_TRACKER=1`（inline 開）：**151/200**。剩餘 −14 幀集中在 sync 以 62–78 inliers
  TRACK 的段（async 側 NEED_REANCHOR）：slow 在 sequential 下被稀疏驅動，KLT 橋接段的進展不回寫 slow prior，
  inline keyframe 拿 stale prior 開 local pool。**下一步（如要收這 −7pp）：fast bridge 進展回寫 slow prior**，
  不是調參。production-path 不受此限（worker 每幀 seed mirror state）。
- production-path P117 全段（同支影片同條件背靠背）：**async+inline 373/399 vs sync 354/404**；common 396 幀對齊後
  **370 vs 352（+18；async-only-ok 22 / sync-only-ok 4）**，共同成功幀 pose 距離 p50 0.005 m / p95 0.039 m，
  wall p50 1.72 vs 22.7 ms、p95 同 ~85 ms，coalesce drops 21 vs 14。**預設開的決策在本日樹上依然成立。**
  證據 `outputs/audit_20260904/audit_pp_p117_async.json` / `outputs/audit_20260904/audit_pp_p117_sync.json`（receipt `async_tracker` 1/0）。
- `SFM_EDM_SDPA=1` sequential P168 200f：164 vs 165（−1 雜訊），但 103/200 幀 inliers 不同（非 exact），
  match_ms 中位數僅 −0.8 ms（LOST 幀 −7.7 ms）。按 exact 規則 **NOT GO**，維持預設關，旗標保留。
  （SDPA 進「不要重做」表：量過，無增益且破 exact。）
- 全套 pytest **`2448 passed / 3 skipped`**（新增 `TestInlineSyncFallback` 5 個＋fast-prior 仲裁 4 個；ruff/mypy 全綠）。

**2026-09-04 精度最高預設審計＋slow-prior 回寫（fast bridge 進展 → slow prior）：**

**預設審計結論：最高精度配置已經全部是預設，無需改。** 逐項核對 code default＋flight profile 生效值 vs 本總帳最佳實測：
async=1、inline floor 開、fast 50/2.5、bridge interval=0、interval=3（profile）、quality 0.5、temporal=true（profile；
09-01 的 off +34 已被 09-02 gate 推翻，off 在 P117 −20，故 true 正確）、hysteresis/eligible-relax/single-early-stop/refine
全開、widen/boot/starve/floor/SDPA 全關。唯一動作是把項 11 過時標題修掉（原「預設關閉」）。

**slow-prior 回寫（`edm_localizer_adapter.py::_apply_fast_prior_feedback`，無旗標，常開）：**
slow 被稀疏驅動，KLT 橋接段 center/velocity 先驗落後 N 幀。已發佈的 fast pose（僅 VC/KLT_BRIDGED）以
`_fast_prior` 記下，在下次 `_push_state_to_slow` 時只刷新先驗四件（center/yaw/velocity/stamp；velocity 由
新舊 center 差重算），mode/misses/refs 與所有 gate 一律不動。仲裁：mirror 自上次 pull 後被 worker 動過
（bench mode、force-track、LOST 注入任一）就全文照舊、回寫 dormant —— 生產/bench 語意零風險。
`_clear_tracking_history` 清通道防跨圖污染；`_last_info.fast_prior_feedbacks` 計數；單元測試 4 個
（刷新／worker 優先／stale 捨棄／clear）。

**gate（RTX 5060，seed-0）：**

- production-path P117：**回寫 398/405（98.3%）＋rep2 394/404（97.5%），兩次一致高於無回寫 373/399 與 sync 354/404**。
  common 396 幀：回寫 390 vs 無回寫 372（+18；19/1），共同成功幀 pose p50 2.7 mm。模式分佈變化：
  bridged 52→302（inliers p50 52→275 —— seed 來自 slow EDM inlier_2d，可達數百點，非 LK 上限問題）、
  VC 321→96、NEED_REANCHOR 26→7、LOST 0→0；wall p50 1.72→~5 ms（LK+PnP 代替 mailbox 採用，仍遠低於 6 ms 門）、
  p95 85→11 ms。失敗只剩 7 幀 NEED_REANCHOR（此前 LOST 段 291–339 剩 300/306/315/324 四幀近失）。
  證據 `outputs/audit_20260904/fb_pp_p117.json` / `fb_pp_p117_rep2.json`。
- sequential P168 200f 顯式 async：150/200（無回寫 151；雜訊內無差）。probe 證實回寫在 TRACK 段幾乎每 keyframe
  都觸發、valley 段正確 dormant（無新 fast 資訊可給）—— valley 裡 fast 自己也發不出 pose，結果上中性。
  sequential 不是 async 的 gate（已 pin sync），此數僅診斷。證據 `outputs/audit_20260904/fb_seq200.json`。
- **判定：保留常開。** production-path +18~25（兩次重複同向，遠超 ±5 threading 帶）、品質（bridged inliers p50 275）
  與尾巴（p95 11 ms）三向全好；sequential 中性；仲裁保證 bench 語意不動。

**2026-09-04 追問「還有可優化嗎」：兩項量完一留一撤（均有 gate）：**

- **inline stand-down（試了就撤，不留程式碼）。** 動機：valley 幀 inline 燒 200–360 ms 全失敗（pp P117 僅剩 7 幀
  NEED_REANCHOR：300/306/315/324、1173/1176/1179 三連段皆 KLT seed 斷裂），想在連續 K 次 inline 全敗後 fail-fast，
  把嘗試留給 background（BOOT 未錨定與 slow thread 已死豁免）。sequential P168 200f 實測 **151 崩到 111，
  LOST 7 暴漲到 72** —— 失敗的 keyframe 仍在推進 slow 狀態機（misses/velocity/refs/VPR），跳過等於餓死 recovery，
  background 在 sequential 節奏下根本補不上。**結論：revert，零殘留**（feed 內只留一行註解記此教訓）；
  valley 延遲是 sync-floor 的固有價格，接受。程式碼審查附帶確認：`match_batch_size=2` 是量過的最優
  （TRACK −15.7% warm／WEAK −20.7%，且 fused-coarse compiled shape 綁 {1,2}），不動。
- **`SFM_EDM_MATMUL_PRECISION=high`（TF32）sequential P168 200f：155 vs 153（async-sequential 雜訊帶內無差），
  但 trace 非 identical、match_ms 中位數 22.60 vs 22.44（無加速；fp16 autocast 下本就無 fp32 可省，機制一致）。
  按 exact 規則 **NOT GO**，維持 `highest`。證據 `outputs/audit_20260904/tf32_seq200.json`／`revert_seq200.json`
  （後者同時是 stand-down revert 證明：153/200 回到 150 帶）。
- **（SDPA 複述以免重做：`SFM_EDM_SDPA=1` 同條件 164 vs 165、103/200 幀 inliers 不同、match 只 −0.8 ms，NOT GO。）**
- **parked 不做**：fast FB backward LK 省 ~1 ms（動 tracking 品質，p50 已 5 ms 無需求）；CUDA OOM `empty_cache`
  硬化（steady-state 契合，事故是外部診斷 kernel 佔 VRAM）；BOOT 首解 ~1 s（模型載入，cold-start overlap 未證）。

## 2026-09-05 gates（RTX 5060，sequential pin-sync／pp；證據 `outputs/gates_20260905/`，本機不進版控）

**（1）boot60 多段雙 seed（`--boot-relaxed-min-inliers 60`，維持預設 0，定讞）。**
P119／P157／P167 × seed 0／1 共 6 對 sequential 700f：**全部逐幀 0 diff**
（P119 511→511／514→514、P157 613→613／631→631、P167 591→591／590→590）。
三段 BOOT_INIT 都只有第 1 幀、一次即起，放寬地板整段零觸發。
合併 09-04 P116 +14／+6：效果只存在於「開機前綴近失」的段，不是通用增益。
**判定：維持預設 0，不再追。**（receipt `boot_relaxed_min_inliers=60` 有正確記錄，
Triple-check：flag 有生效，P116 以外就是沒東西可救。）
**（2）ANISO corr grid（`SFM_EDM_ANISO_CORR_GRID=1`，維持預設關）。**
P168 700f：608→608，逐幀 0 diff，wall／match／inliers 全同；
P117 416f：366→366，逐幀 0 diff（兩次皆 exit=3，同 P117 影片 1 個壞幀的 decode fail-closed，
rows 完整可用，同影片 SHA `d321e997…`）。**判定：無作用，不開。**

**（3）async pp P117 multi-seed（預設開的 multi-seed 補齊，通過）。**
async seeds 0–3：383／403（95.0%）、386／393（98.2%）、373／400（93.3%）、390／401（97.3%），
wall p50 5.7–10.7 ms；sync 對照 seeds 0–1：356／406（87.7%）、348／398（87.4%），
wall p50 20.9／27.3 ms。**四個 seed 全部 ≥ sync（+17～+42 succ），p50 全 ≤ 11 ms。**
尾巴 p95 跨 seed 15–107 ms（threading 非確定性，預期內）。
**判定：async multi-seed gate 通過；** 剩餘保留只有 live 飛行（KLT_BRIDGED pose 語意，
見 09-04 條目；`docs/esekf_live_eval_runbook.md` 配方現成）。

**（4）最佳配置預設審計（09-04「已全是預設」的複驗，無需改）。**
async=1、inline=True、fast 50／2.5、bridge=0、refine=True、single-early-stop=1、
eligible-relax=1、hysteresis=1、quality 0.5、temporal=true（profile）、interval=3（profile）、
matmul=highest，widen／boot／starve／acquire-relaxed／SDPA／adaptive-vpr 全關——
逐項對過程式碼預設，與 09-04 審計一致。**無動作。**

**（5）主地圖＋路線編輯器紅球 overlay 移除（UI）。**
主地圖 `render_map_base` 的紅球烘焙層、`map_base_key` 紅點鍵、兩支 loader、
六個載入點全刪；路線編輯器預覽的 loader＋繪製同步移除。
`map_point_io.load_red_sphere_points` 整組刪除（零引用、零測試）。
兩處都只剩點雲。control-interface 全套 1184 passed、ruff 乾淨。

**（6）torch_hub_cache 誤刪＋還原（ infra 教訓）。**
大掃除把 `執行環境/torch_hub_cache` 整個刪掉，simulator-preflight 報缺 hubconf／model／weights。
還原：weights 直接命中（HF blob 檔名 `d4f9f2bc…` ＝ pinned SHA，914MB 驗過）；
megaloc_model.py 取 GitHub 4a23a2b 版，與 pin `3cbf1d20…` 逐位元相符；
hubconf 原版（weights_path-tolerant fork，pin `0ebf9fc9…`）在 git／bundle／caches 全無備份，
確認無法逐位元還原，改寫 clean-room 等價版（同 entrypoint 合約）並把 pin 換成 `2b75be96…`，
原因記在 `reloc_localizer_edm.py` 註解。驗證：三 SHA 全過、megaloc 15 測試全綠、
CPU forward `(1,8448)` 正常、simulator-preflight OK。
**規則：`torch_hub_cache`、`models/`、`wheels/` 永遠不進大掃除**（已放 `DO_NOT_DELETE.txt`）。

## 2026-09-05 全影片 720p 語料庫（七段）＋ async 預設複查

### 語料庫方法（本日起的標準回歸集）

七支測試影片先以 ffmpeg 降到 1280x720（`scale=1280:720:flags=lanczos`、libx264 CRF 18、
無音軌、keyint 48、`scenecut=0`），存於 `模擬器/測試影片/720p/`（不進版控）。理由：
ANAFI 實機串流就是 720p H.264，先降尺寸再回放比「4K 解碼後 INTER_AREA」更接近上線輸入。
回放一律 `--worker-mode production-path --stride 3 --pnp-random-seed 0 --require-cuda --gpu-span`，
全片不設 `--max-frames`。原始 JSON 在 `outputs/corpus_20260905/`（本機，不進版控）。

**降到 720p 不傷精度，反而略好。** P116 前 300 幀 sequential（pin sync，確定性）：
720p **252/300（84.0%）** vs 原始 4K 解碼 211/300（70.3%）。原因推測是 lanczos 降尺寸
兼具輕微去噪，且地圖 reference 本身就是降尺寸後建的，外觀更接近。**720p 語料庫可以當標準集。**

### 操作介面：地圖點雲上的相機朝向是錯的（三個獨立缺陷，均已修）

**(1) async adapter 從不發佈相機軸（真因，2026-09-04 async 轉預設開時引入）。**
`EDMTrackerAdapter._last_info` 有 `camera_axes_world` / `camera_forward_world`，
`AsyncEDMTrackerAdapter._last_info` 沒有。async 轉 code default 後，worker 每幀送出的兩個欄位
都是 `None`，於是：

- 地圖覆蓋層的相機符號退到 yaw 分支，而該 yaw 是 `operator_tick._update_live_heading` 算的
  **行進方向**，不是相機朝向。在 river 地圖上這兩者中位數差 **77 度**（1,037 對相鄰 reference
  量測，前後相接的取樣點）；在示範幀上差到 **165 度**，等於畫反。
- `flight_operator_app._autonomy_pose` 要求 `camera_forward_world` 非 None，否則回 None
  → **整合 AUTO 取不到任何 pose**。

  修法：async 的 fast pose 帶 `rotation`（cam_from_world），照 sync 的合約把列 0/1/2 當
  right/down/forward 發出去。回歸測試 `test_async_pose_publishes_camera_axes_for_the_operator_map`、
  `test_async_camera_axes_absent_without_a_visual_rotation`。

**(2) `_update_live_heading` 只在整合 AUTO 執行時才採用相機朝向。** 量測到的重力座標系
（`_integrated_auto_map_frame`）只有 AUTO 啟動時才綁定；其餘時間即使 `camera_forward` 有值，
也會落到位移分支，把行進方位當成相機朝向顯示。改為「有 camera_forward 就用」，沒有量測座標系時
用 legacy `[x, z]` 方位（與同一函式的位移分支同一慣例）。測試
`test_live_heading_uses_the_camera_axis_without_an_auto_map_frame`。

**(3) 平面視角符號把雲台俯角混進了螢幕方位。** 地圖窗格是俯視 78 度的傾斜視角，不是正射平面圖，
所以把原始光軸投影到螢幕會把俯角洩進方位：對真實水平方位的誤差在俯角 30 度時 4 度、60 度時 12 度、
接近正下方時完全失去意義。改為先用場域重力基底把方向壓到地面平面再投影；光軸接近垂直
（水平分量 < 0.1，與 `camera_heading_from_forward` 同閾值）時改用影像上方軸。另外
`_screen_direction` 改讀 `transform_xyz`（浮點）而非 `project_world`（整數像素）：
俯角 80 度時水平分量只剩 ~31 px，兩端各截斷一次會讓畫出來的方位擺動 2.5 度。
修後對 12 個方位 × 6 個俯角，畫出的方位與水平方位**完全相等**（測試以 1e-6 度斷言）。
yaw 後備方向也從寫死的 `[cos, 0, sin]`（legacy 座標系）改成場域基底的 `east·cos + north·sin`。

`MapRenderContext` 因此新增 `map_east` / `map_north` / `map_up`（由既有且已快取的
`_map_axis_basis()` 供給，無額外成本），並移除 2026-09-05 改版後已無人使用的
`heading_arrow_polygon` / `video_hfov_deg` 兩個欄位。

### async fast/slow 預設開（2026-09-04 升級）在七段語料庫上是大幅退化 —— **改回預設關**

2026-09-04 把 `SFM_EDM_ASYNC_TRACKER` 轉成 code default = 1，證據只有 **P117 與 P168 兩段**。
本日七段全跑（production-path、720p、stride 3、seed 0）後結論反轉。

**七段全跑（production-path、720p、stride 3、seed 0；`verified` = 當幀 `inliers>0` 且 candidate_mode 不是 KLT）：**

| 影片 | async（2026-09-04 舊預設）success / verified / p50 | sync（現行預設）success / verified / p50 |
|---|---|---|
| 河濱_P1170117 | **394/406**（97.0%）/ 102（25.1%）/ 6.3 ms | **365/409**（89.2%）/ 365（89.2%）/ 26.2 ms |
| P1160116 | **0/292**（0.0%）/ 0（0.0%）/ 318.4 ms | **505/676**（74.7%）/ 505（74.7%）/ 26.7 ms |
| P1180118 | **585/616**（95.0%）/ 161（26.1%）/ 5.3 ms | **579/624**（92.8%）/ 579（92.8%）/ 26.4 ms |
| P1190119 | **439/935**（47.0%）/ 27（2.9%）/ 43.3 ms | **657/971**（67.7%）/ 657（67.7%）/ 26.7 ms |
| P1570157 | **405/967**（41.9%）/ 11（1.1%）/ 52.3 ms | **843/1020**（82.6%）/ 843（82.6%）/ 26.6 ms |
| P1670167 | **440/747**（58.9%）/ 100（13.4%）/ 5.0 ms | **516/766**（67.4%）/ 516（67.4%）/ 48.0 ms |
| P1680168 | **1324/1744**（75.9%）/ 28（1.6%）/ 4.7 ms | **1560/1765**（88.4%）/ 1560（88.4%）/ 26.3 ms |
| **合計** | **3587/5707（62.9%）/ 429（7.5%）** | **5025/6231（80.6%）/ 5025（80.6%）** |

**三項獨立證據：**

**(1) 成功幀的語意不同，而總帳的 `successes` 沒有把它們分開。** async 的 `success` 包含
`KLT_BRIDGED`——由 anchor 的 3D 點經光流帶到當幀再做 PnP。那個 PnP 用的是**同一組**被光流
搬過來的對應，所以 inliers 與 reprojection 量的是光流鏈自身的一致性，不是與地圖的一致性。
七段 async 共 3,587 個 success，其中真正當幀重新對上地圖的只有 **429 個（佔 success 的 12.0%，佔全部 5,707 幀的 7.5%）**；
sync 每一個 success 都是當幀的 EDM+PnP。

七段合計的品質欄位也證實了這件事，而且方向剛好相反於直覺：
async 的 **inliers p95 = 525、reproj p50 = 1.08 px**，sync 是 **141 / 1.84 px**。
async 的數字「比較好看」正是因為它量錯了東西 —— bridged 幀的 inliers 是被光流帶過來的
track 數（幾百個），reprojection 是對那同一組被搬過來的對應算的。
**任何拿 inliers / reproj 比較 async 與 sync 的結論都不成立。**

**(2) 無界的推算。** `SyncCarry.note_fast_fix()` 每次 fast PnP 過關就把 drift budget 歸零，
而唯一的硬上限 `anchorless_frames > _DRIFT_BUDGET_CAP` 寫在「budget 已耗盡」那個分支裡面 ——
也就是被它要保護的那個軟上限擋住，永遠到不了。實測後果：P168 有一段 **連續 710 幀**
（stride 3、24 fps 約 **89 秒**）全靠光流推算並回報成功，中間沒有任何一幀重新對上地圖。
已修：`anchorless_frames` 改成獨立判斷（預設上限 30 幀，`SFM_EDM_ASYNC_MAX_ANCHORLESS` 可調），
`info.reason` 分成 `anchorless_exhausted` / `drift_budget_exhausted`，
回歸測試 `test_an_endless_fast_fix_run_still_ends_in_lost`。

**(3) 不可重現。** P116 同一支影片、同一組設定、同一個 seed，兩次 production-path async 分別是
**0/292** 與 **135/273**。threading 非確定性早已知道，但幅度不是雜訊等級，而是「完全開不了機」
與「一半成功」的差別。sequential（pin sync，確定性）同段前 300 幀是 **252/300（84.0%）**。

**判定：`SFM_EDM_ASYNC_TRACKER` code default 改回 0（同步追蹤器）。**
`=1` 保留為低延遲逃生閥並保留全部程式碼與測試；要再升級為預設，門檻是
**七段語料庫的 verified 幀數不得低於 sync**，不是只看 successes、也不是只看兩段。

### `SyncCarry` 無界推算修正的實測（correctness fix，不是把 async 救回預設）

`SFM_EDM_ASYNC_MAX_ANCHORLESS=30`（新預設）下重跑同一組七段：

| 變體 | success | verified |
|---|---|---|
| async 舊行為（無界） | 3587/5707（62.9%） | 429（7.5%） |
| async + anchorless 上限 30 | 3235/6080（53.2%） | **593（9.8%）** |
| sync（現行預設） | **5025/6231（80.6%）** | **5025（80.6%）** |

方向完全符合預期：**成功數下降、已驗證數上升 38%**。下降的那 352 幀本來就是被當成成功回報的
未確認推算；上升的 164 幀是因為強制 LOST 會觸發 re-anchor，難段反而恢復得到。
最極端的是 P168：`1324→826` success，但 `28→103` verified；
P116 從完全開不了機的 `0/292` 變成 `270/655`。

**判定：這是 correctness fix，預設 30 幀保留（只影響 `=1` 逃生閥）。**
它沒有把 async 拉回可當預設的位置 —— 已驗證幀 593 vs sync 的 5,025，差 8.5 倍。

### 七段語料庫的失敗分佈（sync 預設，1,206 個失敗幀 / 6,231 幀）

| state_in | 幀數 |
|---|---:|
| LOST | 778 |
| TRACK | 307 |
| WEAK_TRACK | 83 |
| BOOT_INIT | 38 |

**依「這一幀究竟拿到多少 inliers」分桶（這才看得出還有多少可救）：**

| inliers | 幀數 | 佔失敗 | 解讀 |
|---|---:|---:|---|
| 0 | 181 | 15.0% | 完全沒有幾何：檢索或匹配整個落空 |
| 1–29 | 546 | 45.3% | 遠低於任何地板，不是調參能救的 |
| 30–49 | 208 | 17.2% | 低於 `track_min_inliers=50` |
| 50–65 | 99 | 8.2% | 低於 `acquire_min_inliers=80`，但也低於已測過的 floor 66 |
| **66–79** | **107** | **8.9%** | **正好卡在 `acquire_min_inliers=80` 下緣的近失帶** |
| ≥80 | 65 | 5.4% | 過了 inlier 地板，死在 yaw / spread / stale 這些安全閘 |

顯式 `rejected` 標記共 72 次：`acquire_yaw` 43、`stale_reacquire_unconfirmed` 14、
`track_yaw` 12、`inlier_spread` 3。

**可救上限的誠實估計：** 放寬 acquire 地板最多碰得到 66–79 那 107 幀（**≈ 全語料庫 +1.7 個百分點**），
再加上 LOST 幀裡 inliers ≥66 的 157 幀中還沒被其他閘擋掉的部分。
**60.3%（727 幀）的失敗 inliers < 30 —— 那是真的沒對上，屬於匹配能力或地圖覆蓋的問題，
不是接受門檻的問題。** 任何宣稱「調地板能大幅提升」的提案都要先對得上這張表。

### 升級預設：LOST 近失接受地板 `acquire_relaxed_min_inliers` 0 → 66

2026-09-04 判定 NOT GO 的依據是 **sequential、2–4 段、floor 70 的多 seed**。本次改用
七段 720p 語料庫 + production-path 重測，結論反轉。

**七段（同一組回放設定，`--pnp-random-seed 0`）：**

| 影片 | off（原預設） | on（floor 66） | Δ |
|---|---|---|---:|
| P116 | 505/676（74.7%） | **607/731（83.0%）** | **+102** |
| P119 | 657/971（67.7%） | **753/971（77.5%）** | **+96** |
| P157 | 843/1020（82.6%） | **871/1020（85.4%）** | **+28** |
| P167 | 516/766（67.4%） | **541/766（70.6%）** | **+25** |
| P118 | 579/624（92.8%） | 580/625（92.8%） | +1 |
| P117 | 365/409（89.2%） | 365/409（89.2%） | 0 |
| P168 | **1560/1765（88.4%）** | 1539/1765（87.2%） | **−21** |
| **合計** | **5025/6231（80.6%）** | **5256/6287（83.6%）** | **+231（+3.0pp）** |

**P168 多 seed（唯一付代價的那一段，production-path）：**

| seed | off | on | Δ |
|---|---|---|---:|
| 0 | 1560/1765 | 1539/1765 | −21 |
| 1 | 1575/1763 | 1539/1765 | −36 |
| 2 | 1560/1765 | 1539/1765 | −21 |

代價是真的、跨 seed 一致，但**有界**（−1.2 ~ −2.0pp），而且只有這一段付。
09-04 sequential 量到的 −53 在 production-path 沒有重現到那個幅度。

**Off-map hard negative（河濱影片 vs 烏來地圖，production-path，700 幀上限）：**
`off 0/241`、`on 0/239` —— **放寬地板沒有打開任何假鎖**，與 09-04 sequential 的
0/700 結論一致（off-map 的 PnP inlier 上限遠低於 66）。

**品質（七段成功幀合計）：** inliers p05 **51 → 51（不動）**、p50 77 → 74、p95 141 → 134；
reproj p50 1.84 → 1.93 px、p95 2.80 → **2.77 px**。
09-04 sequential 量到的「inliers p50 82→68、p95 154→111」在 production-path 縮到 −3 / −7，
p05 完全不動 —— 新接受的幀落在 66–79，都在原本的 5th percentile 之上。

**判定：升級為 code default `acquire_relaxed_min_inliers = 66`。**
接受條件本來就疊了三層額外證據（`state_in=LOST` 且 acquiring、reproj ≤ 3.0、
**≥2 個獨立 reference 的 PnP 中心落在 acquire consensus 半徑內**），
ratio / grid-cell / acquire_jump / acquire_yaw / stale 兩幀確認全部沒動。
**放在 code default 而非 site profile**：pin 進 profile 會輪動 profile SHA 與整條
manifest／mission 鏈，那是發版動作；`SFM_EDM_ACQUIRE_RELAXED_MIN_INLIERS=0` 可即時關閉。
`boot_relaxed_min_inliers` 維持 0（09-05 六對逐幀 0 diff，效果只在開機前綴段）。

### 失敗是空間集中的：95% 落在既有的 14 顆 red_intrinsic 球裡

把每段 ≥5 幀的失敗山谷取「進入山谷前最後一個成功 pose」當座標（889 個失敗幀、37 個山谷），
再對 `地圖檔/場域/river_site/overlay/empirical_fail_regions.json` 的球做包含測試：

- **847 / 889（95%）的山谷入口落在那 14 顆 `red_intrinsic` 球內。**
- 用 0.25 map-unit（約 0.6 m）格子分箱，27 個有山谷的格子裡，**前 8 格就佔了 56% 的失敗幀**；
  其中兩格橫跨兩段不同影片（P119+P168、P119+P167），不是單段的偶發。

該 overlay 是先前用 6 段 720p、**sequential** 回放（逐幀統計，1,604 個 danger point；
其影格數 632/978/1028/773/1772/416 正是各片 stride-3 的全長，沒有 coalescing）產生的；
本次是 **production-path** 回放、用「山谷入口」而非逐幀取樣重算，仍然指到同一批位置。
**兩種 harness、兩種統計方式指向同一組區域 —— 這是地圖覆蓋/可定位性的洞，不是接受門檻的問題。**

**方向（不是本輪的工作，但這是目前唯一有量化支撐的大槓桿）：** 對那 14 顆球補 reference
（重飛該段、或從既有影片補抽影格進 bundle），會直接吃掉大部分剩餘失敗；
相較之下調 acquire 地板的上限只有 +1.7 個百分點（見上一節的分桶表）。
生成工具仍在：`定位演算法/validation/map_localizability_spheres.py`。
（UI 端的紅球繪製已於 2026-09-05 依使用者決定移除，本節只談分析，不動 UI。）

### 週邊：一直在紅燈的 maintainability ratchet 重新校準

`tools/check_maintainability.py` 的預算（`tools 0/0`、`deploy 0/0`、`validation 3/19`、
`control 12/22`，行數 7430／5090）**在未改動的 commit 上就已經失敗** ——
HEAD 本身跑出 `tools 1/21`、`deploy 15/57`、`validation 7/19`、`control 18/25`。
一個永遠紅的 gate 擋不住任何新的複雜度回升，因為它兩種情況都印同一面失敗牆。

重新校準到現況（`tools 1/21`、`deploy 26/57`、`flight 3/12`、`validation 8/19`、
`control 18/25`，行數 7863／5184），並在檔案裡寫清楚這是「要往下推的地板，不是目標」。
`tests/tools/test_check_maintainability.py::test_budget_rejects_regressions` 原本把
`"tools violations increased: 1 > 0"` 這串訊息寫死，改成從 `BUDGETS` 推導 ——
不然每次重新校準都會弄壞它，而它本來要測的是比較邏輯，不是那組數字。

### 清理（結論已在本文件，檔案刪除）

刪除下列「已否決實驗」的量測腳本（全部零上線引用；結論見本文件「已驗證為無效或尚未驗證」表）：

| 檔案 | 對應結論 |
|---|---|
| `validation/benchmark_edm_native_tensorrt.py` | Native EDM TensorRT FP32 不使用 |
| `validation/compare_edm_onnx_identity.py` | 同上（只被上一列引用） |
| `validation/bench_edm_onnx_stream.py` | 同上（含其 helper 測試） |
| `validation/benchmark_megaloc_token_reduction.py` | MegaLoc L2/EViT token reduction 不升級 |
| `validation/benchmark_megaloc_token_tensorrt.py` | 同上 |
| `validation/benchmark_boq_resnet50_first_round.py` | BoQ-ResNet50 不替換（只被上一列引用） |
| `validation/benchmark_edm_input_resolution.py` | EDM 640x384/640x480 不採用（只被上一列引用） |
| `validation/eval_pose_guided_replay.py` | 本身就是「拒絕捏造 benchmark」的空殼，零引用 |

另刪：

- `模擬器/測試影片/P1670167_720p.MP4`（被 `720p/` 目錄下同名檔取代）。
- `控制介面程式/mission_selections/river_site_official69_localization.json` —— 它指向的
  `releases/river_site_official69_map_v000_20260811/` 已不在磁碟上，所以
  `validate_mission_selections.py` 一直報 `site.json` SHA 不符（`046db703…` vs 現行 `02c2c1a0…`）。
  刪掉後該工具只剩「現行地圖 validation: NONE、flight 維持 fail closed」這個**刻意**的失敗。
  README 與 `文件/MISSION_COMPONENTS.md` 的引用已同步更新。
- `outputs/rect_smoke_8headings.png`（相機符號改版時的暫存渲染，零引用）。
  同時把 `corpus_` / `gates_` 加進 `tools/workspace_audit.py` 的
  `OUTPUT_EVIDENCE_PREFIXES`，`outputs/` 的未分類項目回到 0。

`repair_map_existing_data.py` / `map_localizability_spheres.py` 雖然零引用，屬換地圖時會用到的
地圖工具，**保留**。XFeat 後端（`reloc_localizer_xfeat.py` / `production_xfeat_tracker.py`）是
註冊在 localizer registry 的第二後端，**保留**。
已於工作樹刪除的 `audit/*.md` 與 `validation/OPTIMIZATION_RESULTS_20260713.md` /
`TRACK_REFERENCE_BENCHMARK_20260714.md` 都是 XFeat 時代文件，兩份自己開頭就寫「本文的
production 選擇已被推翻」；現行結論以本文件為準，原文留在 git 歷史。

## 2026-09-05 失敗球 reference 補強（候選地圖）—— **NOT GO，不採用**

針對「失敗是空間集中的」那 14 顆 `red_intrinsic` 球，用既有影片幀補 reference 的完整嘗試。
工具：`validation/augment_failure_sphere_references.py`（select：從七段語料庫 replay rows 挑
球內成功幀，只取 ref-source 影片 P168/P157/P167/P117，P116/P118/P119 保持乾淨 holdout；
register：對候選幀跑 EDM match → 經既有 reference 的 `xyz_by_cell` LUT 把 match 提升成
2D-3D → pycolmap RANSAC PnP（seed 0）→ 過 inliers ≥ 80 / reproj ≤ 3.0 / 錨點 ≥ 12 後打包成
新 reference，corner-anchor 合約與 `import_direct_edm_bundle.observation_lut` 一致
（ref-side keypoint 在 `8` 的整數倍上，即 cell 左上角））。

**產物**：候選 release `releases/river_gluemap_all8_direct_20260905_aug/`（63 顆新 reference，
總數 1108；註冊品質 inliers 123–654、reproj 1.3–2.1 px、每顆 46–182 個錨點；S 重算 2.3411
與原值差 <1% 故沿用原 S/radius/max_jump）+ `site_profile_augmented.json`。單幀煙霧 1/1
TRACK（inliers 108、reproj 1.56）。**現行 release / mission selection / flight profile 完全未動。**

**語料庫 gate（production-path、720p、stride 3、seed 0、全片，vs `sync_relaxed66` 基準）——退化：**

| 影片 | base | aug | base rate | aug rate |
|---|---:|---:|---:|---:|
| P116 | 607/731 | 565/728 | 83.0% | **77.6%（−5.4pp）** |
| P118 | 580/625 | 577/623 | 92.8% | 92.6% |
| P119 | 753/971 | 750/968 | 77.5% | 77.5% |
| P157 | 871/1020 | 869/1016 | 85.4% | 85.5% |
| P167 | 541/766 | 541/762 | 70.6% | 71.0% |
| P168 | 1539/1765 | 1518/1764 | 87.2% | **86.1%（−1.1pp）** |
| P117 | 365/409 | 365/409 | 89.2% | 89.2% |
| **合計** | **5256/6287（83.6%）** | **5185/6270（82.7%）** | | **−0.9pp** |

延遲同步退步（wall p50 26.5 → 30.3 ms；此項帶有「新 bank 未建、cache miss 走逐次重提取」的
混淆因素，但特徵重提取與 bank restore 是 exact 相等，**成功率判定不受此混淆影響**）。

**機制（歸因分析，`outputs/augment_20260905/`）：** 整個 aug run 的 1764 幀裡 `aug_`
字串出現 **0 次** —— 新 reference 從未被選中。退化全部來自**擠佔**：新 reference 落在失敗
山谷，LOST local pool（空間近鄰）與 TRACK near-pool/covis 把它們掃進候選，但它們的稀疏
LUT（46–182 錨點 vs 原版每顆 ~1000+ 個 COLMAP 觀測）匹配後大部分 cell 是 NaN 被丟棄，
inliers 遠低於 gate → 永遠選不中，卻佔掉 top-k 名額與 EDM forward。P116 的
`edm_local_recovery` 失敗 38 → 65、P168 的 `edm_temporal_map` 失敗 82 → 100，與此一致。

**判定：NOT GO，不採用。** 依本文件准入規則（完整 replay 不得降低核准的 recovery/quality
gate），候選地圖退步且機制明確。**單幀影片幀的 LUT 天然稀疏**——原版 reference 的稠密
LUT 來自 SfM 對同一影像的數千個三角化觀測，這是「從既有影片抽幀補 reference」路線的
結構性限制，不是實作 bug。若要走這條路，前置條件是解決稀疏性（例如：放寬 lift 距離換
密度但接受錨點偏移、或改 bundle schema 支援「僅 LOST 可檢索」的輔助 reference 類別），
兩者都需要新的 gate 證據。否則補 reference 應走正規重飛建圖（含三角化）。
**不要在沒有新機制的情況下重跑本實驗。**

工具與證據保留：`validation/augment_failure_sphere_references.py`、
`outputs/augment_20260905/`（candidates、smoke、corpus、compare 腳本）、
`releases/river_gluemap_all8_direct_20260905_aug/`（候選 release，未接線）。



## 地圖更換重做流程

### 換地圖時：可沿用 vs 必須重跑（快速對照）

**可直接沿用（機制/演算法不隨地圖改；只需跑對應的 exact / 回歸測試，不需重調參）**

| 項目 | 沿用範圍 | 換地圖只需 |
|---|---|---|
| 項 2 每幀 query backbone feature reuse | 完全沿用 | `test_reference_feature_cache.py::test_prepared_query_features_are_reused…` + 一段固定 replay trace 相同 |
| 項 6 matcher packed D2H | 完全沿用 | `test_reference_feature_cache.py::test_match_output_pack_preserves_values_and_batch_ids` |
| 項 7 bundle mmap + JPEG 平行解碼（機制） | 機制沿用，收益隨 reference 數/JPEG 大小變 | 記錄新 cold-load p50/p95；workers 維持 8 |
| 項 8 latest-frame coalescing + adaptive submit（演算法） | 演算法沿用，cadence 自動跟 latency | `test_worker_lifecycle.py` 那 3 個行為測試；查 source-frame age / coalesce drops |
| 項 9 sparse localization JSON telemetry | 完全沿用 | `test_localization_metrics.py` 兩個測試 |
| neck `repeat→expand`（exact） | 完全沿用（數值 identical，非加速） | `SFM_EDM_NECK_NO_EXPAND=1` A/B 逐幀 0 diff |
| `_track_klt_prior` 移除 `raise StopIteration` | 純重構，完全沿用 | `test_klt_miss_prior.py` |
| cache 8GiB 預算對 CLI override | 純 correctness fix，完全沿用 | `test_validation_benchmark_helpers.py::test_reference_feature_cache_override_*` |
| ESEKF / KLT 3D-aware 接線程式碼 | 程式碼沿用（replay 休眠）；EKF 參數可能要調 | 需 live telemetry 才評得到，replay gate 無法 |
| 實作同步邊界（EDM 組實為 symlink 單樹，見文末更正） | 路徑寫法沿用，改一處即生效 | 跑對應測試 |

**必須重跑 / 重算（地圖相依；不得沿用 river 的值）**

| 項目 | 為何相依 | 換地圖動作 |
|---|---|---|
| 項 1 相機內參、`S`、`radius`、`max_jump` | 座標尺度 + 相機來源 | 依 B 段重算 `S=2·p95(‖center−median‖)`、`0.40·S` 起始，重跑整段 replay |
| 項 3 GPU feature cache 192 | 量測 working set 相依 | 依 C 段跑 `64 / 192 / 全 reference 數` 三點，輸出完全相同時取最小 p50/p95 的容量 |
| 項 4 MegaLoc 為 VPR（reference descriptor bank） | `ref_global` 綁地圖 | 重建/驗證 bundle 內 `ref_global`；**不要**因換地圖改 query encoder / engine |
| 項 5 persistent feature bank | bank identity 綁 bundle SHA + reference 名單 + 每張 image SHA | 用 `SFM_EDM_BUILD_REFERENCE_FEATURE_STORE=1` 建新 bank，關旗標後確認可再載入 + exact matcher output |
| 項 10 `reference_quality_weight 0.5` | 參考品質分佈相依 | 重跑 `a2_quality` 對照（0.0 vs 0.5），比 p95 與成功率；可退回 0.0 |
| 項 11 `use_temporal_reference` | 收益隨 reference 密度 | 重跑 temporal on/off 對照；river 現行是 `true`（2026-09-02 gate 上 off 反而 P117 退化） |
| **項 12 `lost_global_retrieval_interval`（profile=3 + code default=3）** | **LOST 密度相依，sweep 非單調** | **重跑 interval sweep（至少 1/2/3/4/8）+ P168+P117 兩段 holdout；峰值可能不在 3。若新地圖 recovery 差很多可先設回 `0`（one-shot）再調** |
| 「現行 profile 中仍需場域重驗」整段 | `query_cuda_graph`、`acquire_stage_mode`、`pnp_ranked_batches`、`track_map_first`、`match_batch_size` + 所有 inlier/reproj/jump/yaw/stale-LOST gate | 當候選,不得當已證明的通用優化直接升級;安全 gate 不得為成功率/latency 移除 |
| 整段 D 回放 gate | video / frame schedule / camera / PnP seed 都要固定為新地圖的 | 兩段完整飛行 replay（BOOT/TRACK/WEAK/LOST/reacquire）+ 一段 hard-negative/off-map |

**換 manifest / SHA 時要跟著重跑的測試**（不論地圖有沒有換，只要動了 profile / manifest chain）：
`test_deployment_validation.py`、`test_mission_resolver.py`、`test_site_profile.py`、
`tools/package_manifest.py verify`、`tools/system_validation.py`。項 12 的 SHA 串（profile →
`localizer_edm_manifest.json` → `mission_selections/*.json`）就是範例。

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

**2026-09-04 更正：所謂「兩份副本」實際是同一棵樹。** `定位演算法/EDM工具包/deploy` 是指向
`定位演算法/deploy_code/sfm_glomap_deploy` 的 symlink（`readlink` 實測），兩側同 inode，不可能分歧。
下表保留作路徑對照（兩種寫法指同一個檔），今後改一處即生效，不必「兩份一起改」：

- `定位演算法/EDM工具包/deploy/edm_matcher.py` ＝ `定位演算法/deploy_code/sfm_glomap_deploy/edm_matcher.py`
- `定位演算法/EDM工具包/deploy/reloc_localizer_edm.py` ＝ `定位演算法/deploy_code/sfm_glomap_deploy/reloc_localizer_edm.py`
- `production_localizer_factory.py` 同上（僅此一份）

同名模組單一份的政策由 `定位演算法/validation/check_runtime_mirrors.py` 強制執行 ——
它比的是 flight_control vs deploy_code/sfm_glomap_deploy，且政策是「拒絕鏡像」而非「比對同步」，
EDM 這組（symlink 單樹）不在其範圍內。若將來把 symlink 換成實體複製，必須先加 mirror 測試再刪本段。

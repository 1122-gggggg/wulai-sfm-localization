# `sfm_direct_deploy` — `direct` 定位後端的部署 package

`direct` 是目前唯一註冊的正式定位後端，由
`localizer_registry.py` 依 `site_profile.localizer` 建立。它跑的是 two-rate 迴路：

```text
每一幀（CPU）                          背景（GPU，不在關鍵路徑）
  KLT 追蹤 960×540 灰階                  MegaLoc top-2 檢索，失敗才有界擴充
  → pycolmap absolute-pose PnP           → EDM 密集匹配 → lift 2D–3D
  → FAST_TRACK / VO_ONLY / DEAD_RECKON   → PnP reloc → handover **replace** 活點集
```

快迴路完全不碰 GPU，reloc worker 完全不碰關鍵路徑。這個分工是刻意的，不是實作
巧合：改動它就等於改動 P174 那組凍結組態的前提。`FAST_TRACK` 是快迴路標籤，
不代表已達 strong 地圖支撐；map inliers 少於 80 時，即使仍顯示 `FAST_TRACK`，
也會排程 eager 非同步 recovery，不讓大量 VO 點掩蓋地圖錨點流失。
partial handover 經 KLT 走到目前影格後，若混合 map＋VO PnP 未達 strong map
支撐、仍有至少 80 個 map anchors，會另試目前影格的 native map-only PnP，
不是直接採用背景舊影格姿態。接受仍須通過既有六項凍結 strong 幾何門檻：
inlier count、inlier ratio、convex-hull coverage、4×4 occupancy、positive-depth
ratio 與 reprojection P90；來源 identity／epoch／freshness 與 adapter 的 KLT
confirmation 門檻也不變。離線 native pycolmap smoke 使用 91 筆錄製對應點，
得到 91 inliers；79 點則拒絕。這只證明該 correspondence surface 的恢復路徑，
沒有實際 RGB 重播，不能宣稱已治好 80 點 retrieval abstention。

### 返程失敗時的有界檢索（2026-10-03）

`optimizations` 未設定時，預設啟用 `adaptive_retrieval`；明確設定為 `false`
仍只使用原本的 top-2。前兩張得到強定位且有足量稀疏 seed 時不增加 EDM 比對；
否則，若開始擴充前仍有 `reloc.period_s` 預算，從 CPU 排序的最多 24 張候選中
再選兩張，本次最多比對四張參考。這是擴充前的預算檢查，不是整次求解的硬截止。
後續失敗輪替尚未試過的候選，耗盡才重試；可用強地圖錨成功則清除輪替紀錄。
共視排序仍保留最高排名的未嘗試全域候選，避免一直困在相同地圖鄰域。
`RelocFix.reference_names` 記錄本次實際比對的所有參考。GPU feature cache、
spatial selection、資產 SHA 與所有 strong／handover 門檻均不變。

離線以驗證過的河濱地圖影像與正式 BoQ／EDM／native PnP 注入錯誤檢索排序：
前三輪的 top-2 均被拒絕，第三輪擴充找到原排名第七、八張的正確參考，
得到 `LOCALIZED_STRONG`、2947 inliers 與 1142 seed points，每輪最多四張。
此驗證只注入檢索排序，沒有替換 matcher、PnP 或幾何門檻；不是最新實飛 RGB
重播，也不代表返程定位斷線已經過真機驗收。

同一 capture 內，每張參考的二視圖幾何與 sparse／depth lifting 結果只計算一次；
先重用獨立參考證據，再按原規則全域 greedy dedup，避免擴充 top-2+2 重算前兩張。
快取在下次 capture 及 `close()` 清除，不跨影格重用。Vendor 只抽出等價的 private
lift hook，快取與選擇政策仍位於本地 `live_provider.py`。

最新 `075845Z` 場次的 34 張實拍，前後判定／姿態／seed 逐位元一致；失敗求解
配對 median 約 407 → 383 ms。無足量 seed 的 strong 不再提前停止擴充，找不到
更好的可用錨則保留原 strong 結果；該種例外的單次耗時 322 → 615 ms，不能
宣稱每次都加快。返程仍有 31/32 拒絕及一次無 seed 的 strong，不是失鎖已修好。
兩張失敗影格另測排名 25–64 仍未取得可用錨，故不擴大正式 GPU 搜尋或降低
幾何門檻；附近同朝向參考存在，不能把這個有界審查解讀成缺圖或全圖無解。

## 檔案分工

| 檔案 | 責任 |
|---|---|
| `direct_paths.py` | `vendor_sys_path()`：把 `vendor/` 冪等地掛進 `sys.path` |
| `direct_map.py` | `DirectMapAssets.load()`：讀 `direct_bundle.json`、驗 SHA-256、解析 model / keyframes / MegaLoc bank / intersection cells / depth 路徑 |
| `direct_profile.py` | `load_direct_profile()`：讀 `direct_localizer_profile.json`，回傳巢狀 frozen dataclass |
| `live_provider.py` | `LiveMapEDMProvider`：GPU 側 MegaLoc + EDM + PnP，回傳 `RelocFix` |
| `two_rate_tracker.py` | `RelocWorker`（單一背景 thread）與 `TwoRateTracker`（快迴路 + handover + VO + dead reckon） |
| `direct_localizer_adapter.py` | `DirectTrackerAdapter`：每幀只跑一個追蹤器（baseline 階段為 `TwoRateTracker`，工作圖階段為 `OnlineWorkMapTracker`），套用位姿確認後發布 |
| `surface_depth.py` | 選配 Open3D CPU 表面射線查詢；驗證 MVS sidecar 與地圖身分，輸出公尺 camera-Z 深度，未知為 NaN；不接入 tracker 或 flight controls |
| `vendor/` | vendored 上游 package，見下 |
| `VENDOR_PROVENANCE.json` | vendor 來源、逐檔 SHA-256、本地 patch 清單 |

資產契約：`direct_bundle.json`（schema `direct-localization-bundle/v1`）與
`direct_localizer_profile.json`（schema `direct-deployment-profile/v1`）。欄位表見
[`控制介面程式/site_profiles/建圖端輸出規格.md`](../../../控制介面程式/site_profiles/建圖端輸出規格.md)。
建圖流程見 [`定位演算法/EDM工具包/build/README.md`](../../EDM工具包/build/README.md)。

`model_sha256` 必須包含 `cameras.bin`、`images.bin`、`points3D.bin`；
新 COLMAP 模型可另宣告 `rigs.bin`、`frames.bin`。所有宣告的模型檔都驗證存在與
SHA-256，不接受其他模型鍵。這讓福和橋交付包可直接載入，無須刪除 rig/frame
資產或重寫原始 bundle 的 hash。

**生產路徑不讀任何環境變數旋鈕。** P174 那組 env（`P174_AND_NEXT.md:58-65`）已經
逐項寫進 `direct_localizer_profile.json`，並由 `site_profile.asset_sha256` 綁定。
要換數值就是換一份 profile 檔 + 換一次 SHA，不是設環境變數。

## 選配：凍結舊圖，只擴張新區

`online_workmap.expansion` 是 SHA 綁定的實驗開關；省略時三項皆關閉。
區塊存在時必須完整提供三個布林值，任一啟用都要求 `online_workmap.enabled=true`：

```json
{
  "online_workmap": {
    "enabled": true,
    "optimizer": "local_ba",
    "retain_valid_tracks": true,
    "expansion": {
      "frozen_region_gate": true,
      "spatiotemporal_query": true,
      "depth_seeds": true
    }
  }
}
```

`online_workmap.submap_graph_solver`（`batch` 預設 | `isam2`）選擇子圖 Sim(3) 圖的求解器；
`isam2` 需要選配 GTSAM（`requirements/optional-workmap-graph-lock.txt`），缺少時圖層
回報 `gtsam_unavailable` 並停用，前端與併圖不受影響（改用錨點擬合的 Sim(3)）。
release builder 以 `--workmap-submap-graph-solver {batch,isam2}` 寫入（需 `--online-workmap`）。

- `frozen_region_gate`：0.08 地圖單位 cell 中至少 3 個舊圖點才算覆蓋，連同
  26 鄰 cell 禁止永久新地標升格，不阻斷暫時 VO 點。三角化與深度提升共用
  此升格閘門；非有限座標 fail closed。
- `spatiotemporal_query`：最近 100 個 reference-KF 記錄與可見 3D cell 提供
  負 ID 候選。投影只初始化 KLT，PnP 使用來源關鍵幀到當前幀的前後向 LK 實測像素，
  不會把預測位置當觀測，也不重新開啟 `TOPUP`。
- `depth_seeds`：背景 relocalizer worker 串行執行 MoGe-2 ViT-S，與索引及 overlap
  共用公平排程。pending depth 只留最新請求，快迴路不等 GPU。
  深度先以舊圖正 ID 擬合尺度，不足 20 點才用 session 支援；通過尺度一致性、
  第二幀重投影及凍結區域檢查後才提升為負 ID 地標。
  snapshot 的 `origin` 區分 `depth` 與 `triangulated`。

MoGe 選配安裝見 [`requirements/README.md`](../../../requirements/README.md)。
預設使用 SHA-256 固定的本機 `moge-2-vits-normal/model.pt`；
`SFM_MOGE_CHECKPOINT` 只覆寫檔案位置，不放寬 hash。缺少選配層或權重時
`OnlineWorkMapTracker` 建構失敗，adapter 在 worker 啟動階段拋出
`RuntimeError("online workmap tracker construction failed")`，不退回其他追蹤器。
release builder 的 `--online-workmap --workmap-expansion` 同時開啟三項。
診斷以 `workmap_query_*`、`workmap_depth_*`、`workmap_frozen_*` 傳到 adapter，
並由 `localization_metrics.DIRECT_INFO_FIELDS` 白名單保留至 worker IPC 與 session log。
正 ID、凍結參考影像與姿態不變；負 ID 定位仍是 workmap 弱定位。
此開關與開發影片回放不授予品質、mission 或飛行核准。

### 擴圖狀態、子圖與 Sim(3) 併圖

`workmap_state.py` 的 `MappingStateMachine` 每幀以 R_G（舊圖內點／所有有效追蹤對應 `n_corr`）、
hull 覆蓋（≥ 0.15，同 profile `minimum_hull_coverage`）與舊圖錨點連續性決定
`LOCALIZE`（強覆蓋，不建新點、不記 KF、live cap 回到舊圖優先）、`ANCHORED_EXPAND`、
`DETACHED_EXPAND`（連續 12 幀無舊圖錨點且仍有 VO 位姿 → `SubmapRegistry` 開 detached 子圖，
之後的新點／KF 帶該 `submap_id`）、`RELOCALIZE`、`MERGE`。
`workmap_submaps.py` 的 `SubmapRegistry` 記錄子圖狀態（`active`／`relocalizing`／`merging`／
`merged`／`unmerged`）、Sim(3) 與錨點。回到舊圖後，每個強舊圖幀把「舊圖 PnP 位姿」與
「只用該子圖點的 PnP 位姿」配成 `AnchorPair`（子圖點不進混合 PnP，KLT 保留列）；
至少 8 個、跨度 8–48 幀且逐幀（中心 0.03 u、旋轉 2°）與相鄰對（1°、0.02 u）一致、尺度在 0.5–2
才併圖，否則丟最舊 anchor 繼續收，首個 anchor 起 48 幀仍不一致則 `unmerged`。
併圖：merge 邊 → `workmap_submap_graph.solve_submap_graph`（SciPy batch 或 iSAM2）→ Sim(3)
當 `merge_snapshot` 初值 → 既有 stitch 求解與 `apply_loop_candidate` 檢查。GlueMap 點與參考影像
永不移動；重複 landmark 經 overlap 驗證後以 `fused_to` 指回 GlueMap 點。
`unmerged` 子圖的點離開 live 與查詢索引，也不被 workmap 重定位引用；離線升格
（`tools/workmap_reconstruction.py`）略過它們。session 的 `snapshot.json` 帶 `submaps` 記錄。

補充：
- 發布位姿：混合 PnP（舊圖＋VO／session 點，與原版同）為預設；只有「舊圖點在混合位姿下 4 px 內的比例
  < `OLD_SUPPORT_SHARE`（0.8）」，或有 detached 子圖／pre-anchor 子圖需要純舊圖位姿當 anchor 時，才多解一次
  frozen-only PnP 並比較（`workmap_pose_basis` 為 `mixed`／`old_only`）。避免在不需要 anchor 且舊圖支持
  足夠時重複求解；這不代表所有影片的定位精度已達標。
- pre-anchor（independent）子圖併入視同子圖併圖：收集 anchor 時狀態為 RELOCALIZE，轉入 session 時為 MERGE，
  完成時另發 `submap_merged`（`kind: independent`）並計入 `workmap_merge_committed`。
- `SubmapRegistry.awaiting_id` 在 RELOCALIZE 與 MERGE 都持有待併子圖。這些子圖點不進主 PnP，但保留
  有效 KLT 軌跡，直到縫合提交成功；拒絕後重試仍保持隔離，放棄後於下一幀清除。
- stitch 後 loop frame 檢查：p50 ≤ 1.5 px、inlier 比例 ≥ 0.8，且不得比縫合前差 0.5 px。
  相對改善不能繞過絕對品質門檻；縫合被拒絕後清空 anchor 並重新計 48 幀視窗，最多 3 次。
- 縫合事件 `submap_merged`／`submap_merge_rejected` 帶 `stitch_s`；`submap_anchor_rejected` 帶擬合 `scale`、
  `rotation_deg`；每幀輸出 `workmap_anchor_rows`／`workmap_anchor_local_inliers`。
- stitch 預算 `_MAX_NFEV = 12`、`_MAX_ROBUST_NFEV = 3`（記錄快照上 cost 與 loop 殘差不變，求解時間約少 28%）。
- stitch 使用投影與 Rodrigues 旋轉的解析稀疏 Jacobian，避免有限差分重複計算整個觀測圖；
  凍結點權重、Huber loss、迭代預算及提交品質門檻不變。
- Local BA 提交成功後，待成熟 VO 特徵的種子位姿同步到同一 capture 關鍵幀的已提交位姿，
  避免以舊基線三角化、放大深度及尺度誤差；被拒或過期的 BA 不修改種子位姿。
- full 的 VO 關鍵幀只標記 BA 待處理，在下一個非關鍵幀開始組裝 snapshot；BA／stitch
  未完成時仍維持 single-flight，完成後等非關鍵幀再處理 pending。`keyframe_stride=1`
  時於下一 capture 處理，避免永久飢餓。epoch／revision、子圖隔離與提交品質檢查不變。
- iSAM2 與 batch 共用逐分量 Huber loss。每批新邊只提交一次，再以 `1e-5` 分量步長門檻做最多 8 次
  增量收斂更新；沒有新邊不重新更新。固定兩次 update 在大幅 merge 校正時可能尚未收斂。
- session 收尾快照以緊湊 JSON 寫入，只轉換 NumPy／Path 葉節點；保留原子替換、fsync 與 NaN 拒絕。
  不再為縮排與遞迴複製整個 JSON-native 快照付出額外成本；完整吞吐仍含 shutdown drain，不以穩態 FPS 代替。
- 正常 EOF 不再補跑背景幾何：取消未開始的 index／overlap／depth，等待已執行 GPU job，
  終止 CPU pool 並丟棄尚未套用的 BA／stitch／overlap fit／pre-anchor alignment。
  writer 保存串流期間已提交狀態；snapshot 的 `statistics.close_dropped` 記錄丟棄數，
  `close_phase_ms` 分別量測 stop／gpu_wait／cpu_terminate／writer。訊號 fast-close 路徑不變。
- 正常 close 在已接受寫入與最後觀測完成後，以 queue sentinel 喚醒 writer，不等
  0.1 s idle 輪詢；fast-close／未完成 drain 仍走既有停止事件。snapshot 的 keyframe
  catalog 不預先遞迴複製，由既有 JSON encoder 在 NumPy 葉節點轉換，內容與嚴格有限值契約不變。
- `close_phase_ms` 另含 archive／index_flush／reloc_close（例外階段為 `null`）；`SessionWorkMap.close` 在 snapshot
  `statistics.close_session_ms` 記 drain／writer_join／index_join／events_flush，`session_meta.json` 記
  `snapshot_write_ms`／`snapshot_bytes`。`ba_geometry_rejected` 事件帶 `submit_to_apply_ms`。
- VO 關鍵幀的角點偵測（含 occupancy mask）改在 `vo-detect` 執行緒執行，下一幀 `step()` 開頭併入
  `_vo_xy`／`_vo_seed_xy`／`_vo_bid`／`_vo_pose_by_bid`／`_vo_seed_meta`；發布位姿不變，偵測當幀的 `vo_candidates`
  不含尚未併入的種子。`reset()`／`close()` 等待並丟棄未併入的偵測。僅 `OnlineWorkMapTracker` 在無 depth estimator
  時啟用；baseline（`TwoRateTracker`）仍同步偵測。
- Local BA 的窗外觀測者：最多 `BA_MAX_OBSERVER_KEYFRAMES = 20` 個與窗內可變點共享觀測、同 epoch／同子圖的關鍵幀，
  以固定相機（只帶可變點的列）加入代價，`poses` 仍只含視窗幀；可變點上限 `MAX_VARIABLE_POINTS = 160`，
  單一關鍵幀新點觀測上限 500（等於 `fast_loop.track_cap`）。提交時觀測者的 `pose_revision` 須不變，且受影響列
  落在 `OBSERVER_SUPPORT_PX = 4.0` 內的數量不得減少（否則 `observer_support_count`）；未列入 payload 的窗外幀仍走
  逐觀測 `later_keyframe_reprojection` 規則。


### 暫時 VO、永久升格與單一前端

啟用工作圖時，`OnlineWorkMapTracker`（`TwoRateTracker` 子類）是唯一的前端：
每幀只有一份 KLT 活點、VO seed、handover、位姿、速度及確認歷史，也只有
一個 `RelocWorker`（先 global、失敗才查 workmap）。工作圖追蹤只多做
新點升格、KF 記錄與背景 BA；舊圖定位判定沿用原版規則。弱段的 VO_ONLY／
弱 FAST_TRACK 位姿就是用到 session landmark 的混合 PnP，單幀跳變仍由
`ORIGINAL_NO_FREEZE` 規則把關。

VO 特徵照原 keyframe 節奏播種，通過雙視角正深度、視差及重投影檢查就可成為
暫時負 ID 活點。永久升格另外要求 seed 與當前影格都通過可信位姿檢查、
capture／epoch 來源有效，以及工作圖區域閘門。未升格負 ID 仍可供短期 PnP，
但不能進 durable KF、重訪索引、depth 尺度參考或 BA，亦不計為工作圖地標使用。

full 的健康舊圖追蹤收到至少 `fast_loop.reseed_min_points` 個 walked 點時，
先解 frozen-only strong PnP；與前一幀中心差 ≤ 0.1 u、旋轉差 ≤ 5° 才以 refresh
替換正 ID 並保留有效負 ID，不發布 `RELOC_SEED`。不一致或失鎖時仍走原 reseed
與 adapter 確認流程。baseline 不啟用此分支。
可信 capture approval 只在啟用 VO 且符合 `vo.keyframe_stride` 的關鍵幀計算；
其他幀清空 approval，但不影響逐幀視覺 PnP。

建圖側（writer／index／BA／overlap／depth）拋例外或回報不健康時，
`OnlineWorkMapTracker.disable_mapping()` 停止所有建圖副作用（不再升格、
不記 KF、不送 BA／overlap、停用 presubmap），前端繼續定位，
`workmap_mapping_healthy` 變 `False` 並帶 `workmap_mapping_error`；不清空原點集
來「回退」。前端自己的例外仍往上拋。

IPC 診斷 `workmap_output_source`（發布位姿實際用到新點為 `workmap`，
否則 `baseline`）、`workmap_pose_landmarks`（本幀 PnP 內點中本 ordinal 的 session
landmark 數）、`workmap_mapping_healthy`、`workmap_mapping_error`。評估器只在
發布來源為 workmap 時，將新點／BA 使用事件計入發布成效。

### 原版／擴圖版成對回放

[`../../validation/workmap_paired_compare.py`](../../validation/workmap_paired_compare.py)
以三次 baseline／full 交錯順序執行 protocol，支援完整 cell 續跑，拒絕部分目錄、
來源雜湊變更與低於 3 GiB 的可用空間。每次保存 metrics，僅清除該 session 的
keyframe PNG；`report` 驗證所有 run 並輸出 `comparison.json`、逐 run markdown 表。
`winner` 要求每片 fresh 中位數不減、strong ≥ baseline 中位數的 97%，每次 full
FPS ≥ 來源的 98%、core95 ≤ `max(2×baseline 中位數, baseline 中位數+8 ms)`，
latency95 ≤ 100 ms、latency99 ≤ 200 ms。這些比較門檻不構成飛行核准。

即時 GPU query 是 latest-capture mailbox，wall-clock 回放的 fix 選取／交付時序不是
決定論。原版一致性以相同 seed capture 與 delivery ordinal 的區段逐幀比較完整 pose；
不以跨時序的同 repeat 位置差宣稱程式變更，也不剔除失鎖 baseline 或重跑不利 cell。
成對比較的三次中位數與逐 run 效能門檻保持原樣。

```bash
PYTHONPATH=outputs/deps/workmap_graph /home/allen/localization/.venv/bin/python \
  定位演算法/validation/workmap_paired_compare.py run --protocol <protocol.json>
PYTHONPATH=outputs/deps/workmap_graph /home/allen/localization/.venv/bin/python \
  定位演算法/validation/workmap_paired_compare.py report --protocol <protocol.json>
```

## vendor 來源與搬遷規則

`vendor/river_map_quality`（36 檔）與 `vendor/sfm_diagnosis`（94 檔）是從交付包
`river-deploy-5060-20260908/software/` 複製進來的，來源路徑與逐檔 SHA-256 記在
`VENDOR_PROVENANCE.json`：

- `river_map_quality` ← `software/compatibility/river_map_quality/src/river_map_quality`
  （`VENDOR_PROVENANCE.json:59`）
- `sfm_diagnosis` ← `software/normal_map_pipeline/src/sfm_diagnosis`
  （`VENDOR_PROVENANCE.json:159`）

本地 patch 有三筆，逐筆記在 `VENDOR_PROVENANCE.json.local_patches`（每筆帶
`upstream_sha256`；`packages.*.files` 保留的一律是上游 digest）：

1. py3.10 相容性：`river_map_quality/baseline.py` 改成 `from datetime import timezone`
   並 `UTC = timezone.utc`，因為 `datetime.UTC` 是 3.11+ 才有，而本工作區 runtime 是
   CPython 3.10.12。
2. 效能：`river_map_quality/official_edm_adapter.py` 的 `lift_reference_matches` 改成
   向量化鄰域搜尋。半徑、admission 規則與 (distance, Point3D ID) 併發序全部不變，輸出
   逐位元相同；`tests/localization/deploy/test_lift_reference_matches.py` 保留改版前的
   純量實作當 oracle 做差分比對。這筆不屬於下面規則 3 的兩類，是操作員明確核准後落地
   的，仍欠一次回上游。
3. 等價 hook：`sfm_diagnosis/site_pipeline/deployment_localizer.py` 抽出
   `_lift_matched_reference`，讓本地 provider 快取單張參考的 pre-dedup lift 證據。
   confidence、二視圖門檻、depth lifting 與合併順序不變；
   `tests/localization/deploy/test_direct_lift_cache.py` 使用抽取前實作作差分 oracle，
   涵蓋重疊邊界、tie、unmapped／depth，以及快取重用後再次合併不污染證據。

搬遷回上游的規則：

1. 改 vendor 檔之前先確認那不是應該改在 `sfm_direct_deploy/*.py`（本地 runtime）
   的東西。vendor 是第三方邊界，本地邏輯不往裡面塞。
2. 真的要改 vendor，必須在 `local_patches` 加一筆（`change` + `file` + `reason`
   + `upstream_sha256` + `patched_sha256`）。`packages.*.files` 記的一律是**上游**
   digest，不隨本地 patch 改動 —— 那是 re-vendor 時比對用的基準；本地檔案的實際
   digest 由 `patched_sha256` 與 `SHA256SUMS` 各記一份。無 provenance 的改動視為
   不可追溯，等同污染第三方邊界。
3. 本地改動不限類別。py3.10 相容性、上游 backport，以及**任何能證明有收益的優化**
   都可以留在本地。條件是規則 2 的可追溯性，加上 `tests/` 裡一份能證明等價或證明
   收益的測試 —— 對不該改變輸出的優化，測試要用改版前的實作當 oracle 做差分比對。
   本地 patch 仍然欠上游一次回收：規則 4 的 re-vendor 是整包換，沒有回上游的 patch
   會被蓋掉。
4. 重新 vendor 時整包換，不做逐檔挑選，並重算 `VENDOR_PROVENANCE.json`。

## 不許做的事

以下每一條都是 `P174_AND_NEXT.md:91-98` 的明文禁令，來源是已花掉的實驗成本，
不是保守偏好：

| 禁令 | 出處 | 理由 |
|---|---|---|
| 不許再拿 P167 / P173 / P174 回頭調 `TRACK_CAP`、VO lag、PnP、`VO_MIN_LIVE`、`DEAD_RECKON` 門檻 | `P174_AND_NEXT.md:93` | 旋鈕本來就是在 P167 上選的；再用同三條路調就是對測試集過擬合。準度只能靠**一條全新航線、凍結組態只跑一次** |
| LocoTrack-S/B 不進 29 fps 快迴路 | `P174_AND_NEXT.md:94` | 73 ms GPU vs KLT 2 ms CPU；small 只縮 transformer，CNN 共用，而且會跟 EDM 搶 GPU |
| 不做光度前處理（gamma / CLAHE） | `P174_AND_NEXT.md:95` | 部署會遇到所有光線；已量過，P174（mean luma 52）不做也有 99.0% 覆蓋（`FINDINGS.md:91`） |
| `VO_WINDOW_BA` 不開 | `P174_AND_NEXT.md:96` | 更慢（loop 5.48 → 6.47 ms）而黑洞末段誤差沒比較好（0.009 → 0.013）（`FINDINGS.md:96`） |
| 不自寫 CUDA / sm_120 JIT 擴充 | `P174_AND_NEXT.md:97` | SplatHLoc 已因此停掉 |
| `map_model` 不許指回 GlueMap 原圖 | `P174_AND_NEXT.md:98`、`river-deploy-5060-20260908/README.md:78` | GlueMap 觀測上的 baseline smoke lift 率只有 707/35478 = 1.99%，遠低於 15% 放行門檻（`.../model/occupancy.json:21-26`） |

搬機（5060）上**准做**的只有機器相關量測：實測快迴路與 reloc 毫秒、用實測
`reloc.median_ms` 設 `reloc.period_s`、handover 排程改用本機延遲、必要時降
`reference_cache_size`（`P174_AND_NEXT.md:71-89`）。這些都不是準度旋鈕。

## 已知數字（不要當作公尺誤差）

P174 是三條路線裡最接近「新路徑」的一條，但**沒有絕對 GT**：99.40% 是有 pose 的
幀比例，不是公尺精度（`P174_AND_NEXT.md:9,17`）。1829 幀中 `FAST_TRACK` 1507、
`RELOC_SEED` 81、`VO_ONLY` 230、`DEAD_RECKON` 0、`NO_POSE` 11（全在冷啟動）
（`records/replay_p174_ship.summary.json` 的 `status_counts`）。live 誤差未知；
最長一段 173 幀在 reloc 回來前已漂到 0.17 地圖單位（`P174_AND_NEXT.md:52-54`）。

因此新地圖進系統時 `localizer_quality_receipt` 一律先是未通過、`flight.approved`
為 false，要等一條全新航線的一次性現場驗收才能轉正。

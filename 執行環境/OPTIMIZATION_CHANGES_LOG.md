# 定位系統 完整修改紀錄（原始版 → 現在版）

烏來場域無人機定位/巡檢 runtime。本檔是這次調校的**完整變更記錄**:原始行為 → 現在行為、為什麼改、怎麼驗證。

> 說明:原始的自壓縮封存檔已在初期清理時刪除,git 基線 commit 是調校中途才建立的,所以本檔不是機械式 `git diff`,而是**逐項對照的策展記錄**,每項都已對照現行程式碼確認。GitHub 備份:私有 repo `github.com/1122-gggggg/wulai-sfm-localization`(僅程式碼,大檔 gitignore)。

> 2026-07-29 狀態：以下是 XFeat／光流時期的歷史調校紀錄，不再代表目前 EDM
> runtime 或操作介面。現行設定、安全契約與驗證結果以 workspace 根目錄
> `README.md` 為準；現行 UI 的「開始定位」只啟動定位，不代表 AUTO 同意或自主巡檢。

---

## 一、總覽對照表

| 面向 | 原始版 | 現在版 |
|---|---|---|
| **定位策略** | 每幀 MegaLoc→XFeat→LighterGlue→PnP | **光流追蹤 + 每 6 幀 deep refresh**(deep 只佔 14–17%) |
| **FPS(720p @ 5Mbps 真機畫質)** | **7.6** | **47–87**(視場景) |
| **精度** | baseline | deep 部分 **bit-identical**;flow 差 median **0.015–0.017 map-unit**(對飛控 1.5u/3.0u 門檻可忽略);100% 成功 |
| **飛行安全** | 部分且有 bug(起飛逾時留空中、無 NaN 閘、看門狗會誤觸發…) | 完整(獨立 SafetyMonitor、低信心→懸停→MegaLoc→切手動、連續性/跳點/NaN/OOM/串流閘) |
| **操作介面** | 基本顯示 | 開始巡檢閘、失敗指示、無法定位永久紅點、危險區、render 加速、worker 逾時重啟 |
| **結構** | 兩份「鏡像」實為不完整、靠 sys.path 意外決定、config 分裂 | pose_types 解耦、鏡像補完 + 同步檢查、依賴補齊 |

**現在的定位組合**
```
取得/迷路(BOOT_INIT/LOST): MegaLoc top-30 檢索 → XFeat → LighterGlue → PnP
正常追蹤(TRACK/WEAK):     XFeat mutual-NN 快路徑 → 不穩再 LighterGlue adaptive 3→5 → PnP
                          + 光流(KLT)追蹤,deep 只在每 6 幀 refresh 跑;加權 PnP + 信心/跳點閘
座標系: GLOMAP(x/z 水平、-y 上)
```

---

## 二、定位演算法與效能(`production_xfeat_tracker.py`, `reloc_localizer_xfeat.py`)

| # | 改動 | 原始 → 現在 | 無損? |
|---|---|---|---|
| 1 | **ref 特徵常駐 GPU 快取** | 每幀重傳 ref 特徵到 GPU → 一次上傳、快取重用 | ✅ bit-identical |
| 2 | **對應點組裝向量化** | Python 逐點 for-loop → numpy 向量化 | ✅ |
| 3 | **query 特徵抽一次共用** | 每 pass(nn/adaptive/full)重抽,一幀最多 3 次 → 每幀抽一次跨 pass 共用 | ✅ |
| 4 | **LighterGlue full retry 重用匹配** | full retry 重比前 3 ref → 重用 adaptive pass 的匹配 | ✅ |
| — | (sync-gate / normalized-desc cache) | 實測 **0 提升,已還原** | — |
| 5 | **光流追蹤模式**(`SFM_FLOW_TRACK=1`) | 每幀 deep → deep 每 6 幀 refresh + 中間 KLT 追蹤(forward-backward check) | 差 ~0.015u |
| 6 | **加權 PnP** | flow PnP 用全部追蹤點 → 排除**真正 FB 離群漂移點**於 solve 之外(絕對門檻保護,乾淨畫面**零回歸**、不加第二次 solve → FPS 不降) | 乾淨 bit-noise |
| 7 | **flow 信心閘** | flow 一律發布 → reproj>4/inliers<50/ratio<0.25/追蹤點<100 → **不發布**(懸停 + 下幀 refresh) | 增穩健 |
| 8 | **flow max_jump 閘** | flow 無跳點檢查 → 跳離上次發布中心 > `max_jump` 不發布 | 增安全 |
| 9 | **時序快取重建短路** | 每強幀重建 anchor cache(GPU→CPU→GPU) → seed ref 未變時只刷新 age、不重建 | ✅ bit-identical |
| 10 | **ref cache LRU** | 滿 700 整包 `clear()`(長航線 latency spike) → 丟最舊一個 | ✅ |
| 11 | **`xfeat_topk_track` 1300→1700**(2026-07-07 benchmark 優化 pass) | TRACK 每幀抽 1300 個 XFeat 特徵 → 1700。P0710071 512 幀 720p 串流實測:成功 492→501(96.1%→97.9%)、median inliers 189→258、reproj RMS 2.934→2.900、LOST+WEAK 83→66、LG fallback 232→169、p50/p90/p95 延遲 −6.7%/−10.7%/−21%。機制:NN 快路徑通過率 269→335,省下 ~68ms 的 LighterGlue fallback 多於多抽特徵的 ~2ms。1500/1900/2048 皆較差(bracket 驗證)。 | 品質↑速度↑ |

**累積:7.6 → 47–87 fps**(deep 部分 7.6→13.9 為 bit-identical;flow 再 ×3.5)。

**研究結論(2025-2026 光流文獻,5 團隊 + 對抗式驗證)**:對「~150 稀疏點、causal、小位移」場景,**沒有更快又不掉精度的光流可換**;KLT 已在 Pareto 前緣。深度 tracker(CoTracker3/TAPIR)慢 5-50×;GPU-KLT 在 150 點 overhead-bound 無法加速;RLOF 只在 KLT 掉 track 時有價值(穩健度工具,非速度)。

---

## 三、飛行安全強化(`path_follow_flight.py`, `real_path_follow_controller.py`, `olympe_frame_source.py`, `mission_pipeline.py`)

| 項目 | 原始 → 現在 |
|---|---|
| **`airborne` UnboundLocalError** | 起飛逾時把飛機留空中且漏連線 → 修好,一律安全清理 |
| **SafetyMonitor(取代 PcmdWatchdog)** | 看門狗與主迴圈同執行緒(GPU 卡死就一起卡) → **獨立執行緒**,卡死時仍能 LAND/EMERGENCY/切手動;**MANUAL 完全不送 PCMD**(不搶飛手);看門狗只在主迴圈跳動後才啟用(不會在起飛窗誤觸發) |
| **NaN/inf 位姿閘** | 無 → 非有限值一律當無定位 → 懸停 |
| **get_pose 例外 + CUDA OOM** | 未包保護,例外會炸迴圈 → try/except → 懸停;OOM 清快取復原 |
| **低信心 → 懸停 + MegaLoc → 切手動** | 無;弱定位可能被當真飛 → inliers<`SFM_LOW_CONF_INLIERS`(60)/WEAK/無定位 → **懸停 + 逼 tracker 跑 MegaLoc 重定位**;`SFM_LOST_MANUAL_S`(3s)內救不回 → **SkyController 切手動**;純 WiFi 無飛手則保留 `LOST_LAND_S`(4s)自動降落 failsafe |
| **連續性閘(跳點)** | 無 → 跳離上一接受點 > `SFM_MAX_POSE_JUMP_U`(1.5u)視為斷裂不採用,連兩次一致才當重定位 |
| **重定位不污染 heading** | 重定位跳點被 `HeadingEstimator.update()` 當真實移動 → 污染 yaw offset、PCMD 朝錯方向 → `mark_teleport()` 跳過該幀位移更新(保留已學 offset) |
| **MANUAL 分支靜默** | 進 MANUAL 仍送一次 zero PCMD(搶飛手) → 完全靜默 |
| **安全檔啟動重設** | 只在檔案不存在時寫 auto(舊 run 殘留的 land/emergency 會被新任務第一輪讀到) → **啟動一律原子重設為 auto** |
| **body pitch 只看水平** | pitch 用 3D 速度 norm(純垂直修正產生假前傾) → 只看水平 X/Z,gaz 單獨管 Y |
| **起飛前載入模型** | 無 → `ensure_models` 先載入 |
| **看電桿** | 機身朝電桿飛(可能撞線) → 沿航線飛、gimbal 看電桿 |
| **mission_pipeline** | 預設 bundle 錯、安全指令非原子寫、python 寫死 → 預設 v3、原子寫、python fallback(env→存在才用→sys.executable) |
| **legacy 路徑** | 可直接武裝 → 用 `SFM_ALLOW_LEGACY_FLIGHT=1` 擋 + finally 清理強化;autoflight DONE 真飛也降落 |

安全驗證:`path_follow_flight.py --selftest` **9 項全過**(heading 融合、PCMD 正負號、串流丟失 + NaN 懸停閘、SafetyMonitor LAND/HOVER 權威、MANUAL 靜默、看門狗預跳動閘、低信心→懸停+MegaLoc+切手動)。

---

## 四、操作介面(`flight_operator_app.py`, `draw_path.py`, `live_localizer_worker.py`)

| 項目 | 原始 → 現在 |
|---|---|
| **開始巡檢閘** | 一開就跑 → 按「開始巡檢」才開始串流輸入 + 定位(= 進入 AUTO 的同意);顯示**整體 FPS** |
| **`--stream-fps`** | 無 → 可設串流速率(測試用 29) |
| **繪製路徑疊圖** | 無 → 畫的巡檢路徑用**洋紅色**疊在地圖(aligned→glomap),與藍色實際軌跡區分 |
| **地圖預設放大** | 1.85× / 滾輪上限 30× → **3.2× / 上限 120×** |
| **定位失敗/低信心指示** | 無 → 影片上方紅(失敗)/黃(低信心)橫幅 + telemetry 中文大字 + 軌跡上紅/黃健康標記 |
| **無法定位永久紅點** | 無 → 每次定位失敗在最後已知位置留**永久紅點**(`SFM_NO_LOC_DEDUP_U` 去重),累積成失敗地圖(取代先前試作的危險平面) |
| **Render 加速** | 每 tick 重繪兩板 + 逐點投影 + 7 萬次 draw.point → **dirty-flag**(沒變不重繪)+ 批次 `transform_xyz` + 點雲向量化 `fromarray` |
| **worker client 去重** | `LiveLocalizerClient`/`LiveDetectorClient` ~90 行重複 → 抽 `LiveWorkerClient` base |
| **worker IPC 逾時 + 重啟** | `readline()` 可永久阻塞、定位凍住 → `select` 逾時(`SFM_WORKER_TIMEOUT_S`,預設 8s)→ 標 FAIL 解凍 + cooldown 重啟(重載期間跳過 submit) |
| **boot lock 逾時** | 只在鎖定時釋放、第一幀鎖不上永久凍畫面 → 只在巡檢後啟動、`boot_lock_s` 到期未鎖就釋放 |
| **狀態 JSON 降頻** | 每幀寫 `/tmp` indent JSON → ~5Hz + compact |
| **worker fd-level stdout 轉向** | 只改 Python `sys.stdout`,原生 lib(如 LightGlue 載入訊息)fd-1 寫入會污染 JSON IPC → 比照 detector 做 fd 級轉向 |
| **draw_path 導覽** | → Shift+左鍵平移、Alt+左鍵旋轉 |

---

## 五、結構 / 設定 / 穩健性(4-方向 Audit)

| 項目 | 原始 → 現在 |
|---|---|
| **pose_types 解耦** | localizer `from autoflight import Pose,Localizer` → 連帶載入 plan_path(SDF 規劃) → 新增 `pose_types.py`,localizer 改 import 它;autoflight re-export 保相容。**驗證:import localizer 不再載入 plan_path/autoflight** |
| **鏡像補完 + 同步檢查** | `reloc_localizer_xfeat.py` 只在 deploy_code、flight_control 沒有(真飛靠 PYTHONPATH 混用兩份) → 複製補完 + `sync_mirror_check.sh`(8 檔 diff,drift 即報錯) |
| **`find_system_root`** | 5 份複製、硬編 `/media/cihcilab/新增磁碟區/...` fallback(換機靜默解析到外機路徑) → 改吃 `SFM_SYSTEM_ROOT`,否則明確報錯 |
| **env 變數統一** | UI `SFM_LOC_LOW_INLIERS` vs 飛控 `SFM_LOW_CONF_INLIERS` 兩名 → 統一 `SFM_LOW_CONF_INLIERS` |
| **依賴補齊** | `requirements_runtime.txt` 漏 `huggingface_hub`/`safetensors`/`kornia`(MegaLoc/LighterGlue 需要) → 補 pin |
| **eval 吞例外** | `validation/eval_stream_core.py` match/PnP 例外靜默 → 記錄 + error 幀分開計數 |
| **死碼標註**(依決定保留不刪) | safe_volume / pole_cruise / `_localize_with_temporal_cache` / `XFeatLightGlueLocalizer` → 加註「未接入生產路徑」 |
| **避障監控標註** | `SparseCloudCollisionMonitor` 未接入 → 註明「避障靠飛手,接入為待決設計」 |

---

## 六、Benchmark 結果(720p）

| 影片 / 條件 | 模式 | FPS | 成功率 | inliers | 精度(vs deep-every-frame) |
|---|---|---|---|---|---|
| P1210121 乾淨(2.7K→720p) | deep 原始 | 7.6 | 100% | 125 | baseline |
| P1210121 乾淨 | deep 優化(#1-4) | 13.9 | 100% | 125 | **bit-identical** |
| P1210121 **5Mbps 真機畫質** | deep | 13.7 | 100% | 126 | 0.015u |
| P1210121 **5Mbps** | flow | 53 | 100% | 121 | 0.017u |
| **P1230123 5Mbps** | **flow(最終)** | **~85** | **100%** | 255 | median 0.0076u / p90 0.021u |

**結論:5Mbps 真機串流畫質對定位幾乎無影響;光流方法在真機速度 + 真機畫質下驗證通過。** 使用者確認測試影片就是真機錄的、飛行速度也差不多,故此結果具真機代表性(唯一未模擬的是 WiFi 掉幀,已由 flow-refresh + 懸停處理)。

---

## 七、新增檔案
- `sfm_system/定位/deploy_code/sfm_glomap_deploy/pose_types.py`(+ flight_control 鏡像):中性 `Pose`/`Localizer` 型別。
- `sfm_system/定位/sync_mirror_check.sh`:兩份鏡像 drift 檢查。
- `.gitignore`:排除大檔(*.pt/*.bin/*.ply/torch_hub_cache/影片/__pycache__)。
- `OPTIMIZATION_CHANGES_LOG.md`(本檔)。
- `sfm_system/定位/validation/compare_benchmarks.py`(2026-07-07):兩份 benchmark JSON 差異 + speed/accuracy 接受準則判定(容忍度可調,`--json-out` 存判定)。
- benchmark/tracker 新增診斷欄位(2026-07-07,純記錄不影響行為):每幀 `xfeat_extract_count`/`lg_call_count`/`lg_reuse_count`/`nn_call_count`/`megaloc_call_count`、`pnp_failed`、`jump_rejected`、`inlier_coverage`(5×3 格覆蓋率)、`quality_score`(綜合品質分,僅記錄)、`load_ms`、p95 percentile、`intrinsics_check`(benchmark vs 生產 720p 內參一致性,焦距差 >2% 硬警告)。

---

## 八、可調環境變數(完整)

**定位/光流**:`SFM_FLOW_TRACK`(光流開關)、`SFM_FLOW_REFRESH`(refresh 間隔,預設 6)。
**安全門檻**:`SFM_LOW_CONF_INLIERS`(60)、`SFM_LOC_HIGH_REPROJ`(4.0)、`SFM_GATE_WEAK`、`SFM_MAX_POSE_JUMP_U`(1.5)、`SFM_MAX_ROUTE_DEVIATION_U`(3.0)、`SFM_LOST_MANUAL_S`(3)、`SFM_STREAM_LOST_LAND_S`(15)、`SFM_WEAK_HOVER_LAND_S`、`SFM_PCMD_WATCHDOG_S`、`SFM_SAFETY_FILE`、`SFM_ALLOW_LEGACY_FLIGHT`。
**UI**:`SFM_NO_LOC_DEDUP_U`(1.0)、`SFM_WORKER_TIMEOUT_S`(8)。
**路徑/資源**:`SFM_SYSTEM_ROOT`、`SFM_RELOC_BUNDLE`、`SFM_MEGALOC_CACHE`、`SFM_FLIGHT_PATH_JSON`、`SFM_POLES_JSON`、`SFM_MAP_ROOT`、`SFM_SAFEZONE_DIR`、`SFM_LOCALIZER_PYTHON`、`SFM_DETECTOR_PYTHON`、`SFM_OLYMPE_CONTROLLER`、`SFM_GLOMAP`、`SFM_TOOLS_ROOT`。

---

## 九、暫緩項目(有理由,未做)
- **起飛前重力對齊檢查**:起飛 hover 後拿 PnP 相機姿態比對 IMU 重力,地圖 -Y 與真實重力差 >2-3° 就禁 AUTO。**真機才用**,需一次真機 hover 校正 IMU→相機座標慣例,故未實作。
- **#10 frame.tobytes 複製**:實測 IPC ~55-83MB/s 遠低於 pipe 頻寬,非瓶頸。
- **#11 mission_pipeline python probe-import**:`SFM_LOCALIZER_PYTHON` env override 已是逃生口,大致涵蓋。

## 九之一、2026-07-07 benchmark 優化 pass:已測試但**否決**的候選(勿盲目重試)

基準與最終結果:`定位/outputs/optimization_baseline.json` / `optimization_final.json`(P0710071 512 幀,720p,RTX 5060 Laptop)。比較工具:`定位/validation/compare_benchmarks.py`(接受準則 + 可調容忍度)。品質指標跨 run **完全 deterministic**;延遲有 ~5% 熱漂移,故計時結論皆用前後夾測(bracketed A/B)。

| 候選 | 結果 | 否決原因 |
|---|---|---|
| **2D/3D 對應點去重**(`dedup_corr`,保留 flag 預設關) | 否決 | 對應點 median **46.8% 是重複**(cache+多 ref 同 landmark),但所有 inlier 門檻(fast-accept≥100、weak≥30/50/80、seed≥150)都是照「含重複計數」校準的;去重後 inlier 掉到門檻下 → 成功率 −4.5pp、LOST+WEAK +30。門檻屬安全參數不動,故整案否決。 |
| **temporal cache 改用 validated inliers seed**(`temporal_cache_seed_mode=inliers`,保留 flag 預設關;即舊「暫緩 #9」) | 否決 | LOST+WEAK −16 但成功 492→488(尾段難路段 −9/+5)。 |
| `nn_min_score` 0.85→0.80 / 0.82 | 否決 | 0.80:成功 −3;0.82:單獨用全過但延遲無改善,疊在 topk1700 上反而成功 501→491。 |
| `temporal_cache_min_score` 0.85→0.80 | 否決 | LOST+WEAK +9、成功 −1。 |
| `temporal_cache_max_age` 2→3(疊在 topk1700) | 否決 | 成功 501→486。 |
| `temporal_cache_seed_min_inliers` 150→100(疊在 topk1700) | 否決 | p50 最佳(19.8ms)、LOST+WEAK 最少(50)但成功 501→494。 |
| `xfeat_topk_track` 1500 / 1900 / 2048 | 否決 | 1700 為 bracket 驗證後的最佳點(1500:495;1900:495;2048:492)。 |
| 移除 runtime `_sync()` / 預先 normalize ref 描述子 | 未重試 | 前一輪已實測 0 提升並還原(見表二「—」列)。 |

---

## 十、換場域須知(演算法不變)
換另一個場域**只換資料檔 + 指路徑**,演算法完全不動:① reloc bundle `.pt`(`SFM_RELOC_BUNDLE`)、② 顯示點雲 `.ply`、③ 巡檢路徑 `flight_path.json`(`SFM_FLIGHT_PATH_JSON`)、④ 電桿 `poles.json`(`SFM_POLES_JSON`)、⑤ MegaLoc 快取 `.npy`(選用)。同一台 ANAFI 則相機內參 `CAM_720` 不動。①②需用**完整 repo 的建圖 pipeline**(GLOMAP + bundle builder)產生,不在本 runtime 套件內。

---

## 十一、2026-07-14 production 決策：**移除 mutual-NN 快路**，TRACK/WEAK 全程 LighterGlue

操作者決策（依據：NN 快路的準度明顯較差）。此項推翻 §九之一 之前把 `nn_then_lg`
當 production 預設的結論，並改以 20260714 參考軌跡的那條路徑（全量 LighterGlue）
本身作為 production。

**新的 production 定位組合**

| 階段 | 內容 |
|---|---|
| BOOT_INIT / LOST | MegaLoc top-30 → XFeat 2048 → LighterGlue → PnP（不變） |
| TRACK / WEAK_TRACK | XFeat top-K 1700 → **LighterGlue adaptive first top-3、不足再 top-5** → PnP，**每幀都跑** |
| mutual-NN 快路 | **移除**（`matcher_mode` 預設由 `nn_then_lg` 改為 `lighterglue`） |
| temporal anchor cache | **實質失效**：cache 只在 NN 快路分支被查詢（`production_xfeat_tracker.py` 的 `matcher_mode == "nn_then_lg"` 區塊），移除 NN 後 cache-used/accept 恆為 0。設定值保留只為讓舊 benchmark 可重現。 |

**改動檔案**（兩份鏡像已同步，`sync_mirror_check.sh` 通過）

- `production_xfeat_tracker.py`：`ProductionConfig.matcher_mode` 預設 `nn_then_lg` → `lighterglue`
- `path_follow_flight.py`：`production_config()` 同上
- `live_localizer_worker.py`：新增 `--matcher-mode`（空值＝跟隨 `production_config()`）
- `flight_operator_app.py`：新增 `--nn-fast-path`（**僅供 benchmark 重現**已移除的 NN 路徑）

**已知代價（來自 TRACK_REFERENCE_BENCHMARK_20260714 的既有量測）**

- 速度：容易影片變慢。P024 正常 TRACK FPS 37.07（NN）→ 25.41（全量 LG）；
  P072 32.45 → 24.40。困難影片幾乎不變甚至更快：P121 15.82 → 16.65，P017 19.17 → 20.70。
- 準度：全量 LighterGlue 就是該報告的參考軌跡本身，因此不再有「相對參考的偏差」。
  原 NN 路徑相對它的位置差 P95 為 P017 0.109 m、P024 0.173 m、P072 0.298 m、P121 0.130 m。

**仍未解**：本專案沒有非衛星絕對 ground truth，因此「移除 NN 提升絕對準度」只能以
「不再偏離全量 LG 參考」表述，不能宣稱公分級絕對精度改善。

---

## 十二、2026-08-03：EDM reference backbone 特徵快取（**已接受**）

EDM 用一次 backbone 呼叫處理 `cat([image0, image1])`（`edm.py:56`）。localizer 一律
把地圖參考當 image0、把同一張 query 複製成 image1（`match_many_to_one`），所以每幀都在
重算「只取決於不可變地圖影像」的參考特徵。eval() 下 BatchNorm 走既存統計，參考特徵因此
是 image-local、可跨幀保留。

**改動**：`edm_matcher.py` 新增 reference backbone 特徵 LRU（預設 16 筆、8.16 MiB/筆
fp16，約 130 MiB）、hit/miss/eviction 計數與 `reference_feature_cache_stats()`；
`benchmark_edm_site_replay.py` 每列記錄 `refs` 與該幀的 cache 增量。
`SFM_EDM_REF_FEATURE_CACHE=0` 可完全關閉。

**P119 replay A/B/A（河濱 river_site profile 724e35e0、2934 幀、RTX 5060）**

| | cache off (A) | cache off (A2) | cache on (B) |
|---|---:|---:|---:|
| wall p50 | 51.09 ms | 52.01 ms | **49.29 ms（−4.4%）** |
| wall p95 | 118.18 | 119.96 | **102.79（−13.7%）** |
| processing FPS | 14.92 | 14.72 | **15.84（+6.9%）** |
| 成功 | 1955 | 1955 | 2059 |
| inliers p50 | 690.5 | 690.5 | 693.0 |

夾測漂移 1.8%。A 與 A2 成功數完全相同，再次確認品質指標跨 run deterministic。

**收益集中在多參考路徑**（同 mode 同 refcount 的逐幀對照）：TRACK 1 ref 的 match 只有
−1.5%（28.43→27.99 ms），WEAK 3 refs −14.4%（88.37→75.62），LOST 10 refs −19%。
b=1 時省下的是「batch-2 backbone 換成 batch-1」約 3 ms，再扣掉組裝 4 層特徵的 cat +
`contiguous(channels_last)`，淨值很小。**隔離 microbenchmark 當時給的 TRACK −15.7% 沒有
在完整管線上兌現，以 replay 數字為準。**

**兩個設計決定（都有量測支撐）**
1. 每張影像一律以 batch=1 抽特徵。把 miss 和 query 併成一次呼叫在冷幀更快（冷幀變成與
   upstream 逐位元相同、+2.0%），但那會讓一幀的匹配結果取決於「當時剛好有哪些參考在
   快取裡」，`match_many_to_one` 就不再是輸入的純函數。這裡選確定性。代價是冷幀 TRACK
   +5.6%、WEAK −1.8%。
2. 組裝後必須 `contiguous(channels_last)`。cat 一個 expand 過的 query 會靜默產生 NCHW，
   餵進 neck 的損失大於快取的收益（修正前冷幀 TRACK 是 +26%）。

**與 upstream 的數值差**：warm 路徑約 1% 的 anchor cell 會換人，共用 cell 的 query 點
平均位移約 0.005 px、最大 0.073 px（真實地圖影像三組 pair）。原因是 query 改成單獨抽
特徵，屬於 upstream 本來就存在的 fp16 batch-shape 敏感度（同一組 pair 在 b=1 與 b=2 之間
的差異同一量級）。

**不要據此宣稱品質變好**：B 的 quality gate 由 fail 轉 pass，但 inliers p50 剛好落在
門檻值 693.0 上，且成功幀 +104 來自狀態機串聯（限跳閘的硬門檻把一個邊界幀翻面 →
TRACK/WEAK 切換 → 參考集改變）。這是這條管線對次像素擾動敏感的證據，不是定位品質提升。

**測試**：`定位演算法/validation/tests/test_reference_feature_cache.py`（純度：cache hit 與
miss 逐位元相同；anchor 與未快取 backbone 的一致度；hit/miss/eviction 帳；env 關閉後
不掛 patch；channels_last 回歸守門）。

**未做**：LOST recovery 掃描 192 筆 bank 會打穿 16 筆 LRU（B 有 83 次 eviction）；容量
尚未依場域調校。另 `reference_cache_size=32` 的 fp32 輸入 tensor 另佔約 72 MiB，有了特徵
快取後只有 miss 才需要，可再縮。

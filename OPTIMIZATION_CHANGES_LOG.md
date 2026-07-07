# 定位系統優化與修改紀錄

烏來場域定位建圖包 — runtime 優化 / 安全強化 / 光流整合 的完整紀錄。

---

## 一、原始 vs 現在 定位方法 比較表

| 面向 | 原始方法 | 現在方法 |
|---|---|---|
| **每幀是否做 deep match** | 每幀都跑 XFeat + LighterGlue,對 **5 個** local ref | **光流追蹤** + **每 6 幀**才 deep refresh(deep 只佔 14–17% 的幀) |
| **query 特徵抽取** | 每次 `_localize_with_candidates` 都重抽(nn/adaptive/full 一幀最多 3 次) | **每幀抽一次、跨 pass 共用**(#1);flow 幀完全不抽 |
| **LighterGlue full retry** | 把前 3 個 ref **再比一次** | **重用 adaptive pass 的匹配**(#2) |
| **ref 特徵傳輸** | 每幀把 ref 特徵重傳 GPU | **常駐 GPU 快取**(一次上傳) |
| **對應點組裝** | Python 逐點 for-loop | **numpy 向量化** |
| **追蹤策略** | 無(每幀重定位) | KLT 光流追 deep 內點,+ forward-backward check + 品質閘觸發 refresh |
| **FPS(720p @ 5Mbps 真機畫質)** | **7.6** | **47–87**(視場景;richer scene 更快) |
| **精度** | baseline | deep 優化 = **bit-identical**;flow = 差 median **0.015–0.017 map-unit**(對飛控 1.5u/3.0u 門檻可忽略) |
| **成功率(真機畫質)** | — | **100%**(250/250) |
| **安全機制** | 部分且有 bug | 完整(見第三節) |

### 定位組合(現在)
```
取得/迷路(BOOT_INIT/LOST): MegaLoc top-30 檢索 → XFeat → LighterGlue → PnP
正常追蹤(TRACK/WEAK):     XFeat mutual-NN 快路徑 → 不穩再 LighterGlue adaptive 3→5 → PnP
                          + 光流追蹤(refresh=6),deep 只在 refresh 幀跑
座標系: GLOMAP(x/z 水平、-y 上)
```

---

## 二、定位 FPS 優化(`production_xfeat_tracker.py`)

| # | 改動 | 效果 | 無損? |
|---|---|---|---|
| 1 | ref 特徵常駐 GPU 快取(不再每幀重傳) | — | ✅ bit-identical |
| 2 | 對應點收集向量化(取代 Python 逐點迴圈) | 131→72ms 的一部分 | ✅ |
| 3 | query XFeat 每幀抽一次,跨 nn/adaptive/full 共用 | ~8.5ms/幀 | ✅ |
| 4 | adaptive LighterGlue 的匹配在 full retry 重用 | ~10-12ms/幀(147/241 幀) | ✅ |
| — | (sync-gate / normalized-desc cache:實測 0 提升,**已還原**) | — | — |
| 5 | **光流追蹤模式**(`SFM_FLOW_TRACK=1`):deep 每 N 幀 refresh,中間 KLT 追蹤 | **13.9→47-87fps** | 非無損,差 ~0.015u |

累積:**7.6 → 47-87 fps**(deep 部分 7.6→13.9 為 bit-identical;flow 再 ×3.5)。

### 研究結論(2025-2026 光流文獻,5 團隊 + 對抗式驗證)
對「~150 稀疏點、causal、小位移」場景,**沒有更快又不掉精度的光流可換**;KLT 已在 Pareto 前緣。深度 tracker(CoTracker3/TAPIR)慢 5-50×;GPU-KLT 在 150 點是 overhead-bound 無法加速;RLOF 只在 KLT 掉 track 時有價值(是穩健度工具、非速度工具)。

---

## 三、飛行安全強化(3 方稽核:Codex + 2 個工作流)

`path_follow_flight.py` / `real_path_follow_controller.py` / `olympe_frame_source.py` / `mission_pipeline.py` / `live_localizer_worker.py`:

- **`airborne` UnboundLocalError**:起飛逾時會把飛機留空中且漏連線 → 修好。
- **`SafetyMonitor`(取代 PcmdWatchdog)**:獨立執行緒,GPU 卡死時仍能 LAND/EMERGENCY/切手動;MANUAL 完全不送 PCMD(不搶飛手);看門狗只在主迴圈跳動後才啟用。
- **NaN 位姿閘**:非有限值一律當無定位 → 懸停(否則會繞過所有門檻亂飛)。
- **get_pose 例外 + CUDA OOM**:包 try/except → 懸停;OOM 清快取復原。
- **WEAK 低信心 → 懸停**(可 `SFM_GATE_WEAK=0` 關)。
- **起飛前先載入模型**(`ensure_models`)。
- **看電桿**:預設沿航線飛(gimbal 看電桿),不用機身朝電桿飛(避免撞線)。
- **mission_pipeline** 預設 bundle 改回 v3;安全指令改**原子寫入**。
- **連續性閘**(定位):跳離上一接受點 >1.5u → 視為斷裂不採用,連兩次一致才當重定位。
- **低信心 → 懸停 + MegaLoc 重定位 → 切手動**(本次新增,`run_loop`):
  - PnP inliers < `SFM_LOW_CONF_INLIERS`(60)或 WEAK 或無定位 → **懸停**(絕不用低信心 pose 飛),並要求 tracker 進 LOST 跑 **MegaLoc** 重定位。
  - MegaLoc 在 `SFM_LOST_MANUAL_S`(3s)內救不回來 → **切 MANUAL 交給飛手**(僅 SkyController 有真手動;純 WiFi 無飛手則保留 `LOST_LAND_S`=4s 自動降落 failsafe)。
- **flow 也套同一信心閘**(Codex #1):flow 幀若 reproj>4 / inliers<50 / track ratio 低 → **不發布該幀 pose(回 None)** + 下幀強制 deep refresh。增精度、FPS 不變(好畫質時完全不觸發)。
- legacy 路徑(autoflight/cruise_geofence/run_real)finally 清理強化;autoflight DONE 真飛也會降落。

### Codex 稽核修正(本次)
- **#1 flow 信心閘**:見上(低品質 flow 幀不發布)。
- **#3 UI 狀態檔降頻**:`flight_operator_app` 每幀寫 `/tmp` JSON → 降到 ~5Hz + compact,高 FPS 下 UI 更穩,精度不變。
- **#4 兩份副本同步 + LOST→MANUAL 實作**(已完成,見上)。
- **#5 ref cache LRU**:滿 700 改成**丟最舊一個**(取代整包 clear),長航線跨區無 latency spike。
- **#2 CUDA sync**:實測 0 提升(.cpu()/PnP 本就同步),**不改**。

---

## 四、操作介面(`flight_operator_app.py` / `draw_path.py`)

- **「開始巡檢」閘**:按下才開始串流輸入 + 定位;顯示**整體 FPS**。
- **`--stream-fps`**:可設串流速率(測試用 29)。
- **繪製路徑疊圖**:把畫的巡檢路徑以**洋紅色**疊在地圖(aligned→glomap 轉換),與藍色實際軌跡區分。
- **地圖預設放大 3.2×、滾輪最大 120×**。
- **定位失敗 / 低信心 指示**(本次新增):
  - 影片上方**紅色橫幅 = 定位失敗**、**黃色 = 低信心**(inliers/reproj)。
  - telemetry 面板**中文大字**:定位正常 / 定位信心低 / 定位失敗。
  - 地圖軌跡上用**紅/黃點標出「哪個位置」**定位弱或失敗。
  - 門檻:`SFM_LOC_LOW_INLIERS`(預設 60)、`SFM_LOC_HIGH_REPROJ`(預設 4.0)。
- **draw_path 導覽**:Shift+左鍵拖=平移、Alt+左鍵拖=旋轉。

---

## 五、Benchmark 結果(720p)

| 影片 / 條件 | 模式 | FPS | 成功率 | inliers | 精度(vs baseline) |
|---|---|---|---|---|---|
| P1210121 乾淨(2.7K→720p) | deep 原始 | 7.6 | 100% | 125 | 基準 |
| P1210121 乾淨 | deep 優化(#1+#2) | 13.9 | 100% | 125 | **0.0000mm(bit-identical)** |
| P1210121 乾淨 | flow | 47 | 100% | 121 | 0.015u |
| P1210121 **5Mbps 真機畫質** | deep | 13.7 | 100% | 126 | 0.015u |
| P1210121 **5Mbps** | flow | 53 | 100% | 121 | 0.017u |
| **P1230123 5Mbps** | **flow** | **87** | **100%** | 255 | — |

**結論:5Mbps 真機串流畫質對定位幾乎無影響(inliers 不降、成功率 100%);光流方法在真機速度+真機畫質下驗證通過(47-87fps、精度損失可忽略)。**

### 環境相依(原機有裝但 `requirements_runtime.txt` 漏列,已補)
`huggingface_hub`、`safetensors`(MegaLoc 權重)、`kornia`(LighterGlue)。

### 可調環境變數
`SFM_FLOW_TRACK`(光流開關)、`SFM_FLOW_REFRESH`(refresh 間隔)、`SFM_GATE_WEAK`、`SFM_MAX_POSE_JUMP_U`、`SFM_LOW_CONF_INLIERS`、`SFM_LOC_HIGH_REPROJ`、`SFM_LOST_MANUAL_S`、`SFM_STREAM_LOST_LAND_S`、`SFM_SYSTEM_ROOT` 等。

---

## 六、4-方向 Audit 修正(多 agent 稽核 + 修復)

4 個 read-only agent 掃描 architecture / duplication / performance / config-error-test,合併去重後修復(破壞性/設計項先經確認):

### 結構(P1)
- **鏡像補完**:`reloc_localizer_xfeat.py` 原本只在 deploy_code → 複製進 flight_control;新增 `sync_mirror_check.sh`(8 檔 diff,drift 即報錯)。
- **`pose_types.py`**:抽出中性 `Pose`/`Localizer`,localizer 改 `from pose_types import`;autoflight re-export 保相容。**驗證:import localizer 不再載入 plan_path/autoflight**。
- **`mission_pipeline` python fallback**:`--python` 預設改 `SFM_LOCALIZER_PYTHON`→`/usr/bin/python3.12`(存在才用)→`sys.executable`。

### 安全(P1)
- **SafetyMonitor.poll() 失敗記錄**:原 `except: pass` 靜默 → 首次失敗限速印警告(避免操作員接管權被靜默癱瘓)。
- **低信心閘上 selftest**:`--selftest` 新增第 9 項驗證「低信心→懸停+MegaLoc→切手動」。**過程中抓到並修掉一個真 bug**:`if fresh: lost_since=None` 會在每個 fresh-但-低信心幀重置計時器 → 導致低信心永遠懸停、無法升級切手動;改成只在「好定位」(drive 分支)重置。
- **flow 信心閘**(前述 Codex #1)+ **連續性閘** 一併納入 selftest 覆蓋。

### 效能(P1,飛行熱路徑)
- **時序快取重建短路**:seed ref 集合未變時不再 GPU→CPU→GPU 重建,只刷新 age。**驗證:定位 bit-identical(0.00000u)、FPS 不變**。
- (P1-7 lazy payload、P1-6b GPU-gather:flow 模式下實益極小且動到相關性敏感碼,**暫緩**。)

### 設定/路徑/紀錄(P2)
- **env 變數統一**:UI 的 `SFM_LOC_LOW_INLIERS` → `SFM_LOW_CONF_INLIERS`(與飛控一致)。
- **`find_system_root` 硬編路徑**:5 處 `/media/cihcilab/...` fallback → 改吃 `SFM_SYSTEM_ROOT`,否則明確報錯(不再靜默解析到外機路徑)。
- **eval 吞例外**:`validation/eval_stream_core.py` match/PnP 例外改記錄 + error 幀分開計數(不再偽裝成 map 品質不足)。
- **UI 每 tick render**:加 dirty-flag(pose/frame/view 沒變不重繪)、批次投影、點雲向量化;worker client 兩份 → `LiveWorkerClient` base。(非飛行路徑;layout-selftest 通過)
- **ref cache LRU**(前述 Codex #5)。

### 依你確認保留(未動邏輯,只標註)
- **死碼**(safe_volume/pole_cruise/_localize_with_temporal_cache/XFeatLightGlueLocalizer):加註「未接入生產路徑」,不刪。
- **SparseCloudCollisionMonitor**:標註「未接入,避障靠飛手,接入為待決設計」,不接不刪。
- **Pose 三份合一**:座標慣例分歧有飛安風險,**跳過**(只做 pose_types)。

### 驗證
compile 全過、**flight selftest 9 項全過**、flow 定位 **bit-identical**、mirror check OK、介面正常(flow=1)。

---

## 七、第二輪 Codex 稽核修正 + UI

### 定位精度(不降 FPS)
- **加權 PnP**(Codex 首推):flow PnP 只在有「真正 FB 離群漂移點」時把它們排除於 pose solve 之外(絕對門檻保護 → 乾淨畫面**零回歸**、單解不加第二次 solve → FPS 不降)。實測乾淨片精度 bit-noise、13/250 幀觸發。

### 飛行安全(Codex 第二輪)
- **#3 MANUAL 完全靜默**:移除進 MANUAL 時的一次 zero PCMD(不搶飛手)。
- **#4 安全檔啟動原子重設 auto**:防舊 run 殘留的 land/emergency 被新任務第一輪 poll。
- **#1 重定位跳點不污染 heading**:`HeadingEstimator.mark_teleport()` 跳過該幀位移更新,保留已學 yaw offset。
- **#2 flow 加 max_jump gate**:KLT/PnP 翻轉跳太遠不發布。
- **#8 pitch 只看水平 X/Z**:純垂直修正不再產生假前傾(gaz 管 Y)。
- **#6 worker fd-level stdout 轉向**:原生 lib(如 LightGlue 載入訊息)不再污染 JSON IPC。
- **#12 requirements 補** huggingface_hub/safetensors/kornia pin。
- 暫緩(有理由):#5 worker IPC timeout、#7 boot lock timeout(UI robustness,較大改動)、#9 temporal cache inliers(需 benchmark)、#10 tobytes(非瓶頸)、#11 python probe(env override 已覆蓋)。

### 操作介面
- **無法定位永久標記**:每次定位失敗在最後已知位置留**紅點**(NO_LOC_DEDUP_U 去重),永久累積,取代先前的危險平面。

### 備份
code-only 私有 repo: github.com/1122-gggggg/wulai-sfm-localization(大檔 gitignore)。

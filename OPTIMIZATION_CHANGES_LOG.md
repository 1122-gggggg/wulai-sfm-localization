# 定位系統 完整修改紀錄（原始版 → 現在版）

烏來場域無人機定位/巡檢 runtime。本檔是這次調校的**完整變更記錄**:原始行為 → 現在行為、為什麼改、怎麼驗證。

> 說明:原始的自壓縮封存檔已在初期清理時刪除,git 基線 commit 是調校中途才建立的,所以本檔不是機械式 `git diff`,而是**逐項對照的策展記錄**,每項都已對照現行程式碼確認。GitHub 備份:私有 repo `github.com/1122-gggggg/wulai-sfm-localization`(僅程式碼,大檔 gitignore)。

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

---

## 八、可調環境變數(完整)

**定位/光流**:`SFM_FLOW_TRACK`(光流開關)、`SFM_FLOW_REFRESH`(refresh 間隔,預設 6)。
**安全門檻**:`SFM_LOW_CONF_INLIERS`(60)、`SFM_LOC_HIGH_REPROJ`(4.0)、`SFM_GATE_WEAK`、`SFM_MAX_POSE_JUMP_U`(1.5)、`SFM_MAX_ROUTE_DEVIATION_U`(3.0)、`SFM_LOST_MANUAL_S`(3)、`SFM_STREAM_LOST_LAND_S`(15)、`SFM_WEAK_HOVER_LAND_S`、`SFM_PCMD_WATCHDOG_S`、`SFM_SAFETY_FILE`、`SFM_ALLOW_LEGACY_FLIGHT`。
**UI**:`SFM_NO_LOC_DEDUP_U`(1.0)、`SFM_WORKER_TIMEOUT_S`(8)。
**路徑/資源**:`SFM_SYSTEM_ROOT`、`SFM_RELOC_BUNDLE`、`SFM_MEGALOC_CACHE`、`SFM_FLIGHT_PATH_JSON`、`SFM_POLES_JSON`、`SFM_MAP_ROOT`、`SFM_SAFEZONE_DIR`、`SFM_LOCALIZER_PYTHON`、`SFM_DETECTOR_PYTHON`、`SFM_OLYMPE_CONTROLLER`、`SFM_GLOMAP`、`SFM_TOOLS_ROOT`。

---

## 九、暫緩項目(有理由,未做)
- **起飛前重力對齊檢查**:起飛 hover 後拿 PnP 相機姿態比對 IMU 重力,地圖 -Y 與真實重力差 >2-3° 就禁 AUTO。**真機才用**,需一次真機 hover 校正 IMU→相機座標慣例,故未實作。
- **#9 temporal cache 用 validated inliers 而非 full-ref**:品質/調參取捨,需 benchmark 才知優劣。
- **#10 frame.tobytes 複製**:實測 IPC ~55-83MB/s 遠低於 pipe 頻寬,非瓶頸。
- **#11 mission_pipeline python probe-import**:`SFM_LOCALIZER_PYTHON` env override 已是逃生口,大致涵蓋。

---

## 十、換場域須知(演算法不變)
換另一個場域**只換資料檔 + 指路徑**,演算法完全不動:① reloc bundle `.pt`(`SFM_RELOC_BUNDLE`)、② 顯示點雲 `.ply`、③ 巡檢路徑 `flight_path.json`(`SFM_FLIGHT_PATH_JSON`)、④ 電桿 `poles.json`(`SFM_POLES_JSON`)、⑤ MegaLoc 快取 `.npy`(選用)。同一台 ANAFI 則相機內參 `CAM_720` 不動。①②需用**完整 repo 的建圖 pipeline**(GLOMAP + bundle builder)產生,不在本 runtime 套件內。

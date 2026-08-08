# audit/system_inventory.md — 系統全貌清單

稽核日期：2026-08-07
基準：`agent/localization-runtime-optimizations` @ `d0b2250`（工作區 dirty，63 檔變更）
方法：AST 全掃 + `git ls-files` + 實跑 probe。**以程式行為為準，不採信 README 與註解。**

---

## 1. 專案目錄結構（來源樹，排除產出物）

```
/home/allen/localization/
├── 控制介面程式/                    操作介面 + 無人機後端（UI 層）
│   ├── operator_interface/          Tk UI、Olympe 後端、定位 worker client
│   ├── site_profiles/               每場域一份原子化資產設定（JSON）
│   ├── mission_configs/             任務設定
│   ├── authoring/                   航線繪製工具（Blender，離線）
│   ├── 影片模擬串流/                模擬入口（simulated-stream）
│   ├── 真機串流/                    真機入口（real-flight）
│   ├── site_profile.py              設定載入與 fail-closed 驗證（801 行）
│   ├── mission_pipeline.py          任務管線 + 自主鎖（568 行）
│   ├── workspace_layout.py          路徑解析
│   └── SAFETY.md                    **操作員綁定的安全政策（本稽核遵守）**
├── 定位演算法/                      定位與飛控
│   ├── flight_control/              飛控、安全監控、路徑跟隨（UI 程序載入）
│   ├── deploy_code/
│   │   ├── sfm_glomap_deploy/       EDM 定位堆疊（定位 worker 子程序載入）
│   │   └── runtime/EDM/             EDM 模型原始碼與權重
│   ├── EDM工具包/                   建圖與 benchmark 工具
│   ├── validation/                  驗證腳本與測試
│   ├── pipeline/                    離線定位管線
│   └── configs/                     EDM profile JSON
├── 模擬器/
│   ├── parrot_stimulate/            ANAFI PCMD 模擬器（獨立 Python 3.11 venv）
│   └── sphinx_anafi_path_convergence/  Sphinx 收斂測試
├── 地圖檔/場域/                     每場域的地圖、bundle、航線、報告
├── 執行環境/                        runtime 需求、torch_hub_cache、第二套 manifest
├── tools/                           安裝、驗證、打包、稽核工具
├── outputs/                         產出（flight_logs 約 100 個 session、benchmark）
├── 文件/                            ARCHITECTURE.md、SYSTEM_SPEC.md、WORKSPACE_AUDIT.md
└── audit/                           **本次稽核產出**
```

**規模**：222 個 tracked `.py`，約 86,000 行 Python。
`bandit` 掃描的 runtime 程式碼為 40,132 行。

---

## 2. 執行入口

### 2.1 操作員入口（互斥，由 argv 字面值固定）

| 入口 | 實際指令 | 模式 pin 位置 |
|---|---|---|
| 真機 | `控制介面程式/真機串流/啟動.sh` → `operator_interface/start_anafi_live.sh` | `--interface real-flight` 字面值（`start_anafi_live.sh:233`） |
| 模擬 | `控制介面程式/影片模擬串流/啟動.sh` | `--interface simulated-stream` 字面值（`啟動.sh:252`） |
| 模擬（選檔） | `控制介面程式/影片模擬串流/選擇啟動.sh` | 同上 |
| 航線編修 | `控制介面程式/authoring/航線編修.sh` | — |
| 系統驗證 | `./驗證系統.sh` → `tools/system_validation.py`（15 步） | — |

專案根目錄 `開啟介面的終端機代碼` 記錄操作員實際使用的真機指令。

**六層模式強制**（實測 dry-run 確認）：launcher argv 守衛 → argv 字面值 →
`resolve_operator_interface` → backend 分支（Olympe 只在 `if args.live` 內 import）→
`install_network_guard` → `SessionConfig` 的 `INTERFACE_MISMATCH` / `HOT_SWITCH_PROHIBITED`。
**不存在任一方向的 fallback。**

### 2.2 其他 `__main__` 入口

61 個檔案含 `if __name__ == "__main__"`。分類：
- **飛控**（`定位演算法/flight_control/`）：`path_follow_flight.py`（自主，**硬鎖**）、
  `autoflight.py`、`manual_nudge_pilot.py`、`passive_flight_session.py`、`pole_cruise.py`、
  `cruise_geofence.py`、`gravity_calibration.py`、`log_anafi_telemetry.py`、`olympe_frame_source.py`
- **定位／建圖**：`EDM工具包/build/build_reloc_map_edm.py`、`deploy_code/runtime/EDM/{train,test}.py`、
  `pipeline/localize_pipeline.py`、各 `reloc_localizer_*.py`
- **驗證／benchmark**：`定位演算法/validation/*.py`（約 20 個）
- **工具**：`tools/{system_validation,simulator_preflight,workspace_audit,package_manifest,export_simulator_package}.py`

---

## 3. 模組清單（依責任）

| 類別 | 模組 | 行數 | 備註 |
|---|---|---|---|
| **UI** | `operator_interface/flight_operator_app.py` | 7800 | `OperatorApp` 3993 行 / 106 方法 / **144 個 `__init__` 屬性** |
| | `operator_interface/route_editor_window.py` | 1082 | `RouteEditorWindow` 1024 行 / 46 方法 / 45 屬性 |
| | `operator_interface/site_assets_panel.py` | 495 | 場域資產面板 |
| | `operator_interface/operator_actions.py` | — | 按鈕定義表 |
| **無人機 SDK／連線** | `operator_interface/olympe_live_backend.py` | 4357 | `OlympeLiveBackend` 3749 行 / 93 方法 / **113 屬性** |
| | `flight_control/olympe_frame_source.py` | 1071 | PDRAW 影格來源（**與 deploy 副本分歧 332 行**，見 F-07） |
| **定位** | `operator_interface/live_localizer_worker.py` | 938 | 定位子程序主體 |
| | `operator_interface/live_localizer_protocol.py` | 76 | mode ↔ code 對照 |
| | `deploy_code/sfm_glomap_deploy/production_localizer_factory.py` | — | **provider 選擇點** |
| | `deploy_code/sfm_glomap_deploy/production_edm_tracker.py` | 991 | EDM 追蹤器（現行 production） |
| | `flight_control/production_xfeat_tracker.py` | 2091 | XFeat 追蹤器（鏡像檔） |
| **地圖載入** | `operator_interface/local_site_assets.py` | 955 | `LocalSitePackageProvider`；**含 `sys.path[0]` 插入，見 F-17** |
| | `operator_interface/site_asset_interfaces.py` | 41 | 資產抽象介面 |
| | `flight_operator_app.py:2729-2809` | 81 | `read_ply_points` / `read_reference_pose_points` / `read_map_points`（**寫在 UI 檔內**） |
| **控制命令** | `operator_interface/backend_contract.py` | 538 | **typed 契約**：`ControlAction`、`ControlRequest`、`ControlResult`、`FailureReason` |
| | `flight_control/manual_nudge_pilot.py` | 705 | 微移常數來源（**deploy 副本未納 git，見 F-06**） |
| | `operator_interface/scale_free_control_adapter.py` | 40 | 載入模擬器內的權威控制核心 |
| **安全飛控** | `flight_control/path_follow_flight.py` | 2443 | `SafetySwitch`(120) + `SafetyMonitor`(432) + `run_loop` + `fly()`（**硬鎖**） |
| | `flight_control/real_path_follow_controller.py` | 1199 | 控制器 + `CollisionMonitor`（**scipy 選用，見 F-08**） |
| | `flight_control/cruise_geofence.py` / `safe_volume.py` | — | 地理圍欄 |
| **狀態管理** | `flight_operator_app.py:1114-1222` | 108 | `DroneState`：**約 80 欄位的扁平 dataclass**，UI 與 backend 共用 |
| **設定管理** | `控制介面程式/site_profile.py` | 801 | **嚴格 fail-closed**：拒未知鍵、非有限 JSON、SHA 格式 |
| | `deploy_code/sfm_glomap_deploy/edm_profile.py` | — | EDM profile 驗證 |
| **日誌** | `operator_interface/runtime_safety.py` | 629 | `SessionLogs`（4 條 JSONL）+ 磁碟保留 + 網路守衛 + arming 閘門（**名實不符**） |
| | `olympe_live_backend.py:443` | 69 | `_CmdLog` |
| **錯誤處理** | — | — | **無統一例外階層**；失敗以 bool／None／例外混合表達 |

---

## 4. 執行緒、程序與 async task

### 4.1 程序

| 程序 | 啟動者 | sys.path 樹 | 說明 |
|---|---|---|---|
| UI 主程序 | `start_anafi_live.sh` | **flight_control** | Tk + Olympe backend |
| 定位 worker | `flight_operator_app.py:2129` `subprocess.Popen` | **deploy_code** | stdio JSON + shared_memory |
| 偵測 worker | 同上 | — | 選用 |
| 模擬器（獨立） | `tools/system_validation.py:164` | Python **3.11** venv | `parrot_stimulate` |

> **兩棵樹同時在不同程序中存活**——這是 F-06／F-07／F-17 的根因。

### 4.2 執行緒（UI 程序）

| 名稱 | daemon | 位置 | 停止 | 例外處理 |
|---|---|---|---|---|
| Tk 主執行緒 | — | `mainloop()` | `destroy()` | tick 例外會**永久終止 tick 鏈**（無人重新 arm） |
| `nudge-hold-loop` | 是 | `olympe_live_backend.py:3060` | `_nudge_loop_stop` Event | `except/finally`：清 hold + 歸零 + 記錄 ✓ |
| `sc-stick-monitor` | 是 | `olympe_live_backend.py:298` | `stop()` | 斷線回報 |
| `runtime-flight-safety` | 是 | `olympe_live_backend.py:3740` | — | — |
| `olympe-ui-<command>` | 是 | `flight_operator_app.py:4245` | 自然結束 | 經 `_flight_results` 回主執行緒；**但會跨執行緒操作 Tk，見 F-33** |
| `live-localizer-client` / `live-detector-client` | 是 | `flight_operator_app.py:2074` | `_stop_process()`：terminate→1s→kill | 有重啟 + warmup 節流 |
| `anafi-nal-loss` | 是 | `flight_operator_app.py:1911` | — | — |
| `route-editor-map-load` / `-ply-scan` | 是 | `route_editor_window.py:814,925` | — | — |
| `site-asset-import` | 是 | `site_assets_panel.py:246` | — | — |
| Pdraw callback thread | Olympe 擁有 | `olympe_frame_source.py` | Olympe | — |

### 4.3 執行緒（自主飛行，目前不執行）

`safety-monitor`（`path_follow_flight.py:848`，20 Hz，daemon）、
`run_loop` 控制迴圈、`nudge-<name>` pulse threads（`manual_nudge_pilot.py:417`）、
`run_deadman`（`:688`）。

### 4.4 Tk `.after()` 週期迴圈

| 迴圈 | 週期 | 位置 |
|---|---|---|
| `tick` | 100 ms（`next_tick_deadline` 以 deadline 再排程，**不會重疊**） | `:3160`, `:6878` |
| `poll_localization_results` | 105 ms → `_loc_result_poll_ms` | `:3161`, `:3174` |
| `_check_active_site_route` | 1200 ms | `:3165` |
| `_poll_route_editor_load` | 50 ms | `:4673`, `:4681` |
| `_guard_tick`（route editor） | — | `route_editor_window.py:107` |

### 4.5 Queue 與 Lock

| 物件 | 上限 | 位置 |
|---|---|---|
| worker pending | `maxsize=1` | `flight_operator_app.py:2059` |
| worker results | 有界 | `:2060` |
| **`_flight_results`** | **無界** ← F-19 | `:2918` |
| site asset / route editor queues | `maxsize=1` | `site_assets_panel.py:73`、`route_editor_window.py:148,157` |
| frame shared_memory | 2 槽 | `flight_operator_app.py:2042` |

Lock 共 24 處，主要為 `OlympeLiveBackend._lock`（RLock）、`_firmware_limits_lock`、
`_calibration_lock`、`LiveWorkerClient._lock` / `_proc_lock`、`_flight_inflight_lock`、
`SafetyMonitor._io_lock`（**跨阻塞呼叫持有，見 F-10**）。

### 4.6 訊號處理

`flight_operator_app.py:7732` 註冊 SIGINT／SIGTERM／SIGHUP，含 `exit_in_progress` 防重入；
**註冊失敗會明確警告操作員**（`:7744-7749`）。另有 `atexit`。
`path_follow_flight.py:1799`、`manual_nudge_pilot.py:673`、`autoflight.py:221`、
`cruise_geofence.py`、`passive_flight_session.py:58` 各有自己的訊號處理。

---

## 5. 長時間執行的迴圈

| 迴圈 | 頻率 | 退出條件 |
|---|---|---|
| Tk `mainloop` | — | `destroy()` |
| `tick` | 10 Hz | 自我再排程；**例外即永久停止** |
| `nudge-hold-loop` | 20 Hz | `_nudge_loop_stop` / 無 hold / cleanup / landed / pilot_sticks |
| `SafetyMonitor._run_loop` | 20 Hz | `_stop` Event |
| `run_loop`（自主） | 20 Hz | terminal latch / stop flag |
| `LiveWorkerClient._loop` | 事件驅動 | `_closed` Event |
| worker frame loop | 事件驅動 | stdin EOF |
| `start_anafi_live.sh` UI 監看迴圈 | — | 子程序結束 |

---

## 6. 設定管理

**四個獨立來源，無單一 resolver**（見 F-13）：
1. site profile JSON（`site_profile.py`，schema v1/v2，**嚴格驗證**）
2. EDM localizer profile JSON（`edm_profile.py` + `EDMConfig.validate()`）
3. **約 95 個 `SFM_*` 環境變數**（多數在 import 時求值）
4. argparse 預設值（其預設本身即 `os.environ.get(...)`）

顯式優先序只有一處：`--site-profile` 與逐項 flag／env 互斥（`flight_operator_app.py:576-582`），
其餘為載入順序湧現。

**可弱化安全的環境變數**：`SFM_MAX_TILT_DEG`、`SFM_MAX_VERTICAL_SPEED_MS`、
`SFM_MAX_ROTATION_SPEED_DEGS`、`SFM_MIN_TAKEOFF_BATTERY_PCT`、`SFM_RTH_MIN_ALTITUDE_M`、
`SFM_STREAM_LOSS_GRACE_S`、`SFM_ALLOW_LEGACY_FLIGHT=1`（繞過 geofence）、
`SFM_GATE_WEAK=0`（在弱定位上飛行）、`SFM_SAFETY_FILE`。

---

## 7. 硬編碼路徑與全域狀態

**硬編碼絕對路徑**：runtime 程式碼與腳本中**僅 1 處**出現 `/home/allen`，
且為 `定位演算法/flight_control/path_follow_flight.py:89` 說明前次稽核已移除的**註解**。
這是明顯優於一般研究專案的表現。

**`/tmp` 路徑**（bandit B108，共 6 處）：最重要的是
`path_follow_flight.py:258` 的 `SAFETY_FILE = /tmp/sfm_drone_safety.cmd`（見 F-15）。

**模組層可變全域**：
| 名稱 | 檔案 | 性質 |
|---|---|---|
| `_NETWORK_GUARD_INSTALLED` | `runtime_safety.py:471` | 布林旗標，`install_network_guard` 內 `global` 修改（idempotent 守衛） |
| `MODE_TO_CODE` / `CODE_TO_MODE` | `live_localizer_protocol.py:13,22` | 常數對照表 |
| `_ELIGIBLE_LOG_NAMES` / `_PERMANENT_LOG_NAMES` / `_INCIDENT_EVENTS` | `runtime_safety.py:31,36,44` | 常數集合 |
| `_NO_PAYLOAD_ACTIONS` / `_LEGACY_ACTIONS` | `backend_contract.py:159,182` | 常數集合 |
| `STATE` | `demo_stub_http_server.py:18` | demo stub，非 production |

**沒有發現**跨模組以全域變數交換資料的情形。全域幾乎都是常數表。

**真正的 shared mutable state 是 `DroneState` 物件**：
`backend.poll()` **以參考回傳**，UI 直接寫入 `backend.state` 的
`stream`／`mode`／`loc`／`tracker_state`／`pose`／`inliers`，
同時 Olympe callback 執行緒也在修改同一物件。這是本系統最主要的共享可變狀態。

---

## 8. 測試程式

50 個測試檔 / 896 個測試函式 / **1070 個收集到的測試**，30.2 秒跑完。

| 檔案 | 行數 | 層級 |
|---|---|---|
| `operator_interface/test_olympe_live_backend_safety.py` | 3081 | **整合**（125 個中 120 個建構真實 `OlympeLiveBackend`，對假 olympe SDK） |
| `flight_control/test_flight_safety_gates.py` | 2551 | **整合**（31 個驅動真實 `run_loop`、25 個驅動 `SafetyMonitor` 執行緒，虛擬時鐘） |
| `operator_interface/test_worker_lifecycle.py` | 1541 | **整合**（真子程序：崩潰、重啟冷卻、阻塞寫入、EOF） |
| `operator_interface/test_route_editor.py` | 1000+ | 單元 |
| `operator_interface/test_site_profile.py` | 721 | 單元（設定驗證，覆蓋最好） |
| `operator_interface/test_operator_render_perf.py` | 560 | **唯一建構真實 `OperatorApp` 的測試**（23 個，需 X display） |
| `operator_interface/test_operator_command_safety.py` | 607 | 混合（11 個為原始碼字串斷言） |
| `模擬器/parrot_stimulate/tests/*` | 1357 | **被 `pytest.ini` 排除**（見 F-23） |

**mock 與 fake**：假 olympe SDK 注入 `sys.modules`、虛擬時鐘、假 drone。
**46 處以 `__new__` 繞過建構子**，因此多數 `__init__` 未被驗證。

**模擬器**：`模擬器/parrot_stimulate`（ANAFI PCMD 模擬器）、
`模擬器/sphinx_anafi_path_convergence`（Sphinx）。

**端對端 smoke**：`tools/simulated_ui_smoke.sh`（啟動真實 GUI，等定位 feed 就緒），
但為 opt-in（`--ui-smoke`），不在 1070 內。

---

## 9. 部署腳本

| 腳本 | 用途 |
|---|---|
| `tools/install_runtime.sh` | 建立 venv（強制 CPython 3.10、**拒絕 system-site-packages**）、以 hash-locked 需求安裝 |
| `tools/test_clean_install.sh` / `test_portable_runtime.sh` | 乾淨安裝驗證 |
| `tools/system_validation.py` | **15 步無飛行驗證矩陣**，產出 JSON receipt |
| `tools/simulator_preflight.py` | 模擬入口的完整 preflight（資產 SHA、CUDA、模型載入） |
| `tools/workspace_audit.py` | 目錄佈局與磁碟檢查 |
| `tools/package_manifest.py` | MANIFEST.tsv / SHA256SUMS 產生與驗證 |
| `tools/export_simulator_package.py` | 可攜套件匯出 |
| `定位演算法/validation/check_runtime_mirrors.py` | authoritative 鏡像檔漂移檢查；由 system validation 執行 |

---

## 10. 外部相依

**直接 pin（14 個）**：torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 2.2.6、
opencv-python 4.13.0.92、pillow 12.3.0、pycolmap 4.0.4、**protobuf 3.19.4**（綁定 Olympe）、
**parrot-olympe 8.4.0**、safetensors 0.8.0、kornia 0.8.2、einops 0.8.2、joblib 1.5.3、
loguru 0.7.3、yacs 0.1.8。

`requirements-lock.txt`：49 個 pin / 525 行 hash，由 `uv pip compile --generate-hashes` 產生。

**系統層（非 pip，僅註解提及）**：`ffmpeg`、`python3-tk`。

**實測環境漂移**（見 F-04）：`.venv` 有 `include-system-site-packages = true`；
`pyyaml 5.4.1` 與 `markupsafe 2.0.1` 來自 `/usr/lib/python3/dist-packages`（lock 為 6.0.3 / 3.0.3）；
6 個版本不符；37 個 venv 內套件不在 lock；`lingbot-map` 指向不存在的目錄。

---

## 11. 模型與資料檔載入位置

| 資產 | 位置 | 完整性機制 |
|---|---|---|
| XFeat / LighterGlue / MegaLoc | `執行環境/torch_hub_cache/`（package-local torch.hub） | 離線，不從網路抓取 |
| EDM 權重 | `定位演算法/deploy_code/runtime/EDM/weights/` | — |
| 地圖 PLY / bundle / reference poses | `地圖檔/場域/<site>/` | `site_profile.asset_sha256`（**逐項 SHA-256**） |
| 航線 JSON | `地圖檔/mission_routes/`、`地圖檔/場域/<site>/routes/` | site profile |
| 測試影片 | `模擬器/測試影片/` | `simulator_preflight.py` 驗 SHA-256 |

> `MANIFEST.tsv` / `SHA256SUMS` **不涵蓋** `地圖檔/`、`模擬器/測試影片/`、`outputs/`；
> 這些資產的完整性改由 `site_profile.asset_sha256` 與 preflight 提供。

---

## 12. 日誌位置

`outputs/flight_logs/session_<UTC時間>_<mode>_<uuid8>/`，目錄權限 `0o750`。
每 session 四條 JSONL：`commands`、`localization`、`telemetry`、`incidents`，
外加 `session_manifest.json`（啟動時，原子寫入）、`session_summary.json`（關閉時）、
`hardware_inventory.json`、`video_inventory.json`。

**保留政策**：`runtime_safety.enforce_retention()` — 30 天 / 20 GiB，
但**只涵蓋** `localization.jsonl`、`performance.jsonl`、`video_metrics.jsonl`、`loc_metrics_*`；
`commands`／`incidents`／`telemetry` 列為永久保留。

**目前狀態**：約 100 個 session 目錄，`outputs/` 共 2.20 GiB；
磁碟可用 20.18 GiB（**14.5%，已低於 15% 警告門檻**）。

---

## 13. 責任重疊摘要

| 模組 | 應保留 | 應拆出 |
|---|---|---|
| `OperatorApp` | 視窗生命週期、widget 事件繫結、算繪排程 | 定位 pipeline 編排、metrics JSONL 寫入、地圖／視訊點陣化、重力校正、場域切換、命令派送 |
| `OlympeLiveBackend` | Olympe 連線、PCMD wire、telemetry 讀取 | 起飛 preflight（18 道閘門）、韌體限制、磁羅盤校正、RTH 政策、飛行錄影、磁碟健康 |
| `runtime_safety` | — | 目前混合 4 件無關責任：session 日誌、磁碟保留、網路策略、autonomy arming 閘門 |
| `flight_operator_app.py`（模組層） | UI 相關純函式 | `read_ply_points` 等地圖 I/O（應歸地圖 provider） |

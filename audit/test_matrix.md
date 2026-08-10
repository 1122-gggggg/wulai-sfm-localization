# audit/test_matrix.md — 測試矩陣

稽核日期：2026-08-07
基準測試結果：**1070 passed, 1 skipped, 17 warnings in 30.18s**（稽核起點，乾淨執行）
規模：50 個測試檔 / 896 個測試函式 / 1070 個收集到的測試

---

## 1. 現有測試分層（實際層級，非依檔名判斷）

| 檔案 | 測試數 | **實際層級** | 真實性評估 |
|---|---|---|---|
| `tests/control_interface/operator_interface/test_olympe_live_backend_safety.py` | 125 函式 / **157 收集** | **整合** | `make_backend()`（`:584`）建構**真實** `OlympeLiveBackend`（`:624`），被使用 **123 次**，另 2 處直接建構 → 幾乎每個測試都跑真實類別，對注入 `sys.modules` 的假 olympe SDK。**注意**：helper 把 `_connect` monkeypatch 成 no-op，故連線路徑本身只在 `:863` 的 4 個參數化案例中執行。這是全庫最強的測試 |
| `tests/localization/flight_control/test_flight_safety_gates.py` | **149 收集** | **整合** | 31 個驅動真實 `run_loop`，25 個驅動真實 `SafetyMonitor` 執行緒（虛擬時鐘）。96 個故障注入 |
| `tests/control_interface/operator_interface/test_worker_lifecycle.py` | ~90 | **整合（真子程序）** | 實際 `subprocess` 啟動 worker，測崩潰、重啟冷卻、阻塞寫入、EOF |
| `tests/control_interface/operator_interface/test_site_profile.py` | ~60 | 單元 | 設定驗證，覆蓋最完整的區域 |
| `tests/control_interface/operator_interface/test_route_editor.py` | ~70 | 單元 | |
| `tests/control_interface/operator_interface/test_operator_render_perf.py` | 23 | **系統（需 X display）** | **唯一建構真實 `OperatorApp` 的測試** |
| `tests/control_interface/operator_interface/test_operator_command_safety.py` | ~40 | 混合 | **11 個是原始碼字串斷言，非行為驗證** |
| `tests/control_interface/operator_interface/test_autonomy_gate.py` | **43** | 單元 | 本稽核由 20 增至 43（補 NaN/Inf） |
| `定位演算法/validation/tests/*` | ~130 | 單元 + CUDA 條件 | |
| `tests/localization/flight_control/test_olympe_frame_worker.py` | 16 | 整合 | **唯一同時載入兩份鏡像副本的測試**，但斷言是兩者行為的**交集** |
| `模擬器/parrot_stimulate/tests/*` | 12 檔 / 1357 行 | 整合 | **被 `pytest.ini:17` 排除，不在 1070 內**（見 F-23） |
| `tests/tools/test_*.py` | 19 | 單元 | 打包／驗證工具 |

### 「有多少是真的」量化

| 指標 | 數值 |
|---|---|
| 使用真實執行緒的測試 | ~18 |
| 使用真實子程序的測試 | ~23 |
| 使用 `tmp_path` | 183 |
| 以 `__new__` 繞過建構子 | **46 處**（故多數 `__init__` 未驗證） |
| **原始碼字串斷言（非行為）** | **~24 個**（11 個在 `test_operator_command_safety.py`） |
| 建構真實 `OlympeLiveBackend` | 125 個測試函式中 **125 個**（123 經 `make_backend()` + 2 直接），展開為 157 個收集到的測試。但 `_connect` 被 stub，真實連線路徑只在 4 個參數化案例中執行 |
| 建構真實 `OperatorApp` | 23 個（2%，全部 display-gated） |
| **執行 `path_follow_flight.fly()`** | **0** |
| **執行 `flight_operator_app.main()` 完整流程** | **0**（僅 1 次跑到 argparse `SystemExit`） |

**30 秒跑完 1070 個測試**，代表絕大多數為記憶體內測試。這本身不是缺點
（快速回饋是優點），但需搭配理解：真正的整合覆蓋集中在 backend 與 flight-control，
而 UI 與應用組裝層幾乎沒有行為覆蓋。

---

## 2. 二十項關鍵行為覆蓋檢查

**REAL** = 有真實行為測試；**PARTIAL** = 部分覆蓋且有明確缺口；**NONE** = 無。

| # | 行為 | 判定 | 證據 / 缺口 |
|---|---|---|---|
| 1 | 狀態機轉移 | **REAL** | `test_flight_safety_gates.py:949`（MANUAL 不送）、`:956`（HOVER 送零）、`:962`（LAND 歸零後結束）、`:969`（EMERGENCY 不送 PCMD）、`:177`、`:504`、`:451`；backend 側 `test_olympe_live_backend_safety.py:1367, 2018` |
| 2 | 非法轉移拒絕 | **REAL**（監控層） | `test_flight_safety_gates.py:1559`、`:1595`（terminal 不可解除）、`:1475`（僅允許升級 LAND→EMERGENCY）、`:991`（stale AUTO 拒絕）。**缺口**：`SafetySwitch._apply`（`path_follow_flight.py:352`）本身無 transition table，會接受 EMERGENCY→AUTO；閂鎖只存在於 `SafetyMonitor._latch_terminal`，且無測試單獨驗證 switch |
| 3 | 命令仲裁 | **REAL**（backend）/ **NONE**（UI） | epoch/pulse-token 仲裁 `test_olympe_live_backend_safety.py:1771, 2226, 2242, 2275`。**缺口**：`OperatorApp.send` 只對**同名**命令去重，且 `emergency_stop`／`land_now` 走**不同的同步路徑**（不在 `async_commands` 內），無任何 UI 層跨命令仲裁測試 |
| 4 | Stale command 拒絕 | **PARTIAL** | 有：stale AUTO（`:991`）、nudge deadman（`test_olympe_live_backend_safety.py:2156`）、虛擬搖桿 deadman（`:2965`）。**缺口**：`ControlRequest.submitted_mono_ns` 只檢查正值，從不用於拒絕（見 F-25）；`command_is_fresh` 原語無人呼叫、無測試 |
| 5 | Stale localization 拒絕 | **REAL** | `test_flight_safety_gates.py:504`、`:437`（慢推論不能刷新舊影格的 pose）、`:386`（pose 以擷取時刻標記）、`:311`（時間戳非遞增則 fail closed）；`test_worker_lifecycle.py:1482`；`test_autonomy_gate.py:44` |
| 6 | 手動接管 | **REAL** | `test_flight_safety_gates.py:2448, 2478, 2491, 1337`；`test_olympe_live_backend_safety.py:2397, 2435, 2822, 2005` |
| 7 | Emergency 優先級 | **REAL**（backend）/ **字串斷言**（UI） | `test_flight_safety_gates.py:1530, 1475, 969, 1559`；`test_olympe_live_backend_safety.py:1800, 1771`。**缺口**：UI 層唯一的緊急停止測試是 `test_operator_command_safety.py:125`，斷言字串「緊急停止電腦動作」出現在 `inspect.getsource(_build_ui)` 中 |
| 8 | Watchdog timeout | **REAL**（飛控）/ **NONE**（UI 心跳） | `test_flight_safety_gates.py:1726, 1092, 1642, 1737`；`test_olympe_live_backend_safety.py:2156`。**缺口**：`FailureReason.UI_HEARTBEAT_LOST` 宣告了但**零生產程式碼引用、零測試** |
| 9 | Worker 崩潰偵測 | **REAL**，但**未接上 fail-safe** | 真子程序：`test_worker_lifecycle.py:1219, 1329, 1376, 1440, 1299, 1482`。**缺口**：`FailureReason.WORKER_EXIT`／`WORKER_STALL` 從未被 raise；worker 死亡只降級定位健康度，**不觸發 `fail_safe`**，且無端對端測試 |
| 10 | SDK 例外處理 | **REAL 但窄** | `test_olympe_live_backend_safety.py:2186, 837, 2985, 3004, 3015`；`test_flight_safety_gates.py:1011, 1002, 1975`。**注意**：`_FakeDrone` 多以 `Expectation.success()==False` 模擬失敗，而非**拋出** Olympe 例外，故多數測試走的是「命令被拒絕」而非「SDK 拋例外」 |
| 11 | UI 重複點擊 | **PARTIAL** | `test_operator_command_safety.py:250` 驅動真實 `_dispatch_live_command`，斷言第二次相同點擊回 False；`:237`（失焦清空所有 hold）。**缺口**：去重鍵是命令**名稱**且只涵蓋 `async_commands` 的 12 個名字；重複的 `emergency_stop`、`land_now`、`hover`、nudge 點擊**不去重也無測試** |
| 12 | 設定錯誤處理 | **REAL（最強）** | `test_site_profile.py:86,106,138,196,309,361,373,382,507,538,543`；`test_flight_safety_gates.py:565,616,630,637,1656,2119,2127`；`test_start_anafi_live_launcher.py:151,159` |
| 13 | 啟動到一半失敗 | **PARTIAL** | 有：`test_worker_lifecycle.py:570`（ready handshake 阻擋提交）、`test_site_profile.py:373, 553`、`test_olympe_live_backend_safety.py:914, 1353`。**缺口**：應用層完全未測——`main()`（919 行）只執行一次且在 argparse 就 `SystemExit` |
| 14 | 重複 start/stop | **NONE**（session 生命週期） | `start()` 的 `INTERFACE_MISMATCH`／`HOT_SWITCH_PROHIBITED` 分支**無測試**；`cleanup()` 從未被呼叫兩次；`LiveWorkerClient.close()` 從未被呼叫兩次。**有涵蓋的是 re-arm 週期**（worker 重啟、串流健康恢復、降落失敗重試、lost-hold 重置） |
| 15 | Shutdown 完整性 | **REAL**（backend/worker）/ **字串斷言**（app） | `test_olympe_live_backend_safety.py:2053, 2073, 2099, 2140`；`test_worker_lifecycle.py:1403, 1424`；`test_flight_safety_gates.py:2370`。**缺口**：app 退出路徑只以文字驗證——`test_operator_command_safety.py:430` 斷言 `EXIT_SAFETY_SIGNALS` 集合，`:450` 從 `main` 原始碼切 700 字元檢查字串。**處理器從未被執行** |
| 16 | Queue overflow | **REAL** | `test_worker_lifecycle.py:190`（真實 `LiveWorkerClient._publish_result` 對 `Queue(maxsize=2)`，斷言丟最舊）、`:1160`；`test_olympe_frame_worker.py:104`。**注意**：`_flight_results` 是**無界** queue（F-19），無測試 |
| 17 | 執行緒例外傳播 | **REAL**（飛控）/ **NONE**（UI 派送執行緒） | `test_flight_safety_gates.py:2294, 2311, 2491`；`test_olympe_live_backend_safety.py:2186, 3015, 3051`。**缺口**：`_dispatch_live_command` 的 `run()` 把例外放進 `_flight_results`，但唯一讀該 queue 的測試（`test_operator_command_safety.py:268`）只斷言**成功**的 tuple；錯誤分支與其 UI 呈現從未被執行——**這正是 F-33 未被測試發現的原因** |
| 18 | Map／localization provider 不可用 | **PARTIAL** | `test_site_profile.py:373`；`test_worker_lifecycle.py:371, 271, 275`；`test_autonomy_gate.py:38-46, 61`；`test_flight_safety_gates.py:131`。**缺口**：無「執行期損毀／不可讀的 `.ply`」測試，無「飛行中 worker 始終未 ready」測試 |
| 19 | NaN／Infinity 控制輸入 | **REAL（覆蓋最好）** | `test_flight_safety_gates.py:489, 1880, 630, 1656`；`test_olympe_live_backend_safety.py:2940, 1567`；`backend_contract.py:118-124` 對每個 nudge 軸強制有限性；`test_site_profile.py:507,538,543`；`test_worker_lifecycle.py:245,1058`；**本稽核新增** `test_autonomy_gate.py` 的 21 個雙側非有限案例 |
| 20 | 安全不可被一般命令覆蓋 | **REAL** | `test_olympe_live_backend_safety.py:1771`（阻塞式 handoff 期間 latch 勝過 pc_control，epoch 檢查於 `olympe_live_backend.py:2473-2481`）、`:2005`、`:1480`；`test_flight_safety_gates.py:1559, 1595, 1111, 949` |

### 統計

| 判定 | 數量 |
|---|---|
| REAL | 11 |
| REAL（一層）+ 缺口（另一層） | 4（#3、#7、#8、#17） |
| PARTIAL | 4（#4、#11、#13、#18） |
| **NONE** | **1（#14 重複 start/stop）** |

---

## 3. 缺少的測試 — 依優先級

| 優先級 | 缺少的測試 | 對應 finding | 可自動執行 | 風險 |
|---|---|---|---|---|
| **P1** | `_backend_command` 在背景執行緒上呼叫 `write_log` → 應斷言 UI 更新發生在 Tk 執行緒 | F-33 | 是 | 目前此路徑完全無測試，缺陷因此未被發現 |
| **P1** | `takeoff_cmd()` 回傳 False 時 `ControlResult.accepted` 必須為 False（每個 legacy 分支各一） | F-01 | 是 | 修 F-01 的前置保護測試 |
| **P1** | 焦點在 `Entry`／`Text` 時按方向鍵**不得**產生 `nudge_begin` | F-02 | 是（需 X display） | 修 F-02 的前置保護測試 |
| **P1** | `fly()` 的順序保證改以真實執行驗證（需先抽出 `_fly_sequence`） | F-05 | 是 | 目前為字串斷言，重構會失效 |
| **P1** | `main()` 的 exit-safety 註冊與 `_emergency_cleanup` 實際執行 | F-05 | 是 | 同上 |
| **P2** | `cleanup()` / `LiveWorkerClient.close()` 呼叫兩次必須 idempotent | #14 | 是 | 唯一 NONE 的行為 |
| **P2** | `backend.start()` 的 `INTERFACE_MISMATCH`／`HOT_SWITCH_PROHIBITED` 分支 | #14 | 是 | 模式切換保護未驗證 |
| **P2** | worker 死亡應觸發（或明確**不**觸發）`fail_safe`，端對端 | F-20 / #9 | 是 | `WORKER_EXIT` 是宣告了但不存在的能力 |
| **P2** | `SafetySwitch` 單獨的非法轉移（EMERGENCY→AUTO 應被拒） | #2 | 是 | 閂鎖目前只在 monitor 層 |
| **P2** | `SafetyMonitor` 無 SafetySwitch 時必須預設 HOVER 而非 AUTO | F-09 | 是 | fail-open 預設 |
| **P2** | `emergency_stop`／`land_now` 重複點擊 | #11 | 是（需 display） | 不在去重集合內 |
| **P2** | `_flight_results` 有界性與滿載行為 | F-19 | 是 | 無界 queue |
| **P2** | `olympe_frame_source` 在 UI 程序中必須解析到 flight_control 副本 | F-17 | 是 | `sys.path` 順序脆弱性無保護 |
| **P2** | 執行期損毀 `.ply` / bundle | #18 | 是 | |
| **P3** | `scale_free_control_adapter` 契約測試納入主套件 | F-23 | 是 | 權威控制核心只有 1 個測試 |
| **P3** | 兩份 `olympe_frame_source` 的**差集**行為（單調性守衛） | F-07 | 是 | 現有測試只斷言交集 |

---

## 4. 本次稽核新增的測試

| 檔案 | 新增 | 說明 |
|---|---|---|
| `tests/control_interface/operator_interface/test_autonomy_gate.py` | **+22**（21→**43**） | `test_non_finite_evidence_fails_closed`：7 個欄位 × NaN/+Inf/−Inf = 21 個參數化案例，**雙側**（量測值與其上限）；`test_bool_is_not_accepted_as_a_numeric_reading` |
| `定位演算法/validation/tests/test_runtime_mirrors.py` | **+2**（1→**3**） | `test_every_same_named_file_is_either_enforced_or_declared_divergent`（強制新增的同名檔必須被分類）；`test_unenforced_identical_copies_have_not_drifted`（對 `manual_nudge_pilot.py` 做位元組比對） |

**全套件回歸**：`1070 passed, 1 skipped` → **`1094 passed, 1 skipped`**（+24，零失敗，29.5 秒）。

**非空洞性證明**：`test_every_same_named_file_is_either_enforced_or_declared_divergent`
在加入 `UNENFORCED_BUT_MUST_MATCH` 分類**之前實際失敗**，並正確指出
`manual_nudge_pilot.py`。此即該測試確實在檢查真實條件的證據。

---

## 5. 測試基礎設施缺口

| 項目 | 狀態 |
|---|---|
| 覆蓋率量測 | **無**。無 `.coveragerc`／`pyproject.toml`／`tox.ini`；`coverage`、`pytest-cov` 皆未安裝 |
| 測試逾時 | **無**。`pytest-timeout` 未安裝（本稽核嘗試 `--timeout` 時失敗）。套件內有真子程序測試，一個卡住就會無限阻塞 |
| xfail 標記 | 全庫**零**個 |
| 跳過的測試 | 本機 1 個（`test_xfeat_wrapper_optimizations.py:15`，XFeat hub cache 未隨附） |
| **靜默條件式測試** | **約 43 個**。依賴 display（23）、CUDA + EDM checkpoint（15）、site alignment（1）、真實 olympe（1）、simulator profile（1）。在缺少這些的機器上會靜默跳過，**測試數會少而不報錯** |
| conftest | 僅模擬器專案有一個 `sys.path` shim |

**建議**：加入 `pytest-timeout` 與全域逾時；加入 `coverage` 並對
`olympe_live_backend.py`、`path_follow_flight.py`、`runtime_safety.py` 設最低門檻；
把 43 個靜默條件式測試改為明確 `skipif` 並在 CI 記錄跳過數。

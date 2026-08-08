# audit/final_validation.md — 最終驗收

稽核日期：2026-08-07
基準：`agent/localization-runtime-optimizations` @ `d0b2250`（工作區 dirty）
稽核期間**未連線真機、未起飛、未送出任何飛行命令**（遵守 `控制介面程式/SAFETY.md`）。

---

## 二十二項驗收問題

### 1. 系統是否有清楚的模組邊界？

**大致有，但邊界的「粒度」不一致。**

正面：依賴圖是**無環 DAG**（AST 全掃 4 棵來源樹，0 個 circular import），分層清楚
（UI → 服務 → 領域 → 契約）。`backend_contract.py` 是設計良好的穩定介面層，
`olympe_live_backend` 不 import UI（`DroneState` 以 `state_factory` 注入）。
Olympe 全部延遲 import，模組可在無 SDK 環境成功 import（實測通過）。

負面：**模組間邊界清楚，模組內部沒有邊界**。`OperatorApp` 一個 class 3993 行、
106 方法、144 個實例屬性；`OlympeLiveBackend` 3749 行、93 方法、113 屬性。
邊界存在於檔案之間，不存在於責任之間。

### 2. 系統是否高內聚、低耦合？

**耦合低，內聚差。**

低耦合的證據：無循環相依；SDK 被 adapter 隔離；定位在獨立程序；
模組間以 typed 契約或注入交換。

內聚差的證據：`OperatorApp` 混合至少 10 種責任；`runtime_safety.py` 混合 4 件無關的事
（session 日誌、磁碟保留、網路策略、arming 閘門）——**名稱與實際責任不符**。

一項具體的耦合缺陷：UI **繞過自己宣告的 `OperatorBackend` protocol**，
直接呼叫後端私有成員（`_flight_state_name()`、`pilot_sticks`、`stick_override_count`、
`stream_lost_hover()`），並直接寫入 `backend.state`。

### 3. 系統是否存在過度複雜或過度抽象？

**沒有過度抽象——反而是抽象不足。**

全庫**沒有**發現無用的 wrapper、為套設計模式而生的層、或投機性的可設定性。
`backend_contract.py` 的抽象都有實際使用者。這點值得肯定。

複雜度集中在**單一函式過長**：`_build_ui` 564 行、`tick` 351 行、
`poll` 308 行、`read_connection_inventory` 259 行、`__init__` 279/192 行。

### 4. 系統是否容易新增或替換模組？

| 對象 | 判定 | 依據 |
|---|---|---|
| 定位 provider | **容易** | `production_localizer_factory` + `SFM_LOCALIZER_BACKEND`；worker 是獨立程序，介面是 stdio JSON；EDM 與 XFeat 已實際共存 |
| drone adapter | **容易** | typed 契約 + 模擬／真機 backend 已並存；Olympe 延遲 import |
| 地圖 provider | **中等** | 有 `site_asset_interfaces` 抽象，但 `read_ply_points` 等地圖 I/O **寫在 UI 檔內**（`flight_operator_app.py:2729-2809`），換格式要動 UI |
| UI | **困難** | 業務邏輯與 Tk widget 深度交織，無 ViewModel 層 |

### 5. UI、定位、控制、安全是否合理分離？

**定位與控制分離得好；UI 與安全分離得不夠。**

- 定位：獨立程序，程序邊界即隔離邊界。**分離最徹底的一項。**
- 控制：PCMD 單一送出點，仲裁集中。**正確。**
- 安全：核心閂鎖（`pilot_sticks`、`SafetyMonitor.terminated`）集中且正確，
  但安全**相關**邏輯散落——`runtime_safety.py` 的 arming 閘門、
  `olympe_live_backend` 的 18 道起飛閘門、`path_follow_flight` 的 SafetyMonitor、
  `site_profile` 的資產驗證，分屬四處且無統一的 Safety Supervisor 概念。
- UI：**分離最差**。UI 直接寫 `backend.state`，且在背景執行緒上操作 Tk（F-33）。

### 6. 是否存在多個模組共同修改關鍵狀態？

**是。** `DroneState` 是全系統最主要的共享可變狀態：
`backend.poll()` **以參考回傳**，UI 直接寫入其
`stream`／`mode`／`loc`／`tracker_state`／`pose`／`inliers`，
同時 Olympe callback 執行緒也在修改同一物件。約 80 個欄位、無鎖保護。

**但關鍵的安全狀態不是這樣管理的**：`pilot_sticks` 在 `self._lock` 內修改，
`SafetyMonitor.terminated` 是 `threading.Event`。真正的安全權威有正確的併發保護。

### 7. 是否有單一可靠的系統狀態來源？

**沒有單一來源，但有單一**權威**。**

顯示狀態分散（`DroneState` 的 6 個字串欄位、UI 的 12 個布林、
`OperatorApp` 的約 193 個 `self.*` 屬性），且 UI 會把顯示衍生值寫回 `backend.state`。

但**控制權的權威是唯一的**：`pilot_sticks` 布林。所有安全閂鎖
（串流遺失、磁碟／日誌失效、控制器斷線、緊急停止、鏈路遺失）
都經由 `give_to_pilot()` 收斂到這一個布林，而 `send_pcmd` 在鎖內檢查它。
這是本系統設計最正確的地方。

### 8. 是否有明確且可測試的狀態機？

**沒有明確狀態機。**

- 無 Enum、無 transition table、無非法轉移拒絕邏輯。
- `tracker_state` 有 **26 個相異字串字面值**，散落賦值。
- `DroneState.loc` 的 8 個值**混合三種概念**（來源型別 LIVE/SIM、
  定位狀態 OK/STREAM_LOST/LOST_RECOVERY、MegaLoc 子階段）。
- 三套彼此獨立的 `.mode` 詞彙（控制權、定位品質、自主授權），皆為裸字串。

**但轉移行為本身是可測且已測的**：`test_flight_safety_gates.py` 對
MANUAL/HOVER/LAND/EMERGENCY 各分支做了端對端驗證，terminal latch 不可解除也有測試
（`:1559`、`:1595`）。**缺的是型別與集中管理，不是行為正確性。**

一個具體缺口：`SafetySwitch._apply`（`path_follow_flight.py:352`）本身無 transition table，
會接受 EMERGENCY→AUTO；閂鎖只存在於 `SafetyMonitor._latch_terminal`，
且無測試單獨驗證 switch 這一層。

### 9. 控制命令是否有單一仲裁點？

**是——這是本系統最強的設計。**

`olympe_live_backend.py:2235` 的 `self.drone(PCMD(1, r, p, y, g, 0))` 是**全庫唯一**
實際送出 PCMD 的一行。進入該點只有兩條路：
- `send_pcmd()`：在 `self._lock` 內檢查 `_cleanup_done`／`_landed`／`pilot_sticks`／`_maneuver_in_progress`
- `_zero_pcmd_or_log()`：**刻意繞過**上述閘門——歸零必須永遠送得出去

更進一步：全檔**只有一個非零 `send_pcmd` 呼叫點**（`:3019`，nudge hold loop）；
其餘所有 PCMD 都是字面零。

（自主飛行路徑另有 `SafetyMonitor.send_authorized` 作為其單一仲裁點，
但該路徑目前硬鎖。）

### 10. safety command 是否永遠具有最高優先級？

**是（在目前可執行的路徑上）。**

`fail_safe()` → `give_to_pilot()` → 持鎖設 `pilot_sticks = True` + 歸零 PCMD。
此後 `send_pcmd` 對**所有**命令回傳 False（`:2243-2246`），
且 `auto_resume=False` 明確記錄，必須操作員手動「恢復電腦控制」。

有測試證明無法被覆蓋：`test_olympe_live_backend_safety.py:1771`
（阻塞式 handoff 期間 latch 勝過 pc_control，epoch 檢查於 `:2473-2481`）、
`:2005`、`:1480`；`test_flight_safety_gates.py:1559`、`:1595`、`:1111`。

### 11. 背景 worker 崩潰是否能被主系統偵測？

**定位 worker：能，且有真子程序測試。** `flight_operator_app.py:2482-2484` 偵測，
`test_worker_lifecycle.py:1219/1299/1329/1376/1440` 以真實 `subprocess` 驗證崩潰、
重啟冷卻、阻塞寫入、EOF。

**但偵測後不觸發 fail-safe。** `FailureReason.WORKER_EXIT` / `WORKER_STALL`
在生產程式碼與測試中**皆零引用**；worker 死亡只降級定位健康度顯示。
在手動飛行下這是可接受的（定位是顯示用途），但這是宣告了卻不存在的能力（F-20）。

**執行緒：部分。** `nudge-hold-loop` 與 `SafetyMonitor` 的死亡都有處理與測試。
但 `olympe-ui-<command>` 執行緒的例外分支從未被測試執行——**F-33 正是因此未被發現**。
另外，Tk `tick` 若拋出例外會**永久終止** tick 鏈（無人重新 arm），此路徑無測試。

### 12. 系統是否能可靠啟動、停止與重新啟動？

**啟動與停止：可靠。重新啟動：未驗證。**

停止路徑有多重保險：`_on_close` → `backend.cleanup()`（自述 idempotent）→
`mainloop finally`（cleanup + localizer.close + detector.close + session_logs.close）→
`atexit` → SIGINT/SIGTERM/SIGHUP handler（含 `exit_in_progress` 防重入）。
**訊號註冊失敗會明確警告操作員**（`:7744-7749`）——這是很誠實的設計。

**缺口**：「重複 start/stop」是二十項行為中**唯一完全無測試**的一項。
`cleanup()` 從未被呼叫兩次、`LiveWorkerClient.close()` 從未被呼叫兩次、
`start()` 的 `INTERFACE_MISMATCH`／`HOT_SWITCH_PROHIBITED` 分支無測試。
`os.execv` 熱重啟路徑（切換 site profile）無真機驗證。

### 13. 設定是否集中、驗證且可追蹤？

**驗證良好；集中與可追蹤不足。**

`site_profile.py` 的驗證是本系統的正面範例：拒絕未知鍵、非有限 JSON 常數、
錯誤的 SHA-256 格式，並做 cross-field frame-ID 比對，**全部 fail-closed**。

但設定有**四個獨立來源、無單一 resolver**，其中約 95 個 `SFM_*` 環境變數
**完全繞過**該驗證，且數個可弱化安全上限（`SFM_ALLOW_LEGACY_FLIGHT=1` 繞過 geofence、
`SFM_GATE_WEAK=0` 在弱定位上飛行）。優先序只有一處是顯式的，其餘由載入順序湧現。

啟動時**不會印出最終生效設定，也沒有設定 checksum**。

### 14. 日誌是否足以還原事故前後發生的事情？

**大致足夠，但有一個嚴重缺陷。**

正面：每 session 四條結構化 JSONL（commands／localization／telemetry／incidents），
含 `t_utc` + `t_mono_ns`；命令事件帶 `request_id`、`action`、`human_origin`、
`submitted_mono_ns`、`accepted`、`executed`、`reason_code`、`control_owner`——
這正是 §12 要求的欄位。session manifest 記錄執行環境身分，目錄權限 `0o750`，
原子寫入，且 `_INCIDENT_EVENTS` 會把重大事件另存 incidents 串流。

**缺陷（F-01）**：被拒絕的起飛在 `commands.jsonl` 中被記為
`accepted=true, executed=true, reason_code="OK"`——**與成功的起飛無法區分**。
這直接損害事故調查能力，是 F-01 被評為 P1 的主因。

其他缺口：日誌保留只涵蓋部分檔名；沒有 state-before／state-after 欄位。

### 15. 關鍵模組是否具備 unit test？

**是。** 1094 個測試（本稽核前 1070）。設定驗證、NaN/Inf 處理、
命令契約、pose 型別、鏡像一致性皆有單元測試。
`test_site_profile.py`（~60 測試）與新增後的 `test_autonomy_gate.py`（43 測試）品質高。

### 16. 關鍵路徑是否具備 integration test？

**backend 與 flight-control：是，且品質高。**
`test_olympe_live_backend_safety.py` 的 125 個測試函式（157 個收集到的測試）
**全部建構真實 `OlympeLiveBackend`**（123 個經 `make_backend()`、2 個直接），而非 mock；
`test_flight_safety_gates.py`（149 個測試）有 31 個驅動真實 `run_loop`、
25 個驅動真實 `SafetyMonitor` 執行緒；`test_worker_lifecycle.py` 用真實子程序。
（唯一保留：`make_backend()` 把 `_connect` stub 成 no-op，故真實連線路徑只在 4 個參數化案例中執行。）

**應用組裝層：否。**
`path_follow_flight.fly()`（310 行，唯一的 arming 入口）**從未被執行**；
`flight_operator_app.main()`（919 行）只執行到 argparse `SystemExit`。
兩者的保證以 `src.index()` 對**原始碼文字**斷言——任何保留文字但破壞接線的重構都能維持綠燈。
全庫約 24 個測試屬此類。

### 17. 是否仍有 P0 問題？

**否。** 本次稽核未發現任何 P0。具體而言，未發現：無人機失控路徑、
安全機制可被一般命令覆蓋、emergency 無法搶占、舊控制命令持續發送、
關閉系統後仍持續控制，或影響飛控的 race condition／deadlock。

### 18. 是否仍有 P1 問題？

**是，8 項全部未修**（F-01、F-02、F-03、F-04、F-05、F-06、F-07、F-33）。

未修的原因是刻意的：其中 5 項落在 `控制介面程式/SAFETY.md` 明文要求
「只有操作員明確要求並審過風險才可改」的飛行命令與按鍵路徑上；
另 3 項（環境重建、鏡像所有權、啟動器嚴格度）需要操作員決策。
稽核者提供了逐項 patch 建議與前置測試，見 `refactor_plan.md` §B。

### 19. 是否適合進入模擬測試？

**是。** 模擬入口有完整 preflight（5 個資產 SHA-256、CUDA、GPU 型號、bundle、模型載入、
GUI/worker import）、模式六層強制、離線網路守衛。1094 個測試全綠。
F-02 與 F-33 在模擬模式下不會造成實體風險，反而**模擬模式正是驗證這兩項修正的正確場所**。

### 20. 是否適合進入繫留真機測試？

**有條件適合。** 建議先完成 `refactor_plan.md` §B 的 **B1+B2（命令結果真實性 + 跨執行緒 UI）**
與 **B3（鍵盤焦點守衛）**。

理由：繫留狀態下 F-02 的後果有限（機體被繫住），但**繫留測試的價值在於驗證命令與回報的
正確性**，而 F-01 使「命令是否真的被接受」在 UI 與日誌中都不可信——
這會讓繫留測試的結論本身不可靠。

### 21. 是否適合進入非繫留真機測試？

**尚不適合。** 必須先完成 §B 全部七項（B1–B7），特別是：
- B1+B2：否則被拒絕的命令仍會回報成功，事故紀錄不可信
- B3：否則在輸入框打字會對空中飛機送出運動命令
- B5：否則現場環境與可重建環境不是同一個，驗證結果無法轉移

自主航線飛行另需完成 D3（`fly()` 可測序列抽取），且目前三重鎖不應解除。

### 22. 最大的三個剩餘系統風險是什麼？

**風險一：系統對操作員與對事故紀錄的陳述不可信（F-01 + F-33）。**
被拒絕的起飛／降落／懸停／微移在 UI 與 `commands.jsonl` 中都顯示為成功。
操作員的心智模型與事故重建都建立在錯誤資訊上。更糟的是這兩項互相遮蔽：
修 F-01 而不修 F-33，會把拒絕路徑導入一個會拋 `RuntimeError` 的跨執行緒 Tk 呼叫。

**風險二：非飛行的 UI 互動會產生飛行命令（F-02）。**
22 個方向鍵繫結在 toplevel 上且無焦點守衛，經實測 probe 證實：
在 `Entry`／`Text` 中打字會同時插入字元**並**觸發 nudge。後端只在 landed 時拒絕。
這是本次稽核中最接近實體風險的一項。

**風險三：現場執行環境無法由其自身工具重建（F-04）。**
`.venv` 帶有 `include-system-site-packages = true`，而 `tools/install_runtime.sh:35`
**明文拒絕**這種環境——今天執行安裝腳本會直接失敗。系統套件以不同**主版本**滲入
（pyyaml 5.4.1 vs lock 6.0.3、markupsafe 2.0.1 vs 3.0.3），另有 6 個版本不符、
37 個未鎖套件、1 個指向不存在目錄的 editable 安裝。
後果：所有已完成的飛行驗證都無法保證可轉移到重建後的機器。

---

## 稽核執行的驗證

| 動作 | 結果 |
|---|---|
| `pytest` 全套件（稽核起點） | **1070 passed, 1 skipped, 30.18s** |
| `pytest` 全套件（三項修改後） | **1094 passed, 1 skipped, 29.52s**，零失敗 |
| `ruff --select F821` | 修正前 1 項 → 修正後 **0 項** |
| `bandit -ll`（runtime 40,132 行） | **High 0**、Medium 14、Low 254 |
| `pip-audit`（`.venv`） | 10 個漏洞 / 2 個套件（protobuf 為刻意 pin，setuptools 可升） |
| AST import 圖（4 棵樹） | **0 circular import** |
| Tk bindtag probe（實跑） | 證實 F-02 |
| Tk 跨執行緒 probe（實跑） | `RuntimeError: main thread is not in main loop` — 證實 F-33 |
| NaN/Inf arming probe（實跑） | 18/18 fail-closed |
| `check_runtime_mirrors.py` | 8 對全通過；唯一 authoritative mirror gate |
| `tools/workspace_audit.py` | layout OK；磁碟 **14.5% free（已低於 15% 警告門檻）** |
| `tools/system_validation.py`（15 步，完整跑 3 次） | **14 步通過，步驟 6 `workspace_layout` 失敗**。該失敗**與本稽核無關且為既存**：`--strict-output-names` 拒絕上一次稽核留下的 `outputs/audit_20260805`（見 F-34）。步驟 7 `portable_manifest` 在重新產生 manifest 後通過 |
| `tools/package_manifest.py verify` | root authoritative tool；legacy `執行環境/tools/package_manifest.py` 已移除 |

> **註記（manifest）**：`MANIFEST.tsv` / `SHA256SUMS` 在稽核前為 clean。本稽核修改了
> 3 個檔案並新增 `audit/` 目錄，因此以專案既有流程
> （`.venv/bin/python tools/package_manifest.py generate`）重新產生；目前檔案數量
> 以 authoritative tool 的輸出為準，`verify` 回 exit 0。此動作屬「對這些位元組做
> 完整性背書」，明列於此供操作員複驗。
>
> **註記（驗收閘門）**：`驗證系統.sh` 目前回報 `status=failed`，但**唯一失敗的是
> `workspace_layout`，且原因既存於本稽核之前**（`outputs/audit_20260805` 命名不合規）。
> 其餘 14 步全部通過，包含 `root_pytest`、`runtime_mirrors`、`portable_manifest`、
> `cuda_production_smoke`、`offline_model_smoke`。詳見 F-34。

---

## 只能靜態驗證的項目（無真機）

以下**不得**寫成「已驗證安全」，僅為程式碼與離線測試層級確認：

- Olympe 連線、firmware limit 寫入與回讀
- PDRAW 影像串流中斷後的復原
- RTH／lost-link policy 在真實 GPS 不可靠場域的行為
- `os.execv` + `closerange` 熱重啟能否真正釋放 SkyController socket
- 碰撞監控在真實點雲上的判定（本次僅驗證 scipy 缺席時的降級路徑）
- F-02／F-21 在真機空中的實際運動幅度
- SkyController 實體搖桿接管的端對端時延

## 正式真機測試前仍需完成的驗證

1. `refactor_plan.md` §B 全部七項，各自的前置測試先行
2. 在乾淨重建的環境上重跑 1094 測試 + `驗證系統.sh` 15 步，並與現行環境逐項比對
3. 模擬模式下實測：22 個方向鍵在主視窗仍正常、在輸入框中不再產生 nudge
4. 模擬模式下實測：按「自主」按鈕出現「已拒絕 (LOCKED_EXTERNAL_APPROVAL)」且無 `RuntimeError`
5. 繫留狀態下驗證起飛拒絕路徑：刻意讓一道 preflight 閘門失敗，確認 UI 顯示原因且
   `commands.jsonl` 記為 `accepted=false`

---

# 最終結論

## `APPROVED_WITH_CONDITIONS`

**理由。**

這套系統的**安全核心是正確的**，而且是本次稽核中少見的、明顯高於研究原型水準的工程：
PCMD 有全庫唯一的送出點與集中閘門；emergency 透過 `pilot_sticks` 閂鎖真正搶占且無法被
一般命令覆蓋；依賴圖無環；SDK 被延遲 import 隔離；設定驗證嚴格 fail-closed；
關閉路徑有四重保險且會在註冊失敗時誠實告知操作員；NaN/Inf 是覆蓋最完整的行為之一；
1070 個測試中有真正的整合層（120 個建構真實 backend、31 個驅動真實控制迴圈、
真子程序 worker 測試）。**因此沒有 P0。**

但它**不能以目前狀態進入非繫留真機測試**，原因不是它會失控，而是三件事：

1. 它對操作員與對事故紀錄**說謊**（F-01）——被拒絕的飛行命令回報為成功，
   連 `commands.jsonl` 都寫成 `accepted=true`。一個無法信任自己紀錄的系統，
   其飛行測試結論也無法被信任。
2. 一個**非飛行的 UI 互動會產生飛行命令**（F-02，實測 probe 證實）。
3. 現場環境**無法由其自身工具重建**（F-04）——安裝腳本會拒絕現行 `.venv`，
   使既有驗證結果不可轉移。

這三項都是有界、可定位、有明確修正方案的工程問題，不是設計缺陷。
修正範圍小（B1–B3 合計影響約 6 個函式），且都有可先行建立的保護性測試。

**條件**：`refactor_plan.md` §B 的 B1+B2、B3 完成並通過回歸後，可進行繫留真機測試；
§B 全部七項完成後，可進行非繫留真機測試；
自主航線飛行另需完成 D3，且在此之前三重鎖**不得**解除。

**本結論不因系統能成功啟動而給出。** 系統確實能啟動、1094 個測試全綠、15 步驗證矩陣有 14 步通過（且唯一的失敗是既存的命名治理問題）
——這些都不是通過的理由。給出
`APPROVED_WITH_CONDITIONS` 而非 `NOT_APPROVED` 的理由是：安全核心經實測確認正確，
且所有 P1 都屬「資訊正確性」與「環境可重現性」，而非「控制正確性」。

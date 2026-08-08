# AUDIT_REPORT.md — 定位自動巡航系統技術稽核

稽核日期：2026-08-05
稽核範圍：`/home/allen/localization`（視覺定位 + ANAFI 自動巡航 + 操作介面）
Git branch：`agent/localization-runtime-optimizations`
Git commit：`d0b22509b0766e80c0ed43a10944865d4a24321b`（`Replace scrolling controls with responsive tabs`）
Working tree（稽核開始時）：12 個已修改檔案 + 4 個未追蹤檔案（route editor），非乾淨狀態
稽核期間未連線、未起飛、未對真機送出任何命令。所有飛控測試皆為 mock / dry-run / replay。

---

## 1. Executive Summary

### 系統整體狀態

這是一套**已經過多輪安全稽核、成熟度明顯高於一般研究原型**的系統。飛控核心
（`run_loop` + `SafetyMonitor`）採 fail-closed 設計：每一個例外路徑都收斂到 zero PCMD、
hover、人工接管或降落；PCMD 由獨立 20 Hz 執行緒送出，感知延遲無法餓死命令刷新；
`SafetySwitch` 由單一執行緒輪詢；起飛前有 12 道 fail-closed 閘門。既有測試 820 個，
其中 `test_flight_safety_gates.py` 單檔就有 96 個故障注入測試。

因此本次稽核的重點不是「有沒有基本防護」，而是**防護在哪些條件下會失效**。

### 目前的關鍵前提（影響所有判定）

1. **自動航線飛行目前是硬鎖的。**
   `定位演算法/flight_control/path_follow_flight.py:1710` 的 `fly()` 第一行就
   `raise SystemExit("autonomous route flight is LOCKED pending external approval")`，
   `控制介面程式/mission_pipeline.py:59` 亦有 `AUTONOMOUS_ROUTE_EXTERNAL_APPROVAL_LOCKED = True`。
   → `fly()` 之後的所有程式碼目前是死碼，但只差一行就會復活。
2. **目前唯一能實際命令真機的介面是操作 UI 的 hold-to-move（nudge）手動點動**，
   不是自動巡航。本次唯一確認並修復的 High 缺陷就在這條線上。
3. **GPS 前提與程式碼不一致。** 使用者前提是「無 GPS 或 GPS 不可靠」，但
   `olympe_live_backend.py:1062` 把 `lost-link RTH policy` 列入起飛必要條件，
   而該 policy 要求 `home_reachability == reachable`（需 GPS home point）。
   → 在真正 GPS-denied 的場域，目前**根本無法通過起飛閘門**（fail-closed，安全，
   但代表目標情境尚未被支援）。

### 問題數量

計數方式：12 面向掃描產出 60 個候選 → 對抗式複驗 **44 CONFIRMED / 16 REFUTED**；
44 項中 runtime-safety latch 被三個面向各自獨立發現（合併為 1），得 42 個相異項；
再加上稽核者本人獨立確認的 7 項，合計 **49 個相異已確認缺陷**。

| 嚴重度 | 數量 | 已修 | 未修 |
|---|---|---|---|
| **Critical** | **0** | – | – |
| **High** | **15** | **14** | 1 |
| **Medium** | **27** | **12** | 15 |
| **Low** | **7** | 0 | 7 |
| 合計 | **49** | **26** | 23 |

> 第三輪（真機已連線，使用者指示「把要修的部分全部修掉」）完成後更新。
> **唯一未修的 High 是 A04**（inlier 空間分布閘門），因為它需要三場域校準才能安全啟用。
> `production_edm_tracker.py:787`（LOST 重定位繞過跳變／偏航閘門）原被複驗者從
> Critical 下修為 High，唯一理由是 `fly()` 目前硬鎖——**該項現已修正**，
> 因此解鎖 `--fly` 時不再是 Critical。剩餘 5 個未修 High：F04、F05、F06、F13、A04。

### 判定

- **適合進行模擬 / replay 測試：是（GO）。** 已實測：821 測試通過、13 項故障注入 12 項安全、
  400 幀真實 replay p99 26.8 ms、定位成功率 97.5%。
- **適合接上實機：CONDITIONAL GO**，且僅限「操作員手動點動（nudge）＋ 定位只用於顯示」，
  並以 §8 的前置條件為準。此判定的依據是：真正能命令真機的那條路徑上，
  本次找到的 2 個 High 缺陷（A01 deadman、A02 safety latch）**已修並附回歸測試**。
- **自動巡航 `--fly`：仍為 NO-GO，但風險已明顯下降。** 直接影響「交給控制器的姿態是否
  可信」的三項（F01 worker 回應錯配、F02 共享記憶體撕裂讀、F11 LOST 繞過跳變／偏航閘門）
  **均已修正並附回歸測試**。維持外部審核鎖的理由改為：剩餘 5 個未修 High
  （F04、F05、F06、F13、A04）尚未處理，且所有修正都只在 mock／replay 驗證過，未經實機。

> 依使用者訂定的規則——「除非所有 Critical 與 High 飛安問題都已被實際驗證，
> 否則不得判定為 GO」——目前尚有 5 個未修 High，故**仍不得判定為 GO**。

---

## 2. Architecture

### 2.1 實際資料流（以程式碼為準）

```text
ANAFI PDRAW RGB frame  (olympe_frame_source.OlympePdrawGrabber)
   │  每幀帶 capture timestamp；FROZEN_DUP_FRAMES 凍結偵測；is_healthy() 供控制迴圈查詢
   ▼
shared-memory frame → live_localizer_worker（獨立行程，GPU）
   ▼
MegaLoc 全域檢索（BOOT / LOST）  ──┐
   ▼                              │
EDM 局部匹配（TRACK / WEAK）       │  production_edm_tracker.localize()
   ▼                              │
2D–3D correspondences  ───────────┘
   ▼
pycolmap absolute-pose PnP (LO-RANSAC)
   ▼
品質閘門 pose_passes_quality()  ← inliers >= min_inl AND reproj_rms <= limit
   ▼
軌跡閘門：max_jump（硬）+ adaptive_jump（需連續確認）
   ▼
edm_localizer_adapter → Pose(x,y,z,yaw,stamp)
   ▼
run_loop 二次閘門：finite 檢查 / POSE_STALE_S / MAX_POSE_JUMP_U / LOW_CONF_INLIERS / GATE_WEAK
   ▼
HeadingEstimator（Olympe yaw + 地圖運動融合）
   ▼
RouteAutoController.step() → Command
   ▼
YawAlignedPcmdController.update() → (roll,pitch,yaw,gaz)
   ▼
MAX_ROUTE_DEVIATION_U 航道閘門
   ▼
SafetyMonitor.send_authorized()（單一 _io_lock 權威）
   ▼
獨立 20 Hz sender thread → drone(PCMD(...))
```

### 2.2 缺少的階段

| 使用者列出的階段 | 實際狀態 |
|---|---|
| 全域影像檢索 | 有（MegaLoc） |
| 局部特徵與匹配 | 有（EDM；XFeat 程式碼保留但資產已刪） |
| 2D-3D 對應 | 有 |
| PnP / LO-RANSAC | 有（pycolmap） |
| Pose refinement | 有（tracker 內部） |
| **EKF / 濾波器** | **無傳統 EKF。** 採「拒絕優於平滑」策略：max_jump + adaptive_jump 閘門 + 速度預測，沒有共變異數 |
| **安全區 polygon / geofence** | **生產路徑無 polygon geofence。** 只有 (a) `MAX_ROUTE_DEVIATION_U=3.0` 航道半徑（scale-free 單位）、(b) 韌體 MaxAltitude/MaxDistance（需 GPS）。`cruise_geofence.SafeZone2p5D` 被 `SFM_ALLOW_LEGACY_FLIGHT` 鎖住 |
| 路徑規劃 | 有（`plan_path.py`，離線） |
| 任務狀態機 | 有，但分散在三處（見 §2.3） |
| watchdog / heartbeat / manual override | 有（SafetyMonitor） |
| hover / landing / emergency fallback | 有 |

**沒有共變異數，就沒有「距離邊界小於定位誤差時視為風險」的能力**——這是本系統最大的
結構性缺口（見 §3 geofence 相關 finding）。

### 2.3 控制狀態機

實際上有三個獨立狀態機，**沒有統一的 INIT→…→FINISHED 主狀態機**：

**(a) 安全模式（`SafetySwitch` / `SafetyMonitor.mode`）** — 真正的權威
```
AUTO ──► HOVER ──► AUTO
  │        │
  ├──────► MANUAL   （自動端完全不送命令，交還搖桿）
  ├──────► LAND     （latched terminal，送 zero + Landing）
  └──────► EMERGENCY（latched terminal，切馬達）
```
LAND / EMERGENCY 由 `_latch_terminal()` 上鎖，**不可逆**——正確設計。

**(b) 定位狀態（`production_edm_tracker`）**
```
BOOT ──► TRACK ⇄ WEAK ──► LOST ──► BOOT(重新檢索)
```

**(c) 航線狀態（`RouteAutoController.state`）**
```
CRUISE ⇄ INSPECT ──► LANDING
   ▲                    │
   └── HOVER(pose stale)┘
```

**發現的狀態機問題**（詳見 §3）：
- `RouteAutoController` 的 `"DONE"` 狀態**從未被指派**（`path_follow_flight.py:1544` 卻在檢查它）。
- `RouteAutoController.step()` **沒有終端 latch**：`LANDING` 之後若再被呼叫，
  `line 629` 會把 state 改回 `CRUISE`。目前安全**完全依賴呼叫端**在 `line 1544` 立即 `break`。

---


## 3. Findings

稽核方法：12 個獨立面向的深度掃描（72 個 agent、2542 次工具呼叫）＋ 對抗式驗證
（每個 finding 由另一個 agent 以「預設為 REFUTED，除非親眼在程式碼中確認」的立場複驗）。
**60 個候選 finding 中，44 個 CONFIRMED、16 個 REFUTED**（refuted 的多為「上游已有 guard」
或「只存在於 legacy/研究路徑」）。以下另加稽核者本人獨立確認的 8 項。

嚴重度已由複驗者依「是否真的可達生產路徑」下修。**原本被標為 Critical 的
`production_edm_tracker.py:787` 已下修為 High，理由是 `fly()` 目前硬鎖**——這個判斷是誠實的，
但也意味著解鎖 `--fly` 的當下它就會變成 Critical。

### 3.0 稽核者獨立確認並已修正的項目

#### A01 · High · nudge hold loop 的例外會癱瘓 deadman，飛機續飛在最後一個非零 PCMD 上 —— 【已修】

```text
ID:              A01
Severity:        High
Status:          Fixed（附回歸測試，先驗證「不修就失敗」）
Component:       操作 UI 真機後端 / hold-to-move 手動點動
File:            控制介面程式/operator_interface/olympe_live_backend.py
Line:            2533（_ensure_nudge_loop._loop）
```

**Description**：`_loop()` 的「離開迴圈 → 送 zero PCMD → HOVER」收尾**不在 `finally` 內**，
而 deadman 逾時判定（`_nudge_deadline`）**寫在迴圈本體裡**。因此迴圈本體任何例外
（`self.drone(PCMD(...))` 在鏈路異常時拋出、`log.event` 在磁碟滿時拋出、
`_raw_pcmd` 每次呼叫都做的 `from olympe.messages... import PCMD` 在記憶體壓力下失敗）
都會讓執行緒直接死亡，**收尾程式碼被跳過**。

**Trigger**：操作員按住方向鍵不放時，PCMD 送出路徑發生任一例外。

**Impact**：`_nudge_held` 仍保留該方向，但已無人服務它——沒有 PCMD 刷新、
**沒有 zero、deadman 也隨執行緒一起死了**。ANAFI 會維持最後一個非零 PCMD 繼續移動。
唯一的恢復是操作員自己鬆手（`nudge_end` 有獨立的 zero，見下）或按別的鍵。
換言之：**本來要在 `nudge_pulse_s` 後自動停住的保護被完全解除，暴露時間變成人類反應時間。**

**Evidence**（修正前）：
```python
2535    while not self._nudge_loop_stop.is_set():
2540        if held and time.monotonic() >= self._nudge_deadline:   # deadman 在迴圈內
2547        self.send_pcmd(*pcmd, reason="nudge_hold:" + ...)        # 可拋出
2550    # Loop exit: zero if nothing held (release → hover)          # 不在 finally
2553    if empty and not self.pilot_sticks and ...:
2555        self._raw_pcmd(0, 0, 0, 0)
```

**Reproduction**（實測，無真機）：讓 `_FakeDrone.__call__` 在第 2 個非零 PCMD 時
`raise ConnectionError`，按住「前」。修正前結果：執行緒帶著未處理的 `ConnectionError` 死亡、
`_nudge_held == {'前'}`、`tracker_state != "HOVER"`、drone 收到的最後一個 PCMD 是非零。

**Fix**：迴圈本體包 `try`，例外時清空 `_nudge_held` 並記錄 `nudge_loop_error`；
把 zero／HOVER 收尾移進 `finally`；zero 本身若也失敗，改記 `pcmd_zero_failed` 而非 `pass`。

**Verification Test**：`test_olympe_live_backend_safety.py::test_hold_loop_error_drops_the_hold_and_hovers`
——已驗證移除修正後該測試失敗、加回後通過。

---

#### A02 · High · 一次失敗的自動安全動作會永久停用電量／高度／距離保護 —— 【已修】

```text
ID:              A02
Severity:        High
Status:          Fixed（三個獨立面向各自發現同一缺陷，見 F03／F12／F14）
Component:       操作 UI 真機後端 / runtime flight safety
File:            控制介面程式/operator_interface/olympe_live_backend.py
Line:            3149（_execute_runtime_safety_action）／3165（_schedule_runtime_safety_action）
```

**Description**：`_runtime_safety_action_latched` 在排程安全動作時設為 True（3165），
**全檔沒有任何一處把它設回 False**（僅 642 行初始化）。`_evaluate_runtime_safety()`
在 latched 時直接 return False。latch 的原意是「成功的動作不要重複發」——
既有測試 `test_critical_battery_requests_rth_once_when_home_is_reachable` 正是在鎖這個語意。
**缺陷在於動作失敗時也照樣上鎖**：`landed = self.land_cmd(...)` 的回傳值只被寫進 log
（`ok=landed`）就被丟棄。

**Trigger**：電量臨界／超高度／超距離觸發 → 嘗試 RTH（無 GPS home 時失敗）→ 落地未確認。

**Impact**：飛機仍在空中，但該 session 之後**所有**自動安全判定（critical battery、
max altitude、max distance）永久不再評估。只剩操作員手動介入。

**Fix**：動作失敗時解除 latch 並記錄 `runtime_safety_rearmed`，讓下一次輪詢重試；
成功路徑（RTH 成功、已落地）維持原本上鎖語意，既有測試不受影響。

**Verification Test**：`test_olympe_live_backend_safety.py::test_failed_safety_landing_rearms_instead_of_disabling_protection`
——已驗證移除修正後失敗、加回後通過。

---

#### A03 · High · `SFM_GATE_WEAK` 弱定位懸停閘門在生產 EDM 後端上完全失效 —— 【已修】

```text
ID:              A03
Severity:        High
Status:          Fixed（另有 2 個 agent 獨立發現，見 M-list）
Component:       定位 adapter → 飛控 WEAK 閘門
File:            定位演算法/deploy_code/sfm_glomap_deploy/edm_localizer_adapter.py
Line:            187（EDMTrackerAdapter.localize_frame 的 _last_info）
```

**Description**：`path_follow_flight.fly()` 這樣接線（1919 行）：
```python
pose_is_weak=lambda: bool(dict(loc.last_info).get("weak", False)),
```
但 **EDM adapter 的 `_last_info` 從來沒有 `"weak"` 這個 key**（XFeat tracker 有：
`production_xfeat_tracker.py:2037` `info["weak"] = bool(weak)`）。
因此 `pose_is_weak()` 恆為 False，`run_loop:1436-1438` 的 `GATE_WEAK` 分支永遠不成立。
記憶中被當作安全開關的 `SFM_GATE_WEAK`（預設開啟 →「WEAK fix 一律懸停」）在
生產 EDM 後端上**是死的**。

更關鍵的一點：EDM tracker 成功時**一律**寫 `state_out = "TRACK"`（916 行），
只有 miss 才寫真實狀態（934 行）。所以「這個 fix 是從退化狀態贏來的」這件事，
唯一的紀錄是 `state_in`。

**Impact**：在 `WEAK_TRACK` 狀態下以 `weak_min_inliers=30` 接受的 fix，
以及在 `LOST` 狀態下**跳過軌跡跳變閘門**（見 F09）接受的 re-acquisition，
都會被下游當成正常 TRACK。只剩 `LOW_CONF_INLIERS=60` 這一道；inliers ≥ 60 的
弱／重定位 fix 會直接驅動無人機。

**Fix**：adapter 補上 `"weak": info["state_in"] in ("WEAK_TRACK", "LOST")`。
單行、非鏡像檔、只會讓系統更保守（退化狀態的 fix 先懸停，下一幀 `state_in` 即回到 TRACK）。
此修正同時對 F09（LOST 繞過跳變閘門）提供一層縱深防禦。

**Verification Test**：`test_edm_tracker_quality.py::test_adapter_reports_degraded_state_fixes_as_weak_for_the_flight_gate`
——已驗證移除修正後 `KeyError`、加回後通過。

---

### 3.0.1 稽核者獨立確認、**未修**的項目

| ID | Sev | 位置 | 問題 | 為何不修 |
|---|---|---|---|---|
| A04 | High | `production_edm_tracker.py:367,651` `reprojection_metrics`／`pose_passes_quality` | **inlier 空間分布完全沒有閘門**。`inlier_ratio` 與 `inlier_grid_cells`（8×8 網格佔用）每幀都算，但只流向 telemetry／UI（`edm_localizer_adapter.py:192-193`、`live_localizer_worker.py:858-859`、`flight_operator_app.py:4506-4507`），**從未與任何門檻比較**。`min_inlier_ratio=0.35` 只存在於實驗性 NeuFlow tracker（`live_localizer_worker.py:552`），不在生產 EDM 路徑。→ 使用者明列的「PnP inlier 很多但集中在畫面小區域」目前**不會被拒絕** | 這會改變定位接受行為。**已為其備妥校準證據**：本次 390 幀健康 replay 的 `inliers/n_corr` 分布為 p1=0.632、**min=0.571**，即 `inlier_ratio >= 0.35` 的閘門在此語料上**誤拒 0 幀**（餘裕 1.63×）。但這只是**一個場域一支影片**；river_site 與 football_field 未驗證。依稽核原則「無法安全自動修正者只寫入報告」，不擅自加入未校準的飛安門檻 |
| A05 | Medium | `path_follow_flight.py:1263,1264,1437,1438,1453,1461` `run_loop` | **6 個 LoopHooks 回呼沒有例外保護**（`loop_beat`、`safety_poll`、`pose_is_weak`、`pose_confidence`、`force_relocalize`、`request_manual`），對照 `get_pose`／`olympe_yaw`／`stream_healthy`／`emit` 都有包。任一拋出即中止 `run_loop`。故障注入 #13 實測重現 | 後果其實是 fail-safe：`fly()` 的 `finally`（1949）仍會降落。且生產接線下這些回呼都是 trivial（`monitor.beat`、`lambda: monitor.mode`、`setattr`）。修改需同步兩份鏡像檔，風險大於收益 |
| A06 | Medium | `real_path_follow_controller.py:579-629` `RouteAutoController.step` | **狀態機沒有終端 latch**：`state` 進入 `"LANDING"`（613／618）後若再被呼叫一次，629 行會把它改回 `"CRUISE"`。安全**完全依賴呼叫端** `path_follow_flight.py:1544` 立即 `break` | 目前不可達（呼叫端保證 break）。屬縱深防禦缺口，改動控制器語意需重新驗證整條航線行為 |
| A07 | Low | `real_path_follow_controller.py` | `"DONE"` 狀態**從未被指派**（`self.state =` 只出現 CRUISE／HOVER／INSPECT／LANDING），但 `path_follow_flight.py:1544` 在檢查它。`DONE` 屬於另一個控制器 `autoflight.py` | 死狀態檢查，無害 |
| A08 | Low | `benchmark_edm_site_replay.py:429-431` | receipt 在未給 `--quality-baseline` 時輸出 `"enabled": false` 但同時 `"passed": true`。人工審閱 receipt 時極易誤讀為「品質已驗證」 | 無任何自動消費者讀取此欄位；改動 schema 型別的風險大於收益。建議改為 `null` |
| A09 | Low | `test_operator_render_perf.py`（失敗中） | 最小視窗 980×640 下，「操作與定位」分頁有一個 label 被裁掉 19 px。該 label 的內容正是**人工接管按鍵說明**：`空白鍵=全部懸停`、`Esc=交回搖桿`。按鍵功能本身正常，但操作員在最小視窗下讀不到接管方式 | 屬未提交的 UI 改動（`Replace scrolling controls with responsive tabs`），是使用者進行中的工作，不代為改動版面 |
| A10 | Medium | `test_olympe_live_backend_safety.py:717` | `test_connect_keeps_skycontroller_sticks_until_explicit_pc_request[True-True-STICKS]` **直接讀取主機真實硬體**：`find_skycontroller_joystick_path()` 掃 `/dev/input/by-id/usb-Parrot*Skycontroller*-joystick` 與 `/dev/input/js*`。本機目前無 js 裝置 → `STICK_MONITOR_FAIL != STICKS`。**已在 pristine `git HEAD` 上重現，與本次修改無關**。稽核開始時它通過、現在失敗 → 測試結果取決於當下有沒有插搖桿 | 屬測試可重現性缺陷，修正方式（mock 掉裝置探測）會動到既有測試語意，交由團隊決定 |

> A10 的生產端行為是 fail-closed 的（無搖桿 → `STICK_MONITOR_FAIL` → `_takeoff_preflight`
> 以 "takeoff requires a healthy SkyController stick monitor" 擋下起飛），所以這是
> **可重現性問題，不是飛安漏洞**；但它讓「測試全綠」不再是可信的發布訊號。

---

### 3.1 對抗式複驗確認的 High findings（14 項，其中 F03／F12／F14 為同一缺陷的三次獨立發現，已於 A02 修正）

> 以下內容由複驗 agent 產出並經其逐行覆核；稽核者對 F03／F12／F14（runtime-safety latch）、
> F11（LOST 繞過跳變閘門）已親自覆核程式碼確認。其餘 High 項目**未經稽核者本人逐一覆核**，
> 標示為 Confirmed 係指通過對抗式複驗，交付時請以標註的檔案行號自行覆查。


### F01 · High · 【已修 · 第二輪 #2】· Localizer worker responses are paired with the wrong request when a stall's restart is refused by the 30 s cooldown, so an old frame's pose is published as the newest frame's fix

- **Component**: camera-ipc / worker IPC (operator localizer client)
- **File**: `控制介面程式/operator_interface/flight_operator_app.py`  **Line**: 2215  **Symbol**: `LiveWorkerClient._loop / LiveWorkerClient._restart`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Two localization calls exceeding SFM_WORKER_TIMEOUT_S (default 8.0 s, flight_operator_app.py:1849) within the same 30 s window — e.g. a GPU hiccup, CUDA re-init, page-cache/swap pressure, or a slow LOST/MegaLoc recovery. The first timeout consumes the restart budget (`self._last_restart = now`, line 2062); the second timeout hits the cooldown branch and leaves the stalled worker attached.
- **Impact**: A pose computed from a frame that is at least SFM_WORKER_TIMEOUT_S (8 s) old is published to the operator carrying the CURRENT frame's identity and timing. `_attach_client_timing(payload, timing)` (line 2206) copies the new request's `source_frame_stamp_mono` onto the old payload, and `update_live_results` feeds exactly that stamp into the pose smoother (`stamp = result.get("source_frame_stamp_mono", ...)`, flight_operator_app.py:5767) and then into `self.live_pose` (5841-5844). Consequences: the operator's map/HUD shows a position the aircraft left seconds ago while the `pose_age` readout …
- **Description**: The client/worker protocol is strictly synchronous: `_loop` writes one request (control prefix + frame or shm slot byte) and then reads exactly one response line, pairing it with the request it just sent. There is no request id in the protocol: `live_localizer_worker.main()` emits its own `"seq"` counter but the client overwrites the identity with `payload["display_seq"] = seq` / `payload["frame_name"] = frame_name` (flight_operator_app.py:2204-2205) and never compares the worker's `seq` against its own (grep for `\bseq\b` in flight_operator_app.py returns only the client's own local variable at 2078/2080/2185/2204/2247/2268/2296/2319 — no comparison). The synchronous invariant is enforced only by `_restart()` killing the stalled worker …
- **Reproduction**: Read-only, no drone. /tmp/claude-1000/-home-allen/fe9d2d50-0406-4e95-98fb-b72138470034/scratchpad/repro_desync.py builds a real `app.LiveWorkerClient` (timeout_s=0.3) against a stub worker that echoes the frame bytes it read, stalls 1.5 s on its first frame, and sets `client._last_restart = time.monotonic()` to force the cooldown branch. Run with /home/allen/localization/.venv/bin/python from …
- **Recommended fix**: Two minimal, independent changes: (1) put a request id in the protocol — have `live_localizer_protocol.encode_request` already carry the capture stamp, so make `live_localizer_worker.main()` echo the `capture_stamp` it used (and/or an opaque uint32 request id) in the payload, and in `LiveWorkerClient._loop` discard/flag any response whose id does not match the request just written instead of blindly stamping it with `display_seq`/`frame_name`/timing. (2) In `_loop`'s `except` branch, when `self._restart()` returns …
- **Verification test**: Add to 控制介面程式/operator_interface/test_worker_lifecycle.py a test mirroring `test_partial_worker_response_times_out_and_recovers`, but with `client._last_restart = time.monotonic()` set before the first submit and a stub worker that echoes the frame byte it read: assert that every published payload with `success`/`worker_saw` matches the `frame_name` whose bytes produced it (i.e. no payload is published whose content …
- **Verifier**: I read every line cited and then reproduced the defect with a local mock worker (no drone, no network, no edits). CODE VERIFIED VERBATIM - flight_operator_app.py:2051-2062 — `_restart()` takes `_proc_lock`, then `if now - self._last_restart < 30.0: return False` at 2060-2061, i.e. it returns WITHOUT calling …

### F02 · High · 【已修 · 第二輪 #3】· After the response desync the parent overwrites the shared-memory frame slot the worker is currently reading — no lock, seqlock or generation counter protects the 720p slot

- **Component**: camera-ipc / worker IPC (shared-memory frame transport)
- **File**: `控制介面程式/operator_interface/live_localizer_worker.py`  **Line**: 750  **Symbol**: `main (shared-memory frame path) / LiveWorkerClient.submit`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Same trigger as the desync: a second >SFM_WORKER_TIMEOUT_S localization stall inside the 30 s restart cooldown, with SFM_SHARED_FRAMES unset/1 (the production default, flight_operator_app.py:2453-2454). After the pairing slips, the collision window is the worker's read+colour-convert at the start of every subsequent frame, against parent writes that happen at arbitrary UI ticks.
- **Impact**: The localizer runs PnP on an image that is half one camera frame and half another, taken at different times and different aircraft attitudes. There is no detection of this: no per-slot generation counter, no checksum, no seqlock — the worker cannot tell a torn frame from a good one, and the quality gates see a self-consistent-looking image whose two halves disagree geometrically. The outcome is either a spurious localization failure or, worse, a pose fitted to the dominant half and reported as a normal fix to the operator's map. It also silently corrupts the frame identity: the pose is …
- **Description**: The shared-memory transport has two slots (`self._frame_shm_slots = 2`, flight_operator_app.py:1860) and no synchronisation primitive at all. Safety rests entirely on the comment at live_localizer_worker.py:750-751 ("The parent never overwrites the active slot until this synchronous request returns"), implemented by `submit()` choosing `slot = 1 - self._active_shm_slot` while `in_flight` (flight_operator_app.py:2289-2295) and `_loop`'s `finally` advancing `_active_shm_slot` to the coalesced item after the response arrives (2227-2233). `_active_shm_slot` tracks what the CLIENT believes the worker is reading. Once request/response pairing slips by one (see the cooldown-refused-restart finding), the client's parity is inverted relative to the …
- **Reproduction**: Read-only, no drone. /tmp/claude-1000/-home-allen/fe9d2d50-0406-4e95-98fb-b72138470034/scratchpad/repro_shm_torn.py builds a real `app.LiveWorkerClient(..., use_shared_frames=True)` against a stub worker that reads the slot byte, samples the slot's first byte, holds the slot for 0.25 s (standing in for read+cvtColor) and re-samples. Frame i is filled with the constant byte i, and `client._last_restart = …
- **Recommended fix**: Make slot ownership follow the worker, not the client's optimistic bookkeeping. Minimal fix: add a per-slot generation counter in the shared segment (or 8 extra bytes per slot); the client writes `gen` after the pixel copy, sends it alongside the slot byte, and the worker re-reads `gen` immediately after `frame_view_from_shared_memory` and again after it has copied the frame (the EDM adapter's cvtColor), rejecting the frame with an explicit `"error": "torn_frame"` when it changed. Cheaper alternative that also …
- **Verification test**: Add to 控制介面程式/operator_interface/test_worker_lifecycle.py, next to `test_shared_frame_worker_coalesces_latest_slot_and_notifies`, a test using a stub worker that samples its slot at read time and again after a hold, with `client._last_restart = time.monotonic()` and a first-frame stall longer than `timeout_s`: assert every published result reports `torn is False` (or, with the generation-counter fix, that the worker …
- **Verifier**: All cited code exists verbatim. live_localizer_worker.py:741-747 reads a 1-byte slot id then `frame = frame_view_from_shared_memory(frame_shm.buf, slot_byte[0], ...)`; :157-166 returns `np.frombuffer(buffer[start:start+frame_size])` — zero-copy over `frame_shm.buf`; :750-751 is exactly the comment "The parent never …

### F03 · High · 【已修，見 A02】· Runtime safety latch is one-shot and never cleared: an unconfirmed landing permanently disables every automatic safety action for the rest of the flight

- **Component**: concurrency / 控制介面程式/operator_interface (OlympeLiveBackend runtime safety)
- **File**: `控制介面程式/operator_interface/olympe_live_backend.py`  **Line**: 3149  **Symbol**: `OlympeLiveBackend._execute_runtime_safety_action / _schedule_runtime_safety_action / _evaluate_runtime_safety`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Airborne, `state.battery_pct <= _CRITICAL_BATTERY_PCT` (or altitude/distance limit), home not reachable (normal indoors/GPS-denied), and the Landing does not report FlyingStateChanged(state="landed") within 22 s — e.g. touchdown on a surface the ultrasound sensor misreads, the drone snagged on structure, or the state event dropped on a degraded link.
- **Impact**: The aircraft is left airborne at critical battery with `state.tracker_state = "LAND_UNCONFIRMED"` and the host will never re-issue Landing or RTH again, for that or ANY later trigger (geofence breach, altitude limit, controller loss). All host-side runtime geofence/battery protection is silently dead for the rest of the flight; only the aircraft's own firmware failsafe remains.
- **Description**: `_schedule_runtime_safety_action` sets `self._runtime_safety_action_latched = True` (line 3165) BEFORE the background `runtime-flight-safety` thread has done anything, and the latch is never cleared anywhere in the file (`grep -n "_runtime_safety_action_latched"` returns only 642 init-False, 3160 read, 3165 set-True, 3183 read). The thread target `_execute_runtime_safety_action` computes `landed = self.land_cmd(...)` at 3149, logs `ok=landed` at 3150-3155, and then returns unconditionally — the False result is recorded but never acted on. `_land_and_confirm` returns False whenever the `FlyingStateChanged(state="landed")` confirmation does not arrive inside `exp.wait(_timeout=22)` (lines 2139-2140, 2162-2168), and `land_cmd` propagates that …
- **Reproduction**: Extend the existing pytest fixture in 控制介面程式/operator_interface/test_olympe_live_backend_safety.py: `backend = make_backend(); backend.drone.flight_state = "flying"; backend.drone.home_reachable = False; backend.drone.landing_success = False; backend.state.flight_state = "flying"; backend.state.battery_pct = 9.0`. Then `assert backend._evaluate_runtime_safety(); …
- **Recommended fix**: In `_execute_runtime_safety_action`, wrap the whole body in try/except and clear/re-arm the latch when the action did not succeed, e.g. after line 3149: `if not landed: with self._runtime_safety_action_lock: self._runtime_safety_action_latched = False; self._runtime_safety_action_thread = None` and log a `runtime_safety_retry` event, so the next `poll()` re-evaluates and re-issues. Add a bounded retry counter (e.g. 3 attempts) that escalates to `Emergency()`/persistent LAND rather than giving up silently.
- **Verification test**: test_unconfirmed_runtime_safety_landing_is_retried: build a backend flying at 9% battery with home_reachable=False and landing_success=False; assert _evaluate_runtime_safety() schedules and the thread runs; assert the latch is released and a second _evaluate_runtime_safety() call issues another Landing; then set landing_success=True and assert the latch stays set after a confirmed touchdown.
- **Verifier**: Every claimed line is verbatim correct. 3139-3155 `_execute_runtime_safety_action` ends with `landed = self.land_cmd(reason=f"runtime_safety:{reason}")` (3149) then `self.log.event("runtime_safety", ..., ok=landed)` (3150-3155) and returns — the False is logged and discarded, no retry, no un-latch. 3157-3178 …

### F04 · High · 【已修 · 第三輪】· SafetyMonitor swallows every PCMD write failure with no counter and stamps last_pcmd_call_mono_ns before the attempt, so a dead command channel is invisible and the flight log records commands that never reached the wire

- **Component**: concurrency / 定位演算法/flight_control (production flight entry, SafetyMonitor)
- **File**: `定位演算法/flight_control/path_follow_flight.py`  **Line**: 1181  **Symbol**: `SafetyMonitor._run / SafetyMonitor.send_authorized / SafetyMonitor.pcmd_timing_snapshot`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: The ARSDK command channel fails while the pdraw/RTSP video stream keeps delivering fresh frames — Olympe scheduler/expectation error, drone object entering a disconnected state, or any exception out of `drone(PCMD(1, r, p, y, g, 0))` in the `send_pcmd` closure defined at path_follow_flight.py:1799-1802.
- **Impact**: The autonomy believes it is commanding the aircraft at 20 Hz. `run_loop` keeps advancing waypoints, `emit()` writes `"blocked": false` with a fresh `pcmd_call_mono_ns` for every tick, no watchdog fires (the loop is beating), and no LAND is latched. The aircraft is actually uncommanded and flies on its last accepted PCMD / firmware link-loss policy while the operator's console and the post-flight command log both show a nominal mission. Post-incident evidence is actively wrong: the log asserts a wire timestamp for commands that never left the host.
- **Description**: All nine `self._send(...)` call sites in SafetyMonitor are of the form `try: self.last_pcmd_call_mono_ns = time.monotonic_ns(); self._send(...) except Exception: pass` (lines 928-933, 951-955, 965-970, 1109-1114, 1119-1123, 1128-1132, 1138-1142, 1147-1151, 1179-1183). Two consequences: (a) no failure counter, no log line, no state change — nothing in the process learns that the Olympe PCMD wire is rejecting every write; (b) `last_pcmd_call_mono_ns` is assigned BEFORE the call, so `pcmd_timing_snapshot()` (980-986) returns a fresh `pcmd_call_mono_ns` for a command that raised, and `run_loop.emit()` merges that snapshot straight into the per-tick JSONL via `rec.update(hooks.pcmd_timing() or {})` (1231-1235). In the production configuration …
- **Reproduction**: Instantiate `SafetyMonitor(send_pcmd=lambda *a: (_ for _ in ()).throw(RuntimeError("link down")), safety=<AUTO stub>)`, call `.start()`, then `mon.send_authorized((0, 30, 0, 0), lambda: True, lambda: False)`. The call returns `(True, "atomic AUTO authorization; latest PCMD queued", (0, 30, 0, 0))` and `mon.terminated.is_set()` stays False forever; `mon.pcmd_timing_snapshot()["pcmd_call_mono_ns"] `advances on every …
- **Recommended fix**: Add `self.pcmd_send_failures = 0` / `self.last_pcmd_error = ""` to `__init__` and replace each `except Exception: pass` around `self._send` with a helper `self._wire(*pcmd)` that (1) only assigns `self.last_pcmd_call_mono_ns` AFTER the call returns, (2) increments the counter and records the repr on failure, resetting it on success, and (3) once the counter exceeds a small threshold (e.g. 0.5 s worth of consecutive failures) calls `self._latch_terminal("LAND", "PCMD wire failing -> land")` and prints a warning, …
- **Verification test**: test_persistent_pcmd_send_failure_latches_land_and_is_not_timestamped: start a SafetyMonitor whose send_pcmd always raises, feed it authorized AUTO commands for > timeout seconds, assert monitor.pcmd_send_failures > 0, assert monitor.terminated.is_set() and terminal_action == "LAND", and assert pcmd_timing_snapshot()["pcmd_call_mono_ns"] is None (never stamped for a failed write).
- **Verifier**: Verified verbatim, line-by-line, in /home/allen/localization/定位演算法/flight_control/path_follow_flight.py. (1) Claimed code is real and at the claimed lines. `awk NR>=1170,<=1190` prints exactly: 1179: try: / 1180: self.last_pcmd_call_mono_ns = time.monotonic_ns() / 1181: self._send(*desired) / 1182: except Exception: / …

### F05 · High · 【已修 · 第三輪】· Manual-nudge deadman TTL has no upper bound; a large --nudge-pulse-s / NUDGE_PULSE_S silently defeats the "frozen UI decays to hover" guarantee

- **Component**: config-repro / operator_interface / real-drone backend (manual nudge deadman)
- **File**: `控制介面程式/operator_interface/olympe_live_backend.py`  **Line**: 608  **Symbol**: `OlympeLiveBackend.__init__`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Operator launches with `NUDGE_PULSE_S=600 ./start_anafi_live.sh` (or `--nudge-pulse-s 600`), holds a movement key, and the Tk UI then stops delivering the KeyRelease / heartbeat (window focus steal, X/Wayland compositor stall, UI thread blocked on a slow map redraw, localizer GPU hang blocking tick()).
- **Impact**: The drone continues receiving non-zero PCMD at 20 Hz in the held direction for the full TTL with no operator input. At 600 s that is effectively "until battery death or until it hits something". The only remaining brakes are the firmware MaxDistance/MaxAltitude geofence and the safety pilot's sticks. This converts a 0.2 s fail-safe into an unbounded runaway.
- **Description**: The hold-to-move deadman TTL is clamped only from below. `self.nudge_pulse_s = max(0.1, requested_ttl)` accepts any arbitrarily large value. The nudge hold loop re-sends PCMD every 50 ms and only stops when `time.monotonic() >= self._nudge_deadline`; the UI refreshes that deadline from `OperatorApp.tick()`. The design comment in flight_operator_app.py states the intent explicitly: "Backend TTL is refreshed only while Tk continues to observe a physical hold. A frozen UI therefore decays to zero PCMD." That decay time IS `nudge_pulse_s`. Nothing validates an upper bound: `flight_operator_app.py:6434` declares `--nudge-pulse-s` with `type=float` and no range check (while the very same argument block DOES range-check `--max-altitude-m`, …
- **Reproduction**: 1) `grep -n 'nudge_pulse_s' 控制介面程式/operator_interface/olympe_live_backend.py` -> only line 608 bounds it, and only from below. 2) In test_olympe_live_backend_safety.py the existing test `test_nudge_heartbeat_deadman_zeros_stale_hold` (line 1587) calls `make_backend(pulse_s=0.01)` and asserts `backend.nudge_pulse_s == pytest.approx(0.1)` -- it asserts the LOWER clamp only. Change that fixture to `pulse_s=600.0` and …
- **Recommended fix**: Bound the TTL on both sides at the single construction point and reject out-of-range input at the CLI. In olympe_live_backend.py:608 use `self.nudge_pulse_s = min(1.0, max(0.1, requested_ttl))` (1.0 s = 20 held PCMD frames, already generous), and add to flight_operator_app.py next to line 6506: `if not 0.1 <= args.nudge_pulse_s <= 1.0: ap.error("--nudge-pulse-s must be in [0.1, 1.0]")`.
- **Verification test**: Add to test_olympe_live_backend_safety.py: `def test_nudge_deadman_ttl_is_upper_bounded(make_backend): backend = make_backend(skycontroller=False, pulse_s=600.0); assert backend.nudge_pulse_s <= 1.0` plus a launcher/argparse test asserting `flight_operator_app` exits non-zero for `--nudge-pulse-s 600` (mirroring the existing `--min-takeoff-battery-pct` range test style).
- **Verifier**: CONFIRMED as a real defect; every piece of claimed evidence is verbatim correct, but the Critical rating is overstated so I downgrade to High. VERIFIED CODE (re-read with `grep -n "" file | sed -n 'A,Bp'`): - 控制介面程式/operator_interface/olympe_live_backend.py:604-608 exactly as quoted: 604 # Deadman TTL refreshed by UI …

### F06 · High · 【已修 · 第三輪】· EDM runtime profile is optional everywhere at runtime; when absent the localizer silently substitutes unverified hardcoded quality-gate thresholds and skips the profile SHA-256 verification entirely

- **Component**: config-repro / localization / production localizer construction
- **File**: `定位演算法/deploy_code/sfm_glomap_deploy/production_localizer_factory.py`  **Line**: 176  **Symbol**: `build_production_localizer`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: A site profile JSON omits the `localizer_profile` key (it is optional in the schema), or SFM_LOCALIZER_PROFILE is unset for the flight/passive-session entry points. The run proceeds and only emits an informational `profile=defaults` / `variant=production_edm_defaults_...` line on stderr -- no warning, no error, no digest check.
- **Impact**: The localization quality gates that decide whether a pose is trusted by the operator display and (once unlocked) by the controller come from mutable source defaults instead of the SHA-256-pinned, reviewed contract. Any future edit to EDMConfig's defaults, or a tampered/edited deploy tree, silently changes the accept/reject thresholds in the field with no artifact-integrity check, defeating the pinning that artifact_integrity.KNOWN_SHA256 exists to provide for edm_production_profile.json.
- **Description**: Every safety-relevant localization gate (acquire/track/weak min inliers, max_reproj_error_acquire/track, pnp_ransac_max_error, max_jump and the adaptive-jump ladder, prediction_max_dt) lives in the EDM deployment profile, and `edm_profile.apply_edm_tracker_profile` is strict: it rejects a profile that does not name every EDMConfig field. But reaching that strict path is conditional on the profile being supplied at all. `if production_profile:` guards BOTH `verify_sha256(...)` and `apply_edm_tracker_profile(...)`; a falsy value silently falls through to `production_edm_config()` code defaults and a default-constructed EDMMatcher, with `profile_name = "defaults"`. The chain that makes the profile falsy is entirely non-erroring: …
- **Reproduction**: 1) Copy 控制介面程式/site_profiles/river_site_edm.json, delete the `"localizer_profile"` line and the `asset_sha256.localizer_profile` entry. 2) `load_site_profile(copy)` succeeds (no exception) -- `profile.localizer_profile is None`. 3) `flight_operator_app.py --interface real-flight --site-profile <copy> ...` builds LiveLocalizerClient without `--production-profile`; the worker prints `EDM …
- **Recommended fix**: Fail closed for the EDM backend: in build_production_localizer, immediately after `if selected == "edm":`, add `if not production_profile: raise ValueError("EDM backend requires an explicit, SHA-256-pinned edm-deployment-profile/v1")`. Correspondingly make `localizer_profile` and `asset_sha256.localizer_profile` mandatory in site_profile.load_site_profile whenever `localizer == "edm"`, so the failure surfaces at profile load rather than at model construction.
- **Verification test**: Add to test_worker_lifecycle.py / test_site_profile.py: (a) `with pytest.raises(ValueError): build_production_localizer(backend='edm', production_profile=None, ...)`; (b) `with pytest.raises(ValueError, match='localizer_profile'): load_site_profile(profile_without_localizer_profile)` for an EDM site profile.
- **Verifier**: I tried to refute this and could not find any guard. Every cited line is verbatim correct and the fall-through is reachable in the production operator path. 1) The code exists exactly as claimed. `production_localizer_factory.py` lines 171-191 (re-read with `grep -n`): `172 config = production_edm_config()` / `173 …

### F07 · High · 【已修 · 第二輪 #10】· EDM tracker's two-frame jump confirmation accepts the same capture_stamp as an independent confirmation

- **Component**: filter-jump / deploy_code / ProductionEDMTracker trajectory gate
- **File**: `定位演算法/deploy_code/sfm_glomap_deploy/production_edm_tracker.py`  **Line**: 833  **Symbol**: `ProductionEDMTracker.localize (limited_jump confirmation)`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: A PnP spike whose step exceeds adaptive_jump_limit but stays under cfg.max_jump, on a frame that the consumer submits twice because no newer frame has arrived (video gap shorter than stale_s=0.5 s).
- **Impact**: The adaptive trajectory envelope — the tracker's own defence against an isolated bad PnP centre — is defeated by a repeat of the same observation. The spiked centre becomes `self.st.center`, is emitted as an `ok` pose with a fresh capture stamp, and is appended to `accepted_step_norms`, which widens `adaptive_jump_limit` for subsequent frames (base_limit = adaptive_jump_factor * median(step_history)). A single accepted spike therefore both corrupts the current pose and loosens the gate against the next one.
- **Description**: The EDM tracker's second-layer trajectory gate rejects a pose whose step exceeds the adaptive envelope, stashes it as `pending_limited_center/_stamp/_limit`, and requires the next frame to agree with a stationary or constant-velocity prediction before accepting. The time guard on the constant-velocity branch is `next_dt >= 0.0` (line 847), which explicitly ADMITS next_dt == 0 — i.e. the same capture stamp. With next_dt == 0, `predicted = pending + velocity * 0 == pending`, so both the stationary and the constant-velocity residual collapse to `|C - pending|`, which is ~0 when the identical frame is re-localized. `agrees` is therefore True and the out-of-envelope pose is accepted. The author was aware that non-increasing capture stamps occur …
- **Reproduction**: Executed with the project's own fixture. File: /tmp/claude-1000/-home-allen/fe9d2d50-0406-4e95-98fb-b72138470034/scratchpad/test_edm_dup.py, reusing `_jump_gate_tracker` from 定位演算法/validation/tests/test_edm_tracker_quality.py:283. tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.05]) tracker.st.last_capture_stamp = 0.9 first = tracker.localize(frame, capture_stamp=1.0) second = tracker.localize(frame, …
- **Recommended fix**: Require the confirming submission to be strictly newer than the pending one. Change the guard at line 847 from `next_dt >= 0.0` to `next_dt > 1e-6`, AND add an early rejection before the confirmation block so a duplicate stamp can never confirm at all, e.g. right after line 838: if pending_stamp is not None and not (capture_stamp > pending_stamp): info['rejected'] = 'limited_jump_duplicate_stamp' self._on_miss(info); return info The existing tests test_limited_jump_requires_a_second_consistent_pose (stamps …
- **Verification test**: Add to 定位演算法/validation/tests/test_edm_tracker_quality.py, beside test_limited_jump_requires_a_second_consistent_pose: def test_limited_jump_is_not_confirmed_by_a_resubmitted_frame(monkeypatch): tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.05]) tracker.st.last_capture_stamp = 0.9 first = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0) second = tracker.localize(np.zeros((720, 1280, 3), …
- **Verifier**: DEFECT AND IMPACT CONFIRMED; ROOT-CAUSE LINE CORRECTED from 847 to 833. 1) Code exists verbatim. Re-read production_edm_tracker.py 816-872: lines 841-851, 860-871 are exactly as quoted; lines 529-535 are exactly as quoted. 2) The reporter misattributed the cause. I reproduced it (scratch test, no repo file touched): …

### F08 · High · 【已修 · 第二輪 #7】· take_pc_control() re-enables PC PCMD after a concurrent emergency stop / stream-loss / link-loss latch, because it never re-checks the safety epoch after its blocking piloting-source handoff

- **Component**: operator-path / operator UI -> backend -> drone command path (OlympeLiveBackend PC authority)
- **File**: `控制介面程式/operator_interface/olympe_live_backend.py`  **Line**: 2101  **Symbol**: `OlympeLiveBackend.take_pc_control`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Operator presses 恢復電腦控制 (pc_control) on a SkyController link (default `--ip 192.168.53.1 --controller skycontroller3`). While the background `olympe-ui-pc_control` thread is inside `_set_piloting_source("Controller").wait(_timeout=10)`, any of these fires: (a) the operator presses 緊急停止電腦動作 (flight_operator_app.py:3199-3210, dispatched synchronously on the Tk thread), (b) `poll()` -> `_evaluate_video_health()` detects a frozen/stale decoder and …
- **Impact**: The operator's emergency stop — and the automatic stream-loss / link-loss manual handoff — is silently reverted: `pilot_sticks` returns to False, host PCMD is re-enabled, the firmware piloting source is put back to Controller, and the HUD reports `tracker_state="PC"` as if PC control were legitimately held. Nothing re-asserts the latch: `fail_safe` is one-shot, `_evaluate_video_health` is permanently latched (`_stream_failure_latched`, line 3021), and `_latch_total_link_loss` early-returns once `active_incident == control_link_lost` (line 3210). The aircraft is left airborne under host PCMD …
- **Description**: The UI dispatches `pc_control` on a background thread (`OperatorApp._dispatch_live_command`, flight_operator_app.py:3815-3841; "pc_control" is in the `async_commands` set at flight_operator_app.py:3970-3979), while `emergency_stop` and `backend.poll()` run on the Tk main thread. Inside `take_pc_control` the call `self._set_piloting_source("Controller")` (olympe_live_backend.py:2090) does an Olympe `.wait(_timeout=10)` (olympe_live_backend.py:786-788) and holds no lock. Every safety action in this backend — `fail_safe`/`give_to_pilot` (2925-2949, 1928-1962), `land_cmd` (2170), `_latch_total_link_loss` (3208), `_request_rth` (3072) — protects itself by setting `pilot_sticks = True` and bumping `self._pulse_token`. `takeoff_cmd` re-reads …
- **Reproduction**: Reproduced with the repo's own fake-Olympe harness (no drone). Using `make_backend` from 控制介面程式/operator_interface/test_olympe_live_backend_safety.py, wrap `_set_piloting_source` so the "Controller" call blocks on an Event, run `take_pc_control()` in a thread, and while it is blocked call `backend.fail_safe(FailureReason.EMERGENCY_STOP)` (asserted to leave `pilot_sticks=True`, `tracker_state='EMERGENCY_MANUAL'`), …
- **Recommended fix**: Snapshot the epoch before the blocking handoff and re-validate it before granting PC authority, exactly as takeoff_cmd does. In take_pc_control, before `self.nudge_clear(reason="pc_control")` capture `with self._lock: epoch = self._pulse_token`, then change lines 2096-2101 to: with self._lock: if (self._cleanup_done or self.drone is None or epoch != self._pulse_token or self.pilot_sticks): self.pilot_sticks = True self.log.event("pc_control", ok=False, note="superseded_by_safety_command") return False …
- **Verification test**: Add to test_olympe_live_backend_safety.py: `test_pc_control_is_superseded_by_a_concurrent_fail_safe` — block `_set_piloting_source("Controller")` on an Event, start `take_pc_control()` in a thread, call `backend.fail_safe(FailureReason.EMERGENCY_STOP)`, release, join, then assert `take_pc_control() is False`, `backend.pilot_sticks is True`, and `backend.state.tracker_state == "EMERGENCY_MANUAL"`. Add the same test …
- **Verifier**: Verified verbatim: 控制介面程式/operator_interface/olympe_live_backend.py:2065 `def take_pc_control`, 2090 `if not self._set_piloting_source("Controller"):`, 2096 `with self._lock:`, 2101 `self.pilot_sticks = False`, 2110 `self.state.tracker_state = "PC"`. The handoff blocks: `_set_piloting_source` (line 774) returns …

### F09 · High · 【已修 · 第二輪 #8】· Live video-health monitor is armed only once per session: `_stream_failure_latched` is never reset, so a second stream freeze after any recovery triggers no handoff

- **Component**: operator-path / OlympeLiveBackend stream-loss failsafe
- **File**: `控制介面程式/operator_interface/olympe_live_backend.py`  **Line**: 2992  **Symbol**: `OlympeLiveBackend._evaluate_video_health`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Any first stream stall in a session — decoder duplicate_run >= 15 or newest-frame age >= 0.75 s — latches the monitor. The operator then presses 恢復電腦控制 (pc_control) to resume manual nudging. Any later freeze or total video loss during the same flight is never detected: `_evaluate_video_health` returns False at line 2993 before reading `last_frame_age()` at all.
- **Impact**: In a GPS-denied site the operator flies the aircraft by camera. After the first (possibly momentary) decoder stall, the system will never again zero PCMD and hand control back to the safety pilot on a frozen or dead feed. The operator can keep nudging while the video panel shows a stale frame, with no LINK/stream alarm from this path (`state.stream` is also left at whatever the UI last wrote — see flight_operator_app.py:3960-3967, which sets `backend.state.stream = "OK"` on every hover/land/pc_control press).
- **Description**: `_evaluate_video_health` sets `self._stream_failure_latched = True` (line 3021) the first time it fires and returns immediately on every subsequent call (line 2992-2993). Grep for `_stream_failure_latched` in the file returns exactly three sites: 640 (init `False`), 2992 (the early return), 3021 (set `True`). There is no reset anywhere — not in `poll()`, not on stream recovery, not when the operator takes PC control back. Meanwhile the underlying freeze detector in the grabber DOES clear: `olympe_frame_source.py:730` sets `self._dup_n = 0` on the next distinct decoded frame, so `frame_pipeline_stats['frozen']` and `duplicate_run` go back to healthy values. So a transient PDRAW stall permanently disarms the only automatic detector of a …
- **Reproduction**: With the repo harness: `backend = make_backend(); backend.pilot_sticks = False; backend.video_stream = SimpleNamespace(last_stamp=100.0); backend.grabber = SimpleNamespace(last_frame_age=lambda: 0.05, frame_pipeline_stats={'duplicate_run': 15, 'frozen': True})`. `backend._evaluate_video_health(100.1)` -> True (handoff). Now simulate recovery and a second freeze: set `frame_pipeline_stats={'duplicate_run': 0, …
- **Recommended fix**: Make the latch edge-triggered rather than permanent: clear `self._stream_failure_latched = False` inside `_evaluate_video_health` when the stream is observed healthy again (i.e. in the `if not frozen and age_s < _STREAM_STALE_S:` branch at line 3019, set the flag False before `return False`). That re-arms the detector only after a confirmed fresh, non-duplicate frame, so it cannot chatter while the stream is still bad.
- **Verification test**: `test_stream_health_monitor_rearms_after_a_confirmed_recovery`: freeze -> assert handoff and `pilot_sticks is True`; feed a healthy `frame_pipeline_stats` and one `_evaluate_video_health` call; `take_pc_control()`; freeze again -> assert `_evaluate_video_health(...) is True`, `backend.pilot_sticks is True`, and a second `stream_lost` record in `backend.log.records`.
- **Verifier**: Verified by direct reading, not inference. 1) Code is verbatim at the claimed lines. olympe_live_backend.py:2991 `def _evaluate_video_health(self, now: float | None = None) -> bool:`; 2992-2993 `if self._stream_failure_latched or self.video_stream is None:` / `return False`; 3019-3021 `if not frozen and age_s < …

### F10 · High · 【已修 · 第二輪 #9】· Real-flight interface will drive the operator's pose, map track and localization-quality HUD from a recorded replay JSON while a real ANAFI is connected and commandable

- **Component**: operator-path / flight_operator_app main() interface selection / state rendering
- **File**: `控制介面程式/operator_interface/flight_operator_app.py`  **Line**: 6944  **Symbol**: `main / OperatorApp.state_from_replay`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: `python3 flight_operator_app.py --interface real-flight --site-profile <p> --no-live-localize --replay-json <recorded.json>` (or `--live` instead of `--interface real-flight`). Also reachable without typing `--replay-json`: the argparse default is `str(DEFAULT_REPLAY_JSON) if DEFAULT_REPLAY_JSON.exists() else ""`, so on any workstation where `outputs/downloads_validation_20260702/P0230023_v3_temporal.json` exists, `--live --no-live-localize` …
- **Impact**: The operator's primary situational-awareness surfaces — the 3D map drone marker, the flown-track history, `inliers`, `reproj`, `loc`, `tracker_state` — are fabricated from a file while a real aircraft is connected and every flight button (起飛 / 原地降落 / 懸停 / 微移 / 緊急停止) is live. The operator can nudge the aircraft toward a position the map says is safe while the map is replaying an unrelated recorded flight. The only textual clue is the token `REPLAY` in the mode field; `loc=OK`, `tracker_state=TRACK`, `inliers=412` and the green health label are all presented as genuine measurements.
- **Description**: `resolve_operator_interface` (flight_operator_app.py:353-369) correctly refuses `--interface real-flight` together with `--video`, so a recorded *video file* cannot be fed to the real backend. But `--replay-json` has no such guard. Line 6944 loads recorded rows whenever `args.replay_json and not args.live_localize` — with no `and not args.live`. And `want_loc = args.live_localize and (...)` at line 6857 makes `localizer is None` under `--no-live-localize`, so `tick()` takes the `state_from_replay` branch (flight_operator_app.py, tick: `if self.localizer is not None: st = self.state_from_live(st) else: st = self.state_from_replay(st)`). `state_from_replay` (5560-5598) then overwrites `base.pose`, `base.loc`, `base.tracker_state`, …
- **Reproduction**: Executed against the real module (no drone): fed a live-looking DroneState (mode=LIVE, tracker_state=STICKS) plus two recorded rows into `OperatorApp.state_from_replay`. Output: mode= REPLAY loc= OK tracker= TRACK pose= [11.0, -2.0, 33.0, 0.5] inliers= 412 reproj= 0.31 stream= OK HUD identity -> REAL ANAFI | mode=REPLAY | TRACK | OK status strip -> REPLAY | OK | TRACK The HUD line rendered by `render_video` …
- **Recommended fix**: Make the replay flag cross-interface-exclusive exactly like `--video`. Either (a) reject it up front in `resolve_operator_interface` by passing `replay_json` alongside `video` and raising `ValueError` for real-flight, or (b) minimally, at line 6944 change the condition to `if args.replay_json and not args.live_localize and not args.live:` and add an `ap.error(...)` when `args.live and args.replay_json` was passed explicitly, so the operator is told rather than silently given an empty overlay.
- **Verification test**: `test_real_flight_interface_refuses_recorded_replay_rows`: parse `--interface real-flight --site-profile <fixture> --no-live-localize --replay-json <fixture>` through `main()`'s argument handling (or the extracted `resolve_operator_interface`) and assert `SystemExit`/`ValueError`. Plus a unit test asserting `OperatorApp` constructed with `backend.is_live is True` and non-empty `replay_rows` raises, so the two can …
- **Verifier**: I read every cited line and tried hard to find the guard; there is none. 1) The anchor is exact. /home/allen/localization/控制介面程式/operator_interface/flight_operator_app.py: 6943 ` replay_rows = []` 6944 ` if args.replay_json and not args.live_localize:` 6945 ` replay_rows = …

### F11 · High · 【已修 · 第二輪 #1】· EDM tracker skips the translation-jump gate on every LOST reacquisition, and has no rotation/yaw sanity gate at all

- **Component**: pose-gates / localization quality gates / pose acceptance
- **File**: `定位演算法/deploy_code/sfm_glomap_deploy/production_edm_tracker.py`  **Line**: 787  **Symbol**: `ProductionEDMTracker.localize`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Tracker is in state LOST (2 misses in TRACK + 2 in WEAK_TRACK, or the flight loop's `force_relocalize` hook which does `setattr(loc.state, "mode", "LOST")` at path_follow_flight.py:1921), and a subsequent frame yields >= acquire_min_inliers (80) inliers against a reference that is not the true place — repeated texture (football-field line markings, river embankment, identical facades) or a MegaLoc/recovery-bank mismatch.
- **Impact**: A wrong-place pose is accepted with no bound on how far it is from the last known position and no bound on how far the heading rotated. It reaches the flight loop labelled `next_mode="TRACK"` with >= 80 inliers, so run_loop's `low_conf` test (path_follow_flight.py:1436-1438) is False and the drone is driven on it after RECOVERY_GOOD_FIXES=2 ticks. Any teleport under MAX_POSE_JUMP_U = 1.5 map units passes run_loop's own outlier gate (path_follow_flight.py:1405) immediately with no confirmation at all; teleports above 1.5 u need only one agreeing second fix. By contrast, in TRACK the same …
- **Description**: The only geometric sanity check on an accepted PnP pose is the two-layer trajectory gate at lines 787-878, and it is guarded by `info["state_in"] != "LOST"`. Every pose accepted while the tracker is in LOST — i.e. exactly the relocalization case where retrieval or a blind recovery-bank scan can land on the wrong physical place — bypasses both the hard `max_jump` reject and the adaptive `limit_center_step` + two-frame confirmation. In addition, `yaw` is computed at line 781 and published at line 916 but is NEVER compared against the prior heading anywhere in the file: `grep -n "yaw" production_edm_tracker.py` returns only 61,129,173,424,471,480,481,497,498,781,895,916 — lines 471/480/481/497/498 are `_track_candidates` *candidate …
- **Reproduction**: 定位演算法/validation/tests/test_edm_tracker_quality.py already pins the defect. `_jump_gate_tracker(monkeypatch, [1.0])` (line 439) makes PnP return `Rigid3d(Rotation3d(), [-1.0,0,0])`, i.e. centre = [1.0,0,0], against `st.center = np.zeros(3)`: 438 def test_lost_recovery_does_not_train_continuous_step_history(monkeypatch) -> None: 439 tracker = _jump_gate_tracker(monkeypatch, [1.0]) 441 tracker.st.state = "LOST" 449 …
- **Recommended fix**: Apply a bounded acquisition gate in LOST instead of skipping it, mirroring the XFeat port it came from. Minimal change at production_edm_tracker.py:787: keep the existing branch for non-LOST, and add an `elif self.st.center is not None and <prior not expired>:` branch that rejects when `norm(C - self.st.center) > cfg.acquire_max_jump` (new field, default 1.25 to match ProductionConfig.acquire_max_jump) or when `abs(_angle_diff(yaw, self.st.yaw)) > radians(cfg.acquire_max_yaw_diff_deg)` (new field, default 90.0), …
- **Verification test**: In 定位演算法/validation/tests/test_edm_tracker_quality.py add `test_lost_reacquisition_rejects_unbounded_teleport_and_yaw_flip`: build `_jump_gate_tracker(monkeypatch, [1.0])`, set `tracker.st.state = "LOST"`, `tracker.st.yaw = 0.0`, `tracker.st.last_capture_stamp = capture_stamp - 0.05` (prior fresh), and assert `not result["ok"]` and `result["rejected"] == "acquire_jump"`. A second case returning …
- **Verifier**: The defect is real and I reproduced it. production_edm_tracker.py:787 reads verbatim `if self.st.center is not None and info["state_in"] != "LOST":`, so the hard `max_jump` reject (line 793: `if raw_step > cfg.max_jump:` -> line 805 `info["rejected"] = "jump"`) and the adaptive `limit_center_step` + two-frame …

### F12 · High · 【已修，見 A02】· One failed automatic safety action permanently disarms ALL runtime flight safety (battery/altitude/distance/controller-loss)

- **Component**: state-machine / operator live backend — runtime safety supervisor
- **File**: `控制介面程式/operator_interface/olympe_live_backend.py`  **Line**: 3165  **Symbol**: `OlympeLiveBackend._schedule_runtime_safety_action / _evaluate_runtime_safety / _execute_runtime_safety_action`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Any first runtime safety action whose landing does not confirm within 20 s: e.g. altitude-limit action at max_altitude-1.0 m where the descent takes longer than the expectation timeout, or a controller-disconnect action (which skips RTH by design, line 3145) whose Landing does not reach the `landed` state. Afterwards, battery falling to <=10 % (_CRITICAL_BATTERY_PCT) or crossing 95 % of max_distance produces no action at all.
- **Impact**: The drone keeps flying past the geofence / into battery exhaustion with no host-side automatic RTH or landing. Only the onboard firmware geofence/low-battery policy and a human operator remain. The UI shows `SAFETY_ACTION_PENDING` -> `LAND_UNCONFIRMED` and then (see separate finding) reverts to a normal-looking flight state, so the operator is not told that autonomous safety is dead.
- **Description**: `_runtime_safety_action_latched` is a one-way latch: it is set to True the moment the FIRST autonomous safety action is scheduled (line 3165) and is never cleared anywhere in the module (only initialised False at line 642). `_evaluate_runtime_safety()` short-circuits on that flag (lines 3182-3189), so once the latch is set no further battery-critical, altitude-limit, distance-limit or controller-disconnect action can ever be scheduled again. `_execute_runtime_safety_action` (3139-3155) does NOT clear the latch when its action fails — it merely logs `runtime_safety ok=False` (3149-3155). A failed RTH falls back to `land_cmd`, and `land_cmd` returns False whenever touchdown is not confirmed within the 20 s …
- **Reproduction**: Confirmed with a mock-drone pytest repro (no network, no drone): ``` backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0) backend.drone.flight_state = "flying"; backend.state.flight_state = "flying" backend.drone.landing_success = False # touchdown never confirms backend.drone.home_reachable = False assert backend._schedule_runtime_safety_action(FailureReason.CONTROLLER_DISCONNECTED.value) …
- **Recommended fix**: In `_execute_runtime_safety_action`, release the latch when the action did not reach a confirmed terminal state, so a later (or more severe) trigger can act: after `landed = self.land_cmd(...)`, if `not landed`, do `with self._runtime_safety_action_lock: self._runtime_safety_action_latched = False` (and keep `state.active_incident`). Optionally rate-limit re-arming (e.g. one retry per `_action_retry_s`) instead of latching forever, and always allow escalation to a strictly more severe reason (battery_critical) …
- **Verification test**: Add to test_olympe_live_backend_safety.py: `test_failed_safety_land_rearms_runtime_safety` — set `landing_success=False`, schedule CONTROLLER_DISCONNECTED, join the thread, assert `not backend._landed`; then set `battery_pct=5.0` and assert `backend._evaluate_runtime_safety()` is True and a second Landing/return_to_home appears in `backend.drone.events`.
- **Verifier**: Read the code directly. olympe_live_backend.py:3157-3165 `_schedule_runtime_safety_action` sets `self._runtime_safety_action_latched = True` (line 3165) as the first act inside `self._runtime_safety_action_lock`; :3181-3189 `_evaluate_runtime_safety` returns False whenever that flag is set; :3139-3155 …

### F13 · High · 【已修 · 第三輪】· Nudge/PCMD is accepted while an automated TakeOff or Landing is still in progress — no maneuver-in-progress guard

- **Component**: state-machine / operator live backend — PCMD authority state machine + operator UI dispatch
- **File**: `控制介面程式/operator_interface/olympe_live_backend.py`  **Line**: 2602  **Symbol**: `OlympeLiveBackend.nudge_begin / send_pcmd / _land_and_confirm / takeoff_cmd`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Operator presses 原地降落 (or an automatic safety land runs) and then, during the up-to-22 s landing descent, presses any movement key (w/a/s/d/i/j/k/l/…) or a 微移 button. Same during the up-to-17 s TakeOff climb.
- **Impact**: A non-zero translation PCMD is streamed at 20 Hz to the aircraft while the firmware is executing its automated landing descent close to the ground (or its takeoff climb). Host-side autonomy and firmware maneuver fight each other, the drone can translate/climb during touchdown, and the landing can fail to confirm — which then leaves `_landed` False, `pilot_sticks` False and (with the finding above) autonomous safety permanently latched.
- **Description**: The only PCMD gates are `pilot_sticks`, `_landed` and `_cleanup_done` (send_pcmd lines 1908-1911, nudge_begin lines 2599-2604). There is no flag for "an automated Landing/TakeOff maneuver is currently pending". `land_cmd` deliberately sets `pilot_sticks = False` (line 2187) so it may command the aircraft, and `_landed` is only set after touchdown confirmation, so during the whole `exp.wait(_timeout=22)` window of `_land_and_confirm` (line 2139) every PCMD gate is open. `takeoff_cmd` has the same window: it sets `pilot_sticks = False` and `_landed = False` (lines 2278-2279) and then waits up to 17 s outside the lock (line 2297). The operator UI makes this reachable by design: `takeoff`/`land` are dispatched to a background thread precisely …
- **Reproduction**: Mock-drone pytest repro (no network, no drone). Landing case — block the Landing expectation with a barrier, then nudge from the caller thread: ``` backend.drone = BarrierDrone(); backend.drone.flight_state = "flying" Thread(target=lambda: backend.land_cmd("ui_land_now")).start() wait_entered.wait() # Landing on the wire, touchdown not confirmed assert backend.state.tracker_state == "LANDING" and not backend._landed …
- **Recommended fix**: Add an explicit maneuver epoch/flag, e.g. `self._maneuver_pending: str | None`, set under `self._lock` immediately before scheduling TakeOff (line 2280) and Landing (line 2135) and cleared in a `finally` after the expectation resolves. Reject `nudge_begin`, `nudge_heartbeat` and `send_pcmd` (other than the internal zero PCMD used by land/cleanup) while it is set, logging `pcmd_blocked_maneuver`. Cheapest equivalent: capture `_pulse_token` in `nudge_begin`/the hold loop and refuse to emit when it has moved since …
- **Verification test**: Add `test_nudge_is_refused_while_landing_is_pending` and `test_nudge_is_refused_during_takeoff_wait` to test_olympe_live_backend_safety.py, modelled on the existing barrier tests (`test_takeoff_schedule_before_land_orders_landing_after_takeoff`): assert `backend.nudge_begin("前") is False`, `backend.send_pcmd(0, 8, 0, 0, reason="probe") is False`, and that no PCMD with a non-zero axis appears in `backend.drone.pcmds` …
- **Verifier**: Every claimed line is verbatim correct and I reproduced the behaviour. 1) The gates are exactly as claimed. `olympe_live_backend.py:1906 def send_pcmd(...)`, `1908 if self._cleanup_done or self._landed:`, `1910 if self.pilot_sticks:`. `2594 def nudge_begin(...)`, `2599 if self.pilot_sticks:`, `2602 if …

### F14 · High · 【已修，見 A02】· _runtime_safety_action_latched is never cleared after a FAILED safety action, permanently disabling battery/altitude/distance/controller-disconnect auto-actions for the rest of the flight

- **Component**: watchdog / operator_interface / runtime flight-safety supervisor
- **File**: `控制介面程式/operator_interface/olympe_live_backend.py`  **Line**: 3165  **Symbol**: `_schedule_runtime_safety_action / _execute_runtime_safety_action / _evaluate_runtime_safety`
- **Status**: Confirmed (adversarially verified)
- **Trigger**: Any first runtime safety action that does not confirm. Most likely path given this system's stated premise (no reliable GPS): battery reaches _CRITICAL_BATTERY_PCT=10.0 -> _refresh_home_state() returns False (no valid/reachable home) -> land_cmd -> ANAFI aborts or delays touchdown (obstacle under the aircraft, HoveringWarning, high descent from a 50 m ceiling, or a transient link glitch that swallows the FlyingStateChanged event) -> the 22 s …
- **Impact**: For the remainder of the flight _evaluate_runtime_safety returns False on its first line, so battery_critical, altitude_limit, distance_limit and controller_disconnected produce NO automatic action ever again — no retry of the landing, no RTH, nothing. The aircraft is still airborne on a battery below 10 %. As a compounding effect, land_cmd left self.pilot_sticks = False (line 2188) and the firmware piloting source at 'Controller' because the `elif landed:` restore at 2210-2216 was not taken, so the SkyController pilot has no authority until they physically deflect a stick to trigger …
- **Description**: _schedule_runtime_safety_action sets self._runtime_safety_action_latched = True (line 3165) BEFORE the action is attempted, and _evaluate_runtime_safety short-circuits on that flag as its very first condition (line 3184-3185). The flag is assigned in exactly two places — `grep -n "_runtime_safety_action_latched" 控制介面程式/operator_interface/olympe_live_backend.py` returns 642 (constructor, False), 3160 (read), 3165 (set True), 3184 (read) — so it is never reset. _execute_runtime_safety_action (3139-3155) can complete without any effective action: land_cmd returns False when _land_and_confirm's `exp.wait(_timeout=22)` does not observe FlyingStateChanged(state='landed') (lines 2137-2141, 2157-2168), and the failure is only logged with ok=False …
- **Reproduction**: Using the existing make_backend fixture in 控制介面程式/operator_interface/test_olympe_live_backend_safety.py: set backend.drone.flight_state='flying', backend.drone.home_reachable=False, backend.state.battery_pct=9.0, and make the mock's Landing expectation report success()==False. Call backend._evaluate_runtime_safety() and join _runtime_safety_action_thread. Then set backend.state.drone_altitude_m=49.0 / …
- **Recommended fix**: In _execute_runtime_safety_action, clear the latch when the action did not confirm so the supervisor can re-arm: after `landed = self.land_cmd(...)`, if not landed then `with self._runtime_safety_action_lock: self._runtime_safety_action_latched = False` (and record a monotonic retry floor, e.g. self._runtime_safety_retry_at = time.monotonic() + 5.0, checked in _schedule_runtime_safety_action so a persistently failing action retries at a bounded rate instead of spinning). Keep the latch set on a confirmed …
- **Verification test**: test_failed_runtime_safety_action_rearms_the_supervisor: mock drone whose Landing never confirms; drive battery_critical, join the thread, assert _runtime_safety_action_latched is False and that a subsequent altitude_limit condition schedules a new action (a second Landing appears in drone.events). Pair it with test_successful_runtime_safety_action_stays_latched to pin the existing one-shot behaviour on success.
- **Verifier**: Read 控制介面程式/operator_interface/olympe_live_backend.py directly. Line 3165 `self._runtime_safety_action_latched = True` exists verbatim in _schedule_runtime_safety_action (def 3157), set before the worker thread is started (3171-3177). `grep -n "_runtime_safety_action_latched" ` returns exactly 4 hits: 642 (init …


---

### 3.2 Medium / Low（對抗式複驗確認，未修）

| # | Sev | 檔案:行 | 問題 | 觸發 | 建議修正 |
|---|---|---|---|---|---|
| M01 | Medium | `manual_nudge_pilot.py`:357 | manual_nudge_pilot: the nudge pulse thread has no exception guard, so any raise from the PCMD wire or the command log kills it mid-pulse and the … | Disk full / read-only filesystem (ENOSPC OSError from the line-buffered write at 167), a broken stdout pipe, or an Olympe exception from … | Mirror the pattern already applied to olympe_live_backend.py `_ensure_nudge_loop._loop`: wrap the whole `_pulse` body in `try: ... except Exception as exc: ... finally:` … |
| M02 | Medium | `manual_nudge_pilot.py`:403 | manual_nudge_pilot deadman thread calls the unguarded command log BEFORE landing, so a log-write failure kills the last-resort landing thread | The command-log file becomes unwritable (disk full → ENOSPC, filesystem remounted read-only, log directory removed) or stdout is a closed pipe — at … | Reorder to land first and log after (`pilot.land(...)` then `pilot.log.event(...)`), and wrap the whole `run_deadman` while-body in `try/except Exception` that still … |
| M03 | Medium | `flight_operator_app.py`:6432 | Manual nudge command authority (--nudge-pct / NUDGE_PCT) is accepted without any range check, allowing full-stick PCMD from a single keypress | Field operator runs `NUDGE_PCT=100 ./start_anafi_live.sh` (or a typo such as `--nudge-pct 80` instead of `8`). No error, no warning, no startup echo … | Add a bound at the CLI next to the existing envelope checks in flight_operator_app.py (after line 6506): `if not 1 <= args.nudge_pct <= 25: ap.error("--nudge-pct must be … |
| M04 | Medium | `flight_operator_app.py`:663 | env_bool() fails OPEN: a malformed SFM_REQUIRE_GPS_FOR_GEOFENCE silently disables the GPS-fix precondition that both the firmware geofence and the … | Anyone sets `SFM_REQUIRE_GPS_FOR_GEOFENCE=enabled` / `=Y` / `=ON!` / `=` (empty) in the field shell, a wrapper script, or a systemd unit, intending … | Make env_bool strict: accept {"1","true","yes","on"} -> True, {"0","false","no","off"} -> False, and `raise SystemExit(f"{name} must be a boolean (0/1/true/false), got … |
| M05 | Medium | `benchmark_edm_site_replay.py`:380 | Replay quality gate can pass on the wrong map: the pinned baseline binds only the video SHA, never the site profile or bundle, and contains no … | Run the replay with a bundle/point-cloud pairing that is internally self-consistent but wrong for the site (the exact failure the team was previously … | (a) In the baseline schema add `site_profile_sha256`, `bundle_sha256` and `localizer_profile_sha256`, and compare them alongside the video SHA at lines 380-386, raising … |
| M06 | Medium | `runtime_safety.py`:114 | Flight session manifest pins every asset and dependency digest but records no code revision, so a flight log cannot be tied to the software that … | Any real-flight or replay session. Afterwards, an incident review tries to determine which revision of path_follow_flight.py / olympe_live_backend.py … | Extend collect_runtime_identity with a `code` block: `git rev-parse HEAD`, `git rev-parse --abbrev-ref HEAD` and `git status --porcelain` emptiness (all wrapped in … |
| M07 | Medium | `path_follow_flight.py`:627 | HeadingEstimator fuses ANAFI NED yaw into the GLOMAP map heading with the wrong rotational sense, so the heading handed to the controller inverts … | Any pure-yaw segment in AUTO: RouteAutoController returns FOLLOW/REJOIN with a horizontal goal distance > minimum_yaw_alignment_distance (0.25) and … | Make the fusion sense-aware and fix it together with yaw_sign, because the two errors currently mask each other. In HeadingEstimator, learn the offset as a sum and … |
| M08 | Medium | `path_follow_flight.py`:1406 | run_loop's pose-jump confirmation can be satisfied by re-localizing the SAME camera frame, so one bad frame both proposes and confirms a relocation | A wrong global relocalization (perceptual aliasing in a repetitive corridor) on one frame, coinciding with a video gap of at least one control period … | Track the stamp of the measurement that set `pending_jump` and require the confirming fix to be a strictly newer measurement. Minimal change in run_loop: store … |
| M09 | Medium | `edm_localizer_adapter.py`:187 | pose_is_weak() is always False on the EDM backend, so the WEAK-fix hover gate is silently inert in every configured site | Flying any site profile (all select EDM) while the tracker is in WEAK_TRACK — i.e. after `weak_after`=2 consecutive missed fixes — and the recovering … | Publish the weak classification from the EDM adapter. In edm_localizer_adapter.py, inside the `self._last_info = {...}` literal (after line 190), add: "weak": … |
| M10 | Medium | `olympe_live_backend.py`:3219 | Distance geofence silently becomes a no-op when GPS position or Home is unavailable, while the UI still reports the fence as ON | Airborne with the distance geofence enabled; the drone loses its GPS fix (urban canyon, under structure, indoor) or Home was never recorded. … | Treat 'geofence enabled but position unavailable' as a fail-closed condition rather than a skipped branch: in `_evaluate_runtime_safety`, when … |
| M11 | Medium | `flight_operator_app.py`:5882 | Camera zoom above 1.0x silently stops all localization submissions while the status strip, video HUD and health label keep displaying the last TRACK … | Operator drags the 縮放 slider off 1.0x during a live flight (a normal action for inspecting a target), or the aircraft camera reports a non-1.0 zoom … | Make the paused/stale condition a first-class display state rather than an inferred one. In `submit_current_frame_for_localization`, when the zoom gate trips, set an … |
| M12 | Medium | `production_edm_tracker.py`:755 | inlier_ratio and inlier_grid_cells are computed on every frame and never compared against any threshold; no correspondence de-duplication either | Every accepted frame. In TRACK the correspondence set is capped at max_corr_total=900 (line 81) and min_inl is 50, so an inlier ratio of 5.6% is … | Add two thresholds to EDMConfig next to the existing gates at production_edm_tracker.py:64-67 — `min_inlier_ratio: float = 0.15` and `min_inlier_grid_cells: int = 6` … |
| M13 | Medium | `production_edm_tracker.py`:694 | Acquisition staged early-stop accepts a pose from the first 2 retrieved references without evaluating the other 8, and no multi-reference consensus … | BOOT_INIT or a LOST episode in a site with self-similar geometry, where MegaLoc's top-2 results are both from the wrong (but visually similar) place … | Make the early stop require agreement rather than plausibility. In the `staged` branch at line 700, when the first-2 hypothesis passes, additionally require that its … |
| M14 | Medium | `path_follow_flight.py`:1919 | WEAK-fix hover gate (SFM_GATE_WEAK) is silently inert on the production EDM backend: pose_is_weak reads a key the EDM adapter never sets | Any autonomous route flight with SFM_LOCALIZER_BACKEND=edm (i.e. every shipped site profile). The gate is dead on 100% of frames; the practical … | Make the flight hook derive weakness the same way the UI already does, instead of trusting a key one backend does not emit. In 定位演算法/flight_control/path_follow_flight.py … |
| M15 | Medium | `live_non_map_acceptance.py`:345 | Live acceptance harness records E_nudges as PASS when every nudge was rejected: backend.nudge() discards nudge_begin()'s False return | Run the harness in its air phase (`--i-understand-this-will-takeoff` with SFM_ALLOW_AUTO_TAKEOFF=1) in any state where nudges are blocked — most … | Check the return values instead of relying on exceptions. Replace line 337 with `began = self.backend.nudge_begin(name)` and line 339 with `hovered = … |
| M16 | Medium | `live_non_map_acceptance.py`:287 | Acceptance checks C_gimbal_zoom and G_camera_air are hard-coded PASS; the camera calls they wrap are silent no-ops whenever pilot_sticks is set … | Any run of the harness over a SkyController link (`--controller skycontroller3`, or ip 192.168.53.x). C_gimbal_zoom is in the documented ground-only … | Make the backend camera setters report success and have the harness assert on it. Change `set_gimbal_pitch` and `set_zoom` to `-> bool`, returning False on the early … |
| M17 | Medium | `live_non_map_acceptance.py`:294 | Acceptance harness records skipped air-phase checks (D/E/F/G/H) as PASS in the default --no-fly run | Every default (`--no-fly`, or simply not passing `--i-understand-this-will-takeoff` with SFM_ALLOW_AUTO_TAKEOFF=1) invocation of the harness. | Add a third state so a skipped check is never counted as a pass. Give `Check` a `skipped: bool = False` field, record the five air checks with `skipped=True` (and either … |
| M18 | Medium | `olympe_live_backend.py`:2710 | set_zoom writes the requested zoom into operator UI state before the guard that rejects the command, so the UI shows a zoom level that was never … | Operator adjusts zoom while the SkyController holds the sticks (`pilot_sticks` True — the default for every SkyController session per line 699), … | Move the state write after the guard and after the command is issued, mirroring set_gimbal_pitch: delete line 2710, and set `self.state.zoom = z` immediately before … |
| M19 | Medium | `manual_nudge_pilot.py`:297 | manual_nudge_pilot marks the aircraft LANDED the instant Landing() is queued — even when the call raises — and cleanup then refuses to land | Operator presses `l`/原地降落, or the deadman fires (`run_deadman` -> `pilot.land("deadman_no_ui_heartbeat")`), and the Landing command is lost, … | Mirror `OlympeLiveBackend._land_and_confirm`: issue `self.drone(Landing() >> FlyingStateChanged(state="landed", _timeout=20))`, wait, and set `self.safety.landed = True` … |
| M20 | Medium | `cruise_geofence.py`:236 | cruise_geofence leaves the aircraft airborne: connect/TakeOff run outside the try/finally and SIGINT is only hooked after takeoff (SIGTERM/SIGHUP … | Run with SFM_ALLOW_LEGACY_FLIGHT=1 (the gate at lines 224-229) and either (a) takeoff does not confirm hovering within 12 s, or (b) the operator … | Move `drone = olympe.Drone(...)`, `drone.connect()` and the TakeOff into the `try:` block, initialise `drone = None`/`airborne = False` before it, set `airborne = True` … |
| M21 | Medium | `olympe_live_backend.py`:3409 | poll() overwrites terminal/abnormal tracker_state (LAND_UNCONFIRMED, SOURCE_FAIL, TAKEOFF_FAIL, EMERGENCY_MANUAL) with the raw firmware flying state | Any of: a landing that does not confirm touchdown; a piloting-source handoff that fails (PCMD silently blocked); a takeoff that does not reach … | Invert the rule: keep a small allow-list of tracker_state values that firmware telemetry MAY overwrite (e.g. only the plain flight-phase values the poll itself … |
| M22 | Medium | `production_edm_tracker.py`:651 | No inlier spatial-distribution acceptance gate: inlier_grid_cells is computed and logged but never compared to any threshold, and no test covers a … | The query frame is dominated by one repeated/planar structure (a billboard, a pole face, a corridor panel) so all surviving PnP inliers fall in one … | Add `min_inlier_grid_cells` to EDMConfig (and the equivalent to ProductionConfig/_gate for XFeat), validate it in EDMConfig.validate(), and reject in pose_passes_quality … |
| M23 | Medium | `test_flight_safety_gates.py`:115 | test_fly_is_the_only_arming_entrypoint's TakeOff assertion is vacuous: both sides of the equality are 0 | Any future commit that adds an arming call to a non-fly() function using the same multi-line `drone(\n TakeOff() >> ...)` formatting as the existing … | Count the message token itself rather than a formatting-dependent call prefix, and assert the count is non-zero so the check can never go vacuous again: `for token in … |
| M24 | Medium | `path_follow_flight.py`:1144 | SafetyMonitor stall watchdog forces zero PCMD forever but never escalates to LAND — a hung localizer leaves the aircraft airborne indefinitely | hooks.get_pose() (production_edm_tracker / pycolmap PnP / CUDA) blocks permanently — CUDA context hang, driver Xid, pdraw decoder deadlock, or any … | In the stall branch record the first stalled timestamp (e.g. self._stall_since = now when the branch is first entered, cleared in the else at 1157), and once (now - … |
| M25 | Medium | `path_follow_flight.py`:1114 | SafetyMonitor performs blocking Olympe waits (Landing, setPilotingSource) while holding _io_lock, so operator EMERGENCY is unseen for up to 10 s and … | Operator commands LAND (file/keyboard) while the radio link is degraded or the aircraft does not acknowledge Landing. _attempt_callback blocks ~10 s, … | Give _await_confirmed_action a bounded wait: add a `timeout_s: float | None = None` parameter and call `wait(_timeout=timeout_s)` when it is set, then have … |
| M26 | Medium | `olympe_live_backend.py`:2101 | take_pc_control samples stick deflection only before a blocking 10 s piloting-source handoff, and the stick-reclaim path is disabled for that whole … | Operator presses 取得電腦控制 / 恢復電腦控制 and then deflects the SkyController sticks before the setPilotingSource expectation resolves — a window of tens of … | Re-validate stick activity after the handoff, inside the same lock that grants authority: at line 2096-2101 change to `with self._lock: if self._cleanup_done or … |
| M27 | Low | `flight_operator_app.py`:2029 | _readline_with_timeout discards every byte after the first newline, silently dropping complete worker responses | Two or more complete response lines resident in the worker's stdout pipe when the client performs a read. With the synchronous … | Give each client a persistent read buffer: store the leftover bytes on the instance (e.g. `self._stdout_tail: bytearray`, reset in `_spawn`/`_restart`) and seed … |
| M28 | Low | `olympe_live_backend.py`:2529 | _ensure_nudge_loop's is_alive() early return races with the exiting loop's tail, leaving a held nudge with no streaming thread and no deadman | Operator presses a new direction key in the sub-millisecond window between the hold loop clearing `_nudge_held` on deadman expiry (2543, lock … | Make the handoff explicit instead of relying on `is_alive()`. Give the loop a generation token: `_ensure_nudge_loop` should compare a `self._nudge_loop_generation` … |
| M29 | Low | `path_follow_flight.py`:89 | Hardcoded absolute path /home/allen/足球場 in the production flight module silently overrides the default reloc bundle and MegaLoc cache for every … | Run `定位演算法/flight_control/passive_flight_session.py` (or any importer of path_follow_flight) without … | Delete FOOTBALL_FIELD_ROOT/BUNDLE/MEGALOC (lines 89-95) and the two conditional branches that reference them, leaving the workspace-relative fallbacks only; and make the … |
| M30 | Low | `site_profile.py`:357 | Site-profile flight envelope parameters (speed limit, max pose jump, max route deviation, lookahead) are validated only as "> 0" with no upper bound, … | A hand-edited or newly authored schema-v2 site profile with a typo or an over-optimistic value in flight.controller -- e.g. speed_limit_mps 3.0 … | Add explicit ceilings next to the existing three, derived from the approved envelope, e.g. `speed_limit_mps <= 2.0`, `max_pose_jump_map_units <= 5.0`, … |


---

### 3.3 被複驗**推翻**的候選（16 項，記錄以免重複調查）

複驗者在這些項目上找到了上游 guard、或證明只存在於 legacy／研究路徑。摘要：

- run_loop 的 `int(v)` 對 NaN 會拋出 → 但 pose 在 1388 行已做 `pose_finite` 檢查，
  且控制器 `ControlConfig.__post_init__` 全參數 finite 驗證，路徑不可達。
- 「PCMD 未限幅」→ `_raw_pcmd` 有 `_clamp_pct`，`fly()` 的 sink 亦有 [-100,100] 限幅。
- 「reconnect 後自動回到 CRUISING」→ **根本沒有自動 reconnect**（`_connect()` 只呼叫一次）。
- 「兩個執行緒同時送 PCMD」→ 全部經 `SafetyMonitor._io_lock` 單一權威。
- 「geofence polygon 演算法錯誤」→ 生產路徑根本沒有 polygon geofence（改列為架構缺口，見 §2.2）。
- 其餘為 legacy `autoflight.py` / `cruise_geofence.py`（`SFM_ALLOW_LEGACY_FLIGHT` 鎖住）
  與 benchmark 腳本內的問題，不在生產路徑。

---

---

## 4. Safety Invariants — 實際驗證結果

| # | 不變量 | 保證？ | 證據 |
|---|---|---|---|
| 1 | 無有效定位時不得輸出巡航命令 | **是** | `path_follow_flight.py:1439` `if low_conf or not fresh:` → zero PCMD。故障注入 3/4/5/12 通過 |
| 2 | 定位超時後必須在限定時間內停止前進 | **是** | `POSE_STALE_S=0.5s` 內以 last_good 續航，之後 hover；`LOST_LAND_S=4.0s` 降落。實測續航 ≤0.10s |
| 3 | manual override 永遠高於 autonomous | **是** | `run_loop:1289` MANUAL 完全不送；`SafetyMonitor._run:1134` `_desired_valid=False`。故障注入 10 通過 |
| 4 | 飛控 disconnect 後不得繼續更新任務狀態 | **是（生產路徑）** | `send_authorized` 全域 try/except；故障注入 9 通過 |
| 5 | NaN／Inf 不得進入控制器 | **是** | `run_loop:1388` `pose_finite` 檢查；`ControlConfig.__post_init__` 全參數 finite 檢查 |
| 6 | geofence 外移命令必須被拒絕 | **部分** | 只有航道偏離 `MAX_ROUTE_DEVIATION_U`，且是「當前點」判定、無不確定性 margin、無 polygon |
| 7 | landing 狀態不得發出巡航命令 | **是（依賴呼叫端）** | `run_loop:1544` 立即 break；但控制器本身無 latch |
| 8 | reconnect 後不得自動恢復巡航 | **是（因為根本沒有自動 reconnect）** | `olympe_live_backend.py:670` `_connect()` 只呼叫一次，無重連迴圈 |
| 9 | 任務成功不得在降落前回報 | **是** | `_land_and_confirm()` 只在收到 landed 事件後回報成功 |

---

## 5. Test Results

### 5.1 既有測試（基準）

```bash
/home/allen/localization/.venv/bin/python -m pytest -q
```
| | 稽核前 | 第一輪後 | 第二輪後 |
|---|---|---|---|
| passed | 819 | 820 | **832** |
| failed | 1 | 1 | 2 |
| skipped | 1 | 1 | 1 |
| 耗時 | 17.9 s | 18.0 s | 19.4 s |

新增 13 個回歸測試。第二輪的第 2 個 failed 是 A10（讀主機真實搖桿裝置的環境相依測試），
**已在 pristine `git HEAD` 上重現，與本次任何修改無關**；稽核開始時它通過、之後失敗，
因為期間主機上沒有可列舉的 `/dev/input/js*`。

唯一失敗：`test_operator_render_perf.py::test_each_control_tab_fits_the_minimum_window_without_clipping`
（見 §3 的 UI 裁切 finding，稽核前即存在，屬未提交的 UI 改動）。
唯一 skip：`test_xfeat_wrapper_optimizations.py`（XFeat torch_hub_cache 為選配，未隨包提供）。

其他已執行：
```bash
python 定位演算法/flight_control/path_follow_flight.py --selftest   # OK
python 定位演算法/validation/check_runtime_mirrors.py               # OK: 8 compatibility pairs
python -m pip check                                                # No broken requirements
```

### 5.2 靜態分析

專案 `requirements-test.txt` 已釘選 `ruff==0.16.1` 但未安裝於 runtime venv。
為避免污染飛行 runtime，稽核在**獨立 venv** 安裝 ruff 執行，未改動 `.venv`。

```bash
ruff check --select E9,F63,F7,F82,F811,F841,B006,B008 定位演算法 控制介面程式 tools 執行環境
```
結果：**8 個 F841（未使用區域變數），0 個 E9／F63／F7／F82／F811／B006／B008。**
即：無語法錯誤、無未定義名稱、無可變預設參數、無 `except:` 裸捕捉（E722 = 0）。
以 8 萬行安全關鍵程式而言，這是很乾淨的結果。

未安裝／未執行（記錄為限制）：mypy、pyright、bandit、coverage、pytest-timeout。

### 5.3 端對端定位效能（真實 GPU、真實地圖、真實影片）

```bash
python 定位演算法/validation/benchmark_edm_site_replay.py \
  --site-profile 控制介面程式/site_profiles/urai_edm.json \
  --video 模擬器/測試影片/P0240024_720p.mp4 \
  --max-frames 400 --require-cuda --out .../replay_urai_400.json
```
硬體：RTX 5060 Laptop 8 GB / CUDA / `production_edm_..._torch_fp16_coarse3225`

| 指標 | 值 |
|---|---|
| 定位成功率 | **390/400 = 97.5%** |
| end-to-end wall p50 | **21.58 ms**（46.3 FPS） |
| end-to-end wall p90 | **23.89 ms**（41.9 FPS） |
| end-to-end wall p99 | **26.82 ms**（37.3 FPS） |
| wall max | 848.3 ms（第 1 幀，模型 warmup） |
| 超過 15 FPS 預算(66.7 ms) 的幀 | **2 / 400**（皆為 warmup 幀） |
| 超過 20 Hz 控制週期(50 ms) 的幀 | 2 / 400 |
| matching p50 / p90 | 18.64 / 20.71 ms（主要成本） |
| PnP p50 / p90 | 1.70 / 2.58 ms |
| processing FPS | 38.95 |
| 冷啟 startup | 11.30 s |
| inliers p50 | 273（門檻 `track_min_inliers=50`） |
| reproj_rms p50 / p95 / max | 1.35 / 1.71 / 1.88 px（門檻 6.0） |
| 被拒幀 | 10，全部為 `limited_jump_unconfirmed`（跳變閘門正常運作） |
| ref feature cache 命中率 | 99.25% |

**結論：15 FPS 目標達成且有大量餘裕，p99 亦然。**警訊只有一個：**冷啟第一幀 848 ms**，
所以起飛前必須先完成 warmup（現行起飛閘門要求 `< 1.0 s` 的新鮮影像，但不保證 tracker 已 warm）。

### 5.4 故障注入（section 10，全部 mock，無真機）

harness：`run_loop` + `LoopHooks`（虛擬時鐘），與既有 `test_flight_safety_gates.py` 同一手法。

| # | 注入 | 系統實際反應 | 預期安全反應 | 結果 |
|---|---|---|---|---|
| 1 | 每 20 幀丟 1 幀 | 以 last_good 續航，任務完成 | 單幀遺失可被 `POSE_STALE_S` 覆蓋 | PASS |
| 2 | 影像串流停 2 s | 整段 zero PCMD，未降落 | zero PCMD（2s < `STREAM_LOST_LAND_S=15s`） | PASS |
| 3 | 定位延遲 50→800 ms | `pose_age > 0.5s` 後 0 次驅動，`localization lost -> land` | 逾期即停止驅動 | PASS |
| 4 | PnP success 但 pose 含 NaN | NaN 從未驅動；以 last_good 續航 0.10 s 後降落 | NaN 不得steer，續航有上限 | PASS |
| 5 | inliers 崩到 10（退化） | `low confidence -> land`，其後 0 次非零 PCMD | hover → land | PASS |
| 6 | 單次 5 u 位置跳變 | `jump_reject_u` 觸發 1 次，未被平滑 | 拒絕而非平滑 | PASS |
| 7 | 地圖尺度錯 10 倍 | `route deviation 3.30u > 3.0u -> land` | 航道閘門攔截 | PASS |
| 8 | 時鐘倒退 5 s（負 dt） | 無 crash、無超界 PCMD | 降級處理 | PASS |
| 9 | 飛控連線中斷（send 拋例外，生產接線） | `send_authorized` 吸收，迴圈存活 | 不得因例外死亡 | PASS |
| 10 | 巡航中 manual override | 221 個 MANUAL tick，**0 次送命令** | 自動端完全靜默 | PASS |
| 11 | 航道邊界抖動 | 首次越界即 `-> land`，不震盪 | 立即終止 | PASS |
| 12 | 定位 worker 每幀拋例外 | 視為 no-fix，續航 ~0 s 後降落 | 有界續航 → land | PASS |
| 13 | `force_relocalize` hook 拋例外 | **未捕捉例外，run_loop 中止** | 不應中止控制迴圈 | **FAIL** |

12/13 安全。第 13 項為確認缺陷（見 §3）。

未能在本環境注入的項目（記錄為限制）：GPU OOM 真實觸發、磁碟寫入失敗、
真實 Olympe reconnect、真實 SkyController 接管延遲。

---

## 6. Fixes

兩輪修正，共 **10 個 High 已修**。**每一項都先驗證「移除修正 → 新測試失敗；加回修正 → 通過」**，
避免加入空轉測試。測試數：baseline 819 passed → **832 passed**（+13 個新回歸測試）。

### 第一輪（稽核當下）

| ID | 檔案 | 修正 | 對應測試 |
|---|---|---|---|
| A01 | `olympe_live_backend.py` `_ensure_nudge_loop._loop` | 迴圈本體包 `try`，例外時清空 hold；zero／HOVER 收尾移入 `finally`；zero 失敗改記 `pcmd_zero_failed` 而非 `pass` | `test_hold_loop_error_drops_the_hold_and_hovers` |
| A02（=F03/F12/F14） | `olympe_live_backend._execute_runtime_safety_action` | RTH 與 Landing 都失敗時解除 latch 並記 `runtime_safety_rearmed`，讓下次輪詢重試；成功路徑語意不變 | `test_failed_safety_landing_rearms_instead_of_disabling_protection` |
| A03 | `edm_localizer_adapter.py` | `_last_info` 補 `"weak"`，由 `state_in ∈ {WEAK_TRACK, LOST}` 導出，讓 `SFM_GATE_WEAK` 真正生效 | `test_adapter_reports_degraded_state_fixes_as_weak_for_the_flight_gate` |

### 第二輪（使用者指定 #1 #2 #3 #7 #8 #9 #10）

| # | ID | 檔案 | 修正 | 對應測試 |
|---|---|---|---|---|
| 1 | F11 | `production_edm_tracker.localize` | LOST 重定位不再無界。新增 `elif self.st.center is not None:` 分支：prior 仍新鮮（`lost_prior_max_age_s=3.0s`）時，重定位必須落在 `acquire_max_jump_factor × max_jump` 內且偏航變化 ≤ `acquire_max_yaw_diff_deg=90°`，否則 `rejected="acquire_jump"`／`"acquire_yaw"`。prior 過期後仍允許全域重定位（長時間 LOST 不會被鎖死） | `test_lost_reacquisition_rejects_a_teleport_beyond_the_acquire_bound`、`..._rejects_an_impossible_heading_flip`、`..._within_the_bound_is_still_accepted`、`..._is_unbounded_once_the_prior_expires` |
| 10 | F07 | 同上（limited_jump 確認段） | 二次確認必須來自**不同** capture：新增 `independent_capture = pending_stamp is not None and capture_stamp > pending_stamp`。重跑同一幀不再能自我確認 | `test_limited_jump_cannot_be_confirmed_by_relocalizing_the_same_capture` |
| 2 | F01 | `flight_operator_app.LiveWorkerClient._loop` | 用 worker 自己的 `seq` 做請求／回應配對：第 k 個請求的回應必須是 `seq == k-1`，不符即 `_WorkerResponseDesync`，**絕不發布**該 payload | `test_stale_response_after_a_refused_restart_is_never_published_as_a_new_fix` |
| 3 | F02 | 同上 + `_restart(force=True)` | 配對失序時強制重生 worker（**繞過 30 s cooldown**），因為父行程的共享記憶體 slot 記帳此時已落後 worker 一格，繼續寫就會覆寫對方正在讀的 slot | 同上（測試同時斷言 pid 有變） |
| 7 | F08 | `olympe_live_backend.take_pc_control` | 在阻塞式 piloting-source 交接**前**捕捉 `_pulse_token`，交接後比對；不同即拒絕解除 PCMD 封鎖並記 `safety_latched_during_handoff` | `test_pc_control_refuses_to_unblock_pcmd_if_safety_latched_during_handoff` |
| 8 | F09 | `olympe_live_backend._evaluate_video_health` | 串流恢復健康時解除 `_stream_failure_latched` 並記 `stream_health_rearmed`；同一次中斷仍只交接一次 | `test_stream_health_rearms_after_recovery_so_a_second_freeze_still_hands_off` |
| 9 | F10 | `flight_operator_app.main` | `--live` + `--replay-json` + `--no-live-localize` 直接 `SystemExit`，且**在連線前**就拒絕 | `test_real_flight_interface_refuses_a_replay_driven_hud` |

### 依使用者指示的兩項政策調整

| 項目 | 變更 | 理由 |
|---|---|---|
| GPS／RTH 起飛條件 | `read_connection_inventory` 把「機上失聯政策已確認」與「Home Point 可用」拆開。**只有前者是起飛必要條件**；沒有可用 home 時不再擋起飛，改記 `lost_link_fallback: land_in_place` | 使用者：「GPS 飛機上是有，只是常常訊號不好，所以不一定要」 |
| 失聯 fallback | `ending_behavior=landing` + auto_trigger 仍為必要條件，確保「失聯一定會自己下來」；沒有 home 就原地降落。`_execute_runtime_safety_action` 本來就已是 RTH 失敗→`land_cmd` | 使用者：「RTH 的部分如果沒辦法就採取原地降落」 |
| 地圖尺度 | **移除**原報告「必須量化地圖公制尺度」的建議 | 使用者：「整個地圖不需要尺度」。所有新增門檻都刻意設計成尺度無關：`acquire_max_jump_factor` 是 `max_jump` 的倍數，偏航是角度，過期是秒 |

測試：`test_weak_gps_does_not_block_takeoff_and_arms_land_in_place`、
`test_unconfirmed_lost_link_policy_still_blocks_takeoff`（確保放寬 GPS 沒有連帶放寬自動降落保證）。

### 設定檔連動

新增的三個 tracker 欄位是 **scale-free**，所以每個場域同值：

```text
acquire_max_jump_factor = 2.0   acquire_max_yaw_diff_deg = 90.0   lost_prior_max_age_s = 3.0
```

`edm_profile.py` 的 schema 是 fail-closed 的（缺欄位就拒絕載入），所以五個 EDM profile
與 `EDM_REQUIRED_TRACKER_KEYS` 都一起更新，三個 site profile 的
`asset_sha256.localizer_profile` 重新釘選。各場域實際生效的界線：

| profile | max_jump | acquire 界線 |
|---|---|---|
| `edm_production_profile`（urai） | 2.000 | 4.000 u |
| `river_site` | 0.760 | 1.521 u |
| `football_field` | 0.736 | 1.472 u |

### 定位品質回歸驗證

同一支影片、同一個 bundle、固定 PnP seed，修正前後 400 幀 replay：

| | 成功率 | inliers p50 | reproj p50 | 拒絕 |
|---|---|---|---|---|
| 修正前 | 390/400 (97.5%) | 273 | 1.343 | `limited_jump_unconfirmed: 10` |
| 修正後 | 390/400 (97.5%) | 273 | 1.343 | `limited_jump_unconfirmed: 10` |

**逐幀接受／拒絕與 inlier 數完全一致（0 幀差異）**——新的 LOST 界線只在重定位時才作用，
不影響正常追蹤。（wall 時間 21.6→26.6 ms 是 GPU 熱漂移，非程式碼；品質指標是決定性的。）

### 未做的事（刻意）

- 沒有降低任何既有定位品質門檻，只增加了新的拒絕條件。
- 沒有把真正的錯誤改成 warning。
- 沒有刪除任何資料集、地圖、模型、實驗結果或原始日誌。
- 沒有動 `.venv`（ruff 裝在獨立 venv）。
- 沒有解除 `--fly` 的外部審核鎖。
- 沒有動 `example_site_edm.json`（模板，其 SHA 欄位刻意為 null）。

### 第三輪（真機已連線；使用者：「把要修的部分全部修掉」）

飛機已接上，但**全程未連線送命令、未起飛**；驗證仍只用 mock。

| ID | Sev | 檔案 | 修正 | 對應測試 |
|---|---|---|---|---|
| F04 | High | `path_follow_flight.SafetyMonitor` | 新增 `_send_counted()`：**只有送出成功才蓋** `last_pcmd_call_mono_ns`，並累計 `pcmd_send_failures` / `last_pcmd_send_error`，首次與每 20 次印警告。9 個原本 `try/stamp/send/except: pass` 的站點全部改用它；失效計數也進 `pcmd_timing_snapshot()` → 命令日誌。**鏡像檔已同步** | `test_dead_pcmd_channel_is_counted_and_never_stamped_as_delivered`、`test_successful_pcmd_stamps_the_wire_marker` |
| F05 | High | `olympe_live_backend.__init__` | deadman TTL 加上**上限** `NUDGE_TTL_MAX_S=2.0`（安全逾時往短的方向夾是安全方向），被夾時明確印出，不靜默 | `test_nudge_deadman_ttl_is_bounded_above` |
| F06 | High | `production_localizer_factory.build_production_localizer` | 沒有 profile 就 **fail closed**（dataclass 預設是 target_site 尺度，別的場域會靜默拿到鬆很多的門檻，且完全跳過 SHA 驗證）。要用預設必須明確 `allow_profile_defaults=True` | `test_localizer_factory_refuses_unverified_default_thresholds` |
| F13 | High | `olympe_live_backend` takeoff／land／`send_pcmd`／`nudge_begin` | 新增 `_maneuver_in_progress`：自動起飛／降落進行中**拒絕非零 PCMD 與新的 hold**；zero 仍放行（只會強化 hover）。降落未確認時**釋放**此鎖，避免操作員被困在只能懸停 | `test_motion_is_refused_while_an_automated_maneuver_is_in_flight`、`test_unconfirmed_landing_releases_the_maneuver_block` |
| M-geofence | Medium | `_evaluate_runtime_safety` | 無 GPS／無 Home 時距離圍籬其實**什麼都沒檢查**，UI 卻顯示已啟用。新增 `state.distance_guard_active` 與 `distance_guard` 事件，狀態轉換時記錄。**因為本次放寬了 GPS 起飛條件，這項變得更重要** | `test_distance_geofence_reports_itself_inactive_without_gps` |
| M-state | Medium | `poll()` | 韌體飛行狀態輪詢會覆蓋終端／異常狀態，抹掉「起飛被擋／降落未確認／來源交接失敗」的唯一畫面紀錄。改用 `_TRACKER_STATE_STICKY`，把 `TAKEOFF_FAIL`／`TAKEOFF_BLOCKED`／`SOURCE_FAIL`／`LAND_UNCONFIRMED` 納入保護 | `test_telemetry_poll_does_not_erase_a_terminal_state`（4 參數） |
| M-zoom | Medium | `set_zoom` + `submit_current_frame_for_localization` | `state.zoom` 原本在守門與命令**之前**就寫入；而 `state.zoom` 又是定位是否暫停的依據 → 相機拒絕的 zoom 會讓定位在實際 1.0x 下被靜默停掉。改成命令成功才提交；並把暫停狀態寫進 `loc_health="PAUSED_ZOOM"`，不再只靠一行會捲掉的 log | `test_rejected_zoom_is_not_recorded_as_applied`、`test_zoom_blocked_by_sticks_is_not_recorded_as_applied` |
| M-config | Medium | `olympe_live_backend.__init__` | `nudge_pct` 完全沒有範圍檢查（`--nudge-pct 100` = 滿舵，不是 nudge）。改為 `[1, NUDGE_PCT_MAX=25]` 之外直接 `ValueError` fail closed | `test_nudge_authority_outside_the_envelope_is_rejected`（5 參數） |

測試：832 → **851 passed**。`check_runtime_mirrors.py` OK、`--selftest` OK。

### 第四輪（遙控器已開機;使用者指定的 7 項 + 手動／自動權限政策）

| ID | 檔案 | 修正 | 對應測試 |
|---|---|---|---|
| M | `run_loop`（鏡像同步） | 跳變二次確認新增 `pending_jump_stamp`,確認必須來自**不同 capture**。這是第二輪在 tracker 修掉的 F07 在飛控層的孿生版本 | `test_pose_jump_cannot_be_confirmed_by_relocalizing_the_same_capture` |
| M | `SafetyMonitor._run`（鏡像同步） | stall watchdog 不再永遠壓零:持續 `WATCHDOG_LAND_S=5.0s` 後 `_latch_terminal("LAND")`。否則飛機會一直懸停到電池耗盡 | `test_stall_watchdog_escalates_to_land_instead_of_hovering_forever` |
| M | `take_pc_control` | 阻塞交接**後**重新取樣搖桿;交接期間飛手抓桿不再被忽略 | 既有 stick 測試 + 新 epoch 測試 |
| M | `env_bool` | 改為 **fail closed**:無法辨識的值直接 `SystemExit`。`SFM_REQUIRE_GPS_FOR_GEOFENCE=ture` 以前會靜默關掉 GPS 前置條件 | — |
| M | `manual_nudge_pilot.land` | `Landing()` 排隊 ≠ 觸地。改為等 `FlyingStateChanged(state="landed")` 確認後才標記 `landed` | — |
| M | `manual_nudge_pilot._pulse` | pulse 執行緒加 `try/finally`:回 hover 的收尾不再被例外跳過 | — |
| M | `live_non_map_acceptance` | 新增 `Report.skip()`(`ok=None`,印 `[SKIP]`)。`--no-fly` 的 5 個空中檢查不再記成 PASS;`C_gimbal_zoom` 改為實際比對 pitch/zoom;`E_nudges` 改用 `nudge_begin()` 的回傳值,全被拒絕不再算 PASS | — |
| M | `test_fly_is_the_only_arming_entrypoint` | 原斷言比對 `drone(TakeOff`,該樣式**從不出現**(實際是 `TakeOff() >> FlyingStateChanged(...)`),所以一直在比 0 == 0。改為比對 `TakeOff()`/`Emergency()`/`Landing()` 並要求非零 | 該測試本身 |

**手動／自動權限政策(使用者 2026-08-06 定案)：**
手動飛行**不需要**通過定位安全審核 —— 飛手就是控制迴路,定位只是顯示。
切換到自動飛行則必須通過**全部**定位安全審核。

程式現況已符合前半:`_takeoff_preflight` 完全不依賴定位(電量／連線／磁力計／磁碟／
韌體限制／庫存),定位 worker 建不起來也只是 UI 顯示錯誤,不擋起飛與點動。
後半新增 `runtime_safety.autonomous_arming_blockers()`,把自動化前置條件寫成單一
fail-closed 閘門:autonomous 未解鎖、worker 未就緒、profile 未驗證、zoom 暫停、
狀態非 TRACK、pose 過期、inliers 不足、reproj 超標、連續好定位不足 2 次 —— 任一
不成立即拒絕,**證據缺失也視為拒絕**。`manual_flight_blockers()` 恆回空清單,並有測試
鎖住「手動不得被定位條件擋下」這條政策。

**PCMD 方向對應驗證(bench 用):** 新增 15 個測試,走真實 `nudge_begin` →
`_combined_nudge_pcmd` 路徑,驗證 12 個方向的 PCMD 正負號、對角不得夾帶 yaw、
授權不超過 `nudge_pct`、放開歸零、前後同按互相抵消。

測試:851 → **889 passed**。

### 第五輪（實機地面驗證發現，2026-08-06）

**這是唯一一個靠實機硬體發現、四輪程式碼稽核都抓不到的缺陷。**

```text
ID:              H01
Severity:        Medium（安全方向正確，但實務上讓 PC 控制無法使用）
Status:          Fixed（現場複測確認）
Component:       操作 UI 真機後端 / SkyController 搖桿override
File:            控制介面程式/operator_interface/olympe_live_backend.py
Line:            169（axes_active）／2108（_maybe_reclaim_from_sticks）
```

**Description**：`axes_active()` 原本是「**任一**軸超過 deadzone 即判定飛手要接管」。
Parrot SkyController 3 v1.8.1 共有 **6 支軸**，實機地面實測（`tools/stick_override_bench.py`）：
軸 0–3 是兩支飛行搖桿，**軸 4–5 是雲台滾輪與肩鍵**。因此在 PC 持有控制權期間，
**只要轉動雲台調整鏡頭，飛行控制權就會被搶回搖桿**。

**為何程式碼稽核抓不到**：Linux joystick 驅動只回報通用的 `ABS_X/Y/Z/RX/RY/RZ`
軸碼（已用 `JSIOCGAXMAP` 確認），**沒有任何語意名稱**可供靜態判讀。哪一支軸是雲台、
哪一支是飛行桿，只能靠實機撥動測量。

**Impact**：失效方向是安全的（交還給人），但操作上 PC 控制形同不可用——操作員
每次微調鏡頭就掉出 PC 控制，且畫面上看不出原因。PC 持有控制權時遙控器的雲台輸入
本來就不會傳到飛機，所以這個觸發**純粹只有副作用、沒有任何作用**。

**Fix**：新增 `_STICK_FLIGHT_AXES = (0, 1, 2, 3)`，`axes_active()` 只檢查飛行軸；
`stick_override` 的 log 也只記飛行軸。保留 `flight_axes=None` 可回到舊行為（供未來
不同機型）。

**Verification**：`test_camera_axes_do_not_seize_flight_control`（相機軸滿舵不得觸發、
四支飛行軸各自都要觸發、deadzone 以下不算、`flight_axes=None` 回舊行為）——已驗證
移除修正即失敗。**現場複測**：只動雲台 → 不觸發；撥搖桿 → 立即觸發。

**順帶更正的量測值**：SkyController 3 滿舵實測為 **±28715**（非 ±32767），
因此 deadzone 2000 相當於**可用行程的 7%**，不是 6%。

### 第五輪同時完成的實機唯讀量測

| 項目 | 實測值 | 說明 |
|---|---|---|
| MaxTilt | **20°**（韌體可到 40°） | **我們的程式從不設定它**，只讀取 |
| MaxVerticalSpeed | **2.0 m/s**（可到 4.0） | 同上 |
| MaxRotationSpeed | **20°/s**（可到 200） | 同上 |
| NoFlyOverMaxDistance | **0（關閉）** | 機上硬距離圍籬目前未啟用 |
| 串流 | 18.7 fps+、`ntp-mapped` 拍攝時戳、frame age p99 63 ms、凍結誤報 0 | 稽核的時戳／凍結偵測疑慮，真機資料皆正常 |

> **H02（High）已於 2026-08-06 修正**：`_configure_firmware_limits_locked()` 現在把
> `MaxTilt` / `MaxVerticalSpeed` / `MaxRotationSpeed` 一併**設定並回讀確認**，與
> `MaxAltitude` 走同一條 fail-closed 路徑（bounds 檢查 → 下指令 → 回讀比對，任一步
> 失敗即 `_firmware_config_ok=False`，起飛被 `_takeoff_preflight` 擋下）。預設值
> `20° / 2.0 m/s / 20°/s` **就是本次實測、已驗證可用的封套，所以飛行手感不變** ——
> 改變的只是「從此是釘住並驗證過的，而不是繼承上一次 FreeFlight 的殘留值」。
> 新增 `--max-tilt-deg` / `--max-vertical-speed-ms` / `--max-rotation-speed-degs`
> 與對應 `SFM_*` 環境變數，非有限或 ≤0 直接 `ap.error`。
> 測試：`test_speed_envelope_is_pinned_and_read_back_not_inherited`（飛機帶著
> 40°/4.0/200 進來，設定後必須變成預設封套）、`test_speed_envelope_readback_mismatch_is_fail_closed`、
> `test_invalid_speed_envelope_is_rejected`（4 參數）、
> `test_operator_speed_envelope_defaults_match_backend`（UI 與 backend 常數不得漂移）。
>
> 原始問題描述（保留）：`MaxTilt` / `MaxVerticalSpeed` / `MaxRotationSpeed`
> 全域搜尋確認**沒有任何一處設定它們**，只有 `MaxAltitude` / `MaxDistance` /
> `NoFlyOverMaxDistance` 有設定並回讀確認。這三個決定操作員每一次點動的實際速度，
> 目前完全依賴機上殘留值（可能來自 FreeFlight 或上一次 session）。若有人把 MaxTilt
> 調成 40°，同樣的 8% 指令會變成 3.2° 傾角、加速度加倍，而程式不會察覺也不會阻止。
> 建議比照 MaxAltitude 在起飛前設定並回讀確認。

**nudge_pct=8 在 MaxTilt=20° 下的實際輸出**：單軸 1.60°（水平加速度約 0.27 m/s²、
垂直 0.16 m/s）、雙軸 1.20°、三軸 1.00°、轉向 4.0°/s。
**水平是加速度不是速度**：按住 3 秒約 0.8 m/s，放開才會煞停回懸停。

## 7. Remaining Risks（本環境無法驗證）

| 風險 | 為何無法在此驗證 |
|---|---|
| 真機 Olympe reconnect／stale command 行為 | 2026-08-06 真機已接上，但**串流探針連不上**：Olympe 經 USB 找到 `Skycontroller 3`（`192.168.53.1` ping 得到）卻在 45 s 後 `ConnectionState.Timeout`，且沒有其他行程佔用。最可能是飛機未開機／未與遙控器配對。待排除後可用 `outputs/audit_20260805/live_stream_probe.py` 重跑 |
| SkyController 實體接管延遲與優先權 | 遙控器接上後 `/dev/input/js*` 已出現，A10 那個測試**現在會通過**——直接證實它的結果取決於當下有沒有插搖桿 |
| GPU OOM 真實觸發下的 CUDA 快取釋放 | 需刻意耗盡 8 GB VRAM，可能影響主機穩定性 |
| 磁碟寫入失敗（ENOSPC）對安全日誌與起飛閘門的影響 | 需填滿磁碟 |
| GIL-holding CUDA hang 是否真的餓死 SafetyMonitor | 需真實 CUDA 掛起；記憶中已知為殘留風險，SkyController 是真正的後盾 |
| `MAX_ROUTE_DEVIATION_U = 3.0` 對應多少公尺 | 地圖是 scale-free SfM 單位。依使用者決定**不做公制換算**，航道界線就以地圖單位定義與驗收；本次新增的門檻也全部設計成尺度無關 |
| A04 的 inlier_ratio 門檻在其他兩個場域的誤拒率 | 只跑了 urai 一支影片 400 幀 |
| 其餘 2 個場域（river_site、football_field）的 replay 驗證 | football_field 依專案紀錄本來就「尚未 replay 驗證」 |
| 真實鏡頭凍結／解碼失敗／解析度中途變更 | 需真實 PDRAW 串流 |

---

## 8. 建議的下一步驗證順序

原第 1 項「量化地圖公制尺度」**已依使用者指示移除**（地圖不需要尺度；新增門檻皆為尺度無關）。

1. **修剩餘 4 個 High 中最容易驗證的兩個**，兩者都可用現有 mock 手法驗證、不需真機：
   - **F13**：起飛／降落進行中仍接受 nudge/PCMD（缺 maneuver-in-progress 閘門）。
   - **F05**：`--nudge-pulse-s` / `NUDGE_PULSE_S` 沒有上限，一個過大的值會讓
     「UI 凍結會自然衰減成 hover」的保證失效。加上範圍檢查即可。
2. **F04**：`SafetyMonitor` 吞掉所有 PCMD 送出失敗且無計數器，並在嘗試**之前**就蓋上
   `last_pcmd_call_mono_ns`，導致命令通道死掉時看不出來、飛行日誌還記錄了從未上線的命令。
   建議加失敗計數 + 時戳移到成功之後。
3. **F06**：EDM runtime profile 在多處是「可選」的；缺席時 localizer 會靜默套用未經驗證的
   硬編碼門檻並跳過 profile SHA 驗證。建議改為 fail-closed（本次已證明 schema 檢查本身
   是有效的——三個新欄位立刻被擋下並要求更新所有 profile）。
4. **A04（inlier 空間分布閘門）三場域校準**：對 river_site、football_field 各跑一次 replay，
   確認 `inlier_ratio >= 0.35` 的誤拒率（urai 實測 min=0.571，誤拒 0 幀），再決定是否啟用。
5. **A10 測試去硬體化**：`test_connect_keeps_skycontroller_sticks…[True-True-STICKS]` 會讀
   主機真實 `/dev/input/js*`，插不插搖桿會讓它時綠時紅。mock 掉裝置探測，讓「測試全綠」
   重新成為可信的發布訊號。
6. **在 Sphinx 或拆槳狀態下驗證本次的 7 項修正**——目前全部只在 mock／replay 驗證過。
   特別是 F08（緊急停止期間搶 PC 控制權）與 F09（第二次串流凍結）需要人在場實際觸發。
7. 以上完成後再重跑：完整套件 + 故障注入 + 三場域 replay，才討論解鎖 `--fly`。

## 附錄 A · 本次稽核產出的證據檔

```text
outputs/audit_20260805/
├── README.md                     重跑方式與 commit 綁定
├── fault_injection.py            Section-10 故障注入 harness（真實 run_loop，無真機）
├── fault_injection_results.txt   13 情境實測輸出
├── replay_urai_400.json          400 幀 replay receipt（含四項 SHA-256 + PnP seed）
└── verified_findings.json        44 CONFIRMED / 16 REFUTED 完整複驗結果
```

> `python tools/workspace_audit.py --strict-output-names` 目前會對這個新目錄回報
> `WARN: outputs/audit_20260805`。**這是該工具正常運作**——`classify_output()` 的
> 設計意圖（見 `tools/test_workspace_audit.py::test_output_classification_keeps_new_names_visible`）
> 就是讓未知的 output 家族保持可見，直到有人刻意分類它。稽核**刻意不**把 `audit_`
> 加進 `OUTPUT_EVIDENCE_PREFIXES` 來讓警告消失；請團隊自行決定是否納入白名單，
> 或將本目錄移入既有的 evidence 前綴。

---

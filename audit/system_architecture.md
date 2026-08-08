# audit/system_architecture.md — 系統架構、模組依賴與資料流

稽核日期：2026-08-07
基準：`agent/localization-runtime-optimizations` @ `d0b2250`（工作區 dirty，63 檔變更）

本文件所有圖與表皆由**實際程式碼**推導（AST import 圖、`grep` 送出點、實跑 probe），
非由 README 或註解推導。與程式不符之處以程式為準。

---

## 0. 一句話架構

單一 Tk 程序（operator UI）作為 orchestrator，向下透過**注入的 backend 介面**操作
無人機（Olympe）或模擬 backend；定位在**獨立子程序**中執行，以 stdio JSON + shared
memory 與 UI 交換；自主航線飛行目前**硬鎖**，唯一能命令真機的路徑是操作員手動點動。

---

## 1. 模組依賴圖（實測，無環）

以 AST 解析 `控制介面程式/`、`定位演算法/flight_control/`、
`定位演算法/deploy_code/sfm_glomap_deploy/`、`tools/` 內全部非測試 `.py` 得出。

**結果：0 個 circular import。依賴圖是乾淨的 DAG。**

```mermaid
graph TD
    subgraph L4["L4 應用／入口"]
        FOA["flight_operator_app<br/>7800 行 · UI 主程序"]
        LLW["live_localizer_worker<br/>938 行 · 定位子程序"]
        MP["mission_pipeline"]
        ODW["object_detector_worker"]
    end

    subgraph L3["L3 服務"]
        OLB["olympe_live_backend<br/>4357 行 · DroneAdapter"]
        REW["route_editor_window"]
        SAP["site_assets_panel"]
        PFF["path_follow_flight<br/>2443 行 · 自主飛行(鎖)"]
    end

    subgraph L2["L2 領域"]
        LSA["local_site_assets"]
        RPFC["real_path_follow_controller"]
        PLF["production_localizer_factory"]
        OFS["olympe_frame_source"]
        MNP["manual_nudge_pilot"]
        PXT["production_xfeat_tracker"]
        ELA["edm_localizer_adapter"]
    end

    subgraph L1["L1 契約／基礎"]
        BC["backend_contract<br/>typed ControlRequest/Result"]
        RS["runtime_safety"]
        SP["site_profile<br/>fail-closed 驗證"]
        PT["pose_types<br/>Pose/Localizer"]
        WL["workspace_layout"]
        AI["artifact_integrity"]
    end

    FOA --> OLB & REW & SAP & LSA & RPFC & BC & RS & SP & WL
    LLW --> BC & RS & PLF & PXT & PFF & PT & WL & ELA
    OLB --> BC & RS & MNP & OFS & WL
    REW --> LSA & RPFC
    SAP --> LSA
    PFF --> OFS & PLF & RPFC
    PLF --> ELA & PXT & AI
    PXT --> PT & AI
    MNP --> OFS
    RS --> BC
    LSA --> SP
```

**分層評價**：層次清楚，箭頭一律由上往下。`olympe_live_backend` **不 import**
`flight_operator_app`——`DroneState` 以 `state_factory` callable 注入（`olympe_live_backend.py:642`）。
這是正確的依賴反轉。

**扇入最高的模組**（被依賴次數）：`site_profile` 6、`real_path_follow_controller` 6、
`olympe_frame_source` 6、`workspace_layout` 5、`backend_contract` 5、`pose_types` 5。
這些是事實上的穩定介面層，符合預期。

**唯一結構異味**：`olympe_frame_source -> autoflight`——影像來源模組依賴自主飛行模組，
方向與其他邊相反。屬 P3。

---

## 2. 系統元件圖（程序與行程邊界）

```mermaid
graph LR
    subgraph P1["程序 1：operator UI（Tk 主執行緒）"]
        UI["OperatorApp<br/>3993 行 · 106 方法 · 144 屬性"]
        BE["backend<br/>OlympeLiveBackend 或 DroneBackend(sim)"]
        LWC["LiveLocalizerClient"]
        LDC["LiveDetectorClient"]
    end

    subgraph P2["程序 2：定位 worker"]
        LW["live_localizer_worker<br/>EDM / XFeat tracker"]
    end

    subgraph P3["程序 3：物件偵測 worker"]
        OD["object_detector_worker"]
    end

    subgraph EXT["外部"]
        ANAFI["ANAFI 無人機<br/>Olympe 8.4.0"]
        SC["SkyController 3<br/>/dev/input joystick"]
        VID["影片檔（模擬模式）"]
    end

    UI -->|"tk .after 迴圈"| BE
    BE -->|"PCMD / TakeOff / Landing"| ANAFI
    ANAFI -->|"telemetry / PDRAW 視訊"| BE
    SC -->|"搖桿軸值"| BE
    VID -->|"FFmpegFrameStream"| UI
    UI <-->|"stdio JSON + shared_memory"| LW
    UI <-->|"stdio JSON"| OD
```

---

## 3. 相機影像 → 定位輸出路徑

```mermaid
sequenceDiagram
    participant ANAFI as ANAFI PDRAW / 影片檔
    participant GRAB as frame grabber 執行緒
    participant UI as OperatorApp.tick (Tk)
    participant SHM as shared_memory (2 槽)
    participant W as live_localizer_worker 子程序

    ANAFI->>GRAB: 原始畫格
    GRAB->>UI: 最新畫格（queue maxsize=1，丟舊留新）
    UI->>SHM: 寫入 RGB 至槽 n
    UI->>W: stdio JSON 請求 + 槽索引 + client_submit_mono
    W->>W: EDM/XFeat 比對 + PnP
    W->>UI: stdio JSON 結果（dict）
    UI->>UI: normalize_live_localization_result（非有限 XYZ → success=False）
    UI->>UI: annotate_ui_arrival_timing（貼上 ui_arrival_mono_ns）
    UI->>UI: TemporalPoseStabilizer（中位數 + 低通 + 速率限制）
    UI->>UI: LostHoldPolicy（連續失敗 → hold/hover）
```

**關鍵事實**：
- 畫格 queue `maxsize=1`（`flight_operator_app.py:2059`）——**天然丟舊留新，不會累積 backlog**。
- 定位結果帶有 monotonic 時間戳：`client_submit_mono`、`client_response_mono`、
  `source_frame_stamp_mono`、`frame_callback_enter_mono_ns`、`ui_arrival_mono_ns`。
- 定位結果**是無型別 dict**，非 typed 物件（見 findings F-05）。
- `pose_types.Pose` 是給飛控邊界用的 typed 契約（x/y/z/yaw/stamp，明確標注 raw GLOMAP
  座標、stamp 為 monotonic 永不用 wall clock），但**不用於 UI↔worker 這條路徑**。

---

## 4. UI → 控制命令 → SDK 路徑（含仲裁點）

```mermaid
graph TD
    K["鍵盤 22 個方向鍵"] --> OKP["_on_nudge_key_press"]
    B["虛擬搖桿 / 微移鈕"] --> ONB["_on_nudge_btn_press"]
    BTN["飛行按鈕<br/>起飛/降落/懸停/手動"] --> SEND["OperatorApp.send()"]

    OKP --> BCMD["_backend_command()"]
    ONB --> BCMD
    SEND -->|"live 且屬 async_commands"| DLC["_dispatch_live_command<br/>背景執行緒 + _flight_inflight 去重"]
    SEND -->|"其他"| BCMD
    DLC --> BCMD

    BCMD -->|"建立 ControlRequest<br/>human_origin=True"| TC["OlympeLiveBackend.command()"]
    TC --> TYPED["_typed_command()"]
    TYPED -->|"TAKEOFF 且非人類來源"| REJ1["rejected HUMAN_ORIGIN_REQUIRED"]
    TYPED -->|"START_AUTO"| REJ2["rejected LOCKED_EXTERNAL_APPROVAL"]
    TYPED -->|"EMERGENCY_STOP"| FS["fail_safe()"]
    TYPED -->|"其餘"| LEG["legacy 字串 dispatcher"]

    LEG --> TO["takeoff_cmd()"]
    LEG --> LD["land_cmd()"]
    LEG --> HV["hover_cmd()"]
    LEG --> NB["nudge_begin/end"]

    NB --> NL["nudge-hold-loop 執行緒 20Hz"]
    NL --> SPC["send_pcmd()"]
    HV --> SPC
    TO --> RAW
    SPC --> RAW["_raw_pcmd()<br/>唯一 PCMD 送出點"]
    FS --> GTP["give_to_pilot()<br/>pilot_sticks=True 閂鎖"]
    GTP --> RAW
    RAW --> DRONE["drone(PCMD(...))"]

    style RAW fill:#2d5a2d,color:#fff
    style GTP fill:#7a2d2d,color:#fff
```

### 4.1 命令仲裁 — **系統的最大優點**

**PCMD 只有一個實際送出點**：`olympe_live_backend.py:2229 _raw_pcmd()`，其中
`self.drone(PCMD(1, r, p, y, g, 0))` 是全庫唯一一行真正送出 PCMD 的程式。

進入該點只有兩條路：

| 路徑 | 用途 | 閘門 |
|---|---|---|
| `send_pcmd()` (`:2239`) | 一般運動命令 | `self._lock` 下檢查：`_cleanup_done`、`_landed`、`pilot_sticks`、`_maneuver_in_progress`（非零命令時） |
| `_zero_pcmd_or_log()` (`:2200`) | 安全歸零 | 刻意繞過上述閘門——歸零必須永遠送得出去 |

**優先級鏈（實測正確）**：
`fail_safe()` → `give_to_pilot()` → `pilot_sticks = True`（持鎖）+ 歸零 PCMD。
一旦 `pilot_sticks` 為真，`send_pcmd` 對**所有**後續命令回傳 False（`:2243-2246`）。
`auto_resume=False` 明確記錄，不會自動恢復；必須操作員按「恢復電腦控制」。

→ **emergency 確實具有最高優先級，且無法被一般命令覆蓋。**

### 4.2 三套彼此獨立的「模式」詞彙

| 概念 | 欄位 | 取值 | 位置 |
|---|---|---|---|
| 控制權 | `DroneState.mode` | MANUAL / PC_CONTROL / AUTO / CLOSED | UI + backend |
| 定位品質 | `RuntimeState.mode` | TRACK / WEAK_TRACK / LOST | `production_xfeat_tracker.py` |
| 自主飛行授權 | `SafetySwitch.mode` | AUTO / HOVER / MANUAL / LAND / EMERGENCY | `path_follow_flight.py:273` |

三者皆為裸字串、皆無 Enum、無共用型別，且都叫 `.mode`。互不衝突（分屬不同類別），
但命名碰撞使跨模組閱讀困難。見 findings F-08。

---

## 5. 手動接管與緊急停止路徑

```mermaid
stateDiagram-v2
    [*] --> PC_CONTROL: take_pc_control() 確認成功
    PC_CONTROL --> MANUAL: Esc 鍵
    PC_CONTROL --> MANUAL: 實體搖桿動作<br/>(SkyControllerStickMonitor)
    PC_CONTROL --> MANUAL: fail_safe(任何原因)
    PC_CONTROL --> MANUAL: 視訊逾時 stream_lost_hover
    MANUAL --> PC_CONTROL: 操作員按「恢復電腦控制」<br/>且搖桿未動
    PC_CONTROL --> [*]: cleanup（關窗/訊號/atexit）
    MANUAL --> [*]: cleanup

    note right of MANUAL
      pilot_sticks = True
      send_pcmd() 一律 False
      無自動恢復
    end note
```

**四條獨立的接管觸發**（皆導向同一個 `give_to_pilot()`）：
1. Esc 鍵 → `send("manual")`
2. 實體搖桿移動 → `SkyControllerStickMonitor._on_stick_active`
3. `fail_safe(reason)` → 連線失效／視訊逾時／緊急停止
4. `cleanup()` → 關窗、Ctrl+C、SIGTERM、SIGHUP、atexit

---

## 6. 啟動與關閉流程

```mermaid
graph TD
    S["啟動.sh"] --> RD["resolve_display.sh"]
    RD --> SAL["start_anafi_live.sh<br/>preflight 檢查"]
    SAL --> MAIN["flight_operator_app.main()"]
    MAIN --> CFG["site_profile 載入 + fail-closed 驗證"]
    CFG --> NG["install_network_guard()<br/>monkey-patch socket"]
    NG --> SL["SessionLogs.create()<br/>4 條 JSONL 串流"]
    SL --> BEC["backend 建構 + 連線"]
    BEC --> WRK["定位 worker 子程序啟動"]
    WRK --> APP["OperatorApp 建構<br/>__init__ 279 行 · 144 屬性"]
    APP --> SIG["atexit + SIGINT/SIGTERM/SIGHUP 註冊<br/>失敗會明確警告"]
    SIG --> ML["mainloop()"]

    ML --> CLOSE["關閉"]
    CLOSE --> OC["_on_close: backend.cleanup() → session_logs.close() → destroy()"]
    OC --> FIN["mainloop finally:<br/>cleanup / localizer.close / detector.close / session_logs.close"]
    FIN --> EXEC{"requested_site_profile?"}
    EXEC -->|是| RESTART["closerange(3,4096) + execv 重啟"]
    EXEC -->|否| END["結束"]
```

**關閉路徑評價：完整且有多重保險。**
- `cleanup()` 自述 idempotent（`:4274`），且被 `_on_close`、`finally`、`atexit`、訊號處理
  多次呼叫皆安全。
- 訊號處理有 `exit_in_progress` Event 防重入。
- **訊號註冊失敗會明確印出警告**（`:7744-7749`）——「關終端機會降落」若不成立，畫面會說。
  這是很好的誠實設計。

---

## 7. 執行緒與程序圖

```mermaid
graph TD
    subgraph MAIN["UI 主程序"]
        TK["Tk 主執行緒<br/>tick 100ms / poll_localization_results 105ms<br/>_check_active_site_route 1200ms"]
        NLT["nudge-hold-loop<br/>daemon · 20Hz"]
        SCM["SkyControllerStickMonitor<br/>daemon"]
        RSA["_runtime_safety_action_thread"]
        OUC["olympe-ui-&lt;command&gt;<br/>每個 async 命令一條 · daemon"]
        WCL["LiveWorkerClient._loop<br/>每 worker 一條 · daemon"]
        NAL["_nal_thread（FFmpeg NAL pump）"]
        SAI["site-asset-import · daemon"]
    end
    subgraph SUB["子程序"]
        LWP["定位 worker"]
        ODP["偵測 worker"]
    end
    TK --> NLT & OUC & SAI
    WCL <--> LWP
    WCL <--> ODP
    SCM -->|"stick active → give_to_pilot"| TK
```

| 執行緒 | daemon | 停止機制 | 例外處理 |
|---|---|---|---|
| `nudge-hold-loop` | 是 | `_nudge_loop_stop` Event | `try/except/finally`，例外時清空 hold 並歸零 PCMD，記錄 `nudge_loop_error` |
| `SkyControllerStickMonitor` | 是 | `stop()` | 斷線回報 `_on_stick_monitor_disconnect` |
| `olympe-ui-<cmd>` | 是 | 自然結束 | 例外經 `_flight_results` queue 回主執行緒 |
| `LiveWorkerClient._loop` | 是 | `_stop_process()`：terminate→wait 1s→kill→wait | 有重啟 + warmup 節流 |
| `SafetyMonitor`（自主，鎖住） | 是 | `terminated` Event | `_run_loop` 111 行 |

**正面**：`nudge-hold-loop` 的 deadman 計時器放在迴圈**內部**，且迴圈死亡時 `finally`
會清空 hold 並歸零——這正是「執行緒死掉導致最後一筆非零命令持續」的正確解法，
且程式碼註解明確說明了此設計理由（`:3022-3029`）。

---

## 8. 錯誤傳播與恢復流程

```mermaid
graph TD
    E1["Olympe 例外"] --> FS["fail_safe()"]
    E2["視訊逾時 > stream_loss_grace_s"] --> SLH["stream_lost_hover()"] --> FS
    E3["連線中斷"] --> FS
    E4["nudge 迴圈例外"] --> Z["清空 hold + 歸零 PCMD + 記錄"]
    E5["定位 worker 崩潰"] --> WR["LiveWorkerClient 重啟 + warmup 節流"]
    E6["磁碟臨界"] --> TB["封鎖起飛"]
    E7["設定驗證失敗"] --> NOGUI["不建立 GUI/worker，啟動即失敗"]

    FS --> GTP2["give_to_pilot: pilot_sticks 閂鎖<br/>auto_resume=False"]
    GTP2 --> HUMAN["等待操作員手動恢復"]
```

**錯誤分類實況**：系統**沒有**統一的例外型別階層。失敗以三種方式混用表達：
布林回傳、`None`、以及例外。`ControlResult` 是唯一結構化的失敗表達，但只覆蓋
typed command 路徑，且該路徑存在遺失布林的缺陷（findings F-01）。

`ruff` 統計：全庫 364 個 `BLE001` blind-except、104 個 `S110` try-except-pass。
安全關鍵檔案的分布：

| 檔案 | BLE001 | S110 |
|---|---|---|
| `olympe_live_backend.py` | 81 | 18 |
| `flight_operator_app.py` | 41 | 12 |
| `path_follow_flight.py` | 37 | 9 |
| `runtime_safety.py` | 4 | 0 |

全庫**無**裸 `except:`。多數 blind-except 位於日誌／telemetry 讀取等非控制路徑，
且多半有記錄；但數量本身使「哪些例外被刻意吞掉」難以審閱。見 findings F-06。

---

## 9. 模組責任表

| 模組 | 責任 | 輸入 | 輸出 | 對外介面 | 生命週期 | 執行緒 | 逾時 | fallback | 責任重疊 |
|---|---|---|---|---|---|---|---|---|---|
| `OperatorApp` | UI 建構、telemetry 輪詢、定位結果處理、地圖/視訊算繪、命令送出、日誌、狀態追蹤 | 鍵鼠、backend state、worker 結果 | Tk 畫面、backend 命令 | `send()`、`_backend_command()` | 程序生命週期 | Tk 主執行緒 + 多條 daemon | worker timeout | — | **嚴重過載：7 種責任於單一 class** |
| `OlympeLiveBackend` | 連線、telemetry、視訊、PCMD、起降閘門、地理圍欄、安全閂鎖、磁羅盤校正、RTH、日誌 | ControlRequest、搖桿、Olympe 事件 | PCMD、TakeOff/Landing、DroneState 更新 | `command(ControlRequest)` | connect→cleanup（idempotent） | 主 + nudge loop + stick monitor | 多處 | 歸零 PCMD | **過載：10 種責任** |
| `backend_contract` | 型別化命令契約 | — | `ControlRequest/Result`、Enum | dataclass + Enum | 無狀態 | — | — | — | 無（乾淨） |
| `runtime_safety` | session 日誌、磁碟保留、離線網路守衛、自主 arming 閘門 | 快照 dict | JSONL、blockers list | 函式 + `SessionLogs` | session | — | — | fail-closed | **名實不符：4 種不相關責任** |
| `site_profile` | 場域設定載入與驗證 | JSON | frozen dataclass | `load()` | 啟動一次 | — | — | **無 fallback（正確）** | 無（乾淨） |
| `live_localizer_worker` | 定位子程序 | stdio JSON + shm 畫格 | stdio JSON 結果 | 程序邊界 | 由 client 管理 | 子程序 | client 端 timeout | 重啟 | 無 |
| `path_follow_flight` | 自主航線飛行（**硬鎖**） | 航線、pose | PCMD | `fly()`（立即 SystemExit） | — | SafetyMonitor 執行緒 | 多處 | hover/land | 過載但目前不執行 |
| `SafetySwitch` | 檔案／鍵盤安全開關 | `/tmp/sfm_drone_safety.cmd`、stdin | AUTO/HOVER/MANUAL/LAND/EMERGENCY | `poll()` | 自主飛行期間 | 呼叫端輪詢 | mtime 比對 | **一律 fail-closed 到 HOVER** | 無 |

---

## 10. 「能否替換實作」評估

| 要替換的東西 | 是否可行 | 證據 |
|---|---|---|
| **定位 provider** | **可以** | `production_localizer_factory.py` + `SFM_LOCALIZER_BACKEND` + `resolve_localizer_backend()`；worker 是獨立程序，介面是 stdio JSON。EDM↔XFeat 已實際共存 |
| **地圖 provider** | **大致可以** | `local_site_assets.LocalSitePackageProvider` + `site_asset_interfaces.py` 有抽象；但 `read_ply_points` / `read_map_points` 直接寫在 UI 檔內（`flight_operator_app.py:2729-2809`），換格式要動 UI |
| **drone adapter** | **可以** | `backend_contract` 定義 typed 介面；模擬 `DroneBackend` 與 `OlympeLiveBackend` 已並存；Olympe import 全部延遲在函式內，模組可在無 Olympe 環境 import（實測通過） |
| **UI** | **困難** | 業務邏輯與 Tk widget 在 `OperatorApp` 內深度交織（144 個實例屬性），無 ViewModel 層 |

---

## 11. 架構總評

**強項（實測確認）**
1. PCMD 單一送出點，閘門集中，emergency 具真正最高優先級且會閂鎖。
2. 依賴圖無環，分層清楚，SDK 以延遲 import 隔離。
3. `backend_contract` 是設計良好的 typed 命令契約（request_id / timestamp / human_origin）。
4. `site_profile` 嚴格 fail-closed 驗證，拒絕未知欄位與非有限數值。
5. 關閉路徑多重保險且 idempotent；訊號註冊失敗會明說。
6. `SafetySwitch` 有 stale-AUTO 拒絕（mtime 早於 run start 即拒）。
7. 鏡像重複檔以 `check_runtime_mirrors.py` 機械化強制一致，且測試會跑。
8. runtime 程式碼無機器特定絕對路徑。

**弱項（實測確認）**
1. 兩個 God object：`OperatorApp`（3993 行/106 方法/144 屬性）、
   `OlympeLiveBackend`（3749 行/93 方法/113 屬性）。
2. 無顯式狀態機：`tracker_state` 有 26 個字串值，散落賦值，無 Enum、無 transition 驗證。
3. legacy dispatcher 遺失布林 → 被拒絕的飛行命令回報成功（連 forensic log 都寫成 accepted）。
4. UI↔worker 的定位 payload 是無型別 dict。
5. 約 50 個 `SFM_*` 環境變數形成繞過 `site_profile` 驗證的第二套設定通道。
6. 364 個 blind-except 使「哪些錯誤被刻意吞掉」難以審閱。

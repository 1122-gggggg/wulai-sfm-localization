# audit/refactor_plan.md — 重構與修正計畫

稽核日期：2026-08-07

## 指導原則（本計畫遵守）

1. **先建立測試，再修改行為。** 每一項都列出前置測試。
2. **不重寫整個系統。** 本計畫沒有任何「重寫 `OperatorApp`」之類的項目。
3. **不為了美觀改變已驗證行為。** PCMD 仲裁、emergency 閂鎖、fail-closed 設定驗證等
   經確認正確的部分**不動**。
4. **不引入大型框架**，不製造沒有實際需求的抽象層。
5. **不讓檔案數量無限制增加**，不把 manager 拆成更多 manager。
6. **`控制介面程式/SAFETY.md` 具約束力**：飛行命令與按鍵路徑的變更必須由操作員審過風險。
   本計畫將這類項目明確標示為「需操作員授權」，稽核者不自行施行。

---

## A. 立即修正（本次稽核已完成）

| # | 項目 | 收益 | 風險 | 影響模組 | 驗收條件 | 狀態 |
|---|---|---|---|---|---|---|
| A1 | 移除 `preview_site_route` 中 `return` 之後的死碼，並把日誌移到 return 前、改用作用域內存在的 `count` | 消除全庫唯一的 `F821` undefined-name；補上成功路徑的觀測性（原本只有失敗路徑有日誌） | 極低。僅新增一行操作員日誌，位於「僅顯示、不變更場域航線」的預覽路徑，非飛控路徑 | `flight_operator_app.py:4839-4848` | `ruff --select F821` 全庫零；`operator_interface/` 542 測試通過 | **已完成** |
| A2 | 補 autonomy arming 閘門的 NaN／Inf fail-closed 保護性測試（21 個雙側參數化案例 + bool 案例） | 釘住一個已正確但無測試的安全行為。非有限數在比較中會靜默取勝，是典型的靜默失效來源 | 無（純測試新增） | `test_autonomy_gate.py` | 該檔 21→43 測試全通過 | **已完成** |
| A3 | 補鏡像樹的分類強制與漂移偵測測試 | 讓「沒人分類這個同名檔」變成測試失敗而非事後發現；對 `manual_nudge_pilot.py` 實際做位元組比對 | 無（純測試新增） | `test_runtime_mirrors.py` | 該檔 1→3 測試全通過；且在加入分類前**實際失敗**過（非空洞性證明） | **已完成** |

**回歸結果**：`1070 passed, 1 skipped` → **`1094 passed, 1 skipped`**，零失敗。

---

## B. 飛行前必須修正（需操作員授權）

> 以下皆位於 `SAFETY.md` 涵蓋的路徑，或需要操作員決定嚴格程度。
> 建議以 **B1+B2 綁為同一批次**（見下方相依性說明）。

### B1 + B2（必須一起做）｜命令結果的真實性 + 跨執行緒 UI 更新

**相依性（重要）**：目前 F-01（丟棄布林）**正在遮蔽** F-33（背景執行緒操作 Tk）。
若只修 B1，會把大量拒絕路徑導入 B2 的壞掉呼叫，使「靜默誤報」升級為「UI 可能失去回應」。

| 項目 | 內容 |
|---|---|
| **預期收益** | 操作員能看到真正的拒絕原因；`commands.jsonl` 的 `control_result` 不再把被拒絕的起飛記成 `accepted=true`（恢復事故調查能力）；消除 Tk 執行緒違規 |
| **風險** | 中。改動飛行命令派送路徑。**不改變任何送出的命令內容**，只改變回傳值與顯示位置 |
| **影響模組** | `olympe_live_backend.py:3298-3406`（回傳布林）、`backend_contract.py:401-403`（`completed` 的預設）、`flight_operator_app.py:4253-4277`（移除背景執行緒上的 `write_log`）、`:4279`（`_finish_backend_command` 統一顯示） |
| **前置測試** | ①每個 legacy 分支：底層回傳 False → `ControlResult.accepted is False`；②`start_auto` 在 live 模式被拒時，`write_log` 必須發生在 Tk 執行緒（可用 `threading.current_thread() is threading.main_thread()` 斷言）；③現有 125 個 backend 安全測試必須全綠 |
| **遷移方式** | 分兩個 commit。先加測試（會失敗），再修實作。`ControlResult.completed` 增加 `strict` 參數，預設維持舊行為，新呼叫點顯式傳 `strict=True`，避免一次性改變所有既有語意 |
| **驗收條件** | 全套件綠；模擬模式下按「自主」按鈕，日誌出現「已拒絕 (LOCKED_EXTERNAL_APPROVAL)」且無 `RuntimeError`；`commands.jsonl` 中該事件 `accepted=false` |

### B3｜鍵盤焦點守衛（F-02）

| 項目 | 內容 |
|---|---|
| **預期收益** | 消除「在介面自己的欄位打字會對空中的飛機送出 PCMD」——本次稽核最接近實體風險的一項 |
| **風險** | 低—中。**不改變任何閘門或命令語意**，只在四個處理器前加焦點判斷。但屬 `SAFETY.md` 明列的「微移」路徑 |
| **影響模組** | `flight_operator_app.py` 的 `_on_nudge_key_press`、`_on_nudge_key_release`、`_hover_all_nudges`、`reset_camera_defaults`；日誌 `Text` 改 `state="disabled"` |
| **前置測試** | 焦點在 `ttk.Entry`／`tk.Text` 時發送 `<KeyPress-w>`，斷言**沒有** `nudge_begin`；焦點在主視窗時斷言**有**。需 X display（與現有 23 個 render-perf 測試同條件） |
| **遷移方式** | 單一 commit。務必同時確認「焦點在主視窗時所有 22 鍵仍正常」，避免修出「方向鍵全失效」這個更糟的結果 |
| **驗收條件** | 新測試通過；操作員在模擬模式實測 22 個方向鍵仍可用；點進輸入框打字不產生 nudge 事件 |

### B4｜真機啟動器補齊 preflight（F-03）

| 項目 | 內容 |
|---|---|
| **預期收益** | 讓命令真實飛機的路徑，其環境保證不低於只讀影片的路徑 |
| **風險** | 低（只增加失敗條件，不改飛行邏輯）。但會**讓某些目前能啟動的情境無法啟動**——需操作員決定嚴格程度 |
| **影響模組** | `控制介面程式/operator_interface/start_anafi_live.sh` |
| **前置測試** | 擴充 `test_start_anafi_live_launcher.py`：缺 Python 3.10、venv 不合規、`include-system-site-packages=true` 時必須拒絕啟動 |
| **遷移方式** | 先以「警告」形式上線一次現場驗證，確認不會誤擋，再改為硬性失敗 |
| **驗收條件** | 乾淨機器上啟動成功；刻意破壞環境時明確失敗並給出可執行的修正指令 |

### B5｜重建可重現的執行環境（F-04）

| 項目 | 內容 |
|---|---|
| **預期收益** | 讓「乾淨環境重建」與「現場實跑環境」變成同一個環境；所有已完成的飛行驗證才具備可轉移性 |
| **風險** | **高影響**（會取代操作員目前可用的 `.venv`），但可逆——先建到 `SFM_VENV_DIR` 的新目錄並平行比對，確認後才切換 |
| **影響模組** | `.venv`、`執行環境/requirements_runtime.txt`、`requirements-lock.txt` |
| **前置測試** | 在新 venv 上跑完整 1094 測試 + `驗證系統.sh` 15 步，與現行環境逐項比對 |
| **遷移方式** | ①把 `scipy`、`onnxruntime-gpu`、`av` 正式納入 requirements 並重產 lock；②移除 `lingbot-map` 死安裝；③`SFM_VENV_DIR=/tmp/sfm-clean bash tools/install_runtime.sh`；④平行比對；⑤確認後切換 |
| **驗收條件** | 新 venv 的 `pyvenv.cfg` 無 `include-system-site-packages`；`pip freeze` 與 lock 一致；1094 測試全綠；`驗證系統.sh` 15 步全過 |

### B6｜碰撞監控不得靜默降級（F-08）

| 項目 | 內容 |
|---|---|
| **預期收益** | 消除一個 fail-open：`status="OFF"` 目前與「附近沒有障礙物」無法區分 |
| **風險** | 低。只改回報字串與 preflight 日誌，不改判定邏輯 |
| **影響模組** | `real_path_follow_controller.py:660, 676`（`CollisionMonitor`）；`執行環境/requirements_runtime.txt` |
| **前置測試** | monkeypatch `cKDTree = None`，斷言 `update()` 回 `status="UNAVAILABLE"` 而非 `"OFF"`，且 `severity` 不是 0.0 這個「安全值」 |
| **遷移方式** | 與 B5 同批（scipy 入 lock）。檢查所有讀 `status` 的呼叫端能處理新值 |
| **驗收條件** | 新測試通過；preflight 日誌出現 collision monitor 狀態一行 |

### B7｜鏡像樹的所有權決定（F-06、F-07）

| 項目 | 內容 |
|---|---|
| **預期收益** | 消除「一棵樹修好、另一棵樹靜默過期」這整類風險 |
| **風險** | 中。`olympe_frame_source.py` 的合併屬行為性變更（影格發布時序） |
| **影響模組** | `定位演算法/flight_control/`、`定位演算法/deploy_code/sfm_glomap_deploy/` |
| **前置測試** | A3 已建立分類強制與漂移偵測。另需：兩份 `olympe_frame_source` 的**差集**行為測試（單調性守衛：送入時間戳倒退的影格，斷言不被發布） |
| **遷移方式** | 三步：①`manual_nudge_pilot.py` 決定「納入 git + MIRROR_PAIRS」或「刪除 deploy 副本」；②把單調性守衛與 mid-convert 丟棄移植到 deploy 副本；③由 authoritative `validation/check_runtime_mirrors.py` 維護明確的 enforced／divergent／transitional 分類 |
| **驗收條件** | `_same_named_files()` 全部落入 enforced 或有理由的 divergent；差集行為測試通過 |

---

## C. 短期重構（飛行後、下一個維護週期）

### C1｜`OperatorApp` 減負 — **只抽出無狀態邏輯，不新增 manager**

| 項目 | 內容 |
|---|---|
| **預期收益** | 3993 行 / 106 方法 / 144 屬性降到可審閱規模。**目標不是漂亮，是讓下一位工程師能在不讀完 7800 行的情況下安全修改 UI** |
| **風險** | 中。`tick`（351 行）是熱路徑，改動可能影響算繪節奏 |
| **影響模組** | `flight_operator_app.py` |
| **前置測試** | 現有 23 個 `test_operator_render_perf.py` 是唯一建構真實 `OperatorApp` 的測試——**必須先擴充**，否則重構沒有安全網 |
| **遷移方式（建議順序）** | ①先抽**純函式**（該檔已有 43 個模組層函式，是好的既有模式）：`_append_loc_metrics` 的 136 行字面 dict → 一個 `build_loc_metrics_record()` 純函式；②`render_map` / `render_video` → 一個無狀態的 `MapRenderer` / `VideoRenderer`（吃 `DroneState` 回 `Image`）；③定位 pipeline 編排 → `LocalizationPresenter`。**不要**把 `_build_ui` 的 564 行拆成 20 個 `_build_xxx`——那只是把長度換成跳轉次數 |
| **驗收條件** | 每一步後全套件綠；`OperatorApp` 方法數與屬性數實際下降；`_build_ui` 以外的最長方法 < 150 行 |

### C2｜`tracker_state` 收斂為 Enum + 單一寫入點（F-11）

| 項目 | 內容 |
|---|---|
| **預期收益** | 26 個散落的字串字面值變成一個 `StrEnum`；同時取得 §12 要求的 state-before／state-after 日誌 |
| **風險** | 低—中。`tracker_state` 目前主要是顯示與紀錄用途（真正的安全閂鎖是 `pilot_sticks` 與 `terminated`），所以改動不觸及安全語意 |
| **影響模組** | `flight_operator_app.py`（`DroneState`）、`olympe_live_backend.py`（賦值點） |
| **前置測試** | 先加一個「所有賦值都必須是 Enum 成員」的測試；再逐步替換 |
| **遷移方式** | `StrEnum` 保持與現有字串相等，因此既有比較與日誌格式**不變**，可漸進遷移。新增 `set_tracker_state(new, *, reason)` 集中賦值並記錄前後值 |
| **驗收條件** | 全套件綠；`grep 'tracker_state = "'` 歸零；incidents.jsonl 出現 state-before/after |

### C3｜設定單一 resolver（F-13）

| 項目 | 內容 |
|---|---|
| **預期收益** | 消除「約 95 個環境變數繞過嚴格 site profile 驗證」這個第二通道；啟動日誌可印出最終生效設定與 checksum |
| **風險** | 中。可能改變某些現場慣用的環境變數行為 |
| **影響模組** | `site_profile.py`、`flight_operator_app.py` 的 argparse、`workspace_layout.py` |
| **前置測試** | 對每個安全相關 env var，測其超範圍值必須被拒絕而非靜默採用 |
| **遷移方式** | 新增 `resolve_config()` 回傳 frozen dataclass + checksum。**先只做「印出最終設定」**，不改變任何解析行為，讓現場先看到實際生效值；下一步才加範圍檢查 |
| **驗收條件** | 啟動日誌有完整設定 dump 與 checksum；session manifest 記錄之 |

### C4｜定位 payload 型別化（F-14）

| 項目 | 內容 |
|---|---|
| **預期收益** | 跨程序邊界從 60 鍵的 raw dict 變成有欄位、單位、frame、timestamp、validity 的 frozen dataclass |
| **風險** | 低。**不動演算法**，只在邊界加一層驗證與轉換 |
| **影響模組** | `live_localizer_worker.py`（產生端）、`flight_operator_app.py`（消費端） |
| **前置測試** | `LocalizationResult.from_json` 對缺欄位／型別錯誤／非有限值的行為 |
| **遷移方式** | 先只在**消費端**加 `from_json` 驗證並保留 dict 相容存取；確認無迴歸後再收斂 |
| **驗收條件** | 全套件綠；worker 回傳缺欄位時有明確錯誤而非 `KeyError` |

### C5｜測試基礎設施（F-24）

安裝 `pytest-timeout`（全域逾時）與 `coverage`；把 43 個靜默條件式測試改為明確 `skipif`。
風險極低，收益立即。**建議與 B5 同批施行。**

---

## D. 長期改善

| # | 項目 | 說明 |
|---|---|---|
| D1 | `OlympeLiveBackend` 拆出起飛 preflight | 18 道閘門 + 140 行的 `_takeoff_preflight` 可獨立為 `TakeoffPreflight`，輸入快照、輸出 blockers list——**與 `autonomous_arming_blockers` 完全相同的成功模式**（純函式、易測、fail-closed）。這是本系統已被證明有效的設計，值得複製 |
| D2 | `runtime_safety.py` 依責任拆分 | 目前混合 session 日誌、磁碟保留、網路策略、arming 閘門四件無關事。拆成 `session_logs.py`、`disk_policy.py`、`network_policy.py`、`arming_gate.py`。**低風險但檔案數增加**，故列為長期 |
| D3 | `fly()` 抽出可測序列（F-05） | 見 findings F-05。解鎖自主飛行**之前**必須完成 |
| D4 | UI 使用自己宣告的 `OperatorBackend` protocol | 目前 UI 直接呼叫 backend 私有成員（`_flight_state_name` 等）。補齊 protocol 後 UI 才真正可替換 backend |
| D5 | 減少 blind-except（F-16） | 364 個 `BLE001` / 104 個 `S110`。**分批**把 `except Exception: pass` 換成具體例外 + `log.debug`，優先 `olympe_live_backend.py` 的 18 個。**不得**以 suppression 消警告 |

---

## E. 不建議現在修改

| # | 項目 | 理由 |
|---|---|---|
| E1 | **PCMD 單一送出點與其閘門** | 經確認正確。`_raw_pcmd` / `send_pcmd` / `_zero_pcmd_or_log` 的三層結構（含「歸零刻意繞過閘門」）是本系統最強的設計，且有 125 個整合測試。**不要動** |
| E2 | **`fail_safe` → `give_to_pilot` → `pilot_sticks` 閂鎖** | emergency 優先級的實作正確且有測試。不要為了「更漂亮的狀態機」改寫它 |
| E3 | **`site_profile.py` 的嚴格驗證** | fail-closed 且拒絕未知欄位，是設定管理的正面範例。不要為了「更寬容」放寬 |
| E4 | **升級 `protobuf` 3.19.4** | 3 個 CVE，但被 Olympe 8.4.0 的 wire schema 綁定，程式碼已註記此約束。系統離線、區域網路、無外部輸入，實際暴露面極低。**應記錄為已接受風險**，待 Olympe 升版時一併處理 |
| E5 | **把自主飛行解鎖** | 三重鎖（`SystemExit`、`build_controller` 的 `RuntimeError`、`AUTONOMOUS_ROUTE_EXTERNAL_APPROVAL_LOCKED`）目前是系統最重要的安全邊界。在 D3 完成、B1–B7 全數修畢之前不得解鎖 |
| E6 | **導入狀態機框架** | C2 的 `StrEnum` + 單一寫入點已足以解決實際問題。引入 `transitions` 之類的框架會增加相依與概念負擔，收益不明 |
| E7 | **把 `_build_ui` 拆成 20 個小方法** | 564 行確實過長，但拆成 20 個 `_build_xxx` 只是把「長」換成「跳轉」，對可讀性無淨收益。優先做 C1 的無狀態抽取 |
| E8 | **刪除 `flight_control/olympe_frame_source.py` 的 ~100 行死碼** | 已在 findings F-30 提出。但該檔正處於待合併狀態（B7），此時刪碼會使 diff 更難審。**B7 完成後再做** |

---

## F. 建議施行順序

```
1. A1–A3                              [已完成]
2. B1+B2（同批）→ 全套件回歸
3. B3            → 模擬模式實測 22 鍵
4. B5 + C5       → 乾淨環境重建 + 測試基礎設施
5. B4, B6, B7    → 啟動器、碰撞監控、鏡像樹
6. ── 此時可進行繫留真機測試 ──
7. C1–C4         → 下一個維護週期
8. D1–D5, D3     → 自主飛行解鎖的前置條件
9. ── D3 完成後才討論解鎖 fly() ──
```

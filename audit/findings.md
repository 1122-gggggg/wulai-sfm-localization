# audit/findings.md — 軟體工程與系統架構稽核發現

稽核日期：2026-08-07
基準：`agent/localization-runtime-optimizations` @ `d0b2250`（工作區 dirty，63 檔變更）
稽核性質：**軟體工程／架構**稽核。不重新驗證定位精度、建圖品質或演算法效能。
稽核期間**未連線真機、未起飛、未送出任何飛行命令**（遵守 `控制介面程式/SAFETY.md`）。

## 統計

| 嚴重度 | 數量 | 已修 | 未修 |
|---|---|---|---|
| **P0** | **0** | – | – |
| **P1** | 8 | 0 | 8 |
| **P2** | 19 | 0 | 19 |
| **P3** | 7 | 2 | 5 |
| 合計 | **34** | 2 | 32 |

> F-08（`scipy` 未 pin 導致碰撞監控靜默關閉）計為 P2，但**自主飛行解鎖後應升為 P1**。

> 已修的 2 項為 F-26（死碼 + undefined name）與 F-27（補上 NaN/Inf fail-closed 保護性測試）。
> 其餘均**刻意不修**：`控制介面程式/SAFETY.md` 規定飛行命令與按鍵路徑須由操作員審核風險後才可變更，
> 而多數 P1／P2 恰好落在該政策涵蓋的路徑上。詳見 `refactor_plan.md`。

**沒有發現 P0。** 具體而言，本次稽核**未**發現：無人機失控路徑、安全機制可被一般命令
覆蓋、emergency 無法搶占、舊控制命令持續發送、關閉系統後仍持續控制，或影響飛控的
race condition／deadlock。命令仲裁與緊急停止的設計是本系統最強的部分（見 §正面確認）。

P1 的共同性質是：**系統的實體行為是安全的，但它對操作員與對事故紀錄「說謊」**，
以及**執行環境無法由其自身工具重建**。

---

## 正面確認（先講，因為這決定了整體判定）

以下項目經實測確認為**正確且高於一般研究原型水準**，稽核不建議更動：

| # | 事實 | 證據 |
|---|---|---|
| G-1 | **PCMD 全庫只有一個實際送出點** | `olympe_live_backend.py:2235` 是唯一的 `drone(PCMD(...))`。入口僅 `send_pcmd()`（有閘門）與 `_zero_pcmd_or_log()`（刻意繞過，因歸零必須送得出去） |
| G-2 | **emergency 具真正最高優先級且會閂鎖** | `fail_safe()` → `give_to_pilot()` → 持鎖設 `pilot_sticks=True`；此後 `send_pcmd` 對所有命令回 False（`:2243-2246`）。`auto_resume=False`，必須人工恢復 |
| G-3 | **依賴圖無環** | AST 全掃 4 個來源樹，0 個 circular import。`olympe_live_backend` 不 import UI；`DroneState` 以 `state_factory` 注入 |
| G-4 | **SDK 被隔離** | Olympe 全部延遲 import；`olympe_live_backend` 可在無 Olympe 環境成功 import（實測） |
| G-5 | **設定驗證嚴格 fail-closed** | `site_profile.py` 拒絕未知欄位、非有限 JSON 常數、格式錯誤的 SHA-256，並做 cross-field frame-ID 比對 |
| G-6 | **關閉路徑多重保險且 idempotent** | `_on_close` + `mainloop finally` + `atexit` + SIGINT/SIGTERM/SIGHUP；訊號註冊失敗會明確警告操作員（`:7744-7749`） |
| G-7 | **nudge 迴圈 deadman 在迴圈內部，且迴圈死亡會歸零** | `:3022-3058` 的 `except/finally`，附有解釋此設計理由的註解 |
| G-8 | **stale AUTO 會被拒絕** | `SafetySwitch._read_file_command` 比對 mtime 與 run start，早於者拒絕（`:1657-1663`） |
| G-9 | **NaN/Inf 全面 fail-closed** | 20 項行為中覆蓋最好的一項；`autonomous_arming_blockers` 實測 18/18 非有限輸入全部阻擋 |
| G-10 | **命令契約有 request_id／timestamp／human_origin** | `backend_contract.ControlRequest`；起飛強制 `human_origin=True`，非人類來源直接 `HUMAN_ORIGIN_REQUIRED` |
| G-11 | **runtime 程式碼無機器特定絕對路徑** | 全庫僅 1 處 `/home/allen`，且是說明前次已移除的註解 |
| G-12 | **離線與網路策略以 monkey-patch 強制** | `install_network_guard`；worker 一律套用 simulated 策略，永遠無法對飛機說話 |

---

# P1 — 正式飛行前必須修正

---

## F-01｜被拒絕的飛行命令回報為成功，連事故紀錄也寫成 accepted

- **嚴重度**：P1
- **類別**：錯誤處理 / 介面契約 / 可觀測性
- **檔案**：`控制介面程式/operator_interface/olympe_live_backend.py:3298-3406`、`:3274-3275`；`控制介面程式/operator_interface/backend_contract.py:401-403`

### 問題描述

`OlympeLiveBackend.command()` 的 legacy 字串 dispatcher 對數個飛行命令**丟棄回傳的布林值**，
然後統一 `return self.state`。`ControlResult.completed()` 以 `accepted = raw_result is not False`
判定成功，而 `DroneState` 物件 `is not False`，因此**被拒絕的命令一律變成 `accepted=True`**。

### 實際證據

```python
# olympe_live_backend.py:3305-3310  ── 布林被丟棄
elif name == "hover":     self.hover_cmd("ui_hover")      # 回傳 False 時被丟棄
elif name == "land":      self.land_cmd("ui_land")
elif name == "takeoff":   self.takeoff_cmd()              # 18 道 preflight 閘門的結果被丟棄
...
elif name in {"nudge_begin", "nudge_press"}:  self.nudge_begin(d)
...
return self.state                                          # :3406

# backend_contract.py:401-403
accepted = raw_result is not False        # DroneState is not False → True
```

呼叫鏈：`_typed_command:3274` → `raw = self.command(name)` → 得到 `DroneState`
→ `ControlResult.completed(state, raw_result=<DroneState>)` → `accepted=True, executed=True, reason_code="OK"`。

**同一個函式內其他分支卻寫對了**：`land_now`（`:3313`）、`firmware_limits_apply`（`:3315`）、
`auto_speed_limit_apply`、`nudge_vector`（`:3393`）、`emergency_stop`（`:3311`）都正確回傳。
所以這是遺漏，不是設計。

### 可能後果

1. 操作員按「起飛」，18 道 preflight 閘門任一擋下，UI **不顯示拒絕原因**，飛機不動而操作員不知為何。
2. **`commands.jsonl` 的 `control_result` 事件寫入 `accepted=true, reason_code="OK"`**
   （`olympe_live_backend.py:3290-3296`）。事故後重建時序時，一次被拒絕的起飛在紀錄上
   與一次成功的起飛無法區分。這是對事故調查能力的直接損害。
3. `_backend_command`（`flight_operator_app.py:4269-4274`）本來**已經實作了**拒絕訊息
   （`f"{command}: 已拒絕 ({result.reason_code})"`），但因 `accepted` 恆真而永不觸發。
   修好上游即可讓既有機制生效。

### 重現方法

無需真機：

```python
# 以既有的 fake olympe 測試框架，令 takeoff_cmd 的任一 preflight 閘門失敗
result = backend.command(ControlRequest.from_legacy("takeoff", human_origin=True))
assert result.accepted is False   # 目前會失敗：實際為 True
```

### 建議修正

在 legacy dispatcher 中回傳既有的布林值：

```python
elif name == "hover":    return self.hover_cmd("ui_hover")
elif name == "land":     return self.land_cmd("ui_land")
elif name == "takeoff":  return self.takeoff_cmd()
elif name == "manual":   return self.give_to_pilot()
elif name in {"nudge_begin", "nudge_press"}:  return self.nudge_begin(d)
elif name in {"nudge_end", "nudge_release"}:  self.nudge_end(d); return True
elif name == "nudge_heartbeat":               return self.nudge_heartbeat(dirs)
```

並在 `ControlResult.completed` 加一個 `strict` 路徑，讓「未知回傳型別」不再默認為成功。

### 是否已修正

**否。** 本稽核**刻意不修**：`控制介面程式/SAFETY.md` 明文規定
「禁止隨意修改所有『按鍵／按鈕控制無人機飛行』的指令，尤其是起飛、原地降落」，
且要求「只有操作員明確要求並審過風險才可改上述路徑」。此修改雖不改變送出的命令內容，
仍位於該政策涵蓋的路徑上，應由操作員審核後施行。

### 剩餘風險

高（資訊面）。實體面 fail-safe（被拒絕＝飛機不動），但操作員心智模型與事故紀錄皆錯誤。

---

## F-02｜在介面自己的文字欄位打字會對空中的飛機送出 PCMD

- **嚴重度**：P1
- **類別**：UI 架構 / 安全操作性
- **檔案**：`控制介面程式/operator_interface/flight_operator_app.py:3143-3159`（22 個方向鍵繫結）、`:3957-3963`（`_on_nudge_key_press`，無焦點守衛）、`:3752`/`:3755`/`:3787`（三個 `ttk.Entry`）、`:3939`（`tk.Text` 未設 `state="disabled"`）

### 問題描述

22 個方向鍵繫結在 **toplevel** 上（`self.bind(...)`，`self` 即 `OperatorApp(tk.Tk)`）。
Tk 的 bindtag 鏈為 `widget → class → toplevel → all`，因此在 `Entry`／`Text` 中按鍵會
**同時插入字元並觸發 toplevel 的 nudge 處理器**。`_on_nudge_key_press` 沒有任何焦點檢查。

### 實際證據（本稽核實跑 probe，非閱讀推論）

```
Entry bindtags: ('.!entry', 'TEntry', '.', 'all')
Text  bindtags: ('.!text',  'Text',   '.', 'all')
在 Entry 打 'w' → entry 內容: 'w' | 觸發的處理器: ['TOPLEVEL_NUDGE_w']
在 Text  打 'w' → text  內容: 'w' | 觸發的處理器: ['TOPLEVEL_NUDGE_w']
```

受影響按鍵（皆為實際飛行方向）：
`w s a d q e r f u i o j l m , . 1 2 3 7 8 9`，另有 `0`/`Home` 重設雲台。

後端只在 `self._landed` 為真時拒絕（`olympe_live_backend.py:3072`）；**空中時 `nudge_begin` 會被接受**。

三個 `ttk.Entry` 皆**未**設為 disabled，標籤僅寫「限制（landed 才可套用）」——
意思是*套用*限 landed，欄位本身**隨時可編輯**。日誌 `tk.Text`（`:3939`）亦未 disabled，
且設了 `insertbackground`（游標顏色），可取得焦點。

### 可能後果

操作員在懸停中點進高度限制欄位預先輸入「82」→ `8`＝上、`2`＝下，飛機上升後下降。
或點進日誌區閱讀失敗原因，之後每一次按鍵都變成飛行命令，而該分頁上虛擬搖桿並不可見。

實體風險有界（KeyRelease 會歸零，且有 deadman），屬**短暫非預期運動**而非持續失控，
故評為 P1 而非 P0。但這是本次稽核中最接近實體風險的一項。

### 重現方法

啟動模擬介面 → 點進「高度」輸入框 → 輸入 `82` → 觀察 `commands.jsonl` 出現
`nudge_begin` 事件（模擬模式下不會有實體動作）。

### 建議修正

在四個處理器開頭加焦點守衛（不更動任何閘門或命令語意）：

```python
def _keyboard_is_for_flight(self) -> bool:
    w = self.focus_get()
    return not isinstance(w, (ttk.Entry, tk.Entry, tk.Text))
```
於 `_on_nudge_key_press`、`_on_nudge_key_release`、`_hover_all_nudges`、
`reset_camera_defaults` 開頭 early-return；並將日誌 `Text` 設為 `state="disabled"`，
在 `write_log` 內暫時切換。

### 是否已修正

**否。** 同 F-01，`SAFETY.md` 明文將「按鍵控制無人機飛行」列為需操作員審核的路徑
（「微移 | 方向鍵按住/放開 → nudge_begin/end | 放開不歸零 = 持續飛走」）。
本稽核提供 patch 建議但不自行施行。

### 剩餘風險

高。建議列為真機飛行前的首要修正項。

---

## F-03｜真機啟動器完全不跑 preflight，模擬啟動器卻跑完整 preflight

- **嚴重度**：P1
- **類別**：部署 / 一致性
- **檔案**：`控制介面程式/operator_interface/start_anafi_live.sh`（無 preflight）vs `控制介面程式/影片模擬串流/啟動.sh:240-247`

### 問題描述

兩個入口的防護程度**與風險相反**：

| 檢查 | 模擬入口 | **真機入口** |
|---|---|---|
| `simulator_preflight.py --check-runtime --full-runtime` | 有（5 個資產 SHA-256、CUDA、GPU 型號、bundle 載入、EDM+MegaLoc 模型載入、GUI/worker import） | **無** |
| 強制 CPython 3.10 | 有（`啟動.sh:231-234`） | **無版本檢查** |
| 拒絕 venv 外的 Python | 有（除非 `SFM_PORTABLE_ALLOW_EXTERNAL_PYTHON=1`） | 向上找三層 `.venv`，找不到就用 `PATH` 上的 `python3`（`:73`） |
| 固定 `SFM_TORCH_HUB_CACHE` / `SFM_LOCALIZER_PYTHON` | 有（`啟動.sh:29-60`） | 只固定 `SFM_WORKSPACE_ROOT`（`:18`）；其餘沿用操作員 shell 的殘留值 |

### 可能後果

命令真實飛機的路徑，其環境正確性保證**低於**只讀影片的路徑。操作員 shell 中殘留的
`SFM_LOCALIZER_PYTHON` 會被 `default_worker_python`（`flight_operator_app.py:131-141`）
直接採用，可能以錯誤的 Python／錯誤的模型快取啟動定位 worker，而無任何警告。

### 建議修正

在 `start_anafi_live.sh` 加入與模擬入口對等的最小檢查：Python 版本、venv 歸屬、
資產 SHA-256、以及 `SFM_*` 環境變數的顯式固定。真機路徑的檢查不應少於模擬路徑。

### 是否已修正

否（屬啟動腳本與環境政策，需操作員決定嚴格程度）。

### 剩餘風險

中高。

---

## F-04｜正式執行環境無法由其自身安裝腳本重建；系統套件以不同主版本滲入

- **嚴重度**：P1
- **類別**：可重現性 / 部署
- **檔案**：`.venv/pyvenv.cfg`、`tools/install_runtime.sh:35-39`

### 實際證據

```
$ cat .venv/pyvenv.cfg
home = /usr/bin
include-system-site-packages = true      ←
version = 3.10.12

$ sed -n '35,39p' tools/install_runtime.sh
if [[ -f "$venv_dir/pyvenv.cfg" ]] && grep -qi '^include-system-site-packages[[:space:]]*=[[:space:]]*true' ...; then
  echo "[runtime] 拒絕使用 include-system-site-packages=true 的環境" >&2
  exit 1
fi
```

安裝腳本**明文拒絕**這種 venv，因此**現行正式環境不可能由 `tools/install_runtime.sh` 產生**。
今天執行該腳本會直接 exit 1。

實測滲入情形：

```
yaml       → /usr/lib/python3/dist-packages/yaml        5.4.1  （lock 指定 pyyaml==6.0.3）
markupsafe → /usr/lib/python3/dist-packages/markupsafe  2.0.1  （lock 指定 3.0.3）
```

兩者皆為**主版本不同**。另有 6 個 locked 套件版本不符（`setuptools` 59.6.0 vs lock 81.0.0、
`filelock`、`fsspec`、`cuda-bindings`、`cuda-pathfinder`、`typing-extensions`），
37 個 venv 內套件不在 lock 中，以及 `lingbot-map==0.1.0` 是指向**不存在**目錄
`/home/allen/lingbot-map` 的 editable 安裝（實測 `ls` 失敗）。

### 可能後果

「乾淨環境重建」與「現場實跑環境」是兩個不同的環境。飛行驗證結果無法保證可轉移到
重建後的機器上，反之亦然。這使所有已完成的驗證失去可重現性基礎。

### 建議修正

1. 以 `SFM_VENV_DIR` 建立乾淨 venv，`bash tools/install_runtime.sh` 重裝，
   跑完整測試與 `驗證系統.sh` 比對。
2. 補齊 lock（見 F-08 的 `scipy`、`onnxruntime-gpu`、`av`）。
3. 移除 `lingbot-map` 死安裝。
4. 於 `start_anafi_live.sh` 加入 `include-system-site-packages` 檢查，使真機啟動時
   拒絕不合規環境（與安裝腳本一致）。

### 是否已修正

否（重建執行環境會影響操作員現行可用狀態，屬需操作員決策的高影響動作）。

### 剩餘風險

高（對可重現性）；對當下飛行安全影響間接。

---

## F-05｜兩個最大且最關鍵的函式從未被執行，其保證以「原始碼字串比對」斷言

- **嚴重度**：P1
- **類別**：可測試性
- **檔案**：`定位演算法/flight_control/path_follow_flight.py:1779-2088`（`fly()`，310 行）；`控制介面程式/operator_interface/flight_operator_app.py:6881`（`main()`，919 行）
- **測試**：`tests/localization/flight_control/test_flight_safety_gates.py:1366-1379`、`:1662-1685`；`tests/control_interface/operator_interface/test_operator_command_safety.py:430-462`、`:586-598`

### 實際證據

```python
# test_flight_safety_gates.py:1368-1374 ── 對原始碼文字做索引比較
initial_sticks_i = src.index('set_piloting_source(drone, "SkyController")')
preflight_i      = src.index("configure_flight_preflight")
takeoff_i        = src.index("TakeOff() >>")
assert controller_i < src.index("def send_pcmd") < src.index("SafetyMonitor(") < takeoff_i
assert src.index("must_land = True") < src.index("TakeOff() >>")
assert src.rindex("monitor.send_authorized") > src.index("finally:")
```

全庫**沒有任何測試呼叫 `fly()`**。`main()` 只被執行一次，且在 argparse 階段就
`SystemExit`，因此 exit-signal 註冊與 `_emergency_cleanup` 僅以 `inspect.getsource` 字串比對驗證。

全庫約 24 個測試屬此類原始碼字串斷言（其中 11 個在 `test_operator_command_safety.py`）。

### 可能後果

任何「保留原始碼文字但破壞實際接線」的重構都能讓這些測試維持綠燈：把
`_emergency_cleanup` 移出訊號處理器、在 helper 內調換呼叫順序、或讓 `signal.signal` 失敗，
測試都不會發現。而這正是「關終端機會降落」這條保證所依賴的程式碼。

### 公允說明

作者顯然知道這是弱代理：`test_flight_safety_gates.py:123` 有
`f"{token}() no longer appears in fly(); this test can no longer prove ..."` 這類自我防護，
避免測試在重構後靜默失效。這是負責任的做法，但仍不是行為驗證。

### 建議修正

`fly()` 的三重鎖（`SystemExit`、`build_controller` 的 `RuntimeError`、
`AUTONOMOUS_ROUTE_EXTERNAL_APPROVAL_LOCKED`）使其無法直接測試。可行路徑：
把 `fly()` 中「建立 monitor → 設定 piloting source → preflight → TakeOff → finally land」
的**順序邏輯**抽成一個可注入假 drone 的 `_fly_sequence(deps)`，讓 `fly()` 僅剩鎖與 wiring。
如此順序保證可用真實執行驗證，而非字串比對。此為**行為保持**重構，但影響最關鍵函式，
必須由操作員排程並在模擬器上完整回歸。

### 是否已修正

否（屬需設計審查的重構，見 `refactor_plan.md`）。

### 剩餘風險

中。目前 `fly()` 硬鎖，實際不執行；但解鎖前必須先有真實測試。

---

## F-06｜`manual_nudge_pilot.py` 的 deploy 副本未納入 git，且不在任何鏡像檢查清單內

- **嚴重度**：P1
- **類別**：可重現性 / 一致性
- **檔案**：`定位演算法/deploy_code/sfm_glomap_deploy/manual_nudge_pilot.py`（705 行）

### 實際證據

```
$ git ls-files --error-unmatch 定位演算法/deploy_code/sfm_glomap_deploy/manual_nudge_pilot.py
error: 路徑規格 ... 未符合任何 git 已知檔案   ← 未追蹤
$ git ls-files --error-unmatch 定位演算法/flight_control/manual_nudge_pilot.py
定位演算法/flight_control/manual_nudge_pilot.py                ← 已追蹤
$ diff -q （兩者）→ IDENTICAL
$ grep -c manual_nudge_pilot 定位演算法/validation/check_runtime_mirrors.py
1
```

12 個同名檔案中只有 8 對受鏡像檢查保護；`manual_nudge_pilot.py` 不在其中。
它同時是**操作 UI 微移常數的來源**（`olympe_live_backend.py:62` 於模組層 import）。

### 可能後果

工程師調整 `flight_control/manual_nudge_pilot.py` 的 `NUDGE_PCT` 後，deploy 側留下
705 行的過期孿生檔，而：`git diff` 看不到（未追蹤）、鏡像檢查看不到（不在清單）、
`MANIFEST.tsv` 看不到（兩份獨立雜湊，同步重生）、mtime 看不到。
**唯一的發現機制是人工 `diff`。** 今日該孿生檔無 importer，但只要 `sys.path` 順序改變
（見 F-17），UI 就會靜默改用另一組微移常數。

### 建議修正

擇一：(a) 把 `manual_nudge_pilot.py` 加入 `MIRROR_PAIRS` 並 `git add` deploy 副本；
或 (b) 若 deploy 側確實無人使用，直接刪除該副本。**(b) 較佳**——減少一份鏡像即減少一類風險。

### 是否已修正

否（刪檔屬破壞性動作，需操作員確認 deploy 端確無使用者）。

### 剩餘風險

中。

---

## F-07｜`olympe_frame_source.py` 兩份副本行為分歧：deploy 副本缺少影格時間戳單調性保護

- **嚴重度**：P1
- **類別**：一致性 / 資料生命週期
- **檔案**：`定位演算法/flight_control/olympe_frame_source.py:718-722`（有保護）vs `定位演算法/deploy_code/sfm_glomap_deploy/olympe_frame_source.py:468`（無）

### 實際證據

```python
# flight_control 副本（:718-722）
with self._lock:
    # Never let a slower convert overwrite a newer published frame.
    if self._latest is not None and stamp < float(self._stamp):
        self._queue_drops += 1
        return
    if digest == self._digest:
        ...

# deploy_code 副本（:468）── 直接進入 digest 檢查，無單調性守衛
with self._lock:
    if digest == self._digest:
        ...
```

```
$ grep -c 'Never let a slower convert overwrite a newer published frame' <兩檔>
deploy_code/... : 0
flight_control/...: 1
```

兩檔共 332 行差異，且**mtime 完全相同**（`2026-08-03 00:20:57`），因此時間戳無法用於
判斷何者較新。此檔**不在**鏡像強制清單內（屬「刻意分歧」的三檔之一）。

### 可能後果

deploy 副本可讓較舊的影格覆蓋較新的已發布影格，使 `self._stamp` **倒退**。
控制端以 `_stamp` 判斷新鮮度（超過 `stale_s` 回傳 `None`），因此倒退的時間戳可能
讓**當前**畫面看起來過期（誤觸 HOVER），或反過來讓舊影格成為「最新定位依據」。
這正是該模組 docstring（`:5`）自稱要防止的失效模式。

其他分歧：deploy 副本缺少 mid-convert 丟棄（CPU 壓力下影格年齡更差）、
缺少 SkyController media-name 重試（SC3 通常只廣播 `"DefaultVideo"`，deploy 副本
在預設場域設定下會開錯 media name）。反向也有一處：flight_control 副本在
`require_source_timestamps=False`（**兩者的預設值**）時會**發布**無來源時間戳的影格
並**停止累加 `_timestamp_drops`**，使丟棄計數器靜默——此處 flight_control 較不保守。

### 目前是否影響飛行

**目前不影響。** UI 程序載入 flight_control 副本（有保護），定位 worker 子程序載入
deploy 副本但只做定位比對、不做影格發布時序決策。風險在於 F-17 的 `sys.path` 順序脆弱性。

### 建議修正

把單調性守衛與 mid-convert 丟棄移植到 deploy 副本（或反向合併為單一共用模組）。
舊版 `sync_mirror_check.sh:8-11` 曾用註解宣告「兩份必須一起 review」，但**沒有任何
程式碼執行該檢查**；現已由 authoritative Python checker 分類並檢查 transitional copy。

### 是否已修正

否（跨樹合併影格路徑屬行為性變更，需模擬器回歸）。

### 剩餘風險

中。

---

## F-33｜背景執行緒直接操作 Tk widget；按下「自主」按鈕今天就會觸發

- **嚴重度**：P1
- **類別**：併發 / UI 架構
- **檔案**：`控制介面程式/operator_interface/flight_operator_app.py:4233-4243`（背景執行緒）、`:4265` 與 `:4270`（`write_log`）、`:4419-4421`（`write_log` 實作）

### 問題描述

`_dispatch_live_command` 把 `_backend_command` 丟到背景執行緒 `olympe-ui-<command>` 執行。
而 `_backend_command` 在**命令被拒絕時**會呼叫 `self.write_log(...)`，
`write_log` 直接對 `tk.Text` 做 `insert()` / `delete()`。**Tkinter 不是 thread-safe。**

### 實際證據

```python
# :4233-4243 ── 背景執行緒
def run() -> None:
    try:
        result = self._backend_command(command, payload)   # ← 在背景執行緒上
    except Exception as exc:
        error = repr(exc)
    ...
threading.Thread(target=run, name=f"olympe-ui-{command}", daemon=True).start()

# :4264-4271 ── 於同一背景執行緒上操作 Tk
except ValueError as exc:
    self.write_log(f"指令已拒絕: {exc}")          # :4265
...
if not result.accepted:
    self.write_log(f"{command}: 已拒絕 ({result.reason_code})")   # :4270

# :4419-4421 ── write_log 就是 Tk widget 變更
self.log.insert("1.0", ...)
self.log.delete(...)
```

本稽核實跑 probe 證實 Tk 對此的反應：

```
background-thread Tk result: ['RuntimeError: main thread is not in main loop']
```

### 今天就可觸發的路徑（非假設）

`start_auto` 在 `async_commands` 內（`:4385`）→ 走背景執行緒；
`_typed_command` 對 `START_AUTO` 一律回傳
`ControlResult.rejected("LOCKED_EXTERNAL_APPROVAL", ...)`（`olympe_live_backend.py:3266`）
→ `accepted=False` → 觸發 `:4270` 的 `write_log` → **背景執行緒操作 Tk**。

即：**在真機模式下按「自主」按鈕，今天就會在 `olympe-ui-start_auto` 執行緒上違規操作 Tk。**
`firmware_limits_apply`、`auto_speed_limit_apply` 亦然（此二者是 F-01 中**正確**回傳布林的分支，
因此它們的拒絕會確實走到這裡）。

### 可能後果

`run()` 的 `except Exception` 會接住該 `RuntimeError` 並經 `_flight_results` 回報，
所以**不會使程序崩潰**——但：

1. 操作員看到的是通用錯誤，而**不是**「自主已鎖定」這個真正原因。
2. Tcl 直譯器已被跨執行緒觸碰。CPython 的 tkinter 多半能偵測並丟出 `RuntimeError`，
   但這取決於 Tcl 的 threaded/non-threaded 編譯方式；在 non-threaded Tcl 上，
   結果是直譯器狀態損毀而非乾淨例外，可能使 UI 在飛行中失去回應。

### 與 F-01 的交互作用（重要）

**目前 F-01 正在遮蔽 F-33。** 因為 legacy dispatcher 丟棄布林，
takeoff／land／hover／nudge 的拒絕都變成 `accepted=True`，永遠走不到 `:4270`。
**若先修 F-01 而不同時修 F-33，會把大量拒絕路徑導入這個壞掉的跨執行緒呼叫。**
兩者必須一起修。

### 重現方法

模擬或真機模式下按「自主」按鈕，觀察 `_flight_results` 收到
`RuntimeError: main thread is not in main loop` 而非預期的拒絕訊息。

### 建議修正

`_backend_command` 不應直接寫 UI。改為回傳結構化結果，由已存在的
`_finish_backend_command`（`:4279`，**已經在 Tk 執行緒上執行**）負責顯示：

```python
# _backend_command: 不呼叫 write_log，改回傳 (ok, message)
if not result.accepted:
    return False, f"{command}: 已拒絕 ({result.reason_code})"
# _finish_backend_command 於 Tk 執行緒統一 write_log
```

或最小改動：把兩處 `self.write_log(...)` 換成 `self.after(0, self.write_log, msg)`。

### 是否已修正

**否。** 修正需與 F-01 一併設計，且位於飛行命令派送路徑上，依 `SAFETY.md` 應由操作員審核。

### 剩餘風險

中高。建議與 F-01 綁為同一個修正批次。

---

# P2 — 近期應修正

---

## F-08｜`scipy` 未納入 lock，碰撞監控在乾淨安裝上靜默關閉（fail-open）

- **嚴重度**：P2（自主飛行解鎖後升為 P1）
- **檔案**：`requirements/runtime.txt:24-26`（被註解）；`定位演算法/flight_control/real_path_follow_controller.py:59-62, 660, 676`

```python
try:
    from scipy.spatial import cKDTree
except Exception:  # optional; collision monitor becomes unavailable
    cKDTree = None
...
self.tree = cKDTree(self.xyz) if (cKDTree is not None and len(self.xyz)) else None
def update(self, pos):
    if self.tree is None:
        return {"status": "OFF", "distance": None, "point": None, "severity": 0.0}
```

`# scipy==1.17.1` 在 requirements 中被註解掉，故不在 lock 內。依文件流程做乾淨安裝
**不會安裝 scipy**，碰撞監控回傳 `status="OFF"`、`severity=0.0`——與「附近沒有障礙物」
在資料上**無法區分**。本機現有 scipy 1.15.3（版本亦與註解的 1.17.1 不符）。

**公允說明**：該監控自述為 "only an operator warning layer... should not be the only
real-flight safety check"，且位於目前硬鎖的自主路徑上。

**建議**：把 scipy 正式納入 requirements 與 lock；並把缺席狀態改回報
`status="UNAVAILABLE"` 而非 `"OFF"`，於 preflight 明確記錄一行。

**是否已修正**：否。

---

## F-09｜`SafetyMonitor` 在未取得 SafetySwitch 時預設為 AUTO（fail-open），與 SafetySwitch 的 fail-closed 預設相反

- **嚴重度**：P2
- **檔案**：`定位演算法/flight_control/path_follow_flight.py:814` vs `:290`

```python
# SafetyMonitor.__init__ :814
self.mode = str(getattr(safety, "mode", "AUTO")).upper()   # ← 預設 AUTO

# SafetySwitch.__init__ :290
self.mode = "HOVER"      # fail closed until a readable command is applied
```

兩個安全類別的預設方向**相反**，且較寬鬆的那個在 monitor 上。
`fly()` 一定會傳入 switch，故目前不可觸發；但這是留給未來呼叫者的陷阱，
且與同檔案自身的 fail-closed 政策矛盾。

**建議**：改為 `getattr(safety, "mode", "HOVER")`，並補一個「無 switch 時必須 HOVER」的測試。

**是否已修正**：否（屬安全預設值變更，需操作員確認無現存依賴 AUTO 預設的呼叫者）。

---

## F-10｜`SafetyMonitor._run_loop` 在持有 `_io_lock` 期間執行阻塞式 Olympe 等待，與同檔案的註解自相矛盾

- **嚴重度**：P2
- **檔案**：`定位演算法/flight_control/path_follow_flight.py:1107`（整個 tick body 持鎖）、`:913-915`（註解）

整個 `_run_loop` tick 在 `with self._io_lock` 內執行，包含
`_attempt_callback → _await_confirmed_action → result.wait()`（阻塞等待 Landing/Emergency）
與 `_ensure_piloting_source → set_piloting_source`（阻塞等待 + `drone.get_state` 回讀）。

而 `schedule_authorized_takeoff` 的 docstring（`:913-915`）明說在鎖內等待
"would prevent LAND/EMERGENCY from being processed"。程式碼與其自身的設計理由不一致。

**建議**：把阻塞式 callback 移出 `_io_lock`（僅在鎖內取快照與決策），或改用逾時較短的
非阻塞確認。屬併發性重構，需在模擬器上做故障注入回歸。

**是否已修正**：否。

---

## F-11｜沒有顯式狀態機；`tracker_state` 有 26 個字串值，`DroneState.loc` 混用三種概念

- **嚴重度**：P2
- **檔案**：`控制介面程式/operator_interface/flight_operator_app.py:1114-1222`（`DroneState`）

實測全庫賦值：

- `state.tracker_state` 共 **26 個相異字串**：`BOOT, BOOT_INIT, FAIL_SAFE_HOVER, GRAVITY_CAL,
  HOVER, HOVER_LOCK, LAND, LANDING, LAND_UNCONFIRMED, LINK, LINK_LOST_ONBOARD,
  LOCALIZATION_ONLY, NUDGE, PC, PC_FROZEN, RTH, SAFETY_ACTION_PENDING, SOURCE_FAIL,
  STICK_MONITOR_FAIL, STICKS, STREAM_LOST_HOVER, STREAM_LOST_MANUAL, TAKEOFF,
  TAKEOFF_BLOCKED, TAKEOFF_FAIL, TRACK`
- `DroneState.loc` 共 8 值，**混合三種概念**：來源型別（`LIVE`/`SIM`）、
  定位狀態（`OK`/`STREAM_LOST`/`LOST_RECOVERY`/`STARTING`）、MegaLoc 子階段
  （`MEGALOC_LOCKED`/`MEGALOC_LOCKING`）。
- 全庫**無** transition table、無 Enum、無非法轉移拒絕邏輯（僅 `InterfaceMode`、
  `ControlAction`、`FailureReason` 三個 Enum，皆非狀態機）。

**公允說明**：實際的安全閂鎖**不是**靠這些字串實作的——真正的權威是
`pilot_sticks` 布林與 `SafetyMonitor.terminated` Event，兩者都有正確的閂鎖語意與測試。
`tracker_state` 主要是**顯示與紀錄**用途。因此這是可維護性問題，不是安全漏洞。

**建議**：把 `tracker_state` 收斂為 `StrEnum`，集中於一個 `set_tracker_state(new, reason)`
方法內賦值並記錄前後狀態（同時解決 §12 要求的 state-before/state-after 日誌）。
不建議一次導入完整 FSM 框架。

**是否已修正**：否。

---

## F-12｜兩個 God object

- **嚴重度**：P2
- **檔案**：`flight_operator_app.py`（`OperatorApp`）、`olympe_live_backend.py`（`OlympeLiveBackend`）

實測（AST）：

| 類別 | 行數 | 方法數 | `__init__` 設定的實例屬性 | 最長方法 |
|---|---|---|---|---|
| `OperatorApp` | 3993 | 106 | **144** | `_build_ui` 564、`tick` 351、`__init__` 279、`update_localization_metrics` 138 |
| `OlympeLiveBackend` | 3749 | 93 | **113** | `poll` 308、`read_connection_inventory` 259、`__init__` 192、`_takeoff_preflight` 140 |

`OperatorApp` 至少混合 10 種責任：widget 建構、100ms render/telemetry tick、定位提交與
結果處理、飛行命令派送、JSONL metrics 寫入、PIL 地圖／視訊點陣化、重力校正、
航線覆蓋層編排、場域切換、日誌。

另外，UI **繞過自己宣告的 `OperatorBackend` protocol**，直接呼叫後端私有成員：
`_flight_state_name()`、`cleanup()`、`via_skycontroller()`、`pilot_sticks`、
`stick_override_count`、`stream_lost_hover()`。

**建議**：見 `refactor_plan.md`。優先抽出**純函式**（已有 43 個模組層函式是好的起點）與
`_append_loc_metrics`／render 兩塊無狀態邏輯，而非急於拆成更多 manager。

**是否已修正**：否。

---

## F-13｜約 95 個 `SFM_*` 環境變數形成繞過 `site_profile` 驗證的第二套設定通道

- **嚴重度**：P2
- **檔案**：全庫；`控制介面程式/operator_interface/flight_operator_app.py:7080-7126` 等

設定有四個來源（site profile JSON、EDM profile JSON、~95 個 `SFM_*` 環境變數、argparse 預設），
**沒有單一 resolver**。優先序只在一處是顯式的（`--site-profile` 與逐項 flag 互斥，
`flight_operator_app.py:576-582`）；其餘皆為 `os.environ.get(NAME, hardcoded)` 在 import 時
求值而**湧現**的順序。

`site_profile.py` 的驗證嚴格，但幾乎只做**存在性與正負號**檢查，**不做範圍檢查**；
而環境變數這條路徑**完全不經過**該驗證。可影響安全上限者包括
`SFM_MAX_TILT_DEG`、`SFM_MAX_VERTICAL_SPEED_MS`、`SFM_MAX_ROTATION_SPEED_DEGS`、
`SFM_MIN_TAKEOFF_BATTERY_PCT`、`SFM_RTH_MIN_ALTITUDE_M`、`SFM_STREAM_LOSS_GRACE_S`，
以及明確的旁路開關 `SFM_ALLOW_LEGACY_FLIGHT=1`（繞過 geofence）與
`SFM_GATE_WEAK=0`（在弱定位上飛行，import 時求值）。

**緩解**：firmware limit 會在 connect 時寫入並**回讀確認**（`_configure_firmware_limits_locked`），
韌體拒絕的值會使 preflight 失敗，故極端值不會靜默生效。

**建議**：建立一個 `resolve_config()` 單一入口，輸出一個 frozen dataclass 與 checksum，
並於啟動日誌印出「最終生效設定」；把安全相關 env var 納入同一套範圍檢查。

**是否已修正**：否。

---

## F-14｜UI↔worker 的定位 payload 是無型別 dict，且缺少 pose 時間戳／信心值／frame id

- **嚴重度**：P2
- **檔案**：`控制介面程式/operator_interface/live_localizer_worker.py:811-886`；`flight_operator_app.py:725-745`

跨程序的定位結果是 JSON 解析出的 raw `dict`（約 60 個 key）。它**有**：
`seq` 計數器、多個 host monotonic 時間戳、`success`、`inliers`、`reproj_rms`、`mode`/`next_mode`、
`pose{x,y,z,yaw_raw}`。它**沒有**：pose 自身的時間戳（只有 host 端的提交／到達時間）、
`success` 以外的有效性旗標、信心純量、worker 端的 frame id。

系統其實**已有**適當的型別（`定位演算法/flight_control/pose_types.py` 的 `Pose`，
docstring 明確標注單位、frame＝raw GLOMAP、`stamp` 為 monotonic 永不用 wall clock），
但**未用於這條路徑**。且 `pose_types.Pose` 與 `real_path_follow_controller.Pose` 是
**duck typing 交換**（該 docstring 自述），屬有文件但脆弱的契約。

**建議**：在 worker 邊界導入一個 frozen dataclass（`LocalizationResult`），
明列欄位、單位、frame、timestamp、validity、seq；以 `from_json` 做一次驗證。
不需改動演算法。

**是否已修正**：否。

---

## F-15｜安全命令通道位於世界可寫目錄，且只有 AUTO 檢查新鮮度

- **嚴重度**：P2
- **檔案**：`定位演算法/flight_control/path_follow_flight.py:258, 273-281`

`SAFETY_FILE = os.environ.get("SFM_SAFETY_FILE", "/tmp/sfm_drone_safety.cmd")`。
`bandit` 亦標記此行（B108 hardcoded_tmp_directory）。任何本機使用者／程序皆可寫入。
`_ALIASES` 的 15 個 token 含 `e`/`emergency`（**切馬達，飛機會掉落**）。

**正面**：讀取邏輯本身 fail-closed 極佳——檔案不存在→HOVER、token 無效→HOVER、
讀取例外→HOVER，且 AUTO 有 mtime 新鮮度檢查（早於 run start 則拒絕）。

**缺口**：新鮮度檢查**只套用於 AUTO**。`LAND`／`EMERGENCY` 沒有；不過此二者是
「更安全方向」的轉移，故風險有限。真正的問題是路徑可寫性。

**建議**：預設改到 session 目錄下（如 `outputs/flight_logs/<session>/safety.cmd`，
`mkdir(mode=0o750)` 已是既有做法），並檢查檔案 owner 與權限。

**是否已修正**：否。

---

## F-16｜364 個 blind-except、104 個 try-except-pass，使「哪些錯誤被刻意吞掉」無法審閱

- **嚴重度**：P2
- **工具**：`ruff --select BLE001,S110`

| 檔案 | BLE001 | S110 |
|---|---|---|
| `olympe_live_backend.py` | 81 | 18 |
| `flight_operator_app.py` | 41 | 12 |
| `path_follow_flight.py` | 37 | 9 |
| `runtime_safety.py` | 4 | 0 |
| 全庫 | 364 | 104 |

**公允說明**：全庫**沒有**裸 `except:`；多數位於日誌／telemetry 讀取等非控制路徑，
且多半有記錄。安全關鍵路徑上的例外處理（`_zero_pcmd_or_log`、nudge 迴圈、SafetyMonitor）
經檢視是**正確且刻意**的——它們捕捉後會記錄並降級到安全狀態。

問題在**數量**：104 個 `except: pass` 中要辨識哪一個掩蓋了真實問題，成本極高。

**建議**：不要為了消警告而加 suppression。改為分批把 `except Exception: pass` 換成
`except <具體例外>: log.debug(...)`，優先處理 `olympe_live_backend.py` 的 18 個。

**是否已修正**：否。

---

## F-17｜`local_site_assets` 在 UI 程序內把 deploy 樹插到 `sys.path[0]`，模組解析正確性只靠 import 時序

- **嚴重度**：P2
- **檔案**：`控制介面程式/operator_interface/local_site_assets.py:629, 647`

兩處 `sys.path.insert(0, str(deploy))`，位於 `inspect_edm_bundle()` / `load_edm_runtime_profile()` 內，
執行於**UI 程序**。插入後 `olympe_frame_source`、`manual_nudge_pilot`、`path_follow_flight`、
`reloc_localizer_xfeat` 都會解析到 deploy 副本（離線 `find_spec` 模擬已證實）。

目前不會出事，因為 `OlympeLiveBackend` 於 `flight_operator_app.py:7490` 先建構，
其 `_connect` 已把 `olympe_frame_source` 綁入 `sys.modules`（flight_control 副本），
早於任何操作員觸發的場域匯入。**但這個正確性完全依賴「延遲 import 早於延遲 path 插入」的時序，
且沒有任何斷言保護它。** 任何把場域匯入提前、或把 backend connect 延後的重構，
都會讓 UI 靜默切換到 deploy 副本（該副本缺 F-07 的單調性守衛）。

**建議**：不要用 `sys.path` 插入來載入場域資產；改用 `importlib.util.spec_from_file_location`
明確載入，或在 UI 啟動時就固定 `sys.path` 且不再變動。至少補一個測試斷言
`olympe_frame_source.__file__` 指向 flight_control 副本。

**是否已修正**：否。

---

## F-18｜`執行環境/` 的第二套 manifest 因過期分支而結構性失敗（26 項，已於 2026-08-07 修正）

- **嚴重度**：P2
- **檔案**：`執行環境/tools/package_manifest.py`（過期分支）vs `tools/package_manifest.py:24`

修正前稽核記錄（該過期命令目前已不可執行）：

```
$ .venv/bin/python 執行環境/tools/package_manifest.py verify --root 執行環境
... 26 個 FAIL ...   exit=1
$ grep -n 'inductor_cache' tools/package_manifest.py 執行環境/tools/package_manifest.py
tools/package_manifest.py:24:    "inductor_cache",      ← 只有 root 版排除
```

26 項中 25 項是 `inductor_cache/`（torch inductor 編譯快取）的必然變動，
1 項為真實漂移（當時的優化記錄檔 size 18067→21650）。
因為該分支未排除編譯快取，**任何一次編譯後它都會再度失敗**——結構上不可能通過。
且 `system_validation.py` 不呼叫它，故此失敗對 `驗證系統.sh` 不可見。

**建議**：刪除該過期分支，改為呼叫 root 的 `tools/package_manifest.py`；或把它納入
`system_validation.py` 並修正排除清單。目前這份 manifest 提供的是虛假的完整性感。

**是否已修正**：是；第二實作已刪除，見下方 2026-08-07 修正。

**2026-08-07 修正**：已移除 `執行環境/tools/package_manifest.py` 過期第二實作，
並更新 legacy runtime 文件要求所有 package manifest 操作使用 root
`tools/package_manifest.py`。root tool 的 scope 同時排除 cache、editor、audit review
artifact 與其他非 release ignored 資料。

---

## F-19｜`_flight_results` 是無界 queue

- **嚴重度**：P2
- **檔案**：`控制介面程式/operator_interface/flight_operator_app.py:2918`

```python
self._flight_results: queue.Queue[tuple[str, object | None, str | None]] = queue.Queue()
```

其他 queue 都有 `maxsize=1`（`:2059`、`:2931`、`site_assets_panel.py:73`、
`route_editor_window.py:148,157`），唯獨此處無界。生產者是每個 async 飛行命令的背景執行緒，
消費者是 Tk 的 `tick`。若 Tk 主迴圈卡住（GPU 算繪、大地圖重繪），項目會無限累積。

**緩解**：生產速率受操作員按鍵速度限制，且 `_flight_inflight` 對同名命令去重。
實務上難以撐爆，故評 P2 而非 P1。

**建議**：加上 `maxsize` 與滿載時的丟棄＋記錄策略，與其他 queue 一致。

**是否已修正**：否。

---

## F-20｜六個 `FailureReason` 列舉成員從未被使用（宣告了不存在的能力）

- **嚴重度**：P2
- **檔案**：`控制介面程式/operator_interface/backend_contract.py:55-69`

`POSE_STALE`、`WORKER_EXIT`、`WORKER_STALL`、`UI_HEARTBEAT_LOST`、`INVALID_TELEMETRY`、
`SHUTDOWN` 在**生產程式碼與測試中皆零引用**。

其中 `WORKER_EXIT` / `WORKER_STALL` 特別值得注意：定位 worker 崩潰**確實**會被偵測
（`flight_operator_app.py:2482-2484`，且有真子程序測試），但只會降級定位健康度，
**不會觸發 `fail_safe`**。`UI_HEARTBEAT_LOST` 則暗示存在 UI 心跳看門狗——**並不存在**。

**建議**：刪除未使用成員，或把它們接上真正的 fail-safe 路徑。目前狀態會讓讀者
（與未來的稽核者）以為系統具備它並不具備的保護。

**是否已修正**：否。

---

## F-21｜空白鍵會先重新觸發上一個被點擊的飛行按鈕，才執行懸停

- **嚴重度**：P2
- **檔案**：`控制介面程式/operator_interface/flight_operator_app.py:3124`（`<space>` → 全部懸停）、`:3522-3548`（飛行按鈕）

Tk 的 `ttk::button` class binding 綁定 `<space>`（`/usr/share/tcltk/tk8.6/ttk/button.tcl:23`），
且 `<ButtonPress-1>` 會 `ttk::clickToFocus`。bindtag 順序為 widget → class → toplevel，
故 **class binding 先執行**。畫面圖例（`:3595`）教導「空白鍵＝全部懸停」為緊急鍵。

實際後果：按過「恢復電腦控制」後再按空白鍵，會先重新取得 PC 控制再懸停；
按過「起飛」後按空白鍵，會先重新送出 TakeOff 再懸停。

**緩解**：空中再次 TakeOff 會被 `_takeoff_preflight` 的 landed 檢查擋下；
但「恢復電腦控制」「手動／搖桿」沒有對應閘門，會真的重新執行。

**建議**：飛行按鈕加 `takefocus=0`（不影響滑鼠點擊）。屬 SAFETY.md 涵蓋路徑，需操作員審核。

**是否已修正**：否。

---

## F-22｜`send_pcmd` / `_zero_pcmd_or_log` 在 `drone is None` 時回報成功

- **嚴重度**：P2
- **檔案**：`控制介面程式/operator_interface/olympe_live_backend.py:2229-2237, 2200-2227, 2239-2266`

`_raw_pcmd` 在 `self.drone is None` 時回傳 `None`（未送出任何東西），
但 `send_pcmd` 仍記錄 `pcmd` 事件並 `return True`；`_zero_pcmd_or_log` 亦回傳 `True`。

`_zero_pcmd_or_log` 的 docstring 明寫 "A failure here is never silent"，
但「沒有送出」這個情況**是靜默的**，且被記為成功。

**緩解**：`drone is None` 意味已斷線，此時本就無命令可送。

**建議**：`_raw_pcmd` 回傳 `None` 時，`send_pcmd` 應回 `False`、
`_zero_pcmd_or_log` 應記錄 `pcmd_zero_skipped_no_drone`。

**是否已修正**：否。

---

## F-23｜被排除於主測試套件之外的 `parrot_stimulate` 測試，卻是權威控制核心的測試

- **嚴重度**：P2
- **檔案**：`pytest.ini:17`（`norecursedirs` 含 `parrot_stimulate`）；`控制介面程式/operator_interface/scale_free_control_adapter.py`

`模擬器/parrot_stimulate/tests/`（12 檔、1357 行，含 `test_safety.py`、`test_scale_free_control.py`）
**不在** 1070 個測試內。它由 `tools/system_validation.py:164-169` 以 Python 3.11 的獨立 venv 另跑。

問題在於 `scale_free_control_adapter.py` 把
`模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py` 載入為**權威控制核心**，
而主套件中只有**一個**測試涵蓋該 adapter。

另注意：該核心匯出的新鮮度原語 `command_is_fresh`
（`scale_free_control_adapter.py:34` 再匯出）**沒有任何生產程式碼或測試呼叫它**。

**建議**：把 adapter 的契約測試（而非整個模擬器套件）納入主套件，
或在 `pytest.ini` 中以獨立 marker 納入。

**是否已修正**：否。

---

## F-24｜沒有覆蓋率量測，也沒有測試逾時機制

- **嚴重度**：P2
- **檔案**：`pytest.ini`（無 addopts）、`requirements/test.txt`（只有 pytest、ruff）

無 `.coveragerc`、`pyproject.toml`、`setup.cfg`、`tox.ini`；`coverage` 與 `pytest-cov` 皆未安裝
（本稽核嘗試 `--timeout` 時即因缺 `pytest-timeout` 而失敗）。
既然套件內有會啟動**真子程序**的測試（`test_worker_lifecycle.py`），
一個卡住的測試會無限阻塞整個套件。

**建議**：加入 `pytest-timeout` 並設全域逾時；加入 `coverage` 並對
`olympe_live_backend.py`、`path_follow_flight.py`、`runtime_safety.py` 設最低門檻。

**是否已修正**：否。

---

## F-25｜`ControlRequest.submitted_mono_ns` 從未被用於新鮮度判斷

- **嚴重度**：P2
- **檔案**：`控制介面程式/operator_interface/backend_contract.py:227, 235`；`olympe_live_backend.py:3285`

`ControlRequest` 帶有 `submitted_mono_ns`，但只被檢查**正值**（`:235`），
之後僅**記錄**（`:3285`）。沒有任何地方拒絕過期的命令請求。

系統其實**有**新鮮度原語 `command_is_fresh`（見 F-23），但無人呼叫。

**緩解**：實際 stale-command 防護是由別的機制提供的——nudge deadman（TTL）、
`pilot_sticks` 閂鎖、`_maneuver_in_progress`。所以不是防護缺口，是**未實現的設計意圖**。

**建議**：要嘛在 `_typed_command` 開頭加上請求年齡檢查，要嘛移除該欄位的暗示性
（改名為 `logged_submit_mono_ns`），避免讀者誤以為有新鮮度保護。

**是否已修正**：否。

---

## F-34｜專案自身的最高層驗收閘門 `驗證系統.sh` 目前是紅的，且已紅了一段時間

- **嚴重度**：P2
- **類別**：可觀測性 / 流程
- **檔案**：`tools/workspace_audit.py:75`（`OUTPUT_EVIDENCE_PREFIXES`）、`:239-251`（`--strict-output-names`）；`tools/system_validation.py` 步驟 6

### 實際證據

本稽核完整跑了三次 15 步驗證矩陣。最終一次（清乾淨 manifest 後）：

```
OK  root_pytest                      exit=0
OK  parrot_pytest                    exit=0
OK  parrot_ruff_check                exit=0
OK  parrot_ruff_format               exit=0
OK  runtime_mirrors                  exit=0
FAIL workspace_layout                exit=1     ←
OK  portable_manifest                exit=0
OK  flight_selftest                  exit=0
OK  root_dependency_check            exit=0
OK  parrot_lock_check                exit=0
OK  parrot_preflight                 exit=0
OK  profile_asset_validation         exit=0
OK  portable_simulator_preflight     exit=0
OK  cuda_production_smoke            exit=0
OK  offline_model_smoke              exit=0
status=failed   EXIT=1
```

失敗原因：

```
$ .venv/bin/python tools/workspace_audit.py --strict-output-names --no-sizes
layout: OK
output entries not classified: 1
WARN: outputs/audit_20260805        ← 上一次稽核（2026-08-05）留下的證據目錄
```

`outputs/audit_20260805` 的名稱不符合 `OUTPUT_EVIDENCE_PREFIXES`，
因此 `--strict-output-names` 讓整個驗證矩陣回報 `status=failed`。

**與本次稽核無關**：本稽核的 `audit/` 位於 repo 根目錄而非 `outputs/` 下，
實測不影響此步驟（移除本稽核產物後仍然失敗）。該目錄建立於 2026-08-06 14:04，
即上一次稽核期間。

### 可能後果

這是**流程風險而非技術風險**：專案最高層的「無飛行驗收矩陣」目前回報失敗，
而失敗原因是良性的（一個命名不合規的證據目錄）。
**一個長期為紅、且大家知道可以忽略的閘門，就不再是閘門。**
下一次真正的失敗（例如 `runtime_mirrors` 或 `cuda_production_smoke`）
將混在同一個 `status=failed` 中，很可能被同樣忽略。

其餘 14 步全部通過，代表這個閘門本身是有效且有價值的——正因如此更不該讓它一直紅著。

### 建議修正

擇一（皆為分鐘級）：
1. 把 `audit_` 加入 `tools/workspace_audit.py:75` 的 `OUTPUT_EVIDENCE_PREFIXES`；或
2. 把 `outputs/audit_20260805` 歸檔／改名為既有前綴。

並考慮讓 `system_validation.py` 在 receipt 中**逐步列出**哪一步失敗
（目前需自行解析 receipt JSON 才知道，本稽核即是如此發現的）。

### 是否已修正

**否。** 修改 `OUTPUT_EVIDENCE_PREFIXES` 等於放寬專案的產出物治理規則，
屬操作員的政策決定；歸檔既有稽核證據亦然。

### 剩餘風險

中（流程面）。

---

# P3 — 改善項目

---

## F-26｜`preview_site_route` 的 return 之後有無法到達的程式碼，且引用未定義名稱

- **嚴重度**：P3
- **檔案**：`控制介面程式/operator_interface/flight_operator_app.py:4845-4848`（修正前）
- **狀態**：**已修正**

`ruff --select F821` 是全庫唯一的 undefined-name：

```python
return f"地圖已顯示 {name}（{count} 點）；僅為顯示，未變更場域航線"
self.write_log(                                    # ← 永遠不會執行
    f"route preview loaded: {len(points)} waypoints; ...")   # ← points 未定義
```

失敗分支（`:4843`）有記錄，成功分支沒有——觀測性不對稱。

**修正內容**：把日誌移到 `return` 之前，並改用該作用域內實際存在的 `count`。
這**新增一行操作員日誌**（行為變更，但限於顯示用途的航線預覽，非飛控路徑）。

**驗證結果**：`ruff --select F821` → All checks passed；
`控制介面程式/operator_interface/` 542 個測試通過。

**剩餘風險**：無。

---

## F-27｜自主 arming 閘門的 NaN／Inf fail-closed 行為正確但無測試

- **嚴重度**：P3
- **檔案**：`控制介面程式/operator_interface/runtime_safety.py:573-577`
- **狀態**：**已修正（補測試）**

`_num()` 以 `math.isfinite` 拒絕非有限值，使 NaN/Inf 一律被視為「未知」而阻擋 arming。
本稽核實測 18/18 組非有限輸入（7 個欄位 × NaN/+Inf/−Inf）全部正確阻擋，
**但原本沒有任何測試固定此行為**。

非有限數在比較中會靜默取勝（`NaN > x` 與 `NaN < x` 皆為 False；`-inf` 通過任何上界），
因此這是值得釘住的行為。

**修正內容**：在 `test_autonomy_gate.py` 新增
`test_non_finite_evidence_fails_closed`（21 個參數化案例，涵蓋量測值與其上限**雙側**）
與 `test_bool_is_not_accepted_as_a_numeric_reading`。

**驗證結果**：該檔測試由 20 個增至 43 個，全部通過。

**剩餘風險**：無。

---

## F-28｜`live_non_map_acceptance.py` 插入一個不存在的 `sys.path` 項目

- **嚴重度**：P3
- **檔案**：`控制介面程式/operator_interface/live_non_map_acceptance.py:38-41`

```python
_FC = _HERE.parent / "flight_control"      # → 控制介面程式/flight_control
```
實測該目錄**不存在**（`ls` 失敗）。因此該腳本兩棵樹都拿不到。

**建議**：改為 `_HERE.parents[1] / "定位演算法" / "flight_control"`，或直接移除。

**是否已修正**：否（該檔為驗收腳本，非飛行路徑；修正前需確認其實際使用方式）。

---

## F-29｜三套彼此獨立的 `.mode` 詞彙，命名碰撞

- **嚴重度**：P3

`DroneState.mode`（MANUAL/PC_CONTROL/AUTO/CLOSED）、
`RuntimeState.mode`（TRACK/WEAK_TRACK/LOST，在 `production_xfeat_tracker.py`）、
`SafetySwitch.mode`（AUTO/HOVER/MANUAL/LAND/EMERGENCY）。

三者互不衝突（分屬不同類別），但同名 `.mode` 且都掛在名為 `state` 的物件上。
本稽核初次 grep 時即因此誤判為「同一欄位混用兩種概念」，複查後更正——
可見其對閱讀者的實際干擾。

**建議**：更名為 `control_mode` / `tracking_mode` / `safety_mode`。

**是否已修正**：否。

---

## F-30｜`flight_control/olympe_frame_source.py` 有約 100 行無呼叫者的死碼

- **嚴重度**：P3
- **檔案**：`定位演算法/flight_control/olympe_frame_source.py:315, 327, 339, 375, 389, 401`

`_init_pyav_decoder`、`_avcc_to_annexb`、`_decode_h264_payload`、`_coded_payload_bytes`、
`_coded_avcc_cb`、`_coded_bytestream_cb` 全庫無呼叫者；`_av_codec` 只在 `:254` 被賦值 `None`。
另有 `:626-627` 的 `if skipped_stale: pass` 空語句。

**建議**：移除（僅存在於已分歧的那一側，屬純攜帶成本）。依 CLAUDE.md「發現無關死碼應提出而非逕自刪除」，此處僅提出。

**是否已修正**：否。

---

## F-31｜`sync_mirror_check.sh` 是 `check_runtime_mirrors.py` 的重複實作，且無人執行（已於 2026-08-07 修正）

- **嚴重度**：P3

兩者涵蓋**完全相同的 8 對檔案**。前者無任何程式呼叫（僅在歷史優化記錄中被
提及），後者才被
`tools/system_validation.py:184-190` 使用。

更重要的是：舊版 `sync_mirror_check.sh:8-11` 的標頭宣告了
「`reloc_localizer_xfeat.py` 除該行外必須完全相同」這個不變條件，
而**沒有任何程式碼實作該檢查**——一個零強制力的文件化不變條件。

**建議**：刪除 shell 版，把其標頭的分歧政策寫進 `check_runtime_mirrors.py` 的
docstring，並對三個「刻意分歧」檔加上「除白名單行外必須相同」的實際檢查。

**是否已修正**：是；shell duplicate 已刪除，見下方 2026-08-07 修正。

**2026-08-07 修正**：已刪除無人呼叫的 shell duplicate；分歧檔案與 transitional
same-name policy 現由 `定位演算法/validation/check_runtime_mirrors.py` 的
`INTENTIONALLY_DIVERGENT`／`UNENFORCED_BUT_MUST_MATCH` authoritative constants
及 checker 執行。system validation 只呼叫此 Python checker。

---

## F-32｜磁碟已低於警告門檻；`setuptools` 有已知 CVE

- **嚴重度**：P3

實測 `tools/workspace_audit.py`：**20.18 GiB free (14.5%)**。
`runtime_safety.WARNING_FREE_PERCENT = 15.0`，故**目前已處於警告狀態**；
`CRITICAL_FREE_PERCENT = 5.0` 會**封鎖起飛**。最大佔用：`模擬器` 11.52 GiB、`.venv` 7.81 GiB、
`outputs` 2.20 GiB（約 100 個 session 目錄）。

`pip-audit` 對 `.venv`：`setuptools 59.6.0` 有 4 個 CVE（非執行期相依，可安全升級）；
`protobuf 3.19.4` 有 3 個 CVE，但**被 Olympe 8.4.0 的 wire schema 綁定且已在程式碼中註記**——
**不建議盲目升級**，應記錄為已接受風險。

**建議**：清理 `outputs/` 舊 session（`enforce_retention` 已實作但僅涵蓋部分檔名）；升級 setuptools。

**是否已修正**：否。

---

## 附錄 A — 本次稽核執行的驗證

| 動作 | 結果 |
|---|---|
| `pytest`（全套件，稽核起點） | **1070 passed, 1 skipped, 30.18s** |
| `pytest`（三處修改後） | 見 `final_validation.md` |
| `ruff check`（4 個來源樹） | 985 issues（分布見 F-16、F-30） |
| `ruff --select F821` | 修正前 1 項；修正後 **0 項** |
| `bandit -ll`（runtime 程式碼，40132 行） | **High 0**、Medium 14、Low 254 |
| `pip-audit`（`.venv`） | 10 個漏洞／2 個套件（protobuf、setuptools） |
| AST import 圖（4 樹） | **0 circular import** |
| Tk bindtag probe（實跑） | 證實 F-02 |
| `autonomous_arming_blockers` NaN/Inf probe | 18/18 fail-closed |
| `check_runtime_mirrors.py` | 8 對全數通過；唯一 authoritative mirror gate |
| `tools/workspace_audit.py` | layout OK，1 WARN，磁碟 14.5% |
| `tools/package_manifest.py verify` | root authoritative tool；legacy second implementation removed |
| `tools/system_validation.py`（15 步） | 14 步通過；步驟 7 `portable_manifest` 因本稽核的檔案變更而失敗 |

## 附錄 B — 只能靜態驗證的項目

無真機，故以下僅做程式碼與離線測試層級確認，**不得視為「已驗證安全」**：
Olympe 連線與 firmware limit 回讀、PDRAW 串流中斷復原、RTH／lost-link 在 GPS 不可靠場域的行為、
`os.execv` 熱重啟能否真正釋放 SkyController socket、碰撞監控在真實點雲上的判定、
以及 F-02／F-21 在真機空中的實際運動幅度。

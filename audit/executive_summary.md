# audit/executive_summary.md — 執行摘要

**稽核日期**：2026-08-07
**範圍**：軟體工程與系統架構（**不**重新驗證定位精度、建圖品質或演算法效能）
**基準**：`agent/localization-runtime-optimizations` @ `d0b2250`，工作區 dirty（63 檔變更）
**稽核期間未連線真機、未起飛、未送出任何飛行命令**（遵守 `控制介面程式/SAFETY.md`）

# 結論：`APPROVED_WITH_CONDITIONS`

---

## 一句話總結

安全核心正確且明顯優於研究原型（**0 個 P0**），但系統目前**對操作員與對事故紀錄
陳述不實**，且**執行環境無法由其自身工具重建**——這 8 項 P1 必須在真機飛行前修正。

---

## 問題數量

| 嚴重度 | 數量 | 已修 | 未修 |
|---|---|---|---|
| **P0**（立即停止使用） | **0** | – | – |
| **P1**（正式飛行前必須修正） | 8 | 0 | 8 |
| **P2**（近期應修正） | 19 | 0 | 19 |
| **P3**（改善項目） | 7 | 2 | 5 |
| 合計 | **34** | 2 | 32 |

**為何 P1 全部未修**：其中 5 項落在 `控制介面程式/SAFETY.md` 明文規定
「只有操作員明確要求並審過風險才可改」的飛行命令與按鍵路徑上；
另 3 項需操作員決策（重建執行環境、鏡像樹所有權、啟動器嚴格度）。
每一項都附有逐項 patch 建議與前置測試（`refactor_plan.md` §B）。

---

## 先講系統做對了什麼

這些經**實測**確認，且稽核建議**不要動**：

1. **PCMD 全庫只有一個送出點。** `olympe_live_backend.py:2235` 是唯一的
   `drone(PCMD(...))`。全檔只有**一個**非零呼叫點；其餘全是字面零。
2. **Emergency 真正搶占且無法被覆蓋。** `fail_safe()` → `give_to_pilot()` →
   持鎖設 `pilot_sticks=True`，此後 `send_pcmd` 拒絕一切；`auto_resume=False`，
   必須人工恢復。有 6 個測試證明無法被一般命令覆蓋。
3. **依賴圖無環。** AST 全掃 4 棵來源樹，0 個 circular import。
4. **SDK 被隔離。** Olympe 全部延遲 import，模組可在無 SDK 環境成功 import（實測）。
5. **設定驗證嚴格 fail-closed。** `site_profile.py` 拒絕未知欄位、非有限 JSON、錯誤 SHA 格式。
6. **關閉路徑四重保險且 idempotent**，且**訊號註冊失敗會明確警告操作員**。
7. **測試有真正的整合層。** backend 安全測試的 125 個測試函式（157 個收集到的測試）
   **全部**建構真實 `OlympeLiveBackend`（123 個經 `make_backend()`、2 個直接），
   而非 mock；31 個測試驅動真實控制迴圈；worker 測試用真實子程序。
8. **runtime 程式碼無機器特定絕對路徑**（全庫僅 1 處註解提及）。
9. **NaN/Inf 是覆蓋最完整的行為之一**（實測 18/18 fail-closed）。

**沒有發現**過度抽象、無用 wrapper、為套設計模式而生的層，或投機性可設定性——
這在此類專案中不常見，值得記錄。

---

## 三大剩餘風險

### 一、系統對操作員與對事故紀錄說謊（F-01 + F-33）

`OlympeLiveBackend.command()` 的 legacy dispatcher **丟棄** `takeoff_cmd()`、
`land_cmd()`、`hover_cmd()`、`nudge_begin()` 的布林回傳，然後 `return self.state`。
`ControlResult.completed()` 以 `raw_result is not False` 判定成功，而 `DroneState`
物件 `is not False` → **被拒絕的命令變成 `accepted=True`**。

後果不只是 UI 顯示錯誤：`commands.jsonl` 會把被 18 道 preflight 閘門擋下的起飛
記成 `accepted=true, executed=true, reason_code="OK"`——**與成功的起飛無法區分**。
這直接損害事故調查能力。

同一函式的其他分支（`land_now`、`emergency_stop`、`firmware_limits_apply`、
`nudge_vector`）**寫對了**，所以這是遺漏而非設計。

**且兩個缺陷互相遮蔽**：F-33 是背景執行緒直接操作 Tk widget
（實測 probe：`RuntimeError: main thread is not in main loop`）。
目前因 F-01 使 `accepted` 恆真而幾乎不觸發——**先修 F-01 而不修 F-33，
會把大量拒絕路徑導入這個壞掉的跨執行緒呼叫。兩者必須一起修。**
（今天已可觸發：真機模式按「自主」按鈕即走此路徑。）

### 二、非飛行的 UI 互動會產生飛行命令（F-02）

22 個方向鍵繫結在 toplevel 上，`_on_nudge_key_press` **沒有焦點守衛**。
實測 Tk probe：

```
Entry bindtags: ('.!entry', 'TEntry', '.', 'all')
在 Entry 打 'w' → entry 內容: 'w' | 觸發的處理器: ['TOPLEVEL_NUDGE_w']
```

三個 `ttk.Entry` 與日誌 `tk.Text` **皆未 disabled**。後端只在 `_landed` 時拒絕，
**空中會接受**。操作員在懸停中點進高度欄位輸入「82」→ `8`＝上、`2`＝下。

實體風險有界（放開會歸零，且有 deadman），故評 P1 而非 P0，
但這是本次稽核中最接近實體風險的一項。

### 三、現場環境無法由其自身工具重建（F-04）

```
$ cat .venv/pyvenv.cfg
include-system-site-packages = true      ← install_runtime.sh:35 明文拒絕此設定
```

**今天執行 `tools/install_runtime.sh` 會直接 exit 1。** 系統套件以不同**主版本**滲入：
`pyyaml 5.4.1`（lock 6.0.3）、`markupsafe 2.0.1`（lock 3.0.3），皆解析自
`/usr/lib/python3/dist-packages`。另有 6 個版本不符、37 個未鎖套件、
以及指向不存在目錄 `/home/allen/lingbot-map` 的 editable 安裝。

後果：「乾淨重建環境」與「現場實跑環境」是兩個不同的環境，
所有已完成的飛行驗證都**無法保證可轉移**。

---

## 其他值得注意的發現

| # | 發現 | 嚴重度 |
|---|---|---|
| F-05 | **兩個最大且最關鍵的函式從未被執行**：`fly()`（310 行，唯一 arming 入口）與 `main()`（919 行）。其保證以 `src.index()` 對**原始碼文字**斷言——保留文字但破壞接線的重構能維持綠燈 | P1 |
| F-06 | `manual_nudge_pilot.py`（705 行，UI 微移常數來源）的 deploy 副本**未納入 git**，且不在任何鏡像檢查清單內。一側修改對 git diff、鏡像檢查、MANIFEST 皆不可見 | P1 |
| F-07 | `olympe_frame_source.py` 的 deploy 副本**缺少影格時間戳單調性守衛**（flight_control 副本有）。兩檔差 332 行且 **mtime 完全相同**，時間戳無法判斷新舊 | P1 |
| F-03 | **真機啟動器完全不跑 preflight**，模擬啟動器卻跑完整 preflight（資產 SHA、CUDA、模型載入）。防護程度與風險相反 | P1 |
| F-08 | `scipy` 被註解掉未納 lock；缺席時碰撞監控回 `status="OFF"`、`severity=0.0`——與「沒有障礙物」無法區分 | P2 |
| F-11 | **無顯式狀態機**：`tracker_state` 有 26 個字串字面值；`DroneState.loc` 的 8 個值混合來源型別、定位狀態、MegaLoc 子階段三種概念 | P2 |
| F-12 | 兩個 God object：`OperatorApp` 3993 行／106 方法／**144 個實例屬性**；`OlympeLiveBackend` 3749 行／93 方法／113 屬性 | P2 |
| F-13 | 約 **95 個 `SFM_*` 環境變數**形成繞過 `site_profile` 嚴格驗證的第二套設定通道，含 `SFM_ALLOW_LEGACY_FLIGHT=1`（繞過 geofence）、`SFM_GATE_WEAK=0`（在弱定位上飛行） | P2 |
| F-18 | `執行環境/` 的第二套 manifest 因過期分支未排除編譯快取而**結構性失敗**（26 項），且無人呼叫——提供虛假的完整性感 | P2 |
| F-34 | **專案自身的最高層驗收閘門 `驗證系統.sh` 目前是紅的**：15 步中 14 步通過，`workspace_layout` 因上一次稽核留下的 `outputs/audit_20260805` 命名不合規而失敗。原因良性，但長期為紅的閘門就不再是閘門 | P2 |
| — | 磁碟可用 **14.5%**，已低於 `runtime_safety` 的 15% 警告門檻（5% 會封鎖起飛） | P3 |

---

## 本次稽核實際執行的修改

**刻意採取最小介入。** 只做了三項零風險改動：

| # | 改動 | 驗證 |
|---|---|---|
| A1 | 移除 `preview_site_route` 中 `return` 之後的死碼（引用未定義的 `points`），並把日誌移到 return 前、改用作用域內的 `count` | `ruff --select F821` 全庫由 1 項降為 **0 項** |
| A2 | 補 autonomy arming 閘門的 NaN/Inf fail-closed 測試（21 個雙側參數化 + bool 案例） | 該檔 21 → **43** 測試 |
| A3 | 補鏡像樹的分類強制與漂移偵測測試 | 該檔 1 → **3** 測試；**非空洞性已證明**（加入分類前實際失敗並正確指出 `manual_nudge_pilot.py`） |

**回歸結果：`1070 passed, 1 skipped` → `1094 passed, 1 skipped`，零失敗（29.5 秒）。**

所有 P1／P2 均**未**自行修改——它們落在 `SAFETY.md` 要求操作員審核的路徑上，
或需要操作員決策。`refactor_plan.md` 提供逐項的收益、風險、影響模組、前置測試、
遷移方式與驗收條件。

---

## 建議施行順序

```
1. B1+B2（命令結果真實性 + 跨執行緒 UI）── 必須同批，否則前者會惡化後者
2. B3（鍵盤焦點守衛）
   ──▶ 此時可進行繫留真機測試
3. B5+C5（乾淨環境重建 + 測試基礎設施）
4. B4, B6, B7（啟動器 preflight、碰撞監控、鏡像樹所有權）
   ──▶ 此時可進行非繫留真機測試
5. C1–C4（OperatorApp 減負、狀態機收斂、設定 resolver、定位 payload 型別化）
6. D3（fly() 抽出可測序列）
   ──▶ D3 完成前，自主飛行三重鎖不得解除
```

---

## 判定理由

給出 `APPROVED_WITH_CONDITIONS` 而非 `NOT_APPROVED`，是因為：
**安全核心經實測確認正確，且全部 8 項 P1 都屬「資訊正確性」與「環境可重現性」，
而非「控制正確性」。** 系統不會失控；它會在該拒絕時拒絕、該歸零時歸零、
該交回搖桿時交回。問題在於它**沒有誠實地說出自己做了什麼**。

給出 `APPROVED_WITH_CONDITIONS` 而非 `APPROVED`，是因為：
一個無法信任自己事故紀錄的系統，其飛行測試結論也無法被信任；
而一個非飛行的 UI 互動能產生飛行命令，在真機上是不可接受的。

**本結論不因系統能成功啟動、1094 個測試全綠、或 15 步驗證矩陣多數通過而給出。**
（事實上該矩陣目前回報 `status=failed`——見 F-34。）

---

## 文件索引

| 文件 | 內容 |
|---|---|
| `audit/executive_summary.md` | 本文件 |
| `audit/system_inventory.md` | 完整目錄結構、入口、模組、執行緒、queue、全域狀態、設定、日誌、相依 |
| `audit/system_architecture.md` | 元件圖、依賴圖、資料流、控制流、執行緒圖、11 條關鍵路徑、模組責任表 |
| `audit/findings.md` | **34 項發現**，各含編號、嚴重度、檔案行號、證據、後果、重現、建議、狀態、剩餘風險 |
| `audit/test_matrix.md` | 測試分層、20 項行為覆蓋檢查、缺少的測試與優先級 |
| `audit/refactor_plan.md` | 立即／飛行前／短期／長期／**不建議現在修改**，各含收益、風險、前置測試、驗收條件 |
| `audit/reproducibility_report.md` | 平台、版本、安裝、啟動、測試指令、環境限制、不可重現因素 |
| `audit/final_validation.md` | 22 項驗收問題逐一作答 + 最終結論 |

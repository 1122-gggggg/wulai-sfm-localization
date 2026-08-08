# 2026-08-08 全專案維護性與程式品質稽核

## 結論

本次純地面稽核已完成高優先級的可維護性、可讀性、重複模組、文件一致性、CI、
資源生命週期與 P119 品質回歸修正。軟體工程層可接受；真機自主航線仍維持
**永久鎖定／未核准**，本報告不構成飛行核准。

稽核與驗證期間未連線無人機、未載入 Olympe 真機流程、未送出 TakeOff／Landing／
PCMD，也未執行任何會離地的腳本。受保護的起飛、降落、關窗強制降落、方向鍵
hold-to-move、Esc 與 Space 語意沒有修改。

## 範圍與方法

檢查範圍包含：

- Python 與 C++ 第一方／隨附 runtime 程式；
- UI、worker、site profile、validation、flight controller 與 simulator；
- requirements hash locks、CI、portable manifest 與發布驗證入口；
- README、架構、規格、安全與部署操作文件；
- 固定 SHA 的 `P1190119.MP4` CUDA 全片 replay。

方法包含 AST／CodeGraph 結構檢查、Ruff `E9/F/C901`、pytest branch coverage、完全
相同檔案雜湊掃描、乾淨 CLI 匯入、runtime ownership gate、文件契約測試、離線模型
smoke、飛控 selftest 與固定影片品質 gate。

## 維護結果

| 面向 | 稽核前主要問題 | 修正後狀態 |
|---|---|---|
| 模組所有權 | flight/deploy 有 8 組正式 mirror、2 組分歧副本與 1 組過渡副本 | 11 個共用模組各只保留一個 owner；另一側副本全部移除 |
| 重複程式 | 同一 runtime 邏輯重複提交，容易依 `sys.path` 順序載入不同版本 | 移除 11 個副本，共 9,189 行；第一方 Python／shell 無其他完全相同檔案 |
| 可讀性 | UI/worker `main()` 同時建立 parser 與執行副作用；metrics dict 重複且有重複 key | parser 建立可獨立測試；metrics 改成純函式與明確欄位白名單 |
| 安全命令 | mission 預設 `/tmp`，flight 另有路徑／檢查邏輯 | 單一 `safety_command.py`；owner-private 路徑、權限驗證、拒絕 symlink、原子寫入 |
| 靜態品質 | CI 未完整檢查 undefined/unused import；沒有複雜度防回歸 | Ruff `E9/F` 全庫 gate；分區 C901 debt 只能下降、不能增加 |
| Coverage | 只量少數 validation/tools，總門檻缺失 | 第一方核心 branch coverage 53.47%，正式門檻 50.00% |
| CI | root 與 Python 3.11 simulator 驗證分散，依賴未全鎖 | root + parrot 兩個 job；hash-locked dependencies、lint、format、coverage、ownership |
| C++ 資源 | EDM ONNX demo 每次 `match()` 洩漏輸入陣列，session 手動 `new/delete` | 輸入改 `std::vector<float>`，session 改 `std::unique_ptr`；缺參數立即退出 |
| 文件 | mirror 政策、`/tmp` safety path、舊 mission 路徑、SciPy lock、coverage 與 P119 結果互相矛盾 | 現行文件以程式與 lock 為準統一；歷史稽核保留日期／commit 語境 |
| P119 | baseline identity 跟目前 site/profile SHA 不一致，舊結果曾未過 gate | 更新輸入 identity，不放寬品質門檻；2,934 幀全片 gate 通過 |

## 具體修正

### 1. 單一 runtime owner

部署目錄擁有定位 runtime：

- `artifact_integrity.py`
- `megaloc_cache.py`
- `pose_types.py`
- `production_xfeat_tracker.py`
- `reloc_localizer_xfeat.py`

飛控目錄擁有控制與 Olympe runtime：

- `autoflight.py`
- `manual_nudge_pilot.py`
- `olympe_frame_source.py`
- `path_follow_flight.py`
- `plan_path.py`
- `real_path_follow_controller.py`

`validation/check_runtime_mirrors.py` 保留舊檔名以維持 CI／呼叫端相容，但語意已改為
module ownership：缺少 owner、舊副本回流或新同名 runtime 檔都會失敗。worker 與
validation CLI 明確加入 deploy/flight owner 路徑，不依賴測試程序的 module cache。

### 2. 可讀性與責任分離

- `flight_operator_app.py` 與 `live_localizer_worker.py` 的 parser 建立已抽成
  `build_argument_parser()`，匯入與 parser 測試不再觸發執行期副作用。
- UI localization metrics 改由 `localization_metrics.py` 的純函式建立；欄位契約只有
  一份，移除重複 `ui_serialize_ms` key，且不會把未核准的任意 worker 欄位寫入 log。
- 新增的小型維護模組與契約測試全部納入 Ruff format gate；沒有對飛控大型歷史檔做
  純美化式大改，避免掩蓋安全差異。

### 3. 複雜度控制

核心第一方程式目前仍有 111 個 Ruff C901 熱點，屬既有 state machine 與大型 UI
累積債務。直接一次重寫 `run_loop`、PCMD 仲裁或 Olympe backend 會提高飛安風險，
因此採分區 ratchet：

| 區域 | 既有熱點數上限 | 最壞複雜度上限 |
|---|---:|---:|
| tools | 4 | 20 |
| deploy/localizer | 13 | 49 |
| flight | 21 | 86 |
| validation | 13 | 39 |
| control/UI | 60 | 72 |

`tools/check_maintainability.py` 使任一區域新增熱點或提高最壞值都會讓 CI 失敗；後續
拆分可以降低數字，不得以調高預算規避。這不是宣稱複雜度已歸零，而是把未完成的
長期重構變成明確、可驗證、不能惡化的工程契約。

### 4. 文件契約

已對齊根 README、`文件/ARCHITECTURE.md`、`文件/SYSTEM_SPEC.md`、兩個 runtime
README、操作介面 README、`SAFETY.md` 與專案 `AGENTS.md`。新增測試會阻止：

- `/tmp/sfm_drone_safety.cmd` 回流；
- 舊 mirror 政策或舊 `mission/` workspace 路徑回流；
- 架構文件漏列權威 runtime 模組；
- 現行文件出現不存在的相對連結。

日期化 `audit/` 文件描述當時 commit 與測試結果，保留作歷史證據，不把舊數字改寫成
今天的結果。

## P119 CUDA 全片證據

輸入 SHA-256：
`600bbf70227311cab079d77fcb896f97e6d3e55f6bc40b5bef01d74b65f7826c`。
容器宣告 2,935 幀、實際解碼 2,934 幀；只對此固定 SHA 明確接受既有
`KNOWN_INCOMPLETE` 例外。

| 指標 | 2026-08-08 結果 | 既有 gate | 結果 |
|---|---:|---:|---|
| 成功 pose | 2,051 | ≥1,686 | PASS |
| TRACK | 2,658 | ≥2,244 | PASS |
| LOST | 16 | ≤139 | PASS |
| inliers p50 / p95 | 706 / 880 | ≥693 / ≥875 | PASS |
| reprojection RMS p95 | 2.9355 px | ≤2.9565 px | PASS |
| limited jump unconfirmed | 818 | ≤1,187 | PASS |
| processing FPS | 22.9831 | 資訊值 | — |
| wall p50 / p95 | 32.023 / 77.845 ms | 資訊值 | — |

Baseline 只更新 site/localizer profile 的實際 SHA identity；所有品質閾值保持不變。

## 驗證摘要

- Root Python 3.10：1,243 passed、1 skipped；整合驗證 branch coverage 53.47%，通過
  50% 門檻（獨立測試口徑為 53.55%）。
- Parrot simulator Python 3.11：88 passed；Ruff lint 與 27 檔 format 全通過。
- 飛控 `--selftest`：通過；只執行純 Python／mock 路徑。
- Runtime ownership：11 個 canonical modules，無 mirror。
- Ruff `E9/F`、文件契約、複雜度 budget、workspace/profile/SHA 與 dependency gates：通過。
- `./驗證系統.sh --allow-dirty` 開發模式 19/19 步驟通過；收據：
  `outputs/validation_receipts/validation_20260808T082656816196Z.json`。正式 release 模式的
  19 個步驟亦全通過，但因本批修正尚未提交，dirty-worktree 防護按設計拒絕發佈。
- Portable manifest 共 371 個穩定項目；coverage data、HTML/XML coverage 報告不屬發佈
  內容，且已有回歸測試避免動態雜湊污染清單。
- C++ ONNX demo：有 source-level regression test；本機未安裝可選的 OpenCV 與
  ONNX Runtime C++ third-party headers，因此本次不能宣稱已完成實際 C++ link/run。

## 仍存在但已明確隔離的風險

1. 111 個既有 C901 熱點尚未逐一拆完；目前由不可上升的 CI budget 管理。
2. 全庫仍有歷史格式債務；新維護模組與 Python 3.11 simulator 已強制 format，未對
   飛控關鍵大檔做一次性全檔 reformat。
3. 專案未建立 mypy／pyright 全庫門檻；Olympe、Tk、Torch 與動態 backend contract
   仍使完整靜態型別化需要分階段進行。
4. 真機硬體、韌體、總連線中斷策略、PCMD response 與航線淨空仍須現場人員驗證；
   任何 agent 都不能代替操作員起飛。

這些項目不會被描述成「已解決」；前三項是受 CI 約束的後續工程債，第四項是軟體
無法取代的外部安全核准。自主航線入口會繼續在連線前 fail closed。

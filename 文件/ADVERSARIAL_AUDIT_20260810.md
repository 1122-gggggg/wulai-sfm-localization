# 全專案工程品質與對抗式審查

審查日期：2026-08-10  
審查範圍：`/home/allen/localization` 的第一方程式、飛控介面、定位 worker、
場域／航線資產、發行流程、離線安裝、模擬器與操作 UI。匯入的第三方 runtime
以版本、雜湊、SBOM、smoke test 管理，不把其原始碼風格計入第一方品質分數。

## 結論

**Overall Engineering Quality: 7.5 / 10**

系統已具備可重複執行的 no-flight validation、場域資產雜湊綁定、四步 preflight、
AUTO 起飛後定位閘門、手動接管、關閉時降落流程、離線依賴鎖、SBOM 與 portable
package manifest。這一輪以對抗方式找出的飛控競態、舊定位結果回寫、假成功 cleanup、
路線 TOCTOU、portable symlink/path escape、資源無界載入等問題，均以失敗路徑測試
固定後再修正。

這個分數不表示已證明可以安全真飛。仍無法由軟體測試取代的證據包括：真機環境的
端到端飛行、外部簽章信任根、100k reference 的實際壓力資料，以及 NVIDIA／相機／
Olympe 在目標電腦上的硬體相容性。

## 審查方法與成功條件

審查不是只跑 lint，而是逐項驗證下列性質：

| 面向 | 對抗方法 | 通過條件 |
|---|---|---|
| Boundary | 0、負值、上限、未來 timestamp、最少點數、bundle 大小／數量上限 | 明確拒絕或進入安全狀態，不能默認成功 |
| Negative | 壞 JSON、錯 SHA、路徑逃逸、symlink、缺依賴、非法命令型別 | 無副作用地失敗，錯誤可診斷 |
| Invariant | AUTO ownership、場域／route/frame 綁定、freshness、manifest identity | 狀態切換與重啟後仍成立 |
| State Transition | preflight、takeoff、hover、localization gate、route、manual、landing、shutdown | 只允許定義過的轉移，不可跳步或假完成 |
| Fault Injection | worker crash、CUDA OOM、串流 EOF、landing failure、延遲結果、cleanup 競態 | 降級、停止或重試；不得繼續輸出危險控制量 |

最終 release gate 應同時通過：完整 pytest、coverage floor、Parrot simulator tests、
Ruff、format、bounded mypy、maintainability budget、第一方 security rules、dependency
audit/SBOM、runtime ownership、workspace audit、profile asset validation、CUDA/model smoke、
來源 manifest、portable manifest、clean offline install 與 simulated UI pose smoke。

本次整合修正後的開發階段結果：root `1650 passed, 1 skipped`，Parrot simulator
`88 passed`；Ruff、format、bounded mypy、maintainability、security/SBOM、module ownership、
workspace、profile assets、flight selftest、dependency check、CUDA production smoke、EDM/MegaLoc
offline inference 與 simulated UI valid-pose smoke 均通過。正式 release receipt 必須在 clean
commit 與最終 portable package 建立後重跑，避免把 dirty-tree 結果當成發行證據。

## Critical / High findings 與處置

| 嚴重度 | File / symbol | 問題與影響 | 處置 |
|---|---|---|---|
| Critical | `operator_shutdown.py`, AUTO shutdown | cleanup 可能在 AUTO thread 仍輸出非零 PCMD 時進行 | 關閉先 suspend、cancel、送零值並 join；AUTO 未停止時拒絕 backend cleanup |
| Critical | `path_follow_flight.py`, landing state | 降落失敗一次後可能被視為 DONE | 加入有界重試與 `LANDING_UNRESOLVED`；未確認 landed 不得宣告完成 |
| High | `olympe_live_backend.py`, authority/takeoff/landing | 起降與 ownership 交錯時可能出現競態，`None` 也可能被當成功 | 序列化 authority；只有 literal `True` 算命令成功 |
| High | `live_localizer_worker.py`, result ordering | 較舊 TRACK 結果可能晚到並覆蓋較新的 LOST | 依 frame/capture 序拒絕 stale result，並加入 out-of-order regression |
| High | `live_localizer_worker.py`, CUDA OOM | OOM 被降成一般 LOST，或連續 OOM 造成無界 worker 重建 | 將 CUDA OOM 分類為 fatal；最多重建 3 次並退避，健康 inference 才重置，超限標記 unavailable |
| High | `export_edm_onnx_flight.py`, checkpoint loading | 對外部 checkpoint 使用可執行 pickle 載入 | 改用 `weights_only=True` 並拒絕不相容 payload |
| High | `reloc_localizer_edm.py`, bundle decode | reference、JPEG 解碼、descriptor、XYZ、covis 無界，握手前即可耗盡 RAM/VRAM | 解碼前檢查檔案/JPEG header、encoded/decoded bytes、8448 維 descriptor、XYZ 與 edge 上限 |
| High | `localization_contract.py`, timestamps | contract 可接受 pose timestamp 早於 capture | 明確要求 `pose_timestamp >= capture_timestamp` |
| High | `site_profile.py`, asset/deploy path | profile 可用 `..` 或 symlink 逃離 workspace，甚至指定任意 Python | 所有 runtime asset 與 interpreter 必須位於解析後 workspace 內，拒絕 symlink component |
| High | `release_activation.py`, copytree | excluded tree 中的 symlink 可能在 stage 時被跟隨 | stage 保留 symlink，再對完整 stage 拒絕任何 symlink |
| High | `package_manifest.py`, control files | `MANIFEST.tsv`／`SHA256SUMS` 自身或父路徑 symlink 可寫／讀到包外 | generate/verify 在任何控制檔寫入前檢查所有 path components |
| High | `export_simulator_package.py`, artifact paths | runtime artifact 或 destination 的父目錄 symlink 可讀寫包外 | source/seed/destination 全路徑 containment，拒絕任一 symlink component |
| High | `export_simulator_package.py`, publication | 匯出失敗留下半成品；source 在 copy 中變更仍聲稱舊 identity | sibling staging、前後 source identity 驗證、成功後 atomic publish |
| High | portable launch/install | 被排除的既有 `.venv` 可在 manifest 驗證前執行 | 使用系統 Python 先驗 package；package-local venv 必須有目前 manifest identity marker |
| High | `local_site_assets.py`, artifact copy | hash 驗證與 copy 間存在 TOCTOU | 複製至 temporary、重算 size/SHA、再 atomic replace |
| High | `operator_autonomy.py`, manual handoff | UI thread 同步等待 AUTO cancel，介面可能凍結 | 立即 latch cancel/zero，後續 handoff 在 daemon worker 完成 |
| High | `flight_operator_app.py`, emergency close | UI close 可能同步等待 backend stop 而失去回應 | emergency stop 非同步化，shutdown supervisor 保持狀態可見 |
| High | `site_assets_panel.py`, airborne import | 先寫入航線再做飛行中檢查，可能改變已綁定資產 | 在 file selection/import 前檢查 flight/route lock，controller 使用 immutable snapshot |
| High | `path_follow_flight.py`, weak-pose gate | production 可透過環境變數關閉弱定位保護 | production path 固定啟用；測試注入改走明確的非 production seam |
| High | `一鍵啟動.sh`, direct launch | 直接執行 portable launcher 可能略過 package manifest | 啟動任何 UI／網路前先驗證 `PORTABLE_PACKAGE.json`、`MANIFEST.tsv`、`SHA256SUMS` |

## Medium findings 與處置

| File / symbol | 問題 | 處置 |
|---|---|---|
| `flight_operator_app.py`, stream EOF | decoder delay buffer 在 EOF 時不排空 | EOF drain，保留最後可用 frame 與一致的 timestamp |
| localization scale conversion | 單一縮放率假設 16:9，非等比例輸入會扭曲 intrinsics | 分離 `scale_x` / `scale_y` 並驗證正值與 finite |
| XFeat production factory | bundle 與 MegaLoc/VPR metadata 未完全相互驗證 | factory 建立前驗證 localizer、retriever、descriptor contract |
| telemetry freshness | 只看整包更新可能掩蓋個別欄位過期 | altitude、velocity、attitude 等欄位各自記錄 sample time |
| speed semantics | UI 上限與實際 controller clamp 語意不一致 | 單一 domain limit；變更後撤銷本次 session approval |
| post-takeoff fallback | takeoff 後初始化失敗不一定進入 landing | takeoff 成功後任何 gate/factory 失敗都走 landing supervisor |
| typed command / cleanup | truthy object 或缺例外型別可產生 false-success | command 回傳與例外合約明確化；cleanup failure 保留失敗狀態 |
| portable wheelhouse | runtime package 強制攜帶 test/quality locks | live-minimal 只解析 runtime lock；仍保留 exact hashes |
| portable scope | 舊 package 含 authoring、測試與多餘場域 | live-minimal 僅含 operator、所選 river bundle、必要定位／飛控與安裝工具 |
| portable smoke cleanup | 被 manifest 排除的 `outputs` symlink 可能讓 cleanup 指向包外 | 任何寫入／清理前拒絕 symlinked outputs/simulator/venv path，加入 victim-survival regression |
| portable site marker | marker 可漏 profile 宣告 asset 或 reference-index sibling 仍自洽 | 交叉驗證 profiles、files、實際 profile JSON 與 index siblings |

## 已確認的核心 invariants

1. 四步 preflight 完成前，MANUAL 與 AUTO 都不能取得飛行 ownership。
2. AUTO 使用第三步確認時的 immutable route snapshot；後續 UI 切換不能偷換航線。
3. AUTO 狀態順序為 takeoff → hover → post-takeoff localization gates → route tracking。
4. `TRACK`、pose freshness、inliers、reprojection gate 在起飛並懸停後才開始決定是否進入路線。
5. MANUAL 接管會先撤銷 AUTO ownership 並送零控制量；兩者不能同時輸出 PCMD。
6. UI 或終端關閉時先停止 AUTO，再請求原地降落；無法確認 landed 時保留 unresolved failure。
7. GPS 有無不阻擋起飛；沒有可信 GPS fix 時不啟用距離 fence。這是使用者明確接受的操作政策，不代表有地理圍欄保護。
8. Olympe／無人機／SkyController 版本只做診斷，不是 AUTO gate。hardware approval receipt 是選用的稽核證據，不是 runtime gate。
9. profile、route、frame、重力對齊與模型資產以 site/coordinate-frame/SHA 綁定；不接受 workspace 外或 symlink 資產。
10. portable 在安裝、UI 或網路動作前驗證自己的 package manifest，安裝只允許鎖定且有 hash 的離線 wheel。

## 仍存在的風險與不能由本次軟體修正替代的證據

### 1. 真機 E2E 未被證明

測試環境不能安全地自動執行實際 takeoff、hover、manual takeover、landing。模擬 UI 與
backend fault tests 能證明程式狀態機，但不能證明現場 RF、螺旋槳、氣流、相機曝光、
控制器 firmware 或操作者反應。正式河濱試飛仍應使用低高度、短航線與實體接管驗證。

### 2. Manifest 提供完整性，不提供外部來源真實性

SHA 與 package manifest 可發現複製／傳輸後變更，但若攻擊者能同時替換檔案與 manifest，
仍可重簽自己的內容。真正的跨電腦 artifact registry 需要離線保存的 public key、簽章
發行程序、key rotation 與撤銷機制；這不能由 package 內自我簽署解決。

### 3. 100k references 尚未完成實測設計

目前 fallback retrieval 仍可能是 O(N·D)，適合現有小型場域，不應宣稱已支援 100k。
需改成分片、memory-mapped descriptor store 與 FAISS/cuVS 類 ANN index，並用真實 descriptor
分布量測 recall@k、P95 latency、RAM/VRAM、增量更新與 index corruption recovery。

### 4. 相依套件的已接受例外

SBOM 會記錄全部元件；`protobuf 3.19.4` 的三個 advisory 因 Olympe 8.4.0 protocol pin
暫時接受至 2027-08-09，`setuptools 81.0.0` 的 Linux runtime 不適用 advisory 暫時接受至
2027-02-09。PyTorch `2.11.0+cu128` 與 torchvision `0.26.0+cu128` 是官方 CUDA index
wheel，PyPI audit 無法比對，故以 exact hash、離線 wheelhouse 與 SBOM 補償；不是零風險。

### 5. Portable 不是跨作業系統 binary

最小 package 的明確目標是 Linux x86_64、CPython 3.10、相容 NVIDIA/CUDA 12.8 driver、
X11/XWayland、`python3-tk` 與 FFmpeg。它可搬到符合契約的另一台電腦，不是 Windows、
macOS 或無 NVIDIA 環境皆可直接執行的單一檔案。

## 32 項評分

| 項目 | 分數 | 主要依據 |
|---|---:|---|
| Readability | 7.2 | 命名與 seams 已改善；仍有大型流程函式 |
| Maintainability | 7.0 | 高風險責任已拆出；control 尚有 56 個 complexity hot spots |
| Simplicity | 6.8 | 發行／安全流程必要但層數偏多 |
| Modularity | 7.2 | authority、shutdown、site runtime、route domain 已分離 |
| Cohesion | 7.0 | 新模組內聚；backend 與 app 仍負責較多狀態 |
| Coupling | 6.8 | profile 與 EDM/runtime 仍有實際耦合 |
| SOLID | 6.8 | interface seams 增加；localizer abstraction 尚未完全可替換 |
| DRY | 7.4 | route/profile 驗證已收斂；部分 UI state 映射仍重複 |
| API Design | 7.2 | typed command 與 domain model 改善；動態 vendor API 限制仍在 |
| Type Safety | 6.4 | bounded mypy 嚴格通過，但全 UI/ML 尚未納入 |
| Correctness | 8.0 | 座標／時間／ownership invariants 有回歸測試；真機未驗證 |
| Robustness | 8.1 | fault injection 覆蓋 worker、OOM、landing、EOF、cleanup |
| Testability | 7.8 | dependency seams 與 simulator 良好；真硬體仍難隔離 |
| Testing | 8.5 | unit/integration/regression/smoke 與五類 adversarial matrix |
| Performance | 6.7 | UI render 有 budget；100k retrieval 未設計完成 |
| Memory | 7.1 | bundle/resource bounds 已加入；大型模型仍需實機 soak |
| Concurrency | 8.0 | ownership、shutdown、stale-result 與 worker lifecycle 有測試 |
| Security | 7.4 | path/symlink/unsafe load/security gate 已修；缺外部 trust root |
| Logging | 7.6 | validation receipts 與飛行事件可追蹤；尚非完整 structured telemetry |
| Configuration | 8.0 | profile/env/lock 集中且有 validation |
| Portability | 7.8 | 離線 clean install 與一鍵啟動；受硬體平台契約限制 |
| Reproducibility | 8.1 | commit、manifest、SHA、locks、SBOM、receipt 綁定 |
| Documentation | 7.8 | architecture/spec/portable/runbook 已補；現場 E2E 仍需紀錄 |
| Architecture | 7.2 | boundaries 進步；100k registry 與 multi-localizer 待重設計 |
| Consistency | 7.4 | error、route、profile contract 較一致；舊動態 UI code 仍混雜 |
| Dependencies | 7.0 | exact locks 與 offline wheelhouse；存在明列的 audit exceptions |
| Technical Debt | 6.6 | 複雜度已受 budget 控制但熱點仍多 |
| Repository Hygiene | 7.8 | runtime mirror、output naming、manifest gates；模型仍使 repo 大 |
| Extensibility | 6.8 | registry/seams 已有，但 site profile 目前仍以 EDM 為主 |
| Scalability | 5.7 | 現有場域可用，100k reference 尚未完成 |
| Reliability | 7.9 | shutdown/landing/freshness/failure lifecycle 已強化 |
| Developer Experience | 7.6 | 一鍵啟動與驗證入口清楚；GPU/Olympe 環境仍重 |

總分依專案指定權重計算：Maintainability、Architecture、Correctness、Robustness 各
10%；Readability、Modularity 各 8%；Testability、Performance、Reliability 各 7%；
Reproducibility 6%；Type Safety 5%；Configuration、Documentation 各 4%；其餘平均 4%。

## Top 20 refactoring hot spots

這些是風險導向排序，不等同「現在全部重寫」。數值為目前 complexity scanner 的
函式最高值／超預算數量；應一次拆一個 seam 並保留行為測試。

| Rank | File | Max / count | 建議邊界 |
|---:|---|---:|---|
| 1 | `定位演算法/flight_control/path_follow_flight.py` | 86 / 7 | state transition、landing、gate evaluation |
| 2 | `控制介面程式/operator_interface/live_localizer_worker.py` | 60 / 1 | process lifecycle、result admission |
| 3 | `控制介面程式/operator_interface/olympe_live_backend.py` | 50 / 16 | command adapter、telemetry、authority |
| 4 | `控制介面程式/operator_interface/production_xfeat_tracker.py` | 49 / 5 | factory、retrieval、pose conversion |
| 5 | `控制介面程式/operator_interface/production_edm_tracker.py` | 49 / 4 | factory、bundle validation、inference |
| 6 | `定位演算法/validation/benchmark_production_stream.py` | 39 / 1 | source runner、metric aggregation |
| 7 | `控制介面程式/operator_interface/flight_operator_app.py` | 32 / 17 | command coordinator、view model、tick |
| 8 | `定位演算法/validation/benchmark_edm_site_replay.py` | 30 / 1 | decode、replay、quality verdict |
| 9 | `定位演算法/deploy_code/sfm_glomap_deploy/reloc_localizer_edm.py` | 28 / 1 | bundle decoder、resource guard |
| 10 | `控制介面程式/site_profile.py` | 24 / 5 | schema parse、path policy、approval policy |
| 11 | `定位演算法/deploy_code/sfm_glomap_deploy/reloc_localizer_xfeat.py` | 23 / 1 | retrieval、matching、pose estimate |
| 12 | `定位演算法/flight_control/passive_flight_session.py` | 23 / 1 | capture lifecycle、result writer |
| 13 | `控制介面程式/operator_interface/live_non_map_acceptance.py` | 22 / 1 | stream runner、acceptance verdict |
| 14 | `定位演算法/flight_control/route_domain.py` | 21 / 1 | JSON parse、semantic validation |
| 15 | `tools/simulator_preflight.py` | 20 / 3 | dependency probes、artifact verdict |
| 16 | `定位演算法/deploy_code/sfm_glomap_deploy/megaloc_cache.py` | 20 / 1 | cache identity、load policy |
| 17 | `控制介面程式/operator_interface/operator_autonomy.py` | 20 / 1 | cancel handoff、UI event bridge |
| 18 | `定位演算法/flight_control/autoflight.py` | 19 / 2 | orchestration、status reporting |
| 19 | `定位演算法/flight_control/real_path_follow_controller.py` | 18 / 3 | coordinate adapter、command conversion |
| 20 | `定位演算法/validation/stream_integrity.py` | 18 / 1 | stream invariants、failure verdict |

## 接下來怎麼改

### 現在不要改

- 不在沒有 ground-truth dataset 時更換 Tcw/Twc、quaternion 或 coordinate-frame convention。
- 不為了降低數字一次重寫 `flight_operator_app.py` 或 backend；每次只拆一個有測試的 seam。
- 不重新訓練或轉換 EDM checkpoint；portable 必須保持與 profile SHA 綁定的既有模型。
- 不把使用者已明確接受的 GPS 非阻擋政策偷偷改回起飛 gate；應在 UI 保持明顯告警。

### 可以立即、低風險地持續修

- 逐個把 `path_follow_flight.py` 的 gate/landing transition 移到純函式並補 transition table tests。
- 擴大 mypy 到 `route_domain.py`、`localization_contract.py`、`operator_shutdown.py`。
- 將 validation receipt 與現場測試紀錄保留在外部只寫入媒體，避免只存同一台主機。

### 建議逐步重構

- 從 backend 繼續拆 TelemetryFreshnessStore、TakeoffLandingSupervisor 與 vendor adapter。
- 從 OperatorApp 拆剩餘 command coordinator、view model 與 shutdown lifecycle。
- 讓 route 解析、核准、controller conversion 只經過一個 immutable domain model。
- 將 source repository、artifact acquisition/cache、portable package assembly 完全分層。

### 需要重新設計

- 100k reference storage/retrieval：分片、ANN、mmap、版本化 index 與壓力基準。
- artifact registry/offline bundle：外部簽章、key rotation、撤銷、可重建 provenance。
- 多 localizer：穩定的 `Retriever`、`Matcher`、`PoseEstimator`、`LocalizerResult` protocol，
  讓 EDM/XFeat/RoMa 可替換且 profile 不直接綁實作細節。
- hardware approval 信任鏈：若未來恢復強制 gate，必須有設備身分、簽發者 trust root、
  有效期、nonce／anti-replay 與撤銷清單。

## 審查判定原則

- 「測試通過」只表示這個 commit 與 manifest 所綁定的軟硬體契約通過 no-flight 驗證。
- 「portable」表示可搬到契約相容的電腦離線安裝，不表示跨 OS、跨 GPU 或免安裝 driver。
- 「AUTO unlocked」不表示定位品質永遠足夠；起飛後仍必須通過 TRACK、freshness、inliers、
  reprojection gates 才能沿路線飛行。
- 操作者能切回 MANUAL 是重要保護，但不能替代正確的 shutdown、landing、authority、
  telemetry freshness 與 localization safety invariants。

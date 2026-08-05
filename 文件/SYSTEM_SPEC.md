# 場域視覺定位與 Parrot ANAFI 操作系統規格

| 欄位 | 內容 |
|---|---|
| 文件狀態 | 歷史設計與安全決策記錄；現行執行契約以根 README、site profile schema 與 preflight 為準 |
| 規格版本 | 0.1 |
| 日期 | 2026-08-02，Asia/Taipei |
| 適用工作區 | `/home/allen/localization` |
| 規格範圍 | 建圖、定位、控制介面、模擬串流、真機接口、安全、測試、部署、監控 |
| 實作限制 | 本文件保留 To-Be 與當時 As-Is 背景；不得覆蓋現行 fail-closed 程式契約 |

> 2026-08-03 發布註記：第 3 節的 As-Is 原為 2026-08-02 盤點。下列已改成
> 目前實作的關鍵事實；其他分階段與待辦表保留為歷史規劃，不是執行時預設值。

## 1. 決策摘要

本規格採用下列已確認決策，後續實作不得自行改變：

1. 目標優先順序固定為：飛行安全、定位準確、低延遲、失鎖恢復、操作便利、開發效率。
2. 正常操作為單人；未來任何真機自主路徑飛行例外要求兩人確認、實體安全監控員及起飛前 checklist。
3. 系統只有兩個操作接口：`simulated-stream` 與 `real-flight`。保留兩個獨立啟動器，內部共用 Python backend contract；不新增 REST 或 ROS。
4. 接口模式與場域 profile 在程序生命週期內不可變。禁止模擬／真機熱切換，也禁止飛行中或執行中熱換地圖；必須完全關閉後重啟。
5. 模擬接口本階段只保證影片檔。啟動器優先選用工作區內的 P1190119.MP4，否則只有一部匯入影片時自動選用，播放完停在最後一個可解碼畫面，不循環。
6. 模擬介面的起飛、降落、懸停與微移只更新模擬狀態，永不載入 Olympe、連接真機或送出真機命令。
7. 真機接口涵蓋連線與影像、即時定位、人工起飛／降落、人工微移、懸停、手動接管及緊急停止。真機自主路徑飛行保持鎖定，直到外部核准條件全部完成。
8. 地圖及路徑不建立公尺尺度，也不要求 `map_units_per_meter`。地圖座標只作定位、方向、相對路徑進度與畫面顯示。
9. 未來自主路徑的初始實際水平速度上限為 **0.30 m/s**。它是飛機端速度限制，不是 map unit 尺度；必須由真機速度遙測及實機 PCMD 響應測試驗證。速度日後可修改，但只能在確認落地時修改，且每次修改都使原自主飛行核准失效，必須重新測試與核准。
10. 路徑由操作員避開已知障礙物。系統不宣稱具備動態避障或可靠碰撞偵測；稀疏點雲碰撞監控不是正式安全層。每條真機路徑仍須做人工現場淨空審查。
11. 規劃路徑線預設隱藏，操作員可勾選顯示；即時定位軌跡、相機視錐及 XYZ 軸持續顯示。
12. 河濱不是唯一場域。任何具備完整原子化 site profile 的場域都可使用。
13. EDM 河濱 production profile 保留 `max_corr_total=900`。
14. 本次已驗證發布包的正式定位硬體契約是 NVIDIA RTX 5060 + CUDA 12.8，並 fail closed；CPU 只供離線檢查，不可降級成正式飛行定位。其他 GPU 必須重新完成模型、CUDA 與效能驗證後才能另行核准。
15. 系統完全離線運作。模擬模式不可連網；真機模式只允許本機、SkyController／ANAFI 私有網路，不可存取 Internet、DNS 下載模型或執行碼。
16. 主系統維持 Python 3.10；`模擬器/parrot_stimulate` 維持 Python 3.11 獨立環境，但提供一個統一驗證入口。
17. 系統手動啟動，不建立開機或登入自動啟動服務。
18. 高度／距離預設為 30 m／100 m，操作員可在 UI 修改，但只允許在飛機確認落地、連線健康且 firmware readback 可用時套用。
19. 影像、定位或 worker 故障但控制鏈仍在時：立即送零 PCMD、原地懸停、取消自動／電腦連續動作，切換成手動操作。SkyController 搖桿優先；若無 SkyController 但 PC 控制鏈仍健康，保留電腦手動控制。
20. 完全失去飛機控制連線時使用機載安全策略：先懸停／重連；逾時後，GPS 與 Home Point 有效則 RTH，否則依經現場確認的 firmware 策略受控降落。主機端不可假裝斷線後仍能控制飛機。
21. UI 採繁體中文，標準尺寸 1440×900、最低 980×640；SIM 使用藍色識別，REAL 使用紅橘色永久警告。
22. 定位／效能 log 保留 30 天或總量 20 GB，先到者為準；command／safety／incident log 不自動刪除，只能手動封存或匯出。
23. map、bundle、model、runtime profile 與 SHA manifest 的更新由操作員明確核准。
24. 無 ground truth 時不得宣稱公尺級位置誤差或絕對姿態誤差；以固定影格、全部 LOST 恢復事件及完整軌跡的影像人工判讀作為品質驗收。
25. 最終實作完成後，須由 Claude Opus 5 Max 做獨立全系統審核，再由 Codex 執行經操作員接受的修改。若該模型不可用，不得無聲替換，必須先記錄原因並取得操作員同意。

## 2. 目標、非目標與使用者

### 2.1 目標

- 在任意已建圖場域，以同一套 EDM runtime 進行 720p 影像定位。
- 讓單一操作員可安全使用模擬影片或真機 ANAFI，而不可能誤用另一個接口。
- 真機故障時先停止電腦動作，再交回人工，不沿用過期姿態或過期 PCMD。
- 以可重現的資產 SHA、測試、紀錄與監控，證明修改沒有讓目前定位品質或速度退步。
- 未來自主飛行採慢速、短有效期、每次依新 pose 重算的控制，初始速度上限 0.30 m/s。

### 2.2 非目標

- 不增加 REST、HTTP 控制 API、ROS topic/service 或執行中接口切換。
- 模擬接口不保證攝影機、RTSP、網路串流或其他來源，只保證本機影片檔。
- 不把稀疏點雲或 YOLO 當作正式碰撞避免系統。
- 不宣稱絕對公尺定位精度、絕對姿態精度或 ground-truth 飛行誤差。
- 不允許 agent、LLM、腳本或其他自動化替操作員按下真機起飛。
- 本階段不核准真機自主路徑飛行。
- 不在執行時下載模型、程式碼、套件或場域資產。

### 2.3 使用者與角色

| 角色 | 正常模式 | 權限與責任 |
|---|---|---|
| 操作員 | SIM、真機人工飛行 | 選場域、啟動接口、人工起降、微移、懸停、修改落地時的 firmware 限制、判讀定位 |
| 安全監控員 | 僅未來自主真機測試 | 全程握有接管能力，執行第二人確認及 checklist；不得由軟體代替 |
| 系統維護者 | 離線 | 建圖、更新 profile／SHA、執行測試與部署；不得自動觸發真機起飛 |
| 獨立審核者 | 發布前 | 審核整個系統、測試證據與未處理風險，不直接操作真機 |

## 3. As-Is：2026-08-02 盤點（關鍵契約已更新至 2026-08-03）

本節描述 2026-08-02 工作區實際狀態，不把後續要求誤寫成已實作功能。

### 3.1 主機與執行環境

| 項目 | 現況 |
|---|---|
| OS | Ubuntu 22.04.5 LTS，Linux 6.8.0-136-generic |
| 主環境 | `/home/allen/localization/.venv`，Python 3.10.12 |
| PCMD 模擬環境 | `模擬器/parrot_stimulate/.venv`，Python 3.11.15 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB，driver 580.173.02 |
| 正式定位 | EDM PyTorch CUDA FP16；EDM worker 已在 CUDA 不可用時拒絕啟動 |
| 儲存空間 | `/home/allen` 所在檔案系統 140 GB，已使用 91%，只剩約 14 GB |
| 原始碼狀態 | 工作樹有大量既有修改與未追蹤檔案；目前通過測試的狀態尚未形成可重現 release commit/tag |

目前磁碟只剩約 14 GB，因此「最多 20 GB 可自動清理 log」不能單獨作為磁碟安全界線；To-Be 必須另有低磁碟保護。

### 3.2 目前架構

```text
MP4 或 ANAFI PDRAW RGB
        │
        ▼
flight_operator_app.py
        │  兩槽 shared memory，capacity-1/latest-frame
        ▼
live_localizer_worker.py
        ├─ BOOT_INIT / 連續低信心 / LOST：一次性 MegaLoc 全域檢索
        ├─ TRACK / WEAK_TRACK：EDM 局部匹配
        ├─ 2D-3D correspondence
        └─ pycolmap absolute-pose PnP
        │
        ▼
Pose + TRACK / WEAK_TRACK / LOST
        ├─ 桌面 UI
        ├─ JSONL metrics
        └─ 獨立 legacy 路徑控制入口
```

主要所有權如下：

- `控制介面程式/`：site profile、任務入口、桌面 UI、worker protocol、兩個串流 launcher。
- `定位演算法/deploy_code/sfm_glomap_deploy/`：正式 EDM runtime、bundle loader、tracker adapter。
- `定位演算法/flight_control/`：真機路徑控制、安全監控、Olympe 串流與手動工具。
- `定位演算法/validation/`：replay、效能、硬體監控、mirror 與部署檢查。
- `地圖檔/場域/<site>/`：每個場域的 map、bundle、route、report，不納入 Git。
- `模擬器/parrot_stimulate/`：獨立 Python 3.11 Sphinx／PCMD 響應與路徑安全測試工具，不是第三個操作接口，也不能連真機。

### 3.3 目前兩個接口

| 接口 | 現有入口 | 現況 |
|---|---|---|
| 模擬串流 | `控制介面程式/影片模擬串流/選擇啟動.sh` | 固定 `simulated-stream`，拒絕 real/override 參數，不載入 Olympe；由使用者選擇完整場域 profile/PLY 與影片，CLI launcher 預設 river profile 並只在 P119 或唯一影片時自動選擇；EOF 保留最後一幀且不循環 |
| 真機串流 | `控制介面程式/真機串流/啟動.sh` | 固定 `real-flight`，拒絕 `--video`；明確要求 site profile；透過 SkyController 3 或 direct Wi-Fi 連線，PDRAW 失敗不會退回影片 |

兩個 launcher 已有跨接口參數拒絕測試，並共用 `resolve_display.sh`
探測及驗證本機 X/XWayland `DISPLAY`。

### 3.4 場域與建圖資產

- Site profile 必須綁定 UI 顯示 PLY 與 EDM localization bundle；portable full-runtime
  另要求 `query_camera`。reference poses、route、poles 與場域特調 runtime profile 為選配。
- `river_site_edm.json` 目前為 `flight.approved=false`，route 只供顯示。
- 河濱 route 有 20 個 waypoint，但缺少正式 `sfm-flight-route/v1` 的 `site_id`、`coordinate_frame_id`、`purpose` 等欄位，因此不能通過自主飛行 gate。
- 河濱 profile 的 `flight.coordinate_frame_id` 與 controller 都是 `null`，
  `flight.approved=false`；schema v2 不定義也不要求 `map_units_per_meter`。
- 河濱 EDM runtime profile 中的 `S=1.900843` 是參考相機分布所算出的演算法場景尺度，只用於 `radius`／jump gate；它不是 map-unit-to-metre 比例。現有文字 `scale-calibrated` 容易和公尺尺度混淆。
- 同一 profile 實際有提供的 PLY、bundle、reference poses 與 route
  必須來自同一次重建；不同 SfM 重建不可直接混用。

### 3.5 定位現況

- 正式河濱 backend 為 EDM，輸入 1024×576、PyTorch CUDA FP16、coarse top-k 3225。
- TRACK／WEAK／LOST local top-k 為 1／3／5，BOOT MegaLoc top-k 10，先驗證前 2 張，
  不足才展開；LOST scan=2、grace=12、coarse top-k=3225。
- `max_corr_total=900`，inlier gates 80／50／30，prediction max dt 0.25 s。
- worker 使用兩槽 shared memory、latest-frame coalescing、ready handshake、timeout/restart，避免 UI 累積過期影格。
- EDM runtime 已對 checkpoint、bundle、runtime profile 使用 SHA-256；正式 checkpoint loader 使用 `weights_only=True`。
- 正式 EDM worker 已在 CUDA 不可用時 fail closed。
- `offline_model_smoke.py` 會阻擋 DNS/socket 並驗證本地模型；正式 launcher 尚未在 OS 層限制所有外部連線。

### 3.6 P119 現況基準

| 項目 | 值 |
|---|---:|
| 影片 | 匯入到 `模擬器/測試影片/P1190119.MP4`，或透過 `--video` 明確指定 |
| SHA-256 | `600bbf70227311cab079d77fcb896f97e6d3e55f6bc40b5bef01d74b65f7826c` |
| 容器資料 | H.264 Main，2688×1512，約 23.984 fps，宣告 2,935 幀 |
| 實際可解碼 | 2,934 幀，已確認為已知不完整來源 |
| GPU | RTX 5060 Laptop GPU |
| profile | 河濱 EDM，`max_corr_total=900` |
| 成功 pose | 1,686／2,934，57.4642% |
| 狀態數 | TRACK 2,244；WEAK_TRACK 551；LOST 139 |
| processing FPS | 11.7024 |
| wall p50／p95 | 57.134／158.126 ms |
| LOST inference p95／max | 519.554／530.750 ms；每次在下一個 decoded frame 回到 TRACK |
| inliers p50／p95 | 693／875 |
| reprojection RMS p95 | 2.9565 px |

基準證據為 `outputs/validation/edm_p119_pnp_ab_20260802/pnp_cap_900_full.json`。該檔產生時尚未把 stream-integrity verdict 寫入 artifact，因此它只能作為演算法／效能基準，不能代表完整解碼驗收成功。

最近一次循環 UI 記錄的 TRACK submit-to-UI p95 約 116.05 ms，但它未正確包含 280 ms 模擬鏈路的 capture-before-delay 時間，且來源有循環，所以只作資訊，不作正式端到端門檻。

### 3.7 真機控制與安全現況

已存在的保護包括：

- 只有人類可在桌面 UI 按下起飛；agent 與自動化禁止起飛。
- SkyController 搖桿偏轉會強制取回控制權。
- Space 懸停、Esc 手動接管、微移按住才送 PCMD、放開或 heartbeat 過期歸零。
- 關窗、Ctrl+C 或 signal 會嘗試零 PCMD、原地降落，再交還 SkyController。
- 高度／距離預設 30 m／100 m，只有落地時可寫，並要求 firmware ack/readback；距離 geofence 預設開啟。
- 起飛前要求至少 30% 電量；距離 geofence 開啟時要求 GPS fix。
- legacy autonomous runner 有獨立 20 Hz SafetyMonitor、pose freshness、WEAK/LOST、jump、route deviation 與終止安全 gate。
- 目前 UI 真機 backend 的 `start_auto` 只會切成 localization-only PC control，不執行 route。

尚未符合本規格之處：

- 真機 UI 的 stream stale 會送 hover，但尚未一併交還人工。
- Olympe link loss 目前只記錄並顯示警告 `display_only_no_auto_land`，未完成 B1 lost-link firmware policy 驗證。
- UI localization-only 模式的 worker crash／定位 LOST 尚未形成單一、明確的「零 PCMD後交人工」契約。
- 連線時尚未完整讀取並記錄 aircraft/controller 型號、serial、firmware、連線路徑及 RTH/Home policy。
- 現有 `Emergency` safety mode 與 Space／Esc 分散，UI 尚無明確且醒目的「緊急停止電腦動作」控制。

### 3.8 UI、紀錄、部署與測試現況

- UI 已是 1440×900、最小 980×640；route 預設隱藏且有「顯示規劃路徑」checkbox。
- 軌跡、相機視錐及 XYZ 軸已顯示。
- SIM 與 REAL 標題不同，但尚未建立完整藍／紅橘視覺身份與永久 REAL 警告區。
- `loc_metrics_*.jsonl`、`live_ui_cmdlog_*.jsonl`、飛行 telemetry 與硬體監控均已存在。
- 已有 session-based durable logging、disk-pressure gate 與 retention manager；
  localization／performance／video metrics 依 30 天或總量 20 GB 清理，command／
  incident／manifest／summary 不自動刪除。`outputs/flight_logs` 目前約 1.1 GB。
- 系統維持手動啟動。
- 已有 `驗證系統.sh` 單一驗證入口；根 Python 3.10 與 `parrot_stimulate` Python
  3.11 維持分離，由入口依序執行。
- 2026-08-03 最近一次核心驗證：主工作區 757 passed、1 skipped；`parrot_stimulate`
  86 passed；runtime mirrors 8/8；ruff、firmware preflight、`pip check`、flight selftest、profile/SHA、
  workspace layout、CUDA 與完全離線 EDM／MegaLoc 通過。最近一次 P119 全片驗證的
  完整性通過既定 2935/2934 waiver；品質回歸僅 `inliers p50=690` 未達既有 693
  gate，因此該份 P119 receipt 誠實標為 failed，門檻未放寬。

## 4. To-Be：目標架構

```mermaid
flowchart LR
    SP[Site profile + SHA manifest] --> CFG[Immutable SessionConfig]
    MAP[PLY / EDM bundle / camera / route] --> SP

    SIM[影片模擬啟動器] --> CFG
    REAL[真機啟動器] --> CFG
    CFG --> UI[繁體中文 Operator UI]

    UI --> B{OperatorBackend contract}
    B --> SB[SimulatedBackend]
    B --> RB[RealAnafiBackend]

    SB --> VF[本機 MP4 + 720p H264 link simulation]
    RB --> SC[SkyController 3]
    SC --> DRONE[Parrot ANAFI]
    DRONE --> PDRAW[PDRAW 720p]

    VF --> FRAME[Latest FrameSource]
    PDRAW --> FRAME
    FRAME --> SHM[2-slot shared memory]
    SHM --> WORKER[CUDA EDM worker]
    WORKER --> POSE[Pose + TRACK/WEAK/LOST + timing]
    POSE --> UI
    POSE --> SAFE[Independent Safety Supervisor]
    B --> SAFE
    SAFE --> RB

    UI --> LOG[Session / command / localization / incident logs]
    WORKER --> LOG
    SAFE --> LOG
    RB --> LOG
    LOG --> RET[Local retention + manual archive]
```

架構約束：

- `SessionConfig.interface_mode`、`site_profile_sha256` 與所有 asset digest 啟動後唯讀。
- 模擬 backend 與真機 backend 實作同一 Python contract，但打包與 dependency 邊界分離。
- `SimulatedBackend` 不可 import Olympe；`RealAnafiBackend` 不可接受 file video source。
- 安全監控不得依賴 Tk UI tick 或定位 worker；UI 卡住或 worker 崩潰時仍可讓既有 PCMD 過期並歸零。
- 正式 production localizer 只有 EDM；XFeat、NeuFlow 與其他候選方法只留在離線研究／比較路徑。
- 只有兩個操作啟動器；建圖、benchmark、Sphinx、驗證工具是維護工具，不是第三個飛行接口。

## 5. 場域、建圖與資產契約

### 5.1 場域 profile

每個場域必須以單一 profile 原子化選取：

- `site_id`、`display_name` 與 `localizer="edm"`。
- schema/selector/UI 必要的顯示 PLY。
- 姿態估計必要的 EDM localization bundle。
- portable full-runtime 必要、且必須對齊影片管線的 query camera。
- 可選的 map reference poses、EDM runtime profile、route／poles 等資產。
- 實際有提供的場域檔案路徑與 SHA-256；飛行 readiness 另依 schema gate。
- `flight.approved=false` 為新場域預設，任何缺漏均 fail closed。

`coordinate_frame_id` 只識別同一次重建，不代表公尺尺度。To-Be schema 不要求 `map_units_per_meter`。

### 5.2 無公尺尺度控制契約

系統將兩種「尺度」明確分開：

- 演算法場景尺度 `S`：由 reference camera 分布計算，只用於定位 radius／jump gate，單位仍是 map unit。
- 公尺地圖尺度：本系統不建立、不要求，也不從相機軌跡猜測。

未來自主路徑控制使用下列方式避免依賴公尺地圖尺度：

1. 用 pose 到下一 waypoint／lookahead 的 map-space 向量決定方向，送出前只取單位方向。
2. 以已校正的 map-to-body 軸向與機體 yaw 將方向轉成 body forward/right，不把 map 距離換成公尺。
3. 實際速度由 ANAFI 的飛機端速度遙測、短 PCMD pulse 與獨立 20 Hz sender 限制；初始 `speed_limit_mps=0.30`。
4. 每筆 autonomous desired PCMD 的有效期不得超過 0.25 s；沒有更新即歸零。
5. 缺少、過期或不可信的飛機速度遙測時，不允許 autonomous translation。
6. waypoint arrival、route deviation、pose jump 等幾何門檻以場域 profile 的 map unit 欄位保存，不能跨場域複製。
7. 速度上限變更只允許落地時操作；變更後 profile/receipt digest 改變，`flight.approved` 自動視為 false。

真機 PCMD 不是直接的 m/s 命令，因此「0.30 m/s」是必須由速度回授保護的上限，不得只以固定 PCMD 百分比推算。真機 response receipt 尚未產生時，自主功能保持鎖定。

### 5.3 建圖與換圖流程

```mermaid
flowchart TD
    A[收集場域影像] --> B[COLMAP/GLOMAP 重建]
    B --> C[固定 coordinate_frame_id]
    C --> D[輸出顯示 PLY + 選配 reference poses]
    C --> E[建立 EDM localization bundle]
    D --> F[建立 query camera / site profile]
    E --> F
    F --> G[計算並審核 SHA manifest]
    G --> H[離線 replay + 固定影格人工判讀]
    H --> I[操作員核准資產版本]
    I --> J[重啟 SIM 或 REAL 載入該 profile]
```

需求：

- 同一 profile 實際有提供的 map、bundle、reference poses、route 不可跨重建混用；
  真機 route 另必須和 `flight.coordinate_frame_id` 一致。
- 新場域第一次只允許模擬／地面定位；`flight.approved` 必須為 false。
- 換地圖只能關閉目前 UI 後，使用另一 profile 重啟。
- profile 或任一資產 SHA 改變後，舊測試及飛行核准全部失效。
- 操作員負責畫出避開已知障礙物的 route；真機使用前仍要在現場逐段確認淨空。
- 系統不得把稀疏點雲空白區解讀成「沒有障礙物」。

### 5.4 Route 顯示與飛行格式

- 顯示 route 可維持目前簡單 `{frame, units, waypoints}` 格式，但必須明確標記 `purpose=display` 或由 profile 標記 display-only。
- 真機自主 route 必須有正式 schema、`site_id`、`coordinate_frame_id`、`purpose=flight`、有限的三維 waypoint、open polyline、SHA-256 與淨空核准。
- UI 預設不畫 route line；checkbox 只改顯示，不改飛行核准或控制狀態。

## 6. 兩個接口規格

### 6.1 模擬串流接口

唯一入口：

```bash
./控制介面程式/影片模擬串流/啟動.sh [影片檔] [UI 額外參數]
```

預設：

| 項目 | 值 |
|---|---|
| interface | `simulated-stream`，唯讀 |
| site profile | `控制介面程式/site_profiles/river_site_edm.json` |
| video | `模擬器/測試影片/` 內的 P1190119.MP4，或單一匯入影片；多部時必須明確指定 |
| output stream | 1280×720、30 fps、H.264 Main、5,000 kbps |
| artificial latency | 280 ms |
| artificial loss | 0% nominal |
| EOF | 保留最後一個完整 decoded frame，狀態 `EOF_HOLD`，不重新開檔、不循環 |
| localization | CUDA EDM，與真機相同的 BOOT/TRACK/WEAK/LOST pipeline |
| controls | 只修改模擬 pose／flight state |

壓力 preset 必須透過同一啟動器提供，不增加新接口：

- `nominal`：280 ms、0%。
- `loss-1`：280 ms、1%。
- `loss-3`：280 ms、3%。
- `loss-5`：280 ms、5%。

百分比代表合成 slice/NAL loss，僅供壓力測試，不宣稱等於某個真實 Wi-Fi 場景。

若預設影片不存在或 SHA 不符，啟動失敗並清楚提示；不可換成其他影片靜默啟動。操作員可明確傳入另一影片覆寫預設。

P119 容器宣告 2,935 幀但只解碼 2,934 幀。對這個固定 SHA，UI 可播放並以 `KNOWN_INCOMPLETE` 標示；完整性 benchmark 必須產生非通過 verdict，除非驗收命令明確使用這項已核准 waiver。任何不同 SHA 或不同可解碼幀數都不是同一份已知例外。

### 6.2 真機接口

唯一入口：

```bash
SFM_SITE_PROFILE=/absolute/path/to/site.json \
  ./控制介面程式/真機串流/啟動.sh
```

正式飛行連線以 Parrot ANAFI 4K + SkyController 3 為目前預期組合。連線後必須從設備讀取並記錄實際值，而不是只相信啟動參數：

- aircraft product/model、serial、hardware revision、firmware version。
- controller product/model、serial、firmware／software version。
- Olympe version、連線 IP／transport、是否經 SkyController。
- PDRAW codec、解析度、fps 與 source timestamp 能力。
- battery、GPS、Home Point、RTH/lost-link policy、firmware 高度／距離設定與允許範圍。

若設備資訊讀不到、型號與核准清單不符或 firmware 尚未測試，仍可在地面顯示診斷，但必須阻止電腦控制起飛與自主功能。

正式飛行一律經 SkyController 3，確保人工搖桿接管。Direct drone Wi-Fi 只屬地面診斷／實驗模式，沒有 SkyController 時不得啟用自主路徑；是否允許人工 PC 起飛仍受獨立現場核准。

真機接口不得接受 `--video`，PDRAW 不可用時不得用影片代替。啟動時必須明確指定 site profile；不得猜河濱或沿用上一次場域。

### 6.3 模式與地圖不可熱切換

- UI 不提供 SIM/REAL toggle。
- backend 建立後不得替換。
- `interface_mode`、site profile path、profile SHA、asset SHA 在 session log 首筆寫入並保持不變。
- 任何切換需求都執行安全關閉、結束 worker、釋放 shared memory，再由另一啟動器建立新 session。

## 7. 共用 Python backend contract

此 contract 是同一程序內的 Python 契約，不是 REST/ROS。實作可使用 `typing.Protocol` 或 ABC，但不得再以未檢查的任意 command string 作唯一契約。

```python
class OperatorBackend(Protocol):
    mode: Literal["simulated-stream", "real-flight"]
    is_live: bool
    state: OperatorState
    video: FrameSource

    def start(self, config: SessionConfig) -> StartResult: ...
    def poll(self, now_mono_ns: int) -> OperatorState: ...
    def command(self, request: ControlRequest) -> ControlResult: ...
    def fail_safe(self, reason: FailureReason) -> ControlResult: ...
    def close(self, reason: str) -> CloseResult: ...

class FrameSource(Protocol):
    def next_frame(self, *, only_new: bool = True) -> FramePacket | None: ...
    def close(self) -> None: ...
```

必要資料：

- `SessionConfig`：session ID、固定 interface、site profile path/SHA、asset SHA、runtime profile SHA、video 或 real endpoint、firmware limits、offline policy。
- `FramePacket`：RGB、sequence、source timestamp、host receipt timestamp、source identity、`eof`、decode／link timing。
- `OperatorState`：connection、stream、localization、flight state、control owner、pose、pose freshness、battery、GPS/Home、limits readback、last command、active incident。
- `ControlRequest`：enum action、typed payload、request ID、host monotonic timestamp、human-origin flag。
- `ControlResult`：accepted、executed、reason code、ack/readback、resulting state、monotonic timestamp。
- `FailureReason`：stream stale、localization WEAK/LOST、pose stale、worker exit/stall、UI heartbeat lost、control link lost、invalid telemetry、disk/log failure、shutdown。

必要 action 與語意：

| Action | SIM | REAL |
|---|---|---|
| `TAKEOFF` | 只改模擬狀態 | 只接受 UI 人類按鍵；執行全部 preflight |
| `LAND` | 模擬落地 | 一次性 Landing，確認結果並記錄 |
| `HOVER` | 模擬速度歸零 | 立即零 PCMD |
| `MANUAL` | 模擬模式切換 | 零 PCMD，優先交還 SkyController；無 SC 時保留 PC 手動但不恢復 AUTO |
| `NUDGE_BEGIN/HEARTBEAT/END` | 改模擬 pose；deadman 相同 | 有限 PCMD，放開／失焦／TTL 到期歸零 |
| `EMERGENCY_STOP` | 清空所有模擬動作並懸停 | 最高優先：取消 pending command、零 PCMD、停止 AUTO／nudge，交人工；不等同空中切斷馬達 |
| `LAND_NOW` | 模擬降落 | 明確且獨立的緊急降落命令 |
| `APPLY_LIMITS` | 更新模擬 HUD | 只在確認落地時寫 firmware，ack/readback 不一致即失敗 |
| `START_LOCALIZATION` | 啟動 worker feed | 啟動 worker feed，不取回 PC 控制、不起飛 |
| `START_AUTO` | 可進入純模擬 route test | 預設回 `LOCKED_EXTERNAL_APPROVAL`；不得只靠按鈕解鎖 |

未知 action、缺 payload、錯誤狀態或 backend 不支援時，必須回傳明確拒絕理由，不可只寫 log 後假裝成功。

## 8. 操作介面規格

### 8.1 版面與身份

- 語言：繁體中文。
- 標準視窗：1440×900；最小：980×640。
- SIM／REAL 身份以文字區分，不可只靠顏色，避免色覺辨識問題。
- **2026-08-03 操作員決定**：移除最上方的常駐身份列。身份改由影像面板 HUD
  的 `video_hud_identity()` 提供（`SIMULATED ANAFI` / `REAL ANAFI`）。
  **已知代價**：沒有影格可畫時（開機、串流中斷、EOF 之前的空窗）畫面上不會
  出現任何 SIM／REAL 標示。此決定以節省版面為由做出，非疏漏；要恢復常駐標示
  就把身份列加回 `_build_ui`。
- REAL 仍須顯示 aircraft/controller identity、site ID、control owner 及自主鎖定狀態
  （`控制介面程式/operator_interface` 的 ANAFI／控制權面板）。
- LINK LOST、STREAM LOST、WORKER DOWN、LOGGING FAILED 使用最高優先全寬警告
  （`incident_banner`），且只在事件發生時佔用版面；`安全狀態：正常` 這類閒置列不常駐。
- **2026-08-03 操作員決定**：LOCALIZATION LOST／LOW CONFIDENCE／LOST hold 不再有
  全寬警告列，只保留影像面板內的 banner（`render_video`）與「定位儀表」面板的
  `loc_health_label`。**已知代價**：該 banner 被裁切在影像面板內、受 video dirty key
  節流，不具全寬最高優先性質。此決定以避免與介面中段資訊重複為由做出，非疏漏。

### 8.2 地圖與影像

- 左側顯示點雲、即時定位軌跡、相機中心／視錐、XYZ／UP 軸。
- 規劃 route 預設隱藏；「顯示規劃路徑」只控制 overlay。
- 實際軌跡與規劃 route 必須使用不同顏色與圖例。
- 不可因 LOST 把最後姿態當成新 pose 繼續畫線；LOST 區段以斷線或不同健康標記呈現。
- 右側顯示目前 source、frame sequence、source age、stream FPS、localization FPS、p95 latency、inliers、reprojection、TRACK/WEAK/LOST。
- **2026-08-03 操作員決定**：移除地圖上方的固定管線摘要列（與「定位儀表」面板重複）。
  定位結果抵達 UI 的 5 秒滾動 FPS、串流 5 秒滾動 FPS、submit-to-UI 端到端延遲與其
  5 秒 p95 現在只存在於控制區的「定位儀表」面板。**已知代價**：該面板位於可捲動的
  控制區內，捲動後即不可見。計數語意不變 —— SIM 標示「實機鏈路模擬」，計數點位於
  720p30／H.264 Main／5 Mb/s／280 ms backlog／選用丟包模擬之後，且須等 EDM 結果抵達
  UI 才計數；REAL 由 PDRAW 影格經 EDM 到 UI 的實際結果計數；無可信樣本顯示 `N/A`，
  不得以核心理論吞吐量替代端到端 FPS。`format_pipeline_metrics_summary()` 與
  `OperatorApp.pipeline_metrics_summary()` 保留（目前只有測試使用），要恢復摘要列
  直接接回 `_build_ui` 即可。
- 影格年齡與位姿年齡改由飛行列右側的固定讀數提供，隨門檻變色（350／750 ms）。
- SIM 到 EOF 時畫面保持最後一幀並明確顯示 `EOF_HOLD`，不再增加 frame sequence。
- 可捲動控制區必須有獨立「飛控遙測（Olympe 讀回）」面板，顯示飛控融合
  roll/pitch/yaw、相對起飛點高度、AGL、NED 地速、GPS 位置／精度／衛星、
  heading/RTH、風／震動／懸停警告、RSSI／鏈路品質與 IMU／barometer／GPS 等感測器
  健康。這些值只從 Olympe event cache 讀取，不得被當成定位結果或 PCMD 回應。
  Original ANAFI 公開事件沒有 raw IMU 與 raw 氣壓數值，介面必須明確標示為
  飛控融合估計／感測器健康，不可偽裝成 raw telemetry。

### 8.3 控制與優先權

控制優先順序固定為：

```text
EMERGENCY_STOP / lost-link firmware policy
    > LAND_NOW / cleanup landing
    > MANUAL / stick takeover / HOVER
    > manual nudge
    > future autonomous route
```

- Space 永遠為 HOVER。
- Esc 永遠為 MANUAL／交回搖桿。
- 微移必須 hold-to-move，release／focus loss／heartbeat timeout 都歸零。
- 「開始定位」不能取回 PC control、起飛或啟動 route。
- 「恢復電腦控制」只能進入 PC manual，不能自動恢復 autonomous。
- 發生安全故障後，即使定位恢復也不自動重新進入 autonomous；需要新的人工核准流程。

### 8.4 高度與距離

- 預設 desired limit：高度 30 m、距離 100 m、distance geofence ON。
- UI 永遠同時顯示 desired value 與 aircraft readback，不得混為一個值。
- 只在 `LANDED + LINK_OK + bounds_known` 時允許套用；值必須為正、在 firmware bounds 內、寫入後 readback 一致。
- 接近 80% 顯示黃色；接近 95% 或到達限制顯示紅色。
- PC route/manual controller 在 firmware 前先消除繼續向上／向外的命令；firmware geofence 是最後保護層。
- 到達高度／距離限制時阻止繼續上升／向外，不因碰到 geofence 自動 RTH。RTH 只用於獨立的 lost-link／返航流程。

## 9. 定位系統規格

### 9.1 Production pipeline

```text
BOOT_INIT
  MegaLoc retrieval，一次
  -> EDM batch local matching
  -> 2D-3D correspondence
  -> PnP
  -> TRACK

TRACK
  EDM local_topk=1
  -> WEAK_TRACK 時 topk=3
  -> 連續 2 筆低信心時先懸停／凍幀，再執行一次 MegaLoc recovery
  -> LOST 時 topk=5 + 每個 LOST episode 一次 MegaLoc
```

固定要求：

- MegaLoc 只由三種自動事件觸發：落地狀態的起飛初始化 `BOOT_INIT` 一次、連續 2 筆低信心 EDM 結果後一次，以及每個真正 `LOST` episode 進入時一次。單筆 `WEAK_TRACK` 只提高 EDM top-k；低信心升級前 SIM 凍幀、REAL 歸零 PCMD 並交還人工。正式操作 UI 不得提供手動強制 global retrieval／MegaLoc 的控制。
- 河濱 `max_corr_total=900`。
- 正式 device 必須是 `cuda` 且 GPU identity 符合核准 deployment；CPU fallback 禁止。
- query camera 先驗 schema 與幾何參數；bundle、runtime profile 與 EDM checkpoint
  依各自資產合約驗 SHA-256 和結構後再載入。
- 執行時不得下載 MegaLoc、EDM、XFeat、Hugging Face 或 torch hub 資產。
- shared-memory owner 只由 UI 建立／unlink，worker 只能 attach／detach。
- worker busy 時只保留最新 frame，不排隊重播過期 frame。
- source capture、host receipt、worker start/end、UI arrival 使用 monotonic timestamp 並保留 clock semantics。
- GPU profiling synchronization 預設關閉；開啟 profiling 的 FPS 不可當 production FPS。

### 9.2 安全輸出條件

Pose 只有在下列條件全部通過時才能成為可控制 pose：

- XYZ、rotation、reprojection、timestamp 都有限且 schema 正確。
- source timestamp 未過期。
- inlier、reprojection 與 tracker state 通過 runtime profile。
- jump／trajectory continuity gate 通過，或有第二個一致 fix 確認重定位。
- pose 所屬 site/profile/session 與當前 session 相同。

WEAK、LOST、stale pose 或 worker error 都不可沿用上一筆非零 autonomous command。

### 9.3 無 ground truth 的驗收

- 不輸出「誤差 X 公尺」或「姿態誤差 X 度」的正式 claim。
- 固定抽查影格：0、293、587、880、1174、1467、1760、2054、2347、2641、2933。
- 另外抽查每一個 LOST entry、LOST recovery 及前後各一個成功 frame。
- 產生完整軌跡 overlay、狀態色段、跳點清單與 route-relative 視圖供操作員判讀。
- 操作員只核准「大致位置正確／不正確」、「軌跡連續／不連續」與具體例外，不把人工判讀換算成公尺精度。

## 10. 真機安全需求

### 10.1 起飛前置條件

人工起飛按鈕只有在下列條件全部為真時可用：

- interface 固定為 `real-flight`，site profile 與所有 SHA 已驗證。
- aircraft/controller identity、firmware、Olympe、連線路徑已讀取並記錄。
- ANAFI 羅盤狀態已明確回讀且不是 required／failed／calibrating；經
  SkyController 3 連線時，控制器羅盤也必須明確為 Calibrated。firmware 僅回報
  recommended 時顯示警告但不封鎖人工起飛。
- SkyController 搖桿接管已實測；正式飛行不可只相信設定值。
- battery 至少 30%。
- GPS fix、Home Point 與 firmware lost-link/RTH policy 符合 B1。
- 高度 30 m／距離 100 m 或操作員當次設定值已 ack/readback。
- distance geofence 已確認開啟。
- 影像串流、定位 worker、CUDA、logging、磁碟空間健康。
- 真機起飛只能由操作員在本機 UI 實際按下；不得有 CLI、環境變數或自動化旁路。

### 10.2 自主飛行外部核准 gate

所有場域維持 `flight.approved=false`，直到書面／artifact 證據齊全：

1. 同一次 SfM 重建的 coordinate frame 與資產 SHA 綁定確認。
2. 操作員逐段 route 淨空審查。
3. 拆槳 yaw、body-axis、PCMD 正負號與 stick takeover 測試。
4. 在 `landed` 狀態由操作員分別完成 ANAFI 與 SkyController 3 韌體羅盤校正，
   並確認介面能回讀每一軸進度；校正命令不得啟動馬達或移動機體。
5. 真機 PCMD response 測試，證明速度回授與 0.30 m/s 上限可執行；這不是 map scale 校正。
6. 地面測試與低高度、低速測試。
7. 真機 PDRAW + EDM 定位驗證。
8. frame/body 軸向、機體 yaw、camera-to-body yaw offset 校正；不要求 map distance scale。
9. 高度、距離、lost-link、RTH/Home、低電量及 logging fail-closed 測試。
10. `parrot_stimulate` 共用控制器測試、固定 worst-case 測試全部通過。
11. 兩人確認、實體安全監控員及起飛前 checklist 已落實。
12. 操作員明確更新 `flight.approved=true` 並核准新 SHA manifest。

### 10.3 0.30 m/s scale-free 自主控制

- 真機自主 route 實作不得繼續使用目前依賴 `map_units_per_meter` 的 continuous-polyline conversion。
- 正式 route controller 必須和 `parrot_stimulate` 的逐點控制邏輯共用單一實作或單一權威核心，不得維護兩套 PCMD 公式。
- map-space target vector 只提供方向；實際速度只採 airframe telemetry。
- 先轉向、確認穩定、再以短時限 PCMD 平移；每次取得新 pose 後重算。
- 任何速度超過 0.30 m/s、速度未知、pose 過期、WEAK/LOST、route deviation 或 command TTL 過期都立即歸零並進入人工模式。
- 使用者日後可修改速度，但只能落地修改；修改後必須重新做 PCMD response、braking、low-altitude 與 worst-case 測試。

### 10.4 故障策略

| 故障 | 立即動作 | 後續狀態 | 自動恢復 AUTO |
|---|---|---|---|
| 影像 stale／中斷 | 清除 desired PCMD、送零、懸停 | SkyController 手動；無 SC 但 PC link 健康則 PC manual | 否 |
| 定位 WEAK／LOST／pose stale | 同上；停止使用舊 pose | 手動，定位可在背景重抓 | 否 |
| worker exit／stall／OOM | 獨立 safety supervisor 先歸零與交人工，再重啟 worker | 手動 | 否 |
| UI freeze／focus loss | nudge heartbeat/desired PCMD TTL 到期歸零 | 手動或 hover | 否 |
| SkyController stick movement | 立即停止 PC PCMD並交回 sticks | SkyController manual | 否 |
| PC 與 SkyController/aircraft 完全斷線 | 主機停止假設可控；機載懸停／重連 | GPS+Home 有效逾時 RTH；否則經驗證的受控降落 | 否 |
| battery/GPS/Home/limit readback 不合格 | 起飛前拒絕；空中依 firmware 與操作員處置 | 手動／安全降落 | 否 |
| logging 或磁碟無法保證 safety log | 起飛前拒絕；空中發出 incident 並交人工 | 手動 | 否 |
| 關窗／Ctrl+C／SIGTERM | 零 PCMD、停止錄影、原地 Landing、交還 sticks | CLOSED | 不適用 |

### 10.5 緊急停止定義

「緊急停止」指停止電腦／自主移動，不是空中切斷馬達：

- 原子取消所有 pending control request。
- desired PCMD 設為零且鎖存，不因晚到 worker result 恢復。
- 清空所有 nudge hold。
- 切換 SkyController 手動；若只有健康 PC link，留在 PC manual zero state。
- 需要降落時由獨立 `LAND_NOW` 執行。
- 任何 motor emergency／cut-out 功能不放在一般 UI，除非另有硬體製造商程序與外部核准。

## 11. 狀態機

### 11.1 Session／接口狀態

```mermaid
stateDiagram-v2
    [*] --> STOPPED
    STOPPED --> STARTING_SIM: 模擬啟動器
    STOPPED --> STARTING_REAL: 真機啟動器
    STARTING_SIM --> SIM_READY: profile/video/CUDA 通過
    STARTING_REAL --> REAL_GROUND_MANUAL: profile/device/preflight 通過
    STARTING_SIM --> START_FAILED: 任一 gate 失敗
    STARTING_REAL --> START_FAILED: 任一 gate 失敗
    SIM_READY --> SIM_RUNNING
    SIM_RUNNING --> EOF_HOLD: 最後 decoded frame
    REAL_GROUND_MANUAL --> REAL_AIRBORNE_MANUAL: 人類按起飛
    REAL_AIRBORNE_MANUAL --> REAL_GROUND_MANUAL: Landing confirmed
    EOF_HOLD --> STOPPING
    SIM_RUNNING --> STOPPING
    REAL_GROUND_MANUAL --> STOPPING
    REAL_AIRBORNE_MANUAL --> STOPPING: cleanup landing
    STOPPING --> STOPPED
    START_FAILED --> STOPPED
```

SIM 與 REAL 之間沒有 transition；只能回到 STOPPED 後由另一啟動器重建 session。

### 11.2 定位狀態

```mermaid
stateDiagram-v2
    [*] --> WORKER_LOADING
    WORKER_LOADING --> BOOT_INIT: ready handshake
    BOOT_INIT --> TRACK: strong pose
    BOOT_INIT --> LOST: lock timeout/failure
    TRACK --> WEAK_TRACK: weak gate
    WEAK_TRACK --> TRACK: strong pose
    TRACK --> LOST: consecutive failure/stale
    WEAK_TRACK --> LOST: consecutive failure/stale
    LOST --> TRACK: confirmed recovery
    LOST --> MANUAL_HOLD: REAL failure policy
    WORKER_LOADING --> WORKER_DOWN: exit/stall
    TRACK --> WORKER_DOWN: exit/stall
    WEAK_TRACK --> WORKER_DOWN: exit/stall
    LOST --> WORKER_DOWN: exit/stall
    WORKER_DOWN --> WORKER_LOADING: bounded restart
```

SIM 在 LOST 可凍住影片做 bounded recovery；REAL 不可凍 camera，只能懸停並交人工。

### 11.3 控制權狀態

```mermaid
stateDiagram-v2
    [*] --> GROUND_STICKS
    GROUND_STICKS --> AIRBORNE_STICKS: human TAKEOFF
    AIRBORNE_STICKS --> PC_MANUAL: explicit PC control
    PC_MANUAL --> AIRBORNE_STICKS: Esc/stick input/failure
    PC_MANUAL --> HOVER_HANDOFF: stream/pose/worker failure
    FUTURE_AUTO --> HOVER_HANDOFF: any safety failure
    HOVER_HANDOFF --> AIRBORNE_STICKS: SkyController healthy
    HOVER_HANDOFF --> PC_MANUAL_ZERO: no SC, PC link healthy
    AIRBORNE_STICKS --> LANDING: LAND
    PC_MANUAL --> LANDING: LAND
    HOVER_HANDOFF --> LANDING: LAND_NOW
    LANDING --> GROUND_STICKS: touchdown confirmed
    AIRBORNE_STICKS --> LINK_LOST_ONBOARD: total link loss
    PC_MANUAL --> LINK_LOST_ONBOARD: total link loss
    LINK_LOST_ONBOARD --> RTH: timeout + GPS/Home valid
    LINK_LOST_ONBOARD --> ONBOARD_LAND: no valid GPS/Home
```

`FUTURE_AUTO` 在本階段不可進入。

## 12. 效能與品質 KPI

### 12.1 不退步原則

- 先固定 input/profile/model/bundle SHA、RTX 5060、driver/power profile、P119、PnP seed 與 benchmark command。
- 品質 gate 使用固定輸入的 deterministic count，不給「效能誤差」掩蓋品質退步。
- 時間／FPS 取三次 warmed full run 的中位數，排除主機更新、thermal throttle 與 profiling mode；中位數不得比基準差。
- 任何基準更新都需要操作員核准新 artifact/SHA，不可由測試自動接受新結果。

### 12.2 正式 gate

| KPI | 基準／要求 | 判定 |
|---|---:|---|
| P119 decoded frames | 固定 SHA 應為 2,934，容器宣告 2,935 | 必須標記 `KNOWN_INCOMPLETE`；其他數量 fail |
| Pose successes | ≥1,686／2,934 | 低於即品質回歸 |
| TRACK count | ≥2,244 | 低於即品質回歸 |
| LOST count | ≤139 | 高於即品質回歸 |
| inliers p50／p95 | ≥693／875 | 任一低於即品質回歸 |
| reprojection RMS p95 | ≤2.9565 px | 高於即品質回歸 |
| processing FPS | ≥11.7024 | 三次 warmed run 中位數 |
| wall p95 | ≤158.126 ms | 三次 warmed run 中位數 |
| LOST inference max | ≤530.750 ms | 不得變慢；每個 episode 必須有 recovery／terminal verdict |
| 正常 TRACK submit-to-UI p95 | 目前資訊值 116.05 ms | 修正 source timestamp 後先封存 B0；之後不得退步 |
| nominal source-to-UI p95 | 尚無可信 B0 | 必須把 280 ms link delay 納入 timestamp，產生 B0 後才能宣稱達標 |
| worker busy queue depth | ≤1 latest frame | 不允許舊 frame backlog |
| safety command observation-to-zero | ≤100 ms | 由獨立 safety log 驗證；不含故障偵測門檻 |
| pose freshness cutoff | ≤500 ms | 過期即不得控制 |
| stream stale cutoff | ≤750 ms | 過期即 hover/manual |
| nudge deadman | ≤250 ms | UI freeze/release 後必須歸零 |

「定位成功率」只表示演算法在該影片成功輸出通過 gate 的 pose，不表示絕對位置正確。正式品質判定還要通過第 9.3 節的人工影像審查。

## 13. 錯誤處理

| 錯誤 | SIM | REAL | Log／UI |
|---|---|---|---|
| profile／asset 不存在 | 拒絕啟動 | 拒絕啟動 | 列出缺少路徑與 site ID |
| SHA mismatch | 拒絕啟動 | 拒絕啟動 | expected/actual，不載入 artifact |
| CUDA 不可用／GPU 不符 | 拒絕 production localization | 拒絕真機定位與起飛 | 顯示 driver/GPU 診斷，不降級 CPU |
| 影片目錄沒影片／多部未指定 | 拒絕啟動 | 不適用 | 提示匯入影片或明確指定 `VIDEO` |
| P119 2,934/2,935 | 播放、標示已知不完整、EOF hold | 不適用 | integrity verdict 非正常完整 |
| 一般影片 decode error | 保留最後完整 frame，進 `DECODE_ERROR_HOLD` | 不適用 | frame index、ffmpeg status |
| PDRAW 無 frame | 不適用 | 零 PCMD、懸停、手動 | 全寬紅色警告與 incident |
| worker 啟動太久 | 保持首幀／不播放 | 地面不得起飛；空中交人工 | ready/error/log path |
| worker stall／exit | bounded restart | 先安全 handoff，再 restart | 每次 restart reason/count |
| 非有限 pose／錯 schema | 當定位失敗 | 當定位失敗並安全 handoff | 原始錯誤不得污染 state |
| control request 不合法 | 明確拒絕 | 明確拒絕 | request ID、reason code |
| firmware write/readback mismatch | 更新模擬狀態失敗 | 拒絕起飛 | desired、bounds、readback |
| log 開檔／寫入失敗 | 顯示警告，可停止模擬 | 起飛前 fail closed；空中交人工 | stderr fallback + incident banner |
| 磁碟低空間 | 清理可刪 log，必要時停止新模擬記錄 | 清理可刪 log；不足以保證 safety log 時拒絕起飛 | free bytes、purged files、blocked reason |

## 14. 紀錄與監控

### 14.1 Session log

每次啟動建立唯一 session directory 或同 session ID 的 JSONL 組：

- `session_manifest.json`：mode、site/profile/assets/model SHA、video SHA 或 hardware identity、Python/package/CUDA/driver、啟動參數、offline policy。
- `commands.jsonl`：request、ack/readback、control owner、實際送出的 PCMD、human-origin、時間戳。
- `localization.jsonl`：frame/source timestamps、state、pose、inliers、reprojection、latency、worker lifecycle。
- `telemetry.jsonl`：battery、GPS/Home、attitude、airframe speed、height/distance、link、firmware limits。
- `incidents.jsonl`：所有 safety transition、失鎖、handoff、lost link、logging/disk failure、shutdown/landing verdict。
- `session_summary.json`：結束原因、最大／p95 指標、事件計數、未確認 landing 或其他未結案項目。

每筆安全相關資料同時保留 UTC wall clock 與 host monotonic ns。不得只靠 UI 文字 log。

### 14.2 保留策略

- localization/performance/video-derived log：30 天或總量 20 GB，先到者為準，按最舊 session 清理。
- command/safety/incident/session manifest/summary：不自動刪除。
- 手動 archive/export 產生 manifest、SHA-256 與檔案清單。
- 自動清理前後都寫 retention audit，不可刪除當前 session。
- 磁碟 free space <20 GB 或 <15% 時警告並清理可刪 log；<5 GB 或 <5% 時禁止新的真機起飛。門檻採較早觸發者。
- 目前只有約 14 GB free，實作完成前必須先清理／擴充空間或調整經操作員核准的部署容量；不可藉由刪除永久 safety log 解決。

### 14.3 本機監控

完全離線，因此不依賴雲端監控。UI／local monitor 顯示：

- worker ready/restart/stall、CUDA device/memory/OOM。
- stream FPS、frame drop/coalesce、source age。
- TRACK FPS、p50/p95 latency、WEAK/LOST/recovery。
- CPU/GPU temperature、clock、power、thermal slowdown。
- control link、PDRAW、GPS/Home、battery、airframe speed。
- disk free、log write status、retention action。

硬體監控為 read-only，不使用 sudo 取消 thermal protection 或永久修改功耗限制。

## 15. 離線、安全與資產治理

- SIM process 的 outbound network 必須完全禁止，loopback IPC 除外。
- REAL process 只允許 loopback 與已解析的 `192.168.53.1`／`192.168.42.1` 等核准 ANAFI 私有 endpoint；不得查 DNS 或連 Internet。
- 設定 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、固定 package-local cache；任何 cache 缺失都 fail closed。
- model/checkpoint/bundle/profile/route/reference poses/map manifest 在載入前驗 SHA-256。
- PyTorch artifact 優先使用 `weights_only=True`、受限 safe globals 及 schema validation。任何仍使用 unrestricted pickle 的維護工具必須先驗 SHA，且不得進 production startup path。
- command log 不記錄 secret；工作區不可要求 Internet token 才能飛行。
- 操作員核准 artifact 更新時，receipt 必須包含舊／新 SHA、變更理由、測試結果、人工視覺判定與日期。
- working tree 未形成 release receipt 前，不得宣稱可重現部署。

## 16. 部署與操作

### 16.1 部署需求

- Ubuntu 22.04 LTS deployment profile。
- RTX 5060 + 已核准 NVIDIA driver/CUDA；啟動時驗證 GPU identity 與 CUDA smoke。
- 主 Python 3.10 environment 維持 UI、EDM、pycolmap、Olympe 相容依賴。
- `parrot_stimulate` Python 3.11 environment 維持獨立，不被 root pytest 跨版本收集。
- 所有依賴、torch hub repo、模型、權重與 firmware test asset 在部署前本地備妥並驗 manifest。
- 不建立 system boot/login service；由操作員手動執行 SIM 或 REAL launcher。

### 16.2 DISPLAY 自動偵測

Launcher 行為：

1. 已存在且可連線的 `WAYLAND_DISPLAY`／`DISPLAY` 優先。
2. 否則檢查目前登入圖形 session，找出可用 X/Wayland display 與對應 runtime/auth。
3. 可驗證地依序探測常見 display，不只硬寫 `:0` 或 `:1`。
4. 找不到可用桌面時清楚失敗，不在背景啟動無人可見的真機 UI。
5. dry-run 顯示選到的 display、Python、mode、profile 與完整安全摘要，但不連接飛機。

### 16.3 單一驗證入口

To-Be 提供一個不連真機、不起飛的入口，例如：

```bash
./驗證系統.sh
```

它依序執行並彙整：

1. 主 Python 3.10 `pytest -q`。
2. Python 3.11 `parrot_stimulate` tests、ruff check、format check。
3. runtime mirror check。
4. flight selftest。
5. 主環境及模擬環境 dependency check。
6. offline model smoke。
7. artifact/SHA/profile schema validation。
8. CUDA/GPU smoke；沒有 GPU 時整體 production verdict 失敗，但可另外輸出 offline-only 診斷。
9. 選配 P119 full regression；已知 incomplete verdict 必須明確呈現。

輸出 machine-readable receipt，任何必須項失敗時整體 exit code 非零。

## 17. 測試矩陣

| ID | 層級 | 測試 | 環境 | 通過條件 |
|---|---|---|---|---|
| T-001 | Unit | site profile/schema/SHA/coordinate frame | CPU | 完整接受，缺漏與混用 fail closed |
| T-002 | Unit | backend typed command/state contract | CPU | SIM/REAL contract tests 全通過，未知 command 明確拒絕 |
| T-003 | Unit | launcher cross-interface rejection | CPU | SIM 拒 real flags；REAL 拒 video |
| T-004 | Integration | DISPLAY 探測 `:0`／`:1`／Wayland／無桌面 | 桌面 mock | 選可用桌面；無桌面 fail |
| T-005 | Integration | P119 default + EOF hold | RTX 5060 | 2,934 decoded，最後 sequence 不再增加、不循環、顯示 known incomplete |
| T-006 | Integration | nominal/loss-1/loss-3/loss-5 | RTX 5060 | 每個 preset 有 manifest、無 crash、事件與品質報告完整 |
| T-007 | Integration | shared-memory worker lifecycle | RTX 5060 | stall/kill/restart 後可恢復，owner buffer 不被 worker unlink |
| T-008 | Regression | P119 EDM `max_corr_total=900` | RTX 5060 | 第 12 節所有品質／效能 gate 通過 |
| T-009 | Human QA | 固定影格、所有 LOST、完整軌跡 | UI | 操作員簽核，不產生公尺精度 claim |
| T-010 | Security | 完全離線模型載入 | 網路阻斷 | 無 DNS/socket；缺 cache fail closed |
| T-011 | Security | SIM Olympe/network isolation | CPU | Olympe 未 import、無外部 socket、無真機 command |
| T-012 | Safety | CUDA unavailable／wrong GPU | CPU/mock | production localization、REAL takeoff blocked |
| T-013 | Safety | nudge press/release/focus loss/UI freeze | mock backend | ≤250 ms 零 PCMD，無 stale resend |
| T-014 | Safety | stream/pose/worker failure | mock + Sphinx | ≤100 ms observation-to-zero，切人工，不 auto-resume |
| T-015 | Safety | stick takeover | SkyController props-off | 任一 deliberate stick input 立即取回 |
| T-016 | Safety | height/distance write and limit | mock + props-off | landed-only、bounds、ack/readback、越界命令被 clamp |
| T-017 | Safety | total link loss B1 | props-off/ground first | onboard hover/reconnect；GPS/Home 有效走 RTH policy，否則驗證的 land policy |
| T-018 | Hardware | device inventory | 真機地面 | model/serial/firmware/controller/video/RTH 全部記錄 |
| T-019 | Simulation | PCMD response ± axes | Sphinx | 符號、stop time、braking、TTL receipt 完整 |
| T-020 | Simulation | deterministic route worst-case | Sphinx | 全部固定案例安全終止，loss 後零 PCMD |
| T-021 | Field | 真機 PCMD response | 拆槳後低高度 | 建立 speed response receipt，證明 0.30 m/s guard |
| T-022 | Field | ground/low-altitude manual controls | 真機 | 起降、hover、nudge、Esc、Space、E-stop、land 通過 |
| T-023 | Field | 真機 EDM | 人工飛行 | pose/影像/latency/LOST 記錄與人工判讀通過 |
| T-024 | Operations | log retention/disk pressure | temp filesystem | 可刪 log 按策略清理，永久 safety log 不刪，低空間阻止起飛 |
| T-025 | Operations | reboot後手動啟動 | deployment host | 不自啟；SIM/REAL 可透過自動 DISPLAY 探測啟動 |
| T-026 | Release | unified validation receipt | 離線 | 所有 required gate 一個命令得到非歧義 verdict |

任何真機測試都由現場人員實際操作起飛。自動測試、agent 或 LLM 不得觸發 TakeOff、非零 PCMD 或自由飛行。

## 18. Gap analysis

| 優先級 | 現況 | To-Be | 所需工作 |
|---|---|---|---|
| P0 | link loss 只顯示；stream stale 只 hover | 所有感知／worker故障先零 PCMD再交人工；total link 用 B1 | 統一 fail-safe contract、獨立 supervisor、測試 |
| P0 | 連線未完整記錄設備／firmware／RTH | 現場讀取並記錄，未知值阻止 PC control takeoff | device inventory 與 preflight gate |
| P0 | UI 無明確 E-stop | 醒目的 movement E-stop + 獨立 LAND_NOW | UI/backend typed action、鎖存與測試 |
| P0 | autonomous schema／controller 要求 metric scale | 永久無 map metric scale；方向 unit vector + airframe speed guard 0.30 m/s | site schema v2、scale-free controller adapter、舊入口保持鎖定 |
| P0 | route controller 有兩套邏輯 | `parrot_stimulate` 與真機共用權威核心 | 抽出最小共用 controller，雙環境 adapter tests |
| P0 | 無真機 PCMD response receipt | 速度上限不可只靠百分比推測 | props-off、ground、low-altitude response workflow |
| P0 | 無 log retention，磁碟 91% | 保留策略 + 低磁碟 fail closed | retention manager、disk monitor、部署容量處理 |
| P0 | offline smoke 有，runtime OS allowlist 無 | SIM 無網路；REAL 只可 ANAFI 私網 | launcher sandbox/firewall policy 與測試 |
| P1 | SIM 要求明確 video、預設 urai | 河濱 + P119 預設 | launcher default 與 SHA gate |
| P1 | FFmpeg EOF 自動 loop | 最後一幀 `EOF_HOLD` | FrameSource EOF contract、UI state、測試 |
| P1 | link simulation 非預設 | 5 Mb/s、280 ms、0% nominal，1/3/5% presets | launcher preset 與 manifest |
| P1 | `DISPLAY=:1` fallback | 自動辨識有效 desktop | display resolver 與 dry-run tests |
| P1 | SIM/REAL 共用深色外觀 | 藍 SIM、紅橘 REAL 永久識別 | UI theme/status strip |
| P1 | backend 以 duck typing + command string 為主 | typed Python contract + explicit result | Protocol/dataclass/enum，surgical adapter |
| P1 | 多個驗證命令 | 單一 validation entry + receipt | wrapper 與跨 Python test aggregation |
| P1 | current P119 artifact 無 integrity verdict | known-incomplete waiver 與非正常完整 verdict | 重跑 benchmark、寫入 audit fields |
| P2 | `scale-calibrated` 字樣易混淆 | `scene-threshold-calibrated`，清楚非公尺 | profile/docs 術語修正 |
| P2 | source-to-UI timestamp 未含人工 link delay | capture-before-link 到 UI 的可信 p95 | FramePacket timestamp 擴充、B0 receipt |
| P2 | 硬體 monitor 是獨立 CLI | UI/session summary 整合 read-only health | process supervisor 與 health aggregation |
| P2 | mirror source trees 需人工同步 | 單一 package/source owner | 發布後的結構收斂，不與安全功能混改 |
| P2 | working tree 無 release receipt | 可重現 manifest、commit/tag 或 immutable package | 操作員核准後建立 release 流程 |

## 19. 分階段交付計畫

### Phase 0：規格確認

- 操作員審核並確認本文件。
- 確認前不修改程式。

驗收：本文件涵蓋指定範圍，沒有未解的產品決策；現場才能取得的資料列為 receipt，不再當成訪談問題。

### Phase 1：接口契約與 P0 安全基礎

- 建立 shared Python backend contract 與 immutable session config。
- 統一故障後 hover/manual handoff、E-stop、total-link state。
- 加入設備／firmware／RTH inventory 與 preflight fail-closed。
- 保持 real autonomous 按鈕鎖定，不改真機起飛安全路徑。

驗收：unit/integration safety tests 全通過；不需也不得真機起飛。

### Phase 2：模擬接口與 UI

- 河濱／P119 預設、nominal link、1/3/5% presets。
- EOF last-frame hold。
- DISPLAY 自動探測。
- SIM／REAL theme、永久 REAL 警告、route 預設隱藏確認。

驗收：P119 完整 UI run 不循環，2,934 幀 known-incomplete receipt；跨接口隔離測試通過。

### Phase 3：可觀測性、離線與部署

- session manifest、incident、安全 log、retention、disk pressure。
- runtime offline network allowlist。
- 單一 validation entry 與 release receipt。
- 處理目前 91% 磁碟使用率。

驗收：無網路 full smoke、低磁碟測試、雙 Python 驗證及整體 receipt 通過。

### Phase 4：定位回歸與控制器整合

- 固定 P119 B0 與 visual review artifact。
- 保留 `max_corr_total=900`，不得為通過測試而放寬品質 gate。
- 共用 `parrot_stimulate` waypoint controller 核心。
- 建立無 map metric scale的方向控制與 airframe speed guard，但 real auto 仍鎖定。

驗收：第 12 節 KPI、Sphinx response/worst-case、全部 unit/integration tests 通過。

### Phase 5：真機地面與人工低高度測試

- 現場讀取 hardware/firmware。
- 拆槳 axis/yaw/stick/E-stop 測試。
- 人工起飛完成 PCMD response、0.30 m/s guard、braking、stream/worker failure handoff、EDM 定位。

驗收：由人類操作，所有 receipt 完整；任何失敗均保持 `flight.approved=false`。

### Phase 6：自主飛行外部核准

- 完成 route 淨空、兩人確認、安全監控員、checklist 與全部第 10.2 節證據。
- 由操作員核准 profile/SHA 並明確設定 `flight.approved=true`。
- 先進行最低高度、最短 route、0.30 m/s 上限的漸進測試。

驗收：任一 gate 缺失都不得進入 autonomous。

### Phase 7：獨立最終審核與修正

- 將完整系統、diff、spec traceability、測試 receipts 與未解風險交給 Claude Opus 5 Max。
- 審核結果分類為 blocker、safety、quality、performance、maintainability。
- Codex 只執行操作員接受且能追溯至本規格的修正，重新跑所有受影響測試。
- 若 Claude Opus 5 Max 無法使用，先向操作員報告並取得替代審核者授權。

## 20. Definition of Done

本任務的程式實作只有在下列條件全部成立時才算完成：

- 只有兩個操作接口，cross-mode、hot-switch 與 file/live fallback 都被測試拒絕。
- SIM 預設河濱 + P119 + nominal link，EOF 停最後一幀，controls 只改模擬狀態。
- REAL 永久醒目警告，連線時記錄實際 hardware／firmware，無 file fallback。
- planned route 預設隱藏，trajectory／frustum／axes 保留。
- 所有感知／worker故障先停止 PC 動作並交人工；total link 的 B1 policy 有現場證據。
- 高度／距離落地可調、ack/readback，越界不自動 RTH；lost-link RTH 保持獨立。
- 正式 EDM 無 CUDA即 fail closed，完全離線，asset/model/profile SHA 全驗證。
- P119 品質／效能不低於第 12 節基準；人工影像審查通過且不做 ground-truth claim。
- 無 map metric scale；未來 route 使用方向正規化與飛機端 0.30 m/s speed guard。沒有真機 response receipt 前 autonomous 保持鎖定。
- logging、retention、disk pressure、single validation receipt 完成。
- 主 Python 3.10 與 `parrot_stimulate` Python 3.11 維持隔離且一鍵驗證。
- 所有真機起飛均由操作員在 UI 親手執行，agent／腳本沒有旁路。
- 所有 site profile 在外部核准前仍為 `flight.approved=false`。
- Claude Opus 5 Max 最終審核及後續核准修正完成，或已由操作員書面同意替代審核方案。

## 21. 參考

- `README.md`
- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- `控制介面程式/SAFETY.md`
- `控制介面程式/operator_interface/README.md`
- `控制介面程式/site_profiles/README.md`
- `定位演算法/flight_control/README.md`
- `模擬器/parrot_stimulate/README.md`
- `模擬器/parrot_stimulate/INTEGRATION_REVIEW.md`
- [Parrot ANAFI White Paper v1.4](https://www.parrot.com/assets/s3fs-public/2020-07/white-paper_anafi-v1.4-en.pdf)

---

此版本沒有剩餘的產品訪談問題。Aircraft/controller 實際版本、Home/RTH policy、PCMD response、低高度測試結果與自主核准屬現場執行時產生的驗收證據，不應由文件預先猜測。

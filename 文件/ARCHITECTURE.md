# 系統架構與程式碼所有權

## 目標與邊界

本倉庫提供既有場域地圖上的視覺定位、操作介面、任務編修與 ANAFI 飛行安全整合。
本機 workspace 可保存實際場域資產，但點雲、localization bundle、影片、飛行紀錄
與模型權重不提交到 Git，並依下列資料生命週期管理。

系統的部署單位是 **site profile**。同一個 profile 必須原子化地選取：

- PLY 地圖；
- localization bundle；
- query camera 內參；
- localizer backend 與 production profile；
- 航線、電桿及其他任務資產。

地圖、bundle、相機與航線若不在同一座標系，不得進入真機流程。

## 執行資料流

```text
MP4 或 ANAFI Pdraw RGB frame
        │
        ▼
live_localizer_worker
        │
        ├─ BOOT / LOST：MegaLoc 全域檢索
        ├─ TRACK / WEAK：EDM 或 XFeat 局部匹配
        ├─ 2D–3D correspondences
        └─ pycolmap absolute-pose PnP
        │
        ▼
Pose + BOOT_INIT / TRACK / WEAK_TRACK / LOST
        │
        ├─ operator desktop UI
        ├─ validation / replay quality gate
        └─ flight controller safety gates
```

## 目錄所有權

| 目錄 | 所有權 |
|---|---|
| `控制介面程式/` | site profile、任務入口、桌面 UI、worker protocol、串流 launcher |
| `定位演算法/deploy_code/sfm_glomap_deploy/` | 可部署的定位 runtime、localizer adapter、bundle loader |
| `定位演算法/flight_control/` | 航線控制、Olympe 串流與真機安全邏輯 |
| `定位演算法/validation/` | replay、品質 gate、硬體監控、部署一致性檢查 |
| `定位演算法/deploy_code/runtime/EDM/` | 上游 EDM training/runtime 原始碼；視為 third-party boundary |
| `地圖檔/` | 使用者提供的場域資產；預設不納入 Git |
| `outputs/` | 操作 session、正式驗證 receipt 與不可丟失的實驗決策鏈 |
| `文件/` | 架構、系統規格與工作區稽核紀錄 |
| `tools/` | 不改變 runtime 的系統驗證、工作區維護與稽核工具 |

新功能應放入其邏輯所有者，不得為了方便再複製一份到另一個目錄。

## 資料生命週期

| 類型 | 唯一路徑 | 自動清理 |
|---|---|---|
| 場域地圖／bundle／route | `地圖檔/場域/<site>/` | 禁止 |
| 測試影片 | `模擬器/測試影片/` 或 receipt 固定的外部來源 | 禁止 |
| 模型與離線 cache | `定位演算法/deploy_code/runtime/EDM/weights/`、`執行環境/torch_hub_cache/` | 禁止 |
| 編譯 kernel cache | `執行環境/inductor_cache/` | 可刪，冷啟自動重建（約 5 秒） |
| 操作 session | `outputs/flight_logs/session_*/` | 只清可再生效能 log；安全／命令 log 永久 |
| 正式驗證 | `outputs/validation_receipts/` | 禁止 |
| 實驗證據 | `outputs/<實驗>_<YYYYMMDD>/` | 人工審核後封存，不自動刪除 |
| 舊交付內容 | 工作區外的唯讀備份 | 禁止 import／執行，人工管理 |

不為了美觀搬動舊實驗路徑，因為歷史 Markdown／JSON 會引用它們。新資料必須遵守
上述唯一落點；`python tools/workspace_audit.py --strict-output-names` 檢查結構、
symlink、容量與未分類 output。

## 共用 runtime 模組的唯一所有權

部署與飛控入口都可能把兩個 runtime 目錄加入 `sys.path`，因此同名 Python 檔會使
匯入結果依路徑順序而變。每個共用模組只保留一份權威實作：

| owner 目錄 | 權威模組 |
|---|---|
| `deploy_code/sfm_glomap_deploy/` | `artifact_integrity.py`、`localizer_registry.py`、`megaloc_cache.py`、`pose_types.py`、`reference_index.py`、`production_xfeat_tracker.py`、`reloc_localizer_xfeat.py` |
| `flight_control/` | `autoflight.py`、`manual_nudge_pilot.py`、`olympe_frame_source.py`、`path_follow_flight.py`、`plan_path.py`、`real_path_follow_controller.py`、`route_domain.py` |

呼叫端直接從 owner 匯入，另一個目錄不得再放相容副本。CI 由下列命令驗證 owner
存在、舊副本沒有回流，且沒有未分類的同名 runtime 檔案：

```bash
python 定位演算法/validation/check_runtime_mirrors.py
```

兩個目錄都保有各自的 `README.md`，內容描述不同責任，這是唯一允許的同名檔案。

## 飛行權限與安全資料流

```text
OperatorApp
  │
  ├─ OperatorCommandCoordinator：命令序列化、優先佇列與 shutdown suspend
  └─ typed backend command
          │
          ├─ AuthorityController：MANUAL / PC_CONTROL 單一 owner
          ├─ TakeoffLandingSupervisor：起飛後失敗的降落 fallback
          └─ TelemetryFreshnessStore：逐欄位 freshness 與速度限制語意
```

真機 AUTO 另有獨立的 release latch。程式啟動與 profile 同步都會把 latch 設為
關閉；即使一般起飛按鈕可用，也不能建立 live AUTO coordinator。未來解除阻擋前，
必須同時通過四步 preflight、site/route 核准、資產雜湊、runtime lock，以及簽章的
hardware approval receipt v2。GPS 不可用不會阻擋起飛，但只會停用依賴 GPS 的距離
柵欄，不能繞過其他安全 gate。目前隨附的所有 site profile 都不具備 live AUTO
核准。

## 路線與定位替換邊界

`flight_control/route_domain.py` 是 route JSON 的唯一解析、驗證與 controller
conversion 邊界。profile 的 `site_id`、`coordinate_frame_id` 與 route 必須完全
一致。localizer 由 `localizer_registry.py` 依 profile 建立，EDM 與 XFeat 都實作相同
pose provider contract；呼叫端不得直接綁定某個 tracker。

大量 reference 透過 `reference_index.py` 的磁碟式、可驗證 IVF index 讀取，descriptor
與 posting array 使用 memory map，避免把 100k reference 全部載入 RAM。index
metadata、陣列與模型 identity 都由 `SHA256SUMS.json` 固定。

## Portable artifact 與信任邊界

- `MANIFEST.tsv` / `SHA256SUMS`：Git source release 的完整性。
- `RUNTIME_ARTIFACTS.json`：runtime 大型資產的來源、目標與 digest，不允許網路下載。
- `PORTABLE_SITE_ASSETS.json`：匯出時選定 site profile 的離線資產清單。
- hardware receipt、detached signature 與 trust store：分別固定內容、簽章及信任根，
  三者都必須由 profile 以 SHA-256 綁定。

portable exporter 只從 source repository 或明確指定的 offline artifact root 取得資產，
拒絕 traversal、symlink 逸出與 digest 不符。portable 套件因此可在無 Git、無原始場域
目錄的另一台電腦驗證與安裝；它不會因為攜帶 pending request 而取得 AUTO 權限。

## 支援的入口

### 離線影片

```bash
SFM_SITE_PROFILE=/absolute/path/to/site.json \
  ./控制介面程式/影片模擬串流/啟動.sh /absolute/path/to/video.mp4
```

### 任務編修與驗證

```bash
python 控制介面程式/mission_pipeline.py \
  --site-profile /absolute/path/to/site.json \
  --mode draw-path

python 控制介面程式/mission_pipeline.py \
  --site-profile /absolute/path/to/site.json \
  --mode dry-run
```

正式任務模式預設要求 `--site-profile`。`--allow-legacy-assets` 僅供已審查的遷移作業，不是正式部署介面。`flight-selftest` 與 `safety-*` 命令刻意保持 profile-free，確保缺失資產時仍能執行純邏輯檢查或安全動作。

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

## 目前保留的相容鏡像

在完成 packaging 前，少數檔案仍需同時出現在部署包與飛控目錄。其邏輯 owner 如下：

| 邏輯 owner | 相容 mirror |
|---|---|
| `deploy_code/.../artifact_integrity.py` | `flight_control/artifact_integrity.py` |
| `flight_control/autoflight.py` | `deploy_code/.../autoflight.py` |
| `deploy_code/.../megaloc_cache.py` | `flight_control/megaloc_cache.py` |
| `flight_control/path_follow_flight.py` | `deploy_code/.../path_follow_flight.py` |
| `flight_control/plan_path.py` | `deploy_code/.../plan_path.py` |
| `deploy_code/.../pose_types.py` | `flight_control/pose_types.py` |
| `deploy_code/.../production_xfeat_tracker.py` | `flight_control/production_xfeat_tracker.py` |
| `flight_control/real_path_follow_controller.py` | `deploy_code/.../real_path_follow_controller.py` |

修改時先改 owner，再同步 mirror；CI 由下列命令阻止 drift：

```bash
python 定位演算法/validation/check_runtime_mirrors.py
```

名稱相同但未列入表中的檔案是獨立實作，不可假設可以互相覆蓋；目前包括 `olympe_frame_source.py` 與 `reloc_localizer_xfeat.py`。

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

### 操作介面

真機使用 `控制介面程式/operator_interface/flight_operator_app.py`。`demo_stub_http_server.py` 只提供模擬 HTTP API，不連接 Olympe，也不得作為真機入口。

## 安全邊界

- 起飛只能由現場操作員在桌面 UI 親自執行。
- 自動化不得呼叫 TakeOff 或替操作員寫入起飛授權。
- `fly` 前必須完成 self-test、dry-run、模擬、拆槳測試與現場安全審查。
- WEAK、LOST、stale pose、stale stream 或 watchdog timeout 必須導向零 PCMD、懸停、人工接管或降落，不得繼續沿用舊命令。
- 場域資產不完整、路徑不存在或 profile 與 per-asset override 混用時，入口必須 fail closed。

## 後續收斂順序

1. 以 mirror check 維持現有部署包一致性。
2. 將部署包改成由單一 source tree 建置，不再提交鏡像 Python 檔。
3. 把 EDM 上游程式移到明確的 `third_party/EDM` 邊界，保留本專案 adapter 與 tracker glue。
4. 統一成可安裝的 Python package 與單一 CLI。
5. 移除 `--allow-legacy-assets` 及所有舊場域 fallback。

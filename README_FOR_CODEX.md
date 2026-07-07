# Localization Runtime Tuning Transfer Package

這個資料夾是要搬到另一台電腦的定位調參包。用途是調整定位 runtime 組合，例如把
LightGlue 改成 NN、`nn_then_lg`，或調整 topK、threshold、PnP gate、temporal
cache。這包不是建圖包，也不是更新地圖包。

## 這包適合做什麼

- 使用既有 XFeat reloc bundle 做定位。
- 調整 matcher：`nn`、`lighterglue`、`nn_then_lg`。
- 調整 `nn_min_score`、`local_topk`、`weak_local_topk`、`track_xfeat_topk`。
- 調整 adaptive LightGlue 3->5 fallback。
- 調整 PnP / RANSAC / reprojection gate。
- 調整 temporal cache，降低進入 LighterGlue fallback 的比例。
- 開啟桌面飛行介面，左側看點雲和定位軌跡，右側看 720p 串流。

## 這包不適合做什麼

- 不用來重建 GLOMAP 地圖。
- 不用來重建 XFeat 2D-3D anchors。
- 不用來重新抽 MV-RoMa / UFM / GLOMAP 特徵。
- 不包含測試影片。
- 不包含標註系統和 YOLO 訓練資料。
- 不包含完整 `/定位/source/` 舊實驗資料。

如果要重建 bundle 或重新選 reference keyframes，還需要另外搬 base reference
images、完整 source/update workspace，以及重建腳本。

## 主要檔案

```text
requirements_runtime.txt        # 原機 runtime 直譯器的釘選版本
torch_hub_cache/                # 離線模型權重(XFeat/LighterGlue/MegaLoc)
sfm_system/
  定位/
    bundles/
      current_reloc_map_updated_v3.pt
    maps/
      current_realrgb_v3.ply
      base_glomap_fused_0/
    deploy_code/
      sfm_glomap_deploy/
        production_xfeat_tracker.py
        reloc_localizer_xfeat.py
        autoflight.py
        path_follow_flight.py
        real_path_follow_controller.py
        olympe_frame_source.py
        plan_path.py
        map_intrinsics.json
    mission/
      operator_interface/
      flight_control/
      configs/
    pipeline/
      localize_pipeline.py      # 只有 production-stream 模式可用(見下)
    validation/
      benchmark_production_stream.py
      eval_stream_core.py
    configs/
      localization_defaults.json   # 純描述 metadata,runtime 不讀
    DEPLOY_TO_OTHER_PC.md
```

注意:`localize_pipeline.py` 的 `compare` 與 `sim` 模式依賴沒搬的
base bundle / simulation 目錄,在這包裡只有 `--mode production-stream` 可用。
`mission/authoring/` 的 Blender 標註工具依賴沒搬的 source 資料,在新電腦
不可用(僅供參考)。

## 核心資料

固定 reloc bundle：

```text
sfm_system/定位/bundles/current_reloc_map_updated_v3.pt
```

這個 bundle 已包含 XFeat keypoints/descriptors、2D-3D anchors、ref global
descriptors。只換 NN / LightGlue matching 不需要重新建 bundle。

介面顯示用彩色點雲：

```text
sfm_system/定位/maps/current_realrgb_v3.ply
```

benchmark 讀內參用 sparse model：

```text
sfm_system/定位/maps/base_glomap_fused_0/
```

## 建議先做的自檢

到新電腦解壓後，先進入搬移包根目錄：

```bash
cd /path/to/sfm_localization_runtime_tuning_package
```

確認沒有斷掉的 symlink：

```bash
find sfm_system -xtype l
```

正常情況應該沒有輸出。

確認大檔案存在：

```bash
ls -lh sfm_system/定位/bundles/current_reloc_map_updated_v3.pt
ls -lh sfm_system/定位/maps/current_realrgb_v3.ply
ls -lh sfm_system/定位/maps/base_glomap_fused_0/images.bin
```

確認 Python import（以下所有指令都用「裝了 torch+pycolmap+cv2 的 python」跑）：

```bash
PYTHONPATH=sfm_system/定位/deploy_code/sfm_glomap_deploy \
python3 - <<'PY'
from production_xfeat_tracker import ProductionConfig
from reloc_localizer_xfeat import XFeatRelocMap
print("imports ok", ProductionConfig().matcher_mode)
print("bundle load test...")
x = XFeatRelocMap.load("sfm_system/定位/bundles/current_reloc_map_updated_v3.pt")
print("refs", len(x.ref_names), "vpr", x.meta.get("bundle_vpr") or x.meta.get("vpr"))
PY
```

預期輸出 `refs 2095 vpr megaloc`。

確認 pycolmap 版本吃得下 rig 格式 sparse model（這步過不了先解版本問題再調參）：

```bash
python3 -c "import pycolmap; r=pycolmap.Reconstruction('sfm_system/定位/maps/base_glomap_fused_0'); print('images', r.num_images(), 'points', r.num_points3D())"
```

原機用 pycolmap 4.0.4（見 `requirements_runtime.txt`）。這張圖含
`frames.bin`/`rigs.bin`（rig/frame 模型），太舊或差太多的 pycolmap 可能
載不進來或 `estimate_and_refine_absolute_pose` API 不相容。

確認操作介面可以讀點雲：

```bash
python3 sfm_system/定位/mission/operator_interface/flight_operator_app.py \
  --selftest \
  --no-live-localize \
  --no-live-detect \
  --video ""
```

## 雙 Python 架構（重要）

操作介面（UI）與定位 worker 是**兩個 process、可以是兩個不同的 python**：

- UI 本體（tkinter + PIL + numpy）：跑 `flight_operator_app.py` 的直譯器。
- 定位 worker（`live_localizer_worker.py`）：由 UI 用 `--localizer-python`
  指定的直譯器啟動，**必須裝 torch + torchvision + pycolmap + opencv + numpy**。

`--localizer-python` 的預設解析順序：

1. 環境變數 `SFM_LOCALIZER_PYTHON`
2. `/usr/bin/python3.12`（原機的 runtime 直譯器,存在才用）
3. 目前跑 UI 的同一個 python

新電腦最簡單的做法：直接用「裝齊依賴的 python」跑 UI，worker 就會用同一個。
或設 `export SFM_LOCALIZER_PYTHON=/path/to/env/bin/python`。物件偵測 worker
同理用 `SFM_DETECTOR_PYTHON` / `--detector-python`。
`localize_pipeline.py --python` 也遵守 `SFM_LOCALIZER_PYTHON`。

worker 啟動失敗時,stderr 在 `/tmp/sfm_live_localizer_worker.log`。

## 桌面操作介面

若只要開 UI，不連真機、不跑物件偵測：

```bash
python3 sfm_system/定位/mission/operator_interface/flight_operator_app.py \
  --no-live-detect \
  --video /path/to/input.MP4
```

若要暫時不跑定位，只看介面：

```bash
python3 sfm_system/定位/mission/operator_interface/flight_operator_app.py \
  --no-live-localize \
  --no-live-detect \
  --video /path/to/input.MP4
```

目前串流固定以 1280x720 進入定位 worker。若使用影片模擬，程式會用 ffmpeg
把影片縮到 720p stream。

## Production Tracker 狀態機

目前預設決策：

```text
BOOT_INIT / LOST:
  MegaLoc retrieval
  -> XFeat + LighterGlue
  -> PnP

TRACK / WEAK_TRACK:
  XFeat + mutual-NN fast pass
  -> 若不夠穩，再 XFeat + LighterGlue adaptive 3->5
  -> PnP
```

主要程式：

```text
sfm_system/定位/deploy_code/sfm_glomap_deploy/production_xfeat_tracker.py
```

重要參數（`ProductionConfig`，名字以程式碼為準）：

```text
matcher_mode = nn_then_lg | lighterglue | nn   # TRACK/WEAK_TRACK 用
acquire_matcher_mode = lighterglue             # BOOT_INIT/LOST 專用;"" = 跟隨 matcher_mode
nn_min_score = 0.85
adaptive_accept_inliers = 100
adaptive_accept_reproj = 3.5
adaptive_first_topk = 3
local_topk = 5
weak_local_topk = 8
xfeat_topk_track = 1300
xfeat_topk_acquire = 2048
pnp_ransac_max_error = 5.0
max_reproj_error_track = 6.0
max_reproj_error_acquire = 5.0
boot_global_topk = 30
lost_global_topk = 30
weak_global_topk = 0
temporal_cache_enabled = True                  # nn 與 nn_then_lg 的 TRACK 都會用
```

行為注意：

- `matcher_mode` 只影響 TRACK/WEAK_TRACK；BOOT_INIT/LOST 的 acquisition 由
  `acquire_matcher_mode` 控制（預設 lighterglue）。所以把 `matcher_mode`
  改成 `nn` 不會弱化初始鎖定。若真要連 acquisition 也用 NN，設
  `acquire_matcher_mode=""` —— 但注意純 NN acquisition 可能一直過不了
  acquire gate（卡在 BOOT_INIT）。
- temporal cache 在 `nn` 與 `nn_then_lg` 的 TRACK 狀態都會啟用；
  `lighterglue` 模式用不到（LighterGlue 路徑不吃 cache anchors）。
- benchmark 輸出的 `temporal_cache_used_frames` / `temporal_cache_attempted_frames`
  才是 cache 實際貢獻的指標；舊欄位 `temporal_cache_accept` 恆為 0（歷史遺留）。
- 調參一律走 benchmark CLI flags 或改 `ProductionConfig` 預設值；
  `configs/localization_defaults.json` 是描述性 metadata，runtime 不讀。

如果目標是 RTX 5060 上更快，優先調整方向是提高 NN fast pass 成功率，降低
LighterGlue fallback 比例。不要優先砍 PnP，也不要讓 TRACK 狀態跑 MegaLoc。

## 離線 Benchmark

`benchmark_production_stream.py` 已內建 sys.path bootstrap，直接執行即可,
不需要設 PYTHONPATH。這包沒有附測試影片。若新電腦上有自己的 frame
directory，可跑：

```bash
python3 sfm_system/定位/validation/benchmark_production_stream.py \
  --query-dir /path/to/query_frames_jpg \
  --bundle sfm_system/定位/bundles/current_reloc_map_updated_v3.pt \
  --model sfm_system/定位/maps/base_glomap_fused_0 \
  --matcher-mode nn_then_lg \
  --resize-width 1280 \
  --out sfm_system/定位/outputs/benchmark_result.json
```

改成純 NN：

```bash
python3 sfm_system/定位/validation/benchmark_production_stream.py \
  --query-dir /path/to/query_frames_jpg \
  --bundle sfm_system/定位/bundles/current_reloc_map_updated_v3.pt \
  --model sfm_system/定位/maps/base_glomap_fused_0 \
  --matcher-mode nn \
  --nn-min-score 0.85 \
  --resize-width 1280
```

改成只用 LighterGlue：

```bash
python3 sfm_system/定位/validation/benchmark_production_stream.py \
  --query-dir /path/to/query_frames_jpg \
  --bundle sfm_system/定位/bundles/current_reloc_map_updated_v3.pt \
  --model sfm_system/定位/maps/base_glomap_fused_0 \
  --matcher-mode lighterglue \
  --resize-width 1280
```

輸出會包含每幀 latency、FPS、state、inliers、reprojection RMS，並可輸出相機位置
PLY。

## 依賴環境

原機 runtime 直譯器（`/usr/bin/python3.12`）的釘選版本在
`requirements_runtime.txt`：

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
numpy 2.4.6
opencv-python 4.13.0
pillow 10.2.0
pycolmap 4.0.4
```

另需系統層：ffmpeg、tkinter（UI 用）。XFeat / LighterGlue / MegaLoc 不是
pip 套件，權重和 repo 都走 `torch.hub`（見下節）。

## 離線模型權重（torch_hub_cache/）

XFeat、LighterGlue、MegaLoc 首次執行會透過 `torch.hub` 下載。這包已附上
原機的 hub cache（約 230MB），新電腦不能連網也能跑：

```bash
mkdir -p ~/.cache/torch
cp -r torch_hub_cache ~/.cache/torch/hub
# 或不想動 home 目錄：export TORCH_HOME=/path/to/this/package/torch_hub_cache_home
```

內容物：

```text
torch_hub_cache/
  checkpoints/xfeat.pt                          # XFeat 權重
  checkpoints/resnet50-0676ba61.pth             # MegaLoc backbone
  checkpoints/resnet50_16384.pth                # MegaLoc/BoQ 權重
  verlab_accelerated_features_main/             # XFeat repo（含 weights/xfeat-lighterglue.pt）
  gmberton_MegaLoc_main/                        # MegaLoc repo
  amaralibey_bag-of-queries_main/               # MegaLoc 依賴的 BoQ repo
  trusted_list
```

注意 `TORCH_HOME` 指的是 hub 的**上一層**（`$TORCH_HOME/hub/...`）；用
`cp -r` 到 `~/.cache/torch/hub` 最不容易搞錯。

## 真機飛行注意

這包含自動飛行和操作介面程式，但真機飛行還需要新電腦安裝 Parrot Olympe SDK。
生產飛行入口是 `path_follow_flight.py --fly`（其餘 arming 入口都被
`SFM_ALLOW_LEGACY_FLIGHT` gate 擋住）。

內建安全機制（都在 `path_follow_flight.py`，可用環境變數調）：

- 串流斷線 -> 0.5s 內 hover；斷超過 `SFM_STREAM_LOST_LAND_S`（預設 15s）自動降落。
- 定位 LOST -> hover；超過 4s（LOST_LAND_S）自動降落。
- 單幀 pose 跳變 > `SFM_MAX_POSE_JUMP_U`（預設 1.5 map-units）直接拒斥並 hover；
  連續兩幀一致才接受（允許真正的重定位恢復）。
- 偏離航線 > `SFM_MAX_ROUTE_DEVIATION_U`（預設 3.0 map-units）中止並降落，
  不會盲目 REJOIN 長距離。
- PCMD watchdog（`SFM_PCMD_WATCHDOG_S`，預設 0.7s）：主迴圈卡死（GPU/模型 hang）
  時由獨立 thread 強制送零 PCMD hover。
- 安全開關指令：`a`uto / `h`over / `m`anual / `l`and / `e`mergency（**切斷馬達，
  機體直接掉落**，只在旋槳威脅人身時用），寫入 `SFM_SAFETY_FILE`
  （預設 /tmp/sfm_drone_safety.cmd）或終端輸入。
- 起飛後先 hover BOOT_INIT lock（25s 內鎖不到就降落），成功才進 AUTO。
- PCMD 上限遠低於 ANAFI 極限（pitch<=8%、yaw<=25%、gaz<=12%、roll=0）。

**仍未涵蓋、飛行前必須人工確認**：

- 沒有接 safe-volume / 點雲避障（`safe_volume.py` 存在但未接入生產路徑）；
  航線本身必須畫在淨空走廊內。
- 地圖是 GLOMAP 無尺度單位；所有距離門檻（1.5u、3.0u…）的實際公尺數
  取決於地圖尺度，換地圖要重新確認。介面高度顯示為 `u(map)` 不是公尺。
- 起飛時機頭必須朝航線起始方向（heading 種子假設 nose-on-route）；
  `--yaw-sign` 要先在 PROPS-OFF 板凳測試驗證。
- 桌面介面的飛行按鈕是模擬 backend，**不會**控制真機；真機介入只有
  safety file / 終端指令。
- 沒有 RTH（返航）。緊急停止 = Emergency 掉落。

正式飛行前至少跑一次：`--selftest`、`--dry-run`、Sphinx 模擬、PROPS-OFF
`--grab-only` 街機驗證,再進開闊場地 + 人工監控。

## 不要搬進這包的東西

以下資料刻意沒有放進來：

- `sfm_system/建圖/`
- `sfm_system/更新地圖/`
- `sfm_system/定位/validation/source_videos/`
- `sfm_system/定位/source/`
- `/media/cihcilab/新增磁碟區/label_system/`
- YOLO 訓練資料

若要在新電腦上做物件偵測 runtime，需要另外搬：

```text
sfm_system/定位/object_detection/configs/
sfm_system/定位/object_detection/models/
```

若 GPU/driver 不同，TensorRT engine 要從 `.pt` 或 `.onnx` 重新轉。

## 給下一位 Claude / Codex 的工作邊界

預設只調 runtime matching，不要動地圖建置資料。

可改：

- `production_xfeat_tracker.py`
- `path_follow_flight.py`
- `live_localizer_worker.py`
- `flight_operator_app.py`（含 UI 美觀與功能調整——版面、配色、
  顯示資訊、操作流程都可以改；唯安全相關行為（hover/LOST 處理、
  worker 失敗處理）改動前要先跟使用者確認）
- `benchmark_production_stream.py`
- `localize_pipeline.py`

不要改：

- `current_reloc_map_updated_v3.pt`，除非使用者明確要求重建 bundle。
- `current_realrgb_v3.ply`，除非使用者明確要求替換顯示地圖。
- `base_glomap_fused_0/`，它只作為 sparse model / 內參參考。

調參後要記錄：

- matcher mode
- NN fast accept ratio
- LighterGlue fallback ratio
- success rate
- median / p90 latency
- median inliers
- median reprojection RMS
- LOST/WEAK_TRACK frame 數
- 相機位置 PLY 是否合理

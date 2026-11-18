# 工作區工具

## 離線 MVS 與公尺深度查詢

先匯入完整 `direct` 場域，再用原始建圖影像及同一份 COLMAP 模型離線重建。
CUDA 建圖環境必須獨立，不能將 `pycolmap-cuda12` 安裝到定位環境覆蓋 CPU wheel：

```bash
uv venv --python .venv/bin/python .venv-mvs
uv pip install --python .venv-mvs/bin/python --require-hashes -r requirements/mvs-lock.txt
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --require-hashes -r requirements/optional-depth-lock.txt
.venv-mvs/bin/python tools/build_mvs_surface.py \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --output-dir /path/with/enough/free/space/mvs --max-image-size 640
```

預設處理所有已註冊影像，包含 SHA-256、原始解析度與座標系檢查，再執行
去畸變、CUDA PatchMatch、geometric fusion、Open3D CPU Poisson 與量測支援篩選。
中間深度／法向量及 mesh 需要足夠儲存空間，不要將整場 workspace 放入快滿的
系統磁碟。`--image-name` 是明確標示 partial 的除錯選圖，不代表完成全場。
融合會預載全部 geometric 深度／法向量／影像到 RAM，以雙執行緒執行；
避免小型 LRU 快取反覆載入。河濱 640 px 的輸入預估約 4–5 GiB，
另須預留模型、融合點與 mesh 記憶體；提高解析度前先確認 RAM 餘裕。
此模式的 `cache_size` 不限制融合總記憶體，機制見
[COLMAP fusion](https://github.com/colmap/colmap/blob/4.0.4/src/colmap/mvs/fusion.cc)。
CUDA binding 依 [COLMAP 4.0.4](https://github.com/colmap/colmap/tree/4.0.4)；
它包含 4.0.3 的 Blackwell PatchMatch 修正。
Poisson 使用 depth 12，讓網格解析度能保留密集 MVS 取樣；過粗網格會被局部
支援篩選大量移除。仍不保證填滿畫面，無觀測支持的部分維持未知。

成功後原子化發布 `direct_bundle.json` 旁的 `dense_surface.json` 與
`dense_surface/<mesh-sha256>.ply`，不修改定位模型、site profile、航線或飛行授權。
mesh 維持原始 map 座標，沒有套用 UI 重力對齊或公尺縮放；manifest 綁定 bundle、
mesh SHA-256 與 `coordinate_frame_id`。失敗不替換既有 manifest。
MVS sidecar 是匯入後產物，既有場域匯入器不會自動複製它；搬機時須一併攜帶
sidecar 與其相對路徑指向的 mesh。

若已有獨立確認的公尺比例，可在建圖時同時提供
`--meters-per-map-unit <m/u>` 與 `--scale-source "<校正來源>"`；否則保留 null。
不得使用 controller 設定、候選尺度或 `map_scale: 1` 代替有效校正。
操作介面影片右鍵的「MVS 絕對深度」使用同場次、配對影格時可信的 `SessionMapScale`；
沒有可信 session 尺度或有來源的 manifest 校正時，拒絕輸出公尺值。

純離線 CLI 可使用 hash-bound 參考姿態，或帶 `coordinate_frame_id` 的線上定位
JSON（`--pose-json`，須成功、完整相機姿態且不是保留前一幀位置）：

```bash
.venv/bin/python tools/query_surface_depth.py \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --reference-image <reference-image-name.jpg> \
  --pixel 640 360 --output /tmp/camera-depth.npy
```

未在 surface manifest 保存比例時，CLI 同樣需要明確的
`--meters-per-map-unit <m/u> --scale-source "<校正來源>"`。
輸出為 float32 公尺 **camera-Z 深度**，不是視線斜距；
`.npy` 無表面命中處是 NaN，JSON 查詢是 null。
`--width` 可指定較小深度圖，內參按原始相機像素中心映射。
線上 UI 的預覽寬度為 640 像素，CLI 預設原始相機解析度。

Poisson 可能跨越空洞，因此發布前刪除缺少附近融合量測支援或過大的三角面，
不額外補洞。這仍是採樣啟發式，不能證明細小孔洞、玻璃、水面、動態物體或
未拍攝表面的深度。無命中是「未知」，不是「沒有障礙物」；
此功能不接入 PCMD、AUTO、起降或碰撞保護。

河濱全場已使用 1058 張註冊影像執行，1045 張產生 geometric 深度，
融合 4,246,852 點，發布 3,168,950 個受量測支持的三角面。
正式錄影與 direct 定位器的配對姿態已實測；其中一張 640×360 查詢影格
命中率約 88%，不是全場覆蓋率或深度精度。
紀錄見 `outputs/mvs/river_site/receipt.json` 與
`outputs/analysis/mvs_river_surface_smoke.json`。
這次未取得可持久化的公尺校正，manifest 比例維持 null；
實際公尺讀值仍須可信 session 尺度。另以已知 5 m 平面驗證介面讀值與深度圖。

## AUTO 軌跡回顧

新場次自動將完整規劃航點、控制週期位置／指令／偏差、定位來源與 AUTO 起訖
寫入 `outputs/flight_logs/session_*/trajectory.jsonl`。也保留 BOOT／等待期間的
精簡定位結果及失敗紀錄。這個檔案不在診斷日誌的自動輪替刪除清單內。
重新開啟操作介面後，新場次才會使用此記錄格式。

降落後可產生無網路依賴的回顧頁（只讀取日誌，不連接飛機）：

```bash
.venv/bin/python tools/flight_trajectory_review.py outputs/flight_logs/session_場次名稱 \
  --out outputs/analysis/trajectory_review
```

以瀏覽器開啟輸出的 `index.html`，選擇 AUTO 場次、俯視／兩個側視角，拖曳時間軸
或播放。`trajectory.csv` 提供時間、原始 XYZ、定位來源、當時目標航段偏差、
地速與電量；`trajectory.json` 保留路線快照及其座標系／SHA-256。
舊場次尚有 `localization.jsonl`／`telemetry.jsonl` 時也能回顧。

所有軌跡均為定位／IMU 估計，非獨立真值。距離用地圖單位，不假設公尺比例。
偏差以當時目標航段計算，包含首次靠攏段；BOOT 無目標時不計偏差。
缺失／過期位置與超過 0.5 秒的紀錄空檔不連線，不把人工接管間隔接成 AUTO。

### 每趟 AUTO 規劃折線誤差與紀錄健康

```bash
.venv/bin/python tools/autoflight_error_report.py outputs/flight_logs/session_場次名稱 \
  --out /tmp/auto-route-errors.json
```

此報告讀取永久終結摘要；JSON 的 `auto_runs` 分趟保留平均／RMS／P95／最大與
水平偏差、樣本數、結果及路線 SHA，終端顯示最後一趟。
定義是首次到達航點後，獨立且新鮮的觀測到最近 3D 規劃航段
的距離；與上方回顧頁「當時目標航段、包含靠攏」不同。沒有合格樣本會保留
該趟並顯示無法計算。缺少觀測時間證據的舊紀錄標為 `legacy_unverified`，
不可當成已確認的新鮮獨立樣本。分析前檢查場次 `session_summary.json`：
`recording_complete`、`write_failures`、`debug_deferred_dropped`、`unresolved`。
SDK dispatch 表示命令呼叫／enqueue，不代表韌體執行確認。

可用純離線生產者／讀回 smoke 驗證數值與故障標記，輸出限於專用暫存目錄：

```bash
.venv/bin/python tools/verify_flight_diagnostics.py --output-root /tmp/flight-diagnostic-smoke
```

它使用正式 coordinator／controller／記錄器與記憶體 transport，保留檔案讀回、
輪替後不變，並注入寫入、最後 fsync、handle close、佇列溢位及 summary 寫入失敗；
不連飛機、不呼叫真機 SDK、不執行起飛或降落。

背景診斷的回傳契約與讀回方式：

- `localization()`／`telemetry()`／`defer_telemetry()` 回傳 `True` 只代表
  有界佇列接納資料，不代表已寫入、fsync 或錄製健康；`False` 代表未接納。
  overflow 與非同步寫入故障須連同 incident、loss marker 與 summary 檢查。trajectory mirror
  隨已接納的診斷資料處理，不因背景寫入而省略。
- commands／incidents 仍同步 write／flush／fsync；其成功 bool 代表該安全
  sink 的耐久性，不與背景診斷 admission 混用，也不等待診斷 sink 的 I/O。
- 離線工具在 session 尚未關閉、要讀 JSONL 前，明確呼叫
  `wait_for_diagnostics(timeout=...)`，並處理逾時。它等待呼叫前已接納資料與
  loss marker 完成處理，**不是 fsync、也不把寫入失敗改判成功**；
  不得在控制迴圈／UI poll 呼叫。沒有舊 `flush` 相容 shim。
- `close()` 停止 admission，排空已接納資料並 join writer，再做 sink 的最後
  同步及 handle close，最後寫 summary。寫入、fsync、close 或 summary
  失敗均須如實反映；summary 可重試，但不能用重試清掉既有故障。
  讀回須一併檢查 `recording_complete`、`write_failures`、
  `debug_deferred_dropped` 與 `unresolved`，不可只看 barrier 回傳值。

本輪已觀察到上述 CLI 的離線 producer／trace／failure／overflow／summary retry
smoke 通過；僅證明該離線路徑，不代表完整 gates、目前 Tk UI 或實飛核准。

### 離線介面延遲量測

```bash
.venv/bin/python tools/benchmark_operator_latency.py --iterations 500
.venv/bin/python tools/benchmark_operator_latency.py --ui-seconds 20 \
  --site-profile /path/to/site_profile.json --video /path/to/offline.mp4 \
  --window-size 1440x900 --capture /tmp/operator-latency.png
```

GUI 模式固定 `simulated-stream`，量測實際啟動階段、Tk heartbeat 與各 render stage
的時間 count／p50／p95；`--window-size` 指定量測視窗，`--capture` 保存畫面。
不快取／略過場域雜湊檢查、不連真機、不執行飛行。預設不啟動定位，若要量測
模型與結果就緒時間，另加 `--localize` 並使用與場域相符的錄影；
合成影片不能證明定位成功，也不代表 GPU／真機鏈路已就緒。
2026-10-03 的 1440×900、20 秒實際 SIMULATED-STREAM smoke，在 adaptive HUD
修正前量得 render p50 3.34 ms、p95 5.99 ms。沒有同條件的 before baseline
（桌面認證失敗），因此不能宣稱受控前後加速，也不能以此保證最小視窗的控制列皆可容納。
native video presenter 重用 fitted source／PhotoImage 與 canvas overlays；
HUD-only 更新不 resize／upload 影片。這些 render 數字不是網路、GPU 或定位推論速度。
微量測現在以真實 FFmpeg 影格的型態（唯讀 NumPy RGB 陣列，底層為不可變 `bytes`）量測，
並在位元組不同時直接失敗。舊的 `memoryview(bytes)` 輸入不能代表正式路徑：正式影格的
`memoryview.obj` 是 ndarray，舊判斷永遠落到每次 2.76 MB 的複製（`immutable_frame_reused=false`）。
修正後 `LiveWorkerClient._immutable_frame_bytes` 沿 ndarray base 找回原不可變 `bytes`
（唯讀、連續、長度相等才重用；可變來源或局部 view 仍複製）。同程序 3 次 1000 次量測，
複製版本中位數 0.737–0.741 ms、目前提交 0.047–0.061 ms；第一次修正前量測的兩者數值異常
接近（約 0.074 ms），無法作為乾淨配對，因此以 `immutable_frame_reused` 轉為 true 作為機制證據。
這只涵蓋 CPU 提交，並非上述 GUI 的配對 baseline，也不含 IPC、模型或 GPU。

實際定位＋預訓練分割／mask 顯示的 GUI smoke 可加 `--localize --segment`：

```bash
.venv/bin/python tools/benchmark_operator_latency.py \
  --ui-seconds 50 --localize --segment \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --video 模擬器/測試影片/P1980198.MP4 --window-size 1440x900 \
  --capture /tmp/operator-segmentation.png --report /tmp/operator-segmentation.json
```

`--segment-model`／`--segment-python` 可指定已備妥的本機權重與獨立 YOLO 環境；
不下載、不啟動真機。收據 `window.paired_pipeline` 記錄兩組成功的同來源幀
結果都實際呈現後的 FPS／E2E 延遲與 mask 呈現數。此 GUI 交付率受來源及
地圖／影片顯示節拍限制，不是串流 FPS 或 headless 運算吞吐上限。
2026-10-05 的 H264 5 Mb/s、280 ms 名目鏈路模擬、50 秒實際介面驗證完成
1160 次分割呈現及 595 組同幀雙路呈現，收據／截圖見
`outputs/analysis/localization_yolo26n_seg_ui_20261005/`。


無影片／模型時可用受控 20 Hz 完成事件量測真實 Tk 結果消費與 HUD 繪製：

```bash
.venv/bin/python tools/benchmark_operator_latency.py --controlled-results \
  --ui-seconds 5 --window-size 1440x900 --report /tmp/operator-latency.json \
  --capture /tmp/operator-latency.png --render-profile /tmp/operator-latency.prof
```

此模式只建立模擬後端，不連相機或真機，不能搭配 `--localize`。報告保留
`ui_poll_delay_ms`、heartbeat 與 render 的 count／p50／p95／max，以及結果順序與
callback 異常；合成完成事件只驗證介面路徑，不證明實際定位或真機鏈路。

### 定位與 YOLO 偵測／分割並行量測

```bash
.venv/bin/python tools/benchmark_localization_yolo.py \
  --site-profile 地圖檔/場域/river_site/site_profile.json \
  --video 模擬器/測試影片/P1980198.MP4 \
  --mode throughput --warmup-frames 300 --frames 900 \
  --output outputs/analysis/localization_yolo26n_seg_e2e_20261005
```

預設使用官方 COCO 預訓練 `執行環境/models/yolo26n-seg.pt`，不訓練／微調、不把
mask 送進定位或飛控。定位與偵測／分割皆使用正式 worker，沒有另一份量測專用
YOLO 推論實作。先跑定位單獨量測，再跑定位＋獨立 `object_detector_worker.py`；
兩者共用 GPU。預設 `throughput` **不按影片 FPS 限速、不丟幀**：每次從解碼
開始，包含 resize／RGB 轉換、IPC、定位、分割推論與後處理、mask 輪廓 CPU
轉換及 JSON 傳輸；等待同一幀的兩組結果都在 consumer 解析完成，才送下一幀。
定位單跑只等定位結果。FPS = 完成整條運算鏈的影格數 ÷ 實際牆鐘時間，
不是串流 FPS，也不是兩個模型 FPS 相加。此模式同時只有一幀在途，
是該排程的實測吞吐，不宣稱多幀 pipeline 的理論最大吞吐。
預設分割使用 Ultralytics 8.4.173、CUDA GPU 0、FP16、`imgsz=640`（保留預設
矩形 letterbox，不強制 640×640）。預設暖機 300 幀，量測後續 900 幀。
三類電力設備模型另加 `--task detect --model /path/to/checkpoint.pt`；
`--precision fp32` 可對照原始 PyTorch 精度，`--arms concurrent` 只跑雙模型。
TensorRT 可直接指定 `--model /path/to/model.engine`，精度由引擎決定。

本機已備妥 `.venv-yolo/bin/python`，透過本機 `.pth` 重用 `.venv` 的既有
Torch／CUDA 相依，額外套件不寫入正式 `.venv`。這是可選量測環境，
**不在正式 runtime lock／portable 發布契約內**；搬機後不能直接沿用絕對路徑
`.pth`。其他環境可用 `--yolo-python /path/to/python` 指定已安裝 Ultralytics
8.4.173 與相容 CUDA Torch 的獨立 Python，`--localizer-python` 指定正式定位
Python。權重須事先取得；量測程序禁止網路下載，不會自動換其他模型。
官方模型說明：[Ultralytics instance segmentation](https://docs.ultralytics.com/tasks/segment/)。

`report.json` 的 `compute_e2e_fps` 是上述端到端運算 FPS，
`decode_to_both_results_ms` 保留每幀完整運算鏈的 p50／p95／p99／max；
另有成功結果數、地圖確認數、完整一秒的配對完成數、超過 30 FPS 的
33.33 ms 預算幀數，以及每秒 GPU 用量取樣。
`baseline_results.json`／`concurrent_results.json` 保留逐幀結果與完整偵測框／
分割輪廓；`startup` 保存正式 worker readiness 資料。量測不包含 GUI、
逐幀疊圖、真機曝光／傳輸或遙測。舊的 `segmentation_preview.jpg` 只存在於
2026-10-05 的歷史量測，不由目前工具生成。
COCO 80 類也不等於草地／道路／電力設備的專用分割。

2026-10-05，RTX 5060 Laptop GPU，P1980198 暖機 300 幀後量測相同的 900 幀：

| 指標 | 定位單跑 | 定位＋YOLO26n-seg |
|---|---:|---:|
| 完成整條運算鏈的幀數 | 900 | 900 |
| 實際耗時 | 5.823 s | 10.422 s |
| 端到端運算 FPS | 154.55 | 86.36 |
| 有效定位幀數 | 837 | 900 |
| 有效配對 FPS | 143.73 | 86.36 |
| 完整運算鏈延遲 p50／p95／p99 | 6.07／8.11／11.77 ms | 10.90／14.38／18.48 ms |
| 超過 33.33 ms 的幀數 | 3 | 5 |

並行最大延遲 68.58 ms，897 幀有 mask；0 丟幀。這段的整體運算吞吐超過
30 FPS，但不是每幀的硬即時保證。不限速會改變背景重定位與前景追蹤的時間
關係，兩臂成功幀數差異不能解讀成 YOLO 改善定位品質。

需要確認實際來源節奏下是否跟得上串流，另用
`--mode paced --warmup-seconds 10 --seconds 30`。此模式沿用 production client
的最新幀 coalesce，其更新率受影片 FPS 封頂，**不能用來估計運算吞吐上限**。
舊報告 `outputs/analysis/localization_yolo26n_seg_20261005/report.json` 的
30.0 FPS 屬於此來源限速量測，當時分割 IPC 只回傳摘要，不包含現在的 mask 輪廓。
兩種離線量測都不等於失鎖壓力測試或實機端到端驗收。

### YOLO TensorRT FP16 引擎重建

```bash
.venv-yolo/bin/python tools/export_yolo_engine.py \
  --model 執行環境/models/power_equipment_sample50_3class_yolo26n_640_aug_best.pt \
  --output 執行環境/models/power_equipment_yolo26n_384x640_fp16.engine
```

在目標 GPU 執行；需先備妥 Ultralytics、ONNX、TensorRT 與 CUDA Torch。
工具不更改原始 `.pt`，不下載模型、不訓練、不做 INT8 校正，也不覆蓋既有引擎。
固定 batch 1；由 `--width 1280 --height 720 --imgsz 640` 與模型 stride 計算
與 PyTorch 相同的矩形 letterbox，預設引擎輸入為 `1×3×384×640`。
先匯出具明確 FP16 型別的 ONNX，再建立 TensorRT 引擎，支援移除 FP16 builder
旗標的 TensorRT 11。中間 ONNX 使用暫存目錄，結束後移除。
`.engine` 依賴 GPU／TensorRT 版本，換機或版本後需重新匯出，不納入 portable
runtime lock。FP16 與引擎融合可能造成浮點差異，必須以目標影片比較框／類別／
信心值與定位＋YOLO **同幀完整結果**吞吐後再選用，不把單模型推論 FPS 當 E2E。

影片介面沿用既有 `SFM_DETECTOR_MODEL=/path/to/model.engine`，或直接 app
傳 `--detector-model /path/to/model.engine`。分割引擎則用 `SFM_YOLO_SEG_MODEL`；
task 仍須明確選 `detect` 或 `segment`，座標與 mask 輪廓仍以原始 1280×720 回傳。

本機已實際建立三類偵測引擎與
`執行環境/models/yolo26n_seg_384x640_fp16.engine`，皆為 TensorRT 11 的 FP16
binding。分割 `.pt` 與 `.engine` 各完成 100 個定位＋分割同幀結果、0 丟幀；
各回傳 62 個有效 mask，已檢查框與輪廓的原始像素範圍、confidence 與影格 ID。
這是相容性 smoke，不是速度／準確率比較；收據：
[segment_contract_smoke.json](../outputs/analysis/localization_yolo_optimization_20261006/segment_contract_smoke.json)。

### 2026-10-06：福和橋雙模型實測與採用決策

相同 P1130113 4K 原片、福和橋 profile、三類設備模型、640 尺度：
每臂暖機 300 幀後量測 900 個完整同幀結果，4-thread CPU 設定；
既有使用者 UI／GPU 程序未停止。所有量測臂 900/900 有定位結果、0 丟幀，
但這是 mapping-source 影片，不證明獨立定位精度。

| 量測 | 原始 worker | 優化 worker |
|---|---:|---:|
| PyTorch FP32 同後端交錯兩輪的完整鏈 FPS | 56.82、37.72 | 45.26、39.70 |
| FPS 中位數 | 47.27 | 42.48 |
| detector core p50 的兩輪中位數 | 13.34 ms | 11.50 ms |

**沒有確認 E2E FPS 提升，不切換預設 `.pt` 或精度。** 既有背景 CPU／GPU 程序
仍在執行；完整鏈結果差異很大，未隔離波動來源，不能把較低 YOLO 時間當整體加速。
量測保留於
[`paired_worker/summary.json`](../outputs/analysis/localization_yolo_optimization_20261006/paired_worker/summary.json)。
採用的安全改動是 RGB→BGR 使用 OpenCV SIMD、重用原緩衝區，以及偵測框只做
一次 GPU→CPU 傳輸。RGB microprobe p50 為 2.675 → 0.040 ms，輸出位元一致、
destination identity 不變；正式兩臂同 900 幀的所有偵測框也完全一致。
Ultralytics 精度改用現有 `quantize` API，保留原 FP32／FP16 語意。

另各三輪比較 PyTorch FP32、PyTorch FP16、TensorRT FP16。優化 worker 的
完整鏈 FPS 中位數分別為 **52.88、46.43、43.02**；TensorRT 未優於 FP32，
因此引擎只供明確 opt-in。代表性未改 RGB worker 的第 2 輪，YOLO inference
6.324 → 0.659 ms，但完整鏈 40.68 → 39.04 FPS，證明 inference 不是唯一瓶頸。
報告：[optimized/summary.json](../outputs/analysis/localization_yolo_optimization_20261006/optimized/summary.json)。

TensorRT 與 PyTorch FP16 在 898/900 幀有相同的類別／框數；對 FP32 是
891/900。匹配框對 FP16 的 IoU 中位數 1.0，但最低 0.821、信心最大差 0.00464，
有門檻附近的增刪框，不宣稱精度完全相同。證據：
[box_parity.json](../outputs/analysis/localization_yolo_optimization_20261006/box_parity.json)。
來源影片約 24 FPS；上述不限速 compute FPS 不是顯示 FPS，也不會補出新影格。

實際 `SIMULATED-STREAM` 介面另運行約 77 秒，定位與 TensorRT 偵測 worker
皆就緒，收到 1115 個定位結果、1017 個偵測結果，畫面已顯示偵測框；
最後一組同幀定位／偵測均成功。未連硬體、未啟動 AUTO／起飛。
介面收據／截圖：
[ui_smoke.json](../outputs/analysis/localization_yolo_optimization_20261006/ui_smoke.json)、
[trt_detection_ui.png](../outputs/analysis/localization_yolo_optimization_20261006/trt_detection_ui.png)。

| 檔案 | 用途 |
|---|---|
| `verify_wind_control.py` | 正式控制器的持續漂移、變風向、陣風及延遲／掉幀矩陣；輸出真值指標與逐週期紀錄，見 [抗風驗證](../定位演算法/flight_control/WIND_CONTROL.md) |
| `identify_pcmd_response.py` | 由場次 `trajectory.jsonl` 的已下發 PCMD 與融合 NED 速度辨識延遲一階響應，並重播殘餘漂移觀測器，見 [抗風驗證](../定位演算法/flight_control/WIND_CONTROL.md) |
| `autoflight_error_report.py` | 依 AUTO run ID 與永久終結摘要回顧首次到達航點後的規劃折線誤差；無樣本與舊紀錄證據不足明確標示 |
| `verify_flight_diagnostics.py` | 不連飛機的正式控制／紀錄生產者讀回、永久保留與寫入／同步／溢位故障 smoke |
| `benchmark_operator_latency.py` | 唯讀 RGB 提交比較與有限時間 simulated-stream GUI 啟動／heartbeat／render stage count、p50、p95 量測，支援視窗尺寸與擷取畫面，不略過既有驗證 |
| `workspace_audit.py` | 唯讀檢查目錄、必要檔案、symlink、output 命名與容量 |
| `check_maintainability.py` | 以分區 C901 預算阻止複雜度熱點數量或最壞值回升 |
| `build_dense_display_ply.py` | 由 release 的 `model/points3D.bin` 產生航線編輯器用的完整點雲 `map/map_dense.ply`，綁定座標系與 `map.ply` SHA-256；`--mvs-fused <ply>` 改寫稠密化彩色點雲（以固定記憶體串流抽樣、濾離群點、輸出至編輯器 300 萬點上限，需 scipy；release 有 `dense_surface.json` 時須同座標系，沒有時 `map.ply` 至少 30% 要落在有稠密點的格內） |
| `verify_dense_alignment.py` | 證明稠密點雲與 COLMAP 模型同座標系：取樣關鍵影格各自觀測到的 3D 點，檢查稠密點雲在同一像素的最近表面深度是否一致（容差約 5%，預設 60 張影格、至少 30% 一致才通過；烏來正確點雲 0.62、錯座標系舊點雲 0.02）。`build_dense_display_ply.py` 的 `map.ply` 重疊率檢查擋不住幾單位的偏移，換稠密點雲時須先跑這個 |
| `simulator_preflight.py` | 模擬介面啟動前檢查 Python、CUDA、模型、場域資產與影片 |
| `offline_wheelhouse.py` | 建立／驗證與三份 lockfile 及每個 wheel SHA-256 綁定的離線安裝庫 |
| `install_runtime.sh` | 用 hash lock 建立乾淨 CPython 3.10 venv；`--offline` 強制 `--no-index` |
| `export_simulator_package.py` | 依 runtime allowlist 匯出固定程式；可用 `--site-profile` 收錄完整 hash-bound 場域資產，排除 venv、執行輸出與影片 |
| `package_manifest.py` | 驗證目前可攜式發布包的 MANIFEST.tsv / SHA256SUMS |
| `release_activation.py` | 驗證 commit/version-bound package，原子 stage/activate/rollback |
| `test_clean_install.sh` | 在暫存目錄重建 CPython 3.10 venv 並執行 runtime preflight |
| `simulated_ui_smoke.sh` | 實際啟動地圖/影片選擇路徑，等 GUI 產生有效 pose 後安全終止 |
| `test_portable_runtime.sh` | 依 package kind 驗證入口、乾淨離線安裝；有場域時要求 GUI 有效 pose，真機操作包另做 live launcher dry-run |
| `../tests/tools/test_workspace_audit.py` | 驗證工作區結構契約 |
| `security_dependency_gate.py` | 執行 pip-audit、驗證到期中的安全例外，並輸出 CycloneDX SBOM |
| `sim_route_autoflight.py` | 〔離線〕以正式 `RouteAutoController` 與降階 ANAFI 模型做無比例閉環模擬；輸出 `summary.json`、`trace.csv`、圖與掃描 `sweep.csv`，見下方「任務與路線離線工具」 |
| `anafi_wind_env.py` | 〔離線〕Gymnasium 水平定點環境；重用 R5 降階 plant、正式定點控制器與地速包絡，融合式延遲觀測、兩軸 PCMD 殘差，不接 Sphinx／真機 |
| `train_anafi_wind.py` | 〔離線〕CPU PPO 訓練、保存／重載模型、同種子基準配對評估；保存契約、來源 SHA-256、版本與逐情境結果，見下方「ANAFI 強化學習離線訓練」 |
| `verify_preset_routes.py` | 〔離線〕對 `river_site/routes/*.json` 逐條、逐起點做 waypoint-1 起飛的閉環驗收（含風與回航）；輸出 `results.json`、`wind_results.json`、`verification_manifest.json` |
| `test_route_autoflight.py` | 〔離線〕單一 route JSON 的 dry-run 驗收：驗證航線格式並以正式控制器＋簡化動力學跑完；通過／失敗以結束碼表示。名稱雖以 `test_` 開頭，是 CLI 工具，不由 pytest 收集 |
| `imu_flight_test_report.py` | 〔離線，唯讀〕檢查 IMU 飛行測試場次的遙測／影像完整性與時序；輸出 markdown 或 JSON，不構成精度或飛行核准 |
| `report_localization_performance.py` | 〔離線，唯讀〕由 flight-log 場次計算端到端定位交付率（不重複幀 FPS、延遲與品質觀測）並對照 30 FPS 計畫門檻，輸出 JSON。`quality` 另含地圖確認失鎖次數與回復時間（開機前的未確認不算失鎖，結尾未回復另標）及 client 佇列丟幀（累計值取最大） |
| `report_map_height_quality.py` | 〔離線，唯讀〕各地圖的高度**一致性**報告：以地圖 SHA／對齊 SHA 歸屬 flight-log，僅用 AUTO 強定位、非重複、新鮮氣壓高度樣本，將 3 秒窗內氣壓升降與地圖高度同向／反向／凍結以該路線到站半徑正規化分類（同比縮放地圖結果相同）。無該圖實飛資料為 `NO_FLIGHT_EVIDENCE`；最佳結論僅 `CONSISTENT_NOT_VALIDATED`，不輸出高度精度或公尺尺度 |
| `import_direct_edm_bundle.py` | 〔離線〕把可攜 direct BoQ/EDM/COLMAP 地圖轉成正式 EDM bundle；輸出 `localization_bundle.pt`、`reference_poses.json`、`map.ply`、`import_summary.json`；需 torch／pycolmap／OpenCV |
| `resolve_mission.py` | 〔離線〕解析 mission selection 並回報 localization／flight readiness；`--require` 不符時結束碼非零；`--materialize-site-profile` 寫入相容快照 |
| `pin_mission_selection.py` | 〔離線〕由各元件 manifest 建立 SHA-256 綁定的新 selection，不覆寫既有檔，未達定位相容就不建立輸出 |
| `update_mission_route.py` | 〔離線〕一鍵換 selection 的路線：重算 route／route package SHA、改寫兩份檔並以 resolver 驗證，失敗即還原；支援 `--dry-run` |
| `stick_override_bench.py` | **〔地面台架，需實體 SkyController〕** 以真實 `SkyControllerStickMonitor` 讀 HID 搖桿，驗證撥桿會觸發交回控制權的判定；不連飛機、不飛行，但需人工撥桿。**僅由操作員在地面執行，agent／CI 一律不得執行** |

`requirements/runtime-lock.txt` 是 runtime 的唯一 hash lock；`requirements/test-lock.txt`
另外固定 pytest、ruff、coverage 與 timeout plugin。正式 portable 先在來源機建立
wheelhouse：

```bash
.venv/bin/python tools/offline_wheelhouse.py build \
  --output /path/to/approved-wheelhouse \
  --requirements requirements/runtime-lock.txt \
  --requirements requirements/test-lock.txt \
  --requirements requirements/quality-lock.txt
```

乾淨測試可用 `bash tools/install_runtime.sh --offline --test-deps`；含品質工具時使用
`bash tools/install_runtime.sh --offline --test-deps --quality-deps`。
offline 模式會先核對 `WHEELHOUSE.json`、三份 lockfile 與所有 wheel，再以
`--no-index --find-links` 安裝。安裝器會拒絕
`include-system-site-packages=true` 的既有 venv。scipy 已固定於 runtime lock，
供診斷與 deploy 使用；桌面操作介面不再有稀疏點雲近接懸停互鎖，也沒有
`--require-collision-monitor`。不得宣稱具備 collision protection。

Portable manifest 的 scope 由 `tools/package_manifest.py` 唯一決定；editor files、
cache、workspace 執行輸出、audit review artifacts 與其他 `.gitignore` 對應的非發布
資料不會因為偶然存在於 workspace 就進入 MANIFEST.tsv。
本機 `.venv`、`.venv-yolo` 與 `.venv-mvs` 都是執行環境，不納入 source／portable manifest；
其 `lib64` 等環境內 symlink 不會被當成發布來源。產生 manifest 不刪除這些環境。
根目錄 `runs/` 訓練輸出與 `執行環境/last_mission_selection.json` 是本機產物，
不納入發布清單；三類電力設備偵測權重也不列入 Git 來源清單。

發布收據與 `PORTABLE_PACKAGE.json` 必須同時帶 manifest/SHA256SUMS、commit、version
與 dirty 狀態。`release_activation.py` 先驗證完整 package，再以 temporary directory
與 atomic symlink replacement 建立 `current`／`previous`；dirty source 預設拒絕，只有
明確傳入 `--allow-dirty`（development）才可 stage 或 activate，rollback 也會重新驗證
目標 package。即使 manifest 正確，`offline_install.complete` 不是字面 `true` 或
wheelhouse 與 lock/digest 不符也不得 activation。

要重查目錄結構時可執行：

```bash
python tools/workspace_audit.py --strict-output-names
```

## 外部 runtime artifacts

`RUNTIME_ARTIFACTS.json` 是 Git source 內的 runtime artifact registry，不是下載器。
它只列出 portable runtime 真正需要的相對路徑、大小與 SHA-256；模型 cache 的其他
repo、checkpoint 與暫存檔不會被 exporter 掃描或複製。來源工作區的
`MANIFEST.tsv` / `SHA256SUMS` 是 source-only manifest，外部 artifact 則由 registry
獨立綁定；portable 輸出包的 manifest 會同時包含已解析的 allowlist 檔案。
來源 checkout 可用 `python tools/package_manifest.py verify --source-only` 檢查；
portable package 則在其根目錄執行一般的 `verify`。

乾淨 checkout 沒有外部 artifact 時，exporter 會 fail closed，列出缺少的相對路徑與
digest，不會嘗試網路下載。請從受核准的離線 artifact bundle 建立與 registry 相同的
目錄樹，再指定 seed root：

```bash
python tools/export_simulator_package.py /path/to/portable_localization \
  --artifact-root /path/to/seed-root \
  --wheelhouse-root /path/to/approved-wheelhouse \
  --site-profile 地圖檔/場域/river_site/site_profile.json
# 或：SFM_RUNTIME_ARTIFACT_ROOT=/path/to/seed-root python tools/export_simulator_package.py ...
```

`--artifact-root` 下的檔案必須以 repository-relative path 放置；resolver 會先驗證
大小，再驗證 SHA-256。沒有 registry 或 digest 不符時不會產生部分可信的 runtime 包。
沒有 `--wheelhouse-root` 的一般匯出不會假裝成離線包；其
`PORTABLE_PACKAGE.json` 會明確標示 `offline_install.complete=false`，不得當作正式
activation 輸入。
有 `--site-profile` 時，`PORTABLE_SITE_ASSETS.json` 另外綁定 profile 與每個場域檔案；
reference index 不能只帶 manifest，signed hardware approval 也不能缺 receipt、
signature 或 trust store sidecar。

## 工程品質 gates

`requirements/quality.txt` / `requirements/quality-lock.txt` 是獨立的品質工具
清單，不會改寫 runtime 或 test lock。CPython 3.10 可由安裝器加入現有驗證環境：

```bash
bash tools/install_runtime.sh --quality-deps
python -m mypy
ruff check . --select S102,S105,S106,S107,S602,S604,S605,S608 \
  --exclude '**/test*.py' --exclude '**/tests/**' \
  --exclude '定位演算法/deploy_code/runtime/**'
python tools/security_dependency_gate.py
```

`mypy` 只對 `pyproject.toml` 的明確 typed boundary 清單設 fail gate，並不宣稱
整個 Tk/Olympe/Torch 應用程式已完成型別化。安全 gate 的 vulnerability exception 在
`pyproject.toml` 以 package、advisory ID、理由和 ISO 到期日保存；到期或未列出的
漏洞都會 fail。pip-audit 的 `skip_reason` 不能以一般 vulnerability exception
略過，必須在 `[[tool.security_dependency_gate.skip_exceptions]]` 以精確的 package、
version、`skip_reason`、理由和到期日列出。版本、原因改變或例外過期都會 fail，避免
把新的未稽核套件默認視為安全。`security_dependency_gate.py` 會在
`outputs/security/` 產生 `pip-audit.json` 和 `sbom.cyclonedx.json`；非 PyPI 的
CUDA wheel 即使被 pip-audit skip，也會以 `audited=false` 與原始 `skip_reason`
保留在 CycloneDX JSON SBOM 中。

## 系統優化的離線工具（2026-09-22）

- `audit_camera_calibration.py --site <site.json> --out <audit.json>`：檢查內參、畸變與來源收據，量化保留原 K 的校正位移。不能取代尺度量測或實飛驗收。
- `report_relocalization.py --capture <capture.json> --out <report.json>`：按獨立重定位工作統計 p50/p95/p99、缺失階段與最慢案例；不是按重複 UI 記錄計數。
- `replay_telemetry_ekf.py --session <session> --out <directory>`：用前 40% 時間校準水平尺度／方向，後 60% 比較純視覺與速度融合 ESKF；每五個強觀測保留一個不送入濾波器。需要包含地圖軸的 route plan、相機軸、來源時間與速度。資料不足時報告原因，不產生假校準。

上述工具只讀資料與寫入報告，不載入 Olympe 或發送控制命令。詳細結果與限制見
[`system_optimization_20260922.md`](../docs/system_optimization_20260922.md)。

## ANAFI 強化學習離線訓練

使用獨立的 `~/.local/share/venvs/drone-rl` Python 3.10 環境；不向真機 runtime
加入 Gymnasium／Stable-Baselines3，不連飛機、不啟動 Sphinx，也沒有 live 模式。
在 repository 根目錄執行：

```bash
# 第二輪魯棒訓練；--out 必須是尚不存在的目錄
~/.local/share/venvs/drone-rl/bin/python tools/train_anafi_wind.py \
  --timesteps 1000000 --seed 33 --n-envs 4 \
  --reference-model outputs/anafi_rl/ppo_horizontal_hold_seed17_quantization_fixed/policy.zip \
  --eval-seeds 2000017 2000018 2000019 2000020 2000021 2000022 2000023 2000024 2000025 2000026 \
  --out outputs/anafi_rl/ppo_horizontal_hold_robust_seed33

# 在相同目前環境重載新舊 actor，另指定尚不存在的結果目錄
~/.local/share/venvs/drone-rl/bin/python tools/train_anafi_wind.py \
  --evaluate-only outputs/anafi_rl/ppo_horizontal_hold_robust_seed33/policy.zip \
  --reference-model outputs/anafi_rl/ppo_horizontal_hold_seed17_quantization_fixed/policy.zip \
  --eval-seeds 2000017 2000018 2000019 2000020 2000021 2000022 2000023 2000024 2000025 2000026 \
  --out outputs/anafi_rl/ppo_horizontal_hold_robust_seed33_recheck
```

任務固定高度／航向，20 Hz 控制、200 Hz plant 積分。PPO 只輸出 roll／pitch
殘差，每軸最多 10 PCMD 百分點；基準優先分配後仍通過正式加速漸增與地速包絡，
積分器回饋排除殘差。這是離線動作額度，不是已核准的真機設定或物理速度硬保證。
殘差先量化成整數 PCMD 再套分配邊界；不得因 1.9 四捨五入成 2，就把
整個向量衰減。配置非整數額度時整數命令上限向下取整，仍不超過配置額度。

Actor 使用 8 個歷史 frame，共 184 個 float32：機體水平定位誤差、融合式地速、
航向誤差、roll／pitch、基準與先前核准命令、先前實際分配殘差、各來源年齡及有效性。
定位誤差依可用地圖尺度與伺服範圍正規化，不把 map units 當公尺。
SDK 姿態／速度時間是 host receipt，不能代替不可見的 aircraft capture time。
過期或缺失的定位／姿態／速度停用 RL 殘差；快取樣本不重新標成當下時間。
真實位置、漂移及 plant 參數僅供 reward／診斷，不能進入 actor。
`AnafiWindEnv.contract()` 保存欄位順序、尺度、時序及失效契約；
`pack_observation()` 可共用觀測格式，但尚未接上真機推論入口。

訓練情境為 calm、steady、gust、reversal、dropout，stress 的組合／擾動曲線不參與訓練。
第二輪涵蓋其慢響應與延遲元件，不能再把所有 stress 參數稱為分布外。
預設仍用三個評估 seed；上述第二輪命令預先指定十個新保留 seed。
`--reference-model` 驗證舊模型 SHA-256 與輸入／輸出語意，在同一目前環境比較
零補償、舊 actor、新 actor；不宣稱重現舊環境結果，也不移除失敗／提前終止案例。
舊模型維持凍結，新模型從頭訓練。`policy.zip` 為模型；`metadata.json` 保存版本、
來源／模型 SHA-256、實際 steps 與契約；`evaluation.json`／`evaluation.csv` 保存
逐 seed 指標與配對差值。SB3 完整 rollout 會向上取整，實際 steps 以 metadata 為準。
限幅比例比較最近整數 PCMD 與核准命令，包含基準包絡修改及到達命令上限，
不把單純的小數轉整數算成限幅。環境契約目前為 schema 2，舊模型不得直接
搭配 `--evaluate-only` 混用新來源；只能經語意核對的 `--reference-model`
在目前環境比較。第一輪模型與原始報告保留不覆寫。
恢復時間只適用有限脈衝／反向情境：擾動結束後，連續 1 秒位置誤差 ≤0.2 m、
地速 ≤0.1 m/s 的首個穩定窗起點；未恢復或不適用保留 null，不當成 0 秒。

重用的 R5 名目模型使用水平時間常數 3 s、tilt lag 0.4 s、加速度增益 1.09。
第二輪未驗證的魯棒範圍為水平時間常數 2–4.5 s、tilt lag 0.25–0.75 s，
速度遙測延遲 0.12–0.35 s、命令延遲 0–0.10 s；reward 同時抑制超額要求、
未分配補償與地速，不只獎勵位置誤差。這不是新的實飛校準。
擾動單位是**殘餘地面漂移 m/s，不是大氣風速**。
融合觀測與定位雜訊是合成模型，不是實際相機／EKF pipeline；無障礙物、垂直擾動
或完整機體氣動模型。訓練完成或離線改善均不構成抗風認證／實飛許可；
Sphinx／真機時序、命令歸屬與操作員接管仍須另行整合驗證。

Sphinx 地面風場檢查收據：
[`sphinx_ground_wind_probe_seed33/report.json`](../outputs/anafi_rl/sphinx_ground_wind_probe_seed33/report.json)。
已核對本機韌體 SHA-256、啟動 Sphinx／headless UE，對 `world:wind/wind` 寫入並讀回
無風、東／北／西向 2 m/s，最後恢復無風並停止自建程序。沒有 Olympe 連線或飛行命令；
這只驗證原生風場參數接口，不是 PPO 在物理風場的飛行結果。飛行驗證須由操作員
先起飛／保持懸停，再使用受限的模擬器接口；不可把地面讀回當作抗風驗證。

第二輪收據：
[`ppo_horizontal_hold_robust_seed33/metadata.json`](../outputs/anafi_rl/ppo_horizontal_hold_robust_seed33/metadata.json)、
[`evaluation.json`](../outputs/anafi_rl/ppo_horizontal_hold_robust_seed33/evaluation.json)。
完成 1,003,520 steps；十個新 seed、六情境共 60 組三方比較。
在同一 schema 2 環境，新模型平均 RMSE 比凍結舊 actor 低，但 19/60 組仍退化，
嚴苛情境 0/10 達到恢復門檻。新舊環境／seed 不同，不能直接和第一輪表格比百分比；
這是改善但未通過部署門檻的離線模型，沒有 PPO 物理飛行抗風證據。

## 任務與路線離線工具

以下除 `stick_override_bench.py` 外都不連飛機、不載入 Olympe、不發送控制命令。
模擬與驗收的結果是近似模型，不是飛行許可。

```bash
# 閉環模擬：單條路線、3 個隨機種子，產物寫到 --out（預設 outputs/sim_autoflight/run_<時間>）
.venv/bin/python tools/sim_route_autoflight.py \
  --route 地圖檔/場域/river_site/routes/flight_route_<revision>.json --seeds 3 --out /tmp/sim

# 預設路線全數驗收；--out 與 --localization-wait-s 必填，--quick 為每個起點單一種子
.venv/bin/python tools/verify_preset_routes.py --quick --out /tmp/preset-routes \
  --localization-wait-s 0

# 單一路線 dry-run（結束碼 0 通過；--min-progress 預設 0.95）
.venv/bin/python tools/test_route_autoflight.py --route <route.json>

# IMU 飛行測試場次完整性；--json 輸出 JSON，--out 另存 markdown
.venv/bin/python tools/imu_flight_test_report.py --session outputs/flight_logs/session_場次名稱

# 定位交付率；--session 可重複，--window-s／--warmup-s／--nominal-fps 調整統計窗
.venv/bin/python tools/report_localization_performance.py \
  --session outputs/flight_logs/session_場次名稱 > /tmp/localization-performance.json

# 匯入 direct EDM 地圖（輸出目錄新建）
.venv/bin/python tools/import_direct_edm_bundle.py --source /path/to/portable_map --out-dir /tmp/edm-bundle
```

`verify_preset_routes.py` 與 `sim_route_autoflight.py` 的預設路線讀取 `地圖檔/` 內的
私有場域資產；沒有這些資產時請以 `--route` 指定。

mission selection 工具鏈（皆需在 workspace 內，路徑以相對於檔案的 SHA-256 參照綁定）：

```bash
.venv/bin/python tools/resolve_mission.py 控制介面程式/mission_selections/<selection>.json --require valid
.venv/bin/python tools/pin_mission_selection.py --output 控制介面程式/mission_selections/<new>.json \
  --selection-id <new> --vehicle <vehicle.json> --site <site.json> --map <map.json> \
  --localizer <localizer.json> --calibration <quality-receipt.json>
.venv/bin/python tools/update_mission_route.py \
  --selection 控制介面程式/mission_selections/<selection>.json \
  --route 地圖檔/場域/<site>/routes/flight_route_<revision>.json --dry-run
```

`stick_override_bench.py` 是地面台架：連接並開啟遙控器後執行
`python tools/stick_override_bench.py [秒數]`，在時限內撥動飛行搖桿（軸 0–3，超過
deadzone 才算；相機／雲台軸 4–5 不會奪取控制權），PASS 代表判定被觸發。
找不到 SkyController joystick 節點或全程無觸發時結束碼為 1。它使用
`operator_interface/skycontroller_stick.py` 的正式監視器，不經 Olympe。

## 擴圖版（工作圖）建圖工具

| 工具 | 用途 |
|---|---|
| `build_direct_site_release.py` | 建立 SHA-bound direct 場域包；加 `--online-workmap` 時 profile 啟用工作圖，且 `localizer_deploy_dir` 指向 `定位演算法/deploy_code_expand/sfm_direct_deploy`，沒加則指向原始版本 |
| `build_workmap_release.py` | 從已完成工作圖 session 建立離線 deploy pack，不啟用 mission 或 AUTO |
| `workmap_reconstruction.py` | 以可信觀測重建工作圖，排除未併入／已融合的重複地標 |
| `workmap_pack_assets.py` | 打包幾何與來源一致的模型、參考影像與 bundle |

後三者載入擴圖版演算法；測試在 `tests/localization/deploy_expand/`（`pytest -c pytest-expand.ini`）。

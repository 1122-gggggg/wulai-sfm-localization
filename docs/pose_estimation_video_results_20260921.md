# 2026-09-21：兩段影片的姿態估計比較

結論：保留正式 profile 的 2 次 refinement。3～5 次可以作低成本候選，但這兩片尚未提供一致的幾何精度改善證據；純視覺 ESKF 與本版滑動視窗不升級為正式定位來源。

下次需要真機遙測的部分已記錄於 [EKF 同步資料收集清單](next_flight_ekf_capture_20260921.md)，也已從既有下次飛行 runbook 連結。

## 輸入與方法

- P174：`/home/allen/下載/P1740174_720p.MP4`，1280×720，完整 1,829 幀。
- P172：`/home/allen/下載/P1720172.MP4`，2688×1512，實際 2,979 幀；OpenCV 與 ffprobe 的讀取計數一致，容器宣告 2,980 幀。
- 兩片均依來源 PTS 執行，nominal 23.976 fps，沒有 timestamp fallback；不沿用舊報告的 29 fps。
- 兩片沒有可配對的 NED／IMU 遙測，也沒有獨立位置真值。P174 的 720p 轉碼與歷史原片缺少 hash 對照，不把相同幀數當作已證明同源。
- 當前 DIRECT frontend、相同 profile／map bundle，完整抽取每幀 PnP 前的 KLT／VO 對應。模型載入與 warmup 不列入逐幀延遲。
- 每幀只跑一次 RANSAC，seed=0；0／2／3／5／10 次 refinement 使用相同初始解及 inlier mask 的獨立副本。
- ID 可被 5 整除的觀測不送入任何候選估計器；其他約 80% 用於估計。主要幾何指標只看正 ID 的地圖點，排除負 ID 的 VO 自我一致性。
- 所有方法的誤差表使用完全相同的可評估影格：每幀至少 6 個留出地圖點，且各方法均有輸出。
- 這是固定基準前端輸入的條件化比較。過去 KLT／VO／播種歷史來自 baseline，沒有讓各候選重新影響前端；留出觀測也不是獨立公尺真值。

硬體：Intel Core i7-13620H、RTX 5060 Laptop GPU。CPU 估計比較固定 OpenBLAS／OMP／MKL 各 1 執行緒；float64 幾何資料。NumPy 2.2.6、pycolmap 4.0.4；其他版本與輸入／程式雜湊保存在原始輸出。每方法排除前 10 個可求解影格的暖機時間。

## 結果

誤差為留出地圖點的重投影 px，越小越好，但不是位置真值誤差。耗時為 **RANSAC＋refinement＋可選 temporal stage**，不含解碼、KLT、reloc，也不是正式飛行端到端延遲。

### P174

| 方法 | 重投影 p50 / p95（px） | 估計耗時 p50 / p95（ms） |
| --- | ---: | ---: |
| 0 次（僅 RANSAC） | 1.3780 / 3.1914 | 0.344 / 0.556 |
| 2 次（目前） | 1.2867 / 3.0338 | 0.908 / 1.242 |
| 3 次 | 1.2852 / 3.0385 | 1.046 / 1.421 |
| 5 次 | 1.2847 / 3.0450 | 1.334 / 1.795 |
| 10 次 | 1.2847 / 3.0514 | 1.820 / 2.605 |
| 純視覺 ESKF | 1.2946 / 3.0438 | 1.425 / 1.768 |
| 5 幀視窗 | 1.2888 / 3.0364 | 33.570 / 37.902 |
| 10 幀視窗 | 1.2984 / 3.0416 | 53.605 / 60.855 |

共同評估 1,497 幀、78,583 筆留出地圖點；此範圍沒有方法缺失輸出。

### P172

| 方法 | 重投影 p50 / p95（px） | 估計耗時 p50 / p95（ms） |
| --- | ---: | ---: |
| 0 次（僅 RANSAC） | 1.2648 / 3.1588 | 0.350 / 0.576 |
| 2 次（目前） | 1.1544 / 2.9867 | 0.904 / 1.233 |
| 3 次 | 1.1507 / 2.9935 | 1.036 / 1.391 |
| 5 次 | 1.1494 / 3.0013 | 1.305 / 1.756 |
| 10 次 | 1.1474 / 3.0068 | 1.744 / 2.530 |
| 純視覺 ESKF | 1.1855 / 3.1582 | 1.413 / 1.754 |
| 5 幀視窗 | 1.1617 / 2.9921 | 33.375 / 40.985 |
| 10 幀視窗 | 1.1615 / 2.9985 | 52.867 / 64.655 |

共同評估 2,371 幀、117,402 筆留出地圖點；此範圍沒有方法缺失輸出。

## 如何解讀

1. **現有 refinement 有效，但增加次數的收益快速變小。** 0→2 次兩片的重投影誤差都改善；2→3／5 次只有很小的中位數改善，p95 反而略增。沒有證據支持現在就調高正式迭代上限。
2. **幀間變化減少不等於更準。** 2→5 次的估計位置二階差分 p95：P174 8.06→5.58、P172 13.41→9.50 u/s²。這同時包含真實運動與雜訊，沒有真值不能直接稱為抖動誤差下降。
3. **純視覺 ESKF 更平順但幾何一致性未改善。** P172 的地圖點 p95 從 2.9867 升到 3.1582 px。這只測常速模型＋visual pose，不是融合 NED／raw IMU 的 EKF。候選使用固定參數，沒有按這兩片調參找最好結果。
4. **本版視窗不適合直接上線。** 5／10 幀的中位數約 33／53 ms，還沒算 KLT 與其他工作；10 幀已超過約 41.7 ms 的 24 fps 影格間隔，5 幀的 p95 也幾乎吃完這個預算。幾何改善不穩定。
5. **這不否定所有 sliding-window 方法。** 本版是 SciPy CPU 原型、最多 80 個共享 track、5 次 function evaluations，以共享局部點修正耦合多幀。固定首姿態，使用當幀 PnP 正則項，沒有正式 marginalization、NED 因子或 Ceres／GTSAM 工程優化。

## EKF 輸出不能冒充定位覆蓋率

| 影片 | 基準 PnP 有輸出 | ESKF 接受視覺更新 | ESKF 純預測輸出 |
| --- | ---: | ---: |
| P174 | 1818 | 1813 | 5 |
| P172 | 2762 | 2740 | 32 |

ESKF 的 prediction 包含拒絕 visual innovation 後的短暫預測，不能把它加回 visual success。連續超過 0.5 秒沒有接受視覺更新即清除狀態；epoch 改變也會重置。

## 可重跑的工具

- [pose_estimation_experiment.py](../定位演算法/validation/pose_estimation_experiment.py)：提取／配對比較。
- [pose_filter_experiment.py](../定位演算法/validation/pose_filter_experiment.py)：12 維 error state 的純視覺 ESKF，SO(3) transport／reset Jacobian、Joseph covariance update。
- [pose_window_experiment.py](../定位演算法/validation/pose_window_experiment.py)：固定大小多幀 pose／point 原型。
- [pose_experiment_report.py](../定位演算法/validation/pose_experiment_report.py)：共同影格、map-only 留出指標。

```bash
# 純離線，不連接飛機。P172 可換成另一個輸入檔與輸出目錄。
HF_HUB_OFFLINE=1 TORCH_HUB_OFFLINE=1 .venv/bin/python \
  定位演算法/validation/pose_estimation_experiment.py capture \
  --video /home/allen/下載/P1740174_720p.MP4 \
  --out outputs/analysis/pose_experiments_20260921/p174_capture

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  定位演算法/validation/pose_estimation_experiment.py evaluate \
  --capture outputs/analysis/pose_experiments_20260921/p174_capture \
  --out outputs/analysis/pose_experiments_20260921/p174_results

.venv/bin/python 定位演算法/validation/pose_experiment_report.py \
  --capture outputs/analysis/pose_experiments_20260921/p174_capture \
  --results outputs/analysis/pose_experiments_20260921/p174_results
```

資料提取的 relocalizer 在主執行緒量測真實工作時間，再依影片時鐘延後交付，沒有重現正式執行緒併行與 latest-pending 語意；因此不宣稱端到端 realtime coverage 驗收。比較階段重用同一份 captured schedule，避免某方法取得較有利的播種。

## 證據與檢查

原始資料在 `outputs/analysis/pose_experiments_20260921/`：

- [P174 comparison.json](../outputs/analysis/pose_experiments_20260921/p174_results/comparison.json)、[CSV](../outputs/analysis/pose_experiments_20260921/p174_results/comparison.csv)
- [P172 comparison.json](../outputs/analysis/pose_experiments_20260921/p172_results/comparison.json)、[CSV](../outputs/analysis/pose_experiments_20260921/p172_results/comparison.csv)
- 各 `*_capture/capture.json` 包含影片、profile、bundle、觀測檔雜湊及逐次 reloc 工作；`*_results/summary.json` 包含估計器程式雜湊與執行緒環境。

新增離線測試 **19 passed**，覆蓋共享 RANSAC／holdout 隔離、時戳與資料一致性、SO(3) covariance、掉點與 outlier、視窗耦合／大小／世代，以及 map-only 共同影格統計。Ruff 與 `git diff --check` 通過。正式 profile 與飛行控制未因這次實驗更動。

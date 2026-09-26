# 五項系統改善與驗證邊界（2026-09-22）

本輪以開始時的未提交工作區為基準，保留既有飛控修正。新增的去畸變與 ESKF
都是離線候選，尚未接入正式飛控。相機來源已由操作員確認；實測地圖尺度尚未提供。

## 1. 品質門檻與模組拆分

- 修正 AUTO 加速狀態的 optional tuple 型別，mypy 的 21 個既有邊界通過。
- 拆出場域資產雜湊驗證與 EKF 報告段落，不提高複雜度預算。tools 超標函式數
  6 → 5，control 26 → 24；所有分區與行數門檻通過。
- 羅盤軸繪圖移至既有 `operator_rendering.py`；GPS、衛星、訊號、雲台及變焦
  的唯讀輪詢移至既有 `backend_telemetry.py`，原方法保留薄包裝。
- UI 主程式 7,418 → 7,335 行，live backend 5,231 → 5,161 行。
- 比對開始時快照，49 個涉及起飛、降落、清理、微移、freeze、關閉及鍵盤操作
  的方法 AST 未變。README／架構圖的檢索器名稱改為實作中的 BoQ。

## 2. 相機與尺度

`intrinsics_provenance.json` 將操作員「ANAFI 原始串流影像，未另外去畸變」的確認
綁定到 `intrinsics.json` SHA。場次同步保存來源收據。來源確認、數值有效、與
PINHOLE 內參匹配、主機是否已校正，是不同欄位，不能互相替代。

新增檢查拒絕錯誤矩陣末列、非有限畸變及可被 NumPy broadcasting 誤判相同的
畸形相機參數。對本機內參的 375 點網格，保留原 K 的去畸變位移 p50 5.89 px、
p95 32.58 px、最大 50.41 px。這是校正模型的像素位移，不是位置誤差改善量。

離線 capture 可用 `--raw-anafi-intrinsics` 比較原始與去畸變影像；它保留相同
PINHOLE K，明確記錄校準雜湊及前處理時間。合成投影反算、零畸變影像一致性、
尺寸縮放及錯誤相機拒絕測試通過。未把係數套到來源未確認的 P172/P174，也未
在缺少 ANAFI 影像驗證時自動變更正式影像管線。

尺度報告修正三個問題：

1. 使用影像 capture 與速度來源時間，去除重複輪詢；弱定位／IMU 橋不當成地圖觀測。
2. 不跨越超過 0.5 秒的觀測中斷積分；只使用視覺與遙測的共同時間區段。
3. 前後半段漂移依時間順序計算。原先排序後才分半，下降 0.5 倍會錯報成上升 2 倍。

既有實飛快照目前只有 2 個有效尺度片段，報告保持未收斂。沒有產生實測尺度、
高度基準或獨立位置真值。

## 3. 重定位延遲

新增 `edm_match_ms` 與 `lift_prepare_ms`，一路經 tracker、adapter、worker metric
到永久 trajectory 日誌。兩者分割既有 `match_lift_ms`，不能重複相加。
`lift_prepare_ms` 也包含幾何篩選與準備，不能全部歸因於 2D–3D lift。

報告依 epoch／ordinal／capture 時間去重，避免同一次重定位被每個 UI 影格重複
計數；缺少識別或階段的舊紀錄不補零。離線 capture 另保存來源及輸出檔案雜湊。

| 本輪完整影片 | 影格 | 重定位工作 | p50 | p95 | p99 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P172 | 2,979 | 188 | 192.8 ms | 266.3 ms | 326.4 ms | 343.3 ms |
| P174 720p | 1,829 | 78 | 189.7 ms | 284.0 ms | 318.7 ms | 321.7 ms |

P172 EDM 匹配 p95 120.3 ms，其餘 match/lift 處理 p95 130.2 ms。固定影格的
CPU profiler 也看到雙視圖幾何篩選及 PnP 成本，不能只把長尾歸因於 GPU。
舊 1,613 ms 案例仍未重現，原因未證實，本輪不宣稱已修復該長尾或得到加速比。

GPU 工作經暖機，每個 capture job 外層做 CUDA synchronize；內部分段是 host
wall time。P174 當時有離線測試負載，數字不是隔離主機基準。Scheduled capture
沒有重現正式 UI／worker 併行，不能把上述時間當成實飛端到端延遲。

## 4. 離線速度融合 ESKF

沿用既有 12 維視覺 ESKF，新增校準後的水平速度觀測、創新量拒絕、Joseph
協方差更新與來源時間去重。速度不能延長 0.5 秒視覺錨點期限；不使用 raw IMU，
不把機身姿態當相機姿態，也不把未校準高度加入濾波器。

重播以前 40% 時間估水平尺度／方向，後 60% 評估；每五個強視覺觀測保留一個，
不送入任何候選的更新或拒絕判斷。分別記錄接受的視覺觀測與純預測輸出。
韌體速度與視覺可能相關，量測噪聲仍是假設，協方差不是實測誤差保證。

合成圓形運動、三個 seed、真尺度 2.5 m/u，估計為 2.499–2.502 m/u；每組評估
有 234 個共同留出觀測。其位置殘差 p95：

| seed | 純視覺 ESKF | 融合水平速度 ESKF |
| --- | ---: | ---: |
| 7 | 0.01627 u | 0.01356 u |
| 11 | 0.01685 u | 0.01447 u |
| 19 | 0.01703 u | 0.01503 u |

這些只是指定合成模型的結果。9/18 保存的實飛快照，校準前段沒有足夠連續且
有位移的強定位，因此程式明確回報無法執行 A/B；不降低校準門檻或使用評估段
反向調參。正式 EKF 保持關閉。

## 驗證及仍存在的失敗

Ruff、既有格式範圍、mypy、維護性與 module ownership 檢查通過。
完整離線 pytest：**2,652 passed、4 failed、2 skipped**，coverage **59.44%**。
最後的定位、去畸變、尺度、重定位永久記錄、adapter、工具與收集契約回歸為
**281 passed**。另一路 UI 繪圖、live backend 安全及記錄回歸為 **315 passed**；
兩組有重疊，不與完整測試數相加。

4 個失敗都是既有私有航線驗收：八點路線的航跡偏差集合、一個 0.5 m 陣風恢復
案例、六點路線的兩個延遲／噪聲起始案例。以開始時的 AUTO 快照重跑，三個
個別案例仍失敗；八點路線第一個起點也重現完全相同的 p95 0.06979285 u 與最大
0.09556871 u。這輪未修改該控制律，沒有放寬驗收、刪除或跳過失敗案例。
因此整個專案尚未全部驗收通過。

原始輸出：[本輪證據目錄](../outputs/analysis/system_optimization_20260922/)。
其中 `full_tests.txt`、`baseline_route_tests.txt`、`baseline_preset_failure.json`、
`camera_audit.json`、`p172_profile.json`、`p174_profile.json`、
`synthetic_ekf_matrix.json` 與 `ekf_real_replay/summary.json` 保存上述結果。

## 重跑

```bash
.venv/bin/python tools/audit_camera_calibration.py \
  --site 地圖檔/場域/river_site/site_profile.json --out outputs/analysis/camera_audit.json

# 僅對來源確認為原始 ANAFI 的新錄影使用此選項；省略則保留基準輸入。
.venv/bin/python 定位演算法/validation/pose_estimation_experiment.py capture \
  --video /path/to/raw_anafi.mp4 --raw-anafi-intrinsics intrinsics.json \
  --out outputs/analysis/anafi_rectified_capture

.venv/bin/python tools/report_relocalization.py \
  --capture outputs/analysis/anafi_rectified_capture/capture.json \
  --out outputs/analysis/anafi_reloc_profile.json

.venv/bin/python tools/replay_telemetry_ekf.py \
  --session /path/to/recorded_session --out outputs/analysis/telemetry_ekf_replay
```

這些指令不連接飛機。後續仍需要實測尺度、可配對的原始 ANAFI 影像、獨立評估
資料及操作員執行的現場驗收，才能決定是否將候選接入正式定位／控制。

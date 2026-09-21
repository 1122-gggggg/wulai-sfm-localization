# 2026-09-21 專案與最近實飛複查

本次保留工作區原有的 4 個未提交修改，對照飛行日誌、場次保存的原始碼雜湊與
目前程式。所有驗證均為離線測試或模擬，沒有連線操作飛機。

## 實飛紀錄：最新場次與前一場要分開看

以下時間均為台灣時間 2026-09-18。到站距離直接由相同 raw-map 座標系中的
`auto_route_plan.waypoints_u` 與 tick pose 計算；plan 已保存 controller 座標，
不能再做一次 aligned-to-map 轉換。航點編號從 1 起算。

| 場次／AUTO 段 | 紀錄中的行為 | 可支持的結論 |
| --- | --- | --- |
| 最新 `d7d826b3`，17:28:20–17:28:32 | 約 12 秒都在 WP1 轉向；最近 0.47347 u，半徑 0.116921 u；搖桿接管 | 尚未進入近點收斂階段，不能據此判定在到站圈外漂移 |
| 最新 `d7d826b3`，17:29:06–17:29:22 | WP1 於 17:29:08 到站，0.112452 u；WP2 於 17:29:17 到站，0.088899 u；前往 WP3 途中搖桿接管 | 此段已通過前兩點，沒有在 WP3 附近長時間停滯的證據 |
| 前一場 `7dd3f9f7`，WP3 17:25:26–17:26:02 | 約 35.9 秒未到站；最近 0.05629 u，半徑 0.053223 u；最後由遙控器降落接管 | 確實重現「接近航點但未進圈」 |

前一場 WP3 的觀察重點：

- 在「新鮮、標記地圖確認、距離小於兩倍到站半徑」的樣本中，累計約 11.05 秒；
  3D 誤差平均 0.0884 u、中位數 0.0933 u。
- 上述樣本地速平均 0.588 m/s、p95 0.810 m/s；最靠近時仍為 0.740 m/s，
  命令為 `[roll=-6, pitch=-10, yaw=0, gaz=-5]`，不是穩定停在點上。
- 「地圖確認」是程式品質標記，不是獨立真值；無法排除定位誤差。
- 沒有風速／風向量測；`wind_state=ok` 不能代表無風，也不能量化風力。

因此問題應描述為 **近點收斂不足，風是可能因素但未證實**，也要分別檢查
定位品質、轉向時間與接管事件，不能把全部未完成都歸因於風吹。

完整時間線與統計見
[`log_analysis.json`](../outputs/analysis/review_20260921/log_analysis.json)；
可重跑的唯讀分析程式見
[`analyze_flight_logs.py`](../outputs/analysis/review_20260921/analyze_flight_logs.py)。

## 既有修改是否已在當次實飛生效

**不能用「未提交」推定「飛行後才改」。** 最新 session manifest 的原始碼
SHA-256 與工作檔／本次修改前快照比對後，確認下列既有修改當次已在程式中：

| 項目 | 核對結果 | 剩餘限制 |
| --- | --- | --- |
| 弱定位繼續巡航 | `path_follow_flight.py` 場次 SHA `6614c89b…` 與本次開始時工作檔相同 | 桌面 `accept_weak_poses=True` 會關閉 weak gate，因此新增的「距到站圈 4 倍內只接受強定位」並未在桌面 AUTO 強制生效 |
| 轉頭時沿航段緩慢前進、近點鎖 yaw | 控制器場次 SHA `369c9ed9…` 與本次修改前快照相同，已包含這些功能 | 航段上的 crawl 與首次 direct-to-point 靠攏不同；不能把所有水平抵風輸出都當成 crawl，也不能宣稱它已解決近點漂移 |
| 高度／氣壓交叉檢查 | 當次有 `vertical_guard` 記錄 | 最近三場均沒有觸發 `vertical_waived`；最新場次氣壓高度變化未達 1 m 門檻，沒有實飛證據證明此 guard 已解決高度問題 |
| 比例地速限制與超速煞車 | `operator_autonomy.py` 與 backend 雜湊均與當次相同；546 個導航 tick 有 265 次 requested／sent PCMD 不同 | 最新最大地速仍約 0.760 m/s，高於設定 0.6 m/s；這是命令限制器，不是瞬時地速的硬上限 |
| 本次抵風積分與過期地速修正 | 尚未出現在 9/18 的場次原始碼 | 僅完成離線驗證 |

桌面弱定位行為是現有明確設定，本次沒有擅自改成近點停飛。需要先確認
「保持弱定位續飛」與「近點必須強地圖確認」的預期，再設計相應驗證。

## 本次修正：持續擾動下無法進入到站圈

`YawAlignedPcmdController._position_integral` 原本即使定位與地速持續有效，也會
以 2 秒時間常數衰減累積修正。這會使抵抗固定擾動的能力在非零位置誤差處達到
平衡，仍可能停在到站圈外。原有測試只檢查輸出方向、上限和短時間累積，沒有
驗證持續擾動下能否進圈並穩定停留。

修正內容：

- 有效位置觀測及新鮮地速下保留累積修正；只有新的位置時間戳能增加積分。
- 無位置觀測、位置過期或地速不可用時，仍使用原有衰減。
- 補上地速 tuple 存在但已過期時漏走衰減的分支。
- 保留積分上限、飽和時禁止同向累積、反向誤差卸載，以及原有 PCMD 上限。

修改在 [real_path_follow_controller.py](../定位演算法/flight_control/real_path_follow_controller.py)，
新增 [test_sustained_wind_control.py](../tests/localization/flight_control/test_sustained_wind_control.py)。

### 受控離線反例

使用最新場次的小到站半徑 `0.051909 u`、減速距離 `0.4322554798454335 u`，
固定世界座標方向的水平擾動、0.4 秒一階響應及整數 PCMD。尺度假設為
15 m/u，命令響應假設為 0.018 m/s/PCMD，測試持續 60 秒，另驗證兩個機頭方向。

這些是假設的控制器測試條件，**不是實測風速、場域尺度或經辨識的 ANAFI 模型**。
下表驗證一種仍存在的失敗機制，不能據此斷言當次實飛完全由風造成。

| 固定水平擾動 | 修正前最後 10 秒最大位置誤差 | 修正後 | 修正後首次進圈 |
| --- | ---: | ---: | ---: |
| 逆向 0.3 m/s | 0.05905 u，60 秒內未進圈 | 0.01349 u | 5.00 秒 |
| 逆向 0.4 m/s | 0.08042 u，60 秒內未進圈 | 0.01441 u | 5.55 秒 |
| 順向 0.3 m/s | 0.05903 u，進圈後再漂出 | 0.01227 u | 2.25 秒 |
| 無擾動 | 0.00096 u | 0.00983 u | 3.40 秒 |

無擾動案例修正後的殘餘振盪較大，但仍在到站圈內，不能把所有指標都描述為改善。
詳細輸出與版本雜湊保存在
[`servo_comparison.json`](../outputs/analysis/review_20260921/servo_comparison.json)。

### 完整航線的改善與未解問題

使用最新場次 SHA-256 對應的
`flight_route_20260913_145051_ec20651d.json`、相同 seed 7、15 m/u、600 秒預算，
以正式控制核心及桌面地速限制器跑離線模擬：

| 情境 | 修改前 | 修改後 |
| --- | --- | --- |
| 無風擾動 | 219.6 秒完成，7/7 目標 | 217.8 秒完成，7/7 目標 |
| 固定水平地面漂移 0.3 m/s | 600 秒仍卡在 40%，2/7 目標 | 通過 6/7 目標，但終點尚未完成確認 |

後者仍是 **失敗情境**。紀錄中的 `progress_pct=100` 不代表完成；必須同時檢查
`success` 與 `waypoints_reached`。把模擬時長調成 900 秒也不會繞過正式的
600 秒任務預算。修正後最後停在 `action=LAND`、`target_idx=6`，地速約
0.229 m/s，未達降落過渡所需的 0.10 m/s；等待期間零 PCMD，在這個固定擾動模型
下無法消除漂移。這是仍需另外處理的降落前穩定問題，不能以放寬速度門檻冒充修正。
此次沒有更改任務預算或終點／降落條件，遵守
[SAFETY.md](../控制介面程式/SAFETY.md) 與
[定位演算法/AGENTS.md](../定位演算法/AGENTS.md) 的降落控制變更邊界。
無風下路徑長度從 51.3 m 增為 53.2 m；較強持續擾動下亦存在繞行與終點收斂問題，
所以目前只可宣稱修復一種抵風能力不足的機制，不能宣稱整條航線已能穩定抗風完成。

原始結果位於 `outputs/analysis/review_20260921/{baseline,candidate}_{nowind,drift03}_600/`。

## PnP 後的 pose-only refinement

目前已經有做，不能把「新增 refinement」當作尚未採用的優化：

- 快迴路 [two_rate_tracker.py](../定位演算法/deploy_code/sfm_direct_deploy/two_rate_tracker.py)
  使用 `pycolmap.estimate_and_refine_absolute_pose`；正式場域 profile 的
  `fast_loop.pnp.refine_iters` 為 **2**，設定的是最大迭代次數。
- 本機 pycolmap 4.0.4 的 refinement 預設為 `refine_focal_length=False`、
  `refine_extra_params=False`；目前快迴路沒有覆寫它們，內參固定。
- 慢速重定位的 `official_edm_adapter_loo._estimate_absolute_pose` 也使用同一個
  estimate-and-refine API；此處沒有覆寫 refinement 迭代設定。

可以測試 2、3、5 次迭代的配對 A/B，固定影片、輸入匹配、RANSAC seed、內參與
地圖。這是候選實驗，尚未有量測支持「品質提升且速度相近」。驗收應同時看
PnP 與整體迴圈的 p50/p95 耗時、定位成功／失鎖率、位置／yaw 跳變及高度一致性；
不能只看參與優化的同一批點之重投影誤差。錯誤對應或缺乏幾何約束時，追加同一
目標函數的迭代仍可能收斂到錯誤姿態。

API 行為另與 [COLMAP 官方文件](https://colmap.github.io/legacy/4.0/pycolmap/pycolmap.html)
交叉確認。本次沒有更改定位 profile 或把更多 refinement 疊上正式流程。
EKF、5～10 幀滑動視窗、正確速度殘差與實驗順序詳見
[定位方法比較](pose_estimation_options_20260921.md)。

最新場次 `localization.jsonl` 的 1,891 筆 `pnp_ms` 中位數為 1.471 ms，p95 為
2.876 ms，已包含現有 refinement 所在的快迴路階段；這不是 refinement 單獨耗時。
增加迭代的額外成本與收益尚未量測，不能直接用這個數字推算。

## 專案其他優化優先順序

1. **修正飛行報告的距離語意。**
   `autoflight_error_report.py` 原先把到整條航線的 cross-track distance 與
   航點半徑比較，產生 `within_arrival_sphere_share`。貼著航線但離目標很遠也會
   顯示 100%，不能拿來判斷是否進入到站球。本次已改用 `target_distance_u` 與
   `target_radius_u`，支援逐點半徑；舊紀錄從可核對的 pose／target 還原，資料不足
   回傳 `None`。`closest_approach_u` 也改成目標距離。
   最新場次第二段的圈內樣本比例從錯誤的 1.0 改為 0.0；這不表示沒有到站，因為
   到站當幀會切換到下一個目標。真正通過幾點仍以目標切換與該時刻位置核對。
2. **讓不完整日誌可見。**
   [flight_debug_bundle.py](../tools/flight_debug_bundle.py) 與
   [autoflight_error_report.py](../tools/autoflight_error_report.py) 讀 JSONL 時會跳過
   壞行，缺少損壞行數與來源位置提示。建議報告加入 incomplete 狀態與壞行統計，
   避免意外斷電後少了最後事件卻被當成完整紀錄。本次未更改此行為。
3. **融合資料使用各自的時間戳。**
   [localization_metrics.py](../控制介面程式/operator_interface/localization_metrics.py)
   保存 attitude、speed、altitude 的不同時間戳，但
   [live_worker_clients.py](../控制介面程式/operator_interface/live_worker_clients.py)
   的 SFM3/SFM4 發送路徑使用統一的 `fused_telemetry_mono`。建議逐欄檢查新鮮度，
   過期速度／高度不得因 attitude 新鮮而視為同步。現行 direct tracker 主要取 yaw，
   因此尚無證據把此次漂移歸因於這一項；本次未擴大修改融合流程。

以上是針對目前定位、控制、日誌與驗證鏈的優先項目，不代表每個模組均已做完整
形式驗證。既有總帳已否決的 GPU／模型優化沒有重新啟用；PnP 迭代調整仍屬候選。

## 驗證

- 相同的新增回歸測試載入修正前原始碼：9 failed、3 passed。
- 修正後新增回歸：12 passed。
- 飛控、安全閘門、路徑引導、yaw/高度、桌面 AUTO 與離線模擬：486 passed、1 skipped。
  跳過原因是 URAI 場域對齊檔不存在，並非實飛通過。
- 報告工具的相關回歸：19 passed；兩組合計 505 passed、1 skipped。
- `ruff`（修改的核心與新測試）及 `git diff --check` 通過。

本次主要回歸命令：

```bash
.venv/bin/python -m pytest -q \
  tests/localization/flight_control/test_sustained_wind_control.py \
  tests/localization/flight_control/test_flight_safety_gates.py \
  tests/localization/flight_control/test_route_segment_guidance.py \
  tests/localization/flight_control/test_yaw_vertical_control.py \
  tests/control_interface/operator_interface/test_operator_autonomy.py \
  tests/tools/test_sim_route_autoflight.py
```

模擬通過只能證明指定擾動模型下的程式行為。定位誤差、地圖高度可觀測性、
實機動態與戶外風場仍需操作員自行實飛取得新紀錄確認。

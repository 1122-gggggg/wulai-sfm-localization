# PnP refinement、EKF 與滑動視窗：目前專案的可行性

日期：2026-09-21。此文件保留初始程式與文獻調查。
後續已完成兩段影片的 refinement、純視覺 ESKF 與視窗原型比較，最新數據與結論
以 [影片實驗結果](pose_estimation_video_results_20260921.md) 為準；正式定位來源未改動。
飛行問題及已實作修正另見 [飛行複查](flight_review_20260921.md)。

## 結論與建議順序

三種方法處理不同層次，不能只以軌跡看起來平滑來選擇：

| 方法 | 主要可能收益 | 目前專案狀態 | 成本與限制 |
| --- | --- | --- | --- |
| 單幀 pose-only refinement | 修正同一幀的非線性重投影誤差 | 快迴路已使用，最多 2 次迭代 | 最小實驗是 2／3／5 次；新增同一種 refinement 可能只是重複運算 |
| KF／EKF／ESKF 融合 | 時間上的狀態預測、按可信度融合量測、短時掉點銜接 | 現行 DIRECT 沒有完整 ESEKF；歷史版本已退役 | 狀態小時成本較容易控制，但尺度、座標、時序與協方差錯誤會導致過度信任錯定位 |
| 5～10 幀 sliding-window optimization | 重線性化近期多幀狀態、結合地圖錨點與相對運動 | 有 KLT 與 2D–3D 點集，可作前端基礎；目前沒有此後端 | 更有多幀幾何的改善空間，但增加求解、狀態管理與延遲；尚無本專案速度／精度量測 |

建議先做 refinement 的小型配對 A/B，同時準備 **5 幀、只記錄結果、不供飛控使用**
的視窗原型。要納入 NED 速度，先完成公尺尺度、座標與時間戳驗證。10 幀與完整
慣性融合留待 5 幀版本證明有收益後再比較。這是實驗順序，不是已驗證的效能排名。

任何濾波／最佳化都不能替代抵風控制器：若姿態已正確，但抵風命令不足，
仍會漂出到站圈。反過來，強行平滑錯誤姿態也不會讓飛行更準。

## 1. PnP 現在已做 refinement

[two_rate_tracker.py](../定位演算法/deploy_code/sfm_direct_deploy/two_rate_tracker.py)
使用 `pycolmap.estimate_and_refine_absolute_pose`。場域 profile 的
`fast_loop.pnp.refine_iters=2`，相機焦距及其他內參固定。慢速 relocalizer 也使用
帶 refinement 的 API。這與 [COLMAP 官方 API](https://colmap.github.io/legacy/4.0/pycolmap/pycolmap.html)
描述一致。

最新場次的 `pnp_ms`：1,891 筆，中位數 1.471 ms、p95 2.876 ms。這包含現有
refinement 所在的階段，並非 refinement 單獨耗時。不能據此保證把迭代次數
增加一倍只會多花某個固定時間。

增加迭代可能改善尚未收斂的好匹配，但無法增加缺失的幾何資訊，也不會自行
辨識重複場景中的錯誤匹配。應同時比較位置／yaw 跳變、失鎖率與 p95 延遲，
不能只比較參與優化的同一批內點之重投影誤差。

## 2. EKF 可以做什麼，現在缺什麼

這裡按 Extended Kalman Filter 理解。若處理完整旋轉狀態，error-state formulation
是可考慮的設計；旋轉誤差與狀態更新需要一致的座標約定。
[Solà 的 ESKF 推導](https://arxiv.org/abs/1711.02508) 說明了這類狀態表示。

現行輸入是飛控已融合的姿態／NED 速度／高度，不是原始加速度與角速度。
因此可研究的是「視覺姿態與飛控估計的鬆耦合融合」，不能直接稱作 raw-IMU VIO。
例如 [OpenVINS](https://docs.openvins.com/) 的視覺慣性 EKF 還包含慣性訊號處理、
相機與 IMU 外參及時間差校正，這些前提不能從目前 Olympe 遙測中憑空取得。

主要前提：

1. **尺度與座標。** map units 與 m/s 不可直接相加；map 的重力對齊也不等於已
   驗證的 NED-to-map 航向對齊。需要尺度及旋轉，並處理相機中心與機身參考點差異。
2. **不同量測的真實時間戳。** 視覺有曝光／擷取到輸出的延遲，姿態、速度及高度
   也各有時間戳。更新須在量測時刻處理，再預測到控制時刻；不能把舊量測改標成新資料。
3. **可信度與相關性。** PnP、視覺外推、IMU_BRIDGE 可能共享同一個原始地圖錨點，
   不能當成三份獨立觀測。飛控姿態與速度也來自已融合系統。需要保守量測噪聲、
   創新量檢查及失鎖時不確定度增長。

[robot_localization 的官方融合指南](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/configuring_robot_localization.rst)
也明確討論了重複資訊與不合適協方差造成的問題。

特別注意：profile 的 `map_scale=2.344447488498555` **不是 m/map-unit**。
發布包 `provenance/source_manifest.json` 將它定義為註冊相機中心距離其中位中心
之 p95 的兩倍，是地圖空間分布統計；它不能代入下文速度因子的公尺尺度 `s`。
既有 [map_scale.py](../定位演算法/flight_control/map_scale.py) 可以離線估算公尺尺度，
但估算結果、穩定性與 frame alignment 仍須驗證，不能把欄位同名當作同一種量。

若先只做地圖單位的「位置＋速度」常速模型，可用較簡單的 KF 做對照，暫不引入
NED 平移。它能改善部分抖動，但沒有新的獨立運動量測，長時間失鎖仍會漂移；
不能把平滑後軌跡當成更精準的證據。

## 3. 滑動視窗公式需要補上的部分

你的核心方向合理：同時使用地圖定位與跨幀運動資訊。固定視窗內保留近期狀態，
把較舊狀態邊際化，可以限制問題規模。
[GTSAM fixed-lag smoother 文件](https://borglab.github.io/gtsam/fixedlagsmoother/) 說明了此架構。

### 只有固定地圖點的 map term，仍然是逐幀獨立問題

令 `T_t` 是 world-to-camera，地圖點 `X_i` 與相機內參固定。則：

\[
E_{map}=\sum_t E_t(T_t)
\]

不同姿態之間沒有共同的未知量或跨幀因子，因此把 5 幀放進同一個 solver，
數學上仍然可以分解成 5 個獨立 PnP refinement。**改善來自有效的跨幀耦合，
不是來自視窗長度本身。** 這是由使用者提供的目標函數直接推導。

### KLT 對應需要一個明確的跨幀幾何模型

`x_i^t ↔ x_i^{t+1}` 是影像觀測，並不直接等於具尺度的 3D 位移。可研究的做法是
為局部追蹤點保留共享深度／逆深度，或先建構有不確定度的相對姿態因子，再連結
`T_t` 與 `T_{t+1}`。深度與相對位移在低視差／純旋轉時可能無法可靠估計。

若 `E_flow` 只是把目前 PnP 已使用的 KLT 像素再次寫成同一個重投影誤差，
就沒有增加獨立資訊。若使用相鄰兩幀殘差之差作耦合，也要處理它與原始重投影
殘差的相關性，避免重複加權。地圖匹配、視覺追蹤與飛控速度的噪聲不能只靠
兩個任意 `lambda` 當成同量綱資料。

### NED 速度項要轉成相同單位與座標

若 `p_t^M` 為 map 中的機身位置，`s` 為 **m/map-unit**，`R_{MN}` 為 NED 向量
轉到 map 的旋轉，簡化殘差應是：

\[
r_{v,t}=(p^M_{t+1}-p^M_t)
-\frac{1}{s}R_{MN}\frac{v_t^{NED}+v_{t+1}^{NED}}{2}\Delta t
\]

若狀態用的是目前 PnP 的相機中心，還必須處理機身到相機的位移及旋轉造成的
參考點運動；上式不能直接照搬。PnP 回傳的平移 `t` 也不是地圖中的相機中心，
必須用 `C=-R^T t`。速度要按各自時間戳對齊到影格區間，缺失或過期時不加此因子。

這個因子約束的是**量到的地面位移一致性**，不只是「運動要平滑」。有陣風時
真實速度可以變化，不能把這種變化一律懲罰掉；速度也不等於風速。

### 需要歷史先驗與魯棒加權

完整候選應包含邊際化先驗，並用各因子的協方差白化殘差：

\[
\min_{T_{t-N+1:t},\;d}
E_{prior}+\sum\rho(\|r_{map}\|_{\Sigma_{map}^{-1}}^2)
+\sum\rho(\|r_{track}\|_{\Sigma_{track}^{-1}}^2)
+\sum\rho(\|r_v\|_{\Sigma_v^{-1}}^2)
\]

其中 `d` 僅在使用局部追蹤點深度時加入；固定地圖不必一起重建。較舊狀態被移出
視窗時，需保留其資訊或清楚定義較簡單的固定先驗近似。發生 reset／reseed 時，
舊先驗與追蹤身份也必須相容。

[VINS-Mono](https://arxiv.org/abs/1708.03852) 是多幀特徵與慣性資訊聯合最佳化的
參考，但它使用預積分的原始 IMU，不代表本專案可直接用 NED 速度替換整套因子。

## 4. 能否維持接近的速度

有可能，但尚未驗證。5～10 個 pose 約為 30～60 個姿態自由度，實際成本還取決於
追蹤點數、深度未知量、魯棒核、邊際化、迭代次數與 Python/C++ 邊界。
單幀 PnP 的 1.47 ms 不能直接乘上 N 來預測視窗求解時間。

可先用 5 幀、有限且空間分布良好的 track、warm start、少量迭代與求解時間上限。
純研究原型在旁路執行，記錄結果，不阻塞原始定位與飛控。超時、重置或資料版本
過期的結果應丟棄，不把佇列堆積或修改輸出時間戳當作「維持更新率」。

維持最近 N 幀不等於每次必須等待 N 個新影格；啟動後可逐幀更新視窗。真正要
量的是最新姿態的輸出年齡，而不是把歷史軌跡事後修得很平順。輸出較平滑卻更晚，
仍可能讓靠近航點時的煞車與修正變差。

## 5. 比較實驗

保持相同的影片、逐幀匹配、時間戳及控制器，分開比較：

1. 現有 2-iteration PnP baseline。
2. 3／5-iteration 單幀 refinement。
3. 小狀態的濾波器，只使用已驗證單位的量測。
4. 5 幀視窗的 map＋有效跨幀 visual constraints。
5. 完成尺度／座標／時間驗證後，再加入 NED velocity factor；最後測 10 幀。

指標需同時包含定位成功率、錯誤重定位、跳變、短時掉點誤差、來源年齡、p50/p95
耗時，以及閉迴路到站時間、超越量與完成率。有獨立真值時才報位置／姿態精度；
單靠更低重投影誤差、更少抖動或更漂亮的軌跡不夠。

日誌中的 `pnp_observation_sample` 是低頻且最多 128 個已接受內點的抽樣，可做
refinement 初篩；它沒有完整的連續逐幀 track history 與被拒絕對應，不能直接
充當 5～10 幀視窗或完整定位 admission 的驗收資料。

## 6. 現有程式可重用的部分與最小缺口

在 [two_rate_tracker.py](../定位演算法/deploy_code/sfm_direct_deploy/two_rate_tracker.py)：

- `_live_ids` 隨 KLT 與 PnP mask 保留，正值是 map point，負值是 VO point。
  `_live_xy`／`_live_xyz` 只保存當前集合，尚無完整逐幀 track history。
- 當幀 KLT 有前後像素及 keep mask，但之後主要保留更新後陣列與品質統計。
  視窗應在資料被覆寫前保存 ID、前後像素、有效標記與 capture timestamp。
- pose 使用 `cam_from_world`，有原始 capture timestamp；灰階 ring 雖有 48 幀，
  但本身沒有每幀姿態和觀測對應，不能單靠這個 ring 還原求解視窗。
- reset 會清空狀態並重置負 VO ID 計數，strong reseed 也會替換追蹤集合。
  需要資料世代 `epoch`，並界定哪些 handover 可以延續因子、哪些必須清除舊先驗。
- `observe_fused_state()` 目前只保留姿態，沒有保存 NED 速度歷史。協定的總樣本
  150 ms 同步檢查，也不能替代 speed／attitude 各自的新鮮度檢查。

建議研究原型只增加固定長度的不可變觀測快照，包含世代、影格編號、capture time、
姿態及帶 ID 的觀測。旁路 optimizer 使用獨立的 CPU 執行工作，不塞進現有
`RelocWorker`，因為該 worker 已負責耗時的重定位，混用會干擾 handover 延遲。
輸出先只寫診斷與對照，沒有 A/B 證據前不供飛控使用。

既有 fused-yaw bridge、錄製與資料報告相關離線測試共 40 passed；它們證明的是
現有資料契約，不是 EKF 或滑動視窗已有效。

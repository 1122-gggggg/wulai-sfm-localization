# 2026-09-22 實飛問題修正與離線驗證

依同日四個飛行場次、七段 AUTO 的複查，修正高度保護、折返轉向、近點收斂、
地速限制及觀測時間／尺度累積。保留工作區原有修改，沒有操作真機。

## 修正結果

| 問題 | 修改後行為 | 驗證 |
|---|---|---|
| 高度不一致後，每到下一點又重新爬升 | 高度不一致保護保留至這段 AUTO 結束；中途點維持高度、以水平距離到站 | 下一航點及切換當幀均不恢復 gaz；終點仍用原始三維位置確認 |
| REJOIN 打斷折返轉向，未轉正就沿返程移動 | 新航段先連續完成轉向，以轉向起點的位置伺服及地速抵風；完成對準後才回航段、前進 | 165° 折返反例、完整往返航線、既有轉向與接管測試 |
| 小到站球附近反覆切換位置控制／回航段，丟掉縱向抵風補償 | 完成轉向後，FOLLOW／REJOIN 在三倍半徑內都使用完整位置伺服；四倍半徑外才離開置中 | 近點 REJOIN 的縱向積分保留、持續擾動與終點穩定測試 |
| 開始平移時突然送大傾角，延遲速度又使煞車過頭 | 新鮮遙測下水平加速以 1.5 秒建立完整額度；保留超速煞車，低速反向先保留即時阻尼，其餘漸增 | 延遲 250 ms、5 Hz 速度、三種響應時間的壓力模型；停止／接管重置、低額度量化測試 |
| 較晚遙測未配對先前影格；弱影格反覆清除尺度候選 | 保存合格視覺樣本，等待遙測覆蓋曝光時刻後配對；弱定位只可初始化 BOOT 航向一次，之後由強定位校準 | 延遲航向與弱定位啟動反例；既有跨 ±π、過期／重複時戳測試 |
| 固定四秒切片浪費短航段的尺度證據 | 以不重疊、至少 1 m 的實際位移切片；每段 1–4.5 秒，保留至少 8 段、總位移 10 m、方向和尺度一致性條件 | 合成已知尺度、實飛觀測重播；不把預測位置用於到站或降落 |

速度限制另修正「從靜止開始的第一筆速度沒有加速度基準」，以及已下煞車時仍
把舊加速度外推至額外未來時距的問題。桌面、離線模擬與 Sphinx adapter 共用
相同加速限制；Sphinx adapter 只做單元測試，未執行連線或飛行。

主要檔案：
[控制器](../定位演算法/flight_control/real_path_follow_controller.py)、
[飛行迴圈](../定位演算法/flight_control/path_follow_flight.py)、
[控制觀測估計](../定位演算法/flight_control/control_state_estimator.py)、
[桌面 AUTO](../控制介面程式/operator_interface/operator_autonomy.py)。
行為邊界同步更新於 [SAFETY.md](../控制介面程式/SAFETY.md) 與
[WIND_CONTROL.md](../定位演算法/flight_control/WIND_CONTROL.md)。

## 驗證與取捨

- 新增的 14 個針對性案例載入修改前快照：**14 failed**；載入修正後均通過。
- 飛控、轉向、高度、終點穩定、桌面 AUTO、安全及模擬工具：**672 passed、1 skipped**。
  略過原因是測試所需場域對齊檔不存在。
- Sphinx adapter 的離線測試：**11 passed**。兩組合計 **683 passed、1 skipped**。
- 六種擾動／延遲條件、三個 seed：**18／18** 完成整條往返航線並通過原有驗收。
  原有驗收包括靠攏後水平航段偏差 p95 不超過當時半徑、PCMD 上限、風事件覆蓋，
  及事件後十秒內回到半徑內並保持半秒。
- 這組配對模擬的最高地速由 **0.645 降至 0.531 m/s**；最差航段偏差 p95 為
  **0.966 倍半徑**，指定事件後最長回復時間 **2.3 秒**。
- 不是所有指標都改善。無擾動 seed 7 完成時間由 **213.0 增至 233.4 秒**；
  陣風＋延遲／掉幀 seed 7 的偏差 p95 由 **0.518 增至 0.966 倍半徑**，仍通過
  既有門檻。連續完成轉向與較平緩加速有時間及航跡取捨。
- 實飛觀測重播：原本 3,116 個導航取樣全部不啟用位置預測；修正後最新主段
  有 **38 個取樣**通過尺度條件。其他不夠長或不一致的片段仍不啟用。
  這只驗證估計器啟用條件，不是重飛後的軌跡，更不是精度改善的真值驗證。
- `ruff` 與 `git diff --check` 通過。原始飛行日誌的 SHA-256 未變；最終模擬
  receipt 中的核心原始碼 SHA-256 與交付檔案一致。

修改前的配對模擬取自既有 `wind_optimization_20260922/accepted/summary.json`，
其中六個受測來源檔雜湊均與本次開始時快照一致；未把其他版本當成基線。
模型使用 15 m/u 及指定殘餘地面漂移，這是測試假設，不是實測 ANAFI 尺度或風速。

## 尚需實測的部分

高度修正是針對觀測失效的控制保護，**不表示 EDM 高度誤差本身已被校準修復**。
若終點的原始高度仍不可信，程式不會用合成高度放行降落，仍依既有停滯／等待
預算交由操作員處理。保護以一段 AUTO 為範圍，操作員重新開始 AUTO 會重新觀測。

已知弱定位續飛策略保留；沒有以新增停止閘門冒充定位恢復。折返視角的地圖
支持、真機動態與戶外風仍需新紀錄驗證。起飛、人工降落、關閉降落、按住微移、
Esc freeze 與實體搖桿接管的操作語意未修改。

## 這批資料能否做 EKF

**可以開始鬆耦合 EKF 的離線原型與 A/B 比較。** 四個場次有視覺姿態及 capture
時間、飛控融合 NED 速度／姿態／高度及各自時間戳，並有弱定位、VO、IMU 橋接
與恢復片段。可測短暫失鎖預測、重定位跳變、時間對齊與保留觀測的預測殘差。
應分離校準與評估時段，不能拿同一批輸入姿態的平滑程度當精度真值。

四個場次均未保存 `imu_test/frames`，因此目前只能重播已記錄的定位輸出，無法
重跑完整影像定位管線。沒有 raw gyro／accel、核准外參、獨立公尺位置真值，
也不能把短段在線估計尺度當成已完成場域校準。這次沒有把 EKF 接入正式飛控；
目前的 `ControlStateEstimator` 是時間／尺度及有界位置預測，不是 EKF。

## 重現與證據

[新增回歸測試](../tests/localization/flight_control/test_flight_regressions_20260922.py)、
[完整測試紀錄](../outputs/analysis/flight_fixes_20260922/checks_final.txt)、
[修改前失敗紀錄](../outputs/analysis/flight_fixes_20260922/baseline_regressions.txt)、
[18 情境結果及來源雜湊](../outputs/analysis/flight_fixes_20260922/matrix_final/summary.json)、
[實飛觀測重播](../outputs/analysis/flight_fixes_20260922/replay_estimator.py)、
[重播結果](../outputs/analysis/flight_fixes_20260922/estimator_replay.json)。

```bash
.venv/bin/python -m pytest -q -rs \
  tests/localization/flight_control \
  tests/control_interface/operator_interface/test_operator_autonomy.py \
  tests/control_interface/operator_interface/test_operator_flight_safety.py \
  tests/tools/test_sim_route_autoflight.py tests/tools/test_verify_wind_control.py

模擬器/parrot_stimulate/.venv/bin/python -m pytest -q \
  模擬器/parrot_stimulate/tests/test_production_route.py

.venv/bin/python tools/verify_wind_control.py \
  --route 地圖檔/場域/river_site/routes/flight_route_20260913_145051_ec20651d.json \
  --align 地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260908/localization/T_align_gravity.json \
  --meters-per-unit 15 --seeds 7 11 19 \
  --out outputs/analysis/flight_fixes_20260922/matrix_final
```

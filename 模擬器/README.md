# 模擬器（實體）

三種工具保留不同責任，不互相冒充完整系統驗證：

| 工具 | 用途 | 不取代 |
|---|---|---|
| `測試影片/` + `控制介面程式/影片模擬串流/` | EDM、720p 串流與操作 UI 回歸 | Sphinx firmware／PCMD 物理響應 |
| `parrot_stimulate/` | 主要 Sphinx/Olympe PCMD、逐點 route、安全 gate、response 與固定 worst-case 驗證 | EDM 影像定位、真機安全核准 |
| `sphinx_anafi_path_convergence/` | 保留十種 convergence controller 的比較與歷史回歸 | 正式 waypoint controller 或 production readiness gate |

`parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py` 是 Python 3.10
操作介面與 Python 3.11 Sphinx route 共用的純控制核心；真機 autonomous 仍保持鎖定。
完整無飛行驗證由根目錄 `./驗證系統.sh` 一次執行兩個 Python 環境。

整合只使用 `parrot_stimulate/` 的程式、鎖檔、固定 firmware 與測試。
原 `封存/parrot_stimulate_standalone_20260803/` 已於 2026-08-05 經操作員明確
授權永久刪除；它原本不納入外層 Git，因此目前不能由此 repository 回復。
`parrot_stimulate/` 現在是工作區內唯一的正式 Sphinx/Olympe 模擬器來源。

`launch_sphinx_anafi_empty.sh`、`sphinx_path_follow_smoke.py` 為舊 harness 的相容入口，
不應再新增第三套控制公式。

## 離線影片位置

下載的模擬影片集中於 `測試影片/`，原始檔內容不變，不納入 Git：

- `P1130113.MP4`（福和橋）、`P1980198.MP4`（河濱）、`P0170017.MP4`（烏來，4K、
  199 秒、2.5 GB）、`P1990199.MP4`。
- `超過地圖範圍影片/`：`P2300230.MP4`、`P2310231.MP4`、`P2320232.MP4`。

影片選擇介面預設開啟這個目錄。沒有預設 P119 且有多部影片時，啟動器不會
任選影片；請在選擇介面選檔，或從工作區根目錄明確指定，例如：

```bash
SFM_SITE_PROFILE="$PWD/地圖檔/場域/fuhe_bridge/site_profile.json" \
VIDEO="$PWD/模擬器/測試影片/P1130113.MP4" \
./控制介面程式/影片模擬串流/啟動.sh
```

操作介面的「離線模擬」按鈕依目前選的場域自動用對應影片，不再沿用上一個場域
的影片：`地圖檔/場域/fuhe_bridge` → P1130113、`river_site`（含各 release）→
P1980198、`wulai_all9_direct_20261019` → P0170017。對照表是
`operator_launch.SITE_SIMULATION_VIDEOS`；對應影片不在 `測試影片/` 時改開檔案選擇，
不會退回其他場域的影片。未列入對照表的場域沿用原本的記憶／設定影片規則。

影片移入專案不代表與所有場域相容，也不構成定位精度或實飛核准。

## ANAFI 擬真範圍與離線情境

依據 [ANAFI white paper v1.4](https://www.parrot.com/assets/s3fs-public/2020-07/white-paper_anafi-v1.4-en.pdf)，
機體能力上限是水平 15 m/s、升降 4 m/s、角速度 200°/s、飛控迴圈 200 Hz；
320 g 是機體質量。抗風標示為持續 50 km/h、陣風 80 km/h，不代表在這些風速下
仍能維持本系統的航線精度。桌面控制設定另限 tilt 20°、升降 2 m/s、
yaw 20°/s、tilt slew 60°/s，不能把機體能力上限當作桌面命令設定。

* **韌體與物理風：** `parrot_stimulate production-route` 使用正式 AUTO 控制器、
  Sphinx ANAFI 模型及固定韌體。`--wind-mean-mps E N U` 與 `--gust-mps`
  使用 [Sphinx wind](https://developer.parrot.com/docs/sphinx/wind.html) 的風場，
  不移動真值位置。只允許連到 `10.202.0.1`；須由操作員先在 Sphinx 懸停，
  再以 `--attach-hovering` 接上，不新增起飛入口。
  已安裝 ANAFI4k／Sphinx 2.25.2 只有 `world:wind/wind`，陣風以本機
  JSON-RPC 隨 AUTO pose poll 更新實際 ENU 風向量（約 20 Hz），不使用
  該機型不存在的 drone ExprTk 元件。設定回覆 `false` 即失敗，不能只看
  `sphinx-cli` exit code。地面虛擬驗證已讀回 105 筆週期風場變化，未送飛行命令。
* **快速閉迴路：** `../tools/sim_route_autoflight.py` 使用同一正式控制器，
  200 Hz 積分、有界 tilt slew、tilt 加速度、阻力、零 PCMD 煞車及命令延遲。
  最新飛行校準情境 R5 使用水平時間常數 3 s、有效 tilt lag 0.4 s、
  水平加速度增益 1.09。預設的速度代理模型只保留給控制器回歸，不能
  當成真機模型；擬真檢查應使用 R5 或明確指定 `--tau-h-s 3 --tau-tilt-s 0.4`。
  這是降階校準模型，不是完整空氣動力或韌體 EKF 複製。一般模式預設額外
  命令傳輸延遲為零；步階曲線的有效延遲不能直接當作 PCMD 網路延遲重複計入。
  `--gust-m` 是有限脈衝的累積**殘餘漂移公尺**，不是大氣風速：
  `--gust-duration-s 2` 在兩秒內施加半正弦擾動，不再瞬移。
  瀏覽器比較 harness 的位置脈衝不能當成物理陣風驗證。
* **定位中斷：** 兩者共用 `simulated_localization.py`。視覺 blackout 時保留
  最後視覺中心，以獨立、帶時間戳的飛控融合姿態補航向。沒有原始加速度
  積分位置的假保證；IMU_BRIDGE 標記為 `map_confirmed=False`、
  `position_observed=False`，不能冒充新的位置觀測或降落證據。
  姿態樣本超過 0.5 s、tilt 超過 35°、yaw rate 超過 180°/s，或橋接超過
  10 s／300 frames，停止補位，交由原有失鎖處理；視覺恢復後重新提供定位。
  正式迴圈仍保留最多 0.5 s 的既有 pose holdover；之後不再導引航線。
  新鮮地速仍可觸發原有煞車修正，不能把煞車 PCMD 誤讀為位置補位或續航。

2026-10-03 最新有 AUTO 與定位的紀錄為
`session_20261003T091352Z_real-flight_bd7986e4`。較新的 `091729Z` session
沒有定位結果，不能拿來辨識機體動態。前者辨識得到 delay 0.5 s、
水平時間常數 3 s、gain 0.0641 m/s/%、R² 0.701；這是閉迴路融合遙測估計，
不是量測尺度或適航認證。Sphinx roll ±50% 六秒探測得到時間常數
3.01–3.09 s、delay 0.364–0.369 s；降階模型對 roll-right 命令期間
速度 RMSE 0.085 m/s。pitch、yaw、gaz 個別探測完成，但整批 response
曾在 pitch 起飛／懸停階段失敗，不能宣稱整套 response 驗證通過。

以下 preset 使用該航次的水平回應時間常數與弱定位比例，另注入陣風及每 40 秒三秒
blackout，屬於壓力情境，**不是逐時間重播實飛或實測風場**：

```bash
.venv/bin/python tools/sim_route_autoflight.py \
  --route 地圖檔/場域/river_site/routes/flight_route_20260914_141505_e87a66fb.json \
  --scenario R5_flight20261003_gust_imu --meters-per-unit 10 \
  --out outputs/sim_autoflight/latest_gust_imu --no-plot
```

`meters-per-unit` 是模擬真值假設，控制器不知道此尺度。長 blackout 可另用
`--tau-h-s 3 --tau-tilt-s 0.4 --blackout-every-s 40 --blackout-dur-s 12`
（不選 preset）測試校準模型的橋接到期；
完整結果與 `trace.csv` 必須一起看，不能只憑航線完成就稱為「完美模擬」。
白皮書未公開完整氣動參數、韌體控制律、場域光照／紋理與 EDM 視覺失效
分布；這些是目前擬真的界限，仍須由操作員安全地收集實飛校準資料。

### 已執行的驗證

證據保留於 [`outputs/sim_autoflight/anafi_fidelity_20261003/`](../outputs/sim_autoflight/anafi_fidelity_20261003/)：
R5 陣風／三秒 blackout 情境完成 11 個去回航點（272 s），十二秒 blackout 情境
完成 11 個航點（214 s），橋接到期期間沒有新增位置觀測或降落確認。
`native_gust_smoke.json` 位於 `wind_interface_probe/`，記錄地面 Sphinx 的 105 筆
真實風場讀回；`wind_failure_guard/smoke.json` 是無 SDK／網路的中途拒絕設定退出驗證。
這些不代表已在空中聯合驗證 EDM、陣風與真機定位中斷。

最新 Sphinx adapter／遙測／安全回歸為 31 passed、5 skipped；
略過案例依賴目前未提供的舊 river 航線資產，不算通過。

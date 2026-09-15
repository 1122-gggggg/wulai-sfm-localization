# DIRECT 離線修正與驗證

更新：2026-09-15。以下指令不連接飛機。歷史的 ESEKF、KLT LOST、200/200 排名與舊 Sphinx smoke 不能當作目前版本的放行證據。

## 現行行為

- RelocWorker 至多保留一筆執行中工作與一筆最新等待影格。等待影格會被更新，不累積舊觸發；reset／close 後不發布舊結果。
- 完整重新播種必須同時滿足 `fast_loop.reseed_min_points` 與 relocalizer strong inliers。現行值為 120／80；較小的 handover 只能補充既有可追蹤點集，不能繞過門檻初始化空軌跡。
- PnP 的 12 點求解門檻不等於可靠定位。少於設定的 80 個 map inliers 時，即使 direct status 是 FAST_TRACK，發布的 mode 也是 WEAK_TRACK。
- DIRECT 輸出使用 `max_jump_u`、最近可靠位置的中位數。候選超過 `max_jump_u` 時僅保留舊位置，不把估計修正加入速度歷史。控制端不再以航點半徑做來源確認暫停；1.5 map-unit 單次跳變暫停仍在。
- 現行 profile 與新 release builder 的 `reloc.period_s` 為 1.0，`dead_reckon.max_age_s` 明列 10.0，`max_jump_u` 明列 1.5。舊 profile 的相容預設仍可讀取，但新 release 不省略這些值。控制端的弱定位導航期限仍只有最後可靠地圖影格後 0.5 秒。
- IMU 是 firmware fused yaw／姿態及遙測，屬 Level D，沒有 raw gyro/accel 積分或 ESEKF。缺少尺度和相機／機身外參時不能把 NED 公尺速度混入 map 單位。
- 點雲近接懸停互鎖已依操作員指示拿掉。障礙物迴避回到操作員目視與搖桿接管；稀疏點雲不提供避障保證。

## 本機檢查

```bash
.venv/bin/python -m pytest -q --timeout=300
.venv/bin/python -m pytest -q 模擬器/sphinx_anafi_path_convergence/tests
.venv/bin/python tools/system_validation.py --smoke --development
SFM_INSTALL_OFFLINE=1 bash tools/test_clean_install.sh
```

`--development` 只允許驗證尚未提交的工作樹，不能當作 release 或飛行核准。Sphinx tests 在獨立 Python 程序檢查模擬器控制及 SDK message contract，不代表已跑 Sphinx VM 或真機。

先依固定 lock 建立 wheelhouse，之後安裝與驗證可完全不使用套件索引：

```bash
.venv/bin/python tools/offline_wheelhouse.py build \
  --output 執行環境/offline_wheelhouse \
  --requirements requirements/runtime-lock.txt \
  --requirements requirements/test-lock.txt \
  --requirements requirements/quality-lock.txt
```

已存在的 wheelhouse 應用同樣三份 lock 執行 `verify --wheelhouse`；build 拒絕覆寫。套件下載不包含任何飛機連線。

## 任務總時間

路線驗證必須明確填入初始定位等待時間，並計入原本的 300 秒總預算：

```bash
.venv/bin/python tools/verify_preset_routes.py --localization-wait-s 30 \
  --out outputs/analysis/route_budget_30s
```

等待 0 秒只代表已取得定位的控制器情境。不能將它的通過率推論為有定位等待時仍能完成任務；等待吃完預算時，模擬不得產生運動步驟。模擬尚未辨識真機動態，也不會推算 BOOT 等待中的實際風漂。

## Portable 與 CI

- 預設 portable 不含私有場域；`tools/test_portable_runtime.sh PACKAGE` 驗證其離線安裝與模組載入，明列未評估定位／GUI pose。
- 要求場域驗證時使用 `--require-site-bundle`，export 必須指定 `--site-profile` 及 `--wheelhouse-root`。缺少場域不得當作完整 UI 驗證通過。
- Hosted CI 檢查 CPU／模擬 contracts；CUDA、EDM 權重、私有場域與 portable GUI 驗證在 GPU dispatch 執行。這些範圍不等於實飛驗收。
- Sphinx firmware 使用本機 `anafi-pc.ext2.zip` 的大小及 SHA-256 作為固定來源；manifest 的 `retrieval_url` 只記錄歷史取得方式，禁止以 `#latest` 重新取得替代檔案。

## 仍需資料或實機證據

P173 地圖洞不能靠延長 VO 期限證明已修好，P174 的 99.4% 也只是覆蓋率。本工作區未提供 P173／P174 原始影片與獨立位置真值，不能以現有 mapping keyframes 重算出「獨立精度驗證」。需要補足資料，再依固定 profile、影像雜湊與真值來源檢查。

既有 session 的資料完整性可用：

```bash
.venv/bin/python tools/imu_flight_test_report.py --session SESSION --json
```

`video_fail`／`cleanup_incomplete` 的實機重現可先在地面處理；風擾、制動和完整任務動態仍須操作員實飛驗收。離線修改不授權 agent 起飛、操作飛機或把 `validation: NONE` 改成已驗證。

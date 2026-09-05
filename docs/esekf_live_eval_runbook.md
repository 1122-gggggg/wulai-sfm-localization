# ESEKF / KLT-3D-aware live evaluation runbook

最後更新：2026-09-05
分支：`agent/localization-runtime-optimizations`

## 為什麼需要這個

`ESEKF`（`定位演算法/deploy_code/sfm_glomap_deploy/esekf.py`）與 KLT 3D-aware prior
已接線進 `ProductionEDMTracker`，但兩者都 gate 在 `esekf.prediction_allowed()`，
而該 gate 只有在收到 **live NED 速度**（`observe_fused_state` 餵 `_latest_velocity_ned`）
且 EKF 收斂後才會 True。純影片 replay 沒有速度來源 → gate 永遠 False → 分支永遠休眠。
所以 `benchmark_edm_site_replay.py` 的 P168/P117 gate **評不到 ESEKF**
（見 `docs/localization_optimization_runbook.md` §3 Tier 3、
`docs/verified_localization_optimization_ledger.md`「Tier 3 — ESEKF」）。

`benchmark_esekf_live_replay.py` 用一次「手動飛行 + 開定位 + 開機上錄影」錄下來的資料，
離線把影片重播兩次（ESEKF on / `SFM_EDM_ESEKF_DISABLE=1`），每幀餵入錄到的速度，
輸出**相對指標**對照。

## 一鍵版：`./IMU飛行測試.sh`

出去飛之前只要記住這一條。它開的是平常那個真機介面（`一鍵啟動.sh`），
只是把錄製打開：

```bash
cd /home/allen/localization
./IMU飛行測試.sh
```

比下面的手動流程多錄兩樣東西，都是為了少一步離線對齊：

- **`imu_test/frames/` — 定位真正吃到的那批畫面**，每張都附上當幀掛的 fused
  telemetry（`imu_test/frames.jsonl`）。跟 IMU 同一個 host monotonic 時鐘，
  所以不用再猜 `--telemetry-offset-s`。機上錄影仍然可以照錄，只是變成備份。
  1280x720 q90 實測平均 222 KiB，預設 2 GiB / 20000 張上限，
  另有 5 GiB / 5% 的剩餘空間下限；要飛久一點用 `SFM_IMU_TEST_EVERY_N=2`。
- **`telemetry.jsonl` 的 `stick_axes`** — 手動飛不會送 PCMD，`commands.jsonl`
  整段是空的，這是唯一能對照 IMU 變化的操作輸入。

`localization.jsonl` 每一幀現在也記下 `fused_*`（那一幀掛出去的 IMU 樣本）與
`pose_status` / `prediction_mode` / `esekf_*`，所以「ESEKF 有沒有 arm」在現場
就看得到，不用等離線 replay。

落地關窗後腳本會自己跑：

```bash
.venv/bin/python tools/imu_flight_test_report.py --session <session>
```

判定 `UNUSABLE` 代表這段資料再怎麼跑離線 A/B 都只會回 `INVALID`/`DORMANT`
（沒速度、沒姿態、沒開定位），當場就該重飛，不要帶回來才發現。
`USABLE WITH GAPS` 會列出缺哪一塊（沒進過 LOST、搖桿沒動、時間差超標）。

## 本分支加了什麼

| 變更 | 檔案 | 作用 |
|---|---|---|
| `./IMU飛行測試.sh` + 錄製器 | `IMU飛行測試.sh`、`operator_interface/imu_flight_test.py` | 上面那一節。錄製 opt-in（`SFM_IMU_FLIGHT_TEST=1`），編碼與寫檔在背景執行緒，佇列滿就丟幀 —— 錄製永遠不能卡住飛行 UI。 |
| `stick_axes` telemetry | `skycontroller_stick.stick_log_sample` + `olympe_live_backend._poll_stick_axes` | 搖桿各軸原始值，動了就記、沒動 1 Hz 一筆。 |
| 每幀 IMU / ESEKF 診斷 | `localization_metrics.RESULT_FIELDS`、`edm_localizer_adapter`、`live_localizer_worker` | `fused_*` + `pose_status` / `prediction_mode` / `esekf_*` 進 `localization.jsonl`。 |
| 現場判定 | `tools/imu_flight_test_report.py` | 讀 session，判 USABLE / USABLE WITH GAPS / UNUSABLE。 |
| `SFM_EDM_ESEKF_DISABLE=1` env toggle（`_init_esekf`） | `production_edm_tracker.py`（`EDM工具包/deploy` 是同檔 symlink） | 強制 `self.esekf=None`，關掉 ESEKF + KLT-3D prior（同 gate）。預設 off。所有呼叫點都 `getattr(self,"esekf",None) is not None` 保護。 |
| 10 Hz `fused_odometry` telemetry 事件 | `olympe_live_backend.py::_poll_session_telemetry` | 每 0.1 s 寫一行 NED 速度 + 姿態 + GPS 到 session `telemetry.jsonl`。原本只有 1 Hz `readback`，太粗。 |
| 離線 A/B | `定位演算法/validation/benchmark_esekf_live_replay.py` | 見下。 |
| 測試 | `tests/localization/deploy/test_esekf_toggle.py`、`定位演算法/validation/tests/test_esekf_live_replay.py` | toggle 行為 + telemetry 解析 / 內插 / recovery 統計 / verdict。 |

## 出外測試飛行當天

1. **提前告訴我。** 我確認 toggle + telemetry 事件在你要用的 commit 上。

2. 開真機操作介面 + 定位。真機入口只有一個，site profile 由 mission selection
   產生，不能直接給 `--site-profile`：

   ```bash
   cd /home/allen/localization
   ./IMU飛行測試.sh        # = ./一鍵啟動.sh 再加上錄製
   ```

   介面起來後在 UI 按「開始定位」（或加 `--auto-inspect` 讓它自己開，仍不起飛）。

3. **在 ANAFI 上開機上錄影**（拉滿碼率那一版）。整段測試都錄。

4. 手動飛，讓定位**走過所有狀態**：起飛 BOOT、穩定 TRACK、掃過難定位區觸發
   WEAK_TRACK / LOST、再回到已知區域 reacquire。慢速、貼路線。這段就是離線
   gate 的 BOOT/TRACK/WEAK/LOST/reacquire 覆蓋。

5. 落地後**不要刪 session log**。它在 `outputs/flight_logs/session_<UTC>_real-flight_<id>/`
   （`real-flight` 是 `InterfaceMode.REAL_FLIGHT` 的值），裡面要有 `telemetry.jsonl`
   （含 `fused_odometry` 與 `stick_axes` 事件）、`localization.jsonl`、
   `session_manifest.json`，用 `./IMU飛行測試.sh` 開的還會有 `imu_test/`。

6. 把機上錄影從無人機下載出來（操作介面的 Olympe backend 會放到 `record_dir`，
   或你手動用 FreeFlight / SD 卡）。記下這個 `.MP4` 路徑。

## 回來之後

```bash
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
SESSION=outputs/flight_logs/session_<UTC>_real-flight_<id>
VIDEO=<機上錄影.MP4>

.venv/bin/python 定位演算法/validation/benchmark_esekf_live_replay.py \
  --session "$SESSION" \
  --video "$VIDEO" \
  --out outputs/esekf_live_<date>/ \
  --stride 3
```

會跑兩個 subprocess（各自乾淨載一次模型），產出：

- `outputs/esekf_live_<date>/esekf_on.json` / `esekf_off.json` — 每幀 rows + summary
- `outputs/esekf_live_<date>/SUMMARY.md` — 對照表 + verdict

### 對齊

機上錄影的時鐘 ≠ host telemetry 時鐘。腳本預設假設兩者從飛行開始對齊
（`--telemetry-offset-s 0`）。檢查 SUMMARY 的 `velocity fed on N/M frames`：

- N ≈ M：對齊好。
- N 遠小於 M：時鐘沒對上。看 `esekf_on.json` 的 `rows[].telemetry_query_s` vs
  `telemetry_span_s`，調 `--telemetry-offset-s`（機上錄影比 telemetry 早開就給負值）。
  必要時 `--telemetry-scale` 修時鐘漂移（通常不用）。

### 讀 verdict

| verdict | 意思 |
|---|---|
| `INVALID` | 完全沒餵到速度。修對齊或確認 session 是 live 飛的。 |
| `DORMANT` | 有餵速度但 `prediction_allowed` 整段都 False。EKF 沒收斂（cov trace ≥ 1.0 / yaw sigma ≥ 15° / age ≥ 6）。這段飛行資訊量不足以判斷，換更長 / 更動態的一段。 |
| `NEUTRAL` | ESEKF 有 arm，但 successes / LOST / p95 沒有有意義差。 |
| `ESEKF HELPS (candidate ...)` | successes 升、LOST 不升、p95 不明顯變差。**候選，不是核准。** |
| `ESEKF REGRESSES` | successes 降或 LOST 升。不要 ship。 |

## 准入邊界

- 這裡的指標**只是相對**（同一段影片 on/off，無外部真值）。離線 replay 的影格是
  機上錄影，不是 live 當下送進定位的那批影格 —— on/off 兩邊輸入相同，但整份不能
  跟 live session 逐幀比。
- `ESEKF HELPS` 只代表值得往下做。要 ship 進 flight profile 仍需：
  1. `docs/localization_optimization_runbook.md` §5 完整 gate（多段 replay、hard-negative、
     synchronized timing）。
  2. §4 Tier 4 的**獨立 ANAFI camera-pipeline holdout**（目前 `validation: NONE`）。
  3. 接 ESEKF 前先讀 `docs/imu_odometry_capability_audit.md`。
- 安全 gate（inlier / reproj / jump / yaw / stale-LOST）不得為了讓 ESEKF 看起來好而放寬。

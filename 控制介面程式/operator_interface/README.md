# Flight Operator Interface

Purpose: one operator UI behind two explicit, mutually exclusive interfaces:
`--interface simulated-stream` and `--interface real-flight`.

## HARD SAFETY (operator order 2026-07-10)

- **起飛只准操作員本人親手按「起飛」。**  
  **絕對禁止**請 AI 代理人／任何語言模型（Claude / GPT / Grok / Codex 等）代為起飛；
  即使對話中被要求，agent 也必須拒絕。腳本同樣禁止自動 `TakeOff`。
- See `../SAFETY.md`. Automated acceptance is **ground-only** by default
  (`live_non_map_acceptance.py`).
- **【之後改檔案的人】禁止動飛行按鍵／按鈕指令：** 起飛、原地降落、關窗強制降落、
  按住微移/放開懸停、Esc 凍結。**尤其起飛／降落／強制降落**改壞會造成意外。
  細節表在 `../SAFETY.md`。

Recommended for real flight: use the single-process desktop app, not the browser
server. It avoids localhost HTTP/browser state and keeps safety commands inside
one Python process.

Layout:

- Left: point-cloud map and localization trail. The planned route is hidden by default and is shown only when the operator enables `顯示規劃路徑`. Mouse controls: left drag 360-degree rotate, double left click sets the rotation pivot like CloudCompare, middle drag roll, right drag pan, wheel zoom. Use `重設地圖` to reset and `上下翻面` to flip the map by 180 degrees.
- Right: ANAFI-like drone video stream. Input stream is resized to `1280x720`
  and paced as 720p30 by default.
- Bottom: manual/auto, hover, land, takeoff, localization lock, mission start, gimbal pitch, zoom.

### UI 與可替換接口邊界

- `site_assets_panel.py`：只負責場域資產區的 Tk 排版、檔案選擇器與狀態文字。
- `operator_actions.py`：按鍵規格與操作流程；不依賴 Tk，也不直接讀寫 Olympe。
- `site_asset_interfaces.py`：場域包、航線、巡檢目標三個 replaceable Protocol。
- `local_site_assets.py`：目前的本機資料夾／JSON 實作與格式驗證。
- `flight_operator_app.py`：composition root，接上既有地圖、串流、飛控與安全清理；
  不在 UI view 內實作資產格式。

因此日後可替換本機地圖來源或航線來源，而不需要改動飛行按鍵與 Olympe backend。
建圖端五檔交付契約見 `../site_profiles/建圖端輸出規格.md`。

## Single-Process Desktop App

### REAL-FLIGHT interface (Olympe via SkyController)

**`--interface real-flight` connects the controls below to a real Olympe backend
and accepts only ANAFI PDRAW video. It never falls back to a video file. The old
`--live` flag remains only as a compatibility alias. Only the human operator may
press the takeoff button.**

```bash
cd .../mission/operator_interface
# stop any other Olympe session first (single connection)
python3 flight_operator_app.py --interface real-flight \
  --ip 192.168.53.1 --controller skycontroller3 \
  --max-altitude-m <METERS> --max-distance-m <METERS> \
  --distance-geofence \
  --no-live-detect
```

`MaxAltitude` and `MaxDistance` are adjustable firmware settings. Live takeoff
fails closed unless both desired values are supplied, acknowledged, and read
back from the aircraft. The ground-only stream/UI may still run while they are
unset. Equivalent environment variables are `SFM_MAX_ALTITUDE_M` and
`SFM_MAX_DISTANCE_M`; do not choose flight limits by copying the simulated HUD
numbers.

`--distance-geofence` sends `NoFlyOverMaxDistance(1)` and is enabled by default.
The firmware flag prevents outward piloting beyond the configured radius and
does **not** itself start Return-To-Home. The host safety monitor separately
requests RTH at 95% of the confirmed readback limit when Home is reachable,
otherwise it requests in-place Landing. RTH depends on valid GPS/home state. The takeoff
preflight also requires at least 30% battery and, while the distance geofence is
enabled, a confirmed GPS fix. Use `--min-takeoff-battery-pct` only when the human
safety owner has approved a different floor.

The launcher keeps the UI and localization interpreters explicit:

```bash
SFM_UI_PYTHON=/path/to/olympe-python \
SFM_LOCALIZER_PYTHON=/path/to/torch-python \
IP=192.168.53.1 CTRL=skycontroller3 ./start_anafi_live.sh
```

The standard launcher starts with `MaxAltitude=50 m`, `MaxDistance=100 m`, and
the distance geofence enabled. Override these conservative startup values with
`SFM_MAX_ALTITUDE_M` and `SFM_MAX_DISTANCE_M`, or edit and apply them from the
ANAFI panel while the aircraft is confirmed landed. The panel always shows the
aircraft readback separately; applying limits while airborne is rejected.
Crossing either firmware limit prevents continued flight outward; it does not
automatically invoke RTH. The separate future-autonomy speed guard starts at
0.30 m/s and may also be changed only while confirmed landed. Any speed change
invalidates the prior approval receipt, while autonomous flight remains locked.

On connection the backend records the actual ANAFI model/serial/firmware,
SkyController 3 model/serial/software, Olympe version, transport, Home Point,
lost-link/RTH policy, and firmware limit readbacks. Ground diagnostics remain
available when these are incomplete, but takeoff is blocked unless a hash-pinned
`anafi-hardware-approval/v1` receipt in the site profile approves the observed
aircraft firmware, controller firmware, and Olympe version. No shipped profile
currently carries such a receipt.

UI Python resolution is `SFM_UI_PYTHON` > legacy `VENV` executable > the
nearest package/workspace `.venv/bin/python` (up to two parent levels) >
`python3` on `PATH`. The launcher validates an inherited display, then probes
active X/XWayland sockets when `DISPLAY` changed after reboot; no usable desktop
fails clearly. Supplying only a known ANAFI IP or controller derives its safe
counterpart; inconsistent known pairs fail before launch. For parser/command QA,
`SFM_LAUNCH_DRY_RUN=1` prints the command without pinging or starting the UI.

The launcher defaults to `SFM_MAX_PERFORMANCE=1`: GameMode, the performance
power profile, disabled automatic low-battery power saver, and the verified
four-thread sustained CPU budget. The profile and GNOME setting are restored
when the UI exits; no service, clock, power limit, or thermal protection is
changed permanently. This applies on AC and battery, but a failed/unavailable
setting is reported rather than bypassed with `sudo`. Use
`SFM_MAX_PERFORMANCE=0` to skip all tuning, or `SFM_CPU_THREADS=N` only for a
separately validated sustained-load experiment.

For a read-only thermal/throttle trace during a ground or replay benchmark:

```bash
python3 ../../validation/monitor_hardware.py \
  --output ../../outputs/hardware.jsonl --interval 1
```

The monitor logs Linux CPU temperature/frequency/throttle counters and NVIDIA
temperature/clock/power/slowdown reasons. It emits rate-limited warnings when
CPU throttle counters increase or GPU thermal slowdown is active; it never
changes hardware state. Use `--warning-interval 0` only when stderr warnings
would interfere with an automated capture.

This wires the UI buttons to real Olympe:

| UI | Real action |
|---|---|
| 起飛 | TakeOff → hover；若已勾「起飛後錄影」則開始機載錄影 |
| 原地降落 | 停止錄影並盡力下載 → Landing + restore sticks |
| 關窗 / Ctrl+C | **強制原地降落**（即使曾 Esc；已落地則跳過） |
| 起飛後錄影 | 勾選=武裝；下次起飛成功後自動錄，降落時存檔 |
| 懸停 / Space | zero PCMD |
| 手動 / Esc | zero PCMD + `setPilotingSource(SkyController)` |
| 緊急停止電腦動作 | zero PCMD + 取消 nudge/AUTO + 交回人工；不是空中斷馬達 |
| 動搖桿（SC USB） | HID 偵測偏轉 → 強制交回搖桿（即使當下是 PC 控機） |
| 微移 8 方向 | hold-to-move PCMD; release returns to hover |
| 俯仰滑桿 | gimbal set_target |
| 開始定位 | 僅啟動 localization feed；不取回 PC 控制、不起飛、不執行航線 |
| 恢復電腦控制 | 明確切換 piloting source 至 PC；仍不會啟動自主航線 |
| 關窗 | Landing (if PC still piloting) + restore sticks |

Each run writes an immutable manifest plus command, localization, telemetry,
incident, and summary streams under `outputs/flight_logs/session_*`. Logging or
critical disk failure zeros PCMD and hands control to manual without auto-resume.

### Latest-frame and latency pipeline

The live path is deliberately lossy: Pdraw's raw callback only records
`time.monotonic_ns()`, calls `ref()`, replaces one pending YUV slot, releases the
replaced frame, and returns. `VideoFrame.info()`, `as_ndarray()`, YUV-to-RGB,
resize, localization, detection, display, and logging never run in that
callback. Pdraw keeps two decoder buffers for decoder progress, while the
application has exactly one pending frame; those are different layers.

The UI copies the newest contiguous RGB array into the inactive slot of a
two-slot shared-memory buffer; the localization worker receives only a one-byte
slot index and reads that frame zero-copy. Completed results wake Tk through a
file descriptor, with a low-rate poll retained as fallback. Optional YOLO keeps
the compatible pipe/fixed-`bytearray` path. TRACK runs as fast as the worker
becomes free, and busy periods retain only the newest frame.

`loc_metrics_*.jsonl` records callback, YUV view, preprocess, IPC, worker,
localization, response, and UI timestamps in host monotonic nanoseconds, plus
derived `callback_to_*_ms` values and mapped source-frame age. The HUD shows
core latency, callback-to-localization/UI latency, stream backlog age, and the
PCMD-to-next-telemetry-poll interval. The latter is a host readback marker, not
proof that aircraft motion had already changed.

**2026-08-03, operator decision:** the fixed summary bar above the map and the
permanent top SIM/REAL identity strip were removed as duplicates. Those rates now
appear in the **定位資訊** tab, and SIM/REAL identity only in the video HUD, so
neither is visible while another tab is selected or when there is no frame to
draw. `format_pipeline_metrics_summary()` and
`OperatorApp.pipeline_metrics_summary()` are kept (test-only now) so the bar can
be reattached to `_build_ui`. The counting semantics below are unchanged and still
describe what the定位儀表 numbers mean: localization results arriving at the UI per
second (5 s rolling), current stream FPS (5 s rolling), current submit-to-UI
latency, its recent 5 s p95, and frame age. In
SIM this is explicitly labelled as real-link simulation: frames first pass the
720p30/H.264 Main/5 Mb/s encoder-decoder, fixed 280 ms frame backlog, and
optional slice loss, then EDM, and are counted only when results reach Tk. REAL
counts the corresponding PDRAW-to-EDM-to-UI results. Missing or expired
measurements are `N/A`; core model throughput is kept separate and is never
presented as end-to-end FPS.

The **飛控與限制** tab contains **飛控遙測（Olympe 讀回）**, which shows the
flight-state cache independently of visual localization: fused roll/pitch/yaw, altitude above
takeoff, AGL, NED ground speed, GPS fix/location/accuracy/satellites, heading/RTH,
wind/vibration/hover warnings, Wi-Fi/link quality, and IMU/barometer/GPS/etc.
sensor health. These are read-only Olympe event-cache values. Original ANAFI's
public events expose sensor health and flight-controller estimates, not raw IMU
samples or raw barometric pressure; the UI labels that limitation explicitly.

Production does not force a CUDA synchronization for timing. Set
`SFM_LOC_PROFILE_GPU=1` only for a profiling run; it enables CUDA Events and the
required event synchronization, so its FPS is not a production FPS result.

### Render and layout budget

Everything below runs on the Tk main thread, which is also the thread that hands
frames to the localizer, so per-tick waste there is not free.

- Flight actions (手動/搖桿, 恢復電腦控制, 懸停, 原地降落, 緊急停止電腦動作) and the
  pose/frame age readout sit in a fixed bar **above** the control tabs. The fixed
  295 px pane has six tabs and no horizontal or vertical scrollbar: 操作與定位、
  定位資訊、飛控與限制、校正、場域資產、系統紀錄. Command strings and bindings are
  unchanged. `test_operator_render_perf.py` verifies every tab fits the minimum
  980×640 client area and every abort action remains outside the selectable tabs.
- Localization alerts (LOCALIZATION LOST, LOW CONFIDENCE, LOST hold) live only in
  the video panel (`render_video`) and in `loc_health_label`. **2026-08-03 operator
  decision:** the top-level banner was removed as duplicated by the middle of the
  interface; the in-video one is clipped to that panel and gated by the video dirty
  key. `incident_banner` (CONTROL LINK LOST, stream loss, disk pressure) is packed
  only while an incident is active — there is no idle 「安全狀態：正常」 row.
- Panels blit into the existing `ImageTk.PhotoImage` (~0.47 ms) instead of
  allocating a new one and recreating the canvas item every tick (~0.82 ms).
  A panel resize still allocates.
- The point-cloud base is a depth argsort plus a scatter over the whole cloud:
  ~16 ms at 250k points. While the operator drags or zooms it drops to
  `SFM_MAP_INTERACTIVE_POINTS` (default 60k, ~4.9 ms) and settles back to
  `SFM_MAP_STATIC_POINTS` (default 250k) once the view stops moving. The
  decimation level is part of the base-image cache key.
- The planned-route polyline still uses every point; only its dots are thinned to
  `SFM_ROUTE_DOT_MAX` (default 200). At a 2000-point route that is 9.5 ms → 1.1 ms.
- All HUD `StringVar`s are `DedupStringVar`: Tk repaints a label on every `set()`
  even when the text is unchanged, and the tick runs at 100 Hz while these values
  move at most at the localization rate. The comparison reads the live Tcl value,
  so text an operator typed into a bound Entry is still overwritten correctly.
  The rolling pipeline summary and the age readout are additionally throttled to
  10 Hz.

Safety pilot must hold the sticks. Autonomous route flight is unconditionally
`LOCKED`; the retired metric entrypoint exits before connecting, and the UI only
runs localization until external approval, two-person confirmation, and field
receipts exist.

Self-test without opening a GUI:

```bash
cd .../sfm_system/定位/mission/operator_interface
python3 flight_operator_app.py --selftest --max-points 30000
```

Run the desktop app:

```bash
cd .../sfm_system/定位/mission/operator_interface
python3 flight_operator_app.py --max-points 90000
```

Use `--interface simulated-stream` with an explicit `--video` for an ANAFI-like
720p30 replay.
The displayed frame is localized by a worker using the selected localization
runtime. YOLO object detection is **off by default even when a model file exists**;
only `--live-detect` enables it. `--no-live-detect` keeps the intent explicit:

```bash
python3 flight_operator_app.py \
  --interface simulated-stream \
  --video /path/to/input.MP4 \
  --boot-lock-ms 2500
```

Startup behavior matches the real flight sequence: wait for the localization
worker's model-ready handshake, then hold the first 1280x720 frame while
`BOOT_INIT` / MegaLoc / local matching / PnP locks the start pose. The left map
path is produced from the same frame shown on the right; replay JSON is not used
unless `--no-live-localize` is passed.

MegaLoc has three automatic production triggers: one `BOOT_INIT` retrieval
during the landed/takeoff-initialization phase, one recovery after two
consecutive low-confidence EDM results, and one retrieval on entry to each
actual `LOST` episode. A single WEAK result only raises the EDM local reference
count. The operator UI has no manual global-retrieval or benchmark-state
buttons. On sustained LOW, replay freezes its frame; real flight sends zero
PCMD and hands control to the pilot before requesting MegaLoc. Recovery then
continues with EDM and requires consecutive good fixes before motion can resume.

The route contract defaults to the same `outputs/current_safezone/flight_path.json`
used by the production runner. It remains loaded and verified, but its map line is
hidden by default; the operator can enable `顯示規劃路徑` when needed. The app
prints its resolved path and SHA-256 at startup. A draft route is used only when
selected explicitly with `--route-json` or `SFM_FLIGHT_PATH_JSON`. Startup fails
closed if the route is empty, malformed, not three-dimensional, or contains a
non-finite coordinate.

## Live Object Detection

YOLO runs in `object_detector_worker.py`, separate from the localization worker.
This keeps the UI and localization loop responsive while detection runs every
few frames.

Default detector settings:

- Enabled: no; opt in with `--live-detect`
- Model: `../../object_detection/models/power_equipment_yolo26n_640_fp16.engine`
- Python: the current app runtime; use `--detector-python` or
  `SFM_DETECTOR_PYTHON` only for an explicitly prepared alternate environment
- Input stream: same `1280x720` RGB frame shown in the right video panel
- Inference size: `640`
- Detection period: every `3` frames by default
- Confidence: `0.25`
- IoU: `0.7`

Runtime controls:

```bash
python3 flight_operator_app.py \
  --video /path/to/input.MP4 \
  --detect-every-n-frames 5

python3 flight_operator_app.py --no-live-detect
```

If `--live-detect` is requested and the model is absent, startup fails with a
clear error instead of launching a worker that cannot become ready.
Localization and detection worker input/output are time-bounded; an exited,
non-reading, or partial-response worker is reaped and restarted without blocking
the UI shutdown path. A `success` response is accepted only with finite XYZ.

The right video panel overlays the latest detection boxes and the bottom
telemetry panel shows detection FPS, detector latency, object count, and the
frame name used by the detector. The latest detection JSON is also written to:

```text
/tmp/sfm_flight_operator_detection_status.json
```

## Parrot ANAFI Simulation Profile

The desktop simulator uses the ANAFI white-paper values as hard limits and HUD
metadata:

- Video stream: `1280x720 @ 30 fps`, simulated H264/RTP, `5 Mb/s`, `280 ms` link latency.
- Camera: video HFOV `69 deg`, digital zoom `1.0x` to `3.0x`.
- Gimbal: controllable pitch `-90 deg` to `+90 deg`, max control speed `180 deg/s`.
- Flight envelope: max horizontal speed `15 m/s`, vertical speed `4 m/s`, yaw rate `200 deg/s`.
- Flight behavior: takeoff targets `1.0 m` hover before AUTO; `hover`, `manual`, and `land` remain highest-priority safety commands.
- Battery model: linear `25 min` flight-time estimate for simulator HUD only.

Use `--video-stride N` only for debugging. The default `--video-stride 1` keeps
the 720p30 ANAFI-like stream; values above 1 intentionally drop source frames
before the UI and no longer represent the real drone stream.

Offline replay mode is still available for comparison:

```bash
python3 flight_operator_app.py --no-live-localize \
  --replay-json /path/to/localization_replay.json
```

Offline replay uses the simulator backend. `--interface real-flight` selects
`OlympeLiveBackend`; cross-interface video/flag combinations fail before either
backend is created.

Keyboard safety shortcuts:

- `Space`: hover
- `Esc`: manual

### Firmware magnetometer calibration

Panel **韌體羅盤校正** exposes the ANAFI and SkyController 3 calibration APIs as
two separate controls. This replaces the need to enter FreeFlight 6 for the same
firmware operation on the supported hardware, while retaining FreeFlight 6 as a
manual fallback.

- Calibration can start only after the flight state is explicitly read back as
  `landed`, from a local operator button click, with a healthy live link.
- Starting clears pending PC movement and sends zero PCMD only when the PC already
  owns control. It never sends takeoff, motor, or movement commands.
- The aircraft reports whether calibration is required or recommended, the current
  X/Y/Z axis, completion flags, and failure state. SkyController 3 reports its own
  NotCalibrated/CalibratingX/Y/Z/Calibrated state.
- Required, unknown, failed, or in-progress calibration blocks takeoff and
  autonomous start. Firmware `recommended` status produces a warning but does not
  block operator takeoff.
- The operator must lift and rotate the powered aircraft or controller by hand as
  directed by the displayed axis. Keep propellers stopped and move away from metal,
  reinforced concrete, vehicles, magnets, and strong current-carrying cables.

The two devices are calibrated independently. Cancelling one does not start or
cancel the other, and no calibration is started automatically when the application
opens.

### Passive gravity / attitude check

Panel **姿態／重力檢查（只讀；不寫入飛機）** guides three props-off rotations
while sampling body attitude:

1. **水平旋轉 (yaw)** — spin about gravity, deck level  
2. **前後俯仰 (pitch)** — tip nose up/down  
3. **左右側傾 (roll)** — tip sideways  

The UI reports level tilt, per-phase angular coverage, body-frame gravity unit vector, and PASS/FAIL. Results are saved under `outputs/flight_logs/gravity_cal_*.json`. This is a project-side sensor/axis check only; it does not write calibration data to the aircraft and is not a substitute for firmware magnetometer calibration.

- **In this sim app**: starting a phase synthesizes matching attitude so you can practice the flow.
- **On the real airframe** (props off, passive telemetry only):

```bash
cd ../flight_control
python3 gravity_calibration.py --selftest
python3 gravity_calibration.py --live --ip 192.168.53.1 --controller skycontroller3
```

Map note: GLOMAP gravity-up is **-Y**; level-phase `g_body` is NED-down in the body frame (level ≈ `(0,0,1)`).

Micro-move buttons / keys use the selected backend:

- Buttons: 右上前 / 左上前 / 右下前 / 左下前 / 右上後 / 左上後 (+ 下後 補齊)
- Keys: `U I O` 左上前/前/右上前, `J L` 左/右, `M , .` 左下前/後/右下前,
  `7 8 9` 左上後/上/右上後, `1 2 3` 左下後/下/右下後, `W A S D` 前後左右, `R/F` 上下

- With `--interface simulated-stream`, they move only the simulated pose.
- With `--interface real-flight`, they use the protected hold-to-move PCMD path: press sends the
  bounded nudge and release returns to hover. Do not change these bindings or
  release semantics without explicit pilot review.

Safety for the live operator is defined in `../SAFETY.md`; Esc / 手動 returns
control to the safety pilot. Agents must never invoke takeoff or live movement.

## Browser Fallback

This is useful for remote monitoring or quick browser review, but is not the
preferred real-flight control surface.

```bash
cd .../sfm_system/定位/mission/operator_interface
python3 serve_flight_interface.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

Backend API expected by the UI:

- `GET /api/state`
  - Returns mode, localization status, pose, inliers, reprojection error, tracker state.
- `POST /api/control`
  - Body: `{"command": "hover|land|manual|auto|takeoff|boot_lock|start_auto|pause|resume|gimbal_pitch|zoom", "payload": {...}}`
  - This is the hook that should call Olympe / ANAFI commands.
- `GET /api/video.jpg` or future MJPEG/WebRTC endpoint
  - Drone live stream.
- `GET /api/map_points`
  - Downsampled point cloud for the left map panel.

Safety rule for real integration:

- `hover`, `land`, and `manual` must bypass AUTO mission logic and be handled at the highest priority in the Olympe backend.
- Losing the 720p stream should force `STREAM_LOST_HOVER` before localization or path-follow commands run.
- Losing localization should force hover before any path-follow command is sent.

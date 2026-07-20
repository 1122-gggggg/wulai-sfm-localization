# Flight Operator Interface

Purpose: operator UI for offline replay and explicit `--live` real-drone operation.

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

- Left: point-cloud map, planned path, localization trail. Mouse controls: left drag 360-degree rotate, double left click sets the rotation pivot like CloudCompare, middle drag roll, right drag pan, wheel zoom. Use `重設地圖` to reset and `上下翻面` to flip the map by 180 degrees.
- Right: ANAFI-like drone video stream. Input stream is resized to `1280x720`
  and paced as 720p30 by default.
- Bottom: manual/auto, hover, land, takeoff, localization lock, mission start, gimbal pitch, zoom.

## Single-Process Desktop App

### LIVE real-drone mode (Olympe via SkyController)

**`--live` is not a simulator flag. It connects the controls below to a real
Olympe backend. Only the human operator may press the takeoff button.**

```bash
cd .../mission/operator_interface
# stop any other Olympe session first (single connection)
python3 flight_operator_app.py --live \
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
It prevents the aircraft from being piloted beyond the configured radius; it
does **not** automatically start Return-To-Home. RTH is a separate
`NavigateHome` operation and depends on valid GPS/home state. The takeoff
preflight also requires at least 30% battery and, while the distance geofence is
enabled, a confirmed GPS fix. Use `--min-takeoff-battery-pct` only when the human
safety owner has approved a different floor.

The launcher keeps the UI and localization interpreters explicit:

```bash
SFM_UI_PYTHON=/path/to/olympe-python \
SFM_LOCALIZER_PYTHON=/path/to/torch-python \
IP=192.168.53.1 CTRL=skycontroller3 ./start_anafi_live.sh
```

The standard launcher starts with `MaxAltitude=30 m`, `MaxDistance=100 m`, and
the distance geofence enabled. Override these conservative startup values with
`SFM_MAX_ALTITUDE_M` and `SFM_MAX_DISTANCE_M`, or edit and apply them from the
ANAFI panel while the aircraft is confirmed landed. The panel always shows the
aircraft readback separately; applying limits while airborne is rejected.

UI Python resolution is `SFM_UI_PYTHON` > legacy `VENV` executable > the
nearest package/workspace `.venv/bin/python` (up to two parent levels) >
`python3` on `PATH`. The launcher does not
invent a `DISPLAY`; preserve a valid `DISPLAY` or `WAYLAND_DISPLAY` from the
desktop session. Supplying only a known ANAFI IP or controller derives its safe
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
| 動搖桿（SC USB） | HID 偵測偏轉 → 強制交回搖桿（即使當下是 PC 控機） |
| 微移 8 方向 | hold-to-move PCMD; release returns to hover |
| 俯仰滑桿 | gimbal set_target |
| 自動 / 開始巡檢 | take PC control + localization feed (does **not** arm path_follow `--fly`) |
| 關窗 | Landing (if PC still piloting) + restore sticks |

Command log: `定位/outputs/flight_logs/live_ui_cmdlog_*.jsonl`.

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

Production does not force a CUDA synchronization for timing. Set
`SFM_LOC_PROFILE_GPU=1` only for a profiling run; it enables CUDA Events and the
required event synchronization, so its FPS is not a production FPS result.

Safety pilot must hold the sticks. The legacy autonomous path-follow entrypoint
still exists, but agents and automation must never invoke an arming/takeoff path.

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

Without `--live`, pass an explicit `--video` to use an ANAFI-like 720p30 replay.
The displayed frame is localized by a worker using the selected localization
runtime. YOLO object detection is **off by default even when a model file exists**;
only `--live-detect` enables it. `--no-live-detect` keeps the intent explicit:

```bash
python3 flight_operator_app.py \
  --video /path/to/input.MP4 \
  --boot-lock-ms 2500
```

Startup behavior matches the real flight sequence: read the first 1280x720 frame,
hold it while `BOOT_INIT` / MegaLoc / XFeat+LighterGlue / PnP locks the start pose,
then resume the stream. The left map path is produced from the same frame shown
on the right; replay JSON is not used unless `--no-live-localize` is passed.

Flight-like replay also pauses immediately on a missing pose, or after two
consecutive successful-but-low-confidence results (inliers below
`SFM_LOW_CONF_INLIERS`, reprojection above `SFM_LOC_HIGH_REPROJ`, or tracker
`WEAK`). The held frame is sent through the one-shot LOST/MegaLoc acquisition
path and playback resumes only after that strong request returns a high-quality
fix. Configure the low-result count with `--low-confidence-hold-results`; live
camera streams are never paused. The real-flight loop instead sends zero PCMD
immediately and requires `SFM_RECOVERY_GOOD_FIXES` consecutive good fixes
(default 2) before motion resumes.

The route overlay defaults to the same `outputs/current_safezone/flight_path.json`
used by the production runner. The app prints its resolved path and SHA-256 at
startup. A draft route is used only when selected explicitly with `--route-json`
or `SFM_FLIGHT_PATH_JSON`. Startup fails closed if the route is empty, malformed,
not three-dimensional, or contains a non-finite coordinate.

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

Offline replay uses the simulator backend. `--live` selects the existing
`OlympeLiveBackend`; do not treat the two modes as interchangeable.

Keyboard safety shortcuts:

- `Space`: hover
- `Esc`: manual

### Gravity / IMU calibration

Panel **重力校正** guides three props-off rotations while sampling body attitude:

1. **水平旋轉 (yaw)** — spin about gravity, deck level  
2. **前後俯仰 (pitch)** — tip nose up/down  
3. **左右側傾 (roll)** — tip sideways  

The UI reports level tilt, per-phase angular coverage, body-frame gravity unit vector, and PASS/FAIL. Results are saved under `定位/outputs/flight_logs/gravity_cal_*.json`.

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

- Without `--live`, they move only the simulated pose.
- With `--live`, they use the protected hold-to-move PCMD path: press sends the
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

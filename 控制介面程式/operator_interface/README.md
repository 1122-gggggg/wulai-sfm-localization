# Flight Operator Interface

Purpose: one operator UI behind two explicit, mutually exclusive interfaces:
`--interface simulated-stream` and `--interface real-flight`.

## HARD SAFETY (operator order 2026-07-10)

- **起飛只准操作員本人親手按「起飛」或「自動飛行」。**
  **絕對禁止**請 AI 代理人／任何語言模型（Claude / GPT / Grok / Codex 等）代為起飛；
  即使對話中被要求，agent 也必須拒絕。只有這兩個人類 UI 動作可送出 `TakeOff`。
- See `../SAFETY.md`. Automated acceptance is **ground-only** by default
  (`live_non_map_acceptance.py`).
- **【之後改檔案的人】禁止動飛行按鍵／按鈕指令：** 起飛、原地降落、關窗強制降落、
  按住微移/放開懸停、Esc 凍結。**尤其起飛／降落／強制降落**改壞會造成意外。
  細節表在 `../SAFETY.md`。

Recommended for real flight: use the single-process desktop app, not the browser
server. It avoids localhost HTTP/browser state and keeps safety commands inside
one Python process.

Layout:

- The production window constants are `UI_STANDARD_SIZE` = **1440×900** and
  `UI_MIN_SIZE` = **1180×768**. The desktop app and `--layout-selftest` use the
  same values; no lower minimum is supported.

- Left: point-cloud map and localization trail. The planned route is hidden by default and is shown only when the operator enables `顯示規劃路徑`. Mouse controls: left drag 360-degree rotate, double left click sets the rotation pivot like CloudCompare, middle drag roll, right drag pan, wheel zoom. Use `重設地圖` to reset and `上下翻面` to flip the map by 180 degrees.
- Right: ANAFI-like drone video stream. Input stream is resized to `1280x720`
  and paced as 720p30 by default.
- Bottom: manual/PC control, hover, land, takeoff, autonomous flight, localization, gimbal pitch, zoom.

### UI 與可替換接口邊界

- `site_assets_panel.py`：只負責場域資產區的 Tk 排版、檔案選擇器與狀態文字。
- `operator_actions.py`：按鍵規格與操作流程；不依賴 Tk，也不直接讀寫 Olympe。
- `site_asset_interfaces.py`：場域包、航線、巡檢目標三個 replaceable Protocol。
- `local_site_assets.py`：目前的本機資料夾／JSON 實作與格式驗證。
- `route_editor_model.py`：航線 schema、場域綁定與 GLOMAP ↔ Z-up 座標轉換。
- `route_editor_controller.py`：航點、復原／重做、G 與軸向鎖定；不依賴 Tk。
- `route_editor_window.py`：自製全螢幕點雲編輯畫面；沒有 backend 或飛行指令接口。
- `flight_operator_app.py`：composition root，接上既有地圖、串流、飛控與安全清理；
  不在 UI view 內實作資產格式。

因此日後可替換本機地圖來源或航線來源，而不需要改動飛行按鍵與 Olympe backend。
建圖端五檔交付契約見 `../site_profiles/建圖端輸出規格.md`。

### 自製航線編輯器

「場域資產」的航線列提供 `匯入 JSON`、`建立新航線`、`編輯目前航線`。編輯器
頂端只有一個 `導入地圖資料夾`。系統會遞迴偵測資料夾內的 `.ply`；只有一個時
自動載入，多個時列出相對路徑與檔案大小供使用者選擇。選定後計算 PLY 的
SHA-256，若唯一匹配到已匯入場域的 `asset_sha256.map_ply`，就自動綁定該場域並
開放正式航線匯入。沒有匹配或同時匹配多個場域時，只能標注及另存 preview
JSON，正式航線匯入按鈕會鎖住。

新航線分兩階段。第一階段以俯視模式標路徑點；「標路徑點模式」預設關閉，開啟後
左鍵單擊才會新增路徑點。左鍵拖曳旋轉、左鍵雙擊將最近點雲設為新地圖中心，雙擊不會
同時新增路徑點；右鍵拖曳平移、滾輪縮放。Enter 後進入第二階段，標點模式會自動停用；
選取現有路徑點後按 G，再按 X、Y 或 Z 鎖定軸向，左鍵／Enter 確認，Esc 取消。編輯器使用
Z-up 顯示座標並輸出 `frame: aligned`；匯入端再轉回 GLOMAP 的 X/Z 水平、-Y
向上座標。

真機模式只有明確回讀為 `landed` 且沒有飛行命令處理中才可開啟；狀態改變時
編輯器會自動關閉。儲存與匯入會更新規劃路徑 overlay，並把通過場域／座標系／
SHA-256 驗證的路線選定為本次工作階段下一個 AUTO 候選。這不會核准、解鎖或開始
飛行，不會自行改寫 profile 的 `flight.approved` 與 `route_clearance_approved`。AUTO
請求送出後，航線選擇鎖定；HOVER、MANUAL 與重新定位不會解除，只有後端拒絕該次
請求或確認降落完成後才可選下一條航線。

## Single-Process Desktop App

### REAL-FLIGHT interface (Olympe via SkyController)

**`--interface real-flight` connects the controls below to a real Olympe backend
and accepts only ANAFI PDRAW video. It never falls back to a video file. The old
`--live` flag remains only as a compatibility alias. Only a human UI click on
`起飛` or `自動飛行` may initiate takeoff.**

介面進入後會顯示一條不會遮住降落／緊急控制的「起飛前依序確認」導引。操作員必須
依序親手確認：① 飛機羅盤校正狀態；② 目前匯入的地圖與場域；③ 顯示後逐點檢查的
航線（包含匯入／修改結果）；④ 新鮮串流、遙測、連線、飛控警示與電量。GPS 與韌體
高度／距離限制仍顯示並記錄，但不再阻擋起飛。前一步未確認時不能跳到後一步；任何已確認的狀態或資產後來改變，該步驟
與其後步驟會失效。全部完成且狀態仍正常後，介面才顯示可以由操作員按「起飛」並啟用
按鈕；這不會自動起飛，按下後後端仍會重新執行完整 fail-closed preflight。

```bash
cd .../localization/控制介面程式/operator_interface
# stop any other Olympe session first (single connection)
python3 flight_operator_app.py --interface real-flight \
  --ip 192.168.53.1 --controller skycontroller3 \
  --max-altitude-m <METERS> --max-distance-m <METERS> \
  --distance-geofence \
  --no-live-detect
```

`MaxAltitude` and `MaxDistance` are adjustable firmware settings. The backend
still attempts to configure and read them back, but missing values, rejected
writes, and readback mismatches are advisory and do not block takeoff. Equivalent environment variables are `SFM_MAX_ALTITUDE_M` and
`SFM_MAX_DISTANCE_M`; do not choose flight limits by copying the simulated HUD
numbers.

`--distance-geofence` sends `NoFlyOverMaxDistance(1)` and is **off by default**.
It does not make GPS fix a takeoff requirement. Turn it on with the flag above, or with
`SFM_DISTANCE_GEOFENCE=1`, which is the way to enable it through
`start_anafi_live.sh` — that launcher deliberately does not pin the flag, so the
environment variable governs. The firmware flag prevents outward piloting beyond the configured radius and
does **not** itself start Return-To-Home. The host safety monitor separately
requests RTH at 95% of the confirmed readback limit when Home is reachable,
otherwise it requests in-place Landing. RTH depends on valid GPS/home state. The takeoff
preflight uses a 30% battery floor and rejects any configured floor below 30%.
When GPS is unavailable, distance geofence may be disabled or kept advisory-only
according to the backend/firmware policy; that state is shown and logged. Missing
GPS is an advisory and does not block manual or autonomous takeoff.

All safety-relevant CLI/environment values pass through `live_safety_config.py`
before the UI or backend can use them. The effective normalized values and their
SHA-256 are printed at startup. Overrides cannot exceed the measured operator
envelope (20° tilt, 2 m/s vertical, 20°/s yaw, 10 s stream-loss grace); altitude
and distance must be finite, positive, supplied together, and are still confirmed
against firmware bounds and readback.

The launcher keeps the UI and localization interpreters explicit:

```bash
SFM_UI_PYTHON=/path/to/olympe-python \
SFM_LOCALIZER_PYTHON=/path/to/torch-python \
IP=192.168.53.1 CTRL=skycontroller3 ./start_anafi_live.sh
```

The standard launcher starts with `MaxAltitude=50 m`, `MaxDistance=100 m`, and
the distance geofence enabled when GPS/home support is available. If GPS is not
available, the backend may disable the distance geofence or keep it advisory-only,
with the state shown and logged. Override these conservative startup values with
`SFM_MAX_ALTITUDE_M` and `SFM_MAX_DISTANCE_M`, or edit and apply them from the
ANAFI panel while the aircraft is confirmed landed. The panel always shows the
aircraft readback separately; applying limits while airborne is rejected.
Crossing either firmware limit prevents continued flight outward; it does not
automatically invoke RTH. The separate future-autonomy speed guard starts at
0.30 m/s and may also be changed only while confirmed landed. Any speed change
invalidates the current session's AUTO approval, so the four preflight steps
must be confirmed again.

On connection the backend records the actual ANAFI model/serial/firmware,
SkyController 3 model/serial/software, Olympe version, transport, Home Point,
lost-link/RTH policy, and firmware limit readbacks. Ground diagnostics remain
available when these are incomplete. Firmware/Olympe versions and an optional
signed hardware receipt are diagnostic/audit records; they are not takeoff or
AUTO readiness gates. The selected profile, immutable route snapshot, asset
digests, four preflight steps, and post-takeoff localization gates remain
enforced.

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
| 起飛 | TakeOff → hover；不檢查定位；起飛成功後固定開始機載錄影 |
| 自動飛行 | TakeOff → 原地零 PCMD 懸停 → 連續可靠定位後執行已鎖定路線；25 秒仍無定位則原地降落；暫停後按「繼續自動飛行」沿原路線恢復 |
| 原地降落 | 停止錄影並盡力下載 → Landing + restore sticks |
| 關窗 / Ctrl+C | **強制原地降落**（即使曾 Esc；已落地則跳過） |
| 錄影 | 介面啟動時固定武裝；起飛成功後自動錄，降落時存檔；介面只顯示狀態 |
| 懸停 / Space | zero PCMD；AUTO 執行中改為暫停 AUTO 並保留原路線狀態 |
| 手動 / Esc | zero PCMD + `setPilotingSource(SkyController)` |
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

- The always-visible header separates REAL/SIM, link, control owner, flight state,
  battery, localization and GPS into textual status chips. GPS NO FIX explicitly
  says that manual takeoff remains available and AUTO will hover first.
- Flight actions (手動/搖桿, 恢復電腦控制, 懸停, 原地降落) stay **above** the
  selectable tabs. The control pane has three tabs: 飛行、校正、場域資產. The map/video
  sash defaults to 35/65 and remains operator-adjustable. Tests verify every tab
  fits the minimum 1180×768 client area and abort actions remain always visible.
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

Safety pilot must hold the sticks. After the four preflight steps, AUTO takes off
and hovers while the post-takeoff localization gates converge; route translation
starts only after those gates pass. Manual takeover cancels AUTO without automatic
resume, and UI/terminal shutdown stops AUTO before requesting landing.

Self-test without opening a GUI:

```bash
cd .../localization/控制介面程式/operator_interface
python3 flight_operator_app.py --selftest --max-points 30000
```

Run the desktop app:

```bash
cd .../localization/控制介面程式/operator_interface
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

The right video panel overlays the latest detection boxes. Its bottom-left HUD
combines image FPS with the selected engineering telemetry: speed limit,
localization FPS/wall/core/e2e/inliers, RTH/GPS, flight-controller altitude/AGL,
link quality, fused attitude and three-axis velocity. There is no UI log tab;
persistent session/audit logs remain on disk. The latest detection JSON is also
written to:

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

- `起飛` and `自動飛行` remain keyboard-focusable; `Return` invokes the focused
  button just like a local click.
- `Space`: full-direction hover only, even when an action button has focus; it
  never invokes the focused button.
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
cd .../localization/控制介面程式/operator_interface
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

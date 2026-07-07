# Flight Operator Interface

Purpose: operator UI for future real-drone integration.

Recommended for real flight: use the single-process desktop app, not the browser
server. It avoids localhost HTTP/browser state and keeps safety commands inside
one Python process.

Layout:

- Left: point-cloud map, planned path, localization trail. Mouse controls: left drag 360-degree rotate, double left click sets the rotation pivot like CloudCompare, middle drag roll, right drag pan, wheel zoom. Use `重設地圖` to reset and `上下翻面` to flip the map by 180 degrees.
- Right: ANAFI-like drone video stream. Input stream is resized to `1280x720`
  and paced as 720p30 by default.
- Bottom: manual/auto, hover, land, takeoff, localization lock, mission start, gimbal pitch, zoom.

## Single-Process Desktop App

Self-test without opening a GUI:

```bash
cd /media/cihcilab/新增磁碟區/sfm_system/定位/mission/operator_interface
python3 flight_operator_app.py --selftest --max-points 30000
```

Run the desktop app:

```bash
cd /media/cihcilab/新增磁碟區/sfm_system/定位/mission/operator_interface
python3 flight_operator_app.py --max-points 90000
```

By default this reads `/home/cihcilab/Downloads/P0230023.MP4` as an ANAFI-like
720p30 stream, localizes the displayed frame through a live `/usr/bin/python3.12`
worker, and runs YOLO object detection through a separate TensorRT worker:

```bash
python3 flight_operator_app.py \
  --video /home/cihcilab/Downloads/P0230023.MP4 \
  --boot-lock-ms 2500
```

Startup behavior matches the real flight sequence: read the first 1280x720 frame,
hold it while `BOOT_INIT` / MegaLoc / XFeat+LighterGlue / PnP locks the start pose,
then resume the stream. The left map path is produced from the same frame shown
on the right; replay JSON is not used unless `--no-live-localize` is passed.

## Live Object Detection

YOLO runs in `object_detector_worker.py`, separate from the localization worker.
This keeps the UI and localization loop responsive while detection runs every
few frames.

Default detector settings:

- Model: `../../object_detection/models/power_equipment_yolo26n_640_fp16.engine`
- Python: `/home/cihcilab/miniconda3/bin/python3`
- Input stream: same `1280x720` RGB frame shown in the right video panel
- Inference size: `640`
- Detection period: every `3` frames by default
- Confidence: `0.25`
- IoU: `0.7`

Runtime controls:

```bash
python3 flight_operator_app.py \
  --video /home/cihcilab/Downloads/P0230023.MP4 \
  --detect-every-n-frames 5

python3 flight_operator_app.py --no-live-detect
```

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
  --replay-json /media/cihcilab/新增磁碟區/sfm_system/定位/outputs/downloads_validation_20260702/P0230023_v3_temporal.json
```

This uses `DroneBackend` inside `flight_operator_app.py` as the integration
point. Replace that class with Olympe/localizer/control calls for the real drone.

Keyboard safety shortcuts:

- `Space`: hover
- `Esc`: manual

## Browser Fallback

This is useful for remote monitoring or quick browser review, but is not the
preferred real-flight control surface.

```bash
cd /media/cihcilab/新增磁碟區/sfm_system/定位/mission/operator_interface
/usr/bin/python3.12 serve_flight_interface.py --host 127.0.0.1 --port 8765
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

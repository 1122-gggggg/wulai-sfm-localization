# IMU / fused-state capability audit

Date: 2026-08-17
Scope: production live localization (GlueMap/EDM/PnP) and ANAFI Olympe telemetry.
Rule: only quantities that exist in code or sampled logs. No assumed sensors.

## Verdict

| Question | Answer |
|---|---|
| Capability level for **map-frame** short-term odometry | **Level D** (orientation prior only) |
| Recommended propagation | Visual-velocity position (already in `ProductionEDMTracker`) + fused-attitude **yaw increment** as a viewing-direction prior |
| Fused XYZ in GlueMap map frame? | **No** |
| Fused NED velocity usable as map velocity? | **No** (scale-free map; no measured `map_units_per_meter`; frames differ) |
| Raw accel / gyro / quaternion? | **No public path** |
| Camera-body extrinsic on the production river site? | **Not measured.** Operator convention: camera center = UAV origin |
| Localization worker currently receives IMU? | **No** (SFM1/SFM2 = mode + optional capture stamp) |


Do **not** implement Level A/B/C on ANAFI fused telemetry. Interfaces for map-aligned odometry exist for a future calibrated adapter; the ANAFI adapter never claims map alignment.

---

## 1. What we actually have

### Signal table

| Signal | Exists | Raw / fused | Frame | Units | Rate | Timestamp | Source | Runtime to localizer? |
|---|---|---|---|---|---|---|---|---|
| quaternion | **No** | — | — | — | — | — | Not in Olympe message tables or backend | No |
| roll | **Yes** | Firmware fused Euler | Aircraft NED | rad | source ~5 Hz; live poll ~8 Hz | `get_state` poll monotonic; **not** event-UUID fresh | `olympe_live_backend.py:4855-4859` `_poll_mandatory_telemetry` | No (UI/backend only) |
| pitch | **Yes** | Firmware fused Euler | Aircraft NED | rad | same | same | same | No |
| yaw | **Yes** | Firmware fused Euler | NED, 0=North, **CW** | rad | same | same | same; conversion `path_follow_flight.py:783-786` | No |
| gyro xyz | **No** | — | — | — | — | — | Not imported in `log_anafi_telemetry.py` or live backend | No |
| accel xyz | **No** | — | — | — | — | — | same | No |
| linear acceleration | **No** | — | — | — | — | — | same | No |
| gravity-compensated acceleration | **No sensor**. Derived `g_body` from fused roll/pitch only | derived | body NED | unit vector | cal poll 20 Hz | cal `t_mono` | `gravity_calibration.py:62-77` `attitude_to_body_gravity` | No |
| velocity xyz | **Yes** | Firmware fused | NED: `speedX` north, `speedY` east, `speedZ` down | m/s | source ~5 Hz; live poll ~8 Hz | SpeedChanged UUID + `event.date` → `ground_speed_mono_ns` | `olympe_live_backend.py:4871-4886` | No |
| position xyz (map) | **Visual only** | EDM/PnP `C = -Rᵀ t` | GlueMap / COLMAP map | map units | camera rate | `capture_stamp` monotonic | `production_edm_tracker.py:1157-1163` `_pose_components` | Yes (that is the localizer output) |
| position xyz (FC) | Live: **WGS84 only**. Sphinx-only: local NEU from first GPS fix | fused GPS | WGS84 or Sphinx-local NEU, **not** GlueMap | deg / m | GPS ~1 Hz (Sphinx) | GPS event UUID | live `GpsLocationChanged` `olympe_live_backend.py:4916-4944`; Sphinx `telemetry_sources.py:516-584` | No |
| altitude | **Yes** | fused FC | takeoff-relative + AGL | m | poll ~8 Hz | AltitudeChanged freshness-tracked | `olympe_live_backend.py:4861-4870`, `4918-4920` | No |
| barometer | Health bit only | — | — | — | — | — | `SensorsStatesListChanged`; no Pa | No |
| GPS | **Yes** | fused | WGS84 + 1σ accuracies | deg, m | poll | event UUID | `GpsLocationChanged` / `GPSFixStateChanged`. Real sampled session: `gps_fixed=false` | No |
| covariance / uncertainty | **No FC covariance**. Visual gates + heuristic only | software | map | mixed | — | — | `reprojection_metrics`, `localization_uncertainty.py`, unused `ConstantVelocityKalman` | Visual gates yes; IMU cov no |
| flight-controller EKF state | Internal only; not exported | fused events above | NED / WGS84 | — | — | — | Parrot firmware via Olympe 8.4 | No |

### A. Acceleration

There is **no** acceleration sample on the production or logger path.

Closest object: `attitude_to_body_gravity(roll, pitch)` (`gravity_calibration.py:62-77`) reconstructs a **unit gravity direction** from fused Euler under aircraft NED (`g_ned = +Z down`). It is not specific force, not linear acceleration, and not an EKF acceleration state.

### B. Velocity

`SpeedChanged` is firmware-fused **NED** velocity in m/s (`speedX` north, `speedY` east, `speedZ` down). Live backend stores `speed_north_mps` / `speed_east_mps` / `speed_down_mps` and `ground_speed_mps = hypot(N,E)` (`olympe_live_backend.py:4879-4886`).

- Not body-frame. Not GlueMap. Not optical-flow.
- `airspeed_mps` on `DroneState` is a **legacy alias of ground speed**, not `AirSpeedChanged`.
- Freshness: event UUID + wall-age-corrected monotonic. Stale cache is fail-closed for the landing speed gate.
- Reset/drift: firmware hover fusion; no project-level reset API.
- `ScaleFreeVisualImuFusion` (`read_only_flight_advisor.py:379-386`) explicitly refuses to treat NED velocity as a map observation without a measured scale.

### C. Position

Production map position is **visual camera center** in the reconstruction frame. River-site AUTO uses that camera center as the navigation point and does **not** apply `map→site` or `map_units_per_meter` (`文件/ARCHITECTURE.md`, `控制介面程式/site_profiles/README.md`).

Sphinx `SphinxTelemetrySource.get_pose` is GPS lat/lon/alt → local north/east/up from the first fix, remapped to `x=north, y=-up, z=east`. Comment in `telemetry_sources.py:466-483`: **not raw IMU**, **not** simulator ground truth, **not** GlueMap.

### Data flow

```text
ANAFI firmware EKF (internal, unpublished)
        │
        ▼
Olympe event cache (parrot-olympe 8.4)
        │
        ├─ AttitudeChanged / SpeedChanged / AltitudeChanged / GpsLocationChanged
        │     → olympe_live_backend.poll() ~8 Hz
        │     → DroneState
        │     → HUD, gravity_cal, HeadingEstimator, landing speed gate
        │     ✖ live_localizer_worker
        │
        └─ PDRAW 1280x720
              → ntp_raw_timestamp → host monotonic (olympe_frame_source.py)
              → SHM RGB + SFM2 capture_stamp
              → ProductionEDMTracker
              → MegaLoc only BOOT / each LOST episode
              → spatial EDM refs → PnP cam_from_world
              → Pose (raw map x/y/z + yaw)
```

Read-only `FusedState` / `ConstantVelocityKalman` / `ScaleFreeVisualImuFusion` live in `read_only_flight_advisor.py` and tests only. Not wired to the worker.

---

## 2. Coordinate frames

| Frame | Meaning | Axes / convention | Evidence |
|---|---|---|---|
| **M** GlueMap / COLMAP | Reconstruction world | Gauge-free. Up is a **measured** site property (`T_align_gravity.json` / `MapFrame`), not axis names. River AUTO stays in raw map. | `pose_types.py:25-28`, `real_path_follow_controller.py:92-96,135-143` |
| **C** OpenCV camera | Query / reference cameras | Right, down, forward. PnP is **world-to-camera**. | `reloc_localizer_edm.py:768-772`; adapter `edm_localizer_adapter.py:193-196` |
| **B** UAV body | ANAFI FRD | Forward, right, down. Used by extrinsic schema and gravity cal. | `pose_frame_chain.py:66`, `gravity_calibration.py:62-69` |
| **I** IMU | Not separately exposed | Treat as coincident with B for firmware attitude. No I↔B calibration in repo. | — |
| **O** FC odometry | NED + WGS84 / Sphinx-local | Yaw CW-from-North. Velocity NED. Not M. | `path_follow_flight.py:783-786`, `telemetry_sources.py:466-475` |

### PnP pose is `T_C_M` (cam_from_world), not `T_M_C`

`pycolmap.estimate_and_refine_absolute_pose` returns `ret["cam_from_world"]` (`reloc_localizer_edm.py:768`, `production_edm_tracker.py:1157-1162`):

```text
p_cam = R @ p_world + t
C = -R.T @ t          # camera center in M
fwd_world = R.T @ [0,0,1]
yaw = atan2(fwd_y, fwd_x)   # map-XY heading of camera +Z; not MapFrame.heading unless adapter remaps
```

Quaternion ordering: unused in the live visual path. New code uses **wxyz**.

Rotation: active matrices, right-handed, det = +1.

`HeadingEstimator` converts Olympe yaw → map CCW-from-East as `π/2 − yaw_ned` and accepts a visual update only when Δyaw matches IMU Δyaw (`path_follow_flight.py:761-836`). That is a **flight** heading fuse, not a localization prior.

---

## 3. Camera ↔ body extrinsic

Schema exists: `sfm-camera-body-extrinsic/v1` (`pose_frame_chain.py:21,64-142`).

`T_C_B`: `p_cam = R_C_B @ p_body + t_C_B` (body FRD → OpenCV). `approved` must be true before `NavigationPoseTransformer` will compose `T_W_M * inverse(T_C_M) * T_C_B`.

Production river snapshots set `asset_sha256.camera_body_extrinsic: null`. Site README: raw-map AUTO **does not load** `site_alignment` / `camera_body_extrinsic`; camera optical centre is the navigation origin.

Vehicle `parrot_anafi_720p.json`: capability `imu`, `required_calibrations: []`.

**Camera center = UAV origin** (operator decision, `camera_center_is_body_origin`). Lever arm is zero. This is **not** camera axes = body FRD; Δyaw still does not apply a measured `T_C_B` rotation.


---

## 4. Capability level

| Level | Requirement | Available for map-frame localization? |
|---|---|---|
| A fused position + orientation | XYZ + quaternion/RPY in a common O frame | **No** — FC position is WGS84/Sphinx-local, not M |
| B fused velocity + orientation | v + RPY in a frame that can be applied to M | **No** — v is NED m/s; M is scale-free |
| C orientation + acceleration | RPY + accel | **No acceleration** |
| **D orientation only** | RPY / yaw | **Yes** — fused Euler on the backend; not yet on the worker |

Plus, independently of IMU: `ProductionEDMTracker` already does **visual constant-velocity** center prediction, `prediction_max_dt = 0.25 s` (`production_edm_tracker.py:611-620`).

Implemented path: **Level D orientation prior + existing visual-velocity position**. Level A/B adapters accept only samples explicitly labeled `frame="map"` or a caller-supplied `T_M_O`.

---

## 5. Existing architecture impact

### Reuse (do not duplicate)

| Module | Role |
|---|---|
| `ProductionEDMTracker` | BOOT / TRACK / WEAK_TRACK / LOST, MegaLoc policy `boot_and_lost_once`, spatial radius + yaw + covis + last_refs |
| `_pose_quality_rejected` / `reprojection_metrics` | inliers, ratio, RMS, 8×8 grid coverage |
| `_predict_center` | visual velocity |
| `EDMLocalizer.retrieve` | MegaLoc / candidate-restricted MegaLoc |
| `HeadingEstimator` / `ScaleFreeVisualImuFusion` | yaw-increment agreement (flight / advisor) |
| `CameraBodyExtrinsic` | if a future approved JSON appears |
| `TelemetryFreshnessStore` | event-UUID freshness |

### Modify (thin hooks, feature flag default off)

- `production_edm_tracker.py` — optional pose-guided prediction / radius / yaw prior
- `edm_localizer_adapter.py` — `observe_fused_state`, `pose_status`
- `live_localizer_protocol.py` / worker / UI prefix — optional SFM3 fused header
- `production_localizer_factory.py` — attach disabled-by-default controller

### Add

`定位演算法/deploy_code/sfm_glomap_deploy/pose_guided/`
`定位演算法/configs/pose_guided_localization.json`
`tests/localization/deploy/test_pose_guided_*.py`

### Do not modify

EDM matcher, MegaLoc weights, `edm_production_profile.json` required key set, pycolmap PnP, route controller, safety PCMD path.

---

## 6. Timestamp / sync

| Clock | Domain | Notes |
|---|---|---|
| Camera | host monotonic, NTP-mapped from `ntp_raw_timestamp` | Trusted only if `stamp_source==ntp-mapped` (`olympe_frame_source.py`) |
| Attitude | poll monotonic via `get_state` | **Not** UUID-fresh in live backend |
| Speed / alt / GPS | event UUID + `event.date` → monotonic | Fail-closed if frozen |
| PnP pose stamp | image `capture_stamp` | Must not be refreshed by inference delay |

There is **no** IMU–camera interpolator in production. New code interpolates only samples that share host monotonic and are within `max_sync_error_s`. Attitude without a trustworthy stamp is unused.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| IMU drift / wrong local radius | Uncertainty grows with age; WEAK expands radius; LOST → MegaLoc |
| Accel bias / gravity leakage | No accel integration |
| Camera/IMU frame mismatch | No body↔camera convert without approved extrinsic; Δyaw only |
| Timestamp mismatch | Reject fused samples older than `max_sync_error_s` |
| Map scale | NED m/s never added to map-unit state unless `metres_per_map_unit` is set |
| Convention mix (ENU/NED, CW/CCW) | Same `π/2 − yaw_ned` as `HeadingEstimator` |
| Stale `get_state` attitude | Conservative age cap; speed already UUID-fresh |
| Incorrect visual anchor | `is_safe_visual_anchor` uses acquire-level visual gates only |
| Predicted pose labeled confirmed | `PREDICTED_ONLY` vs `VISUALLY_CONFIRMED`; `ok` stays false |
| Perceptual alias / local lock-in | Existing jump / acquire-yaw / MegaLoc LOST fallback |
| GPS unused / unfixed | Do not seed map pose from GPS |

---

## 8. Replay / ablation data

`outputs/flight_logs/session_20260812T092032Z_real-flight_*` has 1 Hz fused speed/altitude and **no** `att_roll/pitch/yaw` in the JSONL, and no paired query frames. Insufficient for Baseline A vs B replay.

Ablation therefore lives in unit tests (synthetic frames, known SE(3)) plus the existing EDM tracker retrieval/quality suites.

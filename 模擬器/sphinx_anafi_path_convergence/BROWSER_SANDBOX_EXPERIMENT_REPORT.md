# Browser Sandbox Experiment Report

Generated on 2026-07-09 from `outputs/browser_sandbox/index.html`.

Historical replay notice: this report predates the 2026-07-10
`anafi_white_paper_v1_4` plant/default update. Its explicit 200 ms synthetic
total delay and recorded replay metrics remain historical settings. The current
280 ms value is ANAFI video end-to-end latency only and therefore a lower bound;
decode plus MegaLoc/XFeat/PnP latency must be measured and added separately.

## Scope

This file records the current browser sandbox random-error, hloc-delay, hloc
outage, route-tube, and final-landing settings, plus the latest replay result.

Important limitation: this browser sandbox does **not** simulate the Parrot
ANAFI firmware. It uses the same experiment controller code, but the aircraft
plant is `KinematicAnafi`, a first-order kinematic approximation. Use the
Sphinx/Olympe backend for Parrot ANAFI firmware/control-state validation.

## Reproduction Command

```bash
cd sfm_system/定位/experiments/sphinx_anafi_path_convergence
/home/allen/localization/.venv/bin/python make_browser_sandbox.py \
  --algorithm translational_waypoint \
  --route-style complex \
  --num-waypoints 10 \
  --return-to-start \
  --inspection-poles \
  --inspection-pole-waypoints 6,7 \
  --inspection-pole-right-offset-m 1.2 \
  --inspection-pole-top-above-waypoint-m 2.0 \
  --map-size-m 20 \
  --duration 220 \
  --seed 7 \
  --pose-source noisy_estimated \
  --pose-noise-m 0.25 \
  --pose-error-max-m 1.0 \
  --telemetry-delay-ms 200 \
  --telemetry-delay-jitter-ms 100 \
  --hloc-outage-interval-s 10 \
  --hloc-outage-duration-s 1.0 \
  --hloc-outage-start-s 10 \
  --camera-yaw-noise-deg 5.0 \
  --camera-yaw-error-max-deg 20.0 \
  --wind-gust-interval-s 5.0 \
  --wind-gust-m 0.5 \
  --arrival-radius 1.0 \
  --route-tube-radius 1.0 \
  --route-tube-exit-s 0.8 \
  --route-tube-exit-updates 6 \
  --max-pose-age-s 0.6 \
  --pose-loss-short-s 1.0 \
  --lost-abort-s 8.0 \
  --final-landing-radius 0.8 \
  --final-landing-hold-s 0.5 \
  --final-landing-max-est-speed 1.5 \
  --final-landing-yaw-stable-deg 20.0
```

The generated page is:

```text
outputs/browser_sandbox/index.html
http://localhost:9002/
```

## Coordinate Convention

The browser sandbox uses the experiment raw frame:

```text
x = horizontal forward/north-like axis
z = horizontal lateral/east-like axis
y = negative up, so height_up = -y
```

When discussing the map projection, the visual horizontal map is therefore
`x/z`, not `x/y`. The route requested as an `xy` projection is represented here
as a mostly straight `x/z` projection, with `y` used for height variation.

## Map And Route

Map:

```text
shape: square
side length: 20.0 map units
bounds: x [-10, 10], z [-10, 10]
route horizontal x span: 12.8 map units
route horizontal z weave span: 1.430 map units
route height span: 1.986 map units
planned waypoints: 10
controller route waypoints: 27
return leg: W1 -> ... -> W10 -> W9 -> ... -> W1
inspection poles: W6 and W7, route-right side
route seed: 7
```

The horizontal projection of the 10 planned waypoints is close to a straight
line with mild side-to-side weaving. Each waypoint height is randomly sampled
from the seed. The executed controller route expands those 10 planned points
into this out-and-back sequence:

```text
W1, W2, W3, W4, W5,
W6, W6 pole top, W6 return height,
W7, W7 pole top, W7 return height,
W8, W9, W10, W9, W8,
W7, W7 pole top, W7 return height,
W6, W6 pole top, W6 return height,
W5, W4, W3, W2, W1
```

Planned waypoints:

| waypoint | x | z | height_up |
|---:|---:|---:|---:|
| 1 | -6.400 | 0.025 | 3.533 |
| 2 | -4.978 | 0.692 | 1.786 |
| 3 | -3.556 | 0.674 | 3.471 |
| 4 | -2.133 | 0.031 | 3.335 |
| 5 | -0.711 | -0.715 | 2.417 |
| 6 | 0.711 | -0.621 | 1.924 |
| 7 | 2.133 | 0.035 | 2.357 |
| 8 | 3.556 | 0.715 | 2.639 |
| 9 | 4.978 | 0.589 | 3.261 |
| 10 | 6.400 | -0.157 | 3.771 |

Inspection poles:

| pole | base_x | base_z | top_height_up | yaw_target_rad |
|---|---:|---:|---:|---:|
| W6 pole | 0.405 | 0.540 | 3.924 | 1.829 |
| W7 pole | 1.623 | 1.121 | 4.357 | 2.010 |

Start condition:

```text
start position: x=-6.313, y=-3.465, z=0.932
start offset from W1: 0.911 map units
start bearing relative to route: 59.419 deg
initial yaw error: 12.0 deg
control dt: 0.05 s
```

## Controller And Safety Settings

```text
algorithm: translational_waypoint
pose source delivered to controller: noisy_estimated
controller sees truth directly: false
waypoint arrival: estimated 3D distance <= 1.0 map unit
route tube radius: 1.0 map unit
tube projection: active segment +/- 1 segment
tube abort debounce: outside tube for 0.8 s and 6 effective pose updates
initial rejoin grace: 10.0 s
manual handoff model: zero PCMD hover / ABORT_OR_MANUAL, not Emergency
```

The controller receives only the estimated pose/yaw stream. The true kinematic
pose and true yaw are added after command generation for visualization and
metrics.

At W10, the controller hovers, yaw-aligns the camera/body toward W9 using the
same horizontal projection angle logic used between forward waypoints, then
follows the reversed route back to W1 before completing.

At W6 and W7, including on the return leg, the controller hovers, yaw-aligns to
the corresponding pole, moves vertically in place to the pole-top waypoint, then
descends in place to the original waypoint height before continuing.

## Hloc-Like Error Model

Position estimate:

```text
pose_noise_seed: 10014
per-axis Gaussian sigma: 0.25 map units
random pose-noise clamp at sampling time: <= 1.0 map unit 3D resultant
```

Yaw estimate:

```text
yaw noise sigma: 5.0 deg
random yaw-noise clamp at sampling time: <= 20.0 deg
```

Delayed telemetry:

```text
base hloc/telemetry delay: 200 ms
delay jitter: uniform +/- 100 ms
controller pose-age stale gate: 0.6 s
```

Periodic complete localization outage:

```text
first outage: t=10.0 s
outage interval: every 10.0 s
outage duration: 1.0 s
behavior: no new hloc samples are delivered; the last delivered pose remains old
```

Because the controller compares a delayed/stale pose against the current true
kinematic pose, the observed current-truth error can exceed the random-noise
clamps. In this replay, observed maxima were:

```text
max current-truth pose error: 1.276 map units
max current-truth yaw error: 30.228 deg
max pose age: 1.400 s
```

That does not mean the random sample clamp failed; it means the delayed sample
aged while the vehicle continued moving or drifted.

## Lost Localization State Machine

The loss handling is based on hloc update age, not control-frame count.

```text
max pose age before stale handling: 0.6 s
short loss stage: age/loss duration <= 1.0 s
medium loss stage: 1.0 s < age/loss duration <= 8.0 s
long loss stage: age/loss duration > 8.0 s
```

Behavior:

```text
short_hover_hold: zero PCMD hover / hold current command context
medium_wait_relocalize: stop forward progress and wait for localization recovery
long_manual_handoff: ABORT_OR_MANUAL manual handoff model
```

Latest replay:

```text
total stale localization time: 11.3 s
stale frames: 226
short_hover_hold frames: 133
medium_wait_relocalize frames: 93
long_manual_handoff frames: 0
```

No single outage lasted long enough to hit the 8.0 s long-loss handoff.

## Final Landing Gate

The final landing decision no longer uses ordinary waypoint arrival alone.

```text
ordinary waypoint arrival radius: 1.0 map unit
final landing radius: 0.8 map unit
final landing hold: 0.5 s
final max estimated speed over hold window: 1.5 map units/s
final max yaw-estimate span over hold window: 20.0 deg
pose must be valid and not stale
```

Latest final gate sample:

```text
landing gate ready at: t=167.85 s
estimated final distance: 0.627 map units
hold time: 0.500 s
estimated speed over hold: 1.356 map units/s
yaw span over hold: 5.231 deg
pose age: 0.300 s
```

Final true position relative to W1:

```text
final true position: x=-6.001, y=-3.523, z=0.058
W1 position: x=-6.400, y=-3.533, z=0.025
3D distance to W1: 0.401 map units
horizontal distance to W1: 0.400 map units
vertical map-unit error to W1: 0.010 map units
```

## Wind Gust Error

Wind gust is applied to the true kinematic plant position, not to the controller
pose estimate. It does not change yaw.

```text
wind seed: 20018
wind interval: every 5.0 s
wind magnitude: 0.5 map units
directions: forward, back, right, left, up, down
direction frame: relative to current true drone body yaw
yaw effect: none
observed wind events: 33
```

## Latest Replay Result

Summary:

```text
clean-run pass: false
reason: telemetry_stale_11.3s
completed route: true
terminal mode: COMPLETED
controller route progress: 100.0%
sim duration: 168.85 s
frames: 3378
raw ticks: 3378
converged: true
time to converge: 2.10 s
```

`pass=false` is expected for this deliberate outage run because the existing
metric gate still treats more than 2.0 s total stale telemetry as a clean-run
failure. It is not a route abort: the controller completed the out-and-back
inspection route after repeated localization dropouts.

Tracking metrics:

```text
mean cross-track: 0.263 map units
p90 cross-track: 0.604 map units
max cross-track: 0.851 map units
yaw flips per minute: 3.909
stop-and-go score: 1.197
```

Tube metrics:

```text
max true tube distance: 1.033 map units
max controller-estimated tube distance: 1.523 map units
manual handoff from tube exit: no
```

Interpretation: the true vehicle briefly exceeded the 1.0-map-unit tube by
about 0.033 map units, and the delayed/noisy controller estimate sometimes
appeared farther outside. The outside-tube duration/update debounce did not
reach both thresholds, so the run completed without a false manual handoff.

## Absolute Map-Unit Thresholds

These values are route/map coordinate units, not physical meters. In Sphinx,
one simulator unit is metric telemetry. In real hloc/SfM flight, the same
numbers mean arbitrary map units such as `1 map unit` or `2 map units`; they
must not be interpreted as physical meters without calibration.

| use | value | scale dependency |
|---|---:|---|
| map square side length | 20.0 map units | browser coordinate extent |
| map margin used for generated waypoints | 2.0 map units | route placement in map coordinates |
| start sampling radius | 1.5 map units | start placement in map coordinates |
| actual start offset from W1 | 0.911 map units | start placement in map coordinates |
| 3D waypoint arrival radius | 1.0 map unit | arrival decision in map coordinates |
| final landing radius | 0.8 map unit | landing decision in map coordinates |
| final landing max estimated speed | 1.5 map units/s | landing decision in map coordinates and time |
| route tube radius | 1.0 map unit | safety decision in map coordinates |
| route tube segment window | active segment +/- 1 | topology window, not metric scale |
| route tube outside-time debounce | 0.8 s | time debounce |
| route tube effective update debounce | 6 pose updates | hloc/update-count debounce |
| route tube initial rejoin grace | 10.0 s | time debounce |
| position estimate Gaussian sigma | 0.25 map units per axis | pose-noise model in map coordinates |
| position estimate clamp at sampling time | 1.0 map unit 3D resultant | pose-error cap in map coordinates |
| camera/body yaw noise sigma | 5.0 deg | angular estimate-noise model |
| camera/body yaw clamp at sampling time | 20.0 deg | angular estimate-error cap |
| telemetry delay | 200 +/- 100 ms | time latency model |
| hloc outage interval | 10.0 s | time localization-failure model |
| hloc outage duration | 1.0 s | time localization-failure model |
| stale pose age limit | 0.6 s | time localization-loss gate |
| short loss threshold | 1.0 s | time localization-loss gate |
| long loss/manual timeout | 8.0 s | time localization-loss gate |
| wind gust interval | 5.0 s | time model |
| wind gust impulse | 0.5 map units | plant disturbance in map coordinates |
| W6/W7 pole horizontal offset | 1.2 map units route-right | inspection geometry in map coordinates |
| pole top above waypoint height | 2.0 map units | inspection geometry in map coordinates |
| ideal success cross-track threshold | 0.5 map units | reporting threshold in map coordinates |
| max allowed cross-track threshold | 3.0 map units | reporting/fail threshold in map coordinates |
| trial hard-stop cross-track guard | 8.5 map units | run-stop guard in map coordinates |
| minimum segment time | 0.6 s | time gate before waypoint switch |
| maximum segment time | 90.0 s | time fail/escape gate |
| waypoint hover time | 1.0 s | time gate at waypoint |
| waypoint yaw-align tolerance | 12.0 deg | angular waypoint gate |
| waypoint yaw-align timeout | 6.0 s | time escape gate |

Control quantities below are not map-distance thresholds:

```text
PCMD caps: pitch <= 10%, roll <= 6%, yaw <= 25%, gaz <= 15%
translational resultant cap: max_horiz_translate <= 10% PCMD
PCMD rate limit: 200 percentage-points/s
```

## Current Interpretation

Under the current browser sandbox assumptions, the `translational_waypoint`
controller completed the 27-point out-and-back pole-inspection route while
using only noisy, delayed, periodically missing estimated pose/yaw input.

This supports the path-control geometry under the simplified kinematic model.
It does not prove ANAFI firmware behavior. Firmware/control-state validation
still requires the Sphinx/Olympe backend.

## Open Design Risks Before hloc / Real ANAFI

| risk | why it matters | next check |
|---|---|---|
| kinematic plant mismatch | real ANAFI velocity response has attitude lag, inertia, braking delay, and firmware state constraints | validate the same controller through Sphinx/Olympe |
| hloc confidence | low-match or high-error poses should not trigger tube abort or command correction | feed match count, inlier ratio, reprojection error, pose jump, and timestamp age into the state machine |
| delayed pose near obstacles | 200-300 ms latency can make the controller believe it is inside the tube while truth has drifted | use tighter speed caps or larger safety margins around obstacles |
| 1-map-unit tube with 1-map-unit pose clamp | estimated pose can appear outside the tube even when truth is near the route | tune tube radius/debounce from real hloc error statistics |
| vertical inspection segments | vertical climb/descent has little horizontal segment length, so a tube alone is not a full hover safety volume | add explicit inspection cylinders and obstacle clearance checks |
| pole geometry | current poles are visual targets, not collision/no-fly objects | add explicit no-fly cylinder clearance logic |
| camera/body/gimbal extrinsics | hloc camera yaw may not equal body yaw when gimbal or camera optical axis has an offset | calibrate camera-to-body yaw/pitch before using yaw-align numbers |
| wind model | current wind is an instantaneous 0.5-map-unit position impulse, not continuous force or velocity disturbance | test continuous drift/velocity bias and gust recovery |
| no absolute scale in real hloc | browser values are map units; real monocular SfM can be arbitrary scale | keep thresholds as map-unit values or apply a calibrated scale factor |

## Verification

Latest full test command:

```bash
/home/allen/localization/.venv/bin/python -m pytest -q tests
```

Latest full test result:

```text
167 passed
```

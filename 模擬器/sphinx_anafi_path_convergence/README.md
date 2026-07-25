# Sphinx ANAFI path-convergence experiment

**SPHINX ANAFI SIMULATION ONLY.** Isolated experiment: can the autonomous
controller take off from a random point within 5 m of the route start and
smoothly converge onto the planned waypoint route, given only simulator
telemetry pose? Nothing here validates visual localization, map quality, or
real-flight safety, and nothing here changes production flight code.

```text
Sphinx validation scale: meters
Real monocular SfM map scale: arbitrary units
Real altitude scale: arbitrary map units, not meters
```

No threshold tuned here (meters) may be copied into the real map's
arbitrary-unit thresholds without a calibrated scale factor (`scale_utils.py`).

## Layout

| file | what |
|---|---|
| `run_sphinx_anafi_convergence.py` | Monte Carlo harness (Sphinx or kinematic backend) |
| `anafi_profile.py` | single ANAFI white-paper capability/sensor/video profile |
| `route_geometry.py` | polyline projection, arclength, adaptive lookahead, route patterns |
| `controllers.py` | 10 comparable algorithms over shared controller machinery |
| `telemetry_sources.py` | Olympe/Sphinx source, kinematic plant, perturbation wrapper, IP guard |
| `metrics.py` | convergence/smoothness/stability metrics, pass criteria, aggregation |
| `scale_utils.py` | meters <-> map-units context + calibration helpers |
| `configs/` | example run configs |
| `tests/` | pure-Python tests (no Sphinx required) |
| `outputs/` | generated logs (gitignored) |

## Run

Pure-Python tests (no Sphinx needed):

```bash
/home/allen/localization/.venv/bin/pytest -q \
  sfm_system/定位/experiments/sphinx_anafi_path_convergence/tests
# expected: 167 passed
```

The three runnable surfaces are deliberately different:

| surface | command | fidelity / purpose |
|---|---|---|
| Parrot Sphinx SIL | launcher below, then `--backend sphinx` | official Gazebo software-in-the-loop using ANAFI PC firmware; required firmware/control validation |
| kinematic batch | `--backend kinematic` | deterministic first-order approximation for paired Monte Carlo/CI; no firmware or aerodynamics |
| browser replay | `make_browser_sandbox.py` | renders a precomputed kinematic run; visualization only, not a live simulator |

Start the real Sphinx ANAFI software-in-the-loop stack in a separate terminal:

```bash
sfm_system/定位/mission/flight_control/launch_sphinx_anafi_empty.sh --check
sfm_system/定位/mission/flight_control/launch_sphinx_anafi_empty.sh
# wait for [ready] and "All drones instantiated"; retry the full stack if
# Sphinx reports "All drones dropped"
```

On this machine the launcher intentionally uses the exact offscreen UE4 binary
with `-RenderOffScreen`; the plain `parrot-ue4-empty` launcher is unstable on
the installed RTX 5060/driver. It starts/checks `firmwared`, waits up to 45 s
for both `All drones instantiated` and `10.202.0.1`, and cleans up both child
process groups with bounded TERM then KILL fallback. The default firmware URL
contains `#latest` and is not pinned. Launcher logs and Sphinx summaries record
the observed Sphinx, firmware and Olympe versions; exact reproduction must not
be claimed until the firmware image is pinned by content hash/revision.

Compare algorithms in Sphinx ANAFI simulation:

```bash
python3 sfm_system/定位/experiments/sphinx_anafi_path_convergence/run_sphinx_anafi_convergence.py \
  --backend sphinx \
  --pattern s_curve --start-radius-m 5.0 --num-trials 20 \
  --initial-yaw-mode route --random-yaw-error-deg 15 \
  --rejoin-algorithm compare \
  --duration 60 --random-seed 42 \
  --out sfm_system/定位/experiments/sphinx_anafi_path_convergence/outputs/algorithm_compare_scurve
```

Height-changing route:

```bash
python3 .../run_sphinx_anafi_convergence.py --pattern s_curve --height-amp-m 1.5 \
  --start-radius-m 5.0 --num-trials 20 --rejoin-algorithm adaptive_lookahead \
  --horizontal-control-mode hybrid --duration 60 --random-seed 42 \
  --out .../outputs/height_profile_test
```

Telemetry perturbation:

```bash
python3 .../run_sphinx_anafi_convergence.py --pattern s_curve --start-radius-m 5.0 \
  --num-trials 20 --rejoin-algorithm adaptive_lookahead --horizontal-control-mode hybrid \
  --pose-noise-m 0.2 --yaw-bias-deg 10 --telemetry-delay-ms 150 --telemetry-drop-rate 0.05 \
  --duration 60 --random-seed 42 --out .../outputs/noisy_telemetry_test
```

Scale-aware reporting:

```bash
python3 .../run_sphinx_anafi_convergence.py --pattern line --start-radius-m 5.0 \
  --num-trials 20 --rejoin-algorithm adaptive_lookahead --map-units-per-meter 2.4 \
  --out .../outputs/scale_report_example
```

No Sphinx available? Wide sweeps use the clearly labeled kinematic
approximation (`--backend kinematic`); Sphinx remains the ANAFI-dynamics
validation step. If Sphinx is not running, `--backend sphinx` fails with a
clear message; the tests above still run.

Kinematic compare mode is paired: each algorithm receives the same scenario
ID, random seed, start quadrant/offset, yaw error and perturbation. Sphinx uses
the same matched inputs in scenario-major order and cyclically rotates the
first algorithm, but it cannot reset firmware, world, battery or thermal state
between trials. Sphinx summaries therefore expose only `exploratory_ranking`,
never a physically paired ranking or recommended algorithm. Every Sphinx
`moveBy` expectation and measured start position/yaw must pass tolerance before
a trial is valid.

Do not apply a global `horizontal_control_mode` in the comparison config
because that would erase lateral versus nose-first variants. Explicit CLI
flags still win over JSON config, including `--flag=value` syntax. Numeric CLI
and JSON config values are rejected when non-finite, wrongly typed or outside
their benchmark bounds; `num_trials` must be positive.

## ANAFI profile and modeling boundary

`anafi_profile.py` is the single source for the public white-paper values used
by the experiment: 320 g; 15 m/s horizontal; 4 m/s ascent/descent; 200 deg/s
maximum angular speed; 50 km/h wind and 80 km/h gust figures; 1 m takeoff
hover; 1.5 cm hover accuracy at 1 m; 200 Hz internal control loop; 720p30,
5 Mbps video with 280 ms end-to-end video latency; gimbal pitch +/-90 deg at up to 180 deg/s;
GPS position/speed standard deviations 1.2 m/0.5 m/s; and 0.2 m barometer
noise. Source: [Parrot ANAFI white paper v1.4](https://www.parrot.com/assets/s3fs-public/2020-07/white-paper_anafi-v1.4-en.pdf).

The kinematic plant uses the speed/angular capability values and 200 Hz
integration substeps, and embeds the complete profile in output metadata. It
does not model mass, aerodynamics, wind resistance, sensors, gimbal, codec, or
firmware. Those values are metadata/test inputs unless an explicit fault model
uses them. Controller caps remain pitch<=10%, yaw<=25%, gaz<=15%, roll<=6%.

Useful extras: `--yaw-sweep` (0..±90° buckets), `--perturbation-suite`,
`--speed-schedule cos|threshold|smoothstep`, `--auto-yaw-calibration`,
`--config configs/default.json`, `--csv`.

## Browser sandbox

For fast visual iteration on the waypoint algorithm, generate a self-contained
browser replay from the same Python controller code and the kinematic plant:

```bash
cd sfm_system/定位/experiments/sphinx_anafi_path_convergence
/home/allen/localization/.venv/bin/python make_browser_sandbox.py \
  --algorithm translational_waypoint --route-style complex \
  --num-waypoints 10 --return-to-start --map-size-m 20 --duration 220 --seed 7 \
  --inspection-poles --inspection-pole-waypoints 6,7 \
  --inspection-pole-right-offset-m 1.2 --inspection-pole-top-above-waypoint-m 2.0 \
  --pose-source noisy_estimated --pose-noise-m 0.25 --pose-error-max-m 1.0 \
  --video-e2e-latency-ms 280 --decode-localization-latency-ms 0 \
  --telemetry-delay-jitter-ms 100 \
  --hloc-outage-interval-s 10 --hloc-outage-duration-s 1.0 --hloc-outage-start-s 10 \
  --camera-yaw-noise-deg 5.0 --camera-yaw-error-max-deg 20.0 \
  --wind-gust-interval-s 5.0 --wind-gust-m 0.5 \
  --arrival-radius 1.0 \
  --route-tube-radius 1.0 --route-tube-exit-s 0.8 --route-tube-exit-updates 6 \
  --max-pose-age-s 0.6 --pose-loss-short-s 1.0 --lost-abort-s 8.0 \
  --final-landing-radius 0.8 --final-landing-hold-s 0.5 \
  --final-landing-max-est-speed 1.5 --final-landing-yaw-stable-deg 20.0
xdg-open outputs/browser_sandbox/index.html

# Optional local URL:
(cd outputs/browser_sandbox && python3 -m http.server 9002 --bind 127.0.0.1)
# then open http://localhost:9002/
```

The HTML shows a 3-D 20 x 20 route/map-unit square with 10 preplanned
waypoints. These browser distances are route/map coordinate units, not physical
meters for hloc/SfM. The horizontal x/z projection is a mostly straight route
with only mild lateral weave, while each waypoint height is randomly sampled
from the run seed. With
`--return-to-start`, the controller route expands the 10 planned points into
`W1 -> ... -> W10 -> W9 -> ... -> W1`: at W10 the drone hovers, yaw-aligns the
camera/body toward W9, then repeats the same translational waypoint algorithm
back to W1 before completing. W6 and W7 each have a right-side cylindrical
inspection pole. At each encounter with W6 or W7, including the return leg, the
route inserts `Wn pole top` and `Wn return height`: the drone hovers,
yaw-aligns the camera/body to the pole, climbs in place to the pole-top height,
descends in place to the original waypoint height, then continues. The HTML also
shows a translucent route safety tube, drone path, the controller's estimated
pose, current target waypoint, body heading, body-frame command vector, PCMD,
cross-track, route progress, `move_mode`, and wind gust events. Use left-drag to
rotate, mouse wheel to zoom, and right/middle-drag to pan. Re-run
`make_browser_sandbox.py` after editing `controllers.py`. This is a
visualization/debug frontend only; it does not simulate Parrot ANAFI firmware.
Use Sphinx/Olympe for firmware/control state validation.

Pose input: browser replay defaults to `--pose-source noisy_estimated`, so the
controller receives a current-position estimate with bounded random 3-D error
inside `--pose-error-max-m 1.0`; the default shape is Gaussian xyz noise with
`--pose-noise-m 0.25`, clamped to that 1-map-unit maximum. The true kinematic pose is
added only after command generation for display and scoring. The controller's
camera/body yaw estimate also gets Gaussian noise `--camera-yaw-noise-deg 5.0`,
clamped to `--camera-yaw-error-max-deg 20.0`. Use `--pose-source truth` only
for ideal controller-only debugging. The default tube radius is kept at
`--route-tube-radius 1.0` map unit.

Hloc latency/outage: the white-paper 280 ms is video end-to-end latency only,
so it is a lower bound before decode plus MegaLoc/XFeat/PnP processing. The
browser total is `--video-e2e-latency-ms` plus a measured
`--decode-localization-latency-ms`; leaving the latter at zero sets metadata
`telemetryDelayIsLowerBound=true`. `--telemetry-delay-ms` remains an explicit
synthetic total-delay override. `--telemetry-delay-jitter-ms 100` adds jitter,
and
`--hloc-outage-interval-s 10 --hloc-outage-duration-s 1.0` simulates complete
localization loss for 1 second every 10 seconds. The controller gates stale pose
by hloc update age, not control-frame count: `--max-pose-age-s 0.6` enters
lost-localization handling, `--pose-loss-short-s 1.0` separates short hover hold
from medium relocalization wait, and `--lost-abort-s 8.0` models long-loss
manual handoff.

Waypoint arrival: `translational_waypoint` treats each waypoint as reached when
the controller-estimated 3-D distance to that waypoint is within
`--arrival-radius 1.0` map unit. The same rule applies on the outbound and
return legs.

Final landing: the last W1 return does not land only because the noisy estimated
pose touched the ordinary waypoint radius. It also requires
`--final-landing-radius 0.8`, `--final-landing-hold-s 0.5`, estimated speed no
greater than `--final-landing-max-est-speed 1.5`, yaw span no greater than
`--final-landing-yaw-stable-deg 20.0`, and a valid non-stale pose.

Wind input: `--wind-gust-interval-s 5.0 --wind-gust-m 0.5` applies a position
only drift impulse every five seconds. Each impulse picks one of
forward/back/left/right/up/down relative to the current drone body yaw and moves
the true kinematic position by 0.5 map units. It does not change the drone yaw.

Tube safety: `--route-tube-radius 1.0` checks full-3D distance in route/map
units, but only against the active segment +/- `--route-tube-segment-window`
segments so a nearby wrong branch in a U-turn/self-crossing route is not
accepted. Because hloc can have brief pose glitches and the browser model allows
up to 1 map unit pose error, abort requires both `--route-tube-exit-s 0.8` continuous
outside-tube time and `--route-tube-exit-updates 6` effective pose updates. A
repeated stale hloc pose
does not count as a new update. `--route-tube-initial-grace-s` allows initial
rejoin from outside the tube until the drone first enters the tube or the grace
expires. In this experiment `ABORT_OR_MANUAL` means zero PCMD hover / manual
handoff model, not `Emergency`.

Hloc confidence gate: if a pose includes `valid=False`, too few matches, too low
an inlier ratio, too high reprojection error, or too large a pose jump, the
controller treats it like a temporary localization loss: zero PCMD hover, no tube
counter update, and manual handoff only after `lost_abort_s`.

## Algorithms compared

`naive_waypoint`, `continuous_path` (production-style full-polyline
FOLLOW/REJOIN that chases the nearest point), `segment_corridor`,
`segment_corridor_hover`, `segment_corridor_lateral`,
`segment_corridor_nose_first`, `adaptive_lookahead` (adaptive lookahead +
hysteresis + anti-oscillation), `adaptive_smoothed` (+ filtered carrot),
`translational_waypoint` and `translational_smoothed` (see below).
See `outputs/*/summary.json` -> `ranking` for kinematic results. Sphinx output
uses `exploratory_ranking` and deliberately leaves `recommended_algorithm`
unset because sequential physical trials are not independent.

### Yaw-locked translational controller (`translational_waypoint`)

The drone does **not yaw while flying a segment**. It reaches the next waypoint
by pure body-frame translation, combining forward/back (pitch), left/right
(roll) and up/down (gaz) velocity components; the body-frame direction to the
target maps to one of **14 movement modes** (6 axis faces + 8 3D corners:
`forward/back/left/right/up/down` and `forward_up_right` … `back_down_left`),
logged per tick as `move_mode`. The mode is a label for analysis/debug; PCMD is
computed from continuous body-frame components, with the normalized 3-D command
resultant capped so diagonal modes do not stack full per-axis commands. It only
**yaws while hovering at a waypoint**, to face the next one (the yaw angle is the
in-XY-plane angle between the Pi→Pi+1 segment and the camera heading). Because
its motion is not nose-first, it uses the drone's actual body yaw directly
rather than the motion-refined map-yaw offset (real GLOMAP integration then
needs a one-time constant NED→map yaw calibration). Run it with
`--rejoin-algorithm translational_waypoint`.

## Safety

Default IP `10.202.0.1`; only valid IPv4 addresses inside `10.202.0.0/16` are
accepted. Real ANAFI/SkyController IPs and prefix lookalikes are refused, with
no override.
No Emergency anywhere; zero PCMD before Landing; land+disconnect in `finally`;
PCMD clamped to [-100,100] with conservative caps (pitch<=10%, yaw<=25%,
gaz<=15%, roll<=6%; nose-first controllers keep roll 0 unless lateral assist is
enabled, while the translational controller uses bounded roll as a primary axis).

## What this validates / does not validate

Validates: route-convergence control geometry and stability using the
simulated Parrot ANAFI (Sphinx/Olympe fused telemetry pose), under initial
offset (<=5 m), initial yaw error, and injected telemetry faults.

Does NOT validate: visual localization (XFeat/SfM/GLOMAP/hloc), monocular map
metric scale, raw IMU, GPS-denied dead reckoning, obstacle avoidance, wind,
real-flight safety, or any real-map threshold values (arbitrary units,
calibration required).

## Next-stage blocker: localization-in-the-loop

The current Sphinx task uses firmware-fused telemetry as controller input. It
does not include a textured field scene, camera exposure/stream validation,
720p video transport, frame/decode timestamps, MegaLoc retrieval, XFeat/PnP,
or localization confidence feeding the controller. The next required stage is:

```text
textured UE scene -> ANAFI 720p30 stream -> timestamped decode
-> MegaLoc/XFeat/PnP -> freshness/confidence gates -> route controller
-> independent Gazebo pose evaluation
```

Until that closed loop exists and has an independent simulator pose channel,
this experiment cannot validate the user's localization algorithm or claim
readiness for the real site.

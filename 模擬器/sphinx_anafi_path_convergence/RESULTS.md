# Sphinx ANAFI path-convergence — results & recommendation

**Scope reminder.** This validates ONLY route-convergence control using the
Parrot ANAFI in Parrot Sphinx / Olympe simulator telemetry. It does NOT
validate visual localization, hloc, SfM/GLOMAP map quality, XFeat/LightGlue/
MegaLoc, raw IMU, real ANAFI sensor fusion, GPS-denied dead reckoning,
monocular map metric scale, physical altitude in meters for the real map,
obstacle avoidance, wind, or real-flight safety. No claim of real-flight
safety is made.

**Methodology update, 2026-07-10.** The runner now reuses the same scenario
ID, seed, start, yaw error and perturbation for every algorithm, and the
kinematic plant references `anafi_white_paper_v1_4`. Older cross-algorithm
tables later in this document were produced before those changes. They remain
historical diagnostics, but their rankings are not paired comparisons and must
not be used as the current recommendation. Individual Sphinx connectivity,
takeoff, telemetry and landing evidence remains valid, but the fused telemetry
channel used there was not independent simulator ground truth.

```text
Sphinx validation scale: meters
Real monocular SfM map scale: arbitrary units
Real altitude scale: arbitrary map units, not meters
```

---

## 0. Current paired clean baseline (2026-07-10)

Command: `configs/algorithm_compare.json` with explicit
`--backend=kinematic`. This ran 20 shared scenarios per algorithm, 200 trials
total, using the white-paper profile and conservative controller PCMD caps.
All 200 trials passed and completed the route. The retained generated summary
and trial aggregates are under `outputs/paired_profile_baseline_20260710/`;
the 203 MB per-tick log was intentionally pruned.

| rank | algorithm | score | mean cross-track m | mean convergence s | yaw flips/min |
|---:|---|---:|---:|---:|---:|
| 1 | adaptive_lookahead | 157.91 | 0.0362 | 2.935 | 3.465 |
| 2 | segment_corridor_nose_first | 157.54 | 0.0391 | 2.860 | 4.136 |
| 3 | translational_waypoint | 157.51 | 0.0974 | 2.982 | 1.938 |
| 4 | segment_corridor_lateral | 157.37 | 0.0357 | 2.945 | 4.327 |
| 5 | segment_corridor_hover | 157.26 | 0.0397 | 2.980 | 4.468 |
| 6 | adaptive_smoothed | 157.08 | 0.0343 | 2.895 | 5.162 |
| 7 | translational_smoothed | 156.73 | 0.1782 | 2.958 | 1.922 |
| 8 | naive_waypoint | 153.64 | 0.1546 | 3.395 | 9.133 |
| 9 | segment_corridor | 153.57 | 0.0619 | 2.828 | 11.352 |
| 10 | continuous_path | 153.34 | 0.0669 | 2.305 | 11.218 |

The clean kinematic scores are close and are not evidence that any controller
is ready for real flight. `adaptive_lookahead` leads this composite score;
`translational_waypoint` has fewer yaw reversals but higher cross-track error.
Paired perturbation sweeps plus counterbalanced Sphinx evaluation with an
independent pose reference are still required for a flight-controller choice.

## 1. Files changed / added

All new code is isolated under
`sfm_system/定位/experiments/sphinx_anafi_path_convergence/`:

```
README.md  RESULTS.md  anafi_sphinx_notes.md  .gitignore
anafi_profile.py   run_sphinx_anafi_convergence.py   make_browser_sandbox.py
route_geometry.py   controllers.py
telemetry_sources.py   metrics.py   scale_utils.py
configs/{default,algorithm_compare,height_profile,noisy_telemetry,translational}.json
tests/{test_anafi_profile_and_fairness,test_sphinx_freshness_and_setup,
       test_route_geometry,test_controllers,test_scale_utils,test_metrics,
       test_anafi_safety_guards,test_telemetry_perturbation}.py + conftest.py
outputs/.gitkeep   (generated logs are gitignored)
```

## 2. Isolation

Confirmed. Every new file lives under the experiment folder; frame/PCMD
conventions were re-derived in-folder (not imported) so the experiment cannot
drift production and vice-versa.

## 3. Simulated aircraft — Parrot ANAFI in Sphinx

Confirmed working (Parrot Sphinx 2.25.2, `anafi.drone` + `anafi-pc.ext2.zip`
firmware, Olympe 8.4.0). Verified live: Sphinx loaded the ANAFI ("All drones
instantiated", "Boot uuid"); Olympe connected at `10.202.0.1`; `TakeOff` →
hovering; fused telemetry flowed (`PositionChanged` ~1.2 Hz, `AttitudeChanged`/
`SpeedChanged` ~5.3 Hz); a forward-pitch PCMD at fused yaw ≈ 1.60 rad moved the
drone +0.49 m east / −0.02 m north (correct sign on real ANAFI dynamics); the
harness ran end-to-end against Sphinx (takeoff → random-start `moveBy` → 20 Hz
control loop → per-tick logging → land), whose logs surfaced a real bug (below).

**Real Sphinx confirmation added 2026-07-09.** With the offscreen UE4 renderer,
the harness completed a clean 2-trial `translational_waypoint` run
(`outputs/sphinx_translational`) and a 20-trial `--rejoin-algorithm compare`
run covering all 10 controllers (`outputs/sphinx_compare`). The targeted
`translational_waypoint` run passed 2/2 with `backend: "sphinx"`; the compare
run historically ranked `translational_waypoint` first. That run used different
scenario order/random inputs by algorithm, did not validate actual `_goto`
start error, and shared sequential firmware/world/battery state. Its ranking is
not a fair physical comparison. Large `ticks.jsonl` logs were pruned;
`outputs/sphinx_compare/trajectory_check.svg` is a fused-telemetry top-down
diagnostic, not an independent ground-truth trajectory.

**Environment limitation (honest).** The Parrot UE4 renderer segfaults on this
machine's RTX 5060 Laptop (Blackwell, driver 580.159.04). `-RenderOffScreen`
avoids the segfault and was used for the real Sphinx runs above, but it provides
no normal UE4 viewport. This is still useful for control validation because the
harness uses Sphinx/Olympe fused telemetry; it does not provide independent
truth metrics and does not visually
validate scene rendering, hloc, camera images, or real visual localization.

## 4. Companion simulator files changed

`mission/flight_control/launch_sphinx_anafi_empty.sh` now uses the verified
offscreen UE4 path with readiness checks and process-group cleanup.
`mission/flight_control/sphinx_path_follow_smoke.py` now enforces the simulator
network with `ipaddress`. No real-flight arming entrypoint was added or used.

## 5. Candidate algorithms implemented (ten)

Over one feature-flagged corridor state machine plus a dedicated
yaw-locked translational controller (states: INIT_REJOIN, SEGMENT_ALIGN,
SEGMENT_FOLLOW, SEGMENT_REJOIN, WAYPOINT_HOVER, NEXT_SEGMENT,
LOST_OR_UNCERTAIN, ABORT_OR_MANUAL):

1. `naive_waypoint` — straight at the next waypoint (nose-first)
2. `continuous_path` — production-style full-polyline FOLLOW/REJOIN (nearest-point chaser)
3. `segment_corridor` — active-segment corridor, fixed lookahead, nearest-point REJOIN
4. `segment_corridor_hover` — + waypoint hover + yaw re-align
5. `segment_corridor_lateral` — + bounded lateral roll assist
6. `segment_corridor_nose_first` — + nose-first lookahead REJOIN (pure pursuit)
7. `adaptive_lookahead` — + adaptive lookahead + hysteresis + anti-oscillation
8. `adaptive_smoothed` — + low-pass on control pose/heading + rate-limited carrot
9. **`translational_waypoint`** — yaw-locked body translation, yaw only at waypoints (see below)
10. **`translational_smoothed`** — 9 + control-pose low-pass

### The yaw-locked translational controller (user-requested design)

The drone **never yaws while flying a segment**. It reaches the next waypoint by
pure body-frame translation, combining forward/back (pitch), left/right (roll)
and up/down (gaz) velocity components; the body-frame direction to the target
maps to one of **14 movement modes** — 6 axis faces (forward/back/left/right/
up/down) + 8 3-D corners (forward_up_right … back_down_left) — logged per tick
as `move_mode`, with the continuous per-axis components applied. On arrival it
**hovers**, then **yaws (only here)** to face the next waypoint: the yaw angle is
the in-XY-plane angle between the Pi→Pi+1 segment and the camera-heading ray.
Because its motion is not nose-first, it uses the drone's actual fused body yaw
directly (the motion-refined map-yaw offset would be invalid); real GLOMAP
integration needs a one-time constant NED→map yaw calibration instead.

Optional route-tube safety is now implemented for hloc-style pose streams:
`route_tube_radius` checks full-3D distance to the active segment window,
`route_tube_exit_s` plus `route_tube_exit_updates` debounce hloc glitches, and
low-confidence hloc poses hold zero PCMD without triggering tube abort. The
Sphinx tables below were measured before enabling this stricter optional safety
gate.

## 6. Current algorithm decision

There is no Sphinx-backed recommendation yet. The current paired kinematic
baseline ranks `adaptive_lookahead` first and `translational_waypoint` third,
with small score separation. Sphinx now uses matched scenario inputs,
scenario-major cyclic order and verified starts, but sequential trials cannot
reset firmware/world/battery state and use no independent ground truth. New
Sphinx summaries therefore publish only `exploratory_ranking` and leave
`recommended_algorithm` unset.

## 7. Why

The targeted historical Sphinx run shows that `translational_waypoint` can
execute against ANAFI firmware telemetry, but it does not establish superiority.
Its main tradeoff remains fewer yaw reversals versus higher paired-kinematic
cross-track error and dependence on accurate NED→map yaw calibration.

## 8. Sphinx ANAFI commands used

Launch: `sphinx anafi.drone::firmware=…anafi-pc.ext2.zip` +
`UnrealApp-Linux-Shipping Empty -RenderOffScreen`. Olympe: `TakeOff`,
`PCMD(1,roll,pitch,yaw,gaz,0)`, `moveBy` (trial setup only), `Landing`; states
`FlyingStateChanged`, `PositionChanged`, `AttitudeChanged`, `SpeedChanged`. No
`Emergency`. Zero PCMD before `Landing`; land + disconnect in `finally`.

## 9–18. Monte Carlo results and Sphinx confirmation

**Sphinx targeted confirmation** — line route, 3.0 m segments, start ≤ 5 m, yaw
±15°, 2 trials, 22 s/trial, seed 5 (`outputs/sphinx_translational`):

| algorithm | backend | pass | complete | conv (s) | route progress | segment switches | ct mean (m) | ct p90 (m) | ct max (m) | vertical mean (m) | yaw flips/min | stop&go | hard aborts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **translational_waypoint** | sphinx | **1.00** | 0.00 | 11.2 | 0.83 | 2.0 | 0.238 | 0.407 | 1.952 | 0.095 | 0.0 | 0.190 | 0 |

This was the historical targeted confirmation run for the requested controller.
Both trials passed strict convergence/corridor criteria with real Sphinx/Olympe
telemetry. The 22 s window did not complete the full 9 m route in this targeted
run, but it did advance through two segment switches on average with no aborts.

**Archived unpaired Sphinx all-controller compare** — same line route, 2 trials/controller, 20
trials total, 22 s/trial, seed 5 (`outputs/sphinx_compare`). Low `n=2`, unmatched
scenario/order bias, unverified setup and fused-reference scoring mean this is
only a historical lifecycle/control diagnostic:

| algorithm | score | pass | complete | conv (s) | route progress | ct mean (m) | ct p90 (m) | vertical mean (m) | yaw flips/min | hard aborts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **translational_waypoint** | **133.76** | **1.00** | 0.50 | 3.5 | 0.92 | **0.278** | **0.503** | **0.038** | **0.0** | 0 |
| adaptive_lookahead | 92.74 | 0.50 | 1.00 | 0.0 | 0.98 | 0.590 | 1.059 | 0.043 | 2.7 | 0 |
| continuous_path (prod. baseline) | 69.28 | 0.50 | 0.50 | 0.0 | 0.65 | 0.511 | 0.877 | 0.049 | 5.3 | 0 |
| segment_corridor_nose_first | 63.21 | 0.50 | 0.50 | 0.0 | 0.78 | 1.384 | 2.101 | 0.049 | 5.9 | 0 |
| translational_smoothed | 50.62 | 0.50 | 0.00 | 9.9 | 0.83 | 0.735 | 1.884 | 0.041 | 0.0 | 0 |
| naive_waypoint (baseline) | -9.76 | 0.00 | 0.00 | 8.9 | 0.61 | 0.958 | 1.634 | 0.095 | 2.7 | 0 |
| adaptive_smoothed | -11.23 | 0.00 | 0.00 | 6.8 | 0.92 | 1.225 | 2.154 | 0.040 | 4.1 | 0 |
| segment_corridor_lateral | -15.96 | 0.00 | 0.00 | n/a | 0.65 | 1.272 | 1.958 | 0.046 | 5.5 | 0 |
| segment_corridor_hover | -16.36 | 0.00 | 0.00 | n/a | 0.39 | 1.513 | 2.258 | 0.106 | 4.1 | 0 |
| segment_corridor | -18.98 | 0.00 | 0.00 | n/a | 0.49 | 1.491 | 2.234 | 0.032 | 5.5 | 0 |

Do not use the archived table to rank algorithms. The current runner rotates
algorithm order per matched scenario, validates command expectations and actual
start position/yaw, stops on invalid setup, and still labels any resulting
physical ranking exploratory because the world/firmware/battery cannot be reset
independently between trials.

**What Sphinx does and does not prove for hloc.** These Sphinx numbers validate
the route controller given a pose stream and real ANAFI simulator dynamics. They
do **not** validate hloc, camera imagery, map matching, scale recovery, or
NED→map yaw calibration. The next better experiment is a pose-log / hloc-replay
backend that feeds timestamped hloc poses, confidence, dropouts, jumps, and
scale/yaw calibration errors into this same harness before any real closed-loop
flight.

**Kinematic statistical source** — s_curve, start ≤ 5 m, yaw ±15°, 20
trials/algo, seed 42 (`outputs/algorithm_compare_scurve`; height variant
identical ranking, `outputs/algorithm_compare_height`):

| algorithm | pass | complete | conv (s) | ct mean (m) | ct p90 (m) | yaw flips/min | stop&go |
|---|---|---|---|---|---|---|---|
| **translational_smoothed** | 1.00 | 1.00 | **3.1** | **0.035** | **0.057** | **1.5** | 0.00 |
| **translational_waypoint** | 1.00 | 1.00 | 3.1 | 0.041 | 0.089 | 1.5 | 0.03 |
| adaptive_lookahead | 1.00 | 1.00 | 5.3 | 0.043 | 0.113 | 2.4 | 0.00 |
| adaptive_smoothed | 1.00 | 1.00 | 5.4 | 0.041 | 0.098 | 2.6 | 0.00 |
| segment_corridor_lateral | 1.00 | 1.00 | 4.6 | 0.047 | 0.129 | 2.4 | 0.01 |
| segment_corridor_nose_first | 1.00 | 1.00 | 4.7 | 0.044 | 0.126 | 2.7 | 0.00 |
| segment_corridor_hover | 1.00 | 1.00 | 5.8 | 0.048 | 0.130 | 3.3 | 0.02 |
| continuous_path (prod. baseline) | 1.00 | 1.00 | 4.2 | 0.086 | 0.201 | 7.4 | 0.02 |
| segment_corridor | 1.00 | 1.00 | 4.9 | 0.085 | 0.197 | 7.5 | 0.02 |
| naive_waypoint (baseline) | 1.00 | 1.00 | 4.3 | 0.225 | 0.428 | 5.8 | 0.03 |

- **Pass rate by |yaw error|** (yaw sweep 0…±90°, `outputs/yaw_sweep`): 100% for
  every algorithm at every bucket **through ±90°** — far beyond the ±10–15°
  required. Nose-first controllers yaw onto the carrot within ~1–2 s; the
  translational controller is heading-agnostic for convergence (it strafes) and
  only cares about heading at the waypoint yaw.
- **Route completion / segment completion**: 100% for all corridor/adaptive/
  translational algorithms on the short route.
- **Average time to converge**: **3.1 s** (translational) to 5.8 s.
- **Average horizontal cross-track**: **0.035 m** (translational_smoothed);
  0.041–0.048 m for the other corridor/adaptive family; 0.086 m (continuous_path)
  and 0.225 m (naive) for the baselines.
- **Average vertical error** (height route, 1.5 m climb): 0.13–0.22 m, within the
  0.6 m arrival-vertical radius; translational lowest (0.14 m). Horizontal and
  vertical are tracked separately (a vertical error never triggers a horizontal
  REJOIN; a large vertical error scales back forward drive — see tests).
- **Oscillation / stop-and-go**: translational 1.5 flips/min (lowest, it barely
  yaws); adaptive/corridor-nose-first 2.4–3.3; baselines 5.8–7.5.

**Speed schedules** (`outputs/sched_*`, yaw ±30°): `cos`, `threshold`,
`smoothstep` all pass 1.00; `cos` marginally best. **Baselines worse?** Yes —
`naive_waypoint` ~5× the best historical candidate's cross-track; `continuous_path`
~2× cross-track and ~5× oscillation.

## 11. Telemetry-perturbation robustness — the key trade-off

Suite = clean, yaw-bias 5/10/15/30°, pose-noise 0.1/0.2 m, delay 100/200 ms,
dropout 5% (10 trials each, `outputs/noisy_*`):

| algorithm | overall | yaw-bias ≤15° | yaw-bias 30° | delay ≤200 ms | dropout 5% | pose-noise 0.1 m | 0.2 m | flips/min |
|---|---|---|---|---|---|---|---|---|
| **translational_smoothed** | 0.90 | 1.00 | route done, wide track* | 1.00 | 1.00 | **1.00** | **1.00** | 1.5 |
| translational_waypoint | 0.90 | 1.00 | route done, wide track* | 1.00 | 1.00 | **1.00** | **1.00** | 1.5 |
| adaptive_smoothed | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 | 0.00 | 6.0 |
| adaptive_lookahead | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 11.2 |
| continuous_path | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 33.5 |

\* Under 30° yaw bias the translational controller **still completes the route
and reaches every waypoint (10/10)**, but its inter-waypoint tracking widens to
~0.41 m cross-track (a curved approach), missing the strict 0.5 m-corridor pass
criterion. Degraded, not broken.

**The trade-off (clear and complementary):**

- **Translational is immune to position noise** (0.1 m *and* 0.2 m white
  per-tick noise both 100%): it uses fused yaw directly and barely yaws, so
  there is no motion-heading jitter to amplify. It is **sensitive to heading
  bias** because it can't self-correct heading mid-segment — a 30° yaw/calibration
  error rotates every velocity command by 30°, curving the path (it still
  reaches the waypoints).
- **Nose-first (adaptive) is the opposite**: it self-corrects heading from
  motion, so it shrugs off yaw bias, but per-tick position noise corrupts that
  motion-heading estimate → yaw oscillation (0.1 m noise fails). `adaptive_
  smoothed` (control pose/heading low-pass + yaw-rate limit + ≥0.30 m offset-
  refinement baseline) recovers 40% of 0.1 m-noise trials at ~1/2 the oscillation
  of `adaptive_lookahead` and ~1/6 of `continuous_path`.

All controllers tolerate yaw bias ≤15°, telemetry delay ≤200 ms, and 5% dropout
at 100%. (0.1 m *white per-tick* noise at 20 Hz is a harsh, unrealistic stress
test — real Olympe fused telemetry noise is smaller and temporally correlated.)

## Lateral-assist test

`hybrid` (bounded roll assist ON) vs `nose_first` (OFF) on `adaptive_lookahead`,
16 trials: cross-track 0.042 vs 0.045 m, same 100% pass, same 2.3 flips/min,
~21 assist ticks/trial — a small improvement; kept OFF-by-default at the
algorithm level and optional via `--horizontal-control-mode`. (The translational
controller uses full omnidirectional roll as a first-class axis, with the
normalized 3-D command resultant capped so diagonal modes do not stack full
per-axis commands.)

## Bugs found and fixed during validation (review findings)

- **Cached-state freshness → event-sequenced fusion.** Sphinx
  `PositionChanged` is ~1 Hz and Olympe `get_state` returns a cached payload.
  Stamping every read with `now` made frozen telemetry appear permanently
  fresh. The source now tracks `get_last_event` UUIDs, preserves the observation
  stamp, dead-reckons only from fresh SpeedChanged events, bounds GPS-anchor age
  and fails closed. `get_fused_reference` is diagnostic firmware-fused position,
  not exact ground truth.
- **Translational heading corruption.** The map-yaw offset estimator refines from
  motion assuming nose-first flight; the translational controller strafes/backs
  up, so the offset flipped and the drone ran away. Fix: `refine_heading_from_
  motion = False` → use the drone's actual fused body yaw directly. Also
  decelerate into the waypoint (removed the min-speed floor near the target) to
  stop overshoot.
- **Rejoin segment catch-up / progress-switch guard / route-seeded heading /
  motion-anchor persistence / self-healing takeoff** — smaller corridor-controller
  and harness fixes (see git/comments).

## 22–27. Scale: meters vs arbitrary map units

Everything measured is **Sphinx meters**. The real monocular SfM/GLOMAP map is
**arbitrary units on all axes, including height**; real vertical control tracks
only relative map-unit height, never physical meters. Real horizontal AND
vertical thresholds **cannot be copied from Sphinx meters** — without a calibrated
factor they stay in map units and are flagged as requiring calibration (the
harness prints the standard warning and, absent a factor, emits provisional
route-geometry-derived map-unit thresholds, explicitly not flight-ready). **No
scale factor was supplied for any validation run** (warning emitted). The
`scale_report_example` run demonstrates dual reporting with a hypothetical
`--map-units-per-meter 2.4` (0.5 m corridor = 1.2 map units, 3.0 m deviation =
7.2 map units) — illustrative, not a measured calibration. Calibrate via
`scale_utils.compute_map_units_per_meter` before physical flight.

## 28. Pure-Python tests

**167 tests, all passing** without Sphinx (`pytest -q tests`): polyline/active
projection, nearest-segment-≠-nearest-waypoint, fixed/adaptive/clamped lookahead,
arrival radius, segment switch + early/late detection, REJOIN/FOLLOW hysteresis,
deadband/correction/hard-abort, vertical interpolation/slope/steep-warning,
vertical-doesn't-trigger-horizontal-REJOIN, high-vertical-slows-forward-drive,
gaz sign, yaw wrap, random-start-within-5 m + front/back/left/right coverage,
stop-and-go & oscillation scoring, PCMD clamp, non-Sphinx IP rejection (+ dual
strict `ipaddress`-based `10.202.0.0/16` gate), scale conversion + missing-factor warning +
map-units-vs-meters, telemetry perturbation wrapper, the **14-mode
classifier**, **translational body-decomposition signs / no-yaw-in-segment /
yaw-only-at-waypoint / mode-label-not-discrete-speed / 3-D resultant-cap**, and
closed-loop kinematic convergence per algorithm (nose-first and translational).

## 29–31. What this validates / does not / residual risks

- **Validates**: route-convergence control geometry and stability on the
  simulated Parrot ANAFI (Sphinx/Olympe fused telemetry pose) under ≤5 m random
  offset start, initial yaw error, and injected telemetry faults — for both the
  nose-first family and the yaw-locked translational family.
- **Does NOT validate**: visual localization, hloc, SfM/GLOMAP quality,
  XFeat/LightGlue/MegaLoc, raw IMU, real sensor fusion, GPS-denied dead
  reckoning, monocular map metric scale, physical altitude in meters, obstacle
  avoidance, wind, real-flight safety.
- **Residual real-world risks**: real map thresholds uncalibrated (arbitrary
  units); the translational controller needs an accurate NED→map yaw calibration
  it cannot self-correct; real localizer pose is lower-rate/noisier/occasionally
  wrong; hloc confidence/dropout/jump behavior has not yet been replayed through
  the controller; wind/aero not modeled; the kinematic backend is an
  approximation; the Sphinx offscreen renderer is unstable on this GPU.

The blocking next stage is a textured UE scene feeding timestamped ANAFI
720p30 video through decode, MegaLoc, XFeat/PnP and freshness/confidence gates
into the controller, evaluated against an independent Gazebo pose channel.
That localization-in-the-loop path is not present in this package and was not
tested here.

## 32. Recommendation for later production integration

Only after separate scale calibration and with the existing safety stack intact.
If the real localizer provides a reliable heading (or a good, held NED→map yaw
calibration): port the **translational waypoint controller** — body-frame 3-D
translation via the 14 modes, yaw only at waypoints, separated horizontal/vertical
handling — behind a flag, keeping production's speed/PCMD limits,
jump/deviation gates, and manual override, and converting every meter threshold
to map units with a measured factor first. Add control-pose low-pass only after
hloc pose replay shows that it improves real localization noise without adding
too much lag. Keep the nose-first `adaptive_lookahead` available as the fallback
for uncertain-heading conditions (it self-corrects heading; the translational
one does not). Because the translational controller uses roll as a primary axis,
treat lateral speed with the same conservatism as forward speed (the resultant
is capped after combining the 3-D body-frame command, not by stacking full
per-axis commands). Do not adopt any meter-valued threshold from this experiment
into the real map without calibration. This experiment does not certify
real-flight safety.

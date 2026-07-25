# HANDOFF — Sphinx ANAFI path-convergence experiment

Read this first, then `README.md` (how to run) and `RESULTS.md` (findings).
Most code lives under `sfm_system/定位/experiments/sphinx_anafi_path_convergence/`.
The only companion files are the simulator-only launcher and smoke test under
`mission/flight_control/`. Do not change real-flight or deploy entrypoints.

## What this is

Validates route-convergence CONTROL (not vision) for the Parrot ANAFI in
Parrot Sphinx / Olympe. A drone takes off within 5 m of a route's first
waypoint and must converge onto the planned waypoint route. Ten algorithms are
compared; the user's requested design is `translational_waypoint`.

## Status

- 10 controllers implemented over a shared state machine (`controllers.py`).
- **167 pytest tests pass** (`/home/allen/localization/.venv/bin/python -m pytest -q tests`).
- Current paired clean kinematic baseline completed: 200/200 trials passed and
  completed. `adaptive_lookahead` led the composite score (157.91), while
  `translational_waypoint` ranked third (157.51) with fewer yaw reversals.
  Older robustness tables are pre-pairing historical evidence and must be
  rerun before choosing a controller.
- Sphinx ANAFI proven live: connect, takeoff, real telemetry (GPS ~1.2 Hz,
  attitude/velocity ~5.3 Hz), PCMD→motion with correct signs, harness ran
  end-to-end (surfaced + fixed the GPS-staleness bug via velocity dead-reckoning).
- Clean Sphinx confirmation completed 2026-07-09:
  `outputs/sphinx_translational` passed 2/2 for `translational_waypoint`, and
  `outputs/sphinx_compare` exercised all 10 controllers. That old compare used
  different random scenarios per algorithm, so its cross-algorithm ranking is
  not a fair current result. The targeted 2/2 result remains useful.
- Sphinx stack rechecked 2026-07-10: Sphinx 2.25.2, ANAFI firmware 1.10.4 and
  offscreen UE4 reached `All drones instantiated`; an 8 s simulator-only smoke
  completed connect, takeoff, telemetry, PCMD, zero PCMD, Landing and
  disconnect. It was an arming/lifecycle smoke, not a convergence result.
- Browser visual sandbox added:
  `make_browser_sandbox.py` generates `outputs/browser_sandbox/index.html` from
  the same controller code and kinematic plant. It is a Three.js 3-D view with
  mouse orbit/pan/zoom for visual/debug review, not firmware simulation.
- Route safety tube added as an optional controller parameter:
  `route_tube_radius` is a full-3D nearest-distance tube around the active
  segment +/- `route_tube_segment_window`, not the entire polyline. Hloc glitches
  are debounced by `route_tube_exit_s` plus `route_tube_exit_updates`, with
  `route_tube_initial_grace_s` for initial rejoin from outside the tube.

## Environment limitation

The Parrot UE4 renderer segfaults on this machine's RTX 5060 Laptop (Blackwell,
driver 580). `-RenderOffScreen` avoids the segfault and was used for the Sphinx
runs above, but it provides no normal UE4 viewport and the render loop can still
drop the drone during boot. Treat this as flaky renderer/hardware behavior, not
a route-controller bug.

## How to launch Sphinx (exact, this machine)

```bash
# Check dependencies and observed Sphinx version:
sfm_system/定位/mission/flight_control/launch_sphinx_anafi_empty.sh --check

# Start firmwared, Sphinx core and the known-working offscreen UE4 renderer:
sfm_system/定位/mission/flight_control/launch_sphinx_anafi_empty.sh
# Continue only after it sees BOTH "All drones instantiated" and 10.202.0.1.
# "All drones dropped" fails startup. Cleanup has bounded TERM -> KILL fallback.
```

The default firmware selector contains `#latest`, so it is unpinned. Each live
harness summary records observed Sphinx, ANAFI firmware and Olympe versions and
sets `fully_reproducible=false`. Do not claim exact reproduction until the
firmware image is pinned by immutable revision/hash.

Retry loop that fits the flaky boot window: `scratchpad/*.sh` in the session
scratchpad (`start_sphinx.sh`, `start_ue4.sh`, `sphinx_retry.sh`) show a
working pattern — restart the stack, poll `ping 10.202.0.1` for up to ~45 s, and
fire the harness the instant the drone is up. If the drone keeps dropping,
retry the whole stack a few times; it is probabilistic.

## Run the harness against Sphinx

```bash
cd sfm_system/定位/experiments/sphinx_anafi_path_convergence
/home/allen/localization/.venv/bin/python run_sphinx_anafi_convergence.py \
  --backend sphinx --pattern line --segment-length-m 3.0 --num-trials 2 \
  --rejoin-algorithm translational_waypoint --random-yaw-error-deg 10 \
  --duration 22 --min-segment-time-s 0.6 --random-seed 5 \
  --out outputs/sphinx_translational
# then --rejoin-algorithm compare for all 10, longer duration, more trials
```

The harness self-heals takeoff (handles a drone left airborne by a prior run),
zeroes PCMD before Landing, lands+disconnects in `finally`, and accepts only
valid IPv4 addresses in `10.202.0.0/16` with no override. If a prior run wedged the Olympe connection, land+disconnect
once (`scratchpad/reset_drone.py`) before retrying.

## Design facts you must not break

- **Translational controller** (`TranslationalWaypointController`): yaw-locked
  body-frame translation via 14 movement modes (6 faces + 8 corners), yaw ONLY at
  waypoints. The 14-mode classifier is a label for analysis/debug; PCMD is
  computed from continuous body-frame components, and the final 3-D command
  resultant is capped so diagonal modes never stack full per-axis commands. It
  sets `refine_heading_from_motion = False` and uses the drone's fused body yaw
  DIRECTLY — because it strafes/backs up, motion ≠ heading, so the motion-refined
  map-yaw offset would flip the heading and cause runaway. In Sphinx the fused
  NED yaw equals the map heading (local frame x=north, z=east); the real GLOMAP
  map needs a one-time constant NED→map yaw calibration instead.
- **Sphinx GPS is ~1 Hz** vs the 20 Hz loop. `SphinxTelemetrySource` observes
  Olympe `get_last_event` UUIDs, snaps to fresh PositionChanged events and
  dead-reckons only with fresh SpeedChanged events. Re-reading a cached state
  never refreshes its stamp; frozen telemetry becomes stale fail-closed.
  `get_fused_reference` is the same firmware-fused position channel, not exact
  or independent ground truth, and is diagnostic only.
- **Scale**: everything measured is Sphinx METERS. The real map is arbitrary
  units on all axes incl. height; never copy a meter threshold to the real map
  without a calibrated factor (`scale_utils`). Keep the warning behavior.
- **Route tube safety**: full-3D route tube distance is in route/map units.
  For hloc, require outside-tube duration plus effective pose updates before
  manual handoff; a single bad localization frame or repeated stale pose must not
  abort. Low-confidence hloc poses must hold zero PCMD and must not update tube
  counters or control.
- **Safety**: no `Emergency`; conservative PCMD caps; sim-IP-only guard. Do not
  weaken these; do not touch production real-flight code.

## Verification checklist for the next agent

1. `/home/allen/localization/.venv/bin/python -m pytest -q tests` → 167 passing.
2. If you edit a controller, re-run the relevant kinematic comparison and
   confirm no regression in pass rate / cross-track / oscillation.
3. If you repeat Sphinx: capture `outputs/sphinx_translational/summary.json`
   with a real `backend: "sphinx …"` and non-trivial per-tick pose, then update
   `RESULTS.md` §3 and §9–18 with any newer measured Sphinx numbers.
4. Keep outputs pruned (delete `ticks.jsonl` after extracting summaries — they
   are gitignored and huge).
5. For browser review, run `make_browser_sandbox.py` and serve
   `outputs/browser_sandbox/` on localhost; `index.html` is the entrypoint.

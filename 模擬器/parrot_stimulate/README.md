# ANAFI Sphinx PCMD diagonal-flight probe

This is a deliberately narrow, **Sphinx-only** prototype. It can run a single PCMD probe or
follow a generated 10-waypoint route with closed-loop PCMD corrections.

> With the ANAFI PC firmware running in Parrot Sphinx, does a simultaneous forward
> `pitch` command and upward `gaz` command create measurable forward-and-upward motion?

Route runs use Sphinx true position only to generate a delayed/noisy localization estimate and to
score the result. PCMD decisions do not read the current true position. The prototype does not run
the GlueMap/EDM vision pipeline, load its maps, or connect to a physical drone. EDM is treated as an
externally validated localization source; this project stresses its operator-observed 0.30 m and
10° error envelope.

## Safety boundary

- The Olympe target is hard-locked to Sphinx's virtual-drone address `10.202.0.1`.
  There is no command-line option for a physical-drone address.
- Every run sends zero PCMD in `finally`, stops the Olympe piloting loop, then requests
  a normal landing. It never sends the emergency motor-cut command.
- Route takeoff is blocked below 30% battery. Stale/low-confidence localization causes zero-PCMD
  hover and repeated failures terminate the flight; altitude or route-geofence violations terminate
  immediately. Ctrl+C follows the same zero-PCMD and landing cleanup path.
- A non-empty output directory is never overwritten.
- Single-probe runs fix the Sphinx spawn pose to yaw zero: `0 0 0.2 0 0 0`.
  Route runs take off there and generate P1 within 2 m, then follow P1 through P10.

## What is measured

The acceptance decision uses Sphinx **true simulator telemetry**, not GPS or flight-state
estimates. `tlm-data-logger inet:127.0.0.1:9060` exposes `worldPosition`; Parrot documents
this as true data not tainted by estimation. The parser accepts only
`omniscient_anafi.worldPosition`, excluding the vertical-camera and UE4-object positions that
share the stream. Sphinx/Gazebo uses ENU metres, so this probe reports:

- forward displacement: initial-body forward axis;
- lateral drift: initial-body right axis;
- upward displacement: Gazebo `z`;
- yaw drift: relative Olympe attitude change.

Default acceptance criteria for the initial `pitch=20, gaz=20, duration=1.5 s` probe are:

| Check | Pass condition |
| --- | --- |
| Forward displacement | at least 0.20 m |
| Upward displacement | at least 0.20 m |
| Lateral drift | at most 0.20 m |
| Absolute yaw drift | at most 5° |

The values are intentionally simple evidence gates, not real-flight safety limits. PCMD is a
firmware control input, not a world-frame velocity vector: `pitch` commands a percentage of
the configured maximum pitch/roll, while `gaz` commands a percentage of the configured
vertical-speed limit.

## Current host status

The project environment is installed and imports Olympe successfully:

- Parrot Sphinx 2.25.2
- Parrot UE4 Empty 2.25.2, launched headlessly with `-RenderOffScreen`
- Olympe 8.4.0 in an isolated uv-managed Python 3.11 environment
- A locally cached ANAFI PC firmware image, SHA-256
  `fcbc8450911e7533763479bed672fd1c4b3a3395c073171ad2d1ce568a15fede`
  (details in [`firmware/manifest.json`](firmware/manifest.json)). Runtime commands verify this
  ignored local asset against the manifest and do not fall back to the remote `#latest` alias.

UEFI Secure Boot is disabled on the current host, so the Sphinx preflight can pass.

## Commands

```bash
# Integrated localization checkout.
cd /home/allen/localization/模擬器/parrot_stimulate
uv sync --locked

# No simulator launch: show exact PC firmware, Sphinx, UE and PCMD configuration.
uv run anafi-pcmd-sim dry-run

# Must return passed=true before an end-to-end run.
uv run anafi-pcmd-sim preflight

# Launch Sphinx and its headless UE world, run the forward-up probe, then cleanly stop both.
uv run anafi-pcmd-sim run --launch-sphinx --duration 1.5 --pitch 20 --gaz 20

# Add --show-window to open the local Unreal Engine visualizer on the desktop.
uv run anafi-pcmd-sim run --launch-sphinx --show-window --duration 1.5 --pitch 20 --gaz 20

# Run all 26 non-zero body-frame translation directions. This remains Sphinx-only.
uv run anafi-pcmd-sim sweep --launch-sphinx --magnitude 20 --duration 1.5

# Measure the Sphinx response to ±10% roll, pitch, yaw, and gaz steps.
uv run anafi-pcmd-sim response --launch-sphinx --magnitude 10

# Generate a reproducible forward-winding 3D route, follow its 10 points, and land.
uv run anafi-pcmd-sim route --show-window --seed 42

# Inspect the exact route and controller settings without launching Sphinx.
uv run anafi-pcmd-sim route --dry-run --seed 42

# Inspect or run five deterministic bounded worst-case route scenarios.
uv run anafi-pcmd-sim worst-case --dry-run
uv run anafi-pcmd-sim worst-case --seed 42
```

Successful runs create a timestamped directory in `artifacts/runs/` containing:

- `report.json` — command, pass/fail decision, displacement and yaw metrics;
- `pcmd_response.csv` — the Sphinx speed curve during the command and after zero-PCMD release;
- `true_trajectory.csv` — Sphinx true ENU world positions;
- `true_telemetry.stderr.log` — telemetry-client diagnostics;
- `simulator/sphinx.log`, `simulator/unreal.log`, and Sphinx `.tlmb` telemetry data.

Route runs additionally create:

- `route_plan.json` and `route_preview.svg` — the 10 points, connected route, start, and settings;
- `pcmd_log.csv` — every feedback cycle's line vector, yaw error, and roll/pitch/yaw/gaz;
- `wind_disturbances.csv` — every bounded true-position offset and its vector length;
- `safety_events.csv` — localization and geofence fail-safe activations;
- `route_report.json` and `route_result.svg` — waypoint arrivals, landing state, and true path.

The true acceptance radius remains 0.5 m. To reserve room for the configured 0.30 m position error
and motion during 0.20 s of latency, the controller requires three consecutive localization updates
inside a conservative 0.15 m estimated radius with estimated speed no greater than 0.25 m/s. It
then hovers, recomputes the camera-to-next-point XY angle once per second, and requires three
consecutive estimates inside a 3° deadband with estimated yaw rate no greater than 5°/s before
translating. An exact measured 0° is not observable when heading estimates contain noise.
During translation, the latest 3D target
line is decomposed into body-forward, body-right, and up PCMD components every second without
re-entering yaw-only alignment. Estimated velocity adds slowdown, overspeed, and closing-speed
braking. Every PCMD expires after 0.75 s, before the next one-second update. When the remaining XY
projection is shorter than 0.25 m, that
noise-sensitive bearing no longer blocks vertical capture; the 0.5 m three-dimensional arrival
requirement is unchanged.
The only injected errors are seeded sudden wind-like true-position displacement and localization
error. Every 20 seconds, `pysphinx.Sphinx.move_drone` applies a random horizontal displacement
whose vector length is at most 0.40 m; each event is retained in `wind_disturbances.csv`. The
controller's three-dimensional position-error vector is at most 0.30 m and its camera-heading
error is at most ±10°. Localization errors are time-correlated, delayed by 0.20 s, and independently
dropped on 3% of control updates; dropped or invalid data sends zero PCMD. The CSV retains both the
true and estimated states, pose age, confidence, estimated speed, safety state, and arrival counter.
The 0.30 m limit is
the operator-observed GlueMap/EDM cap and is also close to the worst-video 0.298 m P95 disagreement
in the repository's image-only trajectory comparison. That comparison is not surveyed ground
truth. PCMD commands are not weakened, dropped, or replaced by artificial axis errors.
Wind disturbance stops after the last waypoint is confirmed, before landing begins.

`worst-case` does not sample random seeds. It runs fixed maximum-bound cases for position bias toward
the target, position bias away from the target, a 0.40 m final-approach displacement, localization
loss immediately after a non-zero PCMD, and their combined limits. A case passes only if all ten
true arrivals remain within 0.50 m and landing is confirmed.

`route_plan.json` makes the simulator assumptions explicit: Gazebo ENU, metres, identity
camera-to-body translation, and zero camera/body yaw offset for the Sphinx model. Production
integration must replace the frame axes and camera-to-body extrinsics, but must not invent a
map-unit-to-metre scale: the shared core normalizes the map-space target vector, uses site-specific
map-unit arrival/deviation gates, and limits physical horizontal speed from airframe telemetry.
The current `parrot-ue4-empty` world has no obstacle course, so the report marks collision clearance
as an external production requirement rather than pretending it was tested.

The generated route advances monotonically along ENU +X in 0.35–0.45 m steps, stays within 0.35 m
of its centreline, and alternates between low and high waypoints. The combined X/Y/Z segment length
is still about one metre. Keeping the final X near 2.7 m also keeps the landing inside the finite
collision ground of `parrot-ue4-empty`; the earlier 8.7 m endpoint had no physical floor. It remains
an overall straight route with small horizontal bends and vertical undulation.

The validated seed-42 run is in
`artifacts/runs/waypoint-route-seed42-full-safety-v6-true-radius-reserve-20260802`. It completed all
10 waypoints and landed on the first attempt. True arrival errors were 0.082–0.178 m; every waypoint
had three low-speed confirmations. The observed maxima were 0.261 m position error, 6.282° yaw
error, 0.3969 m wind displacement, and 0.20 s accepted-pose age. The run recorded 25 seeded
localization-loss events, all of which produced zero-PCMD hover and recovered within the configured
limit.
That run predates the three-update low-yaw-rate alignment gate and remains a random-error baseline.
The new gate's deterministic combined-bound receipt is in
`artifacts/runs/waypoint-route-combined-bounds-20260802-attempt2`. It completed 10/10 points and
landed on the first attempt while holding a 0.30 m target-directed position bias, 10° yaw bias,
injecting localization loss after non-zero PCMD, and applying one 0.40 m final-approach offset.
The largest true arrival error was 0.4405 m. Its 401 decision rows include 29 yaw-alignment holds
and 9 third-sample alignment confirmations.

Route runs use the Sphinx `anafi.drone` model (`anafi4k` hardware) and the ANAFI PC firmware.
For repeatability, the controller pins the firmware's standard limits to 20° maximum tilt,
1 m/s vertical speed, 70°/s yaw, and 150°/s pitch/roll rotation. The white-paper hardware maxima
(15 m/s horizontal, 4 m/s vertical, and 200°/s angular velocity) are retained as model metadata,
not forced as the normal flight settings. Each row in `pcmd_log.csv` includes the PCMD actually
sent, true and estimated navigation states, navigation error, and the configured tilt,
vertical-speed, and yaw-rate setpoints.

The Sphinx `pitch=10%` response receipt at
`artifacts/runs/pcmd-response-pitch10-20260802` measured 0.717 m/s peak horizontal speed,
0.342 m/s mean speed during the three-second command, 0.429 m of movement after release, and
1.875 s to remain below 0.05 m/s for three telemetry samples. These values calibrate only the
current Sphinx firmware/model. The same step test must be measured on the physical ANAFI before
using them as real-aircraft gains.

## Test and lint

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## Run one probe automatically after the next boot

The project includes an opt-in per-user systemd service. It is enabled once but launches nothing
unless an explicit arm marker exists. This host has user-service lingering enabled, so the service
can start after boot even before you reconnect over SSH.

```bash
cd /home/allen/localization/模擬器/parrot_stimulate
./scripts/install_user_boot_service.sh
./scripts/arm_next_boot_probe.sh
```

After the next boot it runs the fixed Sphinx-only command
`run --launch-sphinx --duration 1.5 --pitch 20 --gaz 20`, writes results below
`artifacts/boot-runs/`, and exits. Cancel a pending run with
`./scripts/disarm_next_boot_probe.sh`. See [`docs/boot_autorun.md`](docs/boot_autorun.md) for
verification commands and the one-shot safety behavior.

## Official references

- [Olympe: controller SDK and Sphinx support](https://developer.parrot.com/docs/olympe/index.html)
- [Olympe PCMD semantics](https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html)
- [ANAFI White Paper v1.4](https://www.parrot.com/assets/s3fs-public/2020-07/white-paper_anafi-v1.4-en.pdf)
- [Olympe take-off / hovering sequence](https://developer.parrot.com/docs/olympe/userguide/basics/moving_around.html)
- [Sphinx quick start and PC firmware launch](https://developer.parrot.com/docs/sphinx/quickstart.html)
- [Sphinx true telemetry and `worldPosition`](https://developer.parrot.com/docs/sphinx/inspecting.html)
- [Sphinx coordinate systems](https://developer.parrot.com/docs/sphinx/coordinate_systems.html)
- [Sphinx system requirements, including Secure Boot](https://developer.parrot.com/docs/sphinx/system_requirements.html)

## Scope limitation

Passing this simulation proves behavior for the chosen PC firmware image and simulated
hardware conditions only. Before a real-aircraft trial, pin the exact firmware matching the
physical ANAFI, repeat the test with conservative limits, and add a separate real-flight safety
review. This prototype intentionally cannot connect to a real aircraft.

# ANAFI / Sphinx / Olympe notes for this experiment

## Aircraft

Parrot ANAFI as modeled by Parrot Sphinx (`/opt/parrot-sphinx/usr/share/sphinx/
drones/anafi.drone` + `anafi-pc.ext2.zip` firmware). Sphinx is Parrot's official
Gazebo software-in-the-loop path and runs the ANAFI PC firmware corresponding
to the aircraft firmware. No generic quadrotor model is used; no real hardware
is ever contacted (strict `10.202.0.0/16` IP guard).

Launch (same script production uses):

```bash
systemctl start firmwared          # once per boot; launcher also checks/starts it
sfm_system/定位/mission/flight_control/launch_sphinx_anafi_empty.sh --check
sfm_system/定位/mission/flight_control/launch_sphinx_anafi_empty.sh
# = sphinx anafi.drone::firmware=<anafi-pc.ext2.zip>
#   + exact UE4 binary with -RenderOffScreen on this machine
```

Olympe connects to `10.202.0.1` (Sphinx virtual ANAFI).
The default firmware selector contains `#latest`; it is not immutable. Live
summaries record observed Sphinx, firmware and Olympe versions and explicitly
set `fully_reproducible=false` until an image revision/hash is pinned.

## White-paper profile

`anafi_profile.py` is the single machine-readable source used by the
kinematic plant, browser metadata/defaults and batch summary metadata. It
records the ANAFI v1.4 white-paper values: 0.320 kg; 15 m/s horizontal;
4 m/s ascent/descent; 200 deg/s angular speed; 50 km/h wind / 80 km/h gust;
1 m takeoff hover; 0.015 m hover accuracy at 1 m; 200 Hz internal loop;
1280x720 at 30 fps and 5 Mbps with 280 ms video latency; gimbal +/-90 deg and
180 deg/s; GPS position/speed standard deviations 1.2 m / 0.5 m/s; barometer
noise 0.2 m. Source: <https://www.parrot.com/assets/s3fs-public/2020-07/white-paper_anafi-v1.4-en.pdf>.

These capability figures do not raise controller PCMD caps. Not every profile
field is modeled by the kinematic approximation; unmodeled values remain
explicit metadata rather than implied physics.

## Telemetry used (fused, not raw IMU)

* `PositionChanged` (lat/lon/alt) -> local meters relative to the first fix.
* `AttitudeChanged` -> fused roll/pitch/yaw (NED yaw: 0 = north, + toward east).
* `SpeedChanged` -> NED velocity (consistency logging only).

These are firmware-fused states; this experiment explicitly does NOT claim raw
IMU validation.

## Coordinate frames

| frame | convention |
|---|---|
| Sphinx local | north/east/up meters, origin = first GPS fix |
| controller raw map | x = north, z = east, y = -up (up = -Y), yaw = atan2(z, x) |
| path JSON "aligned" | X/Y horizontal, Z up; aligned (x,y,z) -> raw (x,-z,y) |
| Olympe NED velocity | speedX north, speedY east, speedZ down |
| real SfM/GLOMAP map | same axis convention as raw, but ARBITRARY units (all axes, height included) |

With x=north/z=east the controller heading equals the ANAFI NED fused yaw, so
the learned heading offset is ~0 in simulation; the offset machinery is still
exercised (and tested under injected yaw bias) because the real system needs it.

## PCMD semantics

`PCMD(flag, roll, pitch, yaw, gaz)` percents in [-100,100]:
pitch>0 forward, roll>0 right, yaw>0 clockwise (heading increases), gaz>0 climb.
Conservative caps in this experiment: pitch 10, yaw 25, gaz 15, roll 6 (roll
used only when lateral assist explicitly enabled). `--auto-yaw-calibration`
verifies the yaw sign in-sim with a small hover yaw pulse (no Emergency, zero
PCMD restored).

## Kinematic backend caveat

`--backend kinematic` is a first-order-lag approximation. Its 100% command
scales reference the profile (15 m/s horizontal, 4 m/s vertical, 200 deg/s
yaw) and it integrates at 200 Hz, but the controllers retain conservative
10-25% caps. It has no firmware, mass/aerodynamic, motor, wind-resistance,
gimbal, camera-codec or sensor model. It is used for paired Monte Carlo sweeps
and CI and is NOT Sphinx physics. Browser output is only a replay of this same
kinematic run. Sphinx runs are the ANAFI firmware/dynamics validation.

## Timing

Control loop 20 Hz. Sphinx GPS position updates arrive slower than attitude;
the pose-age gate (0.6 s) plus the LOST/abort ladder handles gaps. The
perturbation wrapper can additionally delay/drop telemetry to probe margins.

Measured rates (this ANAFI, offscreen): `PositionChanged` ~1.2 Hz,
`AttitudeChanged`/`SpeedChanged` ~5.3 Hz. `SphinxTelemetrySource` uses Olympe
event UUIDs as reception sequences, snaps to each fresh fused position event
and dead-reckons with fresh NED velocity events. Cached events retain their old
stamp and become stale fail-closed. `get_fused_reference` exposes the same
firmware-fused position channel for diagnostics; it is not independent or
exact simulator ground truth.

The 280 ms white-paper number is video end-to-end latency, not complete hloc
latency. Browser metadata treats it as a lower bound and adds separately
measured decode plus MegaLoc/XFeat/PnP time. With no such measurement it marks
`telemetryDelayIsLowerBound=true`.

## Heading handling: nose-first vs translational

- **Nose-first controllers** (corridor/adaptive): the map-frame heading is the
  fused yaw plus a learned offset, refined from motion direction (motion ≈ nose
  when flying nose-first). Seeded from the route at takeoff.
- **Translational controller** (`translational_waypoint`): the drone strafes /
  backs up, so motion ≠ heading and the motion-refined offset would be wrong (it
  can flip the heading). The run loop detects this (`refine_heading_from_motion
  = False`) and uses the drone's actual fused body yaw directly. In Sphinx the
  fused NED yaw equals the map heading (local frame x=north, z=east). For the
  real GLOMAP map (arbitrary-rotation frame) this controller needs a one-time
  constant NED→map yaw calibration held fixed thereafter — it must NOT be
  refined from motion.

## GPU / renderer note

`parrot-ue4-empty` (UE4 renderer) segfaults on the RTX 5060 Laptop (Blackwell,
driver 580.159.04). `-RenderOffScreen` avoids the crash (this experiment needs
only telemetry, not camera). Even offscreen, the render loop is unstable on this
GPU and intermittently fails to boot the drone instance, so long multi-trial
Sphinx sessions are unreliable here; the kinematic backend carries the
statistical sweep.

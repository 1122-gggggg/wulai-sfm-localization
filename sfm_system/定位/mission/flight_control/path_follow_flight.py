#!/usr/bin/env python3
"""REAL ANAFI closed-loop path-follow flight (Parrot Olympe / Ground SDK).

This is the ONE missing wiring piece. Every block below already exists and was
sim-validated separately; this file joins them into a single runnable real-flight
entrypoint on ONE Olympe connection:

    OlympePdrawGrabber (720p live stream)              [olympe_frame_source.py]
      -> ProductionXFeatTracker                        [production_xfeat_tracker.py]
             MegaLoc(top30) -> XFeat -> LighterGlue -> PnP,  BOOT_INIT -> TRACK
      -> map-frame heading fusion  (Olympe yaw  <->  map-motion direction)
      -> RouteAutoController.step(pose) -> Command      [real_path_follow_controller.py]
             FOLLOW / REJOIN-nearest-point / LAND  (keeps drone ON the drawn route)
      -> command_to_body_percent(cmd, pose) -> PCMD(roll, pitch, yaw, gaz)
      -> drone(PCMD(...))            same single connection used for the video

SDK      : Parrot Olympe (Ground SDK).  https://developer.parrot.com/docs/olympe/
Map      : fused forward+reverse GLOMAP; reloc bundle reloc_map_xfeat_tri.pt (1920 refs).
Path     : pre-drawn polyline  safezone/flight_path.json  (Blender Z-up waypoints).
Frame    : raw GLOMAP  horizontal = X/Z, gravity-up = -Y   (matches RouteAutoController).

Why heading is fused and NOT read from the localizer
----------------------------------------------------
Both localizers store  pose.yaw = atan2(fwd_y, fwd_x)  -- that is the X/Y plane,
which in this -Y-up map is NOT a horizontal heading (it mixes in the vertical
axis). The sim dashboards only ever used pose.x/y/z, so the bad yaw was never
exercised. A path-follow controller DOES need a real map-frame heading for the
yaw feedback, so we derive it here from the drone's map-motion direction
(atan2(dz, dx), exact because positions are exact) and anchor Olympe's fast,
low-latency yaw telemetry to it with a running offset. We deliberately do not
touch the tracker's internal yaw (its ref_yaws gate depends on that convention).

SAFETY  -- real flight is irreversible and can injure people or destroy the drone.
  --selftest  : pure-python checks (heading fusion + PCMD sign sanity). No deps.
  --dry-run   : no drone; a toy dynamics model closes the loop to exercise logic.
  --grab-only : connect + localize on the LIVE stream, PROPS OFF, never arms motors.
  --fly       : arms + flies. Open space, human on the manual-override controller,
                AFTER verifying PCMD signs (--yaw-sign) on YOUR airframe.
Take off with the drone NOSE POINTED ALONG THE ROUTE START (that seeds the heading
offset); online motion refinement corrects it from there. Speeds are tiny.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import select
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLIGHT_ROOT = Path(__file__).resolve().parent


def _find_system_root(start: Path) -> Path:
    # This file ships in three places (source deploy dir, transfer-package
    # deploy_code/, transfer-package mission/flight_control/); a fixed
    # parents[N] depth is only right for one of them. Walk up instead.
    for p in [start, *start.parents]:
        if p.name == "sfm_system":
            return p
    return start.parents[3] if len(start.parents) > 3 else start


SYSTEM_ROOT = _find_system_root(FLIGHT_ROOT)
LOC_ROOT = SYSTEM_ROOT / "定位"
MISSION_ROOT = LOC_ROOT / "mission"
SOURCE_ROOT = LOC_ROOT / "source" / "sfm_glomap"
DEPLOY_ROOT = LOC_ROOT / "deploy_code" / "sfm_glomap_deploy"
DEFAULT_BUNDLE = (
    LOC_ROOT / "bundles" / "current_reloc_map_updated_v3.pt"
    if (LOC_ROOT / "bundles" / "current_reloc_map_updated_v3.pt").exists()
    else LOC_ROOT / "bundles" / "base_reloc_map_xfeat_tri.pt"
)
DEFAULT_MEGALOC_CACHE = DEPLOY_ROOT / "megaloc_ref_desc_glomap_fused_322.npy"
DEFAULT_PATH_JSON = (
    MISSION_ROOT / "outputs" / "current_safezone" / "flight_path.json"
    if (MISSION_ROOT / "outputs" / "current_safezone" / "flight_path.json").exists()
    else SOURCE_ROOT / "safezone" / "flight_path.json"
)
DEFAULT_POLES_JSON = (
    MISSION_ROOT / "outputs" / "current_safezone" / "poles.json"
    if (MISSION_ROOT / "outputs" / "current_safezone" / "poles.json").exists()
    else SOURCE_ROOT / "safezone" / "poles.json"
)

# artifact paths (same ones sim_dashboard_real.py loads)
XBUN = os.environ.get("SFM_RELOC_BUNDLE", str(DEFAULT_BUNDLE))
MEG = os.environ.get("SFM_MEGALOC_CACHE", str(DEFAULT_MEGALOC_CACHE))
PATH_JSON = os.environ.get("SFM_FLIGHT_PATH_JSON", str(DEFAULT_PATH_JSON))
POLES_JSON = os.environ.get("SFM_POLES_JSON", str(DEFAULT_POLES_JSON))

DRONE_IP_REAL = "192.168.42.1"
DRONE_IP_SKYCTRL = "192.168.53.1"
DRONE_IP_SIM = "10.202.0.1"                     # Parrot Sphinx simulator

# 720p live-stream camera == sc1_AB_720 build intrinsics (f from ANAFI 69deg HFOV).
CAM_720 = ("SIMPLE_RADIAL", 1280, 720, [934.0, 640.0, 360.0, 0.001])

CTRL_HZ = 20
POSE_STALE_S = 0.5          # visual pose older than this -> HOVER
STREAM_STALE_S = 0.5        # 720p live frame older than this -> HOVER before localization
LOST_LAND_S = 4.0           # no fresh pose for this long -> auto-land
# Localization lost + MegaLoc cannot relocalize this long -> hand to the human pilot
# (only if a SkyController manual pilot exists; else the LOST_LAND_S failsafe still lands).
LOST_MANUAL_S = float(os.environ.get("SFM_LOST_MANUAL_S", "3.0"))
LOW_CONF_INLIERS = int(os.environ.get("SFM_LOW_CONF_INLIERS", "60"))  # PnP inliers below -> low-confidence
# Stream lost this long -> auto-land (was: hover forever until battery death).
STREAM_LOST_LAND_S = float(os.environ.get("SFM_STREAM_LOST_LAND_S", "15.0"))
# A WEAK (low-confidence) fix in repetitive line-corridor geometry can be plausible
# but wrong and still pass the jump/deviation gates; treat it as "uncertain" -> hover.
# SFM_GATE_WEAK=0 restores the old behavior of flying on weak fixes (tuning only).
GATE_WEAK = os.environ.get("SFM_GATE_WEAK", "1") != "0"
WEAK_HOVER_LAND_S = float(os.environ.get("SFM_WEAK_HOVER_LAND_S", "8.0"))
# Reject a fresh fix that jumps farther than this from the last accepted fix
# (MAP UNITS). A second consecutive fix agreeing with the first one is accepted,
# so genuine relocalization after hover drift still recovers.
MAX_POSE_JUMP_U = float(os.environ.get("SFM_MAX_POSE_JUMP_U", "1.5"))
# Route-corridor bound (MAP UNITS): if the accepted pose ends up farther than
# this from the drawn route, abort and land instead of REJOIN-ing blindly.
MAX_ROUTE_DEVIATION_U = float(os.environ.get("SFM_MAX_ROUTE_DEVIATION_U", "3.0"))
# If the control loop stalls longer than this (model/GPU hang), a helper thread
# forces zero PCMD so the drone hovers instead of holding the last command.
PCMD_WATCHDOG_S = float(os.environ.get("SFM_PCMD_WATCHDOG_S", "0.7"))
MOVE_EPS = 0.05             # min map displacement (map-units) to trust a motion heading
OFFSET_EMA = 0.25           # heading-offset smoothing on each valid motion sample
FIRST_FIX_TIMEOUT_S = 25.0  # how long to wait for BOOT_INIT -> first TRACK
GIMBAL_PITCH_DEG = -10.0    # camera tilt vs horizon; slightly down, like the map refs
SAFETY_FILE = os.environ.get("SFM_SAFETY_FILE", "/tmp/sfm_drone_safety.cmd")


class SafetySwitch:
    """Runtime safety switch for AUTO / HOVER / MANUAL / LAND.

    Commands can come from either:
      - terminal stdin: a/auto, h/hover, m/manual, l/land;
      - a small text file, default /tmp/sfm_drone_safety.cmd.

    MANUAL means the autonomous loop stops sending PCMD. It is intended for a
    physical SkyController pilot. On direct-drone Wi-Fi, MANUAL is downgraded to
    HOVER because there may be no separate human controller.
    """

    _ALIASES = {
        "a": "AUTO", "auto": "AUTO", "resume": "AUTO",
        "h": "HOVER", "hover": "HOVER", "pause": "HOVER",
        "m": "MANUAL", "manual": "MANUAL", "pilot": "MANUAL",
        "l": "LAND", "land": "LAND", "landing": "LAND",
        # Motor cut. The drone DROPS -- only for when spinning props endanger
        # people and a normal landing is worse.
        "e": "EMERGENCY", "emergency": "EMERGENCY",
    }

    def __init__(self, path: str | Path | None = SAFETY_FILE,
                 allow_manual: bool = False, keyboard: bool = True):
        self.path = Path(path) if path else None
        self.allow_manual = bool(allow_manual)
        self.keyboard = bool(keyboard and sys.stdin and sys.stdin.isatty())
        self.mode = "AUTO"
        self._mtime_ns: int | None = None
        self._last_print = 0.0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Always reset to AUTO on startup (atomic) so a stale land/emergency left in the shared
            # file by a previous run cannot be polled by this mission's first tick.
            _tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            _tmp.write_text("auto\n")
            os.replace(_tmp, self.path)
            self._mtime_ns = self.path.stat().st_mtime_ns
        print(
            "[safety] commands: a/auto, h/hover, m/manual, l/land, e/emergency(motor cut); "
            f"file={self.path or 'disabled'} manual_allowed={self.allow_manual}",
            flush=True,
        )

    def _read_file_command(self) -> str | None:
        if self.path is None or not self.path.exists():
            return None
        try:
            st = self.path.stat()
            if self._mtime_ns == st.st_mtime_ns:
                return None
            self._mtime_ns = st.st_mtime_ns
            parts = self.path.read_text(errors="ignore").strip().split()
            return parts[0].lower() if parts else None
        except Exception as exc:
            print(f"[safety] cannot read safety file: {exc}", flush=True)
            return None

    def _read_keyboard_command(self) -> str | None:
        if not self.keyboard:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return None
            parts = sys.stdin.readline().strip().split()
            return parts[0].lower() if parts else None
        except Exception:
            return None

    def _apply(self, token: str | None) -> None:
        if not token:
            return
        new = self._ALIASES.get(str(token).lower())
        if new is None:
            print(f"[safety] ignoring unknown command={token!r}", flush=True)
            return
        if new == "MANUAL" and not self.allow_manual:
            print("[safety] MANUAL requested but no SkyController manual pilot is configured; using HOVER", flush=True)
            new = "HOVER"
        if new != self.mode:
            print(f"[safety] {self.mode} -> {new}", flush=True)
            self.mode = new

    def force(self, token: str) -> bool:
        """Programmatically request a mode (e.g. auto-switch to MANUAL on total loss).
        Returns True only if a manual pilot actually took over (not downgraded to HOVER)."""
        self._apply(token)
        return self.mode == "MANUAL"

    def poll(self) -> str:
        self._apply(self._read_file_command())
        self._apply(self._read_keyboard_command())
        if self.mode in {"HOVER", "MANUAL"}:
            now = time.monotonic()
            if now - self._last_print > 2.0:
                action = "sending zero PCMD" if self.mode == "HOVER" else "AUTO PCMD suspended"
                print(f"[safety] {self.mode}: {action}; command auto or land to continue", flush=True)
                self._last_print = now
        return self.mode


def manual_override_available(ip: str, controller: str) -> bool:
    ctrl = str(controller or "auto").lower()
    if ctrl == "auto":
        ctrl = "skycontroller3" if str(ip) == DRONE_IP_SKYCTRL else "drone"
    return ctrl in {"skycontroller3", "skyctrl3", "sc3"}


def wait_success(expectation, label: str) -> None:
    res = expectation.wait()
    if hasattr(res, "success") and not res.success():
        raise SystemExit(f"{label} failed or timed out")


# ---------------------------------------------------------------------------
# Map-frame heading: Olympe yaw (fast, drifts) anchored to map-motion (exact, sparse)

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class HeadingEstimator:
    """Estimate the drone body heading IN THE MAP FRAME (X/Z plane, -Y up).

    heading = wrap(olympe_yaw + offset), where `offset` is learned by comparing the
    map-motion direction (from consecutive localized positions) against Olympe yaw
    whenever the drone actually translates. Falls back to raw motion heading if no
    Olympe yaw is available (e.g. --dry-run).
    """

    def __init__(self):
        self.offset = None          # wrap(map_motion_heading - olympe_yaw)
        self._last_C = None
        self._last_motion_heading = None

    def seed_from_path(self, C0: np.ndarray, path_goal: np.ndarray, olympe_yaw: float | None):
        """Operator took off with the nose along the route start: assume the initial
        body heading points from C0 toward the first path goal."""
        h = math.atan2(float(path_goal[2] - C0[2]), float(path_goal[0] - C0[0]))
        self._last_motion_heading = h
        if olympe_yaw is not None:
            self.offset = _wrap(h - olympe_yaw)

    def mark_teleport(self) -> None:
        """Discard the previous position so the NEXT update() computes no displacement.
        Call on a confirmed relocation jump: a position teleport is not real motion, and
        feeding its displacement would corrupt the learned yaw offset (wrong PCMD direction).
        The offset itself is a physical yaw->map calibration and is deliberately preserved."""
        self._last_C = None

    def update(self, C: np.ndarray, olympe_yaw: float | None):
        """Refine the offset from the latest localized map position."""
        C = np.asarray(C, float)
        if self._last_C is not None:
            d = C - self._last_C
            if math.hypot(d[0], d[2]) >= MOVE_EPS:               # translated enough
                h = math.atan2(float(d[2]), float(d[0]))         # X/Z-plane heading
                self._last_motion_heading = h
                if olympe_yaw is not None:
                    new = _wrap(h - olympe_yaw)
                    self.offset = new if self.offset is None else \
                        _wrap(self.offset + OFFSET_EMA * _wrap(new - self.offset))
        self._last_C = C

    def heading(self, olympe_yaw: float | None) -> float | None:
        if olympe_yaw is not None and self.offset is not None:
            return _wrap(olympe_yaw + self.offset)
        return self._last_motion_heading      # dry-run / not-yet-anchored fallback


# ---------------------------------------------------------------------------
# Olympe helpers (thin, so the module imports without olympe for --selftest/--dry-run)

def olympe_yaw_of(drone) -> float | None:
    """Latest fused body yaw (radians, drone NED frame) or None if not yet received."""
    from olympe.messages.ardrone3.PilotingState import AttitudeChanged
    try:
        st = drone.get_state(AttitudeChanged)          # raises KeyError if never received
    except (KeyError, RuntimeError):
        return None
    return None if not st else float(st["yaw"])


def set_gimbal(drone, pitch_deg: float):
    """Point the camera to a fixed absolute (horizon-referenced) tilt so live frames
    resemble the map reference views. Absolute frame == EIS-stabilized vs horizon."""
    from olympe.messages.gimbal import set_target
    res = drone(set_target(
        gimbal_id=0, control_mode="position",
        yaw_frame_of_reference="none", yaw=0.0,
        pitch_frame_of_reference="absolute", pitch=float(pitch_deg),
        roll_frame_of_reference="none", roll=0.0)).wait()
    if hasattr(res, "success") and not res.success():
        print("[flight] warning: gimbal target command did not report success", flush=True)


# ---------------------------------------------------------------------------
# The closed loop (shared by real flight and --dry-run via the `drone_io` adapter)

@dataclass
class LoopHooks:
    get_pose: callable          # () -> localizer Pose | None   (has .x,.y,.z,.stamp)
    olympe_yaw: callable        # () -> float | None
    send_pcmd: callable         # (roll,pitch,yaw,gaz) -> None
    pose_is_weak: callable | None = None  # () -> bool; True if the last fix was a WEAK track
    pose_confidence: callable | None = None  # () -> int; last-fix PnP inliers (low -> hover + relocalize)
    force_relocalize: callable | None = None  # () -> None; ask the tracker to run MegaLoc (LOST)
    request_manual: callable | None = None  # () -> bool; switch to MANUAL on persistent loss (SkyController)
    safety_poll: callable | None = None  # () -> AUTO/HOVER/MANUAL/LAND/EMERGENCY
    stream_healthy: callable | None = None  # () -> bool; false means live stream stale/lost
    stream_status: callable | None = None   # () -> str; operator log detail
    loop_beat: callable | None = None       # () -> None; watchdog heartbeat, called every tick
    pose_info: callable | None = None       # () -> dict; localizer last_info (LOGGING ONLY)
    log_tick: callable | None = None        # (dict) -> None; structured per-tick command log
    now: callable = time.monotonic


class CommandLog:
    """JSONL per-tick command log for pre-real-flight review.

    One line per control tick: gate status, block reason, and the FINAL PCMD (or
    null when autonomy sent nothing). `sink` says where commands went:
    dry-run / replay-mock / olympe. Logging must never disturb the control loop:
    the file is line-buffered and run_loop swallows log exceptions.
    """

    def __init__(self, path: str | Path, sink: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", buffering=1, encoding="utf-8")
        self.sink = sink

    def __call__(self, rec: dict) -> None:
        rec["sink"] = self.sink
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def event(self, **kw) -> None:
        self(kw)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def default_cmd_log_path(tag: str) -> Path:
    return (LOC_ROOT / "outputs" / "flight_logs"
            / f"{tag}_cmdlog_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


class SafetyMonitor:
    """Independent safety-authority + stall watchdog thread.

    Runs OUTSIDE the control loop so operator commands act even when the main
    thread is blocked in synchronous GPU inference (get_pose) or an Olympe wait.
    The loop runs localization inference synchronously; a model/GPU hang would
    otherwise (a) leave the drone holding its last non-zero command and (b) stop
    the loop from ever polling the safety switch again. This thread fixes both:

      - polls the SafetySwitch (file/keyboard) at ~3x per timeout -- the ONLY
        reader of stdin/file, so the main loop just reads cached ``mode``;
      - EMERGENCY  -> emergency_cb() once (motor cut) and set ``terminated``;
      - LAND       -> zero PCMD then land_cb() once and set ``terminated``;
      - HOVER/MANUAL -> stream zero PCMD (hover) regardless of loop state;
      - AUTO       -> if the loop has not beaten within ``timeout`` (stall/GPU
        hang) force zero PCMD; otherwise leave control to the loop.

    So a hung get_pose() can no longer block operator LAND/EMERGENCY: this thread
    issues them directly. It cannot interrupt the stuck C-level CUDA call, but the
    drone is commanded safe from here while the main thread is still blocked.
    """

    def __init__(self, send_pcmd, safety=None, land_cb=None, emergency_cb=None,
                 timeout_s: float = PCMD_WATCHDOG_S):
        self._send = send_pcmd
        self._safety = safety
        self._land_cb = land_cb
        self._emergency_cb = emergency_cb
        self._timeout = float(timeout_s)
        self._beat_t = time.monotonic()
        self._beat_seen = False              # stall watchdog stays off until the loop beats
        self.mode = "AUTO"
        self.reason = ""
        self.terminated = threading.Event()
        self._acted = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="safety-monitor", daemon=True)

    def beat(self) -> None:
        self._beat_t = time.monotonic()
        self._beat_seen = True

    def start(self) -> "SafetyMonitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _safe_send(self, r, p, y, g) -> None:
        try:
            self._send(r, p, y, g)
        except Exception:
            pass

    def _run(self) -> None:
        warned = False
        poll_warned = False
        while not self._stop.wait(self._timeout / 3.0):
            if self._safety is not None:
                try:
                    self.mode = self._safety.poll()
                    poll_warned = False
                except Exception as exc:
                    if not poll_warned:                  # sole safety poller; a stuck poll blinds operator override
                        poll_warned = True
                        print(f"[safety] SafetyMonitor.poll() FAILING ({exc!r}); operator "
                              "LAND/HOVER/EMERGENCY may be unseen; holding last mode", flush=True)
            mode = self.mode
            if mode == "EMERGENCY":
                if not self._acted:
                    self._acted = True
                    self.reason = "EMERGENCY command -> motor cut"
                    self.terminated.set()          # visible to the loop BEFORE the cb, so a
                                                   # resuming loop cannot send a countermanding PCMD
                    if self._emergency_cb is not None:
                        try:
                            self._emergency_cb()
                        except Exception:
                            pass
                continue
            if mode == "LAND":
                if not self._acted:
                    self._acted = True
                    self.reason = "safety LAND command -> land"
                    self.terminated.set()          # set BEFORE we block on the land cb
                    self._safe_send(0, 0, 0, 0)
                    if self._land_cb is not None:
                        try:
                            self._land_cb()
                        except Exception:
                            pass
                continue
            if mode == "HOVER":
                self._safe_send(0, 0, 0, 0)          # active zero-hold
                continue
            if mode == "MANUAL":
                # Hand control to the physical pilot: the autonomous system must send
                # NOTHING here. Even a zero PCMD competes with the pilot's stick inputs.
                continue
            # AUTO: fall back to the pure stall watchdog -- but ONLY once the loop has
            # started beating. Before run_loop (takeoff, gimbal, BOOT_INIT acquisition)
            # the heartbeat is frozen, so an ungated check would false-fire and inject
            # PCMD into the takeoff maneuver.
            if self._beat_seen and (time.monotonic() - self._beat_t) > self._timeout:
                self._safe_send(0, 0, 0, 0)
                if not warned:
                    print(f"[safety] WATCHDOG: control loop stalled > {self._timeout:.1f}s; "
                          "forcing zero PCMD (hover)", flush=True)
                    warned = True
            else:
                warned = False


def run_loop(hooks: LoopHooks, ctrl, waypoints, yaw_sign: int = 1, verbose: bool = True):
    """Localize -> fuse heading -> RouteAutoController -> PCMD, at CTRL_HZ.

    Returns the terminal reason string. `ctrl` is a RouteAutoController; `waypoints`
    are raw-GLOMAP np arrays (its own list). Import here so --selftest needs no deps.
    """
    import real_path_follow_controller as rpf

    heading = HeadingEstimator()
    period = 1.0 / CTRL_HZ
    last_good = None
    lost_since = None
    weak_since = None
    stream_lost_since = None
    last_stream_warn = 0.0
    pending_jump = None
    seeded = False
    steps = 0
    reason = "stopped"
    last_safety_mode = "AUTO"
    manual_requested = False

    def emit(rec, pcmd, blocked, why):
        """Structured per-tick command record. Must never disturb the loop."""
        if hooks.log_tick is None:
            return
        rec["pcmd"] = None if pcmd is None else [int(v) for v in pcmd]
        rec["blocked"] = bool(blocked)
        rec["reason"] = str(why)
        if hooks.pose_info is not None:
            try:
                info = hooks.pose_info() or {}
                rec["loc"] = {k: info.get(k) for k in
                              ("frame", "idx", "mode", "next_mode", "inliers", "reproj_rms", "weak")}
            except Exception:
                pass
        try:
            hooks.log_tick(rec)
        except Exception:
            pass

    while True:
        t0 = hooks.now()
        if hooks.loop_beat is not None:
            hooks.loop_beat()
        safety_mode = hooks.safety_poll() if hooks.safety_poll is not None else "AUTO"
        rec = {"step": steps, "t": round(t0, 3), "safety": safety_mode}
        if safety_mode == "EMERGENCY":
            reason = "EMERGENCY command -> motor cut"
            emit(rec, None, True, reason)
            break
        if safety_mode == "LAND":
            hooks.send_pcmd(0, 0, 0, 0)
            reason = "safety LAND command -> land"
            emit(rec, (0, 0, 0, 0), True, reason)
            break
        if safety_mode == "HOVER":
            hooks.send_pcmd(0, 0, 0, 0)
            emit(rec, (0, 0, 0, 0), True, "safety HOVER: zero PCMD")
            steps += 1
            dt = hooks.now() - t0
            if dt < period:
                time.sleep(period - dt)
            continue
        if safety_mode == "MANUAL":
            last_safety_mode = safety_mode                # MANUAL: send NOTHING (SkyController pilot has the sticks)
            emit(rec, None, True, "MANUAL: autonomy sends nothing (pilot has the sticks)")
            steps += 1
            dt = hooks.now() - t0
            if dt < period:
                time.sleep(period - dt)
            continue
        last_safety_mode = safety_mode

        now = hooks.now()
        if hooks.stream_healthy is not None and not hooks.stream_healthy():
            hooks.send_pcmd(0, 0, 0, 0)
            stream_lost_since = stream_lost_since or now
            lost_since = None
            rec["stream_ok"] = False
            if now - stream_lost_since >= STREAM_LOST_LAND_S:
                reason = (f"stream lost {now - stream_lost_since:.1f}s "
                          f">= {STREAM_LOST_LAND_S:.0f}s -> land")
                emit(rec, (0, 0, 0, 0), True, reason)
                break
            emit(rec, (0, 0, 0, 0), True,
                 f"stream stale/lost {now - stream_lost_since:.1f}s -> hover")
            if verbose and now - last_stream_warn > 1.0:
                detail = hooks.stream_status() if hooks.stream_status is not None else "stream stale/lost"
                print(
                    f"[safety] STREAM_LOST_HOVER: {detail}; "
                    f"lost_for={now - stream_lost_since:.1f}s; zero PCMD, waiting for stream/manual/land",
                    flush=True,
                )
                last_stream_warn = now
            steps += 1
            dt = hooks.now() - t0
            if dt < period:
                time.sleep(period - dt)
            continue
        stream_lost_since = None

        try:
            oyaw = hooks.olympe_yaw()
        except Exception:
            oyaw = None                                    # attitude not yet received
        try:
            ploc = hooks.get_pose()
        except Exception as exc:
            # A single raising frame (CUDA OOM, PnP blow-up, bad decode) must be a
            # missed fix -> hover, never an aborted mission.
            ploc = None
            if verbose:
                print(f"[safety] localizer raised ({exc!r}); treating as no fix -> hover", flush=True)
        now = hooks.now()
        # A NaN/inf pose would pass every `x > threshold` gate below (all False for
        # NaN) and steer the drone with garbage -> reject non-finite as no fix.
        pose_finite = ploc is not None and all(
            math.isfinite(float(v)) for v in (ploc.x, ploc.y, ploc.z, ploc.stamp))
        if ploc is not None and not pose_finite and verbose:
            print("[safety] NON_FINITE_POSE_REJECT: localizer returned NaN/inf -> hover", flush=True)
        if ploc is not None and pose_finite:
            rec["pose"] = [round(float(ploc.x), 3), round(float(ploc.y), 3), round(float(ploc.z), 3)]
            rec["pose_age"] = round(now - float(ploc.stamp), 3)
        elif ploc is not None:
            rec["pose"] = "non-finite"
        fresh = pose_finite and (now - ploc.stamp) <= POSE_STALE_S
        jump_rejected = False
        if fresh and last_good is not None:
            # Outlier gate: one spurious PnP pose must not steer the drone.
            # A single fix jumping > MAX_POSE_JUMP_U is rejected (hover); if the
            # NEXT fix agrees with it, treat it as genuine relocalization.
            cand = np.array([ploc.x, ploc.y, ploc.z], float)
            prev = np.array([last_good.x, last_good.y, last_good.z], float)
            if float(np.linalg.norm(cand - prev)) > MAX_POSE_JUMP_U:
                if (pending_jump is not None
                        and float(np.linalg.norm(cand - pending_jump)) <= MAX_POSE_JUMP_U):
                    pending_jump = None
                    heading.mark_teleport()   # don't let the relocation jump pollute the yaw offset
                    if verbose:
                        print("[safety] POSE_JUMP confirmed by consecutive fix; accepting relocation", flush=True)
                else:
                    pending_jump = cand
                    fresh = False
                    jump_rejected = True
                    rec["jump_reject_u"] = round(float(np.linalg.norm(cand - prev)), 2)
                    if verbose:
                        print(f"[safety] POSE_JUMP_REJECT: fix jumped "
                              f"{float(np.linalg.norm(cand - prev)):.2f}u > {MAX_POSE_JUMP_U}u; "
                              "hover, waiting for confirmation", flush=True)
            else:
                pending_jump = None
        if fresh:
            last_good = ploc            # lost_since resets only on a GOOD fix (drive branch), so the
                                        # low-conf timer accumulates across fresh-but-weak frames too
        elif (not jump_rejected) and last_good is not None and (now - last_good.stamp) <= POSE_STALE_S:
            # Missed/stale fix within the freshness window -> keep last accepted pose.
            # NOT on a jump reject: a contradicting fix means high uncertainty; hover
            # this tick (as documented above) instead of driving on the previous pose.
            ploc, fresh = last_good, True

        # Low confidence (WEAK, or PnP inliers below threshold) OR no fresh fix -> HOVER and
        # ask the tracker to relocalize with MegaLoc. If MegaLoc cannot recover within
        # LOST_MANUAL_S, hand to the human pilot (SkyController); else land at LOST_LAND_S.
        # The drone is never driven on a low-confidence fix.
        low_conf = bool(fresh and (
            (GATE_WEAK and hooks.pose_is_weak is not None and hooks.pose_is_weak())
            or (hooks.pose_confidence is not None and hooks.pose_confidence() < LOW_CONF_INLIERS)))
        if low_conf or not fresh:
            hooks.send_pcmd(0, 0, 0, 0)                    # hover; never fly on a weak/absent fix
            lost_since = lost_since or now
            if hooks.force_relocalize is not None:
                hooks.force_relocalize()                  # ask the tracker to run MegaLoc
            waited = now - lost_since
            emit(rec, (0, 0, 0, 0), True,
                 ("low confidence" if low_conf else
                  "pose jump rejected" if "jump_reject_u" in rec else
                  "no fresh pose (lost/stale/PnP fail/non-finite)")
                 + f" -> hover ({waited:.1f}s)")
            if (not manual_requested and waited >= LOST_MANUAL_S
                    and hooks.request_manual is not None and hooks.request_manual()):
                manual_requested = True
                print("[safety] LOW_CONF/LOST: MegaLoc could not relocalize in "
                      f"{waited:.1f}s -> switching to MANUAL (pilot takeover)", flush=True)
            elif waited >= LOST_LAND_S:
                reason = ("localization lost" if not fresh else "low confidence") + " -> land"
                break
            elif verbose and steps % 10 == 0:
                print(f"[safety] LOW_CONF_HOVER: hover + MegaLoc relocalize; waited={waited:.1f}s", flush=True)
            steps += 1
            dt = hooks.now() - t0
            if dt < period:
                time.sleep(period - dt)
            continue
        else:
            lost_since = None
            manual_requested = False
            C = np.array([ploc.x, ploc.y, ploc.z], float)
            if not seeded:
                # seed heading from route start (operator points nose along path[0->goal])
                _d, _n, _seg, s0 = rpf.project_to_path(C, ctrl.wp, ctrl.cum)
                goal, _gi = rpf.point_at_s(ctrl.wp, ctrl.cum,
                                           min(ctrl.path_len, s0 + ctrl.cfg.lookahead))
                heading.seed_from_path(C, goal, oyaw)
                seeded = True
            heading.update(C, oyaw)
            hdg = heading.heading(oyaw)
            if hdg is None:
                hooks.send_pcmd(0, 0, 0, 0)                # heading unknown -> hover
                emit(rec, (0, 0, 0, 0), True, "heading unavailable -> hover")
            else:
                pose = rpf.Pose(x=ploc.x, y=ploc.y, z=ploc.z, yaw=hdg, stamp=ploc.stamp)
                cmd = ctrl.step(pose, now)
                roll, pitch, yaw, gaz = rpf.command_to_body_percent(
                    cmd, pose, yaw_sign=yaw_sign)
                rec["heading_deg"] = round(math.degrees(hdg), 1)
                rec["path_error_u"] = round(float(cmd.path_error), 3)
                rec["progress"] = round(float(cmd.progress), 4)
                rec["action"] = cmd.status
                if cmd.path_error > MAX_ROUTE_DEVIATION_U:
                    # Route-corridor bound: never REJOIN across long distances.
                    hooks.send_pcmd(0, 0, 0, 0)
                    reason = (f"route deviation {cmd.path_error:.2f}u "
                              f"> {MAX_ROUTE_DEVIATION_U}u -> land")
                    emit(rec, (0, 0, 0, 0), True, reason)
                    break
                if cmd.should_land or ctrl.state in ("LANDING", "DONE"):
                    hooks.send_pcmd(0, 0, 0, 0)
                    reason = "route complete -> land"
                    emit(rec, (0, 0, 0, 0), True, reason)
                    break
                hooks.send_pcmd(roll, pitch, yaw, gaz)
                emit(rec, (roll, pitch, yaw, gaz), False, cmd.status)
                if verbose and steps % 10 == 0:
                    print(f"[{cmd.status:26s}] pos=({ploc.x:5.1f},{ploc.y:5.1f},{ploc.z:4.1f}) "
                          f"hdg={math.degrees(hdg):6.1f} err={cmd.path_error:4.2f} "
                          f"prog={cmd.progress:4.2f} PCMD(p={pitch:+d},y={yaw:+d},g={gaz:+d})")
        steps += 1
        dt = hooks.now() - t0
        if dt < period:
            time.sleep(period - dt)
    return reason


# ---------------------------------------------------------------------------
# Builders

def production_config():
    """The LOCKED production sweep (user's final decision, 2026-06). These already
    match ProductionConfig() defaults; pinned here so the deployment entrypoint is
    self-documenting and a future default change can't silently alter real flights.
        BOOT_INIT/LOST : MegaLoc top30 -> XFeat -> LighterGlue -> PnP
        TRACK/WEAK     : XFeat mutual-NN fast pass -> (if weak) LighterGlue adaptive 3->5 -> PnP
    """
    from production_xfeat_tracker import ProductionConfig
    return ProductionConfig(
        matcher_mode="nn_then_lg", nn_min_score=0.85,
        adaptive_first_topk=3, adaptive_accept_inliers=100, adaptive_accept_reproj=3.5,
        local_topk=5, weak_local_topk=8,
        xfeat_topk_track=1300, xfeat_topk_acquire=2048,
        boot_global_topk=30, lost_global_topk=30, weak_global_topk=0,
        pnp_ransac_max_error=5.0,
        max_reproj_error_track=6.0, max_reproj_error_acquire=5.0,
        max_corr_total=0, max_corr_per_ref=0,
        temporal_cache_enabled=True,
        temporal_cache_min_anchors=80,
        temporal_cache_max_anchors=2048,
        temporal_cache_max_age=2,
        temporal_cache_min_score=0.85,
        temporal_cache_seed_min_inliers=150,
        temporal_cache_seed_max_reproj=3.5,
    )


def build_localizer(frame_source, cam_tuple=CAM_720):
    from production_xfeat_tracker import ProductionXFeatTracker, MegaLocLayer
    from reloc_localizer_xfeat import XFeatRelocMap, Camera
    xmap = XFeatRelocMap.load(XBUN)          # fixed-pose triangulated XFeat layer, 1920 refs
    if xmap.ref_centers is None or len(xmap.ref_centers) != len(xmap.ref_names):
        raise SystemExit(
            f"bundle tracking metadata mismatch: refs={len(xmap.ref_names)} "
            f"ref_centers={0 if xmap.ref_centers is None else len(xmap.ref_centers)}. "
            "Regenerate/augment tracking metadata before real flight deployment."
        )
    try:
        meg = MegaLocLayer.load_cache(Path(MEG), xmap.ref_names, input_size=322)
    except Exception as exc:
        print(f"[flight] MegaLoc cache unavailable/incompatible ({exc}); using bundle ref_global")
        meg = MegaLocLayer(xmap.ref_global, input_size=322)
    cam = Camera(*cam_tuple)
    return ProductionXFeatTracker(xmap, meg, frame_source=frame_source,
                                  query_cam=cam, cfg=production_config())


def build_controller():
    import real_path_follow_controller as rpf
    wp = rpf.load_waypoints(PATH_JSON)
    poles = rpf.load_poles(POLES_JSON)
    ctrl = rpf.RouteAutoController(wp, poles)
    return ctrl, wp


# ---------------------------------------------------------------------------
# Real flight

def fly(ip: str, yaw_sign: int, gimbal_pitch: float, controller: str, safety_file: str,
        cmd_log_path: str = ""):
    import olympe_frame_source as ofs
    from olympe.messages.ardrone3.Piloting import TakeOff, PCMD, Landing, Emergency
    from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

    ctrl, wp = build_controller()
    clog = CommandLog(cmd_log_path or default_cmd_log_path("fly"), sink="olympe")
    print(f"[fly] structured command log: {clog.path}", flush=True)
    safety = SafetySwitch(
        safety_file,
        allow_manual=manual_override_available(ip, controller),
        keyboard=True,
    )
    # Fail fast BEFORE connecting/arming if the operator left a non-AUTO command.
    preflight_safety = safety.poll()
    if preflight_safety != "AUTO":
        raise SystemExit(
            f"[fly] preflight safety mode is {preflight_safety}; "
            "command safety-auto before takeoff"
        )

    # Bind everything the finally touches BEFORE the try, so a failure at connect,
    # stream start, model load, or takeoff can never raise UnboundLocalError in the
    # finally and strand an airborne drone with an abandoned connection.
    drone = None
    grab = None
    monitor = None
    airborne = False
    reason = ""
    send_pcmd = lambda *_: None
    stop = {"f": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("f", True))
    try:
        drone = ofs.connect(ip, controller=controller)
        grab = ofs.OlympePdrawGrabber(drone).start()
        loc = build_localizer(grab)
        loc.ensure_models()                          # load MegaLoc/XFeat NOW (on the ground)

        def send_pcmd(roll, pitch, yaw, gaz):
            # Single wire to the drone: clamp to the PCMD percent domain [-100, 100].
            r, p, y, g = (max(-100, min(100, int(v))) for v in (roll, pitch, yaw, gaz))
            drone(PCMD(1, r, p, y, g, 0))

        # Independent safety authority + stall watchdog: polls the safety switch and
        # can issue LAND/EMERGENCY/hover from its own thread even while the main loop
        # is blocked in synchronous get_pose() GPU inference.
        monitor = SafetyMonitor(
            send_pcmd, safety,
            land_cb=lambda: drone(Landing()),
            emergency_cb=lambda: drone(Emergency()),
        ).start()

        # user's flow: TAKEOFF -> hover -> gimbal -> BOOT_INIT MegaLoc lock -> START AUTO.
        # We lock WHILE HOVERING (camera at flight height/view, like the map refs), not
        # on the ground. If it never locks, we hover then land -- we never blind-fly AUTO.
        wait_success(drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=12)), "takeoff/hovering")
        airborne = True
        set_gimbal(drone, gimbal_pitch)
        print(f"[fly] BOOT_INIT: MegaLoc lock while hovering (<= {FIRST_FIX_TIMEOUT_S:.0f}s)...")

        def _boot_pose():
            try:
                return loc.get_pose()
            except Exception as exc:
                print(f"[fly] BOOT_INIT localizer raised ({exc!r}); retrying", flush=True)
                return None

        t0 = time.monotonic()
        while _boot_pose() is None and not stop["f"] and not monitor.terminated.is_set():
            send_pcmd(0, 0, 0, 0)                       # hold hover during acquisition
            if time.monotonic() - t0 > FIRST_FIX_TIMEOUT_S:
                raise SystemExit("[fly] never localized while hovering -> landing (no AUTO)")
            time.sleep(0.1)
        if monitor.terminated.is_set():
            reason = monitor.reason or "safety command before AUTO -> land"
        else:
            print(f"[fly] locked (mode={dict(loc.last_info).get('next_mode')}, "
                  f"inliers={dict(loc.last_info).get('inliers')}). START AUTO.")
            hooks = LoopHooks(
                get_pose=loc.get_pose,
                pose_is_weak=lambda: bool(dict(loc.last_info).get("weak", False)),
                pose_confidence=lambda: int(dict(loc.last_info).get("inliers", 0) or 0),
                force_relocalize=lambda: setattr(loc.state, "mode", "LOST"),  # run MegaLoc next deep frame
                request_manual=(lambda: safety.force("manual")) if safety.allow_manual else (lambda: False),
                olympe_yaw=lambda: olympe_yaw_of(drone),
                # Once the monitor has commanded LAND/EMERGENCY, a loop resuming from a
                # blocked get_pose() must NOT send a countermanding nonzero PCMD.
                send_pcmd=lambda r, p, y, g: (
                    None if (stop["f"] or monitor.terminated.is_set()) else send_pcmd(r, p, y, g)),
                safety_poll=lambda: monitor.mode,       # the monitor thread is the sole poller
                stream_healthy=grab.is_healthy,
                stream_status=lambda: (lambda age: (
                    "no 720p frame yet" if age is None
                    else f"last 720p frame age={age:.2f}s"
                ))(grab.last_frame_age()),
                loop_beat=monitor.beat,
                pose_info=lambda: dict(loc.last_info),
                log_tick=clog,
            )
            # run_loop returns when route done / lost / (Ctrl-C flips stop -> next hover then we break)
            reason = _run_until(hooks, ctrl, wp, yaw_sign, stop)
            if monitor.terminated.is_set():
                reason = monitor.reason or reason
        print(f"[fly] {reason}")
    finally:
        if monitor is not None:
            monitor.stop()
        if airborne and drone is not None:
            if reason.startswith("EMERGENCY"):
                print("[fly] EMERGENCY: cutting motors (drone will drop)", flush=True)
                try:
                    drone(Emergency()).wait()
                except Exception as exc:
                    print(f"[fly] Emergency() failed ({exc}); falling back to Landing()", flush=True)
                    try:
                        drone(Landing())
                    except Exception as exc2:
                        print(f"[fly] warning: Landing() also failed: {exc2}", flush=True)
            else:
                print("[fly] landing")
                try:
                    send_pcmd(0, 0, 0, 0)
                except Exception:
                    pass
                try:
                    # wait_success raises SystemExit (a BaseException) on a landing-
                    # confirmation timeout; catch it too, or grab.stop()/disconnect leak.
                    wait_success(drone(Landing()), "landing")
                except (Exception, SystemExit) as exc:
                    print(f"[fly] warning: landing did not confirm: {exc}", flush=True)
        elif drone is not None:
            print("[fly] not airborne; closing stream/connection without Landing()", flush=True)
        else:
            print("[fly] never connected; nothing to land", flush=True)
        if grab is not None:
            try:
                grab.stop()
            except Exception:
                pass
        if drone is not None:
            try:
                drone.disconnect()
            except Exception:
                pass
        try:
            clog.event(event="terminal", reason=reason)
        except Exception:
            pass
        clog.close()


def _run_until(hooks, ctrl, wp, yaw_sign, stop):
    """run_loop, but also honor the Ctrl-C stop flag between iterations."""
    orig_now = hooks.now
    def guarded_now():
        if stop["f"]:
            raise KeyboardInterrupt
        return orig_now()
    hooks.now = guarded_now
    try:
        return run_loop(hooks, ctrl, wp, yaw_sign=yaw_sign)
    except KeyboardInterrupt:
        return "operator Ctrl-C -> land"


# ---------------------------------------------------------------------------
# Grab-only: PROPS OFF localization on the live stream (never arms)

def grab_only(ip: str, secs: float, controller: str):
    import olympe_frame_source as ofs
    drone = None
    grab = None
    try:
        drone = ofs.connect(ip, controller=controller)
        grab = ofs.OlympePdrawGrabber(drone).start()
        loc = build_localizer(grab)
        print(f"[grab] localizing live for {secs:.0f}s (PROPS OFF, no arm)...")
        t0 = time.monotonic()
        n = ok = 0
        while time.monotonic() - t0 < secs:
            try:
                p = loc.get_pose()
            except Exception as exc:
                p = None
                print(f"[grab] localizer raised ({exc!r})", flush=True)
            info = dict(loc.last_info)
            mode = info.get("mode") or "?"           # None before the first frame -> "?"
            inl = info.get("inliers", 0) or 0
            n += 1
            if p is not None:
                ok += 1
                print(f"[grab] {mode:>10s} inl={inl:3d} "
                      f"pos=({p.x:6.1f},{p.y:6.1f},{p.z:5.1f})")
            else:
                print(f"[grab] {mode:>10s} inl={inl:3d} (no fix)")
            time.sleep(0.2)
        print(f"[grab] fix rate {ok}/{n}")
    finally:
        if grab is not None:
            try:
                grab.stop()
            except Exception:
                pass
        if drone is not None:
            try:
                drone.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dry-run: toy dynamics + a fake localizer, to exercise the loop with no hardware

def dry_run(yaw_sign: int, steps: int = 12000, cmd_log_path: str = ""):
    """Closed loop against a kinematic ANAFI stand-in in raw-GLOMAP frame.

    Body command -> world motion using the SAME sign conventions the real PCMD
    mapping assumes: +pitch = forward along heading, +gaz = up (-Y), +yaw = turn.
    A perfect localizer returns the sim state; heading fusion is skipped (no Olympe
    yaw) so run_loop falls back to the motion heading -- which is exactly what we
    want to smoke-test.
    """
    import real_path_follow_controller as rpf
    # pure path-follow smoke test: no poles / no inspection stops, so this exercises
    # the wiring + FOLLOW/REJOIN/LAND terminal, not the (separately-validated) look-at
    # inspection behavior.
    wp = rpf.load_waypoints(PATH_JSON)
    ctrl = rpf.RouteAutoController(wp, poles=[],
                                   config=rpf.ControlConfig(inspect_waypoints=()))

    # start on the first waypoint, heading roughly toward the second
    C = wp[0].astype(float).copy()
    h = math.atan2(wp[1][2] - wp[0][2], wp[1][0] - wp[0][0])
    KP, KG, KY = 0.06, 0.05, 0.06     # per-% per-step response (matches autoflight._Sim)

    class _State:
        pass
    S = _State(); S.C = C; S.h = h; S.t = 0.0
    dt = 1.0 / CTRL_HZ

    def get_pose():
        p = rpf.Pose(x=float(S.C[0]), y=float(S.C[1]), z=float(S.C[2]),
                     yaw=0.0, stamp=S.t)     # yaw here is ignored by run_loop's fusion
        return p

    def send_pcmd(roll, pitch, yaw, gaz):
        S.h = _wrap(S.h + KY * yaw_sign * yaw * dt)      # yaw turns heading
        fwd = KP * pitch * dt
        S.C[0] += fwd * math.cos(S.h)                    # +pitch -> forward in X/Z
        S.C[2] += fwd * math.sin(S.h)
        S.C[1] += -KG * gaz * dt                         # +gaz -> up == -Y
        S.t += dt

    # inject the sim time into run_loop so freshness math lines up
    clog = CommandLog(cmd_log_path or default_cmd_log_path("dryrun"), sink="dry-run")
    hooks = LoopHooks(get_pose=get_pose, olympe_yaw=lambda: None,
                      send_pcmd=send_pcmd, log_tick=clog, now=lambda: S.t)
    # cap iterations so a logic bug can't spin forever
    reason = _run_capped(hooks, ctrl, wp, yaw_sign, steps)
    clog.event(event="terminal", reason=reason)
    clog.close()
    _d, _n, _seg, s = rpf.project_to_path(S.C, ctrl.wp, ctrl.cum)
    print(f"[dry] end={reason}  progress={s/ctrl.path_len*100:4.0f}%  "
          f"final_pos=({S.C[0]:.2f},{S.C[1]:.2f},{S.C[2]:.2f})  state={ctrl.state}")
    print(f"[dry] structured command log: {clog.path}")
    return ctrl.state, s / ctrl.path_len


def _run_capped(hooks, ctrl, wp, yaw_sign, max_steps):
    orig_now = hooks.now
    box = {"n": 0}
    def counting_now():
        box["n"] += 1
        if box["n"] > max_steps:
            raise KeyboardInterrupt
        return orig_now()
    hooks.now = counting_now
    try:
        return run_loop(hooks, ctrl, wp, yaw_sign=yaw_sign, verbose=False)
    except KeyboardInterrupt:
        return "step cap"


# ---------------------------------------------------------------------------
# Self-test: heading fusion + PCMD sign sanity (pure python, no deps)

def selftest():
    import real_path_follow_controller as rpf

    # 1) heading fusion recovers the map<->body yaw offset from motion
    he = HeadingEstimator()
    oyaw = 0.3                                   # pretend Olympe body yaw (const)
    C = np.array([0.0, 0.0, 0.0])
    he.update(C, oyaw)
    true_map_heading = math.radians(40.0)        # body-forward points here in the map
    for _ in range(30):                          # walk forward along that heading
        C = C + 0.2 * np.array([math.cos(true_map_heading), 0.0, math.sin(true_map_heading)])
        he.update(C, oyaw)
    est = he.heading(oyaw)
    assert abs(_wrap(est - true_map_heading)) < math.radians(2), \
        f"heading fusion off: {math.degrees(est):.1f} vs {math.degrees(true_map_heading):.1f}"

    # 2) PCMD sign sanity: facing the goal -> forward pitch>0, small yaw; behind -> yaw drives, no forward
    pose = rpf.Pose(x=0, y=0, z=0, yaw=0.0)      # facing +X
    ahead = rpf.Command("FOLLOW", np.array([1.2, 0.0, 0.0]), yaw_target=0.0,
                        goal=np.zeros(3), path_error=0.0, progress=0.0)
    r, p, y, g = rpf.command_to_body_percent(ahead, pose)
    assert p > 0 and abs(y) < 3 and r == 0, (r, p, y, g)
    behind = rpf.Command("FOLLOW", np.array([-1.2, 0.0, 0.0]), yaw_target=math.pi,
                         goal=np.zeros(3), path_error=0.0, progress=0.0)
    r, p, y, g = rpf.command_to_body_percent(behind, pose)
    assert p == 0 and abs(y) == 25, ("should spin in place, not fly backward", r, p, y, g)

    # 3) +gaz == up (-Y): a "go up" command (target y lower) yields gaz>0
    up = rpf.Command("FOLLOW", np.array([0.0, -1.0, 0.0]), yaw_target=0.0,
                     goal=np.zeros(3), path_error=0.0, progress=0.0)
    _, _, _, g = rpf.command_to_body_percent(up, pose)
    assert g > 0, f"+gaz should be ascend, got {g}"

    # 4) stream lost must preempt localization and command hover.
    sent = []
    ticks = {"n": 0, "t": 0.0}

    def fake_now():
        ticks["n"] += 1
        ticks["t"] += 1.0 / CTRL_HZ
        if ticks["n"] > 24:
            raise KeyboardInterrupt
        return ticks["t"]

    def should_not_localize():
        raise AssertionError("stream-lost hover gate must run before get_pose()")

    hooks = LoopHooks(
        get_pose=should_not_localize,
        olympe_yaw=lambda: None,
        send_pcmd=lambda r, p, y, g: sent.append((r, p, y, g)),
        stream_healthy=lambda: False,
        stream_status=lambda: "selftest disconnected",
        now=fake_now,
    )
    try:
        run_loop(hooks, None, None, verbose=False)
    except KeyboardInterrupt:
        pass
    assert sent and all(cmd == (0, 0, 0, 0) for cmd in sent), sent

    # 5) a NaN/inf pose must be rejected (hover), never steer the drone.
    nan_sent = []
    nt = {"n": 0, "t": 0.0}

    def nan_now():
        nt["n"] += 1
        nt["t"] += 1.0 / CTRL_HZ
        if nt["n"] > 8:
            raise KeyboardInterrupt
        return nt["t"]

    wpn = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    ctrln = rpf.RouteAutoController(wpn, poles=[], config=rpf.ControlConfig(inspect_waypoints=()))
    hooks = LoopHooks(
        get_pose=lambda: rpf.Pose(x=float("nan"), y=0.0, z=0.0, yaw=0.0, stamp=nt["t"]),
        olympe_yaw=lambda: None,
        send_pcmd=lambda r, p, y, g: nan_sent.append((r, p, y, g)),
        now=nan_now,
    )
    try:
        run_loop(hooks, ctrln, wpn, verbose=False)
    except KeyboardInterrupt:
        pass
    assert nan_sent and all(cmd == (0, 0, 0, 0) for cmd in nan_sent), ("NaN pose must hover", nan_sent[:5])

    # 6) SafetyMonitor is an INDEPENDENT authority: a LAND command reaches land_cb
    #    and sets terminated with NO control loop running (models "main loop blocked
    #    in get_pose()"); a HOVER command streams zero PCMD.
    class _Seq:
        def __init__(self, seq):
            self.seq = list(seq)
            self.i = 0

        def poll(self):
            v = self.seq[min(self.i, len(self.seq) - 1)]
            self.i += 1
            return v

    mcalls = {"pcmd": [], "land": 0, "emergency": 0}
    mon = SafetyMonitor(
        lambda r, p, y, g: mcalls["pcmd"].append((r, p, y, g)),
        _Seq(["HOVER", "LAND"]),
        land_cb=lambda: mcalls.__setitem__("land", mcalls["land"] + 1),
        emergency_cb=lambda: mcalls.__setitem__("emergency", mcalls["emergency"] + 1),
        timeout_s=0.06,
    ).start()
    time.sleep(0.4)
    mon.stop()
    assert mcalls["land"] == 1, ("LAND must fire exactly once, independent of any loop", mcalls)
    assert mon.terminated.is_set(), "LAND must terminate the mission"
    assert (0, 0, 0, 0) in mcalls["pcmd"], "HOVER must stream zero PCMD from the monitor thread"

    # 7) MANUAL hands control to the physical pilot: the monitor must send NOTHING
    #    (a zero PCMD would fight the pilot's sticks).
    mcalls2 = {"pcmd": []}
    mon2 = SafetyMonitor(lambda r, p, y, g: mcalls2["pcmd"].append((r, p, y, g)),
                         _Seq(["MANUAL"]), timeout_s=0.06).start()
    time.sleep(0.3)
    mon2.stop()
    assert mcalls2["pcmd"] == [], ("MANUAL must send NOTHING", mcalls2)

    # 8) the stall watchdog must stay silent until the loop actually beats, so it can
    #    never false-fire and inject PCMD during the takeoff/BOOT_INIT window.
    mcalls3 = {"pcmd": []}
    mon3 = SafetyMonitor(lambda r, p, y, g: mcalls3["pcmd"].append((r, p, y, g)),
                         _Seq(["AUTO"]), timeout_s=0.06).start()
    time.sleep(0.3)
    mon3.stop()
    assert mcalls3["pcmd"] == [], ("stall watchdog must not fire before first beat", mcalls3)

    # 9) a low-confidence fix (inliers < threshold) must HOVER, ask the tracker to relocalize
    #    (force_relocalize -> MegaLoc), and after LOST_MANUAL_S hand to the pilot (request_manual);
    #    it must never drive the drone on a weak fix.
    lc = {"pcmd": [], "reloc": 0, "manual": 0, "t": 0.0}

    def lc_now():
        lc["t"] += 0.2
        if lc["t"] > LOST_LAND_S + 20.0:
            raise KeyboardInterrupt
        return lc["t"]

    wplc = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    ctrllc = rpf.RouteAutoController(wplc, poles=[], config=rpf.ControlConfig(inspect_waypoints=()))
    hooks = LoopHooks(
        get_pose=lambda: rpf.Pose(x=0.0, y=0.0, z=0.0, yaw=0.0, stamp=lc["t"]),
        olympe_yaw=lambda: None,
        send_pcmd=lambda r, p, y, g: lc["pcmd"].append((r, p, y, g)),
        pose_confidence=lambda: 30,                                  # < LOW_CONF_INLIERS (60)
        force_relocalize=lambda: lc.__setitem__("reloc", lc["reloc"] + 1),
        request_manual=lambda: (lc.__setitem__("manual", lc["manual"] + 1) or True),
        now=lc_now,
    )
    try:
        run_loop(hooks, ctrllc, wplc, verbose=False)
    except KeyboardInterrupt:
        pass
    assert lc["pcmd"] and all(cmd == (0, 0, 0, 0) for cmd in lc["pcmd"]), ("low-conf must hover", lc["pcmd"][:5])
    assert lc["reloc"] > 0, "low-conf must ask the tracker to relocalize (MegaLoc)"
    assert lc["manual"] == 1, ("MegaLoc-fail must hand to MANUAL exactly once", lc)

    print("selftest OK: heading fusion, PCMD signs, stream-lost + NaN-pose hover gates, "
          "SafetyMonitor LAND/HOVER authority, MANUAL-silence, pre-beat watchdog gating, "
          "and low-confidence hover+MegaLoc+MANUAL handoff sane")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ANAFI path-follow closed-loop flight (Olympe).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true", help="pure-python checks, no deps")
    mode.add_argument("--dry-run", action="store_true", help="toy dynamics, no drone")
    mode.add_argument("--grab-only", action="store_true", help="live localize, PROPS OFF, no arm")
    mode.add_argument("--fly", action="store_true", help="ARM + FLY (open space, safety pilot!)")
    ap.add_argument("--ip", default=DRONE_IP_REAL,
                    help=f"real {DRONE_IP_REAL} / skyctrl {DRONE_IP_SKYCTRL} / sphinx {DRONE_IP_SIM}")
    ap.add_argument("--yaw-sign", type=int, default=1, choices=(-1, 1),
                    help="flip if the drone yaws the WRONG way on bench (verify props-off)")
    ap.add_argument("--gimbal-pitch", type=float, default=GIMBAL_PITCH_DEG,
                    help="camera tilt vs horizon (deg, negative=down)")
    ap.add_argument("--controller", default=os.environ.get("SFM_OLYMPE_CONTROLLER", "auto"),
                    help="auto / drone / anafi / skycontroller3")
    ap.add_argument("--safety-file", default=SAFETY_FILE,
                    help="write auto/hover/manual/land here for runtime safety switching")
    ap.add_argument("--secs", type=float, default=20.0, help="--grab-only duration")
    ap.add_argument("--cmd-log", default="",
                    help="JSONL per-tick command log path (default: auto under 定位/outputs/flight_logs/)")
    args = ap.parse_args()

    if args.selftest:
        print("[mode] SELFTEST: no drone -- pure-python checks, no Olympe import, no motors", flush=True)
        selftest()
    elif args.dry_run:
        print("[mode] DRY-RUN: mock commands only -- no drone connection, no real Olympe command", flush=True)
        dry_run(args.yaw_sign, cmd_log_path=args.cmd_log)
    elif args.grab_only:
        print("[mode] GRAB-ONLY: live stream, props off, no arming -- connects and localizes, "
              "never sends TakeOff/PCMD", flush=True)
        grab_only(args.ip, args.secs, args.controller)
    elif args.fly:
        print("[mode] FLY: REAL DRONE COMMANDS ENABLED -- TakeOff/PCMD/Landing/Emergency live", flush=True)
        fly(args.ip, args.yaw_sign, args.gimbal_pitch, args.controller, args.safety_file,
            cmd_log_path=args.cmd_log)


if __name__ == "__main__":
    main()

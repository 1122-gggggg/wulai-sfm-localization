#!/usr/bin/env python3
"""REAL ANAFI closed-loop path-follow flight (Parrot Olympe / Ground SDK).

This is the ONE missing wiring piece. Every block below already exists and was
sim-validated separately; this file joins them into a single runnable real-flight
entrypoint on ONE Olympe connection:

    OlympePdrawGrabber (720p live stream)              [olympe_frame_source.py]
      -> site-selected EDM or XFeat production tracker
             MegaLoc retrieval -> local matching -> PnP, BOOT_INIT -> TRACK
      -> map-frame heading fusion  (visual camera ray <-> Olympe yaw)
      -> RouteAutoController.step(pose) -> Command      [real_path_follow_controller.py]
             FOLLOW / REJOIN-nearest-point / LAND  (keeps drone ON the drawn route)
      -> command_to_body_percent(cmd, pose) -> PCMD(roll, pitch, yaw, gaz)
      -> drone(PCMD(...))            same single connection used for the video

SDK      : Parrot Olympe (Ground SDK).  https://developer.parrot.com/docs/olympe/
Map      : fused forward+reverse GLOMAP; reloc bundle reloc_map_xfeat_tri.pt (1920 refs).
Path     : pre-drawn polyline  safezone/flight_path.json  (Blender Z-up waypoints).
Frame    : raw GLOMAP  horizontal = X/Z, gravity-up = -Y   (matches RouteAutoController).

Heading contract
----------------
The production localizer publishes pose.yaw from the camera forward ray projected
into the site's measured MapFrame. That visual heading anchors Olympe's faster NED
yaw after converting clockwise-from-North NED into counter-clockwise map heading.
Route direction and translation are never treated as heading measurements.

SAFETY  -- real flight is irreversible and can injure people or destroy the drone.
  --selftest  : pure-python checks (heading fusion + PCMD sign sanity). No deps.
  --dry-run   : no drone; a toy dynamics model closes the loop to exercise logic.
  --grab-only : connect + localize on the LIVE stream, PROPS OFF, never arms motors.
  --fly       : rejected. Live TakeOff is only allowed from the operator UI.
The aircraft's initial nose direction is measured, never assumed from the route.
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
    configured = os.environ.get("SFM_WORKSPACE_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / "定位演算法").is_dir() and (root / "控制介面程式").is_dir():
            return root
    for p in [start, *start.parents]:
        if (p / "定位演算法").is_dir() and (p / "控制介面程式").is_dir():
            return p
        if p.name == "sfm_system":
            return p
    return start.parents[3] if len(start.parents) > 3 else start


SYSTEM_ROOT = _find_system_root(FLIGHT_ROOT)
LOC_ROOT = (
    SYSTEM_ROOT / "定位演算法"
    if (SYSTEM_ROOT / "定位演算法").is_dir()
    else SYSTEM_ROOT / "定位"
)
MISSION_ROOT = (
    SYSTEM_ROOT / "控制介面程式"
    if (SYSTEM_ROOT / "控制介面程式").is_dir()
    else LOC_ROOT / "mission"
)
SOURCE_ROOT = LOC_ROOT / "source" / "sfm_glomap"
DEPLOY_ROOT = LOC_ROOT / "deploy_code" / "sfm_glomap_deploy"
if str(FLIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(FLIGHT_ROOT))
if DEPLOY_ROOT.is_dir() and str(DEPLOY_ROOT) not in sys.path:
    sys.path.append(str(DEPLOY_ROOT))

from localization_uncertainty import (  # noqa: E402
    LocalizationState,
    decide_localization_transition,
)
from landing_transition import (  # noqa: E402
    GROUND_SPEED_MAX_AGE_S,
    LANDING_SPEED_THRESHOLD_MPS,
    decide_route_completion_landing,
    landing_speed_allows_land,  # noqa: F401 - compatibility API
)
from heading_fusion import (  # noqa: E402
    OFFSET_EMA,  # noqa: F401 - compatibility API
    OlympeGroundSpeedTracker as OlympeGroundSpeedTracker,
    VISUAL_IMU_YAW_MISMATCH_RAD,  # noqa: F401 - compatibility API
    HeadingEstimator,
    olympe_yaw_of,
)
from safety_command import (  # noqa: E402
    default_safety_file as _default_safety_file,  # noqa: F401 - compatibility API
    prepare_safety_file as _prepare_safety_file,
    safety_file_from_environment as _safety_file_from_environment,
    validate_safety_directory as _validate_safety_directory,
    validate_safety_file_stat as _validate_safety_file_stat,
)
# 2026-08-06 audit: a hard-coded external-site probe used to sit here and,
# whenever that developer-specific directory happened to exist, silently swapped
# DEFAULT_BUNDLE and DEFAULT_MEGALOC_CACHE to the football-field site regardless of
# which site the operator had chosen. The live launcher refuses to start without an
# explicit --site-profile for exactly this reason ("field assets must never fall
# back to a different site map/route/bundle"); this module must not contradict it.
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
REFERENCE_INDEX = os.environ.get("SFM_REFERENCE_INDEX", "").strip()
REFERENCE_INDEX_SHA256 = os.environ.get(
    "SFM_REFERENCE_INDEX_SHA256", ""
).strip()
PATH_JSON = os.environ.get("SFM_FLIGHT_PATH_JSON", str(DEFAULT_PATH_JSON))
POLES_JSON = os.environ.get("SFM_POLES_JSON", str(DEFAULT_POLES_JSON))
MAP_ALIGN = os.environ.get("SFM_MAP_ALIGN", "").strip()
LOCALIZER_BACKEND = os.environ.get("SFM_LOCALIZER_BACKEND", "xfeat").strip().lower()
LOCALIZER_PROFILE = os.environ.get("SFM_LOCALIZER_PROFILE", "").strip()
BUNDLE_SHA256 = os.environ.get("SFM_BUNDLE_SHA256", "").strip()
FLIGHT_CONTRACT_JSON = os.environ.get("SFM_FLIGHT_CONTRACT_JSON", "").strip()

DRONE_IP_REAL = "192.168.42.1"
DRONE_IP_SKYCTRL = "192.168.53.1"
DRONE_IP_SIM = "10.202.0.1"                     # Parrot Sphinx simulator

# Same physical camera as the football-field map, calibrated at 1280x720.
# FULL_OPENCV params: fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6.
CAM_720 = (
    "FULL_OPENCV", 1280, 720,
    [
        960.4853099760471, 958.1961747147875,
        670.8167651412149, 358.7191813450141,
        -0.016359355216362784, 0.256336300878371,
        -0.006099082030819077, 0.019509803298460405,
        -0.1198628127364991, 0.0, 0.0, 0.0,
    ],
)


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def query_camera_from_environment(default=CAM_720, *, required: bool = False):
    raw = os.environ.get("SFM_QUERY_CAMERA_JSON", "").strip()
    if not raw:
        if required:
            raise ValueError("flight contract requires an explicit query camera")
        return default
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid SFM_QUERY_CAMERA_JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"model", "width", "height", "params"}:
        raise ValueError(
            "SFM_QUERY_CAMERA_JSON must contain model, width, height, and params"
        )
    return value["model"], value["width"], value["height"], value["params"]


def _require_approved_flight_contract(contract: dict) -> None:
    blockers = []
    if contract.get("schema_version") != 2:
        blockers.append("schema_version must be 2")
    if contract.get("approved") is not True:
        blockers.append("flight approval is missing")
    if contract.get("route_clearance_approved") is not True:
        blockers.append("route clearance approval is missing")
    for key in ("site_id", "coordinate_frame_id", "route_sha256"):
        if not str(contract.get(key) or "").strip():
            blockers.append(f"{key} is missing")
    if blockers:
        raise ValueError("autonomous flight contract rejected: " + "; ".join(blockers))


def flight_contract_from_environment(*, require_approved: bool) -> dict | None:
    if not FLIGHT_CONTRACT_JSON:
        if require_approved:
            raise ValueError("autonomous flight requires an explicit approved flight contract")
        return None
    try:
        contract = json.loads(
            FLIGHT_CONTRACT_JSON,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid SFM_FLIGHT_CONTRACT_JSON: {exc}") from exc
    if not isinstance(contract, dict):
        raise ValueError("SFM_FLIGHT_CONTRACT_JSON must be an object")
    if require_approved:
        _require_approved_flight_contract(contract)
    return contract


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {raw!r}") from exc
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be finite and in [{minimum}, {maximum}], got {raw!r}")
    return value


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {raw!r}")
    return value


CTRL_HZ = 20
POSE_STALE_S = 0.5          # visual pose older than this -> HOVER
# Reuse one accepted visual position only across a very short failed-frame run.
# The source timestamp must still satisfy POSE_STALE_S; this second bound limits
# how long control may continue after the localizer actually stopped producing fixes.
POSE_HOLDOVER_S = _env_float(
    "SFM_POSE_HOLDOVER_S", 0.20, minimum=0.0, maximum=POSE_STALE_S)
# 720p live frame older than this -> HOVER before localization. Distinct from the
# operator interface's _STREAM_STALE_S (0.75 s), which only feeds its stick-handoff
# timer; this one decides whether autonomy may run the localizer on the frame.
STREAM_STALE_S = 0.5
LOST_LAND_S = 4.0           # no fresh pose for this long -> auto-land
# Give MegaLoc a stationary retry window before moving the camera. The desktop
# AUTO adapter opts into one slow, telemetry-bounded clockwise turn.
LOST_YAW_SEARCH_DELAY_S = 10.0
LOST_YAW_SEARCH_TIMEOUT_S = 300.0
LOST_YAW_SEARCH_TARGET_RAD = 2.0 * math.pi
LOW_CONF_INLIERS = _env_int("SFM_LOW_CONF_INLIERS", 60, minimum=1, maximum=10000)
RECOVERY_GOOD_FIXES = _env_int(
    "SFM_RECOVERY_GOOD_FIXES", 2, minimum=1, maximum=20)
# A WEAK (low-confidence) fix in repetitive line-corridor geometry can be plausible
# but wrong and still pass the jump/deviation gates; treat it as "uncertain" -> hover.
# SFM_GATE_WEAK is retained for explicit offline benchmark/simulation callers only.
# The real-flight runner always enables this gate below, regardless of the env.
GATE_WEAK = os.environ.get("SFM_GATE_WEAK", "1") != "0"
REAL_FLIGHT_WEAK_POSE_GATE = True
WEAK_HOVER_LAND_S = _env_float(
    "SFM_WEAK_HOVER_LAND_S", 8.0, minimum=0.5, maximum=600.0)
# Reject a fresh fix that jumps farther than this from the last accepted fix
# (MAP UNITS). A second consecutive fix agreeing with the first one is accepted,
# so genuine relocalization after hover drift still recovers.
MAX_POSE_JUMP_U = _env_float(
    "SFM_MAX_POSE_JUMP_U", 1.5, minimum=0.01, maximum=100.0)
POSE_CONTINUITY_HISTORY = _env_int(
    "SFM_POSE_CONTINUITY_HISTORY", 5, minimum=2, maximum=20)
LOW_CONF_JUMP_WINDOW_S = _env_float(
    "SFM_LOW_CONF_JUMP_WINDOW_S", 2.0, minimum=0.1, maximum=30.0)
LOW_CONF_JUMP_HOVER_COUNT = _env_int(
    "SFM_LOW_CONF_JUMP_HOVER_COUNT", 3, minimum=2, maximum=20)
# Route-corridor bound (MAP UNITS): if the accepted pose ends up farther than
# this from the drawn route, abort and land instead of REJOIN-ing blindly.
MAX_ROUTE_DEVIATION_U = _env_float(
    "SFM_MAX_ROUTE_DEVIATION_U", 3.0, minimum=0.01, maximum=100.0)
# If the control loop stalls longer than this (model/GPU hang), a helper thread
# forces zero PCMD so the drone hovers instead of holding the last command.
WATCHDOG_LAND_S = _env_float(
    "SFM_WATCHDOG_LAND_S", 5.0, minimum=0.5, maximum=120.0)
PCMD_WATCHDOG_S = _env_float(
    "SFM_PCMD_WATCHDOG_S", 0.7, minimum=0.1, maximum=10.0)
PCMD_CONTROL_HZ = _env_float(
    "SFM_PCMD_CONTROL_HZ", 20.0, minimum=5.0, maximum=50.0)
PCMD_COMMAND_TTL_S = _env_float(
    "SFM_PCMD_COMMAND_TTL_S", 0.15, minimum=0.01, maximum=0.15)
FIRST_FIX_TIMEOUT_S = 25.0  # how long to wait for BOOT_INIT -> first TRACK
AUTO_CONSENT_TIMEOUT_S = 30.0
BOOT_LOCK_FIXES = 3
BOOT_START_MAX_U = _env_float(
    "SFM_BOOT_START_MAX_U", 1.5, minimum=0.05, maximum=100.0)
# The ANAFI whitepaper's 280 ms is a video latency lower bound, not a safe
# pipeline upper bound. Drain a conservative, explicitly tunable source-clock
# interval before an inspection frame can be acknowledged.
INSPECTION_PIPELINE_DRAIN_S = _env_float(
    "SFM_INSPECTION_PIPELINE_DRAIN_S", 1.0, minimum=0.3, maximum=5.0)
GIMBAL_PITCH_DEG = -10.0    # camera tilt vs horizon; slightly down, like the map refs


SAFETY_FILE = str(_safety_file_from_environment())


class SafetySwitch:
    """Runtime safety switch for AUTO / HOVER / MANUAL / LAND.

    Commands can come from either:
      - terminal stdin: a/auto, h/hover, m/manual, l/land;
      - a small owner-only text file, defaulting to the per-user runtime dir.

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
                 allow_manual: bool = False, keyboard: bool = True,
                 require_fresh_auto: bool = False):
        self.path = Path(path) if path else None
        self.allow_manual = bool(allow_manual)
        self.keyboard = bool(keyboard and sys.stdin and sys.stdin.isatty())
        self.require_fresh_auto = bool(require_fresh_auto)
        self.mode = "HOVER"                 # fail closed until a readable command is applied
        self._mtime_ns: int | None = None
        self._file_key: tuple[int, int, int] | None = None
        self._last_print = 0.0
        self._run_start_mtime_ns: int | None = None
        self._stale_file_warned = False
        if self.path:
            st = _prepare_safety_file(self.path)
            self._run_start_mtime_ns = st.st_mtime_ns
            # Leave any existing LAND/HOVER/MANUAL/EMERGENCY command untouched.
            # _mtime_ns stays None so the first poll applies the on-disk value.
        print(
            "[safety] commands: a/auto, h/hover, m/manual, l/land, e/emergency(motor cut); "
            f"file={self.path or 'disabled'} manual_allowed={self.allow_manual}",
            flush=True,
        )

    def _read_file_command(self) -> str | None:
        if self.path is None:
            return None
        try:
            _validate_safety_directory(self.path.parent)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.path, flags)
            with os.fdopen(fd, "r", encoding="utf-8", errors="ignore") as handle:
                st = os.fstat(handle.fileno())
                _validate_safety_file_stat(self.path, st)
                file_key = (st.st_dev, st.st_ino, st.st_mtime_ns)
                if self._file_key == file_key:
                    return None
                self._file_key = file_key
                self._mtime_ns = st.st_mtime_ns
                parts = handle.read().strip().split()
        except FileNotFoundError:
            self._mtime_ns = None
            self._file_key = None
            return "hover"
        except Exception as exc:
            print(f"[safety] cannot read safety file: {exc}", flush=True)
            return "hover"

        try:
            token = parts[0].lower() if parts else "hover"
            if token not in self._ALIASES:
                print(f"[safety] invalid file command={token!r}; failing closed to HOVER", flush=True)
                return "hover"
            if (self._run_start_mtime_ns is not None
                    and self._mtime_ns <= self._run_start_mtime_ns):
                if not self._stale_file_warned:
                    self._stale_file_warned = True
                    print("[safety] stale file command predates this run; write a fresh command", flush=True)
                return "hover"
            return token
        except Exception as exc:
            print(f"[safety] cannot read safety file: {exc}", flush=True)
            return "hover"

    def _read_keyboard_command(self) -> str | None:
        if not self.keyboard:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return None
            parts = sys.stdin.readline().strip().split()
            return parts[0].lower() if parts else "hover"
        except Exception as exc:
            print(f"[safety] keyboard read failed ({exc!r}); failing closed to HOVER", flush=True)
            return "hover"

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


def arming_allowed(safety_mode: str, stream_healthy: bool,
                   stop_requested: bool, terminated: bool) -> tuple[bool, str]:
    if terminated:
        return False, "safety authority already terminated the mission"
    if stop_requested:
        return False, "operator stop signal is set"
    if str(safety_mode).upper() != "AUTO":
        return False, f"safety mode is {safety_mode}, not AUTO"
    if not stream_healthy:
        return False, "720p stream is not healthy"
    return True, "AUTO + healthy stream + no stop/termination"


class BootPoseLock:
    """Require stable, fresh fixes close to any waypoint on the route."""

    def __init__(self, route_waypoints, required_fixes: int = BOOT_LOCK_FIXES,
                 max_start_distance: float = BOOT_START_MAX_U,
                 max_fix_jump: float = MAX_POSE_JUMP_U):
        points = np.asarray(route_waypoints, dtype=float)
        if points.shape == (3,):
            points = points.reshape(1, 3)
        if (
            points.ndim != 2
            or points.shape[0] < 1
            or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("BOOT route waypoints must be finite 3-vectors")
        self.route_waypoints = np.array(points, dtype=float, copy=True)
        self.route_start = self.route_waypoints[0]
        self.required_fixes = int(required_fixes)
        self.max_start_distance = float(max_start_distance)
        self.max_fix_jump = float(max_fix_jump)
        if (self.required_fixes < 2
                or not all(math.isfinite(v) and v > 0.0
                           for v in (self.max_start_distance, self.max_fix_jump))):
            raise ValueError("BOOT lock thresholds must be finite/positive and require >=2 fixes")
        self.count = 0
        self._last = None
        self.nearest_waypoint_index: int | None = None

    @property
    def position(self) -> np.ndarray | None:
        return None if self._last is None else self._last.copy()

    def reset(self) -> None:
        self.count = 0
        self._last = None
        self.nearest_waypoint_index = None

    def observe(self, pose, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        try:
            values = np.array([pose.x, pose.y, pose.z, pose.stamp], dtype=float)
        except (AttributeError, TypeError, ValueError):
            self.reset()
            return False
        age = now - float(values[3])
        pos = values[:3]
        distances = np.linalg.norm(self.route_waypoints - pos, axis=1)
        nearest_index = int(np.argmin(distances))
        valid = (math.isfinite(now) and np.isfinite(values).all()
                 and -0.05 <= age <= POSE_STALE_S
                 and float(distances[nearest_index]) <= self.max_start_distance
                 and (self._last is None
                      or float(np.linalg.norm(pos - self._last)) <= self.max_fix_jump))
        if not valid:
            self.reset()
            return False
        self._last = pos.copy()
        self.nearest_waypoint_index = nearest_index
        self.count += 1
        return self.count >= self.required_fixes


ACTION_WAIT_TIMEOUT_S = 5.0


def _bounded_wait(wait, timeout_s: float, label: str):
    """Call an SDK wait without ever allowing an unbounded caller-side wait."""
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise RuntimeError(f"{label} timeout must be finite and positive")
    try:
        return wait(_timeout=timeout)
    except TypeError:
        # Small test doubles and older SDKs may not accept _timeout. Run their
        # legacy wait signature on a daemon worker so a stuck expectation cannot
        # hold the safety thread or its I/O lock forever.
        result = {}
        done = threading.Event()

        def invoke():
            try:
                result["value"] = wait()
            except BaseException as exc:  # noqa: BLE001 - propagate below
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(target=invoke, name=f"bounded-wait:{label}", daemon=True).start()
        if not done.wait(timeout):
            raise TimeoutError(f"{label} wait exceeded {timeout:.2f}s")
        if "error" in result:
            raise result["error"]
        return result.get("value")


def _bounded_call(callback, timeout_s: float, label: str):
    """Run an SDK callback on a daemon worker with a hard caller timeout."""
    result = {}
    done = threading.Event()

    def invoke():
        try:
            result["value"] = callback()
        except BaseException as exc:  # noqa: BLE001 - propagate below
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=invoke, name=f"bounded-call:{label}", daemon=True).start()
    if not done.wait(float(timeout_s)):
        raise TimeoutError(f"{label} callback exceeded {float(timeout_s):.2f}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _await_confirmed_action(result, label: str, timeout_s: float = ACTION_WAIT_TIMEOUT_S) -> None:
    """Accept explicit local success or a confirmed Olympe expectation."""
    if result is None or result is True:
        return
    if result is False:
        raise RuntimeError(f"{label} returned False")
    wait = getattr(result, "wait", None)
    if not callable(wait):
        raise RuntimeError(f"{label} returned an unconfirmed result")
    waited = _bounded_wait(wait, timeout_s, label)
    confirmation = result if waited is None else waited
    success = getattr(confirmation, "success", None)
    if not callable(success) or not success():
        raise RuntimeError(f"{label} failed or timed out")


def wait_success(expectation, label: str) -> None:
    try:
        _await_confirmed_action(expectation, label)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def set_piloting_source(drone, source: str) -> bool:
    """Set and read back SkyController CoPiloting ownership."""
    from olympe.messages.skyctrl.CoPiloting import setPilotingSource, pilotingSource

    requested = str(source)
    _await_confirmed_action(
        drone(setPilotingSource(source=requested)),
        f"setPilotingSource({requested})",
    )
    state = drone.get_state(pilotingSource)
    actual = state.get("source") if isinstance(state, dict) else None
    actual_name = str(getattr(actual, "name", actual)).rsplit(".", 1)[-1]
    if actual_name.lower() != requested.lower():
        raise RuntimeError(
            f"piloting source readback mismatch: requested={requested} actual={actual_name}")
    return True


def require_landed_for_firmware_config(drone) -> None:
    """Keep the aircraft-state gate separate from advisory firmware setup."""
    from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

    flying_state = drone.get_state(FlyingStateChanged)
    state = flying_state.get("state") if isinstance(flying_state, dict) else None
    state_name = str(getattr(state, "name", state)).rsplit(".", 1)[-1].lower()
    if state_name != "landed":
        raise RuntimeError(
            f"firmware limits may only be changed while landed; current state={state_name}")


def _positive_firmware_limit(value: float | None, label: str) -> float:
    if value is None:
        raise RuntimeError("firmware altitude/distance limits were not provided")
    limit = float(value)
    if not math.isfinite(limit) or limit <= 0.0:
        raise RuntimeError(f"{label} must be finite and > 0 m")
    return limit


def _preflight_battery_percent(drone, battery_message) -> int:
    state = drone.get_state(battery_message)
    battery = state.get("percent") if isinstance(state, dict) else None
    if not isinstance(battery, (int, float)) or not 0 <= float(battery) <= 100:
        raise RuntimeError(f"battery state unavailable/invalid: {battery!r}")
    if float(battery) < 30.0:
        raise RuntimeError(f"battery {float(battery):.0f}% is below the 30% advisory floor")
    return int(battery)


def _require_distance_geofence_gps(drone) -> None:
    from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged

    state = drone.get_state(GPSFixStateChanged)
    fixed = state.get("fixed") if isinstance(state, dict) else None
    if int(fixed or 0) != 1:
        raise RuntimeError("distance geofence requires a confirmed GPS fix")


def _confirm_firmware_limit(
    drone,
    command,
    state_message,
    requested: float,
    label: str,
) -> None:
    before = drone.get_state(state_message)
    minimum = before.get("min") if isinstance(before, dict) else None
    maximum = before.get("max") if isinstance(before, dict) else None
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (minimum, maximum)
    ):
        raise RuntimeError(f"{label} firmware bounds unavailable")
    if not float(minimum) <= requested <= float(maximum):
        raise RuntimeError(
            f"{label} {requested:g} outside firmware range "
            f"[{float(minimum):g}, {float(maximum):g}]"
        )
    _await_confirmed_action(drone(command), label)
    after = drone.get_state(state_message)
    actual = after.get("current") if isinstance(after, dict) else None
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), requested, rel_tol=1e-5, abs_tol=0.05
    ):
        raise RuntimeError(
            f"{label} readback mismatch: requested={requested:g} actual={actual!r}"
        )


def _configure_distance_geofence(drone, command_type, state_message, enabled: bool) -> None:
    requested = int(bool(enabled))
    _await_confirmed_action(
        drone(command_type(shouldNotFlyOver=requested)),
        "NoFlyOverMaxDistance",
    )
    state = drone.get_state(state_message)
    actual = state.get("shouldNotFlyOver") if isinstance(state, dict) else None
    if int(actual if actual is not None else -1) != requested:
        raise RuntimeError(
            "NoFlyOverMaxDistance readback mismatch: "
            f"requested={requested} actual={actual!r}"
        )


def configure_flight_preflight(drone, max_altitude_m: float, max_distance_m: float,
                               distance_geofence: bool = False, *,
                               check_landed: bool = True) -> dict:
    """Check battery and try to apply optional firmware flight limits."""
    if check_landed:
        require_landed_for_firmware_config(drone)
    from olympe.messages.ardrone3.PilotingSettings import (
        MaxAltitude, MaxDistance, NoFlyOverMaxDistance,
    )
    from olympe.messages.ardrone3.PilotingSettingsState import (
        MaxAltitudeChanged, MaxDistanceChanged, NoFlyOverMaxDistanceChanged,
    )
    from olympe.messages.common.CommonState import BatteryStateChanged

    altitude = _positive_firmware_limit(max_altitude_m, "max altitude")
    distance = _positive_firmware_limit(max_distance_m, "max distance")
    battery = _preflight_battery_percent(drone, BatteryStateChanged)

    if distance_geofence:
        _require_distance_geofence_gps(drone)

    _confirm_firmware_limit(
        drone,
        MaxAltitude(current=altitude),
        MaxAltitudeChanged,
        altitude,
        "MaxAltitude",
    )
    _confirm_firmware_limit(
        drone,
        MaxDistance(value=distance),
        MaxDistanceChanged,
        distance,
        "MaxDistance",
    )
    _configure_distance_geofence(
        drone,
        NoFlyOverMaxDistance,
        NoFlyOverMaxDistanceChanged,
        distance_geofence,
    )
    return {
        "battery_percent": battery,
        "max_altitude_m": altitude,
        "max_distance_m": distance,
        "distance_geofence": bool(distance_geofence),
    }


# ---------------------------------------------------------------------------
# Map-frame heading: Olympe yaw (fast) anchored to measured visual orientation

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def inspection_gimbal_pitch_deg(meta: dict, pose, map_frame=None) -> float | None:
    """Target pitch using the site's measured horizontal and vertical axes."""
    import real_path_follow_controller as rpf
    frame = map_frame or rpf.LEGACY_MAP_FRAME
    try:
        target = np.asarray(meta["target"], dtype=float)
        camera = np.array([pose.x, pose.y, pose.z], dtype=float)
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(camera).all():
        return None
    delta = target - camera
    horizontal = frame.horizontal_distance(delta)
    delta_up = frame.vertical(delta)
    pitch = math.degrees(math.atan2(delta_up, horizontal))
    return pitch if math.isfinite(pitch) else None


def _gimbal_state_value(state, key: str):
    if isinstance(state, dict) and key in state:
        return state.get(key)
    if isinstance(state, dict):
        for value in state.values():
            if isinstance(value, dict) and value.get("gimbal_id", 0) == 0 and key in value:
                return value.get(key)
    return None


def _valid_gimbal_request(target: float, timeout_s: float, tolerance_deg: float) -> bool:
    return (
        math.isfinite(target)
        and math.isfinite(timeout_s)
        and timeout_s > 0.0
        and math.isfinite(tolerance_deg)
        and tolerance_deg > 0.0
    )


def _gimbal_target_in_bounds(drone, state_message, target: float) -> bool:
    try:
        bounds = drone.get_state(state_message)
        minimum = float(_gimbal_state_value(bounds, "min_pitch"))
        maximum = float(_gimbal_state_value(bounds, "max_pitch"))
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    return math.isfinite(minimum) and math.isfinite(maximum) and minimum <= target <= maximum


def _send_gimbal_target(drone, command_type, target: float, timeout_s: float) -> bool:
    try:
        result = _bounded_wait(
            drone(
                command_type(
                    gimbal_id=0,
                    control_mode="position",
                    yaw_frame_of_reference="none",
                    yaw=0.0,
                    pitch_frame_of_reference="absolute",
                    pitch=target,
                    roll_frame_of_reference="none",
                    roll=0.0,
                )
            ),
            timeout_s,
            "gimbal target",
        )
    except Exception:
        return False
    if hasattr(result, "success") and not result.success():
        print("[flight] warning: gimbal target command did not report success", flush=True)
        return False
    return True


def _wait_for_gimbal_pitch(
    drone,
    state_message,
    target: float,
    timeout_s: float,
    tolerance_deg: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            state = drone.get_state(state_message)
            actual = float(_gimbal_state_value(state, "pitch_absolute"))
        except (KeyError, TypeError, ValueError, RuntimeError):
            actual = float("nan")
        if math.isfinite(actual) and abs(actual - target) <= tolerance_deg:
            return True
        time.sleep(0.05)
    return False


def set_gimbal(drone, pitch_deg: float, *, require_confirmation: bool = False,
               timeout_s: float = 2.0, tolerance_deg: float = 3.0):
    """Point the camera to a fixed absolute (horizon-referenced) tilt so live frames
    resemble the map reference views. Absolute frame == EIS-stabilized vs horizon."""
    target = float(pitch_deg)
    timeout = float(timeout_s)
    tolerance = float(tolerance_deg)
    if not _valid_gimbal_request(target, timeout, tolerance):
        return False
    from olympe.messages.gimbal import absolute_attitude_bounds, attitude, set_target
    if require_confirmation and not _gimbal_target_in_bounds(
        drone, absolute_attitude_bounds, target
    ):
        return False
    if not _send_gimbal_target(drone, set_target, target, timeout):
        return False
    return (
        _wait_for_gimbal_pitch(drone, attitude, target, timeout, tolerance)
        if require_confirmation
        else True
    )


# ---------------------------------------------------------------------------
# The closed loop (shared by real flight and --dry-run via the `drone_io` adapter)

@dataclass
class LoopHooks:
    get_pose: callable          # () -> localizer Pose | None   (has .x,.y,.z,.stamp)
    olympe_yaw: callable        # () -> float | None
    send_pcmd: callable         # (roll,pitch,yaw,gaz) -> None
    send_authorized_pcmd: callable | None = None  # atomic ((r,p,y,g)) -> (sent, reason, actual_pcmd)
    pose_is_weak: callable | None = None  # () -> bool; True if the last fix was a WEAK track
    pose_confidence: callable | None = None  # () -> int; last-fix PnP inliers (low -> hover + relocalize)
    force_relocalize: callable | None = None  # () -> None; ask the tracker to run MegaLoc (LOST)
    request_manual: callable | None = None  # () -> bool; physical-stick/manual safety handoff
    stick_active: callable | None = None    # () -> bool; the pilot has physically moved the sticks
    safety_poll: callable | None = None  # () -> AUTO/HOVER/MANUAL/LAND/EMERGENCY
    stream_healthy: callable | None = None  # () -> bool; false means live stream stale/lost
    stream_status: callable | None = None   # () -> str; operator log detail
    loop_beat: callable | None = None       # () -> None; watchdog heartbeat, called every tick
    pose_info: callable | None = None       # () -> dict; localizer last_info (LOGGING ONLY)
    pose_jump_pause: callable | None = None  # (distance_map_units) -> latch operator HOVER
    inspection_ack: callable | None = None  # (metadata, pose) -> bool after gimbal+capture success
    inspection_hold: callable | None = None  # (blocking_action) -> bool while zero PCMD is sustained
    pcmd_timing: callable | None = None       # () -> latest desired/wire monotonic_ns markers
    ground_speed: callable | None = None      # () -> (horizontal_mps, monotonic_stamp) | None
    log_tick: callable | None = None        # (dict) -> None; structured per-tick command log
    land_on_localization_loss: bool = True  # false: zero PCMD and wait for recovery/operator
    localization_yaw_search_pcmd: int = 0  # 0 disables; positive ANAFI yaw turns right
    localization_yaw_search_event: callable | None = None  # (state, progress_deg) -> None
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
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
            pass


def default_cmd_log_path(tag: str) -> Path:
    return (SYSTEM_ROOT / "outputs" / "flight_logs"
            / f"{tag}_cmdlog_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


class SafetyMonitor:
    """Independent safety-authority + stall watchdog thread.

    Runs OUTSIDE the control loop so operator commands act even when the main
    thread is blocked in synchronous GPU inference (get_pose) or an Olympe wait.
    The loop runs localization inference synchronously; a model/GPU hang would
    otherwise (a) leave the drone holding its last non-zero command and (b) stop
    the loop from ever polling the safety switch again. This thread fixes both:

      - emits the latest authorized desired PCMD independently at 20 Hz;
      - polls the SafetySwitch (file/keyboard) at least 20 Hz -- the ONLY
        reader of stdin/file, so the main loop just reads cached ``mode``;
      - EMERGENCY  -> emergency_cb() once (motor cut) and set ``terminated``;
      - LAND       -> zero PCMD then land_cb() once and set ``terminated``;
      - HOVER     -> stream zero PCMD regardless of loop state;
      - MANUAL    -> send nothing so the physical pilot owns the sticks;
      - AUTO       -> if the loop has not beaten within ``timeout`` (stall/GPU
        hang) force zero PCMD; otherwise leave control to the loop.

    So a hung get_pose() can no longer block operator LAND/EMERGENCY: this thread
    issues them directly. It cannot interrupt the stuck C-level CUDA call, but the
    drone is commanded safe from here while the main thread is still blocked.
    """

    def __init__(self, send_pcmd, safety=None, land_cb=None, emergency_cb=None,
                 stop_requested=None, timeout_s: float = PCMD_WATCHDOG_S,
                 piloting_source_cb=None,
                 command_ttl_s: float = PCMD_COMMAND_TTL_S,
                 max_translation_pcmd: int = 10,
                 max_yaw_pcmd: int = 20):
        self._send = send_pcmd
        self._safety = safety
        self._land_cb = land_cb
        self._emergency_cb = emergency_cb
        self._stop_requested = stop_requested
        self._timeout = float(timeout_s)
        if not math.isfinite(self._timeout) or not 0.01 <= self._timeout <= 10.0:
            raise ValueError("SafetyMonitor timeout must be finite and in [0.01, 10.0] seconds")
        self._command_ttl_s = float(command_ttl_s)
        if (not math.isfinite(self._command_ttl_s)
                or not 0.01 <= self._command_ttl_s <= 0.15):
            raise ValueError("command_ttl_s must be finite and in [0.01, 0.15] seconds")
        self._max_translation_pcmd = int(max_translation_pcmd)
        self._max_yaw_pcmd = int(max_yaw_pcmd)
        if (isinstance(max_translation_pcmd, bool)
                or isinstance(max_yaw_pcmd, bool)
                or self._max_translation_pcmd != max_translation_pcmd
                or self._max_yaw_pcmd != max_yaw_pcmd
                or not 1 <= self._max_translation_pcmd <= 100
                or not 1 <= self._max_yaw_pcmd <= 100):
            raise ValueError("PCMD limits must be integers in [1,100]")
        self._beat_t = time.monotonic()
        self._beat_seen = False              # stall watchdog stays off until the loop beats
        # No SafetySwitch/first poll means no valid operator command. Never
        # expose an optimistic AUTO default while the command channel is still
        # unknown or unavailable.
        self.mode = "HOVER"
        self.reason = ""
        self.terminal_action = "NONE"
        self._terminal_reason = ""
        self.terminated = threading.Event()
        self._land_acted = False
        self._emergency_acted = False
        self.emergency_issued = False
        self.action_failures = {"LAND": 0, "EMERGENCY": 0}
        self.last_action_error = {"LAND": "", "EMERGENCY": ""}
        self._action_retry_s = max(0.1, self._timeout)
        self._land_retry_at = 0.0
        self._emergency_retry_at = 0.0
        self._piloting_source_cb = piloting_source_cb
        # A callback is installed only after fly() has confirmed Controller ownership.
        self._piloting_source = "Controller" if piloting_source_cb is not None else None
        self._piloting_source_retry_at = 0.0
        self.piloting_source_failures = 0
        self.last_piloting_source_error = ""
        self._inspection_hold = threading.Event()
        self._stop = threading.Event()
        self._control_wake = threading.Event()
        self._control_period_s = 1.0 / PCMD_CONTROL_HZ
        self._desired_pcmd = (0, 0, 0, 0)
        self._desired_valid = False
        self._desired_stream_healthy = None
        self._desired_stop_requested = None
        self._desired_updated_mono_ns: int | None = None
        self.last_pcmd_call_mono_ns: int | None = None
        self.pcmd_send_failures = 0
        self.thread_died_error: str | None = None
        self.last_pcmd_send_error = ""
        self._stall_since = None
        self._io_lock = threading.RLock()
        self._thread = threading.Thread(target=self._run, name="safety-monitor", daemon=True)

    def _send_counted(self, roll: int, pitch: int, yaw: int, gaz: int) -> bool:
        """Send one PCMD, stamping the wire time only on SUCCESS.

        Stamping before the attempt made a dead command channel indistinguishable
        from a healthy one: every failed send still advanced last_pcmd_call_mono_ns,
        so the flight log recorded commands that never reached the aircraft and the
        failures were counted nowhere. Failures are still non-fatal here -- the
        monitor must keep running -- but they are now visible.
        """
        try:
            self._send(roll, pitch, yaw, gaz)
        except Exception as exc:
            self.pcmd_send_failures += 1
            self.last_pcmd_send_error = repr(exc)
            if self.pcmd_send_failures == 1 or self.pcmd_send_failures % 20 == 0:
                print(f"[safety] PCMD SEND FAILED x{self.pcmd_send_failures} ({exc!r}); "
                      "the command channel may be down", flush=True)
            return False
        self.last_pcmd_call_mono_ns = time.monotonic_ns()
        return True

    def beat(self) -> None:
        self._beat_t = time.monotonic()
        self._beat_seen = True

    def start(self) -> "SafetyMonitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._control_wake.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(0.2, self._timeout))

    def request_manual_override(self) -> tuple[bool, str]:
        """Zero autonomy, then confirm that the physical pilot owns the sticks."""
        with self._io_lock:
            if self.terminal_action != "NONE":
                return False, f"terminal action {self.terminal_action} already latched"
            if self.mode == "MANUAL" and self._piloting_source == "SkyController":
                return True, "manual override already active"

            self._desired_pcmd = (0, 0, 0, 0)
            self._desired_valid = False
            zero_ok = self._send_counted(0, 0, 0, 0)
            try:
                manual_ok = bool(
                    self._piloting_source_cb is not None
                    and self._safety is not None
                    and self._safety.force("manual")
                )
            except Exception as exc:
                manual_ok = False
                self.last_piloting_source_error = repr(exc)
            self.mode = "MANUAL" if manual_ok else "HOVER"
            if not manual_ok:
                self._latch_terminal(
                    "LAND", "stick override has no confirmed manual pilot -> land"
                )
                self._control_wake.set()
                return False, self.reason
            if not self._ensure_piloting_source("SkyController"):
                self._latch_terminal(
                    "LAND", "stick override handoff to SkyController failed -> land"
                )
                self._control_wake.set()
                return False, self.reason
            self._control_wake.set()
            return True, (
                "zero PCMD + SkyController manual override"
                if zero_ok
                else "SkyController manual override; zero PCMD send failed"
            )

    def request_land(self, reason: str) -> None:
        """Wake the independent safety thread with an irreversible LAND request."""
        with self._io_lock:
            self._desired_pcmd = (0, 0, 0, 0)
            self._desired_valid = True
            self._latch_terminal("LAND", reason)
            self._control_wake.set()

    def arming_allowed(self, stream_healthy, stop_requested) -> tuple[bool, str]:
        """Atomically refresh safety input and authorize the imminent TakeOff call."""
        with self._io_lock:
            if self._safety is not None:
                try:
                    self.mode = self._safety.poll()
                except (OSError, ValueError, AttributeError, TypeError, RuntimeError) as exc:  # Tier1: safety poll — narrow, no silent pass
                    print(f"[safety] SafetyMonitor.poll() failed ({exc!r}); failing closed to HOVER", flush=True)
                    self.mode = "HOVER"
            try:
                stream_ok = bool(stream_healthy())
            except (OSError, ValueError, AttributeError, TypeError, RuntimeError) as exc:  # Tier1: stream health — narrow, no silent pass
                print(f"[safety] stream_healthy() failed ({exc!r}); assuming unhealthy", flush=True)
                stream_ok = False
            try:
                stopped = bool(stop_requested())
            except (OSError, ValueError, AttributeError, TypeError, RuntimeError) as exc:  # Tier1: stop request — narrow, no silent pass
                print(f"[safety] stop_requested() failed ({exc!r}); assuming stopped", flush=True)
                stopped = True
            if self.mode == "EMERGENCY":
                self._latch_terminal("EMERGENCY", "EMERGENCY command -> motor cut")
            elif self.mode == "LAND":
                self._latch_terminal("LAND", "safety LAND command -> land")
            elif stopped:
                self._latch_terminal("LAND", "operator termination signal -> land")
            return arming_allowed(
                self.mode, stream_ok, stopped, self.terminated.is_set())

    def schedule_authorized_takeoff(self, schedule, stream_healthy, stop_requested):
        """Refresh the final gate and schedule TakeOff under one authority lock.

        The returned expectation is deliberately not waited here: waiting while
        holding the lock would prevent LAND/EMERGENCY from being processed.
        """
        with self._io_lock:
            allowed, reason = self.arming_allowed(stream_healthy, stop_requested)
            if not allowed:
                return None, reason
            if not self._ensure_piloting_source("Controller"):
                return None, "piloting source Controller not confirmed"
            return schedule(), reason

    def _refresh_command_inputs(self, stream_healthy, stop_requested) -> tuple[bool, bool]:
        if self._safety is not None:
            try:
                self.mode = str(self._safety.poll()).upper()
            except (OSError, ValueError, AttributeError, TypeError, RuntimeError) as exc:  # Tier1: safety poll — narrow, no silent pass
                print(f"[safety] SafetyMonitor.poll() failed ({exc!r}); failing closed to HOVER", flush=True)
                self.mode = "HOVER"
        try:
            stream_ok = bool(stream_healthy())
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError) as exc:  # Tier1: stream health — narrow, no silent pass
            print(f"[safety] stream_healthy() failed ({exc!r}); assuming unhealthy", flush=True)
            stream_ok = False
        try:
            stopped = bool(stop_requested())
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError) as exc:  # Tier1: stop request — narrow, no silent pass
            print(f"[safety] stop_requested() failed ({exc!r}); assuming stopped", flush=True)
            stopped = True
        if self.mode == "EMERGENCY":
            self._latch_terminal("EMERGENCY", "EMERGENCY command -> motor cut")
        elif self.mode == "LAND":
            self._latch_terminal("LAND", "safety LAND command -> land")
        elif stopped:
            self._latch_terminal("LAND", "operator termination signal -> land")
        return stream_ok, stopped

    def _remember_desired(
        self,
        pcmd: tuple[int, int, int, int],
        stream_healthy,
        stop_requested,
        *,
        stamp: bool,
    ) -> None:
        self._desired_pcmd = pcmd
        self._desired_valid = True
        self._desired_stream_healthy = stream_healthy
        self._desired_stop_requested = stop_requested
        if stamp:
            self._desired_updated_mono_ns = time.monotonic_ns()

    def _command_source(self) -> str:
        return (
            "SkyController"
            if self.terminal_action == "NONE" and self.mode == "MANUAL"
            else "Controller"
        )

    def _ensure_command_source(self, stream_healthy, stop_requested):
        target = self._command_source()
        source_ok = self.terminal_action == "EMERGENCY" or self._ensure_piloting_source(target)
        if source_ok:
            return None
        self._remember_desired(
            (0, 0, 0, 0),
            stream_healthy,
            stop_requested,
            stamp=False,
        )
        actual = (0, 0, 0, 0) if self._send_counted(0, 0, 0, 0) else None
        return False, f"piloting source {target} not confirmed", actual

    def _validated_authorized_pcmd(self, pcmd) -> tuple[int, int, int, int] | None:
        try:
            desired = tuple(pcmd)
        except TypeError:
            return None
        if len(desired) != 4:
            return None
        if not all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in desired
        ):
            return None
        command = tuple(int(value) for value in desired)
        if (
            abs(command[0]) > self._max_translation_pcmd
            or abs(command[1]) > self._max_translation_pcmd
            or abs(command[2]) > self._max_yaw_pcmd
            or abs(command[3]) > self._max_translation_pcmd
        ):
            return None
        return command

    def _reject_invalid_pcmd(self, stream_healthy, stop_requested):
        zero = (0, 0, 0, 0)
        self._remember_desired(zero, stream_healthy, stop_requested, stamp=True)
        actual = zero if self._send_counted(*zero) else None
        self._control_wake.set()
        return False, "PCMD malformed or outside approved envelope", actual

    def _accept_authorized_pcmd(self, desired, stream_healthy, stop_requested):
        self._remember_desired(desired, stream_healthy, stop_requested, stamp=True)
        if self._thread.is_alive():
            # Wake the independent 20 Hz sender. Perception never waits for
            # command completion and cannot starve PCMD refreshes.
            self._control_wake.set()
        elif not self._send_counted(*desired):
            return False, f"PCMD send failed: {self.last_pcmd_send_error}", None
        return True, "atomic AUTO authorization; latest PCMD queued", desired

    def _block_unauthorized_pcmd(self, why, stream_healthy, stop_requested):
        actual = None
        if (
            self.mode not in {"MANUAL", "EMERGENCY"}
            and self.terminal_action != "EMERGENCY"
        ):
            zero = (0, 0, 0, 0)
            self._remember_desired(zero, stream_healthy, stop_requested, stamp=True)
            if self._send_counted(*zero):
                actual = zero
            self._control_wake.set()
        else:
            self._desired_valid = False
        return False, why, actual

    def send_authorized(self, pcmd, stream_healthy, stop_requested):
        """Authorize every autonomy PCMD, including zero, under one lock."""
        with self._io_lock:
            stream_ok, stopped = self._refresh_command_inputs(
                stream_healthy,
                stop_requested,
            )
            source_failure = self._ensure_command_source(stream_healthy, stop_requested)
            if source_failure is not None:
                return source_failure
            ok, why = arming_allowed(
                self.mode, stream_ok, stopped, self.terminated.is_set())
            if ok:
                desired = self._validated_authorized_pcmd(pcmd)
                if desired is None:
                    return self._reject_invalid_pcmd(stream_healthy, stop_requested)
                return self._accept_authorized_pcmd(
                    desired,
                    stream_healthy,
                    stop_requested,
                )
            return self._block_unauthorized_pcmd(why, stream_healthy, stop_requested)

    def send_auto(self, pcmd, stream_healthy, stop_requested):
        """Backward-compatible alias for the unified command authority."""
        return self.send_authorized(pcmd, stream_healthy, stop_requested)

    def pcmd_timing_snapshot(self) -> dict:
        with self._io_lock:
            return {
                "desired_pcmd_update_mono_ns": self._desired_updated_mono_ns,
                "pcmd_call_mono_ns": self.last_pcmd_call_mono_ns,
                "desired_pcmd": list(self._desired_pcmd) if self._desired_valid else None,
                "pcmd_send_failures": self.pcmd_send_failures,
            }

    def run_while_holding_zero(self, action, stream_healthy, stop_requested) -> bool:
        """Run a blocking inspection action while the monitor sustains zero PCMD.

        The initial zero is authorized atomically. The monitor then repeats zero
        while AUTO remains authoritative; MANUAL and EMERGENCY stay silent.
        """
        self._inspection_hold.set()
        try:
            authorized, _reason, _actual = self.send_authorized(
                (0, 0, 0, 0), stream_healthy, stop_requested)
            if not authorized:
                return False
            return bool(action())
        finally:
            self._inspection_hold.clear()

    def _attempt_callback(self, kind: str, callback, now: float) -> bool:
        """Attempt and confirm a terminal action; retry failures later."""
        retry_attr = "_emergency_retry_at" if kind == "EMERGENCY" else "_land_retry_at"
        with self._io_lock:
            if now < float(getattr(self, retry_attr)):
                return False
        if callback is None:
            exc = RuntimeError(f"{kind} callback is unavailable")
            with self._io_lock:
                self.action_failures[kind] += 1
                self.last_action_error[kind] = repr(exc)
                setattr(self, retry_attr, now + self._action_retry_s)
            return False

        def invoke_and_confirm():
            result = callback()
            _await_confirmed_action(
                result, f"{kind} callback", timeout_s=self._timeout)
            return result

        try:
            _bounded_call(invoke_and_confirm, self._timeout, f"{kind} callback")
        except BaseException as exc:  # noqa: BLE001 - safety action must retry
            with self._io_lock:
                self.action_failures[kind] += 1
                self.last_action_error[kind] = repr(exc)
                setattr(self, retry_attr, now + self._action_retry_s)
            print(f"[safety] {kind} callback failed ({exc!r}); retrying in "
                  f"{self._action_retry_s:.2f}s", flush=True)
            return False
        with self._io_lock:
            if kind == "EMERGENCY":
                self._emergency_acted = True
                self.emergency_issued = True
            else:
                self._land_acted = True
        return True

    def _ensure_piloting_source(self, source: str, now: float | None = None) -> bool:
        if self._piloting_source_cb is None or self._piloting_source == source:
            return True
        now = time.monotonic() if now is None else float(now)
        if now < self._piloting_source_retry_at:
            return False

        def invoke_and_confirm():
            result = self._piloting_source_cb(source)
            _await_confirmed_action(
                result, f"piloting source {source}", timeout_s=self._timeout)
            return result

        try:
            _bounded_call(
                invoke_and_confirm, self._timeout, f"piloting source {source}")
        except BaseException as exc:  # noqa: BLE001 - source failures fail closed
            with self._io_lock:
                self.piloting_source_failures += 1
                self.last_piloting_source_error = repr(exc)
                self._piloting_source_retry_at = now + self._action_retry_s
            print(f"[safety] piloting source {source} failed ({exc!r}); retrying in "
                  f"{self._action_retry_s:.2f}s", flush=True)
            return False
        with self._io_lock:
            self._piloting_source = source
            self.last_piloting_source_error = ""
        return True

    def _latch_terminal(self, action: str, reason: str) -> None:
        """Latch NONE -> LAND -> EMERGENCY; terminal input cannot be undone."""
        rank = {"NONE": 0, "LAND": 1, "EMERGENCY": 2}
        requested = str(action).upper()
        if requested not in rank:
            return
        if rank[requested] > rank[self.terminal_action]:
            self.terminal_action = requested
            self._terminal_reason = str(reason)
        if self.terminal_action != "NONE":
            self.reason = self._terminal_reason
            self.terminated.set()

    def _run(self) -> None:
        """Independent safety thread. It must not be able to stop quietly.

        This thread is the ONLY safety-switch poller and the independent 20 Hz
        PCMD sender. The inner try/excepts below cover the calls that were expected
        to fail; anything outside them used to kill the thread outright, after
        which LAND/EMERGENCY were never read again and the aircraft held its last
        command until the firmware's own link timeout. A crash is now converted
        into the same terminal LAND the operator would have commanded.
        """
        try:
            self._run_loop()
        except BaseException as exc:                      # noqa: BLE001 - see above
            self.thread_died_error = repr(exc)
            print(f"[safety] SAFETY MONITOR THREAD DIED: {exc!r} -> latching LAND",
                  flush=True)
            self._latch_terminal("LAND", f"safety monitor thread died: {exc!r}")
            # The monitor is the last independent safety authority. A latch by
            # itself is only bookkeeping once this thread is dead, so make one
            # bounded LAND attempt before propagating the crash.
            self._attempt_callback("LAND", self._land_cb, time.monotonic())
            raise

    def _external_stop_requested(self) -> bool:
        if self._stop_requested is None:
            return False
        try:
            return bool(self._stop_requested())
        except Exception:
            return True

    def _poll_safety_mode(self, warned: bool) -> tuple[str, bool]:
        if self._safety is None:
            return str(self.mode).upper(), warned
        try:
            return str(self._safety.poll()).upper(), False
        except Exception as exc:
            if not warned:
                print(
                    f"[safety] SafetyMonitor.poll() FAILING ({exc!r}); operator "
                    "LAND/HOVER/EMERGENCY may be unseen; failing closed to HOVER",
                    flush=True,
                )
            return "HOVER", True

    def _latch_polled_terminal(self, mode: str, external_stop: bool) -> None:
        if mode == "EMERGENCY":
            self._latch_terminal("EMERGENCY", "EMERGENCY command -> motor cut")
        elif external_stop:
            self._latch_terminal("LAND", "operator termination signal -> land")
        elif mode == "LAND":
            self._latch_terminal("LAND", "safety LAND command -> land")

    def _terminal_callback_request(self):
        if self.terminal_action == "EMERGENCY":
            now = time.monotonic()
            request = None
            if not self._emergency_acted:
                request = ("EMERGENCY", self._emergency_cb, now)
            return True, request
        if self.terminal_action != "LAND":
            return False, None
        now = time.monotonic()
        request = None
        if not self._land_acted:
            if now >= self._land_retry_at:
                self._send_counted(0, 0, 0, 0)
            request = ("LAND", self._land_cb, now)
        return True, request

    def _handle_non_auto_mode(self, mode: str, source_ok: bool) -> bool:
        if not source_ok:
            self._desired_pcmd = (0, 0, 0, 0)
            self._desired_valid = True
            self._send_counted(0, 0, 0, 0)
            return True
        if mode == "HOVER":
            self._desired_pcmd = (0, 0, 0, 0)
            self._desired_valid = True
            self._send_counted(0, 0, 0, 0)
            return True
        if mode == "MANUAL":
            self._desired_valid = False
            return True
        if self._inspection_hold.is_set():
            self._send_counted(0, 0, 0, 0)
            return True
        return False

    def _handle_control_stall(self, warned: bool) -> tuple[bool, bool]:
        stalled = self._beat_seen and (time.monotonic() - self._beat_t) > self._timeout
        if not stalled:
            self._stall_since = None
            return False, False
        self._desired_pcmd = (0, 0, 0, 0)
        self._desired_valid = True
        self._send_counted(0, 0, 0, 0)
        now = time.monotonic()
        if self._stall_since is None:
            self._stall_since = now
        elif now - self._stall_since >= WATCHDOG_LAND_S:
            print(
                f"[safety] WATCHDOG: control loop stalled > "
                f"{now - self._stall_since:.1f}s -> land",
                flush=True,
            )
            self._latch_terminal("LAND", "control loop stalled -> land")
        if not warned:
            print(
                f"[safety] WATCHDOG: control loop stalled > {self._timeout:.1f}s; "
                "forcing zero PCMD (hover)",
                flush=True,
            )
        return True, True

    def _desired_inputs_are_safe(self) -> bool:
        try:
            stream_ok = (
                self._desired_stream_healthy is None
                or bool(self._desired_stream_healthy())
            )
        except Exception:
            stream_ok = False
        try:
            stopped = (
                self._desired_stop_requested is not None
                and bool(self._desired_stop_requested())
            )
        except Exception:
            stopped = True
        return stream_ok and not stopped

    def _send_fresh_desired(self) -> None:
        if not self._desired_valid:
            return
        stamp = self._desired_updated_mono_ns
        fresh = (
            stamp is not None
            and time.monotonic_ns() - stamp <= int(self._command_ttl_s * 1e9)
        )
        desired = (
            self._desired_pcmd
            if self._desired_inputs_are_safe() and fresh
            else (0, 0, 0, 0)
        )
        if desired == (0, 0, 0, 0):
            self._desired_pcmd = desired
        self._send_counted(*desired)

    def _run_loop(self) -> None:
        warned = False
        poll_warned = False
        poll_period = min(self._control_period_s, self._timeout / 3.0)
        while True:
            self._control_wake.wait(poll_period)
            self._control_wake.clear()
            if self._stop.is_set():
                return
            callback_request = None
            terminal_handled = False
            with self._io_lock:
                mode, poll_warned = self._poll_safety_mode(poll_warned)
                self.mode = mode
                self._latch_polled_terminal(mode, self._external_stop_requested())
                source_ok = (
                    self.terminal_action == "EMERGENCY"
                    or self._ensure_piloting_source(self._command_source())
                )
                terminal_handled, callback_request = self._terminal_callback_request()
                if terminal_handled:
                    pass
                elif self._handle_non_auto_mode(mode, source_ok):
                    continue
                elif not self._handle_control_stall(warned)[0]:
                    warned = False
                    self._send_fresh_desired()
                else:
                    warned = True
            if terminal_handled:
                # LAND/EMERGENCY callbacks may wait on an SDK expectation. Run
                # them after releasing _io_lock so stop(), cleanup, and a safety
                # refresh cannot deadlock behind a broken command channel.
                if callback_request is not None:
                    self._attempt_callback(*callback_request)
                continue


def _handle_physical_stick_override(safety_monitor: SafetyMonitor, _axes) -> None:
    accepted, detail = safety_monitor.request_manual_override()
    if detail != "manual override already active":
        print(f"[safety] physical stick override: {detail}", flush=True)
    if not accepted:
        safety_monitor.request_land(
            "physical stick override could not confirm manual control -> land"
        )


def _handle_stick_monitor_disconnect(
        safety_monitor: SafetyMonitor, reason: str) -> None:
    print(f"[safety] stick monitor disconnected ({reason}) -> land", flush=True)
    safety_monitor.request_land("SkyController stick monitor disconnected -> land")


def _stick_monitor_has_all_axes(stick_monitor, required_axes: set[int]) -> bool:
    axes_deadline = time.monotonic() + 1.0
    while (stick_monitor.healthy
           and not required_axes.issubset(stick_monitor.snapshot_axes())
           and time.monotonic() < axes_deadline):
        time.sleep(0.01)
    return bool(
        stick_monitor.healthy
        and required_axes.issubset(stick_monitor.snapshot_axes())
    )


def start_skycontroller_stick_override(drone, safety_monitor: SafetyMonitor):
    """Arm the shared SC3 HID override before granting the PC piloting authority."""
    operator_root = MISSION_ROOT / "operator_interface"
    if str(operator_root) not in sys.path:
        sys.path.insert(0, str(operator_root))
    from olympe_live_backend import SkyControllerStickMonitor, _STICK_FLIGHT_AXES

    stick_monitor = SkyControllerStickMonitor(
        on_active=lambda axes: _handle_physical_stick_override(
            safety_monitor, axes),
        on_disconnect=lambda reason: _handle_stick_monitor_disconnect(
            safety_monitor, reason),
    )
    if not stick_monitor.start():
        raise RuntimeError(
            "SkyController stick monitor unavailable; refusing autonomous takeoff"
    )
    try:
        required_axes = set(_STICK_FLIGHT_AXES)
        if not _stick_monitor_has_all_axes(stick_monitor, required_axes):
            raise RuntimeError(
                "SkyController stick monitor did not publish all flight axes; "
                "refusing autonomous takeoff"
            )
        if stick_monitor.is_active():
            safety_monitor.request_manual_override()
            raise RuntimeError(
                "SkyController sticks are deflected; refusing autonomous takeoff"
            )
        set_piloting_source(drone, "Controller")
        if not stick_monitor.healthy or safety_monitor.mode == "MANUAL":
            set_piloting_source(drone, "SkyController")
            raise RuntimeError(
                "stick override changed during PC handoff; refusing autonomous takeoff"
            )
    except BaseException:
        stick_monitor.stop()
        raise
    print(
        f"[safety] physical stick override armed at 50 Hz: "
        f"{stick_monitor.device_name or stick_monitor.resolved_path}",
        flush=True,
    )
    return stick_monitor


def stop_skycontroller_stick_override(stick_monitor) -> None:
    if stick_monitor is None:
        return
    try:
        stick_monitor.stop()
    except Exception as exc:
        print(f"[fly] warning: could not stop stick monitor: {exc}", flush=True)


@dataclass(frozen=True)
class _LoopOutcome:
    reason: str | None = None
    sleep: bool = True


class _FlightLoopRunner:
    """Stateful implementation of one localization-to-PCMD control loop."""

    def __init__(
        self,
        hooks: LoopHooks,
        ctrl,
        waypoints,
        *,
        yaw_sign: int,
        verbose: bool,
        enforce_weak_pose_gate: bool,
    ) -> None:
        import real_path_follow_controller as rpf

        self.rpf = rpf
        self.hooks = hooks
        self.ctrl = ctrl
        self.waypoints = waypoints
        self.yaw_sign = yaw_sign
        self.verbose = verbose
        self.enforce_weak_pose_gate = enforce_weak_pose_gate
        self.heading = HeadingEstimator(ctrl.cfg.map_frame if ctrl is not None else None)
        self.pcmd_controller = (
            rpf.YawAlignedPcmdController(ctrl.cfg) if ctrl is not None else None
        )
        self.period = 1.0 / CTRL_HZ
        self.last_good = None
        self.last_good_accepted_at = None
        self.uncertainty = LocalizationState()
        self.stream_lost_since = None
        self.last_stream_warn = 0.0
        self.pending_jump = None
        self.pending_jump_stamp = None
        self.accepted_pose_history: list[np.ndarray] = []
        self.low_conf_jump_times: list[float] = []
        self.pose_jump_pause_latched = False
        self.pose_jump_pause_observed = False
        self.pose_jump_resume_authorized = False
        self.route_rejoin_target = None
        self.route_rejoin_segment: int | None = None
        self.steps = 0
        search_pcmd = hooks.localization_yaw_search_pcmd
        if (
            isinstance(search_pcmd, bool)
            or not isinstance(search_pcmd, int)
            or not 0 <= search_pcmd <= 20
        ):
            raise ValueError("localization_yaw_search_pcmd must be an integer within [0, 20]")
        self.localization_yaw_search_pcmd = search_pcmd
        self.localization_yaw_search_active = False
        self.localization_yaw_search_completed = False
        self.localization_yaw_search_started_at: float | None = None
        self.localization_yaw_search_last_yaw: float | None = None
        self.localization_yaw_search_accumulated_rad = 0.0

    def run(self) -> str:
        while True:
            started_at = self.hooks.now()
            if self.hooks.loop_beat is not None:
                self.hooks.loop_beat()
            safety_mode = (
                self.hooks.safety_poll()
                if self.hooks.safety_poll is not None
                else "AUTO"
            )
            record = {
                "step": self.steps,
                "t": round(started_at, 3),
                "t_mono_ns": time.monotonic_ns(),
                "safety": safety_mode,
            }
            outcome = self._tick(started_at, safety_mode, record)
            if outcome.reason is not None:
                return outcome.reason
            self.steps += 1
            if outcome.sleep:
                self._sleep_remaining(started_at)

    def _sleep_remaining(self, started_at: float) -> None:
        elapsed = self.hooks.now() - started_at
        if elapsed < self.period:
            time.sleep(self.period - elapsed)

    def _reset_pcmd_controller(self) -> None:
        if self.pcmd_controller is not None:
            self.pcmd_controller.reset()

    def _clear_route_rejoin(self) -> None:
        self.route_rejoin_target = None
        self.route_rejoin_segment = None

    def _emit(self, record: dict, pcmd, blocked: bool, why: str) -> None:
        """Write a per-tick record without disturbing the flight loop."""
        if self.hooks.log_tick is None:
            return
        record["pcmd"] = None if pcmd is None else [int(value) for value in pcmd]
        record["blocked"] = bool(blocked)
        record["reason"] = str(why)
        self._add_pose_log(record)
        self._add_pcmd_timing_log(record)
        try:
            self.hooks.log_tick(record)
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
            pass

    def _add_pose_log(self, record: dict) -> None:
        if self.hooks.pose_info is None:
            return
        try:
            info = self.hooks.pose_info() or {}
            record["loc"] = {
                key: info.get(key)
                for key in (
                    "frame",
                    "idx",
                    "mode",
                    "next_mode",
                    "inliers",
                    "reproj_rms",
                    "weak",
                )
            }
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
            pass

    def _add_pcmd_timing_log(self, record: dict) -> None:
        if self.hooks.pcmd_timing is None:
            return
        try:
            record.update(self.hooks.pcmd_timing() or {})
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
            pass

    def _send_command(self, pcmd):
        """Route every loop-originated PCMD through one final authority."""
        command = tuple(int(value) for value in pcmd)
        if self.hooks.send_authorized_pcmd is not None:
            return self.hooks.send_authorized_pcmd(command)
        mode = (
            self.hooks.safety_poll()
            if self.hooks.safety_poll is not None
            else "AUTO"
        )
        try:
            stream_ok = (
                self.hooks.stream_healthy is None
                or bool(self.hooks.stream_healthy())
            )
        except Exception:
            stream_ok = False
        if mode in {"MANUAL", "EMERGENCY"}:
            return False, f"safety mode is {mode}", None
        if mode == "AUTO" and stream_ok:
            self.hooks.send_pcmd(*command)
            return True, "fallback AUTO authorization", command
        self.hooks.send_pcmd(0, 0, 0, 0)
        return False, f"safety mode={mode} stream={stream_ok}", (0, 0, 0, 0)

    def _tick(self, started_at: float, safety_mode: str, record: dict) -> _LoopOutcome:
        outcome = self._initial_safety_outcome(safety_mode, record)
        if outcome is not None:
            return outcome
        outcome = self._pose_jump_pause_outcome(safety_mode, record)
        if outcome is not None:
            return outcome
        outcome = self._stick_override_outcome(record)
        if outcome is not None:
            return outcome
        now = self.hooks.now()
        outcome = self._stream_loss_outcome(now, record)
        if outcome is not None:
            return outcome
        olympe_yaw = self._read_olympe_yaw()
        self._record_olympe_yaw(olympe_yaw, record)
        localized_pose = self._read_localized_pose()
        now = self.hooks.now()
        outcome = self._post_inference_safety_outcome(record)
        if outcome is not None:
            return outcome
        if not self._stream_healthy_after_inference():
            return self._post_inference_stream_loss(now, record)
        localized_pose, fresh, new_visual_fix, low_confidence = self._accepted_pose(
            localized_pose, now, record)
        if fresh and new_visual_fix and not low_confidence:
            if self.heading.update(localized_pose.yaw, olympe_yaw):
                self.last_good = localized_pose
                self.last_good_accepted_at = now
                self._remember_accepted_pose(localized_pose)
                record["pose_source"] = "visual"
            else:
                mismatch = self.heading.last_increment_mismatch_rad
                if mismatch is not None:
                    record["visual_imu_yaw_mismatch_deg"] = round(
                        math.degrees(mismatch), 2)
                fresh = False
        transition = decide_localization_transition(
            self.uncertainty,
            now=now,
            fresh=fresh,
            low_confidence=low_confidence,
            weak_hover_land_s=WEAK_HOVER_LAND_S,
            lost_land_s=LOST_LAND_S,
            recovery_good_fixes_required=RECOVERY_GOOD_FIXES,
        )
        self.uncertainty = transition.state
        outcome = self._localization_transition_outcome(
            transition,
            fresh=fresh,
            low_confidence=low_confidence,
            now=now,
            olympe_yaw=olympe_yaw,
            record=record,
        )
        if outcome is not None:
            return outcome
        return self._localized_motion_outcome(
            localized_pose,
            olympe_yaw,
            now,
            record,
        )

    @staticmethod
    def _record_olympe_yaw(olympe_yaw, record: dict) -> None:
        try:
            logged_imu_yaw = float(olympe_yaw)
        except (TypeError, ValueError, OverflowError):
            logged_imu_yaw = float("nan")
        if math.isfinite(logged_imu_yaw):
            record["imu_yaw_ned_rad"] = round(logged_imu_yaw, 6)

    def _pose_jump_pause_outcome(
        self, mode: str, record: dict
    ) -> _LoopOutcome | None:
        if not self.pose_jump_pause_latched:
            return None
        if mode == "AUTO" and self.pose_jump_pause_observed:
            self.pose_jump_pause_latched = False
            self.pose_jump_resume_authorized = True
            record["pose_jump_resume_authorized"] = True
            return None
        self._reset_pcmd_controller()
        _sent, _why, actual = self._send_command((0, 0, 0, 0))
        self._emit(
            record,
            actual,
            True,
            "pose jump pause latched -> hover; operator must resume AUTO",
        )
        return _LoopOutcome()

    def _initial_safety_outcome(self, mode: str, record: dict) -> _LoopOutcome | None:
        if mode in {"EMERGENCY", "LAND", "HOVER", "MANUAL"}:
            self._interrupt_localization_yaw_search()
            self._clear_route_rejoin()
        if mode == "EMERGENCY":
            reason = "EMERGENCY command -> motor cut"
            self._emit(record, None, True, reason)
            return _LoopOutcome(reason)
        if mode == "LAND":
            _sent, _why, actual = self._send_command((0, 0, 0, 0))
            reason = "safety LAND command -> land"
            self._emit(record, actual, True, reason)
            return _LoopOutcome(reason)
        if mode == "HOVER":
            if self.pose_jump_pause_latched:
                self.pose_jump_pause_observed = True
            self._reset_pcmd_controller()
            _sent, _why, actual = self._send_command((0, 0, 0, 0))
            self._emit(record, actual, True, "safety HOVER: authorized zero PCMD")
            return _LoopOutcome()
        if mode == "MANUAL":
            self._reset_pcmd_controller()
            self._emit(
                record,
                None,
                True,
                "MANUAL: autonomy sends nothing (pilot has the sticks)",
            )
            return _LoopOutcome()
        return None

    def _stick_override_outcome(self, record: dict) -> _LoopOutcome | None:
        if self.hooks.stick_active is None:
            return None
        try:
            sticks_moved = bool(self.hooks.stick_active())
        except Exception as exc:
            sticks_moved = True
            print(
                f"[safety] stick monitor raised ({exc!r}); assuming override",
                flush=True,
            )
        if not sticks_moved:
            return None
        self._interrupt_localization_yaw_search()
        self._clear_route_rejoin()
        _sent, _why, actual = self._send_command((0, 0, 0, 0))
        self._reset_pcmd_controller()
        took_over = bool(self.hooks.request_manual()) if self.hooks.request_manual else False
        record["stick_override"] = True
        self._emit(
            record,
            actual,
            True,
            (
                "stick override -> hover + manual"
                if took_over
                else "stick override -> hover (no manual pilot; staying suspended)"
            ),
        )
        print(
            "[safety] STICK_OVERRIDE: pilot moved the sticks -> "
            "zero PCMD, autonomy suspended",
            flush=True,
        )
        return (
            _LoopOutcome()
            if took_over
            else _LoopOutcome("stick override without a manual pilot -> land")
        )

    def _stream_loss_outcome(self, now: float, record: dict) -> _LoopOutcome | None:
        if self.hooks.stream_healthy is None or self.hooks.stream_healthy():
            self.stream_lost_since = None
            return None
        self._reset_localization_yaw_search()
        self._clear_route_rejoin()
        self._reset_pcmd_controller()
        _sent, _why, actual = self._send_command((0, 0, 0, 0))
        self.stream_lost_since = self.stream_lost_since or now
        self.uncertainty = LocalizationState(
            recovery_good_fixes=self.uncertainty.recovery_good_fixes
        )
        record["stream_ok"] = False
        lost_for = now - self.stream_lost_since
        self._emit(record, actual, True, f"stream stale/lost {lost_for:.1f}s -> hover")
        self._warn_stream_loss(now, lost_for)
        return _LoopOutcome()

    def _warn_stream_loss(self, now: float, lost_for: float) -> None:
        if not self.verbose or now - self.last_stream_warn <= 1.0:
            return
        detail = (
            self.hooks.stream_status()
            if self.hooks.stream_status is not None
            else "stream stale/lost"
        )
        print(
            f"[safety] STREAM_LOST_HOVER: {detail}; lost_for={lost_for:.1f}s; "
            "zero PCMD, waiting for stream/manual/land",
            flush=True,
        )
        self.last_stream_warn = now

    def _read_olympe_yaw(self):
        try:
            return self.hooks.olympe_yaw()
        except Exception:
            return None

    def _read_localized_pose(self):
        try:
            return self.hooks.get_pose()
        except Exception as exc:
            if self.verbose:
                print(
                    f"[safety] localizer raised ({exc!r}); "
                    "treating as no fix -> hover",
                    flush=True,
                )
            return None

    def _post_inference_safety_outcome(self, record: dict) -> _LoopOutcome | None:
        mode = (
            self.hooks.safety_poll()
            if self.hooks.safety_poll is not None
            else "AUTO"
        )
        record["safety_final"] = mode
        if mode in {"EMERGENCY", "LAND", "HOVER", "MANUAL"}:
            self._interrupt_localization_yaw_search()
            self._clear_route_rejoin()
        if mode == "EMERGENCY":
            reason = "EMERGENCY command during inference -> motor cut"
            self._emit(record, None, True, reason)
            return _LoopOutcome(reason)
        if mode == "LAND":
            _sent, _why, actual = self._send_command((0, 0, 0, 0))
            reason = "safety LAND command during inference -> land"
            self._emit(record, actual, True, reason)
            return _LoopOutcome(reason)
        if mode == "HOVER":
            self._reset_pcmd_controller()
            _sent, _why, actual = self._send_command((0, 0, 0, 0))
            self._emit(
                record,
                actual,
                True,
                "safety changed to HOVER during inference -> zero PCMD",
            )
            return _LoopOutcome(sleep=False)
        if mode == "MANUAL":
            self._reset_pcmd_controller()
            self._emit(
                record,
                None,
                True,
                "safety changed to MANUAL during inference -> autonomy sends nothing",
            )
            return _LoopOutcome(sleep=False)
        return None

    def _stream_healthy_after_inference(self) -> bool:
        try:
            return (
                self.hooks.stream_healthy is None
                or bool(self.hooks.stream_healthy())
            )
        except Exception:
            return False

    def _post_inference_stream_loss(self, now: float, record: dict) -> _LoopOutcome:
        self._reset_localization_yaw_search()
        self._reset_pcmd_controller()
        _sent, _why, actual = self._send_command((0, 0, 0, 0))
        self.stream_lost_since = self.stream_lost_since or now
        self.uncertainty = LocalizationState(
            recovery_good_fixes=self.uncertainty.recovery_good_fixes
        )
        record["stream_ok"] = False
        self._emit(
            record,
            actual,
            True,
            "stream became stale/lost during inference -> hover",
        )
        return _LoopOutcome(sleep=False)

    @staticmethod
    def _pose_is_finite(pose) -> bool:
        return pose is not None and all(
            math.isfinite(float(value))
            for value in (pose.x, pose.y, pose.z, pose.yaw, pose.stamp)
        )

    def _record_pose(self, pose, now: float, finite: bool, record: dict) -> None:
        if pose is not None and finite:
            record["pose"] = [
                round(float(pose.x), 3),
                round(float(pose.y), 3),
                round(float(pose.z), 3),
            ]
            record["pose_age"] = round(now - float(pose.stamp), 3)
        elif pose is not None:
            record["pose"] = "non-finite"

    def _remember_accepted_pose(self, pose) -> None:
        self.accepted_pose_history.append(
            np.array([pose.x, pose.y, pose.z], dtype=float)
        )
        del self.accepted_pose_history[:-POSE_CONTINUITY_HISTORY]

    def _continuity_reference(self) -> np.ndarray:
        if self.accepted_pose_history:
            return np.median(np.stack(self.accepted_pose_history), axis=0)
        return np.array(
            [self.last_good.x, self.last_good.y, self.last_good.z],
            dtype=float,
        )

    def _latch_pose_jump_pause(self, distance: float, record: dict) -> None:
        if self.hooks.pose_jump_pause is None:
            return
        self.pose_jump_pause_latched = True
        self.pose_jump_pause_observed = False
        self.pose_jump_resume_authorized = False
        record["pose_jump_pause_latched"] = True
        try:
            self.hooks.pose_jump_pause(distance)
        except Exception as exc:
            record["pose_jump_pause_error"] = repr(exc)

    def _apply_jump_gate(
        self,
        pose,
        fresh: bool,
        low_confidence: bool,
        now: float,
        record: dict,
    ) -> tuple[bool, bool]:
        if not fresh or self.last_good is None:
            return fresh, False
        candidate = np.array([pose.x, pose.y, pose.z], float)
        previous = self._continuity_reference()
        distance = float(np.linalg.norm(candidate - previous))
        if distance <= MAX_POSE_JUMP_U:
            self.pending_jump = None
            self.pending_jump_stamp = None
            self.pose_jump_resume_authorized = False
            return True, False
        independent = (
            self.pending_jump is not None
            and self.pending_jump_stamp is not None
            and float(pose.stamp) > float(self.pending_jump_stamp)
        )
        confirmed = independent and float(
            np.linalg.norm(candidate - self.pending_jump)
        ) <= MAX_POSE_JUMP_U
        if confirmed and not low_confidence and (
            self.hooks.pose_jump_pause is None or self.pose_jump_resume_authorized
        ):
            self.pending_jump = None
            self.pending_jump_stamp = None
            self.accepted_pose_history.clear()
            self.pose_jump_resume_authorized = False
            self.pose_jump_pause_observed = False
            if self.verbose:
                print(
                    "[safety] POSE_JUMP confirmed by consecutive fix; "
                    "accepting relocation",
                    flush=True,
                )
            return True, False
        self.pending_jump = candidate
        self.pending_jump_stamp = float(pose.stamp)
        record["jump_reject_u"] = round(distance, 2)
        record["jump_low_confidence"] = bool(low_confidence)
        if low_confidence:
            cutoff = now - LOW_CONF_JUMP_WINDOW_S
            self.low_conf_jump_times = [
                stamp for stamp in self.low_conf_jump_times if stamp >= cutoff
            ]
            self.low_conf_jump_times.append(now)
            record["low_conf_jump_count"] = len(self.low_conf_jump_times)
            record["low_conf_jump_window_s"] = LOW_CONF_JUMP_WINDOW_S
            if len(self.low_conf_jump_times) >= LOW_CONF_JUMP_HOVER_COUNT:
                self._latch_pose_jump_pause(distance, record)
        else:
            self._latch_pose_jump_pause(distance, record)
        if self.verbose:
            print(
                f"[safety] POSE_JUMP_REJECT: fix jumped {distance:.2f}u > "
                f"{MAX_POSE_JUMP_U}u; hover, waiting for confirmation",
                flush=True,
            )
        return False, True

    def _accepted_pose(self, pose, now: float, record: dict):
        finite = self._pose_is_finite(pose)
        if pose is not None and not finite and self.verbose:
            print(
                "[safety] NON_FINITE_POSE_REJECT: localizer returned "
                "NaN/inf -> hover",
                flush=True,
            )
        self._record_pose(pose, now, finite, record)
        fresh = bool(finite and -0.05 <= now - pose.stamp <= POSE_STALE_S)
        if (
            fresh
            and self.last_good is not None
            and float(pose.stamp) <= float(self.last_good.stamp)
        ):
            # The desktop adapter exposes its latest accepted pose on every 20 Hz
            # control tick.  An unchanged capture is holdover, not another visual
            # observation: it must not refresh the holdover timer or re-anchor yaw.
            record["repeated_pose_stamp"] = round(float(pose.stamp), 6)
            fresh = False
        candidate_low_confidence = self._is_low_confidence(fresh, fresh)
        fresh, jump_rejected = self._apply_jump_gate(
            pose,
            fresh,
            candidate_low_confidence,
            now,
            record,
        )
        new_visual_fix = fresh
        low_confidence = bool(fresh and candidate_low_confidence)
        if not fresh and (
            not jump_rejected
            and self.last_good is not None
            and self.last_good_accepted_at is not None
            and self.uncertainty.uncertain_since is None
            and 0.0 <= now - self.last_good_accepted_at <= POSE_HOLDOVER_S
            and now - self.last_good.stamp <= POSE_STALE_S
        ):
            pose, fresh = self.last_good, True
            record["pose"] = [
                round(float(pose.x), 3),
                round(float(pose.y), 3),
                round(float(pose.z), 3),
            ]
            record["pose_age"] = round(now - float(pose.stamp), 3)
            record["pose_holdover_s"] = round(
                now - self.last_good_accepted_at, 3)
            record["pose_source"] = "holdover"
        return pose, fresh, new_visual_fix, low_confidence

    def _is_low_confidence(self, fresh: bool, new_visual_fix: bool) -> bool:
        weak = (
            self.enforce_weak_pose_gate
            and self.hooks.pose_is_weak is not None
            and self.hooks.pose_is_weak()
        )
        few_inliers = (
            self.hooks.pose_confidence is not None
            and self.hooks.pose_confidence() < LOW_CONF_INLIERS
        )
        return bool(fresh and new_visual_fix and (weak or few_inliers))

    @staticmethod
    def _uncertainty_reason(low_confidence: bool, record: dict) -> str:
        if low_confidence:
            return "low confidence"
        if "jump_reject_u" in record:
            return "pose jump rejected"
        if "visual_imu_yaw_mismatch_deg" in record:
            return "visual/IMU yaw increment mismatch"
        return "no fresh pose (lost/stale/PnP fail/non-finite)"

    def _notify_localization_yaw_search(self, state: str, progress_deg: float) -> None:
        callback = self.hooks.localization_yaw_search_event
        if callback is None:
            return
        try:
            callback(str(state), float(progress_deg))
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
            pass

    def _reset_localization_yaw_search(self, event: str | None = None) -> None:
        was_active = self.localization_yaw_search_active
        progress_deg = math.degrees(self.localization_yaw_search_accumulated_rad)
        self.localization_yaw_search_active = False
        self.localization_yaw_search_completed = False
        self.localization_yaw_search_started_at = None
        self.localization_yaw_search_last_yaw = None
        self.localization_yaw_search_accumulated_rad = 0.0
        if event is not None and was_active:
            self._notify_localization_yaw_search(event, progress_deg)

    def _interrupt_localization_yaw_search(self) -> None:
        if not self.localization_yaw_search_active:
            return
        self.localization_yaw_search_active = False
        self.localization_yaw_search_completed = True
        self.localization_yaw_search_started_at = None
        self.localization_yaw_search_last_yaw = None

    @staticmethod
    def _finite_yaw(value) -> float | None:
        try:
            yaw = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return yaw if math.isfinite(yaw) else None

    def _localization_yaw_search_outcome(
        self,
        *,
        now: float,
        waited: float,
        olympe_yaw,
        fresh: bool,
        record: dict,
    ) -> _LoopOutcome | None:
        if self.localization_yaw_search_pcmd <= 0:
            return None
        if self.localization_yaw_search_completed:
            return None
        if fresh and not self.localization_yaw_search_active:
            return None

        yaw = self._finite_yaw(olympe_yaw)
        if not self.localization_yaw_search_active:
            if waited < LOST_YAW_SEARCH_DELAY_S:
                return None
            if yaw is None:
                record["localization_yaw_search"] = "yaw_telemetry_unavailable"
                return None
            self.localization_yaw_search_active = True
            self.localization_yaw_search_started_at = now
            self.localization_yaw_search_last_yaw = yaw
            self.localization_yaw_search_accumulated_rad = 0.0
            self._notify_localization_yaw_search("started", 0.0)
        elif yaw is None:
            self.localization_yaw_search_active = False
            self.localization_yaw_search_completed = True
            progress_deg = math.degrees(
                self.localization_yaw_search_accumulated_rad
            )
            record["localization_yaw_search"] = "yaw_telemetry_lost"
            self._notify_localization_yaw_search("yaw_telemetry_lost", progress_deg)
            return None
        else:
            previous = self.localization_yaw_search_last_yaw
            if previous is not None:
                delta = (yaw - previous + math.pi) % (2.0 * math.pi) - math.pi
                self.localization_yaw_search_accumulated_rad = max(
                    0.0,
                    self.localization_yaw_search_accumulated_rad + delta,
                )
            self.localization_yaw_search_last_yaw = yaw

        progress_deg = math.degrees(self.localization_yaw_search_accumulated_rad)
        if self.localization_yaw_search_accumulated_rad >= LOST_YAW_SEARCH_TARGET_RAD:
            self.localization_yaw_search_active = False
            self.localization_yaw_search_completed = True
            record["localization_yaw_search"] = "completed"
            record["localization_yaw_search_deg"] = round(progress_deg, 1)
            self._notify_localization_yaw_search("completed", progress_deg)
            return None

        assert self.localization_yaw_search_started_at is not None
        if now - self.localization_yaw_search_started_at >= LOST_YAW_SEARCH_TIMEOUT_S:
            self.localization_yaw_search_active = False
            self.localization_yaw_search_completed = True
            record["localization_yaw_search"] = "timed_out"
            record["localization_yaw_search_deg"] = round(progress_deg, 1)
            self._notify_localization_yaw_search("timed_out", progress_deg)
            return None

        sent, send_reason, actual = self._send_command(
            (0, 0, self.localization_yaw_search_pcmd, 0)
        )
        record["localization_yaw_search"] = "searching_right"
        record["localization_yaw_search_deg"] = round(progress_deg, 1)
        if not sent:
            self.localization_yaw_search_active = False
            self.localization_yaw_search_completed = True
            record["localization_yaw_search_error"] = send_reason
            self._notify_localization_yaw_search("command_rejected", progress_deg)
            return None
        self._emit(
            record,
            actual,
            False,
            f"right yaw localization search {progress_deg:.1f}/360.0deg",
        )
        return _LoopOutcome()

    def _localization_transition_outcome(
        self,
        transition,
        *,
        fresh: bool,
        low_confidence: bool,
        now: float,
        olympe_yaw,
        record: dict,
    ) -> _LoopOutcome | None:
        if transition.action in {"uncertain_hover", "land"}:
            self._clear_route_rejoin()
            self._reset_pcmd_controller()
            waited = float(transition.waited_s)
            if self.hooks.force_relocalize is not None:
                try:
                    self.hooks.force_relocalize()
                except Exception as exc:
                    record["relocalize_error"] = repr(exc)
            reason = self._uncertainty_reason(low_confidence, record)
            search_outcome = self._localization_yaw_search_outcome(
                now=now,
                waited=waited,
                olympe_yaw=olympe_yaw,
                fresh=fresh,
                record=record,
            )
            if search_outcome is not None:
                return search_outcome
            _sent, _why, actual = self._send_command((0, 0, 0, 0))
            self._emit(record, actual, True, f"{reason} -> hover ({waited:.1f}s)")
            # When the desktop adapter opted into the bounded yaw search, keep
            # the aircraft stationary for the full MegaLoc retry window.  The
            # ordinary 4 s landing threshold must not end this recovery path
            # before its configured 10 s delay expires.
            if (
                self.localization_yaw_search_pcmd > 0
                and not fresh
                and not self.localization_yaw_search_completed
                and waited < LOST_YAW_SEARCH_DELAY_S
            ):
                return _LoopOutcome()
            if (
                transition.action == "land"
                and self.hooks.land_on_localization_loss
            ):
                terminal = "localization lost" if not fresh else "low confidence"
                return _LoopOutcome(f"{terminal} -> land")
            if self.verbose and self.steps % 10 == 0:
                print(
                    "[safety] LOW_CONF_HOVER: hover + MegaLoc relocalize; "
                    f"waited={waited:.1f}s",
                    flush=True,
                )
            return _LoopOutcome()
        if transition.action != "recovery_hover":
            return None
        self._reset_localization_yaw_search("recovered")
        self._reset_pcmd_controller()
        _sent, _why, actual = self._send_command((0, 0, 0, 0))
        self._emit(
            record,
            actual,
            True,
            f"localization recovery confirmation "
            f"{self.uncertainty.recovery_good_fixes}/{RECOVERY_GOOD_FIXES} -> hover",
        )
        return _LoopOutcome()

    def _compute_route_pcmd(self, command, pose, now: float):
        if command.look_at_pole is not None:
            self._reset_pcmd_controller()
            return self.rpf.command_to_body_percent(
                command,
                pose,
                config=self.ctrl.cfg,
                yaw_sign=self.yaw_sign,
            )
        return self.pcmd_controller.update(
            command,
            pose,
            now,
            target_key=self.ctrl.target_index,
            yaw_sign=self.yaw_sign,
        )

    def _record_route_command(self, record: dict, command, heading: float) -> None:
        record["heading_deg"] = round(math.degrees(heading), 1)
        record["path_error_u"] = round(float(command.path_error), 3)
        record["progress"] = round(float(command.progress), 4)
        record["action"] = command.status
        record["pcmd_phase"] = self.pcmd_controller.phase

    def _route_command_gate_outcome(self, command, pose, record: dict) -> _LoopOutcome | None:
        route_deviation_limit = float(self.ctrl.cfg.max_route_deviation)
        route_distance, nearest, segment, _progress_s = self.rpf.project_to_path(
            pose.xyz, self.ctrl.wp, self.ctrl.cum
        )
        record["route_distance_u"] = round(float(route_distance), 3)
        if route_distance <= route_deviation_limit:
            return None
        self.route_rejoin_target = np.array(nearest, dtype=float, copy=True)
        self.route_rejoin_segment = int(segment)
        record["route_rejoin_target"] = [
            round(float(value), 6) for value in self.route_rejoin_target
        ]
        record["route_rejoin_segment"] = self.route_rejoin_segment
        _sent, _why, actual = self._send_command((0, 0, 0, 0))
        reason = (
            f"route deviation {route_distance:.2f}u > "
            f"{route_deviation_limit}u -> hover before rejoin"
        )
        self._reset_pcmd_controller()
        self._emit(record, actual, True, reason)
        return _LoopOutcome()

    def _ground_speed_sample(self, command):
        if command.action != "LAND":
            return None
        try:
            return self.hooks.ground_speed() if self.hooks.ground_speed is not None else None
        except Exception:
            return None

    def _route_rejoin_outcome(self, pose, record: dict) -> _LoopOutcome:
        target = self.route_rejoin_target
        assert target is not None
        delta = target - pose.xyz
        distance = float(np.linalg.norm(delta))
        record["route_rejoin_target"] = [round(float(value), 6) for value in target]
        record["route_rejoin_segment"] = self.route_rejoin_segment
        record["route_rejoin_distance_u"] = round(distance, 6)
        record["action"] = "REJOIN"
        record["pcmd_phase"] = "route_rejoin_translate"
        # A percentage PCMD has a finite minimum step. Stop inside twice the
        # ordinary translation tolerance so the aircraft cannot oscillate across
        # an exact projection forever; this remains well inside the route tube.
        rejoin_tolerance = 2.0 * float(self.ctrl.cfg.translation_arrival_tolerance)
        route_limit = float(self.ctrl.cfg.max_route_deviation)
        if route_limit > 0.0:
            rejoin_tolerance = min(rejoin_tolerance, route_limit)
        if distance <= rejoin_tolerance:
            self._clear_route_rejoin()
            self._reset_pcmd_controller()
            _sent, _why, actual = self._send_command((0, 0, 0, 0))
            self._emit(record, actual, True, "route rejoin complete -> hover")
            return _LoopOutcome()

        command = self.rpf.Command(
            "REJOIN",
            delta / distance,
            float(pose.yaw),
            target.copy(),
            distance,
            float(self.ctrl.target_index) / float(max(1, len(self.ctrl.wp) - 1)),
            status="route rejoin translate without yaw",
        )
        pcmd = self.rpf.command_to_body_percent(
            command,
            pose,
            config=self.ctrl.cfg,
            yaw_sign=self.yaw_sign,
            require_yaw_alignment=False,
        )
        # REJOIN preserves the current camera heading by contract.
        pcmd = (pcmd[0], pcmd[1], 0, pcmd[3])
        sent, send_reason, actual = self._send_command(pcmd)
        self._emit(
            record,
            actual,
            not sent,
            command.status if sent else send_reason,
        )
        if not sent:
            terminal = self._blocked_send_outcome(send_reason, context="during route rejoin")
            if terminal is not None:
                return terminal
        return _LoopOutcome()

    def _route_completion_outcome(
        self,
        command,
        now: float,
        record: dict,
    ) -> _LoopOutcome | None:
        if not command.should_land and self.ctrl.state not in ("LANDING", "DONE"):
            return None
        _sent, _why, actual = self._send_command((0, 0, 0, 0))
        transition = decide_route_completion_landing(
            command.action,
            self._ground_speed_sample(command),
            now,
            threshold_mps=LANDING_SPEED_THRESHOLD_MPS,
            max_age_s=GROUND_SPEED_MAX_AGE_S,
        )
        record.update(dict(transition.log_fields))
        if transition.outcome == "hover":
            self._reset_pcmd_controller()
            self._emit(record, actual, True, transition.reason)
            return _LoopOutcome()
        self._emit(record, actual, True, transition.reason)
        return _LoopOutcome(transition.reason)

    @staticmethod
    def _blocked_send_outcome(send_reason: str, *, context: str) -> _LoopOutcome | None:
        upper = str(send_reason).upper()
        if "EMERGENCY" in upper:
            return _LoopOutcome(f"EMERGENCY command {context} -> motor cut")
        if "LAND" in upper or "STOP SIGNAL" in upper or "TERMINATED" in upper:
            return _LoopOutcome(f"{send_reason} -> land")
        return None

    def _inspection_capture(self, command, pose, record: dict) -> None:
        capture_acked = False
        try:
            if self.hooks.inspection_ack is not None:
                action = lambda: self.hooks.inspection_ack(command.look_at_pole, pose)
                capture_acked = bool(
                    self.hooks.inspection_hold(action)
                    if self.hooks.inspection_hold is not None
                    else action()
                )
        except Exception as exc:
            if self.verbose:
                print(
                    f"[inspection] gimbal/capture hook failed ({exc!r})",
                    flush=True,
                )
        post_ok, post_reason, post_actual = self._send_command((0, 0, 0, 0))
        acknowledged = bool(
            capture_acked
            and post_ok
            and self.ctrl.ack_inspection(
                command.look_at_pole,
                orientation_confirmed=True,
            )
        )
        self._emit(
            record,
            post_actual,
            True,
            (
                "inspection target aligned + capture acknowledged"
                if acknowledged
                else f"inspection capture not acknowledged ({post_reason})"
            ),
        )

    def _inspection_outcome(
        self,
        command,
        pose,
        yaw: int,
        record: dict,
    ) -> _LoopOutcome:
        hold_ok, hold_reason, hold_actual = self._send_command((0, 0, 0, 0))
        if not hold_ok:
            self._emit(
                record,
                hold_actual,
                True,
                f"inspection pre-action hold blocked ({hold_reason})",
            )
            terminal = self._blocked_send_outcome(
                hold_reason,
                context="during inspection",
            )
            return terminal or _LoopOutcome()
        yaw_error = _wrap(float(command.yaw_target) - float(pose.yaw))
        tolerance = math.radians(float(self.ctrl.cfg.inspect_yaw_tolerance_deg))
        if abs(yaw_error) > tolerance:
            sent, send_reason, actual = self._send_command((0, 0, yaw, 0))
            detail = (
                f"inspection body-yaw align error={math.degrees(yaw_error):.1f}deg"
                if sent
                else send_reason
            )
            self._emit(record, actual, not sent, detail)
        else:
            self._inspection_capture(command, pose, record)
        return _LoopOutcome()

    def _route_pcmd_outcome(
        self,
        command,
        localized_pose,
        heading: float,
        pcmd,
        record: dict,
    ) -> _LoopOutcome:
        sent, send_reason, actual = self._send_command(pcmd)
        self._emit(record, actual, not sent, command.status if sent else send_reason)
        if not sent:
            terminal = self._blocked_send_outcome(send_reason, context="before PCMD")
            if terminal is not None:
                return terminal
        elif self.verbose and self.steps % 10 == 0:
            roll, pitch, yaw, gaz = pcmd
            print(
                f"[{command.status:26s}] pos=({localized_pose.x:5.1f},"
                f"{localized_pose.y:5.1f},{localized_pose.z:4.1f}) "
                f"hdg={math.degrees(heading):6.1f} err={command.path_error:4.2f} "
                f"prog={command.progress:4.2f} phase={self.pcmd_controller.phase} "
                f"PCMD(r={roll:+d},p={pitch:+d},y={yaw:+d},g={gaz:+d})"
            )
        return _LoopOutcome()

    def _localized_motion_outcome(
        self,
        localized_pose,
        olympe_yaw,
        now: float,
        record: dict,
    ) -> _LoopOutcome:
        self._reset_localization_yaw_search("recovered")
        heading = self.heading.heading(olympe_yaw)
        if heading is None:
            self._reset_pcmd_controller()
            _sent, _why, actual = self._send_command((0, 0, 0, 0))
            self._emit(record, actual, True, "heading unavailable -> hover")
            return _LoopOutcome()
        pose = self.rpf.Pose(
            x=localized_pose.x,
            y=localized_pose.y,
            z=localized_pose.z,
            yaw=heading,
            stamp=localized_pose.stamp,
        )
        if self.route_rejoin_target is not None:
            return self._route_rejoin_outcome(pose, record)
        command = self.ctrl.step(pose, now)
        self._record_route_command(record, command, heading)
        outcome = self._route_command_gate_outcome(command, pose, record)
        if outcome is not None:
            return outcome
        pcmd = self._compute_route_pcmd(command, pose, now)
        outcome = self._route_completion_outcome(command, now, record)
        if outcome is not None:
            return outcome
        if command.look_at_pole is not None:
            return self._inspection_outcome(command, pose, pcmd[2], record)
        return self._route_pcmd_outcome(
            command,
            localized_pose,
            heading,
            pcmd,
            record,
        )


def run_loop(hooks: LoopHooks, ctrl, waypoints, yaw_sign: int = 1, verbose: bool = True,
             *, enforce_weak_pose_gate: bool = True):
    """Localize, authorize, and route PCMD until a terminal condition."""
    return _FlightLoopRunner(
        hooks,
        ctrl,
        waypoints,
        yaw_sign=yaw_sign,
        verbose=verbose,
        enforce_weak_pose_gate=enforce_weak_pose_gate,
    ).run()


# ---------------------------------------------------------------------------
# Builders

def production_config():
    """The LOCKED production sweep (user's final decision). These
    match ProductionConfig() defaults; pinned here so the deployment entrypoint is
    self-documenting and a future default change can't silently alter real flights.
        BOOT_INIT/LOST : MegaLoc top30 -> XFeat -> LighterGlue -> PnP
        TRACK/WEAK     : XFeat -> LighterGlue adaptive 3->5 -> PnP

    2026-07-14: the mutual-NN fast pass was REMOVED from production on accuracy
    grounds (operator decision). LighterGlue now runs on every frame, which is the
    path the 20260714 reference trajectories were measured against. The temporal
    anchor cache is only consulted by the NN fast pass, so it is now inert; its
    settings are kept so matcher_mode="nn_then_lg" stays reproducible for benchmarks.
    """
    from production_localizer_factory import production_xfeat_config

    return production_xfeat_config()


def build_localizer(frame_source, cam_tuple=None):
    from production_localizer_factory import build_production_localizer
    import real_path_follow_controller as rpf

    contract = flight_contract_from_environment(require_approved=False) or {}
    contract_camera = contract.get("query_camera")
    if cam_tuple is None and isinstance(contract_camera, dict):
        cam_tuple = (
            contract_camera.get("model"),
            contract_camera.get("width"),
            contract_camera.get("height"),
            contract_camera.get("params"),
        )
    bundle_sha256 = contract.get("localization_bundle_sha256") or BUNDLE_SHA256
    reference_index = contract.get("reference_index") or REFERENCE_INDEX or None
    reference_index_sha256 = (
        contract.get("reference_index_sha256")
        or REFERENCE_INDEX_SHA256
        or None
    )
    if (
        contract.get("approved") is True
        and BUNDLE_SHA256
        and BUNDLE_SHA256.strip().lower() != str(bundle_sha256).strip().lower()
    ):
        raise ValueError(
            "SFM_BUNDLE_SHA256 does not match the approved flight contract"
        )
    built = build_production_localizer(
        backend=LOCALIZER_BACKEND,
        bundle=XBUN,
        frame_source=frame_source,
        camera_tuple=cam_tuple or query_camera_from_environment(),
        bundle_sha256=bundle_sha256 or None,
        megaloc_cache=MEG or None,
        reference_index=reference_index,
        reference_index_sha256=reference_index_sha256,
        production_profile=LOCALIZER_PROFILE or None,
        production_profile_sha256=contract.get("localizer_profile_sha256"),
        map_frame=(rpf.load_map_frame(MAP_ALIGN) if MAP_ALIGN else None),
    )
    print(
        f"[localizer] backend={built.backend} variant={built.variant} "
        f"camera={built.camera.model}:{built.camera.width}x{built.camera.height}",
        flush=True,
    )
    return built.tracker


def build_controller(*, require_approved: bool = True):
    """Build the global scale-free direction controller for an approved route."""
    import real_path_follow_controller as rpf

    contract = flight_contract_from_environment(require_approved=require_approved)
    if contract is None:
        if require_approved:
            raise ValueError("route control requires an explicit site flight contract")
        map_frame = rpf.load_map_frame(MAP_ALIGN) if MAP_ALIGN else rpf.LEGACY_MAP_FRAME
        waypoints = rpf.load_waypoints(PATH_JSON, map_frame=map_frame)
        route_source = PATH_JSON
    else:
        for key in ("site_id", "coordinate_frame_id", "route_sha256"):
            if not str(contract.get(key) or "").strip():
                raise ValueError(f"route control contract is missing {key}")
        if not MAP_ALIGN:
            raise ValueError("autonomous flight requires a measured map alignment")
        map_frame = rpf.load_map_frame(MAP_ALIGN)
        snapshot = rpf.capture_mission_route_snapshot(
            PATH_JSON,
            expected_sha256=str(contract["route_sha256"]),
            expected_site_id=str(contract["site_id"]),
            expected_coordinate_frame_id=str(contract["coordinate_frame_id"]),
            map_frame=map_frame,
        )
        waypoints = snapshot.controller_waypoints()
        route_source = snapshot
    config = rpf.config_for_route(
        route_source,
        rpf.ControlConfig(
            inspect_waypoints=(),
            map_frame=map_frame,
            max_route_deviation=MAX_ROUTE_DEVIATION_U,
        ),
    )
    return rpf.RouteAutoController(waypoints, poles=[], config=config), waypoints


# ---------------------------------------------------------------------------
# Real flight

def _wait_for_fresh_auto_consent(safety, stop: dict[str, bool], command_log) -> None:
    print(
        f"[fly] waiting <= {AUTO_CONSENT_TIMEOUT_S:.0f}s for a fresh AUTO command "
        "written after this run started",
        flush=True,
    )
    deadline = time.monotonic() + AUTO_CONSENT_TIMEOUT_S
    while True:
        mode = safety.poll()
        if mode == "AUTO":
            return
        timed_out = time.monotonic() >= deadline
        if mode in {"LAND", "EMERGENCY"} or stop["f"] or timed_out:
            if stop["f"]:
                reason = "operator stop"
            elif timed_out:
                reason = "fresh AUTO consent timeout"
            else:
                reason = f"preflight safety mode {mode}"
            command_log.event(event="terminal", reason=reason)
            command_log.close()
            raise SystemExit(f"[fly] {reason}; refusing to connect/take off")
        time.sleep(0.1)


def _install_flight_stop_handlers() -> dict[str, bool]:
    stop = {"f": False}
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        termination_signal = getattr(signal, signal_name, None)
        if termination_signal is not None:
            signal.signal(
                termination_signal,
                lambda *_args: stop.__setitem__("f", True),
            )
    return stop


def _configure_flight_preflight_for_live(
    drone,
    command_log,
    max_altitude_m: float | None,
    max_distance_m: float | None,
    distance_geofence: bool,
) -> None:
    try:
        preflight = configure_flight_preflight(
            drone,
            max_altitude_m,
            max_distance_m,
            distance_geofence,
            check_landed=False,
        )
    except Exception as exc:
        command_log.event(
            event="firmware_preflight",
            confirmed=False,
            warning=str(exc),
            max_altitude_m=max_altitude_m,
            max_distance_m=max_distance_m,
            distance_geofence=bool(distance_geofence),
        )
        print(
            f"[fly] WARNING: {exc}; continuing without confirmed firmware limits",
            flush=True,
        )
        return
    command_log.event(event="firmware_preflight", confirmed=True, **preflight)
    print(
        "[fly] firmware preflight confirmed: "
        f"battery={preflight['battery_percent']}% "
        f"max_altitude={preflight['max_altitude_m']:.1f}m "
        f"max_distance={preflight['max_distance_m']:.1f}m "
        f"distance_geofence={preflight['distance_geofence']}",
        flush=True,
    )


def _boot_pose(localizer):
    try:
        return localizer.get_pose()
    except Exception as exc:
        print(f"[fly] BOOT_INIT localizer raised ({exc!r}); retrying", flush=True)
        return None


def _wait_for_boot_lock(
    monitor, grabber, localizer, route_waypoints, controller, stop
) -> str | None:
    lock = BootPoseLock(route_waypoints)
    locked = False
    started_at = time.monotonic()
    while not locked and not stop["f"] and not monitor.terminated.is_set():
        allowed, _reason = monitor.arming_allowed(grabber.is_healthy, lambda: stop["f"])
        if allowed:
            locked = lock.observe(_boot_pose(localizer), now=time.monotonic())
        else:
            lock.reset()
        monitor.send_authorized(
            (0, 0, 0, 0),
            grabber.is_healthy,
            lambda: stop["f"],
        )
        if time.monotonic() - started_at > FIRST_FIX_TIMEOUT_S:
            raise SystemExit(
                "[fly] BOOT lock failed (need consecutive fresh fixes near a route waypoint) "
                "-> landing (no AUTO)"
            )
        time.sleep(0.1)
    if monitor.terminated.is_set():
        return monitor.reason or "safety command before AUTO -> land"
    if stop["f"]:
        return "operator stop signal before AUTO -> land"
    position = lock.position
    if position is None:
        return "BOOT lock completed without a takeoff position -> land"
    controller.start_after_nearest_waypoint(position)
    return None


def _capture_inspection_frame(drone, grabber, metadata, pose, map_frame) -> bool:
    pitch_target = inspection_gimbal_pitch_deg(metadata, pose, map_frame)
    if pitch_target is None:
        return False
    if not set_gimbal(
        drone,
        pitch_target,
        require_confirmation=True,
        timeout_s=2.0,
        tolerance_deg=3.0,
    ):
        return False
    baseline_source_us = grabber.latest_source_ntp_us()
    if baseline_source_us is None:
        return False
    receipt_after = time.monotonic() + INSPECTION_PIPELINE_DRAIN_S
    sample = None
    frame_deadline = receipt_after + STREAM_STALE_S
    while time.monotonic() < frame_deadline:
        candidate = grabber.inspection_sample_after(
            baseline_source_us,
            min_source_advance_s=INSPECTION_PIPELINE_DRAIN_S,
            receipt_after=receipt_after,
        )
        if candidate is not None:
            sample = candidate
            break
        time.sleep(0.02)
    if sample is None:
        return False
    frame, capture_stamp = sample
    if time.monotonic() - float(capture_stamp) > STREAM_STALE_S:
        return False
    import cv2

    output_dir = SYSTEM_ROOT / "outputs" / "flight_inspections"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"wp{int(metadata['waypoint']):02d}_pole{int(metadata['pole_id']):02d}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}.jpg"
    )
    written = bool(
        cv2.imwrite(str(output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    )
    if written:
        print(
            f"[inspection] body aligned; gimbal pitch={pitch_target:.1f}deg; "
            f"captured {output}",
            flush=True,
        )
    return written


def _live_stream_status(grabber) -> str:
    age = grabber.last_frame_age()
    return (
        "no 720p frame yet"
        if age is None
        else f"last 720p frame age={age:.2f}s source={grabber.stamp_source}"
    )


def _make_live_loop_hooks(
    *,
    localizer,
    safety,
    stick_monitor,
    drone,
    send_pcmd,
    monitor,
    grabber,
    stop,
    ground_speed,
    command_log,
    inspection_ack,
) -> LoopHooks:
    request_manual = (
        (lambda: safety.force("manual"))
        if safety.allow_manual
        else (lambda: False)
    )

    def send_if_active(roll, pitch, yaw, gaz):
        if not stop["f"] and not monitor.terminated.is_set():
            send_pcmd(roll, pitch, yaw, gaz)

    return LoopHooks(
        get_pose=localizer.get_pose,
        pose_is_weak=lambda: bool(dict(localizer.last_info).get("weak", False)),
        pose_confidence=lambda: int(
            dict(localizer.last_info).get("inliers", 0) or 0
        ),
        force_relocalize=lambda: setattr(localizer.state, "mode", "LOST"),
        request_manual=request_manual,
        stick_active=getattr(stick_monitor, "is_active", None),
        olympe_yaw=lambda: olympe_yaw_of(drone),
        send_pcmd=send_if_active,
        send_authorized_pcmd=lambda pcmd: monitor.send_authorized(
            pcmd,
            grabber.is_healthy,
            lambda: stop["f"],
        ),
        safety_poll=lambda: monitor.mode,
        stream_healthy=grabber.is_healthy,
        stream_status=lambda: _live_stream_status(grabber),
        loop_beat=monitor.beat,
        pose_info=lambda: dict(localizer.last_info),
        pose_jump_pause=lambda _distance: safety.force("hover"),
        inspection_ack=inspection_ack,
        inspection_hold=lambda action: monitor.run_while_holding_zero(
            action,
            grabber.is_healthy,
            lambda: stop["f"],
        ),
        pcmd_timing=monitor.pcmd_timing_snapshot,
        ground_speed=ground_speed.sample,
        log_tick=command_log,
        # Desktop AUTO contract: localization loss holds zero PCMD.
        land_on_localization_loss=False,
    )


def _run_live_route_after_takeoff(
    *,
    monitor,
    grabber,
    localizer,
    waypoints,
    stop,
    controller,
    safety,
    stick_monitor,
    drone,
    send_pcmd,
    ground_speed,
    command_log,
    yaw_sign: int,
) -> str:
    boot_reason = _wait_for_boot_lock(
        monitor,
        grabber,
        localizer,
        waypoints,
        controller,
        stop,
    )
    if boot_reason is not None:
        return boot_reason
    auto_ok, auto_reason = monitor.arming_allowed(
        grabber.is_healthy,
        lambda: stop["f"],
    )
    if not auto_ok:
        raise SystemExit(f"[fly] final AUTO authorization failed: {auto_reason}")
    print(
        f"[fly] locked (mode={dict(localizer.last_info).get('next_mode')}, "
        f"inliers={dict(localizer.last_info).get('inliers')}). START AUTO."
    )

    def inspection_ack(metadata, pose):
        return _capture_inspection_frame(
            drone,
            grabber,
            metadata,
            pose,
            controller.cfg.map_frame,
        )

    hooks = _make_live_loop_hooks(
        localizer=localizer,
        safety=safety,
        stick_monitor=stick_monitor,
        drone=drone,
        send_pcmd=send_pcmd,
        monitor=monitor,
        grabber=grabber,
        stop=stop,
        ground_speed=ground_speed,
        command_log=command_log,
        inspection_ack=inspection_ack,
    )
    reason = _run_until(hooks, controller, waypoints, yaw_sign, stop)
    return monitor.reason or reason if monitor.terminated.is_set() else reason


def _authorize_cleanup_zero(monitor, grabber, stop, must_land: bool, reason: str) -> str:
    if monitor is None:
        return reason
    monitor.stop()
    if monitor.reason:
        reason = monitor.reason
    if not must_land or reason.startswith("EMERGENCY"):
        return reason
    _sent, authorization_reason, _actual = monitor.send_authorized(
        (0, 0, 0, 0),
        lambda: grabber is not None and grabber.is_healthy(),
        lambda: stop["f"],
    )
    if "EMERGENCY" in str(authorization_reason).upper():
        return "EMERGENCY command during cleanup -> motor cut"
    return reason


def _perform_terminal_flight_action(
    *,
    must_land: bool,
    drone,
    reason: str,
    monitor,
    emergency_action,
    landing_action,
) -> None:
    if not must_land:
        if drone is not None:
            print(
                "[fly] no TakeOff attempt; closing stream/connection without Landing()",
                flush=True,
            )
        else:
            print("[fly] never connected; nothing to land", flush=True)
        return
    if drone is None:
        print("[fly] never connected; nothing to land", flush=True)
        return
    if not reason.startswith("EMERGENCY"):
        print("[fly] landing")
        try:
            landing_action("landing/landed")
        except (Exception, SystemExit) as exc:
            print(f"[fly] warning: landing did not confirm: {exc}", flush=True)
        return
    print("[fly] EMERGENCY: cutting motors (drone will drop)", flush=True)
    if monitor is not None and monitor.emergency_issued:
        return
    try:
        emergency_action()
    except Exception as exc:
        print(
            f"[fly] Emergency() failed ({exc}); falling back to Landing()",
            flush=True,
        )
        try:
            landing_action("emergency fallback landing/landed")
        except (Exception, SystemExit) as fallback_exc:
            print(
                f"[fly] warning: Landing() also failed: {fallback_exc}",
                flush=True,
            )


def _close_live_flight_resources(
    *,
    grabber,
    drone,
    via_skycontroller: bool,
    command_log,
    reason: str,
) -> None:
    if grabber is not None:
        try:
            grabber.stop()
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
            pass
    if drone is not None and via_skycontroller:
        try:
            set_piloting_source(drone, "SkyController")
        except Exception as exc:
            print(
                f"[fly] warning: could not restore SkyController ownership: {exc}",
                flush=True,
            )
    if drone is not None:
        try:
            drone.disconnect()
        except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
            pass
    try:
        command_log.event(event="terminal", reason=reason)
    except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
        pass
    command_log.close()


def fly(ip: str, yaw_sign: int, gimbal_pitch: float, controller: str, safety_file: str,
        cmd_log_path: str = "", max_altitude_m: float | None = None,
        max_distance_m: float | None = None, distance_geofence: bool = False):
    raise SystemExit(
        "[fly] live TakeOff is only allowed from the operator UI "
        "(控制介面程式/真機串流/啟動.sh). Use --grab-only for live localize "
        "with props off, or --dry-run."
    )


def _run_until(hooks, ctrl, wp, yaw_sign, stop):
    """run_loop, but also honor the Ctrl-C stop flag between iterations."""
    orig_now = hooks.now
    def guarded_now():
        if stop["f"]:
            raise KeyboardInterrupt
        return orig_now()
    hooks.now = guarded_now
    try:
        return run_loop(
            hooks,
            ctrl,
            wp,
            yaw_sign=yaw_sign,
            enforce_weak_pose_gate=REAL_FLIGHT_WEAK_POSE_GATE,
        )
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
            except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
                pass
        if drone is not None:
            try:
                drone.disconnect()
            except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
                pass


# ---------------------------------------------------------------------------
# Dry-run: toy dynamics + a fake localizer, to exercise the loop with no hardware

def dry_run(yaw_sign: int, steps: int = 12000, cmd_log_path: str = ""):
    """Closed loop against a kinematic ANAFI stand-in in raw-GLOMAP frame.

    Body command -> world motion using the SAME sign conventions the real PCMD
    mapping assumes: +pitch = forward, +roll = body-right, +gaz = measured map-up,
    and +yaw = turn. A perfect localizer returns the simulated motion heading.
    """
    import real_path_follow_controller as rpf
    # pure path-follow smoke test: no poles / no inspection stops, so this exercises
    # the wiring + FOLLOW/REJOIN/LAND terminal, not the (separately-validated) look-at
    # inspection behavior.
    ctrl, wp = build_controller(require_approved=False)

    # start on the first waypoint, heading roughly toward the second
    C = wp[0].astype(float).copy()
    h = ctrl.cfg.map_frame.heading(wp[1] - wp[0])
    KP, KG, KY = 0.06, 0.05, 0.06     # per-% per-step response (matches autoflight._Sim)

    class _State:
        pass
    S = _State(); S.C = C; S.h = h; S.t = 0.0
    dt = 1.0 / CTRL_HZ

    def get_pose():
        p = rpf.Pose(x=float(S.C[0]), y=float(S.C[1]), z=float(S.C[2]),
                     yaw=S.h, stamp=S.t)
        return p

    def send_pcmd(roll, pitch, yaw, gaz):
        # command_to_body_percent already applied yaw_sign; the plant must not
        # apply it a second time or a wrong sign looks correct in dry-run.
        S.h = _wrap(S.h + KY * yaw * dt)                 # yaw turns heading
        fwd = KP * pitch * dt
        forward_map = (
            math.cos(S.h) * ctrl.cfg.map_frame.east
            + math.sin(S.h) * ctrl.cfg.map_frame.north
        )
        right_map = (
            math.sin(S.h) * ctrl.cfg.map_frame.east
            - math.cos(S.h) * ctrl.cfg.map_frame.north
        )
        S.C += fwd * forward_map
        S.C += KP * roll * dt * right_map
        S.C += KG * gaz * dt * ctrl.cfg.map_frame.up
        S.t += dt

    # inject the sim time into run_loop so freshness math lines up
    clog = CommandLog(cmd_log_path or default_cmd_log_path("dryrun"), sink="dry-run")
    hooks = LoopHooks(get_pose=get_pose, olympe_yaw=lambda: None,
                      send_pcmd=send_pcmd,
                      ground_speed=lambda: (0.0, S.t),
                      log_tick=clog, now=lambda: S.t)
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
        return run_loop(
            hooks,
            ctrl,
            wp,
            yaw_sign=yaw_sign,
            verbose=False,
            enforce_weak_pose_gate=GATE_WEAK,
        )
    except KeyboardInterrupt:
        return "step cap"


# ---------------------------------------------------------------------------
# Self-test: heading fusion + PCMD sign sanity (pure python, no deps)

def _selftest_heading_and_pcmd(rpf) -> None:
    estimator = HeadingEstimator()
    olympe_yaw = math.radians(110.0)
    true_heading = math.radians(40.0)
    estimator.update(true_heading, olympe_yaw)
    estimate = estimator.heading(olympe_yaw)
    assert abs(_wrap(estimate - true_heading)) < 1e-9, (
        f"heading fusion off: {math.degrees(estimate):.1f} "
        f"vs {math.degrees(true_heading):.1f}"
    )

    pose = rpf.Pose(x=0, y=0, z=0, yaw=0.0)
    ahead = rpf.Command(
        "FOLLOW",
        np.array([1.2, 0.0, 0.0]),
        yaw_target=0.0,
        goal=np.array([1.2, 0.0, 0.0]),
        path_error=0.0,
        progress=0.0,
    )
    roll, pitch, yaw, gaz = rpf.command_to_body_percent(ahead, pose)
    assert pitch > 0 and abs(yaw) < 3 and roll == 0, (roll, pitch, yaw, gaz)
    behind = rpf.Command(
        "FOLLOW",
        np.array([-1.2, 0.0, 0.0]),
        yaw_target=math.pi,
        goal=np.array([-1.2, 0.0, 0.0]),
        path_error=0.0,
        progress=0.0,
    )
    roll, pitch, yaw, gaz = rpf.command_to_body_percent(behind, pose)
    assert pitch == 0 and abs(yaw) == 20, (
        "should spin in place, not fly backward",
        roll,
        pitch,
        yaw,
        gaz,
    )
    upward = rpf.Command(
        "FOLLOW",
        np.array([0.0, -1.0, 0.0]),
        yaw_target=0.0,
        goal=np.array([0.0, -1.0, 0.0]),
        path_error=0.0,
        progress=0.0,
    )
    _, _, _, gaz = rpf.command_to_body_percent(upward, pose)
    assert gaz > 0, f"+gaz should be ascend, got {gaz}"


def _selftest_stream_and_pose_gates(rpf) -> None:
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
        send_pcmd=lambda roll, pitch, yaw, gaz: sent.append(
            (roll, pitch, yaw, gaz)
        ),
        stream_healthy=lambda: False,
        stream_status=lambda: "selftest disconnected",
        now=fake_now,
    )
    try:
        run_loop(hooks, None, None, verbose=False)
    except KeyboardInterrupt:
        pass
    assert sent and all(command == (0, 0, 0, 0) for command in sent), sent

    nan_sent = []
    nan_ticks = {"n": 0, "t": 0.0}

    def nan_now():
        nan_ticks["n"] += 1
        nan_ticks["t"] += 1.0 / CTRL_HZ
        if nan_ticks["n"] > 8:
            raise KeyboardInterrupt
        return nan_ticks["t"]

    waypoints = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    controller = rpf.RouteAutoController(
        waypoints,
        poles=[],
        config=rpf.ControlConfig(inspect_waypoints=()),
    )
    hooks = LoopHooks(
        get_pose=lambda: rpf.Pose(
            x=float("nan"),
            y=0.0,
            z=0.0,
            yaw=0.0,
            stamp=nan_ticks["t"],
        ),
        olympe_yaw=lambda: None,
        send_pcmd=lambda roll, pitch, yaw, gaz: nan_sent.append(
            (roll, pitch, yaw, gaz)
        ),
        now=nan_now,
    )
    try:
        run_loop(hooks, controller, waypoints, verbose=False)
    except KeyboardInterrupt:
        pass
    assert nan_sent and all(command == (0, 0, 0, 0) for command in nan_sent), (
        "NaN pose must hover",
        nan_sent[:5],
    )


class _SafetySequence:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.index = 0

    def poll(self):
        value = self.sequence[min(self.index, len(self.sequence) - 1)]
        self.index += 1
        return value


def _selftest_safety_monitor() -> None:
    calls = {"pcmd": [], "land": 0, "emergency": 0}
    monitor = SafetyMonitor(
        lambda roll, pitch, yaw, gaz: calls["pcmd"].append(
            (roll, pitch, yaw, gaz)
        ),
        _SafetySequence(["HOVER", "LAND"]),
        land_cb=lambda: calls.__setitem__("land", calls["land"] + 1),
        emergency_cb=lambda: calls.__setitem__(
            "emergency", calls["emergency"] + 1
        ),
        timeout_s=0.06,
    ).start()
    time.sleep(0.4)
    monitor.stop()
    assert calls["land"] == 1, (
        "LAND must fire exactly once, independent of any loop",
        calls,
    )
    assert monitor.terminated.is_set(), "LAND must terminate the mission"
    assert (0, 0, 0, 0) in calls["pcmd"], (
        "HOVER must stream zero PCMD from the monitor thread"
    )

    manual_calls = {"pcmd": []}
    manual_monitor = SafetyMonitor(
        lambda roll, pitch, yaw, gaz: manual_calls["pcmd"].append(
            (roll, pitch, yaw, gaz)
        ),
        _SafetySequence(["MANUAL"]),
        timeout_s=0.06,
    ).start()
    time.sleep(0.3)
    manual_monitor.stop()
    assert manual_calls["pcmd"] == [], (
        "MANUAL must send NOTHING",
        manual_calls,
    )

    prebeat_calls = {"pcmd": []}
    prebeat_monitor = SafetyMonitor(
        lambda roll, pitch, yaw, gaz: prebeat_calls["pcmd"].append(
            (roll, pitch, yaw, gaz)
        ),
        _SafetySequence(["AUTO"]),
        timeout_s=0.06,
    ).start()
    time.sleep(0.3)
    prebeat_monitor.stop()
    assert prebeat_calls["pcmd"] == [], (
        "stall watchdog must not fire before first beat",
        prebeat_calls,
    )


def _selftest_low_confidence_hover(rpf) -> None:
    state = {"pcmd": [], "reloc": 0, "manual": 0, "t": 0.0}

    def now():
        state["t"] += 0.2
        if state["t"] > LOST_LAND_S + 20.0:
            raise KeyboardInterrupt
        return state["t"]

    waypoints = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    controller = rpf.RouteAutoController(
        waypoints,
        poles=[],
        config=rpf.ControlConfig(inspect_waypoints=()),
    )
    hooks = LoopHooks(
        get_pose=lambda: rpf.Pose(
            x=0.0,
            y=0.0,
            z=0.0,
            yaw=0.0,
            stamp=state["t"],
        ),
        olympe_yaw=lambda: None,
        send_pcmd=lambda roll, pitch, yaw, gaz: state["pcmd"].append(
            (roll, pitch, yaw, gaz)
        ),
        pose_confidence=lambda: 30,
        force_relocalize=lambda: state.__setitem__("reloc", state["reloc"] + 1),
        request_manual=lambda: (
            state.__setitem__("manual", state["manual"] + 1) or True
        ),
        now=now,
    )
    try:
        run_loop(hooks, controller, waypoints, verbose=False)
    except KeyboardInterrupt:
        pass
    assert state["pcmd"] and all(
        command == (0, 0, 0, 0) for command in state["pcmd"]
    ), ("low-conf must hover", state["pcmd"][:5])
    assert state["reloc"] > 0, "low-conf must ask the tracker to relocalize (MegaLoc)"
    assert state["manual"] == 0, (
        "localization timeout must not request automatic manual takeover",
        state,
    )


def selftest():
    import real_path_follow_controller as rpf

    _selftest_heading_and_pcmd(rpf)
    _selftest_stream_and_pose_gates(rpf)
    _selftest_safety_monitor()
    _selftest_low_confidence_hover(rpf)
    print(
        "selftest OK: heading fusion, PCMD signs, stream-lost + NaN-pose hover gates, "
        "SafetyMonitor LAND/HOVER authority, MANUAL-silence, pre-beat watchdog gating, "
        "and low-confidence hover+MegaLoc without timer-based manual handoff sane"
    )


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ANAFI path-follow closed-loop flight (Olympe).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true", help="pure-python checks, no deps")
    mode.add_argument("--dry-run", action="store_true", help="toy dynamics, no drone")
    mode.add_argument("--grab-only", action="store_true", help="live localize, PROPS OFF, no arm")
    mode.add_argument("--fly", action="store_true",
                      help="rejected: live TakeOff is only allowed from the operator UI")
    ap.add_argument("--ip", default=DRONE_IP_REAL,
                    help=f"real {DRONE_IP_REAL} / skyctrl {DRONE_IP_SKYCTRL} / sphinx {DRONE_IP_SIM}")
    ap.add_argument("--yaw-sign", type=int, default=1, choices=(-1, 1),
                    help="flip if the drone yaws the WRONG way on bench (verify props-off)")
    ap.add_argument("--gimbal-pitch", type=float, default=GIMBAL_PITCH_DEG,
                    help="camera tilt vs horizon (deg, negative=down)")
    ap.add_argument("--max-altitude-m", type=float,
                    help="optional advisory firmware maximum altitude in metres")
    ap.add_argument("--max-distance-m", type=float,
                    help="optional advisory firmware maximum distance from takeoff in metres")
    ap.add_argument("--distance-geofence", action=argparse.BooleanOptionalAction, default=False,
                    help="enable GPS-dependent firmware NoFlyOverMaxDistance (default: disabled)")
    ap.add_argument("--controller", default=os.environ.get("SFM_OLYMPE_CONTROLLER", "auto"),
                    help="auto / drone / anafi / skycontroller3")
    ap.add_argument("--safety-file", default=SAFETY_FILE,
                    help="write auto/hover/manual/land here for runtime safety switching")
    ap.add_argument("--secs", type=float, default=20.0, help="--grab-only duration")
    ap.add_argument("--cmd-log", default="",
                    help="JSONL per-tick command log path (default: outputs/flight_logs/)")
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
        fly(args.ip, args.yaw_sign, args.gimbal_pitch, args.controller, args.safety_file,
            cmd_log_path=args.cmd_log,
            max_altitude_m=args.max_altitude_m,
            max_distance_m=args.max_distance_m,
            distance_geofence=args.distance_geofence)


if __name__ == "__main__":
    main()

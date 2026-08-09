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
  --fly       : approved site-profile flight with runtime safety and firmware limits.
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

from safety_command import (  # noqa: E402
    default_safety_file as _default_safety_file,  # noqa: F401 - compatibility API
    prepare_safety_file as _prepare_safety_file,
    safety_file_from_environment as _safety_file_from_environment,
    validate_safety_directory as _validate_safety_directory,
    validate_safety_file_stat as _validate_safety_file_stat,
)
# 2026-08-06 audit: a hard-coded /home/allen/足球場 probe used to sit here and,
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
# 720p live frame older than this -> HOVER before localization. Distinct from the
# operator interface's _STREAM_STALE_S (0.75 s), which only feeds its stick-handoff
# timer; this one decides whether autonomy may run the localizer on the frame.
STREAM_STALE_S = 0.5
LOST_LAND_S = 4.0           # no fresh pose for this long -> auto-land
# Localization lost + MegaLoc cannot relocalize this long -> hand to the human pilot
# (only if a SkyController manual pilot exists; else the LOST_LAND_S failsafe still lands).
LOST_MANUAL_S = _env_float("SFM_LOST_MANUAL_S", 3.0, minimum=0.0, maximum=300.0)
LOW_CONF_INLIERS = _env_int("SFM_LOW_CONF_INLIERS", 60, minimum=1, maximum=10000)
RECOVERY_GOOD_FIXES = _env_int(
    "SFM_RECOVERY_GOOD_FIXES", 2, minimum=1, maximum=20)
# Stream lost this long -> auto-land (was: hover forever until battery death).
STREAM_LOST_LAND_S = _env_float(
    "SFM_STREAM_LOST_LAND_S", 15.0, minimum=0.5, maximum=600.0)
# A WEAK (low-confidence) fix in repetitive line-corridor geometry can be plausible
# but wrong and still pass the jump/deviation gates; treat it as "uncertain" -> hover.
# SFM_GATE_WEAK=0 restores the old behavior of flying on weak fixes (tuning only).
GATE_WEAK = os.environ.get("SFM_GATE_WEAK", "1") != "0"
WEAK_HOVER_LAND_S = _env_float(
    "SFM_WEAK_HOVER_LAND_S", 8.0, minimum=0.5, maximum=600.0)
# Reject a fresh fix that jumps farther than this from the last accepted fix
# (MAP UNITS). A second consecutive fix agreeing with the first one is accepted,
# so genuine relocalization after hover drift still recovers.
MAX_POSE_JUMP_U = _env_float(
    "SFM_MAX_POSE_JUMP_U", 1.5, minimum=0.01, maximum=100.0)
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
# Keep these fixed to the scale-free safety core's physical-speed contract.
# Map distances are intentionally never converted to metres for this gate.
LANDING_SPEED_THRESHOLD_MPS = 0.10
GROUND_SPEED_MAX_AGE_S = 0.50
OFFSET_EMA = 0.25           # visual-to-attitude offset smoothing after first anchor
FIRST_FIX_TIMEOUT_S = 25.0  # how long to wait for BOOT_INIT -> first TRACK
AUTO_CONSENT_TIMEOUT_S = 30.0
BOOT_LOCK_FIXES = 3
BOOT_START_MAX_U = _env_float(
    "SFM_BOOT_START_MAX_U", 1.5, minimum=0.05, maximum=100.0)
INSPECTION_TIMEOUT_S = _env_float(
    "SFM_INSPECTION_TIMEOUT_S", 15.0, minimum=1.0, maximum=120.0)
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
    """Require stable, fresh consecutive fixes close to the route start."""

    def __init__(self, route_start, required_fixes: int = BOOT_LOCK_FIXES,
                 max_start_distance: float = BOOT_START_MAX_U,
                 max_fix_jump: float = MAX_POSE_JUMP_U):
        self.route_start = np.asarray(route_start, dtype=float)
        if self.route_start.shape != (3,) or not np.isfinite(self.route_start).all():
            raise ValueError("BOOT route start must be a finite 3-vector")
        self.required_fixes = int(required_fixes)
        self.max_start_distance = float(max_start_distance)
        self.max_fix_jump = float(max_fix_jump)
        if (self.required_fixes < 2
                or not all(math.isfinite(v) and v > 0.0
                           for v in (self.max_start_distance, self.max_fix_jump))):
            raise ValueError("BOOT lock thresholds must be finite/positive and require >=2 fixes")
        self.count = 0
        self._last = None

    def reset(self) -> None:
        self.count = 0
        self._last = None

    def observe(self, pose, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        try:
            values = np.array([pose.x, pose.y, pose.z, pose.stamp], dtype=float)
        except (AttributeError, TypeError, ValueError):
            self.reset()
            return False
        age = now - float(values[3])
        pos = values[:3]
        valid = (math.isfinite(now) and np.isfinite(values).all()
                 and -0.05 <= age <= POSE_STALE_S
                 and float(np.linalg.norm(pos - self.route_start)) <= self.max_start_distance
                 and (self._last is None
                      or float(np.linalg.norm(pos - self._last)) <= self.max_fix_jump))
        if not valid:
            self.reset()
            return False
        self._last = pos
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

    if max_altitude_m is None or max_distance_m is None:
        raise RuntimeError("firmware altitude/distance limits were not provided")
    altitude = float(max_altitude_m)
    distance = float(max_distance_m)
    if not math.isfinite(altitude) or altitude <= 0.0:
        raise RuntimeError("max altitude must be finite and > 0 m")
    if not math.isfinite(distance) or distance <= 0.0:
        raise RuntimeError("max distance must be finite and > 0 m")

    battery_state = drone.get_state(BatteryStateChanged)
    battery = battery_state.get("percent") if isinstance(battery_state, dict) else None
    if not isinstance(battery, (int, float)) or not 0 <= float(battery) <= 100:
        raise RuntimeError(f"battery state unavailable/invalid: {battery!r}")
    if float(battery) < 30.0:
        raise RuntimeError(f"battery {float(battery):.0f}% is below the 30% advisory floor")

    if distance_geofence:
        from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
        gps_state = drone.get_state(GPSFixStateChanged)
        gps_fixed = gps_state.get("fixed") if isinstance(gps_state, dict) else None
        if int(gps_fixed or 0) != 1:
            raise RuntimeError("distance geofence requires a confirmed GPS fix")

    def confirm_limit(command, state_message, requested: float, label: str) -> None:
        before = drone.get_state(state_message)
        minimum = before.get("min") if isinstance(before, dict) else None
        maximum = before.get("max") if isinstance(before, dict) else None
        if not all(isinstance(v, (int, float)) and math.isfinite(float(v))
                   for v in (minimum, maximum)):
            raise RuntimeError(f"{label} firmware bounds unavailable")
        if not float(minimum) <= requested <= float(maximum):
            raise RuntimeError(
                f"{label} {requested:g} outside firmware range "
                f"[{float(minimum):g}, {float(maximum):g}]")
        _await_confirmed_action(drone(command), label)
        after = drone.get_state(state_message)
        actual = after.get("current") if isinstance(after, dict) else None
        if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), requested, rel_tol=1e-5, abs_tol=0.05):
            raise RuntimeError(
                f"{label} readback mismatch: requested={requested:g} actual={actual!r}")

    confirm_limit(MaxAltitude(current=altitude), MaxAltitudeChanged,
                  altitude, "MaxAltitude")
    confirm_limit(MaxDistance(value=distance), MaxDistanceChanged,
                  distance, "MaxDistance")
    geofence_value = int(bool(distance_geofence))
    _await_confirmed_action(
        drone(NoFlyOverMaxDistance(shouldNotFlyOver=geofence_value)),
        "NoFlyOverMaxDistance",
    )
    geofence_state = drone.get_state(NoFlyOverMaxDistanceChanged)
    geofence_actual = (geofence_state.get("shouldNotFlyOver")
                       if isinstance(geofence_state, dict) else None)
    if int(geofence_actual if geofence_actual is not None else -1) != geofence_value:
        raise RuntimeError(
            "NoFlyOverMaxDistance readback mismatch: "
            f"requested={geofence_value} actual={geofence_actual!r}")
    return {
        "battery_percent": int(battery),
        "max_altitude_m": altitude,
        "max_distance_m": distance,
        "distance_geofence": bool(distance_geofence),
    }


# ---------------------------------------------------------------------------
# Map-frame heading: Olympe yaw (fast) anchored to measured visual orientation

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class HeadingEstimator:
    """Fuse visual MapFrame heading with Olympe's clockwise-from-North NED yaw."""

    def __init__(self, map_frame=None):
        import real_path_follow_controller as rpf
        self.map_frame = map_frame or rpf.LEGACY_MAP_FRAME
        self.offset = None
        self._last_visual_heading = None

    @staticmethod
    def _olympe_to_map_ccw(olympe_yaw: float) -> float:
        # Olympe AttitudeChanged.yaw is NED: clockwise from North. MapFrame
        # heading is counter-clockwise from East.
        return _wrap(math.pi / 2.0 - float(olympe_yaw))

    def update(self, visual_map_yaw: float, olympe_yaw: float | None):
        """Anchor attitude telemetry to the localized camera/body forward ray."""
        try:
            visual = _wrap(float(visual_map_yaw))
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(visual):
            return
        self._last_visual_heading = visual
        if olympe_yaw is None:
            return
        try:
            attitude = self._olympe_to_map_ccw(float(olympe_yaw))
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(attitude):
            return
        new = _wrap(visual - attitude)
        self.offset = new if self.offset is None else _wrap(
            self.offset + OFFSET_EMA * _wrap(new - self.offset)
        )

    def heading(self, olympe_yaw: float | None) -> float | None:
        if olympe_yaw is not None and self.offset is not None:
            try:
                attitude = self._olympe_to_map_ccw(float(olympe_yaw))
            except (TypeError, ValueError, OverflowError):
                attitude = float("nan")
            if math.isfinite(attitude):
                return _wrap(attitude + self.offset)
        return self._last_visual_heading


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


class OlympeGroundSpeedTracker:
    """Read fresh NED horizontal speed from Olympe ``SpeedChanged`` events.

    ``get_state`` alone cannot distinguish a new event from a cached value.  The
    event UUID and SDK receipt time make the landing gate fail closed when speed
    telemetry stops updating.
    """

    def __init__(self, drone, message=None, *, now=time.monotonic, wall_now=time.time):
        if message is None:
            from olympe.messages.ardrone3.PilotingState import SpeedChanged
            message = SpeedChanged
        self.drone = drone
        self.message = message
        self._now = now
        self._wall_now = wall_now
        self._marker = None
        self._sample = None

    def sample(self) -> tuple[float, float] | None:
        """Return ``(horizontal_mps, receipt_monotonic_s)`` or ``None``."""
        try:
            event = self.drone.get_last_event(self.message)
            marker = str(event.uuid)
            state = dict(event.args)
            now = float(self._now())
            event_age_s = float(self._wall_now()) - float(event.date.timestamp())
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return None
        if (not math.isfinite(now) or not math.isfinite(event_age_s)
                or event_age_s < -0.05):
            return None
        if marker != self._marker:
            self._marker = marker
            self._sample = (state, now - max(0.0, event_age_s))
        if self._sample is None:
            return None
        state, stamp = self._sample
        try:
            speed_x = float(state["speedX"])
            speed_y = float(state["speedY"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in (speed_x, speed_y, stamp)):
            return None
        return math.hypot(speed_x, speed_y), float(stamp)


def landing_speed_allows_land(
    sample,
    now: float,
    *,
    threshold_mps: float = LANDING_SPEED_THRESHOLD_MPS,
    max_age_s: float = GROUND_SPEED_MAX_AGE_S,
) -> tuple[bool, str, float | None]:
    """Fail-closed physical-speed gate for normal end-of-route landing."""
    if sample is None:
        return False, "ground speed unavailable", None
    try:
        speed, stamp = sample
        if isinstance(speed, bool) or isinstance(stamp, bool):
            raise ValueError
        speed = float(speed)
        stamp = float(stamp)
        now = float(now)
    except (TypeError, ValueError, OverflowError):
        return False, "ground speed invalid", None
    if not all(math.isfinite(value) for value in (speed, stamp, now)) or speed < 0.0:
        return False, "ground speed invalid", None
    age_s = now - stamp
    if age_s < -0.05:
        return False, "ground speed timestamp is in the future", speed
    if age_s > float(max_age_s):
        return False, f"ground speed stale ({age_s:.2f}s)", speed
    if speed > float(threshold_mps):
        return False, f"ground speed {speed:.3f} m/s > {threshold_mps:.3f} m/s", speed
    return True, f"ground speed {speed:.3f} m/s <= {threshold_mps:.3f} m/s", speed


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


def set_gimbal(drone, pitch_deg: float, *, require_confirmation: bool = False,
               timeout_s: float = 2.0, tolerance_deg: float = 3.0):
    """Point the camera to a fixed absolute (horizon-referenced) tilt so live frames
    resemble the map reference views. Absolute frame == EIS-stabilized vs horizon."""
    target = float(pitch_deg)
    if (not math.isfinite(target) or not math.isfinite(float(timeout_s))
            or timeout_s <= 0.0 or not math.isfinite(float(tolerance_deg))
            or tolerance_deg <= 0.0):
        return False
    from olympe.messages.gimbal import absolute_attitude_bounds, attitude, set_target
    if require_confirmation:
        try:
            bounds = drone.get_state(absolute_attitude_bounds)
            lo = float(_gimbal_state_value(bounds, "min_pitch"))
            hi = float(_gimbal_state_value(bounds, "max_pitch"))
        except (KeyError, TypeError, ValueError, RuntimeError):
            return False
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= target <= hi):
            return False
    try:
        res = _bounded_wait(
            drone(set_target(
                gimbal_id=0, control_mode="position",
                yaw_frame_of_reference="none", yaw=0.0,
                pitch_frame_of_reference="absolute", pitch=target,
                roll_frame_of_reference="none", roll=0.0)),
            float(timeout_s), "gimbal target")
    except Exception:
        return False
    if hasattr(res, "success") and not res.success():
        print("[flight] warning: gimbal target command did not report success", flush=True)
        return False
    if not require_confirmation:
        return True
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        try:
            state = drone.get_state(attitude)
            actual = float(_gimbal_state_value(state, "pitch_absolute"))
        except (KeyError, TypeError, ValueError, RuntimeError):
            actual = float("nan")
        if math.isfinite(actual) and abs(actual - target) <= float(tolerance_deg):
            return True
        time.sleep(0.05)
    return False


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
    request_manual: callable | None = None  # () -> bool; switch to MANUAL on persistent loss (SkyController)
    stick_active: callable | None = None    # () -> bool; the pilot has physically moved the sticks
    safety_poll: callable | None = None  # () -> AUTO/HOVER/MANUAL/LAND/EMERGENCY
    stream_healthy: callable | None = None  # () -> bool; false means live stream stale/lost
    stream_status: callable | None = None   # () -> str; operator log detail
    loop_beat: callable | None = None       # () -> None; watchdog heartbeat, called every tick
    pose_info: callable | None = None       # () -> dict; localizer last_info (LOGGING ONLY)
    inspection_ack: callable | None = None  # (metadata, pose) -> bool after gimbal+capture success
    inspection_hold: callable | None = None  # (blocking_action) -> bool while zero PCMD is sustained
    pcmd_timing: callable | None = None       # () -> latest desired/wire monotonic_ns markers
    ground_speed: callable | None = None      # () -> (horizontal_mps, monotonic_stamp) | None
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
                except Exception:
                    self.mode = "HOVER"
            try:
                stream_ok = bool(stream_healthy())
            except Exception:
                stream_ok = False
            try:
                stopped = bool(stop_requested())
            except Exception:
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

    def send_authorized(self, pcmd, stream_healthy, stop_requested):
        """Authorize every autonomy PCMD, including zero, under one lock."""
        with self._io_lock:
            if self._safety is not None:
                try:
                    self.mode = str(self._safety.poll()).upper()
                except Exception:
                    self.mode = "HOVER"
            try:
                stream_ok = bool(stream_healthy())
            except Exception:
                stream_ok = False
            try:
                stopped = bool(stop_requested())
            except Exception:
                stopped = True
            if self.mode == "EMERGENCY":
                self._latch_terminal("EMERGENCY", "EMERGENCY command -> motor cut")
            elif self.mode == "LAND":
                self._latch_terminal("LAND", "safety LAND command -> land")
            elif stopped:
                self._latch_terminal("LAND", "operator termination signal -> land")
            target_source = ("SkyController" if self.terminal_action == "NONE"
                             and self.mode == "MANUAL" else "Controller")
            source_ok = (self.terminal_action == "EMERGENCY"
                         or self._ensure_piloting_source(target_source))
            if not source_ok:
                actual = None
                self._desired_pcmd = (0, 0, 0, 0)
                self._desired_valid = True
                self._desired_stream_healthy = stream_healthy
                self._desired_stop_requested = stop_requested
                if self._send_counted(0, 0, 0, 0):
                    actual = (0, 0, 0, 0)
                return False, f"piloting source {target_source} not confirmed", actual
            ok, why = arming_allowed(
                self.mode, stream_ok, stopped, self.terminated.is_set())
            if ok:
                try:
                    desired = tuple(pcmd)
                except TypeError:
                    desired = ()
                valid_shape = len(desired) == 4
                valid_types = valid_shape and all(
                    isinstance(value, (int, np.integer))
                    and not isinstance(value, (bool, np.bool_))
                    for value in desired
                )
                if valid_types:
                    desired = tuple(int(value) for value in desired)
                in_envelope = valid_types and (
                    abs(desired[0]) <= self._max_translation_pcmd
                    and abs(desired[1]) <= self._max_translation_pcmd
                    and abs(desired[2]) <= self._max_yaw_pcmd
                    and abs(desired[3]) <= self._max_translation_pcmd
                )
                if not in_envelope:
                    self._desired_pcmd = (0, 0, 0, 0)
                    self._desired_valid = True
                    self._desired_stream_healthy = stream_healthy
                    self._desired_stop_requested = stop_requested
                    self._desired_updated_mono_ns = time.monotonic_ns()
                    actual = ((0, 0, 0, 0)
                              if self._send_counted(0, 0, 0, 0) else None)
                    self._control_wake.set()
                    return False, "PCMD malformed or outside approved envelope", actual
                self._desired_pcmd = desired
                self._desired_valid = True
                self._desired_stream_healthy = stream_healthy
                self._desired_stop_requested = stop_requested
                self._desired_updated_mono_ns = time.monotonic_ns()
                if self._thread.is_alive():
                    # Wake the independent 20 Hz sender. Perception never waits
                    # for command completion and cannot starve PCMD refreshes.
                    self._control_wake.set()
                else:
                    # Unit tests and pre-start/final cleanup keep the historical
                    # synchronous behavior because no sender thread exists.
                    if not self._send_counted(*desired):
                        return (False,
                                f"PCMD send failed: {self.last_pcmd_send_error}",
                                None)
                return True, "atomic AUTO authorization; latest PCMD queued", desired
            actual = None
            if (self.mode not in {"MANUAL", "EMERGENCY"}
                    and self.terminal_action != "EMERGENCY"):
                self._desired_pcmd = (0, 0, 0, 0)
                self._desired_valid = True
                self._desired_stream_healthy = stream_healthy
                self._desired_stop_requested = stop_requested
                self._desired_updated_mono_ns = time.monotonic_ns()
                if self._send_counted(0, 0, 0, 0):
                    actual = (0, 0, 0, 0)
                self._control_wake.set()
            else:
                self._desired_valid = False
            return False, why, actual

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
                external_stop = False
                if self._stop_requested is not None:
                    try:
                        external_stop = bool(self._stop_requested())
                    except Exception:
                        external_stop = True
                polled_mode = self.mode
                if self._safety is not None:
                    try:
                        polled_mode = self._safety.poll()
                        poll_warned = False
                    except Exception as exc:
                        polled_mode = "HOVER"
                        if not poll_warned:
                            poll_warned = True
                            print(f"[safety] SafetyMonitor.poll() FAILING ({exc!r}); operator "
                                  "LAND/HOVER/EMERGENCY may be unseen; failing closed to HOVER", flush=True)
                self.mode = str(polled_mode).upper()
                mode = self.mode
                if mode == "EMERGENCY":
                    self._latch_terminal("EMERGENCY", "EMERGENCY command -> motor cut")
                elif external_stop:
                    self._latch_terminal("LAND", "operator termination signal -> land")
                elif mode == "LAND":
                    self._latch_terminal("LAND", "safety LAND command -> land")

                target_source = ("SkyController" if self.terminal_action == "NONE"
                                 and mode == "MANUAL" else "Controller")
                source_ok = (self.terminal_action == "EMERGENCY"
                             or self._ensure_piloting_source(target_source))

                if self.terminal_action == "EMERGENCY":
                    terminal_handled = True
                    now = time.monotonic()
                    if not self._emergency_acted:
                        callback_request = ("EMERGENCY", self._emergency_cb, now)
                elif self.terminal_action == "LAND":
                    terminal_handled = True
                    now = time.monotonic()
                    if not self._land_acted:
                        if now >= self._land_retry_at:
                            self._send_counted(0, 0, 0, 0)
                        callback_request = ("LAND", self._land_cb, now)
                elif not source_ok:
                    self._desired_pcmd = (0, 0, 0, 0)
                    self._desired_valid = True
                    self._send_counted(0, 0, 0, 0)
                    continue
                if terminal_handled:
                    pass
                elif mode == "HOVER":
                    self._desired_pcmd = (0, 0, 0, 0)
                    self._desired_valid = True
                    self._send_counted(0, 0, 0, 0)
                    continue
                elif mode == "MANUAL":
                    self._desired_valid = False
                    continue
                elif self._inspection_hold.is_set():
                    self._send_counted(0, 0, 0, 0)
                    continue
                elif self._beat_seen and (time.monotonic() - self._beat_t) > self._timeout:
                    self._desired_pcmd = (0, 0, 0, 0)
                    self._desired_valid = True
                    self._send_counted(0, 0, 0, 0)
                    stall_now = time.monotonic()
                    if self._stall_since is None:
                        self._stall_since = stall_now
                    elif stall_now - self._stall_since >= WATCHDOG_LAND_S:
                        # Holding zero forever leaves the aircraft hovering until the
                        # battery dies with nobody driving it. Escalate to LAND.
                        print(f"[safety] WATCHDOG: control loop stalled > "
                              f"{stall_now - self._stall_since:.1f}s -> land", flush=True)
                        self._latch_terminal(
                            "LAND", "control loop stalled -> land")
                    if not warned:
                        print(f"[safety] WATCHDOG: control loop stalled > {self._timeout:.1f}s; "
                              "forcing zero PCMD (hover)", flush=True)
                        warned = True
                else:
                    warned = False
                    self._stall_since = None
                    if not self._desired_valid:
                        continue
                    try:
                        desired_stream_ok = (
                            self._desired_stream_healthy is None
                            or bool(self._desired_stream_healthy()))
                    except Exception:
                        desired_stream_ok = False
                    try:
                        desired_stopped = (
                            self._desired_stop_requested is not None
                            and bool(self._desired_stop_requested()))
                    except Exception:
                        desired_stopped = True
                    desired = (
                        self._desired_pcmd
                        if (desired_stream_ok and not desired_stopped
                            and self._desired_updated_mono_ns is not None
                            and time.monotonic_ns() - self._desired_updated_mono_ns
                            <= int(self._command_ttl_s * 1e9))
                        else (0, 0, 0, 0)
                    )
                    if desired == (0, 0, 0, 0):
                        self._desired_pcmd = desired
                    self._send_counted(*desired)
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


def run_loop(hooks: LoopHooks, ctrl, waypoints, yaw_sign: int = 1, verbose: bool = True):
    """Localize -> fuse heading -> RouteAutoController -> PCMD, at CTRL_HZ.

    Returns the terminal reason string. `ctrl` is a RouteAutoController; `waypoints`
    are raw-GLOMAP np arrays (its own list). Import here so --selftest needs no deps.
    """
    import real_path_follow_controller as rpf

    heading = HeadingEstimator(ctrl.cfg.map_frame if ctrl is not None else None)
    pcmd_controller = (
        rpf.YawAlignedPcmdController(ctrl.cfg) if ctrl is not None else None
    )

    def reset_pcmd_controller():
        if pcmd_controller is not None:
            pcmd_controller.reset()
    period = 1.0 / CTRL_HZ
    last_good = None
    uncertain_since = None
    uncertain_land_after = None
    recovery_good_fixes = 0
    stream_lost_since = None
    last_stream_warn = 0.0
    pending_jump = None
    pending_jump_stamp = None
    inspection_started = {}
    steps = 0
    reason = "stopped"
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
        if hooks.pcmd_timing is not None:
            try:
                rec.update(hooks.pcmd_timing() or {})
            except Exception:
                pass
        try:
            hooks.log_tick(rec)
        except Exception:
            pass

    def send_command(pcmd):
        """Route every loop-originated PCMD through one final authority."""
        pcmd = tuple(int(v) for v in pcmd)
        if hooks.send_authorized_pcmd is not None:
            return hooks.send_authorized_pcmd(pcmd)
        mode = hooks.safety_poll() if hooks.safety_poll is not None else "AUTO"
        try:
            stream_ok = hooks.stream_healthy is None or bool(hooks.stream_healthy())
        except Exception:
            stream_ok = False
        if mode in {"MANUAL", "EMERGENCY"}:
            return False, f"safety mode is {mode}", None
        authorized = mode == "AUTO" and stream_ok
        if authorized:
            hooks.send_pcmd(*pcmd)
            return True, "fallback AUTO authorization", pcmd
        hooks.send_pcmd(0, 0, 0, 0)
        return False, f"safety mode={mode} stream={stream_ok}", (0, 0, 0, 0)

    while True:
        t0 = hooks.now()
        if hooks.loop_beat is not None:
            hooks.loop_beat()
        safety_mode = hooks.safety_poll() if hooks.safety_poll is not None else "AUTO"
        rec = {
            "step": steps,
            "t": round(t0, 3),
            "t_mono_ns": time.monotonic_ns(),
            "safety": safety_mode,
        }
        if safety_mode == "EMERGENCY":
            reason = "EMERGENCY command -> motor cut"
            emit(rec, None, True, reason)
            break
        if safety_mode == "LAND":
            _sent, _why, actual = send_command((0, 0, 0, 0))
            reason = "safety LAND command -> land"
            emit(rec, actual, True, reason)
            break
        if safety_mode == "HOVER":
            reset_pcmd_controller()
            _sent, _why, actual = send_command((0, 0, 0, 0))
            emit(rec, actual, True, "safety HOVER: authorized zero PCMD")
            steps += 1
            dt = hooks.now() - t0
            if dt < period:
                time.sleep(period - dt)
            continue
        if safety_mode == "MANUAL":
            reset_pcmd_controller()
            emit(rec, None, True, "MANUAL: autonomy sends nothing (pilot has the sticks)")
            steps += 1
            dt = hooks.now() - t0
            if dt < period:
                time.sleep(period - dt)
            continue
        # The physical override. Checked every tick, before anything pose-derived,
        # because it is the pilot taking the aircraft back: nothing the autonomy is
        # in the middle of computing may delay it. Zero PCMD goes out FIRST, while
        # the PC still owns the piloting source; only then is MANUAL latched, after
        # which the loop is silent so it cannot fight the sticks.
        if hooks.stick_active is not None:
            try:
                sticks_moved = bool(hooks.stick_active())
            except Exception as exc:
                # A monitor that cannot answer is not evidence that the pilot is
                # idle. Treat it as an override and let the human sort it out.
                sticks_moved = True
                print(f"[safety] stick monitor raised ({exc!r}); assuming override",
                      flush=True)
            if sticks_moved:
                _sent, _why, actual = send_command((0, 0, 0, 0))
                reset_pcmd_controller()
                took_over = bool(hooks.request_manual()) if hooks.request_manual else False
                rec["stick_override"] = True
                emit(rec, actual, True,
                     "stick override -> hover + manual" if took_over
                     else "stick override -> hover (no manual pilot; staying suspended)")
                print("[safety] STICK_OVERRIDE: pilot moved the sticks -> "
                      "zero PCMD, autonomy suspended", flush=True)
                if not took_over:
                    # No SkyController pilot configured: MANUAL downgrades to HOVER,
                    # so autonomy must not simply carry on as if nothing happened.
                    reason = "stick override without a manual pilot -> land"
                    break
                steps += 1
                dt = hooks.now() - t0
                if dt < period:
                    time.sleep(period - dt)
                continue

        now = hooks.now()
        if hooks.stream_healthy is not None and not hooks.stream_healthy():
            reset_pcmd_controller()
            _sent, _why, actual = send_command((0, 0, 0, 0))
            stream_lost_since = stream_lost_since or now
            uncertain_since = None
            uncertain_land_after = None
            rec["stream_ok"] = False
            if now - stream_lost_since >= STREAM_LOST_LAND_S:
                reason = (f"stream lost {now - stream_lost_since:.1f}s "
                          f">= {STREAM_LOST_LAND_S:.0f}s -> land")
                emit(rec, actual, True, reason)
                break
            emit(rec, actual, True,
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
        # Inference is synchronous and can take long enough for the operator to
        # switch mode or for video to disappear. Re-authorize after inference,
        # before any pose-derived command is allowed to reach the drone.
        final_safety = hooks.safety_poll() if hooks.safety_poll is not None else "AUTO"
        rec["safety_final"] = final_safety
        if final_safety == "EMERGENCY":
            reason = "EMERGENCY command during inference -> motor cut"
            emit(rec, None, True, reason)
            break
        if final_safety == "LAND":
            _sent, _why, actual = send_command((0, 0, 0, 0))
            reason = "safety LAND command during inference -> land"
            emit(rec, actual, True, reason)
            break
        if final_safety == "HOVER":
            reset_pcmd_controller()
            _sent, _why, actual = send_command((0, 0, 0, 0))
            emit(rec, actual, True,
                 "safety changed to HOVER during inference -> zero PCMD")
            steps += 1
            continue
        if final_safety == "MANUAL":
            reset_pcmd_controller()
            emit(rec, None, True,
                 "safety changed to MANUAL during inference -> autonomy sends nothing")
            steps += 1
            continue
        try:
            stream_ok_after_inference = (hooks.stream_healthy is None
                                         or bool(hooks.stream_healthy()))
        except Exception:
            stream_ok_after_inference = False
        if not stream_ok_after_inference:
            reset_pcmd_controller()
            _sent, _why, actual = send_command((0, 0, 0, 0))
            stream_lost_since = stream_lost_since or now
            uncertain_since = None
            uncertain_land_after = None
            rec["stream_ok"] = False
            emit(rec, actual, True,
                 "stream became stale/lost during inference -> hover")
            steps += 1
            continue
        # A NaN/inf pose would pass every `x > threshold` gate below (all False for
        # NaN) and steer the drone with garbage -> reject non-finite as no fix.
        pose_finite = ploc is not None and all(
            math.isfinite(float(v)) for v in (ploc.x, ploc.y, ploc.z, ploc.yaw, ploc.stamp))
        if ploc is not None and not pose_finite and verbose:
            print("[safety] NON_FINITE_POSE_REJECT: localizer returned NaN/inf -> hover", flush=True)
        if ploc is not None and pose_finite:
            rec["pose"] = [round(float(ploc.x), 3), round(float(ploc.y), 3), round(float(ploc.z), 3)]
            rec["pose_age"] = round(now - float(ploc.stamp), 3)
        elif ploc is not None:
            rec["pose"] = "non-finite"
        fresh = pose_finite and -0.05 <= (now - ploc.stamp) <= POSE_STALE_S
        jump_rejected = False
        if fresh and last_good is not None:
            # Outlier gate: one spurious PnP pose must not steer the drone.
            # A single fix jumping > MAX_POSE_JUMP_U is rejected (hover); if the
            # NEXT fix agrees with it, treat it as genuine relocalization.
            cand = np.array([ploc.x, ploc.y, ploc.z], float)
            prev = np.array([last_good.x, last_good.y, last_good.z], float)
            if float(np.linalg.norm(cand - prev)) > MAX_POSE_JUMP_U:
                # The confirming fix must come from a DIFFERENT capture. Re-localizing
                # the same frame reproduces the same centre, so it would agree with
                # itself and promote one bad frame straight to an accepted relocation.
                independent = (
                    pending_jump is not None
                    and pending_jump_stamp is not None
                    and float(ploc.stamp) > float(pending_jump_stamp)
                )
                if (independent
                        and float(np.linalg.norm(cand - pending_jump)) <= MAX_POSE_JUMP_U):
                    pending_jump = None
                    pending_jump_stamp = None
                    if verbose:
                        print("[safety] POSE_JUMP confirmed by consecutive fix; accepting relocation", flush=True)
                else:
                    pending_jump = cand
                    pending_jump_stamp = float(ploc.stamp)
                    fresh = False
                    jump_rejected = True
                    rec["jump_reject_u"] = round(float(np.linalg.norm(cand - prev)), 2)
                    if verbose:
                        print(f"[safety] POSE_JUMP_REJECT: fix jumped "
                              f"{float(np.linalg.norm(cand - prev)):.2f}u > {MAX_POSE_JUMP_U}u; "
                              "hover, waiting for confirmation", flush=True)
            else:
                pending_jump = None
                pending_jump_stamp = None
        if fresh:
            last_good = ploc
        elif (not jump_rejected) and last_good is not None and (now - last_good.stamp) <= POSE_STALE_S:
            # Missed/stale fix within the freshness window -> keep last accepted pose.
            # NOT on a jump reject: a contradicting fix means high uncertainty; hover
            # this tick (as documented above) instead of driving on the previous pose.
            ploc, fresh = last_good, True

        # Low confidence (WEAK, or PnP inliers below threshold) OR no fresh fix -> HOVER and
        # ask the tracker to relocalize with MegaLoc. If MegaLoc cannot recover within
        # LOST_MANUAL_S, hand to the human pilot. Missing fixes land at
        # LOST_LAND_S; fresh-but-weak fixes use WEAK_HOVER_LAND_S.
        # The drone is never driven on a low-confidence fix.
        low_conf = bool(fresh and (
            (GATE_WEAK and hooks.pose_is_weak is not None and hooks.pose_is_weak())
            or (hooks.pose_confidence is not None and hooks.pose_confidence() < LOW_CONF_INLIERS)))
        if low_conf or not fresh:
            reset_pcmd_controller()
            recovery_good_fixes = 0
            _sent, _why, actual = send_command((0, 0, 0, 0))
            current_limit = WEAK_HOVER_LAND_S if low_conf else LOST_LAND_S
            if uncertain_since is None:
                uncertain_since = now
                uncertain_land_after = current_limit
            else:
                # WEAK and LOST are one uninterrupted uncertainty episode. If
                # they alternate, retain the shortest encountered fail-safe.
                uncertain_land_after = min(float(uncertain_land_after), current_limit)
            waited = now - uncertain_since
            land_after = float(uncertain_land_after)
            if hooks.force_relocalize is not None:
                hooks.force_relocalize()                  # ask the tracker to run MegaLoc
            emit(rec, actual, True,
                 ("low confidence" if low_conf else
                  "pose jump rejected" if "jump_reject_u" in rec else
                  "no fresh pose (lost/stale/PnP fail/non-finite)")
                 + f" -> hover ({waited:.1f}s)")
            if (not manual_requested and waited >= LOST_MANUAL_S
                    and hooks.request_manual is not None and hooks.request_manual()):
                manual_requested = True
                print("[safety] LOW_CONF/LOST: MegaLoc could not relocalize in "
                      f"{waited:.1f}s -> switching to MANUAL (pilot takeover)", flush=True)
            elif waited >= land_after:
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
            if uncertain_since is not None and recovery_good_fixes + 1 < RECOVERY_GOOD_FIXES:
                reset_pcmd_controller()
                recovery_good_fixes += 1
                _sent, _why, actual = send_command((0, 0, 0, 0))
                emit(
                    rec,
                    actual,
                    True,
                    f"localization recovery confirmation {recovery_good_fixes}/"
                    f"{RECOVERY_GOOD_FIXES} -> hover",
                )
                # Do not force another global pass: the next frame must prove
                # the normal TRACK path is also trustworthy before motion resumes.
                steps += 1
                dt = hooks.now() - t0
                if dt < period:
                    time.sleep(period - dt)
                continue
            uncertain_since = None
            uncertain_land_after = None
            recovery_good_fixes = 0
            manual_requested = False
            heading.update(ploc.yaw, oyaw)
            hdg = heading.heading(oyaw)
            if hdg is None:
                reset_pcmd_controller()
                _sent, _why, actual = send_command((0, 0, 0, 0))
                emit(rec, actual, True, "heading unavailable -> hover")
            else:
                pose = rpf.Pose(x=ploc.x, y=ploc.y, z=ploc.z, yaw=hdg, stamp=ploc.stamp)
                cmd = ctrl.step(pose, now)
                if cmd.look_at_pole is not None:
                    reset_pcmd_controller()
                    roll, pitch, yaw, gaz = rpf.command_to_body_percent(
                        cmd,
                        pose,
                        config=ctrl.cfg,
                        yaw_sign=yaw_sign,
                    )
                else:
                    roll, pitch, yaw, gaz = pcmd_controller.update(
                        cmd,
                        pose,
                        now,
                        # Alignment belongs to the clicked target waypoint, not
                        # merely the geometric segment number. This guarantees a
                        # fresh hover/turn gate for the first leg and every target.
                        target_key=ctrl.target_index,
                        yaw_sign=yaw_sign,
                    )
                rec["heading_deg"] = round(math.degrees(hdg), 1)
                rec["path_error_u"] = round(float(cmd.path_error), 3)
                rec["progress"] = round(float(cmd.progress), 4)
                rec["action"] = cmd.status
                rec["pcmd_phase"] = pcmd_controller.phase
                if pcmd_controller.abort_reason is not None:
                    _sent, _why, actual = send_command((0, 0, 0, 0))
                    reason = f"{pcmd_controller.abort_reason} -> land"
                    emit(rec, actual, True, reason)
                    break
                if cmd.path_error > MAX_ROUTE_DEVIATION_U:
                    # Route-corridor bound: never REJOIN across long distances.
                    _sent, _why, actual = send_command((0, 0, 0, 0))
                    reason = (f"route deviation {cmd.path_error:.2f}u "
                              f"> {MAX_ROUTE_DEVIATION_U}u -> land")
                    emit(rec, actual, True, reason)
                    break
                if cmd.should_land or ctrl.state in ("LANDING", "DONE"):
                    _sent, _why, actual = send_command((0, 0, 0, 0))
                    if cmd.action == "LAND":
                        try:
                            speed_sample = (
                                hooks.ground_speed() if hooks.ground_speed is not None else None
                            )
                        except Exception:
                            speed_sample = None
                        speed_ok, speed_reason, speed_mps = landing_speed_allows_land(
                            speed_sample, now
                        )
                        rec["ground_speed_mps"] = speed_mps
                        if not speed_ok:
                            reset_pcmd_controller()
                            emit(
                                rec,
                                actual,
                                True,
                                f"route complete; {speed_reason} -> hover",
                            )
                            steps += 1
                            dt = hooks.now() - t0
                            if dt < period:
                                time.sleep(period - dt)
                            continue
                    reason = ("pending inspection abort -> land" if cmd.action == "ABORT"
                              else "route complete -> land")
                    emit(rec, actual, True, reason)
                    break
                oldest_inspection = min(inspection_started.values(), default=None)
                if (cmd.look_at_pole is None and oldest_inspection is not None
                        and now - oldest_inspection >= INSPECTION_TIMEOUT_S):
                    _sent, _why, actual = send_command((0, 0, 0, 0))
                    reason = "inspection alignment/capture timeout -> land"
                    emit(rec, actual, True, reason)
                    break
                if cmd.look_at_pole is not None:
                    key = (cmd.look_at_pole.get("waypoint"), cmd.look_at_pole.get("pole_id"))
                    inspection_since = inspection_started.setdefault(key, now)
                    hold_ok, hold_reason, hold_actual = send_command((0, 0, 0, 0))
                    if not hold_ok:
                        emit(rec, hold_actual, True,
                             f"inspection pre-action hold blocked ({hold_reason})")
                        upper_reason = str(hold_reason).upper()
                        if "EMERGENCY" in upper_reason:
                            reason = "EMERGENCY command during inspection -> motor cut"
                            break
                        if "LAND" in upper_reason or "STOP SIGNAL" in upper_reason or "TERMINATED" in upper_reason:
                            reason = f"{hold_reason} -> land"
                            break
                    elif not cmd.look_at_pole.get("body_yaw_required", False):
                        reason = "inspection body-yaw orientation unavailable -> land"
                        emit(rec, hold_actual, True, reason)
                        break
                    elif now - inspection_since >= INSPECTION_TIMEOUT_S:
                        reason = "inspection alignment/capture timeout -> land"
                        emit(rec, hold_actual, True, reason)
                        break
                    else:
                        yaw_error = _wrap(float(cmd.yaw_target) - float(pose.yaw))
                        yaw_tol = math.radians(float(ctrl.cfg.inspect_yaw_tolerance_deg))
                        if abs(yaw_error) > yaw_tol:
                            yaw_only = (0, 0, yaw, 0)
                            sent, send_reason, actual = send_command(yaw_only)
                            emit(rec, actual, not sent,
                                 (f"inspection body-yaw align error={math.degrees(yaw_error):.1f}deg"
                                  if sent else send_reason))
                        else:
                            capture_acked = False
                            try:
                                if hooks.inspection_ack is not None:
                                    action = lambda: hooks.inspection_ack(cmd.look_at_pole, pose)
                                    capture_acked = bool(
                                        hooks.inspection_hold(action)
                                        if hooks.inspection_hold is not None else action())
                            except Exception as exc:
                                if verbose:
                                    print(f"[inspection] gimbal/capture hook failed ({exc!r})", flush=True)
                            post_ok, post_reason, post_actual = send_command((0, 0, 0, 0))
                            acked = bool(capture_acked and post_ok
                                         and ctrl.ack_inspection(
                                             cmd.look_at_pole, orientation_confirmed=True))
                            if acked:
                                inspection_started.pop(key, None)
                            emit(rec, post_actual, True,
                                 ("inspection target aligned + capture acknowledged" if acked
                                  else f"inspection capture not acknowledged ({post_reason})"))
                else:
                    pcmd = (roll, pitch, yaw, gaz)
                    sent, send_reason, actual = send_command(pcmd)
                    emit(rec, actual, not sent, cmd.status if sent else send_reason)
                    if not sent:
                        upper_reason = str(send_reason).upper()
                        if "EMERGENCY" in upper_reason:
                            reason = "EMERGENCY command before PCMD -> motor cut"
                            break
                        if "LAND" in upper_reason or "STOP SIGNAL" in upper_reason or "TERMINATED" in upper_reason:
                            reason = f"{send_reason} -> land"
                            break
                    elif verbose and steps % 10 == 0:
                        print(f"[{cmd.status:26s}] pos=({ploc.x:5.1f},{ploc.y:5.1f},{ploc.z:4.1f}) "
                              f"hdg={math.degrees(hdg):6.1f} err={cmd.path_error:4.2f} "
                              f"prog={cmd.progress:4.2f} phase={pcmd_controller.phase} "
                              f"PCMD(r={roll:+d},p={pitch:+d},y={yaw:+d},g={gaz:+d})")
        steps += 1
        dt = hooks.now() - t0
        if dt < period:
            time.sleep(period - dt)
    return reason


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
    config = rpf.config_for_route(
        PATH_JSON,
        rpf.ControlConfig(inspect_waypoints=(), map_frame=map_frame),
    )
    return rpf.RouteAutoController(waypoints, poles=[], config=config), waypoints


# ---------------------------------------------------------------------------
# Real flight

def fly(ip: str, yaw_sign: int, gimbal_pitch: float, controller: str, safety_file: str,
        cmd_log_path: str = "", max_altitude_m: float | None = None,
        max_distance_m: float | None = None, distance_geofence: bool = False):
    stop = {"f": False}
    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, lambda *_: stop.__setitem__("f", True))
    ctrl, wp = build_controller()
    import olympe_frame_source as ofs
    from olympe.messages.ardrone3.Piloting import TakeOff, PCMD, Landing, Emergency
    from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

    clog = CommandLog(cmd_log_path or default_cmd_log_path("fly"), sink="olympe")
    print(f"[fly] structured command log: {clog.path}", flush=True)
    safety = SafetySwitch(
        safety_file,
        allow_manual=manual_override_available(ip, controller),
        keyboard=True,
        require_fresh_auto=True,
    )
    print(f"[fly] waiting <= {AUTO_CONSENT_TIMEOUT_S:.0f}s for a fresh AUTO command "
          "written after this run started", flush=True)
    consent_deadline = time.monotonic() + AUTO_CONSENT_TIMEOUT_S
    while True:
        preflight_safety = safety.poll()
        if preflight_safety == "AUTO":
            break
        if preflight_safety in {"LAND", "EMERGENCY"} or stop["f"] \
                or time.monotonic() >= consent_deadline:
            why = ("operator stop" if stop["f"] else
                   "fresh AUTO consent timeout" if time.monotonic() >= consent_deadline else
                   f"preflight safety mode {preflight_safety}")
            clog.event(event="terminal", reason=why)
            clog.close()
            raise SystemExit(f"[fly] {why}; refusing to connect/take off")
        time.sleep(0.1)

    # Bind everything the finally touches BEFORE the try, so a failure at connect,
    # stream start, model load, or takeoff can never raise UnboundLocalError in the
    # finally and strand an airborne drone with an abandoned connection.
    drone = None
    grab = None
    monitor = None
    stick_monitor = None
    must_land = False
    reason = ""
    via_skycontroller = False
    send_pcmd = lambda *_: None
    try:
        drone = ofs.connect(ip, controller=controller)
        via_skycontroller = manual_override_available(ip, controller)
        if via_skycontroller:
            # Keep the physical pilot authoritative throughout ground setup.
            set_piloting_source(drone, "SkyController")
        require_landed_for_firmware_config(drone)
        try:
            preflight = configure_flight_preflight(
                drone, max_altitude_m, max_distance_m, distance_geofence,
                check_landed=False)
        except Exception as exc:
            clog.event(
                event="firmware_preflight", confirmed=False, warning=str(exc),
                max_altitude_m=max_altitude_m, max_distance_m=max_distance_m,
                distance_geofence=bool(distance_geofence),
            )
            print(
                f"[fly] WARNING: {exc}; continuing without confirmed firmware limits",
                flush=True,
            )
        else:
            clog.event(event="firmware_preflight", confirmed=True, **preflight)
            print(
                "[fly] firmware preflight confirmed: "
                f"battery={preflight['battery_percent']}% "
                f"max_altitude={preflight['max_altitude_m']:.1f}m "
                f"max_distance={preflight['max_distance_m']:.1f}m "
                f"distance_geofence={preflight['distance_geofence']}",
                flush=True,
            )
        # True flight rejects callback-receipt timestamps: without source NTP,
        # upstream queue latency cannot be distinguished from a fresh frame.
        grab = ofs.OlympePdrawGrabber(
            drone, require_source_timestamps=True).start()
        loc = build_localizer(grab)
        loc.ensure_models()                          # load MegaLoc/XFeat NOW (on the ground)
        ground_speed = OlympeGroundSpeedTracker(drone)

        def send_pcmd(roll, pitch, yaw, gaz):
            # Single wire to the drone: clamp to the PCMD percent domain [-100, 100].
            r, p, y, g = (max(-100, min(100, int(v))) for v in (roll, pitch, yaw, gaz))
            drone(PCMD(1, r, p, y, g, 0))

        # Independent safety authority + stall watchdog: polls the safety switch and
        # can issue LAND/EMERGENCY/hover from its own thread even while the main loop
        # is blocked in synchronous get_pose() GPU inference.
        monitor = SafetyMonitor(
            send_pcmd, safety,
            # Confirm that Landing was accepted; the final cleanup below waits
            # separately for the terminal landed state.
            land_cb=lambda: drone(Landing()),
            emergency_cb=lambda: drone(Emergency()),
            stop_requested=lambda: stop["f"],
            piloting_source_cb=(
                (lambda source: set_piloting_source(drone, source))
                if via_skycontroller else None),
            max_translation_pcmd=ctrl.cfg.max_translation_pcmd,
            max_yaw_pcmd=ctrl.cfg.max_yaw_pcmd,
        )
        if via_skycontroller:
            # Only after landed preflight, heavy model loading, and the physical
            # stick monitor are ready may the PC acquire authority for TakeOff.
            stick_monitor = start_skycontroller_stick_override(drone, monitor)
        monitor.start()

        # user's flow: TAKEOFF -> hover -> gimbal -> BOOT_INIT MegaLoc lock -> START AUTO.
        # We lock WHILE HOVERING (camera at flight height/view, like the map refs), not
        # on the ground. If it never locks, we hover then land -- we never blind-fly AUTO.
        # Set before sending TakeOff: a timeout/exception can occur after the aircraft
        # has physically lifted but before FlyingStateChanged confirms hovering.
        must_land = True
        takeoff_expectation, arm_reason = monitor.schedule_authorized_takeoff(
            lambda: drone(
                TakeOff() >> FlyingStateChanged(state="hovering", _timeout=12)),
            grab.is_healthy,
            lambda: stop["f"],
        )
        if takeoff_expectation is None:
            raise SystemExit(f"[fly] arming denied: {arm_reason}")
        # Wait outside monitor._io_lock so a post-schedule LAND/EMERGENCY can run.
        wait_success(takeoff_expectation, "takeoff/hovering")
        if not set_gimbal(drone, gimbal_pitch):
            raise SystemExit("[fly] gimbal did not acknowledge target -> landing (no AUTO)")
        print(f"[fly] BOOT_INIT: require {BOOT_LOCK_FIXES} stable fixes near route start "
              f"(<= {BOOT_START_MAX_U:.2f}u, timeout {FIRST_FIX_TIMEOUT_S:.0f}s)...")

        def _boot_pose():
            try:
                return loc.get_pose()
            except Exception as exc:
                print(f"[fly] BOOT_INIT localizer raised ({exc!r}); retrying", flush=True)
                return None

        boot_lock = BootPoseLock(wp[0])
        boot_locked = False
        t0 = time.monotonic()
        while not boot_locked and not stop["f"] and not monitor.terminated.is_set():
            boot_ok, _boot_reason = monitor.arming_allowed(grab.is_healthy, lambda: stop["f"])
            if boot_ok:
                boot_locked = boot_lock.observe(_boot_pose(), now=time.monotonic())
            else:
                boot_lock.reset()
            monitor.send_authorized(
                (0, 0, 0, 0), grab.is_healthy, lambda: stop["f"])
            if time.monotonic() - t0 > FIRST_FIX_TIMEOUT_S:
                raise SystemExit(
                    "[fly] BOOT lock failed (need consecutive fresh fixes near route start) "
                    "-> landing (no AUTO)")
            time.sleep(0.1)
        if monitor.terminated.is_set():
            reason = monitor.reason or "safety command before AUTO -> land"
        elif stop["f"]:
            reason = "operator stop signal before AUTO -> land"
        else:
            auto_ok, auto_reason = monitor.arming_allowed(grab.is_healthy, lambda: stop["f"])
            if not auto_ok:
                raise SystemExit(f"[fly] final AUTO authorization failed: {auto_reason}")
            print(f"[fly] locked (mode={dict(loc.last_info).get('next_mode')}, "
                  f"inliers={dict(loc.last_info).get('inliers')}). START AUTO.")

            def _inspection_ack(meta, pose):
                pitch_target = inspection_gimbal_pitch_deg(
                    meta, pose, ctrl.cfg.map_frame
                )
                if pitch_target is None:
                    return False
                if not set_gimbal(
                        drone, pitch_target, require_confirmation=True,
                        timeout_s=2.0, tolerance_deg=3.0):
                    return False
                baseline_source_us = grab.latest_source_ntp_us()
                if baseline_source_us is None:
                    return False
                # Raw NTP is a source sequence clock, not a host-synchronized
                # capture time. Drain both source time and host receipt time so
                # a pre-confirmation frame arriving late cannot be acknowledged.
                receipt_after = time.monotonic() + INSPECTION_PIPELINE_DRAIN_S
                sample = None
                frame_deadline = receipt_after + STREAM_STALE_S
                while time.monotonic() < frame_deadline:
                    candidate = grab.inspection_sample_after(
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
                out_dir = SYSTEM_ROOT / "outputs" / "flight_inspections"
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / (f"wp{int(meta['waypoint']):02d}_pole{int(meta['pole_id']):02d}_"
                                 f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}.jpg")
                ok = bool(cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
                if ok:
                    print(f"[inspection] body aligned; gimbal pitch={pitch_target:.1f}deg; "
                          f"captured {out}", flush=True)
                return ok

            hooks = LoopHooks(
                get_pose=loc.get_pose,
                pose_is_weak=lambda: bool(dict(loc.last_info).get("weak", False)),
                pose_confidence=lambda: int(dict(loc.last_info).get("inliers", 0) or 0),
                force_relocalize=lambda: setattr(loc.state, "mode", "LOST"),  # run MegaLoc next deep frame
                request_manual=(lambda: safety.force("manual")) if safety.allow_manual else (lambda: False),
                stick_active=getattr(stick_monitor, "is_active", None),
                olympe_yaw=lambda: olympe_yaw_of(drone),
                # Once the monitor has commanded LAND/EMERGENCY, a loop resuming from a
                # blocked get_pose() must NOT send a countermanding nonzero PCMD.
                send_pcmd=lambda r, p, y, g: (
                    None if (stop["f"] or monitor.terminated.is_set()) else send_pcmd(r, p, y, g)),
                send_authorized_pcmd=lambda pcmd: monitor.send_authorized(
                    pcmd, grab.is_healthy, lambda: stop["f"]),
                safety_poll=lambda: monitor.mode,       # the monitor thread is the sole poller
                stream_healthy=grab.is_healthy,
                stream_status=lambda: (lambda age: (
                    "no 720p frame yet" if age is None
                    else f"last 720p frame age={age:.2f}s source={grab.stamp_source}"
                ))(grab.last_frame_age()),
                loop_beat=monitor.beat,
                pose_info=lambda: dict(loc.last_info),
                inspection_ack=_inspection_ack,
                inspection_hold=lambda action: monitor.run_while_holding_zero(
                    action, grab.is_healthy, lambda: stop["f"]),
                pcmd_timing=monitor.pcmd_timing_snapshot,
                ground_speed=ground_speed.sample,
                log_tick=clog,
            )
            # run_loop returns when route done / lost / (Ctrl-C flips stop -> next hover then we break)
            reason = _run_until(hooks, ctrl, wp, yaw_sign, stop)
            if monitor.terminated.is_set():
                reason = monitor.reason or reason
        print(f"[fly] {reason}")
    finally:
        stop_skycontroller_stick_override(stick_monitor)
        if monitor is not None:
            monitor.stop()
            if monitor.reason:
                reason = monitor.reason
            # Even the terminal zero goes through the same authority lock. A
            # late MANUAL/EMERGENCY command therefore cannot receive a PCMD.
            if must_land and not reason.startswith("EMERGENCY"):
                _sent, final_auth_reason, _actual = monitor.send_authorized(
                    (0, 0, 0, 0),
                    lambda: grab is not None and grab.is_healthy(),
                    lambda: stop["f"],
                )
                if "EMERGENCY" in str(final_auth_reason).upper():
                    reason = "EMERGENCY command during cleanup -> motor cut"
        if must_land and drone is not None:
            if reason.startswith("EMERGENCY"):
                print("[fly] EMERGENCY: cutting motors (drone will drop)", flush=True)
                if monitor is None or not monitor.emergency_issued:
                    try:
                        _await_confirmed_action(drone(Emergency()), "emergency")
                    except Exception as exc:
                        print(f"[fly] Emergency() failed ({exc}); falling back to Landing()", flush=True)
                        try:
                            wait_success(
                                drone(Landing() >> FlyingStateChanged(
                                    state="landed", _timeout=20)),
                                "emergency fallback landing/landed",
                            )
                        except (Exception, SystemExit) as exc2:
                            print(f"[fly] warning: Landing() also failed: {exc2}", flush=True)
            else:
                print("[fly] landing")
                try:
                    # wait_success raises SystemExit (a BaseException) on a landing-
                    # confirmation timeout; catch it too, or grab.stop()/disconnect leak.
                    wait_success(
                        drone(Landing() >> FlyingStateChanged(
                            state="landed", _timeout=20)),
                        "landing/landed",
                    )
                except (Exception, SystemExit) as exc:
                    print(f"[fly] warning: landing did not confirm: {exc}", flush=True)
        elif drone is not None:
            print("[fly] no TakeOff attempt; closing stream/connection without Landing()", flush=True)
        else:
            print("[fly] never connected; nothing to land", flush=True)
        if grab is not None:
            try:
                grab.stop()
            except Exception:
                pass
        if drone is not None and via_skycontroller:
            try:
                set_piloting_source(drone, "SkyController")
            except Exception as exc:
                print(f"[fly] warning: could not restore SkyController ownership: {exc}",
                      flush=True)
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
        return run_loop(hooks, ctrl, wp, yaw_sign=yaw_sign, verbose=False)
    except KeyboardInterrupt:
        return "step cap"


# ---------------------------------------------------------------------------
# Self-test: heading fusion + PCMD sign sanity (pure python, no deps)

def selftest():
    import real_path_follow_controller as rpf

    # 1) visual map heading anchors converted Olympe NED yaw exactly.
    he = HeadingEstimator()
    oyaw = math.radians(110.0)
    true_map_heading = math.radians(40.0)
    he.update(true_map_heading, oyaw)
    est = he.heading(oyaw)
    assert abs(_wrap(est - true_map_heading)) < 1e-9, \
        f"heading fusion off: {math.degrees(est):.1f} vs {math.degrees(true_map_heading):.1f}"

    # 2) PCMD sign sanity: facing the goal -> forward pitch>0, small yaw; behind -> yaw drives, no forward
    pose = rpf.Pose(x=0, y=0, z=0, yaw=0.0)      # facing +X
    ahead = rpf.Command("FOLLOW", np.array([1.2, 0.0, 0.0]), yaw_target=0.0,
                        goal=np.array([1.2, 0.0, 0.0]), path_error=0.0, progress=0.0)
    r, p, y, g = rpf.command_to_body_percent(ahead, pose)
    assert p > 0 and abs(y) < 3 and r == 0, (r, p, y, g)
    behind = rpf.Command("FOLLOW", np.array([-1.2, 0.0, 0.0]), yaw_target=math.pi,
                         goal=np.array([-1.2, 0.0, 0.0]), path_error=0.0, progress=0.0)
    r, p, y, g = rpf.command_to_body_percent(behind, pose)
    assert p == 0 and abs(y) == 20, ("should spin in place, not fly backward", r, p, y, g)

    # 3) +gaz == up (-Y): a "go up" command (target y lower) yields gaz>0
    up = rpf.Command("FOLLOW", np.array([0.0, -1.0, 0.0]), yaw_target=0.0,
                     goal=np.array([0.0, -1.0, 0.0]), path_error=0.0, progress=0.0)
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
    mode.add_argument("--fly", action="store_true", help="approved autonomous route flight")
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
        print("[mode] FLY: approved route with runtime safety authority", flush=True)
        fly(args.ip, args.yaw_sign, args.gimbal_pitch, args.controller, args.safety_file,
            cmd_log_path=args.cmd_log,
            max_altitude_m=args.max_altitude_m,
            max_distance_m=args.max_distance_m,
            distance_geofence=args.distance_geofence)


if __name__ == "__main__":
    main()

"""Production-route Sphinx adapter: attach to a hovering Sphinx and fly AUTO.

Thin adapter only: builds the exact production stack (RouteDocument +
measured MapFrame + config_for_route + production_auto_control_config +
apply_desktop_auto_authority) and drives path_follow_flight.run_loop through
LoopHooks. Owns no controller math: PCMD comes from YawAlignedPcmdController
via run_loop, speed limiting from DesktopRouteAutonomy._apply_speed_limit via
a shared-fake-backend shim, and synthetic localization from
simulated_localization (the same capture/receive contract as the fast sim).
Sphinx ENU metres are rotated into the measured map basis and divided by
meters_per_unit. Independent firmware fused attitude bridges heading during
visual loss. ANAFI4k Sphinx 2.25.2 exposes physical wind only on the world
component. Gusts update that native ENU wind vector from the AUTO pose poll;
no unsupported drone ExprTk component or position displacement is used.
No TakeOff: the operator hovers first; --attach-hovering is required.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import http.client
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .safety import SPHINX_DRONE_IP, require_sphinx_target
from .telemetry import TrueTelemetryCollector

PRODUCTION_SUBCOMMAND = "production-route"

LOCALIZATION_SOURCE = "synthetic_pose_not_EDM"
POSITION_SOURCE = "omniscient_anafi.worldPosition_absolute_Gazebo_ENU"
YAW_SOURCE = "perturbed_firmware_attitude"
WIND_DOC = "https://developer.parrot.com/docs/sphinx/wind.html"

PRODUCTION_ENVELOPE = {
    "max_tilt_deg": 20.0,
    "max_vertical_speed_mps": 2.0,
    "max_yaw_rate_deg_s": 20.0,
    "max_pitch_roll_rate_deg_s": 60.0,
}

FIRMWARE_SHA = "fcbc8450911e7533763479bed672fd1c4b3a3395c073171ad2d1ce568a15fede"


@dataclass
class ProductionRouteArgs:
    route: Path
    align: Path | None
    meters_per_unit: float
    output_dir: Path
    dry_run: bool
    attach_hovering: bool
    seed: int
    duration_s: float
    wind_mean_mps: tuple[float, float, float]
    gust_mps: float
    gust_duration_s: float
    gust_transition_s: float
    gust_pause_s: float
    pose_rate_hz: float = 20.0
    latency_ms: float = 120.0
    latency_jitter_ms: float = 0.0
    position_correlation_s: float = 0.0
    yaw_correlation_s: float = 0.0
    bias_walk_u_sqrt_s: float = 0.0
    reseed_confirm_frames: int = 0
    pos_noise_u: float = 0.004
    yaw_noise_deg: float = 1.0
    drop_rate: float = 0.02
    blackout_every_s: float = 0.0
    blackout_dur_s: float = 0.0


def add_production_route_parser(subparsers) -> Any:
    parser = subparsers.add_parser(
        PRODUCTION_SUBCOMMAND,
        help="fly the production AUTO controller against an operator hovering Sphinx",
    )
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--align", type=Path, default=None)
    parser.add_argument("--meters-per-unit", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--attach-hovering", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument(
        "--wind-mean-mps", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("E", "N", "U")
    )
    parser.add_argument("--gust-mps", type=float, nargs="?", const=1.0, default=0.0)
    parser.add_argument("--gust-duration-s", type=float, default=2.0)
    parser.add_argument("--gust-transition-s", type=float, default=1.0)
    parser.add_argument("--gust-pause-s", type=float, default=15.0)
    parser.add_argument("--pose-rate-hz", type=float, default=20.0)
    parser.add_argument("--latency-ms", type=float, default=120.0)
    parser.add_argument("--latency-jitter-ms", type=float, default=0.0)
    parser.add_argument("--position-correlation-s", type=float, default=0.0)
    parser.add_argument("--yaw-correlation-s", type=float, default=0.0)
    parser.add_argument("--bias-walk-u-sqrt-s", type=float, default=0.0)
    parser.add_argument("--reseed-confirm-frames", type=int, default=0)
    parser.add_argument("--pos-noise-u", type=float, default=0.004)
    parser.add_argument("--yaw-noise-deg", type=float, default=1.0)
    parser.add_argument("--drop-rate", type=float, default=0.02)
    parser.add_argument(
        "--blackout-every-s",
        type=float,
        default=0.0,
        help="repeat visual loss every N simulation seconds (0 disables)",
    )
    parser.add_argument(
        "--blackout-dur-s",
        type=float,
        default=0.0,
        help="visual-loss duration; fresh fused attitude holds center and bridges heading",
    )
    return parser


def args_from_namespace(args: argparse.Namespace) -> ProductionRouteArgs:
    gust = args.gust_mps
    if isinstance(gust, (list, tuple)):
        gust = float(gust[0]) if gust else 0.0
    return ProductionRouteArgs(
        route=Path(args.route),
        align=Path(args.align) if getattr(args, "align", None) else None,
        meters_per_unit=float(args.meters_per_unit),
        output_dir=Path(args.output_dir),
        dry_run=bool(getattr(args, "dry_run", False)),
        attach_hovering=bool(getattr(args, "attach_hovering", False)),
        seed=int(getattr(args, "seed", 7)),
        duration_s=float(getattr(args, "duration_s", 300.0)),
        wind_mean_mps=tuple(float(v) for v in getattr(args, "wind_mean_mps", (0.0, 0.0, 0.0))),
        gust_mps=float(gust or 0.0),
        gust_duration_s=float(getattr(args, "gust_duration_s", 2.0)),
        gust_transition_s=float(getattr(args, "gust_transition_s", 1.0)),
        gust_pause_s=float(getattr(args, "gust_pause_s", 15.0)),
        pose_rate_hz=float(getattr(args, "pose_rate_hz", 20.0)),
        latency_ms=float(getattr(args, "latency_ms", 120.0)),
        latency_jitter_ms=float(getattr(args, "latency_jitter_ms", 0.0)),
        position_correlation_s=float(getattr(args, "position_correlation_s", 0.0)),
        yaw_correlation_s=float(getattr(args, "yaw_correlation_s", 0.0)),
        bias_walk_u_sqrt_s=float(getattr(args, "bias_walk_u_sqrt_s", 0.0)),
        reseed_confirm_frames=int(getattr(args, "reseed_confirm_frames", 0)),
        pos_noise_u=float(getattr(args, "pos_noise_u", 0.004)),
        yaw_noise_deg=float(getattr(args, "yaw_noise_deg", 1.0)),
        drop_rate=float(getattr(args, "drop_rate", 0.02)),
        blackout_every_s=float(getattr(args, "blackout_every_s", 0.0)),
        blackout_dur_s=float(getattr(args, "blackout_dur_s", 0.0)),
    )


def _flight_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "定位演算法" / "flight_control"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("flight_control directory not found")


def _ensure_flight_imports():
    root = _flight_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    operator_root = root.parent.parent / "控制介面程式" / "operator_interface"
    if operator_root.is_dir() and str(operator_root) not in sys.path:
        sys.path.insert(0, str(operator_root))


def build_production_stack(route: Path, align: Path | None):
    _ensure_flight_imports()
    import real_path_follow_controller as rpf
    from route_domain import RouteDocument

    frame = rpf.load_map_frame(align) if align is not None else rpf.LEGACY_MAP_FRAME
    doc = RouteDocument.from_path(route, require_map_units=True, map_frame=frame)
    base = rpf.production_auto_control_config(frame)
    cfg = rpf.apply_desktop_auto_authority(rpf.config_for_route(route, base))
    waypoints = doc.controller_waypoints()
    ctrl = rpf.RouteAutoController([np.asarray(p, float) for p in waypoints], poles=[], config=cfg)
    return rpf, doc, frame, cfg, ctrl


def _validate_parsed(parsed: ProductionRouteArgs) -> None:
    scale = float(parsed.meters_per_unit)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("meters-per-unit must be finite and > 0")
    duration = float(parsed.duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration-s must be finite and > 0")
    for name in ("pose_rate_hz",):
        value = float(getattr(parsed, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    for name in (
        "latency_ms",
        "latency_jitter_ms",
        "position_correlation_s",
        "yaw_correlation_s",
        "bias_walk_u_sqrt_s",
        "pos_noise_u",
        "yaw_noise_deg",
        "drop_rate",
        "blackout_every_s",
        "blackout_dur_s",
        "gust_mps",
        "gust_duration_s",
        "gust_transition_s",
        "gust_pause_s",
    ):
        value = float(getattr(parsed, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    if parsed.gust_mps > 0 and parsed.gust_duration_s + 2 * parsed.gust_transition_s <= 0:
        raise ValueError("enabled gust must have a positive duration or transition")
    if not 0.0 <= float(parsed.drop_rate) <= 1.0:
        raise ValueError("drop_rate must be in [0, 1]")
    frames = int(parsed.reseed_confirm_frames)
    if frames < 0:
        raise ValueError("reseed_confirm_frames must be >= 0")
    for value in tuple(parsed.wind_mean_mps):
        if not math.isfinite(float(value)):
            raise ValueError("wind-mean-mps must be finite")


def _route_enu_metres(parsed: ProductionRouteArgs, doc, frame) -> list[tuple[float, float, float]]:
    scale = float(parsed.meters_per_unit)
    east = np.asarray(frame.east, dtype=float)
    north = np.asarray(frame.north, dtype=float)
    up = np.asarray(frame.up, dtype=float)
    points = []
    for raw in doc.controller_waypoints():
        point = np.asarray(raw, dtype=float).reshape(3)
        points.append(
            (
                float(np.dot(point, east) * scale),
                float(np.dot(point, north) * scale),
                float(np.dot(point, up) * scale),
            )
        )
    return points


def describe_dry_run(parsed: ProductionRouteArgs) -> dict:
    _validate_parsed(parsed)
    _rpf, doc, frame, cfg, ctrl = build_production_stack(parsed.route, parsed.align)
    route_sha = hashlib.sha256(Path(parsed.route).read_bytes()).hexdigest()
    wind = _wind_to_sphinx_params(tuple(parsed.wind_mean_mps))
    return {
        "command": PRODUCTION_SUBCOMMAND,
        "dry_run": True,
        "route": str(parsed.route),
        "route_sha256": route_sha,
        "waypoints": len(doc.waypoints),
        "expanded_waypoints": len(ctrl.wp),
        "waypoints_enu_m": [
            [round(v, 4) for v in point] for point in _route_enu_metres(parsed, doc, frame)
        ],
        "meters_per_unit": float(parsed.meters_per_unit),
        "seed": int(parsed.seed),
        "duration_s": float(parsed.duration_s),
        "wind_mean_mps": [float(v) for v in parsed.wind_mean_mps],
        "wind_sphinx": wind,
        "gust_mps": float(parsed.gust_mps),
        "gust_sphinx": {
            "mechanism": "host-driven world/wind SetParam; native physical wind",
            "direction_enu": "East, additive to mean vector",
            "duration_s": float(parsed.gust_duration_s),
            "transition_s": float(parsed.gust_transition_s),
            "pause_s": float(parsed.gust_pause_s),
            "updates": "AUTO pose polls, nominally 20 Hz",
        },
        "target_safety_lock": f"{SPHINX_DRONE_IP} only",
        "firmware_sha256": FIRMWARE_SHA,
        "envelope": dict(PRODUCTION_ENVELOPE),
        "localization_source": LOCALIZATION_SOURCE,
        "position_source": POSITION_SOURCE,
        "truth_telemetry": {
            "endpoint": "inet:127.0.0.1:9060",
            "rate_hz": 50,
            "max_age_s": 0.5,
            "origin": "absolute Gazebo world; no GPS or attach-origin subtraction",
        },
        "yaw_source": YAW_SOURCE,
        "wind_api": WIND_DOC,
        "connect": False,
        "pcmd_sent": False,
        "sensor": {
            "pose_rate_hz": float(parsed.pose_rate_hz),
            "latency_ms": float(parsed.latency_ms),
            "latency_jitter_ms": float(parsed.latency_jitter_ms),
            "position_correlation_s": float(parsed.position_correlation_s),
            "yaw_correlation_s": float(parsed.yaw_correlation_s),
            "bias_walk_u_sqrt_s": float(parsed.bias_walk_u_sqrt_s),
            "reseed_confirm_frames": int(parsed.reseed_confirm_frames),
            "pos_noise_u": float(parsed.pos_noise_u),
            "yaw_noise_deg": float(parsed.yaw_noise_deg),
            "drop_rate": float(parsed.drop_rate),
            "blackout_every_s": float(parsed.blackout_every_s),
            "blackout_dur_s": float(parsed.blackout_dur_s),
            "blackout_window": "simulation_time % blackout_every_s < blackout_dur_s",
            "blackout_compensation": "fused attitude only; last visual center held; no map confirmation",
            "imu_sample_max_age_s": 0.5,
            "imu_bridge_max_age_s": 10.0,
            "imu_bridge_max_frames": 300,
        },
        "controller": {
            "max_translation_pcmd": int(cfg.max_translation_pcmd),
            "max_vertical_pcmd": int(cfg.max_vertical_pcmd),
            "max_yaw_pcmd": int(cfg.max_yaw_pcmd),
        },
    }


def _wind_to_sphinx_params(mean_enu: tuple[float, float, float]) -> dict[str, float]:
    east, north, up = map(float, mean_enu)
    horizontal = math.hypot(east, north)
    magnitude = math.hypot(horizontal, up)
    direction = math.degrees(math.atan2(north, east)) % 360.0
    elevation = math.degrees(math.atan2(up, horizontal))
    return {
        "magnitude_mean": float(magnitude),
        "direction_mean": float(direction),
        "elevation_mean": float(elevation),
    }


def _gust_amplitude(parsed: ProductionRouteArgs, elapsed_s: float) -> float:
    """Rise, hold, fall, pause; zero pause means one cycle, like Sphinx gusts."""
    if parsed.gust_mps <= 0:
        return 0.0
    transition = float(parsed.gust_transition_s)
    duration = float(parsed.gust_duration_s)
    active = duration + 2 * transition
    elapsed = max(0.0, float(elapsed_s))
    if parsed.gust_pause_s == 0:
        if elapsed >= active:
            return 0.0
        phase = elapsed
    else:
        phase = elapsed % (active + float(parsed.gust_pause_s))
    if transition > 0 and phase < transition:
        fraction = phase / transition
    elif phase < transition + duration:
        fraction = 1.0
    elif transition > 0 and phase < active:
        fraction = (active - phase) / transition
    else:
        fraction = 0.0
    return float(parsed.gust_mps) * fraction


class _SphinxWind:
    """Native physical world wind; no thread, SDK commands, or truth movement."""

    def __init__(self, parsed: ProductionRouteArgs):
        self.parsed = parsed
        self._connection = http.client.HTTPConnection("127.0.0.1", 8383, timeout=0.5)
        self._last_params: dict[str, float] = {}
        self._started_at: float | None = None
        try:
            self._apply(tuple(parsed.wind_mean_mps))
        except Exception:
            self.close()
            raise

    def _set_param(self, parameter: str, value: float) -> None:
        # Same RPC and string-valued parameters as installed pysphinx.SetParam.
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SetParam",
                "params": {
                    "machine": "world",
                    "object": "wind/wind",
                    "parameter": parameter,
                    "value": repr(float(value)),
                },
            }
        )
        self._connection.request("POST", "/", body, {"Content-Type": "application/json"})
        response = self._connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"Sphinx wind HTTP status {response.status}")
        result = json.loads(payload)
        if (
            not isinstance(result, dict)
            or result.get("result") is not True
            or result.get("error") is not None
        ):
            raise RuntimeError(f"Sphinx rejected wind parameter {parameter}: {result}")

    def _apply(self, enu: tuple[float, float, float]) -> None:
        for parameter, value in _wind_to_sphinx_params(enu).items():
            if self._last_params.get(parameter) == value:
                continue
            self._set_param(parameter, value)
            self._last_params[parameter] = value

    def update(self, now: float) -> None:
        if self._started_at is None:
            self._started_at = float(now)
        gust = _gust_amplitude(self.parsed, float(now) - self._started_at)
        east, north, up = self.parsed.wind_mean_mps
        self._apply((float(east) + gust, float(north), float(up)))

    def close(self) -> None:
        self._connection.close()


def _write_dry_run_artifacts(parsed: ProductionRouteArgs, payload: dict) -> None:
    out = Path(parsed.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "production_route_plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class _SphinxProductionPoseBridge:
    """Sphinx truth -> synthetic map-unit pose for the production controller.

    Converts Sphinx ENU metres to the measured map basis, pushes them through
    SimulatedLocalizer (the same capture/receive contract as the fast sim),
    and exposes the LoopHooks pose/yaw/velocity surface. Visual yaw is perturbed
    firmware heading; blackout increments use unperturbed independent samples.
    """

    def __init__(self, parsed, frame, localizer, *, rng):
        self.parsed = parsed
        self.frame = frame
        self.localizer = localizer
        self.rng = rng
        self.scale = float(parsed.meters_per_unit)
        self.east = np.asarray(frame.east, dtype=float)
        self.north = np.asarray(frame.north, dtype=float)
        self.up = np.asarray(frame.up, dtype=float)
        self.last_truth_m: np.ndarray | None = None
        self.last_truth_yaw: float | None = None
        self.last_truth_stamp: float | None = None
        self._truth_sample_stamp: float | None = None
        self._truth_fault: str | None = None
        self.last_pose = None
        self.last_pose_accepted_at: float | None = None
        self.last_map_stamp: float | None = None
        self.last_attitude_yaw: float | None = None
        self.last_attitude_stamp: float | None = None
        self.yaw_perturb_deg = 0.0
        self.speed_north_mps = 0.0
        self.speed_east_mps = 0.0
        self.speed_stamp: float | None = None
        self._time_origin: float | None = None

    def _simulation_time(self, now: float) -> float:
        if self._time_origin is None:
            self._time_origin = float(now)
        return float(now) - self._time_origin

    def enu_to_map(self, enu_m) -> np.ndarray:
        vec = np.asarray(enu_m, dtype=float).reshape(3)
        return (vec[0] * self.east + vec[1] * self.north + vec[2] * self.up) / self.scale

    def invalidate_truth(self, fault="truth-missing") -> None:
        self._truth_fault = fault
        self.last_pose = None
        self.localizer.map_confirmed = False
        self.localizer.fault = fault

    def truth_is_fresh(self) -> bool:
        return bool(
            self._truth_fault is None
            and self.last_truth_stamp is not None
            and 0.0 <= time.monotonic() - self.last_truth_stamp <= 0.5
        )

    def observe_truth(
        self,
        now: float,
        enu_m,
        attitude_yaw_ned: float | None,
        vel_ned: tuple[float, float] | None = None,
        *,
        attitude_stamp: float | None = None,
        tilt_deg: float = 0.0,
        truth_sample_stamp: float | None = None,
    ) -> None:
        enu = np.asarray(enu_m, dtype=float).reshape(3)
        source_stamp = float(now if truth_sample_stamp is None else truth_sample_stamp)
        if not math.isfinite(source_stamp) or not np.all(np.isfinite(enu)):
            self.invalidate_truth("truth-invalid")
            return
        if self._truth_sample_stamp is not None and source_stamp < self._truth_sample_stamp:
            self.invalidate_truth("truth-out-of-order")
            return
        new_truth = self._truth_sample_stamp is None or source_stamp > self._truth_sample_stamp
        sim_now = self._simulation_time(now)
        if new_truth:
            self.last_truth_m = enu.copy()
            self.last_truth_stamp = float(now)
            self._truth_sample_stamp = source_stamp
            self._truth_fault = None
        # Rotate ENU metres into the measured map basis before the localizer
        # divides by scale. This is the inverse of _route_enu_metres.
        map_m = enu[0] * self.east + enu[1] * self.north + enu[2] * self.up
        yaw_noise = (
            float(self.rng.normal(0.0, float(self.parsed.yaw_noise_deg)))
            if float(self.parsed.yaw_noise_deg) > 0
            else 0.0
        )
        self.yaw_perturb_deg = yaw_noise
        attitude = float(attitude_yaw_ned) if attitude_yaw_ned is not None else float("nan")
        map_attitude = (math.pi / 2.0 - attitude + math.pi) % (2.0 * math.pi) - math.pi
        visual_map_yaw = float(map_attitude + math.radians(yaw_noise))
        if not math.isfinite(visual_map_yaw):
            visual_map_yaw = float(self.last_truth_yaw or 0.0)
        self.last_truth_yaw = visual_map_yaw
        sample_stamp = float(now if attitude_stamp is None else attitude_stamp)
        yaw_rate = 0.0
        if self.last_attitude_stamp is not None and sample_stamp > self.last_attitude_stamp:
            delta = (attitude - self.last_attitude_yaw + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate = math.degrees(delta) / (sample_stamp - self.last_attitude_stamp)
        if self.last_attitude_stamp != sample_stamp or not math.isfinite(attitude):
            self.localizer.push_imu(
                sample_stamp - self._time_origin,
                map_attitude,
                tilt_deg=tilt_deg,
                yaw_rate_deg_s=yaw_rate,
            )
        if math.isfinite(attitude) and math.isfinite(sample_stamp):
            self.last_attitude_yaw = attitude
            self.last_attitude_stamp = sample_stamp
        if vel_ned is not None:
            self.speed_north_mps = float(vel_ned[0])
            self.speed_east_mps = float(vel_ned[1])
            self.speed_stamp = float(now)
        if new_truth:
            self.localizer.push_truth(sim_now, map_m, visual_map_yaw)

    def get_pose(self):
        now = time.monotonic()
        if self._time_origin is None:
            return None
        if not self.truth_is_fresh():
            self.invalidate_truth(self._truth_fault or "truth-stale")
            return None
        pose = self.localizer.report(self._simulation_time(now))
        if pose is None:
            self.last_pose = None
            return None
        fresh = self.last_pose is None or float(pose.stamp) + self._time_origin > float(
            self.last_pose.stamp
        )
        # Keep localizer stamps in simulation time; hooks consume monotonic time.
        if fresh:
            self.last_pose = copy.copy(pose)
            self.last_pose.stamp = float(pose.stamp) + self._time_origin
            self.last_pose_accepted_at = float(now)
            if bool(getattr(pose, "map_confirmed", False)):
                self.last_map_stamp = self.last_pose.stamp
        return self.last_pose

    def olympe_yaw(self):
        return self.last_attitude_yaw

    def attitude_sample(self):
        if self.last_attitude_stamp is None:
            return None
        return self.last_attitude_yaw, self.last_attitude_stamp

    def ned_velocity(self):
        if self.speed_stamp is None:
            return None
        return self.speed_north_mps, self.speed_east_mps, self.speed_stamp

    def pose_is_weak(self) -> bool:
        pose = self.last_pose
        return bool(pose is not None and not bool(getattr(pose, "map_confirmed", True)))

    def pose_is_predicted(self) -> bool:
        pose = self.last_pose
        return bool(pose is not None and not bool(getattr(pose, "position_observed", True)))

    def pose_confidence(self) -> int:
        pose = self.last_pose
        if pose is None:
            return 0
        return 0 if self.pose_is_weak() else 50

    def ground_speed(self):
        if self.speed_stamp is None:
            return None
        speed = math.hypot(float(self.speed_north_mps), float(self.speed_east_mps))
        return float(speed), float(self.speed_stamp)

    def body_velocity(self):
        sample = self.ground_speed()
        yaw = self.olympe_yaw()
        if sample is None or yaw is None:
            return None
        _speed, stamp = sample
        if not 0.0 <= time.monotonic() - float(stamp) <= 0.5:
            return None
        try:
            north = float(self.speed_north_mps)
            east = float(self.speed_east_mps)
        except (TypeError, ValueError, OverflowError):
            return None
        return (
            math.cos(float(yaw)) * north + math.sin(float(yaw)) * east,
            -math.sin(float(yaw)) * north + math.cos(float(yaw)) * east,
            float(stamp),
        )


class _SharedSpeedLimitShim:
    """Reuse DesktopRouteAutonomy._apply_speed_limit without its flight loop."""

    def __init__(self, controller, *, speed_limit_mps: float = 0.6):
        from types import SimpleNamespace

        self.controller = controller
        self.backend = SimpleNamespace(
            state=SimpleNamespace(
                autonomous_speed_limit_enabled=True,
                autonomous_speed_limit_mps=float(speed_limit_mps),
                autonomous_speed_guard_status="SPEED_WAITING",
                att_yaw=0.0,
                attitude_mono_ns=0,
                speed_north_mps=0.0,
                speed_east_mps=0.0,
                ground_speed_mps=0.0,
                ground_speed_mono_ns=0,
            )
        )
        self.stamp = 0.0
        self.now = lambda: self.stamp
        self._speed_sample = None
        self._speed_rate_mps2 = 0.0
        self._accel_command = None

    def _bind(self, source) -> None:
        import operator_autonomy as autonomy

        for name in (
            "_predicted_speed",
            "_body_velocity",
            "_olympe_yaw",
            "_apply_speed_limit",
            "_limit_acceleration_slew",
            "_ground_speed",
        ):
            bound = getattr(autonomy.DesktopRouteAutonomy, name, None)
            if bound is not None:
                try:
                    object.__setattr__(self, name, bound.__get__(self))
                except (AttributeError, TypeError):
                    pass

    def apply(
        self,
        pcmd,
        *,
        yaw_ned: float,
        north_mps: float,
        east_mps: float,
        stamp: float,
        attitude_stamp: float | None,
        velocity_stamp: float | None,
    ):
        self.stamp = float(stamp)
        state = self.backend.state
        state.att_yaw = float(yaw_ned)
        state.attitude_mono_ns = None if attitude_stamp is None else int(attitude_stamp * 1e9)
        state.speed_north_mps = float(north_mps)
        state.speed_east_mps = float(east_mps)
        state.ground_speed_mps = float(math.hypot(float(north_mps), float(east_mps)))
        state.ground_speed_mono_ns = None if velocity_stamp is None else int(velocity_stamp * 1e9)
        self._bind(self)
        try:
            limited = self._apply_speed_limit(tuple(int(v) for v in pcmd))
            return self._limit_acceleration_slew(limited)
        except AttributeError:
            return tuple(int(v) for v in pcmd)


def _run_attach(parsed: ProductionRouteArgs) -> int:
    _ensure_flight_imports()
    _validate_parsed(parsed)
    require_sphinx_target(SPHINX_DRONE_IP)
    try:
        import olympe
        from olympe.messages.ardrone3.Piloting import PCMD, Landing
        from olympe.messages.ardrone3.PilotingState import (
            AttitudeChanged,
            FlyingStateChanged,
            SpeedChanged,
        )
    except ImportError as exc:
        print(f"production-route needs Olympe for attach: {exc}", file=sys.stderr)
        return 2
    _rpf, _doc, frame, _cfg, ctrl = build_production_stack(parsed.route, parsed.align)
    import path_follow_flight as pff
    from simulated_localization import SimulatedLocalizer, SimulatedLocalizerConfig

    rng = np.random.default_rng(int(parsed.seed))
    loc = SimulatedLocalizer(
        SimulatedLocalizerConfig(
            s=float(parsed.meters_per_unit),
            pos_sigma_u=float(parsed.pos_noise_u),
            yaw_sigma_deg=float(parsed.yaw_noise_deg),
            latency_ms=float(parsed.latency_ms),
            drop_rate=float(parsed.drop_rate),
            pose_rate_hz=float(parsed.pose_rate_hz),
            latency_jitter_ms=float(parsed.latency_jitter_ms),
            position_correlation_s=float(parsed.position_correlation_s),
            yaw_correlation_s=float(parsed.yaw_correlation_s),
            bias_walk_u_sqrt_s=float(parsed.bias_walk_u_sqrt_s),
            reseed_confirm_frames=int(parsed.reseed_confirm_frames),
            blackout_every_s=float(parsed.blackout_every_s),
            blackout_dur_s=float(parsed.blackout_dur_s),
        ),
        rng,
    )
    bridge = _SphinxProductionPoseBridge(parsed, frame, loc, rng=rng)
    speed_shim = _SharedSpeedLimitShim(ctrl)
    out = Path(parsed.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    drone = olympe.Drone(SPHINX_DRONE_IP)
    if not drone.connect():
        print(
            "production-route: unable to connect to hovering Sphinx at 10.202.0.1", file=sys.stderr
        )
        return 2
    from olympe.messages.common.CommonState import BatteryStateChanged

    try:
        battery = int(drone.get_state(BatteryStateChanged).get("percent", -1))
    except (AttributeError, TypeError, ValueError):
        battery = -1
    if battery >= 0 and battery < 30:
        print(
            f"production-route: takeoff battery gate blocks attach ({battery}% < 30%)",
            file=sys.stderr,
        )
        drone.disconnect()
        return 2
    try:
        flying = str(drone.get_state(FlyingStateChanged).get("state", ""))
    except (AttributeError, TypeError, ValueError):
        flying = ""
    if "hover" not in flying and "flying" not in flying:
        print(
            f"production-route: Sphinx is not hovering (state={flying}); operator must hover first",
            file=sys.stderr,
        )
        drone.disconnect()
        return 2
    try:
        wind = _SphinxWind(parsed)
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
        drone.disconnect()
        print(f"production-route: wind setup failed: {exc}", file=sys.stderr)
        return 2
    log_path = out / "production_route_cmdlog.jsonl"
    from path_follow_flight import CommandLog

    cmdlog = CommandLog(log_path, sink="sphinx-production-route")
    sent = {"count": 0}
    attitude_received = {"sample": None}
    wind_failure: dict[str, str | None] = {"error": None}

    def on_attitude(event, _scheduler):
        # Timestamp actual firmware events, never refresh cached get_state data.
        attitude_received["sample"] = (time.monotonic(), dict(event.args))

    attitude_subscription = drone.subscribe(on_attitude, AttitudeChanged())
    collector = TrueTelemetryCollector(stderr_path=out / "true_telemetry.stderr.log")

    def send_authorized_pcmd(pcmd):
        limited = speed_shim.apply(
            pcmd,
            yaw_ned=float(bridge.olympe_yaw() or 0.0),
            north_mps=float(bridge.speed_north_mps),
            east_mps=float(bridge.speed_east_mps),
            stamp=time.monotonic(),
            attitude_stamp=bridge.last_attitude_stamp,
            velocity_stamp=bridge.speed_stamp,
        )
        ok = bool(
            drone(PCMD(1, int(limited[0]), int(limited[1]), int(limited[2]), int(limited[3]), 0))
        )
        sent["count"] += 1 if ok else 0
        return ok, "Sphinx dispatch" if ok else "Sphinx dispatch failed", limited if ok else None

    def get_pose():
        now = time.monotonic()
        try:
            wind.update(now)
        except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
            # The formal loop treats getter exceptions as no fix. Latch a
            # boundary failure separately so it exits instead of waiting forever.
            wind_failure["error"] = str(exc)
            return None
        try:
            sample = collector.latest_sample()
        except RuntimeError:
            bridge.invalidate_truth()
            return None
        att_sample = attitude_received["sample"]
        try:
            spd = drone.get_state(SpeedChanged)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            spd = {}
        # Absolute Gazebo world position matches waypoints_enu_m. No GPS
        # estimate, zeroing at attach, or discarded world-height offset.
        enu = (sample.x_m, sample.y_m, sample.z_m)
        attitude_stamp, att = att_sample if att_sample is not None else (now, {})
        try:
            yaw_ned = float(att["yaw"])
            roll, pitch = float(att["roll"]), float(att["pitch"])
            if not all(math.isfinite(value) for value in (yaw_ned, roll, pitch)):
                raise ValueError("non-finite firmware attitude")
            tilt_deg = math.degrees(
                math.acos(max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch))))
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            yaw_ned = float("nan")
            tilt_deg = float("nan")
        try:
            vel = (float(spd["speedX"]), float(spd["speedY"]))
        except (KeyError, TypeError, ValueError):
            vel = None
        bridge.observe_truth(
            now,
            enu,
            yaw_ned,
            vel_ned=vel,
            attitude_stamp=attitude_stamp,
            tilt_deg=tilt_deg,
            truth_sample_stamp=sample.timestamp_s,
        )
        return bridge.get_pose()

    hooks = pff.LoopHooks(
        get_pose=get_pose,
        olympe_yaw=bridge.olympe_yaw,
        send_pcmd=lambda *pcmd: send_authorized_pcmd(pcmd),
        send_authorized_pcmd=send_authorized_pcmd,
        pose_is_weak=bridge.pose_is_weak,
        pose_is_predicted=bridge.pose_is_predicted,
        pose_confidence=bridge.pose_confidence,
        max_weak_pose_age_s=0.5,
        stream_healthy=lambda: True,
        # LAND terminates this virtual run; MANUAL would leave its loop waiting.
        # The existing finally block performs normal Sphinx landing/cleanup.
        safety_poll=lambda: "LAND" if wind_failure["error"] is not None else "AUTO",
        ground_speed=bridge.ground_speed,
        body_velocity=bridge.body_velocity,
        attitude_sample=bridge.attitude_sample,
        ned_velocity=bridge.ned_velocity,
        log_tick=cmdlog,
        land_on_localization_loss=False,
        wait_expired=lambda _reason, _elapsed_s: False,
    )
    waypoints = [np.asarray(p, dtype=float) for p in ctrl.wp]
    ctrl.start_after_nearest_waypoint(np.asarray(waypoints[0], dtype=float))
    try:
        try:
            collector.start()
            collector.wait_for_sample(timeout_s=15)
        except (OSError, RuntimeError, TimeoutError) as exc:
            print(f"production-route: true world telemetry unavailable: {exc}", file=sys.stderr)
            return 2
        reason = pff.run_loop(hooks, ctrl, waypoints, yaw_sign=1, verbose=True)
        if wind_failure["error"] is not None:
            print(f"production-route: wind update failed: {wind_failure['error']}", file=sys.stderr)
            return 2
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
        print(f"production-route: simulation boundary failed: {exc}", file=sys.stderr)
        return 2
    finally:
        wind.close()
        with contextlib.suppress(Exception):
            attitude_subscription.unsubscribe()
        with contextlib.suppress(Exception):
            drone(PCMD(1, 0, 0, 0, 0, 0))
        try:
            drone(Landing(_no_expect=True)).wait(_timeout=20)
        except Exception as exc:  # noqa: BLE001 -- cleanup retains diagnostics.
            print(f"production-route: landing did not confirm ({exc!r})", file=sys.stderr)
        with contextlib.suppress(Exception):
            collector.stop()
        with contextlib.suppress(Exception):
            drone.disconnect()
        with contextlib.suppress(Exception):
            cmdlog.close()
    print(f"production-route finished: {reason} (pcmd_sent={sent['count'] > 0})", flush=True)
    print((out / "production_route_plan.json").as_posix())
    return 0


def run_production_route(parsed: ProductionRouteArgs) -> int:
    require_sphinx_target(SPHINX_DRONE_IP)
    if parsed.dry_run:
        _validate_parsed(parsed)
        payload = describe_dry_run(parsed)
        _write_dry_run_artifacts(parsed, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not parsed.attach_hovering:
        print(
            "production-route refuses to start: pass --attach-hovering with an "
            "operator-established hovering Sphinx; it never takes off itself",
            file=sys.stderr,
        )
        return 2
    try:
        return _run_attach(parsed)
    except ValueError as exc:
        print(f"production-route failed: {exc}", file=sys.stderr)
        return 2

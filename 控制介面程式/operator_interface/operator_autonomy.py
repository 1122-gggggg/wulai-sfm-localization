"""Desktop AUTO coordinator using the production route-control loop.

The coordinator owns no Olympe connection.  It uses the already-connected
operator backend so takeoff, PCMD, manual takeover, and landing all share one
authority path.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Keep this module importable directly by its unit tests as well as through the
# desktop entrypoint, which normally installs the flight-control path first.
_FLIGHT_CONTROL = Path(__file__).resolve().parents[2] / "定位演算法" / "flight_control"
if str(_FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_CONTROL))

import path_follow_flight as pff
from operator_autonomy_status import leg_stage
import real_path_follow_controller as rpf
from landing_transition import GROUND_SPEED_MAX_AGE_S
from operator_localization_search import BoundedYawSearch, YawSearchConfig


LAND_ATTEMPTS = 2
AUTO_PCMD_FAILURE_LIMIT = 3
AUTO_WORKER_STALL_TIMEOUT_S = 2.0
AUTO_WATCHDOG_POLL_S = 0.05
AUTO_MAX_DURATION_S = 300.0
AUTO_BATTERY_RESERVE_PCT = 20.0
AUTO_TELEMETRY_MAX_AGE_S = 2.0
# Total time AUTO may hold a hover waiting for something it cannot supply itself
# (a usable pose, a fresh ground-speed sample).  boot_timeout_s only bounds the
# yaw search and still leaves the hover running; this bounds the wait itself so
# a stuck AUTO stops burning battery silently and raises auto_failed instead.
AUTO_WAIT_BUDGET_S = 60.0
# Fresh-pose control time persists across motion phases; external waits pause it.
AUTO_WAYPOINT_STALL_S = 30.0
# Ground speed arrives at 5 Hz, 0.18-0.30 s old when a tick uses it, and keeps
# rising for ~0.5 s after tilt is cut (flight 2026-09-15: p95 0.84-0.91 m/s,
# max 1.22 m/s against 0.6). The limit acts on speed extrapolated this far past
# the sample's age at its measured acceleration.
AUTO_SPEED_LIMIT_LOOKAHEAD_S = 0.25
# Lost-pose yaw search runs at the hook-validated ceiling (20); the 50%
# route-yaw cap lives in production_auto_control_config and must not be
# reused here.
Pcmd = tuple[int, int, int, int]
AuthorizedPcmdResult = tuple[bool, str, Pcmd | None]

@dataclass(frozen=True)
class AutonomyEvent:
    kind: str
    detail: str = ""
    command: str | None = None
    result: object | None = None
    error: str | None = None


@dataclass
class _WaypointProgress:
    target: int
    pose_stamp: float
    tick_stamp: float
    reference_distance: float
    progress_stamp: float
    no_progress_s: float = 0.0
    counting: bool = False
    best_errors: dict[str, float] = field(default_factory=dict)


def _fresh_route_progress() -> dict:
    return {
        "initial_target": None,
        "last_target": None,
        "arrived": [],
        "last_tick": {},
        "last_stage": None,
        "last_stage_target": None,
    }


class DesktopRouteAutonomy:
    """Acquire PC control, hold for a trustworthy pose, then run the route."""

    def __init__(
        self,
        *,
        backend: Any,
        snapshot: rpf.MissionRouteSnapshot,
        map_frame: rpf.MapFrame,
        get_pose: Callable[[], rpf.Pose | None],
        pose_is_weak: Callable[[], bool],
        pose_is_predicted: Callable[[], bool] | None = None,
        pose_confidence: Callable[[], int],
        force_relocalize: Callable[[], None],
        stream_healthy: Callable[[], bool],
        takeoff: Callable[[], object],
        land: Callable[[], object],
        take_pc_control: Callable[[], object] | None = None,
        start_airborne: bool = False,
        arming_blockers: Callable[[], list[str]] | None = None,
        boot_timeout_s: float = pff.FIRST_FIX_TIMEOUT_S,
        boot_search_config: YawSearchConfig | None = None,
        accept_weak_poses: bool = False,
        pose_source_pending: Callable[[], bool] | None = None,
        pose_reseed_confirming: Callable[[], bool] | None = None,
        worker_stall_timeout_s: float | None = None,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        run_loop: Callable[..., str] = pff.run_loop,
    ):
        if not isinstance(map_frame, rpf.MapFrame):
            raise ValueError("AUTO requires the active site's measured map frame")
        timeout = float(boot_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("AUTO localization timeout must be finite and positive")
        stall_timeout = float(
            AUTO_WORKER_STALL_TIMEOUT_S
            if worker_stall_timeout_s is None
            else worker_stall_timeout_s
        )
        if not math.isfinite(stall_timeout) or stall_timeout <= 0.0:
            raise ValueError("AUTO worker stall timeout must be finite and positive")
        if start_airborne and not callable(take_pc_control):
            raise ValueError("airborne AUTO requires an explicit PC-control handoff")
        snapshot.verify_file_unchanged()
        waypoints = snapshot.controller_waypoints()

        if boot_search_config is None:
            boot_search_config = YawSearchConfig(yaw_pcmd=20)
            # Operator decision 2026-09-13: no blind rotation on lost
            # localization, hover in place instead. An explicit config still
            # opts into the sweep (tests and special missions); the default
            # path never spins.
            spin_to_search = False
        else:
            spin_to_search = True
        config = rpf.apply_desktop_auto_authority(
            rpf.config_for_route(
                snapshot, rpf.production_auto_control_config(map_frame)
            )
        )

        self.backend = backend
        self.run_id = uuid.uuid4().hex
        self.snapshot = snapshot
        self.waypoints = waypoints
        self.controller = rpf.RouteAutoController(waypoints, poles=[], config=config)
        self.get_pose = get_pose
        self.pose_is_weak = pose_is_weak
        self.pose_is_predicted = pose_is_predicted or (lambda: False)
        self.pose_source_pending = pose_source_pending
        self.pose_reseed_confirming = pose_reseed_confirming
        self.pose_confidence = pose_confidence
        self.force_relocalize = force_relocalize
        self.stream_healthy = stream_healthy
        self.takeoff = takeoff
        self.land = land
        self.take_pc_control = take_pc_control
        self.start_airborne = bool(start_airborne)
        self.arming_blockers = arming_blockers or (lambda: [])
        self.boot_timeout_s = timeout
        self.boot_search_config = boot_search_config
        self.spin_to_search = bool(spin_to_search)
        # After a visual lock, VO / dead-reckon / IMU-bridge / PREDICTED_ONLY
        # may keep translating while visual localization is down. They do not
        # renew the map anchor and cannot unlock BOOT.
        self.accept_weak_poses = bool(accept_weak_poses)
        self.worker_stall_timeout_s = stall_timeout
        self.now = now
        self.sleep = sleep
        self.run_loop = run_loop
        self._speed_sample: tuple[float, float] | None = None
        self._speed_rate_mps2 = 0.0
        self.events: queue.Queue[AutonomyEvent] = queue.Queue()
        self._cancel = threading.Event()
        self._paused = threading.Event()
        self._cancel_reason = ""
        self._manual_handoff_attempted = False
        self._manual_handoff_lock = threading.Lock()
        self._manual_handoff_thread: threading.Thread | None = None
        self._thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._heartbeat_lock = threading.Lock()
        self._heartbeat_mono = time.monotonic()
        self._landing_confirmed: bool | None = None
        self._pcmd_failure_streak = 0
        self._auto_failed = False
        self._auto_failure_detail = ""
        self._failure_lock = threading.Lock()
        self._command_epoch = 0
        self._waypoint_progress: _WaypointProgress | None = None
        self._mission_started: float | None = None
        self._route_progress: dict = _fresh_route_progress()
        self.phase = "IDLE"

    @property
    def active(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def join(self, timeout: float | None = None) -> bool:
        """Wait boundedly for both the route worker and authority handoff."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        for thread in (
            self._thread,
            self._manual_handoff_thread,
            self._watchdog_thread,
        ):
            if thread is None or thread is threading.current_thread():
                continue
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        return all(
            thread is None or thread is threading.current_thread() or not thread.is_alive()
            for thread in (
                self._thread,
                self._manual_handoff_thread,
                self._watchdog_thread,
            )
        )

    def start(self) -> bool:
        handoff = self._manual_handoff_thread
        watchdog = self._watchdog_thread
        if (
            self.active
            or bool(handoff is not None and handoff.is_alive())
            or bool(watchdog is not None and watchdog.is_alive())
        ):
            return False
        self._cancel.clear()
        self._paused.clear()
        self._cancel_reason = ""
        self._manual_handoff_attempted = False
        self._manual_handoff_thread = None
        self._landing_confirmed = None
        self._pcmd_failure_streak = 0
        self._auto_failed = False
        self._auto_failure_detail = ""
        self._waypoint_progress = None
        self._mission_started = None
        self._route_progress = _fresh_route_progress()
        self._watchdog_stop.clear()
        self._touch_heartbeat()
        self._thread = threading.Thread(
            target=self._run,
            name="desktop-route-autonomy",
            daemon=True,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="desktop-route-autonomy-watchdog",
            daemon=True,
        )
        self._thread.start()
        self._watchdog_thread.start()
        return True

    def cancel(self, reason: str) -> None:
        self._cancel_reason = str(reason or "operator")
        self._cancel.set()
        self._watchdog_stop.set()
        self._paused.clear()
        with self._failure_lock:
            self._command_epoch += 1
        self._clear_motion()
        handoff_reason = f"auto_cancel:{self._cancel_reason}"
        self._send_zero(handoff_reason)
        with self._manual_handoff_lock:
            if not self._manual_handoff_attempted:
                self._manual_handoff_attempted = True
                # give_to_pilot() may wait for source readback. Cancellation is
                # already latched and zeroed; retain the thread so shutdown can
                # wait for it boundedly before disconnecting the backend.
                handoff = threading.Thread(
                    target=self._request_manual,
                    kwargs={"reason": handoff_reason},
                    name="desktop-route-manual-handoff",
                    daemon=True,
                )
                self._manual_handoff_thread = handoff
                handoff.start()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self, reason: str = "operator_hover") -> bool:
        """Hold the active AUTO run in place without discarding its route state."""
        if self.phase not in {"TAKEOFF", "HANDOFF", "BOOT_HOVER", "ROUTE"}:
            return False
        self._paused.set()
        if reason == "auto_localization_source_change":
            self.controller.reset_arrival_confirmation()
        self._clear_motion()
        self._send_zero(reason)
        self._emit("paused", "自動飛行已暫停")
        return True

    def resume(self) -> bool:
        """Resume a paused run; the existing worker continues without takeoff."""
        if self.phase not in {"TAKEOFF", "HANDOFF", "BOOT_HOVER", "ROUTE"}:
            return False
        if not self._paused.is_set():
            return False
        self._paused.clear()
        self._emit("resumed", "繼續原航線")
        return True

    def drain_events(self) -> list[AutonomyEvent]:
        drained = []
        while True:
            try:
                drained.append(self.events.get_nowait())
            except queue.Empty:
                return drained

    def _touch_heartbeat(self) -> None:
        with self._heartbeat_lock:
            self._heartbeat_mono = time.monotonic()

    def _watchdog_loop(self) -> None:
        poll_s = min(
            AUTO_WATCHDOG_POLL_S,
            max(0.01, self.worker_stall_timeout_s / 5.0),
        )
        while not self._watchdog_stop.wait(poll_s):
            worker = self._thread
            if worker is None or not worker.is_alive():
                return
            if self._cancel.is_set():
                return
            if self._check_runtime_budget():
                return
            if self._paused.is_set():
                self._touch_heartbeat()
                continue
            if self.phase in {
                "IDLE",
                "TAKEOFF",
                "HANDOFF",
                "LANDING",
                "LANDING_UNRESOLVED",
                "AUTO_FAILED",
                "DONE",
            }:
                self._touch_heartbeat()
                continue
            with self._heartbeat_lock:
                age_s = time.monotonic() - self._heartbeat_mono
            if age_s <= self.worker_stall_timeout_s:
                continue
            if self._cancel.is_set():
                return
            if self._paused.is_set():
                self._touch_heartbeat()
                continue
            self._latch_auto_failure(
                f"AUTO worker stalled: no heartbeat for {age_s:.2f}s",
                error=f"heartbeat timeout after {age_s:.2f}s",
                zero_reason="auto_worker_stall_hover",
            )
            return

    def _emit(
        self,
        kind: str,
        detail: str = "",
        *,
        command: str | None = None,
        result: object | None = None,
        error: str | None = None,
    ) -> None:
        logs = self._session_logs()
        telemetry = getattr(logs, "telemetry", None)
        if callable(telemetry):
            try:
                telemetry("autonomy_event", auto_run_id=self.run_id, kind=kind,
                          phase=self.phase, detail=str(detail), command=command, error=error,
                          event_mono_ns=int(self.now() * 1e9))
            except (OSError, ValueError, AttributeError, TypeError, RuntimeError):
                pass  # The operator must still receive the event if its log fails.
        self.events.put(
            AutonomyEvent(
                kind=kind,
                detail=detail,
                command=command,
                result=result,
                error=error,
            )
        )

    def _pilot_override(self) -> bool:
        if bool(getattr(self.backend, "pilot_sticks", False)):
            return True
        if self._uncommanded_landing():
            return True
        monitor = getattr(self.backend, "_stick_monitor", None)
        is_active = getattr(monitor, "is_active", None)
        try:
            return bool(callable(is_active) and is_active())
        except Exception:
            return True

    def _uncommanded_landing(self) -> bool:
        """Firmware is landing while AUTO flies, and neither AUTO nor the backend asked.

        The SkyController land button (or a firmware emergency landing) leaves no
        stick trace. AUTO never lands from these phases, and the backend's own
        land_cmd marks its maneuver. Flight 2026-09-15 14:25:20: AUTO kept
        commanding and its gaz +8 cancelled the pilot's landing at 0.17 m.
        """
        if self.phase not in {"HANDOFF", "BOOT_HOVER", "ROUTE"}:
            return False
        state = getattr(getattr(self.backend, "state", None), "flight_state", "")
        if str(state or "").lower() not in {"landing", "emergency_landing"}:
            return False
        return getattr(self.backend, "_maneuver_in_progress", None) != "landing"

    def _safety_mode(self) -> str:
        self._check_runtime_budget()
        if self._operator_stop():
            # run_loop exits on LAND.  _run distinguishes operator/manual cancel
            # from an actual landing request and therefore does not auto-land.
            return "LAND"
        if self._runtime_safety_latched():
            return "HOVER"
        if self._paused.is_set():
            return "HOVER"
        return "AUTO"

    def _runtime_safety_latched(self) -> bool:
        return bool(getattr(self.backend, "_runtime_safety_action_latched", False))

    def _latch_auto_failure(
        self,
        detail: str,
        *,
        error: str | None = None,
        zero_reason: str = "auto_failure_hover",
    ) -> None:
        with self._failure_lock:
            if self._auto_failed:
                return
            self._auto_failed = True
            self._auto_failure_detail = str(detail)
            self.phase = "AUTO_FAILED"
            self._cancel.set()
            self._watchdog_stop.set()
            self._command_epoch += 1
        self._clear_motion()
        self._send_zero(zero_reason)
        self._emit("auto_failed", self._auto_failure_detail, error=error)

    def _check_wait_budget(self, reason: str, elapsed_s: float) -> bool:
        if elapsed_s < AUTO_WAIT_BUDGET_S:
            return False
        self._latch_auto_failure(
            f"{reason} for {elapsed_s:.1f}s, past the "
            f"{AUTO_WAIT_BUDGET_S:g}s AUTO wait budget; manual recovery required"
        )
        return True

    def _check_runtime_budget(self) -> bool:
        if self._mission_started is None or self._cancel.is_set():
            return False
        if self.phase not in {"BOOT_HOVER", "ROUTE"}:
            return False
        elapsed = self.now() - self._mission_started
        reason = None
        if elapsed >= AUTO_MAX_DURATION_S:
            reason = f"AUTO exceeded {AUTO_MAX_DURATION_S:g}s mission budget"
        elif bool(getattr(self.backend, "is_live", False)):
            state = self.backend.state
            try:
                battery = float(state.battery_pct)
                age = self.now() - int(state.telemetry_read_mono_ns) * 1e-9
            except (AttributeError, TypeError, ValueError, OverflowError):
                battery, age = float("nan"), float("inf")
            if not math.isfinite(age) or not 0.0 <= age <= AUTO_TELEMETRY_MAX_AGE_S:
                reason = "AUTO telemetry unavailable or stale"
            elif not math.isfinite(battery) or not 0.0 <= battery <= 100.0:
                reason = "AUTO battery unavailable"
            elif battery <= AUTO_BATTERY_RESERVE_PCT:
                reason = f"AUTO battery {battery:g}% reached {AUTO_BATTERY_RESERVE_PCT:g}% reserve"
        if reason is None:
            return False
        self._latch_auto_failure(reason + "; manual recovery required")
        return True

    def _record_pcmd_failure(self, reason: str) -> None:
        self._pcmd_failure_streak += 1
        if self._pcmd_failure_streak < AUTO_PCMD_FAILURE_LIMIT:
            return
        self._latch_auto_failure(
            f"AUTO PCMD failed {self._pcmd_failure_streak} consecutive times; route terminated",
            error=str(reason),
            zero_reason="auto_pcmd_failure_hover",
        )

    def _operator_stop(self) -> bool:
        if self._cancel.is_set() or self._auto_failed:
            return True
        if not self._pilot_override():
            return False
        if not self._manual_handoff_attempted:
            self._manual_handoff_attempted = True
            reason = "auto_stick_override"
            if self._uncommanded_landing():
                reason = "auto_rc_landing"
                self._emit("rc_landing_yield", "遙控器降落中：AUTO 停止送命令並交還遙控器")
            self._send_zero(reason)
            self._request_manual(reason)
        return True

    def _clear_motion(self) -> None:
        clear = getattr(self.backend, "clear_nudge_vector", None)
        if callable(clear):
            try:
                clear()
            except Exception:
                pass

    def _send_zero(self, reason: str) -> bool:
        send = getattr(self.backend, "send_pcmd", None)
        if not callable(send):
            return False
        try:
            return bool(send(0, 0, 0, 0, reason=reason))
        except Exception:
            return False

    def _send_authorized(self, pcmd: Pcmd) -> AuthorizedPcmdResult:
        self._touch_heartbeat()
        self._check_runtime_budget()
        if self._auto_failed:
            self._clear_motion()
            self._send_zero("auto_failure_hover")
            return (
                False,
                f"AUTO failed: {self._auto_failure_detail}",
                (0, 0, 0, 0),
            )
        if self._cancel.is_set():
            self._clear_motion()
            self._send_zero(f"auto_cancel:{self._cancel_reason}")
            return False, f"AUTO cancelled: {self._cancel_reason}", (0, 0, 0, 0)
        if self._pilot_override():
            self._clear_motion()
            if self._uncommanded_landing():
                return False, "RC landing: AUTO yields", None
            return False, "pilot stick override", None
        if self._runtime_safety_latched():
            self._clear_motion()
            self._send_zero("auto_runtime_safety_hover")
            return False, "AUTO runtime safety hover", (0, 0, 0, 0)
        if self._paused.is_set():
            self._clear_motion()
            self._send_zero("auto_operator_pause")
            return False, "AUTO paused", (0, 0, 0, 0)
        if not self.stream_healthy():
            self._clear_motion()
            self._send_zero("auto_stream_unhealthy")
            return False, "AUTO stream unhealthy", (0, 0, 0, 0)

        epoch = int(self._command_epoch)
        pcmd = self._apply_speed_limit(pcmd)
        result = self._dispatch_route_pcmd(pcmd)
        if (
            int(self._command_epoch) != epoch
            or self._auto_failed
            or self._cancel.is_set()
        ):
            self._clear_motion()
            reason = (
                "auto_failure_hover"
                if self._auto_failed
                else f"auto_cancel:{self._cancel_reason}"
            )
            self._send_zero(reason)
            return False, reason, (0, 0, 0, 0)
        return result

    def _predicted_speed(self, speed: float, stamp: float) -> float:
        """Speed expected once this tick's tilt takes effect."""
        previous = self._speed_sample
        if previous is None or stamp > previous[1]:
            gap = None if previous is None else stamp - previous[1]
            self._speed_rate_mps2 = (
                (speed - previous[0]) / gap if gap is not None and gap <= 0.5 else 0.0
            )
            self._speed_sample = (speed, stamp)
        horizon = max(0.0, self.now() - stamp) + AUTO_SPEED_LIMIT_LOOKAHEAD_S
        return speed + max(0.0, self._speed_rate_mps2) * horizon

    def _apply_speed_limit(self, pcmd: Pcmd) -> Pcmd:
        """Shrink horizontal PCMD that would push ground speed past the limit.

        Never latched: the allowed tilt falls quadratically from the full cap at
        standstill to zero at the limit, evaluated on speed predicted past the
        telemetry delay. Commands that do not add speed along the current motion
        (braking, wind hold) pass untouched, as does everything when fresh body
        velocity is unavailable.
        """
        roll, pitch = int(pcmd[0]), int(pcmd[1])
        state = self.backend.state
        if not (roll or pitch):
            return pcmd
        if not bool(getattr(state, "autonomous_speed_limit_enabled", True)):
            state.autonomous_speed_guard_status = "SPEED_LIMIT_DISABLED"
            return pcmd
        try:
            limit = float(state.autonomous_speed_limit_mps)
        except (AttributeError, TypeError, ValueError, OverflowError):
            limit = float("nan")
        velocity = self._body_velocity()
        if not math.isfinite(limit) or limit <= 0.0 or velocity is None:
            state.autonomous_speed_guard_status = "SPEED_UNAVAILABLE"
            return pcmd
        forward_v, right_v, stamp = velocity
        predicted = self._predicted_speed(math.hypot(forward_v, right_v), stamp)
        if pitch * forward_v + roll * right_v <= 0.0:
            state.autonomous_speed_guard_status = "FRESH_SPEED_LIMIT"
            return pcmd
        ceiling = (
            self.controller.cfg.max_translation_pcmd * max(0.0, 1.0 - predicted / limit) ** 2
        )
        magnitude = max(abs(roll), abs(pitch))
        if magnitude <= ceiling:
            state.autonomous_speed_guard_status = "FRESH_SPEED_LIMIT"
            return pcmd
        scale = ceiling / magnitude
        state.autonomous_speed_guard_status = "SPEED_LIMITED"
        return round(roll * scale), round(pitch * scale), pcmd[2], pcmd[3]

    @staticmethod
    def _route_pcmd_reason(accepted: bool) -> str:
        if not accepted:
            return "AUTO PCMD rejected"
        return "desktop AUTO command cap"

    def _pcmd_authority_pct(self) -> int:
        """Percent-of-envelope AUTO's unit vector scales by.

        The widest per-axis cap the controller can emit, so every axis survives
        the round trip through the unit vector unchanged.
        """
        cfg = self.controller.cfg
        return max(1, min(100, int(max(
            cfg.max_translation_pcmd, cfg.max_vertical_pcmd, cfg.max_yaw_pcmd
        ))))

    def _dispatch_route_pcmd(
        self,
        pcmd: Pcmd,
    ) -> AuthorizedPcmdResult:
        """Send through the backend's normalized vector seam or direct PCMD."""

        # Unit axes against AUTO's own authority, not the operator's nudge size.
        # The backend scales them back by the same number, so the percentages the
        # controller computed are what reaches the aircraft.  Dividing by
        # nudge_pct here (and letting the backend multiply by nudge_pct) made the
        # whole chain collapse to clamp(pcmd, +/-nudge_pct): with --nudge-pct 8
        # a commanded 50% yaw left as 8%, i.e. 1.6 deg/s of a 20 deg/s envelope.
        authority = self._pcmd_authority_pct()
        vector = tuple(max(-1.0, min(1.0, float(value) / authority)) for value in pcmd)
        setter = getattr(self.backend, "set_nudge_vector", None)
        if callable(setter):
            try:
                accepted = bool(setter(*vector, authority_pct=authority))
            except TypeError:
                # Backends predating the per-vector authority (the simulated
                # route plant) still take four positional axes.
                accepted = bool(setter(*vector))
            except Exception as exc:
                reason = f"AUTO PCMD failed: {exc!r}"
                self._record_pcmd_failure(reason)
                return False, reason, None
            if accepted:
                self._pcmd_failure_streak = 0
            else:
                self._record_pcmd_failure("AUTO PCMD rejected")
            return (
                accepted,
                self._route_pcmd_reason(accepted),
                pcmd if accepted else None,
            )

        send = getattr(self.backend, "send_pcmd", None)
        if not callable(send):
            self._record_pcmd_failure("AUTO PCMD unavailable")
            return False, "AUTO PCMD unavailable", None
        try:
            accepted = bool(send(*pcmd, reason="desktop_auto_route"))
        except Exception as exc:
            reason = f"AUTO PCMD failed: {exc!r}"
            self._record_pcmd_failure(reason)
            return False, reason, None
        if accepted:
            self._pcmd_failure_streak = 0
        else:
            self._record_pcmd_failure("AUTO PCMD rejected")
        reason = self._route_pcmd_reason(accepted)
        return accepted, reason, pcmd if accepted else None

    def _olympe_yaw(self) -> float | None:
        state = self.backend.state
        if hasattr(state, "attitude_mono_ns"):
            stamp = state.attitude_mono_ns
            if stamp is None or not 0.0 <= self.now() - int(stamp) / 1e9 <= pff.POSE_STALE_S:
                return None
        try:
            yaw = float(getattr(self.backend.state, "att_yaw", float("nan")))
        except (TypeError, ValueError, OverflowError):
            return None
        return yaw if math.isfinite(yaw) else None

    def _ground_speed(self) -> tuple[float, float] | None:
        state = self.backend.state
        try:
            speed = float(state.ground_speed_mps)
            stamp = int(state.ground_speed_mono_ns) / 1_000_000_000.0
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(speed) or not math.isfinite(stamp) or speed < 0.0:
            return None
        return speed, stamp

    def _body_velocity(self) -> tuple[float, float, float] | None:
        """Fresh firmware NED velocity projected into the aircraft's body axes."""
        sample = self._ground_speed()
        yaw = self._olympe_yaw()
        if sample is None or yaw is None:
            return None
        _speed, stamp = sample
        if not 0.0 <= self.now() - stamp <= GROUND_SPEED_MAX_AGE_S:
            return None
        state = self.backend.state
        try:
            north, east = float(state.speed_north_mps), float(state.speed_east_mps)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(north) or not math.isfinite(east):
            return None
        return (math.cos(yaw) * north + math.sin(yaw) * east,
                -math.sin(yaw) * north + math.cos(yaw) * east, stamp)

    def _request_manual(self, reason: str = "auto_stick_override") -> bool:
        give = getattr(self.backend, "give_to_pilot", None)
        if not callable(give):
            return False
        try:
            return bool(give(reason=reason))
        except Exception:
            return False

    def _route_hooks(self) -> pff.LoopHooks:
        return pff.LoopHooks(
            get_pose=self.get_pose,
            olympe_yaw=self._olympe_yaw,
            send_pcmd=lambda r, p, y, g: self.backend.send_pcmd(
                r, p, y, g, reason="desktop_auto_route"
            ),
            send_authorized_pcmd=self._send_authorized,
            pose_is_weak=self.pose_is_weak,
            pose_is_predicted=self.pose_is_predicted,
            pose_source_pending=self.pose_source_pending,
            pose_reseed_confirming=self.pose_reseed_confirming,
            pose_confidence=self.pose_confidence,
            max_weak_pose_age_s=pff.POSE_STALE_S,
            force_relocalize=self.force_relocalize,
            request_manual=self._request_manual,
            stick_active=self._pilot_override,
            safety_poll=self._safety_mode,
            stream_healthy=self.stream_healthy,
            stream_status=lambda: str(getattr(self.backend.state, "stream", "unknown")),
            log_tick=self._log_route_tick,
            loop_beat=self._touch_heartbeat,
            pose_jump_pause=self._pause_for_pose_jump,
            localization_yaw_search_pcmd=self.boot_search_config.yaw_pcmd if self.spin_to_search else 0,
            localization_yaw_search_event=self._on_localization_yaw_search,
            ground_speed=self._ground_speed,
            body_velocity=self._body_velocity,
            wait_expired=self._check_wait_budget,
            # Localization loss never auto-lands the AUTO route: the loop holds
            # zero PCMD through the bounded yaw search until the pose recovers
            # or the operator takes over with the sticks. Physical stick
            # movement still invokes request_manual immediately.
            land_on_localization_loss=False,
            now=self.now,
        )

    def _session_logs(self) -> Any:
        return getattr(self.backend, "session_logs", None)

    def _log_route_tick(self, record: dict) -> None:
        """Persist one route-loop tick so the flown path can be read back.

        The per-frame pose already reaches localization.jsonl, but nothing
        recorded what the route loop made of it -- the cross-track distance, the
        waypoint being flown, the PCMD and why.  Without those the log answers
        "where was the drone" but not "how far off the plan was it", which is the
        question every AUTO debrief actually asks.
        """
        record["auto_run_id"] = self.run_id
        now = self.now()
        state = self.backend.state
        for name in (
            "ground_speed_mps", "speed_north_mps", "speed_east_mps", "speed_down_mps",
            "drone_altitude_m", "att_roll", "att_pitch", "att_yaw", "wind_state",
            "flight_state", "control_owner", "battery_pct",
        ):
            record[name] = getattr(state, name, None)
        for name in ("ground_speed", "attitude", "altitude"):
            stamp = getattr(state, f"{name}_mono_ns", None)
            record[f"{name}_mono_ns"] = stamp
            record[f"{name}_age_s"] = None if stamp is None else now - int(stamp) / 1e9
        record["auto_elapsed_s"] = None if self._mission_started is None else now - self._mission_started
        self._check_waypoint_progress(record)
        self._track_route_progress(record)
        logs = self._session_logs()
        if logs is None:
            return
        try:
            logs.localization("auto_route_tick", **record)
        except (
            OSError,
            ValueError,
            AttributeError,
            TypeError,
            RuntimeError,
        ):  # Tier3: fallback best-effort — narrow, keep pass
            pass

    def _check_waypoint_progress(self, record: dict) -> None:
        """Keep a waypoint's progress budget across turns, braking and recovery."""
        target = self._tick_target(record)
        if (
            self.phase != "ROUTE"
            or target is None or not 0 <= target < len(self.controller.wp)
        ):
            if self._waypoint_progress is not None:
                self._waypoint_progress.tick_stamp = self.now()
                self._waypoint_progress.counting = False
            return
        now = self.now()
        state = self._waypoint_progress
        dt = 0.0 if state is None else now - state.tick_stamp
        if state is not None:
            state.tick_stamp = now
        try:
            position = np.asarray(record.get("pose_u"), dtype=float)
            stamp = float(record.get("pose_stamp", float("nan")))
            valid = (
                position.shape == (3,) and np.isfinite(position).all()
                and math.isfinite(stamp) and math.isfinite(now)
                and -0.05 <= now - stamp <= self.controller.cfg.max_pose_age_s
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        motion_block = "speed" in str(record.get("reason", "")).lower()
        if not valid or self._paused.is_set() or (record.get("blocked") and not motion_block):
            if state is not None:
                state.counting = False
            return
        distance = float(np.linalg.norm(self.controller.wp[target] - position))
        if state is None or state.target != target or dt < 0.0:
            state = _WaypointProgress(target, stamp - 1e-9, now, distance, now)
            self._waypoint_progress = state
        elif state.counting and dt <= self.controller.cfg.max_pose_age_s:
            state.no_progress_s += dt
        state.counting = True
        threshold = max(
            2.0 * self.controller.cfg.translation_arrival_tolerance,
            0.5 * self.controller._arrive_radius_for(target),
        )
        if stamp > state.pose_stamp:
            state.pose_stamp = stamp
            improved = state.reference_distance - distance >= threshold
            if improved:
                state.reference_distance = distance
                # After real spatial progress, a later mid-leg turn is a new
                # heading correction. Mere phase switching cannot renew it.
                state.best_errors.pop("yaw_error_deg", None)
            phase = record.get("pcmd_phase")
            metrics = []
            if phase in {"turn", "yaw_alignment_hold", "yaw_alignment_confirmed"}:
                metrics.append(("yaw_error_deg", 5.0))
            if phase == "route_rejoin":
                metrics.append(("path_error_u", threshold))
            if phase == "turn_drift_recovery":
                metrics.append(("turn_anchor_error_u", threshold))
            for key, required in metrics:
                try:
                    error = abs(float(record.get(key, float("nan"))))
                except (TypeError, ValueError, OverflowError):
                    continue
                if not math.isfinite(error):
                    continue
                previous = state.best_errors.setdefault(key, error)
                if previous - error >= required:
                    state.best_errors[key] = error
                    improved = True
            if improved:
                state.progress_stamp = now
                state.no_progress_s = 0.0
        elapsed = state.no_progress_s
        record["waypoint_no_progress_s"] = round(elapsed, 3)
        record["waypoint_distance_u"] = distance
        if elapsed >= AUTO_WAYPOINT_STALL_S:
            detail = (
                f"waypoint {target + 1} made no measurable progress for {elapsed:.1f}s "
                f"(remaining {distance:.3f} map units); holding for operator"
            )
            self._latch_auto_failure(detail, zero_reason="auto_waypoint_stall_hover")
            record.update(blocked=True, pcmd=[0, 0, 0, 0], reason=detail)

    @staticmethod
    def _tick_target(record: dict) -> int | None:
        try:
            target = record.get("target_index")
        except AttributeError:
            return None
        if target is None:
            return None
        try:
            return int(target)
        except (TypeError, ValueError):
            return None

    def _drawn_waypoint_label(self, target: int) -> str:
        """Drawn-frame waypoint name for a controller target index."""
        drawn = max(1, len(self.waypoints))
        if drawn >= 2 and target > drawn - 1:
            there = 2 * drawn - target - 1
            return f"路徑點{there}（返程）"
        return f"路徑點{target + 1}"

    def _drawn_leg_label(self, target: int) -> str:
        drawn = max(1, len(self.waypoints))
        if drawn >= 2 and target > drawn - 1:
            there = 2 * drawn - target - 1
            return f"返程前往路徑點{there}"
        return f"前往路徑點{target + 1}"

    def _track_route_progress(self, record: dict) -> None:
        """Track arrivals, stage changes, and the latest tick for display.

        Every tick refreshes the live stage line; arrivals and stage
        transitions also enter the event queue so the operator log shows
        exactly which waypoint phase the run reached last.
        """
        try:
            last = {
                "phase": record.get("pcmd_phase"),
                "action": record.get("action"),
                "reason": str(record.get("reason") or ""),
                "blocked": bool(record.get("blocked")),
                "path_error_u": record.get("route_distance_u"),
                "has_pose": record.get("pose_u") is not None,
                "target_index": record.get("target_index"),
                "search": record.get("localization_yaw_search"),
                "search_deg": record.get("localization_yaw_search_deg"),
            }
        except AttributeError:
            return
        state = self._route_progress
        state["last_tick"] = last
        target = self._tick_target(record)
        if target is None:
            target = state["last_target"]
            if target is None:
                target = int(self.controller.target_index)
        if state["initial_target"] is None:
            state["initial_target"] = target
            state["last_target"] = target
        else:
            last_target = state["last_target"]
            if target != last_target:
                if target > last_target:
                    # The retired waypoint is the one just left, not the new target.
                    state["arrived"].append(last_target)
                    self._emit(
                        "waypoint_arrived",
                        f"到達{self._drawn_waypoint_label(last_target)}",
                    )
                else:
                    state["arrived"] = []
                    state["initial_target"] = target
                state["last_target"] = target
        if (last["action"] == "final path reached -> LAND" or str(last["action"]).startswith("LAND")) and target not in state["arrived"]:
            state["arrived"].append(target)
            self._emit("waypoint_arrived", f"到達{self._drawn_waypoint_label(target)}")
        stage_key, stage_text = self._leg_stage(target, last)
        if stage_key != state.get("last_stage") or target != state.get("last_stage_target"):
            state["last_stage"] = stage_key
            state["last_stage_target"] = target
            self._emit("leg_stage", stage_text)

    def _leg_stage(self, target: int, last: dict) -> tuple[str, str]:
        return leg_stage(self._drawn_waypoint_label(int(target)), last)

    #: Coordinator phase text shown ahead of the leg stage when not on route.
    _COORDINATOR_PHASE_TEXT = {
        "TAKEOFF": "起飛中",
        "HANDOFF": "控制權交接中",
        "BOOT_HOVER": "找尋定位中（起飛點懸停）",
        "LANDING": "降落中",
        "LANDING_UNRESOLVED": "降落未確認",
        "AUTO_FAILED": "失敗鎖定懸停",
        "DONE": "已結束",
        "IDLE": "未啟動",
    }

    def auto_leg_status(self) -> dict:
        """Live AUTO leg snapshot for the operator display (UI thread reads)."""
        try:
            target = int(self.controller.target_index)
            total = len(self.controller.wp)
        except (AttributeError, TypeError, ValueError):
            return {"active": False, "line": "AUTO 狀態讀取失敗"}
        state = self._route_progress
        arrived = [self._drawn_waypoint_label(int(value)) for value in state["arrived"]]
        last = state["last_tick"] or {}
        recorded_target = last.get("target_index")
        if isinstance(recorded_target, int) and 0 <= recorded_target < total:
            target = recorded_target
        if last:
            stage_key, action = self._leg_stage(target, last)
        else:
            stage_key, action = "none", "等待路線資料中"
        phase_text = self._COORDINATOR_PHASE_TEXT.get(self.phase, "")
        if self.phase != "ROUTE" and phase_text:
            action = phase_text
            stage_key = self.phase.lower()
        if self.paused:
            action, stage_key = "自動飛行已暫停", "paused"
        try:
            error = float(last.get("path_error_u", float("nan")))
            error_text = f"｜誤差{error:.2f}u" if math.isfinite(error) else ""
        except (TypeError, ValueError):
            error_text = ""
        arrived_text = f"｜已到達：{'、'.join(arrived)}" if arrived else ""
        line = (
            f"AUTO 第{target + 1}站/共{total}站・{self._drawn_leg_label(target)}・"
            f"{action}{error_text}{arrived_text}"
        )
        return {
            "active": True,
            "target": target,
            "total": total,
            "leg": self._drawn_leg_label(target),
            "arrived": arrived,
            "action": action,
            "stage": stage_key,
            "path_error_u": last.get("path_error_u"),
            "line": line,
        }

    def _plan_radii(self) -> list[float] | None:
        radii = getattr(self.controller.cfg, "waypoint_arrive_radii", None)
        if radii is None:
            return None
        try:
            return [round(float(value), 4) for value in radii]
        except (TypeError, ValueError):
            return None

    def _log_route_plan(self) -> None:
        """Write the plan once, so a session log plots without the route file."""
        logs = self._session_logs()
        if logs is None:
            return
        try:
            config = dict(vars(self.controller.cfg))
            frame = config.pop("map_frame")
            config["map_frame"] = {
                "east": frame.east.tolist(), "north": frame.north.tolist(),
                "up": frame.up.tolist(), "source": frame.source,
            }
            logs.localization(
                "auto_route_plan",
                auto_run_id=self.run_id,
                control_config=config,
                route_sha256=self.snapshot.sha256,
                coordinate_frame_id=self.snapshot.coordinate_frame_id,
                map_pose_max_age_s=pff.POSE_STALE_S,
                auto_duration_limit_s=AUTO_MAX_DURATION_S,
                waypoint_stall_limit_s=AUTO_WAYPOINT_STALL_S,
                heading_convention="map CCW; firmware yaw NED CW",
                waypoints_u=[
                    [round(float(value), 4) for value in point] for point in self.controller.wp
                ],
                drawn_waypoint_count=len(self.waypoints),
                return_to_start=bool(self.controller.cfg.return_to_start),
                waypoint_arrive_radius_u=float(self.controller.cfg.waypoint_arrive_radius),
                waypoint_arrive_radii_u=self._plan_radii(),
                target_index=int(self.controller.target_index),
                accept_weak_poses=bool(self.accept_weak_poses),
            )
        except (
            OSError,
            ValueError,
            AttributeError,
            TypeError,
            RuntimeError,
        ):  # Tier3: fallback best-effort — narrow, keep pass
            pass

    def _land_after_failure(self, detail: str) -> None:
        self.phase = "LANDING"
        self._landing_confirmed = False
        self._clear_motion()
        self._send_zero("auto_pre_land")
        last_result = None
        last_error = None
        for attempt in range(1, LAND_ATTEMPTS + 1):
            # land_cmd re-claims PC control on the live backend, so a retry must
            # re-check authority rather than trust the previous attempt: if the
            # operator took the sticks while the last land() was in flight, this
            # run has no standing to command the aircraft again.
            if self._operator_stop():
                return
            self._touch_heartbeat()
            result = None
            error = None
            try:
                result = self.land()
            except Exception as exc:
                error = repr(exc)
            last_result, last_error = result, error
            self._emit(
                "command_result",
                detail,
                command="land",
                result=result,
                error=error,
            )
            if result is True:
                self._landing_confirmed = True
                return
            if attempt < LAND_ATTEMPTS:
                self._clear_motion()
                self._send_zero("auto_pre_land_retry")
        self.phase = "LANDING_UNRESOLVED"
        self._emit(
            "landing_unresolved",
            detail,
            command="land",
            result=last_result,
            error=last_error,
        )

    def _takeoff_command(self) -> tuple[object | None, str | None]:
        self.phase = "TAKEOFF"
        self._touch_heartbeat()
        takeoff_result = None
        takeoff_error = None
        try:
            takeoff_result = self.takeoff()
        except Exception as exc:
            takeoff_error = repr(exc)
        self._emit(
            "command_result",
            "AUTO takeoff completed",
            command="takeoff",
            result=takeoff_result,
            error=takeoff_error,
        )
        return takeoff_result, takeoff_error

    def _airborne_handoff_command(self) -> tuple[object | None, str | None]:
        self.phase = "HANDOFF"
        self._touch_heartbeat()
        handoff_result = None
        handoff_error = None
        try:
            take_pc_control = self.take_pc_control
            if take_pc_control is None:
                raise RuntimeError("AUTO airborne PC control handoff is unavailable")
            handoff_result = take_pc_control()
        except Exception as exc:
            handoff_error = repr(exc)
        self._emit(
            "command_result",
            "AUTO airborne PC control handoff completed",
            command="pc_control",
            result=handoff_result,
            error=handoff_error,
        )
        return handoff_result, handoff_error

    def _hold_boot_hover_paused(self) -> bool:
        while self._paused.is_set():
            self._touch_heartbeat()
            self._check_runtime_budget()
            if self._operator_stop():
                return False
            self._send_zero("auto_operator_pause")
            self.sleep(0.1)
        return True

    def _observe_boot_pose(
        self,
        lock: pff.BootPoseLock,
        last_pose_stamp: float | None,
    ) -> tuple[bool, float | None, bool]:
        pose = self.get_pose()
        if pose is None or self.pose_is_weak() or self.pose_confidence() < pff.LOW_CONF_INLIERS:
            lock.reset()
            return False, last_pose_stamp, False
        try:
            yaw_is_finite = math.isfinite(float(pose.yaw))
        except (AttributeError, TypeError, ValueError, OverflowError):
            yaw_is_finite = False
        if not yaw_is_finite:
            lock.reset()
            return False, last_pose_stamp, False
        pose_stamp = float(pose.stamp)
        if last_pose_stamp is None or pose_stamp > last_pose_stamp:
            last_pose_stamp = pose_stamp
            if lock.observe(pose, now=self.now()):
                return True, last_pose_stamp, True
        return False, last_pose_stamp, True

    def _send_boot_search_yaw(self, yaw_pcmd: int) -> bool:
        self._clear_motion()
        if not callable(getattr(self.backend, "set_nudge_vector", None)):
            return False
        accepted, _reason, _actual = self._send_authorized((0, 0, int(yaw_pcmd), 0))
        return accepted

    def _on_localization_yaw_search(self, state: str, progress_deg: float) -> None:
        progress = max(0.0, float(progress_deg))
        messages = {
            "started": "AUTO 定位仍失敗：保持位置與高度，開始向右旋轉一圈搜尋線索",
            "completed": "AUTO 右旋搜尋已滿一圈仍無可靠定位：停止旋轉並持續懸停重試",
            "recovered": "AUTO 右旋搜尋期間已恢復可靠定位：停止旋轉並懸停確認",
            "timed_out": "AUTO 右旋搜尋逾時：停止旋轉並持續懸停重試",
            "yaw_telemetry_lost": "AUTO 右旋搜尋失去 IMU yaw：立即停止旋轉",
            "command_rejected": "AUTO 右旋搜尋指令遭拒：立即停止旋轉",
        }
        detail = messages.get(state, f"AUTO 右旋搜尋狀態：{state}")
        self._emit(
            "localization_yaw_search",
            f"{detail}（已旋轉 {progress:.1f}°）",
        )

    def _pause_for_pose_jump(self, distance_map_units: float) -> None:
        distance = max(0.0, float(distance_map_units))
        if self.pause("auto_localization_pose_jump"):
            self._emit(
                "pose_jump_paused",
                f"定位位置單次跳變 {distance:.2f} map units",
            )

    def _hold_after_boot_search_failure(self, detail: str) -> None:
        self._clear_motion()
        self._send_zero("auto_boot_localization_failed")
        self._emit(
            "boot_hover_waiting",
            detail + "; holding position and continuing localization retries",
        )

    def _select_route_start(self, position: Any) -> None:
        """Start the route at waypoint 1, disclosing the flight to it.

        AUTO always flies the authored waypoints in order, even when the
        takeoff pose sits nearer a later point: that leg is flown direct to
        waypoint 1, then waypoint 2, and so on. There is no corridor policing
        that opening leg or the drawn route: every tick flies direct at the
        current waypoint from the current localized position, so drift is
        corrected continuously. The join distance is still disclosed here so
        the operator sees how far waypoint 1 is from this takeoff spot.
        """
        route_distance, _nearest_point, _segment, _s = rpf.project_to_path(
            position, self.controller.wp, self.controller.cum
        )
        start = int(self.controller.start_after_nearest_waypoint(position))
        target = self.controller.target_index
        if self.controller.cfg.return_to_start:
            tail = "flies the route to its last waypoint, then returns along it and lands at waypoint 1"
        else:
            tail = "lands at the last waypoint"
        self._emit(
            "route_start_selected",
            f"takeoff pose starts at waypoint {start + 1} "
            f"({float(route_distance):.3f} map units off route); "
            f"first target is waypoint "
            f"{self._drawn_waypoint_number(target)} of {len(self.waypoints)}; {tail}",
        )

    def _drawn_waypoint_number(self, index: int) -> int:
        """1-based number on the *drawn* route for a controller waypoint index.

        A return-to-start route carries the reverse pass as extra indices, so
        past the turnaround the raw index no longer names a waypoint the
        operator drew.  Fold it back onto the drawn numbering.
        """
        drawn = len(self.waypoints)
        if index < drawn:
            return int(index) + 1
        return 2 * drawn - 1 - int(index)

    def _boot_hover(self) -> str | None:
        self.phase = "BOOT_HOVER"
        self._touch_heartbeat()
        self._emit(
            "boot_hover",
            "hovering until stable localization or operator action",
        )
        lock = pff.BootPoseLock(self.waypoints)
        started = self.now()
        next_wait_notice = started + self.boot_timeout_s
        last_relocalize = float("-inf")
        last_pose_stamp: float | None = None
        last_arming_blockers: tuple[str, ...] = ()
        search = BoundedYawSearch(self.boot_search_config)
        search.begin(started)
        search_enabled = self.spin_to_search
        last_search_state: str | None = None
        self._clear_motion()
        while True:
            self._touch_heartbeat()
            self._check_runtime_budget()
            if self._operator_stop():
                return "AUTO cancelled during localization hover"
            if self._paused.is_set():
                paused_at = self.now()
                if not self._hold_boot_hover_paused():
                    return "AUTO cancelled while paused"
                paused_for = max(0.0, self.now() - paused_at)
                started += paused_for
                next_wait_notice += paused_for
                continue
            locked, last_pose_stamp, pose_valid = self._observe_boot_pose(lock, last_pose_stamp)
            if locked:
                blockers = tuple(self.arming_blockers())
                if not blockers:
                    position = lock.position
                    if position is None:
                        lock.reset()
                        continue
                    self._select_route_start(position)
                    self._send_zero("auto_boot_pose_locked")
                    return None
                lock.reset()
                if blockers != last_arming_blockers:
                    self._emit(
                        "boot_hover_waiting",
                        "AUTO arming remains blocked: "
                        + "; ".join(blockers)
                        + "; continuing hover",
                    )
                    last_arming_blockers = blockers
            now = self.now()
            if now - last_relocalize >= 0.5:
                try:
                    self.force_relocalize()
                except Exception:
                    pass
                last_relocalize = now
            if search_enabled and now - started >= self.boot_timeout_s:
                self._hold_after_boot_search_failure(
                    f"stable localization unavailable after {now - started:.1f}s"
                )
                search_enabled = False
                next_wait_notice = now + self.boot_timeout_s
            if self._check_wait_budget("initial localization unavailable", now - started):
                return "initial localization wait expired -> manual"

            if search_enabled:
                safe_to_search = not self._runtime_safety_latched()
                try:
                    safe_to_search = safe_to_search and bool(self.stream_healthy())
                except Exception:
                    safe_to_search = False
                step = search.step(
                    now=now,
                    aircraft_yaw_rad=self._olympe_yaw(),
                    safe=safe_to_search,
                )
                if pose_valid:
                    search_state = "pose_stabilizing"
                    yaw_pcmd = 0
                else:
                    search_state = step.state
                    yaw_pcmd = step.yaw_pcmd
                if search_state != last_search_state:
                    if search_state.startswith("searching_"):
                        direction = "right" if search_state.endswith("right") else "left"
                        self._emit(
                            "boot_hover_search",
                            f"initial localization yaw search: {direction}",
                        )
                    elif search_state == "no_yaw_telemetry":
                        self._emit(
                            "boot_hover_search_waiting",
                            "yaw telemetry unavailable; holding position without blind rotation",
                        )
                    last_search_state = search_state
                if not pose_valid and step.terminal:
                    self._hold_after_boot_search_failure(
                        "initial localization yaw search " + step.state.replace("_", " ")
                    )
                    search_enabled = False
                    yaw_pcmd = 0
                if yaw_pcmd and not self._send_boot_search_yaw(yaw_pcmd):
                    self._hold_after_boot_search_failure(
                        "initial localization yaw command was rejected"
                    )
                    search_enabled = False
                elif not yaw_pcmd:
                    self._send_zero("auto_boot_hover")
            else:
                self._send_zero("auto_boot_hover")

            if now >= next_wait_notice:
                self._emit(
                    "boot_hover_waiting",
                    f"stable localization unavailable after {now - started:.1f}s; continuing hover",
                )
                next_wait_notice = now + self.boot_timeout_s
            self.sleep(0.1)

    def _run_route(self) -> str:
        self.phase = "ROUTE"
        self._touch_heartbeat()
        self._emit("route_started", "stable localization confirmed")
        self._log_route_plan()
        reason = self.run_loop(
            self._route_hooks(),
            self.controller,
            self.waypoints,
            yaw_sign=1,
            verbose=False,
            enforce_weak_pose_gate=not self.accept_weak_poses,
        )
        return str(reason or "AUTO route ended")

    def _run(self) -> None:
        terminal_detail = "AUTO stopped"
        try:
            if self.start_airborne:
                handoff_result, handoff_error = self._airborne_handoff_command()
                if handoff_error is not None or handoff_result is not True:
                    terminal_detail = "AUTO PC control handoff failed"
                    return
            else:
                takeoff_result, takeoff_error = self._takeoff_command()
                if takeoff_error is not None or takeoff_result is not True:
                    terminal_detail = "AUTO takeoff failed"
                    return
            if self._operator_stop():
                terminal_detail = (
                    "AUTO cancelled after PC control handoff"
                    if self.start_airborne
                    else "AUTO cancelled after takeoff"
                )
                return

            self._mission_started = self.now()
            boot_detail = self._boot_hover()
            if boot_detail is not None:
                terminal_detail = boot_detail
                if not self._operator_stop():
                    self._land_after_failure(terminal_detail)
                return

            terminal_detail = self._run_route()
            if self._operator_stop():
                return
            self._land_after_failure(terminal_detail)
        except BaseException as exc:  # safety cleanup must cover worker failures
            terminal_detail = f"AUTO worker failed: {exc!r}"
            self._latch_auto_failure(
                terminal_detail,
                error=repr(exc),
                zero_reason="auto_worker_failure_hover",
            )
        finally:
            self._clear_motion()
            self._paused.clear()
            self._watchdog_stop.set()
            if self._auto_failed:
                self.phase = "AUTO_FAILED"
            elif self._landing_confirmed is False and not self._cancel.is_set():
                self.phase = "LANDING_UNRESOLVED"
            else:
                self.phase = "DONE"
                self._emit("finished", terminal_detail)

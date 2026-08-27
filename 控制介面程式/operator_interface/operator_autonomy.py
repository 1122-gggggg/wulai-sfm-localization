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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Keep this module importable directly by its unit tests as well as through the
# desktop entrypoint, which normally installs the flight-control path first.
_FLIGHT_CONTROL = Path(__file__).resolve().parents[2] / "定位演算法" / "flight_control"
if str(_FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_CONTROL))

import path_follow_flight as pff
import real_path_follow_controller as rpf
from landing_transition import GROUND_SPEED_MAX_AGE_S
from operator_localization_search import BoundedYawSearch, YawSearchConfig


LAND_ATTEMPTS = 2
AUTO_PCMD_FAILURE_LIMIT = 3
AUTO_WORKER_STALL_TIMEOUT_S = 2.0
AUTO_WATCHDOG_POLL_S = 0.05
AUTO_SPEED_GUARD_RELEASE_RATIO = 0.80
Pcmd = tuple[int, int, int, int]
AuthorizedPcmdResult = tuple[bool, str, Pcmd | None]


@dataclass(frozen=True)
class AutonomyEvent:
    kind: str
    detail: str = ""
    command: str | None = None
    result: object | None = None
    error: str | None = None


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
        pose_confidence: Callable[[], int],
        force_relocalize: Callable[[], None],
        stream_healthy: Callable[[], bool],
        takeoff: Callable[[], object],
        land: Callable[[], object],
        take_pc_control: Callable[[], object] | None = None,
        start_airborne: bool = False,
        max_route_deviation_map_units: float = pff.MAX_ROUTE_DEVIATION_U,
        arming_blockers: Callable[[], list[str]] | None = None,
        boot_timeout_s: float = pff.FIRST_FIX_TIMEOUT_S,
        boot_search_config: YawSearchConfig | None = None,
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
        route_deviation_limit = float(max_route_deviation_map_units)
        nudge_pct = int(getattr(backend, "nudge_pct", 10) or 10)
        nudge_pct = max(1, min(100, nudge_pct))
        if boot_search_config is None:
            boot_search_config = YawSearchConfig(yaw_pcmd=min(8, nudge_pct))
        elif boot_search_config.yaw_pcmd > nudge_pct:
            raise ValueError("AUTO yaw search command exceeds the configured nudge limit")
        base_config = rpf.ControlConfig(
            inspect_waypoints=(),
            map_frame=map_frame,
            max_translation_pcmd=min(10, nudge_pct),
            max_yaw_pcmd=min(20, nudge_pct),
            max_route_deviation=route_deviation_limit,
        )
        config = rpf.config_for_route(snapshot, base_config)

        self.backend = backend
        self.snapshot = snapshot
        self.waypoints = waypoints
        self.controller = rpf.RouteAutoController(
            waypoints, poles=[], config=config
        )
        self.get_pose = get_pose
        self.pose_is_weak = pose_is_weak
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
        self.worker_stall_timeout_s = stall_timeout
        self.now = now
        self.sleep = sleep
        self.run_loop = run_loop
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
        self._speed_guard_latched = False
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
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            thread.join(timeout=remaining)
        return all(
            thread is None
            or thread is threading.current_thread()
            or not thread.is_alive()
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
        self._speed_guard_latched = False
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
        self._clear_motion()
        self._send_zero(reason)
        return True

    def resume(self) -> bool:
        """Resume a paused run; the existing worker continues without takeoff."""
        if self.phase not in {"TAKEOFF", "HANDOFF", "BOOT_HOVER", "ROUTE"}:
            return False
        if not self._paused.is_set():
            return False
        self._paused.clear()
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
        monitor = getattr(self.backend, "_stick_monitor", None)
        is_active = getattr(monitor, "is_active", None)
        try:
            return bool(callable(is_active) and is_active())
        except Exception:
            return True

    def _safety_mode(self) -> str:
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
        self._clear_motion()
        self._send_zero(zero_reason)
        self._emit("auto_failed", self._auto_failure_detail, error=error)

    def _record_pcmd_failure(self, reason: str) -> None:
        self._pcmd_failure_streak += 1
        if self._pcmd_failure_streak < AUTO_PCMD_FAILURE_LIMIT:
            return
        self._latch_auto_failure(
            "AUTO PCMD failed "
            f"{self._pcmd_failure_streak} consecutive times; route terminated",
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
            self._send_zero("auto_stick_override")
            self._request_manual()
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
        if self._auto_failed:
            return (
                False,
                f"AUTO failed: {self._auto_failure_detail}",
                (0, 0, 0, 0),
            )
        if self._cancel.is_set():
            self._clear_motion()
            return False, f"AUTO cancelled: {self._cancel_reason}", (0, 0, 0, 0)
        if self._pilot_override():
            self._clear_motion()
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

        speed_guard_active, rejection = self._apply_speed_guard(pcmd)
        if rejection is not None:
            return rejection
        return self._dispatch_route_pcmd(
            pcmd,
            speed_guard_active=speed_guard_active,
        )

    def _apply_speed_guard(
        self,
        pcmd: Pcmd,
    ) -> tuple[bool, AuthorizedPcmdResult | None]:
        """Fail closed on horizontal motion when speed telemetry is not fresh."""

        if not bool(getattr(
            self.backend.state, "autonomous_speed_limit_enabled", True
        )):
            self._speed_guard_latched = False
            self.backend.state.autonomous_speed_guard_status = "SPEED_LIMIT_DISABLED"
            return False, None

        horizontal_motion = int(pcmd[0]) != 0 or int(pcmd[1]) != 0
        speed_limit = self._speed_guard_limit()
        if not math.isfinite(speed_limit) or speed_limit <= 0.0:
            return self._speed_guard_limit_unavailable(horizontal_motion)

        telemetry, rejection = self._speed_guard_telemetry(horizontal_motion)
        if rejection is not None:
            return False, rejection
        if telemetry is None:
            return False, None
        speed_mps, speed_guard_active = telemetry

        release_limit = speed_limit * AUTO_SPEED_GUARD_RELEASE_RATIO
        if self._speed_guard_latched:
            latched_result = self._speed_guard_latched_result(
                speed_mps,
                release_limit,
                horizontal_motion,
            )
            if latched_result is not None:
                return latched_result

        if speed_mps >= speed_limit:
            self._speed_guard_latched = True
            self.backend.state.autonomous_speed_guard_status = "OVERSPEED_HOVER"
            if horizontal_motion:
                return self._speed_guard_hover(
                    "OVERSPEED_HOVER",
                    f"AUTO speed limit {speed_limit:g} m/s reached "
                    f"({speed_mps:g} m/s) -> HOVER",
                )
            return True, None

        self.backend.state.autonomous_speed_guard_status = "FRESH_SPEED_GUARD"
        return speed_guard_active, None

    def _speed_guard_limit(self) -> float:
        try:
            raw_limit = self.backend.state.autonomous_speed_limit_mps
            if isinstance(raw_limit, bool):
                raise ValueError
            return float(raw_limit)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return float("nan")

    def _speed_guard_limit_unavailable(
        self,
        horizontal_motion: bool,
    ) -> tuple[bool, AuthorizedPcmdResult | None]:
        if horizontal_motion:
            return self._speed_guard_hover(
                "SPEED_UNAVAILABLE_HOVER",
                "AUTO speed limit unavailable -> HOVER",
            )
        self.backend.state.autonomous_speed_guard_status = "SPEED_LIMIT_UNAVAILABLE"
        return False, None

    def _speed_guard_telemetry(
        self,
        horizontal_motion: bool,
    ) -> tuple[tuple[float, bool] | None, AuthorizedPcmdResult | None]:
        speed_sample = self._ground_speed()
        if speed_sample is None:
            if horizontal_motion:
                _, rejection = self._speed_guard_hover(
                    "SPEED_UNAVAILABLE_HOVER",
                    "AUTO ground speed unavailable or stale -> HOVER",
                )
                return None, rejection
            self.backend.state.autonomous_speed_guard_status = "SPEED_UNAVAILABLE"
            return None, None

        speed_mps, speed_stamp = speed_sample
        speed_guard_active = 0.0 <= self.now() - speed_stamp <= GROUND_SPEED_MAX_AGE_S
        if not speed_guard_active:
            if horizontal_motion:
                _, rejection = self._speed_guard_hover(
                    "SPEED_UNAVAILABLE_HOVER",
                    "AUTO ground speed unavailable or stale -> HOVER",
                )
                return None, rejection
            self.backend.state.autonomous_speed_guard_status = "SPEED_STALE"
            return None, None
        return (speed_mps, speed_guard_active), None

    def _speed_guard_latched_result(
        self,
        speed_mps: float,
        release_limit: float,
        horizontal_motion: bool,
    ) -> tuple[bool, AuthorizedPcmdResult | None] | None:
        if speed_mps < release_limit:
            self._speed_guard_latched = False
            self.backend.state.autonomous_speed_guard_status = (
                "FRESH_SPEED_GUARD_RELEASED"
            )
            return None
        self.backend.state.autonomous_speed_guard_status = "OVERSPEED_LATCHED"
        if horizontal_motion:
            return self._speed_guard_hover(
                "OVERSPEED_LATCHED",
                f"AUTO speed guard latched until ground speed < "
                f"{release_limit:g} m/s -> HOVER",
            )
        return True, None

    def _speed_guard_hover(
        self,
        status: str,
        detail: str,
    ) -> tuple[bool, AuthorizedPcmdResult | None]:
        self.backend.state.autonomous_speed_guard_status = status
        self._clear_motion()
        self._send_zero("auto_speed_guard_hover")
        return False, (False, detail, (0, 0, 0, 0))

    @staticmethod
    def _route_pcmd_reason(accepted: bool, speed_guard_active: bool) -> str:
        if not accepted:
            return "AUTO PCMD rejected"
        if speed_guard_active:
            return "desktop AUTO fresh-speed guard + command cap"
        return "desktop AUTO command cap only (fresh speed unavailable)"

    def _dispatch_route_pcmd(
        self,
        pcmd: Pcmd,
        *,
        speed_guard_active: bool,
    ) -> AuthorizedPcmdResult:
        """Send through the backend's normalized vector seam or direct PCMD."""

        pct = max(1, int(getattr(self.backend, "nudge_pct", 10) or 10))
        vector = tuple(max(-1.0, min(1.0, float(value) / pct)) for value in pcmd)
        setter = getattr(self.backend, "set_nudge_vector", None)
        if callable(setter):
            try:
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
                self._route_pcmd_reason(accepted, speed_guard_active),
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
        reason = self._route_pcmd_reason(accepted, speed_guard_active)
        return accepted, reason, pcmd if accepted else None

    def _olympe_yaw(self) -> float | None:
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
            pose_confidence=self.pose_confidence,
            force_relocalize=self.force_relocalize,
            request_manual=self._request_manual,
            stick_active=self._pilot_override,
            safety_poll=self._safety_mode,
            stream_healthy=self.stream_healthy,
            stream_status=lambda: str(
                getattr(self.backend.state, "stream", "unknown")
            ),
            loop_beat=self._touch_heartbeat,
            pose_jump_pause=self._pause_for_pose_jump,
            localization_yaw_search_pcmd=self.boot_search_config.yaw_pcmd,
            localization_yaw_search_event=self._on_localization_yaw_search,
            # Localization loss never auto-lands the AUTO route: the loop holds
            # zero PCMD through the bounded yaw search until the pose recovers
            # or the operator takes over with the sticks. Physical stick
            # movement still invokes request_manual immediately.
            land_on_localization_loss=False,
            now=self.now,
        )

    def _land_after_failure(self, detail: str) -> None:
        self.phase = "LANDING"
        self._landing_confirmed = False
        self._clear_motion()
        self._send_zero("auto_pre_land")
        last_result = None
        last_error = None
        for attempt in range(1, LAND_ATTEMPTS + 1):
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
        if pose is None:
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
        accepted, _reason, _actual = self._send_authorized(
            (0, 0, int(yaw_pcmd), 0)
        )
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
        search_enabled = True
        last_search_state: str | None = None
        self._clear_motion()
        while True:
            self._touch_heartbeat()
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
            locked, last_pose_stamp, pose_valid = self._observe_boot_pose(
                lock, last_pose_stamp
            )
            if locked:
                blockers = tuple(self.arming_blockers())
                if not blockers:
                    position = lock.position
                    if position is None:
                        lock.reset()
                        continue
                    nearest = self.controller.start_after_nearest_waypoint(position)
                    target = self.controller.target_index
                    self._emit(
                        "route_start_selected",
                        f"takeoff pose joined at waypoint {nearest + 1}; "
                        f"first target is waypoint {target + 1}",
                    )
                    self._send_zero("auto_boot_pose_locked")
                    return None
                lock.reset()
                if blockers != last_arming_blockers:
                    self._emit(
                        "boot_hover_waiting",
                        "AUTO arming remains blocked: " + "; ".join(blockers)
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
                        "initial localization yaw search "
                        + step.state.replace("_", " ")
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
                    f"stable localization unavailable after "
                    f"{now - started:.1f}s; continuing hover",
                )
                next_wait_notice = now + self.boot_timeout_s
            self.sleep(0.1)

    def _run_route(self) -> str:
        self.phase = "ROUTE"
        self._touch_heartbeat()
        self._emit("route_started", "stable localization confirmed")
        reason = self.run_loop(
            self._route_hooks(),
            self.controller,
            self.waypoints,
            yaw_sign=1,
            verbose=False,
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
            else:
                self.phase = "DONE"
                self._emit("finished", terminal_detail)

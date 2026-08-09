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


@dataclass(frozen=True)
class AutonomyEvent:
    kind: str
    detail: str = ""
    command: str | None = None
    result: object | None = None
    error: str | None = None


class DesktopRouteAutonomy:
    """Take off, hold for a trustworthy pose, then run the route controller."""

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
        arming_blockers: Callable[[], list[str]] | None = None,
        boot_timeout_s: float = pff.FIRST_FIX_TIMEOUT_S,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        run_loop: Callable[..., str] = pff.run_loop,
    ):
        if not isinstance(map_frame, rpf.MapFrame):
            raise ValueError("AUTO requires the active site's measured map frame")
        timeout = float(boot_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("AUTO localization timeout must be finite and positive")

        snapshot.verify_file_unchanged()
        waypoints = snapshot.controller_waypoints()
        nudge_pct = int(getattr(backend, "nudge_pct", 10) or 10)
        nudge_pct = max(1, min(100, nudge_pct))
        base_config = rpf.ControlConfig(
            inspect_waypoints=(),
            map_frame=map_frame,
            max_translation_pcmd=min(10, nudge_pct),
            max_yaw_pcmd=min(20, nudge_pct),
        )
        config = rpf.config_for_route(snapshot.path, base_config)

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
        self.arming_blockers = arming_blockers or (lambda: [])
        self.boot_timeout_s = timeout
        self.now = now
        self.sleep = sleep
        self.run_loop = run_loop
        self.events: queue.Queue[AutonomyEvent] = queue.Queue()
        self._cancel = threading.Event()
        self._paused = threading.Event()
        self._cancel_reason = ""
        self._manual_handoff_attempted = False
        self._thread: threading.Thread | None = None
        self.phase = "IDLE"

    @property
    def active(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> bool:
        if self.active:
            return False
        self._cancel.clear()
        self._paused.clear()
        self._cancel_reason = ""
        self._manual_handoff_attempted = False
        self._thread = threading.Thread(
            target=self._run,
            name="desktop-route-autonomy",
            daemon=True,
        )
        self._thread.start()
        return True

    def cancel(self, reason: str) -> None:
        self._cancel_reason = str(reason or "operator")
        self._cancel.set()
        self._paused.clear()
        self._clear_motion()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self, reason: str = "operator_hover") -> bool:
        """Hold the active AUTO run in place without discarding its route state."""
        if self.phase not in {"TAKEOFF", "BOOT_HOVER", "ROUTE"}:
            return False
        self._paused.set()
        self._clear_motion()
        self._send_zero(reason)
        return True

    def resume(self) -> bool:
        """Resume a paused run; the existing worker continues without takeoff."""
        if self.phase not in {"TAKEOFF", "BOOT_HOVER", "ROUTE"}:
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

    def _emit(self, kind: str, detail: str = "", **fields: object) -> None:
        self.events.put(AutonomyEvent(kind=kind, detail=detail, **fields))

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
        if self._paused.is_set():
            return "HOVER"
        return "AUTO"

    def _operator_stop(self) -> bool:
        if self._cancel.is_set():
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

    def _send_authorized(self, pcmd: tuple[int, int, int, int]):
        if self._cancel.is_set():
            self._clear_motion()
            return False, f"AUTO cancelled: {self._cancel_reason}", (0, 0, 0, 0)
        if self._pilot_override():
            self._clear_motion()
            return False, "pilot stick override", None
        if self._paused.is_set():
            self._clear_motion()
            self._send_zero("auto_operator_pause")
            return False, "AUTO paused", (0, 0, 0, 0)
        if not self.stream_healthy():
            self._clear_motion()
            self._send_zero("auto_stream_unhealthy")
            return False, "AUTO stream unhealthy", (0, 0, 0, 0)

        pct = max(1, int(getattr(self.backend, "nudge_pct", 10) or 10))
        vector = tuple(max(-1.0, min(1.0, float(value) / pct)) for value in pcmd)
        setter = getattr(self.backend, "set_nudge_vector", None)
        if callable(setter):
            try:
                accepted = bool(setter(*vector))
            except Exception as exc:
                return False, f"AUTO PCMD failed: {exc!r}", None
            return (
                accepted,
                "desktop AUTO deadman refreshed" if accepted else "AUTO PCMD rejected",
                tuple(int(value) for value in pcmd) if accepted else None,
            )

        send = getattr(self.backend, "send_pcmd", None)
        try:
            accepted = bool(send(*pcmd, reason="desktop_auto_route"))
        except Exception as exc:
            return False, f"AUTO PCMD failed: {exc!r}", None
        return accepted, "desktop AUTO PCMD" if accepted else "AUTO PCMD rejected", (
            tuple(int(value) for value in pcmd) if accepted else None
        )

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
            stamp = int(state.telemetry_read_mono_ns) / 1_000_000_000.0
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(speed) or not math.isfinite(stamp):
            return None
        return speed, stamp

    def _request_manual(self) -> bool:
        give = getattr(self.backend, "give_to_pilot", None)
        if not callable(give):
            return False
        try:
            return bool(give(reason="auto_stick_override"))
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
            ground_speed=self._ground_speed,
            now=self.now,
        )

    def _land_after_failure(self, detail: str) -> None:
        self.phase = "LANDING"
        self._clear_motion()
        self._send_zero("auto_pre_land")
        result = None
        error = None
        try:
            result = self.land()
        except Exception as exc:
            error = repr(exc)
        self._emit(
            "command_result",
            detail,
            command="land",
            result=result,
            error=error,
        )

    def _run(self) -> None:
        airborne = False
        terminal_detail = "AUTO stopped"
        try:
            self.phase = "TAKEOFF"
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
            if takeoff_error is not None or takeoff_result is False:
                terminal_detail = "AUTO takeoff failed"
                return
            airborne = True
            if self._operator_stop():
                terminal_detail = "AUTO cancelled after takeoff"
                return

            self.phase = "BOOT_HOVER"
            self._emit(
                "boot_hover",
                f"waiting up to {self.boot_timeout_s:g}s for stable localization",
            )
            lock = pff.BootPoseLock(self.waypoints[0])
            started = self.now()
            last_relocalize = float("-inf")
            last_pose_stamp: float | None = None
            while True:
                if self._operator_stop():
                    terminal_detail = "AUTO cancelled during localization hover"
                    return
                if self._paused.is_set():
                    paused_at = self.now()
                    while self._paused.is_set():
                        if self._operator_stop():
                            terminal_detail = "AUTO cancelled while paused"
                            return
                        self._send_zero("auto_operator_pause")
                        self.sleep(0.1)
                    started += max(0.0, self.now() - paused_at)
                    continue
                self._send_zero("auto_boot_hover")
                pose = self.get_pose()
                yaw_ready = bool(
                    pose is not None and math.isfinite(float(pose.yaw))
                )
                if yaw_ready:
                    pose_stamp = float(pose.stamp)
                    if last_pose_stamp is None or pose_stamp > last_pose_stamp:
                        last_pose_stamp = pose_stamp
                        if lock.observe(pose, now=self.now()):
                            break
                now = self.now()
                if now - last_relocalize >= 0.5:
                    try:
                        self.force_relocalize()
                    except Exception:
                        pass
                    last_relocalize = now
                if now - started >= self.boot_timeout_s:
                    terminal_detail = "stable localization unavailable after takeoff hover"
                    self._land_after_failure(terminal_detail)
                    return
                self.sleep(0.1)

            blockers = list(self.arming_blockers())
            if blockers:
                terminal_detail = "AUTO arming blocked: " + "; ".join(blockers)
                self._land_after_failure(terminal_detail)
                return

            self.phase = "ROUTE"
            self._emit("route_started", "stable localization confirmed")
            reason = self.run_loop(
                self._route_hooks(),
                self.controller,
                self.waypoints,
                yaw_sign=1,
                verbose=False,
            )
            terminal_detail = str(reason or "AUTO route ended")
            if self._operator_stop():
                return
            self._land_after_failure(terminal_detail)
        except BaseException as exc:  # safety cleanup must cover worker failures
            terminal_detail = f"AUTO worker failed: {exc!r}"
            if airborne and not self._operator_stop():
                self._land_after_failure(terminal_detail)
        finally:
            self._clear_motion()
            self._paused.clear()
            self.phase = "DONE"
            self._emit("finished", terminal_detail)

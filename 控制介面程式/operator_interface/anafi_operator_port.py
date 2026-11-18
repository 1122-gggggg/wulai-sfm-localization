"""ANAFI's normalized operator-facing port, separate from flight-control policy.

OlympeLiveBackend supplies the existing safety machinery. This base owns only
read-only presentation, explicit connection entry, and typed request admission.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from anafi_connection import connect_or_recover, runtime_switch_status
from backend_contract import (
    ControlRequest, ControlResult, RuntimeSwitchStatus,
    execute_admitted_control, gimbal_observation,
)


class AnafiOperatorPort:
    """Public adapter operations implemented once for the current aircraft."""

    @property
    def connection_target(self) -> str:
        return str(self.ip)

    def flight_state_name(self) -> str:
        """Normalize the adapter's firmware cache read, not UI tracker state."""
        return self._flight_state_name() or "unknown"

    @property
    def maneuver_in_progress(self) -> str | None:
        return self._maneuver_in_progress

    @property
    def runtime_safety_latched(self) -> bool:
        return self._runtime_safety_action_latched

    def pilot_override_active(self) -> bool:
        if self.pilot_sticks:
            return True
        is_active = getattr(self._stick_monitor, "is_active", None)
        try:
            return bool(callable(is_active) and is_active())
        except Exception:
            return True

    def gimbal_pitch_sample(self) -> tuple[float, int] | None:
        """Return an atomic hardware event observation, never a cache poll stamp."""
        sample = self.telemetry_freshness.sample("gimbal")
        if (
            sample is None or not isinstance(sample.marker, tuple)
            or not sample.marker or sample.marker[0] != "event"
            or not isinstance(sample.value, dict)
            or sample.value.get("gimbal_id", 0) != 0
        ):
            return None
        return gimbal_observation(sample.value.get("pitch_absolute"), sample.observed_mono_ns)

    def runtime_switch_status(self) -> RuntimeSwitchStatus:
        return runtime_switch_status(self)

    def connect_live(self) -> None:
        """Explicitly connect/recover a ground session on its persistent SDK owner."""
        # Share shutdown's lock: closing during a pending connect must wait for
        # that connection and then apply the existing landing/cleanup policy.
        with self._cleanup_lock:
            if self._shutdown_latched or self._cleanup_done:
                raise RuntimeError("backend is shutting down")
            self._connection_attempted = True
            self.connection_in_progress = True
            self.state.stream = "CONNECTING"
            try:
                # Olympe loops monitor their creating thread. A one-shot UI
                # worker would disconnect them as soon as connection returns.
                if self._connection_executor is None:
                    self._connection_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="olympe-connection"
                    )
                self._connection_executor.submit(connect_or_recover, self).result()
                if self.with_video and self.video_stream is None:
                    raise RuntimeError("真機已連線，但串流啟動失敗，請再次連接")
            except BaseException as exc:
                self.log.event("connect_failed", error=str(exc) or repr(exc))
                if self.drone is None:
                    self.state.stream = "DISCONNECTED"
                    self.state.link_ok = False
                    self.state.link_status = "DISCONNECTED"
                raise
            finally:
                self.connection_in_progress = False

    def _command_typed_request(self, request: ControlRequest) -> ControlResult:
        self.log.event(
            "control_request",
            request_id=request.request_id,
            action=request.action.value,
            human_origin=request.human_origin,
            submitted_mono_ns=request.submitted_mono_ns,
        )
        result = execute_admitted_control(request, self.state, self._typed_command)
        self.log.event(
            "control_result",
            request_id=request.request_id,
            action=request.action.value,
            accepted=result.accepted,
            executed=result.executed,
            reason_code=result.reason_code,
            control_owner=getattr(self.state, "control_owner", None),
        )
        return result


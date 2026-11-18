"""Read-only link evidence and operator-triggered ANAFI connection recovery.

No background retry, takeoff, landing, or cleanup policy lives here. A connected
unknown/airborne aircraft is never disconnected to repair a desktop session.
"""
from __future__ import annotations

from backend_contract import RuntimeSwitchKind, RuntimeSwitchStatus


def _state_name(value) -> str:
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].lower()


def _link_readback(backend) -> tuple[str, str]:
    """Same caches as _probe_link_ok, without equating unreadable with lost."""
    try:
        host = _state_name(backend.drone.connection_state())
    except Exception:
        return "unknown", "host link readback unavailable"
    if host != "connected":
        return host, f"host link state: {host}"
    if not backend.via_skycontroller():
        return "connected", "direct aircraft connected"
    try:
        from olympe.messages.drone_manager import connection_state

        managed = backend._state_dict(connection_state)
        raw = managed.get("state") if managed is not None else None
        aircraft = _state_name(raw)
    except Exception:
        return "unknown", "aircraft link readback unavailable"
    return aircraft, f"controller aircraft link state: {aircraft}"


def connection_evidence(backend) -> RuntimeSwitchStatus:
    """Inspect caches without SDK waits; caller serializes connection attempts."""
    target = backend.connection_target
    if backend.drone is None:
        kind = (
            RuntimeSwitchKind.UNSAFE if backend._connection_attempted
            else RuntimeSwitchKind.NEVER_CONNECTED
        )
        return RuntimeSwitchStatus(kind, "no confirmed transport", target)
    link, reason = _link_readback(backend)
    if link == "disconnected":
        return RuntimeSwitchStatus(RuntimeSwitchKind.PROVEN_DISCONNECTED, reason, target)
    if link != "connected":
        return RuntimeSwitchStatus(RuntimeSwitchKind.UNSAFE, reason, target)
    flight = backend.flight_state_name()
    kind = RuntimeSwitchKind.LANDED if flight == "landed" else RuntimeSwitchKind.UNSAFE
    return RuntimeSwitchStatus(kind, "connected aircraft flight-state readback", target, flight)


def runtime_switch_status(backend) -> RuntimeSwitchStatus:
    if backend.connection_in_progress:
        return RuntimeSwitchStatus(
            RuntimeSwitchKind.UNSAFE, "connection in progress", backend.connection_target,
        )
    return connection_evidence(backend)


def retry_ground_stick_monitor(backend) -> bool:
    """One explicit retry; preserve manual/safety intent during USB enumeration."""
    if backend._stick_monitor_ready():
        return True
    with backend._lock:
        epoch = backend._pulse_token
        if backend._shutdown_latched or backend._cleanup_done:
            return False
    if connection_evidence(backend).kind is not RuntimeSwitchKind.LANDED:
        return False
    backend._start_stick_monitor()
    with backend._lock:
        if epoch != backend._pulse_token or backend._shutdown_latched or backend._cleanup_done:
            return False
    return (
        backend._stick_monitor_ready()
        and connection_evidence(backend).kind is RuntimeSwitchKind.LANDED
    )


def _release_disconnected_transport(backend) -> None:
    # Retire stale movement before any future session can publish commands.
    backend._latch_total_link_loss()
    backend.nudge_clear(reason="explicit_reconnect")
    backend._stop_stick_monitor()
    if backend.motion_history is not None:
        backend.motion_history.detach()
        backend.motion_history = None
    if backend.grabber is not None:
        backend.grabber.stop()
        backend.grabber = None
    backend.video = None
    backend.video_stream = None
    if backend.drone.disconnect() is False:
        raise RuntimeError("stale SDK session could not be released")
    backend.drone = None
    backend._connection_ready = False


def _refresh_connected_ground(backend) -> None:
    # No initial auto-PC acquisition and no replacement of a healthy video stream.
    if backend.via_skycontroller() and not backend._stick_monitor_ready():
        if not retry_ground_stick_monitor(backend):
            raise RuntimeError("controller stick monitor unavailable; PC control remains blocked")
        backend.read_connection_inventory()
    if backend.with_video and backend.video_stream is None:
        backend._start_video()
    backend.state.stream = "OK" if backend.video_stream is not None or not backend.with_video else "NO_VIDEO"


def connect_or_recover(backend) -> None:
    """Run once on the persistent SDK owner, under the existing cleanup lock."""
    if backend.drone is not None:
        evidence = connection_evidence(backend)
        if evidence.kind is RuntimeSwitchKind.PROVEN_DISCONNECTED:
            _release_disconnected_transport(backend)
        elif evidence.kind is not RuntimeSwitchKind.LANDED:
            raise RuntimeError(
                f"reconnect refused: connected airborne or unknown state ({evidence.reason})"
            )
    if not backend._connection_ready or backend.drone is None:
        backend._connect()
        backend._connection_ready = True
        return
    _refresh_connected_ground(backend)

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from types import SimpleNamespace

import pytest

import path_follow_flight as pff
from backend_contract import (
    ControlAction,
    ControlRequest,
    InterfaceMode,
    SessionConfig,
)
from flight_operator_app import DroneBackend, OperatorApp
from operator_autonomy import DesktopRouteAutonomy
from operator_command_coordinator import OperatorCommandCoordinator
from operator_localization_search import YawSearchConfig
from operator_shutdown import OperatorShutdownCoordinator
from real_path_follow_controller import (
    LEGACY_MAP_FRAME,
    MissionRouteSnapshot,
    Pose,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def now(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += float(duration)


class _ScenarioBackend:
    """Pure recording backend for AUTO authority and safety transitions."""

    mode = InterfaceMode.SIMULATED_STREAM
    is_live = False
    nudge_pct = 10

    def __init__(self, *, manual_available: bool = True) -> None:
        self.manual_available = manual_available
        self.pilot_sticks = False
        self._runtime_safety_action_latched = False
        self.pcmds: list[tuple[int, int, int, int, str]] = []
        self.nudge_vectors: list[tuple[float, float, float, float]] = []
        self.handoffs: list[str] = []
        self.cleanup_calls = 0
        self.state = SimpleNamespace(
            att_yaw=0.0,
            stream="OK",
            autonomous_speed_limit_mps=0.30,
            ground_speed_mps=0.0,
            ground_speed_mono_ns=10_000_000_000,
            telemetry_read_mono_ns=10_000_000_000,
        )

    def send_pcmd(
        self,
        roll: int,
        pitch: int,
        yaw: int,
        gaz: int,
        *,
        reason: str,
    ) -> bool:
        self.pcmds.append((roll, pitch, yaw, gaz, reason))
        if yaw:
            # The mock supplies attitude feedback so BoundedYawSearch can
            # demonstrate both sides of its configured sweep.
            self.state.att_yaw += math.radians(float(yaw) * 2.0)
        return True

    def set_nudge_vector(
        self, roll: float, pitch: float, yaw: float, gaz: float
    ) -> bool:
        self.nudge_vectors.append((roll, pitch, yaw, gaz))
        if yaw:
            self.state.att_yaw += math.radians(
                float(yaw) * self.nudge_pct * 2.0
            )
        return True

    def clear_nudge_vector(self) -> None:
        self.nudge_vectors.append((0.0, 0.0, 0.0, 0.0))

    def give_to_pilot(self, *, reason: str) -> bool:
        self.handoffs.append(reason)
        if self.manual_available:
            self.pilot_sticks = True
        return self.manual_available

    def cleanup(self) -> bool:
        self.cleanup_calls += 1
        self.send_pcmd(0, 0, 0, 0, reason="simulation_cleanup")
        return True


def _snapshot(tmp_path, *, waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))):
    route = tmp_path / "simulation-route.json"
    route.write_text(
        json.dumps({"waypoints": waypoints, "frame": "glomap", "units": "map"}),
        encoding="utf-8",
    )
    return MissionRouteSnapshot(
        path=route,
        sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
        site_id="simulation-site",
        coordinate_frame_id="simulation-frame",
        waypoints=tuple(tuple(float(value) for value in point) for point in waypoints),
    )


def _sim_config() -> SessionConfig:
    return SessionConfig(
        session_id="simulation-session",
        interface_mode=InterfaceMode.SIMULATED_STREAM,
        site_profile="/tmp/simulation-site.json",
        site_profile_sha256="a" * 64,
        asset_sha256={},
        runtime_profile_sha256="b" * 64,
        source="mock-video",
        offline=True,
    )


def _coordinator(backend: object) -> tuple[OperatorCommandCoordinator, queue.Queue]:
    results: queue.Queue = queue.Queue()
    coordinator = OperatorCommandCoordinator(
        backend=backend,
        normal_results=results,
        safety_results=queue.Queue(),
        inflight=set(),
        inflight_lock=threading.Lock(),
        publish_lock=threading.Lock(),
        write_log=None,
        record_drop=None,
        safety_commands={"land", "emergency_stop"},
    )
    return coordinator, results


def test_manual_takeoff_command_lifecycle_is_typed_and_simulated() -> None:
    backend = DroneBackend()
    assert backend.start(_sim_config()).started
    coordinator, results = _coordinator(backend)

    assert coordinator.dispatch("takeoff", {})
    assert results.get(timeout=1.0) == ("takeoff", True, None)
    assert backend.state.tracker_state == "TAKEOFF"

    # Move the mock vehicle directly to its hover altitude; no SDK or motor is
    # involved in this lifecycle check.
    backend.sim_xyz[1] = -1.0
    backend.poll()
    assert backend.state.tracker_state == "HOVER"
    assert backend.state.altitude_m == pytest.approx(1.0)

    landed = backend.command(
        ControlRequest.create(ControlAction.LAND_NOW, human_origin=True)
    )
    assert landed.accepted and landed.executed
    assert backend.state.tracker_state == "LAND"
    backend.sim_xyz[1] = 0.0
    backend.poll()
    assert backend.state.tracker_state == "HOVER"


def test_simulated_route_test_runs_the_production_controller_on_drawn_waypoints(
    tmp_path,
) -> None:
    backend = DroneBackend()
    snapshot = _snapshot(
        tmp_path,
        waypoints=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.0, 0.5)),
    )
    plant = backend.route_test_plant
    assert plant.begin(snapshot, LEGACY_MAP_FRAME)

    autonomy = DesktopRouteAutonomy(
        backend=plant,
        snapshot=snapshot,
        map_frame=LEGACY_MAP_FRAME,
        get_pose=plant.pose,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: False,
        take_pc_control=lambda: True,
        start_airborne=True,
        land=plant.finish,
    )

    assert autonomy.start()
    assert autonomy.join(timeout=12.0)
    assert autonomy.phase == "DONE"
    assert plant.finished
    assert backend.state.flight_state == "landed"
    assert backend.sim_xyz == pytest.approx(snapshot.waypoints[-1], abs=0.08)


def test_simulated_auto_route_is_not_blocked_by_real_flight_preflight() -> None:
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = DroneBackend()
    operator.preflight_guide = SimpleNamespace(complete=False, current_step="compass")
    operator.write_log = lambda _message: None

    assert not OperatorApp._preflight_blocks_flight_command(operator, "start_auto")


def test_saved_route_callback_starts_only_the_bound_simulated_route(tmp_path) -> None:
    route = _snapshot(tmp_path)
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = DroneBackend()
    operator.mission_route_lock = SimpleNamespace(snapshot=route)
    statuses = []
    commands = []
    operator.site_assets_panel = SimpleNamespace(set_status=statuses.append)
    operator.send = commands.append
    operator._integrated_auto_active = lambda: True

    assert OperatorApp.start_saved_route_test(operator, route.path)
    assert commands == ["start_auto"]
    assert statuses == ["航線已儲存；正在用正式控制器執行純模擬航線…"]

    commands.clear()
    assert not OperatorApp.start_saved_route_test(operator, tmp_path / "other.json")
    assert commands == []


class _RouteLock:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.active = False

    def begin_auto(self, *, displayed_sha256):
        assert displayed_sha256 == self.snapshot.sha256
        self.active = True
        return self.snapshot

    def cancel_rejected_auto_start(self) -> None:
        self.active = False


def test_switching_into_auto_clears_manual_nonzero_input_before_activation(
    tmp_path,
) -> None:
    backend = _ScenarioBackend()
    backend.is_live = True
    backend.mode = InterfaceMode.REAL_FLIGHT
    backend.nudge_vectors.append((0.6, 0.0, 0.0, 0.0))
    route = _snapshot(tmp_path)
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = backend
    operator._site_switching = False
    operator._runtime_available = True
    operator._integrated_autonomy = None
    operator._auto_paused = False
    operator.inspecting = True
    operator.mission_route_lock = _RouteLock(route)
    operator._displayed_route_sha256 = route.sha256
    operator._pending_auto_route_activation = False
    operator._nudge_keys_held = {"w"}
    operator._nudge_buttons_held = {"前"}
    operator._stick_vector_active = True
    operator.write_log = lambda _message: None
    operator._preflight_blocks_flight_command = lambda _command: False
    operator._backend_command = (
        lambda command, payload=None: backend.command(command, **(payload or {}))
    )
    started = []
    operator._start_integrated_auto = lambda snapshot: started.append(snapshot) or True

    OperatorApp.send(operator, "start_auto")

    assert started == [route]
    assert operator._nudge_keys_held == set()
    assert operator._nudge_buttons_held == set()
    assert operator._stick_vector_active is False
    assert backend.nudge_vectors[-1] == (0.0, 0.0, 0.0, 0.0)


def test_initial_missing_localization_search_is_bounded_yaw_only_and_stops_on_lock(
    tmp_path,
) -> None:
    clock = _Clock()
    backend = _ScenarioBackend()
    commands: list[str] = []
    route_started_at: list[int] = []

    def pose() -> Pose | None:
        if clock.now() < 10.8:
            return None
        return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())

    def run_loop(*_args, **_kwargs):
        route_started_at.append(len(backend.pcmds))
        return "simulation route complete"

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=pose,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: commands.append("takeoff") or True,
        land=lambda: commands.append("land") or True,
        boot_timeout_s=4.0,
        boot_search_config=YawSearchConfig(
            wait_before_search_s=0.1,
            max_search_s=3.0,
            sweep_angle_deg=20.0,
            yaw_pcmd=6,
            tick_s=0.1,
        ),
        now=clock.now,
        sleep=clock.sleep,
        run_loop=run_loop,
    )

    autonomy.start()
    assert autonomy.join(timeout=1.0)
    search = [
        vector
        for vector in backend.nudge_vectors
        if any(abs(value) > 0.0 for value in vector)
    ]
    assert any(vector[2] > 0 for vector in search)
    assert any(vector[2] < 0 for vector in search)
    assert all(
        vector[:2] == (0.0, 0.0) and vector[3] == 0.0 and abs(vector[2]) <= 0.6
        for vector in search
    )
    assert route_started_at
    assert all(command[:4] == (0, 0, 0, 0) for command in backend.pcmds[route_started_at[0] :])
    assert commands == ["takeoff", "land"]


@pytest.mark.parametrize("manual_available", [True, False])
def test_missing_localization_without_yaw_telemetry_never_rotates_while_hovering(
    tmp_path,
    manual_available: bool,
) -> None:
    clock = _Clock()
    backend = _ScenarioBackend(manual_available=manual_available)
    backend.state.att_yaw = float("nan")
    commands: list[str] = []
    autonomy = None

    def sleep_then_stop(duration: float) -> None:
        clock.sleep(duration)
        if clock.now() >= 10.7:
            autonomy._cancel.set()

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 0,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: commands.append("takeoff") or True,
        land=lambda: commands.append("land") or True,
        boot_timeout_s=0.3,
        boot_search_config=YawSearchConfig(
            wait_before_search_s=0.1,
            max_search_s=1.0,
            sweep_angle_deg=20.0,
            yaw_pcmd=6,
        ),
        now=clock.now,
        sleep=sleep_then_stop,
        run_loop=lambda *_args, **_kwargs: "must not run",
    )

    autonomy._run()

    assert clock.now() - 10.0 <= 0.81
    assert backend.pcmds
    assert all(command[:4] == (0, 0, 0, 0) for command in backend.pcmds)
    assert any(
        event.kind == "boot_hover_search_waiting"
        and "without blind rotation" in event.detail
        for event in autonomy.drain_events()
    )
    assert commands == ["takeoff"]
    assert backend.handoffs == []


def test_airborne_auto_handoff_never_repeats_takeoff(tmp_path) -> None:
    clock = _Clock()
    backend = _ScenarioBackend()
    backend.pilot_sticks = True
    commands: list[str] = []

    def take_pc_control() -> bool:
        commands.append("pc_control")
        backend.pilot_sticks = False
        return True

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: commands.append("takeoff") or True,
        take_pc_control=take_pc_control,
        start_airborne=True,
        land=lambda: commands.append("land") or True,
        boot_timeout_s=1.0,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: "route complete",
    )

    autonomy._run()

    assert commands == ["pc_control", "land"]
    command_events = [
        event for event in autonomy.drain_events() if event.kind == "command_result"
    ]
    assert [(event.command, event.result) for event in command_events] == [
        ("pc_control", True),
        ("land", True),
    ]


@pytest.mark.parametrize("manual_available", [True, False])
def test_auto_localization_loss_hovers_without_timer_handoff_or_land(
    tmp_path,
    monkeypatch,
    manual_available: bool,
) -> None:
    clock = _Clock()
    backend = _ScenarioBackend(manual_available=manual_available)
    commands: list[str] = []
    pose_calls = 0
    loss_started = threading.Event()
    hover_seen = threading.Event()
    send_pcmd = backend.send_pcmd
    set_nudge_vector = backend.set_nudge_vector
    clear_nudge_vector = backend.clear_nudge_vector

    def record_pcmd(roll, pitch, yaw, gaz, *, reason):
        result = send_pcmd(roll, pitch, yaw, gaz, reason=reason)
        if loss_started.is_set() and (roll, pitch, yaw, gaz) == (0, 0, 0, 0):
            hover_seen.set()
        return result

    def record_nudge(roll, pitch, yaw, gaz):
        result = set_nudge_vector(roll, pitch, yaw, gaz)
        if loss_started.is_set() and not any(
            abs(value) > 0.0 for value in (roll, pitch, yaw, gaz)
        ):
            hover_seen.set()
        return result

    def record_clear():
        clear_nudge_vector()
        if loss_started.is_set():
            hover_seen.set()

    backend.send_pcmd = record_pcmd
    backend.set_nudge_vector = record_nudge
    backend.clear_nudge_vector = record_clear

    # Keep the production loop and authority hooks, but make time deterministic.
    monkeypatch.setattr(pff, "POSE_HOLDOVER_S", 0.0)
    monkeypatch.setattr(pff, "LOST_LAND_S", 0.0)
    monkeypatch.setattr(pff.time, "sleep", clock.sleep)

    def pose() -> Pose | None:
        nonlocal pose_calls
        pose_calls += 1
        if pose_calls <= 8:
            return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())
        loss_started.set()
        return None

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=pose,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: commands.append("takeoff") or True,
        land=lambda: commands.append("land") or True,
        boot_timeout_s=1.0,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=pff.run_loop,
    )

    assert autonomy.start()
    assert hover_seen.wait(timeout=2.0), (
        pose_calls,
        autonomy.phase,
        commands,
        backend.pcmds[-5:],
        backend.nudge_vectors[-5:],
    )

    assert backend.pcmds
    assert all(
        command[0] == 0 and command[1] == 0 and command[3] == 0
        for command in backend.pcmds
    )
    assert commands == ["takeoff"]
    assert backend.handoffs == []
    assert not any(
        event.kind == "landing_unresolved" for event in autonomy.drain_events()
    )

    autonomy.cancel("test complete")
    assert autonomy.join(timeout=1.0)
    assert autonomy.phase == "DONE"


class _CommandGate:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def suspend(self) -> None:
        self.calls.append("suspend")

    def resume(self) -> None:
        self.calls.append("resume")


def test_shutdown_cancels_auto_with_bounded_join_and_zero_pcmd(tmp_path) -> None:
    clock = _Clock()
    backend = _ScenarioBackend()
    entered = threading.Event()
    autonomy = None

    def run_loop(hooks, *_args, **_kwargs):
        entered.set()
        while not autonomy._cancel.is_set():
            hooks.send_authorized_pcmd((5, 0, 0, 0))
            time.sleep(0.001)
        return "simulation cancelled"

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=run_loop,
    )
    gate = _CommandGate()
    destroyed: list[bool] = []
    shutdown = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=None,
        write_log=None,
        destroy=lambda: destroyed.append(True),
        command_coordinator=gate,
        autonomy=autonomy,
    )

    autonomy.start()
    assert entered.wait(timeout=1.0)
    started = time.monotonic()
    assert shutdown.shutdown(reason="simulation_close")
    assert time.monotonic() - started < 1.0
    assert not autonomy.active
    assert backend.cleanup_calls == 1
    assert destroyed == [True]
    assert gate.calls == ["suspend"]
    assert backend.pcmds
    assert all(command[:4] == (0, 0, 0, 0) for command in backend.pcmds)

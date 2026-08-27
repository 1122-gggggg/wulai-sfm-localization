from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from operator_autonomy import AUTO_PCMD_FAILURE_LIMIT, DesktopRouteAutonomy
from operator_localization_search import YawSearchConfig
import path_follow_flight as pff
import real_path_follow_controller as rpf
from real_path_follow_controller import LEGACY_MAP_FRAME, MissionRouteSnapshot, Pose


class _Clock:
    def __init__(self):
        self.value = 10.0

    def now(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class _Backend:
    def __init__(self):
        self.pilot_sticks = False
        self._runtime_safety_action_latched = False
        self.nudge_pct = 10
        self.zeros = []
        self.vectors = []
        self.state = SimpleNamespace(
            att_yaw=0.0,
            ground_speed_mps=0.0,
            ground_speed_mono_ns=10_000_000_000,
            telemetry_read_mono_ns=10_000_000_000,
            autonomous_speed_limit_enabled=True,
            autonomous_speed_limit_mps=0.30,
            stream="OK",
        )

    def send_pcmd(self, roll, pitch, yaw, gaz, *, reason):
        self.zeros.append((roll, pitch, yaw, gaz, reason))
        return True

    def set_nudge_vector(self, roll, pitch, yaw, gaz):
        self.vectors.append((roll, pitch, yaw, gaz))
        return True

    def clear_nudge_vector(self):
        self.vectors.append((0.0, 0.0, 0.0, 0.0))

    def give_to_pilot(self, *, reason):
        self.pilot_sticks = True
        return True


def _snapshot(
    tmp_path,
    *,
    max_route_deviation_map_units: float | None = None,
    waypoints=None,
) -> MissionRouteSnapshot:
    route_waypoints = (
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        if waypoints is None
        else tuple(tuple(float(value) for value in point) for point in waypoints)
    )
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps({
            "waypoints": route_waypoints,
            "frame": "glomap",
            "units": "map",
        }),
        encoding="utf-8",
    )
    digest = hashlib.sha256(route.read_bytes()).hexdigest()
    return MissionRouteSnapshot(
        path=route,
        sha256=digest,
        site_id="field-a",
        coordinate_frame_id="glomap-a",
        waypoints=route_waypoints,
        max_route_deviation_map_units=max_route_deviation_map_units,
    )


@pytest.mark.parametrize(
    ("route_limit", "site_limit", "expected"),
    [(0.08, 0.1, 0.08), (0.08, 0.05, 0.05)],
)
def test_desktop_auto_uses_the_tighter_route_or_site_deviation_limit(
    tmp_path,
    route_limit,
    site_limit,
    expected,
) -> None:
    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=_snapshot(
            tmp_path,
            max_route_deviation_map_units=route_limit,
        ),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        max_route_deviation_map_units=site_limit,
    )

    assert autonomy.controller.cfg.max_route_deviation == pytest.approx(expected)


def test_desktop_auto_keeps_route_and_limits_in_raw_map_frame(tmp_path) -> None:
    snapshot = _snapshot(
        tmp_path,
        max_route_deviation_map_units=0.08,
        waypoints=((0, 0, 0), (1, 2, 3)),
    )
    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=snapshot,
        map_frame=rpf.MapFrame(
            east=np.array([1.0, 0.0, 0.0]),
            north=np.array([0.0, 1.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
            source="site-enu",
        ),
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        max_route_deviation_map_units=0.1,
    )

    assert np.asarray(autonomy.waypoints) == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    )
    assert autonomy.controller.cfg.max_route_deviation == pytest.approx(0.08)


def test_cancel_join_tracks_the_async_manual_handoff(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingHandoffBackend(_Backend):
        def give_to_pilot(self, *, reason):
            entered.set()
            release.wait(timeout=1.0)
            return super().give_to_pilot(reason=reason)

    autonomy = DesktopRouteAutonomy(
        backend=BlockingHandoffBackend(),
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )

    autonomy.cancel("test")
    assert entered.wait(timeout=1.0)
    assert autonomy.join(timeout=0.01) is False
    release.set()
    assert autonomy.join(timeout=1.0) is True


def test_worker_stall_watchdog_latches_failure_and_join_cleans_up(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()
    clock = _Clock()
    land = Mock(return_value=True)
    autonomy = None

    def stalled_route(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=1.0)
        return "released after watchdog"

    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=land,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=stalled_route,
        worker_stall_timeout_s=0.05,
    )

    assert autonomy.start()
    assert entered.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while autonomy.phase != "AUTO_FAILED" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert autonomy.phase == "AUTO_FAILED"

    release.set()
    assert autonomy.join(timeout=1.0)
    assert autonomy._watchdog_thread is not None
    assert not autonomy._watchdog_thread.is_alive()
    assert land.call_count == 0
    events = autonomy.drain_events()
    assert any(event.kind == "auto_failed" for event in events)
    assert not any(event.kind == "finished" for event in events)
    assert any(entry[:4] == (0, 0, 0, 0) for entry in autonomy.backend.zeros)


def test_worker_watchdog_does_not_fire_while_route_paused(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()
    clock = _Clock()
    autonomy = None

    def stalled_route(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=1.0)
        return "released while paused"

    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
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
        run_loop=stalled_route,
        worker_stall_timeout_s=0.05,
    )

    assert autonomy.start()
    assert entered.wait(timeout=1.0)
    assert autonomy.pause()
    time.sleep(0.15)
    assert autonomy.phase == "ROUTE"
    assert not any(event.kind == "auto_failed" for event in autonomy.drain_events())

    autonomy.cancel("paused watchdog test")
    release.set()
    assert autonomy.join(timeout=1.0)
    assert autonomy.phase == "DONE"


def test_worker_watchdog_does_not_fire_during_takeoff_or_landing(tmp_path) -> None:
    clock = _Clock()
    takeoff_entered = threading.Event()
    takeoff_release = threading.Event()
    land_entered = threading.Event()
    land_release = threading.Event()
    autonomy = None

    def slow_takeoff():
        takeoff_entered.set()
        takeoff_release.wait(timeout=1.0)
        return True

    def slow_land():
        land_entered.set()
        land_release.wait(timeout=1.0)
        return True

    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=slow_takeoff,
        land=slow_land,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: "route complete",
        worker_stall_timeout_s=0.05,
    )

    assert autonomy.start()
    assert takeoff_entered.wait(timeout=1.0)
    time.sleep(0.15)
    assert autonomy.phase == "TAKEOFF"
    assert not any(event.kind == "auto_failed" for event in autonomy.drain_events())
    takeoff_release.set()
    assert land_entered.wait(timeout=1.0)
    time.sleep(0.15)
    assert autonomy.phase == "LANDING"
    assert not any(event.kind == "auto_failed" for event in autonomy.drain_events())
    land_release.set()
    assert autonomy.join(timeout=1.0)
    assert autonomy.phase == "DONE"


def test_no_localization_after_auto_takeoff_keeps_hovering_without_handoff(
    tmp_path,
) -> None:
    clock = _Clock()
    backend = _Backend()
    commands = []
    autonomy = None

    def sleep_then_stop(duration):
        clock.sleep(duration)
        if clock.now() >= 10.7:
            autonomy._cancel.set()

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: commands.append("takeoff") or True,
        land=lambda: commands.append("land") or True,
        boot_timeout_s=0.3,
        now=clock.now,
        sleep=sleep_then_stop,
        run_loop=lambda *_args, **_kwargs: "must not run",
    )

    autonomy._run()

    assert commands == ["takeoff"]
    assert backend.zeros
    assert all(entry[:4] == (0, 0, 0, 0) for entry in backend.zeros)
    events = autonomy.drain_events()
    assert any(event.kind == "boot_hover" for event in events)
    assert any(event.kind == "boot_hover_waiting" for event in events)
    assert not any(event.kind == "route_started" for event in events)
    assert backend.pilot_sticks is False


def test_initial_localization_timeout_does_not_land_when_handoff_is_unavailable(
    tmp_path,
) -> None:
    class NoPilotBackend(_Backend):
        def give_to_pilot(self, *, reason):
            return False

    clock = _Clock()
    backend = NoPilotBackend()
    commands = []
    autonomy = None

    def sleep_then_stop(duration):
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
        now=clock.now,
        sleep=sleep_then_stop,
        run_loop=lambda *_args, **_kwargs: "must not run",
    )

    autonomy._run()

    assert commands == ["takeoff"]
    assert all(entry[:4] == (0, 0, 0, 0) for entry in backend.zeros)
    assert any(
        event.kind == "boot_hover_waiting"
        for event in autonomy.drain_events()
    )


def test_initial_localization_search_uses_yaw_only_then_stops_on_pose(
    tmp_path,
) -> None:
    clock = _Clock()
    backend = _Backend()
    route_calls = []

    def pose():
        if clock.now() < 10.35:
            return None
        return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=pose,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        boot_timeout_s=2.0,
        boot_search_config=YawSearchConfig(
            wait_before_search_s=0.1,
            max_search_s=1.0,
            sweep_angle_deg=20.0,
            yaw_pcmd=6,
        ),
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: route_calls.append(True) or "route complete",
    )

    autonomy._run()

    moving = [entry for entry in backend.vectors if entry[2] != 0]
    assert moving
    assert all(entry[:2] == (0, 0) and entry[3] == 0 for entry in moving)
    assert all(abs(entry[2]) <= 0.6 for entry in moving)
    assert backend.zeros[-1][:4] == (0, 0, 0, 0)
    assert route_calls == [True]


def test_boot_search_yaw_uses_vector_deadman_authority_not_raw_pcmd(
    tmp_path,
) -> None:
    backend = _Backend()
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 0,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )

    assert autonomy._send_boot_search_yaw(6)
    assert autonomy._send_boot_search_yaw(-6)

    moving = [
        entry
        for entry in backend.vectors
        if any(abs(value) > 0.0 for value in entry)
    ]
    assert moving == [(0.0, 0.0, 0.6, 0.0), (0.0, 0.0, -0.6, 0.0)]
    assert not any(entry[2] for entry in backend.zeros)


def test_position_without_same_frame_orientation_commands_nothing_then_recovers(
    tmp_path,
) -> None:
    clock = _Clock()
    backend = _Backend()
    route_calls = []

    def pose():
        # App-level same-frame contract: a fresh position WITHOUT camera
        # orientation yields no AUTO pose, so AUTO must not translate or yaw.
        if clock.now() < 10.35:
            return None
        return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=pose,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        boot_timeout_s=2.0,
        boot_search_config=YawSearchConfig(
            wait_before_search_s=10.0,
            max_search_s=1.0,
            sweep_angle_deg=20.0,
            yaw_pcmd=6,
        ),
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: route_calls.append(True)
        or "route complete",
    )

    autonomy._run()

    assert route_calls == [True]
    assert all(entry[:4] == (0, 0, 0, 0) for entry in backend.zeros)


def test_desktop_route_localization_loss_never_auto_lands(tmp_path) -> None:
    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 0,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )

    hooks = autonomy._route_hooks()
    assert hooks.land_on_localization_loss is False
    assert hooks.localization_yaw_search_pcmd == 8
    assert hooks.localization_yaw_search_event is not None


def test_stable_localization_starts_route_only_after_hover_lock(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    commands = []
    route_calls = []

    def pose():
        return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())

    def run_loop(hooks, controller, waypoints, **kwargs):
        route_calls.append((hooks, controller, waypoints, kwargs))
        return "route complete -> land"

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
        run_loop=run_loop,
    )

    autonomy._run()

    assert commands == ["takeoff", "land"]
    assert len(route_calls) == 1
    kinds = [event.kind for event in autonomy.drain_events()]
    assert kinds.index("boot_hover") < kinds.index("route_started")


def test_stable_localization_starts_after_nearest_takeoff_waypoint(tmp_path) -> None:
    clock = _Clock()
    commands = []
    route_starts = []
    route = (
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
    )

    def pose():
        return Pose(9.9, 0.0, 0.0, 0.0, stamp=clock.now())

    def run_loop(_hooks, controller, _waypoints, **_kwargs):
        route_starts.append(controller.target_index)
        return "route complete -> land"

    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=_snapshot(tmp_path, waypoints=route),
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
        run_loop=run_loop,
    )

    autonomy._run()

    assert commands == ["takeoff", "land"]
    assert route_starts == [3]
    assert any(
        event.kind == "route_start_selected" and "waypoint 3" in event.detail
        for event in autonomy.drain_events()
    )


def test_boot_pose_failure_resets_consecutive_lock(tmp_path) -> None:
    clock = _Clock()
    poses = iter(
        [
            Pose(0.0, 0.0, 0.0, 0.0, stamp=10.0),
            None,
            Pose(0.0, 0.0, 0.0, 0.0, stamp=10.1),
            Pose(0.0, 0.0, 0.0, 0.0, stamp=10.2),
        ]
    )
    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: next(poses),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        now=clock.now,
    )
    lock = pff.BootPoseLock((0.0, 0.0, 0.0))
    last_stamp = None

    locked, last_stamp, valid = autonomy._observe_boot_pose(lock, last_stamp)
    assert not locked and lock.count == 1
    assert valid
    locked, last_stamp, valid = autonomy._observe_boot_pose(lock, last_stamp)
    assert not locked and lock.count == 0
    assert not valid
    clock.sleep(0.1)
    locked, last_stamp, valid = autonomy._observe_boot_pose(lock, last_stamp)
    assert not locked and lock.count == 1
    assert valid
    clock.sleep(0.1)
    locked, _last_stamp, valid = autonomy._observe_boot_pose(lock, last_stamp)
    assert not locked and lock.count == 2
    assert valid


def test_airborne_manual_start_claims_pc_without_sending_takeoff(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    backend.pilot_sticks = True
    commands = []
    route_calls = []

    def take_pc_control():
        commands.append("pc_control")
        backend.pilot_sticks = False
        return True

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.5, 0.0, 0.0, 0.0, stamp=clock.now()),
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
        run_loop=lambda *_args, **_kwargs: route_calls.append(True) or "route complete",
    )

    autonomy._run()

    assert commands == ["pc_control", "land"]
    assert route_calls == [True]
    assert backend.zeros
    assert backend.zeros[0][:4] == (0, 0, 0, 0)
    command_events = [
        event for event in autonomy.drain_events()
        if event.kind == "command_result"
    ]
    assert command_events[0].command == "pc_control"
    assert command_events[0].result is True


def test_failed_airborne_pc_handoff_stays_manual_without_takeoff_or_land(
    tmp_path,
) -> None:
    backend = _Backend()
    backend.pilot_sticks = True
    commands = []
    run_loop = Mock(return_value="must not run")
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
        take_pc_control=lambda: commands.append("pc_control") or False,
        start_airborne=True,
        land=lambda: commands.append("land") or True,
        run_loop=run_loop,
    )

    autonomy._run()

    assert commands == ["pc_control"]
    assert backend.pilot_sticks is True
    run_loop.assert_not_called()
    events = autonomy.drain_events()
    assert any(
        event.kind == "finished" and "PC control handoff failed" in event.detail
        for event in events
    )


def test_run_normal_mock_sequence_takes_off_routes_and_lands(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    takeoff = Mock(return_value=True)
    run_loop = Mock(return_value="route complete")
    land = Mock(return_value=True)

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=Mock(),
        stream_healthy=lambda: True,
        takeoff=takeoff,
        land=land,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=run_loop,
    )

    autonomy._run()

    takeoff.assert_called_once_with()
    run_loop.assert_called_once()
    land.assert_called_once_with()
    assert autonomy.phase == "DONE"


def test_run_arming_blocker_hovers_until_operator_stops(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    takeoff = Mock(return_value=True)
    run_loop = Mock(return_value="must not run")
    land = Mock(return_value=True)
    autonomy = None

    def sleep_until_cancelled(duration: float) -> None:
        clock.sleep(duration)
        if clock.now() >= 10.8:
            autonomy._cancel.set()

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=Mock(),
        stream_healthy=lambda: True,
        takeoff=takeoff,
        land=land,
        arming_blockers=lambda: ["operator approval required"],
        now=clock.now,
        sleep=sleep_until_cancelled,
        run_loop=run_loop,
    )

    autonomy._run()

    takeoff.assert_called_once_with()
    run_loop.assert_not_called()
    land.assert_not_called()
    assert autonomy.phase == "DONE"
    assert any(
        event.kind == "boot_hover_waiting" and "operator approval required" in event.detail
        for event in autonomy.drain_events()
    )


def test_run_route_exception_enters_auto_failed_and_hovers_without_landing(
    tmp_path,
) -> None:
    clock = _Clock()
    backend = _Backend()
    run_loop = Mock(side_effect=RuntimeError("route boom"))
    land = Mock(return_value=True)

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=Mock(),
        stream_healthy=lambda: True,
        takeoff=Mock(return_value=True),
        land=land,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=run_loop,
    )

    autonomy._run()

    run_loop.assert_called_once()
    land.assert_not_called()
    assert any(
        entry == (0, 0, 0, 0, "auto_worker_failure_hover")
        for entry in backend.zeros
    )
    events = autonomy.drain_events()
    assert any(
        event.kind == "auto_failed" and "AUTO worker failed" in event.detail
        for event in events
    )
    assert not any(event.kind == "finished" for event in events)
    assert backend.pilot_sticks is False
    assert autonomy.phase == "AUTO_FAILED"


def test_repeated_auto_pcmd_rejection_terminates_route_in_hover_failure(
    tmp_path,
) -> None:
    class RejectingBackend(_Backend):
        def set_nudge_vector(self, roll, pitch, yaw, gaz):
            self.vectors.append((roll, pitch, yaw, gaz))
            return False

    backend = RejectingBackend()
    land = Mock(return_value=True)
    route_calls = []
    clock = _Clock()
    autonomy = None

    def run_loop(hooks, *_args, **_kwargs):
        for _ in range(AUTO_PCMD_FAILURE_LIMIT + 1):
            route_calls.append(hooks.send_authorized_pcmd((5, 0, 0, 0)))
            if hooks.safety_poll() == "LAND":
                return "AUTO PCMD failure stopped route"
        raise AssertionError("route continued after persistent PCMD rejection")

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
        land=land,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=run_loop,
    )

    autonomy._run()

    assert len(route_calls) == AUTO_PCMD_FAILURE_LIMIT
    assert all(not result[0] for result in route_calls)
    assert land.call_count == 0
    assert backend.zeros[-1][:4] == (0, 0, 0, 0)
    events = autonomy.drain_events()
    assert any(event.kind == "auto_failed" for event in events)
    assert not any(event.kind == "finished" for event in events)
    assert autonomy.phase == "AUTO_FAILED"


def test_run_stop_after_takeoff_skips_route_and_landing(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    takeoff = Mock()
    run_loop = Mock(return_value="must not run")
    land = Mock(return_value=True)
    autonomy = None

    def stop_during_takeoff():
        takeoff()
        autonomy._cancel.set()
        return True

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=Mock(),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=Mock(),
        stream_healthy=lambda: True,
        takeoff=stop_during_takeoff,
        land=land,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=run_loop,
    )

    autonomy._run()

    takeoff.assert_called_once_with()
    run_loop.assert_not_called()
    land.assert_not_called()
    assert any(
        event.kind == "finished" and "cancelled after takeoff" in event.detail
        for event in autonomy.drain_events()
    )
    assert autonomy.phase == "DONE"


def test_auto_controller_config_stays_bound_to_verified_route_snapshot(
    tmp_path, monkeypatch
) -> None:
    route = tmp_path / "route.json"

    def write_route(radius: float) -> None:
        route.write_text(
            json.dumps({
                "schema": "sfm-flight-route/v1",
                "site_id": "field-a",
                "coordinate_frame_id": "glomap-a",
                "frame": "glomap",
                "units": "map",
                "purpose": "flight",
                "closed": False,
                "arrive_radius_map_units": radius,
                "waypoints": [[0, 0, 0], [1, 0, 0]],
            }),
            encoding="utf-8",
        )

    write_route(0.42)
    snapshot = rpf.capture_mission_route_snapshot(
        route,
        expected_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
        expected_site_id="field-a",
        expected_coordinate_frame_id="glomap-a",
        map_frame=LEGACY_MAP_FRAME,
    )

    original_verify = rpf.MissionRouteSnapshot.verify_file_unchanged

    def verify_then_replace(route_snapshot):
        original_verify(route_snapshot)
        write_route(0.91)

    monkeypatch.setattr(rpf.MissionRouteSnapshot, "verify_file_unchanged", verify_then_replace)
    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=snapshot,
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )

    assert autonomy.controller.cfg.waypoint_arrive_radius == 0.42


def test_successful_auto_run_follows_the_complete_phase_sequence(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    phases = []

    def note_phase() -> None:
        phase = autonomy.phase
        if not phases or phases[-1] != phase:
            phases.append(phase)

    def pose():
        note_phase()
        return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=pose,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: note_phase() or True,
        land=lambda: note_phase() or True,
        boot_timeout_s=1.0,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: note_phase() or "route complete",
    )

    autonomy._run()

    assert phases == ["TAKEOFF", "BOOT_HOVER", "ROUTE", "LANDING"]
    assert autonomy.phase == "DONE"


def test_pause_holds_zero_and_resume_keeps_the_same_route_run(tmp_path) -> None:
    backend = _Backend()
    takeoffs = []
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: takeoffs.append("takeoff") or True,
        land=lambda: True,
    )
    autonomy.phase = "ROUTE"

    assert autonomy.pause()
    assert autonomy.paused
    assert autonomy._safety_mode() == "HOVER"
    accepted, reason, applied = autonomy._send_authorized((5, 0, 0, 0))
    assert not accepted
    assert reason == "AUTO paused"
    assert applied == (0, 0, 0, 0)
    assert backend.zeros[-1][:4] == (0, 0, 0, 0)

    assert autonomy.resume()
    assert not autonomy.paused
    assert autonomy._safety_mode() == "AUTO"
    assert takeoffs == []


def test_pose_jump_pause_emits_event_and_requires_resume(tmp_path) -> None:
    backend = _Backend()
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )
    autonomy.phase = "ROUTE"

    autonomy._pause_for_pose_jump(2.75)

    assert autonomy.paused
    event = autonomy.drain_events()[0]
    assert event.kind == "pose_jump_paused"
    assert "2.75" in event.detail
    assert backend.zeros[-1][:4] == (0, 0, 0, 0)
    assert autonomy.resume()
    assert not autonomy.paused


def test_physical_stick_override_stops_authorized_auto_motion(tmp_path) -> None:
    backend = _Backend()
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )
    autonomy.phase = "ROUTE"
    backend.pilot_sticks = True

    accepted, reason, applied = autonomy._send_authorized((5, 5, 5, 5))

    assert autonomy._safety_mode() == "LAND"
    assert not accepted
    assert reason == "pilot stick override"
    assert applied is None
    assert backend.vectors[-1] == (0.0, 0.0, 0.0, 0.0)


def test_cancel_hands_control_to_manual_without_resuming_auto(tmp_path) -> None:
    class HandoffBackend(_Backend):
        def __init__(self):
            super().__init__()
            self.handoffs = []
            self.handoff_started = threading.Event()
            self.release_handoff = threading.Event()

        def give_to_pilot(self, *, reason):
            self.handoffs.append(reason)
            self.handoff_started.set()
            self.release_handoff.wait(1.0)
            return super().give_to_pilot(reason=reason)

    backend = HandoffBackend()
    commands = []
    route_calls = []
    autonomy = None

    def takeoff():
        commands.append("takeoff")
        autonomy.cancel("manual_takeover")
        return True

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=10.0),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=takeoff,
        land=lambda: commands.append("land") or True,
        run_loop=lambda *_args, **_kwargs: route_calls.append(True) or "unexpected",
    )

    finished = threading.Event()

    def run() -> None:
        autonomy._run()
        finished.set()

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    assert backend.handoff_started.wait(1.0)
    assert finished.wait(1.0)
    assert any(entry[:4] == (0, 0, 0, 0) for entry in backend.zeros)
    backend.release_handoff.set()
    runner.join(timeout=1.0)
    assert not runner.is_alive()

    assert commands == ["takeoff"]
    assert route_calls == []
    assert backend.handoffs == ["auto_cancel:manual_takeover"]
    assert autonomy.phase == "DONE"
    assert autonomy._send_authorized((5, 0, 0, 0))[0] is False
    assert backend.handoffs == ["auto_cancel:manual_takeover"]



@pytest.mark.parametrize("measured_speed", [0.30, 0.31])
def test_fresh_speed_at_or_above_limit_hovers_before_route_command(
    tmp_path,
    measured_speed,
) -> None:
    clock = _Clock()
    backend = _Backend()
    backend.state.ground_speed_mps = measured_speed
    backend.state.ground_speed_mono_ns = 10_000_000_000
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        now=clock.now,
        sleep=clock.sleep,
    )

    accepted, reason, applied = autonomy._send_authorized((5, 0, 0, 0))

    assert not accepted
    assert "speed limit" in reason
    assert applied == (0, 0, 0, 0)
    assert backend.vectors == [(0.0, 0.0, 0.0, 0.0)]
    assert backend.zeros[-1][:4] == (0, 0, 0, 0)
    assert backend.state.autonomous_speed_guard_status == "OVERSPEED_HOVER"


def test_fresh_speed_below_limit_keeps_guard_active_and_sends_route_command(
    tmp_path,
) -> None:
    clock = _Clock()
    backend = _Backend()
    backend.state.ground_speed_mps = 0.29
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        now=clock.now,
        sleep=clock.sleep,
    )

    accepted, reason, applied = autonomy._send_authorized((5, 0, 0, 0))

    assert accepted
    assert "fresh-speed guard" in reason
    assert applied == (5, 0, 0, 0)
    assert backend.vectors == [(0.5, 0.0, 0.0, 0.0)]
    assert backend.state.autonomous_speed_guard_status == "FRESH_SPEED_GUARD"


def test_disabled_speed_limit_bypasses_only_the_ground_speed_guard(tmp_path) -> None:
    backend = _Backend()
    backend.state.autonomous_speed_limit_enabled = False
    backend.state.ground_speed_mps = None
    backend.state.ground_speed_mono_ns = None
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )

    accepted, _reason, applied = autonomy._send_authorized((5, 0, 0, 0))

    assert accepted
    assert applied == (5, 0, 0, 0)
    assert backend.state.autonomous_speed_guard_status == "SPEED_LIMIT_DISABLED"


def test_runtime_safety_latch_holds_auto_at_zero(tmp_path) -> None:
    backend = _Backend()
    backend._runtime_safety_action_latched = True
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )

    accepted, reason, applied = autonomy._send_authorized((5, 0, 0, 0))

    assert autonomy._safety_mode() == "HOVER"
    assert not accepted
    assert reason == "AUTO runtime safety hover"
    assert applied == (0, 0, 0, 0)
    assert backend.vectors == [(0.0, 0.0, 0.0, 0.0)]
    assert backend.zeros[-1] == (0, 0, 0, 0, "auto_runtime_safety_hover")


@pytest.mark.parametrize("speed_sample", [None, (5.0, 1.0)])
def test_missing_or_stale_speed_blocks_horizontal_auto_command(
    tmp_path,
    speed_sample,
) -> None:
    clock = _Clock()
    backend = _Backend()
    if speed_sample is None:
        backend.state.ground_speed_mps = None
        backend.state.ground_speed_mono_ns = None
    else:
        backend.state.ground_speed_mps, speed_stamp = speed_sample
        backend.state.ground_speed_mono_ns = int(speed_stamp * 1_000_000_000)
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        now=clock.now,
        sleep=clock.sleep,
    )

    accepted, reason, applied = autonomy._send_authorized((5, 0, 0, 0))

    assert not accepted
    assert "ground speed" in reason
    assert applied == (0, 0, 0, 0)
    assert all(
        not any(abs(value) > 0.0 for value in vector)
        for vector in backend.vectors
    )
    assert backend.state.autonomous_speed_guard_status == "SPEED_UNAVAILABLE_HOVER"


def test_speed_guard_latch_releases_only_below_hysteresis_threshold(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        now=clock.now,
        sleep=clock.sleep,
    )

    backend.state.ground_speed_mps = 0.30
    accepted, reason, applied = autonomy._send_authorized((5, 0, 0, 0))
    assert not accepted
    assert "speed limit" in reason
    assert applied == (0, 0, 0, 0)
    assert autonomy._speed_guard_latched

    backend.state.ground_speed_mps = 0.25
    accepted, _reason, _applied = autonomy._send_authorized((5, 0, 0, 0))
    assert not accepted
    assert autonomy._speed_guard_latched

    backend.state.ground_speed_mps = 0.24
    accepted, _reason, _applied = autonomy._send_authorized((5, 0, 0, 0))
    assert not accepted
    assert autonomy._speed_guard_latched

    backend.state.ground_speed_mps = 0.23
    accepted, reason, applied = autonomy._send_authorized((5, 0, 0, 0))
    assert accepted
    assert "fresh-speed guard" in reason
    assert applied == (5, 0, 0, 0)
    assert not autonomy._speed_guard_latched


def test_paused_auto_does_not_trip_speed_guard_on_stale_telemetry(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    backend.state.ground_speed_mps = None
    backend.state.ground_speed_mono_ns = None
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        now=clock.now,
        sleep=clock.sleep,
    )
    autonomy.phase = "ROUTE"

    assert autonomy.pause()
    accepted, reason, applied = autonomy._send_authorized((5, 0, 0, 0))

    assert not accepted
    assert reason == "AUTO paused"
    assert applied == (0, 0, 0, 0)
    assert not autonomy._speed_guard_latched


def test_takeoff_must_return_literal_true_before_boot_hover(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    commands = []
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: commands.append("takeoff") or None,
        land=lambda: commands.append("land") or True,
        boot_timeout_s=0.3,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: commands.append("route") or "unexpected",
    )

    autonomy._run()

    assert commands == ["takeoff"]
    assert autonomy.phase == "DONE"
    assert not any(event.kind == "boot_hover" for event in autonomy.drain_events())


@pytest.mark.parametrize("landing_result", [False, RuntimeError("landing failed")])
def test_unconfirmed_landing_retries_and_completes_worker(tmp_path, landing_result) -> None:
    clock = _Clock()
    backend = _Backend()
    commands = []
    land = Mock(side_effect=[landing_result, landing_result])

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
        land=land,
        boot_timeout_s=1.0,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: "route complete",
    )

    autonomy._run()

    assert commands == ["takeoff"]
    assert land.call_count == 2
    events = autonomy.drain_events()
    assert sum(event.kind == "landing_unresolved" for event in events) == 1
    assert sum(event.kind == "finished" for event in events) == 1
    assert autonomy._landing_confirmed is False
    assert autonomy.phase == "DONE"

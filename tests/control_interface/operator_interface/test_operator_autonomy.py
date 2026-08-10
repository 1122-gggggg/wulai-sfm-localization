from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operator_autonomy import DesktopRouteAutonomy
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
        self.nudge_pct = 10
        self.zeros = []
        self.vectors = []
        self.state = SimpleNamespace(
            att_yaw=0.0,
            ground_speed_mps=0.0,
            ground_speed_mono_ns=10_000_000_000,
            telemetry_read_mono_ns=10_000_000_000,
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


def _snapshot(tmp_path) -> MissionRouteSnapshot:
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps({
            "waypoints": [[0, 0, 0], [1, 0, 0]],
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
        waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )


def test_no_localization_after_auto_takeoff_hovers_then_lands(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    commands = []
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
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: "must not run",
    )

    autonomy._run()

    assert commands == ["takeoff", "land"]
    assert backend.zeros
    assert all(entry[:4] == (0, 0, 0, 0) for entry in backend.zeros)
    events = autonomy.drain_events()
    assert any(event.kind == "boot_hover" for event in events)
    assert not any(event.kind == "route_started" for event in events)


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


def test_run_arming_blocker_lands_before_route_loop(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    takeoff = Mock(return_value=True)
    run_loop = Mock(return_value="must not run")
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
        arming_blockers=lambda: ["operator approval required"],
        now=clock.now,
        sleep=clock.sleep,
        run_loop=run_loop,
    )

    autonomy._run()

    takeoff.assert_called_once_with()
    run_loop.assert_not_called()
    land.assert_called_once_with()
    assert autonomy.phase == "DONE"
    assert any(
        event.kind == "landing_unresolved" or "arming blocked" in event.detail
        for event in autonomy.drain_events()
    )


def test_run_route_exception_lands_and_reports_worker_failure(tmp_path) -> None:
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
    land.assert_called_once_with()
    events = autonomy.drain_events()
    assert any(
        event.kind == "finished" and "AUTO worker failed" in event.detail
        for event in events
    )
    assert autonomy.phase == "DONE"


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



def test_fresh_overspeed_hovers_before_sending_a_route_command(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    backend.state.ground_speed_mps = 0.31
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


def test_stale_speed_uses_truthful_command_cap_without_blocking_no_gps_auto(
    tmp_path,
) -> None:
    clock = _Clock()
    backend = _Backend()
    backend.state.ground_speed_mps = 5.0
    backend.state.ground_speed_mono_ns = 1_000_000_000
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
    assert "command cap" in reason
    assert applied == (5, 0, 0, 0)
    assert backend.state.autonomous_speed_guard_status == "COMMAND_CAP_ONLY"


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
        get_pose=lambda: None,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: commands.append("takeoff") or True,
        land=land,
        boot_timeout_s=0.1,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: "unexpected",
    )

    autonomy._run()

    assert commands == ["takeoff"]
    assert land.call_count == 2
    events = autonomy.drain_events()
    assert sum(event.kind == "landing_unresolved" for event in events) == 1
    assert sum(event.kind == "finished" for event in events) == 1
    assert autonomy._landing_confirmed is False
    assert autonomy.phase == "DONE"

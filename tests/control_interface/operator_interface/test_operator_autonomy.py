from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from operator_autonomy import (
    AUTO_BATTERY_RESERVE_PCT,
    AUTO_PCMD_FAILURE_LIMIT,
    AUTO_WAIT_BUDGET_S,
    DesktopRouteAutonomy,
)
from operator_localization_search import YawSearchConfig
import path_follow_flight as pff
import real_path_follow_controller as rpf
from real_path_follow_controller import (
    DESKTOP_AUTO_MAX_TRANSLATION_PCMD,
    DESKTOP_AUTO_MAX_YAW_PCMD,
    DESKTOP_AUTO_YAW_ALIGNMENT_TIMEOUT_S,
    LEGACY_MAP_FRAME,
    MissionRouteSnapshot,
    Pose,
)


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
        self.runtime_safety_latched = False
        self.maneuver_in_progress = None
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

    def pilot_override_active(self):
        return self.pilot_sticks

    def gimbal_pitch_sample(self):
        if getattr(self.state, "gimbal_stamp_source", None) != "simulated":
            return None
        return self.state.gimbal_pitch_deg, self.state.gimbal_mono_ns

    def send_pcmd(self, roll, pitch, yaw, gaz, *, reason, auto_run_id=None):
        self.zeros.append((roll, pitch, yaw, gaz, reason))
        return True

    def set_nudge_vector(self, roll, pitch, yaw, gaz, *, authority_pct=None, auto_run_id=None):
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
    waypoints=None,
) -> MissionRouteSnapshot:
    route_waypoints = (
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        if waypoints is None
        else tuple(tuple(float(value) for value in point) for point in waypoints)
    )
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "waypoints": route_waypoints,
                "frame": "glomap",
                "units": "map",
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(route.read_bytes()).hexdigest()
    return MissionRouteSnapshot(
        path=route,
        sha256=digest,
        site_id="field-a",
        coordinate_frame_id="glomap-a",
        waypoints=route_waypoints,
    )


def test_desktop_auto_keeps_route_in_raw_map_frame(tmp_path) -> None:
    snapshot = _snapshot(
        tmp_path,
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
    )

    assert np.asarray(autonomy.waypoints) == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    )


def test_route_hooks_forward_reseed_confirmation(tmp_path) -> None:
    def reseed_confirming():
        return True

    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
        snapshot=_snapshot(tmp_path),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: None,
        pose_is_weak=lambda: True,
        pose_reseed_confirming=reseed_confirming,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
    )

    assert autonomy._route_hooks().pose_reseed_confirming is reseed_confirming


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
    assert any(event.kind == "boot_hover_waiting" for event in autonomy.drain_events())


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
    clock = _Clock()
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
        now=clock.now,
    )

    assert autonomy._send_boot_search_yaw(6)
    assert autonomy._send_boot_search_yaw(-6)

    moving = [entry for entry in backend.vectors if any(abs(value) > 0.0 for value in entry)]
    # Unit axes are scaled by AUTO's authority, and the backend scales them
    # back, so what reaches the aircraft is still the +/-6% that was asked for.
    authority = autonomy._pcmd_authority_pct()
    assert moving == [
        (0.0, 0.0, 6 / authority, 0.0),
        (0.0, 0.0, -6 / authority, 0.0),
    ]
    assert [round(entry[2] * authority) for entry in moving] == [6, -6]
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
        run_loop=lambda *_args, **_kwargs: route_calls.append(True) or "route complete",
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
    assert hooks.localization_yaw_search_pcmd == 0
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


def test_stable_localization_starts_at_waypoint_one_even_near_a_later_point(tmp_path) -> None:
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
    assert route_starts == [0]
    assert any(
        event.kind == "route_start_selected" and "starts at waypoint 1" in event.detail
        for event in autonomy.drain_events()
    )


def test_takeoff_far_off_the_route_still_starts_at_waypoint_one(tmp_path) -> None:
    # There is no corridor policing the opening leg or the route, so a takeoff
    # far off the drawn polyline must still start at waypoint 1 rather than refuse.
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
        return Pose(9.9, 6.0, 0.0, 0.0, stamp=clock.now())

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
    assert route_starts == [0]
    assert autonomy.phase != "AUTO_FAILED"
    kinds = [event.kind for event in autonomy.drain_events()]
    assert "auto_failed" not in kinds
    assert "route_start_selected" in kinds


def test_auto_yaw_authority_is_not_tied_to_the_manual_nudge_size(tmp_path) -> None:
    # A small --nudge-pct used to shrink AUTO's yaw command with it: at 8 the
    # alignment turned at 8% of the firmware MaxRotationSpeed while translation
    # was held at zero, so AUTO never reached the route (measured 2026-09-12).
    backend = _Backend()
    backend.nudge_pct = 8
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

    cfg = autonomy.controller.cfg
    assert cfg.max_yaw_pcmd == DESKTOP_AUTO_MAX_YAW_PCMD
    assert cfg.max_yaw_pcmd > backend.nudge_pct
    assert cfg.yaw_alignment_timeout_s == pytest.approx(DESKTOP_AUTO_YAW_ALIGNMENT_TIMEOUT_S)
    assert cfg.max_translation_pcmd == DESKTOP_AUTO_MAX_TRANSLATION_PCMD
    # Translation authority is intentionally below the default manual nudge
    # size: the two envelopes are unrelated. Dispatch must scale by AUTO's
    # own authority, verified below.

    # And the dispatch seam carries those caps through: it used to divide by
    # nudge_pct while the backend multiplied by nudge_pct, so the whole chain
    # collapsed to clamp(pcmd, +/-nudge_pct) and a commanded 50% yaw left as 8%.
    authority = autonomy._pcmd_authority_pct()
    assert authority == DESKTOP_AUTO_MAX_YAW_PCMD
    for pcmd in (
        (0, 0, DESKTOP_AUTO_MAX_YAW_PCMD, 0),
        (DESKTOP_AUTO_MAX_TRANSLATION_PCMD, 0, 0, 0),
    ):
        autonomy._dispatch_route_pcmd(pcmd)
        vector = backend.vectors[-1]
        assert [round(value * authority) for value in vector] == list(pcmd)

    # Lost-pose default is hover-in-place, never blind rotation (operator
    # decision 2026-09-13): the search only spins on explicit opt-in.
    assert autonomy.spin_to_search is False
    assert autonomy._route_hooks().localization_yaw_search_pcmd == 0


def test_desktop_auto_matches_path_follow_production_config(tmp_path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(pff, "PATH_JSON", str(snapshot.path))
    monkeypatch.setattr(pff, "MAP_ALIGN", "")
    monkeypatch.setattr(pff, "FLIGHT_CONTRACT_JSON", "")
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
    controller, _waypoints = pff.build_controller(require_approved=False)
    desktop = autonomy.controller.cfg
    production = controller.cfg
    assert desktop.max_yaw_pcmd == production.max_yaw_pcmd
    assert desktop.yaw_alignment_timeout_s == pytest.approx(production.yaw_alignment_timeout_s)
    assert desktop.max_translation_pcmd == production.max_translation_pcmd
    assert desktop.return_to_start == production.return_to_start
    assert desktop.return_to_start is True
    assert desktop.max_yaw_pcmd == DESKTOP_AUTO_MAX_YAW_PCMD
    assert desktop.max_translation_pcmd == DESKTOP_AUTO_MAX_TRANSLATION_PCMD
    assert desktop.yaw_alignment_timeout_s == pytest.approx(DESKTOP_AUTO_YAW_ALIGNMENT_TIMEOUT_S)
    assert desktop.waypoint_arrive_radius == rpf.DESKTOP_AUTO_DEFAULT_ARRIVE_RADIUS


class _SessionLogs:
    """Minimal stand-in for the localization stream the AUTO worker writes to."""

    def __init__(self):
        self.records = []

    def localization(self, event, **fields):
        self.records.append((event, fields))
        return True


def test_route_loop_logs_the_plan_and_each_tick_against_it(tmp_path) -> None:
    # The per-frame pose already reaches localization.jsonl; what was missing is
    # what the route loop made of it, which is what an AUTO debrief needs.
    clock = _Clock()
    backend = _Backend()
    backend.session_logs = _SessionLogs()
    route = (
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
    )

    def pose():
        return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())

    def run_loop(hooks, _controller, _waypoints, **_kwargs):
        hooks.log_tick({"route_distance_u": 0.25, "pose_u": [0.0, 0.0, 0.0]})
        return "route complete -> land"

    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=_snapshot(tmp_path, waypoints=route),
        map_frame=LEGACY_MAP_FRAME,
        get_pose=pose,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        boot_timeout_s=1.0,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=run_loop,
    )

    autonomy._run()

    events = dict(backend.session_logs.records)
    assert "auto_route_plan" in events
    plan = events["auto_route_plan"]
    assert plan["drawn_waypoint_count"] == 3
    assert plan["waypoints_u"][0] == [0.0, 0.0, 0.0]
    assert events["auto_route_tick"]["route_distance_u"] == pytest.approx(0.25)


def test_route_tick_logging_survives_a_backend_without_session_logs(tmp_path) -> None:
    clock = _Clock()
    autonomy = DesktopRouteAutonomy(
        backend=_Backend(),
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

    autonomy._log_route_plan()
    autonomy._log_route_tick({"route_distance_u": 1.0})


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
    command_events = [event for event in autonomy.drain_events() if event.kind == "command_result"]
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
        event.kind == "finished" and "PC control handoff failed" in event.detail for event in events
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
    assert any(entry == (0, 0, 0, 0, "auto_worker_failure_hover") for entry in backend.zeros)
    events = autonomy.drain_events()
    assert any(
        event.kind == "auto_failed" and "AUTO worker failed" in event.detail for event in events
    )
    assert not any(event.kind == "finished" for event in events)
    assert backend.pilot_sticks is False
    assert autonomy.phase == "AUTO_FAILED"


def test_repeated_auto_pcmd_rejection_terminates_route_in_hover_failure(
    tmp_path,
) -> None:
    class RejectingBackend(_Backend):
        def set_nudge_vector(self, roll, pitch, yaw, gaz, *, authority_pct=None, auto_run_id=None):
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
            json.dumps(
                {
                    "schema": "sfm-flight-route/v1",
                    "site_id": "field-a",
                    "coordinate_frame_id": "glomap-a",
                    "frame": "glomap",
                    "units": "map",
                    "purpose": "flight",
                    "closed": False,
                    "arrive_radius_map_units": radius,
                    "waypoints": [[0, 0, 0], [1, 0, 0]],
                }
            ),
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

    autonomy.controller._arrive_frames = 2
    target = autonomy.controller.target_index
    assert autonomy.pause("auto_localization_source_change")
    assert autonomy.controller._arrive_frames == 0
    assert autonomy.controller.target_index == target
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
    events = autonomy.drain_events()
    assert "paused" in [event.kind for event in events]
    event = next(event for event in events if event.kind == "pose_jump_paused")
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


def test_rc_landing_hands_auto_back_to_the_skycontroller(tmp_path) -> None:
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
    backend.state.flight_state = "landing"  # no land_cmd: the SkyController button

    accepted, reason, applied = autonomy._send_authorized((5, 5, 0, 8))

    assert not accepted and applied is None
    assert reason == "RC landing: AUTO yields"
    assert autonomy._safety_mode() == "LAND"
    assert backend.pilot_sticks, "control goes back to the SkyController"
    assert backend.zeros[-1] == (0, 0, 0, 0, "auto_rc_landing")
    assert "rc_landing_yield" in [event.kind for event in autonomy.drain_events()]


@pytest.mark.parametrize(
    "phase,flight_state,maneuver",
    [
        ("ROUTE", "flying", None),
        ("ROUTE", "landing", "landing"),  # the backend's own land_cmd
        ("LANDING", "landing", None),  # AUTO's own end-of-route landing
    ],
)
def test_only_an_uncommanded_landing_during_auto_flight_yields(
    tmp_path, phase, flight_state, maneuver
) -> None:
    backend = _Backend()
    backend.maneuver_in_progress = maneuver
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
    autonomy.phase = phase
    backend.state.flight_state = flight_state

    assert autonomy._safety_mode() == "AUTO"
    assert not backend.pilot_sticks


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


@pytest.mark.parametrize("measured_speed", [0.30, 0.31, None])
def test_ground_speed_does_not_zero_horizontal_auto_command(
    tmp_path,
    measured_speed,
) -> None:
    clock = _Clock()
    backend = _Backend()
    backend.state.ground_speed_mps = measured_speed
    backend.state.ground_speed_mono_ns = None if measured_speed is None else 10_000_000_000
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
    assert "speed" not in reason.lower() or "command cap" in reason
    assert applied == (5, 0, 0, 0)
    authority = autonomy._pcmd_authority_pct()
    assert backend.vectors == [(5 / authority, 0.0, 0.0, 0.0)]
    assert backend.zeros == []


def test_disabled_speed_limit_still_sends_horizontal_auto_command(tmp_path) -> None:
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


def test_runtime_safety_latch_holds_auto_at_zero(tmp_path) -> None:
    backend = _Backend()
    backend.runtime_safety_latched = True
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


def _budget_autonomy(tmp_path, backend, clock) -> DesktopRouteAutonomy:
    return DesktopRouteAutonomy(
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
        run_loop=lambda *_args, **_kwargs: "route complete",
    )


@pytest.mark.parametrize("speed, age", [
    (0.317, 0.0), (0.0, 0.6), (float("nan"), 0.0), (0.0, -1.0),
])
def test_auto_yaw_command_is_not_held_for_horizontal_stop(tmp_path, speed, age):
    clock = _Clock()
    backend = _Backend()
    backend.state.autonomous_speed_limit_mps = 0.6
    backend.state.ground_speed_mps = speed
    backend.state.ground_speed_mono_ns = int((clock.now() - age) * 1e9)
    autonomy = _budget_autonomy(tmp_path, backend, clock)

    accepted, _, applied = autonomy._send_authorized((0, 0, 50, 2))
    assert accepted
    assert applied == (0, 0, 50, 2)
    record = {"pcmd": list(applied), "pcmd_phase": "turn"}
    autonomy._log_route_tick(record)
    assert record["pcmd_phase"] == "turn"


@pytest.mark.parametrize("accept_weak_poses, expected_gate", [(True, False), (False, True)])
def test_desktop_route_weak_pose_gate_follows_accept_weak_poses(
    tmp_path, accept_weak_poses, expected_gate
):
    autonomy = _budget_autonomy(tmp_path, _Backend(), _Clock())
    autonomy.accept_weak_poses = accept_weak_poses
    seen = {}

    def run_loop(_hooks, _controller, _waypoints, **kwargs):
        seen.update(kwargs)
        return "route complete"

    autonomy.run_loop = run_loop
    assert autonomy._run_route() == "route complete"
    assert seen["enforce_weak_pose_gate"] is expected_gate


def test_turn_brake_log_preserves_a_later_manual_handoff(tmp_path):
    autonomy = _budget_autonomy(tmp_path, _Backend(), _Clock())
    record = {"pcmd": None, "pcmd_phase": "turn", "blocked": True, "reason": "MANUAL"}
    autonomy._log_route_tick(record)
    assert record["reason"] == "MANUAL"


def test_boot_pose_lock_rejects_vo_only_even_with_high_inliers(tmp_path):
    clock = _Clock()
    autonomy = _budget_autonomy(tmp_path, _Backend(), clock)
    autonomy.accept_weak_poses = False
    autonomy.pose_is_weak = lambda: True
    autonomy.pose_confidence = lambda: 300
    lock = Mock()
    assert autonomy._observe_boot_pose(lock, None) == (False, None, False)
    lock.observe.assert_not_called()
    lock.reset.assert_called_once()


@pytest.mark.parametrize("source", ["unconfirmed", "unobserved", "reseed", "predicted_hook", "reseed_hook"])
def test_boot_lock_requires_observed_map_pose_even_when_health_is_ok(tmp_path, source):
    clock = _Clock()
    autonomy = _budget_autonomy(tmp_path, _Backend(), clock)
    autonomy.accept_weak_poses = False
    autonomy.pose_is_weak = lambda: False
    autonomy.pose_confidence = lambda: 300
    bad = {"active": False}
    autonomy.get_pose = lambda: Pose(
        0, 0, 0, 0, stamp=clock.now(),
        map_confirmed=not (bad["active"] and source == "unconfirmed"),
        position_observed=not (bad["active"] and source == "unobserved"),
        reseed_confirming=bad["active"] and source == "reseed",
    )
    autonomy.pose_is_predicted = lambda: bad["active"] and source == "predicted_hook"
    autonomy.pose_reseed_confirming = lambda: bad["active"] and source == "reseed_hook"
    lock = pff.BootPoseLock((0, 0, 0))
    _, stamp, valid = autonomy._observe_boot_pose(lock, None)
    assert valid and lock.count == 1
    bad["active"] = True
    for _ in range(lock.required_fixes):
        clock.sleep(0.1)
        locked, stamp, valid = autonomy._observe_boot_pose(lock, stamp)
        assert not locked and not valid and lock.count == 0
    bad["active"] = False
    for index in range(lock.required_fixes):
        clock.sleep(0.1)
        locked, stamp, valid = autonomy._observe_boot_pose(lock, stamp)
        assert valid and locked == (index == lock.required_fixes - 1)


@pytest.mark.parametrize("predicted", [False, True])
def test_boot_lock_accepts_stable_estimates_without_promoting_map_confidence(tmp_path, predicted):
    clock = _Clock()
    autonomy = _budget_autonomy(tmp_path, _Backend(), clock)
    autonomy.accept_weak_poses = True
    autonomy.pose_is_weak = lambda: True
    autonomy.pose_is_predicted = lambda: predicted
    autonomy.pose_confidence = lambda: 0
    autonomy.get_pose = lambda: Pose(
        0, 0, 0, 0, stamp=clock.now(),
        map_confirmed=False, position_observed=not predicted,
    )
    lock = pff.BootPoseLock((0, 0, 0))
    stamp = None
    for index in range(lock.required_fixes):
        clock.sleep(0.1)
        locked, stamp, valid = autonomy._observe_boot_pose(lock, stamp)
        assert valid and locked == (index == lock.required_fixes - 1)
        # Polling the same estimate must not count as another fix.
        autonomy._observe_boot_pose(lock, stamp)
        assert lock.count == index + 1
    assert not autonomy.get_pose().map_confirmed
    # Exercise the actual BOOT loop, including the post-lock runtime gate.
    autonomy.arming_blockers = Mock(return_value=[])
    assert autonomy._boot_hover() is None
    autonomy.arming_blockers.assert_called_once()


@pytest.mark.parametrize("failure", ["missing", "stale", "future", "nan", "yaw", "jump", "reseed"])
def test_estimated_boot_lock_keeps_pose_safety_checks(tmp_path, failure):
    clock = _Clock()
    autonomy = _budget_autonomy(tmp_path, _Backend(), clock)
    autonomy.accept_weak_poses = True
    autonomy.pose_is_weak = lambda: True
    lock = pff.BootPoseLock((0, 0, 0))
    _, stamp, _ = autonomy._observe_boot_pose(lock, None)
    clock.sleep(0.1)
    pose = Pose(
        float("nan") if failure == "nan" else 10.0 if failure == "jump" else 0.0,
        0, 0, float("nan") if failure == "yaw" else 0.0,
        stamp=clock.now() + (1.0 if failure == "future" else -1.0 if failure == "stale" else 0.0),
        map_confirmed=False, position_observed=False,
        reseed_confirming=failure == "reseed",
    )
    autonomy.get_pose = lambda: None if failure == "missing" else pose
    locked, _, _ = autonomy._observe_boot_pose(lock, stamp)
    assert not locked and lock.count == 0


def _forward_motion_backend(clock, forward_mps):
    backend = _Backend()
    backend.state.autonomous_speed_limit_mps = 0.6
    backend.state.ground_speed_mps = abs(forward_mps)
    backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
    # att_yaw 0: body forward is north.
    backend.state.speed_north_mps = forward_mps
    backend.state.speed_east_mps = 0.0
    return backend


def test_drawn_arrival_radius_does_not_accept_unfinished_descent(tmp_path):
    autonomy = _budget_autonomy(tmp_path, _Backend(), _Clock())
    snapshot = replace(autonomy.snapshot, arrive_radius_map_units=0.030215517241379313)
    config = rpf.apply_desktop_auto_authority(rpf.config_for_route(
        snapshot, rpf.production_auto_control_config(LEGACY_MAP_FRAME)
    ))
    controller = rpf.RouteAutoController(snapshot.controller_waypoints(), poles=[], config=config)
    # 0.10 map unit above the first point: inside the old forced 0.15 circle,
    # but still over three times the route's authored arrival radius.
    pose = Pose(0.0, -0.10, 0.0, 0.0, stamp=10.0)
    command = controller.step(pose, now=10.0)
    assert controller.target_index == 0
    assert config.waypoint_arrive_radius == pytest.approx(0.030215517241379313)
    pcmd = rpf.command_to_body_percent(command, pose, config=config)
    assert pcmd[3] < 0


@pytest.mark.parametrize("command", [(0, 50, 7, -3), (0, 0, 7, -3), (0, -1, 7, -3)])
def test_overspeed_requires_braking_without_changing_yaw_or_descent(tmp_path, command):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 1.17)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    accepted, _, applied = autonomy._send_authorized(command)
    assert accepted
    assert applied[1] <= -11
    assert applied[2:] == command[2:]
    assert backend.zeros == []


def test_manual_takeover_finished_event_does_not_claim_auto_landed(tmp_path):
    clock = _Clock()
    backend = _Backend()
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy.land = Mock(return_value=True)

    def run_loop(*_args, **_kwargs):
        backend.pilot_sticks = True
        return "safety LAND command -> land"

    autonomy.run_loop = run_loop
    autonomy._run()
    autonomy.land.assert_not_called()
    finished = next(event for event in autonomy.drain_events() if event.kind == "finished")
    assert "manual" in finished.detail.lower()
    assert "-> land" not in finished.detail
    assert autonomy.interrupted_target_index == autonomy.controller.target_index


@pytest.mark.parametrize("target", [2, 4])
def test_airborne_restart_resumes_unfinished_target_after_boot_lock(tmp_path, target):
    clock = _Clock()
    backend = _Backend()
    snapshot = _snapshot(tmp_path, waypoints=((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)))
    commands = []
    seen_targets = []

    def run_loop(_hooks, controller, *_args, **_kwargs):
        seen_targets.append(controller.target_index)
        assert controller._arrive_frames == 0
        assert controller._final_confirm_frames == 0
        backend.pilot_sticks = True
        return "safety LAND command -> land"

    autonomy = DesktopRouteAutonomy(
        backend=backend, snapshot=snapshot, map_frame=LEGACY_MAP_FRAME,
        get_pose=lambda: Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now()),
        pose_is_weak=lambda: False, pose_confidence=lambda: 100,
        force_relocalize=lambda: None, stream_healthy=lambda: True,
        takeoff=lambda: commands.append("takeoff") or True,
        land=lambda: commands.append("land") or True,
        take_pc_control=lambda: commands.append("pc_control") or True,
        start_airborne=True, resume_target_index=target,
        now=clock.now, sleep=clock.sleep, run_loop=run_loop,
    )
    autonomy._run()
    assert commands == ["pc_control"]
    assert seen_targets == [target]
    assert autonomy.interrupted_target_index == target
    events = autonomy.drain_events()
    assert any(event.kind == "boot_hover" for event in events)
    assert any("manual recovery resumes" in event.detail for event in events)


@pytest.mark.parametrize("heading", np.linspace(-np.pi, np.pi, 17))
def test_overspeed_braking_is_bounded_and_opposes_diagonal_motion(tmp_path, heading):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 1.17)
    backend.state.speed_north_mps = 1.17 * np.cos(heading)
    backend.state.speed_east_mps = 1.17 * np.sin(heading)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    for roll, pitch in [(50, 50), (-50, 50), (50, -50), (-50, -50), (0, 0)]:
        applied = autonomy._apply_speed_limit((roll, pitch, 11, -7))
        projection = applied[1] * np.cos(heading) + applied[0] * np.sin(heading)
        assert projection <= -11.4 + 1e-9
        assert max(abs(applied[0]), abs(applied[1])) <= 50
        assert applied[2:] == (11, -7)


def test_recorded_manual_restart_descends_toward_unfinished_third_waypoint():
    # Flight 2026-09-18 12:11:33: the old restart targeted WP1 above the
    # localized camera. WP3 was unfinished and lies below it. No altimeter input.
    frame = rpf.MapFrame(
        east=np.array([0.9999409166106423, 0.01083349348623922, 0.000893703951008892]),
        north=np.array([0.0, -0.08221525243898248, 0.9966145956518973]),
        up=np.array([0.010870293826223672, -0.9965557122837028, -0.08221039488321148]),
        source="recorded-river-frame",
    )
    config = replace(rpf.production_auto_control_config(frame), waypoint_arrive_radius=0.0302155)
    points = [[0.4494, -0.1904, -1.3942], [0.3144, -0.1999, -1.2688], [0.2687, 0.0629, -1.1334]]
    pose = Pose(0.5537, 0.023, -1.5092, yaw=0.0, stamp=10.0)
    controller = rpf.RouteAutoController(points, poles=[], config=config)
    for target, direction in [(0, 1), (2, -1)]:
        controller.start_after_nearest_waypoint(pose.xyz, target_index=target)
        command = controller.step(pose, now=10.0)
        pcmd = rpf.command_to_body_percent(command, pose, config=config, require_yaw_alignment=False)
        assert controller.target_index == target
        assert pcmd[3] * direction > 0


@pytest.mark.parametrize("speed, command, expected", [
    (0.0, (0, 50, 7, 3), (0, 50, 7, 3)),
    (0.12, (0, 50, 7, 3), (0, 14, 7, 3)),
    (0.3, (0, 50, 7, 3), (0, 9, 7, 3)),
    # Orthogonal roll is preserved: only the forward projection is limited.
    (0.3, (-30, 40, 0, 0), (-30, 9, 0, 0)),
    (0.3, (0, 10, 0, 0), (0, 9, 0, 0)),
    (0.6, (0, 50, 7, 3), (0, 0, 7, 3)),
    (0.9, (0, 50, 0, 0), (0, -9, 0, 0)),
])
def test_horizontal_auto_command_tapers_to_zero_at_speed_limit(
    tmp_path, speed, command, expected
):
    clock = _Clock()
    backend = _forward_motion_backend(clock, speed)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    assert autonomy.controller.cfg.max_translation_pcmd == 50
    # The steady speed envelope is independent of the startup acceleration ramp.
    assert autonomy._apply_speed_limit(command) == expected


@pytest.mark.parametrize("limit", [0.15, 0.3, 0.6, 2.0])
@pytest.mark.parametrize("yaw", [-2.7, 0.0, 1.2])
@pytest.mark.parametrize("heading", [-2.0, 0.0, 0.7, 2.8])
def test_accelerating_tilt_spends_damping_equivalent_speed_headroom(
    tmp_path, limit, yaw, heading
):
    # Full-cap taper used to permit 33 PCMD at .1 m/s with a .3 limit:
    # a high thrust burst during .4-.5 s delay and 1.4-2 s response. Budget
    # headroom in the existing braking units instead of fitting a plant gain.
    clock = _Clock()
    speed = limit / 3.0
    backend = _forward_motion_backend(clock, speed)
    backend.state.autonomous_speed_limit_mps = limit
    backend.state.att_yaw = yaw
    backend.state.speed_north_mps = speed * math.cos(yaw + heading)
    backend.state.speed_east_mps = speed * math.sin(yaw + heading)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    command = (round(50 * math.sin(heading)), round(50 * math.cos(heading)), 7, -3)

    applied = autonomy._apply_speed_limit(command)

    projection = applied[1] * math.cos(heading) + applied[0] * math.sin(heading)
    assert projection <= rpf.DESKTOP_AUTO_WIND_REJECT_PCMD_PER_MPS * (limit - speed) + 1e-9
    assert projection > 0
    assert max(abs(applied[0]), abs(applied[1])) <= 50
    assert applied[2:] == command[2:]


@pytest.mark.parametrize("speed", [0.1, 0.3, 0.6])
def test_headroom_budget_preserves_orthogonal_hold_and_stronger_braking(tmp_path, speed):
    clock = _Clock()
    backend = _forward_motion_backend(clock, speed)
    backend.state.autonomous_speed_limit_mps = 0.3
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    # A speed guard must not scale the position/wind servo as a whole. Full
    # opposing pitch, or a lateral correction at/below the limit, still fits.
    assert autonomy._apply_speed_limit((35, -40, 7, -3)) == (35, -40, 7, -3)
    if speed <= 0.3:
        assert autonomy._apply_speed_limit((35, 0, 7, -3)) == (35, 0, 7, -3)


def test_speed_limit_extrapolates_measured_acceleration_past_telemetry_delay(tmp_path):
    """Flight 2026-09-15: speed kept rising ~0.5 s after the limiter cut tilt."""
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.2)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy._apply_speed_limit((0, 50, 0, 0))
    clock.sleep(0.2)
    backend.state.ground_speed_mps = 0.3
    backend.state.speed_north_mps = 0.3
    backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
    clock.sleep(0.1)

    applied = autonomy._apply_speed_limit((0, 50, 0, 0))

    # Rising delayed telemetry needs less accelerating tilt than the steady
    # .3m/s headroom (9 PCMD), without changing the prediction horizon.
    assert applied[1] < 9
    assert applied[0] == 0 and applied[2:] == (0, 0)
    assert backend.state.autonomous_speed_guard_status == "SPEED_LIMITED"


def test_authorized_brake_does_not_cancel_prediction_before_aircraft_decelerates(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.2)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy._apply_speed_limit((0, 50, 7, -3))
    clock.sleep(0.2)
    backend.state.ground_speed_mps = backend.state.speed_north_mps = 0.4
    backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
    # Brake authorized in the previous tick, but the new measured sample is
    # still accelerating under old tilt. The aircraft has not acted on it yet.
    autonomy._accel_command = (clock.now() - 0.05, np.array([-12.0, 0.0]))
    applied = autonomy._apply_speed_limit((0, 20, 7, -3))
    assert applied[1] < 0 and applied[2:] == (7, -3)
    # Reusing that sample cannot remove the command-delay horizon.
    clock.sleep(0.05)
    assert autonomy._apply_speed_limit((0, 20, 7, -3))[1] <= applied[1]
    # A genuinely decelerating new sample releases prediction, without a latch.
    clock.sleep(0.15)
    backend.state.ground_speed_mps = backend.state.speed_north_mps = 0.3
    backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
    assert autonomy._apply_speed_limit((0, 20, 7, -3)) == (0, 9, 7, -3)


@pytest.mark.parametrize("heading", np.linspace(-np.pi, np.pi, 17))
def test_predicted_overspeed_brake_survives_slew_at_measured_underspeed(tmp_path, heading):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.0)
    backend.state.autonomous_speed_limit_mps = 0.3
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy._send_authorized((0, 50, 7, -3))
    clock.sleep(0.2)
    stamp = clock.now()
    backend.state.ground_speed_mps = 0.29
    backend.state.speed_north_mps = 0.29 * np.cos(heading)
    backend.state.speed_east_mps = 0.29 * np.sin(heading)
    backend.state.ground_speed_mono_ns = int(stamp * 1e9)
    clock.sleep(0.2)

    accepted, _, applied = autonomy._send_authorized((20, 50, 7, -3))

    # 5 Hz telemetry: actual underspeed, but acceleration plus sample age
    # and command lookahead require more than immediate velocity damping.
    predicted = 0.29 + (0.29 / 0.2) * (0.2 + 0.25)
    projection = applied[1] * np.cos(heading) + applied[0] * np.sin(heading)
    assert accepted
    assert projection <= -rpf.DESKTOP_AUTO_WIND_REJECT_PCMD_PER_MPS * (predicted - 0.3) + 1e-9
    assert max(abs(applied[0]), abs(applied[1])) <= 50
    assert applied[2:] == (7, -3)


def test_speed_envelope_does_not_bank_unapproved_acceleration(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.3)
    backend.state.autonomous_speed_limit_mps = 0.3
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    for _ in range(50):
        backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
        assert autonomy._send_authorized((0, 50, 7, -3))[2] == (0, 0, 7, -3)
        clock.sleep(0.05)
    # A new decelerated sample releases the speed envelope without a latch,
    # but cannot resurrect tilt withheld during those stationary commands.
    backend.state.ground_speed_mps = backend.state.speed_north_mps = 0.0
    backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
    accepted, _, applied = autonomy._send_authorized((0, 50, 7, -3))
    assert accepted and 0 < applied[1] <= 2
    assert applied[2:] == (7, -3)


def test_final_speed_envelope_preserves_available_crosswind_authority(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.0)
    backend.state.autonomous_speed_limit_mps = 0.3
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy._send_authorized((20, 50, 7, -3))
    clock.sleep(0.2)
    backend.state.ground_speed_mps = backend.state.speed_north_mps = 0.29
    backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
    clock.sleep(0.2)
    # Crosswind tilt is already established; final braking must reserve
    # along-motion authority without discarding the orthogonal correction.
    autonomy._accel_command = (clock.now() - 0.05, np.array([50.0, 20.0]))
    accepted, _, applied = autonomy._send_authorized((20, 50, 7, -3))
    assert accepted
    assert applied[0] == 20 and applied[1] <= -19
    assert applied[2:] == (7, -3)


def test_quantized_low_speed_reversal_does_not_restore_full_reverse_tilt(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.03)
    backend.state.autonomous_speed_limit_mps = 0.3
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    for event in range(8):
        velocity = 0.03 if event % 2 == 0 else -0.03
        backend.state.ground_speed_mps = abs(velocity)
        backend.state.speed_north_mps = velocity
        backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
        for _ in range(4):
            accepted, _, applied = autonomy._send_authorized(
                (0, -50 if velocity > 0.0 else 50, 7, -3)
            )
            assert accepted
            # Immediate low-speed damping plus four 20 Hz ramp increments,
            # not full opposite tilt held on a delayed quantized speed sign.
            assert 0 < abs(applied[1]) <= 6
            assert applied[1] * velocity < 0.0
            assert applied[2:] == (7, -3)
            clock.sleep(0.05)


def test_braking_command_is_not_limited_above_speed_limit(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.9)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    accepted, _, applied = autonomy._send_authorized((0, -50, 0, 0))
    assert accepted
    assert applied == (0, -50, 0, 0)
    assert backend.state.autonomous_speed_guard_status == "FRESH_SPEED_LIMIT"


def test_stale_velocity_leaves_horizontal_command_unlimited(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.9)
    backend.state.ground_speed_mono_ns = int((clock.now() - 1.0) * 1e9)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    accepted, _, applied = autonomy._send_authorized((0, 50, 0, 0))
    assert accepted
    assert applied == (0, 50, 0, 0)
    assert backend.state.autonomous_speed_guard_status == "SPEED_UNAVAILABLE"


def _progress_tick(clock, *, x=0.0, target=1, **overrides):
    tick = {
        "t": clock.now(), "pose_stamp": clock.now(), "pose_u": [x, 0.0, 0.0],
        "target_index": target, "pcmd_phase": "translate", "action": "FOLLOW",
        "pcmd": [0, 3, 0, 0], "blocked": False, "reason": "FOLLOW",
    }
    tick.update(overrides)
    return tick


def test_stalled_waypoint_stops_motion_and_reports_failure_without_landing(tmp_path):
    clock = _Clock()
    backend = _Backend()
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy.phase = "ROUTE"
    autonomy.land = Mock()
    last = None
    for _ in range(301):
        # Bounded localization jitter must not masquerade as sustained progress.
        last = _progress_tick(clock, x=0.002 if int(clock.now() * 10) % 2 else 0.0)
        autonomy._log_route_tick(last)
        clock.sleep(0.1)
    assert autonomy._auto_failed
    assert "waypoint 2" in autonomy._auto_failure_detail
    assert "progress" in autonomy._auto_failure_detail
    assert backend.zeros[-1][:4] == (0, 0, 0, 0)
    assert any(event.kind == "auto_failed" for event in autonomy.drain_events())
    autonomy.land.assert_not_called()
    accepted, _, applied = autonomy._send_authorized((0, 3, 0, 0))
    assert not accepted and applied == (0, 0, 0, 0)


def test_slow_but_steady_waypoint_progress_does_not_time_out(tmp_path):
    clock = _Clock()
    autonomy = _budget_autonomy(tmp_path, _Backend(), clock)
    autonomy.phase = "ROUTE"
    for index in range(700):
        autonomy._log_route_tick(_progress_tick(clock, x=index * 0.001))
        clock.sleep(0.1)
    assert not autonomy._auto_failed


@pytest.mark.parametrize("interruption", ["turn", "blocked", "missing_pose", "stale_pose", "target", "paused"])
def test_waypoint_stall_history_survives_interruptions_until_target_changes(tmp_path, interruption):
    clock = _Clock()
    autonomy = _budget_autonomy(tmp_path, _Backend(), clock)
    autonomy.phase = "ROUTE"
    for _ in range(200):
        autonomy._log_route_tick(_progress_tick(clock))
        clock.sleep(0.1)
    changes = {
        "turn": {"pcmd_phase": "turn", "pcmd": [0, 0, 20, 0]},
        "blocked": {"blocked": True, "pcmd": [0, 0, 0, 0]},
        "missing_pose": {"pose_u": None},
        "stale_pose": {"pose_stamp": clock.now() - 1.0},
        "target": {"target_index": 2},
        "paused": {},
    }[interruption]
    if interruption == "paused":
        autonomy._paused.set()
    autonomy._log_route_tick(_progress_tick(clock, **changes))
    autonomy._paused.clear()
    clock.sleep(0.1)
    for _ in range(200):
        autonomy._log_route_tick(_progress_tick(clock))
        clock.sleep(0.1)
    assert autonomy._auto_failed is (interruption != "target")


def test_repeated_pose_cannot_refresh_progress_or_trigger_a_false_stall(tmp_path):
    clock = _Clock()
    autonomy = _budget_autonomy(tmp_path, _Backend(), clock)
    autonomy.phase = "ROUTE"
    captured = clock.now()
    for _ in range(400):
        autonomy._log_route_tick(_progress_tick(clock, pose_stamp=captured))
        clock.sleep(0.05)
    assert not autonomy._auto_failed
    assert not autonomy._waypoint_progress.counting
    assert autonomy._waypoint_progress.no_progress_s <= autonomy.controller.cfg.max_pose_age_s


def test_real_route_loop_stops_stalled_translation_without_landing(tmp_path, monkeypatch):
    clock = _Clock()
    backend = _Backend()
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy.phase = "ROUTE"
    autonomy.controller.target_index = 1
    autonomy.land = Mock()
    hooks = autonomy._route_hooks()

    def pose():
        assert clock.now() < 50.0, "stalled route failed to stop within its budget"
        backend.state.ground_speed_mono_ns = int((clock.now() - 0.001) * 1e9)
        return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())

    hooks.get_pose = pose
    monkeypatch.setattr(pff.time, "sleep", clock.sleep)
    pff.run_loop(hooks, autonomy.controller, autonomy.waypoints, verbose=False)
    assert autonomy._auto_failed
    assert "progress" in autonomy._auto_failure_detail
    assert backend.zeros[-1][:4] == (0, 0, 0, 0)
    autonomy.land.assert_not_called()


def test_auto_wait_budget_latches_failure_instead_of_hovering_forever(tmp_path) -> None:
    """A wait AUTO cannot end by itself is bounded, not held indefinitely."""
    clock = _Clock()
    autonomy = _budget_autonomy(tmp_path, _Backend(), clock)

    assert autonomy._check_wait_budget("localization unavailable", 1.0) is False
    assert autonomy._auto_failed is False

    assert autonomy._check_wait_budget("localization unavailable", AUTO_WAIT_BUDGET_S) is True
    assert autonomy._auto_failed is True
    assert autonomy.phase == "AUTO_FAILED"
    events = autonomy.drain_events()
    assert any(event.kind == "auto_failed" for event in events)
    # The failure stops autonomous motion; it must not hand the sticks to a
    # pilot who may not be holding the controller.
    assert autonomy.backend.pilot_sticks is False


@pytest.mark.parametrize(
    ("battery", "expected"),
    [
        (80.0, False),
        (20.0, False),
        (11.0, False),
        (AUTO_BATTERY_RESERVE_PCT, True),
        (9.0, True),
        (float("nan"), True),
    ],
    ids=["healthy", "old-reserve", "above-reserve", "at-reserve", "below-reserve", "unreadable"],
)
def test_auto_runtime_safety_latches_on_battery_reserve(tmp_path, battery, expected) -> None:
    """In-flight AUTO stops on its own battery reserve, not only on firmware RTH."""
    clock = _Clock()
    backend = _Backend()
    backend.is_live = True
    backend.state.battery_pct = battery
    backend.state.telemetry_read_mono_ns = int(clock.now() * 1_000_000_000)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy._mission_started = clock.now()
    autonomy.phase = "ROUTE"

    assert autonomy._check_runtime_safety() is expected
    assert autonomy._auto_failed is expected


def test_auto_runtime_has_no_fixed_duration_but_retains_battery_guard(tmp_path) -> None:
    clock = _Clock()
    backend = _Backend()
    backend.is_live = True
    backend.state.battery_pct = 80.0
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy._mission_started = clock.now()
    autonomy.phase = "ROUTE"

    clock.sleep(601.0)
    backend.state.telemetry_read_mono_ns = int(clock.now() * 1_000_000_000)
    accepted, _reason, applied = autonomy._send_authorized((0, 5, 0, 0))
    assert accepted and applied[1] > 0
    assert autonomy._auto_failed is False

    backend.state.battery_pct = AUTO_BATTERY_RESERVE_PCT
    accepted, _reason, applied = autonomy._send_authorized((0, 5, 0, 0))
    assert not accepted and applied == (0, 0, 0, 0)
    assert autonomy.phase == "AUTO_FAILED"


@pytest.mark.parametrize("landing_result", [False, RuntimeError("landing failed")])
def test_manual_takeover_during_failed_landing_retires_the_auto_retry(
    tmp_path, landing_result
) -> None:
    """Takeover while the first land() is in flight: no second autonomous land.

    ``land_cmd`` re-claims PC control on the real backend, so a retry fired after
    the operator already has the sticks would take the aircraft back mid-recovery.
    The retry must re-check authority, not just the result of the previous call.
    """
    clock = _Clock()
    backend = _Backend()
    autonomy = None

    def take_over_then_fail():
        # The operator grabs the sticks while this first land() is still running.
        autonomy._cancel.set()
        backend.pilot_sticks = True
        if isinstance(landing_result, Exception):
            raise landing_result
        return landing_result

    land = Mock(side_effect=take_over_then_fail)
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
        boot_timeout_s=1.0,
        now=clock.now,
        sleep=clock.sleep,
        run_loop=lambda *_args, **_kwargs: "route complete",
    )

    autonomy._land_after_failure("route complete -> land")

    assert land.call_count == 1
    assert autonomy._landing_confirmed is False
    events = autonomy.drain_events()
    assert not any(event.kind == "landing_unresolved" for event in events)


@pytest.mark.parametrize("landing_result", [False, RuntimeError("landing failed")])
def test_unconfirmed_landing_retries_and_stays_unresolved(tmp_path, landing_result) -> None:
    """Both land() attempts unconfirmed: the worker ends, the mission does not.

    ``finished``/``DONE`` is the operator's evidence that the aircraft is on the
    ground, so an unconfirmed landing must not claim it.  The worker thread still
    exits; the phase stays ``LANDING_UNRESOLVED`` so the unresolved landing keeps
    its supervision owner.
    """
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
    assert not any(event.kind == "finished" for event in events)
    assert autonomy._landing_confirmed is False
    assert autonomy.phase == "LANDING_UNRESOLVED"


def test_cancelled_run_releases_for_a_fresh_start_after_join(tmp_path) -> None:
    """Manual takeover during ROUTE, then AUTO again: the old run must join to DONE."""
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
        run_loop=lambda *_args, **_kwargs: "route done",
    )

    assert autonomy.start() is True
    autonomy.cancel("manual_takeover")
    assert autonomy.join(timeout=2.0) is True
    assert autonomy.phase == "DONE"
    assert any(event.kind == "finished" for event in autonomy.drain_events())

    assert autonomy.start() is True
    autonomy.cancel("manual_takeover")
    assert autonomy.join(timeout=2.0) is True
    assert autonomy.phase == "DONE"


def test_send_authorized_race_with_failure_clears_nudge_and_zeros_pcmd(
    tmp_path,
) -> None:
    class RacingBackend(_Backend):
        def __init__(self):
            super().__init__()
            self._nudge_vector = ("unset",)
            self._entered = threading.Event()
            self._release = threading.Event()
            self.state.autonomous_speed_limit_enabled = False

        def set_nudge_vector(self, roll, pitch, yaw, gaz, *, authority_pct=10, auto_run_id=None):
            self._entered.set()
            assert self._release.wait(timeout=2.0)
            self._nudge_vector = (roll, pitch, yaw, gaz)
            self.vectors.append((roll, pitch, yaw, gaz))
            return True

        def clear_nudge_vector(self):
            self._nudge_vector = None
            self.vectors.append((0.0, 0.0, 0.0, 0.0))

        def send_pcmd(self, roll, pitch, yaw, gaz, *, reason, auto_run_id=None):
            self.zeros.append((roll, pitch, yaw, gaz, reason))
            return True

    backend = RacingBackend()
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
    worker = threading.Thread(
        target=autonomy._send_authorized,
        args=((10, 0, 0, 0),),
        daemon=True,
    )
    worker.start()
    assert backend._entered.wait(timeout=2.0)
    autonomy._latch_auto_failure("race")
    backend._release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert backend._nudge_vector is None
    assert backend.zeros[-1][:4] == (0, 0, 0, 0)


def test_route_hooks_feed_firmware_altitude_and_announce_a_height_waiver(tmp_path):
    clock = _Clock()
    backend = _Backend()
    backend.state.drone_altitude_m = 2.5
    backend.state.altitude_mono_ns = int(clock.now() * 1e9)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    assert autonomy._route_hooks().altitude() == (2.5, pytest.approx(clock.now()))
    backend.state.drone_altitude_m = float("nan")
    assert autonomy._altitude() is None

    autonomy._log_route_tick({
        "target_index": 0,
        "vertical_waived_now": True,
        "vertical_guard": {"baro_moved_m": 1.02, "localized_followed_u": -0.09},
    })
    events = []
    while not autonomy.events.empty():
        events.append(autonomy.events.get_nowait())
    waiver = [event for event in events if event.kind == "vertical_unobservable"]
    assert len(waiver) == 1
    assert "1.02 m" in waiver[0].detail and "-0.09" in waiver[0].detail


def test_accelerating_tilt_ramps_and_pause_discards_previous_acceleration(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    for index in range(4):
        clock.value = 10. + index * .05
        backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
        accepted, _, applied = autonomy._send_authorized((0, 50, 7, -3))
        assert accepted and 0 < applied[1] <= 3 * (index + 1)
        assert applied[2:] == (7, -3)
    autonomy._clear_motion()
    clock.sleep(.05)
    assert autonomy._send_authorized((0, 50, 7, -3))[2][1] <= 3
    backend.state.speed_north_mps = .9
    assert autonomy._send_authorized((0, -50, 7, -3))[2] == (0, -50, 7, -3)


def test_low_authority_acceleration_accumulates_fractional_pcmd(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy.controller.cfg = replace(autonomy.controller.cfg, max_translation_pcmd=3)
    commands = []
    for index in range(30):
        clock.value = 10. + index * .05
        backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
        commands.append(autonomy._send_authorized((0, 3, 0, 0))[2][1])
    assert commands[0] == 0 and max(commands) >= 2


def test_first_motion_sample_measures_acceleration_from_the_stationary_sample(tmp_path):
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    autonomy._apply_speed_limit((0, 50, 0, 0))
    clock.sleep(.2)
    backend.state.speed_north_mps = backend.state.ground_speed_mps = .2
    backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
    clock.sleep(.1)
    assert autonomy._apply_speed_limit((0, 50, 0, 0))[1] <= 5


@pytest.mark.parametrize('response_s', [.2, .4, .6])
def test_delayed_five_hz_speed_does_not_allow_a_full_tilt_launch(tmp_path, response_s):
    # Stress model only: 0.04 m/s per PCMD, 250 ms delayed 5 Hz telemetry.
    # This identifies a startup overshoot mechanism, not an ANAFI calibration.
    clock = _Clock()
    backend = _forward_motion_backend(clock, 0.)
    autonomy = _budget_autonomy(tmp_path, backend, clock)
    speed, history = 0., []
    for index in range(160):
        clock.value = 10. + index * .05
        history.append(speed)
        if index % 4 == 0:
            sampled = max(0, index - 5)
            backend.state.speed_north_mps = history[sampled]
            backend.state.ground_speed_mps = abs(history[sampled])
            backend.state.ground_speed_mono_ns = int((10. + sampled * .05) * 1e9)
        accepted, _, applied = autonomy._send_authorized((0, 50, 0, 0))
        assert accepted
        speed += (.04 * applied[1] - speed) * (1. - np.exp(-.05 / response_s))
    assert max(abs(v) for v in history) <= .6
    assert sum(history) * .05 > 1., 'passing requires forward travel, not a zero-command stop'


def test_real_route_loop_reports_observed_geometry_not_holdover_or_prediction(tmp_path, monkeypatch):
    """Real coordinator/controller/loop; backend and localization are offline only."""
    clock = _Clock()
    backend = _Backend()
    backend.session_logs = _SessionLogs()
    auto = _budget_autonomy(tmp_path, backend, clock)
    auto.run_loop = pff.run_loop
    monkeypatch.setattr(pff.time, "sleep", clock.sleep)
    post_reads = 0
    previous_stamp = None

    def pose():
        nonlocal post_reads, previous_stamp
        assert clock.now() < 20.0, "offline probe must not leave a worker running"
        backend.state.ground_speed_mono_ns = int(clock.now() * 1e9)
        if auto.phase != "ROUTE" or auto.controller.target_index == 0:
            return Pose(0.0, 0.0, 0.0, 0.0, stamp=clock.now())
        post_reads += 1
        if post_reads == 12:
            auto.cancel("offline geometry probe")
        stamp = clock.now()
        if post_reads == 2:
            stamp = previous_stamp
        elif post_reads == 3:
            stamp -= 1.0
        else:
            previous_stamp = stamp
        return Pose(
            0.5, 0.3, 0.4, 0.0, stamp=stamp,
            position_observed=post_reads != 5,
        )

    auto.get_pose = pose
    auto.pose_info = lambda: {
        "display_seq": post_reads, "pose_capture_mono_ns": int(clock.now() * 1e9),
        "direct_status": "OFFLINE_OBSERVATION",
    }
    auto._run()
    summaries = [event for event in auto.drain_events() if event.kind == "route_error_summary"]
    assert len(summaries) == 1
    summary = summaries[0].result
    ticks = [fields for event, fields in backend.session_logs.records if event == "auto_route_tick"]
    accepted = [tick for tick in ticks if tick["route_error_sample"]]
    offset_ticks = [tick for tick in accepted if tick["pose_u"] == [0.5, 0.3, 0.4]]
    assert offset_ticks
    assert all(tick["route_distance_u"] == pytest.approx(0.5) for tick in offset_ticks)
    assert all(tick["route_nearest_u"] == pytest.approx([0.5, 0.0, 0.0]) for tick in offset_ticks)
    assert any(tick.get("pose_source") == "holdover" and not tick["route_error_sample"] for tick in ticks)
    assert any(tick.get("position_observed") is False and not tick["route_error_sample"] for tick in ticks)
    assert len({tick["pose_stamp"] for tick in accepted}) == len(accepted)
    assert summary["samples"] == len(accepted)
    assert summary["mean_u"] == round(sum(tick["route_distance_u"] for tick in accepted) / len(accepted), 4)
    assert summary["rms_u"] == round(np.sqrt(sum(tick["route_distance_u"] ** 2 for tick in accepted) / len(accepted)), 4)
    assert summary["max_u"] == summary["p95_u"] == 0.5
    assert summary["horizontal_max_u"] == 0.4
    assert summary["outcome"] == "cancelled"
    assert all(tick["auto_run_id"] == summary["run_id"] for tick in ticks)
    assert any(tick.get("loc_pose_stamp_matches") is False for tick in ticks)
    auto._report_route_error()
    assert not auto.drain_events()


@pytest.mark.parametrize("terminal", ["complete", "cancel", "failure"])
def test_worker_always_emits_one_terminal_route_error_summary(tmp_path, terminal):
    clock = _Clock()
    auto = _budget_autonomy(tmp_path, _Backend(), clock)

    def route_loop(*_args, **_kwargs):
        if terminal == "cancel":
            auto.cancel("offline cancellation")
        elif terminal == "failure":
            raise RuntimeError("offline worker failure")
        return "offline complete"

    auto.run_loop = route_loop
    auto._run()
    auto._report_route_error("must not duplicate")
    events = [event for event in auto.drain_events() if event.kind == "route_error_summary"]
    assert len(events) == 1
    assert events[0].result["state"] == "no_samples"
    assert events[0].result["outcome"] == {
        "complete": "landed", "cancel": "cancelled", "failure": "failed",
    }[terminal]


def test_restarting_same_coordinator_keeps_route_error_runs_independent(tmp_path):
    clock = _Clock()
    auto = _budget_autonomy(tmp_path, _Backend(), clock)
    offsets = iter((0.1, 0.2))

    def route_loop(hooks, *_args, **_kwargs):
        distance = next(offsets)
        hooks.log_tick({"t": clock.now(), "target_index": 0})
        hooks.log_tick({
            "t": clock.now(), "pose_stamp": clock.now(), "target_index": 1,
            "pose_u": [0.5, 0.0, distance], "route_nearest_u": [0.5, 0.0, 0.0],
            "route_distance_u": distance, "action": "FOLLOW",
        })
        return "offline complete"

    auto.run_loop = route_loop
    results = []
    for _ in range(2):
        assert auto.start()
        assert auto.join(timeout=2.0)
        events = [event for event in auto.drain_events() if event.kind == "route_error_summary"]
        assert len(events) == 1
        results.append(events[0].result)
    assert results[0]["run_id"] != results[1]["run_id"]
    assert [result["samples"] for result in results] == [1, 1]
    assert [result["mean_u"] for result in results] == [0.1, 0.2]


def test_auto_backend_dispatch_carries_run_identity_without_changing_commands(tmp_path):
    auto = _budget_autonomy(tmp_path, _Backend(), _Clock())
    setter = Mock(wraps=auto.backend.set_nudge_vector)
    auto.backend.set_nudge_vector = setter
    accepted, _, actual = auto._dispatch_route_pcmd((0, 4, 7, -2))
    assert accepted and actual == (0, 4, 7, -2)
    setter.assert_called_once()
    assert setter.call_args.kwargs["auto_run_id"] == auto.run_id
    assert setter.call_args.kwargs["authority_pct"] == auto._pcmd_authority_pct()

    auto.backend.set_nudge_vector = None
    send = Mock(wraps=auto.backend.send_pcmd)
    auto.backend.send_pcmd = send
    accepted, _, actual = auto._dispatch_route_pcmd((0, 4, 7, -2))
    assert accepted and actual == (0, 4, 7, -2)
    send.assert_called_once_with(
        0, 4, 7, -2, reason="desktop_auto_route", auto_run_id=auto.run_id,
    )
    send.reset_mock()
    assert auto._send_zero("offline stop")
    send.assert_called_once_with(
        0, 0, 0, 0, reason="offline stop", auto_run_id=auto.run_id,
    )


def test_waypoint_camera_dispatch_uses_typed_command_and_fresh_event_readback(tmp_path):
    from backend_contract import ControlAction, ControlResult
    from operator_flight_safety import TelemetryFreshnessStore
    from olympe_live_backend import OlympeLiveBackend
    from types import MethodType

    clock = _Clock()
    backend = _Backend()
    backend.is_live = True
    backend.state.battery_pct = 80
    backend.state.gimbal_pitch_deg = -30.
    backend.state.gimbal_mono_ns = 10_000_000_000
    backend.state.gimbal_stamp_source = "observed_cache_value"
    backend.telemetry_freshness = TelemetryFreshnessStore()
    backend.gimbal_pitch_sample = MethodType(OlympeLiveBackend.gimbal_pitch_sample, backend)
    backend.telemetry_freshness.observe(
        "gimbal", {"gimbal_id": 0, "pitch_absolute": -30.},
        marker=("cache", "old"), observed_mono_ns=10_000_000_000,
    )
    requests = []

    def command(request):
        requests.append(request)
        return ControlResult.completed(backend.state, raw_result=True)

    backend.command = command
    auto = _budget_autonomy(tmp_path, backend, clock)
    auto.phase = "ROUTE"
    assert auto._request_waypoint_gimbal_pitch(0)
    assert requests[0].action is ControlAction.GIMBAL_PITCH
    assert requests[0].payload.value == 0 and not requests[0].human_origin
    assert auto._read_waypoint_gimbal_pitch() is None
    backend.telemetry_freshness.observe(
        "gimbal", {"gimbal_id": 0, "pitch_absolute": -30.},
        marker=("event", "confirmed"), observed_mono_ns=10_000_000_000,
    )
    backend.state.gimbal_pitch_deg = 70.  # Unpaired UI state cannot fabricate readback.
    assert auto._read_waypoint_gimbal_pitch() == (-30., 10.)
    assert auto.pause()
    assert not auto._request_waypoint_gimbal_pitch(45)
    assert auto._read_waypoint_gimbal_pitch() is None
    assert auto.resume()
    backend.pilot_sticks = True
    assert not auto._request_waypoint_gimbal_pitch(45)
    backend.pilot_sticks = False
    auto._cancel.set()
    assert not auto._request_waypoint_gimbal_pitch(45)
    assert len(requests) == 1


def test_waypoint_camera_session_dispatch_confirmation_pause_and_result_events(tmp_path):
    from backend_contract import ControlResult
    from route_domain import WaypointAction

    clock = _Clock()
    backend = _Backend()
    backend.is_live = False
    backend.state.gimbal_pitch_deg = 20.
    backend.state.gimbal_mono_ns = 9_000_000_000
    backend.state.gimbal_stamp_source = "simulated"
    requests = []

    def command(request):
        requests.append(request)
        backend.state.gimbal_pitch_deg = request.payload.value
        return ControlResult.completed(backend.state, raw_result=True)

    backend.command = command
    auto = _budget_autonomy(tmp_path, backend, clock)
    auto.phase = "ROUTE"
    auto.controller = rpf.RouteAutoController(
        auto.waypoints,
        config=replace(auto.controller.cfg, waypoint_arrive_confirm_frames=1,
                       waypoint_actions=((WaypointAction("gimbal_pitch", 0),), ())),
        request_gimbal_pitch=auto._request_waypoint_gimbal_pitch,
        read_gimbal_pitch=auto._read_waypoint_gimbal_pitch,
    )

    def tick(now):
        clock.value = now
        return auto.controller.step(Pose(0, 0, 0, 0., stamp=now), now=now)

    assert tick(10.).action == "WAYPOINT_ACTION" and len(requests) == 1
    assert auto.pause()
    clock.value = 50.
    assert auto.resume()
    assert tick(50.).waypoint_action["elapsed_s"] == 0
    backend.state.gimbal_mono_ns = 50_100_000_000
    assert tick(50.1).action == "WAYPOINT_ACTION"
    backend.state.gimbal_mono_ns = 50_200_000_000
    assert tick(50.2).action == "FOLLOW"
    assert len(requests) == 1
    auto._track_route_progress({"target_index": 1})
    auto._track_route_progress({"target_index": 1})
    results = [event for event in auto.drain_events() if event.kind == "waypoint_camera_action"]
    assert len(results) == 1 and results[0].result["state"] == "confirmed"


def test_simulated_route_consumer_delegates_camera_only_and_emits_simulated_readback():
    from backend_contract import ControlAction, ControlRequest, ScalarPayload
    from simulated_backend import DroneBackend

    backend = DroneBackend()
    plant = backend.route_test_plant
    camera = ControlRequest.create(
        ControlAction.GIMBAL_PITCH, payload=ScalarPayload(-45), human_origin=False,
    )
    assert not plant.command(camera).accepted  # No active simulated session.
    plant._map_frame = LEGACY_MAP_FRAME
    plant._vertical_pcmd_gain = 0.06
    assert plant.command(camera).accepted
    assert backend.state.gimbal_pitch_deg == -45
    plant.send_pcmd(0, 0, 0, 0, reason="offline_camera_observation")
    assert backend.state.gimbal_stamp_source == "simulated"
    assert backend.state.gimbal_mono_ns > 0
    zoom = ControlRequest.create(ControlAction.ZOOM, payload=ScalarPayload(2), human_origin=False)
    assert not plant.command(zoom).accepted
    plant.pilot_sticks = True
    assert not plant.command(camera).accepted
    assert backend.state.gimbal_pitch_deg == -45  # No implicit reset on cancellation.


@pytest.mark.parametrize("backend_kind", ["simulated", "live-deferred"])
def test_session_speed_limit_controls_real_guard_and_idempotent_start(
    tmp_path, monkeypatch, backend_kind,
):
    from backend_contract import SessionConfig
    from simulated_backend import DroneBackend

    if backend_kind == "simulated":
        backend = DroneBackend()
    else:
        import flight_operator_app as app
        import olympe_live_backend as live

        monkeypatch.setattr(live, "_DEFAULT_RECORD_DIR", tmp_path / "recordings")
        backend = live.OlympeLiveBackend(
            app.DroneState, app.ANAFI,
            event_log=SimpleNamespace(event=lambda *a, **kw: None),
            auto_pc_control=False, with_video=False, defer_connect=True,
        )
        assert backend.drone is None and not backend._connection_attempted
    config = SessionConfig(
        session_id="speed-policy", interface_mode=backend.mode,
        site_profile="", site_profile_sha256="", asset_sha256={},
        runtime_profile_sha256="", source="", offline=True,
        autonomous_speed_limit_mps=.35,
    )
    assert backend.start(config).started
    clock = _Clock()
    state = backend.state
    state.autonomous_speed_limit_enabled = True
    state.att_yaw = 0.
    state.att_roll = state.att_pitch = 0.
    state.attitude_mono_ns = state.ground_speed_mono_ns = round(clock.now() * 1e9)
    state.ground_speed_mps = state.speed_north_mps = .5
    state.speed_east_mps = 0.
    auto = _budget_autonomy(tmp_path, backend, clock)

    applied = auto._apply_speed_limit((0, 50, 7, -3))
    assert applied[1] < 0
    assert applied[2:] == (7, -3)
    state.autonomous_speed_limit_mps = .8
    assert backend.start(config).started
    assert auto._apply_speed_limit((0, 50, 7, -3))[1] > 0

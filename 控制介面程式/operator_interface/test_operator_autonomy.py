from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from operator_autonomy import DesktopRouteAutonomy
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
            telemetry_read_mono_ns=10_000_000_000,
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

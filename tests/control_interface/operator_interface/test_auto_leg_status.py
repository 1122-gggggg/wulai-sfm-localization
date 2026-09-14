"""Live AUTO leg/arrival display stays truthful across target advances."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from operator_autonomy import DesktopRouteAutonomy
from real_path_follow_controller import LEGACY_MAP_FRAME, MissionRouteSnapshot


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


def _snapshot(tmp_path) -> MissionRouteSnapshot:
    route_waypoints = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps({"waypoints": route_waypoints, "frame": "glomap", "units": "map"}),
        encoding="utf-8",
    )
    return MissionRouteSnapshot(
        path=route,
        sha256=hashlib.sha256(route.read_bytes()).hexdigest(),
        site_id="field-a",
        coordinate_frame_id="glomap-a",
        waypoints=route_waypoints,
    )


def _tick_record(*, target, phase="translate", reason="FOLLOW", distance=0.12):
    return {
        "t": 100.0,
        "action": "FOLLOW",
        "pcmd_phase": phase,
        "target_index": target,
        "route_distance_u": distance,
        "pose_u": [0.4, 0.0, -1.6],
        "pcmd": [0, 1, 0, 0],
        "blocked": False,
        "reason": reason,
    }


def _basic_autonomy(tmp_path, **kwargs):
    return DesktopRouteAutonomy(
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
        **kwargs,
    )


def test_route_progress_announces_arrivals_and_leg_line(tmp_path) -> None:
    autonomy = _basic_autonomy(tmp_path)
    autonomy.phase = "ROUTE"
    autonomy.controller.target_index = 1
    autonomy._log_route_tick(_tick_record(target=1))
    autonomy._log_route_tick(_tick_record(target=1, distance=0.10))
    kinds = [event.kind for event in autonomy.drain_events()]
    assert "waypoint_arrived" not in kinds
    assert "leg_stage" in kinds
    autonomy.controller.target_index = 2
    autonomy._log_route_tick(_tick_record(target=2, phase="turn", distance=0.05))

    arrivals = [
        event for event in autonomy.drain_events() if event.kind == "waypoint_arrived"
    ]
    assert len(arrivals) == 1
    assert arrivals[0].detail == "到達路徑點2"
    status = autonomy.auto_leg_status()
    assert status["active"] is True
    assert status["target"] == 2
    assert status["leg"] == "返程前往路徑點1"
    assert status["arrived"] == ["路徑點2"]
    assert status["action"] == "旋轉機頭朝向路徑點1（返程）"
    assert "返程前往路徑點1" in status["line"]
    assert "已到達" in status["line"]


def test_leg_line_walks_through_search_turn_aligned_enroute_arrival(tmp_path) -> None:
    autonomy = _basic_autonomy(tmp_path)
    autonomy.phase = "ROUTE"
    autonomy.controller.target_index = 1

    autonomy._log_route_tick(_tick_record(target=1, phase=None, reason="FOLLOW"))
    autonomy._log_route_tick(_tick_record(target=1, phase=None, reason="FOLLOW"))
    no_pose = _tick_record(target=1, phase=None, reason="FOLLOW")
    no_pose["pose_u"] = None
    no_pose["route_distance_u"] = None
    autonomy._log_route_tick(no_pose)
    assert autonomy.auto_leg_status()["action"] == "找尋路徑點2中"
    assert autonomy.auto_leg_status()["stage"] == "searching_pose"

    search = _tick_record(
        target=1, phase=None, reason="right yaw localization search 12.0/360.0deg"
    )
    search["localization_yaw_search"] = "searching_right"
    search["localization_yaw_search_deg"] = 12.0
    autonomy._log_route_tick(search)
    status = autonomy.auto_leg_status()
    assert status["stage"] == "yaw_search"
    assert "找定位旋轉中" in status["action"]

    autonomy._log_route_tick(_tick_record(target=1, phase="turn"))
    assert autonomy.auto_leg_status()["action"] == "旋轉機頭朝向路徑點2"
    autonomy._log_route_tick(_tick_record(target=1, phase="yaw_alignment_confirmed"))
    status = autonomy.auto_leg_status()
    assert status["action"] == "已朝向路徑點2"
    assert status["stage"] == "aligned"
    autonomy._log_route_tick(_tick_record(target=1, phase="translate"))
    assert autonomy.auto_leg_status()["action"] == "前往路徑點2"

    arriving = _tick_record(target=1, phase="translate")
    arriving["action"] = "waypoint 2 arrival confirm 1/1"
    autonomy._log_route_tick(arriving)
    status = autonomy.auto_leg_status()
    assert status["stage"] == "arriving"
    assert "到達路徑點2確認中" in status["action"]
    stages = [event.kind for event in autonomy.drain_events() if event.kind == "leg_stage"]
    assert stages, "every waypoint phase must emit a stage event for the log"


def test_rejoin_and_centering_are_not_reported_as_hover(tmp_path) -> None:
    autonomy = _basic_autonomy(tmp_path)
    autonomy.phase = "ROUTE"
    for phase, text in (
        ("route_rejoin", "回到航段中"), ("waypoint_centering", "位置與高度調整中"),
        ("turn_brake", "水平減速後轉向"),
        ("height_adjust", "修正高度中"),
    ):
        autonomy._log_route_tick(_tick_record(target=1, phase=phase))
        status = autonomy.auto_leg_status()
        assert status["stage"] == phase
        assert text in status["action"]
        blocked = _tick_record(target=1, phase=phase, reason="AUTO speed guard -> HOVER")
        blocked["blocked"] = True
        blocked["pcmd"] = [0, 0, 0, 0]
        autonomy._log_route_tick(blocked)
        assert autonomy.auto_leg_status()["stage"] == "speed_guard"


def test_map_confirmation_wait_is_not_reported_as_an_arrival(tmp_path) -> None:
    autonomy = _basic_autonomy(tmp_path)
    autonomy.phase = "ROUTE"
    autonomy.controller.target_index = 1
    record = _tick_record(target=1, phase="idle")
    record["action"] = "WAIT_MAP_CONFIRMATION"
    record["pcmd"] = [0, 0, 0, 0]
    autonomy._log_route_tick(record)
    status = autonomy.auto_leg_status()
    assert status["stage"] == "map_confirmation"
    assert status["action"] == "等待地圖定位確認（路徑點2）"
    assert not status["arrived"]


def test_boot_phase_prefixes_the_leg_stage(tmp_path) -> None:
    autonomy = _basic_autonomy(tmp_path)
    autonomy.phase = "BOOT_HOVER"
    autonomy.controller.target_index = 1
    autonomy._log_route_tick(_tick_record(target=1, phase="translate"))
    status = autonomy.auto_leg_status()
    assert "找尋定位中" in status["line"]
    assert "前往路徑點2" in status["line"]


def test_leg_status_reports_waiting_without_pose(tmp_path) -> None:
    autonomy = _basic_autonomy(tmp_path)
    autonomy.phase = "ROUTE"
    autonomy.controller.target_index = 1
    record = _tick_record(target=1, phase=None, reason="no fresh pose -> hover (3.0s)")
    record["pose_u"] = None
    record["route_distance_u"] = None
    autonomy._log_route_tick(record)

    status = autonomy.auto_leg_status()
    assert status["action"] == "找尋路徑點2中"
    assert status["stage"] == "searching_pose"
    assert "前往路徑點2" in status["line"]


def test_weak_accept_flag_opens_loop_gate_only_when_opted_in(tmp_path) -> None:
    for flag, expected_gate in ((False, True), (True, False)):
        captured: dict = {}

        def run_loop(_hooks, _controller, _waypoints, **kwargs):
            captured.update(kwargs)
            return "route complete"

        autonomy = _basic_autonomy(
            tmp_path, accept_weak_poses=flag, run_loop=run_loop
        )
        assert autonomy.accept_weak_poses is flag
        assert autonomy._run_route() == "route complete"
        assert captured["enforce_weak_pose_gate"] is expected_gate


def test_explicit_search_config_still_opts_into_spin(tmp_path) -> None:
    from operator_localization_search import YawSearchConfig

    autonomy = _basic_autonomy(
        tmp_path, boot_search_config=YawSearchConfig(yaw_pcmd=12)
    )
    assert autonomy.spin_to_search is True
    assert autonomy._route_hooks().localization_yaw_search_pcmd == 12


def test_blocked_tick_without_pose_keeps_the_actual_stop_reason(tmp_path):
    autonomy = _basic_autonomy(tmp_path)
    autonomy.phase = "ROUTE"
    record = _tick_record(target=1, reason="AUTO speed guard -> HOVER")
    record.update(blocked=True, pose_u=None, pcmd=[0, 0, 0, 0])
    autonomy._log_route_tick(record)
    assert autonomy.auto_leg_status()["stage"] == "speed_guard"


def test_final_arrival_is_emitted_once_and_done_replaces_motion(tmp_path):
    autonomy = _basic_autonomy(tmp_path)
    autonomy.phase = "ROUTE"
    record = _tick_record(target=2, phase="idle")
    record["action"] = "final path reached -> LAND"
    autonomy._log_route_tick(record)
    autonomy._log_route_tick(record)
    arrivals = [event for event in autonomy.drain_events() if event.kind == "waypoint_arrived"]
    assert len(arrivals) == 1
    assert arrivals[0].detail == "到達路徑點1（返程）"
    autonomy.phase = "DONE"
    assert autonomy.auto_leg_status()["action"] == "已結束"


def test_status_panel_keeps_finished_run_and_does_not_show_stale_tracking():
    from flight_operator_app import OperatorApp

    lines = []
    app = SimpleNamespace(
        auto_leg_var=SimpleNamespace(set=lines.append),
        backend=SimpleNamespace(state=SimpleNamespace(flight_state="landed", control_owner="PC")),
        _integrated_autonomy=None,
        _last_auto_status={"line": "已結束｜已到達：路徑點2、路徑點1（返程）"},
        _latest_tracking_mode="FAST_TRACK", loc_health="FAIL",
    )
    OperatorApp._update_auto_leg_readout(app)
    assert "已結束" in lines[-1] and "路徑點2" in lines[-1]
    assert "定位失效" in lines[-1] and "FAST_TRACK" not in lines[-1]
    assert "已落地" in lines[-1]


def test_event_drain_displays_failure_and_invokes_the_finish_handler():
    from unittest.mock import Mock
    from flight_operator_app import OperatorApp

    events = [SimpleNamespace(kind=kind, detail=text) for kind, text in (
        ("leg_stage", "修正朝向"), ("waypoint_arrived", "到達路徑點2"),
        ("auto_failed", "定位中斷"), ("finished", "已懸停"),
    )]
    app = Mock()
    app._integrated_autonomy = SimpleNamespace(drain_events=lambda: events)
    app.auto_status_events = Mock()
    app.auto_status_events.index.return_value = "5.0"
    OperatorApp._drain_integrated_autonomy_events(app)
    app._log_integrated_auto_failed.assert_called_once_with(events[2])
    app._finish_integrated_auto_event.assert_called_once_with(events[3])
    lines = [call.args[1] for call in app.auto_status_events.insert.call_args_list]
    assert len(lines) == 4
    assert "到達路徑點2" in lines[1] and "自動飛行失敗" in lines[2]


def test_finish_handler_preserves_last_status_before_clearing_coordinator():
    from flight_operator_app import OperatorApp

    status = {"line": "已結束｜已到達：路徑點2"}
    app = SimpleNamespace(
        _pending_auto_route_activation=False,
        _integrated_autonomy=SimpleNamespace(auto_leg_status=lambda: status),
        write_log=lambda text: None, _set_auto_paused=lambda value: None,
    )
    OperatorApp._finish_integrated_auto_event(app, SimpleNamespace(detail="complete"))
    assert app._integrated_autonomy is None
    assert app._last_auto_status == status

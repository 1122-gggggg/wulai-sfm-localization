from __future__ import annotations

import flight_operator_app as app
import inspect
import queue
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flight_operator_app import (
    DroneState,
    OperatorApp,
    PREFLIGHT_GUIDE_STEPS,
    SequentialPreflightGuide,
    UI_LOG_MAX_LINES,
    _positive_env_float,
    _positive_env_int,
    format_magnetometer_calibration,
    format_olympe_telemetry,
    gravity_phase_guidance,
    next_tick_deadline,
    video_hud_identity,
)


class _Backend:
    def __init__(self):
        self.calls = []

    def command(self, name, **payload):
        self.calls.append((name, payload))
        return name


def _bare_app(backend=None):
    app = OperatorApp.__new__(OperatorApp)
    app.backend = backend or _Backend()
    app._nudge_keys_held = set()
    app._nudge_buttons_held = set()
    app._nudge_key_map = {"w": "前"}
    app._stick_vector_active = False
    app._flight_results = queue.Queue()
    app._flight_inflight = set()
    app._flight_inflight_lock = threading.Lock()
    app.control_owner_var = SimpleNamespace(set=lambda _value: None)
    app.write_log = lambda _message: None
    return app


class _RouteLock:
    def __init__(self, *, digest="a" * 64, fail=None):
        self.active = False
        self.fail = fail
        self.snapshot = SimpleNamespace(
            path=Path("/tmp/selected-route.json"),
            sha256=digest,
            site_id="field-a",
            coordinate_frame_id="glomap-a",
        )

    def begin_auto(self, *, displayed_sha256):
        if self.fail:
            raise ValueError(self.fail)
        assert displayed_sha256 == self.snapshot.sha256
        self.active = True
        return self.snapshot

    def confirm_auto_started(self):
        self.active = True


def test_preflight_guide_is_sequential_and_invalidates_downstream() -> None:
    guide = SequentialPreflightGuide()

    assert guide.current_step == PREFLIGHT_GUIDE_STEPS[0]
    for index, step in enumerate(PREFLIGHT_GUIDE_STEPS):
        assert guide.current_step == step
        guide.confirm_current((step, index))
    assert guide.complete

    invalidated = guide.sync({
        "compass": ("compass", "changed"),
        "map": ("map", 1),
        "route": ("route", 2),
        "system": ("system", 3),
    })

    assert invalidated == "compass"
    assert guide.current_step == "compass"
    assert not guide.complete


def test_takeoff_button_path_refuses_until_human_preflight_is_complete() -> None:
    operator = OperatorApp.__new__(OperatorApp)
    operator.preflight_guide = SequentialPreflightGuide()
    operator.write_log = lambda message: operator.logs.append(message)
    operator.logs = []
    operator.sent = []
    operator.send = operator.sent.append
    operator.focus_set = lambda: None

    operator._send_flight_button("takeoff")

    assert operator.sent == []
    assert any("起飛前確認尚未完成" in message for message in operator.logs)

    for step in PREFLIGHT_GUIDE_STEPS:
        operator.preflight_guide.confirm_current((step, "confirmed"))
    operator._send_flight_button("takeoff")

    assert operator.sent == ["takeoff"]


def test_preflight_system_step_requires_fresh_stream_and_telemetry() -> None:
    now = time.monotonic()
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = SimpleNamespace(
        is_live=True,
        min_takeoff_battery_pct=15.0,
        desired_distance_geofence=True,
        require_gps_for_geofence=True,
        via_skycontroller=lambda: False,
        log=SimpleNamespace(durable=True, healthy=True),
    )
    operator.video_frame = object()
    operator.video_stream = SimpleNamespace(last_stamp=now)
    operator._video_frame_stamp = now
    state = DroneState(
        stream="PREVIEW",
        link_ok=True,
        link_status="OK",
        flight_state="landed",
        alert_state="none",
        battery_pct=80.0,
        gps_fixed=True,
        max_altitude_m=10.0,
        max_distance_m=50.0,
        distance_geofence_enabled=True,
        telemetry_read_mono_ns=int(now * 1_000_000_000),
    )

    evidence, reason = operator._preflight_step_evidence(
        "system", state, now=now
    )
    assert evidence is not None, reason

    state.telemetry_read_mono_ns = int((now - 3.0) * 1_000_000_000)
    evidence, reason = operator._preflight_step_evidence(
        "system", state, now=now
    )
    assert evidence is None
    assert "遙測" in reason


def test_preflight_route_step_requires_visible_hash_verified_route(tmp_path) -> None:
    route = tmp_path / "route.json"
    route.write_text("{}", encoding="utf-8")
    verified = []
    snapshot = SimpleNamespace(
        path=route,
        sha256="a" * 64,
        site_id="field-a",
        coordinate_frame_id="glomap-a",
        waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        verify_file_unchanged=lambda: verified.append(True),
    )
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = SimpleNamespace(is_live=False)
    operator.mission_route_lock = SimpleNamespace(snapshot=snapshot)
    operator._displayed_route_sha256 = snapshot.sha256
    operator.route_pts = list(snapshot.waypoints)
    operator.route_visible = True

    evidence, reason = operator._preflight_step_evidence(
        "route", DroneState(), verify_route_hash=True
    )

    assert evidence is not None, reason
    assert verified == [True]

    operator.route_visible = False
    evidence, reason = operator._preflight_step_evidence("route", DroneState())
    assert evidence is None
    assert "顯示規劃路徑" in reason


def test_start_auto_dispatches_the_selected_route_identity_and_locks_it() -> None:
    operator = _bare_app()
    operator.inspecting = True
    operator.mission_route_lock = _RouteLock()
    operator._displayed_route_sha256 = "a" * 64
    operator._reset_virtual_sticks = lambda: None

    operator.send("start_auto")

    assert operator.mission_route_lock.active
    assert operator.backend.calls == [("start_auto", {
        "route_path": "/tmp/selected-route.json",
        "route_sha256": "a" * 64,
        "site_id": "field-a",
        "coordinate_frame_id": "glomap-a",
    })]


def test_start_auto_does_not_dispatch_when_route_identity_mismatches() -> None:
    operator = _bare_app()
    operator.inspecting = True
    operator.mission_route_lock = _RouteLock(fail="displayed route mismatch")
    operator._displayed_route_sha256 = "b" * 64
    operator._reset_virtual_sticks = lambda: None

    operator.send("start_auto")

    assert operator.backend.calls == []


def test_route_editor_is_refused_while_auto_route_is_locked() -> None:
    operator = _bare_app()
    operator.backend.is_live = True
    operator.mission_route_lock = _RouteLock()
    operator.mission_route_lock.active = True

    allowed, reason = operator.route_editor_safety_check()

    assert not allowed
    assert "AUTO" in reason


def test_tick_deadline_does_not_accumulate_callback_runtime():
    deadline, delay = next_tick_deadline(1.0, 1.006, 0.010)
    assert deadline == pytest.approx(1.010)
    assert delay == 4

    deadline, delay = next_tick_deadline(deadline, 1.035, 0.010)
    assert deadline == pytest.approx(1.040)
    assert delay == 5


@pytest.mark.parametrize(
    ("reader", "value"),
    [
        (_positive_env_float, "nan"),
        (_positive_env_float, "0"),
        (_positive_env_int, "1.5"),
        (_positive_env_int, "-1"),
    ],
)
def test_safety_display_environment_rejects_invalid_values(
    monkeypatch, reader, value
) -> None:
    monkeypatch.setenv("SFM_TEST_VALUE", value)
    with pytest.raises(ValueError):
        reader("SFM_TEST_VALUE", 1)


def test_no_localization_markers_are_bounded() -> None:
    app = _bare_app()
    app.no_loc_markers = deque(maxlen=3)
    for index in range(5):
        app.live_last_xyz = np.array([float(index) * 2.0, 0.0, 0.0])
        app._record_no_loc()

    assert len(app.no_loc_markers) == 3
    assert np.allclose(app.no_loc_markers[0], [4.0, 0.0, 0.0])


def test_ui_log_is_trimmed_to_a_bounded_number_of_lines() -> None:
    class FakeLog:
        def __init__(self):
            self.insert_calls = []
            self.delete_calls = []

        def insert(self, *args):
            self.insert_calls.append(args)

        def delete(self, *args):
            self.delete_calls.append(args)

    app = OperatorApp.__new__(OperatorApp)
    app.log = FakeLog()

    app.write_log("bounded")

    assert app.log.insert_calls
    assert app.log.delete_calls == [(f"{UI_LOG_MAX_LINES + 1}.0", "end")]


def test_ui_starts_localization_without_exposing_fake_autonomy() -> None:
    source = inspect.getsource(OperatorApp._build_ui)
    assert 'text="開始定位"' in source
    assert "command=self.begin_auto_inspect" in source
    assert '"start_auto"' not in source
    assert 'text="自主巡檢未接入此介面"' in source


def test_operator_ui_cannot_force_megaloc_outside_boot_or_lost() -> None:
    source = inspect.getsource(OperatorApp._build_ui)

    assert '"BOOT/LOST", "global"' not in source
    assert "set_localization_benchmark_mode" not in source
    assert "定位測速：" not in source


def test_ui_has_textual_sim_real_identity_and_emergency_stop() -> None:
    """SIM/REAL must always be distinguishable by TEXT, not colour alone.

    2026-08-06, operator decision: the video HUD and the status bar below it were
    showing the same fields twice, so the shared ones were merged into the bar.
    Identity moved DOWN with them, which also restores the guarantee the 2026-08-03
    change had accepted as lost: it stays readable when there is no video frame.
    """
    build_source = inspect.getsource(OperatorApp._build_ui)
    assert "緊急停止電腦動作" in build_source

    # Identity belongs to the always-visible status bar, not the overlay.
    status_source = inspect.getsource(OperatorApp.tick)
    assert "video_hud_identity" in status_source, (
        "SIM/REAL identity must be composed into the status bar"
    )
    render_source = inspect.getsource(OperatorApp.render_video)
    assert "video_hud_identity" not in render_source, (
        "identity on the video would vanish exactly when the stream is lost"
    )

    simulated = video_hud_identity(False, "LIVE")
    real = video_hud_identity(True, "MANUAL")
    assert "SIMULATED" in simulated and "REAL" in real
    assert simulated != real


def test_ui_exposes_read_only_olympe_flight_telemetry() -> None:
    source = inspect.getsource(OperatorApp._build_ui)
    # The panel became rows of the single video overlay on 2026-08-06; what must
    # still hold is that the readouts exist and are composed read-only.
    assert "olympe_state_var" in source
    assert "olympe_altitude_var" in source
    assert "raw IMU／raw 氣壓數值" in source  # the fused-estimate disclosure

    state = DroneState(
        flight_state="hovering",
        alert_state="none",
        heading_state="ok",
        navigate_home_state="available",
        navigate_home_reason="enabled",
        drone_altitude_m=2.5,
        agl_altitude_m=1.8,
        speed_north_mps=0.18,
        speed_east_mps=0.24,
        speed_down_mps=-0.04,
        ground_speed_mps=0.30,
        gps_fixed=True,
        gps_latitude_deg=25.012345,
        gps_longitude_deg=121.543210,
        gps_altitude_m=12.5,
        gps_latitude_accuracy_m=1.0,
        gps_longitude_accuracy_m=2.0,
        gps_altitude_accuracy_m=3.0,
        gps_satellites=14,
        wind_state="warning",
        vibration_state="ok",
        hover_no_gps_too_high=True,
        wifi_rssi_dbm=-55,
        link_signal_quality_raw=0x84,
        sensor_states={"IMU": True, "barometer": True, "GPS": False},
        telemetry_read_mono_ns=1_000_000_000,
    )
    state.att_roll = 0.1
    state.att_pitch = -0.2
    state.att_yaw = 0.3

    telemetry = format_olympe_telemetry(
        state, now_mono_ns=1_025_000_000
    )

    assert "飛行 hovering" in telemetry["state"]
    assert "飛控融合姿態" in telemetry["attitude"]
    assert "N 0.18" in telemetry["speed"] and "水平 0.30" in telemetry["speed"]
    assert "相對起飛點" in telemetry["altitude"] and "AGL 1.80" in telemetry["altitude"]
    assert "GPS FIX" in telemetry["gps"] and "衛星 14" in telemetry["gps"]
    assert "外部干擾" in telemetry["environment"]
    assert "GPS:FAULT" in telemetry["sensors"]
    assert "cache age 25 ms" in telemetry["sensors"]


def test_ui_separates_firmware_magnetometer_calibration_from_passive_gravity_check() -> None:
    source = inspect.getsource(OperatorApp._build_ui)

    assert "飛機羅盤校正（只允許 landed；使用者手持旋轉）" in source
    assert "姿態／重力檢查（只讀；不寫入飛機）" in source
    assert "drone_magnetometer_start" in source
    assert "不會啟動馬達" in source
    # Operator decision 2026-08-06: the SkyController compass panel was removed.
    # It only feeds pilot-referenced features this system never uses, and it no
    # longer gates takeoff, so exposing it invited pointless field work.
    assert "skycontroller_magnetometer_start" not in source
    assert "每一軸請轉滿三圈" in source

    state = DroneState(
        drone_magnetometer_required=2,
        drone_magnetometer_started=True,
        drone_magnetometer_axis="yAxis",
        drone_magnetometer_x_done=True,
        drone_magnetometer_y_done=False,
        drone_magnetometer_z_done=False,
        drone_magnetometer_failed=False,
        skycontroller_magnetometer_state="CalibratingZ",
    )

    status = format_magnetometer_calibration(state)

    assert "建議" in status["drone"]
    assert "Y/pitch" in status["drone"]
    assert "進行中：Z 軸" in status["controller"]


def test_magnetometer_status_shows_reusable_and_latest_firmware_result() -> None:
    valid = format_magnetometer_calibration(DroneState(
        drone_magnetometer_required=0,
        drone_magnetometer_started=False,
        drone_magnetometer_x_done=True,
        drone_magnetometer_y_done=True,
        drone_magnetometer_z_done=True,
        drone_magnetometer_failed=False,
    ))["drone"]
    assert "最新讀回校正結果：PASS" in valid
    assert "X/Y/Z 已完成" in valid and "可沿用" in valid

    recommended = format_magnetometer_calibration(DroneState(
        drone_magnetometer_required=2,
        drone_magnetometer_started=False,
        drone_magnetometer_failed=False,
    ))["drone"]
    assert "機上既有校正結果：可沿用" in recommended
    assert "韌體建議重新校正" in recommended

    failed = format_magnetometer_calibration(DroneState(
        drone_magnetometer_required=1,
        drone_magnetometer_started=False,
        drone_magnetometer_failed=True,
    ))["drone"]
    assert "最新讀回校正結果：FAIL" in failed
    assert "不可沿用" in failed


def test_gravity_guide_tells_the_operator_each_motion_and_next_action() -> None:
    ready = gravity_phase_guidance(None)
    assert "拆除螺旋槳" in ready and "landed" in ready
    assert "水平旋轉 → 前後俯仰 → 左右側傾" in ready

    yaw = gravity_phase_guidance("yaw", sample_count=12, span_deg=46.4)
    assert "1/3 YAW" in yaw and "保持水平" in yaw and "轉一整圈" in yaw
    assert "樣本 12/15" in yaw and "角度變化 46°/90°" in yaw
    assert "下一階段" in yaw

    pitch = gravity_phase_guidance("pitch", sample_count=18, span_deg=31.0)
    assert "2/3 PITCH" in pitch and "機頭先抬高再壓低" in pitch
    assert "角度變化 31°/25°" in pitch and "下一階段" in pitch

    roll = gravity_phase_guidance("roll", sample_count=20, span_deg=28.0)
    assert "3/3 ROLL" in roll and "先向左再向右側傾" in roll
    assert "角度變化 28°/25°" in roll and "完成並分析" in roll


def test_focus_loss_clears_all_holds_and_backend_nudges():
    app = _bare_app()
    app._nudge_keys_held.add("w")
    app._nudge_buttons_held.add("上")

    app._on_input_focus_lost()

    assert not app._nudge_keys_held
    assert not app._nudge_buttons_held
    assert not app._stick_vector_active
    assert app.backend.calls == [("nudge_clear", {"reason": "ui_focus_lost"})]


@pytest.mark.parametrize("widget_class", ["Entry", "TEntry", "Text"])
def test_text_input_keypress_does_not_start_nudge(widget_class):
    app = _bare_app()
    event = SimpleNamespace(
        widget=SimpleNamespace(winfo_class=lambda: widget_class)
    )

    app._on_nudge_key_press("w", "前", event)

    assert not app._nudge_keys_held
    assert app.backend.calls == []


def test_non_input_keypress_still_starts_nudge():
    app = _bare_app()
    event = SimpleNamespace(
        widget=SimpleNamespace(winfo_class=lambda: "TFrame")
    )

    app._on_nudge_key_press("w", "前", event)

    assert app._nudge_keys_held == {"w"}
    assert app.backend.calls == [("nudge_begin", {"dir": "前"})]


def test_slow_backend_command_is_dispatched_without_blocking_caller():
    entered = threading.Event()
    release = threading.Event()

    class SlowBackend(_Backend):
        def command(self, name, **payload):
            entered.set()
            assert release.wait(1.0)
            return super().command(name, **payload)

    app = _bare_app(SlowBackend())
    start = time.monotonic()
    assert app._dispatch_live_command("takeoff", {})
    assert time.monotonic() - start < 0.1
    assert entered.wait(0.5)
    assert not app._dispatch_live_command("takeoff", {})

    release.set()
    command, result, error = app._flight_results.get(timeout=1.0)
    assert (command, result, error) == ("takeoff", "takeoff", None)


def test_flight_result_queue_drops_old_normal_results_but_preserves_safety():
    operator = _bare_app()
    operator._flight_results = queue.Queue(maxsize=2)
    operator._flight_safety_results = queue.Queue()
    incidents = []
    operator.session_logs = SimpleNamespace(
        incident=lambda event, **fields: incidents.append((event, fields))
    )

    operator._publish_flight_result(("hover", "old", None))
    operator._publish_flight_result(("manual", "middle", None))
    operator._publish_flight_result(("pc_control", "new", None))
    operator._publish_flight_result(("land", "landed", None))
    operator._publish_flight_result(("emergency_stop", "stopped", None))

    assert [operator._flight_results.get_nowait()[0] for _ in range(2)] == [
        "manual", "pc_control",
    ]
    assert [operator._flight_safety_results.get_nowait()[0] for _ in range(2)] == [
        "land", "emergency_stop",
    ]
    assert any(event == "flight_result_dropped" for event, _ in incidents)


def test_typed_sim_control_request_rejects_stale_normal_action_but_not_safety():
    backend = app.DroneBackend()
    stale_ns = time.monotonic_ns() - app.CONTROL_REQUEST_MAX_AGE_NS - 1

    stale_hover = backend.command(app.ControlRequest(
        action=app.ControlAction.HOVER,
        submitted_mono_ns=stale_ns,
        human_origin=True,
    ))
    assert not stale_hover.accepted
    assert stale_hover.reason_code == "STALE_CONTROL_REQUEST"

    stale_land = backend.command(app.ControlRequest(
        action=app.ControlAction.LAND_NOW,
        submitted_mono_ns=stale_ns,
        human_origin=True,
    ))
    assert stale_land.accepted
    assert stale_land.executed

    stale_emergency = backend.command(app.ControlRequest(
        action=app.ControlAction.EMERGENCY_STOP,
        submitted_mono_ns=stale_ns,
        human_origin=False,
    ))
    assert stale_emergency.accepted
    assert stale_emergency.executed


def test_space_hover_handler_consumes_event_and_flight_buttons_drop_focus():
    operator = _bare_app()
    calls = []
    operator._hover_all_nudges = lambda: calls.append("hover")

    assert operator._on_space_hover(None) == "break"
    assert calls == ["hover"]

    source = inspect.getsource(OperatorApp._build_ui)
    assert "takefocus=0" in source


def test_tick_poll_failure_fails_safe_and_schedules_next_tick():
    class PollFailureBackend:
        state = SimpleNamespace()

        def poll(self):
            raise RuntimeError("telemetry poll failed")

    incidents = []
    fail_safe_reasons = []
    scheduled = []
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = PollFailureBackend()
    operator.session_logs = SimpleNamespace(
        incident=lambda event, **fields: incidents.append((event, fields))
    )
    operator._drain_flight_command_results = lambda: None
    operator._active_nudge_directions = lambda: []
    operator._is_live_backend = lambda: False
    operator._stick_vector_active = False
    operator.write_log = lambda _message: None
    operator.after = lambda delay, callback: scheduled.append((delay, callback))
    operator._tick_period_s = 0.01
    operator._next_tick_deadline = 0.0
    operator.backend.fail_safe = lambda reason: fail_safe_reasons.append(reason)

    OperatorApp.tick(operator)

    assert fail_safe_reasons == [app.FailureReason.INVALID_TELEMETRY]
    assert incidents and incidents[0][0] == "backend_poll_failed"
    assert scheduled and getattr(scheduled[0][1], "__func__", None) is OperatorApp.tick


def test_diagnostic_write_failure_is_recorded_as_incident():
    incidents = []
    operator = OperatorApp.__new__(OperatorApp)
    operator.session_logs = SimpleNamespace(
        incident=lambda event, **fields: incidents.append((event, fields))
    )
    operator.backend = SimpleNamespace(log=SimpleNamespace(event=lambda *_a, **_k: None))
    operator.write_log = lambda _message: None

    OperatorApp._record_diagnostic_failure(
        operator, "/tmp/sfm_flight_operator_live_status.json", OSError("read-only")
    )

    assert incidents and incidents[0][0] == "diagnostic_write_failed"
    assert incidents[0][1]["resolved"] is False


def test_sim_takeoff_requires_typed_human_origin_request():
    backend = app.DroneBackend()

    legacy = backend.command("takeoff")
    assert isinstance(legacy, app.ControlResult)
    assert not legacy.accepted
    assert legacy.reason_code == "TYPED_TAKEOFF_REQUIRED"

    automated = backend.command(
        app.ControlRequest.create(app.ControlAction.TAKEOFF, human_origin=False)
    )
    assert not automated.accepted
    human = backend.command(
        app.ControlRequest.create(app.ControlAction.TAKEOFF, human_origin=True)
    )
    assert human.accepted
    assert backend.state.tracker_state == "TAKEOFF"


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_limit_editor_validates_then_sends_one_firmware_command():
    app = _bare_app()
    app.max_altitude_input_var = _Var("30")
    app.max_distance_input_var = _Var("100")
    app.distance_geofence_input_var = _Var(True)

    assert app.apply_firmware_limits_from_ui()
    assert app.backend.calls == [(
        "firmware_limits_apply",
        {
            "max_altitude_m": 30.0,
            "max_distance_m": 100.0,
            "distance_geofence": True,
        },
    )]


@pytest.mark.parametrize("altitude,distance", [("", "100"), ("nan", "100"), ("30", "0")])
def test_limit_editor_rejects_invalid_values(altitude, distance):
    app = _bare_app()
    app.max_altitude_input_var = _Var(altitude)
    app.max_distance_input_var = _Var(distance)
    app.distance_geofence_input_var = _Var(True)

    assert not app.apply_firmware_limits_from_ui()
    assert app.backend.calls == []


def test_limit_editor_can_load_current_firmware_readback_without_writing():
    backend = _Backend()
    backend.state = SimpleNamespace(
        max_altitude_m=18.0,
        max_distance_m=75.0,
        distance_geofence_enabled=False,
    )
    app = _bare_app(backend)
    app.max_altitude_input_var = _Var("")
    app.max_distance_input_var = _Var("")
    app.distance_geofence_input_var = _Var(True)

    assert app.load_firmware_limits_from_state()
    assert app.max_altitude_input_var.get() == "18"
    assert app.max_distance_input_var.get() == "75"
    assert app.distance_geofence_input_var.get() is False
    assert backend.calls == []


def test_autonomous_speed_editor_validates_then_sends_one_command():
    app = _bare_app()
    app.autonomous_speed_limit_input_var = _Var("0.30")

    assert app.apply_autonomous_speed_limit_from_ui()
    assert app.backend.calls == [(
        "auto_speed_limit_apply",
        {"speed_limit_mps": 0.3},
    )]


@pytest.mark.parametrize("value", ["", "nan", "0", "-0.1"])
def test_autonomous_speed_editor_rejects_invalid_values(value):
    app = _bare_app()
    app.autonomous_speed_limit_input_var = _Var(value)

    assert not app.apply_autonomous_speed_limit_from_ui()
    assert app.backend.calls == []


def test_real_flight_interface_refuses_a_replay_driven_hud(monkeypatch, tmp_path):
    """--live must never render the operator HUD from a recorded trajectory.

    state_from_replay() drives pose, map track, heading and localization quality.
    Feeding that from a JSON recording while a live, commandable ANAFI is connected
    shows a plausible trajectory unrelated to where the aircraft actually is.
    The refusal must happen before anything connects to the drone.
    """
    import sys

    import flight_operator_app as app

    replay = tmp_path / "replay.json"
    replay.write_text('{"rows": []}', encoding="utf-8")

    def _must_not_run(*_a, **_k):
        raise AssertionError("resolved site assets / connected before refusing")

    monkeypatch.setattr(app, "resolve_operator_site_assets", _must_not_run)
    monkeypatch.setattr(sys, "argv", [
        "flight_operator_app.py",
        "--live", "--ip", "192.168.53.1",
        "--no-live-localize",
        "--replay-json", str(replay),
    ])

    with pytest.raises(SystemExit) as excinfo:
        app.main()
    assert "--replay-json cannot be combined" in str(excinfo.value)


def test_operator_speed_envelope_defaults_match_backend():
    """The UI duplicates the envelope defaults because olympe_live_backend is
    imported lazily (it pulls Olympe). Keep the two copies in step."""
    import flight_operator_app as app
    import olympe_live_backend as backend

    assert app.DEFAULT_MAX_TILT_DEG == backend.DEFAULT_MAX_TILT_DEG
    assert app.DEFAULT_MAX_VERTICAL_SPEED_MS == backend.DEFAULT_MAX_VERTICAL_SPEED_MS
    assert (app.DEFAULT_MAX_ROTATION_SPEED_DEGS
            == backend.DEFAULT_MAX_ROTATION_SPEED_DEGS)


def test_magnetometer_axis_guide_maps_every_firmware_axis_form():
    """The firmware reports the axis in several shapes; all must map, and an
    absent/finished axis must map to None so a stale rotation is not shown."""
    from flight_operator_app import magnetometer_axis_guide as guide

    for raw in ("xAxis", "x_axis", "X", "MagnetoCalibrationAxis.xAxis"):
        assert guide(raw)[0] == "X/roll", raw
    for raw in ("yAxis", "y_axis", "Y"):
        assert guide(raw)[0] == "Y/pitch", raw
    for raw in ("zAxis", "z_axis", "Z"):
        assert guide(raw)[0] == "Z/yaw", raw
    for raw in ("none", "unknown", "", None):
        assert guide(raw) is None, raw


def test_magnetometer_axis_guide_views_are_distinct_and_instructive():
    from flight_operator_app import MAGNETOMETER_AXIS_GUIDE

    views = {view for _label, view, _text in MAGNETOMETER_AXIS_GUIDE.values()}
    assert views == {"top", "side", "front"}, "each axis needs its own drawing"
    for label, _view, text in MAGNETOMETER_AXIS_GUIDE.values():
        assert text.strip(), f"{label} has no operator instruction"


def test_takeoff_button_gate_mirrors_backend_compass_policy():
    """The UI must not grey out takeoff on a policy the backend no longer enforces.

    Backend (_magnetometer_control_error): an UNCALIBRATED controller compass is a
    warning; a calibration actively RUNNING still blocks. The UI's takeoff-button
    gate has to say the same thing, or the operator sees a dead button with no
    explanation anywhere.
    """
    source = inspect.getsource(OperatorApp.update_magnetometer_metrics)
    assert "controller_ready = not (via_controller and controller_active)" in source
    assert 'controller_key == "calibrated"' not in source


def test_exit_safety_signals_are_declared_and_armed_visibly():
    """'Close the terminal and it lands' must be verifiable, not assumed.

    The registration used to swallow failures silently, so the guarantee could be
    absent with nothing on screen or in the log to reveal it.
    """
    import flight_operator_app as app

    assert app.EXIT_SAFETY_SIGNALS == {"SIGINT", "SIGTERM", "SIGHUP"}
    source = inspect.getsource(app.main)
    # every one of them is armed
    assert "signal.SIGINT, signal.SIGTERM, signal.SIGHUP" in source
    # a failure is reported, not swallowed
    assert "exit safety armed" in source
    assert "may NOT land" in source
    assert '"exit_safety"' in source
    # the handler must run the landing cleanup
    assert "_emergency_cleanup(f\"signal_{signum}\")" in source


def test_emergency_cleanup_lands_and_survives_a_failing_backend():
    """The exit path must attempt the landing, and must not itself explode when
    the backend is already broken -- otherwise later exit steps never run."""
    import flight_operator_app as app

    source = inspect.getsource(app.main)
    start = source.index("def _emergency_cleanup")
    body = source[start:start + 700]
    assert "live_backend.cleanup()" in body, "exit path does not land"
    assert "except Exception" in body, "a broken backend would abort the exit path"
    assert "session_logs.close" in body, "exit path does not close the session log"


class _FakeStick:
    """Stands in for VirtualStick so the mapping is testable without Tk."""

    def __init__(self):
        self.value = (0.0, 0.0)
        self.recentred = 0

    def recenter(self):
        self.value = (0.0, 0.0)
        self.recentred += 1


def _stick_app(backend=None):
    app = _bare_app(backend)
    app.stick_left = _FakeStick()
    app.stick_right = _FakeStick()
    return app


def test_virtual_sticks_map_to_yaw_gaz_and_roll_pitch():
    """Mode 2 layout: left stick turns/climbs, right stick translates."""
    app = _stick_app()

    app.stick_left.value = (-0.5, 0.75)
    app._on_virtual_stick(app.stick_left, -0.5, 0.75)
    app.stick_right.value = (0.25, -1.0)
    app._on_virtual_stick(app.stick_right, 0.25, -1.0)

    assert app.backend.calls[-1] == (
        "nudge_vector",
        {"roll": 0.25, "pitch": -1.0, "yaw": -0.5, "gaz": 0.75},
    )
    assert app._stick_vector_active


def test_releasing_the_virtual_sticks_clears_only_the_stick_hold():
    """nudge_clear wipes the WHOLE hold set, including a key still being pressed.

    An all-zero nudge_vector releases only the stick's own sentinel, so a keyboard
    direction the operator is still physically holding keeps flying the aircraft.
    """
    app = _stick_app()
    app.stick_right.value = (0.5, 0.5)
    app._on_virtual_stick(app.stick_right, 0.5, 0.5)
    app.backend.calls.clear()

    app.stick_right.value = (0.0, 0.0)
    app._on_virtual_stick(app.stick_right, 0.0, 0.0)

    assert app.backend.calls == [
        ("nudge_vector", {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "gaz": 0.0})
    ], "a whole-set nudge_clear would drop a still-held key"
    assert not app._stick_vector_active

    # An already-centred stick must not spam the backend.
    app._on_virtual_stick(app.stick_right, 0.0, 0.0)
    assert len(app.backend.calls) == 1

def test_a_refused_vector_snaps_the_knob_home():
    """The knob must never show a deflection the aircraft is not following."""

    class _Refusing(_Backend):
        def command(self, name, **payload):
            super().command(name, **payload)
            return False

    app = _stick_app(_Refusing())
    app.stick_right.value = (0.9, 0.0)
    app._on_virtual_stick(app.stick_right, 0.9, 0.0)

    assert app.stick_right.recentred == 1
    assert app.stick_right.value == (0.0, 0.0)
    assert not app._stick_vector_active


def test_a_failing_vector_command_snaps_the_knob_home():
    class _Raising(_Backend):
        def command(self, name, **payload):
            raise RuntimeError("link down")

    app = _stick_app(_Raising())
    app.stick_left.value = (0.0, 1.0)
    app._on_virtual_stick(app.stick_left, 0.0, 1.0)

    assert app.stick_left.recentred == 1
    assert not app._stick_vector_active


def test_hover_button_recentres_both_virtual_sticks():
    app = _stick_app()
    app.send = lambda *a, **k: None
    app.stick_left.value = (0.5, 0.5)
    app.stick_right.value = (-0.5, 0.5)

    app._hover_all_nudges()

    assert app.stick_left.value == (0.0, 0.0)
    assert app.stick_right.value == (0.0, 0.0)
    assert not app._stick_vector_active


def test_a_held_virtual_stick_keeps_refreshing_the_backend_ttl():
    """Holding the knob still must re-send: the backend TTL decays to zero."""
    app = _stick_app()
    app.stick_right.value = (0.0, 0.6)
    app._on_virtual_stick(app.stick_right, 0.0, 0.6)
    app.backend.calls.clear()

    # What tick() does each frame while the knob is held but not moving.
    app._send_stick_vector(app.stick_left.value, app.stick_right.value)

    assert app.backend.calls == [
        ("nudge_vector", {"roll": 0.0, "pitch": 0.6, "yaw": 0.0, "gaz": 0.0})
    ]


def test_tick_drives_the_held_virtual_stick_heartbeat():
    """Without this wiring the knob would look held while PCMD decayed to zero."""
    source = inspect.getsource(OperatorApp.tick)
    assert "_stick_vector_active" in source
    assert "_send_stick_vector(self.stick_left.value, self.stick_right.value)" in source


def test_restart_closes_inherited_descriptors_before_exec() -> None:
    """execv inherits open fds, and Olympe routinely fails to release its pomp loop.

    The replacement image then held the SkyController socket the dying backend
    never closed, and could not connect -- observed as "Error while destroying
    pomp loop: -16" followed by a restart that died immediately.
    """
    source = inspect.getsource(app.main)
    exec_index = source.index("os.execv(")
    assert "os.closerange(3" in source[:exec_index], (
        "descriptors are still inherited by the restarted interface"
    )


def test_startup_asks_about_a_missing_route_for_the_active_site() -> None:
    """After a site switch the interface restarts, so the question must be re-asked."""
    source = inspect.getsource(app.OperatorApp._check_active_site_route)
    assert "follow_up_route_for_site" in source
    assert "route_json" in source

    scheduled = inspect.getsource(app.OperatorApp.__init__)
    assert "_check_active_site_route" in scheduled, "nothing schedules the check"

from __future__ import annotations

import flight_operator_app as app
import inspect
import operator_tick
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
    VIRTUAL_STICK_KEY_MAP,
    _positive_env_float,
    _positive_env_int,
    format_magnetometer_calibration,
    format_olympe_telemetry,
    gps_operator_message,
    gravity_phase_guidance,
    inventory_ui_status,
    next_tick_deadline,
    video_hud_identity,
)
from operator_shutdown import OperatorShutdownCoordinator


class _Backend:
    def __init__(self):
        self.calls = []

    def command(self, name, **payload):
        self.calls.append((name, payload))
        return name


def _bare_app(backend=None):
    app = OperatorApp.__new__(OperatorApp)
    app.backend = backend or _Backend()
    app.preflight_guide = SequentialPreflightGuide()
    for step in PREFLIGHT_GUIDE_STEPS:
        app.preflight_guide.confirm_current((step, "test fixture"))
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


class _CollisionBackend:
    is_live = False

    def __init__(self):
        self.state = SimpleNamespace(
            pose=np.zeros(4, dtype=float),
            tracker_state="CRUISE",
            last_command="",
        )
        self.calls = []

    def nudge_clear(self, *, reason):
        self.calls.append(("nudge_clear", reason))

    def send_pcmd(self, *pcmd, reason):
        self.calls.append(("send_pcmd", pcmd, reason))
        return True


def _collision_guard_app() -> OperatorApp:
    operator = _bare_app(_CollisionBackend())
    operator.map_radius = 10.0
    operator.default_map_center = np.zeros(3, dtype=float)
    operator.collision_guard_enabled = True
    operator.collision_guard_radius = None
    operator._collision_interlock_latched = False
    operator._collision_guard_monitor = None
    operator.collision_guard_snapshot = {}
    operator.inspecting = False
    operator.current_state = operator.backend.state
    operator._is_live_backend = lambda: False
    operator._pause_integrated_auto = lambda _reason: False
    operator.session_logs = None
    operator._configure_sparse_cloud_collision_guard(
        np.array([[0.0, 0.0, 0.0]], dtype=float)
    )
    if operator._collision_guard_monitor.tree is None:
        pytest.skip("scipy cKDTree is unavailable")
    return operator


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

    def cancel_rejected_auto_start(self):
        self.active = False


def _complete_preflight(operator) -> None:
    operator.preflight_guide = SequentialPreflightGuide()
    for step in PREFLIGHT_GUIDE_STEPS:
        operator.preflight_guide.confirm_current((step, "confirmed"))


def _make_autonomy_runtime_ready(operator, monkeypatch=None) -> None:
    operator._autonomy_profile_verified = True
    operator.localizer = SimpleNamespace(
        ready=True, request_relocalize=lambda: None
    )
    operator.loc_health = "OK"
    operator.loc_health_inliers = 100
    operator.loc_health_reproj = 0.1
    operator._loc_consecutive_good_fixes = 2
    operator._zoom_localization_paused = False
    operator.live_locked = True
    operator.live_result = None
    operator.live_pose = np.zeros(4, dtype=float)
    operator.camera_forward_world = np.array([1.0, 0.0, 0.0])
    operator.loc_pose_updated_mono = time.monotonic()
    operator._autonomy_map_frame = app.LEGACY_MAP_FRAME
    if monkeypatch is not None:
        monkeypatch.setattr(app, "validate_profile_flight_assets", lambda _profile: None)
        monkeypatch.setattr(
            app, "resolve_site_map_frame", lambda _profile: app.LEGACY_MAP_FRAME
        )


def test_draw_detections_renders_a_label_box_without_crashing() -> None:
    """Regression: the label background rectangle used to reference the
    undefined names ty1/tw (NameError on any non-empty detection result)."""
    from PIL import Image, ImageDraw

    operator = OperatorApp.__new__(OperatorApp)
    operator.detection_result = {
        "success": True,
        "boxes": [
            {"xyxy": [10, 10, 100, 100], "class_id": 0, "class_name": "person", "confidence": 0.87},
        ],
    }
    img = Image.new("RGB", (200, 200))
    draw = ImageDraw.Draw(img)

    operator.draw_detections(draw, 1.0, 0, 0, 200, 200)


def test_keyboard_keys_match_the_two_virtual_sticks() -> None:
    assert VIRTUAL_STICK_KEY_MAP == {
        "a": "左旋",
        "d": "右旋",
        "w": "上",
        "s": "下",
        "j": "左",
        "l": "右",
        "i": "前",
        "k": "後",
    }


def test_sparse_cloud_center_camera_is_preview_only() -> None:
    operator = _collision_guard_app()

    operator.update_sparse_cloud_collision_interlock(operator.backend.state)

    assert operator.collision_guard_snapshot["status"] == "PREVIEW_HIT"
    assert operator.collision_guard_snapshot["preview"] is True
    assert operator.collision_guard_snapshot["center"] == pytest.approx(
        operator.default_map_center
    )
    assert operator._collision_interlock_latched is False
    assert operator.backend.calls == []


def test_sparse_cloud_hit_clears_motion_hovers_once_and_requires_clearance() -> None:
    operator = _collision_guard_app()
    operator.inspecting = True
    operator.live_locked = True
    operator.loc_health = "OK"
    operator.live_pose = np.array([0.02, 0.0, 0.0], dtype=float)
    operator.loc_pose_updated_mono = time.monotonic()
    pauses = []
    incidents = []
    operator._pause_integrated_auto = lambda reason: pauses.append(reason) or True
    operator.session_logs = SimpleNamespace(
        incident=lambda event, **fields: incidents.append((event, fields))
    )

    operator.update_sparse_cloud_collision_interlock(operator.backend.state)
    operator.update_sparse_cloud_collision_interlock(operator.backend.state)

    assert operator.collision_guard_snapshot["status"] == "COLLISION"
    assert operator._collision_interlock_latched is True
    assert pauses == ["sparse_cloud_collision"]
    assert operator.backend.calls == [
        ("nudge_clear", "sparse_cloud_collision"),
        (
            "send_pcmd",
            (0, 0, 0, 0),
            "sparse_cloud_collision_hover",
        ),
    ]
    assert operator.backend.state.tracker_state == "HOVER"
    assert operator._collision_motion_blocked("test motion") is True
    assert incidents[0][0] == "sparse_cloud_collision_hover"
    assert incidents[0][1]["resolved"] is False
    calls_at_latch = list(operator.backend.calls)
    operator._on_nudge_btn_press("前")
    assert operator.backend.calls == calls_at_latch
    assert operator._nudge_buttons_held == set()

    operator.inspecting = False
    operator.update_sparse_cloud_collision_interlock(operator.backend.state)
    assert operator.collision_guard_snapshot["status"] == "PREVIEW_HIT"
    assert operator._collision_interlock_latched is True

    operator.inspecting = True
    operator.live_pose = np.array([2.0, 0.0, 0.0], dtype=float)
    operator.loc_pose_updated_mono = time.monotonic()
    operator.update_sparse_cloud_collision_interlock(operator.backend.state)
    assert operator.collision_guard_snapshot["status"] == "CLEAR"
    assert operator._collision_interlock_latched is False
    assert incidents[-1][1]["resolved"] is True


def test_sparse_cloud_inspect_without_lock_does_not_latch() -> None:
    operator = _collision_guard_app()
    operator.inspecting = True
    operator.live_locked = False
    operator.loc_health = "FAIL"
    operator.backend.state.pose = np.array([0.0, 0.0, 0.0, 0.0])

    operator.update_sparse_cloud_collision_interlock(operator.backend.state)

    assert operator.collision_guard_snapshot["status"] == "WAITING"
    assert operator._collision_interlock_latched is False
    assert operator.backend.calls == []


def test_sparse_cloud_radius_change_controls_the_collision_boundary() -> None:
    operator = _collision_guard_app()
    operator.inspecting = True
    operator.live_locked = True
    operator.loc_health = "OK"
    operator.live_pose = np.array([0.04, 0.0, 0.0], dtype=float)
    operator.loc_pose_updated_mono = time.monotonic()
    operator.collision_guard_radius = 0.025

    operator.update_sparse_cloud_collision_interlock(operator.backend.state)
    assert operator.collision_guard_snapshot["status"] == "CLEAR"

    operator.collision_guard_radius = 0.05
    operator.update_sparse_cloud_collision_interlock(operator.backend.state)
    assert operator.collision_guard_snapshot["status"] == "COLLISION"


def test_sparse_cloud_collision_still_hovers_when_auto_pause_raises() -> None:
    operator = _collision_guard_app()
    operator.inspecting = True
    operator.live_locked = True
    operator.loc_health = "OK"
    operator.live_pose = np.array([0.01, 0.0, 0.0], dtype=float)
    operator.loc_pose_updated_mono = time.monotonic()

    def fail_pause(_reason):
        raise RuntimeError("coordinator failed")

    operator._pause_integrated_auto = fail_pause
    operator.update_sparse_cloud_collision_interlock(operator.backend.state)

    assert operator._collision_interlock_latched is True
    assert operator.backend.calls[-1] == (
        "send_pcmd",
        (0, 0, 0, 0),
        "sparse_cloud_collision_hover",
    )


def test_keyboard_indicator_moves_without_emitting_a_stick_command() -> None:
    stick = app.VirtualStick.__new__(app.VirtualStick)
    stick._x = 0.0
    stick._y = 0.0
    stick._draw = lambda: None
    stick._on_change = lambda *_args: pytest.fail(
        "keyboard visualization must not emit nudge_vector"
    )

    stick.show_keyboard(-1.0, 1.0)

    assert stick.value == (0.0, 0.0)
    assert (stick._keyboard_x, stick._keyboard_y) == (-1.0, 1.0)


def test_keyboard_press_and_release_update_the_matching_stick_visual() -> None:
    class Stick:
        def __init__(self):
            self.keyboard = None

        def show_keyboard(self, x, y):
            self.keyboard = (x, y)

    operator = _bare_app()
    operator.stick_left = Stick()
    operator.stick_right = Stick()
    event = SimpleNamespace(
        widget=SimpleNamespace(winfo_class=lambda: "TFrame")
    )

    operator._on_nudge_key_press("w", "上", event)
    operator._on_nudge_key_press("a", "左旋", event)

    assert operator.stick_left.keyboard == (-1.0, 1.0)
    assert operator.stick_right.keyboard == (0.0, 0.0)

    operator._on_nudge_key_release("w", "上", event)
    operator._on_nudge_key_release("a", "左旋", event)

    assert operator.stick_left.keyboard == (0.0, 0.0)
    assert operator.backend.calls == [
        ("nudge_begin", {"dir": "上"}),
        ("nudge_begin", {"dir": "左旋"}),
        ("nudge_end", {"dir": "上"}),
        ("nudge_end", {"dir": "左旋"}),
    ]


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
    operator.inspecting = False
    operator.live_locked = False
    operator.loc_health = "FAIL"
    operator._send_flight_button("takeoff")

    assert operator.sent == ["takeoff"]


def test_completing_four_step_preflight_enables_auto_button() -> None:
    class Var:
        def __init__(self):
            self.value = ""

        def set(self, value):
            self.value = value

    class Widget:
        def __init__(self):
            self.states = []

        def state(self, values):
            self.states.append(tuple(values))

    operator = OperatorApp.__new__(OperatorApp)
    operator.preflight_guide = SequentialPreflightGuide()
    operator.preflight_guide_var = Var()
    operator.preflight_confirm_button = Widget()
    operator.takeoff_button = Widget()
    operator.start_auto_button = Widget()
    operator._preflight_auto_collapsed = False
    operator._is_live_backend = lambda: False
    operator._preflight_step_evidence = lambda step, *_args, **_kwargs: (
        (step, "confirmed"),
        "ready",
    )
    operator._update_preflight_cards = lambda _state: None
    operator._update_flight_action_guidance = lambda _state: None
    operator._set_preflight_visible = lambda _visible: None
    operator._select_flight_tab = lambda: None

    for step in PREFLIGHT_GUIDE_STEPS[:-1]:
        operator.preflight_guide.confirm_current((step, "confirmed"))
    state = SimpleNamespace(flight_state="landed")
    operator._update_preflight_guide(state)
    assert operator.start_auto_button.states[-1] == ("disabled",)

    final_step = PREFLIGHT_GUIDE_STEPS[-1]
    operator.preflight_guide.confirm_current((final_step, "confirmed"))
    operator._update_preflight_guide(state)

    assert operator.start_auto_button.states[-1] == ("!disabled",)
    assert "「自動飛行」" in operator.preflight_guide_var.value


def test_valid_firmware_compass_readback_is_auto_confirmed() -> None:
    class Var:
        def set(self, _value):
            pass

    class Widget:
        def state(self, _values):
            pass

    operator = OperatorApp.__new__(OperatorApp)
    operator.preflight_guide = SequentialPreflightGuide()
    operator.preflight_guide_var = Var()
    operator.preflight_confirm_button = Widget()
    operator.takeoff_button = Widget()
    operator.start_auto_button = Widget()
    operator._preflight_auto_collapsed = False
    operator._is_live_backend = lambda: True
    operator._preflight_step_evidence = lambda step, *_args, **_kwargs: (
        (("compass", 0), "校正有效")
        if step == "compass"
        else (None, "等待人工確認")
    )
    operator._update_preflight_cards = lambda _state: None
    operator._update_flight_action_guidance = lambda _state: None
    operator._select_preflight_tab = lambda step: operator.tabs.append(step)
    operator._set_preflight_visible = lambda _visible: None
    operator._set_preflight_expanded = lambda _expanded: None
    operator.logs = []
    operator.tabs = []
    operator.write_log = operator.logs.append
    state = DroneState(
        flight_state="landed",
        drone_magnetometer_required=0,
        drone_magnetometer_started=False,
        drone_magnetometer_failed=False,
    )

    operator._update_preflight_guide(state)

    assert operator.preflight_guide.confirmed_steps == ("compass",)
    assert operator.preflight_guide.current_step == "map"
    assert operator.tabs == ["map"]
    assert any("韌體有效回讀自動確認" in message for message in operator.logs)


@pytest.mark.parametrize(
    ("required", "started", "failed"),
    [
        (2, False, False),
        (0, True, False),
        (0, False, True),
        (None, False, False),
    ],
    ids=["recommended", "in-progress", "failed", "unknown"],
)
def test_invalid_compass_state_is_not_auto_confirmed(
    required: int | None,
    started: bool,
    failed: bool,
) -> None:
    operator = OperatorApp.__new__(OperatorApp)
    operator.preflight_guide = SequentialPreflightGuide()
    operator._preflight_step_evidence = lambda *_args, **_kwargs: (
        ("compass", required),
        "羅盤狀態尚未有效",
    )
    operator.write_log = lambda _message: None
    operator._select_preflight_tab = lambda _step: None
    state = DroneState(
        flight_state="landed",
        drone_magnetometer_required=required,
        drone_magnetometer_started=started,
        drone_magnetometer_failed=failed,
    )

    assert operator._auto_confirm_valid_compass_preflight(state, live=True) is False
    assert operator.preflight_guide.current_step == "compass"


@pytest.mark.parametrize(
    "stream_state",
    ["PREVIEW", "OK", "LOCALIZING", "HOLD_720P"],
)
def test_preflight_system_step_requires_fresh_stream_and_telemetry(
    stream_state: str,
) -> None:
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
        stream=stream_state,
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
    assert evidence == ("system", "live"), reason

    for advisory_battery in (1.0, float("nan")):
        state.battery_pct = advisory_battery
        evidence, reason = operator._preflight_step_evidence(
            "system", state, now=now
        )
        assert evidence is not None, reason
        assert "電量" in reason and "不阻擋起飛" in reason
    state.battery_pct = 80.0

    state.gps_fixed = False
    state.max_altitude_m = None
    state.max_distance_m = None
    state.distance_geofence_enabled = None
    operator.backend._firmware_config_ok = False
    evidence, reason = operator._preflight_step_evidence(
        "system", state, now=now
    )
    assert evidence is not None, reason
    assert "不阻擋起飛" in reason

    for unavailable_stream in ("WAIT", "LOST", "LOST_HOLD"):
        state.stream = unavailable_stream
        evidence, reason = operator._preflight_step_evidence(
            "system", state, now=now
        )
        assert evidence is None
        assert "串流狀態" in reason
    state.stream = stream_state

    state.telemetry_read_mono_ns = int((now - 3.0) * 1_000_000_000)
    evidence, reason = operator._preflight_step_evidence(
        "system", state, now=now
    )
    assert evidence is None
    assert "遙測" in reason


def test_preflight_system_step_does_not_block_on_airborne_state_or_alerts() -> None:
    now = time.monotonic()
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = SimpleNamespace(
        is_live=True,
        min_takeoff_battery_pct=15.0,
        via_skycontroller=lambda: False,
        log=SimpleNamespace(durable=True, healthy=True),
    )
    operator.video_frame = object()
    operator.video_stream = SimpleNamespace(last_stamp=now)
    operator._video_frame_stamp = now
    state = DroneState(
        stream="PREVIEW",
        link_ok=True,
        flight_state="hovering",
        active_incident="operator_acknowledged",
        alert_state="critical",
        battery_pct=80.0,
        telemetry_read_mono_ns=int(now * 1_000_000_000),
    )

    evidence, reason = operator._preflight_step_evidence(
        "system", state, now=now
    )

    assert evidence is not None, reason


@pytest.mark.parametrize("gps_fixed", [False, None], ids=["no-fix", "unknown"])
def test_missing_gps_with_disabled_geofence_is_not_a_preflight_blocker(gps_fixed) -> None:
    now = time.monotonic()
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = SimpleNamespace(
        is_live=True,
        min_takeoff_battery_pct=15.0,
        desired_distance_geofence=False,
        require_gps_for_geofence=True,
        via_skycontroller=lambda: False,
        log=SimpleNamespace(durable=True, healthy=True),
    )
    operator.video_frame = object()
    operator.video_stream = SimpleNamespace(last_stamp=now)
    state = DroneState(
        stream="PREVIEW",
        link_ok=True,
        flight_state="landed",
        alert_state="none",
        battery_pct=80.0,
        gps_fixed=gps_fixed,
        distance_geofence_enabled=False,
        telemetry_read_mono_ns=int(now * 1_000_000_000),
    )

    evidence, reason = operator._preflight_step_evidence(
        "system", state, now=now
    )

    assert evidence is not None, reason
    assert "不阻擋起飛" in reason


@pytest.mark.parametrize("gps_fixed", [False, None], ids=["no-fix", "unknown"])
def test_manual_takeoff_does_not_require_gps_fix(gps_fixed) -> None:
    operator = OperatorApp.__new__(OperatorApp)
    _complete_preflight(operator)
    operator.backend = SimpleNamespace(
        state=DroneState(gps_fixed=gps_fixed, distance_geofence_enabled=False),
        desired_distance_geofence=False,
        require_gps_for_geofence=True,
    )
    operator.sent = []
    operator.send = operator.sent.append
    operator.write_log = lambda _message: None
    operator.focus_set = lambda: None

    operator._send_flight_button("takeoff")

    assert operator.sent == ["takeoff"]


def test_preflight_route_step_requires_hash_verified_route(tmp_path) -> None:
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

    evidence, reason = operator._preflight_step_evidence(
        "route", DroneState(), verify_route_hash=True
    )

    assert evidence is not None, reason
    assert verified == [True]


def test_display_only_import_can_be_confirmed_without_becoming_auto_candidate(
    tmp_path, monkeypatch
) -> None:
    route = tmp_path / "route.json"
    route.write_text("{}", encoding="utf-8")
    snapshot = SimpleNamespace(
        path=route.resolve(),
        sha256="a" * 64,
        site_id="river_site_edm",
        coordinate_frame_id="river_site_glomap",
        waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        controller_waypoints=lambda: [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ],
        verify_file_unchanged=lambda: None,
    )
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = SimpleNamespace(is_live=False)
    operator.mission_route_lock = SimpleNamespace(active=False, snapshot=None)
    operator.site_profile_path = tmp_path / "profile.json"
    operator._active_site_map_frame = lambda: None
    operator.redraw_map_only = lambda: None
    monkeypatch.setattr(
        app,
        "load_site_profile",
        lambda _path: SimpleNamespace(
            site_id="river_site_edm",
            flight=SimpleNamespace(coordinate_frame_id="river_site_glomap"),
        ),
    )
    monkeypatch.setattr(app, "flight_readiness_errors", lambda _profile: ["not approved"])
    monkeypatch.setattr(app, "file_sha256", lambda _path: snapshot.sha256)
    monkeypatch.setattr(
        app, "capture_mission_route_snapshot", lambda *_args, **_kwargs: snapshot
    )

    operator._show_route_overlay(route, bind_for_auto=False)
    evidence, reason = operator._preflight_step_evidence(
        "route", DroneState(), verify_route_hash=True
    )

    assert evidence is not None, reason
    assert operator.mission_route_lock.snapshot is None


def test_approved_editor_import_refreshes_and_binds_the_active_auto_route(
    tmp_path,
) -> None:
    profile_path = tmp_path / "site_profile.json"
    route_path = tmp_path / "flight_route.json"
    calls = []
    operator = OperatorApp.__new__(OperatorApp)
    operator.site_profile_path = profile_path
    operator._sync_autonomy_profile_approval = lambda: calls.append("sync")
    operator._show_route_overlay = lambda path, *, bind_for_auto: calls.append(
        (Path(path), bind_for_auto)
    )
    operator.write_log = lambda message: calls.append(("log", message))

    operator.apply_route_import_preview(
        SimpleNamespace(
            asset_path=route_path,
            profile_path=profile_path,
            approved_for_auto=True,
        )
    )

    assert calls[:2] == ["sync", (route_path, True)]


def test_approved_editor_import_switches_to_the_updated_managed_profile(
    tmp_path, monkeypatch
) -> None:
    active_profile = tmp_path / "system-profile.json"
    managed_profile = tmp_path / "managed" / "site_profile.json"
    route_path = managed_profile.parent / "routes" / "flight_route.json"
    calls = []
    scheduled = []
    operator = OperatorApp.__new__(OperatorApp)
    operator.site_profile_path = active_profile
    operator._show_route_overlay = lambda path, *, bind_for_auto: calls.append(
        ("show", Path(path), bind_for_auto)
    )
    operator.request_site_profile_restart = lambda path: calls.append(
        ("apply", Path(path))
    )
    operator.after_idle = scheduled.append
    operator.write_log = lambda message: calls.append(("log", message))
    operator.site_assets_panel = SimpleNamespace(
        set_status=lambda message: calls.append(("status", message))
    )
    monkeypatch.setattr(
        app,
        "site_pack_root_for_profile",
        lambda _profile, _root: tmp_path / "managed",
    )

    operator.apply_route_import_preview(
        SimpleNamespace(
            asset_path=route_path,
            profile_path=managed_profile,
            approved_for_auto=True,
        )
    )

    assert calls[0] == ("show", route_path, False)
    assert len(scheduled) == 1
    scheduled[0]()
    assert ("apply", managed_profile) in calls


def test_confirming_route_step_binds_the_displayed_snapshot_for_auto() -> None:
    old_snapshot = SimpleNamespace(
        path=Path("/tmp/old-route.json"),
        sha256="a" * 64,
    )
    selected_snapshot = SimpleNamespace(
        path=Path("/tmp/selected-route.json"),
        sha256="b" * 64,
        site_id="river_site_edm",
        coordinate_frame_id="river_site_glomap",
        waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    events = []

    class RouteLock:
        active = False
        snapshot = old_snapshot

        def bind(self, snapshot):
            events.append(("bind", snapshot))
            self.snapshot = snapshot

    class SessionLogs:
        def command(self, event, **fields):
            events.append((event, fields))
            return True

    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = SimpleNamespace(state=DroneState())
    operator.current_state = operator.backend.state
    operator.preflight_guide = SequentialPreflightGuide()
    operator.preflight_guide.confirm_current(("compass", "confirmed"))
    operator.preflight_guide.confirm_current(("map", "confirmed"))
    operator.mission_route_lock = RouteLock()
    operator._displayed_route_snapshot = selected_snapshot
    operator._displayed_route_sha256 = selected_snapshot.sha256
    operator.session_logs = SessionLogs()
    operator._preflight_step_evidence = lambda *_args, **_kwargs: (
        ("route", selected_snapshot.sha256),
        "route verified",
    )
    operator._update_preflight_guide = lambda _state: None
    operator._select_preflight_tab = lambda _step: None
    operator.write_log = lambda _message: None

    assert operator.confirm_current_preflight_step() is True

    assert operator.mission_route_lock.snapshot is selected_snapshot
    assert events[0][0] == "route_selected"
    assert events[0][1]["route_sha256"] == selected_snapshot.sha256
    assert events[1] == ("bind", selected_snapshot)
    assert operator.preflight_guide.current_step == "system"


def test_route_selection_is_logged_before_the_snapshot_is_bound(
    tmp_path, monkeypatch
) -> None:
    route = tmp_path / "route.json"
    route.write_text("{}", encoding="utf-8")
    snapshot = SimpleNamespace(
        path=route.resolve(),
        sha256="a" * 64,
        site_id="river_site_edm",
        coordinate_frame_id="river_site_glomap",
        waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        controller_waypoints=lambda: [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ],
    )
    events = []

    class RouteLock:
        active = False
        snapshot = None

        def bind(self, value):
            events.append(("bind", value))
            self.snapshot = value

    class SessionLogs:
        def command(self, event, **fields):
            events.append((event, fields))
            return True

    operator = OperatorApp.__new__(OperatorApp)
    operator.mission_route_lock = RouteLock()
    operator.session_logs = SessionLogs()
    operator.site_profile_path = tmp_path / "profile.json"
    operator._active_site_map_frame = lambda: None
    operator.redraw_map_only = lambda: None
    monkeypatch.setattr(
        app,
        "load_site_profile",
        lambda _path: SimpleNamespace(
            site_id="river_site_edm",
            flight=SimpleNamespace(coordinate_frame_id="river_site_glomap"),
        ),
    )
    monkeypatch.setattr(app, "flight_readiness_errors", lambda _profile: [])
    monkeypatch.setattr(app, "file_sha256", lambda _path: snapshot.sha256)
    monkeypatch.setattr(
        app, "capture_mission_route_snapshot", lambda *_args, **_kwargs: snapshot
    )

    count = operator._show_route_overlay(route)

    assert count == 2
    assert events[0] == (
        "route_selected",
        {
            "route_path": str(route.resolve()),
            "route_sha256": "a" * 64,
            "site_id": "river_site_edm",
            "coordinate_frame_id": "river_site_glomap",
            "waypoint_count": 2,
        },
    )
    assert events[1] == ("bind", snapshot)


def test_route_selection_is_not_bound_when_the_audit_log_fails(
    tmp_path, monkeypatch
) -> None:
    route = tmp_path / "route.json"
    route.write_text("{}", encoding="utf-8")
    snapshot = SimpleNamespace(
        path=route.resolve(),
        sha256="a" * 64,
        site_id="river_site_edm",
        coordinate_frame_id="river_site_glomap",
        waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    bound = []
    operator = OperatorApp.__new__(OperatorApp)
    operator.mission_route_lock = SimpleNamespace(
        active=False, bind=bound.append
    )
    operator.session_logs = SimpleNamespace(command=lambda *_args, **_kwargs: False)
    operator.site_profile_path = tmp_path / "profile.json"
    operator._active_site_map_frame = lambda: None
    monkeypatch.setattr(
        app,
        "load_site_profile",
        lambda _path: SimpleNamespace(
            site_id="river_site_edm",
            flight=SimpleNamespace(coordinate_frame_id="river_site_glomap"),
        ),
    )
    monkeypatch.setattr(app, "flight_readiness_errors", lambda _profile: [])
    monkeypatch.setattr(app, "file_sha256", lambda _path: snapshot.sha256)
    monkeypatch.setattr(
        app, "capture_mission_route_snapshot", lambda *_args, **_kwargs: snapshot
    )

    with pytest.raises(ValueError, match="耐久記錄"):
        operator._show_route_overlay(route)

    assert bound == []


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


def test_live_start_auto_uses_integrated_takeoff_sequence_not_legacy_backend() -> None:
    operator = _bare_app()
    operator.backend.is_live = True
    operator.inspecting = False
    operator.mission_route_lock = _RouteLock()
    operator._displayed_route_sha256 = "a" * 64
    operator._begin_inspection_feed = lambda: setattr(operator, "inspecting", True)
    started = []
    operator._start_integrated_auto = lambda snapshot: started.append(snapshot) or True

    operator.send("start_auto")

    assert operator.inspecting
    assert started == [operator.mission_route_lock.snapshot]
    assert operator.backend.calls == []


def test_hover_pauses_integrated_auto_and_auto_button_resumes_without_takeoff() -> None:
    class Coordinator:
        phase = "ROUTE"
        paused = False

        def __init__(self):
            self.pause_calls = []
            self.resume_calls = 0

        def pause(self, reason):
            self.pause_calls.append(reason)
            self.paused = True
            return True

        def resume(self):
            self.resume_calls += 1
            self.paused = False
            return True

    coordinator = Coordinator()
    operator = _bare_app()
    operator._integrated_autonomy = coordinator
    operator._reset_virtual_sticks = lambda: None

    operator.send("hover")

    assert coordinator.pause_calls == ["operator_hover"]
    assert operator._auto_paused
    assert operator.backend.calls == [("hover", {})]

    operator.send("start_auto")

    assert coordinator.resume_calls == 1
    assert not operator._auto_paused
    assert operator.backend.calls == [("hover", {})]


def test_start_auto_does_not_dispatch_when_route_identity_mismatches() -> None:
    operator = _bare_app()
    operator.inspecting = True
    operator.mission_route_lock = _RouteLock(fail="displayed route mismatch")
    operator._displayed_route_sha256 = "b" * 64
    operator._reset_virtual_sticks = lambda: None

    operator.send("start_auto")

    assert operator.backend.calls == []


@pytest.mark.parametrize(
    ("profile_approved", "route_clearance_approved"),
    [(False, True), (True, False)],
    ids=["profile-not-approved", "route-not-cleared"],
)
def test_unapproved_profile_or_route_is_not_bound_as_auto_candidate(
    tmp_path, monkeypatch, profile_approved, route_clearance_approved
) -> None:
    route = tmp_path / "route.json"
    route.write_text("{}", encoding="utf-8")
    snapshot = SimpleNamespace(
        path=route.resolve(),
        sha256="a" * 64,
        site_id="field-a",
        coordinate_frame_id="glomap-a",
        waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        controller_waypoints=lambda: [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ],
    )
    bound = []

    class RouteLock:
        active = False
        snapshot = None

        def bind(self, value):
            bound.append(value)
            self.snapshot = value

    profile = SimpleNamespace(
        site_id="field-a",
        flight=SimpleNamespace(
            approved=profile_approved,
            route_clearance_approved=route_clearance_approved,
            coordinate_frame_id="glomap-a",
        ),
    )
    operator = OperatorApp.__new__(OperatorApp)
    operator.mission_route_lock = RouteLock()
    operator.site_profile_path = tmp_path / "profile.json"
    operator._active_site_map_frame = lambda: None
    operator.redraw_map_only = lambda: None
    operator.write_log = lambda _message: None
    monkeypatch.setattr(app, "load_site_profile", lambda _path: profile)
    monkeypatch.setattr(app, "file_sha256", lambda _path: snapshot.sha256)
    monkeypatch.setattr(
        app, "capture_mission_route_snapshot", lambda *_args, **_kwargs: snapshot
    )
    monkeypatch.setattr(
        app,
        "flight_readiness_errors",
        lambda selected: (
            []
            if selected.flight.approved and selected.flight.route_clearance_approved
            else ["AUTO approval is incomplete"]
        ),
        raising=False,
    )

    with pytest.raises(ValueError):
        operator._show_route_overlay(route)

    assert bound == []
    assert operator.mission_route_lock.snapshot is None


def test_unapproved_route_choice_switches_display_without_binding_auto(
    tmp_path, monkeypatch
) -> None:
    route = tmp_path / "alternate-route.json"
    route.write_text("{}", encoding="utf-8")
    snapshot = SimpleNamespace(
        path=route,
        sha256="b" * 64,
        waypoints=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    calls = []
    operator = OperatorApp.__new__(OperatorApp)
    operator.site_profile_path = tmp_path / "profile.json"
    operator.mission_route_lock = SimpleNamespace(snapshot=None)
    operator.write_log = lambda _message: None

    def show(path, *, bind_for_auto=True):
        calls.append((Path(path), bind_for_auto))
        operator._displayed_route_snapshot = snapshot
        return len(snapshot.waypoints)

    operator._show_route_overlay = show
    monkeypatch.setattr(app, "load_site_profile", lambda _path: object())
    monkeypatch.setattr(
        app,
        "flight_readiness_errors",
        lambda _profile: ["AUTO approval is incomplete"],
    )

    message = operator.preview_site_route(route)

    assert calls == [(route, False)]
    assert operator.mission_route_lock.snapshot is None
    assert "僅切換顯示" in message


def test_start_auto_does_not_dispatch_when_four_step_preflight_is_incomplete() -> None:
    operator = _bare_app()
    operator.inspecting = True
    operator.preflight_guide = SequentialPreflightGuide()
    for step in PREFLIGHT_GUIDE_STEPS[:-1]:
        operator.preflight_guide.confirm_current((step, "confirmed"))
    operator.mission_route_lock = _RouteLock()
    operator._displayed_route_sha256 = "a" * 64
    operator._reset_virtual_sticks = lambda: None
    operator._pending_auto_route_activation = False
    operator.logs = []
    operator.write_log = operator.logs.append

    operator.send("start_auto")

    assert operator.backend.calls == []
    assert not operator.mission_route_lock.active
    assert not getattr(operator, "_pending_auto_route_activation", False)
    assert operator.preflight_guide.current_step == PREFLIGHT_GUIDE_STEPS[-1]


@pytest.mark.parametrize(
    ("autonomous_locked", "autonomous_approval_valid"),
    [(True, True), (False, False)],
    ids=["runtime-lock", "invalid-approval"],
)
def test_integrated_auto_does_not_create_coordinator_when_runtime_approval_is_invalid(
    tmp_path,
    monkeypatch,
    autonomous_locked,
    autonomous_approval_valid,
) -> None:
    created = []

    class Coordinator:
        phase = "IDLE"
        boot_timeout_s = 1.0

        def __init__(self, **_kwargs):
            created.append(self)

        def start(self):
            return True

    profile = SimpleNamespace(
        site_id="field-a",
        flight=SimpleNamespace(
            approved=True,
            route_clearance_approved=True,
            coordinate_frame_id="glomap-a",
        ),
    )
    operator = _bare_app()
    operator.backend.is_live = True
    operator.backend.state = SimpleNamespace(
        autonomous_locked=autonomous_locked,
        autonomous_approval_valid=autonomous_approval_valid,
    )
    operator.inspecting = True
    _make_autonomy_runtime_ready(operator, monkeypatch)
    _complete_preflight(operator)
    operator.mission_route_lock = _RouteLock()
    operator._displayed_route_sha256 = "a" * 64
    operator.site_profile_path = tmp_path / "profile.json"
    operator._active_site_map_frame = lambda: object()
    operator._reset_virtual_sticks = lambda: None
    monkeypatch.setattr(app, "DesktopRouteAutonomy", Coordinator)
    monkeypatch.setattr(app, "load_site_profile", lambda _path: profile)
    monkeypatch.setattr(
        app, "flight_readiness_errors", lambda _profile: [], raising=False
    )

    operator.send("start_auto")

    assert created == []
    assert operator.__dict__.get("_integrated_autonomy") is None
    assert not operator.mission_route_lock.active
    assert operator.backend.calls == []


def test_integrated_auto_starts_without_a_separate_live_release_flag(
    tmp_path, monkeypatch
) -> None:
    created = []

    class Coordinator:
        phase = "IDLE"
        boot_timeout_s = 1.0

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def start(self):
            return True

    profile = SimpleNamespace(
        site_id="field-a",
        pose_chain=SimpleNamespace(site_frame_id="field-a-enu"),
        flight=SimpleNamespace(
            approved=True,
            route_clearance_approved=True,
            coordinate_frame_id="glomap-a",
            controller=SimpleNamespace(max_route_deviation_map_units=0.9),
        ),
    )
    operator = _bare_app()
    operator.backend.is_live = True
    operator.backend.state = SimpleNamespace(
        autonomous_locked=False,
        autonomous_approval_valid=True,
        flight_state="landed",
    )
    operator.inspecting = True
    _make_autonomy_runtime_ready(operator, monkeypatch)
    _complete_preflight(operator)
    operator.mission_route_lock = _RouteLock()
    operator._displayed_route_sha256 = "a" * 64
    operator.site_profile_path = tmp_path / "profile.json"
    operator._active_site_map_frame = lambda: object()
    operator._reset_virtual_sticks = lambda: None
    operator.logs = []
    operator.write_log = operator.logs.append
    monkeypatch.setattr(app, "DesktopRouteAutonomy", Coordinator)
    monkeypatch.setattr(app, "load_site_profile", lambda _path: profile)
    monkeypatch.setattr(
        app, "flight_readiness_errors", lambda _profile: [], raising=False
    )

    operator.send("start_auto")

    assert len(created) == 1
    assert operator.__dict__.get("_integrated_autonomy") is created[0]
    assert created[0].kwargs["max_route_deviation_map_units"] == pytest.approx(0.9)
    assert operator.mission_route_lock.active
    assert not any("發布阻擋" in message for message in operator.logs)


def test_integrated_auto_while_airborne_selects_pc_handoff(tmp_path, monkeypatch) -> None:
    created = []

    class Coordinator:
        phase = "IDLE"
        boot_timeout_s = 1.0

        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            return True

    profile = SimpleNamespace(
        site_id="field-a",
        pose_chain=SimpleNamespace(site_frame_id="field-a-enu"),
        flight=SimpleNamespace(
            approved=True,
            route_clearance_approved=True,
            coordinate_frame_id="glomap-a",
        ),
    )
    operator = _bare_app()
    operator.backend.is_live = True
    operator.backend.state = SimpleNamespace(
        autonomous_locked=False,
        autonomous_approval_valid=True,
        flight_state="hovering",
    )
    operator.backend.pilot_sticks = True
    operator.inspecting = True
    _make_autonomy_runtime_ready(operator, monkeypatch)
    _complete_preflight(operator)
    operator.mission_route_lock = _RouteLock()
    operator._displayed_route_sha256 = "a" * 64
    operator.site_profile_path = tmp_path / "profile.json"
    operator._active_site_map_frame = lambda: object()
    operator._reset_virtual_sticks = lambda: None
    monkeypatch.setattr(app, "DesktopRouteAutonomy", Coordinator)
    monkeypatch.setattr(app, "load_site_profile", lambda _path: profile)
    monkeypatch.setattr(
        app, "flight_readiness_errors", lambda _profile: [], raising=False
    )

    operator.send("start_auto")

    assert len(created) == 1
    assert created[0]["start_airborne"] is True
    assert callable(created[0]["take_pc_control"])


@pytest.mark.parametrize("command", ["takeoff", "pc_control"])
def test_integrated_auto_start_event_requires_literal_true(command) -> None:
    operator = OperatorApp.__new__(OperatorApp)
    operator._pending_auto_route_activation = True
    operator.cancelled_activation = False
    operator.confirmed_activation = False
    operator._cancel_pending_auto_route_activation = (
        lambda: setattr(operator, "cancelled_activation", True)
    )
    operator.mission_route_lock = SimpleNamespace(
        confirm_auto_started=lambda: setattr(operator, "confirmed_activation", True)
    )
    operator._finish_backend_command = lambda *_args: None

    OperatorApp._finish_integrated_auto_command_event(
        operator,
        SimpleNamespace(command=command, result=None, error=None),
    )

    assert not operator.confirmed_activation
    assert operator.cancelled_activation


def test_integrated_auto_failed_event_is_visible_and_keeps_coordinator_for_shutdown(
) -> None:
    logs = []
    incidents = []
    coordinator = SimpleNamespace(phase="AUTO_FAILED")
    operator = OperatorApp.__new__(OperatorApp)
    operator._integrated_autonomy = coordinator
    operator._pending_auto_route_activation = False
    operator.session_logs = SimpleNamespace(
        incident=lambda event, **fields: incidents.append((event, fields))
    )
    operator.write_log = logs.append
    operator._set_auto_paused = lambda _paused: None

    OperatorApp._log_integrated_auto_failed(
        operator,
        SimpleNamespace(detail="AUTO worker stalled", error="heartbeat timeout"),
    )

    assert operator._integrated_autonomy is coordinator
    assert logs == [
        "AUTO 失敗並已鎖定懸停：AUTO worker stalled；"
        "請使用「停止電腦動作」或「原地降落」，確認安全後重新啟動介面"
    ]
    assert incidents == [
        (
            "auto_failed",
            {
                "detail": "AUTO worker stalled",
                "error": "heartbeat timeout",
                "resolved": False,
            },
        )
    ]
def test_integrated_auto_pc_handoff_event_activates_route() -> None:
    operator = OperatorApp.__new__(OperatorApp)
    operator._pending_auto_route_activation = True
    operator.cancelled_activation = False
    operator.confirmed_activation = False
    operator._cancel_pending_auto_route_activation = (
        lambda: setattr(operator, "cancelled_activation", True)
    )
    operator.mission_route_lock = SimpleNamespace(
        confirm_auto_started=lambda: setattr(operator, "confirmed_activation", True)
    )
    operator._finish_backend_command = lambda *_args: None

    OperatorApp._finish_integrated_auto_command_event(
        operator,
        SimpleNamespace(command="pc_control", result=True, error=None),
    )

    assert operator.confirmed_activation
    assert not operator.cancelled_activation
    assert operator._pending_auto_route_activation is False


def test_integrated_auto_waiting_for_localization_logs_continued_hover() -> None:
    operator = OperatorApp.__new__(OperatorApp)
    messages = []
    operator.write_log = messages.append

    OperatorApp._log_integrated_auto_boot_hover_waiting(
        operator,
        SimpleNamespace(detail="stable localization unavailable after 25.0s"),
    )

    assert len(messages) == 1
    assert "持續原地懸停" in messages[0]
    assert "不會自動降落" in messages[0]


def test_integrated_auto_pose_jump_enables_explicit_continue_button() -> None:
    operator = OperatorApp.__new__(OperatorApp)
    messages = []
    paused = []
    operator.write_log = messages.append
    operator._set_auto_paused = paused.append

    OperatorApp._log_integrated_auto_pose_jump_paused(
        operator,
        SimpleNamespace(detail="定位位置單次跳變 2.75 map units"),
    )

    assert paused == [True]
    assert "鎖定懸停" in messages[0]
    assert "繼續自動飛行" in messages[0]


def test_profile_verification_sets_only_the_runtime_approval_state() -> None:
    operator = _bare_app()
    operator.backend.is_live = True
    operator.backend.session_config = SimpleNamespace(autonomous_locked=False)
    operator.backend.state = SimpleNamespace()
    operator._active_site_autonomy_errors = lambda: ()

    operator._sync_autonomy_profile_approval()

    assert operator._autonomy_profile_verified is True
    assert operator.backend.state.autonomous_locked is False
    assert operator.backend.state.autonomous_approval_valid is True
    assert not hasattr(operator.backend.state, "live_auto_release_ready")


def test_runtime_unlock_cannot_bypass_site_profile_readiness() -> None:
    operator = _bare_app()
    operator.backend.is_live = True
    operator.backend.session_config = SimpleNamespace(autonomous_locked=False)
    operator.backend.state = SimpleNamespace()
    operator._active_site_autonomy_errors = lambda: ("signed receipt missing",)

    operator._sync_autonomy_profile_approval()

    assert operator._autonomy_profile_verified is False
    assert operator.backend.state.autonomous_locked is True
    assert operator.backend.state.autonomous_approval_valid is False
    assert not hasattr(operator.backend.state, "live_auto_release_ready")


def test_runtime_lock_remains_an_additional_site_profile_blocker() -> None:
    operator = _bare_app()
    operator.backend.is_live = True
    operator.backend.session_config = SimpleNamespace(autonomous_locked=True)
    operator.backend.state = SimpleNamespace()
    operator._active_site_autonomy_errors = lambda: ()

    operator._sync_autonomy_profile_approval()

    assert operator._autonomy_profile_verified is False
    assert operator.backend.state.autonomous_locked is True
    assert operator.backend.state.autonomous_approval_valid is False
    assert "runtime configuration" in operator._autonomy_profile_errors[0]
    assert not hasattr(operator.backend.state, "live_auto_release_ready")


@pytest.mark.parametrize("gps_fixed", [False, None], ids=["no-fix", "unknown"])
def test_integrated_auto_takeoff_allows_missing_gps_when_geofence_is_off(
    tmp_path, monkeypatch, gps_fixed
) -> None:
    created = []

    class Coordinator:
        phase = "IDLE"
        boot_timeout_s = 1.0

        def __init__(self, **_kwargs):
            created.append(self)

        def start(self):
            return True

    profile = SimpleNamespace(
        site_id="field-a",
        pose_chain=SimpleNamespace(site_frame_id="field-a-enu"),
        flight=SimpleNamespace(
            approved=True,
            route_clearance_approved=True,
            coordinate_frame_id="glomap-a",
        ),
    )
    operator = _bare_app()
    operator.backend.is_live = True
    operator.backend.desired_distance_geofence = False
    operator.backend.require_gps_for_geofence = True
    operator.backend.state = SimpleNamespace(
        autonomous_locked=False,
        autonomous_approval_valid=True,
        gps_fixed=gps_fixed,
        distance_geofence_enabled=False,
        flight_state="landed",
        link_ok=True,
    )
    operator.inspecting = True
    _make_autonomy_runtime_ready(operator, monkeypatch)
    _complete_preflight(operator)
    operator.mission_route_lock = _RouteLock()
    operator._displayed_route_sha256 = "a" * 64
    operator.site_profile_path = tmp_path / "profile.json"
    operator._active_site_map_frame = lambda: object()
    operator._reset_virtual_sticks = lambda: None
    monkeypatch.setattr(app, "DesktopRouteAutonomy", Coordinator)
    monkeypatch.setattr(app, "load_site_profile", lambda _path: profile)
    monkeypatch.setattr(
        app, "flight_readiness_errors", lambda _profile: [], raising=False
    )

    operator.send("start_auto")

    assert len(created) == 1
    assert operator.__dict__.get("_integrated_autonomy") is created[0]
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


def test_write_log_remains_available_without_a_ui_log_widget(capsys) -> None:
    operator = OperatorApp.__new__(OperatorApp)

    operator.write_log("terminal-only")

    assert "terminal-only" in capsys.readouterr().out
    assert "log" not in operator.__dict__


def test_ui_exposes_integrated_autonomy_and_separate_localization() -> None:
    source = inspect.getsource(OperatorApp._build_ui)
    assert 'text="開始定位"' in source
    assert "command=self.toggle_localization" in source
    assert any(
        button.command == "emergency_stop" and button.label == "停止電腦動作"
        for button in app.FLIGHT_MODE_BUTTONS
    )
    assert any(
        button.command == "start_auto" and button.label == "自動飛行"
        for button in app.MISSION_MODE_BUTTONS
    )
    assert 'text="自主巡檢未接入此介面"' not in source


def test_flight_tab_exposes_truthful_autonomous_speed_guard() -> None:
    source = inspect.getsource(OperatorApp._build_ui)

    assert "autonomous_speed_limit_input_var" in source
    assert "套用速度限制" in source
    assert "新鮮遙測達上限即送零 PCMD" in source
    assert "地速缺失或過期時禁止水平 AUTO" in source
    assert "非硬上限" in source


def test_operator_ui_cannot_force_megaloc_outside_boot_or_lost() -> None:
    source = inspect.getsource(OperatorApp._build_ui)

    assert '"BOOT/LOST", "global"' not in source
    assert "set_localization_benchmark_mode" not in source
    assert "定位測速：" not in source


def test_ui_has_textual_sim_real_identity_without_separate_auto_stop() -> None:
    """SIM/REAL must always be distinguishable by TEXT, not colour alone.

    2026-08-06, operator decision: the video HUD and the status bar below it were
    showing the same fields twice, so the shared ones were merged into the bar.
    Identity moved DOWN with them, which also restores the guarantee the 2026-08-03
    change had accepted as lost: it stays readable when there is no video frame.
    """
    build_source = inspect.getsource(OperatorApp._build_ui)
    assert "停止自動並懸停" not in build_source

    # Identity belongs to the always-visible status header, not the video.
    status_source = inspect.getsource(OperatorApp._update_flight_header)
    assert "REAL ANAFI" in status_source and "SIMULATED" in status_source
    render_source = inspect.getsource(OperatorApp.render_video)
    assert "video_hud_identity" not in render_source, (
        "identity on the video would vanish exactly when the stream is lost"
    )

    simulated = video_hud_identity(False, "LIVE")
    real = video_hud_identity(True, "MANUAL")
    assert "SIMULATED" in simulated and "REAL" in real
    assert simulated != real


def test_auto_localization_timeout_guidance_keeps_zero_pcmd_hover_policy() -> None:
    class Var:
        def __init__(self):
            self.value = ""

        def set(self, value):
            self.value = value

    operator = OperatorApp.__new__(OperatorApp)
    operator.preflight_guide = SimpleNamespace(
        complete=False,
        current_step="system",
    )
    operator._preflight_step_evidence = lambda *_args: (None, "等待新鮮定位")
    operator._auto_paused = False
    operator.flight_action_hint_var = Var()
    operator.flight_action_hint = SimpleNamespace(configure=lambda **_kwargs: None)

    OperatorApp._update_flight_action_guidance(
        operator,
        SimpleNamespace(flight_state="landed", gps_fixed=False),
    )

    assert "零 PCMD" in operator.flight_action_hint_var.value
    assert "定位恢復" in operator.flight_action_hint_var.value
    assert "人工接管" in operator.flight_action_hint_var.value
    assert "降落" in operator.flight_action_hint_var.value
    assert "不會自動降落" in operator.flight_action_hint_var.value
    assert "逾時原地降落" not in operator.flight_action_hint_var.value


def test_stop_computer_action_dispatches_existing_emergency_stop_command() -> None:
    operator = _bare_app()

    operator.send("emergency_stop")

    assert operator.backend.calls == [("emergency_stop", {})]


def test_gps_header_is_advisory_and_invalid_values_are_hidden() -> None:
    state = DroneState(
        gps_fixed=False,
        gps_altitude_m=500.0,
        gps_latitude_deg=500.0,
        gps_longitude_deg=500.0,
    )
    message, severity = gps_operator_message(state, live=True)
    assert severity == "warning"
    assert "手動可起飛" in message
    assert "自動先懸停" in message

    telemetry = format_olympe_telemetry(state)
    assert "GPS — m" in telemetry["altitude"]
    assert "lat —, lon —" in telemetry["gps"]
    assert "500" not in telemetry["altitude"] + telemetry["gps"]


def test_inventory_status_separates_hardware_version_and_lost_link() -> None:
    backend = SimpleNamespace(
        is_live=True,
        connection_inventory={
            "aircraft": {"name": "ANAFI", "software": "1.8.2"},
            "controller": {"variant": None, "software": "1.8.1"},
            "runtime": {"olympe_version": "8.4.0"},
            "lost_link": {
                "fallback": "land_in_place",
                "policy_confirmed": True,
            },
            "block_reasons": [
                "connected controller is not confirmed SkyController 3 "
                "(variant='unavailable', hid='unavailable')",
                "controller firmware '1.8.1' lacks an approved receipt",
            ],
        },
    )
    statuses = inventory_ui_status(backend)
    assert statuses["hardware"][0] == "blocked"
    assert "SkyController 3" in statuses["hardware"][1]
    assert statuses["version"][0] == "blocked"
    assert "控制器韌體" in statuses["version"][1]
    assert statuses["lost_link"] == ("good", "已確認：無 Home 時原地降落")


def test_ui_exposes_read_only_olympe_flight_telemetry() -> None:
    source = inspect.getsource(OperatorApp._build_ui)
    # The panel became rows of the single video overlay on 2026-08-06; what must
    # still hold is that the readouts exist and are composed read-only.
    assert "olympe_state_var" in source
    assert "olympe_altitude_var" in source

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
    assert telemetry["rth"] == "RTH available/enabled"
    assert "飛控融合姿態" in telemetry["attitude"]
    assert "N 0.18" in telemetry["speed"] and "水平 0.30" in telemetry["speed"]
    assert "三軸速度" in telemetry["velocity"] and "水平" not in telemetry["velocity"]
    assert "相對起飛點" in telemetry["altitude"] and "AGL 1.80" in telemetry["altitude"]
    assert "GPS" not in telemetry["altitude_agl"]
    assert "GPS FIX" in telemetry["gps"] and "衛星 14" in telemetry["gps"]
    assert telemetry["link_quality"] == "連接品質 4/5 (外部干擾)"
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


def test_focus_change_without_active_input_sends_no_flight_command():
    app = _bare_app()

    app._on_input_focus_lost()

    assert app.backend.calls == []


@pytest.mark.parametrize(
    "widget_class",
    [
        "Entry", "TEntry", "Text", "Button", "TButton", "Scale", "TScale",
        "TNotebook", "TCombobox", "Spinbox", "TSpinbox", "Listbox",
        "Checkbutton", "TCheckbutton", "Radiobutton", "TRadiobutton",
    ],
)
def test_interactive_widget_keypress_does_not_start_nudge(widget_class):
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


def test_live_emergency_stop_is_dispatched_without_blocking_caller():
    entered = threading.Event()
    release = threading.Event()

    class SlowSafetyBackend(_Backend):
        is_live = True

        def command(self, name, **payload):
            entered.set()
            assert release.wait(1.0)
            return super().command(name, **payload)

    operator = _bare_app(SlowSafetyBackend())
    started = time.monotonic()
    operator.send("emergency_stop")

    assert time.monotonic() - started < 0.1
    assert entered.wait(0.5)
    release.set()
    command, result, error = operator._flight_safety_results.get(timeout=1.0)
    assert (command, result, error) == ("emergency_stop", "emergency_stop", None)


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


def test_space_hover_handler_consumes_event_and_flight_buttons_are_focusable():
    operator = _bare_app()
    calls = []
    operator._hover_all_nudges = lambda: calls.append("hover")

    assert operator._on_space_hover(None) == "break"
    assert calls == ["hover"]

    source = inspect.getsource(OperatorApp._build_ui)
    assert "takefocus=True" in source
    assert 'widget.bind("<space>", self._on_space_hover)' in source
    assert 'widget.bind("<Return>", self._on_flight_button_return)' in source

    invoked = []
    event = SimpleNamespace(widget=SimpleNamespace(invoke=lambda: invoked.append(True)))
    assert operator._on_flight_button_return(event) == "break"
    assert invoked == [True]


def test_tick_poll_failure_hovers_without_handoff_and_schedules_next_tick():
    class PollFailureBackend:
        state = SimpleNamespace()

        def poll(self):
            raise RuntimeError("telemetry poll failed")

    incidents = []
    hover_commands = []
    cleared = []
    pauses = []
    scheduled = []
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = PollFailureBackend()
    operator.localizer = None
    operator.session_logs = SimpleNamespace(
        incident=lambda event, **fields: incidents.append((event, fields))
    )
    operator._drain_flight_command_results = lambda: None
    operator._active_nudge_directions = lambda: []
    operator._is_live_backend = lambda: False
    operator._stick_vector_active = False
    operator.write_log = lambda _message: None
    operator._pause_integrated_auto = lambda reason: pauses.append(reason)
    operator.after = lambda delay, callback: scheduled.append((delay, callback))
    operator._tick_period_s = 0.01
    operator._next_tick_deadline = 0.0
    operator.backend.nudge_clear = lambda *, reason: cleared.append(reason)
    operator.backend.send_pcmd = (
        lambda *pcmd, reason: hover_commands.append((pcmd, reason)) or True
    )

    OperatorApp.tick(operator)

    assert cleared == ["backend_poll_failed"]
    assert hover_commands == [((0, 0, 0, 0), "backend_poll_failed_hover")]
    assert pauses == ["backend_poll_failed"]
    assert incidents and incidents[0][0] == "backend_poll_failed"
    assert scheduled and getattr(scheduled[0][1], "__func__", None) is OperatorApp.tick


def test_airborne_manual_state_enables_restarting_auto_cruise() -> None:
    class Widget:
        def __init__(self):
            self.states = []

        def state(self, values):
            self.states.append(tuple(values))

    operator = OperatorApp.__new__(OperatorApp)
    _complete_preflight(operator)
    operator._is_live_backend = lambda: True
    operator._integrated_auto_active = lambda: False
    operator._auto_paused = False
    operator.preflight_guide_var = _Var("")
    operator.preflight_confirm_button = Widget()
    operator.takeoff_button = Widget()
    operator.start_auto_button = Widget()
    operator._update_preflight_cards = lambda _state: None
    operator._update_flight_action_guidance = lambda _state: None

    operator._update_preflight_guide(
        SimpleNamespace(flight_state="hovering")
    )

    assert operator.takeoff_button.states[-1] == ("disabled",)
    assert operator.start_auto_button.states[-1] == ("!disabled",)


def test_airborne_manual_state_cannot_start_auto_with_final_preflight_step_pending(
) -> None:
    class Widget:
        def __init__(self):
            self.states = []

        def state(self, values):
            self.states.append(tuple(values))

    operator = OperatorApp.__new__(OperatorApp)
    operator.preflight_guide = SequentialPreflightGuide()
    for step in PREFLIGHT_GUIDE_STEPS[:-1]:
        operator.preflight_guide.confirm_current((step, "confirmed"))
    operator._is_live_backend = lambda: True
    operator._integrated_auto_active = lambda: False
    operator._auto_paused = False
    operator.preflight_guide_var = _Var("")
    operator.preflight_confirm_button = Widget()
    operator.takeoff_button = Widget()
    operator.start_auto_button = Widget()
    operator._preflight_step_evidence = lambda *_args, **_kwargs: (
        ("system", "live"),
        "ready",
    )
    operator._update_preflight_cards = lambda _state: None
    operator._update_flight_action_guidance = lambda _state: None
    operator.current_state = SimpleNamespace(flight_state="hovering")
    operator.backend = SimpleNamespace(state=operator.current_state)

    operator._update_preflight_guide(operator.current_state)

    assert operator.preflight_guide.current_step == "system"
    assert operator.preflight_confirm_button.states[-1] == ("!disabled",)
    assert operator.takeoff_button.states[-1] == ("disabled",)
    assert operator.start_auto_button.states[-1] == ("disabled",)
    assert operator._preflight_blocks_flight_command("start_auto")


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
    app.autonomous_speed_limit_enabled_var = _Var(False)

    assert app.apply_autonomous_speed_limit_from_ui()
    assert app.backend.calls == [(
        "auto_speed_limit_apply",
        {"speed_limit_mps": 0.3, "enabled": False},
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


def test_exit_safety_signals_are_declared_and_armed_visibly(monkeypatch, capsys):
    """'Close the terminal and it lands' must be verifiable, not assumed.

    The registration used to swallow failures silently, so the guarantee could be
    absent with nothing on screen or in the log to reveal it.
    """
    import flight_operator_app as app

    assert app.EXIT_SAFETY_SIGNALS == {"SIGINT", "SIGTERM", "SIGHUP"}
    handlers = {}
    atexit_callbacks = []
    log_events = []
    cleanup_reasons = []
    after_calls = []
    destroyed = []
    fake_app = SimpleNamespace(
        backend=SimpleNamespace(
            log=SimpleNamespace(
                event=lambda name, **fields: log_events.append((name, fields))
            )
        ),
        after=lambda delay, callback: after_calls.append((delay, callback)),
        destroy=lambda: destroyed.append(True),
    )
    monkeypatch.setattr(
        app.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler)
    )
    monkeypatch.setattr(
        app.atexit, "register", lambda callback: atexit_callbacks.append(callback)
    )

    def cleanup(reason):
        cleanup_reasons.append(reason)
        return True

    app._install_live_exit_safety(fake_app, cleanup)

    assert {sig.name for sig in handlers} == app.EXIT_SAFETY_SIGNALS
    assert len(atexit_callbacks) == 1
    assert "exit safety armed" in capsys.readouterr().out
    assert log_events[0][0] == "exit_safety"
    assert log_events[0][1]["ok"] is True
    with pytest.raises(SystemExit):
        handlers[app.signal.SIGTERM](app.signal.SIGTERM, None)
    assert cleanup_reasons == [f"signal_{app.signal.SIGTERM}"]
    assert after_calls == []
    assert destroyed == [True]


def test_exit_signal_keeps_ui_open_until_landing_is_confirmed_then_retries(
    monkeypatch,
) -> None:
    import flight_operator_app as app

    handlers = {}
    destroyed = []
    cleanup_results = iter([False, True])
    fake_app = SimpleNamespace(
        backend=SimpleNamespace(
            log=SimpleNamespace(event=lambda *_args, **_kwargs: None)
        ),
        destroy=lambda: destroyed.append(True),
    )
    monkeypatch.setattr(
        app.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler)
    )
    monkeypatch.setattr(app.atexit, "register", lambda _callback: None)

    app._install_live_exit_safety(
        fake_app,
        lambda _reason: next(cleanup_results),
    )

    handlers[app.signal.SIGINT](app.signal.SIGINT, None)
    assert destroyed == []

    with pytest.raises(SystemExit):
        handlers[app.signal.SIGINT](app.signal.SIGINT, None)
    assert destroyed == [True]


def test_main_arms_ctrl_c_shutdown_for_simulation_and_live_modes() -> None:
    import flight_operator_app as app

    source = inspect.getsource(app.main)

    assert "_install_live_exit_safety(app, _emergency_cleanup)" in source
    assert "if live_backend is not None:" not in source


def test_emergency_cleanup_lands_and_survives_a_failing_backend():
    """The exit path must attempt the landing, and must not itself explode when
    the backend is already broken -- otherwise later exit steps never run."""
    import flight_operator_app as app

    source = inspect.getsource(app.main)
    start = source.index("def _emergency_cleanup")
    body = source[start:start + 2500]
    assert "coordinator.shutdown(reason=reason)" in body, "exit path does not land"
    assert 'app.__dict__.get("backend")' in body, "exit path can use a stale backend"
    assert "except Exception" in body, "a broken backend would abort the exit path"
    assert "return result is True" in body, "exit path must fail closed"


@pytest.mark.parametrize(
    ("command", "false_success"),
    (("takeoff", "起飛 expectation 完成"), ("land", "降落 expectation 完成")),
)
def test_false_takeoff_or_land_completion_is_not_reported_as_success(
    command: str,
    false_success: str,
) -> None:
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = SimpleNamespace(
        pilot_sticks=False,
        state=SimpleNamespace(tracker_state="HOVER", flight_state="hovering"),
    )
    messages = []
    operator.write_log = messages.append

    OperatorApp._finish_backend_command(operator, command, False, None)

    assert not any(false_success in message for message in messages)
    assert any("未確認" in message for message in messages)


@pytest.mark.parametrize("result", [False, None, 1, "ok"])
def test_non_true_auto_completion_does_not_confirm_route(result) -> None:
    operator = OperatorApp.__new__(OperatorApp)
    confirmed = []
    operator._pending_auto_route_activation = True
    operator.mission_route_lock = SimpleNamespace(
        confirm_auto_started=lambda: confirmed.append(True),
        cancel_rejected_auto_start=lambda: None,
    )

    OperatorApp._finish_auto_route_activation(operator, result)

    assert confirmed == []
    assert operator._pending_auto_route_activation is False


@pytest.mark.parametrize("command", ["takeoff", "land"])
@pytest.mark.parametrize("result", [False, None, 1, "ok"])
def test_non_true_flight_completion_does_not_confirm_success_or_landing(
    command: str, result,
) -> None:
    operator = OperatorApp.__new__(OperatorApp)
    messages = []
    released = []
    operator.backend = SimpleNamespace(
        state=SimpleNamespace(tracker_state="HOVER", flight_state="landed"),
        _flight_state_name=lambda: (_ for _ in ()).throw(
            AssertionError("landed state must not be checked after failed cleanup")
        ),
    )
    operator.write_log = messages.append
    operator.mission_route_lock = SimpleNamespace(
        active=True,
        release_after_confirmed_landed=lambda _state: released.append(True),
    )

    assert OperatorApp._finish_flight_expectation(operator, command, result) is True

    assert released == []
    assert any("未確認" in message for message in messages)


def test_flight_completion_message_shows_the_plain_state_name_not_the_enum_repr() -> None:
    """str(TrackerState.HOVER) is "TrackerState.HOVER" (Enum.__str__, not the
    plain value) unless unwrapped via .value first -- that must never reach
    an operator-visible log line."""
    from operator_state import TrackerState

    operator = OperatorApp.__new__(OperatorApp)
    messages = []
    operator.backend = SimpleNamespace(
        state=SimpleNamespace(tracker_state=TrackerState.HOVER, flight_state="landed"),
    )
    operator.write_log = messages.append
    operator.mission_route_lock = SimpleNamespace(active=False)

    assert OperatorApp._finish_flight_expectation(operator, "takeoff", False) is True

    assert messages == ["起飛未確認執行成功：HOVER"]


class _ShutdownSessionLog:
    def __init__(self):
        self.reasons = []

    def close(self, *, reason):
        self.reasons.append(reason)


class _ShutdownBackend:
    is_live = True

    def __init__(self, result=True, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def cleanup(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_shutdown_coordinator_keeps_live_session_open_when_cleanup_rejected():
    backend = _ShutdownBackend(result=False)
    session_logs = _ShutdownSessionLog()
    destroyed = []
    messages = []
    coordinator = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=session_logs,
        write_log=messages.append,
        destroy=lambda: destroyed.append(True),
    )

    assert coordinator.shutdown() is False
    assert backend.calls == 1
    assert session_logs.reasons == []
    assert destroyed == []
    assert any("重試" in message for message in messages)


def test_shutdown_coordinator_keeps_live_session_open_when_cleanup_raises():
    backend = _ShutdownBackend(error=RuntimeError("landing unavailable"))
    session_logs = _ShutdownSessionLog()
    destroyed = []
    messages = []
    coordinator = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=session_logs,
        write_log=messages.append,
        destroy=lambda: destroyed.append(True),
    )

    assert coordinator.shutdown() is False
    assert session_logs.reasons == []
    assert destroyed == []
    assert any("例外" in message for message in messages)


def test_shutdown_coordinator_allows_legacy_non_live_cleanup_none():
    backend = _ShutdownBackend(result=None)
    backend.is_live = False
    session_logs = _ShutdownSessionLog()
    destroyed = []
    coordinator = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=session_logs,
        write_log=lambda _message: None,
        destroy=lambda: destroyed.append(True),
    )

    assert coordinator.shutdown(reason="sim_close") is True
    assert session_logs.reasons == ["sim_close"]
    assert destroyed == [True]


def test_operator_on_close_keeps_window_when_live_cleanup_fails(monkeypatch):
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = _ShutdownBackend(result=False)
    operator.session_logs = _ShutdownSessionLog()
    operator.logs = []
    operator.write_log = operator.logs.append
    destroyed = []
    operator.destroy = lambda: destroyed.append(True)
    callbacks = []
    threads = []
    operator.after = lambda _delay, callback: callbacks.append(callback)

    class _Thread:
        def __init__(self, *, target, **_kwargs):
            self.target = target
            threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(app.threading, "Thread", _Thread)

    operator._on_close()
    assert operator.backend.calls == 0
    assert len(threads) == 1
    threads[0].target()
    assert len(callbacks) == 1
    callbacks[0]()

    assert operator.session_logs.reasons == []
    assert destroyed == []
    assert any("重試" in message for message in operator.logs)


def test_operator_on_close_runs_cleanup_in_background_and_retries_after_failure(
    monkeypatch,
):
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = SimpleNamespace(is_live=True)
    operator.write_log = lambda _message: None
    operator.destroy = lambda: destroyed.append(True)
    destroyed = []
    callbacks = []
    threads = []

    class _Coordinator:
        def __init__(self):
            self.results = iter((False, True))

        def shutdown(self, *, reason):
            assert reason == "ui_window_close"
            return next(self.results)

    operator._shutdown_coordinator = _Coordinator()
    operator.after = lambda _delay, callback: callbacks.append(callback)

    class _Thread:
        def __init__(self, *, target, **_kwargs):
            self.target = target
            threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(app.threading, "Thread", _Thread)

    operator._on_close()
    assert len(threads) == 1
    assert destroyed == []
    threads[0].target()
    callbacks.pop(0)()
    assert destroyed == []

    operator._on_close()
    assert len(threads) == 2
    threads[1].target()
    callbacks.pop(0)()
    assert destroyed == [True]
    assert operator._shutdown_completed is True


def test_mainloop_exit_does_not_repeat_completed_shutdown_cleanup():
    import flight_operator_app as app

    launch = SimpleNamespace(app=SimpleNamespace(_shutdown_completed=True))

    app._close_operator_launch(launch)


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
    source = inspect.getsource(operator_tick._refresh_held_controls)
    assert "_stick_vector_active" in source
    assert "_send_stick_vector(app.stick_left.value, app.stick_right.value)" in source


def test_site_switch_keeps_the_tk_process_and_rebuilds_runtime_off_thread() -> None:
    source = inspect.getsource(app.OperatorApp.request_site_profile_restart)
    assert "_on_close" not in source
    assert "threading.Thread" in source
    assert "replace_active_site_runtime" in source
    assert "os.execv(" not in inspect.getsource(app.main)


def test_startup_asks_about_a_missing_route_for_the_active_site() -> None:
    """Direct startup still asks when the profile has no route."""
    source = inspect.getsource(app.OperatorApp._check_active_site_route)
    assert "follow_up_route_for_site" in source
    assert "route_json" in source

    scheduled = inspect.getsource(app.OperatorApp.__init__)
    assert "_check_active_site_route" in scheduled, "nothing schedules the check"


def _localization_timing_from_state(*, source_stamp: float, telemetry_ns):
    operator = OperatorApp.__new__(OperatorApp)
    operator.video_display_frame_name = "stream_000001"
    operator.video_frame = object()
    operator._video_frame_timing = {}
    operator._frame_rgb_bytes_for_worker = lambda _frame: b"rgb"
    operator.backend = SimpleNamespace(
        state=SimpleNamespace(
            att_roll=0.1,
            att_pitch=-0.2,
            att_yaw=1.3,
            speed_north_mps=0.4,
            speed_east_mps=-0.1,
            speed_down_mps=0.0,
            telemetry_read_mono_ns=telemetry_ns,
        )
    )
    prepared = OperatorApp._prepare_localization_submission(
        operator,
        source_stamp=source_stamp,
        boot_retry=False,
        lost_retry=False,
        hold_retry=False,
    )
    assert prepared is not None
    return prepared[2]


def test_localization_submission_stamps_fused_telemetry_at_acquisition() -> None:
    capture = 100.0
    fused_ns = 100_050_000_000
    timing = _localization_timing_from_state(
        source_stamp=capture, telemetry_ns=fused_ns,
    )
    assert timing["source_frame_stamp_mono"] == capture
    assert timing["fused_telemetry_mono"] == pytest.approx(100.05)
    assert timing["fused_telemetry_mono"] != timing["source_frame_stamp_mono"]
    assert timing["fused_roll"] == pytest.approx(0.1)
    assert timing["fused_yaw"] == pytest.approx(1.3)
    assert timing["fused_speed_north"] == pytest.approx(0.4)


def test_localization_submission_omits_unstamped_fused_telemetry() -> None:
    timing = _localization_timing_from_state(
        source_stamp=100.0, telemetry_ns=None,
    )
    assert "fused_telemetry_mono" not in timing
    assert timing["fused_roll"] == pytest.approx(0.1)
    assert timing["source_frame_stamp_mono"] == 100.0


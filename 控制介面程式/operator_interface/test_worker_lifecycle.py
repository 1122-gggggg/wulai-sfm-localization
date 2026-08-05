from __future__ import annotations

import json
import importlib.util
import inspect
import io
import queue
import select
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image

import flight_operator_app as app
import live_localizer_protocol as localizer_protocol
import live_localizer_worker as localizer_worker
from site_profile import QueryCamera


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_default_overlay_uses_the_portable_route_fallback() -> None:
    assert app.DEFAULT_ROUTE.resolve() == app._DEFAULT_ROUTE_FALLBACK.resolve()


def test_pipeline_metrics_summary_uses_operator_visible_rates_and_e2e_latency() -> None:
    text = app.format_pipeline_metrics_summary(
        localization_fps=7.25,
        stream_fps=29.94,
        e2e_ms=143.21,
        e2e_p95_ms=188.76,
        frame_age_ms=41.55,
    )

    assert text == (
        "定位端到端 FPS 7.2 | 影像串流 FPS 29.9 | "
        "端到端延遲 143.2 ms | p95（近 5 秒）188.8 ms | 影格年齡 41.5 ms"
    )


def test_pipeline_metrics_summary_marks_unavailable_measurements() -> None:
    text = app.format_pipeline_metrics_summary(
        localization_fps=None,
        stream_fps=None,
        e2e_ms=None,
        e2e_p95_ms=None,
        frame_age_ms=None,
    )

    assert text == (
        "定位端到端 FPS N/A | 影像串流 FPS N/A | "
        "端到端延遲 N/A | p95（近 5 秒）N/A | 影格年齡 N/A"
    )


def test_rolling_event_fps_expires_when_deliveries_stop() -> None:
    recent, rate = app._rolling_event_fps([94.0, 95.0, 96.0], now=96.0)
    assert recent == [94.0, 95.0, 96.0]
    assert rate == pytest.approx(1.0)

    recent, rate = app._rolling_event_fps(recent, now=102.0)
    assert recent == []
    assert rate == 0.0


def test_localization_metrics_prunes_e2e_samples_without_crashing(monkeypatch) -> None:
    monkeypatch.setattr(app.time, "monotonic", lambda: 100.0)
    operator = SimpleNamespace(
        live_result_times=[],
        loc_fps=0.0,
        loc_latency_ms=None,
        loc_wall_ms=None,
        loc_e2e_ms=None,
        _loc_e2e_ms_samples=[(94.0, 80.0)],
        loc_core_fps=0.0,
        loc_stage="-",
        _loc_wall_ms_samples=[],
        loc_health="OK",
        _loc_fail_count=0,
        _loc_ok_count=0,
        loc_pose_updated_mono=None,
        loc_health_inliers=0,
        loc_health_reproj=None,
        loc_hold_engage_count=0,
        loc_recovery_fix_count=0,
        loc_recovery_text="",
        _append_loc_metrics=lambda _result: None,
    )

    app.OperatorApp.update_localization_metrics(
        operator,
        {
            "success": True,
            "inliers": app.LOC_LOW_INLIERS,
            "core_wall_ms": 50.0,
            "wall_ms": 70.0,
            "e2e_submit_to_ui_ms": 90.0,
            "mode": "TRACK",
            "next_mode": "TRACK",
        },
    )

    assert operator._loc_e2e_ms_samples == [(100.0, 90.0)]
    assert operator.loc_e2e_ms == 90.0


def test_low_confidence_hold_is_an_engagement_event() -> None:
    assert "ENGAGE_LOW_CONF" in app.CONFIDENCE_HOLD_ENGAGE_EVENTS
    assert "ENGAGE_LOW" not in app.CONFIDENCE_HOLD_ENGAGE_EVENTS


def test_pipeline_metrics_expire_old_e2e_samples() -> None:
    operator = SimpleNamespace(
        inspecting=True,
        loc_fps=8.0,
        stream_fps_instant=30.0,
        live_result_times=[97.0, 97.125, 97.25],
        _stream_frame_times=[99.90, 99.95, 100.0],
        loc_e2e_ms=120.0,
        _loc_e2e_ms_samples=[(93.0, 80.0), (98.0, 110.0), (99.0, 120.0)],
        video_frame=object(),
        _video_frame_stamp=99.9,
        _is_live_backend=lambda: False,
    )

    text = app.OperatorApp.pipeline_metrics_summary(operator, now=100.0)

    assert operator._loc_e2e_ms_samples == [(98.0, 110.0), (99.0, 120.0)]
    assert text.startswith("定位端到端 FPS（實機鏈路模擬） 8.0")
    assert "p95（近 5 秒）120.0 ms" in text
    assert "影格年齡 100.0 ms" in text


def test_pipeline_metrics_mark_simulated_eof_as_stopped() -> None:
    operator = SimpleNamespace(
        inspecting=True,
        loc_fps=12.7,
        stream_fps_instant=26.0,
        live_result_times=[99.8, 99.9, 100.0],
        _stream_frame_times=[99.90, 99.95, 100.0],
        loc_e2e_ms=95.8,
        _loc_e2e_ms_samples=[(99.0, 95.8)],
        video_frame=object(),
        _video_frame_stamp=99.9,
        current_state=SimpleNamespace(stream="EOF_HOLD"),
        _is_live_backend=lambda: False,
    )

    text = app.OperatorApp.pipeline_metrics_summary(operator, now=100.0)

    assert text.startswith("定位端到端 FPS（實機鏈路模擬） N/A")
    assert "影像串流 FPS 0.0" in text
    assert "端到端延遲 N/A" in text
    assert text.endswith("狀態 影片已播完，保留最後一幀")


def test_pillow_overlay_font_resolves_to_a_cjk_capable_local_font() -> None:
    font_path = app.resolve_pil_ui_font_path()

    assert font_path is not None
    assert font_path.is_file()
    assert "CJK" in font_path.name or "NotoSansTC" in font_path.name


def test_simulated_video_hud_never_claims_to_be_live() -> None:
    simulated = app.video_hud_identity(False, "LIVE")
    real = app.video_hud_identity(True, "MANUAL")

    assert simulated == "SIMULATED ANAFI | mode=SIM"
    assert "LIVE" not in simulated
    assert real == "REAL ANAFI | mode=MANUAL"
    assert app.operator_mode_label(False, "LIVE") == "SIM"
    assert app.operator_mode_label(True, "MANUAL") == "MANUAL"
    assert app.localization_fps_metric_label(False) == "定位端到端 FPS（實機鏈路模擬）"
    assert app.localization_fps_metric_label(True) == "定位端到端 FPS（實機）"


def test_worker_result_backlog_keeps_only_the_newest_items(monkeypatch) -> None:
    client = app.LiveWorkerClient.__new__(app.LiveWorkerClient)
    client.results = queue.Queue(maxsize=2)
    client._result_notify_write_fd = -1
    monkeypatch.setattr(app.os, "write", lambda *_args: 1)

    client._publish_result({"seq": 1})
    client._publish_result({"seq": 2})
    client._publish_result({"seq": 3})

    assert [row["seq"] for row in client.poll_results()] == [2, 3]


def test_heading_arrow_polygon_points_at_the_current_heading() -> None:
    arrow = app.heading_arrow_polygon(100.0, 80.0, 120.0, 80.0)

    assert arrow is not None
    tip, *body = arrow
    assert tip == (118.0, 80.0)
    assert all(point[0] < tip[0] for point in body)


def test_route_visibility_toggle_does_not_remove_loaded_route() -> None:
    operator = app.OperatorApp.__new__(app.OperatorApp)
    operator.route_pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    operator.route_visible = True
    operator.show_route_var = SimpleNamespace(get=lambda: False)
    operator._map_dirty_key = ("cached",)
    redraws = []
    operator.redraw_map_only = lambda: redraws.append(True)

    app.OperatorApp.set_route_visibility(operator)

    assert not operator.route_visible
    assert operator.route_pts == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert operator._map_dirty_key is None
    assert redraws == [True]


def test_boot_hold_waits_for_worker_even_when_timed_hold_is_disabled() -> None:
    operator = SimpleNamespace(
        inspecting=True,
        boot_lock_done=True,
        localizer=SimpleNamespace(ready=False),
    )

    assert app.OperatorApp.boot_holding(operator)
    operator.localizer.ready = True
    assert not app.OperatorApp.boot_holding(operator)


def test_heading_arrow_polygon_skips_zero_length_heading() -> None:
    assert app.heading_arrow_polygon(100.0, 80.0, 100.0, 80.0) is None


def test_normalize_camera_forward_requires_a_finite_nonzero_vector() -> None:
    forward = app.normalize_camera_forward([0.0, 3.0, 4.0])

    assert np.allclose(forward, [0.0, 0.6, 0.8])
    assert app.normalize_camera_forward([0.0, 0.0, 0.0]) is None
    assert app.normalize_camera_forward([0.0, float("nan"), 1.0]) is None


def test_camera_frustum_square_face_is_ahead_of_camera() -> None:
    points = app.camera_frustum_world_points(
        np.zeros(3), np.eye(3), length=2.0, hfov_deg=90.0, aspect_ratio=2.0,
    )

    apex, face_center, *corners = points
    assert np.allclose(apex, [0.0, 0.0, 0.0])
    assert np.allclose(face_center, [0.0, 0.0, 2.0])
    assert np.allclose(np.mean(corners, axis=0), face_center)
    assert {round(point[0], 6) for point in corners} == {-2.0, 2.0}
    assert {round(point[1], 6) for point in corners} == {-1.0, 1.0}


def test_normalize_camera_axes_rejects_non_orthogonal_axes() -> None:
    assert app.normalize_camera_axes(np.eye(3)) is not None
    assert app.normalize_camera_axes(np.ones((3, 3))) is None


def test_missing_detector_is_disabled_by_default() -> None:
    assert app.default_live_detect(Path("/definitely/missing/model.engine")) is False


def test_yolo_detector_default_off_even_if_model_path_looks_valid(tmp_path: Path) -> None:
    fake = tmp_path / "fake.engine"
    fake.write_bytes(b"x")
    assert app.default_live_detect(fake) is False


def test_worker_python_defaults_to_current_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFM_TEST_WORKER_PYTHON", raising=False)
    assert app.default_worker_python("SFM_TEST_WORKER_PYTHON") == sys.executable
    monkeypatch.setenv("SFM_TEST_WORKER_PYTHON", "/explicit/python")
    assert app.default_worker_python("SFM_TEST_WORKER_PYTHON") == "/explicit/python"


@pytest.mark.parametrize(
    ("requested", "legacy_live", "expected", "is_live"),
    (
        ("simulated-stream", False, "simulated-stream", False),
        ("real-flight", False, "real-flight", True),
        ("", False, "simulated-stream", False),
        ("", True, "real-flight", True),
    ),
)
def test_operator_interface_modes_are_exactly_two_and_mutually_exclusive(
    requested: str, legacy_live: bool, expected: str, is_live: bool,
) -> None:
    mode, resolved_live = app.resolve_operator_interface(
        requested, legacy_live, ""
    )

    assert mode == expected
    assert resolved_live is is_live


def test_real_flight_interface_rejects_video_file_fallback() -> None:
    with pytest.raises(ValueError, match="accepts only the ANAFI PDRAW stream"):
        app.resolve_operator_interface("real-flight", False, "/tmp/replay.mp4")


def test_simulated_interface_rejects_legacy_live_flag() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        app.resolve_operator_interface("simulated-stream", True, "")


def test_auto_inspect_enables_feed_without_backend_command() -> None:
    class FakeOperator:
        _begin_inspection_feed = app.OperatorApp._begin_inspection_feed
        begin_auto_inspect = app.OperatorApp.begin_auto_inspect
        send = app.OperatorApp.send

        def __init__(self):
            self.inspecting = False
            self.inspect_start = None
            self.processed_frames = 9
            self.overall_fps = 9.0
            self.next_stream_frame_time = 9.0
            self.stream_fps_instant = 9.0
            self._lifetime_stream_fps = 9.0
            self._stream_frame_times = []
            self._loc_e2e_ms_samples = []
            self.logs = []
            self.backend_calls = []
            self.backend = SimpleNamespace(
                is_live=True,
                command=lambda name, **payload: self.backend_calls.append((name, payload)),
            )
            self._nudge_keys_held = set()
            self.stream_lost_since = None

        def write_log(self, text: str) -> None:
            self.logs.append(text)

    operator = FakeOperator()
    operator.begin_auto_inspect()
    assert operator.inspecting
    assert operator.backend_calls == []

    # The human button path keeps its existing backend command semantics.
    manual = FakeOperator()
    manual.send("start_auto")
    assert manual.backend_calls == [("start_auto", {})]


@pytest.mark.parametrize("mode", ["auto", "global", "weak", "track", "relocalize"])
def test_live_localizer_mode_header_round_trip(mode: str) -> None:
    assert localizer_protocol.decode_mode(localizer_protocol.encode_mode(mode)) == mode


def test_live_localizer_mode_header_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="control header"):
        localizer_protocol.decode_mode(b"BAD!\x00")
    with pytest.raises(ValueError, match="mode code"):
        localizer_protocol.decode_mode(localizer_protocol.MAGIC + b"\xff")
    with pytest.raises(ValueError, match="unsupported"):
        localizer_protocol.encode_mode("fly")


def test_edm_worker_fails_clearly_when_cuda_is_unavailable() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )

    with pytest.raises(RuntimeError, match="requires a working CUDA device"):
        localizer_worker.require_edm_cuda(fake_torch)


def test_timed_localizer_header_round_trip() -> None:
    header = localizer_protocol.encode_request("auto", 123.456)

    mode, capture_stamp = localizer_protocol.decode_request(header)

    assert mode == "auto"
    assert capture_stamp == pytest.approx(123.456)


def test_live_localizer_client_snapshots_only_localizer_mode_prefix() -> None:
    client = object.__new__(app.LiveLocalizerClient)
    client._runtime_benchmark_control = True
    client._benchmark_mode_lock = threading.Lock()
    client._benchmark_mode = "auto"
    client._relocalize_once = False
    assert client._request_prefix() == localizer_protocol.encode_mode("auto")
    assert client.set_benchmark_mode("weak") == "weak"
    assert client._request_prefix() == localizer_protocol.encode_mode("weak")
    client.request_relocalize()
    assert client._request_prefix() == localizer_protocol.encode_mode("relocalize")
    assert client._request_prefix() == localizer_protocol.encode_mode("weak")
    with pytest.raises(ValueError, match="unsupported"):
        client.set_benchmark_mode("takeoff")


@pytest.mark.parametrize("enabled", [False, True])
def test_live_localizer_client_passes_neuflow_flag_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, enabled: bool,
) -> None:
    captured = {}

    def fake_worker_init(_self, cmd, *_args, **_kwargs) -> None:
        captured["cmd"] = cmd
        captured["kwargs"] = _kwargs

    monkeypatch.setattr(app.LiveWorkerClient, "__init__", fake_worker_init)

    app.LiveLocalizerClient(
        Path("worker.py"),
        sys.executable,
        1280,
        720,
        Path("bundle.pt"),
        neuflow_track=enabled,
    )

    assert ("--neuflow-track" in captured["cmd"]) is enabled
    assert "--startup-handshake" in captured["cmd"]
    assert captured["kwargs"]["expect_ready_event"] is True


def test_live_localizer_client_passes_projection_sidecar(monkeypatch) -> None:
    captured = {}

    def fake_worker_init(_self, cmd, *_args, **_kwargs) -> None:
        captured["cmd"] = cmd

    monkeypatch.setattr(app.LiveWorkerClient, "__init__", fake_worker_init)

    app.LiveLocalizerClient(
        Path("worker.py"),
        sys.executable,
        1280,
        720,
        Path("bundle.pt"),
        projection_track=True,
        track_landmarks=Path("track_landmarks_v1.npz"),
    )

    index = captured["cmd"].index("--track-landmarks")
    assert "--projection-track" in captured["cmd"]
    assert captured["cmd"][index + 1] == "track_landmarks_v1.npz"


def test_live_localizer_client_passes_profile_query_camera(monkeypatch) -> None:
    captured = {}

    def fake_worker_init(_self, cmd, *_args, **_kwargs) -> None:
        captured["cmd"] = cmd

    monkeypatch.setattr(app.LiveWorkerClient, "__init__", fake_worker_init)
    camera = QueryCamera(
        model="SIMPLE_RADIAL",
        width=1280,
        height=720,
        params=(934.139423, 640.0, 360.0, 0.001061),
    )

    app.LiveLocalizerClient(
        Path("worker.py"),
        sys.executable,
        1280,
        720,
        Path("bundle.pt"),
        query_camera=camera,
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--query-camera-model") + 1] == "SIMPLE_RADIAL"
    assert cmd[cmd.index("--query-camera-width") + 1] == "1280"
    assert cmd[cmd.index("--query-camera-height") + 1] == "720"
    start = cmd.index("--query-camera-params") + 1
    assert [float(value) for value in cmd[start:start + 4]] == list(camera.params)


def test_shipped_edm_production_profile_pins_validated_parameters() -> None:
    profile_path = (
        Path(__file__).resolve().parents[2]
        / "定位演算法"
        / "configs"
        / "edm_production_profile.json"
    )

    profile = localizer_worker.load_edm_production_profile(profile_path)

    assert profile["matcher"] == {
        "coarse_topk": 3225,
        "mconf_thr": 0.2,
        "fp16": True,
        "input_size": [1024, 576],
        "reference_cache_size": 32,
    }
    assert profile["tracker"]["acquire_initial_topk"] == 2
    assert profile["tracker"]["max_reproj_error_acquire"] == 5.0
    assert profile["tracker"]["max_reproj_error_track"] == 6.0
    assert profile["tracker"]["prediction_max_dt"] == 0.25
    assert profile["tracker"]["local_topk"] == 1
    assert profile["tracker"]["weak_local_topk"] == 3
    assert profile["tracker"]["lost_local_topk"] == 5
    assert profile["tracker"]["lost_local_grace_frames"] == 12
    assert profile["tracker"]["recovery_bank_size"] == 192
    assert profile["tracker"]["recovery_scan_topk"] == 2


def test_xfeat_topk_override_keeps_validated_three_then_four_retry() -> None:
    cfg = SimpleNamespace(
        matcher_mode="lighterglue",
        acquire_matcher_mode="lighterglue",
        local_topk=5,
        weak_local_topk=8,
        adaptive_first_topk=3,
    )

    localizer_worker.apply_xfeat_runtime_overrides(
        cfg, matcher_mode="", local_topk=4,
    )

    assert cfg.local_topk == 4
    assert cfg.weak_local_topk == 8
    assert cfg.adaptive_first_topk == 3


def test_live_localizer_switches_mode_without_restarting_worker(tmp_path: Path) -> None:
    worker = (
        "import json,sys; "
        "[(lambda h,r: print(json.dumps({'success':False,'header':h.hex()}),flush=True))"
        "(sys.stdin.buffer.read(5),sys.stdin.buffer.read(3)) for _ in range(2)]"
    )
    client = object.__new__(app.LiveLocalizerClient)
    client._runtime_benchmark_control = True
    client._benchmark_mode_lock = threading.Lock()
    client._benchmark_mode = "auto"
    client._relocalize_once = False
    app.LiveWorkerClient.__init__(
        client,
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "mode-worker.log"),
        "mode-worker",
        "localizer",
        timeout_s=1.0,
    )
    client.restart_warmup_s = 0.0
    pid = client.proc.pid
    try:
        assert client.submit(1, "auto", Image.new("RGB", (1, 1)))
        assert _wait_for(lambda: not client.results.empty())
        first = client.poll_results()[0]
        client.set_benchmark_mode("weak")
        assert client.submit(2, "weak", Image.new("RGB", (1, 1)))
        assert _wait_for(lambda: not client.results.empty())
        second = client.poll_results()[0]
    finally:
        client.close()
    assert bytes.fromhex(first["header"]) == localizer_protocol.encode_mode("auto")
    assert bytes.fromhex(second["header"]) == localizer_protocol.encode_mode("weak")
    assert client.proc.pid == pid


def test_ready_handshake_blocks_submit_until_model_is_loaded(tmp_path: Path) -> None:
    worker = (
        "import json,sys,time; "
        "time.sleep(0.15); "
        "print(json.dumps({'event':'ready','startup_ms':150.0}),flush=True); "
        "raw=sys.stdin.buffer.read(3); "
        "print(json.dumps({'success':bool(raw)}),flush=True)"
    )
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "ready-worker.log"),
        "ready-worker",
        "ready",
        timeout_s=1.0,
        expect_ready_event=True,
    )
    client.restart_warmup_s = 1.0
    try:
        assert not client.ready
        assert not client.submit(1, "early", Image.new("RGB", (1, 1)))
        assert _wait_for(lambda: client.ready)
        assert client.startup_info["startup_ms"] == 150.0
        assert client.submit(2, "ready", Image.new("RGB", (1, 1)))
        assert _wait_for(lambda: not client.results.empty())
        assert client.poll_results()[0]["success"] is True
    finally:
        client.close()


@pytest.mark.parametrize("mode", ["global", "weak", "track"])
def test_offline_localization_benchmark_control_never_touches_backend(mode: str) -> None:
    class FakeLocalizer:
        def __init__(self):
            self.calls = []

        def set_benchmark_mode(self, selected: str) -> str:
            self.calls.append(selected)
            return selected

    class FakeVar:
        def __init__(self):
            self.value = ""

        def set(self, value: str) -> None:
            self.value = value

    class FakeOperator:
        _begin_inspection_feed = app.OperatorApp._begin_inspection_feed
        begin_auto_inspect = app.OperatorApp.begin_auto_inspect
        set_localization_benchmark_mode = app.OperatorApp.set_localization_benchmark_mode

        def __init__(self):
            self.localizer = FakeLocalizer()
            self.backend = SimpleNamespace(command=lambda *_a, **_k: pytest.fail(
                "localization benchmark mode touched the drone backend"))
            self.inspecting = False
            self.inspect_start = None
            self.processed_frames = 0
            self.overall_fps = 0.0
            self.next_stream_frame_time = 0.0
            self.stream_fps_instant = 0.0
            self._lifetime_stream_fps = 0.0
            self._stream_frame_times = []
            self._loc_e2e_ms_samples = []
            self.loc_benchmark_requested = "auto"
            self._loc_benchmark_pending = None
            self.loc_benchmark_mode_var = FakeVar()
            self.logs = []

        def write_log(self, text: str) -> None:
            self.logs.append(text)

    operator = FakeOperator()
    assert operator.set_localization_benchmark_mode(mode)
    assert operator.localizer.calls == [mode]
    assert operator.inspecting
    assert operator._loc_benchmark_pending == mode
    assert "不起飛" in operator.logs[-1]


def test_runtime_benchmark_modes_prepare_only_the_requested_tracker_branch() -> None:
    class State:
        def __init__(self):
            self.mode = "BOOT_INIT"
            self.last_center = None
            self.last_refs = []
            self.fail_count = 0
            self.bad_count = 0

    class Tracker:
        def __init__(self):
            self.state = State()
            self.clear_count = 0

        def _clear_tracking_history(self) -> None:
            self.clear_count += 1
            self.state.last_center = None
            self.state.last_refs = []

    tracker = Tracker()

    def seed(target: str) -> None:
        tracker.state.mode = target
        tracker.state.last_center = object()
        tracker.state.last_refs = [2]
        tracker.state.fail_count = 0
        tracker.state.bad_count = 0

    untouched = tracker.state
    assert not localizer_worker.apply_runtime_benchmark_mode(
        tracker, "auto", "auto", seed)
    assert tracker.state is untouched

    assert not localizer_worker.apply_runtime_benchmark_mode(
        tracker, "global", "auto", seed)
    assert tracker.state.mode == "LOST"
    assert tracker.state.last_center is None

    assert localizer_worker.apply_runtime_benchmark_mode(
        tracker, "weak", "global", seed)
    assert tracker.state.mode == "WEAK_TRACK"
    assert tracker.state.last_refs == [2]

    assert localizer_worker.apply_runtime_benchmark_mode(
        tracker, "track", "weak", seed)
    assert tracker.state.mode == "TRACK"
    assert tracker.state.last_refs == [2]

    assert not localizer_worker.apply_runtime_benchmark_mode(
        tracker, "auto", "track", seed)
    assert tracker.state.mode == "BOOT_INIT"
    assert tracker.state.last_center is None

    with pytest.raises(ValueError, match="unsupported"):
        localizer_worker.apply_runtime_benchmark_mode(
            tracker, "takeoff", "auto", seed)


def test_busy_skip_counts_unique_display_frames_and_all_attempts() -> None:
    operator = SimpleNamespace(
        inspecting=True,
        localizer=SimpleNamespace(busy=lambda: True),
        video_frame=object(),
        video_display_index=7,
        last_submitted_index=6,
        loc_every_n_frames=1,
        _submit_skip_busy=0,
        _submit_busy_attempts=0,
        _last_busy_skip_index=-1,
        _video_frame_stamp=0.0,
        _last_coalesce_stamp=0.0,
        _last_coalesce_mono=0.0,
        lost_hold=None,
        boot_holding=lambda: False,
        lost_holding=lambda: False,
    )
    app.OperatorApp.submit_current_frame_for_localization(operator)
    app.OperatorApp.submit_current_frame_for_localization(operator)
    assert operator._submit_skip_busy == 1
    assert operator._submit_busy_attempts == 2
    operator.video_display_index = 8
    app.OperatorApp.submit_current_frame_for_localization(operator)
    assert operator._submit_skip_busy == 2
    assert operator._submit_busy_attempts == 3


def test_empty_result_poll_preserves_pose_until_render_tick() -> None:
    operator = SimpleNamespace(
        live_new_pose=True,
        localizer=SimpleNamespace(poll_results=lambda: []),
    )
    app.OperatorApp.update_live_results(operator)
    assert operator.live_new_pose is True


def test_localization_result_poll_reschedules_independently() -> None:
    calls = []
    class FakeOperator:
        _loc_result_poll_ms = 5
        poll_localization_results = app.OperatorApp.poll_localization_results

        def update_live_results(self) -> None:
            calls.append("poll")

        def after(self, delay, callback) -> None:
            calls.append((delay, callback))

    operator = FakeOperator()
    operator.poll_localization_results()
    assert calls[0] == "poll"
    assert calls[1] == (5, operator.poll_localization_results)


def test_stream_lost_display_precedes_busy_localizer_status() -> None:
    operator = SimpleNamespace(
        live_pose=app.np.zeros(4),
        live_result={
            "success": False,
            "mode": "TRACK",
            "next_mode": "WEAK_TRACK",
            "inliers": 0,
            "reproj_rms": None,
        },
        localizer=SimpleNamespace(busy=lambda: True),
        video_frame_fresh=False,
        stream_lost_since=time.monotonic(),
        boot_holding=lambda: False,
    )
    state = app.DroneState(
        stream="LOST", loc="STREAM_LOST", tracker_state="STREAM_LOST_HOVER")
    result = app.OperatorApp.state_from_live(operator, state)
    assert result.stream == "LOST"
    assert result.loc == "STREAM_LOST"
    assert result.tracker_state == "STREAM_LOST_MANUAL"


def test_ui_arrival_timing_uses_submit_and_neutral_source_stamp_names() -> None:
    result = app.annotate_ui_arrival_timing(
        {
            "client_submit_mono": 9.90,
            "client_response_mono": 9.98,
            "source_frame_stamp_mono": 9.80,
            "source_stamp_semantics": "stream_monotonic_mapped_or_receipt",
        },
        now=10.0,
    )
    assert result["e2e_submit_to_ui_ms"] == pytest.approx(100.0)
    assert result["ui_poll_delay_ms"] == pytest.approx(20.0)
    assert result["source_stamp_age_at_ui_ms"] == pytest.approx(200.0)
    assert "capture" not in " ".join(result).lower()


def test_composite_stage_does_not_hide_weak_tracker_state() -> None:
    assert app.localization_result_is_weak({
        "composite_stage": "nn_fast_accept",
        "next_mode": "WEAK_TRACK",
    })
    assert app.localization_result_is_weak({
        "composite_stage": "lg_full_after_nn",
        "next_mode": "TRACK",
        "weak": True,
    })
    assert not app.localization_result_is_weak({
        "composite_stage": "lg_full_after_nn",
        "next_mode": "TRACK",
        "weak": False,
    })


def test_confidence_hold_does_not_engage_on_low_results() -> None:
    policy = app.LostHoldPolicy(max_attempts=3, timeout_s=10.0)

    assert policy.on_result(
        success=True, low_confidence=True, strong_relocalize=False, next_mode="TRACK",
        frame_index=10, now=1.0) is None
    assert policy.on_result(
        success=True, low_confidence=True, strong_relocalize=False, next_mode="WEAK_TRACK",
        frame_index=11, now=1.1) is None
    assert not policy.active


def test_confidence_hold_engages_only_after_tracker_is_lost() -> None:
    policy = app.LostHoldPolicy(max_attempts=3, timeout_s=10.0)

    assert policy.on_result(
        success=False, low_confidence=False, strong_relocalize=False,
        next_mode="WEAK_TRACK", frame_index=19, now=1.9) is None
    assert not policy.active
    assert policy.on_result(
        success=False, low_confidence=False, strong_relocalize=False,
        next_mode="LOST", frame_index=20, now=2.0) == "ENGAGE_FAIL"
    assert policy.active


def test_real_low_confidence_escalation_hands_off_and_requests_megaloc_once() -> None:
    fail_safe_calls = []
    relocalize_calls = []
    operator = app.OperatorApp.__new__(app.OperatorApp)
    operator.lost_hold = app.LostHoldPolicy(
        max_attempts=3,
        timeout_s=10.0,
        low_confidence_results=2,
        hold_on_low_confidence=True,
    )
    operator.inspecting = True
    operator.video_display_frame_name = "frame_000011.jpg"
    operator.video_display_index = 11
    operator._is_live_backend = lambda: True
    operator.backend = SimpleNamespace(
        fail_safe=lambda reason: fail_safe_calls.append(reason)
    )
    operator.localizer = SimpleNamespace(
        request_relocalize=lambda: relocalize_calls.append(True)
    )
    operator.write_log = lambda _message: None

    for index in range(2):
        result = {
            "success": True,
            "inliers": 20,
            "reproj_rms": 2.0,
            "next_mode": "WEAK_TRACK",
            "mode": "WEAK_TRACK",
        }
        operator._apply_lost_hold_result(result)
        if index == 0:
            assert fail_safe_calls == []
            assert relocalize_calls == []

    assert fail_safe_calls == [app.FailureReason.LOCALIZATION_WEAK]
    assert relocalize_calls == [True]


def test_metrics_keep_wall_compatibility_and_explicit_core_timing() -> None:
    sink = io.StringIO()
    operator = SimpleNamespace(
        _ensure_loc_metrics_log=lambda: None,
        _loc_metrics_f=sink,
        loc_fps=12.5,
        _submit_ok=3,
        _submit_skip_busy=2,
        _submit_busy_attempts=5,
    )
    app.OperatorApp._append_loc_metrics(operator, {
        "success": False,
        "wall_ms": 55.0,
        "core_wall_ms": 50.0,
        "e2e_submit_to_ui_ms": 70.0,
        "display_seq": 42,
        "hold_retry": True,
        "hold_kind": "lost",
        "relocalize_requested": True,
        "confidence_hold_event": "RELEASE_FIX",
        "limited_jump": {
            "confirmation_model": "constant_velocity",
            "confirmation_residual": 0.04,
        },
        "limited_jump_confirmed": True,
        "pose": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw_raw": 0.5},
    })
    record = json.loads(sink.getvalue())
    assert record["wall_ms"] == 55.0
    assert record["core_wall_ms"] == 50.0
    assert record["e2e_submit_to_ui_ms"] == 70.0
    assert record["display_seq"] == 42
    assert record["hold_retry"] is True
    assert record["relocalize_requested"] is True
    assert record["confidence_hold_event"] == "RELEASE_FIX"
    assert record["limited_jump"]["confirmation_model"] == "constant_velocity"
    assert record["limited_jump_confirmed"] is True
    assert record["pose"]["z"] == 3.0
    assert record["submit_skip_busy"] == 2
    assert record["submit_busy_attempts"] == 5


def test_temporal_pose_stabilizer_suppresses_single_frame_flip() -> None:
    stabilizer = app.TemporalPoseStabilizer(
        tau_s=0.15, max_speed_u_s=2.0, step_slack_u=0.03, max_step_u=0.15)
    samples = [
        np.array([0.00, 0.0, 0.0]),
        np.array([0.02, 0.0, 0.0]),
        np.array([1.00, 0.0, 0.0]),
        np.array([0.03, 0.0, 0.0]),
    ]
    outputs = [stabilizer.update(value, 1.0 + i * 0.06)[0] for i, value in enumerate(samples)]
    steps = [float(np.linalg.norm(b - a)) for a, b in zip(outputs, outputs[1:])]

    assert max(steps) < 0.03
    assert outputs[-1][0] < 0.03


def test_temporal_pose_stabilizer_rate_limits_sustained_relocation() -> None:
    stabilizer = app.TemporalPoseStabilizer(
        tau_s=0.15, max_speed_u_s=2.0, step_slack_u=0.03, max_step_u=0.15)
    first, _ = stabilizer.update(np.zeros(3), 1.0)
    outputs = [first]
    diagnostics = []
    for i in range(1, 8):
        value, info = stabilizer.update(np.array([1.0, 0.0, 0.0]), 1.0 + i * 0.06)
        outputs.append(value)
        diagnostics.append(info)
    steps = [float(np.linalg.norm(b - a)) for a, b in zip(outputs, outputs[1:])]

    assert max(steps) <= 0.150000001
    assert diagnostics[1]["limited"] is True
    assert outputs[-1][0] > 0.6


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"waypoints": []},
        {"waypoints": [[0.0, 1.0]]},
        {"waypoints": [[0.0, 1.0, float("nan")]]},
    ],
)
def test_invalid_route_fails_closed(tmp_path: Path, payload: dict) -> None:
    route = tmp_path / "route.json"
    route.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        app.load_route_glomap(str(route))


def test_valid_route_is_finite_and_converted(tmp_path: Path) -> None:
    route = tmp_path / "route.json"
    route.write_text(json.dumps({"waypoints": [[1.0, 2.0, 3.0]]}), encoding="utf-8")
    points = app.load_route_glomap(str(route))
    assert len(points) == 1
    assert points[0].tolist() == [1.0, -3.0, 2.0]


@pytest.mark.parametrize(
    "pose",
    [
        None,
        {},
        {"x": float("nan"), "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": float("inf"), "z": 0.0},
        {"x": 0.0, "y": 0.0, "z": -float("inf")},
    ],
)
def test_success_payload_with_invalid_pose_is_downgraded(pose: object) -> None:
    result, xyz = app.normalize_live_localization_result(
        {"success": True, "pose": pose, "frame_name": "bad"})
    assert result["success"] is False
    assert "invalid localization pose" in result["error"]
    assert xyz is None


def test_operator_anafi_profile_matches_simulator_contract() -> None:
    profile_path = (
        app.SYSTEM_ROOT / "模擬器" /
        "sphinx_anafi_path_convergence" / "anafi_profile.py"
    )
    if not profile_path.is_file():
        pytest.skip("simulator profile is not shipped in the portable repository")
    spec = importlib.util.spec_from_file_location("_operator_test_anafi_profile", profile_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    sim = module.ANAFI_PROFILE
    ui = app.ANAFI
    assert ui.model == sim.name
    assert ui.weight_g == round(sim.mass_kg * 1000)
    assert ui.max_horizontal_speed_mps == sim.max_horizontal_speed_mps
    assert ui.max_vertical_speed_mps == sim.max_ascent_speed_mps == sim.max_descent_speed_mps
    assert ui.max_yaw_rate_dps == sim.max_angular_speed_deg_s
    assert ui.max_wind_kmh == sim.wind_resistance_kmh
    assert ui.takeoff_hover_m == sim.takeoff_hover_height_m
    assert ui.gimbal_pitch_min_deg == sim.gimbal_pitch_min_deg
    assert ui.gimbal_pitch_max_deg == sim.gimbal_pitch_max_deg
    assert ui.gimbal_pitch_rate_dps == sim.gimbal_max_speed_deg_s
    assert (ui.stream_width, ui.stream_height, ui.stream_fps) == (
        sim.video_width_px, sim.video_height_px, sim.video_fps)
    assert ui.stream_latency_ms == sim.video_latency_ms
    assert ui.stream_mbps * 1_000_000 == sim.video_bitrate_bps


def test_worker_rgb_frame_view_is_zero_copy_and_lifetime_safe() -> None:
    raw = bytes(range(18))
    frame = localizer_worker.frame_view_from_rgb_bytes(raw, width=3, height=2)
    assert frame.shape == (2, 3, 3)
    assert frame.flags["C_CONTIGUOUS"]
    assert not frame.flags["OWNDATA"]
    assert frame.tobytes() == raw
    with pytest.raises(ValueError):
        localizer_worker.frame_view_from_rgb_bytes(raw[:-1], width=3, height=2)


def test_worker_read_buffer_is_preallocated_and_writable() -> None:
    expected = bytes(range(18))
    raw = localizer_worker.read_exact(io.BytesIO(expected), len(expected))
    frame = localizer_worker.frame_view_from_rgb_bytes(raw, width=3, height=2)
    assert isinstance(raw, bytearray)
    assert frame.flags["WRITEABLE"]
    assert frame.tobytes() == expected


def test_worker_can_refill_the_same_fixed_frame_buffer() -> None:
    first = bytes(range(18))
    second = bytes(reversed(range(18)))
    buffer = bytearray(18)

    assert localizer_worker.read_exact_into(io.BytesIO(first), buffer) == 18
    frame = localizer_worker.frame_view_from_rgb_bytes(buffer, width=3, height=2)
    assert frame.tobytes() == first
    assert localizer_worker.read_exact_into(io.BytesIO(second), buffer) == 18
    assert frame.tobytes() == second


def test_live_rgb_worker_input_is_a_zero_copy_memoryview() -> None:
    frame = app.np.zeros((app.STREAM_HEIGHT, app.STREAM_WIDTH, 3), dtype=app.np.uint8)
    raw = app.OperatorApp._frame_rgb_bytes_for_worker(SimpleNamespace(), frame)
    assert isinstance(raw, memoryview)
    assert raw.nbytes == frame.nbytes
    frame[0, 0, 0] = 73
    assert raw[0] == 73


def test_non_calibrated_zoom_pauses_localization_before_submit() -> None:
    logs = []
    operator = SimpleNamespace(
        inspecting=True,
        localizer=SimpleNamespace(busy=lambda: pytest.fail("must not submit at non-1x zoom")),
        video_frame=object(),
        video_display_index=1,
        backend=SimpleNamespace(state=SimpleNamespace(zoom=1.5)),
        _zoom_localization_paused=False,
        write_log=logs.append,
    )

    app.OperatorApp.submit_current_frame_for_localization(operator)

    assert operator._zoom_localization_paused
    assert "1.5x" in logs[0]
    assert app.localization_zoom_is_calibrated(1.0)
    assert not app.localization_zoom_is_calibrated(float("nan"))


def test_shared_frame_worker_coalesces_latest_slot_and_notifies(tmp_path: Path) -> None:
    worker = """
import argparse, json, sys, time
from multiprocessing import shared_memory
p = argparse.ArgumentParser()
p.add_argument('--frame-shm-name', required=True)
p.add_argument('--frame-shm-slots', required=True, type=int)
a = p.parse_args()
shm = shared_memory.SharedMemory(name=a.frame_shm_name)
try:
    for index in range(2):
        slot = sys.stdin.buffer.read(1)
        if not slot:
            break
        if index == 0:
            time.sleep(0.1)
        start = slot[0] * 6
        print(json.dumps({'success': False, 'pixel_sum': sum(shm.buf[start:start + 6])}), flush=True)
finally:
    shm.close()
"""
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker], 2, 1,
        str(tmp_path / "shared.log"), "shared-worker", "shared",
        timeout_s=1.0, use_shared_frames=True,
    )
    client.restart_warmup_s = 0.0
    try:
        assert client.submit(1, "first", np.full((1, 2, 3), 1, np.uint8))
        assert _wait_for(client.busy)
        assert client.submit(2, "middle", np.full((1, 2, 3), 2, np.uint8))
        assert client.submit(3, "latest", np.full((1, 2, 3), 3, np.uint8))
        ready, _, _ = select.select([client.result_notify_fd], [], [], 2.0)
        assert ready
        assert _wait_for(lambda: client.results.qsize() == 2)
        client.drain_result_notifications()
        results = client.poll_results()
    finally:
        client.close()
    assert [(row["display_seq"], row["pixel_sum"]) for row in results] == [
        (1, 6), (3, 18),
    ]


@pytest.mark.parametrize(
    ("requested", "count", "expected"),
    [(-1, 5, 2), (0, 5, 0), (4, 5, 4)],
)
def test_force_track_ref_resolution_is_explicit(requested: int, count: int,
                                                expected: int) -> None:
    assert localizer_worker.resolve_force_track_ref(requested, count) == expected


@pytest.mark.parametrize("requested", [-2, 5])
def test_force_track_ref_rejects_out_of_range(requested: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        localizer_worker.resolve_force_track_ref(requested, 5)


def test_submit_restarts_an_idle_exited_worker(tmp_path: Path) -> None:
    client = app.LiveWorkerClient(
        [sys.executable, "-c", "raise SystemExit(3)"],
        1,
        1,
        str(tmp_path / "worker.log"),
        "test-worker",
        "test",
        timeout_s=0.1,
    )
    try:
        old_pid = client.proc.pid
        assert _wait_for(lambda: client.proc.poll() is not None)
        client._last_restart = 0.0
        assert client.submit(1, "frame", Image.new("RGB", (1, 1))) is False
        assert _wait_for(lambda: client.proc.pid != old_pid)
    finally:
        client.close()
    assert not client.thread.is_alive()


def test_worker_client_reports_submit_pipe_worker_and_response_timing(tmp_path: Path) -> None:
    worker = (
        "import json,sys,time; "
        "sys.stdin.buffer.read(3); "
        "rn=time.monotonic_ns(); sn=time.monotonic_ns(); dn=time.monotonic_ns(); "
        "r=rn*1e-9; s=sn*1e-9; d=dn*1e-9; "
        "print(json.dumps({'success':False,'wall_ms':1.0,'core_wall_ms':1.0,"
        "'worker_read_done_mono':r,'worker_core_start_mono':s,"
        "'worker_core_done_mono':d,'worker_read_done_mono_ns':rn,"
        "'worker_core_start_mono_ns':sn,'worker_core_done_mono_ns':dn}),flush=True)"
    )
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker], 1, 1,
        str(tmp_path / "timing.log"), "timing-worker", "timing", timeout_s=1.0)
    client.restart_warmup_s = 0.0
    try:
        assert client.submit(
            1, "timed", Image.new("RGB", (1, 1)),
            timing_metadata={
                "ui_serialize_ms": 0.25,
                "source_frame_stamp_mono": time.monotonic() - 0.01,
                "source_stamp_semantics": "stream_monotonic_mapped_or_receipt",
                "frame_callback_enter_mono_ns": time.monotonic_ns() - 20_000_000,
                "frame_preprocess_start_mono_ns": time.monotonic_ns() - 15_000_000,
                "frame_yuv_ready_mono_ns": time.monotonic_ns() - 14_000_000,
                "frame_preprocess_done_mono_ns": time.monotonic_ns() - 10_000_000,
            })
        assert _wait_for(lambda: not client.results.empty())
        result = client.poll_results()[0]
    finally:
        client.close()
    for key in (
        "client_submit_mono", "client_dequeue_mono", "client_write_start_mono",
        "client_write_done_mono", "worker_read_done_mono", "worker_core_start_mono",
        "worker_core_done_mono", "client_response_mono", "client_queue_wait_ms",
        "client_pipe_write_ms", "client_roundtrip_ms", "submit_to_worker_read_ms",
        "worker_done_to_client_ms", "source_stamp_age_at_submit_ms",
        "client_submit_mono_ns", "client_dequeue_mono_ns",
        "client_write_start_mono_ns", "client_write_done_mono_ns",
        "worker_read_done_mono_ns", "worker_core_start_mono_ns",
        "worker_core_done_mono_ns", "client_response_mono_ns",
        "callback_to_preprocess_start_ms", "yuv_view_ms",
        "frame_preprocess_ms", "callback_to_submit_ms",
        "callback_to_inference_start_ms", "callback_to_localization_done_ms",
    ):
        assert key in result
    assert result["core_wall_ms"] == 1.0
    assert result["ui_serialize_ms"] == 0.25
    assert result["timing_clock"] == "host_monotonic_seconds"
    assert result["timing_clock_ns"] == "host_monotonic_ns"


def test_production_localizer_has_no_unconditional_device_wide_sync() -> None:
    source = inspect.getsource(localizer_worker)
    assert "torch.cuda.synchronize()" not in source
    assert "SFM_LOC_PROFILE_GPU" in source
    assert "gpu_end_event.synchronize()" in source


def test_restart_cooldown_check_is_atomic(tmp_path: Path) -> None:
    client = app.LiveWorkerClient(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        1,
        1,
        str(tmp_path / "restart.log"),
        "restart-worker",
        "restart",
        timeout_s=0.1,
    )
    results: list[bool] = []
    client._last_restart = 0.0
    client._proc_lock.acquire()
    threads = [threading.Thread(target=lambda: results.append(client._restart()))
               for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        assert _wait_for(lambda: all(thread.is_alive() for thread in threads))
    finally:
        client._proc_lock.release()
    for thread in threads:
        thread.join(timeout=3.0)
    try:
        assert results.count(True) == 1
        assert results.count(False) == 1
    finally:
        client.close()


def test_blocked_worker_write_times_out_and_recovers(tmp_path: Path) -> None:
    client = app.LiveWorkerClient(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        1024,
        1024,
        str(tmp_path / "blocked.log"),
        "blocked-worker",
        "blocked",
        timeout_s=0.05,
    )
    client.restart_warmup_s = 0.0
    try:
        old_pid = client.proc.pid
        assert client.submit(1, "large", Image.new("RGB", (1024, 1024))) is True
        assert _wait_for(lambda: bool(client.poll_results()), timeout=3.0)
        assert _wait_for(lambda: client.proc.pid != old_pid, timeout=3.0)
    finally:
        client.close()
    assert not client.thread.is_alive()


def test_ply_sampling_never_exceeds_requested_limit(tmp_path: Path) -> None:
    ply = tmp_path / "points.ply"
    header = (
        "ply\nformat ascii 1.0\nelement vertex 101\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rows = "".join(f"{i} 0 0 1 2 3\n" for i in range(101))
    ply.write_text(header + rows, encoding="ascii")
    assert len(app.read_ply_points(ply, max_points=100)) <= 100


def test_ply_reader_uses_rgb_after_normals(tmp_path: Path) -> None:
    ply = tmp_path / "normals.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        "1 2 3 0.1 0.2 0.3 4 5 6\n",
        encoding="ascii",
    )

    assert app.read_ply_points(ply, max_points=1).tolist() == [[1, 2, 3, 4, 5, 6]]


def test_partial_worker_response_times_out_and_recovers(tmp_path: Path) -> None:
    worker = (
        "import sys,time; "
        "sys.stdin.buffer.read(3); "
        "sys.stdout.write('{\"success\":'); sys.stdout.flush(); "
        "time.sleep(60)"
    )
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "partial.log"),
        "partial-worker",
        "partial",
        timeout_s=0.05,
    )
    client.restart_warmup_s = 0.0
    try:
        old_pid = client.proc.pid
        assert client.submit(1, "partial", Image.new("RGB", (1, 1))) is True
        assert _wait_for(lambda: bool(client.poll_results()), timeout=3.0)
        assert _wait_for(lambda: client.proc.pid != old_pid, timeout=3.0)
    finally:
        client.close()
    assert not client.thread.is_alive()


def test_close_unblocks_an_in_progress_worker_write(tmp_path: Path) -> None:
    client = app.LiveWorkerClient(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        1024,
        1024,
        str(tmp_path / "closing.log"),
        "closing-worker",
        "closing",
        timeout_s=60.0,
    )
    client.restart_warmup_s = 0.0
    assert client.submit(1, "large", Image.new("RGB", (1024, 1024))) is True
    assert _wait_for(client.busy)

    started = time.monotonic()
    client.close()

    assert time.monotonic() - started < 2.5
    assert not client.thread.is_alive()


def test_close_allows_worker_to_exit_cleanly_on_stdin_eof(tmp_path: Path) -> None:
    client = app.LiveWorkerClient(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        1,
        1,
        str(tmp_path / "eof.log"),
        "eof-worker",
        "eof",
        timeout_s=1.0,
    )
    time.sleep(0.05)
    client.close()
    assert client.proc.returncode == 0
    assert not client.thread.is_alive()


def test_attached_frame_shm_survives_worker_death(tmp_path):
    """A killed worker must not take the operator's frame buffer with it.

    CPython's resource_tracker unlinks any SharedMemory the process touched, so a
    worker that merely attached used to destroy the operator-owned segment on its
    way out. Every replacement worker then failed to attach and died, and
    localization stopped completely.
    """
    import subprocess
    import sys
    import time
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        child = (
            "import sys, time\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "from live_localizer_worker import attach_frame_shm\n"
            "attach_frame_shm(sys.argv[1])\n"
            "print('attached', flush=True)\n"
            "time.sleep(30)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", child, shm.name], stdout=subprocess.PIPE)
        try:
            assert proc.stdout.readline().strip() == b"attached"
            proc.terminate()
            proc.wait(timeout=10)
            time.sleep(0.5)
            survivor = shared_memory.SharedMemory(name=shm.name, create=False)
            survivor.close()
        finally:
            if proc.poll() is None:
                proc.kill()
            if proc.stdout is not None:
                proc.stdout.close()
    finally:
        shm.close()
        shm.unlink()

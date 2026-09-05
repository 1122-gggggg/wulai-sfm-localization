from __future__ import annotations

import json
import importlib.util
import inspect
import io
import math
import os
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
import operator_tick
from localization_contract import InvalidLocalizationResult, LocalizationResult
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
        "定位端到端 FPS N/A | 影像串流 FPS N/A | 端到端延遲 N/A | p95（近 5 秒）N/A | 影格年齡 N/A"
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


def test_delayed_success_keeps_source_pose_age_instead_of_ui_arrival(monkeypatch) -> None:
    monkeypatch.setattr(app.time, "monotonic", lambda: 100.0)
    operator = SimpleNamespace(
        live_result_times=[],
        loc_fps=0.0,
        loc_latency_ms=None,
        loc_wall_ms=None,
        loc_e2e_ms=None,
        _loc_e2e_ms_samples=[],
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
            "reproj_rms": 0.1,
            "source_frame_stamp_mono": 99.0,
            "capture_mono_ns": 99_000_000_000,
            "pose_mono_ns": 99_500_000_000,
        },
    )

    assert operator.loc_pose_updated_mono == 99.0


def test_autonomy_pose_rejects_source_pose_older_than_half_second(monkeypatch) -> None:
    monkeypatch.setattr(app.time, "monotonic", lambda: 100.0)
    operator = SimpleNamespace(
        live_locked=True,
        loc_health="OK",
        loc_pose_updated_mono=99.0,
        live_pose=np.array([1.0, 2.0, 3.0, 0.0], dtype=float),
    )

    assert app.OperatorApp._autonomy_pose(operator) is None


def test_autonomy_pose_uses_raw_camera_center_and_6dof_heading(monkeypatch) -> None:
    monkeypatch.setattr(app.time, "monotonic", lambda: 100.0)

    operator = SimpleNamespace(
        live_locked=True,
        loc_health="OK",
        loc_pose_updated_mono=99.8,
        live_pose=np.array([1.0, 2.0, 3.0, np.nan], dtype=float),
        camera_forward_world=np.array([0.0, 0.0, 1.0]),
        _autonomy_map_frame=app.LEGACY_MAP_FRAME,
        backend=SimpleNamespace(is_live=True),
    )

    pose = app.OperatorApp._autonomy_pose(operator)

    assert pose is not None
    assert (pose.x, pose.y, pose.z, pose.yaw, pose.stamp) == pytest.approx(
        (1.0, 2.0, 3.0, math.pi / 2.0, 99.8)
    )


def test_autonomy_pose_rejects_camera_looking_nearly_vertical(monkeypatch) -> None:
    monkeypatch.setattr(app.time, "monotonic", lambda: 100.0)
    operator = SimpleNamespace(
        live_locked=True,
        loc_health="OK",
        loc_pose_updated_mono=99.8,
        live_pose=np.array([1.0, 2.0, 3.0, 0.0], dtype=float),
        camera_forward_world=np.array([0.0, -1.0, 0.01]),
        _autonomy_map_frame=app.LEGACY_MAP_FRAME,
        backend=SimpleNamespace(is_live=True),
    )

    assert app.OperatorApp._autonomy_pose(operator) is None


def test_autonomy_pose_requires_same_frame_orientation_even_when_not_live(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app.time, "monotonic", lambda: 100.0)
    for is_live in (True, False):
        operator = SimpleNamespace(
            live_locked=True,
            loc_health="OK",
            loc_pose_updated_mono=99.8,
            live_pose=np.array([1.0, 2.0, 3.0, 0.0], dtype=float),
            camera_forward_world=None,
            _autonomy_map_frame=app.LEGACY_MAP_FRAME,
            backend=SimpleNamespace(is_live=is_live),
        )

        assert app.OperatorApp._autonomy_pose(operator) is None


def test_live_result_without_orientation_clears_stale_camera_orientation(
    monkeypatch,
) -> None:
    operator = SimpleNamespace(
        camera_forward_world=np.array([0.0, 0.0, 1.0]),
        camera_axes_world=np.eye(3),
        live_last_xyz=np.zeros(3),
        _integrated_auto_map_frame=None,
        live_heading=0.5,
        live_pose=np.zeros(4),
    )

    app.OperatorApp._update_live_camera_orientation(
        operator, {"pose": [0.0, 0.0, 0.0]}, np.zeros(3)
    )

    assert operator.camera_forward_world is None
    assert operator.camera_axes_world is None


def test_autonomy_pose_rejects_invalid_current_orientation_then_recovers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app.time, "monotonic", lambda: 100.0)
    operator = SimpleNamespace(
        live_locked=True,
        loc_health="OK",
        loc_pose_updated_mono=99.8,
        live_pose=np.array([1.0, 2.0, 3.0, 0.0], dtype=float),
        live_last_xyz=np.array([1.0, 2.0, 3.0], dtype=float),
        camera_forward_world=np.array([0.0, 0.0, 1.0]),
        camera_axes_world=np.eye(3),
        _integrated_auto_map_frame=app.LEGACY_MAP_FRAME,
        _autonomy_map_frame=app.LEGACY_MAP_FRAME,
        live_heading=0.5,
    )

    for result in (
        {},
        {"camera_forward_world": [0.0, 0.0, 0.0]},
        {"camera_forward_world": [0.0, float("nan"), 1.0]},
    ):
        app.OperatorApp._update_live_camera_orientation(
            operator,
            result,
            np.array([1.1, 2.0, 3.0], dtype=float),
        )
        assert app.OperatorApp._autonomy_pose(operator) is None

    new_forward = np.array([1.0, 0.0, 0.0], dtype=float)
    app.OperatorApp._update_live_camera_orientation(
        operator,
        {"camera_forward_world": new_forward.tolist()},
        np.array([1.1, 2.0, 3.0], dtype=float),
    )

    pose = app.OperatorApp._autonomy_pose(operator)
    assert pose is not None
    expected_heading = app.camera_heading_from_forward(
        new_forward,
        app.LEGACY_MAP_FRAME,
    )
    assert expected_heading is not None
    assert pose.yaw == pytest.approx(expected_heading)


def test_state_from_replay_clears_camera_orientation_when_row_lacks_it(
    monkeypatch,
) -> None:
    operator = SimpleNamespace(
        replay_rows=[
            {
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                "success": True,
            }
        ],
        replay_headings=[0.0],
        replay_index=0,
        replay_last_pose=np.zeros(4),
        camera_forward_world=np.array([0.0, 0.0, 1.0]),
        camera_axes_world=np.eye(3),
        video_frame_fresh=True,
        boot_holding=lambda: False,
    )
    base = SimpleNamespace(
        stream="OK",
        mode="",
        loc="",
        tracker_state="",
        inliers=0,
        reproj=None,
        pose=np.zeros(4),
    )

    app.OperatorApp.state_from_replay(operator, base)

    assert operator.camera_forward_world is None
    assert operator.camera_axes_world is None


def test_guided_site_alignment_is_removed_from_the_operator_ui() -> None:
    source = inspect.getsource(app.OperatorApp._build_ui)

    assert "現場座標校正" not in source
    assert not hasattr(app.OperatorApp, "open_guided_site_alignment")


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


def test_worker_readline_preserves_second_line_from_one_read() -> None:
    read_fd, write_fd = os.pipe()
    writer = os.fdopen(write_fd, "wb", buffering=0)
    writer.write(b'{"seq":1}\n{"seq":2}\n')
    writer.close()
    stdout = os.fdopen(read_fd, "rb", buffering=0)
    client = app.LiveWorkerClient.__new__(app.LiveWorkerClient)
    client.label = "tail-worker"
    client.timeout_s = 0.2
    client.MAX_RESPONSE_BYTES = 1024
    client._closed = threading.Event()
    client._stdout_buffer = bytearray()
    proc = SimpleNamespace(stdout=stdout, poll=lambda: None)
    try:
        assert client._readline_with_timeout(proc) == b'{"seq":1}\n'
        assert client._readline_with_timeout(proc) == b'{"seq":2}\n'
    finally:
        stdout.close()


@pytest.mark.parametrize("payload", [{}, {"seq": True}, {"seq": 1.0}, {"seq": "1"}, {"seq": 2}])
def test_worker_sequence_rejects_missing_or_non_exact_sequence(payload) -> None:
    client = app.LiveWorkerClient.__new__(app.LiveWorkerClient)
    client.label = "sequence-worker"

    with pytest.raises(app._WorkerResponseDesync, match="desync"):
        client._validate_worker_sequence(payload, 1)


def test_worker_sequence_accepts_only_matching_builtin_int() -> None:
    client = app.LiveWorkerClient.__new__(app.LiveWorkerClient)
    client.label = "sequence-worker"

    client._validate_worker_sequence({"seq": 1}, 1)


def test_localization_exception_payload_is_explicitly_lost() -> None:
    payload = localizer_worker.localization_exception_payload(
        seq=4, error=RuntimeError("tracker failed")
    )

    assert payload["success"] is False
    assert payload["mode"] == "LOST"
    assert payload["next_mode"] == "LOST"
    assert payload["localization_exception"] is True


def test_localization_result_boundary_validates_and_preserves_dict_payload() -> None:
    payload = {
        "localization_contract_version": 1,
        "seq": 4,
        "frame_id": "worker-4",
        "capture_mono_ns": 10_000,
        "pose_mono_ns": 20_000,
        "validity": True,
        "confidence": 0.75,
        "success": True,
        "pose": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw_raw": 0.5},
    }

    result = LocalizationResult.from_payload(payload)

    assert result.seq == 4
    assert result.frame_id == "worker-4"
    assert result.pose == (1.0, 2.0, 3.0, 0.5)
    assert result.validity is True
    assert result.to_payload()["pose"] == payload["pose"]

    with pytest.raises(InvalidLocalizationResult, match="pose"):
        LocalizationResult.from_payload(
            {
                **payload,
                "pose": {"x": 1.0, "y": float("nan"), "z": 3.0},
            }
        )


def test_localization_result_boundary_rejects_future_timestamps() -> None:
    payload = {
        "localization_contract_version": 1,
        "seq": 4,
        "frame_id": "worker-4",
        "capture_mono_ns": 1_001,
        "pose_mono_ns": 1_002,
        "validity": False,
        "confidence": 0.0,
        "success": False,
    }

    with pytest.raises(InvalidLocalizationResult, match="future"):
        LocalizationResult.from_payload(payload, now_mono_ns=1_000)


def test_worker_exception_payload_carries_localization_contract_fields() -> None:
    payload = localizer_worker.localization_exception_payload(
        seq=4, error=RuntimeError("tracker failed")
    )

    result = LocalizationResult.from_payload(payload)
    assert result.validity is False
    assert result.confidence == 0.0
    assert result.pose is None


def test_edm_cuda_oom_is_structured_fatal_and_cleans_cache() -> None:
    class FakeCudaOOM(RuntimeError):
        pass

    cleanup_calls = []
    fake_torch = SimpleNamespace(
        OutOfMemoryError=FakeCudaOOM,
        cuda=SimpleNamespace(
            OutOfMemoryError=FakeCudaOOM,
            empty_cache=lambda: cleanup_calls.append("empty_cache"),
            ipc_collect=lambda: cleanup_calls.append("ipc_collect"),
        ),
    )
    error = FakeCudaOOM("CUDA out of memory")

    assert localizer_worker.edm_cuda_oom_requires_restart("edm", error, fake_torch)
    assert not localizer_worker.edm_cuda_oom_requires_restart("xfeat", error, fake_torch)

    payload = localizer_worker.localization_exception_payload(seq=4, error=error)
    payload.update(localizer_worker.cuda_oom_failure_fields(error))
    payload["cuda_cache_cleanup"] = localizer_worker.cleanup_cuda_cache(fake_torch)

    assert payload["success"] is False
    assert payload["validity"] is False
    assert payload["error"] == "cuda_oom"
    assert payload["failure_kind"] == "cuda_oom"
    assert payload["worker_fatal"] is True
    assert payload["restart_required"] is True
    assert payload["cuda_cache_cleanup"] == {
        "empty_cache": "ok",
        "ipc_collect": "ok",
    }
    assert cleanup_calls == ["empty_cache", "ipc_collect"]


def test_localization_exception_result_triggers_live_fail_safe() -> None:
    payload = localizer_worker.localization_exception_payload(
        seq=5, error=RuntimeError("tracker failed")
    )
    reasons = []
    operator = SimpleNamespace(
        localizer=SimpleNamespace(poll_results=lambda: [payload]),
        _loc_benchmark_pending=None,
        loc_benchmark_active="auto",
        _apply_lost_hold_result=lambda _result: None,
        update_localization_metrics=lambda _result: None,
        _is_live_backend=lambda: True,
        _engage_real_localization_recovery=lambda reason: reasons.append(reason),
        _last_status_write=float("inf"),
        _last_loc_fail_log=float("inf"),
        _loc_ok_count=0,
        _loc_fail_count=0,
        loc_fps=0.0,
        live_result=None,
        live_result_frame_name="",
        write_log=lambda _message: None,
    )

    app.OperatorApp.update_live_results(operator)
    assert reasons == [app.FailureReason.LOCALIZATION_LOST]


def test_live_results_drop_publication_inversions_before_autonomy() -> None:
    metrics = []
    recovery_calls = []
    rows = [
        {
            "seq": 2,
            "display_seq": 20,
            "frame_id": "frame-20",
            "frame_name": "frame-20",
            "success": False,
            "mode": "LOST",
            "next_mode": "LOST",
            "localization_exception": True,
            "error": "worker restarting",
        },
        {
            "seq": 1,
            "display_seq": 10,
            "frame_id": "frame-10",
            "frame_name": "frame-10",
            "success": True,
            "pose": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw_raw": 0.1},
            "inliers": app.LOC_LOW_INLIERS,
            "mode": "TRACK",
            "next_mode": "TRACK",
        },
        {
            "seq": 3,
            "display_seq": 30,
            "frame_id": "frame-30",
            "frame_name": "frame-30",
            "success": True,
            "pose": {"x": 4.0, "y": 5.0, "z": 6.0, "yaw_raw": 0.2},
            "inliers": app.LOC_LOW_INLIERS,
            "mode": "TRACK",
            "next_mode": "TRACK",
        },
    ]
    operator = SimpleNamespace(
        localizer=SimpleNamespace(poll_results=lambda: rows),
        _loc_benchmark_pending=None,
        loc_benchmark_active="auto",
        _last_localization_exception_seq=None,
        _last_applied_live_result_display_seq=None,
        _apply_lost_hold_result=lambda _result: None,
        update_localization_metrics=lambda result: metrics.append(result),
        _is_live_backend=lambda: True,
        _engage_real_localization_recovery=lambda reason: recovery_calls.append(reason),
        _last_status_write=float("inf"),
        _last_loc_fail_log=float("inf"),
        _loc_ok_count=0,
        _loc_fail_count=0,
        loc_fps=0.0,
        live_result=None,
        live_result_frame_name="",
        live_last_xyz=None,
        _live_pending=None,
        camera_forward_world=None,
        camera_axes_world=None,
        _integrated_auto_map_frame=None,
        live_heading=None,
        live_pose=np.zeros(4, dtype=float),
        live_locked=False,
        live_new_pose=False,
        boot_holding=lambda: False,
        pose_stabilizer=None,
        write_log=lambda _message: None,
    )

    app.OperatorApp.update_live_results(operator)

    assert [result["display_seq"] for result in metrics] == [20, 30]
    assert operator.live_result["display_seq"] == 30
    assert operator.live_result_frame_name == "frame-30"
    assert np.allclose(operator.live_pose[:3], [4.0, 5.0, 6.0])
    assert recovery_calls == [app.FailureReason.LOCALIZATION_LOST]
    assert operator._last_applied_live_result_display_seq == 30


def test_held_frame_retry_with_same_display_seq_is_not_dropped() -> None:
    operator = SimpleNamespace(_last_applied_live_result_display_seq=42)

    assert app.OperatorApp._live_result_is_new(operator, {"display_seq": 42, "hold_retry": True})
    assert not app.OperatorApp._live_result_is_new(
        operator, {"display_seq": 42, "hold_retry": False}
    )
    assert not app.OperatorApp._live_result_is_new(
        operator, {"display_seq": 41, "hold_retry": True}
    )
    assert operator._last_applied_live_result_display_seq == 42


def test_low_confidence_recovery_holds_pose_until_trustworthy_fix() -> None:
    batches = [
        [
            {
                "seq": 1,
                "display_seq": 1,
                "frame_id": "frame-low",
                "frame_name": "frame-low",
                "success": True,
                "pose": {"x": 9.0, "y": 9.0, "z": 9.0, "yaw_raw": 0.1},
                "inliers": 20,
                "reproj_rms": 2.0,
                "mode": "WEAK_TRACK",
                "next_mode": "WEAK_TRACK",
            }
        ],
        [
            {
                "seq": 2,
                "display_seq": 2,
                "frame_id": "frame-fix",
                "frame_name": "frame-fix",
                "success": True,
                "pose": {"x": 1.2, "y": 2.1, "z": 3.0, "yaw_raw": 0.1},
                "inliers": 80,
                "reproj_rms": 2.0,
                "mode": "LOST",
                "next_mode": "TRACK",
                "relocalize_requested": True,
            }
        ],
    ]
    recovery_calls = []
    operator = SimpleNamespace(
        localizer=SimpleNamespace(poll_results=lambda: batches.pop(0)),
        lost_hold=app.LostHoldPolicy(
            max_attempts=3,
            timeout_s=0.0,
            low_confidence_results=1,
            hold_on_low_confidence=True,
        ),
        inspecting=True,
        video_display_index=11,
        video_display_frame_name="frame-low",
        _loc_benchmark_pending=None,
        loc_benchmark_active="auto",
        _last_localization_exception_seq=None,
        _last_applied_live_result_display_seq=None,
        _apply_lost_hold_result=lambda result: app.OperatorApp._apply_lost_hold_result(
            operator, result
        ),
        update_localization_metrics=lambda _result: None,
        _is_live_backend=lambda: False,
        _integrated_auto_active=lambda: False,
        _engage_real_localization_recovery=lambda reason: recovery_calls.append(reason),
        _last_status_write=float("inf"),
        _last_loc_fail_log=float("inf"),
        _loc_ok_count=0,
        _loc_fail_count=0,
        loc_fps=0.0,
        live_result=None,
        live_result_frame_name="",
        live_last_xyz=np.array([1.0, 2.0, 3.0], dtype=float),
        _live_pending=None,
        camera_forward_world=None,
        camera_axes_world=None,
        _integrated_auto_map_frame=None,
        live_heading=0.0,
        live_pose=np.array([1.0, 2.0, 3.0, 0.0], dtype=float),
        live_locked=True,
        live_new_pose=False,
        boot_holding=lambda: False,
        pose_stabilizer=None,
        yaw_stabilizer=None,
        write_log=lambda _message: None,
    )

    app.OperatorApp.update_live_results(operator)

    assert operator.lost_hold.active
    assert operator.live_result["confidence_hold_event"] == "ENGAGE_LOW_CONF"
    assert np.allclose(operator.live_pose[:3], [1.0, 2.0, 3.0])
    assert np.allclose(operator.live_last_xyz, [1.0, 2.0, 3.0])
    assert recovery_calls == [app.FailureReason.LOCALIZATION_WEAK]

    app.OperatorApp.update_live_results(operator)

    assert not operator.lost_hold.active
    assert operator.live_result["confidence_hold_event"] == "RELEASE_FIX"
    assert np.allclose(operator.live_pose[:3], [1.2, 2.1, 3.0])


def test_install_site_runtime_allows_localization_exception_seq_zero(monkeypatch) -> None:
    profile = SimpleNamespace(
        site_id="new-site",
        source=Path("/tmp/new-site.yaml"),
        display_name="New site",
    )
    localizer = SimpleNamespace(benchmark_mode="auto")
    prepared = app.PreparedSiteRuntime(
        args=SimpleNamespace(),
        interface_mode="sim",
        profile=profile,
        hardware_approval=None,
        map_points=np.empty((0, 3), dtype=float),
        route_points=[],
        mission_route_snapshot=None,
        replay_rows=[],
    )
    runtime = app.ActiveSiteRuntime(
        prepared=prepared,
        session_logs=SimpleNamespace(),
        backend=SimpleNamespace(state=SimpleNamespace(), is_live=False),
        video_stream=SimpleNamespace(output_fps=30.0),
        live_backend=None,
        localizer=localizer,
        detector=None,
        lost_hold=None,
    )
    operator = app.OperatorApp.__new__(app.OperatorApp)
    operator._loc_metrics_f = None
    operator.site_asset_actions = SimpleNamespace(current_profile=None)
    operator.history = []
    operator.history_health = []
    operator.no_loc_markers = []
    operator.write_log = lambda _message: None
    operator.destroy = lambda: None
    operator.boot_lock_s = 0.0
    operator._last_localization_exception_seq = 0
    operator._last_applied_live_result_display_seq = 99
    operator._make_command_coordinator = lambda: object()
    operator._map_view_bounds = lambda _points: (np.zeros(2), 1.0)
    operator._set_preflight_visible = lambda _visible: None
    operator._reset_localization_benchmark_metrics = lambda: None
    operator._derive_motion_headings = lambda _rows: []
    operator._sync_autonomy_profile_approval = lambda: None
    operator.title = lambda _title: None
    operator._attach_localizer_file_handler = lambda: None
    operator._sync_record_status_label = lambda: None
    operator.site_assets_panel = SimpleNamespace(activate_site=lambda *_args: None)
    monkeypatch.setattr(
        app,
        "OperatorShutdownCoordinator",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        app,
        "site_pack_root_for_profile",
        lambda *_args: Path("/tmp/site-pack"),
    )

    app.OperatorApp._install_site_runtime(operator, runtime)

    recovery_calls = []
    operator.localizer.poll_results = lambda: [
        {
            "seq": 0,
            "display_seq": 0,
            "frame_id": "frame-0",
            "frame_name": "frame-0",
            "success": False,
            "mode": "LOST",
            "next_mode": "LOST",
            "localization_exception": True,
            "error": "new worker failure",
        }
    ]
    operator._loc_benchmark_pending = None
    operator.loc_benchmark_active = "auto"
    operator.loc_benchmark_mode_var = SimpleNamespace(set=lambda _value: None)
    operator._apply_lost_hold_result = lambda _result: None
    operator.update_localization_metrics = lambda _result: None
    operator._is_live_backend = lambda: True
    operator._engage_real_localization_recovery = lambda reason: recovery_calls.append(reason)
    operator._last_status_write = float("inf")
    operator._last_loc_fail_log = float("inf")
    operator._loc_ok_count = 0
    operator._loc_fail_count = 0
    operator.loc_fps = 0.0
    operator.live_result = None
    operator.live_result_frame_name = ""
    operator.live_last_xyz = None
    operator._live_pending = None
    operator.camera_forward_world = None
    operator.camera_axes_world = None
    operator._integrated_auto_map_frame = None
    operator.live_heading = None
    operator.live_pose = np.zeros(4, dtype=float)
    operator.live_locked = False
    operator.live_new_pose = False
    operator.boot_holding = lambda: False
    operator.pose_stabilizer = None

    assert operator._last_localization_exception_seq is None
    operator.inspecting = True
    app.OperatorApp.update_live_results(operator)
    assert recovery_calls == [app.FailureReason.LOCALIZATION_LOST]
    assert operator._last_localization_exception_seq == 0


def test_heading_arrow_polygon_points_at_the_current_heading() -> None:
    arrow = app.heading_arrow_polygon(100.0, 80.0, 120.0, 80.0)

    assert arrow is not None
    tip, *body = arrow
    assert tip == (118.0, 80.0)
    assert all(point[0] < tip[0] for point in body)


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
    requested: str,
    legacy_live: bool,
    expected: str,
    is_live: bool,
) -> None:
    mode, resolved_live = app.resolve_operator_interface(requested, legacy_live, "")

    assert mode == expected
    assert resolved_live is is_live


def test_real_flight_interface_rejects_video_file_fallback() -> None:
    with pytest.raises(ValueError, match="accepts only the ANAFI PDRAW stream"):
        app.resolve_operator_interface("real-flight", False, "/tmp/replay.mp4")


def test_simulated_interface_rejects_legacy_live_flag() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        app.resolve_operator_interface("simulated-stream", True, "")


def test_auto_inspect_while_manual_airborne_preserves_control_without_command() -> None:
    class FakeOperator:
        _begin_inspection_feed = app.OperatorApp._begin_inspection_feed
        begin_auto_inspect = app.OperatorApp.begin_auto_inspect
        _preflight_blocks_flight_command = app.OperatorApp._preflight_blocks_flight_command
        _is_live_backend = app.OperatorApp._is_live_backend
        send = app.OperatorApp.send

        def _resume_integrated_auto(self):
            return False

        def _select_preflight_tab(self, _step):
            return None

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
            self.preflight_guide = SimpleNamespace(complete=True, current_step=None)
            self.backend = SimpleNamespace(
                is_live=True,
                pilot_sticks=True,
                state=SimpleNamespace(flight_state="hovering"),
                command=lambda name, **payload: self.backend_calls.append((name, payload)),
            )
            self._nudge_keys_held = set()
            self.stream_lost_since = None

        def write_log(self, text: str) -> None:
            self.logs.append(text)

    operator = FakeOperator()
    operator.begin_auto_inspect()
    assert operator.inspecting
    assert operator.backend.pilot_sticks is True
    assert operator.backend_calls == []

    # AUTO without a validated, selected route now fails closed before dispatch.
    manual = FakeOperator()
    manual.send("start_auto")
    assert manual.backend_calls == []
    assert any("沒有路線鎖" in line for line in manual.logs)


def test_localization_button_toggles_start_and_cancel_without_flight_command() -> None:
    class Variable:
        def __init__(self):
            self.text = "開始定位"

        def configure(self, *, text):
            self.text = text

    class LostHold:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class FakeOperator:
        _sync_localization_button = app.OperatorApp._sync_localization_button
        _begin_inspection_feed = app.OperatorApp._begin_inspection_feed
        cancel_localization = app.OperatorApp.cancel_localization
        toggle_localization = app.OperatorApp.toggle_localization
        begin_auto_inspect = app.OperatorApp.begin_auto_inspect

        def __init__(self):
            self.inspecting = False
            self.start_localization_button = Variable()
            self.inspect_start = None
            self.processed_frames = 0
            self.overall_fps = 0.0
            self.stream_fps_instant = 0.0
            self._stream_frame_times = []
            self._lifetime_stream_fps = 0.0
            self.loc_wall_ms = None
            self.loc_e2e_ms = None
            self._loc_e2e_ms_samples = []
            self.loc_pose_updated_mono = None
            self._loc_consecutive_good_fixes = 0
            self.loc_hold_engage_count = 0
            self.loc_recovery_fix_count = 0
            self.loc_recovery_text = ""
            self.next_stream_frame_time = 0.0
            self.live_locked = True
            self.live_new_pose = True
            self.live_result = {"success": True}
            self.live_result_frame_name = "frame"
            self.live_pending_frame_name = "frame"
            self.detection_result = {"success": True}
            self.detection_result_frame_name = "frame"
            self.detection_pending_frame_name = "frame"
            self._live_pending = object()
            self.last_submitted_index = 7
            self.last_detect_submitted_index = 7
            self.video_display_index = 8
            self._last_applied_live_result_display_seq = 7
            self.boot_lock_start = 1.0
            self.boot_lock_s = 20.0
            self.boot_lock_done = True
            self.lost_hold = LostHold()
            self.logs = []
            self.backend_calls = []

        def write_log(self, text):
            self.logs.append(text)

    operator = FakeOperator()

    operator.toggle_localization()
    assert operator.inspecting is True
    assert operator.start_localization_button.text == "取消定位"

    operator.toggle_localization()
    assert operator.inspecting is False
    assert operator.start_localization_button.text == "開始定位"
    assert operator.live_locked is False
    assert operator.live_result is None
    assert operator.detection_result is None
    assert operator._last_applied_live_result_display_seq == 8
    assert operator.lost_hold.reset_calls == 1
    assert operator.backend_calls == []
    assert "未發送飛控指令" in operator.logs[-1]


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


def test_timed_localizer_header_rejects_future_capture_timestamp() -> None:
    with pytest.raises(ValueError, match="future"):
        localizer_protocol.encode_request("auto", time.monotonic() + 1.0)


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


def test_live_localizer_sends_imu_even_when_pose_guided_search_is_off() -> None:
    client = object.__new__(app.LiveLocalizerClient)
    client._runtime_benchmark_control = True
    client._benchmark_mode_lock = threading.Lock()
    client._benchmark_mode = "auto"
    client._relocalize_once = False
    header = client._request_prefix(
        {
            "source_frame_stamp_mono": 10.0,
            "fused_telemetry_mono": 10.05,
            "fused_roll": 0.1,
            "fused_pitch": -0.2,
            "fused_yaw": 1.3,
            "fused_speed_north": 0.4,
            "fused_speed_east": -0.1,
            "fused_speed_down": 0.0,
        }
    )
    mode, stamp, fused = localizer_protocol.decode_control_header(header)
    assert mode == "auto"
    assert stamp == pytest.approx(10.0)
    assert fused is not None
    assert fused.stamp == pytest.approx(10.05)
    assert fused.yaw == pytest.approx(1.3)
    assert fused.speed_north == pytest.approx(0.4)


def test_live_localizer_sends_gnss_with_its_independent_stamp() -> None:
    client = object.__new__(app.LiveLocalizerClient)
    client._runtime_benchmark_control = True
    client._benchmark_mode_lock = threading.Lock()
    client._benchmark_mode = "auto"
    client._relocalize_once = False

    header = client._request_prefix(
        {
            "source_frame_stamp_mono": 10.0,
            "fused_telemetry_mono": 10.05,
            "fused_roll": 0.1,
            "fused_pitch": -0.2,
            "fused_yaw": 1.3,
            "fused_gps_mono": 9.5,
            "fused_gps_latitude": 25.033,
            "fused_gps_longitude": 121.5654,
            "fused_gps_altitude": 18.2,
            "fused_gps_latitude_accuracy": 0.8,
            "fused_gps_longitude_accuracy": 0.9,
            "fused_gps_altitude_accuracy": 1.4,
        }
    )

    assert header.startswith(localizer_protocol.GNSS_FUSED_MAGIC)
    _mode, _stamp, fused = localizer_protocol.decode_control_header(header)
    assert fused is not None and fused.has_gnss
    assert fused.gps_stamp == pytest.approx(9.5)
    assert fused.latitude == pytest.approx(25.033)


@pytest.mark.parametrize("enabled", [False, True])
def test_live_localizer_client_passes_neuflow_flag_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
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
    assert [float(value) for value in cmd[start : start + 4]] == list(camera.params)


def test_live_localizer_client_passes_measured_map_alignment(monkeypatch) -> None:
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
        map_align=Path("T_align_gravity.json"),
    )

    index = captured["cmd"].index("--map-align")
    assert captured["cmd"][index + 1] == "T_align_gravity.json"


def test_live_localizer_client_passes_reference_index_manifest(monkeypatch) -> None:
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
        reference_index=Path("reference-index/SHA256SUMS.json"),
        reference_index_sha256="a" * 64,
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--reference-index") + 1] == ("reference-index/SHA256SUMS.json")
    assert cmd[cmd.index("--reference-index-sha256") + 1] == "a" * 64


def test_live_localizer_client_passes_tensorrt_megaloc_contract(monkeypatch) -> None:
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
        megaloc_backend="tensorrt",
        megaloc_engine=Path("megaloc.engine"),
        megaloc_engine_sha256="b" * 64,
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--megaloc-backend") + 1] == "tensorrt"
    assert cmd[cmd.index("--megaloc-engine") + 1] == "megaloc.engine"
    assert cmd[cmd.index("--megaloc-engine-sha256") + 1] == "b" * 64


def test_shipped_edm_production_profile_pins_validated_parameters() -> None:
    profile_path = (
        Path(__file__).resolve().parents[3]
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
    assert profile["tracker"]["lost_local_grace_frames"] == 2
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
        cfg,
        matcher_mode="",
        local_topk=4,
    )

    assert cfg.local_topk == 4
    assert cfg.weak_local_topk == 8
    assert cfg.adaptive_first_topk == 3


def test_live_localizer_switches_mode_without_restarting_worker(tmp_path: Path) -> None:
    worker = (
        "import json,sys; "
        "[(lambda h,r,seq: print(json.dumps({'seq':seq,'success':False,'header':h.hex()}),flush=True))"
        "(sys.stdin.buffer.read(5),sys.stdin.buffer.read(3),seq) for seq in range(2)]"
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
        "print(json.dumps({'seq':0,'success':bool(raw)}),flush=True)"
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
            self.backend = SimpleNamespace(
                command=lambda *_a, **_k: pytest.fail(
                    "localization benchmark mode touched the drone backend"
                )
            )
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
    assert not localizer_worker.apply_runtime_benchmark_mode(tracker, "auto", "auto", seed)
    assert tracker.state is untouched

    assert not localizer_worker.apply_runtime_benchmark_mode(tracker, "global", "auto", seed)
    assert tracker.state.mode == "LOST"
    assert tracker.state.last_center is None

    assert localizer_worker.apply_runtime_benchmark_mode(tracker, "weak", "global", seed)
    assert tracker.state.mode == "WEAK_TRACK"
    assert tracker.state.last_refs == [2]

    assert localizer_worker.apply_runtime_benchmark_mode(tracker, "track", "weak", seed)
    assert tracker.state.mode == "TRACK"
    assert tracker.state.last_refs == [2]

    assert not localizer_worker.apply_runtime_benchmark_mode(tracker, "auto", "track", seed)
    assert tracker.state.mode == "BOOT_INIT"
    assert tracker.state.last_center is None

    with pytest.raises(ValueError, match="unsupported"):
        localizer_worker.apply_runtime_benchmark_mode(tracker, "takeoff", "auto", seed)


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


def test_adaptive_localization_submit_tracks_half_the_latest_latency() -> None:
    operator = SimpleNamespace(adaptive_loc_submit=True, loc_latency_ms=80.0)
    assert app.OperatorApp._localization_coalesce_interval_s(operator) == pytest.approx(0.04)
    operator.loc_latency_ms = 5.0
    assert app.OperatorApp._localization_coalesce_interval_s(operator) == pytest.approx(0.02)
    operator.loc_latency_ms = 500.0
    assert app.OperatorApp._localization_coalesce_interval_s(operator) == pytest.approx(0.1)
    operator.adaptive_loc_submit = False
    assert app.OperatorApp._localization_coalesce_interval_s(operator) == pytest.approx(0.02)


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


def test_localization_result_poll_logs_failure_and_reschedules_in_finally() -> None:
    logs = []
    scheduled = []
    incidents = []
    pauses = []

    class FailingLocalizer:
        def drain_result_notifications(self) -> None:
            raise RuntimeError("result pipe broke")

    class FakeOperator:
        _loc_result_poll_ms = 5
        localizer = FailingLocalizer()
        session_logs = SimpleNamespace(
            incident=lambda event, **fields: incidents.append((event, fields))
        )
        poll_localization_results = app.OperatorApp.poll_localization_results

        def update_live_results(self) -> None:
            raise AssertionError("update should not run after drain failure")

        def write_log(self, message: str) -> None:
            logs.append(message)

        def _pause_integrated_auto(self, reason: str) -> None:
            pauses.append(reason)

        def after(self, delay, callback) -> None:
            scheduled.append((delay, callback))

    operator = FakeOperator()

    operator.poll_localization_results()

    assert logs == ["LOCALIZATION_RESULT_POLL_FAILED: RuntimeError('result pipe broke')"]
    assert pauses == ["localization_poll_failed"]
    assert incidents == [
        (
            "localization_poll_failed",
            {"error": "RuntimeError('result pipe broke')", "resolved": False},
        )
    ]
    assert scheduled == [(5, operator.poll_localization_results)]


def test_tick_stage_failure_is_logged_paused_and_rescheduled_once(monkeypatch) -> None:
    state = SimpleNamespace()
    scheduled = []
    logs = []
    incidents = []
    pauses = []

    class FakeOperator:
        backend = SimpleNamespace(poll=lambda: state)
        session_logs = SimpleNamespace(
            incident=lambda event, **fields: incidents.append((event, fields))
        )
        _next_tick_deadline = 0.0
        _tick_period_s = 0.01
        _stick_vector_active = False
        tick = object()

        def _drain_flight_command_results(self):
            return None

        def _drain_integrated_autonomy_events(self):
            return None

        def _active_nudge_directions(self):
            return []

        def _is_live_backend(self):
            return False

        def _gravity_tick(self, _state):
            return None

        def update_boot_lock(self):
            return None

        def update_live_results(self):
            return None

        def update_lost_hold(self):
            return None

        def update_detection_results(self):
            return None

        def _pause_integrated_auto(self, reason):
            pauses.append(reason)

        def write_log(self, message):
            logs.append(message)

        def after(self, delay, callback):
            scheduled.append((delay, callback))

    def fail_update_stream(*_args, **_kwargs):
        raise RuntimeError("synthetic stream-stage failure")

    monkeypatch.setattr(operator_tick, "_update_stream", fail_update_stream)

    with pytest.raises(RuntimeError, match="synthetic stream-stage failure"):
        operator_tick.run_tick(
            FakeOperator(),
            rolling_event_fps=lambda values, now: (values, now),
            next_tick_deadline=lambda _previous, _now, _period: (1.0, 7),
            stream_terminal_text=lambda _state: None,
        )

    assert scheduled == [(7, FakeOperator.tick)]
    assert pauses == ["operator_tick_failed:update_stream"]
    assert logs == [
        "OPERATOR_TICK_FAILED stage=update_stream "
        "error=RuntimeError('synthetic stream-stage failure')"
    ]
    assert incidents == [
        (
            "operator_tick_failed",
            {
                "stage": "update_stream",
                "error": "RuntimeError('synthetic stream-stage failure')",
                "resolved": False,
            },
        )
    ]


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
    state = app.DroneState(stream="LOST", loc="STREAM_LOST", tracker_state="STREAM_LOST_HOVER")
    result = app.OperatorApp.state_from_live(operator, state)
    assert result.stream == "LOST"
    assert result.loc == "STREAM_LOST"
    assert result.tracker_state == "STREAM_LOST_HOVER"


def test_stream_lost_detection_recognizes_the_enum_tracker_state_not_just_a_plain_string() -> None:
    """str(TrackerState.STREAM_LOST_HOVER) is "TrackerState.STREAM_LOST_HOVER"
    (Enum.__str__), which would never match the plain-string membership check
    below and silently defeat stream-lost detection whenever tracker_state
    holds an enum member rather than a bare str."""
    from operator_state import TrackerState

    operator = SimpleNamespace(
        live_pose=app.np.zeros(4),
        live_result=None,
        localizer=SimpleNamespace(busy=lambda: False),
        video_frame_fresh=True,
        stream_lost_since=None,
        boot_holding=lambda: False,
    )
    state = app.DroneState(
        stream="OK", active_incident="", tracker_state=TrackerState.STREAM_LOST_HOVER
    )
    result = app.OperatorApp.state_from_live(operator, state)
    assert result.stream == "LOST"
    assert result.loc == "STREAM_LOST"


def test_live_stream_incident_pauses_integrated_auto_without_handoff() -> None:
    pauses = []
    state = app.DroneState(
        stream="LOST",
        active_incident=app.FailureReason.STREAM_STALE.value,
    )
    operator = SimpleNamespace(
        backend=SimpleNamespace(state=state),
        _pause_integrated_auto=lambda reason: pauses.append(reason),
    )

    result = operator_tick._handle_live_missing_frame(operator, state)

    assert result is state
    assert pauses == ["stream_stale"]


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
    assert app.localization_result_is_weak(
        {
            "composite_stage": "nn_fast_accept",
            "next_mode": "WEAK_TRACK",
        }
    )
    assert app.localization_result_is_weak(
        {
            "composite_stage": "lg_full_after_nn",
            "next_mode": "TRACK",
            "weak": True,
        }
    )
    assert not app.localization_result_is_weak(
        {
            "composite_stage": "lg_full_after_nn",
            "next_mode": "TRACK",
            "weak": False,
        }
    )


def test_confidence_hold_does_not_engage_on_low_results() -> None:
    policy = app.LostHoldPolicy(max_attempts=3, timeout_s=10.0)

    assert (
        policy.on_result(
            success=True,
            low_confidence=True,
            strong_relocalize=False,
            next_mode="TRACK",
            frame_index=10,
            now=1.0,
        )
        is None
    )
    assert (
        policy.on_result(
            success=True,
            low_confidence=True,
            strong_relocalize=False,
            next_mode="WEAK_TRACK",
            frame_index=11,
            now=1.1,
        )
        is None
    )
    assert not policy.active


def test_confidence_hold_engages_only_after_tracker_is_lost() -> None:
    policy = app.LostHoldPolicy(max_attempts=3, timeout_s=10.0)

    assert (
        policy.on_result(
            success=False,
            low_confidence=False,
            strong_relocalize=False,
            next_mode="WEAK_TRACK",
            frame_index=19,
            now=1.9,
        )
        is None
    )
    assert not policy.active
    assert (
        policy.on_result(
            success=False,
            low_confidence=False,
            strong_relocalize=False,
            next_mode="LOST",
            frame_index=20,
            now=2.0,
        )
        == "ENGAGE_FAIL"
    )
    assert policy.active


def test_real_low_confidence_escalation_hovers_and_requests_megaloc_once() -> None:
    clears = []
    pcmds = []
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
        nudge_clear=lambda *, reason: clears.append(reason),
        send_pcmd=lambda *pcmd, reason: pcmds.append((pcmd, reason)) or True,
    )
    operator.localizer = SimpleNamespace(request_relocalize=lambda: relocalize_calls.append(True))
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
            assert pcmds == []
            assert relocalize_calls == []

    assert clears == ["localization_recovery"]
    assert pcmds == [((0, 0, 0, 0), "localization_recovery_hover")]
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
    app.OperatorApp._append_loc_metrics(
        operator,
        {
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
        },
    )
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


def test_operator_and_worker_parsers_are_side_effect_free() -> None:
    operator_parser = app.build_argument_parser()
    worker_parser = localizer_worker.build_argument_parser()

    operator_args = operator_parser.parse_args(["--video", "sample.mp4"])
    worker_args = worker_parser.parse_args(["--frame-shm-name", "", "--frame-shm-slots", "0"])

    assert operator_args.video == "sample.mp4"
    assert operator_args.live is False
    assert worker_args.width == 1280
    assert worker_args.localizer_backend in {"auto", "edm", "xfeat"}


def test_temporal_pose_stabilizer_suppresses_single_frame_flip() -> None:
    stabilizer = app.TemporalPoseStabilizer(
        tau_s=0.15, max_speed_u_s=2.0, step_slack_u=0.03, max_step_u=0.15
    )
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
        tau_s=0.15, max_speed_u_s=2.0, step_slack_u=0.03, max_step_u=0.15
    )
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


def _circular_angle_error(angle: float, reference: float) -> float:
    return abs((float(angle) - float(reference) + math.pi) % (2.0 * math.pi) - math.pi)


def test_temporal_yaw_stabilizer_handles_pi_crossing() -> None:
    stabilizer = app.TemporalYawStabilizer(tau_s=0.15)
    samples = [
        math.pi - 0.04,
        -math.pi + 0.04,
        -math.pi + 0.08,
        -math.pi + 0.12,
    ]
    outputs = [
        stabilizer.update(value, 1.0 + index * 0.06)[0] for index, value in enumerate(samples)
    ]

    assert all(
        _circular_angle_error(output, sample) < 0.15
        for output, sample in zip(outputs[1:], samples[1:])
    )
    assert abs(outputs[-1]) > 3.0


def test_temporal_yaw_stabilizer_rejects_one_frame_spike() -> None:
    stabilizer = app.TemporalYawStabilizer(tau_s=0.15)
    samples = [0.0, 0.04, 2.8, 0.04, 0.08]
    outputs = [
        stabilizer.update(value, 1.0 + index * 0.06)[0] for index, value in enumerate(samples)
    ]

    assert abs(outputs[2]) < 0.1
    assert abs(outputs[3]) < 0.1
    assert _circular_angle_error(outputs[-1], samples[-1]) < 0.1


def test_temporal_yaw_stabilizer_reset_drops_previous_history() -> None:
    stabilizer = app.TemporalYawStabilizer()
    stabilizer.update(1.2, 1.0)
    stabilizer.update(1.3, 1.1)
    stabilizer.reset()

    output, info = stabilizer.update(-2.0, 2.0)

    assert output == pytest.approx(-2.0)
    assert info["published_delta_rad"] == 0.0


def test_yaw_publish_keeps_raw_pose_and_leaves_xyz_unchanged() -> None:
    operator = SimpleNamespace(
        pose_stabilizer=None,
        yaw_stabilizer=app.TemporalYawStabilizer(),
    )
    result = {
        "pose": {"x": 1, "y": 2, "z": 3, "yaw_raw": 0.25},
    }
    xyz = np.asarray([1.0, 2.0, 3.0])

    published_xyz = app.OperatorApp._stabilize_live_result_pose(
        operator,
        result,
        xyz,
    )

    assert published_xyz is xyz
    assert result["pose_raw"] == {
        "x": 1,
        "y": 2,
        "z": 3,
        "yaw_raw": 0.25,
    }
    assert result["pose"]["x"] == 1
    assert result["pose"]["y"] == 2
    assert result["pose"]["z"] == 3
    assert result["pose"]["yaw_raw"] == pytest.approx(0.25)


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
    route.write_text(
        json.dumps({"waypoints": [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]}),
        encoding="utf-8",
    )
    points = app.load_route_glomap(str(route))
    assert len(points) == 2
    assert points[0].tolist() == [1.0, -3.0, 2.0]
    assert points[1].tolist() == [2.0, -3.0, 2.0]


def test_raw_glomap_route_is_not_axis_converted(tmp_path: Path) -> None:
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "frame": "glomap",
                "waypoints": [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
            }
        ),
        encoding="utf-8",
    )

    points = app.load_route_glomap(str(route))

    assert points[0].tolist() == [1.0, 2.0, 3.0]
    assert points[1].tolist() == [2.0, 2.0, 3.0]


def test_preview_and_flight_loader_agree_on_every_route(tmp_path: Path) -> None:
    """The pink overlay and the path autonomy flies must come from one rule.

    load_route_glomap (preview) and real_path_follow_controller.load_waypoints
    (flight) each decide which basis to invert an "aligned" route with. If those
    two rules ever drift, the operator validates a route on the map that is NOT
    the one the flight loader produces -- rotated by the angle the two bases
    differ by, 22.5 deg on target_site_v1. This pins them together.
    """
    import numpy as np
    import real_path_follow_controller as rpf

    measured = rpf.MapFrame.from_gravity([0.009067509372034937, 0.924, 0.383])
    waypoints = [[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]

    cases = []
    for frame in ("aligned", "glomap"):
        for align_source in (None, "legacy", "measured", "bogus"):
            for site in (None, measured):
                body = {"frame": frame, "units": "map", "waypoints": waypoints}
                if align_source is not None:
                    body["align_source"] = align_source
                cases.append((body, site))

    for index, (body, site) in enumerate(cases):
        route = tmp_path / f"route_{index}.json"
        route.write_text(json.dumps(body), encoding="utf-8")

        preview_error = flight_error = None
        try:
            preview = app.load_route_glomap(str(route), map_frame=site)
        except Exception as exc:  # noqa: BLE001 - comparing the decision
            preview, preview_error = None, type(exc)
        try:
            flown = rpf.load_waypoints(
                route,
                map_frame=site if site is not None else rpf.LEGACY_MAP_FRAME,
            )
        except Exception as exc:  # noqa: BLE001 - comparing the decision
            flown, flight_error = None, type(exc)

        assert (preview_error is None) == (flight_error is None), (
            f"{body} at site measured={site is not None}: preview "
            f"{'rejected' if preview_error else 'accepted'} but flight loader "
            f"{'rejected' if flight_error else 'accepted'}"
        )
        if preview_error is None:
            assert np.allclose(np.asarray(preview), np.asarray(flown)), (
                f"{body}: preview and flight loader produced different points"
            )


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
        {"success": True, "pose": pose, "frame_name": "bad"}
    )
    assert result["success"] is False
    assert "invalid localization pose" in result["error"]
    assert xyz is None


def test_failed_visual_result_still_exposes_imu_guess_xyz() -> None:
    result, xyz = app.normalize_live_localization_result(
        {
            "success": False,
            "validity": False,
            "pose_status": "PREDICTED_ONLY",
            "pose": {"x": 1.5, "y": 2.5, "z": 3.5, "yaw_raw": 0.4},
        }
    )
    assert result["success"] is False
    assert xyz == pytest.approx([1.5, 2.5, 3.5])


def test_operator_anafi_profile_matches_simulator_contract() -> None:
    profile_path = app.SYSTEM_ROOT / "模擬器" / "sphinx_anafi_path_convergence" / "anafi_profile.py"
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
        sim.video_width_px,
        sim.video_height_px,
        sim.video_fps,
    )
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


def test_worker_loop_emits_one_json_result_for_one_mock_frame(monkeypatch) -> None:
    calls = []

    class FakeTracker:
        last_info = {"mode": "TRACK", "next_mode": "TRACK", "confidence": 0.8}

        def localize_frame(self, frame, *, capture_stamp=None):
            calls.append((frame.shape, capture_stamp))
            return SimpleNamespace(x=1.0, y=2.0, z=3.0, yaw=0.4)

    args = SimpleNamespace(
        width=1,
        height=1,
        frame_shm_name="",
        frame_shm_slots=0,
        runtime_benchmark_control=False,
        force_track_bench=False,
        force_track_ref=-1,
        max_frames=1,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(
        localizer_worker.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b"\x01\x02\x03")),
    )
    output = io.StringIO()

    localizer_worker._run_worker_loop(
        args,
        output,
        "xfeat",
        fake_torch,
        FakeTracker(),
        "mock-tracker",
        None,
        None,
        None,
    )

    payload = json.loads(output.getvalue())
    assert payload["seq"] == 0
    assert payload["success"] is True
    assert payload["frame_id"] == "worker-0"
    assert payload["tracker_variant"] == "mock-tracker"
    assert calls == [((1, 1, 3), None)]


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
        print(json.dumps({'seq': index, 'success': False, 'pixel_sum': sum(shm.buf[start:start + 6])}), flush=True)
finally:
    shm.close()
"""
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        2,
        1,
        str(tmp_path / "shared.log"),
        "shared-worker",
        "shared",
        timeout_s=1.0,
        use_shared_frames=True,
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
        (1, 6),
        (3, 18),
    ]


@pytest.mark.parametrize(
    ("requested", "count", "expected"),
    [(-1, 5, 2), (0, 5, 0), (4, 5, 4)],
)
def test_force_track_ref_resolution_is_explicit(requested: int, count: int, expected: int) -> None:
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


def test_fatal_worker_result_forces_immediate_restart(tmp_path: Path) -> None:
    worker = (
        "import json,sys; sys.stdin.buffer.read(3); "
        "print(json.dumps({'seq':0,'success':False,'restart_required':True}),flush=True)"
    )
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "fatal.log"),
        "fatal-worker",
        "fatal",
        timeout_s=1.0,
    )
    client.restart_warmup_s = 0.0
    old_pid = client.proc.pid
    try:
        assert client.submit(1, "frame", Image.new("RGB", (1, 1))) is True
        assert _wait_for(lambda: not client.results.empty())
        result = client.poll_results()[0]
        assert result["restart_required"] is True
        assert _wait_for(lambda: client.proc.pid != old_pid)
    finally:
        client.close()


def test_fatal_worker_restarts_are_bounded_and_publish_unavailable(tmp_path: Path) -> None:
    worker = (
        "import json,sys\n"
        "i=0\n"
        "while True:\n"
        "  raw=sys.stdin.buffer.read(3)\n"
        "  if not raw: break\n"
        "  print(json.dumps({'seq':i,'success':False,'restart_required':True}),flush=True)\n"
        "  i+=1"
    )
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "fatal-bounded.log"),
        "fatal-bounded-worker",
        "fatal",
        timeout_s=1.0,
    )
    client.restart_warmup_s = 0.0
    client.max_fatal_restart_attempts = 2
    client.fatal_restart_backoff_s = 0.0
    client.fatal_restart_backoff_max_s = 0.0
    frame = Image.new("RGB", (1, 1))
    try:
        for seq in (1, 2, 3):
            old_pid = client.proc.pid
            assert client.submit(seq, f"frame-{seq}", frame) is True
            assert _wait_for(lambda: not client.results.empty())
            result = client.poll_results()[0]
            assert result["restart_required"] is True
            if seq < 3:
                assert _wait_for(lambda: client.proc.pid != old_pid)
            else:
                assert _wait_for(lambda: client.unavailable)
                assert client.proc.pid == old_pid

        assert client.ready is False
        assert client._fatal_restart_attempts == 2
        assert "unavailable" in (client.startup_error or "")
        assert client.submit(4, "frame-4", frame) is False
        assert _wait_for(lambda: not client.results.empty())
        unavailable = client.poll_results()[0]
        assert unavailable["success"] is False
        assert unavailable["validity"] is False
        assert unavailable["worker_unavailable"] is True
        assert unavailable["failure_kind"] == "worker_unavailable"
        assert unavailable["restart_required"] is False
    finally:
        client.close()


def test_healthy_worker_response_resets_fatal_restart_budget(tmp_path: Path) -> None:
    worker = (
        "import json,sys\n"
        "i=0\n"
        "while True:\n"
        "  raw=sys.stdin.buffer.read(3)\n"
        "  if not raw: break\n"
        "  print(json.dumps({'seq':i,'success':False}),flush=True)\n"
        "  i+=1"
    )
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "fatal-reset.log"),
        "fatal-reset-worker",
        "fatal",
        timeout_s=1.0,
    )
    client.restart_warmup_s = 0.0
    client._fatal_restart_attempts = 2
    client._startup_error = "previous fatal failure"
    try:
        assert client.submit(1, "frame", Image.new("RGB", (1, 1))) is True
        assert _wait_for(lambda: not client.results.empty())
        result = client.poll_results()[0]
        assert result["success"] is False
        assert client._fatal_restart_attempts == 0
        assert client.startup_error is None
    finally:
        client.close()


def test_startup_failure_restarts_are_bounded(tmp_path: Path) -> None:
    client = app.LiveWorkerClient(
        [sys.executable, "-c", "raise SystemExit(3)"],
        1,
        1,
        str(tmp_path / "startup-fatal.log"),
        "startup-fatal-worker",
        "fatal",
        timeout_s=0.05,
        expect_ready_event=True,
    )
    client.restart_warmup_s = 0.05
    client.max_fatal_restart_attempts = 2
    client.fatal_restart_backoff_s = 0.0
    client.fatal_restart_backoff_max_s = 0.0
    try:
        assert _wait_for(lambda: client.unavailable, timeout=2.0)
        assert client.ready is False
        assert client._fatal_restart_attempts == 2
        assert "unavailable" in (client.startup_error or "")
    finally:
        client.close()


def test_worker_client_reports_submit_pipe_worker_and_response_timing(tmp_path: Path) -> None:
    worker = (
        "import json,sys,time; "
        "sys.stdin.buffer.read(3); "
        "rn=time.monotonic_ns(); sn=time.monotonic_ns(); dn=time.monotonic_ns(); "
        "r=rn*1e-9; s=sn*1e-9; d=dn*1e-9; "
        "print(json.dumps({'seq':0,'success':False,'wall_ms':1.0,'core_wall_ms':1.0,"
        "'worker_read_done_mono':r,'worker_core_start_mono':s,"
        "'worker_core_done_mono':d,'worker_read_done_mono_ns':rn,"
        "'worker_core_start_mono_ns':sn,'worker_core_done_mono_ns':dn}),flush=True)"
    )
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "timing.log"),
        "timing-worker",
        "timing",
        timeout_s=1.0,
    )
    client.restart_warmup_s = 0.0
    try:
        assert client.submit(
            1,
            "timed",
            Image.new("RGB", (1, 1)),
            timing_metadata={
                "ui_serialize_ms": 0.25,
                "source_frame_stamp_mono": time.monotonic() - 0.01,
                "source_stamp_semantics": "stream_monotonic_mapped_or_receipt",
                "frame_callback_enter_mono_ns": time.monotonic_ns() - 20_000_000,
                "frame_preprocess_start_mono_ns": time.monotonic_ns() - 15_000_000,
                "frame_yuv_ready_mono_ns": time.monotonic_ns() - 14_000_000,
                "frame_preprocess_done_mono_ns": time.monotonic_ns() - 10_000_000,
            },
        )
        assert _wait_for(lambda: not client.results.empty())
        result = client.poll_results()[0]
    finally:
        client.close()
    for key in (
        "client_submit_mono",
        "client_dequeue_mono",
        "client_write_start_mono",
        "client_write_done_mono",
        "worker_read_done_mono",
        "worker_core_start_mono",
        "worker_core_done_mono",
        "client_response_mono",
        "client_queue_wait_ms",
        "client_pipe_write_ms",
        "client_roundtrip_ms",
        "submit_to_worker_read_ms",
        "worker_done_to_client_ms",
        "source_stamp_age_at_submit_ms",
        "client_submit_mono_ns",
        "client_dequeue_mono_ns",
        "client_write_start_mono_ns",
        "client_write_done_mono_ns",
        "worker_read_done_mono_ns",
        "worker_core_start_mono_ns",
        "worker_core_done_mono_ns",
        "client_response_mono_ns",
        "callback_to_preprocess_start_ms",
        "yuv_view_ms",
        "frame_preprocess_ms",
        "callback_to_submit_ms",
        "callback_to_inference_start_ms",
        "callback_to_localization_done_ms",
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
    threads = [threading.Thread(target=lambda: results.append(client._restart())) for _ in range(2)]
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


def test_binary_ply_reader_preserves_xyz_and_rgb(tmp_path: Path) -> None:
    ply = tmp_path / "binary.ply"
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    ply.write_bytes(
        header
        + app.struct.pack("<fffBBB", 1.0, 2.0, 3.0, 4, 5, 6)
        + app.struct.pack("<fffBBB", 7.0, 8.0, 9.0, 10, 11, 12)
    )

    assert app.read_ply_points(ply, max_points=2).tolist() == [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
    ]


@pytest.mark.parametrize(
    ("loss_pct", "expected"),
    [
        (0.0, b"\x00\x00\x01\x01a\x00\x00\x01\x05b\x00\x00\x01\x07c"),
        (100.0, b"\x00\x00\x01\x05b\x00\x00\x01\x07c"),
    ],
)
def test_annex_b_pump_only_drops_non_idr_slices(loss_pct: float, expected: bytes) -> None:
    class RecordingSink(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            self.closed_by_pump = False

        def close(self) -> None:
            self.closed_by_pump = True

    source = io.BytesIO(b"\x00\x00\x01\x01a\x00\x00\x01\x05b\x00\x00\x01\x07c")
    destination = RecordingSink()

    app.FFmpegFrameStream._pump_nals(source, destination, loss_pct, seed=5)

    assert destination.getvalue() == expected
    assert destination.closed_by_pump is True


def test_motion_headings_bridge_missing_poses_and_backfill_start() -> None:
    rows = [
        {},
        {"pose": {"x": 0.0, "z": 0.0}},
        {},
        {"pose": {"x": 1.0, "z": 1.0}},
    ]

    headings = app.OperatorApp._derive_motion_headings(rows)

    assert headings == pytest.approx([math.pi / 4.0] * 4)


def test_motion_headings_ignore_subthreshold_motion() -> None:
    rows = [
        {"pose": {"x": 0.0, "z": 0.0}},
        {"pose": {"x": 0.05, "z": 0.05}},
    ]

    assert app.OperatorApp._derive_motion_headings(rows) == [None, None]


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
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[3] / '控制介面程式' / 'operator_interface')!r})\n"
            "from live_localizer_worker import attach_frame_shm\n"
            "attach_frame_shm(sys.argv[1])\n"
            "print('attached', flush=True)\n"
            "time.sleep(30)\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", child, shm.name], stdout=subprocess.PIPE)
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


def test_stale_response_after_a_refused_restart_is_never_published_as_a_new_fix(
    tmp_path: Path,
) -> None:
    """A worker response must never be attributed to a frame it did not answer.

    Two timeouts inside the 30 s restart cooldown used to leave the stalled worker
    attached: the client then read request N's answer while writing request N+1 and
    stamped the OLD pose with the NEW frame's identity and timing. The worker numbers
    every payload, so the mismatch is detectable — and must force a restart even
    though the cooldown would normally refuse one.
    """
    worker = (
        "import sys,time,json\n"
        "seq=0\n"
        "while True:\n"
        "    b=sys.stdin.buffer.read(3)\n"
        "    if not b: break\n"
        "    if seq==0: time.sleep(0.8)\n"  # first frame stalls past the timeout
        "    sys.stdout.write(json.dumps({'seq':seq,'success':True,"
        "'x':float(seq)})+'\\n'); sys.stdout.flush()\n"
        "    seq+=1\n"
    )
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "desync.log"),
        "desync-worker",
        "desync",
        timeout_s=0.2,
    )
    client.restart_warmup_s = 0.0
    try:
        # Burn the restart budget so the stall's _restart() is refused by the cooldown.
        client._last_restart = time.monotonic()
        old_pid = client.proc.pid

        assert client.submit(1, "frame_a", Image.new("RGB", (1, 1))) is True
        assert _wait_for(lambda: bool(client.poll_results()), timeout=3.0)

        # Let the stalled worker finish frame_a, so its late answer is sitting in the
        # pipe when frame_b's response is read.
        time.sleep(1.2)
        assert _wait_for(lambda: client.submit(2, "frame_b", Image.new("RGB", (1, 1))), timeout=3.0)
        published: list[dict] = []
        assert _wait_for(
            lambda: bool(published.extend(client.poll_results()) or published),
            timeout=5.0,
        )
        frame_b = [p for p in published if p.get("frame_name") == "frame_b"]
        assert frame_b, "frame_b produced no result at all"
        for payload in frame_b:
            assert not payload.get("success"), (
                f"frame_a's pose was published as frame_b's fix: {payload}"
            )
            assert "desync" in str(payload.get("error", "")).lower()
        # Desync must force a respawn despite the cooldown.
        assert _wait_for(lambda: client.proc.pid != old_pid, timeout=5.0)
    finally:
        client.close()
    assert not client.thread.is_alive()


def test_dead_localizer_triggers_worker_exit_fail_safe() -> None:
    from backend_contract import FailureReason

    calls: list[object] = []
    app = SimpleNamespace(
        localizer=SimpleNamespace(unavailable=True),
        backend=SimpleNamespace(fail_safe=lambda reason: calls.append(reason)),
        write_log=lambda _text: None,
    )
    operator_tick._fail_safe_dead_localizer(app)
    operator_tick._fail_safe_dead_localizer(app)
    assert calls == [FailureReason.WORKER_EXIT]


def test_fused_request_keeps_independent_telemetry_stamp() -> None:
    header = localizer_protocol.encode_fused_request(
        "auto",
        10.0,
        localizer_protocol.FusedTelemetry(
            stamp=10.05,
            roll=0.1,
            pitch=0.0,
            yaw=0.2,
        ),
        telemetry_stamp=10.05,
    )
    mode, capture_stamp, fused = localizer_protocol.decode_control_header(header)
    assert mode == "auto"
    assert capture_stamp == pytest.approx(10.0)
    assert fused is not None
    assert fused.stamp == pytest.approx(10.05)
    assert fused.roll == pytest.approx(0.1)


@pytest.mark.parametrize(
    "telemetry_stamp",
    [float("nan"), float("inf"), -1.0, 0.0],
)
def test_fused_request_rejects_non_finite_or_non_positive_telemetry(
    telemetry_stamp: float,
) -> None:
    with pytest.raises(ValueError, match="telemetry stamp"):
        localizer_protocol.encode_fused_request(
            "auto",
            10.0,
            localizer_protocol.FusedTelemetry(roll=0.0, pitch=0.0, yaw=0.0),
            telemetry_stamp=telemetry_stamp,
        )


def test_fused_request_rejects_future_and_stale_telemetry() -> None:
    fused = localizer_protocol.FusedTelemetry(roll=0.0, pitch=0.0, yaw=0.0)
    with pytest.raises(ValueError, match="future"):
        localizer_protocol.encode_fused_request(
            "auto",
            10.0,
            fused,
            telemetry_stamp=time.monotonic() + 1.0,
        )
    with pytest.raises(ValueError, match="stale"):
        localizer_protocol.encode_fused_request(
            "auto",
            10.0,
            fused,
            telemetry_stamp=10.0 + localizer_protocol.MAX_FUSED_SYNC_ERROR_S + 0.01,
        )


def test_legacy_fused_header_is_rejected() -> None:
    legacy = localizer_protocol._LEGACY_FUSED_HEADER.pack(
        localizer_protocol.FUSED_MAGIC,
        0,
        10.0,
        localizer_protocol.FLAG_HAS_ATTITUDE,
        0.1,
        0.0,
        0.2,
        float("nan"),
        float("nan"),
        float("nan"),
    )
    with pytest.raises(ValueError, match="legacy fused"):
        localizer_protocol.decode_control_header(legacy)
    with pytest.raises(ValueError, match="header size"):
        localizer_protocol.decode_request(legacy)


def test_live_localizer_does_not_backdate_unstamped_telemetry() -> None:
    client = object.__new__(app.LiveLocalizerClient)
    client._runtime_benchmark_control = True
    client._benchmark_mode_lock = threading.Lock()
    client._benchmark_mode = "auto"
    client._relocalize_once = False
    header = client._request_prefix(
        {
            "source_frame_stamp_mono": 10.0,
            "fused_roll": 0.1,
            "fused_pitch": -0.2,
            "fused_yaw": 1.3,
        }
    )
    mode, stamp, fused = localizer_protocol.decode_control_header(header)
    assert mode == "auto"
    assert stamp == pytest.approx(10.0)
    assert fused is None


def test_shared_frame_coalesce_copies_only_on_promote(tmp_path: Path) -> None:
    worker = """
import argparse, json, sys, time
from multiprocessing import shared_memory
p = argparse.ArgumentParser()
p.add_argument('--frame-shm-name', required=True)
p.add_argument('--frame-shm-slots', required=True, type=int)
a = p.parse_args()
shm = shared_memory.SharedMemory(name=a.frame_shm_name)
try:
    slot = sys.stdin.buffer.read(1)
    time.sleep(0.2)
    start = slot[0] * 6
    print(json.dumps({'seq': 0, 'success': False, 'pixel_sum': sum(shm.buf[start:start + 6])}), flush=True)
    slot = sys.stdin.buffer.read(1)
    if slot:
        start = slot[0] * 6
        print(json.dumps({'seq': 1, 'success': False, 'pixel_sum': sum(shm.buf[start:start + 6])}), flush=True)
finally:
    shm.close()
"""
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        2,
        1,
        str(tmp_path / "coalesce-promote.log"),
        "coalesce-promote",
        "shared",
        timeout_s=1.0,
        use_shared_frames=True,
        frame_shm_slots=2,
    )
    client.restart_warmup_s = 0.0
    first = np.full((1, 2, 3), 1, np.uint8)
    latest = np.full((1, 2, 3), 9, np.uint8)
    try:
        assert client.submit(1, "first", first)
        assert _wait_for(client.busy)
        active = client._active_shm_slot
        assert active == 0
        free = 1 - active
        before_free = bytes(client._frame_shm.buf[free * 6 : (free + 1) * 6])
        middle = np.full((1, 2, 3), 2, np.uint8)
        latest[0, 0, 0] = 4
        assert client.submit(2, "middle", middle)
        assert client.submit(3, "latest", latest)
        latest[:] = 0
        middle[:] = 0
        after_free = bytes(client._frame_shm.buf[free * 6 : (free + 1) * 6])
        assert after_free == before_free
        assert client._coalesce is not None
        assert client._coalesce_drops == 1
        assert _wait_for(lambda: client.results.qsize() == 2, timeout=3.0)
        results = client.poll_results()
    finally:
        client.close()
        assert client._coalesce is None
        assert client._frame_shm is None
    assert [(row["display_seq"], row["pixel_sum"]) for row in results] == [
        (1, 6),
        (3, 4 + 9 * 5),
    ]
    assert results[0]["circuit_breaker_state"] == "closed"
    assert results[0]["rejected_submits"] == 0
    assert results[0]["ready_latency_ms"] is not None
    assert results[0]["first_result_latency_ms"] is not None


def test_shared_frame_three_slot_coalesce_semantics(tmp_path: Path) -> None:
    """3-slot ring coalescing delivers the newest complete frame and drops intermediates without bytearray double-copy."""
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
            time.sleep(0.2)
        start = slot[0] * 6
        print(json.dumps({'seq': index, 'success': False, 'slot': slot[0], 'pixel_sum': sum(shm.buf[start:start + 6])}), flush=True)
finally:
    shm.close()
"""
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        2,
        1,
        str(tmp_path / "three-slot-coalesce.log"),
        "three-slot-coalesce",
        "shared",
        timeout_s=1.0,
        use_shared_frames=True,
        frame_shm_slots=3,
    )
    client.restart_warmup_s = 0.0
    assert client._coalesce_scratch is None
    assert client._frame_shm_slots == 3

    first = np.full((1, 2, 3), 1, np.uint8)
    middle1 = np.full((1, 2, 3), 2, np.uint8)
    middle2 = np.full((1, 2, 3), 3, np.uint8)
    latest = np.full((1, 2, 3), 7, np.uint8)
    try:
        assert client.submit(1, "first", first)
        assert _wait_for(client.busy)
        assert client.submit(2, "middle1", middle1)
        assert client.submit(3, "middle2", middle2)
        assert client.submit(4, "latest", latest)
        assert client._coalesce_drops == 2
        assert _wait_for(lambda: client.results.qsize() == 2, timeout=3.0)
        results = client.poll_results()
    finally:
        client.close()
    assert [(row["display_seq"], row["pixel_sum"]) for row in results] == [
        (1, 6),
        (4, 42),
    ]


def test_shared_frame_three_slot_tearing_protection(monkeypatch) -> None:
    """Torn or incomplete slots are not promoted; the previous valid slot is preserved."""
    from multiprocessing import shared_memory
    client = app.LiveWorkerClient.__new__(app.LiveWorkerClient)
    client.width = 2
    client.height = 1
    client._frame_size = 6
    client._frame_shm_slots = 3
    client._frame_shm = shared_memory.SharedMemory(create=True, size=18)
    client._active_shm_slot = 0
    client._ready_shm_slot = None
    client._writing_shm_slot = None
    client._slot_seq = [0, 1, 2]
    client._slot_ready = [True, False, True]
    client._last_valid_slot = 0
    client._coalesce = (2, "torn-frame", b"", 1, {})
    client._closed = threading.Event()
    client._lock = threading.Lock()
    client.in_flight = True
    client.pending = queue.Queue(maxsize=1)
    client._coalesce_drops = 0
    try:
        client._promote_coalesced_work()
        assert client._active_shm_slot == 0
        promoted = client.pending.get_nowait()
        assert promoted[3] == 0

        worker_args = SimpleNamespace(width=2, height=1, frame_shm_slots=3)
        stdin_buf = io.BytesIO(bytes([0]))
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=stdin_buf))
        f1 = localizer_worker._read_worker_frame(worker_args, None, client._frame_shm, 6)
        assert f1 is not None

        stdin_buf_corrupt = io.BytesIO(bytes([9]))
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=stdin_buf_corrupt))
        f_fallback = localizer_worker._read_worker_frame(worker_args, None, client._frame_shm, 6)
        assert f_fallback is f1
    finally:
        f1 = None
        f_fallback = None
        localizer_worker._close_worker_frame_shm(None)
        client._frame_shm.close()
        client._frame_shm.unlink()


def test_shared_frame_three_slot_rotation() -> None:
    """3-slot ring rotates across {0, 1, 2} ensuring active, ready, and writing are mutually disjoint."""
    from multiprocessing import shared_memory
    client = app.LiveWorkerClient.__new__(app.LiveWorkerClient)
    client.width = 2
    client.height = 1
    client._frame_size = 6
    client._frame_shm_slots = 3
    client._frame_shm = shared_memory.SharedMemory(create=True, size=18)
    client._active_shm_slot = None
    client._ready_shm_slot = None
    client._writing_shm_slot = None
    client._slot_seq = [None, None, None]
    client._slot_ready = [False, False, False]
    client._last_valid_slot = None
    client._coalesce = None
    client._closed = threading.Event()
    client._lock = threading.Lock()
    client.in_flight = False
    client.pending = queue.Queue(maxsize=1)
    client._coalesce_drops = 0
    try:
        raw1 = memoryview(b"\x01" * 6)
        client._submit_shared_frame(1, "f1", b"", raw1, {})
        assert client._active_shm_slot == 0
        assert client.in_flight is True
        assert client._slot_ready[0] is True

        raw2 = memoryview(b"\x02" * 6)
        client._submit_shared_frame(2, "f2", b"", raw2, {})
        assert client._active_shm_slot == 0
        assert client._ready_shm_slot == 1
        assert client._slot_ready[1] is True

        raw3 = memoryview(b"\x03" * 6)
        client._submit_shared_frame(3, "f3", b"", raw3, {})
        assert client._active_shm_slot == 0
        assert client._ready_shm_slot == 2
        assert client._slot_ready[2] is True
        assert client._coalesce_drops == 1

        client.pending.get_nowait()
        client._promote_coalesced_work()
        assert client._active_shm_slot == 2
        assert client._ready_shm_slot is None
        assert client.in_flight is True

        raw4 = memoryview(b"\x04" * 6)
        client._submit_shared_frame(4, "f4", b"", raw4, {})
        assert client._active_shm_slot == 2
        assert client._ready_shm_slot == 0
        assert client._slot_ready[0] is True
    finally:
        client._frame_shm.close()
        client._frame_shm.unlink()

def test_rejected_submit_and_oom_transition_are_bounded(tmp_path: Path) -> None:
    worker = """
import json, sys
raw = sys.stdin.buffer.read(3)
if not raw:
    raise SystemExit
print(json.dumps({
    "seq": 0, "success": False, "error": "cuda_oom", "failure_kind": "cuda_oom",
    "restart_required": True, "worker_fatal": True,
}), flush=True)
sys.stdin.buffer.read()
"""
    client = app.LiveWorkerClient(
        [sys.executable, "-c", worker],
        1,
        1,
        str(tmp_path / "oom.log"),
        "oom-worker",
        "oom",
        timeout_s=1.0,
    )
    client.restart_warmup_s = 0.0
    try:
        assert not client.submit(1, "bad", np.zeros((2, 2, 3), np.uint8))
        assert client._rejected_submits == 1
        frame = np.zeros((1, 1, 3), np.uint8)
        assert client.submit(2, "oom", frame)
        assert _wait_for(lambda: client.results.qsize() >= 1, timeout=3.0)
        results = client.poll_results()
    finally:
        client.close()
    from localization_metrics import CIRCUIT_BREAKER_STATES, RESTART_REASONS

    oom_rows = [
        row for row in results if row.get("oom_transition") or row.get("error") == "cuda_oom"
    ]
    assert oom_rows
    assert oom_rows[0]["restart_reason"] in RESTART_REASONS
    assert oom_rows[0]["circuit_breaker_state"] in CIRCUIT_BREAKER_STATES
    assert oom_rows[0]["oom_transition"] is True


def test_worker_fused_odometry_uses_telemetry_stamp_not_capture() -> None:
    fused = localizer_protocol.FusedTelemetry(
        stamp=10.05,
        roll=0.1,
        pitch=-0.2,
        yaw=1.3,
        speed_north=0.4,
        speed_east=-0.1,
        speed_down=0.0,
    )
    sample = localizer_worker._fused_odometry_sample_from_request(
        fused,
        capture_stamp=10.0,
        now_mono=10.1,
    )
    assert sample is not None
    assert sample.timestamp == pytest.approx(10.05)


def test_worker_fused_odometry_keeps_independent_gnss_stamp() -> None:
    fused = localizer_protocol.FusedTelemetry(
        stamp=10.05,
        roll=0.1,
        pitch=-0.2,
        yaw=1.3,
        gps_stamp=9.5,
        latitude=25.033,
        longitude=121.5654,
        altitude=18.2,
        latitude_accuracy=0.8,
        longitude_accuracy=0.9,
        altitude_accuracy=1.4,
    )

    sample = localizer_worker._fused_odometry_sample_from_request(
        fused,
        capture_stamp=10.0,
        now_mono=10.1,
    )

    assert sample is not None and sample.has_gnss
    assert sample.timestamp == pytest.approx(10.05)
    assert sample.geodetic_timestamp == pytest.approx(9.5)
    assert sample.geodetic_lla == pytest.approx((25.033, 121.5654, 18.2))


def test_worker_fused_odometry_does_not_backdate_unstamped_telemetry() -> None:
    fused = localizer_protocol.FusedTelemetry(
        stamp=None,
        roll=0.1,
        pitch=-0.2,
        yaw=1.3,
    )
    assert (
        localizer_worker._fused_odometry_sample_from_request(
            fused,
            capture_stamp=10.0,
            now_mono=10.1,
        )
        is None
    )
    assert (
        localizer_worker._fused_odometry_sample_from_request(
            None,
            capture_stamp=10.0,
            now_mono=10.1,
        )
        is None
    )


def test_worker_fused_odometry_rejects_future_and_stale_stamps() -> None:
    future = localizer_protocol.FusedTelemetry(
        stamp=10.2,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )
    assert (
        localizer_worker._fused_odometry_sample_from_request(
            future,
            capture_stamp=10.0,
            now_mono=10.1,
        )
        is None
    )
    stale = localizer_protocol.FusedTelemetry(
        stamp=10.0 + localizer_protocol.MAX_FUSED_SYNC_ERROR_S + 0.01,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )
    assert (
        localizer_worker._fused_odometry_sample_from_request(
            stale,
            capture_stamp=10.0,
            now_mono=stale.stamp + 1.0,
        )
        is None
    )


def test_worker_publishes_sampled_cuda_and_per_class_edm_cache_metrics() -> None:
    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def memory_allocated(self) -> int:
            return 10

        def memory_reserved(self) -> int:
            return 20

        def max_memory_allocated(self) -> int:
            return 30

        def synchronize(self) -> None:
            raise AssertionError("must not force CUDA synchronization")

    tracker = SimpleNamespace(
        loc=SimpleNamespace(
            matcher=SimpleNamespace(
                feature_cache_stats_by_class=lambda: {
                    "map": {"hits": 8, "misses": 1, "evictions": 0},
                    "temporal": {"hits": 2, "misses": 1, "evictions": 3},
                }
            )
        ),
        last_info={},
    )
    payload = {}
    last_mono, last_stats = localizer_worker._attach_sampled_runtime_metrics(
        payload,
        SimpleNamespace(cuda=FakeCuda()),
        tracker,
        None,
        None,
    )
    assert last_mono is not None
    assert last_stats == {
        "cuda_allocated_bytes": 10,
        "cuda_reserved_bytes": 20,
        "cuda_peak_allocated_bytes": 30,
    }
    assert payload["cuda_allocated_bytes"] == 10
    assert payload["cuda_reserved_bytes"] == 20
    assert payload["cuda_peak_allocated_bytes"] == 30
    assert payload["edm_cache_hits"] == 10
    assert payload["edm_cache_misses"] == 2
    assert payload["edm_cache_evictions"] == 3


def test_worker_reads_edm_cache_metrics_through_adapter_tracker() -> None:
    tracker = SimpleNamespace(
        trk=SimpleNamespace(
            loc=SimpleNamespace(
                matcher=SimpleNamespace(
                    feature_cache_stats_by_class=lambda: {
                        "map": {"hits": 9, "misses": 2, "evictions": 1},
                        "temporal": {"hits": 3, "misses": 1, "evictions": 0},
                    }
                )
            )
        ),
        last_info={},
    )

    assert localizer_worker._edm_cache_stats_from_tracker(tracker) == {
        "edm_cache_hits": 12,
        "edm_cache_misses": 3,
        "edm_cache_evictions": 1,
    }


def test_worker_loop_observes_independent_fused_stamp(monkeypatch) -> None:
    observed = []

    class FakeTracker:
        last_info = {"mode": "TRACK", "next_mode": "TRACK", "confidence": 0.8}
        loc = SimpleNamespace(
            matcher=SimpleNamespace(
                feature_cache_stats_by_class=lambda: {
                    "map": {"hits": 4, "misses": 1, "evictions": 0},
                    "temporal": {"hits": 1, "misses": 0, "evictions": 2},
                }
            )
        )

        def observe_fused_state(self, sample) -> None:
            observed.append(sample.timestamp)

        def localize_frame(self, frame, *, capture_stamp=None):
            return SimpleNamespace(x=1.0, y=2.0, z=3.0, yaw=0.4)

    now = time.monotonic()
    capture = now - 0.05
    fused_stamp = now - 0.02
    header = localizer_protocol.encode_fused_request(
        "auto",
        capture,
        localizer_protocol.FusedTelemetry(
            stamp=fused_stamp,
            roll=0.1,
            pitch=0.0,
            yaw=0.2,
        ),
        telemetry_stamp=fused_stamp,
    )
    args = SimpleNamespace(
        width=1,
        height=1,
        frame_shm_name="",
        frame_shm_slots=0,
        runtime_benchmark_control=True,
        force_track_bench=False,
        force_track_ref=-1,
        max_frames=1,
    )
    monkeypatch.setattr(
        localizer_worker.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(header + b"\x01\x02\x03")),
    )
    output = io.StringIO()
    localizer_worker._run_worker_loop(
        args,
        output,
        "xfeat",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        FakeTracker(),
        "mock-tracker",
        None,
        None,
        lambda *_args, **_kwargs: None,
    )
    payload = json.loads(output.getvalue())
    assert observed == [pytest.approx(fused_stamp)]
    assert payload["success"] is True
    assert payload["edm_cache_hits"] == 5
    assert payload["edm_cache_misses"] == 1
    assert payload["edm_cache_evictions"] == 2


def test_client_lifecycle_metrics_follow_outage_and_ready_transitions() -> None:
    client = object.__new__(app.LiveWorkerClient)
    client._lock = threading.Lock()
    client._restart_reason = None
    client._outage_start_mono = None
    client._outage_duration_s = None
    client._rejected_submits = 0
    client._ready_mono = None
    client._ready_latency_ms = None
    client._first_result_latency_ms = None
    client._circuit_breaker_state = "closed"
    client._pending_oom_transition = False
    client._ready_event = threading.Event()
    client._spawned_at = time.monotonic() - 0.05
    client._note_outage_start("stall")
    time.sleep(0.02)
    client._note_worker_ready()
    payload = {}
    client._attach_lifecycle(payload)
    assert payload["restart_reason"] == "stall"
    assert payload["outage_duration_s"] >= 0.02
    assert payload["rejected_submits"] == 0
    assert payload["coalesced_submit_drops"] == 0
    assert payload["ready_latency_ms"] >= 50.0
    assert payload["first_result_latency_ms"] is not None
    assert payload["circuit_breaker_state"] == "closed"
    assert payload["oom_transition"] is False


def test_live_heading_uses_the_camera_axis_without_an_auto_map_frame() -> None:
    # Regression: the measured gravity frame is only bound while integrated
    # AUTO runs, and the camera branch demanded one. Everywhere else a measured
    # camera forward fell through to the displacement branch, so the operator
    # was shown the TRAVEL bearing labelled as a camera heading -- on the river
    # map the two differ by 77 deg at the median.
    operator = SimpleNamespace(
        camera_forward_world=None,
        camera_axes_world=None,
        live_last_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
        _integrated_auto_map_frame=None,
        live_heading=None,
        live_pose=np.zeros(4),
    )

    app.OperatorApp._update_live_camera_orientation(
        operator,
        {"camera_forward_world": [0.0, 0.0, 1.0]},
        np.array([1.0, 0.0, 0.0], dtype=float),  # travelling +x, looking +z
    )

    assert operator.live_heading == pytest.approx(math.pi / 2)


def test_live_heading_falls_back_to_travel_only_without_a_camera_axis() -> None:
    operator = SimpleNamespace(
        camera_forward_world=None,
        camera_axes_world=None,
        live_last_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
        _integrated_auto_map_frame=None,
        live_heading=None,
        live_pose=np.zeros(4),
    )

    app.OperatorApp._update_live_camera_orientation(
        operator, {}, np.array([1.0, 0.0, 0.0], dtype=float)
    )

    assert operator.live_heading == pytest.approx(0.0)

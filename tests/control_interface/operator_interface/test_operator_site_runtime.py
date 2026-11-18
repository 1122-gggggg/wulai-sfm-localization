from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
import queue
import threading
from unittest.mock import Mock

from backend_contract import InterfaceMode, RuntimeSwitchKind, RuntimeSwitchStatus
import flight_operator_app as app
import operator_launch as launch
import operator_stream_modes as stream_modes

import pytest

from operator_site_runtime import (
    ActiveSiteRuntime,
    PreparedSiteRuntime,
    close_active_site_runtime,
    replace_active_site_runtime,
)


class _Closer:
    def __init__(self, name: str = "resource", *, fail: bool = False) -> None:
        self.closed = 0
        self.name = name
        self.fail = fail

    def close(self, *args, **kwargs) -> None:
        self.closed += 1
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")


class _Logs(_Closer):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__("session_logs", fail=fail)
        self.reason = None

    def close(self, *, reason: str) -> None:
        self.reason = reason
        super().close()


class _LiveBackend:
    is_live = True

    def __init__(self, cleanup_result=True) -> None:
        self.cleanup_result = cleanup_result
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1
        return self.cleanup_result


def _prepared(name: str) -> PreparedSiteRuntime:
    return PreparedSiteRuntime(
        args=SimpleNamespace(site_profile=f"/{name}/site_profile.json"),
        interface_mode="real-flight",
        profile=SimpleNamespace(site_id=name),
        hardware_approval=None,
        map_points=(),
        route_points=(),
        mission_route_snapshot=None,
        replay_rows=(),
    )


def _active(
    name: str,
    *,
    cleanup_result=True,
    failures: tuple[str, ...] = (),
) -> ActiveSiteRuntime:
    return ActiveSiteRuntime(
        prepared=_prepared(name),
        session_logs=_Logs(fail="session_logs" in failures),
        backend=_LiveBackend(cleanup_result),
        video_stream=object(),
        live_backend=object(),
        localizer=_Closer("localizer", fail="localizer" in failures),
        detector=_Closer("detector", fail="detector" in failures),
        lost_hold=None,
    )


def test_close_runtime_requires_verified_live_cleanup_before_workers_close() -> None:
    runtime = _active("old", cleanup_result=False)

    with pytest.raises(RuntimeError, match="cleanup"):
        close_active_site_runtime(runtime, reason="site_switch")

    assert runtime.backend.cleanup_calls == 1
    assert runtime.localizer.closed == 0
    assert runtime.detector.closed == 0
    assert runtime.session_logs.closed == 0


@pytest.mark.parametrize(
    "failures",
    [
        ("localizer",),
        ("detector",),
        ("session_logs",),
        ("localizer", "detector"),
        ("localizer", "session_logs"),
        ("detector", "session_logs"),
        ("localizer", "detector", "session_logs"),
    ],
)
def test_close_runtime_attempts_every_dependent_once_and_reports_failures(
    failures: tuple[str, ...],
) -> None:
    runtime = _active("old", failures=failures)

    with pytest.raises(RuntimeError) as raised:
        close_active_site_runtime(runtime, reason="site_switch")

    assert runtime.backend.cleanup_calls == 1
    assert runtime.localizer.closed == 1
    assert runtime.detector.closed == 1
    assert runtime.session_logs.closed == 1
    message = str(raised.value)
    assert all(failure in message for failure in failures)


@pytest.mark.parametrize("cleanup_result", [None, 1, "ok"])
def test_live_cleanup_accepts_only_literal_true(cleanup_result) -> None:
    runtime = _active("old", cleanup_result=cleanup_result)

    with pytest.raises(RuntimeError, match="not confirmed"):
        close_active_site_runtime(runtime, reason="site_switch")

    assert runtime.localizer.closed == 0
    assert runtime.detector.closed == 0
    assert runtime.session_logs.closed == 0


def test_replace_runtime_closes_old_then_activates_new() -> None:
    old = _active("old")
    new_plan = _prepared("new")
    started = []

    def start(plan):
        started.append(plan.profile.site_id)
        return _active(plan.profile.site_id)

    result = replace_active_site_runtime(old, new_plan, start)

    assert result.applied
    assert result.runtime is not None
    assert result.runtime.prepared.profile.site_id == "new"
    assert started == ["new"]
    assert old.backend.cleanup_calls == 1
    assert old.localizer.closed == 1
    assert old.detector.closed == 1
    assert old.session_logs.reason == "site_switch"


def test_replace_runtime_rebuilds_old_site_when_new_start_fails() -> None:
    old = _active("old")
    new_plan = _prepared("new")
    started = []

    def start(plan):
        started.append(plan.profile.site_id)
        if plan.profile.site_id == "new":
            raise RuntimeError("new connection failed")
        return _active(plan.profile.site_id)

    result = replace_active_site_runtime(old, new_plan, start)

    assert not result.applied
    assert result.rolled_back
    assert result.runtime is not None
    assert result.runtime.prepared.profile.site_id == "old"
    assert "new connection failed" in (result.error or "")
    assert started == ["new", "old"]


def test_replace_runtime_keeps_old_site_when_cleanup_is_not_confirmed() -> None:
    old = _active("old", cleanup_result=False)
    started = []

    result = replace_active_site_runtime(
        old,
        _prepared("new"),
        lambda plan: started.append(plan.profile.site_id),
    )

    assert not result.applied
    assert not result.old_closed
    assert not result.rolled_back
    assert result.runtime is old
    assert "not confirmed" in (result.error or "")
    assert started == []


@pytest.mark.parametrize("failed_resource", ["localizer", "detector", "session_logs"])
def test_replace_runtime_does_not_publish_old_after_dependent_cleanup_failure(
    failed_resource: str,
) -> None:
    old = _active("old", failures=(failed_resource,))
    started = []

    result = replace_active_site_runtime(
        old,
        _prepared("new"),
        lambda plan: started.append(plan.profile.site_id),
    )

    assert not result.applied
    assert result.runtime is None
    assert result.old_closed
    assert not result.rolled_back
    assert failed_resource in (result.error or "")
    assert started == []


def test_replace_runtime_reports_disconnected_state_when_rollback_fails() -> None:
    old = _active("old")
    started = []

    def fail_start(plan):
        started.append(plan.profile.site_id)
        raise RuntimeError(f"{plan.profile.site_id} unavailable")

    result = replace_active_site_runtime(old, _prepared("new"), fail_start)

    assert not result.applied
    assert result.old_closed
    assert not result.rolled_back
    assert result.runtime is None
    assert result.error == "new unavailable"
    assert result.rollback_error == "old unavailable"
    assert started == ["new", "old"]


def _offline_runtime(*, failures=()) -> ActiveSiteRuntime:
    runtime = _active("offline", failures=failures)
    runtime.backend.is_live = False
    runtime.live_backend = None
    runtime.video_stream = _Closer("video_stream", fail="video_stream" in failures)
    return runtime


def test_warm_offline_source_starts_before_outgoing_resources_are_closed() -> None:
    old = _active("old")
    candidate = _offline_runtime()

    def start(_plan):
        assert old.backend.cleanup_calls == 0
        assert old.session_logs.closed == 0
        return candidate

    result = replace_active_site_runtime(old, candidate.prepared, start, start_before_close=True)

    assert result.applied and result.runtime is candidate
    assert old.session_logs.closed == 1
    assert candidate.video_stream.closed == 0


def test_offline_source_start_failure_keeps_current_runtime_untouched() -> None:
    old = _active("old")

    def fail(_plan):
        raise RuntimeError("decoder failed")

    result = replace_active_site_runtime(old, _prepared("new"), fail, start_before_close=True)

    assert result.runtime is old and not result.applied and not result.old_closed
    assert result.error == "decoder failed"
    assert old.backend.cleanup_calls == old.localizer.closed == old.session_logs.closed == 0


@pytest.mark.parametrize("old_usable", [True, False])
def test_double_cleanup_failure_never_republishes_a_closed_outgoing_runtime(old_usable) -> None:
    old = _active("old", cleanup_result=not old_usable, failures=() if old_usable else ("localizer",))
    candidate = _offline_runtime(failures=("detector",))

    result = replace_active_site_runtime(
        old, candidate.prepared, lambda _plan: candidate, start_before_close=True,
    )

    assert not result.applied
    assert result.runtime is old if old_usable else result.runtime is None
    assert result.old_closed is not old_usable
    assert "candidate cleanup" in result.error
    assert candidate.localizer.closed == candidate.detector.closed == candidate.video_stream.closed == candidate.session_logs.closed == 1
    assert old.session_logs.closed == (0 if old_usable else 1)


def test_repeated_source_replacements_close_every_file_stream_and_writer_once() -> None:
    first = _offline_runtime()
    second = _offline_runtime()
    third = _offline_runtime()
    result = replace_active_site_runtime(first, second.prepared, lambda _plan: second, start_before_close=True)
    result = replace_active_site_runtime(result.runtime, third.prepared, lambda _plan: third, start_before_close=True)
    close_active_site_runtime(result.runtime, reason="test_exit")
    for runtime in (first, second, third):
        assert runtime.backend.cleanup_calls == 1
        assert runtime.localizer.closed == runtime.detector.closed == runtime.video_stream.closed == runtime.session_logs.closed == 1


def _mode_ui(kind=RuntimeSwitchKind.NEVER_CONNECTED):
    status = RuntimeSwitchStatus(kind, "unconfirmed airborne", "adapter-address")
    current = _active("old")
    current.backend.state = SimpleNamespace(mode="MANUAL")
    current.backend.runtime_switch_status = lambda: status
    ui = SimpleNamespace(
        backend=current.backend, _site_runtime=current, _site_switching=False,
        _connect_live_busy=False, _flight_inflight=set(),
        _flight_inflight_lock=threading.Lock(), _integrated_auto_active=lambda: False,
        mission_route_lock=SimpleNamespace(active=False, snapshot=None),
        _prepare_mode_runtime=Mock(), _start_site_runtime=Mock(),
        _resolve_simulation_video=lambda *_args: None, _simulation_video=None,
        _mode_switch_selection=None, _site_switch_results=queue.Queue(),
        connect_live_status_var=Mock(), write_log=Mock(),
        _set_site_switch_controls=Mock(), _get_command_coordinator=lambda: SimpleNamespace(suspend=Mock()),
        _detach_localizer_file_handler=Mock(), after=Mock(), _poll_site_switch_result=Mock(),
        route_pts=[(1, 2, 3)], map_center=object(), pivot_point=object(), map_pan=object(),
        field_scale=object(),
    )
    return ui


@pytest.mark.parametrize("kind", [RuntimeSwitchKind.NEVER_CONNECTED, RuntimeSwitchKind.PROVEN_DISCONNECTED, RuntimeSwitchKind.LANDED])
def test_ui_accepts_explicit_adapter_safe_evidence_without_inventing_touchdown(kind) -> None:
    ui = _mode_ui(kind)
    stream_modes.require_runtime_switch_safe(ui)
    assert ui.backend.state.mode == "MANUAL"
    assert not hasattr(ui.backend.state, "touchdown_confirmed")


@pytest.mark.parametrize("blocker", ["airborne", "auto", "command", "connecting"])
def test_mode_transition_rejects_unsafe_or_conflicting_operation(blocker) -> None:
    ui = _mode_ui(RuntimeSwitchKind.UNSAFE if blocker == "airborne" else RuntimeSwitchKind.NEVER_CONNECTED)
    if blocker == "auto":
        ui.mission_route_lock.active = True
    elif blocker == "command":
        ui._flight_inflight.add("takeoff")
    elif blocker == "connecting":
        ui._connect_live_busy = True
    with pytest.raises(ValueError):
        stream_modes.require_runtime_switch_safe(ui)


def test_native_picker_cancellation_leaves_current_mode_and_selection_untouched(monkeypatch) -> None:
    ui = _mode_ui()
    monkeypatch.setattr(stream_modes.filedialog, "askopenfilename", lambda **_kwargs: "")
    current = ui._site_runtime

    stream_modes.request_stream_mode(ui, InterfaceMode.SIMULATED_STREAM, video_directory=Path("/recordings"))

    assert ui._site_runtime is current and not ui._site_switching
    assert ui._mode_switch_selection is None
    ui._prepare_mode_runtime.assert_not_called()
    ui._set_site_switch_controls.assert_not_called()
    assert ui.route_pts == [(1, 2, 3)]


def test_picker_nested_loop_rechecks_safety_before_preparation(monkeypatch) -> None:
    ui = _mode_ui()

    def select(**_kwargs):
        ui.mission_route_lock.active = True
        return "/recordings/test.mp4"

    monkeypatch.setattr(stream_modes.filedialog, "askopenfilename", select)
    stream_modes.request_stream_mode(ui, InterfaceMode.SIMULATED_STREAM, video_directory=Path("/recordings"))

    assert not ui._site_switching
    ui._prepare_mode_runtime.assert_not_called()
    assert "AUTO" in ui.connect_live_status_var.set.call_args.args[0]


def test_failed_source_preparation_retains_original_runtime_and_does_not_remember_video(monkeypatch) -> None:
    ui = _mode_ui()
    video = Path("/recordings/broken.mp4")
    ui._resolve_simulation_video = lambda *_args: video
    ui._prepare_mode_runtime.side_effect = ValueError("recording has no frame")

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target
        def start(self):
            self.target()

    monkeypatch.setattr(stream_modes.threading, "Thread", ImmediateThread)
    picker = Mock(side_effect=AssertionError("remembered source should not open picker"))
    monkeypatch.setattr(stream_modes.filedialog, "askopenfilename", picker)
    stream_modes.request_stream_mode(ui, InterfaceMode.SIMULATED_STREAM, video_directory=Path("/recordings"))
    result = ui._site_switch_results.get_nowait()
    assert result.runtime is ui._site_runtime and not result.applied and not result.old_closed
    assert ui._simulation_video is None
    assert ui._site_runtime.backend.cleanup_calls == 0
    assert ui._mode_switch_selection["route_points"] == ((1, 2, 3),)
    picker.assert_not_called()


def test_mode_only_plan_preserves_explicit_map_route_and_clears_recorded_pose_on_live(monkeypatch, tmp_path) -> None:
    video = tmp_path / "operator-recording.mp4"
    video.write_bytes(b"fixture")
    profile = _identity_profile("current", "c" * 64)
    profile.source = tmp_path / "current.json"
    profile.hardware_approval = None
    monkeypatch.setattr(app, "load_site_profile", lambda _path: profile)
    monkeypatch.setattr(app, "_verify_operator_profile_assets", lambda *_args: None)
    monkeypatch.setattr(app, "profile_for_selected_route", lambda *_args: profile)
    monkeypatch.setattr(
        launch, "selected_route_mission_readiness",
        lambda _evidence, selected, snapshot: SimpleNamespace(
            flight_ready=selected is profile and snapshot is chosen_ref[0],
            flight_errors=(), route_errors=(),
        ),
    )
    chosen_ref = []
    prepared = replace(
        _prepared("current"), profile=profile, map_points=object(),
        args=SimpleNamespace(video="old.mp4", live=False, replay_json="replay.json", mission_flight_ready=True),
        route_points=((99, 99, 99),), replay_rows=({"pose": [99, 99, 99]},),
    )
    chosen = SimpleNamespace(path=tmp_path / "chosen-route.json", verify_file_unchanged=Mock())
    chosen_ref.append(chosen)
    # Exactly one decoded RGB pixel constitutes the source viability check.
    decoder = Mock(return_value=SimpleNamespace(stdout=bytes((0, 1, 2))))
    monkeypatch.setattr(launch.subprocess, "run", decoder)

    offline = launch._prepare_operator_mode_runtime(
        prepared, InterfaceMode.SIMULATED_STREAM, video, route_snapshot=chosen, route_points=((1, 2, 3),),
    )
    live = launch._prepare_operator_mode_runtime(
        offline, InterfaceMode.REAL_FLIGHT, None, route_snapshot=chosen, route_points=((1, 2, 3),),
    )
    assert offline.map_points is prepared.map_points is live.map_points
    assert offline.mission_route_snapshot is chosen is live.mission_route_snapshot
    assert offline.route_points == live.route_points == ((1, 2, 3),)
    assert offline.args.live is False and offline.args.video == str(video)
    assert live.args.live is True and live.args.video == live.args.replay_json == ""
    assert live.replay_rows == ()
    assert live.args.mission_flight_ready is True
    decoder.assert_called_once()
    assert prepared.args.replay_json == "replay.json"


def test_live_replay_loader_never_reads_recorded_pose(monkeypatch) -> None:
    monkeypatch.setattr(launch.json, "loads", Mock(side_effect=AssertionError("replay must not load")))
    assert launch._load_operator_replay(SimpleNamespace(live=True, live_localize=False, replay_json="/replay.json")) == []


def test_live_source_metadata_never_inherits_recording_environment(monkeypatch) -> None:
    monkeypatch.setenv("SFM_SOURCE_INTEGRITY", "KNOWN_INCOMPLETE")
    monkeypatch.setenv("SFM_SOURCE_DECLARED_FRAMES", "2935")
    metadata = launch._source_manifest_metadata(InterfaceMode.REAL_FLIGHT, "")
    assert metadata == {"source_integrity": "LIVE_STREAM", "source_declared_frames": "", "source_decoded_frames": ""}
    assert launch._source_manifest_metadata(InterfaceMode.SIMULATED_STREAM, "a" * 64)["source_integrity"] == "UNVERIFIED"


def test_stale_connect_completion_cannot_replace_new_runtime_or_control_intent() -> None:
    old = _active("old")
    current = _offline_runtime()
    ui = SimpleNamespace(
        _connect_live_results=queue.Queue(), _site_runtime=current, _runtime_generation=2,
        _connect_live_busy=False, video_stream=current.video_stream,
        backend=SimpleNamespace(state=SimpleNamespace(mode="MANUAL", control_owner="PC_FROZEN")),
    )
    ui._connect_live_results.put((old, 1, None))
    app.OperatorApp._poll_live_connect_result(ui)
    assert ui.video_stream is current.video_stream
    assert ui.backend.state.control_owner == "PC_FROZEN"
    assert ui._connect_live_busy is False


def test_real_button_success_remains_retryable_without_claiming_control_ownership() -> None:
    runtime = _active("current")
    state = SimpleNamespace(mode="MANUAL", control_owner="PC_FROZEN")
    ui = SimpleNamespace(
        _connect_live_results=queue.Queue(), _site_runtime=runtime, _runtime_generation=1,
        _connect_live_busy=True, backend=SimpleNamespace(connection_target="adapter-address", state=state),
        _set_widget_enabled=Mock(), connect_live_button=Mock(), connect_live_status_var=Mock(), write_log=Mock(),
    )
    ui._connect_live_results.put((runtime, 1, None))
    app.OperatorApp._poll_live_connect_result(ui)
    assert ui.connect_live_button.configure.call_args.kwargs["state"] == "normal"
    assert ui._connect_live_busy is False
    assert state.control_owner == "PC_FROZEN"
    assert "起飛仍須另行操作" in ui.connect_live_status_var.set.call_args.args[0]


@pytest.mark.parametrize("async_result", [False, True])
def test_rejected_control_command_never_reports_ownership_success(async_result) -> None:
    ui = SimpleNamespace(backend=SimpleNamespace(pilot_sticks=False), control_owner_var=Mock(), write_log=Mock())
    if async_result:
        app.OperatorApp._finish_backend_command(ui, "resume_pc", False, None)
    else:
        app.OperatorApp._finish_sync_operator_command(ui, "resume_pc", False)
    ui.control_owner_var.set.assert_not_called()
    assert "未確認" in ui.write_log.call_args.args[0]


def test_simulation_source_resolver_remembers_chosen_video_and_uses_existing_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VIDEO", raising=False)
    monkeypatch.delenv("SFM_SIMULATION_VIDEO", raising=False)
    directory = tmp_path / "recordings"
    directory.mkdir()
    monkeypatch.setenv("VIDEO_DIR", str(directory))
    prepared = replace(_prepared("current"), args=SimpleNamespace(video=""), profile=None)
    one = directory / "one.mp4"
    one.write_bytes(b"fixture")
    assert launch.resolve_simulation_video(prepared) == one
    two = directory / "two.mp4"
    two.write_bytes(b"fixture")
    assert launch.resolve_simulation_video(prepared) is None
    assert launch.resolve_simulation_video(prepared, remembered=two) == two
    default = directory / "P1190119.MP4"
    default.write_bytes(b"fixture")
    assert launch.resolve_simulation_video(prepared) == default
    assert launch.resolve_simulation_video(prepared, remembered=two) == two


def test_simulation_source_resolver_uses_each_sites_own_recording(monkeypatch, tmp_path) -> None:
    for name in ("VIDEO", "SFM_SIMULATION_VIDEO"):
        monkeypatch.delenv(name, raising=False)
    directory = tmp_path / "recordings"
    directory.mkdir()
    monkeypatch.setenv("VIDEO_DIR", str(directory))
    sites = tmp_path / "sites"
    monkeypatch.setattr(app, "_WS", SimpleNamespace(root=tmp_path, site_packages=sites))

    def prepared_for(relative: str):
        profile = SimpleNamespace(source=sites / relative / "site_profile.json")
        return replace(_prepared("current"), args=SimpleNamespace(video=""), profile=profile)

    expected = {
        "fuhe_bridge": "P1130113.MP4",
        "river_site/releases/river_gluemap_all8_direct_20260908": "P1980198.MP4",
        "wulai_all9_direct_20261019": "P0170017.MP4",
    }
    for name in set(expected.values()):
        (directory / name).write_bytes(b"fixture")
    other_site_video = directory / "other.mp4"
    other_site_video.write_bytes(b"fixture")
    for relative, video in expected.items():
        # The previous site's remembered video must not leak into this site.
        assert launch.resolve_simulation_video(prepared_for(relative), other_site_video) == directory / video

    # A mapped site without its recording asks the operator instead of replaying another site's.
    (directory / "P0170017.MP4").unlink()
    assert launch.resolve_simulation_video(prepared_for("wulai_all9_direct_20261019"), other_site_video) is None
    # A site with no mapping keeps the remembered/configured behaviour.
    assert launch.resolve_simulation_video(prepared_for("new_site"), other_site_video) == other_site_video


def _identity_profile(site_id, map_sha, *, route_sha="a" * 64):
    from site_profile import AssetDigests
    return SimpleNamespace(
        site_id=site_id, coordinate_frame="bound-frame", query_camera="measured-camera",
        asset_sha256=AssetDigests(map_ply=map_sha, route_json=route_sha, map_align="b" * 64),
    )


def test_same_map_authored_route_changes_do_not_discard_independent_mission_readiness(monkeypatch) -> None:
    previous = _identity_profile("map", "c" * 64)
    current = _identity_profile("map", "c" * 64, route_sha="d" * 64)
    base = SimpleNamespace(site_profile="/current.json", mission_selection="/validated.json", mission_flight_ready=True)
    args = SimpleNamespace(**vars(base))
    monkeypatch.setattr(app, "load_site_profile", lambda _path: previous)
    launch._refresh_site_mission_provenance(args, base, current)
    assert args.mission_selection == "/validated.json" and args.mission_flight_ready is True


def test_new_map_cannot_inherit_another_maps_mission_readiness(monkeypatch, tmp_path) -> None:
    previous = _identity_profile("old", "c" * 64)
    current = _identity_profile("new", "d" * 64)
    base = SimpleNamespace(site_profile="/current.json", mission_selection="/old.json", mission_flight_ready=True)
    args = SimpleNamespace(**vars(base))
    monkeypatch.setattr(app, "load_site_profile", lambda _path: previous)
    monkeypatch.setattr(app, "_CTRL_ROOT", tmp_path)
    monkeypatch.setattr(launch, "evaluation_only_requested", lambda: False)
    launch._refresh_site_mission_provenance(args, base, current)
    assert args.mission_selection == args.mission_snapshot_id == ""
    assert args.mission_flight_ready is False
    assert "selected map" in args.mission_flight_errors[0]


def test_recorded_source_delivers_actual_ffmpeg_frames_without_loading_aircraft_sdk(tmp_path) -> None:
    import shutil
    import subprocess
    import sys
    import time
    from aircraft_adapter import CURRENT_AIRCRAFT, build_aircraft_runtime

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("offline FFmpeg fixture requires ffmpeg and ffprobe")
    video = tmp_path / "fixture.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=red:s=16x16:r=30:d=0.2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True, timeout=10, capture_output=True,
    )
    definition = replace(CURRENT_AIRCRAFT, profile=replace(
        CURRENT_AIRCRAFT.profile, aircraft_id="offline-test", stream_width=16, stream_height=16,
    ))
    modules_before = {name for name in sys.modules if name == "olympe" or name.startswith("olympe.")}
    backend, stream, live = build_aircraft_runtime(
        SimpleNamespace(live=False, video=str(video), video_stride=1), None, None, definition=definition,
    )
    runtime = ActiveSiteRuntime(
        prepared=_prepared("fixture"), session_logs=_Logs(), backend=backend,
        video_stream=stream, live_backend=live, localizer=None, detector=None, lost_hold=None,
    )
    process = stream.proc
    try:
        packet = None
        deadline = time.monotonic() + 5
        while packet is None and time.monotonic() < deadline:
            packet = backend.video.next_frame()
            time.sleep(0.01)
        assert packet is not None and packet.rgb.shape == (16, 16, 3)
        assert backend.mode is InterfaceMode.SIMULATED_STREAM and backend.is_live is False
        assert live is None
        assert {name for name in sys.modules if name == "olympe" or name.startswith("olympe.")} == modules_before
    finally:
        close_active_site_runtime(runtime, reason="test_exit")
    assert process.poll() is not None
    assert runtime.session_logs.closed == 1


def test_safety_preemption_discards_warmed_source_and_keeps_outgoing_runtime() -> None:
    old = _active("old")
    candidate = _offline_runtime()
    cancel = threading.Event()

    def start(_plan):
        cancel.set()
        return candidate

    result = replace_active_site_runtime(
        old, candidate.prepared, start, start_before_close=True, cancel_requested=cancel.is_set,
    )
    assert result.runtime is old and not result.applied and not result.old_closed
    assert old.backend.cleanup_calls == 0
    assert candidate.video_stream.closed == candidate.session_logs.closed == 1


def test_fresh_airborne_evidence_after_preparation_prevents_cutover() -> None:
    old = _active("old")
    candidate = _offline_runtime()
    result = replace_active_site_runtime(
        old, candidate.prepared, lambda _plan: candidate,
        start_before_close=True, cutover_allowed=lambda: False,
    )
    assert result.runtime is old and not result.old_closed
    assert old.backend.cleanup_calls == 0 and candidate.video_stream.closed == 1


def test_safety_cancellation_during_teardown_rebuilds_old_mode_not_incoming_mode() -> None:
    old = _active("old")
    candidate = _offline_runtime()
    restored = _active("old")
    cancel = threading.Event()
    original_cleanup = old.backend.cleanup

    def cleanup():
        result = original_cleanup()
        cancel.set()
        return result

    old.backend.cleanup = cleanup
    plans = []

    def start(plan):
        plans.append(plan)
        return candidate if plan is candidate.prepared else restored

    result = replace_active_site_runtime(
        old, candidate.prepared, start, start_before_close=True, cancel_requested=cancel.is_set,
    )
    assert result.runtime is restored and result.rolled_back and not result.applied
    assert plans == [candidate.prepared, old.prepared]
    assert candidate.video_stream.closed == candidate.session_logs.closed == 1


@pytest.mark.parametrize(
    ("command", "payload"),
    [(name, {}) for name in ("manual", "hover", "land", "land_now", "emergency_stop", "nudge_clear")]
    + [("nudge_end", {"dir": "前"})],
)
def test_busy_transition_safety_is_sent_only_to_captured_outgoing_backend(command, payload) -> None:
    outgoing = Mock()
    incoming = Mock()
    coordinator = SimpleNamespace(execute=outgoing.command)
    cancellation = threading.Event()
    ui = SimpleNamespace(
        _site_switching=True, _runtime_available=True, _runtime_switch_cancel=cancellation,
        _get_command_coordinator=lambda: coordinator, write_log=Mock(),
    )
    app.OperatorApp._backend_command(ui, command, payload)
    outgoing.command.assert_called_once_with(command, payload)
    incoming.command.assert_not_called()
    assert cancellation.is_set()


@pytest.mark.parametrize("busy_flag", ["_site_switching", "_connect_live_busy"])
def test_busy_runtime_operation_refuses_ordinary_commands_without_dispatch(busy_flag) -> None:
    execute = Mock()
    ui = SimpleNamespace(
        _runtime_available=True, _runtime_switch_cancel=threading.Event(),
        _get_command_coordinator=lambda: SimpleNamespace(execute=execute), write_log=Mock(),
    )
    setattr(ui, busy_flag, True)
    assert app.OperatorApp._backend_command(ui, "takeoff", {}) is False
    execute.assert_not_called()
    assert not ui._runtime_switch_cancel.is_set()


def test_transition_zero_vector_is_safety_but_nonzero_vector_is_not() -> None:
    ui = SimpleNamespace(_runtime_switch_cancel=threading.Event())
    assert not stream_modes.allow_busy_safety_command(ui, "nudge_vector", {"roll": 1, "pitch": 0, "yaw": 0, "gaz": 0})
    assert stream_modes.allow_busy_safety_command(ui, "nudge_vector", {"roll": 0, "pitch": 0, "yaw": 0, "gaz": 0})
    assert ui._runtime_switch_cancel.is_set()


def test_switching_to_live_requires_the_same_explicit_site_boundary_as_launch() -> None:
    prepared = replace(_prepared("legacy"), profile=None, args=SimpleNamespace(live=False))
    with pytest.raises(ValueError, match="select a map"):
        launch._prepare_operator_mode_runtime(
            prepared, InterfaceMode.REAL_FLIGHT, None, route_snapshot=None, route_points=(),
        )


def test_live_ui_rejects_recorded_pose_before_creating_tk_or_connecting() -> None:
    backend = SimpleNamespace(is_live=True, command=Mock())
    with pytest.raises(ValueError, match="recorded replay"):
        app.OperatorApp.__init__(SimpleNamespace(), backend, None, replay_rows=[{"pose": [1, 2, 3]}])
    backend.command.assert_not_called()


def test_system_exit_from_backend_start_rolls_back_without_republishing_closed_runtime() -> None:
    old = _active("old")
    candidate = _prepared("new")
    restored = _active("old")

    def start(plan):
        if plan is candidate:
            raise SystemExit("start rejected")
        return restored

    result = replace_active_site_runtime(old, candidate, start)
    assert result.runtime is restored and result.rolled_back and result.old_closed
    assert result.error == "start rejected"
    assert old.session_logs.closed == 1

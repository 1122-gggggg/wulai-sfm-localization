"""Same-window source selection over the canonical operator runtime transaction."""
from __future__ import annotations

import argparse
from dataclasses import replace
import threading
from pathlib import Path
from tkinter import filedialog

from operator_route_readiness import selected_route_mission_readiness
from backend_contract import ControlRequest, InterfaceMode, InvalidControlRequest
from operator_command_coordinator import is_safety_request
from operator_site_runtime import SiteRuntimeSwitchResult, replace_active_site_runtime
from last_mission_memory import remember_mission_selection


def allow_busy_safety_command(app, command: str, payload: dict | None = None) -> bool:
    """Admit only canonical safety requests against the captured outgoing adapter."""
    try:
        request = ControlRequest.from_legacy(command, human_origin=True, **(payload or {}))
    except InvalidControlRequest:
        return False
    if not is_safety_request(request):
        return False
    cancellation = app.__dict__.get("_runtime_switch_cancel")
    if cancellation is not None:
        cancellation.set()
    return True


def capture_runtime_selection(app) -> dict:
    return {
        "displayed": app.__dict__.get("_displayed_route_snapshot"),
        "displayed_sha": app.__dict__.get("_displayed_route_sha256"),
        "route_snapshot": getattr(app.mission_route_lock, "snapshot", None),
        "route_points": tuple(app.route_pts),
        "map_view": (app.map_center, app.pivot_point, app.map_pan, app.field_scale),
    }


def record_runtime_route_selection(app, profile, snapshot) -> None:
    """Keep canonical rollback/session inputs aligned with a durable UI selection."""
    runtime = app.__dict__.get("_site_runtime")
    if runtime is None:
        return
    args = argparse.Namespace(**vars(runtime.prepared.args))
    args.route_json = str(snapshot.path)
    readiness = selected_route_mission_readiness(args, profile, snapshot)
    args.mission_flight_ready = readiness.flight_ready
    args.mission_flight_errors = readiness.flight_errors
    args.mission_route_errors = readiness.route_errors
    app._mission_readiness_evidence = args
    runtime.prepared = replace(
        runtime.prepared, args=args, profile=profile, mission_route_snapshot=snapshot,
        route_points=tuple(app.route_pts),
    )


def cutover_is_safe(app, runtime) -> bool:
    with app._flight_inflight_lock:
        if app._flight_inflight:
            return False
    if app.backend is not runtime.backend or str(getattr(runtime.backend.state, "mode", "")).upper() == "AUTO":
        return False
    return runtime.backend.runtime_switch_status().can_switch


def require_runtime_switch_safe(app) -> None:
    """Require adapter evidence; a lost control link is not touchdown evidence."""
    if app._site_switching or app._connect_live_busy:
        raise ValueError("連線或切換正在進行，請等待結果")
    if app.__dict__.get("_shutdown_in_progress", False) or app.__dict__.get("_shutdown_completed", False):
        raise ValueError("介面正在關閉")
    with app._flight_inflight_lock:
        if app._flight_inflight:
            raise ValueError("飛行指令尚未完成，禁止切換")
    lock = app.__dict__.get("mission_route_lock")
    if (lock is not None and lock.active) or app._integrated_auto_active():
        raise ValueError("AUTO 任務進行中，禁止切換")
    if str(getattr(app.backend.state, "mode", "")).upper() == "AUTO":
        raise ValueError("AUTO 模式禁止切換")
    reader = getattr(app.backend, "runtime_switch_status", None)
    if not callable(reader):
        raise ValueError("backend 沒有可驗證的切換狀態")
    status = reader()
    if not status.can_switch:
        raise ValueError(status.reason or "飛行狀態不明或尚未落地，禁止切換")


def _choose_video(app, video_directory: Path) -> Path | None:
    resolver = app._resolve_simulation_video
    video = resolver(app._site_runtime.prepared, app._simulation_video) if resolver is not None else None
    if video is None:
        selected = filedialog.askopenfilename(
            parent=app, title="選擇離線模擬影片", initialdir=str(video_directory),
            filetypes=[("影片", "*.mp4 *.MP4 *.mov *.MOV *.mkv *.avi"), ("所有檔案", "*")],
        )
        if not selected:
            return None
        video = Path(selected).expanduser().resolve()
    # A native picker runs a nested Tk event loop; safety may have changed.
    require_runtime_switch_safe(app)
    return video


def request_stream_mode(app, mode: InterfaceMode, *, video_directory: Path) -> None:
    """Prepare before cutover; cancellation/failure never claims a new mode."""
    try:
        require_runtime_switch_safe(app)
        if app._site_runtime is None or app._prepare_mode_runtime is None or app._start_site_runtime is None:
            raise ValueError("runtime 切換尚未設定")
        if app.__dict__.get("_route_editor_window") is not None or app.__dict__.get("_route_editor_loading", False):
            raise ValueError("請先關閉航線編輯器再切換串流")
        video = _choose_video(app, video_directory) if mode is InterfaceMode.SIMULATED_STREAM else None
        if mode is InterfaceMode.SIMULATED_STREAM and video is None:
            return
        old = app._site_runtime
        snapshot = getattr(app.mission_route_lock, "snapshot", None)
        points = tuple(app.route_pts)
        app._mode_switch_selection = capture_runtime_selection(app)
        app._mode_switch_selection.update(video=video, requested_mode=mode)
        cancellation = threading.Event()
        app._runtime_switch_cancel = cancellation
        app._site_switching = True
        app._set_site_switch_controls(busy=True)
        app._get_command_coordinator().suspend()
        app._detach_localizer_file_handler()
        results = app._site_switch_results

        def worker() -> None:
            try:
                plan = app._prepare_mode_runtime(
                    old.prepared, mode, video, route_snapshot=snapshot, route_points=points,
                )
                result = replace_active_site_runtime(
                    old, plan, app._start_site_runtime,
                    start_before_close=mode is InterfaceMode.SIMULATED_STREAM,
                    cancel_requested=cancellation.is_set,
                    cutover_allowed=lambda: cutover_is_safe(app, old),
                )
            except BaseException as exc:
                result = SiteRuntimeSwitchResult(
                    runtime=old, applied=False, old_closed=False, error=str(exc) or repr(exc),
                )
            results.put(result)

        try:
            threading.Thread(target=worker, name="operator-stream-switch", daemon=True).start()
        except Exception as exc:
            results.put(SiteRuntimeSwitchResult(
                runtime=old, applied=False, old_closed=False, error=str(exc) or repr(exc),
            ))
        app.after(50, app._poll_site_switch_result)
    except Exception as exc:
        message = f"串流模式未切換：{exc}"
        app.connect_live_status_var.set(message)
        app.write_log(message)


def _activate_runtime_profile(app, runtime, site_root_for_profile) -> None:
    profile = runtime.prepared.profile
    if profile is not None:
        app.site_assets_panel.activate_site(profile.source, site_root_for_profile(profile.source))


def _install_switch_outcome(app, result, selection, site_root_for_profile) -> None:
    if result.old_closed:
        app._get_command_coordinator().drain(app._finish_backend_command)
    if result.runtime is not None and (result.applied or result.rolled_back):
        if result.rolled_back and selection is None:
            app._mode_switch_selection = app.__dict__.get("_site_switch_previous_selection")
        app._install_site_runtime(result.runtime)
        if result.rolled_back and selection is None:
            _activate_runtime_profile(app, result.runtime, site_root_for_profile)
    elif result.runtime is app._site_runtime and not result.old_closed:
        app._attach_localizer_file_handler()
        app._get_command_coordinator().resume()
        if selection is None:
            _activate_runtime_profile(app, result.runtime, site_root_for_profile)


def _switch_result_message(app, result, selection) -> str:
    if result.applied:
        if selection is not None:
            offline = result.runtime.prepared.interface_mode is InterfaceMode.SIMULATED_STREAM
            if offline:
                app._simulation_video = selection["video"]
            return (
                "離線模擬已啟動（SIM）：同一地圖與航線，沒有真機連線"
                if offline else "已切回真機串流；正在以既有流程連接，不會起飛"
            )
        message = f"已切換至 {result.runtime.prepared.profile.display_name}；runtime 已重建，真機需另行連線"
        if app._is_live_backend() and not app._mission_flight_ready:
            errors = getattr(result.runtime.prepared.args, "mission_flight_errors", ())
            message += "；AUTO 任務守門：" + "；".join(errors)
        return message
    if result.rolled_back:
        return f"切換失敗，舊模式 runtime 已重建（真機需重新連線）：{result.error}"
    if result.runtime is not None:
        return f"切換已取消，原模式保留：{result.error}"
    return (
        f"新 runtime 失敗，舊模式也無法重建：{result.error}；"
        f"回復失敗：{result.rollback_error}。介面保持開啟，飛行控制已停用，可重新選擇場域／串流重試。"
    )


def finish_runtime_switch(app, result, *, site_root_for_profile, close_recorder) -> None:
    """Present only the committed outcome, and invalidate outgoing session data."""
    selection = app.__dict__.get("_mode_switch_selection")
    _install_switch_outcome(app, result, selection, site_root_for_profile)
    if result.applied and result.runtime is not None:
        remember_mission_selection(getattr(result.runtime.prepared.args, "mission_selection", ""))
    app._site_switching = False
    app._runtime_available = result.runtime is not None
    if result.runtime is None:
        close_recorder(app.__dict__.get("imu_flight_test"))
        app.imu_flight_test = None
        app.video_frame = None
        app.live_result = app.detection_result = None
        canvas = app.__dict__.get("video_label")
        if canvas is not None:
            canvas.delete("video-frame", "video-overlay")
    message = _switch_result_message(app, result, selection)
    app.write_log(message)
    app.site_assets_panel.set_status(message)
    status = app.__dict__.get("connect_live_status_var")
    if status is not None:
        status.set(message)
    if result.applied and selection is None:
        app.after(300, app._check_active_site_route)
    app._mode_switch_selection = None
    app.__dict__.pop("_site_switch_previous_selection", None)
    app.__dict__.pop("_runtime_switch_cancel", None)
    saved_route = app.__dict__.pop("_route_to_select_after_site_switch", None)
    if result.applied and saved_route is not None:
        app.preview_site_route(saved_route)
    app._set_site_switch_controls(busy=False)
    if result.applied and selection is not None and selection["requested_mode"] is InterfaceMode.REAL_FLIGHT:
        runtime = result.runtime
        generation = app.__dict__.get("_runtime_generation")

        def connect_selected_runtime() -> None:
            if app._site_runtime is runtime and app.__dict__.get("_runtime_generation") == generation:
                app.connect_live_stream()

        app.after(0, connect_selected_runtime)

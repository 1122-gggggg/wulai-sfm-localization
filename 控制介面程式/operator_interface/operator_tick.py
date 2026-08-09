"""Tick-cycle phases for the desktop operator interface.

The Tk callback is intentionally a small coordinator.  These helpers keep the
existing phase order visible while giving the stream, state, HUD, and render
paths independently testable boundaries.  ``app`` is an explicit dependency so
this module does not become an implicit-self mixin.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from backend_contract import FailureReason


RollingEventFps = Callable[[list[float], float], tuple[list[float], float]]
NextTickDeadline = Callable[[float, float, float], tuple[float, int]]
StreamTerminalText = Callable[[object], str | None]


def _refresh_held_controls(app: Any) -> None:
    """Refresh held movement TTLs while Tk is still receiving input."""
    active_nudges = app._active_nudge_directions()
    if active_nudges and app._is_live_backend():
        # Backend TTL is refreshed only while Tk continues to observe a
        # physical hold. A frozen UI therefore decays to zero PCMD.
        try:
            app._backend_command("nudge_heartbeat", {"dirs": active_nudges})
        except Exception as exc:
            app._nudge_keys_held.clear()
            app._nudge_buttons_held.clear()
            app.write_log(f"微移 heartbeat 失敗，已清除輸入: {exc!r}")
    if app._stick_vector_active and app._is_live_backend():
        # Same contract as the button heartbeat: a frozen UI stops refreshing
        # the backend TTL, so a held stick decays to zero PCMD on its own.
        app._send_stick_vector(app.stick_left.value, app.stick_right.value)


def _schedule_after_poll_failure(
    app: Any, *, next_tick_deadline: NextTickDeadline,
) -> None:
    now = time.monotonic()
    app._next_tick_deadline, delay_ms = next_tick_deadline(
        getattr(app, "_next_tick_deadline", 0.0),
        now,
        getattr(app, "_tick_period_s", 0.1),
    )
    app.after(delay_ms, app.tick)


def _poll_backend(
    app: Any, *, next_tick_deadline: NextTickDeadline,
) -> Any | None:
    """Poll telemetry and retain the original fail-safe/schedule contract."""
    try:
        return app.backend.poll()
    except Exception as exc:
        incident = getattr(getattr(app, "session_logs", None), "incident", None)
        if callable(incident):
            try:
                incident(
                    "backend_poll_failed",
                    error=repr(exc),
                    resolved=False,
                )
            except Exception:
                pass
        try:
            app.write_log(f"BACKEND_POLL_FAILED: {exc!r}")
        except Exception:
            pass
        try:
            app.backend.fail_safe(FailureReason.INVALID_TELEMETRY)
        except Exception as safe_exc:
            if callable(incident):
                try:
                    incident(
                        "backend_poll_fail_safe_failed",
                        error=repr(safe_exc),
                        resolved=False,
                    )
                except Exception:
                    pass
        _schedule_after_poll_failure(app, next_tick_deadline=next_tick_deadline)
        return None


def _handle_stick_override(app: Any, state: Any) -> None:
    """Reflect a physical SkyController handback in the operator HUD."""
    if not app._is_live_backend() or not hasattr(app, "control_owner_var"):
        return
    sticks_now = bool(getattr(app.backend, "pilot_sticks", False))
    prev = bool(getattr(app, "_prev_pilot_sticks", sticks_now))
    if sticks_now and not prev:
        count = int(getattr(app.backend, "stick_override_count", 0) or 0)
        last_cmd = str(getattr(state, "last_command", "") or "")
        if last_cmd == "stick_override" or count > 0:
            app.write_log("搖桿輸入偵測：控制權已強制交回 SkyController 搖桿")
            app.control_owner_var.set(
                "控制權: 搖桿 (動搖桿強制交回) — 按「恢復電腦控制」拿回"
            )
            app.backend.state.stream = "STICKS"
    elif (not sticks_now) and prev:
        app.control_owner_var.set("控制權: 電腦 (LIVE) — 動搖桿立即交回")
    app._prev_pilot_sticks = sticks_now


def _read_next_stream_frame(
    app: Any, *, now: float, hold_boot: bool, hold_lost: bool,
) -> Any | None:
    """Read one frame using the live latest-frame or replay pacing contract."""
    if app._is_live_backend():
        can_read = not hold_boot or app.video_frame is None
        if can_read:
            try:
                return app.video_stream.next_frame(only_new=True)
            except TypeError:
                return app.video_stream.next_frame()
        return None

    stream_due = now >= app.next_stream_frame_time
    can_read = stream_due and (
        not (hold_boot or hold_lost) or app.video_frame is None
    )
    if not can_read:
        return None
    frame = app.video_stream.next_frame()
    if app.next_stream_frame_time <= 0.0:
        app.next_stream_frame_time = now
    while app.next_stream_frame_time <= now:
        app.next_stream_frame_time += app.stream_period_s
    return frame


def _accept_stream_frame(
    app: Any,
    state: Any,
    frame: Any,
    *,
    now: float,
    hold_boot: bool,
    rolling_event_fps: RollingEventFps,
) -> Any:
    app.video_frame = frame
    app.video_display_index = max(0, app.video_stream.output_index - 1)
    app.video_display_frame_name = app.video_stream.last_frame_name
    app._video_frame_stamp = float(
        getattr(app.video_stream, "last_stamp", now) or now
    )
    app._video_frame_timing = dict(
        getattr(app.video_stream, "last_timing", {}) or {}
    )
    app.video_frame_fresh = not hold_boot
    app.stream_lost_since = None
    if hold_boot:
        state.stream = "HOLD_720P"
    elif app.inspecting:
        state.stream = "OK"
        # Rolling 5s stream FPS (truth for "is the pipe real-time now?")
        # plus lifetime average for long-run stats.
        app.processed_frames += 1
        app._stream_frame_times.append(now)
        app._stream_frame_times, app.stream_fps_instant = rolling_event_fps(
            app._stream_frame_times, now
        )
        app.overall_fps = app.stream_fps_instant
        if app.inspect_start is not None:
            elapsed = now - app.inspect_start
            if elapsed > 0:
                app._lifetime_stream_fps = app.processed_frames / elapsed
    else:
        state.stream = "PREVIEW"   # live preview, not yet in inspection
    return state


def _handle_missing_stream_frame(
    app: Any,
    state: Any,
    *,
    now: float,
    hold_boot: bool,
    hold_lost: bool,
) -> Any:
    """Apply the intentional hold, EOF, transport, and stale-frame states."""
    # only_new=True often returns None between 30 Hz frames — that is NOT
    # stream loss. Only declare LOST when we have no recent frame at all.
    if hold_boot and app.video_frame is not None:
        state.stream = "HOLD_720P"
    elif hold_lost and app.video_frame is not None:
        # Paused on purpose: the held frame is stale by design, so the
        # staleness failsafe below must not read it as transport loss.
        state.stream = "LOST_HOLD"
    elif (
        not app._is_live_backend()
        and bool(getattr(app.video_stream, "eof", False))
        and app.video_frame is not None
    ):
        state.stream = str(
            getattr(app.video_stream, "terminal_state", "EOF_HOLD")
        )
        app.stream_lost_since = None
    elif app.inspecting and app._is_live_backend():
        return _handle_live_missing_frame(app, state)
    elif app.inspecting:
        return _handle_replay_missing_frame(app, state, now=now)
    elif app.video_frame is not None:
        state.stream = "PREVIEW"
    else:
        state.stream = "WAIT"
    return state


def _handle_live_missing_frame(app: Any, state: Any) -> Any:
    # Live transport health is decided by the backend from the decoder's
    # newest frame age and duplicate-frame counter. The UI may deliberately
    # hold or render an older frame while localization is busy; that is not a
    # stream fault.
    if (
        str(getattr(state, "active_incident", ""))
        == FailureReason.STREAM_STALE.value
    ):
        return app.backend.state
    state.stream = "OK"
    return state


def _handle_replay_missing_frame(app: Any, state: Any, *, now: float) -> Any:
    age_s = (
        now - float(app._video_frame_stamp)
        if app.video_frame is not None and app._video_frame_stamp > 0
        else 1e9
    )
    # stale_s grabber default ~0.35; allow a bit more before hover failsafe
    if app.video_frame is not None and age_s < 0.75:
        state.stream = "OK"
        app.stream_lost_since = None
        # Hold last rolling estimate; do not inflate with fake frames.
        return state
    if app.stream_lost_since is None:
        app.stream_lost_since = time.monotonic()
        app.backend.stream_lost_hover(f"720p frame stale age_s={age_s:.2f}")
        app.write_log(f"STREAM_LOST_HOVER: 720p frame stale age_s={age_s:.2f}")
    return app.backend.state


def _update_stream(
    app: Any, state: Any, *, rolling_event_fps: RollingEventFps,
) -> Any:
    """Pull and classify the right-hand stream without changing flight state."""
    app.video_frame_fresh = False
    if app.video_stream is None:
        if app.video_frame is None:
            state.stream = "NO_SOURCE"
        return state
    now = time.monotonic()
    hold_boot = app.inspecting and app.boot_holding()
    # LOST: freeze the file stream on the current frame so MegaLoc reacquires
    # from the scene a hovering aircraft would still be looking at.
    hold_lost = app.lost_holding()
    frame = _read_next_stream_frame(
        app, now=now, hold_boot=hold_boot, hold_lost=hold_lost
    )
    if frame is not None:
        return _accept_stream_frame(
            app,
            state,
            frame,
            now=now,
            hold_boot=hold_boot,
            rolling_event_fps=rolling_event_fps,
        )
    return _handle_missing_stream_frame(
        app, state, now=now, hold_boot=hold_boot, hold_lost=hold_lost
    )


def _update_state_and_history(app: Any, state: Any) -> Any:
    app.submit_current_frame_for_localization()
    app.submit_current_frame_for_detection()
    if app.localizer is not None:
        state = app.state_from_live(state)
    else:
        state = app.state_from_replay(state)
    app.current_state = state
    app.update_anafi_metrics(state)
    if (app.localizer is not None and app.live_new_pose) or (app.localizer is None):
        app.history.append(state.pose.copy())
        app.history_health.append(getattr(app, "loc_health", "OK"))
        if len(app.history) > 300:
            app.history.pop(0)
            if app.history_health:
                app.history_health.pop(0)
    if app.localizer is not None:
        # A result may have arrived on the independent 5 ms poll callback.
        # Consume its one-shot history notification only after this render tick.
        app.live_new_pose = False
    app._update_age_readout(state)
    return state


def _update_hud_startup(app: Any) -> None:
    localizer_ready = (
        app.localizer is None
        or bool(getattr(app.localizer, "ready", True))
    )
    if app.localizer is not None and not localizer_ready:
        startup_error = getattr(app.localizer, "startup_error", None)
        text = "定位 worker 暖機重試中" if startup_error else "定位模型暖機中"
        app._set_loc_health_display(text=text, colour="#b26a00")
    elif (
        app.localizer is not None
        and not app.live_result_frame_name
        and not (app._loc_ok_count or app._loc_fail_count)
    ):
        info = getattr(app.localizer, "startup_info", {})
        device = info.get("device") if isinstance(info, dict) else None
        text = "定位就緒，等待首筆結果" if app.inspecting else "定位就緒，待開始"
        if device:
            text += f" ({str(device).upper()})"
        color = "#24733f" if device in (None, "cuda") else "#b26a00"
        app._set_loc_health_display(text=text, colour=color)


def _update_hud_idle(app: Any) -> None:
    app.overall_fps_var.set("串流 FPS - (待命，按開始定位)")
    if hasattr(app, "loc_fps_var"):
        app.loc_fps_var.set("定位 FPS -")
    if hasattr(app, "loc_latency_var"):
        app.loc_latency_var.set("wall_ms - | core - | e2e -")
    if hasattr(app, "loc_quality_var"):
        app.loc_quality_var.set("inliers -")
    if hasattr(app, "loc_recovery_var"):
        app.loc_recovery_var.set(app.loc_recovery_text)


def _update_hud_active(
    app: Any,
    state: Any,
    *,
    rolling_event_fps: RollingEventFps,
    stream_terminal_text: StreamTerminalText,
) -> None:
    now_hud = time.monotonic()
    app._stream_frame_times, app.stream_fps_instant = rolling_event_fps(
        app._stream_frame_times, now_hud
    )
    app.live_result_times, app.loc_fps = rolling_event_fps(
        app.live_result_times, now_hud
    )
    age_ms = (
        (now_hud - float(app._video_frame_stamp)) * 1000.0
        if app._video_frame_stamp > 0 else -1.0
    )
    age_txt = f"{age_ms:.0f}" if age_ms >= 0 else "-"
    terminal_text = (
        None if app._is_live_backend() else stream_terminal_text(state.stream)
    )
    if terminal_text is not None:
        app.overall_fps_var.set(
            f"整體/串流 FPS 0.0 | {terminal_text} | 幀 {app.processed_frames}"
        )
        if hasattr(app, "loc_fps_var"):
            app.loc_fps_var.set("定位 FPS 0.0")
        if hasattr(app, "loc_latency_var"):
            app.loc_latency_var.set("wall_ms - | core - | e2e -")
    else:
        # Primary overall metric = 5s rolling stream FPS only.
        app.overall_fps = float(app.stream_fps_instant)
        life = float(getattr(app, "_lifetime_stream_fps", 0.0) or 0.0)
        app.overall_fps_var.set(
            f"整體/串流 FPS {app.stream_fps_instant:.1f} (5s滾動) | "
            f"stream_age {age_txt}ms | 累計 {life:.1f} | 幀 {app.processed_frames}"
        )
        # Refresh wall/core/age every tick (not only on new loc result).
        if hasattr(app, "loc_fps_var"):
            app.loc_fps_var.set(f"定位 FPS {app.loc_fps:.1f}")
    if terminal_text is None and hasattr(app, "loc_latency_var"):
        wall_txt = f"{app.loc_wall_ms:.1f}" if app.loc_wall_ms is not None else "-"
        core_txt = f"{app.loc_latency_ms:.1f}" if app.loc_latency_ms is not None else "-"
        e2e_txt = f"{app.loc_e2e_ms:.1f}" if app.loc_e2e_ms is not None else "-"
        app.loc_latency_var.set(
            f"wall_ms {wall_txt}ms | core {core_txt}ms | e2e {e2e_txt}ms"
        )
    if hasattr(app, "loc_recovery_var"):
        app.loc_recovery_var.set(app.loc_recovery_text)


def _update_hud_metrics(
    app: Any,
    state: Any,
    *,
    rolling_event_fps: RollingEventFps,
    stream_terminal_text: StreamTerminalText,
) -> None:
    if not hasattr(app, "overall_fps_var"):
        return
    _update_hud_startup(app)
    if not app.inspecting:
        _update_hud_idle(app)
        return
    _update_hud_active(
        app,
        state,
        rolling_event_fps=rolling_event_fps,
        stream_terminal_text=stream_terminal_text,
    )


def _map_dirty_key(app: Any, state: Any, *, width: int, height: int) -> tuple:
    pose_key = tuple(
        round(float(value), 4) if np.isfinite(value) else None
        for value in np.asarray(state.pose, dtype=float)
    )
    camera_forward_key = (
        tuple(round(float(value), 4) for value in app.camera_forward_world)
        if app.camera_forward_world is not None else None
    )
    camera_axes_key = (
        tuple(round(float(value), 4) for value in app.camera_axes_world.reshape(-1))
        if app.camera_axes_world is not None else None
    )
    return (
        app.map_base_key(width, height),
        len(app.history),
        id(app.history[-1]) if app.history else 0,
        id(app.history[0]) if app.history else 0,
        len(app.route_pts),
        app.route_visible,
        len(app.no_loc_markers),
        pose_key,
        camera_axes_key,
        camera_forward_key,
        int(state.inliers),
        None if state.reproj is None else round(float(state.reproj), 4),
    )


def _video_dirty_key(app: Any, state: Any, *, width: int, height: int) -> tuple:
    return (
        # stamp changes only on new grabber frame (avoids work on idle ticks)
        round(float(getattr(app, "_video_frame_stamp", 0.0)), 4),
        width,
        height,
        app.inspecting,
        app.loc_health,
        app.loc_health_inliers,
        None if app.loc_health_reproj is None else round(float(app.loc_health_reproj), 4),
        id(app.detection_result),
        app.video_display_index,
        # Held frames stop changing the stamp; repaint the retry counter anyway.
        app.lost_holding(),
        0 if app.lost_hold is None else app.lost_hold.attempts,
        # Re-render when link/GPS/age band changes (LINK LOST banner + HUD).
        bool(getattr(state, "link_ok", True)),
        getattr(state, "gps_fixed", None),
        None if getattr(state, "frame_age_ms", None) is None
        else int(float(state.frame_age_ms) // 50),  # ~50 ms buckets
        int(float(getattr(state, "stream_fps", 0.0) or 0.0)),
        round(float(state.battery_pct), 0),
        app._video_diagnostic_lines(),
    )


def _render_if_dirty(app: Any, state: Any) -> None:
    width_map = max(300, app.map_label.winfo_width())
    height_map = max(220, app.map_label.winfo_height())
    width_video = max(300, app.video_label.winfo_width())
    height_video = max(220, app.video_label.winfo_height())
    # Map: re-render only when the view, flown history, drawn route or pose/quality
    # readouts changed. Between live fixes these are all static, so reuse the PhotoImage.
    map_key = _map_dirty_key(
        app, state, width=width_map, height=height_map
    )
    if map_key != app._map_dirty_key:
        app._map_dirty_key = map_key
        app._present_frame(
            app.map_label, "map_photo", app.render_map(width_map, height_map, state)
        )
    # Video: re-render only when the source frame, panel size, the localization
    # alert banner inputs or the detection overlay changed. The expensive 720p resize
    # + PhotoImage is skipped on ticks where the same frame is shown again.
    video_key = _video_dirty_key(
        app, state, width=width_video, height=height_video
    )
    if video_key != app._video_dirty_key:
        app._video_dirty_key = video_key
        app._present_frame(
            app.video_label,
            "video_photo",
            app.render_video(width_video, height_video, state),
        )


def _schedule_next_tick(app: Any, *, next_tick_deadline: NextTickDeadline) -> None:
    now = time.monotonic()
    app._next_tick_deadline, delay_ms = next_tick_deadline(
        app._next_tick_deadline, now, app._tick_period_s
    )
    app.after(delay_ms, app.tick)


def run_tick(
    app: Any,
    *,
    rolling_event_fps: RollingEventFps,
    next_tick_deadline: NextTickDeadline,
    stream_terminal_text: StreamTerminalText,
) -> None:
    """Run one UI cycle in the same order as the legacy callback."""
    app._drain_flight_command_results()
    app._drain_integrated_autonomy_events()
    _refresh_held_controls(app)
    state = _poll_backend(app, next_tick_deadline=next_tick_deadline)
    if state is None:
        return
    _handle_stick_override(app, state)
    app._gravity_tick(state)
    app.video_frame_fresh = False
    app.update_boot_lock()
    app.update_live_results()
    app.update_lost_hold()
    app.update_detection_results()
    state = _update_stream(app, state, rolling_event_fps=rolling_event_fps)
    state = _update_state_and_history(app, state)
    _update_hud_metrics(
        app,
        state,
        rolling_event_fps=rolling_event_fps,
        stream_terminal_text=stream_terminal_text,
    )
    _render_if_dirty(app, state)
    _schedule_next_tick(app, next_tick_deadline=next_tick_deadline)

"""Tick-cycle phases for the desktop operator interface.

The Tk callback is intentionally a small coordinator.  These helpers keep the
existing phase order visible while giving the stream, state, HUD, and render
paths independently testable boundaries.  ``app`` is an explicit dependency so
this module does not become an implicit-self mixin.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable

import numpy as np

from backend_contract import FailureReason
from operator_state import TrackerState
from operator_localization_config import (
    LIVE_STATUS_PATH,
    POSE_JUMP_U,
    normalize_camera_axes,
    normalize_camera_forward,
)
from localization_result_ui import (
    annotate_ui_arrival_timing,
    normalize_live_localization_result,
    validate_live_localization_result,
)


RollingEventFps = Callable[[list[float], float], tuple[list[float], float]]
NextTickDeadline = Callable[[float, float, float], tuple[float, int]]
StreamTerminalText = Callable[[object], str | None]


def _rolling_event_fps(
    event_times: list[float], now: float, window_s: float = 5.0,
) -> tuple[list[float], float]:
    """Return recent event timestamps and their observed delivery rate."""
    cutoff = float(now) - float(window_s)
    recent = [stamp for stamp in event_times if stamp >= cutoff]
    if len(recent) < 2:
        return recent, 0.0
    span = recent[-1] - recent[0]
    return recent, (len(recent) - 1) / span if span > 0.0 else 0.0


def stream_terminal_text(stream_state: object) -> str | None:
    return {
        "EOF_HOLD": "影片已播完，保留最後一幀",
        "DECODE_ERROR_HOLD": "解碼錯誤，保留最後一幀",
    }.get(str(stream_state or ""))


def next_tick_deadline(
        previous_deadline: float, now: float, period_s: float) -> tuple[float, int]:
    """Advance a fixed-rate UI deadline without accumulating callback runtime."""
    deadline = float(previous_deadline)
    period = max(0.001, float(period_s))
    if deadline <= 0.0:
        deadline = float(now) + period
    while deadline <= now:
        deadline += period
    delay_ms = max(1, int(math.ceil((deadline - now) * 1000.0 - 1e-9)))
    return deadline, delay_ms


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


def _fail_safe_dead_localizer(app: Any) -> None:
    localizer = getattr(app, "localizer", None)
    if localizer is None or not getattr(localizer, "unavailable", False):
        return
    if getattr(app, "_worker_exit_fail_safe_done", False):
        return
    fail_safe = getattr(getattr(app, "backend", None), "fail_safe", None)
    if not callable(fail_safe):
        return
    fail_safe(FailureReason.WORKER_EXIT)
    app._worker_exit_fail_safe_done = True
    write_log = getattr(app, "write_log", None)
    if callable(write_log):
        write_log("WORKER_EXIT: 定位 worker 已不可用，已 fail-safe")


def _poll_backend(app: Any) -> Any | None:
    """Poll telemetry; on failure hold zero without changing control owner."""
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
            except Exception:  # Tier3: incident logging best-effort — keep fallback
                pass
        try:
            app.write_log(f"BACKEND_POLL_FAILED: {exc!r}")
        except Exception:  # Tier3: log sink best-effort — keep fallback
            pass
        pause_auto = getattr(app, "_pause_integrated_auto", None)
        if callable(pause_auto):
            try:
                pause_auto("backend_poll_failed")
            except Exception:  # Tier3: pause AUTO best-effort — keep fallback
                pass
        try:
            clear_motion = getattr(app.backend, "nudge_clear", None)
            if callable(clear_motion):
                clear_motion(reason="backend_poll_failed")
            send_pcmd = getattr(app.backend, "send_pcmd", None)
            if callable(send_pcmd):
                hovered = bool(send_pcmd(
                    0, 0, 0, 0, reason="backend_poll_failed_hover"
                ))
            else:
                state = getattr(app.backend, "state", None)
                if state is not None:
                    state.tracker_state = TrackerState.HOVER
                    state.last_command = "backend_poll_failed_hover"
                hovered = True
            if not hovered:
                raise RuntimeError("zero PCMD was rejected")
        except Exception as safe_exc:
            if callable(incident):
                try:
                    incident(
                        "backend_poll_hover_failed",
                        error=repr(safe_exc),
                        resolved=False,
                    )
                except Exception:  # Tier3: incident logging best-effort — keep fallback
                    pass
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
        pause_auto = getattr(app, "_pause_integrated_auto", None)
        if callable(pause_auto):
            try:
                pause_auto("stream_stale")
            except Exception:  # Tier3: fallback best-effort — keep pass
                pass
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
    guard = getattr(app, "collision_guard_snapshot", {}) or {}
    guard_center = guard.get("center")
    guard_point = guard.get("point")
    guard_key = (
        str(guard.get("status", "")),
        round(float(guard.get("radius", 0.0) or 0.0), 4),
        tuple(round(float(value), 4) for value in guard_center)
        if guard_center is not None else None,
        tuple(round(float(value), 4) for value in guard_point)
        if guard_point is not None else None,
        bool(guard.get("preview", False)),
    )
    return (
        app.map_base_key(width, height),
        len(app.history),
        id(app.history[-1]) if app.history else 0,
        id(app.history[0]) if app.history else 0,
        len(app.route_pts),
        len(app.no_loc_markers),
        pose_key,
        camera_axes_key,
        camera_forward_key,
        int(state.inliers),
        None if state.reproj is None else round(float(state.reproj), 4),
        guard_key,
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


def _record_tick_failure(app: Any, *, stage: str, error: str) -> None:
    incident = getattr(getattr(app, "session_logs", None), "incident", None)
    if callable(incident):
        try:
            incident(
                "operator_tick_failed",
                stage=stage,
                error=error,
                resolved=False,
            )
        except Exception as incident_exc:
            try:
                app.write_log(
                    f"OPERATOR_TICK_INCIDENT_FAILED stage={stage} "
                    f"error={incident_exc!r}"
                )
            except Exception:  # Tier3: fallback best-effort — keep pass
                pass
    try:
        app.write_log(f"OPERATOR_TICK_FAILED stage={stage} error={error}")
    except Exception:  # Tier3: fallback best-effort — keep pass
        pass


def _handle_tick_failure(app: Any, *, stage: str, exc: Exception) -> None:
    """Record a failed UI stage and use the existing AUTO pause seam."""
    error = repr(exc)
    _record_tick_failure(app, stage=stage, error=error)
    reason = f"operator_tick_failed:{stage}"
    pause_auto = getattr(app, "_pause_integrated_auto", None)
    if callable(pause_auto):
        try:
            pause_auto(reason)
        except Exception as safety_exc:
            try:
                app.write_log(
                    f"OPERATOR_TICK_AUTO_PAUSE_FAILED stage={stage} "
                    f"error={safety_exc!r}"
                )
            except Exception:  # Tier3: fallback best-effort — keep pass
                pass
        return

    # Narrow test doubles or older integrations may expose only the backend's
    # existing fail-safe seam. Do not synthesize a real-flight command here.
    fail_safe = getattr(getattr(app, "backend", None), "fail_safe", None)
    if callable(fail_safe):
        try:
            fail_safe(FailureReason.INVALID_TELEMETRY)
        except Exception as safety_exc:
            try:
                app.write_log(
                    f"OPERATOR_TICK_FAIL_SAFE_FAILED stage={stage} "
                    f"error={safety_exc!r}"
                )
            except Exception:  # Tier3: fallback best-effort — keep pass
                pass


def _publish_live_result_state(app: Any, result: dict) -> None:
    app.live_result = result
    app.live_result_frame_name = str(result.get("frame_name", ""))
    app._apply_lost_hold_result(result)
    app.update_localization_metrics(result)
    now = time.monotonic()
    if now - app._last_status_write < 0.2:
        return
    app._last_status_write = now
    try:
        LIVE_STATUS_PATH.write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        app._record_diagnostic_failure(LIVE_STATUS_PATH, exc)


def _live_result_failed(app: Any, result: dict, *, invalid_success_pose: bool) -> bool:
    if result.get("success") and result.get("pose"):
        return False
    if invalid_success_pose:
        app.loc_health = "FAIL"
        app._record_no_loc()
    now = time.monotonic()
    if now - app._last_loc_fail_log >= 2.0:
        app._last_loc_fail_log = now
        error = result.get("error", "no pose")
        app.write_log(
            f"LIVE_LOCALIZE_FAIL (throttled) last={app.live_result_frame_name} "
            f"err={error} | ok={app._loc_ok_count} fail={app._loc_fail_count} "
            f"loc_fps={app.loc_fps:.1f} wall_ms={result.get('wall_ms')}"
        )
    return True


def _live_pose_is_continuous(app: Any, xyz: np.ndarray) -> bool:
    if app.live_last_xyz is None:
        app._live_pending = None
        return True
    jump = float(np.linalg.norm(xyz - app.live_last_xyz))
    if jump <= POSE_JUMP_U:
        app._live_pending = None
        return True
    if (
        app._live_pending is not None
        and float(np.linalg.norm(xyz - app._live_pending)) <= POSE_JUMP_U
    ):
        app._live_pending = None
        return True
    app._live_pending = xyz
    app.write_log(
        f"POSE_JUMP_REJECT {app.live_result_frame_name}: "
        f"{jump:.2f}u > {POSE_JUMP_U}u; trajectory break, skipped"
    )
    app.loc_health = "FAIL"
    app._record_no_loc()
    return False


def _update_live_camera_orientation(app: Any, result: dict, xyz: np.ndarray) -> None:
    camera_forward = normalize_camera_forward(result.get("camera_forward_world"))
    camera_axes = normalize_camera_axes(result.get("camera_axes_world"))
    app.camera_forward_world = camera_forward
    app.camera_axes_world = camera_axes
    _update_live_heading(app, camera_forward, xyz)


def _update_live_heading(
        app: Any, camera_forward: np.ndarray | None, xyz: np.ndarray) -> None:
    map_frame = getattr(app, "_integrated_auto_map_frame", None)
    if camera_forward is not None:
        # Where the camera looks, whenever the localizer measured it. The
        # measured gravity frame is only bound while integrated AUTO runs; the
        # legacy [x, z] azimuth is the same convention the displacement branch
        # below already falls back to, and it beats reporting the travel
        # direction as a camera heading (they differ by ~77 deg median on the
        # river map, where the drone flies along the bank looking sideways).
        heading = (
            float(map_frame.heading(camera_forward))
            if map_frame is not None
            else math.atan2(float(camera_forward[2]), float(camera_forward[0]))
        )
        if math.isfinite(heading):
            app.live_heading = heading
        return
    if app.live_last_xyz is None:
        return
    displacement = xyz - app.live_last_xyz
    horizontal = (
        map_frame.horizontal_distance(displacement)
        if map_frame is not None
        else math.hypot(float(displacement[0]), float(displacement[2]))
    )
    if horizontal < 0.05:
        return
    app.live_heading = (
        float(map_frame.heading(displacement))
        if map_frame is not None
        else math.atan2(float(displacement[2]), float(displacement[0]))
    )


def _accept_live_pose(app: Any, result: dict, xyz: np.ndarray) -> None:
    _update_live_camera_orientation(app, result, xyz)
    app.live_last_xyz = xyz
    app.live_pose[:] = [
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
        np.nan if app.live_heading is None else float(app.live_heading),
    ]
    app.live_locked = True
    app.live_new_pose = True
    # KLT-bridged frames carry no fresh EDM match: accept for continuity (the
    # jump gate below still bounds them) but count them so recovery-heavy
    # segments cannot silently run on optical flow. Counter resets on EDM.
    # Both spellings matter: the in-tracker bridge reports "klt_bridge", the
    # async fast path reports "klt_fast" with pose_status KLT_BRIDGED. Counting
    # only the first meant the async tracker -- which was the code default from
    # 2026-09-04 to 2026-09-05 -- never incremented this at all, and its
    # unverified runs reached 710 consecutive frames on the 720p corpus.
    bridged = (
        result.get("candidate_mode") in ("klt_bridge", "klt_fast")
        or result.get("pose_status") == "KLT_BRIDGED"
        or result.get("bridge")
    )
    if bridged:
        app.loc_bridge_run = int(getattr(app, "loc_bridge_run", 0) or 0) + 1
    else:
        app.loc_bridge_run = 0
    if app.boot_holding():
        app.boot_lock_done = True
        app.write_log(
            f"BOOT_INIT: live MegaLoc/PnP locked on {app.live_result_frame_name}"
        )


def _accept_predicted_pose(app: Any, result: dict, xyz: np.ndarray) -> None:
    """Show an IMU guess. Never a visual lock or BOOT fix."""
    pose_value = result.get("pose")
    pose = pose_value if isinstance(pose_value, dict) else {}
    yaw_raw = pose.get("yaw_raw")
    try:
        heading = float("nan") if yaw_raw is None else float(yaw_raw)
    except (TypeError, ValueError, OverflowError):
        heading = float("nan")
    if not math.isfinite(heading):
        heading = app.live_heading
    app.live_last_xyz = xyz
    app.live_pose[:] = [
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
        np.nan if heading is None else float(heading),
    ]
    app.live_new_pose = True


def update_live_results(
    app: Any,
    *,
    live_result_is_new: Callable[[Any, dict], bool],
    handle_localization_exception: Callable[[Any, dict], None],
    update_benchmark_status: Callable[[Any, dict], None],
    stabilize_result_pose: Callable[[Any, dict, np.ndarray | None], np.ndarray | None],
) -> None:
    if app.localizer is None:
        return
    results = app.localizer.poll_results()
    if not getattr(app, "inspecting", True):
        return
    for raw_result in results:
        validated = validate_live_localization_result(raw_result)
        result, xyz = normalize_live_localization_result(validated)
        if not live_result_is_new(app, result):
            continue
        annotate_ui_arrival_timing(result)
        handle_localization_exception(app, result)
        update_benchmark_status(app, result)
        xyz = stabilize_result_pose(app, result, xyz)
        try:
            invalid_success_pose = bool(
                isinstance(raw_result, dict) and raw_result.get("success")
                and not result.get("success"))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:  # Tier1: success flag shape — narrow, no silent pass
            incident = getattr(getattr(app, "session_logs", None), "incident", None)
            if callable(incident):
                incident("localization_success_shape_failed", error=repr(exc), resolved=False)
            app.write_log(f"LOCALIZATION_SUCCESS_SHAPE_FAILED: {exc!r}")
            invalid_success_pose = False
        _publish_live_result_state(app, result)
        try:
            hold_active = bool(result.get("confidence_hold_active"))
            low = bool(result.get("confidence_low"))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:  # Tier1: confidence hold shape — narrow, no silent pass
            incident = getattr(getattr(app, "session_logs", None), "incident", None)
            if callable(incident):
                incident("localization_confidence_shape_failed", error=repr(exc), resolved=False)
            app.write_log(f"LOCALIZATION_CONFIDENCE_SHAPE_FAILED: {exc!r}")
            hold_active = False
            low = False
        confidence_hold = getattr(app, "lost_hold", None)
        if (
            hold_active
            or (
                low
                and bool(getattr(
                    confidence_hold, "hold_on_low_confidence", False
                ))
            )
        ):
            # Keep low-confidence and recovery-only poses in telemetry, but
            # hold the last trustworthy public pose until recovery releases.
            continue
        if _live_result_failed(
                app, result, invalid_success_pose=invalid_success_pose,
        ):
            try:
                pose_status = result.get("pose_status")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:  # Tier1: pose_status shape — narrow, no silent pass
                incident = getattr(getattr(app, "session_logs", None), "incident", None)
                if callable(incident):
                    incident("localization_pose_status_shape_failed", error=repr(exc), resolved=False)
                app.write_log(f"LOCALIZATION_POSE_STATUS_FAILED: {exc!r}")
                pose_status = None
            if pose_status == "PREDICTED_ONLY" and xyz is not None:
                _accept_predicted_pose(app, result, xyz)
            continue
        assert xyz is not None
        if not _live_pose_is_continuous(app, xyz):
            continue
        _accept_live_pose(app, result, xyz)


def run_tick(
    app: Any,
    *,
    rolling_event_fps: RollingEventFps = _rolling_event_fps,
    next_tick_deadline: NextTickDeadline = next_tick_deadline,
    stream_terminal_text: StreamTerminalText = stream_terminal_text,
) -> None:
    """Run one UI cycle in the same order as the legacy callback."""
    stage = "drain_flight_command_results"
    try:
        app._drain_flight_command_results()
        stage = "drain_integrated_autonomy_events"
        app._drain_integrated_autonomy_events()
        stage = "refresh_held_controls"
        _refresh_held_controls(app)
        stage = "poll_backend"
        state = _poll_backend(app)
        stage = "fail_safe_dead_localizer"
        _fail_safe_dead_localizer(app)
        stage = "handle_stick_override"
        if state is None:
            return
        stage = "handle_stick_override"
        _handle_stick_override(app, state)
        stage = "gravity_tick"
        app._gravity_tick(state)
        app.video_frame_fresh = False
        stage = "update_boot_lock"
        app.update_boot_lock()
        stage = "update_live_results"
        app.update_live_results()
        stage = "update_lost_hold"
        app.update_lost_hold()
        stage = "update_detection_results"
        app.update_detection_results()
        stage = "update_stream"
        state = _update_stream(app, state, rolling_event_fps=rolling_event_fps)
        stage = "update_state_and_history"
        state = _update_state_and_history(app, state)
        stage = "update_sparse_cloud_collision_interlock"
        update_collision_interlock = getattr(
            app, "update_sparse_cloud_collision_interlock", None
        )
        if callable(update_collision_interlock):
            update_collision_interlock(state)
        stage = "update_hud_metrics"
        _update_hud_metrics(
            app,
            state,
            rolling_event_fps=rolling_event_fps,
            stream_terminal_text=stream_terminal_text,
        )
        stage = "render_if_dirty"
        _render_if_dirty(app, state)
    except Exception as exc:
        _handle_tick_failure(app, stage=stage, exc=exc)
        raise
    finally:
        _schedule_next_tick(app, next_tick_deadline=next_tick_deadline)

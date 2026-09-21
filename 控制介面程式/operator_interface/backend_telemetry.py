"""Read-only session telemetry and video-delivery diagnostics for the live backend.

These helpers log observations; they never issue aircraft commands.
"""

from __future__ import annotations

import time


def video_delivery_snapshot(backend, now: float) -> dict:
    """Continuous video-delivery health for readback telemetry + stale alert.

    Both clocks are host-monotonic seconds. Never raises; returns None
    fields when no stream is attached so offline/sim backends keep working.
    """
    stream = getattr(backend, "video_stream", None)
    if stream is None:
        return {
            "video_fps": None,
            "video_frames_delivered": 0,
            "video_frame_age_s": None,
            "video_stale": None,
        }
    try:
        fps = float(getattr(stream, "fps", 0.0) or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    try:
        delivered = int(getattr(stream, "output_index", 0) or 0)
    except (TypeError, ValueError):
        delivered = 0
    try:
        last = float(getattr(stream, "last_stamp", 0.0) or 0.0)
    except (TypeError, ValueError):
        last = 0.0
    age = (float(now) - last) if last > 0.0 else None
    stale = bool(age is not None and age > backend.VIDEO_STALE_S)
    previously_stale = bool(getattr(backend, "_video_stale_latched", False))
    if stale != previously_stale:
        backend._video_stale_latched = stale
        try:
            backend.log.event(
                "video_stale" if stale else "video_fresh",
                video_frame_age_s=round(age, 3) if age is not None else None,
                video_fps=round(fps, 2),
                video_frames_delivered=delivered,
            )
        except Exception:
            pass
    return {
        "video_fps": round(fps, 2),
        "video_frames_delivered": delivered,
        "video_frame_age_s": round(age, 3) if age is not None else None,
        "video_stale": stale,
    }


def poll_session_telemetry(backend, now: float) -> None:
    if backend.session_logs is None:
        return
    flush = getattr(backend.session_logs, "flush_deferred", None)
    if callable(flush):
        flush()
    backend._poll_stick_axes(now)
    # High-rate NED velocity + attitude for offline ESEKF/KLT-3D A/B replay
    # (定位演算法/validation/benchmark_esekf_live_replay.py). The 1 Hz
    # "readback" event below is too coarse to feed observe_fused_state.
    last_fused = getattr(backend, "_last_fused_odometry_t", 0.0)
    if now - last_fused >= 0.1:
        backend._last_fused_odometry_t = now
        backend.session_logs.telemetry(
            "fused_odometry",
            t_mono_ns=time.monotonic_ns(),
            poll_mono_ns=getattr(backend.state, "telemetry_read_mono_ns", None),
            attitude_mono_ns=getattr(backend.state, "attitude_mono_ns", None),
            attitude_stamp_source=getattr(backend.state, "attitude_stamp_source", None),
            speed_stamp_source=getattr(backend.state, "ground_speed_stamp_source", None),
            speed_mono_ns=getattr(backend.state, "ground_speed_mono_ns", None),
            altitude_mono_ns=getattr(backend.state, "altitude_mono_ns", None),
            gps_mono_ns=getattr(backend.state, "gps_location_mono_ns", None),
            velocity_frame="NED",
            attitude_kind="firmware_fused_euler",
            flight_state=getattr(backend.state, "flight_state", None),
            control_owner=getattr(backend.state, "control_owner", None),
            wind_state=getattr(backend.state, "wind_state", None),
            gimbal_pitch_deg=getattr(backend.state, "gimbal_pitch_deg", None),
            zoom=getattr(backend.state, "zoom", None),
            gps_fixed=getattr(backend.state, "gps_fixed", None),
            gps_latitude_accuracy_m=getattr(backend.state, "gps_latitude_accuracy_m", None),
            gps_longitude_accuracy_m=getattr(backend.state, "gps_longitude_accuracy_m", None),
            gps_altitude_accuracy_m=getattr(backend.state, "gps_altitude_accuracy_m", None),
            speed_north_mps=getattr(backend.state, "speed_north_mps", None),
            speed_east_mps=getattr(backend.state, "speed_east_mps", None),
            speed_down_mps=getattr(backend.state, "speed_down_mps", None),
            att_roll=getattr(backend.state, "att_roll", None),
            att_pitch=getattr(backend.state, "att_pitch", None),
            att_yaw=getattr(backend.state, "att_yaw", None),
            drone_altitude_m=getattr(backend.state, "drone_altitude_m", None),
            agl_altitude_m=getattr(backend.state, "agl_altitude_m", None),
            gps_latitude_deg=getattr(backend.state, "gps_latitude_deg", None),
            gps_longitude_deg=getattr(backend.state, "gps_longitude_deg", None),
            gps_altitude_m=getattr(backend.state, "gps_altitude_m", None),
        )
    if backend.session_logs is not None:
        last_session_tel = getattr(backend, "_last_session_telemetry_t", 0.0)
        if now - last_session_tel >= 1.0:
            backend._last_session_telemetry_t = now
            video_health = backend._video_delivery_snapshot(now)
            backend.session_logs.telemetry(
                "readback",
                battery_pct=getattr(backend.state, "battery_pct", None),
                gps_fixed=getattr(backend.state, "gps_fixed", None),
                home_valid=getattr(backend.state, "home_valid", None),
                home_reachable=getattr(backend.state, "home_reachable", None),
                distance_from_home_m=getattr(backend.state, "distance_from_home_m", None),
                rth_policy_valid=getattr(backend.state, "rth_policy_valid", None),
                rth_policy_configured=getattr(backend.state, "rth_policy_configured", None),
                stick_monitor_ok=getattr(backend.state, "stick_monitor_ok", None),
                active_incident=getattr(backend.state, "active_incident", None),
                altitude_m=getattr(backend.state, "drone_altitude_m", None),
                agl_altitude_m=getattr(backend.state, "agl_altitude_m", None),
                ground_speed_mps=getattr(backend.state, "ground_speed_mps", None),
                speed_north_mps=getattr(backend.state, "speed_north_mps", None),
                speed_east_mps=getattr(backend.state, "speed_east_mps", None),
                speed_down_mps=getattr(backend.state, "speed_down_mps", None),
                airspeed_mps=getattr(backend.state, "airspeed_mps", None),
                heading_state=getattr(backend.state, "heading_state", None),
                alert_state=getattr(backend.state, "alert_state", None),
                wind_state=getattr(backend.state, "wind_state", None),
                vibration_state=getattr(backend.state, "vibration_state", None),
                link_status=getattr(backend.state, "link_status", None),
                max_altitude_m=getattr(backend.state, "max_altitude_m", None),
                max_distance_m=getattr(backend.state, "max_distance_m", None),
                distance_geofence=getattr(backend.state, "distance_geofence_enabled", None),
                **video_health,
            )

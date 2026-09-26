"""Read-only session telemetry and video-delivery diagnostics for the live backend.

These helpers log observations; they never issue aircraft commands.
"""

from __future__ import annotations

from arsdk_ng import arsdk_message

import time
import math

_LIST_FLAG_EMPTY = 1 << 2
_LIST_FLAG_REMOVE = 1 << 3


def control_telemetry_history(backend):
    """Return yaw/stamp and horizontal NED/stamp events on the receipt clock.

    This bypasses the UI polling interval, not the aircraft/network delay.
    Consumers must still enforce freshness and reject duplicate samples.
    """
    read = getattr(getattr(backend, "drone", None), "get_event_history", None)
    if not callable(read):
        return [], []
    histories = []
    for message, fields in (("AttitudeChanged", ("yaw",)), ("SpeedChanged", ("speedX", "speedY"))):
        samples = []
        for event in read(arsdk_message("Ardrone3.PilotingState." + message)):
            try:
                sample = (
                    *[float(event.args[key]) for key in fields],
                    int(event.monotonic_ns) * 1e-9,
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if all(math.isfinite(value) for value in sample):
                samples.append(sample)
        histories.append(samples)
    return tuple(histories)


def _log_control_history(backend, now):
    histories = control_telemetry_history(backend)
    previous = getattr(backend, "_logged_control_stamps", (0.0, 0.0))
    fresh = [
        [row for row in rows if previous[index] < row[-1] <= now]
        for index, rows in enumerate(histories)
    ]
    if any(fresh):
        backend.session_logs.telemetry(
            "control_motion_history",
            timestamp_kind="host_monotonic_receipt",
            t_mono_ns=round(now * 1e9),
            attitude_samples=fresh[0],
            horizontal_ned_samples=fresh[1],
        )
        backend._logged_control_stamps = tuple(
            rows[-1][-1] if rows else previous[index] for index, rows in enumerate(fresh)
        )


def _active_drone_rssi(backend) -> int | None:
    """SkyController 3 RSSI (dBm) of the connected drone; None on direct Wi-Fi."""
    items = backend._poll_get_state(arsdk_message("Drone_manager.Drone_list_item"))
    if not isinstance(items, dict):
        return None
    for entry in items.values():
        if not isinstance(entry, dict):
            continue
        try:
            active = int(entry.get("active") or 0)
            flags = int(entry.get("list_flags") or 0)
        except (TypeError, ValueError):
            continue
        if active != 1 or flags & (_LIST_FLAG_EMPTY | _LIST_FLAG_REMOVE):
            continue
        rssi = backend._finite_float(entry.get("rssi"))
        return None if rssi is None else int(rssi)
    return None




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
    # Preserve source timestamps: polling this at 10 Hz does not turn cached
    # firmware data into independent 10 Hz measurements.
    last_fused = getattr(backend, "_last_fused_odometry_t", 0.0)
    if now - last_fused >= 0.1:
        backend._last_fused_odometry_t = now
        _log_control_history(backend, now)
        backend.session_logs.telemetry(
            "fused_odometry",
            t_mono_ns=time.monotonic_ns(),
            poll_mono_ns=getattr(backend.state, "telemetry_read_mono_ns", None),
            attitude_mono_ns=getattr(backend.state, "attitude_mono_ns", None),
            attitude_stamp_source=getattr(backend.state, "attitude_stamp_source", None),
            speed_stamp_source=getattr(backend.state, "ground_speed_stamp_source", None),
            speed_mono_ns=getattr(backend.state, "ground_speed_mono_ns", None),
            altitude_mono_ns=getattr(backend.state, "altitude_mono_ns", None),
            altitude_stamp_source=getattr(backend.state, "altitude_stamp_source", None),
            gps_mono_ns=getattr(backend.state, "gps_location_mono_ns", None),
            velocity_frame="NED",
            attitude_kind="firmware_fused_euler",
            flight_state=getattr(backend.state, "flight_state", None),
            control_owner=getattr(backend.state, "control_owner", None),
            wind_state=getattr(backend.state, "wind_state", None),
            gimbal_pitch_deg=getattr(backend.state, "gimbal_pitch_deg", None),
            gimbal_yaw_deg=getattr(backend.state, "gimbal_yaw_deg", None),
            gimbal_roll_deg=getattr(backend.state, "gimbal_roll_deg", None),
            gimbal_relative_pitch_deg=getattr(backend.state, "gimbal_relative_pitch_deg", None),
            gimbal_relative_yaw_deg=getattr(backend.state, "gimbal_relative_yaw_deg", None),
            gimbal_relative_roll_deg=getattr(backend.state, "gimbal_relative_roll_deg", None),
            gimbal_mono_ns=getattr(backend.state, "gimbal_mono_ns", None),
            gimbal_stamp_source=getattr(backend.state, "gimbal_stamp_source", None),
            zoom=getattr(backend.state, "zoom", None),
            camera_zoom_level=getattr(backend.state, "camera_zoom_level", None),
            camera_zoom_mono_ns=getattr(backend.state, "camera_zoom_mono_ns", None),
            camera_zoom_stamp_source=getattr(backend.state, "camera_zoom_stamp_source", None),
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


def poll_gps_fix(backend) -> None:
    try:
        message = arsdk_message("Ardrone3.GPSSettingsState.GPSFixStateChanged")
        gps = backend._poll_get_state(message)
        if gps is not None and gps.get("fixed") is not None:
            backend.state.gps_fixed = bool(int(gps.get("fixed") or 0))
    except Exception:
        pass


def poll_satellites(backend) -> None:
    try:
        message = arsdk_message("Ardrone3.GPSState.NumberOfSatelliteChanged")
        satellites = backend._poll_get_state(message)
        if satellites and satellites.get("numberOfSatellite") is not None:
            number = int(satellites["numberOfSatellite"])
            backend.state.gps_satellites = number if number >= 0 else None
    except Exception:
        pass


def poll_signal_sensors(backend) -> None:
    try:
        link_quality_message = arsdk_message("Common.CommonState.LinkSignalQuality")
        sensors_message = arsdk_message("Common.CommonState.SensorsStatesListChanged")
        link_quality = backend._poll_get_state(link_quality_message)
        if link_quality and link_quality.get("value") is not None:
            quality = int(link_quality["value"])
            backend.state.link_signal_quality_raw = quality if 0 <= quality <= 255 else None
        sensor_updates = backend._sensor_state_updates(
            backend._poll_get_state(sensors_message)
        )
        if sensor_updates:
            current = dict(getattr(backend.state, "sensor_states", {}) or {})
            current.update(sensor_updates)
            backend.state.sensor_states = current
    except Exception:
        pass
    try:
        backend.state.wifi_rssi_dbm = _active_drone_rssi(backend)
    except Exception:
        pass


def poll_gimbal(backend) -> None:
    try:
        message = arsdk_message("Gimbal.Attitude")
        state = backend._poll_get_state(message)
        if not isinstance(state, dict) or state.get("pitch_absolute") is None:
            return
        if int(state.get("gimbal_id") or 0) != 0:
            return
        sample = backend._observe_telemetry(
            "gimbal", message, state, now_mono_ns=time.monotonic_ns()
        )
        for attribute, key in (
            ("gimbal_pitch_deg", "pitch_absolute"),
            ("gimbal_yaw_deg", "yaw_absolute"),
            ("gimbal_roll_deg", "roll_absolute"),
            ("gimbal_relative_pitch_deg", "pitch_relative"),
            ("gimbal_relative_yaw_deg", "yaw_relative"),
            ("gimbal_relative_roll_deg", "roll_relative"),
        ):
            value = backend._finite_float(sample.get(key))
            if attribute != "gimbal_pitch_deg" or value is not None:
                setattr(backend.state, attribute, value)
    except Exception:
        pass


def poll_camera_zoom(backend) -> None:
    try:
        message = arsdk_message("Camera.Zoom_level")
        raw = backend._poll_get_state(message)
        if not isinstance(raw, dict) or raw.get("level") is None:
            return
        if int(raw.get("cam_id") or 0) != 0:
            return
        sample = backend._observe_telemetry(
            "camera_zoom", message, raw, now_mono_ns=time.monotonic_ns()
        )
        value = backend._finite_float(sample.get("level"))
        backend.state.camera_zoom_level = value if value is not None and value > 0 else None
    except Exception:
        pass

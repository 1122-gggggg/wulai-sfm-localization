"""Read-only session telemetry and video-delivery diagnostics for the live backend.

These helpers log observations; they never issue aircraft commands.
"""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections import deque

#: Attitude/speed events kept for the control loop between UI polls.
MOTION_HISTORY_SIZE = 512
#: Olympe stamps ``event.date`` when the event object is created on receipt.
#: Ages outside this window are clock faults, not usable receipt times.
_MAX_EVENT_AGE_S = 5.0


def olympe_message(module: str, name: str):
    """Resolve ``olympe.messages.<module>.<name>`` lazily (Olympe is heavy)."""
    return getattr(importlib.import_module(f"olympe.messages.{module}"), name)


def _map_entry(state, key_field: str, key: int) -> dict | None:
    """Return one MAP_ITEM entry from Olympe's keyed state (or a flat dict)."""
    if not isinstance(state, dict) or not state:
        return None
    if key_field in state:
        return state
    for entry in state.values():
        if isinstance(entry, dict) and int(entry.get(key_field) or 0) == key:
            return entry
    return None


class MotionEventHistory:
    """Bounded yaw and horizontal-NED event history on the host monotonic clock.

    Olympe's ``get_state`` only exposes the newest value, so the UI poll would
    drop intermediate turns and accelerations. Each event is converted from its
    Olympe receipt time (``event.date``) to ``time.monotonic()``; samples whose
    receipt time cannot be trusted are discarded rather than restamped.
    """

    FIELDS = (("AttitudeChanged", ("yaw",)), ("SpeedChanged", ("speedX", "speedY")))

    def __init__(
        self, maxlen: int = MOTION_HISTORY_SIZE, *, now=time.monotonic, wall_now=time.time
    ):
        self._lock = threading.RLock()
        self._now = now
        self._wall_now = wall_now
        self._rows = tuple(deque(maxlen=maxlen) for _ in self.FIELDS)
        self._evicted_counts = [0 for _ in self.FIELDS]
        self._evicted_through = [None for _ in self.FIELDS]
        self._subscriptions: list = []

    def record(self, index: int, event) -> None:
        fields = self.FIELDS[index][1]
        try:
            values = [float(event.args[key]) for key in fields]
            age_s = float(self._wall_now()) - float(event.date.timestamp())
            stamp = float(self._now()) - max(0.0, age_s)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return
        if not -0.05 <= age_s <= _MAX_EVENT_AGE_S:
            return
        sample = (*values, stamp)
        if not all(math.isfinite(value) for value in sample):
            return
        with self._lock:
            rows = self._rows[index]
            if rows and stamp <= rows[-1][-1]:
                return
            if len(rows) == rows.maxlen:
                self._evicted_counts[index] += 1
                self._evicted_through[index] = rows[0][-1]
            rows.append(sample)

    def attach(self, drone) -> None:
        for index, (name, _fields) in enumerate(self.FIELDS):
            message = olympe_message("ardrone3.PilotingState", name)
            self._subscriptions.append(
                drone.subscribe(
                    lambda event, _scheduler, index=index: self.record(index, event),
                    message(_policy="wait"),
                )
            )

    def detach(self) -> None:
        subscriptions, self._subscriptions = self._subscriptions, []
        for subscription in subscriptions:
            try:
                subscription.unsubscribe()
            except Exception:
                pass

    def since(self, since_stamps: tuple[float | None, ...] | None = None) -> tuple[list, list]:
        with self._lock:
            result = []
            for index, rows in enumerate(self._rows):
                if not rows:
                    result.append([])
                    continue
                cutoff = None
                if since_stamps is not None and index < len(since_stamps):
                    cutoff = since_stamps[index]
                if cutoff is None or cutoff < rows[0][-1]:
                    result.append(list(rows))
                elif cutoff >= rows[-1][-1]:
                    result.append([])
                else:
                    selected = []
                    for row in reversed(rows):
                        if row[-1] <= cutoff:
                            break
                        selected.append(row)
                    selected.reverse()
                    result.append(selected)
            return tuple(result)

    def diagnostic_snapshot(self, since_stamps=None) -> dict:
        """Ring turnover is not necessarily loss; flag evictions beyond the log cursor."""
        with self._lock:
            unread = []
            for index, stamp in enumerate(self._evicted_through):
                cutoff = (
                    since_stamps[index]
                    if since_stamps is not None and index < len(since_stamps)
                    else None
                )
                unread.append(stamp is not None and (cutoff is None or stamp > cutoff))
            return {
                "capacity_per_source": self._rows[0].maxlen,
                "source_names": [name for name, _ in self.FIELDS],
                "evicted_total": list(self._evicted_counts),
                "evicted_through_mono": list(self._evicted_through),
                "unread_eviction_since_log_cursor": unread,
            }

    def since_with_diagnostics(self, since_stamps=None):
        """Snapshot samples and their loss indicator under the same source lock."""
        with self._lock:
            return self.since(since_stamps), self.diagnostic_snapshot(since_stamps)


def control_telemetry_history(
    backend, since_stamps: tuple[float | None, ...] | None = None
) -> tuple[list, list]:
    """Return yaw/stamp and horizontal NED/stamp events on the receipt clock.

    This bypasses the UI polling interval, not the aircraft/network delay.
    Consumers must still enforce freshness and reject duplicate samples.
    """
    history = getattr(backend, "motion_history", None)
    if history is None:
        return [], []
    return history.since(since_stamps)


def _log_control_history(backend, now):
    previous = getattr(backend, "_logged_control_stamps", None)
    history = getattr(backend, "motion_history", None)
    if hasattr(history, "since_with_diagnostics"):
        (attitudes, velocities), history_health = history.since_with_diagnostics(previous)
    else:
        attitudes, velocities = control_telemetry_history(backend, previous)
        history_health = None
    fresh_attitudes = [row for row in attitudes if row[-1] <= now]
    fresh_velocities = [row for row in velocities if row[-1] <= now]
    if fresh_attitudes or fresh_velocities:
        backend.session_logs.telemetry(
            "control_motion_history",
            timestamp_kind="host_monotonic_receipt",
            t_mono_ns=round(now * 1e9),
            attitude_samples=fresh_attitudes,
            horizontal_ned_samples=fresh_velocities,
            history_health=history_health,
        )
        prev_att = previous[0] if previous is not None else None
        prev_vel = previous[1] if previous is not None else None
        backend._logged_control_stamps = (
            fresh_attitudes[-1][-1] if fresh_attitudes else prev_att,
            fresh_velocities[-1][-1] if fresh_velocities else prev_vel,
        )


def _active_drone_rssi(backend) -> int | None:
    """SkyController 3 RSSI (dBm) of the active drone entry, or None.

    Olympe already drops LIST_ITEM entries flagged Empty/Remove from its state.
    """
    items = backend._poll_get_state(olympe_message("drone_manager", "drone_list_item"))
    if not isinstance(items, dict):
        return None
    for entry in items.values():
        if not isinstance(entry, dict):
            continue
        try:
            active = int(entry.get("active") or 0)
        except (TypeError, ValueError):
            continue
        if active != 1:
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
                battery_info=getattr(backend.state, "battery_info", {}),
                camera_info=getattr(backend.state, "camera_info", {}),
                gimbal_info=getattr(backend.state, "gimbal_info", {}),
                device_info_read_mono_ns=getattr(backend.state, "device_info_read_mono_ns", None),
            )


def poll_gps_fix(backend) -> None:
    try:
        message = olympe_message("ardrone3.GPSSettingsState", "GPSFixStateChanged")
        gps = backend._poll_get_state(message)
        if gps is not None and gps.get("fixed") is not None:
            backend.state.gps_fixed = bool(int(gps.get("fixed") or 0))
    except Exception:
        pass


def poll_satellites(backend) -> None:
    try:
        message = olympe_message("ardrone3.GPSState", "NumberOfSatelliteChanged")
        satellites = backend._poll_get_state(message)
        if satellites and satellites.get("numberOfSatellite") is not None:
            number = int(satellites["numberOfSatellite"])
            backend.state.gps_satellites = number if number >= 0 else None
    except Exception:
        pass


def poll_signal_sensors(backend) -> None:
    try:
        link_quality_message = olympe_message("common.CommonState", "LinkSignalQuality")
        sensors_message = olympe_message("common.CommonState", "SensorsStatesListChanged")
        link_quality = backend._poll_get_state(link_quality_message)
        if link_quality and link_quality.get("value") is not None:
            quality = int(link_quality["value"])
            backend.state.link_signal_quality_raw = quality if 0 <= quality <= 255 else None
        sensor_updates = backend._sensor_state_updates(backend._poll_get_state(sensors_message))
        if sensor_updates:
            current = dict(getattr(backend.state, "sensor_states", {}) or {})
            current.update(sensor_updates)
            backend.state.sensor_states = current
    except Exception:
        pass
    try:
        if backend.via_skycontroller():
            backend.state.wifi_rssi_dbm = _active_drone_rssi(backend)
        else:
            wifi = backend._poll_get_state(
                olympe_message("common.CommonState", "WifiSignalChanged")
            )
            rssi = backend._finite_float(wifi.get("rssi")) if wifi else None
            backend.state.wifi_rssi_dbm = None if rssi is None else int(rssi)
    except Exception:
        pass


def poll_gimbal(backend) -> None:
    try:
        message = olympe_message("gimbal", "attitude")
        state = _map_entry(backend._poll_get_state(message), "gimbal_id", 0)
        if state is None or state.get("pitch_absolute") is None:
            return
        sample = backend._observe_telemetry(
            "gimbal", message, state, map_key=0, now_mono_ns=time.monotonic_ns()
        )
        if int(sample.get("gimbal_id") or 0) != 0:
            return  # Olympe's last event may belong to another MAP_ITEM key.
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
        message = olympe_message("camera", "zoom_level")
        raw = _map_entry(backend._poll_get_state(message), "cam_id", 0)
        if raw is None or raw.get("level") is None:
            return
        sample = backend._observe_telemetry(
            "camera_zoom", message, raw, map_key=0, now_mono_ns=time.monotonic_ns()
        )
        if int(sample.get("cam_id") or 0) != 0:
            return
        value = backend._finite_float(sample.get("level"))
        backend.state.camera_zoom_level = value if value is not None and value > 0 else None
    except Exception:
        pass


def _device_json(value):
    """Preserve raw units while making SDK enums/map keys JSON-safe."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(_device_json(key)): _device_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_device_json(item) for item in value]
    return str(getattr(value, "name", value))


def _device_event(backend, module: str, name: str, key_field: str | None = None):
    try:
        raw = backend._poll_get_state(olympe_message(module, name))
        if key_field is not None:
            raw = _map_entry(raw, key_field, 0)
            if raw is not None and int(raw.get(key_field, -1)) != 0:
                return None
        return _device_json(raw) if isinstance(raw, dict) and raw else None
    except (ImportError, AttributeError, TypeError, ValueError, OverflowError, RuntimeError):
        return None


def poll_device_info(backend, now: float) -> None:
    """Read optional battery/camera/gimbal caches at 1 Hz; never send commands."""
    previous = getattr(backend, "_last_device_info_t", None)
    if previous is not None and now - previous < 1.0:
        return
    backend._last_device_info_t = now
    battery = {
        name: _device_event(backend, "battery", name)
        for name in ("voltage", "health", "cycle_count", "alert")
    }
    for name in ("BatteryChanged", "BatteryState"):
        battery[name] = (
            _device_event(backend, "skyctrl.SkyControllerState", name)
            if backend.via_skycontroller()
            else None
        )
    backend.state.battery_info = battery
    backend.state.camera_info = {
        name: _device_event(backend, "camera", name, "cam_id")
        for name in (
            "camera_mode",
            "exposure",
            "exposure_settings",
            "white_balance",
            "recording_mode",
            "recording_state",
            "zoom_info",
        )
    }
    backend.state.gimbal_info = {
        name: _device_event(backend, "gimbal", name, "gimbal_id")
        for name in ("alert", "calibration_state", "stabilization_state", "axis_lock_state")
    }
    # This is cache-read time, not sensor/event freshness.
    backend.state.device_info_read_mono_ns = time.monotonic_ns()

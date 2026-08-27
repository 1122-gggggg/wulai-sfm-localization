"""Shared ANAFI profile and operator telemetry state."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


STREAM_WIDTH = 1280
STREAM_HEIGHT = 720


@dataclass(frozen=True)
class AnafiProfile:
    model: str = "Parrot ANAFI"
    weight_g: int = 320
    max_horizontal_speed_mps: float = 15.0
    max_vertical_speed_mps: float = 4.0
    max_yaw_rate_dps: float = 200.0
    max_wind_kmh: float = 50.0
    flight_time_s: float = 25.0 * 60.0
    takeoff_hover_m: float = 1.0
    gimbal_pitch_min_deg: float = -90.0
    gimbal_pitch_max_deg: float = 90.0
    gimbal_pitch_rate_dps: float = 180.0
    stream_width: int = STREAM_WIDTH
    stream_height: int = STREAM_HEIGHT
    stream_fps: float = 30.0
    stream_latency_ms: float = 280.0
    stream_mbps: float = 5.0
    video_hfov_deg: float = 69.0
    digital_zoom_max: float = 3.0
    lossless_zoom_fhd: float = 2.8


ANAFI = AnafiProfile()


@dataclass
class DroneState:
    mode: str = "MANUAL"
    tracker_state: str = "HOVER"
    loc: str = "SIM"
    stream: str = "WAIT"
    pose: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    inliers: int = 0
    reproj: float | None = None
    battery_pct: float = 100.0
    altitude_m: float = 0.0
    drone_altitude_m: float | None = None
    gimbal_pitch_deg: float = -20.0
    zoom: float = 1.0
    link_latency_ms: float = ANAFI.stream_latency_ms
    frame_age_ms: float | None = None
    telemetry_read_mono_ns: int | None = None
    last_pcmd_call_mono_ns: int | None = None
    pcmd_to_telemetry_poll_ms: float | None = None
    stream_fps: float = ANAFI.stream_fps
    stream_mbps: float = ANAFI.stream_mbps
    last_command: str = "ready"
    link_ok: bool = True
    link_status: str = "OK"
    gps_fixed: bool | None = None
    max_altitude_m: float | None = None
    max_distance_m: float | None = None
    distance_geofence_enabled: bool | None = None
    max_tilt_deg: float | None = None
    max_vertical_speed_mps: float | None = None
    max_rotation_speed_dps: float | None = None
    preflight_ok: bool | None = None
    preflight_reason: str = "not checked"
    control_owner: str = "SIM"
    active_incident: str = ""
    aircraft_identity: str = "SIMULATED ANAFI"
    controller_identity: str = "SIMULATED CONTROLLER"
    autonomous_locked: bool = True
    home_valid: bool | None = None
    home_reachable: bool | None = None
    rth_policy_valid: bool | None = None
    rth_policy_configured: bool = False
    rth_min_altitude_m: float | None = None
    skycontroller_magnetometer_readable: bool = True
    stick_monitor_ok: bool = False
    distance_from_home_m: float | None = None
    distance_guard_active: bool | None = None
    flight_state: str = "UNKNOWN"
    agl_altitude_m: float | None = None
    speed_north_mps: float | None = None
    speed_east_mps: float | None = None
    speed_down_mps: float | None = None
    ground_speed_mps: float | None = None
    gps_latitude_deg: float | None = None
    gps_longitude_deg: float | None = None
    gps_altitude_m: float | None = None
    gps_latitude_accuracy_m: float | None = None
    gps_longitude_accuracy_m: float | None = None
    gps_altitude_accuracy_m: float | None = None
    gps_satellites: int | None = None
    heading_state: str = "UNKNOWN"
    alert_state: str = "UNKNOWN"
    navigate_home_state: str = "UNKNOWN"
    navigate_home_reason: str = "UNKNOWN"
    wind_state: str = "UNKNOWN"
    vibration_state: str = "UNKNOWN"
    hover_no_gps_too_dark: bool | None = None
    hover_no_gps_too_high: bool | None = None
    wifi_rssi_dbm: int | None = None
    link_signal_quality_raw: int | None = None
    sensor_states: dict[str, bool] = field(default_factory=dict)
    airspeed_mps: float | None = None
    disk_free_bytes: int | None = None
    disk_free_percent: float | None = None
    disk_warning: bool = False
    autonomous_speed_limit_enabled: bool = True
    autonomous_speed_limit_mps: float = 0.30
    autonomous_speed_guard_status: str = "SPEED_WAITING"
    autonomous_approval_valid: bool = False
    drone_magnetometer_required: int | None = None
    drone_magnetometer_started: bool | None = None
    drone_magnetometer_axis: str = "unknown"
    drone_magnetometer_x_done: bool | None = None
    drone_magnetometer_y_done: bool | None = None
    drone_magnetometer_z_done: bool | None = None
    drone_magnetometer_failed: bool | None = None
    skycontroller_magnetometer_state: str = "not_applicable"
    att_roll: float = 0.0
    att_pitch: float = 0.0
    att_yaw: float = 0.0

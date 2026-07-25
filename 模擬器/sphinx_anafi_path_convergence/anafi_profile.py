#!/usr/bin/env python3
"""Single public ANAFI specification used by the simulator experiment.

These are vehicle capability and sensor/video characteristics from Parrot's
ANAFI white paper v1.4. They do not relax controller PCMD caps and do not turn
the pure-Python kinematic approximation into a firmware or aerodynamic model.
Parrot Sphinx remains the software-in-the-loop fidelity gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AnafiSpec:
    profile_id: str = "anafi_white_paper_v1_4"
    name: str = "Parrot ANAFI"
    source_url: str = (
        "https://www.parrot.com/assets/s3fs-public/2020-07/"
        "white-paper_anafi-v1.4-en.pdf"
    )

    mass_kg: float = 0.320
    max_horizontal_speed_mps: float = 15.0
    max_ascent_speed_mps: float = 4.0
    max_descent_speed_mps: float = 4.0
    max_angular_speed_deg_s: float = 200.0
    wind_resistance_kmh: float = 50.0
    wind_gust_kmh: float = 80.0
    takeoff_hover_height_m: float = 1.0
    hover_accuracy_m_at_1m: float = 0.015
    internal_control_loop_hz: float = 200.0

    video_width_px: int = 1280
    video_height_px: int = 720
    video_fps: int = 30
    video_bitrate_bps: int = 5_000_000
    video_latency_ms: float = 280.0

    gimbal_pitch_min_deg: float = -90.0
    gimbal_pitch_max_deg: float = 90.0
    gimbal_max_speed_deg_s: float = 180.0
    gps_position_std_m: float = 1.2
    gps_speed_std_mps: float = 0.5
    barometer_noise_std_m: float = 0.2

    def to_metadata(self) -> dict:
        return asdict(self)


ANAFI_PROFILE = AnafiSpec()


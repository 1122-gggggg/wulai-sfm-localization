"""Transform Sphinx true ENU displacement into the initial body frame."""

from __future__ import annotations

from math import atan2, cos, degrees, sin

from .models import MotionDelta, TruePosition


def _wrapped_angle_rad(angle_rad: float) -> float:
    return atan2(sin(angle_rad), cos(angle_rad))


def delta_in_initial_body_frame(
    start: TruePosition,
    end: TruePosition,
    *,
    initial_yaw_rad: float,
    final_yaw_rad: float,
) -> MotionDelta:
    """Project Gazebo ENU displacement onto the starting forward/right axes.

    Gazebo uses a right-handed ENU world frame. The deterministic probe starts
    at yaw zero, but keeping this transform yaw-aware makes the result valid if
    the spawn pose is later changed.
    """
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    forward_m = dx * cos(initial_yaw_rad) + dy * sin(initial_yaw_rad)
    right_m = -dx * sin(initial_yaw_rad) + dy * cos(initial_yaw_rad)
    yaw_change_rad = _wrapped_angle_rad(final_yaw_rad - initial_yaw_rad)
    return MotionDelta(
        forward_m=forward_m,
        right_m=right_m,
        up_m=end.z_m - start.z_m,
        yaw_change_deg=degrees(yaw_change_rad),
    )

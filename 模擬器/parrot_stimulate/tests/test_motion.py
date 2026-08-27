from math import pi

import pytest

from anafi_pcmd_sim.models import TruePosition
from anafi_pcmd_sim.motion import delta_in_initial_body_frame


def test_yaw_zero_uses_gazebo_x_as_forward_and_z_as_up() -> None:
    start = TruePosition(timestamp_s=1.0, x_m=0.0, y_m=0.0, z_m=0.2)
    end = TruePosition(timestamp_s=2.0, x_m=1.2, y_m=-0.1, z_m=0.8)

    delta = delta_in_initial_body_frame(start, end, initial_yaw_rad=0.0, final_yaw_rad=0.0)

    assert delta.forward_m == pytest.approx(1.2)
    assert delta.right_m == pytest.approx(-0.1)
    assert delta.up_m == pytest.approx(0.6)
    assert delta.yaw_change_deg == pytest.approx(0.0)


def test_yaw_aware_projection_keeps_forward_motion_forward() -> None:
    start = TruePosition(timestamp_s=1.0, x_m=0.0, y_m=0.0, z_m=0.0)
    end = TruePosition(timestamp_s=2.0, x_m=0.0, y_m=1.0, z_m=0.0)

    delta = delta_in_initial_body_frame(start, end, initial_yaw_rad=pi / 2, final_yaw_rad=pi / 2)

    assert round(delta.forward_m, 8) == 1.0
    assert round(delta.right_m, 8) == 0.0


def test_yaw_change_is_wrapped_to_the_shortest_signed_angle() -> None:
    start = TruePosition(timestamp_s=1.0, x_m=0.0, y_m=0.0, z_m=0.0)
    end = TruePosition(timestamp_s=2.0, x_m=0.0, y_m=0.0, z_m=0.0)

    delta = delta_in_initial_body_frame(
        start,
        end,
        initial_yaw_rad=179 * pi / 180,
        final_yaw_rad=-179 * pi / 180,
    )

    assert round(delta.yaw_change_deg, 8) == 2.0

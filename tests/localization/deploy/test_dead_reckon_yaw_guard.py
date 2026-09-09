"""The dead-reckon corner guard has to measure yaw in the map's own horizontal plane.

`azimuth_turn_exceeds` halves a dead-reckoned step when the camera swings more
than a few degrees in one frame, because DR extrapolates on a constant-velocity
prior that a corner breaks.  Which plane the angle is measured in decides
whether it fires on anything real: this site's gravity runs along +Y, so the
raw `atan2(forward[1], forward[0])` convention used elsewhere in the tracker
spans a *vertical* plane.  These tests pin the gravity-aligned basis and show,
with the site's own measured gravity, that the raw convention both misses
real corners in level flight and fires on pure gimbal tilt.
"""

from __future__ import annotations

import math

# Source modules are supplied by the repository's pytest pythonpath.
import numpy as np
import pytest
from real_path_follow_controller import MapFrame

from two_rate_tracker import DEAD_RECKON_MAX_TURN_DEG, azimuth_turn_exceeds

#: `T_align_gravity.json` of river_gluemap_all8_direct_20260908: gravity is
#: within a degree of +Y, so the map's horizontal plane is X-Z.
RIVER_GRAVITY = (-0.010870293826223672, 0.9965557122837028, 0.08221039488321148)


@pytest.fixture
def river_frame() -> MapFrame:
    return MapFrame.from_gravity(RIVER_GRAVITY, source="test")


def camera_rotation(frame: MapFrame, azimuth_deg: float, depression_deg: float) -> np.ndarray:
    """A `cam_from_world` rotation looking along `azimuth`, tilted down."""

    azimuth = math.radians(azimuth_deg)
    depression = math.radians(depression_deg)
    forward = (
        math.cos(azimuth) * math.cos(depression) * frame.east
        + math.sin(azimuth) * math.cos(depression) * frame.north
        - math.sin(depression) * frame.up
    )
    right = np.cross(forward, frame.up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    # Rows of cam_from_world are the camera axes in world coordinates.
    return np.stack([right, down, forward / np.linalg.norm(forward)])


def relative_rotation(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """`after = relative @ before`, the shape recoverPose hands the guard."""

    return after @ before.T


def raw_azimuth(rotation: np.ndarray) -> float:
    """The pre-fix convention: atan2 over world X and Y."""

    forward = rotation[2]
    return math.atan2(float(forward[1]), float(forward[0]))


@pytest.mark.parametrize("turn_deg", [8.0, 15.0, 30.0, -12.0, 90.0])
def test_a_real_corner_fires_the_guard(river_frame: MapFrame, turn_deg: float) -> None:
    before = camera_rotation(river_frame, 20.0, 30.0)
    after = camera_rotation(river_frame, 20.0 + turn_deg, 30.0)

    assert azimuth_turn_exceeds(river_frame, before, relative_rotation(before, after))


@pytest.mark.parametrize("turn_deg", [0.0, 1.0, 3.0, -4.0, 4.9])
def test_level_flight_below_the_threshold_does_not_fire(
    river_frame: MapFrame, turn_deg: float
) -> None:
    before = camera_rotation(river_frame, 20.0, 30.0)
    after = camera_rotation(river_frame, 20.0 + turn_deg, 30.0)

    assert not azimuth_turn_exceeds(river_frame, before, relative_rotation(before, after))


def test_the_threshold_is_the_documented_one(river_frame: MapFrame) -> None:
    before = camera_rotation(river_frame, 0.0, 25.0)
    just_under = camera_rotation(river_frame, DEAD_RECKON_MAX_TURN_DEG - 0.2, 25.0)
    just_over = camera_rotation(river_frame, DEAD_RECKON_MAX_TURN_DEG + 0.2, 25.0)

    assert not azimuth_turn_exceeds(river_frame, before, relative_rotation(before, just_under))
    assert azimuth_turn_exceeds(river_frame, before, relative_rotation(before, just_over))


@pytest.mark.parametrize("depression_deg", [0.0, 2.0, 5.0, 10.0])
def test_the_raw_xy_convention_misses_a_corner_in_level_flight(
    river_frame: MapFrame, depression_deg: float
) -> None:
    """Why the fix exists, half one: near level, the old angle barely moves."""

    before = camera_rotation(river_frame, 20.0, depression_deg)
    after = camera_rotation(river_frame, 50.0, depression_deg)  # a 30 degree corner

    measured = abs(math.degrees(river_frame.heading(after[2]) - river_frame.heading(before[2])))
    raw = abs(math.degrees(raw_azimuth(after) - raw_azimuth(before)))

    assert measured == pytest.approx(30.0, abs=0.5)
    # 1.0 to 3.9 deg over this band: the corner never reached the 5 deg gate.
    assert raw < DEAD_RECKON_MAX_TURN_DEG
    assert azimuth_turn_exceeds(river_frame, before, relative_rotation(before, after))


@pytest.mark.parametrize(("from_deg", "to_deg"), [(5.0, 10.0), (20.0, 25.0), (15.0, 45.0)])
def test_gimbal_pitch_alone_is_not_a_corner(
    river_frame: MapFrame, from_deg: float, to_deg: float
) -> None:
    """Why the fix exists, half two: the old angle fires on pure tilt."""

    before = camera_rotation(river_frame, 20.0, from_deg)
    after = camera_rotation(river_frame, 20.0, to_deg)  # same heading, more tilt

    assert river_frame.heading(after[2]) == pytest.approx(river_frame.heading(before[2]))
    assert not azimuth_turn_exceeds(river_frame, before, relative_rotation(before, after))
    # The pre-fix expression read that pure tilt as a heading change and halved
    # the dead-reckoned step for it.
    assert abs(math.degrees(raw_azimuth(after) - raw_azimuth(before))) > DEAD_RECKON_MAX_TURN_DEG


def test_without_a_map_frame_the_guard_stands_down(river_frame: MapFrame) -> None:
    """No horizontal basis means no defensible angle, so the guard must not fire."""

    before = camera_rotation(river_frame, 20.0, 30.0)
    after = camera_rotation(river_frame, 110.0, 30.0)

    assert not azimuth_turn_exceeds(None, before, relative_rotation(before, after))


def test_a_near_vertical_optical_axis_has_no_azimuth(river_frame: MapFrame) -> None:
    straight_down = camera_rotation(river_frame, 20.0, 90.0)
    after = camera_rotation(river_frame, 110.0, 90.0)

    assert river_frame.horizontal_distance(straight_down[2]) < 1e-9
    assert not azimuth_turn_exceeds(
        river_frame, straight_down, relative_rotation(straight_down, after)
    )

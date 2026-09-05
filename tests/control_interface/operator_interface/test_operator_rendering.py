"""Focused tests for the pure video rendering helpers."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


HERE = (
    Path(__file__).resolve().parents[3]
    / "控制介面程式"
    / "operator_interface"
)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from operator_rendering import draw_video_hud  # noqa: E402


class _RecordingDraw:
    def __init__(self) -> None:
        self.rectangles: list[tuple[tuple[int, ...], dict]] = []
        self.texts: list[tuple[tuple[int, int], str, dict]] = []

    def rectangle(self, box, **kwargs) -> None:
        self.rectangles.append((tuple(box), kwargs))

    def text(self, xy, text, **kwargs) -> None:
        self.texts.append((tuple(xy), text, kwargs))


def test_video_hud_draws_black_text_directly_on_stream_without_bottom_panel() -> None:
    draw = _RecordingDraw()
    state = SimpleNamespace(
        frame_age_ms=12.4,
        stream_fps=30.0,
        gimbal_pitch_deg=-20.0,
        zoom=1.0,
        link_ok=False,
    )

    link_ok = draw_video_hud(
        draw,
        320,
        240,
        state,
        ("diagnostic one", "diagnostic two"),
        60,
        object(),
        object(),
        object(),
        True,
        280.0,
    )

    assert link_ok is False
    assert draw.rectangles == [
        ((18, 18, 302, 222), {"outline": "#363c44", "width": 2}),
    ]
    assert [text for _, text, _ in draw.texts] == [
        "影像 30.0 FPS · 影格新鮮度 12 ms · 鏡頭 -20° / 1.0x",
        "diagnostic one",
        "diagnostic two",
    ]
    assert [xy for xy, _, _ in draw.texts] == [(28, 185), (28, 202), (28, 219)]
    assert all(kwargs["fill"] == "#000000" for _, _, kwargs in draw.texts)
    assert all(kwargs["stroke_width"] == 1 for _, _, kwargs in draw.texts)
    assert all(kwargs["stroke_fill"] == "#f4f4f4" for _, _, kwargs in draw.texts)


def _map_context(**overrides):
    """A map context whose projection is a plain top-down orthographic view."""
    from operator_rendering import MapRenderContext

    width = overrides.pop("width", 620)
    height = overrides.pop("height", 430)
    map_zoom = overrides.pop("map_zoom", 3.2)
    map_radius = overrides.pop("map_radius", 12.0)
    scale = min(width, height) * 0.46 * map_zoom / map_radius

    def transform_xyz(points):
        return np.asarray(points, dtype=float).reshape(-1, 3)

    def project_world(point, w, h):
        v = transform_xyz(point)[0]
        return (int(w * 0.5 + v[0] * scale), int(h * 0.5 - v[1] * scale))

    defaults = dict(
        width=width,
        height=height,
        map_zoom=map_zoom,
        map_radius=map_radius,
        map_pan=np.zeros(2),
        no_loc_markers=[],
        route_pts=[],
        history=[],
        history_health=[],
        pose=(0.0, 0.0, 0.0, float("nan")),
        # right / down / forward, orthonormal. Forward is +X, i.e. screen +x,
        # and right is -Z, the axis this projection drops, so the picture plane
        # is seen edge-on: the honest 3D baseline.
        camera_axes=np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
        camera_forward=None,
        collision_center=None,
        collision_radius=0.0,
        collision_status="DISABLED",
        collision_point=None,
        collision_preview=False,
        transform_xyz=transform_xyz,
        project_world=project_world,
        route_color="#ff3ea5",
        health_color={"LOW": "#e0a92e"},
        route_dot_max=200,
        no_loc_max_markers=40,
        video_aspect_ratio=1280 / 720,
        overlay_font=None,
        # This projection paints the world XY plane, so its ground plane is XY
        # and its up axis is +Z.
        map_east=np.array([1.0, 0.0, 0.0]),
        map_north=np.array([0.0, 1.0, 0.0]),
        map_up=np.array([0.0, 0.0, 1.0]),
    )
    defaults.update(overrides)
    return MapRenderContext(**defaults)


# right / down / forward for a camera whose picture faces this view's depth
# axis: seen face-on, the whole picture lands on the screen.
_FACE_ON_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class _RecordingMapDraw:
    def __init__(self) -> None:
        self.lines: list[tuple[tuple, dict]] = []
        self.polygons: list[tuple[list, dict]] = []
        self.ellipses: list[tuple[tuple, dict]] = []

    def line(self, xy, **kwargs) -> None:
        self.lines.append((tuple(xy), kwargs))

    def polygon(self, xy, **kwargs) -> None:
        self.polygons.append((list(xy), kwargs))

    def ellipse(self, xy, **kwargs) -> None:
        self.ellipses.append((tuple(xy), kwargs))


def _camera_overlay(context) -> _RecordingMapDraw:
    from operator_rendering import _draw_camera_overlay

    draw = _RecordingMapDraw()
    _draw_camera_overlay(draw, context)
    return draw


def _quad(draw: _RecordingMapDraw) -> list:
    """The picture quad: drawn first, before the direction arrowhead."""
    assert len(draw.polygons) == 2, "expected picture fill plus arrowhead"
    return draw.polygons[0][0]


def test_camera_picture_plane_geometry_is_a_screen_ahead_of_the_drone() -> None:
    from operator_rendering import (
        CAMERA_VIEW_RECT_GAP_PX,
        CAMERA_VIEW_RECT_PX,
        camera_view_plane_corners,
    )

    scale = 20.0
    aspect = 1280 / 720
    drone = np.array([3.0, -1.0, 2.0])
    right = np.array([0.0, 1.0, 0.0])
    image_up = np.array([0.0, 0.0, 1.0])
    forward = np.array([1.0, 0.0, 0.0])
    corners = camera_view_plane_corners(drone, right, image_up, forward, scale, aspect)

    center = corners.mean(axis=0)
    # The quad is perpendicular to the optical axis and sits GAP ahead of it.
    assert np.allclose(center, drone + forward * CAMERA_VIEW_RECT_GAP_PX / scale)
    assert np.allclose((corners - center) @ forward, 0.0, atol=1e-12)
    # Width along image-right, height along image-up, in the stream's own
    # picture shape.
    width = float(np.linalg.norm(corners[1] - corners[0]))
    height = float(np.linalg.norm(corners[2] - corners[1]))
    assert width == pytest.approx(CAMERA_VIEW_RECT_PX / scale)
    assert height == pytest.approx(CAMERA_VIEW_RECT_PX / (aspect * scale))
    # Image order: top-left first, counter-clockwise on screen.
    assert np.allclose(
        corners[0],
        center - right * width * 0.5 + image_up * height * 0.5,
    )


def test_camera_picture_seen_face_on_is_the_calibrated_size() -> None:
    from operator_rendering import CAMERA_VIEW_RECT_PX

    context = _map_context(camera_axes=_FACE_ON_AXES)
    quad = _quad(_camera_overlay(context))
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    # Regression: the image plane used to sit at map_radius * 0.015, which
    # cancels against the map scale and left a 6-10 px marker at the default
    # view -- the same size as the position dot, so the operator could not read
    # the camera heading off the map cloud at all.
    assert width == pytest.approx(CAMERA_VIEW_RECT_PX)
    assert height == pytest.approx(CAMERA_VIEW_RECT_PX / (1280 / 720))
    # ... and the correction must not swing the other way. The symbol sits on
    # top of the point cloud and the route, so it stays several times the 8 px
    # position dot for readability but well under the 72 px that hid them.
    assert 24.0 <= width <= 56.0
    # Face-on, the quad centres on the drone: the plane sits straight ahead
    # along this view's depth axis.
    assert (max(xs) + min(xs)) * 0.5 == pytest.approx(context.width * 0.5)
    assert (max(ys) + min(ys)) * 0.5 == pytest.approx(context.height * 0.5)


def test_camera_picture_keeps_one_calibrated_size_across_view_changes() -> None:
    baseline = _quad(_camera_overlay(_map_context(camera_axes=_FACE_ON_AXES)))
    baseline_width = max(pt[0] for pt in baseline) - min(pt[0] for pt in baseline)

    # The facing is free 3D, but the extent is normalised to the view scale:
    # neither vanishing at far zoom nor swallowing the cloud at near zoom.
    for overrides in (
        {"map_zoom": 0.4},
        {"map_zoom": 40.0},
        {"map_radius": 120.0},
        {"width": 440, "height": 140},
    ):
        quad = _quad(_camera_overlay(_map_context(camera_axes=_FACE_ON_AXES, **overrides)))
        width = max(pt[0] for pt in quad) - min(pt[0] for pt in quad)
        assert width == pytest.approx(baseline_width)


def test_camera_picture_seen_from_above_foreshortens_like_a_real_plane() -> None:
    from operator_rendering import (
        CAMERA_VIEW_ARROW_PX,
        CAMERA_VIEW_RECT_GAP_PX,
        CAMERA_VIEW_RECT_PX,
    )

    # Default axes: forward +X across the screen, right -Z straight into this
    # top-down view. The old flat screen-space symbol stayed 44 px deep here no
    # matter the attitude; a real plane collapses to its picture height only.
    context = _map_context()
    draw = _camera_overlay(context)
    quad = _quad(draw)
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    assert max(xs) - min(xs) == pytest.approx(0.0, abs=1e-9)
    assert max(ys) - min(ys) == pytest.approx(CAMERA_VIEW_RECT_PX / (1280 / 720))
    assert (max(xs) + min(xs)) * 0.5 == pytest.approx(
        context.width * 0.5 + CAMERA_VIEW_RECT_GAP_PX
    )

    # The direction arrow keeps its full screen length -- forward stays inside
    # the view's picture plane -- so the bearing stays readable edge-on.
    gaze = draw.lines[0][0]
    assert gaze[2] == pytest.approx(
        context.width * 0.5 + CAMERA_VIEW_RECT_GAP_PX + CAMERA_VIEW_ARROW_PX
    )
    assert gaze[2] > gaze[0]


def test_camera_overlay_marks_the_facing_and_the_drone_end() -> None:
    context = _map_context()
    draw = _camera_overlay(context)

    # Draw order: gaze line, picture fill, picture border, arrowhead, dot.
    assert len(draw.lines) == 2
    gaze, border = draw.lines
    assert gaze[1] == {"fill": "#ffffff", "width": 2}
    assert border[1] == {"fill": "#00d4ff", "width": 3, "joint": "curve"}
    # The arrowhead is the white fill beyond the picture: the front side.
    assert draw.polygons[1][1] == {"fill": "#ffffff"}

    assert len(draw.ellipses) == 1
    box = draw.ellipses[0][0]
    gaze_xy = gaze[0]
    assert (box[0] + box[2]) * 0.5 == pytest.approx(gaze_xy[0])
    assert (box[1] + box[3]) * 0.5 == pytest.approx(gaze_xy[1])


def test_camera_picture_border_colours_mark_the_orientation_source() -> None:
    # Cyan: a camera-measured attitude, from full axes or from the optical axis.
    assert _camera_overlay(_map_context()).lines[1][1]["fill"] == "#00d4ff"
    forward_context = _map_context(
        camera_axes=None, camera_forward=np.array([1.0, 0.0, 0.0])
    )
    assert _camera_overlay(forward_context).lines[1][1]["fill"] == "#00d4ff"
    # Green: yaw+gimbal telemetry fallback.
    yaw_context = _map_context(
        camera_axes=None, camera_forward=None, pose=(0.0, 0.0, 0.0, 0.0)
    )
    assert _camera_overlay(yaw_context).lines[1][1]["fill"] == "#3fbf7f"


def test_camera_overlay_falls_back_to_a_ring_without_any_orientation() -> None:
    # Default pose yaw is NaN: position without bearing.
    draw = _camera_overlay(_map_context(camera_axes=None, camera_forward=None))
    assert draw.polygons == []
    assert draw.lines == []
    assert len(draw.ellipses) == 1


def test_camera_picture_survives_a_degenerate_right_axis_row() -> None:
    # Nadir axes whose right row collapsed to zero -- what a raw
    # cross(forward, up) produces at exactly 90 deg of depression. The quad
    # must still draw, rebuilt against map north, not fall back to the ring.
    context = _map_context(
        camera_axes=np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
    )
    quad = _quad(_camera_overlay(context))
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    # Forward is -Z, this view's depth axis: the horizontal quad shows full.
    assert max(xs) - min(xs) == pytest.approx(44.0)
    assert max(ys) - min(ys) == pytest.approx(44.0 / (1280 / 720))


def _pitched_map_context(**overrides):
    """The operator map's real view: pitched 78 deg, gravity basis from river_site."""
    width, height = 900, 700
    map_zoom, map_radius = 1.0, 2.33
    map_yaw, map_pitch, map_roll = 0.0, (math.radians(78.0) + math.pi) % (2.0 * math.pi), 0.0

    def transform_xyz(points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        cy, sy = math.cos(map_yaw), math.sin(map_yaw)
        cp, sp = math.cos(map_pitch), math.sin(map_pitch)
        cr, sr = math.cos(map_roll), math.sin(map_roll)
        x1 = cy * pts[:, 0] + sy * pts[:, 2]
        z1 = -sy * pts[:, 0] + cy * pts[:, 2]
        y2 = cp * pts[:, 1] - sp * z1
        z2 = sp * pts[:, 1] + cp * z1
        return np.stack([cr * x1 - sr * y2, sr * x1 + cr * y2, z2], axis=1)

    def project_world(point, w, h):
        v = transform_xyz(point)[0]
        scale = min(w, h) * 0.46 * map_zoom / map_radius
        return int(w * 0.5 + v[0] * scale), int(h * 0.5 - v[1] * scale)

    # river_site's measured gravity: 4.78 deg off the legacy -Y assumption.
    up = np.array([0.010903244125662598, -0.996517323960189, -0.08267008113435093])
    up = up / np.linalg.norm(up)
    east = np.array([1.0, 0.0, 0.0]) - float(np.dot(np.array([1.0, 0.0, 0.0]), up)) * up
    east = east / np.linalg.norm(east)

    return _map_context(
        width=width, height=height, map_zoom=map_zoom, map_radius=map_radius,
        transform_xyz=transform_xyz, project_world=project_world,
        map_east=east, map_north=np.cross(up, east), map_up=up,
        **overrides,
    )


def _camera_axes_for(context, azimuth_deg: float, depression_deg: float) -> np.ndarray:
    """right/down/forward rows for a gimballed camera at a known bearing."""
    east = np.asarray(context.map_east, dtype=float)
    north = np.asarray(context.map_north, dtype=float)
    up = np.asarray(context.map_up, dtype=float)
    a, d = math.radians(azimuth_deg), math.radians(depression_deg)
    horizontal = math.cos(a) * east + math.sin(a) * north
    forward = math.cos(d) * horizontal + math.sin(d) * -up
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    return np.stack([right, np.cross(forward, right), forward])


def test_camera_picture_is_painted_where_the_map_projection_puts_it() -> None:
    """The overlay is anchored in the map's world, not a screen-space sticker.

    Ground truth: the app's own project_world applied to the pure geometry
    corners. Each drawn corner has to land within a pixel of where the pane
    puts that world point, or the operator reads the mark against the map and
    it sits where the camera is not.
    """
    from operator_rendering import (
        _camera_picture_basis,
        camera_view_plane_corners,
    )

    base = _pitched_map_context()
    for azimuth in range(0, 360, 45):
        for depression in (0.0, 30.0, 60.0, 85.0):
            context = _pitched_map_context(
                camera_axes=_camera_axes_for(base, azimuth, depression),
            )
            basis = _camera_picture_basis(context)
            assert basis is not None
            right, image_up, forward, _from_yaw = basis
            drone = np.asarray(context.pose[:3], dtype=float)
            scale = (
                min(context.width, context.height)
                * 0.46
                * context.map_zoom
                / context.map_radius
            )
            expected = camera_view_plane_corners(
                drone, right, image_up, forward, scale, context.video_aspect_ratio
            )
            drawn = _quad(_camera_overlay(context))
            for (world_x, world_y, _world_z), (drawn_x, drawn_y) in zip(expected, drawn):
                px, py = context.project_world(
                    (world_x, world_y, _world_z), context.width, context.height
                )
                assert abs(px - drawn_x) <= 1.0, (azimuth, depression)
                assert abs(py - drawn_y) <= 1.0, (azimuth, depression)


def test_camera_picture_points_where_the_stream_points() -> None:
    """Bearing checked against the map, not against the symbol's own math.

    Ground truth is the projection of where the optical axis reaches -- the
    centre plus a long step along the full 3D forward, pushed through the same
    transform the pane uses for the cloud and the route. Unlike the old
    plan-view mark, the 3D quad carries the depression too, so tilt must show.
    """
    base = _pitched_map_context()
    for azimuth in range(0, 360, 45):
        for depression in (0.0, 30.0, 60.0, 85.0):
            axes = _camera_axes_for(base, azimuth, depression)
            context = _pitched_map_context(camera_axes=axes)
            forward = np.asarray(axes[2], dtype=float)
            drone = np.asarray(context.pose[:3], dtype=float)
            here = context.project_world(drone, context.width, context.height)
            there = context.project_world(
                drone + 1.5 * forward, context.width, context.height
            )
            lever = math.hypot(there[0] - here[0], there[1] - here[1])
            if lever < 25.0:
                # Near the view axis the projected bearing itself is noise;
                # the positional test above still covers those attitudes.
                continue
            expected = math.degrees(
                math.atan2(-(there[1] - here[1]), there[0] - here[0])
            )
            gaze = _camera_overlay(context).lines[0][0]
            drawn = math.degrees(math.atan2(-(gaze[3] - gaze[1]), gaze[2] - gaze[0]))
            error = (drawn - expected + 180.0) % 360.0 - 180.0
            # project_world rounds to whole pixels on both lever ends, so the
            # angular budget widens as the projected lever shortens; on a long
            # lever it stays a fraction of a degree.
            budget = math.degrees(math.atan2(2.0, lever)) + 0.3
            assert abs(error) < budget, (azimuth, depression, error, lever)


def test_yaw_picture_follows_the_measured_gravity_basis() -> None:
    # The yaw fallback used to hard-code [cos, 0, sin], i.e. the legacy frame.
    # On a site with a measured basis that is a silent rotation of the symbol.
    # It must compose the heading off map east/north, landing exactly where an
    # explicit camera_forward with the same direction lands.
    yaw = math.radians(37.0)
    context = _pitched_map_context(
        camera_axes=None, camera_forward=None, pose=(0.0, 0.0, 0.0, yaw)
    )
    drawn = _quad(_camera_overlay(context))
    reference = _pitched_map_context(
        camera_axes=None,
        camera_forward=(
            math.cos(yaw) * np.asarray(context.map_east, dtype=float)
            + math.sin(yaw) * np.asarray(context.map_north, dtype=float)
        ),
    )
    assert drawn == pytest.approx(_quad(_camera_overlay(reference)))


def test_yaw_picture_tilts_with_the_telemetry_gimbal_pitch() -> None:
    from operator_rendering import _camera_picture_basis

    yaw = math.radians(37.0)
    level = _pitched_map_context(
        camera_axes=None, camera_forward=None, pose=(0.0, 0.0, 0.0, yaw)
    )
    tilted = _pitched_map_context(
        camera_axes=None, camera_forward=None, pose=(0.0, 0.0, 0.0, yaw),
        gimbal_pitch_deg=-45.0,
    )
    level_basis = _camera_picture_basis(level)
    tilted_basis = _camera_picture_basis(tilted)
    assert level_basis is not None and tilted_basis is not None

    horizontal = (
        math.cos(yaw) * np.asarray(level.map_east, dtype=float)
        + math.sin(yaw) * np.asarray(level.map_north, dtype=float)
    )
    up = np.asarray(level.map_up, dtype=float)
    # Gimbal pitch is positive up, so -45 deg looks 45 deg below the horizon.
    expected = (
        math.cos(math.radians(-45.0)) * horizontal
        + math.sin(math.radians(-45.0)) * up
    )
    assert np.allclose(level_basis[2], horizontal, atol=1e-12)
    assert np.allclose(tilted_basis[2], expected, atol=1e-12)

    # Junk telemetry degrades to the level picture instead of breaking the map.
    junk = _pitched_map_context(
        camera_axes=None, camera_forward=None, pose=(0.0, 0.0, 0.0, yaw),
        gimbal_pitch_deg="not-a-number",
    )
    junk_basis = _camera_picture_basis(junk)
    assert junk_basis is not None
    assert np.allclose(junk_basis[2], horizontal, atol=1e-12)

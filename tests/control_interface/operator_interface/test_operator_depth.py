"""Depth must never attach a different capture's pose to displayed pixels."""
from collections import OrderedDict
import time

import numpy as np
import pytest

from operator_depth import (
    DepthCapture,
    DepthFrame,
    DepthInspector,
    DepthViewState,
    depth_to_color,
    draw_depth_view,
    nearest_in_grid,
    paired_capture,
)


def _inputs():
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frames = {3: (frame, "capture-3", 12.)}
    result = {"success": True, "seq": 98, "display_seq": 3, "frame_name": "capture-3",
              "capture_mono_ns": 12_000_000_000, "pose_capture_mono_ns": 12_000_000_000,
              "camera_attitude_kind": "full", "camera_axes_world": np.eye(3).tolist(),
              "pose": {"x": 1., "y": 2., "z": 3.}, "mode": "TRACK"}
    return frame, frames, result


def test_capture_pairs_display_sequence_not_worker_request_counter():
    frame, frames, result = _inputs()
    capture = paired_capture(result, frames, 0.7)
    assert capture.frame is frame
    np.testing.assert_array_equal(capture.center, [1., 2., 3.])
    assert capture.meters_per_map_unit == 0.7


def test_depth_uses_capture_pose_not_smoothed_display_position():
    _, frames, result = _inputs()
    result["pose_raw"] = {"x": 8., "y": 9., "z": 10.}
    capture = paired_capture(result, frames, 0.7)
    np.testing.assert_array_equal(capture.center, [8., 9., 10.])


@pytest.mark.parametrize("changes", [
    {"display_seq": 4}, {"frame_name": "another-capture"}, {"capture_mono_ns": 13_000_000_000},
    {"pose_capture_mono_ns": 11_900_000_000}, {"pose": None},
    {"camera_attitude_kind": "heading_only"},
    {"camera_axes_world": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]},
    {"pose": {"x": float("nan"), "y": 2, "z": 3}},
])
def test_mismatched_held_or_unmeasured_pose_is_not_inspectable(changes):
    _, frames, result = _inputs()
    result.update(changes)
    assert paired_capture(result, frames) is None


@pytest.mark.parametrize("changes", [
    {"mode": "WEAK_TRACK", "weak": True}, {"success": False},
])
def test_weak_fix_with_own_pose_and_full_attitude_is_inspectable(changes):
    _, frames, result = _inputs()
    result.update(changes)
    assert paired_capture(result, frames) is not None


def test_unknown_scale_is_preserved_not_defaulted_to_one():
    _, frames, result = _inputs()
    assert paired_capture(result, frames).meters_per_map_unit is None
    assert paired_capture(result, frames, float("nan")) is None
    assert paired_capture(result, frames, True) is None


def test_frame_retention_evicts_old_capture_and_reset_forgets_old_site():
    inspector = object.__new__(DepthInspector)
    inspector.bundle = "bundle"
    inspector.closed = False
    inspector.frames = OrderedDict()
    inspector.window = None
    inspector.generation = 0
    inspector.depth = inspector.latest = inspector.selected = None
    for seq in range(20):
        inspector.remember(seq, str(seq), 12. + seq, np.zeros((4, 8, 3)))
    assert 0 not in inspector.frames and 19 in inspector.frames
    inspector.reset()
    _, _, result = _inputs()
    assert paired_capture(result, inspector.frames) is None


def test_grid_reports_nearest_known_depth_and_pixel_per_cell():
    depth = np.full((6, 9), 10., dtype=np.float32)
    depth[1, 4] = 2.5               # cell (0, 1)
    depth[5, 8] = 0.5               # cell (1, 2)
    depth[0, 0] = np.nan            # unknown never wins
    depth[1, 1] = 0.                # zero depth is unknown, not "touching"
    depth[3:6, 0:3] = np.nan        # a whole cell without surface
    cells = nearest_in_grid(depth, rows=2, cols=3)
    assert [c.distance_m for c in cells] == [10., 2.5, 10., None, 10., 0.5]
    assert cells[1].nearest_xy == (4, 1)
    assert cells[5].nearest_xy == (8, 5)
    assert cells[3].nearest_xy is None
    assert cells[5].bounds == (6, 3, 9, 6)


def test_colour_map_warm_near_cool_far_black_unknown():
    colour = depth_to_color(np.array([[1., 29., np.nan, 0.]], dtype=np.float32))
    near, far, nan, zero = colour[0].astype(int)
    assert near[0] > near[2] and far[2] > far[0]
    assert not nan.any() and not zero.any()


class _Surface:
    def __init__(self, depth):
        self.depth, self.calls = depth, []

    def render(self, camera, center, axes, **kwargs):
        self.calls.append((center.copy(), kwargs))
        return self.depth


def _view_inspector(surface):
    inspector = object.__new__(DepthInspector)
    inspector.bundle, inspector.closed, inspector.executor = "bundle", False, None
    inspector.camera = {"width": 8, "height": 4}
    inspector.surface, inspector.latest, inspector.frames = surface, None, OrderedDict()
    return inspector


def _capture(center, scale=0.7, frame=None):
    frame = np.zeros((4, 8, 3), dtype=np.uint8) if frame is None else frame
    return DepthCapture(frame, "f", 1., np.array(center, dtype=float), np.eye(3), scale)


def _await_view(inspector, accept):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = inspector.view_frame()
        if accept(state):
            return state
        time.sleep(0.01)
    raise AssertionError("depth view did not settle")


def test_view_renders_newest_capture_pose_and_scale_only_while_active():
    surface = _Surface(np.full((4, 8), 3., dtype=np.float32))
    inspector = _view_inspector(surface)
    inspector.latest = _capture([1., 2., 3.])
    assert inspector.view_frame() is None and not surface.calls
    assert inspector.toggle_view() is True
    state = _await_view(inspector, lambda s: s.frame is not None)
    assert state.frame.capture is inspector.latest and state.message is None
    np.testing.assert_array_equal(surface.calls[0][0], [1., 2., 3.])
    assert surface.calls[0][1]["meters_per_map_unit"] == 0.7
    inspector.latest = _capture([4., 5., 6.])
    state = _await_view(inspector, lambda s: s.frame.capture is inspector.latest)
    np.testing.assert_array_equal(surface.calls[-1][0], [4., 5., 6.])
    assert inspector.toggle_view() is False and inspector.view_frame() is None
    inspector.executor.shutdown()


def test_view_explains_missing_capture_surface_failure_and_crop():
    class _Broken:
        def render(self, *args, **kwargs):
            raise ValueError("mesh unreadable")

    inspector = _view_inspector(_Broken())
    inspector.toggle_view()
    assert "等待" in inspector.view_frame().message
    inspector.latest = _capture([0., 0., 0.])
    state = _await_view(inspector, lambda s: s.message and "mesh unreadable" in s.message)
    assert state.frame is None
    inspector.latest = _capture([0., 0., 0.], frame=np.zeros((4, 7, 3), dtype=np.uint8))
    assert "比例不符" in inspector.view_frame().message
    inspector.executor.shutdown()


def test_unscaled_capture_still_renders_depth_in_map_units():
    surface = _Surface(np.array([[0.5, 1., 2., np.nan]] * 4, dtype=np.float32))
    inspector = _view_inspector(surface)
    inspector.latest = _capture([0., 0., 0.], scale=None)
    inspector.toggle_view()
    state = _await_view(inspector, lambda s: s.frame is not None)
    assert surface.calls[0][1]["meters_per_map_unit"] == 1.
    assert state.frame.unit == "u" and 0.5 < state.frame.range_max <= 2.
    inspector.executor.shutdown()


class _Draw:
    def __init__(self):
        self.polygons, self.texts, self.rectangles = [], [], []

    def polygon(self, points, **options):
        self.polygons.append((points, options))

    def text(self, xy, text, **options):
        self.texts.append((xy, text, options))

    def rectangle(self, box, **options):
        self.rectangles.append(box)


def test_overlay_places_stars_and_distances_in_canvas_pixels():
    depth = np.full((6, 9), 10., dtype=np.float32)
    depth[4, 7] = 1.25
    cells = nearest_in_grid(depth)
    frame = DepthFrame(_capture([0., 0., 0.]), depth_to_color(depth), cells)
    draw = _Draw()
    draw_depth_view(draw, DepthViewState(frame, None), 2., 100, 50, None, None, 1.5)
    assert len(draw.rectangles) == 9 and len(draw.polygons) == 9
    shown = [text for _, text, options in draw.texts if options["fill"] == "#ffffff"]
    assert shown.count("1.2 m") == 1 and shown.count("10.0 m") == 8
    star, options = next(p for p in draw.polygons if p[1]["fill"] == "#ff2a2a")
    xs, ys = [x for x, _ in star], [y for _, y in star]
    assert (sum(xs[::2]) / 5, sum(ys[::2]) / 5) == pytest.approx((100 + 7.5 * 2, 50 + 4.5 * 2), abs=1.5)
    assert sum(1 for _, o in draw.polygons if o["fill"] == "#ff2a2a") == 1

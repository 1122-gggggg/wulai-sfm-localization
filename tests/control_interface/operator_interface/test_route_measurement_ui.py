from __future__ import annotations

import math
import queue
import re
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from route_editor_controller import OrthoView, RouteEditorController
from route_editor_model import MapDistanceScale, RouteDocument
from route_editor_objects import ObjectBoxController
from route_editor_window import RouteEditorWindow


class Variable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Widget:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)

    def focus_set(self):
        pass


class DistanceList(Widget):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.scroll = 0.0

    def yview(self):
        return self.scroll, 1.0

    def yview_moveto(self, value):
        self.scroll = value

    def delete(self, first, last):
        self.rows.clear()

    def insert(self, where, value):
        self.rows.append(value)


def make_editor(points=(), scale=None):
    """Real interaction/label methods, with only Tk and rendering replaced."""
    editor = SimpleNamespace(
        controller=RouteEditorController(points),
        document=RouteDocument(site_id="yard", coordinate_frame_id="frame"),
        objects=ObjectBoxController(),
        _boxing=False,
        _box_draw_start=None,
        _box_rectangle=None,
        profile=SimpleNamespace(display_name="yard"),
        distance_scale=scale,
        distance_scale_loader=lambda profile: scale,
        measurement_points=[],
        _measurement_selected=None,
        _measurement_move_start=None,
        _measurement_move_axis=None,
        _measuring=False,
        _closed=False,
        _asset_loading=False,
        _preview_only=False,
        _single_click_after=None,
        _left_press=None,
        _left_last=None,
        _left_dragging=False,
        _left_confirmed_move=False,
        _suppress_left_release=False,
        _mouse=(0, 0),
        _move_mouse_start=(0, 0),
        map_frame=None,
        map_loaded_callback=None,
        map_points=np.array([[0, 0, 0, 190, 190, 190]], dtype=np.float32),
        view=OrthoView(center=np.zeros(3), radius=10),
        route_distance_list=DistanceList(),
        test_button=None,
        canvas=Widget(),
        redraws=[],
        pending={},
        cancelled=[],
        title=lambda value: None,
    )
    for name in (
        "mode_var", "selection_var", "distance_scale_var", "measurement_var",
        "route_total_var", "status_var", "title_var",
    ):
        setattr(editor, name, Variable(""))
    editor.measurement_mode_var = Variable(False)
    editor.marking_mode_var = Variable(False)
    editor.arrive_radius_var = Variable(0.02)
    editor.show_arrive_var = Variable(False)
    for name in (
        "finish_button", "marking_mode_button", "measurement_mode_button",
        "save_button", "ply_button",
    ):
        setattr(editor, name, Widget())
    editor._sync_point_radius_box = lambda: None
    editor._sync_action_list = lambda: None
    editor._refresh_object_labels = lambda: None
    editor._on_arrive_radius = lambda value: None
    editor.request_redraw = lambda: editor.redraws.append(True)
    editor._canvas_size = lambda: (800, 600)
    editor._bounds = RouteEditorWindow._bounds

    def after(delay, callback):
        handle = str(len(editor.pending) + len(editor.cancelled))
        editor.pending[handle] = callback
        return handle

    def after_cancel(handle):
        editor.cancelled.append(handle)
        editor.pending.pop(handle, None)

    editor.after = after
    editor.after_cancel = after_cancel
    for name in (
        "_signature", "_refresh_labels", "_refresh_distance_labels", "_format_distance",
        "toggle_measurement_mode", "toggle_marking_mode", "clear_measurement",
        "_measure_at", "_handle_single_left", "_cancel_pending_single_click",
        "_handle_escape", "_handle_begin_move", "_handle_move_axis", "_handle_delete", "_handle_enter",
        "_handle_undo_redo", "_update_move_preview", "finish_layout", "on_motion",
        "on_left_press", "on_left_drag", "on_left_release", "on_left_double",
        "_poll_map_load", "_aligned_map_points", "on_key", "_repair_look_at_references",
        "_clear_object_rectangle",
    ):
        setattr(editor, name, MethodType(getattr(RouteEditorWindow, name), editor))
    editor._nearest_waypoint = lambda x, y: None
    editor._nearest_cloud = lambda x, y, threshold=20: [float(x), float(y), 0.0]
    editor._refresh_labels()
    return editor


def enable_measurement(editor):
    editor.measurement_mode_var.set(True)
    editor.toggle_measurement_mode()


def display_values(text):
    # Compare numeric results, independent of translated wording and units.
    return [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text.split("：")[-1])]


def test_measurement_uses_waypoint_height_without_changing_route_or_history():
    scale = MapDistanceScale(2.0, True, "offline", 0.1)
    editor = make_editor([[0, 0, 0], [3, 0, 4]], scale)
    editor._nearest_waypoint = lambda x, y: int(x)
    editor._nearest_cloud = lambda x, y: pytest.fail("Waypoint must win over cloud")
    signature = editor._signature()
    document_state = repr(editor.document)
    history = repr((editor.controller._undo, editor.controller._redo))

    enable_measurement(editor)
    editor._handle_single_left(0, 0)
    editor._handle_single_left(1, 0)

    assert editor.measurement_points == [[0, 0, 0], [3, 0, 4]]
    assert editor.measurement_distance_u == pytest.approx(5)
    assert editor.measurement_distance_m == pytest.approx(10)
    assert display_values(editor.measurement_var.get()) == pytest.approx([5, 10])
    assert editor._signature() == signature
    assert repr(editor.document) == document_state
    assert repr((editor.controller._undo, editor.controller._redo)) == history


def test_third_valid_click_starts_a_new_pair_and_empty_click_preserves_pair():
    editor = make_editor(scale=MapDistanceScale(3.0, True, "offline"))
    enable_measurement(editor)
    editor._handle_single_left(0, 0)
    editor._handle_single_left(3, 4)
    pair = [point.copy() for point in editor.measurement_points]
    editor._nearest_cloud = lambda x, y: None
    editor._handle_single_left(100, 100)
    assert editor.measurement_points == pair
    editor._nearest_cloud = lambda x, y: [x, y, 7]
    editor._handle_single_left(1, 2)
    assert editor.measurement_points == [[1, 2, 7]]
    assert editor.measurement_distance_u is None
    assert editor.measurement_distance_m is None
    assert editor.controller.points == []


def test_missing_scale_keeps_units_without_inventing_meters():
    editor = make_editor([[0, 0, 0], [3, 0, 4]])
    enable_measurement(editor)
    editor._handle_single_left(0, 0)
    editor._handle_single_left(3, 4)
    assert editor.measurement_distance_u == pytest.approx(5)
    assert editor.measurement_distance_m is None
    assert editor.route_segment_distances_u == pytest.approx([5])
    assert editor.route_segment_distances_m is None
    assert editor.route_total_m is None
    assert display_values(editor.measurement_var.get()) == pytest.approx([5])
    assert display_values(editor.route_distance_list.rows[0]) == pytest.approx([5])
    assert display_values(editor.route_total_var.get()) == pytest.approx([5])


def test_modes_are_mutually_exclusive_and_cancel_delayed_placement():
    editor = make_editor()
    editor.marking_mode_var.set(True)
    editor.toggle_marking_mode()
    editor.on_left_press(SimpleNamespace(x=0, y=0))
    editor.on_left_release(SimpleNamespace(x=0, y=0))
    handle = editor._single_click_after
    assert editor.controller.placement_enabled
    enable_measurement(editor)
    assert handle in editor.cancelled
    assert not editor.pending
    assert not editor.controller.placement_enabled
    assert not editor.marking_mode_var.get()
    editor._handle_single_left(0, 0)
    assert editor.controller.points == []
    editor.marking_mode_var.set(True)
    editor.toggle_marking_mode()
    assert editor.controller.placement_enabled
    assert not editor._measuring
    assert not editor.measurement_mode_var.get()
    assert editor.measurement_points == []


def test_move_must_be_confirmed_or_cancelled_before_measurement():
    editor = make_editor([[0, 0, 0], [1, 0, 0]])
    editor._handle_begin_move()
    editor.controller.preview_move([0, 0, 3])
    before = editor.controller.points[0].copy()
    enable_measurement(editor)
    assert not editor._measuring
    assert not editor.measurement_mode_var.get()
    assert editor.controller.moving
    assert editor.controller.points[0] == before
    editor._handle_escape()
    assert not editor.controller.moving
    assert editor.controller.points[0] == [0, 0, 0]
    enable_measurement(editor)
    editor._handle_single_left(0, 0)
    editor._handle_escape()
    assert not editor._measuring
    assert editor.measurement_points == []
    assert not editor._closed

def test_measurement_mode_does_not_edit_route_through_keyboard_shortcuts():
    editor = make_editor([[0, 0, 0], [3, 0, 4]], MapDistanceScale(2.0, True, "offline"))
    editor.finish_layout()
    editor.controller.selected = 1
    signature = editor._signature()
    enable_measurement(editor)
    editor._handle_single_left(0, 0)
    for key, state in (("g", 0), ("Delete", 0), ("z", 4), ("Return", 0)):
        editor.on_key(SimpleNamespace(widget=None, keysym=key, state=state))
        assert editor._signature() == signature
        assert not editor.controller.moving
        assert editor._measuring
    editor.on_key(SimpleNamespace(widget=None, keysym="Escape", state=0))
    assert not editor._measuring
    assert editor.measurement_points == []
    assert not editor._closed



def test_drag_and_double_click_do_not_count_as_measurements():
    editor = make_editor()
    enable_measurement(editor)
    editor.on_left_press(SimpleNamespace(x=0, y=0))
    editor.on_left_drag(SimpleNamespace(x=15, y=10))
    editor.on_left_release(SimpleNamespace(x=15, y=10))
    assert editor.view.yaw != 0
    assert editor.measurement_points == []
    assert not editor.pending
    editor.on_left_press(SimpleNamespace(x=3, y=4))
    editor.on_left_release(SimpleNamespace(x=3, y=4))
    handle = editor._single_click_after
    editor.on_left_double(SimpleNamespace(x=3, y=4))
    editor.on_left_release(SimpleNamespace(x=3, y=4))
    assert handle in editor.cancelled
    assert not editor.pending
    assert editor.measurement_points == []
    np.testing.assert_allclose(editor.view.center, [3, 4, 0])


@pytest.mark.parametrize("new_scale", [None, MapDistanceScale(4.0, True, "new")])
def test_map_rebinding_clears_measurement_and_reloads_scale(new_scale, tmp_path):
    editor = make_editor(scale=MapDistanceScale(2.0, True, "old"))
    enable_measurement(editor)
    editor._handle_single_left(0, 0)
    editor._handle_single_left(3, 4)
    editor.on_left_release(SimpleNamespace(x=0, y=0))
    loaded_profiles = []
    profile = (
        None if new_scale is None else SimpleNamespace(
            coordinate_frame=SimpleNamespace(id="new-frame"),
            site_id="new-site", display_name="new-site", map_align=None,
        )
    )
    editor.distance_scale_loader = lambda value: loaded_profiles.append(value) or new_scale
    editor._asset_load_queue = queue.Queue()
    editor._asset_load_queue.put((
        profile, np.array([[1, 2, 3]]), tmp_path / "new.ply", "loaded", None,
    ))
    editor._poll_map_load()
    assert loaded_profiles == [profile]
    assert editor.distance_scale is new_scale
    assert editor.measurement_points == []
    assert not editor._measuring
    assert editor.measurement_distance_m is None
    assert not editor.pending
    assert editor.controller.points == []
    assert editor.route_segment_distances_u == []
    assert editor.route_segment_distances_m == ([] if new_scale is not None else None)


def assert_route_display(editor, expected):
    assert editor.route_segment_distances_u == pytest.approx(expected)
    assert editor.route_segment_distances_m == pytest.approx([2 * value for value in expected])
    assert editor.route_total_u == pytest.approx(sum(expected))
    assert editor.route_total_m == pytest.approx(2 * sum(expected))
    assert len(editor.route_distance_list.rows) == len(expected)
    for row, value in zip(editor.route_distance_list.rows, expected):
        assert display_values(row) == pytest.approx([value, 2 * value], abs=0.0006)
    assert display_values(editor.route_total_var.get()) == pytest.approx(
        [sum(expected), 2 * sum(expected)], abs=0.0006,
    )


def test_all_open_route_segments_update_after_add_move_delete_undo_and_redo():
    editor = make_editor(scale=MapDistanceScale(2.0, True, "offline"))
    editor.marking_mode_var.set(True)
    editor.toggle_marking_mode()
    for x, y in ((0, 0), (3, 4), (3, 8)):
        editor._handle_single_left(x, y)
    assert_route_display(editor, [5, 4])
    editor.finish_layout()
    editor.controller.selected = 1
    editor._handle_begin_move()
    editor._mouse = (0, -editor.view.scale(800, 600) * 2)
    editor._update_move_preview()
    assert_route_display(editor, [math.sqrt(45), 2])
    editor._handle_enter()
    editor._handle_undo_redo(shift=False)
    assert_route_display(editor, [5, 4])
    editor._handle_undo_redo(shift=True)
    assert_route_display(editor, [math.sqrt(45), 2])
    editor._handle_delete()
    assert_route_display(editor, [math.sqrt(73)])
    editor._handle_undo_redo(shift=False)
    assert_route_display(editor, [math.sqrt(45), 2])
    editor._handle_undo_redo(shift=True)
    assert_route_display(editor, [math.sqrt(73)])


def test_measurement_cloud_picking_preserves_depth():
    editor = make_editor(scale=MapDistanceScale(2.0, True, "offline"))
    editor.map_points = np.array([
        [0, 0, -3, 190, 190, 190],
        [0, 0, 2, 190, 190, 190],
        [1, 0, 6, 190, 190, 190],
    ], dtype=np.float32)
    editor._nearest_cloud = MethodType(RouteEditorWindow._nearest_cloud, editor)
    editor._nearest_waypoint = MethodType(RouteEditorWindow._nearest_waypoint, editor)
    enable_measurement(editor)
    editor._handle_single_left(400, 300)
    editor._handle_single_left(428, 300)
    assert editor.measurement_points == [[0, 0, 2], [1, 0, 6]]
    assert editor.measurement_distance_u == pytest.approx(math.sqrt(17))
    assert editor.measurement_distance_m == pytest.approx(2 * math.sqrt(17))




@pytest.fixture
def measurement_editor():
    editor = make_editor([[3, 4, 5], [8, 9, 10]], MapDistanceScale(2.0, True, "offline"))
    enable_measurement(editor)
    editor.measurement_points = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    editor._measurement_selected = 1
    editor._mouse = (100, 100)
    editor._refresh_distance_labels()
    return editor


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("orientation", ["top", "front", "right"])
def test_measurement_ball_axis_move_preserves_other_coordinates_and_route(
    measurement_editor, axis, orientation
):
    editor = measurement_editor
    getattr(editor.view, orientation)()
    original = np.array(editor.measurement_points[1])
    route = [point.copy() for point in editor.controller.points]
    editor.on_key(SimpleNamespace(widget=None, keysym="g", state=0))
    editor.on_key(SimpleNamespace(widget=None, keysym=axis, state=0))
    editor.on_motion(SimpleNamespace(x=180, y=40))

    coordinate = {"x": 0, "y": 1, "z": 2}[axis]
    # At this view scale, horizontal pixels move X except in right view,
    # where Y is horizontal. Edge-on axes retain the vertical-drag fallback.
    pixels = 80 if (axis == "x" and orientation != "right") or (axis == "y" and orientation == "right") else 60
    expected = original.copy()
    expected[coordinate] += pixels / 27.6
    np.testing.assert_allclose(editor.measurement_points[1], expected)
    distance = float(np.linalg.norm(expected))
    assert editor.measurement_distance_u == pytest.approx(distance)
    assert editor.measurement_distance_m == pytest.approx(2 * distance)
    editor.on_key(SimpleNamespace(widget=None, keysym="Return", state=0))
    np.testing.assert_allclose(editor.measurement_points[1], expected)
    assert editor._measurement_move_start is None
    assert editor._measuring
    assert editor.controller.points == route


def test_measurement_ball_escape_rolls_back_without_leaving_mode_or_touching_route(
    measurement_editor,
):
    editor = measurement_editor
    original = [point.copy() for point in editor.measurement_points]
    route = [point.copy() for point in editor.controller.points]
    editor.on_key(SimpleNamespace(widget=None, keysym="g", state=0))
    editor.on_motion(SimpleNamespace(x=180, y=40))
    editor.on_key(SimpleNamespace(widget=None, keysym="Escape", state=0))

    assert editor.measurement_points == original
    assert editor.measurement_distance_u == pytest.approx(math.sqrt(14))
    assert editor.measurement_distance_m == pytest.approx(2 * math.sqrt(14))
    assert editor._measuring
    assert editor._measurement_move_start is None
    assert editor.controller.points == route


def test_measurement_ball_cannot_move_without_a_selected_endpoint(measurement_editor):
    editor = measurement_editor
    editor._measurement_selected = None
    original = [point.copy() for point in editor.measurement_points]
    editor.on_key(SimpleNamespace(widget=None, keysym="g", state=0))
    editor.on_motion(SimpleNamespace(x=180, y=40))
    assert editor.measurement_points == original
    assert editor._measurement_move_start is None

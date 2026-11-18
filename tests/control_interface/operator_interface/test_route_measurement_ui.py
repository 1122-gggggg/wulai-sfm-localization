from __future__ import annotations

import math
import queue
import re
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

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
        self.horizontal = 0.0
        self.selected = set()
        self.active = 0
        self.anchor = 0
        self.edits = []

    def yview(self):
        return self.scroll, 1.0

    def yview_moveto(self, value):
        self.scroll = value

    def xview(self):
        return self.horizontal, 1.0

    def xview_moveto(self, value):
        self.horizontal = value

    def get(self, first, last):
        return tuple(self.rows)

    def delete(self, first, last=None):
        self.edits.append(("delete", first, last))
        stop = len(self.rows) if last == "end" else first + 1 if last is None else last + 1
        del self.rows[first:stop]
        self.selected = {
            index if index < first else index - (stop - first)
            for index in self.selected if index < first or index >= stop
        }

    def insert(self, where, value):
        index = len(self.rows) if where == "end" else where
        self.edits.append(("insert", index, value))
        self.rows.insert(index, value)
        self.selected = {old + 1 if old >= index else old for old in self.selected}

    def curselection(self):
        return tuple(sorted(self.selected))

    def index(self, where):
        return self.active if where == "active" else self.anchor

    def selection_clear(self, first, last):
        self.selected.clear()

    def selection_set(self, index):
        self.selected.add(index)

    def activate(self, index):
        self.active = index

    def selection_anchor(self, index):
        self.anchor = index


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
        _cloud_revision=0,
        _raster_cache_key=None,
        _raster_cache=None,
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
        "route_total_var", "status_var", "title_var", "map_status_var", "save_note_var",
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
        "_update_distance_list", "_replace_distance_list_rows",
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


def enable_frame_redraw(editor):
    """Keep Tk timers/real redraw cadence; isolate only the cloud display."""
    for name in ("request_redraw", "_redraw", "_poll_render", "_current_raster_key"):
        setattr(editor, name, MethodType(getattr(RouteEditorWindow, name), editor))
    editor._render_future = None
    editor._render_after = None
    editor._redraw_pending = False
    editor._raster_cache_key = editor._current_raster_key()
    editor._raster_cache = object()
    editor._paint_render = lambda *args: None


def block_cloud_render(editor):
    """Start a real render poll around a deterministic, unfinished worker future."""
    future = Future()
    editor._render_executor = Mock()
    editor._render_executor.submit.return_value = future
    editor._cloud_worker = SimpleNamespace(render=Mock())
    editor._raster_cache_key = None
    editor.request_redraw()
    run_frame(editor)
    return future


def run_frame(editor):
    callbacks = list(editor.pending.values())
    editor.pending.clear()
    for callback in callbacks:
        callback()


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
        editor.on_key(SimpleNamespace(widget=editor.canvas, keysym=key, state=state))
        assert editor._signature() == signature
        assert not editor.controller.moving
        assert editor._measuring
    editor.on_key(SimpleNamespace(widget=editor.canvas, keysym="Escape", state=0))
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
    enable_frame_redraw(editor)
    editor._mouse = (0, -editor.view.scale(800, 600) * 2)
    editor._update_move_preview()
    run_frame(editor)
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
    editor.on_key(SimpleNamespace(widget=editor.canvas, keysym="g", state=0))
    editor.on_key(SimpleNamespace(widget=editor.canvas, keysym=axis, state=0))
    enable_frame_redraw(editor)
    editor.on_motion(SimpleNamespace(x=180, y=40))
    run_frame(editor)

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
    editor.on_key(SimpleNamespace(widget=editor.canvas, keysym="Return", state=0))
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
    editor.on_key(SimpleNamespace(widget=editor.canvas, keysym="g", state=0))
    editor.on_motion(SimpleNamespace(x=180, y=40))
    editor.on_key(SimpleNamespace(widget=editor.canvas, keysym="Escape", state=0))

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
    editor.on_key(SimpleNamespace(widget=editor.canvas, keysym="g", state=0))
    editor.on_motion(SimpleNamespace(x=180, y=40))
    assert editor.measurement_points == original
    assert editor._measurement_move_start is None


def test_drag_sidebar_coalesces_live_preview_and_preserves_unchanged_rows():
    editor = make_editor(
        [[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]],
        MapDistanceScale(2.0, True, "offline"),
    )
    editor.controller.selected = 1
    editor._handle_begin_move()
    enable_frame_redraw(editor)
    rows = editor.route_distance_list
    rows.scroll = 0.45
    rows.horizontal = 0.35
    rows.selected = {1, 2}
    rows.active = 2
    rows.anchor = 1
    rows.edits.clear()
    original_rows = rows.rows.copy()
    for y in range(1, 101):
        editor.on_motion(SimpleNamespace(x=0, y=y))
    assert rows.edits == []
    assert rows.rows == original_rows
    assert len(editor.pending) == 1
    # Coordinates still follow every pointer event, not only committed points.
    assert f"Y {editor.controller.points[1][1]:.3f}" in editor.selection_var.get()
    run_frame(editor)
    expected = [
        math.dist(first, second)
        for first, second in zip(editor.controller.points, editor.controller.points[1:])
    ]
    assert_route_display(editor, expected)
    assert rows.rows[2] == original_rows[2]
    assert [edit[1] for edit in rows.edits if edit[0] == "delete"] == [0, 1]
    assert rows.scroll == 0.45 and rows.horizontal == 0.35
    assert rows.curselection() == (1, 2)
    assert rows.active == 2 and rows.anchor == 1
    rows.edits.clear()
    editor._refresh_distance_labels()
    assert rows.edits == []



@pytest.mark.parametrize("measurement", [False, True])
def test_blocked_cloud_render_keeps_sidebar_live_and_cancels_poll_on_close(
    measurement_editor, measurement,
):
    editor = measurement_editor if measurement else make_editor(
        [[0, 0, 0], [3, 0, 0], [6, 0, 0]], MapDistanceScale(2.0, True, "offline"),
    )
    editor.controller.selected = 1
    editor._handle_begin_move()
    enable_frame_redraw(editor)
    future = block_cloud_render(editor)
    for start in (0, 60):
        previous = (editor.route_total_u, editor.measurement_distance_u)
        for offset in range(start + 1, start + 61):
            editor.on_motion(SimpleNamespace(x=100 + offset, y=100 - offset))
        assert (editor.route_total_u, editor.measurement_distance_u) == previous
        assert editor._distance_labels_dirty
        assert len(editor.pending) == 1
        run_frame(editor)
        assert not future.done()
        assert not editor._distance_labels_dirty
        if measurement:
            expected = math.dist(*editor.measurement_points)
            assert editor.measurement_distance_u == pytest.approx(expected)
            assert editor.measurement_distance_m == pytest.approx(2 * expected)
            assert display_values(editor.measurement_var.get()) == pytest.approx(
                [expected, 2 * expected], abs=0.0006,
            )
        else:
            expected = [
                math.dist(first, second)
                for first, second in zip(editor.controller.points, editor.controller.points[1:])
            ]
            assert_route_display(editor, expected)
        editor._render_executor.submit.assert_called_once()
    handle, callback = next(iter(editor.pending.items()))
    editor.attributes = Mock()
    editor.destroy = Mock()
    editor.on_close_callback = None
    RouteEditorWindow.close_editor(editor, force=True)
    assert handle in editor.cancelled
    assert editor.pending == {}
    editor._render_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    callback()
    assert editor.pending == {}
    editor.destroy.assert_called_once()

@pytest.mark.parametrize("blocked_render", [False, True])
@pytest.mark.parametrize("finish", ["enter", "click", "escape", "save"])
def test_pending_preview_finalization_forces_exact_route_distances(
    finish, tmp_path, blocked_render,
):
    import json

    editor = make_editor(
        [[0, 0, 0], [3, 0, 0], [6, 0, 0]], MapDistanceScale(2.0, True, "offline"),
    )
    editor.controller.selected = 1
    editor._handle_begin_move()
    enable_frame_redraw(editor)
    if blocked_render:
        block_cloud_render(editor)
    editor.on_motion(SimpleNamespace(x=0, y=40))
    run_frame(editor)
    editor.on_motion(SimpleNamespace(x=0, y=91))
    preview = [point.copy() for point in editor.controller.points]
    assert len(editor.pending) == 1
    if finish == "enter":
        editor._handle_enter()
    elif finish == "click":
        editor.on_left_press(SimpleNamespace(x=0, y=91))
    elif finish == "escape":
        editor._handle_escape()
    else:
        editor._preview_only = True
        target = tmp_path / "route_preview.json"
        editor._choose_preview_path = lambda: target
        editor.arrive_radius_var.set(0.02)
        editor._update_document = MethodType(RouteEditorWindow._update_document, editor)
        saved = RouteEditorWindow.save_route(editor)
        assert saved == target
        assert json.loads(target.read_text())["purpose"] == "preview_only"
        assert editor.document.points == preview
    final = [[0, 0, 0], [3, 0, 0], [6, 0, 0]] if finish == "escape" else preview
    expected = [math.dist(first, second) for first, second in zip(final, final[1:])]
    assert editor.controller.points == final
    assert not editor.controller.moving
    assert_route_display(editor, expected)
    run_frame(editor)
    assert_route_display(editor, expected)


@pytest.mark.parametrize("finish", ["enter", "click", "escape"])
@pytest.mark.parametrize("blocked_render", [False, True])
def test_measurement_pending_preview_finalization_has_exact_displayed_distance(
    measurement_editor, finish, blocked_render,
):
    editor = measurement_editor
    original = [point.copy() for point in editor.measurement_points]
    route = [point.copy() for point in editor.controller.points]
    editor._handle_begin_move()
    enable_frame_redraw(editor)
    if blocked_render:
        block_cloud_render(editor)
    for offset in range(1, 61):
        editor.on_motion(SimpleNamespace(x=100 + offset, y=100 - offset))
    assert len(editor.pending) == 1
    if finish == "escape":
        editor._handle_escape()
        assert editor.measurement_points == original
    elif finish == "enter":
        editor._handle_enter()
    else:
        editor.on_left_press(SimpleNamespace(x=160, y=40))
    expected = math.dist(*editor.measurement_points)
    assert editor.measurement_distance_u == pytest.approx(expected)
    assert editor.measurement_distance_m == pytest.approx(2 * expected)
    assert display_values(editor.measurement_var.get()) == pytest.approx(
        [expected, 2 * expected], abs=0.0006,
    )
    assert editor._measurement_move_start is None
    assert editor.controller.points == route

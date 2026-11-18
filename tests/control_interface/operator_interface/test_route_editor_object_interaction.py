from types import SimpleNamespace

import numpy as np
import pytest

from route_editor_controller import OrthoView
from route_editor_objects import ObjectBoxController, PointerScale
from route_editor_window import RouteEditorWindow


def _scale_editor(objects, start, mouse):
    return SimpleNamespace(
        objects=objects,
        view=OrthoView(center=np.zeros(3), radius=3.0),
        _canvas_size=lambda: (800, 600),
        _mouse=mouse,
        _mouse_precise=False,
        _scale_gesture=PointerScale((400, 300), start),
        _move_mouse_start=start,
        _box_numeric="",
        _refresh_object_labels=lambda: None,
        request_redraw=lambda: None,
        _object_transform_status=lambda: None,
    )


def test_scale_started_at_box_center_grows_gently_and_shrinks_the_other_way():
    objects = ObjectBoxController([
        {"center": [0.0, 0.0, 0.0], "size": [2.0, 4.0, 6.0]},
    ])
    objects.begin("s")
    objects.constrain(0)
    editor = _scale_editor(objects, (400, 300), (400, 300))

    RouteEditorWindow._preview_object_transform(editor)
    assert objects.boxes[0]["size"] == [2.0, 4.0, 6.0]

    editor._mouse = (460, 300)
    RouteEditorWindow._preview_object_transform(editor)
    size = objects.boxes[0]["size"]
    assert size[0] == pytest.approx(2.0 * np.exp(60 / np.sqrt(2) / 250))
    assert size[0] < 3.0  # the old mapping quadrupled it at the same 60 px
    assert size[1:] == [4.0, 6.0]

    editor._mouse = (160, 300)
    RouteEditorWindow._preview_object_transform(editor)
    assert objects.boxes[0]["size"][0] < 2.0


def test_scale_can_shrink_far_below_the_start_size_in_one_gesture():
    gesture = PointerScale((400, 300), (700, 300))
    assert gesture.move((100, 300)) == pytest.approx(np.exp(-600 / 250))
    assert gesture.move((-200, 300)) < 0.1


def test_precision_changes_rate_without_jumping_the_size():
    gesture = PointerScale((400, 300), (500, 300))
    coarse = gesture.move((600, 300))
    assert gesture.move((600, 300), precise=True) == coarse
    fine = gesture.move((700, 300), precise=True)
    assert fine == pytest.approx(coarse * np.exp(10 / 250))


def test_zoom_mid_move_keeps_the_box_under_the_same_displacement():
    objects = ObjectBoxController([
        {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0]},
    ])
    objects.begin("g")
    editor = SimpleNamespace(
        objects=objects,
        view=OrthoView(center=np.zeros(3), radius=3.0),
        _canvas_size=lambda: (800, 600),
        _mouse=(500, 300),
        _move_mouse_start=(400, 300),
        request_redraw=lambda: None,
    )
    displacement = 100 / editor.view.scale(800, 600)

    RouteEditorWindow._set_zoom(editor, 2.0)

    delta = editor.view.screen_delta_to_world(
        editor._mouse[0] - editor._move_mouse_start[0],
        editor._mouse[1] - editor._move_mouse_start[1], 800, 600, None,
    )
    assert delta[0] == pytest.approx(displacement)


def test_object_g_z_confirm_cancel_undo_redo_repaint_only_pristine_cached_cloud(monkeypatch):
    from types import MethodType
    from PIL import Image
    import route_editor_window as rew
    from route_editor_controller import RouteEditorController
    from route_editor_rendering import PointCloudRenderer, cloud_raster_key

    objects = ObjectBoxController([
        {"center": [0.0, 0.0, 0.0], "size": [2.0, 4.0, 6.0]},
    ])
    editor = _scale_editor(objects, (400, 300), (400, 250))
    editor.view.front()
    editor.controller = RouteEditorController()
    editor.measurement_points = []
    editor._boxing = True
    editor._box_draw_start = None
    editor._cloud_revision = 0
    editor.map_points = np.array([[0, 0, 0, 50, 80, 120]], dtype=np.float32)
    raster = PointCloudRenderer(editor.map_points, use_cuda=False).render(editor.view, 800, 600)
    editor._raster_cache = Image.fromarray(raster)
    editor._raster_cache_key = cloud_raster_key(editor.map_points, 0, editor.view, 800, 600)
    editor._closed = False
    editor._redraw_pending = False
    editor._distance_labels_dirty = False
    editor._render_future = None
    editor._render_after = None
    editor._canvas_item = None
    editor._photo = None
    editor.callbacks = []
    editor.after = lambda delay, callback: editor.callbacks.append(callback) or "timer"
    editor._render_executor = SimpleNamespace(
        submit=lambda *args: pytest.fail("Object-only transforms must not rasterize the cloud"),
    )
    editor.canvas = SimpleNamespace(
        create_image=lambda *args, **kwargs: 1,
        itemconfigure=lambda *args, **kwargs: None,
    )
    editor.status_var = SimpleNamespace(set=lambda value: None)
    editor._repair_look_at_references = lambda: None
    monkeypatch.setattr(rew.ImageTk, "PhotoImage", lambda image: image.copy())
    for name in (
        "request_redraw", "_redraw", "_current_raster_key", "_paint_render",
        "_draw_axes", "_draw_route", "_draw_measurement", "_draw_object_boxes",
        "_finish_object_transform", "_object_escape", "_object_undo_redo",
    ):
        setattr(editor, name, MethodType(getattr(RouteEditorWindow, name), editor))
    editor.request_redraw()
    editor.callbacks.pop(0)()
    initial = np.asarray(editor._photo).copy()
    objects.begin("g")
    objects.constrain(2)
    RouteEditorWindow._preview_object_transform(editor)
    expected_height = 50 / editor.view.scale(800, 600)
    assert objects.boxes[0]["center"] == pytest.approx([0, 0, expected_height])
    editor.callbacks.pop(0)()
    assert not np.array_equal(np.asarray(editor._photo), initial)
    editor._finish_object_transform()
    editor.callbacks.pop(0)()
    confirmed = np.asarray(editor._photo).copy()
    editor._object_undo_redo(False)
    editor.callbacks.pop(0)()
    assert objects.boxes[0]["center"] == [0, 0, 0]
    np.testing.assert_array_equal(np.asarray(editor._photo), initial)
    editor._object_undo_redo(True)
    editor.callbacks.pop(0)()
    assert objects.boxes[0]["center"] == pytest.approx([0, 0, expected_height])
    np.testing.assert_array_equal(np.asarray(editor._photo), confirmed)
    objects.begin("g")
    objects.constrain(2)
    editor._mouse = (400, 150)
    RouteEditorWindow._preview_object_transform(editor)
    editor._object_escape()
    assert len(editor.callbacks) == 1
    editor.callbacks.pop(0)()
    assert objects.boxes[0]["center"] == pytest.approx([0, 0, expected_height])
    np.testing.assert_array_equal(np.asarray(editor._photo), confirmed)
    np.testing.assert_array_equal(np.asarray(editor._raster_cache), raster)

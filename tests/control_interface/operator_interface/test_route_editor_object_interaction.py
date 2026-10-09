from types import SimpleNamespace

import numpy as np

from route_editor_controller import OrthoView
from route_editor_objects import ObjectBoxController
from route_editor_window import RouteEditorWindow


def test_scale_started_at_box_center_keeps_size_until_pointer_moves():
    objects = ObjectBoxController([
        {"center": [0.0, 0.0, 0.0], "size": [2.0, 4.0, 6.0]},
    ])
    objects.begin("s")
    objects.constrain(0)
    editor = SimpleNamespace(
        objects=objects,
        view=OrthoView(center=np.zeros(3), radius=3.0),
        _canvas_size=lambda: (800, 600),
        _mouse=(400, 300),
        _move_mouse_start=(400, 300),
        _box_numeric="",
        _refresh_object_labels=lambda: None,
        request_redraw=lambda: None,
        _object_transform_status=lambda: None,
    )

    RouteEditorWindow._preview_object_transform(editor)
    assert objects.boxes[0]["size"] == [2.0, 4.0, 6.0]

    editor._mouse = (460, 300)
    RouteEditorWindow._preview_object_transform(editor)
    assert objects.boxes[0] == {
        "center": [0.0, 0.0, 0.0], "size": [8.0, 4.0, 6.0],
    }

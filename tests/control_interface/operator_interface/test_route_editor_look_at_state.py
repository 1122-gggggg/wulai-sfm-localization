from __future__ import annotations

import pytest

from route_editor_controller import RouteEditorController
from route_editor_objects import ObjectBoxController


def _points(count: int) -> list[list[float]]:
    return [[float(index), 0.0, 0.0] for index in range(count)]


def _boxes(count: int) -> list[dict]:
    return [
        {"center": [float(index), 2.0, 3.0], "size": [1.0, 1.0, 1.0]}
        for index in range(count)
    ]


def _target_box(route: RouteEditorController, objects: ObjectBoxController, leg: int) -> dict:
    return objects.boxes[objects.keys.index(route.segment_look_at[leg])]


def test_ranges_address_departing_legs_and_preserve_other_spans() -> None:
    route = RouteEditorController(_points(6), segment_look_at=[2, 2, 2, 2, 2])

    assert route.set_look_at_range(1, 4, 7)
    assert route.segment_look_at == [2, 7, 7, 7, 2]
    assert route.set_look_at_range(0, 1, 3)
    assert route.set_look_at_range(4, 5, 9)
    assert route.segment_look_at == [3, 7, 7, 7, 9]
    assert route.set_look_at_range(2, 4, None)
    assert route.segment_look_at == [3, 7, None, None, 9]


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 0), (2, 1), (0, 4), (True, 2), (0, 2.0)])
def test_invalid_range_does_not_change_links_or_history(start, end) -> None:
    route = RouteEditorController(_points(4), segment_look_at=[4, 4, 4])

    with pytest.raises(ValueError, match="range"):
        route.set_look_at_range(start, end, 8)

    assert route.segment_look_at == [4, 4, 4]
    assert not route.undo()


@pytest.mark.parametrize("key", [True, False, -1, 2.5, "2"])
def test_invalid_object_key_cannot_record_a_range(key) -> None:
    route = RouteEditorController(_points(3), segment_look_at=[1, 1])

    with pytest.raises(ValueError, match="object key"):
        route.set_look_at_range(0, 2, key)

    assert route.segment_look_at == [1, 1]
    assert not route.undo()


def test_unchanged_range_does_not_consume_undo_or_invalidate_redo() -> None:
    route = RouteEditorController(_points(3), segment_look_at=[None, None])
    assert route.set_look_at_range(0, 2, 4)
    assert not route.set_look_at_range(0, 2, 4)
    assert route.undo()
    assert route.segment_look_at == [None, None]
    assert not route.set_look_at_range(0, 2, None)
    assert route.redo()
    assert route.segment_look_at == [4, 4]
    assert route.undo()
    assert not route.undo()


@pytest.mark.parametrize("selected,expected", [(0, [6, 7, 8]), (4, [5, 6, 7])])
def test_endpoint_deletion_removes_only_its_adjacent_leg(selected, expected) -> None:
    points = _points(5)
    route = RouteEditorController(points, segment_look_at=[5, 6, 7, 8])
    route.selected = selected

    assert route.delete_selected()
    assert route.points == points[:selected] + points[selected + 1:]
    assert route.segment_look_at == expected
    assert route.undo()
    assert route.points == points
    assert route.segment_look_at == [5, 6, 7, 8]
    assert route.redo()
    assert route.segment_look_at == expected


@pytest.mark.parametrize(
    "incoming,outgoing,bridge",
    [(6, 6, 6), (6, 7, None), (6, None, None), (None, 6, None), (None, None, None)],
)
def test_interior_deletion_joins_only_matching_targets(incoming, outgoing, bridge) -> None:
    points = _points(5)
    actions = [[{"type": "hover", "seconds": index + 1}] for index in range(5)]
    links = [5, incoming, outgoing, 8]
    route = RouteEditorController(points, actions, links)
    route.selected = 2

    assert route.delete_selected()
    assert route.points == points[:2] + points[3:]
    assert route.actions == actions[:2] + actions[3:]
    assert route.segment_look_at == [5, bridge, 8]
    assert route.undo()
    assert route.points == points
    assert route.actions == actions
    assert route.segment_look_at == links
    assert route.redo()
    assert route.segment_look_at == [5, bridge, 8]


def test_last_remaining_waypoints_can_be_deleted_without_a_phantom_leg() -> None:
    route = RouteEditorController(_points(2), segment_look_at=[3])
    assert route.delete_selected()
    assert route.segment_look_at == []
    assert route.delete_selected()
    assert route.points == []
    assert route.segment_look_at == []
    assert route.undo()
    assert len(route.points) == 1
    assert route.segment_look_at == []
    assert route.undo()
    assert route.points == _points(2)
    assert route.segment_look_at == [3]


def test_appending_a_waypoint_keeps_existing_span_and_adds_unassigned_leg() -> None:
    route = RouteEditorController()
    route.set_placement_enabled(True)
    for point in _points(3):
        route.add(point)
    assert route.set_look_at_range(0, 2, 4)

    route.add([3.0, 0.0, 0.0])
    assert route.segment_look_at == [4, 4, None]
    assert route.undo()
    assert route.points == _points(3)
    assert route.segment_look_at == [4, 4]
    assert route.redo()
    assert route.points == _points(4)
    assert route.segment_look_at == [4, 4, None]


def test_range_motion_and_action_edits_share_coherent_route_history() -> None:
    route = RouteEditorController(_points(3), segment_look_at=[2, None])
    route.selected = 1
    assert route.set_look_at_range(1, 2, 5)
    assert route.begin_move()
    route.preview_move([0.0, 0.0, 3.0])
    assert route.confirm_move()
    assert route.add_action({"type": "hover", "seconds": 2})
    assert route.segment_look_at == [2, 5]
    assert route.remove_last_action()
    assert route.segment_look_at == [2, 5]
    assert route.undo()
    assert route.actions[1] == [{"type": "hover", "seconds": 2}]
    assert route.undo()
    assert route.actions[1] == []
    assert route.points[1] == [1.0, 0.0, 3.0]
    assert route.segment_look_at == [2, 5]
    assert route.undo()
    assert route.points == _points(3)
    assert route.segment_look_at == [2, 5]
    assert route.undo()
    assert route.segment_look_at == [2, None]
    assert route.redo()
    assert route.redo()
    assert route.points[1] == [1.0, 0.0, 3.0]
    assert route.segment_look_at == [2, 5]
    assert route.redo()
    assert route.actions[1] == [{"type": "hover", "seconds": 2}]


def test_cancelling_waypoint_motion_preserves_leg_targets_and_redo() -> None:
    route = RouteEditorController(_points(3), segment_look_at=[1, 2])
    assert route.set_look_at_range(0, 1, 3)
    assert route.undo()
    route.selected = 1
    assert route.begin_move()
    route.preview_move([2.0, 0.0, 1.0])
    assert route.undo()
    assert route.points == _points(3)
    assert route.segment_look_at == [1, 2]
    assert route.redo()
    assert route.segment_look_at == [3, 2]


def test_deleting_an_unreferenced_box_does_not_shift_the_target() -> None:
    boxes = _boxes(3)
    objects = ObjectBoxController(boxes)
    route = RouteEditorController(_points(2), segment_look_at=[2])
    objects.selected = 0

    assert objects.delete_selected()
    assert objects.keys == [1, 2]
    assert _target_box(route, objects, 0) == boxes[2]
    assert not route.discard_missing_look_at(objects.keys)
    assert objects.undo()
    assert objects.keys == [0, 1, 2]
    assert _target_box(route, objects, 0) == boxes[2]
    assert objects.redo()
    assert objects.keys == [1, 2]
    assert _target_box(route, objects, 0) == boxes[2]


def test_undone_add_never_reuses_a_missing_target_key() -> None:
    objects = ObjectBoxController(_boxes(1))
    first_index = objects.add_bounds([4, 0, 0], [6, 2, 2])
    missing_key = objects.keys[first_index]
    route = RouteEditorController(_points(3), segment_look_at=[0, None])
    assert route.set_look_at_range(1, 2, missing_key)
    assert objects.undo()
    assert objects.redo()
    assert objects.keys[first_index] == missing_key
    assert _target_box(route, objects, 1)["center"] == [5.0, 1.0, 1.0]
    assert objects.undo()

    new_index = objects.add_bounds([8, 0, 0], [10, 2, 2])
    assert objects.keys[new_index] > missing_key
    assert not objects.redo()
    assert route.discard_missing_look_at(iter(objects.keys))
    assert route.segment_look_at == [0, None]
    assert not route.discard_missing_look_at(objects.keys)
    assert route.undo()
    assert route.segment_look_at == [0, None]
    assert not route.undo()
    assert route.redo()
    assert route.segment_look_at == [0, None]


def test_independent_history_repair_clears_restored_missing_links_without_retargeting() -> None:
    objects = ObjectBoxController(_boxes(3))
    route = RouteEditorController(_points(3), segment_look_at=[1, 2])
    assert route.set_look_at_range(0, 1, 0)
    objects.selected = 1
    assert objects.delete_selected()
    assert objects.keys == [0, 2]
    assert route.undo()
    assert route.segment_look_at == [1, 2]
    assert route.discard_missing_look_at(objects.keys)
    assert route.segment_look_at == [None, 2]
    assert _target_box(route, objects, 1) == _boxes(3)[2]
    assert objects.undo()
    assert not route.discard_missing_look_at(objects.keys)
    assert route.segment_look_at == [None, 2]
    assert route.redo()
    assert route.segment_look_at == [0, 2]
    assert route.undo()
    assert route.segment_look_at == [None, 2]
    assert not route.undo()


@pytest.mark.parametrize("operation", ["g", "s"])
def test_object_transforms_keep_links_on_same_identity_across_history(operation) -> None:
    objects = ObjectBoxController(_boxes(3))
    objects.selected = 0
    assert objects.delete_selected()
    route = RouteEditorController(_points(2), segment_look_at=[2])
    objects.selected = 1
    original = _boxes(3)[2]
    assert objects.begin(operation)
    objects.preview(delta=[5, 0, 0], factor=2.0)
    transformed = (
        {"center": [7.0, 2.0, 3.0], "size": [1.0, 1.0, 1.0]}
        if operation == "g" else
        {"center": [2.0, 2.0, 3.0], "size": [2.0, 2.0, 2.0]}
    )
    assert _target_box(route, objects, 0) == transformed
    assert objects.confirm()
    assert objects.undo()
    assert objects.keys == [1, 2]
    assert _target_box(route, objects, 0) == original
    assert objects.redo()
    assert _target_box(route, objects, 0) == transformed
    assert objects.begin(operation)
    objects.preview(delta=[9, 0, 0], factor=4.0)
    assert objects.cancel()
    assert objects.keys == [1, 2]
    assert _target_box(route, objects, 0) == transformed

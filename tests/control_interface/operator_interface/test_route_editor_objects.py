from __future__ import annotations

import copy

import numpy as np
import pytest

from route_editor_objects import ObjectBoxController, box_corners


BOX = {"center": [1.0, 2.0, 3.0], "size": [2.0, 4.0, 6.0]}


def test_corners_have_the_renderers_product_order() -> None:
    np.testing.assert_array_equal(box_corners(BOX), [
        [0.0, 0.0, 0.0], [0.0, 0.0, 6.0],
        [0.0, 4.0, 0.0], [0.0, 4.0, 6.0],
        [2.0, 0.0, 0.0], [2.0, 0.0, 6.0],
        [2.0, 4.0, 0.0], [2.0, 4.0, 6.0],
    ])


def test_empty_state_and_no_selection_are_noops() -> None:
    controller = ObjectBoxController()
    assert controller.boxes == []
    assert controller.selected is None
    assert not controller.begin("g")
    controller.preview(delta=[1.0, 2.0, 3.0])
    controller.constrain(0)
    assert controller.axis is None
    assert not controller.confirm()
    assert not controller.cancel()
    assert not controller.delete_selected()
    assert not controller.undo()
    assert not controller.redo()

    controller.add_bounds([0, 0, 0], [1, 1, 1])
    for selected in (None, -1, 1):
        controller.selected = selected
        assert not controller.begin("s")
        assert not controller.delete_selected()
    with pytest.raises(ValueError):
        controller.begin("rotate")


def test_bounds_add_delete_undo_redo_and_selection_clamping() -> None:
    controller = ObjectBoxController()
    assert controller.add_bounds([-1, -2, -3], [3, 6, 9]) == 0
    first = {"center": [1.0, 2.0, 3.0], "size": [4.0, 8.0, 12.0]}
    assert controller.boxes == [first]
    assert controller.add_bounds([0, 0, 0], [1, 2, 3]) == 1
    both = copy.deepcopy(controller.boxes)
    assert controller.selected == 1
    assert controller.delete_selected()
    assert controller.boxes == [first]
    assert controller.selected == 0
    assert controller.undo()
    assert controller.boxes == both
    assert controller.redo()
    assert controller.boxes == [first]
    assert controller.delete_selected()
    assert controller.boxes == []
    assert controller.selected is None
    assert controller.undo()
    assert controller.boxes == [first]
    assert controller.selected == 0


def test_move_previews_do_not_accumulate_and_only_confirm_records_history() -> None:
    controller = ObjectBoxController([BOX])
    assert controller.begin("g")
    assert controller.transforming
    assert controller.operation == "g"
    controller.preview(delta=[1, 2, 3])
    controller.preview(delta=[4, 5, 6])
    assert controller.boxes == [{"center": [5.0, 7.0, 9.0], "size": BOX["size"]}]
    assert controller.confirm()
    assert not controller.transforming
    assert controller.operation is None
    assert controller.axis is None
    assert controller.undo()
    assert controller.boxes == [BOX]
    assert not controller.undo()
    assert controller.redo()
    assert controller.boxes[0]["center"] == [5.0, 7.0, 9.0]
    assert not controller.redo()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_move_axis_constrains_and_toggles_back_to_free_motion(axis) -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("g")
    controller.constrain(axis)
    controller.preview(delta=[10, 20, 30])
    expected = BOX["center"].copy()
    expected[axis] += [10, 20, 30][axis]
    assert controller.boxes[0]["center"] == expected
    assert controller.boxes[0]["size"] == BOX["size"]
    controller.constrain(axis)
    assert controller.axis is None
    controller.preview(delta=[10, 20, 30])
    assert controller.boxes[0]["center"] == [11.0, 22.0, 33.0]
    controller.constrain((axis + 1) % 3)
    controller.constrain(None)
    assert controller.axis is None


def test_scale_is_centered_and_not_cumulative() -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("s")
    controller.preview(factor=3.0)
    controller.preview(factor=0.5)
    assert controller.boxes == [{"center": BOX["center"], "size": [1.0, 2.0, 3.0]}]
    assert controller.confirm()
    assert controller.undo()
    assert controller.boxes == [BOX]
    assert controller.redo()
    assert controller.boxes[0]["size"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_scale_axis_changes_only_one_original_dimension(axis) -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("s")
    controller.preview(factor=4.0)
    controller.constrain(axis)
    controller.preview(factor=2.0)
    expected = BOX["size"].copy()
    expected[axis] *= 2.0
    assert controller.boxes == [{"center": BOX["center"], "size": expected}]
    controller.constrain(axis)
    controller.preview(factor=2.0)
    assert controller.boxes[0]["size"] == [4.0, 8.0, 12.0]


def test_repeated_begin_preserves_original_operation_axis_and_snapshot() -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("g")
    controller.constrain(0)
    controller.preview(delta=[5, 6, 7])
    assert not controller.begin("g")
    assert not controller.begin("s")
    assert controller.operation == "g"
    assert controller.axis == 0
    controller.preview(delta=[2, 9, 9])
    assert controller.boxes[0]["center"] == [3.0, 2.0, 3.0]
    assert controller.cancel()
    assert controller.boxes == [BOX]
    assert not controller.undo()


def test_transform_target_is_fixed_even_if_selection_changes() -> None:
    controller = ObjectBoxController([BOX, BOX])
    controller.begin("g")
    controller.selected = 1
    controller.preview(delta=[10, 0, 0])
    assert controller.boxes[0]["center"] == [11.0, 2.0, 3.0]
    assert controller.boxes[1] == BOX
    controller.cancel()
    assert controller.boxes == [BOX, BOX]


@pytest.mark.parametrize("operation", ["g", "s"])
def test_cancel_and_undo_during_transform_restore_original_without_using_history(operation) -> None:
    controller = ObjectBoxController()
    controller.add_bounds([0, 0, 0], [2, 4, 6])
    original = copy.deepcopy(controller.boxes)
    controller.begin(operation)
    controller.preview(delta=[4, 5, 6], factor=2.0)
    assert controller.cancel()
    assert controller.boxes == original
    assert not controller.transforming
    assert controller.operation is None
    assert controller.axis is None
    controller.begin(operation)
    controller.constrain(2)
    controller.preview(delta=[4, 5, 6], factor=2.0)
    assert controller.undo()
    assert controller.boxes == original
    assert not controller.transforming
    assert controller.axis is None
    assert controller.undo()
    assert controller.boxes == []
    assert controller.redo()
    assert controller.boxes == original


def test_no_change_confirm_and_cancel_keep_redo_but_new_edit_clears_it() -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("g")
    controller.preview(delta=[1, 0, 0])
    controller.confirm()
    controller.undo()
    controller.begin("s")
    controller.preview(factor=1.0)
    assert not controller.confirm()
    controller.begin("g")
    controller.preview(delta=[0, 1, 0])
    controller.cancel()
    assert controller.redo()
    assert controller.boxes[0]["center"] == [2.0, 2.0, 3.0]
    controller.undo()
    controller.begin("g")
    controller.preview(delta=[0, 0, 1])
    assert controller.confirm()
    assert not controller.redo()
    controller.undo()
    controller.add_bounds([0, 0, 0], [1, 1, 1])
    assert not controller.redo()


def test_structural_edits_and_redo_do_not_interrupt_a_preview() -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("g")
    controller.preview(delta=[1, 0, 0])
    preview = copy.deepcopy(controller.boxes)
    with pytest.raises(ValueError):
        controller.add_bounds([0, 0, 0], [1, 1, 1])
    assert not controller.delete_selected()
    assert not controller.redo()
    assert controller.transforming
    assert controller.boxes == preview
    assert controller.cancel()
    assert controller.boxes == [BOX]


@pytest.mark.parametrize("lo,hi", [
    ([0, 0, 0], [0, 1, 1]),
    ([0, 0, 0], [1, -1, 1]),
    ([float("nan"), 0, 0], [1, 1, 1]),
    ([0, 0, 0], [1, float("inf"), 1]),
    ([float("-inf"), 0, 0], [1, 1, 1]),
    ([-1e308, 0, 0], [1e308, 1, 1]),
    ([0, 0], [1, 1, 1]),
    ([[0, 0, 0]], [1, 1, 1]),
    (None, [1, 1, 1]),
])
def test_invalid_bounds_do_not_mutate_boxes_selection_or_history(lo, hi) -> None:
    controller = ObjectBoxController([BOX])
    controller.add_bounds([0, 0, 0], [1, 1, 1])
    controller.undo()
    before = copy.deepcopy(controller.boxes)
    with pytest.raises(ValueError):
        controller.add_bounds(lo, hi)
    assert controller.boxes == before
    assert controller.selected == 0
    assert controller.redo()
    assert controller.boxes == [
        BOX, {"center": [0.5, 0.5, 0.5], "size": [1.0, 1.0, 1.0]},
    ]


@pytest.mark.parametrize("box", [
    {},
    {"center": [0, 0, 0], "size": [0, 1, 1]},
    {"center": [0, 0, 0], "size": [1, -1, 1]},
    {"center": [0, float("nan"), 0], "size": [1, 1, 1]},
    {"center": [0, 0, 0], "size": [1, 1, float("inf")]},
    {"center": [0, 0], "size": [1, 1, 1]},
])
def test_constructor_and_corners_reject_invalid_boxes(box) -> None:
    with pytest.raises(ValueError):
        ObjectBoxController([box])
    with pytest.raises(ValueError):
        box_corners(box)


@pytest.mark.parametrize("factor", [0.0, -1.0, float("nan"), float("inf"), float("-inf"), 1e308, [], "invalid"])
def test_invalid_scale_retains_last_valid_preview_and_can_still_confirm(factor) -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("s")
    controller.preview(factor=2.0)
    before = copy.deepcopy(controller.boxes)
    with pytest.raises(ValueError):
        controller.preview(factor=factor)
    assert controller.boxes == before
    assert controller.transforming
    assert controller.confirm()
    assert controller.undo()
    assert controller.boxes == [BOX]


def test_scale_underflow_rejected_without_losing_preview() -> None:
    original = {"center": [0, 0, 0], "size": [1e-300, 1, 1]}
    controller = ObjectBoxController([original])
    controller.begin("s")
    controller.preview(factor=2.0)
    before = copy.deepcopy(controller.boxes)
    with pytest.raises(ValueError):
        controller.preview(factor=1e-300)
    assert controller.boxes == before
    assert controller.cancel()
    assert controller.boxes == [original]


@pytest.mark.parametrize("delta", [[float("nan"), 0, 0], [0, float("inf"), 0], [1, 2], [[1, 2, 3]], "invalid"])
def test_invalid_displacement_retains_last_valid_preview(delta) -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("g")
    controller.preview(delta=[2, 3, 4])
    before = copy.deepcopy(controller.boxes)
    with pytest.raises(ValueError):
        controller.preview(delta=delta)
    assert controller.boxes == before
    assert controller.cancel()
    assert controller.boxes == [BOX]


def test_move_overflow_rejected_before_mutation() -> None:
    original = {"center": [1e308, 0, 0], "size": [1, 1, 1]}
    controller = ObjectBoxController([original])
    controller.begin("g")
    with pytest.raises(ValueError):
        controller.preview(delta=[1e308, 0, 0])
    assert controller.boxes == [original]
    assert not controller.confirm()


@pytest.mark.parametrize("axis", [-1, 3, "x", 1.5])
def test_invalid_axis_leaves_constraint_and_preview_unchanged(axis) -> None:
    controller = ObjectBoxController([BOX])
    controller.begin("g")
    controller.constrain(1)
    with pytest.raises(ValueError):
        controller.constrain(axis)
    assert controller.axis == 1
    assert controller.boxes == [BOX]

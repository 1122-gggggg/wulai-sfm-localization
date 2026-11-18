from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from real_path_follow_controller import MapFrame
from route_editor_model import RouteDocument, editor_to_glomap, glomap_to_editor
from route_editor_wires import WireController, sample_wire, wire_annotations


def _wire():
    return {"a": [0.0, 0.0, 3.0], "b": [4.0, 2.0, 5.0], "sag": 1.5, "up": [0.0, 0.0, 1.0]}


def test_wire_keeps_attachments_and_midpoint_sag_with_unequal_heights():
    wire = _wire()
    curve = sample_wire(wire)
    np.testing.assert_array_equal(curve[[0, -1]], [wire["a"], wire["b"]])
    np.testing.assert_allclose(curve[32], [2.0, 1.0, 2.5])
    wire["sag"] = 0.0
    np.testing.assert_allclose(sample_wire(wire)[32], [2.0, 1.0, 4.0])


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1", 10 ** 400])
def test_invalid_sag_cannot_replace_last_valid_preview(value):
    controller = WireController([_wire()])
    controller.selected = 0
    controller.begin_sag()
    controller.preview_sag(2.0)
    with pytest.raises(ValueError):
        controller.preview_sag(value)
    assert controller.wires[0]["sag"] == 2.0
    controller.cancel()
    assert controller.wires[0]["sag"] == 1.5


def test_wire_transactions_restore_geometry_without_affecting_other_spans():
    controller = WireController([_wire()])
    controller.pick_endpoint([1, 1, 3])
    with pytest.raises(ValueError):
        controller.pick_endpoint([1, 1, 3])
    assert controller.pending == [1, 1, 3]
    assert controller.pick_endpoint([5, 1, 3])
    original = copy.deepcopy(controller.wires)
    assert controller.begin_sag()
    controller.preview_sag(1.0)
    controller.preview_sag(2.0)
    assert controller.confirm()
    assert controller.wires[0] == original[0]
    assert controller.undo()
    assert controller.wires == original
    assert controller.redo()
    assert controller.wires[1]["sag"] == 2.0
    assert controller.delete_selected()
    assert controller.wires == [original[0]]
    assert controller.undo()
    assert controller.wires[1]["sag"] == 2.0
    assert controller.begin_sag()
    controller.preview_sag(8)
    assert controller.cancel()
    assert controller.wires[1]["sag"] == 2.0


@pytest.mark.parametrize("field,value", [
    ("a", [float("nan"), 0, 1]), ("b", [0, 0, 3]),
    ("up", [0, 0, 2]), ("sag", -0.1),
])
def test_invalid_annotation_is_rejected_on_save_and_load(tmp_path, field, value):
    wire = _wire()
    wire[field] = value
    document = RouteDocument("yard", "frame", points=[[0, 0, 0], [1, 0, 0]], wires=[wire])
    with pytest.raises(ValueError):
        document.save(tmp_path / "invalid.json")
    raw = RouteDocument("yard", "frame", points=[[0, 0, 0], [1, 0, 0]]).payload()
    raw["wires"] = [wire]
    target = tmp_path / "invalid.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        RouteDocument.load(target, site_id="yard", coordinate_frame_id="frame")


def test_wire_roundtrip_and_alignment_change_preserve_physical_curve(tmp_path):
    document = RouteDocument("yard", "frame", points=[[0, 0, 0], [1, 0, 0]], wires=[_wire()])
    target = document.save(tmp_path / "wire.json")
    reopened = RouteDocument.load(target, site_id="yard", coordinate_frame_id="frame")
    assert reopened.wires == document.wires
    frame = MapFrame.from_gravity(np.array([0.1, 0.96, 0.25]))
    measured = RouteDocument.load(
        target, site_id="yard", coordinate_frame_id="frame", map_frame=frame,
        align_source="measured",
    )
    expected = glomap_to_editor(editor_to_glomap(sample_wire(document.wires[0])), frame)
    np.testing.assert_allclose(sample_wire(measured.wires[0]), expected, atol=1e-12)
    target2 = measured.save(tmp_path / "measured.json")
    reloaded = RouteDocument.load(
        target2, site_id="yard", coordinate_frame_id="frame", map_frame=frame,
        align_source="measured",
    )
    np.testing.assert_allclose(sample_wire(reloaded.wires[0]), expected, atol=1e-12)
    assert wire_annotations(reloaded.wires) == reloaded.wires


@pytest.mark.parametrize("endpoint", ["a", "b"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_endpoint_move_is_axis_locked_relative_to_start_and_independent(endpoint, axis):
    controller = WireController([_wire(), _wire()])
    controller.selected = 0
    controller.selected_endpoint = endpoint
    before = copy.deepcopy(controller.wires)
    assert controller.begin_move()
    controller.constrain(axis)
    controller.preview_move([0.1, 0.2, 0.3])
    controller.preview_move([0.4, 0.5, 0.6])
    expected = copy.deepcopy(before)
    expected[0][endpoint][axis] += [0.4, 0.5, 0.6][axis]
    assert controller.wires == expected
    assert controller.confirm()
    np.testing.assert_array_equal(
        sample_wire(controller.wires[0])[[0, -1]],
        [expected[0]["a"], expected[0]["b"]],
    )
    assert controller.undo()
    assert controller.wires == before
    assert controller.redo()
    assert controller.wires == expected


def test_endpoint_cancel_and_invalid_move_preserve_geometry_and_redo():
    controller = WireController([_wire()])
    controller.selected = 0
    controller.selected_endpoint = "a"
    before = copy.deepcopy(controller.wires)
    assert controller.begin_move()
    controller.preview_move([0.5, -0.2, 0.1])
    valid = copy.deepcopy(controller.wires)
    for invalid in ([4, 2, 2], [float("nan"), 0, 0], [0, float("inf"), 0]):
        with pytest.raises(ValueError):
            controller.preview_move(invalid)
        assert controller.wires == valid
    assert not controller.begin_sag()
    assert not controller.delete_selected()
    assert controller.cancel()
    assert controller.wires == before
    assert not controller.undo()
    assert controller.begin_move()
    controller.preview_move([0.5, -0.2, 0.1])
    assert controller.confirm()
    assert controller.undo()
    controller.selected_endpoint = "b"
    assert controller.begin_move()
    controller.preview_move([0.1, 0, 0])
    assert controller.cancel()
    assert controller.redo()
    assert controller.wires == valid


def test_endpoint_axis_toggle_and_move_roundtrip(tmp_path):
    controller = WireController([_wire()])
    controller.selected = 0
    assert not controller.begin_move()
    controller.selected_endpoint = "b"
    assert controller.begin_move()
    controller.constrain(2)
    controller.preview_move([1, 2, 0.25])
    assert controller.wires[0]["b"] == [4, 2, 5.25]
    controller.constrain(2)
    controller.preview_move([0.1, -0.2, 0.3])
    assert controller.wires[0]["b"] == [4.1, 1.8, 5.3]
    assert controller.confirm()
    document = RouteDocument(
        "yard", "frame", points=[[0, 0, 0], [1, 0, 0]], wires=controller.wires,
    )
    saved = document.save(tmp_path / "moved.json")
    reloaded = RouteDocument.load(saved, site_id="yard", coordinate_frame_id="frame")
    assert reloaded.wires == controller.wires
    assert reloaded.wires[0]["a"] == [0, 0, 3]
    assert reloaded.wires[0]["sag"] == 1.5

"""Offline contract tests for authored-leg yaw targets and route identity."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

FLIGHT_CONTROL_ROOT = Path(__file__).resolve().parents[3] / "定位演算法" / "flight_control"
sys.path.insert(0, str(FLIGHT_CONTROL_ROOT))

import real_path_follow_controller as rpf
from route_domain import (
    RouteDocument,
    capture_mission_route_snapshot,
    parse_segment_look_at,
    validate_flight_route_fields,
)


def _route(**fields):
    data = {
        "schema": "sfm-flight-route/v1",
        "site_id": "field-a",
        "coordinate_frame_id": "glomap-a",
        "frame": "glomap",
        "units": "map",
        "purpose": "flight",
        "closed": False,
        "waypoints": [[0, 0, 0], [2, 0, 0], [2, 0, 3]],
        "object_boxes": [
            {"center": [4, 5, 6], "size": [1, 2, 3]},
            {"center": [-3, 7, 8], "size": [2, 3, 4]},
        ],
    }
    data.update(fields)
    return data


def _validate(data):
    validate_flight_route_fields(
        data, expected_site_id="field-a", expected_coordinate_frame_id="glomap-a",
    )


def _snapshot(path, **kwargs):
    return capture_mission_route_snapshot(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_site_id="field-a",
        expected_coordinate_frame_id="glomap-a",
        **kwargs,
    )


def test_zero_based_table_applies_to_exact_authored_legs():
    route = RouteDocument.from_data(_route(segment_look_at=[1, 0]))
    assert route.segment_look_at_points == ((-3.0, -8.0, 7.0), (4.0, -6.0, 5.0))
    assert len(route.segment_look_at_points) == len(route.controller_points) - 1
    partial = RouteDocument.from_data(_route(segment_look_at=[None, 1]))
    assert partial.segment_look_at_points == (None, (-3.0, -8.0, 7.0))


@pytest.mark.parametrize(
    "raw",
    [[], [0], [0, None, 1], (0, 1), {}, "0,1", True,
     [True, None], [False, None], [0.0, None], [-1, None], [2, None], ["0", None]],
)
def test_invalid_segment_table_rejected_before_conversion_and_during_load(raw):
    data = _route(segment_look_at=raw)
    with pytest.raises(ValueError, match="segment_look_at"):
        _validate(data)
    with pytest.raises(ValueError, match="segment_look_at"):
        RouteDocument.from_data(data)


def test_parser_preserves_nulls_and_repeated_box_references():
    assert parse_segment_look_at(
        [0, None, 0], waypoint_count=4, box_count=1,
    ) == [0, None, 0]
    with pytest.raises(ValueError, match="segment_look_at"):
        parse_segment_look_at([0], waypoint_count=2, box_count=0)


INVALID_BOXES = [
    None,
    [],
    {"center": [1, 2, 3]},
    {"center": [1, 2, 3], "size": [1, 1, 1], "rotation": [0, 0, 0]},
    {"center": [1, 2], "size": [1, 1, 1]},
    {"center": "123", "size": [1, 1, 1]},
    {"center": [True, 2, 3], "size": [1, 1, 1]},
    {"center": ["1", 2, 3], "size": [1, 1, 1]},
    {"center": [float("nan"), 2, 3], "size": [1, 1, 1]},
    {"center": [1, float("inf"), 3], "size": [1, 1, 1]},
    {"center": [10**400, 2, 3], "size": [1, 1, 1]},
    {"center": [1, 2, 3], "size": [1, 1]},
    {"center": [1, 2, 3], "size": [True, 1, 1]},
    {"center": [1, 2, 3], "size": ["1", 1, 1]},
    {"center": [1, 2, 3], "size": [0, 1, 1]},
    {"center": [1, 2, 3], "size": [-1, 1, 1]},
    {"center": [1, 2, 3], "size": [float("inf"), 1, 1]},
    {"center": [1, 2, 3], "size": [1, float("nan"), 1]},
]


@pytest.mark.parametrize("box", INVALID_BOXES)
def test_referenced_box_must_be_well_formed_before_import(box):
    data = _route(object_boxes=[box], segment_look_at=[0, None])
    with pytest.raises(ValueError, match="object_boxes"):
        _validate(data)
    with pytest.raises(ValueError, match="object_boxes"):
        RouteDocument.from_data(data)


@pytest.mark.parametrize("boxes", [None, {}, "boxes", []])
def test_reference_requires_an_existing_object_box(boxes):
    data = _route(object_boxes=boxes, segment_look_at=[0, None])
    with pytest.raises(ValueError):
        _validate(data)
    with pytest.raises(ValueError):
        RouteDocument.from_data(data)


@pytest.mark.parametrize("table", [None, [None, None]])
def test_no_targets_preserve_annotation_only_route_behavior(table):
    data = _route(object_boxes=[{"old_annotation": "not a flight target"}], segment_look_at=table)
    _validate(data)
    route = RouteDocument.from_data(data)
    assert route.segment_look_at_points == (None if table is None else (None, None))
    assert route.controller_points == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 3.0))
    data.pop("segment_look_at")
    assert RouteDocument.from_data(data).segment_look_at_points is None


def test_only_referenced_annotations_are_validated():
    data = _route(segment_look_at=[None, 1])
    data["object_boxes"][0] = {"old_annotation": "unused"}
    _validate(data)
    assert RouteDocument.from_data(data).segment_look_at_points == (None, (-3.0, -8.0, 7.0))


@pytest.mark.parametrize("waypoint_frame", ["aligned", "glomap"])
@pytest.mark.parametrize("alignment", ["legacy", "measured"])
def test_tilted_site_target_uses_box_alignment_not_waypoint_frame(waypoint_frame, alignment):
    frame = rpf.MapFrame.from_gravity([0.0, 0.9238, 0.3828])
    data = _route(frame=waypoint_frame, align_source=alignment, segment_look_at=[1, None])
    route = RouteDocument.from_data(data, map_frame=frame)
    expected = (
        frame.east * -3 + frame.north * 7 + frame.up * 8
        if alignment == "measured" else np.array([-3.0, -8.0, 7.0])
    )
    np.testing.assert_allclose(route.segment_look_at_points[0], expected)
    assert route.segment_look_at_points[1] is None
    if waypoint_frame == "glomap":
        np.testing.assert_allclose(route.controller_points, data["waypoints"])


def test_missing_alignment_is_legacy_only_when_site_is_legacy():
    data = _route(segment_look_at=[0, None])
    assert RouteDocument.from_data(data).segment_look_at_points[0] == (4.0, -6.0, 5.0)
    measured = rpf.MapFrame.from_gravity([0.0, 0.9238, 0.3828])
    with pytest.raises(ValueError, match="no align_source"):
        RouteDocument.from_data(data, map_frame=measured)
    data["segment_look_at"] = [None, None]
    assert RouteDocument.from_data(data, map_frame=measured).segment_look_at_points == (None, None)


def test_measured_box_alignment_fails_closed_at_legacy_site_even_for_glomap_waypoints():
    data = _route(align_source="measured", segment_look_at=[0, None])
    with pytest.raises(ValueError, match="no measured gravity alignment"):
        RouteDocument.from_data(data)
    # The pre-import check intentionally does not need site coordinate conversion.
    _validate(data)


def test_snapshot_and_both_config_inputs_keep_same_resolved_targets(tmp_path):
    measured = rpf.MapFrame.from_gravity([0.0, 0.9238, 0.3828])
    path = tmp_path / "route.json"
    path.write_text(json.dumps(_route(align_source="measured", segment_look_at=[1, None])), encoding="utf-8")
    snapshot = _snapshot(path, map_frame=measured)
    expected = measured.east * -3 + measured.north * 7 + measured.up * 8
    path_config = rpf.config_for_route(path, rpf.ControlConfig(map_frame=measured))
    snapshot_config = rpf.config_for_route(snapshot, rpf.ControlConfig(map_frame=measured))
    for targets in (snapshot.segment_look_at_points, path_config.segment_look_at_points,
                    snapshot_config.segment_look_at_points):
        np.testing.assert_allclose(targets[0], expected)
        assert targets[1] is None
    snapshot.verify_file_unchanged()
    changed = _route(align_source="measured", segment_look_at=[0, None])
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after selection"):
        snapshot.verify_file_unchanged()
    # Config from a snapshot must still use its approved bytes, not the changed file.
    np.testing.assert_allclose(rpf.config_for_route(snapshot).segment_look_at_points[0], expected)
    with pytest.raises(AttributeError):
        snapshot.segment_look_at_points = None

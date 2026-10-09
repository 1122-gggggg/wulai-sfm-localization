"""Behavioral tests for yaw-target links through route and preview storage."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from route_domain import RouteDocument as FlightRouteDocument
from route_editor_model import RouteDocument, editor_to_glomap


SITE = "yard"
FRAME_ID = "yard-map-v1"


def _document(**fields):
    return RouteDocument(
        site_id=SITE,
        coordinate_frame_id=FRAME_ID,
        points=[[0, 0, 0], [1, 2, 3], [3, 2, 1]],
        object_boxes=[
            {"center": [2, -3, 4], "size": [1.5, 2, 3]},
            {"center": [-5, 7, 2], "size": [2, 3, 1]},
        ],
        **fields,
    )


def _load(path, **kwargs):
    return RouteDocument.load(
        path, site_id=SITE, coordinate_frame_id=FRAME_ID, **kwargs,
    )


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _tilted_frame():
    angle = math.radians(22.51)
    return SimpleNamespace(
        east=np.array([1.0, 0.0, 0.0]),
        north=np.array([0.0, math.sin(angle), math.cos(angle)]),
        up=np.array([0.0, -math.cos(angle), math.sin(angle)]),
        source="measured",
    )


def test_save_reopen_and_change_preserve_exact_leg_and_box_indexes(tmp_path):
    document = _document(segment_look_at=[1, None])
    path = document.save(tmp_path / "route.json")
    reopened = _load(path)
    assert reopened.segment_look_at == [1, None]
    assert reopened.payload()["segment_look_at"] == [1, None]
    route = FlightRouteDocument.from_path(path)
    assert route.segment_look_at_points == ((-5.0, -2.0, 7.0), None)
    reopened.segment_look_at = [None, 0]
    reopened.save(path)
    assert _load(path).segment_look_at == [None, 0]
    assert FlightRouteDocument.from_path(path).segment_look_at_points == (None, (2.0, -4.0, -3.0))


@pytest.mark.parametrize(
    "table",
    [[], [0], [0, None, 1], "0,1", {}, (0, 1),
     [True, None], [False, None], [0.0, None], [-1, None], [2, None], ["0", None]],
)
def test_payload_rejects_wrong_length_type_and_box_references(table):
    document = _document(segment_look_at=table)
    with pytest.raises(ValueError, match="segment_look_at"):
        document.payload()
    with pytest.raises(ValueError, match="segment_look_at"):
        document.payload(preview_only=True)


@pytest.mark.parametrize("table", [[], [0], [0, None, 1], "0,1", {}, [True, None], [2, None]])
def test_load_rejects_invalid_file_table(tmp_path, table):
    payload = _document().payload()
    payload["segment_look_at"] = table
    path = _write(tmp_path / "invalid.json", payload)
    with pytest.raises(ValueError, match="segment_look_at"):
        _load(path)


def test_saved_reference_cannot_silently_survive_missing_box(tmp_path):
    document = _document(segment_look_at=[1, None])
    document.object_boxes.pop()
    with pytest.raises(ValueError, match="segment_look_at"):
        document.save(tmp_path / "missing.json")
    payload = _document(segment_look_at=[1, None]).payload()
    payload["object_boxes"].pop()
    with pytest.raises(ValueError, match="segment_look_at"):
        _load(_write(tmp_path / "invalid.json", payload))


@pytest.mark.parametrize("waypoint_frame", ["aligned", "glomap"])
def test_basis_conversion_keeps_indexes_and_physical_target(tmp_path, waypoint_frame):
    document = _document(segment_look_at=[1, 0])
    original_targets = editor_to_glomap([box["center"] for box in document.object_boxes])
    payload = document.payload()
    payload["frame"] = waypoint_frame
    if waypoint_frame == "glomap":
        payload["waypoints"] = editor_to_glomap(payload["waypoints"]).tolist()
    path = _write(tmp_path / "legacy.json", payload)
    measured = _tilted_frame()
    reopened = _load(path, map_frame=measured, align_source="measured")
    assert reopened.segment_look_at == [1, 0]
    assert reopened.object_boxes[1]["size"] != document.object_boxes[1]["size"]
    physical_centers = editor_to_glomap(
        [box["center"] for box in reopened.object_boxes], measured,
    )
    np.testing.assert_allclose(physical_centers, original_targets, atol=1e-12)
    reopened.save(path)
    route = FlightRouteDocument.from_path(path, map_frame=measured)
    np.testing.assert_allclose(route.segment_look_at_points[0], original_targets[1], atol=1e-12)
    np.testing.assert_allclose(route.segment_look_at_points[1], original_targets[0], atol=1e-12)
    assert _load(path, map_frame=measured, align_source="measured").segment_look_at == [1, 0]


def test_absent_old_route_links_do_not_invent_targets(tmp_path):
    document = _document()
    payload = document.payload()
    assert "segment_look_at" not in payload
    reopened = _load(_write(tmp_path / "old.json", payload))
    assert reopened.segment_look_at is None
    assert "segment_look_at" not in reopened.payload()
    payload["segment_look_at"] = None
    reopened = _load(_write(tmp_path / "null.json", payload))
    assert reopened.segment_look_at is None
    assert FlightRouteDocument.from_data(reopened.payload()).segment_look_at_points is None


def test_explicit_empty_overrides_keep_per_leg_nulls_through_storage(tmp_path):
    document = _document(segment_look_at=[None, None])
    path = document.save(tmp_path / "no_overrides.json")
    assert _load(path).segment_look_at == [None, None]
    assert FlightRouteDocument.from_path(path).segment_look_at_points == (None, None)


def test_preview_save_preserves_linked_boxes_but_is_not_flight_importable(tmp_path):
    document = _document(segment_look_at=[None, 1])
    path = document.save(tmp_path / "preview.json", preview_only=True)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["segment_look_at"] == [None, 1]
    assert saved["object_boxes"] == document.object_boxes
    assert saved["site_id"] is None
    assert saved["coordinate_frame_id"] is None
    assert saved["purpose"] == "preview_only"
    with pytest.raises(ValueError, match="schema"):
        FlightRouteDocument.from_path(
            path,
            require_flight_contract=True,
            expected_site_id=SITE,
            expected_coordinate_frame_id=FRAME_ID,
        )

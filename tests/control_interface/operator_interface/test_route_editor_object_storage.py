from __future__ import annotations

import json
import math
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from route_editor_model import RouteDocument, editor_to_glomap, glomap_to_editor


SITE = "yard"
FRAME_ID = "yard-map-v1"
BOX = {"center": [2.0, -3.0, 4.0], "size": [1.5, 2.0, 3.0]}


def _document(**kwargs) -> RouteDocument:
    return RouteDocument(
        site_id=SITE,
        coordinate_frame_id=FRAME_ID,
        points=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        **kwargs,
    )


def _write_route(tmp_path, **fields):
    raw = _document().payload()
    raw.update(fields)
    path = tmp_path / "route.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _load(path, **kwargs) -> RouteDocument:
    return RouteDocument.load(
        path, site_id=SITE, coordinate_frame_id=FRAME_ID, **kwargs,
    )


def _tilted_frame():
    angle = math.radians(22.51)
    return SimpleNamespace(
        east=np.array([1.0, 0.0, 0.0]),
        north=np.array([0.0, math.sin(angle), math.cos(angle)]),
        up=np.array([0.0, -math.cos(angle), math.sin(angle)]),
    )


def _corners(box):
    return (
        np.asarray(box["center"], dtype=float)
        + np.asarray(list(product((-1, 1), repeat=3)))
        * np.asarray(box["size"], dtype=float) / 2.0
    )


@pytest.mark.parametrize("align_source", ["legacy", "measured"])
def test_saved_reopened_boxes_preserve_exact_current_basis(tmp_path, align_source):
    # A same-basis annotation must not round-trip through floating-point corners.
    boxes = [BOX, {"center": [2**53 + 1, 0, -7], "size": [1, 2, 3]}]
    document = _document(object_boxes=boxes, align_source=align_source)
    path = document.save(tmp_path / "authored.json")

    raw = json.loads(path.read_text(encoding="utf-8"))
    reopened = _load(path, align_source=align_source, map_frame=_tilted_frame())

    assert raw["object_boxes"] == boxes
    assert reopened.object_boxes == boxes
    assert reopened.payload()["object_boxes"] == boxes
    reopened.object_boxes[0]["center"][0] = 9.0
    reopened.save(path)
    assert _load(path, align_source=align_source).object_boxes[0]["center"][0] == 9.0


def test_absent_boxes_remain_backward_compatible(tmp_path):
    assert "object_boxes" not in _document().payload()
    assert _load(_write_route(tmp_path)).object_boxes == []


INVALID_BOXES = [
    None,
    {},
    "boxes",
    [None],
    [[]],
    [{"center": [0, 0, 0]}],
    [{"size": [1, 1, 1]}],
    [{"center": [0, 0, 0], "size": [1, 1, 1], "rotation": [0, 0, 0]}],
    [{"center": [0, 0], "size": [1, 1, 1]}],
    [{"center": [0, 0, 0], "size": [1, 1, 1, 1]}],
    [{"center": "000", "size": [1, 1, 1]}],
    [{"center": [False, 0, 0], "size": [1, 1, 1]}],
    [{"center": [0, 0, 0], "size": [True, 1, 1]}],
    [{"center": ["1", 0, 0], "size": [1, 1, 1]}],
    [{"center": [0, 0, 0], "size": [None, 1, 1]}],
    [{"center": [float("nan"), 0, 0], "size": [1, 1, 1]}],
    [{"center": [0, 0, float("inf")], "size": [1, 1, 1]}],
    [{"center": [0, 0, 0], "size": [float("nan"), 1, 1]}],
    [{"center": [0, 0, 0], "size": [1, float("inf"), 1]}],
    [{"center": [0, 0, 0], "size": [1, 1, float("-inf")]}],
    [{"center": [0, 0, 0], "size": [0, 1, 1]}],
    [{"center": [0, 0, 0], "size": [1, -1, 1]}],
    [{"center": [10**400, 0, 0], "size": [1, 1, 1]}],
]


@pytest.mark.parametrize("boxes", INVALID_BOXES)
def test_invalid_boxes_are_rejected_when_loading(tmp_path, boxes):
    path = _write_route(tmp_path, object_boxes=boxes)
    with pytest.raises(ValueError, match="object_boxes"):
        _load(path)


@pytest.mark.parametrize("boxes", INVALID_BOXES)
def test_invalid_boxes_are_rejected_before_saving(tmp_path, boxes):
    document = _document(object_boxes=boxes)
    with pytest.raises(ValueError, match="object_boxes"):
        document.save(tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("route_frame", ["aligned", "glomap"])
@pytest.mark.parametrize("declared_alignment", [None, "legacy"])
def test_tilted_basis_aabb_contains_original_physical_corners(
    tmp_path, route_frame, declared_alignment,
):
    raw = _document(object_boxes=[BOX]).payload()
    raw["frame"] = route_frame
    if declared_alignment is None:
        raw.pop("align_source")
    else:
        raw["align_source"] = declared_alignment
    if route_frame == "glomap":
        raw["waypoints"] = editor_to_glomap(raw["waypoints"]).tolist()
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    measured_frame = _tilted_frame()

    converted = _load(path, map_frame=measured_frame, align_source="measured")
    box = converted.object_boxes[0]
    original_physical_corners = editor_to_glomap(_corners(BOX))
    current_corners = glomap_to_editor(original_physical_corners, measured_frame)
    center, size = np.asarray(box["center"]), np.asarray(box["size"])
    lo, hi = center - size / 2.0, center + size / 2.0

    assert np.all(current_corners >= lo - 1e-12)
    assert np.all(current_corners <= hi + 1e-12)
    np.testing.assert_allclose(lo, current_corners.min(axis=0), atol=1e-12)
    np.testing.assert_allclose(hi, current_corners.max(axis=0), atol=1e-12)
    assert np.all(size > 0)
    assert box["size"] != BOX["size"]
    converted.save(path)
    assert _load(path, map_frame=measured_frame, align_source="measured").object_boxes == [box]


def test_glomap_route_boxes_are_still_aligned_editor_coordinates(tmp_path):
    path = _write_route(tmp_path, frame="glomap", object_boxes=[BOX])
    loaded = _load(path)
    assert loaded.object_boxes == [BOX]
    assert loaded.points == [[0.0, 0.0, -0.0], [1.0, 3.0, -2.0]]


@pytest.mark.parametrize("route_frame", ["aligned", "glomap"])
def test_measured_boxes_fail_closed_without_reconstructable_basis(tmp_path, route_frame):
    path = _write_route(
        tmp_path, frame=route_frame, align_source="measured", object_boxes=[BOX],
    )
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        _load(path, align_source="legacy")


def test_preview_payload_keeps_offline_annotations_without_site_binding():
    payload = _document(object_boxes=[BOX]).payload(preview_only=True)
    assert payload["object_boxes"] == [BOX]
    assert payload["site_id"] is None
    assert payload["purpose"] == "preview_only"

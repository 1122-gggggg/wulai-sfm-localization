from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import operator_wire_overlay as overlay
from real_path_follow_controller import MapFrame, capture_mission_route_snapshot
from route_editor_model import RouteDocument, editor_to_glomap
from route_editor_wires import sample_wire


def _wire():
    return {"a": [0.0, 0.0, 2.0], "b": [4.0, 2.0, 3.0], "sag": 0.4, "up": [0.0, 0.0, 1.0]}


@pytest.fixture
def site(tmp_path, monkeypatch):
    gravity = np.array([0.1, 0.96, 0.25])
    frame = MapFrame.from_gravity(gravity)
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({
        "schema": "sfm-align/v2", "gravity_glomap": gravity.tolist(),
        "R": np.vstack([frame.east, frame.north, frame.up]).tolist(),
    }), encoding="utf-8")
    profile = SimpleNamespace(
        site_id="yard", coordinate_frame=SimpleNamespace(id="yard-frame"),
        map_align=alignment, asset_sha256=SimpleNamespace(map_ply="a" * 64),
    )
    monkeypatch.setattr(overlay, "load_site_profile", lambda _path: profile)
    return profile, frame, tmp_path / "site_profile.json"


def _route(tmp_path, *, wires=None, alignment="legacy", map_frame=None, name="route.json"):
    document = RouteDocument(
        "yard", "yard-frame", points=[[0, 0, 0], [1, 0, 0]],
        wires=[_wire()] if wires is None else wires, align_source=alignment,
    )
    path = document.save(tmp_path / name)
    kwargs = {} if map_frame is None else {"map_frame": map_frame}
    return capture_mission_route_snapshot(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_site_id="yard", expected_coordinate_frame_id="yard-frame", **kwargs,
    )


@pytest.mark.parametrize("alignment", ["legacy", "measured"])
def test_main_map_curve_uses_authored_basis_not_current_editor_basis(tmp_path, site, alignment):
    _profile, frame, profile_path = site
    snapshot = _route(tmp_path, alignment=alignment, map_frame=frame)
    notes = []
    curves = overlay.RouteWireOverlay().polylines(snapshot, profile_path, notes.append)
    assert not notes
    expected = editor_to_glomap(sample_wire(_wire()), frame if alignment == "measured" else None)
    np.testing.assert_allclose(curves[0], expected, atol=1e-12)
    assert not curves[0].flags.writeable


def test_changing_route_clears_wires_and_reload_updates_changed_endpoint(tmp_path, site):
    _profile, _frame, profile_path = site
    cache = overlay.RouteWireOverlay()
    notes = []
    first = _route(tmp_path)
    curves = cache.polylines(first, profile_path, notes.append)
    np.testing.assert_allclose(curves[0][0], [0, -2, 0])
    wire = _wire()
    wire["a"] = [0.25, 0.5, 2.75]
    changed = _route(tmp_path, wires=[wire])
    updated = cache.polylines(changed, profile_path, notes.append)
    np.testing.assert_allclose(updated[0][0], [0.25, -2.75, 0.5])
    np.testing.assert_allclose(updated[0][-1], curves[0][-1])
    empty = _route(tmp_path, wires=[], name="no-wires.json")
    assert cache.polylines(empty, profile_path, notes.append) == ()
    assert cache.polylines(changed, profile_path, notes.append)
    assert cache.polylines(None, profile_path, notes.append) == ()
    assert not notes


@pytest.mark.parametrize("failure", ["hash", "site", "frame", "map", "annotation", "alignment"])
def test_rejected_wire_selection_never_leaves_previous_map_geometry(tmp_path, site, failure):
    profile, frame, profile_path = site
    cache = overlay.RouteWireOverlay()
    notes = []
    snapshot = _route(tmp_path, alignment="measured", map_frame=frame)
    assert cache.polylines(snapshot, profile_path, notes.append)
    bad = replace(snapshot)
    if failure == "hash":
        snapshot.path.write_text("{}", encoding="utf-8")
    elif failure == "site":
        bad = replace(snapshot, site_id="another-site")
    elif failure == "frame":
        bad = replace(snapshot, coordinate_frame_id="another-frame")
    elif failure == "alignment":
        profile.map_align = None
    else:
        raw = json.loads(snapshot.path.read_text(encoding="utf-8"))
        if failure == "map":
            raw["map_ply_sha256"] = "b" * 64
        else:
            raw["wires"][0]["sag"] = -1
        snapshot.path.write_text(json.dumps(raw), encoding="utf-8")
        bad = replace(snapshot, sha256=hashlib.sha256(snapshot.path.read_bytes()).hexdigest())
    assert cache.polylines(bad, profile_path, notes.append) == ()
    assert len(notes) == 1


def test_reselecting_restored_route_recovers_from_rejected_snapshot(tmp_path, site):
    _profile, _frame, profile_path = site
    snapshot = _route(tmp_path)
    payload = snapshot.path.read_bytes()
    snapshot.path.write_text("{}", encoding="utf-8")
    cache = overlay.RouteWireOverlay()
    notes = []
    assert cache.polylines(snapshot, profile_path, notes.append) == ()
    snapshot.path.write_bytes(payload)
    curves = cache.polylines(replace(snapshot), profile_path, notes.append)
    np.testing.assert_allclose(curves[0], editor_to_glomap(sample_wire(_wire())))

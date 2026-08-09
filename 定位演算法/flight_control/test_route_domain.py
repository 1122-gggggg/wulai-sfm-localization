#!/usr/bin/env python3
"""Focused tests for the shared route domain contract.

These tests are deliberately offline.  They exercise the parser/converter
seam used by both the deployment controller and the legacy path loader.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_path
import real_path_follow_controller as rpf
from route_domain import RouteDocument


MEASURED_GRAVITY = [
    0.009067509372034937,
    0.9237964066045453,
    0.38277667042064856,
]


def _route(**extra):
    payload = {
        "schema": "sfm-flight-route/v1",
        "site_id": "field-a",
        "coordinate_frame_id": "glomap-a",
        "frame": "glomap",
        "units": "map",
        "purpose": "flight",
        "closed": False,
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
    payload.update(extra)
    return payload


def test_route_domain_is_immutable_and_preserves_contract_metadata():
    route = RouteDocument.from_data(
        _route(),
        expected_site_id="field-a",
        expected_coordinate_frame_id="glomap-a",
        require_flight_contract=True,
    )

    assert route.site_id == "field-a"
    assert route.coordinate_frame_id == "glomap-a"
    assert route.waypoints == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(AttributeError):
        route.site_id = "other"


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"site_id": "other"}, "site_id"),
        ({"coordinate_frame_id": "other"}, "coordinate_frame_id"),
        ({"purpose": "preview_only"}, "purpose"),
        ({"closed": True}, "closed"),
        ({"units": "meters"}, "units"),
    ],
)
def test_route_domain_rejects_flight_contract_mismatch(extra, message):
    with pytest.raises(ValueError, match=message):
        RouteDocument.from_data(
            _route(**extra),
            expected_site_id="field-a",
            expected_coordinate_frame_id="glomap-a",
            require_flight_contract=True,
        )


@pytest.mark.parametrize(
    "waypoints",
    [
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0, 0.0]],
    ],
)
def test_route_domain_rejects_nonfinite_short_or_zero_length_segments(waypoints):
    with pytest.raises(ValueError):
        RouteDocument.from_data({"waypoints": waypoints})


def test_route_domain_rejects_json_nan_and_infinity_tokens(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        '{"waypoints": [[0, 0, 0], [NaN, 0, 0]]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        RouteDocument.from_path(path)


@pytest.mark.parametrize(
    "radius",
    [True, False, 0, -0.1, float("nan"), float("inf"), float("-inf"), "0.5"],
)
def test_route_domain_rejects_invalid_arrival_radius(radius):
    with pytest.raises(ValueError, match="arrive_radius_map_units"):
        RouteDocument.from_data(_route(arrive_radius_map_units=radius))


def test_route_arrival_radius_is_preserved_by_controller_conversion(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps(_route(arrive_radius_map_units=0.42)),
        encoding="utf-8",
    )

    route = RouteDocument.from_path(path)
    config = rpf.config_for_route(path)

    assert route.arrive_radius_map_units == pytest.approx(0.42)
    assert config.waypoint_arrive_radius == pytest.approx(0.42)
    assert config.progress_jump_slack >= 0.42


def test_aligned_conversion_uses_the_declared_legacy_or_measured_authoring_frame():
    frame = rpf.MapFrame.from_gravity(MEASURED_GRAVITY)
    aligned = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]

    legacy = RouteDocument.from_data(
        {"frame": "aligned", "align_source": "legacy", "waypoints": aligned},
        map_frame=frame,
    )
    measured = RouteDocument.from_data(
        {"frame": "aligned", "align_source": "measured", "waypoints": aligned},
        map_frame=frame,
    )

    assert np.allclose(legacy.controller_waypoints()[1], [1.0, -3.0, 2.0])
    expected = frame.east * 1.0 + frame.north * 2.0 + frame.up * 3.0
    assert np.allclose(measured.controller_waypoints()[1], expected)
    assert not np.allclose(legacy.controller_waypoints()[1], measured.controller_waypoints()[1])


def test_closed_route_is_retained_by_domain_and_legacy_loader_can_close_it(tmp_path):
    path = tmp_path / "closed.json"
    path.write_text(
        json.dumps({
            "frame": "aligned",
            "closed": True,
            "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }),
        encoding="utf-8",
    )

    route = RouteDocument.from_path(path)
    assert route.closed is True
    assert len(route.controller_waypoints()) == 2
    loaded = load_path.load_waypoints(path)
    open_route = load_path.load_waypoints(path, close=False)
    closed = load_path.load_waypoints(path, close=True)
    assert len(loaded) == 3
    assert len(open_route) == 2
    assert len(closed) == 3
    assert np.allclose(closed[0], closed[-1])


def test_controller_loader_delegates_to_the_same_domain_conversion(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(json.dumps(_route()), encoding="utf-8")
    domain = RouteDocument.from_path(
        path,
        expected_site_id="field-a",
        expected_coordinate_frame_id="glomap-a",
        require_flight_contract=True,
    )
    loaded = rpf.load_waypoints(
        path,
        expected_site_id="field-a",
        expected_coordinate_frame_id="glomap-a",
        require_flight_contract=True,
    )
    assert np.allclose(loaded[1], domain.controller_waypoints()[1])

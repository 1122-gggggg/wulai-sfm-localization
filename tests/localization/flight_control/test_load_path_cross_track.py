#!/usr/bin/env python3
"""cross_track's default-route fallback must track the route file on disk."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FLIGHT_CONTROL_ROOT = Path(__file__).resolve().parents[3] / "定位演算法" / "flight_control"
sys.path.insert(0, str(FLIGHT_CONTROL_ROOT))

import load_path


def _write_route(path: Path, waypoints) -> None:
    path.write_text(
        json.dumps({"frame": "glomap", "waypoints": waypoints}),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _fresh_default_route_cache(monkeypatch):
    monkeypatch.setattr(load_path, "_DEFAULT_WAYPOINTS", None)


def test_default_route_is_reloaded_when_the_file_changes(tmp_path, monkeypatch):
    route = tmp_path / "route.json"
    monkeypatch.setattr(load_path, "PATH_JSON", str(route))

    _write_route(route, [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert load_path.cross_track([12.0, 0.0, 10.0]) == pytest.approx((4.0 + 100.0) ** 0.5)

    # Different content (and size): the cached route must not be reused.
    _write_route(route, [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 20.0]])
    assert load_path.cross_track([12.0, 0.0, 10.0]) == pytest.approx(2.0)
    assert load_path.on_path([12.0, 0.0, 10.0], 2.5)


def test_default_route_missing_or_unset_keeps_the_uncached_errors(tmp_path, monkeypatch):
    route = tmp_path / "route.json"
    _write_route(route, [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    monkeypatch.setattr(load_path, "PATH_JSON", str(route))
    assert load_path.cross_track([5.0, 3.0, 0.0]) == pytest.approx(3.0)

    route.unlink()
    with pytest.raises(FileNotFoundError):
        load_path.cross_track([5.0, 3.0, 0.0])

    monkeypatch.setattr(load_path, "PATH_JSON", "")
    with pytest.raises(ValueError):
        load_path.cross_track([5.0, 3.0, 0.0])


def test_explicit_waypoints_bypass_the_default_route(monkeypatch):
    monkeypatch.setattr(load_path, "PATH_JSON", "")
    waypoints = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    assert load_path.cross_track([5.0, 3.0, 0.0], waypoints) == pytest.approx(3.0)

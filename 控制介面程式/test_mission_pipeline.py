from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CONTROL_ROOT = Path(__file__).resolve().parent
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

SPEC = importlib.util.spec_from_file_location(
    "mission_pipeline_under_test", CONTROL_ROOT / "mission_pipeline.py"
)
assert SPEC is not None and SPEC.loader is not None
mission_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mission_pipeline)


def _args(**overrides):
    values = {
        "site_profile": "",
        "allow_legacy_assets": False,
        "map_ply": None,
        "bundle": None,
        "megaloc_cache": None,
        "track_landmarks": None,
        "path_json": None,
        "poles_json": None,
        "safezone_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="mission-test")


def test_operational_mode_requires_site_profile():
    with pytest.raises(SystemExit) as exc:
        mission_pipeline.resolve_mission_site_assets(
            _args(), _parser(), mode="dry-run"
        )
    assert exc.value.code == 2


def test_selftest_remains_profile_free():
    args = _args()
    assert (
        mission_pipeline.resolve_mission_site_assets(
            args, _parser(), mode="flight-selftest"
        )
        is None
    )
    assert args.bundle.endswith("your_site_reloc_map_edm.pt")


def test_profile_uses_canonical_route_paths(monkeypatch, tmp_path):
    profile = SimpleNamespace(
        source=tmp_path / "site.json",
        site_id="alpha",
        display_name="Alpha",
        map_ply=tmp_path / "alpha.ply",
        localization_bundle=tmp_path / "alpha.pt",
        route_json=None,
        poles_json=None,
        megaloc_cache=None,
        track_landmarks=None,
    )
    monkeypatch.setattr(mission_pipeline, "load_site_profile", lambda _: profile)
    monkeypatch.setattr(
        mission_pipeline,
        "_WS",
        SimpleNamespace(mission_routes=tmp_path / "mission_routes"),
    )

    args = _args(site_profile=str(profile.source))
    result = mission_pipeline.resolve_mission_site_assets(
        args, _parser(), mode="draw-path"
    )

    assert result is profile
    assert Path(args.path_json) == tmp_path / "mission_routes" / "alpha" / "flight_path.json"
    assert Path(args.poles_json) == tmp_path / "mission_routes" / "alpha" / "poles.json"


def test_flight_mode_requires_existing_route_and_poles(monkeypatch, tmp_path):
    route = tmp_path / "route.json"
    poles = tmp_path / "poles.json"
    profile = SimpleNamespace(
        source=tmp_path / "site.json",
        site_id="alpha",
        display_name="Alpha",
        map_ply=tmp_path / "alpha.ply",
        localization_bundle=tmp_path / "alpha.pt",
        route_json=route,
        poles_json=poles,
        megaloc_cache=None,
        track_landmarks=None,
    )
    monkeypatch.setattr(mission_pipeline, "load_site_profile", lambda _: profile)

    with pytest.raises(SystemExit):
        mission_pipeline.resolve_mission_site_assets(
            _args(site_profile=str(profile.source)), _parser(), mode="fly"
        )

    route.write_text("[]", encoding="utf-8")
    poles.write_text("[]", encoding="utf-8")
    args = _args(site_profile=str(profile.source))
    assert mission_pipeline.resolve_mission_site_assets(
        args, _parser(), mode="fly"
    ) is profile

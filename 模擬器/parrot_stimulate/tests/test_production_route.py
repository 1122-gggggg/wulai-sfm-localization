"""production-route adapter contracts: thin wiring, no controller math."""

import json
from pathlib import Path as _Path

import pytest

from anafi_pcmd_sim import production_route
from anafi_pcmd_sim.cli import main


def _repo_root():
    from pathlib import Path

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "定位演算法" / "flight_control").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


def _route_and_align():
    root = _repo_root()
    route = (
        root
        / "地圖檔"
        / "場域"
        / "river_site"
        / "routes"
        / "flight_route_20260913_145051_ec20651d.json"
    )
    align = (
        root
        / "地圖檔"
        / "場域"
        / "river_site"
        / "releases"
        / "river_gluemap_all8_direct_20260908"
        / "localization"
        / "T_align_gravity.json"
    )
    if not route.is_file() or not align.is_file():
        pytest.skip("requires the operator's river site assets")
    return route, align


def test_production_stack_matches_offline_sim_builder():
    route, align = _route_and_align()
    _rpf, doc, frame, cfg, ctrl = production_route.build_production_stack(route, align)
    assert len(doc.waypoints) == 4
    assert len(ctrl.wp) == 7
    assert int(cfg.max_translation_pcmd) == 50
    # Same production stack the offline sim builds: same frame, same config,
    # same expanded return-to-start waypoints.
    import sys as _sys
    from pathlib import Path as _Path
    root = _Path(__file__).resolve()
    for parent in root.parents:
        if (parent / "tools" / "sim_route_autoflight.py").is_file():
            if str(parent / "tools") not in _sys.path:
                _sys.path.insert(0, str(parent / "tools"))
            break
    import sim_route_autoflight as _sim
    sim_ctrl, sim_cfg, sim_frame, _doc = _sim.build_production_controller(
        route, align, return_to_start=True
    )
    assert [list(map(float, p)) for p in ctrl.wp] == [list(map(float, p)) for p in sim_ctrl.wp]
    assert sim_frame == frame
    assert int(sim_cfg.max_translation_pcmd) == int(cfg.max_translation_pcmd)


def test_bridge_truth_conversion_round_trips_through_map_frame():
    import tempfile as _tf

    import numpy as _np
    route, align = _route_and_align()
    _rpf, _doc, frame, _cfg, ctrl = production_route.build_production_stack(route, align)
    parsed = production_route.ProductionRouteArgs(
        route=route, align=align, meters_per_unit=5.0, output_dir=_Path(_tf.mkdtemp()),
        dry_run=True, attach_hovering=False, seed=7, duration_s=300.0,
        wind_mean_mps=(0.0, 0.0, 0.0), gust_mps=0.0,
        gust_duration_s=2.0, gust_transition_s=1.0, gust_pause_s=15.0,
    )
    import simulated_localization as _loc
    from heading_fusion import HeadingEstimator as _Heading
    _rng = _np.random.default_rng(7)
    _loc_obj = _loc.SimulatedLocalizer(_loc.SimulatedLocalizerConfig(s=5.0), _rng)
    _bridge = production_route._SphinxProductionPoseBridge(
        parsed, frame, _loc_obj, _Heading(frame), rng=_rng
    )
    target = _np.asarray(ctrl.wp[0], float)
    enu = (
        float(_np.dot(target * 5.0, _np.asarray(frame.east, float))),
        float(_np.dot(target * 5.0, _np.asarray(frame.north, float))),
        float(_np.dot(target * 5.0, _np.asarray(frame.up, float))),
    )
    back = _bridge.enu_to_map(enu)
    assert float(_np.linalg.norm(back - target)) < 1e-9
    _bridge.observe_truth(0.0, enu, 0.0, vel_ned=(0.0, 0.0))
    # World-frame metres pass through unscaled (the localizer divides by
    # scale itself), and nothing is delivered before latency elapses.
    pushed = _loc_obj._truth[-1][1]
    assert float(_np.linalg.norm(pushed - _np.asarray(enu, float))) < 1e-9
    assert _bridge.get_pose() is None


def test_bridge_predicted_flag_mirrors_pose_observation():
    import types as _types
    _ns = _types.SimpleNamespace(
        yaw_noise_deg=0.0, meters_per_unit=5.0, seed=7,
    )
    _bridge = production_route._SphinxProductionPoseBridge.__new__(
        production_route._SphinxProductionPoseBridge
    )
    _bridge.last_pose = _types.SimpleNamespace(position_observed=False)
    assert _bridge.pose_is_predicted() is True
    _bridge.last_pose = _types.SimpleNamespace(position_observed=True)
    assert _bridge.pose_is_predicted() is False
    _bridge.last_pose = None
    assert _bridge.pose_is_predicted() is False


def test_dry_run_reports_no_connection_and_no_pcmd(tmp_path, capsys):
    route, align = _route_and_align()
    exit_code = main(
        [
            "production-route",
            "--route",
            str(route),
            "--align",
            str(align),
            "--meters-per-unit",
            "5.0",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--wind-mean-mps",
            "1.0",
            "0.0",
            "0.0",
            "--gust-mps",
            "2.0",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_safety_lock"] == "10.202.0.1 only"
    assert payload["connect"] is False
    assert payload["pcmd_sent"] is False
    assert payload["localization_source"] == "synthetic_pose_not_EDM"
    assert payload["wind_sphinx"]["magnitude_mean"] == pytest.approx(1.0)
    assert payload["wind_sphinx"]["direction_mean"] == pytest.approx(0.0)
    assert payload["gust_sphinx_expr"] == "2 * gust_magnitude(2, 1, 15)"
    assert (tmp_path / "production_route_plan.json").is_file()


def test_dry_run_rejects_nonpositive_scale(tmp_path):
    route, align = _route_and_align()
    exit_code = main(
        [
            "production-route",
            "--route",
            str(route),
            "--align",
            str(align),
            "--meters-per-unit",
            "0.0",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert exit_code == 2


def test_attach_refuses_without_operator_hover_flag(tmp_path):
    route, _align = _route_and_align()
    exit_code = main(
        [
            "production-route",
            "--route",
            str(route),
            "--meters-per-unit",
            "5.0",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 2


def test_wind_mapping_matches_documented_sphinx_example():
    east_5 = production_route._wind_to_sphinx_params((5.0, 0.0, 0.0))
    assert east_5["magnitude_mean"] == pytest.approx(5.0)
    assert east_5["direction_mean"] == pytest.approx(0.0)
    north = production_route._wind_to_sphinx_params((0.0, 2.0, 0.0))
    assert north["direction_mean"] == pytest.approx(90.0)

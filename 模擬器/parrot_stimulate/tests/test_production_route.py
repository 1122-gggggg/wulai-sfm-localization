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
        route=route,
        align=align,
        meters_per_unit=5.0,
        output_dir=_Path(_tf.mkdtemp()),
        dry_run=True,
        attach_hovering=False,
        seed=7,
        duration_s=300.0,
        wind_mean_mps=(0.0, 0.0, 0.0),
        gust_mps=0.0,
        gust_duration_s=2.0,
        gust_transition_s=1.0,
        gust_pause_s=15.0,
    )
    import simulated_localization as _loc

    _rng = _np.random.default_rng(7)
    _loc_obj = _loc.SimulatedLocalizer(
        _loc.SimulatedLocalizerConfig(s=5.0, pos_sigma_u=0.0, yaw_sigma_deg=0.0, drop_rate=0.0),
        _rng,
    )
    _bridge = production_route._SphinxProductionPoseBridge(parsed, frame, _loc_obj, rng=_rng)
    target = _np.asarray(ctrl.wp[0], float)
    enu = (
        float(_np.dot(target * 5.0, _np.asarray(frame.east, float))),
        float(_np.dot(target * 5.0, _np.asarray(frame.north, float))),
        float(_np.dot(target * 5.0, _np.asarray(frame.up, float))),
    )
    back = _bridge.enu_to_map(enu)
    assert float(_np.linalg.norm(back - target)) < 1e-9
    _bridge.observe_truth(0.0, enu, 0.0, vel_ned=(0.0, 0.0))
    from unittest.mock import patch

    # ENU must be rotated into the measured map basis, not just divided by
    # scale. A full delivered pose round-trips to the controller waypoint.
    pushed = _loc_obj._truth[-1][1]
    _np.testing.assert_allclose(pushed, target * 5.0, atol=1e-9)
    with patch.object(production_route.time, "monotonic", return_value=0.0):
        assert _bridge.get_pose() is None
    with patch.object(production_route.time, "monotonic", return_value=0.12):
        _np.testing.assert_allclose(_bridge.get_pose().xyz, target, atol=1e-9)


def test_bridge_predicted_flag_mirrors_pose_observation():
    import types as _types

    _ns = _types.SimpleNamespace(
        yaw_noise_deg=0.0,
        meters_per_unit=5.0,
        seed=7,
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
            "--blackout-every-s",
            "18.0",
            "--blackout-dur-s",
            "2.5",
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
    assert payload["sensor"]["blackout_every_s"] == pytest.approx(18.0)
    assert payload["sensor"]["blackout_dur_s"] == pytest.approx(2.5)
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


def test_rejected_native_wind_setting_aborts_setup_even_with_http_success(monkeypatch):
    from types import SimpleNamespace

    connection = SimpleNamespace(
        request=lambda *_args: None,
        getresponse=lambda: SimpleNamespace(status=200, read=lambda: b'{"result":false}'),
        close=lambda: None,
    )
    monkeypatch.setattr(
        production_route.http.client, "HTTPConnection", lambda *_args, **_kw: connection
    )
    with pytest.raises(RuntimeError):
        production_route._SphinxWind(SimpleNamespace(wind_mean_mps=(0.0, 0.0, 0.0)))


def test_sphinx_limiter_keeps_sample_age_instead_of_refreshing_cached_speed():
    from types import SimpleNamespace

    production_route._ensure_flight_imports()
    import real_path_follow_controller as rpf

    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    shim = production_route._SharedSpeedLimitShim(SimpleNamespace(cfg=cfg))
    fresh = shim.apply(
        (0, 50, 0, 0),
        yaw_ned=0.0,
        north_mps=1.17,
        east_mps=0.0,
        stamp=10.0,
        attitude_stamp=10.0,
        velocity_stamp=10.0,
    )
    assert fresh[1] < 0
    stale = shim.apply(
        (0, 50, 0, 0),
        yaw_ned=0.0,
        north_mps=1.17,
        east_mps=0.0,
        stamp=11.0,
        attitude_stamp=11.0,
        velocity_stamp=10.0,
    )
    assert stale == (0, 50, 0, 0)
    assert shim.backend.state.autonomous_speed_guard_status == "SPEED_UNAVAILABLE"


def test_sphinx_adapter_uses_the_production_acceleration_ramp():
    from types import SimpleNamespace

    production_route._ensure_flight_imports()
    import real_path_follow_controller as rpf

    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    shim = production_route._SharedSpeedLimitShim(SimpleNamespace(cfg=cfg))
    applied = shim.apply(
        (0, 50, 7, -3),
        yaw_ned=0.0,
        north_mps=0.0,
        east_mps=0.0,
        stamp=10.0,
        attitude_stamp=10.0,
        velocity_stamp=10.0,
    )
    assert 0 < applied[1] <= 3
    assert applied[2:] == (7, -3)


def _blackout_bridge():
    from types import SimpleNamespace

    import numpy as np

    production_route._ensure_flight_imports()
    import real_path_follow_controller as rpf
    from simulated_localization import SimulatedLocalizer, SimulatedLocalizerConfig

    rng = np.random.default_rng(7)
    localizer = SimulatedLocalizer(
        SimulatedLocalizerConfig(
            s=5.0,
            pos_sigma_u=0.0,
            yaw_sigma_deg=0.0,
            latency_ms=0.0,
            drop_rate=0.0,
            blackout_every_s=2.0,
            blackout_dur_s=1.0,
        ),
        rng,
    )
    frame = rpf.LEGACY_MAP_FRAME
    return production_route._SphinxProductionPoseBridge(
        SimpleNamespace(meters_per_unit=5.0, yaw_noise_deg=0.0),
        frame,
        localizer,
        rng=rng,
    )


def test_sphinx_blackout_uses_firmware_attitude_and_never_advances_map_confirmation(monkeypatch):
    import math

    import numpy as np

    bridge = _blackout_bridge()
    now = [100.0]
    monkeypatch.setattr(production_route.time, "monotonic", lambda: now[0])
    bridge.observe_truth(now[0], [1.0, 2.0, 3.0], 0.0)
    assert bridge.get_pose() is None
    now[0] = 101.0
    bridge.observe_truth(now[0], [1.0, 2.0, 3.0], 0.0)
    anchor = bridge.get_pose()
    assert anchor is not None and anchor.stamp == pytest.approx(101.0)
    map_stamp = bridge.last_map_stamp
    now[0] = 102.0
    bridge.observe_truth(
        now[0],
        [900.0, 800.0, 700.0],
        math.radians(20.0),
        attitude_stamp=now[0],
        tilt_deg=10.0,
    )
    pose = bridge.get_pose()
    np.testing.assert_array_equal(pose.xyz, anchor.xyz)
    assert pose.yaw == pytest.approx(anchor.yaw - math.radians(20.0))
    assert bridge.pose_is_weak() and bridge.pose_is_predicted()
    assert bridge.pose_confidence() == 0
    assert bridge.last_map_stamp == map_stamp
    assert pose.stamp == pytest.approx(102.0)
    now[0] = 102.6
    bridge.observe_truth(
        now[0],
        [901.0, 801.0, 701.0],
        math.radians(20.0),
        attitude_stamp=102.0,
        tilt_deg=10.0,
    )
    assert bridge.get_pose() is None
    assert bridge.last_pose is None
    assert bridge.localizer.fault == "blackout-imu-stale"
    assert bridge.attitude_sample()[1] == pytest.approx(102.0)
    assert bridge.last_map_stamp == map_stamp
    now[0] = 103.1
    bridge.observe_truth(now[0], [2.0, 3.0, 4.0], math.radians(20.0))
    recovered = bridge.get_pose()
    assert recovered is not None and recovered.map_confirmed and recovered.position_observed
    np.testing.assert_allclose(recovered.xyz, bridge.enu_to_map([2.0, 3.0, 4.0]))
    assert bridge.last_map_stamp == pytest.approx(103.1)


def test_sphinx_blackout_unsafe_firmware_tilt_has_no_cached_pose(monkeypatch):
    bridge = _blackout_bridge()
    now = [100.0]
    monkeypatch.setattr(production_route.time, "monotonic", lambda: now[0])
    bridge.observe_truth(now[0], [1.0, 2.0, 3.0], 0.0)
    now[0] = 101.0
    bridge.observe_truth(now[0], [1.0, 2.0, 3.0], 0.0)
    assert bridge.get_pose() is not None
    now[0] = 102.0
    bridge.observe_truth(now[0], [1.0, 2.0, 3.0], 0.1, tilt_deg=36.0)
    assert bridge.get_pose() is None
    assert bridge.localizer.fault == "blackout-imu-tilt"
    assert bridge.pose_confidence() == 0


def test_absolute_sphinx_world_position_preserves_spawn_offset_and_height(monkeypatch):
    import numpy as np

    bridge = _blackout_bridge()
    bridge.localizer.blackout_every = 0.0
    now = [100.0]
    monkeypatch.setattr(production_route.time, "monotonic", lambda: now[0])
    world = np.array([12.0, -7.0, 6.5])
    bridge.observe_truth(
        now[0],
        world,
        0.0,
        truth_sample_stamp=1234.0,
        attitude_stamp=now[0],
    )
    pose = bridge.get_pose()
    assert pose is not None and pose.map_confirmed and pose.position_observed
    np.testing.assert_allclose(pose.xyz, bridge.enu_to_map(world))
    # No subtracting attach location/altitude: inverse frame projection retains
    # the absolute east/north offsets and nonzero Gazebo world height.
    projected = (
        np.array(
            [
                np.dot(pose.xyz, bridge.east),
                np.dot(pose.xyz, bridge.north),
                np.dot(pose.xyz, bridge.up),
            ]
        )
        * bridge.scale
    )
    np.testing.assert_allclose(projected, world, atol=1e-9)
    truth_count = len(bridge.localizer._truth)
    now[0] = 100.4
    bridge.observe_truth(
        now[0],
        world,
        0.05,
        truth_sample_stamp=1234.0,
        attitude_stamp=now[0],
    )
    assert len(bridge.localizer._truth) == truth_count
    assert bridge.last_truth_stamp == pytest.approx(100.0)
    assert bridge.get_pose() is not None
    now[0] = 100.51
    bridge.observe_truth(
        now[0],
        world,
        0.1,
        truth_sample_stamp=1234.0,
        attitude_stamp=now[0],
    )
    assert bridge.get_pose() is None
    assert bridge.pose_confidence() == 0
    assert bridge.localizer.fault == "truth-stale"
    assert len(bridge.localizer._truth) == truth_count
    now[0] = 100.6
    next_world = np.array([12.2, -6.8, 6.7])
    bridge.observe_truth(
        now[0],
        next_world,
        0.1,
        truth_sample_stamp=1234.6,
        attitude_stamp=now[0],
    )
    recovered = bridge.get_pose()
    assert recovered is not None and recovered.map_confirmed and recovered.position_observed
    np.testing.assert_allclose(recovered.xyz, bridge.enu_to_map(next_world))


def test_invalid_or_reversed_sphinx_world_samples_fail_closed(monkeypatch):
    bridge = _blackout_bridge()
    bridge.localizer.blackout_every = 0.0
    now = [100.0]
    monkeypatch.setattr(production_route.time, "monotonic", lambda: now[0])
    bridge.observe_truth(now[0], [1.0, 2.0, 3.0], 0.0, truth_sample_stamp=50.0)
    assert bridge.get_pose() is not None
    now[0] = 100.1
    bridge.observe_truth(now[0], [1.0, 2.0, float("nan")], 0.0, truth_sample_stamp=50.1)
    assert bridge.get_pose() is None and bridge.localizer.fault == "truth-invalid"
    bridge.observe_truth(now[0], [1.0, 2.0, 3.0], 0.0, truth_sample_stamp=49.0)
    assert bridge.get_pose() is None and bridge.localizer.fault == "truth-out-of-order"


@pytest.mark.parametrize(
    "elapsed,amplitude",
    [
        (0.0, 0.0),
        (0.5, 1.5),
        (1.0, 3.0),
        (2.5, 3.0),
        (3.0, 3.0),
        (3.5, 1.5),
        (4.0, 0.0),
        (18.0, 0.0),
        (19.5, 1.5),
    ],
)
def test_native_gust_rises_holds_falls_and_repeats_after_pause(elapsed, amplitude):
    from types import SimpleNamespace

    parsed = SimpleNamespace(
        gust_mps=3.0, gust_duration_s=2.0, gust_transition_s=1.0, gust_pause_s=15.0
    )
    assert production_route._gust_amplitude(parsed, elapsed) == pytest.approx(amplitude)


def test_zero_pause_is_one_gust_not_continuous_repetition():
    from types import SimpleNamespace

    parsed = SimpleNamespace(
        gust_mps=3.0, gust_duration_s=2.0, gust_transition_s=1.0, gust_pause_s=0.0
    )
    assert production_route._gust_amplitude(parsed, 2.0) == pytest.approx(3.0)
    assert production_route._gust_amplitude(parsed, 4.0) == 0.0
    assert production_route._gust_amplitude(parsed, 6.0) == 0.0


def test_wind_vector_preserves_vertical_component_and_signed_direction():
    import math

    params = production_route._wind_to_sphinx_params((0.0, -3.0, 4.0))
    assert params["magnitude_mean"] == pytest.approx(5.0)
    assert params["direction_mean"] == pytest.approx(270.0)
    assert params["elevation_mean"] == pytest.approx(math.degrees(math.atan2(4.0, 3.0)))

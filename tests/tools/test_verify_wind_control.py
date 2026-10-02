import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import real_path_follow_controller as rpf
import sim_route_autoflight as sim
from verify_wind_control import main, score


def test_wind_acceptance_excludes_join_but_requires_all_events_and_settling():
    cfg = rpf.ControlConfig(inspect_waypoints=(), waypoint_arrive_radius=0.1)
    ctrl = rpf.RouteAutoController([np.zeros(3), np.array([1.0, 0.0, 0.0])], config=cfg)
    rows = [dict(t=0.0, target_idx=0, true_map_u=[50.0, 0.0, 50.0], pcmd=[0, 0, 0, 0])]
    for index in range(1, 101):
        now = index * 0.05
        # A one-frame return at 2.0 s must not count as stable recovery.
        error = 0.2 if 1.0 <= now < 2.5 and now != 2.0 else 0.01
        rows.append(dict(t=now, target_idx=1, true_map_u=[0.5, 0.0, error], pcmd=[0, 0, 0, 0]))
    result = SimpleNamespace(trace=rows, success=True, reason="complete", time_s=5.0)
    params = sim.SimParams(route=None, align=None, meters_per_unit=15.0)
    metrics = score(result, params, ctrl, cfg.map_frame, [1.0])
    assert metrics["cross_track_p95_m"] < 4.0
    assert metrics["recovery"] == [{"event_s": 1.0, "recovery_s": 1.5}]
    assert not metrics["accepted"]  # p95 exceeds the authored radius.
    metrics = score(result, params, ctrl, cfg.map_frame, [1.0, 10.0])
    assert metrics["wind_events_covered"] == 1
    assert not metrics["accepted"]  # Ending before the gust is not a pass.


@pytest.mark.parametrize(
    "speed_option", [[], ["--speed-limit-mps", "nan"], ["--speed-limit-mps", "0"]]
)
def test_wind_verification_requires_approved_finite_speed_limit(
    monkeypatch, capsys, tmp_path, speed_option
):
    output = tmp_path / "wind"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_wind_control.py",
            "--route",
            str(Path("unused.json")),
            "--out",
            str(output),
            *speed_option,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--speed-limit-mps" in capsys.readouterr().err
    assert not output.exists()

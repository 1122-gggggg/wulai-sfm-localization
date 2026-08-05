import pytest

from anafi_pcmd_sim.flight import analyze_pcmd_response, pcmd_response_rows
from anafi_pcmd_sim.models import PilotingCommand, Scenario, TruePosition


def test_pcmd_response_measures_speed_braking_distance_and_stopping_time() -> None:
    scenario = Scenario(
        "pitch-forward-10",
        PilotingCommand(roll=0, pitch=10, yaw=0, gaz=0),
        duration_s=2.0,
        settle_s=4.0,
    )
    samples = tuple(
        TruePosition(timestamp_s=float(t), x_m=x, y_m=0.0, z_m=1.0)
        for t, x in ((0, 0.0), (1, 1.0), (2, 2.0), (3, 2.2), (4, 2.2), (5, 2.2), (6, 2.2))
    )

    metrics = analyze_pcmd_response(samples, scenario, yaw_change_deg=4.0)
    rows = pcmd_response_rows(samples, scenario)

    assert metrics.peak_horizontal_speed_m_s == pytest.approx(1.0)
    assert metrics.mean_active_horizontal_speed_m_s == pytest.approx(1.0)
    assert metrics.braking_distance_3d_m == pytest.approx(0.2)
    assert metrics.stopping_time_after_release_s == pytest.approx(3.5)
    assert metrics.effective_average_yaw_rate_deg_s == pytest.approx(2.0)
    assert {row["phase"] for row in rows} == {"command", "braking"}


def test_pcmd_response_requires_active_telemetry() -> None:
    scenario = Scenario("short", PilotingCommand.zero(), duration_s=0.01)
    samples = (
        TruePosition(timestamp_s=1.0, x_m=0.0, y_m=0.0, z_m=1.0),
        TruePosition(timestamp_s=2.0, x_m=0.0, y_m=0.0, z_m=1.0),
    )

    with pytest.raises(ValueError, match="no command-active"):
        analyze_pcmd_response(samples, scenario, yaw_change_deg=0.0)

import json

from anafi_pcmd_sim.assessment import DiagonalAssessment
from anafi_pcmd_sim.flight import ProbeResult, write_probe_artifacts
from anafi_pcmd_sim.models import MotionDelta, PilotingCommand, Scenario, TruePosition


def test_writes_a_self_contained_truth_trajectory_receipt(tmp_path) -> None:
    start = TruePosition(timestamp_s=1.0, x_m=0.0, y_m=0.0, z_m=0.5)
    end = TruePosition(timestamp_s=2.0, x_m=1.0, y_m=0.0, z_m=1.0)
    result = ProbeResult(
        scenario=Scenario("forward-up", PilotingCommand(0, 20, 0, 20), duration_s=1.0),
        start=start,
        end=end,
        delta=MotionDelta(forward_m=1.0, right_m=0.0, up_m=0.5, yaw_change_deg=0.0),
        assessment=DiagonalAssessment(passed=True, failures=()),
        samples=(start, end),
    )

    write_probe_artifacts(result, tmp_path)

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["assessment"]["passed"] is True
    assert report["delta_in_initial_body_frame"]["up_m"] == 0.5
    assert (tmp_path / "true_trajectory.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "timestamp_s,x_m,y_m,z_m"
    )

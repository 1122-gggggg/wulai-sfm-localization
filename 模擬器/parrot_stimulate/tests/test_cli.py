import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from anafi_pcmd_sim import cli
from anafi_pcmd_sim.assessment import DirectionalCriteria
from anafi_pcmd_sim.cli import build_parser, main
from anafi_pcmd_sim.models import PilotingCommand, Scenario


def test_sweep_dry_run_prints_the_complete_fixed_yaw_direction_matrix(capsys) -> None:
    exit_code = main(
        [
            "sweep",
            "--dry-run",
            "--duration",
            "1.5",
            "--magnitude",
            "20",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario_count"] == 26
    assert payload["target_safety_lock"] == "10.202.0.1 only"
    assert payload["yaw_policy"] == "fixed zero for body-frame translation checks"
    by_name = {row["name"]: row["command"] for row in payload["scenarios"]}
    assert by_name["right-forward-up"] == {
        "roll": 20,
        "pitch": 20,
        "yaw": 0,
        "gaz": 20,
    }


def test_run_can_opt_in_to_the_unreal_window() -> None:
    args = build_parser().parse_args(["run", "--launch-sphinx", "--show-window"])

    assert args.show_window is True


def test_route_dry_run_prints_ten_points_and_a_bounded_spawn(capsys) -> None:
    exit_code = main(["route", "--dry-run", "--seed", "42", "--show-window"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["waypoint_count"] == 10
    assert payload["first_waypoint_distance_from_ground_spawn_m"] <= 2.0
    assert payload["ue_command"] == ["parrot-ue4-empty", "-quality=low"]
    assert (
        "::pose=" + payload["sphinx_ground_spawn"]["sphinx_pose"] in payload["sphinx_command"][-1]
    )


def test_response_dry_run_uses_ten_percent_on_all_four_signed_axes(capsys) -> None:
    exit_code = main(["response", "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario_count"] == 8
    assert payload["magnitude"] == 10
    assert payload["real_aircraft_calibrated"] is False
    by_name = {row["name"]: row["command"] for row in payload["scenarios"]}
    assert by_name["pitch-forward"] == {"roll": 0, "pitch": 10, "yaw": 0, "gaz": 0}
    assert by_name["yaw-counter-clockwise"] == {
        "roll": 0,
        "pitch": 0,
        "yaw": -10,
        "gaz": 0,
    }


def test_worst_case_dry_run_is_fixed_not_random(capsys) -> None:
    exit_code = main(["worst-case", "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["random_sampling"] is False
    assert payload["position_error_bound_m"] == 0.30
    assert payload["yaw_error_bound_deg"] == 10.0
    assert payload["wind_displacement_bound_m"] == 0.40
    assert {scenario["name"] for scenario in payload["scenarios"]} == {
        "false-arrival-bias",
        "overshoot-bias",
        "final-approach-wind",
        "post-command-localization-loss",
        "combined-bounds",
    }


def test_custom_run_uses_directional_criteria(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProbe:
        def run(self, _scenario: Scenario, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(assessment=SimpleNamespace(passed=True))

    monkeypatch.setattr(cli, "OlympeProbe", FakeProbe)
    monkeypatch.setattr(cli, "TrueTelemetryCollector", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "write_probe_artifacts", lambda *_args: None)

    exit_code = main(
        [
            "run",
            "--roll",
            "20",
            "--pitch",
            "0",
            "--gaz",
            "0",
            "--scenario-name",
            "right",
            "--output-dir",
            str(tmp_path / "right"),
        ]
    )

    assert exit_code == 0
    assert captured["criteria"] is None
    assert isinstance(captured["directional_criteria"], DirectionalCriteria)


def test_sweep_runs_each_scenario_in_an_isolated_child_process(tmp_path: Path, monkeypatch) -> None:
    scenarios = (
        Scenario("right", PilotingCommand(roll=20, pitch=0, yaw=0, gaz=0), 1.5),
        Scenario("left", PilotingCommand(roll=-20, pitch=0, yaw=0, gaz=0), 1.5),
    )
    output_dir = tmp_path / "sweep"
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        attempt_dir = Path(command[command.index("--output-dir") + 1])
        if len(commands) == 1:
            return SimpleNamespace(returncode=2)
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "report.json").write_text(
            json.dumps({"assessment": {"passed": True, "failures": []}}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "all_direction_scenarios", lambda **_kwargs: scenarios)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    exit_code = cli._run_sweep(
        Namespace(
            duration=1.5,
            settle=2.0,
            magnitude=20,
            output_dir=str(output_dir),
            launch_sphinx=True,
            dry_run=False,
        )
    )

    assert exit_code == 0
    assert len(commands) == 3
    assert commands[0][1:4] == ["-m", "anafi_pcmd_sim", "run"]
    assert commands[1][commands[1].index("--scenario-name") + 1] == "right"
    assert commands[1][commands[1].index("--roll") + 1] == "20"
    assert commands[2][commands[2].index("--scenario-name") + 1] == "left"
    assert commands[2][commands[2].index("--roll") + 1] == "-20"
    summary = json.loads((output_dir / "sweep_report.json").read_text(encoding="utf-8"))
    assert [result["command"] for result in summary["results"]] == [
        {"roll": 20, "pitch": 0, "yaw": 0, "gaz": 0},
        {"roll": -20, "pitch": 0, "yaw": 0, "gaz": 0},
    ]
    assert [result["attempts"] for result in summary["results"]] == [2, 1]

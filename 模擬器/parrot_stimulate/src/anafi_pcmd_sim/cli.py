"""Command-line entry point for the Sphinx-only diagonal PCMD probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .assessment import DiagonalCriteria, DirectionalCriteria
from .flight import OlympeProbe, write_probe_artifacts
from .models import PilotingCommand, Scenario, all_direction_scenarios
from .route import (
    OlympeRouteFollower,
    RouteConfig,
    generate_route,
    route_plan_payload,
    route_stress_scenario,
    worst_case_route_scenarios,
    write_route_artifacts,
    write_route_plan,
)
from .simulator import (
    ANAFI_ZERO_YAW_POSE,
    SphinxFinalApproachDisplacement,
    SphinxInstance,
    SphinxWindDisplacements,
    build_sphinx_command,
    build_ue_command,
    resolve_firmware_source,
    run_preflight,
)
from .telemetry import TrueTelemetryCollector


def _scenario_from_args(args: argparse.Namespace) -> Scenario:
    if args.roll or args.yaw or args.scenario_name:
        return Scenario(
            name=args.scenario_name or "custom",
            command=PilotingCommand(
                roll=args.roll,
                pitch=args.pitch,
                yaw=args.yaw,
                gaz=args.gaz,
            ),
            duration_s=args.duration,
            settle_s=args.settle,
        )
    return Scenario.forward_up(
        duration_s=args.duration,
        pitch=args.pitch,
        gaz=args.gaz,
        settle_s=args.settle,
    )


def _output_dir(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (Path.cwd() / "artifacts" / "runs" / timestamp).resolve()


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _sweep_plan(*, duration_s: float, settle_s: float, magnitude: int) -> dict[str, Any]:
    scenarios = all_direction_scenarios(
        duration_s=duration_s,
        settle_s=settle_s,
        magnitude=magnitude,
    )
    return {
        "scenario_count": len(scenarios),
        "magnitude": magnitude,
        "duration_s": duration_s,
        "settle_s": settle_s,
        "target_safety_lock": "10.202.0.1 only",
        "yaw_policy": "fixed zero for body-frame translation checks",
        "scenarios": [
            {"name": scenario.name, "command": asdict(scenario.command)} for scenario in scenarios
        ],
    }


def _response_scenarios(
    *,
    duration_s: float,
    settle_s: float,
    magnitude: int,
) -> tuple[Scenario, ...]:
    if not 1 <= magnitude <= 100:
        raise ValueError("response magnitude must be in 1..100")
    axes = (
        ("roll-right", PilotingCommand(magnitude, 0, 0, 0)),
        ("roll-left", PilotingCommand(-magnitude, 0, 0, 0)),
        ("pitch-forward", PilotingCommand(0, magnitude, 0, 0)),
        ("pitch-backward", PilotingCommand(0, -magnitude, 0, 0)),
        ("yaw-clockwise", PilotingCommand(0, 0, magnitude, 0)),
        ("yaw-counter-clockwise", PilotingCommand(0, 0, -magnitude, 0)),
        ("gaz-up", PilotingCommand(0, 0, 0, magnitude)),
        ("gaz-down", PilotingCommand(0, 0, 0, -magnitude)),
    )
    return tuple(
        Scenario(name, command, duration_s=duration_s, settle_s=settle_s) for name, command in axes
    )


def _response_plan(*, duration_s: float, settle_s: float, magnitude: int) -> dict[str, Any]:
    scenarios = _response_scenarios(
        duration_s=duration_s,
        settle_s=settle_s,
        magnitude=magnitude,
    )
    return {
        "scenario_count": len(scenarios),
        "magnitude": magnitude,
        "duration_s": duration_s,
        "settle_s": settle_s,
        "measurement_source": "Sphinx omniscient_anafi worldPosition plus Olympe yaw",
        "real_aircraft_calibrated": False,
        "scenarios": [
            {"name": scenario.name, "command": asdict(scenario.command)} for scenario in scenarios
        ],
    }


def _write_sweep_summary(
    output_dir: Path,
    summary: dict[str, Any],
    filename: str = "sweep_report.json",
) -> None:
    (output_dir / filename).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_sweep(args: argparse.Namespace) -> int:
    plan = _sweep_plan(
        duration_s=args.duration,
        settle_s=args.settle,
        magnitude=args.magnitude,
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    output_dir = _output_dir(args.output_dir)
    _prepare_output_dir(output_dir)
    summary: dict[str, Any] = {
        **plan,
        "output_dir": str(output_dir),
        "status": "RUNNING",
        "results": [],
    }
    scenarios = all_direction_scenarios(
        duration_s=args.duration,
        settle_s=args.settle,
        magnitude=args.magnitude,
    )
    try:
        for scenario in scenarios:
            for attempt in range(1, 4):
                attempt_dir = output_dir / scenario.name / f"attempt-{attempt}"
                command = [
                    sys.executable,
                    "-m",
                    "anafi_pcmd_sim",
                    "run",
                    "--duration",
                    str(scenario.duration_s),
                    "--settle",
                    str(scenario.settle_s),
                    "--roll",
                    str(scenario.command.roll),
                    "--pitch",
                    str(scenario.command.pitch),
                    "--gaz",
                    str(scenario.command.gaz),
                    "--scenario-name",
                    scenario.name,
                    "--output-dir",
                    str(attempt_dir),
                ]
                if args.launch_sphinx:
                    command.append("--launch-sphinx")
                child = subprocess.run(command, check=False)
                report_path = attempt_dir / "report.json"
                if child.returncode in (0, 1) and report_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    break
                if attempt < 3:
                    time.sleep(2.0)
            else:
                raise RuntimeError(
                    f"scenario {scenario.name} failed after 3 isolated attempts "
                    f"(last exit code: {child.returncode})"
                )
            summary["results"].append(
                {
                    "name": scenario.name,
                    "command": asdict(scenario.command),
                    "report": str(report_path.relative_to(output_dir)),
                    "assessment": report["assessment"],
                    "attempts": attempt,
                }
            )
    except Exception as error:  # noqa: BLE001 -- CLI reports operational failures.
        summary["status"] = "OPERATIONAL_FAILURE"
        summary["error"] = str(error)
        _write_sweep_summary(output_dir, summary)
        print(f"direction sweep failed: {error}", file=sys.stderr)
        return 2

    passed_count = sum(int(result["assessment"]["passed"]) for result in summary["results"])
    summary["status"] = "PASSED" if passed_count == len(scenarios) else "FAILED"
    summary["passed_count"] = passed_count
    summary["failed_count"] = len(scenarios) - passed_count
    _write_sweep_summary(output_dir, summary)
    print((output_dir / "sweep_report.json").as_posix())
    return 0 if summary["status"] == "PASSED" else 1


def _run_response(args: argparse.Namespace) -> int:
    plan = _response_plan(
        duration_s=args.duration,
        settle_s=args.settle,
        magnitude=args.magnitude,
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    output_dir = _output_dir(args.output_dir)
    try:
        _prepare_output_dir(output_dir)
    except Exception as error:  # noqa: BLE001 -- command boundary reports operational failures.
        print(f"PCMD response suite failed: {error}", file=sys.stderr)
        return 2
    summary: dict[str, Any] = {**plan, "status": "RUNNING", "results": []}
    for scenario in _response_scenarios(
        duration_s=args.duration,
        settle_s=args.settle,
        magnitude=args.magnitude,
    ):
        scenario_dir = output_dir / scenario.name
        command = [
            sys.executable,
            "-m",
            "anafi_pcmd_sim",
            "run",
            "--duration",
            str(scenario.duration_s),
            "--settle",
            str(scenario.settle_s),
            "--roll",
            str(scenario.command.roll),
            "--pitch",
            str(scenario.command.pitch),
            "--yaw",
            str(scenario.command.yaw),
            "--gaz",
            str(scenario.command.gaz),
            "--scenario-name",
            scenario.name,
            "--response-only",
            "--output-dir",
            str(scenario_dir),
        ]
        if args.firmware:
            command.extend(("--firmware", args.firmware))
        if args.launch_sphinx:
            command.append("--launch-sphinx")
        if args.show_window:
            command.append("--show-window")
        child = subprocess.run(command, check=False)
        report_path = scenario_dir / "report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
        )
        metrics = report.get("pcmd_response") if report is not None else None
        summary["results"].append(
            {
                "name": scenario.name,
                "command": asdict(scenario.command),
                "passed": child.returncode == 0 and metrics is not None,
                "exit_code": child.returncode,
                "response": metrics,
                "report": str(report_path.relative_to(output_dir)),
            }
        )
        if child.returncode == 2 and report is None:
            summary["status"] = "OPERATIONAL_FAILURE"
            break
    if summary["status"] == "RUNNING":
        summary["status"] = (
            "COMPLETED" if all(result["passed"] for result in summary["results"]) else "FAILED"
        )
    _write_sweep_summary(output_dir, summary, "pcmd_response_report.json")
    print((output_dir / "pcmd_response_report.json").as_posix())
    if summary["status"] == "COMPLETED":
        return 0
    return 2 if summary["status"] == "OPERATIONAL_FAILURE" else 1


def _run_route(args: argparse.Namespace) -> int:
    plan = generate_route(args.seed)
    config = RouteConfig()
    stress_scenario = (
        route_stress_scenario(args.stress_scenario, config)
        if args.stress_scenario is not None
        else None
    )
    output_dir = _output_dir(args.output_dir)
    if args.dry_run:
        payload = route_plan_payload(plan, config)
        payload["stress_scenario"] = asdict(stress_scenario) if stress_scenario else None
        payload["sphinx_command"] = build_sphinx_command(
            output_dir / "simulator",
            firmware_source=args.firmware,
            spawn_pose=plan.sphinx_spawn.sphinx_pose,
            disable_front_camera=True,
        )
        payload["ue_command"] = build_ue_command(show_window=args.show_window)
        payload["target_safety_lock"] = "10.202.0.1 only"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    try:
        _prepare_output_dir(output_dir)
    except Exception as error:  # noqa: BLE001 -- CLI reports operational failures.
        print(f"route failed: {error}", file=sys.stderr)
        return 2

    write_route_plan(plan, config, output_dir)
    telemetry = TrueTelemetryCollector(stderr_path=output_dir / "true_telemetry.stderr.log")
    follower = OlympeRouteFollower(config=config, stress_scenario=stress_scenario)
    if stress_scenario is None:
        wind_disturbances = SphinxWindDisplacements(
            seed=plan.seed ^ 0x51A7,
            maximum_displacement_m=config.error_model.wind_maximum_displacement_m,
            interval_s=config.error_model.wind_displacement_interval_s,
        )
        control_disturbance = None
    else:
        wind_disturbances = SphinxFinalApproachDisplacement(
            displacement_m=stress_scenario.final_approach_wind_m,
        )
        control_disturbance = wind_disturbances.maybe_apply
    status = "RUNNING"
    error_message: str | None = None
    exit_code = 0
    try:
        with (
            SphinxInstance(
                output_dir / "simulator",
                firmware_source=args.firmware,
                show_window=args.show_window,
                spawn_pose=plan.sphinx_spawn.sphinx_pose,
                disable_front_camera=True,
            ),
            wind_disturbances,
        ):
            follower.run(
                plan,
                telemetry=telemetry,
                before_landing=wind_disturbances.stop,
                control_disturbance=control_disturbance,
            )
        status = "COMPLETED"
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        error_message = "route interrupted by operator"
        exit_code = 130
    except Exception as error:  # noqa: BLE001 -- CLI reports operational failures.
        status = "OPERATIONAL_FAILURE"
        error_message = str(error)
        exit_code = 2
    finally:
        write_route_artifacts(
            follower,
            plan,
            telemetry,
            output_dir,
            status=status,
            error=error_message,
            wind_events=wind_disturbances.events,
        )

    if error_message:
        print(f"route failed: {error_message}", file=sys.stderr)
    print((output_dir / "route_report.json").as_posix())
    return exit_code


def _worst_case_plan(config: RouteConfig) -> dict[str, Any]:
    scenarios = worst_case_route_scenarios(config)
    return {
        "scenario_count": len(scenarios),
        "random_sampling": False,
        "position_error_bound_m": config.error_model.maximum_position_error_m,
        "yaw_error_bound_deg": config.error_model.maximum_yaw_error_deg,
        "wind_displacement_bound_m": config.error_model.wind_maximum_displacement_m,
        "scenarios": [asdict(scenario) for scenario in scenarios],
    }


def _run_worst_case(args: argparse.Namespace) -> int:
    config = RouteConfig()
    plan = _worst_case_plan(config)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    output_dir = _output_dir(args.output_dir)
    try:
        _prepare_output_dir(output_dir)
    except Exception as error:  # noqa: BLE001 -- command boundary reports operational failures.
        print(f"worst-case suite failed: {error}", file=sys.stderr)
        return 2

    summary: dict[str, Any] = {**plan, "status": "RUNNING", "results": []}
    for scenario in worst_case_route_scenarios(config):
        scenario_dir = output_dir / scenario.name
        command = [
            sys.executable,
            "-m",
            "anafi_pcmd_sim",
            "route",
            "--seed",
            str(args.seed),
            "--stress-scenario",
            scenario.name,
            "--output-dir",
            str(scenario_dir),
        ]
        if args.firmware:
            command.extend(("--firmware", args.firmware))
        if args.show_window:
            command.append("--show-window")
        child = subprocess.run(command, check=False)
        report_path = scenario_dir / "route_report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
        )
        wind_triggered = bool(
            scenario.final_approach_wind_m == 0.0
            or (report is not None and report["wind_disturbance_event_count"] == 1)
        )
        loss_triggered = bool(
            not scenario.drop_after_nonzero_pcmd
            or (
                report is not None
                and any(event["reason"] == "localization_lost" for event in report["safety_events"])
            )
        )
        passed = bool(
            child.returncode == 0
            and report is not None
            and report["status"] == "COMPLETED"
            and report["landed"]
            and report["completed_waypoint_count"] == report["waypoint_count"]
            and all(
                arrival["arrival_error_m"] <= config.arrival_tolerance_m
                for arrival in report["arrivals"]
            )
            and wind_triggered
            and loss_triggered
        )
        summary["results"].append(
            {
                "name": scenario.name,
                "passed": passed,
                "exit_code": child.returncode,
                "wind_triggered": wind_triggered,
                "post_command_loss_triggered": loss_triggered,
                "report": str(report_path.relative_to(output_dir)),
            }
        )
        if child.returncode == 2 and report is None:
            summary["status"] = "OPERATIONAL_FAILURE"
            break

    if summary["status"] == "RUNNING":
        summary["status"] = (
            "PASSED" if all(result["passed"] for result in summary["results"]) else "FAILED"
        )
    _write_sweep_summary(output_dir, summary, "worst_case_report.json")
    print((output_dir / "worst_case_report.json").as_posix())
    if summary["status"] == "PASSED":
        return 0
    return 2 if summary["status"] == "OPERATIONAL_FAILURE" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sphinx-only PCMD diagonal-flight probe for Parrot ANAFI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="check local Sphinx prerequisites without starting it"
    )
    preflight.add_argument(
        "--firmware",
        default=None,
        help="verified local ANAFI PC firmware image",
    )

    dry_run = subparsers.add_parser("dry-run", help="print the exact simulator and PCMD plan")
    run = subparsers.add_parser("run", help="run one PCMD probe against Sphinx only")
    for command in (dry_run, run):
        command.add_argument("--duration", type=float, default=1.5)
        command.add_argument("--settle", type=float, default=2.0)
        command.add_argument("--roll", type=int, default=0)
        command.add_argument("--pitch", type=int, default=20)
        command.add_argument("--yaw", type=int, default=0)
        command.add_argument("--gaz", type=int, default=20)
        command.add_argument(
            "--scenario-name",
            default=None,
            help="label for a custom PCMD vector",
        )
        command.add_argument("--output-dir", default=None)
        command.add_argument(
            "--firmware",
            default=None,
            help="local ANAFI PC firmware image; default is the verified project cache",
        )
        command.add_argument(
            "--show-window",
            action="store_true",
            help="open the local Unreal Engine visualizer instead of rendering off-screen",
        )
    run.add_argument(
        "--launch-sphinx",
        action="store_true",
        help="launch and tear down the local headless Sphinx instance automatically",
    )
    run.add_argument(
        "--response-only",
        action="store_true",
        help="record the PCMD response without applying directional pass/fail thresholds",
    )
    sweep = subparsers.add_parser(
        "sweep",
        help="run all 26 non-zero roll/pitch/gaz directions against Sphinx only",
    )
    sweep.add_argument("--duration", type=float, default=1.5)
    sweep.add_argument("--settle", type=float, default=2.0)
    sweep.add_argument("--magnitude", type=int, default=20)
    sweep.add_argument("--output-dir", default=None)
    sweep.add_argument(
        "--launch-sphinx",
        action="store_true",
        help="launch and tear down the local headless Sphinx instance automatically",
    )
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="print the 26-case PCMD matrix without starting any process",
    )
    response = subparsers.add_parser(
        "response",
        help="measure speed, braking distance, and yaw response for each PCMD axis",
    )
    response.add_argument("--duration", type=float, default=3.0)
    response.add_argument("--settle", type=float, default=3.0)
    response.add_argument("--magnitude", type=int, default=10)
    response.add_argument("--output-dir", default=None)
    response.add_argument("--firmware", default=None)
    response.add_argument("--launch-sphinx", action="store_true")
    response.add_argument("--show-window", action="store_true")
    response.add_argument("--dry-run", action="store_true")
    route = subparsers.add_parser(
        "route",
        help="generate and follow a winding 10-waypoint route with closed-loop PCMD",
    )
    route.add_argument("--seed", type=int, default=42)
    route.add_argument("--output-dir", default=None)
    route.add_argument(
        "--firmware",
        default=None,
        help="local ANAFI PC firmware image; default is the verified project cache",
    )
    route.add_argument(
        "--show-window",
        action="store_true",
        help="open the local Unreal Engine visualizer",
    )
    route.add_argument(
        "--dry-run",
        action="store_true",
        help="print the generated route and controller plan without starting any process",
    )
    route.add_argument(
        "--stress-scenario",
        choices=[scenario.name for scenario in worst_case_route_scenarios()],
        default=None,
        help="replace random errors with one deterministic bounded worst-case scenario",
    )
    worst_case = subparsers.add_parser(
        "worst-case",
        help="run the deterministic route-error matrix instead of relying on random seeds",
    )
    worst_case.add_argument("--seed", type=int, default=42)
    worst_case.add_argument("--output-dir", default=None)
    worst_case.add_argument("--firmware", default=None)
    worst_case.add_argument("--show-window", action="store_true")
    worst_case.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args.firmware)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return 0 if result.passed else 2

    if args.command == "sweep":
        return _run_sweep(args)

    if args.command == "response":
        return _run_response(args)

    if args.command == "route":
        return _run_route(args)

    if args.command == "worst-case":
        return _run_worst_case(args)

    output_dir = _output_dir(args.output_dir)
    scenario = _scenario_from_args(args)
    if args.command == "dry-run":
        print(
            json.dumps(
                {
                    "sphinx_command": build_sphinx_command(
                        output_dir / "simulator", firmware_source=args.firmware
                    ),
                    "ue_command": build_ue_command(show_window=args.show_window),
                    "firmware_source": resolve_firmware_source(args.firmware),
                    "spawn_pose": ANAFI_ZERO_YAW_POSE,
                    "scenario": {
                        "name": scenario.name,
                        "duration_s": scenario.duration_s,
                        "settle_s": scenario.settle_s,
                        "command": scenario.command.__dict__,
                    },
                    "target_safety_lock": "10.202.0.1 only",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    directional_criteria = (
        DirectionalCriteria()
        if (args.roll or args.scenario_name) and not args.response_only
        else None
    )
    criteria = None if directional_criteria is not None else DiagonalCriteria()
    try:
        _prepare_output_dir(output_dir)
        telemetry = TrueTelemetryCollector(stderr_path=output_dir / "true_telemetry.stderr.log")
        if args.launch_sphinx:
            with SphinxInstance(
                output_dir / "simulator",
                firmware_source=args.firmware,
                show_window=args.show_window,
            ):
                result = OlympeProbe().run(
                    scenario,
                    criteria=criteria,
                    directional_criteria=directional_criteria,
                    telemetry=telemetry,
                    assess=not args.response_only,
                )
        else:
            result = OlympeProbe().run(
                scenario,
                criteria=criteria,
                directional_criteria=directional_criteria,
                telemetry=telemetry,
                assess=not args.response_only,
            )
    except Exception as error:  # noqa: BLE001 -- command boundary reports all operational failures.
        print(f"probe failed: {error}", file=sys.stderr)
        return 2

    write_probe_artifacts(result, output_dir)
    print((output_dir / "report.json").as_posix())
    return 0 if result.assessment is None or result.assessment.passed else 1

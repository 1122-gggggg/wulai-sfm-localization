"""Olympe-driven PCMD probe against Sphinx's virtual ANAFI endpoint."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from .assessment import (
    DiagonalAssessment,
    DiagonalCriteria,
    DirectionalCriteria,
    assess_diagonal_motion,
    assess_directional_motion,
)
from .models import MotionDelta, Scenario, TruePosition
from .motion import delta_in_initial_body_frame
from .safety import SPHINX_DRONE_IP, require_sphinx_target
from .telemetry import TrueTelemetryCollector


@dataclass(frozen=True)
class ProbeResult:
    scenario: Scenario
    start: TruePosition
    end: TruePosition
    delta: MotionDelta
    assessment: DiagonalAssessment | None
    samples: tuple[TruePosition, ...]
    response: PcmdResponseMetrics | None = None


@dataclass(frozen=True)
class PcmdResponseMetrics:
    """Measured Sphinx response to one bounded PCMD step and zero release."""

    command_duration_s: float
    peak_horizontal_speed_m_s: float
    peak_vertical_speed_m_s: float
    mean_active_horizontal_speed_m_s: float
    mean_active_vertical_speed_m_s: float
    braking_distance_3d_m: float
    stopping_time_after_release_s: float | None
    effective_average_yaw_rate_deg_s: float


def pcmd_response_rows(
    samples: tuple[TruePosition, ...],
    scenario: Scenario,
) -> list[dict[str, float | str]]:
    """Derive velocity samples and label command-active versus braking phases."""
    if not samples:
        return []
    started_at = samples[0].timestamp_s
    released_at = started_at + scenario.duration_s
    rows: list[dict[str, float | str]] = []
    for before, after in pairwise(samples):
        dt = after.timestamp_s - before.timestamp_s
        if dt <= 1e-6:
            continue
        vx = (after.x_m - before.x_m) / dt
        vy = (after.y_m - before.y_m) / dt
        vz = (after.z_m - before.z_m) / dt
        midpoint = (before.timestamp_s + after.timestamp_s) / 2.0
        rows.append(
            {
                "elapsed_s": midpoint - started_at,
                "phase": "command" if midpoint <= released_at else "braking",
                "velocity_x_m_s": vx,
                "velocity_y_m_s": vy,
                "velocity_z_m_s": vz,
                "horizontal_speed_m_s": math.hypot(vx, vy),
                "vertical_speed_m_s": vz,
                "speed_3d_m_s": math.sqrt(vx * vx + vy * vy + vz * vz),
            }
        )
    return rows


def analyze_pcmd_response(
    samples: tuple[TruePosition, ...],
    scenario: Scenario,
    *,
    yaw_change_deg: float,
    stopped_speed_m_s: float = 0.05,
) -> PcmdResponseMetrics:
    """Measure speed, release-to-rest distance, and average commanded yaw rate."""
    if len(samples) < 2:
        raise ValueError("PCMD response analysis requires at least two telemetry samples")
    rows = pcmd_response_rows(samples, scenario)
    started_at = samples[0].timestamp_s
    release_elapsed_s = scenario.duration_s
    active = [row for row in rows if row["phase"] == "command"]
    if not active:
        raise ValueError("PCMD response contains no command-active telemetry interval")
    release_sample = min(
        samples,
        key=lambda sample: abs((sample.timestamp_s - started_at) - release_elapsed_s),
    )
    end = samples[-1]
    braking_distance = math.dist(
        (release_sample.x_m, release_sample.y_m, release_sample.z_m),
        (end.x_m, end.y_m, end.z_m),
    )
    stopping_time: float | None = None
    quiet_count = 0
    for row in rows:
        if row["phase"] != "braking":
            continue
        quiet_count = quiet_count + 1 if float(row["speed_3d_m_s"]) <= stopped_speed_m_s else 0
        if quiet_count >= 3:
            stopping_time = max(0.0, float(row["elapsed_s"]) - release_elapsed_s)
            break
    return PcmdResponseMetrics(
        command_duration_s=scenario.duration_s,
        peak_horizontal_speed_m_s=max(float(row["horizontal_speed_m_s"]) for row in rows),
        peak_vertical_speed_m_s=max(abs(float(row["vertical_speed_m_s"])) for row in rows),
        mean_active_horizontal_speed_m_s=sum(float(row["horizontal_speed_m_s"]) for row in active)
        / len(active),
        mean_active_vertical_speed_m_s=sum(float(row["vertical_speed_m_s"]) for row in active)
        / len(active),
        braking_distance_3d_m=braking_distance,
        stopping_time_after_release_s=stopping_time,
        effective_average_yaw_rate_deg_s=yaw_change_deg / scenario.duration_s,
    )


class OlympeProbe:
    """Run a bounded PCMD input and collect Sphinx's unestimated true trajectory.

    The pilot uses ``Drone.piloting`` instead of emitting one-off PCMD messages:
    Olympe owns the 50 ms PCMD cadence. See Parrot's ARDrone3 PCMD reference:
    https://developer.parrot.com/docs/olympe/arsdkng_ardrone3_piloting.html
    """

    def __init__(self, target_ip: str = SPHINX_DRONE_IP) -> None:
        require_sphinx_target(target_ip)
        self.target_ip = target_ip

    @staticmethod
    def _expect_success(result: Any, operation: str) -> None:
        if not result.success():
            raise RuntimeError(f"Olympe operation failed: {operation}")

    @staticmethod
    def _read_yaw_rad(drone: Any, attitude_message: Any) -> float:
        state = drone.get_state(attitude_message)
        yaw = state.get("yaw") if state is not None else None
        if yaw is None:
            raise RuntimeError("Olympe did not provide an AttitudeChanged yaw state")
        return float(yaw)

    def run(
        self,
        scenario: Scenario,
        *,
        criteria: DiagonalCriteria | None = None,
        directional_criteria: DirectionalCriteria | None = None,
        telemetry: TrueTelemetryCollector | None = None,
        assess: bool = True,
    ) -> ProbeResult:
        """Take off, apply one PCMD vector, release it, then land safely."""
        require_sphinx_target(self.target_ip)
        if criteria is not None and directional_criteria is not None:
            raise ValueError("choose either diagonal or directional criteria")
        if criteria is None:
            criteria = DiagonalCriteria()
        import olympe
        from olympe.messages.ardrone3.Piloting import Landing, TakeOff
        from olympe.messages.ardrone3.PilotingState import AttitudeChanged, FlyingStateChanged

        collector = telemetry or TrueTelemetryCollector()
        drone = olympe.Drone(self.target_ip)
        connected = False
        airborne = False
        piloting_started = False
        try:
            if not drone.connect():
                raise RuntimeError(f"unable to connect to Sphinx virtual ANAFI at {self.target_ip}")
            connected = True
            # Official Olympe guidance: wait for hovering after TakeOff before movement.
            takeoff = drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=15)).wait()
            self._expect_success(takeoff, "takeoff and hover")
            airborne = True

            # Start here, not before takeoff: the initial position must not include ascent.
            collector.start()
            start = collector.wait_for_sample(timeout_s=15)
            initial_yaw = self._read_yaw_rad(drone, AttitudeChanged)
            command = scenario.command
            if not drone.piloting(
                command.roll,
                command.pitch,
                command.yaw,
                command.gaz,
                scenario.duration_s,
            ):
                raise RuntimeError("Olympe refused to start PCMD piloting")
            piloting_started = True

            # Olympe sends the active command periodically and zeros it after piloting_time.
            time.sleep(scenario.duration_s + 0.15)
            if not drone.piloting(0, 0, 0, 0, 0):
                raise RuntimeError("Olympe refused the zero-PCMD safety command")
            time.sleep(scenario.settle_s)

            end = collector.latest_sample()
            final_yaw = self._read_yaw_rad(drone, AttitudeChanged)
            # The launcher fixes Gazebo spawn yaw at zero. Olympe attitude is
            # used only to measure relative yaw change, not to rotate ENU axes.
            delta = delta_in_initial_body_frame(
                start,
                end,
                initial_yaw_rad=0.0,
                final_yaw_rad=final_yaw - initial_yaw,
            )
            assessment = None
            if assess:
                assessment = (
                    assess_directional_motion(
                        delta,
                        scenario.command,
                        directional_criteria,
                    )
                    if directional_criteria is not None
                    else assess_diagonal_motion(delta, criteria)
                )
            samples = collector.samples()
            return ProbeResult(
                scenario=scenario,
                start=start,
                end=end,
                delta=delta,
                assessment=assessment,
                samples=samples,
                response=analyze_pcmd_response(
                    samples,
                    scenario,
                    yaw_change_deg=delta.yaw_change_deg,
                ),
            )
        finally:
            if connected:
                with contextlib.suppress(Exception):
                    drone.piloting(0, 0, 0, 0, 0)
                if piloting_started:
                    with contextlib.suppress(Exception):
                        drone.stop_piloting()
                if airborne:
                    with contextlib.suppress(Exception):
                        landing = drone(
                            Landing() >> FlyingStateChanged(state="landed", _timeout=20)
                        ).wait()
                        self._expect_success(landing, "landing")
                with contextlib.suppress(Exception):
                    drone.disconnect()
            collector.stop()


def write_probe_artifacts(result: ProbeResult, output_dir: Path) -> None:
    """Persist a small, self-contained receipt of the simulator experiment."""
    from .route import pcmd_physical_setpoints

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario": {
            "name": result.scenario.name,
            "duration_s": result.scenario.duration_s,
            "settle_s": result.scenario.settle_s,
            "command": asdict(result.scenario.command),
        },
        "sphinx_truth_coordinate_frame": "Gazebo ENU metres",
        "start": asdict(result.start),
        "end": asdict(result.end),
        "delta_in_initial_body_frame": asdict(result.delta),
        "configured_anafi_setpoints": pcmd_physical_setpoints(result.scenario.command),
        "assessment": asdict(result.assessment) if result.assessment is not None else None,
        "pcmd_response": asdict(result.response) if result.response is not None else None,
        "sample_count": len(result.samples),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "true_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_s", "x_m", "y_m", "z_m"])
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in result.samples)
    response_rows = pcmd_response_rows(result.samples, result.scenario)
    if response_rows:
        with (output_dir / "pcmd_response.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(response_rows[0]))
            writer.writeheader()
            writer.writerows(response_rows)

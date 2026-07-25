from types import SimpleNamespace

import numpy as np
import pytest

from controllers import ALGORITHM_ORDER
from route_geometry import RouteModel
from run_sphinx_anafi_convergence import (
    TrialSetupError,
    _goto,
    _product_version_message,
    apply_backend_interpretation,
    build_trial_plan,
    parse_args,
)
from telemetry_sources import OlympeStateTracker, SphinxTelemetrySource


def test_installed_olympe_exposes_runtime_product_version_message():
    pytest.importorskip("olympe")
    message = _product_version_message()
    assert message.args_name == ["software", "hardware"]


class FakeEvent:
    def __init__(self, uuid, args):
        self.uuid = uuid
        self.args = dict(args)


class FakeEventDrone:
    def __init__(self):
        self.events = {}

    def set_event(self, message, uuid, **args):
        self.events[message] = FakeEvent(uuid, args)

    def get_last_event(self, message):
        return self.events[message]


def _telemetry_drone():
    drone = FakeEventDrone()
    drone.set_event("pos", "p1", latitude=48.0, longitude=2.0, altitude=3.0)
    drone.set_event("att", "a1", yaw=0.25)
    drone.set_event("spd", "s1", speedX=0.0, speedY=0.0, speedZ=0.0)
    return drone


def test_cached_olympe_events_do_not_refresh_pose_or_health():
    drone = _telemetry_drone()
    src = SphinxTelemetrySource(
        drone,
        messages=("pos", "att", "spd"),
        max_event_age_s=0.5,
        max_position_age_s=2.0,
    )

    first = src.get_pose(10.0)
    assert first is not None and first.stamp == 10.0
    assert src.yaw(10.0) == pytest.approx(0.25)
    assert src.telemetry_healthy(10.4)

    cached = src.get_pose(10.6)
    assert cached is not None and cached.stamp == 10.0
    assert src.yaw(10.6) is None
    assert src.velocity_ned(10.6) is None
    assert not src.telemetry_healthy(10.6)

    # A new Olympe event UUID refreshes freshness even when a hovering payload
    # has identical values.
    drone.set_event("spd", "s2", speedX=0.0, speedY=0.0, speedZ=0.0)
    drone.set_event("att", "a2", yaw=0.25)
    refreshed = src.get_pose(10.7)
    assert refreshed is not None and refreshed.stamp == 10.7
    assert src.telemetry_healthy(10.7)


def test_fused_position_reference_keeps_event_stamp_and_is_not_called_truth():
    src = SphinxTelemetrySource(
        _telemetry_drone(),
        messages=("pos", "att", "spd"),
        max_event_age_s=0.5,
        max_position_age_s=2.0,
    )
    ref = src.get_fused_reference(20.0)
    assert ref is not None and ref.stamp == 20.0
    assert src.get_fused_reference(20.4).stamp == 20.0
    assert not hasattr(src, "get_truth")


def test_state_only_fallback_refreshes_only_when_payload_changes():
    class StateDrone:
        def __init__(self):
            self.state = {"value": 1}

        def get_state(self, _message):
            return dict(self.state)

    drone = StateDrone()
    tracker = OlympeStateTracker(drone, "message")
    assert tracker.read(1.0)[1] == 1.0
    assert tracker.read(1.4)[1] == 1.0
    drone.state["value"] = 2
    assert tracker.read(1.5)[1] == 1.5


def test_compare_plan_is_scenario_round_robin_and_counterbalanced():
    args = parse_args([
        "--backend", "kinematic",
        "--rejoin-algorithm", "compare",
        "--num-trials", str(len(ALGORITHM_ORDER)),
        "--random-seed", "7",
    ])
    plan = build_trial_plan(args, np.random.default_rng(args.random_seed))
    n = len(ALGORITHM_ORDER)

    assert [row["scenario_id"] for row in plan[:n]] == [0] * n
    by_scenario = {
        sid: [row for row in plan if row["scenario_id"] == sid]
        for sid in range(n)
    }
    assert [rows[0]["algorithm"] for rows in by_scenario.values()] == ALGORITHM_ORDER
    for algorithm in ALGORITHM_ORDER:
        positions = sorted(
            row["scenario_order_index"]
            for rows in by_scenario.values()
            for row in rows
            if row["algorithm"] == algorithm
        )
        assert positions == list(range(n))


class FakeExpectation:
    def __init__(self, success):
        self._success = success

    def wait(self):
        return self

    def success(self):
        return self._success


class FakeCommandDrone:
    def __init__(self, success=True):
        self.success = success

    def __call__(self, _expectation):
        return FakeExpectation(self.success)


class StaticSource:
    def __init__(self, xyz, yaw):
        self.pose = SimpleNamespace(xyz=np.asarray(xyz, float))
        self._yaw = yaw

    def get_pose(self, _now):
        return self.pose

    def yaw(self, _now):
        return self._yaw


class FakeMessage:
    def __rshift__(self, _other):
        return self


def test_goto_rejects_failed_expectation():
    src = StaticSource([0.0, 0.0, 0.0], 0.0)
    with pytest.raises(TrialSetupError, match="moveBy"):
        _goto(
            FakeCommandDrone(success=False), src,
            np.array([2.0, 0.0, 0.0]), 0.0,
            lambda *_args: FakeMessage(), lambda **_kwargs: FakeMessage(),
        )


def test_goto_rejects_actual_start_outside_tolerance():
    src = StaticSource([0.0, 0.0, 0.0], 0.0)
    with pytest.raises(TrialSetupError, match="position error"):
        _goto(
            FakeCommandDrone(success=True), src,
            np.array([2.0, 0.0, 0.0]), 0.0,
            lambda *_args: FakeMessage(), lambda **_kwargs: FakeMessage(),
        )


def test_sphinx_summary_cannot_claim_physical_pairing_or_recommendation():
    runtime = {
        "sphinx_version": "2.25.2",
        "firmware_version": "1.10.4",
        "olympe_version": "8.4.0",
        "firmware_image_pinned": False,
    }
    args = SimpleNamespace(
        backend="sphinx", rejoin_algorithm="compare",
        sphinx_runtime_metadata=runtime,
    )
    summary = {"by_algorithm": {}}
    apply_backend_interpretation(summary, args, [])
    assert summary["paired_scenarios"] is False
    assert summary["scenario_inputs_matched"] is True
    assert summary["physical_trials_independent"] is False
    assert summary["recommended_algorithm"] is None
    assert "exploratory_ranking" in summary
    assert summary["runtime_versions"] == runtime
    assert summary["fully_reproducible"] is False

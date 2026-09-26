import json
import math

import numpy as np
import pytest

from identify_pcmd_response import identify


def test_identifies_fused_speed_response(tmp_path):
    rng = np.random.default_rng(37)
    trajectory = tmp_path / "trajectory.jsonl"
    commands = rng.integers(-35, 36, size=(100, 2))
    velocity = np.zeros(2)
    with trajectory.open("w") as stream:
        for index in range(2000):
            t = index * 0.05
            delayed = max(0, int(math.floor((t - 0.4 - 1e-8) / 1.0)))
            if t >= 0.4:
                velocity += (0.03 * commands[delayed] - velocity) * (1 - math.exp(-0.05 / 1.2))
            if index % 4 == 0:
                sampled = velocity + [-0.1, 0.05]
                speed_stamp = t
            pcmd = [int(commands[index // 20, 1]), int(commands[index // 20, 0]), 0, 0]
            stream.write(
                json.dumps(
                    {
                        "event": "auto_route_tick",
                        "t_mono_ns": round((1000 + t) * 1e9),
                        "command_authorized_mono_ns": round((1000 + t) * 1e9),
                        "ground_speed_mono_ns": round((1000 + speed_stamp) * 1e9),
                        "speed_north_mps": float(sampled[0]),
                        "speed_east_mps": float(sampled[1]),
                        "att_yaw": 0.0,
                        "pcmd": pcmd,
                    }
                )
                + "\n"
            )
    result = identify(trajectory)
    best = result["best"]
    assert abs(best["delay_s"] - 0.4) <= 0.1
    assert abs(best["tau_s"] - 1.2) <= 0.2
    assert best["gain_mps_per_pct"] == pytest.approx(0.03, rel=0.1)
    assert best["r2"] > 0.95
    assert result["velocity_samples"] == 500
    assert result["observer_replay"]["divergence_resets"] == 0


def test_rejects_insufficient_ticks(tmp_path):
    trajectory = tmp_path / "short.jsonl"
    trajectory.write_text('{"event": "auto_route_tick"}\n')
    with pytest.raises(ValueError, match="insufficient auto_route_tick rows"):
        identify(trajectory)

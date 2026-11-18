"""Offline policy boundaries: repeatable resets, stale telemetry and no truth leakage."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from anafi_wind_env import AnafiWindEnv, FRAME_FIELDS, WindEnvConfig


def _rollout(env, seed, actions, scenario="gust"):
    observation, _ = env.reset(seed=seed, options={"scenario": scenario})
    rows = [observation.copy()]
    rewards = []
    final = None
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append(observation.copy())
        rewards.append(reward)
        if terminated or truncated:
            final = info["episode_metrics"]
            break
    return np.stack(rows), np.asarray(rewards), final


def test_reset_restores_complete_seeded_episode_after_different_control_history():
    env = AnafiWindEnv(WindEnvConfig(episode_seconds=6.0))
    actions = np.random.default_rng(81).uniform(-0.25, 0.25, size=(120, 2))
    first = _rollout(env, 1234, actions)
    _rollout(env, 5678, np.ones((120, 2)), scenario="dropout")
    repeated = _rollout(env, 1234, actions)
    np.testing.assert_array_equal(first[0], repeated[0])
    np.testing.assert_array_equal(first[1], repeated[1])
    assert first[2] == repeated[2]
    assert first[2]["duration_s"] == pytest.approx(6.0)
    assert not first[2]["failure"]
    env.close()


def test_repeated_velocity_sample_keeps_event_stamp_and_increases_age():
    env = AnafiWindEnv(WindEnvConfig(episode_seconds=3.0, randomize=False))
    env.reset(seed=12, options={"scenario": "steady"})
    previous = None
    cached_ticks = 0
    for _ in range(60):
        _, _, _, _, info = env.step(np.zeros(2))
        stamp = info["next_sample_stamps"]["velocity"]
        age = info["next_sample_ages"]["velocity"]
        if stamp is not None:
            if previous is not None and stamp["source_or_event_s"] == previous[0]:
                assert age["source_or_event_s"] == pytest.approx(previous[1] + 0.05)
                cached_ticks += 1
            previous = (stamp["source_or_event_s"], age["source_or_event_s"])
    assert cached_ticks > 0
    env.close()


def test_telemetry_outages_disable_residual_without_restamping_stale_samples():
    env = AnafiWindEnv(WindEnvConfig(episode_seconds=8.0, randomize=False))
    env.reset(seed=29, options={"scenario": "dropout"})
    stale_ticks = 0
    active_corrections = 0
    for _ in range(160):
        _, _, terminated, truncated, info = env.step(np.array([0.3, -0.3]))
        if any(age is not None and age["source_or_event_s"] > 0.5 + 1e-9
               for age in info["sample_ages"].values()):
            stale_ticks += 1
            assert not info["residual_enabled"]
            assert info["allocated_residual_pct"] == [0.0, 0.0]
        if info["residual_enabled"] and any(info["allocated_residual_pct"]):
            active_corrections += 1
        assert info["approved_pcmd"][2:] == [0, 0]
        assert max(abs(value) for value in info["allocated_residual_pct"]) <= 3.0
        if terminated or truncated:
            break
    assert stale_ticks > 0
    assert active_corrections > 0
    env.close()


def test_actor_frame_cannot_read_hidden_plant_or_drift_between_events():
    env = AnafiWindEnv(WindEnvConfig(randomize=False))
    env.reset(seed=21, options={"scenario": "steady"})
    for _ in range(10):
        env.step(np.zeros(2))
    before = env._frame()
    assert before["pose_valid"] and before["attitude_valid"] and before["velocity_valid"]
    env._state.pos_m += np.array([2.0, -3.0, 0.0])
    env._state.vel_m += np.array([4.0, -5.0, 0.0])
    env._state.tilt_fwd += 0.2
    env._state.tilt_right -= 0.1
    env._wind += np.array([1.0, 1.0, 0.0])
    assert env._frame() == before
    assert len(before) == len(FRAME_FIELDS)
    env.close()


@pytest.mark.parametrize("action", [[float("nan"), 0.0], [0.0, float("inf")], [0.0], [0.0, 0.0, 0.0]])
def test_invalid_actions_rejected_before_advancing_episode(action):
    env = AnafiWindEnv(WindEnvConfig(episode_seconds=0.1))
    env.reset(seed=4, options={"scenario": "calm"})
    with pytest.raises(ValueError, match="two finite"):
        env.step(action)
    env.step(np.zeros(2))
    _, _, terminated, truncated, info = env.step(np.zeros(2))
    assert not terminated and truncated
    assert info["episode_metrics"]["duration_s"] == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(2))
    env.close()


def test_pcmd_integer_quantization_alone_is_not_reported_as_saturation():
    env = AnafiWindEnv(WindEnvConfig(randomize=False))
    env.reset(seed=2, options={"scenario": "calm"})
    for _ in range(200):
        env.step(np.zeros(2))
    _, _, _, _, info = env.step(np.array([0.11, 0.11]))
    assert info["baseline_pcmd"] == info["baseline_requested_pcmd"]
    np.testing.assert_allclose(info["requested_residual_pct"], [1.1, 1.1])
    assert info["allocated_residual_pct"] == [1.0, 1.0]
    assert not info["saturated"]
    env.close()


def test_rounding_up_one_pcmd_axis_does_not_halve_both_residual_axes():
    env = AnafiWindEnv(WindEnvConfig(randomize=False))
    env.reset(seed=2, options={"scenario": "calm"})
    for _ in range(200):
        env.step(np.zeros(2))
    _, _, _, _, info = env.step(np.array([0.19, 0.19]))
    # The integer pitch correction fits the envelope and the 10-point cap.
    # Rounding 1.9 to 2 must not trigger the wrapper's whole-vector backoff.
    assert info["allocated_residual_pct"][1] == 2.0
    assert max(abs(v) for v in info["allocated_residual_pct"]) <= env.config.residual_cap_pct
    env.close()

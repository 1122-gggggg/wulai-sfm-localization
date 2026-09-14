"""Convergence contract for the offline map-scale estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from map_scale import convergence_trace, estimate_map_scale


def _synthetic_flight(
    *,
    scale: float = 2.5,
    duration_s: float = 120.0,
    speed_hz: float = 10.0,
    visual_hz: float = 24.0,
    seed: int = 7,
    noise_m: float = 0.05,
    drift: float = 0.0,
    park_first_s: float = 10.0,
):
    """A metric circle flown with visual poses at 1/scale plus noise."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration_s, 1.0 / speed_hz)
    # Metric truth: circle radius 8 m, one lap per 40 s, parked at start.
    angle = np.where(times < park_first_s, 0.0, 2.0 * math.pi * (times - park_first_s) / 40.0)
    truth = 8.0 * np.column_stack([np.cos(angle) - 1.0, np.sin(angle), np.zeros_like(angle)])
    truth[times < park_first_s] = 0.0
    metric_speed = np.linalg.norm(np.gradient(truth, 1.0 / speed_hz, axis=0), axis=1)
    speeds = [
        (float(t), float(max(0.0, s + rng.normal(0, 0.03)))) for t, s in zip(times, metric_speed)
    ]
    # Visual: truth scaled down, slow scale drift, pose noise in map units.
    visual_times = np.arange(0.0, duration_s, 1.0 / visual_hz)
    visual_angle = np.where(
        visual_times < park_first_s,
        0.0,
        2.0 * math.pi * (visual_times - park_first_s) / 40.0,
    )
    local_scale = scale * (1.0 + drift * (visual_times / duration_s))
    visual_xyz = (
        8.0
        * np.column_stack(
            [np.cos(visual_angle) - 1.0, np.sin(visual_angle), np.zeros_like(visual_angle)]
        )
        / local_scale[:, None]
    )
    visual_xyz[visual_times < park_first_s] = 0.0
    visual_xyz += rng.normal(0, noise_m / scale, visual_xyz.shape)
    visual = [
        (float(t), float(x), float(y), float(z)) for t, (x, y, z) in zip(visual_times, visual_xyz)
    ]
    return visual, speeds


def test_known_scale_converges_with_more_trajectory() -> None:
    visual, speeds = _synthetic_flight()
    est = estimate_map_scale(visual, speeds)
    assert est.converged, est.reason
    assert est.scale_m_per_unit == pytest.approx(2.5, rel=0.05)
    assert est.n_segments >= 8
    assert est.metric_distance_m > 10.0
    trace = convergence_trace(visual, speeds)
    assert len(trace) >= 3
    # The running IQR shrinks overall as segments accumulate.
    first_iqr = trace[0][2]
    last_iqr = trace[-1][2]
    assert first_iqr is not None and last_iqr is not None
    assert last_iqr <= first_iqr


def test_parked_recording_reports_unconverged_not_bogus() -> None:
    visual, speeds = _synthetic_flight(duration_s=30.0, park_first_s=30.0)
    est = estimate_map_scale(visual, speeds)
    assert not est.converged
    assert est.scale_m_per_unit is None
    assert est.n_segments == 0


def test_short_hop_reports_unconverged_for_lack_of_segments() -> None:
    visual, speeds = _synthetic_flight(duration_s=12.0, park_first_s=10.0)
    est = estimate_map_scale(visual, speeds)
    assert not est.converged
    assert "segments" in est.reason


def test_heavy_scale_drift_reports_spread_not_scale() -> None:
    visual, speeds = _synthetic_flight(drift=1.0)
    est = estimate_map_scale(visual, speeds)
    assert not est.converged
    assert "IQR" in est.reason
    assert est.drift_ratio is not None and est.drift_ratio > 1.2


def test_non_overlapping_series_are_rejected() -> None:
    visual, speeds = _synthetic_flight(duration_s=20.0)
    shifted = [(t + 1000.0, s) for t, s in speeds]
    est = estimate_map_scale(visual, shifted)
    assert not est.converged
    assert "overlap" in est.reason


def test_garbage_rows_are_ignored_not_fatal() -> None:
    visual, speeds = _synthetic_flight()
    visual = visual + [(float("nan"), 1.0, 2.0, 3.0), (50.0, 1.0, 2.0, float("inf"))]
    speeds = speeds + [(60.0, -1.0)]
    est = estimate_map_scale(visual, speeds)
    assert est.converged, est.reason
    assert est.scale_m_per_unit == pytest.approx(2.5, rel=0.05)

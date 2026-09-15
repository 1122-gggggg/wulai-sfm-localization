#!/usr/bin/env python3
"""Offline map-scale estimation from a flown trajectory plus IMU speed.

The SfM map is scale-free (map units) while the aircraft's fused velocity is
metric (m/s). Over any flown chunk, ``metric path length / visual endpoint
displacement`` is one noisy observation of the metres-per-map-unit scale
(endpoints, not visual path length: per-sample pose noise inflates path
length while endpoints stay unbiased). More trajectory means more chunks,
and the median over chunks converges while the IQR measures how much
monocular scale drift is still in the data.

Inputs are plain sequences on the SAME host-monotonic clock
(``fused_odometry.t_mono_ns`` and the visual pose clock already share it)::

    visual = [(t_mono_s, x, y, z), ...]   # map units, success poses only
    speeds = [(t_mono_s, speed_mps), ...] # metric, e.g. NED speed norm

Stationary time contributes no information and is excluded by the metric
speed gate, so a parked recording reports ``converged=False`` instead of a
bogus scale. Pure rotation without translation likewise carries no scale
signal and is rejected by the per-segment visual-length floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: A sample only counts as motion when the aircraft really moves in metres.
S_MIN_MPS = 0.3
#: Long motion runs are cut into chunks of at most this length so that more
#: trajectory always means more independent scale observations. Kept short so
#: endpoint displacement stays close to path length on curved legs.
SEG_MAX_S = 5.0
#: Chunks shorter than this carry more sync jitter than scale signal.
SEG_MIN_S = 1.0
#: Chunks with less visual travel than this are degenerate (e.g. pure yaw).
SEG_MIN_VISUAL_U = 0.02
#: A chunk with less metric travel than this is parked for scale purposes.
SEG_MIN_M = 1.0
#: A chunk must be in motion for at least this fraction of its steps.
MOTION_FRAC_MIN = 0.5
#: Minimum independent segments before the median is trusted.
N_MIN_SEGMENTS = 8
#: Minimum total metric travel before the median is trusted.
D_MIN_M = 10.0
#: IQR/median above this means scale drift still dominates: unconverged.
SPREAD_TOL = 0.25


@dataclass(frozen=True)
class MapScaleEstimate:
    """One scale read-out for a whole session."""

    scale_m_per_unit: float | None
    q1_m_per_unit: float | None
    q3_m_per_unit: float | None
    n_segments: int
    metric_distance_m: float
    visual_distance_u: float
    duration_s: float
    converged: bool
    reason: str
    #: Late-half median / early-half median. Far from 1 hints the visual
    #: scale itself drifted mid-flight rather than the estimator failing.
    drift_ratio: float | None = None


def _finite_rows(visual: Sequence, speeds: Sequence) -> tuple[np.ndarray, np.ndarray]:
    clean_visual = [
        (float(t), float(x), float(y), float(z))
        for t, x, y, z in visual
        if all(isinstance(v, (int, float)) and np.isfinite(v) for v in (t, x, y, z))
    ]
    clean_speeds = [
        (float(t), float(s))
        for t, s in speeds
        if all(isinstance(v, (int, float)) and np.isfinite(v) for v in (t, s))
        and float(s) >= 0.0
    ]
    v = np.array(sorted(clean_visual), dtype=float)
    s = np.array(sorted(clean_speeds), dtype=float)
    return v, s


def _interpolate_visual(
    v: np.ndarray, times: np.ndarray
) -> np.ndarray | None:
    """Visual xyz at each speed-sample time, or None without overlap."""
    if len(v) < 2 or len(times) < 2:
        return None
    if times[0] < v[0, 0] or times[-1] > v[-1, 0]:
        # Require full coverage: partial overlap would bias segment lengths.
        return None
    return np.column_stack(
        [np.interp(times, v[:, 0], v[:, 1 + axis]) for axis in range(3)]
    )


def estimate_map_scale(
    visual: Sequence[tuple[float, float, float, float]],
    speeds: Sequence[tuple[float, float]],
    *,
    s_min_mps: float = S_MIN_MPS,
    seg_max_s: float = SEG_MAX_S,
    seg_min_s: float = SEG_MIN_S,
    seg_min_visual_u: float = SEG_MIN_VISUAL_U,
    seg_min_m: float = SEG_MIN_M,
    motion_frac_min: float = MOTION_FRAC_MIN,
    n_min_segments: int = N_MIN_SEGMENTS,
    d_min_m: float = D_MIN_M,
    spread_tol: float = SPREAD_TOL,
) -> MapScaleEstimate:
    """Estimate metres-per-map-unit from trajectory + metric speed."""
    v, s = _finite_rows(visual, speeds)
    empty = MapScaleEstimate(
        None, None, None, 0, 0.0, 0.0, 0.0, False, "insufficient data"
    )
    if len(v) < 2 or len(s) < 2:
        return empty
    times = s[:, 0]
    xyz = _interpolate_visual(v, times)
    if xyz is None:
        return MapScaleEstimate(
            None, None, None, 0, 0.0, 0.0, 0.0, False,
            "visual and speed series do not overlap",
        )
    speed = s[:, 1]
    dt = np.diff(times)
    ok_step = dt > 0.0
    if not np.any(ok_step):
        return empty
    # Per-step metric lengths on the shared timeline (visual uses chunk
    # endpoints, which stay unbiased under per-sample pose noise).
    step_metric = 0.5 * (speed[:-1] + speed[1:]) * dt
    motion = (speed[:-1] >= s_min_mps) & (speed[1:] >= s_min_mps) & ok_step

    # Cut the timeline into fixed chunks so that more trajectory always means
    # more independent scale observations. A chunk only counts when it is
    # mostly in motion with enough travel on both clocks.
    ratios: list[float] = []
    total_metric = 0.0
    total_visual = 0.0
    chunk_start = times[0]
    chunk_xyz0 = xyz[0].copy()
    chunk_metric = 0.0
    chunk_motion = 0
    chunk_steps = 0

    def close_chunk(end_t: float, xyz1: np.ndarray) -> None:
        nonlocal chunk_metric, chunk_motion, chunk_steps, chunk_xyz0
        nonlocal total_metric, total_visual
        duration = end_t - chunk_start
        frac = (chunk_motion / chunk_steps) if chunk_steps else 0.0
        # Endpoint displacement, not path length: per-sample visual noise
        # inflates path length (random walk) while endpoints stay unbiased.
        # Totals only cover kept chunks, so scale ~= total_metric/total_visual.
        chunk_visual = float(np.linalg.norm(xyz1 - chunk_xyz0))
        if (
            duration >= seg_min_s
            and chunk_metric >= seg_min_m
            and chunk_visual >= seg_min_visual_u
            and frac >= motion_frac_min
        ):
            ratios.append(chunk_metric / chunk_visual)
            total_metric += chunk_metric
            total_visual += chunk_visual
        chunk_metric = 0.0
        chunk_motion = chunk_steps = 0
        chunk_xyz0 = xyz1.copy()

    for index in range(len(dt)):
        if dt[index] <= 0.0:
            continue
        if times[index] - chunk_start >= seg_max_s:
            close_chunk(times[index], xyz[index])
            chunk_start = times[index]
        chunk_metric += float(step_metric[index])
        chunk_steps += 1
        if motion[index]:
            chunk_motion += 1
    close_chunk(times[-1], xyz[-1])

    duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    return _summarize_scale(ratios, total_metric, total_visual, duration,
                            n_min_segments, d_min_m, spread_tol)


def _summarize_scale(ratios, total_metric, total_visual, duration,
                     n_min_segments, d_min_m, spread_tol) -> MapScaleEstimate:
    if len(ratios) < n_min_segments:
        return MapScaleEstimate(
            None, None, None, len(ratios), total_metric, total_visual,
            duration, False,
            f"only {len(ratios)} motion segments (< {n_min_segments})",
        )
    if total_metric < d_min_m:
        return MapScaleEstimate(
            None, None, None, len(ratios), total_metric, total_visual,
            duration, False,
            f"only {total_metric:.1f} m of motion (< {d_min_m:.0f} m)",
        )
    ordered = np.sort(np.asarray(ratios, dtype=float))
    median = float(np.median(ordered))
    q1 = float(np.percentile(ordered, 25))
    q3 = float(np.percentile(ordered, 75))
    spread = (q3 - q1) / median if median > 0 else float("inf")
    half = len(ordered) // 2
    drift = None
    if half >= 2:
        early = float(np.median(ordered[:half]))
        late = float(np.median(ordered[half:]))
        drift = (late / early) if early > 0 else None
    if not np.isfinite(median) or median <= 0.0:
        return MapScaleEstimate(
            None, q1, q3, len(ratios), total_metric, total_visual,
            duration, False, "non-positive scale", drift,
        )
    if spread > spread_tol:
        return MapScaleEstimate(
            median, q1, q3, len(ratios), total_metric, total_visual,
            duration, False,
            f"IQR/median {spread:.2f} > {spread_tol:.2f}: scale still drifting",
            drift,
        )
    return MapScaleEstimate(
        median, q1, q3, len(ratios), total_metric, total_visual,
        duration, True, "converged", drift,
    )


def convergence_trace(
    visual: Sequence[tuple[float, float, float, float]],
    speeds: Sequence[tuple[float, float]],
    **kwargs,
) -> list[tuple[int, float | None, float | None]]:
    """Running (n_segments, median, iqr) after each new segment, in order.

    Lets the caller plot "scale vs trajectory grown" without re-running the
    estimator: the median should settle while the IQR shrinks.
    """
    # Reuse the estimator's segmentation by re-estimating on growing prefixes
    # of the speed series. Quadratic in segments, but sessions have tens, not
    # millions, of them; clarity beats cleverness here.
    _, s = _finite_rows(visual, speeds)
    trace: list[tuple[int, float | None, float | None]] = []
    if len(s) < 2:
        return trace
    step = max(1, len(s) // 50)
    last_n = -1
    for end in range(2, len(s) + 1, step):
        est = estimate_map_scale(visual, [tuple(row) for row in s[:end]], **kwargs)
        if est.n_segments != last_n and est.n_segments > 0:
            if est.q1_m_per_unit is None or est.q3_m_per_unit is None:
                continue
            iqr = est.q3_m_per_unit - est.q1_m_per_unit
            trace.append((est.n_segments, est.scale_m_per_unit, iqr))
            last_n = est.n_segments
    return trace

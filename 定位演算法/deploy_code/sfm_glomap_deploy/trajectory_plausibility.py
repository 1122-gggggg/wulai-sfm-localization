"""Windowed trajectory-plausibility voter for visual pose accepts.

This module is intentionally independent of pixels, optical flow, the ESEKF,
and NED telemetry. It only looks at the already-accepted pose chain
``(center, yaw, capture_stamp)`` plus one candidate pose, and answers whether
the candidate continues that chain plausibly.

Relationship to the existing single-step gate
(:meth:`ProductionEDMTracker._trajectory_would_allow`): the existing gate
compares the candidate against the *last* accepted pose only
(``max_jump`` + yaw limit + adaptive ``limit_center_step``). A mislock that
lands just under ``max_jump`` therefore becomes the new reference and pollutes
every later check. This voter compares the candidate against the *window
median rate* instead, so one wild accept cannot move the goal.

Everything here is a RATE, never a raw distance. Accepted frames are not
evenly spaced: replay strides differ, frames are coalesced, and after a run of
misses the previous accept can be a second or more old. The first version of
this file compared raw step against a median step and, measured over the seven
720p videos, produced exactly the failure that mistake predicts -- a 1-2 s gap
made a legal 0.3-0.7 map-unit move look like a teleport, the veto then blocked
the frame that would have refreshed the window, and the stale window vetoed
the next accept too (P167 -35 frames, 河濱_P117 -30). Hence three hard rules:

1. Compare rates (per second), not distances.
2. Skip every check when the previous accept is older than
   ``MAX_CONTINUITY_GAP_S``: a gap that long carries no usable continuity
   prior, so there is nothing to judge against.
3. Cap consecutive vetoes (``MAX_CONSECUTIVE_VETOES``). A voter that can
   sustain itself on its own refusals is a deadlock, not a gate.

Fail-closed contract:

* The voter only ever *vetoes* (returns ``False``). It never creates,
  repairs, or re-scores a pose.
* Any missing / non-finite / stale input votes *allow* (``True``). A voter
  that cannot judge must not kill a possibly-good recovery.
* All thresholds are relative (factors of ``max_jump`` or of the window
  median rate). Nothing assumes metres or any site scale.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


# Minimum accepted points (excluding the candidate) required before any
# veto can fire. With fewer points there is no median to judge against.
MIN_HISTORY_POINTS = 3

# Previous-accept age above which every check is skipped. Continuity across a
# longer gap is not observable from the pose chain alone.
MAX_CONTINUITY_GAP_S = 0.5

# Consecutive vetoes after which the voter stands down and resets, so a stale
# window can never lock the tracker out of re-acquiring.
MAX_CONSECUTIVE_VETOES = 2

# Candidate speed must exceed BOTH of these to trip the jump check: a factor
# of the window median speed (catches teleports hiding under max_jump after a
# slow hover) and a fraction of the max_jump budget per second (protects
# fast-but-legal flight where every step is large).
MEDIAN_SPEED_FACTOR = 5.0
MAX_JUMP_RATE_FRACTION = 0.5

# Yaw-rate cap for the window check. Deliberately generous: normal flight
# (including yaw search) stays far below this; only a teleport-style flip
# trips it. The rate is measured against the WINDOW MEDIAN yaw over a baseline
# of at least MIN_YAW_BASELINE_S seconds -- single-frame PnP yaw noise (10-20
# deg in weak geometry) divided by one 0.125 s frame interval would otherwise
# read as a 100+ deg/s phantom rate and veto healthy frames (measured on P168
# and P119 before the median baseline was introduced).
MAX_YAW_RATE_DEG_S = 120.0
MIN_YAW_BASELINE_S = 0.3

# Reversal check: angle (deg) between the previous leg and the candidate leg
# above which a sharp zig-zag votes veto -- but only when both legs are
# significant (each longer than this fraction of max_jump), so hover jitter
# never trips it.
REVERSAL_ANGLE_DEG = 150.0
REVERSAL_MIN_LEG_FRACTION = 0.25

# Vertical check: |dz| must exceed BOTH this multiple of the horizontal step
# and this fraction of max_jump, measured across a gap no longer than
# MAX_CONTINUITY_GAP_S. Level flight with noisy altitude never trips it; only
# a floor/ceiling relocalization does. The 2026-09-06 seven-video gate caught
# the earlier version firing on legal 1-2 s climbs (P167 -35 frames), which is
# what the gap bound and the larger ratio now prevent.
VERTICAL_VS_HORIZONTAL = 4.0
VERTICAL_MIN_RATE_FRACTION = 0.5


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b in radians, in [-pi, pi]."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _finite3(vec) -> bool:
    try:
        return len(vec) == 3 and all(math.isfinite(float(v)) for v in vec)
    except (TypeError, ValueError):
        return False


@dataclass
class TrajectoryWindow:
    """Bounded history of accepted poses; append-only, drops stale points."""

    max_points: int = 8
    max_age_s: float = 3.0
    _points: deque = field(default_factory=lambda: deque(maxlen=8), repr=False)
    _consecutive_vetoes: int = 0

    def __post_init__(self) -> None:
        self.max_points = max(2, int(self.max_points))
        self.max_age_s = float(self.max_age_s)
        self._points = deque(maxlen=self.max_points)
        self._consecutive_vetoes = 0

    def reset(self) -> None:
        self._points.clear()
        self._consecutive_vetoes = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._points)

    def append(self, center, yaw: float, stamp) -> None:
        try:
            c = (float(center[0]), float(center[1]), float(center[2]))
            y = float(yaw)
            s = float(stamp)
        except (TypeError, ValueError, IndexError):
            return
        if not all(math.isfinite(v) for v in (*c, y, s)):
            return
        self._points.append((c, y, s))
        self._consecutive_vetoes = 0

    def fresh_points(self, now_s) -> list:
        """Points within ``max_age_s`` of ``now_s``; unusable clock -> []."""
        try:
            now = float(now_s)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(now):
            return []
        return [
            p
            for p in self._points
            if math.isfinite(p[2]) and 0.0 <= now - p[2] <= self.max_age_s
        ]

    def last(self):
        """Most recent point ``(center, yaw, stamp)`` or ``None``."""
        try:
            return self._points[-1]
        except IndexError:
            return None

    def note_veto(self) -> None:
        self._consecutive_vetoes += 1

    def stood_down(self) -> bool:
        """True once consecutive vetoes hit the cap; the window then resets."""
        if self._consecutive_vetoes < MAX_CONSECUTIVE_VETOES:
            return False
        self.reset()
        return True


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def vote_allow(
    history: TrajectoryWindow | list,
    candidate_center,
    candidate_yaw: float,
    candidate_stamp,
    *,
    max_jump: float,
) -> tuple[bool, str]:
    """Return ``(allow, reason)`` for one candidate accept.

    ``allow=True`` means "no objection". ``allow=False`` means veto with a
    machine-readable ``reason`` (``jump_vs_median`` / ``yaw_rate`` /
    ``reversal`` / ``vertical``). Any unusable input returns ``(True, ...)``.
    """
    try:
        mj = float(max_jump)
    except (TypeError, ValueError):
        return True, "bad_max_jump"
    if not math.isfinite(mj) or mj <= 0.0:
        return True, "bad_max_jump"
    if not _finite3(candidate_center):
        return True, "bad_candidate"
    try:
        cyaw = float(candidate_yaw)
        cstamp = float(candidate_stamp)
    except (TypeError, ValueError):
        return True, "bad_candidate"
    if not math.isfinite(cyaw) or not math.isfinite(cstamp):
        return True, "bad_candidate"

    window = history if isinstance(history, TrajectoryWindow) else None
    if window is not None and window.stood_down():
        # Cap reached: the window just reset, so this frame is judged by the
        # production gates alone and gets to refresh the history.
        return True, "stand_down"

    pts = window.fresh_points(cstamp) if window is not None else list(history or [])
    if len(pts) < MIN_HISTORY_POINTS:
        return True, "too_few_points"

    cx, cyy, cz = (
        float(candidate_center[0]),
        float(candidate_center[1]),
        float(candidate_center[2]),
    )
    px, py, pz = pts[-1][0]
    prev_stamp = pts[-1][2]

    dt = cstamp - prev_stamp
    if not math.isfinite(dt) or dt <= 1e-6:
        return True, "bad_dt"
    if dt > MAX_CONTINUITY_GAP_S:
        return True, "gap_too_long"

    step = math.dist((cx, cyy, cz), (px, py, pz))
    speed = step / dt

    # --- 1. speed vs window median speed ---------------------------------
    speeds: list[float] = []
    for older, newer in zip(pts[-4:-1], pts[-3:]):
        leg_dt = newer[2] - older[2]
        if not math.isfinite(leg_dt) or leg_dt <= 1e-6:
            continue
        speeds.append(math.dist(newer[0], older[0]) / leg_dt)
    if speeds:
        median_speed = _median(speeds)
        if (
            median_speed > 1e-9
            and speed > MEDIAN_SPEED_FACTOR * median_speed
            and step > MAX_JUMP_RATE_FRACTION * mj
        ):
            return False, "jump_vs_median"

    # --- 2. yaw rate (median baseline, not last point) --------------------
    ref_yaws = [float(p[1]) for p in pts]
    unwrapped = [ref_yaws[0]]
    for prev_y, y in zip(ref_yaws, ref_yaws[1:]):
        unwrapped.append(unwrapped[-1] + _angle_diff(y, prev_y))
    median_yaw = _median(unwrapped)
    median_stamp = _median([float(p[2]) for p in pts])
    span = cstamp - median_stamp
    if math.isfinite(span) and span + 1e-9 >= MIN_YAW_BASELINE_S:
        yaw_delta = abs(_angle_diff(cyaw, median_yaw))
        if math.isfinite(yaw_delta) and yaw_delta / span > math.radians(MAX_YAW_RATE_DEG_S):
            return False, "yaw_rate"

    # --- 3. reversal (zig-zag teleport) ----------------------------------
    if len(pts) >= 2:
        qx, qy, qz = pts[-2][0]
        leg0 = (px - qx, py - qy, pz - qz)
        leg1 = (cx - px, cyy - py, cz - pz)
        n0 = math.sqrt(sum(v * v for v in leg0))
        n1 = math.sqrt(sum(v * v for v in leg1))
        if n0 > REVERSAL_MIN_LEG_FRACTION * mj and n1 > REVERSAL_MIN_LEG_FRACTION * mj:
            cosang = sum(a * b for a, b in zip(leg0, leg1)) / (n0 * n1)
            cosang = max(-1.0, min(1.0, cosang))
            if math.degrees(math.acos(cosang)) > REVERSAL_ANGLE_DEG:
                return False, "reversal"

    # --- 4. vertical -------------------------------------------------------
    # Vertical motion dominating horizontal motion by this much, and large in
    # absolute terms against the jump budget, is a floor/ceiling relocalization
    # rather than flight. Both conditions are needed: a hover has a tiny
    # horizontal step, so the ratio alone fires on altitude noise.
    dz = abs(cz - pz)
    horizontal = math.dist((cx, cyy), (px, py))
    if dz > VERTICAL_VS_HORIZONTAL * horizontal and dz > VERTICAL_MIN_RATE_FRACTION * mj:
        return False, "vertical"

    return True, "ok"

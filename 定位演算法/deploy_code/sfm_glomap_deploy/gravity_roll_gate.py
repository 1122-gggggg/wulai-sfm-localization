"""Gravity-consistency veto for visual poses (measured gimbal-roll invariant).

The ANAFI gimbal stabilizes roll, so every camera right axis is horizontal --
perpendicular to gravity. That is not an assumption here: it is the same
measurement ``定位演算法/validation/derive_map_gravity.py`` already makes when
it derives ``T_align_gravity.json`` (smallest eigenvector of the camera right
axes). On the current river map the residual over all 1045 reference poses is
p50 0.44 deg, p95 1.53 deg, max 2.14 deg.

A wrong PnP solution has no reason to respect that invariant: RANSAC can reach
a high inlier count in repetitive geometry by tilting the camera, and the
existing gates (inlier floor, reprojection RMS, jump/yaw limits) all measure
consistency with the *correspondences*, never with gravity. This module adds
the orthogonal check.

Fail-closed contract:

* Veto only. It never produces, repairs, or rescores a pose.
* Missing / non-finite gravity or rotation -> allow. A check that cannot be
  evaluated must not kill a possibly-good frame.
* Thresholds are angles in degrees, independent of map scale, so they carry
  across sites unchanged (the invariant is a property of the gimbal, not of
  the reconstruction).
* Defaults are deliberately far outside anything observed: the roll limit is
  several times the worst reference residual, and the down limit sits beyond
  a legal nadir view (90 deg) so full gimbal travel never trips it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


#: Camera-right vs horizontal plane, in degrees. Reference max is 2.14 deg.
DEFAULT_MAX_ROLL_DEG = 12.0

#: Camera-down vs gravity, in degrees. A nadir gimbal view legitimately
#: reaches 90 deg, so only a beyond-nadir / flipped solution can exceed this.
DEFAULT_MAX_DOWN_DEG = 110.0


def load_gravity(path: str | Path) -> list[float] | None:
    """Return the unit gravity vector from a ``sfm-align/v2`` file, or None."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    raw = payload.get("gravity_glomap") if isinstance(payload, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    try:
        vec = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(v * v for v in vec))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return [v / norm for v in vec]


def _unit(vec) -> list[float] | None:
    try:
        out = [float(vec[0]), float(vec[1]), float(vec[2])]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    norm = math.sqrt(sum(v * v for v in out))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return [v / norm for v in out]


def gravity_residuals(rotation, gravity) -> tuple[float, float] | None:
    """Return ``(roll_deg, down_deg)`` for a ``cam_from_world`` rotation.

    ``rotation`` rows are the camera axes expressed in world coordinates:
    row 0 right, row 1 down, row 2 forward (the same convention
    ``edm_localizer_adapter`` uses to publish ``camera_axes_world``).
    ``roll_deg`` is how far the right axis leaves the horizontal plane;
    ``down_deg`` is the angle between the camera down axis and gravity.
    Returns ``None`` when either input is unusable.
    """
    g = _unit(gravity)
    if g is None:
        return None
    try:
        right = _unit(rotation[0])
        down = _unit(rotation[1])
    except (TypeError, IndexError):
        return None
    if right is None or down is None:
        return None
    dot_right = max(-1.0, min(1.0, sum(a * b for a, b in zip(right, g))))
    dot_down = max(-1.0, min(1.0, sum(a * b for a, b in zip(down, g))))
    return abs(math.degrees(math.asin(dot_right))), math.degrees(math.acos(dot_down))


def vote_allow(
    rotation,
    gravity,
    *,
    max_roll_deg: float = DEFAULT_MAX_ROLL_DEG,
    max_down_deg: float = DEFAULT_MAX_DOWN_DEG,
) -> tuple[bool, str, dict]:
    """Return ``(allow, reason, diagnostics)`` for one candidate rotation."""
    residuals = gravity_residuals(rotation, gravity)
    if residuals is None:
        return True, "unavailable", {}
    roll_deg, down_deg = residuals
    diag = {
        "gravity_roll_deg": round(roll_deg, 4),
        "gravity_down_deg": round(down_deg, 4),
    }
    try:
        roll_limit = float(max_roll_deg)
        down_limit = float(max_down_deg)
    except (TypeError, ValueError):
        return True, "bad_limits", diag
    if not math.isfinite(roll_limit) or roll_limit <= 0.0:
        return True, "bad_limits", diag
    if roll_deg > roll_limit:
        return False, "gravity_roll", diag
    if math.isfinite(down_limit) and down_limit > 0.0 and down_deg > down_limit:
        return False, "gravity_down", diag
    return True, "ok", diag

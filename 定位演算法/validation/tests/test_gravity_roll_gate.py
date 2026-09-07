from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from gravity_roll_gate import (  # noqa: E402
    DEFAULT_MAX_DOWN_DEG,
    DEFAULT_MAX_ROLL_DEG,
    gravity_residuals,
    load_gravity,
    vote_allow,
)


GRAVITY = [-0.010903244125662598, 0.996517323960189, 0.08267008113435093]

WORKSPACE = Path(__file__).resolve().parents[3]
REFERENCE_POSES = (
    WORKSPACE
    / "地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260831"
    / "localization/reference_poses.json"
)


def _rotation_from_axes(right, down, forward):
    return np.asarray([right, down, forward], dtype=float)


def _gravity_aligned_rotation(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Rotation whose right axis leaves horizontal by exactly ``roll_deg``."""
    g = np.asarray(GRAVITY, dtype=float)
    g /= np.linalg.norm(g)
    # Build an orthonormal basis with e0, e1 horizontal (perpendicular to g).
    seed = np.asarray([1.0, 0.0, 0.0])
    e0 = seed - (seed @ g) * g
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(g, e0)
    yaw = math.radians(yaw_deg)
    right = math.cos(yaw) * e0 + math.sin(yaw) * e1
    # Tilt right out of the horizontal plane by roll_deg toward gravity.
    roll = math.radians(roll_deg)
    right_tilted = math.cos(roll) * right + math.sin(roll) * g
    right_tilted /= np.linalg.norm(right_tilted)
    # Down axis: gravity tilted by pitch about the right axis.
    pitch = math.radians(pitch_deg)
    forward0 = np.cross(right_tilted, g)
    forward0 /= np.linalg.norm(forward0)
    down = math.cos(pitch) * g + math.sin(pitch) * forward0
    down /= np.linalg.norm(down)
    forward = np.cross(right_tilted, down)
    forward /= np.linalg.norm(forward)
    return _rotation_from_axes(right_tilted, down, forward)


def test_level_pose_allowed():
    rot = _gravity_aligned_rotation(yaw_deg=37.0, pitch_deg=5.0, roll_deg=0.0)
    allow, reason, diag = vote_allow(rot, GRAVITY)
    assert allow is True
    assert reason == "ok"
    assert diag["gravity_roll_deg"] < 1e-6


def test_nadir_gimbal_allowed():
    # Full nadir: camera down axis is horizontal (90 deg from gravity). Legal
    # gimbal travel must never be vetoed.
    rot = _gravity_aligned_rotation(pitch_deg=90.0)
    _, down_deg = gravity_residuals(rot, GRAVITY)
    assert down_deg > 89.0
    allow, reason, _ = vote_allow(rot, GRAVITY)
    assert allow is True, reason


def test_rolled_pose_vetoed():
    rot = _gravity_aligned_rotation(roll_deg=DEFAULT_MAX_ROLL_DEG + 5.0)
    allow, reason, diag = vote_allow(rot, GRAVITY)
    assert allow is False
    assert reason == "gravity_roll"
    assert diag["gravity_roll_deg"] > DEFAULT_MAX_ROLL_DEG


def test_roll_just_inside_limit_allowed():
    rot = _gravity_aligned_rotation(roll_deg=DEFAULT_MAX_ROLL_DEG - 1.0)
    allow, _, _ = vote_allow(rot, GRAVITY)
    assert allow is True


def test_upside_down_pose_vetoed():
    rot = _gravity_aligned_rotation(pitch_deg=180.0)
    allow, reason, diag = vote_allow(rot, GRAVITY)
    assert allow is False
    assert reason == "gravity_down"
    assert diag["gravity_down_deg"] > DEFAULT_MAX_DOWN_DEG


def test_missing_gravity_allows():
    rot = _gravity_aligned_rotation(roll_deg=80.0)
    for bad in (None, [0.0, 0.0, 0.0], [1.0, 2.0], "x"):
        allow, reason, _ = vote_allow(rot, bad)
        assert allow is True, reason


def test_nonfinite_rotation_allows():
    rot = np.full((3, 3), np.nan)
    allow, _, _ = vote_allow(rot, GRAVITY)
    assert allow is True


def test_zero_limit_disables_roll_check():
    rot = _gravity_aligned_rotation(roll_deg=80.0)
    allow, reason, _ = vote_allow(rot, GRAVITY, max_roll_deg=0.0)
    assert allow is True
    assert reason == "bad_limits"


def test_load_gravity_reads_align_file(tmp_path):
    path = tmp_path / "T_align_gravity.json"
    path.write_text(json.dumps({"gravity_glomap": [0.0, 2.0, 0.0]}), encoding="utf-8")
    assert load_gravity(path) == [0.0, 1.0, 0.0]
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert load_gravity(bad) is None
    assert load_gravity(tmp_path / "missing.json") is None


def test_reference_poses_satisfy_the_invariant():
    """The 1045 map references must all pass; otherwise the limit is wrong.

    This is the measurement the threshold is derived from: if a future map
    build breaks the gimbal-roll invariant, this fails before the veto can
    start killing healthy frames in flight.
    """
    if not REFERENCE_POSES.is_file():
        import pytest

        pytest.skip("river reference poses not present in this workspace")
    poses = json.loads(REFERENCE_POSES.read_text(encoding="utf-8"))["poses"]
    rolls = []
    downs = []
    for entry in poses.values():
        rot = np.asarray(entry["R"], dtype=float)
        roll_deg, down_deg = gravity_residuals(rot, GRAVITY)
        rolls.append(roll_deg)
        downs.append(down_deg)
        allow, reason, _ = vote_allow(rot, GRAVITY)
        assert allow is True, f"reference pose vetoed: {reason} roll={roll_deg:.2f}"
    assert max(rolls) < DEFAULT_MAX_ROLL_DEG / 3.0, max(rolls)
    assert max(downs) < DEFAULT_MAX_DOWN_DEG

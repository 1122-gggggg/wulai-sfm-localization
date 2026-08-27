"""The map's vertical axis is a MEASUREMENT. These tests keep it one."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION = Path(__file__).resolve().parents[1]
ROOT = VALIDATION.parents[1]
sys.path.insert(0, str(VALIDATION))
sys.path.insert(0, str(ROOT / "定位演算法" / "flight_control"))

import derive_map_gravity as dmg  # noqa: E402
import real_path_follow_controller as rpf  # noqa: E402

SITE_ALIGNMENTS = {
    "urai": (
        "地圖檔/場域/urai/maps/edm_v1/target_site_ref_poses.json",
        "地圖檔/場域/urai/maps/edm_v1/T_align_gravity.json",
    ),
    "river_site": (
        "地圖檔/場域/river_site/maps/river_site_ref_poses.json",
        "地圖檔/場域/river_site/maps/T_align_gravity.json",
    ),
}


def _horizontal_basis(gravity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = -gravity / np.linalg.norm(gravity)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, up))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    east = seed - float(np.dot(seed, up)) * up
    east /= np.linalg.norm(east)
    return east, np.cross(up, east)


def synthetic_rotations(gravity, count=400, roll_deg=0.0, pitch_deg=20.0):
    """Cameras whose right axis is horizontal (roll_deg=0), spun around the site.

    Rows of each R are the camera axes in world coordinates: [right, down, forward].
    """
    gravity = np.asarray(gravity, dtype=float)
    gravity = gravity / np.linalg.norm(gravity)
    up = -gravity
    east, north = _horizontal_basis(gravity)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    rotations = []
    for index in range(count):
        theta = 2.0 * math.pi * index / count
        forward_h = math.cos(theta) * east + math.sin(theta) * north
        right_h = np.cross(forward_h, up)
        forward = math.cos(pitch) * forward_h - math.sin(pitch) * up
        down = np.cross(forward, right_h)
        down /= np.linalg.norm(down)
        # A deliberate roll tilts the right axis out of horizontal, which is
        # exactly the premise the derivation depends on.
        right = math.cos(roll) * right_h + math.sin(roll) * down
        down = np.cross(forward, right)
        down /= np.linalg.norm(down)
        rotations.append(np.vstack([right / np.linalg.norm(right), down, forward]))
    return np.asarray(rotations)


def _write_poses(tmp_path, rotations):
    poses = {
        f"seq/{index:06d}.jpg": {"R": R.tolist(), "t": [0.0, 0.0, 0.0], "camera_id": 1}
        for index, R in enumerate(rotations)
    }
    path = tmp_path / "ref_poses.json"
    path.write_text(json.dumps({"poses": poses}), encoding="utf-8")
    return path


def test_recovers_a_known_gravity_direction():
    gravity = np.array([0.2, 0.9, 0.35])
    gravity /= np.linalg.norm(gravity)

    measured = dmg.derive_gravity(synthetic_rotations(gravity))

    angle = math.degrees(math.acos(
        float(np.clip(np.dot(measured["gravity"], gravity), -1.0, 1.0))))
    assert angle < 0.01, f"recovered gravity is {angle:.3f} deg off"


def test_sign_comes_from_the_cameras_not_from_a_world_axis():
    """The eigenvector is sign-free; resolving it against +Y reintroduces the bug."""
    gravity = np.array([0.1, -0.95, 0.2])      # DOWN is roughly -Y here
    gravity /= np.linalg.norm(gravity)

    measured = dmg.derive_gravity(synthetic_rotations(gravity))

    assert float(np.dot(measured["gravity"], gravity)) > 0.99, (
        "gravity came back inverted -- the sign was taken from a world axis"
    )
    assert measured["gravity"][1] < 0.0


def test_refuses_a_capture_whose_gimbal_was_not_level():
    """Rolled cameras break the premise, so the answer is not a measurement."""
    rolled = synthetic_rotations([0.0, 1.0, 0.0], roll_deg=25.0)
    with pytest.raises(ValueError, match="not horizontal enough"):
        dmg.derive_gravity(rolled)


def test_refuses_too_few_poses():
    with pytest.raises(ValueError, match="at least"):
        dmg.derive_gravity(synthetic_rotations([0.0, 1.0, 0.0], count=10))


def test_refuses_when_the_down_direction_is_undetermined():
    """Level cameras carry no information about which end of the axis is down."""
    rotations = synthetic_rotations([0.0, 1.0, 0.0], pitch_deg=0.0)
    # Half the cameras upside down: the mean down axis cancels out.
    flip = np.diag([1.0, -1.0, -1.0])
    rotations[::2] = rotations[::2] @ flip
    with pytest.raises(ValueError, match="which end of the gravity axis"):
        dmg.derive_gravity(rotations)


def test_basis_is_orthonormal_and_matches_the_runtime_reader(tmp_path):
    gravity = np.array([0.2, 0.9, 0.35])
    gravity /= np.linalg.norm(gravity)
    poses = _write_poses(tmp_path, synthetic_rotations(gravity))

    assert dmg.main(["--poses", str(poses), "--frame-name", "unit_test"]) == 0
    written = tmp_path / "T_align_gravity.json"
    frame = rpf.load_map_frame(written)

    assert np.allclose(np.cross(frame.east, frame.north), frame.up, atol=1e-9)
    assert float(np.dot(frame.up, -gravity)) > 0.9999
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["schema"] == "sfm-align/v2"
    assert document["frame_to"] == "unit_test_gravity_aligned_Zup"


def test_check_mode_detects_a_stale_alignment(tmp_path, capsys):
    poses = _write_poses(tmp_path, synthetic_rotations([0.2, 0.9, 0.35]))
    assert dmg.main(["--poses", str(poses)]) == 0
    written = tmp_path / "T_align_gravity.json"
    assert dmg.main(["--poses", str(poses), "--check"]) == 0

    document = json.loads(written.read_text(encoding="utf-8"))
    document["gravity_glomap"] = [0.0, 1.0, 0.0]
    written.write_text(json.dumps(document), encoding="utf-8")

    assert dmg.main(["--poses", str(poses), "--check"]) == 1
    assert "differs from a fresh derivation" in capsys.readouterr().err


@pytest.mark.parametrize("site", sorted(SITE_ALIGNMENTS))
def test_every_site_alignment_matches_a_fresh_derivation(site):
    """A shipped alignment must stay reproducible from the poses it came from."""
    poses_rel, align_rel = SITE_ALIGNMENTS[site]
    poses, align = ROOT / poses_rel, ROOT / align_rel
    if not poses.is_file() or not align.is_file():
        pytest.skip(f"{site} assets not present")

    fresh = dmg.build_document(poses, site, None)
    problems = dmg.compare(json.loads(align.read_text(encoding="utf-8")), fresh)

    assert not problems, f"{site}: {problems}"


@pytest.mark.parametrize("site", sorted(SITE_ALIGNMENTS))
def test_no_site_up_axis_is_the_legacy_guess(site):
    """Every site measured so far disagrees with 'GLOMAP -Y is up'."""
    align = ROOT / SITE_ALIGNMENTS[site][1]
    if not align.is_file():
        pytest.skip(f"{site} alignment not present")

    frame = rpf.load_map_frame(align)
    tilt = math.degrees(math.acos(
        float(np.clip(np.dot(frame.up, rpf.LEGACY_MAP_FRAME.up), -1.0, 1.0))))

    assert tilt > 1.0, (
        f"{site} is within 1 deg of the legacy guess; if a site really is aligned "
        "that closely, say so explicitly rather than letting the guess pass"
    )


def test_check_rejects_a_document_whose_R_contradicts_its_own_gravity(tmp_path):
    """R is what every consumer applies; comparing only gravity_glomap let a
    document publish a rotation it was never derived from."""
    poses = _write_poses(tmp_path, synthetic_rotations(_MEASURED := [0.2, 0.9, 0.35]))
    assert dmg.main(["--poses", str(poses)]) == 0
    written = tmp_path / "T_align_gravity.json"

    document = json.loads(written.read_text(encoding="utf-8"))
    rows = document["R"]
    # Same gravity, different (still orthonormal) rotation: swap the two horizontal
    # axes and flip one so the matrix stays a proper rotation.
    document["R"] = [[-value for value in rows[1]], [-value for value in rows[0]],
                     [-value for value in rows[2]]]
    written.write_text(json.dumps(document), encoding="utf-8")

    assert dmg.main(["--poses", str(poses), "--check"]) == 1


def test_check_rejects_a_non_zero_translation(tmp_path):
    poses = _write_poses(tmp_path, synthetic_rotations([0.2, 0.9, 0.35]))
    assert dmg.main(["--poses", str(poses)]) == 0
    written = tmp_path / "T_align_gravity.json"
    document = json.loads(written.read_text(encoding="utf-8"))
    document["t"] = [0.0, 0.5, 0.0]
    written.write_text(json.dumps(document), encoding="utf-8")

    assert dmg.main(["--poses", str(poses), "--check"]) == 1

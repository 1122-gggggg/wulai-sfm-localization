"""Fast-loop PnP RANSAC must be deterministic (seed=0).

pycolmap defaults ``RANSACOptions.random_seed`` to -1 (nondeterministic).
On P173 that alone swings coverage 3-4pp run-to-run: marginal frames flip
across the >=12 inlier gate on RNG noise, so no paired A/B can measure a
real change. The vendor reloc path (``_solve`` via ``_estimate_pnp``) already
uses seed=0; the fast loop must match.

These tests pin both halves: the synthetic determinism proof, and the
tracker's actual option value read from source (constructing a full
TwoRateTracker needs map assets + GPU models, so we assert on the
construction statement instead of an instance).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pycolmap


def _synthetic_scene() -> tuple[np.ndarray, np.ndarray, pycolmap.Camera]:
    rng = np.random.default_rng(0)
    cam = pycolmap.Camera(
        model="PINHOLE", width=960, height=540, params=[800.0, 800.0, 480.0, 270.0]
    )
    K = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]])
    pts3d = rng.uniform([-5.0, -2.0, 5.0], [5.0, 2.0, 20.0], size=(80, 3))
    proj = pts3d @ K.T
    proj = proj[:, :2] / proj[:, 2:3]
    proj += rng.normal(0.0, 1.0, size=proj.shape)
    # 25% gross outliers so RANSAC actually has to choose.
    proj[60:] = rng.uniform([0.0, 0.0], [960.0, 540.0], size=(20, 2))
    return proj, pts3d, cam


def _solve(proj: np.ndarray, pts3d: np.ndarray, cam: pycolmap.Camera, seed: int) -> tuple:
    opt = pycolmap.AbsolutePoseEstimationOptions()
    opt.ransac.max_error = 4.0
    opt.ransac.min_num_trials = 10
    opt.ransac.max_num_trials = 100
    opt.ransac.confidence = 0.999
    opt.ransac.random_seed = seed
    ref = pycolmap.AbsolutePoseRefinementOptions()
    ref.max_num_iterations = 20
    ans = pycolmap.estimate_and_refine_absolute_pose(proj, pts3d, cam, opt, ref)
    assert ans is not None
    cf = ans["cam_from_world"]
    return (
        np.asarray(cf.rotation.matrix()),
        np.asarray(cf.translation),
        int(ans["num_inliers"]),
        bytes(np.asarray(ans["inlier_mask"]).tobytes()),
    )


def test_seed_zero_is_bit_identical_across_runs() -> None:
    proj, pts3d, cam = _synthetic_scene()
    runs = [_solve(proj, pts3d, cam, seed=0) for _ in range(5)]
    for run in runs[1:]:
        assert run[2] == runs[0][2]
        assert run[3] == runs[0][3]
        assert np.array_equal(run[0], runs[0][0])
        assert np.array_equal(run[1], runs[0][1])


def test_seed_minus_one_is_nondeterministic_on_same_input() -> None:
    """Documents *why* the pin exists: default seed wanders on identical input."""
    proj, pts3d, cam = _synthetic_scene()
    runs = [_solve(proj, pts3d, cam, seed=-1) for _ in range(5)]
    assert len({run[3] for run in runs}) > 1


def test_fast_loop_tracker_pins_ransac_seed_zero() -> None:
    src = (
        pathlib.Path(__file__).resolve().parents[3]
        / "定位演算法"
        / "deploy_code"
        / "sfm_direct_deploy"
        / "two_rate_tracker.py"
    ).read_text(encoding="utf-8")
    assert "estimation.ransac.random_seed = 0" in src

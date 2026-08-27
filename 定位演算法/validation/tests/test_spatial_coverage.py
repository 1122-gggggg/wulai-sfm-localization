from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from analyze_spatial_coverage import (
    PinholeCamera,
    camera_voxel_rows,
    evaluate_pose,
    load_view_calibration,
    minimum_viewing_angle_deg,
    pose_information,
    projection_distribution,
    query_rotation,
    run_analysis,
)
from edm_coverage_sidecar import CoverageRecords, file_sha256, write_coverage_sidecar


def test_query_axes_and_minimum_viewing_angle() -> None:
    rotation = query_rotation(0.0, 0.0)
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0])
    records = CoverageRecords(
        np.array([0, 0], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 2, 3], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int32),
    )
    xyz = np.zeros((2, 3), dtype=float)
    obs_landmark = np.array([0, 0, 1], dtype=np.int32)
    obs_direction = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=float)

    theta = minimum_viewing_angle_deg(
        np.array([1.0, 0.0, 0.0]), xyz, records, obs_landmark, obs_direction
    )

    assert theta == pytest.approx([0.0, 90.0])


def test_projection_distribution_rewards_spread_geometry() -> None:
    clustered = np.array([[5, 5], [6, 5], [5, 6], [6, 6]], dtype=float)
    spread = np.array([[5, 5], [95, 5], [5, 95], [95, 95]], dtype=float)

    clustered_metrics = projection_distribution(clustered, 100, 100)
    spread_metrics = projection_distribution(spread, 100, 100)

    assert spread_metrics["convex_hull_ratio"] > clustered_metrics["convex_hull_ratio"]
    assert spread_metrics["grid4_occupancy"] > clustered_metrics["grid4_occupancy"]
    assert spread_metrics["quadrant_occupancy"] == 4


def test_camera_yaw_bins_wrap_and_empty_voxels_are_retained() -> None:
    angles = np.deg2rad([359.0, 1.0])
    forwards = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(2)))
    rows, outside, shape = camera_voxel_rows(
        np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]),
        forwards,
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 1.0, 1.0]),
        1.0,
        30.0,
    )

    assert outside == 0
    assert shape.tolist() == [2, 1, 1]
    assert rows[0]["yaw_0_count"] == 2
    assert rows[0]["occupied_heading_bins"] == 1
    assert rows[1]["camera_count"] == 0


def test_uncalibrated_pose_keeps_view_support_unavailable() -> None:
    camera = PinholeCamera("PINHOLE", 100, 100, 50.0, 50.0, 50.0, 50.0)

    scalar, _detail = evaluate_pose(
        np.zeros(3),
        0.0,
        0.0,
        np.array([[2.0, 0.0, 0.0]]),
        np.array([10.0]),
        camera,
        None,
    )

    assert scalar["n_frustum_visible"] == 1
    assert scalar["n_view_support"] is None
    assert scalar["expected_matches"] is None
    assert scalar["geometry_population"] == "frustum_visible_upper_bound"


def _project(points: np.ndarray, camera: PinholeCamera) -> np.ndarray:
    return np.column_stack(
        (
            camera.fx * points[:, 0] / points[:, 2] + camera.cx,
            camera.fy * points[:, 1] / points[:, 2] + camera.cy,
        )
    )


def test_pose_information_matches_finite_difference_and_detects_degeneracy() -> None:
    camera = PinholeCamera("PINHOLE", 640, 480, 500.0, 510.0, 320.0, 240.0)
    points = np.array(
        [
            [-1.0, -0.5, 3.0],
            [1.0, -0.4, 3.5],
            [-0.7, 0.8, 4.0],
            [0.9, 0.7, 4.5],
            [0.2, -0.9, 5.0],
            [-0.2, 1.0, 5.5],
        ]
    )
    result = pose_information(points, np.ones(len(points)), camera)
    numeric = np.zeros((len(points), 2, 6), dtype=float)
    epsilon = 1e-6
    for dof in range(6):
        if dof < 3:
            delta = np.zeros(3)
            delta[dof] = epsilon
            plus = points + delta
            minus = points - delta
        else:
            axis = np.zeros(3)
            axis[dof - 3] = epsilon
            plus_rotation = cv2.Rodrigues(axis)[0]
            minus_rotation = cv2.Rodrigues(-axis)[0]
            plus = points @ plus_rotation.T
            minus = points @ minus_rotation.T
        numeric[:, :, dof] = (_project(plus, camera) - _project(minus, camera)) / (
            2.0 * epsilon
        )
    numeric_h = np.einsum("nki,nkj->ij", numeric, numeric)

    assert np.asarray(result["H"]) == pytest.approx(numeric_h, rel=2e-6, abs=2e-5)
    assert result["rank"] == 6
    degenerate = pose_information(
        np.array([[0.0, 0.0, depth] for depth in (2.0, 3.0, 4.0, 5.0)]),
        np.ones(4),
        camera,
    )
    assert degenerate["rank"] < 6
    assert degenerate["covariance_over_sigma2"] is None


def test_view_calibration_validates_and_applies_piecewise_bins(tmp_path: Path) -> None:
    path = tmp_path / "view.json"
    path.write_text(
        json.dumps(
            {
                "schema": "edm-viewpoint-envelope/v1",
                "source": "held-out-validation",
                "sample_count": 100,
                "view_support_min_probability": 0.5,
                "bins": [
                    {"min_deg": 0, "max_deg": 30, "match_survival": 0.9},
                    {"min_deg": 30, "max_deg": 180, "match_survival": 0.2},
                ],
            }
        ),
        encoding="utf-8",
    )
    calibration = load_view_calibration(path)

    assert calibration.probabilities(np.array([0.0, 29.9, 30.0, 180.0])) == pytest.approx(
        [0.9, 0.9, 0.2, 0.2]
    )
    assert calibration.theta_max_deg == 30.0
    raw = json.loads(path.read_text())
    raw["bins"][1]["min_deg"] = 31
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_view_calibration(path)


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    names = ["a.jpg", "b.jpg"]
    nan = np.full((4, 3), np.nan, dtype=np.float32)
    a_xyz = nan.copy()
    b_xyz = nan.copy()
    a_xyz[0] = [2.0, -0.2, 0.0]
    b_xyz[1] = [2.0, 0.2, 0.0]
    bundle = {
        "meta": {"edm_grid_w": 2, "edm_grid_h": 2},
        "ref_names": names,
        "ref_centers": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "refs": {
            "a.jpg": {"xyz_by_cell": a_xyz},
            "b.jpg": {"xyz_by_cell": b_xyz},
        },
    }
    bundle_path = tmp_path / "bundle.pt"
    torch.save(bundle, bundle_path)
    sidecar = tmp_path / "coverage.npz"
    write_coverage_sidecar(
        sidecar,
        CoverageRecords(
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2, 4], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int32),
        ),
        bundle_path=bundle_path,
        ref_names=names,
        cell_count=4,
    )
    rotation = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    poses_path = tmp_path / "poses.json"
    poses_path.write_text(
        json.dumps(
            {
                "coordinate_frame_id": "synthetic-map",
                "ref_names": names,
                "poses": {
                    "a.jpg": {"R": rotation, "t": [0, 0, 0]},
                    "b.jpg": {"R": rotation, "t": [0, 0, -1]},
                },
            }
        ),
        encoding="utf-8",
    )
    alignment = tmp_path / "align.json"
    alignment.write_text(
        json.dumps({"schema": "sfm-align/v2", "R": np.eye(3).tolist()}),
        encoding="utf-8",
    )
    profile = tmp_path / "site_profile.json"
    profile.write_text(
        json.dumps(
            {
                "site_id": "synthetic",
                "coordinate_frame": {"id": "synthetic-map", "units": "map"},
                "assets": {"localization_bundle": bundle_path.name},
                "map_reference_poses": poses_path.name,
                "map_align": alignment.name,
                "query_camera": {
                    "model": "PINHOLE",
                    "width": 100,
                    "height": 100,
                    "params": [50, 50, 50, 50],
                },
                "asset_sha256": {
                    "localization_bundle": file_sha256(bundle_path),
                    "map_reference_poses": file_sha256(poses_path),
                    "map_align": file_sha256(alignment),
                },
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "edm-viewpoint-envelope/v1",
                "source": "synthetic",
                "sample_count": 10,
                "view_support_min_probability": 0.5,
                "bins": [{"min_deg": 0, "max_deg": 180, "match_survival": 0.8}],
            }
        ),
        encoding="utf-8",
    )
    return profile, sidecar, calibration


def test_end_to_end_writes_machine_and_human_reports(tmp_path: Path) -> None:
    profile, sidecar, calibration = _write_synthetic_inputs(tmp_path)
    output = tmp_path / "report"
    args = argparse.Namespace(
        site_profile=str(profile),
        coverage_sidecar=str(sidecar),
        bbox=[0.0, -0.1, -0.1, 0.1, 0.1, 0.1],
        voxel_size=1.0,
        query_spacing=1.0,
        yaw_step_deg=360.0,
        pitch_deg=[0.0],
        view_calibration=str(calibration),
        out_dir=str(output),
    )

    summary = run_analysis(args)

    assert summary["sampling"]["poses"] == 1
    assert (output / "summary.json").is_file()
    assert (output / "query_pose_information.jsonl").is_file()
    assert (output / "heatmaps/query_z000_expected_matches.svg").is_file()
    with (output / "query_poses.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert int(row["n_frustum_visible"]) == 2
    assert int(row["n_view_support"]) == 2
    assert float(row["expected_matches"]) == pytest.approx(1.6)



#!/usr/bin/env python3
"""Offline spatial, viewpoint, projection, and PnP observability coverage audit."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from edm_coverage_sidecar import (
    CoverageRecords,
    file_sha256,
    load_coverage_sidecar,
)


REPORT_SCHEMA = "edm-spatial-coverage-audit/v1"
CALIBRATION_SCHEMA = "edm-viewpoint-envelope/v1"


@dataclass(frozen=True)
class PinholeCamera:
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class SiteInputs:
    source: Path
    site_id: str
    coordinate_frame_id: str
    bundle: Path
    reference_poses: Path
    alignment: Path
    camera: PinholeCamera


@dataclass(frozen=True)
class ViewCalibration:
    minimum_deg: np.ndarray
    maximum_deg: np.ndarray
    probability: np.ndarray
    support_probability: float
    source: str
    sample_count: int

    @property
    def theta_max_deg(self) -> float:
        supported = self.maximum_deg[self.probability >= self.support_probability]
        return float(supported[-1]) if len(supported) else 0.0

    def probabilities(self, theta_deg: np.ndarray) -> np.ndarray:
        theta = np.clip(np.asarray(theta_deg, dtype=np.float64), 0.0, 180.0)
        index = np.searchsorted(self.maximum_deg, theta, side="right")
        index = np.minimum(index, len(self.probability) - 1)
        return self.probability[index]


@dataclass(frozen=True)
class CoverageMap:
    ref_names: tuple[str, ...]
    xyz_aligned: np.ndarray
    ref_centers_aligned: np.ndarray
    ref_forward_aligned: np.ndarray
    records: CoverageRecords
    obs_landmark_idx: np.ndarray
    obs_direction: np.ndarray


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _resolved_asset(base: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"site profile is missing {label}")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"site profile {label} does not exist: {path}")
    return path


def _load_pinhole_camera(raw: object) -> PinholeCamera:
    if not isinstance(raw, dict):
        raise ValueError("site profile query_camera must be an object")
    model = str(raw.get("model", "")).upper()
    width = int(raw.get("width", 0))
    height = int(raw.get("height", 0))
    params = raw.get("params")
    if width <= 0 or height <= 0 or not isinstance(params, list):
        raise ValueError("site profile query_camera dimensions/params are invalid")
    values = [_finite_float(value, "query camera parameter") for value in params]
    if model == "PINHOLE" and len(values) == 4:
        fx, fy, cx, cy = values
    elif model == "SIMPLE_PINHOLE" and len(values) == 3:
        fx, cx, cy = values
        fy = fx
    else:
        raise ValueError(
            "spatial coverage observability supports only PINHOLE/SIMPLE_PINHOLE cameras"
        )
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("query camera focal lengths must be positive")
    return PinholeCamera(model, width, height, fx, fy, cx, cy)


def load_site_inputs(path: str | Path) -> SiteInputs:
    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("site profile root must be an object")
    frame = raw.get("coordinate_frame")
    if not isinstance(frame, dict) or frame.get("units") != "map":
        raise ValueError("coverage analysis requires a scale-free map-unit coordinate frame")
    frame_id = str(frame.get("id", "")).strip()
    if not frame_id:
        raise ValueError("site profile coordinate frame id is required")
    assets = raw.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("site profile assets must be an object")
    base = source.parent
    bundle = _resolved_asset(base, assets.get("localization_bundle"), "localization_bundle")
    poses = _resolved_asset(base, raw.get("map_reference_poses"), "map_reference_poses")
    alignment = _resolved_asset(base, raw.get("map_align"), "map_align")

    digests = raw.get("asset_sha256")
    if isinstance(digests, dict):
        for label, asset in (
            ("localization_bundle", bundle),
            ("map_reference_poses", poses),
            ("map_align", alignment),
        ):
            expected = digests.get(label)
            if expected is not None and str(expected).lower() != file_sha256(asset):
                raise ValueError(f"site profile {label} SHA-256 mismatch")
    return SiteInputs(
        source=source,
        site_id=str(raw.get("site_id", "")).strip(),
        coordinate_frame_id=frame_id,
        bundle=bundle,
        reference_poses=poses,
        alignment=alignment,
        camera=_load_pinhole_camera(raw.get("query_camera")),
    )


def load_alignment(path: str | Path) -> np.ndarray:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "sfm-align/v2":
        raise ValueError("coverage analysis requires an sfm-align/v2 alignment")
    rotation = np.asarray(raw.get("R"), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("alignment rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError("alignment rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("alignment rotation must be right-handed")
    return rotation


def load_view_calibration(path: str | Path) -> ViewCalibration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported viewpoint calibration schema")
    bins = raw.get("bins")
    if not isinstance(bins, list) or not bins:
        raise ValueError("viewpoint calibration bins must be a non-empty list")
    minimum: list[float] = []
    maximum: list[float] = []
    probability: list[float] = []
    for index, item in enumerate(bins):
        if not isinstance(item, dict):
            raise ValueError("viewpoint calibration bin must be an object")
        low = _finite_float(item.get("min_deg"), f"bin[{index}].min_deg")
        high = _finite_float(item.get("max_deg"), f"bin[{index}].max_deg")
        chance = _finite_float(
            item.get("match_survival"), f"bin[{index}].match_survival"
        )
        if high <= low or not 0.0 <= chance <= 1.0:
            raise ValueError("viewpoint calibration bin range/probability is invalid")
        minimum.append(low)
        maximum.append(high)
        probability.append(chance)
    if not math.isclose(minimum[0], 0.0, abs_tol=1e-9) or not math.isclose(
        maximum[-1], 180.0, abs_tol=1e-9
    ):
        raise ValueError("viewpoint calibration bins must cover 0 through 180 degrees")
    if any(
        not math.isclose(maximum[index - 1], minimum[index], abs_tol=1e-9)
        for index in range(1, len(minimum))
    ):
        raise ValueError("viewpoint calibration bins must be contiguous")
    if np.any(np.diff(np.asarray(probability, dtype=np.float64)) > 1e-12):
        raise ValueError("viewpoint match survival must be monotonically non-increasing")
    threshold = _finite_float(
        raw.get("view_support_min_probability"), "view_support_min_probability"
    )
    if not 0.0 < threshold <= 1.0:
        raise ValueError("view support probability must be in (0, 1]")
    source = str(raw.get("source", "")).strip()
    sample_count = raw.get("sample_count")
    if not source or isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("viewpoint calibration source and integer sample_count are required")
    if sample_count <= 0:
        raise ValueError("viewpoint calibration sample_count must be positive")
    return ViewCalibration(
        np.asarray(minimum, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
        np.asarray(probability, dtype=np.float64),
        threshold,
        source,
        sample_count,
    )


def _load_reference_geometry(
    path: Path,
    ref_names: Sequence[str],
    bundle_centers: np.ndarray,
    alignment: np.ndarray,
    coordinate_frame_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("poses"), dict):
        raise ValueError("reference poses file is invalid")
    declared_frame = raw.get("coordinate_frame_id")
    if declared_frame not in (None, coordinate_frame_id):
        raise ValueError("reference poses coordinate frame does not match the site profile")
    declared_names = raw.get("ref_names")
    if declared_names is not None and list(declared_names) != list(ref_names):
        raise ValueError("reference poses ordered names do not match the bundle")
    centers: list[np.ndarray] = []
    forwards: list[np.ndarray] = []
    for name in ref_names:
        pose = raw["poses"].get(name)
        if not isinstance(pose, dict):
            raise ValueError(f"reference poses file is missing {name}")
        rotation = np.asarray(pose.get("R"), dtype=np.float64)
        translation = np.asarray(pose.get("t"), dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(f"reference pose {name} has an invalid shape")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5
        ):
            raise ValueError(f"reference pose {name} rotation is not proper")
        centers.append(-rotation.T @ translation)
        forwards.append(rotation.T @ np.array([0.0, 0.0, 1.0]))
    centers_raw = np.asarray(centers, dtype=np.float64)
    if bundle_centers.shape != centers_raw.shape or not np.allclose(
        bundle_centers, centers_raw, atol=1e-5
    ):
        raise ValueError("reference pose centers do not match the production bundle")
    return centers_raw @ alignment.T, np.asarray(forwards) @ alignment.T


def _bundle_anchor_pairs(bundle: dict, ref_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    anchor_ref: list[np.ndarray] = []
    anchor_cell: list[np.ndarray] = []
    for ref_idx, name in enumerate(ref_names):
        xyz = np.asarray(bundle["refs"][name]["xyz_by_cell"], dtype=np.float32)
        valid = np.flatnonzero(np.isfinite(xyz).all(axis=1)).astype(np.int32)
        anchor_ref.append(np.full(len(valid), ref_idx, dtype=np.int32))
        anchor_cell.append(valid)
    return np.concatenate(anchor_ref), np.concatenate(anchor_cell)


def load_coverage_map(inputs: SiteInputs, sidecar_path: str | Path) -> CoverageMap:
    bundle = torch.load(inputs.bundle, map_location="cpu", weights_only=False)
    ref_names = tuple(str(value) for value in bundle["ref_names"])
    grid_w = int(bundle["meta"]["edm_grid_w"])
    grid_h = int(bundle["meta"]["edm_grid_h"])
    cell_count = grid_w * grid_h
    records = load_coverage_sidecar(
        sidecar_path,
        bundle_path=inputs.bundle,
        ref_names=ref_names,
        cell_count=cell_count,
    )
    expected_ref, expected_cell = _bundle_anchor_pairs(bundle, ref_names)
    if not np.array_equal(records.anchor_ref_idx, expected_ref) or not np.array_equal(
        records.anchor_cell_idx, expected_cell
    ):
        raise ValueError("coverage sidecar does not cover every finite production EDM anchor")

    xyz_raw = np.empty((len(expected_ref), 3), dtype=np.float32)
    for ref_idx in np.unique(expected_ref):
        mask = expected_ref == ref_idx
        name = ref_names[int(ref_idx)]
        xyz_raw[mask] = np.asarray(bundle["refs"][name]["xyz_by_cell"], dtype=np.float32)[
            expected_cell[mask]
        ]
    alignment = load_alignment(inputs.alignment)
    bundle_centers = np.asarray(bundle.get("ref_centers"), dtype=np.float64)
    centers_aligned, forward_aligned = _load_reference_geometry(
        inputs.reference_poses,
        ref_names,
        bundle_centers,
        alignment,
        inputs.coordinate_frame_id,
    )
    xyz_aligned = xyz_raw.astype(np.float64) @ alignment.T
    observation_count = np.diff(records.obs_offsets)
    obs_landmark_idx = np.repeat(
        np.arange(len(records.anchor_ref_idx), dtype=np.int32), observation_count
    )
    obs_delta = centers_aligned[records.obs_ref_idx] - xyz_aligned[obs_landmark_idx]
    obs_norm = np.linalg.norm(obs_delta, axis=1)
    if np.any(~np.isfinite(obs_norm) | (obs_norm <= 1e-9)):
        raise ValueError("coverage sidecar contains a zero-length observation direction")
    obs_direction = (obs_delta / obs_norm[:, None]).astype(np.float32)
    return CoverageMap(
        ref_names,
        xyz_aligned,
        centers_aligned,
        forward_aligned,
        records,
        obs_landmark_idx,
        obs_direction,
    )


def axis_values(low: float, high: float, spacing: float) -> np.ndarray:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("bbox maximum must be greater than its minimum")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("grid spacing must be finite and positive")
    count = int(math.floor((high - low) / spacing + 1e-10)) + 1
    return low + np.arange(count, dtype=np.float64) * spacing


def query_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    forward = np.array(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    )
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    down = np.cross(forward, right)
    return np.stack((right, down, forward), axis=0)


def minimum_viewing_angle_deg(
    position: np.ndarray,
    xyz: np.ndarray,
    records: CoverageRecords,
    obs_landmark_idx: np.ndarray,
    obs_direction: np.ndarray,
) -> np.ndarray:
    query_delta = np.asarray(position, dtype=np.float64)[None, :] - xyz
    norm = np.linalg.norm(query_delta, axis=1)
    query_direction = np.zeros_like(query_delta)
    valid = norm > 1e-9
    query_direction[valid] = query_delta[valid] / norm[valid, None]
    dot = np.einsum(
        "ij,ij->i",
        query_direction[obs_landmark_idx],
        obs_direction,
        optimize=True,
    )
    maximum_dot = np.maximum.reduceat(dot, records.obs_offsets[:-1])
    return np.degrees(np.arccos(np.clip(maximum_dot, -1.0, 1.0)))


def project_frustum(
    landmark_delta: np.ndarray,
    rotation_camera_from_aligned: np.ndarray,
    camera: PinholeCamera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_xyz = landmark_delta @ rotation_camera_from_aligned.T
    positive = np.flatnonzero(np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 1e-8))
    if not len(positive):
        return positive, np.zeros((0, 2)), np.zeros((0, 3))
    points = camera_xyz[positive]
    u = camera.fx * points[:, 0] / points[:, 2] + camera.cx
    v = camera.fy * points[:, 1] / points[:, 2] + camera.cy
    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < camera.width)
        & (v >= 0.0)
        & (v < camera.height)
    )
    return positive[inside], np.column_stack((u[inside], v[inside])), points[inside]


def projection_distribution(points: np.ndarray, width: int, height: int) -> dict[str, float | int]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = len(points)
    hull_ratio = 0.0
    if count >= 3:
        hull = cv2.convexHull(points.astype(np.float32))
        hull_ratio = float(cv2.contourArea(hull) / max(float(width * height), 1.0))
    if count:
        gx = np.clip((points[:, 0] * 4.0 / width).astype(int), 0, 3)
        gy = np.clip((points[:, 1] * 4.0 / height).astype(int), 0, 3)
        grid = int(len(np.unique(gy * 4 + gx)))
        qx = points[:, 0] >= width / 2.0
        qy = points[:, 1] >= height / 2.0
        quadrants = int(len(np.unique(qy.astype(int) * 2 + qx.astype(int))))
        center = (
            (points[:, 0] >= width * 0.25)
            & (points[:, 0] <= width * 0.75)
            & (points[:, 1] >= height * 0.25)
            & (points[:, 1] <= height * 0.75)
        )
        edge = (
            (points[:, 0] < width * 0.10)
            | (points[:, 0] >= width * 0.90)
            | (points[:, 1] < height * 0.10)
            | (points[:, 1] >= height * 0.90)
        )
        center_fraction = float(np.mean(center))
        edge_fraction = float(np.mean(edge))
    else:
        grid = quadrants = 0
        center_fraction = edge_fraction = 0.0
    return {
        "convex_hull_ratio": hull_ratio,
        "grid4_occupancy": grid,
        "grid4_occupancy_ratio": grid / 16.0,
        "quadrant_occupancy": quadrants,
        "center_fraction": center_fraction,
        "edge_fraction": edge_fraction,
    }


def pose_information(
    camera_xyz: np.ndarray,
    weights: np.ndarray,
    camera: PinholeCamera,
    *,
    batch_size: int = 100_000,
) -> dict[str, object]:
    points = np.asarray(camera_xyz, dtype=np.float64).reshape(-1, 3)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(points) != len(weight) or np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("pose information weights are invalid")
    information = np.zeros((6, 6), dtype=np.float64)
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        w = weight[start : start + batch_size]
        x, y, z = batch.T
        a = camera.fx / z
        b = camera.fy / z
        c = -camera.fx * x / (z * z)
        d = -camera.fy * y / (z * z)
        j0 = np.column_stack((a, np.zeros_like(a), c, c * y, a * z - c * x, -a * y))
        j1 = np.column_stack((np.zeros_like(b), b, d, -b * z + d * y, -d * x, b * x))
        information += np.einsum("ni,nj,n->ij", j0, j0, w, optimize=True)
        information += np.einsum("ni,nj,n->ij", j1, j1, w, optimize=True)
    information = (information + information.T) * 0.5
    total_weight = float(np.sum(weight))
    normalized = information / total_weight if total_weight > 0.0 else information.copy()
    eigenvalues = np.linalg.eigvalsh(information)
    normalized_eigenvalues = np.linalg.eigvalsh(normalized)
    largest = max(float(eigenvalues[-1]), 0.0)
    tolerance = max(largest * 1e-10, 1e-12)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    covariance = None
    condition = None
    if rank == 6:
        covariance_array = np.linalg.inv(information)
        covariance = covariance_array.tolist()
        condition = float(eigenvalues[-1] / eigenvalues[0])
    return {
        "H": information.tolist(),
        "H_normalized": normalized.tolist(),
        "eigenvalues": [max(float(value), 0.0) for value in eigenvalues],
        "normalized_eigenvalues": [max(float(value), 0.0) for value in normalized_eigenvalues],
        "rank": rank,
        "condition": condition,
        "covariance_over_sigma2": covariance,
        "total_weight": total_weight,
    }


def _optional_percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if len(values) else None


def evaluate_pose(
    position: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    landmark_delta: np.ndarray,
    theta_min: np.ndarray,
    camera: PinholeCamera,
    calibration: ViewCalibration | None,
) -> tuple[dict[str, object], dict[str, object]]:
    rotation = query_rotation(yaw_deg, pitch_deg)
    visible, projected, camera_xyz = project_frustum(landmark_delta, rotation, camera)
    theta = theta_min[visible]
    if calibration is None:
        probabilities = np.ones(len(visible), dtype=np.float64)
        support_mask = np.ones(len(visible), dtype=bool)
        n_view_support: int | None = None
        expected_matches: float | None = None
        population = "frustum_visible_upper_bound"
    else:
        probabilities = calibration.probabilities(theta)
        support_mask = probabilities >= calibration.support_probability
        n_view_support = int(np.count_nonzero(support_mask))
        expected_matches = float(np.sum(probabilities))
        population = "calibrated_view_supported"
    distribution = projection_distribution(projected[support_mask], camera.width, camera.height)
    information = pose_information(camera_xyz, probabilities, camera)
    eigen = information["eigenvalues"]
    normalized_eigen = information["normalized_eigenvalues"]
    covariance = information["covariance_over_sigma2"]
    covariance_max = None
    covariance_diag: list[float] | None = None
    if covariance is not None:
        covariance_array = np.asarray(covariance, dtype=np.float64)
        covariance_diag = [float(value) for value in np.diag(covariance_array)]
        covariance_max = float(np.max(np.diag(covariance_array)))
    scalar = {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "n_frustum_visible": int(len(visible)),
        "theta_p50_deg": _optional_percentile(theta, 50),
        "theta_p90_deg": _optional_percentile(theta, 90),
        "theta_p95_deg": _optional_percentile(theta, 95),
        "theta_max_deg": float(np.max(theta)) if len(theta) else None,
        "n_view_support": n_view_support,
        "expected_matches": expected_matches,
        "geometry_population": population,
        **distribution,
        "information_rank": int(information["rank"]),
        "lambda_min": float(eigen[0]),
        "lambda_max": float(eigen[-1]),
        "lambda_min_normalized": float(normalized_eigen[0]),
        "information_condition": information["condition"],
        "covariance_max_diag_over_sigma2": covariance_max,
        "covariance_diag_over_sigma2": covariance_diag,
    }
    detail = {
        "x": scalar["x"],
        "y": scalar["y"],
        "z": scalar["z"],
        "yaw_deg": scalar["yaw_deg"],
        "pitch_deg": scalar["pitch_deg"],
        "H": information["H"],
        "H_normalized": information["H_normalized"],
        "covariance_over_sigma2": covariance,
    }
    return scalar, detail


def _voxel_shape(bounds_min: np.ndarray, bounds_max: np.ndarray, size: float) -> np.ndarray:
    return np.maximum(1, np.ceil((bounds_max - bounds_min) / size).astype(int))


def camera_voxel_rows(
    centers: np.ndarray,
    forwards: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    voxel_size: float,
    yaw_step_deg: float,
) -> tuple[list[dict[str, object]], int, np.ndarray]:
    shape = _voxel_shape(bounds_min, bounds_max, voxel_size)
    bin_count = int(round(360.0 / yaw_step_deg))
    counts: dict[tuple[int, int, int], np.ndarray] = {}
    outside = 0
    for center, forward in zip(centers, forwards):
        if np.any(center < bounds_min) or np.any(center > bounds_max):
            outside += 1
            continue
        horizontal = math.hypot(float(forward[0]), float(forward[1]))
        if horizontal <= 1e-9:
            raise ValueError("reference camera forward direction has undefined yaw")
        index = np.floor((center - bounds_min) / voxel_size).astype(int)
        index = np.minimum(index, shape - 1)
        yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0]))) % 360.0
        yaw_bin = int(math.floor((yaw + yaw_step_deg / 2.0) / yaw_step_deg)) % bin_count
        key = tuple(int(value) for value in index)
        counts.setdefault(key, np.zeros(bin_count, dtype=np.int64))[yaw_bin] += 1

    rows: list[dict[str, object]] = []
    for iz in range(shape[2]):
        for iy in range(shape[1]):
            for ix in range(shape[0]):
                values = counts.get((ix, iy, iz), np.zeros(bin_count, dtype=np.int64))
                occupied = np.flatnonzero(values > 0)
                if len(occupied):
                    angles = occupied.astype(float) * yaw_step_deg
                    gaps = np.diff(np.r_[angles, angles[0] + 360.0])
                    maximum_gap: float | None = float(np.max(gaps))
                else:
                    maximum_gap = None
                lower = bounds_min + np.array([ix, iy, iz]) * voxel_size
                upper = np.minimum(lower + voxel_size, bounds_max)
                row: dict[str, object] = {
                    "ix": ix,
                    "iy": iy,
                    "iz": iz,
                    "x": float((lower[0] + upper[0]) * 0.5),
                    "y": float((lower[1] + upper[1]) * 0.5),
                    "z": float((lower[2] + upper[2]) * 0.5),
                    "camera_count": int(np.sum(values)),
                    "occupied_heading_bins": int(len(occupied)),
                    "max_heading_gap_deg": maximum_gap,
                }
                for bin_index, value in enumerate(values):
                    row[f"yaw_{int(round(bin_index * yaw_step_deg))}_count"] = int(value)
                rows.append(row)
    return rows, outside, shape


def _numeric_values(rows: Sequence[dict[str, object]], key: str) -> np.ndarray:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return np.asarray(values, dtype=np.float64)


def summarize_position(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    higher_is_better = (
        "n_frustum_visible",
        "n_view_support",
        "expected_matches",
        "convex_hull_ratio",
        "grid4_occupancy_ratio",
        "quadrant_occupancy",
        "information_rank",
        "lambda_min_normalized",
    )
    lower_is_better = (
        "theta_p90_deg",
        "information_condition",
        "covariance_max_diag_over_sigma2",
    )
    for key in higher_is_better + lower_is_better:
        values = _numeric_values(rows, key)
        result[f"median_{key}"] = float(np.median(values)) if len(values) else None
        if len(values):
            result[f"worst_{key}"] = (
                float(np.min(values)) if key in higher_is_better else float(np.max(values))
            )
        else:
            result[f"worst_{key}"] = None
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _color(value: float | None, low: float, high: float, higher_is_better: bool) -> str:
    if value is None or not math.isfinite(value):
        return "#bdbdbd"
    ratio = 0.5 if high <= low else (value - low) / (high - low)
    ratio = min(max(ratio, 0.0), 1.0)
    if not higher_is_better:
        ratio = 1.0 - ratio
    red = int(round(190 * (1.0 - ratio) + 33 * ratio))
    green = int(round(45 * (1.0 - ratio) + 145 * ratio))
    blue = int(round(45 * (1.0 - ratio) + 140 * ratio))
    return f"#{red:02x}{green:02x}{blue:02x}"


def write_heatmap_svg(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    nx: int,
    ny: int,
    keys: Sequence[str],
    labels: Sequence[str],
    title: str,
    higher_is_better: bool,
) -> None:
    values = [
        float(row[key])
        for row in rows
        for key in keys
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    low = min(values) if values else 0.0
    high = max(values) if values else 1.0
    cell = max(5, min(24, int(720 / max(nx, ny, 1))))
    panel_w = nx * cell
    panel_h = ny * cell
    gap = 70
    margin_x = 45
    margin_y = 70
    width = margin_x * 2 + len(keys) * panel_w + (len(keys) - 1) * gap
    height = margin_y + panel_h + 55
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-size="16">{html.escape(title)}</text>',
    ]
    by_index = {(int(row["ix"]), int(row["iy"])): row for row in rows}
    for panel, (key, label) in enumerate(zip(keys, labels)):
        origin_x = margin_x + panel * (panel_w + gap)
        chunks.append(
            f'<text x="{origin_x + panel_w / 2}" y="50" text-anchor="middle" '
            f'font-size="13">{html.escape(label)}</text>'
        )
        for iy in range(ny):
            for ix in range(nx):
                row = by_index[(ix, iy)]
                raw = row.get(key)
                value = None if raw is None else float(raw)
                x = origin_x + ix * cell
                y = margin_y + (ny - 1 - iy) * cell
                chunks.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{_color(value, low, high, higher_is_better)}" stroke="#ffffff" '
                    'stroke-width="0.4"/>'
                )
    chunks.extend(
        [
            f'<text x="{margin_x}" y="{height - 15}" font-size="11">low={low:.6g}</text>',
            f'<text x="{width - margin_x}" y="{height - 15}" text-anchor="end" '
            f'font-size="11">high={high:.6g}</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def _clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    inputs = load_site_inputs(args.site_profile)
    coverage = load_coverage_map(inputs, args.coverage_sidecar)
    calibration = (
        load_view_calibration(args.view_calibration) if args.view_calibration else None
    )
    bounds = np.asarray(args.bbox, dtype=np.float64)
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("bbox must contain six finite values")
    bounds_min, bounds_max = bounds[:3], bounds[3:]
    if np.any(bounds_max <= bounds_min):
        raise ValueError("bbox maximum must be greater than its minimum")
    voxel_size = _finite_float(args.voxel_size, "voxel_size")
    query_spacing = _finite_float(args.query_spacing, "query_spacing")
    yaw_step = _finite_float(args.yaw_step_deg, "yaw_step_deg")
    if voxel_size <= 0.0 or query_spacing <= 0.0 or yaw_step <= 0.0:
        raise ValueError("coverage sizes and yaw step must be positive")
    yaw_count = 360.0 / yaw_step
    if not math.isclose(yaw_count, round(yaw_count), abs_tol=1e-9):
        raise ValueError("yaw_step_deg must divide 360 exactly")
    pitches = [_finite_float(value, "pitch angle") for value in args.pitch_deg]
    if not pitches or any(abs(value) >= 90.0 for value in pitches):
        raise ValueError("pitch angles must be non-empty and strictly within (-90, 90)")

    out = Path(args.out_dir).expanduser().resolve()
    heatmap_dir = out / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    camera_rows, outside_camera_count, voxel_shape = camera_voxel_rows(
        coverage.ref_centers_aligned,
        coverage.ref_forward_aligned,
        bounds_min,
        bounds_max,
        voxel_size,
        yaw_step,
    )
    _write_csv(out / "camera_voxels.csv", camera_rows)

    x_values = axis_values(bounds_min[0], bounds_max[0], query_spacing)
    y_values = axis_values(bounds_min[1], bounds_max[1], query_spacing)
    z_values = axis_values(bounds_min[2], bounds_max[2], query_spacing)
    yaw_values = np.arange(int(round(yaw_count)), dtype=float) * yaw_step
    position_rows: list[dict[str, object]] = []
    pose_fields: list[str] | None = None
    pose_count = 0
    query_csv = out / "query_poses.csv"
    information_jsonl = out / "query_pose_information.jsonl"
    with query_csv.open("w", encoding="utf-8", newline="") as pose_stream, information_jsonl.open(
        "w", encoding="utf-8"
    ) as information_stream:
        writer = None
        position_id = 0
        for iz, z in enumerate(z_values):
            for iy, y in enumerate(y_values):
                for ix, x in enumerate(x_values):
                    position = np.array([x, y, z], dtype=np.float64)
                    landmark_delta = coverage.xyz_aligned - position[None, :]
                    theta_min = minimum_viewing_angle_deg(
                        position,
                        coverage.xyz_aligned,
                        coverage.records,
                        coverage.obs_landmark_idx,
                        coverage.obs_direction,
                    )
                    rows_at_position: list[dict[str, object]] = []
                    for pitch in pitches:
                        for yaw in yaw_values:
                            scalar, detail = evaluate_pose(
                                position,
                                float(yaw),
                                pitch,
                                landmark_delta,
                                theta_min,
                                inputs.camera,
                                calibration,
                            )
                            scalar = {
                                "pose_id": pose_count,
                                "position_id": position_id,
                                "ix": ix,
                                "iy": iy,
                                "iz": iz,
                                **scalar,
                            }
                            detail = {"pose_id": pose_count, **detail}
                            if writer is None:
                                pose_fields = list(scalar)
                                writer = csv.DictWriter(pose_stream, fieldnames=pose_fields)
                                writer.writeheader()
                            writer.writerow(scalar)
                            information_stream.write(
                                json.dumps(_clean_json(detail), separators=(",", ":")) + "\n"
                            )
                            rows_at_position.append(scalar)
                            pose_count += 1
                    position_rows.append(
                        {
                            "position_id": position_id,
                            "ix": ix,
                            "iy": iy,
                            "iz": iz,
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            **summarize_position(rows_at_position),
                        }
                    )
                    position_id += 1
    if pose_fields is None:
        raise RuntimeError("query grid generated no hypothetical poses")
    _write_csv(out / "position_summary.csv", position_rows)

    heatmaps: list[str] = []
    query_metrics = [
        ("n_frustum_visible", True),
        ("convex_hull_ratio", True),
        ("grid4_occupancy_ratio", True),
        ("lambda_min_normalized", True),
        ("information_condition", False),
    ]
    if calibration is not None:
        query_metrics[1:1] = [("n_view_support", True), ("expected_matches", True)]
    for iz, z in enumerate(z_values):
        slice_rows = [row for row in position_rows if int(row["iz"]) == iz]
        for metric, higher in query_metrics:
            path = heatmap_dir / f"query_z{iz:03d}_{metric}.svg"
            write_heatmap_svg(
                path,
                slice_rows,
                nx=len(x_values),
                ny=len(y_values),
                keys=(f"worst_{metric}", f"median_{metric}"),
                labels=("worst orientation", "median orientation"),
                title=f"{metric}, aligned z={z:.6g} map units",
                higher_is_better=higher,
            )
            heatmaps.append(str(path.relative_to(out)))
    for iz in range(int(voxel_shape[2])):
        slice_rows = [row for row in camera_rows if int(row["iz"]) == iz]
        for metric in ("camera_count", "occupied_heading_bins"):
            path = heatmap_dir / f"camera_z{iz:03d}_{metric}.svg"
            write_heatmap_svg(
                path,
                slice_rows,
                nx=int(voxel_shape[0]),
                ny=int(voxel_shape[1]),
                keys=(metric,),
                labels=(metric,),
                title=f"reference {metric}, voxel z-index={iz}",
                higher_is_better=True,
            )
            heatmaps.append(str(path.relative_to(out)))

    calibration_summary = None
    if calibration is not None:
        calibration_summary = {
            "source_path": str(Path(args.view_calibration).resolve()),
            "source": calibration.source,
            "sample_count": calibration.sample_count,
            "view_support_min_probability": calibration.support_probability,
            "theta_max_deg": calibration.theta_max_deg,
        }
    summary: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "site_id": inputs.site_id,
        "coordinate_frame_id": inputs.coordinate_frame_id,
        "units": "map",
        "inputs": {
            "site_profile": str(inputs.source),
            "site_profile_sha256": file_sha256(inputs.source),
            "bundle": str(inputs.bundle),
            "bundle_sha256": file_sha256(inputs.bundle),
            "reference_poses": str(inputs.reference_poses),
            "reference_poses_sha256": file_sha256(inputs.reference_poses),
            "alignment": str(inputs.alignment),
            "alignment_sha256": file_sha256(inputs.alignment),
            "coverage_sidecar": str(Path(args.coverage_sidecar).resolve()),
            "coverage_sidecar_sha256": file_sha256(args.coverage_sidecar),
        },
        "sampling": {
            "bbox": bounds.tolist(),
            "voxel_size": voxel_size,
            "query_spacing": query_spacing,
            "yaw_step_deg": yaw_step,
            "pitch_deg": pitches,
            "positions": len(position_rows),
            "poses": pose_count,
        },
        "map": {
            "references": len(coverage.ref_names),
            "production_edm_anchors": len(coverage.xyz_aligned),
            "anchor_observations": len(coverage.records.obs_ref_idx),
            "reference_cameras_outside_bbox": outside_camera_count,
        },
        "view_calibration": calibration_summary,
        "heatmaps": heatmaps,
        "limitations": [
            "frustum_visible checks positive depth and image bounds only; it does not model occlusion",
            "distances are scale-free map units, not metres",
            "coverage is a global geometric upper bound and does not model retrieval top-k",
            "appearance, repeated structure, wrong correspondences, and PnP/RANSAC failures are not simulated",
            "covariance_over_sigma2 is normalized to one-pixel observation variance",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(_clean_json(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    calibration_note = (
        f"Calibrated viewpoint support: {calibration.source}, "
        f"theta_max={calibration.theta_max_deg:.3g} deg."
        if calibration is not None
        else "No viewpoint calibration was supplied; view-support and expected-match maps are unavailable."
    )
    report = f"""# Production EDM spatial coverage audit

- Site: `{inputs.site_id}`
- Frame: `{inputs.coordinate_frame_id}`
- Units: map units (not metres)
- Reference cameras: {len(coverage.ref_names)}
- Production EDM anchors: {len(coverage.xyz_aligned)}
- Hypothetical positions / poses: {len(position_rows)} / {pose_count}
- {calibration_note}

## Outputs

- `camera_voxels.csv`: per-voxel camera and yaw-bin counts
- `query_poses.csv`: per-pose coverage metrics
- `query_pose_information.jsonl`: full information/covariance matrices
- `position_summary.csv`: worst and median orientation summaries
- `heatmaps/`: aligned-height SVG coverage slices

## Interpretation limits

This is a frustum-only, global geometric coverage audit. It does not claim occlusion-aware
visibility or end-to-end localization probability, and it does not simulate retrieval,
appearance change, repeated structure, correspondence errors, or PnP/RANSAC failures.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-profile",
        required=True,
        help="site profile providing the production bundle, poses, alignment, and camera",
    )
    parser.add_argument(
        "--coverage-sidecar",
        required=True,
        help="exact edm-coverage-observations/v1 sidecar produced during EDM map build",
    )
    parser.add_argument(
        "--bbox",
        nargs=6,
        type=float,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        required=True,
        help="analysis bounds in gravity-aligned Z-up map units",
    )
    parser.add_argument(
        "--voxel-size", type=float, required=True, help="reference-camera voxel edge in map units"
    )
    parser.add_argument(
        "--query-spacing", type=float, required=True, help="hypothetical-query lattice spacing"
    )
    parser.add_argument(
        "--yaw-step-deg", type=float, default=30.0, help="yaw step; must divide 360 exactly"
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        nargs="+",
        default=[-30, -15, 0, 15, 30],
        help="hypothetical camera pitch angles; positive points upward",
    )
    parser.add_argument(
        "--view-calibration",
        help="optional edm-viewpoint-envelope/v1 JSON; no threshold is assumed without it",
    )
    parser.add_argument("--out-dir", required=True, help="directory for JSON/CSV/JSONL/SVG reports")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_analysis(args)
    print(json.dumps(_clean_json(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

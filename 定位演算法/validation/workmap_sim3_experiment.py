#!/usr/bin/env python3
"""Offline, fixed-main-map Sim3 closure diagnostics; never applies corrections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "定位演算法" / "deploy_code_expand" / "sfm_direct_deploy"))
from workmap_submap_graph import edge_residual  # noqa: E402
from workmap_submaps import AnchorPair, Sim3, anchor_consistency, fit_anchor_similarity  # noqa: E402


def pose(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        if not np.allclose(array[3], [0, 0, 0, 1]):
            raise ValueError("invalid homogeneous pose")
        array = array[:3]
    if array.shape != (3, 4) or not np.isfinite(array).all():
        raise ValueError("pose must be finite cam_from_world 3x4 or 4x4")
    rotation = array[:, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("pose rotation is not SO(3)")
    return array.copy()


def camera_sim(value: np.ndarray) -> Sim3:
    rotation = value[:, :3].T
    return Sim3(1.0, rotation, -rotation @ value[:, 3])


def camera_pose(value: Sim3) -> np.ndarray:
    rotation = value.rotation.T
    return np.column_stack([rotation, -rotation @ value.translation])


def make_anchor(record: dict) -> AnchorPair:
    if record["local_inliers"] < 30 or record["main_inliers"] < 80:
        raise ValueError("anchor does not meet independent local/main inlier gates")
    local_xy = np.asarray(record["local_xy"], dtype=np.float64).reshape(-1, 2)
    local_xyz = np.asarray(record["local_xyz"], dtype=np.float64).reshape(-1, 3)
    if len(local_xy) != len(local_xyz) or len(local_xy) < 30:
        raise ValueError("anchor is missing raw local inlier correspondences")
    ids = np.asarray(record["local_point_ids"])
    if (
        ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != len(local_xy)
        or (ids >= 0).any() or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("local anchor must retain unique measured negative landmark IDs")
    return AnchorPair(
        int(record["ordinal"]), pose(record["map_pose"]), pose(record["local_pose"]),
        ids.astype(np.int64, copy=False), local_xy,
    )


def error_summary(records: list[dict], predicted: list[np.ndarray]) -> dict:
    centers, rotations, reprojections = [], [], []
    for record, estimate in zip(records, predicted, strict=True):
        measured = pose(record["map_pose"])
        centers.append(float(np.linalg.norm(camera_sim(estimate).translation - camera_sim(measured).translation)))
        cosine = np.clip((np.trace(estimate[:, :3] @ measured[:, :3].T) - 1) * 0.5, -1, 1)
        rotations.append(float(np.degrees(np.arccos(cosine))))
        xyz = np.asarray(record["main_xyz"], dtype=np.float64).reshape(-1, 3)
        xy = np.asarray(record["main_xy"], dtype=np.float64).reshape(-1, 2)
        intrinsic = np.asarray(record["K"], dtype=np.float64).reshape(3, 3)
        if len(xyz) != len(xy) or len(xyz) < 80:
            raise ValueError("anchor is missing raw frozen-main inlier correspondences")
        projected = (xyz @ estimate[:, :3].T + estimate[:, 3]) @ intrinsic.T
        if (projected[:, 2] <= 0).any():
            return {"ok": False, "reason": "behind_camera"}
        errors = np.linalg.norm(projected[:, :2] / projected[:, 2:] - xy, axis=1)
        if not np.isfinite(errors).all():
            raise ValueError("nonfinite held-out reprojection")
        reprojections.extend(errors.tolist())
    return {
        "ok": True, "count": len(records),
        "center_median_u": float(np.median(centers)),
        "center_max_u": float(np.max(centers)),
        "rotation_median_deg": float(np.median(rotations)),
        "rotation_max_deg": float(np.max(rotations)),
        "main_reprojection_median_px": float(np.median(reprojections)),
        "main_reprojection_p90_px": float(np.percentile(reprojections, 90)),
    }


def fit_window(records: list[dict]) -> dict:
    kinds = {record.get("submap_kind", "detached") for record in records}
    if len(kinds) != 1 or not kinds <= {"detached", "independent"}:
        raise ValueError("one window must have one recognized submap coordinate kind")
    # Independent monocular initialization has arbitrary gauge; production
    # PreAnchorSubmap uses the same geometry gates with scale_bounds=None.
    scale_bounds = None if kinds == {"independent"} else (0.5, 2.0)
    anchors = [make_anchor(record) for record in records]
    sim, residual = fit_anchor_similarity(anchors)
    local_centers = np.array([camera_sim(anchor.local_pose).translation for anchor in anchors])
    centered = local_centers - local_centers.mean(axis=0)
    baseline = float(np.linalg.norm(centered) * sim.scale)
    # Conservative scale sensitivity using the existing 0.03-u center admission tolerance.
    sigma_log_scale_bound = 0.03 / baseline if baseline > 1e-12 else None
    jackknife = []
    for omitted in range(len(anchors)):
        fitted, _ = fit_anchor_similarity(anchors[:omitted] + anchors[omitted + 1:])
        if fitted.scale > 0 and np.isfinite(fitted.scale):
            jackknife.append(float(np.log(fitted.scale)))
    jackknife_sigma = None
    if len(jackknife) == len(anchors):
        values = np.array(jackknife)
        jackknife_sigma = float(np.sqrt((len(values) - 1) / len(values) * np.sum((values - values.mean()) ** 2)))
    return {
        "ordinals": [anchor.ordinal for anchor in anchors], "scale": float(sim.scale),
        "sim3": sim.to_vector().tolist(), "center_rms_u": float(residual),
        "rejection": anchor_consistency(anchors, sim, scale_bounds=scale_bounds),
        "main_center_baseline_norm_u": baseline,
        "scale_sensitivity_sigma_log_at_0_03_u": sigma_log_scale_bound,
        "scale_bounds": scale_bounds,
        "scale_outside_existing_bounds": (
            not scale_bounds[0] <= sim.scale <= scale_bounds[1] if scale_bounds else None
        ),
        "jackknife_sigma_log_scale": jackknife_sigma,
    }


def graph_predictions(records: list[dict], train: set[int], initial: Sim3, prior: bool) -> tuple[list[np.ndarray], dict]:
    local = [camera_sim(pose(record["local_pose"])) for record in records]
    seed = np.array([initial.compose(value).to_vector() for value in local])
    relative = [local[i].inverse().compose(local[i + 1]).to_vector() for i in range(len(local) - 1)]
    measured = {i: camera_sim(pose(records[i]["map_pose"])).to_vector() for i in train}
    odometry_sigma = np.array([np.deg2rad(1)] * 3 + [0.02] * 3 + [0.2])
    anchor_sigma = np.array([np.deg2rad(0.5)] * 3 + [0.01] * 3)
    frozen_root = np.zeros(7)

    def residuals(vector: np.ndarray) -> np.ndarray:
        nodes = vector.reshape(-1, 7)
        residual = [edge_residual(nodes[i], nodes[i + 1], value, odometry_sigma) for i, value in enumerate(relative)]
        # EDM supplies SE3, not a seventh absolute-scale observation.
        residual.extend(edge_residual(frozen_root, nodes[i], value, np.ones(7))[:6] / anchor_sigma for i, value in measured.items())
        if prior:
            # One reference-node soft regularizer, in main-aligned units. Its mean is
            # anchor-derived initialization, NOT an independent new measurement.
            residual.append(np.array([(nodes[0, 6] - seed[0, 6]) / 0.2]))
        return np.concatenate(residual)

    started = time.perf_counter()
    solution = least_squares(residuals, seed.ravel(), method="trf", loss="huber", f_scale=1, max_nfev=50)
    nodes = solution.x.reshape(-1, 7)
    if not np.isfinite(nodes).all():
        raise ValueError("nonfinite graph solution")
    scales = np.exp(nodes[:, 6])
    predictions = [camera_pose(Sim3.from_vector(value)) for value in nodes]
    return predictions, {
        "solver_success": bool(solution.success), "solver_message": str(solution.message),
        "solver_ms": (time.perf_counter() - started) * 1000,
        "nfev": int(solution.nfev), "cost": float(solution.cost),
        "node_scales": scales.tolist(), "fixed_main_root": frozen_root.tolist(),
        "scale_prior": {"reference_node": 1, "mean": float(initial.scale), "sigma_log": 0.2,
                        "independent_measurement": False} if prior else None,
    }


def analyze(records: list[dict], holdout_every: int = 3) -> dict:
    ordered = sorted(records, key=lambda value: int(value["ordinal"]))
    if len({int(record["ordinal"]) for record in ordered}) != len(ordered):
        raise ValueError("duplicate ordinal within one method/submap/source epoch")
    for record in ordered:
        make_anchor(record)
    windows = [fit_window(ordered[start:start + 8]) for start in range(0, len(ordered) - 7, 4)]
    if len(ordered) < 12:
        return {"status": "insufficient_independent_anchors", "count": len(ordered), "required": 12,
                "windows": windows,
                "reason": "window estimates are diagnostic only; need >=8 training and >=4 held-out anchors for graph comparison"}
    if int(ordered[-1]["ordinal"]) - int(ordered[0]["ordinal"]) > 48:
        return {"status": "anchor_span_exceeds_current_48_frame_window", "count": len(ordered)}
    heldout = {i for i in range(len(ordered)) if i % holdout_every == holdout_every - 1}
    train = set(range(len(ordered))) - heldout
    if len(train) < 8 or len(heldout) < 4:
        return {"status": "insufficient_train_or_holdout", "train": len(train), "heldout": len(heldout)}
    train_records = [ordered[i] for i in sorted(train)]
    fit = fit_window(train_records)
    result = {"status": "diagnosed", "count": len(ordered), "training": fit, "windows": windows,
              "heldout_ordinals": [int(ordered[i]["ordinal"]) for i in sorted(heldout)],
              "accuracy_claim": "agreement with independently solved frozen-map PnP; provenance distinguishes EDM from propagated KLT; no external ground truth"}
    if fit["rejection"] is not None:
        result.update(status="initial_sim3_rejected", graph="not_admitted; existing geometry gates preserved")
        return result
    initial = Sim3.from_vector(fit["sim3"])
    holdout_records = [ordered[i] for i in sorted(heldout)]
    result["rigid_submap_sim3"] = error_summary(holdout_records, [initial.apply_pose(pose(record["local_pose"])) for record in holdout_records])
    for name, prior in (("fine_sim3_graph", False), ("fine_sim3_graph_scale_prior", True)):
        predicted, solver = graph_predictions(ordered, train, initial, prior)
        result[name] = {"solver": solver, "heldout": error_summary(holdout_records, [predicted[i] for i in sorted(heldout)])}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.anchors.read_bytes()
    payload = json.loads(source)
    if payload["schema"] != "workmap-closure-anchors/v1":
        raise ValueError("unexpected anchor schema")
    groups: dict[tuple, list[dict]] = {}
    for record in payload["records"]:
        key = (record["method"], record["submap_id"], record["source_epoch"])
        groups.setdefault(key, []).append(record)
    report = {
        "schema": "workmap-sim3-diagnosis/v1", "anchor_sha256": hashlib.sha256(source).hexdigest(),
        "corrections_applied": False, "frozen_main_poses_or_landmarks_modified": False,
        "status": "no_independent_anchors" if not groups else "analyzed",
        "groups": [{"method": key[0], "submap_id": key[1], "source_epoch": key[2], **analyze(records)} for key, records in groups.items()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

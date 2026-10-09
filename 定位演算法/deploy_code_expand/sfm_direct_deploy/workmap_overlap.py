"""Conservative Sim(3) validation for measured session-to-release overlap.

All correspondences are EDM observations.  The solver does not infer matches
from spatial proximity, and the identities reserved for holdout never take part
in RANSAC or the final fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import math
import random
from typing import Any

import numpy as np

from direct_paths import vendor_sys_path

vendor_sys_path()
from sfm_diagnosis.sim3 import (  # noqa: E402
    SimilarityTransform,
    estimate_similarity_3d,
    evaluate_holdout_similarity,
)

_RANSAC_MAX_TRIPLES = 512
_HOLDOUT_MAX_RELATIVE_RESIDUAL = 0.05
_RANSAC_MEDIAN_SPAN_FRACTION = 0.02
_COORDINATE_TOLERANCE = 1e-7


def _failure(reason: str, source_keyframes: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": reason,
        "source_keyframes": list(source_keyframes),
        "fit_ids": [],
        "holdout_ids": [],
        "fit_new_ids": [],
        "holdout_new_ids": [],
    }


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _revision(row: Mapping[str, Any], new_id: int) -> int:
    revisions = row.get("landmark_revisions")
    if isinstance(revisions, Mapping):
        value = revisions.get(new_id, revisions.get(str(new_id)))
    else:
        value = row.get("landmark_revision")
    if value is None:
        raise ValueError(f"missing landmark revision for new_id {new_id}")
    return _integer(value, f"landmark revision for {new_id}")


def _vector(row: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        value = np.asarray(row[name], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid {name}") from error
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


def _frame_metadata(
    row: Mapping[str, Any],
) -> tuple[str, int, int, np.ndarray, np.ndarray, tuple[int, int]]:
    name = row.get("keyframe_name", row.get("source_keyframe_name"))
    if not isinstance(name, str) or not name:
        raise ValueError("missing source keyframe_name")
    try:
        keyframe_revision = _integer(row["keyframe_revision"], "keyframe_revision")
        pose_revision = _integer(row["pose_revision"], "pose_revision")
    except KeyError as error:
        raise ValueError(f"missing {error.args[0]}") from error
    if keyframe_revision < 0 or pose_revision < 0:
        raise ValueError("keyframe revisions must be non-negative")

    pose = _vector(row, "keyframe_pose", (3, 4))
    intrinsics = _vector(row, "keyframe_K", (3, 3))
    rotation = pose[:, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-3)
        or np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError("keyframe_pose rotation must be proper and orthonormal")
    if (
        intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
        or not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-7)
    ):
        raise ValueError("keyframe_K must be a finite pinhole calibration")
    try:
        size = np.asarray(row["image_size"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid image_size") from error
    if (
        size.shape != (2,)
        or not np.isfinite(size).all()
        or np.any(size <= 0.0)
        or np.any(size != np.floor(size))
    ):
        raise ValueError("image_size must contain positive integer width and height")
    return (
        name,
        keyframe_revision,
        pose_revision,
        pose.copy(),
        intrinsics.copy(),
        (int(size[0]), int(size[1])),
    )


def _same_metadata(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0:3] == right[0:3]
        and left[5] == right[5]
        and np.array_equal(left[3], right[3])
        and np.array_equal(left[4], right[4])
    )


def _row_identity(row: Any) -> tuple[int, int]:
    if not isinstance(row, Mapping):
        raise ValueError("each evidence row must be a mapping")
    try:
        new_id = _integer(row["new_id"], "new_id")
        old_id = _integer(row["old_id"], "old_id")
    except KeyError as error:
        raise ValueError(f"missing {error.args[0]}") from error
    if new_id >= 0 or old_id <= 0:
        raise ValueError("overlap identities must be new_id < 0 and old_id > 0")
    if (
        "old_reference_name" not in row
        or not isinstance(row["old_reference_name"], str)
        or not row["old_reference_name"]
    ):
        raise ValueError("missing old_reference_name provenance")
    return new_id, old_id


def _register_frame(frames: dict[str, dict[str, Any]], row: Mapping[str, Any]) -> tuple[Any, ...]:
    frame_meta = _frame_metadata(row)
    frame_name = frame_meta[0]
    existing_frame = frames.get(frame_name)
    if existing_frame is None:
        frames[frame_name] = {"meta": frame_meta}
    elif not _same_metadata(existing_frame["meta"], frame_meta):
        raise ValueError(f"conflicting revisions or frame calibration for {frame_name}")
    return frame_meta


def _row_measurement(
    row: Mapping[str, Any], new_id: int, frame_meta: tuple[Any, ...]
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    revision = _revision(row, new_id)
    point_xyz = _vector(row, "new_xyz", (3,))
    old_xyz = _vector(row, "old_xyz", (3,))
    new_xy = _vector(row, "new_xy", (2,))
    _vector(row, "old_xy", (2,))
    width, height = frame_meta[5]
    if not (0.0 <= new_xy[0] < width and 0.0 <= new_xy[1] < height):
        raise ValueError(f"new_xy falls outside source image for {frame_meta[0]}")
    try:
        confidence = float(row["confidence"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid confidence") from error
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite")
    return revision, point_xyz, old_xyz, new_xy, confidence


def _check_identity_links(
    identities: Mapping[int, Mapping[str, Any]],
    reverse: Mapping[int, int],
    seen_frame_ids: set[tuple[str, int]],
    frame_name: str,
    new_id: int,
    old_id: int,
) -> None:
    mapped_old = identities.get(new_id)
    if mapped_old is not None and mapped_old["old_id"] != old_id:
        raise ValueError(f"new_id {new_id} maps to conflicting old identities")
    mapped_new = reverse.get(old_id)
    if mapped_new is not None and mapped_new != new_id:
        raise ValueError(f"old_id {old_id} maps to conflicting new identities")
    if (frame_name, new_id) in seen_frame_ids:
        raise ValueError(f"duplicate identity observation for {new_id} in {frame_name}")


def _merge_identity(
    identities: dict[int, dict[str, Any]],
    reverse: dict[int, int],
    new_id: int,
    old_id: int,
    revision: int,
    point_xyz: np.ndarray,
    old_xyz: np.ndarray,
) -> dict[str, Any]:
    mapped_old = identities.get(new_id)
    if mapped_old is None:
        identities[new_id] = {
            "new_id": new_id,
            "old_id": old_id,
            "new_xyz": point_xyz.copy(),
            "old_xyz": old_xyz.copy(),
            "landmark_revision": revision,
            "observations": [],
        }
        reverse[old_id] = new_id
        return identities[new_id]
    if revision != mapped_old["landmark_revision"]:
        raise ValueError(f"conflicting landmark revisions for {new_id}")
    if not np.allclose(point_xyz, mapped_old["new_xyz"], rtol=0.0, atol=_COORDINATE_TOLERANCE):
        raise ValueError(f"conflicting new_xyz for identity {new_id}")
    if not np.allclose(old_xyz, mapped_old["old_xyz"], rtol=0.0, atol=_COORDINATE_TOLERANCE):
        raise ValueError(f"conflicting old_xyz for identity {old_id}")
    return mapped_old


def _parse_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    identities: dict[int, dict[str, Any]] = {}
    reverse: dict[int, int] = {}
    frames: dict[str, dict[str, Any]] = {}
    seen_frame_ids: set[tuple[str, int]] = set()
    sessions: set[Any] = set()
    epochs: set[Any] = set()

    for row in rows:
        new_id, old_id = _row_identity(row)
        frame_meta = _register_frame(frames, row)
        frame_name = frame_meta[0]
        if row.get("session_id") is not None:
            sessions.add(row["session_id"])
        if row.get("source_epoch") is not None:
            epochs.add(row["source_epoch"])
        revision, point_xyz, old_xyz, new_xy, confidence = _row_measurement(row, new_id, frame_meta)
        _check_identity_links(identities, reverse, seen_frame_ids, frame_name, new_id, old_id)
        seen_frame_ids.add((frame_name, new_id))
        identity = _merge_identity(
            identities, reverse, new_id, old_id, revision, point_xyz, old_xyz
        )
        identity["observations"].append(
            {
                "keyframe_name": frame_name,
                "keyframe_pose": frame_meta[3],
                "keyframe_K": frame_meta[4],
                "image_size": frame_meta[5],
                "new_xy": new_xy.copy(),
                "confidence": confidence,
            }
        )

    if len(sessions) > 1 or len(epochs) > 1:
        raise ValueError("evidence spans multiple sessions or source epochs")
    if len(frames) < 3:
        raise ValueError("overlap evidence requires at least three distinct source keyframes")
    return identities, frames


def _noncollinear(points: np.ndarray) -> bool:
    if points.shape[0] < 3:
        return False
    centered = points - points.mean(axis=0)
    return int(np.linalg.matrix_rank(centered)) >= 2


def _split_identities(ids: Sequence[int]) -> tuple[list[int], list[int]]:
    # Hash only identity labels: the holdout is fixed without inspecting either
    # coordinate set, residuals, or reprojection quality.
    ordered = sorted(
        ids,
        key=lambda value: hashlib.blake2b(str(value).encode("ascii"), digest_size=16).digest(),
    )
    split = len(ordered) // 2
    return sorted(ordered[:split]), sorted(ordered[split:])


def _ransac_triples(ids: Sequence[int]) -> list[tuple[int, int, int]]:
    all_count = math.comb(len(ids), 3)
    if all_count <= _RANSAC_MAX_TRIPLES:
        return list(itertools.combinations(ids, 3))
    seed_material = ",".join(str(value) for value in ids).encode("ascii")
    seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
    rng = random.Random(seed)
    triples: set[tuple[int, int, int]] = set()
    attempts = 0
    while len(triples) < _RANSAC_MAX_TRIPLES and attempts < _RANSAC_MAX_TRIPLES * 20:
        triples.add(tuple(sorted(rng.sample(list(ids), 3))))
        attempts += 1
    return sorted(triples)


def _points(
    identities: Mapping[int, Mapping[str, Any]], ids: Sequence[int], field: str
) -> np.ndarray:
    return np.asarray([identities[key][field] for key in ids], dtype=np.float64)


def _ransac_threshold(fit_target: np.ndarray) -> float:
    separations = np.linalg.norm(fit_target[:, None, :] - fit_target[None, :, :], axis=2)
    nonzero = separations[separations > np.finfo(np.float64).eps]
    if not len(nonzero):
        raise ValueError("fit target points have zero span")
    return max(1e-9, _RANSAC_MEDIAN_SPAN_FRACTION * float(np.median(nonzero)))


def _ransac_best_inliers(
    identities: Mapping[int, Mapping[str, Any]],
    fit_ids: Sequence[int],
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    threshold: float,
) -> list[int]:
    best_ids: list[int] = []
    best_score: tuple[int, float, float] | None = None
    for triple in _ransac_triples(fit_ids):
        try:
            candidate = estimate_similarity_3d(
                _points(identities, triple, "new_xyz"), _points(identities, triple, "old_xyz")
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = np.linalg.norm(candidate.apply(fit_source) - fit_target, axis=1)
        inlier_mask = residuals <= threshold
        inlier_ids = [key for key, keep in zip(fit_ids, inlier_mask) if keep]
        if len(inlier_ids) < 6:
            continue
        inlier_residuals = residuals[inlier_mask]
        score = (
            len(inlier_ids),
            -float(np.median(inlier_residuals)),
            -float(np.dot(inlier_residuals, inlier_residuals)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_ids = inlier_ids
    return best_ids


def _refine_inliers(
    identities: Mapping[int, Mapping[str, Any]],
    fit_ids: Sequence[int],
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    threshold: float,
    current: list[int],
) -> tuple[SimilarityTransform, list[int]]:
    for _ in range(4):
        transform = estimate_similarity_3d(
            _points(identities, current, "new_xyz"), _points(identities, current, "old_xyz")
        )
        residuals = np.linalg.norm(transform.apply(fit_source) - fit_target, axis=1)
        updated = [key for key, residual in zip(fit_ids, residuals) if residual <= threshold]
        if len(updated) < 6:
            raise ValueError("RANSAC refinement retained fewer than six fit inliers")
        if updated == current:
            return transform, current
        current = updated
    raise ValueError("RANSAC inlier set did not stabilize")


def _ransac_fit(
    identities: Mapping[int, Mapping[str, Any]], fit_ids: Sequence[int]
) -> tuple[SimilarityTransform, list[int], float]:
    fit_source = _points(identities, fit_ids, "new_xyz")
    fit_target = _points(identities, fit_ids, "old_xyz")
    if not _noncollinear(fit_source) or not _noncollinear(fit_target):
        raise ValueError("fit identity points must be non-collinear")
    threshold = _ransac_threshold(fit_target)
    best_ids = _ransac_best_inliers(identities, fit_ids, fit_source, fit_target, threshold)
    if len(best_ids) < 6:
        raise ValueError("deterministic RANSAC found fewer than six fit inliers")
    transform, current = _refine_inliers(
        identities, fit_ids, fit_source, fit_target, threshold, best_ids
    )
    inlier_source = _points(identities, current, "new_xyz")
    inlier_target = _points(identities, current, "old_xyz")
    if not _noncollinear(inlier_source) or not _noncollinear(inlier_target):
        raise ValueError("RANSAC fit inliers must be non-collinear")
    residuals = np.linalg.norm(transform.apply(inlier_source) - inlier_target, axis=1)
    fit_residual = float(np.sqrt(np.mean(residuals**2)))
    return transform, sorted(current), fit_residual


def _transform_parts(transform: SimilarityTransform) -> tuple[float, np.ndarray, np.ndarray]:
    return (
        float(transform.scale),
        np.asarray(transform.rotation, dtype=np.float64),
        np.asarray(transform.translation, dtype=np.float64),
    )


def _holdout_reprojection(
    observation: Mapping[str, Any],
    old_xyz: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, float]:
    """Pixel error of one old holdout point before and after the candidate Sim3."""
    pose = observation["keyframe_pose"]
    intrinsics = observation["keyframe_K"]
    measured = observation["new_xy"]
    width, height = observation["image_size"]
    if not (0.0 <= measured[0] < width and 0.0 <= measured[1] < height):
        raise ValueError("holdout observation falls outside its source image")

    camera_before = pose[:, :3] @ old_xyz + pose[:, 3]
    # If X_global = s R X_local + t, the equivalent rigid camera pose
    # has R_cam' = R_cam R^T and t_cam' = s t_cam - R_cam' t.
    camera_rotation = pose[:, :3] @ rotation.T
    camera_translation = scale * pose[:, 3] - camera_rotation @ translation
    camera_after = camera_rotation @ old_xyz + camera_translation
    if not np.isfinite(camera_before).all() or not np.isfinite(camera_after).all():
        raise ValueError("non-finite holdout camera coordinates")
    if camera_after[2] <= 1e-8:
        raise ValueError("candidate Sim3 puts an old holdout point behind the camera")
    if abs(camera_before[2]) <= 1e-8:
        raise ValueError("unchanged workmap pose cannot project an old holdout point")

    pixel_before_h = intrinsics @ camera_before
    pixel_after_h = intrinsics @ camera_after
    if not np.isfinite(pixel_before_h).all() or not np.isfinite(pixel_after_h).all():
        raise ValueError("non-finite holdout reprojection")
    pixel_before = pixel_before_h[:2] / pixel_before_h[2]
    pixel_after = pixel_after_h[:2] / pixel_after_h[2]
    if not np.isfinite(pixel_before).all() or not np.isfinite(pixel_after).all():
        raise ValueError("non-finite holdout reprojection")
    return float(np.linalg.norm(pixel_before - measured)), float(
        np.linalg.norm(pixel_after - measured)
    )


def _project_holdout(
    identities: Mapping[int, Mapping[str, Any]],
    holdout_ids: Sequence[int],
    transform: SimilarityTransform,
) -> tuple[np.ndarray, np.ndarray]:
    scale, rotation, translation = _transform_parts(transform)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("candidate Sim3 scale must be finite and positive")
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
    ):
        raise ValueError("candidate Sim3 must be finite")

    before: list[float] = []
    after: list[float] = []
    for new_id in holdout_ids:
        old_xyz = np.asarray(identities[new_id]["old_xyz"], dtype=np.float64)
        for observation in identities[new_id]["observations"]:
            error_before, error_after = _holdout_reprojection(
                observation, old_xyz, scale, rotation, translation
            )
            before.append(error_before)
            after.append(error_after)
    return np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)


def _checked_split(identities: Mapping[int, Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    fit_ids, holdout_ids = _split_identities(sorted(identities))
    if len(fit_ids) < 6 or len(holdout_ids) < 6:
        raise ValueError("fit and holdout each require at least six unique identities")
    if not _noncollinear(_points(identities, fit_ids, "new_xyz")) or not _noncollinear(
        _points(identities, fit_ids, "old_xyz")
    ):
        raise ValueError("fit identity points must be non-collinear")
    if not _noncollinear(_points(identities, holdout_ids, "new_xyz")) or not _noncollinear(
        _points(identities, holdout_ids, "old_xyz")
    ):
        raise ValueError("holdout identity points must be non-collinear")
    if set(fit_ids) & set(holdout_ids):
        raise ValueError("fit and holdout identities must be disjoint")
    return fit_ids, holdout_ids


def _holdout_3d(
    identities: Mapping[int, Mapping[str, Any]], fit_ids: Sequence[int], holdout_ids: Sequence[int]
) -> tuple[SimilarityTransform, np.ndarray, np.ndarray]:
    holdout_source = _points(identities, holdout_ids, "new_xyz")
    holdout_target = _points(identities, holdout_ids, "old_xyz")
    evaluation = evaluate_holdout_similarity(
        _points(identities, fit_ids, "new_xyz"),
        _points(identities, fit_ids, "old_xyz"),
        holdout_source,
        holdout_target,
        max_relative_residual=_HOLDOUT_MAX_RELATIVE_RESIDUAL,
    )
    transform = evaluation["transform"]
    if not isinstance(transform, SimilarityTransform):
        raise ValueError("holdout evaluator returned an invalid transform")
    if not evaluation["consistent"]:
        raise ValueError("independent 3D holdout residual exceeds the relative limit")
    before_3d = np.linalg.norm(holdout_source - holdout_target, axis=1)
    after_3d = np.asarray(evaluation["holdout_residuals"], dtype=np.float64)
    return transform, before_3d, after_3d


def fit_overlap_sim3(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Estimate and independently validate a local-to-global Sim(3).

    Rows represent actual EDM-verified 2D↔3D matches.  Repeated landmark
    identities across keyframes are retained only as reprojection observations;
    each identity pair contributes once to 3D fitting and holdout evaluation.
    """

    try:
        return _fit_overlap_sim3(evidence)
    except (ValueError, TypeError, KeyError, OverflowError, np.linalg.LinAlgError) as error:
        return _failure(str(error), _evidence_keyframes(evidence))


def _evidence_keyframes(evidence: Any) -> list[str]:
    try:
        return sorted(
            {
                str(row.get("keyframe_name", row.get("source_keyframe_name")))
                for row in evidence
                if isinstance(row, Mapping)
                and row.get("keyframe_name", row.get("source_keyframe_name"))
            }
        )
    except (TypeError, ValueError):
        return []


def _fit_overlap_sim3(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise ValueError("evidence must be a sequence of measured identity rows")
    rows = list(evidence)
    if not rows:
        raise ValueError("no overlap evidence")
    identities, frames = _parse_evidence(rows)
    source_keyframes = sorted(frames)
    if len(identities) < 12:
        raise ValueError("at least twelve unique overlap identities are required")
    fit_ids, holdout_ids = _checked_split(identities)

    _ransac_transform, fit_ids, fit_residual = _ransac_fit(identities, fit_ids)
    transform, before_3d, after_3d = _holdout_3d(identities, fit_ids, holdout_ids)
    before_3d_p90 = float(np.percentile(before_3d, 90.0))
    after_3d_p90 = float(np.percentile(after_3d, 90.0))
    if not after_3d_p90 < before_3d_p90 - max(1e-9, 1e-9 * before_3d_p90):
        raise ValueError("independent 3D holdout error does not genuinely improve")

    before_2d, after_2d = _project_holdout(identities, holdout_ids, transform)
    before_2d_p90 = float(np.percentile(before_2d, 90.0))
    after_2d_p90 = float(np.percentile(after_2d, 90.0))
    if not after_2d_p90 < before_2d_p90 - max(1e-6, 1e-9 * before_2d_p90):
        raise ValueError("independent 2D reprojection holdout error does not improve")

    scale, rotation, translation = _transform_parts(transform)
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
    ):
        raise ValueError("candidate Sim3 must be finite with positive scale")
    fit_new_ids = sorted(int(identities[key]["new_id"]) for key in fit_ids)
    holdout_new_ids = sorted(int(identities[key]["new_id"]) for key in holdout_ids)
    return {
        "ok": True,
        "reason": "",
        "transform": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "fit_ids": fit_new_ids,
        "holdout_ids": holdout_new_ids,
        "fit_new_ids": fit_new_ids,
        "holdout_new_ids": holdout_new_ids,
        "fit_residual": fit_residual,
        "holdout_before": before_3d.tolist(),
        "holdout_after": after_3d.tolist(),
        "before_3d_p90": before_3d_p90,
        "after_3d_p90": after_3d_p90,
        "before_2d_p90": before_2d_p90,
        "after_2d_p90": after_2d_p90,
        "source_keyframes": source_keyframes,
    }


__all__ = ["fit_overlap_sim3"]

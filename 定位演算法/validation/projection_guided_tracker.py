"""Opt-in projection-guided TRACK fast path with unchanged deep fallback."""

from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pose_types import Pose
from production_xfeat_tracker import (
    ProductionXFeatTracker,
    _ret_inlier_mask,
)
from reloc_localizer_xfeat import DEVICE, _to_device_feats, extract_xfeat


SIDECAR_SCHEMA = "xfeat-track-landmarks"


def ordered_names_sha256(names: Iterable[str]) -> str:
    """Hash ordered names using the sidecar's uint64-length-prefixed contract."""
    digest = hashlib.sha256()
    for name in names:
        encoded = str(name).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    omega = _skew(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * omega + (omega @ omega) / 6.0
    theta2 = theta * theta
    return (
        np.eye(3)
        + ((1.0 - math.cos(theta)) / theta2) * omega
        + ((theta - math.sin(theta)) / (theta2 * theta)) * (omega @ omega)
    )


def _se3_log(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform, dtype=np.float64)
    phi = cv2.Rodrigues(transform[:3, :3])[0].reshape(3)
    rho = np.linalg.solve(_so3_left_jacobian(phi), transform[:3, 3])
    return rho, phi


def _se3_exp(rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(phi)[0]
    transform[:3, 3] = _so3_left_jacobian(phi) @ np.asarray(rho, dtype=np.float64)
    return transform


def extrapolate_tcw(
    previous_tcw: np.ndarray,
    last_tcw: np.ndarray,
    previous_stamp: float,
    last_stamp: float,
    query_stamp: float,
    max_scale: float = 3.0,
) -> tuple[np.ndarray, float]:
    """SE(3) constant-velocity extrapolation for world-to-camera transforms."""
    previous_tcw = np.asarray(previous_tcw, dtype=np.float64)
    last_tcw = np.asarray(last_tcw, dtype=np.float64)
    if previous_tcw.shape != (4, 4) or last_tcw.shape != (4, 4):
        raise ValueError("Tcw samples must be 4x4 matrices")
    dt = float(last_stamp) - float(previous_stamp)
    if not math.isfinite(dt) or dt <= 1e-6:
        raise ValueError("pose history timestamps must be strictly increasing")
    scale = (float(query_stamp) - float(last_stamp)) / dt
    if not math.isfinite(scale):
        raise ValueError("query timestamp produced a non-finite prediction scale")
    scale = float(np.clip(scale, 0.0, max_scale))
    delta = last_tcw @ np.linalg.inv(previous_tcw)
    rho, phi = _se3_log(delta)
    predicted = _se3_exp(scale * rho, scale * phi) @ last_tcw
    predicted[3] = (0.0, 0.0, 0.0, 1.0)
    return predicted, scale


def _opencv_camera(cam) -> tuple[np.ndarray, np.ndarray]:
    model = str(cam.model).upper()
    params = [float(value) for value in cam.params]
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        fx, fy = f, f
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx, fy = f, f
        dist = np.zeros(5, dtype=np.float64)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        dist = np.zeros(5, dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy = params[:4]
        dist = np.asarray(params[4:8], dtype=np.float64)
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy = params[:4]
        dist = np.asarray(params[4:12], dtype=np.float64)
    else:
        raise ValueError(f"unsupported projection camera model: {cam.model}")
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return camera_matrix, dist


def project_visible_landmarks(
    xyz: np.ndarray,
    tcw: np.ndarray,
    cam,
) -> tuple[np.ndarray, np.ndarray]:
    """Project positive-depth landmarks and retain only finite image points."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    tcw = np.asarray(tcw, dtype=np.float64)
    if tcw.shape != (4, 4):
        raise ValueError("Tcw must be a 4x4 matrix")
    if not len(xyz):
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    camera_matrix, dist = _opencv_camera(cam)
    rotation = tcw[:3, :3]
    translation = tcw[:3, 3]
    camera_xyz = (rotation @ xyz.astype(np.float64).T).T + translation
    positive_indices = np.flatnonzero(
        np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 1e-6)
    )
    if not len(positive_indices):
        return positive_indices, np.zeros((0, 2), dtype=np.float32)
    rvec = cv2.Rodrigues(rotation)[0]
    projected, _ = cv2.projectPoints(
        xyz[positive_indices].astype(np.float64),
        rvec,
        translation,
        camera_matrix,
        dist,
    )
    projected = projected.reshape(-1, 2)
    inside = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < float(cam.width))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < float(cam.height))
    )
    return (
        positive_indices[inside].astype(np.int64, copy=False),
        projected[inside].astype(np.float32, copy=False),
    )


class TrackLandmarkSidecar:
    """Validated unique XFeat landmark descriptors and per-reference visibility."""

    REQUIRED_KEYS = {
        "point3D_id",
        "xyz",
        "descriptor",
        "obs_offsets",
        "obs_ref_idx",
        "obs_kp_idx",
        "ref_kp_offsets",
        "ref_kp_landmark_idx",
        "schema_name",
        "schema_version",
        "ref_bundle_sha256",
        "ref_names_sha256",
        "source_features_sha256",
        "source_images_sha256",
        "source_points3D_sha256",
    }

    def __init__(
        self,
        point3d_id: np.ndarray,
        xyz: np.ndarray,
        descriptor: np.ndarray,
        ref_kp_offsets: np.ndarray,
        ref_kp_landmark_idx: np.ndarray,
    ) -> None:
        self.point3d_id = point3d_id
        self.xyz = xyz
        self.descriptor = descriptor
        self.ref_kp_offsets = ref_kp_offsets
        self.ref_kp_landmark_idx = ref_kp_landmark_idx
        self._descriptor_tensor: torch.Tensor | None = None

    @staticmethod
    def read_binding(path: str | Path) -> tuple[str, str]:
        with np.load(Path(path), allow_pickle=False) as archive:
            if str(np.asarray(archive["schema_name"]).item()) != SIDECAR_SCHEMA:
                raise ValueError("unsupported TRACK landmark sidecar schema")
            if int(np.asarray(archive["schema_version"]).item()) != 1:
                raise ValueError("unsupported TRACK landmark sidecar version")
            return (
                str(np.asarray(archive["ref_bundle_sha256"]).item()),
                str(np.asarray(archive["ref_names_sha256"]).item()),
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        ref_names: list[str],
        verified_bundle_sha256: str,
    ) -> "TrackLandmarkSidecar":
        path = Path(path)
        arrays = cls._read_track_landmark_arrays(
            path,
            ref_names,
            verified_bundle_sha256,
        )
        cls._validate_track_landmark_arrays(*arrays, ref_count=len(ref_names))
        return cls(*arrays)

    @classmethod
    def _read_track_landmark_arrays(
        cls,
        path: Path,
        ref_names: list[str],
        verified_bundle_sha256: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            missing = cls.REQUIRED_KEYS - set(archive.files)
            if missing:
                raise ValueError(f"TRACK landmark sidecar missing keys: {sorted(missing)}")
            bundle_hash, names_hash = cls.read_binding(path)
            if bundle_hash != str(verified_bundle_sha256):
                raise ValueError("TRACK landmark sidecar bundle SHA-256 mismatch")
            if names_hash != ordered_names_sha256(ref_names):
                raise ValueError("TRACK landmark sidecar ordered reference names mismatch")
            return (
                np.ascontiguousarray(archive["point3D_id"], dtype=np.int64),
                np.ascontiguousarray(archive["xyz"], dtype=np.float32),
                np.ascontiguousarray(archive["descriptor"], dtype=np.float16),
                np.ascontiguousarray(archive["ref_kp_offsets"], dtype=np.int64),
                np.ascontiguousarray(archive["ref_kp_landmark_idx"], dtype=np.int32),
            )

    @staticmethod
    def _validate_track_landmark_arrays(
        point3d_id: np.ndarray,
        xyz: np.ndarray,
        descriptor: np.ndarray,
        ref_kp_offsets: np.ndarray,
        ref_kp_landmark_idx: np.ndarray,
        *,
        ref_count: int,
    ) -> None:
        count = len(point3d_id)
        if xyz.shape != (count, 3) or descriptor.shape != (count, 64):
            raise ValueError("TRACK landmark xyz/descriptor shapes are inconsistent")
        if not np.isfinite(xyz).all() or not np.isfinite(descriptor).all():
            raise ValueError("TRACK landmark sidecar contains non-finite values")
        if len(np.unique(point3d_id)) != count:
            raise ValueError("TRACK landmark point3D_id values are not unique")
        if ref_kp_offsets.shape != (ref_count + 1,):
            raise ValueError("TRACK landmark reference offsets do not match the bundle")
        if (
            ref_kp_offsets[0] != 0
            or ref_kp_offsets[-1] != len(ref_kp_landmark_idx)
            or np.any(np.diff(ref_kp_offsets) < 0)
        ):
            raise ValueError("TRACK landmark reference offsets are invalid")
        if np.any((ref_kp_landmark_idx < -1) | (ref_kp_landmark_idx >= count)):
            raise ValueError("TRACK landmark reverse lookup contains invalid indices")
        norms = np.linalg.norm(descriptor.astype(np.float32), axis=1)
        if np.any(np.abs(norms - 1.0) > 0.02):
            raise ValueError("TRACK landmark representative descriptors are not normalized")

    def landmarks_for_refs(self, ref_ids: Iterable[int]) -> np.ndarray:
        chunks = []
        n_refs = len(self.ref_kp_offsets) - 1
        for ref_id in dict.fromkeys(int(value) for value in ref_ids):
            if not 0 <= ref_id < n_refs:
                continue
            start, end = self.ref_kp_offsets[ref_id : ref_id + 2]
            values = self.ref_kp_landmark_idx[int(start) : int(end)]
            chunks.append(values[values >= 0])
        if not chunks:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(chunks)).astype(np.int64, copy=False)

    def descriptor_tensor(self, device: str = DEVICE) -> torch.Tensor:
        if self._descriptor_tensor is None or str(self._descriptor_tensor.device) != str(device):
            tensor = torch.from_numpy(self.descriptor.astype(np.float32, copy=False)).to(device)
            self._descriptor_tensor = F.normalize(tensor, dim=-1)
        return self._descriptor_tensor


def _spatial_candidate_pairs(
    projected_xy: np.ndarray,
    query_xy: np.ndarray,
    max_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_size = float(max_radius)
    grid: dict[tuple[int, int], list[int]] = {}
    for query_index, point in enumerate(query_xy):
        cell = (int(math.floor(point[0] / cell_size)), int(math.floor(point[1] / cell_size)))
        grid.setdefault(cell, []).append(query_index)
    landmark_rows: list[int] = []
    query_rows: list[int] = []
    distances: list[float] = []
    max_radius_sq = max_radius * max_radius
    for landmark_row, point in enumerate(projected_xy):
        cell_x = int(math.floor(point[0] / cell_size))
        cell_y = int(math.floor(point[1] / cell_size))
        candidates = []
        for y in range(cell_y - 1, cell_y + 2):
            for x in range(cell_x - 1, cell_x + 2):
                candidates.extend(grid.get((x, y), ()))
        if not candidates:
            continue
        candidate_array = np.asarray(candidates, dtype=np.int64)
        delta = query_xy[candidate_array] - point[None, :]
        distance_sq = np.sum(delta * delta, axis=1)
        keep = distance_sq <= max_radius_sq
        if np.any(keep):
            selected = candidate_array[keep]
            landmark_rows.extend([landmark_row] * len(selected))
            query_rows.extend(selected.tolist())
            distances.extend(np.sqrt(distance_sq[keep]).tolist())
    return (
        np.asarray(landmark_rows, dtype=np.int64),
        np.asarray(query_rows, dtype=np.int64),
        np.asarray(distances, dtype=np.float32),
    )


@torch.inference_mode()
def match_projected_landmarks(
    projected_xy: np.ndarray,
    projected_landmark_idx: np.ndarray,
    query_xy: np.ndarray,
    query_descriptor: torch.Tensor,
    landmark_descriptor: torch.Tensor,
    radii: tuple[float, ...] = (15.0, 25.0, 40.0),
    min_score: float = 0.85,
    ratio: float = 0.8,
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Match pre-normalized descriptors with spatial, ratio and one-to-one gates."""
    projected_xy = np.asarray(projected_xy, dtype=np.float32).reshape(-1, 2)
    projected_landmark_idx = np.asarray(projected_landmark_idx, dtype=np.int64)
    query_xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    if len(projected_xy) != len(projected_landmark_idx):
        raise ValueError("projected coordinates and landmark indices must align")
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("projection search radii must be positive")
    pair_landmark, pair_query, pair_distance = _spatial_candidate_pairs(
        projected_xy, query_xy, max(radii)
    )
    results = {}
    if not len(pair_landmark):
        empty_i = np.zeros(0, dtype=np.int64)
        empty_s = np.zeros(0, dtype=np.float32)
        return {float(radius): (empty_i.copy(), empty_i.copy(), empty_s.copy()) for radius in radii}
    query_descriptor = query_descriptor.float()
    landmark_descriptor = landmark_descriptor.float()
    score_chunks = []
    for start in range(0, len(pair_landmark), 65536):
        end = min(start + 65536, len(pair_landmark))
        landmark_ids = torch.as_tensor(
            projected_landmark_idx[pair_landmark[start:end]],
            device=landmark_descriptor.device,
            dtype=torch.long,
        )
        query_ids = torch.as_tensor(
            pair_query[start:end], device=query_descriptor.device, dtype=torch.long
        )
        score_chunks.append(
            torch.sum(landmark_descriptor[landmark_ids] * query_descriptor[query_ids], dim=1)
            .detach()
            .cpu()
        )
    scores = torch.cat(score_chunks).numpy().astype(np.float32, copy=False)

    for radius in radii:
        eligible = np.flatnonzero(pair_distance <= float(radius))
        candidate_scores = np.zeros(0, dtype=np.float32)
        candidate_landmarks = np.zeros(0, dtype=np.int64)
        candidate_queries = np.zeros(0, dtype=np.int64)
        if len(eligible):
            rows = pair_landmark[eligible]
            row_count = len(projected_landmark_idx)
            best_scores = np.full(row_count, -np.inf, dtype=np.float32)
            np.maximum.at(best_scores, rows, scores[eligible])
            is_best = scores[eligible] == best_scores[rows]
            sentinel = len(scores)
            best_pairs = np.full(row_count, sentinel, dtype=np.int64)
            np.minimum.at(best_pairs, rows[is_best], eligible[is_best])
            not_best = eligible != best_pairs[rows]
            second_scores = np.full(row_count, -np.inf, dtype=np.float32)
            if np.any(not_best):
                np.maximum.at(
                    second_scores,
                    rows[not_best],
                    scores[eligible[not_best]],
                )
            present = best_pairs < sentinel
            best_distance = 1.0 - best_scores
            second_distance = np.maximum(1.0 - second_scores, 1e-8)
            ratio_ok = ~np.isfinite(second_scores) | (
                best_distance <= float(ratio) * second_distance
            )
            accepted_rows = np.flatnonzero(present & (best_scores >= float(min_score)) & ratio_ok)
            accepted_pairs = best_pairs[accepted_rows]
            candidate_scores = best_scores[accepted_rows]
            candidate_landmarks = projected_landmark_idx[accepted_rows]
            candidate_queries = pair_query[accepted_pairs]
        used_query: set[int] = set()
        accepted_rows = []
        order = np.lexsort((candidate_queries, candidate_landmarks, -candidate_scores))
        for row in order:
            query_index = int(candidate_queries[row])
            if query_index in used_query:
                continue
            used_query.add(query_index)
            accepted_rows.append(int(row))
        accepted_rows = np.asarray(accepted_rows, dtype=np.int64)
        results[float(radius)] = (
            candidate_landmarks[accepted_rows],
            candidate_queries[accepted_rows],
            candidate_scores[accepted_rows],
        )
    return results


@dataclass(frozen=True)
class _PoseSample:
    tcw: np.ndarray
    stamp: float


class ProjectionGuidedTracker(ProductionXFeatTracker):
    """TRACK-only projection fast path; every insufficient attempt falls back deep."""

    def __init__(
        self,
        *args,
        landmark_sidecar: TrackLandmarkSidecar,
        search_radii: tuple[float, ...] = (15.0, 25.0, 40.0),
        min_score: float = 0.60,
        ratio: float = 0.95,
        max_prediction_scale: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if len(landmark_sidecar.ref_kp_offsets) != len(self.map.ref_names) + 1:
            raise ValueError("TRACK landmark sidecar reference count mismatch")
        self.landmark_sidecar = landmark_sidecar
        self.projection_search_radii = tuple(float(value) for value in search_radii)
        self.projection_min_score = float(min_score)
        self.projection_ratio = float(ratio)
        self.projection_max_scale = float(max_prediction_scale)
        self._projection_history: list[_PoseSample] = []
        self._projection_pending: _PoseSample | None = None

    def _clear_tracking_history(self) -> None:
        super()._clear_tracking_history()
        if hasattr(self, "_projection_history"):
            self._projection_history.clear()
            self._projection_pending = None

    @staticmethod
    def _sample_from_ret(ret, stamp: float) -> _PoseSample:
        transform = ret["cam_from_world"]
        tcw = np.eye(4, dtype=np.float64)
        tcw[:3, :3] = np.asarray(transform.rotation.matrix(), dtype=np.float64)
        tcw[:3, 3] = np.asarray(transform.translation, dtype=np.float64)
        return _PoseSample(tcw=tcw, stamp=float(stamp))

    def _gate(self, ret, info: dict, mode: str):
        ok, weak, pose = super()._gate(ret, info, mode)
        if ok and pose is not None:
            self._projection_pending = self._sample_from_ret(ret, self._frame_capture_stamp)
        return ok, weak, pose

    def _publish_success(self, pose: Pose, info: dict, weak: bool, source_mode: str | None = None):
        sample = self._projection_pending
        super()._publish_success(pose, info, weak, source_mode=source_mode)
        if sample is not None:
            self._projection_history.append(sample)
            self._projection_history = self._projection_history[-2:]
        self._projection_pending = None

    def localize_frame(self, frame: np.ndarray, capture_stamp: float | None = None) -> Pose | None:
        stamp = time.monotonic() if capture_stamp is None else float(capture_stamp)
        if not math.isfinite(stamp) or stamp > time.monotonic() + 0.05:
            self._last_info = {
                "mode": self.state.mode,
                "next_mode": self.state.mode,
                "error": "invalid_capture_stamp",
                "inliers": 0,
            }
            return None
        self._frame_capture_stamp = stamp
        if self.state.mode != "TRACK" or len(self._projection_history) < 2:
            return self._localize_frame_deep(frame)
        return self._localize_frame_projection(frame)

    def _localize_frame_projection(self, frame: np.ndarray) -> Pose | None:
        start = time.perf_counter()
        self._frame_counters = {"projection_call_count": 1}
        candidates = self._track_candidates(weak=False)
        anchor_indices = self.landmark_sidecar.landmarks_for_refs(candidates)
        fallback = {
            "projection_fallback": True,
            "projection_candidates": [int(value) for value in candidates],
            "projection_anchor_count": int(len(anchor_indices)),
        }
        if len(anchor_indices) < self.cfg.weak_min_inliers:
            pose = self._localize_frame_deep(frame, prior_counters=self._frame_counters)
            self._last_info.update(fallback, projection_reason="too_few_local_anchors")
            return pose

        previous, last = self._projection_history
        try:
            predicted_tcw, prediction_scale = extrapolate_tcw(
                previous.tcw,
                last.tcw,
                previous.stamp,
                last.stamp,
                self._frame_capture_stamp,
                self.projection_max_scale,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            pose = self._localize_frame_deep(frame, prior_counters=self._frame_counters)
            self._last_info.update(fallback, projection_reason=f"prediction:{exc}")
            return pose

        projection_t0 = time.perf_counter()
        visible_rows, projected_xy = project_visible_landmarks(
            self.landmark_sidecar.xyz[anchor_indices], predicted_tcw, self.cam
        )
        visible_landmarks = anchor_indices[visible_rows]
        projection_ms = (time.perf_counter() - projection_t0) * 1000.0
        fallback.update(
            {
                "projection_prediction_scale": prediction_scale,
                "projection_visible_count": int(len(visible_landmarks)),
                "projection_project_ms": projection_ms,
            }
        )
        if len(visible_landmarks) < self.cfg.weak_min_inliers:
            pose = self._localize_frame_deep(frame, prior_counters=self._frame_counters)
            self._last_info.update(fallback, projection_reason="too_few_visible_anchors")
            return pose

        xfeat = self.ensure_xfeat()
        feature_t0 = time.perf_counter()
        query_features = extract_xfeat(xfeat, frame, self.cfg.xfeat_topk_track)
        self._frame_counters["xfeat_extract_count"] = 1
        query_device = _to_device_feats(query_features)
        query_keypoints = query_features["keypoints"].detach().cpu().numpy()
        query_descriptor = F.normalize(query_device["descriptors"].float(), dim=-1)
        query_cache = {
            "q_dev": query_device,
            "qkp": query_keypoints,
            "q_nn_desc": query_descriptor,
        }
        feature_ms = (time.perf_counter() - feature_t0) * 1000.0

        landmark_descriptors = self.landmark_sidecar.descriptor_tensor(DEVICE)
        match_ms = 0.0
        attempts = []
        best_inliers = 0
        best_reproj = None
        for radius in self.projection_search_radii:
            match_t0 = time.perf_counter()
            radius_matches = match_projected_landmarks(
                projected_xy,
                visible_landmarks,
                query_keypoints,
                query_descriptor,
                landmark_descriptors,
                radii=(float(radius),),
                min_score=self.projection_min_score,
                ratio=self.projection_ratio,
            )
            match_ms += (time.perf_counter() - match_t0) * 1000.0
            landmark_ids, query_ids, scores = radius_matches[float(radius)]
            attempt = {"radius_px": float(radius), "matches": int(len(landmark_ids))}
            if len(landmark_ids) < self.cfg.weak_min_inliers:
                attempt.update({"inliers": 0, "reproj_rms": None})
                attempts.append(attempt)
                continue
            points2d = query_keypoints[query_ids].astype(np.float64)
            points3d = self.landmark_sidecar.xyz[landmark_ids].astype(np.float64)
            pnp_t0 = time.perf_counter()
            ret = self._pnp(points2d, points3d)
            pnp_ms = (time.perf_counter() - pnp_t0) * 1000.0
            inliers = 0 if ret is None else int(ret.get("num_inliers", 0))
            reproj = None if ret is None else self._estimate_reproj_rms(ret, points2d, points3d)
            attempt.update({"inliers": inliers, "reproj_rms": reproj, "pnp_ms": pnp_ms})
            attempts.append(attempt)
            if inliers > best_inliers:
                best_inliers, best_reproj = inliers, reproj
            info = {
                "mode": "TRACK",
                "next_mode": "TRACK",
                "candidates": [int(value) for value in candidates],
                "used_refs": [int(value) for value in candidates],
                "composite_stage": "projection_guided",
                "matcher_mode": "projection_cosine_ratio",
                "projection_radius_px": float(radius),
                "projection_prediction_scale": prediction_scale,
                "projection_anchor_count": int(len(anchor_indices)),
                "projection_visible_count": int(len(visible_landmarks)),
                "projection_match_count": int(len(landmark_ids)),
                "projection_project_ms": projection_ms,
                "feature_ms": feature_ms,
                "match_ms": match_ms,
                "pnp_ms": pnp_ms,
                "corr3d": int(len(landmark_ids)),
                "raw_matches": int(len(landmark_ids)),
                "inliers": inliers,
                "reproj_rms": reproj,
                "unique_query_inliers": inliers,
                "projection_score_mean": float(scores.mean()) if len(scores) else None,
            }
            ok, weak, pose = self._gate(ret, info, "TRACK")
            strong = (
                ok
                and not weak
                and pose is not None
                and inliers >= self.cfg.adaptive_accept_inliers
                and (reproj is None or reproj <= self.cfg.adaptive_accept_reproj)
            )
            if not strong:
                continue
            mask = _ret_inlier_mask(ret, len(points3d))
            if mask is not None:
                self._last_inl_2d = points2d[mask].astype(np.float32)
                self._last_inl_3d = points3d[mask].astype(np.float32)
            self._publish_success(pose, info, weak=False, source_mode="TRACK")
            self._age_temporal_cache()
            info.update(
                {
                    "accepted": True,
                    "weak": False,
                    "projection_fallback": False,
                    "projection_attempts": attempts,
                    "next_mode": self.state.mode,
                    "temporal_cache_updated": False,
                    "temporal_cache_size_after": int(len(self.temporal_cache)),
                    "total_ms": (time.perf_counter() - start) * 1000.0,
                    **self._frame_counters,
                }
            )
            self._last_info = info
            return pose

        fallback.update(
            {
                "projection_reason": "strong_gate_not_met",
                "projection_attempts": attempts,
                "projection_best_inliers": int(best_inliers),
                "projection_best_reproj_rms": best_reproj,
                "projection_feature_ms": feature_ms,
                "projection_match_ms": match_ms,
            }
        )
        counters = dict(self._frame_counters)
        pose = self._localize_frame_deep(frame, q_cache=query_cache, prior_counters=counters)
        self._last_info.update(fallback)
        return pose

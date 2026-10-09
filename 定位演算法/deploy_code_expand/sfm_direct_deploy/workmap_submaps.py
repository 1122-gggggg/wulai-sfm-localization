"""Submaps of the online workmap: Sim(3), anchor pairs, consistency checks, registry.

Submap 0 is the anchored map (GlueMap coordinates, positive IDs frozen, plus the session
points built while anchored). A detached or independent submap lives in its own local
frame until multi-frame anchor pairs confirm a Sim(3) ``X_G = s R X + t`` into submap 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ANCHOR_MIN_COUNT = 8
ANCHOR_MIN_SPAN = 8
ANCHOR_MAX_SPAN = 48
#: Local PnP inliers a frame needs before it can contribute an anchor pair.
ANCHOR_MIN_LOCAL_INLIERS = 30
ANCHOR_CENTER_TOL_U = 0.03
ANCHOR_ROT_TOL_DEG = 2.0
ANCHOR_PAIR_ROT_TOL_DEG = 1.0
ANCHOR_PAIR_TRANS_TOL_U = 0.02
SCALE_BOUNDS = (0.5, 2.0)

#: Submap states. Submap 0 is permanently ``anchored``.
STATES = ("anchored", "active", "relocalizing", "merging", "merged", "unmerged")
KINDS = ("detached", "independent")


def _center(pose: np.ndarray) -> np.ndarray:
    return -pose[:, :3].T @ pose[:, 3]


def _rotvec_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Rotation vector of ``rotation``, exact for tiny angles (OpenCV zeroes those)."""
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / 2.0
    sine = float(np.linalg.norm(skew))
    angle = float(np.arctan2(sine, cosine))
    if sine > 1e-12 and angle < np.pi - 1e-3:
        return skew * (angle / sine)
    if cosine > 0.0:
        return skew
    # Within 1e-3 rad of a half turn: the axis comes from the symmetric part.
    diagonal = np.sqrt(np.clip((np.diag(rotation) + 1.0) / 2.0, 0.0, None))
    axis = diagonal * np.where(skew >= 0.0, 1.0, -1.0)
    return axis / max(float(np.linalg.norm(axis)), 1e-12) * angle


def _matrix_from_rotvec(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    skew = np.array(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]
    )
    if angle < 1e-8:
        return np.eye(3) + skew + skew @ skew / 2.0
    return (
        np.eye(3)
        + np.sin(angle) / angle * skew
        + (1.0 - np.cos(angle)) / angle**2 * skew @ skew
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(_rotvec_from_matrix(rotation))))


def fit_similarity(
    local: list[tuple[np.ndarray, np.ndarray]], mapped: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Sim(3) local world -> map world from camera poses.

    Each pose is ``(R cam_from_world, camera center)``. Rotation comes from the camera
    orientations (well determined), scale and translation from the centres. Returns
    ``(s, R, t, rms centre residual)`` with ``x_map = s R x_local + t``.
    """
    stacked = sum(m[0].T @ l[0] for l, m in zip(local, mapped, strict=True))
    u, _, vt = np.linalg.svd(stacked)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    cl = np.array([l[1] for l in local])
    cm = np.array([m[1] for m in mapped])
    ql = (rotation @ (cl - cl.mean(0)).T).T
    qm = cm - cm.mean(0)
    scale = float((ql * qm).sum() / max((ql * ql).sum(), 1e-12))
    translation = cm.mean(0) - scale * rotation @ cl.mean(0)
    residual = float(np.sqrt(np.mean(np.sum((scale * (rotation @ cl.T).T + translation - cm) ** 2, 1))))
    return scale, rotation, translation, residual


def map_pose(pose_local: np.ndarray, similarity: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    """Local ``cam_from_world`` -> map ``cam_from_world`` under ``x_map = s R x_local + t``."""
    scale, rotation, translation = similarity
    r_cam = pose_local[:, :3] @ rotation.T
    center = scale * rotation @ _center(pose_local) + translation
    return np.hstack([r_cam, (-r_cam @ center).reshape(3, 1)])


@dataclass(frozen=True, eq=False)
class Sim3:
    """``X_G = scale * rotation @ X + translation``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def identity(cls) -> "Sim3":
        return cls(1.0, np.eye(3), np.zeros(3))

    def compose(self, other: "Sim3") -> "Sim3":
        """``self`` applied after ``other``."""
        return Sim3(
            self.scale * other.scale,
            self.rotation @ other.rotation,
            self.scale * self.rotation @ other.translation + self.translation,
        )

    def inverse(self) -> "Sim3":
        inverse_rotation = self.rotation.T
        return Sim3(
            1.0 / self.scale, inverse_rotation, -(inverse_rotation @ self.translation) / self.scale
        )

    def apply_points(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        return self.scale * points @ self.rotation.T + self.translation

    def apply_pose(self, cam_from_world: np.ndarray) -> np.ndarray:
        return map_pose(
            np.asarray(cam_from_world, dtype=np.float64)[:3, :4],
            (self.scale, self.rotation, self.translation),
        )

    def to_vector(self) -> np.ndarray:
        """``[rotvec(3), t(3), log s]``."""
        return np.concatenate(
            [_rotvec_from_matrix(self.rotation), self.translation, [np.log(self.scale)]]
        )

    @classmethod
    def from_vector(cls, vector: Any) -> "Sim3":
        v = np.asarray(vector, dtype=np.float64).reshape(7)
        rotation = _matrix_from_rotvec(v[:3])
        return cls(float(np.exp(v[6])), rotation, v[3:6].copy())

    def rotation_deg(self) -> float:
        return _rotation_angle_deg(self.rotation)


@dataclass(frozen=True, eq=False)
class AnchorPair:
    """One frame seen both in GlueMap coordinates and in a submap's local frame.

    ``map_pose`` and ``local_pose`` are 3x4 ``cam_from_world``; ``ids`` and ``xy`` are the
    local PnP inliers (submap landmark IDs and pixels) of that frame.
    """

    ordinal: int
    map_pose: np.ndarray
    local_pose: np.ndarray
    ids: np.ndarray
    xy: np.ndarray


def fit_anchor_similarity(anchors: list[AnchorPair]) -> tuple[Sim3, float]:
    """Sim(3) local -> GlueMap from anchor camera poses, and its RMS centre residual."""
    local = [(a.local_pose[:, :3], _center(a.local_pose)) for a in anchors]
    mapped = [(a.map_pose[:, :3], _center(a.map_pose)) for a in anchors]
    scale, rotation, translation, residual = fit_similarity(local, mapped)
    return Sim3(scale, rotation, translation), residual


def _relative(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera ``first`` -> camera ``second`` rotation and translation of two ``cam_from_world``."""
    rotation = second[:, :3] @ first[:, :3].T
    return rotation, second[:, 3] - rotation @ first[:, 3]


def anchor_consistency(
    anchors: list[AnchorPair],
    sim: Sim3,
    *,
    scale_bounds: tuple[float, float] | None = SCALE_BOUNDS,
) -> str | None:
    """None when every anchor and every neighbouring pair agrees with ``sim``.

    ``scale_bounds=None`` skips the scale range check, for a monocular submap whose scale
    is only fixed by the anchors themselves.
    """
    ordered = sorted(anchors, key=lambda a: a.ordinal)
    if (
        len(ordered) < ANCHOR_MIN_COUNT
        or ordered[-1].ordinal - ordered[0].ordinal < ANCHOR_MIN_SPAN
    ):
        return "insufficient_anchors"
    if not np.isfinite(sim.scale) or (
        scale_bounds is not None and not scale_bounds[0] <= sim.scale <= scale_bounds[1]
    ):
        return "scale_bounds"
    mapped = [sim.apply_pose(a.local_pose) for a in ordered]
    for pose, anchor in zip(mapped, ordered, strict=True):
        if float(np.linalg.norm(_center(pose) - _center(anchor.map_pose))) > ANCHOR_CENTER_TOL_U:
            return "center_residual"
        rotation = pose[:, :3] @ anchor.map_pose[:, :3].T
        if _rotation_angle_deg(rotation) > ANCHOR_ROT_TOL_DEG:
            return "rotation_residual"
    for index in range(len(ordered) - 1):
        rotation_g, translation_g = _relative(ordered[index].map_pose, ordered[index + 1].map_pose)
        rotation_s, translation_s = _relative(mapped[index], mapped[index + 1])
        # DeltaT = (T_ij^G)^-1 T_ij^S' with S' the scaled local relative pose.
        delta_rotation = rotation_g.T @ rotation_s
        delta_translation = rotation_g.T @ (translation_s - translation_g)
        if _rotation_angle_deg(delta_rotation) > ANCHOR_PAIR_ROT_TOL_DEG:
            return "pair_rotation"
        if float(np.linalg.norm(delta_translation)) > ANCHOR_PAIR_TRANS_TOL_U:
            return "pair_translation"
    return None


class SubmapRegistry:
    """IDs, states, applied Sim(3), anchors and landmark/keyframe labels of the submaps."""

    def __init__(self) -> None:
        self._records: dict[int, dict[str, Any]] = {
            0: {
                "id": 0, "kind": "anchored", "parent": None, "state": "anchored",
                "opened_ordinal": 0, "applied": Sim3.identity(),
            }
        }
        self._anchors: dict[int, list[AnchorPair]] = {0: []}
        self._landmark_submap: dict[int, int] = {}
        self._keyframe_submap: dict[str, int] = {}
        self._landmark_cache: dict[int, np.ndarray] = {}
        self.active_id = 0
        self.relocalizing_id: int | None = None
        self.awaiting_id: int | None = None
        self._next_id = 1

    def open(self, kind: str, parent_id: int | None, ordinal: int) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown submap kind {kind!r}")
        if parent_id is not None and parent_id not in self._records:
            raise KeyError(parent_id)
        submap_id = self._next_id
        self._next_id += 1
        self._records[submap_id] = {
            "id": submap_id, "kind": kind, "parent": parent_id, "state": "active",
            "opened_ordinal": int(ordinal), "applied": None,
        }
        self._anchors[submap_id] = []
        return submap_id

    def get(self, submap_id: int) -> dict[str, Any]:
        return self._records[submap_id]

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, submap_id: object) -> bool:
        return submap_id in self._records

    def has_state(self, state: str) -> bool:
        return any(record["state"] == state for record in self._records.values())

    def is_unmerged(self, landmark_id: int) -> bool:
        """Whether a session landmark belongs to a submap that gave up merging."""
        return self._records[self.submap_of(landmark_id)]["state"] == "unmerged"

    def set_active(self, submap_id: int) -> None:
        if submap_id not in self._records:
            raise KeyError(submap_id)
        self.active_id = submap_id

    def set_state(self, submap_id: int, state: str) -> None:
        if state not in STATES or submap_id == 0:
            raise ValueError(f"invalid state {state!r} for submap {submap_id}")
        self._records[submap_id]["state"] = state
        if state == "relocalizing":
            self.relocalizing_id = submap_id
        elif self.relocalizing_id == submap_id:
            self.relocalizing_id = None
        # Anchor collection ends before stitch commits. Keep owning the local tracks
        # through both phases so they cannot leak into the map-coordinate PnP.
        if state in {"relocalizing", "merging"}:
            self.awaiting_id = submap_id
        elif self.awaiting_id == submap_id:
            self.awaiting_id = None

    def set_applied(self, submap_id: int, sim: Sim3) -> None:
        self._records[submap_id]["applied"] = sim

    def add_anchor(self, submap_id: int, anchor: AnchorPair) -> None:
        self._anchors[submap_id].append(anchor)

    def anchors(self, submap_id: int) -> list[AnchorPair]:
        return self._anchors[submap_id]

    def drop_oldest_anchor(self, submap_id: int) -> None:
        if self._anchors[submap_id]:
            del self._anchors[submap_id][0]

    def clear_anchors(self, submap_id: int) -> None:
        self._anchors[submap_id].clear()

    def tag_keyframe(self, name: str, submap_id: int) -> None:
        self._keyframe_submap[str(name)] = int(submap_id)

    def keyframe_submap(self, name: str) -> int:
        return self._keyframe_submap.get(str(name), 0)

    def tag_landmarks(self, ids: Any, submap_id: int) -> None:
        touched = {self._landmark_submap.get(int(key), 0) for key in ids}
        for key in ids:
            self._landmark_submap[int(key)] = int(submap_id)
        for affected in touched | {int(submap_id)}:
            self._landmark_cache.pop(affected, None)

    def submap_of(self, landmark_id: int) -> int:
        return self._landmark_submap.get(int(landmark_id), 0)

    def landmark_ids(self, submap_id: int) -> np.ndarray:
        """Sorted int64 IDs tagged to ``submap_id`` (rebuilt only when tags change)."""
        cached = self._landmark_cache.get(submap_id)
        if cached is None:
            cached = np.array(
                sorted(key for key, owner in self._landmark_submap.items() if owner == submap_id),
                dtype=np.int64,
            )
            self._landmark_cache[submap_id] = cached
        return cached

    def records(self) -> list[dict[str, Any]]:
        out = []
        for submap_id, record in sorted(self._records.items()):
            applied = record["applied"]
            out.append(
                {
                    "id": submap_id,
                    "kind": record["kind"],
                    "parent": record["parent"],
                    "state": record["state"],
                    "opened_ordinal": record["opened_ordinal"],
                    "applied": None if applied is None else [float(v) for v in applied.to_vector()],
                    "anchors": len(self._anchors[submap_id]),
                    "landmarks": int(len(self.landmark_ids(submap_id))),
                }
            )
        return out

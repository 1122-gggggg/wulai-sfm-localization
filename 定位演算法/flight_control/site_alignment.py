"""Measured map-to-site Sim(3) alignment.

Frame convention is explicit throughout this module::

    p_site = scale * R_site_from_map @ p_map + t_site_from_map

The solver is intentionally independent of UI and flight code so a calibration
can be produced, reviewed, and verified without connecting to an aircraft.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = "sfm-site-alignment/v1"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _vec3(value: object, label: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric 3-vector") from exc
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite numeric 3-vector")
    return tuple(float(item) for item in array)


def _points(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(tuple(value), dtype=float)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric 3-vectors") from exc
    if array.ndim != 2 or array.shape[1:] != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite Nx3 array")
    return array


def _rotation(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric 3x3 matrix") from exc
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} must be orthonormal")
    if not math.isclose(float(np.linalg.det(array)), 1.0, abs_tol=1e-8):
        raise ValueError(f"{label} must be right-handed (det=+1)")
    return array


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ControlPoint:
    map_point: tuple[float, float, float]
    site_point: tuple[float, float, float]
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_point", _vec3(self.map_point, "map_point"))
        object.__setattr__(self, "site_point", _vec3(self.site_point, "site_point"))
        if self.label is not None:
            object.__setattr__(self, "label", _identifier(self.label, "control point label"))

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "map": list(self.map_point),
            "site": list(self.site_point),
        }
        if self.label is not None:
            value["label"] = self.label
        return value


@dataclass(frozen=True, slots=True)
class SiteFromMapTransform:
    scale: float
    rotation_site_from_map: tuple[tuple[float, float, float], ...]
    translation_site_from_map: tuple[float, float, float]

    def __post_init__(self) -> None:
        scale = float(self.scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        rotation = _rotation(self.rotation_site_from_map, "R_site_from_map")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(
            self,
            "rotation_site_from_map",
            tuple(tuple(float(value) for value in row) for row in rotation),
        )
        object.__setattr__(
            self,
            "translation_site_from_map",
            _vec3(self.translation_site_from_map, "t_site_from_map"),
        )

    @property
    def R(self) -> np.ndarray:
        return np.asarray(self.rotation_site_from_map, dtype=float)

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.translation_site_from_map, dtype=float)

    def map_to_site_point(self, point: object) -> np.ndarray:
        return self.scale * (self.R @ np.asarray(_vec3(point, "map point"))) + self.t

    def site_to_map_point(self, point: object) -> np.ndarray:
        return self.R.T @ ((np.asarray(_vec3(point, "site point")) - self.t) / self.scale)

    def map_to_site_vector(self, vector: object) -> np.ndarray:
        return self.scale * (self.R @ np.asarray(_vec3(vector, "map vector")))

    def site_to_map_vector(self, vector: object) -> np.ndarray:
        return self.R.T @ (np.asarray(_vec3(vector, "site vector")) / self.scale)

    def map_to_site_rotation(self, rotation_map_from_local: object) -> np.ndarray:
        return self.R @ _rotation(rotation_map_from_local, "R_map_from_local")

    def site_to_map_rotation(self, rotation_site_from_local: object) -> np.ndarray:
        return self.R.T @ _rotation(rotation_site_from_local, "R_site_from_local")

    def to_dict(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "R_site_from_map": [list(row) for row in self.rotation_site_from_map],
            "t_site_from_map": list(self.translation_site_from_map),
        }


@dataclass(frozen=True, slots=True)
class AlignmentQuality:
    residuals_m: tuple[float, ...]
    rmse_m: float
    max_error_m: float
    singular_values: tuple[float, float, float]
    rank: int
    reflection_corrected: bool

    def __post_init__(self) -> None:
        residuals = tuple(float(value) for value in self.residuals_m)
        singular = tuple(float(value) for value in self.singular_values)
        numeric = (*residuals, self.rmse_m, self.max_error_m, *singular)
        if (
            not residuals
            or len(singular) != 3
            or not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in numeric)
        ):
            raise ValueError("alignment quality must contain finite non-negative values")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or not 2 <= self.rank <= 3:
            raise ValueError("alignment rank must be 2 or 3")
        if not isinstance(self.reflection_corrected, bool):
            raise ValueError("reflection_corrected must be boolean")
        object.__setattr__(self, "residuals_m", residuals)
        object.__setattr__(self, "rmse_m", float(self.rmse_m))
        object.__setattr__(self, "max_error_m", float(self.max_error_m))
        object.__setattr__(self, "singular_values", singular)

    def to_dict(self) -> dict[str, object]:
        return {
            "residuals_m": list(self.residuals_m),
            "rmse_m": self.rmse_m,
            "max_error_m": self.max_error_m,
            "singular_values": list(self.singular_values),
            "rank": self.rank,
            "reflection_corrected": self.reflection_corrected,
        }


@dataclass(frozen=True, slots=True)
class AlignmentFit:
    control_points: tuple[ControlPoint, ...]
    transform: SiteFromMapTransform
    quality: AlignmentQuality


def solve_similarity_alignment(
    map_points: Iterable[Iterable[float]],
    site_points: Iterable[Iterable[float]],
    *,
    labels: Iterable[str | None] | None = None,
) -> AlignmentFit:
    """Solve a proper, positive-scale Umeyama alignment."""
    map_array = _points(map_points, "map_points")
    site_array = _points(site_points, "site_points")
    if len(map_array) != len(site_array) or len(map_array) < 3:
        raise ValueError("alignment requires at least three paired control points")
    if len(np.unique(map_array, axis=0)) != len(map_array):
        raise ValueError("map control points must be unique")
    if len(np.unique(site_array, axis=0)) != len(site_array):
        raise ValueError("site control points must be unique")

    map_centered = map_array - map_array.mean(axis=0)
    site_centered = site_array - site_array.mean(axis=0)
    map_rank = int(np.linalg.matrix_rank(map_centered))
    site_rank = int(np.linalg.matrix_rank(site_centered))
    rank = min(map_rank, site_rank)
    if rank < 2:
        raise ValueError("control points are degenerate or collinear")

    covariance = site_centered.T @ map_centered / len(map_array)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.ones(3)
    reflection_corrected = float(np.linalg.det(left @ right_t)) < 0.0
    if reflection_corrected:
        correction[-1] = -1.0
    rotation = left @ np.diag(correction) @ right_t
    map_variance = float(np.mean(np.sum(map_centered * map_centered, axis=1)))
    scale = float(np.dot(singular_values, correction) / map_variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("control points do not admit a positive-scale proper rotation")
    translation = site_array.mean(axis=0) - scale * rotation @ map_array.mean(axis=0)
    predicted = (scale * (rotation @ map_array.T)).T + translation
    residuals = np.linalg.norm(predicted - site_array, axis=1)

    label_values = tuple(labels) if labels is not None else (None,) * len(map_array)
    if len(label_values) != len(map_array):
        raise ValueError("labels must match the number of control points")
    controls = tuple(
        ControlPoint(tuple(map_point), tuple(site_point), label)
        for map_point, site_point, label in zip(map_array, site_array, label_values)
    )
    transform = SiteFromMapTransform(
        scale,
        tuple(tuple(float(value) for value in row) for row in rotation),
        tuple(float(value) for value in translation),
    )
    quality = AlignmentQuality(
        tuple(float(value) for value in residuals),
        float(np.sqrt(np.mean(residuals * residuals))),
        float(np.max(residuals)),
        tuple(float(value) for value in singular_values),
        rank,
        reflection_corrected,
    )
    return AlignmentFit(controls, transform, quality)


def solve_rigid_alignment(
    map_points: Iterable[Iterable[float]],
    site_points: Iterable[Iterable[float]],
    *,
    labels: Iterable[str | None] | None = None,
) -> AlignmentFit:
    """Solve a proper rigid alignment with scale fixed exactly to one."""
    map_array = _points(map_points, "map_points")
    site_array = _points(site_points, "site_points")
    if len(map_array) != len(site_array) or len(map_array) < 3:
        raise ValueError("alignment requires at least three paired control points")
    if len(np.unique(map_array, axis=0)) != len(map_array):
        raise ValueError("map control points must be unique")
    if len(np.unique(site_array, axis=0)) != len(site_array):
        raise ValueError("site control points must be unique")

    map_centered = map_array - map_array.mean(axis=0)
    site_centered = site_array - site_array.mean(axis=0)
    rank = min(
        int(np.linalg.matrix_rank(map_centered)),
        int(np.linalg.matrix_rank(site_centered)),
    )
    if rank < 2:
        raise ValueError("control points are degenerate or collinear")

    covariance = site_centered.T @ map_centered / len(map_array)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.ones(3)
    reflection_corrected = float(np.linalg.det(left @ right_t)) < 0.0
    if reflection_corrected:
        correction[-1] = -1.0
    rotation = left @ np.diag(correction) @ right_t
    translation = site_array.mean(axis=0) - rotation @ map_array.mean(axis=0)
    predicted = (rotation @ map_array.T).T + translation
    residuals = np.linalg.norm(predicted - site_array, axis=1)

    label_values = tuple(labels) if labels is not None else (None,) * len(map_array)
    if len(label_values) != len(map_array):
        raise ValueError("labels must match the number of control points")
    controls = tuple(
        ControlPoint(tuple(map_point), tuple(site_point), label)
        for map_point, site_point, label in zip(map_array, site_array, label_values)
    )
    transform = SiteFromMapTransform(
        1.0,
        tuple(tuple(float(value) for value in row) for row in rotation),
        tuple(float(value) for value in translation),
    )
    quality = AlignmentQuality(
        tuple(float(value) for value in residuals),
        float(np.sqrt(np.mean(residuals * residuals))),
        float(np.max(residuals)),
        tuple(float(value) for value in singular_values),
        rank,
        reflection_corrected,
    )
    return AlignmentFit(controls, transform, quality)


@dataclass(frozen=True, slots=True)
class SiteAlignment:
    map_frame_id: str
    site_frame_id: str
    control_points: tuple[ControlPoint, ...]
    transform: SiteFromMapTransform
    quality: AlignmentQuality
    approved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_frame_id", _identifier(self.map_frame_id, "map_frame_id"))
        object.__setattr__(self, "site_frame_id", _identifier(self.site_frame_id, "site_frame_id"))
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be boolean")
        if len(self.control_points) < 3:
            raise ValueError("site alignment needs at least three control points")
        if len(self.control_points) != len(self.quality.residuals_m):
            raise ValueError("quality residual count does not match control points")

    @classmethod
    def from_fit(
        cls,
        *,
        map_frame_id: str,
        site_frame_id: str,
        fit: AlignmentFit,
        approved: bool = False,
    ) -> "SiteAlignment":
        return cls(
            map_frame_id,
            site_frame_id,
            fit.control_points,
            fit.transform,
            fit.quality,
            approved,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "map_frame_id": self.map_frame_id,
            "site_frame_id": self.site_frame_id,
            "approved": self.approved,
            "control_points": [point.to_dict() for point in self.control_points],
            "transform": self.transform.to_dict(),
            "quality": self.quality.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _strict_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return value


def site_alignment_from_dict(
    value: object,
    *,
    expected_map_frame_id: str | None = None,
    expected_site_frame_id: str | None = None,
) -> SiteAlignment:
    raw = _strict_keys(
        value,
        {"schema", "map_frame_id", "site_frame_id", "approved", "control_points", "transform", "quality"},
        "site alignment",
    )
    if raw["schema"] != SCHEMA:
        raise ValueError(f"site alignment schema must be {SCHEMA}")
    map_frame_id = _identifier(raw["map_frame_id"], "map_frame_id")
    site_frame_id = _identifier(raw["site_frame_id"], "site_frame_id")
    if expected_map_frame_id is not None and map_frame_id != expected_map_frame_id:
        raise ValueError("site alignment map frame does not match the active map")
    if expected_site_frame_id is not None and site_frame_id != expected_site_frame_id:
        raise ValueError("site alignment site frame does not match the active site")
    if not isinstance(raw["approved"], bool):
        raise ValueError("site alignment approved must be boolean")
    if not isinstance(raw["control_points"], list):
        raise ValueError("site alignment control_points must be a list")
    controls = []
    for index, item in enumerate(raw["control_points"]):
        if not isinstance(item, dict) or set(item) not in ({"map", "site"}, {"map", "site", "label"}):
            raise ValueError(f"control_points[{index}] is malformed")
        controls.append(ControlPoint(item["map"], item["site"], item.get("label")))
    transform = _strict_keys(
        raw["transform"],
        {"scale", "R_site_from_map", "t_site_from_map"},
        "site alignment transform",
    )
    persisted_transform = SiteFromMapTransform(
        transform["scale"],
        transform["R_site_from_map"],
        transform["t_site_from_map"],
    )
    quality = _strict_keys(
        raw["quality"],
        {"residuals_m", "rmse_m", "max_error_m", "singular_values", "rank", "reflection_corrected"},
        "site alignment quality",
    )
    persisted_quality = AlignmentQuality(
        quality["residuals_m"],
        quality["rmse_m"],
        quality["max_error_m"],
        quality["singular_values"],
        quality["rank"],
        quality["reflection_corrected"],
    )

    def matches(candidate: AlignmentFit) -> bool:
        return bool(
            math.isclose(
                persisted_transform.scale,
                candidate.transform.scale,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and np.allclose(
                persisted_transform.R, candidate.transform.R, rtol=1e-9, atol=1e-9
            )
            and np.allclose(
                persisted_transform.t, candidate.transform.t, rtol=1e-9, atol=1e-9
            )
            and persisted_quality == candidate.quality
        )

    solvers = [solve_similarity_alignment]
    if math.isclose(persisted_transform.scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
        solvers.insert(0, solve_rigid_alignment)
    recomputed = next(
        (
            candidate
            for solver in solvers
            if matches(
                candidate := solver(
                    (point.map_point for point in controls),
                    (point.site_point for point in controls),
                    labels=(point.label for point in controls),
                )
            )
        ),
        None,
    )
    if recomputed is None:
        raise ValueError("persisted site alignment does not match its control points")
    return SiteAlignment.from_fit(
        map_frame_id=map_frame_id,
        site_frame_id=site_frame_id,
        fit=recomputed,
        approved=raw["approved"],
    )


def load_site_alignment(
    path: str | Path,
    *,
    expected_map_frame_id: str | None = None,
    expected_site_frame_id: str | None = None,
) -> SiteAlignment:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read site alignment {source}: {exc}") from exc
    return site_alignment_from_dict(
        raw,
        expected_map_frame_id=expected_map_frame_id,
        expected_site_frame_id=expected_site_frame_id,
    )


def save_site_alignment(alignment: SiteAlignment, path: str | Path) -> Path:
    if not isinstance(alignment, SiteAlignment):
        raise TypeError("alignment must be a SiteAlignment")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(alignment.to_json(), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


__all__ = [
    "AlignmentFit",
    "AlignmentQuality",
    "ControlPoint",
    "SCHEMA",
    "SiteAlignment",
    "SiteFromMapTransform",
    "load_site_alignment",
    "save_site_alignment",
    "site_alignment_from_dict",
    "solve_rigid_alignment",
    "solve_similarity_alignment",
]

"""Uncertainty-aware local GlueMap reference selection and ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pose_guided.config import PoseGuidedConfig, SearchStageConfig
from pose_guided.types import PosePrediction


@dataclass(frozen=True)
class SearchLimits:
    radius: float
    max_yaw_rad: float
    max_refs: int
    mahalanobis_threshold: float
    hard_max_radius: float
    stage: str


@dataclass(frozen=True)
class RankedReference:
    index: int
    name: str
    score: float
    distance: float
    view_angle_rad: float
    mahalanobis_sq: float | None


def search_limits(
    config: PoseGuidedConfig,
    *,
    state: str,
    misses: int,
    weak_after: int,
    base_radius: float,
    base_yaw_rad: float,
    base_max_refs: int,
) -> SearchLimits:
    stage_name = "TRACK"
    if state == "LOST":
        stage_name = "LOST"
    elif state == "WEAK_TRACK":
        stage_name = "WEAK_TRACK_2" if misses >= weak_after + max(1, weak_after) else "WEAK_TRACK"
    stage = config.search.get(stage_name) or config.search.get(state) or SearchStageConfig(1.0, 1.0, 1.0)
    radius = max(1e-6, float(base_radius) * stage.radius_factor)
    if (
        config.enabled
        and config.uncertainty.hard_max_radius_factor > 0.0
    ):
        hard = float(base_radius) * config.uncertainty.hard_max_radius_factor
    else:
        hard = radius
    max_refs = max(1, int(math.ceil(float(base_max_refs) * stage.max_refs_factor)))
    return SearchLimits(
        radius=radius,
        max_yaw_rad=float(base_yaw_rad) * stage.yaw_factor,
        max_refs=max_refs,
        mahalanobis_threshold=config.uncertainty.mahalanobis_threshold,
        hard_max_radius=max(radius, hard),
        stage=stage_name,
    )


def viewing_angle(predicted_yaw: float | None, reference_yaw: float | None) -> float | None:
    if predicted_yaw is None or reference_yaw is None:
        return None
    if not math.isfinite(predicted_yaw) or not math.isfinite(reference_yaw):
        return None
    pred = np.array([math.cos(predicted_yaw), math.sin(predicted_yaw), 0.0])
    ref = np.array([math.cos(reference_yaw), math.sin(reference_yaw), 0.0])
    cosine = float(np.clip(np.dot(pred, ref), -1.0, 1.0))
    return math.acos(cosine)


def mahalanobis_sq(delta: np.ndarray, covariance: np.ndarray) -> float | None:
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return None
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        return None
    value = float(delta @ inverse @ delta)
    return value if math.isfinite(value) else None


def validated_score_row(name: str, values: object, count: int) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, dict):
        row = np.full(count, np.nan, dtype=float)
        for key, value in values.items():
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
                raise ValueError(f"{name} keys must be integer row indices")
            index = int(key)
            if index < 0 or index >= count:
                raise ValueError(f"{name} index {index} is outside row count {count}")
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
                raise ValueError(f"{name}[{index}] must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name}[{index}] must be finite")
            row[index] = number
        return row
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape[0] != count:
        raise ValueError(f"{name} must be row-aligned with names, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _aligned_quality_length(
    quality: np.ndarray | None,
    stability: np.ndarray | None,
) -> int | None:
    length = None if quality is None else int(quality.shape[0])
    if stability is None:
        return length
    stability_length = int(stability.shape[0])
    if length is None:
        return stability_length
    if stability_length != length:
        raise ValueError("quality and stability rows must have the same length")
    return length


def _accumulate_quality_row(
    total: np.ndarray,
    weight: np.ndarray,
    values: np.ndarray | None,
) -> None:
    if values is None:
        return
    finite = np.isfinite(values)
    total[finite] += values[finite]
    weight[finite] += 1.0


def _mean_quality_rows(
    quality: np.ndarray | None,
    stability: np.ndarray | None,
    length: int,
) -> np.ndarray:
    total = np.zeros(length, dtype=float)
    weight = np.zeros(length, dtype=float)
    _accumulate_quality_row(total, weight, quality)
    _accumulate_quality_row(total, weight, stability)
    combined = np.full(length, np.nan, dtype=float)
    usable = weight > 0.0
    combined[usable] = total[usable] / weight[usable]
    return combined


def _rescale_quality_unit_interval(combined: np.ndarray) -> np.ndarray:
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return np.array(combined, dtype=float, copy=True)
    max_value = float(np.max(finite))
    min_value = float(np.min(finite))
    calibrated = np.array(combined, dtype=float, copy=True)
    if max_value > 1.0 or min_value < 0.0:
        span = max_value - min_value
        if span > 0.0:
            calibrated = (calibrated - min_value) / span
        else:
            calibrated = np.where(np.isfinite(calibrated), 0.0, calibrated)
    return np.clip(calibrated, 0.0, 1.0)


def _coverage_quality_floor(coverage_count: int, coverage_ref: int) -> float:
    if coverage_ref > 0 and coverage_count < coverage_ref:
        return 1.0 - (float(coverage_count) / float(coverage_ref))
    return 0.0


def calibrate_reference_quality(
    quality: np.ndarray | None,
    stability: np.ndarray | None,
    *,
    coverage_count: int,
    coverage_ref: int,
) -> dict[int, float] | None:
    length = _aligned_quality_length(quality, stability)
    if length is None:
        return None
    combined = _mean_quality_rows(quality, stability, length)
    if not np.any(np.isfinite(combined)):
        return None
    calibrated = _rescale_quality_unit_interval(combined)
    floor = _coverage_quality_floor(coverage_count, coverage_ref)
    calibrated = np.where(np.isfinite(calibrated), np.maximum(calibrated, floor), calibrated)
    return {
        index: float(value)
        for index, value in enumerate(calibrated)
        if math.isfinite(float(value))
    }




def _rank_reference(
    *,
    config: PoseGuidedConfig,
    name: str,
    index: int,
    deltas: np.ndarray,
    distances: np.ndarray,
    yaws: np.ndarray | None,
    prediction: PosePrediction,
    limits: SearchLimits,
    last: set[int],
    covis: set[int],
    quality_scores: dict[int, float] | None,
) -> RankedReference | None:
    distance = float(distances[index])
    if not math.isfinite(distance):
        return None
    if distance > limits.hard_max_radius:
        return None
    view = viewing_angle(
        prediction.yaw,
        None if yaws is None else float(yaws[index]),
    )
    if view is not None and view > limits.max_yaw_rad:
        return None
    maha = None
    if prediction.position_covariance is not None:
        maha = mahalanobis_sq(deltas[index], prediction.position_covariance)
        if maha is not None and maha > limits.mahalanobis_threshold and distance > limits.radius:
            return None
    elif distance > limits.radius and index not in last and index not in covis:
        return None
    height_ok = True
    if config.height_axis is not None:
        height_ok = abs(float(deltas[index][config.height_axis])) <= limits.radius
    if not height_ok:
        return None
    score = _score(
        config,
        distance=distance,
        radius=limits.radius,
        view=view,
        last=index in last,
        covisible=index in covis,
        quality=None if quality_scores is None else quality_scores.get(index),
    )
    return RankedReference(
        index=index,
        name=name,
        score=score,
        distance=distance,
        view_angle_rad=0.0 if view is None else view,
        mahalanobis_sq=maha,
    )


def select_and_rank(
    *,
    config: PoseGuidedConfig,
    names: list[str],
    centers: np.ndarray,
    yaws: np.ndarray | None,
    prediction: PosePrediction,
    limits: SearchLimits,
    last_indices: list[int],
    covisible_indices: list[int],
    quality_scores: object = None,
    stability_scores: object = None,
) -> list[RankedReference]:
    if not prediction.valid or prediction.position is None:
        return []
    predicted = np.asarray(prediction.position, dtype=float).reshape(3)
    if not np.isfinite(predicted).all():
        return []
    deltas = np.asarray(centers, dtype=float) - predicted[None, :]
    distances = np.linalg.norm(deltas, axis=1)
    last = set(last_indices)
    covis = set(covisible_indices)
    count = len(names)
    quality_row = validated_score_row("quality_scores", quality_scores, count)
    stability_row = validated_score_row("stability_scores", stability_scores, count)
    nearby = int(
        np.count_nonzero(np.isfinite(distances) & (distances <= limits.hard_max_radius))
    )
    calibrated = calibrate_reference_quality(
        quality_row,
        stability_row,
        coverage_count=nearby,
        coverage_ref=max(int(limits.max_refs), 1),
    )
    ranked: list[RankedReference] = []
    for index, name in enumerate(names):
        item = _rank_reference(
            config=config,
            name=name,
            index=index,
            deltas=deltas,
            distances=distances,
            yaws=yaws,
            prediction=prediction,
            limits=limits,
            last=last,
            covis=covis,
            quality_scores=calibrated,
        )
        if item is not None:
            ranked.append(item)
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[: limits.max_refs]



def _score(
    config: PoseGuidedConfig,
    *,
    distance: float,
    radius: float,
    view: float | None,
    last: bool,
    covisible: bool,
    quality: float | None,
) -> float:
    position_score = math.exp(-distance / max(radius, 1e-6))
    orientation_score = 0.5 if view is None else math.exp(-view / math.radians(45.0))
    temporal_score = 1.0 if last else 0.0
    covis_score = 1.0 if covisible else 0.0
    quality_score = 0.0 if quality is None else float(quality)
    weights = config.ranking
    return (
        weights.w_position * position_score
        + weights.w_orientation * orientation_score
        + weights.w_covisibility * covis_score
        + weights.w_temporal * temporal_score
        + weights.w_quality * quality_score
    )

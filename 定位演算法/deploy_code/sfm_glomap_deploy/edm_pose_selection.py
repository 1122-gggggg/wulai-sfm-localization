"""Consensus and pose-candidate selection for ProductionEDMTracker."""
from __future__ import annotations

import math
import os
from typing import Any, Callable

import numpy as np


CONSENSUS_MODES = frozenset({"pairwise", "cluster"})
_LAST_CONSENSUS: dict[str, Any] = {
    "mode": "pairwise",
    "reason": "none",
    "strong_count": 0,
    "cluster_sizes": (),
    "outlier_count": 0,
    "chosen": None,
}


def acquire_consensus_limit(cfg: Any) -> float:
    return float(cfg.acquire_max_jump_factor) * float(cfg.max_jump)


def candidate_inliers(candidate: tuple) -> int:
    return int(candidate[0]["num_inliers"])


def last_consensus_info() -> dict[str, Any]:
    """Deterministic telemetry from the most recent candidate decision."""
    return dict(_LAST_CONSENSUS)


def count_agreeing_refs(
    scored: list[tuple[str, tuple]],
    center: Any,
    min_inliers: int,
    limit: float,
) -> int:
    """Independent references whose PnP center lands within ``limit`` of ``center``.

    Counts every scored candidate at or above ``min_inliers`` inliers, so the
    chosen reference counts itself. Used by the relaxed LOST acquire tier: a
    lone weak reference is exactly the shape of a wrong-place lock, two
    references landing on the same center is not.
    """
    anchor = np.asarray(center, dtype=float).reshape(3)
    floor = int(min_inliers)
    bound = float(limit)
    agreeing = 0
    for _name, candidate in scored:
        if candidate_inliers(candidate) < floor:
            continue
        other = _pose_center(candidate[0])
        if float(np.linalg.norm(other - anchor)) <= bound:
            agreeing += 1
    return agreeing


def consensus_mode(cfg: Any) -> str:
    mode = getattr(cfg, "pose_consensus_mode", "pairwise")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    if not isinstance(mode, str):
        raise ValueError("pose_consensus_mode must be 'pairwise' or 'cluster'")
    mode = mode.strip()
    if mode not in CONSENSUS_MODES:
        raise ValueError("pose_consensus_mode must be 'pairwise' or 'cluster'")
    return mode


def consensus_rotation_limit_deg(cfg: Any) -> float:
    value = getattr(cfg, "consensus_max_rotation_deg", None)
    if value is None:
        value = getattr(cfg, "acquire_max_yaw_diff_deg", 90.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("consensus_max_rotation_deg must be finite and > 0")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("consensus_max_rotation_deg must be finite and > 0")
    return number


def _pose_center(result: Any) -> np.ndarray:
    transform = result["cam_from_world"]
    rotation = transform.rotation.matrix()
    return -rotation.T @ np.asarray(transform.translation)


def _pose_rotation(result: Any) -> np.ndarray | None:
    try:
        matrix = np.asarray(result["cam_from_world"].rotation.matrix(), dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    return matrix


def _rotation_geodesic_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = first.T @ second
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _reproj_rank(value: Any) -> float:
    if value is None:
        return -math.inf
    number = float(value)
    if not math.isfinite(number):
        return -math.inf
    return -number


def _metric_rank(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return -math.inf
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -math.inf
    if not math.isfinite(number):
        return -math.inf
    return number


def _composite_metric(value: Any, missing: float) -> float:
    """Finite non-negative metric for the composite score; missing/garbage -> missing."""
    if value is None or isinstance(value, bool):
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(number):
        return missing
    return max(0.0, number)


def _rank_key(selection: Any) -> Callable[[tuple[str, tuple]], tuple]:
    refs = list(getattr(selection, "refs", ()) or ())
    vpr_index = {name: index for index, name in enumerate(refs)}
    last = len(vpr_index)
    # SFM_EDM_COMPOSITE_POSE_RANK=1: rank by S = n_inliers / (1 + 0.5*rms) * (cells/64)**0.5 * (1 + 0.2*ratio), vpr index tiebreak.
    if os.environ.get("SFM_EDM_COMPOSITE_POSE_RANK", "0") == "1":
        def composite_key(item: tuple[str, tuple]) -> tuple[float, int]:
            name, candidate = item
            metrics = candidate[3]
            rms = _composite_metric(metrics.get("reproj_rms"), math.inf)
            ratio = _composite_metric(metrics.get("inlier_ratio"), 0.0)
            cells = _composite_metric(metrics.get("inlier_grid_cells"), 0.0)
            score = (
                float(candidate_inliers(candidate))
                / (1.0 + 0.5 * rms)
                * (cells / 64.0) ** 0.5
                * (1.0 + 0.2 * ratio)
            )
            return (score, -int(vpr_index.get(name, last)))

        return composite_key


    def key(item: tuple[str, tuple]) -> tuple[int, float, float, float, int]:
        name, candidate = item
        metrics = candidate[3]
        return (
            candidate_inliers(candidate),
            _reproj_rank(metrics.get("reproj_rms")),
            _metric_rank(metrics.get("inlier_ratio")),
            _metric_rank(metrics.get("inlier_grid_cells")),
            -int(vpr_index.get(name, last)),
        )

    return key


def _strong_items(
    scored: list[tuple[str, tuple]],
    min_inl: int,
) -> list[tuple[str, tuple]]:
    return [
        (name, candidate)
        for name, candidate in scored
        if candidate_inliers(candidate) >= min_inl
    ]


def _strong_centers_disagree(
    scored: list[tuple[str, tuple]],
    min_inl: int,
    cfg: Any,
) -> bool:
    centers: list[np.ndarray] = []
    for _name, candidate in scored:
        if candidate_inliers(candidate) < min_inl:
            continue
        centers.append(_pose_center(candidate[0]))
    if len(centers) < 2:
        return False
    limit = acquire_consensus_limit(cfg)
    for index, center in enumerate(centers):
        for other in centers[index + 1 :]:
            if float(np.linalg.norm(center - other)) > limit:
                return True
    return False


def _cluster_strong_centers(
    strong: list[tuple[str, tuple]],
    center_limit: float,
) -> list[list[tuple[str, tuple]]]:
    """Group strong candidates by camera-center distance only (pairwise rule).

    Pairwise mode deliberately ignores rotation: two fixes that put the camera
    in the same place agree even if their yaw differs (yaw is re-gated
    downstream by the track-yaw slack). Single-link union-find, same shape as
    _cluster_strong.
    """
    count = len(strong)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    centers = [_pose_center(candidate[0]) for _name, candidate in strong]
    for index in range(count):
        for other in range(index + 1, count):
            if float(np.linalg.norm(centers[index] - centers[other])) <= center_limit:
                parent[find(index)] = find(other)
    groups: dict[int, list[tuple[str, tuple]]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(strong[index])
    return list(groups.values())


def _poses_agree(
    first: tuple,
    second: tuple,
    center_limit: float,
    rotation_limit_deg: float,
) -> bool:
    rotation_a = _pose_rotation(first[0])
    rotation_b = _pose_rotation(second[0])
    if rotation_a is None or rotation_b is None:
        return False
    if float(np.linalg.norm(_pose_center(first[0]) - _pose_center(second[0]))) > center_limit:
        return False
    return _rotation_geodesic_deg(rotation_a, rotation_b) <= rotation_limit_deg


def _cluster_strong(
    strong: list[tuple[str, tuple]],
    center_limit: float,
    rotation_limit_deg: float,
) -> list[list[tuple[str, tuple]]]:
    count = len(strong)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index in range(count):
        for other in range(index + 1, count):
            if _poses_agree(strong[index][1], strong[other][1], center_limit, rotation_limit_deg):
                parent[find(index)] = find(other)
    groups: dict[int, list[tuple[str, tuple]]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(strong[index])
    return list(groups.values())


def _record_consensus(
    *,
    mode: str,
    reason: str,
    strong_count: int,
    cluster_sizes: tuple[int, ...],
    outlier_count: int,
    chosen: str | None,
) -> None:
    _LAST_CONSENSUS["mode"] = mode
    _LAST_CONSENSUS["reason"] = reason
    _LAST_CONSENSUS["strong_count"] = int(strong_count)
    _LAST_CONSENSUS["cluster_sizes"] = tuple(sorted(cluster_sizes, reverse=True))
    _LAST_CONSENSUS["outlier_count"] = int(outlier_count)
    _LAST_CONSENSUS["chosen"] = chosen


def _pick_ranked(
    pool: list[tuple[str, tuple]],
    selection: Any,
) -> tuple[str, tuple] | None:
    if not pool:
        return None
    return max(pool, key=_rank_key(selection))


def _select_from_clusters(
    selection: Any,
    scored: list[tuple[str, tuple]],
    cfg: Any,
    *,
    prefer_acquire_top1: bool,
) -> tuple[str, tuple] | None:
    mode = consensus_mode(cfg)
    min_inl = int(selection.min_inliers)
    strong = _strong_items(scored, min_inl)
    if mode == "pairwise":
        if _strong_centers_disagree(scored, min_inl, cfg):
            # No veto-all: elect the largest center-cluster and drop the
            # minority as outliers. A pure 1v1 tie still fails closed.
            clusters = _cluster_strong_centers(strong, acquire_consensus_limit(cfg))
            sizes = tuple(len(cluster) for cluster in clusters)
            biggest = max(sizes) if sizes else 0
            winners = [cluster for cluster in clusters if len(cluster) == biggest]
            outliers = sum(len(cluster) for cluster in clusters if len(cluster) < biggest)
            if len(winners) != 1:
                _record_consensus(
                    mode=mode,
                    reason="pairwise_conflict",
                    strong_count=len(strong),
                    cluster_sizes=tuple(1 for _ in strong),
                    outlier_count=0,
                    chosen=None,
                )
                return None
            chosen = _pick_ranked(winners[0], selection)
            _record_consensus(
                mode=mode,
                reason="pairwise_majority",
                strong_count=len(strong),
                cluster_sizes=sizes,
                outlier_count=outliers,
                chosen=None if chosen is None else chosen[0],
            )
            return chosen
        if prefer_acquire_top1:
            top1_name = selection.refs[0] if selection.refs else None
            by_name = {name: candidate for name, candidate in scored}
            top1 = None if top1_name is None else by_name.get(top1_name)
            if top1 is not None and candidate_inliers(top1) >= min_inl:
                _record_consensus(
                    mode=mode,
                    reason="acquire_top1",
                    strong_count=len(strong),
                    cluster_sizes=(len(strong),) if strong else (),
                    outlier_count=0,
                    chosen=top1_name,
                )
                return top1_name, top1
        chosen = _pick_ranked(strong or scored, selection)
        _record_consensus(
            mode=mode,
            reason="ranked",
            strong_count=len(strong),
            cluster_sizes=(len(strong),) if strong else (),
            outlier_count=0,
            chosen=None if chosen is None else chosen[0],
        )
        return chosen

    if not scored:
        _record_consensus(
            mode=mode,
            reason="empty",
            strong_count=0,
            cluster_sizes=(),
            outlier_count=0,
            chosen=None,
        )
        return None
    if len(strong) <= 1:
        chosen = _pick_ranked(strong or scored, selection)
        _record_consensus(
            mode=mode,
            reason="ranked",
            strong_count=len(strong),
            cluster_sizes=(len(strong),) if strong else (),
            outlier_count=0,
            chosen=None if chosen is None else chosen[0],
        )
        return chosen

    clusters = _cluster_strong(
        strong,
        acquire_consensus_limit(cfg),
        consensus_rotation_limit_deg(cfg),
    )
    sizes = tuple(len(cluster) for cluster in clusters)
    max_size = max(sizes)
    winners = [cluster for cluster in clusters if len(cluster) == max_size]
    outliers = sum(len(cluster) for cluster in clusters if len(cluster) < max_size)
    if len(winners) != 1:
        _record_consensus(
            mode=mode,
            reason="cluster_conflict",
            strong_count=len(strong),
            cluster_sizes=sizes,
            outlier_count=outliers,
            chosen=None,
        )
        return None
    chosen = _pick_ranked(winners[0], selection)
    _record_consensus(
        mode=mode,
        reason="cluster_majority" if outliers else "ranked",
        strong_count=len(strong),
        cluster_sizes=sizes,
        outlier_count=outliers,
        chosen=None if chosen is None else chosen[0],
    )
    return chosen


def _select_acquire_candidate(
    selection: Any,
    scored: list[tuple[str, tuple]],
    cfg: Any,
) -> tuple[str, tuple] | None:
    return _select_from_clusters(selection, scored, cfg, prefer_acquire_top1=True)


def _select_track_candidate(
    selection: Any,
    scored: list[tuple[str, tuple]],
    cfg: Any,
) -> tuple[str, tuple] | None:
    if not scored:
        _record_consensus(
            mode=consensus_mode(cfg),
            reason="empty",
            strong_count=0,
            cluster_sizes=(),
            outlier_count=0,
            chosen=None,
        )
        return None
    return _select_from_clusters(selection, scored, cfg, prefer_acquire_top1=False)

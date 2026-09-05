"""Anchor acceptance reuses ProductionEDMTracker quality fields. No second metric set."""

from __future__ import annotations

import math
from typing import Mapping

from pose_guided.config import PoseGuidedConfig


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe_anchor_state(pose_result: Mapping[str, object], allow_weak_anchors: bool) -> bool:
    if pose_result.get("ok") is not True:
        return False
    if pose_result.get("rejected"):
        return False
    # Predicted-only records never pass; KLT-bridged frames carry no fresh EDM
    # match, so they must not become visual anchors either (drift would latch).
    if pose_result.get("pose_status") in ("PREDICTED_ONLY", "KLT_BRIDGED"):
        return False
    if pose_result.get("candidate_mode") == "klt_bridge" or pose_result.get("bridge"):
        return False
    state_in = str(pose_result.get("state_in") or "")
    if state_in == "WEAK_TRACK" and not allow_weak_anchors:
        return False
    if state_in == "LOST" and not allow_weak_anchors:
        return False
    return True


def _safe_anchor_metrics(pose_result: Mapping[str, object], config: PoseGuidedConfig) -> bool:
    inliers = pose_result.get("inliers")
    if isinstance(inliers, bool) or not isinstance(inliers, (int, float)):
        return False
    if int(inliers) < config.anchor.min_inliers:
        return False
    ratio = _finite_number(pose_result.get("inlier_ratio"))
    if ratio is None or ratio < config.anchor.min_inlier_ratio:
        return False
    cells = pose_result.get("inlier_grid_cells")
    if isinstance(cells, bool) or not isinstance(cells, (int, float)):
        return False
    if int(cells) < config.anchor.min_inlier_grid_cells:
        return False
    reproj = _finite_number(pose_result.get("reproj_rms"))
    if reproj is None or reproj > config.anchor.max_reproj_rms:
        return False
    return True


def is_safe_visual_anchor(pose_result: Mapping[str, object], config: PoseGuidedConfig) -> bool:
    """True only for a high-confidence visually confirmed fix.

    Requires the tracker to have already accepted the pose (`ok`) and to meet
    acquire-level gates. WEAK_TRACK / LOST fixes are refused unless explicitly
    allowed. Predicted-only records never pass.
    """
    return _safe_anchor_state(pose_result, config.allow_weak_anchors) and _safe_anchor_metrics(
        pose_result, config
    )

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from edm_pose_selection import (
    _select_acquire_candidate,
    _select_track_candidate,
    consensus_mode,
    last_consensus_info,
)
from pose_guided.config import PoseGuidedConfig, RankingConfig, load_pose_guided_config
from pose_guided.controller import PoseGuidedController
from pose_guided.quality import is_safe_visual_anchor
from pose_guided.reference_selector import (
    calibrate_reference_quality,
    search_limits,
    select_and_rank,
    validated_score_row,
)
from pose_guided.se3 import quaternion_wxyz_from_matrix, transform_from_rt
from pose_guided.types import (
    Confirmation,
    FusedOdometrySample,
    OdometryFrame,
    PosePrediction,
    PropagationMode,
)
from pose_guided.visual_anchor import VisualAnchorManager
from reposed_motion_validator import (
    RePoseDMotionValidator,
    match_spatial_support,
    rank_and_cap_matches,
)


def _enabled_config(**overrides) -> PoseGuidedConfig:
    config = load_pose_guided_config()
    values = dict(config.__dict__)
    values["enabled"] = True
    values.update(overrides)
    return PoseGuidedConfig(**values)


def _visual_ok(**fields) -> dict:
    result = {
        "ok": True,
        "state_in": "TRACK",
        "inliers": 90,
        "inlier_ratio": 0.4,
        "inlier_grid_cells": 8,
        "reproj_rms": 1.2,
    }
    result.update(fields)
    return result


def test_default_config_is_disabled() -> None:
    from pathlib import Path
    import json

    path = (
        Path(__file__).resolve().parents[3]
        / "定位演算法"
        / "configs"
        / "pose_guided_localization.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["enabled"] is False
    assert raw["camera_center_is_body_origin"] is True
    assert raw["camera_to_body"]["approved"] is False


def test_anchor_only_from_strong_visual() -> None:
    config = _enabled_config()
    assert is_safe_visual_anchor(_visual_ok(), config)
    assert not is_safe_visual_anchor(_visual_ok(ok=False), config)
    assert not is_safe_visual_anchor(_visual_ok(inliers=20), config)
    assert not is_safe_visual_anchor(_visual_ok(state_in="WEAK_TRACK"), config)
    assert not is_safe_visual_anchor(_visual_ok(pose_status="PREDICTED_ONLY"), config)
    # KLT-bridged frames must not become visual anchors (drift would latch),
    # even if they arrive with ok=True and TRACK state.
    assert not is_safe_visual_anchor(_visual_ok(pose_status="KLT_BRIDGED"), config)
    assert not is_safe_visual_anchor(_visual_ok(candidate_mode="klt_bridge"), config)
    assert not is_safe_visual_anchor(_visual_ok(bridge=True), config)


def test_visual_anchor_resets_old_odom_drift() -> None:
    manager = VisualAnchorManager()
    manager.update(
        timestamp=1.0,
        rotation_cam_from_world=np.eye(3),
        center=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
        odom_pose=transform_from_rt(np.eye(3), [100.0, 0.0, 0.0]),
    )
    first = manager.anchor
    assert first is not None
    manager.update(
        timestamp=2.0,
        rotation_cam_from_world=np.eye(3),
        center=np.array([1.0, 0.0, 0.0]),
        yaw=0.0,
        odom_pose=transform_from_rt(np.eye(3), [0.0, 0.0, 0.0]),
    )
    assert manager.anchor is not None
    assert manager.anchor.timestamp == 2.0
    assert manager.anchor.map_center == pytest.approx([1.0, 0.0, 0.0])
    assert manager.anchor.odom_pose[:3, 3] == pytest.approx([0.0, 0.0, 0.0])


def test_orientation_prior_applies_ned_yaw_increment_only() -> None:
    controller = PoseGuidedController(_enabled_config())
    controller.observe_fused(
        FusedOdometrySample.from_anafi(timestamp=1.0, roll=0.0, pitch=0.0, yaw=0.0)
    )
    controller.maybe_update_anchor(
        _visual_ok(),
        timestamp=1.0,
        rotation_cam_from_world=np.eye(3),
        center=np.array([0.0, 0.0, 0.0]),
        yaw=0.2,
    )
    controller.observe_fused(
        FusedOdometrySample.from_anafi(timestamp=1.1, roll=0.0, pitch=0.0, yaw=math.pi / 2.0)
    )
    prediction = controller.predict(
        1.1,
        visual_center=np.array([0.0, 0.0, 0.0]),
        visual_yaw=0.2,
        visual_velocity=None,
        visual_stamp=1.0,
    )
    assert prediction.valid
    assert prediction.propagation_mode is PropagationMode.ORIENTATION_PRIOR
    assert prediction.confirmation is Confirmation.PREDICTED_ONLY
    # NED +90° CW-from-North is -90° in map CCW-from-East.
    assert prediction.yaw == pytest.approx(0.2 - math.pi / 2.0, abs=1e-6)
    assert prediction.position == pytest.approx([0.0, 0.0, 0.0])


def test_ned_velocity_is_not_applied_to_map_position() -> None:
    controller = PoseGuidedController(_enabled_config())
    controller.observe_fused(
        FusedOdometrySample.from_anafi(
            timestamp=1.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            speed_north=5.0,
            speed_east=0.0,
            speed_down=0.0,
        )
    )
    controller.maybe_update_anchor(
        _visual_ok(),
        timestamp=1.0,
        rotation_cam_from_world=np.eye(3),
        center=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
    )
    controller.observe_fused(
        FusedOdometrySample.from_anafi(
            timestamp=1.1,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            speed_north=5.0,
            speed_east=0.0,
            speed_down=0.0,
        )
    )
    prediction = controller.predict(
        1.1,
        visual_center=np.array([0.0, 0.0, 0.0]),
        visual_yaw=0.0,
        visual_velocity=None,
        visual_stamp=1.0,
    )
    assert prediction.position == pytest.approx([0.0, 0.0, 0.0])


def test_map_aligned_fused_pose_is_used_when_explicitly_labeled() -> None:
    controller = PoseGuidedController(_enabled_config())
    rotation = np.eye(3)
    quat = tuple(float(value) for value in quaternion_wxyz_from_matrix(rotation))
    start = FusedOdometrySample(
        timestamp=1.0,
        frame=OdometryFrame.MAP,
        position=(0.0, 0.0, 0.0),
        quaternion_wxyz=quat,
        source="test",
    )
    controller.observe_fused(start)
    controller.maybe_update_anchor(
        _visual_ok(),
        timestamp=1.0,
        rotation_cam_from_world=rotation,
        center=np.array([3.0, 4.0, 0.0]),
        yaw=0.0,
    )
    moved = FusedOdometrySample(
        timestamp=1.1,
        frame=OdometryFrame.MAP,
        position=(1.0, 0.0, 0.0),
        quaternion_wxyz=quat,
        source="test",
    )
    controller.observe_fused(moved)
    prediction = controller.predict(
        1.1,
        visual_center=np.array([3.0, 4.0, 0.0]),
        visual_yaw=0.0,
        visual_velocity=None,
        visual_stamp=1.0,
    )
    assert prediction.propagation_mode is PropagationMode.FUSED_POSE
    assert prediction.position == pytest.approx([4.0, 4.0, 0.0])


def test_search_expands_from_track_to_lost() -> None:
    config = _enabled_config()
    track = search_limits(
        config,
        state="TRACK",
        misses=0,
        weak_after=2,
        base_radius=0.8,
        base_yaw_rad=math.radians(90.0),
        base_max_refs=1,
    )
    weak = search_limits(
        config,
        state="WEAK_TRACK",
        misses=2,
        weak_after=2,
        base_radius=0.8,
        base_yaw_rad=math.radians(90.0),
        base_max_refs=3,
    )
    lost = search_limits(
        config,
        state="LOST",
        misses=8,
        weak_after=2,
        base_radius=0.8,
        base_yaw_rad=math.radians(90.0),
        base_max_refs=5,
    )
    assert track.radius < weak.radius < lost.radius
    assert track.stage == "TRACK"
    assert lost.stage == "LOST"


def test_selector_prefers_nearby_similar_view() -> None:
    config = _enabled_config()
    prediction = PosePrediction(
        timestamp=1.0,
        position=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
        rotation_cam_from_world=None,
        position_covariance=np.eye(3) * 0.04,
        orientation_covariance=0.1,
        age_since_visual_anchor=0.05,
        propagation_mode=PropagationMode.ORIENTATION_PRIOR,
        source="test",
        confidence=0.9,
        valid=True,
    )
    limits = search_limits(
        config,
        state="TRACK",
        misses=0,
        weak_after=2,
        base_radius=1.0,
        base_yaw_rad=math.radians(40.0),
        base_max_refs=2,
    )
    ranked = select_and_rank(
        config=config,
        names=["near", "far", "turned"],
        centers=np.array([[0.1, 0.0, 0.0], [8.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        yaws=np.array([0.0, 0.0, math.pi]),
        prediction=prediction,
        limits=limits,
        last_indices=[],
        covisible_indices=[],
    )
    assert [item.name for item in ranked] == ["near"]


def test_failures_do_not_emit_confirmed_pose() -> None:
    controller = PoseGuidedController(_enabled_config())
    for sample in (
        None,
        FusedOdometrySample.from_anafi(timestamp=float("nan"), roll=0.0, pitch=0.0, yaw=0.0),
    ):
        controller.observe_fused(sample)
    prediction = controller.predict(
        1.0,
        visual_center=np.array([float("nan"), 0.0, 0.0]),
        visual_yaw=float("nan"),
        visual_velocity=None,
        visual_stamp=None,
    )
    assert prediction.valid is False
    assert prediction.confirmation is Confirmation.NONE
    stale = controller.predict(
        10.0,
        visual_center=np.array([0.0, 0.0, 0.0]),
        visual_yaw=0.0,
        visual_velocity=None,
        visual_stamp=1.0,
    )
    # No anchor yet, so this is visual hold, not a confirmed pose.
    assert stale.confirmation is Confirmation.PREDICTED_ONLY
    assert controller.predicted_only(stale)["pose_status"] == "PREDICTED_ONLY"


def test_camera_center_is_treated_as_body_origin() -> None:
    config = load_pose_guided_config()
    assert config.camera_center_is_body_origin is True
    assert config.camera_body_ready is True
    assert config.camera_to_body.approved is False
    assert config.camera_to_body.translation_m == (0.0, 0.0, 0.0)
    assert config.metres_per_map_unit is None
    assert config.height_axis is None


def test_imu_rotates_visual_velocity_after_visual_loss() -> None:
    controller = PoseGuidedController(_enabled_config())
    controller.observe_fused(
        FusedOdometrySample.from_anafi(timestamp=1.0, roll=0.0, pitch=0.0, yaw=0.0)
    )
    controller.maybe_update_anchor(
        _visual_ok(),
        timestamp=1.0,
        rotation_cam_from_world=np.eye(3),
        center=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
    )
    controller.observe_fused(
        FusedOdometrySample.from_anafi(timestamp=2.0, roll=0.0, pitch=0.0, yaw=math.pi / 2.0)
    )
    prediction = controller.predict(
        2.0,
        visual_center=np.array([0.0, 0.0, 0.0]),
        visual_yaw=0.0,
        visual_velocity=np.array([1.0, 0.0, 0.0]),
        visual_stamp=1.0,
    )
    assert prediction.valid
    assert prediction.confirmation is Confirmation.PREDICTED_ONLY
    assert prediction.propagation_mode is PropagationMode.VELOCITY_INTEGRATION
    # +90° NED CW is -90° map CCW, so +X visual velocity becomes -Y.
    assert prediction.position == pytest.approx([0.0, -1.0, 0.0])
    assert prediction.yaw == pytest.approx(-math.pi / 2.0, abs=1e-6)


def test_ned_speed_ratio_scales_visual_velocity() -> None:
    controller = PoseGuidedController(_enabled_config())
    controller.observe_fused(
        FusedOdometrySample.from_anafi(
            timestamp=1.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            speed_north=1.0,
            speed_east=0.0,
            speed_down=0.0,
        )
    )
    controller.maybe_update_anchor(
        _visual_ok(),
        timestamp=1.0,
        rotation_cam_from_world=np.eye(3),
        center=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
    )
    controller.observe_fused(
        FusedOdometrySample.from_anafi(
            timestamp=2.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            speed_north=2.0,
            speed_east=0.0,
            speed_down=0.0,
        )
    )
    prediction = controller.predict(
        2.0,
        visual_center=np.array([0.0, 0.0, 0.0]),
        visual_yaw=0.0,
        visual_velocity=np.array([1.0, 0.0, 0.0]),
        visual_stamp=1.0,
    )
    assert prediction.propagation_mode is PropagationMode.VELOCITY_INTEGRATION
    assert prediction.position == pytest.approx([2.0, 0.0, 0.0])


def test_calibrated_gnss_prior_recovers_map_center_for_relocalization() -> None:
    controller = PoseGuidedController(_enabled_config())
    earth_radius_m = 6_378_137.0
    latitude0 = 25.033
    longitude0 = 121.5654

    def sample(timestamp: float, east_m: float, north_m: float) -> FusedOdometrySample:
        latitude = latitude0 + math.degrees(north_m / earth_radius_m)
        longitude = longitude0 + math.degrees(
            east_m / (earth_radius_m * math.cos(math.radians(latitude0)))
        )
        fused = FusedOdometrySample.from_anafi(
            timestamp=timestamp,
            gps_timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude=18.0,
            latitude_accuracy=0.3,
            longitude_accuracy=0.3,
            altitude_accuracy=0.6,
        )
        assert fused is not None
        return fused

    for index, (east_m, north_m) in enumerate(
        ((0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0), (8.0, 0.0), (0.0, 8.0))
    ):
        timestamp = 100.0 + index
        controller.observe_fused(sample(timestamp, east_m, north_m))
        assert controller.maybe_update_anchor(
            _visual_ok(),
            timestamp=timestamp,
            rotation_cam_from_world=np.eye(3),
            center=np.array([2.0 + 0.1 * east_m, 4.0 + 0.1 * north_m, 7.0]),
            yaw=0.2,
        )

    controller.observe_fused(sample(200.0, 6.0, 3.0))
    prediction = controller.predict(
        200.0,
        visual_center=np.array([2.0, 4.0, 7.0]),
        visual_yaw=0.2,
        visual_velocity=None,
        visual_stamp=105.0,
        allow_gnss_prior=True,
    )

    assert prediction.valid
    assert prediction.propagation_mode is PropagationMode.GNSS_PRIOR
    assert prediction.position == pytest.approx([2.6, 4.3, 7.0], abs=1e-4)


class _Rotation:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = np.asarray(matrix, dtype=float)

    def matrix(self) -> np.ndarray:
        return self._matrix


class _Transform:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.rotation = _Rotation(rotation)
        self.translation = np.asarray(translation, dtype=float)


def _rz(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _pose_cfg(**overrides) -> SimpleNamespace:
    values = {
        "acquire_max_jump_factor": 2.0,
        "max_jump": 2.0,
        "acquire_max_yaw_diff_deg": 90.0,
        "pose_consensus_mode": "pairwise",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _pose_selection(refs: list[str], min_inliers: int = 80) -> SimpleNamespace:
    return SimpleNamespace(refs=refs, min_inliers=min_inliers)


def _pose_candidate(
    center: list[float],
    inliers: int,
    *,
    reproj: float = 1.0,
    ratio: float = 1.0,
    cells: int = 8,
    rotation: np.ndarray | None = None,
) -> tuple:
    matrix = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    translation = -matrix.T @ np.asarray(center, dtype=float)
    return (
        {
            "cam_from_world": _Transform(matrix, translation),
            "num_inliers": inliers,
        },
        np.zeros((inliers, 2)),
        np.zeros((inliers, 3)),
        {
            "reproj_rms": reproj,
            "inlier_ratio": ratio,
            "inlier_grid_cells": cells,
        },
    )


def _prediction() -> PosePrediction:
    return PosePrediction(
        timestamp=1.0,
        position=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
        rotation_cam_from_world=None,
        position_covariance=np.eye(3) * 0.04,
        orientation_covariance=0.1,
        age_since_visual_anchor=0.05,
        propagation_mode=PropagationMode.ORIENTATION_PRIOR,
        source="test",
        confidence=0.9,
        valid=True,
    )


def test_pairwise_consensus_still_vetoes_two_strong_disagreements() -> None:
    selection = _pose_selection(["a", "b"])
    chosen = _select_acquire_candidate(
        selection,
        [
            ("a", _pose_candidate([0.0, 0.0, 0.0], 90)),
            ("b", _pose_candidate([5.0, 0.0, 0.0], 90)),
        ],
        _pose_cfg(),
    )
    assert chosen is None
    info = last_consensus_info()
    assert info["mode"] == "pairwise"
    assert info["reason"] == "pairwise_conflict"


def test_pairwise_consensus_elects_majority_over_singleton_outlier() -> None:
    selection = _pose_selection(["keep_a", "keep_b", "outlier"])
    chosen = _select_track_candidate(
        selection,
        [
            ("keep_a", _pose_candidate([0.0, 0.0, 0.0], 80, ratio=0.4, cells=6)),
            ("keep_b", _pose_candidate([0.2, 0.0, 0.0], 85, ratio=0.5, cells=8)),
            ("outlier", _pose_candidate([8.0, 0.0, 0.0], 400, ratio=0.9, cells=12)),
        ],
        _pose_cfg(),
    )
    # Same geometry as the cluster test, but pairwise ignores rotation and
    # ranks the majority cluster by the standard rank key.
    assert chosen is not None
    assert chosen[0] in ("keep_a", "keep_b")
    info = last_consensus_info()
    assert info["mode"] == "pairwise"
    assert info["reason"] == "pairwise_majority"
    assert info["outlier_count"] == 1


def test_pairwise_acquire_still_keeps_strong_vpr_top1() -> None:
    selection = _pose_selection(["top", "other"])
    chosen = _select_acquire_candidate(
        selection,
        [
            ("top", _pose_candidate([0.0, 0.0, 0.0], 90)),
            ("other", _pose_candidate([0.1, 0.0, 0.0], 200)),
        ],
        _pose_cfg(),
    )
    assert chosen is not None
    assert chosen[0] == "top"
    assert last_consensus_info()["reason"] == "acquire_top1"


def test_cluster_consensus_ignores_singleton_outlier() -> None:
    selection = _pose_selection(["keep_a", "keep_b", "outlier"])
    chosen = _select_track_candidate(
        selection,
        [
            ("keep_a", _pose_candidate([0.0, 0.0, 0.0], 80, ratio=0.4, cells=6)),
            ("keep_b", _pose_candidate([0.2, 0.0, 0.0], 85, ratio=0.5, cells=8)),
            ("outlier", _pose_candidate([8.0, 0.0, 0.0], 400, ratio=0.9, cells=12)),
        ],
        _pose_cfg(pose_consensus_mode="cluster"),
    )
    assert chosen is not None
    assert chosen[0] == "keep_b"
    info = last_consensus_info()
    assert info["mode"] == "cluster"
    assert info["reason"] == "cluster_majority"
    assert info["outlier_count"] == 1


def test_cluster_consensus_uses_rotation_and_fails_closed_on_equal_clusters() -> None:
    selection = _pose_selection(["a1", "a2", "b1", "b2"])
    scored = [
        ("a1", _pose_candidate([0.0, 0.0, 0.0], 90, rotation=_rz(0.0))),
        ("a2", _pose_candidate([0.1, 0.0, 0.0], 88, rotation=_rz(5.0))),
        ("b1", _pose_candidate([0.0, 0.0, 0.0], 91, rotation=_rz(180.0))),
        ("b2", _pose_candidate([0.1, 0.0, 0.0], 89, rotation=_rz(175.0))),
    ]
    assert (
        _select_track_candidate(selection, scored, _pose_cfg(pose_consensus_mode="cluster")) is None
    )
    assert last_consensus_info()["reason"] == "cluster_conflict"


def test_cluster_ranking_uses_vpr_only_as_final_tie_break() -> None:
    selection = _pose_selection(["vpr", "better_ratio"])
    chosen = _select_track_candidate(
        selection,
        [
            ("vpr", _pose_candidate([0.0, 0.0, 0.0], 80, reproj=1.0, ratio=0.2, cells=6)),
            ("better_ratio", _pose_candidate([0.1, 0.0, 0.0], 80, reproj=1.0, ratio=0.8, cells=6)),
        ],
        _pose_cfg(pose_consensus_mode="cluster"),
    )
    assert chosen is not None
    assert chosen[0] == "better_ratio"
    tied = _select_track_candidate(
        selection,
        [
            ("vpr", _pose_candidate([0.0, 0.0, 0.0], 80, reproj=1.0, ratio=0.5, cells=8)),
            ("better_ratio", _pose_candidate([0.1, 0.0, 0.0], 80, reproj=1.0, ratio=0.5, cells=8)),
        ],
        _pose_cfg(pose_consensus_mode="cluster"),
    )
    assert tied is not None
    assert tied[0] == "vpr"


def test_invalid_consensus_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="pose_consensus_mode"):
        consensus_mode(_pose_cfg(pose_consensus_mode="best_effort"))


def test_default_quality_weight_does_not_change_reference_order() -> None:
    config = _enabled_config()
    assert config.ranking.w_quality == 0.0
    limits = search_limits(
        config,
        state="TRACK",
        misses=0,
        weak_after=2,
        base_radius=2.0,
        base_yaw_rad=math.radians(40.0),
        base_max_refs=2,
    )
    kwargs = dict(
        config=config,
        names=["near", "also"],
        centers=np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        yaws=np.array([0.0, 0.0]),
        prediction=_prediction(),
        limits=limits,
        last_indices=[],
        covisible_indices=[],
    )
    plain = [item.name for item in select_and_rank(**kwargs)]
    weighted = [
        item.name
        for item in select_and_rank(
            **kwargs,
            quality_scores=np.array([0.0, 1.0]),
            stability_scores=np.array([0.0, 1.0]),
        )
    ]
    assert plain == ["near", "also"]
    assert weighted == plain


def test_controller_applies_validated_quality_when_weight_enabled() -> None:
    ranking = RankingConfig(1.0, 0.8, 0.4, 0.5, 2.0)
    controller = PoseGuidedController(_enabled_config(ranking=ranking))
    names = controller.select_references(
        names=["low", "high"],
        centers=np.array([[0.15, 0.0, 0.0], [0.16, 0.0, 0.0]]),
        yaws=np.array([0.0, 0.0]),
        prediction=_prediction(),
        state="TRACK",
        misses=0,
        weak_after=2,
        base_radius=2.0,
        base_yaw_rad=math.radians(40.0),
        base_max_refs=2,
        last_indices=[],
        covisible_indices=[],
        quality_scores=np.array([0.1, 0.9]),
        stability_scores=np.array([0.1, 0.9]),
    )
    assert names[0] == "high"


def test_misaligned_quality_rows_fail_closed() -> None:
    controller = PoseGuidedController(_enabled_config())
    with pytest.raises(ValueError, match="row-aligned"):
        controller.select_references(
            names=["only"],
            centers=np.array([[0.0, 0.0, 0.0]]),
            yaws=np.array([0.0]),
            prediction=_prediction(),
            state="TRACK",
            misses=0,
            weak_after=2,
            base_radius=1.0,
            base_yaw_rad=math.radians(40.0),
            base_max_refs=1,
            last_indices=[],
            covisible_indices=[],
            quality_scores=np.array([0.1, 0.2]),
        )
    with pytest.raises(ValueError, match="finite"):
        validated_score_row("quality_scores", np.array([float("nan")]), 1)


def test_quality_calibration_clips_and_keeps_sparse_views() -> None:
    calibrated = calibrate_reference_quality(
        np.array([-2.0, 8.0]),
        None,
        coverage_count=1,
        coverage_ref=4,
    )
    assert calibrated is not None
    assert calibrated[0] == pytest.approx(0.75)
    assert calibrated[1] == pytest.approx(1.0)


def test_rank_and_cap_matches_spreads_across_both_images() -> None:
    dense0 = np.array([[4.0, 4.0]] * 8, dtype=float)
    dense1 = np.array([[4.0, 4.0]] * 8, dtype=float)
    spread0 = np.array([[4.0, 4.0], [900.0, 20.0], [20.0, 500.0], [900.0, 500.0]], dtype=float)
    spread1 = np.array([[900.0, 500.0], [20.0, 500.0], [900.0, 20.0], [4.0, 4.0]], dtype=float)
    mkpts0 = np.vstack((dense0, spread0))
    mkpts1 = np.vstack((dense1, spread1))
    mconf = np.array([0.99] * 8 + [0.4, 0.41, 0.42, 0.43], dtype=float)
    points0, points1, conf = rank_and_cap_matches(
        mkpts0, mkpts1, mconf, 4, width=1024, height=576, grid=4
    )
    assert len(points0) == 4
    assert len(points1) == 4
    assert float(np.max(np.linalg.norm(points0 - np.array([4.0, 4.0]), axis=1))) > 100.0
    assert float(np.min(conf)) < 0.99


def test_reposed_geometry_gates_stay_unavailable() -> None:
    camera = SimpleNamespace(
        model="PINHOLE",
        width=1280,
        height=720,
        params=[900.0, 900.0, 640.0, 360.0],
    )
    validator = RePoseDMotionValidator(
        camera,
        matcher=None,
        model_path=".",
        model_sha256="0" * 64,
        min_inliers=2,
        min_inlier_ratio=0.5,
        min_spatial_support=3,
        match_grid=4,
    )
    geometry = SimpleNamespace(
        pose=SimpleNamespace(R=np.eye(3), t=np.array([0.2, 0.0, 0.1])),
        scale=1.0,
    )
    predicted = np.eye(4)
    predicted[:3, 3] = np.array([0.2, 0.0, 0.1])
    points = np.array([[10.0, 10.0], [12.0, 12.0], [14.0, 14.0]], dtype=float)
    low_ratio = validator._compare_geometry(
        geometry,
        {"num_inliers": 2, "inlier_mask": np.array([True, True, False])},
        predicted,
        8,
        0.0,
        0.0,
        0.0,
        0.0,
        points,
        points + 1.0,
    )
    assert low_ratio.status == "unavailable"
    assert low_ratio.reason == "inlier_ratio"
    clustered = validator._compare_geometry(
        geometry,
        {"num_inliers": 3, "inlier_mask": np.array([True, True, True])},
        predicted,
        3,
        0.0,
        0.0,
        0.0,
        0.0,
        points,
        points + 1.0,
    )
    assert clustered.status == "unavailable"
    assert clustered.reason == "spatial_support"
    assert (
        match_spatial_support(
            points,
            points + 1.0,
            width=1280,
            height=720,
            grid=8,
            mask=np.array([True, True, True]),
        )
        == 1
    )
    with pytest.raises(ValueError, match="min_inlier_ratio"):
        RePoseDMotionValidator(
            camera,
            matcher=None,
            model_path=".",
            model_sha256="0" * 64,
            min_inlier_ratio=1.5,
        )

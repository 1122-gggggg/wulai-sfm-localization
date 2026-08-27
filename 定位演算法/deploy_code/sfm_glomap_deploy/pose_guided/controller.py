"""Facade used by ProductionEDMTracker. Disabled by default: every method no-ops."""

from __future__ import annotations

import numpy as np

from pose_guided.config import PoseGuidedConfig
from pose_guided.fused_state import ImuStateProvider
from pose_guided.propagator import PosePropagator
from pose_guided.quality import is_safe_visual_anchor
from pose_guided.reference_selector import (
    SearchLimits,
    select_and_rank,
    search_limits,
    validated_score_row,
)

from pose_guided.types import (
    Confirmation,
    FusedOdometrySample,
    PosePrediction,
    PropagationMode,
)
from pose_guided.visual_anchor import VisualAnchorManager, odom_pose_from_map_sample


class TrackingStateMachine:
    """Search-stage policy only. Visual BOOT/TRACK/WEAK/LOST stays in the tracker."""

    def limits(
        self,
        config: PoseGuidedConfig,
        *,
        state: str,
        misses: int,
        weak_after: int,
        base_radius: float,
        base_yaw_rad: float,
        base_max_refs: int,
    ) -> SearchLimits:
        return search_limits(
            config,
            state=state,
            misses=misses,
            weak_after=weak_after,
            base_radius=base_radius,
            base_yaw_rad=base_yaw_rad,
            base_max_refs=base_max_refs,
        )


class PoseGuidedController:
    def __init__(self, config: PoseGuidedConfig) -> None:
        self.config = config
        self.provider = ImuStateProvider()
        self.propagator = PosePropagator(config)
        self.anchors = VisualAnchorManager()
        self.policy = TrackingStateMachine()
        self.last_prediction: PosePrediction | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def observe_fused(self, sample: FusedOdometrySample | None) -> None:
        self.provider.observe(sample)

    def reset(self) -> None:
        self.provider.clear()
        self.anchors.clear()
        self.last_prediction = None

    def fused_at(self, timestamp: float) -> FusedOdometrySample | None:
        return self.provider.interpolate(timestamp, max_sync_error_s=self.config.max_sync_error_s)

    def predict(
        self,
        query_timestamp: float,
        *,
        visual_center: np.ndarray | None,
        visual_yaw: float | None,
        visual_velocity: np.ndarray | None,
        visual_stamp: float | None,
    ) -> PosePrediction:
        fused = self.fused_at(query_timestamp)
        prediction = self.propagator.predict(
            query_timestamp=query_timestamp,
            anchor=self.anchors.anchor,
            fused=fused,
            visual_center=visual_center,
            visual_yaw=visual_yaw,
            visual_velocity=visual_velocity,
            visual_stamp=visual_stamp,
        )
        self.last_prediction = prediction
        return prediction

    def maybe_update_anchor(
        self,
        pose_result: dict,
        *,
        timestamp: float,
        rotation_cam_from_world: np.ndarray,
        center: np.ndarray,
        yaw: float,
    ) -> bool:
        if not is_safe_visual_anchor(pose_result, self.config):
            return False
        fused = self.fused_at(timestamp)
        odom_pose = None if fused is None else odom_pose_from_map_sample(fused)
        self.anchors.update(
            timestamp=timestamp,
            rotation_cam_from_world=rotation_cam_from_world,
            center=center,
            yaw=yaw,
            fused=fused,
            odom_pose=odom_pose,
        )
        return True

    def select_references(
        self,
        *,
        names: list[str],
        centers: np.ndarray,
        yaws: np.ndarray | None,
        prediction: PosePrediction,
        state: str,
        misses: int,
        weak_after: int,
        base_radius: float,
        base_yaw_rad: float,
        base_max_refs: int,
        last_indices: list[int],
        covisible_indices: list[int],
        quality_scores: object = None,
        stability_scores: object = None,
    ) -> list[str]:
        if not self.enabled or not prediction.valid:
            return []
        count = len(names)
        quality = validated_score_row("quality_scores", quality_scores, count)
        stability = validated_score_row("stability_scores", stability_scores, count)
        limits = self.policy.limits(
            self.config,
            state=state,
            misses=misses,
            weak_after=weak_after,
            base_radius=base_radius,
            base_yaw_rad=base_yaw_rad,
            base_max_refs=base_max_refs,
        )
        ranked = select_and_rank(
            config=self.config,
            names=names,
            centers=centers,
            yaws=yaws,
            prediction=prediction,
            limits=limits,
            last_indices=last_indices,
            covisible_indices=covisible_indices,
            quality_scores=quality,
            stability_scores=stability,
        )
        return [item.name for item in ranked]


    def search_limits(
        self,
        *,
        state: str,
        misses: int,
        weak_after: int,
        base_radius: float,
        base_yaw_rad: float,
        base_max_refs: int,
    ) -> SearchLimits:
        return self.policy.limits(
            self.config,
            state=state,
            misses=misses,
            weak_after=weak_after,
            base_radius=base_radius,
            base_yaw_rad=base_yaw_rad,
            base_max_refs=base_max_refs,
        )

    @staticmethod
    def predicted_only(prediction: PosePrediction | None) -> dict:
        if prediction is None:
            return {
                "pose_status": Confirmation.NONE.value,
                "prediction_valid": False,
                "prediction_mode": PropagationMode.UNAVAILABLE.value,
            }
        info = prediction.as_info()
        if prediction.valid:
            info["pose_status"] = Confirmation.PREDICTED_ONLY.value
        return info

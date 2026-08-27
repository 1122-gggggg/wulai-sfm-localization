"""IMU / fused-state helpers for pose-guided GlueMap reference selection.

Production default is off. ANAFI fused telemetry is Level D: orientation prior
only. Map-aligned odometry is accepted only when a sample is explicitly labeled
as map-frame; NED velocity is never treated as GlueMap motion.
"""

from pose_guided.config import PoseGuidedConfig, load_pose_guided_config
from pose_guided.controller import PoseGuidedController
from pose_guided.quality import is_safe_visual_anchor
from pose_guided.types import (
    Confirmation,
    FusedOdometrySample,
    PosePrediction,
    PropagationMode,
    VisualAnchor,
)

__all__ = [
    "Confirmation",
    "FusedOdometrySample",
    "PoseGuidedConfig",
    "PoseGuidedController",
    "PosePrediction",
    "PropagationMode",
    "VisualAnchor",
    "is_safe_visual_anchor",
    "load_pose_guided_config",
]

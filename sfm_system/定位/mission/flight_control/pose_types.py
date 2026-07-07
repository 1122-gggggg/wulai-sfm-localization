"""Neutral pose/localizer types shared by the localizer and the flight controller.

Extracted from autoflight.py so the visual localizer (production_xfeat_tracker,
reloc_localizer_xfeat) can import Pose/Localizer WITHOUT pulling in autoflight's
top-level `from plan_path import ...` (the A*/SDF tour planner). autoflight.py
re-exports these names for backward compatibility.
"""
from dataclasses import dataclass


@dataclass
class Pose:
    x: float; y: float; z: float; yaw: float; stamp: float


class Localizer:
    def get_pose(self) -> "Pose | None":
        raise NotImplementedError("plug visual relocalizer (map-frame pose)")

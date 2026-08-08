"""Neutral pose/localizer types shared by the localizer and the flight controller.

Extracted from autoflight.py so the visual localizer (production_xfeat_tracker,
reloc_localizer_xfeat) can import Pose/Localizer WITHOUT pulling in autoflight's
top-level `from plan_path import ...` (the A*/SDF tour planner). autoflight.py
re-exports these names for backward compatibility.
"""
from dataclasses import dataclass


@dataclass
class Pose:
    """Visual-localizer pose crossing the localization/flight-control boundary.

    x/y/z are RAW GLOMAP coordinates. They carry no axis convention: which way is
    up is a per-site measurement held in T_align_gravity.json, applied by
    real_path_follow_controller.MapFrame. ``yaw`` is the heading in that measured
    horizontal plane; ``stamp`` is a time.monotonic() reading, never wall clock.

    Deliberately field-compatible with real_path_follow_controller.Pose, which
    adds an .xyz helper. The two are exchanged by duck typing, so the field names
    and their meaning must stay identical.
    """

    x: float; y: float; z: float; yaw: float; stamp: float


class Localizer:
    def get_pose(self) -> "Pose | None":
        raise NotImplementedError("plug visual relocalizer (map-frame pose)")

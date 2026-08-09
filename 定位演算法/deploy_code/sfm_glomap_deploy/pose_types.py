"""Neutral pose/localizer types shared by the localizer and the flight controller.

Extracted from autoflight.py so the visual localizer (production_xfeat_tracker,
reloc_localizer_xfeat) can import Pose/Localizer WITHOUT pulling in autoflight's
top-level `from plan_path import ...` (the A*/SDF tour planner). autoflight.py
re-exports these names for backward compatibility.
"""

from collections.abc import Callable
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

    x: float
    y: float
    z: float
    yaw: float
    stamp: float


class Localizer:
    def get_pose(self) -> "Pose | None":
        raise NotImplementedError("plug visual relocalizer (map-frame pose)")


@dataclass(frozen=True)
class BuiltLocalizer:
    """Verified production localizer and the assets/configuration it owns."""

    tracker: Localizer
    reloc_map: object
    camera: object
    config: object
    backend: str
    variant: str
    device: str


@dataclass(frozen=True)
class LocalizerCapabilities:
    """Static asset and configuration contract for one localizer backend."""

    name: str
    required_assets: tuple[str, ...] = ()
    optional_assets: tuple[str, ...] = ()
    unsupported_assets: tuple[str, ...] = ()
    supports_production_profile: bool = False


LocalizerBuilder = Callable[..., BuiltLocalizer]


@dataclass(frozen=True)
class LocalizerProvider:
    """Named backend provider registered with the production factory."""

    name: str
    capabilities: LocalizerCapabilities
    builder: LocalizerBuilder | None = None

    def build(self, **kwargs: object) -> BuiltLocalizer:
        if self.builder is None:
            raise RuntimeError(f"localizer provider {self.name!r} has no production builder")
        return self.builder(**kwargs)

"""Onboard SD recording profiles. Live stream stays 720p.

ANAFI white paper v1.4: recording is a separate encoder from the 5 Mb/s live
stream. Only 16:9 modes are listed so a later query_camera scale to 1280x720 is
a single ratio.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecordingProfile:
    profile_id: str
    label: str
    width: int
    height: int
    resolution: str
    framerate: str
    mode: str = "standard"
    hyperlapse: str = "ratio_15"


FHD_30 = RecordingProfile(
    "fhd_30",
    "FHD 1080p30",
    1920,
    1080,
    "res_1080p",
    "fps_30",
)
UHD_4K_30 = RecordingProfile(
    "uhd_4k_30",
    "4K UHD 30",
    3840,
    2160,
    "res_uhd_4k",
    "fps_30",
)
RECORDING_PROFILES: dict[str, RecordingProfile] = {
    FHD_30.profile_id: FHD_30,
    UHD_4K_30.profile_id: UHD_4K_30,
}
DEFAULT_RECORDING_PROFILE = FHD_30


def resolve_recording_profile(profile_id: object) -> RecordingProfile:
    key = str(profile_id or "").strip()
    try:
        return RECORDING_PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported recording profile: {profile_id!r}") from exc


def recording_profile_labels() -> tuple[str, ...]:
    return tuple(profile.label for profile in RECORDING_PROFILES.values())


def recording_profile_by_label(label: object) -> RecordingProfile:
    wanted = str(label or "").strip()
    for profile in RECORDING_PROFILES.values():
        if profile.label == wanted:
            return profile
    raise ValueError(f"unknown recording profile label: {label!r}")


def format_record_status(
    *,
    active: bool,
    armed: bool,
    profile: RecordingProfile,
    detail: str = "",
) -> str:
    if detail:
        return f"錄影: {detail} {profile.label}"
    if active:
        return f"錄影: 錄影中 ● {profile.label}"
    if armed:
        return f"錄影: 待命 {profile.label}（起飛後自動開始）"
    return f"錄影: 關 {profile.label}"


def _enum_token(value: object) -> str:
    if value is None:
        return ""
    text = str(getattr(value, "name", value)).strip().lower()
    return text.rsplit(".", 1)[-1]


def unwrap_camera_state(state: object) -> dict | None:
    """Normalize Olympe camera get_state maps to one cam_id=0 field dict."""
    if not isinstance(state, dict) or not state:
        return None
    if "resolution" in state or "framerate" in state:
        return state
    for key in (0, "0"):
        inner = state.get(key)
        if isinstance(inner, dict):
            return inner
    for value in state.values():
        if isinstance(value, dict) and (
            "resolution" in value or "framerate" in value
        ):
            return value
    return None


def recording_mode_matches(state: object, profile: RecordingProfile) -> bool | None:
    """None = no usable readback; True/False = confirmed match or mismatch."""
    fields = unwrap_camera_state(state)
    if fields is None:
        return None
    resolution = _enum_token(fields.get("resolution"))
    framerate = _enum_token(fields.get("framerate"))
    if not resolution and not framerate:
        return None
    wanted_res = {profile.resolution, profile.resolution.removeprefix("res_")}
    wanted_fps = {profile.framerate, profile.framerate.removeprefix("fps_")}
    resolution_ok = (not resolution) or resolution in wanted_res
    framerate_ok = (not framerate) or framerate in wanted_fps
    return resolution_ok and framerate_ok

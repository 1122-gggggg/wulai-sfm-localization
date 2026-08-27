"""Small binary header shared by the live localizer client and worker."""
from __future__ import annotations

import math
import os
import struct
import time





MAGIC = b"SFM1"
HEADER_SIZE = len(MAGIC) + 1
TIMED_MAGIC = b"SFM2"
_TIMED_HEADER = struct.Struct("!4sBd")
TIMED_HEADER_SIZE = _TIMED_HEADER.size
FUSED_MAGIC = b"SFM3"
_LEGACY_FUSED_HEADER = struct.Struct("!4sBdI6d")
LEGACY_FUSED_HEADER_SIZE = _LEGACY_FUSED_HEADER.size
_FUSED_HEADER = struct.Struct("!4sBddI6d")
FUSED_HEADER_SIZE = _FUSED_HEADER.size
FLAG_HAS_ATTITUDE = 1
FLAG_HAS_VELOCITY = 2
# Fail-closed sync bound for independent fused telemetry vs image capture.
# Matches pose-guided max_sync_error_s; do not backdate current IMU to the frame.
MAX_FUSED_SYNC_ERROR_S = 0.15


class FusedTelemetry:
    __slots__ = (
        "stamp",
        "roll",
        "pitch",
        "yaw",
        "speed_north",
        "speed_east",
        "speed_down",
    )

    def __init__(
        self,
        *,
        stamp: float | None = None,
        roll: float | None = None,
        pitch: float | None = None,
        yaw: float | None = None,
        speed_north: float | None = None,
        speed_east: float | None = None,
        speed_down: float | None = None,
    ) -> None:
        self.stamp = stamp
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.speed_north = speed_north
        self.speed_east = speed_east
        self.speed_down = speed_down

    @property
    def has_attitude(self) -> bool:
        return self.roll is not None and self.pitch is not None and self.yaw is not None

    @property
    def has_velocity(self) -> bool:
        return (
            self.speed_north is not None
            and self.speed_east is not None
            and self.speed_down is not None
        )

MODE_TO_CODE = {
    "auto": 0,
    "global": 1,
    "weak": 2,
    "track": 3,
    # One-frame safety request. Unlike the benchmark modes, this does not
    # reset the worker back to BOOT_INIT on the following automatic frame.
    "relocalize": 4,
}
CODE_TO_MODE = {code: mode for mode, code in MODE_TO_CODE.items()}


def _validate_monotonic_stamp(stamp: float, *, label: str) -> float:
    if not math.isfinite(stamp) or stamp <= 0:
        raise ValueError(f"invalid live-localizer {label}: {stamp!r}")
    if stamp > time.monotonic():
        raise ValueError(f"future live-localizer {label}: {stamp!r}")
    return stamp


def _validate_capture_stamp(stamp: float) -> float:
    return _validate_monotonic_stamp(stamp, label="capture stamp")


def _validate_telemetry_stamp(stamp: float, capture_stamp: float) -> float:
    stamp = _validate_monotonic_stamp(stamp, label="telemetry stamp")
    skew = abs(stamp - capture_stamp)
    if not math.isfinite(skew) or skew > MAX_FUSED_SYNC_ERROR_S:
        raise ValueError(
            f"stale live-localizer telemetry stamp: {stamp!r} vs capture {capture_stamp!r}"
        )
    return stamp


def encode_mode(mode: str) -> bytes:
    try:
        code = MODE_TO_CODE[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unsupported localization control mode: {mode!r}") from exc
    return MAGIC + bytes((code,))


def decode_mode(header: bytes | bytearray | memoryview) -> str:
    raw = bytes(header)
    if len(raw) != HEADER_SIZE or raw[:len(MAGIC)] != MAGIC:
        raise ValueError("invalid live-localizer control header")
    try:
        return CODE_TO_MODE[raw[-1]]
    except KeyError as exc:
        raise ValueError(f"invalid live-localizer mode code: {raw[-1]}") from exc


def encode_request(mode: str, capture_stamp: float | None) -> bytes:
    """Encode a timed request while retaining the legacy SFM1 mode header."""
    if capture_stamp is None:
        return encode_mode(mode)
    try:
        stamp = float(capture_stamp)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"invalid live-localizer capture stamp: {capture_stamp!r}"
        ) from exc
    stamp = _validate_capture_stamp(stamp)
    try:
        code = MODE_TO_CODE[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unsupported localization control mode: {mode!r}") from exc
    return _TIMED_HEADER.pack(TIMED_MAGIC, code, stamp)


def decode_request(header: bytes | bytearray | memoryview) -> tuple[str, float | None]:
    raw = bytes(header)
    if len(raw) == HEADER_SIZE:
        return decode_mode(raw), None
    if len(raw) != TIMED_HEADER_SIZE:
        raise ValueError("invalid live-localizer request header size")
    magic, code, stamp = _TIMED_HEADER.unpack(raw)
    if magic != TIMED_MAGIC:
        raise ValueError("invalid timed live-localizer control header")
    try:
        mode = CODE_TO_MODE[code]
    except KeyError as exc:
        raise ValueError(f"invalid live-localizer mode code: {code}") from exc
    stamp = _validate_capture_stamp(stamp)
    return mode, stamp

def _finite_or_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    number = float(value)
    return number if math.isfinite(number) else float("nan")


def _optional_finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def encode_fused_request(
    mode: str,
    capture_stamp: float,
    fused: FusedTelemetry,
    telemetry_stamp: float | None = None,
) -> bytes:
    stamp = _validate_capture_stamp(float(capture_stamp))
    try:
        code = MODE_TO_CODE[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unsupported localization control mode: {mode!r}") from exc
    flags = 0
    if fused.has_attitude:
        flags |= FLAG_HAS_ATTITUDE
    if fused.has_velocity:
        flags |= FLAG_HAS_VELOCITY
    if flags == 0:
        return encode_request(mode, stamp)
    fused_stamp = telemetry_stamp if telemetry_stamp is not None else fused.stamp
    if fused_stamp is None:
        # Existing three-argument callers still encode a fused header. The extra
        # field is required on the wire; using capture here only fills that slot
        # for the legacy encode signature. Live clients must pass an independent
        # telemetry stamp and must not take this path.
        fused_stamp = stamp
    fused_stamp = _validate_telemetry_stamp(float(fused_stamp), stamp)
    return _FUSED_HEADER.pack(
        FUSED_MAGIC,
        code,
        stamp,
        fused_stamp,
        flags,
        _finite_or_nan(fused.roll),
        _finite_or_nan(fused.pitch),
        _finite_or_nan(fused.yaw),
        _finite_or_nan(fused.speed_north),
        _finite_or_nan(fused.speed_east),
        _finite_or_nan(fused.speed_down),
    )


def decode_control_header(
    header: bytes | bytearray | memoryview,
) -> tuple[str, float | None, FusedTelemetry | None]:
    raw = bytes(header)
    if len(raw) == HEADER_SIZE or (
        len(raw) == TIMED_HEADER_SIZE and raw[:4] == TIMED_MAGIC
    ):
        mode, stamp = decode_request(raw)
        return mode, stamp, None
    if len(raw) == LEGACY_FUSED_HEADER_SIZE and raw[:4] == FUSED_MAGIC:
        raise ValueError("legacy fused live-localizer control header")
    if len(raw) != FUSED_HEADER_SIZE:
        raise ValueError("invalid live-localizer request header size")
    (
        magic,
        code,
        stamp,
        fused_stamp,
        flags,
        roll,
        pitch,
        yaw,
        vn,
        ve,
        vd,
    ) = _FUSED_HEADER.unpack(raw)
    if magic != FUSED_MAGIC:
        raise ValueError("invalid fused live-localizer control header")
    try:
        mode = CODE_TO_MODE[code]
    except KeyError as exc:
        raise ValueError(f"invalid live-localizer mode code: {code}") from exc
    stamp = _validate_capture_stamp(stamp)
    fused_stamp = _validate_telemetry_stamp(fused_stamp, stamp)
    fused = FusedTelemetry(
        stamp=fused_stamp,
        roll=_optional_finite(roll) if flags & FLAG_HAS_ATTITUDE else None,
        pitch=_optional_finite(pitch) if flags & FLAG_HAS_ATTITUDE else None,
        yaw=_optional_finite(yaw) if flags & FLAG_HAS_ATTITUDE else None,
        speed_north=_optional_finite(vn) if flags & FLAG_HAS_VELOCITY else None,
        speed_east=_optional_finite(ve) if flags & FLAG_HAS_VELOCITY else None,
        speed_down=_optional_finite(vd) if flags & FLAG_HAS_VELOCITY else None,
    )
    return mode, stamp, fused


def pose_guided_live_enabled() -> bool:
    override = os.environ.get("SFM_POSE_GUIDED", "").strip()
    if override == "1":
        return True
    if override == "0":
        return False
    try:
        from pose_guided.config import load_pose_guided_config

        return bool(load_pose_guided_config().enabled)
    except Exception:
        return False

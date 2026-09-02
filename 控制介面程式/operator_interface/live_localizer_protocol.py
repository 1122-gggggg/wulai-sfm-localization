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
GNSS_FUSED_MAGIC = b"SFM4"
_GNSS_FUSED_HEADER = struct.Struct("!4sBdddI12d")
GNSS_FUSED_HEADER_SIZE = _GNSS_FUSED_HEADER.size
FLAG_HAS_ATTITUDE = 1
FLAG_HAS_VELOCITY = 2
FLAG_HAS_GNSS = 4
# Fail-closed sync bounds for independent telemetry vs image capture.
MAX_FUSED_SYNC_ERROR_S = 0.15
MAX_GNSS_AGE_S = 2.5


class FusedTelemetry:
    __slots__ = (
        "stamp",
        "roll",
        "pitch",
        "yaw",
        "speed_north",
        "speed_east",
        "speed_down",
        "gps_stamp",
        "latitude",
        "longitude",
        "altitude",
        "latitude_accuracy",
        "longitude_accuracy",
        "altitude_accuracy",
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
        gps_stamp: float | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        altitude: float | None = None,
        latitude_accuracy: float | None = None,
        longitude_accuracy: float | None = None,
        altitude_accuracy: float | None = None,
    ) -> None:
        self.stamp = stamp
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.speed_north = speed_north
        self.speed_east = speed_east
        self.speed_down = speed_down
        self.gps_stamp = gps_stamp
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.latitude_accuracy = latitude_accuracy
        self.longitude_accuracy = longitude_accuracy
        self.altitude_accuracy = altitude_accuracy

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

    @property
    def has_gnss(self) -> bool:
        return (
            self.gps_stamp is not None
            and self.latitude is not None
            and self.longitude is not None
            and self.altitude is not None
            and self.latitude_accuracy is not None
            and self.longitude_accuracy is not None
            and self.altitude_accuracy is not None
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


def _validate_gnss_stamp(stamp: float, capture_stamp: float) -> float:
    stamp = _validate_monotonic_stamp(stamp, label="GNSS stamp")
    age = capture_stamp - stamp
    if not math.isfinite(age) or age < -1e-6 or age > MAX_GNSS_AGE_S:
        raise ValueError(f"stale live-localizer GNSS stamp: {stamp!r} vs capture {capture_stamp!r}")
    return stamp


def _validated_gnss_value(name: str, value: float | None) -> float:
    number = _finite_or_nan(value)
    if not math.isfinite(number):
        raise ValueError(f"invalid live-localizer GNSS {name}: {value!r}")
    return number


def encode_mode(mode: str) -> bytes:
    try:
        code = MODE_TO_CODE[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unsupported localization control mode: {mode!r}") from exc
    return MAGIC + bytes((code,))


def decode_mode(header: bytes | bytearray | memoryview) -> str:
    raw = bytes(header)
    if len(raw) != HEADER_SIZE or raw[: len(MAGIC)] != MAGIC:
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
        raise ValueError(f"invalid live-localizer capture stamp: {capture_stamp!r}") from exc
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
    if fused.has_gnss:
        flags |= FLAG_HAS_GNSS
    if flags == 0:
        return encode_request(mode, stamp)
    fused_stamp = telemetry_stamp if telemetry_stamp is not None else fused.stamp
    if fused_stamp is None:
        fused_stamp = stamp
    fused_stamp = _validate_telemetry_stamp(float(fused_stamp), stamp)
    common = (
        _finite_or_nan(fused.roll),
        _finite_or_nan(fused.pitch),
        _finite_or_nan(fused.yaw),
        _finite_or_nan(fused.speed_north),
        _finite_or_nan(fused.speed_east),
        _finite_or_nan(fused.speed_down),
    )
    if not fused.has_gnss:
        return _FUSED_HEADER.pack(
            FUSED_MAGIC,
            code,
            stamp,
            fused_stamp,
            flags,
            *common,
        )
    gps_stamp = _validate_gnss_stamp(float(fused.gps_stamp), stamp)
    latitude = _validated_gnss_value("latitude", fused.latitude)
    longitude = _validated_gnss_value("longitude", fused.longitude)
    altitude = _validated_gnss_value("altitude", fused.altitude)
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError("invalid live-localizer GNSS coordinates")
    accuracies = tuple(
        _validated_gnss_value(name, value)
        for name, value in (
            ("latitude accuracy", fused.latitude_accuracy),
            ("longitude accuracy", fused.longitude_accuracy),
            ("altitude accuracy", fused.altitude_accuracy),
        )
    )
    if any(value < 0.0 for value in accuracies):
        raise ValueError("invalid live-localizer GNSS accuracy")
    return _GNSS_FUSED_HEADER.pack(
        GNSS_FUSED_MAGIC,
        code,
        stamp,
        fused_stamp,
        gps_stamp,
        flags,
        *common,
        latitude,
        longitude,
        altitude,
        *accuracies,
    )


def decode_control_header(
    header: bytes | bytearray | memoryview,
) -> tuple[str, float | None, FusedTelemetry | None]:
    raw = bytes(header)
    if len(raw) == HEADER_SIZE or (len(raw) == TIMED_HEADER_SIZE and raw[:4] == TIMED_MAGIC):
        mode, stamp = decode_request(raw)
        return mode, stamp, None
    if len(raw) == LEGACY_FUSED_HEADER_SIZE and raw[:4] == FUSED_MAGIC:
        raise ValueError("legacy fused live-localizer control header")
    is_gnss = len(raw) == GNSS_FUSED_HEADER_SIZE and raw[:4] == GNSS_FUSED_MAGIC
    if is_gnss:
        (
            magic,
            code,
            stamp,
            fused_stamp,
            gps_stamp,
            flags,
            roll,
            pitch,
            yaw,
            vn,
            ve,
            vd,
            latitude,
            longitude,
            altitude,
            latitude_accuracy,
            longitude_accuracy,
            altitude_accuracy,
        ) = _GNSS_FUSED_HEADER.unpack(raw)
    elif len(raw) == FUSED_HEADER_SIZE and raw[:4] == FUSED_MAGIC:
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
        gps_stamp = latitude = longitude = altitude = float("nan")
        latitude_accuracy = longitude_accuracy = altitude_accuracy = float("nan")
    else:
        raise ValueError("invalid live-localizer request header size")
    expected_magic = GNSS_FUSED_MAGIC if is_gnss else FUSED_MAGIC
    if magic != expected_magic:
        raise ValueError("invalid fused live-localizer control header")
    try:
        mode = CODE_TO_MODE[code]
    except KeyError as exc:
        raise ValueError(f"invalid live-localizer mode code: {code}") from exc
    stamp = _validate_capture_stamp(stamp)
    fused_stamp = _validate_telemetry_stamp(fused_stamp, stamp)
    has_gnss = bool(flags & FLAG_HAS_GNSS)
    if has_gnss:
        gps_stamp = _validate_gnss_stamp(gps_stamp, stamp)
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not math.isfinite(altitude)
            or not -90.0 <= latitude <= 90.0
            or not -180.0 <= longitude <= 180.0
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (
                    latitude_accuracy,
                    longitude_accuracy,
                    altitude_accuracy,
                )
            )
        ):
            raise ValueError("invalid fused live-localizer GNSS payload")
    fused = FusedTelemetry(
        stamp=fused_stamp,
        roll=_optional_finite(roll) if flags & FLAG_HAS_ATTITUDE else None,
        pitch=_optional_finite(pitch) if flags & FLAG_HAS_ATTITUDE else None,
        yaw=_optional_finite(yaw) if flags & FLAG_HAS_ATTITUDE else None,
        speed_north=_optional_finite(vn) if flags & FLAG_HAS_VELOCITY else None,
        speed_east=_optional_finite(ve) if flags & FLAG_HAS_VELOCITY else None,
        speed_down=_optional_finite(vd) if flags & FLAG_HAS_VELOCITY else None,
        gps_stamp=float(gps_stamp) if has_gnss else None,
        latitude=float(latitude) if has_gnss else None,
        longitude=float(longitude) if has_gnss else None,
        altitude=float(altitude) if has_gnss else None,
        latitude_accuracy=float(latitude_accuracy) if has_gnss else None,
        longitude_accuracy=float(longitude_accuracy) if has_gnss else None,
        altitude_accuracy=float(altitude_accuracy) if has_gnss else None,
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

"""Small binary header shared by the live localizer client and worker."""
from __future__ import annotations

import math
import struct
import time


MAGIC = b"SFM1"
HEADER_SIZE = len(MAGIC) + 1
TIMED_MAGIC = b"SFM2"
_TIMED_HEADER = struct.Struct("!4sBd")
TIMED_HEADER_SIZE = _TIMED_HEADER.size
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


def _validate_capture_stamp(stamp: float) -> float:
    if not math.isfinite(stamp) or stamp <= 0:
        raise ValueError(f"invalid live-localizer capture stamp: {stamp!r}")
    if stamp > time.monotonic():
        raise ValueError(f"future live-localizer capture stamp: {stamp!r}")
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

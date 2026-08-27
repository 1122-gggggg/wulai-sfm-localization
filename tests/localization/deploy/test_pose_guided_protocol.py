from __future__ import annotations

import time

import pytest

from live_localizer_protocol import (
    FUSED_HEADER_SIZE,
    FUSED_MAGIC,
    FusedTelemetry,
    decode_control_header,
    decode_request,
    encode_fused_request,
    encode_request,
)


def test_sfm2_round_trip_unchanged() -> None:
    header = encode_request("auto", 123.456)
    mode, stamp = decode_request(header)
    assert mode == "auto"
    assert stamp == pytest.approx(123.456)
    mode, stamp, fused = decode_control_header(header)
    assert fused is None
    assert mode == "auto"


def test_sfm3_fused_round_trip() -> None:
    header = encode_fused_request(
        "track",
        10.5,
        FusedTelemetry(roll=0.1, pitch=-0.2, yaw=1.3, speed_north=0.4, speed_east=-0.1, speed_down=0.0),
    )
    assert header.startswith(FUSED_MAGIC)
    assert len(header) == FUSED_HEADER_SIZE
    mode, stamp, fused = decode_control_header(header)
    assert mode == "track"
    assert stamp == pytest.approx(10.5)
    assert fused is not None
    assert fused.roll == pytest.approx(0.1)
    assert fused.yaw == pytest.approx(1.3)
    assert fused.speed_north == pytest.approx(0.4)


def test_sfm3_without_usable_fields_falls_back_to_sfm2() -> None:
    header = encode_fused_request("auto", 4.0, FusedTelemetry())
    mode, stamp = decode_request(header)
    assert mode == "auto"
    assert stamp == pytest.approx(4.0)


def test_legacy_decode_rejects_sfm3_size() -> None:
    header = encode_fused_request(
        "auto",
        3.0,
        FusedTelemetry(roll=0.0, pitch=0.0, yaw=0.0),
    )
    with pytest.raises(ValueError, match="header size"):
        decode_request(header)


def test_future_capture_stamp_still_rejected() -> None:
    with pytest.raises(ValueError, match="future"):
        encode_fused_request(
            "auto",
            time.monotonic() + 1.0,
            FusedTelemetry(roll=0.0, pitch=0.0, yaw=0.0),
        )

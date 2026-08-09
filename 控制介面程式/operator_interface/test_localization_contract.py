from __future__ import annotations

import pytest

from localization_contract import InvalidLocalizationResult, LocalizationResult


def _payload(*, capture_mono_ns: int, pose_mono_ns: int) -> dict:
    return {
        "seq": 1,
        "frame_id": "frame-1",
        "capture_mono_ns": capture_mono_ns,
        "pose_mono_ns": pose_mono_ns,
        "validity": True,
        "confidence": 0.9,
        "pose": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.1},
    }


def test_localization_contract_rejects_pose_before_capture() -> None:
    with pytest.raises(InvalidLocalizationResult, match="precede capture"):
        LocalizationResult.from_payload(
            _payload(capture_mono_ns=200, pose_mono_ns=199),
            now_mono_ns=300,
        )


def test_localization_contract_accepts_pose_at_or_after_capture() -> None:
    result = LocalizationResult.from_payload(
        _payload(capture_mono_ns=200, pose_mono_ns=200),
        now_mono_ns=300,
    )

    assert result.capture_mono_ns == result.pose_mono_ns == 200

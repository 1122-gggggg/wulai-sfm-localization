from __future__ import annotations

import pytest

from flight_operator_app import classify_localization_health
from localization_contract import LocalizationResult


def _typed(**overrides) -> LocalizationResult:
    payload = {
        "seq": 1,
        "frame_id": "frame-1",
        "capture_mono_ns": 200,
        "pose_mono_ns": 200,
        "validity": True,
        "success": True,
        "confidence": 0.9,
        "inliers": 400,
        "reproj_rms": 1.2,
        "mode": "TRACK",
        "next_mode": "TRACK",
        "pose": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.1},
    }
    payload.update(overrides)
    return LocalizationResult.from_payload(payload, now_mono_ns=300)


def test_typed_localization_result_with_a_good_fix_is_ok() -> None:
    # Regression: the live pipeline hands classify_localization_health a typed
    # LocalizationResult (dict-like, but not a dict). It must classify on the
    # actual fix quality, not fall straight through to LOST on the type.
    result = _typed()

    assert not isinstance(result, dict)
    assert classify_localization_health(result) == "OK"


def test_typed_localization_result_without_success_is_lost() -> None:
    result = _typed(success=False, validity=False, pose=None)

    assert classify_localization_health(result) == "LOST"


def test_typed_localization_result_with_low_inliers_is_degraded() -> None:
    result = _typed(inliers=10)

    assert classify_localization_health(result) == "DEGRADED"


def test_plain_dict_result_still_classifies_on_quality() -> None:
    assert classify_localization_health({"success": True, "inliers": 400}) == "OK"


@pytest.mark.parametrize("not_a_result", [None, 42, "nope", ["x"]])
def test_a_non_result_object_is_lost(not_a_result: object) -> None:
    assert classify_localization_health(not_a_result) == "LOST"

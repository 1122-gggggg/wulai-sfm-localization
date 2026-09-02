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


@pytest.mark.parametrize("missing", ["seq", "frame_id", "capture_mono_ns", "pose_mono_ns", "confidence"])
def test_missing_required_field_raises_a_named_error_not_a_keyerror(missing: str) -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)
    del payload[missing]

    with pytest.raises(InvalidLocalizationResult):
        LocalizationResult.from_payload(payload, now_mono_ns=300)


def test_frame_id_must_be_a_non_empty_string() -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)
    payload["frame_id"] = "   "

    with pytest.raises(InvalidLocalizationResult, match="frame_id"):
        LocalizationResult.from_payload(payload, now_mono_ns=300)


def test_seq_must_be_a_non_negative_integer_not_a_bool_or_float() -> None:
    for bad in (-1, True, 1.5, "1"):
        payload = _payload(capture_mono_ns=200, pose_mono_ns=200)
        payload["seq"] = bad
        with pytest.raises(InvalidLocalizationResult, match="seq"):
            LocalizationResult.from_payload(payload, now_mono_ns=300)


@pytest.mark.parametrize("bad_confidence", [float("nan"), float("inf"), -0.1, 1.1, "0.5", True])
def test_confidence_must_be_a_finite_number_in_zero_one(bad_confidence) -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)
    payload["confidence"] = bad_confidence

    with pytest.raises(InvalidLocalizationResult, match="confidence"):
        LocalizationResult.from_payload(payload, now_mono_ns=300)


def test_validity_and_success_disagreeing_is_rejected() -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)
    payload["validity"] = True
    payload["success"] = False

    with pytest.raises(InvalidLocalizationResult, match="disagree"):
        LocalizationResult.from_payload(payload, now_mono_ns=300)


def test_a_valid_result_without_a_pose_is_rejected() -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)
    payload["pose"] = None

    with pytest.raises(InvalidLocalizationResult, match="pose"):
        LocalizationResult.from_payload(payload, now_mono_ns=300)


def test_an_invalid_result_without_a_pose_is_accepted() -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)
    payload["validity"] = False
    payload["success"] = False
    payload["pose"] = None

    result = LocalizationResult.from_payload(payload, now_mono_ns=300)

    assert result.pose is None
    assert result.validity is False


def test_non_finite_pose_components_are_rejected() -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)
    payload["pose"] = {"x": float("nan"), "y": 0.0, "z": 0.0, "yaw": 0.0}

    with pytest.raises(InvalidLocalizationResult, match="pose"):
        LocalizationResult.from_payload(payload, now_mono_ns=300)


def test_payload_must_be_a_mapping() -> None:
    with pytest.raises(InvalidLocalizationResult, match="object"):
        LocalizationResult.from_payload([1, 2, 3], now_mono_ns=300)  # type: ignore[arg-type]


def test_from_json_rejects_malformed_json_with_the_contract_error_not_a_raw_json_error() -> None:
    with pytest.raises(InvalidLocalizationResult, match="JSON"):
        LocalizationResult.from_json(b"{not valid json", now_mono_ns=300)


def test_from_json_round_trips_a_well_formed_payload() -> None:
    import json

    raw = json.dumps(_payload(capture_mono_ns=200, pose_mono_ns=200))

    result = LocalizationResult.from_json(raw, now_mono_ns=300)

    assert result.seq == 1
    assert result.frame_id == "frame-1"
    assert result.pose == (1.0, 2.0, 3.0, 0.1)


def test_a_capture_timestamp_in_the_future_is_rejected() -> None:
    with pytest.raises(InvalidLocalizationResult, match="future"):
        LocalizationResult.from_payload(
            _payload(capture_mono_ns=500, pose_mono_ns=500),
            now_mono_ns=300,
        )


def test_to_payload_round_trips_the_canonical_fields() -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)

    result = LocalizationResult.from_payload(payload, now_mono_ns=300)
    round_tripped = result.to_payload()

    assert round_tripped["seq"] == 1
    assert round_tripped["frame_id"] == "frame-1"
    assert round_tripped["capture_mono_ns"] == 200
    assert round_tripped["pose_mono_ns"] == 200
    assert round_tripped["validity"] is True
    assert round_tripped["confidence"] == pytest.approx(0.9)


def test_dict_compatibility_shim_supports_getitem_and_get() -> None:
    payload = _payload(capture_mono_ns=200, pose_mono_ns=200)

    result = LocalizationResult.from_payload(payload, now_mono_ns=300)

    assert result["seq"] == 1
    assert result.get("nonexistent_key", "fallback") == "fallback"
    with pytest.raises(KeyError):
        result["nonexistent_key"]

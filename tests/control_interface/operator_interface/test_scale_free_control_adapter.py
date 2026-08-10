from __future__ import annotations

import pytest

from scale_free_control_adapter import (
    ScaleFreeSample,
    authoritative_core_path,
    decide_scale_free,
)


def test_operator_uses_parrot_stimulate_authoritative_scale_free_core() -> None:
    now_ns = 5_000_000_000
    decision = decide_scale_free(
        ScaleFreeSample(
            position_map=(0.0, 0.0, 0.0),
            target_map=(100.0, 0.0, 0.0),
            camera_yaw_rad=0.0,
            pose_mono_ns=now_ns,
            localization_state="TRACK",
            airframe_horizontal_speed_mps=0.1,
            speed_mono_ns=now_ns,
            route_deviation_ok=True,
            target_reached=False,
            approval_locked=True,
        ),
        now_mono_ns=now_ns,
    )

    assert "parrot_stimulate" in str(authoritative_core_path())
    assert decision.reason == "LOCKED_EXTERNAL_APPROVAL"
    assert decision.speed_limit_mps == pytest.approx(0.30)
    assert decision.zero_motion

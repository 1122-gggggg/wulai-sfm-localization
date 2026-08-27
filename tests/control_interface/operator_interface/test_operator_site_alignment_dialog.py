from __future__ import annotations

import json
import math

import numpy as np
import pytest

from operator_site_alignment_dialog import (
    GUIDED_ALIGNMENT_STEPS,
    GuidedSiteAlignmentDialog,
    GuidedSiteAlignmentModel,
    MIN_CONTROL_POINTS,
    ManualSiteAlignmentModel,
    MotionObservation,
    motion_sample_matches_step,
    validate_hover_reference,
    validate_motion_evidence,
)
from site_alignment import load_site_alignment


MAP_POINTS = (
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 3.0, 0.0),
    (1.0, 1.0, 1.0),
    (-1.0, 2.0, 0.5),
    (2.0, -1.0, 1.5),
)
SITE_OFFSET = np.array((4.0, -2.0, 3.0))
SITE_POINTS = tuple(tuple(np.asarray(point) + SITE_OFFSET) for point in MAP_POINTS)


def _model() -> ManualSiteAlignmentModel:
    current = iter(MAP_POINTS)
    return ManualSiteAlignmentModel(
        map_frame_id="map-v1",
        site_frame_id="world-enu-v1",
        get_current_map_position=lambda: next(current),
    )


def _add_six_points(model: ManualSiteAlignmentModel) -> None:
    for index, site_point in enumerate(SITE_POINTS):
        model.capture_current_map_position(site_point, label=f"P{index + 1}")


def test_capture_lists_and_removes_control_points_through_public_seams() -> None:
    model = _model()

    first = model.capture_current_map_position(SITE_POINTS[0], label="origin")
    second = model.capture_current_map_position(SITE_POINTS[1], label="east")

    assert first.map_point == MAP_POINTS[0]
    assert second.site_point == SITE_POINTS[1]
    assert model.control_points == (first, second)

    removed = model.remove_control_point(0)

    assert removed == first
    assert model.control_points == (second,)


def test_solve_requires_six_points_and_reports_rmse_and_max_residual() -> None:
    model = _model()
    for site_point in SITE_POINTS[:-1]:
        model.capture_current_map_position(site_point)

    with pytest.raises(ValueError, match=f"at least {MIN_CONTROL_POINTS}"):
        model.solve()

    model.capture_current_map_position(SITE_POINTS[-1])
    fit = model.solve()

    assert fit.quality.rmse_m == pytest.approx(0.0, abs=1e-12)
    assert fit.quality.max_error_m == pytest.approx(0.0, abs=1e-12)
    assert model.fit is fit
    assert model.approved is False


def test_write_requires_explicit_approval_and_persists_validated_schema_atomically(
    tmp_path,
) -> None:
    model = _model()
    _add_six_points(model)
    model.solve()
    target = tmp_path / "site_alignment.json"
    target.write_text("old content", encoding="utf-8")

    with pytest.raises(PermissionError, match="approval"):
        model.write(target)
    assert target.read_text(encoding="utf-8") == "old content"

    model.approve()
    saved = model.write(target)

    assert saved == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "sfm-site-alignment/v1"
    assert payload["approved"] is True
    loaded = load_site_alignment(
        target,
        expected_map_frame_id="map-v1",
        expected_site_frame_id="world-enu-v1",
    )
    assert loaded.approved is True
    assert not list(tmp_path.glob(".site_alignment.json.*.tmp"))


def test_editing_points_invalidates_fit_and_approval() -> None:
    model = _model()
    _add_six_points(model)
    model.solve()
    model.approve()

    model.add_control_point((10.0, 10.0, 10.0), (14.0, 8.0, 13.0))

    assert model.fit is None
    assert model.approved is False
    with pytest.raises(ValueError, match="solve"):
        model.write("unused.json")


def _motion(
    pcmd: tuple[int, int, int, int],
    *,
    north: float = 0.0,
    east: float = 0.0,
    down: float = 0.0,
    pilot_sticks: bool = False,
    yaw_rad: float = 0.0,
    gimbal_pitch_deg: float = -20.0,
    stick_axes: dict[int, int] | None = None,
    flight_state: str = "hovering",
) -> MotionObservation:
    return MotionObservation.from_mapping(
        {
            "pcmd": pcmd,
            "pcmd_age_s": 0.02,
            "pilot_sticks": pilot_sticks,
            "stick_axes": stick_axes or {},
            "speed_north_mps": north,
            "speed_east_mps": east,
            "speed_down_mps": down,
            "yaw_rad": yaw_rad,
            "gimbal_pitch_deg": gimbal_pitch_deg,
            "heading_state": "OK",
            "telemetry_age_s": 0.03,
            "flight_state": flight_state,
        }
    )


@pytest.mark.parametrize(
    ("key", "pcmd", "velocity"),
    (
        ("right", (10, 0, 0, 0), (0.0, 0.2, 0.0)),
        ("left", (-10, 0, 0, 0), (0.0, -0.2, 0.0)),
        ("forward", (0, 10, 0, 0), (0.2, 0.0, 0.0)),
        ("back", (0, -10, 0, 0), (-0.2, 0.0, 0.0)),
        ("up", (0, 0, 0, 10), (0.0, 0.0, -0.2)),
        ("down", (0, 0, 0, -10), (0.0, 0.0, 0.2)),
    ),
)
def test_motion_evidence_requires_command_and_matching_measured_direction(
    key, pcmd, velocity,
) -> None:
    step = next(item for item in GUIDED_ALIGNMENT_STEPS if item.key == key)
    sample = _motion(
        pcmd,
        north=velocity[0],
        east=velocity[1],
        down=velocity[2],
    )

    validate_motion_evidence(step, (sample, sample))
    assert motion_sample_matches_step(step, sample)


def test_motion_evidence_rejects_wrong_direction_and_physical_takeover() -> None:
    step = next(item for item in GUIDED_ALIGNMENT_STEPS if item.key == "right")
    wrong = _motion((10, 0, 0, 0), east=-0.2)
    with pytest.raises(ValueError, match="沒有穩定向右"):
        validate_motion_evidence(step, (wrong, wrong))

    physical = _motion(
        (0, 0, 0, 0),
        east=0.2,
        pilot_sticks=True,
        stick_axes={0: 12_000},
    )
    validate_motion_evidence(step, (physical, physical))
    assert motion_sample_matches_step(step, physical)

    mixed = (physical, _motion((10, 0, 0, 0), east=0.2))
    with pytest.raises(ValueError, match="控制來源在步驟中切換"):
        validate_motion_evidence(step, mixed)


def test_motion_evidence_rejects_diagonal_or_rotating_command() -> None:
    step = next(item for item in GUIDED_ALIGNMENT_STEPS if item.key == "right")
    diagonal = _motion((10, 8, 0, 0), east=0.2)
    with pytest.raises(ValueError, match="旋轉或斜向 PCMD"):
        validate_motion_evidence(step, (diagonal, diagonal))

    rotating = _motion((10, 0, 5, 0), east=0.2)
    with pytest.raises(ValueError, match="旋轉或斜向 PCMD"):
        validate_motion_evidence(step, (rotating, rotating))

    yaw_changed = _motion((10, 0, 0, 0), east=0.2, yaw_rad=0.2)
    normal = _motion((10, 0, 0, 0), east=0.2, yaw_rad=0.0)
    with pytest.raises(ValueError, match="yaw"):
        validate_motion_evidence(step, (normal, yaw_changed))

    gimbal_changed = _motion(
        (10, 0, 0, 0), east=0.2, gimbal_pitch_deg=-18.0
    )
    with pytest.raises(ValueError, match="gimbal pitch"):
        validate_motion_evidence(step, (normal, gimbal_changed))

    wrong_axis = _motion((0, 10, 0, 0), north=0.2)
    with pytest.raises(ValueError, match="非預期方向"):
        validate_motion_evidence(step, (normal, normal, wrong_axis))


def test_hover_reference_requires_stationary_fresh_heading_and_gimbal() -> None:
    validate_hover_reference(_motion((0, 0, 0, 0)))

    moving = _motion((0, 0, 0, 0), north=0.25)
    with pytest.raises(ValueError, match="穩定懸停"):
        validate_hover_reference(moving)

    commanded = _motion((0, 0, 5, 0))
    with pytest.raises(ValueError, match="PCMD"):
        validate_hover_reference(commanded)

    physical = _motion(
        (0, 0, 0, 0), pilot_sticks=True, stick_axes={1: -12_000}
    )
    with pytest.raises(ValueError, match="實體搖桿"):
        validate_hover_reference(physical)

    landed = _motion((0, 0, 0, 0), flight_state="landed")
    with pytest.raises(ValueError, match="起飛"):
        validate_hover_reference(landed)


def test_guided_alignment_collects_all_axes_and_solves_only_after_validation() -> None:
    map_points = iter(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 0.3),
            (0.0, 0.0, -0.3),
        )
    )
    model = GuidedSiteAlignmentModel(
        map_frame_id="map-v1",
        site_frame_id="world-v1",
        get_current_map_position=lambda: next(map_points),
    )
    model.capture_origin()
    evidence = {
        "right": _motion((10, 0, 0, 0), east=0.2),
        "left": _motion((-10, 0, 0, 0), east=-0.2),
        "forward": _motion((0, 10, 0, 0), north=0.2),
        "back": _motion((0, -10, 0, 0), north=-0.2),
        "up": _motion((0, 0, 0, 10), down=-0.2),
        "down": _motion((0, 0, 0, -10), down=0.2),
    }
    for step in GUIDED_ALIGNMENT_STEPS[1:]:
        sample = evidence[step.key]
        model.finish_step(step.key, (sample, sample))

    fit = model.solve()

    assert model.complete is True
    assert model.point_count == 7
    assert fit.transform.scale == 1.0
    assert fit.quality.rmse_m == pytest.approx(0.0, abs=1e-12)


def test_guided_alignment_uses_measured_distance_without_estimating_scale() -> None:
    map_points = iter(
        (
            (3.0, -2.0, 0.7),
            (3.82, -1.98, 0.7),
            (2.38, -2.01, 0.7),
            (3.01, -0.91, 0.71),
            (2.99, -2.93, 0.69),
            (3.0, -2.0, 1.14),
            (3.0, -2.0, 0.34),
        )
    )
    model = GuidedSiteAlignmentModel(
        map_frame_id="map-v1",
        site_frame_id="world-v1",
        get_current_map_position=lambda: next(map_points),
    )
    model.capture_origin()
    evidence = {
        "right": _motion((10, 0, 0, 0), east=0.2),
        "left": _motion((-10, 0, 0, 0), east=-0.2),
        "forward": _motion((0, 10, 0, 0), north=0.2),
        "back": _motion((0, -10, 0, 0), north=-0.2),
        "up": _motion((0, 0, 0, 10), down=-0.2),
        "down": _motion((0, 0, 0, -10), down=0.2),
    }
    for step in GUIDED_ALIGNMENT_STEPS[1:]:
        sample = evidence[step.key]
        model.finish_step(step.key, (sample, sample))

    fit = model.solve()

    assert fit.transform.scale == 1.0
    assert 0.0 < fit.quality.rmse_m < 0.05


def test_automatic_monitor_captures_after_command_then_hover_and_requires_return() -> None:
    current_map_point = [(4.0, 3.0, 2.0)]
    model = GuidedSiteAlignmentModel(
        map_frame_id="map-v1",
        site_frame_id="world-v1",
        get_current_map_position=lambda: current_map_point[0],
    )

    class Status:
        value = ""

        def set(self, value: str) -> None:
            self.value = value

    dialog = object.__new__(GuidedSiteAlignmentDialog)
    dialog.model = model
    dialog._auto_monitoring = True
    dialog._collecting = False
    dialog._observations = []
    dialog._active_step = None
    dialog._awaiting_origin_return = False
    dialog._hover_sample_count = 0
    dialog._status_var = Status()
    dialog._refresh = lambda: None

    hover = _motion((0, 0, 0, 0))
    for _ in range(3):
        dialog._process_auto_sample(hover)
    assert model.current_step.key == "right"

    current_map_point[0] = (4.8, 3.0, 2.0)
    right = _motion((10, 0, 0, 0), east=0.2)
    dialog._process_auto_sample(right)
    dialog._process_auto_sample(right)
    for _ in range(3):
        dialog._process_auto_sample(hover)

    assert model.current_step.key == "left"
    assert dialog._awaiting_origin_return is True
    assert model.control_points[1].map_point == (4.8, 3.0, 2.0)

    current_map_point[0] = (4.0, 3.0, 2.0)
    for _ in range(3):
        dialog._process_auto_sample(hover)
    assert dialog._awaiting_origin_return is False


def test_guided_alignment_builds_camera_center_navigation_extrinsic() -> None:
    map_points = iter(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 0.3),
            (0.0, 0.0, -0.3),
        )
    )
    rotation_site_from_body = np.asarray(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    )
    expected_camera_from_body = np.asarray(
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    )
    rotation_camera_from_map = (
        expected_camera_from_body @ rotation_site_from_body.T
    )
    rotations = iter([rotation_camera_from_map] * len(GUIDED_ALIGNMENT_STEPS))
    model = GuidedSiteAlignmentModel(
        map_frame_id="map-v1",
        site_frame_id="world-v1",
        get_current_map_position=lambda: next(map_points),
        get_current_camera_rotation=lambda: next(rotations),
    )
    hover = _motion((0, 0, 0, 0), gimbal_pitch_deg=-20.0)
    model.capture_origin(hover)
    evidence = {
        "right": _motion((10, 0, 0, 0), east=0.2),
        "left": _motion((-10, 0, 0, 0), east=-0.2),
        "forward": _motion((0, 10, 0, 0), north=0.2),
        "back": _motion((0, -10, 0, 0), north=-0.2),
        "up": _motion((0, 0, 0, 10), down=-0.2),
        "down": _motion((0, 0, 0, -10), down=0.2),
    }
    for step in GUIDED_ALIGNMENT_STEPS[1:]:
        sample = evidence[step.key]
        model.finish_step(step.key, (sample, sample))
    model.solve()
    model.approve()

    extrinsic = model.camera_center_extrinsic(
        vehicle_id="anafi-serial-hash",
        body_frame_id="anafi-camera-center-frd",
        camera_frame_id="anafi-camera-opencv",
    )

    assert extrinsic.approved is True
    assert extrinsic.translation_camera_from_body_m == (0.0, 0.0, 0.0)
    assert extrinsic.R_C_B == pytest.approx(expected_camera_from_body)
    assert extrinsic.fixed_gimbal_pitch_deg == pytest.approx(-20.0)
    assert "camera optical centre" in extrinsic.evidence

    angle = math.radians(12.0)
    model._camera_rotations[-1] = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    ) @ model._camera_rotations[-1]
    with pytest.raises(ValueError, match="相機／機體方向"):
        model.camera_center_extrinsic(
            vehicle_id="anafi-serial-hash",
            body_frame_id="anafi-camera-center-frd",
            camera_frame_id="anafi-camera-opencv",
        )


def test_guided_alignment_rejects_left_right_geometry_that_is_not_opposed() -> None:
    map_points = iter(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.8, 0.2, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 0.3),
            (0.0, 0.0, -0.3),
        )
    )
    model = GuidedSiteAlignmentModel(
        map_frame_id="map-v1",
        site_frame_id="world-v1",
        get_current_map_position=lambda: next(map_points),
    )
    model.capture_origin()
    evidence = (
        _motion((10, 0, 0, 0), east=0.2),
        _motion((-10, 0, 0, 0), east=-0.2),
        _motion((0, 10, 0, 0), north=0.2),
        _motion((0, -10, 0, 0), north=-0.2),
        _motion((0, 0, 0, 10), down=-0.2),
        _motion((0, 0, 0, -10), down=0.2),
    )
    for step, sample in zip(GUIDED_ALIGNMENT_STEPS[1:], evidence):
        model.finish_step(step.key, (sample, sample))

    with pytest.raises(ValueError, match="左／右"):
        model.solve()

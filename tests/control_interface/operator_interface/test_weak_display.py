"""Weak-pose display: every computed pose reaches the map, flight untouched.

Covers the WeakDisplay acceptance slice:
* VO_ONLY / DEAD_RECKON (including inliers=0) / FAST_TRACK payloads shaped
  like live_localizer_worker._build_success_payload output run through the
  real OperatorApp.update_live_results acceptance chain headless (no Tk).
* Weak fixes mirror into history_weak (capped like history) with DEGRADED
  health -- including ones the confidence hold swallows for flight control.
* live_pose / live_last_xyz keep exactly the pre-change semantics: the hold
  still freezes flight on the last trustworthy pose.
* classify_localization_health mapping is unchanged (OK/DEGRADED/LOST).
* _draw_route_and_history: OK keeps line-only, weak draws hollow diamonds in
  the health colour, and mismatched history/health lengths align explicitly.
* HUD direct status line carries weak_run=consecutive-weak-frames.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

import pytest

import flight_operator_app as app
import operator_tick
from operator_rendering import MapRenderContext, _draw_route_and_history


# --------------------------------------------------------------------------
# payload builders (shaped like _build_success_payload output, dict form)


def _payload(
    seq: int,
    xyz: tuple[float, float, float] | None,
    *,
    mode: str,
    next_mode: str,
    inliers: int,
    direct_status: str | None = None,
    pose_status: str = "VISUALLY_CONFIRMED",
    success: bool | None = None,
    source_frame_stamp_mono: float | None = None,
) -> dict:
    pose = None if xyz is None else {"x": xyz[0], "y": xyz[1], "z": xyz[2], "yaw_raw": 0.1}
    payload = {
        "seq": seq,
        "display_seq": seq,
        "frame_id": f"frame-{seq}",
        "frame_name": f"frame-{seq}",
        "success": (xyz is not None) if success is None else success,
        "pose": pose,
        "pose_status": pose_status,
        "mode": mode,
        "next_mode": next_mode,
        "inliers": inliers,
        "reproj_rms": None,
        "direct_status": direct_status,
        "map_inliers": inliers,
        "vo_inliers": 0,
        "live_points": inliers,
        "dead_reckon_age": 0,
        "reloc_status": None,
    }
    if source_frame_stamp_mono is not None:
        payload["source_frame_stamp_mono"] = source_frame_stamp_mono
    return payload


def _fast(seq: int, x: float) -> dict:
    return _payload(
        seq,
        (x, 2.0, 3.0),
        mode="TRACK",
        next_mode="TRACK",
        inliers=400,
        direct_status="FAST_TRACK",
    )


def _vo(seq: int, x: float) -> dict:
    return _payload(
        seq,
        (x, 2.0, 3.0),
        mode="WEAK_TRACK",
        next_mode="WEAK_TRACK",
        inliers=45,
        direct_status="VO_ONLY",
        pose_status="NONE",
    )


def _dead_reckon_zero(seq: int, x: float) -> dict:
    return _payload(
        seq,
        (x, 2.0, 3.0),
        mode="WEAK_TRACK",
        next_mode="WEAK_TRACK",
        inliers=0,
        direct_status="DEAD_RECKON",
        pose_status="NONE",
    )


def _no_pose(seq: int) -> dict:
    return _payload(
        seq,
        None,
        mode="LOST",
        next_mode="LOST",
        inliers=0,
        direct_status="NO_POSE",
        pose_status="NONE",
    )


# --------------------------------------------------------------------------
# operator double (mirrors test_worker_lifecycle.py, plus the weak trail)


def _classify_stub(operator: SimpleNamespace, result: dict) -> None:
    health = app.classify_localization_health(result)
    if health == "LOST":
        operator.loc_health = "FAIL"
        operator._loc_fail_count += 1
    elif health == "DEGRADED":
        operator.loc_health = "LOW"
        operator._loc_ok_count += 1
    else:
        operator.loc_health = "OK"
        operator._loc_ok_count += 1
    operator.loc_health_inliers = int(result.get("inliers", 0) or 0)
    operator.loc_health_reproj = result.get("reproj_rms")


def _make_operator(batches: list, policy) -> SimpleNamespace:
    recovery_calls: list = []
    operator = SimpleNamespace(
        localizer=SimpleNamespace(poll_results=lambda: batches.pop(0)),
        lost_hold=policy,
        inspecting=True,
        video_display_index=11,
        video_display_frame_name="frame-11",
        _loc_benchmark_pending=None,
        loc_benchmark_active="auto",
        _last_localization_exception_seq=None,
        _last_applied_live_result_display_seq=None,
        _apply_lost_hold_result=lambda result: app.OperatorApp._apply_lost_hold_result(
            operator, result
        ),
        update_localization_metrics=None,  # bound below (needs operator)
        _is_live_backend=lambda: False,
        _integrated_auto_active=lambda: False,
        _engage_real_localization_recovery=lambda reason: recovery_calls.append(reason),
        _last_status_write=float("inf"),
        _last_loc_fail_log=float("inf"),
        _loc_ok_count=0,
        _loc_fail_count=0,
        loc_fps=0.0,
        live_result=None,
        live_result_frame_name="",
        live_last_xyz=None,
        _live_pending=None,
        camera_forward_world=None,
        camera_axes_world=None,
        _integrated_auto_map_frame=None,
        live_heading=0.0,
        live_pose=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        live_locked=False,
        live_new_pose=False,
        _autonomy_pose_snapshot=None,
        _integrated_autonomy=None,
        loc_bridge_run=0,
        boot_holding=lambda: False,
        boot_lock_done=True,
        pose_stabilizer=None,
        yaw_stabilizer=None,
        history_weak=[],
        history_weak_health=[],
        no_loc_count=0,
        _record_no_loc=None,  # bound below (needs operator)
        write_log=lambda _message: None,
    )
    operator.update_localization_metrics = lambda result: _classify_stub(operator, result)
    operator._record_no_loc = lambda: setattr(operator, "no_loc_count", operator.no_loc_count + 1)
    operator.recovery_calls = recovery_calls
    return operator


def _drain(operator: SimpleNamespace, payloads: list[dict]) -> None:
    for _payload in payloads:
        operator.live_new_pose = False
        app.OperatorApp.update_live_results(operator)


_EAST = np.array([1.0, 0.0, 0.0])
_NORTH = np.array([0.0, 0.0, 1.0])
_UP = np.array([0.0, -1.0, 0.0])


def _camera_axes(pitch_deg: float) -> list:
    pitch = math.radians(pitch_deg)
    forward = math.cos(pitch) * _NORTH + math.sin(pitch) * _UP
    right = _EAST
    down = np.cross(forward, right)
    return [right.tolist(), down.tolist(), forward.tolist()]


def _attitude_payload(seq: int, camera_pitch_deg: float) -> dict:
    payload = _fast(seq, 1.0)
    payload.update(
        source_frame_stamp_mono=12.0,
        camera_axes_world=_camera_axes(camera_pitch_deg),
        gimbal_pitch_deg=-20.0,
        gimbal_roll_deg=0.0,
        gimbal_mono_ns=12_000_000_000,
        gimbal_stamp_source="event",
    )
    return payload


def test_camera_attitude_mismatch_rejects_fix_and_requests_relocalization() -> None:
    operator = _make_operator([[_attitude_payload(1, 25.0)]], app.LostHoldPolicy())
    calls = []
    operator._map_axis_basis = lambda: (_EAST, _NORTH, _UP, True)
    operator.localizer.request_relocalize = lambda: calls.append(1)
    _drain(operator, [_attitude_payload(1, 25.0)])

    assert operator.live_new_pose is False
    assert operator.loc_health == "FAIL"
    assert calls == [1]
    assert operator.live_result["error"].startswith("camera attitude mismatch")
    assert operator.live_result["camera_pitch_error_deg"] == pytest.approx(45.0, abs=0.01)


def test_camera_attitude_consistent_fix_is_accepted_and_logged() -> None:
    operator = _make_operator([[_attitude_payload(1, -21.0)]], app.LostHoldPolicy())
    calls = []
    operator._map_axis_basis = lambda: (_EAST, _NORTH, _UP, True)
    operator.localizer.request_relocalize = lambda: calls.append(1)
    _drain(operator, [_attitude_payload(1, -21.0)])

    assert operator.live_new_pose is True
    assert np.allclose(operator.live_last_xyz, [1.0, 2.0, 3.0])
    assert operator.live_result["camera_pitch_error_deg"] == pytest.approx(1.0, abs=0.01)
    assert calls == []


def test_camera_attitude_check_skips_unmeasured_gravity() -> None:
    operator = _make_operator([[_attitude_payload(1, 25.0)]], app.LostHoldPolicy())
    calls = []
    operator._map_axis_basis = lambda: (_EAST, _NORTH, _UP, False)
    operator.localizer.request_relocalize = lambda: calls.append(1)
    _drain(operator, [_attitude_payload(1, 25.0)])

    assert operator.live_new_pose is True
    assert "camera_pitch_error_deg" not in operator.live_result
    assert calls == []


# --------------------------------------------------------------------------
# acceptance: weak poses reach the map-draw input


def test_weak_poses_mirror_into_history_weak_default_policy() -> None:
    fast = [_fast(seq, 1.0 + 0.01 * seq) for seq in range(1, 6)]
    vo = [_vo(seq, 1.1 + 0.01 * seq) for seq in range(6, 11)]
    dr = [_dead_reckon_zero(seq, 1.2 + 0.01 * seq) for seq in range(11, 16)]
    batches = [[payload] for payload in fast + vo + dr]
    operator = _make_operator(batches, app.LostHoldPolicy())

    _drain(operator, fast + vo + dr)

    # 5 VO_ONLY + 5 DEAD_RECKON(inliers=0) mirrored, all DEGRADED.
    assert len(operator.history_weak) == 10
    assert operator.history_weak_health == ["DEGRADED"] * 10
    for point in operator.history_weak:
        assert np.asarray(point, dtype=float).shape == (3,)
        assert bool(np.all(np.isfinite(point)))
    # Flight semantics unchanged: without a hold, weak still flows to the
    # public pose exactly as before this change (autonomy gates on loc_health).
    assert operator.live_new_pose is True
    assert np.allclose(operator.live_last_xyz, [1.35, 2.0, 3.0])
    assert operator.loc_health == "LOW"


def test_drained_weak_points_reach_map_draw_input() -> None:
    # End to end: acceptance output feeds the real map-draw function, with
    # at least 5 frames of each of FAST_TRACK / VO_ONLY / DEAD_RECKON.
    fast = [_fast(seq, 1.0 + 0.01 * seq) for seq in range(1, 6)]
    vo = [_vo(seq, 1.1 + 0.01 * seq) for seq in range(6, 11)]
    dr = [_dead_reckon_zero(seq, 1.2 + 0.01 * seq) for seq in range(11, 16)]
    batches = [[payload] for payload in fast + vo + dr]
    operator = _make_operator(batches, app.LostHoldPolicy())
    _drain(operator, fast + vo + dr)

    draw = _RecordingDraw()
    history = [tuple(float(v) for v in point) for point in operator.history_weak]
    context = _map_context(
        history=[(1.0 + 0.01 * seq, 2.0, 3.0) for seq in range(1, 6)],
        history_health=["OK"] * 5,
        history_weak=history,
        history_weak_health=list(operator.history_weak_health),
    )
    _draw_route_and_history(draw, context)

    assert len(draw.lines) == 1  # OK backbone unchanged
    assert len(draw.polygons) == 10  # 5 VO_ONLY + 5 DEAD_RECKON diamonds
    assert all(kwargs.get("fill") is None for _, kwargs in draw.polygons)


def test_hold_swallowed_weak_still_displays_while_flight_holds() -> None:
    # Production-like policy: hold engages on the 2nd consecutive weak frame.
    policy = app.LostHoldPolicy(low_confidence_results=2, hold_on_low_confidence=True)
    anchor = _fast(1, 1.0)
    weak_run = [_vo(2, 1.01), _vo(3, 1.02), _vo(4, 1.03)]
    lost = _no_pose(5)
    tail = [_vo(6, 1.04), _dead_reckon_zero(7, 1.05)]
    batches = [[payload] for payload in [anchor] + weak_run + [lost] + tail]
    operator = _make_operator(batches, policy)

    _drain(operator, [anchor] + weak_run + [lost] + tail)

    # Every computed weak pose is displayed, including hold-swallowed ones.
    assert len(operator.history_weak) == 5
    assert set(operator.history_weak_health) == {"DEGRADED"}
    # ... while flight froze on the FAST anchor from the very first weak
    # frame (the hold gate swallows every low frame when hold_on is set;
    # the low_streak only gates the ENGAGE recovery event) and stays in hold.
    assert operator.lost_hold.active
    assert np.allclose(operator.live_last_xyz, [1.0, 2.0, 3.0])
    assert operator.live_new_pose is False


def test_predicted_only_mirrors_into_weak_trail() -> None:
    predicted = _payload(
        1,
        (7.0, 8.0, 9.0),
        mode="WEAK_TRACK",
        next_mode="WEAK_TRACK",
        inliers=0,
        pose_status="PREDICTED_ONLY",
        success=False,
    )
    operator = _make_operator([[predicted]], app.LostHoldPolicy())

    app.OperatorApp.update_live_results(operator)

    # Pre-existing behaviour preserved (IMU guess still published) ...
    assert operator.live_new_pose is True
    # ... and the guess is now also on the weak trail.
    assert len(operator.history_weak) == 1
    assert operator.history_weak_health == ["DEGRADED"]
    assert np.allclose(operator.history_weak[0], [7.0, 8.0, 9.0])
    assert operator._autonomy_pose_snapshot is None
    assert not getattr(operator, "_autonomy_pose_predicted", False)


def test_predicted_only_feeds_auto_snapshot_after_visual_lock() -> None:
    predicted = _payload(
        2,
        (7.0, 8.0, 9.0),
        mode="WEAK_TRACK",
        next_mode="WEAK_TRACK",
        inliers=0,
        pose_status="PREDICTED_ONLY",
        success=False,
        source_frame_stamp_mono=12.0,
    )
    predicted["pose"]["yaw_raw"] = 0.4
    operator = _make_operator([[predicted]], app.LostHoldPolicy())
    operator.live_locked = True
    operator._integrated_autonomy = SimpleNamespace(accept_weak_poses=True)

    app.OperatorApp.update_live_results(operator)

    snapshot = operator._autonomy_pose_snapshot
    assert snapshot is not None
    assert not snapshot.position_observed
    assert (snapshot.x, snapshot.y, snapshot.z, snapshot.yaw, snapshot.stamp) == pytest.approx(
        (7.0, 8.0, 9.0, 0.4, 12.0)
    )


def test_weak_trail_is_capped_like_history() -> None:
    operator = _make_operator([], app.LostHoldPolicy())
    result = _vo(1, 1.0)
    for index in range(operator_tick._WEAK_HISTORY_MAX + 20):
        assert operator_tick._record_weak_display_pose(
            operator, result, np.array([float(index), 0.0, 0.0])
        )
    assert len(operator.history_weak) == operator_tick._WEAK_HISTORY_MAX
    assert len(operator.history_weak_health) == operator_tick._WEAK_HISTORY_MAX


def test_classify_mapping_unchanged() -> None:
    assert app.classify_localization_health(_fast(1, 1.0)) == "OK"
    assert app.classify_localization_health(_vo(2, 1.0)) == "DEGRADED"
    assert app.classify_localization_health(_dead_reckon_zero(3, 1.0)) == "DEGRADED"
    assert app.classify_localization_health(_no_pose(4)) == "LOST"


# --------------------------------------------------------------------------
# rendering


class _RecordingDraw:
    def __init__(self) -> None:
        self.lines: list = []
        self.ellipses: list = []
        self.polygons: list = []

    def line(self, xy, **kwargs) -> None:
        self.lines.append((list(xy), kwargs))

    def ellipse(self, xy, **kwargs) -> None:
        self.ellipses.append((tuple(xy), kwargs))

    def polygon(self, xy, **kwargs) -> None:
        self.polygons.append((list(xy), kwargs))


def _map_context(**overrides):
    width = overrides.pop("width", 620)
    height = overrides.pop("height", 430)

    def transform_xyz(points):
        return np.asarray(points, dtype=float).reshape(-1, 3)

    def project_world(point, w, h):  # pragma: no cover - unused here
        return (int(w * 0.5), int(h * 0.5))

    defaults = dict(
        width=width,
        height=height,
        map_zoom=3.2,
        map_radius=12.0,
        map_pan=np.zeros(2),
        no_loc_markers=[],
        route_pts=[],
        history=[],
        history_health=[],
        pose=(0.0, 0.0, 0.0, float("nan")),
        camera_axes=None,
        camera_forward=None,
        transform_xyz=transform_xyz,
        project_world=project_world,
        route_color="#ff3ea5",
        health_color=dict(app.HEALTH_COLOR),
        route_dot_max=200,
        no_loc_max_markers=40,
        video_aspect_ratio=1280 / 720,
        overlay_font=None,
        map_east=np.array([1.0, 0.0, 0.0]),
        map_north=np.array([0.0, 1.0, 0.0]),
        map_up=np.array([0.0, 0.0, 1.0]),
    )
    defaults.update(overrides)
    return MapRenderContext(**defaults)


def test_ok_history_keeps_line_only() -> None:
    draw = _RecordingDraw()
    context = _map_context(
        history=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        history_health=["OK", "OK"],
    )

    _draw_route_and_history(draw, context)

    assert len(draw.lines) == 1
    assert draw.lines[0][1]["fill"] == "#5aa7e8"
    assert draw.ellipses == []
    assert draw.polygons == []


def test_weak_points_draw_hollow_diamonds_in_health_colour() -> None:
    draw = _RecordingDraw()
    context = _map_context(
        history=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        history_health=["OK", "OK"],
        history_weak=[(0.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
        history_weak_health=["DEGRADED", "DEGRADED"],
    )

    _draw_route_and_history(draw, context)

    # OK behaviour untouched: one blue line, no dots.
    assert len(draw.lines) == 1
    assert draw.ellipses == []
    # Weak: one hollow diamond per point, amber outline, no solid fill.
    assert len(draw.polygons) == 2
    for diamond, kwargs in draw.polygons:
        assert len(diamond) == 4
        assert kwargs.get("outline") == app.HEALTH_COLOR["DEGRADED"]
        assert kwargs.get("fill") is None


def test_weak_diamonds_are_centred_on_their_own_points() -> None:
    """Position, not just count: the diamonds used to stack on a leaked x/y."""

    draw = _RecordingDraw()
    context = _map_context(
        history=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        history_health=["OK", "OK"],
        # Far apart on purpose, so identical centres cannot be a coincidence.
        history_weak=[(5.0, 0.0, 0.0), (-5.0, 0.0, 0.0)],
        history_weak_health=["DEGRADED", "DEGRADED"],
    )

    _draw_route_and_history(draw, context)

    assert len(draw.polygons) == 2
    centres = [
        (sum(p[0] for p in poly) / 4.0, sum(p[1] for p in poly) / 4.0) for poly, _ in draw.polygons
    ]
    assert centres[0] != centres[1]
    # +5 is right of -5 on screen, and neither sits on the history tail.
    assert centres[0][0] > centres[1][0]
    history_tail = (draw.lines[0][0][-1][0], draw.lines[0][0][-1][1])
    assert all(centre != history_tail for centre in centres)


def test_weak_fixes_render_without_a_route_or_history() -> None:
    """The first weak fix of a flight arrives before any OK history exists."""

    draw = _RecordingDraw()
    context = _map_context(
        route_pts=[],
        history=[],
        history_health=[],
        history_weak=[(1.0, 0.0, 0.0)],
        history_weak_health=["DEGRADED"],
    )

    _draw_route_and_history(draw, context)

    assert len(draw.polygons) == 1


def test_mismatched_history_health_aligns_explicitly() -> None:
    draw = _RecordingDraw()
    context = _map_context(
        history=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        history_health=["LOW"],
    )

    _draw_route_and_history(draw, context)

    # The flown line still uses every point; markers cover the shared prefix.
    assert len(draw.lines) == 1
    assert len(draw.lines[0][0]) == 3
    assert len(draw.ellipses) == 1


# --------------------------------------------------------------------------
# HUD


def _hud_operator() -> SimpleNamespace:
    return SimpleNamespace(
        loc_hold_engage_count=0,
        loc_recovery_fix_count=0,
        loc_weak_run=0,
        loc_recovery_text="",
    )


def test_hud_direct_line_reports_consecutive_weak_frames() -> None:
    operator = _hud_operator()

    app.OperatorApp._update_localization_recovery(operator, _fast(1, 1.0))
    assert operator.loc_weak_run == 0
    assert "weak_run=0" in operator.loc_recovery_text

    app.OperatorApp._update_localization_recovery(operator, _vo(2, 1.0))
    app.OperatorApp._update_localization_recovery(operator, _vo(3, 1.0))
    assert operator.loc_weak_run == 2
    assert "weak_run=2" in operator.loc_recovery_text
    assert "direct VO_ONLY" in operator.loc_recovery_text

    app.OperatorApp._update_localization_recovery(operator, _fast(4, 1.0))
    assert operator.loc_weak_run == 0

    app.OperatorApp._update_localization_recovery(operator, _no_pose(5))
    assert operator.loc_weak_run == 0


def test_hud_without_direct_status_has_no_weak_run_fragment() -> None:
    operator = _hud_operator()
    edm_weak = _payload(
        1,
        (1.0, 2.0, 3.0),
        mode="WEAK_TRACK",
        next_mode="WEAK_TRACK",
        inliers=20,
        direct_status=None,
    )
    app.OperatorApp._update_localization_recovery(operator, edm_weak)


def test_held_weak_feeds_autonomy_snapshot_only_when_opted_in() -> None:
    for flag, expected_x in ((False, 1.0), (True, 1.03)):
        policy = app.LostHoldPolicy(low_confidence_results=2, hold_on_low_confidence=True)
        anchor = _fast(1, 1.0)
        weak = [_vo(2, 1.01), _vo(3, 1.02), _vo(4, 1.03)]
        for payload in weak:
            payload["confidence_hold_active"] = True
        operator = _make_operator([[payload] for payload in [anchor] + weak], policy)
        operator._integrated_autonomy = SimpleNamespace(accept_weak_poses=flag)

        _drain(operator, [anchor])
        frozen_live = operator.live_pose.copy()
        _drain(operator, weak)

        assert operator.lost_hold.active
        assert np.allclose(operator.live_pose, frozen_live)
        snapshot = operator._autonomy_pose_snapshot
        assert snapshot is not None
        assert snapshot.x == pytest.approx(expected_x)
        assert snapshot.position_observed is True


def test_weak_trail_records_estimate_source_kind() -> None:
    import operator_tick

    operator = _make_operator([], app.LostHoldPolicy())
    operator.history_weak_kind = []
    klt = _vo(1, 1.0)
    klt["candidate_mode"] = "klt_fast"
    imu = _vo(2, 2.0)
    imu["direct_status"] = "IMU_BRIDGE"
    vo = _vo(3, 3.0)

    assert operator_tick._record_weak_display_pose(operator, klt, np.array([1.0, 2.0, 3.0]))
    assert operator_tick._record_weak_display_pose(operator, imu, np.array([2.0, 2.0, 3.0]))
    assert operator_tick._record_weak_display_pose(operator, vo, np.array([3.0, 2.0, 3.0]))

    assert operator.history_weak_kind == ["KLT", "IMU", "KLT"]
    assert len(operator.history_weak) == 3


@pytest.mark.parametrize("source", ["KLT", "IMU"])
def test_held_estimate_renders_camera_at_weak_center_without_moving_flight(
    monkeypatch,
    tmp_path,
    source,
) -> None:
    """No backend/connection: production acceptance + app map renderer + PIL."""
    import os
    from PIL import Image, ImageDraw
    from operator_rendering import draw_map_overlays

    monkeypatch.setattr("localization_result_ui.time.monotonic", lambda: 12.1)
    anchor = _fast(1, 1.0)
    weak = _vo(2, 1.3)
    weak.update(
        confidence_hold_active=True,
        source_frame_stamp_mono=12.0,
        camera_axes_world=[[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        camera_forward_world=[1.0, 0.0, 0.0],
        camera_attitude_kind="full",
    )
    if source == "IMU":
        weak.update(
            direct_status="IMU_BRIDGE", imu_bridge=True, camera_attitude_kind="heading_only"
        )
    policy = app.LostHoldPolicy(low_confidence_results=2, hold_on_low_confidence=True)
    operator = _make_operator([[anchor], [weak]], policy)
    operator.history_weak_kind = []
    _drain(operator, [anchor])
    frozen_pose = operator.live_pose.copy()
    _drain(operator, [weak])
    np.testing.assert_array_equal(operator.live_pose, frozen_pose)
    np.testing.assert_allclose(operator.history_weak[-1], [1.3, 2.0, 3.0])
    # Map state only; no drone backend, widgets, command hooks or telemetry IO.
    operator._is_live_backend = lambda: True
    operator.map_base_image = lambda w, h: Image.new("RGB", (w, h), "#15181c")
    operator.map_base_key = lambda w, h: (w, h, tuple(operator.map_pan))
    operator.map_zoom, operator.map_radius, operator.map_pan = 3.2, 12.0, np.zeros(2)
    operator.map_yaw = operator.map_pitch = operator.map_roll = 0.0
    operator.route_pts, operator.history, operator.history_health, operator.no_loc_markers = (
        [],
        [],
        [],
        [],
    )
    operator._map_axis_basis = lambda: (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        True,
    )
    # Oblique projection reveals the actual right/down picture plane, not an
    # arbitrary screen-aligned rectangle.
    operator.transform_xyz = lambda points: (
        np.asarray(points, dtype=float)
        @ np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.8, 0.6],
                [0.0, -0.6, 0.8],
            ]
        )
    )
    scale = 430 * 0.46 * operator.map_zoom / operator.map_radius
    operator.project_world = lambda p, w, h: (
        int(w * 0.5 + operator.transform_xyz(np.asarray(p))[0] * scale),
        int(h * 0.5 - operator.transform_xyz(np.asarray(p))[1] * scale),
    )
    state = SimpleNamespace(pose=frozen_pose.copy(), gimbal_pitch_deg=-65.0)

    def render_estimate():
        context = app.OperatorApp._map_render_context(operator, 620, 430, state)
        image = operator.map_base_image(620, 430)
        draw_map_overlays(ImageDraw.Draw(image), context)
        return image, context

    first_image, context = render_estimate()
    np.testing.assert_allclose(context.pose[:3], [1.3, 2.0, 3.0])
    assert context.camera_estimate_kind == source
    assert context.camera_attitude == ("heading" if source == "IMU" else "full")
    assert context.camera_estimate_stale is False
    color = np.array([255, 214, 10] if source == "IMU" else [63, 191, 127])
    assert np.count_nonzero(np.all(np.asarray(first_image) == color, axis=2)) > 40
    first_image.save(tmp_path / f"{source.lower()}_held_estimate.png")
    # Rotate this estimate at the same point while the held flight pose stays
    # fixed: the visible camera must change rather than borrowing that pose.
    if source == "IMU":
        operator.live_result["pose"]["yaw_raw"] = 0.9
    else:
        angle = math.radians(35.0)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        operator.live_result["camera_axes_world"] = (
            np.asarray(weak["camera_axes_world"]) @ rotation.T
        ).tolist()
    second_image, _ = render_estimate()
    assert not np.array_equal(np.asarray(first_image), np.asarray(second_image))
    second_image.save(tmp_path / f"{source.lower()}_rotated_estimate.png")
    np.testing.assert_array_equal(operator.live_pose, frozen_pose)
    np.testing.assert_array_equal(state.pose, frozen_pose)
    # Optional real Tk proof on an Xvfb display, still without constructing a
    # backend or the operator controls. Parent verification opts in explicitly.
    if os.environ.get("SFM_ESTIMATED_CAMERA_TK_PROOF") == "1":
        import tkinter as tk
        from PIL import ImageGrab, ImageTk

        root = tk.Tk()
        try:
            root.title(f"Offline {source} estimated camera attitude — no hardware")
            root.geometry("1240x430+0+0")
            left, right = ImageTk.PhotoImage(first_image), ImageTk.PhotoImage(second_image)
            tk.Label(root, image=left).pack(side="left")
            tk.Label(root, image=right).pack(side="left")
            root.lift()
            root.focus_force()

            def capture():
                ImageGrab.grab(
                    bbox=(
                        root.winfo_rootx(),
                        root.winfo_rooty(),
                        root.winfo_rootx() + root.winfo_width(),
                        root.winfo_rooty() + root.winfo_height(),
                    ),
                    xdisplay=os.environ.get("DISPLAY"),
                ).save(tmp_path / f"{source.lower()}_tk_proof.png")
                root.quit()

            root.after(600, capture)
            root.mainloop()
        finally:
            root.destroy()
    print(f"Offline camera rendering proof: {tmp_path}")


def test_backfilled_submap_on_failed_pose_stays_out_of_control_history() -> None:
    anchored = _fast(1, 1.0)
    no_fix = _no_pose(2)
    no_fix["workmap_output_source"] = "baseline"
    no_fix["workmap_trajectory_backfill"] = {
        "segment_id": 12,
        "points": [[189, 3.0, 2.0, 1.0], [919, 4.0, 2.0, 1.0]],
    }
    operator = _make_operator([[anchored], [no_fix]], app.LostHoldPolicy())
    operator.history = [np.array([0.0, 2.0, 3.0])]
    app.OperatorApp.update_live_results(operator)
    pose_before = operator.live_pose.copy()
    last_before = operator.live_last_xyz.copy()
    snapshot_before = operator._autonomy_pose_snapshot
    history_before = [point.copy() for point in operator.history]
    operator.live_new_pose = False
    app.OperatorApp.update_live_results(operator)

    assert operator.workmap_trajectories == {12: [(3.0, 2.0, 1.0), (4.0, 2.0, 1.0)]}
    np.testing.assert_array_equal(operator.live_pose, pose_before)
    np.testing.assert_array_equal(operator.live_last_xyz, last_before)
    np.testing.assert_array_equal(operator.history, history_before)
    assert operator._autonomy_pose_snapshot is snapshot_before
    assert operator.history_weak == []
    assert operator.live_new_pose is False

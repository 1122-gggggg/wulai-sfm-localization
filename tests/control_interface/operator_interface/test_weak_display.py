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

from types import SimpleNamespace

import numpy as np

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
) -> dict:
    pose = (
        None
        if xyz is None
        else {"x": xyz[0], "y": xyz[1], "z": xyz[2], "yaw_raw": 0.1}
    )
    return {
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


def _fast(seq: int, x: float) -> dict:
    return _payload(
        seq, (x, 2.0, 3.0), mode="TRACK", next_mode="TRACK",
        inliers=400, direct_status="FAST_TRACK",
    )


def _vo(seq: int, x: float) -> dict:
    return _payload(
        seq, (x, 2.0, 3.0), mode="WEAK_TRACK", next_mode="WEAK_TRACK",
        inliers=45, direct_status="VO_ONLY",
        pose_status="NONE",
    )


def _dead_reckon_zero(seq: int, x: float) -> dict:
    return _payload(
        seq, (x, 2.0, 3.0), mode="WEAK_TRACK", next_mode="WEAK_TRACK",
        inliers=0, direct_status="DEAD_RECKON",
        pose_status="NONE",
    )


def _no_pose(seq: int) -> dict:
    return _payload(
        seq, None, mode="LOST", next_mode="LOST",
        inliers=0, direct_status="NO_POSE",
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
    operator.update_localization_metrics = (
        lambda result: _classify_stub(operator, result)
    )
    operator._record_no_loc = lambda: setattr(
        operator, "no_loc_count", operator.no_loc_count + 1
    )
    operator.recovery_calls = recovery_calls
    return operator


def _drain(operator: SimpleNamespace, payloads: list[dict]) -> None:
    for _payload in payloads:
        operator.live_new_pose = False
        app.OperatorApp.update_live_results(operator)
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
    policy = app.LostHoldPolicy(
        low_confidence_results=2, hold_on_low_confidence=True
    )
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
        1, (7.0, 8.0, 9.0), mode="WEAK_TRACK", next_mode="WEAK_TRACK",
        inliers=0, pose_status="PREDICTED_ONLY", success=False,
    )
    operator = _make_operator([[predicted]], app.LostHoldPolicy())

    app.OperatorApp.update_live_results(operator)

    # Pre-existing behaviour preserved (IMU guess still published) ...
    assert operator.live_new_pose is True
    # ... and the guess is now also on the weak trail.
    assert len(operator.history_weak) == 1
    assert operator.history_weak_health == ["DEGRADED"]
    assert np.allclose(operator.history_weak[0], [7.0, 8.0, 9.0])


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
        collision_center=None,
        collision_radius=0.0,
        collision_status="DISABLED",
        collision_point=None,
        collision_preview=False,
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
        1, (1.0, 2.0, 3.0), mode="WEAK_TRACK", next_mode="WEAK_TRACK",
        inliers=20, direct_status=None,
    )
    app.OperatorApp._update_localization_recovery(operator, edm_weak)
    assert operator.loc_weak_run == 1
    assert "weak_run" not in operator.loc_recovery_text

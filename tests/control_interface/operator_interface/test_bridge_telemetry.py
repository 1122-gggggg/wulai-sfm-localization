"""Bridge telemetry: KLT-bridged poses are accepted for continuity but counted."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import operator_tick


def _app() -> SimpleNamespace:
    return SimpleNamespace(
        live_last_xyz=None,
        live_heading=None,
        live_pose=np.zeros(4, dtype=float),
        live_locked=False,
        live_new_pose=False,
        live_result_frame_name="f000123",
        _integrated_auto_map_frame=None,
        boot_holding=lambda: False,
        write_log=lambda *a, **k: None,
    )


def test_bridge_frames_increment_run_counter() -> None:
    app = _app()
    xyz = np.array([1.0, 2.0, 3.0])
    operator_tick._accept_live_pose(
        app, {"candidate_mode": "klt_bridge", "bridge": True}, xyz)
    assert app.loc_bridge_run == 1
    operator_tick._accept_live_pose(
        app, {"candidate_mode": "klt_bridge", "bridge": True}, xyz)
    assert app.loc_bridge_run == 2
    assert app.live_locked is True


def test_edm_frame_resets_bridge_run_counter() -> None:
    app = _app()
    xyz = np.array([1.0, 2.0, 3.0])
    operator_tick._accept_live_pose(
        app, {"candidate_mode": "klt_bridge", "bridge": True}, xyz)
    assert app.loc_bridge_run == 1
    operator_tick._accept_live_pose(app, {"candidate_mode": "edm_track"}, xyz)
    assert app.loc_bridge_run == 0


def test_async_fast_path_bridges_are_counted_too() -> None:
    # The async fast path reports candidate_mode "klt_fast" with pose_status
    # KLT_BRIDGED, not "klt_bridge"; counting only the in-tracker spelling left
    # the async default (2026-09-04..09-05) with a counter that never moved.
    app = _app()
    xyz = np.array([1.0, 2.0, 3.0])
    operator_tick._accept_live_pose(
        app, {"candidate_mode": "klt_fast", "pose_status": "KLT_BRIDGED"}, xyz)
    assert app.loc_bridge_run == 1
    operator_tick._accept_live_pose(
        app, {"candidate_mode": "track", "pose_status": "VISUALLY_CONFIRMED"}, xyz)
    assert app.loc_bridge_run == 0

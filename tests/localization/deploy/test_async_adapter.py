"""AsyncEDMTrackerAdapter contract tests (no GPU, no tracker construction)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY = REPO_ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from edm_localizer_adapter import AsyncEDMTrackerAdapter  # noqa: E402


def _adapter(async_loc, trk=None) -> AsyncEDMTrackerAdapter:
    import threading
    adapter = object.__new__(AsyncEDMTrackerAdapter)
    adapter.async_loc = async_loc
    adapter.trk = trk
    adapter.map_frame = None
    adapter.state = SimpleNamespace(last_pose=None, mode="BOOT_INIT",
                                    last_center=None, prev_center=None,
                                    last_yaw=None, prev_yaw=None,
                                    last_refs=[], fail_count=0, bad_count=0)
    adapter._last_info = {}
    adapter._last_slow_info = {}
    adapter._state_lock = threading.Lock()
    return adapter


def _result(status, pose=None, **extra):
    payload = {
        "pose": pose,
        "pose_status": status,
        "candidate_mode": extra.pop("candidate_mode", "klt_fast"),
        "inliers": 40,
        "tracked": 50,
        "stale": False,
        "reproj_rms": 1.0,
    }
    payload.update(extra)
    stub = SimpleNamespace(as_dict=lambda: payload)
    return stub


def test_bridged_pose_maps_to_worker_pose() -> None:
    adapter = _adapter(SimpleNamespace(
        feed_frame=lambda gray, stamp: _result(
            "KLT_BRIDGED", {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.5}),
    ))
    frame = np.zeros((720, 1280, 3), np.uint8)
    pose = adapter.localize_frame(frame, capture_stamp=1.0)
    assert pose is not None
    assert (pose.x, pose.y, pose.z) == (1.0, 2.0, 3.0)
    assert adapter._last_info["pose_status"] == "KLT_BRIDGED"
    assert adapter._last_info["candidate_mode"] == "klt_fast"


def test_lost_status_yields_no_pose_but_records_info() -> None:
    adapter = _adapter(SimpleNamespace(
        feed_frame=lambda gray, stamp: _result("LOST"),
    ))
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert adapter.localize_frame(frame, capture_stamp=1.0) is None
    assert adapter._last_info["pose_status"] == "LOST"


def test_slow_keyframe_rejects_non_visual() -> None:
    adapter = _adapter(None, trk=SimpleNamespace(
        localize=lambda bgr, capture_stamp: {"ok": True, "pose_status": "KLT_BRIDGED"},
    ))
    gray = np.zeros((576, 1024), np.uint8)
    assert adapter._slow_keyframe(gray, 1.0) is None


def test_slow_keyframe_accepts_visual_anchor() -> None:
    mat = np.eye(4)
    trk = SimpleNamespace(
        localize=lambda bgr, capture_stamp: {
            "ok": True, "pose_status": "VISUALLY_CONFIRMED",
            "inliers": 90, "inlier_ratio": 0.8, "reproj_rms": 1.2,
            "inlier_grid_cells": 8,
        },
        _last_accepted_cam_from_world=mat,
        _klt_2d=np.zeros((90, 2), np.float32),
        _klt_3d=np.zeros((90, 3), np.float32),
    )
    adapter = _adapter(None, trk=trk)
    gray = np.zeros((576, 1024), np.uint8)
    out = adapter._slow_keyframe(gray, 1.0)
    assert out is not None and out["ok"] is True
    assert out["inliers"] == 90
    assert out["pose_status"] == "VISUALLY_CONFIRMED"

def test_slow_keyframe_caches_telemetry_for_gate() -> None:
    mat = np.eye(4)
    trk = SimpleNamespace(
        localize=lambda bgr, capture_stamp: {
            "ok": True, "pose_status": "VISUALLY_CONFIRMED",
            "inliers": 90, "inlier_ratio": 0.8, "reproj_rms": 1.2,
            "inlier_grid_cells": 8, "vpr_ms": 5.0, "match_ms": 20.0,
            "pnp_ms": 1.0, "refs": ["r1"], "reference_count": 1,
            "requested_reference_count": 2, "candidate_mode": "track",
            "global_retrieval_calls": 3,
        },
        _last_accepted_cam_from_world=mat,
        _klt_2d=np.zeros((90, 2), np.float32),
        _klt_3d=np.zeros((90, 3), np.float32),
    )
    adapter = _adapter(None, trk=trk)
    out = adapter._slow_keyframe(np.zeros((576, 1024), np.uint8), 1.0)
    assert out is not None and out["match_ms"] == 20.0
    assert adapter._last_slow_info["match_ms"] == 20.0
    assert adapter._last_slow_info["slow_stamp"] == 1.0


def test_fast_path_surfaces_cached_slow_telemetry() -> None:
    adapter = _adapter(SimpleNamespace(
        feed_frame=lambda gray, stamp: _result(
            "KLT_BRIDGED", {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.5}),
    ))
    adapter._last_slow_info = {"match_ms": 20.0, "pnp_ms": 1.0, "vpr_ms": 5.0,
                               "slow_stamp": 0.9}
    pose = adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8),
                                  capture_stamp=1.0)
    assert pose is not None
    assert adapter._last_info["match_ms"] == 20.0
    assert adapter._last_info["pnp_ms"] == 1.0
    assert adapter._last_info["slow_stamp"] == 0.9


def test_map_frame_heading_wins_over_fast_yaw() -> None:
    adapter = _adapter(SimpleNamespace(
        feed_frame=lambda gray, stamp: _result(
            "KLT_BRIDGED",
            {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.5,
             "rotation": np.eye(3)}),
    ))
    adapter.map_frame = SimpleNamespace(heading=lambda fwd: 1.25)
    pose = adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8),
                                  capture_stamp=1.0)
    assert pose is not None and pose.yaw == 1.25


def test_observe_and_ensure_models_forward_to_slow_tracker() -> None:
    seen = {}
    loc = SimpleNamespace(megaloc=object(),
                          matcher=SimpleNamespace(
                              warmup_fused_coarse=lambda: seen.setdefault("warm", True)))
    trk = SimpleNamespace(loc=loc,
                          observe_fused_state=lambda s: seen.setdefault("obs", s))
    adapter = _adapter(None, trk=trk)
    adapter.ensure_models()
    assert seen.get("warm") is True
    adapter.observe_fused_state("s")
    assert seen.get("obs") == "s"


def test_push_seeded_state_reaches_slow_tracker() -> None:
    st = SimpleNamespace(state="TRACK", center=None, yaw=None, last_refs=[],
                         misses=0, velocity=np.zeros(3), last_capture_stamp=1.0)
    trk = SimpleNamespace(st=st,
                          localize=lambda bgr, capture_stamp: {"ok": False})
    adapter = _adapter(None, trk=trk)
    adapter.state.mode = "LOST"
    adapter.state.fail_count = 3
    assert adapter._slow_keyframe(np.zeros((576, 1024), np.uint8), 2.0) is None
    assert st.state == "LOST" and st.misses == 3


def test_pull_mirrors_slow_recovery_state() -> None:
    st = SimpleNamespace(state="LOST", center=None, yaw=None, last_refs=[],
                         misses=3, velocity=None, last_capture_stamp=1.0)
    mat = np.eye(4)

    def _accept(bgr, capture_stamp):
        # Emulate ProductionEDMTracker accepting a pose: mutate st, then report.
        st.state = "TRACK"
        st.center = np.zeros(3, np.float32)
        st.yaw = 0.5
        st.last_refs = ["r1"]
        st.misses = 1
        return {"ok": True, "pose_status": "VISUALLY_CONFIRMED",
                "inliers": 90, "inlier_ratio": 0.8, "reproj_rms": 1.2,
                "inlier_grid_cells": 8}

    trk = SimpleNamespace(
        st=st,
        localize=_accept,
        _last_accepted_cam_from_world=mat,
        _klt_2d=np.zeros((90, 2), np.float32),
        _klt_3d=np.zeros((90, 3), np.float32),
    )
    adapter = _adapter(None, trk=trk)
    out = adapter._slow_keyframe(np.zeros((576, 1024), np.uint8), 2.0)
    assert out is not None
    assert adapter.state.mode == "TRACK"
    assert adapter.state.last_refs == ["r1"]
    assert adapter.state.fail_count == 1


def test_clear_tracking_history_resets_fast_and_slow() -> None:
    calls = []
    st = SimpleNamespace(state="LOST", center=np.ones(3), yaw=1.0,
                         last_refs=["r9"], misses=5)
    trk = SimpleNamespace(
        loc=SimpleNamespace(),
        _clear_visual_motion_cache=lambda: calls.append("visual"),
        _clear_klt_cache=lambda: calls.append("klt"),
        pose_guided=SimpleNamespace(reset=lambda: calls.append("guided")),
    )
    async_loc = SimpleNamespace(
        mailbox=SimpleNamespace(clear=lambda: calls.append("mailbox")),
        transform_chain=SimpleNamespace(clear=lambda: calls.append("chain")),
        sync_carry=SimpleNamespace(reset=lambda: calls.append("carry")),
        fast_path=SimpleNamespace(reset=lambda: calls.append("fast")),
        scheduler=SimpleNamespace(reset=lambda: calls.append("sched")),
    )
    # AsyncLocalizer.reset coordinates the five; emulate for the fake.
    async_loc.reset = lambda: [getattr(async_loc, k).clear()
                               if k in ("mailbox", "transform_chain")
                               else getattr(async_loc, k).reset()
                               for k in ("mailbox", "transform_chain", "sync_carry",
                                         "fast_path", "scheduler")]
    adapter = _adapter(async_loc, trk=trk)
    adapter.cfg = SimpleNamespace(adaptive_jump_history_size=120)
    trk.st = st
    adapter._clear_tracking_history()
    assert adapter.state.last_center is None
    assert adapter.state.last_refs == []
    assert adapter._last_slow_info == {}
    for key in ("visual", "klt", "guided", "mailbox", "chain", "carry",
                "fast", "sched"):
        assert key in calls, key


def test_prev_pose_trails_last_pose_every_frame() -> None:
    adapter = _adapter(SimpleNamespace(
        feed_frame=lambda gray, stamp: _result("LOST"),
    ))
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert adapter.localize_frame(frame, capture_stamp=1.0) is None
    assert adapter.state.prev_pose is None
    adapter.async_loc = SimpleNamespace(
        feed_frame=lambda gray, stamp: _result(
            "KLT_BRIDGED", {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.5}),
    )
    first = adapter.localize_frame(frame, capture_stamp=2.0)
    assert first is not None and adapter.state.prev_pose is None
    second = adapter.localize_frame(frame, capture_stamp=3.0)
    assert second is not None
    assert adapter.state.prev_pose is first
    assert adapter.state.last_pose is second


def test_slow_keyframe_rescales_seeds_to_edm_grid() -> None:
    from edm_matcher import EDM_H, EDM_W
    cam_w, cam_h = 1280, 720
    mat = np.eye(4)
    trk = SimpleNamespace(
        cam=SimpleNamespace(width=cam_w, height=cam_h),
        localize=lambda bgr, capture_stamp: {
            "ok": True, "pose_status": "VISUALLY_CONFIRMED",
            "inliers": 90, "inlier_ratio": 0.8, "reproj_rms": 1.2,
            "inlier_grid_cells": 8,
        },
        _last_accepted_cam_from_world=mat,
        _klt_2d=np.array([[1280.0, 720.0]], np.float32),
        _klt_3d=np.zeros((1, 3), np.float32),
    )
    adapter = _adapter(None, trk=trk)
    out = adapter._slow_keyframe(np.zeros((576, 1024), np.uint8), 1.0)
    assert out is not None
    got = np.asarray(out["inlier_2d"], dtype=float)[0]
    assert abs(got[0] - EDM_W) < 1e-3
    assert abs(got[1] - EDM_H) < 1e-3


def _slow_state(center=(0.0, 0.0, 0.0), stamp=90.0):
    return SimpleNamespace(state="TRACK", center=np.array(center, np.float32),
                           yaw=0.0, velocity=None, last_capture_stamp=stamp,
                           last_refs=[], misses=0)


def _publishing_adapter(trk):
    return _adapter(SimpleNamespace(
        feed_frame=lambda gray, stamp: _result(
            "KLT_BRIDGED", {"x": 10.0, "y": 0.0, "z": 0.0, "yaw": 0.1}),
    ), trk=trk)


def test_fast_prior_refreshes_stale_slow_prior() -> None:
    trk = SimpleNamespace(st=_slow_state())
    adapter = _publishing_adapter(trk)
    adapter._pull_state_from_slow()  # provenance anchor
    pose = adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8),
                                  capture_stamp=100.0)
    assert pose is not None
    adapter._push_state_to_slow()
    # Prior fields follow the fast bridge; recovery state is untouched.
    assert list(trk.st.center) == [10.0, 0.0, 0.0]
    assert list(trk.st.velocity) == [1.0, 0.0, 0.0]
    assert trk.st.last_capture_stamp == 100.0
    assert trk.st.state == "TRACK" and trk.st.misses == 0
    assert trk.st.last_refs == []
    assert adapter._last_info["fast_prior_feedbacks"] == 0  # counted on push, read next frame
    adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8),
                           capture_stamp=101.0)
    assert adapter._last_info["fast_prior_feedbacks"] == 1


def test_worker_seed_beats_fast_prior() -> None:
    trk = SimpleNamespace(st=_slow_state())
    adapter = _publishing_adapter(trk)
    adapter._pull_state_from_slow()
    adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8),
                           capture_stamp=100.0)
    # Worker injects LOST between keyframes (bench mode / fault injection).
    adapter.state.mode = "LOST"
    adapter.state.fail_count = 3
    adapter._push_state_to_slow()
    assert trk.st.state == "LOST" and trk.st.misses == 3
    assert list(trk.st.center) == [0.0, 0.0, 0.0]  # fast pose NOT applied
    assert trk.st.last_capture_stamp == 90.0
    assert getattr(adapter, "_fast_prior_feedbacks", 0) == 0


def test_stale_fast_prior_is_ignored() -> None:
    trk = SimpleNamespace(st=_slow_state())
    adapter = _publishing_adapter(trk)
    adapter._pull_state_from_slow()
    adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8),
                           capture_stamp=100.0)
    trk.st.last_capture_stamp = 110.0  # slow moved beyond the fast prior
    adapter._push_state_to_slow()
    assert list(trk.st.center) == [0.0, 0.0, 0.0]
    assert trk.st.last_capture_stamp == 110.0


def test_clear_drops_fast_prior_channel() -> None:
    trk = SimpleNamespace(st=_slow_state())
    adapter = _publishing_adapter(trk)
    adapter._pull_state_from_slow()
    adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8),
                           capture_stamp=100.0)
    assert adapter._fast_prior is not None
    adapter._clear_tracking_history()
    assert adapter._fast_prior is None
    assert adapter._last_push_snapshot is None


def test_async_pose_publishes_camera_axes_for_the_operator_map() -> None:
    # Regression: the async adapter became the code default while only the sync
    # adapter published camera_axes_world/camera_forward_world. The operator map
    # then fell through to the yaw symbol -- which carries the TRAVEL direction,
    # not the camera's -- and _autonomy_pose, which requires a camera forward,
    # returned None for every frame.
    rotation = np.array([
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    adapter = _adapter(SimpleNamespace(
        feed_frame=lambda gray, stamp: _result(
            "VISUALLY_CONFIRMED",
            {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.0, "rotation": rotation},
        ),
    ))
    adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert adapter._last_info["camera_axes_world"] == rotation.tolist()
    # Row 2 of cam_from_world is the optical axis in world coordinates.
    assert adapter._last_info["camera_forward_world"] == [1.0, 0.0, 0.0]


def test_async_camera_axes_absent_without_a_visual_rotation() -> None:
    adapter = _adapter(SimpleNamespace(
        feed_frame=lambda gray, stamp: _result("NO_ANCHOR"),
    ))
    adapter.localize_frame(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert adapter._last_info["camera_axes_world"] is None
    assert adapter._last_info["camera_forward_world"] is None

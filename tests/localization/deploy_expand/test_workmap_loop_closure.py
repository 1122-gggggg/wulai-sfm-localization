"""A detached submap is merged onto GlueMap through multi-frame anchors and a Sim(3) stitch."""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from online_workmap import LOOP_MAX_CHAIN, OnlineWorkMapTracker, SessionWorkMap
from workmap_stitch import solve_stitch_snapshot
from workmap_submap_graph import solve_submap_graph
from workmap_submaps import (
    ANCHOR_MAX_SPAN,
    ANCHOR_MIN_COUNT,
    AnchorPair,
    Sim3,
    SubmapRegistry,
    anchor_consistency,
    fit_anchor_similarity,
)

K = np.array([[180.0, 0.0, 160.0], [0.0, 180.0, 120.0], [0.0, 0.0, 1.0]])
STEP = np.array([0.5, 0.0, 0.0])
#: Submap frame -> GlueMap: X_G = s R X_L + t.
TRUE = Sim3(
    1.1,
    cv2.Rodrigues(np.array([[0.0], [0.0], [np.radians(3.0)]]))[0],
    np.array([0.2, 0.0, 0.0]),
)
GLUE_IDS = np.arange(101, 109)


def _pose(center):
    return np.column_stack((np.eye(3), -np.asarray(center, dtype=float)))


def _project(xyz, pose):
    camera = np.atleast_2d(xyz) @ pose[:, :3].T + pose[:, 3]
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2, None]


def _local(true_pose):
    """The pose a submap-frame tracker would hold for a camera whose GlueMap pose is ``true_pose``."""
    return TRUE.inverse().apply_pose(true_pose)


def _submap(tmp_path):
    """Keyframe 0 (submap 0) sees GlueMap; keyframes 1-4 belong to submap 1 in its own frame."""
    rng = np.random.default_rng(20260929)
    true_centers = [STEP * i for i in range(6)]
    frozen = rng.uniform([-0.6, -0.5, 3.5], [0.6, 0.5, 5.0], size=(8, 3))
    session = SessionWorkMap(
        tmp_path / "session",
        session_id="merge-session",
        base_release_id="immutable-r1",
        coordinate_frame_id="fixed-global-frame",
        min_reference_points=6,
    )
    truth, landmarks, hosted = {}, {}, {}
    next_id = -1
    for host in range(1, 5):
        points = rng.uniform([-0.6, -0.5, 3.5], [0.6, 0.5, 5.0], size=(10, 3)) + true_centers[host]
        for point in points:
            key, next_id = next_id, next_id - 1
            truth[key], hosted[key] = point, host
            local_point = TRUE.inverse().apply_points(point)[0]
            views = [
                (o, 1.0 + 0.1 * o, _project(point, _pose(true_centers[o]))[0], _local(_pose(true_centers[o])))
                for o in (host, min(host + 1, 4))
            ]
            landmarks[key] = {
                "xyz": local_point,
                "observations": tuple(views),
                "last_inlier": views[-1][:3],
                "parallax_deg": 3.0,
                "valid_observation_count": 2,
            }
    for key, revision in session.archive_landmarks(landmarks, submap_id=1).items():
        landmarks[key]["revision"] = revision
    image = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    for ordinal in range(5):
        seen = [key for key, host in hosted.items() if host in (ordinal, ordinal - 1)]
        ids = np.array(seen, dtype=np.int64)
        true_pose = _pose(true_centers[ordinal])
        xy = _project(np.stack([truth[key] for key in seen]), true_pose) if seen else np.zeros((0, 2))
        xyz = np.stack([landmarks[key]["xyz"] for key in seen]) if seen else np.zeros((0, 3))
        pose = true_pose
        if ordinal == 0:
            ids = np.concatenate((GLUE_IDS, ids))
            xy = np.concatenate((_project(frozen, true_pose), xy))
            xyz = np.concatenate((frozen, xyz))
        else:
            pose = _local(true_pose)
        assert session.record_keyframe(
            image,
            ordinal=ordinal,
            stamp=1.0 + 0.1 * ordinal,
            source_epoch=0,
            pose=pose,
            K=K,
            image_size=(320, 240),
            ids=ids,
            xy=xy,
            xyz=xyz,
            landmarks=landmarks,
            submap_id=0 if ordinal == 0 else 1,
        )
        session.flush_writer()
    return session, true_centers, truth, hosted, frozen


def _anchors(truth, hosted, count=ANCHOR_MIN_COUNT, first=10, step=2):
    """Frames after the outage: GlueMap pose from the old map, submap pose from submap points only."""
    keys = np.array([key for key, host in hosted.items() if host >= 3], dtype=np.int64)
    pairs = []
    for index in range(count):
        true_pose = _pose(STEP * (4.0 + 0.1 * index))
        pairs.append(
            AnchorPair(
                first + step * index,
                true_pose,
                _local(true_pose),
                keys,
                _project(np.stack([truth[key] for key in keys]), true_pose),
            )
        )
    return pairs


def _merge(session, anchors, anchor_min_old=6):
    sim, _ = fit_anchor_similarity(anchors)
    snapshot = session.merge_snapshot(
        1, sim, anchors, source_epoch=0, K=K, anchor_min_old=anchor_min_old
    )
    if snapshot is None:
        return None, None
    candidate = solve_stitch_snapshot(snapshot)
    return snapshot, session.apply_loop_candidate(snapshot, candidate)


def test_eight_consistent_anchors_fit_the_submap_similarity():
    session_less = _anchors(
        {-1: np.array([0.0, 0.0, 4.0])}, {-1: 4}, count=ANCHOR_MIN_COUNT
    )
    sim, residual = fit_anchor_similarity(session_less)
    assert anchor_consistency(session_less, sim) is None and residual < 1e-9
    assert sim.scale == pytest.approx(TRUE.scale, rel=1e-6)
    np.testing.assert_allclose(sim.rotation, TRUE.rotation, atol=1e-9)
    np.testing.assert_allclose(sim.translation, TRUE.translation, atol=1e-9)


def test_merge_pulls_the_detached_submap_onto_glue_map_and_keeps_glue_points(tmp_path):
    session, true_centers, truth, hosted, _frozen = _submap(tmp_path)
    try:
        frames = sorted(session.keyframes.values(), key=lambda frame: frame["ordinal"])
        glue_pose = np.array(frames[0]["pose"])
        revision = session.revision
        snapshot, closed = _merge(session, _anchors(truth, hosted))
        assert closed is not None, session.last_loop_rejection_reason
        assert snapshot["submap_id"] == 1 and snapshot["scale"] == pytest.approx(TRUE.scale)
        assert session.revision == revision + 1
        np.testing.assert_array_equal(frames[0]["pose"], glue_pose)
        assert not any(key > 0 for key in closed["ids"])
        errors = [
            np.linalg.norm(-f["pose"][:, :3].T @ f["pose"][:, 3] - true_centers[f["ordinal"]])
            for f in frames[1:]
        ]
        assert max(errors) < 0.03
        point_errors = [
            np.linalg.norm(session.landmarks[key]["xyz"] - truth[key])
            for key in truth
            if key in session.landmarks
        ]
        assert np.median(point_errors) < 0.05
    finally:
        session.close()


def test_merge_without_a_pinned_glue_keyframe_is_pinned_by_the_anchors_alone(tmp_path):
    session, _centers, truth, hosted, _frozen = _submap(tmp_path)
    try:
        snapshot, closed = _merge(session, _anchors(truth, hosted), anchor_min_old=99)
        assert closed is not None, session.last_loop_rejection_reason
        assert all(f.get("loop") for f in snapshot["keyframes"] if f["fixed"])
    finally:
        session.close()


def test_a_submap_far_from_any_plausible_drift_is_rejected(tmp_path):
    session, _centers, truth, hosted, _frozen = _submap(tmp_path)
    try:
        anchors = _anchors(truth, hosted)
        far = Sim3(1.0, np.eye(3), np.array([0.0, 3.0, 0.0]))
        revision = session.revision
        assert (
            session.merge_snapshot(1, far, anchors, source_epoch=0, K=K, anchor_min_old=6) is None
        )
        assert session.last_loop_rejection_reason == "implausible_drift"
        assert session.revision == revision
        assert (
            session.merge_snapshot(7, TRUE, anchors, source_epoch=0, K=K, anchor_min_old=6) is None
        )
        assert session.last_loop_rejection_reason == "empty_submap"
    finally:
        session.close()


def test_stitch_recovers_rotated_initial_cameras_without_moving_frozen_points(tmp_path):
    session, centers, truth, hosted, _frozen = _submap(tmp_path)
    try:
        anchors = _anchors(truth, hosted)
        sim, _ = fit_anchor_similarity(anchors)
        snapshot = session.merge_snapshot(1, sim, anchors, source_epoch=0, K=K, anchor_min_old=6)
        turn = cv2.Rodrigues(np.array([0.025, -0.02, 0.015]))[0]
        for frame in snapshot["keyframes"]:
            if not frame["fixed"]:
                pose = np.asarray(frame["pose"])
                frame["pose"] = (turn @ pose).tolist()
        candidate = solve_stitch_snapshot(snapshot)
        assert candidate["ok"], candidate.get("reason")
        assert candidate["loop_after"]["p50"] < 0.1
        closed = session.apply_loop_candidate(snapshot, candidate)
        assert closed is not None, session.last_loop_rejection_reason
        for name, pose in closed["poses"].items():
            pose = np.asarray(pose)
            center = -pose[:, :3].T @ pose[:, 3]
            assert np.linalg.norm(center - centers[session.keyframes[name]["ordinal"]]) < 0.03
            assert np.linalg.norm(cv2.Rodrigues(pose[:, :3])[0]) < 0.01
    finally:
        session.close()


def test_stale_pose_revision_rejects_the_solution_without_committing(tmp_path):
    session, _centers, truth, hosted, _frozen = _submap(tmp_path)
    try:
        anchors = _anchors(truth, hosted)
        sim, _ = fit_anchor_similarity(anchors)
        snapshot = session.merge_snapshot(1, sim, anchors, source_epoch=0, K=K, anchor_min_old=6)
        candidate = solve_stitch_snapshot(snapshot)
        assert candidate["ok"]
        session.keyframes[snapshot["chain"][1]]["pose_revision"] += 1
        revision = session.revision
        assert session.apply_loop_candidate(snapshot, candidate) is None
        assert session.last_loop_rejection_reason == "keyframe_pose_revision"
        assert session.revision == revision
    finally:
        session.close()


def test_long_outage_samples_a_time_spanning_bounded_stitch_chain():
    session = object.__new__(SessionWorkMap)
    frames = {}
    for ordinal in range(LOOP_MAX_CHAIN + 40):
        frames[str(ordinal)] = {
            "name": str(ordinal), "source_epoch": 7, "ordinal": ordinal, "submap_id": 0 if ordinal == 0 else 1,
            "ba_ids": np.arange(101, 109) if ordinal == 0 else np.array([-1]),
            "pose": _pose([0.1 * ordinal, 0, 0]),
        }
    session.keyframes = frames
    parent, chain, path, drift = session._merge_chain_locked(1, Sim3.identity(), 7, 8)
    assert parent["ordinal"] == 0 and len(chain) == LOOP_MAX_CHAIN
    assert chain[0]["ordinal"] == 1 and chain[-1]["ordinal"] == LOOP_MAX_CHAIN + 39
    assert path == pytest.approx(0.1 * (LOOP_MAX_CHAIN + 38)) and drift == pytest.approx(0.0)


# ---------------------------------------------------------------- tracker-side flow


class _Session:
    def __init__(self):
        self.events = []
        self.snapshot_calls = []
        self.writer_healthy = True
        self.geometry_commit_ready = True
        self.last_loop_rejection_reason = None
        self.records = ()
        self.commit = None

    def note_event(self, event):
        self.events.append(event)

    def set_submap_records(self, records):
        self.records = tuple(records)

    def merge_snapshot(self, submap_id, sim, anchors, **kwargs):
        self.snapshot_calls.append((submap_id, sim, len(anchors), kwargs))
        return {
            "source_epoch": 0, "submap_id": submap_id, "scale": float(sim.scale),
            "angle_deg": sim.rotation_deg(), "drift_u": 0.3, "loop_ordinal": anchors[-1].ordinal,
            "keyframes": [
                {"name": "a", "ordinal": 4, "pose0": _pose([0.0, 0.0, 0.0]).tolist(), "pose": []},
                {"name": "__anchor_9__", "loop": True},
            ],
        }

    def apply_loop_candidate(self, snapshot, candidate):
        return self.commit


def _tracker(session):
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker._closing = False
    tracker.profile = SimpleNamespace(
        reloc=SimpleNamespace(frozen_pnp_thresholds={"strong_inliers": 80}),
    )
    tracker.workmap_eval_stage = "full"
    tracker.mapping_enabled = True
    tracker.workmap_session_id = "merge-session"
    tracker._merge_context, tracker._merge_ready = {}, {}
    tracker._graph_edges, tracker._graph_future, tracker._graph_payload = [], None, None
    tracker._graph_pending, tracker._graph_error = False, None
    tracker._graph_solves = tracker._graph_applied = 0
    tracker._graph_solver = "batch"
    tracker.graph_payloads = []
    tracker._unmerged_dirty = False
    tracker._ordinal = 9
    tracker._source_epoch = 0
    tracker._K = K
    tracker._query = None
    tracker._live_ids = -np.arange(1, 41, dtype=np.int64)
    tracker._live_xy = np.zeros((40, 2))
    tracker._live_xyz = np.zeros((40, 3))
    tracker._vo_seed_meta = {3: (8,)}
    tracker._vo_pose_by_bid = {3: _pose([0.0, 0.0, 0.0])}
    tracker._speed_history = [1.0]
    tracker._graph_generation = 0
    tracker._workmap_merge_submitted = tracker._workmap_merge_committed = 0
    tracker._workmap_merge_rejected = tracker._workmap_merge_noop = 0
    tracker._workmap_merge_scale = None
    tracker._stitch_future = tracker._stitch_snapshot = tracker._stitch_started_mono = None
    tracker._stitch_sim = None
    tracker._ba_error = None
    tracker._ba_future = None
    tracker._ba_pending = False
    tracker._ba_pending_mono = None
    tracker._trajectory_backfill = None
    tracker._backfill_repeats = 0
    tracker._backfill_next_ordinal = 0
    tracker.session_workmap = session
    tracker._submaps = SubmapRegistry()
    tracker._submap_archive = []
    tracker._reloc_started, tracker._reloc_first_anchor = {}, {}
    tracker._merge_failures, tracker._merge_residual = {}, {}
    tracker.refreshed = None
    tracker._refresh_live_points = lambda ids: setattr(tracker, "refreshed", ids)
    tracker._ensure_ba_process = lambda: None
    tracker.submitted = []

    def submit(fn, payload):
        future = Future()
        if fn is solve_submap_graph:
            tracker.graph_payloads.append(payload)
            future.set_result(fn(payload))
        else:
            tracker.submitted.append(payload)
            future.set_result({"ok": True})
        return future

    tracker._ba_executor = SimpleNamespace(submit=submit)
    submap_id = tracker._submaps.open("detached", 0, 1)
    tracker._submaps.set_state(submap_id, "relocalizing")
    return tracker, submap_id


def _pairs():
    return _anchors({-1: np.array([0.0, 0.0, 4.0])}, {-1: 4})


def test_seven_anchors_are_never_sent_and_the_eighth_submits_a_stitch_the_poll_commits():
    session = _Session()
    tracker, submap_id = _tracker(session)
    stitched = _pose([0.0, 0.3, 0.0])
    session.commit = {"ids": {-1, -2}, "poses": {"a": stitched}}
    pairs = _pairs()
    for pair in pairs[:-1]:
        tracker._submaps.add_anchor(submap_id, pair)
    tracker._try_merge_submap(submap_id)
    assert tracker.submitted == [] and session.snapshot_calls == []

    tracker._submaps.add_anchor(submap_id, pairs[-1])     # the eighth consistent pair arrives
    tracker._try_merge_submap(submap_id)
    # The merge edge goes to the submap graph first; the stitch waits for its solution.
    assert tracker._submaps.get(submap_id)["state"] == "merging"
    assert tracker.submitted == [] and len(tracker.graph_payloads) == 1
    assert [edge["kind"] for edge in tracker.graph_payloads[0]["edges"]] == ["merge"]
    tracker._poll_graph()
    assert tracker._graph_solves == 1
    tracker._advance_pending_merges()
    assert len(tracker.submitted) == 1 and tracker._workmap_merge_submitted == 1
    assert session.snapshot_calls[0][3]["anchor_min_old"] == 80
    solved = session.snapshot_calls[0][1]                  # the graph solved the merge edge exactly
    assert solved.scale == pytest.approx(TRUE.scale, rel=1e-4)
    # Another pass while the stitch is in flight does not queue a second one.
    tracker._advance_pending_merges()
    assert len(tracker.submitted) == 1

    assert tracker._poll_stitch()
    assert tracker.refreshed == {-1, -2} and tracker._workmap_merge_committed == 1
    assert tracker._workmap_merge_scale == pytest.approx(TRUE.scale)
    assert tracker._submaps.get(submap_id)["state"] == "merged"
    assert tracker._speed_history == [] and tracker._stitch_future is None
    # The pending seed (ordinal 8, after keyframe 4) follows that keyframe's correction.
    center = -tracker._vo_pose_by_bid[3][:, :3].T @ tracker._vo_pose_by_bid[3][:, 3]
    np.testing.assert_allclose(center, [0.0, 0.3, 0.0], atol=1e-12)
    assert any(e["event"] == "submap_merged" and not e["noop"] for e in session.events)


def test_an_inconsistent_anchor_is_dropped_with_its_reason_and_nothing_is_sent():
    session = _Session()
    tracker, submap_id = _tracker(session)
    pairs = _anchors({-1: np.array([0.0, 0.0, 4.0])}, {-1: 4})
    bad = pairs[3]
    turned = cv2.Rodrigues(np.array([[0.0], [0.0], [np.radians(3.0)]]))[0] @ bad.map_pose[:, :3]
    center = -bad.map_pose[:, :3].T @ bad.map_pose[:, 3]
    pairs[3] = AnchorPair(
        bad.ordinal, np.hstack([turned, (-turned @ center).reshape(3, 1)]),
        bad.local_pose, bad.ids, bad.xy,
    )
    for pair in pairs:
        tracker._submaps.add_anchor(submap_id, pair)
    tracker._try_merge_submap(submap_id)
    assert tracker.submitted == [] and session.snapshot_calls == []
    assert len(tracker._submaps.anchors(submap_id)) == ANCHOR_MIN_COUNT - 1
    rejected = [e for e in session.events if e["event"] == "submap_anchor_rejected"]
    assert rejected and rejected[0]["reason"] == "rotation_residual"


def test_three_rejected_stitches_in_a_row_give_the_submap_up_and_drop_its_points():
    session = _Session()
    tracker, submap_id = _tracker(session)
    tracker._submaps.tag_landmarks([-1, -2, -3], submap_id)
    for attempt in range(3):
        tracker._submaps.set_state(submap_id, "merging")
        tracker._merge_rejected(submap_id, "loop_reprojection")
        expected = "unmerged" if attempt == 2 else "relocalizing"
        assert tracker._submaps.get(submap_id)["state"] == expected
    assert tracker._workmap_merge_rejected == 3
    assert -1 in tracker._live_ids                  # never removed in the middle of a pose solve
    tracker._purge_unmerged_rows()
    assert -1 not in tracker._live_ids and -4 in tracker._live_ids
    assert tracker._submaps.is_unmerged(-2) and not tracker._submaps.is_unmerged(-4)
    assert tracker._submaps.relocalizing_id is None


def test_uncommitted_submap_rows_stay_isolated_through_stitch_and_retry():
    session = _Session()
    tracker, submap_id = _tracker(session)
    tracker._live_ids = np.array([5, 6, -1, -2, -3], dtype=np.int64)
    tracker._submaps.tag_landmarks([-2, -3], submap_id)
    for state in ("relocalizing", "merging", "relocalizing"):
        tracker._submaps.set_state(submap_id, state)
        np.testing.assert_array_equal(tracker._mixed_pnp_rows(), [0, 1, 2])
        np.testing.assert_array_equal(
            tracker._retained_rows(np.zeros(5, dtype=bool)), [False, False, False, True, True]
        )
    tracker._submaps.set_state(submap_id, "merged")
    assert tracker._mixed_pnp_rows() is None


def test_bundle_adjustment_waits_while_a_stitch_owns_the_keyframes():
    class Session:
        def ba_snapshot(self, *args, **kwargs):
            raise AssertionError("BA must not snapshot during a stitch")

    tracker, _ = _tracker(Session())
    tracker._ba_optimizer = "local_ba"
    tracker._stitch_future = object()
    tracker._submit_ba_if_ready()
    assert tracker._ba_pending and tracker._ba_pending_mono is not None


@pytest.mark.parametrize(
    ("before_p50", "after_p50", "committed"),
    [
        (2.6, 1.55, False),    # improvement cannot bypass the absolute quality gate
        (1.9, 1.5, True),      # absolute boundary is inclusive
        (1.2, 1.4, True),      # below the absolute bound
    ],
)
def test_loop_frame_gate_keeps_the_absolute_quality_bound(
    tmp_path, before_p50, after_p50, committed
):
    session, _centers, truth, hosted, _frozen = _submap(tmp_path)
    try:
        anchors = _anchors(truth, hosted)
        sim, _ = fit_anchor_similarity(anchors)
        snapshot = session.merge_snapshot(1, sim, anchors, source_epoch=0, K=K, anchor_min_old=6)
        candidate = solve_stitch_snapshot(snapshot)
        assert candidate["ok"]
        candidate["loop_before"] = {**candidate["loop_before"], "p50": before_p50}
        candidate["loop_after"] = {**candidate["loop_after"], "p50": after_p50, "inlier_fraction": 0.9}
        closed = session.apply_loop_candidate(snapshot, candidate)
        assert (closed is not None) is committed
        if not committed:
            assert session.last_loop_rejection_reason == "loop_frame_not_stitched"
    finally:
        session.close()


def test_a_rejected_stitch_starts_a_fresh_anchor_window_for_the_retry():
    session = _Session()
    tracker, submap_id = _tracker(session)
    for pair in _pairs():
        tracker._submaps.add_anchor(submap_id, pair)
    tracker._reloc_first_anchor[submap_id] = 100
    tracker._ordinal = 400
    tracker._submaps.set_state(submap_id, "merging")
    tracker._merge_rejected(submap_id, "loop_reprojection")
    assert tracker._submaps.get(submap_id)["state"] == "relocalizing"
    assert tracker._submaps.anchors(submap_id) == []
    tracker._expire_relocalization()          # 300 frames after the first anchor, 0 after the reject
    assert tracker._submaps.get(submap_id)["state"] == "relocalizing"
    tracker._ordinal = 400 + ANCHOR_MAX_SPAN + 1
    tracker._expire_relocalization()
    assert tracker._submaps.get(submap_id)["state"] == "unmerged"


def _carry_scene(tmp_path):
    session = SessionWorkMap(
        tmp_path / "carry-session",
        session_id="carry-session",
        base_release_id="immutable-r1",
        coordinate_frame_id="fixed-global-frame",
    )
    # Neither numeric ID order nor observation row order matches archive order.
    points = {
        key: {"xyz": np.array([index * 0.1, 0.0, 4.0]), "observations": ()}
        for index, key in enumerate((-3, -1, -8, -5, -7))
    }
    session.archive_landmarks(points)
    names = []
    for ordinal, seen in ((20, [-1, -3, -8, -5, -7, -999]), (10, [-8]), (30, [-5])):
        xyz = np.stack([points.get(key, {"xyz": [0.0, 0.0, 4.0]})["xyz"] for key in seen])
        pose = _pose([0.0, 0.0, 0.0])
        names.append(session.record_keyframe(
            np.zeros((8, 8), dtype=np.uint8),
            ordinal=ordinal,
            stamp=float(ordinal),
            source_epoch=0,
            pose=pose,
            K=K,
            image_size=(8, 8),
            ids=np.asarray(seen, dtype=np.int64),
            xy=_project(xyz, pose),
            xyz=xyz,
            landmarks=points,
        ))
        session.flush_writer()
    session._frames_by_landmark.pop(-7)
    return session, points, names


class _TraversalCountedLandmarks(dict):
    def __init__(self, source):
        super().__init__(source)
        self.traversed = 0

    def __iter__(self):
        for key in super().__iter__():
            self.traversed += 1
            yield key

    def items(self):
        for item in super().items():
            self.traversed += 1
            yield item

    def values(self):
        for value in super().values():
            self.traversed += 1
            yield value


@pytest.mark.parametrize("unrelated", [1, 1000])
def test_carry_only_visits_chain_candidates_and_keeps_update_order(tmp_path, unrelated):
    session, points, names = _carry_scene(tmp_path)
    try:
        # Replacing an archived record must not move its insertion rank.
        session.archive_landmarks({-3: points[-3]})
        session.archive_landmarks({
            -1000 - index: {"xyz": np.array([0.0, 0.0, 4.0]), "observations": ()}
            for index in range(unrelated)
        })
        landmarks = _TraversalCountedLandmarks(session.landmarks)
        session.landmarks = landmarks
        refined = np.array([0.2, 0.1, 4.1])
        updates = {-8: refined.copy()}
        poses = {names[0]: _pose([0.2, 0.0, 0.0]), names[1]: _pose([0.4, 0.0, 0.0])}
        with session._lock:
            session._carry_single_view_points_locked(set(poses), poses, updates)
        assert list(updates) == [-8, -3, -1]
        np.testing.assert_array_equal(updates[-8], refined)
        for key in (-3, -1):
            np.testing.assert_allclose(updates[key], points[key]["xyz"] + [0.2, 0.0, 0.0])
            np.testing.assert_array_equal(session.landmarks[key]["xyz"], points[key]["xyz"])
        # -5 has an outside observer, -7 lacks an index, and -999 has no archive record.
        assert landmarks.traversed <= 5
    finally:
        session.close()


def test_carry_multiple_invalid_candidates_keep_archive_exception_precedence(tmp_path):
    session, _points, names = _carry_scene(tmp_path)
    try:
        # First archived candidate is malformed; the next lacks a host pose.
        session.landmarks[-3]["xyz"] = np.array([0.0, 4.0])
        session._frames_by_landmark[-1] = {names[1]}
        poses = {names[0]: _pose([0.2, 0.0, 0.0])}
        chain = {names[0], names[1]}
        with session._lock:
            with pytest.raises(ValueError):
                session._carry_single_view_points_locked(chain, poses, {})
            # Skip the earlier malformed candidate to expose the independent second error.
            with pytest.raises(KeyError) as rejection:
                session._carry_single_view_points_locked(chain, poses, {-3: np.zeros(3)})
        assert rejection.value.args == (names[1],)
    finally:
        session.close()

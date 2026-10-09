"""Session archive, durability, indexing, and revision gates for P2."""

from __future__ import annotations

import json
import math
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import online_workmap as online_workmap_module
from online_workmap import OnlineWorkMapTracker, SessionWorkMap
from two_rate_tracker import RelocWorker


class _IndexProvider:
    def __init__(self) -> None:
        self.indexed: list[tuple[str, str]] = []
        self.indexed_event = threading.Event()

    def localize_array(self, _gray):
        return SimpleNamespace(ok=False)

    def index_workmap_reference(self, reference):
        image_path = Path(reference["image_path"])
        assert image_path.is_file()
        assert image_path.stat().st_size > 0
        name = f"workmap:{reference['session_id']}:{reference['name']}"
        self.indexed.append((name, threading.current_thread().name))
        self.indexed_event.set()
        return name


class _Resettable:
    def reset(self) -> None:
        pass


def _synthetic_landmarks() -> dict[int, dict]:
    pose0 = np.column_stack((np.eye(3), np.array([-0.1, 0.0, 0.0])))
    pose1 = np.column_stack((np.eye(3), np.zeros(3)))
    landmarks = {}
    for index, point_id in enumerate(range(-1, -13, -1)):
        xyz = np.array([0.03 * index, 0.02 * (index % 4), 3.0 + 0.01 * index])
        first_xy = np.array([8.0 + (index % 4) * 12.0, 8.0 + (index // 4) * 14.0])
        second_xy = first_xy + np.array([2.0, 1.0])
        landmarks[point_id] = {
            "xyz": xyz,
            "revision": 0,
            "parallax_deg": 4.0,
            "valid_observation_count": 2,
            "observations": (
                (0, 1.0, first_xy, pose0),
                (2, 1.2, second_xy, pose1),
            ),
            "last_inlier": (2, 1.2, second_xy),
        }
    return landmarks


def test_durable_negative_points_survive_reset_and_stale_snapshot_is_rejected(tmp_path):
    provider = _IndexProvider()
    worker = RelocWorker(provider)
    worker.start()
    session_id = "workmap-test-session"
    session_dir = tmp_path / "session"
    session = SessionWorkMap(
        session_dir,
        session_id=session_id,
        base_release_id="base-release-r1",
        coordinate_frame_id="frame-r1",
        base_bundle_sha256="a" * 64,
        on_durable_reference=lambda reference, callback: worker.submit_index(reference, callback),
        min_reference_points=8,
    )
    try:
        points = _synthetic_landmarks()
        session.archive_landmarks(points)
        archived = session.tracker_landmarks()
        ids = np.asarray(sorted(archived), dtype=np.int64)
        xy = np.asarray([archived[int(point_id)]["observations"][-1][2] for point_id in ids])
        xyz = np.asarray([archived[int(point_id)]["xyz"] for point_id in ids])
        name = session.record_keyframe(
            np.arange(64 * 64, dtype=np.uint8).reshape(64, 64),
            ordinal=2,
            stamp=1.2,
            source_epoch=0,
            pose=np.column_stack((np.eye(3), np.zeros(3))),
            K=np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]),
            image_size=(64, 64),
            ids=ids,
            xy=xy,
            xyz=xyz,
            landmarks=archived,
        )
        assert name is not None
        assert provider.indexed_event.wait(timeout=5.0)
        session.flush_writer()
        deadline = time.monotonic() + 5.0
        while (
            not session.keyframes.get(name, {}).get("index_ready") and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert session.stats()["workmap_writer_healthy"] is True
        assert session.keyframes[name]["durable"] is True
        assert session.keyframes[name]["index_ready"] is True
        assert worker.workmap_diagnostics()["workmap_index_healthy"] is True
        assert provider.indexed[0][1] == "direct-reloc-worker"

        reference = session.keyframes[name]
        provider_name = reference["provider_reference_name"]
        snapshot = {
            "session_id": session_id,
            "revision": reference["revision"],
            "keyframe_revision": reference["pose_revision"],
            "keyframe_revisions": {provider_name: reference["pose_revision"]},
            "reference_name": provider_name,
            "landmark_revisions": reference["landmark_revisions"],
        }

        # Tracker reset drops active state but retains archive identity and the
        # next negative ID, so returning to this region cannot recycle a point.
        tracker = object.__new__(OnlineWorkMapTracker)
        tracker.session_workmap = session
        tracker._source_epoch = 0
        tracker._tracker = _Resettable()
        tracker._worker = _Resettable()
        tracker._workmap_landmarks = session.tracker_landmarks()
        tracker._vo_next_id = -13
        tracker._workmap_initial_anchor = True
        tracker._depth_ready = {}
        tracker.reset()
        assert set(tracker._workmap_landmarks) == set(range(-1, -13, -1))
        assert tracker._vo_next_id == -13
        assert tracker._workmap_initial_anchor is True
        assert tracker._source_epoch == 1
        assert session.validate_snapshot(snapshot, (provider_name,))
        session.landmarks[-1]["revision"] += 1
        assert not session.validate_snapshot(snapshot, (provider_name,))
    finally:
        worker.close()
        session.close()

    snapshot_file = json.loads((session_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot_file["session_id"] == session_id
    assert snapshot_file["base_release_id"] == "base-release-r1"
    assert snapshot_file["base_bundle_sha256"] == "a" * 64
    session_meta = json.loads((session_dir / "session_meta.json").read_text(encoding="utf-8"))
    assert session_meta["base_bundle_sha256"] == "a" * 64
    assert set(map(int, snapshot_file["landmarks"])) == set(range(-1, -13, -1))
    assert (session_dir / "observations.jsonl").is_file()
    assert (session_dir / "events.jsonl").is_file()


def test_first_view_image_is_durable_before_landmarks_mature(tmp_path):
    directory = tmp_path / "session"
    session = SessionWorkMap(
        directory,
        session_id="seed-image-session",
        base_release_id="base-release-r1",
        coordinate_frame_id="frame-r1",
        min_reference_points=8,
    )
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])
    gray = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    pose = np.column_stack((np.eye(3), np.array([-0.1, 0.0, 0.0])))
    try:
        seed_name = session.record_keyframe(
            gray,
            ordinal=0,
            stamp=1.0,
            source_epoch=0,
            pose=pose,
            K=K,
            image_size=(64, 64),
            ids=np.zeros(0, dtype=np.int64),
            xy=np.zeros((0, 2)),
            xyz=np.zeros((0, 3)),
            landmarks={},
        )
        points = _synthetic_landmarks()
        session.archive_landmarks(points)
        ids = np.array(sorted(points), dtype=np.int64)
        session.record_keyframe(
            gray,
            ordinal=2,
            stamp=1.2,
            source_epoch=0,
            pose=np.column_stack((np.eye(3), np.zeros(3))),
            K=K,
            image_size=(64, 64),
            ids=ids,
            xy=np.array([points[int(key)]["observations"][1][2] for key in ids]),
            xyz=np.array([points[int(key)]["xyz"] for key in ids]),
            landmarks=points,
        )
    finally:
        session.close()

    frames = [json.loads(row) for row in (directory / "keyframes.jsonl").read_text().splitlines()]
    observations = [
        json.loads(row) for row in (directory / "observations.jsonl").read_text().splitlines()
    ]
    seed = next(frame for frame in frames if frame["name"] == seed_name)
    assert seed["ordinal"] == 0 and seed["ids"] == []
    assert Path(seed["image_path"]).is_file()
    assert any(
        row.get("landmark_id") == -1
        and row.get("view_index") == 0
        and row.get("ordinal") == seed["ordinal"]
        for row in observations
    )
    seed_snapshot = json.loads((directory / "snapshot.json").read_text())["keyframes"][seed_name]
    assert set(seed_snapshot["ids"]) == set(range(-1, -13, -1))
    row = seed_snapshot["ba_ids"].index(-1)
    assert np.array_equal(seed_snapshot["ba_xy"][row], points[-1]["observations"][0][2])


def test_saturated_index_dispatcher_does_not_block_writer_and_drains_catalog(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    indexed_names = []
    index_threads = []

    def delayed_index(reference, callback):
        indexed_names.append(reference["name"])
        index_threads.append(threading.current_thread().name)
        if len(indexed_names) == 1:
            entered.set()
            assert release.wait(timeout=30.0)
        callback(reference, f"workmap:{reference['name']}", None)
        return True

    directory = tmp_path / "session"
    session = SessionWorkMap(
        directory,
        session_id="bounded-index-handoff",
        base_release_id="base-release-r1",
        coordinate_frame_id="frame-r1",
        on_durable_reference=delayed_index,
        min_reference_points=8,
    )
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])
    gray = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    pose = np.column_stack((np.eye(3), np.zeros(3)))
    names = []
    try:
        for index in range(6):
            ordinal = 2 + index * 2
            stamp = 1.2 + index * 0.1
            source = _synthetic_landmarks()
            points = {}
            for point_id, landmark in source.items():
                first, second = landmark["observations"]
                key = -(1000 * (index + 1) + abs(point_id))
                points[key] = {
                    **landmark,
                    "xyz": landmark["xyz"].copy(),
                    "observations": (
                        (-(100 + index), stamp - 0.2, first[2], first[3]),
                        (ordinal, stamp, second[2], pose),
                    ),
                    "last_inlier": (ordinal, stamp, second[2]),
                }
            session.archive_landmarks(points)
            ids = np.asarray(sorted(points), dtype=np.int64)
            name = session.record_keyframe(
                gray,
                ordinal=ordinal,
                stamp=stamp,
                source_epoch=0,
                pose=pose,
                K=K,
                image_size=(64, 64),
                ids=ids,
                xy=np.array([points[int(key)]["observations"][-1][2] for key in ids]),
                xyz=np.array([points[int(key)]["xyz"] for key in ids]),
                landmarks=points,
            )
            assert name is not None
            names.append(name)
            if index == 0:
                assert entered.wait(timeout=5.0)
            deadline = time.monotonic() + 5.0
            while not session.keyframes[name]["durable"] and time.monotonic() < deadline:
                time.sleep(0.005)
            assert session.keyframes[name]["durable"]

        assert all(session.keyframes[name]["durable"] for name in names)
        assert all((session.images_dir / f"{name}.png").is_file() for name in names)
        assert session.writer_healthy
        assert session.stats()["workmap_writer_overflow_count"] == 0
        assert session._queue.maxsize == 4
        assert session.index_handoff_pending
        assert session.index_handoff_oldest_age_ms() > 0.0
        assert indexed_names == names[:1]
        assert index_threads == ["direct-workmap-index-dispatch"]

        release.set()
        session.flush_writer()
        session.flush_index_dispatcher()
        assert indexed_names == names
        assert index_threads == ["direct-workmap-index-dispatch"] * len(names)
        assert all(session.keyframes[name]["index_ready"] for name in names)
    finally:
        release.set()
        session.close()


_REAL_REQUIRE_FREE_SPACE = online_workmap_module._require_free_space


def test_keyframe_free_space_floor_stops_before_the_flight_disk_block(monkeypatch, tmp_path):
    usage = shutil._ntuple_diskusage
    gib = 1024**3
    for free_gib, allowed in ((20.0, True), (9.5, True), (8.5, False), (7.9, False)):
        monkeypatch.setattr(
            online_workmap_module.shutil,
            "disk_usage",
            lambda _path, free=free_gib: usage(140 * gib, 0, int(free * gib)),
        )
        if allowed:
            _REAL_REQUIRE_FREE_SPACE(tmp_path)
        else:
            with pytest.raises(OSError, match="workmap keyframes stopped"):
                _REAL_REQUIRE_FREE_SPACE(tmp_path)


def test_low_disk_fails_the_writer_closed_instead_of_filling_the_disk(monkeypatch, tmp_path):
    def full(_path):
        raise OSError("workmap keyframes stopped: 4.00 GiB / 3.0% free")

    monkeypatch.setattr(online_workmap_module, "_require_free_space", full)
    session = online_workmap_module.SessionWorkMap(
        tmp_path / "session",
        session_id="disk-floor",
        base_release_id="release",
        coordinate_frame_id="frame",
    )
    try:
        name = session.record_keyframe(
            np.zeros((8, 8), dtype=np.uint8),
            ordinal=1,
            stamp=1.0,
            source_epoch=0,
            pose=np.hstack([np.eye(3), np.zeros((3, 1))]),
            K=np.eye(3),
            image_size=(8, 8),
            ids=np.zeros((0,), dtype=np.int64),
            xy=np.zeros((0, 2)),
            xyz=np.zeros((0, 3)),
            landmarks={},
        )
        assert name is not None
        assert session.flush_writer(timeout=5.0)
        assert not session.writer_healthy
        assert not list((tmp_path / "session" / "keyframes").glob("*.png"))
    finally:
        session.close()


def test_signalled_close_writes_a_partial_snapshot_within_the_deadline(tmp_path, monkeypatch):
    release = threading.Event()
    real_write = SessionWorkMap._write_task

    def slow_write(self, task):
        release.wait(timeout=10.0)  # a stalled disk
        return real_write(self, task)

    monkeypatch.setattr(SessionWorkMap, "_write_task", slow_write)
    session = SessionWorkMap(
        tmp_path / "session",
        session_id="fast-close",
        base_release_id="release",
        coordinate_frame_id="frame",
    )
    for ordinal in (1, 2, 3):
        session.record_keyframe(
            np.zeros((8, 8), dtype=np.uint8),
            ordinal=ordinal,
            stamp=float(ordinal),
            source_epoch=0,
            pose=np.hstack([np.eye(3), np.zeros((3, 1))]),
            K=np.eye(3),
            image_size=(8, 8),
            ids=np.zeros((0,), dtype=np.int64),
            xy=np.zeros((0, 2)),
            xyz=np.zeros((0, 3)),
            landmarks={},
        )
    session.request_abort()
    started = time.monotonic()
    session.close(shutdown_timeout_s=0.5)
    assert time.monotonic() - started < 1.5
    release.set()
    snapshot = json.loads((tmp_path / "session" / "snapshot.json").read_text())
    assert snapshot["statistics"]["drain_complete"] is False
    meta = json.loads((tmp_path / "session" / "session_meta.json").read_text())
    assert meta["drain_complete"] is False


def test_solver_process_exits_when_its_worker_is_killed(tmp_path):
    import os
    import signal
    import subprocess
    import sys

    script = tmp_path / "parent.py"
    script.write_text(
        "import multiprocessing, os, sys, time\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "import online_workmap as workmap\n"
        "if __name__ == '__main__':\n"
        "    pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn'),\n"
        "        initializer=workmap._ba_process_initialize, initargs=(os.getpid(),))\n"
        "    print(pool.submit(os.getpid).result(), flush=True)\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    parent = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True, env=env
    )
    try:
        child = int(parent.stdout.readline())
        parent.send_signal(signal.SIGKILL)  # no executor shutdown can run
        parent.wait(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(child, signal.SIGKILL)
            raise AssertionError("orphaned workmap solver survived its worker")
    finally:
        if parent.poll() is None:
            parent.kill()


def test_copy_landmark_single_copy_preserves_detached_geometry_and_unknown_fields():
    from online_workmap import _copy_landmark

    xyz = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    xy1 = np.array([10.0, 20.0], dtype=np.float64)
    pose1 = np.eye(4)[:3]
    xy2 = np.array([30.0, 40.0], dtype=np.float64)
    pose2 = np.eye(4)[:3]
    inlier_xy = np.array([50.0, 60.0], dtype=np.float64)
    custom_arr = np.array([9.0, 8.0, 7.0])

    landmark = {
        "xyz": xyz,
        "revision": 3,
        "observations": [
            (0, 1.0, xy1, pose1),
            (1, 2.0, xy2, pose2),
        ],
        "last_inlier": (1, 2.0, inlier_xy),
        "custom_str": "extra_data",
        "custom_arr": custom_arr,
        "custom_dict": {"alpha": [1, 2]},
        "custom_list": [1, [2, 3]],
    }
    copied = _copy_landmark(landmark)

    # Detached checks: modifying copied must not alter landmark
    copied["xyz"][0] = 999.0
    assert landmark["xyz"][0] == 1.0

    copied["observations"][0][2][0] = 888.0
    assert landmark["observations"][0][2][0] == 10.0

    copied["observations"][0][3][0, 0] = 777.0
    assert landmark["observations"][0][3][0, 0] == 1.0

    copied["last_inlier"][2][0] = 666.0
    assert landmark["last_inlier"][2][0] == 50.0

    copied["custom_arr"][0] = 555.0
    assert landmark["custom_arr"][0] == 9.0

    copied["custom_dict"]["alpha"].append(3)
    assert landmark["custom_dict"]["alpha"] == [1, 2]

    copied["custom_list"][1].append(4)
    assert landmark["custom_list"][1] == [2, 3]
    # Content checks
    assert copied["revision"] == 3
    assert copied["custom_str"] == "extra_data"
    assert len(copied["observations"]) == 2
    assert copied["observations"][0][0] == 0
    assert len(copied["observations"][0]) == 4


def test_select_keyframe_points_with_pending_ids_and_active_cap_overflow():
    from online_workmap import _select_keyframe_points

    # Landmark -1: active PnP track, 2 observations, second is ordinal 5
    # Landmark -2: active cap overflow (not in ids!), 2 observations, second is ordinal 5
    # Landmark -3: seed point, only 1 observation (ordinal 5) -> must be excluded
    # Landmark -4: older landmark, 2 observations, second was ordinal 4 -> must be excluded
    landmarks = {
        -1: {
            "xyz": np.array([1.0, 1.0, 5.0]),
            "observations": [
                (4, 1.0, np.array([10.0, 10.0]), np.eye(4)[:3]),
                (5, 2.0, np.array([15.0, 15.0]), np.eye(4)[:3]),
            ],
        },
        -2: {
            "xyz": np.array([2.0, 2.0, 6.0]),
            "observations": [
                (3, 0.5, np.array([20.0, 20.0]), np.eye(4)[:3]),
                (5, 2.0, np.array([25.0, 25.0]), np.eye(4)[:3]),
            ],
        },
        -3: {
            "xyz": np.array([3.0, 3.0, 7.0]),
            "observations": [
                (5, 2.0, np.array([30.0, 30.0]), np.eye(4)[:3]),
            ],
        },
        -4: {
            "xyz": np.array([4.0, 4.0, 8.0]),
            "observations": [
                (2, 0.2, np.array([40.0, 40.0]), np.eye(4)[:3]),
                (4, 1.0, np.array([45.0, 45.0]), np.eye(4)[:3]),
            ],
        },
    }

    ids = np.array([-1], dtype=np.int64)
    xy = np.array([[15.0, 15.0]])
    xyz = np.array([[1.0, 1.0, 5.0]])

    candidate_ids = {-2}

    sel_ids, sel_xy, sel_xyz = _select_keyframe_points(
        ids, xy, xyz, landmarks, ordinal=5, candidate_ids=candidate_ids
    )

    assert set(sel_ids) == {-2, -1}
    assert -3 not in sel_ids
    assert -4 not in sel_ids

    empty_cand_ids, empty_cand_xy, empty_cand_xyz = _select_keyframe_points(
        ids, xy, xyz, landmarks, ordinal=5, candidate_ids=()
    )
    assert set(empty_cand_ids) == {-1}
    assert -2 not in empty_cand_ids


def test_newer_overlap_reference_supersedes_waiting_one_without_failing_health():
    tracker = object.__new__(OnlineWorkMapTracker)
    submitted = []
    tracker._worker = SimpleNamespace(
        submit_overlap=lambda ref, callback: submitted.append(ref["name"]) and False
    )
    tracker._on_overlap_match = lambda *_: None
    tracker._overlap_retry_lock = threading.Lock()
    tracker._overlap_attempted = {}
    tracker._overlap_retry_ref = None
    tracker._overlap_retry_started_mono = None
    tracker._overlap_last_match_submitted_mono = None
    tracker._overlap_error = None
    tracker._overlap_superseded = 0
    older, newer = {"name": "kf-1", "ids": [1] * 20}, {"name": "kf-2", "ids": [1] * 12}
    tracker._submit_or_retry_overlap(older)
    tracker._submit_or_retry_overlap(newer)
    assert submitted == ["kf-1", "kf-2"]
    assert tracker._overlap_retry_ref is newer
    assert tracker._overlap_superseded == 1
    assert tracker._overlap_error is None


def test_display_queue_holds_each_new_or_moved_landmark_once_in_order(tmp_path):
    session = SessionWorkMap(
        tmp_path / "session", session_id="display", base_release_id="r1",
        coordinate_frame_id="frame", min_reference_points=6,
    )
    try:
        pose = np.column_stack((np.eye(3), np.zeros(3)))

        def landmark(x):
            obs = ((0, 1.0, np.zeros(2), pose), (1, 1.1, np.zeros(2), pose))
            return {
                "xyz": np.array([x, 0.0, 4.0]), "observations": obs,
                "last_inlier": (1, 1.1, np.zeros(2)), "rgb": [10, int(x * 10), 30],
            }

        session.archive_landmarks({-1: landmark(1.0), -2: landmark(2.0)})
        session.archive_landmarks({-2: landmark(2.5), -3: landmark(3.0)})  # -2 moved, -1 unchanged
        ids, xyz, rgb, total = session.pop_display_points(2)
        assert ids == [-1, -2] and total == 3 and xyz[1] == [2.5, 0.0, 4.0]
        assert rgb == [[10, 10, 30], [10, 25, 30]]
        assert session.pop_display_points(10)[0] == [-3]
        assert session.pop_display_points(10)[0] == []
    finally:
        session.close()


def test_abandon_maintenance_keeps_only_running_index_callback():
    entered, release = threading.Event(), threading.Event()
    callbacks = []

    class Provider:
        def index_workmap_reference(self, reference):
            entered.set()
            assert release.wait(2.0)
            return reference["name"]

    worker = RelocWorker(Provider())
    worker.start()
    callback = lambda *args: callbacks.append(args)
    try:
        assert worker.submit_index({"name": "running"}, callback)
        assert entered.wait(1.0)
        assert worker.submit_index({"name": "pending"}, callback)
        assert worker.submit_overlap({"name": "overlap"}, callback)
        assert worker.abandon_maintenance() == {"index": 1, "overlap": 1, "depth": 0}
        assert worker.running_kind() == "index"
        assert not worker.submit_index({"name": "late"}, callback)
        assert not worker.submit_overlap({}, callback)
        release.set()
        started = time.monotonic()
        worker.close()
        assert time.monotonic() - started < 1.0
        assert callbacks == [({"name": "running"}, "running", None)]
    finally:
        release.set()
        worker.close()


def test_stop_index_dispatch_preserves_health_and_counts_unfinished_frames(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def submit(reference, callback):
        entered.set()
        assert release.wait(2.0)
        return False

    session = SessionWorkMap(
        tmp_path / "closing", session_id="closing", base_release_id="base",
        coordinate_frame_id="frame", on_durable_reference=submit, min_reference_points=8,
    )
    try:
        points = _synthetic_landmarks()
        ids = np.asarray(sorted(points), dtype=np.int64)
        xy = np.asarray([points[int(key)]["observations"][-1][2] for key in ids])
        xyz = np.asarray([points[int(key)]["xyz"] for key in ids])
        names = []
        for ordinal in (2, 3):
            names.append(session.record_keyframe(
                np.arange(64 * 64, dtype=np.uint8).reshape(64, 64),
                ordinal=ordinal, stamp=1.2 + ordinal * 0.1, source_epoch=0,
                pose=np.column_stack((np.eye(3), np.zeros(3))),
                K=np.array([[50., 0., 32.], [0., 50., 32.], [0., 0., 1.]]),
                image_size=(64, 64), ids=ids, xy=xy, xyz=xyz, landmarks=points,
            ))
        assert all(name is not None for name in names)
        assert session.flush_writer(timeout=1.0)
        assert entered.wait(1.0)
        with session._index_condition:
            second = session.keyframes[names[1]]
            session._request_index_locked(second, second["revision"])
        session.stop_index_dispatch()
        release.set()
        assert session.flush_index_dispatcher(timeout=1.0)
        frame = session.keyframes[names[0]]
        session.mark_indexed({"name": names[0], "revision": frame["revision"]}, "indexed")
        session.set_close_report({"dropped": {}, "phase_ms": {}})
        session.close()
        snapshot = json.loads((session.directory / "snapshot.json").read_text())
        assert snapshot["statistics"]["close_dropped"]["index"] == 1
        assert snapshot["statistics"]["writer_healthy"] is True
        assert snapshot["statistics"]["index_errors"] == 0
    finally:
        release.set()
        session.close()


def test_close_drops_reindex_revision_even_when_previous_index_is_ready(tmp_path):
    session = SessionWorkMap(
        tmp_path / "reindex-close", session_id="reindex", base_release_id="base",
        coordinate_frame_id="frame",
    )
    try:
        # The previous revision remains searchable while a newer geometry revision waits.
        frame = {
            "name": "kf", "ordinal": 1, "revision": 2, "pose_revision": 1,
            "durable": True, "durable_revision": 2, "index_ready": True,
            "index_requested_revision": 2, "index_submitted_revision": 1,
            "index_completed_revision": 1,
        }
        with session._index_condition:
            session.keyframes["kf"] = frame
            session._refresh_index_waiting_locked(frame)
            assert session._has_durable_index_candidate_locked()
        session.stop_index_dispatch()
        with session._index_condition:
            assert not session._has_durable_index_candidate_locked()
            assert frame["index_dropped_at_close"] is True
        session.mark_indexed({"name": "kf", "revision": 2}, "new-revision")
        assert "index_dropped_at_close" not in frame
        assert frame["index_completed_revision"] == 2
    finally:
        session.close()


def test_normal_close_wakes_idle_writer_without_losing_accepted_geometry(tmp_path, monkeypatch):
    idle = threading.Event()
    original_queue = online_workmap_module.queue.Queue

    class LongIdleQueue(original_queue):
        def get(self, block=True, timeout=None):
            idle.set()
            return super().get(block=block, timeout=5.0)

    monkeypatch.setattr(online_workmap_module.queue, "Queue", LongIdleQueue)
    session = SessionWorkMap(
        tmp_path / "wake-close", session_id="wake", base_release_id="base",
        coordinate_frame_id="frame",
    )
    assert idle.wait(1.0)
    session.archive_landmarks(_synthetic_landmarks())
    started = time.monotonic()
    session.close(shutdown_timeout_s=1.0)
    assert time.monotonic() - started < 1.0
    assert not session._writer_thread.is_alive()
    assert session._queue.unfinished_tasks == 0
    snapshot = json.loads((session.directory / "snapshot.json").read_text())
    assert set(snapshot["landmarks"]) == {str(key) for key in _synthetic_landmarks()}
    assert snapshot["statistics"]["drain_complete"] is True
    observations = (session.directory / "observations.jsonl").read_text().splitlines()
    assert any(json.loads(line).get("event") == "geometry_archive_complete" for line in observations)


@pytest.mark.parametrize("invalid", [np.array([[np.nan]]), np.float64(np.inf)])
def test_snapshot_rejects_nonfinite_numpy_keyframe_geometry(tmp_path, invalid):
    session = SessionWorkMap(
        tmp_path / "nonfinite", session_id="nonfinite", base_release_id="base",
        coordinate_frame_id="frame",
    )
    session.keyframes["bad"] = {"pose": invalid}
    with pytest.raises(ValueError, match="Out of range float"):
        session.close()
    assert not (session.directory / "snapshot.json").exists()


def test_close_records_phase_timings_and_snapshot_size(tmp_path):
    session = SessionWorkMap(
        tmp_path / "timing", session_id="timing", base_release_id="base",
        coordinate_frame_id="frame",
    )
    session.close()
    snapshot = json.loads((session.directory / "snapshot.json").read_text())
    timings = snapshot["statistics"]["close_session_ms"]
    assert set(timings) == {"drain", "writer_join", "index_join", "events_flush"}
    assert all(isinstance(value, float) and math.isfinite(value) and value >= 0.0 for value in timings.values())
    meta = json.loads((session.directory / "session_meta.json").read_text())
    assert meta["snapshot_bytes"] == (session.directory / "snapshot.json").stat().st_size
    assert math.isfinite(meta["snapshot_write_ms"]) and meta["snapshot_write_ms"] >= 0.0


def test_ba_rejection_event_carries_submit_to_apply_latency(tmp_path):
    session = SessionWorkMap(
        tmp_path / "reject", session_id="reject", base_release_id="base",
        coordinate_frame_id="frame",
    )
    session.record_ba_rejection("old_support_count", {"snapshot_revision": 3, "source_epoch": 0}, 12.5)
    session.record_ba_rejection("source_epoch", None)
    session.close()
    events = [json.loads(line) for line in (session.directory / "events.jsonl").read_text().splitlines()]
    rejected = [event for event in events if event["event"] == "ba_geometry_rejected"]
    assert [event["submit_to_apply_ms"] for event in rejected] == [12.5, None]

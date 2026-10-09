from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import live_provider as live_provider_module
from live_provider import BOQ_DIM, LiveMapEDMProvider
from river_map_quality import official_edm_adapter
from two_rate_tracker import RelocWorker


def _project(points: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = np.asarray(points) @ pose[:, :3].T + pose[:, 3]
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2:3]


def _descriptor() -> np.ndarray:
    result = np.zeros(BOQ_DIM, dtype=np.float32)
    result[0] = 1.0
    return result


def _write_image(path: Path) -> Path:
    assert cv2.imwrite(str(path), np.full((480, 640, 3), 90, dtype=np.uint8))
    return path


def _fixture(tmp_path: Path, *, top_k: int = 2, conflict: bool = False):
    rng = np.random.default_rng(20260926)
    K = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    points = rng.uniform([-1.1, -0.8, 6.0], [1.1, 0.8, 9.0], size=(24, 3))
    new_pose = np.column_stack((np.eye(3), np.array([0.17, -0.08, 0.0])))
    new_xy = _project(points, new_pose, K)
    new_ids = -np.arange(1, len(points) + 1, dtype=np.int64)
    refs = []
    observations = {}
    point_rows = {int(101 + i): points[i] for i in range(len(points))}
    if conflict:
        point_rows[900] = points[0] + np.array([4.0, 0.0, 0.0])

    for index, translation in enumerate(([0.0, 0.0, 0.0], [0.45, 0.04, 0.0])):
        name = f"frozen-{index}"
        pose = np.column_stack((np.eye(3), np.asarray(translation, dtype=np.float64)))
        old_xy = _project(points, pose, K)
        old_ids = 101 + np.arange(len(points), dtype=np.int64)
        if conflict and index == 1:
            old_ids[0] = 900
        path = _write_image(tmp_path / f"{name}.png")
        refs.append((name, path, pose, old_xy))
        observations[name] = (old_xy, old_ids)

    all_point_ids = np.asarray(sorted(point_rows), dtype=np.int64)
    all_xyz = np.asarray([point_rows[int(point_id)] for point_id in all_point_ids])
    provider = object.__new__(LiveMapEDMProvider)
    provider.profile = SimpleNamespace(
        reloc=SimpleNamespace(top_k=top_k, lift_distance_px=1.5),
        optimizations=SimpleNamespace(gpu_cache_size=0),
    )
    provider._lock = threading.Lock()
    provider._models_ready = True
    provider._index = SimpleNamespace(index_id="global")
    provider._subsets = {
        "global": SimpleNamespace(indices=tuple(range(len(refs))), excluded_sessions=frozenset())
    }
    provider._reference_descriptors = np.zeros((len(refs), BOQ_DIM), dtype=np.float32)
    provider._reference_descriptors[:, 0] = 1.0
    provider._reference_descriptors[:, 1] = np.arange(len(refs), dtype=np.float32) * 0.01
    provider._reference_norms = np.linalg.norm(provider._reference_descriptors, axis=1)
    provider._reference_names = tuple(row[0] for row in refs)
    provider._reference_sessions = tuple("frozen-release" for _ in refs)
    provider._reference_occupied_bins = tuple(64 for _ in refs)
    provider.min_reference_occupied_bins = 0
    provider._reference_paths = tuple(row[1] for row in refs)
    provider._observations = observations
    provider._point_ids = all_point_ids
    provider._point_xyz = all_xyz
    provider.edm_config = {"confidence_threshold": 0.0}
    provider._workmap_session_id = "session-a"
    provider._query_matches = {}
    provider._gpu_matcher = None
    provider._matcher_runtime = lambda: object()
    provider._vpr = lambda: object()
    provider._get_prepared_reference = lambda name, _path, _runtime: (
        ("reference", name),
        (480, 640),
    )

    query_image = _write_image(tmp_path / "session-kf.png")
    durable_reference = {
        "name": "kf-17",
        "image_path": str(query_image),
        "ids": new_ids,
        "xy": new_xy,
        "xyz": points,
        "pose": new_pose,
        "K": K,
        "revision": 7,
        "session_id": "session-a",
        "source_epoch": 4,
        "landmark_revisions": {int(point_id): 3 for point_id in new_ids},
        "pose_revision": 9,
        "ordinal": 17,
        "stamp": 1.25,
        "image_size": [640, 480],
    }
    return provider, durable_reference, refs, new_xy, new_ids


def _install_matching(monkeypatch, provider, refs, new_xy, *, reciprocal: bool = True):
    prepared_query = object()
    monkeypatch.setattr(live_provider_module, "boq_descriptor_from_array", lambda *_: _descriptor())
    monkeypatch.setattr(
        official_edm_adapter,
        "prepare_official_megadepth_image_from_array",
        lambda *_: prepared_query,
    )
    match_calls: list[tuple[str, str]] = []

    def match(_runtime, query, reference):
        if query is prepared_query:
            name = reference[1]
            match_calls.append(("forward", name))
            old_xy = next(row[3] for row in refs if row[0] == name)
            return {
                "mkpts0_f": new_xy,
                "mkpts1_f": old_xy,
                "mconf": np.full(len(new_xy), 0.9, dtype=np.float32),
            }
        if reference is prepared_query:
            name = query[1]
            match_calls.append(("reverse", name))
            old_xy = next(row[3] for row in refs if row[0] == name)
            rows = np.arange(len(new_xy) if reciprocal else 0, dtype=np.int64)
            if reciprocal:
                rows = rows[:-2]
            return {
                "mkpts0_f": old_xy[rows],
                "mkpts1_f": new_xy[rows],
                "mconf": np.full(len(rows), 0.8, dtype=np.float32),
            }
        raise AssertionError("unexpected prepared-image direction")

    provider._match_prepared = match
    return match_calls


def test_overlap_requires_reciprocal_matches_and_deduplicates_old_identities(tmp_path, monkeypatch):
    provider, reference, refs, new_xy, new_ids = _fixture(tmp_path, top_k=2)
    calls = _install_matching(monkeypatch, provider, refs, new_xy)

    result = provider.match_workmap_overlap(reference)

    assert calls == [
        ("forward", "frozen-0"),
        ("reverse", "frozen-0"),
        ("forward", "frozen-1"),
        ("reverse", "frozen-1"),
    ]
    assert result["keyframe_name"] == "kf-17"
    assert result["keyframe_revision"] == 7
    assert result["pose_revision"] == 9
    assert result["source_epoch"] == 4
    assert result["image_size"] == [640, 480]
    assert len(result["pairs"]) == len(new_ids) - 2
    assert len({(pair["new_id"], pair["old_id"]) for pair in result["pairs"]}) == len(
        result["pairs"]
    )
    by_new_id = {pair["new_id"]: pair for pair in result["pairs"]}
    for index in range(len(new_ids) - 2):
        pair = by_new_id[int(new_ids[index])]
        assert pair["old_id"] == 101 + index
        assert pair["old_reference_name"] == "frozen-0"
        np.testing.assert_allclose(pair["new_xy"], reference["xy"][index])
        np.testing.assert_allclose(pair["old_xy"], refs[0][3][index])
        np.testing.assert_allclose(pair["new_xyz"], reference["xyz"][index])
        np.testing.assert_allclose(pair["old_xyz"], reference["xyz"][index])
        np.testing.assert_allclose(pair["confidence"], 0.8)
        assert pair["keyframe_name"] == "kf-17"
        assert pair["keyframe_revision"] == 7
        assert pair["pose_revision"] == 9
        assert pair["keyframe_pose"] == reference["pose"].tolist()
        assert pair["keyframe_K"] == reference["K"].tolist()
        assert pair["image_size"] == [640, 480]


def test_overlap_retrieves_no_more_than_profile_top_k_frozen_references(tmp_path, monkeypatch):
    provider, reference, refs, new_xy, _new_ids = _fixture(tmp_path, top_k=1)
    calls = _install_matching(monkeypatch, provider, refs, new_xy)

    result = provider.match_workmap_overlap(reference)

    assert calls == [("forward", "frozen-0"), ("reverse", "frozen-0")]
    assert result["pairs"]


def test_conflicting_new_to_old_identity_is_rejected(tmp_path, monkeypatch):
    provider, reference, refs, new_xy, new_ids = _fixture(tmp_path, top_k=2, conflict=True)
    _install_matching(monkeypatch, provider, refs, new_xy)

    result = provider.match_workmap_overlap(reference)

    assert len({pair["new_id"] for pair in result["pairs"]}) >= 12
    assert len({pair["old_id"] for pair in result["pairs"]}) == len(result["pairs"])
    assert all(pair["new_id"] != int(new_ids[0]) for pair in result["pairs"])
    assert all(pair["old_id"] not in {101, 900} for pair in result["pairs"])


def test_duplicate_frozen_track_id_is_excluded_without_disabling_the_reference(
    tmp_path, monkeypatch
):
    provider, reference, refs, new_xy, _new_ids = _fixture(tmp_path, top_k=1)
    old_xy, old_ids = provider._observations["frozen-0"]
    provider._observations["frozen-0"] = (
        np.vstack((old_xy, old_xy[2] + [0.3, 0.0])),
        np.append(old_ids, old_ids[2]),
    )
    _install_matching(monkeypatch, provider, refs, new_xy)

    pairs = provider.match_workmap_overlap(reference)["pairs"]

    assert len({row["new_id"] for row in pairs}) >= 12
    assert all(row["old_id"] != int(old_ids[2]) for row in pairs)
    assert all(row["old_reference_name"] == "frozen-0" for row in pairs)


def test_overlap_returns_metadata_but_no_pairs_without_reciprocal_evidence(tmp_path, monkeypatch):
    provider, reference, refs, new_xy, _new_ids = _fixture(tmp_path, top_k=2)
    _install_matching(monkeypatch, provider, refs, new_xy, reciprocal=False)

    result = provider.match_workmap_overlap(reference)

    assert result["pairs"] == []
    assert result["session_id"] == "session-a"
    assert result["source_epoch"] == 4
    assert result["keyframe_name"] == "kf-17"
    np.testing.assert_array_equal(result["keyframe_pose"], reference["pose"])
    np.testing.assert_array_equal(result["keyframe_K"], reference["K"])


def test_overlap_worker_is_bounded_serial_and_fair_with_query_and_index_jobs():
    class OrderedProvider:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.overlap_done = threading.Event()
            self.calls: list[str] = []
            self.thread_ids: list[int] = []

        def localize_array(self, gray):
            marker = int(gray[0, 0])
            self.calls.append(f"query:{marker}")
            self.thread_ids.append(threading.get_ident())
            if marker == 1:
                self.entered.set()
                assert self.release.wait(timeout=5.0)
            return SimpleNamespace(ok=False)

        def match_workmap_overlap(self, reference):
            self.calls.append("overlap")
            self.thread_ids.append(threading.get_ident())
            return {"keyframe_name": reference["name"], "pairs": []}

        def index_workmap_reference(self, reference):
            self.calls.append("index")
            self.thread_ids.append(threading.get_ident())
            return f"workmap:{reference['name']}"

    provider = OrderedProvider()
    worker = RelocWorker(provider)
    worker.start()
    overlap_result = []
    index_callback = threading.Event()
    index_submit_done = threading.Event()
    submitter = None
    try:
        assert worker.submit(np.full((4, 4), 1, np.uint8), 1, 1.0, 0)
        assert provider.entered.wait(timeout=5.0)
        assert worker.submit(np.full((4, 4), 2, np.uint8), 2, 2.0, 0)

        def overlap_callback(result, error):
            overlap_result.append((result, error))
            provider.overlap_done.set()

        assert worker.submit_overlap({"name": "kf-7"}, overlap_callback)
        assert not worker.submit_overlap({"name": "kf-8"}, lambda *_: None)

        def queue_index():
            assert worker.submit_index(
                {"name": "kf-7"},
                lambda _reference, _result, _error: index_callback.set(),
            )
            index_submit_done.set()

        submitter = threading.Thread(target=queue_index)
        submitter.start()
        deadline = time.monotonic() + 5.0
        while (
            not worker.workmap_diagnostics()["workmap_index_pending"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        diagnostics = worker.workmap_diagnostics()
        assert diagnostics["workmap_index_pending"]
        assert diagnostics["workmap_overlap_pending"]
        assert diagnostics["workmap_overlap_oldest_age_ms"] >= 0.0

        assert worker.submit(np.full((4, 4), 3, np.uint8), 3, 3.0, 0)
        assert worker.submit(np.full((4, 4), 4, np.uint8), 4, 4.0, 0)
        provider.release.set()
        assert provider.overlap_done.wait(timeout=5.0)
        assert index_callback.wait(timeout=5.0)
        assert index_submit_done.wait(timeout=5.0)
        submitter.join(timeout=5.0)

        assert provider.calls == ["query:1", "query:4", "overlap", "index"]
        assert len(set(provider.thread_ids)) == 1
        assert overlap_result == [({"keyframe_name": "kf-7", "pairs": []}, None)]
        diagnostics = worker.workmap_diagnostics()
        assert diagnostics["workmap_overlap_pending"] is False
        assert diagnostics["workmap_overlap_healthy"] is True
    finally:
        provider.release.set()
        worker.close()
        if submitter is not None:
            submitter.join(timeout=5.0)


def test_overlap_worker_passes_provider_errors_to_callback():
    class FailingProvider:
        def match_workmap_overlap(self, _reference):
            raise ValueError("bad durable keyframe")

    worker = RelocWorker(FailingProvider())
    worker.start()
    completed = threading.Event()
    results = []
    try:
        assert worker.submit_overlap(
            {"name": "kf-bad"},
            lambda result, error: (results.append((result, error)), completed.set()),
        )
        assert completed.wait(timeout=5.0)
        assert results[0][0] is None
        assert results[0][1] == "ValueError: bad durable keyframe"
        assert worker.workmap_diagnostics()["workmap_overlap_healthy"] is False
    finally:
        worker.close()


def _reference_reciprocal(forward, reverse, tolerance_px):
    """The original grid implementation, kept verbatim as the oracle."""
    forward0 = np.asarray(forward["mkpts0_f"], dtype=np.float64).reshape(-1, 2)
    forward1 = np.asarray(forward["mkpts1_f"], dtype=np.float64).reshape(-1, 2)
    forward_conf = np.asarray(forward["mconf"], dtype=np.float64).reshape(-1)
    reverse0 = np.asarray(reverse["mkpts0_f"], dtype=np.float64).reshape(-1, 2)
    reverse1 = np.asarray(reverse["mkpts1_f"], dtype=np.float64).reshape(-1, 2)
    reverse_conf = np.asarray(reverse["mconf"], dtype=np.float64).reshape(-1)
    reverse_grid = {}
    for index, point in enumerate(reverse1):
        reverse_grid.setdefault(tuple(np.floor(point / tolerance_px).astype(np.int64)), []).append(
            index
        )
    reciprocal = []
    radius_sq = tolerance_px * tolerance_px
    for forward_index, (point0, point1) in enumerate(zip(forward0, forward1)):
        cell = np.floor(point0 / tolerance_px).astype(np.int64)
        best = None
        for x_offset in (-1, 0, 1):
            for y_offset in (-1, 0, 1):
                for reverse_index in reverse_grid.get(
                    (int(cell[0] + x_offset), int(cell[1] + y_offset)), ()
                ):
                    delta0 = point0 - reverse1[reverse_index]
                    delta1 = point1 - reverse0[reverse_index]
                    distance0, distance1 = float(delta0 @ delta0), float(delta1 @ delta1)
                    if distance0 > radius_sq or distance1 > radius_sq:
                        continue
                    confidence = min(
                        float(forward_conf[forward_index]), float(reverse_conf[reverse_index])
                    )
                    candidate = (distance0 + distance1, -confidence, reverse_index)
                    if best is None or candidate < best:
                        best = candidate
        if best is not None:
            reciprocal.append((best[0], best[1], forward_index, best[2]))
    reciprocal.sort()
    used_forward, used_reverse, rows, confidences = set(), set(), [], []
    for _distance, negative_confidence, forward_index, reverse_index in reciprocal:
        if forward_index in used_forward or reverse_index in used_reverse:
            continue
        used_forward.add(forward_index)
        used_reverse.add(reverse_index)
        rows.append(forward_index)
        confidences.append(-negative_confidence)
    return rows, confidences


def test_vectorized_reciprocal_matches_equal_the_grid_reference():
    rng = np.random.default_rng(7)
    for trial in range(40):
        count = int(rng.integers(0, 300))
        truth0 = rng.uniform(0, 640, size=(count, 2))
        truth1 = rng.uniform(0, 640, size=(count, 2))
        # Integer-snapped copies create exact distance and confidence ties.
        noise = np.round(rng.normal(scale=1.2, size=(count, 2)) * 2) / 2
        forward = {
            "mkpts0_f": truth0,
            "mkpts1_f": truth1,
            "mconf": np.round(rng.uniform(size=count), 1),
        }
        permutation = rng.permutation(count)
        extra = rng.uniform(0, 640, size=(count // 3, 2))
        reverse = {
            "mkpts0_f": np.vstack([truth1[permutation] + noise, extra]),
            "mkpts1_f": np.vstack([truth0[permutation] + noise[::-1], extra[::-1]]),
            "mconf": np.round(rng.uniform(size=count + len(extra)), 1),
        }
        tolerance = float(rng.choice([0.5, 1.0, 2.0]))
        rows, confidences = live_provider_module._reciprocal_edm_matches(
            forward, reverse, tolerance_px=tolerance
        )
        expected_rows, expected_confidences = _reference_reciprocal(forward, reverse, tolerance)
        assert rows.tolist() == expected_rows, trial
        assert confidences.tolist() == expected_confidences, trial

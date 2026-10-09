from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import live_provider as live_provider_module
from live_provider import BOQ_DIM, LiveMapEDMProvider, RelocFix
from river_map_quality import official_edm_adapter
from sfm_diagnosis.viewpoint_policy import FROZEN_PNP_THRESHOLDS


class PreparedCache:
    def get_or_prepare(self, key, path, runtime):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert image is not None
        return key, image.shape

    def clear(self):
        pass


def _descriptor(*_args):
    result = np.zeros(BOQ_DIM, dtype=np.float32)
    result[0] = 1.0
    return result


def _provider(tmp_path):
    provider = object.__new__(LiveMapEDMProvider)
    provider.profile = SimpleNamespace(
        reloc=SimpleNamespace(
            min_points=12,
            top_k=1,
            batch_refs=False,
            period_s=0.1,
            frozen_pnp_thresholds=dict(FROZEN_PNP_THRESHOLDS),
        ),
        optimizations=SimpleNamespace(adaptive_retrieval=False, gpu_cache_size=0),
    )
    provider._lock = threading.Lock()
    provider._models_ready = True
    provider._vpr_runtime = object()
    provider._index = SimpleNamespace(index_id="global")
    provider._subsets = {"global": SimpleNamespace(indices=(), excluded_sessions=frozenset())}
    provider._reference_descriptors = np.empty((0, BOQ_DIM), dtype=np.float32)
    provider._reference_norms = np.empty((0,), dtype=np.float32)
    provider._reference_names = ()
    provider._reference_sessions = ()
    provider._reference_occupied_bins = ()
    provider.min_reference_occupied_bins = 0
    provider.top_k = 1
    provider.lift_distance_px = 2.0
    provider.thresholds = dict(FROZEN_PNP_THRESHOLDS)
    provider.edm_config = {"confidence_threshold": 0.0}
    provider._queries = {"live_frame": {"image_path": str(tmp_path / "query.png")}}
    provider._intersection_cells = np.empty((0, 3), dtype=np.float64)
    provider.intrinsics_calibration = {
        "image_width": 640,
        "image_height": 480,
        "K": [[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]],
        "images_are_undistorted": True,
    }
    provider._gpu_matcher = None
    provider._query_matches = {}
    provider._query_lifts = {}
    provider._recovery_attempted = set()
    provider._last_strong_refs = ()
    provider.last_stage_ms = {}
    provider._reference_norms = np.empty((0,), dtype=np.float32)
    provider.startup_stage_ms = {}
    provider._workmap_session_id = None
    provider._workmap_references = {}
    provider._workmap_order = []
    provider._workmap_landmark_refs = {}
    provider._workmap_descriptors = np.empty((0, BOQ_DIM), dtype=np.float32)
    provider._workmap_descriptor_count = 0
    provider._workmap_descriptor_capacity = 0
    provider._last_successful_workmap_reference = None
    provider._prepared_reference_cache = PreparedCache()
    provider._matcher_runtime = lambda: object()
    return provider


def _write_reference_image(path):
    assert cv2.imwrite(str(path), np.full((480, 640, 3), 90, dtype=np.uint8))
    return path


def _project(points, K, translation):
    camera_points = points + np.asarray(translation, dtype=np.float64)
    pixels = camera_points @ K.T
    return pixels[:, :2] / pixels[:, 2:3]


def _reference_payload(path, points, reference_xy, K, *, session_id="session-a"):
    ids = -np.arange(1, len(points) + 1, dtype=np.int64)
    return {
        "name": "kf-7",
        "image_path": str(path),
        "ids": ids,
        "xy": reference_xy,
        "xyz": points,
        "pose": np.column_stack((np.eye(3), np.array([0.25, 0.0, 0.0]))),
        "K": K,
        "revision": 7,
        "session_id": session_id,
        "source_epoch": 0,
        "landmark_revisions": {int(point_id): 3 for point_id in ids},
        "pose_revision": 4,
        "ordinal": 7,
        "stamp": 1.25,
        "image_size": [640, 480],
    }


def test_index_failure_is_not_published_as_a_ready_session_reference(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    path = _write_reference_image(tmp_path / "keyframe.png")
    K = np.asarray(provider.intrinsics_calibration["K"], dtype=np.float64)
    points = np.column_stack(
        (
            np.linspace(-1.5, 1.5, 20),
            np.linspace(-1.0, 1.0, 20),
            np.full(20, 7.0),
        )
    )
    payload = _reference_payload(path, points, _project(points, K, [0.25, 0, 0]), K)
    monkeypatch.setattr(
        live_provider_module,
        "boq_descriptor_from_array",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("descriptor failed")),
    )

    with pytest.raises(RuntimeError, match="descriptor failed"):
        provider.index_workmap_reference(payload)

    assert provider.session_reference_count == 0
    assert provider._workmap_references == {}
    assert provider._workmap_session_id is None


def test_index_decodes_session_keyframe_as_grayscale_for_boq(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    path = _write_reference_image(tmp_path / "keyframe.png")
    K = np.asarray(provider.intrinsics_calibration["K"], dtype=np.float64)
    points = np.column_stack(
        (
            np.linspace(-1.5, 1.5, 20),
            np.linspace(-1.0, 1.0, 20),
            np.full(20, 7.0),
        )
    )
    payload = _reference_payload(path, points, _project(points, K, [0.25, 0, 0]), K)
    extracted = []

    def descriptor(_extractor, image):
        extracted.append(image.copy())
        return _descriptor()

    monkeypatch.setattr(live_provider_module, "boq_descriptor_from_array", descriptor)
    provider.index_workmap_reference(payload)

    assert provider.session_reference_count == 1
    assert len(extracted) == 1
    assert extracted[0].shape == (480, 640)
    assert extracted[0].dtype == np.uint8
    assert np.all(extracted[0] == 90)


def test_global_winner_skips_workmap_candidate_search(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    descriptor = _descriptor()
    provider._reference_descriptors = descriptor[None, :]
    provider._reference_norms = np.array([1.0], dtype=np.float32)
    provider._reference_sessions = ("frozen-release",)
    provider._reference_occupied_bins = (64,)
    provider._reference_names = ("global-kf",)
    provider._subsets["global"] = SimpleNamespace(indices=(0,), excluded_sessions=frozenset())
    provider._workmap_descriptor_count = 1
    provider._workmap_descriptors = descriptor[None, :]
    query_path = tmp_path / "query.png"
    _write_reference_image(query_path)
    monkeypatch.setattr(live_provider_module, "boq_descriptor_from_array", _descriptor)
    monkeypatch.setattr(
        official_edm_adapter,
        "prepare_official_megadepth_image_from_array",
        lambda image, runtime: ("prepared-query", image.shape),
    )
    calls = []
    global_fix = RelocFix(
        ok=True,
        status="LOCALIZED_STRONG",
        cam_from_world=np.column_stack((np.eye(3), np.zeros(3))),
        query_xy=np.zeros((80, 2)),
        point3d_ids=np.arange(80, dtype=np.int64),
        point_xyz=np.zeros((80, 3)),
        inliers=80,
        inlier_ratio=1.0,
        reproj_p90=0.1,
        reference_names=("global-kf",),
        runtime_ms=2.0,
    )
    provider._solve_references = lambda image, prepared, ranked, started, fallback: (
        calls.append(tuple(ranked)) or global_fix
    )
    provider._rank_workmap_references = lambda _descriptor: pytest.fail(
        "session retrieval must not run after a global winner"
    )

    fix = provider.localize_array(np.zeros((480, 640), dtype=np.uint8))

    assert calls == [(0,)]
    assert fix.ok and fix.reference_scope == "global"
    assert fix.status == "LOCALIZED_STRONG"


def test_workmap_batch_is_capped_and_keeps_new_boq_winner(tmp_path):
    provider = _provider(tmp_path)
    provider.top_k = 3
    names = ("old-zone", "old-neighbor", "new-zone-winner", "boq-second", "boq-third")
    descriptors = np.zeros((len(names), BOQ_DIM), dtype=np.float32)
    descriptors[:, 0] = [0.5, 0.4, 1.0, 0.9, 0.8]
    provider._workmap_order = list(names)
    provider._workmap_descriptors = descriptors
    provider._workmap_descriptor_count = len(names)
    provider._workmap_references = {
        "old-zone": {"ids": np.array([-1, -2], dtype=np.int64)},
        "old-neighbor": {"ids": np.array([-1, -2, -3], dtype=np.int64)},
        "new-zone-winner": {"ids": np.array([-4], dtype=np.int64)},
        "boq-second": {"ids": np.array([-5], dtype=np.int64)},
        "boq-third": {"ids": np.array([-6], dtype=np.int64)},
    }
    for index, name in enumerate(names):
        provider._workmap_references[name]["descriptor_index"] = index
        provider._reindex_workmap_landmarks(name, (), provider._workmap_references[name]["ids"])
    provider._last_successful_workmap_reference = "old-zone"

    candidates = provider._rank_workmap_references(_descriptor())

    assert candidates == ("new-zone-winner", "old-zone", "old-neighbor")
    assert len(candidates) <= provider.top_k


def test_session_boq_candidate_runs_real_pnp_and_restores_negative_ids(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    K = np.asarray(provider.intrinsics_calibration["K"], dtype=np.float64)
    rng = np.random.default_rng(20260926)
    points = rng.uniform([-2.7, -2.0, 6.0], [2.7, 2.0, 9.0], size=(180, 3))
    reference_xy = _project(points, K, [0.25, 0.0, 0.0])
    query_xy = _project(points, K, [-0.1, 0.06, 0.0])
    reference_path = _write_reference_image(tmp_path / "session-kf.png")
    query_path = _write_reference_image(tmp_path / "query.png")
    provider._queries["live_frame"]["image_path"] = str(query_path)

    monkeypatch.setattr(live_provider_module, "boq_descriptor_from_array", _descriptor)
    monkeypatch.setattr(
        official_edm_adapter,
        "prepare_official_megadepth_image_from_array",
        lambda image, runtime: ("prepared-query", image.shape),
    )
    indexed_name = provider.index_workmap_reference(
        _reference_payload(reference_path, points, reference_xy, K)
    )
    provider._match_prepared = lambda runtime, query, reference: {
        "mkpts0_f": query_xy,
        "mkpts1_f": reference_xy,
        "mconf": np.ones(len(points), dtype=np.float32),
    }

    fix = provider.localize_array(np.zeros((480, 640), dtype=np.uint8))

    assert provider.session_reference_count == 1
    assert fix.ok
    assert fix.reference_scope == "workmap"
    assert fix.status == "VO_ONLY"
    assert fix.reference_names == (indexed_name,)
    assert len(fix.point3d_ids) >= FROZEN_PNP_THRESHOLDS["strong_inliers"]
    assert np.all(fix.point3d_ids < 0)
    np.testing.assert_allclose(fix.cam_from_world[:3, :3], np.eye(3), atol=0.01)
    np.testing.assert_allclose(fix.cam_from_world[:3, 3], [-0.1, 0.06, 0.0], atol=0.03)
    np.testing.assert_allclose(
        fix.point_xyz,
        points[-fix.point3d_ids - 1],
        atol=1e-10,
    )
    assert fix.workmap_snapshot is not None
    assert fix.workmap_snapshot["session_id"] == "session-a"
    assert fix.workmap_snapshot["keyframe_revision"] == 4
    assert set(fix.workmap_snapshot["landmark_revisions"]) == set(fix.point3d_ids.tolist())
    assert provider._reference_names == ()
    assert provider._reference_descriptors.shape == (0, BOQ_DIM)

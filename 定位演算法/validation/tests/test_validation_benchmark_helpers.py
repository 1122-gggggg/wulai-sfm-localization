from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "定位演算法" / "validation"


def load_script(name: str, filename: str):
    path = VALIDATION / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval_script(monkeypatch):
    fake_root = Path("/tmp/sfm_system_eval_helpers")
    monkeypatch.setenv("SFM_SYSTEM_ROOT", str(fake_root))

    reloc = types.ModuleType("reloc_localizer_xfeat")
    reloc.MegaLocQuery = object
    reloc.bundle_vpr_kind = lambda _meta: "mock"
    reloc.extract_xfeat = lambda *_args: None
    reloc.load_verified_bundle = lambda *_args, **_kwargs: {}
    reloc.load_xfeat = lambda _qk: object()
    cache = types.ModuleType("megaloc_cache")
    cache.load_megaloc_cache = lambda *_args, **_kwargs: None
    integrity = types.ModuleType("stream_integrity")
    integrity.StreamAudit = object
    integrity.ffprobe_frame_count = lambda _path: (None, None)
    integrity.iter_rgb_frames = lambda *_args: iter(())
    pycolmap = types.ModuleType("pycolmap")
    for name, module in (
        ("reloc_localizer_xfeat", reloc),
        ("megaloc_cache", cache),
        ("stream_integrity", integrity),
        ("pycolmap", pycolmap),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    path = VALIDATION / "eval_stream_core.py"
    spec = importlib.util.spec_from_file_location("eval_stream_core_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_replay_helpers_limit_rows_and_keep_summary_fields(tmp_path, capsys):
    replay = load_script("replay_route_commands_helpers", "replay_route_commands.py")
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"rows": [{"idx": 1}, {"idx": 2}]}), encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"rows": []}), encoding="utf-8")

    assert replay._load_rows(str(bench), 1) == [{"idx": 1}]
    with pytest.raises(SystemExit, match="no rows"):
        replay._load_rows(str(empty), 0)

    replay._print_replay_summary(
        "stop",
        [{"idx": 1}],
        [
            {"blocked": False, "path_error_u": 1.0},
            {"blocked": True, "reason": "lost (timeout)", "path_error_u": None},
        ],
        [(0, 0, 0, 0), (1, 0, 0, 0)],
        {"i": 1, "reloc": 2},
        tmp_path / "commands.jsonl",
    )
    output = capsys.readouterr().out
    assert "ticks logged         : 2  (driven=1, blocked=1)" in output
    assert "PCMD sent            : 2  (nonzero=1, zero/hover=1)" in output
    assert "lost" in output


def test_xfeat_pair_selection_skips_short_references_and_honors_limit():
    xfeat = load_script("benchmark_xfeat_lg_onnx_helpers", "benchmark_xfeat_lg_onnx.py")
    reloc_map = SimpleNamespace(
        ref_names=["a", "b", "c"],
        covis={"a": [1, 2], "b": [2], "c": []},
        refs={
            "a": SimpleNamespace(feats={"keypoints": [0, 0]}),
            "b": SimpleNamespace(feats={"keypoints": [0]}),
            "c": SimpleNamespace(feats={"keypoints": [0, 0, 0]}),
        },
    )

    assert xfeat._select_covisible_pairs(reloc_map, 2, 2, 1) == [(0, 2)]
    assert xfeat._select_covisible_pairs(reloc_map, 2, 2, 3) == [(0, 2)]


def test_edm_stream_helper_handles_stride_warmup_and_metrics(monkeypatch):
    stream = load_script("bench_edm_onnx_stream_helpers", "bench_edm_onnx_stream.py")

    class Capture:
        def __init__(self):
            self.frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(5)]

        def isOpened(self):
            return True

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self):
            pass

    capture = Capture()
    monkeypatch.setattr(stream.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(stream.cv2, "resize", lambda frame, _size, interpolation: frame)
    monkeypatch.setattr(stream.time, "perf_counter", iter((10.0, 12.5)).__next__)
    args = SimpleNamespace(
        video=Path("mock.mp4"), max_frames=2, warmup_frames=1, stride=2,
    )
    infos = iter((
        {"state_out": "BOOT", "ok": True, "inliers": 5, "total_ms": 10.0,
         "vpr_ms": 1.0, "match_ms": 2.0, "pnp_ms": 3.0},
        {"state_out": "TRACK", "ok": True, "inliers": 7, "total_ms": 20.0,
         "vpr_ms": 2.0, "match_ms": 3.0, "pnp_ms": 4.0},
        {"state_out": "LOST", "ok": False, "inliers": 0, "total_ms": 30.0,
         "vpr_ms": 3.0, "match_ms": 4.0, "pnp_ms": 5.0},
    ))
    tracker = SimpleNamespace(localize=lambda _frame: next(infos))

    states, totals, vprs, matches, pnps, inliers, n_ok, used, timed, wall_s = (
        stream._run_stream(args, tracker)
    )

    assert (states, totals, vprs, matches, pnps, inliers) == (
        ["TRACK", "LOST"], [20.0, 30.0], [2.0, 3.0], [3.0, 4.0],
        [4.0, 5.0], [7],
    )
    assert (n_ok, used, timed, wall_s) == (1, 3, 2, 2.5)


def test_engine_runner_binds_inputs_and_returns_copies(monkeypatch):
    xfeat = load_script("benchmark_xfeat_lg_onnx_engine_helpers", "benchmark_xfeat_lg_onnx.py")

    class Tensor:
        def __init__(self, pointer, value):
            self.pointer = pointer
            self.value = np.asarray(value)

        def data_ptr(self):
            return self.pointer

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Context:
        def __init__(self):
            self.addresses = {}

        def set_tensor_address(self, name, pointer):
            self.addresses[name] = pointer

        def execute_async_v3(self, *, stream_handle):
            assert stream_handle == 17
            return True

    monkeypatch.setattr(xfeat.torch.cuda, "synchronize", lambda: None)
    context = Context()
    matches = Tensor(20, [[1, -1]])
    scores = Tensor(30, [[0.5, 0.0]])
    actual_matches, actual_scores = xfeat._run_engine(
        context, 17, {"keypoints0": Tensor(10, [[0.0, 1.0]])}, matches, scores,
    )

    assert context.addresses == {"keypoints0": 10}
    np.testing.assert_array_equal(actual_matches, [[1, -1]])
    np.testing.assert_array_equal(actual_scores, [[0.5, 0.0]])


def test_eval_stream_input_and_report_helpers_are_mockable(monkeypatch, tmp_path):
    stream = load_eval_script(monkeypatch)
    video = tmp_path / "flight.mp4"
    video.write_bytes(b"mock")
    monkeypatch.setattr(stream, "ffprobe_frame_count", lambda _path: (120, "ffprobe"))
    args = SimpleNamespace(
        resize="640x360",
        test_dir=str(tmp_path),
        stride=10,
        min_sampled_frames=30,
        expected_raw_frames=["flight.mp4=120"],
    )

    resize_wh, sets = stream._prepare_eval_inputs(args)

    assert resize_wh == (640, 360)
    assert sets == [("flight", str(video), 10, 120, "cli:flight.mp4+ffprobe")]

    output = tmp_path / "eval.json"
    stream._configure_runtime(SimpleNamespace(topk=7, min_conf=0.25, min_inliers=4))
    stream._write_eval_json(
        SimpleNamespace(
            out_json=str(output), resize="640x360", stride=10,
            min_sampled_frames=30,
        ),
        [{"set": "flight", "n": 1}],
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["topk"] == 7
    assert report["min_conf"] == 0.25
    assert report["min_inliers"] == 4
    assert report["rows"] == [{"set": "flight", "n": 1}]


def test_edm_replay_loop_keeps_mock_frame_audit_and_row_schema():
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")

    class Capture:
        def __init__(self):
            self.frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
            self.released = False

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self):
            self.released = True

    pose = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    tracker = SimpleNamespace(
        trk=None,
        last_info={"next_mode": "TRACK", "inliers": 12},
        localize_frame=lambda _rgb, capture_stamp: pose,
    )
    built = SimpleNamespace(tracker=tracker)
    site = SimpleNamespace(query_camera=SimpleNamespace(width=2, height=2))
    audit = replay.StreamAudit(expected_source="partial", capture_opened=True)
    args = SimpleNamespace(max_frames=1, stride=1)
    capture = Capture()

    result = replay._run_replay(args, capture, 10.0, site, built, audit)

    rows = result[0]
    assert capture.released is True
    assert audit.decoded_raw_frames == 1
    assert audit.sampled_frames == 1
    assert result[8] == 1
    assert rows[0]["source_index"] == 0
    assert rows[0]["capture_stamp"] == 0.0
    assert rows[0]["success"] is True
    assert rows[0]["next_mode"] == "TRACK"
    assert rows[0]["inliers"] == 12


P167_MEASURED = {
    "frames": 2318,
    "source_fps": 23.976,
    "processing_fps": 20.864,
    "wall_ms": {"p50": 28.27, "p95": 105.51},
    "match_ms": {"p50": 24.01, "p95": 94.28},
    "pnp_ms": {"p50": 3.03, "p95": 13.62},
    "accepted": 743,
    "accepted_rate": 0.3205,
    "state_counts": {"TRACK": 1233, "WEAK_TRACK": 541, "LOST": 544},
    "limited_jump_unconfirmed": 495,
    "cache_hit_rate": 0.875,
    "startup_ms": 3576.6,
}


def _replay_receipt(**overrides):
    receipt = {
        "radius": 0.5881852149963379,
        "mconf_thr": 0.2,
        "coarse_topk": 3225,
        "cache_capacity": 32,
        "lost_strategy": "boot_and_lost_once",
        "lost_global_retrieval_interval": 15,
        "fused_coarse_mode": "fused",
        "runtime_sigma_mode": "reference_grid",
        "temporal_feature_cache_size": 0,
        "query_cuda_graph": False,
        "acquire_stage_mode": "full_set",
        "track_map_first": False,
        "pnp_ranked_batches": False,
        "lost_prior_strategy": "restrict_nearby",
        "lost_prior_fusion_weight": 1.0,
        "worker_mode": "sequential",
    }
    receipt.update(overrides)
    return receipt



def test_receipt_identity_fails_closed_on_override_mismatch():
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")
    actual = _replay_receipt()
    baseline = {"receipt": dict(actual, worker_mode="production-path")}
    failures = replay.receipt_identity_failures(baseline, actual)
    assert any("receipt.worker_mode mismatch" in item for item in failures)
    assert replay.receipt_identity_failures({"receipt": actual}, actual) == []


def test_receipt_identity_requires_every_override_key():
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")
    actual = _replay_receipt()
    failures = replay.receipt_identity_failures({"receipt": {"radius": actual["radius"]}}, actual)
    assert any("baseline is missing receipt.mconf_thr" in item for item in failures)
    assert any("baseline is missing receipt.coarse_topk" in item for item in failures)
    assert any("baseline is missing receipt.cache_capacity" in item for item in failures)
    assert any("baseline is missing receipt.lost_strategy" in item for item in failures)
    assert any("baseline is missing receipt.fused_coarse_mode" in item for item in failures)
    assert any("baseline is missing receipt.runtime_sigma_mode" in item for item in failures)
    assert any("baseline is missing receipt.temporal_feature_cache_size" in item for item in failures)
    assert any("baseline is missing receipt.acquire_stage_mode" in item for item in failures)
    assert any("baseline is missing receipt.lost_prior_strategy" in item for item in failures)
    assert any("baseline is missing receipt.lost_prior_fusion_weight" in item for item in failures)
    assert any("baseline is missing receipt.worker_mode" in item for item in failures)
    assert not any("baseline is missing receipt.sigma_mode" in item for item in failures)


def test_invalid_public_overrides_fail_closed(tmp_path):
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mock")
    args = SimpleNamespace(
        stride=1, max_frames=0, max_corr_total=0, radius=None,
        site_profile=tmp_path / "site.json", video=video, require_cuda=False,
        worker_mode="sequential", mconf_thr=1.5, coarse_topk=None,
        cache_capacity=None, lost_strategy=None,
        lost_global_retrieval_interval=None, sigma_mode=None,
        runtime_sigma_mode=None, temporal_feature_cache_size=None,
        acquire_stage_mode=None, lost_prior_strategy=None,
        lost_prior_fusion_weight=None,
    )
    with pytest.raises(SystemExit, match="mconf-thr"):
        replay._validate_run_inputs(args)
    args.mconf_thr = 0.2
    args.coarse_topk = 0
    with pytest.raises(SystemExit, match="coarse-topk"):
        replay._validate_run_inputs(args)
    args.coarse_topk = 3225
    args.worker_mode = "inline"
    with pytest.raises(SystemExit, match="worker-mode"):
        replay._validate_run_inputs(args)
    args.worker_mode = "sequential"
    args.runtime_sigma_mode = "fused"
    with pytest.raises(SystemExit, match="runtime-sigma-mode"):
        replay._validate_run_inputs(args)
    args.runtime_sigma_mode = "bidirectional"
    args.temporal_feature_cache_size = -1
    with pytest.raises(SystemExit, match="temporal-feature-cache-size"):
        replay._validate_run_inputs(args)
    args.temporal_feature_cache_size = 2
    args.acquire_stage_mode = "skip_gates"
    with pytest.raises(SystemExit, match="acquire-stage-mode"):
        replay._validate_run_inputs(args)
    args.acquire_stage_mode = "initial_topk"
    args.lost_prior_strategy = "teleport"
    with pytest.raises(SystemExit, match="lost-prior-strategy"):
        replay._validate_run_inputs(args)
    args.lost_prior_strategy = "score_fusion"
    args.lost_prior_fusion_weight = 0.0
    with pytest.raises(SystemExit, match="lost-prior-fusion-weight"):
        replay._validate_run_inputs(args)



class _FakeLiveClient:
    def __init__(self):
        self.ready = True
        self.unavailable = False
        self.timeout_s = 1.0
        self.restart_warmup_s = 0.0
        self._coalesce_drops = 0
        self._coalesce = None
        self._active = None
        self.in_flight = False
        self.submits = []
        self.closed = False
        self.startup_info = {"event": "ready", "ready_latency_ms": 12.0}

    def submit(self, seq, frame_name, frame, *, timing_metadata=None):
        self.submits.append((seq, frame_name))
        item = (seq, frame_name, frame, dict(timing_metadata or {}))
        if self.in_flight:
            if self._coalesce is not None:
                self._coalesce_drops += 1
            self._coalesce = item
            return True
        self.in_flight = True
        self._active = item
        return True

    def poll_results(self):
        return []


    def busy(self):
        return self.in_flight or self._coalesce is not None

    def close(self):
        self.closed = True




class _Clock:
    def __init__(self, t=1000.0):
        self.t = t
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def test_source_cadence_coalesces_to_latest_frame_and_records_drops():
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")
    clock = _Clock()
    client = _FakeLiveClient()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    decoded = [
        {
            "source_index": i, "rgb": rgb, "decode_ms": 1.0, "copy_ms": 0.5,
            "recorded_stamp": i / 23.976, "frame_id": f"src_{i:08d}",
            "frame_sha256": f"{i:064d}",
        }
        for i in range(3)
    ]
    rows, stats = replay.run_source_cadence(
        decoded, 23.976, client, sleep=clock.sleep, monotonic=clock.monotonic,
        drain_timeout_s=0.0,
    )
    assert stats["decoded"] == 3
    assert stats["submit_ok"] == 3
    assert stats["coalesce_drops"] == 1
    assert [name for _seq, name in client.submits] == [
        "src_00000000", "src_00000001", "src_00000002",
    ]
    assert clock.sleeps == pytest.approx([1.0 / 23.976, 1.0 / 23.976])
    assert client._coalesce[1] == "src_00000002"
    client.close()
    assert client.closed is True




def test_pipeline_summary_uses_measured_p167_shape():
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")
    rows = [{
        "success": True, "source_index": 0, "next_mode": "TRACK",
        "capture_source_age_ms": 0.0, "submit_source_age_ms": 1.0,
        "promote_source_age_ms": 2.0, "inference_source_age_ms": 28.27,
        "result_source_age_ms": 28.27, "copy_ms": 0.4, "gpu_span_ms": None,
        "coalesce_drops": 0, "submit_drop": 0,
        "edm_cache_hits": int(0.875 * 16), "edm_cache_misses": 2,
        "edm_cache_evictions": 0,
    }]
    summary = replay.pipeline_summary(rows, startup_ms=P167_MEASURED["startup_ms"])
    assert summary["startup_ms"] == P167_MEASURED["startup_ms"]
    assert summary["first_fix_frame"] == 0
    assert summary["copies"] == 1
    assert summary["cache"]["hits"] == 14


def test_production_stream_receipt_includes_behavior_changing_overrides():
    stream = load_script("benchmark_production_stream_helpers", "benchmark_production_stream.py")
    args = SimpleNamespace(
        radius=0.8, nn_min_score=0.85, boot_topk=30, temporal_cache_max_anchors=2048,
        lost_topk=30, worker_mode="sequential", sigma_mode=None,
        mconf_thr=None, coarse_topk=None, cache_capacity=None, lost_strategy=None,
        lost_global_retrieval_interval=None,
    )
    receipt = stream.build_receipt(args)
    assert receipt["radius"] == 0.8
    assert receipt["mconf_thr"] == 0.85
    assert receipt["coarse_topk"] == 30
    assert receipt["cache_capacity"] == 2048
    assert receipt["worker_mode"] == "sequential"
    assert receipt["fused_coarse_mode"] == "fused"
    assert "runtime_sigma_mode" in receipt
    assert "temporal_feature_cache_size" in receipt
    assert "acquire_stage_mode" in receipt
    assert "lost_prior_strategy" in receipt
    assert "lost_prior_fusion_weight" in receipt
    assert "sigma_mode" not in receipt
    mismatched = stream.receipt_identity_failures(
        {"receipt": dict(receipt, radius=1.25)}, receipt,
    )
    assert any("receipt.radius mismatch" in item for item in mismatched)


def test_streaming_corpus_receipt_defaults_preserve_current_profile():
    path = ROOT / "定位演算法" / "EDM工具包" / "tests" / "bench_streaming_corpus_edm.py"
    spec = importlib.util.spec_from_file_location("bench_streaming_corpus_edm_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        mconf_thr=0.2, edm_topk=3225, radius=None, cache_capacity=32,
        lost_strategy=None, lost_global_retrieval_interval=15,
        sigma_mode=None, worker_mode="sequential",
    )
    receipt = module.build_receipt(args)
    assert receipt["mconf_thr"] == 0.2
    assert receipt["coarse_topk"] == 3225
    assert receipt["cache_capacity"] == 32
    assert receipt["lost_global_retrieval_interval"] == 15
    assert receipt["lost_strategy"] == "boot_and_lost_once"
    assert receipt["worker_mode"] == "sequential"
    assert receipt.get("fused_coarse_mode", receipt.get("sigma_mode")) == "fused"



def _fake_built():
    return SimpleNamespace(
        config=SimpleNamespace(
            radius=0.5881852149963379,
            global_retrieval_policy="boot_and_lost_once",
            lost_global_retrieval_interval=15,
            acquire_stage_mode="full_set",
            track_map_first=False,
            pnp_ranked_batches=False,
            lost_prior_strategy="restrict_nearby",
            lost_prior_fusion_weight=1.0,
            validate=lambda: None,
        ),
        _matcher=SimpleNamespace(
            mconf_thr=0.2,
            topk=3225,
            reference_cache_size=32,
            query_cuda_graph=False,
            runtime_sigma_mode="reference_grid",
            temporal_feature_cache_size=0,
            _feature_cache_capacity={"map": 32, "temporal": 0},
        ),
    )


def test_apply_runtime_overrides_mutate_built_objects_and_receipt():
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")
    built = _fake_built()
    args = SimpleNamespace(
        worker_mode="sequential",
        sigma_mode="upstream",
        radius=None,
        mconf_thr=None,
        coarse_topk=None,
        cache_capacity=None,
        lost_strategy=None,
        lost_global_retrieval_interval=None,
        runtime_sigma_mode="bidirectional",
        temporal_feature_cache_size=3,
        query_cuda_graph=True,
        acquire_stage_mode="initial_topk",
        track_map_first=True,
        pnp_ranked_batches=True,
        lost_prior_strategy="score_fusion",
        lost_prior_fusion_weight=2.5,
    )
    replay.apply_runtime_overrides(args, built)
    assert built.config.acquire_stage_mode == "initial_topk"
    assert built.config.track_map_first is True
    assert built.config.pnp_ranked_batches is True
    assert built.config.lost_prior_strategy == "score_fusion"
    assert built.config.lost_prior_fusion_weight == 2.5
    assert built._matcher.runtime_sigma_mode == "bidirectional"
    assert built._matcher.query_cuda_graph is True
    assert built._matcher.temporal_feature_cache_size == 3
    assert built._matcher._feature_cache_capacity["temporal"] == 3
    receipt = replay.build_receipt(args, built)
    assert receipt["fused_coarse_mode"] == "upstream"
    assert receipt["runtime_sigma_mode"] == "bidirectional"
    assert receipt["temporal_feature_cache_size"] == 3
    assert receipt["acquire_stage_mode"] == "initial_topk"
    assert receipt["query_cuda_graph"] is True
    assert receipt["track_map_first"] is True
    assert receipt["pnp_ranked_batches"] is True
    assert receipt["lost_prior_strategy"] == "score_fusion"
    assert receipt["lost_prior_fusion_weight"] == 2.5
    assert receipt["fused_coarse_mode"] != receipt["runtime_sigma_mode"]
    assert "sigma_mode" not in receipt


def test_runtime_stub_defaults_reproduce_active_profile(tmp_path):
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")
    profile = tmp_path / "edm_runtime_profile.json"
    profile.write_text(json.dumps({
        "matcher": {
            "mconf_thr": 0.2,
            "coarse_topk": 3225,
            "reference_cache_size": 32,
        },
        "tracker": {
            "radius": 0.5881852149963379,
            "global_retrieval_policy": "boot_and_lost_once",
            "lost_global_retrieval_interval": 15,
            "use_temporal_reference": False,
        },
    }), encoding="utf-8")
    site = SimpleNamespace(localizer_profile=profile)
    args = SimpleNamespace(
        worker_mode="sequential",
        sigma_mode=None,
        radius=None,
        mconf_thr=None,
        coarse_topk=None,
        cache_capacity=None,
        lost_strategy=None,
        lost_global_retrieval_interval=None,
        runtime_sigma_mode=None,
        temporal_feature_cache_size=None,
        acquire_stage_mode=None,
        lost_prior_strategy=None,
        lost_prior_fusion_weight=None,
    )
    stub = replay.runtime_stub_from_profile(site, args)
    assert stub.config.acquire_stage_mode == "full_set"
    assert stub.config.lost_prior_strategy == "restrict_nearby"
    assert stub.config.lost_prior_fusion_weight == 1.0
    assert stub._matcher.runtime_sigma_mode == "reference_grid"
    assert stub._matcher.temporal_feature_cache_size == 0
    receipt = replay.build_receipt(args, stub)
    assert receipt["fused_coarse_mode"] == "fused"
    assert receipt["runtime_sigma_mode"] == "reference_grid"
    assert receipt["temporal_feature_cache_size"] == 0
    assert receipt["acquire_stage_mode"] == "full_set"
    assert receipt["lost_prior_strategy"] == "restrict_nearby"
    assert receipt["lost_prior_fusion_weight"] == 1.0
    args.runtime_sigma_mode = "bidirectional"
    args.temporal_feature_cache_size = 4
    args.acquire_stage_mode = "initial_topk"
    args.lost_prior_strategy = "full_global"
    args.lost_prior_fusion_weight = 3.0
    overridden = replay.runtime_stub_from_profile(site, args)
    receipt = replay.build_receipt(args, overridden)
    assert overridden._matcher.runtime_sigma_mode == "bidirectional"
    assert overridden.config.acquire_stage_mode == "initial_topk"
    assert receipt["runtime_sigma_mode"] == "bidirectional"
    assert receipt["temporal_feature_cache_size"] == 4
    assert receipt["acquire_stage_mode"] == "initial_topk"
    assert receipt["lost_prior_strategy"] == "full_global"
    assert receipt["lost_prior_fusion_weight"] == 3.0
    assert receipt["fused_coarse_mode"] == "fused"


def test_production_path_fails_closed_on_untransmitted_overrides(tmp_path):
    replay = load_script("benchmark_edm_site_replay_helpers", "benchmark_edm_site_replay.py")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mock")
    args = SimpleNamespace(
        stride=1, max_frames=0, max_corr_total=0, radius=None,
        site_profile=tmp_path / "site.json", video=video, require_cuda=False,
        worker_mode="production-path", mconf_thr=None, coarse_topk=None,
        cache_capacity=None, lost_strategy=None,
        lost_global_retrieval_interval=None, sigma_mode="fused",
        runtime_sigma_mode="bidirectional", temporal_feature_cache_size=None,
        acquire_stage_mode=None, lost_prior_strategy=None,
        lost_prior_fusion_weight=None,
    )
    with pytest.raises(SystemExit, match="production-path cannot apply runtime-sigma-mode"):
        replay._validate_run_inputs(args)
    args.runtime_sigma_mode = None
    profile = tmp_path / "edm_runtime_profile.json"
    profile.write_text(json.dumps({
        "matcher": {
            "mconf_thr": 0.2,
            "coarse_topk": 3225,
            "reference_cache_size": 32,
        },
        "tracker": {
            "radius": 0.8,
            "global_retrieval_policy": "boot_and_lost_once",
            "use_temporal_reference": False,
        },
    }), encoding="utf-8")
    site = SimpleNamespace(localizer_profile=profile)
    args.lost_prior_strategy = "score_fusion"
    with pytest.raises(SystemExit, match="production-path cannot apply lost-prior-strategy"):
        replay.runtime_stub_from_profile(site, args)



def _cache_override_stub_matcher():
    return SimpleNamespace(
        reference_feature_cache_size=192,
        host_reference_feature_cache_size=0,
        temporal_feature_cache_size=0,
        reference_cache_size=32,
        _feature_cache_capacity={"map": 192, "temporal": 0},
    )


def test_reference_feature_cache_override_validates_before_mutating():
    """Regression: an oversized --reference-feature-cache-size must fail closed
    BEFORE the matcher fields are mutated (SUMMARY.md s1_cache1045 finding)."""
    replay = load_script("benchmark_edm_site_replay_cache", "benchmark_edm_site_replay.py")
    matcher = _cache_override_stub_matcher()
    args = SimpleNamespace(
        reference_feature_cache_size=200_000,  # ~1.7 TiB of feature entries, far over the 8 GiB bound
        host_reference_feature_cache_size=None,
    )
    with pytest.raises(ValueError):
        replay._apply_reference_feature_cache_overrides(matcher, args)
    # the rejected override left the matcher untouched
    assert matcher.reference_feature_cache_size == 192
    assert matcher._feature_cache_capacity["map"] == 192


def test_reference_feature_cache_override_applies_within_budget():
    replay = load_script("benchmark_edm_site_replay_cache_ok", "benchmark_edm_site_replay.py")
    matcher = _cache_override_stub_matcher()
    args = SimpleNamespace(
        reference_feature_cache_size=100,  # ~0.9 GiB, within the 8 GiB bound
        host_reference_feature_cache_size=None,
    )
    replay._apply_reference_feature_cache_overrides(matcher, args)
    assert matcher.reference_feature_cache_size == 100
    assert matcher._feature_cache_capacity["map"] == 100


def test_host_reference_feature_cache_override_validates_before_mutating():
    replay = load_script("benchmark_edm_site_replay_host_cache", "benchmark_edm_site_replay.py")
    matcher = _cache_override_stub_matcher()
    args = SimpleNamespace(
        reference_feature_cache_size=None,
        host_reference_feature_cache_size=200_000,  # far over the 4 GiB pinned-host bound
    )
    with pytest.raises(ValueError):
        replay._apply_reference_feature_cache_overrides(matcher, args)
    assert matcher.host_reference_feature_cache_size == 0


def test_known_incomplete_uses_stream_frame_count_for_coalesced_worker():
    replay = load_script("benchmark_edm_site_replay_integrity", "benchmark_edm_site_replay.py")
    summary = {
        "frames": 2175,
        "successes": 1622,
        "state_counts": {"TRACK": 1622, "LOST": 295},
        "inliers": {"p50": 127.0, "p95": 677.9},
        "reproj_rms": {"p95": 3.15},
        "rejection_counts": {"limited_jump_unconfirmed": 0},
    }
    baseline = {
        "thresholds": {
            "frames": 2175,
            "min_successes": 1600,
            "min_track": 1600,
            "max_lost": 320,
            "min_inliers_p50": 120,
            "min_inliers_p95": 650,
            "max_reproj_rms_p95": 3.2,
            "max_limited_jump_unconfirmed": 0,
        },
        "stream_integrity": {"expected_decoded_frames": 3082},
    }
    audit = SimpleNamespace(
        reported_raw_frames=3083,
        decoded_raw_frames=3082,
        sampled_frames=3082,
        decode_errors=1,
    )
    failures, accepted = replay._evaluate_replay_quality(
        SimpleNamespace(
            quality_baseline=Path("baseline.json"),
            accept_known_incomplete=True,
        ),
        summary,
        baseline,
        audit,
        True,
    )
    assert failures == []
    assert accepted is True

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

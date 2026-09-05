"""The corpus summariser must separate a fresh match from a carried-forward pose."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

VALIDATION = Path(__file__).resolve().parents[1]


def _module():
    path = VALIDATION / "summarize_corpus_runs.py"
    spec = importlib.util.spec_from_file_location("summarize_corpus_runs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _module()


def _row(**kw):
    row = {"success": True, "inliers": 90, "candidate_mode": "edm_temporal_map"}
    row.update(kw)
    return row


def test_klt_bridged_frames_are_successes_but_not_verified() -> None:
    # The whole point of the metric: the async fast path publishes a pose whose
    # PnP re-fits the same map 3D that optical flow moved, so it is a success
    # with no new evidence from the map.
    assert mod.is_verified(_row()) is True
    assert mod.is_verified(_row(candidate_mode="klt_fast")) is False
    assert mod.is_verified(_row(candidate_mode="klt_bridge")) is False
    assert mod.is_verified(_row(candidate_mode=None)) is False


def test_zero_inlier_and_failed_frames_are_not_verified() -> None:
    assert mod.is_verified(_row(inliers=0)) is False
    assert mod.is_verified(_row(success=False)) is False


def test_summary_counts_verified_separately_from_successes() -> None:
    payload = {
        "summary": {"frames": 4, "successes": 3, "wall_ms": {"p50": 5.0, "p95": 9.0},
                    "state_counts": {"VISUALLY_CONFIRMED": 1, "KLT_BRIDGED": 2, "LOST": 1}},
        "rows": [
            _row(),
            _row(candidate_mode="klt_fast", inliers=310),
            _row(candidate_mode="klt_fast", inliers=280),
            _row(success=False, inliers=0, candidate_mode=None),
        ],
    }
    out = mod.summarize_run(payload)
    assert (out["frames"], out["successes"], out["verified"]) == (4, 3, 1)


def test_failure_bands_split_on_the_production_floors() -> None:
    rows = [
        _row(success=False, inliers=0),
        _row(success=False, inliers=12),
        _row(success=False, inliers=45),
        _row(success=False, inliers=60),
        _row(success=False, inliers=70),   # the near-miss band under acquire 80
        _row(success=False, inliers=120),  # cleared the floor, refused elsewhere
        _row(),                            # a success must not be counted
    ]
    counts = {key: count for key, count, _note in mod.failure_bands(rows)}
    assert counts == {"0-0": 1, "1-29": 1, "30-49": 1, "50-65": 1, "66-79": 1, "80-inf": 1}


def test_cli_reports_both_numbers(tmp_path, capsys) -> None:
    run = tmp_path / "sync"
    run.mkdir()
    (run / "P1680168.json").write_text(json.dumps({
        "summary": {"frames": 2, "successes": 2, "wall_ms": {"p50": 26.0, "p95": 80.0},
                    "state_counts": {"TRACK": 2}},
        "rows": [_row(), _row(candidate_mode="klt_fast")],
    }), encoding="utf-8")

    assert mod.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert "P1680168" in out
    assert "2/2" in out          # successes
    assert "50.0%" in out        # verified share


def test_missing_directory_fails_closed(tmp_path) -> None:
    with pytest.raises(SystemExit):
        mod.main([str(tmp_path / "nope")])

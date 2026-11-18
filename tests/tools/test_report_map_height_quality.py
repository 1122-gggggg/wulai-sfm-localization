"""Behaviour of the map height consistency report (pure classification + gates)."""

from types import SimpleNamespace

import pytest

import report_map_height_quality as report


def _climb(unit, *, follow, seconds=40, radius=0.03):
    """Barometer climbs 8 m (0.2 m/s); the map follows by `follow` of 0.95 u."""
    samples = []
    for k in range(int(seconds * 10) + 1):
        t = k / 10.0
        samples.append((t, follow * 0.95 * unit * t / seconds, 1.0 + 8.0 * t / seconds))
    return samples, radius * unit


@pytest.mark.parametrize("unit", [0.01, 1.0, 100.0])
def test_classification_is_unchanged_by_map_unit_size(unit):
    followed, radius = _climb(unit, follow=1.0)
    frozen, _ = _climb(unit, follow=0.0)
    opposed = [(t, -h, b) for t, h, b in followed]
    assert report.classify_windows(followed, radius)["consistent"] >= 3
    assert report.classify_windows(frozen, radius)["map_flat"] >= 3
    assert report.classify_windows(opposed, radius)["opposed"] >= 3


def test_vertical_unobservable_or_flat_map_is_inconsistent_not_a_pass():
    flat = {"consistent": 0, "opposed": 0, "map_flat": 4, "baro_flat": 0, "quiet": 0}
    assert report.verdict(flat, 0) == "HEIGHT_INCONSISTENT"
    good = {"consistent": 5, "opposed": 0, "map_flat": 0, "baro_flat": 0, "quiet": 2}
    assert report.verdict(good, 1) == "HEIGHT_INCONSISTENT"
    assert report.verdict(good, 0) == "CONSISTENT_NOT_VALIDATED"


def test_no_evidence_and_too_little_climb_are_never_a_pass():
    none = dict.fromkeys(("consistent", "opposed", "map_flat", "baro_flat", "quiet"), 0)
    assert report.verdict(none, 0) == "NO_FLIGHT_EVIDENCE"
    quiet = {**none, "quiet": 10}
    assert report.verdict(quiet, 0) == "INSUFFICIENT_VERTICAL_MOTION"
    few = {**none, "consistent": 1}
    assert report.verdict(few, 0) == "INSUFFICIENT_VERTICAL_MOTION"


def test_session_from_another_map_is_rejected_not_attributed_by_name():
    profile = SimpleNamespace(asset_sha256=SimpleNamespace(map_ply="a" * 64, map_align="b" * 64))
    manifest = {"asset_sha256": {"map_ply": "a" * 64, "map_align": "c" * 64}}
    assert report.attribution_errors(manifest, profile) == ["map_align SHA-256 does not match this map"]
    manifest["asset_sha256"]["map_align"] = "b" * 64
    assert report.attribution_errors(manifest, profile) == []


def test_only_an_unterminated_last_line_is_a_live_partial_write(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"a": 1}\n{"b": 2', encoding="utf-8")
    assert report.read_jsonl(path) == ([{"a": 1}], True)
    path.write_text('{"a": 1}\n{broken}\n{"c": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="t.jsonl:2"):
        report.read_jsonl(path)

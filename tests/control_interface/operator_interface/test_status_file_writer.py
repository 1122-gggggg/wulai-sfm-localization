"""The background status-file writer keeps the old file contents and failure reports."""

from __future__ import annotations

from pathlib import Path
import sys

HERE = (
    Path(__file__).resolve().parents[3]
    / "控制介面程式"
    / "operator_interface"
)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from status_file_writer import StatusFileWriter  # noqa: E402


def test_latest_text_wins_and_unchanged_text_is_not_rewritten(tmp_path) -> None:
    writer = StatusFileWriter()
    target = tmp_path / "status.json"
    for text in ('{"n": 1}', '{"n": 2}', '{"n": 3}'):
        assert writer.submit(target, text) == []
    assert writer.flush()
    assert target.read_text(encoding="utf-8") == '{"n": 3}'

    target.unlink()
    assert writer.submit(target, '{"n": 3}') == []
    assert writer.flush()
    assert not target.exists()

    assert writer.submit(target, '{"n": 4}') == []
    assert writer.flush()
    assert target.read_text(encoding="utf-8") == '{"n": 4}'


def test_each_path_keeps_its_own_latest_text(tmp_path) -> None:
    writer = StatusFileWriter()
    live, detector = tmp_path / "live.json", tmp_path / "detector.json"
    writer.submit(live, "live")
    writer.submit(detector, "detector")
    assert writer.flush()
    assert (live.read_text(encoding="utf-8"), detector.read_text(encoding="utf-8")) == (
        "live",
        "detector",
    )


def test_write_failure_is_reported_once_to_the_next_caller_and_retried(tmp_path) -> None:
    writer = StatusFileWriter()
    target = tmp_path / "missing_dir" / "status.json"
    assert writer.submit(target, "payload") == []
    assert writer.flush()

    target.parent.mkdir()
    failures = writer.submit(target, "payload")  # same text: the failed write is retried
    assert [path for path, _exc in failures] == [target]
    assert isinstance(failures[0][1], OSError)
    assert writer.flush()
    assert target.read_text(encoding="utf-8") == "payload"
    assert writer.submit(target, "payload") == []

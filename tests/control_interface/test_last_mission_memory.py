from __future__ import annotations

import json

import pytest

import last_mission_memory as memory
from mission_manifest import ManifestError


@pytest.fixture
def selection(tmp_path):
    path = tmp_path / "map_a.json"
    path.write_text("{}", encoding="utf-8")
    return path


def test_remembered_selection_round_trips_when_it_still_resolves(tmp_path, selection, monkeypatch):
    monkeypatch.setattr(memory, "resolve_mission", lambda *_a, **_k: object())
    record = tmp_path / "last.json"
    memory.remember_mission_selection(selection, record)
    assert memory.load_remembered_selection(record) == selection.resolve()


def test_selection_that_no_longer_validates_is_not_reused(tmp_path, selection, monkeypatch):
    record = tmp_path / "last.json"
    memory.remember_mission_selection(selection, record)

    def reject(*_a, **_k):
        raise ManifestError("assets changed")

    monkeypatch.setattr(memory, "resolve_mission", reject)
    assert memory.load_remembered_selection(record) is None


def test_missing_or_deleted_or_corrupt_memory_yields_default(tmp_path, selection, monkeypatch):
    monkeypatch.setattr(memory, "resolve_mission", lambda *_a, **_k: object())
    record = tmp_path / "last.json"
    assert memory.load_remembered_selection(record) is None
    memory.remember_mission_selection(selection, record)
    selection.unlink()
    assert memory.load_remembered_selection(record) is None
    record.write_text("not json", encoding="utf-8")
    assert memory.load_remembered_selection(record) is None
    record.write_text(json.dumps({"schema": "other", "mission_selection": "x"}), encoding="utf-8")
    assert memory.load_remembered_selection(record) is None


def test_empty_selection_keeps_previous_memory(tmp_path, selection):
    record = tmp_path / "last.json"
    memory.remember_mission_selection(selection, record)
    memory.remember_mission_selection("", record)
    assert json.loads(record.read_text(encoding="utf-8"))["mission_selection"] == str(
        selection.resolve()
    )

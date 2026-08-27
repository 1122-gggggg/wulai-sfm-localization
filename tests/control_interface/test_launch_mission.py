from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "控制介面程式"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

SPEC = importlib.util.spec_from_file_location(
    "launch_mission_under_test",
    CONTROL / "launch_mission.py",
)
assert SPEC is not None and SPEC.loader is not None
launch_mission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch_mission)


def test_check_only_is_parsed_after_selection_without_reaching_operator() -> None:
    args, operator_args = launch_mission._parser().parse_known_args(
        ["selection.json", "--check-only"]
    )

    assert args.selection == "selection.json"
    assert args.check_only is True
    assert operator_args == []


def test_operator_arguments_remain_explicit_passthrough() -> None:
    args, operator_args = launch_mission._parser().parse_known_args(
        ["selection.json", "--", "--video", "replay.mp4"]
    )

    assert args.selection == "selection.json"
    assert args.check_only is False
    assert operator_args == ["--video", "replay.mp4"]


@pytest.mark.parametrize("argument", ["--site-profile", "--site-profile=old.json"])
def test_launcher_rejects_direct_site_profile_arguments(
    argument: str, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("SFM_SITE_PROFILE", raising=False)

    assert launch_mission.main(["selection with spaces.json", argument]) == 2

    assert "SFM_MISSION_SELECTION" in capsys.readouterr().err


def test_launcher_rejects_existing_site_profile_environment(monkeypatch, capsys) -> None:
    monkeypatch.setenv("SFM_SITE_PROFILE", "/old/site profile.json")

    assert launch_mission.main(["selection.json", "--check-only"]) == 2

    assert "SFM_SITE_PROFILE" in capsys.readouterr().err


def test_launcher_rejects_not_localization_ready_selection(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    selection = tmp_path / "selection with spaces.json"
    mission = SimpleNamespace(
        readiness=SimpleNamespace(
            localization_ready=False,
            localization_errors=("localizer map revision does not match",),
        )
    )
    monkeypatch.delenv("SFM_SITE_PROFILE", raising=False)
    monkeypatch.setattr(launch_mission, "resolve_mission", lambda *_args, **_kwargs: mission)

    assert launch_mission.main([str(selection), "--check-only"]) == 2

    captured = capsys.readouterr()
    assert "not localization-ready" in captured.err
    assert "localizer map revision" in captured.err

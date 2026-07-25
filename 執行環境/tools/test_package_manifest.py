from pathlib import Path

from tools.package_manifest import included


def test_generated_and_tool_state_are_excluded_without_hiding_mission_inputs():
    assert not included(Path(".codegraph/codegraph.db"))
    assert not included(Path(".cursor/rules/codegraph.mdc"))
    assert not included(Path(".venv/lib/python/site.py"))
    assert not included(Path("sfm_system/定位/outputs/flight_logs/session.jsonl"))
    assert not included(Path(
        "sfm_system/定位/experiments/sphinx_anafi_path_convergence/outputs/run/ticks.jsonl"
    ))

    assert included(Path(
        "sfm_system/定位/mission/outputs/current_safezone/flight_path.json"
    ))

import importlib.util
from pathlib import Path
import sys


ROOT_MANIFEST = Path(__file__).resolve().parents[2] / "tools" / "package_manifest.py"
SPEC = importlib.util.spec_from_file_location("_root_package_manifest", ROOT_MANIFEST)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
included = MODULE.included


def test_generated_and_tool_state_are_excluded_by_root_authoritative_manifest():
    assert not included(Path(".codegraph/codegraph.db"))
    assert not included(Path(".cursor/rules/codegraph.mdc"))
    assert not included(Path(".venv/lib/python/site.py"))
    assert not included(Path("sfm_system/定位/outputs/flight_logs/session.jsonl"))
    assert not included(Path(
        "sfm_system/定位/experiments/sphinx_anafi_path_convergence/outputs/run/ticks.jsonl"
    ))
    assert not included(Path(
        "sfm_system/定位/mission/outputs/current_safezone/flight_path.json"
    ))
    assert included(Path("定位演算法/validation/check_runtime_mirrors.py"))

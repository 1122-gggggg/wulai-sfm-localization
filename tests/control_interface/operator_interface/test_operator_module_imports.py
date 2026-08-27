from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


OPERATOR_INTERFACE = (
    Path(__file__).resolve().parents[3]
    / "控制介面程式"
    / "operator_interface"
)
PUBLIC_LAUNCH_SYMBOLS = (
    "OperatorSessionIdentity",
    "build_argument_parser",
    "main",
)


def _run_clean_import(scenario: str) -> subprocess.CompletedProcess[str]:
    child = textwrap.dedent(
        f"""
        import importlib
        import importlib.util
        import json
        import sys
        from pathlib import Path

        interface = Path({str(OPERATOR_INTERFACE)!r})
        app_path = interface / "flight_operator_app.py"
        scenario = {scenario!r}
        expected = {PUBLIC_LAUNCH_SYMBOLS!r}
        sys.path.insert(0, str(interface))

        if scenario == "direct":
            module = importlib.import_module("flight_operator_app")
        elif scenario == "arbitrary":
            spec = importlib.util.spec_from_file_location(
                "operator_app_arbitrary_name", app_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("could not create arbitrary-name import spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        elif scenario == "operator_launch":
            module = importlib.import_module("operator_launch")
        else:
            raise AssertionError(f"unknown import scenario: {{scenario!r}}")

        missing = [name for name in expected if not hasattr(module, name)]
        if missing:
            raise AssertionError(
                f"{{scenario}} missing public launch symbols: {{missing}}"
            )
        if not isinstance(module.OperatorSessionIdentity, type):
            raise AssertionError("OperatorSessionIdentity is not a class")
        if not callable(module.build_argument_parser) or not callable(module.main):
            raise AssertionError("launch functions are not callable")
        print(
            "SFM_OPERATOR_IMPORT_RESULT:"
            + json.dumps(
                {{"module": module.__name__, "symbols": list(expected)}},
                sort_keys=True,
            )
        )
        """
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", child],
        cwd=OPERATOR_INTERFACE,
        env={},
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("scenario", ("direct", "arbitrary", "operator_launch"))
def test_operator_launch_symbols_import_in_a_clean_process(scenario: str) -> None:
    result = _run_clean_import(scenario)

    assert result.returncode == 0, (
        f"{scenario} import failed with exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    result_line = next(
        (
            line
            for line in result.stdout.splitlines()
            if line.startswith("SFM_OPERATOR_IMPORT_RESULT:")
        ),
        None,
    )
    assert result_line is not None, result.stdout
    payload = json.loads(result_line.split(":", 1)[1])
    assert set(payload["symbols"]) == set(PUBLIC_LAUNCH_SYMBOLS)

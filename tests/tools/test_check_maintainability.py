from __future__ import annotations

from pathlib import Path

import pytest

from check_maintainability import (
    BUDGETS,
    budget_failures,
    line_budget_failures,
    summarize_findings,
)


def _finding(path: str, complexity: int) -> dict[str, object]:
    return {
        "filename": str(Path(__file__).resolve().parents[2] / path),
        "message": f"`example` is too complex ({complexity} > 10)",
    }


def test_summary_groups_production_findings_and_ignores_tests() -> None:
    summary = summarize_findings(
        [
            _finding("tools/simulator_preflight.py", 20),
            _finding("控制介面程式/operator_interface/flight_operator_app.py", 72),
            _finding("控制介面程式/operator_interface/test_example.py", 99),
        ]
    )

    assert summary["tools"] == {
        "violations": 1,
        "max_complexity": 20,
        "worst": "tools/simulator_preflight.py",
    }
    assert summary["control"]["violations"] == 1
    assert summary["control"]["max_complexity"] == 72


def test_budget_rejects_regressions() -> None:
    summary = {
        group: {
            "violations": budget["violations"],
            "max_complexity": budget["max_complexity"],
            "worst": "example.py",
        }
        for group, budget in BUDGETS.items()
    }
    summary["tools"]["violations"] = BUDGETS["tools"]["violations"] + 1
    summary["flight"]["max_complexity"] = BUDGETS["flight"]["max_complexity"] + 1

    assert budget_failures(summary) == [
        "tools violations increased: 1 > 0",
        "flight max_complexity increased: 13 > 12",
    ]


def test_line_budget_rejects_growth_and_missing_files(tmp_path: Path) -> None:
    source = tmp_path / "large.py"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")

    assert line_budget_failures(tmp_path, {"large.py": 2, "missing.py": 10}) == [
        "large.py lines increased: 3 > 2",
        "line-budget file missing: missing.py",
    ]


def test_help_exits_before_running_the_scan(monkeypatch, capsys) -> None:
    import check_maintainability

    def fail_if_scanned() -> list[dict[str, object]]:
        pytest.fail("--help must not run the Ruff scan")

    monkeypatch.setattr(check_maintainability, "collect_ruff_findings", fail_if_scanned)

    with pytest.raises(SystemExit) as error:
        check_maintainability.main(["--help"])

    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out

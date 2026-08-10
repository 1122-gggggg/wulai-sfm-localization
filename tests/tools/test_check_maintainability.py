from __future__ import annotations

from pathlib import Path

from check_maintainability import BUDGETS, budget_failures, summarize_findings


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


def test_zero_budget_rejects_any_violation() -> None:
    summary = {
        group: {
            "violations": budget["violations"],
            "max_complexity": budget["max_complexity"],
            "worst": "example.py",
        }
        for group, budget in BUDGETS.items()
    }
    summary["tools"]["violations"] = 1
    summary["flight"]["max_complexity"] = 11

    assert budget_failures(summary) == [
        "tools violations increased: 1 > 0",
        "flight max_complexity increased: 11 > 0",
    ]

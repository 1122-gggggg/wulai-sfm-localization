#!/usr/bin/env python3
"""Ratchet existing cyclomatic-complexity debt without rewriting flight logic.

The project contains mature, highly tested control and localization state
machines whose Ruff C901 complexity cannot be removed safely in one change.
This gate records the current debt by subsystem: later changes may reduce it,
but may not add violations or increase the worst function in any subsystem.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_PATHS = (
    "tools",
    "定位演算法/deploy_code/sfm_glomap_deploy",
    "定位演算法/flight_control",
    "定位演算法/validation",
    "控制介面程式",
)

# Measured with Ruff 0.16.1 on 2026-08-08 after the maintenance refactor.
# Values are ceilings, not targets. Reductions do not require this table to be
# updated immediately; any increase fails CI.
BUDGETS = {
    "tools": {"violations": 4, "max_complexity": 20},
    "deploy": {"violations": 13, "max_complexity": 49},
    "flight": {"violations": 21, "max_complexity": 86},
    "validation": {"violations": 13, "max_complexity": 39},
    "control": {"violations": 60, "max_complexity": 72},
}


def _relative_path(filename: str) -> Path:
    path = Path(filename)
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def _group_for(path: Path) -> str | None:
    text = path.as_posix()
    if text.startswith("tools/"):
        return "tools"
    if text.startswith("定位演算法/deploy_code/"):
        return "deploy"
    if text.startswith("定位演算法/flight_control/"):
        return "flight"
    if text.startswith("定位演算法/validation/"):
        return "validation"
    if text.startswith("控制介面程式/"):
        return "control"
    return None


def _is_test(path: Path) -> bool:
    return "tests" in path.parts or path.name.startswith("test_")


def _complexity(message: str) -> int:
    marker = " is too complex ("
    if marker not in message:
        raise ValueError(f"unexpected Ruff C901 message: {message!r}")
    value = message.split(marker, 1)[1].split(" > ", 1)[0]
    return int(value)


def summarize_findings(findings: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    summary = {group: {"violations": 0, "max_complexity": 0, "worst": ""} for group in BUDGETS}
    for finding in findings:
        path = _relative_path(str(finding["filename"]))
        if _is_test(path):
            continue
        group = _group_for(path)
        if group is None:
            continue
        complexity = _complexity(str(finding["message"]))
        group_summary = summary[group]
        group_summary["violations"] = int(group_summary["violations"]) + 1
        if complexity > int(group_summary["max_complexity"]):
            group_summary["max_complexity"] = complexity
            group_summary["worst"] = path.as_posix()
    return summary


def budget_failures(summary: dict[str, dict[str, object]]) -> list[str]:
    failures: list[str] = []
    for group, budget in BUDGETS.items():
        actual = summary[group]
        for metric in ("violations", "max_complexity"):
            if int(actual[metric]) > budget[metric]:
                failures.append(f"{group} {metric} increased: {actual[metric]} > {budget[metric]}")
    return failures


def collect_ruff_findings() -> list[dict[str, object]]:
    command = (
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        "C901",
        "--output-format",
        "json",
        *SCAN_PATHS,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "Ruff complexity scan failed")
    try:
        findings = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ruff did not return valid JSON") from exc
    if not isinstance(findings, list):
        raise RuntimeError("Ruff C901 output must be a JSON list")
    return findings


def main() -> int:
    summary = summarize_findings(collect_ruff_findings())
    failures = budget_failures(summary)
    for group, values in summary.items():
        print(
            f"[maintainability] {group}: violations={values['violations']} "
            f"max={values['max_complexity']} worst={values['worst'] or '-'}"
        )
    if failures:
        for failure in failures:
            print(f"[maintainability] FAIL: {failure}")
        return 1
    print("[maintainability] OK: complexity debt did not increase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

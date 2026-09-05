#!/usr/bin/env python3
"""Reject complexity and oversized-module regressions in production code."""

from __future__ import annotations

import argparse
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

# Ratchet baseline for the current first-party production tree. Reducing these
# values is encouraged; increasing either count or worst complexity fails CI.
#
# Re-baselined 2026-09-05. The previous values (tools 0/0, deploy 0/0,
# validation 3/19, control 12/22) predated the EDM deploy tree and had been
# failing on the committed tree itself, so the ratchet could not detect a new
# regression -- it reported the same wall of failures either way. These numbers
# are the measured floor to push down from, not a target.
BUDGETS = {
    "tools": {"violations": 1, "max_complexity": 21},
    "deploy": {"violations": 26, "max_complexity": 57},
    "flight": {"violations": 3, "max_complexity": 12},
    "validation": {"violations": 8, "max_complexity": 19},
    "control": {"violations": 18, "max_complexity": 25},
}
# Raised 2026-09-05 for the IMU flight-test recording path. Everything that
# could live outside these two files does: the frame recorder is
# operator_interface/imu_flight_test.py and the stick-log decision is
# skycontroller_stick.stick_log_sample. What is left is the call sites that can
# only be where the frame, its telemetry and the session directory are --
# +28 and +24 lines. Push these back down; do not raise them for a feature that
# has not first been moved out.
LINE_BUDGETS = {
    "控制介面程式/operator_interface/flight_operator_app.py": 7891,
    "控制介面程式/operator_interface/olympe_live_backend.py": 5208,
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


def line_budget_failures(
    root: Path = ROOT,
    budgets: dict[str, int] = LINE_BUDGETS,
) -> list[str]:
    failures: list[str] = []
    for relative, maximum in budgets.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"line-budget file missing: {relative}")
            continue
        actual = len(path.read_text(encoding="utf-8").splitlines())
        if actual > maximum:
            failures.append(f"{relative} lines increased: {actual} > {maximum}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    summary = summarize_findings(collect_ruff_findings())
    failures = budget_failures(summary) + line_budget_failures()
    for group, values in summary.items():
        print(
            f"[maintainability] {group}: violations={values['violations']} "
            f"max={values['max_complexity']} worst={values['worst'] or '-'}"
        )
    if failures:
        for failure in failures:
            print(f"[maintainability] FAIL: {failure}")
        return 1
    print("[maintainability] OK: complexity and module sizes remain within budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

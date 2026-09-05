"""Acceptance tests for tools/test_route_autoflight.py (drawn-route dry-run gate)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


TOOL = Path(__file__).resolve().parents[3] / "tools" / "test_route_autoflight.py"

DRAWN_ROUTE = {
    "waypoints": [[0, 0, 4.5], [10, 0, 4.5], [10, 8, 4.5], [0, 8, 4.5]],
    "closed": True,
    "frame": "aligned",
    "units": "map",
    "source": "blender_draw_path",
    "default_alt": 4.5,
}


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_help_exits_zero() -> None:
    proc = _run("--help")
    assert proc.returncode == 0
    assert "--route" in proc.stdout


def test_missing_route_argument_exits_one() -> None:
    proc = _run()
    assert proc.returncode == 1


def test_nonexistent_route_reports_error() -> None:
    proc = _run("--route", "/tmp/route_autoflight_nope.json")
    assert proc.returncode == 1
    assert "[route-test] error=" in proc.stdout


def test_single_waypoint_route_rejected(tmp_path: Path) -> None:
    route = tmp_path / "bad.json"
    route.write_text(json.dumps({"waypoints": [[0, 0, 0]]}), encoding="utf-8")
    proc = _run("--route", str(route))
    assert proc.returncode == 1
    assert "[route-test] error=" in proc.stdout


def test_drawn_format_route_passes_dry_run(tmp_path: Path) -> None:
    route = tmp_path / "drawn.json"
    route.write_text(json.dumps(DRAWN_ROUTE), encoding="utf-8")
    log = tmp_path / "cmd.jsonl"
    proc = _run("--route", str(route), "--cmd-log", str(log))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[route-test] waypoints=4" in proc.stdout
    assert "closed=true" in proc.stdout
    assert "state=LANDING" in proc.stdout
    assert log.exists()


def test_impossible_progress_threshold_fails(tmp_path: Path) -> None:
    route = tmp_path / "drawn.json"
    route.write_text(json.dumps(DRAWN_ROUTE), encoding="utf-8")
    proc = _run("--route", str(route), "--min-progress", "1.0")
    assert proc.returncode == 2

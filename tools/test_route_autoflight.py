#!/usr/bin/env python3
"""Route autoflight dry-run acceptance tool.

Validates an authored flight route and executes a dry-run closed-loop flight
using RouteAutoController and kinematic toy-dynamics without hardware.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import TextIO

# Prohibit importing olympe strictly across the entire execution
sys.modules["olympe"] = None
sys.modules["olympe.messages"] = None
sys.modules["olympe.messages.ardrone3"] = None
sys.modules["olympe.messages.ardrone3.Piloting"] = None

REPO_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_CONTROL_ROOT = REPO_ROOT / "定位演算法" / "flight_control"
if str(FLIGHT_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(FLIGHT_CONTROL_ROOT))

# Ensure ambient flight contract environment does not enforce SHA-256 verification
os.environ.pop("SFM_FLIGHT_CONTRACT_JSON", None)

from route_domain import RouteDocument, LEGACY_MAP_FRAME  # noqa: E402
from real_path_follow_controller import load_map_frame  # noqa: E402
import path_follow_flight as pff  # noqa: E402


def _site_map_align(route_path: Path, explicit_profile: str) -> Path | None:
    """Locate the T_align_gravity.json that belongs to the route being tested.

    The route editor stamps ``align_source='measured'`` on every route authored
    at a site that has one, and such a route cannot be converted back to
    controller coordinates without that same frame.  Resolving it here is what
    lets this tool accept the routes the editor actually writes; without it the
    legacy [x,z,-y] assumption is the only frame on offer and every measured
    route is rejected.  Returns None when no site profile is in play, which is
    what a bare hand-written route wants.
    """
    if explicit_profile:
        profile_path = Path(explicit_profile).expanduser()
    else:
        # Routes live at <site>/routes/<name>.json; the profile sits at <site>/.
        profile_path = route_path.resolve().parent.parent / "site_profile.json"
    if not profile_path.is_file():
        return None
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    align = raw.get("map_align")
    if not align:
        return None
    return (profile_path.parent / align).resolve()


class RouteTestArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that fails with exit code 1 instead of 2 on usage errors."""

    def error(self, message: str) -> None:
        sys.stderr.write(f"error: {message}\n")
        print(f"[route-test] error={message}")
        sys.exit(1)


class _StdoutFilter:
    """Filter that only forwards lines starting with [dry] or [route-test] to the real stdout."""

    def __init__(self, target: TextIO) -> None:
        self.target = target
        self.buffer = ""
        self.captured_dry_lines: list[str] = []

    def write(self, s: str) -> int:
        self.buffer += s
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._process_line(line)
        return len(s)

    def _process_line(self, line: str) -> None:
        if line.startswith("[dry]"):
            self.captured_dry_lines.append(line)
            self.target.write(line + "\n")
            self.target.flush()
        elif line.startswith("[route-test]"):
            self.target.write(line + "\n")
            self.target.flush()

    def flush(self) -> None:
        if self.buffer:
            line = self.buffer
            self.buffer = ""
            self._process_line(line)
        self.target.flush()


@contextlib.contextmanager
def _filter_stdout():
    real_stdout = sys.stdout
    filtr = _StdoutFilter(real_stdout)
    sys.stdout = filtr
    try:
        yield filtr
    finally:
        filtr.flush()
        sys.stdout = real_stdout


def _build_parser() -> argparse.ArgumentParser:
    parser = RouteTestArgumentParser(
        description="Route autoflight dry-run acceptance tool."
    )
    parser.add_argument(
        "--route",
        required=True,
        type=str,
        help="Path to route JSON (e.g. flight_path.json)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12000,
        help="Max simulation steps (default: 12000)",
    )
    parser.add_argument(
        "--min-progress",
        type=float,
        default=0.95,
        help="Minimum progress fraction to pass [0.0..1.0] (default: 0.95)",
    )
    parser.add_argument(
        "--cmd-log",
        type=str,
        default="",
        help="Optional path to output command log JSONL",
    )
    parser.add_argument(
        "--yaw-sign",
        type=int,
        default=1,
        help="Yaw sign convention (default: 1)",
    )
    parser.add_argument(
        "--site-profile",
        type=str,
        default="",
        help=(
            "site_profile.json supplying the measured gravity alignment "
            "(default: <route>/../../site_profile.json when present)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.steps <= 0:
        parser.error("--steps must be a positive integer")
    if not (0.0 <= args.min_progress <= 1.0):
        parser.error("--min-progress must be between 0.0 and 1.0")

    route_path = Path(args.route).expanduser()

    # 1. RouteDocument.from_path 驗證
    try:
        map_align = _site_map_align(route_path, args.site_profile)
        map_frame = LEGACY_MAP_FRAME if map_align is None else load_map_frame(map_align)
        doc = RouteDocument.from_path(
            route_path, require_map_units=True, map_frame=map_frame
        )
    except Exception as exc:
        print(f"[route-test] error={exc}")
        return 1
    print(f"[route-test] map_frame={map_frame.source}")

    # Compute route properties: waypoints count, total length, closed, height range
    wps = doc.waypoints
    num_wps = len(wps)
    closed = bool(doc.closed)
    calc_pts = list(wps)
    if closed and calc_pts[-1] != calc_pts[0]:
        calc_pts.append(calc_pts[0])
    total_len = sum(math.dist(a, b) for a, b in zip(calc_pts[:-1], calc_pts[1:]))
    zs = [p[2] for p in wps]
    z_min, z_max = min(zs), max(zs)

    # Print route verification line
    print(
        f"[route-test] waypoints={num_wps} length={total_len:.2f} total_length={total_len:.2f} "
        f"closed={str(closed).lower()} alt_min={z_min:.2f} alt_max={z_max:.2f} "
        f"z_min={z_min:.2f} z_max={z_max:.2f} alt_range=[{z_min:.2f},{z_max:.2f}]"
    )

    # 2. 覆寫 path_follow_flight.PATH_JSON 後調 dry_run
    resolved_route = str(route_path.resolve())
    pff.PATH_JSON = resolved_route
    os.environ["SFM_FLIGHT_PATH_JSON"] = resolved_route
    # build_controller reloads the route itself, so it needs the same frame the
    # validation above used. Left unset it falls back to the legacy assumption
    # and rejects the very route that just parsed.
    resolved_align = "" if map_align is None else str(map_align)
    pff.MAP_ALIGN = resolved_align
    os.environ["SFM_MAP_ALIGN"] = resolved_align
    # Ensure contract check is disabled for dry_run (no SHA chain verification)
    pff.FLIGHT_CONTRACT_JSON = ""
    os.environ.pop("SFM_FLIGHT_CONTRACT_JSON", None)

    cmd_log_target = str(Path(args.cmd_log).expanduser().resolve()) if args.cmd_log else ""

    captured_dry: list[str] = []
    try:
        with _filter_stdout() as filtr:
            state, progress = pff.dry_run(
                args.yaw_sign,
                steps=args.steps,
                cmd_log_path=cmd_log_target,
            )
            captured_dry = filtr.captured_dry_lines
    except Exception as exc:
        print(f"[route-test] error={exc}")
        return 1

    # 3. 印 [route-test] end/reason/progress/state/cmdlog 行
    reason = "unknown"
    cmdlog = cmd_log_target
    for line in captured_dry:
        if "end=" in line:
            m_end = re.search(r"end=(.*?)(?:\s+progress=|\s*$)", line)
            if m_end:
                reason = m_end.group(1).strip()
        if "structured command log:" in line:
            m_log = re.search(r"structured command log:\s*(\S+)", line)
            if m_log:
                cmdlog = m_log.group(1).strip()

    if (reason == "unknown" or not cmdlog) and cmdlog and Path(cmdlog).is_file():
        try:
            with open(cmdlog, "r", encoding="utf-8") as f:
                last_line = ""
                for line in f:
                    if line.strip():
                        last_line = line
                if last_line:
                    data = json.loads(last_line)
                    if "reason" in data and reason == "unknown":
                        reason = str(data["reason"])
        except Exception:
            pass

    end = reason
    print(
        f"[route-test] end={end} reason={reason} progress={progress:.4f} "
        f"state={state} cmdlog={cmdlog}"
    )

    # 4. progress>=min-progress 則 exit 0 否則 exit 2
    if progress >= args.min_progress:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fault-injection measurement for the pose vetoes, off recorded replay rows.

A veto that never fires on a clean corpus is unmeasured, not proven. This
script takes the accepted-pose sequence from a real replay run
(``benchmark_edm_site_replay.py`` JSON), injects the failure modes the vetoes
exist to catch, and reports detection rate against false-kill rate on the
untouched rows.

Two independent vetoes are evaluated:

* ``trajectory_plausibility.vote_allow`` -- windowed median voter over
  ``(center, yaw, stamp)``.
* ``gravity_roll_gate.vote_allow`` -- measured gimbal-roll invariant, needs a
  rotation, so it is only evaluated for the rotation-flip injections.

Injections (all bounded so the existing gates would *not* catch them, which is
the point -- a fault big enough to trip ``max_jump`` needs no new voter):

* ``teleport``: displace the center by a fraction of ``max_jump`` below the
  single-step limit.
* ``yaw_flip``: rotate yaw by a large angle within one frame interval.
* ``roll``: tilt the camera right axis out of horizontal (gravity veto only).

Example::

    定位演算法/validation/eval_pose_veto_injection.py \\
        --run /tmp/gate_r3/P1190119_720p_t0_baseline.json \\
        --max-jump 0.9338766098022462 --rate 0.01 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


VALIDATION_ROOT = Path(__file__).resolve().parent
DEPLOY_ROOT = VALIDATION_ROOT.parent / "deploy_code" / "sfm_glomap_deploy"
for candidate in (VALIDATION_ROOT, DEPLOY_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from gravity_roll_gate import vote_allow as gravity_vote  # noqa: E402
from trajectory_plausibility import TrajectoryWindow  # noqa: E402
from trajectory_plausibility import vote_allow as trajectory_vote  # noqa: E402


def accepted_track(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accepted frames carrying a pose and a capture stamp, in order."""
    out = []
    for row in rows:
        if not row.get("success"):
            continue
        center = row.get("pose_xyz") or row.get("center")
        stamp = row.get("capture_stamp")
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            continue
        if not isinstance(stamp, (int, float)):
            continue
        point = {"center": [float(v) for v in center], "stamp": float(stamp)}
        yaw = row.get("pose_yaw")
        if isinstance(yaw, (int, float)) and math.isfinite(float(yaw)):
            point["yaw"] = float(yaw)
        out.append(point)
    return out


def derive_yaw(track: list[dict[str, Any]]) -> None:
    """Fill in any missing yaw from the direction of travel.

    Replay rows written before ``pose_yaw`` existed carry no heading. The
    travel-direction stand-in is far noisier than real yaw (it spins freely
    while hovering), so a run that needs this fallback reports an upper bound
    on false kills, never a clean number. Runs with ``pose_yaw`` skip it.
    """
    for index, point in enumerate(track):
        if "yaw" in point:
            continue
        if index == 0:
            point["yaw"] = 0.0
            continue
        prev = track[index - 1]["center"]
        cur = point["center"]
        dx, dy = cur[0] - prev[0], cur[1] - prev[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            point["yaw"] = track[index - 1].get("yaw", 0.0)
        else:
            point["yaw"] = math.atan2(dy, dx)


def _level_rotation(gravity: list[float], yaw: float, roll_deg: float) -> np.ndarray:
    g = np.asarray(gravity, dtype=float)
    g /= np.linalg.norm(g)
    seed = np.asarray([1.0, 0.0, 0.0])
    e0 = seed - (seed @ g) * g
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(g, e0)
    right = math.cos(yaw) * e0 + math.sin(yaw) * e1
    roll = math.radians(roll_deg)
    right = math.cos(roll) * right + math.sin(roll) * g
    right /= np.linalg.norm(right)
    down = g - (g @ right) * right
    down /= np.linalg.norm(down)
    forward = np.cross(right, down)
    forward /= np.linalg.norm(forward)
    return np.asarray([right, down, forward], dtype=float)


def evaluate(
    track: list[dict[str, Any]],
    *,
    max_jump: float,
    rate: float,
    seed: int,
    gravity: list[float],
    teleport_fraction: float,
    yaw_flip_deg: float,
    roll_deg: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    window = TrajectoryWindow(max_points=8, max_age_s=3.0)
    stats = {
        "frames": len(track),
        "injected_teleport": 0,
        "caught_teleport": 0,
        "injected_yaw": 0,
        "caught_yaw": 0,
        "injected_roll": 0,
        "caught_roll": 0,
        "clean_frames": 0,
        "clean_trajectory_vetoes": 0,
        "clean_gravity_vetoes": 0,
        "veto_reasons": {},
    }
    for point in track:
        center = list(point["center"])
        yaw = float(point["yaw"])
        stamp = float(point["stamp"])
        kind = "clean"
        if rng.random() < rate:
            kind = rng.choice(("teleport", "yaw_flip", "roll"))

        if kind == "teleport":
            direction = [rng.gauss(0.0, 1.0) for _ in range(3)]
            norm = math.sqrt(sum(v * v for v in direction)) or 1.0
            offset = teleport_fraction * max_jump
            probe_center = [c + offset * d / norm for c, d in zip(center, direction)]
            probe_yaw = yaw
        elif kind == "yaw_flip":
            probe_center = center
            probe_yaw = yaw + math.radians(yaw_flip_deg)
        else:
            probe_center = center
            probe_yaw = yaw

        allow, reason = trajectory_vote(
            window, probe_center, probe_yaw, stamp, max_jump=max_jump
        )
        rotation = _level_rotation(gravity, probe_yaw, roll_deg if kind == "roll" else 0.0)
        g_allow, g_reason, _ = gravity_vote(rotation, gravity)

        if kind == "teleport":
            stats["injected_teleport"] += 1
            stats["caught_teleport"] += int(not allow)
        elif kind == "yaw_flip":
            stats["injected_yaw"] += 1
            stats["caught_yaw"] += int(not allow)
        elif kind == "roll":
            stats["injected_roll"] += 1
            stats["caught_roll"] += int(not g_allow)
        else:
            stats["clean_frames"] += 1
            stats["clean_trajectory_vetoes"] += int(not allow)
            stats["clean_gravity_vetoes"] += int(not g_allow)

        if not allow:
            stats["veto_reasons"][reason] = stats["veto_reasons"].get(reason, 0) + 1
        if not g_allow:
            stats["veto_reasons"][g_reason] = stats["veto_reasons"].get(g_reason, 0) + 1

        # Only genuine (uninjected) poses enter the history, mirroring the
        # tracker: a vetoed candidate never becomes the new reference.
        if kind == "clean" and allow:
            window.append(center, yaw, stamp)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, nargs="+", required=True, help="replay JSON")
    parser.add_argument("--max-jump", type=float, required=True)
    parser.add_argument("--rate", type=float, default=0.01, help="injection probability")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--teleport-fraction", type=float, default=0.9,
                        help="injected step as a fraction of max_jump (<1 so the "
                             "existing single-step gate would still accept it)")
    parser.add_argument("--yaw-flip-deg", type=float, default=90.0)
    parser.add_argument("--roll-deg", type=float, default=25.0)
    parser.add_argument(
        "--gravity",
        type=Path,
        default=None,
        help="T_align_gravity.json; default uses the river map value",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.gravity is not None:
        from gravity_roll_gate import load_gravity

        gravity = load_gravity(args.gravity)
        if gravity is None:
            raise SystemExit(f"cannot read gravity from {args.gravity}")
    else:
        gravity = [-0.010903244125662598, 0.996517323960189, 0.08267008113435093]

    totals: dict[str, int] = {}
    per_run = []
    for run_path in args.run:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        track = accepted_track(payload.get("rows") or [])
        derive_yaw(track)
        for seed in args.seeds:
            stats = evaluate(
                track,
                max_jump=args.max_jump,
                rate=args.rate,
                seed=seed,
                gravity=gravity,
                teleport_fraction=args.teleport_fraction,
                yaw_flip_deg=args.yaw_flip_deg,
                roll_deg=args.roll_deg,
            )
            stats["run"] = run_path.name
            stats["seed"] = seed
            per_run.append(stats)
            for key, value in stats.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value

    def pct(part: int, whole: int) -> float:
        return 100.0 * part / whole if whole else float("nan")

    summary = {
        "runs": len(args.run),
        "seeds": args.seeds,
        "injection_rate": args.rate,
        "teleport_detection_pct": pct(
            totals.get("caught_teleport", 0), totals.get("injected_teleport", 0)
        ),
        "yaw_detection_pct": pct(totals.get("caught_yaw", 0), totals.get("injected_yaw", 0)),
        "roll_detection_pct": pct(totals.get("caught_roll", 0), totals.get("injected_roll", 0)),
        "clean_false_kill_pct_trajectory": pct(
            totals.get("clean_trajectory_vetoes", 0), totals.get("clean_frames", 0)
        ),
        "clean_false_kill_pct_gravity": pct(
            totals.get("clean_gravity_vetoes", 0), totals.get("clean_frames", 0)
        ),
        "totals": totals,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.write_text(
            json.dumps({"summary": summary, "per_run": per_run}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

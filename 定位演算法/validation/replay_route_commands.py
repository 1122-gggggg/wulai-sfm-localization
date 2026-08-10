#!/usr/bin/env python3
"""Replay-to-command validation: what WOULD the drone have been commanded?

Takes the per-frame output of benchmark_production_stream.py (recorded-video
localization results: pose / success / inliers / weak / reproj_rms) and feeds
it through the REAL flight control path -- path_follow_flight.run_loop with the
production RouteAutoController and every safety gate active (freshness, NaN,
pose jump, low-confidence, route deviation, completion) -- into a MOCK command
sink. No Olympe, no drone, no motors.

Output: a JSONL command log (one line per control tick: gate status, block
reason, final PCMD) plus a summary. Use it to review dangerous behavior
(unexpected nonzero commands, deviation aborts, lost-localization landings)
BEFORE real flight.

Approximation: one recorded frame is presented per control tick (the recorded
video has no wall-clock alignment with the 20 Hz loop), so time-based gates
(LOST_LAND_S etc.) run on that virtual clock.

Usage:
  python3 replay_route_commands.py --bench-json <benchmark_result.json> \
      [--path-json <flight_path.json>] [--out <cmdlog.jsonl>] [--yaw-sign 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FLIGHT_CONTROL = _HERE.parent / "flight_control"
if str(_FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_CONTROL))


def _load_rows(path: str, limit: int) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if limit and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def _print_replay_summary(
    reason: str,
    rows: list[dict],
    records: list[dict],
    sent: list[tuple],
    state: dict,
    out: Path,
) -> None:
    driven = [r for r in records if not r.get("blocked")]
    blocked = [r for r in records if r.get("blocked")]
    reasons: dict[str, int] = {}
    for record in blocked:
        key = str(record.get("reason", "?")).split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    errs = [r["path_error_u"] for r in records if r.get("path_error_u") is not None]
    nonzero = [command for command in sent if tuple(command) != (0, 0, 0, 0)]

    print("\n=== replay-to-command summary ===")
    print(f"terminal reason      : {reason}")
    print(f"frames consumed      : {state['i']}/{len(rows)}")
    print(f"ticks logged         : {len(records)}  (driven={len(driven)}, blocked={len(blocked)})")
    print(f"PCMD sent            : {len(sent)}  (nonzero={len(nonzero)}, zero/hover={len(sent) - len(nonzero)})")
    print(f"relocalize requests  : {state['reloc']}")
    if errs:
        errs_sorted = sorted(errs)
        print(f"path error (u)       : max={errs_sorted[-1]:.3f}  "
              f"p90={errs_sorted[int(0.9 * (len(errs_sorted) - 1))]:.3f}")
    if reasons:
        print("block reasons:")
        for key, value in sorted(reasons.items(), key=lambda item: -item[1]):
            print(f"  {value:5d}  {key}")
    print(f"command log          : {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-json", required=True,
                    help="output of benchmark_production_stream.py (has per-frame rows)")
    ap.add_argument("--path-json", default="",
                    help="route waypoints JSON (default: production SFM_FLIGHT_PATH_JSON)")
    ap.add_argument("--out", default="",
                    help="JSONL command log (default: <bench-json stem>_replay_cmdlog.jsonl)")
    ap.add_argument("--yaw-sign", type=int, default=1, choices=(-1, 1))
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    if args.path_json:
        os.environ["SFM_FLIGHT_PATH_JSON"] = str(Path(args.path_json).resolve())
    # import AFTER the env override: path_follow_flight reads SFM_* at import time
    import path_follow_flight as pff
    import real_path_follow_controller as rpf

    print("[mode] REPLAY-VALIDATE: mock command sink only -- no drone, no Olympe", flush=True)

    rows = _load_rows(args.bench_json, args.limit)

    wp = rpf.load_waypoints(pff.PATH_JSON)
    ctrl = rpf.RouteAutoController(wp, poles=[],
                                   config=rpf.ControlConfig(inspect_waypoints=()))
    print(f"route: {pff.PATH_JSON} ({len(wp)} waypoints, len={ctrl.path_len:.2f}u)")
    print(f"frames: {len(rows)} from {args.bench_json}")

    out = Path(args.out) if args.out else (
        Path(args.bench_json).with_name(Path(args.bench_json).stem + "_replay_cmdlog.jsonl"))
    clog = pff.CommandLog(out, sink="replay-mock")

    tick = 1.0 / pff.CTRL_HZ
    S = {"i": 0, "t": 0.0, "row": {}, "reloc": 0}
    records: list[dict] = []
    sent: list[tuple] = []

    def get_pose():
        i = S["i"]
        if i >= len(rows):
            raise KeyboardInterrupt          # recording exhausted
        S["i"] = i + 1
        S["t"] += tick                       # 1 frame per control tick (virtual clock)
        row = rows[i]
        S["row"] = row
        p = row.get("pose")
        if not row.get("success") or not p:
            return None                      # LOST / PnP fail / reproj reject on this frame
        return rpf.Pose(x=float(p["x"]), y=float(p["y"]), z=float(p["z"]),
                        yaw=float(p.get("yaw", 0.0)), stamp=S["t"])

    def log_tick(rec):
        records.append(rec)
        clog(rec)

    hooks = pff.LoopHooks(
        get_pose=get_pose,
        olympe_yaw=lambda: None,             # heading from map motion, like --dry-run
        send_pcmd=lambda r, p, y, g: sent.append((r, p, y, g)),
        pose_is_weak=lambda: bool(S["row"].get("weak", False)),
        pose_confidence=lambda: int(S["row"].get("inliers", 0) or 0),
        force_relocalize=lambda: S.__setitem__("reloc", S["reloc"] + 1),
        request_manual=lambda: False,        # no pilot in replay -> LOST lands
        stream_healthy=lambda: True,         # recorded frames exist by definition
        pose_info=lambda: {k: S["row"].get(k) for k in
                           ("frame", "idx", "mode", "next_mode", "inliers", "reproj_rms", "weak")},
        log_tick=log_tick,
        now=lambda: S["t"],
    )

    try:
        reason = pff.run_loop(hooks, ctrl, wp, yaw_sign=args.yaw_sign, verbose=False)
    except KeyboardInterrupt:
        reason = "end of recorded frames"
    clog.event(event="terminal", reason=reason, frames_consumed=S["i"], frames_total=len(rows))
    clog.close()

    _print_replay_summary(reason, rows, records, sent, S, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only AUTO trajectory-vs-route error debrief for one flight session.

Answers the operator's question: after the drone joined the route, how far
was the flown trajectory from the planned path? Join-phase wandering (takeoff
to first waypoint arrival) is reported separately and never mixed into the
error statistics.

Reads only ``localization.jsonl`` from the session directory (the
``auto_route_plan`` / ``auto_route_tick`` / ``pose_result`` events the AUTO
loop already emits). Writes nothing unless ``--out`` is given.
"""

import argparse
import bisect
import json
import math
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[int(rank)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def _phase_bucket(tick: dict[str, Any]) -> str:
    reason = str(tick.get("reason") or "").lower()
    if tick.get("blocked") is True:
        if "speed" in reason:
            return "guard_hover"
        if "no fresh pose" in reason or "localization recovery" in reason:
            return "search_lost"
        return "hover_other"
    if "yaw localization search" in reason:
        return "search_lost"
    if "no fresh pose" in reason or "localization recovery" in reason:
        return "search_lost"
    phase = tick.get("pcmd_phase")
    if isinstance(phase, str) and phase not in ("", "idle"):
        return phase
    return "hover_other"

def _vo_share(
    post: list[dict[str, Any]], pose_timeline: list[tuple[float, bool]]
) -> float | None:
    """Share of post-join positioned ticks whose nearest pose was VO-only.

    Both clocks are host-monotonic seconds, so each tick is attributed to the
    latest pose fix not newer than the tick (plus a small tolerance). Ticks
    without a pose within 2 s stay unattributed and out of the denominator.
    """
    if not pose_timeline:
        return None
    times = [stamp for stamp, _ in pose_timeline]
    attributed = 0
    vo = 0
    for tick in post:
        if _finite(tick.get("route_distance_u")) is None:
            continue
        moment = _finite(tick.get("t"))
        if moment is None:
            continue
        index = bisect.bisect_right(times, moment + 0.05) - 1
        if index < 0 or moment - times[index] > 2.0:
            continue
        attributed += 1
        if pose_timeline[index][1]:
            vo += 1
    if not attributed:
        return None
    return round(vo / attributed, 3)


def _split_auto_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split plans+ticks by AUTO run so manual gaps never leak into one debrief.

    Newer rows carry ``auto_run_id``; older logs fall back to a fresh
    ``auto_route_plan`` boundary (the caller passes plans before ticks, so a
    plan always opens a new run even though it carries no ``t``/``step``).
    A plan row anchors its own run and never lands in the previous run's
    tick list.
    """
    runs: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for row in rows:
        if row.get("event") == "auto_route_plan":
            run_id = row.get("auto_run_id")
            current = {"auto_run_id": run_id, "plan": row, "ticks": []}
            runs.append(current)
            if isinstance(run_id, str) and run_id:
                by_id[run_id] = current
            continue
        run_id = row.get("auto_run_id")
        if isinstance(run_id, str) and run_id:
            selected = by_id.get(run_id)
            if selected is None:
                selected = {"auto_run_id": run_id, "plan": None, "ticks": []}
                by_id[run_id] = selected
                runs.append(selected)
            current = selected
        elif current is None:
            current = {"auto_run_id": run_id, "plan": None, "ticks": []}
            runs.append(current)
        current["ticks"].append(row)
    return [run for run in runs if run["ticks"] or isinstance(run.get("plan"), dict)]


def _run_plan_for(run: dict[str, Any], plans: list[dict[str, Any]]) -> dict[str, Any] | None:
    if isinstance(run.get("plan"), dict):
        return run["plan"]
    run_id = run.get("auto_run_id")
    if isinstance(run_id, str) and run_id:
        for plan in plans:
            if plan.get("auto_run_id") == run_id:
                return plan
    if len(plans) == 1:
        return plans[0]
    return None


def _analyze_run(
    run: dict[str, Any],
    plan: dict[str, Any],
    pose_timeline: list[tuple[float, bool]],
) -> dict[str, Any]:
    ticks = sorted(run["ticks"], key=lambda tick: float(tick.get("t", 0.0) or 0.0))
    indexed = [tick for tick in ticks if isinstance(tick.get("target_index"), int)]
    initial_target = indexed[0]["target_index"] if indexed else None
    join_tick: dict[str, Any] | None = None
    for tick in indexed:
        if tick["target_index"] > initial_target:
            join_tick = tick
            break
    distances = [
        _finite(tick.get("route_distance_u"))
        for tick in ticks
        if _finite(tick.get("route_distance_u")) is not None
    ]
    entry: dict[str, Any] = {
        "auto_run_id": run.get("auto_run_id"),
        "tick_count": len(ticks),
        "auto_window_s": round(ticks[-1].get("t", 0.0) - ticks[0].get("t", 0.0), 1),
        "plan": {
            "arrive_radius_u": _finite(plan.get("waypoint_arrive_radius_u")),
            "waypoint_arrive_radii_u": plan.get("waypoint_arrive_radii_u"),
            "drawn_waypoint_count": plan.get("drawn_waypoint_count"),
            "return_to_start": bool(plan.get("return_to_start")),
            "accept_weak_poses": bool(plan.get("accept_weak_poses", False)),
        },
        "join": {
            "joined": join_tick is not None,
            "initial_target": initial_target,
            "closest_approach_u": round(min(distances), 3) if distances else None,
        },
    }
    if join_tick is not None:
        entry["join"]["join_t"] = join_tick.get("t")
        entry["join"]["time_to_join_s"] = round(
            join_tick.get("t", 0.0) - ticks[0].get("t", 0.0), 1
        )
    arrivals: list[dict[str, Any]] = []
    last_target = initial_target
    for tick in indexed:
        if tick["target_index"] != last_target:
            arrivals.append({"target_index": tick["target_index"], "t": tick.get("t")})
            last_target = tick["target_index"]
    entry["arrivals"] = arrivals
    if join_tick is None:
        entry["verdict"] = "NEVER_JOINED"
        return entry
    post = [tick for tick in ticks if float(tick.get("t", 0.0) or 0.0) >= float(join_tick.get("t", 0.0) or 0.0)]
    post_errors = sorted(
        distance
        for tick in post
        if (distance := _finite(tick.get("route_distance_u"))) is not None
    )
    progs = [
        _finite(tick.get("progress"))
        for tick in post
        if _finite(tick.get("progress")) is not None
    ]
    phase_seconds: dict[str, float] = {}
    for tick, nxt in zip(post, post[1:]):
        span = float(nxt.get("t", 0.0) or 0.0) - float(tick.get("t", 0.0) or 0.0)
        if not math.isfinite(span) or span < 0.0 or span > 5.0:
            continue
        bucket = _phase_bucket(tick)
        phase_seconds[bucket] = phase_seconds.get(bucket, 0.0) + span
    blocked_s = 0.0
    for tick, nxt in zip(post, post[1:]):
        if tick.get("blocked") is True:
            span = float(nxt.get("t", 0.0) or 0.0) - float(tick.get("t", 0.0) or 0.0)
            if math.isfinite(span) and 0.0 <= span <= 5.0:
                blocked_s += span
    post_join: dict[str, Any] = {
        "samples": len(post_errors),
        "pose_coverage": round(len(post_errors) / max(1, len(post)), 3),
        "cross_track_u": {
            "mean": round(sum(post_errors) / len(post_errors), 3) if post_errors else None,
            "p95": round(p95, 3) if (p95 := _percentile(post_errors, 95)) is not None else None,
            "max": round(post_errors[-1], 3) if post_errors else None,
        },
        "progress_from": progs[0] if progs else None,
        "progress_to": progs[-1] if progs else None,
        "phase_seconds": {name: round(seconds, 1) for name, seconds in phase_seconds.items()},
        "blocked_seconds": round(blocked_s, 1),
        "vo_share": _vo_share(post, pose_timeline),
    }
    entry["post_join"] = post_join
    if post_errors and (radius := _finite(plan.get("waypoint_arrive_radius_u"))):
        post_join["within_arrival_sphere_share"] = round(
            sum(1 for value in post_errors if value <= radius) / len(post_errors), 3
        )
    entry["verdict"] = "JOINED"
    return entry


def analyze_session(session: Path) -> dict[str, Any]:
    """Debrief every AUTO run in one flight session, separately.

    ``auto_route_plan`` rows anchor their own run by ``auto_run_id``; ticks
    without an id fall back to plan/step boundaries like the debug bundle.
    Manual time between runs never enters join timing or post-join error.
    """
    events: list[dict[str, Any]] = []
    pose_modes: dict[str, int] = {}
    vo_only = 0
    pose_success = 0
    pose_timeline: list[tuple[float, bool]] = []
    for record in read_jsonl(session / "localization.jsonl"):
        event = record.get("event")
        if event in ("auto_route_plan", "auto_route_tick"):
            events.append(record)
        elif event == "pose_result":
            mode = str(record.get("mode") or "UNKNOWN")
            pose_modes[mode] = pose_modes.get(mode, 0) + 1
            stamp = _finite(record.get("t_mono_ns"))
            stamp_s = stamp / 1e9 if stamp is not None else None
            if record.get("success"):
                pose_success += 1
                is_vo = record.get("direct_status") == "VO_ONLY"
                if is_vo:
                    vo_only += 1
                if stamp_s is not None:
                    pose_timeline.append((stamp_s, is_vo))
    pose_timeline.sort()
    plans = [row for row in events if row.get("event") == "auto_route_plan"]
    ticks = [row for row in events if row.get("event") == "auto_route_tick"]
    report: dict[str, Any] = {
        "session": session.name,
        "has_auto": bool(plans) and len(ticks) > 0,
        "pose_modes": pose_modes,
        "pose_vo_only": vo_only,
        "pose_success": pose_success,
    }
    if not plans or not ticks:
        report["verdict"] = "NO_AUTO"
        report["auto_runs"] = []
        return report
    runs = _split_auto_runs(events)
    entries: list[dict[str, Any]] = []
    for run in runs:
        if not run["ticks"]:
            continue
        plan = _run_plan_for(run, plans)
        if plan is None:
            entries.append({
                "auto_run_id": run.get("auto_run_id"),
                "tick_count": len(run["ticks"]),
                "verdict": "NO_PLAN",
            })
            continue
        entries.append(_analyze_run(run, plan, pose_timeline))
    # Legacy single-run shape: the newest/last run stays top-level so old
    # callers keep working while multi-run sessions gain auto_runs.
    latest = entries[-1] if entries else None
    if latest is not None:
        for key in ("tick_count", "plan", "join", "arrivals", "verdict", "post_join"):
            if key in latest:
                report[key] = latest[key]
        if "auto_window_s" in latest:
            report["auto_window_s"] = latest["auto_window_s"]
    else:
        report["verdict"] = "NO_AUTO"
    report["auto_runs"] = entries
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"AUTO 軌跡誤差：{report['session']}"]
    if report.get("verdict") == "NO_AUTO":
        return "\n".join(lines + ["此班沒有 AUTO（無 auto_route_plan/tick），無法分析軌跡誤差。"])
    plan = report.get("plan", {})
    radii = plan.get("waypoint_arrive_radii_u")
    radii_text = f"，逐點 {radii}" if radii else ""
    lines.append(
        f"AUTO {report['tick_count']} tick / {report['auto_window_s']}s，"
        f"到達球 {plan.get('arrive_radius_u')}u{radii_text}，航點 {plan.get('drawn_waypoint_count')} 個"
    )
    if len(report.get("auto_runs", [])) > 1:
        lines.append(f"共 {len(report['auto_runs'])} 段 AUTO，以下為最後一段；各段見 auto_runs。")
    join = report["join"]
    if not join["joined"]:
        lines.append(
            f"從未進站：最接近 {join['closest_approach_u']}u（球 {plan.get('arrive_radius_u')}u），"
            "誤差不計（起飛靠攏段不算誤差）。"
        )
    else:
        lines.append(f"進站耗時 {join['time_to_join_s']}s，之後才開始計誤差。")
        post = report["post_join"]
        cross = post["cross_track_u"]
        lines.append(
            f"站後誤差（{post['samples']} 樣本，位姿覆蓋 {post['pose_coverage']}）："
            f"mean {cross['mean']}u / p95 {cross['p95']}u / max {cross['max']}u；"
            f"進度 {post['progress_from']} → {post['progress_to']}"
        )
        phases = ", ".join(f"{name} {secs}s" for name, secs in sorted(post["phase_seconds"].items()))
        lines.append(f"站後時間分配：{phases}；受阻 {post['blocked_seconds']}s")
        if "within_arrival_sphere_share" in post:
            lines.append(f"站後樣本在到達球內佔比 {post['within_arrival_sphere_share']}")
    arrivals = report.get("arrivals", [])
    if arrivals:
        lines.append("到站事件：" + ", ".join(f"wp{a['target_index']}@{a['t']:.0f}s" for a in arrivals))
    modes = report.get("pose_modes", {})
    if modes:
        lines.append("定位：" + ", ".join(f"{name} {count}" for name, count in sorted(modes.items())))
    lines.append(f"判定：{report['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="flight session directory")
    parser.add_argument("--out", type=Path, default=None, help="write JSON report here")
    args = parser.parse_args(argv)
    report = analyze_session(args.session)
    print(format_report(report))
    if args.out is not None:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

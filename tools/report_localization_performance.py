#!/usr/bin/env python3
"""Report end-to-end localization delivery rate from a flight-log session.

Read-only measurement for the two-rate 30 FPS plan: how many *distinct*
frames' localization results arrived and were consumed by the UI per second,
plus the latency/quality observables that must not regress while the UI
schedule is changed. No execution behaviour is modified.

Counting contract (exact):
  * identity key is (display_seq, frame_name); rows without display_seq are
    excluded as missing_identity_rows;
  * hold_retry rows are excluded from every FPS figure;
  * repeated identity keys count as duplicate_rows, first row wins;
  * the clock is ui_arrival_mono, falling back to t_mono; rows with neither
    are excluded as missing_clock_rows.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

#: direct_status values that prove the pose was confirmed against the map.
MAP_CONFIRMED_STATUSES = ("FAST_TRACK", "RELOC_SEED")

#: Acceptance bounds for the 30 FPS plan (per session).
MEAN_WINDOW_RATIO = 0.99
MIN_WINDOW_RATIO = 0.96
E2E_P95_MAX_MS = 50.0
E2E_P99_MAX_MS = 100.0

#: missing_clock_rows above this share of counted rows means the session
#: cannot answer the rate question at all.
MAX_MISSING_CLOCK_RATIO = 0.05

#: Latency fields summarised with linear-interpolation percentiles.
LATENCY_FIELDS = (
    "core_wall_ms",
    "e2e_submit_to_ui_ms",
    "ui_poll_delay_ms",
    "source_stamp_age_at_ui_ms",
    "client_roundtrip_ms",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile_linear(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (pct in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _median(values: list[float]) -> float | None:
    return _percentile_linear(values, 50.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file strictly; any parse error is an IO-class failure."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: JSON 解析失敗：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: 必須是 JSON object")
            rows.append(record)
    return rows


def _row_clock(row: dict[str, Any]) -> float | None:
    stamp = _finite(row.get("ui_arrival_mono"))
    if stamp is None:
        stamp = _finite(row.get("t_mono"))
    return stamp


def _pose_valid(row: dict[str, Any]) -> bool:
    if row.get("success") is not True:
        return False
    pose = row.get("pose")
    if not isinstance(pose, dict):
        return False
    return all(_finite(pose.get(axis)) is not None for axis in ("x", "y", "z"))


def _map_confirmed(row: dict[str, Any]) -> bool:
    return row.get("direct_status") in MAP_CONFIRMED_STATUSES and _pose_valid(row)


def _row_on_hold(row: dict[str, Any]) -> bool:
    """A real BOOT/LOST hold; the logger writes "none" when not holding."""
    kind = row.get("hold_kind")
    if isinstance(kind, str):
        if kind.strip().lower() not in ("", "none"):
            return True
    elif kind:
        return True
    return row.get("mode") == "LOST" or row.get("next_mode") == "LOST"


def _ffprobe_fps(path: str) -> float | None:
    """Source frame rate via ffprobe r_frame_rate (e.g. 24000/1001)."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
        rate = payload["streams"][0]["r_frame_rate"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    try:
        text = str(rate).strip()
        if "/" in text:
            num, den = text.split("/", 1)
            fps = float(num) / float(den)
        else:
            fps = float(text)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if math.isfinite(fps) and fps > 0 else None


def _resolve_source(manifest: dict[str, Any]) -> dict[str, Any]:
    mode = str(manifest.get("mode") or "")
    candidates = [manifest.get("source"), manifest.get("video")]
    path = next(
        (c.strip() for c in candidates if isinstance(c, str) and c.strip()),
        "",
    )
    if path and Path(path).is_file():
        fps = _ffprobe_fps(path)
        return {
            "source_kind": "file",
            "source_path": path,
            "source_fps": fps,
            "nominal_fps": fps,
            "nominal_fps_source": "ffprobe" if fps is not None else None,
        }
    if "real" in mode or "live" in mode or "anafi" in mode:
        return {
            "source_kind": "live",
            "source_path": path or None,
            "source_fps": None,
            "nominal_fps": 30.0,
            "nominal_fps_source": "live_default",
        }
    return {
        "source_kind": "unknown",
        "source_path": path or None,
        "source_fps": None,
        "nominal_fps": None,
        "nominal_fps_source": None,
    }


def _summarize_tick_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    p50s: dict[str, list[float]] = {}
    p95s: dict[str, list[float]] = {}
    rates: list[float] = []
    periods: list[float] = []
    for row in rows:
        stages = row.get("stages")
        if not isinstance(stages, dict):
            continue
        for name, stats in stages.items():
            if not isinstance(stats, dict):
                continue
            p50 = _finite(stats.get("p50"))
            p95 = _finite(stats.get("p95"))
            if p50 is not None:
                p50s.setdefault(str(name), []).append(p50)
            if p95 is not None:
                p95s.setdefault(str(name), []).append(p95)
        ticks = _finite(row.get("ticks"))
        window_s = _finite(row.get("window_s"))
        if ticks is not None and window_s:
            rates.append(ticks / window_s)
        period = _finite(row.get("tick_period_ms"))
        if period is not None:
            periods.append(period)
    stages_out = {
        name: {"p50": _median(vals), "p95": max(p95s.get(name, []), default=None)}
        for name, vals in p50s.items()
    }
    return {
        "emits": len(rows),
        "stages": stages_out,
        "tick_rate_hz": _median(rates),
        "tick_period_ms": _median(periods),
    }


def _longest_result_streak(
    counted: list[tuple[float, dict[str, Any]]],
    matches: Callable[[dict[str, Any]], bool],
) -> float:
    """Observed state duration on the delivery clock, ending at recovery or log end."""
    started = None
    longest = 0.0
    for stamp, row in counted:
        if started is not None:
            longest = max(longest, stamp - started)
        if matches(row):
            if started is None:
                started = stamp
        else:
            started = None
    return longest


def build_session_report(
    session: Path,
    *,
    window_s: float,
    warmup_s: float,
    nominal_override: float | None,
) -> dict[str, Any]:
    """Build the per-session report dict (never raises on thin data)."""
    manifest = json.loads((session / "session_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{session}/session_manifest.json: 必須是 JSON object")
    loc_rows = [
        row
        for row in read_jsonl(session / "localization.jsonl")
        if row.get("event") == "pose_result"
    ]
    tick_rows = [
        row
        for row in read_jsonl(session / "telemetry.jsonl")
        if row.get("event") == "ui_tick_profile"
    ]

    counted, missing_identity, hold_retry, duplicate, missing_clock = _count_pose_rows(loc_rows)

    source = _resolve_source(manifest)
    nominal = nominal_override
    nominal_origin = "cli" if nominal_override is not None else source["nominal_fps_source"]
    if nominal_override is not None:
        source = {**source, "nominal_fps": nominal, "nominal_fps_source": "cli"}
    else:
        nominal = source["nominal_fps"]

    counts = {
        "total_pose_rows": len(loc_rows),
        "counted_rows": len(counted),
        "hold_retry_rows": hold_retry,
        "duplicate_rows": duplicate,
        "missing_identity_rows": missing_identity,
        "missing_clock_rows": missing_clock,
    }

    identity = {
        key: manifest.get(key)
        for key in (
            "mission_snapshot_id",
            "site_profile_sha256",
            "asset_sha256",
            "runtime_profile_sha256",
            "argv",
            "source_sha256",
        )
    }
    tick_profile = _summarize_tick_profile(tick_rows)

    # Thin data: still return a report, but mark it insufficient so the
    # caller exits 3 and never reports a pass.
    insufficient = _insufficient_samples(nominal, counted, warmup_s, window_s, missing_clock)

    if insufficient is not None or not counted:
        return {
            "session": str(session),
            "counts": counts,
            "source": source,
            "identity": identity,
            "tick_profile": tick_profile,
            "windows": [],
            "fps": {"unique_result_fps": None, "valid_pose_fps": None, "map_confirmed_fps": None},
            "latency_ms": {},
            "quality": {},
            "acceptance": {"pass": False, "reason": insufficient or "no counted rows"},
            "insufficient": insufficient or "no counted rows",
        }

    first_t = counted[0][0]
    last_t = counted[-1][0]
    span = last_t - first_t
    valid = [(t, r) for t, r in counted if _pose_valid(r)]
    confirmed = [(t, r) for t, r in counted if _map_confirmed(r)]
    fps_block = {
        "unique_result_fps": len(counted) / span if span > 0 else 0.0,
        "valid_pose_fps": len(valid) / span if span > 0 else 0.0,
        "map_confirmed_fps": len(confirmed) / span if span > 0 else 0.0,
    }

    # Windows use absolute mono seconds on the counted timeline.
    t0 = first_t + warmup_s
    n_windows = int((last_t - t0) // window_s)
    raw_times: list[tuple[float, dict[str, Any]]] = []
    for row in loc_rows:
        stamp = _row_clock(row)
        if stamp is not None:
            raw_times.append((stamp, row))
    windows = []
    for index in range(max(n_windows, 0)):
        start = t0 + index * window_s
        end = start + window_s
        unique = sum(1 for t, _ in counted if start <= t < end)
        hold_or_lost = any(start <= t < end and _row_on_hold(row) for t, row in raw_times)
        windows.append(
            {
                "start_s": start,
                "end_s": end,
                "unique_results": unique,
                "fps": unique / window_s,
                "hold_or_lost": hold_or_lost,
            }
        )

    latency: dict[str, Any] = {}
    for field in LATENCY_FIELDS:
        vals = [_finite(r.get(field)) for _, r in counted]
        vals = [v for v in vals if v is not None]
        entry: dict[str, Any] = {
            "p50": _percentile_linear(vals, 50.0),
            "p95": _percentile_linear(vals, 95.0),
            "p99": _percentile_linear(vals, 99.0),
        }
        if field == "source_stamp_age_at_ui_ms" and source["source_kind"] == "file":
            # File replay stamps measure receipt pacing, never camera
            # exposure-to-UI latency.
            entry["semantics"] = "file_receipt"
        latency[field] = entry

    # reloc_ms persists in last_info between deliveries. Count each completed
    # job once, rather than weighting its runtime by the following frame count.
    reloc_times = [
        value
        for _, row in counted
        if row.get("reloc_delivered") is True
        and (value := _finite(row.get("reloc_ms"))) is not None
        and value >= 0.0
    ]
    latency["reloc_ms"] = {
        "samples": len(reloc_times),
        "p50": _percentile_linear(reloc_times, 50.0),
        "p95": _percentile_linear(reloc_times, 95.0),
        "p99": _percentile_linear(reloc_times, 99.0),
    }

    status_counts: dict[str, int] = {}
    for _, row in counted:
        status = row.get("direct_status") or "UNKNOWN"
        status_counts[str(status)] = status_counts.get(str(status), 0) + 1
    map_ages = [
        value
        for _, row in counted
        if (value := _finite(row.get("map_constraint_age_s"))) is not None and value >= 0.0
    ]
    reproj = [v for _, r in counted if (v := _finite(r.get("reproj_rms"))) is not None]
    inliers = [v for _, r in counted if (v := _finite(r.get("inliers"))) is not None]
    quality = {
        "success_fraction": sum(1 for _, r in counted if r.get("success") is True) / len(counted),
        "map_confirmed_fraction": len(confirmed) / len(counted),
        "status_counts": status_counts,
        "longest_no_pose_streak_s": _longest_result_streak(counted, lambda r: not _pose_valid(r)),
        "longest_vo_only_streak_s": _longest_result_streak(
            counted, lambda r: _pose_valid(r) and r.get("direct_status") == "VO_ONLY"
        ),
        "longest_map_unconfirmed_streak_s": _longest_result_streak(
            counted, lambda r: not _map_confirmed(r)
        ),
        "max_map_constraint_age_s": max(map_ages, default=None),
        "streak_clock": "ui_arrival_mono_or_t_mono",
        "reproj_rms_p90": _percentile_linear(reproj, 90.0),
        "inliers_p50": _median(inliers),
    }

    eligible = [w["fps"] for w in windows if not w["hold_or_lost"]]
    mean_fps = sum(eligible) / len(eligible) if eligible else None
    min_fps = min(eligible) if eligible else None
    e2e = latency["e2e_submit_to_ui_ms"]
    assert nominal is not None
    mean_ok = mean_fps is not None and mean_fps >= MEAN_WINDOW_RATIO * nominal
    min_ok = min_fps is not None and min_fps >= MIN_WINDOW_RATIO * nominal
    e2e_ok = (
        e2e["p95"] is not None
        and e2e["p95"] <= E2E_P95_MAX_MS
        and e2e["p99"] is not None
        and e2e["p99"] <= E2E_P99_MAX_MS
    )
    acceptance = {
        "pass": bool(mean_ok and min_ok and e2e_ok),
        "mean_full_window_fps": mean_fps,
        "min_full_window_fps": min_fps,
        "nominal_fps": nominal,
        "nominal_fps_source": nominal_origin,
        "mean_ge_0_99_nominal": mean_ok,
        "min_ge_0_96_nominal": min_ok,
        "e2e_p95_le_50_p99_le_100": e2e_ok,
    }
    return {
        "session": str(session),
        "counts": counts,
        "source": source,
        "identity": identity,
        "tick_profile": tick_profile,
        "windows": windows,
        "fps": fps_block,
        "latency_ms": latency,
        "quality": quality,
        "acceptance": acceptance,
        "insufficient": None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        action="append",
        dest="sessions",
        required=True,
        help="flight_logs session 目錄（可重複）",
    )
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--nominal-fps", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.sessions:
        print("至少需要一個 --session", file=sys.stderr)
        return 2
    if not args.window_s or args.window_s <= 0:
        print("--window-s 必須為正數", file=sys.stderr)
        return 2
    if args.warmup_s is None or args.warmup_s < 0:
        print("--warmup-s 不得為負數", file=sys.stderr)
        return 2
    if args.nominal_fps is not None and args.nominal_fps <= 0:
        print("--nominal-fps 必須為正數", file=sys.stderr)
        return 2
    reports = []
    for raw in args.sessions:
        session = Path(raw).expanduser()
        try:
            if not session.is_dir():
                raise ValueError(f"session 目錄不存在：{session}")
            for name in ("localization.jsonl", "session_manifest.json", "telemetry.jsonl"):
                if not (session / name).is_file():
                    raise ValueError(f"缺少必要檔案：{session / name}")
            reports.append(
                build_session_report(
                    session,
                    window_s=args.window_s,
                    warmup_s=args.warmup_s,
                    nominal_override=args.nominal_fps,
                )
            )
        except (OSError, ValueError) as exc:
            print(f"讀取失敗：{exc}", file=sys.stderr)
            return 2
    print(json.dumps({"sessions": reports}, ensure_ascii=False, indent=2))
    if any(rep.get("insufficient") for rep in reports):
        return 3
    if all(rep["acceptance"]["pass"] for rep in reports):
        return 0
    return 1


def _count_pose_rows(loc_rows):
    missing_identity = 0
    hold_retry = 0
    duplicate = 0
    missing_clock = 0
    seen: set[tuple[Any, Any]] = set()
    counted: list[tuple[float, dict[str, Any]]] = []
    for row in loc_rows:
        if row.get("display_seq") is None:
            missing_identity += 1
            continue
        if row.get("hold_retry") is True:
            hold_retry += 1
            continue
        key = (row.get("display_seq"), row.get("frame_name"))
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        stamp = _row_clock(row)
        if stamp is None:
            missing_clock += 1
            continue
        counted.append((stamp, row))
    counted.sort(key=lambda item: item[0])

    return counted, missing_identity, hold_retry, duplicate, missing_clock


def _insufficient_samples(nominal, counted, warmup_s, window_s, missing_clock):
    insufficient: str | None = None
    if nominal is None:
        insufficient = "no nominal fps (unknown source, pass --nominal-fps)"
    elif not counted:
        insufficient = "no counted rows after warmup-independent filtering"
    else:
        first_t = counted[0][0]
        full_windows = int((counted[-1][0] - first_t - warmup_s) // window_s)
        if full_windows < 1:
            insufficient = "fewer than 1 full window after warmup"
        elif len(counted) > 0 and missing_clock > MAX_MISSING_CLOCK_RATIO * len(counted):
            insufficient = "missing_clock_rows exceed 5% of counted rows"

    return insufficient


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarise one or more seven-video corpus runs produced by benchmark_edm_site_replay.

`summary.successes` counts every frame that published a pose, and under the async
tracker that includes KLT-bridged frames -- poses carried forward by optical flow
from an older anchor. Their PnP re-fits the same map 3D that flow moved, so their
inlier and reprojection numbers measure the chain's own consistency, not agreement
with the map. On the 2026-09-05 corpus that gap was 3,587 successes against 429
frames that actually re-matched the map, including one 710-frame run.

So this reports two numbers per run:

    success   frames that published a pose (the replay summary's own count)
    verified  frames whose pose came from a fresh match on that frame:
              success, inliers > 0, and a candidate_mode that is not a KLT path

and, with --failures, buckets every failed frame by how many inliers it did get,
which is what says whether an acceptance floor could have saved it.

    python 定位演算法/validation/summarize_corpus_runs.py outputs/corpus_*/sync
    python 定位演算法/validation/summarize_corpus_runs.py --failures outputs/corpus_*/sync
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

#: candidate_mode values that mean "no fresh match on this frame". ``None`` shows
#: up as the string "None" once a run is written to JSON, and covers the async
#: NO_ANCHOR rows that carry no candidate at all.
KLT_CANDIDATE_MODES = frozenset({"klt_fast", "klt_bridge", "None", "none", ""})

#: Inlier bands, upper bound exclusive. The boundaries are the production floors:
#: weak 30, track 50, the measured relaxed-acquire floor 66, acquire 80.
INLIER_BANDS: tuple[tuple[int, int | None, str], ...] = (
    (0, 1, "no geometry at all"),
    (1, 30, "far below every floor"),
    (30, 50, "below track_min_inliers"),
    (50, 66, "below the measured relaxed floor"),
    (66, 80, "near-miss under acquire_min_inliers"),
    (80, None, "cleared the inlier floor; refused elsewhere"),
)


def is_verified(row: dict[str, Any]) -> bool:
    """True when this frame's pose came from a fresh match on this frame."""
    if not row.get("success"):
        return False
    if int(row.get("inliers") or 0) <= 0:
        return False
    return str(row.get("candidate_mode")) not in KLT_CANDIDATE_MODES


def summarize_run(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    rows = payload.get("rows") or []
    frames = int(summary["frames"])
    verified = sum(1 for row in rows if is_verified(row))
    return {
        "frames": frames,
        "successes": int(summary["successes"]),
        "verified": verified,
        "wall_p50_ms": summary["wall_ms"]["p50"],
        "wall_p95_ms": summary["wall_ms"]["p95"],
        "state_counts": dict(summary.get("state_counts") or {}),
    }


def failure_bands(rows: Iterable[dict[str, Any]]) -> list[tuple[str, int, str]]:
    """Failed frames per inlier band, in band order."""
    failures = [row for row in rows if not row.get("success")]
    counts: Counter[str] = Counter()
    for row in failures:
        inliers = int(row.get("inliers") or 0)
        for low, high, _note in INLIER_BANDS:
            if inliers >= low and (high is None or inliers < high):
                counts[f"{low}-{'inf' if high is None else high - 1}"] += 1
                break
    out = []
    for low, high, note in INLIER_BANDS:
        key = f"{low}-{'inf' if high is None else high - 1}"
        out.append((key, counts.get(key, 0), note))
    return out


def _load(directory: Path) -> dict[str, dict[str, Any]]:
    runs = {}
    for path in sorted(directory.glob("*.json")):
        try:
            runs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not runs:
        raise SystemExit(f"no run JSON in {directory}")
    return runs


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path,
                        help="corpus run directories, each holding one JSON per video")
    parser.add_argument("--failures", action="store_true",
                        help="also bucket failed frames by inlier count")
    args = parser.parse_args(argv)

    loaded = [(d.name, _load(d)) for d in args.directories]
    videos = sorted({name for _tag, runs in loaded for name in runs})

    width = max(len(v) for v in videos) + 2
    header = f"{'video':<{width}}" + "".join(
        f"{tag:>34}" for tag, _ in loaded)
    print(header)
    print(f"{'':<{width}}" + "".join(
        f"{'success        verified   p50':>34}" for _ in loaded))
    print("-" * len(header))

    totals = {tag: [0, 0, 0] for tag, _ in loaded}
    for video in videos:
        line = f"{video:<{width}}"
        for tag, runs in loaded:
            if video not in runs:
                line += f"{'-':>34}"
                continue
            s = summarize_run(runs[video])
            totals[tag][0] += s["successes"]
            totals[tag][1] += s["frames"]
            totals[tag][2] += s["verified"]
            line += (f"{s['successes']:>6}/{s['frames']:<5}"
                     f"{_pct(s['successes'], s['frames']):>5.1f}%"
                     f"{s['verified']:>7}{_pct(s['verified'], s['frames']):>6.1f}%"
                     f"{s['wall_p50_ms']:>5.0f}")
        print(line)

    print("-" * len(header))
    line = f"{'TOTAL':<{width}}"
    for tag, _ in loaded:
        ok, frames, verified = totals[tag]
        line += (f"{ok:>6}/{frames:<5}{_pct(ok, frames):>5.1f}%"
                 f"{verified:>7}{_pct(verified, frames):>6.1f}%{'':>5}")
    print(line)

    if args.failures:
        for tag, runs in loaded:
            rows = [row for run in runs.values() for row in run.get("rows") or []]
            bands = failure_bands(rows)
            total = sum(count for _key, count, _note in bands)
            print(f"\n{tag}: {total} failed frames by inliers")
            for key, count, note in bands:
                print(f"  {key:>8}  {count:>5}  {_pct(count, total):>5.1f}%  {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Interleaved, resumable baseline/full offline replay and comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import re
from statistics import median
import sys
from typing import Any

VALIDATION_DIR = Path(__file__).resolve().parent
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from online_workmap_eval import (  # noqa: E402
    EvaluationError, WORKER_PATH, load_protocol, _verify_full_profile,
    _prepare_protocol_snapshot, _run_one, _validate_run, _run_metrics, _visual_fresh,
    _percentile,
)

STAGE_FIELDS = (
    "track_ms", "pnp_ms", "vo_ms", "handover_ms", "bookkeeping_ms",
    "workmap_pre_ms", "workmap_core_ms", "workmap_post_ms",
)

_KEYFRAME_NAME = re.compile(r"keyframe \S+")
ROOT = VALIDATION_DIR.parents[1]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_hashes() -> dict[str, str]:
    paths = set((ROOT / "定位演算法/deploy_code_expand/sfm_direct_deploy").rglob("*.py"))
    paths.update(WORKER_PATH.parent.glob("*.py"))
    paths.add(VALIDATION_DIR / "online_workmap_eval.py")
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if "__pycache__" not in path.parts
    }


def _cells(protocol):
    for repeat in (1, 2, 3):
        for sequence in protocol.sequences:
            for mode in (("full", "baseline") if repeat == 2 else ("baseline", "full")):
                path = protocol.output_root / mode / sequence["raw"]["name"] / f"repeat-{repeat}"
                yield sequence, mode, repeat, path


def run(protocol_path: str | Path) -> None:
    protocol = load_protocol(protocol_path)
    _verify_full_profile(protocol, "full")
    _prepare_protocol_snapshot(protocol)
    before = _source_hashes()
    source_path = protocol.output_root / "source_before.json"
    if source_path.exists():
        if _read_json(source_path) != before:
            raise EvaluationError("sources changed since cohort start")
    else:
        _write_json(source_path, before)
    completed = []
    expected = len(protocol.sequences) * 6
    for sequence, mode, repeat, path in _cells(protocol):
        metadata = path / "run.json"
        complete = metadata.is_file() and _read_json(metadata).get("complete") is True
        if not complete:
            if path.exists():
                raise EvaluationError(f"partial run directory: {path}")
            if _source_hashes() != before:
                raise EvaluationError("sources changed since cohort start")
            if shutil.disk_usage(protocol.output_root).free < 3 * 1024 ** 3:
                raise EvaluationError("less than 3 GiB free disk space")
            _run_one(protocol, mode, sequence, repeat)
        record, rows = _validate_run(protocol, mode, sequence, repeat)
        metrics = _run_metrics(record, rows, sequence, protocol.raw["gates"])
        _write_json(path / "metrics.json", metrics)
        for image in (path / "session/keyframes").glob("*.png"):
            image.unlink()
        completed.append({"sequence": sequence["raw"]["name"], "mode": mode, "repeat": repeat,
                          "path": str(path), "frames": len(rows)})
        _write_json(protocol.output_root / "completion.json", {
            "runs": completed, "expected_runs": expected, "all_complete": len(completed) == expected,
        })
        print(path, flush=True)
    after = _source_hashes()
    _write_json(protocol.output_root / "source_after.json", after)
    if after != before:
        raise EvaluationError("sources changed since cohort start")


def _stage_ms(rows) -> dict[str, dict[str, float | None] | None]:
    result: dict[str, dict[str, float | None] | None] = {}
    for field in STAGE_FIELDS:
        values = [
            float(value) for row in rows
            for value in (row["response"].get(field),)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        result[field] = (
            {name: _percentile(values, q) for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}
            if values else None
        )
    return result


def _full_run_fields(session_dir, events, statistics) -> dict:
    committed = [event for event in events if event.get("event") == "ba_geometry_committed"]
    latencies = [
        float(event["submit_to_apply_ms"]) for event in committed
        if isinstance(event.get("submit_to_apply_ms"), (int, float))
    ]
    anchor_scales = [
        float(event["scale"]) for event in events
        if event.get("event") == "submap_anchor_rejected" and isinstance(event.get("scale"), (int, float))
    ]
    meta_path = Path(session_dir) / "session_meta.json"
    meta = _read_json(meta_path) if meta_path.is_file() else {}

    def reasons(name: str) -> dict:
        return dict(Counter(
            _KEYFRAME_NAME.sub("keyframe <name>", str(event.get("reason")))
            for event in events if event.get("event") == name
        ))

    return {
        "ba_committed": len(committed),
        "ba_rejected": reasons("ba_geometry_rejected"),
        "ba_updated_points_median": (
            median(len(event.get("updated_landmarks") or {}) for event in committed) if committed else None
        ),
        "ba_submit_to_apply_ms_p95": _percentile(latencies, 0.95),
        "loop_rejected": reasons("loop_geometry_rejected"),
        "anchor_rejected": reasons("submap_anchor_rejected"),
        "anchor_rejected_scale_median": median(anchor_scales) if anchor_scales else None,
        "closure_reseeded": sum(event.get("event") == "closure_reseeded" for event in events),
        "closure_rejected": reasons("closure_reseed_rejected"),
        "close_session_ms": statistics.get("close_session_ms"),
        "snapshot_write_ms": meta.get("snapshot_write_ms"),
        "snapshot_bytes": meta.get("snapshot_bytes"),
    }


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else str(value)


def _stage_p95(row: dict, field: str):
    stage = (row.get("stage_ms") or {}).get(field)
    return None if stage is None else stage["p95"]


def full_table_lines(sequences: dict) -> list[str]:
    lines = [
        "| 序列 | repeat | ba_committed | ba_rejected | ba_pts_med | merges | unmerged | closure | pre95 | vo95 | drain_s | snap_ms |",
        "|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in sequences.items():
        for row in item["full"]:
            values = [
                row["ba_committed"], row["ba_rejected"], row["ba_updated_points_median"], row["merges"],
                row["unmerged_reasons"], row["closure_reseeded"], _stage_p95(row, "workmap_pre_ms"),
                _stage_p95(row, "vo_ms"), row["drain_s"], row["snapshot_write_ms"],
            ]
            lines.append(f"| {name} | {row['repeat']} | " + " | ".join(_cell(value) for value in values) + " |")
    return lines


def summarize_run(record, rows, metrics, session_dir, mode, reseed_min_points) -> dict:
    fresh = strong = reloc_seed = refreshes = longest = streak = 0
    for row in rows:
        response = row["response"]
        visual = _visual_fresh(row)
        status = response.get("direct_status")
        fresh += int(visual)
        strong += int(visual and status in {"FAST_TRACK", "RELOC_SEED"} and response.get("map_inliers", 0) >= 80)
        reloc_seed += int(status == "RELOC_SEED")
        refreshes += int(bool(response.get("reloc_delivered")) and not response.get("klt_reseeded", False)
                         and response.get("handover_points", 0) >= reseed_min_points
                         and status in {"FAST_TRACK", "VO_ONLY"})
        streak = 0 if visual else streak + 1
        longest = max(longest, streak)
    result = {
        "frames": len(rows), "fresh": fresh, "strong": strong,
        "reloc_seed": reloc_seed, "refreshes": refreshes,
        "e2e_fps": metrics["e2e_fps"], "source_fps": record["nominal_source_fps"],
        "core95": metrics["foreground_core_wall_ms"]["p95"],
        "latency95": metrics["capture_to_consumer_ms"]["p95"],
        "latency99": metrics["capture_to_consumer_ms"]["p99"],
        "drain_s": record["run_wall_s"] - (rows[-1]["received_mono_ns"] - record["run_start_monotonic_ns"]) * 1e-9,
        "longest_nonfresh_s": longest / record["nominal_source_fps"],
        "stage_ms": _stage_ms(rows),
    }
    if mode == "full":
        snapshot = _read_json(Path(session_dir) / "snapshot.json")
        events = [json.loads(line) for line in (Path(session_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        statistics = snapshot["statistics"]
        counts = Counter(point["submap_id"] for point in snapshot["landmarks"].values())
        submaps = [{"id": item["id"], "kind": item["kind"], "state": item["state"],
                    "landmarks": counts[item["id"]]} for item in snapshot["submaps"]]
        kinds = {item["id"]: item["kind"] for item in snapshot["submaps"]}
        result.update({
            "points_created": statistics["points_created"], "submaps": submaps,
            "merges": dict(Counter(event.get("kind", kinds.get(event.get("submap_id"), "unknown"))
                                   for event in events if event.get("event") == "submap_merged" and not event.get("noop"))),
            "unmerged_reasons": dict(Counter(event["reason"] for event in events if event.get("event") == "submap_unmerged")),
            "presubmap_merged": sum(event.get("event") == "presubmap_merged" for event in events),
            "close_dropped": statistics.get("close_dropped"), "close_phase_ms": statistics.get("close_phase_ms"),
            **_full_run_fields(session_dir, events, statistics),
        })
    return result


def sequence_gates(baseline: list[dict], full: list[dict]) -> dict:
    if not baseline or not full:
        raise EvaluationError("comparison requires baseline and full runs")
    core = median(item["core95"] for item in baseline)
    gates = {
        "fresh": median(item["fresh"] for item in full) >= median(item["fresh"] for item in baseline),
        "strong": median(item["strong"] for item in full) >= 0.97 * median(item["strong"] for item in baseline),
        "throughput": all(item["e2e_fps"] >= 0.98 * item["source_fps"] for item in full),
        "core": all(item["core95"] <= max(2 * core, core + 8.0) for item in full),
        "latency": all(item["latency95"] <= 100 and item["latency99"] <= 200 for item in full),
    }
    gates["pass"] = all(gates.values())
    return gates


def report(protocol_path: str | Path) -> dict:
    protocol = load_protocol(protocol_path)
    profile = _read_json(protocol.profile_path)
    minimum = profile["fast_loop"]["reseed_min_points"]
    sequences = {sequence["raw"]["name"]: {"baseline": [], "full": []} for sequence in protocol.sequences}
    for sequence, mode, repeat, path in _cells(protocol):
        record, rows = _validate_run(protocol, mode, sequence, repeat)
        metrics = _run_metrics(record, rows, sequence, protocol.raw["gates"])
        summary = summarize_run(record, rows, metrics, path / "session", mode, minimum)
        summary["repeat"] = repeat
        sequences[sequence["raw"]["name"]][mode].append(summary)
    for item in sequences.values():
        item["gates"] = sequence_gates(item["baseline"], item["full"])
        for field in ("fresh", "strong"):
            item[field + "_gain"] = median(row[field] for row in item["full"]) - median(row[field] for row in item["baseline"])
    comparison = {"sequences": sequences, "winner": all(item["gates"]["pass"] for item in sequences.values())}
    _write_json(protocol.output_root / "comparison.json", comparison)
    print("| 序列 | 版本 | fresh | strong | FPS | core95 | lat99 | drain_s | reloc_seed | refreshes |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, item in sequences.items():
        for mode in ("baseline", "full"):
            for row in item[mode]:
                values = [row[field] for field in ("fresh", "strong", "e2e_fps", "core95", "latency99", "drain_s", "reloc_seed", "refreshes")]
                print(f"| {name} | {mode}/{row['repeat']} | " + " | ".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in values) + " |")
    print()
    for line in full_table_lines(sequences):
        print(line)
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "report"):
        subparsers.add_parser(command).add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    try:
        (run if args.command == "run" else report)(args.protocol)
    except (EvaluationError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"workmap_paired_compare: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

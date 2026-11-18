#!/usr/bin/env python3
"""Per-map height *consistency* report from flight logs; offline, read-only.

The controller chases a waypoint's direction in the map's measured up axis and
needs neither metres nor a calibrated height. This report asks only whether the
map height and the firmware barometric height moved together in AUTO flight.
It is not an accuracy claim: with no independent ground truth the best verdict
is CONSISTENT_NOT_VALIDATED. Every distance is a multiple of the route's own
arrival sphere, so a rescaled map gives the same classification.

A session counts for a map only when its manifest map and alignment SHA-256
match the profile. Manual, weak/bridged, repeated or stale-altitude samples
are excluded; a map with none of the remaining evidence is NO_FLIGHT_EVIDENCE.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "控制介面程式", ROOT / "定位演算法/flight_control"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import real_path_follow_controller as rpf  # noqa: E402
import site_profile as sp  # noqa: E402

STRONG = {"FAST_TRACK", "RELOC_SEED"}
WINDOW_S = 10.0
# Same comparison the controller's VerticalConsistencyGuard makes, so a window
# is "map_flat" exactly when that guard would have latched.
BARO_MIN_M = rpf.VERTICAL_GUARD_BARO_M
FOLLOW_FRACTION = rpf.VERTICAL_GUARD_FOLLOW_FRACTION
ALTITUDE_MAX_AGE_S = 1.0
MIN_MOVED_WINDOWS = 3
LIMITATION = (
    "consistency between map height and barometer only; no independent ground "
    "truth and no metric scale is asserted"
)


def read_jsonl(path: Path) -> tuple[list[dict], bool]:
    """Strict JSONL. Only an unterminated last line may be a live partial write."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows, partial = [], False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not text.endswith("\n"):
                partial = True
                continue
            raise ValueError(f"{path}:{index + 1}: {exc}") from exc
    return rows, partial


def classify_windows(samples, radius_u: float) -> dict[str, int]:
    """samples: (t_s, map_height_u, baro_m), time ordered. Scale-free by radius.

    Mirrors VerticalConsistencyGuard: a reference is the median of the first
    `WINDOW` samples; each time the median of the latest `WINDOW` samples is a
    barometric BARO_MIN_M away, one comparison is scored and a new reference
    starts. Whatever is left when the run ends without that much barometric
    motion is `baro_flat` (map moved) or `quiet`.
    """
    counts = dict.fromkeys(
        ("consistent", "opposed", "map_flat", "baro_flat", "quiet"), 0
    )
    window = rpf.VERTICAL_GUARD_WINDOW
    threshold = FOLLOW_FRACTION * radius_u
    reference: list = []
    recent: list = []

    def median(rows, index):
        return float(np.median([row[index] for row in rows]))

    for sample in samples:
        if len(reference) < window:
            reference.append(sample)
            continue
        recent = (recent + [sample])[-window:]
        if len(recent) < window:
            continue
        d_baro = median(recent, 2) - median(reference, 2)
        d_map = median(recent, 1) - median(reference, 1)
        if abs(d_baro) < BARO_MIN_M:
            continue
        if abs(d_map) >= threshold:
            counts["consistent" if d_map * d_baro > 0 else "opposed"] += 1
        else:
            counts["map_flat"] += 1
        reference, recent = [], []
    if len(reference) == window and recent:
        moved = abs(median(recent, 1) - median(reference, 1)) >= threshold
        counts["baro_flat" if moved else "quiet"] += 1
    return counts


def verdict(counts: dict[str, int], unobservable_events: int) -> str:
    moved = counts["consistent"] + counts["opposed"] + counts["map_flat"]
    if unobservable_events:
        return "HEIGHT_INCONSISTENT"
    if not any(counts.values()):
        return "NO_FLIGHT_EVIDENCE"
    if moved < MIN_MOVED_WINDOWS:
        return "INSUFFICIENT_VERTICAL_MOTION"
    if (counts["opposed"] + counts["map_flat"]) * 3 >= moved:
        return "HEIGHT_INCONSISTENT"
    return "CONSISTENT_NOT_VALIDATED"


def attribution_errors(manifest: dict, profile) -> list[str]:
    digests = manifest.get("asset_sha256") or {}
    errors = []
    for key in ("map_ply", "map_align"):
        expected = getattr(profile.asset_sha256, key)
        if digests.get(key) != expected:
            errors.append(f"{key} SHA-256 does not match this map")
    return errors


def session_samples(session: Path, profile):
    manifest = json.loads((session / "session_manifest.json").read_text(encoding="utf-8"))
    errors = attribution_errors(manifest, profile)
    if errors:
        return None, errors, {}
    trajectory, partial = read_jsonl(session / "trajectory.jsonl")
    plan = next((r for r in trajectory if r.get("event") == "auto_route_plan"), None)
    if plan is None:
        return [], [], {"partial_final_line": partial}
    frame = rpf.load_map_frame(profile.map_align)
    plan_up = np.asarray(plan["control_config"]["map_frame"]["up"], dtype=float)
    if not np.allclose(plan_up, frame.up, atol=1e-6):
        return None, ["session map up axis differs from the profile alignment"], {}
    up = np.asarray(frame.up, dtype=float)
    radii = plan["control_config"].get("waypoint_arrive_radii") or [
        plan["control_config"]["waypoint_arrive_radius"]
    ]
    seen, samples = set(), []
    for row in trajectory:
        loc = row.get("loc") or {}
        if (
            row.get("event") != "auto_route_tick"
            or row.get("auto_run_id") != plan.get("auto_run_id")
            or row.get("control_owner") != "PC"
            or row.get("pcmd") is None
            or loc.get("direct_status") not in STRONG
            or row.get("position_observed") is False
            or row.get("pose_new") is False
            or row.get("pose_stamp") in seen
            or row.get("observed_pose_u") is None
            or row.get("drone_altitude_m") is None
            or not 0.0 <= (row.get("altitude_age_s") if row.get("altitude_age_s") is not None else 9.0) <= ALTITUDE_MAX_AGE_S
        ):
            continue
        seen.add(row.get("pose_stamp"))
        samples.append((
            row["t_mono_ns"] * 1e-9,
            float(up @ np.asarray(row["observed_pose_u"], dtype=float)),
            float(row["drone_altitude_m"]),
        ))
    events = sum(
        1 for r in trajectory
        if r.get("event") == "autonomy_event" and r.get("kind") == "vertical_unobservable"
        and r.get("auto_run_id") == plan.get("auto_run_id")
    )
    return samples, [], {
        "radius_u": float(np.median(radii)),
        "vertical_unobservable_events": events,
        "partial_final_line": partial,
    }


def build_report(profiles: list[Path], sessions: list[Path]) -> dict:
    loaded = [sp.load_site_profile(path) for path in profiles]
    maps = []
    for profile in loaded:
        entry = {
            "site_id": profile.site_id,
            "coordinate_frame_id": profile.coordinate_frame.id,
            "map_sha256": profile.asset_sha256.map_ply,
            "up_axis_glomap": list(rpf.load_map_frame(profile.map_align).up),
            "sessions": [],
            "rejected_sessions": [],
        }
        totals = dict.fromkeys(("consistent", "opposed", "map_flat", "baro_flat", "quiet"), 0)
        unobservable = 0
        for session in sessions:
            try:
                samples, errors, meta = session_samples(session, profile)
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(f"{session.name}: {exc}") from exc
            if errors:
                entry["rejected_sessions"].append({"session": session.name, "reasons": errors})
                continue
            counts = classify_windows(samples, meta.get("radius_u", math.inf)) if samples else dict.fromkeys(totals, 0)
            for key, value in counts.items():
                totals[key] += value
            unobservable += meta.get("vertical_unobservable_events", 0)
            entry["sessions"].append({
                "session": session.name,
                "accepted_samples": len(samples or []),
                "windows": counts,
                **{k: v for k, v in meta.items() if k != "radius_u"},
            })
        entry["windows"] = totals
        entry["verdict"] = verdict(totals, unobservable)
        entry["limitation"] = LIMITATION
        maps.append(entry)
    return {"maps": maps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", action="append", type=Path, required=True)
    parser.add_argument("--session", action="append", type=Path)
    parser.add_argument("--flight-logs", type=Path, default=ROOT / "outputs/flight_logs")
    args = parser.parse_args(argv)
    sessions = args.session or sorted(
        p for p in args.flight_logs.glob("session_*real-flight*") if (p / "trajectory.jsonl").is_file()
    )
    try:
        report = build_report(args.site_profile, sessions)
    except ValueError as exc:
        print(f"height report failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

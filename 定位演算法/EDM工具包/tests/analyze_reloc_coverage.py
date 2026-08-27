#!/usr/bin/env python3
"""Audit extracted-frame -> EDM relocation-reference coverage by sequence and time."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.frame_manifest).read_text(encoding="utf-8"))
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    bundle_names = set(bundle["ref_names"])
    frames_by_seq: dict[str, list[dict]] = defaultdict(list)
    for row in manifest["frames"]:
        frames_by_seq[row["seq"]].append(row)

    sequences = []
    worst_gaps = []
    for seq, frames in sorted(frames_by_seq.items()):
        frames.sort(key=lambda row: float(row["t"]))
        refs = [row for row in frames if row["name"] in bundle_names]
        gaps = []
        for left, right in zip(refs, refs[1:]):
            gap = float(right["t"]) - float(left["t"])
            gaps.append(gap)
            worst_gaps.append({
                "sequence": seq,
                "left_ref": left["name"],
                "left_time_s": float(left["t"]),
                "right_ref": right["name"],
                "right_time_s": float(right["t"]),
                "gap_seconds": gap,
            })
        anchored = []
        for row in refs:
            xyz = np.asarray(bundle["refs"][row["name"]]["xyz_by_cell"])
            anchored.append(int(np.isfinite(xyz).all(axis=1).sum()))
        sequences.append({
            "sequence": seq,
            "extracted_frames": len(frames),
            "bundle_references": len(refs),
            "conversion_rate": len(refs) / len(frames) if frames else 0.0,
            "time_start_s": float(frames[0]["t"]),
            "time_end_s": float(frames[-1]["t"]),
            "max_registered_gap_s": max(gaps) if gaps else None,
            "registered_gap_p95_s": float(np.percentile(gaps, 95)) if gaps else None,
            "anchored_cells_median": float(np.median(anchored)) if anchored else None,
            "anchored_cells_p05": float(np.percentile(anchored, 5)) if anchored else None,
        })

    result = {
        "schema": "edm-reloc-coverage-audit/v1",
        "frame_manifest": str(Path(args.frame_manifest).resolve()),
        "bundle": str(Path(args.bundle).resolve()),
        "extracted_frames": sum(len(rows) for rows in frames_by_seq.values()),
        "bundle_references": len(bundle_names),
        "conversion_rate": len(bundle_names) / max(sum(len(rows) for rows in frames_by_seq.values()), 1),
        "sequences": sequences,
        "worst_registered_gaps": sorted(
            worst_gaps, key=lambda row: row["gap_seconds"], reverse=True
        )[:30],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

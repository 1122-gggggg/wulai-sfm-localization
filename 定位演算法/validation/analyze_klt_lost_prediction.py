#!/usr/bin/env python3
"""Score the KLT-in-LOST experiment from replay corpus runs.

Two questions, two arms.

``--shadow`` scores accuracy. The shadow chain runs on every accepted frame
without being re-seeded, so it ages exactly as it would through a long LOST
episode while EDM keeps supplying a fix on the same frame. That fix is the only
ground truth available -- on a frame that actually failed there is nothing to
measure against -- so drift is reported against horizon in seconds.

``--base``/``--lost`` score the regression: whether letting the prediction run
in LOST moves successes or verified at all. It should not, because the
prediction branch only writes to the reported info dict, but "should not" is
what the arm is for.

Reports both corpus numbers per runbook 5a: successes (which counts KLT-bridged
frames) and verified (frames that genuinely re-matched the map).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

#: Horizon buckets in seconds. 0.5 is the shipped cap; the rest ask what a
#: longer chain would have cost. At the measured 8 Hz cadence one frame is
#: ~0.125 s, so 0.5 s is only four frames.
HORIZON_BUCKETS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))

#: Site-calibrated scales the drift has to be judged against, from
#: 地圖檔/場域/river_site/site_profile.json and its EDM runtime profile.
SITE_ARRIVAL_TOLERANCE_U = 0.3175450813764711
SITE_MAX_JUMP_U = 0.9338766098022462


def _rows(directory: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(directory.glob("*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    return out


def _determinism(directory: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        summary = json.loads(path.read_text(encoding="utf-8")).get("summary") or {}
        out[path.stem] = summary.get("deterministic") or {}
    return out


def report_determinism(base: Path, lost: Path) -> None:
    """Bit-exact check, which is the ledger's bar for a no-change claim.

    The reference trace is every reference the matcher touched, in order. If the
    LOST prediction really only writes to the reported info dict, this hash is
    identical -- a far stronger statement than equal success counts.
    """
    base_d, lost_d = _determinism(base), _determinism(lost)
    print(f"\n# 逐位元對照   base={base}  lost={lost}")
    same = differ = 0
    for video in sorted(set(base_d) & set(lost_d)):
        b, l = base_d[video], lost_d[video]
        keys = sorted(set(b) | set(l))
        equal = all(b.get(k) == l.get(k) for k in keys)
        same += equal
        differ += not equal
        mark = "同" if equal else "**不同**"
        print(f"  {video:<16} {mark}")
        if not equal:
            for k in keys:
                if b.get(k) != l.get(k):
                    print(f"      {k}: {str(b.get(k))[:16]}... != {str(l.get(k))[:16]}...")
    print(f"  相同 {same} / 不同 {differ}")


def _verified(rows: list[dict]) -> int:
    return sum(
        1
        for row in rows
        if row.get("success")
        and isinstance(row.get("inliers"), (int, float))
        and row["inliers"] > 0
        and str(row.get("candidate_mode")) not in {"klt_fast", "klt_bridge", "None"}
    )


def _bucket(horizon: float) -> float:
    for edge in HORIZON_BUCKETS:
        if horizon <= edge:
            return edge
    return float("inf")


def report_accuracy(directory: Path) -> None:
    per_bucket: dict[float, list[float]] = {edge: [] for edge in HORIZON_BUCKETS}
    alive_checks = 0
    alive_true = 0
    for video, rows in _rows(directory).items():
        for row in rows:
            if row.get("klt_shadow_alive") is not None:
                alive_checks += 1
                alive_true += bool(row["klt_shadow_alive"])
            error = row.get("klt_shadow_error")
            horizon = row.get("klt_shadow_horizon_s")
            if error is None or horizon is None:
                continue
            error = float(error)
            horizon = float(horizon)
            if not (math.isfinite(error) and math.isfinite(horizon)):
                continue
            per_bucket[_bucket(horizon)].append(error)

    print(f"# KLT 預測精度 vs 時窗   ({directory})")
    print(f"  shadow 步進次數 {alive_checks}，其中鏈仍存活 {alive_true} "
          f"({alive_true / alive_checks:.1%})" if alive_checks else "  no shadow samples")
    print()
    print(f"{'時窗上限':>10}{'樣本':>8}{'p50':>10}{'p90':>10}{'max':>10}"
          f"{'p90/到達容差':>14}{'p90/max_jump':>14}")
    for edge in HORIZON_BUCKETS:
        values = sorted(per_bucket[edge])
        if not values:
            continue
        p50 = statistics.median(values)
        p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
        label = "inf" if edge == float("inf") else f"{edge:.2f}s"
        print(f"{label:>10}{len(values):>8}{p50:>10.4f}{p90:>10.4f}{values[-1]:>10.4f}"
              f"{p90 / SITE_ARRIVAL_TOLERANCE_U:>14.3f}{p90 / SITE_MAX_JUMP_U:>14.3f}")
    print()
    print(f"  參考尺度：到達容差 {SITE_ARRIVAL_TOLERANCE_U:.4f} u，"
          f"單幀允許跳動 max_jump {SITE_MAX_JUMP_U:.4f} u（皆為 map units）")


def report_regression(base: Path, lost: Path) -> None:
    base_rows = _rows(base)
    lost_rows = _rows(lost)
    print(f"\n# LOST 預測回歸   base={base}  lost={lost}")
    print(f"{'video':<16}{'frames':>8}{'succ base':>11}{'succ lost':>11}{'d':>5}"
          f"{'ver base':>10}{'ver lost':>10}{'d':>5}{'LOST 預測':>11}")
    totals = [0, 0, 0, 0, 0, 0]
    for video in sorted(set(base_rows) & set(lost_rows)):
        b, l = base_rows[video], lost_rows[video]
        sb = sum(1 for r in b if r.get("success"))
        sl = sum(1 for r in l if r.get("success"))
        vb, vl = _verified(b), _verified(l)
        predicted = sum(
            1
            for r in l
            if str(r.get("pose_status")) == "PREDICTED_ONLY"
            and str(r.get("prediction_state")) == "LOST"
        )
        totals = [totals[0] + len(b), totals[1] + sb, totals[2] + sl,
                  totals[3] + vb, totals[4] + vl, totals[5] + predicted]
        print(f"{video:<16}{len(b):>8}{sb:>11}{sl:>11}{sl - sb:>5}"
              f"{vb:>10}{vl:>10}{vl - vb:>5}{predicted:>11}")
    print(f"{'TOTAL':<16}{totals[0]:>8}{totals[1]:>11}{totals[2]:>11}"
          f"{totals[2] - totals[1]:>5}{totals[3]:>10}{totals[4]:>10}"
          f"{totals[4] - totals[3]:>5}{totals[5]:>11}")
    if totals[0]:
        print(f"  successes {totals[1] / totals[0]:.1%} -> {totals[2] / totals[0]:.1%}   "
              f"verified {totals[3] / totals[0]:.1%} -> {totals[4] / totals[0]:.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow", type=Path, help="shadow arm directory")
    parser.add_argument("--base", type=Path, help="control arm directory")
    parser.add_argument("--lost", type=Path, help="LOST-predict arm directory")
    args = parser.parse_args(argv)
    if args.shadow is not None:
        report_accuracy(args.shadow)
    if args.base is not None and args.lost is not None:
        report_regression(args.base, args.lost)
        report_determinism(args.base, args.lost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

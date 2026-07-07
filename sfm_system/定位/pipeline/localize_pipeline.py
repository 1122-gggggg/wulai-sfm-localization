#!/usr/bin/env python3
"""Localization pipeline entrypoint.

Modes:
  compare            Compare base vs current bundle on MP4 streams.
  production-stream  Run the production state-machine benchmark on frame dirs.
  sim                Run one of the simulator / dashboard scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import os.path
from pathlib import Path


LOC_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = LOC_ROOT.parent
DEPLOY = LOC_ROOT / "deploy_code" / "sfm_glomap_deploy"
SCRIPTS = LOC_ROOT / "source" / "sfm_glomap" / "scripts"
EVAL_CORE = LOC_ROOT / "validation" / "eval_stream_core.py"
PROD_STREAM = LOC_ROOT / "validation" / "benchmark_production_stream.py"
SIM_DIR = LOC_ROOT / "simulation"

DEFAULT_BASE = LOC_ROOT / "bundles" / "base_reloc_map_xfeat_tri.pt"
DEFAULT_FINAL = LOC_ROOT / "bundles" / "current_reloc_map_updated_v3.pt"
DEFAULT_BASE_CACHE = LOC_ROOT / "bundles" / "base_megaloc_cache_v3.npz"
DEFAULT_TEST = SYSTEM_ROOT / "更新地圖" / "inputs" / "補拍影片" / "test"
DEFAULT_MODEL = LOC_ROOT / "maps" / "base_glomap_fused_0"

# Interpreter for the subprocess runs (needs torch/pycolmap/cv2):
# env override > original machine's known-good python (if present) > current python.
_LEGACY_PY = "/usr/bin/python3.12"
DEFAULT_PYTHON = (
    os.environ.get("SFM_LOCALIZER_PYTHON")
    or (_LEGACY_PY if Path(_LEGACY_PY).exists() else sys.executable)
)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[localize_pipeline] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def py_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DEPLOY) + os.pathsep + str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def rel_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified localization / validation pipeline.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--mode", choices=["compare", "production-stream", "sim"], default="compare")
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--final", default=str(DEFAULT_FINAL))
    parser.add_argument("--base-megaloc-cache", default=str(DEFAULT_BASE_CACHE))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--resize", default="1280x720")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--min-conf", type=float, default=0.1)
    parser.add_argument("--min-inliers", type=int, default=50)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--query-dir", default="")
    parser.add_argument("--bundle", default=str(DEFAULT_FINAL))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--resize-width", type=int, default=1280)
    parser.add_argument("--sim-script", default="sim_harness.py")
    parser.add_argument("--min-success", type=float, default=0.90)
    parser.add_argument("--max-ok-to-fail", type=int, default=0)
    parser.add_argument("--max-final-fail-run", type=int, default=30)
    parser.add_argument("--allow-quality-fail", action="store_true")
    args, passthrough = parser.parse_known_args()

    if args.mode == "compare":
        out = Path(args.out_json) if args.out_json else LOC_ROOT / "outputs" / f"compare_{timestamp()}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python, str(EVAL_CORE),
            "--base", args.base,
            "--final", args.final,
            "--base-megaloc-cache", args.base_megaloc_cache,
            "--test-dir", args.test_dir,
            "--stride", str(args.stride),
            "--resize", args.resize,
            "--topk", str(args.topk),
            "--min-conf", str(args.min_conf),
            "--min-inliers", str(args.min_inliers),
            "--out-json", str(out),
        ] + passthrough
        run(cmd, py_env())
        quality_ok = write_quality_report(out, args.min_success, args.max_ok_to_fail, args.max_final_fail_run)
        if not quality_ok and not args.allow_quality_fail:
            raise SystemExit("localization quality gate failed")
        latest = LOC_ROOT / "outputs" / "latest_compare.json"
        rel_symlink(out, latest)
        print(f"[localize_pipeline] saved {out}", flush=True)
        return

    if args.mode == "production-stream":
        out = LOC_ROOT / "outputs" / f"production_stream_{timestamp()}.json"
        cmd = [
            args.python, str(PROD_STREAM),
            "--bundle", args.bundle,
            "--model", args.model,
            "--resize-width", str(args.resize_width),
            "--out", str(out),
        ]
        if args.query_dir:
            cmd += ["--query-dir", args.query_dir]
        cmd += passthrough
        out.parent.mkdir(parents=True, exist_ok=True)
        run(cmd, py_env())
        return

    script = Path(args.sim_script)
    if not script.is_absolute():
        script = SIM_DIR / script
    if not script.exists():
        raise SystemExit(f"simulation script not found: {script}")
    run([args.python, str(script), *passthrough], py_env())


def write_quality_report(result_json: Path, min_success: float, max_ok_to_fail: int,
                         max_final_fail_run: int) -> bool:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    failures = []
    lines = ["# Localization Quality Report", ""]
    lines.append(f"- Source: `{result_json}`")
    lines.append(
        f"- Gate: final_success >= {min_success:.0%}, or baseline-improved when base is below target; "
        f"ok_to_fail <= {max_ok_to_fail}, final_max_fail_run <= {max_final_fail_run}"
    )
    lines.append("")
    lines.append("| Set | Frames | Base success | Final success | Gain | ok->fail | maxfail final | Result |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        name = row.get("set", "")
        n = int(row.get("n", 0))
        base = float(row.get("base_success", 0.0))
        final = float(row.get("final_success", 0.0))
        gain = float(row.get("gain_pp", 0.0))
        reg = int(row.get("ok_to_fail", 0))
        maxfail = int(row.get("final_max_fail_run", 0))
        base_maxfail = int(row.get("base_max_fail_run", max_final_fail_run))
        regression_ok = reg <= max_ok_to_fail and maxfail <= max_final_fail_run
        absolute_ok = final >= min_success
        baseline_improved = (
            base < min_success
            and final >= base
            and maxfail <= base_maxfail
            and regression_ok
        )
        ok = regression_ok and (absolute_ok or baseline_improved)
        result = "PASS" if absolute_ok else "PASS_BASELINE_IMPROVED" if baseline_improved else "FAIL"
        if not ok:
            failures.append(name)
        lines.append(
            f"| {name} | {n} | {base:.1%} | {final:.1%} | {gain:+.1f}pp | "
            f"{reg} | {maxfail} | {result} |"
        )
    lines.append("")
    lines.append(f"Overall: {'PASS' if not failures else 'FAIL'}")
    if failures:
        lines.append(f"Failed sets: {', '.join(failures)}")
    report = result_json.with_suffix(".quality_report.md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest = LOC_ROOT / "outputs" / "latest_quality_report.md"
    rel_symlink(report, latest)
    print(f"[localize_pipeline] quality report {report}", flush=True)
    return not failures


if __name__ == "__main__":
    main()

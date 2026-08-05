#!/usr/bin/env python3
"""Run the complete no-flight validation matrix and write a JSON receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_PYTHON = ROOT / ".venv/bin/python"
PARROT_ROOT = ROOT / "模擬器/parrot_stimulate"
PARROT_PYTHON = PARROT_ROOT / ".venv/bin/python"
P119_VIDEO = Path(
    os.environ.get(
        "SFM_P119_VIDEO",
        str(ROOT / "模擬器/測試影片/P1190119.MP4"),
    )
).expanduser()
P119_QUALITY_BASELINE = ROOT / "定位演算法/validation/baselines/p119_edm_quality.json"
RELEASE_FILES = (
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-test.txt",
    "執行環境/requirements_runtime.txt",
    "tools/install_runtime.sh",
    "tools/test_clean_install.sh",
    "tools/test_portable_runtime.sh",
    "tools/simulated_ui_smoke.sh",
    "tools/simulator_preflight.py",
    "tools/export_simulator_package.py",
    "tools/package_manifest.py",
    "控制介面程式/影片模擬串流/啟動.sh",
    "控制介面程式/影片模擬串流/選擇啟動.py",
    "控制介面程式/operator_interface/flight_operator_app.py",
    "控制介面程式/operator_interface/live_localizer_worker.py",
    "控制介面程式/operator_interface/scale_free_control_adapter.py",
    "模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py",
    "定位演算法/deploy_code/sfm_glomap_deploy/edm_matcher.py",
    "定位演算法/deploy_code/sfm_glomap_deploy/reloc_localizer_edm.py",
    "定位演算法/deploy_code/runtime/EDM/weights/edm_outdoor.ckpt",
    "執行環境/torch_hub_cache/gmberton_MegaLoc_main/hubconf.py",
    "執行環境/torch_hub_cache/gmberton_MegaLoc_main/megaloc_model.py",
    "執行環境/torch_hub_cache/checkpoints/megaloc/7cb9f7970d366fdf059963d04d372e503e8e9df9/model.safetensors",
)


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    cwd: str
    timeout_s: int


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _git_metadata() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run("status", "--porcelain").splitlines()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "changed_path_count": len(status),
    }


def _portable_metadata(package_root: Path) -> dict[str, object]:
    package_root = package_root.expanduser().resolve()
    files: dict[str, object] = {}
    missing: list[str] = []
    for name in ("PORTABLE_PACKAGE.json", "MANIFEST.tsv", "SHA256SUMS"):
        path = package_root / name
        if not path.is_file():
            files[name] = {"status": "missing"}
            missing.append(name)
            continue
        files[name] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}

    source_matches = False
    metadata_error = ""
    metadata_path = package_root / "PORTABLE_PACKAGE.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_matches = metadata.get("source_release", {}) == {
                "manifest_sha256": _sha256(ROOT / "MANIFEST.tsv"),
                "sha256sums_sha256": _sha256(ROOT / "SHA256SUMS"),
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            metadata_error = str(exc)
    return {
        "path": str(package_root),
        "complete": not missing and source_matches and not metadata_error,
        "missing": missing,
        "source_release_matches": source_matches,
        "metadata_error": metadata_error,
        "files": files,
    }


def _release_metadata(portable_package: Path | None = None) -> dict[str, object]:
    files: dict[str, object] = {}
    missing: list[str] = []
    for relative in RELEASE_FILES:
        path = ROOT / relative
        if not path.is_file():
            files[relative] = {"status": "missing"}
            missing.append(relative)
            continue
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    for name in ("PORTABLE_PACKAGE.json", "MANIFEST.tsv", "SHA256SUMS"):
        path = ROOT / name
        if path.is_file():
            files[name] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    portable = (
        _portable_metadata(portable_package) if portable_package is not None else None
    )
    return {
        "schema": "sfm-release-inputs/v1",
        "complete": not missing and (portable is None or bool(portable["complete"])),
        "missing": missing,
        "files": files,
        "portable_package": portable,
    }


def _steps(
    *,
    p119: bool,
    accept_p119: bool,
    p119_quality: bool,
    quality_out: Path | None,
    portable_package: Path | None = None,
    clean_install: bool = False,
    ui_smoke: bool = False,
) -> list[Step]:
    uv = shutil.which("uv") or "uv"
    steps = [
        Step("root_pytest", (str(ROOT_PYTHON), "-m", "pytest", "-q"), str(ROOT), 2400),
        Step(
            "parrot_pytest",
            (str(PARROT_PYTHON), "-m", "pytest", "-q"),
            str(PARROT_ROOT),
            1200,
        ),
        Step(
            "parrot_ruff_check",
            (str(PARROT_PYTHON), "-m", "ruff", "check", "."),
            str(PARROT_ROOT),
            300,
        ),
        Step(
            "parrot_ruff_format",
            (str(PARROT_PYTHON), "-m", "ruff", "format", "--check", "."),
            str(PARROT_ROOT),
            300,
        ),
        Step(
            "runtime_mirrors",
            (
                str(ROOT_PYTHON),
                str(ROOT / "定位演算法/validation/check_runtime_mirrors.py"),
            ),
            str(ROOT),
            120,
        ),
        Step(
            "workspace_layout",
            (
                str(ROOT_PYTHON),
                str(ROOT / "tools/workspace_audit.py"),
                "--strict-output-names",
                "--no-sizes",
            ),
            str(ROOT),
            120,
        ),
        Step(
            "portable_manifest",
            (
                str(ROOT_PYTHON),
                str(ROOT / "tools/package_manifest.py"),
                "verify",
                "--root",
                str(ROOT),
            ),
            str(ROOT),
            600,
        ),
        Step(
            "flight_selftest",
            (
                str(ROOT_PYTHON),
                str(ROOT / "控制介面程式/mission_pipeline.py"),
                "--mode",
                "flight-selftest",
            ),
            str(ROOT),
            300,
        ),
        Step(
            "root_dependency_check",
            (str(ROOT_PYTHON), "-m", "pip", "check"),
            str(ROOT),
            120,
        ),
        Step(
            "parrot_lock_check",
            (uv, "sync", "--locked", "--offline", "--check"),
            str(PARROT_ROOT),
            300,
        ),
        Step(
            "parrot_preflight",
            (str(PARROT_PYTHON), "-m", "anafi_pcmd_sim", "preflight"),
            str(PARROT_ROOT),
            120,
        ),
        Step(
            "profile_asset_validation",
            (
                str(ROOT_PYTHON),
                str(ROOT / "控制介面程式/validate_system_profiles.py"),
            ),
            str(ROOT),
            300,
        ),
        Step(
            "portable_simulator_preflight",
            (
                str(ROOT_PYTHON),
                str(ROOT / "tools/simulator_preflight.py"),
                "--workspace-root",
                str(ROOT),
                "--site-profile",
                str(ROOT / "控制介面程式/site_profiles/river_site_edm.json"),
                "--video",
                str(ROOT / "模擬器/測試影片/河濱_P1180118_first_2s.mp4"),
                "--check-runtime",
                "--full-runtime",
            ),
            str(ROOT),
            1800,
        ),
        Step(
            "cuda_production_smoke",
            (
                str(ROOT_PYTHON),
                str(ROOT / "定位演算法/validation/cuda_production_smoke.py"),
            ),
            str(ROOT),
            300,
        ),
        Step(
            "offline_model_smoke",
            (
                str(ROOT_PYTHON),
                str(ROOT / "定位演算法/validation/offline_model_smoke.py"),
                "--model",
                "edm",
            ),
            str(ROOT),
            1800,
        ),
    ]
    if portable_package is not None:
        package_root = portable_package.expanduser().resolve()
        steps.insert(
            7,
            Step(
                "portable_output_manifest",
                (
                    str(ROOT_PYTHON),
                    str(package_root / "tools/package_manifest.py"),
                    "verify",
                    "--root",
                    str(package_root),
                ),
                str(package_root),
                600,
            ),
        )
    if clean_install:
        steps.append(
            Step(
                "clean_install_full_runtime",
                ("bash", str(ROOT / "tools/test_clean_install.sh")),
                str(ROOT),
                3600,
            )
        )
    if ui_smoke:
        steps.append(
            Step(
                "simulated_ui_smoke",
                ("bash", str(ROOT / "tools/simulated_ui_smoke.sh")),
                str(ROOT),
                180,
            )
        )
    if portable_package is not None and clean_install and ui_smoke:
        steps.append(
            Step(
                "portable_clean_install_ui_pose",
                (
                    "bash",
                    str(ROOT / "tools/test_portable_runtime.sh"),
                    str(portable_package.expanduser().resolve()),
                ),
                str(ROOT),
                3600,
            )
        )
    if p119:
        argv = [
            str(ROOT_PYTHON),
            str(ROOT / "控制介面程式/validate_p119_source.py"),
        ]
        if accept_p119:
            argv.append("--accept-known-incomplete")
        steps.append(Step("p119_integrity", tuple(argv), str(ROOT), 600))
    if p119_quality:
        if quality_out is None:
            raise ValueError("quality_out is required for P119 quality validation")
        steps.append(
            Step(
                "p119_quality",
                (
                    str(ROOT_PYTHON),
                    str(ROOT / "定位演算法/validation/benchmark_edm_site_replay.py"),
                    "--site-profile",
                    str(ROOT / "控制介面程式/site_profiles/river_site_edm.json"),
                    "--video",
                    str(P119_VIDEO),
                    "--out",
                    str(quality_out),
                    "--quality-baseline",
                    str(P119_QUALITY_BASELINE),
                    "--accept-known-incomplete",
                    "--require-cuda",
                ),
                str(ROOT),
                1800,
            )
        )
    return steps


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p119-integrity", action="store_true")
    parser.add_argument(
        "--p119-quality",
        action="store_true",
        help="run the complete pinned P119 EDM replay and fail on quality regression",
    )
    parser.add_argument(
        "--accept-p119-known-incomplete",
        action="store_true",
        help="accept only the exact pinned 2935-declared/2934-decoded source",
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=ROOT / "outputs/validation_receipts",
    )
    default_portable = ROOT.parent / f"{ROOT.name}_portable"
    parser.add_argument(
        "--portable-package",
        type=Path,
        default=default_portable if default_portable.is_dir() else None,
        help="verify this exported package and bind its identity to the receipt",
    )
    parser.add_argument(
        "--clean-install",
        action="store_true",
        help="rebuild a clean CPython 3.10 venv and run full-runtime preflight",
    )
    parser.add_argument(
        "--ui-smoke",
        action="store_true",
        help="launch the simulated GUI until its localization feed is ready",
    )
    args = parser.parse_args()
    if args.accept_p119_known_incomplete and not args.p119_integrity:
        parser.error("--accept-p119-known-incomplete requires --p119-integrity")
    if args.p119_quality and not (
        args.p119_integrity and args.accept_p119_known_incomplete
    ):
        parser.error(
            "--p119-quality requires --p119-integrity and "
            "--accept-p119-known-incomplete"
        )
    if not ROOT_PYTHON.is_file() or not PARROT_PYTHON.is_file():
        parser.error("both pinned Python environments must exist")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt_dir = args.receipt_dir.expanduser().resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"validation_{stamp}.json"
    log_dir = receipt_dir / f"validation_{stamp}_logs"
    log_dir.mkdir(mode=0o750)
    started = time.time()
    portable_package = (
        args.portable_package.expanduser().resolve()
        if args.portable_package is not None
        else None
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "host": platform.node(),
        "platform": platform.platform(),
        "system_spec_sha256": _sha256(ROOT / "文件/SYSTEM_SPEC.md"),
        "git": _git_metadata(),
        "release_inputs": _release_metadata(portable_package),
        "p119_waiver_requested": bool(args.accept_p119_known_incomplete),
        "steps": [],
    }
    _write_receipt(receipt_path, payload)

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "PYTHONNOUSERSITE": "1",
            "SFM_WORKSPACE_ROOT": str(ROOT),
            "SFM_TORCH_HUB_CACHE": str(ROOT / "執行環境/torch_hub_cache"),
            "SFM_UI_PYTHON": str(ROOT_PYTHON),
            "SFM_LOCALIZER_PYTHON": str(ROOT_PYTHON),
            "SFM_SITE_PROFILE": str(
                ROOT / "控制介面程式/site_profiles/river_site_edm.json"
            ),
        }
    )
    release_inputs = payload["release_inputs"]
    assert isinstance(release_inputs, dict)
    failed = not bool(release_inputs.get("complete"))
    for index, step in enumerate(
        _steps(
            p119=bool(args.p119_integrity),
            accept_p119=bool(args.accept_p119_known_incomplete),
            p119_quality=bool(args.p119_quality),
            quality_out=log_dir / "p119_quality.json",
            portable_package=portable_package,
            clean_install=bool(args.clean_install),
            ui_smoke=bool(args.ui_smoke),
        ),
        start=1,
    ):
        print(f"[validation {index}] {step.name}", flush=True)
        step_started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                step.argv,
                cwd=step.cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=step.timeout_s,
                check=False,
            )
            exit_code = completed.returncode
            output = completed.stdout + completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = (
                exc.stdout.decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode()
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            output = stdout + stderr + f"\nTIMEOUT after {step.timeout_s}s\n"
        duration = time.monotonic() - step_started
        log_path = log_dir / f"{index:02d}_{step.name}.log"
        log_path.write_text(output, encoding="utf-8")
        if output:
            print(output.rstrip(), flush=True)
        ok = exit_code == 0
        failed = failed or not ok
        payload["steps"].append(  # type: ignore[union-attr]
            {
                **asdict(step),
                "argv": list(step.argv),
                "duration_s": round(duration, 3),
                "exit_code": exit_code,
                "timed_out": timed_out,
                "ok": ok,
                "log": str(log_path),
            }
        )
        _write_receipt(receipt_path, payload)

    payload.update(
        status="failed" if failed else "passed",
        finished_utc=datetime.now(timezone.utc).isoformat(),
        duration_s=round(time.time() - started, 3),
        receipt=str(receipt_path),
    )
    _write_receipt(receipt_path, payload)
    print(f"[validation] receipt={receipt_path} status={payload['status']}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

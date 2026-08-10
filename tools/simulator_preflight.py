#!/usr/bin/env python3
"""Fail-closed preflight for the portable simulated-stream interface."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MEGALOC_REVISION = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
MEGALOC_SHA256 = "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
MEGALOC_HUBCONF_SHA256 = (
    "0ebf9fc9c455ca38b9e52c69bcfee4b136a0f8872fdfc4307d00469013228b98"
)
MEGALOC_MODEL_SOURCE_SHA256 = (
    "3cbf1d20515b1da423998a8edab787031eaa7bb273c5a86a5c41c4f6d84e2a6d"
)
EDM_CHECKPOINT_SHA256 = (
    "f686bebdd9705bf6918621a1a83695f83d698cbd8c3eed932847fe3678d13a97"
)
SCALE_FREE_CORE = "模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py"
REQUIRED_GPU_SUBSTRING = "rtx 5060"
SCIPY_LOCK_PACKAGE = "scipy"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha(path: Path, expected: str, label: str, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing {label}: {path}")
        return
    actual = _sha256(path)
    if actual != expected:
        failures.append(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _check_video(video: Path, failures: list[str]) -> dict[str, str]:
    if not video.is_file() or video.stat().st_size <= 0:
        failures.append(f"video is missing or empty: {video}")
        return {}
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        failures.append("missing system command: ffprobe (install ffmpeg)")
        return {}
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        failures.append(
            f"ffprobe cannot read video: {video}: {completed.stderr.strip()}"
        )
        return {}
    return {"duration_s": completed.stdout.strip()}


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _resolved_env_path(name: str, *, resolve_symlinks: bool = True) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if resolve_symlinks else _absolute_path(path)


def _check_environment_contract(
    root: Path, profile_path: Path, failures: list[str], runtime: dict[str, object]
) -> None:
    venv_config = Path(sys.prefix) / "pyvenv.cfg"
    system_site_packages = False
    if venv_config.is_file():
        for line in venv_config.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip().lower() == "include-system-site-packages":
                system_site_packages = value.strip().lower() == "true"
                break
    runtime["system_site_packages"] = system_site_packages
    if system_site_packages:
        failures.append(
            f"runtime venv has include-system-site-packages=true: {venv_config}"
        )

    expected_paths = {
        "SFM_WORKSPACE_ROOT": root.resolve(),
        "SFM_TORCH_HUB_CACHE": (root / "執行環境/torch_hub_cache").resolve(),
        "SFM_UI_PYTHON": _absolute_path(Path(sys.executable)),
        "SFM_LOCALIZER_PYTHON": _absolute_path(Path(sys.executable)),
        "SFM_SITE_PROFILE": profile_path.resolve(),
    }
    for name, expected in expected_paths.items():
        actual = _resolved_env_path(
            name,
            resolve_symlinks=name not in {"SFM_UI_PYTHON", "SFM_LOCALIZER_PYTHON"},
        )
        if actual is not None and actual != expected:
            failures.append(
                f"{name} does not match selected runtime: {actual} != {expected}"
            )
        runtime[f"{name.lower()}_resolved"] = str(actual or expected)


def _check_runtime(
    root: Path, profile_path: Path, failures: list[str]
) -> dict[str, object]:
    runtime: dict[str, object] = {"python": sys.version.split()[0]}
    _check_environment_contract(root, profile_path, failures, runtime)
    if sys.version_info[:2] != (3, 10):
        failures.append(f"Python 3.10 required; got {sys.version.split()[0]}")
    if shutil.which("ffmpeg") is None:
        failures.append("missing system command: ffmpeg")
    for module_name in ("numpy", "cv2", "PIL", "torch", "pycolmap", "tkinter"):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            failures.append(f"missing Python module {module_name}: {exc}")
            continue
        if module_name == "torch":
            runtime["torch"] = getattr(module, "__version__", "unknown")
            if not module.cuda.is_available():
                failures.append("CUDA is unavailable; production EDM requires CUDA")
            else:
                device = module.cuda.get_device_name(0)
                runtime["cuda_device"] = device
                if REQUIRED_GPU_SUBSTRING not in device.lower():
                    failures.append(
                        f"production runtime requires an NVIDIA RTX 5060; got {device}"
                    )

    edm_checkpoint = (
        root / "定位演算法/deploy_code/runtime/EDM/weights/edm_outdoor.ckpt"
    )
    _check_sha(edm_checkpoint, EDM_CHECKPOINT_SHA256, "EDM checkpoint", failures)
    megaloc_root = root / "執行環境/torch_hub_cache/gmberton_MegaLoc_main"
    _check_sha(
        megaloc_root / "hubconf.py",
        MEGALOC_HUBCONF_SHA256,
        "MegaLoc hubconf",
        failures,
    )
    _check_sha(
        megaloc_root / "megaloc_model.py",
        MEGALOC_MODEL_SOURCE_SHA256,
        "MegaLoc model source",
        failures,
    )
    megaloc_weights = (
        root
        / "執行環境/torch_hub_cache/checkpoints/megaloc"
        / MEGALOC_REVISION
        / "model.safetensors"
    )
    _check_sha(megaloc_weights, MEGALOC_SHA256, "MegaLoc weights", failures)
    edm_config = (
        root / "定位演算法/deploy_code/runtime/EDM/configs/edm/outdoor/edm_base.py"
    )
    if not edm_config.is_file():
        failures.append(f"missing EDM config: {edm_config}")
    deploy_dir = root / "定位演算法/deploy_code/sfm_glomap_deploy"
    edm_repo = root / "定位演算法/deploy_code/runtime/EDM"
    for path in (deploy_dir, edm_repo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        edm_matcher = importlib.import_module("edm_matcher")
        edm_matcher._import_edm()
        runtime["edm_import"] = "ok"
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        failures.append(f"EDM runtime import failed: {exc}")
    return runtime


def _collision_monitor_status(
    root: Path,
    failures: list[str],
    *,
    production_required: bool,
) -> dict[str, object]:
    """Report the effective sparse-cloud monitor policy.

    The controller has a guarded SciPy import, so importing SciPy from an
    operator's ambient environment is not enough to make this capability
    reproducible.  A clean runtime can only rely on packages present in the
    hash-locked runtime requirements.  The monitor is currently a warning-only
    research aid, not an approved production collision-protection layer.
    """
    lock_path = root / "requirements/runtime-lock.txt"
    hash_locked = False
    lock_error = ""
    if lock_path.is_file():
        try:
            hash_locked = any(
                line.lstrip().startswith(f"{SCIPY_LOCK_PACKAGE}==")
                for line in lock_path.read_text(encoding="utf-8").splitlines()
            )
        except OSError as exc:
            lock_error = f"cannot read requirements/runtime-lock.txt: {exc}"

    runtime_import = False
    import_error = ""
    try:
        scipy_spatial = importlib.import_module("scipy.spatial")
        runtime_import = getattr(scipy_spatial, "cKDTree", None) is not None
        if not runtime_import:
            import_error = "scipy.spatial.cKDTree is unavailable"
    except Exception as exc:  # noqa: BLE001 - report optional monitor as unavailable
        import_error = str(exc)

    reasons: list[str] = []
    if not hash_locked:
        reasons.append(
            "scipy is not present in requirements/runtime-lock.txt; clean lock-only installs "
            "must treat SparseCloudCollisionMonitor as unavailable"
        )
    if lock_error:
        reasons.append(lock_error)
    if not runtime_import:
        reasons.append(import_error or "scipy.spatial.cKDTree import failed")
    reasons.append(
        "SparseCloudCollisionMonitor is not an approved production safety layer; "
        "collision_protection_claim=false"
    )

    effective_available = hash_locked and runtime_import
    if production_required:
        if not hash_locked:
            failures.append(
                "production collision monitor required but scipy is absent from "
                "requirements/runtime-lock.txt; preflight is fail-closed"
            )
        if not runtime_import:
            failures.append(
                "production collision monitor required but scipy.spatial.cKDTree "
                "is unavailable; preflight is fail-closed"
            )
        failures.append(
            "SparseCloudCollisionMonitor is not an approved production safety layer; "
            "production preflight is fail-closed"
        )

    return {
        "available": effective_available,
        "status": "available_non_production" if effective_available else "unavailable",
        "runtime_import": runtime_import,
        "hash_locked": hash_locked,
        "required_for_production": production_required,
        "production_safety": False,
        "collision_protection_claim": False,
        "reason": " ".join(reasons),
    }


def _check_ply(path: Path, failures: list[str]) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        failures.append(f"map PLY is missing or empty: {path}")
        return
    try:
        with path.open("rb") as stream:
            header = stream.read(64 * 1024)
    except OSError as exc:
        failures.append(f"cannot read map PLY {path}: {exc}")
        return
    if not header.startswith(b"ply\n") or b"end_header" not in header:
        failures.append(f"map PLY header is invalid: {path}")


def _check_full_runtime(
    root: Path,
    profile,
    failures: list[str],
    runtime: dict[str, object],
) -> None:
    if profile is None:
        failures.append("full runtime check requires a valid site profile")
        return
    for path, label in (
        (profile.map_ply, "map PLY"),
        (profile.localization_bundle, "localization bundle"),
        (profile.map_reference_poses, "map reference poses"),
        (profile.localizer_deploy_dir, "localizer deploy directory"),
        (profile.localizer_profile, "localizer profile"),
    ):
        if path is not None and not path.exists():
            failures.append(f"missing {label}: {path}")
    _check_ply(profile.map_ply, failures)
    if profile.query_camera is None:
        failures.append("site profile has no query camera")

    deploy_dir = profile.localizer_deploy_dir or (
        root / "定位演算法/deploy_code/sfm_glomap_deploy"
    )
    expected_deploy_dir = (root / "定位演算法/deploy_code/sfm_glomap_deploy").resolve()
    if deploy_dir.resolve() != expected_deploy_dir:
        failures.append(
            "site profile localizer_deploy_dir is outside the fixed package: "
            f"{deploy_dir} != {expected_deploy_dir}"
        )
    if profile.localizer_profile is not None:
        expected_profile_root = (root / "定位演算法/configs").resolve()
        if not profile.localizer_profile.resolve().is_relative_to(expected_profile_root):
            failures.append(
                "site profile localizer_profile is outside the fixed package: "
                f"{profile.localizer_profile}"
            )
    edm_repo = root / "定位演算法/deploy_code/runtime/EDM"
    for path in (deploy_dir, edm_repo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        from production_localizer_factory import validate_camera_tuple

        camera = profile.query_camera
        if camera is not None:
            validate_camera_tuple(
                (camera.model, camera.width, camera.height, list(camera.params))
            )
            runtime["camera"] = f"{camera.model} {camera.width}x{camera.height}"
    except Exception as exc:  # noqa: BLE001 - report clean-install failures
        failures.append(f"camera/deploy validation failed: {exc}")

    try:
        from reloc_localizer_edm import EDMRelocMap

        expected = profile.asset_sha256.localization_bundle
        reloc_map = EDMRelocMap.load(
            profile.localization_bundle, expected_sha256=expected
        )
        runtime["bundle_refs"] = len(reloc_map.ref_names)
        del reloc_map
    except Exception as exc:  # noqa: BLE001 - report corrupt bundle failures
        failures.append(f"EDM bundle load failed: {exc}")

    try:
        import numpy as np
        from edm_matcher import EDMMatcher

        matcher = EDMMatcher(device="cuda", fp16=True)
        runtime["edm_model"] = "loaded"
        del matcher
    except Exception as exc:  # noqa: BLE001 - report model/runtime failures
        failures.append(f"EDM CUDA matcher load failed: {exc}")

    try:
        from reloc_localizer_edm import MegaLocQuery

        extractor = MegaLocQuery(device="cuda")
        descriptor = extractor.extract_one(np.zeros((322, 322, 3), dtype=np.uint8))
        if descriptor.shape != (8448,) or not np.isfinite(descriptor).all():
            raise ValueError(f"unexpected MegaLoc descriptor: {descriptor.shape}")
        runtime["megaloc_model"] = "loaded_and_inferred"
        del extractor, descriptor
    except Exception as exc:  # noqa: BLE001 - report model/runtime failures
        failures.append(f"MegaLoc CUDA load/inference failed: {exc}")

    operator_dir = root / "控制介面程式/operator_interface"
    control_dir = root / "控制介面程式"
    for path in (control_dir, operator_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for module_name in ("flight_operator_app", "live_localizer_worker"):
        try:
            importlib.import_module(module_name)
            runtime[f"import_{module_name}"] = "ok"
        except Exception as exc:  # noqa: BLE001 - report UI/worker import failures
            failures.append(f"{module_name} import failed: {exc}")


def run_preflight(
    *,
    root: Path,
    profile_path: Path,
    video_path: Path,
    check_runtime: bool,
    full_runtime: bool = False,
    require_collision_monitor: bool = False,
) -> dict[str, object]:
    failures: list[str] = []
    profile = None
    control_dir = root / "控制介面程式"
    scale_free_core = root / SCALE_FREE_CORE
    if not scale_free_core.is_file():
        failures.append(
            f"missing authoritative scale-free control core: {scale_free_core}"
        )
    if str(control_dir) not in sys.path:
        sys.path.insert(0, str(control_dir))
    try:
        from site_profile import load_site_profile

        profile = load_site_profile(profile_path)
        profile_report = {
            "site_id": profile.site_id,
            "profile": str(profile.source),
            "map_ply": str(profile.map_ply),
            "localization_bundle": str(profile.localization_bundle),
        }
    except (ImportError, OSError, TypeError, ValueError) as exc:
        failures.append(f"site profile invalid: {exc}")
        profile_report = {}
    video_report = _check_video(video_path, failures)
    runtime_report = (
        _check_runtime(root, profile_path, failures) if check_runtime else {}
    )
    runtime_report["collision_monitor"] = _collision_monitor_status(
        root,
        failures,
        production_required=require_collision_monitor,
    )
    if full_runtime:
        if not check_runtime:
            failures.append("full runtime check requires --check-runtime")
        else:
            _check_full_runtime(root, profile, failures, runtime_report)
    return {
        "ok": not failures,
        "root": str(root.resolve()),
        "profile": profile_report,
        "video": {"path": str(video_path.resolve()), **video_report},
        "runtime": runtime_report,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=str(_workspace_root()))
    parser.add_argument("--site-profile", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument(
        "--require-collision-monitor",
        action="store_true",
        help="require a hash-locked, approved collision monitor; otherwise fail closed",
    )
    parser.add_argument(
        "--full-runtime",
        action="store_true",
        help="load the selected bundle, EDM CUDA model, MegaLoc model, GUI and worker",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_preflight(
        root=Path(args.workspace_root).expanduser().resolve(),
        profile_path=Path(args.site_profile).expanduser().resolve(),
        video_path=Path(args.video).expanduser().resolve(),
        check_runtime=bool(args.check_runtime or args.full_runtime),
        full_runtime=bool(args.full_runtime),
        require_collision_monitor=bool(args.require_collision_monitor),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"[simulator-preflight] {'OK' if report['ok'] else 'FAIL'}")
        if report["profile"]:
            print(f"[simulator-preflight] site={report['profile']['site_id']}")
        print(f"[simulator-preflight] video={report['video']['path']}")
        collision_monitor = report["runtime"].get("collision_monitor")
        if isinstance(collision_monitor, dict):
            print(
                "[simulator-preflight] collision-monitor="
                f"{collision_monitor['status']} "
                "production_safety=false collision_protection_claim=false"
            )
        for failure in report["failures"]:
            print(f"[simulator-preflight] ERROR: {failure}", file=sys.stderr)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()

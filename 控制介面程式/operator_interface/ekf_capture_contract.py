"""Immutable calibration and reference-frame evidence for offline fusion work."""

from pathlib import Path
import hashlib
import json

import numpy as np


def camera_calibration_evidence(intrinsics: dict, camera: dict) -> dict:
    """Check numerical compatibility without claiming an unmeasured image domain."""
    issues = []
    matches = False
    try:
        matrix = np.asarray(intrinsics["K"], dtype=float)
        width, height = intrinsics["image_width"], intrinsics["image_height"]
        valid = (matrix.shape == (3, 3) and np.isfinite(matrix).all()
                 and matrix[0, 0] > 0 and matrix[1, 1] > 0
                 and isinstance(width, int) and isinstance(height, int)
                 and not isinstance(width, bool) and not isinstance(height, bool)
                 and width > 0 and height > 0)
        if valid:
            valid = bool(np.allclose(matrix[2], [0., 0., 1.], rtol=0., atol=1e-9)
                         and abs(matrix[0, 1]) < 1e-9 and abs(matrix[1, 0]) < 1e-9)
        params = np.asarray(camera.get("params", []), dtype=float)
        matches = bool(valid and camera.get("model") == "PINHOLE"
                       and camera.get("width") == width and camera.get("height") == height
                       and params.shape == (4,) and np.isfinite(params).all()
                       and np.allclose(params, [matrix[0, 0], matrix[1, 1],
                                              matrix[0, 2], matrix[1, 2]]))
    except (KeyError, TypeError, ValueError, IndexError):
        valid = False
    if not valid:
        issues.append("intrinsics dimensions/K are missing or invalid")
    if not matches:
        issues.append("supplied intrinsics do not match the active PINHOLE camera")
    try:
        distortion = np.asarray(intrinsics.get("dist", []), dtype=float)
        distortion_valid = bool(distortion.shape == (5,) and np.isfinite(distortion).all())
    except (TypeError, ValueError):
        distortion_valid = False
    if not distortion_valid:
        issues.append("expected five finite OpenCV distortion coefficients")
    return {"intrinsics_valid": bool(valid), "intrinsics_match_active_pinhole": matches,
            "distortion_coefficients_valid": distortion_valid,
            "distortion_domain_verified": False, "issues": issues}


def verify_image_domain(provenance: dict, intrinsic_sha256: str | None) -> dict:
    verified = bool(intrinsic_sha256 and provenance.get("intrinsics_sha256") == intrinsic_sha256
                    and provenance.get("image_domain") == "anafi_raw_stream"
                    and provenance.get("confirmed_by") == "operator")
    return {"distortion_domain_verified": verified,
            "image_domain": "anafi_raw_stream" if verified else "unknown",
            "domain_evidence": "operator confirmation, digest-bound" if verified else "unverified",
            "host_rectification_applied": False,
            "pinhole_image_domain_verified": False}


def write_capture_contract(directory: Path, *, root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[2]
    directory.mkdir(parents=True, exist_ok=True)
    issues = []

    def snapshot(path, filename):
        try:
            data = Path(path).read_bytes()
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            (directory / filename).write_bytes(data)
            return value, {"source": str(path), "file": filename,
                           "sha256": hashlib.sha256(data).hexdigest()}
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"{filename}: {exc}")
            return {}, None

    try:
        manifest = json.loads((directory.parent / "session_manifest.json").read_text())
    except (OSError, ValueError):
        manifest = {}
    intrinsics, intrinsic_receipt = snapshot(root / "intrinsics.json", "intrinsics.json")
    provenance, provenance_receipt = {}, None
    if (root / "intrinsics_provenance.json").is_file():
        provenance, provenance_receipt = snapshot(
            root / "intrinsics_provenance.json", "intrinsics_provenance.json")
    domain = verify_image_domain(provenance, (intrinsic_receipt or {}).get("sha256"))
    profile_path = Path(manifest.get("site_profile") or directory / "missing_site_profile")
    profile, profile_receipt = snapshot(profile_path, "site_profile.json")
    assets = {}
    for field in ("localizer_profile", "map_align"):
        if profile.get(field):
            path = Path(profile[field])
            if not path.is_absolute():
                path = profile_path.parent / path
            _value, assets[field] = snapshot(path, field + ".json")
    camera = profile.get("query_camera") or {}
    evidence = camera_calibration_evidence(intrinsics, camera)
    issues.extend(evidence["issues"])
    contract = {
        "schema_version": 1,
        "purpose": "offline loosely coupled visual + firmware telemetry evaluation",
        "position_reference": "camera_center",
        "coordinate_frame": profile.get("coordinate_frame"),
        "camera_body_translation_m": [0., 0., 0.],
        "translation_basis": "operator-requested coincident-center approximation; not measured",
        "orientation_reference": "camera optical pose; firmware body Euler and gimbal recorded separately",
        "velocity_frame": "NED; firmware-fused metres/second",
        "altitude_reference": "firmware takeoff-relative altitude; AGL recorded separately; not independent truth",
        "metric_scale_status": "requires independent calibration; map units are not metres",
        "independent_ground_truth_available": False,
        "ekf_running": False,
        "covariance_status": "not supplied by firmware; EKF covariance/innovations do not exist yet",
        "clock": "host monotonic; SDK event receipt mapping is not a sensor exposure guarantee",
        "recording_encoding": "JPEG; lossy reproduction of submitted RGB, shared input for offline A/B",
        "intrinsics": intrinsic_receipt,
        "intrinsics_provenance": provenance_receipt,
        "site_profile": profile_receipt,
        "assets": assets,
        "active_query_camera": camera,
        "intrinsics_match_active_pinhole": evidence["intrinsics_match_active_pinhole"],
        "intrinsics_valid": evidence["intrinsics_valid"],
        "distortion_coefficients_valid": evidence["distortion_coefficients_valid"],
        "supplied_distortion_coefficients": (
            intrinsics.get("dist") if evidence["distortion_coefficients_valid"] else None),
        **domain,
        "distortion_note": "host path resizes without undistorting; PINHOLE assumes rectified input; supplied distortion is preserved, not silently applied",
        "source_files_at_session_start": manifest.get("debug_contract", {}).get("source_files_at_session_start", {}),
        "issues": issues,
    }
    (directory / "capture_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return contract

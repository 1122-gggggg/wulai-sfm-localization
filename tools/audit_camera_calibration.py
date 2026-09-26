#!/usr/bin/env python3
"""Audit raw-stream calibration and quantify same-K rectification, offline only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "控制介面程式/operator_interface"))
from ekf_capture_contract import camera_calibration_evidence, verify_image_domain  # noqa: E402


def rectify_observations(points, intrinsics, image_size):
    """Return corrected pixels with the existing PINHOLE K at the input size."""
    width, height = image_size
    source = (intrinsics["image_width"], intrinsics["image_height"])
    if width <= 0 or height <= 0 or not np.isclose(width / height, source[0] / source[1]):
        raise ValueError("rectification requires a positive size and unchanged aspect ratio")
    K = np.asarray(intrinsics["K"], dtype=float).copy()
    K[0] *= width / source[0]
    K[1] *= height / source[1]
    points = np.asarray(points, dtype=float).reshape(-1, 1, 2)
    if not np.isfinite(points).all():
        raise ValueError("observed pixels must be finite")
    corrected = cv2.undistortPoints(points, K, np.asarray(intrinsics["dist"], dtype=float), P=K)
    return corrected.reshape(-1, 2), K


def raw_anafi_rectifier(intrinsics, camera, image_size):
    """Offline candidate only: resize and rectify raw ANAFI frames with unchanged K."""
    evidence = camera_calibration_evidence(intrinsics, camera)
    if (
        not evidence["intrinsics_match_active_pinhole"]
        or not evidence["distortion_coefficients_valid"]
    ):
        raise ValueError("raw ANAFI calibration must match the active query camera")
    _points, K = rectify_observations([[0.0, 0.0]], intrinsics, image_size)
    maps = cv2.initUndistortRectifyMap(
        K, np.asarray(intrinsics["dist"]), None, K, tuple(image_size), cv2.CV_32FC1
    )

    def rectify(frame):
        height, width = frame.shape[:2]
        if not np.isclose(width / height, image_size[0] / image_size[1]):
            raise ValueError("raw ANAFI frame aspect ratio differs from calibration")
        resized = cv2.resize(frame, tuple(image_size), interpolation=cv2.INTER_AREA)
        return cv2.remap(resized, *maps, interpolation=cv2.INTER_LINEAR)

    return rectify


def audit(intrinsics_path, site_path, provenance_path):
    raw = Path(intrinsics_path).read_bytes()
    intrinsics = json.loads(raw)
    camera = json.loads(Path(site_path).read_text())["query_camera"]
    provenance = json.loads(Path(provenance_path).read_text())
    evidence = camera_calibration_evidence(intrinsics, camera)
    domain = verify_image_domain(provenance, hashlib.sha256(raw).hexdigest())
    result = {
        "schema": "sfm-camera-calibration-audit/v1",
        **evidence,
        **domain,
        "intrinsics_sha256": hashlib.sha256(raw).hexdigest(),
        "site_sha256": hashlib.sha256(Path(site_path).read_bytes()).hexdigest(),
        "metric_scale_measured": False,
        "formal_control_ready": False,
    }
    if evidence["intrinsics_valid"] and evidence["distortion_coefficients_valid"]:
        size = (intrinsics["image_width"], intrinsics["image_height"])
        x, y = np.meshgrid(np.linspace(0, size[0] - 1, 25), np.linspace(0, size[1] - 1, 15))
        points = np.column_stack((x.ravel(), y.ravel()))
        corrected, _K = rectify_observations(points, intrinsics, size)
        displacement = np.linalg.norm(corrected - points, axis=1)
        result["rectification_displacement_px"] = {
            "p50": float(np.median(displacement)),
            "p95": float(np.percentile(displacement, 95)),
            "max": float(displacement.max()),
            "sample_count": len(points),
        }
    result["next_validation"] = (
        "Compare raw versus rectified ANAFI frames using held-out map observations; "
        "do not automatically apply the coefficients to a different camera/video or rectify twice."
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, default=ROOT / "intrinsics.json")
    parser.add_argument("--provenance", type=Path, default=ROOT / "intrinsics_provenance.json")
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.intrinsics, args.site, args.provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

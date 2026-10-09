#!/usr/bin/env python3
"""Query/render static-map metric camera-Z depth without opening an aircraft link."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "控制介面程式"))
sys.path.insert(0, str(ROOT / "定位演算法/deploy_code/sfm_direct_deploy"))
from direct_paths import verify_file_sha256
from site_profile import load_site_profile
from surface_depth import SurfaceDepth


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path, required=True)
    pose = parser.add_mutually_exclusive_group(required=True)
    pose.add_argument(
        "--pose-json", type=Path, help="raw localization payload with pose and camera_axes_world"
    )
    pose.add_argument(
        "--reference-image",
        help="named reference pose from this site's hash-bound reference_poses.json",
    )
    parser.add_argument("--pixel", nargs=2, type=float, action="append", metavar=("X", "Y"))
    parser.add_argument(
        "--output",
        type=Path,
        help="write native-resolution float32 metric Z-depth .npy; unknown is NaN",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="optional depth render width; height follows calibration aspect ratio",
    )
    parser.add_argument(
        "--meters-per-map-unit",
        type=float,
        help="explicit verified scale, otherwise use calibrated surface manifest",
    )
    parser.add_argument("--scale-source", help="required provenance when supplying a scale")
    return parser


def _validate_args(parser, args):
    if not args.pixel and args.output is None:
        parser.error("specify --pixel and/or --output")
    if (args.meters_per_map_unit is None) != (args.scale_source is None):
        parser.error("--meters-per-map-unit and --scale-source must be supplied together")
    if args.scale_source is not None and not args.scale_source.strip():
        parser.error("--scale-source must identify the verified calibration")
    if args.output is not None and args.output.suffix != ".npy":
        parser.error("--output must end in .npy")


def _reference_pose(args, profile):
    verify_file_sha256(profile.map_reference_poses, profile.asset_sha256.map_reference_poses)
    poses = json.loads(profile.map_reference_poses.read_text(encoding="utf-8"))
    reference = poses["poses"][args.reference_image]
    axes = np.asarray(reference["R"], dtype=float)
    center = -axes.T @ np.asarray(reference["t"], dtype=float)
    return center, axes


def _payload_pose(args, profile):
    payload = json.loads(args.pose_json.read_text(encoding="utf-8"))
    if not payload.get("success") or payload.get("camera_attitude_kind") != "full":
        raise ValueError(
            "pose-json must contain a successful full camera pose, not a yaw-only prediction"
        )
    capture = payload.get("capture_mono_ns")
    held = payload.get("pose_capture_mono_ns")
    if held is not None and (capture is None or abs(int(held) - int(capture)) > 1000):
        raise ValueError("pose-json contains a held pose from a different capture")
    raw_pose = payload.get("pose_raw") or payload["pose"]
    center = [raw_pose[key] for key in ("x", "y", "z")]
    axes = payload["camera_axes_world"]
    if payload.get("coordinate_frame_id") != profile.coordinate_frame.id:
        raise ValueError("pose-json must declare the matching coordinate_frame_id")
    return center, axes


def _render_output(args, surface, camera, center, axes):
    height = (
        None
        if args.width is None
        else max(1, round(args.width * camera["height"] / camera["width"]))
    )
    depths = surface.render(
        camera,
        center,
        axes,
        width=args.width,
        height=height,
        meters_per_map_unit=args.meters_per_map_unit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, depths, allow_pickle=False)
    return {
        "output": str(args.output),
        "shape": list(depths.shape),
        "valid_fraction": float(np.isfinite(depths).mean()),
    }


def _depth_summary(args, profile, camera, center, axes):
    surface = SurfaceDepth.load(
        profile.localization_bundle.parent / "dense_surface.json",
        coordinate_frame_id=profile.coordinate_frame.id,
        source_bundle_path=profile.localization_bundle,
    )
    summary = {
        "coordinate_frame_id": surface.coordinate_frame_id,
        "units": "m",
        "depth_definition": "camera_Z",
        "source": "static_mvs_map",
        "meters_per_map_unit": args.meters_per_map_unit or surface.meters_per_map_unit,
        "scale_source": args.scale_source or surface.scale_source,
    }
    if args.pixel:
        depths = surface.sample(
            camera, center, axes, args.pixel, meters_per_map_unit=args.meters_per_map_unit
        )
        summary["samples"] = [
            {"pixel": p, "depth_m": float(d) if np.isfinite(d) else None}
            for p, d in zip(args.pixel, depths, strict=False)
        ]
    if args.output is not None:
        summary.update(_render_output(args, surface, camera, center, axes))
    return summary


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        profile = load_site_profile(args.site_profile)
        verify_file_sha256(profile.localization_bundle, profile.asset_sha256.localization_bundle)
        camera = asdict(profile.query_camera)
        if args.reference_image:
            center, axes = _reference_pose(args, profile)
        else:
            center, axes = _payload_pose(args, profile)
        summary = _depth_summary(args, profile, camera, center, axes)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    except (ImportError, OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Depth query failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

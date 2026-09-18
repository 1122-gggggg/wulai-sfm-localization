#!/usr/bin/env python3
"""Build a `direct` localizer site release from an EDM fixed-pose deploy pack.

The deploy pack ships a retriangulated COLMAP model, the keyframe JPEGs the
runtime matches against, MoGe-3 reference depth, a MegaLoc descriptor bank and
the frozen P174 configuration.  This tool turns that pack into the on-disk
release layout the operator interface loads (site_profile.json + map/ +
model/ + keyframes/ + depth_moge3/ + localization/ + compat/ + provenance/).

Coordinate frame identity
-------------------------
``coordinate_frame_id`` is ``<release-id>_<sha12>`` where ``<sha12>`` is the
first 12 hex characters of the SHA-256 taken over the concatenated *contents*
of ``cameras.bin``, ``images.bin`` and ``points3D.bin`` -- streamed in exactly
that order, with nothing else mixed in (no file names, no other model files).
The same digest is published in full as ``source_model_sha256`` in
reference_poses.json, so a release whose model bytes changed can never keep
claiming the old frame.

Transfer modes
--------------
``--mode link`` (default) hard-links the bulk payload (model/*.bin,
keyframes/images/**, depth_moge3/**) so a 3 GB pack costs no extra disk.  A
hard link is not a symlink, so site_profile's symlink rejection still passes.
Hard links cannot cross filesystems; if the pack lives on another device the
tool fails loudly and asks for ``--mode copy``.  Small text/JSON/NPY assets are
always physically copied so that editing a release never mutates the pack.

Everything is written into ``<releases>/.staging-<uuid>/``, verified there, and
only then ``os.replace``d onto the release name, so an interrupted build never
leaves a half-written release behind.

Usage
-----
    python tools/build_direct_site_release.py \
        --pack river-deploy-5060-20260908 \
        --site river_site \
        --release-id river_gluemap_all8_direct_20260908
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np

TOOL_VERSION = "direct-site-release/1"
TOOL_PATH = "tools/build_direct_site_release.py"

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAVITY_TOOL = REPO_ROOT / "定位演算法" / "validation" / "derive_map_gravity.py"
DEPLOY_DIR_RELATIVE = "../../../../../定位演算法/deploy_code/sfm_direct_deploy"

MODEL_BINS = ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")
#: Order is contractual: the frame digest streams exactly these three, in this order.
FRAME_DIGEST_BINS = ("cameras.bin", "images.bin", "points3D.bin")

#: The stream is 1280x720 undistorted PINHOLE; the model itself is stored at the
#: 2688x1512 source resolution, so the query camera is declared explicitly.
QUERY_CAMERA = {
    "model": "PINHOLE",
    "width": 1280,
    "height": 720,
    "params": [960.4853099760471, 958.1961747147875, 670.8167651412149, 358.7191813450141],
}
QUERY_INTRINSICS = {
    "schema_version": 1,
    "image_width": 1280,
    "image_height": 720,
    "K": [
        [960.4853099760471, 0.0, 670.8167651412149],
        [0.0, 958.1961747147875, 358.7191813450141],
        [0.0, 0.0, 1.0],
    ],
    "images_are_undistorted": True,
}

#: Display cloud budget.  The full model has 2.44M points (~60 MB of PLY) which
#: makes the operator map pane crawl; the release ships a deterministic stride
#: sample instead.  Localization never reads map.ply.
PLY_TARGET_POINTS = 150_000

#: The 20260831 release, whose controller envelope was signed off against
#: S = 2.3346915245056152.  Every *_map_units guard is re-derived from the same
#: ratio against the new map scale so the physical envelope is preserved.
REFERENCE_CONTROLLER_SCALE = 2.3346915245056152
REFERENCE_CONTROLLER = {
    "model": "scale_free_direction_speed_guard_v1",
    "speed_limit_mps": 0.3,
    "pose_max_age_ms": 500,
    "speed_max_age_ms": 500,
    "command_ttl_ms": 150,
    "yaw_tolerance_deg": 3.0,
    "horizontal_axes": [0, 2],
    "vertical_axis": 1,
    "camera_to_body_yaw_deg": 0.0,
    "body_right_sign": 1,
    "lookahead_map_units": 1.2701803255058843,
    "rejoin_tolerance_map_units": 0.6350901627529422,
    "arrival_tolerance_map_units": 0.3175450813764711,
    "inspect_radius_map_units": 0.7938627034411776,
    "inspect_resume_margin_map_units": 0.3175450813764711,
    "max_pose_jump_map_units": 0.9338766098022462,
    "progress_jump_slack_map_units": 0.7938627034411776,
    "max_progress_regression_map_units": 0.15877254068823553,
    "segment_window": 2,
    "progress_speed_factor": 2.0,
    "inspect_waypoints": [],
}
SCALED_CONTROLLER_KEYS = tuple(key for key in REFERENCE_CONTROLLER if key.endswith("_map_units"))

#: P174 frozen knobs (P174_AND_NEXT.md).  Production reads these from the
#: SHA-bound profile; there is no environment-variable override path.
#: min_reference_occupied_bins=1 is the frozen *runtime* argument: replay passes
#: min_reference_occupied_bins=1 (replay_two_rate_reference.py:308) so the
#: geometry loader keeps every reference with >=1 occupied cell and the frozen
#: sideview bank applies cleanly.  The 431 in the pack's localizer_config.json
#: documents bank *construction*, not runtime, and is never read at runtime.
FROZEN_RELOC = {
    "top_k": 2,
    "lift_distance_px": 2.0,
    "period_s": 1.0,
    "min_points": 60,
    "reference_bank": "sideview-boq",
    "min_reference_occupied_bins": 1,
    "bank_occupancy_percentile": 10,
    "descriptor_batch_size": 8,
    "batch_refs": False,
    "reference_cache_size": 1024,
    "frozen_pnp_thresholds": {
        "strong_inliers": 80,
        "minimum_inlier_ratio": 0.25,
        "minimum_hull_coverage": 0.15,
        "minimum_occupancy_4x4": 6,
        "minimum_positive_depth_ratio": 0.99,
        "maximum_reprojection_p90_px": 3.0,
    },
}
FROZEN_FAST_LOOP = {
    "resolution": [960, 540],
    "tracker": "klt",
    "track_cap": 500,
    "reseed_min_points": 120,
    "topup": False,
    # 實測：klt.win 21→15 省 ~0.2ms/幀、慢速段 keep 同級；levels 保持 3 不動
    # （L2 在大位移 keep 掉 1-4pp，已否決）。
    # 警告：凍結值已變更，release profile 需重生，本輪由 Phase 3 處理，此檔只改常數。
    "klt": {"fb_max_px": 1.0, "win": 15, "levels": 3},
    "pnp": {
        "max_error_px": 4.0,
        "min_trials": 10,
        "max_trials": 100,
        "confidence": 0.999,
        # 實測：pnp.refine_iters 20→2，真實 1845 點 fix 上 5.20→4.04ms、
        # 位姿差 dR=0.0005deg；合成 500 點 1.44→1.15ms。
        # 警告：凍結值已變更，release profile 需重生，本輪由 Phase 3 處理，此檔只改常數。
        "refine_iters": 2,
        "min_points": 12,
        "min_inliers": 12,
    },
}
FROZEN_VO = {
    "enabled": True,
    "keyframe_stride": 8,
    "batch_lag": 6,
    "min_live": 350,
    "detect_cap": 300,
    "min_distance": 8,
    "min_parallax_deg": 0.5,
    "max_reproj_px": 2.0,
    "refine": False,
    "window_ba": False,
}
FROZEN_DEAD_RECKON = {"enabled": True, "max_frames": 300, "max_age_s": 10.0}

QUALITY_NOTE = (
    "P174 reports in-sample pose coverage 99.40% over the mapping keyframes only. "
    "There is no absolute ground truth for this map, no held-out evaluation, and no "
    "on-site acceptance flight on the new coordinate frame or any route. "
    "Not eligible for AUTO or vehicle approval."
)


class BuildError(RuntimeError):
    """A build precondition failed; the staging directory is removed."""


# --------------------------------------------------------------------------- io


_SHA256_CACHE: dict[tuple[int, int, int, int], str] = {}
_COPY_SOURCE_TARGET: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}


def sha256_file(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        st = resolved.stat()
        key = (int(st.st_dev), int(st.st_ino), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        key = None
    if key is not None:
        if key in _SHA256_CACHE:
            return _SHA256_CACHE[key]
        alt_key = _COPY_SOURCE_TARGET.get(key)
        if alt_key is not None and alt_key in _SHA256_CACHE:
            val = _SHA256_CACHE[alt_key]
            _SHA256_CACHE[key] = val
            return val

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    val = digest.hexdigest()
    if key is not None:
        _SHA256_CACHE[key] = val
        alt_key = _COPY_SOURCE_TARGET.get(key)
        if alt_key is not None:
            _SHA256_CACHE[alt_key] = val
    return val


def frame_digest(model_dir: Path) -> str:
    """SHA-256 over cameras.bin || images.bin || points3D.bin, contents only."""
    digest = hashlib.sha256()
    for name in FRAME_DIGEST_BINS:
        with (model_dir / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def transfer(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            os.link(source, target)
        except OSError as exc:
            hint = (
                " The pack and the release directory are on different filesystems; "
                "rerun with --mode copy (needs the full payload size in free space)."
                if exc.errno == errno.EXDEV
                else ""
            )
            raise BuildError(f"cannot hard-link {source} -> {target}: {exc}.{hint}") from exc
    elif mode == "copy":
        shutil.copyfile(source, target)
        try:
            st_s = Path(source).resolve().stat()
            st_t = Path(target).resolve().stat()
            src_k = (int(st_s.st_dev), int(st_s.st_ino), int(st_s.st_mtime_ns), int(st_s.st_size))
            tgt_k = (int(st_t.st_dev), int(st_t.st_ino), int(st_t.st_mtime_ns), int(st_t.st_size))
            if src_k in _SHA256_CACHE:
                _SHA256_CACHE[tgt_k] = _SHA256_CACHE[src_k]
            _COPY_SOURCE_TARGET[tgt_k] = src_k
            _COPY_SOURCE_TARGET[src_k] = tgt_k
        except OSError:
            pass
    elif mode == "move":
        shutil.move(str(source), str(target))
    else:
        raise BuildError(f"unknown transfer mode: {mode}")


def transfer_tree(source: Path, target: Path, mode: str) -> list[Path]:
    """Transfer every regular file under *source*, returning target-relative paths."""
    written: list[Path] = []
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise BuildError(f"deploy pack must not contain symlinks: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        transfer(item, target / relative, mode)
        written.append(relative)
    return written


def tree_digest(root: Path, relatives: Iterable[Path]) -> dict:
    """A single digest over a whole tree: sha256 of "<relpath> <sha256>" lines."""
    digest = hashlib.sha256()
    count = 0
    total = 0
    for relative in sorted(relatives, key=lambda item: item.as_posix()):
        path = root / relative
        line = f"{relative.as_posix()} {sha256_file(path)}\n"
        digest.update(line.encode("utf-8"))
        count += 1
        total += path.stat().st_size
    return {
        "file_count": count,
        "total_bytes": total,
        "tree_sha256": digest.hexdigest(),
        "tree_sha256_definition": "sha256 of sorted '<relpath> <sha256>\\n' lines",
    }


# ------------------------------------------------------------------- generators


def write_map_ply(reconstruction, path: Path, target_points: int) -> dict:
    """Deterministic stride sample of the sparse cloud as binary little-endian PLY."""
    ids = np.fromiter(reconstruction.points3D.keys(), dtype=np.int64)
    ids.sort()
    stride = max(1, int(math.ceil(ids.size / max(1, target_points))))
    selected = ids[::stride]
    record = struct.Struct("<fffBBB")
    points3D = reconstruction.points3D
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"comment deterministic stride sample of the sparse model: every {stride}"
            f" point of {ids.size} sorted by point3D_id; display only\n"
            "comment mean decoded RGB over source-frame track observations;"
            " no recoloring\n"
            f"element vertex {selected.size}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n".encode("ascii")
        )
        chunk = bytearray()
        for identifier in selected.tolist():
            point = points3D[identifier]
            xyz = point.xyz
            color = point.color
            chunk += record.pack(
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                int(color[0]),
                int(color[1]),
                int(color[2]),
            )
            if len(chunk) >= 1 << 20:
                stream.write(chunk)
                chunk = bytearray()
        stream.write(chunk)
    return {
        "method": "stride sample over point3D_id ascending",
        "source_points": int(ids.size),
        "stride": stride,
        "written_points": int(selected.size),
        "format": "binary_little_endian 1.0, float x y z + uchar red green blue",
    }


def write_reference_poses(
    reconstruction,
    path: Path,
    *,
    coordinate_frame_id: str,
    model_digest: str,
    ref_names: list[str],
) -> tuple[str, dict]:
    """Every registered image's world->camera pose, keyed by COLMAP image name."""
    poses: dict[str, dict] = {}
    centers = np.zeros((reconstruction.num_images(), 3), dtype=np.float64)
    for index, image in enumerate(
        sorted(reconstruction.images.values(), key=lambda item: item.name)
    ):
        cam_from_world = image.cam_from_world()
        rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(cam_from_world.translation, dtype=np.float64)
        centers[index] = -rotation.T @ translation
        poses[image.name] = {
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "camera_id": "query_camera",
        }
    missing = [name for name in ref_names if name not in poses]
    if missing:
        raise BuildError(
            f"{len(missing)} reference images are not registered in the model, first: {missing[0]}"
        )
    document = {
        "schema": "reference-poses/v2",
        "schema_version": 2,
        "coordinate_frame_id": coordinate_frame_id,
        "source_model_sha256": model_digest,
        "camera": QUERY_CAMERA,
        "cameras": {"query_camera": QUERY_CAMERA},
        "ref_names": ref_names,
        "poses": poses,
    }
    digest = write_json(path, document)
    scale_center = np.median(centers, axis=0)
    map_scale = float(2.0 * np.percentile(np.linalg.norm(centers - scale_center, axis=1), 95))
    return digest, {
        "pose_count": len(poses),
        "ref_name_count": len(ref_names),
        "map_scale": map_scale,
        "map_scale_definition": (
            "S = 2 * p95(||C - componentwise_median(C)||) over every registered "
            "camera centre in the release model"
        ),
    }


def derive_gravity(poses_path: Path, out_path: Path, frame_name: str) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(GRAVITY_TOOL),
            "--poses",
            str(poses_path),
            "--out",
            str(out_path),
            "--frame-name",
            frame_name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not out_path.is_file():
        raise BuildError(f"derive_map_gravity.py failed:\n{result.stdout}\n{result.stderr}")
    print(result.stdout.strip())
    return json.loads(out_path.read_text(encoding="utf-8"))


def controller_for_scale(map_scale: float) -> tuple[dict, dict]:
    coefficients = {
        key: REFERENCE_CONTROLLER[key] / REFERENCE_CONTROLLER_SCALE
        for key in SCALED_CONTROLLER_KEYS
    }
    controller = dict(REFERENCE_CONTROLLER)
    for key, coefficient in coefficients.items():
        controller[key] = coefficient * map_scale
    return controller, coefficients


def write_shard_manifest(
    localization: Path,
    keyframes_root: Path,
    image_relatives: Iterable[Path],
    *,
    shard_size: int = 64,
    depth_root: Path | None = None,
    depth_relatives: Iterable[Path] | None = None,
) -> tuple[Path, str]:
    """Write a deterministic per-shard manifest of bulk keyframe images and optional depth."""
    relatives = sorted(image_relatives)
    shards = []
    for idx, start in enumerate(range(0, len(relatives), shard_size)):
        chunk = relatives[start : start + shard_size]
        entries = []
        for rel in chunk:
            full = keyframes_root / rel
            entries.append(
                {
                    "path": rel.as_posix(),
                    "sha256": sha256_file(full),
                    "size_bytes": full.stat().st_size,
                }
            )
        shard_root_sha256 = hashlib.sha256(
            "".join(f"{e['path']} {e['sha256']}\n" for e in entries).encode("utf-8")
        ).hexdigest()
        shards.append(
            {
                "shard_id": f"shard_{idx:04d}",
                "shard_root_sha256": shard_root_sha256,
                "file_count": len(entries),
                "total_bytes": sum(e["size_bytes"] for e in entries),
                "files": entries,
            }
        )
    manifest_payload: dict[str, object] = {
        "schema": "direct-shard-manifest/v1",
        "shard_size": shard_size,
        "num_shards": len(shards),
        "total_files": len(relatives),
        "shards": shards,
    }
    if depth_root is not None and depth_relatives is not None:
        depth_list = sorted(depth_relatives)
        depth_shards = []
        for idx, start in enumerate(range(0, len(depth_list), shard_size)):
            chunk = depth_list[start : start + shard_size]
            entries = []
            for rel in chunk:
                full = depth_root / rel
                entries.append(
                    {
                        "path": rel.as_posix(),
                        "sha256": sha256_file(full),
                        "size_bytes": full.stat().st_size,
                    }
                )
            d_root_sha = hashlib.sha256(
                "".join(f"{e['path']} {e['sha256']}\n" for e in entries).encode("utf-8")
            ).hexdigest()
            depth_shards.append(
                {
                    "shard_id": f"depth_shard_{idx:04d}",
                    "shard_root_sha256": d_root_sha,
                    "file_count": len(entries),
                    "total_bytes": sum(e["size_bytes"] for e in entries),
                    "files": entries,
                }
            )
        manifest_payload["depth_shards"] = depth_shards
        manifest_payload["total_depth_files"] = len(depth_list)
    manifest_path = localization / "shard_manifest.json"
    manifest_sha = write_json(manifest_path, manifest_payload)
    return manifest_path, manifest_sha


def write_inductor_prewarm(
    staging: Path,
    *,
    batch_sizes: tuple[int, ...] = (1, 2),
) -> tuple[Path, str]:
    """Generate hardware-matched inductor prewarm metadata.

    Target sm_ capability is detected from torch.cuda if available.
    On hardware mismatch or absent GPU, runtime safely falls back
    to JIT compilation (SFM_EDM_FUSED_COARSE=0 / eager JIT).
    Note for PackExport: exporter excludes inductor_cache per package_manifest policy.
    """
    sm_arch = "none"
    has_cuda = False
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            sm_arch = f"sm_{major}{minor}"
            has_cuda = True
    except Exception:
        pass

    prewarm_payload = {
        "schema": "direct-inductor-prewarm/v1",
        "target_sm": sm_arch,
        "has_cuda": has_cuda,
        "warmup_batches": list(batch_sizes),
        "fallback_policy": "fail-safe-jit",
        "note": (
            "Hardware-matched Inductor prewarm package. Runtime falls back to JIT "
            "compilation whenever hardware SM capability does not match."
        ),
    }
    prewarm_path = staging / "compat" / "inductor_prewarm.json"
    prewarm_sha = write_json(prewarm_path, prewarm_payload)
    return prewarm_path, prewarm_sha


# ------------------------------------------------------------------------ build


def resolve_map_dir(pack: Path, explicit: str) -> Path:
    if explicit:
        candidate = (pack / explicit).resolve()
        if not (candidate / "experiments/edm_fixedpose_retriangulate/model").is_dir():
            raise BuildError(f"--map-dir has no retriangulated model: {candidate}")
        return candidate
    candidates = sorted(
        path.parents[2]
        for path in pack.glob("*/experiments/edm_fixedpose_retriangulate/model")
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise BuildError(
            "expected exactly one */experiments/edm_fixedpose_retriangulate/model "
            f"under {pack}, found {len(candidates)}; pass --map-dir"
        )
    return candidates[0]


def prepare_release_geometry(bundle_path: Path) -> str:
    """Precompute CPU geometry and bind it to the existing bundle manifest."""
    direct_root = REPO_ROOT / "定位演算法/deploy_code/sfm_direct_deploy"
    for path in (direct_root, direct_root / "vendor"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from direct_geometry import GEOMETRY_ARCHIVE, prepare_geometry_archive
    from direct_map import DirectMapAssets

    assets = DirectMapAssets.load(bundle_path)
    if any(entry["path"] == GEOMETRY_ARCHIVE for entry in assets.raw["files"]):
        return assets.sha256
    destination = bundle_path.parent / GEOMETRY_ARCHIVE
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    try:
        prepare_geometry_archive(assets, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    raw = dict(assets.raw)
    raw["files"] = [entry for entry in raw["files"] if entry["path"] != GEOMETRY_ARCHIVE]
    raw["files"].append(
        {
            "path": GEOMETRY_ARCHIVE,
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }
    )
    return write_json(bundle_path, raw)


def build(args: argparse.Namespace) -> int:
    import pycolmap

    pack = Path(args.pack).expanduser().resolve(strict=True)
    map_dir = resolve_map_dir(pack, args.map_dir)
    experiment = map_dir / "experiments/edm_fixedpose_retriangulate"
    source_model = experiment / "model"
    source_depth = experiment / "depth_moge3"
    source_keyframes = map_dir / "keyframes"
    source_localization = map_dir / "localization"

    releases = (REPO_ROOT / "地圖檔" / "場域" / args.site / "releases").resolve()
    if not releases.is_dir():
        raise BuildError(f"site releases directory does not exist: {releases}")
    final = releases / args.release_id
    if final.exists() and not args.force:
        raise BuildError(f"release already exists (pass --force to replace): {final}")

    staging = releases / f".staging-{uuid.uuid4().hex}"
    try:
        report = populate(
            staging,
            args,
            pack=pack,
            map_dir=map_dir,
            source_model=source_model,
            source_depth=source_depth,
            source_keyframes=source_keyframes,
            source_localization=source_localization,
            pycolmap=pycolmap,
        )
        verify_staging(staging, pycolmap)
        if final.exists():
            trash = releases / f".trash-{uuid.uuid4().hex}"
            os.replace(final, trash)
            shutil.rmtree(trash, ignore_errors=True)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[release] wrote {final}")
    return 0


def populate(
    staging: Path,
    args: argparse.Namespace,
    *,
    pack: Path,
    map_dir: Path,
    source_model: Path,
    source_depth: Path,
    source_keyframes: Path,
    source_localization: Path,
    pycolmap,
) -> dict:
    mode = args.mode
    release_id = args.release_id
    staging.mkdir(parents=True)

    # --- bulk payload -------------------------------------------------------
    model_dir = staging / "model"
    for name in MODEL_BINS:
        transfer(source_model / name, model_dir / name, mode)
    shutil.copyfile(source_model / "occupancy.json", model_dir / "occupancy.json")
    image_relatives = transfer_tree(source_keyframes / "images", staging / "keyframes/images", mode)
    depth_relatives = transfer_tree(source_depth, staging / "depth_moge3", mode)
    shutil.copyfile(source_keyframes / "keyframes.jsonl", staging / "keyframes/keyframes.jsonl")

    # --- small localization assets -----------------------------------------
    localization = staging / "localization"
    localization.mkdir(parents=True)
    copied_localization = (
        "reference_manifest.jsonl",
        "boq_references_sideview.npy",
        "boq_references_sideview.names.json",
        "fim_lwtl_intersection_cells.json",
    )

    # --- identity -----------------------------------------------------------
    model_digest = frame_digest(model_dir)
    coordinate_frame_id = f"{release_id}_{model_digest[:12]}"
    coordinate_frame = {
        "id": coordinate_frame_id,
        "convention": "glomap",
        "horizontal_axes": ["x", "z"],
        "up_axis": "-y",
        "handedness": "right",
        "units": "map",
    }

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    occupancy = json.loads((model_dir / "occupancy.json").read_text(encoding="utf-8"))
    if (
        reconstruction.num_images() != occupancy["n_images"]
        or reconstruction.num_points3D() != occupancy["n_points3D"]
    ):
        raise BuildError(
            "model disagrees with occupancy.json: "
            f"{reconstruction.num_images()}/{reconstruction.num_points3D()} vs "
            f"{occupancy['n_images']}/{occupancy['n_points3D']}"
        )

    # --- generated assets ---------------------------------------------------
    ply_stats = write_map_ply(reconstruction, staging / "map/map.ply", PLY_TARGET_POINTS)
    ref_names = [
        json.loads(line)["image_name"]
        for line in (localization / "reference_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    poses_path = localization / "reference_poses.json"
    poses_sha, pose_stats = write_reference_poses(
        reconstruction,
        poses_path,
        coordinate_frame_id=coordinate_frame_id,
        model_digest=model_digest,
        ref_names=ref_names,
    )
    gravity_path = localization / "T_align_gravity.json"
    gravity = derive_gravity(poses_path, gravity_path, coordinate_frame_id)
    gravity_sha = sha256_file(gravity_path)

    map_scale = pose_stats["map_scale"]
    controller, coefficients = controller_for_scale(map_scale)

    profile_path = localization / "direct_localizer_profile.json"
    profile_sha = write_json(
        profile_path,
        {
            "schema": "direct-deployment-profile/v1",
            "name": "river_p174_frozen",
            "map_scale": map_scale,
            "intrinsics": QUERY_INTRINSICS,
            "reloc": FROZEN_RELOC,
            "fast_loop": FROZEN_FAST_LOOP,
            "vo": FROZEN_VO,
            "dead_reckon": FROZEN_DEAD_RECKON,
            "max_jump_u": 1.5,
        },
    )

    shard_manifest_path, shard_manifest_sha = write_shard_manifest(
        localization,
        staging / "keyframes/images",
        image_relatives,
        depth_root=staging / "depth_moge3",
        depth_relatives=depth_relatives,
    )
    prewarm_path, prewarm_sha = write_inductor_prewarm(staging)

    # --- bundle -------------------------------------------------------------
    bundle_files = [
        *(f"../model/{name}" for name in sorted(MODEL_BINS)),
        "reference_manifest.jsonl",
        "boq_references_sideview.npy",
        "boq_references_sideview.names.json",
        "fim_lwtl_intersection_cells.json",
        "../keyframes/keyframes.jsonl",
        "shard_manifest.json",
        "../compat/inductor_prewarm.json",
    ]
    bundle_path = localization / "direct_bundle.json"
    bundle_sha = write_json(
        bundle_path,
        {
            "schema": "direct-localization-bundle/v1",
            "map_revision_id": release_id,
            "coordinate_frame_id": coordinate_frame_id,
            "model_dir": "../model",
            "keyframes_manifest": "../keyframes/keyframes.jsonl",
            "keyframes_images_root": "../keyframes/images",
            "reference_manifest": "reference_manifest.jsonl",
            "reference_bank": {
                "name": "sideview-boq",
                "descriptors": "boq_references_sideview.npy",
                "names": "boq_references_sideview.names.json",
            },
            "intersection_cells": "fim_lwtl_intersection_cells.json",
            "reference_depth_dir": "../depth_moge3",
            "model_sha256": {name: sha256_file(model_dir / name) for name in FRAME_DIGEST_BINS},
            "files": [
                {
                    "path": relative,
                    "sha256": sha256_file(localization / relative),
                    "size_bytes": (localization / relative).stat().st_size,
                }
                for relative in bundle_files
            ],
        },
    )

    # --- site profile -------------------------------------------------------
    bundle_sha = prepare_release_geometry(bundle_path)
    site_profile_path = staging / "site_profile.json"
    ply_sha = sha256_file(staging / "map/map.ply")
    write_json(
        site_profile_path,
        {
            "schema_version": 2,
            "site_id": release_id,
            "display_name": "河濱全八段 DIRECT（未獨立驗證）",
            "localizer": "direct",
            "localizer_deploy_dir": DEPLOY_DIR_RELATIVE,
            "localizer_profile": "localization/direct_localizer_profile.json",
            "map_reference_poses": "localization/reference_poses.json",
            "map_align": "localization/T_align_gravity.json",
            "query_camera": QUERY_CAMERA,
            "coordinate_frame": coordinate_frame,
            "asset_sha256": {
                "map_ply": ply_sha,
                "localization_bundle": bundle_sha,
                "route_json": None,
                "localizer_profile": profile_sha,
                "map_reference_poses": poses_sha,
                "map_align": gravity_sha,
                "shard_manifest": shard_manifest_sha,
                "inductor_prewarm": prewarm_sha,
            },
            "hardware_approval": None,
            "flight": {
                "approved": False,
                "coordinate_frame_id": coordinate_frame_id,
                "route_clearance_approved": False,
                "approval_note": (
                    "新座標系、無航線、未做現場驗收；P174 只有 in-sample pose coverage，"
                    "禁止 AUTO/真機核准。"
                ),
                "controller": controller,
            },
            "assets": {
                "map_ply": "map/map.ply",
                "route_json": None,
                "localization_bundle": "localization/direct_bundle.json",
                "megaloc_cache": None,
                "track_landmarks": None,
                "poles_json": None,
                "shard_manifest": "localization/shard_manifest.json",
                "inductor_prewarm": "compat/inductor_prewarm.json",
            },
        },
    )
    site_profile_sha = sha256_file(site_profile_path)

    # --- compat -------------------------------------------------------------
    compat = staging / "compat"
    quality_gate_id = f"{release_id}_direct_unvalidated"
    write_json(
        compat / "map_manifest.json",
        {
            "schema": "sfm-map-revision/v1",
            "site_id": f"{args.site}_direct",
            "map_revision_id": release_id,
            "coordinate_frame": coordinate_frame,
            "assets": {
                "map_ply": {"path": "../map/map.ply", "sha256": ply_sha},
                "reference_poses": {
                    "path": "../localization/reference_poses.json",
                    "sha256": poses_sha,
                },
                "map_align": {
                    "path": "../localization/T_align_gravity.json",
                    "sha256": gravity_sha,
                },
            },
        },
    )
    write_json(
        compat / "localizer_direct_manifest.json",
        {
            "schema": "sfm-localizer-variant/v1",
            "algorithm_id": "direct",
            "variant_id": f"{release_id}_direct",
            "provider_api_version": 1,
            "pose_contract_version": 1,
            "map_revision_id": release_id,
            "coordinate_frame_id": coordinate_frame_id,
            "camera_profiles": ["river_b0_p116_p117_map_scaled_1280x720_pinhole_v1"],
            "required_vehicle_capabilities": ["rgb_stream_1280x720"],
            "quality_gate_id": quality_gate_id,
            "artifacts": {
                "bundle": {
                    "path": "../localization/direct_bundle.json",
                    "sha256": bundle_sha,
                },
                "profile": {
                    "path": "../localization/direct_localizer_profile.json",
                    "sha256": profile_sha,
                },
            },
            "runtime": {
                "device": "cuda",
                "matcher": "edm",
                "retrieval": "megaloc",
                "deploy_dir": "定位演算法/deploy_code/sfm_direct_deploy",
            },
        },
    )
    write_json(
        compat / "localizer_quality_receipt.json",
        {
            "schema": "sfm-calibration-receipt/v1",
            "receipt_id": quality_gate_id,
            "kind": "localizer_quality",
            "subject": quality_gate_id,
            "passed": False,
            "issued_at": args.issued_at,
            "expires_at": None,
            "details": {
                "source_status": "MAP_BUILT_UNVALIDATED_ALL_INPUTS",
                "validation": "NONE",
                "in_sample_pose_coverage": 0.9940,
                "absolute_ground_truth": "NONE",
                "site_acceptance_flight": "NONE",
                "reference_pose_count": pose_stats["pose_count"],
                "bundle_sha256": bundle_sha,
                "profile_sha256": profile_sha,
                "source_evidence": "../provenance/P174_AND_NEXT.md",
                "notes": QUALITY_NOTE,
            },
        },
    )

    # --- provenance ---------------------------------------------------------
    provenance = staging / "provenance"
    provenance.mkdir(parents=True)
    for name in ("README.md", "P174_AND_NEXT.md"):
        shutil.copyfile(pack / name, provenance / name)

    small_inputs = [
        ("model/occupancy.json", source_model / "occupancy.json"),
        ("keyframes/keyframes.jsonl", source_keyframes / "keyframes.jsonl"),
        *((f"localization/{name}", source_localization / name) for name in copied_localization),
        ("provenance/README.md", pack / "README.md"),
        ("provenance/P174_AND_NEXT.md", pack / "P174_AND_NEXT.md"),
    ]
    report = {
        "schema": "direct-release-provenance/v1",
        "release_id": release_id,
        "site": args.site,
        "tool": TOOL_PATH,
        "tool_version": TOOL_VERSION,
        "transfer_mode": mode,
        "source": {
            "pack_root": str(pack),
            "map_dir": str(map_dir.relative_to(pack)),
            "model": str(source_model.relative_to(pack)),
            "depth": str(source_depth.relative_to(pack)),
            "keyframes": str(source_keyframes.relative_to(pack)),
            "localization": str(source_localization.relative_to(pack)),
        },
        "model": {
            "coordinate_frame_id": coordinate_frame_id,
            "frame_digest_definition": "sha256 over the concatenated contents of cameras.bin, images.bin, "
            "points3D.bin in exactly that order",
            "model_sha256": model_digest,
            "n_images": reconstruction.num_images(),
            "n_points3D": reconstruction.num_points3D(),
            "bins": [
                {
                    "path": f"model/{name}",
                    "sha256": sha256_file(model_dir / name),
                    "size_bytes": (model_dir / name).stat().st_size,
                    "transfer": mode,
                }
                for name in sorted(MODEL_BINS)
            ],
        },
        "small_inputs": [
            {
                "path": relative,
                "source": str(source.relative_to(pack)),
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
                "transfer": "copy",
            }
            for relative, source in small_inputs
        ],
        "bulk_trees": {
            "keyframes/images": {
                **tree_digest(staging / "keyframes/images", image_relatives),
                "transfer": mode,
            },
            "depth_moge3": {
                **tree_digest(staging / "depth_moge3", depth_relatives),
                "transfer": mode,
            },
        },
        "map_ply_downsample": ply_stats,
        "map_scale": {
            "S": map_scale,
            "definition": pose_stats["map_scale_definition"],
            "n_camera_centres": pose_stats["pose_count"],
        },
        "controller_scaling": {
            "reference_release": "river_gluemap_all8_direct_20260831",
            "reference_map_scale": REFERENCE_CONTROLLER_SCALE,
            "coefficients": coefficients,
            "derivation": "coefficient = reference_value / reference_map_scale; "
            "new_value = coefficient * S",
            "values": {key: controller[key] for key in SCALED_CONTROLLER_KEYS},
        },
        "gravity": gravity["derivation"],
        "generated": {
            "map/map.ply": ply_sha,
            "localization/reference_poses.json": poses_sha,
            "localization/T_align_gravity.json": gravity_sha,
            "localization/direct_bundle.json": bundle_sha,
            "localization/direct_localizer_profile.json": profile_sha,
            "site_profile.json": site_profile_sha,
            "localization/shard_manifest.json": shard_manifest_sha,
            "compat/inductor_prewarm.json": prewarm_sha,
        },
        "reference_counts": {
            "reference_manifest": len(ref_names),
            "poses": pose_stats["pose_count"],
        },
    }
    write_json(provenance / "source_manifest.json", report)
    return report


def _verify_shard_entries(
    shards: list[dict],
    root: Path,
    label: str,
) -> None:
    for shard in shards:
        files = shard.get("files", [])
        if "shard_root_sha256" in shard:
            expected_root = hashlib.sha256(
                "".join(f"{e['path']} {e['sha256']}\n" for e in files).encode("utf-8")
            ).hexdigest()
            if shard["shard_root_sha256"] != expected_root:
                raise BuildError(f"{label} shard {shard.get('shard_id')} root hash mismatch")
        for entry in files:
            full = root / entry["path"]
            if not full.is_file():
                raise BuildError(f"{label} file missing: {entry['path']}")
            if sha256_file(full) != entry["sha256"]:
                raise BuildError(f"{label} hash mismatch: {entry['path']}")


def _verify_shard_manifest(staging: Path, digests: dict[str, str]) -> None:
    shard_manifest = staging / "localization/shard_manifest.json"
    if not shard_manifest.is_file():
        return
    actual = sha256_file(shard_manifest)
    if "shard_manifest" in digests and actual != digests["shard_manifest"]:
        raise BuildError(
            f"site_profile asset_sha256.shard_manifest mismatch: {actual} != {digests['shard_manifest']}"
        )
    manifest_obj = json.loads(shard_manifest.read_text(encoding="utf-8"))
    _verify_shard_entries(
        manifest_obj.get("shards", []),
        staging / "keyframes/images",
        label="keyframe",
    )
    _verify_shard_entries(
        manifest_obj.get("depth_shards", []),
        staging / "depth_moge3",
        label="depth",
    )


def _verify_colmap_and_keyframes(staging: Path, pycolmap) -> None:
    reconstruction = pycolmap.Reconstruction(str(staging / "model"))
    occupancy = json.loads((staging / "model/occupancy.json").read_text(encoding="utf-8"))
    if (
        reconstruction.num_images() != occupancy["n_images"]
        or reconstruction.num_points3D() != occupancy["n_points3D"]
    ):
        raise BuildError("promoted model does not reload with the expected counts")

    keyframes = staging / "keyframes/images"
    manifest_lines = (
        (staging / "keyframes/keyframes.jsonl").read_text(encoding="utf-8").splitlines()
    )
    for line in manifest_lines:
        if not line.strip():
            continue
        record = json.loads(line)
        image = keyframes / Path(record["image_uri"]).parent.name / Path(record["image_uri"]).name
        if not image.is_file():
            raise BuildError(f"keyframe listed in the manifest is missing: {image}")


def _verify_bundle(staging: Path) -> None:
    bundle = json.loads((staging / "localization/direct_bundle.json").read_text(encoding="utf-8"))
    localization = staging / "localization"
    for entry in bundle["files"]:
        path = (localization / entry["path"]).resolve()
        actual = sha256_file(path)
        if actual != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            raise BuildError(f"bundle file digest mismatch: {entry['path']}")
    for name, expected in bundle["model_sha256"].items():
        if sha256_file(staging / "model" / name) != expected:
            raise BuildError(f"model digest mismatch: {name}")


def verify_staging(staging: Path, pycolmap) -> None:
    """Re-hash every published digest and re-open the model before promoting."""
    _verify_bundle(staging)
    profile = json.loads((staging / "site_profile.json").read_text(encoding="utf-8"))
    digests = profile["asset_sha256"]
    targets = {
        "map_ply": profile["assets"]["map_ply"],
        "localization_bundle": profile["assets"]["localization_bundle"],
        "localizer_profile": profile["localizer_profile"],
        "map_reference_poses": profile["map_reference_poses"],
        "map_align": profile["map_align"],
    }
    if profile.get("assets", {}).get("shard_manifest"):
        targets["shard_manifest"] = profile["assets"]["shard_manifest"]
    if profile.get("assets", {}).get("inductor_prewarm"):
        targets["inductor_prewarm"] = profile["assets"]["inductor_prewarm"]
    for key, relative in targets.items():
        if relative is None:
            continue
        if sha256_file(staging / relative) != digests[key]:
            raise BuildError(f"site_profile asset_sha256.{key} mismatch")

    _verify_shard_manifest(staging, digests)

    prewarm_file = staging / "compat/inductor_prewarm.json"
    if prewarm_file.is_file():
        actual_prewarm = sha256_file(prewarm_file)
        if "inductor_prewarm" in digests and actual_prewarm != digests["inductor_prewarm"]:
            raise BuildError("site_profile asset_sha256.inductor_prewarm mismatch")

    _verify_colmap_and_keyframes(staging, pycolmap)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", required=True, help="deploy pack root directory")
    parser.add_argument("--site", required=True, help="site directory under 地圖檔/場域")
    parser.add_argument("--release-id", required=True, help="release directory name")
    parser.add_argument(
        "--map-dir", default="", help="map directory inside the pack (default: auto-detected)"
    )
    parser.add_argument(
        "--mode",
        default="link",
        choices=("link", "copy", "move"),
        help="how the bulk payload is transferred (default: link)",
    )
    parser.add_argument(
        "--issued-at",
        default="",
        help="quality receipt timestamp (default: derived from the release id date suffix)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing release directory"
    )
    args = parser.parse_args(argv)
    if not args.issued_at:
        stamp = args.release_id[-8:]
        if not stamp.isdigit():
            parser.error("--issued-at is required when the release id has no date suffix")
        args.issued_at = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}T00:00:00+08:00"
    try:
        return build(args)
    except BuildError as exc:
        print(f"[release] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

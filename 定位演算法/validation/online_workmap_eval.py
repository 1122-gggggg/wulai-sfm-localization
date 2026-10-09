#!/usr/bin/env python3
"""Preregistered full-frame online workmap evaluation through the production worker.

Protocol paths are relative to the protocol file unless absolute. The protocol
schema is ``online-workmap-eval/v1``; see :func:`load_protocol` for its strict
contract. Each ``run`` mode executes every sequence three times in fresh worker
processes, at the declared source frame rate. It never calls a tracker directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# The interface is shared; only --deploy-dir selects the expansion algorithm.
WORKER_PATH = ROOT / "控制介面程式" / "operator_interface" / "live_localizer_worker.py"
WORKER_MODULES = WORKER_PATH.parent
if str(WORKER_MODULES) not in sys.path:
    sys.path.insert(0, str(WORKER_MODULES))

from live_localizer_protocol import encode_request  # noqa: E402

SCHEMA = "online-workmap-eval/v1"
RUN_SCHEMA = "online-workmap-run/v1"
REPORT_SCHEMA = "online-workmap-report/v1"
MODES = ("baseline", "reuse", "ba", "full")
REPETITIONS = 3
_ALLOWED_SEGMENT_ROLES = frozenset(
    {"old_map", "new_region", "occlusion_or_loss", "new_region_revisit", "old_map_return"}
)
_THREAD_ENV_NAMES = frozenset(
    {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"}
)
_FIXED_GATES = {
    "min_e2e_fps": 15.0,
    "min_visual_fresh_fps": 15.0,
    "max_capture_to_consumer_p95_ms": 100.0,
    "max_capture_to_consumer_p99_ms": 200.0,
    "max_foreground_p95_ms": 50.0,
    "frame_count_metadata_tolerance_frames": 1,
    "fps_metadata_tolerance": 0.01,
    "window_s": 10.0,
    "response_timeout_s": 120.0,
}

_WORKMAP_HEALTH_FIELDS = (
    "workmap_writer_healthy",
    "workmap_index_healthy",
    "workmap_ba_healthy",
    "workmap_overlap_healthy",
)
_WORKMAP_QUEUE_FIELDS = (
    ("reloc", "workmap_reloc_pending", "workmap_reloc_oldest_age_ms"),
    ("index", "workmap_index_pending", "workmap_index_oldest_age_ms"),
    ("ba", "workmap_ba_pending", "workmap_ba_oldest_age_ms"),
    ("writer", "workmap_writer_pending", "workmap_writer_oldest_age_ms"),
    ("overlap", "workmap_overlap_pending", "workmap_overlap_oldest_age_ms"),
)
_WORKMAP_FAILURE_COUNTERS = (
    "workmap_writer_overflow_count",
    "workmap_index_errors",
)
_WORKMAP_MAX_INDEX_READY_MS = 1_000.0
_WORKMAP_MAX_BA_APPLY_MS = 1_000.0
_REQUIRED_PROTOCOL_KEYS = frozenset(
    {"schema", "run_id", "repetitions", "output_root", "bundle", "profile", "query_camera", "sequences", "hardware", "threads", "gates"}
)


class EvaluationError(RuntimeError):
    """Protocol, worker, or evidence-integrity failure."""


@dataclass(frozen=True)
class Protocol:
    raw: dict[str, Any]
    path: Path
    output_root: Path
    bundle_path: Path
    profile_path: Path
    sequences: tuple[dict[str, Any], ...]


def _json_no_constants(value: str):
    raise EvaluationError(f"non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_json_no_constants)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root must be an object: {path}")
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_number(value: object, what: str) -> float:
    if not _is_number(value) or float(value) <= 0.0:
        raise EvaluationError(f"{what} must be a positive finite number")
    return float(value)


def _safe_name(value: object, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise EvaluationError(f"{what} must be a simple path-safe identifier")
    if value in {".", ".."}:
        raise EvaluationError(f"invalid {what}: {value!r}")
    return value


def _sha_string(value: object, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise EvaluationError(f"{what} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _reject_symlink_path(path: Path, *, allow_missing: bool, what: str) -> Path:
    """Reject symlinks in any existing component, including a symlinked file."""
    absolute = Path(os.path.abspath(path.expanduser()))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise EvaluationError(f"{what} traverses a symlink: {cursor}")
        if not cursor.exists() and not allow_missing:
            raise EvaluationError(f"{what} does not exist: {cursor}")
    return absolute


def _resolve_input_path(raw: object, base: Path, *, what: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise EvaluationError(f"{what} path must be a non-empty string")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = _reject_symlink_path(candidate, allow_missing=False, what=what)
    if not candidate.is_file():
        raise EvaluationError(f"{what} must be a regular file: {candidate}")
    return candidate


def _resolve_output_path(raw: object, base: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise EvaluationError("output_root must be a non-empty path string")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return _reject_symlink_path(candidate, allow_missing=True, what="output_root")


def _asset(raw: object, base: Path, label: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
        raise EvaluationError(f"{label} must contain exactly path and sha256")
    digest = _sha_string(raw["sha256"], f"{label}.sha256")
    path = _resolve_input_path(raw["path"], base, what=label)
    normalized = dict(raw)
    normalized["path"] = str(path)
    normalized["sha256"] = digest
    return normalized, path


def _validate_segments(raw: object, sequence_name: str) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise EvaluationError(f"sequence {sequence_name}.segments must be a list")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    intervals: list[tuple[float, float]] = []
    for index, segment in enumerate(raw):
        if not isinstance(segment, dict) or set(segment) - {"name", "start_s", "end_s", "role"}:
            raise EvaluationError(f"sequence {sequence_name} segment {index} has invalid fields")
        name = _safe_name(segment.get("name"), f"sequence {sequence_name} segment name")
        if name in names:
            raise EvaluationError(f"sequence {sequence_name} has duplicate segment {name!r}")
        names.add(name)
        start = segment.get("start_s")
        end = segment.get("end_s")
        if not _is_number(start) or not _is_number(end) or float(start) < 0 or float(end) <= float(start):
            raise EvaluationError(f"sequence {sequence_name} segment {name} needs finite 0 <= start_s < end_s")
        role = segment.get("role")
        if role is not None and role not in _ALLOWED_SEGMENT_ROLES:
            raise EvaluationError(f"sequence {sequence_name} segment {name} has unsupported role {role!r}")
        start_f, end_f = float(start), float(end)
        if any(start_f < previous_end and end_f > previous_start for previous_start, previous_end in intervals):
            raise EvaluationError(f"sequence {sequence_name} segments must not overlap")
        intervals.append((start_f, end_f))
        item = {"name": name, "start_s": start_f, "end_s": end_f}
        if role is not None:
            item["role"] = role
        result.append(item)
    return sorted(result, key=lambda segment: segment["start_s"])


def _load_protocol_source(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = _reject_symlink_path(Path(path), allow_missing=False, what="protocol")
    source = source.resolve(strict=True)
    if not source.is_file():
        raise EvaluationError(f"protocol is not a regular file: {source}")
    raw = _read_json(source)
    missing = _REQUIRED_PROTOCOL_KEYS - raw.keys()
    unknown = raw.keys() - _REQUIRED_PROTOCOL_KEYS
    if missing or unknown:
        raise EvaluationError(f"protocol keys mismatch (missing={sorted(missing)}, unknown={sorted(unknown)})")
    if raw.get("schema") != SCHEMA:
        raise EvaluationError(f"protocol schema must be {SCHEMA!r}")
    return source, raw


def _validate_query_camera(camera: object) -> dict[str, Any]:
    if not isinstance(camera, dict) or set(camera) != {"model", "width", "height", "params"}:
        raise EvaluationError("query_camera must contain exactly model, width, height, params")
    if not isinstance(camera["model"], str) or not camera["model"].strip():
        raise EvaluationError("query_camera.model must be a non-empty string")
    for dimension in ("width", "height"):
        if isinstance(camera[dimension], bool) or not isinstance(camera[dimension], int) or camera[dimension] <= 0:
            raise EvaluationError(f"query_camera.{dimension} must be a positive integer")
    params = camera["params"]
    if not isinstance(params, list) or len(params) < 4 or any(not _is_number(value) for value in params):
        raise EvaluationError("query_camera.params must contain at least four finite numbers")
    return camera


def _load_sequence(sequence: object, base: Path, seen_sequences: set[str]) -> dict[str, Any]:
    allowed = {"name", "video", "video_sha256", "fps", "segments", "ground_truth"}
    required = {"name", "video", "video_sha256", "fps"}
    if not isinstance(sequence, dict) or required - sequence.keys() or sequence.keys() - allowed:
        raise EvaluationError(f"sequence must contain {sorted(required)} and no unknown fields")
    name = _safe_name(sequence["name"], "sequence name")
    if name in seen_sequences:
        raise EvaluationError(f"duplicate sequence name: {name}")
    seen_sequences.add(name)
    fps = _positive_number(sequence["fps"], f"sequence {name}.fps")
    video_sha = _sha_string(sequence["video_sha256"], f"sequence {name}.video_sha256")
    video_path = _resolve_input_path(sequence["video"], base, what=f"sequence {name} video")
    segments = _validate_segments(sequence.get("segments", []), name)
    ground_truth = sequence.get("ground_truth")
    gt_path = None
    normalized_gt = None
    if ground_truth is not None:
        if not isinstance(ground_truth, dict) or set(ground_truth) != {"path", "sha256"}:
            raise EvaluationError(f"sequence {name}.ground_truth must be null or contain path and sha256")
        gt_path = _resolve_input_path(ground_truth["path"], base, what=f"sequence {name} ground truth")
        normalized_gt = {"path": str(gt_path), "sha256": _sha_string(ground_truth["sha256"], f"sequence {name}.ground_truth.sha256")}
    normalized = dict(sequence)
    normalized["name"] = name
    normalized["video"] = str(video_path)
    normalized["video_sha256"] = video_sha
    normalized["fps"] = fps
    normalized["segments"] = segments
    normalized["ground_truth"] = normalized_gt
    return {"raw": normalized, "video_path": video_path, "ground_truth_path": gt_path}


def _load_sequences(sequences_raw: object, base: Path) -> list[dict[str, Any]]:
    if not isinstance(sequences_raw, list) or not sequences_raw:
        raise EvaluationError("sequences must be a non-empty list")
    seen_sequences: set[str] = set()
    return [_load_sequence(sequence, base, seen_sequences) for sequence in sequences_raw]


def _validate_threads(threads: object) -> dict[str, Any]:
    if not isinstance(threads, dict) or not threads:
        raise EvaluationError("threads must be a non-empty environment-variable object")
    unknown_threads = threads.keys() - _THREAD_ENV_NAMES
    if unknown_threads:
        raise EvaluationError(f"unsupported thread environment variables: {sorted(unknown_threads)}")
    missing_threads = _THREAD_ENV_NAMES - threads.keys()
    if missing_threads:
        raise EvaluationError(f"thread settings must pin every supported thread variable: {sorted(missing_threads)}")
    for name, value in threads.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvaluationError(f"threads.{name} must be a positive integer")
    return threads


def _validate_gates(gates: object) -> dict[str, Any]:
    if not isinstance(gates, dict) or set(gates) != set(_FIXED_GATES):
        raise EvaluationError(f"gates must contain exactly {sorted(_FIXED_GATES)}")
    for name, expected in _FIXED_GATES.items():
        actual = gates[name]
        if not _is_number(actual) or float(actual) != float(expected):
            raise EvaluationError(f"gates.{name} must be preregistered as {expected}")
    return gates


def _release_roots(*asset_paths: Path) -> set[Path]:
    # Site releases live under releases/<release-id>. Also recognize the
    # builder's portable release marker when a release is copied elsewhere.
    release_roots: set[Path] = set()
    for asset_path in asset_paths:
        for candidate in asset_path.parents:
            if candidate.parent.name == "releases":
                release_roots.add(candidate)
            if (candidate / "provenance" / "source_manifest.json").is_file() and (candidate / "compat" / "localizer_direct_manifest.json").is_file():
                release_roots.add(candidate)
    return release_roots


def load_protocol(path: str | Path) -> Protocol:
    """Load and fail-closed validate an ``online-workmap-eval/v1`` protocol.

    All paths are resolved relative to the protocol file. Gate numbers are
    preregistered constants, not per-run defaults; repetitions is exactly 3.
    """
    source, raw = _load_protocol_source(path)
    run_id = _safe_name(raw.get("run_id"), "run_id")
    if isinstance(raw.get("repetitions"), bool) or raw.get("repetitions") != REPETITIONS:
        raise EvaluationError(f"repetitions must be exactly {REPETITIONS}")
    base = source.parent
    output_root = _resolve_output_path(raw["output_root"], base)
    bundle_raw, bundle_path = _asset(raw["bundle"], base, "bundle")
    profile_raw, profile_path = _asset(raw["profile"], base, "profile")
    camera = _validate_query_camera(raw["query_camera"])
    sequences = _load_sequences(raw["sequences"], base)

    hardware = raw["hardware"]
    if not isinstance(hardware, dict):
        raise EvaluationError("hardware must be an object containing fixed machine details")
    threads = _validate_threads(raw["threads"])
    gates = _validate_gates(raw["gates"])
    release_roots = _release_roots(bundle_path, profile_path)
    if any(output_root == release_root or release_root in output_root.parents for release_root in release_roots):
        raise EvaluationError(f"output_root must be outside frozen release {sorted(map(str, release_roots))}")

    raw_normalized = dict(raw)
    raw_normalized["run_id"] = run_id
    raw_normalized["output_root"] = str(output_root)
    raw_normalized["bundle"] = bundle_raw
    raw_normalized["profile"] = profile_raw
    raw_normalized["query_camera"] = dict(camera)
    raw_normalized["sequences"] = [item["raw"] for item in sequences]
    raw_normalized["threads"] = dict(threads)
    raw_normalized["gates"] = dict(gates)
    raw_normalized["hardware"] = dict(hardware)
    return Protocol(raw_normalized, source, output_root, bundle_path, profile_path, tuple(sequences))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _dump_json_line(stream, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


class _InputSnapshots:
    """Size, mtime, and SHA-256 of each input file; one hash per distinct path."""

    def __init__(self) -> None:
        self.snapshots: dict[str, dict[str, Any]] = {}
        self._file_cache: dict[Path, dict[str, Any]] = {}

    def add_file(
        self,
        name: str,
        path: Path,
        *,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
    ) -> None:
        checked_path = _reject_symlink_path(path, allow_missing=False, what=f"{name} input")
        if not checked_path.is_file():
            raise EvaluationError(f"{name} input is not a regular file: {checked_path}")
        snapshot = self._file_cache.get(checked_path)
        if snapshot is None:
            stat = checked_path.stat()
            snapshot = {
                "path": str(checked_path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(checked_path),
            }
            self._file_cache[checked_path] = snapshot
        if expected_size_bytes is not None and snapshot["size_bytes"] != expected_size_bytes:
            raise EvaluationError(f"{name} size differs from bundle manifest: {checked_path}")
        if expected_sha256 is not None and snapshot["sha256"] != _sha_string(expected_sha256, f"{name}.sha256"):
            raise EvaluationError(f"{name} SHA-256 differs from manifest: {checked_path}")
        self.snapshots[name] = dict(snapshot)


def _snapshot_bundle_files(inputs: _InputSnapshots, bundle_manifest: dict[str, Any], bundle_dir: Path) -> None:
    files = bundle_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise EvaluationError("bundle must contain a non-empty files manifest for frozen asset verification")
    seen_bundle_paths: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict) or not {"path", "sha256", "size_bytes"} <= item.keys():
            raise EvaluationError(f"bundle files[{index}] must contain path, sha256, and size_bytes")
        relative_name = item["path"]
        if not isinstance(relative_name, str) or not relative_name or relative_name in seen_bundle_paths:
            raise EvaluationError(f"bundle files[{index}].path must be a unique non-empty string")
        seen_bundle_paths.add(relative_name)
        size_bytes = item["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise EvaluationError(f"bundle files[{index}].size_bytes must be a non-negative integer")
        bundle_file = _resolve_input_path(relative_name, bundle_dir, what=f"bundle files[{index}]")
        inputs.add_file(
            f"bundle_file:{relative_name}",
            bundle_file,
            expected_sha256=item["sha256"],
            expected_size_bytes=size_bytes,
        )


def _snapshot_model_files(inputs: _InputSnapshots, bundle_manifest: dict[str, Any], bundle_dir: Path) -> None:
    model_hashes = bundle_manifest.get("model_sha256")
    if model_hashes is not None and not isinstance(model_hashes, dict):
        raise EvaluationError("bundle model_sha256 must be an object")
    if not model_hashes:
        return
    model_dir_raw = bundle_manifest.get("model_dir")
    if not isinstance(model_dir_raw, str) or not model_dir_raw:
        raise EvaluationError("bundle model_sha256 requires a model_dir")
    model_dir = Path(model_dir_raw)
    if not model_dir.is_absolute():
        model_dir = bundle_dir / model_dir
    model_dir = _reject_symlink_path(model_dir, allow_missing=False, what="bundle model directory")
    if not model_dir.is_dir():
        raise EvaluationError(f"bundle model_dir is not a directory: {model_dir}")
    if any(not isinstance(name, str) for name in model_hashes):
        raise EvaluationError("bundle model_sha256 keys must be strings")
    for relative_name, expected_sha in sorted(model_hashes.items()):
        relative_path = Path(relative_name)
        if not relative_name or relative_path.is_absolute() or ".." in relative_path.parts:
            raise EvaluationError(f"invalid bundle model_sha256 path: {relative_name!r}")
        model_path = _reject_symlink_path(model_dir / relative_path, allow_missing=False, what="bundle model file")
        if model_dir not in model_path.parents:
            raise EvaluationError(f"bundle model file escapes model_dir: {relative_name!r}")
        inputs.add_file(f"model:{relative_name}", model_path, expected_sha256=expected_sha)


def _keyframe_image_root(bundle_manifest: dict[str, Any], bundle_dir: Path) -> tuple[Path, Path] | None:
    manifest_raw = bundle_manifest.get("keyframes_manifest")
    image_root_raw = bundle_manifest.get("keyframes_images_root")
    if manifest_raw is None and image_root_raw is None:
        return None
    if not isinstance(manifest_raw, str) or not manifest_raw or not isinstance(image_root_raw, str) or not image_root_raw:
        raise EvaluationError("bundle keyframes_manifest and keyframes_images_root must both be non-empty paths")
    keyframes_manifest = _resolve_input_path(manifest_raw, bundle_dir, what="keyframes manifest")
    image_root = Path(image_root_raw).expanduser()
    if not image_root.is_absolute():
        image_root = bundle_dir / image_root
    image_root = _reject_symlink_path(image_root, allow_missing=False, what="keyframe image root")
    if not image_root.is_dir():
        raise EvaluationError(f"keyframe image root is not a directory: {image_root}")
    return keyframes_manifest, image_root


def _snapshot_keyframe_images(inputs: _InputSnapshots, bundle_manifest: dict[str, Any], bundle_dir: Path) -> None:
    located = _keyframe_image_root(bundle_manifest, bundle_dir)
    if located is None:
        return
    keyframes_manifest, image_root = located
    keyframes = _records(keyframes_manifest)
    if not keyframes:
        raise EvaluationError("keyframes manifest has no reference images")
    for ordinal, keyframe in enumerate(keyframes):
        output_name = keyframe.get("output_name")
        image_sha = keyframe.get("image_sha256")
        if not isinstance(output_name, str) or not output_name:
            raise EvaluationError(f"keyframes manifest row {ordinal} has no output_name")
        relative_image = Path(output_name)
        if relative_image.is_absolute() or ".." in relative_image.parts:
            raise EvaluationError(f"keyframes manifest row {ordinal} image escapes its image root")
        image_path = _reject_symlink_path(image_root / relative_image, allow_missing=False, what="keyframe image")
        if image_root not in image_path.parents:
            raise EvaluationError(f"keyframes manifest row {ordinal} image escapes its image root")
        inputs.add_file(f"keyframe_image:{relative_image.as_posix()}", image_path, expected_sha256=image_sha)


def _snapshot_inputs(protocol: Protocol, sequence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inputs = _InputSnapshots()
    expected_inputs = {
        "bundle": (protocol.bundle_path, protocol.raw["bundle"]["sha256"]),
        "profile": (protocol.profile_path, protocol.raw["profile"]["sha256"]),
        "video": (sequence["video_path"], sequence["raw"]["video_sha256"]),
    }
    for name, (path, expected_sha256) in expected_inputs.items():
        inputs.add_file(name, path, expected_sha256=expected_sha256)

    bundle_manifest = _read_json(protocol.bundle_path)
    bundle_dir = protocol.bundle_path.parent
    _snapshot_bundle_files(inputs, bundle_manifest, bundle_dir)
    _snapshot_model_files(inputs, bundle_manifest, bundle_dir)
    _snapshot_keyframe_images(inputs, bundle_manifest, bundle_dir)

    ground_truth_path = sequence.get("ground_truth_path")
    if ground_truth_path is not None:
        inputs.add_file(
            "ground_truth",
            ground_truth_path,
            expected_sha256=sequence["raw"]["ground_truth"]["sha256"],
        )
    return inputs.snapshots


def _snapshots_match(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> bool:
    if not isinstance(before, dict) or not isinstance(after, dict) or before.keys() != after.keys():
        return False
    required_fields = {"path", "size_bytes", "mtime_ns", "sha256"}
    for name in before:
        left, right = before[name], after[name]
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if required_fields - left.keys() or required_fields - right.keys():
            return False
        if any(left[field] != right[field] for field in required_fields):
            return False
    return True


def _verify_full_profile(protocol: Protocol, mode: str) -> None:
    if mode != "full":
        return
    profile = _read_json(protocol.profile_path)
    workmap = profile.get("online_workmap")
    if not isinstance(workmap, dict) or workmap.get("enabled") is not True:
        raise EvaluationError("full mode requires a SHA-bound profile with online_workmap.enabled=true")


def _effective_query_camera(camera: dict[str, Any], source_size: tuple[int, int]) -> dict[str, Any]:
    source_width, source_height = source_size
    if (camera["width"], camera["height"]) == source_size:
        return dict(camera)
    if camera["model"] != "PINHOLE":
        raise EvaluationError("native stream requires PINHOLE intrinsics scaling")
    sx = source_width / camera["width"]
    sy = source_height / camera["height"]
    fx, fy, cx, cy = camera["params"]
    return {
        "model": "PINHOLE",
        "width": source_width,
        "height": source_height,
        "params": [fx * sx, fy * sy, cx * sx, cy * sy],
    }


def _worker_command(
    protocol: Protocol,
    session_dir: Path,
    mode: str,
    source_size: tuple[int, int],
) -> list[str]:
    camera = _effective_query_camera(protocol.raw["query_camera"], source_size)
    command = [
        sys.executable,
        str(WORKER_PATH),
        "--localizer-backend", "direct",
        "--width", str(source_size[0]),
        "--height", str(source_size[1]),
        "--query-camera-model", str(camera["model"]),
        "--query-camera-width", str(camera["width"]),
        "--query-camera-height", str(camera["height"]),
        "--query-camera-params", *(str(float(value)) for value in camera["params"]),
        "--bundle", str(protocol.bundle_path),
        "--bundle-sha256", protocol.raw["bundle"]["sha256"],
        "--production-profile", str(protocol.profile_path),
        "--production-profile-sha256", protocol.raw["profile"]["sha256"],
        "--deploy-dir", str(ROOT / "定位演算法" / "deploy_code_expand" / "sfm_direct_deploy"),
        "--runtime-benchmark-control",
        "--startup-handshake",
        "--workmap-eval-stage", mode,
        "--workmap-session-dir", str(session_dir),
    ]
    return command


def _read_worker_line(process: subprocess.Popen, timeout_s: float, context: str) -> bytes:
    ready, _, _ = select.select([process.stdout], [], [], timeout_s)
    if not ready:
        raise EvaluationError(f"worker timed out waiting for {context} after {timeout_s:g}s")
    line = process.stdout.readline()
    if not line:
        code = process.poll()
        raise EvaluationError(f"worker exited before {context} (returncode={code})")
    return line


def _parse_worker_json(line: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(line.decode("utf-8"), parse_constant=_json_no_constants)
    except (UnicodeError, json.JSONDecodeError, EvaluationError) as exc:
        raise EvaluationError(f"worker emitted invalid JSON for {context}: {line[:300]!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"worker response for {context} must be a JSON object")
    return value
def _write_all(stream, data: bytes | memoryview) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if not written:
            raise BrokenPipeError("worker stdin accepted zero bytes")
        offset += written




def _kill_worker(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait()


def _segment_for_time(segments: list[dict[str, Any]], time_s: float) -> list[dict[str, Any]]:
    return [segment for segment in segments if segment["start_s"] <= time_s < segment["end_s"]]


def _proc_memory_kib(proc: Path) -> dict[str, float]:
    memory = {}
    for line in (proc / "status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, value, *_ = line.split()
            memory[name.rstrip(":")] = float(value)
    return memory


class _ResourceSampler:
    """Sample the worker process tree and device, including background BA."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.peaks: dict[str, float] = {}
        self.samples = 0
        self.gpu_samples = 0
        self.expected_exit_diagnostics: set[str] = set()
        self.worker_exit_observed = threading.Event()
        self.errors: set[str] = set()

    def mark_worker_exit_observed(self) -> None:
        self.worker_exit_observed.set()

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> dict[str, Any]:
        self.stop.set()
        if self.thread.ident is None:
            self.errors.add("resource sampler did not start")
        else:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.errors.add("resource sampler did not stop")
        complete = (
            self.samples > 1
            and self.gpu_samples > 0
            and not self.errors
            and all(
                self.peaks.get(key) is not None
                for key in (
                    "cpu_percent_one_core_100",
                    "ram_mib",
                    "rss_hwm_mib",
                    "gpu_utilization_percent_device",
                    "vram_mib_device",
                )
            )
        )
        return {
            "sampling_status": "COMPLETE" if complete else "PARTIAL",
            "source": "linux /proc/<pid>/stat+status; nvidia-smi --id=0 --query-gpu=utilization.gpu,memory.used",
            "cpu_scope": "worker process tree including child processes",
            "gpu_scope": "device-level total for GPU 0; includes other GPU users",
            "sample_interval_s": 1.0,
            "samples": self.samples,
            "gpu_samples": self.gpu_samples,
            "peak_cpu_percent_one_core_100": self.peaks.get("cpu_percent_one_core_100"),
            "peak_ram_mib": self.peaks.get("ram_mib"),
            "peak_process_tree_rss_hwm_mib": self.peaks.get("rss_hwm_mib"),
            "peak_gpu_utilization_percent_device": self.peaks.get("gpu_utilization_percent_device"),
            "peak_vram_mib_device": self.peaks.get("vram_mib_device"),
            "errors": sorted(self.errors),
            "expected_exit_diagnostics": sorted(self.expected_exit_diagnostics),
        }

    def _process_tree(self) -> tuple[float, float, float] | None:
        pending = [self.pid]
        seen: set[int] = set()
        sampled = 0
        zombie_seen = False
        cpu = rss_kib = hwm_kib = 0.0
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            proc = Path("/proc") / str(pid)
            try:
                sample = self._process_sample(pid, proc)
            except FileNotFoundError:
                continue
            if isinstance(sample, bool):
                zombie_seen = zombie_seen or sample
                continue
            cpu += sample[0]
            rss_kib += sample[1]
            hwm_kib += sample[2]
            sampled += 1
            try:
                children = proc / "task" / str(pid) / "children"
                pending.extend(int(child) for child in children.read_text().split())
            except FileNotFoundError:
                continue
        if sampled == 0:
            return self._empty_process_tree(zombie_seen)
        return cpu, rss_kib / 1024.0, hwm_kib / 1024.0

    def _process_sample(self, pid: int, proc: Path) -> tuple[float, float, float] | bool:
        """CPU seconds, RSS KiB, and HWM KiB; True for a zombie, False if it vanished."""
        stat_text = (proc / "stat").read_text()
        fields = stat_text.rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            self.expected_exit_diagnostics.add(
                f"process {pid} was zombie during the /proc snapshot; omitted its exited RSS"
            )
            return True
        cpu = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        memory = _proc_memory_kib(proc)
        if "VmRSS" not in memory or "VmHWM" not in memory:
            try:
                latest_state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                return False
            if latest_state == "Z":
                self.expected_exit_diagnostics.add(
                    f"process {pid} exited during the /proc snapshot; omitted its zombie RSS"
                )
                return True
            raise ValueError(f"missing RSS fields for process {pid}")
        return cpu, memory["VmRSS"], memory["VmHWM"]

    def _empty_process_tree(self, zombie_seen: bool) -> None:
        if zombie_seen:
            self.expected_exit_diagnostics.add(
                f"worker process tree {self.pid} was exited/zombie during the final snapshot; prior peaks retained"
            )
            return None
        if self.worker_exit_observed.is_set():
            self.expected_exit_diagnostics.add(
                f"worker process tree {self.pid} had exited before the final snapshot; prior peaks retained"
            )
            return None
        raise ProcessLookupError(f"worker process tree {self.pid} is no longer available")

    def _sample(self) -> None:
        last_cpu = last_time = None
        while not self.stop.is_set():
            sample_started = time.monotonic()
            try:
                snapshot = self._process_tree()
                if snapshot is not None:
                    cpu, ram, hwm = snapshot
                    now = time.monotonic()
                    self.peaks["ram_mib"] = max(self.peaks.get("ram_mib", 0.0), ram)
                    self.peaks["rss_hwm_mib"] = max(self.peaks.get("rss_hwm_mib", 0.0), hwm)
                    if last_cpu is not None and now > last_time:
                        percent = max(0.0, 100.0 * (cpu - last_cpu) / (now - last_time))
                        self.peaks["cpu_percent_one_core_100"] = max(
                            self.peaks.get("cpu_percent_one_core_100", 0.0), percent
                        )
                    last_cpu, last_time = cpu, now
                    self.samples += 1
            except ProcessLookupError as exc:
                if self.worker_exit_observed.is_set():
                    self.expected_exit_diagnostics.add(
                        f"worker process tree disappeared after observed exit: {exc}"
                    )
                else:
                    self.errors.add(f"process sampling failed: {type(exc).__name__}: {exc}")
            except (OSError, ValueError, IndexError) as exc:
                self.errors.add(f"process sampling failed: {type(exc).__name__}: {exc}")
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=0.75, check=True,
                )
                utilization, vram = (float(item.strip()) for item in result.stdout.strip().split(","))
                self.peaks["gpu_utilization_percent_device"] = max(
                    self.peaks.get("gpu_utilization_percent_device", 0.0), utilization
                )
                self.peaks["vram_mib_device"] = max(self.peaks.get("vram_mib_device", 0.0), vram)
                self.gpu_samples += 1
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.errors.add(f"device sampling failed: {type(exc).__name__}: {exc}")
            self.stop.wait(max(0.0, 1.0 - (time.monotonic() - sample_started)))


class _WorkerRun:
    """One fresh worker process fed one sequence at its source frame rate."""

    def __init__(self, protocol: Protocol, mode: str, sequence: dict[str, Any], repeat: int) -> None:
        self.protocol = protocol
        self.mode = mode
        self.sequence = sequence
        self.repeat = repeat
        self.raw_sequence = sequence["raw"]
        self.sequence_name = self.raw_sequence["name"]
        run_dir = protocol.output_root / mode / self.sequence_name / f"repeat-{repeat}"
        _reject_symlink_path(run_dir, allow_missing=True, what="run output")
        if run_dir.exists() and any(run_dir.iterdir()):
            raise EvaluationError(f"run output already exists; refusing to overwrite: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        _reject_symlink_path(run_dir, allow_missing=False, what="run output")
        self.frames_path = run_dir / "frames.jsonl"
        self.run_path = run_dir / "run.json"
        self.stderr_path = run_dir / "worker.stderr.log"
        self.session_dir = run_dir / "session"
        _reject_symlink_path(self.session_dir, allow_missing=True, what="workmap session directory")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = uuid.uuid4().hex
        self.resources: _ResourceSampler | None = None
        self.process: subprocess.Popen | None = None
        self.capture: Any = None
        self.frame_sink = None
        self.stderr_sink = None
        self.startup: dict[str, Any] | None = None
        self.input_before: dict[str, dict[str, Any]] = {}
        self.input_after: dict[str, dict[str, Any]] = {}
        self.rows = 0
        self.declared_frames: int | None = None
        self.decoded_frames = 0
        self.source_fps: float | None = None
        self.source_size: list[int] | None = None
        self.decoder_eof = False
        self.started_perf: float | None = None
        self.started_mono_ns: int | None = None
        self.started_wall_s: float | None = None
        self.finished_perf: float | None = None
        self.returncode: int | None = None
        self.failure: str | None = None
        self.command: list[str] = []
        self.record: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "run_id": self.run_id,
            "experiment_run_id": protocol.raw["run_id"],
            "mode": mode,
            "sequence": self.sequence_name,
            "repeat": repeat,
            "complete": False,
            "worker_process_instance": uuid.uuid4().hex,
            "process_id": None,
            "worker_started": False,
            "decoder_read_terminated": False,
            "decoded_until_eof": False,
            "decoded_frames": 0,
            "frame_count_integrity_pass": False,
            "input_video": str(sequence["video_path"]),
            "session_dir": str(self.session_dir),
            "run_dir": str(run_dir),
        }

    def execute(self) -> Path:
        try:
            self._check_inputs()
            self._open_video()
            self._start_worker()
            self._stream_frames()
            self._drain_worker()
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            if self.process is not None:
                _kill_worker(self.process)
                self.returncode = self.process.poll()
        finally:
            self._release_capture_and_worker()
            self._close_sinks()
            self._finish_resources()
            self._record_input_integrity()
            self._record_outcome()
            _atomic_json(self.run_path, self.record)
        label = f"{self.mode}/{self.sequence_name}/repeat-{self.repeat}"
        if self.failure is not None:
            raise EvaluationError(f"{label} failed; see {self.run_path}: {self.failure}")
        if not self.record["complete"]:
            raise EvaluationError(f"{label} did not complete; see {self.run_path}")
        return self.run_path

    def _check_inputs(self) -> None:
        protocol = self.protocol
        _verify_full_profile(protocol, self.mode)
        self.input_before = _snapshot_inputs(protocol, self.sequence)
        for name, expected in (
            ("bundle", protocol.raw["bundle"]["sha256"]),
            ("profile", protocol.raw["profile"]["sha256"]),
            ("video", self.raw_sequence["video_sha256"]),
        ):
            if self.input_before[name]["sha256"] != expected:
                raise EvaluationError(f"{name} SHA-256 differs from protocol")
        self.record["input_snapshot_before"] = self.input_before
        self.record["input_video_sha256"] = self.input_before["video"]["sha256"]
        self.record["input_bundle_sha256"] = self.input_before["bundle"]["sha256"]
        self.record["input_profile_sha256"] = self.input_before["profile"]["sha256"]

    def _open_video(self) -> None:
        import cv2

        raw_sequence = self.raw_sequence
        camera = self.protocol.raw["query_camera"]
        self.capture = cv2.VideoCapture(str(self.sequence["video_path"]), cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            raise EvaluationError(f"cannot open protocol video: {self.sequence['video_path']}")
        self.source_fps = source_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        reported_frames = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.declared_frames = int(round(reported_frames)) if math.isfinite(reported_frames) and reported_frames > 0 else None
        self.width = width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_size = [width, height]
        if width <= 0 or height <= 0:
            raise EvaluationError(f"video has no usable source resolution: {width}x{height}")
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise EvaluationError(f"video has no usable source FPS: {source_fps!r}")
        if abs(source_fps - float(raw_sequence["fps"])) > float(self.protocol.raw["gates"]["fps_metadata_tolerance"]):
            raise EvaluationError(f"video FPS metadata {source_fps} differs from protocol {raw_sequence['fps']}")
        if self.declared_frames is None:
            raise EvaluationError("video has no positive CAP_PROP_FRAME_COUNT for full-run integrity")
        effective_camera = _effective_query_camera(camera, (width, height))
        self.record.update({
            "nominal_source_fps": source_fps,
            "declared_frame_count": self.declared_frames,
            "source_size": self.source_size,
            "query_camera": camera,
            "effective_query_camera": effective_camera,
        })
        clip_duration_s = self.declared_frames / float(raw_sequence["fps"])
        if any(segment["end_s"] > clip_duration_s + 1.0 / float(raw_sequence["fps"]) for segment in raw_sequence["segments"]):
            raise EvaluationError(f"sequence {self.sequence_name} segment extends beyond video duration {clip_duration_s:.3f}s")

    def _start_worker(self) -> None:
        protocol = self.protocol
        self.command = _worker_command(protocol, self.session_dir, self.mode, (self.width, self.height))
        env = os.environ.copy()
        env.update({name: str(value) for name, value in protocol.raw["threads"].items()})
        if self.mode == "full":
            env.pop("SFM_EVALUATION_ONLY", None)
        else:
            env["SFM_EVALUATION_ONLY"] = "1"
        self.record["thread_settings"] = dict(protocol.raw["threads"])
        self.record["evaluation_only_mode"] = env.get("SFM_EVALUATION_ONLY")
        self.stderr_sink = self.stderr_path.open("wb")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_sink,
            cwd=ROOT,
            env=env,
            bufsize=0,
            start_new_session=True,
        )
        self.record["worker_started"] = True
        self.record["process_id"] = self.process.pid
        self.resources = _ResourceSampler(self.process.pid)
        self.resources.start()
        self.record["worker_command"] = self.command
        self.record["python"] = sys.executable
        self.record["worker_path"] = str(WORKER_PATH)
        self.startup = _parse_worker_json(
            _read_worker_line(self.process, float(protocol.raw["gates"]["response_timeout_s"]), "startup ready event"),
            "startup ready event",
        )
        if self.startup.get("event") != "ready":
            raise EvaluationError(f"worker startup event is not ready: {self.startup!r}")
        if self.startup.get("backend") != "direct":
            raise EvaluationError(f"worker selected unexpected backend: {self.startup.get('backend')!r}")
        self.record["worker_startup"] = self.startup

    def _stream_frames(self) -> None:
        self.frame_sink = self.frames_path.open("w", encoding="utf-8", newline="\n")
        self.started_mono_ns = time.monotonic_ns()
        self.started_perf = time.perf_counter()
        wall_probe_before_ns = time.monotonic_ns()
        wall_probe_unix_s = time.time()
        wall_probe_after_ns = time.monotonic_ns()
        wall_probe_mono_ns = (wall_probe_before_ns + wall_probe_after_ns) // 2
        self.started_wall_s = wall_probe_unix_s + (self.started_mono_ns - wall_probe_mono_ns) * 1e-9
        while True:
            decode_started = time.perf_counter()
            ok, bgr = self.capture.read()
            if not ok:
                self._reach_eof()
                break
            if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.size == 0:
                raise EvaluationError(f"decoder returned invalid frame ordinal {self.decoded_frames}")
            if (bgr.shape[1], bgr.shape[0]) != (self.width, self.height):
                raise EvaluationError(f"video resolution changed at frame {self.decoded_frames}")
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            _dump_json_line(self.frame_sink, self._exchange_frame(bgr, decode_ms))
            self.rows += 1
            self.decoded_frames += 1

    def _reach_eof(self) -> None:
        self.decoder_eof = True
        tolerance = int(self.protocol.raw["gates"]["frame_count_metadata_tolerance_frames"])
        assert self.declared_frames is not None
        if abs(self.declared_frames - self.decoded_frames) > tolerance:
            raise EvaluationError(
                f"decoder reached EOF after {self.decoded_frames} frames; metadata declares {self.declared_frames}"
            )

    def _exchange_frame(self, bgr: Any, decode_ms: float) -> dict[str, Any]:
        import cv2

        assert self.process is not None and self.started_mono_ns is not None and self.started_perf is not None
        fps = float(self.raw_sequence["fps"])
        timeout_s = float(self.protocol.raw["gates"]["response_timeout_s"])
        ordinal = self.decoded_frames
        video_time_s = ordinal / fps
        frame_segments = _segment_for_time(self.raw_sequence["segments"], video_time_s)

        # Keep the source timeline while ensuring encode_request's future-stamp
        # guard has elapsed. One millisecond is a fixed send safety margin,
        # not a per-frame delay: deadlines remain anchored to the original FPS.
        source_capture_s = self.started_mono_ns * 1e-9 + video_time_s
        deadline = self.started_perf + video_time_s + 0.001
        pace_started = time.perf_counter()
        wait = deadline - pace_started
        if wait > 0:
            time.sleep(wait)
        pace_wait_ms = max(0.0, (time.perf_counter() - pace_started) * 1000.0)
        request_header = encode_request("auto", source_capture_s)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not rgb.flags.c_contiguous:
            rgb = cv2.ascontiguousarray(rgb)
        send_mono_ns = time.monotonic_ns()
        try:
            _write_all(self.process.stdin, request_header)
            _write_all(self.process.stdin, memoryview(rgb).cast("B"))
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EvaluationError(f"worker stdin failed at frame {ordinal}: {exc}") from exc
        response = _parse_worker_json(_read_worker_line(self.process, timeout_s, f"frame {ordinal} response"), f"frame {ordinal}")
        received_mono_ns = time.monotonic_ns()
        _check_frame_response(response, ordinal, source_capture_s)
        return {
            "experiment_run_id": self.protocol.raw["run_id"],
            "run_id": self.run_id,
            "mode": self.mode,
            "sequence": self.sequence_name,
            "repeat": self.repeat,
            "ordinal": ordinal,
            "video_time_s": video_time_s,
            "segments": [{"name": item["name"], "role": item.get("role")} for item in frame_segments],
            "decode_ms": decode_ms,
            "pace_wait_ms": pace_wait_ms,
            "send_mono_ns": send_mono_ns,
            "received_mono_ns": received_mono_ns,
            "response": response,
        }

    def _drain_worker(self) -> None:
        process = self.process
        assert process is not None
        self.capture.release()
        self.capture = None
        process.stdin.close()
        process.stdin = None
        try:
            self.returncode = process.wait(timeout=float(self.protocol.raw["gates"]["response_timeout_s"]))
        except subprocess.TimeoutExpired as exc:
            raise EvaluationError("worker did not drain/exit after input EOF") from exc
        if self.resources is not None:
            self.resources.mark_worker_exit_observed()
        extra = process.stdout.read()
        if extra:
            raise EvaluationError(f"worker emitted unexpected stdout after final frame: {extra[:300]!r}")
        if self.returncode != 0:
            raise EvaluationError(f"worker exited with status {self.returncode}; see {self.stderr_path}")
        if self.rows != self.decoded_frames or self.decoded_frames <= 0:
            raise EvaluationError(f"worker returned {self.rows} rows for {self.decoded_frames} decoded frames")
        self.record["worker_exit_observed"] = True
        self.record["worker_close_and_session_drain_observed"] = True

    def _release_capture_and_worker(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        process = self.process
        if process is not None and process.poll() is None:
            _kill_worker(process)
        if process is not None and self.returncode is None:
            self.returncode = process.poll()

    def _close_sinks(self) -> None:
        if self.frame_sink is not None:
            try:
                self.frame_sink.flush()
                self.frame_sink.close()
            except Exception as exc:
                self.failure = self.failure or f"frame output flush/close failed: {type(exc).__name__}: {exc}"
        if self.stderr_sink is not None:
            try:
                self.stderr_sink.close()
            except Exception as exc:
                self.failure = self.failure or f"worker stderr close failed: {type(exc).__name__}: {exc}"
        if self.started_perf is not None:
            self.finished_perf = time.perf_counter()

    def _finish_resources(self) -> None:
        if self.resources is None:
            return
        if self.process is not None and self.process.poll() is not None:
            self.resources.mark_worker_exit_observed()
        try:
            self.record["resource_peaks"] = self.resources.finish()
        except Exception as exc:
            self.record["resource_peaks"] = {
                "sampling_status": "FAILED",
                "source": "linux /proc process tree; nvidia-smi GPU device 0",
                "cpu_scope": "worker process tree",
                "gpu_scope": "device-level total; includes other GPU users",
                "errors": [f"resource sampler finalization failed: {type(exc).__name__}: {exc}"],
            }

    def _record_input_integrity(self) -> None:
        if not self.input_before:
            try:
                self.input_before = _snapshot_inputs(self.protocol, self.sequence)
            except Exception as exc:
                self.record["input_snapshot_before_error"] = f"{type(exc).__name__}: {exc}"
        try:
            self.input_after = _snapshot_inputs(self.protocol, self.sequence)
        except Exception as exc:
            self.record["input_snapshot_after_error"] = f"{type(exc).__name__}: {exc}"
        self.record["input_snapshot_before"] = self.input_before
        self.record["input_snapshot_after"] = self.input_after
        self.record["input_integrity_pass"] = bool(
            self.input_before and self.input_after and _snapshots_match(self.input_before, self.input_after)
        )

    def _record_outcome(self) -> None:
        record = self.record
        if self.returncode is not None:
            record["worker_returncode"] = self.returncode
        record["decoder_read_terminated"] = bool(self.decoder_eof)
        record["decoded_until_eof"] = bool(self.decoder_eof)
        record["decoded_frames"] = self.decoded_frames
        record["frame_count_integrity_pass"] = bool(
            self.decoder_eof and self.declared_frames is not None and abs(self.declared_frames - self.decoded_frames) <= int(self.protocol.raw["gates"]["frame_count_metadata_tolerance_frames"])
        )
        record["worker_startup"] = self.startup
        record["worker_command"] = self.command
        record["frames_jsonl"] = str(self.frames_path)
        record["stderr_log"] = str(self.stderr_path)
        record["session_dir"] = str(self.session_dir)
        optional = {
            "nominal_source_fps": self.source_fps,
            "declared_frame_count": self.declared_frames,
            "source_size": self.source_size,
        }
        record.update({key: value for key, value in optional.items() if value is not None})
        if self.started_perf is not None:
            end = self.finished_perf if self.finished_perf is not None else time.perf_counter()
            record["run_wall_s"] = end - self.started_perf
            record["run_start_monotonic_ns"] = self.started_mono_ns
            record["run_start_unix_s"] = self.started_wall_s
        if self.frames_path.is_file():
            record["frames_jsonl_sha256"] = _sha256(self.frames_path)
        record["complete"] = bool(
            self.failure is None
            and self.returncode == 0
            and self.decoder_eof
            and self.rows == self.decoded_frames
            and self.decoded_frames > 0
            and record["input_integrity_pass"]
            and record["frame_count_integrity_pass"]
        )
        record["error"] = self.failure
        record["completed_unix_s"] = time.time()


def _check_frame_response(response: dict[str, Any], ordinal: int, source_capture_s: float) -> None:
    if "error" in response:
        raise EvaluationError(f"worker reported localization error at frame {ordinal}: {response['error']}")
    seq = response.get("seq")
    if isinstance(seq, bool) or seq != ordinal:
        raise EvaluationError(f"worker response seq {seq!r} does not match decoded ordinal {ordinal}")
    expected_frame_id = f"worker-{ordinal}"
    if response.get("frame_id") != expected_frame_id:
        raise EvaluationError(f"worker response frame_id {response.get('frame_id')!r} != {expected_frame_id!r}")
    response_capture_ns = response.get("capture_mono_ns")
    if isinstance(response_capture_ns, bool) or not isinstance(response_capture_ns, int):
        raise EvaluationError(f"worker response has no integer capture_mono_ns at frame {ordinal}")
    expected_capture_ns = int(round(source_capture_s * 1_000_000_000.0))
    if abs(response_capture_ns - expected_capture_ns) > 2:
        raise EvaluationError(f"worker capture timestamp identity mismatch at frame {ordinal}")


def _run_one(protocol: Protocol, mode: str, sequence: dict[str, Any], repeat: int) -> Path:
    return _WorkerRun(protocol, mode, sequence, repeat).execute()


def _prepare_protocol_snapshot(protocol: Protocol) -> Path:
    _reject_symlink_path(protocol.output_root, allow_missing=True, what="output_root")
    protocol.output_root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(protocol.output_root, allow_missing=False, what="output_root")
    snapshot = protocol.output_root / "protocol.json"
    _reject_symlink_path(snapshot, allow_missing=True, what="protocol snapshot")
    if snapshot.exists():
        existing = load_protocol(snapshot).raw
        if json.dumps(existing, ensure_ascii=False, sort_keys=True, allow_nan=False) != json.dumps(protocol.raw, ensure_ascii=False, sort_keys=True, allow_nan=False):
            raise EvaluationError(f"output protocol snapshot differs from {protocol.path}: {snapshot}")
    else:
        _atomic_json(snapshot, protocol.raw)
    return snapshot


def run_mode(protocol_path: str | Path, mode: str) -> list[Path]:
    if mode not in MODES:
        raise EvaluationError(f"unsupported mode {mode!r}; choose from {MODES}")
    protocol = load_protocol(protocol_path)
    if not WORKER_PATH.is_file():
        raise EvaluationError(f"production worker is missing: {WORKER_PATH}")
    _prepare_protocol_snapshot(protocol)
    outputs: list[Path] = []
    for sequence in protocol.sequences:
        for repeat in range(1, REPETITIONS + 1):
            outputs.append(_run_one(protocol, mode, sequence, repeat))
    return outputs


def _records(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    raise EvaluationError(f"blank JSONL row at {path}:{line_no}")
                try:
                    row = json.loads(line, parse_constant=_json_no_constants)
                except (json.JSONDecodeError, EvaluationError) as exc:
                    raise EvaluationError(f"invalid JSONL row at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise EvaluationError(f"JSONL row must be an object at {path}:{line_no}")
                result.append(row)
    except OSError as exc:
        raise EvaluationError(f"cannot read frames {path}: {exc}") from exc
    return result


def _percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _visual_fresh(row: dict[str, Any]) -> bool:
    response = row.get("response")
    if not isinstance(response, dict) or response.get("validity") is not True:
        return False
    if not isinstance(response.get("pose"), dict):
        return False
    capture_ns = response.get("capture_mono_ns")
    pose_ns = response.get("pose_capture_mono_ns")
    if isinstance(capture_ns, bool) or isinstance(pose_ns, bool) or not isinstance(capture_ns, int) or not isinstance(pose_ns, int):
        return False
    if abs(capture_ns - pose_ns) > 2:
        return False
    # Direct tracker statuses separate a fresh PnP solution (including weak
    # VO-only visual poses) from propagated DEAD_RECKON/IMU_BRIDGE outputs.
    direct_status = response.get("direct_status")
    if direct_status is not None:
        return direct_status in {"FAST_TRACK", "VO_ONLY", "RELOC_SEED"}
    return response.get("pose_status") == "VISUALLY_CONFIRMED"


def _run_paths(protocol: Protocol, mode: str, name: str, repeat: int) -> tuple[Path, Path, Path]:
    run_dir = protocol.output_root / mode / name / f"repeat-{repeat}"
    _reject_symlink_path(run_dir, allow_missing=False, what=f"run directory {mode}/{name}/repeat-{repeat}")
    session_dir = run_dir / "session"
    _reject_symlink_path(session_dir, allow_missing=False, what="workmap session directory")
    if not session_dir.is_dir():
        raise EvaluationError(f"missing session directory for {mode}/{name}/repeat-{repeat}")
    run_path, frames_path = run_dir / "run.json", run_dir / "frames.jsonl"
    if not run_path.is_file() or not frames_path.is_file():
        raise EvaluationError(f"missing run.json or frames.jsonl for {mode}/{name}/repeat-{repeat}")
    _reject_symlink_path(run_path, allow_missing=False, what="run metadata")
    _reject_symlink_path(frames_path, allow_missing=False, what="frame output")
    return session_dir, run_path, frames_path


def _check_run_completion(
    record: dict[str, Any], expected_identity: dict[str, Any], run_path: Path
) -> None:
    for key, expected in expected_identity.items():
        if record.get(key) != expected:
            raise EvaluationError(f"run identity mismatch {key} in {run_path}")
    if (
        record.get("complete") is not True
        or record.get("decoder_read_terminated") is not True
        or record.get("decoded_until_eof") is not True
    ):
        raise EvaluationError(f"run did not consume the decoder through EOF and complete cleanly: {run_path}")
    if record.get("worker_returncode") != 0 or record.get("worker_exit_observed") is not True:
        raise EvaluationError(f"worker did not exit successfully: {run_path}")
    if (
        not isinstance(record.get("worker_process_instance"), str)
        or not record["worker_process_instance"]
        or isinstance(record.get("process_id"), bool)
        or not isinstance(record.get("process_id"), int)
    ):
        raise EvaluationError(f"missing independent worker process identity: {run_path}")
    if record.get("input_integrity_pass") is not True or record.get("frame_count_integrity_pass") is not True:
        raise EvaluationError(f"input/frame-count integrity failed: {run_path}")
    if record.get("worker_close_and_session_drain_observed") is not True:
        raise EvaluationError(f"worker close/session drain was not observed: {run_path}")


def _check_command_identity(
    record: dict[str, Any], camera: dict[str, Any], run_path: Path
) -> tuple[list[str], list[int], dict[str, Any]]:
    command = record.get("worker_command")
    if (
        not isinstance(command, list)
        or any(not isinstance(argument, str) for argument in command)
        or str(WORKER_PATH) not in command
    ):
        raise EvaluationError(f"run did not record the production worker command: {run_path}")
    source_size = record.get("source_size")
    if (
        not isinstance(source_size, list)
        or len(source_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in source_size)
        or record.get("query_camera") != camera
    ):
        raise EvaluationError(f"run native resolution/calibrated camera metadata is invalid: {run_path}")
    effective_camera = _effective_query_camera(camera, (source_size[0], source_size[1]))
    if record.get("effective_query_camera") != effective_camera:
        raise EvaluationError(f"run effective stream camera differs from scaled protocol calibration: {run_path}")
    if (
        len(command) < 2
        or command[0] != record.get("python")
        or command[1] != str(WORKER_PATH)
        or record.get("worker_path") != str(WORKER_PATH)
    ):
        raise EvaluationError(f"run command does not invoke the recorded production worker: {run_path}")
    return command, source_size, effective_camera


def _command_camera_params(command: list[str], run_path: Path) -> list[str]:
    try:
        params_index = command.index("--query-camera-params") + 1
    except ValueError as exc:
        raise EvaluationError(f"worker command is missing --query-camera-params: {run_path}") from exc
    actual_params = []
    for value in command[params_index:]:
        if value.startswith("--"):
            break
        actual_params.append(value)
    return actual_params


def _check_command_options(
    command: list[str],
    expected_options: tuple[tuple[str, str], ...],
    effective_camera: dict[str, Any],
    run_path: Path,
) -> None:
    for flag, expected in expected_options:
        try:
            actual = command[command.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise EvaluationError(f"worker command is missing {flag}: {run_path}") from exc
        if actual != expected:
            raise EvaluationError(f"worker command {flag}={actual!r}, expected {expected!r}: {run_path}")
    expected_params = [str(float(value)) for value in effective_camera["params"]]
    if _command_camera_params(command, run_path) != expected_params:
        raise EvaluationError(f"worker query-camera parameters differ from protocol: {run_path}")
    if "--runtime-benchmark-control" not in command:
        raise EvaluationError(f"worker command did not enable the timed stdin protocol: {run_path}")
    if "--startup-handshake" not in command:
        raise EvaluationError(f"worker command did not enable the startup handshake: {run_path}")


def _check_worker_command(
    record: dict[str, Any], protocol: Protocol, mode: str, session_dir: Path, run_path: Path
) -> None:
    command, source_size, effective_camera = _check_command_identity(
        record, protocol.raw["query_camera"], run_path
    )
    expected_options = (
        ("--localizer-backend", "direct"),
        ("--width", str(source_size[0])),
        ("--height", str(source_size[1])),
        ("--query-camera-model", str(effective_camera["model"])),
        ("--query-camera-width", str(effective_camera["width"])),
        ("--query-camera-height", str(effective_camera["height"])),
        ("--bundle", str(protocol.bundle_path)),
        ("--bundle-sha256", protocol.raw["bundle"]["sha256"]),
        ("--production-profile", str(protocol.profile_path)),
        ("--production-profile-sha256", protocol.raw["profile"]["sha256"]),
        ("--deploy-dir", str(ROOT / "定位演算法" / "deploy_code_expand" / "sfm_direct_deploy")),
        ("--workmap-eval-stage", mode),
        ("--workmap-session-dir", str(session_dir)),
    )
    _check_command_options(command, expected_options, effective_camera, run_path)
    if not isinstance(record.get("worker_startup"), dict) or record["worker_startup"].get("event") != "ready":
        raise EvaluationError(f"worker startup handshake is missing: {run_path}")
    expected_evaluation_only = None if mode == "full" else "1"
    if record.get("evaluation_only_mode") != expected_evaluation_only:
        raise EvaluationError(f"worker evaluation-only stage gate mismatch: {run_path}")
    if record.get("thread_settings") != protocol.raw["threads"]:
        raise EvaluationError(f"worker thread settings differ from protocol: {run_path}")


def _expected_snapshot_names(protocol: Protocol, sequence: dict[str, Any]) -> set[str]:
    bundle_manifest = _read_json(protocol.bundle_path)
    expected_snapshot_names = {"bundle", "profile", "video"}
    expected_snapshot_names.update(f"bundle_file:{item['path']}" for item in bundle_manifest["files"])
    expected_snapshot_names.update(f"model:{name}" for name in (bundle_manifest.get("model_sha256") or {}))
    if bundle_manifest.get("keyframes_manifest") is not None:
        manifest_path = _resolve_input_path(bundle_manifest["keyframes_manifest"], protocol.bundle_path.parent, what="keyframes manifest")
        expected_snapshot_names.update(
            f"keyframe_image:{Path(row['output_name']).as_posix()}"
            for row in _records(manifest_path)
        )
    if sequence.get("ground_truth_path") is not None:
        expected_snapshot_names.add("ground_truth")
    return expected_snapshot_names


def _check_run_inputs(
    record: dict[str, Any], protocol: Protocol, sequence: dict[str, Any], frames_path: Path, run_path: Path
) -> None:
    if record.get("frames_jsonl_sha256") != _sha256(frames_path):
        raise EvaluationError(f"frames.jsonl SHA-256 mismatch: {frames_path}")
    snapshots_before = record.get("input_snapshot_before")
    snapshots_after = record.get("input_snapshot_after")
    if not isinstance(snapshots_before, dict) or not isinstance(snapshots_after, dict) or not _snapshots_match(snapshots_before, snapshots_after):
        raise EvaluationError(f"source hashes changed during run or are missing: {run_path}")
    expected_hashes = {
        "bundle": protocol.raw["bundle"]["sha256"],
        "profile": protocol.raw["profile"]["sha256"],
        "video": sequence["raw"]["video_sha256"],
        **(
            {"ground_truth": sequence["raw"]["ground_truth"]["sha256"]}
            if sequence["raw"].get("ground_truth") is not None
            else {}
        ),
    }
    if snapshots_before.keys() != _expected_snapshot_names(protocol, sequence):
        raise EvaluationError(f"run input snapshot does not cover every frozen release asset: {run_path}")
    for key, expected in expected_hashes.items():
        snapshot = snapshots_before.get(key)
        if not isinstance(snapshot, dict) or snapshot.get("sha256") != expected:
            raise EvaluationError(f"{key} hash in run does not match protocol: {run_path}")


def _check_run_timing(
    record: dict[str, Any], protocol: Protocol, sequence: dict[str, Any], rows: list[dict[str, Any]], run_path: Path
) -> int:
    count = record.get("decoded_frames")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or len(rows) != count:
        raise EvaluationError(f"frames.jsonl row count does not match complete decoded frame count: {run_path}")
    if record.get("declared_frame_count") is None or abs(int(record["declared_frame_count"]) - count) > int(protocol.raw["gates"]["frame_count_metadata_tolerance_frames"]):
        raise EvaluationError(f"decoded frame count metadata mismatch: {run_path}")
    fps = record.get("nominal_source_fps")
    if not _is_number(fps) or abs(float(fps) - float(sequence["raw"]["fps"])) > float(protocol.raw["gates"]["fps_metadata_tolerance"]):
        raise EvaluationError(f"source FPS in run differs from protocol: {run_path}")
    duration = record.get("run_wall_s")
    if not _is_number(duration) or float(duration) <= 0:
        raise EvaluationError(f"missing measured run wall duration: {run_path}")
    start_ns = record.get("run_start_monotonic_ns")
    if isinstance(start_ns, bool) or not isinstance(start_ns, int):
        raise EvaluationError(f"missing monotonic run origin: {run_path}")
    if not _is_number(record.get("run_start_unix_s")) or float(record["run_start_unix_s"]) <= 0.0:
        raise EvaluationError(f"missing wall-clock replay origin: {run_path}")
    return start_ns


def _frame_row_timing(
    row: dict[str, Any], ordinal: int, fps: float, start_ns: int, frames_path: Path
) -> tuple[int, int]:
    response = row.get("response")
    if not isinstance(response, dict):
        raise EvaluationError(f"worker result is not an object at ordinal {ordinal}: {frames_path}")
    seq = response.get("seq")
    if isinstance(seq, bool) or seq != ordinal:
        raise EvaluationError(f"worker response sequence mismatch at ordinal {ordinal}: {frames_path}")
    video_time = row.get("video_time_s")
    if not _is_number(video_time) or abs(float(video_time) - ordinal / fps) > 1e-6:
        raise EvaluationError(f"frame source timestamp mismatch at ordinal {ordinal}: {frames_path}")
    capture_ns = response.get("capture_mono_ns")
    receive_ns = row.get("received_mono_ns")
    send_ns = row.get("send_mono_ns")
    if isinstance(capture_ns, bool) or not isinstance(capture_ns, int):
        raise EvaluationError(f"worker result lacks capture timestamp at ordinal {ordinal}: {frames_path}")
    if isinstance(receive_ns, bool) or not isinstance(receive_ns, int) or receive_ns < capture_ns:
        raise EvaluationError(f"invalid capture-to-consumer timestamp at ordinal {ordinal}: {frames_path}")
    if isinstance(send_ns, bool) or not isinstance(send_ns, int) or send_ns < capture_ns or receive_ns < send_ns:
        raise EvaluationError(f"invalid worker send/receive timestamp at ordinal {ordinal}: {frames_path}")
    expected_capture_ns = int(round(start_ns * 1e-9 * 1_000_000_000.0 + float(video_time) * 1_000_000_000.0))
    if abs(capture_ns - expected_capture_ns) > 2:
        raise EvaluationError(f"capture timestamp is not source-aligned at ordinal {ordinal}: {frames_path}")
    return capture_ns, receive_ns


def _check_frame_timing(rows: list[dict[str, Any]], fps: float, start_ns: int, frames_path: Path) -> None:
    previous_capture_ns = previous_receive_ns = None
    for ordinal, row in enumerate(rows):
        capture_ns, receive_ns = _frame_row_timing(row, ordinal, fps, start_ns, frames_path)
        if previous_capture_ns is not None and capture_ns <= previous_capture_ns:
            raise EvaluationError(f"capture timestamps did not increase at ordinal {ordinal}: {frames_path}")
        if previous_receive_ns is not None and receive_ns <= previous_receive_ns:
            raise EvaluationError(f"consumer timestamps did not increase at ordinal {ordinal}: {frames_path}")
        previous_capture_ns, previous_receive_ns = capture_ns, receive_ns
        if row["response"].get("error") is not None:
            raise EvaluationError(f"worker frame error at ordinal {ordinal}: {frames_path}")


def _check_frame_identity(
    rows: list[dict[str, Any]], expected_identity: dict[str, Any], run_id: Any, frames_path: Path
) -> None:
    for ordinal, row in enumerate(rows):
        if row.get("experiment_run_id") != expected_identity["experiment_run_id"] or row.get("run_id") != run_id:
            raise EvaluationError(f"frame row identity mismatch at ordinal {ordinal}: {frames_path}")
        if (
            row.get("mode") != expected_identity["mode"]
            or row.get("sequence") != expected_identity["sequence"]
            or row.get("repeat") != expected_identity["repeat"]
            or row.get("ordinal") != ordinal
        ):
            raise EvaluationError(f"frame rows are not complete and contiguous at ordinal {ordinal}: {frames_path}")
        response = row.get("response")
        if not isinstance(response, dict) or response.get("seq") != ordinal or response.get("frame_id") != f"worker-{ordinal}":
            raise EvaluationError(f"worker response identity mismatch at ordinal {ordinal}: {frames_path}")
        if isinstance(row.get("received_mono_ns"), bool) or not isinstance(row.get("received_mono_ns"), int):
            raise EvaluationError(f"missing consumer receive timestamp at ordinal {ordinal}: {frames_path}")
        if response.get("error") is not None:
            raise EvaluationError(f"worker frame error at ordinal {ordinal}: {frames_path}")


def _validate_run(protocol: Protocol, mode: str, sequence: dict[str, Any], repeat: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = sequence["raw"]["name"]
    session_dir, run_path, frames_path = _run_paths(protocol, mode, name, repeat)
    record = _read_json(run_path)
    expected_identity = {
        "schema": RUN_SCHEMA,
        "experiment_run_id": protocol.raw["run_id"],
        "mode": mode,
        "sequence": name,
        "repeat": repeat,
    }
    _check_run_completion(record, expected_identity, run_path)
    _check_worker_command(record, protocol, mode, session_dir, run_path)
    _check_run_inputs(record, protocol, sequence, frames_path, run_path)
    rows = _records(frames_path)
    start_ns = _check_run_timing(record, protocol, sequence, rows, run_path)
    _check_frame_timing(rows, float(sequence["raw"]["fps"]), start_ns, frames_path)
    _check_frame_identity(rows, expected_identity, record.get("run_id"), frames_path)
    return record, rows


def _nonnegative_counter(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_BACKLOG_QUEUES = frozenset({"index", "ba", "writer", "overlap"})


def _health_flag_state(
    response: dict[str, Any], missing_by_field: dict[str, int], unhealthy_by_field: dict[str, int]
) -> tuple[bool, bool]:
    missing = False
    healthy = True
    for field in _WORKMAP_HEALTH_FIELDS:
        value = response.get(field)
        if not isinstance(value, bool):
            missing_by_field[field] += 1
            missing = True
            healthy = False
        elif not value:
            unhealthy_by_field[field] += 1
            healthy = False
    return missing, healthy


def _failure_counter_state(
    response: dict[str, Any], missing_by_field: dict[str, int], failure_counts: dict[str, int]
) -> tuple[bool, bool]:
    missing = False
    healthy = True
    for field in _WORKMAP_FAILURE_COUNTERS:
        value = response.get(field)
        if not _nonnegative_counter(value):
            missing_by_field[field] += 1
            missing = True
            healthy = False
        elif value:
            failure_counts[field] += 1
            healthy = False
    return missing, healthy


def _record_queue_samples(
    response: dict[str, Any],
    missing_by_field: dict[str, int],
    queue_values: dict[str, list[tuple[bool | None, float | None]]],
) -> bool:
    """Append one frame's pending/age sample per queue; False if any is missing."""
    complete = True
    for name, pending_field, age_field in _WORKMAP_QUEUE_FIELDS:
        pending = response.get(pending_field)
        age = response.get(age_field)
        pending_ok = isinstance(pending, bool)
        age_ok = _is_number(age) and float(age) >= 0.0
        if not pending_ok:
            missing_by_field[pending_field] += 1
        if not age_ok:
            missing_by_field[age_field] += 1
        if not pending_ok or not age_ok:
            complete = False
        queue_values[name].append(
            (pending if pending_ok else None, float(age) if age_ok else None)
        )
    return complete


def _queue_window(samples: list[tuple[bool | None, float | None]]) -> dict[str, Any]:
    valid_ages = [age for pending, age in samples if pending is True and age is not None]
    all_valid = all(pending is not None and age is not None for pending, age in samples)
    ages = [float(age) for _, age in samples if age is not None]
    pending_flags = [pending for pending, _ in samples]
    continuous = bool(
        all_valid
        and pending_flags
        and all(pending_flags)
        and len(ages) == len(samples)
        and all(later >= earlier for earlier, later in zip(ages, ages[1:]))
        and ages[-1] > ages[0]
    )
    return {
        "pending_frames": sum(pending is True for pending, _ in samples),
        "oldest_pending_age_ms": max(valid_ages) if valid_ages else 0.0,
        "start_age_ms": ages[0] if ages else None,
        "end_age_ms": ages[-1] if ages else None,
        "age_grew_continuously": continuous,
    }


def _idle_queue_report() -> dict[str, Any]:
    return {
        name: {
            "pending_frames": 0,
            "oldest_pending_age_ms": None,
            "start_age_ms": None,
            "end_age_ms": None,
            "age_grew_continuously": False,
        }
        for name, _, _ in _WORKMAP_QUEUE_FIELDS
    }


def _window_queues(
    window: dict[str, Any],
    selected: list[int],
    queue_values: dict[str, list[tuple[bool | None, float | None]]],
    backlog_growth_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    queue_report: dict[str, Any] = {}
    for name, _, _ in _WORKMAP_QUEUE_FIELDS:
        item = _queue_window([queue_values[name][index] for index in selected])
        queue_report[name] = item
        if name in _BACKLOG_QUEUES and item["age_grew_continuously"]:
            backlog_growth_windows.append({
                "start_s": window["start_s"],
                "end_s": window["end_s"],
                "queue": name,
                "start_age_ms": item["start_age_ms"],
                "end_age_ms": item["end_age_ms"],
            })
    return queue_report


def _workmap_health_metrics(
    rows: list[dict[str, Any]],
    relative_receives: list[float],
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed on missing or unhealthy full-stage worker diagnostics."""
    frame_metrics: list[dict[str, Any]] = []
    missing_by_field = {
        field: 0
        for field in (
            *_WORKMAP_HEALTH_FIELDS,
            *_WORKMAP_FAILURE_COUNTERS,
            *(
                field
                for _, pending_field, age_field in _WORKMAP_QUEUE_FIELDS
                for field in (pending_field, age_field)
            ),
        )
    }
    unhealthy_by_field = {field: 0 for field in _WORKMAP_HEALTH_FIELDS}
    failure_counts = {field: 0 for field in _WORKMAP_FAILURE_COUNTERS}
    queue_values: dict[str, list[tuple[bool | None, float | None]]] = {
        name: [] for name, _, _ in _WORKMAP_QUEUE_FIELDS
    }

    for row in rows:
        response = row.get("response")
        response = response if isinstance(response, dict) else {}
        flags_missing, flags_healthy = _health_flag_state(response, missing_by_field, unhealthy_by_field)
        counters_missing, counters_healthy = _failure_counter_state(response, missing_by_field, failure_counts)
        queues_complete = _record_queue_samples(response, missing_by_field, queue_values)
        frame_metrics.append({
            "healthy": flags_healthy and counters_healthy and queues_complete,
            "missing": flags_missing or counters_missing or not queues_complete,
        })

    windows_with_ages: list[dict[str, Any]] = []
    backlog_growth_windows: list[dict[str, Any]] = []
    for window in windows:
        selected = [
            index
            for index, receive_s in enumerate(relative_receives)
            if float(window["start_s"]) <= receive_s < float(window["end_s"])
        ]
        queue_report = (
            _window_queues(window, selected, queue_values, backlog_growth_windows)
            if selected
            else _idle_queue_report()
        )
        windows_with_ages.append({
            "start_s": window["start_s"],
            "end_s": window["end_s"],
            "frames_received": len(selected),
            "queues": queue_report,
        })

    last_frame = frame_metrics[-1] if frame_metrics else {"healthy": False, "missing": True}
    frame_health_pass = bool(frame_metrics) and all(item["healthy"] for item in frame_metrics)
    frame_diagnostics_complete = bool(frame_metrics) and not any(missing_by_field.values())
    backlog_windows_pass = bool(windows) and not backlog_growth_windows
    return {
        "pass": frame_health_pass and frame_diagnostics_complete and backlog_windows_pass,
        "required_health_fields": list(_WORKMAP_HEALTH_FIELDS),
        "required_failure_counters": list(_WORKMAP_FAILURE_COUNTERS),
        "frame_count": len(rows),
        "frame_health_pass": frame_health_pass,
        "frame_diagnostics_complete": frame_diagnostics_complete,
        "missing_by_field": missing_by_field,
        "unhealthy_frame_count_by_field": unhealthy_by_field,
        "frames_with_overflow_or_errors_by_field": failure_counts,
        "last_frame_health_pass": bool(last_frame["healthy"]),
        "last_frame_health_missing": bool(last_frame["missing"]),
        "ten_second_pending_age_windows": windows_with_ages,
        "backlog_growth_windows": backlog_growth_windows,
        "backlog_windows_pass": backlog_windows_pass,
    }


def _run_metrics(record: dict[str, Any], rows: list[dict[str, Any]], sequence: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    duration = float(record["run_wall_s"])
    fps = len(rows) / duration
    start_ns = record.get("run_start_monotonic_ns")
    if isinstance(start_ns, bool) or not isinstance(start_ns, int):
        raise EvaluationError("run has no valid monotonic timing origin")
    replay_start_wall_s = (
        float(record["run_start_unix_s"])
        if _is_number(record.get("run_start_unix_s"))
        else None
    )
    replay_end_wall_s = (
        replay_start_wall_s + (rows[-1]["received_mono_ns"] - start_ns) * 1e-9
        if replay_start_wall_s is not None
        else None
    )
    relative_receives = [(row["received_mono_ns"] - start_ns) * 1e-9 for row in rows]
    if any(not math.isfinite(value) or value < 0 for value in relative_receives):
        raise EvaluationError("frame consumer timestamps precede run start")
    if any(value > duration for value in relative_receives):
        raise EvaluationError("frame consumer timestamps fall outside measured run duration")
    window_s = float(gates["window_s"])
    windows = []
    full_windows = int(math.floor(duration / window_s))
    for index in range(full_windows):
        start, end = index * window_s, (index + 1) * window_s
        count = sum(start <= value < end for value in relative_receives)
        measured_fps = count / window_s
        windows.append({"start_s": start, "end_s": end, "frames_received": count, "e2e_fps": measured_fps, "pass": measured_fps > float(gates["min_e2e_fps"])})
    workmap_health = (
        _workmap_health_metrics(rows, relative_receives, windows)
        if record.get("mode") == "full"
        else None
    )
    latencies_ms = [
        (row["received_mono_ns"] - row["response"]["capture_mono_ns"]) / 1_000_000.0
        for row in rows
        if isinstance(row["response"].get("capture_mono_ns"), int) and not isinstance(row["response"].get("capture_mono_ns"), bool)
    ]
    foreground_ms = [
        float(row["response"]["core_wall_ms"])
        for row in rows
        if _is_number(row["response"].get("core_wall_ms"))
    ]
    if len(latencies_ms) != len(rows):
        raise EvaluationError("one or more worker results lack capture-to-consumer identity timestamps")
    if len(foreground_ms) != len(rows):
        raise EvaluationError("one or more worker results lack measured foreground processing time")
    visual_rows = [row for row in rows if _visual_fresh(row)]
    segment_metrics: list[dict[str, Any]] = []
    for segment in sequence["raw"]["segments"]:
        if segment.get("role") not in {"new_region", "new_region_revisit"}:
            continue
        duration_s = segment["end_s"] - segment["start_s"]
        segment_rows = [row for row in rows if segment["start_s"] <= row["video_time_s"] < segment["end_s"]]
        visual_count = sum(_visual_fresh(row) for row in segment_rows)
        visual_fps = visual_count / duration_s
        segment_metrics.append({
            "name": segment["name"],
            "role": segment["role"],
            "start_s": segment["start_s"],
            "end_s": segment["end_s"],
            "frames_in_segment": len(segment_rows),
            "fresh_visual_pose_frames": visual_count,
            "fresh_visual_pose_fps": visual_fps,
            "pass": visual_fps > float(gates["min_visual_fresh_fps"]),
        })
    return {
        "decoded_frames": len(rows),
        "run_wall_s_including_worker_drain": duration,
        "e2e_fps": fps,
        "e2e_fps_pass": fps > float(gates["min_e2e_fps"]),
        "ten_second_windows": windows,
        "ten_second_window_gate_pass": bool(windows) and all(window["pass"] for window in windows),
        "trailing_partial_window_s": duration - full_windows * window_s,
        "fresh_visual_pose_frames": len(visual_rows),
        "fresh_visual_pose_fps_full_run": len(visual_rows) / duration,
        "new_region_segments": segment_metrics,
        "workmap_health": workmap_health,
        "timed_replay_wall_bounds": (
            {
                "start_unix_s": replay_start_wall_s,
                "end_unix_s": replay_end_wall_s,
                "start_monotonic_ns": start_ns,
            }
            if replay_start_wall_s is not None and replay_end_wall_s is not None
            else None
        ),
        "capture_to_consumer_ms": {
            "p95": _percentile(latencies_ms, 0.95),
            "p99": _percentile(latencies_ms, 0.99),
            "max": max(latencies_ms) if latencies_ms else None,
            "pass": (
                (_percentile(latencies_ms, 0.95) is not None and _percentile(latencies_ms, 0.95) <= float(gates["max_capture_to_consumer_p95_ms"]))
                and (_percentile(latencies_ms, 0.99) is not None and _percentile(latencies_ms, 0.99) <= float(gates["max_capture_to_consumer_p99_ms"]))
            ),
        },
        "foreground_core_wall_ms": {
            "p95": _percentile(foreground_ms, 0.95),
            "max": max(foreground_ms) if foreground_ms else None,
            "pass": _percentile(foreground_ms, 0.95) is not None and _percentile(foreground_ms, 0.95) <= float(gates["max_foreground_p95_ms"]),
        },
    }


def _segment_roles(sequence: dict[str, Any]) -> list[str]:
    return [segment.get("role") for segment in sequence["raw"]["segments"] if segment.get("role") is not None]


def _has_revisit_contract(protocol: Protocol) -> bool:
    required = ["old_map", "new_region", "occlusion_or_loss", "new_region_revisit", "old_map_return"]
    for sequence in protocol.sequences:
        roles = _segment_roles(sequence)
        position = 0
        for role in roles:
            if role == required[position]:
                position += 1
                if position == len(required):
                    return True
    return False


def _wall_stamp_in_replay(value: Any, start_s: Any, end_s: Any) -> bool:
    return (
        _is_number(value)
        and _is_number(start_s)
        and _is_number(end_s)
        and float(start_s) <= float(value) <= float(end_s)
    )


def _event_int(event: dict[str, Any], key: str) -> int | None:
    value = event.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


_WORKMAP_EVIDENCE_COUNTERS = (
    "workmap_points_created",
    "workmap_points_used",
    "workmap_keyframes_ready",
    "workmap_reloc_accepted",
    "workmap_ba_submitted",
    "workmap_ba_applied",
    "workmap_ba_stale",
    "workmap_scale_applied",
)


def _session_identity(rows: list[dict[str, Any]]) -> tuple[str | None, bool]:
    response_session_ids = {
        row["response"].get("workmap_session_id")
        for row in rows
        if isinstance(row["response"].get("workmap_session_id"), str)
        and row["response"]["workmap_session_id"]
    }
    session_id = next(iter(response_session_ids)) if len(response_session_ids) == 1 else None
    session_identity_pass = session_id is not None and all(
        row["response"].get("workmap_session_id") == session_id for row in rows
    )
    return session_id, session_identity_pass


def _replay_bounds(timed_replay_wall_bounds: dict[str, Any] | None) -> tuple[Any, Any, Any, bool]:
    bounds = timed_replay_wall_bounds if isinstance(timed_replay_wall_bounds, dict) else {}
    replay_start_wall_s = bounds.get("start_unix_s")
    replay_end_wall_s = bounds.get("end_unix_s")
    replay_start_monotonic_ns = bounds.get("start_monotonic_ns")
    replay_wall_bounds_pass = (
        _is_number(replay_start_wall_s)
        and _is_number(replay_end_wall_s)
        and float(replay_end_wall_s) >= float(replay_start_wall_s)
        and isinstance(replay_start_monotonic_ns, int)
        and not isinstance(replay_start_monotonic_ns, bool)
        and replay_start_monotonic_ns >= 0
    )
    return replay_start_wall_s, replay_end_wall_s, replay_start_monotonic_ns, replay_wall_bounds_pass


def _session_events(events_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        _reject_symlink_path(events_path, allow_missing=True, what="workmap session events")
        if not events_path.is_file():
            return [], f"missing events.jsonl: {events_path}"
        return _records(events_path), None
    except (EvaluationError, OSError) as exc:
        return [], str(exc)


def _first_use_events(
    events: list[dict[str, Any]], rows_by_ordinal: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    first_use_events = []
    for event in events:
        if event.get("event") != "first_new_landmark_pnp_use":
            continue
        landmark_id = event.get("landmark_id")
        created = _event_int(event, "created_ordinal")
        used = _event_int(event, "used_ordinal")
        if (
            isinstance(landmark_id, int)
            and not isinstance(landmark_id, bool)
            and landmark_id < 0
            and created is not None
            and used == created + 1
            and created in rows_by_ordinal
            and used in rows_by_ordinal
            and _visual_fresh(rows_by_ordinal[used])
            # Older single-tracker sessions have no output selector. "workmap" marks a
            # fresh published pose whose PnP used session landmarks of that ordinal.
            and rows_by_ordinal[used]["response"].get("workmap_output_source", "workmap")
            == "workmap"
            and _nonnegative_counter(
                rows_by_ordinal[used]["response"].get("workmap_points_used")
            )
            and rows_by_ordinal[used]["response"]["workmap_points_used"] > 0
        ):
            first_use_events.append({
                "landmark_id": landmark_id,
                "created_ordinal": created,
                "used_ordinal": used,
                "revision": event.get("revision"),
            })
    return first_use_events


def _ba_applied_rows(rows_by_ordinal: dict[int, dict[str, Any]]) -> list[tuple[int, int]]:
    applied_rows: list[tuple[int, int]] = []
    for ordinal, row in rows_by_ordinal.items():
        response = row["response"]
        revision = response.get("workmap_revision")
        if (
            response.get("workmap_ba_applied_this_frame") is True
            and isinstance(revision, int)
            and not isinstance(revision, bool)
            and revision >= 0
        ):
            applied_rows.append((ordinal, revision))
    return applied_rows


def _updated_negative_ids(commit: dict[str, Any]) -> set[int]:
    updated = commit.get("updated_landmarks")
    updated_ids = set()
    if isinstance(updated, dict):
        for key in updated:
            try:
                point_id = int(key)
            except (TypeError, ValueError):
                continue
            if point_id < 0:
                updated_ids.add(point_id)
    return updated_ids


def _ba_use_confirms(
    use: dict[str, Any],
    revision: int,
    updated_ids: set[int],
    applied_ordinal: int,
    applied_row: dict[str, Any],
    rows_by_ordinal: dict[int, dict[str, Any]],
) -> tuple[bool, int | None]:
    """Whether a later fresh PnP frame used landmarks this BA revision updated."""
    used_ordinal = _event_int(use, "used_ordinal")
    raw_used_ids = use.get("landmark_ids")
    used_ids = {
        value for value in raw_used_ids
        if isinstance(value, int) and not isinstance(value, bool) and value < 0
    } if isinstance(raw_used_ids, list) else set()
    use_revision = _event_int(use, "revision")
    used_row = rows_by_ordinal.get(used_ordinal) if used_ordinal is not None else None
    used_row_revision = (
        used_row["response"].get("workmap_revision")
        if used_row is not None
        else None
    )
    confirmed = bool(
        used_ordinal is not None
        and used_ordinal in rows_by_ordinal
        and _visual_fresh(applied_row)
        and used_row is not None
        and isinstance(used_row_revision, int)
        and not isinstance(used_row_revision, bool)
        and used_row_revision >= revision
        and _visual_fresh(used_row)
        and used_row["response"].get("workmap_output_source", "workmap") == "workmap"
        and _nonnegative_counter(
            used_row["response"].get("workmap_points_used")
        )
        and used_row["response"]["workmap_points_used"] > 0
        and used_ordinal >= applied_ordinal
        and use_revision is not None
        and use_revision >= revision
        and updated_ids.intersection(used_ids)
    )
    return confirmed, used_ordinal


def _ba_pairs(
    ba_commits: list[dict[str, Any]],
    ba_uses: list[dict[str, Any]],
    rows_by_ordinal: dict[int, dict[str, Any]],
    replay_start_wall_s: Any,
    replay_end_wall_s: Any,
    replay_start_monotonic_ns: Any,
) -> list[dict[str, Any]]:
    applied_rows = _ba_applied_rows(rows_by_ordinal)
    ba_pairs = []
    for commit in ba_commits:
        revision = _event_int(commit, "revision")
        updated_ids = _updated_negative_ids(commit)
        submit_to_apply_ms = commit.get("submit_to_apply_ms")
        if (
            revision is None
            or not _is_number(submit_to_apply_ms)
            or float(submit_to_apply_ms) < 0.0
            or not _wall_stamp_in_replay(
                commit.get("stamp_unix_s"), replay_start_wall_s, replay_end_wall_s
            )
        ):
            continue
        for applied_ordinal, applied_revision in applied_rows:
            if applied_revision < revision:
                continue
            applied_row = rows_by_ordinal[applied_ordinal]
            applied_row_wall_s = float(replay_start_wall_s) + (
                applied_row["received_mono_ns"] - replay_start_monotonic_ns
            ) * 1e-9
            if not _wall_stamp_in_replay(
                commit.get("stamp_unix_s"), replay_start_wall_s, applied_row_wall_s
            ):
                continue
            for use in ba_uses:
                confirmed, used_ordinal = _ba_use_confirms(
                    use, revision, updated_ids, applied_ordinal, applied_row, rows_by_ordinal
                )
                if confirmed:
                    ba_pairs.append({
                        "revision": revision,
                        "applied_ordinal": applied_ordinal,
                        "frame_workmap_revision": applied_revision,
                        "used_ordinal": used_ordinal,
                        "submit_to_apply_ms": float(submit_to_apply_ms),
                    })
    return ba_pairs


def _eligible_queued_keyframes(events: list[dict[str, Any]]) -> tuple[set[str], bool]:
    eligible_queued_names: set[str] = set()
    index_evidence_complete = True
    for event in events:
        if event.get("event") != "keyframe_queued":
            continue
        name = event.get("name")
        eligible = event.get("index_requested")
        if not isinstance(name, str) or not name or not isinstance(eligible, bool):
            index_evidence_complete = False
        elif eligible:
            eligible_queued_names.add(name)
    return eligible_queued_names, index_evidence_complete


def _first_index_requests(
    events: list[dict[str, Any]], replay_start_wall_s: Any, replay_end_wall_s: Any
) -> dict[str, dict[str, Any]]:
    first_requests: dict[str, dict[str, Any]] = {}
    for event in events:
        name = event.get("name")
        if (
            event.get("event") == "reference_index_requested"
            and isinstance(name, str)
            and name
            and _event_int(event, "revision") is not None
            and _wall_stamp_in_replay(
                event.get("stamp_unix_s"), replay_start_wall_s, replay_end_wall_s
            )
        ):
            previous = first_requests.get(name)
            if previous is None or float(event["stamp_unix_s"]) < float(previous["stamp_unix_s"]):
                first_requests[name] = event
    return first_requests


def _indexed_reference_events(
    events: list[dict[str, Any]], replay_start_wall_s: Any, replay_end_wall_s: Any
) -> dict[str, list[dict[str, Any]]]:
    indexed_events: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        name = event.get("name")
        if (
            event.get("event") == "reference_indexed"
            and isinstance(name, str)
            and name
            and isinstance(event.get("provider_reference_name"), str)
            and event["provider_reference_name"]
            and _event_int(event, "revision") is not None
            and _wall_stamp_in_replay(
                event.get("stamp_unix_s"), replay_start_wall_s, replay_end_wall_s
            )
        ):
            indexed_events.setdefault(name, []).append(event)
    return indexed_events


def _index_ready_latencies(
    expected_names: set[str],
    first_requests: dict[str, dict[str, Any]],
    durable_names: set[str],
    indexed_events: dict[str, list[dict[str, Any]]],
) -> tuple[list[float], list[str]]:
    index_latencies_ms = []
    missing_ready_names = []
    for name in sorted(expected_names):
        request = first_requests.get(name)
        if request is None or name not in durable_names:
            missing_ready_names.append(name)
            continue
        indexed = min(
            (
                event for event in indexed_events.get(name, ())
                if int(event["revision"]) >= int(request["revision"])
                and float(event["stamp_unix_s"]) >= float(request["stamp_unix_s"])
            ),
            key=lambda event: float(event["stamp_unix_s"]),
            default=None,
        )
        if indexed is None:
            missing_ready_names.append(name)
            continue
        index_latencies_ms.append(
            (float(indexed["stamp_unix_s"]) - float(request["stamp_unix_s"])) * 1000.0
        )
    return index_latencies_ms, missing_ready_names


def _workmap_evidence(
    rows: list[dict[str, Any]],
    sequence: dict[str, Any],
    session_dir: Path,
    health: dict[str, Any] | None,
    timed_replay_wall_bounds: dict[str, Any] | None,
) -> dict[str, Any]:
    observed_fields = {
        key
        for row in rows
        for key in row["response"]
        if key.startswith("workmap_") or key == "reference_scope"
    }
    counters = {
        key: max(
            (
                int(row["response"][key])
                for row in rows
                if _nonnegative_counter(row["response"].get(key))
            ),
            default=None,
        )
        for key in _WORKMAP_EVIDENCE_COUNTERS
    }
    session_id, session_identity_pass = _session_identity(rows)
    (
        replay_start_wall_s,
        replay_end_wall_s,
        replay_start_monotonic_ns,
        replay_wall_bounds_pass,
    ) = _replay_bounds(timed_replay_wall_bounds)
    events_path = session_dir / "events.jsonl"
    events, events_error = _session_events(events_path)
    matching_session_events = (
        [event for event in events if event.get("session_id") == session_id]
        if session_identity_pass
        else []
    )
    invalid_session_event_count = len(events) - len(matching_session_events)
    event_stream_pass = (
        events_error is None
        and replay_wall_bounds_pass
        and session_identity_pass
        and bool(events)
        and invalid_session_event_count == 0
    )
    rows_by_ordinal = {
        int(row["ordinal"]): row
        for row in rows
        if isinstance(row.get("ordinal"), int) and not isinstance(row.get("ordinal"), bool)
    }

    first_use_events = _first_use_events(matching_session_events, rows_by_ordinal)
    first_use_pass = event_stream_pass and bool(first_use_events)

    ba_commits = [
        event for event in matching_session_events
        if event.get("event") == "ba_geometry_committed"
    ]
    ba_uses = [
        event for event in matching_session_events
        if event.get("event") == "ba_geometry_used"
    ]
    ba_pairs = _ba_pairs(
        ba_commits,
        ba_uses,
        rows_by_ordinal,
        replay_start_wall_s,
        replay_end_wall_s,
        replay_start_monotonic_ns,
    )
    matched_ba_revisions = {pair["revision"] for pair in ba_pairs}
    ba_apply_values = [
        float(event["submit_to_apply_ms"])
        for event in ba_commits
        if _event_int(event, "revision") in matched_ba_revisions
        and _is_number(event.get("submit_to_apply_ms"))
        and float(event["submit_to_apply_ms"]) >= 0.0
        and _is_number(event.get("stamp_unix_s"))
    ]
    ba_pass = (
        event_stream_pass
        and bool(ba_commits)
        and bool(ba_pairs)
        and len(ba_apply_values) == len(matched_ba_revisions)
        and _percentile(ba_apply_values, 0.95) is not None
        and _percentile(ba_apply_values, 0.95) <= _WORKMAP_MAX_BA_APPLY_MS
    )

    eligible_queued_names, index_evidence_complete = _eligible_queued_keyframes(matching_session_events)
    durable_names = {
        event["name"]
        for event in matching_session_events
        if event.get("event") == "keyframe_durable"
        and isinstance(event.get("name"), str)
        and event["name"]
        and _wall_stamp_in_replay(event.get("stamp_unix_s"), replay_start_wall_s, replay_end_wall_s)
    }
    first_requests = _first_index_requests(matching_session_events, replay_start_wall_s, replay_end_wall_s)
    indexed_events = _indexed_reference_events(matching_session_events, replay_start_wall_s, replay_end_wall_s)
    expected_names = eligible_queued_names | first_requests.keys()
    index_latencies_ms, missing_ready_names = _index_ready_latencies(
        expected_names, first_requests, durable_names, indexed_events
    )
    index_pass = (
        event_stream_pass
        and index_evidence_complete
        and bool(expected_names)
        and not missing_ready_names
        and len(index_latencies_ms) == len(expected_names)
        and max(index_latencies_ms) <= _WORKMAP_MAX_INDEX_READY_MS
    )
    roles = [segment.get("role") for segment in sequence["raw"]["segments"]]
    event_evidence_pass = first_use_pass and ba_pass and index_pass
    return {
        "diagnostic_fields_present": sorted(observed_fields),
        "cumulative_counters_observed": counters,
        "session_writer_healthy_ever": any(
            row["response"].get("workmap_writer_healthy") is True for row in rows
        ),
        "frame_health": health,
        "session_events": {
            "path": str(events_path),
            "readable": events_error is None and events_path.is_file(),
            "error": events_error,
            "session_identity_pass": session_identity_pass,
            "event_count": len(events),
            "invalid_session_event_count": invalid_session_event_count,
            "timed_replay_wall_bounds": {
                "start_unix_s": replay_start_wall_s,
                "end_unix_s": replay_end_wall_s,
            },
            "timed_replay_wall_bounds_pass": replay_wall_bounds_pass,
            "event_stream_pass": event_stream_pass,
        },
        "first_new_landmark_pnp_use": {
            "pass": first_use_pass,
            "valid_next_frame_event_count": len(first_use_events),
            "examples": first_use_events[:10],
        },
        "ba_commit_and_use_during_replay": {
            "pass": ba_pass,
            "commit_event_count": len(ba_commits),
            "commit_events_matched_to_applied_and_used_frames": len(matched_ba_revisions),
            "commit_events_without_timed_apply_and_use": len(ba_commits) - len(matched_ba_revisions),
            "p95_submit_to_apply_ms": _percentile(ba_apply_values, 0.95),
            "max_submit_to_apply_ms": max(ba_apply_values) if ba_apply_values else None,
            "max_submit_to_apply_gate_ms": _WORKMAP_MAX_BA_APPLY_MS,
            "examples": ba_pairs[:10],
            "close_only_commit_is_not_counted": True,
        },
        "keyframe_durable_to_index_ready": {
            "pass": index_pass,
            "eligible_keyframe_count": len(expected_names),
            "eligibility_events_complete": index_evidence_complete,
            "missing_ready_keyframes": missing_ready_names[:10],
            "missing_ready_keyframe_count": len(missing_ready_names),
            "indexed_keyframe_count": len(index_latencies_ms),
            "max_durable_to_index_ready_ms": max(index_latencies_ms) if index_latencies_ms else None,
            "max_ready_gate_ms": _WORKMAP_MAX_INDEX_READY_MS,
        },
        "event_evidence_pass": event_evidence_pass,
        "stage_behavior_assessment": (
            "SESSION_EVENT_BACKED; next-frame landmark use, BA geometry use during replay, "
            "and durable-to-index readiness are verified from events.jsonl and timed frame rows"
            if event_evidence_pass
            else (
                "SESSION_EVENT_EVIDENCE_FAILED_OR_INCOMPLETE; one or more next-frame landmark, "
                "BA geometry use, or durable-to-index checks lack valid in-replay evidence"
            )
        ),
        "required_revisit_roles": ["old_map", "new_region", "occlusion_or_loss", "new_region_revisit", "old_map_return"],
        "sequence_roles": roles,
    }


def _resource_sampling_pass(resource: Any) -> bool:
    if not isinstance(resource, dict):
        return False
    errors = resource.get("errors")
    peaks = (
        "peak_cpu_percent_one_core_100",
        "peak_ram_mib",
        "peak_process_tree_rss_hwm_mib",
        "peak_gpu_utilization_percent_device",
        "peak_vram_mib_device",
    )
    return (
        resource.get("sampling_status") == "COMPLETE"
        and isinstance(resource.get("source"), str)
        and "/proc/" in resource["source"]
        and "nvidia-smi" in resource["source"]
        and resource.get("cpu_scope") == "worker process tree including child processes"
        and resource.get("gpu_scope") == "device-level total for GPU 0; includes other GPU users"
        and _is_number(resource.get("sample_interval_s"))
        and 0.0 < float(resource["sample_interval_s"]) <= 1.0
        and isinstance(resource.get("samples"), int)
        and not isinstance(resource.get("samples"), bool)
        and resource["samples"] > 1
        and isinstance(resource.get("gpu_samples"), int)
        and not isinstance(resource.get("gpu_samples"), bool)
        and resource["gpu_samples"] > 0
        and isinstance(errors, list)
        and not errors
        and all(_is_number(resource.get(key)) for key in peaks)
        and 0.0 <= float(resource["peak_cpu_percent_one_core_100"])
        and float(resource["peak_ram_mib"]) >= 0.0
        and float(resource["peak_process_tree_rss_hwm_mib"]) >= 0.0
        and 0.0 <= float(resource["peak_gpu_utilization_percent_device"]) <= 100.0
        and float(resource["peak_vram_mib_device"]) >= 0.0
    )


_RunCell = tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]


def _report_preflight(protocol_path: str | Path) -> tuple[Protocol, Path]:
    protocol = load_protocol(protocol_path)
    snapshot = protocol.output_root / "protocol.json"
    if not snapshot.is_file():
        raise EvaluationError(f"missing preregistered protocol snapshot: {snapshot}")
    saved_protocol = load_protocol(snapshot).raw
    if json.dumps(saved_protocol, ensure_ascii=False, sort_keys=True, allow_nan=False) != json.dumps(protocol.raw, ensure_ascii=False, sort_keys=True, allow_nan=False):
        raise EvaluationError("protocol changed after runs began; refusing to report")
    _verify_full_profile(protocol, "full")
    for sequence in protocol.sequences:
        current = _snapshot_inputs(protocol, sequence)
        expected_hashes = {
            "bundle": protocol.raw["bundle"]["sha256"],
            "profile": protocol.raw["profile"]["sha256"],
            "video": sequence["raw"]["video_sha256"],
            **(
                {"ground_truth": sequence["raw"]["ground_truth"]["sha256"]}
                if sequence["raw"].get("ground_truth") is not None
                else {}
            ),
        }
        for name, expected in expected_hashes.items():
            if current[name]["sha256"] != expected:
                raise EvaluationError(f"{name} changed after evaluation runs completed; refusing to report")
    return protocol, snapshot


def _collect_runs(protocol: Protocol) -> tuple[dict[tuple[str, str, int], _RunCell], set[str], list[int]]:
    runs: dict[tuple[str, str, int], _RunCell] = {}
    process_instances: set[str] = set()
    pids: list[int] = []
    for mode in MODES:
        for sequence in protocol.sequences:
            for repeat in range(1, REPETITIONS + 1):
                record, rows = _validate_run(protocol, mode, sequence, repeat)
                instance = str(record["worker_process_instance"])
                if instance in process_instances:
                    raise EvaluationError("multiple evaluation cells claim the same worker process instance")
                process_instances.add(instance)
                pids.append(int(record["process_id"]))
                metrics = _run_metrics(record, rows, sequence, protocol.raw["gates"])
                runs[(mode, sequence["raw"]["name"], repeat)] = (record, rows, metrics)
    expected_run_count = len(MODES) * len(protocol.sequences) * REPETITIONS
    if len(runs) != expected_run_count:
        raise EvaluationError(f"expected {expected_run_count} complete mode/sequence/repeat runs, found {len(runs)}")
    return runs, process_instances, pids


def _check_run_consistency(protocol: Protocol, runs: dict[tuple[str, str, int], _RunCell]) -> None:
    for sequence in protocol.sequences:
        name = sequence["raw"]["name"]
        records = [
            runs[(mode, name, repeat)][0]
            for mode in MODES
            for repeat in range(1, REPETITIONS + 1)
        ]
        counts = {record["decoded_frames"] for record in records}
        if len(counts) != 1:
            raise EvaluationError(f"mode/repetition runs decoded different frame counts for {name}: {sorted(counts)}")
        sizes = {tuple(record["source_size"]) for record in records}
        if len(sizes) != 1:
            raise EvaluationError(f"mode/repetition runs used different native video resolutions for {name}: {sorted(sizes)}")
        effective_cameras = {
            json.dumps(
                record.get("effective_query_camera"),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            for record in records
        }
        if len(effective_cameras) != 1:
            raise EvaluationError(f"mode/repetition runs used different effective camera calibrations for {name}")


def _mode_reports(
    name: str,
    runs: dict[tuple[str, str, int], _RunCell],
    all_run_metrics: list[dict[str, Any]],
    full_visual_gates: list[bool],
) -> dict[str, Any]:
    mode_reports: dict[str, Any] = {}
    baseline_fps: list[float] = []
    for mode in MODES:
        repetitions: list[dict[str, Any]] = []
        for repeat in range(1, REPETITIONS + 1):
            record, rows, metrics = runs[(mode, name, repeat)]
            all_run_metrics.append(metrics)
            metrics_with_id = {**metrics, "repeat": repeat, "process_id": record["process_id"], "process_instance": record["worker_process_instance"]}
            metrics_with_id["resource_peaks"] = record.get("resource_peaks")
            repetitions.append(metrics_with_id)
            if mode == "baseline":
                baseline_fps.append(metrics["e2e_fps"])
            if mode == "full":
                full_visual_gates.extend(bool(segment_metric["pass"]) for segment_metric in metrics["new_region_segments"])
        mode_reports[mode] = {"repetitions": repetitions}
    for repeat in range(1, REPETITIONS + 1):
        for mode in MODES[1:]:
            mode_reports[mode]["repetitions"][repeat - 1]["e2e_fps_delta_vs_baseline"] = (
                mode_reports[mode]["repetitions"][repeat - 1]["e2e_fps"] - baseline_fps[repeat - 1]
            )
    return mode_reports


def _full_mode_activity(
    protocol: Protocol, sequence: dict[str, Any], runs: dict[tuple[str, str, int], _RunCell]
) -> list[dict[str, Any]]:
    name = sequence["raw"]["name"]
    activity = []
    for repeat in range(1, REPETITIONS + 1):
        _, full_rows, full_metrics = runs[("full", name, repeat)]
        session_dir = protocol.output_root / "full" / name / f"repeat-{repeat}" / "session"
        activity.append(
            _workmap_evidence(
                full_rows,
                sequence,
                session_dir,
                full_metrics["workmap_health"],
                full_metrics["timed_replay_wall_bounds"],
            )
        )
    return activity


def _sequence_report(
    protocol: Protocol,
    sequence: dict[str, Any],
    runs: dict[tuple[str, str, int], _RunCell],
    mode_reports: dict[str, Any],
) -> dict[str, Any]:
    name = sequence["raw"]["name"]
    gt = sequence["raw"].get("ground_truth")
    return {
        "video": sequence["raw"]["video"],
        "video_sha256": sequence["raw"]["video_sha256"],
        "protocol_fps": sequence["raw"]["fps"],
        "segments": sequence["raw"]["segments"],
        "query_camera_base": protocol.raw["query_camera"],
        "effective_query_camera": runs[("baseline", name, 1)][0]["effective_query_camera"],
        "ground_truth": {
            "artifact_sha_bound": gt is not None,
            "format_validation": "NOT_VALIDATED_BY_P0_REPORT" if gt else "NO_GROUND_TRUTH_ARTIFACT",
            "path": gt["path"] if gt else None,
            "sha256": gt["sha256"] if gt else None,
            "absolute_accuracy_metrics": None,
            "accuracy_status": (
                "NOT_COMPUTED_BY_P0_REPORT_FORMAT_UNVERIFIED"
                if gt else "MISSING_INDEPENDENT_GROUND_TRUTH"
            ),
        },
        "modes": mode_reports,
    }


def _gate_results(
    protocol: Protocol,
    runs: dict[tuple[str, str, int], _RunCell],
    all_run_metrics: list[dict[str, Any]],
    full_visual_gates: list[bool],
    activity_by_sequence: dict[str, Any],
) -> dict[str, bool]:
    speed_pass = all(
        metric["e2e_fps_pass"] and metric["ten_second_window_gate_pass"]
        for metric in all_run_metrics
    )
    latency_pass = all(
        metric["capture_to_consumer_ms"]["pass"] and metric["foreground_core_wall_ms"]["pass"]
        for metric in all_run_metrics
    )
    has_new_region_segments = any(
        segment.get("role") in {"new_region", "new_region_revisit"}
        for sequence in protocol.sequences
        for segment in sequence["raw"]["segments"]
    )
    visual_pass = has_new_region_segments and bool(full_visual_gates) and all(full_visual_gates)
    ground_truth_artifact_sha_bound = all(
        sequence["raw"].get("ground_truth") is not None for sequence in protocol.sequences
    )
    ground_truth_format_validated = False
    full_evidences = [
        evidence
        for evidences in activity_by_sequence.values()
        for evidence in evidences
    ]
    feature_evidence_available = all(
        all(evidence["cumulative_counters_observed"][key] is not None for key in ("workmap_points_created", "workmap_points_used", "workmap_keyframes_ready", "workmap_ba_applied"))
        for evidence in full_evidences
    )
    full_run_metrics = [
        metrics
        for (mode, _, _), (_, _, metrics) in runs.items()
        if mode == "full"
    ]
    workmap_health_pass = bool(full_run_metrics) and all(
        isinstance(metrics.get("workmap_health"), dict)
        and metrics["workmap_health"].get("pass") is True
        for metrics in full_run_metrics
    )
    first_landmark_use_pass = bool(full_evidences) and all(
        evidence["first_new_landmark_pnp_use"]["pass"]
        for evidence in full_evidences
    )
    ba_commit_use_pass = bool(full_evidences) and all(
        evidence["ba_commit_and_use_during_replay"]["pass"]
        for evidence in full_evidences
    )
    durable_index_pass = bool(full_evidences) and all(
        evidence["keyframe_durable_to_index_ready"]["pass"]
        for evidence in full_evidences
    )
    return {
        "speed": speed_pass,
        "latency": latency_pass,
        "has_new_region_segments": has_new_region_segments,
        "visual": visual_pass,
        "revisit_contract": _has_revisit_contract(protocol),
        "ground_truth_artifact_sha_bound": ground_truth_artifact_sha_bound,
        "ground_truth_format_validated": ground_truth_format_validated,
        "ground_truth_available": ground_truth_artifact_sha_bound and ground_truth_format_validated,
        "gt_metrics_available": False,
        "feature_evidence_available": feature_evidence_available,
        "workmap_health": workmap_health_pass,
        "first_landmark_use": first_landmark_use_pass,
        "ba_commit_use": ba_commit_use_pass,
        "durable_index": durable_index_pass,
        "workmap_event_evidence": first_landmark_use_pass and ba_commit_use_pass and durable_index_pass,
        "resource_available": all(
            _resource_sampling_pass(record.get("resource_peaks"))
            for record, _, _ in runs.values()
        ),
    }


def _acceptance_blockers(gates: dict[str, bool]) -> list[str]:
    checks = (
        (gates["speed"], "one or more full runs or complete 10-second windows did not strictly exceed 15 FPS"),
        (gates["latency"], "capture-to-consumer or foreground latency gate failed or lacked measurements"),
        (gates["has_new_region_segments"], "no designated new-region/revisit segment is present"),
        (
            not gates["has_new_region_segments"] or gates["visual"],
            "full-mode fresh visual pose FPS did not strictly exceed 15 in every designated new-region segment",
        ),
        (gates["revisit_contract"], "no sequence establishes old map → new region → occlusion/loss → new-region revisit → old-map return"),
        (gates["ground_truth_artifact_sha_bound"], "independent ground truth is missing"),
        (
            not gates["ground_truth_artifact_sha_bound"] or gates["ground_truth_format_validated"],
            "ground-truth artifact is SHA-bound, but its format is not validated by P0",
        ),
        (gates["gt_metrics_available"], "independent ground-truth accuracy metrics are not computed by this P0 report"),
        (gates["feature_evidence_available"], "required online workmap activity counters are unavailable"),
        (
            gates["workmap_health"],
            "full-mode writer/index/BA/overlap health was missing or unhealthy, an overflow/error occurred, "
            "or a 10-second window showed continuously growing backlog age",
        ),
        (
            gates["first_landmark_use"],
            "events.jsonl does not prove a newly created workmap landmark was used by PnP on the next ordinal",
        ),
        (
            gates["ba_commit_use"],
            "events.jsonl and timed frames do not prove BA geometry was committed and used during replay",
        ),
        (
            gates["durable_index"],
            "events.jsonl does not prove a durable keyframe became index-ready within 1 second",
        ),
        (gates["resource_available"], "CPU/GPU/RAM/VRAM peak samples missing or failed"),
    )
    return [message for passed, message in checks if not passed]


def report_protocol(protocol_path: str | Path) -> Path:
    protocol, snapshot = _report_preflight(protocol_path)
    runs, process_instances, pids = _collect_runs(protocol)
    _check_run_consistency(protocol, runs)

    sequence_reports: dict[str, Any] = {}
    all_run_metrics: list[dict[str, Any]] = []
    full_visual_gates: list[bool] = []
    activity_by_sequence: dict[str, Any] = {}
    for sequence in protocol.sequences:
        name = sequence["raw"]["name"]
        mode_reports = _mode_reports(name, runs, all_run_metrics, full_visual_gates)
        activity_by_sequence[name] = _full_mode_activity(protocol, sequence, runs)
        sequence_reports[name] = _sequence_report(protocol, sequence, runs, mode_reports)
    gates = _gate_results(protocol, runs, all_run_metrics, full_visual_gates, activity_by_sequence)
    full_acceptance_blockers = _acceptance_blockers(gates)
    report_data = {
        "schema": REPORT_SCHEMA,
        "experiment_run_id": protocol.raw["run_id"],
        "protocol": str(protocol.path),
        "protocol_snapshot": str(snapshot),
        "repetitions_required": REPETITIONS,
        "modes_required": list(MODES),
        "hardware": protocol.raw["hardware"],
        "threads": protocol.raw["threads"],
        "gates": protocol.raw["gates"],
        "integrity": {
            "all_required_runs_complete": True,
            "expected_worker_process_instances": len(runs),
            "distinct_worker_process_instances": len(process_instances),
            "worker_pids": pids,
            "inputs_sha_bound_before_and_after_each_run": True,
            "all_modes_use_full_decoded_frame_sequence": True,
        },
        "gate_results": {
            "e2e_fps_strictly_above_15_all_runs_and_each_complete_10s_window": gates["speed"],
            "full_mode_fresh_visual_pose_fps_strictly_above_15_in_new_region_segments": gates["visual"],
            "capture_to_consumer_and_foreground_latency": gates["latency"],
            "new_region_revisit_sequence_contract_present": gates["revisit_contract"],
            "independent_ground_truth_artifact_sha_bound": gates["ground_truth_artifact_sha_bound"],
            "independent_ground_truth_format_validated": gates["ground_truth_format_validated"],
            "independent_ground_truth_available": gates["ground_truth_available"],
            "independent_ground_truth_accuracy_metrics_computed": gates["gt_metrics_available"],
            "workmap_activity_counters_available": gates["feature_evidence_available"],
            "full_mode_writer_index_ba_overlap_health_and_bounded_backlog": gates["workmap_health"],
            "first_new_landmark_used_by_next_pnp_ordinal": gates["first_landmark_use"],
            "ba_commit_and_geometry_use_during_replay": gates["ba_commit_use"],
            "keyframe_durable_to_index_ready_within_one_second": gates["durable_index"],
            "workmap_session_event_evidence_available": gates["workmap_event_evidence"],
            "cpu_gpu_ram_vram_peak_samples_available": gates["resource_available"],
        },
        "workmap_activity": activity_by_sequence,
        "sequences": sequence_reports,
        "decision": {
            "speed_only": "PASS" if gates["speed"] else "FAIL_OR_UNVERIFIED",
            "latency": "PASS" if gates["latency"] else "FAIL_OR_UNVERIFIED",
            "visual_freshness": "PASS" if gates["visual"] else ("NOT_VERIFIABLE" if not gates["has_new_region_segments"] else "FAIL_OR_UNVERIFIED"),
            "absolute_accuracy": "NOT_ASSESSED" if not gates["gt_metrics_available"] else "ASSESSED",
            "full_acceptance": "NOT_PASSED" if full_acceptance_blockers else "PASS",
            "full_acceptance_pass": not full_acceptance_blockers,
            "blockers": full_acceptance_blockers,
            "flight_permission": "NOT_GRANTED_OR_ASSESSED_OFFLINE_ONLY",
        },
    }
    report_path = protocol.output_root / "report.json"
    _reject_symlink_path(report_path, allow_missing=True, what="report output")
    _atomic_json(report_path, report_data)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one full sequence matrix for one evaluation stage")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--mode", required=True, choices=MODES)
    report_parser = subparsers.add_parser("report", help="validate and report all four complete stage matrices")
    report_parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            paths = run_mode(args.protocol, args.mode)
            for path in paths:
                print(path)
        else:
            print(report_protocol(args.protocol))
    except (EvaluationError, OSError, ValueError) as exc:
        parser.exit(2, f"online_workmap_eval: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

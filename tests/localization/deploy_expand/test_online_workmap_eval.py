from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
VALIDATION_DIR = ROOT / "定位演算法" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import online_workmap_eval as evaluator  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_fixture(tmp_path: Path, *, frame_rate: float = 16.0) -> Path:
    model_dir = tmp_path / "assets" / "model"
    model_dir.mkdir(parents=True)
    model = model_dir / "model.bin"
    model.write_bytes(b"synthetic model bytes")

    keyframe_root = tmp_path / "assets" / "keyframes"
    image = keyframe_root / "images" / "ref" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic reference image")
    keyframe_manifest = keyframe_root / "keyframes.jsonl"
    keyframe_manifest.write_text(
        json.dumps(
            {
                "output_name": "ref/frame.jpg",
                "image_sha256": _sha(image),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = tmp_path / "assets" / "bundle.json"
    bundle_files = [
        {"path": "model/model.bin", "sha256": _sha(model), "size_bytes": model.stat().st_size},
        {
            "path": "keyframes/keyframes.jsonl",
            "sha256": _sha(keyframe_manifest),
            "size_bytes": keyframe_manifest.stat().st_size,
        },
    ]
    bundle.write_text(
        json.dumps(
            {
                "model_dir": "model",
                "model_sha256": {"model.bin": _sha(model)},
                "keyframes_manifest": "keyframes/keyframes.jsonl",
                "keyframes_images_root": "keyframes/images",
                "files": bundle_files,
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "assets" / "profile.json"
    profile.write_text(json.dumps({"online_workmap": {"enabled": True}}), encoding="utf-8")
    video = tmp_path / "sequence.mp4"
    video.write_bytes(b"protocol fixture video placeholder")

    protocol = {
        "schema": evaluator.SCHEMA,
        "run_id": "synthetic-report",
        "repetitions": evaluator.REPETITIONS,
        "output_root": "results",
        "bundle": {"path": "assets/bundle.json", "sha256": _sha(bundle)},
        "profile": {"path": "assets/profile.json", "sha256": _sha(profile)},
        "query_camera": {
            "model": "PINHOLE",
            "width": 1280,
            "height": 720,
            "params": [1000.0, 1000.0, 640.0, 360.0],
        },
        "sequences": [
            {
                "name": "new-area",
                "video": "sequence.mp4",
                "video_sha256": _sha(video),
                "fps": frame_rate,
                "segments": [
                    {"name": "new-area", "start_s": 0.0, "end_s": 10.0, "role": "new_region"}
                ],
                "ground_truth": None,
            }
        ],
        "hardware": {"machine": "synthetic-fixture"},
        "threads": {name: 1 for name in evaluator._THREAD_ENV_NAMES},
        "gates": dict(evaluator._FIXED_GATES),
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def _write_synthetic_events(session_dir: Path, session_id: str, *, ba_enabled: bool) -> None:
    start_wall_s = 1_800_000_000.0
    keyframe_queued_wall_s = start_wall_s + 20 / 16.0
    events = [
        {
            "event": "keyframe_queued",
            "name": "kf-20",
            "revision": 2,
            "ordinal": 20,
            "index_requested": True,
            "stamp_unix_s": keyframe_queued_wall_s,
        },
        {
            "event": "keyframe_durable",
            "name": "kf-20",
            "revision": 2,
            "ordinal": 20,
            "image_path": str(session_dir / "images" / "kf-20.png"),
            "stamp_unix_s": keyframe_queued_wall_s + 0.1,
        },
        {
            "event": "reference_index_requested",
            "name": "kf-20",
            "revision": 2,
            "ordinal": 20,
            "stamp_unix_s": keyframe_queued_wall_s + 0.1,
        },
        {
            "event": "reference_indexed",
            "name": "kf-20",
            "provider_reference_name": "session-kf-20",
            "revision": 2,
            "pose_revision": 0,
            "stamp_unix_s": keyframe_queued_wall_s + 0.3,
        },
        {
            "event": "first_new_landmark_pnp_use",
            "landmark_id": -10,
            "created_ordinal": 10,
            "used_ordinal": 11,
            "stamp": 5_000.6875,
            "revision": 1,
        },
    ]
    if ba_enabled:
        events.extend(
            [
                {
                    "event": "ba_geometry_committed",
                    "revision": 3,
                    "source_epoch": 0,
                    "updated_landmarks": {"-20": [1.0, 2.0, 3.0]},
                    "submit_to_apply_ms": 250.0,
                    "stamp_unix_s": start_wall_s + 50 / 16.0 + 0.005,
                },
                {
                    "event": "ba_geometry_used",
                    "landmark_ids": [-20],
                    "used_ordinal": 51,
                    "stamp": 5_003.1875,
                    "revision": 3,
                },
            ]
        )
    path = session_dir / "events.jsonl"
    path.write_text(
        "".join(
            json.dumps({"session_id": session_id, **event}, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )


def _write_synthetic_matrix(protocol: evaluator.Protocol) -> None:
    sequence = protocol.sequences[0]
    name = sequence["raw"]["name"]
    fps = float(sequence["raw"]["fps"])
    frame_count = int(fps * 10)
    start_ns = 5_000_000_000_000
    snapshots = evaluator._snapshot_inputs(protocol, sequence)
    for mode in evaluator.MODES:
        for repeat in range(1, evaluator.REPETITIONS + 1):
            run_dir = protocol.output_root / mode / name / f"repeat-{repeat}"
            session_dir = run_dir / "session"
            session_dir.mkdir(parents=True)
            frames_path = run_dir / "frames.jsonl"
            rows = []
            for ordinal in range(frame_count):
                video_time = ordinal / fps
                capture_ns = start_ns + round(video_time * 1_000_000_000)
                response = {
                    "seq": ordinal,
                    "frame_id": f"worker-{ordinal}",
                    "capture_mono_ns": capture_ns,
                    "pose_capture_mono_ns": capture_ns,
                    "validity": True,
                    "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "direct_status": "FAST_TRACK",
                    "core_wall_ms": 10.0,
                }
                if mode != "baseline":
                    ba_enabled = mode in {"ba", "full"}
                    ba_applied = ba_enabled and ordinal >= 50
                    response.update(
                        {
                            "workmap_session_id": f"{mode}-{repeat}-session",
                            "workmap_session_dir": str(session_dir),
                            "workmap_revision": (
                                int(ordinal >= 10) + int(ordinal >= 20) + int(ba_applied)
                            ),
                            "workmap_points_created": int(ordinal >= 10),
                            "workmap_points_used": int(ordinal >= 11),
                            "workmap_keyframes_ready": int(ordinal >= 26),
                            "workmap_reloc_accepted": 0,
                            "workmap_ba_submitted": int(ba_enabled and ordinal >= 40),
                            "workmap_ba_applied": int(ba_applied),
                            "workmap_ba_stale": 0,
                            "workmap_scale_applied": 0,
                            "workmap_writer_healthy": True,
                            "workmap_writer_pending": False,
                            "workmap_writer_oldest_age_ms": 0.0,
                            "workmap_writer_overflow_count": 0,
                            "workmap_index_healthy": True,
                            "workmap_index_errors": 0,
                            "workmap_index_pending": False,
                            "workmap_index_oldest_age_ms": 0.0,
                            "workmap_reloc_pending": False,
                            "workmap_reloc_oldest_age_ms": 0.0,
                            "workmap_ba_pending": False,
                            "workmap_ba_oldest_age_ms": 0.0,
                            "workmap_ba_healthy": True,
                            "workmap_ba_rejected": 0,
                            "workmap_ba_applied_this_frame": bool(ba_enabled and ordinal == 50),
                            "workmap_scale_applied_this_frame": False,
                            "workmap_overlap_evidence_count": 0,
                            "workmap_overlap_stale": 0,
                            "workmap_overlap_rejected": 0,
                            "workmap_overlap_superseded": 0,
                            "workmap_overlap_healthy": True,
                            "workmap_overlap_pending": False,
                            "workmap_overlap_oldest_age_ms": 0.0,
                        }
                    )
                rows.append(
                    {
                        "experiment_run_id": protocol.raw["run_id"],
                        "run_id": f"{mode}-{repeat}",
                        "mode": mode,
                        "sequence": name,
                        "repeat": repeat,
                        "ordinal": ordinal,
                        "video_time_s": video_time,
                        "segments": [{"name": "new-area", "role": "new_region"}],
                        "decode_ms": 0.1,
                        "pace_wait_ms": 1.0,
                        "send_mono_ns": capture_ns + 1_000_000,
                        "received_mono_ns": capture_ns + 10_000_000,
                        "response": response,
                    }
                )
            frames_path.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            if mode != "baseline":
                _write_synthetic_events(
                    session_dir,
                    f"{mode}-{repeat}-session",
                    ba_enabled=mode in {"ba", "full"},
                )
            command = evaluator._worker_command(protocol, session_dir, mode, (2688, 1512))
            run = {
                "schema": evaluator.RUN_SCHEMA,
                "experiment_run_id": protocol.raw["run_id"],
                "run_id": f"{mode}-{repeat}",
                "mode": mode,
                "sequence": name,
                "repeat": repeat,
                "worker_process_instance": uuid.uuid4().hex,
                "process_id": 10000 + evaluator.MODES.index(mode) * evaluator.REPETITIONS + repeat,
                "worker_returncode": 0,
                "complete": True,
                "decoder_read_terminated": True,
                "decoded_until_eof": True,
                "worker_exit_observed": True,
                "worker_close_and_session_drain_observed": True,
                "worker_command": command,
                "python": sys.executable,
                "worker_path": str(evaluator.WORKER_PATH),
                "worker_startup": {"event": "ready", "backend": "direct"},
                "thread_settings": dict(protocol.raw["threads"]),
                "evaluation_only_mode": None if mode == "full" else "1",
                "frames_jsonl_sha256": _sha(frames_path),
                "input_snapshot_before": snapshots,
                "input_snapshot_after": copy.deepcopy(snapshots),
                "input_integrity_pass": True,
                "frame_count_integrity_pass": True,
                "decoded_frames": frame_count,
                "declared_frame_count": frame_count,
                "nominal_source_fps": fps,
                "run_wall_s": 10.1,
                "run_start_monotonic_ns": start_ns,
                "run_start_unix_s": 1_800_000_000.0,
                "source_size": [2688, 1512],
                "query_camera": dict(protocol.raw["query_camera"]),
                "effective_query_camera": evaluator._effective_query_camera(
                    protocol.raw["query_camera"], (2688, 1512)
                ),
                "resource_peaks": {
                    "sampling_status": "COMPLETE",
                    "source": "linux /proc/<pid>/stat+status; nvidia-smi --id=0 --query-gpu=utilization.gpu,memory.used",
                    "cpu_scope": "worker process tree including child processes",
                    "gpu_scope": "device-level total for GPU 0; includes other GPU users",
                    "sample_interval_s": 1.0,
                    "samples": 11,
                    "gpu_samples": 11,
                    "peak_cpu_percent_one_core_100": 25.0,
                    "peak_ram_mib": 512.0,
                    "peak_process_tree_rss_hwm_mib": 768.0,
                    "peak_gpu_utilization_percent_device": 75.0,
                    "peak_vram_mib_device": 4096.0,
                    "errors": [],
                },
            }
            (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")


def _rewrite_frame_response(
    run_dir: Path,
    ordinal: int,
    *,
    updates: dict[str, object] | None = None,
    remove: tuple[str, ...] = (),
) -> None:
    frames_path = run_dir / "frames.jsonl"
    rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
    response = rows[ordinal]["response"]
    for field in remove:
        response.pop(field, None)
    if updates:
        response.update(updates)
    frames_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["frames_jsonl_sha256"] = _sha(frames_path)
    run_path.write_text(json.dumps(run), encoding="utf-8")


def test_synthetic_matrix_keeps_workmap_stages_separate(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)

    baseline = protocol.output_root / "baseline" / "new-area" / "repeat-1" / "frames.jsonl"
    reuse = protocol.output_root / "reuse" / "new-area" / "repeat-1" / "frames.jsonl"
    ba = protocol.output_root / "ba" / "new-area" / "repeat-1" / "frames.jsonl"
    full = protocol.output_root / "full" / "new-area" / "repeat-1" / "frames.jsonl"
    baseline_row = json.loads(baseline.read_text(encoding="utf-8").splitlines()[0])
    reuse_row = json.loads(reuse.read_text(encoding="utf-8").splitlines()[0])
    ba_rows = [json.loads(line) for line in ba.read_text(encoding="utf-8").splitlines()]
    full_rows = [json.loads(line) for line in full.read_text(encoding="utf-8").splitlines()]

    assert not any(key.startswith("workmap_") for key in baseline_row["response"])
    assert reuse_row["response"]["workmap_ba_submitted"] == 0
    assert reuse_row["response"]["workmap_ba_applied"] == 0
    assert ba_rows[50]["response"]["workmap_ba_applied_this_frame"] is True
    assert ba_rows[0]["response"]["workmap_scale_applied"] == 0
    assert full_rows[50]["response"]["workmap_ba_applied_this_frame"] is True
    assert full_rows[0]["response"]["workmap_overlap_healthy"] is True


def test_protocol_fixes_three_repetitions_and_hash_bound_assets(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    assert protocol.raw["repetitions"] == 3
    assert protocol.raw["bundle"]["path"] == str((tmp_path / "assets" / "bundle.json").resolve())
    assert protocol.raw["sequences"][0]["video"] == str((tmp_path / "sequence.mp4").resolve())

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["repetitions"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(evaluator.EvaluationError, match="exactly 3"):
        evaluator.load_protocol(path)


def test_input_snapshot_rejects_modified_keyframe_reference_image(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    image = tmp_path / "assets" / "keyframes" / "images" / "ref" / "frame.jpg"
    image.write_bytes(b"changed reference image")

    with pytest.raises(evaluator.EvaluationError, match="SHA-256 differs from manifest"):
        evaluator._snapshot_inputs(protocol, protocol.sequences[0])


def test_full_stage_rejects_disabled_profile_even_when_hash_bound(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    profile_path = tmp_path / "assets" / "profile.json"
    profile_path.write_text(json.dumps({"online_workmap": {"enabled": False}}), encoding="utf-8")
    raw["profile"]["sha256"] = _sha(profile_path)
    path.write_text(json.dumps(raw), encoding="utf-8")
    protocol = evaluator.load_protocol(path)
    snapshot = evaluator._snapshot_inputs(protocol, protocol.sequences[0])
    assert snapshot["profile"]["sha256"] == raw["profile"]["sha256"]

    with pytest.raises(evaluator.EvaluationError, match="online_workmap.enabled=true"):
        evaluator._verify_full_profile(protocol, "full")
    evaluator._verify_full_profile(protocol, "baseline")


def test_protocol_rejects_output_inside_frozen_release(tmp_path: Path) -> None:
    release = tmp_path / "site" / "releases" / "release-1"
    (release / "assets").mkdir(parents=True)
    bundle = release / "assets" / "bundle.json"
    profile = release / "assets" / "profile.json"
    video = tmp_path / "sequence.mp4"
    model = release / "assets" / "model.bin"
    model.write_bytes(b"model")
    bundle.write_text(
        json.dumps({"model_dir": ".", "model_sha256": {"model.bin": _sha(model)}}), encoding="utf-8"
    )
    profile.write_text(json.dumps({"online_workmap": {"enabled": True}}), encoding="utf-8")
    video.write_bytes(b"video")
    raw = {
        "schema": evaluator.SCHEMA,
        "run_id": "release-safety",
        "repetitions": 3,
        "output_root": str(release / "results"),
        "bundle": {"path": str(bundle), "sha256": _sha(bundle)},
        "profile": {"path": str(profile), "sha256": _sha(profile)},
        "query_camera": {"model": "PINHOLE", "width": 2, "height": 2, "params": [1, 1, 1, 1]},
        "sequences": [
            {
                "name": "seq",
                "video": str(video),
                "video_sha256": _sha(video),
                "fps": 16,
                "ground_truth": None,
            }
        ],
        "hardware": {"machine": "fixture"},
        "threads": {name: 1 for name in evaluator._THREAD_ENV_NAMES},
        "gates": dict(evaluator._FIXED_GATES),
    }
    path = tmp_path / "release-protocol.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(evaluator.EvaluationError, match="outside frozen release"):
        evaluator.load_protocol(path)


def test_report_requires_every_stage_and_repeat(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    with pytest.raises(evaluator.EvaluationError, match="run directory"):
        evaluator.report_protocol(path)


def test_report_uses_strict_fps_gate_and_calls_missing_gt_unassessed(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)

    report_path = evaluator.report_protocol(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["integrity"]["all_required_runs_complete"] is True
    assert (
        report["gate_results"]["e2e_fps_strictly_above_15_all_runs_and_each_complete_10s_window"]
        is True
    )
    assert report["gate_results"]["cpu_gpu_ram_vram_peak_samples_available"] is True
    assert report["gate_results"]["independent_ground_truth_available"] is False
    assert (
        report["gate_results"][
            "full_mode_fresh_visual_pose_fps_strictly_above_15_in_new_region_segments"
        ]
        is True
    )
    assert (
        report["gate_results"]["full_mode_writer_index_ba_overlap_health_and_bounded_backlog"]
        is True
    )
    assert report["gate_results"]["workmap_session_event_evidence_available"] is True
    assert report["gate_results"]["new_region_revisit_sequence_contract_present"] is False
    assert any(
        "no sequence establishes old map" in blocker for blocker in report["decision"]["blockers"]
    )
    activity = report["workmap_activity"]["new-area"][0]
    assert activity["first_new_landmark_pnp_use"]["pass"] is True
    assert activity["ba_commit_and_use_during_replay"]["pass"] is True
    assert activity["keyframe_durable_to_index_ready"]["pass"] is True
    assert activity["keyframe_durable_to_index_ready"][
        "max_durable_to_index_ready_ms"
    ] == pytest.approx(200.0)
    assert report["decision"]["full_acceptance"] == "NOT_PASSED"
    assert "independent ground truth is missing" in report["decision"]["blockers"]
    sequence_report = report["sequences"]["new-area"]
    assert sequence_report["ground_truth"]["accuracy_status"] == "MISSING_INDEPENDENT_GROUND_TRUTH"
    assert sequence_report["query_camera_base"] == protocol.raw["query_camera"]
    assert sequence_report["effective_query_camera"] == evaluator._effective_query_camera(
        protocol.raw["query_camera"], (2688, 1512)
    )
    assert sequence_report["modes"]["full"]["repetitions"][0]["e2e_fps"] > 15.0
    health = sequence_report["modes"]["full"]["repetitions"][0]["workmap_health"]
    queue_ages = health["ten_second_pending_age_windows"][0]["queues"]
    assert set(queue_ages) == {"reloc", "index", "ba", "writer", "overlap"}
    assert all(item["oldest_pending_age_ms"] == 0.0 for item in queue_ages.values())


@pytest.mark.parametrize("missing", ("ready", "eligibility"))
def test_report_rejects_promised_references_without_complete_index_evidence(
    tmp_path: Path, missing: str
) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)
    events_path = (
        protocol.output_root / "full" / "new-area" / "repeat-1" / "session" / "events.jsonl"
    )
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    if missing == "ready":
        stamp = 1_800_000_000.0 + 21 / 16.0
        events.extend(
            [
                {
                    "session_id": events[0]["session_id"],
                    "event": "keyframe_queued",
                    "name": "kf-21",
                    "revision": 4,
                    "ordinal": 21,
                    "index_requested": True,
                    "stamp_unix_s": stamp,
                },
                {
                    "session_id": events[0]["session_id"],
                    "event": "keyframe_durable",
                    "name": "kf-21",
                    "revision": 4,
                    "ordinal": 21,
                    "stamp_unix_s": stamp + 0.1,
                },
                {
                    "session_id": events[0]["session_id"],
                    "event": "reference_index_requested",
                    "name": "kf-21",
                    "revision": 4,
                    "ordinal": 21,
                    "stamp_unix_s": stamp + 0.1,
                },
            ]
        )
    else:
        next(event for event in events if event["event"] == "keyframe_queued").pop(
            "index_requested"
        )
    events_path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )

    report = json.loads(evaluator.report_protocol(path).read_text(encoding="utf-8"))
    evidence = report["workmap_activity"]["new-area"][0]["keyframe_durable_to_index_ready"]
    assert evidence["pass"] is False
    assert evidence["indexed_keyframe_count"] == 1
    assert evidence["missing_ready_keyframe_count"] == (1 if missing == "ready" else 0)
    assert evidence["eligibility_events_complete"] is (missing == "ready")
    assert report["gate_results"]["workmap_session_event_evidence_available"] is False


def test_report_fails_on_a_transient_unhealthy_full_frame(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)
    run_dir = protocol.output_root / "full" / "new-area" / "repeat-1"
    _rewrite_frame_response(
        run_dir,
        80,
        updates={"workmap_writer_healthy": False},
    )

    report_path = evaluator.report_protocol(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert (
        report["gate_results"]["full_mode_writer_index_ba_overlap_health_and_bounded_backlog"]
        is False
    )
    health = report["sequences"]["new-area"]["modes"]["full"]["repetitions"][0]["workmap_health"]
    assert health["frame_health_pass"] is False
    assert health["last_frame_health_pass"] is True
    assert health["unhealthy_frame_count_by_field"]["workmap_writer_healthy"] == 1
    assert report["decision"]["full_acceptance"] == "NOT_PASSED"


def _same_frame_ba_evidence(
    tmp_path: Path,
    *,
    used_ordinal: int = 1742,
    commit_stamp: float = 1_800_000_001.0,
    use_revision: int = 1752,
    updated_landmarks: dict[str, list[float]] | None = None,
    used_landmark_ids: list[int] | None = None,
    output_source: str | None = None,
) -> dict[str, object]:
    session_id = "same-frame-session"
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    events = [
        {
            "session_id": session_id,
            "event": "ba_geometry_committed",
            "revision": 1752,
            "source_epoch": 0,
            "updated_landmarks": updated_landmarks or {"-20": [1.0, 2.0, 3.0]},
            "submit_to_apply_ms": 250.0,
            "stamp_unix_s": commit_stamp,
        },
        {
            "session_id": session_id,
            "event": "ba_geometry_used",
            "landmark_ids": used_landmark_ids or [-20],
            "used_ordinal": used_ordinal,
            "stamp": 5_003.1875,
            "revision": use_revision,
        },
    ]
    (session_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )
    start_mono_ns = 10_000_000_000
    rows = []
    for ordinal, offset_ns in ((1741, 1_900_000_000), (1742, 2_000_000_000), (1743, 2_100_000_000)):
        capture_ns = start_mono_ns + offset_ns
        rows.append(
            {
                "ordinal": ordinal,
                "received_mono_ns": capture_ns,
                "response": {
                    "workmap_session_id": session_id,
                    "workmap_revision": 1752,
                    "workmap_ba_applied_this_frame": ordinal == 1742,
                    "workmap_points_used": 1,
                    "validity": True,
                    "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "capture_mono_ns": capture_ns,
                    "pose_capture_mono_ns": capture_ns,
                    "direct_status": "FAST_TRACK",
                    **({"workmap_output_source": output_source} if output_source else {}),
                },
            }
        )
    return evaluator._workmap_evidence(
        rows,
        {"raw": {"name": "same-frame", "segments": []}},
        session_dir,
        None,
        {
            "start_unix_s": 1_800_000_000.0,
            "end_unix_s": 1_800_000_003.0,
            "start_monotonic_ns": start_mono_ns,
        },
    )


def test_ba_geometry_commit_and_use_on_same_fresh_pnp_frame_passes(tmp_path: Path) -> None:
    evidence = _same_frame_ba_evidence(tmp_path)

    ba = evidence["ba_commit_and_use_during_replay"]
    assert ba["pass"] is True
    assert ba["examples"] == [
        {
            "revision": 1752,
            "applied_ordinal": 1742,
            "frame_workmap_revision": 1752,
            "used_ordinal": 1742,
            "submit_to_apply_ms": 250.0,
        }
    ]


@pytest.mark.parametrize("source, accepted", [("baseline", False), ("workmap", True)])
def test_auxiliary_ba_use_requires_published_workmap_pose(tmp_path, source, accepted):
    evidence = _same_frame_ba_evidence(tmp_path, output_source=source)
    assert evidence["ba_commit_and_use_during_replay"]["pass"] is accepted


@pytest.mark.parametrize("source, accepted", [("baseline", False), ("workmap", True)])
def test_first_landmark_use_requires_published_workmap_pose(source, accepted):
    event = {
        "event": "first_new_landmark_pnp_use",
        "landmark_id": -1,
        "created_ordinal": 10,
        "used_ordinal": 11,
    }
    rows = {
        10: {"response": {}},
        11: {
            "response": {
                "workmap_output_source": source,
                "workmap_points_used": 1,
                "validity": True,
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                "capture_mono_ns": 100,
                "pose_capture_mono_ns": 100,
                "direct_status": "VO_ONLY",
            }
        },
    }
    assert bool(evaluator._first_use_events([event], rows)) is accepted


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"used_ordinal": 1741}, "use preceded apply"),
        ({"use_revision": 1751}, "stale revision"),
        ({"updated_landmarks": {"20": [1.0, 2.0, 3.0]}}, "no updated negative ID"),
        ({"used_landmark_ids": [-21]}, "used point was not updated"),
        ({"commit_stamp": 1_800_000_002.5}, "commit followed applied frame"),
        ({"commit_stamp": 1_800_000_004.0}, "commit occurred after replay close"),
    ],
)
def test_ba_evidence_rejects_preceding_stale_and_close_events(
    tmp_path: Path,
    changes: dict[str, object],
    reason: str,
) -> None:
    evidence = _same_frame_ba_evidence(tmp_path, **changes)

    assert evidence["ba_commit_and_use_during_replay"]["pass"] is False, reason


def test_resource_sampler_counts_zombie_rss_race_as_expected_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = 23456
    proc_root = tmp_path / "proc"
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "stat").write_text(f"{pid} (worker) Z 1\n", encoding="utf-8")
    real_path = Path
    monkeypatch.setattr(
        evaluator,
        "Path",
        lambda value: proc_root if value == "/proc" else real_path(value),
    )

    sampler = evaluator._ResourceSampler(pid)
    assert sampler._process_tree() is None
    assert any("zombie" in item for item in sampler.expected_exit_diagnostics)
    sampler.samples = 128
    sampler.gpu_samples = 129
    sampler.peaks.update(
        {
            "cpu_percent_one_core_100": 264.0,
            "ram_mib": 2935.0,
            "rss_hwm_mib": 2935.0,
            "gpu_utilization_percent_device": 84.0,
            "vram_mib_device": 1528.0,
        }
    )
    sampler.thread = threading.Thread(target=lambda: None)
    sampler.thread.start()
    sampler.thread.join()

    resource = sampler.finish()
    assert resource["sampling_status"] == "COMPLETE"
    assert resource["errors"] == []
    assert resource["expected_exit_diagnostics"]


def test_live_process_rss_sampling_error_is_not_expected_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = 23457
    proc_root = tmp_path / "proc"
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "stat").write_text(
        f"{pid} (worker) R " + " ".join(["0"] * 16) + "\n",
        encoding="utf-8",
    )
    (process_dir / "status").write_text("Name:\tworker\n", encoding="utf-8")
    real_path = Path
    monkeypatch.setattr(
        evaluator,
        "Path",
        lambda value: proc_root if value == "/proc" else real_path(value),
    )

    sampler = evaluator._ResourceSampler(pid)
    with pytest.raises(ValueError, match="missing RSS fields"):
        sampler._process_tree()
    assert not sampler.expected_exit_diagnostics


def test_report_does_not_count_close_only_ba_or_index_events(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)
    events_path = (
        protocol.output_root / "full" / "new-area" / "repeat-1" / "session" / "events.jsonl"
    )
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event["event"] in {"ba_geometry_committed", "reference_indexed"}:
            event["stamp_unix_s"] = 1_800_000_020.0
    events_path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )

    report_path = evaluator.report_protocol(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gate_results"]["workmap_activity_counters_available"] is True
    assert report["gate_results"]["workmap_session_event_evidence_available"] is False
    evidence = report["workmap_activity"]["new-area"][0]
    assert evidence["session_events"]["event_stream_pass"] is True
    assert evidence["ba_commit_and_use_during_replay"]["pass"] is False
    assert evidence["keyframe_durable_to_index_ready"]["pass"] is False
    assert evidence["stage_behavior_assessment"].startswith(
        "SESSION_EVENT_EVIDENCE_FAILED_OR_INCOMPLETE"
    )
    assert report["decision"]["full_acceptance"] == "NOT_PASSED"


def test_report_rejects_ba_commit_after_applied_frame(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)
    events_path = (
        protocol.output_root / "full" / "new-area" / "repeat-1" / "session" / "events.jsonl"
    )
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event["event"] == "ba_geometry_committed":
            event["stamp_unix_s"] = 1_800_000_004.0
    events_path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )

    report_path = evaluator.report_protocol(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = report["workmap_activity"]["new-area"][0]
    assert evidence["session_events"]["event_stream_pass"] is True
    assert evidence["ba_commit_and_use_during_replay"]["pass"] is False
    assert evidence["keyframe_durable_to_index_ready"]["pass"] is True
    assert report["gate_results"]["workmap_session_event_evidence_available"] is False
    assert report["decision"]["full_acceptance"] == "NOT_PASSED"


def test_report_fails_on_missing_final_health_and_session_events(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)
    run_dir = protocol.output_root / "full" / "new-area" / "repeat-1"
    _rewrite_frame_response(run_dir, 159, remove=("workmap_overlap_healthy",))
    (run_dir / "session" / "events.jsonl").unlink()

    report_path = evaluator.report_protocol(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert (
        report["gate_results"]["full_mode_writer_index_ba_overlap_health_and_bounded_backlog"]
        is False
    )
    assert report["gate_results"]["workmap_session_event_evidence_available"] is False
    assert report["gate_results"]["workmap_activity_counters_available"] is True
    repetition = report["sequences"]["new-area"]["modes"]["full"]["repetitions"][0]
    assert repetition["workmap_health"]["last_frame_health_pass"] is False
    assert repetition["workmap_health"]["last_frame_health_missing"] is True
    activity = report["workmap_activity"]["new-area"][0]
    assert activity["cumulative_counters_observed"]["workmap_ba_applied"] == 1
    assert activity["session_events"]["readable"] is False
    assert activity["event_evidence_pass"] is False
    assert report["decision"]["full_acceptance"] == "NOT_PASSED"


def test_report_fails_when_full_window_backlog_age_grows_continuously(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)
    run_dir = protocol.output_root / "full" / "new-area" / "repeat-1"
    frames_path = run_dir / "frames.jsonl"
    rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
    for ordinal, row in enumerate(rows):
        row["response"]["workmap_writer_pending"] = True
        row["response"]["workmap_writer_oldest_age_ms"] = ordinal * 62.5
    frames_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["frames_jsonl_sha256"] = _sha(frames_path)
    run_path.write_text(json.dumps(run), encoding="utf-8")

    report_path = evaluator.report_protocol(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert (
        report["gate_results"]["full_mode_writer_index_ba_overlap_health_and_bounded_backlog"]
        is False
    )
    health = report["sequences"]["new-area"]["modes"]["full"]["repetitions"][0]["workmap_health"]
    assert health["backlog_windows_pass"] is False
    assert health["backlog_growth_windows"][0]["queue"] == "writer"


def test_report_fails_resource_gate_when_sampler_data_is_missing(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)

    run_path = protocol.output_root / "baseline" / "new-area" / "repeat-1" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    del run["resource_peaks"]
    run_path.write_text(json.dumps(run), encoding="utf-8")

    report_path = evaluator.report_protocol(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gate_results"]["cpu_gpu_ram_vram_peak_samples_available"] is False
    assert (
        report["sequences"]["new-area"]["modes"]["baseline"]["repetitions"][0]["resource_peaks"]
        is None
    )
    assert "CPU/GPU/RAM/VRAM peak samples missing or failed" in report["decision"]["blockers"]
    assert report["decision"]["full_acceptance"] == "NOT_PASSED"


def test_gt_artifact_remains_unverified_without_a_declared_format(tmp_path: Path) -> None:
    path = _protocol_fixture(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    gt_path = tmp_path / "opaque-ground-truth.csv"
    gt_path.write_text("unparsed ground truth artifact", encoding="utf-8")
    raw["sequences"][0]["ground_truth"] = {
        "path": gt_path.name,
        "sha256": _sha(gt_path),
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    protocol = evaluator.load_protocol(path)
    evaluator._prepare_protocol_snapshot(protocol)
    _write_synthetic_matrix(protocol)

    report_path = evaluator.report_protocol(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ground_truth = report["sequences"]["new-area"]["ground_truth"]
    assert ground_truth["artifact_sha_bound"] is True
    assert ground_truth["format_validation"] == "NOT_VALIDATED_BY_P0_REPORT"
    assert ground_truth["absolute_accuracy_metrics"] is None
    assert ground_truth["accuracy_status"] == "NOT_COMPUTED_BY_P0_REPORT_FORMAT_UNVERIFIED"
    assert report["gate_results"]["independent_ground_truth_artifact_sha_bound"] is True
    assert report["gate_results"]["independent_ground_truth_format_validated"] is False
    assert report["gate_results"]["independent_ground_truth_available"] is False
    assert (
        "ground-truth artifact is SHA-bound, but its format is not validated by P0"
        in report["decision"]["blockers"]
    )


def test_exactly_fifteen_fps_fails_strict_run_and_window_gates() -> None:
    fps = 15.0
    start_ns = 5_000_000_000_000
    rows = []
    for ordinal in range(150):
        capture_ns = start_ns + round(ordinal / fps * 1_000_000_000)
        rows.append(
            {
                "video_time_s": ordinal / fps,
                "received_mono_ns": capture_ns + 1_000_000,
                "response": {"capture_mono_ns": capture_ns, "core_wall_ms": 10.0},
            }
        )
    metrics = evaluator._run_metrics(
        {"run_wall_s": 10.0, "run_start_monotonic_ns": start_ns},
        rows,
        {"raw": {"fps": fps, "segments": []}},
        dict(evaluator._FIXED_GATES),
    )
    assert metrics["e2e_fps"] == 15.0
    assert metrics["e2e_fps_pass"] is False
    assert metrics["ten_second_window_gate_pass"] is False


def test_existing_protocol_at_output_root_is_verified_without_rewriting(tmp_path: Path) -> None:
    original = _protocol_fixture(tmp_path)
    raw = json.loads(original.read_text(encoding="utf-8"))
    raw["output_root"] = "."
    raw["bundle"]["path"] = "../assets/bundle.json"
    raw["profile"]["path"] = "../assets/profile.json"
    raw["sequences"][0]["video"] = "../sequence.mp4"
    output_root = tmp_path / "results"
    output_root.mkdir()
    snapshot = output_root / "protocol.json"
    snapshot.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    before = snapshot.read_bytes()

    protocol = evaluator.load_protocol(snapshot)
    assert evaluator._prepare_protocol_snapshot(protocol) == snapshot
    assert snapshot.read_bytes() == before


_FAKE_WORKER = r"""
import json
import struct
import sys

args = sys.argv[1:]
width = int(args[args.index("--width") + 1])
height = int(args[args.index("--height") + 1])
wrong_seq_at = int(sys.argv[-1]) if sys.argv[-1].lstrip("-").isdigit() else -1
header = struct.Struct("!4sBd")
out = sys.stdout
out.write(json.dumps({"event": "ready", "backend": "direct"}) + "\n")
out.flush()
source = sys.stdin.buffer
seq = 0
while True:
    raw = source.read(header.size)
    if not raw:
        break
    _magic, _code, stamp = header.unpack(raw)
    frame = source.read(width * height * 3)
    assert len(frame) == width * height * 3
    out.write(json.dumps({
        "seq": seq + (1 if seq == wrong_seq_at else 0),
        "frame_id": f"worker-{seq}",
        "capture_mono_ns": int(round(stamp * 1_000_000_000.0)),
        "validity": False,
    }) + "\n")
    out.flush()
    seq += 1
"""


def _video_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, wrong_seq_at: int = -1
) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = _protocol_fixture(tmp_path)
    video = tmp_path / "sequence.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write an mp4v test video")
    for index in range(12):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sequences"][0]["video_sha256"] = _sha(video)
    raw["sequences"][0]["segments"] = [
        {"name": "new-area", "start_s": 0.0, "end_s": 0.5, "role": "new_region"}
    ]
    path.write_text(json.dumps(raw), encoding="utf-8")

    worker = tmp_path / "fake_worker.py"
    worker.write_text(_FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(evaluator, "WORKER_PATH", worker)
    original_command = evaluator._worker_command

    def command(*args, **kwargs):
        return [*original_command(*args, **kwargs), str(wrong_seq_at)]

    monkeypatch.setattr(evaluator, "_worker_command", command)
    return path


def test_run_one_streams_every_decoded_frame_through_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = evaluator.load_protocol(_video_protocol(tmp_path, monkeypatch))
    evaluator._prepare_protocol_snapshot(protocol)

    run_path = evaluator._run_one(protocol, "baseline", protocol.sequences[0], 1)

    run = json.loads(run_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in Path(run["frames_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert run["complete"] is True
    assert run["error"] is None
    assert run["worker_returncode"] == 0
    assert run["decoded_until_eof"] is True
    assert run["decoded_frames"] == len(rows) == run["declared_frame_count"]
    assert run["input_integrity_pass"] is True
    assert run["frame_count_integrity_pass"] is True
    assert run["evaluation_only_mode"] == "1"
    assert run["source_size"] == [64, 48]
    assert run["effective_query_camera"]["width"] == 64
    assert run["frames_jsonl_sha256"] == _sha(Path(run["frames_jsonl"]))
    assert run["resource_peaks"]["sampling_status"] in {"COMPLETE", "PARTIAL"}
    assert [row["ordinal"] for row in rows] == list(range(len(rows)))
    assert rows[0]["segments"] == [{"name": "new-area", "role": "new_region"}]
    assert rows[-1]["segments"] == []
    assert all(row["response"]["frame_id"] == f"worker-{row['ordinal']}" for row in rows)

    record, validated_rows = evaluator._validate_run(protocol, "baseline", protocol.sequences[0], 1)
    assert record["run_id"] == run["run_id"]
    assert validated_rows == rows

    tampered = dict(run, thread_settings={**run["thread_settings"], "OMP_NUM_THREADS": 99})
    run_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(evaluator.EvaluationError, match="thread settings differ"):
        evaluator._validate_run(protocol, "baseline", protocol.sequences[0], 1)


def test_run_one_records_failure_when_worker_breaks_frame_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = evaluator.load_protocol(_video_protocol(tmp_path, monkeypatch, wrong_seq_at=3))
    evaluator._prepare_protocol_snapshot(protocol)

    with pytest.raises(evaluator.EvaluationError, match="failed; see"):
        evaluator._run_one(protocol, "full", protocol.sequences[0], 1)

    run_path = protocol.output_root / "full" / "new-area" / "repeat-1" / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert run["complete"] is False
    assert "does not match decoded ordinal 3" in run["error"]
    assert run["evaluation_only_mode"] is None
    assert run["decoded_frames"] == 3
    assert run["decoded_until_eof"] is False
    assert run["input_integrity_pass"] is True
    assert run["worker_returncode"] is not None

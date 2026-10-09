from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import pytest

from tools.workmap_reconstruction import (
    _Keyframe,
    _Session,
    _add_session_geometry,
    _camera_calibrations,
    _fit_original_gauge,
    _parse_session,
    _point_image_components,
    _triangulate_measured_track,
    _retain_triangulated,
    _triangulate_session,
)


def _K(width: int = 640, height: int = 480, focal: float = 420.0) -> np.ndarray:
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pose(center: np.ndarray) -> np.ndarray:
    pose = np.column_stack((np.eye(3), -np.asarray(center, dtype=np.float64)))
    return pose


def _pixel(K: np.ndarray, pose: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    camera = pose[:, :3] @ xyz + pose[:, 3]
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def test_measured_pose_triangulation_recovers_real_point() -> None:
    K = _K()
    xyz = np.array([0.25, -0.15, 5.4], dtype=np.float64)
    centers = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.85, 0.0, 0.0]),
        np.array([0.05, 0.65, 0.0]),
    )
    observations = [(_pixel(K, _pose(center), xyz), K, _pose(center)) for center in centers]

    triangulated, metrics = _triangulate_measured_track(observations)

    np.testing.assert_allclose(triangulated, xyz, rtol=0.0, atol=1e-9)
    assert metrics["observation_count"] == 3
    assert metrics["parallax_deg"] > 1.5
    assert metrics["reprojection_p90_px"] < 1e-8


def test_low_parallax_track_is_excluded_without_losing_measured_good_track(tmp_path: Path) -> None:
    snapshot, bundle_sha = _write_snapshot(
        tmp_path / "session", observation_source="keyframe_image_measurement"
    )
    session = _parse_session(
        snapshot, tmp_path / "session", "base-release", "base-frame", bundle_sha
    )
    far = np.array([0.2, 0.1, 200.0], dtype=np.float64)
    extra = []
    frames = []
    for frame in session.keyframes:
        pixel = _pixel(frame.K, frame.pose, far)
        extra.append((frame.name, pixel))
        frames.append(
            replace(
                frame,
                negative_ids=np.append(frame.negative_ids, -9),
                negative_xy=np.vstack((frame.negative_xy, pixel)),
                landmark_revisions={**frame.landmark_revisions, -9: 0},
            )
        )
    session = replace(
        session,
        keyframes=tuple(frames),
        points_xyz={**session.points_xyz, -9: far},
        negative_observations={**session.negative_observations, -9: tuple(extra)},
    )
    points, _metrics, rejected = _triangulate_session(session)
    assert set(points) == {-7}
    assert rejected == {"minimum_parallax": 1}
    retained = _retain_triangulated(session, points)
    assert all(frame.negative_ids.tolist() == [-7] for frame in retained.keyframes)


def test_old_only_keyframe_component_needs_six_measured_base_anchors(tmp_path: Path) -> None:
    from tools.workmap_reconstruction import _validate_component_anchors

    snapshot, bundle_sha = _write_snapshot(
        tmp_path / "session", observation_source="keyframe_image_measurement"
    )
    parsed = _parse_session(
        snapshot, tmp_path / "session", "base-release", "base-frame", bundle_sha
    )
    frame = replace(
        parsed.keyframes[0],
        negative_ids=np.empty(0, dtype=np.int64),
        negative_xy=np.empty((0, 2), dtype=np.float64),
    )
    measured_old = tuple((point_id, np.array([float(point_id), 1.0])) for point_id in range(1, 7))
    component = {"keyframe_names": [frame.name]}
    old_only = replace(parsed, keyframes=(frame,), positive_observations={frame.name: measured_old})
    anchors = _validate_component_anchors(old_only, [component], [])
    assert anchors[0]["exact_old_track_point_ids"] == list(range(1, 7))
    assert anchors[0]["edm_anchored_negative_ids"] == []
    with pytest.raises(ValueError):
        _validate_component_anchors(
            replace(old_only, positive_observations={frame.name: measured_old[:5]}),
            [component],
            [],
        )


def _write_snapshot(root: Path, *, observation_source: str) -> tuple[dict, str]:
    root.mkdir(parents=True)
    session_id = "a23fdd50-session"
    bundle_sha = "a" * 64
    K = _K(64, 48, 50.0)
    point = np.array([0.2, 0.1, 4.0], dtype=np.float64)
    keyframes = {}
    manifests = []
    observations = []
    for ordinal, center in enumerate((np.array([0.0, 0.0, 0.0]), np.array([0.8, 0.0, 0.0]))):
        name = f"workmap_test_{ordinal:08d}"
        pose = _pose(center)
        xy = _pixel(K, pose, point)
        image_path = root / "images" / f"{name}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), np.zeros((48, 64), dtype=np.uint8))
        record = {
            "name": name,
            "session_id": session_id,
            "ordinal": ordinal,
            "source_epoch": 0,
            "stamp": float(ordinal + 1),
            "revision": ordinal + 1,
            "pose_revision": 0,
            "image_path": str(image_path),
            "image_size": [64, 48],
            "cam_from_world": pose.tolist(),
            "pose": pose.tolist(),
            "K": K.tolist(),
            "ids": [-7],
            "xy": [xy.tolist()],
            "xyz": [point.tolist()],
            "landmark_revisions": {"-7": 0},
            "durable_complete": True,
        }
        keyframes[name] = {
            **record,
            "durable": True,
            "ba_ids": [-7],
            "ba_xy": [xy.tolist()],
            "ba_xyz": [point.tolist()],
        }
        manifests.append(record)
        observations.append(
            {
                "session_id": session_id,
                "keyframe_name": name,
                "landmark_id": -7,
                "landmark_revision": 0,
                "ordinal": ordinal,
                "stamp": float(ordinal + 1),
                "xy": xy.tolist(),
                "xyz": point.tolist(),
                "cam_from_world": pose.tolist(),
                "image_path": str(image_path),
                "observation_source": observation_source,
            }
        )
    (root / "keyframes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifests), encoding="utf-8"
    )
    (root / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8"
    )
    (root / "session_meta.json").write_text(
        json.dumps(
            {
                "schema": "direct-online-workmap-session/v1",
                "session_id": session_id,
                "base_release_id": "base-release",
                "coordinate_frame_id": "base-frame",
                "base_bundle_sha256": bundle_sha,
                "source_epoch": 0,
                "writer_healthy": True,
            }
        ),
        encoding="utf-8",
    )
    snapshot = {
        "schema": "direct-online-workmap-session/v1",
        "session_id": session_id,
        "base_release_id": "base-release",
        "coordinate_frame_id": "base-frame",
        "base_bundle_sha256": bundle_sha,
        "revision": 2,
        "source_epoch": 0,
        "statistics": {"writer_healthy": True},
        "landmarks": {"-7": {"xyz": point.tolist(), "revision": 0}},
        "keyframes": keyframes,
    }
    return snapshot, bundle_sha


def test_snapshot_will_not_admit_pose_projected_pixels_as_observations(tmp_path: Path) -> None:
    snapshot, bundle_sha = _write_snapshot(
        tmp_path / "session", observation_source="pose_projected_pixel"
    )

    with pytest.raises(ValueError):
        _parse_session(snapshot, tmp_path / "session", "base-release", "base-frame", bundle_sha)


def test_wrong_observation_provenance_is_rejected_for_negative_id(tmp_path: Path) -> None:
    snapshot, bundle_sha = _write_snapshot(
        tmp_path / "session", observation_source="frozen_map_keyframe_image_measurement"
    )

    with pytest.raises(ValueError):
        _parse_session(snapshot, tmp_path / "session", "base-release", "base-frame", bundle_sha)


def test_archived_seed_measurements_bind_to_the_durable_capture(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    snapshot, bundle_sha = _write_snapshot(
        session_dir, observation_source="tracked_image_measurement"
    )
    manifest_path = session_dir / "keyframes.jsonl"
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    for row in manifest_rows:
        row["ids"] = []
        row["xy"] = []
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    # Durable first-view measurements may outlive the bounded BA candidate set.
    seed = snapshot["keyframes"]["workmap_test_00000000"]
    seed["ba_ids"] = []
    seed["ba_xy"] = []
    seed["ba_xyz"] = []

    observations_path = session_dir / "observations.jsonl"
    observation_rows = [json.loads(line) for line in observations_path.read_text().splitlines()]
    for row in observation_rows:
        row.pop("keyframe_name")
        row.pop("image_path")
    observations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in observation_rows), encoding="utf-8"
    )

    session = _parse_session(snapshot, session_dir, "base-release", "base-frame", bundle_sha)

    assert len(session.negative_observations[-7]) == 2
    assert {name for name, _xy in session.negative_observations[-7]} == {
        "workmap_test_00000000",
        "workmap_test_00000001",
    }


def test_multi_epoch_reused_ordinals_and_earlier_epoch_keyframes(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True)
    session_id = "multi-epoch-session"
    bundle_sha = "b" * 64
    K = _K(64, 48, 50.0)
    point = np.array([0.2, 0.1, 4.0], dtype=np.float64)

    # Frame 0: epoch 0, ordinal 0
    pose0 = _pose(np.array([0.0, 0.0, 0.0]))
    xy0 = _pixel(K, pose0, point)
    img0 = session_dir / "images" / "kf_ep0.png"
    img0.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(img0), np.zeros((48, 64), dtype=np.uint8))
    rec0 = {
        "name": "kf_ep0",
        "session_id": session_id,
        "ordinal": 0,
        "source_epoch": 0,
        "stamp": 10.0,
        "revision": 1,
        "pose_revision": 0,
        "image_path": str(img0),
        "image_size": [64, 48],
        "cam_from_world": pose0.tolist(),
        "pose": pose0.tolist(),
        "K": K.tolist(),
        "ids": [-7],
        "xy": [xy0.tolist()],
        "xyz": [point.tolist()],
        "landmark_revisions": {"-7": 0},
        "durable_complete": True,
    }

    # Frame 1: epoch 1, ordinal 0 (reused ordinal!)
    pose1 = _pose(np.array([0.8, 0.0, 0.0]))
    xy1 = _pixel(K, pose1, point)
    img1 = session_dir / "images" / "kf_ep1.png"
    assert cv2.imwrite(str(img1), np.zeros((48, 64), dtype=np.uint8))
    rec1 = {
        "name": "kf_ep1",
        "session_id": session_id,
        "ordinal": 0,
        "source_epoch": 1,
        "stamp": 20.0,
        "revision": 2,
        "pose_revision": 0,
        "image_path": str(img1),
        "image_size": [64, 48],
        "cam_from_world": pose1.tolist(),
        "pose": pose1.tolist(),
        "K": K.tolist(),
        "ids": [-7],
        "xy": [xy1.tolist()],
        "xyz": [point.tolist()],
        "landmark_revisions": {"-7": 0},
        "durable_complete": True,
    }

    manifests = [rec0, rec1]
    (session_dir / "keyframes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifests), encoding="utf-8"
    )

    observations = [
        # Tracked row with source_epoch 0, ordinal 0 -> must bind to kf_ep0
        {
            "session_id": session_id,
            "landmark_id": -7,
            "landmark_revision": 0,
            "source_epoch": 0,
            "ordinal": 0,
            "stamp": 10.0,
            "xy": xy0.tolist(),
            "xyz": point.tolist(),
            "cam_from_world": pose0.tolist(),
            "observation_source": "tracked_image_measurement",
        },
        # Tracked row without source_epoch, stamp 20.0 -> must bind to kf_ep1
        {
            "session_id": session_id,
            "landmark_id": -7,
            "landmark_revision": 0,
            "ordinal": 0,
            "stamp": 20.0,
            "xy": xy1.tolist(),
            "xyz": point.tolist(),
            "cam_from_world": pose1.tolist(),
            "observation_source": "tracked_image_measurement",
        },
    ]
    (session_dir / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8"
    )

    (session_dir / "session_meta.json").write_text(
        json.dumps(
            {
                "schema": "direct-online-workmap-session/v1",
                "session_id": session_id,
                "base_release_id": "base-release",
                "coordinate_frame_id": "base-frame",
                "base_bundle_sha256": bundle_sha,
                "source_epoch": 1,
                "writer_healthy": True,
            }
        ),
        encoding="utf-8",
    )

    snapshot = {
        "schema": "direct-online-workmap-session/v1",
        "session_id": session_id,
        "base_release_id": "base-release",
        "coordinate_frame_id": "base-frame",
        "base_bundle_sha256": bundle_sha,
        "revision": 2,
        "source_epoch": 1,
        "statistics": {"writer_healthy": True},
        "landmarks": {"-7": {"xyz": point.tolist(), "revision": 0}},
        "keyframes": {
            "kf_ep0": {
                **rec0,
                "durable": True,
                "ba_ids": [-7],
                "ba_xy": [xy0.tolist()],
                "ba_xyz": [point.tolist()],
            },
            "kf_ep1": {
                **rec1,
                "durable": True,
                "ba_ids": [-7],
                "ba_xy": [xy1.tolist()],
                "ba_xyz": [point.tolist()],
            },
        },
    }

    # 1. Multi-epoch parsing succeeds and correctly resolves bindings
    session = _parse_session(snapshot, session_dir, "base-release", "base-frame", bundle_sha)
    assert len(session.negative_observations[-7]) == 2
    assert {name for name, _xy in session.negative_observations[-7]} == {"kf_ep0", "kf_ep1"}

    # 2. Refusal of forged keyframe_name in tracked_image_measurement
    bad_obs = list(observations)
    bad_obs[0] = dict(bad_obs[0])
    bad_obs[0]["keyframe_name"] = "kf_ep1"
    (session_dir / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in bad_obs), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="forged or conflicting keyframe_name"):
        _parse_session(snapshot, session_dir, "base-release", "base-frame", bundle_sha)

    # 3. Refusal of duplicate capture identity in keyframes
    dup_manifests = [
        rec0,
        {**rec1, "source_epoch": 0, "name": "kf_dup"},
    ]
    (session_dir / "keyframes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dup_manifests), encoding="utf-8"
    )
    dup_snapshot = dict(snapshot)
    dup_snapshot["keyframes"] = {
        "kf_ep0": snapshot["keyframes"]["kf_ep0"],
        "kf_dup": {**snapshot["keyframes"]["kf_ep1"], "source_epoch": 0, "name": "kf_dup"},
    }
    with pytest.raises(ValueError, match="duplicate keyframe capture identity"):
        _parse_session(dup_snapshot, session_dir, "base-release", "base-frame", bundle_sha)


def test_real_pycolmap_merge_keeps_measured_tracks_and_camera_calibrations() -> None:
    reconstruction = pycolmap.Reconstruction()
    old_K = _K(640, 480, 400.0)
    old_camera = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 400.0, 640, 480)
    old_camera.params = np.array([400.0, 400.0, 320.0, 240.0])
    reconstruction.add_camera_with_trivial_rig(old_camera)
    old_xyz = np.array([0.2, 0.1, 5.0], dtype=np.float64)
    old_pixel = _pixel(old_K, _pose(np.array([0.0, 0.0, 0.0])), old_xyz)
    old_image = pycolmap.Image(
        name="old/frame.jpg",
        keypoints=np.asarray([old_pixel]),
        camera_id=1,
        image_id=1,
    )
    reconstruction.add_image_with_trivial_frame(
        old_image,
        pycolmap.Rigid3d(_pose(np.array([0.0, 0.0, 0.0]))),
    )
    reconstruction.add_point3D_with_id(
        1,
        pycolmap.Point3D(
            xyz=old_xyz,
            track=pycolmap.Track([pycolmap.TrackElement(1, 0)]),
            color=np.zeros(3, dtype=np.uint8),
        ),
    )

    session_id = "deadbeef-1234-session"
    new_xyz = np.array([0.4, 0.25, 6.0], dtype=np.float64)
    frames = []
    for ordinal, (name, width, height, focal, center) in enumerate(
        (
            ("kf_a", 640, 480, 420.0, np.array([0.7, 0.0, 0.0])),
            ("kf_b", 800, 600, 510.0, np.array([1.5, 0.2, 0.0])),
        )
    ):
        K = _K(width, height, focal)
        pose = _pose(center)
        new_xy = _pixel(K, pose, new_xyz)
        frames.append(
            _Keyframe(
                name=name,
                ordinal=ordinal,
                revision=ordinal + 1,
                pose_revision=0,
                source_epoch=0,
                stamp=float(ordinal),
                pose=pose,
                K=K,
                width=width,
                height=height,
                image_path=Path(f"{name}.png"),
                negative_ids=np.array([-9], dtype=np.int64),
                negative_xy=new_xy.reshape(1, 2),
                landmark_revisions={-9: 0},
            )
        )
    old_measured_xy = _pixel(frames[0].K, frames[0].pose, old_xyz)
    session = _Session(
        session_id=session_id,
        source_epoch=0,
        keyframes=tuple(frames),
        points_xyz={-9: new_xyz},
        negative_observations={
            -9: (("kf_a", frames[0].negative_xy[0]), ("kf_b", frames[1].negative_xy[0]))
        },
        positive_observations={"kf_a": ((1, old_measured_xy),), "kf_b": ()},
        input_hashes={},
    )
    before_cameras = _camera_calibrations(reconstruction)

    mapped, names, _indices = _add_session_geometry(
        reconstruction,
        session,
        {-9: new_xyz},
        {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        {1},
    )

    assert mapped == {-9: 2}
    assert names == {
        "kf_a": "workmap/deadbeef_kf_a.png",
        "kf_b": "workmap/deadbeef_kf_b.png",
    }
    assert reconstruction.point3D(1).track.length() == 2
    assert reconstruction.point3D(2).track.length() == 2
    first = reconstruction.find_image_with_name(names["kf_a"])
    second = reconstruction.find_image_with_name(names["kf_b"])
    assert first is not None and second is not None
    assert int(first.num_points3D) == 2
    assert int(second.num_points3D) == 1
    after_cameras = _camera_calibrations(reconstruction)
    assert after_cameras[1] == before_cameras[1]
    assert after_cameras[first.camera_id][1:3] == (640, 480)
    assert after_cameras[first.camera_id][3] == (420.0, 420.0, 320.0, 240.0)
    assert after_cameras[second.camera_id][1:3] == (800, 600)
    assert after_cameras[second.camera_id][3] == (510.0, 510.0, 400.0, 300.0)
    assert _point_image_components(reconstruction, {"old/frame.jpg"}, list(names.values())) == []


def test_original_gauge_uses_disjoint_fit_holdout_and_rejects_non_sim3_drift() -> None:
    original = {}
    for index in range(18):
        original[f"old/{index:02d}.jpg"] = np.array(
            [float(index % 4), float(index // 4), 0.13 * float((index * index) % 7)],
            dtype=np.float64,
        )
    exact = {name: center.copy() for name, center in original.items()}

    fitted = _fit_original_gauge(exact, original)

    assert fitted["pass"] is True
    assert fitted["fit"]["count"] >= 6
    assert fitted["holdout"]["count"] >= 6
    assert not set(fitted["fit"]["image_names"]) & set(fitted["holdout"]["image_names"])
    assert fitted["holdout"]["residual_p90"] < 1e-10

    distorted = {name: center.copy() for name, center in original.items()}
    holdout_name = fitted["holdout"]["image_names"][0]
    distorted[holdout_name] += np.array([0.0, 20.0, -8.0])
    with pytest.raises(ValueError):
        _fit_original_gauge(distorted, original)


def test_global_fixed_k_ba_optimizes_full_synthetic_reconstruction() -> None:
    from sfm_diagnosis.site_pipeline.robust_filter import RobustFilterConfig
    from tools.workmap_reconstruction import _fixed_k_global_ba

    reconstruction = pycolmap.Reconstruction()
    K = _K(640, 480, 420.0)
    camera = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 420.0, 640, 480)
    camera.params = np.array([420.0, 420.0, 320.0, 240.0])
    reconstruction.add_camera_with_trivial_rig(camera)
    centers = [
        np.array([float(index % 4) * 0.35, float(index // 4) * 0.3, 0.0]) for index in range(12)
    ]
    points = [
        np.array([0.2 * (index % 5) - 0.35, 0.18 * (index // 5) - 0.2, 5.5 + 0.15 * (index % 3)])
        for index in range(18)
    ]
    for image_index, center in enumerate(centers, start=1):
        pose = _pose(center)
        pixels = np.asarray([_pixel(K, pose, point) for point in points], dtype=np.float64)
        image = pycolmap.Image(
            name=f"old/{image_index:02d}.jpg",
            keypoints=pixels,
            camera_id=1,
            image_id=image_index,
        )
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(pose))
    for point_index, true_point in enumerate(points, start=1):
        noisy = true_point + np.array([0.025, -0.02, 0.035])
        track = pycolmap.Track(
            [
                pycolmap.TrackElement(image_id, point_index - 1)
                for image_id in range(1, len(centers) + 1)
            ]
        )
        reconstruction.add_point3D_with_id(
            point_index,
            pycolmap.Point3D(
                xyz=noisy,
                track=track,
                color=np.zeros(3, dtype=np.uint8),
            ),
        )
    cameras_before = _camera_calibrations(reconstruction)
    reconstruction.update_point_3d_errors()
    mean_before = reconstruction.compute_mean_reprojection_error()

    receipt = _fixed_k_global_ba(
        reconstruction,
        RobustFilterConfig(
            max_reprojection_error_px=3.0,
            minimum_triangulation_angle_deg=1.5,
            minimum_track_length=2,
            bundle_adjustment_iterations=25,
            bundle_adjustment_threads=2,
        ),
    )

    assert receipt["invoked"] is True
    assert receipt["global_registered_images"] == 12
    assert receipt["summary"]["num_residuals"] > 0
    assert receipt["final_cost"] < receipt["initial_cost"]
    assert reconstruction.compute_mean_reprojection_error() < mean_before
    assert _camera_calibrations(reconstruction) == cameras_before


def test_unmerged_submaps_and_fused_duplicates_stay_out_of_the_offline_model() -> None:
    from tools.workmap_reconstruction import _without_unmerged_submaps

    def frame(name, submap_id, ids):
        return {
            "name": name, "submap_id": submap_id, "ids": list(ids),
            "xy": [[float(i), 0.0] for i in ids], "xyz": [[0.0, 0.0, float(-i)] for i in ids],
            "landmark_revisions": {str(i): 1 for i in ids},
        }

    snapshot = {
        "submaps": [
            {"id": 0, "state": "anchored"},
            {"id": 1, "state": "merged"},
            {"id": 2, "state": "unmerged"},
        ],
        "landmarks": {
            "-1": {"submap_id": 0},
            "-2": {"submap_id": 1},
            "-3": {"submap_id": 2},
            "-4": {"submap_id": 0, "fused_to": 17},
        },
        "keyframes": {
            "kept": frame("kept", 0, [-1, -2, -3, -4]),
            "lost": frame("lost", 2, [-3]),
        },
    }
    manifest = {"kept": frame("kept", 0, [-1, -3]), "lost": frame("lost", 2, [-3])}
    filtered, filtered_manifest, skip = _without_unmerged_submaps(snapshot, manifest)
    assert set(filtered["landmarks"]) == {"-1", "-2"}
    assert [row["name"] for row in filtered["keyframes"]] == ["kept"]
    assert filtered["keyframes"][0]["ids"] == [-1, -2]
    assert set(filtered["keyframes"][0]["landmark_revisions"]) == {"-1", "-2"}
    assert set(filtered_manifest) == {"kept"} and filtered_manifest["kept"]["ids"] == [-1]
    assert skip({"landmark_id": -3}) and skip({"landmark_id": -1, "keyframe_name": "lost"})
    assert not skip({"landmark_id": -2, "keyframe_name": "kept"})
    # Nothing to drop: the original objects pass through untouched.
    plain = {"landmarks": {"-1": {}}, "keyframes": {"a": frame("a", 0, [-1])}}
    assert _without_unmerged_submaps(plain, {"a": {}}) == (plain, {"a": {}}, None)

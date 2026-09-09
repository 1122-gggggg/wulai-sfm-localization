#!/usr/bin/env python3
"""Rebuild GlueMap observations with official EDM on frozen poses (RTX 5060 live set)."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = WORKSPACE_ROOT / "定位演算法" / "deploy_code" / "sfm_direct_deploy" / "vendor"
DEFAULT_MAP_ROOT = (
    WORKSPACE_ROOT / "river-deploy-5060-20260908" / "river_gluemap_all8_direct_20260831"
)
DEFAULT_EXPERIMENT_SUBDIR = Path("experiments") / "edm_fixedpose_retriangulate"
DEFAULT_EDM_REPO = WORKSPACE_ROOT / "定位演算法" / "deploy_code" / "runtime" / "EDM"
DEFAULT_EDM_CHECKPOINT = DEFAULT_EDM_REPO / "weights" / "edm_outdoor.ckpt"
DEFAULT_MEGALOC_SOURCE = (
    WORKSPACE_ROOT / "執行環境" / "torch_hub_cache" / "gmberton_MegaLoc_main"
)
MEGALOC_CHECKPOINT_GLOB = "models--gberton--MegaLoc/snapshots/*/model.safetensors"


def _vendor_sys_path() -> None:
    """Make the vendored river_map_quality / sfm_diagnosis packages importable (idempotent)."""
    entry = str(VENDOR_ROOT)
    if VENDOR_ROOT.is_dir() and entry not in sys.path:
        sys.path.insert(0, entry)


_vendor_sys_path()


def _default_megaloc_checkpoint() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    found = sorted(hub.glob(MEGALOC_CHECKPOINT_GLOB))
    return found[0] if found else None


P172_PREVIOUS_STRONG = 5
P172_PREVIOUS_N = 124

EXPECTED_N_IMAGES = 1058
EXPECTED_POSE_ONLY = 12
CELL_PX = 2.0
COVIS_MIN_SHARED = 15
COVIS_TOP_K = 20
TEMPORAL_ORDINAL = 8
MAX_PAIRS = 25000
SAMPSON_MAX_PX = 3.0
MIN_TRIANGULATION_ANGLE_DEG = 1.0
MIN_INLIERS = 30
KEYPOINT_CAP = int(os.environ.get("P172_KEYPOINT_CAP", "50000"))
KEYPOINT_CAP_OOM = 20000
TEST_RES = int(os.environ.get("P172_TEST_RES", "640"))
EDM_TOPK = int(os.environ.get("P172_EDM_TOPK", "2240"))
MEGALOC_TOP_K = int(os.environ.get("P172_MEGALOC_TOP_K", "5"))
MEGALOC_INPUT_SIZE = 322
CONFIDENCE_THRESHOLD = 0.0
VRAM_LIMIT_MIB = 7000.0
SCALE_5090_TO_5060 = 4.0
OCCUPIED_2PX_MIN = 0.00495
NN_SAMPLE_IMAGES = 40
NN_MAX_POINTS = 2000
CELL_SIZES_PX = (2, 8, 16)
SMOKE_IDENTITY = "vid_349c83c4bf56a785/frame_00000000.jpg"
DIAGNOSE_NEIGHBORS = 8
R_COVER_MIN = 0.15
LIFT_DISTANCE_PX = float(os.environ.get("P172_LIFT_PX", "4.0"))  # 2x CELL_PX: tolerate cell-representative churn
SMOKE_QUERY_NAME = "vid_349c83c4bf56a785/frame_00000000.jpg"
BENCH_REF_NAME = "vid_349c83c4bf56a785/frame_00000024.jpg"


def _apply_roots(
    *,
    map_root: Path,
    experiment_root: Path | None = None,
    edm_repo: Path = DEFAULT_EDM_REPO,
    edm_checkpoint: Path = DEFAULT_EDM_CHECKPOINT,
    megaloc_source: Path = DEFAULT_MEGALOC_SOURCE,
    megaloc_checkpoint: Path | None = None,
    baseline_smoke: Path | None = None,
) -> None:
    """Bind every filesystem-dependent global. Called at import with defaults, then from main()."""
    global MAP_ROOT, EXPERIMENT_ROOT, MODEL, IMAGES, KEYFRAMES, LOCALIZER_CONFIG
    global RT5060_EDM_BASE, RT5060_MEGADEPTH, BASELINE_SMOKE, SAME_QUERY, BENCH_REF
    global P172_FRAMES, DEPTH_DIR, HEDGE_DEPTH_DIR
    global EDM_REPO, EDM_CHECKPOINT, MEGALOC_SOURCE, MEGALOC_CHECKPOINT

    MAP_ROOT = Path(map_root).resolve()
    EXPERIMENT_ROOT = (
        Path(experiment_root).resolve()
        if experiment_root is not None
        else MAP_ROOT / DEFAULT_EXPERIMENT_SUBDIR
    )
    MODEL = MAP_ROOT / "model"
    IMAGES = MAP_ROOT / "keyframes" / "images"
    KEYFRAMES = MAP_ROOT / "keyframes" / "keyframes.jsonl"
    LOCALIZER_CONFIG = MAP_ROOT / "localization" / "localizer_config.json"
    RT5060_EDM_BASE = EXPERIMENT_ROOT / "rt5060_edm_base.py"
    RT5060_MEGADEPTH = EXPERIMENT_ROOT / "rt5060_megadepth.py"
    BASELINE_SMOKE = (
        Path(baseline_smoke).resolve()
        if baseline_smoke is not None
        else EXPERIMENT_ROOT / "records" / "smoke_localization.json"
    )
    SAME_QUERY = IMAGES / SMOKE_QUERY_NAME
    BENCH_REF = IMAGES / BENCH_REF_NAME
    P172_FRAMES = MAP_ROOT / "experiments" / "cell_anchor_dense_lift" / "p172_heldout" / "frames"
    DEPTH_DIR = MAP_ROOT / "experiments" / "cell_anchor_dense_lift" / "depth"
    # map-driven weak-zone subset (183 refs, missing -> tracks fallback)
    HEDGE_DEPTH_DIR = EXPERIMENT_ROOT / "hedge_depth"
    EDM_REPO = Path(edm_repo)
    EDM_CHECKPOINT = Path(edm_checkpoint)
    MEGALOC_SOURCE = Path(megaloc_source)
    MEGALOC_CHECKPOINT = Path(megaloc_checkpoint) if megaloc_checkpoint is not None else None


_apply_roots(map_root=DEFAULT_MAP_ROOT, megaloc_checkpoint=_default_megaloc_checkpoint())


def _require_path(value: Path | None, flag: str) -> Path:
    if value is None:
        raise SystemExit(f"{flag} is required; no default could be resolved on this machine")
    if not value.exists():
        raise SystemExit(f"{flag} path does not exist: {value}")
    return value


def _load_localizer_config() -> dict[str, Any]:
    """Read the map's localizer_config.json and override the weight paths with the CLI values."""
    if not LOCALIZER_CONFIG.is_file():
        raise SystemExit(f"localizer_config.json missing: {LOCALIZER_CONFIG}")
    config = json.loads(LOCALIZER_CONFIG.read_text(encoding="utf-8"))
    config["megaloc_source"] = str(_require_path(MEGALOC_SOURCE, "--megaloc-source"))
    config["megaloc_checkpoint"] = str(_require_path(MEGALOC_CHECKPOINT, "--megaloc-checkpoint"))
    config["edm_root"] = str(EDM_REPO)
    config["edm_checkpoint"] = str(EDM_CHECKPOINT)
    return config


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _occupied_fraction(xy: np.ndarray, width: int, height: int, cell_px: float) -> float:
    nx = max(1, int(math.ceil(width / cell_px)))
    ny = max(1, int(math.ceil(height / cell_px)))
    if xy.size == 0:
        return 0.0
    cells = {
        (int(math.floor(x / cell_px)), int(math.floor(y / cell_px)))
        for x, y in xy
        if 0.0 <= x < width and 0.0 <= y < height
    }
    return len(cells) / float(nx * ny)


def _nn_spacings(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 2:
        return np.empty(0, dtype=np.float64)
    delta = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def _stratified_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    sample = min(k, n)
    if sample == 1:
        return [0]
    return [int(round(i * (n - 1) / (sample - 1))) for i in range(sample)]


def _compute_pids() -> list[tuple[str, str]]:
    import subprocess

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"nvidia-smi failed: {result.stderr.strip()}")
    rows: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, name = line.partition(",")
        rows.append((pid.strip(), name.strip()))
    return rows


def require_free_gpu() -> None:
    busy = _compute_pids()
    if busy:
        pretty = ", ".join(f"{pid} {name}" for pid, name in busy)
        raise SystemExit(f"GPU compute busy ({pretty}); not starting CUDA work")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _admit_runtime() -> None:
    from sfm_diagnosis.site_pipeline.deployment_localizer import (
        _configure_audited_runtime_site_packages,
        _resolve_audited_site_packages,
    )

    _configure_audited_runtime_site_packages(_resolve_audited_site_packages(None))


def _keyframes() -> dict[str, dict]:
    rows = [json.loads(line) for line in KEYFRAMES.read_text(encoding="utf-8").splitlines() if line]
    return {str(row["output_name"]): row for row in rows}


def quantize_xy(uv: np.ndarray | tuple[float, float], cell: float = CELL_PX) -> tuple[int, int]:
    u, v = float(uv[0]), float(uv[1])
    return (int(math.floor(u / cell)), int(math.floor(v / cell)))


def _skew(t: np.ndarray) -> np.ndarray:
    tx, ty, tz = (float(t[0]), float(t[1]), float(t[2]))
    return np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]], dtype=np.float64)


def _w2c_4x4(image: Any) -> np.ndarray:
    matrix = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        pose = np.eye(4, dtype=np.float64)
        pose[:3] = matrix
        return pose
    raise ValueError(f"unexpected cam_from_world matrix shape {matrix.shape}")


def _k_from_camera(camera: Any) -> np.ndarray:
    model = str(getattr(camera.model, "name", camera.model)).rsplit(".", 1)[-1]
    params = [float(value) for value in camera.params]
    if model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        raise ValueError(f"expected PINHOLE camera, got {camera.model!r}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def sampson_pixels(uv1: np.ndarray, uv2: np.ndarray, essential: np.ndarray, k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    points1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(uv2, dtype=np.float64).reshape(-1, 2)
    fundamental = np.linalg.inv(k2).T @ essential @ np.linalg.inv(k1)
    ones = np.ones((len(points1), 1), dtype=np.float64)
    x1 = np.concatenate([points1, ones], axis=1)
    x2 = np.concatenate([points2, ones], axis=1)
    fx1 = x1 @ fundamental.T
    ftx2 = x2 @ fundamental
    num = np.sum(x2 * fx1, axis=1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.sqrt(np.maximum(num / np.maximum(den, 1e-18), 0.0))


def triangulate_linear(uv1: np.ndarray, uv2: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    points1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(uv2, dtype=np.float64).reshape(-1, 2)
    n = len(points1)
    design = np.empty((n, 4, 4), dtype=np.float64)
    design[:, 0] = points1[:, 0:1] * p1[2] - p1[0]
    design[:, 1] = points1[:, 1:2] * p1[2] - p1[1]
    design[:, 2] = points2[:, 0:1] * p2[2] - p2[0]
    design[:, 3] = points2[:, 1:2] * p2[2] - p2[1]
    _, _, vh = np.linalg.svd(design)
    homogeneous = vh[:, -1, :]
    w = homogeneous[:, 3:4]
    return homogeneous[:, :3] / np.where(np.abs(w) < 1e-12, np.nan, w)


def camera_centers(t_w2c: np.ndarray) -> np.ndarray:
    rotation = t_w2c[:3, :3]
    translation = t_w2c[:3, 3]
    return -rotation.T @ translation


def triangulation_angles_deg(xyz: np.ndarray, center_a: np.ndarray, center_b: np.ndarray) -> np.ndarray:
    ray_a = xyz - center_a
    ray_b = xyz - center_b
    norm_a = np.linalg.norm(ray_a, axis=1)
    norm_b = np.linalg.norm(ray_b, axis=1)
    denom = np.maximum(norm_a * norm_b, 1e-18)
    cosine = np.clip(np.sum(ray_a * ray_b, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def frozen_pose_inliers(
    uv1: np.ndarray,
    uv2: np.ndarray,
    t_i: np.ndarray,
    t_j: np.ndarray,
    k_i: np.ndarray,
    k_j: np.ndarray,
    *,
    sampson_max_px: float = SAMPSON_MAX_PX,
    min_angle_deg: float = MIN_TRIANGULATION_ANGLE_DEG,
) -> np.ndarray:
    points1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(uv2, dtype=np.float64).reshape(-1, 2)
    if len(points1) == 0:
        return np.zeros(0, dtype=bool)
    relative = t_j @ np.linalg.inv(t_i)
    rotation = relative[:3, :3]
    translation = relative[:3, 3]
    essential = _skew(translation) @ rotation
    sampson = sampson_pixels(points1, points2, essential, k_i, k_j)
    p1 = k_i @ t_i[:3]
    p2 = k_j @ t_j[:3]
    xyz = triangulate_linear(points1, points2, p1, p2)
    cam_z_i = (t_i[:3, :3] @ xyz.T + t_i[:3, 3:4])[2]
    cam_z_j = (t_j[:3, :3] @ xyz.T + t_j[:3, 3:4])[2]
    angles = triangulation_angles_deg(xyz, camera_centers(t_i), camera_centers(t_j))
    return (
        np.isfinite(sampson)
        & np.isfinite(xyz).all(axis=1)
        & np.isfinite(cam_z_i)
        & np.isfinite(cam_z_j)
        & np.isfinite(angles)
        & (sampson <= sampson_max_px)
        & (cam_z_i > 0.0)
        & (cam_z_j > 0.0)
        & (angles >= min_angle_deg)
    )


def pair_id(name_a: str, name_b: str) -> str:
    return hashlib.sha256((name_a + "\0" + name_b).encode("utf-8")).hexdigest()[:16]


def _empty_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_megaloc(runtime: Any) -> None:
    del runtime
    _empty_cuda()


def _load_edm_runtime():
    from river_map_quality.official_edm_adapter_loo import load_official_edm_runtime

    runtime = load_official_edm_runtime(
        edm_repo=EDM_REPO,
        checkpoint=EDM_CHECKPOINT,
        config_path=RT5060_EDM_BASE,
        data_config_path=RT5060_MEGADEPTH,
        device="cuda",
    )
    print(
        f"load_official_edm_runtime input_width={runtime.input_width} "
        f"input_height={runtime.input_height} coarse_topk={runtime.coarse_topk}",
        flush=True,
    )
    if int(runtime.input_width) != TEST_RES or int(runtime.input_height) != TEST_RES:
        raise SystemExit(
            f"rt5060 configs did not merge: input_width={runtime.input_width} (expected {TEST_RES})"
        )
    if int(runtime.coarse_topk) != EDM_TOPK:
        raise SystemExit(
            f"rt5060 TOPK did not merge: coarse_topk={runtime.coarse_topk} (expected {EDM_TOPK})"
        )
    return runtime


def _match_paths(runtime: Any, path_a: Path, path_b: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2
    from river_map_quality.official_edm_adapter_loo import prepare_official_megadepth_image
    from sfm_diagnosis.site_pipeline.deployment_localizer import (
        _match_official_prepared,
        _prepare_official_image,
        _valid_matches,
    )

    image_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if image_a is None:
        raise FileNotFoundError(path_a)
    if image_b is None:
        raise FileNotFoundError(path_b)
    prepared_a = _prepare_official_image(path_a, runtime, prepare_image=prepare_official_megadepth_image)
    prepared_b = _prepare_official_image(path_b, runtime, prepare_image=prepare_official_megadepth_image)
    matched = _match_official_prepared(runtime, prepared_a, prepared_b)
    mkpts0 = np.asarray(matched["mkpts0_f"], dtype=np.float32).reshape(-1, 2)
    mkpts1 = np.asarray(matched["mkpts1_f"], dtype=np.float32).reshape(-1, 2)
    mconf = np.asarray(matched["mconf"], dtype=np.float32).reshape(-1)
    valid = _valid_matches(
        mkpts0,
        mkpts1,
        mconf,
        query_shape=image_a.shape,
        reference_shape=image_b.shape,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    return mkpts0[valid], mkpts1[valid], mconf[valid]


class _PreparedCache:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._prepared: dict[Path, Any] = {}
        self._shape: dict[Path, tuple[int, int]] = {}

    def get(self, path: Path):
        import cv2
        from river_map_quality.official_edm_adapter_loo import prepare_official_megadepth_image
        from sfm_diagnosis.site_pipeline.deployment_localizer import _prepare_official_image

        resolved = path.resolve(strict=True)
        if resolved not in self._prepared:
            image = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(resolved)
            self._shape[resolved] = (int(image.shape[0]), int(image.shape[1]))
            self._prepared[resolved] = _prepare_official_image(
                resolved, self.runtime, prepare_image=prepare_official_megadepth_image
            )
        return self._prepared[resolved], self._shape[resolved]


def _match_cached(cache: _PreparedCache, path_a: Path, path_b: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sfm_diagnosis.site_pipeline.deployment_localizer import _match_official_prepared, _valid_matches

    prepared_a, shape_a = cache.get(path_a)
    prepared_b, shape_b = cache.get(path_b)
    matched = _match_official_prepared(cache.runtime, prepared_a, prepared_b)
    mkpts0 = np.asarray(matched["mkpts0_f"], dtype=np.float32).reshape(-1, 2)
    mkpts1 = np.asarray(matched["mkpts1_f"], dtype=np.float32).reshape(-1, 2)
    mconf = np.asarray(matched["mconf"], dtype=np.float32).reshape(-1)
    valid = _valid_matches(
        mkpts0,
        mkpts1,
        mconf,
        query_shape=shape_a,
        reference_shape=shape_b,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    return mkpts0[valid], mkpts1[valid], mconf[valid]


def _covered_fraction(query_xy: np.ndarray, map_xy: np.ndarray, cell: float = CELL_PX) -> float:
    query = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    mapped = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
    if len(query) == 0:
        return 0.0
    if len(mapped) == 0:
        return 0.0
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(mapped):
        grid[(int(math.floor(point[0] / cell)), int(math.floor(point[1] / cell)))].append(index)
    max_distance_sq = cell * cell
    hits = 0
    for point in query:
        x_cell = int(math.floor(point[0] / cell))
        y_cell = int(math.floor(point[1] / cell))
        candidates = [
            index
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for index in grid.get((x_cell + dx, y_cell + dy), ())
        ]
        if not candidates:
            continue
        deltas = mapped[np.asarray(candidates, dtype=np.int64)] - point
        if np.any(np.einsum("ij,ij->i", deltas, deltas) <= max_distance_sq):
            hits += 1
    return hits / float(len(query))


def _pose_only_and_tracks(reconstruction: Any) -> tuple[set[str], dict[str, set[int]], dict[str, int]]:
    keyframes = _keyframes()
    pose_only_mode = {
        name for name, row in keyframes.items() if str(row.get("mapping_mode") or "") == "POSE_ONLY"
    }
    pose_only: set[str] = set()
    tracks: dict[str, set[int]] = {}
    image_ids: dict[str, int] = {}
    for image_id, image in reconstruction.images.items():
        name = str(image.name)
        image_ids[name] = int(image_id)
        if name in pose_only_mode:
            pose_only.add(name)
            continue
        tracks[name] = {int(point.point3D_id) for point in image.points2D if point.has_point3D()}
    return pose_only, tracks, image_ids


def cmd_selftest(_args: argparse.Namespace) -> int:
    same = quantize_xy((0.05, 0.05), cell=2.0)
    near = quantize_xy((1.95, 0.05), cell=2.0)
    far = quantize_xy((2.15, 0.05), cell=2.0)
    if same != near:
        raise SystemExit(f"1.9 px pair must share a cell: {same} vs {near}")
    if same == far:
        raise SystemExit(f"2.1 px pair must not share a cell: {same} vs {far}")

    k = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t_i = np.eye(4, dtype=np.float64)
    t_j = np.eye(4, dtype=np.float64)
    t_j[:3, 3] = (1.0, 0.0, 0.0)
    xyz = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    uv1 = np.array([[10.0, 10.0]], dtype=np.float64)
    uv2 = np.array([[30.0, 10.0]], dtype=np.float64)
    cam1 = t_i[:3, :3] @ xyz.T + t_i[:3, 3:4]
    cam2 = t_j[:3, :3] @ xyz.T + t_j[:3, 3:4]
    if cam1[2, 0] <= 0.0 or cam2[2, 0] <= 0.0:
        raise SystemExit("cheirality failed on the known-pose point")
    relative = t_j @ np.linalg.inv(t_i)
    essential = _skew(relative[:3, 3]) @ relative[:3, :3]
    good = float(sampson_pixels(uv1, uv2, essential, k, k)[0])
    if good >= 0.1:
        raise SystemExit(f"perfect projection Sampson={good} px, expected < 0.1")
    perturbed = np.array([[30.0, 20.0]], dtype=np.float64)
    bad = float(sampson_pixels(uv1, perturbed, essential, k, k)[0])
    if bad <= 3.0:
        raise SystemExit(f"perturbed Sampson={bad} px, expected > 3")
    keep_good = frozen_pose_inliers(uv1, uv2, t_i, t_j, k, k)
    keep_bad = frozen_pose_inliers(uv1, perturbed, t_i, t_j, k, k)
    if not bool(keep_good[0]):
        raise SystemExit("known-pose correspondence was rejected")
    if bool(keep_bad[0]):
        raise SystemExit("perturbed correspondence was kept")
    print("selftest ok", flush=True)
    return 0


def cmd_pairs(_args: argparse.Namespace) -> int:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(MODEL))
    n_reg = int(reconstruction.num_reg_images())
    if n_reg != EXPECTED_N_IMAGES:
        raise SystemExit(f"num_reg_images={n_reg} != {EXPECTED_N_IMAGES}; abort")
    pose_only, tracks, _image_ids = _pose_only_and_tracks(reconstruction)
    if len(pose_only) != EXPECTED_POSE_ONLY:
        raise SystemExit(f"pose-only images={len(pose_only)} != {EXPECTED_POSE_ONLY}; abort")
    keyframes = _keyframes()

    point_to_images: dict[int, list[str]] = defaultdict(list)
    for name, pids in tracks.items():
        for pid in pids:
            point_to_images[pid].append(name)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for names in point_to_images.values():
        unique = sorted(set(names))
        for i, left in enumerate(unique):
            for right in unique[i + 1 :]:
                shared[(left, right)] += 1

    neighbors: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (left, right), count in shared.items():
        if count < COVIS_MIN_SHARED:
            continue
        neighbors[left].append((count, right))
        neighbors[right].append((count, left))

    covis: dict[tuple[str, str], int] = {}
    for name, items in neighbors.items():
        items.sort(key=lambda row: (-row[0], row[1]))
        for count, other in items[:COVIS_TOP_K]:
            pair = (name, other) if name < other else (other, name)
            covis[pair] = shared[pair]

    temporal: set[tuple[str, str]] = set()
    by_video: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name in tracks:
        row = keyframes[name]
        by_video[name.split("/")[0]].append((int(row["frame_index"]), name))
    for rows in by_video.values():
        rows.sort()
        ordered = [name for _index, name in rows]
        for i, left in enumerate(ordered):
            for delta in range(1, TEMPORAL_ORDINAL + 1):
                j = i + delta
                if j >= len(ordered):
                    break
                right = ordered[j]
                pair = (left, right) if left < right else (right, left)
                temporal.add(pair)

    all_pairs = set(covis) | temporal
    n_pairs = len(all_pairs)
    summary = {
        "n_pairs": n_pairs,
        "n_images": n_reg,
        "n_pose_only": len(pose_only),
        "n_covisibility": len(covis),
        "n_temporal": len(temporal),
        "n_both": len(set(covis) & temporal),
        "pose_only": sorted(pose_only),
    }
    if n_pairs > MAX_PAIRS:
        _write_json(EXPERIMENT_ROOT / "pairs_summary.json", summary)
        raise SystemExit(f"n_pairs={n_pairs} > {MAX_PAIRS}; abort, summary dumped")

    out = EXPERIMENT_ROOT / "pairs.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for name_a, name_b in sorted(all_pairs):
            in_covis = (name_a, name_b) in covis
            in_temporal = (name_a, name_b) in temporal
            if in_covis and in_temporal:
                reason = "both"
            elif in_covis:
                reason = "covisibility"
            else:
                reason = "temporal"
            payload = {
                "name_a": name_a,
                "name_b": name_b,
                "reason": reason,
                "shared_points": int(covis.get((name_a, name_b), 0)),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _write_json(EXPERIMENT_ROOT / "pairs_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def _load_pairs() -> list[dict[str, Any]]:
    path = EXPERIMENT_ROOT / "pairs.jsonl"
    if not path.is_file():
        raise SystemExit("pairs.jsonl missing; run pairs first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _bench_gate() -> dict[str, Any]:
    verdict = EXPERIMENT_ROOT / "bench" / "verdict.json"
    if verdict.is_file():
        payload = json.loads(verdict.read_text(encoding="utf-8"))
        if payload.get("status") == "OOM_RISK_8GB":
            raise SystemExit("bench VRAM gate failed (OOM_RISK_8GB); stop")
    path = EXPERIMENT_ROOT / "bench" / "rt5060.json"
    if not path.is_file():
        raise SystemExit("bench/rt5060.json missing; run bench first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if float(payload["peak_allocated_mib"]) > VRAM_LIMIT_MIB:
        raise SystemExit("bench peak_allocated_mib exceeds 7000; stop")
    return payload


def _diagnose_gate() -> dict[str, Any]:
    verdict = EXPERIMENT_ROOT / "diagnose" / "verdict.json"
    if verdict.is_file():
        payload = json.loads(verdict.read_text(encoding="utf-8"))
        if payload.get("status") == "INSUFFICIENT":
            raise SystemExit(f"diagnose failed: {payload.get('reason')}")
    path = EXPERIMENT_ROOT / "diagnose" / "cover.json"
    if not path.is_file():
        raise SystemExit("diagnose/cover.json missing; run diagnose first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if float(payload["mean_r_cover"]) < R_COVER_MIN:
        raise SystemExit(f"mean r_cover={payload['mean_r_cover']} < {R_COVER_MIN}; stop")
    return payload


def cmd_bench(_args: argparse.Namespace) -> int:
    require_free_gpu()
    _admit_runtime()
    from river_map_quality.megaloc_edm_catalog import (
        MEGALOC_INPUT_SIZE as CATALOG_INPUT_SIZE,
        extract_megaloc_descriptors,
        load_offline_megaloc_runtime,
    )

    config = _load_localizer_config()
    if int(CATALOG_INPUT_SIZE) != MEGALOC_INPUT_SIZE:
        raise SystemExit(f"MegaLoc input_size={CATALOG_INPUT_SIZE} != {MEGALOC_INPUT_SIZE}")

    megaloc = load_offline_megaloc_runtime(
        source=Path(str(config["megaloc_source"])),
        checkpoint=Path(str(config["megaloc_checkpoint"])),
        device="cuda",
    )
    extract_megaloc_descriptors(megaloc, [SAME_QUERY], batch_size=1)
    _unload_megaloc(megaloc)

    runtime = _load_edm_runtime()
    _match_paths(runtime, SAME_QUERY, BENCH_REF)
    _match_paths(runtime, SAME_QUERY, BENCH_REF)
    del runtime
    _empty_cuda()

    import torch

    torch.cuda.reset_peak_memory_stats()
    megaloc = load_offline_megaloc_runtime(
        source=Path(str(config["megaloc_source"])),
        checkpoint=Path(str(config["megaloc_checkpoint"])),
        device="cuda",
    )
    t0 = time.perf_counter()
    extract_megaloc_descriptors(megaloc, [SAME_QUERY], batch_size=1)
    runtime_megaloc = time.perf_counter() - t0
    _unload_megaloc(megaloc)

    runtime = _load_edm_runtime()
    t1 = time.perf_counter()
    _match_paths(runtime, SAME_QUERY, BENCH_REF)
    runtime_edm = time.perf_counter() - t1
    peak_allocated_mib = float(torch.cuda.max_memory_allocated()) / 1048576.0
    del runtime
    _empty_cuda()

    runtime_total = runtime_megaloc + runtime_edm
    estimated_5060_s = runtime_total * SCALE_5090_TO_5060
    payload = {
        "edm_test_res": TEST_RES,
        "edm_topk": EDM_TOPK,
        "megaloc_top_k": MEGALOC_TOP_K,
        "megaloc_input_size": MEGALOC_INPUT_SIZE,
        "runtime_megaloc_s": runtime_megaloc,
        "runtime_edm_s": runtime_edm,
        "runtime_total_s": runtime_total,
        "peak_allocated_mib": peak_allocated_mib,
        "estimated_5060_s": estimated_5060_s,
        "estimated_5060_hz": (1.0 / estimated_5060_s) if estimated_5060_s > 0 else float("inf"),
        "scale_5090_to_desktop_5060": SCALE_5090_TO_5060,
    }
    _write_json(EXPERIMENT_ROOT / "bench" / "rt5060.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if peak_allocated_mib > VRAM_LIMIT_MIB:
        verdict = {"status": "OOM_RISK_8GB", "peak_allocated_mib": peak_allocated_mib}
        _write_json(EXPERIMENT_ROOT / "bench" / "verdict.json", verdict)
        raise SystemExit("peak_allocated_mib > 7000; stop")
    if estimated_5060_s > 0.20:
        print(
            f"estimated_5060_s={estimated_5060_s:.4f} > 0.20; keeping K=1/640 and continuing",
            flush=True,
        )
    return 0


def cmd_diagnose(_args: argparse.Namespace) -> int:
    require_free_gpu()
    _admit_runtime()
    _bench_gate()
    import pycolmap

    pairs = _load_pairs()
    baseline = json.loads(BASELINE_SMOKE.read_text(encoding="utf-8"))
    reference_ids = [str(name) for name in baseline["result"]["reference_ids"] if str(name) != SMOKE_IDENTITY]
    if len(reference_ids) != 4:
        raise SystemExit(f"expected 4 non-identity smoke refs, got {reference_ids}")

    incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        incident[str(row["name_a"])].append(row)
        incident[str(row["name_b"])].append(row)

    reconstruction = pycolmap.Reconstruction(str(MODEL))
    observations: dict[str, np.ndarray] = {}
    for image in reconstruction.images.values():
        points = [point.xy for point in image.points2D if point.has_point3D()]
        observations[str(image.name)] = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    runtime = _load_edm_runtime()
    cache = _PreparedCache(runtime)
    per_ref = []
    for ref in reference_ids:
        mkpts_q, mkpts_r, _conf = _match_cached(cache, SAME_QUERY, IMAGES / ref)
        q_r = mkpts_r
        neighbors = incident.get(ref, [])
        neighbors = sorted(
            neighbors,
            key=lambda row: (
                -int(row["shared_points"]),
                0 if str(row["reason"]) in {"temporal", "both"} else 1,
                str(row["name_a"]),
                str(row["name_b"]),
            ),
        )
        neighbor_names = []
        for row in neighbors:
            other = str(row["name_b"]) if str(row["name_a"]) == ref else str(row["name_a"])
            if other not in neighbor_names:
                neighbor_names.append(other)
            if len(neighbor_names) >= DIAGNOSE_NEIGHBORS:
                break
        mapping_xy = []
        for neighbor in neighbor_names:
            mk0, _mk1, _c = _match_cached(cache, IMAGES / ref, IMAGES / neighbor)
            mapping_xy.append(mk0)
        m_r = np.concatenate(mapping_xy, axis=0) if mapping_xy else np.empty((0, 2), dtype=np.float64)
        r_cover = _covered_fraction(q_r, m_r, cell=CELL_PX)
        r_sift = _covered_fraction(q_r, observations.get(ref, np.empty((0, 2))), cell=CELL_PX)
        per_ref.append(
            {
                "reference": ref,
                "n_query_ref": int(len(q_r)),
                "n_mapping_mkpts": int(len(m_r)),
                "n_neighbors": len(neighbor_names),
                "neighbors": neighbor_names,
                "r_cover": r_cover,
                "r_sift": r_sift,
            }
        )
        print(json.dumps(per_ref[-1], ensure_ascii=False), flush=True)

    del runtime
    _empty_cuda()
    mean_r_cover = float(np.mean([row["r_cover"] for row in per_ref])) if per_ref else 0.0
    mean_r_sift = float(np.mean([row["r_sift"] for row in per_ref])) if per_ref else 0.0
    payload = {
        "refs": per_ref,
        "mean_r_cover": mean_r_cover,
        "mean_r_sift": mean_r_sift,
        "edm_test_res": TEST_RES,
        "edm_topk": EDM_TOPK,
        "megaloc_top_k": MEGALOC_TOP_K,
    }
    _write_json(EXPERIMENT_ROOT / "diagnose" / "cover.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if mean_r_cover < R_COVER_MIN:
        verdict = {
            "status": "INSUFFICIENT",
            "reason": "query-ref EDM mkpts are not 2px-covered by ref-ref EDM",
            "mean_r_cover": mean_r_cover,
        }
        _write_json(EXPERIMENT_ROOT / "diagnose" / "verdict.json", verdict)
        raise SystemExit("diagnose INSUFFICIENT; not running match/triangulate")
    return 0


def _npz_path(name_a: str, name_b: str) -> Path:
    return EXPERIMENT_ROOT / "matches" / f"{pair_id(name_a, name_b)}.npz"


def _inlier_path(name_a: str, name_b: str) -> Path:
    return EXPERIMENT_ROOT / "matches" / "inliers" / f"{pair_id(name_a, name_b)}.npz"


def _filter_all_pairs(reconstruction: Any, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    poses: dict[str, np.ndarray] = {}
    ks: dict[str, np.ndarray] = {}
    for image in reconstruction.images.values():
        name = str(image.name)
        poses[name] = _w2c_4x4(image)
        ks[name] = _k_from_camera(reconstruction.cameras[image.camera_id])

    inlier_root = EXPERIMENT_ROOT / "matches" / "inliers"
    if inlier_root.exists():
        shutil.rmtree(inlier_root)
    inlier_root.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    inlier_counts: list[int] = []
    for row in pairs:
        name_a = str(row["name_a"])
        name_b = str(row["name_b"])
        path = _npz_path(name_a, name_b)
        if not path.is_file():
            raise SystemExit(f"missing matches npz: {path}")
        with np.load(path, allow_pickle=False) as payload:
            mkpts0 = np.asarray(payload["mkpts0"], dtype=np.float64)
            mkpts1 = np.asarray(payload["mkpts1"], dtype=np.float64)
            mconf = np.asarray(payload["mconf"], dtype=np.float64)
        mask = frozen_pose_inliers(mkpts0, mkpts1, poses[name_a], poses[name_b], ks[name_a], ks[name_b])
        n_inliers = int(np.count_nonzero(mask))
        inlier_counts.append(n_inliers)
        if n_inliers < MIN_INLIERS:
            dropped += 1
            continue
        kept += 1
        np.savez_compressed(
            _inlier_path(name_a, name_b),
            mkpts0=np.asarray(mkpts0[mask], dtype=np.float32),
            mkpts1=np.asarray(mkpts1[mask], dtype=np.float32),
            mconf=np.asarray(mconf[mask], dtype=np.float32),
        )
    counts = np.asarray(inlier_counts, dtype=np.float64)
    summary = {
        "pairs_total": len(pairs),
        "pairs_kept": kept,
        "pairs_dropped": dropped,
        "inlier_count": {
            "p10": _percentile(counts, 10),
            "median": _percentile(counts, 50),
            "p90": _percentile(counts, 90),
        },
        "min_inliers": MIN_INLIERS,
        "sampson_max_px": SAMPSON_MAX_PX,
        "min_angle_deg": MIN_TRIANGULATION_ANGLE_DEG,
    }
    _write_json(EXPERIMENT_ROOT / "matches" / "filter_summary.json", summary)
    return summary


def cmd_match(_args: argparse.Namespace) -> int:
    require_free_gpu()
    _admit_runtime()
    _bench_gate()
    _diagnose_gate()
    import pycolmap

    pairs = _load_pairs()
    match_root = EXPERIMENT_ROOT / "matches"
    match_root.mkdir(parents=True, exist_ok=True)
    missing = [row for row in pairs if not _npz_path(str(row["name_a"]), str(row["name_b"])).is_file()]
    if missing:
        runtime = _load_edm_runtime()
        cache = _PreparedCache(runtime)
        started = time.perf_counter()
        for index, row in enumerate(missing, start=1):
            name_a = str(row["name_a"])
            name_b = str(row["name_b"])
            path = _npz_path(name_a, name_b)
            mkpts0, mkpts1, mconf = _match_cached(cache, IMAGES / name_a, IMAGES / name_b)
            np.savez_compressed(path, mkpts0=mkpts0, mkpts1=mkpts1, mconf=mconf)
            if index == 1 or index % 25 == 0 or index == len(missing):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-6)
                print(
                    f"match {index}/{len(missing)} n={len(mkpts0)} "
                    f"{elapsed:.1f}s {rate:.2f} pair/s",
                    flush=True,
                )
        del runtime
        del cache
        _empty_cuda()
    reconstruction = pycolmap.Reconstruction(str(MODEL))
    summary = _filter_all_pairs(reconstruction, pairs)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def _iter_inlier_pairs(pairs: list[dict[str, Any]]) -> Iterable[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    for row in pairs:
        name_a = str(row["name_a"])
        name_b = str(row["name_b"])
        path = _inlier_path(name_a, name_b)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as payload:
            yield (
                name_a,
                name_b,
                np.asarray(payload["mkpts0"], dtype=np.float64),
                np.asarray(payload["mkpts1"], dtype=np.float64),
                np.asarray(payload["mconf"], dtype=np.float64),
            )


def _build_database(reconstruction: Any, pairs: list[dict[str, Any]], *, keypoint_cap: int, database_path: Path) -> dict[str, Any]:
    import pycolmap
    from sfm_diagnosis.site_pipeline.densesfm_worker import open_database_compat

    pose_only, _tracks, image_ids = _pose_only_and_tracks(reconstruction)
    cells: dict[str, dict[tuple[int, int], tuple[float, float, float]]] = defaultdict(dict)
    for name_a, name_b, mkpts0, mkpts1, mconf in _iter_inlier_pairs(pairs):
        if name_a in pose_only or name_b in pose_only:
            continue
        for xy, conf in zip(mkpts0, mconf, strict=True):
            key = quantize_xy(xy)
            prev = cells[name_a].get(key)
            if prev is None or float(conf) > prev[0]:
                cells[name_a][key] = (float(conf), float(xy[0]), float(xy[1]))
        for xy, conf in zip(mkpts1, mconf, strict=True):
            key = quantize_xy(xy)
            prev = cells[name_b].get(key)
            if prev is None or float(conf) > prev[0]:
                cells[name_b][key] = (float(conf), float(xy[0]), float(xy[1]))

    discarded_cells = 0
    keypoints: dict[str, np.ndarray] = {}
    index_of: dict[str, dict[tuple[int, int], int]] = {}
    for name, grid in cells.items():
        items = list(grid.items())
        if len(items) > keypoint_cap:
            items.sort(key=lambda item: -item[1][0])
            discarded_cells += len(items) - keypoint_cap
            items = items[:keypoint_cap]
        items.sort(key=lambda item: (item[0][1], item[0][0]))
        xy = np.asarray([[row[1][1], row[1][2]] for row in items], dtype=np.float32)
        keypoints[name] = xy
        index_of[name] = {cell: i for i, (cell, _value) in enumerate(items)}

    if database_path.exists():
        database_path.unlink()
    database = open_database_compat(pycolmap.Database, database_path)
    n_match_pairs = 0
    n_match_rows = 0
    try:
        for camera_id in sorted(reconstruction.cameras):
            database.write_camera(reconstruction.cameras[camera_id], use_camera_id=True)
        for rig_id in sorted(reconstruction.rigs):
            database.write_rig(reconstruction.rigs[rig_id], use_rig_id=True)
        for frame_id in sorted(reconstruction.frames):
            database.write_frame(reconstruction.frames[frame_id], use_frame_id=True)
        for image_id in sorted(int(value) for value in reconstruction.reg_image_ids()):
            image = reconstruction.images[image_id]
            database.write_image(image, use_image_id=True)
            name = str(image.name)
            points = keypoints.get(name, np.zeros((0, 2), dtype=np.float32))
            if name in pose_only:
                points = np.zeros((0, 2), dtype=np.float32)
            database.write_keypoints(image_id, points)
        for name_a, name_b, mkpts0, mkpts1, mconf in _iter_inlier_pairs(pairs):
            if name_a in pose_only or name_b in pose_only:
                continue
            id_a = image_ids[name_a]
            id_b = image_ids[name_b]
            index_a = index_of.get(name_a, {})
            index_b = index_of.get(name_b, {})
            rows = []
            seen: set[tuple[int, int]] = set()
            for xy0, xy1 in zip(mkpts0, mkpts1, strict=True):
                ia = index_a.get(quantize_xy(xy0))
                ib = index_b.get(quantize_xy(xy1))
                if ia is None or ib is None:
                    continue
                pair = (ia, ib)
                if pair in seen:
                    continue
                seen.add(pair)
                rows.append(pair)
            if not rows:
                continue
            values = np.asarray(rows, dtype=np.uint32)
            left, right = id_a, id_b
            if left > right:
                left, right = right, left
                values = values[:, ::-1]
            database.write_matches(left, right, values)
            geometry = pycolmap.TwoViewGeometry()
            geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
            geometry.inlier_matches = values
            database.write_two_view_geometry(left, right, geometry)
            n_match_pairs += 1
            n_match_rows += int(len(values))
    finally:
        database.close()
    n_keypoints = [len(xy) for name, xy in keypoints.items() if name not in pose_only]
    counts = np.asarray(n_keypoints or [0], dtype=np.float64)
    return {
        "database": str(database_path),
        "keypoint_cap": keypoint_cap,
        "n_images_with_keypoints": len(n_keypoints),
        "n_pose_only": len(pose_only),
        "discarded_cells": discarded_cells,
        "keypoint_count": {
            "p10": _percentile(counts, 10),
            "median": _percentile(counts, 50),
            "p90": _percentile(counts, 90),
            "max": int(counts.max()) if counts.size else 0,
        },
        "n_match_pairs": n_match_pairs,
        "n_match_rows": n_match_rows,
    }


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "bad_alloc" in text or "out of memory" in text or "std::bad_alloc" in text


def _retriangulate_fixed_poses(
    *,
    input_model: Path,
    database: Path,
    image_root: Path,
    output_model: Path,
    minimum_angle_deg: float,
    ignore_two_view_tracks: bool,
) -> dict[str, Any]:
    import pycolmap
    from river_v4_optimizer.runner import analyze_model, atomic_json

    output = output_model.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    reconstruction = pycolmap.Reconstruction(str(input_model.resolve(strict=True)))
    options = pycolmap.IncrementalPipelineOptions()
    options.extract_colors = False
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.triangulation.min_angle = float(minimum_angle_deg)
    options.triangulation.ignore_two_view_tracks = bool(ignore_two_view_tracks)
    triangulation = options.triangulation
    if hasattr(triangulation, "complete_max_reproj_error"):
        triangulation.complete_max_reproj_error = 3.0
    if hasattr(triangulation, "merge_max_reproj_error"):
        triangulation.merge_max_reproj_error = 3.0
    result = pycolmap.triangulate_points(
        reconstruction,
        database.resolve(strict=True),
        image_root.resolve(strict=True),
        output,
        clear_points=True,
        options=options,
        refine_intrinsics=False,
    )
    metrics = analyze_model(output)
    receipt = {
        "schema_version": 1,
        "artifact_type": "RIVER_V4_FIXED_POSE_RETRIANGULATION",
        "input_model": str(input_model.resolve()),
        "database": str(database.resolve()),
        "image_root": str(image_root.resolve()),
        "output_model": str(output),
        "minimum_angle_deg": minimum_angle_deg,
        "ignore_two_view_tracks": ignore_two_view_tracks,
        "refine_intrinsics": False,
        "complete_max_reproj_error": getattr(triangulation, "complete_max_reproj_error", None),
        "merge_max_reproj_error": getattr(triangulation, "merge_max_reproj_error", None),
        "returned_registered_images": result.num_reg_images(),
        **metrics,
    }
    atomic_json(output.parent / "retriangulation_receipt.json", receipt)
    return receipt


def _occupancy_payload(model: Path) -> dict[str, Any]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model))
    images = list(reconstruction.images.values())
    n_images = len(images)
    n_points3d = int(reconstruction.num_points3D())
    obs_counts: list[int] = []
    records: list[tuple[str, np.ndarray, int, int]] = []
    for image in images:
        points = [point for point in image.points2D if point.has_point3D()]
        xy = np.asarray([point.xy for point in points], dtype=np.float64).reshape(-1, 2)
        camera = reconstruction.cameras[image.camera_id]
        obs_counts.append(len(xy))
        records.append((str(image.name), xy, int(camera.width), int(camera.height)))
    counts = np.asarray(obs_counts, dtype=np.float64)
    records.sort(key=lambda row: row[0])
    sampled = [records[i] for i in _stratified_indices(len(records), NN_SAMPLE_IMAGES)]
    nn_all: list[np.ndarray] = []
    occupied: dict[int, list[float]] = {size: [] for size in CELL_SIZES_PX}
    for _name, xy, width, height in sampled:
        if len(xy) > NN_MAX_POINTS:
            take = np.linspace(0, len(xy) - 1, NN_MAX_POINTS, dtype=np.int64)
            sampled_xy = xy[take]
        else:
            sampled_xy = xy
        nn_all.append(_nn_spacings(sampled_xy))
        for size in CELL_SIZES_PX:
            occupied[size].append(_occupied_fraction(xy, width, height, float(size)))
    nn = np.concatenate(nn_all) if nn_all else np.empty(0, dtype=np.float64)
    baseline = json.loads(BASELINE_SMOKE.read_text(encoding="utf-8"))
    raw_matches = int(baseline["result"]["raw_matches"])
    valid_2d3d = int(baseline["result"]["valid_2d3d"])
    return {
        "n_images": n_images,
        "n_points3D": n_points3d,
        "observation_count": {
            "p10": _percentile(counts, 10),
            "median": _percentile(counts, 50),
            "p90": _percentile(counts, 90),
        },
        "nn_spacing_px": {
            "p10": _percentile(nn, 10),
            "median": _percentile(nn, 50),
            "p90": _percentile(nn, 90),
            "n_images_sampled": len(sampled),
            "max_points_per_image": NN_MAX_POINTS,
        },
        "occupied_cell_fraction": {
            str(size): float(np.mean(occupied[size])) if occupied[size] else float("nan")
            for size in CELL_SIZES_PX
        },
        "baseline_smoke": {
            "raw_matches": raw_matches,
            "valid_2d3d": valid_2d3d,
            "r_lift": round(valid_2d3d / raw_matches, 4) if raw_matches else None,
            "source": str(BASELINE_SMOKE),
        },
    }


def cmd_triangulate(_args: argparse.Namespace) -> int:
    import pycolmap

    filter_summary = EXPERIMENT_ROOT / "matches" / "filter_summary.json"
    if not filter_summary.is_file():
        raise SystemExit("matches/filter_summary.json missing; run match first")
    _diagnose_gate()
    pairs = _load_pairs()
    output_model = EXPERIMENT_ROOT / "model"
    points_bin = output_model / "points3D.bin"
    database_path = EXPERIMENT_ROOT / "database.db"
    original = pycolmap.Reconstruction(str(MODEL))
    pose_only, _tracks, _ids = _pose_only_and_tracks(original)

    if points_bin.is_file():
        print(f"skip triangulate; {points_bin} exists", flush=True)
    else:
        db_receipt = _build_database(original, pairs, keypoint_cap=KEYPOINT_CAP, database_path=database_path)
        _write_json(EXPERIMENT_ROOT / "database_receipt.json", db_receipt)
        print(json.dumps(db_receipt, indent=2, ensure_ascii=False), flush=True)
        attempts = [
            {"ignore_two_view_tracks": False, "keypoint_cap": KEYPOINT_CAP},
            {"ignore_two_view_tracks": True, "keypoint_cap": KEYPOINT_CAP},
            {"ignore_two_view_tracks": True, "keypoint_cap": KEYPOINT_CAP_OOM},
        ]
        last_error: BaseException | None = None
        rebuilt_20k = False
        for index, attempt in enumerate(attempts):
            if attempt["keypoint_cap"] == KEYPOINT_CAP_OOM and not rebuilt_20k:
                db_receipt = _build_database(
                    original, pairs, keypoint_cap=KEYPOINT_CAP_OOM, database_path=database_path
                )
                _write_json(EXPERIMENT_ROOT / "database_receipt.json", db_receipt)
                rebuilt_20k = True
            if output_model.exists():
                shutil.rmtree(output_model)
            try:
                receipt = _retriangulate_fixed_poses(
                    input_model=MODEL,
                    database=database_path,
                    image_root=IMAGES,
                    output_model=output_model,
                    minimum_angle_deg=MIN_TRIANGULATION_ANGLE_DEG,
                    ignore_two_view_tracks=bool(attempt["ignore_two_view_tracks"]),
                )
                _write_json(EXPERIMENT_ROOT / "triangulate_receipt.json", receipt)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if output_model.exists():
                    shutil.rmtree(output_model)
                if not _is_oom(exc):
                    raise
                print(f"triangulate OOM on attempt {index}: {exc}", flush=True)
                continue
        if last_error is not None:
            payload = {
                "status": "OOM",
                "error": str(last_error),
                "attempts": attempts,
            }
            _write_json(EXPERIMENT_ROOT / "triangulate_oom.json", payload)
            raise SystemExit("triangulate OOM after retries; leaving triangulate_oom.json")

    rebuilt = pycolmap.Reconstruction(str(output_model))
    n_reg = int(rebuilt.num_reg_images())
    if n_reg != EXPECTED_N_IMAGES:
        raise SystemExit(f"triangulated num_reg_images={n_reg} != {EXPECTED_N_IMAGES}")
    gained = []
    for image in rebuilt.images.values():
        name = str(image.name)
        if name not in pose_only:
            continue
        n_obs = sum(1 for point in image.points2D if point.has_point3D())
        if n_obs:
            gained.append((name, n_obs))
    if gained:
        raise SystemExit(f"pose-only images gained 3D observations: {gained[:5]}")
    occupancy = _occupancy_payload(output_model)
    _write_json(output_model / "occupancy.json", occupancy)
    print(json.dumps(occupancy, indent=2, ensure_ascii=False), flush=True)
    frac2 = float(occupancy["occupied_cell_fraction"]["2"])
    if frac2 <= OCCUPIED_2PX_MIN:
        raise SystemExit(f"occupied 2px fraction {frac2} <= {OCCUPIED_2PX_MIN}")
    return 0


def _edm_config() -> dict[str, Any]:
    return {
        "repo": str(EDM_REPO),
        "checkpoint": str(EDM_CHECKPOINT),
        "model_config": str(RT5060_EDM_BASE),
        "data_config": str(RT5060_MEGADEPTH),
        "input_size": [TEST_RES, TEST_RES],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


def _make_query_manifest(query: Path, cache: Path) -> tuple[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    query_id = "query_" + hashlib.sha256(query.read_bytes()).hexdigest()[:16]
    query_manifest = cache / f"{query_id}.jsonl"
    query_manifest.write_text(
        json.dumps(
            {
                "query_id": query_id,
                "session_id": "DEPLOYMENT_QUERY",
                "timestamp": 0.0,
                "image_path": str(query),
                "pose_provenance": "NONE",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return query_id, query_manifest


def _localize_one(query: Path, output: Path) -> dict[str, Any]:
    _admit_runtime()
    from river_map_quality.megaloc_edm_catalog import (
        extract_megaloc_descriptors,
        load_offline_megaloc_runtime,
    )
    from sfm_diagnosis.edm_risk.edm_loo import EDMQuery
    from sfm_diagnosis.site_pipeline.deployment_localizer import (
        FinalMapEDMProvider,
        build_localization_payload,
    )
    from sfm_diagnosis.viewpoint_policy import resolve_megaloc_bank

    config = _load_localizer_config()
    cache = EXPERIMENT_ROOT / "cache"
    query_id, query_manifest = _make_query_manifest(query, cache)
    map_model = EXPERIMENT_ROOT / "model"
    provider = FinalMapEDMProvider(
        map_model=str(map_model),
        keyframes=str(KEYFRAMES),
        query_manifest=str(query_manifest),
        cache_dir=str(cache / "provider"),
        edm_config=_edm_config(),
        megaloc_source=str(config["megaloc_source"]),
        megaloc_checkpoint=str(config["megaloc_checkpoint"]),
        intrinsics_calibration=config["intrinsics"],
        top_k=MEGALOC_TOP_K,
        lift_distance_px=LIFT_DISTANCE_PX,
        descriptor_batch_size=1,
        min_reference_occupied_bins=1,
        reference_depth_dir=str(DEPTH_DIR),
    )
    provider._load_geometry()
    bank_descriptors, bank_names, bank_kind = resolve_megaloc_bank(MAP_ROOT / "localization")
    bundled_names = json.loads(Path(bank_names).read_text(encoding="utf-8"))
    descriptors = np.load(bank_descriptors, allow_pickle=False)
    provider.apply_reference_bank(bundled_names, descriptors)

    megaloc_runtime = load_offline_megaloc_runtime(
        source=Path(str(config["megaloc_source"])),
        checkpoint=Path(str(config["megaloc_checkpoint"])),
        device="cuda",
    )
    import torch

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    query_descriptor = extract_megaloc_descriptors(megaloc_runtime, [query], batch_size=1)[0]
    runtime_megaloc = time.perf_counter() - t0
    _unload_megaloc(megaloc_runtime)
    provider._query_descriptors = {query_id: query_descriptor}
    provider._prepared = True
    index = provider.build_reference_index(excluded_sessions=frozenset(), strict=False)
    result = provider.localize(
        EDMQuery(
            query_id=query_id,
            session_id="DEPLOYMENT_QUERY",
            timestamp=0.0,
            image_path=str(query),
            pose_provenance="NONE",
        ),
        index,
    )
    payload = build_localization_payload(
        query=query,
        map_name="river_gluemap_all8_direct_20260831",
        result=result,
        reference_bank=bank_kind,
    )
    peak_allocated_mib = float(torch.cuda.max_memory_allocated()) / 1048576.0
    runtime_total = float(runtime_megaloc) + float(payload["result"].get("runtime_total") or 0.0)
    extra = {
        "runtime_megaloc": runtime_megaloc,
        "runtime_edm": payload["result"].get("runtime_edm"),
        "runtime_pnp": payload["result"].get("runtime_pnp"),
        "runtime_total": runtime_total,
        "peak_allocated_mib": peak_allocated_mib,
        "estimated_5060_s": runtime_total * SCALE_5090_TO_5060,
        "edm_test_res": TEST_RES,
        "megaloc_top_k": MEGALOC_TOP_K,
    }
    payload["result"] = {**payload["result"], **extra}
    payload.update({key: extra[key] for key in extra})
    _write_json(output, payload)
    return payload


def cmd_smoke(_args: argparse.Namespace) -> int:
    require_free_gpu()
    _diagnose_gate()
    model = EXPERIMENT_ROOT / "model" / "points3D.bin"
    if not model.is_file():
        raise SystemExit("experiment model missing; run triangulate first")
    smoke_dir = EXPERIMENT_ROOT / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    payload = _localize_one(SAME_QUERY, smoke_dir / "smoke_edm_tracks.json")
    result = payload["result"]
    raw = int(result.get("raw_matches") or 0)
    valid = int(result.get("valid_2d3d") or 0)
    r_lift = (valid / raw) if raw else 0.0
    gates = {
        "status": payload.get("status"),
        "raw_matches": raw,
        "valid_2d3d": valid,
        "r_lift": r_lift,
        "ransac_inliers": result.get("ransac_inliers"),
        "reprojection_p90": result.get("reprojection_p90"),
        "runtime_megaloc": result.get("runtime_megaloc"),
        "runtime_edm": result.get("runtime_edm"),
        "runtime_pnp": result.get("runtime_pnp"),
        "runtime_total": result.get("runtime_total"),
        "peak_allocated_mib": result.get("peak_allocated_mib"),
        "estimated_5060_s": result.get("estimated_5060_s"),
    }
    print(json.dumps(gates, indent=2, ensure_ascii=False), flush=True)
    ok = (
        payload.get("status") == "LOCALIZED_STRONG"
        and r_lift >= 0.15
        and int(result.get("ransac_inliers") or 0) >= 80
        and float(result.get("reprojection_p90") or 1e9) <= 3.0
    )
    if not ok:
        _write_json(smoke_dir / "smoke_fail.json", gates)
        raise SystemExit(
            "smoke gate failed: "
            f"status={payload.get('status')} raw_matches={raw} valid_2d3d={valid} "
            f"r_lift={r_lift} ransac_inliers={result.get('ransac_inliers')} "
            f"reprojection_p90={result.get('reprojection_p90')} "
            f"runtimes megaloc={result.get('runtime_megaloc')} edm={result.get('runtime_edm')} "
            f"pnp={result.get('runtime_pnp')} total={result.get('runtime_total')}"
        )
    return 0


def _provider_class():
    """EXPERIMENTAL arm switch: flow/union temporal backends via FlowTemporalBridge."""
    backend = os.environ.get("P172_TEMPORAL_BACKEND", "klt")
    if backend in ("flow", "union", "neuflow"):
        from sfm_diagnosis.site_pipeline.flow_temporal_bridge import make_flow_provider

        return make_flow_provider()
    from sfm_diagnosis.site_pipeline.deployment_localizer import FinalMapEDMProvider

    return FinalMapEDMProvider


def _provider_extra_kwargs(Provider: Any) -> dict[str, Any]:
    if getattr(Provider, "__name__", "") == "FlowTemporalBridge":
        return {
            "temporal_backend": os.environ.get("P172_TEMPORAL_BACKEND", "flow"),
            "weak_anchor_birth": os.environ.get("P172_WEAK_ANCHOR_BIRTH", "0") == "1",
        }
    return {}


def _run_heldout(
    *,
    output_root: Path,
    reference_depth_dir: Path | None,
    temporal: bool,
    method: str,
    protocol: str,
    previous_strong: str,
) -> int:
    require_free_gpu()
    _admit_runtime()
    smoke_path = EXPERIMENT_ROOT / "smoke" / "smoke_edm_tracks.json"
    if not smoke_path.is_file():
        raise SystemExit("smoke/smoke_edm_tracks.json missing; run smoke first")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "LOCALIZED_STRONG":
        raise SystemExit("smoke did not pass LOCALIZED_STRONG; not running P172")

    from river_map_quality.megaloc_edm_catalog import (
        extract_megaloc_descriptors,
        load_offline_megaloc_runtime,
    )
    from sfm_diagnosis.edm_risk.edm_loo import EDMQuery
    from sfm_diagnosis.site_pipeline.deployment_localizer import (
        FinalMapEDMProvider,
        build_localization_payload,
    )
    from sfm_diagnosis.viewpoint_policy import resolve_megaloc_bank

    heldout_tag = os.environ.get("HELDOUT_TAG", "p172")
    heldout_session = os.environ.get("HELDOUT_SESSION", "P1720172")
    heldout_dir = os.environ.get("HELDOUT_FRAMES_DIR", "")
    frames_root = Path(heldout_dir) if heldout_dir else P172_FRAMES
    frames = sorted(frames_root.glob("frame_*.jpg"))
    only = {
        token.strip()
        for token in os.environ.get("P172_ONLY", "").split(",")
        if token.strip()
    }
    if only:
        frames = [
            path for path in frames if path.name in only or path.stem in only
        ]
    if len(frames) < (len(only) if only else 50):
        raise SystemExit(f"expected heldout {heldout_tag} frames, found {len(frames)}")
    mapping_names = set(
        json.loads((MAP_ROOT / "localization" / "megaloc_references.names.json").read_text())
    )
    overlap = [
        path.name
        for path in frames
        if f"{heldout_tag}/{path.name}" in mapping_names or path.name in mapping_names
    ]
    if overlap:
        raise SystemExit(f"{heldout_tag} frame names leak into mapping bank: {overlap[:5]}")

    config = _load_localizer_config()
    cache = output_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_root / "results.jsonl"
    if results_jsonl.is_file():
        results_jsonl.unlink()
    queries: list[tuple[str, Path]] = []
    manifest_rows = []
    for path in frames:
        query_id = f"{heldout_tag}_{path.stem}"
        queries.append((query_id, path))
        manifest_rows.append(
            json.dumps(
                {
                    "query_id": query_id,
                    "session_id": heldout_session,
                    "timestamp": 0.0,
                    "image_path": str(path),
                    "pose_provenance": "NONE",
                }
            )
        )
    query_manifest = cache / f"{heldout_tag}_queries.jsonl"
    query_manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    Provider = _provider_class()
    provider = Provider(
        map_model=str(EXPERIMENT_ROOT / "model"),
        keyframes=str(KEYFRAMES),
        query_manifest=str(query_manifest),
        cache_dir=str(cache / "provider"),
        edm_config=_edm_config(),
        megaloc_source=str(config["megaloc_source"]),
        megaloc_checkpoint=str(config["megaloc_checkpoint"]),
        intrinsics_calibration=config["intrinsics"],
        top_k=MEGALOC_TOP_K,
        lift_distance_px=LIFT_DISTANCE_PX,
        descriptor_batch_size=1,
        min_reference_occupied_bins=int(os.environ.get("P172_MIN_REF_BINS", "1")),
        reference_depth_dir=None if reference_depth_dir is None else str(reference_depth_dir),
        **_provider_extra_kwargs(Provider),
    )
    provider._load_geometry()
    retrieval = os.environ.get("P172_RETRIEVAL", "megaloc")
    if retrieval == "boq":
        import boq_runtime

        boq_bank = EXPERIMENT_ROOT / os.environ.get("BOQ_BANK_OUT", "boq_bank")
        bank_names = boq_bank / "references.names.json"
        bank_descriptors = boq_bank / "references.npy"
        if not bank_descriptors.is_file():
            raise SystemExit(f"{boq_bank} is absent; run bake_boq_bank.py first")
        # queries must use exactly the transform the bank was baked with
        receipt = json.loads((boq_bank / "bank_receipt.json").read_text(encoding="utf-8"))
        height, width = receipt["input_size_hw"]
        os.environ["P172_BOQ_RES"] = f"{width}x{height}"
        os.environ["P172_BOQ_BACKBONE"] = str(receipt["backbone"])
        bank_kind = f"boq_{receipt['backbone']}_{width}x{height}"
    else:
        bank_descriptors, bank_names, bank_kind = resolve_megaloc_bank(
            MAP_ROOT / "localization"
        )
    bundled_names = json.loads(Path(bank_names).read_text(encoding="utf-8"))
    descriptors = np.load(bank_descriptors, allow_pickle=False)
    if os.environ.get("P172_MIN_REF_BINS", "1") != "1":
        # Q_ref gate prunes low-occupancy map refs at geometry load; drop the same
        # rows from the frozen MegaLoc bank so identities stay aligned.
        keep_names = set(provider._reference_names)
        idx = [i for i, n in enumerate(bundled_names) if str(n) in keep_names]
        print(
            f"Q_ref gate: bank {len(bundled_names)} -> {len(idx)} refs "
            f"(min_bins={os.environ['P172_MIN_REF_BINS']})",
            flush=True,
        )
        bundled_names = [bundled_names[i] for i in idx]
        descriptors = descriptors[idx]
    provider.apply_reference_bank(bundled_names, descriptors)

    pending_ids = [query_id for query_id, _path in queries]
    pending_paths = [path for _query_id, path in queries]
    if retrieval == "boq":
        extracted = boq_runtime.extract_boq_descriptors(pending_paths, batch_size=8)
        boq_runtime.unload_boq()
    else:
        megaloc_runtime = load_offline_megaloc_runtime(
            source=Path(str(config["megaloc_source"])),
            checkpoint=Path(str(config["megaloc_checkpoint"])),
            device="cuda",
        )
        extracted = extract_megaloc_descriptors(megaloc_runtime, pending_paths, batch_size=1)
        _unload_megaloc(megaloc_runtime)
    provider._query_descriptors.update(dict(zip(pending_ids, extracted, strict=True)))
    provider._prepared = True
    index = provider.build_reference_index(excluded_sessions=frozenset(), strict=False)

    done: dict[str, dict] = {}
    anchor = None
    results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with results_jsonl.open("a", encoding="utf-8") as handle:
        for query_id, path in queries:
            t0 = time.perf_counter()
            result = provider.localize(
                EDMQuery(
                    query_id=query_id,
                    session_id=heldout_session,
                    timestamp=0.0,
                    image_path=str(path),
                    pose_provenance="NONE",
                ),
                index,
                **({"anchor": anchor} if temporal else {}),
            )
            payload = build_localization_payload(
                query=path,
                map_name="river_gluemap_all8_direct_20260831",
                result=result,
                reference_bank=bank_kind,
            )
            runtime_total = float(
                payload["result"].get("runtime_total") or (time.perf_counter() - t0)
            )
            source = provider.last_retrieval_source if temporal else "megaloc"
            row = {
                "query_name": path.name,
                "query_id": query_id,
                "status": payload["status"],
                "in_intersection": payload.get("in_intersection"),
                "viewpoint_abstain": payload.get("viewpoint_abstain"),
                "raw_matches": payload["result"].get("raw_matches"),
                "valid_2d3d": payload["result"].get("valid_2d3d"),
                "n_track_2d3d": payload["result"].get("n_track_2d3d"),
                "n_depth_2d3d": payload["result"].get("n_depth_2d3d"),
                "track_inliers": payload["result"].get("track_inliers"),
                "ransac_inliers": payload["result"].get("ransac_inliers"),
                "reprojection_p90": payload["result"].get("reprojection_p90"),
                "inlier_ratio": payload["result"].get("inlier_ratio"),
                "occupancy_4x4": payload["result"].get("occupancy_4x4"),
                "convex_hull_coverage": payload["result"].get("convex_hull_coverage"),
                "occupied_frac_30": payload["result"].get("occupied_frac_30"),
                "pose_consistency": payload["result"].get("pose_consistency"),
                "runtime_edm": payload["result"].get("runtime_edm"),
                "runtime_pnp": payload["result"].get("runtime_pnp"),
                "runtime_total": runtime_total,
                "estimated_5060_s": runtime_total * SCALE_5090_TO_5060,
                "retrieval_source": source,
                "n_transferred": provider.last_n_transferred if temporal else None,
                "anchor_age": None if anchor is None else int(anchor.age),
                "estimated_position": list(result.estimated_position)
                if result.estimated_position is not None
                else None,
                "estimated_R_wc": [list(row_) for row_ in result.estimated_R_wc]
                if result.estimated_R_wc is not None
                else None,
                "reference_ids": list(result.reference_ids or ()),
                "point_ids": [int(v_) for v_ in (result.point_ids or ())],
                "inlier_query_xy": [
                    [float(x), float(y)] for x, y in (result.inlier_query_xy or ())
                ],
                "inlier_confidence_mean": result.inlier_confidence_mean,
                "n_modes": len(provider.last_modes),
                "modes_top2": [
                    {
                        "inliers": int(mode.pnp_inliers),
                        "score": float(mode.absolute_score),
                        "quality_pass": bool(mode.quality_pass),
                        "refs": list(mode.reference_ids),
                        "local_conditioning": str(mode.local_conditioning),
                        "pose": [[float(x) for x in row] for row in mode.representative_pose],
                    }
                    for mode in sorted(
                        provider.last_modes,
                        key=lambda mode: int(mode.pnp_inliers),
                        reverse=True,
                    )[:2]
                ],
                "mode_decision": {
                    "status": provider.last_mode_decision.status,
                    "reasons": list(provider.last_mode_decision.reasons),
                    "failures": list(provider.last_mode_decision.failure_cases),
                    "selected": provider.last_mode_decision.selected_mode_id,
                }
                if provider.last_mode_decision is not None
                else None,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done[path.name] = row
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if temporal:
                if provider.last_track_anchor is not None:
                    anchor = provider.last_track_anchor
                elif anchor is not None:
                    anchor = anchor.next_age()

    statuses = Counter(str(row["status"]) for row in done.values())
    n = len(done)
    strong = statuses.get("LOCALIZED_STRONG", 0)
    totals = [float(row["runtime_total"] or 0.0) for row in done.values()]
    estimated = [float(row["estimated_5060_s"] or 0.0) for row in done.values()]
    sources = Counter(str(row.get("retrieval_source") or "megaloc") for row in done.values())
    summary = {
        "protocol": protocol,
        "map": "river_gluemap_all8_direct_20260831",
        "n_extracted": len(frames),
        "n_eval": n,
        "method": method,
        "reference_depth_dir": None if reference_depth_dir is None else str(reference_depth_dir),
        "lift_distance_px": LIFT_DISTANCE_PX,
        "top_k": MEGALOC_TOP_K,
        "edm_test_res": TEST_RES,
        "edm_topk": EDM_TOPK,
        "temporal": temporal,
        "validation": "NONE",
        "gt": "none_admission_gates_only",
        "status_counts": dict(statuses),
        "retrieval_source_counts": dict(sources),
        "success_rate_localized_strong": strong / n if n else 0.0,
        "localized_strong": strong,
        "previous_localized_strong": previous_strong,
        "median_runtime_total": float(np.median(totals)) if totals else None,
        "median_estimated_5060_s": float(np.median(estimated)) if estimated else None,
        "bank_kind": bank_kind,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def cmd_p172(_args: argparse.Namespace) -> int:
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172",
        reference_depth_dir=DEPTH_DIR,
        temporal=False,
        method="MegaLoc+officialEDM+PnP_track_ratio+MoGe2_depth_unproject",
        protocol="heldout_video_p1720172_edm_fixedpose_tracks",
        previous_strong=f"{P172_PREVIOUS_STRONG}/{P172_PREVIOUS_N}",
    )


def cmd_p172_tracks(_args: argparse.Namespace) -> int:
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_tracks",
        reference_depth_dir=None,
        temporal=False,
        method="MegaLoc+officialEDM+PnP_edm_tracks",
        protocol="heldout_video_p1720172_edm_tracks_only",
        previous_strong="20/124",
    )


def cmd_p172_temporal(_args: argparse.Namespace) -> int:
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_temporal",
        reference_depth_dir=None,
        temporal=True,
        method="ACCEPT_anchor_covisible_map_refs+union_PnP+MegaLoc_fallback",
        protocol="heldout_video_p1720172_temporal_map_refs",
        previous_strong="23/124",
    )


def cmd_p172_klt(_args: argparse.Namespace) -> int:
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_klt",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc+KLT_map3d_bridge+union_PnP",
        protocol="heldout_video_p1720172_klt_bridge",
        previous_strong="23/124",
    )


def cmd_p172_modes(_args: argparse.Namespace) -> int:
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_modes",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc+KLT_map3d_bridge+union_PnP+modes_logging",
        protocol="heldout_video_p1720172_modes_readjudication",
        previous_strong="24/124",
    )


def cmd_p172_dossier(_args: argparse.Namespace) -> int:
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_dossier",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc+KLT_map3d_bridge+union_PnP+forensics_logging",
        protocol="heldout_video_p1720172_dossier_logging",
        previous_strong="24/124",
    )


def cmd_p172_hedge_depth(_args: argparse.Namespace) -> int:
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_hedge_depth",
        reference_depth_dir=HEDGE_DEPTH_DIR,
        temporal=True,
        method="MegaLoc+KLT_map3d_bridge+union_PnP+hedge_depth_unproject",
        protocol="heldout_video_p1720172_hedge_depth_bridge",
        previous_strong="24/124",
    )


def cmd_p172_hr960_top10(_args: argparse.Namespace) -> int:
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_hr960_top10",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc10+officialEDM960_TOPK4096+KLT_map3d_bridge+union_PnP+pose_dump",
        protocol="heldout_video_p1720172_hr960_top10_bridge",
        previous_strong="24/124",
    )


def cmd_p172_flow(_args: argparse.Namespace) -> int:
    """Arm 1: SEA-RAFT flow direct replacement of KLT bridge."""
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "flow")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_flow",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc+SeaRaft_flow_bridge+union_PnP",
        protocol="heldout_video_p1720172_flow_bridge",
        previous_strong="24/124",
    )


def cmd_p172_union(_args: argparse.Namespace) -> int:
    """Arm 2: merged MegaLoc + flow correspondences, single PnP."""
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_union",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc+SeaRaft_flow_union_PnP",
        protocol="heldout_video_p1720172_flow_union",
        previous_strong="24/124",
    )


def cmd_p172_flow_weakbirth(_args: argparse.Namespace) -> int:
    """Arm 3: flow bridge + relaxed WEAK anchor birth (hedge continuity)."""
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "flow")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_flow_weakbirth",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc+SeaRaft_flow_bridge+weak_anchor_birth",
        protocol="heldout_video_p1720172_flow_weakbirth",
        previous_strong="24/124",
    )


def cmd_p172_union_weakbirth(_args: argparse.Namespace) -> int:
    """Arm 4: merged union PnP + relaxed WEAK anchor birth."""
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_union_weakbirth",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc+SeaRaft_flow_union_PnP+weak_anchor_birth",
        protocol="heldout_video_p1720172_flow_union_weakbirth",
        previous_strong="24/124",
    )


def cmd_p172_full(_args: argparse.Namespace) -> int:
    """Arm 5: union + weak birth + causal Fix1 sideview WEAK admission."""
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_full",
        reference_depth_dir=None,
        temporal=True,
        method="MegaLoc+SeaRaft_union+weakbirth+sideview_weak",
        protocol="heldout_video_p1720172_full_combo",
        previous_strong="24/124",
    )


def cmd_p172_full_hedge3(_args: argparse.Namespace) -> int:
    """Arm 6: full stack + MoGe3-G hedge depth lift (reference_depth_dir)."""
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_full_hedge3",
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method="MegaLoc+SeaRaft_union+weakbirth+sideview_weak+moge3G_hedge",
        protocol="heldout_video_p1720172_full_hedge3",
        previous_strong="24/124",
    )

def cmd_p172_full_init(_args: argparse.Namespace) -> int:
    """Arm 7: full stack + MoGe3 hedge + route-start init birth policy."""
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    os.environ.setdefault("P172_INIT_BIRTH_N", "8")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_full_init",
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method="MegaLoc+SeaRaft_union+weakbirth+sideview_weak+moge3G_hedge+initbirth",
        protocol="heldout_video_p1720172_full_init",
        previous_strong="24/124",
    )


def cmd_p172_tierb(_args: argparse.Namespace) -> int:
    """Arm 8: Arm 7 + Tier-B kinematic-continuity WEAK admission."""
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    os.environ.setdefault("P172_INIT_BIRTH_N", "8")
    os.environ.setdefault("P172_TIERB", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p172_tierb",
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method="MegaLoc+SeaRaft_union+weakbirth+sideview_weak+moge3G_hedge+initbirth+tierb",
        protocol="heldout_video_p1720172_tierb",
        previous_strong="24/124",
    )
def _set_heldout(tag: str, session: str, frames_subdir: str) -> None:
    import os as _os

    base = MAP_ROOT / "experiments" / "cell_anchor_dense_lift"
    _os.environ.setdefault("HELDOUT_TAG", tag)
    _os.environ.setdefault("HELDOUT_SESSION", session)
    _os.environ.setdefault("HELDOUT_FRAMES_DIR", str(base / frames_subdir / "frames"))


def cmd_p173_full_hedge3(_args: argparse.Namespace) -> int:
    """P173 second-route validation: best stack (union+weakbirth+sideview+moge3 hedge)."""
    _set_heldout("p173", "P1730173", "p173_heldout")
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p173_full_hedge3",
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method="MegaLoc+SeaRaft_union+weakbirth+sideview_weak+moge3G_hedge",
        protocol="heldout_video_p1730173_full_hedge3",
        previous_strong="none_first_run",
    )


def cmd_p174_full_hedge3(_args: argparse.Namespace) -> int:
    """P174 second-route validation: best stack (union+weakbirth+sideview+moge3 hedge)."""
    _set_heldout("p174", "P1740174", "p174_heldout")
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p174_full_hedge3",
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method="MegaLoc+SeaRaft_union+weakbirth+sideview_weak+moge3G_hedge",
        protocol="heldout_video_p1740174_full_hedge3",
        previous_strong="none_first_run",
    )

def cmd_p173_tierb(_args: argparse.Namespace) -> int:
    """P173 held-out route under Arm 8 (Arm 6 + init birth + Tier-B kinematic)."""
    _set_heldout("p173", "P1730173", "p173_heldout")
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    os.environ.setdefault("P172_INIT_BIRTH_N", "8")
    os.environ.setdefault("P172_TIERB", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p173_tierb",
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method="MegaLoc+SeaRaft_union+weakbirth+sideview_weak+moge3G_hedge+initbirth+tierb",
        protocol="heldout_video_p1730173_tierb",
        previous_strong="36/122",
    )


def cmd_p174_tierb(_args: argparse.Namespace) -> int:
    """P174 held-out route under Arm 8 (Arm 6 + init birth + Tier-B kinematic)."""
    _set_heldout("p174", "P1740174", "p174_heldout")
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    os.environ.setdefault("P172_INIT_BIRTH_N", "8")
    os.environ.setdefault("P172_TIERB", "1")
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / "p174_tierb",
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method="MegaLoc+SeaRaft_union+weakbirth+sideview_weak+moge3G_hedge+initbirth+tierb",
        protocol="heldout_video_p1740174_tierb",
        previous_strong="46/76",
    )


def _arm8_env() -> None:
    os.environ.setdefault("P172_TEMPORAL_BACKEND", "union")
    os.environ.setdefault("P172_WEAK_ANCHOR_BIRTH", "1")
    os.environ.setdefault("P172_SIDEVIEW_WEAK", "1")
    os.environ.setdefault("P172_INIT_BIRTH_N", "8")
    os.environ.setdefault("P172_TIERB", "1")


def _loco_env() -> None:
    """E1: LocoTrack point tracker replaces SEA-RAFT in the anchor transfer."""
    _arm8_env()
    os.environ.setdefault("P172_POINT_TRACKER", "locotrack")
    os.environ.setdefault("P172_TRACK_RES", "512x512")
    os.environ.setdefault("P172_TRACK_CHAIN", "6")
    os.environ.setdefault("P172_LOCOTRACK_SIZE", "base")


def _loco_method() -> str:
    return (
        "MegaLoc+LocoTrack_union+weakbirth+sideview_weak+moge3G_hedge+initbirth+tierb"
        f"[res={os.environ.get('P172_TRACK_RES')},chain={os.environ.get('P172_TRACK_CHAIN')}]"
    )


def cmd_p172_loco(_args: argparse.Namespace) -> int:
    """E1 on the P172 dev route: Arm 8 with the LocoTrack anchor transfer."""
    _loco_env()
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / os.environ.get("LOCO_OUT", "p172_loco"),
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method=_loco_method(),
        protocol="heldout_video_p1720172_loco",
        previous_strong="37/124",
    )


def cmd_p173_loco(_args: argparse.Namespace) -> int:
    """E1 on the P173 held-out route (the map-hole route)."""
    _set_heldout("p173", "P1730173", "p173_heldout")
    _loco_env()
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / os.environ.get("LOCO_OUT", "p173_loco"),
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method=_loco_method(),
        protocol="heldout_video_p1730173_loco",
        previous_strong="36/122",
    )


def cmd_p174_loco(_args: argparse.Namespace) -> int:
    """E1 on the P174 held-out route."""
    _set_heldout("p174", "P1740174", "p174_heldout")
    _loco_env()
    return _run_heldout(
        output_root=EXPERIMENT_ROOT / os.environ.get("LOCO_OUT", "p174_loco"),
        reference_depth_dir=EXPERIMENT_ROOT / "depth_moge3",
        temporal=True,
        method=_loco_method(),
        protocol="heldout_video_p1740174_loco",
        previous_strong="46/76",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-root",
        type=Path,
        default=DEFAULT_MAP_ROOT,
        help=f"map root holding model/, keyframes/, localization/ (default: {DEFAULT_MAP_ROOT})",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help=(
            "where pairs/matches/database/model receipts are written "
            f"(default: <map-root>/{DEFAULT_EXPERIMENT_SUBDIR})"
        ),
    )
    parser.add_argument(
        "--edm-repo",
        type=Path,
        default=DEFAULT_EDM_REPO,
        help=f"EDM repository root (default: {DEFAULT_EDM_REPO})",
    )
    parser.add_argument(
        "--edm-checkpoint",
        type=Path,
        default=DEFAULT_EDM_CHECKPOINT,
        help=f"EDM outdoor checkpoint (default: {DEFAULT_EDM_CHECKPOINT})",
    )
    parser.add_argument(
        "--megaloc-source",
        type=Path,
        default=DEFAULT_MEGALOC_SOURCE,
        help=f"MegaLoc source tree (default: {DEFAULT_MEGALOC_SOURCE})",
    )
    parser.add_argument(
        "--megaloc-checkpoint",
        type=Path,
        default=_default_megaloc_checkpoint(),
        help=(
            "MegaLoc safetensors checkpoint "
            f"(default: first match of ~/.cache/huggingface/hub/{MEGALOC_CHECKPOINT_GLOB})"
        ),
    )
    parser.add_argument(
        "--baseline-smoke",
        type=Path,
        default=None,
        help=(
            "pre-retriangulation smoke_localization.json used for the occupancy baseline "
            "(default: <experiment-root>/records/smoke_localization.json)"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    sub.add_parser("pairs")
    sub.add_parser("bench")
    sub.add_parser("diagnose")
    sub.add_parser("match")
    sub.add_parser("triangulate")
    sub.add_parser("smoke")
    sub.add_parser("p172")
    sub.add_parser("p172_tracks")
    sub.add_parser("p172_temporal")
    sub.add_parser("p172_klt")
    sub.add_parser("p172_hr960_top10")
    sub.add_parser("p172_flow")
    sub.add_parser("p172_union")
    sub.add_parser("p172_flow_weakbirth")
    sub.add_parser("p172_union_weakbirth")
    sub.add_parser("p172_full")
    sub.add_parser("p172_full_hedge3")
    sub.add_parser("p172_full_init")
    sub.add_parser("p172_tierb")
    sub.add_parser("p173_full_hedge3")
    sub.add_parser("p174_full_hedge3")
    sub.add_parser("p173_tierb")
    sub.add_parser("p174_tierb")
    sub.add_parser("p172_loco")
    sub.add_parser("p173_loco")
    sub.add_parser("p174_loco")
    sub.add_parser("p172_dossier")
    sub.add_parser("p172_modes")
    args = parser.parse_args()
    _apply_roots(
        map_root=args.map_root,
        experiment_root=args.experiment_root,
        edm_repo=args.edm_repo,
        edm_checkpoint=args.edm_checkpoint,
        megaloc_source=args.megaloc_source,
        megaloc_checkpoint=args.megaloc_checkpoint,
        baseline_smoke=args.baseline_smoke,
    )
    commands = {
        "selftest": cmd_selftest,
        "pairs": cmd_pairs,
        "bench": cmd_bench,
        "diagnose": cmd_diagnose,
        "match": cmd_match,
        "triangulate": cmd_triangulate,
        "smoke": cmd_smoke,
        "p172": cmd_p172,
        "p172_tracks": cmd_p172_tracks,
        "p172_temporal": cmd_p172_temporal,
        "p172_klt": cmd_p172_klt,
        "p172_hr960_top10": cmd_p172_hr960_top10,
        "p172_flow": cmd_p172_flow,
        "p172_union": cmd_p172_union,
        "p172_flow_weakbirth": cmd_p172_flow_weakbirth,
        "p172_union_weakbirth": cmd_p172_union_weakbirth,
        "p172_full": cmd_p172_full,
        "p172_full_hedge3": cmd_p172_full_hedge3,
        "p172_full_init": cmd_p172_full_init,
        "p172_tierb": cmd_p172_tierb,
        "p173_full_hedge3": cmd_p173_full_hedge3,
        "p174_full_hedge3": cmd_p174_full_hedge3,
        "p173_tierb": cmd_p173_tierb,
        "p174_tierb": cmd_p174_tierb,
        "p172_loco": cmd_p172_loco,
        "p173_loco": cmd_p173_loco,
        "p174_loco": cmd_p174_loco,
        "p172_dossier": cmd_p172_dossier,
        "p172_modes": cmd_p172_modes,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

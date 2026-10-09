"""Online, single-frame relocalizer built on the frozen offline EDM provider.

``FinalMapEDMProvider`` is a vendored *offline batch* localizer:
it wants a query manifest on disk, pre-extracts VPR descriptors for the
whole query set, and reads every image back from a path.  Deployment needs the
opposite shape -- one in-memory frame at a time, arriving from a camera.

``LiveMapEDMProvider`` subclasses it and adapts these stages:

* the descriptor stage: the reference bank is loaded from the release's frozen
  ``.npy`` (BoQ-ResNet50, 16384-D); query descriptors are extracted from the
  live array with :class:`BoQExtractor` instead of a file set;
* the keyframe index: rebuilt from :class:`DirectMapAssets` so a release stays
  relocatable, with a per-image SHA-256 check the first time a reference is
  prepared;
* the geometry stage: a bundle-verified archive can replace COLMAP parsing
  with equivalent arrays; older releases retain the original loader;
* the entry point: :meth:`localize_array` runs retrieval -> EDM match/lift ->
  PnP in the same order as the vendored ``localize()``, minus the KLT bridge
  (the deployed fast loop owns tracking).

Every matching, lifting, PnP and admission decision stays inside vendor code.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from boq_extractor import BOQ_DIM, BoQExtractor
from direct_map import DirectMapAssets
from direct_paths import sha256_file, vendor_sys_path
from direct_profile import DirectProfile
from edm_inference import CachedEDM
from workmap_reference import (
    OverlapQuery,
    index_reference_points,
    parse_overlap_reference,
    reference_camera,
    reference_image_size,
    reference_integer,
    reference_names,
    require_reference_fields,
    unique_overlap_pairs,
)

vendor_sys_path()

from sfm_diagnosis.edm_risk.edm_loo import EDMQuery  # noqa: E402
from sfm_diagnosis.site_pipeline.deployment_localizer import (  # noqa: E402
    FinalMapEDMProvider,
    _canonical_sha256,
    _configure_audited_runtime_site_packages,
    rank_reference_indices,
)


# The retriangulated river map was matched at 640x640 with COARSE.TOPK 2240
# (`P172_TEST_RES` / `P172_EDM_TOPK` in the mapping driver).  The transfer
# package shipped the driver but not the two yacs override files it merges, so
# they are regenerated here verbatim from those frozen constants and the
# packaged `configs/edm/outdoor/edm_base.py`.  `_load_edm_runtime` refuses to
# continue unless the loaded runtime really reports these numbers.
EDM_TEST_RES = 640
EDM_COARSE_TOPK = 2240
EDM_CONFIDENCE_THRESHOLD = 0.0

_EDM_BASE_CONFIG_SOURCE = f"""from src.config.default import _CN as cfg

cfg.EDM.COARSE.MCONF_THR = 0.05
cfg.EDM.FINE.SIGMA_THR = 1e-6
cfg.EDM.COARSE.BORDER_RM = 0
cfg.EDM.TEST_RES_H = {EDM_TEST_RES}
cfg.EDM.TEST_RES_W = {EDM_TEST_RES}
cfg.EDM.COARSE.TOPK = {EDM_COARSE_TOPK}
# RoPE is rescaled from the training resolution (832x832) to the test
# resolution; without this the transformer asserts (npe is None) and the
# positional encoding would be wrong.  Same rule as the production
# edm_matcher.py NPE block.
cfg.EDM.NECK.NPE = [cfg.EDM.TRAIN_RES_H, cfg.EDM.TRAIN_RES_W, {EDM_TEST_RES}, {EDM_TEST_RES}]
"""
_EDM_DATA_CONFIG_SOURCE = """from src.config.default import _CN as cfg

cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DF = 8
"""
# (BoQ preprocessing, including ImageNet normalisation, lives in
# ``boq_extractor`` so the in-memory path cannot drift from the bank build.)

_PLACEHOLDER_QUERY_ID = "live_frame"
_PLACEHOLDER_SESSION_ID = "LIVE_STREAM"


@dataclass(frozen=True)
class RelocFix:
    """One relocalization attempt, in the pixel frame of the submitted array."""

    ok: bool
    status: str
    cam_from_world: np.ndarray | None
    query_xy: np.ndarray
    point3d_ids: np.ndarray
    point_xyz: np.ndarray
    inliers: int
    inlier_ratio: float
    reproj_p90: float | None
    reference_names: tuple[str, ...]
    runtime_ms: float
    reason: str = ""
    stage_ms: dict[str, float] | None = None
    reference_scope: str = "global"
    workmap_snapshot: dict[str, Any] | None = None

    @classmethod
    def abstained(cls, reason: str, runtime_ms: float) -> "RelocFix":
        return cls(
            ok=False,
            status="ABSTAINED",
            cam_from_world=None,
            query_xy=np.zeros((0, 2), dtype=np.float64),
            point3d_ids=np.zeros((0,), dtype=np.int64),
            point_xyz=np.zeros((0, 3), dtype=np.float64),
            inliers=0,
            inlier_ratio=0.0,
            reproj_p90=None,
            reference_names=(),
            runtime_ms=runtime_ms,
            reason=reason,
        )


def clone_reloc_fix(fix: RelocFix) -> RelocFix:
    """Return an independent copy of RelocFix with copied numpy arrays to prevent aliasing."""
    return replace(
        fix,
        cam_from_world=None if fix.cam_from_world is None else fix.cam_from_world.copy(),
        query_xy=fix.query_xy.copy(),
        point3d_ids=fix.point3d_ids.copy(),
        point_xyz=fix.point_xyz.copy(),
        stage_ms=None if fix.stage_ms is None else dict(fix.stage_ms),
        workmap_snapshot=None if fix.workmap_snapshot is None else copy.deepcopy(fix.workmap_snapshot),
    )

def boq_descriptor_from_array(extractor: BoQExtractor, image: np.ndarray) -> np.ndarray:
    """Extract one 16,384-D BoQ descriptor from an in-memory RGB or grayscale array."""
    if image.ndim == 2:
        import torch
        import torch.nn.functional as F
        from boq_extractor import _IMAGENET_MEAN, _IMAGENET_STD

        device = torch.device(extractor.device)
        arr = image if image.flags.c_contiguous else np.ascontiguousarray(image)
        source = torch.from_numpy(arr)
        if device.type == "cuda":
            source = source.pin_memory()
        x = source.to(device, non_blocking=True)
        x = x.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
        x = F.interpolate(
            x,
            size=(extractor.input_size, extractor.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).repeat(1, 3, 1, 1)
        normalization = extractor._normalization_tensors
        if normalization is None or normalization[0].device != device:
            mean = torch.tensor(list(_IMAGENET_MEAN), dtype=torch.float32, device=device).view(
                1, 3, 1, 1
            )
            std = torch.tensor(list(_IMAGENET_STD), dtype=torch.float32, device=device).view(
                1, 3, 1, 1
            )
            normalization = (mean, std)
            extractor._normalization_tensors = normalization
        x = ((x - normalization[0]) / normalization[1]).contiguous()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=extractor.fp16):
                output = extractor.model(x)
            desc = F.normalize(output.float(), dim=1, eps=1e-12)[0]
            descriptor = desc.cpu().numpy().astype(np.float32, copy=False)
    else:
        descriptor = extractor.extract_from_array(image)

    if descriptor.shape != (BOQ_DIM,):
        raise RuntimeError(
            f"BoQ descriptor shape {descriptor.shape!r} violates the {BOQ_DIM}-D contract"
        )
    if not np.isfinite(descriptor).all():
        raise RuntimeError("BoQ emitted a non-finite descriptor")
    return descriptor


def _reciprocal_edm_matches(
    forward: Mapping[str, Any],
    reverse: Mapping[str, Any],
    *,
    tolerance_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-to-one forward rows confirmed by the reverse EDM pass."""
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("reciprocal match tolerance must be finite and positive")

    def arrays(result: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points0 = np.asarray(result["mkpts0_f"], dtype=np.float64).reshape(-1, 2)
        points1 = np.asarray(result["mkpts1_f"], dtype=np.float64).reshape(-1, 2)
        confidence = np.asarray(result["mconf"], dtype=np.float64).reshape(-1)
        if len(points0) != len(points1) or len(points0) != len(confidence):
            raise ValueError("EDM correspondence arrays have different lengths")
        return points0, points1, confidence

    forward0, forward1, forward_conf = arrays(forward)
    reverse0, reverse1, reverse_conf = arrays(reverse)
    if not len(forward0) or not len(reverse0):
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)

    from scipy.spatial import cKDTree

    # Candidate pairs agree in both images within the tolerance. The KD-tree
    # only narrows the search; the inclusive squared-distance tests below are
    # the admission rule.
    radius_sq = tolerance_px * tolerance_px
    neighbours = cKDTree(reverse1).query_ball_point(forward0, r=tolerance_px * (1.0 + 1e-12))
    counts = np.fromiter(
        (len(items) for items in neighbours), dtype=np.int64, count=len(neighbours)
    )
    if not counts.sum():
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    forward_index = np.repeat(np.arange(len(forward0), dtype=np.int64), counts)
    reverse_index = np.fromiter(
        (item for items in neighbours for item in items), dtype=np.int64, count=int(counts.sum())
    )
    delta0 = forward0[forward_index] - reverse1[reverse_index]
    delta1 = forward1[forward_index] - reverse0[reverse_index]
    distance0 = np.einsum("ij,ij->i", delta0, delta0)
    distance1 = np.einsum("ij,ij->i", delta1, delta1)
    admitted = (distance0 <= radius_sq) & (distance1 <= radius_sq)
    forward_index, reverse_index = forward_index[admitted], reverse_index[admitted]
    total = distance0[admitted] + distance1[admitted]
    confidence = np.minimum(forward_conf[forward_index], reverse_conf[reverse_index])
    # Best reverse row per forward row: smallest (distance, -confidence, index).
    order = np.lexsort((reverse_index, -confidence, total, forward_index))
    first = np.ones(len(order), dtype=bool)
    first[1:] = forward_index[order][1:] != forward_index[order][:-1]
    best = order[first]
    # Greedy one-to-one assignment over the per-forward winners, best first.
    ranked = best[
        np.lexsort((reverse_index[best], forward_index[best], -confidence[best], total[best]))
    ]
    used_forward: set[int] = set()
    used_reverse: set[int] = set()
    forward_rows: list[int] = []
    confidences: list[float] = []
    for row in ranked:
        forward_row, reverse_row = int(forward_index[row]), int(reverse_index[row])
        if forward_row in used_forward or reverse_row in used_reverse:
            continue
        used_forward.add(forward_row)
        used_reverse.add(reverse_row)
        forward_rows.append(forward_row)
        confidences.append(float(confidence[row]))
    return (
        np.asarray(forward_rows, dtype=np.int64),
        np.asarray(confidences, dtype=np.float64),
    )


def _strict_two_view_inliers(
    points0: np.ndarray,
    points1: np.ndarray,
    *,
    threshold_px: float = 3.0,
) -> np.ndarray:
    """Require an actual robust fundamental-matrix fit before admitting overlap."""
    import cv2

    first = np.asarray(points0, dtype=np.float32).reshape(-1, 2)
    second = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    rejected = np.zeros(len(first), dtype=bool)
    if len(first) < 8 or len(first) != len(second):
        return rejected
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    fundamental, mask = cv2.findFundamentalMat(
        first, second, method, float(threshold_px), 0.999, 10000
    )
    if fundamental is None or mask is None or np.asarray(fundamental).shape != (3, 3):
        return rejected
    inliers = np.asarray(mask, dtype=bool).reshape(-1)
    if len(inliers) != len(first):
        return rejected
    minimum = max(6, int(np.ceil(0.25 * len(first))))
    return inliers if int(inliers.sum()) >= minimum else rejected


def _overlap_pair(
    query: OverlapQuery,
    old_name: str,
    *,
    new_row: int,
    old_id: int,
    old_xyz: np.ndarray,
    old_xy: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    new_id = int(query.new_ids[new_row])
    return {
        "new_id": new_id,
        "old_id": old_id,
        "new_xyz": query.new_xyz[new_row].tolist(),
        "old_xyz": np.asarray(old_xyz).tolist(),
        "new_xy": query.new_xy[new_row].tolist(),
        "old_xy": np.asarray(old_xy).tolist(),
        "old_reference_name": old_name,
        "confidence": confidence,
        "keyframe_name": query.keyframe_name,
        "keyframe_revision": query.keyframe_revision,
        "pose_revision": query.pose_revision,
        "landmark_revisions": {new_id: int(query.landmark_revisions[new_id])},
        "keyframe_pose": query.pose.tolist(),
        "keyframe_K": query.camera_matrix.tolist(),
        "image_size": [query.width, query.height],
    }


def _restore_session_ids(matches: Any, record: Mapping[str, Any]) -> list[Any]:
    restored = []
    for match in matches:
        row = int(match.point3d_id) - 1
        if row < 0 or row >= len(record["ids"]):
            raise RuntimeError("workmap lift returned an invalid local row ID")
        restored.append(
            replace(
                match,
                point3d_id=int(record["ids"][row]),
                xyz=tuple(float(value) for value in record["xyz"][row]),
            )
        )
    return restored


class LiveMapEDMProvider(FinalMapEDMProvider):
    """Single-frame, in-memory relocalizer against one frozen direct release."""

    def __init__(
        self,
        assets: DirectMapAssets,
        profile: DirectProfile,
        *,
        edm_repo: Path,
        edm_checkpoint: Path,
        boq_weights: Path,
        cache_dir: Path,
    ) -> None:
        self.assets = assets
        self.profile = profile
        cache = Path(cache_dir).expanduser().absolute()
        cache.mkdir(parents=True, exist_ok=True)
        keyframes = assets.keyframe_index()
        weights = Path(boq_weights).expanduser().resolve(strict=True)
        self.boq_weights = weights

        # The vendored provider reads these two switches from the environment.
        # The profile, not the shell, decides them: bind them before any vendor
        # code observes them so the SHA-bound profile stays authoritative.
        os.environ["EDM_BATCH_REFS"] = "1" if profile.reloc.batch_refs else "0"
        os.environ["EDM_REF_CACHE_SIZE"] = str(profile.reloc.reference_cache_size)

        super().__init__(
            map_model=str(assets.model_dir),
            keyframes=str(assets.keyframes_manifest),
            keyframe_index=keyframes,
            query_manifest=str(self._write_placeholder_manifest(cache, keyframes)),
            cache_dir=str(cache),
            edm_config=self._edm_config(cache, edm_repo, edm_checkpoint),
            # Vendored ``FinalMapEDMProvider`` still names these slots
            # ``megaloc_source``/``megaloc_checkpoint`` and hashes the checkpoint
            # into its fingerprint.  Its MegaLoc loader is dead code on the live
            # path (``_load_descriptors`` is overridden below to install the
            # frozen bank), so both slots point at the BoQ weights file: the
            # ``"megaloc_checkpoint"`` fingerprint entry keeps varying with the
            # weights even though the field name is frozen by vendor code.
            megaloc_source=str(weights),
            megaloc_checkpoint=str(weights),
            intrinsics_calibration=profile.intrinsics.as_calibration(),
            top_k=profile.reloc.top_k,
            lift_distance_px=profile.reloc.lift_distance_px,
            thresholds=profile.reloc.frozen_pnp_thresholds,
            descriptor_batch_size=profile.reloc.descriptor_batch_size,
            intersection_cells_path=(
                None if assets.intersection_cells is None else str(assets.intersection_cells)
            ),
            min_reference_occupied_bins=profile.reloc.min_reference_occupied_bins,
            reference_depth_dir=(
                None if assets.reference_depth_dir is None else str(assets.reference_depth_dir)
            ),
        )
        self._vpr_runtime: BoQExtractor | None = None
        self._index: Any | None = None
        self._models_ready = False
        self._lock = threading.Lock()
        self._gpu_matcher: CachedEDM | None = None
        self._query_matches: dict[int, dict] = {}
        self._query_lifts: dict[str, tuple[list, int]] = {}
        self._last_strong_refs: tuple[int, ...] = ()
        self._recovery_attempted: set[int] = set()
        self.last_stage_ms: dict[str, float] = {}
        self._reference_norms: np.ndarray | None = None
        self.startup_stage_ms: dict[str, float] = {}
        self._workmap_session_id: str | None = None
        self._workmap_references: dict[str, dict[str, Any]] = {}
        self._workmap_order: list[str] = []
        # landmark ID -> registered references observing it, so covisibility
        # ranking does not intersect against every session reference.
        self._workmap_landmark_refs: dict[int, set[str]] = {}
        self._workmap_descriptors = np.empty((0, BOQ_DIM), dtype=np.float32)
        self._workmap_descriptor_count = 0
        self._workmap_descriptor_capacity = 0
        self._last_successful_workmap_reference: str | None = None

    # -- construction helpers ---------------------------------------------

    @staticmethod
    def _write_placeholder_manifest(cache: Path, keyframes: dict[str, dict]) -> Path:
        """One-row query manifest so the vendored ``__init__`` stays satisfied.

        ``_query_index`` insists on a non-empty manifest whose images exist, and
        the provider fingerprint hashes them.  Live localization never reads a
        query from disk, so the row points at a reference image that is already
        part of the release rather than at an invented file.
        """

        name = next(iter(sorted(keyframes)))
        manifest = cache / "live_placeholder_queries.jsonl"
        row = {
            "query_id": _PLACEHOLDER_QUERY_ID,
            "session_id": _PLACEHOLDER_SESSION_ID,
            "timestamp": 0.0,
            "image_path": str(keyframes[name]["image_uri"]),
            "pose_provenance": "NONE",
        }
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return manifest

    @staticmethod
    def _edm_config(cache: Path, edm_repo: Path, edm_checkpoint: Path) -> dict:
        config_dir = cache / "edm_runtime_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        base = config_dir / "edm_base_640.py"
        data = config_dir / "megadepth_640.py"
        for path, source in ((base, _EDM_BASE_CONFIG_SOURCE), (data, _EDM_DATA_CONFIG_SOURCE)):
            if not path.is_file() or path.read_text(encoding="utf-8") != source:
                path.write_text(source, encoding="utf-8")
        return {
            "repo": str(Path(edm_repo).resolve(strict=True)),
            "checkpoint": str(Path(edm_checkpoint).resolve(strict=True)),
            "model_config": str(base),
            "data_config": str(data),
            "input_size": [EDM_TEST_RES, EDM_TEST_RES],
            "confidence_threshold": EDM_CONFIDENCE_THRESHOLD,
        }

    def _compute_fingerprint(self) -> str:
        """Live identity from verified bundle/profile plus weight/config content."""
        edm_checkpoint = Path(str(self.edm_config["checkpoint"]))
        model_config = Path(str(self.edm_config["model_config"]))
        data_config = Path(str(self.edm_config["data_config"]))
        return _canonical_sha256(
            {
                "implementation": "LIVE_DIRECT_FROZEN_BANK_V1",
                "bundle": str(self.assets.sha256),
                "profile": str(self.profile.sha256),
                "edm_checkpoint": sha256_file(edm_checkpoint),
                "boq_checkpoint": sha256_file(self.boq_weights),
                "model_config_sha256": sha256_file(model_config),
                "data_config_sha256": sha256_file(data_config),
                "input_size": list(self.edm_config.get("input_size", [])),
                "confidence_threshold": self.edm_config.get("confidence_threshold"),
            }
        )

    # -- vendor overrides --------------------------------------------------

    def _load_geometry(self) -> None:
        if self._geometry_locked and self._reference_names:
            return
        from direct_geometry import GEOMETRY_ARCHIVE, load_geometry_archive

        if any(entry["path"] == GEOMETRY_ARCHIVE for entry in self.assets.raw["files"]):
            load_geometry_archive(
                self, self.assets.root / GEOMETRY_ARCHIVE, self.assets.raw["model_sha256"]
            )
        else:
            super()._load_geometry()

    def _load_descriptors(self) -> None:
        """Install the frozen reference bank; never extract a query catalog."""

        names = self.assets.ref_names
        descriptors = np.load(self.assets.bank_descriptors, allow_pickle=False)
        self.apply_reference_bank(names, descriptors)
        self._query_descriptors = {}

    def _get_prepared_reference(self, name: str, path: Path, runtime: Any):
        """Verify the reference JPEG against the manifest before it is used."""

        self.assets.verify_reference_image(name)
        return super()._get_prepared_reference(name, path, runtime)

    # -- runtime -----------------------------------------------------------

    def _matcher_runtime(self) -> Any:
        runtime = super()._matcher_runtime()
        if int(runtime.input_width) != EDM_TEST_RES or int(runtime.input_height) != EDM_TEST_RES:
            raise RuntimeError(
                f"EDM configs did not merge: input {runtime.input_width}x{runtime.input_height}"
                f" (expected {EDM_TEST_RES} square)"
            )
        if int(runtime.coarse_topk) != EDM_COARSE_TOPK:
            raise RuntimeError(
                f"EDM COARSE.TOPK did not merge: {runtime.coarse_topk} (expected {EDM_COARSE_TOPK})"
            )
        return runtime

    def _vpr(self) -> BoQExtractor:
        # BoQ is self-contained (torch/torchvision/numpy/cv2 only): no audited
        # site-packages dance, no vendor loader.  fp16 defaults on for CUDA
        # (``SFM_BOQ_FP16`` may switch it off); the extractor SHA-pins the
        # weights at construction.
        if self._vpr_runtime is None:
            self._vpr_runtime = BoQExtractor(device="cuda", weights_path=self.boq_weights)
        return self._vpr_runtime

    def ensure_models(self) -> None:
        """Load geometry, bank, EDM and BoQ, then warm the GPU kernels."""
        if self._models_ready:
            return
        stage_ms: dict[str, float] = {}
        mark = time.perf_counter()
        _configure_audited_runtime_site_packages(self.audited_edm_site_packages)
        self._prepare()
        stage_ms["geometry_bank_ms"] = (time.perf_counter() - mark) * 1000.0
        mark = time.perf_counter()
        self._matcher_runtime()
        stage_ms["edm_load_ms"] = (time.perf_counter() - mark) * 1000.0
        mark = time.perf_counter()
        self._vpr()
        stage_ms["boq_load_ms"] = (time.perf_counter() - mark) * 1000.0
        mark = time.perf_counter()
        self._index = self.build_reference_index(excluded_sessions=frozenset(), strict=False)
        self._reference_norms = np.linalg.norm(self._reference_descriptors, axis=1)
        stage_ms["reference_index_ms"] = (time.perf_counter() - mark) * 1000.0
        mark = time.perf_counter()
        self._warm_models()
        stage_ms["kernel_warmup_ms"] = (time.perf_counter() - mark) * 1000.0
        self.startup_stage_ms = dict(stage_ms)
        self._models_ready = True

    def _warm_models(self) -> None:
        """Run one real forward pass per network without touching the map anchor."""
        width, height = self.profile.fast_loop.resolution
        rng = np.random.default_rng(20260916)
        texture = rng.integers(0, 256, size=(int(height), int(width)), dtype=np.uint8)
        texture = np.ascontiguousarray(texture)
        # BoQ sees one real descriptor forward on textured input.
        boq_descriptor_from_array(self._vpr(), texture)
        from river_map_quality.official_edm_adapter import (
            prepare_official_megadepth_image_from_array,
        )

        runtime = self._matcher_runtime()
        prepared = prepare_official_megadepth_image_from_array(texture, runtime)
        # EDM runs the production matcher branch: batched when the profile
        # enables it, single-reference otherwise. A temporary GPU cache keeps
        # the warmup off the production reference cache.
        saved_matcher = self._gpu_matcher
        saved_matches = self._query_matches
        self._query_matches = {}
        self._gpu_matcher = None
        try:
            if bool(self.profile.reloc.batch_refs):
                self._match_prepared_batch(
                    runtime, prepared, [prepared] * int(self.profile.reloc.top_k)
                )
            else:
                self._match_prepared(runtime, prepared, prepared)
        finally:
            self._query_matches = saved_matches
            self._gpu_matcher = saved_matcher
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (AttributeError, RuntimeError, ValueError):
            pass

    @property
    def ref_names(self) -> tuple[str, ...]:
        return tuple(self._reference_names)

    def frozen_point_xyz(self) -> np.ndarray:
        """Read-only view of the frozen site-map points loaded by ensure_models()."""
        if not self._models_ready:
            raise RuntimeError("frozen points are available only after ensure_models()")
        points = self._point_xyz.view()
        points.setflags(write=False)
        return points

    @property
    def session_reference_count(self) -> int:
        """Number of durable session references with a ready CPU BoQ descriptor."""
        return self._workmap_descriptor_count

    def index_workmap_reference(self, reference: Mapping[str, Any]) -> str:
        """Atomically admit one durable workmap keyframe to the session catalog.

        The caller is the existing background relocalization/GPU worker.  The
        image and its two-view landmark geometry must already be durable before
        this synchronous method is called.  The frozen global bank and all of
        its geometry/descriptor arrays are deliberately untouched.
        """
        require_reference_fields(reference, "workmap reference")
        source_name, session_id = reference_names(
            reference, "workmap reference name and session_id must be non-empty"
        )
        revision = reference_integer(reference["revision"], "workmap", "revision")
        pose_revision = reference_integer(reference["pose_revision"], "workmap", "pose_revision")
        source_epoch = reference_integer(reference["source_epoch"], "workmap", "source_epoch")
        ordinal = reference_integer(reference["ordinal"], "workmap", "ordinal")
        stamp = float(reference["stamp"])
        if not np.isfinite(stamp):
            raise ValueError("workmap stamp must be finite")
        image_path = Path(str(reference["image_path"])).expanduser().resolve(strict=True)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        width, height = reference_image_size(reference, "workmap")
        # Session retrieval is intentionally restricted to session landmarks.
        # Positive IDs remain the immutable release's responsibility and must
        # never enter a workmap-only PnP candidate.
        ids, xy, xyz, revisions = index_reference_points(
            reference, width, height, max(6, int(self.profile.reloc.min_points))
        )
        pose, camera_matrix = reference_camera(reference, "workmap", "cam_from_world matrix")
        record = {
            "source_name": source_name,
            "image_path": image_path,
            "ids": ids,
            "xy": xy,
            "xyz": xyz,
            "pose": np.ascontiguousarray(pose, dtype=np.float64).copy(),
            "K": np.ascontiguousarray(camera_matrix, dtype=np.float64).copy(),
            "revision": revision,
            "session_id": session_id,
            "source_epoch": source_epoch,
            "landmark_revisions": revisions,
            "pose_revision": pose_revision,
            "ordinal": ordinal,
            "stamp": stamp,
            "image_size": (width, height),
        }
        for field in ("ids", "xy", "xyz", "pose", "K"):
            record[field].flags.writeable = False
        with self._lock:
            if not self._models_ready:
                raise RuntimeError("ensure_models() must run before indexing workmap references")
            if self._workmap_session_id is not None and self._workmap_session_id != session_id:
                raise ValueError("one provider cannot index references from multiple sessions")
            registered_name = f"workmap:{session_id}:{source_name}"
            record["name"] = registered_name
            prior = self._workmap_references.get(registered_name)
            if prior is not None:
                self._update_workmap_reference_locked(record, prior)
            else:
                self._add_workmap_reference_locked(record)
            return registered_name

    def _update_workmap_reference_locked(
        self, record: dict[str, Any], prior: Mapping[str, Any]
    ) -> None:
        """A BA revision reuses the stored descriptor and measured image."""
        if prior["image_path"] != record["image_path"]:
            raise ValueError("a workmap keyframe name cannot be rebound to another image")
        if record["image_size"] != prior["image_size"]:
            raise ValueError("a workmap keyframe cannot change image_size")
        unchanged = all(
            record[key] == prior[key]
            for key in (
                "revision",
                "pose_revision",
                "source_epoch",
                "ordinal",
                "stamp",
                "image_size",
                "landmark_revisions",
            )
        ) and all(
            np.array_equal(record[key], prior[key]) for key in ("ids", "xy", "xyz", "pose", "K")
        )
        if unchanged:
            return
        if record["revision"] <= prior["revision"]:
            raise ValueError("workmap keyframe updates require a newer revision")
        record["descriptor_index"] = prior["descriptor_index"]
        self._workmap_references[record["name"]] = record
        self._reindex_workmap_landmarks(record["name"], prior["ids"], record["ids"])

    def _add_workmap_reference_locked(self, record: dict[str, Any]) -> None:
        import cv2

        # Only a new keyframe needs image decode and BoQ inference.
        width, height = record["image_size"]
        gray = cv2.imread(str(record["image_path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"workmap image cannot be decoded: {record['image_path']}")
        if gray.shape[1] != width or gray.shape[0] != height:
            raise ValueError("workmap image dimensions disagree with image_size")
        descriptor = boq_descriptor_from_array(self._vpr(), gray)
        if descriptor.shape != (BOQ_DIM,):
            raise RuntimeError("workmap BoQ descriptor violates the catalog dimension")
        row = self._workmap_descriptor_count
        capacity = self._workmap_descriptor_capacity
        descriptors = self._workmap_descriptors
        if row == capacity:
            capacity = max(4, capacity * 2)
            expanded = np.empty((capacity, BOQ_DIM), dtype=np.float32)
            expanded[:row] = descriptors[:row]
            descriptors = expanded
        descriptors[row] = descriptor
        record["descriptor_index"] = row
        # Publish catalog metadata and descriptor row together only after
        # image decode, validation, and BoQ inference have all succeeded.
        self._workmap_session_id = record["session_id"]
        self._workmap_descriptors = descriptors
        self._workmap_descriptor_capacity = capacity
        self._workmap_references[record["name"]] = record
        self._workmap_order.append(record["name"])
        self._reindex_workmap_landmarks(record["name"], (), record["ids"])
        self._workmap_descriptor_count = row + 1

    def _reindex_workmap_landmarks(self, name: str, old_ids, new_ids) -> None:
        for point_id in old_ids:
            refs = self._workmap_landmark_refs.get(int(point_id))
            if refs is not None:
                refs.discard(name)
        for point_id in new_ids:
            self._workmap_landmark_refs.setdefault(int(point_id), set()).add(name)

    def match_workmap_overlap(self, reference: Mapping[str, Any]) -> dict[str, Any]:
        """Match one durable session KF against a bounded frozen-map BoQ shortlist.

        Both EDM directions must agree, and the reciprocal matches must pass a
        robust two-view fit before either image is lifted through its measured
        observations. This path only reports evidence; it never changes either
        map or performs an XYZ-nearest association.
        """
        query = parse_overlap_reference(reference)
        import cv2

        color_bgr = cv2.imread(str(query.image_path), cv2.IMREAD_COLOR)
        if color_bgr is None or color_bgr.shape[:2] != (query.height, query.width):
            raise ValueError("workmap overlap image cannot be decoded at its declared size")
        result: dict[str, Any] = {
            "session_id": query.session_id,
            "source_epoch": query.source_epoch,
            "keyframe_name": query.keyframe_name,
            "keyframe_revision": query.keyframe_revision,
            "pose_revision": query.pose_revision,
            "landmark_revisions": query.landmark_revisions,
            "keyframe_pose": query.pose.tolist(),
            "keyframe_K": query.camera_matrix.tolist(),
            "image_size": [query.width, query.height],
            "pairs": [],
        }
        if len(query.new_ids) == 0:
            return result
        with self._lock:
            if not self._models_ready or self._index is None:
                raise RuntimeError("ensure_models() must run before workmap overlap matching")
            if (
                self._workmap_session_id is not None
                and self._workmap_session_id != query.session_id
            ):
                raise ValueError("one provider cannot match overlap for multiple sessions")
            ranked = self._overlap_shortlist(color_bgr)
            if not ranked:
                return result
            candidates = self._overlap_candidates(color_bgr, ranked, query)
        result["pairs"] = unique_overlap_pairs(candidates)
        return result

    def _overlap_shortlist(self, color_bgr: np.ndarray) -> tuple[int, ...]:
        import cv2

        descriptor = boq_descriptor_from_array(
            self._vpr(), cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        )
        subset = self._subsets[self._index.index_id]
        top_k = int(self.profile.reloc.top_k)
        ranked, _fallback = rank_reference_indices(
            self._reference_descriptors,
            descriptor,
            reference_sessions=self._reference_sessions,
            excluded_sessions=subset.excluded_sessions,
            top_k=top_k,
            occupied_bins=self._reference_occupied_bins,
            min_occupied_bins=self.min_reference_occupied_bins,
            retrieve_pool=None,
            reference_norms=self._reference_norms,
        )
        allowed = set(subset.indices)
        return tuple(index for index in ranked if index in allowed)[:top_k]

    def _overlap_candidates(
        self, color_bgr: np.ndarray, ranked: tuple[int, ...], query: OverlapQuery
    ) -> dict[tuple[int, int], dict[str, Any]]:
        import cv2
        from river_map_quality.official_edm_adapter import (
            prepare_official_megadepth_image_from_array,
        )

        runtime = self._matcher_runtime()
        prepared_query = prepare_official_megadepth_image_from_array(
            cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY), runtime
        )
        lift_distance = float(self.profile.reloc.lift_distance_px)
        if not np.isfinite(lift_distance) or lift_distance <= 0:
            raise ValueError("workmap overlap lift distance must be finite and positive")
        candidates: dict[tuple[int, int], dict[str, Any]] = {}
        self._query_matches.clear()
        try:
            for reference_index in ranked:
                for pair in self._overlap_pairs_for_reference(
                    runtime, prepared_query, reference_index, query, lift_distance
                ):
                    identity = (pair["new_id"], pair["old_id"])
                    previous = candidates.get(identity)
                    if previous is None or pair["confidence"] > previous["confidence"]:
                        candidates[identity] = pair
        finally:
            self._query_matches.clear()
        return candidates

    def _reciprocal_reference_matches(
        self,
        runtime,
        prepared_query,
        reference_index: int,
        query: OverlapQuery,
        tolerance_px: float,
    ):
        """Forward/reverse EDM rows that agree and fit one fundamental matrix."""
        from sfm_diagnosis.site_pipeline.deployment_localizer import _valid_matches

        old_name = self._reference_names[reference_index]
        prepared_reference, reference_shape = self._get_prepared_reference(
            old_name, self._reference_paths[reference_index], runtime
        )
        threshold = float(self.edm_config.get("confidence_threshold") or 0.0)
        directions = []
        for first, second, shape0, shape1 in (
            (prepared_query, prepared_reference, (query.height, query.width), reference_shape),
            (prepared_reference, prepared_query, reference_shape, (query.height, query.width)),
        ):
            raw = self._match_prepared(runtime, first, second)
            points0 = np.asarray(raw["mkpts0_f"], dtype=np.float64).reshape(-1, 2)
            points1 = np.asarray(raw["mkpts1_f"], dtype=np.float64).reshape(-1, 2)
            confidence = np.asarray(raw["mconf"], dtype=np.float64).reshape(-1)
            valid = _valid_matches(
                points0,
                points1,
                confidence,
                query_shape=shape0,
                reference_shape=shape1,
                confidence_threshold=threshold,
            )
            directions.append(
                {"mkpts0_f": points0[valid], "mkpts1_f": points1[valid], "mconf": confidence[valid]}
            )
        forward, reverse = directions
        rows, confidences = _reciprocal_edm_matches(forward, reverse, tolerance_px=tolerance_px)
        if len(rows) < 8:
            return None
        new_xy, old_xy = forward["mkpts0_f"][rows], forward["mkpts1_f"][rows]
        geometric = _strict_two_view_inliers(new_xy, old_xy)
        if not geometric.any():
            return None
        return new_xy[geometric], old_xy[geometric], confidences[geometric]

    def _unambiguous_observations(self, old_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        observation_xy, observation_ids = self._observations[old_name]
        observation_xy = np.asarray(observation_xy, dtype=np.float64).reshape(-1, 2)
        observation_ids = np.asarray(observation_ids, dtype=np.int64).reshape(-1)
        positive = observation_ids > 0
        observation_xy, observation_ids = observation_xy[positive], observation_ids[positive]
        if len(observation_xy) != len(observation_ids) or not np.isfinite(observation_xy).all():
            raise RuntimeError(f"frozen reference {old_name!r} has invalid measured observations")
        # Some immutable COLMAP frames contain multiple 2D rows for one
        # positive 3D ID. Neither row proves which pixel should be paired
        # with the EDM feature: discard that ID.
        unique_ids, counts = np.unique(observation_ids, return_counts=True)
        unambiguous = np.isin(observation_ids, unique_ids[counts == 1])
        if np.count_nonzero(unambiguous) < 8:
            return None
        return observation_xy[unambiguous], observation_ids[unambiguous]

    def _overlap_pairs_for_reference(
        self,
        runtime,
        prepared_query,
        reference_index: int,
        query: OverlapQuery,
        lift_distance: float,
    ) -> list[dict[str, Any]]:
        from river_map_quality.official_edm_adapter import _nearest_observation_within

        old_name = self._reference_names[reference_index]
        matched = self._reciprocal_reference_matches(
            runtime, prepared_query, reference_index, query, min(2.0, lift_distance)
        )
        observations = None if matched is None else self._unambiguous_observations(old_name)
        if matched is None or observations is None:
            return []
        edm_new_xy, edm_old_xy, confidences = matched
        old_xy, old_ids = observations
        new_rows, _new_distances, new_lifted = _nearest_observation_within(
            edm_new_xy,
            query.new_xy,
            np.arange(1, len(query.new_ids) + 1, dtype=np.int64),
            lift_distance,
        )
        old_rows, _old_distances, old_lifted = _nearest_observation_within(
            edm_old_xy, old_xy, np.arange(1, len(old_ids) + 1, dtype=np.int64), lift_distance
        )
        lifted_new = np.zeros(len(edm_new_xy), dtype=np.int64)
        lifted_old = np.zeros(len(edm_old_xy), dtype=np.int64)
        lifted_new[new_lifted] = new_rows
        lifted_old[old_lifted] = old_rows
        accepted = np.flatnonzero(new_lifted & old_lifted)
        if not len(accepted):
            return []
        accepted_old_ids = np.asarray(
            [old_ids[int(lifted_old[row]) - 1] for row in accepted], dtype=np.int64
        )
        accepted_old_xyz = self._point_xyz_for_ids(accepted_old_ids)
        if not np.isfinite(accepted_old_xyz).all():
            raise RuntimeError(f"frozen reference {old_name!r} contains non-finite 3D landmarks")
        inliers = _strict_two_view_inliers(
            query.new_xy[lifted_new[accepted] - 1], old_xy[lifted_old[accepted] - 1]
        )
        return [
            _overlap_pair(
                query,
                old_name,
                new_row=int(lifted_new[edm_row]) - 1,
                old_id=int(old_id),
                old_xyz=old_xyz,
                old_xy=old_xy[int(lifted_old[edm_row]) - 1],
                confidence=float(confidences[edm_row]),
            )
            for edm_row, old_id, old_xyz in zip(
                accepted[inliers], accepted_old_ids[inliers], accepted_old_xyz[inliers], strict=True
            )
        ]

    # -- localization ------------------------------------------------------

    def _match_prepared(self, runtime, query, reference):
        from river_map_quality.official_edm_adapter import match_official_prepared

        key = id(reference)
        if key not in self._query_matches:
            started = time.perf_counter()
            capacity = self.profile.optimizations.gpu_cache_size
            if capacity:
                if self._gpu_matcher is None:
                    self._gpu_matcher = CachedEDM(runtime, capacity)
                matched = self._gpu_matcher.match(query, reference)
            else:
                matched = match_official_prepared(runtime, query, reference)
            self._query_matches[key] = (reference, matched)
            self.last_stage_ms["edm_match_ms"] = (
                self.last_stage_ms.get("edm_match_ms", 0.0)
                + (time.perf_counter() - started) * 1000.0
            )
        return self._query_matches[key][1]

    def _match_prepared_batch(self, runtime, query, references):
        from river_map_quality.official_edm_adapter import match_official_prepared_batch

        missing = [
            reference for reference in references if id(reference) not in self._query_matches
        ]
        if missing:
            started = time.perf_counter()
            matched = match_official_prepared_batch(runtime, query, missing)
            self._query_matches.update(
                (id(ref), (ref, result)) for ref, result in zip(missing, matched, strict=True)
            )
            self.last_stage_ms["edm_match_ms"] = (
                self.last_stage_ms.get("edm_match_ms", 0.0)
                + (time.perf_counter() - started) * 1000.0
            )
        return [self._query_matches[id(reference)][1] for reference in references]

    def _rank_workmap_references(self, descriptor: np.ndarray) -> tuple[str, ...]:
        count = self._workmap_descriptor_count
        if count == 0:
            return ()
        scores = self._workmap_descriptors[:count] @ np.asarray(
            descriptor, dtype=np.float32
        ).reshape(BOQ_DIM)
        boq_order = np.argsort(-scores, kind="stable")[: self.top_k]
        names = [self._workmap_order[int(index)] for index in boq_order]
        if not names:
            return ()

        # Keep the full-session BoQ winner even when the last successful area
        # has many covisible references. Fill the remaining bounded batch with
        # that area's true covisibility neighborhood, then BoQ fallbacks.
        selected = [names[0]]
        previous = self._last_successful_workmap_reference
        prioritized = (
            [previous, *self._covisible_workmap_references(previous, scores)]
            if previous in self._workmap_references
            else []
        )
        for name in (*prioritized, *names[1:]):
            if name not in selected and len(selected) < self.top_k:
                selected.append(name)
        return tuple(selected)

    def _covisible_workmap_references(self, previous: str, scores: np.ndarray) -> list[str]:
        """References sharing session points with ``previous``, most shared first."""
        shared_counts: dict[str, int] = {}
        for point_id in self._workmap_references[previous]["ids"]:
            for name in self._workmap_landmark_refs.get(int(point_id), ()):
                if name != previous:
                    shared_counts[name] = shared_counts.get(name, 0) + 1
        neighbors = []
        for name, shared in shared_counts.items():
            index = int(self._workmap_references[name]["descriptor_index"])
            neighbors.append((shared, float(scores[index]), index, name))
        neighbors.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [name for _, _, _, name in neighbors]

    def _solve_workmap_references(
        self,
        image: np.ndarray,
        prepared_query: Any,
        reference_names: tuple[str, ...],
        started: float,
    ) -> RelocFix:
        match_started = time.perf_counter()

        if not reference_names:
            return RelocFix.abstained("workmap retrieval returned no references", 0.0)

        from sfm_diagnosis.site_pipeline.deployment_localizer import (
            _pose_fields,
            evaluate_localization_admission,
        )

        lifted, raw_matches = self._lift_workmap_matches(image, prepared_query, reference_names)
        self.last_stage_ms["workmap_valid_matches"] = float(raw_matches)
        self.last_stage_ms["workmap_lifted_matches"] = float(len(lifted))
        query_path = Path(self._queries[_PLACEHOLDER_QUERY_ID]["image_path"])
        self.last_retrieval_source = "workmap_boq"
        self.last_n_transferred = 0
        self.last_track_anchor = None
        self.last_modes = ()
        self.last_mode_decision = None
        match_elapsed = time.perf_counter() - match_started
        pnp_started = time.perf_counter()
        solved, metrics, decision_status = self._solve(
            query_path,
            lifted,
            session_groups=False,
            query_image=image,
        )
        pnp_elapsed = time.perf_counter() - pnp_started
        self.last_stage_ms["workmap_match_lift_ms"] = max(0.0, match_elapsed * 1000.0)
        self.last_stage_ms["workmap_pnp_ms"] = pnp_elapsed * 1000.0

        center = None if solved is None else _pose_fields(solved.pose)[0]
        accepted, _global_status, _, _ = evaluate_localization_admission(
            registration_success=solved is not None,
            metrics=metrics,
            decision_status=decision_status,
            position=center,
            intersection_cells=self._intersection_cells,
            requested_thresholds=self.thresholds,
        )
        reproj_p90 = (
            None if metrics.get("reprojection_p90") is None else float(metrics["reprojection_p90"])
        )
        if not accepted or solved is None:
            return RelocFix(
                ok=False,
                status="ABSTAINED",
                cam_from_world=(
                    None
                    if solved is None
                    else np.asarray(solved.pose, dtype=np.float64)[:3, :4].copy()
                ),
                query_xy=np.zeros((0, 2), dtype=np.float64),
                point3d_ids=np.zeros((0,), dtype=np.int64),
                point_xyz=np.zeros((0, 3), dtype=np.float64),
                inliers=int(metrics.get("inlier_count") or 0),
                inlier_ratio=float(metrics.get("inlier_ratio") or 0.0),
                reproj_p90=reproj_p90,
                reference_names=(),
                runtime_ms=(time.perf_counter() - started) * 1000.0,
                reason=f"workmap {decision_status}: geometric admission failed",
                reference_scope="workmap",
            )
        return self._workmap_fix(solved, metrics, lifted, reference_names, started, reproj_p90)

    def _lift_workmap_matches(
        self, image: np.ndarray, prepared_query: Any, reference_names: tuple[str, ...]
    ) -> tuple[tuple[Any, ...], int]:
        """EDM matches to session references, lifted onto their session points."""
        from river_map_quality.official_edm_adapter import (
            deduplicate_lifted_matches,
            lift_reference_matches,
        )
        from sfm_diagnosis.site_pipeline.deployment_localizer import (
            _two_view_inliers,
            _valid_matches,
        )

        runtime = self._matcher_runtime()
        prepared_references = []
        reference_shapes: dict[str, tuple[int, int]] = {}
        for name in reference_names:
            prepared, shape = self._prepared_reference_cache.get_or_prepare(
                f"session-workmap:{name}", self._workmap_references[name]["image_path"], runtime
            )
            prepared_references.append(prepared)
            reference_shapes[name] = shape
        if bool(self.profile.reloc.batch_refs) and len(prepared_references) > 1:
            matched_list = self._match_prepared_batch(runtime, prepared_query, prepared_references)
        else:
            matched_list = [
                self._match_prepared(runtime, prepared_query, reference)
                for reference in prepared_references
            ]
        confidence_threshold = float(self.edm_config.get("confidence_threshold") or 0.0)
        lifted = []
        raw_matches = 0
        for name, matched in zip(reference_names, matched_list, strict=True):
            record = self._workmap_references[name]
            query_points = np.asarray(matched["mkpts0_f"], dtype=np.float64).reshape(-1, 2)
            reference_points = np.asarray(matched["mkpts1_f"], dtype=np.float64).reshape(-1, 2)
            confidences = np.asarray(matched["mconf"], dtype=np.float64).reshape(-1)
            valid = _valid_matches(
                query_points,
                reference_points,
                confidences,
                query_shape=image.shape,
                reference_shape=reference_shapes[name],
                confidence_threshold=confidence_threshold,
            )
            query_points, reference_points = query_points[valid], reference_points[valid]
            confidences = confidences[valid]
            geometry = _two_view_inliers(query_points, reference_points, threshold_px=3.0)
            query_points, reference_points = query_points[geometry], reference_points[geometry]
            confidences = confidences[geometry]
            raw_matches += len(query_points)
            # The vendored lift intentionally rejects negative IDs.  Give it a
            # dense, positive, per-reference row namespace, then restore the
            # session ID and immutable XYZ on every accepted correspondence.
            lifted_rows = lift_reference_matches(
                query_points=query_points,
                reference_points=reference_points,
                observation_points=record["xy"],
                observation_point3d_ids=np.arange(1, len(record["ids"]) + 1, dtype=np.int64),
                confidences=confidences,
                maximum_distance_px=self.lift_distance_px,
                reference_name=name,
            )
            lifted.extend(_restore_session_ids(lifted_rows.matches, record))
        deduplicated = deduplicate_lifted_matches(lifted, query_conflict_distance_px=1.0)
        return tuple(deduplicated.matches), raw_matches

    def _workmap_fix(
        self,
        solved: Any,
        metrics: Mapping[str, Any],
        lifted: tuple[Any, ...],
        reference_names: tuple[str, ...],
        started: float,
        reproj_p90: float | None,
    ) -> RelocFix:
        inlier_mask = np.asarray(solved.inlier_mask, dtype=bool)
        if inlier_mask.shape != (len(lifted),):
            raise RuntimeError("workmap PnP returned a misaligned inlier mask")
        inlier_matches = [match for match, keep in zip(lifted, inlier_mask, strict=True) if keep]
        if not inlier_matches or any(
            match.point3d_id is None or int(match.point3d_id) >= 0 or match.xyz is None
            for match in inlier_matches
        ):
            raise RuntimeError("workmap PnP inliers must carry negative IDs and XYZ")
        reference_order = {name: index for index, name in enumerate(reference_names)}
        snapshot, counts = self._workmap_snapshot(inlier_matches, reference_order)
        self._last_successful_workmap_reference = snapshot["reference_name"]
        return RelocFix(
            ok=True,
            status="VO_ONLY",
            cam_from_world=np.asarray(solved.pose, dtype=np.float64)[:3, :4].copy(),
            query_xy=np.ascontiguousarray(
                np.asarray([match.query_xy for match in inlier_matches], dtype=np.float64)
            ),
            point3d_ids=np.ascontiguousarray(
                np.asarray([int(match.point3d_id) for match in inlier_matches], dtype=np.int64)
            ),
            point_xyz=np.ascontiguousarray(
                np.asarray([match.xyz for match in inlier_matches], dtype=np.float64)
            ),
            inliers=int(np.count_nonzero(inlier_mask)),
            inlier_ratio=float(metrics.get("inlier_ratio") or 0.0),
            reproj_p90=reproj_p90,
            reference_names=tuple(sorted(counts, key=lambda name: reference_order[name])),
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            reason="session workmap visual localization; weak by reference scope",
            reference_scope="workmap",
            workmap_snapshot=snapshot,
        )

    def _workmap_snapshot(
        self, inlier_matches: list[Any], reference_order: Mapping[str, int]
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Revisions the fast loop must re-validate before trusting this fix."""
        counts: dict[str, int] = {}
        landmark_revisions: dict[int, int] = {}
        keyframe_revisions: dict[str, int] = {}
        for match in inlier_matches:
            name = str(match.reference_name)
            record = self._workmap_references[name]
            counts[name] = counts.get(name, 0) + 1
            point_id = int(match.point3d_id)
            point_revision = int(record["landmark_revisions"][point_id])
            if landmark_revisions.setdefault(point_id, point_revision) != point_revision:
                raise RuntimeError("workmap PnP mixed revisions of one landmark")
            keyframe_revisions[name] = int(record["pose_revision"])
        anchor_name = min(counts, key=lambda name: (-counts[name], reference_order[name]))
        anchor = self._workmap_references[anchor_name]
        snapshot = {
            "session_id": str(anchor["session_id"]),
            "revision": max(int(self._workmap_references[name]["revision"]) for name in counts),
            "keyframe_revision": int(anchor["pose_revision"]),
            "keyframe_revisions": keyframe_revisions,
            "landmark_revisions": landmark_revisions,
            "reference_name": anchor_name,
        }
        return snapshot, counts

    def _lift_matched_reference(self, query_shape, name, reference_shape, matched, threshold):
        # Cache before greedy cross-reference deduplication. Filtering an already
        # deduplicated baseline would discard alternatives needed by expansion.
        # Names identify immutable reference geometry; matching/shape/threshold
        # are fixed within this locked capture. Clear on every capture and close.
        if name not in self._query_lifts:
            self._query_lifts[name] = super()._lift_matched_reference(
                query_shape, name, reference_shape, matched, threshold
            )
        return self._query_lifts[name]

    def _has_usable_anchor(self, fix: RelocFix) -> bool:
        return fix.ok and len(fix.point3d_ids) >= int(
            self.profile.reloc.frozen_pnp_thresholds["strong_inliers"]
        )

    def _recovery_references(self, ranked: tuple[int, ...]) -> tuple[int, ...]:
        """Search beyond failed near-duplicate views, retaining a global winner.

        Only the additional matching budget rotates. The baseline pair is
        always tried at the current ranking, so a changed camera view can
        immediately recover without waiting for the exploratory cycle.
        """
        extra = [index for index in ranked[self.top_k :] if index not in self._recovery_attempted]
        if not extra:
            self._recovery_attempted.clear()
            extra = list(ranked[self.top_k :])
        if len(extra) < 2 or not self._last_strong_refs:
            return tuple(extra[: self.top_k])
        seeds = np.unique(
            np.concatenate(
                [
                    self._observations[self._reference_names[index]][1]
                    for index in self._last_strong_refs
                ]
            )
        )
        # Covisibility may reorder only the nearest bounded alternatives, not
        # drag an outbound prior into a distant return-facing retrieval rank.
        nearest = extra[: self.top_k * 3]
        overlap = [
            len(np.intersect1d(seeds, self._observations[self._reference_names[index]][1]))
            for index in nearest[1:]
        ]
        neighbor = nearest[1 + int(np.argmax(overlap))]
        return tuple(dict.fromkeys([extra[0], neighbor, *extra]))[: self.top_k]

    def localize_array(
        self,
        gray: np.ndarray,
        *,
        color_bgr: np.ndarray | None = None,
        scope: str = "all",
        on_global_fix: Any = None,
    ) -> RelocFix:
        """Relocalize one live frame.  Never raises; failures become ABSTAINED.

        ``gray`` is the EDM input.  ``color_bgr`` — when the caller still holds
        the colour frame — is what BoQ sees; the frozen reference bank was
        extracted from colour keyframes, so passing it keeps retrieval on the
        distribution the bank was built from.  Without it the grey frame is
        replicated across the three channels.
        """

        started = time.perf_counter()
        try:
            with self._lock:
                return self._localize_array(
                    gray, color_bgr, scope=scope, on_global_fix=on_global_fix
                )
        except Exception as error:  # deployment contract: the caller keeps flying
            return RelocFix.abstained(
                f"{type(error).__name__}: {error}",
                (time.perf_counter() - started) * 1000.0,
            )

    def _localize_array(
        self,
        gray: np.ndarray,
        color_bgr: np.ndarray | None,
        *,
        scope: str = "all",
        on_global_fix: Any = None,
    ) -> RelocFix:
        import cv2

        if scope not in {"all", "global", "workmap"}:
            raise ValueError("localization scope must be all, global, or workmap")
        if scope == "workmap" and not self._workmap_descriptor_count:
            return replace(
                RelocFix.abstained("no indexed workmap references", 0.0),
                reference_scope="workmap",
            )
        self.last_stage_ms = {}
        if self._index is None:
            raise RuntimeError("ensure_models() must run before localize_array()")
        started = time.perf_counter()
        self._query_matches.clear()
        self._query_lifts.clear()
        image = np.ascontiguousarray(gray)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("live query must be an HxW uint8 grayscale array")
        if color_bgr is None:
            descriptor = boq_descriptor_from_array(self._vpr(), image)
        else:
            rgb = cv2.cvtColor(np.ascontiguousarray(color_bgr), cv2.COLOR_BGR2RGB)
            descriptor = boq_descriptor_from_array(self._vpr(), rgb)
        if scope == "workmap":
            # Auxiliary-only queries cannot change the baseline's adaptive
            # frozen-map retrieval history.
            fix = self._workmap_relocalization(image, None, descriptor, started)
            return replace(
                fix,
                runtime_ms=(time.perf_counter() - started) * 1000.0,
                stage_ms=dict(self.last_stage_ms),
            )

        global_fix, prepared = self._global_relocalization(image, descriptor, started)
        self.last_stage_ms["lift_prepare_ms"] = max(
            0.0,
            self.last_stage_ms.get("match_lift_ms", 0.0)
            - self.last_stage_ms.get("edm_match_ms", 0.0),
        )
        packed_global = replace(
            global_fix,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            stage_ms=dict(self.last_stage_ms),
        )
        if callable(on_global_fix):
            on_global_fix(packed_global)

        if scope == "global":
            return packed_global

        # Global evidence always wins. A session-only fix is never mixed into
        # the frozen bank or allowed to replace a successful global result.
        fix = global_fix
        if not global_fix.ok and self._workmap_descriptor_count:
            if time.perf_counter() - started >= float(self.profile.reloc.period_s):
                # The same per-attempt budget as adaptive global retrieval: a
                # slow global attempt must not also queue a session batch ahead
                # of the next relocalization while the fast loop has no anchor.
                self.last_stage_ms["workmap_skipped_budget"] = 1.0
            else:
                fix = self._workmap_relocalization(image, prepared, descriptor, started)
        return replace(
            fix,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            stage_ms=dict(self.last_stage_ms),
        )

    def _global_relocalization(
        self, image: np.ndarray, descriptor: np.ndarray, started: float
    ) -> tuple[RelocFix, Any]:
        """Frozen-map retrieval, EDM and PnP; returns the fix and prepared query."""
        subset = self._subsets[self._index.index_id]
        ranked, viewpoint_pool_fallback = rank_reference_indices(
            self._reference_descriptors,
            descriptor,
            reference_sessions=self._reference_sessions,
            excluded_sessions=subset.excluded_sessions,
            # Ranking is CPU-only. Explore a wider global shortlist over
            # failed attempts, but match at most one extra top_k per capture.
            top_k=self.top_k * (12 if self.profile.optimizations.adaptive_retrieval else 1),
            occupied_bins=self._reference_occupied_bins,
            min_occupied_bins=self.min_reference_occupied_bins,
            retrieve_pool=None,
            reference_norms=self._reference_norms,
        )
        allowed = set(subset.indices)
        ranked = tuple(index for index in ranked if index in allowed)
        self.last_stage_ms["retrieval_ms"] = (time.perf_counter() - started) * 1000.0
        if not ranked:
            self._last_strong_refs = ()
            return RelocFix.abstained(
                "retrieval returned no admissible reference",
                (time.perf_counter() - started) * 1000.0,
            ), None
        from river_map_quality.official_edm_adapter import (
            prepare_official_megadepth_image_from_array,
        )

        base = ranked[: self.top_k]
        matched_refs = base
        prepared = prepare_official_megadepth_image_from_array(image, self._matcher_runtime())
        global_fix = self._solve_references(image, prepared, base, started, viewpoint_pool_fallback)
        self.last_stage_ms["matched_references"] = float(len(base))
        # Depth-assisted strong poses may have no admissible sparse-map seed.
        # Keep their decision, but continue bounded recovery for a usable anchor.
        # Adaptive global recovery runs before the workmap gets its one later batch.
        if (
            not self._has_usable_anchor(global_fix)
            and self.profile.optimizations.adaptive_retrieval
            and time.perf_counter() - started < self.profile.reloc.period_s
        ):
            extra = self._recovery_references(ranked)
            if extra:
                refs = base + extra
                expanded = self._solve_references(
                    image, prepared, refs, started, viewpoint_pool_fallback
                )
                self._recovery_attempted.update(extra)
                matched_refs = refs
                self.last_stage_ms["matched_references"] = float(len(refs))
                if expanded.ok and (not global_fix.ok or self._has_usable_anchor(expanded)):
                    global_fix = expanded
                    base = refs
        if self._has_usable_anchor(global_fix):
            self._last_strong_refs = base
            self._recovery_attempted.clear()
        else:
            self._last_strong_refs = ()
        return (
            replace(
                global_fix,
                reference_names=tuple(self._reference_names[index] for index in matched_refs),
            ),
            prepared,
        )

    def _workmap_relocalization(
        self, image: np.ndarray, prepared: Any, descriptor: np.ndarray, started: float
    ) -> RelocFix:
        self.last_track_anchor = None
        self.last_n_transferred = 0
        workmap_started = time.perf_counter()
        try:
            if prepared is None:
                from river_map_quality.official_edm_adapter import (
                    prepare_official_megadepth_image_from_array,
                )

                prepared = prepare_official_megadepth_image_from_array(
                    image, self._matcher_runtime()
                )
            workmap_ranked = self._rank_workmap_references(descriptor)
            self.last_stage_ms["workmap_retrieval_ms"] = (
                time.perf_counter() - workmap_started
            ) * 1000.0
            self.last_stage_ms["workmap_matched_references"] = float(len(workmap_ranked))
            return self._solve_workmap_references(image, prepared, workmap_ranked, started)
        except Exception as error:
            return replace(
                RelocFix.abstained(
                    f"{type(error).__name__}: {error}",
                    (time.perf_counter() - started) * 1000.0,
                ),
                reference_scope="workmap",
            )

    def _solve_references(self, image, prepared, ranked, started, viewpoint_pool_fallback):
        self.last_retrieval_source = "boq"
        self.last_n_transferred = 0
        self.last_track_anchor = None
        self.last_modes = ()
        self.last_mode_decision = None

        query_path = Path(self._queries[_PLACEHOLDER_QUERY_ID]["image_path"])
        match_started = time.perf_counter()
        lifted, raw_matches = self._match_and_lift(
            query_path, ranked, query_image=image, prepared_query=prepared
        )
        runtime_edm = time.perf_counter() - match_started
        pnp_started = time.perf_counter()
        solved, metrics, decision_status = self._solve(query_path, lifted, query_image=image)
        runtime_pnp = time.perf_counter() - pnp_started
        self.last_stage_ms["match_lift_ms"] = (
            self.last_stage_ms.get("match_lift_ms", 0.0) + runtime_edm * 1000.0
        )
        self.last_stage_ms["reloc_pnp_ms"] = (
            self.last_stage_ms.get("reloc_pnp_ms", 0.0) + runtime_pnp * 1000.0
        )
        result = self._pack_query_result(
            query=EDMQuery(
                query_id=_PLACEHOLDER_QUERY_ID,
                session_id=_PLACEHOLDER_SESSION_ID,
                timestamp=time.time(),
                image_path=str(query_path),
                pose_provenance="NONE",
            ),
            started=started,
            lifted=lifted,
            raw_matches=raw_matches,
            result=solved,
            metrics=metrics,
            decision_status=decision_status,
            reference_ids=tuple(self._reference_names[value] for value in ranked),
            viewpoint_pool_fallback=bool(viewpoint_pool_fallback),
            runtime_edm=runtime_edm,
            runtime_pnp=runtime_pnp,
            temporal_consistency=None,
        )
        return self._pack_fix(result, solved, (time.perf_counter() - started) * 1000.0)

    def _pack_fix(self, result: Any, solved: Any, runtime_ms: float) -> RelocFix:
        """Project one vendored query result onto the fast loop's seed contract.

        The 2D/3D arrays come from ``last_track_anchor``, which the vendored
        ``_anchor_if_admissible`` builds from the PnP inliers that carry a real
        map point id, gated on the frozen ACCEPT thresholds.  That is exactly the
        seed the reference two-rate replay hands over, so the same gate applies.
        """

        anchor = self.last_track_anchor
        if anchor is None:
            query_xy = np.zeros((0, 2), dtype=np.float64)
            point_ids = np.zeros((0,), dtype=np.int64)
            point_xyz = np.zeros((0, 3), dtype=np.float64)
        else:
            query_xy = np.ascontiguousarray(anchor.xy, dtype=np.float64)
            point_ids = np.ascontiguousarray(anchor.point3d_ids, dtype=np.int64)
            point_xyz = np.ascontiguousarray(self._point_xyz_for_ids(point_ids), dtype=np.float64)
        cam_from_world = (
            None if solved is None else np.asarray(solved.pose, dtype=np.float64)[:3, :4].copy()
        )
        status = str(result.query_status)
        return RelocFix(
            ok=status == "LOCALIZED_STRONG" and cam_from_world is not None,
            status=status,
            cam_from_world=cam_from_world,
            query_xy=query_xy,
            point3d_ids=point_ids,
            point_xyz=point_xyz,
            inliers=int(result.ransac_inliers or 0),
            inlier_ratio=float(result.inlier_ratio or 0.0),
            reproj_p90=(
                None if result.reprojection_p90 is None else float(result.reprojection_p90)
            ),
            reference_names=tuple(str(name) for name in (result.reference_ids or ())),
            runtime_ms=runtime_ms,
            reason=str(getattr(result, "admission_reason", "") or ""),
        )

    def close(self) -> None:
        """Release both GPU runtimes."""

        self._matcher = None
        self._gpu_matcher = None
        self._query_matches.clear()
        self._query_lifts.clear()
        self._vpr_runtime = None
        self._models_ready = False
        self._empty_cuda_cache()
        self._prepared_reference_cache.clear()
        self._workmap_references.clear()
        self._workmap_order.clear()
        self._workmap_landmark_refs.clear()
        self._workmap_descriptors = np.empty((0, BOQ_DIM), dtype=np.float32)
        self._workmap_descriptor_count = 0
        self._workmap_descriptor_capacity = 0
        self._workmap_session_id = None
        self._last_successful_workmap_reference = None

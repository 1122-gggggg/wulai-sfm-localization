"""Online, single-frame relocalizer built on the frozen offline EDM provider.

``FinalMapEDMProvider`` (vendored, unmodified) is an *offline batch* localizer:
it wants a query manifest on disk, pre-extracts VPR descriptors for the
whole query set, and reads every image back from a path.  Deployment needs the
opposite shape -- one in-memory frame at a time, arriving from a camera.

``LiveMapEDMProvider`` subclasses it and replaces exactly three things:

* the descriptor stage: the reference bank is loaded from the release's frozen
  ``.npy`` (BoQ-ResNet50, 16384-D); query descriptors are extracted from the
  live array with :class:`BoQExtractor` instead of a file set;
* the keyframe index: rebuilt from :class:`DirectMapAssets` so a release stays
  relocatable, with a per-image SHA-256 check the first time a reference is
  prepared;
* the entry point: :meth:`localize_array` runs retrieval -> EDM match/lift ->
  PnP in the same order as the vendored ``localize()``, minus the KLT bridge
  (the deployed fast loop owns tracking).

Every matching, lifting, PnP and admission decision stays inside vendor code.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from boq_extractor import BOQ_DIM, BoQExtractor
from direct_map import DirectMapAssets
from direct_paths import vendor_sys_path
from direct_profile import DirectProfile


vendor_sys_path()

from sfm_diagnosis.edm_risk.edm_loo import EDMQuery  # noqa: E402
from sfm_diagnosis.site_pipeline.deployment_localizer import (  # noqa: E402
    FinalMapEDMProvider,
    _configure_audited_runtime_site_packages,
    _resolve_audited_site_packages,
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


def boq_descriptor_from_array(extractor: BoQExtractor, rgb: np.ndarray) -> np.ndarray:
    """Extract one 16,384-D BoQ descriptor from an in-memory RGB array.

    Thin wrapper over ``BoQExtractor.extract_from_array`` (bicubic
    ``antialias=True`` resize to the ``BOQ_INPUT`` square, 384; ImageNet
    mean/std; model; L2 normalise).  The frozen reference bank was produced
    by that exact transform; any deviation silently re-ranks retrieval.
    """

    descriptor = extractor.extract_from_array(rgb)
    if descriptor.shape != (BOQ_DIM,):
        raise RuntimeError(
            f"BoQ descriptor shape {descriptor.shape!r} violates the "
            f"{BOQ_DIM}-D contract"
        )
    if not np.isfinite(descriptor).all():
        raise RuntimeError("BoQ emitted a non-finite descriptor")
    norm = float(np.linalg.norm(descriptor))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("BoQ emitted a zero-norm descriptor")
    return np.ascontiguousarray(descriptor / norm, dtype=np.float32)


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
        # Replace the mapping machine's absolute /home/cihcilab image URIs with
        # this release's own paths, before any geometry load resolves them.
        self._keyframes = keyframes
        self._vpr_runtime: BoQExtractor | None = None
        self._index: Any | None = None
        self._models_ready = False
        self._lock = threading.Lock()

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

    # -- vendor overrides --------------------------------------------------

    def _load_descriptors(self) -> None:
        """Install the frozen reference bank; never extract a query catalog."""

        names = tuple(
            str(name)
            for name in json.loads(self.assets.bank_names.read_text(encoding="utf-8"))
        )
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
                f"EDM COARSE.TOPK did not merge: {runtime.coarse_topk} "
                f"(expected {EDM_COARSE_TOPK})"
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
        _configure_audited_runtime_site_packages(self.audited_edm_site_packages)
        self._prepare()
        self._matcher_runtime()
        self._vpr()
        self._index = self.build_reference_index(
            excluded_sessions=frozenset(), strict=False
        )
        width, height = self.profile.fast_loop.resolution
        # A real forward pass through both networks; an exception here is a
        # broken deployment and must not be hidden behind the first stream frame.
        # The colour argument is warmed too, because that is the deployed path:
        # the fast loop hands its BGR frame over with every relocalization.
        self._localize_array(
            np.zeros((height, width), dtype=np.uint8),
            np.zeros((height, width, 3), dtype=np.uint8),
        )
        self._models_ready = True

    @property
    def ref_names(self) -> tuple[str, ...]:
        return tuple(self._reference_names)

    # -- localization ------------------------------------------------------

    def localize_array(self, gray: np.ndarray, *, color_bgr: np.ndarray | None = None) -> RelocFix:
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
                return self._localize_array(gray, color_bgr)
        except Exception as error:  # deployment contract: the caller keeps flying
            return RelocFix.abstained(
                f"{type(error).__name__}: {error}",
                (time.perf_counter() - started) * 1000.0,
            )

    def _localize_array(self, gray: np.ndarray, color_bgr: np.ndarray | None) -> RelocFix:
        import cv2

        if self._index is None:
            raise RuntimeError("ensure_models() must run before localize_array()")
        started = time.perf_counter()
        image = np.ascontiguousarray(gray)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("live query must be an HxW uint8 grayscale array")
        if color_bgr is None:
            rgb = np.repeat(image[:, :, None], 3, axis=2)
        else:
            rgb = cv2.cvtColor(np.ascontiguousarray(color_bgr), cv2.COLOR_BGR2RGB)

        descriptor = boq_descriptor_from_array(self._vpr(), rgb)
        subset = self._subsets[self._index.index_id]
        ranked, viewpoint_pool_fallback = rank_reference_indices(
            self._reference_descriptors,
            descriptor,
            reference_sessions=self._reference_sessions,
            excluded_sessions=subset.excluded_sessions,
            top_k=self.top_k,
            occupied_bins=self._reference_occupied_bins,
            min_occupied_bins=self.min_reference_occupied_bins,
            retrieve_pool=None,
        )
        allowed = set(subset.indices)
        ranked = tuple(index for index in ranked if index in allowed)
        if not ranked:
            return RelocFix.abstained(
                "retrieval returned no admissible reference",
                (time.perf_counter() - started) * 1000.0,
            )

        self.last_retrieval_source = "boq"
        self.last_n_transferred = 0
        self.last_track_anchor = None
        self.last_modes = ()
        self.last_mode_decision = None

        query_path = Path(self._queries[_PLACEHOLDER_QUERY_ID]["image_path"])
        match_started = time.perf_counter()
        lifted, raw_matches = self._match_and_lift(query_path, ranked, query_image=image)
        runtime_edm = time.perf_counter() - match_started
        pnp_started = time.perf_counter()
        solved, metrics, decision_status = self._solve(query_path, lifted, query_image=image)
        runtime_pnp = time.perf_counter() - pnp_started
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
            point_xyz = np.ascontiguousarray(
                self._point_xyz_for_ids(point_ids), dtype=np.float64
            )
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
        self._vpr_runtime = None
        self._models_ready = False
        self._empty_cuda_cache()

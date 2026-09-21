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

import json
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from boq_extractor import BOQ_DIM, BoQExtractor
from direct_map import DirectMapAssets
from direct_paths import sha256_file, vendor_sys_path
from direct_profile import DirectProfile
from edm_inference import CachedEDM

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
        self._last_strong_refs: tuple[int, ...] = ()
        self.last_stage_ms: dict[str, float] = {}
        self._reference_norms: np.ndarray | None = None
        self.startup_stage_ms: dict[str, float] = {}

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

    # -- localization ------------------------------------------------------

    def _match_prepared(self, runtime, query, reference):
        from river_map_quality.official_edm_adapter import match_official_prepared

        key = id(reference)
        if key not in self._query_matches:
            capacity = self.profile.optimizations.gpu_cache_size
            if capacity:
                if self._gpu_matcher is None:
                    self._gpu_matcher = CachedEDM(runtime, capacity)
                matched = self._gpu_matcher.match(query, reference)
            else:
                matched = match_official_prepared(runtime, query, reference)
            self._query_matches[key] = (reference, matched)
        return self._query_matches[key][1]

    def _match_prepared_batch(self, runtime, query, references):
        from river_map_quality.official_edm_adapter import match_official_prepared_batch

        missing = [
            reference for reference in references if id(reference) not in self._query_matches
        ]
        if missing:
            matched = match_official_prepared_batch(runtime, query, missing)
            self._query_matches.update(
                (id(ref), (ref, result)) for ref, result in zip(missing, matched, strict=True)
            )
        return [self._query_matches[id(reference)][1] for reference in references]

    def _recovery_references(self, ranked: tuple[int, ...]) -> tuple[int, ...]:
        """Keep the global winner; add a covisible alternative when available."""
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
        # Only rank the bounded global shortlist, never lock recovery to a prior.
        overlap = [
            len(np.intersect1d(seeds, self._observations[self._reference_names[index]][1]))
            for index in extra[1:]
        ]
        neighbor = extra[1 + int(np.argmax(overlap))]
        return tuple(dict.fromkeys([extra[0], neighbor, *extra]))[: self.top_k]

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
        self._query_matches.clear()
        self.last_stage_ms = {}
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
        subset = self._subsets[self._index.index_id]
        ranked, viewpoint_pool_fallback = rank_reference_indices(
            self._reference_descriptors,
            descriptor,
            reference_sessions=self._reference_sessions,
            excluded_sessions=subset.excluded_sessions,
            top_k=self.top_k * (3 if self.profile.optimizations.adaptive_retrieval else 1),
            occupied_bins=self._reference_occupied_bins,
            min_occupied_bins=self.min_reference_occupied_bins,
            retrieve_pool=None,
            reference_norms=self._reference_norms,
        )
        allowed = set(subset.indices)
        ranked = tuple(index for index in ranked if index in allowed)
        self.last_stage_ms["retrieval_ms"] = (time.perf_counter() - started) * 1000.0
        if not ranked:
            return RelocFix.abstained(
                "retrieval returned no admissible reference",
                (time.perf_counter() - started) * 1000.0,
            )

        base = ranked[: self.top_k]
        from river_map_quality.official_edm_adapter import (
            prepare_official_megadepth_image_from_array,
        )

        prepared = prepare_official_megadepth_image_from_array(image, self._matcher_runtime())
        fix = self._solve_references(image, prepared, base, started, viewpoint_pool_fallback)
        self.last_stage_ms["matched_references"] = float(len(base))
        # Preserve every strong baseline decision. Expansion only rescues failures,
        # and only while this attempt still has budget within its normal period.
        if (
            not fix.ok
            and self.profile.optimizations.adaptive_retrieval
            and time.perf_counter() - started < self.profile.reloc.period_s
        ):
            extra = self._recovery_references(ranked)
            if extra:
                refs = base + extra
                expanded = self._solve_references(
                    image, prepared, refs, started, viewpoint_pool_fallback
                )
                self.last_stage_ms["matched_references"] = float(len(refs))
                if expanded.ok:
                    fix = expanded
                    base = refs
        if fix.ok:
            self._last_strong_refs = base
        else:
            self._last_strong_refs = ()
        return replace(
            fix,
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            stage_ms=dict(self.last_stage_ms),
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
        self._vpr_runtime = None
        self._models_ready = False
        self._empty_cuda_cache()

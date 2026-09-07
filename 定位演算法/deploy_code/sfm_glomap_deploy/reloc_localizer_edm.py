#!/usr/bin/env python3
"""EDM relocalizer: BoQ-ResNet50 retrieval -> EDM matching -> PnP.

Drop-in replacement for the XFeat + LighterGlue + mutual-NN local matcher. Retrieval is
separate: the bundle carries BoQ-ResNet50 global descriptors, so only the local
matching and the 2D->3D lookup change.

WHY THERE IS NO KD-TREE / DEPTH MAP / SNAP RADIUS HERE:
  The reference is passed as image0, so every direction-01 match puts the reference side
  exactly on its coarse cell centre -- the same anchor the map build triangulated. The 3D
  lookup is therefore an exact O(1) index into xyz_by_cell, with no positional error
  between what was triangulated and what is matched at flight time.

  Direction-10 matches (reference side refined, query side on the grid) are DISCARDED:
  their reference point is not an anchor, so their 3D would be off by up to half a cell.
  Half the matches are dropped and there are still ~1000+ left per reference.

The bundle embeds the reference images (EDM must SEE them), so no external image
directory is needed at flight time.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from artifact_integrity import verify_sha256
from boq_query import BOQ_DIM, BoQQuery
from edm_matcher import EDM_H, EDM_W, EDMMatcher
from reference_index import ReferenceIndex

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The river bundle currently has 454 references.  Keep a fixed load-time budget
# safely above that map while rejecting 100k-reference retrieval, which this
# embedded-image plus per-reference-XYZ architecture does not support.
EDM_MAX_REFERENCE_COUNT = 4096
EDM_MAX_BUNDLE_FILE_BYTES = 2 * 1024 * 1024 * 1024
EDM_MAX_IMAGE_ENCODED_BYTES = 4 * 1024 * 1024
EDM_MAX_TOTAL_ENCODED_IMAGE_BYTES = 512 * 1024 * 1024
EDM_MAX_TOTAL_DECODED_IMAGE_BYTES = 1024 * 1024 * 1024
EDM_MAX_XYZ_BYTES = 512 * 1024 * 1024
EDM_GLOBAL_DESCRIPTOR_DIM = BOQ_DIM
EDM_MAX_GLOBAL_DESCRIPTOR_BYTES = 256 * 1024 * 1024
EDM_MAX_COVIS_EDGES = EDM_MAX_REFERENCE_COUNT * 128

_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


def _next_jpeg_marker(raw: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(raw) or raw[offset] != 0xFF:
        raise ValueError("embedded EDM reference JPEG has an invalid marker")
    while offset < len(raw) and raw[offset] == 0xFF:
        offset += 1
    if offset >= len(raw):
        raise ValueError("embedded EDM reference JPEG has a truncated marker")
    return raw[offset], offset + 1


def _jpeg_segment_length(raw: bytes, offset: int) -> int:
    if offset + 2 > len(raw):
        raise ValueError("embedded EDM reference JPEG has a truncated segment")
    length = int.from_bytes(raw[offset : offset + 2], "big")
    if length < 2 or offset + length > len(raw):
        raise ValueError("embedded EDM reference JPEG has an invalid segment length")
    return length


def _jpeg_sof_dimensions(raw: bytes, offset: int, length: int) -> tuple[int, int]:
    if length < 7:
        raise ValueError("embedded EDM reference JPEG has a truncated SOF")
    height = int.from_bytes(raw[offset + 3 : offset + 5], "big")
    width = int.from_bytes(raw[offset + 5 : offset + 7], "big")
    if width <= 0 or height <= 0:
        raise ValueError("embedded EDM reference JPEG has invalid dimensions")
    return width, height


def _jpeg_dimensions(encoded: np.ndarray) -> tuple[int, int]:
    """Read JPEG SOF dimensions without allocating its decoded pixel buffer."""
    raw = encoded.tobytes()
    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        raise ValueError("embedded EDM reference image is not a JPEG")
    offset = 2
    while offset < len(raw):
        marker, offset = _next_jpeg_marker(raw, offset)
        if marker in {0xD9, 0xDA}:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        length = _jpeg_segment_length(raw, offset)
        if marker in _JPEG_SOF_MARKERS:
            return _jpeg_sof_dimensions(raw, offset, length)
        offset += length
    raise ValueError("embedded EDM reference JPEG has no supported SOF marker")


def _decode_reference_image(item: tuple[str, object]) -> tuple[str, np.ndarray]:
    name, encoded = item
    decoded = cv2.imdecode(
        np.asarray(encoded, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if decoded is None or decoded.shape != (576, 1024):
        raise ValueError(f"EDM reference image failed to decode at 1024x576: {name}")
    return name, decoded


@dataclass
class EDMRelocMap:
    ref_names: list
    ref_global: np.ndarray  # (N, BOQ_DIM) L2-normalised BoQ
    xyz_by_cell: dict  # name -> (N_CELLS, 3) float32, NaN = unanchored
    images: dict  # name -> (576, 1024) uint8 grayscale
    meta: dict
    ref_centers: np.ndarray | None = None
    ref_yaws: np.ndarray | None = None
    covis: dict | None = None

    @staticmethod
    def load(
        path: str | Path,
        expected_sha256: str | None = None,
    ) -> "EDMRelocMap":
        artifact = Path(path)
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(
                f"EDM relocation bundle must be a regular non-symlink file: {artifact}"
            )
        artifact_size = artifact.stat().st_size
        if artifact_size <= 0 or artifact_size > EDM_MAX_BUNDLE_FILE_BYTES:
            raise ValueError(
                f"EDM relocation bundle size {artifact_size} exceeds the load budget "
                f"{EDM_MAX_BUNDLE_FILE_BYTES}"
            )
        verify_sha256(artifact, expected_sha256)
        dtypes = (
            np.float16,
            np.float32,
            np.float64,
            np.uint8,
            np.int32,
            np.int64,
            np.bool_,
        )
        numpy_safe_globals = [
            np._core.multiarray._reconstruct,
            np._core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            *(type(np.dtype(dtype)) for dtype in dtypes),
        ]
        with torch.serialization.safe_globals(numpy_safe_globals):
            b = torch.load(
                artifact,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        b = _validate_edm_bundle_schema(b)
        names = list(b["ref_names"])
        xyz = {name: np.asarray(b["refs"][name]["xyz_by_cell"], np.float32) for name in names}
        encoded_images = [(name, b["refs"][name]["image_jpg"]) for name in names]
        decode_workers = min(16, len(names), os.cpu_count() or 1)
        if decode_workers <= 1:
            decoded_images = map(_decode_reference_image, encoded_images)
            imgs = dict(decoded_images)
        else:
            with ThreadPoolExecutor(
                max_workers=decode_workers,
                thread_name_prefix="edm-map-decode",
            ) as executor:
                imgs = dict(executor.map(_decode_reference_image, encoded_images))
        g = np.asarray(b["ref_global"], np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        return EDMRelocMap(
            ref_names=names,
            ref_global=g,
            xyz_by_cell=xyz,
            images=imgs,
            meta=dict(b["meta"]),
            ref_centers=None
            if "ref_centers" not in b
            else np.asarray(b["ref_centers"], np.float32),
            ref_yaws=None if "ref_yaws" not in b else np.asarray(b["ref_yaws"], np.float32),
            covis=b.get("covis"),
        )


def _validate_edm_bundle_structure(bundle: object) -> tuple[dict, list, dict]:
    if not isinstance(bundle, dict):
        raise ValueError("EDM relocation bundle must be a dictionary")
    required = {"meta", "ref_names", "ref_global", "refs"}
    allowed = required | {"ref_centers", "ref_yaws", "covis"}
    if not required.issubset(bundle) or not set(bundle).issubset(allowed):
        raise ValueError(
            f"unexpected EDM bundle keys: {sorted(set(bundle) - allowed)}; "
            f"missing: {sorted(required - set(bundle))}"
        )
    meta = bundle["meta"]
    if not isinstance(meta, dict) or str(meta.get("feature", "")).lower() != "edm":
        feature = None if not isinstance(meta, dict) else meta.get("feature")
        raise ValueError(f"not an EDM bundle: feature={feature!r}")
    names = bundle["ref_names"]
    refs = bundle["refs"]
    if not isinstance(names, list) or not names:
        raise ValueError("EDM bundle ref_names must be unique non-empty strings")
    if len(names) > EDM_MAX_REFERENCE_COUNT:
        raise ValueError(
            f"EDM bundle reference count {len(names)} exceeds supported bound "
            f"{EDM_MAX_REFERENCE_COUNT}; 100k-reference retrieval is not supported "
            "by this architecture"
        )
    if not all(isinstance(name, str) and name for name in names) or len(names) != len(set(names)):
        raise ValueError("EDM bundle ref_names must be unique non-empty strings")
    if not isinstance(refs, dict):
        raise ValueError("EDM bundle refs must be a dictionary")
    return meta, names, refs


def _validate_edm_bundle_dimensions(meta: dict, names: list) -> tuple[int, int, int, int]:
    dimensions = (
        meta.get("edm_grid_w"),
        meta.get("edm_grid_h"),
        meta.get("edm_input_w"),
        meta.get("edm_input_h"),
    )
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in dimensions
    ) or dimensions != (128, 72, 1024, 576):
        raise ValueError("EDM bundle grid/input dimensions are incompatible with this runtime")
    _, _, input_w, input_h = dimensions
    projected_decoded_bytes = len(names) * input_w * input_h
    if projected_decoded_bytes > EDM_MAX_TOTAL_DECODED_IMAGE_BYTES:
        raise ValueError(
            "EDM bundle projected decoded image bytes "
            f"{projected_decoded_bytes} exceed the load budget "
            f"{EDM_MAX_TOTAL_DECODED_IMAGE_BYTES}"
        )
    return dimensions


def _validate_edm_bundle_refs(refs: dict, names: list) -> None:
    if set(refs) != set(names):
        raise ValueError("EDM bundle refs do not exactly match ref_names")


def _validate_edm_global_descriptors(ref_global: object, names: list) -> None:
    if (
        not isinstance(ref_global, np.ndarray)
        or ref_global.ndim != 2
        or ref_global.shape != (len(names), EDM_GLOBAL_DESCRIPTOR_DIM)
        or not np.issubdtype(ref_global.dtype, np.floating)
        or ref_global.nbytes > EDM_MAX_GLOBAL_DESCRIPTOR_BYTES
        or not np.isfinite(ref_global).all()
        or np.any(np.linalg.norm(ref_global, axis=1) <= 1e-12)
    ):
        raise ValueError("EDM bundle ref_global has an invalid shape, dtype, size, or value")


def _validate_edm_reference_entry(
    name: str, entry: object, cell_count: int
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(entry, dict) or set(entry) != {"xyz_by_cell", "image_jpg"}:
        raise ValueError(f"invalid EDM reference entry: {name}")
    xyz = entry["xyz_by_cell"]
    image_jpg = entry["image_jpg"]
    if not isinstance(xyz, np.ndarray) or xyz.shape != (cell_count, 3) or xyz.dtype != np.float32:
        raise ValueError(f"invalid EDM xyz_by_cell: {name}")
    if (
        not isinstance(image_jpg, np.ndarray)
        or image_jpg.ndim != 1
        or image_jpg.dtype != np.uint8
        or image_jpg.size == 0
    ):
        raise ValueError(f"invalid embedded EDM reference image: {name}")
    return xyz, image_jpg


def _validate_edm_reference_budgets(
    name: str,
    xyz: np.ndarray,
    image_jpg: np.ndarray,
    total_xyz_bytes: int,
    total_encoded_image_bytes: int,
) -> tuple[int, int]:
    total_xyz_bytes += int(xyz.nbytes)
    if total_xyz_bytes > EDM_MAX_XYZ_BYTES:
        raise ValueError(
            f"EDM bundle XYZ memory {total_xyz_bytes} exceeds the load budget {EDM_MAX_XYZ_BYTES}"
        )
    if image_jpg.nbytes > EDM_MAX_IMAGE_ENCODED_BYTES:
        raise ValueError(
            f"EDM reference encoded image is too large at {name}: "
            f"{image_jpg.nbytes} > {EDM_MAX_IMAGE_ENCODED_BYTES} bytes"
        )
    total_encoded_image_bytes += int(image_jpg.nbytes)
    if total_encoded_image_bytes > EDM_MAX_TOTAL_ENCODED_IMAGE_BYTES:
        raise ValueError(
            "EDM bundle encoded image bytes "
            f"{total_encoded_image_bytes} exceed the load budget "
            f"{EDM_MAX_TOTAL_ENCODED_IMAGE_BYTES}"
        )
    return total_xyz_bytes, total_encoded_image_bytes


def _validate_edm_reference_image(
    name: str, image_jpg: np.ndarray, input_w: int, input_h: int
) -> None:
    if _jpeg_dimensions(image_jpg) != (input_w, input_h):
        raise ValueError(f"EDM reference JPEG dimensions are incompatible at {name}")


def _validate_edm_reference_xyz(name: str, xyz: np.ndarray) -> None:
    valid_xyz_rows = np.isfinite(xyz).all(axis=1) | np.isnan(xyz).all(axis=1)
    if np.isinf(xyz).any() or not valid_xyz_rows.all():
        raise ValueError(f"invalid non-finite EDM anchors: {name}")


def _validate_edm_references(
    names: list, refs: dict, dimensions: tuple[int, int, int, int]
) -> None:
    grid_w, grid_h, input_w, input_h = dimensions
    cell_count = grid_w * grid_h
    total_xyz_bytes = 0
    total_encoded_image_bytes = 0
    for name in names:
        xyz, image_jpg = _validate_edm_reference_entry(name, refs[name], cell_count)
        total_xyz_bytes, total_encoded_image_bytes = _validate_edm_reference_budgets(
            name, xyz, image_jpg, total_xyz_bytes, total_encoded_image_bytes
        )
        _validate_edm_reference_image(name, image_jpg, input_w, input_h)
        _validate_edm_reference_xyz(name, xyz)


def _validate_edm_optional_fields(bundle: dict, names: list) -> None:
    for key, shape in (("ref_centers", (len(names), 3)), ("ref_yaws", (len(names),))):
        value = bundle.get(key)
        if value is not None and (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"invalid EDM bundle {key}")


def _validate_edm_covis(bundle: dict, names: list) -> None:
    covis = bundle.get("covis")
    if covis is not None:
        if not isinstance(covis, dict) or set(covis) != set(names):
            raise ValueError("invalid EDM bundle covis keys")
        total_covis_edges = 0
        for index, name in enumerate(names):
            neighbors = covis[name]
            if (
                not isinstance(neighbors, list)
                or any(
                    type(value) is not int or value < 0 or value >= len(names)
                    for value in neighbors
                )
                or index in neighbors
                or len(neighbors) != len(set(neighbors))
            ):
                raise ValueError(f"invalid EDM bundle covis neighbors: {name}")
            total_covis_edges += len(neighbors)
            if total_covis_edges > EDM_MAX_COVIS_EDGES:
                raise ValueError(f"EDM bundle covis edge count exceeds {EDM_MAX_COVIS_EDGES}")


def _validate_edm_bundle_schema(bundle: object) -> dict:
    meta, names, refs = _validate_edm_bundle_structure(bundle)
    dimensions = _validate_edm_bundle_dimensions(meta, names)
    _validate_edm_bundle_refs(refs, names)
    _validate_edm_global_descriptors(bundle["ref_global"], names)
    _validate_edm_references(names, refs, dimensions)
    _validate_edm_optional_fields(bundle, names)
    _validate_edm_covis(bundle, names)
    return bundle


@dataclass
class Camera:
    model: str
    width: int
    height: int
    params: list = field(default_factory=list)


class EDMLocalizer:
    def __init__(
        self,
        reloc_map: EDMRelocMap,
        camera: Camera,
        matcher: EDMMatcher | None = None,
        vpr: BoQQuery | None = None,
        topk: int = 5,
        min_conf: float = 0.2,
        pnp_max_error: float = 5.0,
        min_inliers: int = 50,
        reference_index: ReferenceIndex | None = None,
        vpr_factory=None,
    ):
        self.map = reloc_map
        self._ref_index_by_name = {name: index for index, name in enumerate(reloc_map.ref_names)}
        self.cam = camera
        self.matcher = matcher or EDMMatcher(
            mconf_thr=min_conf, runtime_sigma_mode="reference_grid"
        )
        self._vpr = vpr
        self._vpr_factory = vpr_factory
        self.topk = topk
        self.min_inliers = min_inliers
        self.pnp_max_error = pnp_max_error
        self.scale_x = camera.width / EDM_W
        self.scale_y = camera.height / EDM_H
        # Keep the historical ``scale`` seam for the temporal EDM LUT.  Numpy
        # broadcasting makes it an x/y pair instead of assuming a 16:9 camera.
        self.scale = np.asarray((self.scale_x, self.scale_y), dtype=np.float32)
        self.reference_index = reference_index
        self._ref_global_tensor = None
        if reference_index is not None:
            names = tuple(reloc_map.ref_names)
            if reference_index.count != len(names) or set(reference_index.names) != set(names):
                raise ValueError("reference index names do not match EDM relocation bundle")

    def _reference_descriptors_on(self, device: torch.device) -> torch.Tensor:
        cached = self._ref_global_tensor
        if cached is None or cached.device != device:
            cached = torch.from_numpy(
                np.ascontiguousarray(self.map.ref_global, dtype=np.float32)
            ).to(device)
            self._ref_global_tensor = cached
        return cached

    def _retrieve_scored_device(
        self,
        descriptor: torch.Tensor,
        k: int,
        exclude: set[str] | None,
        candidates: list[str] | None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        descriptor = descriptor.to(dtype=torch.float32)
        refs = self._reference_descriptors_on(descriptor.device)
        if candidates is not None:
            names = [
                name for name in dict.fromkeys(candidates) if not exclude or name not in exclude
            ]
            if not names or k <= 0:
                return []
            indices = torch.tensor(
                [self._ref_index_by_name[name] for name in names],
                dtype=torch.long,
                device=descriptor.device,
            )
            scores = refs.index_select(0, indices) @ descriptor
            order = torch.argsort(scores, descending=True, stable=True)[:k]
            order_cpu = order.cpu().tolist()
            score_cpu = scores.index_select(0, order).cpu().tolist()
            return [(names[int(index)], float(score)) for index, score in zip(order_cpu, score_cpu)]
        scores = refs @ descriptor
        order_cpu = torch.argsort(scores, descending=True, stable=True).cpu().tolist()
        scores_cpu = scores.cpu().tolist()
        out = []
        for index in order_cpu:
            name = self.map.ref_names[int(index)]
            if exclude and name in exclude:
                continue
            out.append((name, float(scores_cpu[int(index)])))
            if len(out) == k:
                break
        return out

    @property
    def vpr(self) -> BoQQuery:
        if self._vpr is None:
            self._vpr = self._vpr_factory() if self._vpr_factory is not None else BoQQuery()
        return self._vpr

    def retrieve_scored(
        self,
        frame_rgb: np.ndarray,
        k: int,
        exclude: set[str] | None = None,
        candidates: list[str] | None = None,
        descriptor: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        extract_tensor = getattr(self.vpr, "extract_one_tensor", None)
        if self.reference_index is None and callable(extract_tensor):
            d_tensor = (
                extract_tensor(frame_rgb)
                if descriptor is None
                else torch.as_tensor(descriptor, device=self.vpr.device)
            )
            return self._retrieve_scored_device(d_tensor, k, exclude, candidates)
        d = self.vpr.extract_one(frame_rgb) if descriptor is None else descriptor
        if torch.is_tensor(d):
            d = d.detach().cpu().numpy()
        if candidates is not None:
            names = [
                name for name in dict.fromkeys(candidates) if not exclude or name not in exclude
            ]
            indices = np.asarray(
                [self._ref_index_by_name[name] for name in names],
                dtype=np.int64,
            )
            if not len(indices) or k <= 0:
                return []
            sim = self.map.ref_global[indices] @ d
            order = np.argsort(-sim)[:k]
            return [(names[int(index)], float(sim[int(index)])) for index in order]
        if self.reference_index is not None:
            requested = max(0, int(k))
            if requested == 0:
                return []
            candidate_count = min(
                self.reference_index.count,
                requested + len(exclude or ()),
            )
            matches = self.reference_index.query(d, top_k=candidate_count)
            out = [
                (match.name, float(match.score))
                for match in matches
                if not exclude or match.name not in exclude
            ]
            return out[:requested]
        sim = self.map.ref_global @ d
        order = np.argsort(-sim)
        out: list[tuple[str, float]] = []
        for i in order:
            name = self.map.ref_names[i]
            if exclude and name in exclude:
                continue
            out.append((name, float(sim[int(i)])))
            if len(out) == k:
                break
        return out

    def retrieve(
        self,
        frame_rgb: np.ndarray,
        k: int,
        exclude: set[str] | None = None,
        candidates: list[str] | None = None,
    ) -> list[str]:
        return [
            name
            for name, _score in self.retrieve_scored(
                frame_rgb, k, exclude=exclude, candidates=candidates
            )
        ]

    def correspondences(
        self,
        query_gray: np.ndarray,
        ref_names: list[str],
        batch_size: int | None = None,
    ):
        """-> (camera-pixel 2D, map 3D, confidence, per-ref match counts)."""
        by_ref = self.correspondences_by_ref(query_gray, ref_names, batch_size=batch_size)
        p2 = [row[0] for row in by_ref if len(row[0])]
        p3 = [row[1] for row in by_ref if len(row[1])]
        confidence = [row[2] for row in by_ref if len(row[2])]
        counts = [row[3] for row in by_ref]
        if not p2:
            return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), counts
        return np.concatenate(p2), np.concatenate(p3), np.concatenate(confidence), counts

    def correspondences_by_ref(
        self,
        query_gray: np.ndarray,
        ref_names: list[str],
        batch_size: int | None = None,
        *,
        prepared_query=None,
        on_batch=None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """Return 2D/3D correspondences separately for each reference.

        Keeping these boundaries lets EDM-only map recovery score each unrelated
        global reference with its own PnP hypothesis instead of mixing their 3D points.
        """
        return self.correspondences_for_sources(
            query_gray,
            [self.map.images[name] for name in ref_names],
            [self.map.xyz_by_cell[name] for name in ref_names],
            batch_size=batch_size,
            prepared_query=prepared_query,
            on_batch=on_batch,
        )

    def correspondences_for_sources(
        self,
        query_gray: np.ndarray,
        reference_images: list[np.ndarray],
        xyz_luts: list[np.ndarray],
        batch_size: int | None = None,
        source_kinds: list[str] | None = None,
        *,
        prepared_query=None,
        on_batch=None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """Match arbitrary reference images whose coarse cells carry map-frame 3D."""
        if len(reference_images) != len(xyz_luts):
            raise ValueError("reference_images and xyz_luts must have equal length")
        if source_kinds is not None and len(source_kinds) != len(reference_images):
            raise ValueError("source_kinds must match the reference count")
        if prepared_query is None:
            prepare = getattr(self.matcher, "prepare_query", None)
            if callable(prepare):
                prepared_query = prepare(query_gray)
        size = len(reference_images) if not batch_size or batch_size <= 0 else batch_size
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        for start in range(0, len(reference_images), max(size, 1)):
            images = reference_images[start : start + max(size, 1)]
            kwargs = {}
            if source_kinds is not None:
                kwargs["source_kinds"] = source_kinds[start : start + max(size, 1)]
            if prepared_query is not None:
                kwargs["prepared_query"] = prepared_query
            results = self.matcher.match_many_to_one(images, query_gray, **kwargs)
            chunk_rows = []
            for xyz_lut, result in zip(xyz_luts[start : start + len(images)], results):
                chunk_rows.append(self._correspondence_row(xyz_lut, result))
            out.extend(chunk_rows)
            if callable(on_batch):
                on_batch(start, chunk_rows)
        return out

    def _correspondence_row(
        self,
        xyz_lut: np.ndarray,
        result: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        k0, k1 = result["mkpts0"], result["mkpts1"]
        confidence = np.asarray(result["mconf"], dtype=np.float32)
        empty = (
            np.zeros((0, 2)),
            np.zeros((0, 3)),
            np.zeros(0),
            0,
        )
        if len(k0) == 0:
            return empty
        if len(confidence) != len(k0) or len(k1) != len(k0):
            raise ValueError("EDM match confidence is misaligned with keypoints")
        # keep direction-01 only: the reference side must sit on its cell centre,
        # which is the anchor the map build triangulated.
        d01 = ~EDMMatcher.is_refined(k0)
        if not d01.any():
            return empty
        cells = EDMMatcher.cell_ids(k0[d01])
        xyz = xyz_lut[cells]
        conf = confidence[d01]
        query_pts = k1[d01]
        ok = np.isfinite(xyz).all(1)
        ok &= np.isfinite(conf)
        ok &= np.isfinite(query_pts).all(1)
        if not ok.any():
            return empty
        return (
            query_pts[ok] * self.scale,
            xyz[ok],
            conf[ok],
            int(ok.sum()),
        )

    def localize(
        self,
        frame_bgr: np.ndarray,
        ref_names: list[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict | None:
        import pycolmap

        gray = EDMMatcher.load_gray(frame_bgr)
        if ref_names is None:
            rgb = (
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_bgr.ndim == 3
                else cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
            )
            ref_names = self.retrieve(rgb, self.topk, exclude=exclude)

        p2, p3, _confidence, counts = self.correspondences(gray, ref_names)
        if len(p3) < 6:
            return None
        cam = pycolmap.Camera(
            model=self.cam.model,
            width=self.cam.width,
            height=self.cam.height,
            params=self.cam.params,
        )
        opts = pycolmap.AbsolutePoseEstimationOptions()
        opts.ransac.max_error = self.pnp_max_error
        ret = pycolmap.estimate_and_refine_absolute_pose(
            np.asarray(p2, float), np.asarray(p3, float), cam, opts
        )
        if ret is None:
            return None
        T = ret["cam_from_world"]
        R = T.rotation.matrix()
        t = np.asarray(T.translation)
        C = -R.T @ t
        fwd = R.T @ np.array([0, 0, 1.0])
        n_in = int(ret["num_inliers"])
        return {
            "R": R,
            "t": t,
            "center": C,
            "yaw": float(math.atan2(fwd[1], fwd[0])),
            "inliers": n_in,
            "n_corr": int(len(p3)),
            "refs": list(ref_names),
            "per_ref": counts,
            "ok": n_in >= self.min_inliers,
        }


if __name__ == "__main__":
    print(__doc__)

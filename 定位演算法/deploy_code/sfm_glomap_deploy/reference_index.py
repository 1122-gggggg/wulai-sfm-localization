"""Deterministic, bounded IVF reference-descriptor index.

This module is an intentionally small replacement seam for large reference
maps.  It stores normalized descriptors in an on-disk NumPy array and uses a
coarse inverted file (IVF) lookup before exact reranking.  It does not change
the existing EDM bundle loader or tracker.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np


_SCHEMA = "localization-reference-index"
_FORMAT_VERSION = 1
_MANIFEST_SCHEMA = "localization-reference-index-sha256"
_REQUIRED_FILES = (
    "metadata.json",
    "centroids.npy",
    "postings_offsets.npy",
    "postings_indices.npy",
    "descriptors.npy",
    "names.json",
)
_DEFAULT_BATCH_SIZE = 4096
_DEFAULT_KMEANS_ITERATIONS = 8
_DEFAULT_MAX_QUERY_PROBES = 8
_DEFAULT_MAX_QUERY_CANDIDATES = 4096
_NORMALIZATION_TOLERANCE = 1e-3


class IndexFormatError(ValueError):
    """Raised when an index is missing, corrupt, or does not match its contract."""


@dataclass(frozen=True)
class ReferenceMatch:
    """One exact-reranked reference match.

    ``score`` is cosine similarity because both the stored and query vectors
    are L2-normalized.  ``distance`` is the corresponding squared L2 distance.
    """

    name: str
    score: float
    distance: float


@dataclass(frozen=True)
class QueryStats:
    """Bounded work performed by the most recent query."""

    probes: int
    candidate_count: int
    exact_rerank_count: int


@runtime_checkable
class ReferenceIndex(Protocol):
    """Replaceable lookup interface for localizers and future backends."""

    model_identity: str
    dimension: int
    count: int

    @property
    def names(self) -> tuple[str, ...]:
        """Stable identifiers in descriptor-storage order."""
        ...

    def query(
        self,
        descriptor: Sequence[float] | np.ndarray,
        *,
        top_k: int = 10,
        probes: int | None = None,
        max_candidates: int | None = None,
    ) -> tuple[ReferenceMatch, ...]:
        """Return exact-reranked matches under explicit work bounds."""


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_finite_normalized_descriptors(
    descriptors: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    try:
        values = np.asarray(descriptors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("descriptors must be a numeric 2-D array") from exc
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("descriptors must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("descriptors must contain only finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("descriptors must have finite, non-zero norms")
    if not np.all(np.abs(norms - 1.0) <= _NORMALIZATION_TOLERANCE):
        raise ValueError("descriptors must be L2-normalized")
    return np.ascontiguousarray(values, dtype=np.float32)


def _as_query(descriptor: Sequence[float] | np.ndarray, dimension: int) -> np.ndarray:
    try:
        value = np.asarray(descriptor, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("query descriptor must be a numeric 1-D array") from exc
    if value.ndim != 1 or value.shape[0] != dimension:
        raise ValueError(f"query descriptor dimension must be {dimension}")
    if not np.isfinite(value).all():
        raise ValueError("query descriptor must contain only finite values")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("query descriptor must have a finite, non-zero norm")
    return np.ascontiguousarray(value / norm, dtype=np.float32)


def _validate_names(names: Iterable[str], count: int) -> list[str]:
    values = list(names)
    if len(values) != count:
        raise ValueError(f"names count {len(values)} does not match descriptor count {count}")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError("names must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("names must be unique stable identifiers")
    return values


def _assign(descriptors: np.ndarray, centroids: np.ndarray, batch_size: int) -> np.ndarray:
    assignments = np.empty(descriptors.shape[0], dtype=np.int32)
    for start in range(0, descriptors.shape[0], batch_size):
        end = min(start + batch_size, descriptors.shape[0])
        scores = descriptors[start:end] @ centroids.T
        # argmax chooses the lowest centroid id on an exact tie.
        assignments[start:end] = np.argmax(scores, axis=1).astype(np.int32, copy=False)
    return assignments


def _fit_centroids(
    descriptors: np.ndarray,
    nlist: int,
    seed: int,
    iterations: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    initial = np.sort(rng.choice(descriptors.shape[0], size=nlist, replace=False))
    centroids = descriptors[initial].astype(np.float32, copy=True)
    for _ in range(iterations):
        assignments = _assign(descriptors, centroids, batch_size)
        sums = np.zeros((nlist, descriptors.shape[1]), dtype=np.float64)
        counts = np.bincount(assignments, minlength=nlist)
        for dimension in range(descriptors.shape[1]):
            sums[:, dimension] = np.bincount(
                assignments,
                weights=descriptors[:, dimension].astype(np.float64, copy=False),
                minlength=nlist,
            )
        for centroid_id, count in enumerate(counts):
            if count == 0:
                continue
            mean = sums[centroid_id] / float(count)
            norm = float(np.linalg.norm(mean))
            if math.isfinite(norm) and norm > 0.0:
                centroids[centroid_id] = (mean / norm).astype(np.float32)
    return centroids, _assign(descriptors, centroids, batch_size)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class IVFReferenceIndex:
    """On-disk deterministic IVF index with bounded query work."""

    def __init__(
        self,
        root: Path,
        metadata: dict[str, object],
        centroids: np.ndarray,
        descriptors: np.ndarray,
        postings_offsets: np.ndarray,
        postings_indices: np.ndarray,
        names: tuple[str, ...],
    ) -> None:
        self.root = root
        self.metadata = dict(metadata)
        self.model_identity = str(metadata["model_identity"])
        self.dimension = int(metadata["dimension"])
        self.count = int(metadata["count"])
        self.nlist = int(metadata["nlist"])
        self.max_query_probes = int(metadata["max_query_probes"])
        self.max_query_candidates = int(metadata["max_query_candidates"])
        self._centroids = centroids
        self._descriptors = descriptors
        self._postings_offsets = postings_offsets
        self._postings_indices = postings_indices
        self._names = names
        self._last_query_stats = QueryStats(0, 0, 0)

    @property
    def last_query_stats(self) -> QueryStats:
        return self._last_query_stats

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @classmethod
    def build(
        cls,
        output_dir: str | os.PathLike[str],
        descriptors: Sequence[Sequence[float]] | np.ndarray,
        names: Iterable[str],
        *,
        model_identity: str,
        nlist: int | None = None,
        seed: int = 0,
        kmeans_iterations: int = _DEFAULT_KMEANS_ITERATIONS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_query_probes: int | None = None,
        max_query_candidates: int | None = None,
    ) -> Path:
        """Build an immutable index and publish it with one atomic rename."""
        if not isinstance(model_identity, str) or not model_identity:
            raise ValueError("model_identity must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        iterations = _positive_int(kmeans_iterations, "kmeans_iterations")
        batch = _positive_int(batch_size, "batch_size")
        values = _as_finite_normalized_descriptors(descriptors)
        stable_names = _validate_names(names, values.shape[0])
        order = np.asarray(
            sorted(range(values.shape[0]), key=stable_names.__getitem__), dtype=np.int64
        )
        values = np.ascontiguousarray(values[order], dtype=np.float32)
        stable_names = [stable_names[int(index)] for index in order]
        if nlist is None:
            nlist_value = max(1, min(256, math.ceil(math.sqrt(values.shape[0]))))
        else:
            nlist_value = _positive_int(nlist, "nlist")
        nlist_value = min(nlist_value, values.shape[0])
        max_probes = (
            min(_DEFAULT_MAX_QUERY_PROBES, nlist_value)
            if max_query_probes is None
            else _positive_int(max_query_probes, "max_query_probes")
        )
        max_candidates = (
            min(_DEFAULT_MAX_QUERY_CANDIDATES, values.shape[0])
            if max_query_candidates is None
            else _positive_int(max_query_candidates, "max_query_candidates")
        )
        if max_probes > nlist_value:
            raise ValueError("max_query_probes cannot exceed nlist")
        if max_candidates > values.shape[0]:
            raise ValueError("max_query_candidates cannot exceed descriptor count")

        centroids, assignments = _fit_centroids(
            values, nlist_value, int(seed), iterations, batch
        )
        posting_order = np.argsort(assignments, kind="stable").astype(np.int32, copy=False)
        counts = np.bincount(assignments, minlength=nlist_value)
        offsets = np.zeros(nlist_value + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        metadata: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "model_identity": model_identity,
            "dimension": int(values.shape[1]),
            "count": int(values.shape[0]),
            "dtype": "float32",
            "nlist": nlist_value,
            "seed": int(seed),
            "kmeans_iterations": iterations,
            "batch_size": batch,
            "max_query_probes": max_probes,
            "max_query_candidates": max_candidates,
            "normalization_tolerance": _NORMALIZATION_TOLERANCE,
        }

        output = Path(output_dir)
        parent = output.parent
        parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"index output already exists: {output}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
        try:
            _write_json(temporary / "metadata.json", metadata)
            np.save(temporary / "centroids.npy", centroids, allow_pickle=False)
            np.save(temporary / "postings_offsets.npy", offsets, allow_pickle=False)
            np.save(temporary / "postings_indices.npy", posting_order, allow_pickle=False)
            mmap = np.lib.format.open_memmap(
                temporary / "descriptors.npy",
                mode="w+",
                dtype=np.float32,
                shape=values.shape,
            )
            try:
                for start in range(0, values.shape[0], batch):
                    end = min(start + batch, values.shape[0])
                    mmap[start:end] = values[start:end]
                mmap.flush()
            finally:
                del mmap
            _write_json(temporary / "names.json", stable_names)
            manifest = {
                "schema": _MANIFEST_SCHEMA,
                "format_version": _FORMAT_VERSION,
                "files": {name: _sha256(temporary / name) for name in _REQUIRED_FILES},
            }
            _write_json(temporary / "SHA256SUMS.json", manifest)
            os.replace(temporary, output)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return output

    @classmethod
    def open(
        cls,
        index_dir: str | os.PathLike[str],
        *,
        expected_model_identity: str | None = None,
        expected_dimension: int | None = None,
    ) -> "IVFReferenceIndex":
        """Open and fully validate a published index, or fail closed."""
        root = Path(index_dir)
        if not root.is_dir():
            raise IndexFormatError(f"index directory does not exist: {root}")
        metadata = _load_index_metadata(
            root,
            expected_model_identity=expected_model_identity,
            expected_dimension=expected_dimension,
        )
        _verify_manifest(root)
        centroids, descriptors, offsets, indices, names = _load_index_arrays(root)
        stable_names = _validate_loaded_arrays(
            metadata, centroids, descriptors, offsets, indices, names
        )
        return cls(root, metadata, centroids, descriptors, offsets, indices, stable_names)

    def query(
        self,
        descriptor: Sequence[float] | np.ndarray,
        *,
        top_k: int = 10,
        probes: int | None = None,
        max_candidates: int | None = None,
    ) -> tuple[ReferenceMatch, ...]:
        top = _positive_int(top_k, "top_k")
        if top > self.count:
            raise ValueError("top_k cannot exceed descriptor count")
        probe_count = self.max_query_probes if probes is None else _positive_int(probes, "probes")
        candidate_limit = (
            self.max_query_candidates
            if max_candidates is None
            else _positive_int(max_candidates, "max_candidates")
        )
        if probe_count > self.max_query_probes:
            raise ValueError("probes exceeds the index query bound")
        if candidate_limit > self.max_query_candidates:
            raise ValueError("max_candidates exceeds the index query bound")
        if candidate_limit < top:
            raise ValueError("max_candidates cannot be smaller than top_k")
        query = _as_query(descriptor, self.dimension)
        centroid_scores = self._centroids @ query
        centroid_order = np.argsort(-centroid_scores, kind="stable")[:probe_count]
        candidate_ids = np.empty(candidate_limit, dtype=np.int32)
        candidate_count = 0
        for centroid_id in centroid_order:
            start = int(self._postings_offsets[centroid_id])
            end = int(self._postings_offsets[centroid_id + 1])
            take = min(end - start, candidate_limit - candidate_count)
            if take <= 0:
                break
            candidate_ids[candidate_count : candidate_count + take] = self._postings_indices[
                start : start + take
            ]
            candidate_count += take
        candidates = candidate_ids[:candidate_count]
        scores = self._descriptors[candidates] @ query
        ranked_slots = sorted(
            range(candidate_count),
            key=lambda slot: (-float(scores[slot]), self._names[int(candidates[slot])]),
        )[:top]
        self._last_query_stats = QueryStats(probe_count, candidate_count, candidate_count)
        return tuple(
            ReferenceMatch(
                name=self._names[int(candidates[slot])],
                score=float(scores[slot]),
                distance=max(0.0, float(2.0 - 2.0 * scores[slot])),
            )
            for slot in ranked_slots
        )


def _load_index_metadata(
    root: Path,
    *,
    expected_model_identity: str | None,
    expected_dimension: int | None,
) -> dict[str, object]:
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndexFormatError("invalid metadata.json") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != _SCHEMA:
        raise IndexFormatError("unsupported index schema")
    if metadata.get("format_version") != _FORMAT_VERSION:
        raise IndexFormatError("unsupported index format version")
    model_identity = metadata.get("model_identity")
    dimension = metadata.get("dimension")
    if not isinstance(model_identity, str) or not model_identity:
        raise IndexFormatError("invalid model identity")
    if expected_model_identity is not None and model_identity != expected_model_identity:
        raise IndexFormatError("model identity does not match expected identity")
    if expected_dimension is not None and dimension != expected_dimension:
        raise IndexFormatError("dimension does not match expected dimension")
    _validate_metadata(metadata)
    return metadata


def _load_index_arrays(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    try:
        centroids = np.load(root / "centroids.npy", mmap_mode="r", allow_pickle=False)
        descriptors = np.load(root / "descriptors.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(root / "postings_offsets.npy", allow_pickle=False)
        indices = np.load(root / "postings_indices.npy", allow_pickle=False)
        names = json.loads((root / "names.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndexFormatError("unable to load index arrays") from exc
    return centroids, descriptors, offsets, indices, names


def _validate_loaded_arrays(
    metadata: dict[str, object],
    centroids: np.ndarray,
    descriptors: np.ndarray,
    offsets: np.ndarray,
    indices: np.ndarray,
    names: object,
) -> tuple[str, ...]:
    _validate_descriptor_arrays(metadata, centroids, descriptors)
    _validate_postings(metadata, offsets, indices)
    return _validate_loaded_names(names, int(metadata["count"]))


def _validate_descriptor_arrays(
    metadata: dict[str, object], centroids: np.ndarray, descriptors: np.ndarray
) -> None:
    if not isinstance(centroids, np.ndarray) or not isinstance(descriptors, np.ndarray):
        raise IndexFormatError("descriptor arrays are not NumPy arrays")
    if centroids.dtype != np.dtype(np.float32) or descriptors.dtype != np.dtype(np.float32):
        raise IndexFormatError("index arrays must use float32")
    nlist = int(metadata["nlist"])
    count = int(metadata["count"])
    dimension = int(metadata["dimension"])
    if centroids.shape != (nlist, dimension):
        raise IndexFormatError("centroid shape does not match metadata")
    if descriptors.shape != (count, dimension):
        raise IndexFormatError("descriptor shape does not match metadata")
    if not np.isfinite(centroids).all():
        raise IndexFormatError("index descriptors must be finite")
    _validate_stored_descriptors(descriptors, count)
    centroid_norms = np.linalg.norm(centroids, axis=1)
    if np.any(centroid_norms <= 0.0) or not np.isfinite(centroid_norms).all():
        raise IndexFormatError("stored centroids are invalid")


def _validate_stored_descriptors(descriptors: np.ndarray, count: int) -> None:
    for start in range(0, count, _DEFAULT_BATCH_SIZE):
        end = min(start + _DEFAULT_BATCH_SIZE, count)
        block = np.asarray(descriptors[start:end])
        if not np.isfinite(block).all():
            raise IndexFormatError("index descriptors must be finite")
        norms = np.linalg.norm(block, axis=1)
        if np.any(norms <= 0.0) or not np.all(
            np.abs(norms - 1.0) <= _NORMALIZATION_TOLERANCE
        ):
            raise IndexFormatError("stored descriptors are not L2-normalized")


def _validate_postings(
    metadata: dict[str, object], offsets: np.ndarray, indices: np.ndarray
) -> None:
    nlist = int(metadata["nlist"])
    count = int(metadata["count"])
    if offsets.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
        raise IndexFormatError("postings must use integer arrays")
    if offsets.shape != (nlist + 1,) or indices.shape != (count,):
        raise IndexFormatError("posting shape does not match metadata")
    if offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) < 0):
        raise IndexFormatError("posting offsets are invalid")
    if np.any(indices < 0) or np.any(indices >= count):
        raise IndexFormatError("posting indices are out of range")
    if not np.array_equal(np.sort(indices), np.arange(count, dtype=indices.dtype)):
        raise IndexFormatError("postings must contain each descriptor exactly once")
    for start, end in zip(offsets[:-1], offsets[1:]):
        if end - start > 1 and np.any(np.diff(indices[start:end]) <= 0):
            raise IndexFormatError("posting entries must be stable sorted indices")


def _validate_loaded_names(names: object, count: int) -> tuple[str, ...]:
    if not isinstance(names, list) or len(names) != count or not all(
        isinstance(name, str) and name for name in names
    ):
        raise IndexFormatError("names do not match descriptor count")
    if len(set(names)) != count or names != sorted(names):
        raise IndexFormatError("names must be unique and stable sorted identifiers")
    return tuple(names)


def _validate_metadata(metadata: dict[str, object]) -> None:
    positive_int_fields = (
        "dimension",
        "count",
        "nlist",
        "kmeans_iterations",
        "batch_size",
        "max_query_probes",
        "max_query_candidates",
    )
    for field in positive_int_fields:
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise IndexFormatError(f"invalid metadata field: {field}")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise IndexFormatError("invalid metadata field: seed")
    if metadata["nlist"] > metadata["count"]:
        raise IndexFormatError("nlist exceeds descriptor count")
    if metadata["max_query_probes"] > metadata["nlist"]:
        raise IndexFormatError("max_query_probes exceeds nlist")
    if metadata["max_query_candidates"] > metadata["count"]:
        raise IndexFormatError("max_query_candidates exceeds descriptor count")
    if metadata.get("dtype") != "float32":
        raise IndexFormatError("unsupported descriptor dtype")
    tolerance = metadata.get("normalization_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise IndexFormatError("invalid normalization tolerance")


def _verify_manifest(root: Path) -> None:
    try:
        manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndexFormatError("invalid SHA256SUMS.json") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != _MANIFEST_SCHEMA:
        raise IndexFormatError("unsupported digest manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(_REQUIRED_FILES):
        raise IndexFormatError("digest manifest does not cover required files")
    for name in _REQUIRED_FILES:
        digest = files.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise IndexFormatError(f"invalid digest for {name}")
        actual_path = root / name
        if not actual_path.is_file() or _sha256(actual_path) != digest:
            raise IndexFormatError(f"digest mismatch for {name}")


def build_reference_index(
    output_dir: str | os.PathLike[str],
    descriptors: Sequence[Sequence[float]] | np.ndarray,
    names: Iterable[str],
    *,
    model_identity: str,
    **kwargs: object,
) -> Path:
    """Functional wrapper for callers that do not need the concrete class."""
    return IVFReferenceIndex.build(
        output_dir,
        descriptors,
        names,
        model_identity=model_identity,
        **kwargs,
    )


def open_reference_index(
    index_dir: str | os.PathLike[str],
    *,
    expected_model_identity: str | None = None,
    expected_dimension: int | None = None,
) -> IVFReferenceIndex:
    """Functional wrapper for the fail-closed index opener."""
    return IVFReferenceIndex.open(
        index_dir,
        expected_model_identity=expected_model_identity,
        expected_dimension=expected_dimension,
    )


__all__ = [
    "IndexFormatError",
    "IVFReferenceIndex",
    "QueryStats",
    "ReferenceIndex",
    "ReferenceMatch",
    "build_reference_index",
    "open_reference_index",
]

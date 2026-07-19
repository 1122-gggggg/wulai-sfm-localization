"""Name-bound MegaLoc descriptor cache I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


NPY_SCHEMA = "megaloc-npy-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_names_sha256(ref_names: list[str]) -> str:
    payload = json.dumps(ref_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_names(ref_names: list[str]) -> None:
    if (not isinstance(ref_names, list) or not ref_names
            or not all(isinstance(name, str) and name for name in ref_names)
            or len(ref_names) != len(set(ref_names))):
        raise ValueError("MegaLoc ordered reference names must be unique non-empty strings")


def _validate_desc(desc: np.ndarray, ref_names: list[str]) -> np.ndarray:
    if (not isinstance(desc, np.ndarray) or desc.ndim != 2
            or desc.shape[0] != len(ref_names) or desc.shape[1] == 0
            or desc.dtype not in (np.dtype(np.float16), np.dtype(np.float32))
            or not np.isfinite(desc).all()):
        raise ValueError(
            f"MegaLoc cache has invalid shape, dtype, or values: "
            f"shape={getattr(desc, 'shape', None)} dtype={getattr(desc, 'dtype', None)}"
        )
    result = np.asarray(desc, dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("MegaLoc cache contains zero-norm descriptors")
    return np.ascontiguousarray(result / norms)


def load_megaloc_cache(path: str | Path, ref_names: list[str],
                        meta_path: str | Path | None = None) -> np.ndarray:
    """Load a named NPZ, or a NPY whose sidecar binds exact names and hashes."""
    cache = Path(path)
    _validate_names(ref_names)
    loaded = np.load(cache, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if cache.suffix.lower() != ".npz":
            loaded.close()
            raise ValueError("MegaLoc named archive must use the .npz extension")
        try:
            keys = set(loaded.files)
            portable_keys = {
                "schema_name", "schema_version", "model", "input_size", "desc", "names",
            }
            if keys == portable_keys:
                if str(np.asarray(loaded["schema_name"]).item()) != "sfm_system.megaloc_cache":
                    raise ValueError("unsupported MegaLoc NPZ schema")
                if int(np.asarray(loaded["schema_version"]).item()) != 1:
                    raise ValueError("unsupported MegaLoc NPZ schema version")
                if str(np.asarray(loaded["model"]).item()) != "MegaLoc":
                    raise ValueError("MegaLoc NPZ model metadata mismatch")
                if int(np.asarray(loaded["input_size"]).item()) <= 0:
                    raise ValueError("MegaLoc NPZ input_size must be positive")
            elif keys != {"desc", "names"}:
                raise ValueError(
                    "MegaLoc NPZ must use the legacy named schema or "
                    "sfm_system.megaloc_cache/v1"
                )
            desc = np.asarray(loaded["desc"])
            names_array = np.asarray(loaded["names"])
        finally:
            loaded.close()
        if names_array.ndim != 1:
            raise ValueError("MegaLoc NPZ names must be one-dimensional")
        names = [str(name) for name in names_array.tolist()]
        if names != ref_names:
            raise ValueError("MegaLoc NPZ ordered reference names do not exactly match the bundle")
        return _validate_desc(desc, ref_names)

    if cache.suffix.lower() != ".npy":
        raise ValueError("raw MegaLoc NPY content must use the .npy extension")
    sidecar = Path(meta_path) if meta_path else cache.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(f"MegaLoc NPY requires a binding sidecar: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MegaLoc NPY sidecar {sidecar}: {exc}") from exc
    required = {"schema", "ref_names", "shape", "cache_sha256", "names_sha256"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError(f"MegaLoc NPY sidecar is missing required fields: {sidecar}")
    if metadata["schema"] != NPY_SCHEMA:
        raise ValueError(f"unsupported MegaLoc NPY sidecar schema: {metadata['schema']!r}")
    if metadata["ref_names"] != ref_names:
        raise ValueError("MegaLoc NPY sidecar ordered reference names do not match the bundle")
    expected_names_hash = canonical_names_sha256(ref_names)
    if metadata["names_sha256"] != expected_names_hash:
        raise ValueError("MegaLoc NPY sidecar canonical names SHA-256 mismatch")
    actual_cache_hash = _file_sha256(cache)
    if metadata["cache_sha256"] != actual_cache_hash:
        raise ValueError(
            f"MegaLoc cache SHA-256 mismatch: expected {metadata['cache_sha256']}, "
            f"got {actual_cache_hash}"
        )
    desc = np.asarray(loaded)
    if metadata["shape"] != list(desc.shape):
        raise ValueError("MegaLoc NPY sidecar shape does not match the cache")
    return _validate_desc(desc, ref_names)


def write_megaloc_cache(path: str | Path, desc: np.ndarray, ref_names: list[str],
                         meta_path: str | Path | None = None,
                         metadata: dict | None = None) -> None:
    """Write canonical named NPZ, or NPY plus its required binding sidecar."""
    cache = Path(path)
    _validate_names(ref_names)
    normalized = _validate_desc(np.asarray(desc), ref_names)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.suffix.lower() == ".npz":
        with cache.open("wb") as stream:
            np.savez(stream, desc=normalized, names=np.asarray(ref_names, dtype=np.str_))
    elif cache.suffix.lower() == ".npy":
        with cache.open("wb") as stream:
            np.save(stream, normalized)
    else:
        raise ValueError("MegaLoc cache path must end in .npz or .npy")

    if cache.suffix.lower() == ".npy" or meta_path is not None:
        sidecar = Path(meta_path) if meta_path else cache.with_suffix(".json")
        payload = dict(metadata or {})
        payload.update({
            "schema": NPY_SCHEMA if cache.suffix.lower() == ".npy" else "megaloc-npz-v1",
            "ref_names": ref_names,
            "shape": list(normalized.shape),
            "cache_sha256": _file_sha256(cache),
            "names_sha256": canonical_names_sha256(ref_names),
        })
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

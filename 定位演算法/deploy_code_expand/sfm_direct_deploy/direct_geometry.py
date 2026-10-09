"""Precomputed map geometry, consumed only through the bundle's digest chain."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GEOMETRY_ARCHIVE = "direct_geometry.npz"
GEOMETRY_SCHEMA = "direct-geometry/v1"


def prepare_geometry_archive(assets, destination: Path) -> None:
    """Run the authoritative geometry loader once, without loading any models."""
    from sfm_diagnosis.site_pipeline.deployment_localizer import FinalMapEDMProvider

    provider = object.__new__(FinalMapEDMProvider)
    provider.map_model = assets.model_dir
    provider._keyframes = assets.keyframe_index()
    provider._reference_names = ()
    provider._geometry_locked = False
    # Store every reference. Runtime applies the active profile's selection.
    provider._drop_empty_side_references = lambda: None
    provider._load_geometry()
    names = provider._reference_names
    offsets = np.r_[0, np.cumsum([len(provider._observations[n][0]) for n in names])]
    metadata = json.dumps(
        {"schema": GEOMETRY_SCHEMA, "model_sha256": dict(assets.raw["model_sha256"])}
    )
    with destination.open("wb") as output:
        np.savez(
            output,
            metadata=np.asarray(metadata),
            names=np.asarray(names),
            point_ids=provider._point_ids,
            point_xyz=provider._point_xyz,
            observation_offsets=offsets,
            observation_xy=np.concatenate([provider._observations[n][0] for n in names]),
            observation_ids=np.concatenate([provider._observations[n][1] for n in names]),
            occupied_bins=np.asarray(provider._reference_occupied_bins, dtype=np.int64),
            cam_from_world=np.asarray([provider._cam_from_world[n] for n in names]),
            native_wh=np.asarray([provider._native_wh[n] for n in names], dtype=np.int64),
            camera_params=np.asarray([provider._camera_params[n] for n in names]),
            median_sparse_z=np.asarray([provider._median_sparse_z[n] for n in names]),
        )


def load_geometry_archive(provider, path: Path, model_sha256: dict) -> None:
    """Restore verified arrays; derive paths and sessions from current keyframes."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata != {"schema": GEOMETRY_SCHEMA, "model_sha256": dict(model_sha256)}:
            raise ValueError("precomputed geometry does not match the map model/schema")
        names = tuple(str(name) for name in archive["names"])
        if not names or len(set(names)) != len(names):
            raise ValueError("precomputed geometry requires unique reference names")
        if set(names) - provider._keyframes.keys():
            raise ValueError("precomputed reference identity is absent from keyframes")
        offsets = archive["observation_offsets"]
        xy, ids = archive["observation_xy"], archive["observation_ids"]
        if (
            offsets.shape != (len(names) + 1,)
            or offsets[0] != 0
            or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(ids)
            or xy.shape != (len(ids), 2)
        ):
            raise ValueError("precomputed geometry has inconsistent observation offsets")
        point_ids, point_xyz = archive["point_ids"], archive["point_xyz"]
        if point_ids.ndim != 1 or point_xyz.shape != (len(point_ids), 3):
            raise ValueError("precomputed geometry has inconsistent point arrays")
        fields = {
            "occupied_bins": (),
            "cam_from_world": (3, 4),
            "native_wh": (2,),
            "camera_params": (4,),
            "median_sparse_z": (),
        }
        rows = {name: archive[name] for name in fields}
        if any(rows[key].shape != (len(names), *shape) for key, shape in fields.items()):
            raise ValueError("precomputed geometry has inconsistent reference arrays")

    provider._reference_names = names
    provider._reference_sessions = tuple(str(provider._keyframes[n]["video_id"]) for n in names)
    provider._reference_paths = tuple(
        Path(provider._keyframes[n]["image_uri"]).resolve(strict=True) for n in names
    )
    provider._point_ids, provider._point_xyz = point_ids, point_xyz
    provider._observations = {
        name: (xy[offsets[i] : offsets[i + 1]], ids[offsets[i] : offsets[i + 1]])
        for i, name in enumerate(names)
    }
    provider._reference_occupied_bins = tuple(int(v) for v in rows["occupied_bins"])
    provider._cam_from_world = dict(zip(names, rows["cam_from_world"]))
    provider._native_wh = {
        n: tuple(int(v) for v in row) for n, row in zip(names, rows["native_wh"])
    }
    provider._camera_params = {
        n: tuple(float(v) for v in row) for n, row in zip(names, rows["camera_params"])
    }
    provider._median_sparse_z = {n: float(v) for n, v in zip(names, rows["median_sparse_z"])}
    provider._drop_empty_side_references()
    provider._geometry_locked = True

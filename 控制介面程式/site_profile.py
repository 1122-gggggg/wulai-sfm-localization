#!/usr/bin/env python3
"""Load one atomic set of site-specific localization and mission assets."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1


LOCALIZER_BACKENDS = ("xfeat", "edm")


@dataclass(frozen=True)
class QueryCamera:
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class SiteProfile:
    source: Path
    site_id: str
    display_name: str
    map_ply: Path
    route_json: Path | None
    localization_bundle: Path
    # Local matcher / tracker family. Retrieval (MegaLoc) is shared.
    # "xfeat" = ProductionXFeatTracker; "edm" = ProductionEDMTracker adapter.
    localizer: str = "xfeat"
    localizer_deploy_dir: Path | None = None
    localizer_profile: Path | None = None
    map_reference_poses: Path | None = None
    megaloc_cache: Path | None = None
    track_landmarks: Path | None = None
    poles_json: Path | None = None
    query_camera: QueryCamera | None = None


def _resolve_asset(profile_dir: Path, value, key: str, *, required: bool) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"site profile asset {key!r} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"site profile asset {key!r} must be a path string or null")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = profile_dir / path
    return path.resolve()


def _load_query_camera(raw, source: Path) -> QueryCamera | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"site profile query_camera must be an object: {source}")
    model = raw.get("model")
    width = raw.get("width")
    height = raw.get("height")
    params = raw.get("params")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"site profile query_camera.model must be non-empty: {source}")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError(f"site profile query_camera.width must be > 0: {source}")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError(f"site profile query_camera.height must be > 0: {source}")
    if not isinstance(params, list) or not params:
        raise ValueError(f"site profile query_camera.params must be a non-empty list: {source}")
    try:
        parsed_params = tuple(float(value) for value in params)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"site profile query_camera.params must contain numbers: {source}"
        ) from exc
    if not all(math.isfinite(value) for value in parsed_params):
        raise ValueError(f"site profile query_camera.params must be finite: {source}")
    return QueryCamera(
        model=model.strip().upper(),
        width=width,
        height=height,
        params=parsed_params,
    )


def load_site_profile(path: str | Path, *, validate_files: bool = True) -> SiteProfile:
    """Load a profile and resolve relative asset paths beside the JSON file."""
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read site profile {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid site profile JSON {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"site profile root must be an object: {source}")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"site profile schema_version must be {SCHEMA_VERSION}: {source}"
        )
    site_id = raw.get("site_id")
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError(f"site profile site_id must be a non-empty string: {source}")
    display_name = raw.get("display_name", site_id)
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError(f"site profile display_name must be a non-empty string: {source}")
    localizer = raw.get("localizer", "xfeat")
    if not isinstance(localizer, str) or localizer.strip().lower() not in LOCALIZER_BACKENDS:
        raise ValueError(
            f"site profile localizer must be one of {LOCALIZER_BACKENDS}: {source}"
        )
    localizer = localizer.strip().lower()
    assets = raw.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(f"site profile assets must be an object: {source}")

    base = source.parent
    profile = SiteProfile(
        source=source,
        site_id=site_id.strip(),
        display_name=display_name.strip(),
        map_ply=_resolve_asset(base, assets.get("map_ply"), "map_ply", required=True),
        route_json=_resolve_asset(
            base, assets.get("route_json"), "route_json", required=False
        ),
        localization_bundle=_resolve_asset(
            base,
            assets.get("localization_bundle"),
            "localization_bundle",
            required=True,
        ),
        localizer=localizer,
        localizer_deploy_dir=_resolve_asset(
            base,
            raw.get("localizer_deploy_dir"),
            "localizer_deploy_dir",
            required=False,
        ),
        localizer_profile=_resolve_asset(
            base,
            raw.get("localizer_profile"),
            "localizer_profile",
            required=False,
        ),
        map_reference_poses=_resolve_asset(
            base,
            raw.get("map_reference_poses"),
            "map_reference_poses",
            required=False,
        ),
        megaloc_cache=_resolve_asset(
            base, assets.get("megaloc_cache"), "megaloc_cache", required=False
        ),
        track_landmarks=_resolve_asset(
            base, assets.get("track_landmarks"), "track_landmarks", required=False
        ),
        poles_json=_resolve_asset(
            base, assets.get("poles_json"), "poles_json", required=False
        ),
        query_camera=_load_query_camera(raw.get("query_camera"), source),
    )
    if validate_files:
        missing = [
            f"{name}={asset}"
            for name, asset in (
                ("map_ply", profile.map_ply),
                ("route_json", profile.route_json),
                ("localization_bundle", profile.localization_bundle),
                ("megaloc_cache", profile.megaloc_cache),
                ("track_landmarks", profile.track_landmarks),
                ("poles_json", profile.poles_json),
                ("map_reference_poses", profile.map_reference_poses),
                ("localizer_profile", profile.localizer_profile),
            )
            if asset is not None and not asset.is_file()
        ]
        if profile.localizer_deploy_dir is not None and not profile.localizer_deploy_dir.is_dir():
            missing.append(
                "localizer_deploy_dir=" + str(profile.localizer_deploy_dir)
            )
        if missing:
            raise ValueError(
                f"site profile {source} has missing asset(s): " + "; ".join(missing)
            )
    return profile

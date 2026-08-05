"""Shared production localizer construction for the UI worker and flight runner."""
from __future__ import annotations

import math
from numbers import Real
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from artifact_integrity import verify_sha256
from edm_profile import apply_edm_tracker_profile, load_edm_production_profile


XFEAT_CAMERA_720 = (
    "FULL_OPENCV",
    1280,
    720,
    [
        960.4853099760471,
        958.1961747147875,
        670.8167651412149,
        358.7191813450141,
        -0.016359355216362784,
        0.256336300878371,
        -0.006099082030819077,
        0.019509803298460405,
        -0.1198628127364991,
        0.0,
        0.0,
        0.0,
    ],
)


@dataclass(frozen=True)
class BuiltLocalizer:
    tracker: object
    reloc_map: object
    camera: object
    config: object
    backend: str
    variant: str
    device: str


def validate_camera_tuple(camera_tuple) -> tuple[str, int, int, list[float]]:
    if not isinstance(camera_tuple, (list, tuple)) or len(camera_tuple) != 4:
        raise ValueError("query camera must be (model, width, height, params)")
    model, width, height, params = camera_tuple
    if not isinstance(model, str) or not model.strip():
        raise ValueError("query camera model must be a non-empty string")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("query camera dimensions must be positive integers")
    if (
        not isinstance(params, (list, tuple))
        or not params
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in params
        )
    ):
        raise ValueError("query camera params must be finite numbers")
    parsed = (
        model.strip().upper(),
        width,
        height,
        [float(value) for value in params],
    )
    if not parsed[0] or not parsed[3]:
        raise ValueError("query camera needs a model, positive dimensions, and parameters")
    try:
        camera = pycolmap.Camera(
            model=parsed[0],
            width=parsed[1],
            height=parsed[2],
            params=parsed[3],
        )
        if not camera.verify_params():
            raise ValueError(
                f"{parsed[0]} expects parameters {camera.params_info}"
            )
    except Exception as exc:
        raise ValueError(f"invalid pycolmap query camera: {exc}") from exc
    return parsed


def production_xfeat_config():
    """Pinned production XFeat/LighterGlue configuration."""
    from production_xfeat_tracker import ProductionConfig

    return ProductionConfig(
        matcher_mode="lighterglue",
        acquire_matcher_mode="lighterglue",
        nn_min_score=0.85,
        min_conf=0.1,
        adaptive_first_topk=3,
        adaptive_accept_inliers=100,
        adaptive_accept_reproj=3.5,
        local_topk=5,
        weak_local_topk=8,
        near_pool=24,
        covis_per_ref=20,
        radius=0.8,
        max_yaw_diff_deg=90.0,
        xfeat_topk_track=1700,
        xfeat_topk_acquire=2048,
        boot_global_topk=30,
        lost_global_topk=30,
        weak_global_topk=0,
        acquire_min_inliers=80,
        track_min_inliers=50,
        weak_min_inliers=30,
        good_inliers=80,
        pnp_ransac_max_error=5.0,
        pnp_ransac_random_seed=-1,
        max_reproj_error_track=6.0,
        max_reproj_error_acquire=5.0,
        acquire_max_jump=1.25,
        acquire_max_yaw_diff_deg=90.0,
        max_jump=2.0,
        max_speed_mps=10.0,
        jump_slack_m=0.4,
        lost_pure_global_after_s=3.0,
        weak_after=2,
        lost_after=2,
        max_corr_total=0,
        max_corr_per_ref=0,
        dedup_corr=False,
        flow_enabled=False,
        temporal_cache_enabled=True,
        temporal_cache_seed_mode="full_ref",
        temporal_cache_min_anchors=80,
        temporal_cache_max_anchors=2048,
        temporal_cache_max_age=2,
        temporal_cache_min_score=0.85,
        temporal_cache_seed_min_inliers=150,
        temporal_cache_seed_max_reproj=3.5,
    )


def build_production_localizer(
    *,
    backend: str,
    bundle: str | Path,
    frame_source,
    camera_tuple,
    bundle_sha256: str | None = None,
    megaloc_cache: str | Path | None = None,
    production_profile: str | Path | None = None,
    production_profile_sha256: str | None = None,
    matcher_mode: str = "",
    local_topk: int = 0,
) -> BuiltLocalizer:
    """Build one verified production tracker without allocating models twice."""
    selected = str(backend).strip().lower()
    if selected not in {"edm", "xfeat"}:
        raise ValueError(f"unsupported production localizer backend: {backend!r}")
    camera_spec = validate_camera_tuple(camera_tuple)

    if selected == "edm":
        from edm_localizer_adapter import (
            EDMTrackerAdapter,
            production_edm_config,
        )
        from edm_matcher import EDMMatcher
        from reloc_localizer_edm import Camera, DEVICE, EDMRelocMap

        reloc_map = EDMRelocMap.load(bundle, expected_sha256=bundle_sha256 or None)
        config = production_edm_config()
        matcher = None
        profile_name = "defaults"
        matcher_tag = "torch_fp16_defaults"
        if production_profile:
            verify_sha256(production_profile, production_profile_sha256 or None)
            profile = load_edm_production_profile(production_profile)
            apply_edm_tracker_profile(config, profile)
            matcher_cfg = profile["matcher"]
            matcher = EDMMatcher(
                mconf_thr=float(matcher_cfg["mconf_thr"]),
                topk=int(matcher_cfg["coarse_topk"]),
                fp16=bool(matcher_cfg["fp16"]),
                reference_cache_size=int(matcher_cfg["reference_cache_size"]),
            )
            profile_name = str(profile.get("name") or "unnamed")
            matcher_tag = (
                f"torch_{'fp16' if matcher_cfg['fp16'] else 'fp32'}"
                f"_coarse{matcher_cfg['coarse_topk']}"
            )
        if local_topk > 0:
            config.local_topk = int(local_topk)
            config.weak_local_topk = max(config.weak_local_topk, config.local_topk)
            config.validate()
        camera = Camera(*camera_spec)
        tracker = EDMTrackerAdapter(
            reloc_map,
            camera,
            cfg=config,
            matcher=matcher,
            frame_source=frame_source,
        )
        return BuiltLocalizer(
            tracker=tracker,
            reloc_map=reloc_map,
            camera=camera,
            config=config,
            backend=selected,
            variant=(
                f"production_edm_{profile_name}_topk{config.local_topk}_{matcher_tag}"
            ),
            device=DEVICE,
        )

    from production_xfeat_tracker import MegaLocLayer, ProductionXFeatTracker
    from reloc_localizer_xfeat import Camera, DEVICE, XFeatRelocMap

    reloc_map = XFeatRelocMap.load(str(bundle), bundle_sha256 or None)
    if (
        reloc_map.ref_centers is None
        or len(reloc_map.ref_centers) != len(reloc_map.ref_names)
    ):
        raise ValueError(
            "bundle tracking metadata mismatch: "
            f"refs={len(reloc_map.ref_names)} "
            f"ref_centers={0 if reloc_map.ref_centers is None else len(reloc_map.ref_centers)}"
        )
    cache = Path(megaloc_cache) if megaloc_cache else None
    if cache is not None and cache.is_file():
        megaloc = MegaLocLayer.load_cache(
            cache,
            reloc_map.ref_names,
            input_size=322,
            device=DEVICE,
        )
    else:
        megaloc = MegaLocLayer(
            reloc_map.ref_global,
            input_size=322,
            device=DEVICE,
        )
    config = production_xfeat_config()
    if matcher_mode:
        config.matcher_mode = str(matcher_mode)
    if local_topk > 0:
        config.local_topk = int(local_topk)
        config.weak_local_topk = max(config.weak_local_topk, config.local_topk + 1)
        config.adaptive_first_topk = min(config.adaptive_first_topk, config.local_topk)
    camera = Camera(*camera_spec)
    tracker = ProductionXFeatTracker(
        reloc_map,
        megaloc,
        frame_source=frame_source,
        query_cam=camera,
        cfg=config,
    )
    return BuiltLocalizer(
        tracker=tracker,
        reloc_map=reloc_map,
        camera=camera,
        config=config,
        backend=selected,
        variant=f"matcher_{config.matcher_mode}_topk{config.local_topk}",
        device=DEVICE,
    )

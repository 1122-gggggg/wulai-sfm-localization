"""Shared production localizer construction for the UI worker and flight runner."""

from __future__ import annotations

import math
import shutil
import uuid
from numbers import Real
from pathlib import Path

import pycolmap

from artifact_integrity import expected_sha256
from localizer_registry import get_localizer_provider, register_localizer_provider
from pose_types import BuiltLocalizer, Camera, LocalizerProvider


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
            raise ValueError(f"{parsed[0]} expects parameters {camera.params_info}")
    except Exception as exc:
        raise ValueError(f"invalid pycolmap query camera: {exc}") from exc
    return parsed


DIRECT_DEPLOY_DIR = Path(__file__).resolve().parents[1] / "sfm_direct_deploy"


def _import_direct_runtime():
    """Import the direct deployment package for the single production backend."""
    if not DIRECT_DEPLOY_DIR.is_dir():
        raise ValueError(
            f"direct localizer deployment directory is missing: {DIRECT_DEPLOY_DIR}"
        )
    import sys

    if str(DIRECT_DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(DIRECT_DEPLOY_DIR))
    from direct_paths import default_cache_dir, default_runtime_paths, vendor_sys_path

    vendor_sys_path()
    from direct_localizer_adapter import DirectTrackerAdapter
    from direct_map import DirectMapAssets
    from direct_profile import load_direct_profile
    from live_provider import LiveMapEDMProvider

    return (
        DirectMapAssets,
        load_direct_profile,
        LiveMapEDMProvider,
        DirectTrackerAdapter,
        default_runtime_paths,
        default_cache_dir,
    )


def _prune_empty_workmap_sessions(root: Path) -> None:
    """Drop default-location sessions that never stored a keyframe image.

    Every worker start creates a session; restarts and ground-only runs leave
    metadata-only directories. Sessions with any keyframe are post-flight
    evidence and are never removed here.
    """
    if not root.is_dir():
        return
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        if any((candidate / "keyframes").glob("*.png")):
            continue
        shutil.rmtree(candidate, ignore_errors=True)


def _build_direct_localizer(
    *,
    bundle: str | Path,
    frame_source,
    camera_tuple,
    bundle_sha256: str | None = None,
    megaloc_cache: str | Path | None = None,
    reference_index: str | Path | None = None,
    reference_index_sha256: str | None = None,
    vpr_weights: str | Path | None = None,
    vpr_weights_sha256: str | None = None,
    production_profile: str | Path | None = None,
    production_profile_sha256: str | None = None,
    matcher_mode: str = "",
    local_topk: int = 0,
    allow_profile_defaults: bool = False,
    map_frame=None,
    workmap_session_dir: str | Path | None = None,
    workmap_eval_stage: str | None = None,
) -> BuiltLocalizer:
    if not production_profile:
        # Unlike EDM there is no built-in threshold set to fall back to: the
        # direct-deployment-profile IS the frozen P174 configuration (map scale,
        # PnP gates, reference-bank occupancy floor). allow_profile_defaults
        # therefore cannot make this safe and is deliberately not honoured.
        raise ValueError(
            "the direct localizer backend requires a SHA-verified "
            "direct-deployment-profile/v1 JSON (production_profile=...); it has no "
            "built-in thresholds, so allow_profile_defaults cannot substitute for one"
        )
    del allow_profile_defaults
    # Second gate behind LocalizerCapabilities.unsupported_assets: retrieval is the
    # bundle's own MegaLoc bank, so an operator who wires a BoQ cache or an IVF
    # index here must be told, not silently ignored.
    for value, label in (
        (megaloc_cache, "megaloc_cache"),
        (reference_index, "reference_index"),
        (reference_index_sha256, "reference_index_sha256"),
        (matcher_mode, "matcher_mode"),
        (vpr_weights, "vpr_weights"),
        (vpr_weights_sha256, "vpr_weights_sha256"),
    ):
        if value:
            raise ValueError(
                f"the direct localizer backend does not support {label}: retrieval and "
                "matching are pinned by the bundle's MegaLoc bank and the deployment profile"
            )
    if int(local_topk) > 0:
        raise ValueError(
            "the direct localizer backend does not support local_topk overrides: "
            "reloc.top_k is pinned by the SHA-verified deployment profile"
        )

    (
        DirectMapAssets,
        load_direct_profile,
        LiveMapEDMProvider,
        DirectTrackerAdapter,
        default_runtime_paths,
        default_cache_dir,
    ) = _import_direct_runtime()

    assets = DirectMapAssets.load(
        bundle,
        expected_sha256=expected_sha256(bundle, bundle_sha256),
    )
    profile = load_direct_profile(
        production_profile,
        expected_sha256=production_profile_sha256 or None,
    )
    if workmap_eval_stage == "full" and not profile.online_workmap.enabled:
        raise ValueError(
            "workmap evaluation stage 'full' requires online_workmap.enabled=true"
        )
    if (
        workmap_eval_stage == "full"
        or (workmap_eval_stage is None and profile.online_workmap.enabled)
    ) and not production_profile_sha256:
        raise ValueError(
            "the full online workmap stage requires a trusted SHA-256 for the enabled profile"
        )
    runtime = default_runtime_paths()
    cache_dir = default_cache_dir()
    if workmap_session_dir:
        session_dir = Path(workmap_session_dir).expanduser().resolve()
    else:
        sessions_root = cache_dir / "workmap_sessions"
        _prune_empty_workmap_sessions(sessions_root)
        session_dir = (sessions_root / uuid.uuid4().hex).resolve()
    release_root = assets.root.resolve()
    try:
        session_dir.relative_to(release_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"workmap session directory must be outside the base release: {session_dir}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    provider = LiveMapEDMProvider(
        assets,
        profile,
        edm_repo=runtime.edm_repo,
        edm_checkpoint=runtime.edm_checkpoint,
        boq_weights=runtime.boq_weights,
        cache_dir=cache_dir,
    )
    camera = Camera(*camera_tuple)
    tracker = DirectTrackerAdapter(
        assets,
        profile,
        provider,
        map_frame=map_frame,
        frame_source=frame_source,
        workmap_session_dir=session_dir,
        workmap_eval_stage=workmap_eval_stage,
    )
    return BuiltLocalizer(
        tracker=tracker,
        reloc_map=assets,
        camera=camera,
        config=profile,
        backend="direct",
        variant=(
            f"direct_{profile.name}_topk{profile.reloc.top_k}_megaloc_edm_pnp_klt"
            + (f"_workmap_{tracker.workmap_eval_stage}" if tracker.workmap_eval_stage != "baseline" else "")
        ),
        device="cuda",
    )


register_localizer_provider(
    LocalizerProvider(
        name="direct",
        capabilities=get_localizer_provider("direct").capabilities,
        builder=_build_direct_localizer,
    )
)


def build_production_localizer(
    *,
    backend: str,
    bundle: str | Path,
    frame_source,
    camera_tuple,
    bundle_sha256: str | None = None,
    megaloc_cache: str | Path | None = None,
    reference_index: str | Path | None = None,
    reference_index_sha256: str | None = None,
    vpr_weights: str | Path | None = None,
    vpr_weights_sha256: str | None = None,
    production_profile: str | Path | None = None,
    production_profile_sha256: str | None = None,
    matcher_mode: str = "",
    local_topk: int = 0,
    allow_profile_defaults: bool = False,
    map_frame=None,
    workmap_session_dir: str | Path | None = None,
    workmap_eval_stage: str | None = None,
) -> BuiltLocalizer:
    """Build one verified production tracker through the named provider registry."""
    if str(backend).strip().lower() != "direct":
        raise ValueError(
            f"unsupported localizer backend {backend!r}; only 'direct' is supported"
        )
    provider = get_localizer_provider(backend)
    if production_profile and not provider.capabilities.supports_production_profile:
        raise ValueError(f"localizer backend {provider.name!r} does not support production_profile")
    camera_spec = validate_camera_tuple(camera_tuple)
    return provider.build(
        bundle=bundle,
        frame_source=frame_source,
        camera_tuple=camera_spec,
        bundle_sha256=bundle_sha256,
        megaloc_cache=megaloc_cache,
        reference_index=reference_index,
        reference_index_sha256=reference_index_sha256,
        vpr_weights=vpr_weights,
        vpr_weights_sha256=vpr_weights_sha256,
        production_profile=production_profile,
        production_profile_sha256=production_profile_sha256,
        matcher_mode=matcher_mode,
        local_topk=local_topk,
        allow_profile_defaults=allow_profile_defaults,
        map_frame=map_frame,
        workmap_session_dir=workmap_session_dir,
        workmap_eval_stage=workmap_eval_stage,
    )

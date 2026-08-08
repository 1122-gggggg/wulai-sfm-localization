#!/usr/bin/env python3
"""Persistent stdin/stdout production localizer worker.

Protocol:
  input  : optional SFM1 mode header, then raw RGB bytes or one shared-memory slot byte
  output : one JSON line per frame

All model logs are redirected to stderr so stdout remains machine-readable.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from multiprocessing import resource_tracker, shared_memory
import os
import sys
import time
from pathlib import Path

import numpy as np

from live_localizer_protocol import (
    HEADER_SIZE,
    TIMED_HEADER_SIZE,
    TIMED_MAGIC,
    decode_request,
)


# Workspace layout (physical dirs beneath the repository root).
import sys as _sys
_CTRL = Path(__file__).resolve().parents[1]
if str(_CTRL) not in _sys.path:
    _sys.path.insert(0, str(_CTRL))
from workspace_layout import workspace_from_file  # noqa: E402
from backend_contract import InterfaceMode  # noqa: E402
from runtime_safety import (  # noqa: E402
    configure_offline_environment,
    install_network_guard,
)

_WS = workspace_from_file(__file__)
SYSTEM_ROOT = _WS.root  # workspace root (replaces old sfm_system parent layout)
LOC_ROOT = _WS.algorithms  # algorithmic root
DEPLOY_DIR = _WS.deploy_code
if not DEPLOY_DIR.exists():
    DEPLOY_DIR = _WS.algorithms / "source" / "sfm_glomap" / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))
VALIDATION_DIR = _WS.validation
PACKAGE_ROOT = _WS.runtime
DEFAULT_BUNDLE = Path(os.environ.get(
    "SFM_RELOC_BUNDLE",
    str(_WS.bundles / "your_site_reloc_map_edm.pt"),
))
DEFAULT_MEGALOC = os.environ.get(
    "SFM_MEGALOC_CACHE",
    "",
)
DEFAULT_NEUFLOW_REPO = Path(os.environ.get(
    "SFM_NEUFLOW_REPO",
    str(_WS.root / ".experiment_deps" / "neuflow_v2"),
))
DEFAULT_NEUFLOW_WEIGHTS = Path(os.environ.get(
    "SFM_NEUFLOW_WEIGHTS",
    str(DEFAULT_NEUFLOW_REPO / "neuflow_mixed.pth"),
))

from edm_profile import (  # noqa: E402
    apply_edm_tracker_profile,
    load_edm_production_profile,
)


def resolve_localizer_backend(requested: str, bundle: Path) -> str:
    """Pick edm|xfeat from an explicit flag, else from the bundle file name."""
    req = (requested or "auto").strip().lower()
    if req in {"edm", "xfeat"}:
        return req
    if req not in {"", "auto"}:
        raise ValueError(f"unsupported localizer backend: {requested!r}")
    name = bundle.name.lower()
    if "edm" in name:
        return "edm"
    return "xfeat"


def require_edm_cuda(torch_module) -> None:
    """Fail clearly instead of letting the production EDM matcher crash mid-load."""
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "EDM production localization requires a working CUDA device; "
            "check nvidia-smi and reboot after an NVIDIA driver update"
        )


def attach_frame_shm(name: str):
    """Attach the operator's frame buffer without taking ownership of it.

    CPython registers every SharedMemory it touches with resource_tracker, which
    unlinks the segment when *this* process dies -- even though the operator
    created it and is still using it. So killing a worker (a stalled one gets
    SIGTERM while it is still loading its bundle) destroyed the buffer, and every
    replacement worker then died on FileNotFoundError attaching a segment that no
    longer existed. The operator only creates the segment once, so nothing ever
    recovered and localization stopped entirely.

    The creator stays responsible for unlinking; we only detach.
    """
    shm = shared_memory.SharedMemory(name=name, create=False)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
    except Exception:
        # Older/newer CPython may not track it; attaching still works either way.
        pass
    return shm


def read_exact_into(stream, data: bytearray) -> int:
    """Fill an existing buffer and return the number of bytes received."""
    view = memoryview(data)
    offset = 0
    readinto = getattr(stream, "readinto", None)
    while offset < len(data):
        if callable(readinto):
            count = readinto(view[offset:])
        else:
            chunk = stream.read(len(data) - offset)
            count = len(chunk)
            if count:
                view[offset:offset + count] = chunk
        if not count:
            break
        offset += int(count)
    view.release()
    return offset


def read_exact(stream, size: int) -> bytearray:
    """Compatibility helper that allocates once and fills through readinto()."""
    data = bytearray(int(size))
    offset = read_exact_into(stream, data)
    return data if offset == len(data) else data[:offset]


def frame_view_from_rgb_bytes(
    raw: bytes | bytearray | memoryview, width: int, height: int,
) -> np.ndarray:
    """Return a zero-copy RGB view valid for as long as ``raw`` is referenced."""
    expected = int(width) * int(height) * 3
    if len(raw) != expected:
        raise ValueError(f"RGB frame size mismatch: {len(raw)}/{expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape((int(height), int(width), 3))


def frame_view_from_shared_memory(buffer, slot: int, slots: int,
                                  width: int, height: int) -> np.ndarray:
    """Return the selected zero-copy frame slot after validating its bounds."""
    slot = int(slot)
    slots = int(slots)
    if slot < 0 or slot >= slots:
        raise ValueError(f"shared frame slot {slot} outside [0,{slots})")
    frame_size = int(width) * int(height) * 3
    start = slot * frame_size
    return frame_view_from_rgb_bytes(buffer[start:start + frame_size], width, height)


def resolve_force_track_ref(requested: int, ref_count: int) -> int:
    if ref_count <= 0:
        raise ValueError("bundle has no reference centers")
    requested = int(requested)
    if requested == -1:
        return int(ref_count) // 2
    if requested < 0 or requested >= int(ref_count):
        raise ValueError(
            f"--force-track-ref {requested} out of range [0,{int(ref_count) - 1}] (or use -1)")
    return requested


def resolve_query_camera_override(
    model: str,
    width: int,
    height: int,
    params: list[float] | None,
    stream_width: int,
    stream_height: int,
) -> tuple[str, int, int, list[float]] | None:
    provided = bool(model) or width != 0 or height != 0 or params is not None
    if not provided:
        return None
    if not model or width <= 0 or height <= 0 or not params:
        raise ValueError(
            "query camera override requires model, positive width/height, and params"
        )
    if width != stream_width or height != stream_height:
        raise ValueError(
            "query camera resolution must match the worker stream: "
            f"camera={width}x{height} stream={stream_width}x{stream_height}"
        )
    from production_localizer_factory import validate_camera_tuple

    return validate_camera_tuple((model, width, height, params))


def apply_xfeat_runtime_overrides(cfg, *, matcher_mode: str, local_topk: int) -> None:
    if matcher_mode:
        cfg.matcher_mode = str(matcher_mode)
    else:
        cfg.matcher_mode = "lighterglue"
        cfg.acquire_matcher_mode = "lighterglue"
    if local_topk > 0:
        cfg.local_topk = int(local_topk)
        cfg.weak_local_topk = max(
            int(cfg.weak_local_topk),
            int(local_topk) + 1,
            int(cfg.local_topk) + 1,
        )
        # Keep the validated adaptive first pass. local_topk is the candidate
        # budget; the fourth ref is added only when the first three are weak.
        cfg.adaptive_first_topk = min(int(cfg.adaptive_first_topk), cfg.local_topk)


def apply_runtime_benchmark_mode(tracker, mode: str, previous_mode: str,
                                 seed_local_prior) -> bool:
    """Prepare one forced benchmark branch at a frame boundary.

    Returns True when a fixed local prior was seeded for this frame. ``auto``
    leaves an already-automatic tracker untouched; leaving a forced mode resets
    to a clean BOOT_INIT state so synthetic priors cannot leak into production.
    """
    if mode not in {"auto", "global", "weak", "track"}:
        raise ValueError(f"unsupported localization benchmark mode: {mode!r}")

    def reset_to_boot() -> None:
        tracker._clear_tracking_history()
        tracker.state = type(tracker.state)()

    if mode == "auto":
        if previous_mode != "auto":
            reset_to_boot()
        return False

    if mode == "global":
        if previous_mode != "global":
            reset_to_boot()
        tracker.state.mode = "LOST"
        tracker.state.fail_count = 0
        tracker.state.bad_count = 0
        return False

    target = "WEAK_TRACK" if mode == "weak" else "TRACK"
    needs_prior = (
        previous_mode != mode
        or tracker.state.last_center is None
        or not tracker.state.last_refs
    )
    if previous_mode != mode:
        reset_to_boot()
    if needs_prior:
        seed_local_prior(target)
        return True
    tracker.state.mode = target
    tracker.state.fail_count = 0
    tracker.state.bad_count = 0
    return False


def localization_exception_payload(
    *,
    seq: int,
    error: BaseException | str,
    frame_id: str | None = None,
    capture_mono_ns: int | None = None,
    pose_mono_ns: int | None = None,
) -> dict:
    """Build the explicit LOST result used for tracker-side exceptions."""
    now_ns = time.monotonic_ns()
    return {
        "seq": int(seq),
        "frame_id": str(frame_id or f"worker-{int(seq)}"),
        "capture_mono_ns": int(capture_mono_ns or now_ns),
        "pose_mono_ns": int(pose_mono_ns or now_ns),
        "localization_contract_version": 1,
        "validity": False,
        "confidence": 0.0,
        "success": False,
        "mode": "LOST",
        "next_mode": "LOST",
        "localization_exception": True,
        "error": error if isinstance(error, str) else repr(error),
    }


@contextlib.contextmanager
def redirect_native_stdout_to_stderr():
    """Route native fd-1 logs away from the JSON stdout pipe."""
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def main() -> None:
    configure_offline_environment()
    install_network_guard(InterfaceMode.SIMULATED_STREAM)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--query-camera-model", default="")
    ap.add_argument("--query-camera-width", type=int, default=0)
    ap.add_argument("--query-camera-height", type=int, default=0)
    ap.add_argument("--query-camera-params", type=float, nargs="+", default=None)
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument(
        "--map-align",
        default="",
        help="Measured sfm-align/v2 T_align_gravity.json for public pose yaw.",
    )
    ap.add_argument(
        "--bundle-sha256",
        default="",
        help="trusted SHA-256 for the selected localization bundle",
    )
    ap.add_argument("--megaloc-cache", default=DEFAULT_MEGALOC)
    ap.add_argument("--deploy-dir", default=str(DEPLOY_DIR))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--frame-shm-name", default="")
    ap.add_argument("--frame-shm-slots", type=int, default=0)
    ap.add_argument(
        "--startup-handshake",
        action="store_true",
        help="Emit one JSON ready event after models are loaded, before reading frames.",
    )
    ap.add_argument(
        "--localizer-backend",
        choices=("auto", "edm", "xfeat"),
        default=os.environ.get("SFM_LOCALIZER_BACKEND", "auto"),
        help=(
            "Local matcher family: edm (ProductionEDMTracker) or xfeat "
            "(ProductionXFeatTracker). auto picks from the bundle file name."
        ),
    )
    ap.add_argument(
        "--neuflow-track",
        action="store_true",
        help="XFeat only: NeuFlow-v2 refresh=3 hybrid for TRACK; keep deep fallback.",
    )
    ap.add_argument(
        "--projection-track",
        action="store_true",
        help="XFeat only: projection-guided TRACK fast path with deep fallback.",
    )
    ap.add_argument("--track-landmarks", default="")
    ap.add_argument(
        "--matcher-mode",
        choices=("", "nn_then_lg", "lighterglue"),
        default="",
        help=(
            "XFeat only: override TRACK/WEAK matcher. Empty follows production_config() "
            "(LighterGlue). 'nn_then_lg' restores the mutual-NN fast pass for benches only."
        ),
    )
    ap.add_argument(
        "--local-topk",
        type=int,
        default=0,
        help=(
            "Override TRACK local_topk (0 = production default). "
            "For XFeat also clamps adaptive_first_topk. For EDM overrides production_edm_config."
        ),
    )
    ap.add_argument(
        "--runtime-benchmark-control",
        action="store_true",
        help="Read a per-frame SFM1 mode header for UI global/weak/track benchmarks.",
    )
    ap.add_argument(
        "--force-track-bench",
        action="store_true",
        help=(
            "Bench only: before each frame force TRACK mode with a map-ref pose prior "
            "and reset temporal cache (skips MegaLoc/BOOT_INIT). Measures a fixed-prior "
            "cold-cache TRACK microbench, not stateful flight tracking. "
            "Pose success still expected to fail off-field."
        ),
    )
    ap.add_argument(
        "--force-track-ref",
        type=int,
        default=-1,
        help="Ref index for --force-track-bench seed (-1 = middle ref).",
    )
    ap.add_argument(
        "--edm-matcher",
        choices=("torch",),
        default="torch",
        help=(
            "Verified EDM production backend: PyTorch CUDA FP16. Rejected "
            "ONNX/TensorRT experiments are intentionally unavailable here."
        ),
    )
    ap.add_argument(
        "--production-profile",
        default="",
        help="Validated edm-deployment-profile/v1 JSON applied to matcher and tracker.",
    )
    ap.add_argument(
        "--production-profile-sha256",
        default="",
        help="trusted SHA-256 for --production-profile",
    )
    args = ap.parse_args()
    if bool(args.frame_shm_name) != (args.frame_shm_slots > 0):
        ap.error("--frame-shm-name and positive --frame-shm-slots must be used together")
    if args.frame_shm_slots > 255:
        ap.error("--frame-shm-slots must fit in one byte")
    try:
        query_camera_override = resolve_query_camera_override(
            args.query_camera_model,
            args.query_camera_width,
            args.query_camera_height,
            args.query_camera_params,
            args.width,
            args.height,
        )
    except ValueError as exc:
        ap.error(str(exc))
    if args.neuflow_track and args.projection_track:
        ap.error("--neuflow-track and --projection-track are separate alternatives")
    if args.projection_track and not args.track_landmarks:
        ap.error("--projection-track requires --track-landmarks")
    try:
        backend = resolve_localizer_backend(args.localizer_backend, Path(args.bundle))
    except ValueError as exc:
        ap.error(str(exc))
    if backend == "edm":
        for enabled, flag in (
            (args.neuflow_track, "--neuflow-track"),
            (args.projection_track, "--projection-track"),
            (args.matcher_mode, "--matcher-mode"),
        ):
            if enabled:
                ap.error(f"{flag} is XFeat-only; use --localizer-backend xfeat")
        if args.production_profile_sha256 and not args.production_profile:
            ap.error("--production-profile-sha256 requires --production-profile")
    elif args.production_profile:
        ap.error("--production-profile is EDM-only")
    elif args.production_profile_sha256:
        ap.error("--production-profile-sha256 is EDM-only")

    startup_started = time.perf_counter()
    json_fd = os.dup(sys.stdout.fileno())
    json_out = os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    selected_deploy_dir = Path(args.deploy_dir).resolve()
    sys.path.insert(0, str(selected_deploy_dir))
    # A transferred EDM package contains its matcher/tracker implementation but
    # deliberately omits the UI adapter. Keep the UI adapter from the workspace
    # available after the selected deployment so its imports resolve to the
    # package's EDM modules first.
    if selected_deploy_dir != DEPLOY_DIR.resolve():
        sys.path.insert(1, str(DEPLOY_DIR))
    if backend == "xfeat" and (args.neuflow_track or args.projection_track):
        sys.path.insert(0, str(VALIDATION_DIR))
    map_frame = None
    if args.map_align:
        from real_path_follow_controller import load_map_frame
        map_frame = load_map_frame(args.map_align)

    with redirect_native_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
        import torch

        if backend == "edm":
            require_edm_cuda(torch)
            from edm_localizer_adapter import CAM_720_EDM
            from production_localizer_factory import build_production_localizer

            if args.megaloc_cache:
                print(
                    f"[live_worker] ignoring --megaloc-cache {args.megaloc_cache}: "
                    "EDM retrieval reads the bundle's own MegaLoc ref_global",
                    file=sys.stderr, flush=True,
                )
            built = build_production_localizer(
                backend="edm",
                bundle=args.bundle,
                bundle_sha256=args.bundle_sha256 or None,
                frame_source=lambda: None,
                camera_tuple=query_camera_override or CAM_720_EDM,
                production_profile=args.production_profile or None,
                production_profile_sha256=(
                    args.production_profile_sha256 or None
                ),
                local_topk=int(args.local_topk),
                map_frame=map_frame,
            )
            tracker = built.tracker
            xmap = built.reloc_map
            cam = built.camera
            cfg = built.config
            tracker_variant = built.variant
            DEVICE = built.device
            edm_matcher = tracker.trk.loc.matcher
            print(
                f"[live_worker] EDM tracker={tracker_variant} "
                f"mconf={getattr(edm_matcher, 'mconf_thr', 0.2)} "
                f"track/weak/lost={cfg.local_topk}/{cfg.weak_local_topk}/"
                f"{getattr(cfg, 'lost_local_topk', 'n/a')} "
                f"boot_topk={cfg.boot_global_topk} "
                f"staged_first={getattr(cfg, 'acquire_initial_topk', 'n/a')} "
                f"batch={getattr(cfg, 'match_batch_size', 'n/a')} "
                f"lost_grace={getattr(cfg, 'lost_local_grace_frames', 'n/a')} "
                f"recovery_bank/scan={getattr(cfg, 'recovery_bank_size', 'n/a')}/"
                f"{getattr(cfg, 'recovery_scan_topk', 'n/a')} "
                f"max_corr={cfg.max_corr_total} "
                f"reproj_gate={getattr(cfg, 'max_reproj_error_acquire', 'n/a')}/"
                f"{getattr(cfg, 'max_reproj_error_track', 'n/a')} "
                f"min_inliers={cfg.acquire_min_inliers}/{cfg.track_min_inliers}/{cfg.weak_min_inliers}",
                file=sys.stderr, flush=True,
            )
        else:
            from path_follow_flight import CAM_720, production_config
            from production_xfeat_tracker import MegaLocLayer, ProductionXFeatTracker
            from reloc_localizer_xfeat import Camera, DEVICE, XFeatRelocMap

            xmap = XFeatRelocMap.load(args.bundle)
            cache = Path(args.megaloc_cache) if args.megaloc_cache else None
            if cache is None:
                print(
                    "[live_worker] using bundle ref_global MegaLoc descriptors",
                    file=sys.stderr, flush=True,
                )
                megaloc = MegaLocLayer(xmap.ref_global, input_size=322, device=DEVICE)
            elif cache.exists():
                try:
                    megaloc = MegaLocLayer.load_cache(
                        cache, xmap.ref_names, input_size=322, device=DEVICE)
                except Exception as exc:
                    print(
                        f"[live_worker] MegaLoc cache incompatible ({exc}); "
                        "using bundle ref_global",
                        file=sys.stderr, flush=True,
                    )
                    megaloc = MegaLocLayer(xmap.ref_global, input_size=322, device=DEVICE)
            else:
                print(
                    f"[live_worker] MegaLoc cache missing: {cache}; using bundle ref_global",
                    file=sys.stderr, flush=True,
                )
                megaloc = MegaLocLayer(xmap.ref_global, input_size=322, device=DEVICE)

            cam = Camera(*(query_camera_override or CAM_720))
            cfg = production_config()
            apply_xfeat_runtime_overrides(
                cfg,
                matcher_mode=str(args.matcher_mode),
                local_topk=int(args.local_topk),
            )
            tracker_variant = (
                f"matcher_{cfg.matcher_mode}_topk{cfg.local_topk}"
                f"_adapt{cfg.adaptive_first_topk}"
            )
            print(
                f"[live_worker] XFeat TRACK matcher={cfg.matcher_mode} "
                f"local_topk={cfg.local_topk} weak_local_topk={cfg.weak_local_topk} "
                f"adaptive_first={cfg.adaptive_first_topk} "
                f"track_min_inliers={cfg.track_min_inliers} "
                f"max_reproj_track={cfg.max_reproj_error_track} "
                f"max_jump={cfg.max_jump} weak_after={cfg.weak_after} "
                f"lost_after={cfg.lost_after} xfeat_track={cfg.xfeat_topk_track}",
                file=sys.stderr, flush=True,
            )
            if args.neuflow_track:
                from neuflow_refresh_experiment import NeuFlowRefreshTracker, NeuFlowV2Backend
                neuflow_backend = NeuFlowV2Backend(
                    repo=DEFAULT_NEUFLOW_REPO,
                    weights=DEFAULT_NEUFLOW_WEIGHTS,
                    width=520, height=320,
                )
                tracker = NeuFlowRefreshTracker(
                    xmap, megaloc, frame_source=lambda: None, query_cam=cam, cfg=cfg,
                    neuflow_backend=neuflow_backend, refresh_interval=3,
                    min_seed=80, min_track=60, min_inliers=50,
                    min_inlier_ratio=0.35, max_reproj=6.0,
                )
                tracker_variant = "neuflow_v2_r3_s80_t60"
            elif args.projection_track:
                from projection_guided_tracker import (
                    ProjectionGuidedTracker, TrackLandmarkSidecar,
                )
                landmark_sidecar = TrackLandmarkSidecar.load(
                    Path(args.track_landmarks), xmap.ref_names)
                tracker = ProjectionGuidedTracker(
                    xmap, megaloc, frame_source=lambda: None, query_cam=cam, cfg=cfg,
                    landmark_sidecar=landmark_sidecar,
                    search_radii=(15.0, 25.0, 40.0),
                    min_score=0.60, ratio=0.95,
                )
                tracker_variant = "projection_guided_r15_25_40_s060_ratio095"
            else:
                tracker = ProductionXFeatTracker(
                    xmap, megaloc, frame_source=lambda: None, query_cam=cam,
                    cfg=cfg, map_frame=map_frame)

        # Experimental XFeat wrappers inherit the production pose gate but have
        # older constructor signatures. Apply the same verified frame uniformly.
        tracker.map_frame = map_frame

        if args.force_track_bench:
            # Hard lock: never MegaLoc / never leave TRACK for the whole session.
            if hasattr(tracker, "lock_track_only"):
                tracker.lock_track_only = True
            elif hasattr(tracker, "trk") and hasattr(tracker.trk, "lock_track_only"):
                tracker.trk.lock_track_only = True  # EDM adapter
            print(
                "[live_worker] lock_track_only=1: MegaLoc disabled; mode pinned to TRACK "
                "(success/fail does not escalate to WEAK/LOST)",
                file=sys.stderr, flush=True,
            )

        print(
            f"[live_worker] initializing backend={backend} device={DEVICE} "
            f"cuda={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} "
            f"bundle={args.bundle} tracker={tracker_variant} refs={len(xmap.ref_names)} "
            f"query_camera={cam.model}:{cam.width}x{cam.height}",
            file=sys.stderr,
            flush=True,
        )
        if args.force_track_bench:
            # Forced TRACK microbench never takes BOOT/LOST VPR; keep MegaLoc off GPU
            # so its cost is not attributed to the matching path being measured.
            ensure = getattr(tracker, "ensure_xfeat", None) or getattr(
                tracker, "ensure_edm", None)
            if callable(ensure):
                ensure()
        else:
            tracker.ensure_models()

        force_track_ref_resolved: int | None = None
        force_track_label = None
        seed_local_prior = None
        if args.force_track_bench or args.runtime_benchmark_control:
            from pose_types import Pose
            centers = xmap.ref_centers
            if centers is None or len(centers) == 0:
                raise RuntimeError(
                    "bundle missing ref_centers; cannot seed local benchmark prior")
            n = len(centers)
            try:
                force_track_ref_resolved = resolve_force_track_ref(args.force_track_ref, n)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            C = np.asarray(centers[force_track_ref_resolved], np.float32)
            seed_yaw = 0.0
            if xmap.ref_yaws is not None and len(xmap.ref_yaws) == n:
                seed_yaw = float(xmap.ref_yaws[force_track_ref_resolved])
            d = np.linalg.norm(np.asarray(centers, np.float32) - C[None, :], axis=1)
            seed_near = [int(j) for j in np.argsort(d)[:5]]
            if args.force_track_bench:
                force_track_label = "fixed_prior_cold_cache"

            def seed_local_prior(target_mode: str, *, reset_temporal: bool = False) -> None:
                """Seed a fixed map-reference pose for TRACK/WEAK speed tests."""
                pose0 = Pose(
                    x=float(C[0]), y=float(C[1]), z=float(C[2]),
                    yaw=seed_yaw, stamp=time.monotonic(),
                )
                st = tracker.state
                st.mode = str(target_mode)
                st.last_pose = pose0
                st.prev_pose = pose0
                st.last_center = C.copy()
                st.prev_center = C.copy()
                st.last_yaw = seed_yaw
                st.prev_yaw = seed_yaw
                st.last_refs = list(seed_near)
                st.fail_count = 0
                st.bad_count = 0
                if reset_temporal:
                    tracker.temporal_cache = type(tracker.temporal_cache)()

            if args.force_track_bench:
                seed_local_prior("TRACK", reset_temporal=True)
                print(
                    f"[live_worker] FORCE_TRACK_BENCH label={force_track_label} "
                    f"ref_requested={args.force_track_ref} "
                    f"ref_resolved={force_track_ref_resolved} "
                    f"(fixed prior, temporal cache reset, MegaLoc/BOOT_INIT skipped each frame)",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[live_worker] runtime benchmark modes ready ref={force_track_ref_resolved} "
                    "(default auto; UI controls global/weak/track at frame boundaries)",
                    file=sys.stderr,
                    flush=True,
                )

    startup_ms = (time.perf_counter() - startup_started) * 1000.0
    print(
        f"[live_worker] ready backend={backend} device={DEVICE} "
        f"tracker={tracker_variant} startup_ms={startup_ms:.1f}",
        file=sys.stderr,
        flush=True,
    )
    if args.startup_handshake:
        json_out.write(json.dumps({
            "event": "ready",
            "backend": backend,
            "device": DEVICE,
            "tracker_variant": tracker_variant,
            "startup_ms": startup_ms,
        }, ensure_ascii=False) + "\n")
        json_out.flush()

    frame_size = int(args.width) * int(args.height) * 3
    frame_buffer = None if args.frame_shm_name else bytearray(frame_size)
    frame_shm = (
        attach_frame_shm(args.frame_shm_name) if args.frame_shm_name else None
    )
    if frame_shm is not None and len(frame_shm.buf) < frame_size * args.frame_shm_slots:
        frame_shm.close()
        raise RuntimeError("shared frame memory is smaller than the declared slot count")
    # CUDA events require a synchronization to read. Keep them strictly opt-in;
    # production never inserts a device-wide/per-frame timing fence.
    gpu_timing_profiled = (
        os.environ.get("SFM_LOC_PROFILE_GPU", "0").strip().lower()
        in {"1", "true", "yes", "on"}
        and torch.cuda.is_available()
    )
    gpu_start_event = (
        torch.cuda.Event(enable_timing=True) if gpu_timing_profiled else None)
    gpu_end_event = (
        torch.cuda.Event(enable_timing=True) if gpu_timing_profiled else None)
    seq = 0
    previous_runtime_mode = "auto"
    while True:
        benchmark_mode_requested = "track" if args.force_track_bench else "auto"
        capture_stamp = None
        if args.runtime_benchmark_control:
            header = read_exact(sys.stdin.buffer, HEADER_SIZE)
            if not header:
                break
            if len(header) != HEADER_SIZE:
                print(
                    f"[live_worker] partial control header: {len(header)}/{HEADER_SIZE}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            if bytes(header[:4]) == TIMED_MAGIC:
                suffix = read_exact(
                    sys.stdin.buffer, TIMED_HEADER_SIZE - HEADER_SIZE)
                if len(suffix) != TIMED_HEADER_SIZE - HEADER_SIZE:
                    print(
                        "[live_worker] partial timed control header",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                header.extend(suffix)
            try:
                benchmark_mode_requested, capture_stamp = decode_request(header)
            except ValueError as exc:
                print(f"[live_worker] {exc}", file=sys.stderr, flush=True)
                break
        if frame_shm is None:
            assert frame_buffer is not None
            frame_bytes = read_exact_into(sys.stdin.buffer, frame_buffer)
            if not frame_bytes:
                break
            if frame_bytes != frame_size:
                print(f"[live_worker] partial frame: {frame_bytes}/{frame_size}", file=sys.stderr, flush=True)
                break
            frame = frame_view_from_rgb_bytes(frame_buffer, args.width, args.height)
        else:
            slot_byte = read_exact(sys.stdin.buffer, 1)
            if not slot_byte:
                break
            frame = frame_view_from_shared_memory(
                frame_shm.buf, slot_byte[0], args.frame_shm_slots,
                args.width, args.height,
            )
        worker_read_done_mono_ns = time.monotonic_ns()
        worker_read_done_mono = worker_read_done_mono_ns * 1e-9
        # The parent never overwrites the active slot until this synchronous request
        # returns, so both pipe and shared-memory paths need no second 720p copy.
        wall_t0 = time.perf_counter()
        core_t0: float | None = None
        worker_core_start_mono: float | None = None
        worker_core_start_mono_ns: int | None = None
        gpu_span_ms: float | None = None
        force_track_seed_ms = 0.0
        benchmark_setup_ms = 0.0
        benchmark_seeded = False
        benchmark_mode_active = "track" if args.force_track_bench else previous_runtime_mode
        try:
            if args.force_track_bench:
                seed_t0 = time.perf_counter()
                assert seed_local_prior is not None
                seed_local_prior("TRACK", reset_temporal=True)
                force_track_seed_ms = (time.perf_counter() - seed_t0) * 1000.0
                benchmark_setup_ms = force_track_seed_ms
                benchmark_seeded = True
            elif args.runtime_benchmark_control:
                setup_t0 = time.perf_counter()
                assert seed_local_prior is not None
                if benchmark_mode_requested == "relocalize":
                    # Preserve the last trustworthy spatial prior and force this
                    # frame into LOST recovery. EDM itself allows MegaLoc only on
                    # the first frame of each LOST episode; later retries stay local.
                    tracker.state.mode = "LOST"
                    tracker.state.fail_count = 0
                    tracker.state.bad_count = 0
                    benchmark_mode_active = previous_runtime_mode
                else:
                    benchmark_seeded = apply_runtime_benchmark_mode(
                        tracker,
                        benchmark_mode_requested,
                        previous_runtime_mode,
                        lambda target: seed_local_prior(target, reset_temporal=False),
                    )
                    previous_runtime_mode = benchmark_mode_requested
                    benchmark_mode_active = benchmark_mode_requested
                benchmark_setup_ms = (time.perf_counter() - setup_t0) * 1000.0
            worker_core_start_mono_ns = time.monotonic_ns()
            worker_core_start_mono = worker_core_start_mono_ns * 1e-9
            core_t0 = time.perf_counter()
            if gpu_start_event is not None:
                gpu_start_event.record()
            with redirect_native_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
                pose = tracker.localize_frame(frame, capture_stamp=capture_stamp)
            if gpu_end_event is not None:
                gpu_end_event.record()
                gpu_end_event.synchronize()
                gpu_span_ms = float(gpu_start_event.elapsed_time(gpu_end_event))
            worker_core_done_mono_ns = time.monotonic_ns()
            worker_core_done_mono = worker_core_done_mono_ns * 1e-9
            core_wall_ms = (time.perf_counter() - core_t0) * 1000.0
            # Backward-compatible wall_ms retains the old forced-bench seed cost.
            wall_ms = (time.perf_counter() - wall_t0) * 1000.0
            info = dict(tracker.last_info)
            # A non-finite pose (NaN/inf) would serialize as the bare token `NaN`
            # (invalid JSON) and could steer the UI trajectory -> report it as no fix.
            pose_ok = pose is not None and bool(
                np.isfinite([pose.x, pose.y, pose.z, pose.yaw]).all())
            raw_confidence = info.get("confidence", info.get("inlier_ratio"))
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError, OverflowError):
                confidence = 1.0 if pose_ok else 0.0
            if not math.isfinite(confidence):
                confidence = 1.0 if pose_ok else 0.0
            confidence = min(1.0, max(0.0, confidence))
            capture_mono_ns = (
                int(round(float(capture_stamp) * 1_000_000_000.0))
                if capture_stamp is not None
                else worker_read_done_mono_ns
            )
            payload = {
                "seq": seq,
                "frame_id": f"worker-{seq}",
                "capture_mono_ns": capture_mono_ns,
                "pose_mono_ns": worker_core_done_mono_ns,
                "localization_contract_version": 1,
                "validity": bool(pose_ok),
                "confidence": confidence,
                "success": pose_ok,
                "wall_ms": wall_ms,
                "core_wall_ms": core_wall_ms,
                "worker_read_done_mono": worker_read_done_mono,
                "worker_core_start_mono": worker_core_start_mono,
                "worker_core_done_mono": worker_core_done_mono,
                "worker_read_done_mono_ns": worker_read_done_mono_ns,
                "worker_core_start_mono_ns": worker_core_start_mono_ns,
                "worker_core_done_mono_ns": worker_core_done_mono_ns,
                "gpu_timing_profiled": gpu_timing_profiled,
                "gpu_span_ms": gpu_span_ms,
                "tracker_variant": tracker_variant,
                "force_track_bench": bool(args.force_track_bench),
                "force_track_label": force_track_label,
                "force_track_ref_requested": (
                    int(args.force_track_ref) if args.force_track_bench else None),
                "force_track_ref_resolved": (
                    force_track_ref_resolved if args.force_track_bench else None),
                "force_track_seed_ms": force_track_seed_ms,
                "benchmark_mode_requested": benchmark_mode_requested,
                "benchmark_mode_active": benchmark_mode_active,
                "relocalize_requested": benchmark_mode_requested == "relocalize",
                "benchmark_prior_kind": (
                    "fixed_ref" if benchmark_mode_active in {"weak", "track"} else None),
                "benchmark_prior_ref": (
                    force_track_ref_resolved
                    if benchmark_mode_active in {"weak", "track"} else None),
                "benchmark_setup_ms": benchmark_setup_ms,
                "benchmark_seeded": benchmark_seeded,
                "pose": None if not pose_ok else {
                    "x": float(pose.x),
                    "y": float(pose.y),
                    "z": float(pose.z),
                    "yaw_raw": float(pose.yaw),
                },
                "mode": info.get("mode"),
                "next_mode": info.get("next_mode"),
                "inliers": int(info.get("inliers", 0) or 0),
                "n_corr": info.get("n_corr"),
                "reference_count": info.get("reference_count"),
                "candidate_mode": info.get("candidate_mode"),
                "global_retrieval_calls": info.get("global_retrieval_calls"),
                "camera_axes_world": info.get("camera_axes_world"),
                "camera_forward_world": info.get("camera_forward_world"),
                "reproj_rms": info.get("reproj_rms"),
                "inlier_ratio": info.get("inlier_ratio"),
                "inlier_grid_cells": info.get("inlier_grid_cells"),
                "requested_reference_count": info.get("requested_reference_count"),
                "staged_early_stop": info.get("staged_early_stop"),
                "rejected": info.get("rejected"),
                "limited_jump": info.get("limited_jump"),
                "limited_jump_confirmed": info.get(
                    "limited_jump_confirmed", False
                ),
                "composite_stage": info.get("composite_stage"),
                "vpr_ms": info.get("vpr_ms"),
                "feature_ms": info.get("feature_ms"),
                "match_ms": info.get("match_ms"),
                "pnp_ms": info.get("pnp_ms"),
                "neuflow_stage": info.get("neuflow_stage"),
                "neuflow_anchor_count": info.get("neuflow_anchor_count"),
                "neuflow_flow_ms": info.get("neuflow_flow_ms"),
                "neuflow_gpu_ms": info.get("neuflow_gpu_ms"),
                "projection_fallback": info.get("projection_fallback"),
                "projection_reason": info.get("projection_reason"),
                "projection_radius_px": info.get("projection_radius_px"),
                "projection_anchor_count": info.get("projection_anchor_count"),
                "projection_visible_count": info.get("projection_visible_count"),
                "projection_match_count": info.get("projection_match_count"),
                "projection_best_inliers": info.get("projection_best_inliers"),
                "projection_project_ms": info.get("projection_project_ms"),
                "projection_feature_ms": info.get("projection_feature_ms"),
                "projection_match_ms": info.get("projection_match_ms"),
            }
        except Exception as exc:
            worker_core_done_mono_ns = time.monotonic_ns()
            worker_core_done_mono = worker_core_done_mono_ns * 1e-9
            core_wall_ms = (
                (time.perf_counter() - core_t0) * 1000.0 if core_t0 is not None else None)
            wall_ms = (time.perf_counter() - wall_t0) * 1000.0
            payload = localization_exception_payload(
                seq=seq,
                error=exc,
                frame_id=f"worker-{seq}",
                capture_mono_ns=(
                    int(round(float(capture_stamp) * 1_000_000_000.0))
                    if capture_stamp is not None
                    else worker_read_done_mono_ns
                ),
                pose_mono_ns=worker_core_done_mono_ns or worker_read_done_mono_ns,
            )
            payload.update({
                "wall_ms": wall_ms,
                "core_wall_ms": core_wall_ms,
                "worker_read_done_mono": worker_read_done_mono,
                "worker_core_start_mono": worker_core_start_mono,
                "worker_core_done_mono": worker_core_done_mono,
                "worker_read_done_mono_ns": worker_read_done_mono_ns,
                "worker_core_start_mono_ns": worker_core_start_mono_ns,
                "worker_core_done_mono_ns": worker_core_done_mono_ns,
                "gpu_timing_profiled": gpu_timing_profiled,
                "gpu_span_ms": None,
                "tracker_variant": tracker_variant,
                "force_track_bench": bool(args.force_track_bench),
                "force_track_label": force_track_label,
                "force_track_ref_requested": (
                    int(args.force_track_ref) if args.force_track_bench else None),
                "force_track_ref_resolved": (
                    force_track_ref_resolved if args.force_track_bench else None),
                "force_track_seed_ms": force_track_seed_ms,
                "benchmark_mode_requested": benchmark_mode_requested,
                "benchmark_mode_active": benchmark_mode_active,
                "benchmark_prior_kind": (
                    "fixed_ref" if benchmark_mode_active in {"weak", "track"} else None),
                "benchmark_prior_ref": (
                    force_track_ref_resolved
                    if benchmark_mode_active in {"weak", "track"} else None),
                "benchmark_setup_ms": benchmark_setup_ms,
                "benchmark_seeded": benchmark_seeded,
            })
            print(f"[live_worker] localization error: {exc!r}", file=sys.stderr, flush=True)
        json_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        json_out.flush()
        seq += 1
        if args.max_frames and seq >= args.max_frames:
            break
    if frame_shm is not None:
        if "frame" in locals():
            del frame
        frame_shm.close()


if __name__ == "__main__":
    main()

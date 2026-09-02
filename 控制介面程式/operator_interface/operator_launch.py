"""CLI and process launch for the desktop operator interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import flight_operator_app as app
from flight_operator_app import (
    ANAFI,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_DETECTOR_WORKER,
    DEFAULT_REPLAY_JSON,
    DEFAULT_WORKER,
    DroneBackend,
    DroneState,
    EXIT_SAFETY_SIGNALS,
    MissionRouteSnapshot,
    OperatorApp,
    REAL_FLIGHT_INTERFACE,
    SIMULATED_STREAM_INTERFACE,
    STREAM_HEIGHT,
    STREAM_WIDTH,
    UI_MIN_SIZE,
    UI_STANDARD_SIZE,
    default_worker_python,
    env_bool,
    optional_env_float,
)
from backend_contract import InterfaceMode, LegacyFrameSourceAdapter, SessionConfig
from ffmpeg_frame_stream import FFmpegFrameStream
from live_safety_config import (
    DEFAULT_MAX_ROTATION_SPEED_DEGS,
    DEFAULT_MAX_TILT_DEG,
    DEFAULT_MAX_VERTICAL_SPEED_MS,
    DEFAULT_RTH_MIN_ALTITUDE_M,
    LiveSafetyConfig,
)
from live_worker_clients import LiveDetectorClient, LiveLocalizerClient
from lost_hold_policy import LostHoldPolicy
from mission_manifest import ManifestError
from operator_shutdown import OperatorShutdownCoordinator
from operator_site_runtime import (
    ActiveSiteRuntime,
    PreparedSiteRuntime,
    close_active_site_runtime,
)
from runtime_safety import (
    SessionLogs,
    assess_disk_space,
    collect_runtime_identity,
    configure_offline_environment,
    enforce_retention,
    install_network_guard,
)
from site_profile import SiteProfile


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the operator CLI without starting UI, models, or hardware."""
    ap = argparse.ArgumentParser(description=app.__doc__)
    ap.add_argument(
        "--site-profile",
        default=os.environ.get("SFM_SITE_PROFILE", ""),
        help=(
            "JSON profile that atomically selects map, planned route, localization "
            "bundle and optional MegaLoc cache"
        ),
    )
    ap.add_argument("--map-ply", default=None)
    ap.add_argument("--max-points", type=int, default=250000)
    ap.add_argument("--video", default="")
    ap.add_argument(
        "--video-stride",
        type=int,
        default=1,
        help="1 means ANAFI-like 720p30 stream; >1 keeps every Nth source frame",
    )
    ap.add_argument(
        "--replay-json", default=str(DEFAULT_REPLAY_JSON) if DEFAULT_REPLAY_JSON.exists() else ""
    )
    ap.add_argument("--live-localize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--localizer-python", default=default_worker_python("SFM_LOCALIZER_PYTHON"))
    ap.add_argument("--localizer-worker", default=str(DEFAULT_WORKER))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--bundle-sha256", default=None)
    ap.add_argument(
        "--localizer-backend",
        default=None,
        choices=("auto", "edm", "xfeat"),
        help=(
            "Local matcher family (edm|xfeat). Default auto from bundle name; "
            "site profiles set this atomically with the map/bundle."
        ),
    )
    ap.add_argument(
        "--localizer-deploy-dir",
        default=None,
        help="EDM deployment directory selected directly or by --site-profile",
    )
    ap.add_argument(
        "--localizer-profile",
        default=None,
        help="EDM deployment profile selected directly or by --site-profile",
    )
    ap.add_argument("--localizer-profile-sha256", default=None)
    ap.add_argument(
        "--edm-matcher",
        default="torch",
        choices=("torch",),
        help=(
            "EDM matching engine: verified PyTorch CUDA FP16 production path. "
            "Rejected ONNX/TensorRT experiments are not selectable in flight UI."
        ),
    )
    ap.add_argument("--megaloc-cache", default=None)
    ap.add_argument(
        "--megaloc-backend",
        choices=("pytorch", "pytorch_fp16", "tensorrt"),
        default=os.environ.get("SFM_MEGALOC_BACKEND", "tensorrt"),
        help="MegaLoc query backend; defaults to TensorRT.",
    )
    ap.add_argument(
        "--megaloc-engine",
        default=os.environ.get("SFM_MEGALOC_TENSORRT_ENGINE", ""),
    )
    ap.add_argument(
        "--megaloc-engine-sha256",
        default=os.environ.get("SFM_MEGALOC_TENSORRT_ENGINE_SHA256", ""),
    )
    ap.add_argument(
        "--reference-index",
        default=None,
        help="Profile-pinned IVF index SHA256SUMS.json for large reference maps",
    )
    ap.add_argument("--reference-index-sha256", default=None)
    ap.add_argument(
        "--track-landmarks",
        default=None,
        help="XFeat TRACK landmark sidecar selected directly or by site profile",
    )
    ap.add_argument(
        "--live-detect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="OPTIONAL YOLO overlay (default OFF; not used for localization/flight)",
    )
    ap.add_argument("--detector-python", default=default_worker_python("SFM_DETECTOR_PYTHON"))
    ap.add_argument("--detector-worker", default=str(DEFAULT_DETECTOR_WORKER))
    ap.add_argument("--detector-model", default=str(DEFAULT_DETECTOR_MODEL))
    ap.add_argument("--detect-every-n-frames", type=int, default=3)
    ap.add_argument("--detector-conf", type=float, default=0.25)
    ap.add_argument("--detector-iou", type=float, default=0.7)
    ap.add_argument("--detector-max-det", type=int, default=300)
    ap.add_argument(
        "--loc-every-n-frames",
        type=int,
        default=1,
        help="Submit every Nth stream frame to the localizer (1=every new frame; 2≈half rate)",
    )
    ap.add_argument(
        "--adaptive-loc-submit",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SFM_ADAPTIVE_LOC_SUBMIT", "1").strip()
        in {"1", "true", "TRUE", "yes"},
        help=(
            "Adapt busy-worker coalesce cadence to half the latest localization "
            "latency while preserving latest-frame delivery"
        ),
    )
    ap.add_argument("--tick-ms", type=int, default=8)
    ap.add_argument(
        "--stream-fps",
        type=float,
        default=ANAFI.stream_fps,
        help="720p frame-source rate fed to the localizer (real ANAFI live stream is 30)",
    )
    ap.add_argument(
        "--boot-lock-ms",
        type=int,
        default=2500,
        help="hold the first 720p frame to simulate takeoff hover + MegaLoc BOOT_INIT",
    )
    ap.add_argument(
        "--lost-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable bounded localization recovery. File/video pauses its frame; "
            "real flight sends zero PCMD, hands control to the pilot, and keeps "
            "consuming live frames."
        ),
    )
    ap.add_argument(
        "--lost-hold-max-attempts",
        type=int,
        default=5,
        help="total LOST recovery attempts before the held stream is released",
    )
    ap.add_argument(
        "--hold-on-low-confidence",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SFM_HOLD_ON_LOW_CONF", "1").strip() in {"1", "true", "TRUE", "yes"},
        help=(
            "after consecutive low-confidence EDM results, hover/hold and run "
            "staged MegaLoc/EDM recovery (enabled by default)"
        ),
    )
    ap.add_argument(
        "--low-confidence-hold-results",
        type=int,
        default=int(os.environ.get("SFM_LOW_CONF_HOLD_RESULTS", "2")),
        help="consecutive low-confidence EDM results before staged MegaLoc recovery",
    )
    ap.add_argument(
        "--lost-hold-timeout-ms",
        type=int,
        default=10000,
        help="release the held frame if the retries stall (0 = no timeout)",
    )
    ap.add_argument(
        "--auto-inspect",
        action="store_true",
        help="after UI start, auto 開始定位 (does NOT send flight commands)",
    )
    ap.add_argument(
        "--loc-force-track-bench",
        action="store_true",
        help=(
            "Pass --force-track-bench to localizer worker: skip MegaLoc/BOOT_INIT, "
            "seed a fixed TRACK prior and cold cache each frame "
            "(TRACK microbench only; no takeoff)."
        ),
    )
    ap.add_argument(
        "--loc-force-track-ref",
        type=int,
        default=-1,
        help="Map ref index for track bench seed (-1=middle)",
    )
    ap.add_argument(
        "--neuflow-track",
        action="store_true",
        help="Use NeuFlow-v2 refresh=3 hybrid in TRACK; deep matcher remains fallback.",
    )
    ap.add_argument(
        "--pose-stabilize",
        action="store_true",
        help=(
            "Publish a causal median+low-pass pose while retaining raw PnP in telemetry. "
            "Intended for verified replay/site profiles with visibly noisy framewise PnP."
        ),
    )
    ap.add_argument(
        "--projection-track",
        action="store_true",
        help="Use projection-guided TRACK fast path; unchanged deep matcher remains fallback.",
    )
    ap.add_argument(
        "--nn-fast-path",
        action="store_true",
        help=(
            "BENCHMARK ONLY: restore the mutual-NN fast pass that was removed from "
            "production on 2026-07-14 (accuracy). Production runs LighterGlue every frame."
        ),
    )
    ap.add_argument(
        "--local-topk",
        type=int,
        default=0,
        help=(
            "Override TRACK local_topk for the live worker (0 = production default). "
            "XFeat: also clamps adaptive_first_topk. EDM: overrides production_edm_config."
        ),
    )
    ap.add_argument(
        "--route-json",
        default=None,
        help="drawn flight path (aligned frame) to overlay on the map in a distinct colour",
    )
    ap.add_argument("--layout-selftest", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--interface",
        dest="interface_mode",
        choices=(SIMULATED_STREAM_INTERFACE, REAL_FLIGHT_INTERFACE),
        default="",
        help="explicit input/control boundary: local video or real ANAFI/Olympe",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="legacy alias for --interface real-flight",
    )
    ap.add_argument(
        "--ip",
        default="192.168.53.1",
        help="192.168.53.1 SkyController / 192.168.42.1 direct drone",
    )
    ap.add_argument("--controller", default="skycontroller3", help="skycontroller3 / drone / auto")
    ap.add_argument(
        "--nudge-pct",
        type=int,
        default=8,
        help="PCMD percent for micro-moves (keep small; default 8)",
    )
    ap.add_argument(
        "--nudge-pulse-s",
        type=float,
        default=0.20,
        help="nudge heartbeat TTL/deadman seconds (default 0.20)",
    )
    ap.add_argument(
        "--max-altitude-m",
        type=float,
        default=optional_env_float("SFM_MAX_ALTITUDE_M"),
        help="advisory desired firmware MaxAltitude; failure does not block takeoff",
    )
    ap.add_argument(
        "--max-distance-m",
        type=float,
        default=optional_env_float("SFM_MAX_DISTANCE_M"),
        help="advisory desired firmware MaxDistance; failure does not block takeoff",
    )
    ap.add_argument(
        "--distance-geofence",
        action=argparse.BooleanOptionalAction,
        # OFF by default. GPS and firmware readback are advisory takeoff inputs;
        # containment for
        # manual flight comes from MaxAltitude plus the operator. Link loss is
        # handled by the onboard policy (RTH when Home is usable, else land).
        default=env_bool("SFM_DISTANCE_GEOFENCE", False),
        help=(
            "enable NoFlyOverMaxDistance; host monitor also triggers RTH/Landing "
            "at 95%% of the confirmed limit"
        ),
    )
    ap.add_argument(
        "--max-tilt-deg",
        type=float,
        default=float(os.environ.get("SFM_MAX_TILT_DEG", DEFAULT_MAX_TILT_DEG)),
        help="firmware MaxTilt pinned at connect; scales every stick command",
    )
    ap.add_argument(
        "--max-vertical-speed-ms",
        type=float,
        default=float(os.environ.get("SFM_MAX_VERTICAL_SPEED_MS", DEFAULT_MAX_VERTICAL_SPEED_MS)),
        help="firmware MaxVerticalSpeed pinned at connect",
    )
    ap.add_argument(
        "--max-rotation-speed-degs",
        type=float,
        default=float(
            os.environ.get("SFM_MAX_ROTATION_SPEED_DEGS", DEFAULT_MAX_ROTATION_SPEED_DEGS)
        ),
        help="firmware MaxRotationSpeed pinned at connect",
    )
    ap.add_argument(
        "--stream-loss-grace-s",
        type=float,
        default=float(os.environ.get("SFM_STREAM_LOSS_GRACE_S", 10.0)),
        help=(
            "how long the video stream must be CONTINUOUSLY stale before control "
            "is handed back to the sticks; brief latency spikes recover inside it"
        ),
    )
    ap.add_argument(
        "--auto-pc-control",
        action=argparse.BooleanOptionalAction,
        default=env_bool("SFM_AUTO_PC_CONTROL", True),
        help=(
            "take PC control automatically on connect (landed + sticks idle "
            "only); stick movement always hands control back"
        ),
    )
    ap.add_argument(
        "--rth-min-altitude-m",
        type=float,
        default=float(os.environ.get("SFM_RTH_MIN_ALTITUDE_M", DEFAULT_RTH_MIN_ALTITUDE_M)),
        help=(
            "RTH climb altitude pinned at connect; keep at or below "
            "--max-altitude-m or the recovery breaks your own ceiling"
        ),
    )
    ap.add_argument(
        "--min-takeoff-battery-pct",
        type=float,
        default=float(os.environ.get("SFM_MIN_TAKEOFF_BATTERY_PCT", "30")),
        help="advisory takeoff battery threshold; never blocks takeoff (default 30%%)",
    )
    ap.add_argument(
        "--require-gps-for-geofence",
        action=argparse.BooleanOptionalAction,
        default=env_bool("SFM_REQUIRE_GPS_FOR_GEOFENCE", True),
        help="legacy compatibility flag; GPS status is advisory and never gates takeoff",
    )
    ap.add_argument(
        "--no-live-video", action="store_true", help="skip PDRAW video (control+telemetry only)"
    )
    ap.add_argument("--cmd-log", default="", help="JSONL path for live command log")
    return ap


def _autonomy_profile_readiness_errors(
    site_profile: SiteProfile | None,
) -> list[str]:
    if site_profile is None:
        return ["missing site profile"]
    return app.flight_readiness_errors(site_profile)


def _load_startup_route(
    route_path: Path | None,
    site_profile: SiteProfile | None,
) -> tuple[str | None, list, MissionRouteSnapshot | None]:
    if route_path is None:
        return None, [], None
    if not route_path.is_file():
        raise SystemExit(f"route JSON not found: {route_path}")

    route_hash = app.file_sha256(route_path)
    mission_snapshot = None
    try:
        route_map_frame = app.resolve_site_map_frame(site_profile)
        flight = site_profile.flight if site_profile is not None else None
        coordinate_frame_id = flight.coordinate_frame_id if flight is not None else None
        if site_profile is not None and coordinate_frame_id:
            candidate = app.capture_mission_route_snapshot(
                route_path,
                expected_sha256=route_hash,
                expected_site_id=site_profile.site_id,
                expected_coordinate_frame_id=coordinate_frame_id,
                map_frame=route_map_frame or app.LEGACY_MAP_FRAME,
            )
            route_points = candidate.controller_waypoints()
            readiness_errors = app.flight_readiness_errors(site_profile)
            if readiness_errors:
                print(
                    "[operator] route loaded for display only; AUTO binding "
                    "rejected: " + "; ".join(readiness_errors),
                    flush=True,
                )
            else:
                mission_snapshot = candidate
        else:
            route_points = app.load_route_glomap(str(route_path), map_frame=route_map_frame)
    except Exception as exc:
        # This also covers malformed gravity alignment. It must be an
        # operator-readable startup refusal, not a traceback.
        raise SystemExit(f"cannot load route {route_path} for this site: {exc}") from exc
    return route_hash, route_points, mission_snapshot


def _parse_operator_arguments(ap: argparse.ArgumentParser) -> argparse.Namespace:
    args = ap.parse_args()
    try:
        args.interface_mode, args.live = app.resolve_operator_interface(
            args.interface_mode, bool(args.live), str(args.video or "")
        )
    except ValueError as exc:
        ap.error(str(exc))
    if (
        args.interface_mode == SIMULATED_STREAM_INTERFACE
        and not str(args.video or "").strip()
        and not args.selftest
        and not args.layout_selftest
    ):
        ap.error(
            "simulated-stream requires a video file; use the dedicated launcher "
            "for the approved P119 default"
        )
    if args.live and args.replay_json and not args.live_localize:
        # Real-flight interface. state_from_replay() feeds the operator's pose, map
        # track, heading and localization-quality HUD; sourcing that from a recording
        # while a live, commandable ANAFI is connected would show a plausible
        # trajectory that has nothing to do with where the aircraft actually is.
        # Refuse before anything connects.
        raise SystemExit(
            "--replay-json cannot be combined with the real-flight interface: the "
            "operator HUD must never be driven from a recording while a live aircraft "
            "is connected. Use --live-localize, or drop --live."
        )
    if args.neuflow_track and args.projection_track:
        ap.error("--neuflow-track and --projection-track are separate alternatives")
    if args.projection_track and not args.track_landmarks:
        ap.error("--projection-track requires --track-landmarks or a profile sidecar")
    return args


def _resolve_startup_site(
    args: argparse.Namespace,
    ap: argparse.ArgumentParser,
) -> tuple[SiteProfile | None, object | None]:
    site_profile = app.resolve_operator_site_assets(args, ap)
    if args.live and site_profile is None:
        raise SystemExit(
            "--live requires an explicit --site-profile; field assets must never "
            "fall back to a different site's map/route/bundle"
        )
    if args.live:
        selection_arg = os.environ.get("SFM_MISSION_SELECTION", "").strip()
        if not selection_arg:
            raise SystemExit(
                "real-flight requires SFM_MISSION_SELECTION; a direct site profile "
                "is not an independent flight authorization source"
            )
        try:
            mission = app.resolve_mission(selection_arg, workspace_root=app._WS.root)
            if not mission.readiness.localization_ready:
                raise ManifestError(
                    "mission is not localization-ready: "
                    + "; ".join(mission.readiness.localization_errors)
                )
            expected_profile = mission.materialize_legacy_site_profile(
                app._WS.runtime / "mission_snapshots"
            ).resolve()
        except (ManifestError, OSError, ValueError) as exc:
            raise SystemExit(f"mission selection rejected: {exc}") from exc
        assert site_profile is not None
        if site_profile.source.resolve() != expected_profile:
            raise SystemExit(
                "real-flight site profile does not match mission selection snapshot: "
                f"expected {expected_profile}, got {site_profile.source.resolve()}"
            )
        args.mission_selection = str(mission.selection.source)
        args.mission_selection_sha256 = mission.selection_sha256
        args.mission_snapshot_id = mission.identity
        args.mission_flight_ready = bool(mission.readiness.flight_ready)
        args.mission_flight_errors = tuple(mission.readiness.flight_errors)
    hardware_approval = None
    if site_profile is not None and site_profile.hardware_approval is not None:
        try:
            hardware_approval = app.load_hardware_approval_receipt(site_profile.hardware_approval)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return site_profile, hardware_approval


def _resolve_live_safety(
    args: argparse.Namespace,
    ap: argparse.ArgumentParser,
) -> LiveSafetyConfig:
    try:
        live_safety_config = LiveSafetyConfig.resolve(
            nudge_pct=args.nudge_pct,
            nudge_pulse_s=args.nudge_pulse_s,
            max_altitude_m=args.max_altitude_m,
            max_distance_m=args.max_distance_m,
            max_tilt_deg=args.max_tilt_deg,
            max_vertical_speed_ms=args.max_vertical_speed_ms,
            max_rotation_speed_degs=args.max_rotation_speed_degs,
            rth_min_altitude_m=args.rth_min_altitude_m,
            stream_loss_grace_s=args.stream_loss_grace_s,
            min_takeoff_battery_pct=args.min_takeoff_battery_pct,
            distance_geofence=args.distance_geofence,
            require_gps_for_geofence=args.require_gps_for_geofence,
        )
    except ValueError as exc:
        ap.error(str(exc))
    args.nudge_pulse_s = live_safety_config.nudge_pulse_s
    print(
        "[operator] effective_live_safety="
        f"{json.dumps(live_safety_config.as_log_fields(), sort_keys=True)} "
        f"sha256={live_safety_config.checksum()}",
        flush=True,
    )
    return live_safety_config


def _announce_interface(args: argparse.Namespace) -> None:
    if args.live:
        try:
            from olympe_live_backend import quiet_olympe_logs

            quiet_olympe_logs()
        except Exception:
            pass
        print(
            "[mode] REAL-FLIGHT interface: LIVE Olympe TakeOff/PCMD/Landing via "
            f"ip={args.ip} controller={args.controller}. "
            "Esc/手動=交回搖桿; 關窗=Landing+還搖桿. "
            "Safety pilot must hold the sticks.",
            flush=True,
        )
        return
    print(
        "[mode] SIMULATED-STREAM interface: no drone commands; this app uses a "
        "sim backend and never sends TakeOff/PCMD/Landing/Emergency to a real drone",
        flush=True,
    )


def _announce_site_assets(
    args: argparse.Namespace,
    site_profile: SiteProfile | None,
    route_path: Path | None,
    route_hash: str | None,
) -> None:
    if site_profile is not None:
        print(
            f"[operator] site={site_profile.site_id!r} "
            f"name={site_profile.display_name!r} profile={site_profile.source}",
            flush=True,
        )
        if site_profile.query_camera is not None:
            query_camera = site_profile.query_camera
            print(
                f"[operator] query_camera={query_camera.model}:"
                f"{query_camera.width}x{query_camera.height} "
                f"params={list(query_camera.params)}",
                flush=True,
            )
        if site_profile.localizer_profile is not None:
            print(
                f"[operator] localizer_profile={site_profile.localizer_profile}",
                flush=True,
            )
    print(
        f"[operator] map={Path(args.map_ply).resolve()} "
        f"bundle={Path(args.bundle).resolve()} "
        f"localizer={args.localizer_backend} "
        f"megaloc={Path(args.megaloc_cache).resolve() if args.megaloc_cache else 'bundle:ref_global'}",
        flush=True,
    )
    if route_path is None:
        print("[operator] route=none (replay-only profile)", flush=True)
    else:
        print(f"[operator] route={route_path.resolve()} sha256={route_hash}", flush=True)


def _resolve_startup_localizer(args: argparse.Namespace) -> None:
    try:
        args.localizer_backend = app.resolve_localizer_backend(
            str(getattr(args, "localizer_backend", None) or "auto"),
            Path(args.bundle),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _load_runtime_map(
    args: argparse.Namespace,
    site_profile: SiteProfile | None,
) -> np.ndarray:
    points = app.read_map_points(Path(args.map_ply), args.max_points)
    if site_profile is None or site_profile.map_reference_poses is None:
        return points
    reference_centers = app.read_reference_pose_points(
        site_profile.map_reference_poses, args.max_points
    )[:, :3]
    lower = reference_centers.min(axis=0) - 5.0
    upper = reference_centers.max(axis=0) + 5.0
    keep = ((points[:, :3] >= lower) & (points[:, :3] <= upper)).all(axis=1)
    before = len(points)
    points = points[keep]
    if not len(points):
        raise SystemExit("map RGB points do not overlap the reference-pose coordinate frame")
    print(
        f"[operator] RGB map reference-bound filter kept {len(points)}/{before} points",
        flush=True,
    )
    return points


def _run_operator_selftest(args: argparse.Namespace, points: np.ndarray) -> bool:
    if not args.selftest:
        return False
    print(f"loaded {len(points)} sampled map points from {args.map_ply}")
    if len(points):
        xyz = points[:, :3]
        print(
            f"bbox min={xyz.min(axis=0).round(3).tolist()} max={xyz.max(axis=0).round(3).tolist()}"
        )
    if args.video:
        print(
            f"video stream fixed at {STREAM_WIDTH}x{STREAM_HEIGHT}"
            f"@{ANAFI.stream_fps:g}: {args.video} stride={args.video_stride}"
        )
    if args.replay_json:
        replay = json.loads(Path(args.replay_json).read_text(encoding="utf-8"))
        print(f"loaded replay rows={len(replay.get('rows', []))} from {args.replay_json}")
    print(
        f"live localize={args.live_localize} worker={args.localizer_worker} "
        f"neuflow_track={args.neuflow_track} projection_track={args.projection_track} "
        f"track_landmarks={args.track_landmarks or None} "
        f"pose_stabilize={args.pose_stabilize}"
    )
    print(
        f"live detect={args.live_detect} worker={args.detector_worker} "
        f"model={args.detector_model} every={args.detect_every_n_frames}f"
    )
    print(f"boot lock hold={args.boot_lock_ms} ms")
    return True


def _run_layout_selftest(
    args: argparse.Namespace,
    points: np.ndarray,
    interface_mode: InterfaceMode,
    standard_size: tuple[int, int],
    minimum_viewport: tuple[int, int],
) -> bool:
    if not args.layout_selftest:
        return False
    layout_backend = DroneBackend()
    if interface_mode is InterfaceMode.REAL_FLIGHT:
        layout_backend.is_live = True
        layout_backend.pilot_sticks = True
        layout_backend.desired_max_altitude_m = args.max_altitude_m
        layout_backend.desired_max_distance_m = args.max_distance_m
        layout_backend.desired_distance_geofence = bool(args.distance_geofence)
    app = OperatorApp(layout_backend, points, tick_ms=args.tick_ms, boot_lock_ms=0)
    app.update_idletasks()
    sizes = []
    for _ in range(5):
        app.tick()
        app.update_idletasks()
        sizes.append((app.winfo_reqwidth(), app.winfo_reqheight()))
    print(f"layout requested sizes: {sizes}")
    if len(set(sizes[-3:])) != 1:
        raise SystemExit("layout requested size is not stable")
    requested_width, requested_height = sizes[-1]
    if requested_width > standard_size[0] or requested_height > standard_size[1]:
        raise SystemExit(
            f"layout exceeds the standard viewport: {requested_width}x{requested_height}"
        )
    app.geometry(f"{minimum_viewport[0]}x{minimum_viewport[1]}")
    app.update_idletasks()
    actual_minimum = (app.winfo_width(), app.winfo_height())
    print(f"layout minimum viewport: {actual_minimum}")
    if actual_minimum != minimum_viewport:
        raise SystemExit(f"layout cannot hold the minimum viewport: {actual_minimum}")
    app.destroy()
    return True


@dataclass(frozen=True)
class OperatorSessionIdentity:
    asset_hashes: dict[str, str]
    site_profile_sha256: str
    runtime_profile_sha256: str
    site_profile_schema_version: int
    autonomous_speed_limit_mps: float
    autonomy_profile_errors: tuple[str, ...]
    source_identity: str
    source_sha256: str

    @property
    def autonomous_locked(self) -> bool:
        return bool(self.autonomy_profile_errors)


def _operator_session_identity(
    args: argparse.Namespace,
    site_profile: SiteProfile | None,
) -> OperatorSessionIdentity:
    asset_hashes = {}
    if site_profile is not None:
        asset_hashes = {
            key: value for key, value in asdict(site_profile.asset_sha256).items() if value
        }
    runtime_profile_sha256 = ""
    if site_profile is not None and site_profile.localizer_profile is not None:
        runtime_profile_sha256 = app.file_sha256(site_profile.localizer_profile)
    autonomous_speed_limit_mps = 0.30
    if (
        site_profile is not None
        and site_profile.flight is not None
        and site_profile.flight.controller is not None
    ):
        autonomous_speed_limit_mps = float(site_profile.flight.controller.speed_limit_mps)
    return OperatorSessionIdentity(
        asset_hashes=asset_hashes,
        site_profile_sha256=(
            app.file_sha256(site_profile.source) if site_profile is not None else ""
        ),
        runtime_profile_sha256=runtime_profile_sha256,
        site_profile_schema_version=(
            int(site_profile.schema_version) if site_profile is not None else 0
        ),
        autonomous_speed_limit_mps=autonomous_speed_limit_mps,
        autonomy_profile_errors=tuple(_autonomy_profile_readiness_errors(site_profile)),
        source_identity=(str(Path(args.video).resolve()) if args.video else args.ip),
        source_sha256=(app.file_sha256(Path(args.video)) if args.video else ""),
    )


def _hardware_approval_manifest(hardware_approval: object | None) -> dict | None:
    if hardware_approval is None:
        return None
    return {
        "path": str(hardware_approval.source),
        "sha256": hardware_approval.sha256,
        "approved": hardware_approval.approved,
        "aircraft_product": hardware_approval.aircraft_product,
        "controller_product": hardware_approval.controller_product,
        "aircraft_firmware_versions": list(hardware_approval.aircraft_firmware_versions),
        "controller_firmware_versions": list(hardware_approval.controller_firmware_versions),
        "olympe_versions": list(hardware_approval.olympe_versions),
    }


def _operator_session_manifest(
    args: argparse.Namespace,
    interface_mode: InterfaceMode,
    site_profile: SiteProfile | None,
    hardware_approval: object | None,
    identity: OperatorSessionIdentity,
) -> dict:
    return {
        "site_id": site_profile.site_id if site_profile is not None else "",
        "site_profile": (str(site_profile.source) if site_profile is not None else ""),
        "site_profile_sha256": identity.site_profile_sha256,
        "site_profile_schema_version": identity.site_profile_schema_version,
        "mission_selection": str(getattr(args, "mission_selection", "") or ""),
        "mission_selection_sha256": str(getattr(args, "mission_selection_sha256", "") or ""),
        "mission_snapshot_id": str(getattr(args, "mission_snapshot_id", "") or ""),
        "mission_flight_ready": bool(getattr(args, "mission_flight_ready", False)),
        "mission_flight_errors": list(getattr(args, "mission_flight_errors", ()) or ()),
        "asset_sha256": identity.asset_hashes,
        "runtime_profile_sha256": identity.runtime_profile_sha256,
        "source": identity.source_identity,
        "source_sha256": identity.source_sha256,
        "source_integrity": os.environ.get("SFM_SOURCE_INTEGRITY", "UNVERIFIED"),
        "source_declared_frames": os.environ.get("SFM_SOURCE_DECLARED_FRAMES", ""),
        "source_decoded_frames": os.environ.get("SFM_SOURCE_DECODED_FRAMES", ""),
        "python": sys.version,
        "runtime": collect_runtime_identity(),
        "argv": list(sys.argv),
        "offline": True,
        "autonomous_speed_limit_mps": identity.autonomous_speed_limit_mps,
        "autonomous_locked": identity.autonomous_locked,
        "autonomous_approval_valid": not identity.autonomous_locked,
        "autonomy_profile_errors": list(identity.autonomy_profile_errors),
        "runtime_inventory_receipts": (
            {"hardware": "hardware_inventory.json", "video": "video_inventory.json"}
            if interface_mode is InterfaceMode.REAL_FLIGHT
            else {}
        ),
        "firmware_limits_requested": {
            "max_altitude_m": args.max_altitude_m,
            "max_distance_m": args.max_distance_m,
            "distance_geofence": bool(args.distance_geofence),
        },
        "hardware_approval": _hardware_approval_manifest(hardware_approval),
    }


def _create_operator_session(
    args: argparse.Namespace,
    interface_mode: InterfaceMode,
    site_profile: SiteProfile | None,
    hardware_approval: object | None,
) -> tuple[SessionLogs, SessionConfig]:
    identity = _operator_session_identity(args, site_profile)
    session_logs = SessionLogs.create(
        app._WS.flight_logs,
        mode=interface_mode,
        manifest=_operator_session_manifest(
            args,
            interface_mode,
            site_profile,
            hardware_approval,
            identity,
        ),
    )
    session_config = SessionConfig(
        session_id=session_logs.directory.name,
        interface_mode=interface_mode,
        site_profile=(str(site_profile.source) if site_profile is not None else ""),
        site_profile_sha256=identity.site_profile_sha256,
        asset_sha256=identity.asset_hashes,
        runtime_profile_sha256=identity.runtime_profile_sha256,
        source=identity.source_identity,
        offline=True,
        site_profile_schema_version=identity.site_profile_schema_version,
        autonomous_speed_limit_mps=identity.autonomous_speed_limit_mps,
        autonomous_locked=identity.autonomous_locked,
        firmware_limits={
            "max_altitude_m": args.max_altitude_m,
            "max_distance_m": args.max_distance_m,
            "distance_geofence": bool(args.distance_geofence),
        },
    )
    retention = enforce_retention(
        app._WS.flight_logs,
        current_session=session_logs.directory,
    )
    disk_status = assess_disk_space(session_logs.directory)
    print(
        f"[session] {session_logs.directory} | disk={disk_status.reason} | "
        f"retention_removed={len(retention.removed)}",
        flush=True,
    )
    if disk_status.warning:
        session_logs.incident(
            "disk_warning",
            free_bytes=disk_status.free_bytes,
            free_percent=disk_status.free_percent,
            takeoff_blocked=disk_status.takeoff_blocked,
            reason=disk_status.reason,
        )
    app.atexit.register(lambda: session_logs.close(reason="process_atexit"))
    return session_logs, session_config


def _approved_versions(
    hardware_approval: object | None,
    attribute: str,
) -> tuple[str, ...]:
    if hardware_approval is None or not hardware_approval.approved:
        return ()
    return tuple(getattr(hardware_approval, attribute))


def _build_operator_backend(
    args: argparse.Namespace,
    session_logs: SessionLogs,
    hardware_approval: object | None,
) -> tuple[object, object | None, object | None]:
    if not args.live:
        backend = DroneBackend(session_logs=session_logs)
        video_stream = None
        if args.video:
            video_stream = FFmpegFrameStream(
                Path(args.video),
                STREAM_WIDTH,
                STREAM_HEIGHT,
                stride=args.video_stride,
                fps=ANAFI.stream_fps,
                link_sim=app.resolve_anafi_link_sim(),
            )
            backend.video = LegacyFrameSourceAdapter(video_stream, str(Path(args.video).resolve()))
        return backend, video_stream, None

    from olympe_live_backend import OlympeLiveBackend

    cmd_log = session_logs.directory / "commands.jsonl"
    live_backend = OlympeLiveBackend(
        DroneState,
        ANAFI,
        ip=args.ip,
        controller=args.controller,
        nudge_pct=args.nudge_pct,
        nudge_pulse_s=args.nudge_pulse_s,
        max_altitude_m=args.max_altitude_m,
        max_distance_m=args.max_distance_m,
        distance_geofence=bool(args.distance_geofence),
        min_takeoff_battery_pct=float(args.min_takeoff_battery_pct),
        max_tilt_deg=float(args.max_tilt_deg),
        max_vertical_speed_ms=float(args.max_vertical_speed_ms),
        max_rotation_speed_degs=float(args.max_rotation_speed_degs),
        rth_min_altitude_m=float(args.rth_min_altitude_m),
        auto_pc_control=bool(args.auto_pc_control),
        stream_loss_grace_s=float(args.stream_loss_grace_s),
        require_gps_for_geofence=bool(args.require_gps_for_geofence),
        approved_aircraft_firmware=_approved_versions(
            hardware_approval, "aircraft_firmware_versions"
        ),
        approved_controller_firmware=_approved_versions(
            hardware_approval, "controller_firmware_versions"
        ),
        approved_olympe_versions=_approved_versions(hardware_approval, "olympe_versions"),
        cmd_log=cmd_log,
        event_log=session_logs.command_log,
        session_logs=session_logs,
        with_video=not args.no_live_video,
    )
    video_stream = live_backend.video_stream
    if video_stream is None:
        print(
            "[live] PDRAW unavailable; file fallback is prohibited in real-flight mode",
            flush=True,
        )
    print(f"[live] command log -> {cmd_log}", flush=True)
    return live_backend, video_stream, live_backend


def _start_operator_backend(
    backend: object,
    session_config: SessionConfig,
    session_logs: SessionLogs,
) -> None:
    started = backend.start(session_config)
    if started.started:
        return
    session_logs.incident("session_start_rejected", reason=started.reason_code)
    session_logs.close(reason="session_start_rejected")
    raise SystemExit(f"backend start rejected: {started.reason_code}")


def _announce_localizer_mode(args: argparse.Namespace) -> None:
    if args.localizer_backend == "edm":
        edm_m = str(getattr(args, "edm_matcher", "torch") or "torch")
        topk = int(getattr(args, "local_topk", 0) or 0) or 1
        profile_txt = str(getattr(args, "localizer_profile", "") or "defaults")
        print(
            "[operator] TRACK/WEAK matcher = EDM (detector-free), "
            f"{topk} ref/frame (local_topk={topk}), engine={edm_m}; "
            "BOOT MegaLoc staged 2 then 10; LOST first nearby EDM, then profile-scheduled MegaLoc; "
            f"profile={profile_txt}",
            flush=True,
        )
    else:
        topk = int(getattr(args, "local_topk", 0) or 0)
        topk_txt = str(topk) if topk > 0 else "production"
        override = "; nn_then_lg OVERRIDE" if args.nn_fast_path else ""
        print(
            "[operator] TRACK/WEAK matcher = XFeat + LighterGlue ONLY "
            f"(no MNN; local_topk={topk_txt}{override}); "
            "BOOT/LOST acquisition = MegaLoc top-30 -> LighterGlue",
            flush=True,
        )
    if args.neuflow_track:
        print(
            "[operator] TRACK=NeuFlow-v2 refresh3; deep XFeat/NN/LighterGlue "
            "retained for refresh/fallback and BOOT/LOST",
            flush=True,
        )
    if args.projection_track:
        print(
            "[operator] TRACK=projection-guided 15/25/40px; original "
            "XFeat/NN/LighterGlue retained as same-frame fallback",
            flush=True,
        )
    if args.loc_force_track_bench:
        print(
            "[operator] loc FORCE_TRACK_BENCH: fixed-prior cold-cache TRACK "
            "microbench; MegaLoc/BOOT_INIT skipped (success may still be false "
            "off-field)",
            flush=True,
        )


def _build_operator_localizer(
    args: argparse.Namespace,
    site_profile: SiteProfile | None,
    video_stream: object | None,
) -> object | None:
    want_loc = args.live_localize and (video_stream is not None or not args.live)
    if not want_loc:
        return None
    if args.localizer_backend == "edm" and (
        args.neuflow_track or args.projection_track or args.nn_fast_path
    ):
        raise SystemExit(
            "NeuFlow / projection-track / --nn-fast-path are XFeat-only; "
            "use --localizer-backend xfeat or an XFeat site profile"
        )
    localizer = LiveLocalizerClient(
        Path(args.localizer_worker),
        args.localizer_python,
        STREAM_WIDTH,
        STREAM_HEIGHT,
        Path(args.bundle),
        args.megaloc_cache,
        megaloc_backend=str(args.megaloc_backend),
        megaloc_engine=str(args.megaloc_engine or ""),
        megaloc_engine_sha256=str(args.megaloc_engine_sha256 or ""),
        reference_index=str(getattr(args, "reference_index", "") or ""),
        reference_index_sha256=str(getattr(args, "reference_index_sha256", "") or ""),
        force_track_bench=bool(args.loc_force_track_bench),
        force_track_ref=int(args.loc_force_track_ref),
        neuflow_track=bool(args.neuflow_track),
        projection_track=bool(args.projection_track),
        track_landmarks=args.track_landmarks,
        matcher_mode=(
            "nn_then_lg"
            if args.nn_fast_path
            else ("lighterglue" if str(args.localizer_backend) == "xfeat" else "")
        ),
        localizer_backend=str(args.localizer_backend),
        localizer_deploy_dir=str(getattr(args, "localizer_deploy_dir", "") or ""),
        localizer_profile=str(getattr(args, "localizer_profile", "") or ""),
        bundle_sha256=str(getattr(args, "bundle_sha256", "") or ""),
        localizer_profile_sha256=str(getattr(args, "localizer_profile_sha256", "") or ""),
        local_topk=int(getattr(args, "local_topk", 0) or 0),
        query_camera=(site_profile.query_camera if site_profile is not None else None),
        map_align=(site_profile.map_align if site_profile is not None else ""),
    )
    _announce_localizer_mode(args)
    return localizer


def _build_operator_detector(
    args: argparse.Namespace,
    video_stream: object | None,
) -> object | None:
    if not args.live_detect or video_stream is None:
        return None
    return LiveDetectorClient(
        Path(args.detector_worker),
        args.detector_python,
        STREAM_WIDTH,
        STREAM_HEIGHT,
        Path(args.detector_model),
        imgsz=640,
        conf=args.detector_conf,
        iou=args.detector_iou,
        max_det=args.detector_max_det,
    )


def _load_operator_replay(args: argparse.Namespace) -> list:
    if not args.replay_json or args.live_localize:
        return []
    replay = json.loads(Path(args.replay_json).read_text(encoding="utf-8"))
    return replay.get("rows", [])


def _build_lost_hold_policy(
    args: argparse.Namespace,
    localizer: object | None,
    video_stream: object | None,
) -> LostHoldPolicy | None:
    if not args.lost_hold or localizer is None or video_stream is None:
        return None
    lost_hold = LostHoldPolicy(
        max_attempts=int(args.lost_hold_max_attempts),
        timeout_s=max(0, int(args.lost_hold_timeout_ms)) / 1000.0,
        low_confidence_results=int(args.low_confidence_hold_results),
        hold_on_low_confidence=bool(args.hold_on_low_confidence),
    )
    recovery_action = "zero PCMD hover" if args.live else "freeze simulated frame"
    print(
        f"[operator] localization recovery ON: action={recovery_action}; "
        f"low-confidence threshold={lost_hold.low_confidence_results}; "
        "LOST first uses nearby EDM, then profile-scheduled MegaLoc; retry EDM "
        f"on the held frame up to {lost_hold.max_attempts}x "
        f"(timeout {lost_hold.timeout_s:.1f}s), then release",
        flush=True,
    )
    return lost_hold


_SITE_SWITCH_ASSET_ARGUMENTS = (
    "map_ply",
    "route_json",
    "bundle",
    "megaloc_cache",
    "reference_index",
    "reference_index_sha256",
    "track_landmarks",
    "localizer_backend",
    "localizer_deploy_dir",
    "localizer_profile",
    "bundle_sha256",
    "localizer_profile_sha256",
)


class _SiteSwitchParser:
    """Small argparse-compatible error boundary for a running Tk process."""

    @staticmethod
    def error(message: str) -> None:
        raise ValueError(message)


def _prepare_operator_site_runtime(
    base_args: argparse.Namespace,
    profile_path: str | Path,
) -> PreparedSiteRuntime:
    """Validate and load new-site assets without touching the active runtime."""
    args = argparse.Namespace(**vars(base_args))
    args.site_profile = str(Path(profile_path).expanduser().resolve())
    for name in _SITE_SWITCH_ASSET_ARGUMENTS:
        setattr(args, name, None)

    profile = app.resolve_operator_site_assets(args, _SiteSwitchParser())
    if profile is None:
        raise ValueError("site switch requires an explicit site profile")
    query_camera = profile.query_camera
    if query_camera is not None and (
        query_camera.width != STREAM_WIDTH or query_camera.height != STREAM_HEIGHT
    ):
        raise ValueError(
            "query_camera must describe the exact frame sent to localization: "
            f"profile={query_camera.width}x{query_camera.height}, "
            f"stream={STREAM_WIDTH}x{STREAM_HEIGHT}. Scale intrinsics or calibrate "
            "the actual stream; changing only the JSON dimensions is invalid."
        )

    hardware_approval = None
    if profile.hardware_approval is not None:
        hardware_approval = app.load_hardware_approval_receipt(profile.hardware_approval)
    _resolve_startup_localizer(args)
    route_path = Path(args.route_json) if args.route_json else None
    _route_hash, route_points, route_snapshot = _load_startup_route(
        route_path,
        profile,
    )
    points = _load_runtime_map(args, profile)
    replay_rows = _load_operator_replay(args)
    return PreparedSiteRuntime(
        args=args,
        interface_mode=InterfaceMode(args.interface_mode),
        profile=profile,
        hardware_approval=hardware_approval,
        map_points=points,
        route_points=tuple(route_points),
        mission_route_snapshot=route_snapshot,
        replay_rows=tuple(replay_rows),
    )


def _best_effort_close(resource: object | None) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        pass


def _discard_partial_operator_runtime(
    *,
    backend: object | None,
    video_stream: object | None,
    live_backend: object | None,
    localizer: object | None,
    detector: object | None,
    session_logs: SessionLogs | None,
) -> None:
    _best_effort_close(localizer)
    _best_effort_close(detector)
    cleanup = getattr(backend, "cleanup", None)
    if callable(cleanup):
        try:
            cleanup()
        except Exception:
            pass
    elif live_backend is None:
        _best_effort_close(video_stream)
    if session_logs is not None:
        try:
            session_logs.close(reason="site_start_failed")
        except Exception:
            pass


def _start_operator_site_runtime(
    prepared: PreparedSiteRuntime,
) -> ActiveSiteRuntime:
    """Create one complete runtime, cleaning partial resources on any failure."""
    args = prepared.args
    session_logs = None
    backend = None
    video_stream = None
    live_backend = None
    localizer = None
    detector = None
    try:
        session_logs, session_config = _create_operator_session(
            args,
            prepared.interface_mode,
            prepared.profile,
            prepared.hardware_approval,
        )
        backend, video_stream, live_backend = _build_operator_backend(
            args,
            session_logs,
            prepared.hardware_approval,
        )
        _start_operator_backend(backend, session_config, session_logs)
        localizer = _build_operator_localizer(
            args,
            prepared.profile,
            video_stream,
        )
        detector = _build_operator_detector(args, video_stream)
        lost_hold = _build_lost_hold_policy(args, localizer, video_stream)
        return ActiveSiteRuntime(
            prepared=prepared,
            session_logs=session_logs,
            backend=backend,
            video_stream=video_stream,
            live_backend=live_backend,
            localizer=localizer,
            detector=detector,
            lost_hold=lost_hold,
        )
    except BaseException:
        _discard_partial_operator_runtime(
            backend=backend,
            video_stream=video_stream,
            live_backend=live_backend,
            localizer=localizer,
            detector=detector,
            session_logs=session_logs,
        )
        raise


@dataclass(frozen=True)
class _OperatorLaunchInputs:
    args: argparse.Namespace
    site_profile: SiteProfile | None
    hardware_approval: object | None
    interface_mode: InterfaceMode
    route_points: list
    mission_route_snapshot: MissionRouteSnapshot | None
    points: np.ndarray


@dataclass(frozen=True)
class _OperatorLaunch:
    app: OperatorApp
    live_backend: object | None
    video_stream: FFmpegFrameStream | None
    localizer: LiveLocalizerClient | None
    detector: LiveDetectorClient | None
    session_logs: SessionLogs


def _prepare_operator_launch_inputs() -> _OperatorLaunchInputs | None:
    parser = build_argument_parser()
    args = _parse_operator_arguments(parser)
    site_profile, hardware_approval = _resolve_startup_site(args, parser)
    _resolve_live_safety(args, parser)
    route_path = Path(args.route_json) if args.route_json else None
    route_hash, route_points, mission_route_snapshot = _load_startup_route(
        route_path,
        site_profile,
    )
    if args.live_detect and not Path(args.detector_model).is_file():
        raise SystemExit(
            f"--live-detect set but model missing: {args.detector_model}; "
            "omit --live-detect (default) for localization-only"
        )
    _announce_interface(args)
    configure_offline_environment()
    interface_mode = InterfaceMode(args.interface_mode)
    install_network_guard(
        interface_mode,
        allowed_real_hosts=(args.ip,) if args.live else (),
    )
    _resolve_startup_localizer(args)
    _announce_site_assets(args, site_profile, route_path, route_hash)
    object.__setattr__(ANAFI, "stream_fps", float(args.stream_fps))
    points = _load_runtime_map(args, site_profile)
    if _run_operator_selftest(args, points):
        return None
    if _run_layout_selftest(
        args,
        points,
        interface_mode,
        UI_STANDARD_SIZE,
        UI_MIN_SIZE,
    ):
        return None
    return _OperatorLaunchInputs(
        args=args,
        site_profile=site_profile,
        hardware_approval=hardware_approval,
        interface_mode=interface_mode,
        route_points=route_points,
        mission_route_snapshot=mission_route_snapshot,
        points=points,
    )


def _main_site_runtime(
    inputs: _OperatorLaunchInputs,
    *,
    session_logs: SessionLogs,
    backend: object,
    video_stream: FFmpegFrameStream | None,
    live_backend: object | None,
    localizer: LiveLocalizerClient | None,
    detector: LiveDetectorClient | None,
    lost_hold: LostHoldPolicy | None,
    replay_rows: list[dict],
) -> ActiveSiteRuntime | None:
    if inputs.site_profile is None:
        return None
    prepared = PreparedSiteRuntime(
        args=inputs.args,
        interface_mode=inputs.interface_mode,
        profile=inputs.site_profile,
        hardware_approval=inputs.hardware_approval,
        map_points=inputs.points,
        route_points=tuple(inputs.route_points),
        mission_route_snapshot=inputs.mission_route_snapshot,
        replay_rows=tuple(replay_rows),
    )
    return ActiveSiteRuntime(
        prepared=prepared,
        session_logs=session_logs,
        backend=backend,
        video_stream=video_stream,
        live_backend=live_backend,
        localizer=localizer,
        detector=detector,
        lost_hold=lost_hold,
    )


def _build_operator_launch(inputs: _OperatorLaunchInputs) -> _OperatorLaunch:
    args = inputs.args
    session_logs, session_config = _create_operator_session(
        args,
        inputs.interface_mode,
        inputs.site_profile,
        inputs.hardware_approval,
    )
    backend, video_stream, live_backend = _build_operator_backend(
        args,
        session_logs,
        inputs.hardware_approval,
    )
    _start_operator_backend(backend, session_config, session_logs)
    localizer = _build_operator_localizer(args, inputs.site_profile, video_stream)
    detector = _build_operator_detector(args, video_stream)
    replay_rows = _load_operator_replay(args)
    lost_hold = _build_lost_hold_policy(args, localizer, video_stream)
    site_runtime = _main_site_runtime(
        inputs,
        session_logs=session_logs,
        backend=backend,
        video_stream=video_stream,
        live_backend=live_backend,
        localizer=localizer,
        detector=detector,
        lost_hold=lost_hold,
        replay_rows=replay_rows,
    )
    app = OperatorApp(
        backend,
        inputs.points,
        video_stream=video_stream,
        localizer=localizer,
        detector=detector,
        detect_every_n_frames=args.detect_every_n_frames,
        loc_every_n_frames=args.loc_every_n_frames,
        adaptive_loc_submit=bool(args.adaptive_loc_submit),
        replay_rows=replay_rows,
        tick_ms=args.tick_ms,
        boot_lock_ms=args.boot_lock_ms,
        lost_hold=lost_hold,
        pose_stabilize=bool(args.pose_stabilize),
        session_logs=session_logs,
        site_id=inputs.site_profile.site_id if inputs.site_profile is not None else "",
        site_profile_path=(inputs.site_profile.source if inputs.site_profile is not None else None),
        mission_route_snapshot=inputs.mission_route_snapshot,
        site_runtime=site_runtime,
        prepare_site_runtime=_prepare_operator_site_runtime,
        start_site_runtime=_start_operator_site_runtime,
    )
    return _OperatorLaunch(
        app=app,
        live_backend=live_backend,
        video_stream=video_stream,
        localizer=localizer,
        detector=detector,
        session_logs=session_logs,
    )


def _configure_operator_launch(launch: _OperatorLaunch, inputs: _OperatorLaunchInputs) -> None:
    app = launch.app
    app.route_pts = inputs.route_points
    if app.route_pts:
        print(
            f"[operator] mission route loaded but hidden: {len(app.route_pts)} waypoints "
            f"from {inputs.args.route_json}",
            flush=True,
        )
    if not inputs.args.auto_inspect:
        return

    def _auto_start_inspect() -> None:
        try:
            app.begin_auto_inspect()
            print(
                "[operator] auto-inspect: started localization feed "
                "(no flight-control command, no takeoff)",
                flush=True,
            )
        except Exception as exc:
            print(f"[operator] auto-inspect failed: {exc!r}", flush=True)

    app.after(2500, _auto_start_inspect)


def _close_operator_launch(launch: _OperatorLaunch) -> None:
    app = launch.app
    if app.__dict__.get("_shutdown_completed", False):
        return
    active_runtime = app.__dict__.get("_site_runtime")
    if active_runtime is not None:
        try:
            close_active_site_runtime(active_runtime, reason="mainloop_exit")
        except Exception as exc:
            print(f"[operator] runtime cleanup error: {exc!r}", flush=True)
        return
    _best_effort_close(launch.video_stream)
    _best_effort_close(launch.localizer)
    _best_effort_close(launch.detector)
    launch.session_logs.close(reason="mainloop_exit")


def _install_live_exit_safety(app: OperatorApp, emergency_cleanup) -> None:
    import flight_operator_app as _foa

    _foa.atexit.register(lambda: emergency_cleanup("atexit"))
    exit_in_progress = threading.Event()

    def _sig_handler(signum, _frame):
        if exit_in_progress.is_set():
            print(
                "[operator] exit already in progress; ignoring extra signal",
                flush=True,
            )
            return
        exit_in_progress.set()
        result = emergency_cleanup(f"signal_{signum}")
        if result is True:
            try:
                app.destroy()
            except Exception:
                pass
            raise SystemExit(0)
        exit_in_progress.clear()
        print(
            "[operator] exit safety could not confirm landing; "
            "keeping the interface and connection for retry",
            flush=True,
        )

    armed, failed = [], []
    for sig in (_foa.signal.SIGINT, _foa.signal.SIGTERM, _foa.signal.SIGHUP):
        try:
            _foa.signal.signal(sig, _sig_handler)
            armed.append(sig.name)
        except Exception as exc:
            failed.append(f"{sig.name}:{exc!r}")
    exit_safety_armed = EXIT_SAFETY_SIGNALS.issubset(set(armed))
    failure_text = f"  FAILED: {'; '.join(failed)}" if failed else ""
    print(
        f"[operator] exit safety armed: {'+'.join(armed) or 'NONE'}{failure_text}",
        flush=True,
    )
    if not exit_safety_armed:
        print(
            "[operator] WARNING: closing the terminal / Ctrl-C may NOT land "
            "the aircraft. Land manually before exiting.",
            flush=True,
        )
    try:
        app.backend.log.event(
            "exit_safety",
            ok=exit_safety_armed,
            armed=armed,
            failed=failed,
        )
    except Exception:
        pass


def main() -> None:
    inputs = _prepare_operator_launch_inputs()
    if inputs is None:
        return
    launch = _build_operator_launch(inputs)
    _configure_operator_launch(launch, inputs)
    app = launch.app

    # Exit safety: Ctrl-C / kill terminal / SIGTERM / SIGHUP / atexit all land.
    def _emergency_cleanup(reason: str = "signal") -> bool:
        print(f"[operator] exit safety ({reason}) -> land + restore sticks", flush=True)
        current_backend = app.__dict__.get("backend")
        coordinator = app.__dict__.get("_shutdown_coordinator")
        if coordinator is None:
            try:
                coordinator = OperatorShutdownCoordinator(
                    backend=current_backend,
                    session_logs=app.__dict__.get("session_logs"),
                    write_log=app.__dict__.get("write_log"),
                    destroy=None,
                    command_coordinator=app._get_command_coordinator(),
                    get_autonomy=lambda: app.__dict__.get("_integrated_autonomy"),
                )
                app._shutdown_coordinator = coordinator
            except Exception as exc:
                print(f"[operator] exit safety setup error: {exc!r}", flush=True)
                return False
        try:
            result = coordinator.shutdown(reason=reason)
        except Exception as exc:
            print(f"[operator] exit safety cleanup error: {exc!r}", flush=True)
            return False
        if result is not True:
            print(
                f"[operator] exit safety cleanup not confirmed: {result!r}",
                flush=True,
            )
        else:
            app._shutdown_completed = True
        return result is True

    _install_live_exit_safety(app, _emergency_cleanup)

    try:
        app.mainloop()
    finally:
        _close_operator_launch(launch)

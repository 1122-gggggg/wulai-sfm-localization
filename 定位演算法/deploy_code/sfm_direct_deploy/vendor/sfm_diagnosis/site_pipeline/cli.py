from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig
from .pipeline import ApprovalRequired, SitePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="site-sfm-pipeline",
        description="Graph-aware, segment-centric multi-video SfM pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="Create immutable corpus inventory")
    initialize.add_argument("--config", type=Path, required=True)
    initialize.add_argument("--corpus", type=Path, required=True)
    initialize.add_argument("--metadata", type=Path)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--force", action="store_true")

    run = commands.add_parser("run", help="Run or resume pipeline stages")
    run.add_argument("--run", type=Path, required=True)
    run.add_argument("--from-stage")
    run.add_argument("--to-stage")
    run.add_argument("--force", action="store_true")

    status = commands.add_parser("status", help="Show stage and approval status")
    status.add_argument("--run", type=Path, required=True)

    approve = commands.add_parser("approve-final", help="Approve exact Final input")
    approve.add_argument("--run", type=Path, required=True)
    approve.add_argument("--decision-sha", required=True)
    approve.add_argument("--approver", required=True)

    export = commands.add_parser("export", help="Regenerate final manifests")
    export.add_argument("--run", type=Path, required=True)

    backfill = commands.add_parser("backfill", help="Import legacy manifests as candidate evidence")
    backfill.add_argument("--run", type=Path, required=True)
    backfill.add_argument("--legacy-run", type=Path, required=True)

    normalize = commands.add_parser(
        "normalize-selection",
        help="Remove zero-degree keyframes before final approval",
    )
    normalize.add_argument("--run", type=Path, required=True)

    enrich = commands.add_parser(
        "enrich-bridge",
        help="Add evidence-selected bridge keyframes and invalidate prior products",
    )
    enrich.add_argument("--run", type=Path, required=True)
    enrich.add_argument("--segment", action="append", default=[])
    enrich.add_argument("--keyframe", action="append", default=[])
    enrich.add_argument("--attempt-name", required=True)

    layers = commands.add_parser(
        "materialize-layers",
        help="Create robust base geometry and retain the dense localization layer",
    )
    layers.add_argument("--run", type=Path, required=True)
    layers.add_argument("--export-ply", action="store_true")

    validate = commands.add_parser(
        "validate-layers",
        help="Run strict robust/dense localization and fuse the results",
    )
    validate.add_argument("--run", type=Path, required=True)
    validate.add_argument("--maximum-position-normalized", type=float, default=0.02)
    validate.add_argument("--maximum-rotation-deg", type=float, default=2.0)
    validate.add_argument("--target-rate", type=float, default=0.95)
    validate.add_argument("--outer-holdout-frozen", action="store_true")

    fuse = commands.add_parser(
        "fuse-localization",
        help="Fuse strict robust/dense localization validations",
    )
    fuse.add_argument("--run", type=Path, required=True)
    fuse.add_argument("--robust-validation", type=Path, required=True)
    fuse.add_argument("--dense-validation", type=Path, required=True)
    fuse.add_argument("--scene-scale", type=float)
    fuse.add_argument("--maximum-position-normalized", type=float, default=0.02)
    fuse.add_argument("--maximum-rotation-deg", type=float, default=2.0)
    fuse.add_argument("--target-rate", type=float, default=0.95)
    fuse.add_argument("--outer-holdout-frozen", action="store_true")

    refinement = commands.add_parser(
        "refine-map", help="Run an immutable PixSfM or Dense-SfM candidate refinement"
    )
    refinement.add_argument(
        "--backend", choices=("pixsfm", "densesfm-refine", "densesfm-full"), required=True
    )
    refinement.add_argument("--input-model", type=Path, required=True)
    refinement.add_argument("--images", type=Path, required=True)
    refinement.add_argument("--pairs", type=Path)
    refinement.add_argument("--intrinsics", type=Path, required=True)
    refinement.add_argument("--run", type=Path, required=True)
    refinement.add_argument("--cache-dir", type=Path)
    refinement.add_argument("--runtime-python", type=Path, required=True)
    refinement.add_argument("--runtime-root", type=Path)
    refinement.add_argument("--continuation-receipt", type=Path)
    refinement.add_argument("--timeout-seconds", type=int, default=86_400)
    refinement.add_argument("--max-cache-gb", type=int, default=500)
    refinement.add_argument("--pixsfm-patch-size", type=int, default=4)
    refinement.add_argument("--max-reprojection-error-px", type=float, default=3.0)
    refinement.add_argument("--minimum-triangulation-angle-deg", type=float, default=1.5)
    refinement.add_argument("--minimum-track-length", type=int, default=3)
    refinement.add_argument("--bundle-adjustment-iterations", type=int, default=100)
    refinement.add_argument("--bundle-adjustment-threads", type=int, default=8)
    refinement.add_argument("--dry-run", action="store_true")
    refinement.add_argument("--resume", action="store_true")

    mvroma = commands.add_parser(
        "mvroma-augment", help="Add strictly gated MV-RoMa tracks to an immutable map copy"
    )
    mvroma.add_argument("--scope", choices=("targeted", "full"), required=True)
    mvroma.add_argument("--input-model", type=Path, required=True)
    mvroma.add_argument("--images", type=Path, required=True)
    mvroma.add_argument("--selection", type=Path, required=True)
    mvroma.add_argument("--intrinsics", type=Path, required=True)
    mvroma.add_argument("--run", type=Path, required=True)
    mvroma.add_argument("--runtime-python", type=Path, required=True)
    mvroma.add_argument("--runtime-root", type=Path, required=True)
    mvroma.add_argument("--weight", type=Path, required=True)
    mvroma.add_argument("--continuation-receipt", type=Path)
    mvroma.add_argument("--coarse-height", type=int, default=378)
    mvroma.add_argument("--coarse-width", type=int, default=672)
    mvroma.add_argument("--target-height", type=int, default=756)
    mvroma.add_argument("--target-width", type=int, default=1344)
    mvroma.add_argument("--certainty-threshold", type=float, default=0.5)
    mvroma.add_argument("--max-samples-per-group", type=int, default=1024)
    mvroma.add_argument("--timeout-seconds", type=int, default=86_400)
    mvroma.add_argument("--sparse-video", action="append", default=[])
    mvroma.add_argument("--weak-keyframe", action="append", default=[])
    mvroma.add_argument("--dry-run", action="store_true")
    mvroma.add_argument("--resume", action="store_true")

    direct = commands.add_parser(
        "direct-map",
        help="Map every adaptively sampled sequence in one native GLUEMAP job",
    )
    direct.add_argument("--site-name", required=True)
    direct.add_argument("--corpus", type=Path, required=True)
    direct.add_argument("--run", type=Path, required=True)
    direct.add_argument("--intrinsics", type=Path, required=True)
    direct.add_argument("--gluemap-root", type=Path, required=True)
    direct.add_argument("--gluemap-config", type=Path, required=True)
    direct.add_argument("--workspace-root", type=Path, required=True)
    direct.add_argument("--megaloc-source", type=Path, required=True)
    direct.add_argument("--megaloc-checkpoint", type=Path, required=True)
    direct.add_argument("--edm-root", type=Path)
    direct.add_argument("--edm-checkpoint", type=Path)
    direct.add_argument("--probe-fps", type=float, default=4.0)
    direct.add_argument("--baseline-fps", type=float, default=1.0)
    direct.add_argument("--fast-fps", type=float, default=2.0)
    direct.add_argument("--hover-fps", type=float, default=0.2)
    direct.add_argument("--rotation-fps", type=float, default=2.0)
    direct.add_argument("--min-gap-seconds", type=float, default=0.25)
    direct.add_argument("--preprocess-only", action="store_true")
    direct.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "direct-map":
            from .direct_mapping import (
                DirectMappingRequest,
                DirectMappingRuntime,
                run_direct_mapping,
            )
            from .preprocessing import DirectSamplingPolicy

            request = DirectMappingRequest(
                site_name=args.site_name,
                corpus_root=args.corpus,
                run_dir=args.run,
                intrinsics_path=args.intrinsics,
                policy=DirectSamplingPolicy(
                    probe_fps=args.probe_fps,
                    baseline_fps=args.baseline_fps,
                    fast_fps=args.fast_fps,
                    hover_fps=args.hover_fps,
                    rotation_fps=args.rotation_fps,
                    min_gap_seconds=args.min_gap_seconds,
                ),
            )
            runtime = DirectMappingRuntime(
                gluemap_root=args.gluemap_root,
                base_config_path=args.gluemap_config,
                workspace_root=args.workspace_root,
                megaloc_source=args.megaloc_source,
                megaloc_checkpoint=args.megaloc_checkpoint,
                edm_root=args.edm_root,
                edm_checkpoint=args.edm_checkpoint,
            )
            payload = run_direct_mapping(
                request,
                runtime,
                resume=args.resume,
                preprocess_only=args.preprocess_only,
            )
        elif args.command == "init":
            pipeline = SitePipeline(PipelineConfig.from_toml(args.config))
            result = pipeline.initialize(
                args.corpus,
                args.output,
                metadata=args.metadata,
                force=args.force,
            )
            payload = _result_payload(result)
        elif args.command == "backfill":
            from .migration import backfill_legacy_run

            receipt = backfill_legacy_run(args.legacy_run, args.run)
            payload = {"backfill_receipt": str(receipt)}
        elif args.command == "normalize-selection":
            from .selection_tools import normalize_run_selection

            payload = {"normalization_receipt": str(normalize_run_selection(args.run))}
        elif args.command == "enrich-bridge":
            from .selection_tools import enrich_run_selection

            payload = {
                "bridge_enrichment_receipt": str(
                    enrich_run_selection(
                        args.run,
                        target_segments=set(args.segment),
                        target_keyframes=set(args.keyframe),
                        attempt_name=args.attempt_name,
                    )
                )
            }
        elif args.command == "materialize-layers":
            from .release import materialize_layers

            payload = {
                "robust_filter_receipt": str(
                    materialize_layers(args.run, export_ply=args.export_ply)
                )
            }
        elif args.command == "validate-layers":
            from .release import validate_localization_layers

            payload = {
                "ensemble_validation_receipt": str(
                    validate_localization_layers(
                        args.run,
                        maximum_position_normalized=args.maximum_position_normalized,
                        maximum_rotation_deg=args.maximum_rotation_deg,
                        target_rate=args.target_rate,
                        outer_holdout_frozen=args.outer_holdout_frozen,
                    )
                )
            }
        elif args.command == "fuse-localization":
            from .release import fuse_localization_results

            payload = {
                "ensemble_validation_receipt": str(
                    fuse_localization_results(
                        args.run,
                        args.robust_validation,
                        args.dense_validation,
                        scene_scale=args.scene_scale,
                        maximum_position_normalized=args.maximum_position_normalized,
                        maximum_rotation_deg=args.maximum_rotation_deg,
                        target_rate=args.target_rate,
                        outer_holdout_frozen=args.outer_holdout_frozen,
                    )
                )
            }
        elif args.command == "refine-map":
            from .map_refinement import RefinementRequest, run_refinement
            from .robust_filter import RobustFilterConfig

            cache_dir = args.cache_dir or (args.run.parent / "refinement_cache" / args.backend)
            request = RefinementRequest(
                backend=args.backend,
                input_model=args.input_model,
                images=args.images,
                intrinsics=args.intrinsics,
                pairs=args.pairs,
                run_dir=args.run,
                cache_dir=cache_dir,
                runtime_python=args.runtime_python,
                runtime_root=args.runtime_root,
                continuation_receipt=args.continuation_receipt,
                timeout_seconds=args.timeout_seconds,
                max_cache_gb=args.max_cache_gb,
                pixsfm_patch_size=args.pixsfm_patch_size,
                robust_filter=RobustFilterConfig(
                    max_reprojection_error_px=args.max_reprojection_error_px,
                    minimum_triangulation_angle_deg=args.minimum_triangulation_angle_deg,
                    minimum_track_length=args.minimum_track_length,
                    bundle_adjustment_iterations=args.bundle_adjustment_iterations,
                    bundle_adjustment_threads=args.bundle_adjustment_threads,
                ),
            )
            payload = run_refinement(request, dry_run=args.dry_run, resume=args.resume)
        elif args.command == "mvroma-augment":
            from river_mvroma.runner import (
                RIVER_V3_SPARSE_VIDEOS,
                RIVER_V3_WEAK_KEYFRAMES,
                MVRoMaRequest,
                run_mvroma,
            )

            request = MVRoMaRequest(
                scope=args.scope,
                input_model=args.input_model,
                images=args.images,
                selection=args.selection,
                intrinsics=args.intrinsics,
                run_dir=args.run,
                runtime_python=args.runtime_python,
                runtime_root=args.runtime_root,
                weight=args.weight,
                continuation_receipt=args.continuation_receipt,
                coarse_size=(args.coarse_height, args.coarse_width),
                target_size=(args.target_height, args.target_width),
                certainty_threshold=args.certainty_threshold,
                max_samples_per_group=args.max_samples_per_group,
                sparse_video_ids=tuple(args.sparse_video) or RIVER_V3_SPARSE_VIDEOS,
                weak_keyframe_ids=tuple(args.weak_keyframe) or RIVER_V3_WEAK_KEYFRAMES,
                timeout_seconds=args.timeout_seconds,
            )
            payload = run_mvroma(request, dry_run=args.dry_run, resume=args.resume)
        else:
            pipeline = SitePipeline.open(args.run)
            if args.command == "run":
                payload = _result_payload(
                    pipeline.run(
                        args.run,
                        from_stage=args.from_stage,
                        to_stage=args.to_stage,
                        force=args.force,
                    )
                )
            elif args.command == "status":
                payload = pipeline.status(args.run)
            elif args.command == "approve-final":
                approval = pipeline.approve_final(
                    args.run, args.decision_sha, approver=args.approver
                )
                payload = {"approval": str(approval)}
            else:
                payload = _result_payload(pipeline.export(args.run))
    except ApprovalRequired as error:
        print(json.dumps({"status": "APPROVAL_REQUIRED", "reason": str(error)}, indent=2))
        return 3
    except (OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {"status": "FAILED", "error_type": type(error).__name__, "reason": str(error)},
                indent=2,
            )
        )
        return 2
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _result_payload(result) -> dict[str, object]:
    return {
        "run_dir": str(result.run_dir),
        "status": result.status,
        "receipt": str(result.receipt),
        "stages": [{"name": stage.name, "status": stage.status} for stage in result.stages],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]

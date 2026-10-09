#!/usr/bin/env python3
"""Offline independent VO + fresh global EDM anchors; never applies a correction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '定位演算法/deploy_code_expand/sfm_direct_deploy'))
from direct_map import DirectMapAssets  # noqa: E402
from direct_paths import default_runtime_paths, vendor_sys_path  # noqa: E402
from direct_profile import load_direct_profile  # noqa: E402
from presubmap import PreAnchorSubmap  # noqa: E402
from workmap_closure_experiment import (  # noqa: E402
    SIZE, apply_frozen_gates, camera_matrix, frozen_geometry, solve_pnp, verify_frozen_points,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--first-anchor', required=True, type=int)
    parser.add_argument('--last-anchor', required=True, type=int)
    parser.add_argument('--output-dir', required=True, type=Path)
    return parser.parse_args()


def _select_sequence(protocol, name):
    sequences = [s for s in protocol['sequences'] if s['name'] == name]
    if len(sequences) != 1:
        raise ValueError('sequence must identify exactly one video')
    return sequences[0]


def _bind_inputs(protocol_file, protocol, sequence):
    def asset(value):
        path = Path(value)
        return path if path.is_absolute() else protocol_file.resolve().parent / path
    video = asset(sequence['video'])
    bundle_path = asset(protocol['bundle']['path'])
    profile_path = asset(protocol['profile']['path'])
    bound = {video: sequence['video_sha256'], bundle_path: protocol['bundle']['sha256'],
             profile_path: protocol['profile']['sha256']}
    return video, bundle_path, profile_path, bound


def _verify_bound_digests(bound, message):
    for path, expected in bound.items():
        if digest(path) != expected:
            raise ValueError(f'{message}: {path}')


def _build_provider(bundle_path, profile_path, protocol, output):
    vendor_sys_path()
    from live_provider import LiveMapEDMProvider
    from sfm_diagnosis.site_pipeline.deployment_localizer import (
        _configure_audited_runtime_site_packages, _resolve_audited_site_packages,
    )
    _configure_audited_runtime_site_packages(_resolve_audited_site_packages(None))
    paths = default_runtime_paths()
    provider = LiveMapEDMProvider(
        DirectMapAssets.load(bundle_path, expected_sha256=protocol['bundle']['sha256']),
        load_direct_profile(profile_path, expected_sha256=protocol['profile']['sha256']),
        edm_repo=paths.edm_repo, edm_checkpoint=paths.edm_checkpoint,
        boq_weights=paths.boq_weights, cache_dir=output / 'provider_cache',
    )
    provider.ensure_models()
    matched = []
    original_match = provider._match_and_lift
    def capture_match(*positional, **keywords):
        result = original_match(*positional, **keywords)
        matched[:] = result[0]
        return result
    provider._match_and_lift = capture_match
    return provider, matched


def _measure_anchor(ordinal, gray, local, provider, matched, K, frozen_ids, frozen_xyz, profile_json):
    raw_ids = -(local._ids + 1)
    local_xyz, local_xy = local._xyz.copy(), local._xy.copy()
    local_fit = solve_pnp(raw_ids, local_xyz, local_xy, K, minimum=30, seed=0, max_error_px=3.)
    matched.clear()
    fix = provider.localize_array(gray, scope='global')
    rows = [row for row in matched if row.point3d_id is not None and row.point3d_id > 0]
    ids = np.array([row.point3d_id for row in rows], dtype=np.int64)
    xy = np.array([row.query_xy for row in rows], dtype=np.float64).reshape(-1, 2)
    xyz = provider._point_xyz_for_ids(ids).reshape(-1, 3)
    verify_frozen_points(ids, xyz, frozen_ids, frozen_xyz)
    limits = profile_json['reloc']['frozen_pnp_thresholds']
    main_fit = apply_frozen_gates(solve_pnp(
        ids, xyz, xy, K, minimum=int(limits['strong_inliers']), seed=0,
        max_error_px=float(limits['maximum_reprojection_p90_px'])), limits)
    valid = bool(fix.ok and local_fit['ok'] and main_fit['ok'] and ordinal in local.poses)
    return {'raw_ids': raw_ids, 'local_xyz': local_xyz, 'local_xy': local_xy,
            'local_fit': local_fit, 'fix': fix, 'ids': ids, 'xy': xy, 'xyz': xyz,
            'main_fit': main_fit, 'valid': valid}


def _anchor_record(measured, ordinal, stamp, K, sequence):
    local_fit, main_fit = measured['local_fit'], measured['main_fit']
    li = np.array(local_fit['inlier_indices'], dtype=int)
    mi = np.array(main_fit['inlier_indices'], dtype=int)
    return {
        'ordinal': ordinal, 'method': 'independent_VO/fresh_global_EDM',
        'submap_id': 1, 'submap_kind': 'independent', 'source_epoch': 0,
        'local_pose': local_fit['cam_from_world'], 'map_pose': main_fit['cam_from_world'],
        'local_inliers': len(li), 'main_inliers': len(mi),
        'local_point_ids': measured['raw_ids'][li].tolist(),
        'main_point_ids': measured['ids'][mi].tolist(),
        'local_xyz': measured['local_xyz'][li].tolist(), 'local_xy': measured['local_xy'][li].tolist(),
        'main_xyz': measured['xyz'][mi].tolist(), 'main_xy': measured['xy'][mi].tolist(),
        'K': K.tolist(), 'image_size_px': list(SIZE),
        'provenance': {'video_sha256': sequence['video_sha256'], 'video_time_s': stamp,
            'capture_identity': 'offline decoded video ordinal; not a live monoclock capture',
            'local_ids': 'bijective negative session namespace: -(real PreAnchorSubmap ID + 1)',
            'local_coordinates': 'capture-time premerge independent VO; no published/main pose seed',
            'main_coordinates': 'immutable bundle positive-ID XYZ; raw fresh EDM matches',
            'solver': main_fit['solver'], 'global_reference_scope': 'global',
            'KLT_main_anchor_propagation': False},
    }


def _scan_video(args, sequence, cap, local, provider, matched, K, frozen_ids, frozen_xyz,
                profile_json):
    records, attempts = [], []
    for ordinal in range(args.last_anchor + 1):
        ok, bgr = cap.read()
        if not ok:
            raise ValueError(f'video ended before requested ordinal {ordinal}')
        gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), SIZE, interpolation=cv2.INTER_AREA)
        stamp = ordinal / float(sequence['fps'])
        if ordinal % 2 == 0:
            # Synchronous real VO: no main pose/scale prior enters initialization or tracking.
            local.step(ordinal, stamp, gray)
        if ordinal < args.first_anchor or ordinal % 2:
            continue
        measured = _measure_anchor(
            ordinal, gray, local, provider, matched, K, frozen_ids, frozen_xyz, profile_json)
        local_fit, main_fit, fix = measured['local_fit'], measured['main_fit'], measured['fix']
        valid = measured['valid']
        attempts.append({'ordinal': ordinal, 'video_time_s': stamp, 'global_status': fix.status,
                         'global_ok': bool(fix.ok), 'local_pnp': local_fit, 'main_pnp': main_fit,
                         'paired_anchor': valid, 'local_state': local.state})
        if valid:
            records.append(_anchor_record(measured, ordinal, stamp, K, sequence))
        print(json.dumps({'ordinal': ordinal, 'paired_anchor': valid,
                          'local_inliers': local_fit['inlier_count'],
                          'main_inliers': main_fit['inlier_count'], 'global_status': fix.status}), flush=True)
    return records, attempts


def main() -> int:
    args = _parse_args()
    if not 0 <= args.first_anchor <= args.last_anchor or args.last_anchor - args.first_anchor > 48:
        raise ValueError('anchor window must be nonnegative and <=48 frames')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    sequence = _select_sequence(protocol, args.sequence)
    video, bundle_path, profile_path, bound = _bind_inputs(args.protocol, protocol, sequence)
    _verify_bound_digests(bound, 'input SHA256 mismatch')
    bundle = json.loads(bundle_path.read_text())
    profile_json = json.loads(profile_path.read_text())
    K = camera_matrix(protocol, profile_json)
    frozen_ids, frozen_xyz, geometry = frozen_geometry(bundle_path, bundle)
    provider, matched = _build_provider(bundle_path, profile_path, protocol, output)
    local = PreAnchorSubmap(K)
    cv2.setRNGSeed(0)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('video did not open')
    try:
        records, attempts = _scan_video(
            args, sequence, cap, local, provider, matched, K, frozen_ids, frozen_xyz, profile_json)
    finally:
        cap.release()
        local.close()
    _verify_bound_digests(bound, 'input changed during experiment')
    frozen_geometry(bundle_path, bundle)
    (output / 'anchors.json').write_text(json.dumps(
        {'schema': 'workmap-closure-anchors/v1', 'records': records}, indent=2, allow_nan=False) + '\n')
    (output / 'attempts.json').write_text(json.dumps(attempts, indent=2, allow_nan=False) + '\n')
    manifest = {'schema': 'workmap-independent-closure-experiment/v1',
                'sequence': args.sequence, 'first_anchor': args.first_anchor, 'last_anchor': args.last_anchor,
                'anchors': len(records), 'source_sha256': digest(Path(__file__)),
                'input_hashes': {str(path): value for path, value in bound.items()},
                'frozen_geometry': geometry, 'frozen_inputs_unchanged': True,
                'corrections_applied': False, 'independent_ground_truth': False,
                'scheduling': 'synchronous offline VO/EDM, not a frontend throughput benchmark'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

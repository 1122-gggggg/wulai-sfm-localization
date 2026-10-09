"""Differential oracle for the unchanged pre-cache lifting and greedy dedup path."""

from pathlib import Path
from typing import Any
import os

import numpy as np
import pytest

from live_provider import LiveMapEDMProvider
from river_map_quality import official_edm_adapter as adapter
from sfm_diagnosis.site_pipeline import deployment_localizer as vendor
from sfm_diagnosis.site_pipeline.deployment_localizer import (
    _match_official_prepared,
    _match_official_prepared_batch,
    _two_view_inliers,
    _valid_matches,
)


# Frozen pre-extraction body. Keep independent of the production hook/cache.
def _original_match_and_lift(
    self,
    query_path: Path,
    ranked: tuple[int, ...],
    *,
    query_image: Any = None,
    prepared_query: Any = None,
):
    import cv2
    from river_map_quality.official_edm_adapter import (
        deduplicate_lifted_matches,
        lift_reference_matches,
        lift_unmapped_with_depth,
        prepare_official_megadepth_image_from_array,
    )
    from river_map_quality.official_edm_adapter_loo import (
        prepare_official_megadepth_image,  # noqa: F401 - frozen oracle import
    )

    if query_image is None:
        query_image = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
        if query_image is None:
            raise FileNotFoundError(query_path)
    lifted = []
    raw_matches = 0
    matcher = self._matcher_runtime()
    if prepared_query is None:
        # Change 2: Prepare query directly from in-memory array, avoiding redundant disk read.
        # Cannot change decision because preprocessing follows the deterministic official transform.
        prepared_query = prepare_official_megadepth_image_from_array(query_image, matcher)
    threshold = float(self.edm_config.get("confidence_threshold") or 0.0)

    # Change 2: Retrieve prepared references from in-memory LRU cache.
    # Removes repeated cv2.imread and resizing of 2688x1512 reference JPEGs across calls.
    # Cannot change decision because the prepared arrays and native shapes are deterministic
    # and bitwise identical to on-the-fly preparation.
    ref_items = []
    for reference_index in ranked:
        reference_path = self._reference_paths[reference_index]
        name = self._reference_names[reference_index]
        prep_ref, ref_shape = self._get_prepared_reference(name, reference_path, matcher)
        ref_items.append((reference_index, name, prep_ref, ref_shape))

    # Change 1: Batch top_k reference pairs into a single EDM forward when EDM_BATCH_REFS=1.
    # Removes sequential model forward passes (saving ~33 ms on 5090, ~132 ms on 5060).
    # Cannot change decision because EDM natively processes mini-batches with separable outputs.
    # Gated behind EDM_BATCH_REFS (default 0) to guard against GPU cuBLAS floating-point
    # reduction order differences across different batch sizes on GPU.
    batch_enabled = os.environ.get("EDM_BATCH_REFS", "0") == "1"
    if batch_enabled and len(ref_items) > 1:
        matched_list = getattr(self, "_match_prepared_batch", _match_official_prepared_batch)(
            matcher, prepared_query, [item[2] for item in ref_items]
        )
    else:
        matched_list = [
            getattr(self, "_match_prepared", _match_official_prepared)(
                matcher, prepared_query, item[2]
            )
            for item in ref_items
        ]

    for (reference_index, name, prepared_reference, reference_shape), matched in zip(
        ref_items, matched_list, strict=True
    ):
        query_points = np.asarray(matched["mkpts0_f"], dtype=float).reshape(-1, 2)
        reference_points = np.asarray(matched["mkpts1_f"], dtype=float).reshape(-1, 2)
        confidences = np.asarray(matched["mconf"], dtype=float).reshape(-1)
        valid = _valid_matches(
            query_points,
            reference_points,
            confidences,
            query_shape=query_image.shape,
            reference_shape=reference_shape,
            confidence_threshold=threshold,
        )
        query_points = query_points[valid]
        reference_points = reference_points[valid]
        confidences = confidences[valid]
        geometry = _two_view_inliers(query_points, reference_points, threshold_px=3.0)
        query_points = query_points[geometry]
        reference_points = reference_points[geometry]
        confidences = confidences[geometry]
        raw_matches += len(query_points)
        observation_xy, observation_ids = self._observations[name]
        lift_result = lift_reference_matches(
            query_points=query_points,
            reference_points=reference_points,
            observation_points=observation_xy,
            observation_point3d_ids=observation_ids,
            confidences=confidences,
            maximum_distance_px=self.lift_distance_px,
            reference_name=name,
        )
        lifted.extend(lift_result.matches)
        depth = self._depth_for_reference(name)
        if depth is not None and lift_result.unmapped_pairs:
            fx, fy, cx, cy = self._camera_params[name]
            lifted.extend(
                lift_unmapped_with_depth(
                    unmapped=lift_result.unmapped_pairs,
                    depth=depth,
                    native_wh=self._native_wh[name],
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    cam_from_world_3x4=self._cam_from_world[name],
                    median_sparse_z=self._median_sparse_z[name],
                )
            )
    deduplicated = deduplicate_lifted_matches(
        lifted,
        query_conflict_distance_px=1.0,
    )
    return deduplicated.matches, raw_matches


@pytest.mark.parametrize("depth", [False, True])
def test_cached_reference_lifts_match_original_overlap_boundary_oracle(monkeypatch, depth):
    provider = object.__new__(LiveMapEDMProvider)
    provider._query_lifts = {}
    provider._reference_names = ("a", "b", "c", "d")
    provider._reference_paths = tuple(Path(name) for name in provider._reference_names)
    provider.edm_config = {"confidence_threshold": 0.0}
    provider.lift_distance_px = 2.0
    provider._matcher_runtime = lambda: None
    provider._get_prepared_reference = lambda name, path, runtime: (name, (32, 32))
    provider._observations = {
        name: (
            np.array([[5.0, 5.0], [5.0, 5.0], [15.0, 15.0]]),
            np.array([10 + i, 100 + i, 20 + i]),
        )
        for i, name in enumerate(provider._reference_names)
    }
    provider._camera_params = {name: (30.0, 30.0, 16.0, 16.0) for name in provider._reference_names}
    provider._native_wh = {name: (32, 32) for name in provider._reference_names}
    provider._cam_from_world = {
        name: np.column_stack((np.eye(3), np.zeros(3))) for name in provider._reference_names
    }
    provider._median_sparse_z = {name: 2.0 for name in provider._reference_names}
    provider._depth_for_reference = lambda name: np.full((32, 32), 2.0) if depth else None
    # Query conflicts straddle one-pixel boundaries; reference lifts exercise
    # equal-distance point-ID ties, repeated IDs, unmapped/depth and invalid rows.
    matched = {
        name: {
            "mkpts0_f": np.array(
                [
                    [5.0 + i * 0.4, 5.0],
                    [5.9 + i * 0.4, 5.0],
                    [15.0, 15.0],
                    [24.0, 24.0],
                    [-1.0, 2.0],
                ]
            ),
            "mkpts1_f": np.array([[5.0, 5.0], [5.0, 5.0], [15.0, 15.0], [24.0, 24.0], [2.0, 2.0]]),
            "mconf": np.array([0.8 - i * 0.1, 0.7, 0.6, 0.5, 0.9]),
        }
        for i, name in enumerate(provider._reference_names)
    }
    provider._match_prepared_batch = lambda runtime, query, refs: [matched[name] for name in refs]
    monkeypatch.setenv("EDM_BATCH_REFS", "1")
    monkeypatch.setattr(
        vendor, "_two_view_inliers", lambda q, r, threshold_px: np.ones(len(q), dtype=bool)
    )
    # Oracle resolves its global geometry helper independently.
    monkeypatch.setitem(
        _original_match_and_lift.__globals__, "_two_view_inliers", vendor._two_view_inliers
    )
    image = np.zeros((32, 32), dtype=np.uint8)
    expected = _original_match_and_lift(
        provider, Path("query"), (0, 1, 2, 3), query_image=image, prepared_query=object()
    )
    calls = []
    actual_lift = adapter.lift_reference_matches

    def record_lift(**kwargs):
        calls.append(kwargs["reference_name"])
        return actual_lift(**kwargs)

    monkeypatch.setattr(adapter, "lift_reference_matches", record_lift)
    provider._match_and_lift(Path("query"), (0, 1), query_image=image, prepared_query=object())
    cached_baseline = {
        name: (tuple(lifted), raw) for name, (lifted, raw) in provider._query_lifts.items()
    }
    actual = provider._match_and_lift(
        Path("query"), (0, 1, 2, 3), query_image=image, prepared_query=object()
    )
    assert actual == expected
    assert calls == ["a", "b", "c", "d"]
    assert all(
        (tuple(provider._query_lifts[name][0]), provider._query_lifts[name][1]) == original
        for name, original in cached_baseline.items()
    )
    provider._query_lifts.clear()
    assert (
        provider._match_and_lift(
            Path("query"), (0, 1, 2, 3), query_image=image, prepared_query=object()
        )
        == expected
    )
    assert calls == ["a", "b", "c", "d"] * 2

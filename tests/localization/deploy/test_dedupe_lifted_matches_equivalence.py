"""Differential test for the vectorised `deduplicate_lifted_matches`.

The pre-vectorisation scalar body is reproduced here as the oracle.  The greedy
admission is order dependent, so the deployed implementation is only admissible
if it yields the same accepted tuple and the same counters on every input,
including exact ties and matches sitting exactly on the conflict radius.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

# Source modules are supplied by the repository's pytest pythonpath.
import numpy as np
import pytest

from river_map_quality.official_edm_adapter import (
    DeduplicatedMatches,
    LiftedMatch,
    deduplicate_lifted_matches,
)


def scalar_deduplicate_lifted_matches(
    matches: Sequence[LiftedMatch],
    *,
    query_conflict_distance_px: float,
) -> DeduplicatedMatches:
    """The pre-vectorisation body, unchanged. Oracle only; never imported by runtime."""

    if not math.isfinite(query_conflict_distance_px) or query_conflict_distance_px <= 0:
        raise ValueError("query_conflict_distance_px must be finite and positive")
    ordered = sorted(
        matches,
        key=lambda match: (
            0 if match.point3d_id is not None else 1,
            -match.confidence,
            match.lift_distance_px,
            match.point3d_id if match.point3d_id is not None else -1,
            match.reference_name,
            match.query_xy,
        ),
    )
    grid: dict[tuple[int, int], list[LiftedMatch]] = defaultdict(list)
    accepted: list[LiftedMatch] = []
    point_ids: set[int] = set()
    conflicts = 0
    duplicates = 0
    radius_sq = query_conflict_distance_px * query_conflict_distance_px
    for match in ordered:
        x_cell = math.floor(match.query_xy[0] / query_conflict_distance_px)
        y_cell = math.floor(match.query_xy[1] / query_conflict_distance_px)
        nearby = [
            earlier
            for x_offset in (-1, 0, 1)
            for y_offset in (-1, 0, 1)
            for earlier in grid.get((x_cell + x_offset, y_cell + y_offset), ())
            if (earlier.query_xy[0] - match.query_xy[0]) ** 2
            + (earlier.query_xy[1] - match.query_xy[1]) ** 2
            <= radius_sq
        ]
        if any(earlier.point3d_id != match.point3d_id for earlier in nearby):
            conflicts += 1
            continue
        if nearby or (match.point3d_id is not None and match.point3d_id in point_ids):
            duplicates += 1
            continue
        accepted.append(match)
        if match.point3d_id is not None:
            point_ids.add(match.point3d_id)
        grid[(x_cell, y_cell)].append(match)
    return DeduplicatedMatches(
        matches=tuple(accepted),
        conflicting_query_match_count=conflicts,
        duplicate_point3d_match_count=duplicates,
    )


def _random_matches(
    rng: np.random.Generator, count: int, *, extent: float, id_count: int
) -> list[LiftedMatch]:
    # Half-pixel quantised coordinates and few distinct confidences / distances
    # force exact sort-key ties and pairs lying exactly on the conflict radius.
    xy = np.round(rng.uniform(-extent, extent, size=(count, 2)) * 2.0) / 2.0
    confidences = rng.integers(0, 4, size=count) / 4.0
    lifts = rng.integers(0, 3, size=count) / 2.0
    ids = rng.integers(-1, id_count, size=count)
    return [
        LiftedMatch(
            query_xy=(float(xy[row, 0]), float(xy[row, 1])),
            point3d_id=None if ids[row] < 0 else int(ids[row]),
            reference_name=f"ref{int(rng.integers(0, 3))}",
            confidence=float(confidences[row]),
            lift_distance_px=float(lifts[row]),
        )
        for row in range(count)
    ]


@pytest.mark.parametrize(
    ("seed", "count", "extent", "id_count", "distance"),
    [
        (0, 200, 12.0, 30, 2.0),
        (1, 600, 30.0, 400, 2.0),
        (2, 400, 6.0, 8, 1.0),
        (3, 3000, 40.0, 900, 2.0),  # spans several processing blocks
        (4, 500, 1000.0, 500, 3.5),
        (5, 300, 8.0, 1000, 0.5),
    ],
)
def test_matches_scalar_reference(
    seed: int, count: int, extent: float, id_count: int, distance: float
) -> None:
    matches = _random_matches(np.random.default_rng(seed), count, extent=extent, id_count=id_count)
    expected = scalar_deduplicate_lifted_matches(matches, query_conflict_distance_px=distance)
    actual = deduplicate_lifted_matches(matches, query_conflict_distance_px=distance)
    assert actual == expected
    assert expected.matches  # the scenario exercises acceptance, not only rejection


def test_empty_input() -> None:
    assert deduplicate_lifted_matches([], query_conflict_distance_px=2.0) == (
        scalar_deduplicate_lifted_matches([], query_conflict_distance_px=2.0)
    )


def test_exact_duplicates_and_negative_coordinates() -> None:
    matches = [
        LiftedMatch((-0.5, -0.5), 1, "a", 0.9, 0.0),
        LiftedMatch((-0.5, -0.5), 1, "b", 0.9, 0.0),
        LiftedMatch((-0.5, -0.5), 2, "c", 0.8, 0.0),
        LiftedMatch((1.5, -0.5), 3, "a", 0.7, 0.0),  # exactly on the radius of the first
        LiftedMatch((1.5001, -0.5), 4, "a", 0.6, 0.0),
        LiftedMatch((-0.5, -0.5), None, "a", 0.5, 0.0),
    ]
    expected = scalar_deduplicate_lifted_matches(matches, query_conflict_distance_px=2.0)
    assert deduplicate_lifted_matches(matches, query_conflict_distance_px=2.0) == expected


@pytest.mark.parametrize("distance", [0.0, -1.0, math.inf, math.nan])
def test_invalid_distance_rejected(distance: float) -> None:
    with pytest.raises(ValueError):
        deduplicate_lifted_matches([], query_conflict_distance_px=distance)

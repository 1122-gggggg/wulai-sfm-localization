"""Differential tests for the vectorised `lift_reference_matches`.

The vendored scalar implementation that shipped in
`river-deploy-5060-20260908` is reproduced here as the oracle, unchanged
except for this repository's line length.  It is
the definition of correct behaviour for this function, so every test compares
the deployed implementation against it field by field rather than against
hand-written expectations: the vectorisation is only admissible if it is
indistinguishable from the scan it replaced.
"""

from __future__ import annotations

import math
from collections import defaultdict

# Source modules are supplied by the repository's pytest pythonpath.
import numpy as np
import pytest

from river_map_quality.official_edm_adapter import (
    LiftedMatch,
    ReferenceLiftResult,
    UnmappedPair,
    _validated_arrays,
    lift_reference_matches,
)


def scalar_lift_reference_matches(
    *,
    query_points: np.ndarray,
    reference_points: np.ndarray,
    observation_points: np.ndarray,
    observation_point3d_ids: np.ndarray,
    confidences: np.ndarray,
    maximum_distance_px: float,
    reference_name: str,
) -> ReferenceLiftResult:
    """The pre-vectorisation body, unchanged. Oracle only; never imported by runtime."""

    if not math.isfinite(maximum_distance_px) or maximum_distance_px <= 0:
        raise ValueError("maximum_distance_px must be finite and positive")
    if not reference_name:
        raise ValueError("reference_name is required")
    query, reference, observations, point_ids, scores = _validated_arrays(
        query_points,
        reference_points,
        observation_points,
        observation_point3d_ids,
        confidences,
    )
    valid = point_ids >= 0
    observations, point_ids = observations[valid], point_ids[valid]
    cell_size = float(maximum_distance_px)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(observations):
        grid[(math.floor(point[0] / cell_size), math.floor(point[1] / cell_size))].append(index)

    max_distance_sq = cell_size * cell_size
    lifted: list[LiftedMatch] = []
    unmapped_pairs: list[UnmappedPair] = []
    for query_point, reference_point, confidence in zip(query, reference, scores, strict=True):
        x_cell = math.floor(reference_point[0] / cell_size)
        y_cell = math.floor(reference_point[1] / cell_size)
        candidates = [
            index
            for x_offset in (-1, 0, 1)
            for y_offset in (-1, 0, 1)
            for index in grid.get((x_cell + x_offset, y_cell + y_offset), ())
        ]
        if not candidates:
            unmapped_pairs.append(
                UnmappedPair(
                    query_xy=(float(query_point[0]), float(query_point[1])),
                    reference_xy=(float(reference_point[0]), float(reference_point[1])),
                    reference_name=reference_name,
                    confidence=float(confidence),
                )
            )
            continue
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        deltas = observations[candidate_indices] - reference_point
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)
        within = distances_sq <= max_distance_sq
        if not within.any():
            unmapped_pairs.append(
                UnmappedPair(
                    query_xy=(float(query_point[0]), float(query_point[1])),
                    reference_xy=(float(reference_point[0]), float(reference_point[1])),
                    reference_name=reference_name,
                    confidence=float(confidence),
                )
            )
            continue
        candidate_indices, distances_sq = candidate_indices[within], distances_sq[within]
        selection = np.lexsort((point_ids[candidate_indices], distances_sq))[0]
        observation_index = int(candidate_indices[selection])
        lifted.append(
            LiftedMatch(
                query_xy=(float(query_point[0]), float(query_point[1])),
                point3d_id=int(point_ids[observation_index]),
                reference_name=reference_name,
                confidence=float(confidence),
                lift_distance_px=float(math.sqrt(distances_sq[selection])),
            )
        )
    return ReferenceLiftResult(
        matches=tuple(lifted),
        unmapped_match_count=len(unmapped_pairs),
        unmapped_pairs=tuple(unmapped_pairs),
    )


def assert_identical(**kwargs) -> ReferenceLiftResult:
    """Both implementations must agree on every field, bit for bit."""

    expected = scalar_lift_reference_matches(**kwargs)
    actual = lift_reference_matches(**kwargs)
    assert actual.unmapped_match_count == expected.unmapped_match_count
    assert actual.unmapped_pairs == expected.unmapped_pairs
    assert len(actual.matches) == len(expected.matches)
    for got, want in zip(actual.matches, expected.matches, strict=True):
        assert got.query_xy == want.query_xy
        assert got.point3d_id == want.point3d_id
        assert got.reference_name == want.reference_name
        assert got.confidence == want.confidence
        # Exact, not approximate: a different lift distance means a different
        # observation was selected, or the distance was computed differently.
        assert got.lift_distance_px == want.lift_distance_px
        assert got.xyz == want.xyz
    assert actual == expected
    return actual


def build_case(
    rng: np.random.Generator,
    *,
    n_matches: int,
    n_observations: int,
    spread: float,
    radius: float,
    invalid_ratio: float = 0.2,
    quantum: float | None = None,
) -> dict:
    """One randomised lift problem.

    ``quantum`` snaps both point sets onto a lattice, which is what manufactures
    the exact distance ties that the tie-break rule exists to resolve.
    """

    def points(count: int) -> np.ndarray:
        values = rng.uniform(-spread, spread, size=(count, 2))
        if quantum is not None:
            values = np.round(values / quantum) * quantum
        return values

    point_ids = rng.integers(0, 40, size=n_observations).astype(np.int64)
    point_ids[rng.random(n_observations) < invalid_ratio] = -1
    return {
        "query_points": points(n_matches),
        "reference_points": points(n_matches),
        "observation_points": points(n_observations),
        "observation_point3d_ids": point_ids,
        "confidences": rng.uniform(0.0, 1.0, size=n_matches),
        "maximum_distance_px": radius,
        "reference_name": "ref_0001.jpg",
    }


@pytest.mark.parametrize("seed", range(12))
def test_matches_the_scalar_scan_on_random_problems(seed: int) -> None:
    rng = np.random.default_rng(seed)
    case = build_case(rng, n_matches=180, n_observations=900, spread=60.0, radius=2.0)
    result = assert_identical(**case)
    # A vacuous comparison would pass; the fixture must exercise both outcomes.
    assert result.matches
    assert result.unmapped_match_count


@pytest.mark.parametrize("seed", range(8))
def test_matches_the_scalar_scan_when_distances_tie(seed: int) -> None:
    """A lattice makes equal distances common, so the tie-break is exercised."""

    rng = np.random.default_rng(1000 + seed)
    case = build_case(
        rng,
        n_matches=150,
        n_observations=700,
        spread=12.0,
        radius=2.0,
        quantum=0.5,
    )
    assert_identical(**case)


@pytest.mark.parametrize(
    ("n_matches", "n_observations", "spread", "radius"),
    [
        (1, 1, 3.0, 2.0),
        (40, 3, 50.0, 2.0),  # observations far sparser than matches
        (3, 400, 4.0, 2.0),  # every match buried in candidates
        (60, 200, 5.0, 0.25),  # radius well below the point spacing
        (60, 200, 5.0, 25.0),  # radius swallowing the whole scene
    ],
)
def test_matches_the_scalar_scan_across_densities(
    n_matches: int, n_observations: int, spread: float, radius: float
) -> None:
    rng = np.random.default_rng(7)
    assert_identical(
        **build_case(
            rng,
            n_matches=n_matches,
            n_observations=n_observations,
            spread=spread,
            radius=radius,
        )
    )


def test_distance_tie_is_broken_by_the_smaller_point3d_id() -> None:
    """Equidistant candidates resolve on the stable M0 Point3D ID, not on order."""

    case = {
        "query_points": np.array([[10.0, 10.0]]),
        "reference_points": np.array([[0.0, 0.0]]),
        # Both exactly 1.0 away; the higher ID is listed first on purpose.
        "observation_points": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "observation_point3d_ids": np.array([500, 200], dtype=np.int64),
        "confidences": np.array([0.9]),
        "maximum_distance_px": 2.0,
        "reference_name": "ref.jpg",
    }
    result = assert_identical(**case)
    assert result.matches[0].point3d_id == 200
    assert result.matches[0].lift_distance_px == 1.0


def test_radius_boundary_is_inclusive_and_exclusive_on_the_right_side() -> None:
    inside = {
        "query_points": np.array([[0.0, 0.0]]),
        "reference_points": np.array([[0.0, 0.0]]),
        "observation_points": np.array([[2.0, 0.0]]),
        "observation_point3d_ids": np.array([7], dtype=np.int64),
        "confidences": np.array([1.0]),
        "maximum_distance_px": 2.0,
        "reference_name": "ref.jpg",
    }
    result = assert_identical(**inside)
    assert result.matches[0].lift_distance_px == 2.0
    assert result.unmapped_match_count == 0

    outside = dict(inside, observation_points=np.array([[2.0000001, 0.0]]))
    result = assert_identical(**outside)
    assert result.matches == ()
    assert result.unmapped_match_count == 1


def test_negative_coordinates_use_floor_not_truncation() -> None:
    """Cell indices straddle zero; truncating instead of flooring merges cells."""

    rng = np.random.default_rng(11)
    case = build_case(rng, n_matches=120, n_observations=500, spread=6.0, radius=2.0)
    assert (case["reference_points"] < 0).any()
    assert (case["observation_points"] < 0).any()
    assert_identical(**case)


def test_observations_split_across_a_huge_coordinate_gap() -> None:
    """Dense cell ranks must not overflow when two clusters sit far apart."""

    near = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]])
    far = np.array([[1e15, 1e15], [1e15 + 1.0, 1e15]])
    case = {
        "query_points": np.array([[0.0, 0.0], [1e15, 1e15], [5e14, 0.0]]),
        "reference_points": np.array([[0.4, 0.4], [1e15 + 0.5, 1e15], [5e14, 0.0]]),
        "observation_points": np.concatenate([near, far]),
        "observation_point3d_ids": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "confidences": np.array([0.5, 0.6, 0.7]),
        "maximum_distance_px": 2.0,
        "reference_name": "ref.jpg",
    }
    result = assert_identical(**case)
    assert len(result.matches) == 2
    assert result.unmapped_match_count == 1


def test_reference_points_outside_the_observed_extent_are_unmapped() -> None:
    case = {
        "query_points": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "reference_points": np.array([[-900.0, -900.0], [900.0, 900.0]]),
        "observation_points": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "observation_point3d_ids": np.array([3, 4], dtype=np.int64),
        "confidences": np.array([0.5, 0.5]),
        "maximum_distance_px": 2.0,
        "reference_name": "ref.jpg",
    }
    result = assert_identical(**case)
    assert result.matches == ()
    assert result.unmapped_match_count == 2


def test_every_observation_without_a_point3d_id_is_dropped() -> None:
    case = {
        "query_points": np.array([[0.0, 0.0]]),
        "reference_points": np.array([[0.0, 0.0]]),
        "observation_points": np.array([[0.1, 0.1], [0.2, 0.2]]),
        "observation_point3d_ids": np.array([-1, -1], dtype=np.int64),
        "confidences": np.array([0.5]),
        "maximum_distance_px": 2.0,
        "reference_name": "ref.jpg",
    }
    result = assert_identical(**case)
    assert result.matches == ()
    assert result.unmapped_match_count == 1


def test_empty_match_set_and_empty_observation_set() -> None:
    empty_matches = {
        "query_points": np.zeros((0, 2)),
        "reference_points": np.zeros((0, 2)),
        "observation_points": np.array([[0.0, 0.0]]),
        "observation_point3d_ids": np.array([1], dtype=np.int64),
        "confidences": np.zeros(0),
        "maximum_distance_px": 2.0,
        "reference_name": "ref.jpg",
    }
    result = assert_identical(**empty_matches)
    assert result == ReferenceLiftResult(matches=(), unmapped_match_count=0)

    empty_observations = dict(
        empty_matches,
        query_points=np.array([[0.0, 0.0]]),
        reference_points=np.array([[0.0, 0.0]]),
        observation_points=np.zeros((0, 2)),
        observation_point3d_ids=np.zeros(0, dtype=np.int64),
        confidences=np.array([0.5]),
    )
    result = assert_identical(**empty_observations)
    assert result.unmapped_match_count == 1


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_radius_must_be_finite_and_positive(radius: float) -> None:
    with pytest.raises(ValueError, match="maximum_distance_px"):
        lift_reference_matches(
            query_points=np.zeros((1, 2)),
            reference_points=np.zeros((1, 2)),
            observation_points=np.zeros((1, 2)),
            observation_point3d_ids=np.array([1], dtype=np.int64),
            confidences=np.array([0.5]),
            maximum_distance_px=radius,
            reference_name="ref.jpg",
        )


def test_reference_name_is_required() -> None:
    with pytest.raises(ValueError, match="reference_name is required"):
        lift_reference_matches(
            query_points=np.zeros((1, 2)),
            reference_points=np.zeros((1, 2)),
            observation_points=np.zeros((1, 2)),
            observation_point3d_ids=np.array([1], dtype=np.int64),
            confidences=np.array([0.5]),
            maximum_distance_px=2.0,
            reference_name="",
        )


def test_non_finite_and_misaligned_inputs_are_still_rejected() -> None:
    base = {
        "query_points": np.zeros((2, 2)),
        "reference_points": np.zeros((2, 2)),
        "observation_points": np.zeros((2, 2)),
        "observation_point3d_ids": np.array([1, 2], dtype=np.int64),
        "confidences": np.array([0.5, 0.5]),
        "maximum_distance_px": 2.0,
        "reference_name": "ref.jpg",
    }
    with pytest.raises(ValueError, match="must be finite"):
        lift_reference_matches(**dict(base, reference_points=np.array([[0.0, 0.0], [np.nan, 0.0]])))
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        lift_reference_matches(**dict(base, reference_points=np.zeros((3, 2))))
    with pytest.raises(ValueError, match="confidences must have shape"):
        lift_reference_matches(**dict(base, confidences=np.array([0.5])))
    with pytest.raises(ValueError, match="COLMAP observations must be aligned"):
        lift_reference_matches(**dict(base, observation_point3d_ids=np.array([1], dtype=np.int64)))

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from site_alignment import (
    SiteAlignment,
    load_site_alignment,
    save_site_alignment,
    site_alignment_from_dict,
    solve_rigid_alignment,
    solve_similarity_alignment,
)


def _rotation() -> np.ndarray:
    yaw = math.radians(31.0)
    pitch = math.radians(-8.0)
    return np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ) @ np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )


def _points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 2.0, 0.5],
            [2.0, -1.0, 1.5],
        ]
    )


def test_umeyama_recovers_known_scale_rotation_and_translation() -> None:
    map_points = _points()
    rotation = _rotation()
    site_points = (2.4 * (rotation @ map_points.T)).T + [4.0, -3.0, 1.2]

    fit = solve_similarity_alignment(map_points, site_points)

    assert fit.transform.scale == pytest.approx(2.4)
    assert fit.transform.R == pytest.approx(rotation)
    assert fit.transform.t == pytest.approx([4.0, -3.0, 1.2])
    assert fit.quality.rmse_m < 1e-12
    assert np.linalg.det(fit.transform.R) == pytest.approx(1.0)


def test_rigid_alignment_never_estimates_scale() -> None:
    rng = np.random.default_rng(91)
    map_points = _points()
    rotation = _rotation()
    expected = (rotation @ map_points.T).T + [4.0, -3.0, 1.2]
    site_points = expected + rng.normal(scale=0.002, size=expected.shape)

    fit = solve_rigid_alignment(map_points, site_points)

    assert fit.transform.scale == 1.0
    assert fit.transform.R == pytest.approx(rotation, abs=0.005)
    assert fit.transform.t == pytest.approx([4.0, -3.0, 1.2], abs=0.005)
    assert 0.0 < fit.quality.rmse_m < 0.01


def test_rigid_alignment_roundtrip_keeps_scale_one(tmp_path) -> None:
    map_points = _points()
    site_points = (_rotation() @ map_points.T).T + [0.5, -0.8, 2.0]
    fit = solve_rigid_alignment(map_points, site_points)
    alignment = SiteAlignment.from_fit(
        map_frame_id="map-v1",
        site_frame_id="site-v1",
        fit=fit,
        approved=True,
    )

    loaded = load_site_alignment(save_site_alignment(alignment, tmp_path / "rigid.json"))

    assert loaded == alignment
    assert loaded.transform.scale == 1.0


def test_noisy_overdetermined_fit_reports_residual_quality() -> None:
    rng = np.random.default_rng(12)
    map_points = rng.normal(size=(20, 3))
    expected = (1.7 * (_rotation() @ map_points.T)).T + [0.5, -0.8, 2.0]
    site_points = expected + rng.normal(scale=0.004, size=expected.shape)

    fit = solve_similarity_alignment(map_points, site_points)

    assert fit.quality.rank == 3
    assert 0.0 < fit.quality.rmse_m < 0.02
    assert fit.quality.max_error_m < 0.03
    assert len(fit.quality.residuals_m) == 20


def test_point_vector_rotation_and_inverse_roundtrip() -> None:
    fit = solve_similarity_alignment(
        _points(),
        (1.3 * (_rotation() @ _points().T)).T + [2.0, 4.0, -1.0],
    )
    point = np.array([0.2, -0.4, 1.5])
    vector = np.array([1.0, 2.0, 3.0])

    assert fit.transform.site_to_map_point(
        fit.transform.map_to_site_point(point)
    ) == pytest.approx(point)
    assert fit.transform.site_to_map_vector(
        fit.transform.map_to_site_vector(vector)
    ) == pytest.approx(vector)
    composed = fit.transform.map_to_site_rotation(np.eye(3))
    assert fit.transform.site_to_map_rotation(composed) == pytest.approx(np.eye(3))


def test_reflection_input_never_returns_a_left_handed_rotation() -> None:
    reflected = _points().copy()
    reflected[:, 0] *= -1.0

    fit = solve_similarity_alignment(_points(), reflected)

    assert fit.quality.reflection_corrected
    assert np.linalg.det(fit.transform.R) == pytest.approx(1.0)
    assert fit.quality.rmse_m > 0.0


@pytest.mark.parametrize(
    ("map_points", "site_points", "message"),
    [
        ([[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0]], "at least three"),
        (
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            "degenerate",
        ),
        (
            [[0, 0, 0], [1, 0, 0], [0, 1, float("nan")]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "finite",
        ),
        (
            [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "unique",
        ),
    ],
)
def test_invalid_control_points_are_rejected(map_points, site_points, message) -> None:
    with pytest.raises(ValueError, match=message):
        solve_similarity_alignment(map_points, site_points)


def test_json_roundtrip_recomputes_and_binds_frame_ids(tmp_path) -> None:
    fit = solve_similarity_alignment(_points(), _points() + [1.0, 2.0, 3.0])
    alignment = SiteAlignment.from_fit(
        map_frame_id="map-v1",
        site_frame_id="site-enu-v1",
        fit=fit,
        approved=True,
    )
    path = save_site_alignment(alignment, tmp_path / "alignment.json")

    loaded = load_site_alignment(
        path,
        expected_map_frame_id="map-v1",
        expected_site_frame_id="site-enu-v1",
    )

    assert loaded == alignment
    with pytest.raises(ValueError, match="active map"):
        load_site_alignment(path, expected_map_frame_id="another-map")


def test_tampered_persisted_transform_is_rejected() -> None:
    fit = solve_similarity_alignment(_points(), _points() + [1.0, 2.0, 3.0])
    raw = SiteAlignment.from_fit(
        map_frame_id="map-v1",
        site_frame_id="site-v1",
        fit=fit,
    ).to_dict()
    raw["transform"]["scale"] = 9.0

    with pytest.raises(ValueError, match="does not match"):
        site_alignment_from_dict(raw)


def test_malformed_and_nonfinite_json_are_rejected(tmp_path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"schema": NaN}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_site_alignment(malformed)
    with pytest.raises(ValueError, match="exactly"):
        site_alignment_from_dict({"schema": "wrong"})


def test_alignment_models_are_immutable() -> None:
    fit = solve_similarity_alignment(_points(), _points())

    with pytest.raises(FrozenInstanceError):
        fit.transform.scale = 2.0


def test_serialized_json_contains_no_nonstandard_numbers() -> None:
    alignment = SiteAlignment.from_fit(
        map_frame_id="map-v1",
        site_frame_id="site-v1",
        fit=solve_similarity_alignment(_points(), _points()),
    )

    assert json.loads(alignment.to_json())["schema"] == "sfm-site-alignment/v1"

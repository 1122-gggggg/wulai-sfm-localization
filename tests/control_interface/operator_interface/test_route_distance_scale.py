from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from route_editor_model import (
    MAP_DISTANCE_SCALE_SCHEMA,
    MapDistanceScale,
    RouteDocument,
    load_map_distance_scale,
)
from site_profile import AssetDigests, CoordinateFrame, SiteProfile


@pytest.fixture
def scale_site(tmp_path):
    canonical_map = tmp_path / "canonical.ply"
    canonical_map.write_bytes(b"canonical sparse map asset")
    profile = SiteProfile(
        source=tmp_path / "site_profile.json",
        schema_version=2,
        site_id="yard",
        display_name="Yard",
        map_ply=canonical_map,
        route_json=None,
        localization_bundle=tmp_path / "localization_bundle.pkl",
        coordinate_frame=CoordinateFrame(
            id="yard-map-v1",
            convention="glomap",
            horizontal_axes=("x", "z"),
            up_axis="-y",
            handedness="right",
            units="map",
        ),
    )
    payload = {
        "schema": MAP_DISTANCE_SCALE_SCHEMA,
        "coordinate_frame_id": profile.coordinate_frame.id,
        "map_ply_sha256": hashlib.sha256(canonical_map.read_bytes()).hexdigest(),
        "m_per_unit": 2.5,
        "estimated": True,
        "source": "Two offline flights; candidate, not measured calibration",
        "uncertainty_relative": 0.27,
    }
    return profile, payload


def _write_scale(profile, payload):
    path = profile.map_ply.parent / "map_distance_scale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_arbitrary_three_dimensional_distance_is_scaled_without_mutation():
    scale = MapDistanceScale(2.5, estimated=True, source="offline candidate")
    first = [1.0, -2.0, 3.0]
    second = [3.0, 1.0, 9.0]
    original = copy.deepcopy([first, second])

    assert scale.distance_m(first, second) == pytest.approx(17.5)
    assert scale.distance_m(second, first) == pytest.approx(17.5)
    assert [first, second] == original
    assert scale.distance_m(first, first) == 0.0


def test_route_segments_and_total_are_an_open_three_dimensional_polyline():
    scale = MapDistanceScale(2.5, estimated=True, source="offline candidate")
    points = [[1, -2, 3], [3, 1, 9], [3, 1, 9], [0, 5, 21]]
    original = copy.deepcopy(points)

    segments = scale.route_segment_distances(points)

    assert segments == pytest.approx([17.5, 0.0, 32.5])
    assert sum(segments) == pytest.approx(50.0)
    assert len(segments) == len(points) - 1
    assert points == original


@pytest.mark.parametrize("points", [[], [[1, 2, 3]]])
def test_empty_or_single_waypoint_route_has_no_segments_or_total(points):
    scale = MapDistanceScale(2.5, estimated=True, source="offline candidate")
    segments = scale.route_segment_distances(points)
    assert segments == []
    assert sum(segments) == 0.0


@pytest.mark.parametrize(
    "value",
    [0, -1, float("nan"), float("inf"), -float("inf"), True, False, "2.5", None],
)
def test_invalid_scale_is_rejected(value):
    with pytest.raises(ValueError, match="m_per_unit"):
        MapDistanceScale(value, estimated=True, source="offline candidate")


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "0.2"])
def test_invalid_relative_uncertainty_is_rejected(value):
    with pytest.raises(ValueError, match="uncertainty_relative"):
        MapDistanceScale(
            2.5, estimated=True, source="offline candidate", uncertainty_relative=value,
        )


@pytest.mark.parametrize("value", [None, "true", 0, 1])
def test_estimated_status_must_be_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="estimated"):
        MapDistanceScale(2.5, estimated=value, source="offline candidate")


@pytest.mark.parametrize("value", [None, "", "  ", 123])
def test_source_cannot_be_missing_or_empty(value):
    with pytest.raises(ValueError, match="source"):
        MapDistanceScale(2.5, estimated=True, source=value)


@pytest.mark.parametrize(
    "point",
    [[1, 2], [1, 2, 3, 4], [1, float("nan"), 3], [1, 2, float("inf")]],
)
def test_distance_rejects_non_three_dimensional_or_nonfinite_points(point):
    scale = MapDistanceScale(2.5, estimated=True, source="offline candidate")
    with pytest.raises(ValueError):
        scale.distance_m(point, [0, 0, 0])


@pytest.mark.parametrize("estimated", [True, False])
def test_loader_preserves_declared_status_source_and_uncertainty(scale_site, estimated):
    profile, payload = scale_site
    payload["estimated"] = estimated
    _write_scale(profile, payload)

    scale = load_map_distance_scale(profile)

    assert scale is not None
    assert scale.m_per_unit == 2.5
    assert scale.estimated is estimated
    assert scale.source == payload["source"]
    assert scale.uncertainty_relative == 0.27
    assert scale.distance_m([1, -2, 3], [3, 1, 9]) == pytest.approx(17.5)


def test_missing_optional_uncertainty_is_unknown_not_zero(scale_site):
    profile, payload = scale_site
    payload.pop("uncertainty_relative")
    _write_scale(profile, payload)
    scale = load_map_distance_scale(profile)
    assert scale is not None
    assert scale.uncertainty_relative is None


def test_missing_scale_does_not_assume_one_meter_per_map_unit(scale_site):
    profile, _ = scale_site
    assert load_map_distance_scale(None) is None
    assert load_map_distance_scale(profile) is None


@pytest.mark.parametrize(
    "key",
    ["schema", "coordinate_frame_id", "map_ply_sha256", "m_per_unit", "estimated", "source"],
)
def test_missing_required_metadata_makes_meter_conversion_unavailable(scale_site, key):
    profile, payload = scale_site
    payload.pop(key)
    _write_scale(profile, payload)
    assert load_map_distance_scale(profile) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "unsupported"),
        ("coordinate_frame_id", "another-map-v1"),
        ("map_ply_sha256", "0" * 64),
        ("map_ply_sha256", "not-a-hash"),
        ("map_ply_sha256", "g" * 64),
        ("m_per_unit", 0),
        ("m_per_unit", float("nan")),
        ("m_per_unit", float("inf")),
        ("m_per_unit", True),
        ("estimated", "true"),
        ("source", ""),
        ("uncertainty_relative", -0.1),
    ],
)
def test_invalid_or_mismatched_metadata_never_applies_a_scale(scale_site, key, value):
    profile, payload = scale_site
    payload[key] = value
    _write_scale(profile, payload)
    assert load_map_distance_scale(profile) is None


@pytest.mark.parametrize("content", [b"{", b"[]", b"null", b"\xff"])
def test_damaged_sidecar_makes_meter_conversion_unavailable(scale_site, content):
    profile, _ = scale_site
    (profile.map_ply.parent / "map_distance_scale.json").write_bytes(content)
    assert load_map_distance_scale(profile) is None


def test_missing_coordinate_frame_cannot_bind_a_scale(scale_site):
    profile, payload = scale_site
    _write_scale(profile, payload)
    assert load_map_distance_scale(replace(profile, coordinate_frame=None)) is None


def test_profile_asset_digest_binds_scale_without_reopening_the_large_map(scale_site):
    profile, payload = scale_site
    _write_scale(profile, payload)
    profile = replace(
        profile, asset_sha256=AssetDigests(map_ply=payload["map_ply_sha256"]),
    )
    profile.map_ply.unlink()

    assert load_map_distance_scale(profile) is not None
    wrong_profile = replace(profile, asset_sha256=AssetDigests(map_ply="0" * 64))
    assert load_map_distance_scale(wrong_profile) is None


def test_fallback_uses_canonical_map_not_dense_display_substitute(scale_site):
    profile, payload = scale_site
    dense_map = profile.map_ply.parent / "scene_dense.ply"
    dense_map.write_bytes(b"dense display-only asset; different digest")
    _write_scale(profile, payload)
    assert load_map_distance_scale(profile) is not None

    payload["map_ply_sha256"] = hashlib.sha256(dense_map.read_bytes()).hexdigest()
    _write_scale(profile, payload)
    assert load_map_distance_scale(profile) is None


def test_reloads_small_sidecar_and_refuses_a_replaced_canonical_map(scale_site):
    profile, payload = scale_site
    _write_scale(profile, payload)
    assert load_map_distance_scale(profile).m_per_unit == 2.5

    payload["m_per_unit"] = 3.0
    _write_scale(profile, payload)
    assert load_map_distance_scale(profile).m_per_unit == 3.0

    profile.map_ply.write_bytes(b"a replacement canonical map with a different size")
    assert load_map_distance_scale(profile) is None


def test_meter_measurement_does_not_rewrite_waypoints_or_route_file(tmp_path):
    document = RouteDocument(
        site_id="yard",
        coordinate_frame_id="yard-map-v1",
        points=[[1, -2, 3], [3, 1, 9], [0, 5, 21]],
    )
    saved_route = document.save(tmp_path / "route.json")
    original_file = saved_route.read_bytes()
    original_points = copy.deepcopy(document.points)
    scale = MapDistanceScale(2.5, estimated=True, source="offline candidate")

    assert sum(scale.route_segment_distances(document.points)) == pytest.approx(50.0)
    assert document.points == original_points
    assert saved_route.read_bytes() == original_file
    assert json.loads(original_file)["units"] == "map"
    assert json.loads(original_file)["waypoints"] == original_points


def test_route_of_repeated_waypoints_has_zero_length():
    scale = MapDistanceScale(2.5, estimated=True, source="offline candidate")
    points = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    segments = scale.route_segment_distances(points)
    assert segments == [0.0, 0.0]
    assert sum(segments) == 0.0


def test_missing_canonical_map_without_an_asset_digest_cannot_bind_scale(scale_site):
    profile, payload = scale_site
    _write_scale(profile, payload)
    profile.map_ply.unlink()
    assert load_map_distance_scale(profile) is None

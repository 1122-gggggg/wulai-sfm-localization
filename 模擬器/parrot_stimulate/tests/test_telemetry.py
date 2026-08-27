import pytest

from anafi_pcmd_sim.models import TruePosition
from anafi_pcmd_sim.telemetry import TrueTelemetryCollector, TrueTelemetryParser


def test_parser_emits_a_true_world_position_sample() -> None:
    parser = TrueTelemetryParser()

    assert parser.feed("omniscient_anafi.timestamp: 4.098000") is None
    assert parser.feed("omniscient_anafi.worldPosition.x: 1.250000") is None
    assert parser.feed("omniscient_anafi.worldPosition.y: -0.250000") is None

    sample = parser.feed("omniscient_anafi.worldPosition.z: 0.750000")

    assert sample is not None
    assert sample.timestamp_s == 4.098
    assert (sample.x_m, sample.y_m, sample.z_m) == (1.25, -0.25, 0.75)


def test_parser_ignores_non_true_data() -> None:
    parser = TrueTelemetryParser()

    assert parser.feed("battery.percent: 80") is None


def test_parser_ignores_other_sphinx_objects_with_world_positions() -> None:
    parser = TrueTelemetryParser()

    parser.feed("omniscient___UE4__magic_tile_anafi.timestamp: 4.0")
    parser.feed("omniscient___UE4__magic_tile_anafi.worldPosition.x: 99.0")
    parser.feed("omniscient___UE4__magic_tile_anafi.worldPosition.y: 99.0")
    parser.feed("omniscient___UE4__magic_tile_anafi.worldPosition.z: 99.0")
    parser.feed("vertical_camera_anafi.timestamp: 4.5")
    parser.feed("vertical_camera_anafi.worldPosition.x: 9.0")
    parser.feed("vertical_camera_anafi.worldPosition.y: 9.0")
    parser.feed("vertical_camera_anafi.worldPosition.z: 0.2")

    parser.feed("omniscient_anafi.timestamp: 5.0")
    parser.feed("omniscient_anafi.worldPosition.x: 1.0")
    parser.feed("omniscient_anafi.worldPosition.y: 2.0")
    sample = parser.feed("omniscient_anafi.worldPosition.z: 3.0")

    assert sample == TruePosition(timestamp_s=5.0, x_m=1.0, y_m=2.0, z_m=3.0)


def test_parser_accepts_scientific_notation() -> None:
    parser = TrueTelemetryParser()

    parser.feed("omniscient_anafi.timestamp: 1.2e1")
    parser.feed("omniscient_anafi.worldPosition.x: 1e-2")
    parser.feed("omniscient_anafi.worldPosition.y: 2e-2")
    sample = parser.feed("omniscient_anafi.worldPosition.z: 3e-2")

    assert sample is not None
    assert sample.timestamp_s == 12.0
    assert (sample.x_m, sample.y_m, sample.z_m) == (0.01, 0.02, 0.03)


def test_parser_discards_an_implausible_world_coordinate() -> None:
    parser = TrueTelemetryParser()

    parser.feed("omniscient_anafi.timestamp: 4.0")
    parser.feed("omniscient_anafi.worldPosition.x: 1.0")
    parser.feed("omniscient_anafi.worldPosition.y: 2.0")
    assert parser.feed("omniscient_anafi.worldPosition.z: 99999.8") is None

    parser.feed("omniscient_anafi.timestamp: 5.0")
    parser.feed("omniscient_anafi.worldPosition.x: 3.0")
    parser.feed("omniscient_anafi.worldPosition.y: 4.0")
    sample = parser.feed("omniscient_anafi.worldPosition.z: 5.0")

    assert sample is not None
    assert (sample.x_m, sample.y_m, sample.z_m) == (3.0, 4.0, 5.0)


def test_collector_selects_a_delayed_sample_without_using_future_pose() -> None:
    collector = TrueTelemetryCollector()
    assert collector._command == (
        "tlm-data-logger",
        "-r",
        "50",
        "inet:127.0.0.1:9060",
    )
    collector._samples = [
        TruePosition(timestamp_s=1.0, x_m=1, y_m=0, z_m=1),
        TruePosition(timestamp_s=1.2, x_m=2, y_m=0, z_m=1),
        TruePosition(timestamp_s=1.4, x_m=3, y_m=0, z_m=1),
    ]

    selected = collector.sample_at_or_before(1.25)

    assert selected is not None
    assert selected.timestamp_s == 1.2
    assert collector.sample_at_or_before(0.5) is None


def test_collector_interpolates_a_delayed_pose_without_extrapolating() -> None:
    collector = TrueTelemetryCollector()
    collector._samples = [
        TruePosition(timestamp_s=1.0, x_m=1, y_m=2, z_m=3),
        TruePosition(timestamp_s=1.2, x_m=2, y_m=4, z_m=5),
        TruePosition(timestamp_s=1.4, x_m=3, y_m=6, z_m=7),
    ]

    selected = collector.interpolated_sample_at(1.25)

    assert selected is not None
    assert selected.timestamp_s == 1.25
    assert (selected.x_m, selected.y_m, selected.z_m) == pytest.approx((2.25, 4.5, 5.5))
    assert collector.interpolated_sample_at(0.5) is None
    assert collector.interpolated_sample_at(1.5) is None

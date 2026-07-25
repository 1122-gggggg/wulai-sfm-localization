import pytest

from scale_utils import (NO_SCALE_WARNING, ScaleContext, compute_map_units_per_meter,
                         compute_meters_per_map_unit, provisional_map_unit_thresholds)


def test_calibration_formulas():
    # 12 m measured between two points that are 30 map units apart
    assert compute_meters_per_map_unit(12.0, 30.0) == pytest.approx(0.4)
    assert compute_map_units_per_meter(12.0, 30.0) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        compute_meters_per_map_unit(0.0, 30.0)
    with pytest.raises(ValueError):
        compute_map_units_per_meter(12.0, -1.0)


def test_threshold_conversion_example():
    # spec example: desired 1.0 m deviation -> map units via map_units_per_meter
    sc = ScaleContext(map_units_per_meter=2.4)
    assert sc.to_map_units(1.0) == pytest.approx(2.4)
    assert sc.to_meters(2.4) == pytest.approx(1.0)
    assert sc.meters_per_map_unit == pytest.approx(1.0 / 2.4)


def test_reciprocal_factors_validated():
    ScaleContext(map_units_per_meter=2.0, meters_per_map_unit=0.5)  # consistent
    with pytest.raises(ValueError):
        ScaleContext(map_units_per_meter=2.0, meters_per_map_unit=0.6)
    with pytest.raises(ValueError):
        ScaleContext(map_units_per_meter=-1.0)


def test_missing_scale_factor_is_never_invented():
    sc = ScaleContext()
    assert not sc.has_scale
    with pytest.raises(ValueError):
        sc.to_map_units(1.0)
    with pytest.raises(ValueError):
        sc.to_meters(1.0)
    d = sc.dual(0.5)
    assert d["meters"] == 0.5 and d["map_units"] is None
    assert "calibration" in d["note"]
    header = "\n".join(sc.report_header())
    assert NO_SCALE_WARNING in header
    assert "arbitrary units" in header
    assert "not meters" in header               # height is map units, not meters


def test_dual_reporting_with_scale():
    sc = ScaleContext(meters_per_map_unit=0.5)
    d = sc.dual(3.0)                            # 3 m == 6 map units
    assert d["map_units"] == pytest.approx(6.0)
    header = "\n".join(sc.report_header())
    assert NO_SCALE_WARNING not in header
    assert "meters_per_map_unit=0.5" in header


def test_provisional_thresholds_require_calibration():
    th = provisional_map_unit_thresholds(route_len_u=30.0, n_segments=6)
    assert th["requires_calibration"] is True
    assert th["units"] == "arbitrary map units"
    assert th["arrival_radius_u"] == pytest.approx(0.15 * 5.0)
    assert th["horizontal_hard_abort_threshold_u"] > th["horizontal_correction_threshold_u"] \
        > th["horizontal_deadband_u"]

#!/usr/bin/env python3
"""Scale handling: Sphinx meters vs real monocular SfM map units.

The real GLOMAP/SfM map has NO reliable metric scale on ANY axis (including
height). Sphinx simulation is metric. This module keeps the two worlds apart:

    Sphinx validation scale: meters
    Real monocular SfM map scale: arbitrary units
    Real altitude scale: arbitrary map units, not meters

A ScaleContext converts between the two ONLY when the operator supplies a
calibrated factor. Without a factor, every real-map threshold stays in
arbitrary map units and is flagged as requiring calibration. No factor is ever
invented or defaulted.
"""
from __future__ import annotations

from dataclasses import dataclass

NO_SCALE_WARNING = (
    "WARNING: Sphinx convergence thresholds are in meters. Real monocular SfM "
    "map thresholds are arbitrary map units and must be calibrated before "
    "physical flight."
)

SCALE_STATEMENT = (
    "Sphinx validation scale: meters\n"
    "Real monocular SfM map scale: arbitrary units\n"
    "Real altitude scale: arbitrary map units, not meters"
)


def compute_meters_per_map_unit(real_distance_m: float, map_distance_units: float) -> float:
    """Calibration: measure a known real distance between two points visible on
    the route, find the same two points in the SfM/GLOMAP map, then

        meters_per_map_unit = real_distance_m / map_distance_units
    """
    if real_distance_m <= 0.0 or map_distance_units <= 0.0:
        raise ValueError("calibration distances must be positive")
    return real_distance_m / map_distance_units


def compute_map_units_per_meter(real_distance_m: float, map_distance_units: float) -> float:
    """Inverse calibration: map_units_per_meter = map_distance_units / real_distance_m."""
    if real_distance_m <= 0.0 or map_distance_units <= 0.0:
        raise ValueError("calibration distances must be positive")
    return map_distance_units / real_distance_m


@dataclass
class ScaleContext:
    """Optional meters <-> map-units conversion. Both factors accepted; if both
    are given they must agree (reciprocals)."""
    map_units_per_meter: float | None = None
    meters_per_map_unit: float | None = None

    def __post_init__(self):
        m2u, u2m = self.map_units_per_meter, self.meters_per_map_unit
        if m2u is not None and m2u <= 0.0:
            raise ValueError("map_units_per_meter must be positive")
        if u2m is not None and u2m <= 0.0:
            raise ValueError("meters_per_map_unit must be positive")
        if m2u is not None and u2m is not None:
            if abs(m2u * u2m - 1.0) > 1e-6:
                raise ValueError(
                    f"map_units_per_meter={m2u} and meters_per_map_unit={u2m} "
                    "are not reciprocals")
        elif m2u is not None:
            self.meters_per_map_unit = 1.0 / m2u
        elif u2m is not None:
            self.map_units_per_meter = 1.0 / u2m

    @property
    def has_scale(self) -> bool:
        return self.map_units_per_meter is not None

    def to_map_units(self, meters: float) -> float:
        if not self.has_scale:
            raise ValueError("no scale factor supplied; real map thresholds "
                             "remain arbitrary map units (calibration required)")
        return meters * self.map_units_per_meter

    def to_meters(self, map_units: float) -> float:
        if not self.has_scale:
            raise ValueError("no scale factor supplied; cannot express map "
                             "units in meters (calibration required)")
        return map_units * self.meters_per_map_unit

    def dual(self, value_m: float) -> dict:
        """Report a Sphinx-meter quantity in both units (map units only if a
        calibrated factor exists)."""
        out = {"meters": float(value_m)}
        if self.has_scale:
            out["map_units"] = float(value_m) * self.map_units_per_meter
        else:
            out["map_units"] = None
            out["note"] = "no scale factor: map-unit value requires calibration"
        return out

    def report_header(self) -> list[str]:
        lines = list(SCALE_STATEMENT.splitlines())
        if self.has_scale:
            lines.append(f"scale factor: meters_per_map_unit={self.meters_per_map_unit:.6g}, "
                         f"map_units_per_meter={self.map_units_per_meter:.6g}")
        else:
            lines.append("scale factor: meters_per_map_unit or map_units_per_meter, "
                         "unknown unless calibrated")
            lines.append(NO_SCALE_WARNING)
        return lines


def provisional_map_unit_thresholds(route_len_u: float, n_segments: int) -> dict:
    """When NO scale factor exists, estimate provisional real-map thresholds
    from the route's own geometry (fractions of segment length). These are NOT
    meters and NOT flight-ready: they only give an order of magnitude that MUST
    be calibrated before physical flight."""
    seg = route_len_u / max(1, n_segments)
    return {
        "units": "arbitrary map units",
        "requires_calibration": True,
        "arrival_radius_u": 0.15 * seg,
        "arrival_vertical_radius_u": 0.10 * seg,
        "horizontal_deadband_u": 0.05 * seg,
        "horizontal_correction_threshold_u": 0.20 * seg,
        "horizontal_hard_abort_threshold_u": 1.00 * seg,
        "note": ("provisional fractions of mean segment length; calibrate with "
                 "compute_map_units_per_meter() before physical flight"),
    }

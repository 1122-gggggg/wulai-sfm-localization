"""Online GNSS-to-map calibration from visually confirmed anchors."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from pose_guided.types import FusedOdometrySample

_EARTH_RADIUS_M = 6_378_137.0
_MAX_GNSS_AGE_S = 2.5
_MAX_HORIZONTAL_ACCURACY_M = 5.0
_MIN_SAMPLES = 6
_MIN_AXIS_SPREAD_M = 1.0
_MAX_BASIS_ANISOTROPY = 1.35


@dataclass(frozen=True)
class GnssMapEstimate:
    center: np.ndarray
    covariance: np.ndarray
    confidence: float


@dataclass(frozen=True)
class _CalibrationObservation:
    gps_timestamp: float
    east_north_m: np.ndarray
    map_center: np.ndarray
    horizontal_accuracy_m: float


class GnssMapPrior:
    """Learn a local 2-D GNSS to 3-D map similarity from safe visual fixes.

    No map scale or axis convention is assumed. The prior stays disabled until
    independently timestamped GNSS and visually confirmed centers span two
    horizontal directions and fit an approximately isotropic similarity.
    """

    def __init__(self, *, max_samples: int = 96) -> None:
        self._observations: deque[_CalibrationObservation] = deque(maxlen=max_samples)
        self._origin_latitude: float | None = None
        self._origin_longitude: float | None = None
        self._linear: np.ndarray | None = None
        self._offset: np.ndarray | None = None
        self._map_units_per_meter: float | None = None
        self._fit_rms_m: float | None = None
        self._last_gps_timestamp: float | None = None

    @property
    def ready(self) -> bool:
        return self._linear is not None and self._offset is not None

    def clear(self) -> None:
        self._observations.clear()
        self._origin_latitude = None
        self._origin_longitude = None
        self._linear = None
        self._offset = None
        self._map_units_per_meter = None
        self._fit_rms_m = None
        self._last_gps_timestamp = None

    def observe_visual(self, sample: FusedOdometrySample | None, center: np.ndarray) -> bool:
        validated = self._validated_sample(sample)
        map_center = np.asarray(center, dtype=float)
        if validated is None or map_center.shape != (3,) or not np.isfinite(map_center).all():
            return False
        gps_timestamp, latitude, longitude, horizontal_accuracy = validated
        if self._last_gps_timestamp is not None and gps_timestamp <= self._last_gps_timestamp:
            return False
        if self._origin_latitude is None:
            self._origin_latitude = latitude
            self._origin_longitude = longitude
        east_north = self._east_north_m(latitude, longitude)
        self._observations.append(
            _CalibrationObservation(
                gps_timestamp=gps_timestamp,
                east_north_m=east_north,
                map_center=map_center.copy(),
                horizontal_accuracy_m=horizontal_accuracy,
            )
        )
        self._last_gps_timestamp = gps_timestamp
        self._fit()
        return True

    def estimate(self, sample: FusedOdometrySample | None) -> GnssMapEstimate | None:
        validated = self._validated_sample(sample)
        if validated is None or not self.ready:
            return None
        _gps_timestamp, latitude, longitude, horizontal_accuracy = validated
        assert self._linear is not None
        assert self._offset is not None
        assert self._map_units_per_meter is not None
        center = self._linear @ self._east_north_m(latitude, longitude) + self._offset
        if not np.isfinite(center).all():
            return None
        sigma_m = max(horizontal_accuracy, self._fit_rms_m or 0.0, 0.5)
        sigma_map = self._map_units_per_meter * sigma_m
        confidence = 1.0 / (1.0 + sigma_m)
        return GnssMapEstimate(
            center=center,
            covariance=np.eye(3, dtype=float) * sigma_map * sigma_map,
            confidence=confidence,
        )

    def _validated_sample(
        self, sample: FusedOdometrySample | None
    ) -> tuple[float, float, float, float] | None:
        if sample is None or not sample.has_gnss:
            return None
        assert sample.geodetic_timestamp is not None
        assert sample.geodetic_lla is not None
        assert sample.geodetic_accuracy_m is not None
        age = float(sample.timestamp) - float(sample.geodetic_timestamp)
        if not math.isfinite(age) or age < -1e-6 or age > _MAX_GNSS_AGE_S:
            return None
        latitude, longitude, _altitude = sample.geodetic_lla
        lat_accuracy, lon_accuracy, _alt_accuracy = sample.geodetic_accuracy_m
        horizontal_accuracy = max(float(lat_accuracy), float(lon_accuracy))
        if (
            not math.isfinite(horizontal_accuracy)
            or horizontal_accuracy > _MAX_HORIZONTAL_ACCURACY_M
        ):
            return None
        return (
            float(sample.geodetic_timestamp),
            float(latitude),
            float(longitude),
            horizontal_accuracy,
        )

    def _east_north_m(self, latitude: float, longitude: float) -> np.ndarray:
        assert self._origin_latitude is not None
        assert self._origin_longitude is not None
        latitude_rad = math.radians(latitude)
        origin_latitude_rad = math.radians(self._origin_latitude)
        north = math.radians(latitude - self._origin_latitude) * _EARTH_RADIUS_M
        east = (
            math.radians(longitude - self._origin_longitude)
            * _EARTH_RADIUS_M
            * math.cos(0.5 * (latitude_rad + origin_latitude_rad))
        )
        return np.asarray([east, north], dtype=float)

    def _fit(self) -> None:
        if len(self._observations) < _MIN_SAMPLES:
            return
        east_north = np.stack([row.east_north_m for row in self._observations])
        centered = east_north - np.mean(east_north, axis=0)
        covariance = centered.T @ centered / len(centered)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] < _MIN_AXIS_SPREAD_M * _MIN_AXIS_SPREAD_M:
            return
        centers = np.stack([row.map_center for row in self._observations])
        accuracies = np.asarray([max(row.horizontal_accuracy_m, 0.5) for row in self._observations])
        design = np.column_stack([east_north, np.ones(len(east_north))])
        weighted_design = design / accuracies[:, None]
        weighted_centers = centers / accuracies[:, None]
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(
            weighted_design,
            weighted_centers,
            rcond=None,
        )
        if rank != 3:
            return
        linear = coefficients[:2].T
        basis_scales = np.linalg.svd(linear, compute_uv=False)
        if basis_scales[-1] <= 1e-9 or basis_scales[0] / basis_scales[-1] > _MAX_BASIS_ANISOTROPY:
            return
        scale = float(np.mean(basis_scales))
        predictions = design @ coefficients
        residual_m = np.linalg.norm(predictions - centers, axis=1) / scale
        rms_m = float(np.sqrt(np.mean(residual_m * residual_m)))
        median_accuracy = float(np.median(accuracies))
        if rms_m > max(1.5, 1.5 * median_accuracy):
            return
        self._linear = linear
        self._offset = coefficients[2]
        self._map_units_per_meter = scale
        self._fit_rms_m = rms_m

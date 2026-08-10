#!/usr/bin/env python3
"""Shared, immutable route domain model.

The operator controller and the older ``load_path`` module used to parse the
same JSON with slightly different rules.  This module owns the route contract:
JSON decoding, finite waypoint validation, route metadata, and conversion from
the authoring frame to controller GLOMAP coordinates.  It intentionally has no
flight or UI dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NoReturn, Protocol

import numpy as np


ROUTE_SCHEMA = "sfm-flight-route/v1"


def reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def finite_vec3(value: object, label: str) -> np.ndarray:
    """Return a finite numeric 3-vector or raise a contract error."""
    try:
        out: np.ndarray = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric 3-vector") from exc
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(f"{label} must be a finite numeric 3-vector, got shape={out.shape}")
    return out


def _validated_waypoints(waypoints: object) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(waypoints, (list, tuple)) or len(waypoints) < 2:
        count = len(waypoints) if isinstance(waypoints, (list, tuple)) else 0
        raise ValueError(f"need >=2 waypoints, got {count}")
    checked = [finite_vec3(point, f"waypoint[{index}]") for index, point in enumerate(waypoints)]
    for index, (start, end) in enumerate(zip(checked[:-1], checked[1:])):
        if float(np.linalg.norm(end - start)) <= 1e-9:
            raise ValueError(f"route segment {index}->{index + 1} has zero length")
    return tuple((float(point[0]), float(point[1]), float(point[2])) for point in checked)


class MapFrameLike(Protocol):
    """Minimal map-frame surface needed by route conversion.

    ``real_path_follow_controller.MapFrame`` satisfies this protocol.  Keeping
    this structural avoids a route-domain/controller import cycle.
    """

    @property
    def east(self) -> np.ndarray: ...

    @property
    def north(self) -> np.ndarray: ...

    @property
    def up(self) -> np.ndarray: ...

    @property
    def source(self) -> str: ...


@dataclass(frozen=True)
class _LegacyMapFrame:
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray
    source: str = "legacy_assumption"


LEGACY_MAP_FRAME = _LegacyMapFrame(
    east=np.array([1.0, 0.0, 0.0]),
    north=np.array([0.0, 0.0, 1.0]),
    up=np.array([0.0, -1.0, 0.0]),
)


def aligned_to_glomap(
    point: Iterable[float],
    frame: MapFrameLike = LEGACY_MAP_FRAME,
) -> np.ndarray:
    """Convert an aligned (east, north, up) point to raw GLOMAP."""
    value = finite_vec3(point, "aligned coordinate")
    result: np.ndarray = (
        float(value[0]) * np.asarray(frame.east, dtype=float)
        + float(value[1]) * np.asarray(frame.north, dtype=float)
        + float(value[2]) * np.asarray(frame.up, dtype=float)
    )
    return result


def _read_json_bytes(payload: bytes, source: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"), parse_constant=reject_json_constant)
    except UnicodeDecodeError as exc:
        raise ValueError(f"route is not valid UTF-8 JSON: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"route is not valid JSON: {source}") from exc


def _strict_bool(value: object, key: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"route {key} must be a boolean, got {value!r}")
    return value


def _optional_text(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"route {key} must be a non-empty string")
    return value


def _route_core_fields(
    data: dict[str, Any],
    *,
    require_map_units: bool,
) -> tuple[tuple[tuple[float, float, float], ...], str, str, bool]:
    source_points = _validated_waypoints(data["waypoints"])
    frame = data.get("frame", "aligned")
    if not isinstance(frame, str) or frame not in {"aligned", "glomap"}:
        raise ValueError("route frame must be 'aligned' or 'glomap'")
    units = data.get("units", "map")
    if not isinstance(units, str) or not units:
        raise ValueError("route units must be a non-empty string")
    if require_map_units and units != "map":
        raise ValueError("route units must be 'map'")
    return source_points, frame, units, _strict_bool(data.get("closed"), "closed")


def _validate_flight_contract(
    data: dict[str, Any],
    *,
    expected_site_id: str | None,
    expected_coordinate_frame_id: str | None,
    frame: str,
    closed: bool,
) -> None:
    if "frame" not in data:
        raise ValueError("flight route frame must be declared")
    if "closed" not in data:
        raise ValueError("flight route closed must be declared False")
    expected = {
        "schema": ROUTE_SCHEMA,
        "site_id": expected_site_id,
        "coordinate_frame_id": expected_coordinate_frame_id,
        "units": "map",
        "purpose": "flight",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"flight route {key} must be {value!r}, got {data.get(key)!r}")
    if frame not in {"aligned", "glomap"}:
        raise ValueError("flight route frame must be 'aligned' or 'glomap'")
    if closed is not False:
        raise ValueError(f"flight route closed must be False, got {data.get('closed')!r}")


def _controller_coordinates(
    source_points: tuple[tuple[float, float, float], ...],
    *,
    frame: str,
    align_source: str | None,
    map_frame: MapFrameLike,
) -> tuple[tuple[tuple[float, float, float], ...], str | None]:
    if frame != "aligned":
        return source_points, align_source

    site_is_measured = getattr(map_frame, "source", "") != "legacy_assumption"
    if align_source is None:
        if site_is_measured:
            raise ValueError(
                "route uses frame='aligned' but declares no align_source, and "
                "this site has a measured gravity alignment; add "
                "align_source='legacy' or 'measured' (or export frame='glomap')"
            )
        align_source = "legacy"
    if align_source == "legacy":
        authoring_frame: MapFrameLike = LEGACY_MAP_FRAME
    elif align_source == "measured":
        if not site_is_measured:
            raise ValueError(
                "route declares align_source='measured' but this site has no "
                "measured gravity alignment to invert it with"
            )
        authoring_frame = map_frame
    else:
        raise ValueError(f"unknown route align_source: {align_source!r}")
    controller_points = tuple(
        (float(value[0]), float(value[1]), float(value[2]))
        for value in (aligned_to_glomap(point, authoring_frame) for point in source_points)
    )
    return controller_points, align_source


def _arrival_radius(data: dict[str, Any]) -> float | None:
    raw_radius = data.get("arrive_radius_map_units")
    if raw_radius is None:
        return None
    if isinstance(raw_radius, bool) or not isinstance(raw_radius, (int, float)):
        raise ValueError("route arrive_radius_map_units must be a number")
    arrive_radius = float(raw_radius)
    if not math.isfinite(arrive_radius) or arrive_radius <= 0.0:
        raise ValueError("route arrive_radius_map_units must be finite and > 0")
    return arrive_radius


@dataclass(frozen=True)
class RouteDocument:
    """One validated route and its canonical controller coordinates.

    ``waypoints`` is the immutable source/authoring representation.  The
    controller representation is kept separately so legacy path consumers can
    still receive aligned coordinates while the flight controller receives
    GLOMAP coordinates.
    """

    waypoints: tuple[tuple[float, float, float], ...]
    controller_points: tuple[tuple[float, float, float], ...]
    frame: str
    units: str
    closed: bool
    schema: str | None = None
    site_id: str | None = None
    coordinate_frame_id: str | None = None
    purpose: str | None = None
    align_source: str | None = None
    arrive_radius_map_units: float | None = None

    @classmethod
    def from_data(
        cls,
        data: object,
        *,
        expected_site_id: str | None = None,
        expected_coordinate_frame_id: str | None = None,
        require_flight_contract: bool = False,
        require_map_units: bool = False,
        map_frame: MapFrameLike = LEGACY_MAP_FRAME,
    ) -> "RouteDocument":
        if not isinstance(data, dict) or not isinstance(data.get("waypoints"), list):
            raise ValueError("route JSON must contain a waypoints list")

        source_points, frame, units, closed = _route_core_fields(
            data,
            require_map_units=require_map_units,
        )

        schema = _optional_text(data, "schema")
        site_id = _optional_text(data, "site_id")
        coordinate_frame_id = _optional_text(data, "coordinate_frame_id")
        purpose = _optional_text(data, "purpose")
        align_source = _optional_text(data, "align_source")

        if require_flight_contract:
            _validate_flight_contract(
                data,
                expected_site_id=expected_site_id,
                expected_coordinate_frame_id=expected_coordinate_frame_id,
                frame=frame,
                closed=closed,
            )
        controller_points, align_source = _controller_coordinates(
            source_points,
            frame=frame,
            align_source=align_source,
            map_frame=map_frame,
        )
        arrive_radius = _arrival_radius(data)

        return cls(
            waypoints=source_points,
            controller_points=controller_points,
            frame=frame,
            units=units,
            closed=closed,
            schema=schema,
            site_id=site_id,
            coordinate_frame_id=coordinate_frame_id,
            purpose=purpose,
            align_source=align_source,
            arrive_radius_map_units=arrive_radius,
        )

    @classmethod
    def from_bytes(cls, payload: bytes, **kwargs: Any) -> "RouteDocument":
        return cls.from_data(_read_json_bytes(payload, "route bytes"), **kwargs)

    @classmethod
    def from_path(cls, path_json: str | Path, **kwargs: Any) -> "RouteDocument":
        path = Path(path_json)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read route: {path}") from exc
        try:
            data = _read_json_bytes(payload, str(path))
        except ValueError:
            raise
        return cls.from_data(data, **kwargs)

    def source_waypoints(
        self, *, close: bool | None = None
    ) -> tuple[tuple[float, float, float], ...]:
        should_close = self.closed if close is None else bool(close)
        if should_close and self.waypoints[-1] != self.waypoints[0]:
            return self.waypoints + (self.waypoints[0],)
        return self.waypoints

    def controller_waypoints(self, *, close: bool = False) -> list[np.ndarray]:
        points = self.controller_points
        if close and points[-1] != points[0]:
            points = points + (points[0],)
        return [np.array(point, dtype=float, copy=True) for point in points]


def _waypoints_from_route_data(
    data: object,
    *,
    expected_site_id: str | None = None,
    expected_coordinate_frame_id: str | None = None,
    require_flight_contract: bool = False,
    map_frame: MapFrameLike = LEGACY_MAP_FRAME,
) -> list[np.ndarray]:
    """Compatibility adapter for the controller's historical private helper."""
    return RouteDocument.from_data(
        data,
        expected_site_id=expected_site_id,
        expected_coordinate_frame_id=expected_coordinate_frame_id,
        require_flight_contract=require_flight_contract,
        require_map_units=True,
        map_frame=map_frame,
    ).controller_waypoints()


def load_waypoints(
    path_json: str | Path,
    *,
    expected_site_id: str | None = None,
    expected_coordinate_frame_id: str | None = None,
    require_flight_contract: bool = False,
    map_frame: MapFrameLike = LEGACY_MAP_FRAME,
) -> list[np.ndarray]:
    return RouteDocument.from_path(
        path_json,
        expected_site_id=expected_site_id,
        expected_coordinate_frame_id=expected_coordinate_frame_id,
        require_flight_contract=require_flight_contract,
        require_map_units=True,
        map_frame=map_frame,
    ).controller_waypoints()


@dataclass(frozen=True)
class MissionRouteSnapshot:
    """Immutable route identity captured before an AUTO mission."""

    path: Path
    sha256: str
    site_id: str
    coordinate_frame_id: str
    waypoints: tuple[tuple[float, float, float], ...]
    arrive_radius_map_units: float | None = None

    def controller_waypoints(self) -> list[np.ndarray]:
        return [np.array(point, dtype=float, copy=True) for point in self.waypoints]

    def verify_file_unchanged(self) -> None:
        try:
            current = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"selected route is no longer readable: {self.path}") from exc
        if not hmac.compare_digest(current, self.sha256):
            raise ValueError(
                "selected route changed after selection; re-open and validate it "
                "before starting AUTO"
            )


def capture_mission_route_snapshot(
    path_json: str | Path,
    *,
    expected_sha256: str,
    expected_site_id: str,
    expected_coordinate_frame_id: str,
    map_frame: MapFrameLike = LEGACY_MAP_FRAME,
) -> MissionRouteSnapshot:
    path = Path(path_json).expanduser().resolve()
    expected_digest = str(expected_sha256 or "").strip().lower()
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise ValueError("expected route SHA-256 must be 64 lowercase hex characters")
    site_id = str(expected_site_id or "").strip()
    coordinate_frame_id = str(expected_coordinate_frame_id or "").strip()
    if not site_id or not coordinate_frame_id:
        raise ValueError("route snapshot requires site_id and coordinate_frame_id")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read selected route: {path}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(digest, expected_digest):
        raise ValueError(
            f"selected route SHA-256 mismatch: expected {expected_digest}, got {digest}"
        )
    route = RouteDocument.from_bytes(
        payload,
        expected_site_id=site_id,
        expected_coordinate_frame_id=coordinate_frame_id,
        require_flight_contract=True,
        require_map_units=True,
        map_frame=map_frame,
    )
    return MissionRouteSnapshot(
        path=path,
        sha256=digest,
        site_id=site_id,
        coordinate_frame_id=coordinate_frame_id,
        waypoints=tuple((point[0], point[1], point[2]) for point in route.controller_points),
        arrive_radius_map_units=route.arrive_radius_map_units,
    )


class MissionRouteLock:
    """Keep one route bound across AUTO/HOVER/MANUAL/resume states."""

    def __init__(self, snapshot: MissionRouteSnapshot | None = None):
        self._snapshot = snapshot
        self._state = "idle"

    @property
    def snapshot(self) -> MissionRouteSnapshot | None:
        return self._snapshot

    @property
    def active(self) -> bool:
        return self._state != "idle"

    def bind(self, snapshot: MissionRouteSnapshot) -> None:
        if self.active:
            raise ValueError("cannot switch route while an AUTO mission is active")
        self._snapshot = snapshot

    def begin_auto(self, *, displayed_sha256: str | None) -> MissionRouteSnapshot:
        snapshot = self._snapshot
        if snapshot is None:
            raise ValueError("no validated route is selected for AUTO")
        displayed = str(displayed_sha256 or "").strip().lower()
        if not hmac.compare_digest(displayed, snapshot.sha256):
            raise ValueError("displayed route does not match the selected AUTO route")
        snapshot.verify_file_unchanged()
        if self._state == "idle":
            self._state = "pending"
        return snapshot

    def confirm_auto_started(self) -> None:
        if self._state != "pending":
            raise ValueError("no pending AUTO route start to confirm")
        self._state = "active"

    def resume_auto(self) -> MissionRouteSnapshot:
        if self._state != "active" or self._snapshot is None:
            raise ValueError("no active AUTO mission route to resume")
        return self._snapshot

    def cancel_rejected_auto_start(self) -> None:
        if self._state == "pending":
            self._state = "idle"

    def release_after_confirmed_landed(self, flight_state: str) -> None:
        if str(flight_state or "").strip().lower() != "landed":
            raise ValueError("route lock releases only after confirmed landed")
        self._state = "idle"

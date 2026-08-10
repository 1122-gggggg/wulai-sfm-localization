"""Route-authoring data model and the explicit editor/GLOMAP frame boundary."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROUTE_SCHEMA = "sfm-flight-route/v1"
PREVIEW_ROUTE_SCHEMA = "sfm-route-preview/v1"


def glomap_to_editor(points, frame=None) -> np.ndarray:
    """Raw GLOMAP -> operator-facing Z-up (east, north, up) components.

    ``frame`` is the site's measured MapFrame. Without it this falls back to the
    legacy [x, z, -y] swap, which ASSUMES GLOMAP -Y is up -- wrong by 22.51 deg on
    target_site_v1. Getting it wrong tilts the whole editing frame: "up" on screen
    stops being up, so raising a waypoint also shoves it sideways in the real world.
    """
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    if frame is None:
        return np.column_stack((values[:, 0], values[:, 2], -values[:, 1]))
    return np.column_stack((values @ frame.east, values @ frame.north,
                            values @ frame.up))


def editor_to_glomap(points, frame=None) -> np.ndarray:
    """Operator-facing Z-up -> raw GLOMAP, through the SAME basis used to display.

    Must mirror glomap_to_editor exactly; a mismatched pair silently rotates every
    exported waypoint by the angle the two bases differ by.
    """
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    if frame is None:
        return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))
    return (values[:, 0:1] * frame.east[None, :]
            + values[:, 1:2] * frame.north[None, :]
            + values[:, 2:3] * frame.up[None, :])


def _waypoints(raw: object, label: str) -> list[list[float]]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} waypoints must be a list")
    parsed: list[list[float]] = []
    for index, point in enumerate(raw):
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"{label} waypoint[{index}] must contain three numbers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in point
        ):
            raise ValueError(f"{label} waypoint[{index}] must contain finite numbers")
        parsed.append([float(value) for value in point])
    return parsed


def _arrive_radius(raw: object) -> float | None:
    """Validate the operator-confirmed arrival sphere carried by a route file."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("route arrive_radius_map_units must be a number")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("route arrive_radius_map_units must be finite and > 0")
    return value


def _editor_points_from_route(
    raw: dict,
    *,
    map_frame,
    align_source: str,
) -> np.ndarray:
    points = _waypoints(raw.get("waypoints"), "route")
    frame = raw.get("frame", "aligned")
    if frame == "glomap":
        return glomap_to_editor(points, map_frame)
    if frame != "aligned":
        raise ValueError("route frame must be 'aligned' or 'glomap'")

    declared = raw.get("align_source")
    if declared is None:
        declared = "legacy"
    if declared not in ("legacy", "measured"):
        raise ValueError(f"unsupported route align_source: {declared!r}")
    if declared == align_source:
        return np.asarray(points, dtype=float).reshape(-1, 3)
    if declared == "legacy":
        # Legacy-authored route opened at a measured site: round-trip
        # through raw GLOMAP so the points land in the basis on screen.
        return glomap_to_editor(editor_to_glomap(points), map_frame)
    raise ValueError(
        "route declares align_source='measured' but this site has no "
        "measured gravity alignment, so the basis it was authored in "
        "cannot be reconstructed"
    )


@dataclass
class RouteDocument:
    """An open waypoint polyline stored internally in editor Z-up coordinates."""

    site_id: str
    coordinate_frame_id: str
    points: list[list[float]] = field(default_factory=list)
    source_path: Path | None = None
    #: Which alignment produced the editor frame these points were marked in.
    #: "aligned" alone is ambiguous, and reading it back with the wrong basis
    #: rotates every waypoint; the flight loader refuses a route that omits this.
    align_source: str = "legacy"
    #: Arrival sphere the operator confirmed, in map units.
    arrive_radius_map_units: float | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        site_id: str,
        coordinate_frame_id: str,
        map_frame=None,
        align_source: str = "legacy",
    ) -> RouteDocument:
        """Read a route INTO the editor's current basis.

        ``map_frame``/``align_source`` describe the basis this editor session
        displays the point cloud through. A file authored against the other basis
        is CONVERTED through raw GLOMAP, never relabelled: stamping the session's
        align_source onto untouched legacy coordinates produces a file whose
        declaration and contents disagree, and the flight loader then applies the
        measured inverse to legacy points -- rotating every waypoint by the site's
        tilt, which is the exact failure align_source was added to prevent.
        """
        source = Path(path).expanduser().resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("route root must be a JSON object")
        declared_schema = raw.get("schema")
        if declared_schema not in (None, ROUTE_SCHEMA):
            raise ValueError(f"unsupported route schema: {declared_schema!r}")
        if raw.get("units", "map") != "map":
            raise ValueError("route units must be 'map'")
        if raw.get("purpose", "flight") != "flight":
            raise ValueError("route purpose must be 'flight'")
        if raw.get("closed", False) is not False:
            raise ValueError("route editor accepts only open routes")
        declared_site = raw.get("site_id")
        if declared_site not in (None, site_id):
            raise ValueError(
                f"route site_id {declared_site!r} does not match {site_id!r}"
            )
        declared_frame_id = raw.get("coordinate_frame_id")
        if declared_frame_id not in (None, coordinate_frame_id):
            raise ValueError(
                "route coordinate_frame_id does not match the selected site"
            )
        editor_points = _editor_points_from_route(
            raw,
            map_frame=map_frame,
            align_source=align_source,
        )
        return cls(
            site_id=site_id,
            coordinate_frame_id=coordinate_frame_id,
            points=editor_points.tolist(),
            source_path=source,
            align_source=align_source,
            arrive_radius_map_units=_arrive_radius(raw.get("arrive_radius_map_units")),
        )

    def payload(self, *, preview_only: bool = False) -> dict:
        points = _waypoints(self.points, "editor")
        if len(points) < 2:
            raise ValueError("航線至少需要兩個航點")
        if all(a == b for a, b in zip(points, points[1:])):
            raise ValueError("航線必須包含至少一段非零長度的線段")
        payload = {
            "schema": PREVIEW_ROUTE_SCHEMA if preview_only else ROUTE_SCHEMA,
            "site_id": None if preview_only else self.site_id,
            "coordinate_frame_id": None if preview_only else self.coordinate_frame_id,
            "frame": "aligned",
            "align_source": self.align_source,
            "units": "map",
            "purpose": "preview_only" if preview_only else "flight",
            "closed": False,
            "waypoints": points,
            "arrive_radius_map_units": self.arrive_radius_map_units,
            "source": "operator_route_editor",
            "note": (
                "unbound PLY preview; cannot be imported as a flight route"
                if preview_only
                else "draft route; flight clearance remains unapproved"
            ),
        }
        return payload

    def save(self, path: str | Path, *, preview_only: bool = False) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(
                self.payload(preview_only=preview_only),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        self.source_path = target
        return target

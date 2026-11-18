"""Route-authoring data model and the explicit editor/GLOMAP frame boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from route_editor_wires import wire_annotations

# Authoring fields are validated by the flight route contract's owner; keep
# this module importable before the desktop entrypoint installs that path.
_FLIGHT_CONTROL = Path(__file__).resolve().parents[2] / "定位演算法" / "flight_control"
if str(_FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_CONTROL))

from route_domain import (  # noqa: E402
    MAX_ACTIONS_PER_WAYPOINT,
    parse_segment_look_at,
    parse_waypoint_action,
)

ROUTE_SCHEMA = "sfm-flight-route/v1"
PREVIEW_ROUTE_SCHEMA = "sfm-route-preview/v1"
MAP_DISTANCE_SCALE_SCHEMA = "sfm_map_distance_scale_v1"


@dataclass(frozen=True)
class MapDistanceScale:
    """Display-only map scale; an estimate is not a flight-runtime calibration."""

    m_per_unit: float
    estimated: bool
    source: str
    uncertainty_relative: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.m_per_unit, bool)
            or not isinstance(self.m_per_unit, (int, float))
            or not math.isfinite(self.m_per_unit)
            or self.m_per_unit <= 0
        ):
            raise ValueError("m_per_unit must be finite and > 0")
        if not isinstance(self.estimated, bool):
            raise ValueError("estimated must be a boolean")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        uncertainty = self.uncertainty_relative
        if uncertainty is not None and (
            isinstance(uncertainty, bool)
            or not isinstance(uncertainty, (int, float))
            or not math.isfinite(uncertainty)
            or uncertainty < 0
        ):
            raise ValueError("uncertainty_relative must be finite and >= 0")

    def distance_m(self, first, second) -> float:
        """Return the three-dimensional straight-line distance without mutation."""
        if len(first) != 3 or len(second) != 3:
            raise ValueError("distance points must contain three coordinates")
        distance = math.dist(first, second)
        if not math.isfinite(distance):
            raise ValueError("distance points must contain finite coordinates")
        return distance * self.m_per_unit

    def route_segment_distances(self, points) -> list[float]:
        """Measure consecutive segments of an open route, never closing it."""
        return [self.distance_m(first, second) for first, second in zip(points, points[1:])]


@lru_cache(maxsize=8)
def _canonical_map_sha256(
    path: Path, _size: int, _mtime_ns: int, _ctime_ns: int,
) -> str:
    """Cache only the large canonical asset, invalidating when its stat changes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_map_distance_scale(profile) -> MapDistanceScale | None:
    """Reload a map/frame-bound sidecar, or leave metric display unavailable.

    Only the canonical ``profile.map_ply`` identifies the map, never a dense
    rendering substitute. This metadata cannot authorize flight control or
    change route coordinates. Missing or invalid metadata never implies 1 m/u.
    """
    if profile is None:
        return None
    try:
        map_path = Path(profile.map_ply)
        frame_id = profile.coordinate_frame.id
        if not isinstance(frame_id, str) or not frame_id:
            return None
        raw = json.loads(
            (map_path.parent / "map_distance_scale.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(raw, dict)
            or raw.get("schema") != MAP_DISTANCE_SCALE_SCHEMA
            or raw.get("coordinate_frame_id") != frame_id
        ):
            return None
        declared_digest = raw.get("map_ply_sha256")
        if (
            not isinstance(declared_digest, str)
            or len(declared_digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in declared_digest)
        ):
            return None
        scale = MapDistanceScale(
            m_per_unit=raw.get("m_per_unit"),
            estimated=raw.get("estimated"),
            source=raw.get("source"),
            uncertainty_relative=raw.get("uncertainty_relative"),
        )
        asset_digest = getattr(getattr(profile, "asset_sha256", None), "map_ply", None)
        if asset_digest:
            if (
                not isinstance(asset_digest, str)
                or len(asset_digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in asset_digest)
            ):
                return None
            map_digest = asset_digest
        else:
            canonical_path = map_path.expanduser().resolve()
            stat = canonical_path.stat()
            map_digest = _canonical_map_sha256(
                canonical_path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
            )
        return scale if declared_digest.lower() == map_digest.lower() else None
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError, OverflowError):
        return None


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


def _waypoint_radii(raw: object, count: int) -> list[float] | None:
    """Validate per-waypoint arrival spheres; None means all-global."""
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError(
            "route waypoint_arrive_radii must be a list with one radius per waypoint"
        )
    radii = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"route waypoint_arrive_radii[{index}] must be a number")
        radius = float(value)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                f"route waypoint_arrive_radii[{index}] must be finite and > 0"
            )
        radii.append(radius)
    return radii


def waypoint_actions_json(
    raw: object, count: int, key: str = "waypoint_actions"
) -> list[list[dict]] | None:
    """Validate per-waypoint actions with the flight contract; None = none."""
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError(f"route {key} must be a list with one list per waypoint")
    parsed = []
    for index, actions in enumerate(raw):
        if not isinstance(actions, list) or len(actions) > MAX_ACTIONS_PER_WAYPOINT:
            raise ValueError(
                f"route {key}[{index}] must be a list of at most "
                f"{MAX_ACTIONS_PER_WAYPOINT} actions"
            )
        parsed.append([
            parse_waypoint_action(action, f"route {key}[{index}][{step}]").to_json()
            for step, action in enumerate(actions)
        ])
    return parsed if any(parsed) else None


def return_waypoint_actions_json(raw: object, count: int) -> list[list[dict]] | None:
    """Return-pass actions: the landing point (first) and turnaround (last) cannot carry any."""
    parsed = waypoint_actions_json(raw, count, "return_waypoint_actions")
    if parsed is not None:
        for index in sorted({0, count - 1}):
            if 0 <= index < count and parsed[index]:
                raise ValueError(
                    f"回程動作不能設在第 {index + 1} 個航點：第一點是降落點，"
                    "最後一點是只飛一次的折返點"
                )
    return parsed


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


def _box_field(box: dict, key: str, label: str) -> list:
    values = box[key]
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{label}.{key} must contain three numbers")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}.{key} must contain finite numbers")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{label}.{key} must contain finite numbers")
        if key == "size" and value <= 0:
            raise ValueError(f"{label}.size must be > 0")
    return list(values)


def _object_boxes(raw: object) -> list[dict]:
    """Validate editor AABBs, including boxes selected as segment yaw targets."""
    if not isinstance(raw, list):
        raise ValueError("route object_boxes must be a list")
    boxes = []
    for index, box in enumerate(raw):
        label = f"route object_boxes[{index}]"
        if not isinstance(box, dict) or set(box) != {"center", "size"}:
            raise ValueError(f"{label} must contain exactly center and size")
        boxes.append({key: _box_field(box, key, label) for key in ("center", "size")})
    return boxes


def _editor_boxes_from_route(
    raw: dict, *, map_frame, align_source: str,
) -> list[dict]:
    """Rebound all physical corners when changing the editor alignment basis.

    Boxes are always aligned editor Z-up AABBs, even for GLOMAP waypoints.
    Legacy-to-measured conversion may enlarge them because rotation is not
    stored. An unreconstructable measured basis fails closed like waypoints.
    """
    boxes = _object_boxes(raw.get("object_boxes", []))
    declared = raw.get("align_source")
    if declared is None:
        declared = "legacy"
    if declared in ("legacy", "measured") and declared == align_source:
        return boxes
    converted = []
    for box in boxes:
        center = np.asarray(box["center"], dtype=float)
        half_size = np.asarray(box["size"], dtype=float) / 2.0
        corners = center + np.asarray(list(product((-1, 1), repeat=3))) * half_size
        points = _editor_points_from_route(
            {
                "frame": "aligned",
                "align_source": declared,
                "waypoints": corners.tolist(),
            },
            map_frame=map_frame,
            align_source=align_source,
        )
        lo, hi = points.min(axis=0), points.max(axis=0)
        converted.append({
            "center": (lo / 2.0 + hi / 2.0).tolist(),
            "size": (hi - lo).tolist(),
        })
    return _object_boxes(converted)

def _editor_wires_from_route(raw: dict, *, map_frame, align_source: str) -> list[dict]:
    """Convert both attachments and the sag axis, preserving the entire curve."""
    wires = wire_annotations(raw.get("wires", []))
    declared = raw.get("align_source", "legacy")
    if declared == align_source:
        return wires
    converted = []
    for wire in wires:
        points = _editor_points_from_route(
            {
                "frame": "aligned",
                "align_source": declared,
                "waypoints": [wire["a"], wire["b"], [0.0, 0.0, 0.0], wire["up"]],
            },
            map_frame=map_frame, align_source=align_source,
        )
        converted.append({
            "a": points[0].tolist(), "b": points[1].tolist(),
            "sag": wire["sag"], "up": (points[3] - points[2]).tolist(),
        })
    return wire_annotations(converted)



@dataclass
class RouteDocument:
    """An open polyline plus offline object AABBs in editor Z-up coordinates."""

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
    #: Per-waypoint arrival spheres drawn by the operator; None = all-global.
    waypoint_arrive_radii: list[float] | None = None
    #: Per-waypoint actions (route_domain JSON form), run after arrival; None = none.
    waypoint_actions: list[list[dict]] | None = None
    #: Per-waypoint actions run when the RETURN pass reaches that waypoint; None = none.
    return_waypoint_actions: list[list[dict]] | None = None
    #: Object annotations in this document's aligned editor basis, not obstacles.
    object_boxes: list[dict] = field(default_factory=list)
    #: Zero-based object_boxes indexes for authored legs i -> i + 1.
    segment_look_at: list[int | None] | None = None
    #: Offline wire annotations, never flight obstacles.
    wires: list[dict] = field(default_factory=list)

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
        boxes = _editor_boxes_from_route(
            raw, map_frame=map_frame, align_source=align_source,
        )
        return cls(
            site_id=site_id,
            coordinate_frame_id=coordinate_frame_id,
            points=editor_points.tolist(),
            source_path=source,
            align_source=align_source,
            arrive_radius_map_units=_arrive_radius(raw.get("arrive_radius_map_units")),
            waypoint_arrive_radii=_waypoint_radii(
                raw.get("waypoint_arrive_radii"), len(editor_points)
            ),
            waypoint_actions=waypoint_actions_json(
                raw.get("waypoint_actions"), len(editor_points)
            ),
            return_waypoint_actions=return_waypoint_actions_json(
                raw.get("return_waypoint_actions"), len(editor_points)
            ),
            object_boxes=boxes,
            wires=_editor_wires_from_route(
                raw, map_frame=map_frame, align_source=align_source,
            ),
            segment_look_at=parse_segment_look_at(
                raw.get("segment_look_at"),
                waypoint_count=len(editor_points),
                box_count=len(boxes),
            ),
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
            "waypoint_arrive_radii": self.aligned_point_radii(len(points)),
            "waypoint_actions": waypoint_actions_json(self.waypoint_actions, len(points)),
            "return_waypoint_actions": return_waypoint_actions_json(
                self.return_waypoint_actions, len(points)
            ),
            "source": "operator_route_editor",
            "note": (
                "unbound PLY preview; cannot be imported as a flight route"
                if preview_only
                else "latest operator route; AUTO clearance is controlled by the site profile"
            ),
        }
        boxes = _object_boxes(self.object_boxes)
        if boxes:
            payload["object_boxes"] = boxes
        wires = wire_annotations(self.wires)
        if wires:
            payload["wires"] = wires
        references = parse_segment_look_at(
            self.segment_look_at, waypoint_count=len(points), box_count=len(boxes),
        )
        if references is not None:
            payload["segment_look_at"] = references
        return payload

    def aligned_point_radii(self, count: int) -> list[float] | None:
        """Per-point spheres padded/truncated to the current point list.

        The editor window keeps this aligned on add/delete; this backstop
        covers undo/redo, which restores points without touching radii.
        """
        radii = self.waypoint_arrive_radii
        if radii is None:
            return None
        fallback = self.arrive_radius_map_units
        if fallback is None or not isinstance(fallback, (int, float)):
            raise ValueError("route needs a global arrival radius to align per-point radii")
        aligned = list(radii[:count])
        while len(aligned) < count:
            aligned.append(float(fallback))
        return aligned

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

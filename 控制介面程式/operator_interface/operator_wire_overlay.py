"""Display-only wire geometry bound to the operator's immutable route selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from route_editor_model import editor_to_glomap
from route_editor_wires import sample_wire, wire_annotations
from site_profile import load_site_profile


class RouteWireOverlay:
    """Read/sample once per route identity, never in the steady-state render loop."""

    def __init__(self):
        self._key: tuple | None = None
        self._snapshot = None
        self._polylines: tuple[np.ndarray, ...] = ()

    def polylines(self, snapshot, profile_path, note) -> tuple[np.ndarray, ...]:
        if snapshot is None or profile_path is None:
            self._key = None
            self._polylines = ()
            self._snapshot = None
            return self._polylines
        key = (
            snapshot.path, snapshot.sha256, snapshot.site_id,
            snapshot.coordinate_frame_id, Path(profile_path),
        )
        if key == self._key and snapshot is self._snapshot:
            return self._polylines
        # Discard the previous site's geometry even when the new data is invalid.
        self._key = key
        self._snapshot = snapshot
        self._polylines = ()
        try:
            payload = Path(snapshot.path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != snapshot.sha256:
                raise ValueError("航線內容已變更，請重新選取航線")
            profile = load_site_profile(profile_path)
            frame_id = getattr(profile.coordinate_frame, "id", None)
            if profile.site_id != snapshot.site_id or frame_id != snapshot.coordinate_frame_id:
                raise ValueError("電線所屬場域／座標與目前地圖不符")
            raw = json.loads(payload)
            if not isinstance(raw, dict):
                raise ValueError("電線航線必須是 JSON 物件")
            if raw.get("map_ply_sha256", profile.asset_sha256.map_ply) != profile.asset_sha256.map_ply:
                raise ValueError("電線所屬地圖 SHA-256 不符")
            wires = wire_annotations(raw.get("wires", []))
            if not wires:
                return self._polylines
            # Wire coordinates are always aligned editor coordinates, even if
            # outer frame='glomap' describes the route's waypoint coordinates.
            alignment = raw.get("align_source") or "legacy"
            map_frame = None
            if alignment == "measured":
                if profile.map_align is None:
                    raise ValueError("電線宣告重力對齊，但目前場域沒有對齊資料")
                from real_path_follow_controller import load_map_frame

                map_frame = load_map_frame(profile.map_align)
            elif alignment != "legacy":
                raise ValueError(f"不支援的電線對齊座標：{alignment}")
            curves = []
            for wire in wires:
                points = editor_to_glomap(sample_wire(wire), map_frame)
                if not np.isfinite(points).all():
                    raise ValueError("電線座標轉換結果不是有限值")
                points.setflags(write=False)
                curves.append(points)
            self._polylines = tuple(curves)
        except (OSError, ValueError, TypeError, AttributeError, OverflowError) as exc:
            note(f"電線圖層未顯示：{exc}")
        return self._polylines
